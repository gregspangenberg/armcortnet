import pathlib
import os
import huggingface_hub
import numpy as np
import SimpleITK as sitk
import armcrop
from typing import List

# make nnunet stop spitting out warnings from environment variables the author declared
os.environ["nnUNet_raw"] = "None"
os.environ["nnUNet_preprocessed"] = "None"
os.environ["nnUNet_results"] = "None"

import nnunetv2
import nnunetv2.inference
import nnunetv2.inference.predict_from_raw_data


class Net:
    # def __init__(self, bone_type: str, save_obb_dir: str | None = None):
    def __init__(self, bone_type: str):

        self.bone_type = bone_type
        # self._save_obb_dir = save_obb_dir
        self._model_path = self._get_nnunet_model(bone_type)
        self._nnunet_predictor = (
            nnunetv2.inference.predict_from_raw_data.nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                verbose=False,
                verbose_preprocessing=False,
            )
        )
        if self.bone_type == "scapula":
            fold = (1,)
        elif self.bone_type == "humerus":
            fold = (0,)

        self._nnunet_predictor.initialize_from_trained_model_folder(
            self._model_path,
            use_folds=fold,
            checkpoint_name="checkpoint_best.pth",
        )

    def _get_nnunet_model(self, bone_type) -> str:
        """
        Download the ML model from hugginface for inference

        Returns:
            model_path: Path to the ML model
        """

        if bone_type not in ["scapula", "humerus"]:
            raise ValueError("bone_type must be either 'scapula' or 'humerus'")

        model_dir = pathlib.Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = huggingface_hub.snapshot_download(
            repo_id=f"gregspangenberg/armcortnet",
            allow_patterns=f"{bone_type}/*",
            local_dir=model_dir,
        )
        model_path = pathlib.Path(model_path) / bone_type
        return str(model_path)

    def _obb(self, vol_path):
        # this could be spedup if armcrop was modified to load its model once not every time it
        # recieves a  new volume
        return armcrop.OBBCrop2Bone(vol_path)

    def _convert_sitk_to_nnunet(self, vols_sitk: list):
        # this needs some work
        vols = []
        props = []
        for v in vols_sitk:
            vols.append(np.expand_dims(sitk.GetArrayFromImage(v), 0).astype(np.float32))
            print(v.GetSize())
            props.append(
                {
                    "sitk_stuff": {
                        # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                        "spacing": v.GetSpacing(),
                        "origin": v.GetOrigin(),
                        "direction": v.GetDirection(),
                    },
                    # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong
                    # Image arrays are returned x,y,z but spacing is returned z,y,x. Duh.
                    "spacing": list(np.abs(v.GetSpacing())[::-1]),
                }
            )
        return vols, props

    def _convert_nnunet_to_sitk(self, result_arr, vols_sitk):
        result_sitk = []
        # for each volume in the batch
        for i, r in enumerate(result_arr):
            r_sitk = sitk.GetImageFromArray(r)
            r_sitk.CopyInformation(vols_sitk[i])
            result_sitk.append(r_sitk)

        return result_sitk

    def post_process(self, seg_sitk: sitk.Image) -> sitk.Image:
        # Create binary mask of classes 2-4 which is the entire bone
        b_mask = sitk.BinaryThreshold(
            seg_sitk, lowerThreshold=2, upperThreshold=4, insideValue=1, outsideValue=0
        )
        # Get contour of the bone binary mask
        contour = sitk.BinaryContour(
            b_mask, fullyConnected=True, backgroundValue=0, foregroundValue=1
        )

        # Get locations where contour=1 AND class of seg_stik = 3
        contour_on_class3 = sitk.Multiply(contour, sitk.Equal(seg_sitk, 3))

        # Subtract contour from class 3 to make it class 2
        result = sitk.Subtract(seg_sitk, contour_on_class3)  # Turn class 3 to 2

        return result

    def predict(
        self, vol_path: str | pathlib.Path, post_process=True
    ) -> List[sitk.Image]:
        print(
            f"initalized {self._nnunet_predictor}, {self._nnunet_predictor.dataset_json}"
        )
        # memory could be better managed by deleting objects after they are used
        if self.bone_type == "scapula":
            vols_sitk = self._obb(vol_path).scapula([0.5, 0.5, 0.5])
        elif self.bone_type == "humerus":
            vols_sitk = self._obb(vol_path).humerus([0.5, 0.5, 0.5])

        output_segs = []
        vols, props = self._convert_sitk_to_nnunet(vols_sitk)
        for i in range(len(vols_sitk)):
            r = self._nnunet_predictor.predict_single_npy_array(
                vols[i],
                props[i],
            )
            r = sitk.GetImageFromArray(r)
            r.CopyInformation(vols_sitk[i])
            if self.bone_type == "scapula":
                Unaligner = armcrop.UnalignOBBSegmentation(
                    vol_path,
                    thin_regions={2: (2, 3)},
                    face_connectivity_regions=[2],
                    face_connectivity_repeats=2,
                )
            elif self.bone_type == "humerus":
                Unaligner = armcrop.UnalignOBBSegmentation(
                    vol_path, thin_regions={2: (2, 3)}
                )

            # perform the unalignment
            r_unalign = Unaligner(r)

            if post_process:
                # perform post processing on the unaligned segmentation
                r_unalign = self.post_process(r_unalign)

            output_segs.append(r_unalign)  # append the segmentation in og csys

        return output_segs


if __name__ == "__main__":
    scapula_segmentations = Net("scapula").predict(
        "/mnt/slowdata/arthritic-clinical-half-arm/AAW/AAW.nrrd",
    )
    print(scapula_segmentations)
    for i, s in enumerate(scapula_segmentations):
        sitk.WriteImage(s, f"AAW_scapula_{i}.seg.nrrd", useCompression=True)
