import pathlib
import os
import huggingface_hub
import numpy as np
import SimpleITK as sitk
import armcrop

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
        self._nnunet_predictor = nnunetv2.inference.predict_from_raw_data.nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            verbose=False,
            verbose_preprocessing=False,
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

    def predict(self, vol_path: str | pathlib.Path, output_seg_path: str | pathlib.Path):

        # memory could be better managed by deleting objects after they are used
        if self.bone_type == "scapula":
            vols_sitk = self._obb(vol_path).scapula([0.5, 0.5, 0.5])
        elif self.bone_type == "humerus":
            # vols_sitk = self._obb(vol_path).humerus([0.5, 0.5, 0.5])
            vols_sitk = [
                sitk.ReadImage(
                    "/home/greg/projects/segment/stage2_net_training/database/vol_obb/humerus/AAW-0.nrrd"
                )
            ]

        vols_nnunet, props_nnunet = self._convert_sitk_to_nnunet(vols_sitk)

        result_arr = self._nnunet_predictor.predict_from_list_of_npy_arrays(
            vols_nnunet,
            None,
            props_nnunet,
            None,
            num_processes=len(vols_nnunet),
        )
        # convert the result back to sitk
        result_sitk = self._convert_nnunet_to_sitk(result_arr, vols_sitk)

        # in the futre we can switch armcrop to take an sitk object instead of a path
        for r in result_sitk:
            sitk.WriteImage(r, output_seg_path)

            if self.bone_type == "scapula":
                Unaligner = armcrop.UnalignOBBSegmentation(
                    vol_path,
                    thin_regions={2: (2, 3)},
                    face_connectivity_regions=[2],
                    face_connectivity_repeats=2,
                )
            elif self.bone_type == "humerus":
                Unaligner = armcrop.UnalignOBBSegmentation(vol_path, thin_regions={2: (2, 3)})

            # perform the unalignment and save the result
            r_unalign = Unaligner(output_seg_path)
            sitk.WriteImage(r_unalign, output_seg_path)

            # we should add in a step where we apply shell thickness

            return r_unalign


if __name__ == "__main__":
    res = Net("humerus").predict(
        "/mnt/slowdata/arthritic-clinical-half-arm/AAW/AAW.nrrd",
        "/home/greg/projects/armcortnet/armcortnet/AAW.seg.nrrd",
    )
