import pathlib
import os
import huggingface_hub
import numpy as np
import SimpleITK as sitk
import SimpleITK.utilities.vtk
import vtk
import armcrop
import torch
from typing import List


# make nnunet stop spitting out warnings from environment variables the author declared
os.environ["nnUNet_raw"] = "None"
os.environ["nnUNet_preprocessed"] = "None"
os.environ["nnUNet_results"] = "None"

import nnunetv2
import nnunetv2.inference
import nnunetv2.inference.predict_from_raw_data


class Net:
    def __init__(self, bone_type: str):
        """
        Initialize the ML model for inference. Downloads the model from huggingface hub. Placing this in inside a for loop will cause the model to be loaded into memory multiple times. This is not ideal.

        Args:
            bone_type: Type of bone to detect and segment. Must be either 'scapula' or 'humerus'
        """
        # Initialize cache variables
        self._cache_key = None
        self._cache_result = None

        self.bone_type = bone_type
        self._model_path = self._get_nnunet_model(bone_type)

        if torch.cuda.is_available():
            self._nnunet_predictor = nnunetv2.inference.predict_from_raw_data.nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                verbose=False,
                verbose_preprocessing=False,
            )
        else:
            self._nnunet_predictor = nnunetv2.inference.predict_from_raw_data.nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                verbose=False,
                verbose_preprocessing=False,
                device=torch.device("cpu"),
                perform_everything_on_device=False,
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
        # we default to a lower confidence threshold as we care more about complete capture
        # of the bone than an accurate bounding box
        return armcrop.OBBCrop2Bone(
            vol_path,
            confidence_threshold=0.2,
            iou_supress_threshold=0.4,
        )

    def _convert_sitk_to_nnunet(self, vol_sitk: sitk.Image):
        # this needs some work
        arr = np.expand_dims(sitk.GetArrayFromImage(vol_sitk), 0).astype(np.float32)
        prop = {
            "sitk_stuff": {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                "spacing": vol_sitk.GetSpacing(),
                "origin": vol_sitk.GetOrigin(),
                "direction": vol_sitk.GetDirection(),
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong
            # Image arrays are returned x,y,z but spacing is returned z,y,x. Duh.
            "spacing": list(np.abs(vol_sitk.GetSpacing())[::-1]),
        }

        return arr, prop

    def _convert_nnunet_to_sitk(self, result_arr, vols_sitk):
        result_sitk = []
        # for each volume in the batch
        for i, r in enumerate(result_arr):
            r_sitk = sitk.GetImageFromArray(r)
            r_sitk.CopyInformation(vols_sitk[i])
            result_sitk.append(r_sitk)

        return result_sitk

    def post_process(self, seg_sitk: sitk.Image, detection_mean=None) -> sitk.Image:
        """This makes the cortical watertight and deletes the other bones."""

        # Create binary mask of classes 2-4 which is the entire bone
        b_mask = sitk.BinaryThreshold(
            seg_sitk, lowerThreshold=2, upperThreshold=4, insideValue=1, outsideValue=0
        )
        # get connected components and remove small components
        cc = sitk.RelabelComponent(
            sitk.ConnectedComponent(b_mask),
            sortByObjectSize=True,
            minimumObjectSize=5000,
        )

        if detection_mean is not None:
            # keep the connected component closest to bone_centroid
            cc_stats = sitk.LabelShapeStatisticsImageFilter()
            cc_stats.ComputeOrientedBoundingBoxOn()
            cc_stats.Execute(cc)

            # if more than 1 object
            if len(cc_stats.GetLabels()) > 1:
                # keep the connected component closest to the origin that matches the obb size
                dists = []
                for label in cc_stats.GetLabels():
                    # filter out any components less than 80 % of the obb z-dim
                    cc_size = cc_stats.GetOrientedBoundingBoxSize(label)
                    obb_size = (
                        seg_sitk.GetSize()[2] * seg_sitk.GetSpacing()[2]
                    ) - 2 * self.z_padding
                    if cc_size[2] > 0.50 * obb_size:
                        label_centroid = cc_stats.GetCentroid(label)
                        dist = np.linalg.norm(np.array(label_centroid) - np.array(detection_mean))
                        dists.append(dist)
                # if no components are greater than 80% of the obb z-dim
                if len(dists) == 0:
                    b_mask = cc == 1
                else:
                    # keep the closest label
                    b_mask = cc == cc_stats.GetLabels()[np.argmin(dists)]
            else:
                b_mask = cc == 1

        else:
            # keep the largest connected component
            b_mask = cc == 1

        del cc
        # Get contour of the bone binary mask
        contour = sitk.BinaryContour(
            b_mask, fullyConnected=True, backgroundValue=0, foregroundValue=1
        )
        # Get locations where contour=1 AND class of seg_stik = 3
        contour_on_class3 = sitk.Multiply(contour, sitk.Equal(seg_sitk, 3))
        del contour
        # Subtract contour from class 3 to make it class 2
        result = sitk.Subtract(seg_sitk, contour_on_class3)  # Turn class 3 to 2

        # retain class 2 and class 3 only where overlapping b_mask
        result = sitk.Multiply(result, b_mask)

        return result

    def _predict_obb(self, vol_path: str, vol_input: sitk.Image) -> List[sitk.Image]:
        if self._cache_key == vol_path:
            return self._cache_result

        obb_cropper = self._obb(vol_input)
        if self.bone_type == "scapula":
            self.xy_padding = 60
            self.z_padding = 60
            vols_obb = obb_cropper.scapula(
                [0.5, 0.5, 0.5],
                xy_padding=self.xy_padding,
                z_padding=self.z_padding,
                z_iou_interval=80,
                z_length_min=30,
            )
        elif self.bone_type == "humerus":
            self.xy_padding = 10
            self.z_padding = 60
            vols_obb = obb_cropper.humerus(
                [0.5, 0.5, 0.5],
                xy_padding=self.xy_padding,
                z_padding=self.z_padding,
                z_iou_interval=80,
                z_length_min=40,
            )
        # get detection means
        obb_segs = []
        for vol_obb, dmean in zip(vols_obb, obb_cropper.detection_means):
            v, p = self._convert_sitk_to_nnunet(vol_obb)
            r = self._nnunet_predictor.predict_single_npy_array(v, p)
            del v, p

            # create a sitk image from the prediction
            r = sitk.GetImageFromArray(r)
            r.CopyInformation(vol_obb)

            # post process the segmentation
            r = self.post_process(r, dmean)
            obb_segs.append(r)

        # update cache
        self._cache_key = vol_path
        self._cache_result = obb_segs

        return obb_segs

    def predict(
        self,
        vol_path: str | pathlib.Path,
    ) -> List[sitk.Image]:
        """Predicts the segmentation of the bone.

        Args:
            vol_path: Path to the volume to segment

        Returns:
            List of sitk.Image objects

            The list is structured as follows:
            [
                detected_bone1,
                detected_bone2,
                ...
            ]
        """

        vol_input = sitk.ReadImage(str(vol_path))
        if self.bone_type == "scapula":
            Unaligner = armcrop.UnalignOBBSegmentation(
                vol_input,
                thin_regions={2: (2, 3)},
                face_connectivity_regions=[2],
                face_connectivity_repeats=2,
            )
        elif self.bone_type == "humerus":
            Unaligner = armcrop.UnalignOBBSegmentation(
                vol_input,
                thin_regions={2: (2, 3)},
            )

        output_segs = []
        for r in self._predict_obb(str(vol_path), vol_input):
            # unalign the segmentation
            r = Unaligner(r)
            # post process the segmentation
            r = self.post_process(r)
            output_segs.append(r)

        return output_segs

    def predict_poly(
        self,
        vol_path: str | pathlib.Path,
        smooth_iter=30,
        smooth_passband=0.01,
    ) -> List[List[vtk.vtkPolyData]]:
        """Predicts the segmentation of the bone and returns a list of vtkPolyData objects.

        Args:
            vol_path: Path to the volume to segment

        Returns:
            List of vtkPolyData objects

            The list is structured as follows:
            [
                [detected_bone1-cortical, detected_bone1-trabecular],
                [detected_bone2-cortical, detected_bone2-trabecular],
                ...
            ]
        """
        results = []
        for r in self._predict_obb(str(vol_path), sitk.ReadImage(str(vol_path))):
            polys = []
            for label in [2, 3]:  # Iterate through labels 2 and 3

                # removes in the internal surface of the cortical bone
                if label == 2:
                    _r = sitk.BinaryThreshold(
                        r, lowerThreshold=2, upperThreshold=4, insideValue=2, outsideValue=0
                    )
                    r_vtk = SimpleITK.utilities.vtk.sitk2vtk(_r)
                    del _r
                else:
                    r_vtk = SimpleITK.utilities.vtk.sitk2vtk(r)

                # pad the image incase the contour is on the edge
                pad = vtk.vtkImageConstantPad()
                pad.SetInputData(r_vtk)
                extents = r_vtk.GetExtent()
                pad.SetOutputWholeExtent(
                    extents[0] - 1,
                    extents[1] + 1,
                    extents[2] - 1,
                    extents[3] + 1,
                    extents[4] - 1,
                    extents[5] + 1,
                )
                pad.SetConstant(0)
                pad.Update()
                r_vtk = pad.GetOutput()

                # the spacing here is always (0.5, 0.5, 0.5)
                # which makes conversion parameters like smoothing consitent
                # convert to polydata
                # Generate contour for current label
                flying_edges = vtk.vtkDiscreteFlyingEdges3D()
                flying_edges.SetInputData(r_vtk)
                flying_edges.GenerateValues(1, label, label)
                flying_edges.Update()
                poly = flying_edges.GetOutput()

                # decimate the polydata it is super dense
                decimate = vtk.vtkQuadricDecimation()
                decimate.SetInputData(poly)
                decimate.SetTargetReduction(0.5)
                decimate.VolumePreservationOn()
                decimate.Update()
                poly = decimate.GetOutput()

                # apply windowed sinc filter
                smoother = vtk.vtkWindowedSincPolyDataFilter()
                smoother.SetInputData(poly)
                # less smoothing
                smoother.SetNumberOfIterations(smooth_iter)
                smoother.SetPassBand(smooth_passband)
                smoother.BoundarySmoothingOff()
                smoother.FeatureEdgeSmoothingOff()
                smoother.NonManifoldSmoothingOn()
                smoother.Update()  # Update smoother

                poly = smoother.GetOutput()

                polys.append(poly)  # Append smoothed polydata

            results.append(polys)  # Append list of polydata to results

        return results


if __name__ == "__main__":
    from utils import write_polydata

    model = Net("humerus")
    ct = "/mnt/slowdata/ct/arthritic-clinical-half-arm/AAW/AAW.nrrd"
    scapula_segmentations = model.predict(ct)

    for i, s in enumerate(scapula_segmentations):
        sitk.WriteImage(s, f"AAW_scapula_{i}.seg.nrrd", useCompression=True)

    scapula_polydata = model.predict_poly(ct)
    for i, s in enumerate(scapula_polydata):
        for j, p in enumerate(s):
            write_polydata(p, f"AAW_scapula_{i}_{j}.ply")
