import gc
import os
import pathlib
import tempfile
from typing import List

import armcrop
import huggingface_hub
import numpy as np
import SimpleITK as sitk
import SimpleITK.utilities.vtk
import torch
import vtk

# make nnunet stop spitting out warnings from environment variables the author declared
os.environ["nnUNet_raw"] = "None"
os.environ["nnUNet_preprocessed"] = "None"
os.environ["nnUNet_results"] = "None"

import cv2
import nnunetv2
import nnunetv2.inference
import nnunetv2.inference.predict_from_raw_data
from nnunetv2.utilities.helpers import empty_cache


def _clear_memory(device=None):
    """Aggressively clear memory after predictions."""
    gc.collect()
    if device is not None:
        empty_cache(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _force_face_connectivity(arr_og: np.ndarray) -> np.ndarray:
    """Force 4-point connectivity on each slice in each direction in the segmentation. This bridges small gaps in the segmentation that closing can't fix."""

    def force_4_connectivity_2d(arr):
        # Label components using 4-connectivity
        _, labeled = cv2.connectedComponents(arr, connectivity=4)

        # Find both left and right diagonal neighbors with different labels in one operation
        upleft = np.roll(np.roll(labeled, -1, axis=0), -1, axis=1)  # Up-Left
        upright = np.roll(np.roll(labeled, -1, axis=0), 1, axis=1)  # Up-Right
        downleft = np.roll(np.roll(labeled, 1, axis=0), -1, axis=1)  # Down-Left
        downright = np.roll(np.roll(labeled, 1, axis=0), 1, axis=1)  # Down-Right

        diag_diff = (
            ((labeled != upleft) & (labeled > 0) & (upleft > 0))
            | ((labeled != upright) & (labeled > 0) & (upright > 0))
            | ((labeled != downleft) & (labeled > 0) & (downleft > 0))
            | ((labeled != downright) & (labeled > 0) & (downright > 0))
        )

        # Create orthogonal connections once
        result = arr.copy()
        result |= np.roll(diag_diff, 1, axis=1)  # Horizontal connection
        result |= np.roll(diag_diff, 1, axis=0)  # Vertical connection

        return result

    arr = arr_og.copy().astype(np.uint8)
    for z in range(arr.shape[0]):
        arr[z, :, :] = force_4_connectivity_2d(arr[z, :, :])
    for x in range(arr.shape[1]):
        arr[:, x, :] = force_4_connectivity_2d(arr[:, x, :])
    for x in range(arr.shape[2]):
        arr[:, :, x] = force_4_connectivity_2d(arr[:, :, x])

    return arr


class Net:
    def __init__(self, bone_type: str, enable_cache: bool = True):
        """
        Initialize the ML model for inference. Downloads the model from huggingface hub. Placing this in inside a for loop will cause the model to be loaded into memory multiple times. This is not ideal.

        Args:
            bone_type: Type of bone to detect and segment. Must be either 'scapula' or 'humerus'
            enable_cache: Whether to cache prediction results. Disable to save RAM when
                         processing many volumes. Default is True.
        """
        # Initialize cache variables
        self._cache_key = None
        self._cache_result = None
        self._caching_enabled = enable_cache

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

    def clear_cache(self):
        """Clear prediction cache and free memory. Call this between processing large volumes."""
        self._cache_key = None
        self._cache_result = None
        _clear_memory(self._nnunet_predictor.device)

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

    def post_process(self, seg_sitk: sitk.Image, detection_means=None) -> sitk.Image:
        """This makes the cortical watertight and deletes the other bones."""

        # Create binary mask of classes 2-4 which is the entire bone
        b_mask = sitk.BinaryThreshold(seg_sitk, lowerThreshold=2, upperThreshold=4, insideValue=1, outsideValue=0)
        # get connected components and remove small components
        cc = sitk.RelabelComponent(
            sitk.ConnectedComponent(b_mask),
            sortByObjectSize=True,
            minimumObjectSize=5000,
        )
        if detection_means is not None:
            # iterate over the connected components
            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(cc)
            lbl_dists = []
            for lbl in shape_stats.GetLabels():
                # get the centroid of the component
                centroid = shape_stats.GetCentroid(lbl)
                # get the closest detection mean to the centroid
                dists = detection_means - centroid
                dists = np.linalg.norm(dists, axis=1)
                lbl_dists.append(np.min(dists))

            # use the label with the smallest distance
            closest_lbl = shape_stats.GetLabels()[np.argmin(lbl_dists)]
            b_mask = sitk.Equal(cc, closest_lbl)

        else:
            # keep the largest connected component
            b_mask = sitk.Equal(cc, 1)

        del cc
        # Get contour of the bone binary mask
        contour = sitk.BinaryContour(b_mask, fullyConnected=True, backgroundValue=0, foregroundValue=1)
        # Get locations where contour=1 AND class of seg_stik = 3
        contour_on_class3 = sitk.Multiply(contour, sitk.Equal(seg_sitk, 3))
        del contour
        # Subtract contour from class 3 to make it class 2
        result = sitk.Subtract(seg_sitk, contour_on_class3)  # Turn class 3 to 2

        # retain class 2 and class 3 only where overlapping b_mask
        result = sitk.Multiply(result, b_mask)

        return result

    def _predict_obb(self, vol_path: str, vol_input: sitk.Image, low_memory: bool = False) -> List[sitk.Image]:
        if self._caching_enabled and self._cache_key == vol_path:
            return self._cache_result

        # get oriented bounding boxes from the volume
        obb_cropper = armcrop.CropOriented(vol_input, detection_confidence=0.2, detection_iou=0.5)
        vols_obb = obb_cropper.process(
            bone=self.bone_type,
            grouping_iou=0.3,
            grouping_interval=50,
            grouping_min_depth=50,
            spacing=(0.5, 0.5, 0.5),
        )
        # # get detection means
        centroider = armcrop.Centroids(vol_input, detection_confidence=0.2, detection_iou=0.5)
        detection_means = centroider.process(
            bone=self.bone_type,
            grouping_iou=0.3,
            grouping_interval=50,
            grouping_min_depth=50,
        )

        obb_segs = []
        for vol_obb, dmean in zip(vols_obb, detection_means):
            if low_memory:
                r = self._predict_via_file(vol_obb)
            else:
                v, p = self._convert_sitk_to_nnunet(vol_obb)
                r = self._nnunet_predictor.predict_single_npy_array(v, p)
                del v, p
                r = sitk.GetImageFromArray(r)
                r.CopyInformation(vol_obb)

            # post process the segmentation
            r = self.post_process(r, dmean)
            obb_segs.append(r)

            # Clear memory after each prediction in low_memory mode
            if low_memory:
                _clear_memory(self._nnunet_predictor.device)

        # update cache (only if caching is enabled)
        if self._caching_enabled:
            self._cache_key = vol_path
            self._cache_result = obb_segs

        return obb_segs

    def _predict_via_file(self, vol_sitk: sitk.Image) -> sitk.Image:
        """Memory-efficient prediction using file-based I/O instead of array.

        This avoids RAM duplication by letting nnUNet handle file loading internally
        with its optimized preprocessing pipeline. Uses sequential (no multiprocessing)
        to avoid spawn context issues.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            input_file = tmpdir / "input_0000.nrrd"
            output_file = tmpdir / "input.nrrd"

            # Write input to temp file
            sitk.WriteImage(vol_sitk, str(input_file), useCompression=False)

            # Use sequential file-based prediction (no multiprocessing, more memory efficient)
            self._nnunet_predictor.predict_from_files_sequential(
                [[str(input_file)]],
                str(tmpdir),
                save_probabilities=False,
                overwrite=True,
            )

            # Read result
            result = sitk.ReadImage(str(output_file))
            result.CopyInformation(vol_sitk)

        _clear_memory(self._nnunet_predictor.device)
        return result

    def _predict_og(self, vol_path: str, vol_input: sitk.Image, low_memory: bool = False) -> List[sitk.Image]:
        """Predicts the segmentation of the bone. Only supports Identity Direction Matrix volumes. i.e nii.gz are currently unsupported, due to flip along 2nd axis.

        Args:
            vol_path: Path to the volume to segment
            low_memory: Use file-based prediction to reduce RAM usage

        Returns:
            List of sitk.Image objects

            The list is structured as follows:
            [
                detected_bone1,
                detected_bone2,
                ...
            ]
        """
        if self._caching_enabled and self._cache_key == vol_path:
            return self._cache_result

        # get detection means
        centroider = armcrop.Centroids(vol_input, detection_confidence=0.2, detection_iou=0.5)
        detection_means = centroider.process(
            bone=self.bone_type,
            grouping_iou=0.3,
            grouping_interval=50,
            grouping_min_depth=50,
        )

        segs = []
        for dmean in detection_means:
            # Crop around centroid, resample crop, predict, then place back
            crop_seg = self._predict_cropped_region(vol_input, dmean, low_memory=low_memory)
            # post process the segmentation
            crop_seg = self.post_process(crop_seg, dmean)
            segs.append(crop_seg)

            if low_memory:
                _clear_memory(self._nnunet_predictor.device)

        # update cache (only if caching is enabled)
        if self._caching_enabled:
            self._cache_key = vol_path
            self._cache_result = segs

        return segs

    def _predict_cropped_region(
        self,
        vol_input: sitk.Image,
        detection_centroids: np.ndarray,
        low_memory: bool = False,
        padding_mm: float = 50.0,
        min_size_mm: tuple = (200, 200, 200),
    ) -> sitk.Image:
        """Crop around a centroid, resample to 0.5mm, predict, and place result back.

        Args:
            vol_input: Full input volume
            detection_centroids: Array of physical coordinates (N, 3) for detection centroids
            low_memory: Use file-based prediction
            padding_mm: Padding around centroid in mm
            min_size_mm: Minimum crop size in mm (x, y, z)

        Returns:
            Segmentation in original volume space (same size as vol_input)
        """
        spacing = np.array(vol_input.GetSpacing())
        size = np.array(vol_input.GetSize())
        origin = np.array(vol_input.GetOrigin())

        # Get center of all detection centroids for this bone
        detection_centroids = np.atleast_2d(detection_centroids)
        centroid = np.mean(detection_centroids, axis=0)

        # Convert centroid from physical to index coordinates
        centroid_idx = np.array(vol_input.TransformPhysicalPointToIndex(centroid.tolist()))

        # Calculate padding in voxels
        padding_vox = (padding_mm / spacing).astype(int)

        # Calculate minimum size in voxels
        min_size_vox = (np.array(min_size_mm) / spacing).astype(int)

        # Calculate crop region: start from centroid, extend by padding or min_size/2
        half_size = np.maximum(padding_vox, min_size_vox // 2)

        crop_start = centroid_idx - half_size
        crop_end = centroid_idx + half_size

        # Constrain to image bounds
        crop_start = np.maximum(crop_start, 0)
        crop_end = np.minimum(crop_end, size)
        crop_size = crop_end - crop_start

        # Ensure minimum size (expand if constrained on one side)
        for i in range(3):
            if crop_size[i] < min_size_vox[i]:
                deficit = min_size_vox[i] - crop_size[i]
                # Try to expand on the side with room
                if crop_start[i] > 0:
                    expand = min(deficit, crop_start[i])
                    crop_start[i] -= expand
                    deficit -= expand
                if deficit > 0 and crop_end[i] < size[i]:
                    expand = min(deficit, size[i] - crop_end[i])
                    crop_end[i] += expand
                crop_size[i] = crop_end[i] - crop_start[i]

        # Extract ROI
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetIndex(crop_start.astype(int).tolist())
        roi_filter.SetSize(crop_size.astype(int).tolist())
        vol_crop = roi_filter.Execute(vol_input)

        # Resample crop to 0.5mm isotropic
        target_spacing = (0.5, 0.5, 0.5)
        new_size = [int(round(s * sp / 0.5)) for s, sp in zip(vol_crop.GetSize(), vol_crop.GetSpacing())]
        vol_res = sitk.Resample(
            vol_crop,
            new_size,
            sitk.Transform(),
            sitk.sitkBSpline,
            vol_crop.GetOrigin(),
            target_spacing,
            vol_crop.GetDirection(),
            -1023,
        )
        del vol_crop

        # Run prediction on resampled crop
        if low_memory:
            seg_res = self._predict_via_file(vol_res)
        else:
            v, p = self._convert_sitk_to_nnunet(vol_res)
            seg_res = self._nnunet_predictor.predict_single_npy_array(v, p)
            del v, p
            seg_res = sitk.GetImageFromArray(seg_res)
            seg_res.CopyInformation(vol_res)

        del vol_res

        # Resample segmentation back to original crop spacing using nearest neighbor
        crop_origin = origin + crop_start * spacing
        seg_crop = sitk.Resample(
            seg_res,
            crop_size.astype(int).tolist(),
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            crop_origin.tolist(),  # Physical origin of crop region
            spacing.tolist(),
            vol_input.GetDirection(),
            0,
        )
        del seg_res

        # Place segmentation back into full volume space
        result = sitk.Image(size.astype(int).tolist(), sitk.sitkUInt8)
        result.SetSpacing(spacing.tolist())
        result.SetOrigin(origin.tolist())
        result.SetDirection(vol_input.GetDirection())

        # Paste the crop into the result
        result = sitk.Paste(
            result,
            seg_crop,
            seg_crop.GetSize(),
            [0, 0, 0],
            crop_start.astype(int).tolist(),
        )
        del seg_crop

        return result

    def predict(
        self,
        vol_path: str | pathlib.Path,
        crop=True,
        low_memory=False,
    ) -> List[sitk.Image]:
        """Predicts the segmentation of the bone. Only supports Identity Direction Matrix volumes. i.e nii.gz are currently unsupported, due to flip along 2nd axis.

        Args:
            vol_path: Path to the volume to segment
            crop: Use oriented bounding box cropping for prediction
            low_memory: Use file-based prediction to reduce RAM usage.
                       Trades some speed for significantly lower memory footprint.
                       Recommended when processing large volumes or running out of RAM.

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

        output_segs = []
        if crop:
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
            for r in self._predict_obb(str(vol_path), vol_input, low_memory=low_memory):
                # unalign the segmentation
                r = Unaligner(r)
                # post process the segmentation
                r = self.post_process(r)
                output_segs.append(r)
        else:
            for r in self._predict_og(str(vol_path), vol_input, low_memory=low_memory):
                # post process the segmentation
                r = self.post_process(r)
                output_segs.append(r)

        return output_segs

    def predict_poly(
        self,
        vol_path: str | pathlib.Path,
        crop=True,
        smooth_iter=30,
        smooth_passband=0.01,
        closing=True,
        low_memory=False,
    ) -> List[List[vtk.vtkPolyData]]:
        """Predicts the segmentation of the bone and returns a list of vtkPolyData objects.

        Args:
            vol_path: Path to the volume to segment
            crop: Use oriented bounding box cropping for prediction
            smooth_iter: Number of smoothing iterations
            smooth_passband: Passband for the windowed sinc filter
            closing: Apply morphological closing to the cortical bone
            low_memory: Use file-based prediction to reduce RAM usage.
                       Trades some speed for significantly lower memory footprint.

        Returns:
            List of vtkPolyData objects

            The list is structured as follows:
            [
                [detected_bone1-cortical, detected_bone1-trabecular],
                [detected_bone2-cortical, detected_bone2-trabecular],
                ...
            ]
        """
        vol_input = sitk.ReadImage(str(vol_path))
        if crop:
            outputs = self._predict_obb(str(vol_path), vol_input, low_memory=low_memory)
        else:
            outputs = self._predict_og(str(vol_path), vol_input, low_memory=low_memory)

        results = []
        for r in outputs:
            polys = []
            for label in [2, 3]:  # Iterate through labels 2 and 3

                # removes in the internal surface of the cortical bone
                if label == 2:
                    _r = sitk.BinaryThreshold(r, lowerThreshold=2, upperThreshold=4, insideValue=1, outsideValue=0)
                    if closing:
                        for _ in range(3):
                            arr = _force_face_connectivity(sitk.GetArrayFromImage(_r))
                        _r = sitk.GetImageFromArray(arr)
                        _r.CopyInformation(r)

                        _r = sitk.BinaryClosingByReconstruction(_r, kernelRadius=((7, 7, 7)), fullyConnected=True)
                        _r = sitk.Multiply(_r, label)

                    r_vtk = SimpleITK.utilities.vtk.sitk2vtk(_r)
                    # del _r
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

    model = Net("scapula")
    ct = "/mnt/slowdata/ct/arthritic-clinical-half-arm/AAW/AAW.nrrd"
    # scapula_segmentations = model.predict(ct)

    # for i, s in enumerate(scapula_segmentations):
    #     sitk.WriteImage(s, f"AAW_scapula_{i}.seg.nrrd", useCompression=True)

    scapula_polydata = model.predict_poly(ct, crop=False)
    for i, s in enumerate(scapula_polydata):
        for j, p in enumerate(s):
            write_polydata(p, f"AAW_scapula_{i}_{j}.ply")
