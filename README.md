# *armcortnet*
[![PyPI Latest Release](https://img.shields.io/pypi/v/armcortnet.svg)](https://pypi.org/project/armcortnet)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Armcortnet provides automatic segmentation of the humerus and scapula from CT scans. The deep learning model is trained to also segment out the cortical and trabecular subregions from each bone as well.


The deep learning pipeple consists of using [armcrop](https://pypi.org/project/armcrop/) to crop to an oriented bounding box around each humerus or scapula in the image and then a neural network based traine from the nnUNet framework segments that cropped volume. The segmetnation is then transformed back to the original coordinate system, post-processed and finally saved as a .seg.nrrd file.

## Installation
Please install pytorch first before installing armcortnet. You can learn about installing pytorch from the official website [here](https://pytorch.org/get-started/locally/).

Then install armcortnet using pip:
```bash
pip install armcortnet
```
For faster oriented bounding box cropping you can replace onnxruntime with onnxruntime-gpu.

## Usage
To generate a segmentation of the humerus or scapula from a CT volume, use the following:
```python
import armcortnet
import SimpleITK as sitk

# initialize the segmentation model
model = armcortnet.Net(bone_type="scapula")  # or "humerus"

# perform segmentation prediction on a CT volume
pred_segmentations = model.predict(
    vol_path="path/to/input/ct.nrrd"
)
# output is a list of SimpleITK images, one for each bone_type detected in the CT
for i, pred_seg in enumerate(pred_segmentations):
    # write each of the segmentations to the disk
    sitk.WriteImage(pred_seg, f"scapula-{i}.seg.nrrd")

```
A mesh of the predicted bone can be generated using the following:
```python
# perform mesh prediction on a CT volume, returns list of vtkPolyData objects
pred_meshes = model.predict_poly(
    vol_path="path/to/input/ct.nrrd"
)

# iterate over each detected object
for i, cort_trab_polys in enumerate(pred_meshes):
    # iterate over the cortical and trabecular meshes
    for j, poly in enumerate(cort_trab_polys):
        armcortnet.write_polydata(p, f"scapula_{i}_{j}.ply")
```


## Output Labels

The segmentation output contains the following labels:
- 0: Background
- 1: Other adjacent bones ("i.e clavicle, radius, ulna, etc.")
- 2: Cortical region of bone of interest
- 3: Trabecular region of bone of interest

Note: label 1 is removed when post-processing is used

## Models
Trained models are automatically downloaded from HuggingFace Hub (`gregspangenberg/armcortnet`) on first use.

