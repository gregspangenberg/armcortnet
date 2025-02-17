# armcortnet

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

```python
from armcortnet import Net
import SimpleITK as sitk

# initialize the segmentation model
model = Net(bone_type="scapula")  # or "humerus"

# perform segmentation prediction on a CT volume
pred_segmentations = model.predict(
    vol_path="path/to/input/ct.nrrd",
)
# output is a list of SimpleITK images, one for each bone_type detected in the CT
for i, pred_seg in enumerate(pred_segmentations):
    # write each of the segmentations to the disk
    sitk.WriteImage(pred_seg, f"scapula-{i}.seg.nrrd")

```

## Output Labels

The segmentation output contains the following labels:
- 0: Background
- 1: Other adjacent bones ("i.e clavicle, radius, ulna, etc.")
- 2: Cortical region of bone of interest
- 3: Trabecular region of bone of interest

## Models
Trained models are automatically downloaded from HuggingFace Hub (`gregspangenberg/armcortnet`) on first use.

