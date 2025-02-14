# armcortnet

Armcortnet provides automatic segmentation of the humerus and scapula from CT scans. The deep learning model is trained to also segment out the cortical and trabecular subregions from each bone as well.


The deep learning pipeple consists of using armcrop to crop to an oriented bounding box around each humerus or scapula in the image and then a neural network based traine from the nnUNet framework segments that cropped volume. The segmetnation is then transformed back to the original coordinate system, post-processed and finally saved as a .seg.nrrd file.

## Installation

```bash
pip install armcortnet
```

## Usage

```python
from armcortnet import Net

# Initialize segmentation model
model = Net(bone_type="scapula")  # or "humerus"

# Perform segmentation
model.predict(
    vol_path="path/to/input/ct.nrrd",
    output_seg_path="path/to/output/segmentation.seg.nrrd"
)
```

## Output Labels

The segmentation output contains the following labels:
- 0: Background
- 1: Other adjacent bones ("i.e clavicle, radius, ulna, etc.")
- 2: Cortical region of bone of interest
- 3: Trabecular region of bone of interest

## Models
Trained models are automatically downloaded from HuggingFace Hub (`gregspangenberg/armcortnet`) on first use.

