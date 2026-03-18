from pathlib import Path

import SimpleITK as sitk

import armcortnet

BONE_TYPE = "scapula"  # options are 'humerus' or 'scapula'


if __name__ == "__main__":
    input_folder = Path(__file__).parent / "input"
    output_folder = Path(__file__).parent / "output" / BONE_TYPE
    output_folder.mkdir(parents=True, exist_ok=True)

    # initialize the segmentation model
    model = armcortnet.Net(bone_type=BONE_TYPE)

    for file in input_folder.glob("*.nii.gz"):
        # perform mesh prediction on a CT volume, returns list of vtkPolyData objects
        pred = model.predict(vol_path=file)
        # iterate over each detected object
        for i, p in enumerate(pred):
            sitk.WriteImage(p, output_folder / f"{file.stem}.gz", useCompression=True)

        # Clear memory between files
        model.clear_cache()
