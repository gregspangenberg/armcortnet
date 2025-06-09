import SimpleITK as sitk
import os
import pathlib


def load_dicom_series(directory):
    # Read the DICOM series using SimpleITK
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def convert_to_nrrd(image, output_path):
    # Save the image as an NRRD file
    sitk.WriteImage(image, output_path, useCompression=True)


# folder containing folders of dicoms
for folder in pathlib.Path("/mnt/slowdata/ct/cadaveric-full-arm").glob("*"):
    print(folder.name)
    if not folder.is_dir():
        continue
    nrrd_name = folder / f"{folder.name}.nrrd"
    if nrrd_name.exists():
        print(f"Skipping {nrrd_name} as it already exists.")
        continue

    dicom_dirs = [f for f in folder.glob("*") if f.is_dir()]
    for dcm_dir in dicom_dirs:
        image = load_dicom_series(dcm_dir)
        convert_to_nrrd(image, nrrd_name)
