import SimpleITK as sitk
import os
import pathlib



# folder containing folders of dicoms
all_folder = pathlib.Path(r"C:\Users\hulcuser\Desktop\Shoulder CT data for autosegmentation_20241115")


def load_dicom_series(directory):
    # Read the DICOM series using SimpleITK
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def convert_to_nrrd(image, output_path):
    # Save the image as an NRRD file
    sitk.WriteImage(image, output_path)


dicom_dirs = [f for f in all_folder.glob("*") if f.is_dir()]

for dcm_dir in dicom_dirs:
    image = load_dicom_series(dcm_dir)
    convert_to_nrrd(image, all_folder / f"{dcm_dir.stem}.nrrd")



