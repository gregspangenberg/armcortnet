import armcortnet
import SimpleITK as sitk
from pathlib import Path

BONE_TYPE = "humerus"  # options are 'humerus' or 'scapula'


input_folder = Path(__file__).parent / "input"
output_folder = Path(__file__).parent / "output" / BONE_TYPE
output_folder.mkdir(parents=True, exist_ok=True)


# initialize the segmentation model
model = armcortnet.Net(bone_type=BONE_TYPE)

for file in input_folder.glob("*[!seg].nrrd"):

    # perform mesh prediction on a CT volume, returns list of vtkPolyData objects
    pred_meshes = model.predict_poly(vol_path=file)
    # iterate over each detected object
    for i, cort_trab_polys in enumerate(pred_meshes):
        # iterate over the cortical and trabecular meshes
        for j, poly in enumerate(cort_trab_polys):
            if j == 0:
                ext = ""
            elif j == 1:
                ext = "_trab"
            else:
                raise ValueError("Unexpected number of meshes returned by the model.")
            armcortnet.write_polydata(poly, output_folder / f"{file.stem}_{i}{ext}.ply")
