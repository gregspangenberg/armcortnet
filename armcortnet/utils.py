import vtk
import pathlib


def write_polydata(polydata: vtk.vtkPolyData, path: str | pathlib.Path) -> None:
    """Writes a vtkPolyData object to a file. Supported formats are: .stl, .ply, .obj, .vtk

    Args:
        polydata: vtkPolyData object to write
        path: Path to write the object to
    """
    path = pathlib.Path(path)  # Ensure path is a Path object

    if path.suffix == ".stl":
        writer = vtk.vtkSTLWriter()
    elif path.suffix == ".ply":
        writer = vtk.vtkPLYWriter()
    elif path.suffix == ".obj":
        writer = vtk.vtkOBJWriter()
    elif path.suffix == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    else:
        raise ValueError("Unsupported file format. Supported filetypes are: .stl, .ply, .obj, .vtk")

    writer.SetInputData(polydata)
    writer.SetFileName(str(path))  # Convert back to string for VTK
    writer.Write()
