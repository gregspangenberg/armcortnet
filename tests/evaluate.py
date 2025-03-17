import armcortnet
import SimpleITK as sitk
import pathlib
import json
from tqdm import tqdm

# set type
BONE_TYPE = "scapula"

vols = pathlib.Path("/mnt/slowdata/ct")
vols = [f for f in vols.rglob("*[!seg].nrrd")]
segs = pathlib.Path(f"/home/greg/projects/segment/stage2_net_training/database/seg/{BONE_TYPE}")
segs = [f for f in segs.rglob("*seg.nrrd")]

# load the json
with open(f"tests/seg/splits_{BONE_TYPE}.json", "r") as f:
    splits = json.load(f)

vals = splits["val"]
vals = [s.replace("-1.", ".").replace("-0.", ".") for s in vals]
tests_ext = splits["test"]
tests_ext = [s.replace("-1.", ".").replace("-0.", ".") for s in tests_ext]


# load model
model = armcortnet.Net(BONE_TYPE)
for seg in tqdm(segs):
    # get volume
    vol = [v for v in vols if v.stem == seg.stem.split(".")[0]]
    if len(vol) == 0:
        print(f"Volume not found for {seg}")
        continue
    vol = vol[0]

    if vol.stem in vals:
        split_type = "val"
    elif vol.stem in tests_ext:
        split_type = "test"
    else:
        continue

    print(f"Processing {vol} and {seg}")
    if (pathlib.Path("tests/seg") / BONE_TYPE / split_type / f"{vol.stem}-0.seg.nrrd").exists():
        continue
    if (pathlib.Path("tests/seg") / BONE_TYPE / split_type / f"{vol.stem}-1.seg.nrrd").exists():
        continue
    # make prediction
    segs = model.predict(vol)

    # save prediction
    for i, s in enumerate(segs):
        print(f"{vol.stem}-{i}.seg.nrrd")
        sitk.WriteImage(
            s,
            pathlib.Path("tests/seg") / BONE_TYPE / split_type / f"{vol.stem}-{i}.seg.nrrd",
            useCompression=True,
        )
