import monai.metrics
import monai.transforms
import torch
import numpy as np
import pathlib
import SimpleITK as sitk
import yaml
from pprint import pprint
import gc
import scipy.stats


gc.enable()
torch.set_default_dtype(torch.float32)

# select parameters
BONE_TYPE = "scapula"
SPLIT_TYPE = "val"


def get_data(bone_type: str, split_type: str):
    pred_folder = pathlib.Path("tests/seg") / bone_type / split_type
    pred_files = sorted(pred_folder.rglob("*seg.nrrd"))

    gt_folder = (
        pathlib.Path("/home/greg/projects/segment/stage2_net_training/database/seg") / bone_type
    )
    gt_files = [gt_folder / gt.name.replace("-0.", ".").replace("-1.", ".") for gt in pred_files]

    return zip(pred_files, gt_files)


def closest_object(pred, gt):
    gt_mask = sitk.BinaryThreshold(
        gt, lowerThreshold=2, upperThreshold=4, insideValue=1, outsideValue=0
    )
    pred_mask = sitk.BinaryThreshold(
        pred, lowerThreshold=2, upperThreshold=4, insideValue=1, outsideValue=0
    )

    gt_cc = sitk.RelabelComponent(sitk.ConnectedComponent(gt_mask))

    gt_cc_stats = sitk.LabelShapeStatisticsImageFilter()
    gt_cc_stats.ComputeOrientedBoundingBoxOn()
    gt_cc_stats.Execute(gt_cc)

    pred_stats = sitk.LabelShapeStatisticsImageFilter()
    pred_stats.ComputeOrientedBoundingBoxOn()
    pred_stats.Execute(pred_mask)
    pred_centroid = pred_stats.GetCentroid(1)

    dists = []
    gt_labels = gt_cc_stats.GetLabels()
    for l in gt_labels:
        gt_centroid = gt_cc_stats.GetCentroid(l)
        dists.append(np.linalg.norm(np.array(pred_centroid) - np.array(gt_centroid)))

    cloest_label = gt_labels[np.argmin(dists)]

    return sitk.Multiply(gt, gt_cc == cloest_label)


metrics = {
    # Set to_onehot=True to handle label format inputs
    "dice": monai.metrics.DiceMetric(include_background=False, reduction="none"),
    "hausdorff": monai.metrics.HausdorffDistanceMetric(include_background=False, reduction="none"),
    "hausdorff99": monai.metrics.HausdorffDistanceMetric(
        include_background=False,
        reduction="none",
        percentile=99,
    ),
    "asd": monai.metrics.SurfaceDistanceMetric(include_background=False, reduction="none"),
}
results_file = pathlib.Path(f"tests/seg/{BONE_TYPE}_{SPLIT_TYPE}_metrics.yaml")
if not results_file.exists():
    metrics_records = {name: {"all": [], "cort": [], "trab": []} for name in metrics.keys()}
    for pred, gt in get_data(BONE_TYPE, SPLIT_TYPE):

        print("\n", pred.name, gt.name)

        pred = sitk.ReadImage(str(pred), sitk.sitkUInt8)
        gt = sitk.ReadImage(str(gt), sitk.sitkUInt8)

        # discard further away bones
        gt = closest_object(pred, gt)
        spacing = gt.GetSpacing()

        gt = torch.Tensor(sitk.GetArrayFromImage(gt).astype(np.uint8))
        pred = torch.Tensor(sitk.GetArrayFromImage(pred).astype(np.uint8))

        gc.collect()

        # one hot encode for each class
        gt = gt.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)

        # get the masks
        pall = ((pred == 2) | (pred == 3)).to(torch.bool)
        gall = ((gt == 2) | (gt == 3)).to(torch.bool)
        p2 = (pred == 2).to(torch.bool)
        g2 = (gt == 2).to(torch.bool)
        p3 = (pred == 3).to(torch.bool)
        g3 = (gt == 3).to(torch.bool)
        del gt, pred
        gc.collect()

        # Calculate each metric
        for name, metric in metrics.items():
            if name == "dice":
                all_measure = float(metric(pall, gall))
                cort_measure = float(metric(p2, g2))
                trab_measure = float(metric(p3, g3))
            else:
                all_measure = float(metric(pall, gall, spacing=spacing))
                cort_measure = float(metric(p2, g2, spacing=spacing))
                trab_measure = float(metric(p3, g3, spacing=spacing))
            # print(f"{name}: all={all_measure:.3f}, cort={cort_measure:.3f}, trab={trab_measure:.3f}")
            print(f"{name}: all={all_measure}, cort={cort_measure}, trab={trab_measure:.3f}")

            # break
            metric.reset()
            metrics_records[name]["all"].append(all_measure)
            metrics_records[name]["cort"].append(cort_measure)
            metrics_records[name]["trab"].append(trab_measure)

    print(metrics_records)

    # save metrics to yaml
    with open(results_file, "w") as f:
        yaml.dump(metrics_records, f)

else:
    with open(results_file, "r") as f:
        metrics_records = yaml.load(f, Loader=yaml.FullLoader)

    # calculate descriptive statistics
    for name, metric in metrics_records.items():
        for key, value in metric.items():
            print(
                f"{name:<16} {key:<5}: mean={np.median(value):>8.2f} {scipy.stats.iqr(value,axis=0):>8.2f}"
            )
