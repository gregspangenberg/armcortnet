import scipy.stats
import yaml
import pathlib
import numpy as np

# select parameters
BONE_TYPE = "humerus"
SPLIT_TYPE = "test"

results_file = pathlib.Path(f"tests/seg/{BONE_TYPE}_{SPLIT_TYPE}_metrics.yaml")
with open(results_file, "r") as f:
    metrics_records = yaml.load(f, Loader=yaml.FullLoader)

# calculate descriptive statistics
for name, metric in metrics_records.items():
    for key, value in metric.items():
        q1 = np.percentile(value, 25)
        q3 = np.percentile(value, 75)
        median = np.median(value)
        print(
            f"{name:<16} {key:<5}: {median:>6.2f} ({q1:>6.2f}-{q3:>6.2f}) mm"
        )
