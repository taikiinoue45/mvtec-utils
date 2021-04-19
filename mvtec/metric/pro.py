from statistics import mean

import mlflow
import numpy as np
import pandas as pd
from numpy import ndarray as NDArray
from skimage import measure
from sklearn.metrics import auc


def pro(masks: NDArray, amaps: NDArray, fpr_th: float = 0.3, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 fpr
    Args:
        amaps (NDArray): All anomaly maps in the test dataset. amaps.shape -> (num_test_data, h, w)
        masks (NDArray): All binary masks in the test dataset. masks.shape -> (num_test_data, h, w)
        num_th (int): Number of thresholds
    """

    assert isinstance(amaps, NDArray), "Type of amaps must be NDArray"
    assert isinstance(masks, NDArray), "Type of masks must be NDArray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "Shape of amaps and masks must be same"
    assert set(masks.flatten()) == {0, 1}, "Elements of masks must be 0 or 1"
    assert isinstance(num_th, int), "Type of num_th must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Save pro.csv
    df.to_csv("pro.csv", index=False)

    # Compute pro30
    df = df[df["fpr"] < fpr_th]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro30_auc = auc(df["fpr"], df["pro"])

    # Logging pro30 auc to mlflow server
    mlflow.log_metric("pro30_auc", value=pro30_auc)

    # Logging pro30 curve to mlflow server
    # TODO(inoue): step in log_metric only accept int, so fpr is multiplied by 100 and rounded.
    for fpr, pro in zip(df["fpr"], df["pro"]):
        mlflow.log_metric("pro30_curve", value=round(pro * 100), step=round(fpr * 100))
