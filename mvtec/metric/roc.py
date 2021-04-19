from typing import List, Optional

import mlflow
import pandas as pd
from numpy import ndarray as NDArray
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, roc_curve


def roc(
    y_trues: NDArray,
    y_preds: NDArray,
    epoch: Optional[int] = None,
    stems: Optional[List[str]] = None,
) -> None:

    """Compute image-wise receiver operating characteristic (ROC)
    Args:
        y_trues (NDArray):
            True binary labels. y_true.shape -> (num_test_data,)
        y_preds (NDArray):
            Target predictions. y_pred.shape -> (num_test_data,)
        epoch (Optional[int]):
            During validation,
        stems (Optional[List[str]]):
            List of image filenames withuot suffix. len(stems) -> num_test_data
            During test,
    """

    assert isinstance(y_trues, NDArray), "Type of y_trues must be NDArray"
    assert isinstance(y_preds, NDArray), "Type of y_preds must be NDArray"
    assert y_trues.ndim == 1, "y_trues.ndim must be 1 (num_test_data,)"
    assert y_preds.ndim == 1, "y_preds.ndim must be 1 (num_test_data,)"
    assert set(y_trues) == {0, 1}, "Elements of y_true must be 0 or 1"

    assert (epoch is None) & (stems is None), ""
    assert (epoch is not None) & (stems is not None), ""

    fprs, tprs, thresholds = roc_curve(y_trues, y_preds, pos_label=1, drop_intermediate=False)
    roc_auc = roc_auc_score(y_trues, y_preds)

    if epoch:
        # Logging roc auc to mlflow server
        mlflow.log_metric("val - roc_auc", value=roc_auc, step=epoch)

    if stems:
        # Logging roc auc to mlflow server
        mlflow.log_metric("test - roc_auc", value=roc_auc)

        # Logging roc curve to mlflow server
        # TODO(inoue): step in log_metric only accept int, so fpr is multiplied by 100 and rounded
        for fpr, tpr in zip(fprs, tprs):
            mlflow.log_metric("roc_curve", value=round(tpr * 100), step=round(fpr * 100))

        # Save image_roc.csv
        keys = [f"threshold_{i}" for i in range(len(thresholds))]
        roc_df = pd.DataFrame({"key": keys, "fpr": fprs, "tpr": tprs, "threshold": thresholds})
        roc_df.to_csv("roc_curve.csv", index=False)

        # Update test_info.csv
        info_csv = pd.merge(
            pd.DataFrame({"stem": stems, "y_true": y_trues, "y_pred": y_preds}),
            pd.read_csv("test_info.csv"),
            on="stem",
        )
        for i, th in enumerate(thresholds):
            info_csv[f"threshold_{i}"] = info_csv["y_score"].apply(lambda x: 1 if x >= th else 0)
        info_csv.to_csv("test_info.csv", index=False)
