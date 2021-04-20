from typing import List

import mlflow
import pandas as pd
from numpy import ndarray
from sklearn.metrics import roc_auc_score, roc_curve


def compute_roc(y_trues: ndarray, y_preds: ndarray, stems: List[str]) -> None:

    """Compute the area under the curve of receiver operating characteristic (ROC)

    Args:
        y_trues (ndarray): All binary labels in test. y_trues.shape -> (num_test_data,)
        y_preds (ndarray): All predictions in test. y_preds.shape -> (num_test_data,)
        stems (List[str]): All image filename without suffix in test. len(stems) -> num_test_data
    """

    assert isinstance(y_trues, ndarray), "type(y_trues) must be ndarray"
    assert isinstance(y_preds, ndarray), "type(y_preds) must be ndarray"
    assert y_trues.ndim == 1, "y_trues.ndim must be 1 (num_test_data,)"
    assert y_preds.ndim == 1, "y_preds.ndim must be 1 (num_test_data,)"
    assert set(y_trues) == {0, 1}, "set(y_trues) must be {0, 1}"

    fprs, tprs, thresholds = roc_curve(y_trues, y_preds, pos_label=1, drop_intermediate=False)
    roc_auc = roc_auc_score(y_trues, y_preds)

    # Logging roc_auc to mlflow server
    mlflow.log_metric("roc_auc", value=roc_auc)

    # Logging roc_curve to mlflow server
    # TODO(inoue): step in log_metric only accept int, so fpr is multiplied by 100 and rounded
    for fpr, tpr in zip(fprs, tprs):
        mlflow.log_metric("roc_curve", value=round(tpr * 100), step=round(fpr * 100))

    # Save roc_curve.csv
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
        info_csv[f"threshold_{i}"] = info_csv["y_pred"].apply(lambda x: 1 if x >= th else 0)
    info_csv.to_csv("test_info.csv", index=False)
