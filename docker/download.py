import os
import tarfile
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def download_mvtec(data_dir: Path) -> None:

    os.makedirs(data_dir, exist_ok=True)
    base_url = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/"
    filename = "mvtec_anomaly_detection.tar.xz"
    url = base_url + filename
    tarxz_path = data_dir / filename
    urlretrieve(url, tarxz_path, reporthook=_gen_bar_updater())

    with tarfile.open(tarxz_path, "r:xz") as tar:
        tar.extractall(path=data_dir)


def _gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def _bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return _bar_update


def create_info_csv(data_dir: Path) -> DataFrame:

    df = pd.DataFrame({})

    for data_type in ["train", "test"]:
        for p in data_dir.glob(f"*/{data_type}/*/*.png"):

            raw_stem = p.stem
            defect = p.parents[0].name
            data_type = p.parents[1].name
            category = p.parents[2].name

            df = df.append(
                {
                    "raw_img_path": str(p),
                    "raw_stem": raw_stem,
                    "defect": defect,
                    "data_type": data_type,
                    "category": category,
                },
                ignore_index=True,
            )

    for category in df["category"].unique():

        category_df = df.query("data_type=='train' & category==@category")
        _, val_index = train_test_split(
            category_df.index.tolist(),
            train_size=0.8,
            test_size=0.2,
            random_state=5,
            shuffle=True,
        )
        df.loc[val_index, "data_type"] = "val"

    df["stem"] = df.apply(
        lambda x: f"{x.category}_{x.data_type}_{x.defect}_{x.raw_stem}",
        axis=1,
    )

    df["raw_mask_path"] = df.apply(
        lambda x: f"{data_dir}/{x.category}/ground_truth/{x.defect}/{x.raw_stem}_mask.png",
        axis=1,
    )

    return df


def move_images_and_masks(data_dir: Path, df: DataFrame) -> None:

    os.makedirs(f"{data_dir}/images", exist_ok=True)
    os.makedirs(f"{data_dir}/masks", exist_ok=True)

    for i in tqdm(df.index):
        raw_img_path, raw_mask_path, stem = df.loc[i, ["raw_img_path", "raw_mask_path", "stem"]]

        if os.path.exists(raw_mask_path):
            os.rename(raw_mask_path, f"{data_dir}/masks/{stem}.png")
        else:
            # create masks for train images
            img = cv2.imread(raw_img_path)
            mask = np.zeros(img.shape)
            cv2.imwrite(f"{data_dir}/masks/{stem}.png", mask)

        os.rename(raw_img_path, f"{data_dir}/images/{stem}.png")

    df.drop(columns=["raw_stem", "raw_img_path", "raw_mask_path"])
    df.to_csv(f"{data_dir}/info.csv", index=False)


if __name__ == "__main__":

    data_dir = Path("/data/MVTec")
    download_mvtec(data_dir)
    df = create_info_csv(data_dir)
    move_images_and_masks(data_dir, df)
