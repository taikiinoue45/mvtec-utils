import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def create_info_csv(mvtec_dir: Path) -> DataFrame:

    df = pd.DataFrame({})

    for data_type in ["train", "test"]:
        for p in mvtec_dir.glob(f"*/{data_type}/*/*.png"):

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
        lambda x: f"{mvtec_dir}/{x.category}/ground_truth/{x.defect}/{x.raw_stem}_mask.png",
        axis=1,
    )

    return df


def move_images_and_masks(df: DataFrame) -> None:

    os.makedirs("/data/images", exist_ok=True)
    os.makedirs("/data/masks", exist_ok=True)

    for i in df.index:
        raw_img_path, raw_mask_path, stem = df.loc[i, ["raw_img_path", "raw_mask_path", "stem"]]

        if os.path.exists(raw_mask_path):
            os.rename(raw_mask_path, f"/data/masks/{stem}.png")
        else:
            # create masks for train images
            img = cv2.imread(raw_img_path)
            mask = np.zeros(img.shape)
            cv2.imwrite(f"/data/masks/{stem}.png", mask)

        os.rename(raw_img_path, f"/data/images/{stem}.png")

    df.drop(columns=["raw_stem", "raw_img_path", "raw_mask_path"])
    df.to_csv("/data/info.csv", index=False)


if __name__ == "__main__":

    mvtec_dir = Path("/data/MVTec")
    df = create_info_csv(mvtec_dir)
    move_images_and_masks(df)
