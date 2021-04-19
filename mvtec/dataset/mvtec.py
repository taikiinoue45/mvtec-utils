from pathlib import Path
from typing import List, Tuple, Union

import cv2
import pandas as pd
from albumentations import Compose
from torch import Tensor
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        queries: List[str],
        transforms: Compose,
    ) -> None:

        """PyTorch custom dataset for MVTec AD

        Args:
            data_dir (Union[Path, str]): Path to MVTec AD dataset
            queries (List[str]): List of queries to extract arbitrary rows from info.csv
            transforms (Compose): List of albumentations transforms
        """

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        info_csv = pd.read_csv(self.data_dir / "info.csv")
        self.df = pd.concat([info_csv.query(q) for q in queries])

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor]:

        stem = self.df.loc[index, "stem"]

        img_path = str(self.data_dir / f"images/{stem}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = str(self.data_dir / f"masks/{stem}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1

        data = self.transforms(image=img, mask=mask)
        return (stem, data["image"], data["mask"])

    def __len__(self) -> int:

        return len(self.df)
