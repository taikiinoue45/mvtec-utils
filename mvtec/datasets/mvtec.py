from pathlib import Path
from typing import Tuple, Union

import cv2
import pandas as pd
from albumentations import Compose
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import Literal


class MVTecDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        data_type: Literal["train", "val", "test"],
        category: str,
        transform: Compose,
    ) -> None:

        """PyTorch custom dataset for MVTec AD dataset

        Args:
            data_dir (Union[Path, str]): Path to MVTec AD dataset.
            data_type (Literal['train', 'val', 'test']): Type of data.
            category (str): Category of product.
            transform (Compose): Preprocessing defined by albumentations.
        """

        self.data_dir = Path(data_dir)
        self.transform = transform
        info_csv = pd.read_csv(f"{self.data_dir}/info.csv")
        info_csv = info_csv.query("data_type==@data_type & category==@category")
        info_csv.to_csv(f"{data_type}_info.csv", index=False)
        self.stems = info_csv["stem"].tolist()

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor]:

        stem = self.stems[index]
        img = cv2.imread(f"{self.data_dir}/images/{stem}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f"{self.data_dir}/masks/{stem}.png", cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1
        data = self.transform(image=img, mask=mask)
        return (stem, data["image"], data["mask"])

    def __len__(self) -> int:

        return len(self.stems)
