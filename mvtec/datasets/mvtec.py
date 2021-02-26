import os
import tarfile
from pathlib import Path
from typing import Callable, List, Tuple, Union
from urllib.request import urlretrieve

import cv2
import pandas as pd
from albumentations import Compose
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


class MVTecDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        queries: List[str],
        transforms: Compose,
        download: bool = False,
    ) -> None:

        """PyTorch custom dataset for MVTec AD

        Args:
            data_dir (Union[Path, str]): Path to MVTec AD dataset
            queries (List[str]): List of queries to extract arbitrary rows from info.csv
            transforms (Compose): List of albumentations transforms
        """

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        if download:
            self.download()

        info_csv = pd.read_csv(self.data_dir / "info.csv")
        self.df = pd.concat([info_csv.query(q) for q in queries])

    def download(self) -> None:

        os.makedirs(self.data_dir, exist_ok=True)
        base_url = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/"
        filename = "mvtec_anomaly_detection.tar.xz"
        url = base_url + filename
        tarxz_path = self.data_dir / filename
        urlretrieve(url, tarxz_path, reporthook=_gen_bar_updater())

        with tarfile.open(tarxz_path, "r:xz") as tar:
            tar.extractall(path=self.data_dir)

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


def _gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def _bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return _bar_update
