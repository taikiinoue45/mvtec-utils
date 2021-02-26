import tarfile
from pathlib import Path
from typing import Callable, Union
from urllib.request import urlretrieve

from tqdm import tqdm


BASE_URL = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/"

CATEGORY_TO_FILENAME = {
    "all": "mvtec_anomaly_detection.tar.xz",
    "bottle": "bottle.tar.xz",
    "cable": "cable.tar.xz",
    "capsule": "capsule.tar.xz",
    "carpet": "carpet.tar.xz",
    "grid": "grid.tar.xz",
    "hazelnut": "hazelnut.tar.xz",
    "leather": "leather.tar.xz",
    "metal_nut": "metal_nut.tar.xz",
    "pill": "pill.tar.xz",
    "screw": "screw.tar.xz",
    "tile": "tile.tar.xz",
    "toothbrush": "toothbrush.tar.xz",
    "transistor": "transistor.tar.xz",
    "wood": "wood.tar.xz",
    "zipper": "zipper.tar.xz",
}


def download(save_dir: Union[Path, str], category: str = "all") -> None:

    save_dir = Path(save_dir)
    filename = CATEGORY_TO_FILENAME[category]
    url = BASE_URL + filename
    urlretrieve(url, save_dir / filename, reporthook=gen_bar_updater())

    with tarfile.open(from_path, "r:xz") as tar:
        tar.extractall(path=to_path)


def _gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update
