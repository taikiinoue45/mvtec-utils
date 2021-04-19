import os
import subprocess

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
from tqdm import tqdm


def savegif(imgs: NDArray, masks: NDArray, amaps: NDArray) -> None:

    os.mkdir("results")
    pbar = tqdm(enumerate(zip(imgs, masks, amaps)), desc="savefig")
    for i, (img, mask, amap) in pbar:

        grid = ImageGrid(
            fig=plt.figure(figsize=(12, 4)),
            rect=111,
            nrows_ncols=(1, 3),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        grid[0].imshow(img)
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[0].set_title("Input Image", fontsize=18)

        grid[1].imshow(img)
        grid[1].imshow(mask, alpha=0.5)
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("Ground Truth", fontsize=18)

        grid[2].imshow(img)
        im = grid[2].imshow(amap, alpha=0.3, cmap="jet", vmin=0, vmax=1)
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].cax.toggle_label(True)
        grid[2].set_title("Anomaly Map", fontsize=18)

        plt.colorbar(im, cax=grid.cbar_axes[0])
        plt.savefig(f"results/{i}.png", bbox_inches="tight")
        plt.close()

    # NOTE(inoue): The gif files converted by PIL or imageio were low-quality.
    #              So, I used ImageMagick command, instead of them.
    subprocess.run("convert -delay 100 -loop 0 results/*.png result.gif", shell=True)
