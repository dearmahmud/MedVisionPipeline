# task1_classification/plot_grids.py
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def _flatten_axes(axes):
    # axes can be Axes, ndarray of Axes, or a list of Axes
    if isinstance(axes, np.ndarray):
        return axes.flatten().tolist()
    if isinstance(axes, (list, tuple)):
        return list(axes)
    # single Axes
    return [axes]

def show_images_grid(
    paths,
    cols=2,
    rows=2,
    figsize=(12, 10),
    suptitle=None,
    paginate=True,
    tight=True
):
    """
    Display images in a grid of (rows x cols). If paginate=True, it shows
    multiple pages when there are more images than slots.
    """
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("No valid image paths.")
        return

    per_page = cols * rows
    total = len(paths)
    pages = 1 if not paginate else math.ceil(total / per_page)

    for pg in range(pages):
        batch = paths if not paginate else paths[pg * per_page : (pg + 1) * per_page]
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = _flatten_axes(axes)

        for ax, p in zip(axes, batch):
            img = mpimg.imread(p)
            ax.imshow(img)
            ax.set_title(os.path.basename(p), fontsize=11)
            ax.axis("off")

        # Hide unused axes on the last page
        for k in range(len(batch), len(axes)):
            axes[k].axis("off")

        if suptitle:
            fig.suptitle(
                f"{suptitle}" + (f" (page {pg+1}/{pages})" if pages > 1 else ""),
                fontsize=14, y=0.98
            )
        if tight:
            plt.tight_layout()
        plt.show()