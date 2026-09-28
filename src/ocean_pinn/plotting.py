"""Figures comparing observed and predicted temperature."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Recessive chrome so the data carries the contrast.
_STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.titlecolor": "#0b0b0b",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#c3c2b7",
    "ytick.color": "#c3c2b7",
    "xtick.labelcolor": "#52514e",
    "ytick.labelcolor": "#52514e",
}


def plot_temperature_comparison(
    time: np.ndarray,
    depth: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    gridsize: int = 200,
    cmap: str = "YlOrRd",
) -> Figure:
    """Side-by-side maps of observed and predicted temperature over time and depth.

    Each hexagon shows the mean temperature (°C) of the points it contains.
    `time` holds a datetime64 and `depth` a depth in m (positive down) per point.
    """
    vmin = min(observed.min(), predicted.min())
    vmax = max(observed.max(), predicted.max())
    dates = mdates.date2num(time)

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
        panels = zip(axes, (observed, predicted), ("Observed", "PINN prediction"), strict=True)
        for ax, values, title in panels:
            hexes = ax.hexbin(
                dates,
                depth,
                C=values,
                gridsize=gridsize,
                reduce_C_function=np.mean,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.set_title(title)
        axes[0].set_ylabel("Depth (m)")
        axes[0].invert_yaxis()  # shared with the second panel
        fig.colorbar(hexes, ax=axes, label="Temperature (°C)")
    return fig
