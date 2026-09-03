"""Shared utilities for the thesis figures."""

from __future__ import annotations

import io
import os
import pickle
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "generated"
CORRECTED_LONG = (
    ROOT / "evidence_overrides" / "generated" / "results_long_author_corrected.csv"
)
CORRECTED_AGGREGATE = (
    ROOT / "evidence_overrides" / "generated" / "results_aggregate_author_corrected.csv"
)
DL4BI = Path(os.environ.get("DEEPRV_RESULTS_ROOT", ROOT))

TEACHERS = ["Bilinear", "Cubic", "DTC", "FITC"]
COLORS = {
    "Bilinear": "#3775BA",
    "Cubic": "#188977",
    "DTC": "#8B5EA7",
    "FITC": "#B64342",
    "Exact": "#272727",
    "Full GP": "#767676",
}
MARKERS = {8: "o", 16: "s", 32: "^"}
MEAN_MARKER_SIZE = 3.5
MEAN_LINE_WIDTH = 1.2
ERRORBAR_CAP_SIZE = 2.0


def apply_style(font_size: float = 8.0) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["axes.titlesize"] = font_size + 0.5
    plt.rcParams["xtick.labelsize"] = font_size - 0.5
    plt.rcParams["ytick.labelsize"] = font_size - 0.5
    plt.rcParams["legend.fontsize"] = font_size - 0.5
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["pdf.fonttype"] = 42


def add_panel_label(ax, label: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig, stem: str, dpi: int = 600) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def read_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(CORRECTED_LONG), pd.read_csv(CORRECTED_AGGREGATE)


def aggregate_value(
    aggregates: pd.DataFrame, grid: str, model: str, metric: str
) -> tuple[float, float]:
    row = aggregates[
        (aggregates["grid"] == grid)
        & (aggregates["model"] == model)
        & (aggregates["metric"] == metric)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one aggregate row: {grid}, {model}, {metric}")
    return float(row.iloc[0]["mean"]), float(row.iloc[0]["sample_sd"])


def _reconstruct_pickled_jax_array(
    reconstruct_func, reconstruct_args, state, metadata=None
):
    obj = reconstruct_func(*reconstruct_args)
    obj.__setstate__(state)
    return obj


class _JaxArraySafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "jax._src.array" and name == "_reconstruct_array":
            return _reconstruct_pickled_jax_array
        return super().find_class(module, name)


def read_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return _JaxArraySafeUnpickler(handle).load()


def read_zip_pickle(zip_path: Path, member: str) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        return _JaxArraySafeUnpickler(io.BytesIO(archive.read(member))).load()


def posterior_mean_local(path: Path, side: int) -> np.ndarray:
    with np.load(path) as loaded:
        return np.asarray(loaded["obs"], dtype=float).mean(axis=0).reshape(side, side)


def posterior_mean_zip(zip_path: Path, member: str, side: int) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(member)
    with np.load(io.BytesIO(payload)) as loaded:
        return np.asarray(loaded["obs"], dtype=float).mean(axis=0).reshape(side, side)


def posterior_sd_local(path: Path, side: int) -> np.ndarray:
    """Sample SD of retained posterior-predictive count draws by location."""
    with np.load(path) as loaded:
        return np.asarray(loaded["obs"], dtype=float).std(ddof=1, axis=0).reshape(
            side, side
        )


def posterior_sd_zip(zip_path: Path, member: str, side: int) -> np.ndarray:
    """Sample SD of archived posterior-predictive count draws by location."""
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(member)
    with np.load(io.BytesIO(payload)) as loaded:
        return np.asarray(loaded["obs"], dtype=float).std(ddof=1, axis=0).reshape(
            side, side
        )


def common_image_limits(images: list[np.ndarray]) -> tuple[float, float]:
    transformed = np.concatenate([np.log1p(image).ravel() for image in images])
    return float(transformed.min()), float(transformed.max())


def draw_metric_panel(
    ax,
    long: pd.DataFrame,
    aggregates: pd.DataFrame,
    grid: str,
    metric: str,
    ylabel: str,
    sides: tuple[int, int, int],
    log_scale: bool = False,
    coverage: bool = False,
    show_seed_points: bool = True,
) -> None:
    subset = long[(long["grid"] == grid) & (long["family"] == "DeepRV")]
    for teacher in TEACHERS:
        means, sds = [], []
        for side in sides:
            mean, sd = aggregate_value(
                aggregates, grid, f"DeepRV {teacher} {side}", metric
            )
            means.append(mean)
            sds.append(sd)
            values = subset[
                (subset["teacher"] == teacher)
                & (subset["inducing_side"] == side)
            ][metric]
            if show_seed_points:
                ax.scatter(
                    np.full(len(values), side),
                    values,
                    s=9,
                    facecolors="white",
                    edgecolors=COLORS[teacher],
                    linewidths=0.7,
                    zorder=3,
                )
        ax.errorbar(
            sides,
            means,
            yerr=sds,
            color=COLORS[teacher],
            marker="o",
            markersize=MEAN_MARKER_SIZE,
            linewidth=MEAN_LINE_WIDTH,
            capsize=ERRORBAR_CAP_SIZE,
            label=teacher,
            zorder=2,
        )
    if coverage:
        full, _ = aggregate_value(aggregates, grid, "Full GP", metric)
        exact, _ = aggregate_value(aggregates, grid, "Exact DeepRV", metric)
        ax.axhline(0.9, color="#9A9A9A", linestyle=":", linewidth=0.9)
        ax.axhline(full, color=COLORS["Full GP"], linestyle="--", linewidth=0.9)
        ax.axhline(exact, color=COLORS["Exact"], linestyle="-.", linewidth=0.9)
    else:
        exact, _ = aggregate_value(aggregates, grid, "Exact DeepRV", metric)
        ax.axhline(exact, color=COLORS["Exact"], linestyle="--", linewidth=0.9)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(sides)
    ax.set_xlabel("Inducing side")
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out", length=2.5, width=0.8)


def draw_map_row(
    axes,
    images: list[np.ndarray],
    titles: list[str],
    panel_labels: list[str],
):
    vmin, vmax = common_image_limits(images)
    artist = None
    for ax, image, title, label in zip(axes, images, titles, panel_labels):
        artist = ax.imshow(
            np.log1p(image),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(title, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label, x=-0.08, y=1.02)
    return artist


def draw_difference_and_mask_row(
    axes,
    reference: np.ndarray,
    comparisons: list[np.ndarray],
    titles: list[str],
    panel_labels: list[str],
    observed_mask: np.ndarray,
):
    """Draw count-mean absolute differences and the paired observation mask."""
    differences = [np.abs(image - reference) for image in comparisons]
    vmax = max(float(difference.max()) for difference in differences)
    artist = None
    for ax, difference, title, label in zip(
        axes[:-1], differences, titles, panel_labels[:-1]
    ):
        artist = ax.imshow(
            difference,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(title, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label, x=-0.08, y=1.02)

    mask_axis = axes[-1]
    mask_axis.imshow(
        observed_mask,
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="lower",
    )
    mask_axis.set_title("Observation mask", pad=3)
    mask_axis.set_xticks([])
    mask_axis.set_yticks([])
    for spine in mask_axis.spines.values():
        spine.set_visible(False)
    add_panel_label(mask_axis, panel_labels[-1], x=-0.08, y=1.02)
    return artist
