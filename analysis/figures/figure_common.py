"""Shared helpers for the thesis figures."""

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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "generated"
SOURCE_DATA = ROOT / "figure_sources" / "data"
DL4BI = Path(os.environ.get("DEEPRV_RESULTS_ROOT", ROOT))
ARCHIVE_DIR = (
    DL4BI / "outputs" / "oliver_64x64_uniform_obs50_seed0_seed1_unit_zips"
)
DEEPRV_SEED0 = (
    ARCHIVE_DIR / "deeprv_target64_uniform_obs50_allmodels_ls30_decoderseed0_seed0.zip"
)
FULL_GP_SEED0 = (
    ARCHIVE_DIR / "directgp_target64_uniform_obs50_fullgp_reference_ls30_seed0.zip"
)
DIRECT32_SEED0 = (
    ARCHIVE_DIR
    / "directgp_target64_uniform_obs50_inducing32_approxmethods_with_fullref_ls30_seed0.zip"
)

TEACHERS = ("Bilinear", "Cubic", "DTC", "FITC")
COLORS = {
    "Bilinear": "#3B6FA5",
    "Cubic": "#16877A",
    "DTC": "#7A5A9E",
    "FITC": "#B34B4B",
    "Exact": "#264B73",
    "Full GP": "#6C737A",
}


def apply_style(font_size: float = 7.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.3,
            "xtick.labelsize": font_size - 0.4,
            "ytick.labelsize": font_size - 0.4,
            "legend.fontsize": font_size - 0.4,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
    )


def save_figure(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


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


def read_zip_pickle(zip_path: Path, member: str) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(member)
    return _JaxArraySafeUnpickler(io.BytesIO(payload)).load()


def posterior_mean_zip(zip_path: Path, member: str, side: int = 64) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(member)
    with np.load(io.BytesIO(payload)) as loaded:
        draws = np.asarray(loaded["obs"], dtype=float)
    if draws.ndim < 2:
        raise ValueError(f"Unexpected posterior array shape {draws.shape}: {member}")
    mean = draws.mean(axis=0).reshape(side, side)
    if not np.isfinite(mean).all() or np.any(mean < 0):
        raise ValueError(f"Invalid posterior predictive mean: {member}")
    return mean


def load_and_verify_seed0_data() -> dict:
    """Load Seed 0 and prove that all three archives use the same data/mask."""
    archives = (DEEPRV_SEED0, FULL_GP_SEED0, DIRECT32_SEED0)
    for path in archives:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = [read_zip_pickle(path, "seed_0/observed_data.pkl") for path in archives]
    exact_keys = ("y_full", "obs_mask", "s")
    for key in exact_keys:
        arrays = [np.asarray(record[key]) for record in records]
        if not all(np.array_equal(arrays[0], other) for other in arrays[1:]):
            raise ValueError(f"Seed 0 source mismatch across archives: {key}")
    latent_arrays = [np.asarray(record["latent_f"], dtype=float) for record in records]
    if not all(
        np.allclose(latent_arrays[0], other, rtol=1e-6, atol=5e-6)
        for other in latent_arrays[1:]
    ):
        raise ValueError("Seed 0 latent field mismatch exceeds float32 tolerance")
    data = records[0]
    latent = np.asarray(data["latent_f"], dtype=float).reshape(64, 64)
    counts = np.asarray(data["y_full"], dtype=float).reshape(64, 64)
    mask = np.asarray(data["obs_mask"], dtype=bool).reshape(64, 64)
    if int(mask.sum()) != 2048 or mask.size != 4096:
        raise ValueError("Expected exactly 2,048 of 4,096 locations observed")
    return {"latent": latent, "counts": counts, "mask": mask}


def read_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    long_path = SOURCE_DATA / "results_long_author_corrected.csv"
    aggregate_path = SOURCE_DATA / "results_aggregate_author_corrected.csv"
    return pd.read_csv(long_path), pd.read_csv(aggregate_path)


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


def draw_metric_panel(
    ax,
    long: pd.DataFrame,
    aggregates: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    log_scale: bool = False,
    coverage: bool = False,
) -> None:
    sides = (8, 16, 32)
    subset = long[(long["grid"] == "64x64") & (long["family"] == "DeepRV")]
    for teacher in TEACHERS:
        means, sds = [], []
        for side in sides:
            mean, sd = aggregate_value(
                aggregates, "64x64", f"DeepRV {teacher} {side}", metric
            )
            means.append(mean)
            sds.append(sd)
            values = subset[
                (subset["teacher"] == teacher)
                & (subset["inducing_side"] == side)
            ][metric]
            if len(values) != 3:
                raise RuntimeError(f"Expected three seeds for {teacher}{side}: {metric}")
            ax.scatter(
                np.full(3, side),
                values,
                s=11,
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
            markersize=3.4,
            linewidth=1.1,
            capsize=2,
            label=teacher,
            zorder=2,
        )
    if coverage:
        full, _ = aggregate_value(aggregates, "64x64", "Full GP", metric)
        exact, _ = aggregate_value(aggregates, "64x64", "Exact DeepRV", metric)
        ax.axhline(0.90, color="#9A9A9A", linestyle=":", linewidth=0.8)
        ax.axhline(full, color=COLORS["Full GP"], linestyle="--", linewidth=0.8)
        ax.axhline(exact, color=COLORS["Exact"], linestyle="-.", linewidth=0.8)
        ax.set_ylim(0.91, 0.975)
    else:
        exact, _ = aggregate_value(aggregates, "64x64", "Exact DeepRV", metric)
        ax.axhline(exact, color=COLORS["Exact"], linestyle="--", linewidth=0.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(sides)
    ax.set_xlabel("Inducing side")
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out", length=2.4, width=0.7)
