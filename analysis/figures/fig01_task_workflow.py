#!/usr/bin/env python3
"""Figure 1: Seed-0 task illustration and pretraining/inference boundary."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from figure_common import add_panel_label, apply_style, load_and_verify_seed0_data, save_figure

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
OUTPUT_FORMATS = (".svg", ".pdf", ".png")
RASTER_DPI = 600


EXACT_FILL = "#DCEAF5"
APPROX_FILL = "#F7E3C7"
NEURAL_FILL = "#E8DDF2"
COMMON_FILL = "#ECEFF1"
INFERENCE_FILL = "#E2F0E7"
TEXT = "#28323C"
LINE = "#56616B"


def rounded_box(ax, x, y, w, h, text, face, edge=LINE, fontsize=6.2, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.75,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.05,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, colour=LINE, connectionstyle="arc3,rad=0"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.72,
            color=colour,
            connectionstyle=connectionstyle,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=4,
        )
    )


def region(ax, x, y, w, h, title, face, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.text(
        x + 0.012,
        y + h - 0.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=edge,
        fontsize=6.7,
        fontweight="bold",
        zorder=5,
    )


def main() -> None:
    apply_style(6.6)
    data = load_and_verify_seed0_data()
    latent, counts, mask = data["latent"], data["counts"], data["mask"]
    if np.any(counts < 0):
        raise ValueError("Counts must be nonnegative before the log1p pseudocount transform")
    log_counts = np.log1p(counts)
    masked_log_counts = np.where(mask, log_counts, np.nan)

    fig = plt.figure(figsize=(7.20, 6.35))
    grid = fig.add_gridspec(
        2,
        4,
        left=0.035,
        right=0.985,
        bottom=0.025,
        top=0.985,
        hspace=0.22,
        wspace=0.18,
        height_ratios=(1.00, 2.28),
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]

    latent_limit = float(np.max(np.abs(latent)))
    latent_artist = axes[0].imshow(
        latent,
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-latent_limit,
        vmax=latent_limit,
    )
    count_vmin, count_vmax = float(log_counts.min()), float(log_counts.max())
    count_cmap = plt.get_cmap("viridis").copy()
    count_cmap.set_bad("#D9DDE0")
    count_artist = axes[1].imshow(
        log_counts,
        origin="lower",
        interpolation="nearest",
        cmap=count_cmap,
        vmin=count_vmin,
        vmax=count_vmax,
    )
    mask_artist = axes[2].imshow(
        mask.astype(int),
        origin="lower",
        interpolation="nearest",
        cmap=ListedColormap(["#D9DDE0", "#3977A8"]),
        vmin=0,
        vmax=1,
    )
    axes[3].imshow(
        masked_log_counts,
        origin="lower",
        interpolation="nearest",
        cmap=count_cmap,
        vmin=count_vmin,
        vmax=count_vmax,
    )
    titles = (
        "True latent field $f$",
        "Complete count field, $\\log(1+y)$",
        "Uniform observation mask",
        "Observed count field, $\\log(1+y)$",
    )
    for axis, title, label in zip(axes, titles, "abcd"):
        axis.set_title(title, pad=2.5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        add_panel_label(axis, label, x=-0.06, y=1.01)

    latent_cbar = fig.colorbar(latent_artist, ax=axes[0], orientation="horizontal", fraction=0.055, pad=0.04)
    latent_cbar.set_label("Latent field value", labelpad=1.2)
    latent_cbar.ax.tick_params(length=2, pad=1)
    # A single count colour bar documents the range shared by panels (b) and (d).
    count_cbar = fig.colorbar(count_artist, ax=axes[1], orientation="horizontal", fraction=0.055, pad=0.04)
    count_cbar.set_label("Displayed $\\log(1+\\mathrm{count})$", labelpad=1.2)
    count_cbar.ax.tick_params(length=2, pad=1)
    axes[2].text(
        0.5,
        -0.13,
        "grey: unobserved    blue: observed (50%)",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=5.7,
        color=TEXT,
    )

    workflow = fig.add_subplot(grid[1, :])
    workflow.set_axis_off()
    add_panel_label(workflow, "e", x=-0.018, y=1.00)

    region(
        workflow,
        0.005,
        0.625,
        0.990,
        0.360,
        "ONE-TIME SUPERVISED PRETRAINING",
        "#FBFCFD",
        "#7A8792",
    )
    rounded_box(workflow, 0.025, 0.735, 0.135, 0.095, "Hyperparameters $\\ell$\n+ Gaussian input $\\xi_b$", COMMON_FILL)
    rounded_box(workflow, 0.200, 0.790, 0.135, 0.065, "Exact GP fields", EXACT_FILL, edge="#527A9D")
    rounded_box(
        workflow,
        0.200,
        0.675,
        0.135,
        0.095,
        "Approximate GP fields\nBilinear · Cubic\nDTC · FITC",
        APPROX_FILL,
        edge="#B07A34",
        fontsize=5.6,
    )
    workflow.text(0.2675, 0.875, "Pretraining field source", transform=workflow.transAxes, ha="center", va="center", fontsize=6.2, fontweight="bold", color=TEXT)
    rounded_box(workflow, 0.385, 0.735, 0.125, 0.095, "Supervised\nfield pairs", COMMON_FILL)
    rounded_box(workflow, 0.550, 0.735, 0.115, 0.095, "Reconstruction\nloss", COMMON_FILL)
    rounded_box(workflow, 0.705, 0.735, 0.125, 0.095, "DeepRV decoder\ntraining", NEURAL_FILL, edge="#755C8E")
    rounded_box(workflow, 0.865, 0.735, 0.110, 0.095, "Frozen DeepRV\ndecoder", NEURAL_FILL, edge="#755C8E", weight="bold")
    arrow(workflow, (0.160, 0.782), (0.200, 0.822))
    arrow(workflow, (0.160, 0.782), (0.200, 0.722))
    arrow(workflow, (0.335, 0.822), (0.385, 0.795))
    arrow(workflow, (0.335, 0.722), (0.385, 0.770))
    arrow(workflow, (0.510, 0.782), (0.550, 0.782))
    arrow(workflow, (0.665, 0.782), (0.705, 0.782))
    arrow(workflow, (0.830, 0.782), (0.865, 0.782), colour="#755C8E")
    workflow.text(0.848, 0.842, "freeze", transform=workflow.transAxes, ha="center", va="bottom", fontsize=5.6, color="#755C8E")

    region(
        workflow,
        0.005,
        0.012,
        0.990,
        0.565,
        "PER-DATASET POSTERIOR INFERENCE",
        "#FAFCFA",
        "#4D7C60",
    )
    workflow.text(
        0.320,
        0.515,
        "Matched likelihood, inferential priors and NUTS protocol",
        transform=workflow.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
        color="#4D7C60",
        fontweight="bold",
    )
    rounded_box(
        workflow,
        0.660,
        0.468,
        0.155,
        0.070,
        "Observed counts\n+ uniform mask",
        COMMON_FILL,
        fontsize=5.8,
        weight="bold",
    )

    row_y = (0.350, 0.205, 0.060)
    row_names = ("DeepRV", "Full GP reference", "Matched Direct GP")
    row_fills = (NEURAL_FILL, EXACT_FILL, APPROX_FILL)
    row_edges = ("#755C8E", "#527A9D", "#B07A34")
    input_text = ("Latents + parameters\n$z,\\ell,\\beta$", "Covariance parameter\n$\\ell$", "Covariance parameter\n$\\ell$")
    construction_text = ("Frozen DeepRV decoder\n(no GP construction downstream)", "Exact GP covariance\n+ field construction", "Matched approximate covariance\n(no decoder)")
    for y, name, face, edge, first, construction in zip(row_y, row_names, row_fills, row_edges, input_text, construction_text):
        rounded_box(workflow, 0.025, y, 0.105, 0.085, name, face, edge=edge, fontsize=5.8, weight="bold")
        rounded_box(workflow, 0.155, y, 0.120, 0.085, first, COMMON_FILL, fontsize=5.8)
        rounded_box(workflow, 0.310, y, 0.185, 0.085, construction, face, edge=edge, fontsize=5.7)
        rounded_box(workflow, 0.530, y, 0.095, 0.085, "Latent field\n$f$", COMMON_FILL, fontsize=5.8)
        rounded_box(workflow, 0.675, y, 0.135, 0.085, "Common Poisson\nlikelihood", COMMON_FILL, fontsize=5.8)
        rounded_box(workflow, 0.850, y, 0.125, 0.085, "NUTS posterior\ninference", INFERENCE_FILL, edge="#4D7C60", fontsize=5.8)
        arrow(workflow, (0.130, y + 0.0425), (0.155, y + 0.0425), colour=edge)
        arrow(workflow, (0.275, y + 0.0425), (0.310, y + 0.0425), colour=edge)
        arrow(workflow, (0.495, y + 0.0425), (0.530, y + 0.0425), colour=edge)
        arrow(workflow, (0.625, y + 0.0425), (0.675, y + 0.0425))
        arrow(workflow, (0.810, y + 0.0425), (0.850, y + 0.0425), colour="#4D7C60")

    # A neutral data bus feeds each likelihood and never enters pretraining.
    workflow.plot([0.655, 0.655], [0.102, 0.503], transform=workflow.transAxes, color="#7B858D", linewidth=0.65, zorder=1)
    workflow.plot([0.655, 0.660], [0.503, 0.503], transform=workflow.transAxes, color="#7B858D", linewidth=0.65, zorder=1)
    for y in row_y:
        arrow(workflow, (0.655, y + 0.0425), (0.675, y + 0.0425), colour="#7B858D")

    save_figure(fig, "fig01_task_workflow")


if __name__ == "__main__":
    main()
