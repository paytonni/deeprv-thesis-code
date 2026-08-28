#!/usr/bin/env python3
"""Figure 6: prespecified Seed-0 64x64 posterior predictive map audit."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from figure_common import (
    DEEPRV_SEED0,
    DIRECT32_SEED0,
    FULL_GP_SEED0,
    add_panel_label,
    apply_style,
    load_and_verify_seed0_data,
    posterior_mean_zip,
    save_figure,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
OUTPUT_FORMATS = (".svg", ".pdf", ".png")
RASTER_DPI = 600


DEEPRV_MEMBERS = {
    "Exact DeepRV": "seed_0/inference/initial/deeprv_exact/posterior_predictive.npz",
    "DeepRV Bilinear32": "seed_0/inference/initial/deeprv_lowres_bilinear_32x32/posterior_predictive.npz",
    "DeepRV Cubic32": "seed_0/inference/initial/deeprv_lowres_cubic_32x32/posterior_predictive.npz",
    "DeepRV DTC32": "seed_0/inference/initial/deeprv_lowres_dtc_32x32/posterior_predictive.npz",
    "DeepRV FITC32": "seed_0/inference/initial/deeprv_lowres_fitc_32x32/posterior_predictive.npz",
}
FULL_MEMBER = "seed_0/inference/initial/full_gp/posterior_predictive.npz"
DIRECT_CUBIC_MEMBER = "seed_0/inference/initial/kissgp_ski_cubic/posterior_predictive.npz"


def clean_map_axis(axis, title: str, label: str) -> None:
    axis.set_title(title, pad=2.4, fontsize=6.5)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    add_panel_label(axis, label, x=-0.06, y=1.01)
    axis.set_box_aspect(1)


def main() -> None:
    apply_style(6.5)
    data = load_and_verify_seed0_data()
    latent, mask = data["latent"], data["mask"]
    full = posterior_mean_zip(FULL_GP_SEED0, FULL_MEMBER)
    deep = {
        title: posterior_mean_zip(DEEPRV_SEED0, member)
        for title, member in DEEPRV_MEMBERS.items()
    }
    direct_cubic = posterior_mean_zip(DIRECT32_SEED0, DIRECT_CUBIC_MEMBER)

    prediction_titles = (
        "Full GP reference",
        "Exact DeepRV",
        "DeepRV Bilinear32",
        "DeepRV Cubic32",
        "DeepRV DTC32",
        "DeepRV FITC32",
        "Direct Cubic32",
    )
    prediction_images = (
        full,
        deep["Exact DeepRV"],
        deep["DeepRV Bilinear32"],
        deep["DeepRV Cubic32"],
        deep["DeepRV DTC32"],
        deep["DeepRV FITC32"],
        direct_cubic,
    )
    if any(np.any(image < 0) for image in prediction_images):
        raise ValueError("Posterior mean counts must be nonnegative for log1p")
    # log1p uses the declared pseudocount of one and only changes display.
    log_images = tuple(np.log1p(image) for image in prediction_images)
    count_vmin = float(min(image.min() for image in log_images))
    count_vmax = float(max(image.max() for image in log_images))
    differences = (
        np.abs(deep["Exact DeepRV"] - full),
        np.abs(deep["DeepRV Cubic32"] - full),
        np.abs(deep["DeepRV FITC32"] - full),
    )
    diff_vmax = float(max(image.max() for image in differences))
    latent_limit = float(np.max(np.abs(latent)))

    fig = plt.figure(figsize=(7.20, 5.62))
    grid = fig.add_gridspec(
        3,
        5,
        left=0.025,
        right=0.955,
        bottom=0.035,
        top=0.975,
        width_ratios=(1, 1, 1, 1, 0.055),
        wspace=0.13,
        hspace=0.22,
    )
    axes = [[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(3)]
    caxes = [fig.add_subplot(grid[row, 4]) for row in range(3)]

    latent_artist = axes[0][0].imshow(
        latent,
        cmap="RdBu_r",
        vmin=-latent_limit,
        vmax=latent_limit,
        origin="lower",
        interpolation="nearest",
    )
    clean_map_axis(axes[0][0], "True latent field", "a")
    axes[0][1].imshow(
        mask.astype(int),
        cmap=ListedColormap(["#D9DDE0", "#3977A8"]),
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
    )
    clean_map_axis(axes[0][1], "Uniform observation mask", "b")
    axes[0][1].text(0.5, -0.07, "grey: unobserved  ·  blue: observed", transform=axes[0][1].transAxes, ha="center", va="top", fontsize=5.2)

    panel_positions = (
        (axes[0][2], log_images[0], prediction_titles[0], "c"),
        (axes[0][3], log_images[1], prediction_titles[1], "d"),
        (axes[1][0], log_images[2], prediction_titles[2], "e"),
        (axes[1][1], log_images[3], prediction_titles[3], "f"),
        (axes[1][2], log_images[4], prediction_titles[4], "g"),
        (axes[1][3], log_images[5], prediction_titles[5], "h"),
        (axes[2][0], log_images[6], prediction_titles[6], "i"),
    )
    count_artist = None
    for axis, image, title, label in panel_positions:
        count_artist = axis.imshow(
            image,
            cmap="viridis",
            vmin=count_vmin,
            vmax=count_vmax,
            origin="lower",
            interpolation="nearest",
        )
        clean_map_axis(axis, title, label)

    difference_titles = (
        "$|$Exact DeepRV $-$ Full GP$|$",
        "$|$DeepRV Cubic32 $-$ Full GP$|$",
        "$|$DeepRV FITC32 $-$ Full GP$|$",
    )
    difference_artist = None
    for axis, image, title, label in zip(axes[2][1:], differences, difference_titles, "jkl"):
        difference_artist = axis.imshow(
            image,
            cmap="magma",
            vmin=0.0,
            vmax=diff_vmax,
            origin="lower",
            interpolation="nearest",
        )
        clean_map_axis(axis, title, label)

    latent_cbar = fig.colorbar(latent_artist, cax=caxes[0])
    latent_cbar.set_label("Latent field value", rotation=270, rotation_mode="anchor", labelpad=10)
    count_cbar = fig.colorbar(count_artist, cax=caxes[1])
    count_cbar.set_label("$\\log(1+$ posterior mean count$)$", rotation=270, rotation_mode="anchor", labelpad=11)
    diff_cbar = fig.colorbar(difference_artist, cax=caxes[2])
    diff_cbar.set_label("Absolute mean-count difference", rotation=270, rotation_mode="anchor", labelpad=11)
    for cax in caxes:
        cax.tick_params(length=2, pad=1)

    save_figure(fig, "fig06_grid64_posterior_maps")


if __name__ == "__main__":
    main()
