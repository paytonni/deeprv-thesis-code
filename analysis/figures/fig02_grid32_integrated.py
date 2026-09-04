#!/usr/bin/env python3
"""Figure 2: integrated all-seed metrics and Seed-0 maps at 32x32."""

from __future__ import annotations

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

EXPECTED_OUTPUTS = (
    "fig02_grid32_integrated.svg",
    "fig02_grid32_integrated.pdf",
    "fig02_grid32_integrated.png",
)
RASTER_DPI = 600

from plot_common import (
    RESULTS_ROOT,
    add_panel_label,
    apply_style,
    draw_difference_and_mask_row,
    draw_map_row,
    draw_metric_panel,
    posterior_mean_local,
    read_pickle,
    read_results,
    save_figure,
)


def main() -> None:
    apply_style()
    long, aggregates = read_results()
    fig = plt.figure(figsize=(7.20, 6.85))
    outer = fig.add_gridspec(
        2,
        1,
        left=0.08,
        right=0.90,
        bottom=0.07,
        top=0.88,
        hspace=0.40,
        height_ratios=[1.15, 1.70],
    )
    top = outer[0].subgridspec(1, 3, wspace=0.62)
    bottom = outer[1].subgridspec(2, 4, wspace=0.08, hspace=0.30)
    metric_axes = [fig.add_subplot(top[0, i]) for i in range(3)]
    draw_metric_panel(metric_axes[0], long, aggregates, "32x32", "predictive_mse_vs_full_gp", "Predictive MSE vs Full GP", (4, 8, 16), log_scale=True, show_seed_points=False)
    draw_metric_panel(metric_axes[1], long, aggregates, "32x32", "lengthscale_w1_vs_full_gp", "Lengthscale W1", (4, 8, 16), log_scale=True, show_seed_points=False)
    draw_metric_panel(metric_axes[2], long, aggregates, "32x32", "coverage_90_unobserved", "Unobserved coverage", (4, 8, 16), coverage=True, show_seed_points=False)
    metric_axes[2].set_ylim(0.72, 0.99)
    for axis, label in zip(metric_axes, ("a", "b", "c")):
        add_panel_label(axis, label)
    handles, labels = metric_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.49, 0.985))

    base = (
        RESULTS_ROOT
        / "outputs"
        / "grid32"
        / "grid32_thesis_comparison"
        / "seed_0"
        / "inference"
        / "initial"
    )
    paths = [
        base / "full_gp" / "posterior_predictive.npz",
        base / "deeprv_exact" / "posterior_predictive.npz",
        base / "deeprv_lowres_cubic_16x16" / "posterior_predictive.npz",
        base / "deeprv_lowres_fitc_16x16" / "posterior_predictive.npz",
    ]
    images = [posterior_mean_local(path, 32) for path in paths]
    map_axes = [fig.add_subplot(bottom[0, i]) for i in range(4)]
    artist = draw_map_row(
        map_axes,
        images,
        ["Full GP", "Exact DeepRV", "Cubic16", "FITC16"],
        ["d", "e", "f", "g"],
    )
    difference_axes = [fig.add_subplot(bottom[1, i]) for i in range(4)]
    observed = read_pickle(
        RESULTS_ROOT
        / "outputs"
        / "grid32"
        / "grid32_thesis_comparison"
        / "seed_0"
        / "observed_data.pkl"
    )
    difference_artist = draw_difference_and_mask_row(
        difference_axes,
        images[0],
        images[1:],
        ["|Exact - Full|", "|Cubic16 - Full|", "|FITC16 - Full|"],
        ["h", "i", "j", "k"],
        observed["obs_mask"].reshape(32, 32),
    )
    colourbar_axis = fig.add_axes([0.925, 0.36, 0.016, 0.22])
    colourbar = fig.colorbar(artist, cax=colourbar_axis)
    colourbar.ax.set_title("log1p\nmean", pad=3)
    difference_colourbar_axis = fig.add_axes([0.925, 0.08, 0.016, 0.22])
    difference_colourbar = fig.colorbar(
        difference_artist, cax=difference_colourbar_axis
    )
    difference_colourbar.ax.set_title("|mean -\nFull|", pad=3)
    save_figure(fig, "fig02_grid32_integrated", dpi=RASTER_DPI)


if __name__ == "__main__":
    main()
