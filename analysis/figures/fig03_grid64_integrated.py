#!/usr/bin/env python3
"""Figure 3: all-seed quantitative evidence at 64x64 (no duplicated maps)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from figure_common import add_panel_label, apply_style, draw_metric_panel, read_results, save_figure

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
OUTPUT_FORMATS = (".svg", ".pdf", ".png")
RASTER_DPI = 600


def main() -> None:
    apply_style(8.0)
    long, aggregates = read_results()
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.65))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.77, wspace=0.49)
    draw_metric_panel(axes[0], long, aggregates, "predictive_mse_vs_full_gp", "Predictive MSE vs Full GP", log_scale=True)
    draw_metric_panel(axes[1], long, aggregates, "lengthscale_w1_vs_full_gp", "Lengthscale $W_1$", log_scale=True)
    draw_metric_panel(axes[2], long, aggregates, "coverage_90_unobserved", "Unobserved 90% coverage", coverage=True)
    for axis, label in zip(axes, "abc"):
        add_panel_label(axis, label, x=-0.16, y=1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.52, 0.985))
    save_figure(fig, "fig03_grid64_integrated")


if __name__ == "__main__":
    main()
