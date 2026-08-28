#!/usr/bin/env python3
"""Figure 5: measured 64x64 runtime components and decoder amortisation."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

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
    "fig05_runtime_amortisation.svg",
    "fig05_runtime_amortisation.pdf",
    "fig05_runtime_amortisation.png",
)
RASTER_DPI = 600

from plot_common import (
    COLORS,
    add_panel_label,
    aggregate_value,
    apply_style,
    read_results,
    save_figure,
)


MODELS = [
    "Full GP",
    "Exact DeepRV",
    "DeepRV Cubic 32",
    "Direct Cubic 32",
    "DeepRV FITC 32",
]
MODEL_COLOURS = ["#4D4D4D", "#272727", COLORS["Cubic"], COLORS["Cubic"], COLORS["FITC"]]
MODEL_STYLES = ["-", "-", "-", "--", "-"]


def main() -> None:
    apply_style()
    _, aggregates = read_results()
    grid = "64x64"
    components = ["field_generation_seconds", "training_seconds", "inference_seconds", "total_seconds"]
    component_labels = ["Teacher fields", "Optimisation", "Inference", "Saved total"]
    component_markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.75), constrained_layout=True)
    y = np.arange(len(MODELS))[::-1]
    for metric, label, marker in zip(components, component_labels, component_markers):
        values = []
        for model in MODELS:
            try:
                mean, _ = aggregate_value(aggregates, grid, model, metric)
            except RuntimeError:
                mean = np.nan
            values.append(mean)
        axes[0].scatter(values, y, marker=marker, s=22, label=label)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(MODELS)
    axes[0].set_xlabel("Measured seconds (log scale)")
    axes[0].tick_params(direction="out", length=2.5, width=0.8)

    k_values = np.array([1, 2, 5, 10, 20, 50], dtype=float)
    for model, colour, line_style in zip(MODELS, MODEL_COLOURS, MODEL_STYLES):
        total, _ = aggregate_value(aggregates, grid, model, "total_seconds")
        if model == "Full GP" or model.startswith("Direct "):
            recurring, _ = aggregate_value(
                aggregates, grid, model, "inference_seconds"
            )
            one_time = 0.0
        else:
            recurring, _ = aggregate_value(aggregates, grid, model, "inference_seconds")
            one_time = total - recurring
        cumulative = one_time + k_values * recurring
        axes[1].plot(k_values, cumulative, color=colour, linestyle=line_style, marker="o", markersize=3, linewidth=1.2, label=model)
        axes[2].plot(k_values, cumulative / k_values, color=colour, linestyle=line_style, marker="o", markersize=3, linewidth=1.2, label=model)
    for axis in axes[1:]:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(k_values)
        axis.set_xticklabels([str(int(x)) for x in k_values])
        axis.set_xlabel("Datasets using one decoder, K")
        axis.tick_params(direction="out", length=2.5, width=0.8)
    axes[1].set_ylabel("Cumulative seconds")
    axes[2].set_ylabel("Seconds per dataset")
    for axis, panel in zip(axes, ("a", "b", "c")):
        add_panel_label(axis, panel)
    model_handles = [Line2D([0], [0], color=c, linestyle=s, lw=1.5, label=m) for m, c, s in zip(MODELS, MODEL_COLOURS, MODEL_STYLES)]
    fig.legend(model_handles, MODELS, loc="upper center", ncol=5, bbox_to_anchor=(0.60, 1.075))
    save_figure(fig, "fig05_runtime_amortisation", dpi=RASTER_DPI)


if __name__ == "__main__":
    main()
