#!/usr/bin/env python3
"""Paired Target-128 comparison using the author-corrected Seed 0/1/2 labels."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figure_common import COLORS, SOURCE_DATA, add_panel_label, apply_style, save_figure


def draw_panel(ax, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    x = [0, 1]
    markers = {0: "o", 1: "s", 2: "^"}
    grey = {0: "#4A4A4A", 1: "#777777", 2: "#A0A0A0"}
    for seed in (0, 1, 2):
        rows = data[data["seed"] == seed].set_index("model")
        values = [rows.loc["Exact128", metric], rows.loc["Cubic64", metric]]
        ax.plot(x, values, color=grey[seed], linewidth=0.9, zorder=1)
        ax.scatter(0, values[0], s=30, marker=markers[seed],
                   facecolor="white", edgecolor=COLORS["Exact"],
                   linewidth=1.1, zorder=2, label=f"Seed {seed}")
        ax.scatter(1, values[1], s=30, marker=markers[seed],
                   facecolor="white", edgecolor=COLORS["Cubic"],
                   linewidth=1.1, zorder=2)
    ax.set_xticks(x, ["Exact128", "Cubic64"])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.25, 1.25)
    ax.tick_params(direction="out", length=2.4, width=0.7)


def main() -> None:
    apply_style(8.0)
    data = pd.read_csv(SOURCE_DATA / "target128_exact_cubic_author_corrected.csv")
    expected = {(model, seed) for model in ("Exact128", "Cubic64") for seed in (0, 1, 2)}
    observed = set(zip(data["model"], data["seed"]))
    if observed != expected or data[["predictive_count_mse", "ell_abs_error"]].isna().any().any():
        raise ValueError("Target-128 paired figure input is incomplete")

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.75))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.80, wspace=0.38)
    draw_panel(axes[0], data, "predictive_count_mse", "Predictive-count MSE")
    draw_panel(axes[1], data, "ell_abs_error", "Absolute lengthscale error")
    for axis, label in zip(axes, "ab"):
        add_panel_label(axis, label, x=-0.13, y=1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 0.985))
    save_figure(fig, "fig07_target128_paired")


if __name__ == "__main__":
    main()
