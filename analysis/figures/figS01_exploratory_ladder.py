#!/usr/bin/env python3
"""Supplementary Figure S1: 8x8 and 16x16 exploratory evidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_common import SOURCE_DATA, add_panel_label, apply_style, save_figure


def value(data, resolution, variant, construction, side, metric, ell=30):
    rows = data[
        (data["target_resolution"] == resolution)
        & data["experiment_variant"].str.contains(variant)
        & (data["teacher"] == construction)
        & (data["inducing_grid_size"].astype(str) == str(side))
        & (data["metric_name"] == metric)
        & (data["true_lengthscale"] == ell)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            (resolution, variant, construction, side, metric, ell, len(rows))
        )
    return float(rows.iloc[0]["metric_value"]), str(
        rows.iloc[0]["diagnostic_status"]
    )


def main() -> None:
    apply_style(8.0)
    source = SOURCE_DATA / "8x8_16x16_exploratory_metrics.csv"
    data = pd.read_csv(source, keep_default_na=False)
    data["true_lengthscale"] = pd.to_numeric(data["true_lengthscale"])

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.55), constrained_layout=True)
    labels = [
        "8: Exact",
        "8: Lowres4",
        "8: Local8",
        "16: Exact",
        "16: Lowres8",
        "16: Local8",
        "16: Local16",
    ]
    specs = [
        ("8x8", "exact_lowres_local", "Exact", "NOT_APPLICABLE"),
        ("8x8", "exact_lowres_local", "Lowres", "4"),
        ("8x8", "exact_lowres_local", "Local", "8"),
        ("16x16", "exact_lowres_local", "Exact", "NOT_APPLICABLE"),
        ("16x16", "exact_lowres_local", "Lowres", "8"),
        ("16x16", "exact_lowres_local", "Local", "8"),
        ("16x16", "exact_lowres_local", "Local", "16"),
    ]
    mse = [value(data, *spec, "posterior_mean_mse_vs_full_gp")[0] for spec in specs]
    axes[0].barh(
        np.arange(len(labels))[::-1],
        mse,
        color=["#272727", "#3775BA", "#B64342", "#272727", "#3775BA", "#B64342", "#E9A6A1"],
    )
    axes[0].set_yticks(np.arange(len(labels))[::-1])
    axes[0].set_yticklabels(labels)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Predictive MSE vs Full GP")

    sides = [4, 6, 8, 12]
    colours = {10: "#B64342", 30: "#3775BA", 50: "#188977"}
    for ell in (10, 30, 50):
        vals, statuses = [], []
        for side in sides:
            val, status = value(
                data,
                "16x16",
                f"frontier_gt_ls_{ell}",
                "Lowres",
                str(side),
                "posterior_mean_log1p_rmse_vs_full_gp",
                ell,
            )
            vals.append(val)
            statuses.append(status)
        axes[1].plot(
            sides,
            vals,
            color=colours[ell],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"ell={ell}",
        )
        for side, val, status in zip(sides, vals, statuses):
            if status.startswith("UNVERIFIED"):
                axes[1].scatter(
                    [side],
                    [val],
                    marker="x",
                    s=28,
                    color="#B64342",
                    linewidths=1.1,
                    zorder=4,
                )
    axes[1].set_xlabel("Low-resolution side")
    axes[1].set_ylabel("log1p predictive RMSE")
    axes[1].set_xticks(sides)
    axes[1].legend()

    saving = [
        100
        * value(
            data,
            "16x16",
            "frontier_gt_ls_30",
            "Lowres",
            str(side),
            "measured_pretraining_cost_saving_vs_exact",
            30,
        )[0]
        for side in sides
    ]
    axes[2].plot(
        sides, saving, color="#3775BA", marker="o", markersize=3, linewidth=1.2
    )
    axes[2].set_xlabel("Low-resolution side")
    axes[2].set_ylabel("Measured pretraining saving (%)")
    axes[2].set_xticks(sides)
    axes[2].set_ylim(bottom=0)

    for axis, panel in zip(axes, ("a", "b", "c")):
        add_panel_label(axis, panel)
        axis.tick_params(direction="out", length=2.5, width=0.8)
    save_figure(fig, "figS01_exploratory_ladder")


if __name__ == "__main__":
    main()
