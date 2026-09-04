"""Deterministic mapping from experiment metrics to the analysis schema."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


_TEACHER_NAMES = {
    "bilinear": "Bilinear",
    "cubic": "Cubic",
    "dtc": "DTC",
    "fitc": "FITC",
}
_DIRECT_TEACHERS = {
    "bilinear": "Bilinear",
    "kissgp_ski_bilinear": "Bilinear",
    "cubic": "Cubic",
    "kissgp_ski_cubic": "Cubic",
    "dtc": "DTC",
    "dtc_sor": "DTC",
    "fitc": "FITC",
}


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _model_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(_first(row, "model_name", "model") or "")
    side = _first(row, "inducing_grid_size", "inducing_side")
    if raw == "full_gp":
        return {"model": "Full GP", "family": "Reference", "teacher": "Full GP"}
    if raw == "deeprv_exact" or re.fullmatch(r"deeprv_exact_\d+x\d+", raw):
        return {"model": "Exact DeepRV", "family": "DeepRV", "teacher": "Exact"}
    if raw == "Exact128":
        return {
            "model": raw,
            "family": "DeepRV",
            "teacher": "Exact",
            "inducing_side": 128,
        }
    target_match = re.fullmatch(r"(Bilinear|Cubic|DTC|FITC)(\d+)", raw)
    if target_match:
        return {
            "model": raw,
            "family": "DeepRV",
            "teacher": target_match.group(1),
            "inducing_side": int(target_match.group(2)),
        }
    exploratory_match = re.fullmatch(r"deeprv_(lowres|local)(?:_(\d+)x\d+)?", raw)
    if exploratory_match:
        teacher = exploratory_match.group(1).title()
        inducing_side = exploratory_match.group(2)
        if inducing_side is None:
            locations = _first(row, "num_pretrain_locations")
            if locations is not None:
                candidate = int(round(math.sqrt(int(locations))))
                if candidate * candidate == int(locations):
                    inducing_side = str(candidate)
        fields: dict[str, Any] = {"family": "DeepRV", "teacher": teacher}
        if inducing_side is not None:
            fields["inducing_side"] = int(inducing_side)
            fields["model"] = f"DeepRV {teacher} {inducing_side}"
        return fields
    match = re.fullmatch(
        r"deeprv_lowres_(bilinear|cubic|dtc|fitc)_(\d+)x\d+", raw
    )
    if match:
        teacher = _TEACHER_NAMES[match.group(1)]
        inducing_side = int(match.group(2))
        return {
            "model": f"DeepRV {teacher} {inducing_side}",
            "family": "DeepRV",
            "teacher": teacher,
            "inducing_side": inducing_side,
        }
    if raw in _DIRECT_TEACHERS and side is not None:
        teacher = _DIRECT_TEACHERS[raw]
        inducing_side = int(side)
        return {
            "model": f"Direct {teacher} {inducing_side}",
            "family": "Direct",
            "teacher": teacher,
            "inducing_side": inducing_side,
        }
    return {}


def normalize_analysis_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return metrics with non-destructive aliases consumed by ``analysis/``."""

    row = dict(metrics)
    grid_size = _first(row, "grid_size", "target_grid_size")
    if grid_size is not None:
        row.setdefault("grid", f"{int(grid_size)}x{int(grid_size)}")
    seed = _first(row, "seed", "data_seed")
    if seed is not None:
        row.setdefault("seed", int(seed))
    for key, value in _model_fields(row).items():
        row.setdefault(key, value)

    aliases = {
        "predictive_mse_vs_full_gp": ("posterior_mean_mse_vs_full_gp",),
        "lengthscale_w1_vs_full_gp": ("ls_wasserstein_vs_full_gp",),
        "lengthscale_wasserstein_vs_full_gp": ("ls_wasserstein_vs_full_gp",),
        "log1p_mse_counts": ("log1p_mse_all",),
        "predictive_count_mse": ("mse_all",),
        "ell_abs_error": ("ell_mean_distance_to_true_30",),
        "posterior_lengthscale_mean": ("posterior_mean_ls", "ell_mean"),
        "posterior_lengthscale_median": ("posterior_median_ls", "ell_median"),
        "posterior_lengthscale_sd": ("posterior_sd_ls", "ell_sd"),
        "posterior_lengthscale_q05": ("posterior_q05_ls", "ell_q05"),
        "posterior_lengthscale_q95": ("posterior_q95_ls", "ell_q95"),
        "posterior_beta_mean": ("posterior_mean_beta", "beta_mean"),
        "posterior_beta_median": ("posterior_median_beta", "beta_median"),
        "posterior_beta_sd": ("posterior_sd_beta", "beta_sd"),
        "posterior_beta_q05": ("posterior_q05_beta", "beta_q05"),
        "posterior_beta_q95": ("posterior_q95_beta", "beta_q95"),
        "field_generation_seconds": ("pretraining_gp_field_generation_time",),
        "training_seconds": ("pretraining_neural_optimization_time",),
    }
    for destination, sources in aliases.items():
        value = _first(row, *sources)
        if destination not in row and value is not None:
            row[destination] = value

    if (
        "predictive_count_mse" not in row
        and row.get("posterior_predictive_count_mse_all") is not None
    ):
        row["predictive_count_mse"] = row["posterior_predictive_count_mse_all"]
    if "true_lengthscale" not in row and row.get("gt_ls") is not None:
        row["true_lengthscale"] = row["gt_ls"]
    if "divergences" not in row and row.get("num_divergences") is not None:
        row["divergences"] = row["num_divergences"]
    if "max_depth_hits" not in row and row.get("max_tree_depth_hits") is not None:
        row["max_depth_hits"] = row["max_tree_depth_hits"]

    if "inference_seconds" not in row:
        value = _first(
            row,
            "inference_total_time",
            "inference_total_seconds",
            "posterior_inference_time",
        )
        if value is not None:
            row["inference_seconds"] = value
    if "total_seconds" not in row:
        value = _first(
            row,
            "total_time_with_pretraining",
            "total_time_with_precompute",
            "inference_total_seconds",
            "inference_total_time",
        )
        if value is not None:
            row["total_seconds"] = value
    if "formal_pass" not in row:
        if row.get("diagnostics_passed") is not None:
            row["formal_pass"] = bool(row["diagnostics_passed"])
        elif row.get("status") is not None:
            row["formal_pass"] = row["status"] == "PASS"
    return row
