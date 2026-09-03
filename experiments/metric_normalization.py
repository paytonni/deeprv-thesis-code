"""Deterministic mapping from experiment metrics to the analysis schema."""

from __future__ import annotations

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
    if raw == "deeprv_exact":
        return {"model": "Exact DeepRV", "family": "DeepRV", "teacher": "Exact"}
    if raw == "Exact128":
        return {"family": "DeepRV", "teacher": "Exact", "inducing_side": 128}
    target_match = re.fullmatch(r"(Bilinear|Cubic|DTC|FITC)(\d+)", raw)
    if target_match:
        return {
            "family": "DeepRV",
            "teacher": target_match.group(1),
            "inducing_side": int(target_match.group(2)),
        }
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
        "predictive_mse_vs_full_gp": "posterior_mean_mse_vs_full_gp",
        "lengthscale_w1_vs_full_gp": "ls_wasserstein_vs_full_gp",
        "lengthscale_wasserstein_vs_full_gp": "ls_wasserstein_vs_full_gp",
        "log1p_mse_counts": "log1p_mse_all",
        "predictive_count_mse": "mse_all",
        "posterior_lengthscale_mean": "posterior_mean_ls",
        "posterior_lengthscale_sd": "posterior_sd_ls",
        "field_generation_seconds": "pretraining_gp_field_generation_time",
        "training_seconds": "pretraining_neural_optimization_time",
    }
    for destination, source in aliases.items():
        if destination not in row and row.get(source) is not None:
            row[destination] = row[source]

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
