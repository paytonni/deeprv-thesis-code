#!/usr/bin/env python3
"""Build deterministic analysis tables from completed experiment metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from metric_normalization import normalize_analysis_metrics  # noqa: E402


PUBLIC_SEEDS = (0, 1, 2)
ANALYSIS_METRICS = (
    "predictive_mse_vs_full_gp",
    "lengthscale_w1_vs_full_gp",
    "coverage_90_all",
    "coverage_90_unobserved",
    "log1p_mse_counts",
    "posterior_lengthscale_mean",
    "posterior_lengthscale_median",
    "posterior_lengthscale_sd",
    "posterior_lengthscale_q05",
    "posterior_lengthscale_q95",
    "posterior_beta_mean",
    "posterior_beta_median",
    "posterior_beta_sd",
    "posterior_beta_q05",
    "posterior_beta_q95",
    "field_generation_seconds",
    "training_seconds",
    "inference_seconds",
    "total_seconds",
    "divergences",
    "max_depth_hits",
    "predictive_count_mse",
    "ell_abs_error",
    "posterior_mean_log1p_rmse_vs_full_gp",
    "measured_pretraining_cost_saving_vs_exact",
    "relative_lengthscale_wasserstein_vs_full_gp",
)
CONTEXT_FILES = (
    "experiment_config.json",
    "seed_config.json",
    "scenario_config.json",
    "config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root containing completed experiment output directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "DEEPRV_ANALYSIS_DATA_ROOT", ROOT / "analysis" / "data"
            )
        ),
        help="Destination for results_long.csv and results_aggregate.csv.",
    )
    return parser.parse_args()


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _coerce(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return None
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def _context_for(path: Path, results_root: Path) -> dict[str, Any]:
    parents = [path.parent]
    parents.extend(parent for parent in path.parents if parent != path.parent)
    parents = [parent for parent in parents if parent == results_root or results_root in parent.parents]
    context: dict[str, Any] = {}
    for parent in reversed(parents):
        for name in CONTEXT_FILES:
            candidate = parent / name
            if candidate.is_file():
                context.update(_json_mapping(candidate))
    return context


def _grid_from_path(path: Path) -> int | None:
    text = path.as_posix().lower()
    match = re.search(r"(?<!\d)(8|16|32|64|128)x\1(?!\d)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"target[_-]?(8|16|32|64|128)(?!\d)", text)
    return int(match.group(1)) if match else None


def _seed_from_path(path: Path) -> int | None:
    matches = re.findall(r"seed[_-]?([012])(?!\d)", path.as_posix().lower())
    return int(matches[-1]) if matches else None


def _study(row: Mapping[str, Any], path: Path) -> str:
    grid = str(row["grid"])
    text = path.as_posix().lower()
    if grid == "128x128":
        return "target128"
    if "scenarios" in path.parts or "spacing" in text or "frontier" in text:
        return "spacing"
    if grid == "8x8":
        return "foundations"
    if grid == "16x16":
        return "support"
    return "systematic"


def _target128_context() -> dict[str, Any]:
    config = _json_mapping(EXPERIMENTS / "configs" / "target128.json")
    return {
        "target_grid_size": config["target_grid_size"],
        "true_lengthscale": config["true_lengthscale"],
    }


def _normalise(
    raw: Mapping[str, Any], path: Path, results_root: Path
) -> dict[str, Any]:
    row = normalize_analysis_metrics(raw)
    if row.get("grid") is None:
        grid_size = _grid_from_path(path)
        if grid_size is not None:
            row["grid"] = f"{grid_size}x{grid_size}"
    if row.get("seed") is None:
        seed = _seed_from_path(path)
        if seed is not None:
            row["seed"] = seed
    if row.get("grid") == "128x128":
        for key, value in _target128_context().items():
            row.setdefault(key, value)
        row = normalize_analysis_metrics(row)
    if row.get("true_lengthscale") is None and row.get("gt_ls") is not None:
        row["true_lengthscale"] = row["gt_ls"]
    if row.get("budget") == "extended" or bool(row.get("probe", False)):
        return {}
    if "runtime_estimates" in path.parts:
        return {}
    missing = [key for key in ("grid", "model", "family", "teacher", "seed") if row.get(key) is None]
    if missing:
        raise RuntimeError(f"Cannot normalize {path}; missing fields: {', '.join(missing)}")
    row["seed"] = int(row["seed"])
    if row["seed"] not in PUBLIC_SEEDS:
        raise RuntimeError(f"Unsupported seed {row['seed']} in {path}; expected only 0, 1, 2")
    row["study"] = _study(row, path)
    row["source_path"] = path.relative_to(results_root).as_posix()
    return row


def _completed_json_rows(results_root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(results_root.rglob("metrics.json")):
        if not (path.parent / "complete.json").is_file():
            continue
        yield path, {**_context_for(path, results_root), **_json_mapping(path)}


def _standalone_csv_rows(results_root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(results_root.rglob("metrics.csv")):
        if path.with_suffix(".json").is_file() or any(path.parent.rglob("metrics.json")):
            continue
        context = _context_for(path, results_root)
        if not context:
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as error:
            raise RuntimeError(f"Cannot read CSV input {path}: {error}") from error
        for csv_row in rows:
            yield path, {**context, **{key: _coerce(value) for key, value in csv_row.items()}}


def discover_rows(results_root: Path) -> list[dict[str, Any]]:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Experiment results root does not exist: {results_root}")
    candidates = [*_completed_json_rows(results_root), *_standalone_csv_rows(results_root)]
    if not candidates:
        raise RuntimeError(
            f"No completed metric inputs found below {results_root}. Expected leaf metrics.json files with complete.json markers or a standalone metrics.csv."
        )
    rows = [
        row
        for path, raw in candidates
        if (row := _normalise(raw, path, results_root))
    ]
    if not rows:
        raise RuntimeError(f"No analysis-eligible metric rows found below {results_root}")
    return rows


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_tables(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = pd.DataFrame(rows)
    identity = [
        "study",
        "grid",
        "model",
        "family",
        "teacher",
        "inducing_side",
        "true_lengthscale",
        "seed",
        "budget",
    ]
    present_identity = [column for column in identity if column in long.columns]
    duplicate_key = [column for column in present_identity if column not in {"budget"}]
    duplicates = long.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        details = long.loc[duplicates, duplicate_key + ["source_path"]]
        raise RuntimeError("Duplicate analysis rows detected:\n" + details.to_string(index=False))

    group_columns = [column for column in identity[:-2] if column in long.columns]
    incomplete = []
    for keys, group in long.groupby(group_columns, dropna=False, sort=True):
        seeds = tuple(sorted(int(seed) for seed in group["seed"].unique()))
        if seeds != PUBLIC_SEEDS:
            incomplete.append((keys, seeds))
    if incomplete:
        lines = [f"{keys}: seeds={seeds}" for keys, seeds in incomplete]
        raise RuntimeError(
            "Three-seed analysis input is incomplete; expected Seeds 0, 1 and 2:\n"
            + "\n".join(lines)
        )

    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in long.groupby(group_columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values))
        for metric in ANALYSIS_METRICS:
            if metric not in group.columns:
                continue
            values = [_finite_number(value) for value in group[metric]]
            values = [value for value in values if value is not None]
            if not values:
                continue
            if len(values) != len(PUBLIC_SEEDS):
                raise RuntimeError(
                    f"Metric {metric} is missing for one or more seeds in {base}"
                )
            series = pd.Series(values, dtype=float)
            aggregate_rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(series.mean()),
                    "sample_sd": float(series.std(ddof=1)),
                    "n": int(series.count()),
                }
            )
    if not aggregate_rows:
        raise RuntimeError("No supported numeric analysis metrics were found")

    ordered_long = present_identity + sorted(
        column for column in long.columns if column not in present_identity
    )
    long = long[ordered_long].sort_values(
        [column for column in ("study", "grid", "model", "seed") if column in long.columns],
        kind="stable",
        na_position="first",
    )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        [column for column in ("study", "grid", "model", "metric") if column in pd.DataFrame(aggregate_rows).columns],
        kind="stable",
        na_position="first",
    )
    return long.reset_index(drop=True), aggregate.reset_index(drop=True)


def write_tables(long: pd.DataFrame, aggregate: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_dir / "results_long.csv", index=False, lineterminator="\n")
    aggregate.to_csv(
        output_dir / "results_aggregate.csv", index=False, lineterminator="\n"
    )


def main() -> None:
    args = parse_args()
    long, aggregate = build_tables(discover_rows(args.results_root.resolve()))
    write_tables(long, aggregate, args.output_dir.resolve())
    print(f"Wrote {len(long)} normalized rows to {args.output_dir / 'results_long.csv'}")
    print(
        f"Wrote {len(aggregate)} three-seed summaries to "
        f"{args.output_dir / 'results_aggregate.csv'}"
    )


if __name__ == "__main__":
    main()
