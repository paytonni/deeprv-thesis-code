#!/usr/bin/env python3
"""Build thesis configuration tables from the public experiment defaults."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
DEFAULT_OUT = ROOT / "tables" / "generated"
SOURCES = {
    "8x8": EXPERIMENTS / "exploratory_8x8.py",
    "16x16": EXPERIMENTS / "exploratory_16x16.py",
    "32x32": EXPERIMENTS / "grid32.py",
    "64x64": EXPERIMENTS / "grid64.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for generated LaTeX tables.",
    )
    return parser.parse_args()


def config_defaults(path: Path) -> dict[str, Any]:
    """Read literal Config defaults without importing GPU-oriented modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            values: dict[str, Any] = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.value is None:
                        continue
                    try:
                        values[item.target.id] = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        if isinstance(item.value, ast.Name) and item.value.id in constants:
                            values[item.target.id] = constants[item.value.id]
            return values
    raise RuntimeError(f"Config class not found in {path}")


def target128_defaults() -> dict[str, Any]:
    path = EXPERIMENTS / "configs" / "target128.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read Target-128 configuration {path}: {error}") from error


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def integer(value: Any) -> str:
    return f"{int(value):,}"


def build_rows() -> list[tuple[str, list[str]]]:
    configs = {label: config_defaults(path) for label, path in SOURCES.items()}
    target = target128_defaults()
    spacing = config_defaults(EXPERIMENTS / "exploratory_16x16_spacing.py")
    columns = [configs[label] for label in SOURCES]

    def values(key: str, target_key: str | None = None) -> list[Any]:
        return [config[key] for config in columns] + [target[target_key or key]]

    domain = [f"[0, {config['domain_stop']:g}]" for config in columns]
    domain.append(f"[{target['domain'][0]:g}, {target['domain'][1]:g}]")
    lengthscales = values("gt_ls", "true_lengthscale")
    masks = [
        f"spatial, {100 * columns[0]['obs_ratio']:g}% observed",
        f"spatial, {100 * columns[1]['obs_ratio']:g}% observed",
        f"{columns[2]['obs_mask_type']}, {100 * columns[2]['obs_ratio']:g}% observed",
        f"{columns[3]['obs_mask_type']}, {100 * columns[3]['obs_ratio']:g}% observed",
        f"{target['mask_type']}, {100 * target['observation_ratio']:g}% observed",
    ]
    chains = [
        columns[0]["num_chains"],
        columns[1]["num_chains"],
        columns[2]["num_chains"],
        columns[3]["num_chains"],
        target["num_chains"],
    ]
    warmup = [
        columns[0]["mcmc_warmup"],
        columns[1]["mcmc_warmup"],
        columns[2]["initial_warmup"],
        columns[3]["initial_warmup"],
        target["num_warmup"],
    ]
    retained = [
        columns[0]["mcmc_samples"],
        columns[1]["mcmc_samples"],
        columns[2]["initial_samples"],
        columns[3]["initial_samples"],
        target["num_samples"],
    ]
    train_steps = values("train_steps", "formal_train_steps")
    batch_sizes = values("batch_size", "microbatch_size")
    validation_intervals = values("validation_interval")
    validation_batches = values("validation_batches")
    checkpoint_intervals = [
        "--",
        integer(columns[1]["checkpoint_save_interval"]),
        integer(columns[2]["checkpoint_save_interval"]),
        integer(columns[3]["checkpoint_save_interval"]),
        integer(target["checkpoint_save_interval"]),
    ]
    target_checkpoints = ", ".join(
        f"{model}: {integer(step) if step is not None else 'none'}"
        for model, step in target["formal_checkpoint_steps"].items()
    )
    small_models = ["Full GP"]
    for mode in columns[0]["pretrain_modes"]:
        if mode == "exact":
            small_models.append("Exact")
        elif mode == "lowres":
            small_models.append(f"Lowres{columns[0]['pretrain_grid_size']}")
        elif mode == "local":
            small_models.append(f"Local{columns[0]['local_grid_size']}")
    def support_label(name: str) -> str:
        if name == "full_gp":
            return "Full GP"
        if name == "deeprv_exact":
            return "Exact"
        return name.removeprefix("deeprv_").replace("_", " ").title().replace("X", "x")

    support_models = [support_label(name) for name in columns[1]["models"]]
    comparisons = [
        "; ".join(small_models),
        "; ".join(support_models),
        "Full GP; Exact DeepRV; DeepRV "
        + "/".join(value.upper() for value in columns[2]["weightings"])
        + " at sides "
        + ", ".join(map(str, columns[2]["inducing_grid_sizes"])),
        "Full GP; Exact DeepRV; DeepRV "
        + "/".join(value.upper() for value in columns[3]["weightings"])
        + " at sides "
        + ", ".join(map(str, columns[3]["inducing_grid_sizes"]))
        + "; matched Direct GP "
        + "/".join(value.upper() for value in columns[3]["weightings"])
        + " at sides "
        + ", ".join(map(str, columns[3]["inducing_grid_sizes"])),
        "; ".join(target["models"]),
    ]
    spacing_summary = (
        "lengthscales "
        + ", ".join(f"{value:g}" for value in spacing["lengthscales"])
        + "; sides "
        + ", ".join(map(str, spacing["pretrain_grid_sizes"]))
    )

    return [
        ("Domain", domain),
        ("Latent field", [f"Matérn-1/2, ell={value:g}" for value in lengthscales]),
        ("Likelihood", ["Poisson with log link"] * 5),
        ("Observation protocol", masks),
        ("Dataset seeds", ["0, 1, 2"] * 5),
        ("Compared constructions", comparisons),
        ("Spacing study", ["--", spacing_summary, "--", "--", "--"]),
        ("NUTS chains", [str(value) for value in chains]),
        ("NUTS warmup per chain", [integer(value) for value in warmup]),
        ("NUTS retained per chain", [integer(value) for value in retained]),
        ("Decoder", ["gMLPDeepRV, 2 blocks"] * 5),
        ("Training steps", [integer(value) for value in train_steps]),
        ("Batch size", [integer(value) for value in batch_sizes]),
        (
            "Validation",
            [
                f"every {integer(interval)} steps; {integer(batches)} batches"
                for interval, batches in zip(validation_intervals, validation_batches)
            ],
        ),
        ("Optimiser", ["Yogi; cosine schedule; gradient clip 3"] * 5),
        ("Checkpoint interval", checkpoint_intervals),
        ("Formal checkpoint", ["--", "--", "--", "--", target_checkpoints]),
    ]


def table_lines(rows: list[tuple[str, list[str]]], *, longtable: bool) -> list[str]:
    environment = "longtable" if longtable else "tabular"
    column_spec = r">{\raggedright\arraybackslash}p{0.18\textwidth}" + " " + " ".join(
        [r">{\raggedright\arraybackslash}p{0.145\textwidth}"] * 5
    )
    lines = [
        "% Generated by analysis/tables/build_configuration_table.py",
        "% Source: public experiment Config defaults and experiments/configs/target128.json",
        rf"\begin{{{environment}}}{{{column_spec}}}",
    ]
    if longtable:
        lines.append(r"\caption{Complete experimental configuration by target scale.}\label{tab:supp-full-configuration}\\")
    lines.extend(
        [
            r"\toprule",
            r"Setting & $8\times8$ & $16\times16$ & $32\times32$ & $64\times64$ & Target-128 \\",
            r"\midrule",
        ]
    )
    if longtable:
        lines.extend(
            [
                r"\endfirsthead",
                r"\toprule",
                r"Setting & $8\times8$ & $16\times16$ & $32\times32$ & $64\times64$ & Target-128 \\",
                r"\midrule",
                r"\endhead",
            ]
        )
    for label, entries in rows:
        lines.append(
            " & ".join([latex_escape(label), *[latex_escape(str(value)) for value in entries]])
            + r" \\"
        )
    lines.extend([r"\bottomrule", rf"\end{{{environment}}}", ""])
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    (args.output_dir / "tab01_configuration.tex").write_text(
        "\n".join(table_lines(rows, longtable=False)), encoding="utf-8"
    )
    (args.output_dir / "tabS01_full_configuration.tex").write_text(
        "\n".join(table_lines(rows, longtable=True)), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
