#!/usr/bin/env python3
"""16x16 low-resolution pretraining spacing experiment.

DeepRV models are trained once per seed and pretraining grid, then reused across
multiple synthetic-data lengthscales. This isolates the relationship between
pretraining spacing, process lengthscale, posterior fidelity, and measured cost.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import jax.numpy as jnp
import matplotlib
import numpy as np
import numpyro
import wandb
from jax import Array, random
from numpyro import distributions as dist

import exploratory_16x16 as base

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Config(base.Config):
    lengthscales: tuple[float, ...] = (10.0, 30.0, 50.0)
    pretrain_grid_sizes: tuple[int, ...] = (4, 6, 8, 12, 16)
    output_root: str = "outputs/deeprv_lowres_frontier_16x16"
    run_name: str = "lowres_frontier_matern12"
    coverage_loss_threshold: float = 0.03
    log1p_rmse_threshold: float = 0.15
    relative_ls_wasserstein_threshold: float = 0.20
    beta_wasserstein_threshold: float = 0.20
    meaningful_cost_saving_threshold: float = 0.20


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=Config.seed)
    parser.add_argument("--grid-size", type=int, default=Config.grid_size)
    parser.add_argument("--domain-stop", type=float, default=Config.domain_stop)
    parser.add_argument(
        "--lengthscales",
        type=float,
        nargs="+",
        default=list(Config.lengthscales),
    )
    parser.add_argument(
        "--pretrain-grid-sizes",
        type=int,
        nargs="+",
        default=list(Config.pretrain_grid_sizes),
    )
    parser.add_argument("--obs-ratio", type=float, default=Config.obs_ratio)
    parser.add_argument("--train-steps", type=int, default=Config.train_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--valid-steps", type=int, default=Config.valid_steps)
    parser.add_argument("--mcmc-warmup", type=int, default=Config.mcmc_warmup)
    parser.add_argument("--mcmc-samples", type=int, default=Config.mcmc_samples)
    parser.add_argument("--num-chains", type=int, default=Config.num_chains)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--beta-true", type=float, default=Config.beta_true)
    parser.add_argument("--prior-loc", type=float, default=Config.prior_loc)
    parser.add_argument("--prior-scale", type=float, default=Config.prior_scale)
    parser.add_argument("--coverage-level", type=float, default=Config.coverage_level)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=Config.checkpoint_interval
    )
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument(
        "--coverage-loss-threshold",
        type=float,
        default=Config.coverage_loss_threshold,
    )
    parser.add_argument(
        "--log1p-rmse-threshold",
        type=float,
        default=Config.log1p_rmse_threshold,
    )
    parser.add_argument(
        "--relative-ls-wasserstein-threshold",
        type=float,
        default=Config.relative_ls_wasserstein_threshold,
    )
    parser.add_argument(
        "--beta-wasserstein-threshold",
        type=float,
        default=Config.beta_wasserstein_threshold,
    )
    parser.add_argument(
        "--meaningful-cost-saving-threshold",
        type=float,
        default=Config.meaningful_cost_saving_threshold,
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    args.lengthscales = tuple(dict.fromkeys(args.lengthscales))
    args.pretrain_grid_sizes = tuple(dict.fromkeys(args.pretrain_grid_sizes))
    args.models = ()
    cfg = Config(**vars(args))
    if cfg.grid_size != 16:
        raise ValueError("The formal frontier requires --grid-size 16.")
    if not cfg.lengthscales or any(value <= 0 for value in cfg.lengthscales):
        raise ValueError("--lengthscales must contain positive values.")
    if not cfg.pretrain_grid_sizes or any(
        value < 2 or value > cfg.grid_size for value in cfg.pretrain_grid_sizes
    ):
        raise ValueError(
            "--pretrain-grid-sizes must contain values from 2 through grid-size."
        )
    if cfg.grid_size not in cfg.pretrain_grid_sizes:
        raise ValueError(
            "Include the target grid size as the exact full-resolution baseline."
        )
    if not 0.0 < cfg.obs_ratio < 1.0:
        raise ValueError("--obs-ratio must be in (0, 1).")
    if min(
        cfg.train_steps,
        cfg.valid_steps,
        cfg.mcmc_warmup,
        cfg.mcmc_samples,
        cfg.num_chains,
        cfg.checkpoint_interval,
    ) < 1:
        raise ValueError("Training, MCMC, chain, and checkpoint values must be positive.")
    return cfg


def scientific_config(cfg: Config) -> dict:
    values = asdict(cfg)
    for key in ("seed", "force_rerun", "output_root", "models", "gt_ls"):
        values.pop(key, None)
    return json.loads(json.dumps(values))


def prepare_run(cfg: Config) -> Path:
    run_dir = Path(cfg.output_root).expanduser() / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "experiment_config.json"
    current = scientific_config(cfg)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != current:
            raise ValueError(
                f"Configuration mismatch in {run_dir}. Use a new --run-name."
            )
    else:
        base.write_json(manifest_path, current)
    base.write_json(
        run_dir / f"environment_seed_{cfg.seed}.json", base.environment_info()
    )
    (run_dir / f"command_seed_{cfg.seed}.txt").write_text(
        shlex.join(sys.argv) + "\n"
    )
    return run_dir


def grid_label(grid_size: int, target_grid_size: int) -> str:
    if grid_size == target_grid_size:
        return f"deeprv_exact_{grid_size}x{grid_size}"
    return f"deeprv_lowres_{grid_size}x{grid_size}"


def training_dir(run_dir: Path, seed: int, grid_size: int) -> Path:
    return run_dir / "training" / f"seed_{seed}" / f"grid_{grid_size}x{grid_size}"


def scenario_dir(run_dir: Path, gt_ls: float, seed: int) -> Path:
    return run_dir / "scenarios" / f"ls_{gt_ls:g}" / f"seed_{seed}"


def prepare_training(
    cfg: Config,
    run_dir: Path,
    target_s: Array,
    priors: dict,
) -> dict[int, tuple[Callable, dict]]:
    trained = {}
    for grid_size in cfg.pretrain_grid_sizes:
        model_dir = training_dir(run_dir, cfg.seed, grid_size)
        if cfg.force_rerun and model_dir.exists():
            base.reset_output(model_dir.parent, model_dir.name)
        model_dir.mkdir(parents=True, exist_ok=True)
        sample_s = base.make_grid(grid_size, 0.0, cfg.domain_stop)
        base.write_json(
            model_dir / "config.json",
            {
                **asdict(cfg),
                "pretrain_grid_size": grid_size,
                "num_pretrain_locations": int(sample_s.shape[0]),
            },
        )
        print(
            f"Training {grid_label(grid_size, cfg.grid_size)}: "
            f"{sample_s.shape[0]} locations"
        )
        decoder, result = base.train_deeprv(
            cfg,
            "deeprv_lowres_8x8",
            target_s,
            sample_s,
            priors,
            model_dir,
        )
        trained[grid_size] = (decoder, result)
    return trained


def load_or_create_scenario_data(
    cfg: Config,
    run_dir: Path,
    target_s: Array,
    gt_ls: float,
) -> tuple[Path, dict]:
    seed_dir = scenario_dir(run_dir, gt_ls, cfg.seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": cfg.seed,
        "gt_ls": gt_ls,
        "grid_size": cfg.grid_size,
        "domain_stop": cfg.domain_stop,
        "obs_ratio": cfg.obs_ratio,
        "beta_true": cfg.beta_true,
    }
    manifest_path = seed_dir / "scenario_config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Scenario configuration mismatch in {seed_dir}.")
    base.write_json(manifest_path, manifest)
    data_path = seed_dir / "observed_data.pkl"
    if data_path.exists():
        with data_path.open("rb") as f:
            import pickle

            return seed_dir, pickle.load(f)
    rng_data, rng_mask = random.split(random.key(cfg.seed))
    y_full, latent_f, rate = base.gen_y_obs(
        rng_data, target_s, gt_ls, cfg.beta_true
    )
    obs_mask = base.gen_spatial_obs_mask(
        rng_mask, (cfg.grid_size, cfg.grid_size), cfg.obs_ratio
    )
    data = {
        "s": target_s,
        "latent_f": latent_f,
        "rate": rate,
        "y_full": y_full,
        "obs_mask": obs_mask,
        "gt_ls": gt_ls,
        "beta_true": cfg.beta_true,
        "kernel": "matern_1_2",
    }
    base.write_pickle(data_path, data)
    base.plot_truth_and_mask(seed_dir, replace(cfg, gt_ls=gt_ls), data)
    return seed_dir, data


def run_inference(
    cfg: Config,
    model_name: str,
    model_dir: Path,
    data: dict,
    infer_model: Callable,
    surrogate_decoder: Optional[Callable],
    training: Optional[dict],
    sample_s: Optional[Array],
    full_reference: Optional[tuple[dict, dict]],
    hmc_id: int,
) -> tuple[dict, dict, dict]:
    existing = base.load_model_outputs(model_dir)
    if existing is not None and not cfg.force_rerun:
        print(f"{model_name}: complete output exists; skipping")
        return existing
    if cfg.force_rerun and model_dir.exists():
        base.reset_output(model_dir.parent, model_dir.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = {**asdict(cfg), "model_name": model_name}
    base.write_json(model_dir / "config.json", model_cfg)
    hmc_key = random.fold_in(random.key(cfg.seed + 1_000_000), hmc_id)
    print(f"{model_name}: running MCMC")
    samples, posterior, infer_time, ess, diagnostics = base.run_hmc(
        cfg,
        hmc_key,
        infer_model,
        data["y_full"],
        data["obs_mask"],
        surrogate_decoder,
    )
    metrics = base.summarize_model(
        cfg,
        model_name,
        sample_s,
        data,
        samples,
        posterior,
        infer_time,
        ess,
        diagnostics,
        training,
        full_reference,
    )
    base.save_model_outputs(model_dir, samples, posterior, metrics)
    return samples, posterior, metrics


def enrich_scenario_metrics(
    cfg: Config,
    seed_dir: Path,
    gt_ls: float,
    training_by_grid: dict[int, tuple[Callable, dict]],
) -> None:
    full_loaded = base.load_model_outputs(seed_dir / "full_gp")
    if full_loaded is None:
        return
    full_metrics = full_loaded[2]
    coverage_key = f"coverage_{int(cfg.coverage_level * 100)}_unobserved"
    exact_training = training_by_grid[cfg.grid_size][1]
    exact_cost = exact_training["gp_sample_generation_time"] + exact_training[
        "neural_optimization_time"
    ]
    rows = [full_metrics]
    for grid_size in cfg.pretrain_grid_sizes:
        name = grid_label(grid_size, cfg.grid_size)
        loaded = base.load_model_outputs(seed_dir / name)
        if loaded is None:
            continue
        metrics = loaded[2]
        training = training_by_grid[grid_size][1]
        cost = training["gp_sample_generation_time"] + training[
            "neural_optimization_time"
        ]
        spacing = cfg.domain_stop / (grid_size - 1)
        coverage_loss = full_metrics[coverage_key] - metrics[coverage_key]
        log1p_rmse = math.sqrt(metrics["posterior_mean_log1p_mse_vs_full_gp"])
        relative_ls_wasserstein = (
            metrics["ls_wasserstein_vs_full_gp"]
            / full_metrics["posterior_mean_ls"]
        )
        cost_saving = 1.0 - cost / exact_cost
        predictive_acceptable = (
            coverage_loss <= cfg.coverage_loss_threshold
            and log1p_rmse <= cfg.log1p_rmse_threshold
        )
        parameter_acceptable = (
            relative_ls_wasserstein <= cfg.relative_ls_wasserstein_threshold
            and metrics["beta_wasserstein_vs_full_gp"]
            <= cfg.beta_wasserstein_threshold
        )
        metrics.update(
            {
                "gt_ls": gt_ls,
                "pretrain_grid_size": grid_size,
                "pretrain_spacing": spacing,
                "spacing_to_gt_ls_ratio": spacing / gt_ls,
                "pretrain_point_fraction_vs_exact": (
                    grid_size**2 / cfg.grid_size**2
                ),
                "measured_pretraining_cost": cost,
                "measured_pretraining_cost_ratio_vs_exact": cost / exact_cost,
                "measured_pretraining_cost_saving_vs_exact": cost_saving,
                "coverage_loss_vs_full_gp": coverage_loss,
                "posterior_mean_log1p_rmse_vs_full_gp": log1p_rmse,
                "relative_ls_wasserstein_vs_full_gp": relative_ls_wasserstein,
                "predictive_acceptable": predictive_acceptable,
                "parameter_acceptable": parameter_acceptable,
                "fidelity_acceptable": (
                    predictive_acceptable and parameter_acceptable
                ),
                "meaningful_cost_saving": (
                    cost_saving >= cfg.meaningful_cost_saving_threshold
                ),
                "cost_fidelity_acceptable": (
                    predictive_acceptable
                    and parameter_acceptable
                    and cost_saving >= cfg.meaningful_cost_saving_threshold
                ),
            }
        )
        base.write_json(seed_dir / name / "metrics.json", metrics)
        base.write_csv(seed_dir / name / "metrics.csv", [metrics])
        rows.append(metrics)
    base.write_csv(seed_dir / "metrics.csv", rows)


def aggregate_results(cfg: Config, run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.glob("scenarios/ls_*/seed_*/*/metrics.json")):
        row = json.loads(path.read_text())
        if row.get("model_name") != "full_gp":
            rows.append(row)
    if not rows:
        return
    base.write_csv(run_dir / "combined_lowres_metrics.csv", rows)
    group_keys = ("gt_ls", "pretrain_grid_size")
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    summary = []
    for gt_ls in cfg.lengthscales:
        for grid_size in cfg.pretrain_grid_sizes:
            selected = [
                row
                for row in rows
                if row.get("gt_ls") == gt_ls
                and row.get("pretrain_grid_size") == grid_size
            ]
            if not selected:
                continue
            item = {
                "gt_ls": gt_ls,
                "pretrain_grid_size": grid_size,
                "num_seeds": len(selected),
            }
            for key in numeric_keys:
                values = [
                    float(row[key])
                    for row in selected
                    if row.get(key) is not None
                    and np.isfinite(float(row[key]))
                ]
                if values:
                    item[f"{key}_mean"] = float(np.mean(values))
                    item[f"{key}_std"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    )
            for key in (
                "predictive_acceptable",
                "parameter_acceptable",
                "fidelity_acceptable",
                "meaningful_cost_saving",
                "cost_fidelity_acceptable",
            ):
                item[f"{key}_rate"] = float(
                    np.mean([bool(row.get(key, False)) for row in selected])
                )
            summary.append(item)
    base.write_csv(run_dir / "aggregate_lowres_frontier.csv", summary)
    plot_frontier(cfg, run_dir, rows, summary)


def plot_frontier(
    cfg: Config, run_dir: Path, rows: list[dict], summary: list[dict]
) -> None:
    fig, axes = plt.subplots(
        1, len(cfg.lengthscales), figsize=(5 * len(cfg.lengthscales), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, gt_ls in zip(axes, cfg.lengthscales):
        selected = [row for row in summary if row["gt_ls"] == gt_ls]
        for row in selected:
            ax.scatter(
                row["measured_pretraining_cost_ratio_vs_exact_mean"],
                row["posterior_mean_log1p_rmse_vs_full_gp_mean"],
                s=70,
            )
            ax.annotate(
                f'{int(row["pretrain_grid_size"])}x'
                f'{int(row["pretrain_grid_size"])}',
                (
                    row["measured_pretraining_cost_ratio_vs_exact_mean"],
                    row["posterior_mean_log1p_rmse_vs_full_gp_mean"],
                ),
                fontsize=8,
            )
        ax.axhline(cfg.log1p_rmse_threshold, color="black", linestyle="--")
        ax.axvline(
            1.0 - cfg.meaningful_cost_saving_threshold,
            color="black",
            linestyle=":",
        )
        ax.set_title(f"true lengthscale = {gt_ls:g}")
        ax.set_xlabel("pretraining cost / exact cost")
        ax.set_ylabel("log1p posterior mean RMSE vs full GP")
    fig.savefig(run_dir / "cost_fidelity_frontier.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    metrics = (
        ("posterior_mean_log1p_rmse_vs_full_gp", "log1p mean RMSE"),
        ("coverage_loss_vs_full_gp", "unobserved coverage loss"),
        ("relative_ls_wasserstein_vs_full_gp", "relative ls Wasserstein"),
    )
    for ax, (metric, label) in zip(axes, metrics):
        for gt_ls in cfg.lengthscales:
            selected = sorted(
                [row for row in summary if row["gt_ls"] == gt_ls],
                key=lambda row: row["spacing_to_gt_ls_ratio_mean"],
            )
            ax.errorbar(
                [row["spacing_to_gt_ls_ratio_mean"] for row in selected],
                [row[f"{metric}_mean"] for row in selected],
                yerr=[
                    row.get(f"{metric}_std") or 0.0
                    for row in selected
                ],
                marker="o",
                capsize=3,
                label=f"ls={gt_ls:g}",
            )
        ax.set_xlabel("pretraining spacing / true lengthscale")
        ax.set_ylabel(label)
        ax.legend()
    fig.savefig(run_dir / "fidelity_vs_spacing_ratio.png", dpi=180)
    plt.close(fig)

    grids = list(cfg.pretrain_grid_sizes)
    matrix = np.full((len(cfg.lengthscales), len(grids)), np.nan)
    for i, gt_ls in enumerate(cfg.lengthscales):
        for j, grid_size in enumerate(grids):
            selected = [
                row
                for row in rows
                if row["gt_ls"] == gt_ls
                and row["pretrain_grid_size"] == grid_size
            ]
            if selected:
                matrix[i, j] = np.mean(
                    [row["cost_fidelity_acceptable"] for row in selected]
                )
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(grids)), [f"{g}x{g}" for g in grids])
    ax.set_yticks(
        range(len(cfg.lengthscales)), [f"ls={value:g}" for value in cfg.lengthscales]
    )
    ax.set_xlabel("pretraining grid")
    ax.set_title("fraction of seeds meeting cost and fidelity criteria")
    fig.colorbar(image, ax=ax)
    fig.savefig(run_dir / "acceptance_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    numpyro.set_host_device_count(cfg.num_chains)
    wandb.init(mode="disabled")
    run_dir = prepare_run(cfg)
    log_path = run_dir / f"console_seed_{cfg.seed}.log"
    log_file = log_path.open("a", buffering=1)
    sys.stdout = base.Tee(sys.__stdout__, log_file)
    sys.stderr = base.Tee(sys.__stderr__, log_file)
    print("=" * 80)
    print("Started:", base.utc_now())
    print("Output:", run_dir)
    print("Config:", json.dumps(asdict(cfg), indent=2))

    target_s = base.make_grid(cfg.grid_size, 0.0, cfg.domain_stop)
    priors = {
        "ls": dist.LogNormal(loc=cfg.prior_loc, scale=cfg.prior_scale),
        "beta": dist.Normal(0.0, 1.0),
    }
    infer_model = base.inference_model(target_s, priors)
    trained = prepare_training(cfg, run_dir, target_s, priors)

    for ls_index, gt_ls in enumerate(cfg.lengthscales):
        scenario_cfg = replace(cfg, gt_ls=gt_ls)
        seed_dir, data = load_or_create_scenario_data(
            scenario_cfg, run_dir, target_s, gt_ls
        )
        full_samples, full_posterior, _ = run_inference(
            scenario_cfg,
            "full_gp",
            seed_dir / "full_gp",
            data,
            infer_model,
            None,
            None,
            None,
            None,
            10_000 + ls_index,
        )
        full_reference = (full_samples, full_posterior)
        for grid_index, grid_size in enumerate(cfg.pretrain_grid_sizes):
            decoder, training = trained[grid_size]
            sample_s = base.make_grid(grid_size, 0.0, cfg.domain_stop)
            run_inference(
                scenario_cfg,
                grid_label(grid_size, cfg.grid_size),
                seed_dir / grid_label(grid_size, cfg.grid_size),
                data,
                infer_model,
                decoder,
                training,
                sample_s,
                full_reference,
                20_000 + ls_index * 100 + grid_index,
            )
            enrich_scenario_metrics(scenario_cfg, seed_dir, gt_ls, trained)
            aggregate_results(cfg, run_dir)

    print("Finished:", base.utc_now())
    print("Combined metrics:", run_dir / "combined_lowres_metrics.csv")
    print("Aggregate metrics:", run_dir / "aggregate_lowres_frontier.csv")


if __name__ == "__main__":
    main()
