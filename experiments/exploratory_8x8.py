#!/usr/bin/env python3
"""Paper-like 8x8 DeepRV pretraining audit.

This experiment checks full GP MCMC, exact full-resolution DeepRV, low-resolution
full-domain pretraining, and local-region pretraining under settings closer to
the DeepRV synthetic benchmark than the earlier smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import numpyro
import optax
import wandb
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from scipy.stats import wasserstein_distance

from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder
from dl4bi_sps.kernels import matern_1_2
from dl4bi_sps.utils import build_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Config:
    seed: int = 0
    grid_size: int = 8
    domain_stop: float = 100.0
    gt_ls: float = 30.0
    obs_ratio: float = 0.5
    train_steps: int = 200_000
    batch_size: int = 32
    valid_steps: int = 500
    mcmc_warmup: int = 4_000
    mcmc_samples: int = 6_000
    num_chains: int = 2
    lr: float = 5.0e-3
    beta_true: float = 1.0
    prior_loc: float = 3.0
    prior_scale: float = 0.4
    coverage_level: float = 0.9
    output_root: str = "outputs/deeprv_paperlike_8x8_pretraining_audit"
    run_name: str = ""
    pretrain_modes: tuple[str, ...] = ("exact", "lowres", "local")
    pretrain_grid_size: int = 4
    local_grid_size: Optional[int] = 8
    local_region_width: float = 0.5


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Paper-like 8x8 pretraining audit.")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=Config.seed)
    parser.add_argument("--grid-size", type=int, default=Config.grid_size)
    parser.add_argument("--domain-stop", type=float, default=Config.domain_stop)
    parser.add_argument("--gt-ls", type=float, default=Config.gt_ls)
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
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--pretrain-modes",
        nargs="+",
        choices=["exact", "lowres", "local"],
        default=list(Config.pretrain_modes),
    )
    parser.add_argument("--pretrain-grid-size", type=int, default=Config.pretrain_grid_size)
    parser.add_argument("--local-grid-size", type=int, default=Config.local_grid_size)
    parser.add_argument("--local-region-width", type=float, default=Config.local_region_width)
    args = parser.parse_args()
    cfg = Config(**vars(args))
    cfg.pretrain_modes = tuple(cfg.pretrain_modes)
    if cfg.grid_size < 4:
        raise ValueError("--grid-size should be at least 4.")
    if not 0.0 < cfg.obs_ratio < 1.0:
        raise ValueError("--obs-ratio should be in (0, 1).")
    if "lowres" in cfg.pretrain_modes and cfg.pretrain_grid_size >= cfg.grid_size:
        raise ValueError("--pretrain-grid-size must be smaller than --grid-size.")
    if cfg.local_grid_size is not None and cfg.local_grid_size < 2:
        raise ValueError("--local-grid-size should be at least 2.")
    if not 0.0 < cfg.local_region_width <= 1.0:
        raise ValueError("--local-region-width should be in (0, 1].")
    return cfg


def make_output_dir(cfg: Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes = "-".join(cfg.pretrain_modes)
    run_name = cfg.run_name or (
        f"paperlike8x8_matern12_ls{int(cfg.gt_ls)}_{modes}_seed{cfg.seed}_{timestamp}"
    )
    out_dir = Path(cfg.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def make_grid(grid_size: int, domain_start: float, domain_stop: float) -> Array:
    return build_grid(
        [{"start": domain_start, "stop": domain_stop, "num": grid_size}] * 2
    ).reshape(-1, 2)


def gen_spatial_obs_mask(rng: Array, grid_shape: tuple[int, int], obs_ratio: float) -> Array:
    h, w = grid_shape
    total = h * w
    target = int(obs_ratio * total)
    mask = jnp.zeros((h, w), dtype=bool)
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    points_collected = 0
    max_tries = 200
    tries = 0
    while points_collected < target and tries < max_tries:
        tries += 1
        rng, rng_blob = random.split(rng)
        rngs = random.split(rng_blob, 4)
        center_x = random.randint(rngs[0], (), 0, h)
        center_y = random.randint(rngs[1], (), 0, w)
        rx_low, ry_low = max(1, h // 8), max(1, w // 8)
        rx_high, ry_high = max(rx_low + 1, h // 3 + 1), max(ry_low + 1, w // 3 + 1)
        radius_x = random.randint(rngs[2], (), rx_low, rx_high)
        radius_y = random.randint(rngs[3], (), ry_low, ry_high)
        ellipse = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0
        mask = jnp.logical_or(mask, ellipse)
        points_collected = int(mask.sum())
    flat = mask.flatten()
    if int(flat.sum()) < target:
        missing = target - int(flat.sum())
        false_idxs = jnp.argwhere(~flat).reshape(-1)
        rng, rng_fill = random.split(rng)
        chosen = random.choice(rng_fill, false_idxs, shape=(missing,), replace=False)
        flat = flat.at[chosen].set(True)
    elif int(flat.sum()) > target:
        true_idxs = jnp.argwhere(flat).reshape(-1)
        rng, rng_trim = random.split(rng)
        chosen = random.choice(rng_trim, true_idxs, shape=(target,), replace=False)
        flat = jnp.zeros(total, dtype=bool).at[chosen].set(True)
    return flat


def nearest_neighbor_indices(source_s: Array, target_s: Array) -> Array:
    squared_dist = jnp.sum((target_s[:, None, :] - source_s[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(squared_dist, axis=1)


def bilinear_interpolation_matrix(source_s: Array, target_s: Array) -> Array:
    source_np = np.asarray(source_s)
    target_np = np.asarray(target_s)
    x_axis = np.unique(source_np[:, 0])
    y_axis = np.unique(source_np[:, 1])
    if len(x_axis) < 2 or len(y_axis) < 2:
        raise ValueError("Interpolation source must have at least a 2x2 grid.")
    index_by_xy = {(float(x), float(y)): idx for idx, (x, y) in enumerate(source_np)}
    weights = np.zeros((target_np.shape[0], source_np.shape[0]), dtype=np.float32)
    for row, (x_raw, y_raw) in enumerate(target_np):
        x = float(np.clip(x_raw, x_axis[0], x_axis[-1]))
        y = float(np.clip(y_raw, y_axis[0], y_axis[-1]))
        ix0 = int(np.clip(np.searchsorted(x_axis, x, side="right") - 1, 0, len(x_axis) - 2))
        iy0 = int(np.clip(np.searchsorted(y_axis, y, side="right") - 1, 0, len(y_axis) - 2))
        ix1, iy1 = ix0 + 1, iy0 + 1
        x0, x1 = x_axis[ix0], x_axis[ix1]
        y0, y1 = y_axis[iy0], y_axis[iy1]
        wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        corners = [
            (ix0, iy0, (1.0 - wx) * (1.0 - wy)),
            (ix1, iy0, wx * (1.0 - wy)),
            (ix0, iy1, (1.0 - wx) * wy),
            (ix1, iy1, wx * wy),
        ]
        for ix, iy, weight in corners:
            weights[row, index_by_xy[(float(x_axis[ix]), float(y_axis[iy]))]] += weight
    return jnp.asarray(weights)


def build_pretrain_locations(cfg: Config, target_s: Array, mode: str) -> Array:
    if mode == "exact":
        return target_s
    if mode == "lowres":
        return make_grid(cfg.pretrain_grid_size, 0.0, cfg.domain_stop)
    if mode == "local":
        local_grid_size = cfg.local_grid_size or cfg.grid_size
        center = cfg.domain_stop / 2.0
        half_width = cfg.domain_stop * cfg.local_region_width / 2.0
        return make_grid(local_grid_size, center - half_width, center + half_width)
    raise ValueError(f"Unknown mode: {mode}")


def gen_y_obs(rng: Array, s: Array, gt_ls: float, beta_true: float):
    rng_mu, rng_poiss = random.split(rng)
    kernel = matern_1_2(s, s, 1.0, gt_ls) + 5e-4 * jnp.eye(s.shape[0])
    latent_f = dist.MultivariateNormal(jnp.zeros(s.shape[0]), kernel).sample(rng_mu)
    rate = jnp.exp(beta_true + latent_f)
    y_full = dist.Poisson(rate=rate).sample(rng_poiss)
    return y_full, latent_f, rate


def inference_model(s: Array, priors: dict):
    surrogate_kwargs = {"s": s}

    def poisson(surrogate_decoder: Optional[Callable] = None, obs_mask=True, y=None):
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            k = matern_1_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.deterministic("mu", jnp.linalg.cholesky(k) @ z[0])
        else:
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze(),
            )
        rate = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=rate), obs=y)

    return poisson


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_train_dataloader(target_s: Array, sample_s: Array, priors: dict, batch_size: int):
    jitter = 5e-4 * jnp.eye(sample_s.shape[0])
    interpolation = bilinear_interpolation_matrix(sample_s, target_s)
    latent_idx = nearest_neighbor_indices(target_s, sample_s)
    kernel_jit = jit(lambda locs, var, ls: matern_1_2(locs, locs, var, ls) + jitter)
    f_jit = jit(lambda l_chol, z: jnp.einsum("ij,bj->bi", l_chol, z))
    interpolate_jit = jit(lambda f_sample: jnp.einsum("ts,bs->bt", interpolation, f_sample))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            ls = priors["ls"].sample(rng_ls)
            z_target = dist.Normal().sample(rng_z, sample_shape=(batch_size, target_s.shape[0]))
            z_sample = z_target[:, latent_idx]
            k = kernel_jit(sample_s, 1.0, ls)
            f_sample = f_jit(jnp.linalg.cholesky(k), z_sample)
            f_target = interpolate_jit(f_sample)
            yield {"s": target_s, "z": z_target, "conditionals": jnp.array([ls]), "f": f_target}

    return dataloader


def train_deeprv(
    cfg: Config,
    rng_train: Array,
    rng_test: Array,
    target_s: Array,
    sample_s: Array,
    priors: dict,
):
    model = gMLPDeepRV(num_blks=2)
    loader = gen_train_dataloader(target_s, sample_s, priors, cfg.batch_size)
    lr_schedule = cosine_annealing_lr(cfg.train_steps, cfg.lr)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optax.yogi(lr_schedule))
    valid_interval = max(1, min(10_000, cfg.train_steps))
    start = perf_counter()
    state = train(
        rng_train,
        model,
        optimizer,
        deep_rv_train_step,
        cfg.train_steps,
        loader,
        valid_step,
        valid_interval,
        cfg.valid_steps,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
        log_loss_interval=max(1, min(1_000, cfg.train_steps)),
    )
    train_time = perf_counter() - start
    eval_mse = evaluate(rng_test, state, valid_step, loader, cfg.valid_steps)["norm MSE"]
    return model, state, generate_surrogate_decoder(state, model), train_time, float(eval_mse)


def run_hmc(
    cfg: Config,
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Array,
    surrogate_decoder: Optional[Callable] = None,
):
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(
        nuts,
        num_chains=cfg.num_chains,
        num_samples=cfg.mcmc_samples,
        num_warmup=cfg.mcmc_warmup,
        progress_bar=True,
    )
    rng_run, rng_pred = random.split(rng)
    start = perf_counter()
    mcmc.run(rng_run, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = perf_counter() - start
    mcmc.print_summary()
    samples_all = mcmc.get_samples()
    posterior = Predictive(model, samples_all)(rng_pred, surrogate_decoder=surrogate_decoder)
    samples_small = {k: v for k, v in samples_all.items() if k in ["ls", "beta"]}
    ess = az.ess(mcmc, method="mean")
    return samples_small, posterior, infer_time, ess


def mse_values(y: Array, y_hat: Array, mask: Array):
    sq = (y - y_hat) ** 2
    log_sq = (jnp.log1p(y) - jnp.log1p(y_hat)) ** 2
    denom = jnp.var(y) + 1e-8
    return {
        "mse_all": float(sq.mean()),
        "mse_observed": float(sq[mask].mean()),
        "mse_unobserved": float(sq[jnp.logical_not(mask)].mean()),
        "log1p_mse_all": float(log_sq.mean()),
        "log1p_mse_observed": float(log_sq[mask].mean()),
        "log1p_mse_unobserved": float(log_sq[jnp.logical_not(mask)].mean()),
        "nrmse_all": float(jnp.sqrt(sq.mean() / denom)),
        "nrmse_observed": float(jnp.sqrt(sq[mask].mean() / denom)),
        "nrmse_unobserved": float(jnp.sqrt(sq[jnp.logical_not(mask)].mean() / denom)),
    }


def coverage_values(y: Array, posterior_obs: Array, mask: Array, level: float):
    alpha = 1.0 - level
    lo = jnp.quantile(posterior_obs, alpha / 2.0, axis=0)
    hi = jnp.quantile(posterior_obs, 1.0 - alpha / 2.0, axis=0)
    covered = jnp.logical_and(y >= lo, y <= hi)
    return {
        f"coverage_{int(level * 100)}_all": float(covered.mean()),
        f"coverage_{int(level * 100)}_observed": float(covered[mask].mean()),
        f"coverage_{int(level * 100)}_unobserved": float(covered[jnp.logical_not(mask)].mean()),
    }


def y_summary(y: Array, mask: Array):
    return {
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_observed_mean": float(y[mask].mean()),
        "y_observed_max": float(y[mask].max()),
        "y_unobserved_mean": float(y[jnp.logical_not(mask)].mean()),
        "y_unobserved_max": float(y[jnp.logical_not(mask)].max()),
    }


def summarize_model(
    model_name: str,
    pretrain_mode: str,
    pretrain_locations: Optional[Array],
    y_full: Array,
    obs_mask: Array,
    samples: dict,
    posterior: dict,
    infer_time: float,
    ess,
    coverage_level: float,
    train_time: Optional[float] = None,
    train_norm_mse: Optional[float] = None,
    full_gp_mean: Optional[Array] = None,
    full_gp_samples: Optional[dict] = None,
):
    y_hat = posterior["obs"].mean(axis=0)
    row = {
        "model_name": model_name,
        "pretrain_mode": pretrain_mode,
        "num_pretrain_locations": None if pretrain_locations is None else int(pretrain_locations.shape[0]),
        "pretrain_domain_min": None if pretrain_locations is None else float(pretrain_locations.min()),
        "pretrain_domain_max": None if pretrain_locations is None else float(pretrain_locations.max()),
        "train_time": train_time,
        "train_norm_mse": train_norm_mse,
        "infer_time": infer_time,
        "posterior_mean_ls": float(samples["ls"].mean()),
        "posterior_mean_beta": float(samples["beta"].mean()),
    }
    row.update(y_summary(y_full, obs_mask))
    row.update(mse_values(y_full, y_hat, obs_mask))
    row.update(coverage_values(y_full, posterior["obs"], obs_mask, coverage_level))
    if ess is not None:
        for key in ["ls", "beta"]:
            try:
                row[f"ESS_{key}"] = float(ess[key].mean().item())
            except Exception:
                row[f"ESS_{key}"] = None
    if full_gp_mean is not None:
        row["posterior_mean_mse_vs_full_gp"] = float(jnp.mean((y_hat - full_gp_mean) ** 2))
        row["posterior_mean_log1p_mse_vs_full_gp"] = float(
            jnp.mean((jnp.log1p(y_hat) - jnp.log1p(full_gp_mean)) ** 2)
        )
    if full_gp_samples is not None:
        row["ls_wasserstein_vs_full_gp"] = float(
            wasserstein_distance(full_gp_samples["ls"], samples["ls"])
        )
        row["beta_wasserstein_vs_full_gp"] = float(
            wasserstein_distance(full_gp_samples["beta"], samples["beta"])
        )
    return row


def save_metrics(out_dir: Path, rows: list[dict]):
    keys = sorted({key for row in rows for key in row.keys()})
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def plot_truth_and_mask(out_dir: Path, cfg: Config, latent_f: Array, rate: Array, y_full: Array, obs_mask: Array):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fields = [
        (latent_f.reshape(cfg.grid_size, cfg.grid_size), "latent f"),
        (jnp.log(rate).reshape(cfg.grid_size, cfg.grid_size), "log rate beta+f"),
        (jnp.log1p(y_full).reshape(cfg.grid_size, cfg.grid_size), "log1p counts y"),
        (obs_mask.reshape(cfg.grid_size, cfg.grid_size), "observed mask"),
    ]
    for ax, (field, title) in zip(axes, fields):
        im = ax.imshow(np.asarray(field), origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(out_dir / "diagnostic_truth_and_mask.png", dpi=180)
    plt.close(fig)


def plot_predictive_means(out_dir: Path, cfg: Config, y_full: Array, obs_mask: Array, posteriors: dict[str, dict]):
    model_names = list(posteriors)
    ncols = 2 + len(model_names)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    observed_log = np.ma.masked_where(
        ~np.asarray(obs_mask.reshape(cfg.grid_size, cfg.grid_size)),
        np.asarray(jnp.log1p(y_full).reshape(cfg.grid_size, cfg.grid_size)),
    )
    panels = [(observed_log, "observed log1p(y)"), (jnp.log1p(y_full).reshape(cfg.grid_size, cfg.grid_size), "full log1p(y)")]
    for name in model_names:
        mean = posteriors[name]["obs"].mean(axis=0)
        panels.append((jnp.log1p(mean).reshape(cfg.grid_size, cfg.grid_size), f"{name} mean"))
    for ax, (field, title) in zip(axes, panels):
        im = ax.imshow(np.asarray(field), origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(out_dir / "posterior_predictive_means.png", dpi=180)
    plt.close(fig)


def plot_difference_to_full_gp(out_dir: Path, cfg: Config, posteriors: dict[str, dict]):
    full_gp_mean = posteriors["full_gp"]["obs"].mean(axis=0)
    model_names = [name for name in posteriors if name != "full_gp"]
    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 4), constrained_layout=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        mean = posteriors[name]["obs"].mean(axis=0)
        field = (mean - full_gp_mean).reshape(cfg.grid_size, cfg.grid_size)
        im = ax.imshow(np.asarray(field), origin="lower", cmap="coolwarm")
        ax.set_title(f"{name} - full GP")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(out_dir / "posterior_mean_differences_vs_full_gp.png", dpi=180)
    plt.close(fig)


def plot_lengthscale_posteriors(out_dir: Path, samples_by_model: dict[str, dict], gt_ls: float):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name, samples in samples_by_model.items():
        ax.hist(np.asarray(samples["ls"]), bins=40, alpha=0.5, density=True, label=name)
    ax.axvline(gt_ls, color="black", linestyle="--", label="true ls")
    ax.set_xlabel("lengthscale")
    ax.set_ylabel("density")
    ax.legend()
    fig.savefig(out_dir / "lengthscale_posteriors.png", dpi=180)
    plt.close(fig)


def save_runtime_info(out_dir: Path):
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "numpyro_version": numpyro.__version__,
        "cwd": str(Path.cwd()),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
    }
    with (out_dir / "environment.json").open("w") as f:
        json.dump(info, f, indent=2)
    with (out_dir / "command.txt").open("w") as f:
        f.write(" ".join(sys.argv) + "\n")


def main():
    cfg = parse_args()
    numpyro.set_host_device_count(cfg.num_chains)
    wandb.init(mode="disabled")
    out_dir = make_output_dir(cfg)
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)
    save_runtime_info(out_dir)

    print("JAX devices:", jax.devices())
    print("Writing outputs to:", out_dir)
    print("Config:", json.dumps(asdict(cfg), indent=2))

    rng = random.key(cfg.seed)
    rng_data, rng_mask, rng_full, rng_train, rng_test, rng_drv = random.split(rng, 6)
    s = make_grid(cfg.grid_size, 0.0, cfg.domain_stop)
    priors = {
        "ls": dist.LogNormal(loc=cfg.prior_loc, scale=cfg.prior_scale),
        "beta": dist.Normal(),
    }
    y_full, latent_f, rate = gen_y_obs(rng_data, s, cfg.gt_ls, cfg.beta_true)
    obs_mask = gen_spatial_obs_mask(rng_mask, (cfg.grid_size, cfg.grid_size), cfg.obs_ratio)
    infer_model = inference_model(s, priors)

    with (out_dir / "observed_data.pkl").open("wb") as f:
        pickle.dump(
            {
                "s": s,
                "latent_f": latent_f,
                "rate": rate,
                "y_full": y_full,
                "obs_mask": obs_mask,
                "gt_ls": cfg.gt_ls,
                "beta_true": cfg.beta_true,
                "kernel": "matern_1_2",
            },
            f,
        )
    plot_truth_and_mask(out_dir, cfg, latent_f, rate, y_full, obs_mask)

    print("Running full GP MCMC reference...")
    samples_gp, posterior_gp, infer_time_gp, ess_gp = run_hmc(
        cfg, rng_full, infer_model, y_full, obs_mask
    )
    with (out_dir / "samples_full_gp.pkl").open("wb") as f:
        pickle.dump(samples_gp, f)

    rows = [
        summarize_model(
            "full_gp",
            "",
            None,
            y_full,
            obs_mask,
            samples_gp,
            posterior_gp,
            infer_time_gp,
            ess_gp,
            cfg.coverage_level,
        )
    ]
    posteriors = {"full_gp": posterior_gp}
    samples_by_model = {"full_gp": samples_gp}
    full_gp_mean = posterior_gp["obs"].mean(axis=0)

    mode_train_keys = random.split(rng_train, len(cfg.pretrain_modes))
    mode_test_keys = random.split(rng_test, len(cfg.pretrain_modes))
    mode_hmc_keys = random.split(rng_drv, len(cfg.pretrain_modes))
    for mode, mode_rng_train, mode_rng_test, mode_rng_hmc in zip(
        cfg.pretrain_modes, mode_train_keys, mode_test_keys, mode_hmc_keys
    ):
        s_pretrain = build_pretrain_locations(cfg, s, mode)
        model_name = f"deeprv_{mode}"
        print(f"Training {model_name} on {s_pretrain.shape[0]} pretraining locations...")
        model, state, surrogate_decoder, train_time, train_norm_mse = train_deeprv(
            cfg, mode_rng_train, mode_rng_test, s, s_pretrain, priors
        )
        del model, state
        print(f"Running MCMC for {model_name}...")
        samples_drv, posterior_drv, infer_time_drv, ess_drv = run_hmc(
            cfg, mode_rng_hmc, infer_model, y_full, obs_mask, surrogate_decoder
        )
        with (out_dir / f"samples_deeprv_{mode}.pkl").open("wb") as f:
            pickle.dump(samples_drv, f)
        rows.append(
            summarize_model(
                model_name,
                mode,
                s_pretrain,
                y_full,
                obs_mask,
                samples_drv,
                posterior_drv,
                infer_time_drv,
                ess_drv,
                cfg.coverage_level,
                train_time=train_time,
                train_norm_mse=train_norm_mse,
                full_gp_mean=full_gp_mean,
                full_gp_samples=samples_gp,
            )
        )
        posteriors[model_name] = posterior_drv
        samples_by_model[model_name] = samples_drv

    with (out_dir / "posterior_predictives.pkl").open("wb") as f:
        pickle.dump(posteriors, f)
    save_metrics(out_dir, rows)
    plot_predictive_means(out_dir, cfg, y_full, obs_mask, posteriors)
    plot_difference_to_full_gp(out_dir, cfg, posteriors)
    plot_lengthscale_posteriors(out_dir, samples_by_model, cfg.gt_ls)

    print("\nPaper-like 8x8 pretraining audit complete.")
    print("Output directory:", out_dir)
    print("Metrics:", out_dir / "metrics.csv")
    print("Truth/mask plot:", out_dir / "diagnostic_truth_and_mask.png")
    print("Predictive plot:", out_dir / "posterior_predictive_means.png")
    print("Difference plot:", out_dir / "posterior_mean_differences_vs_full_gp.png")
    print("Lengthscale plot:", out_dir / "lengthscale_posteriors.png")


if __name__ == "__main__":
    main()
