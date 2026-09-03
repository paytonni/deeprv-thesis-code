#!/usr/bin/env python3
"""16x16 synthetic Poisson GP pretraining comparison.

The experiment separates DeepRV approximation error, cheap-pretraining error,
and local-domain distribution shift using a full GP reference and four DeepRV
pretraining designs. Each seed/model is persisted independently so interrupted
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from flax.training import orbax_utils
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from orbax.checkpoint import PyTreeCheckpointer
from scipy.stats import wasserstein_distance

from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import TrainState, cosine_annealing_lr, evaluate
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder
from dl4bi_sps.kernels import matern_1_2
from dl4bi_sps.utils import build_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ALL_MODELS = (
    "full_gp",
    "deeprv_exact",
    "deeprv_lowres_8x8",
    "deeprv_local_8x8",
    "deeprv_local_16x16",
)
MODEL_IDS = {name: idx for idx, name in enumerate(ALL_MODELS)}


@dataclass
class Config:
    seed: int = 0
    grid_size: int = 16
    domain_stop: float = 100.0
    gt_ls: float = 30.0
    obs_ratio: float = 0.5
    train_steps: int = 200_000
    batch_size: int = 32
    validation_interval: int = 10_000
    validation_batches: int = 500
    mcmc_warmup: int = 4_000
    mcmc_samples: int = 6_000
    num_chains: int = 2
    lr: float = 5.0e-3
    beta_true: float = 1.0
    prior_loc: float = 3.0
    prior_scale: float = 0.4
    coverage_level: float = 0.9
    checkpoint_save_interval: int = 10_000
    output_root: str = "outputs/deeprv_paperlike_16x16_pretraining_comparison"
    run_name: str = "paperlike16x16_matern12_ls30"
    models: tuple[str, ...] = ALL_MODELS
    force_rerun: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=Config.seed)
    parser.add_argument("--grid-size", type=int, default=Config.grid_size)
    parser.add_argument("--domain-stop", type=float, default=Config.domain_stop)
    parser.add_argument("--gt-ls", type=float, default=Config.gt_ls)
    parser.add_argument("--obs-ratio", type=float, default=Config.obs_ratio)
    parser.add_argument("--train-steps", type=int, default=Config.train_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument(
        "--validation-interval", type=int, default=Config.validation_interval
    )
    parser.add_argument(
        "--validation-batches", type=int, default=Config.validation_batches
    )
    parser.add_argument("--mcmc-warmup", type=int, default=Config.mcmc_warmup)
    parser.add_argument("--mcmc-samples", type=int, default=Config.mcmc_samples)
    parser.add_argument("--num-chains", type=int, default=Config.num_chains)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--beta-true", type=float, default=Config.beta_true)
    parser.add_argument("--prior-loc", type=float, default=Config.prior_loc)
    parser.add_argument("--prior-scale", type=float, default=Config.prior_scale)
    parser.add_argument("--coverage-level", type=float, default=Config.coverage_level)
    parser.add_argument(
        "--checkpoint-save-interval",
        type=int,
        default=Config.checkpoint_save_interval,
    )
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(Config.models),
        help="Models to run. DeepRV-only runs require a completed full_gp reference.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    cfg = Config(**vars(args))
    cfg.models = tuple(dict.fromkeys(cfg.models))
    if cfg.grid_size != 16:
        raise ValueError("The formal experiment requires --grid-size 16.")
    if not 0.0 < cfg.obs_ratio < 1.0:
        raise ValueError("--obs-ratio must be in (0, 1).")
    if min(cfg.train_steps, cfg.validation_interval, cfg.validation_batches) < 1:
        raise ValueError("Training and validation steps must be positive.")
    if cfg.mcmc_warmup < 1 or cfg.mcmc_samples < 1 or cfg.num_chains < 1:
        raise ValueError("MCMC warmup, samples, and chains must be positive.")
    if cfg.checkpoint_save_interval < 1:
        raise ValueError("--checkpoint-save-interval must be positive.")
    if not 0.0 < cfg.coverage_level < 1.0:
        raise ValueError("--coverage-level must be in (0, 1).")
    return cfg


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
    tmp.replace(path)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class MemoryMonitor:
    """Poll process RSS and NVIDIA process memory when those sources exist."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.pid = os.getpid()
        self.peak_rss_mb: Optional[float] = None
        self.peak_gpu_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _poll(self):
        while not self._stop.is_set():
            rss = self._read_rss_mb()
            gpu = self._read_gpu_mb()
            if rss is not None:
                self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss)
            if gpu is not None:
                self.peak_gpu_mb = max(self.peak_gpu_mb or 0.0, gpu)
            self._stop.wait(self.interval)

    def _read_rss_mb(self) -> Optional[float]:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak <= 0:
            return None
        divisor = 1024.0 if platform.system() != "Darwin" else 1024.0**2
        return float(peak) / divisor

    def _read_gpu_mb(self) -> Optional[float]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        used = 0.0
        found = False
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid):
                used += float(parts[1])
                found = True
        return used if found else None


def git_provenance() -> dict:
    def run_git(*args):
        result = subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "branch": run_git("branch", "--show-current"),
        "commit": run_git("rev-parse", "HEAD"),
        "status_short": run_git("status", "--short"),
    }


def environment_info() -> dict:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "numpyro_version": numpyro.__version__,
        "cwd": str(Path.cwd()),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "git": git_provenance(),
    }


def scientific_config(cfg: Config) -> dict:
    values = asdict(cfg)
    for key in ("models", "force_rerun", "output_root"):
        values.pop(key)
    return values


def prepare_seed_dir(cfg: Config) -> tuple[Path, Path]:
    run_dir = Path(cfg.output_root).expanduser() / cfg.run_name
    seed_dir = run_dir / f"seed_{cfg.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = seed_dir / "seed_config.json"
    current = scientific_config(cfg)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != current:
            raise ValueError(
                f"Configuration mismatch in {seed_dir}. Use a new --run-name "
                "instead of mixing incompatible results."
            )
    else:
        write_json(manifest_path, current)
    write_json(seed_dir / "environment.json", environment_info())
    (seed_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    return run_dir, seed_dir


def reset_output(seed_dir: Path, model_name: str) -> None:
    model_dir = seed_dir / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)


def make_grid(grid_size: int, domain_start: float, domain_stop: float) -> Array:
    return build_grid(
        [{"start": domain_start, "stop": domain_stop, "num": grid_size}] * 2
    ).reshape(-1, 2)


def gen_spatial_obs_mask(
    rng: Array, grid_shape: tuple[int, int], obs_ratio: float
) -> Array:
    h, w = grid_shape
    total = h * w
    target = int(obs_ratio * total)
    mask = jnp.zeros((h, w), dtype=bool)
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    points_collected = 0
    tries = 0
    while points_collected < target and tries < 200:
        tries += 1
        rng, rng_blob = random.split(rng)
        rngs = random.split(rng_blob, 4)
        center_x = random.randint(rngs[0], (), 0, h)
        center_y = random.randint(rngs[1], (), 0, w)
        rx_low, ry_low = max(1, h // 8), max(1, w // 8)
        rx_high = max(rx_low + 1, h // 3 + 1)
        ry_high = max(ry_low + 1, w // 3 + 1)
        radius_x = random.randint(rngs[2], (), rx_low, rx_high)
        radius_y = random.randint(rngs[3], (), ry_low, ry_high)
        ellipse = (
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
            <= 1.0
        )
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


def bilinear_interpolation_matrix(source_s: Array, target_s: Array) -> Array:
    source_np = np.asarray(source_s)
    target_np = np.asarray(target_s)
    x_axis = np.unique(source_np[:, 0])
    y_axis = np.unique(source_np[:, 1])
    if len(x_axis) < 2 or len(y_axis) < 2:
        raise ValueError("Interpolation source must have at least a 2x2 grid.")
    index_by_xy = {(float(x), float(y)): i for i, (x, y) in enumerate(source_np)}
    weights = np.zeros((target_np.shape[0], source_np.shape[0]), dtype=np.float32)
    for row, (x_raw, y_raw) in enumerate(target_np):
        x = float(np.clip(x_raw, x_axis[0], x_axis[-1]))
        y = float(np.clip(y_raw, y_axis[0], y_axis[-1]))
        ix0 = int(
            np.clip(
                np.searchsorted(x_axis, x, side="right") - 1, 0, len(x_axis) - 2
            )
        )
        iy0 = int(
            np.clip(
                np.searchsorted(y_axis, y, side="right") - 1, 0, len(y_axis) - 2
            )
        )
        ix1, iy1 = ix0 + 1, iy0 + 1
        x0, x1 = x_axis[ix0], x_axis[ix1]
        y0, y1 = y_axis[iy0], y_axis[iy1]
        wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        corners = (
            (ix0, iy0, (1.0 - wx) * (1.0 - wy)),
            (ix1, iy0, wx * (1.0 - wy)),
            (ix0, iy1, (1.0 - wx) * wy),
            (ix1, iy1, wx * wy),
        )
        for ix, iy, weight in corners:
            idx = index_by_xy[(float(x_axis[ix]), float(y_axis[iy]))]
            weights[row, idx] += weight
    return jnp.asarray(weights)


def nearest_target_indices(target_s: Array, sample_s: Array) -> Array:
    squared_dist = jnp.sum(
        (sample_s[:, None, :] - target_s[None, :, :]) ** 2, axis=-1
    )
    return jnp.argmin(squared_dist, axis=1)


def pretrain_locations(model_name: str, target_s: Array, domain_stop: float) -> Array:
    if model_name == "deeprv_exact":
        return target_s
    if model_name == "deeprv_lowres_8x8":
        return make_grid(8, 0.0, domain_stop)
    if model_name == "deeprv_local_8x8":
        return make_grid(8, domain_stop * 0.25, domain_stop * 0.75)
    if model_name == "deeprv_local_16x16":
        return make_grid(16, domain_stop * 0.25, domain_stop * 0.75)
    raise ValueError(f"No pretraining design for {model_name}.")


def gen_y_obs(rng: Array, s: Array, gt_ls: float, beta_true: float):
    rng_mu, rng_poiss = random.split(rng)
    kernel = matern_1_2(s, s, 1.0, gt_ls) + 5e-4 * jnp.eye(s.shape[0])
    latent_f = dist.MultivariateNormal(jnp.zeros(s.shape[0]), kernel).sample(rng_mu)
    rate = jnp.exp(beta_true + latent_f)
    y_full = dist.Poisson(rate=rate).sample(rng_poiss)
    return y_full, latent_f, rate


def load_or_create_data(cfg: Config, seed_dir: Path, s: Array) -> dict:
    path = seed_dir / "observed_data.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    rng = random.key(cfg.seed)
    rng_data, rng_mask = random.split(rng)
    y_full, latent_f, rate = gen_y_obs(rng_data, s, cfg.gt_ls, cfg.beta_true)
    obs_mask = gen_spatial_obs_mask(
        rng_mask, (cfg.grid_size, cfg.grid_size), cfg.obs_ratio
    )
    data = {
        "s": s,
        "latent_f": latent_f,
        "rate": rate,
        "y_full": y_full,
        "obs_mask": obs_mask,
        "gt_ls": cfg.gt_ls,
        "beta_true": cfg.beta_true,
        "kernel": "matern_1_2",
    }
    write_pickle(path, data)
    plot_truth_and_mask(seed_dir, cfg, data)
    return data


def inference_model(s: Array, priors: dict):
    def poisson(surrogate_decoder: Optional[Callable] = None, obs_mask=True, y=None):
        ls = numpyro.sample("ls", priors["ls"])
        beta = numpyro.sample("beta", priors["beta"])
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            kernel = matern_1_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.deterministic("mu", jnp.linalg.cholesky(kernel) @ z[0])
        else:
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), s=s).squeeze(),
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


def make_batch_generator(
    target_s: Array, sample_s: Array, priors: dict, batch_size: int
):
    jitter = 5e-4 * jnp.eye(sample_s.shape[0])
    interpolation = bilinear_interpolation_matrix(sample_s, target_s)
    latent_idx = nearest_target_indices(target_s, sample_s)

    @jit
    def generate(rng_data):
        rng_ls, rng_z = random.split(rng_data)
        ls = priors["ls"].sample(rng_ls)
        z_target = dist.Normal().sample(
            rng_z, sample_shape=(batch_size, target_s.shape[0])
        )
        z_sample = z_target[:, latent_idx]
        kernel = matern_1_2(sample_s, sample_s, 1.0, ls) + jitter
        f_sample = jnp.einsum(
            "ij,bj->bi", jnp.linalg.cholesky(kernel), z_sample
        )
        f_target = jnp.einsum("ts,bs->bt", interpolation, f_sample)
        return {
            "s": target_s,
            "z": z_target,
            "conditionals": jnp.array([ls]),
            "f": f_target,
        }

    return generate


def initialize_train_state(model, optimizer, batch, rng_init: Array) -> TrainState:
    rng_params, rng_extra = random.split(rng_init)
    variables = model.init(
        {"params": rng_params, "extra": rng_extra},
        **batch,
    )
    params = variables.pop("params")
    return TrainState.create(
        apply_fn=model.apply, params=params, kwargs=variables, tx=optimizer
    )


def checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"step_{step:08d}"


def save_training_checkpoint(
    checkpoint_dir: Path,
    state: TrainState,
    best_params,
    best_kwargs,
    best_metric: float,
    cumulative_gp_time: float,
    cumulative_optimization_time: float,
    cumulative_wall_time: float,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(checkpoint_dir, int(state.step))
    if path.exists():
        return path
    payload = {
        "params": state.params,
        "kwargs": state.kwargs,
        "opt_state": state.opt_state,
        "step": state.step,
        "best_params": best_params,
        "best_kwargs": best_kwargs,
        "best_metric": np.asarray(best_metric),
        "cumulative_gp_sample_generation_time": np.asarray(cumulative_gp_time),
        "cumulative_neural_optimization_time": np.asarray(
            cumulative_optimization_time
        ),
        "cumulative_train_wall_time": np.asarray(cumulative_wall_time),
    }
    checkpointer = PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(payload)
    checkpointer.save(path.absolute(), payload, save_args=save_args)
    write_json(
        checkpoint_dir / "latest.json",
        {"step": int(state.step), "path": path.name, "saved_at_utc": utc_now()},
    )
    return path


def restore_training_checkpoint(
    checkpoint_dir: Path, model, optimizer
) -> Optional[tuple]:
    latest_path = checkpoint_dir / "latest.json"
    if not latest_path.exists():
        return None
    latest = json.loads(latest_path.read_text())
    path = checkpoint_dir / latest["path"]
    payload = PyTreeCheckpointer().restore(path.absolute())
    state = TrainState.create(
        apply_fn=model.apply,
        params=payload["params"],
        kwargs=payload["kwargs"],
        tx=optimizer,
    )
    state = state.replace(
        step=payload["step"],
        opt_state=payload["opt_state"],
    )
    return (
        state,
        payload["best_params"],
        payload["best_kwargs"],
        float(payload["best_metric"]),
        float(payload["cumulative_gp_sample_generation_time"]),
        float(payload["cumulative_neural_optimization_time"]),
        float(payload.get("cumulative_train_wall_time", 0.0)),
    )


def append_training_history(path: Path, row: dict) -> None:
    existing = []
    if path.exists():
        with path.open(newline="") as f:
            existing = list(csv.DictReader(f))
    existing = [item for item in existing if int(item["step"]) != int(row["step"])]
    existing.append(row)
    existing.sort(key=lambda item: int(item["step"]))
    write_csv(path, existing)


def evaluate_model(
    cfg: Config,
    state: TrainState,
    generate_batch: Callable,
    base_key: Array,
) -> float:
    def loader(_):
        for idx in range(cfg.validation_batches):
            yield generate_batch(random.fold_in(base_key, idx))

    return float(
        evaluate(
            base_key,
            state,
            valid_step,
            loader,
            cfg.validation_batches,
        )["norm MSE"]
    )


def train_deeprv(
    cfg: Config,
    model_name: str,
    target_s: Array,
    sample_s: Array,
    priors: dict,
    model_dir: Path,
) -> tuple[Callable, dict]:
    model = gMLPDeepRV(num_blks=2)
    lr_schedule = cosine_annealing_lr(cfg.train_steps, cfg.lr)
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.yogi(lr_schedule),
    )
    generate_batch = make_batch_generator(
        target_s, sample_s, priors, cfg.batch_size
    )
    model_key = random.fold_in(random.key(cfg.seed), MODEL_IDS[model_name])
    init_key, train_key, valid_key = random.split(model_key, 3)
    init_batch = generate_batch(random.fold_in(train_key, 0))
    jax.block_until_ready(init_batch["f"])
    checkpoint_dir = model_dir / "checkpoints"
    restored = restore_training_checkpoint(checkpoint_dir, model, optimizer)
    if restored is None:
        state = initialize_train_state(model, optimizer, init_batch, init_key)
        best_params, best_kwargs = state.params, state.kwargs
        best_metric = float("inf")
        gp_time = 0.0
        optimization_time = 0.0
        previous_wall_time = 0.0
        print(f"{model_name}: starting training from step 0")
    else:
        (
            state,
            best_params,
            best_kwargs,
            best_metric,
            gp_time,
            optimization_time,
            previous_wall_time,
        ) = restored
        print(f"{model_name}: resumed training from step {int(state.step)}")

    history_path = model_dir / "training_history.csv"
    prior_result_path = model_dir / "training_result.json"
    if int(state.step) >= cfg.train_steps and prior_result_path.exists():
        print(f"{model_name}: training already complete; loading saved result")
        best_state = state.replace(params=best_params, kwargs=best_kwargs)
        return (
            generate_surrogate_decoder(best_state, model),
            json.loads(prior_result_path.read_text()),
        )

    wall_start = perf_counter()
    with MemoryMonitor() as memory:
        for step in range(int(state.step) + 1, cfg.train_steps + 1):
            batch_start = perf_counter()
            batch = generate_batch(random.fold_in(train_key, step))
            jax.block_until_ready(batch["f"])
            gp_time += perf_counter() - batch_start

            optimize_start = perf_counter()
            step_key = random.fold_in(train_key, cfg.train_steps + step)
            state, loss = deep_rv_train_step(step_key, state, batch)
            jax.block_until_ready((state.params, loss))
            optimization_time += perf_counter() - optimize_start

            should_validate = (
                step % cfg.validation_interval == 0 or step == cfg.train_steps
            )
            should_checkpoint = (
                step % cfg.checkpoint_save_interval == 0
                or step == cfg.train_steps
            )
            metric = None
            if should_validate:
                metric = evaluate_model(
                    cfg,
                    state,
                    generate_batch,
                    random.fold_in(valid_key, step),
                )
                if metric < best_metric:
                    best_metric = metric
                    best_params, best_kwargs = state.params, state.kwargs
            checkpoint = None
            if should_checkpoint:
                checkpoint = save_training_checkpoint(
                    checkpoint_dir,
                    state,
                    best_params,
                    best_kwargs,
                    best_metric,
                    gp_time,
                    optimization_time,
                    previous_wall_time + perf_counter() - wall_start,
                )
            if should_validate or should_checkpoint:
                row = {
                    "step": step,
                    "train_loss": float(loss),
                    "valid_norm_mse": metric,
                    "best_valid_norm_mse": best_metric,
                    "cumulative_gp_sample_generation_time": gp_time,
                    "cumulative_neural_optimization_time": optimization_time,
                    "checkpoint": None if checkpoint is None else str(checkpoint),
                    "saved_at_utc": utc_now(),
                }
                append_training_history(history_path, row)
                message = f"{model_name}: step={step} loss={float(loss):.6g}"
                if metric is not None:
                    message += f" valid={metric:.6g}"
                if checkpoint is not None:
                    message += f" checkpoint={checkpoint.name}"
                print(message)

    best_state = state.replace(params=best_params, kwargs=best_kwargs)
    result = {
        "train_time": previous_wall_time + perf_counter() - wall_start,
        "gp_sample_generation_time": gp_time,
        "neural_optimization_time": optimization_time,
        "train_norm_mse": best_metric,
        "training_steps_completed": int(state.step),
        "num_pretrain_locations": int(sample_s.shape[0]),
        "pretrain_domain_min": float(sample_s.min()),
        "pretrain_domain_max": float(sample_s.max()),
        "train_peak_host_rss_mb": memory.peak_rss_mb,
        "train_peak_gpu_memory_mb": memory.peak_gpu_mb,
    }
    write_json(model_dir / "training_result.json", result)
    return generate_surrogate_decoder(best_state, model), result


def run_hmc(
    cfg: Config,
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Array,
    surrogate_decoder: Optional[Callable] = None,
) -> tuple[dict, dict, float, dict, dict]:
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(
        nuts,
        num_chains=cfg.num_chains,
        num_samples=cfg.mcmc_samples,
        num_warmup=cfg.mcmc_warmup,
        progress_bar=True,
    )
    rng_run, rng_pred = random.split(rng)
    with MemoryMonitor() as memory:
        start = perf_counter()
        mcmc.run(
            rng_run,
            surrogate_decoder=surrogate_decoder,
            obs_mask=obs_mask,
            y=y_obs,
            extra_fields=("diverging",),
        )
        infer_time = perf_counter() - start
        samples_all = mcmc.get_samples()
        posterior = Predictive(model, samples_all)(
            rng_pred, surrogate_decoder=surrogate_decoder
        )
        jax.block_until_ready(posterior["obs"])
    samples_small = {key: samples_all[key] for key in ("ls", "beta")}
    try:
        ess_data = az.ess(mcmc, method="mean")
        ess = {
            key: float(ess_data[key].mean().item()) for key in ("ls", "beta")
        }
    except Exception as exc:
        print("ESS calculation unavailable:", repr(exc))
        ess = {"ls": None, "beta": None}
    extra = mcmc.get_extra_fields()
    diagnostics = {
        "num_divergences": int(np.asarray(extra.get("diverging", [])).sum()),
        "inference_peak_host_rss_mb": memory.peak_rss_mb,
        "inference_peak_gpu_memory_mb": memory.peak_gpu_mb,
    }
    posterior_small = {
        "obs": posterior["obs"],
        "mu": posterior.get("mu"),
    }
    return samples_small, posterior_small, infer_time, ess, diagnostics


def mse_values(y: Array, y_hat: Array, mask: Array) -> dict:
    sq = (y - y_hat) ** 2
    log_sq = (jnp.log1p(y) - jnp.log1p(y_hat)) ** 2
    return {
        "mse_all": float(sq.mean()),
        "mse_observed": float(sq[mask].mean()),
        "mse_unobserved": float(sq[~mask].mean()),
        "log1p_mse_all": float(log_sq.mean()),
        "log1p_mse_observed": float(log_sq[mask].mean()),
        "log1p_mse_unobserved": float(log_sq[~mask].mean()),
    }


def uncertainty_values(
    y: Array, posterior_obs: Array, mask: Array, level: float
) -> dict:
    alpha = 1.0 - level
    lo = jnp.quantile(posterior_obs, alpha / 2.0, axis=0)
    hi = jnp.quantile(posterior_obs, 1.0 - alpha / 2.0, axis=0)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    prefix = f"{int(level * 100)}"
    return {
        f"coverage_{prefix}_all": float(covered.mean()),
        f"coverage_{prefix}_observed": float(covered[mask].mean()),
        f"coverage_{prefix}_unobserved": float(covered[~mask].mean()),
        f"interval_width_{prefix}_all": float(width.mean()),
        f"interval_width_{prefix}_observed": float(width[mask].mean()),
        f"interval_width_{prefix}_unobserved": float(width[~mask].mean()),
    }


def summarize_model(
    cfg: Config,
    model_name: str,
    sample_s: Optional[Array],
    data: dict,
    samples: dict,
    posterior: dict,
    infer_time: float,
    ess: dict,
    diagnostics: dict,
    training: Optional[dict],
    full_gp_reference: Optional[tuple[dict, dict]],
) -> dict:
    y_full = data["y_full"]
    obs_mask = data["obs_mask"]
    y_hat = posterior["obs"].mean(axis=0)
    row = {
        "seed": cfg.seed,
        "model_name": model_name,
        "num_pretrain_locations": (
            None if sample_s is None else int(sample_s.shape[0])
        ),
        "pretrain_domain_min": (
            None if sample_s is None else float(sample_s.min())
        ),
        "pretrain_domain_max": (
            None if sample_s is None else float(sample_s.max())
        ),
        "posterior_mean_ls": float(samples["ls"].mean()),
        "posterior_mean_beta": float(samples["beta"].mean()),
        "ESS_ls": ess.get("ls"),
        "ESS_beta": ess.get("beta"),
        "ESS_ls_per_second": (
            None if ess.get("ls") is None else ess["ls"] / infer_time
        ),
        "ESS_beta_per_second": (
            None if ess.get("beta") is None else ess["beta"] / infer_time
        ),
        "infer_time": infer_time,
        **diagnostics,
    }
    row.update(mse_values(y_full, y_hat, obs_mask))
    row.update(
        uncertainty_values(
            y_full, posterior["obs"], obs_mask, cfg.coverage_level
        )
    )
    if training is not None:
        row.update(training)
    if full_gp_reference is not None:
        full_samples, full_posterior = full_gp_reference
        full_mean = full_posterior["obs"].mean(axis=0)
        row["posterior_mean_mse_vs_full_gp"] = float(
            jnp.mean((y_hat - full_mean) ** 2)
        )
        row["posterior_mean_log1p_mse_vs_full_gp"] = float(
            jnp.mean((jnp.log1p(y_hat) - jnp.log1p(full_mean)) ** 2)
        )
        row["ls_wasserstein_vs_full_gp"] = float(
            wasserstein_distance(
                np.asarray(full_samples["ls"]).reshape(-1),
                np.asarray(samples["ls"]).reshape(-1),
            )
        )
        row["beta_wasserstein_vs_full_gp"] = float(
            wasserstein_distance(
                np.asarray(full_samples["beta"]).reshape(-1),
                np.asarray(samples["beta"]).reshape(-1),
            )
        )
    return row


def save_model_outputs(
    model_dir: Path,
    samples: dict,
    posterior: dict,
    metrics: dict,
) -> None:
    write_pickle(model_dir / "posterior_samples.pkl", samples)
    np.savez_compressed(
        model_dir / "posterior_predictive.npz",
        obs=np.asarray(posterior["obs"]),
        mu=(
            np.asarray(posterior["mu"])
            if posterior.get("mu") is not None
            else np.empty((0,))
        ),
    )
    write_json(model_dir / "metrics.json", metrics)
    write_csv(model_dir / "metrics.csv", [metrics])
    write_json(
        model_dir / "complete.json",
        {"completed_at_utc": utc_now(), "expected_files_present": True},
    )


def load_model_outputs(model_dir: Path) -> Optional[tuple[dict, dict, dict]]:
    required = (
        model_dir / "complete.json",
        model_dir / "posterior_samples.pkl",
        model_dir / "posterior_predictive.npz",
        model_dir / "metrics.json",
    )
    if not all(path.exists() for path in required):
        return None
    with (model_dir / "posterior_samples.pkl").open("rb") as f:
        samples = pickle.load(f)
    predictive = np.load(model_dir / "posterior_predictive.npz")
    posterior = {
        "obs": jnp.asarray(predictive["obs"]),
        "mu": (
            None
            if predictive["mu"].size == 0
            else jnp.asarray(predictive["mu"])
        ),
    }
    metrics = json.loads((model_dir / "metrics.json").read_text())
    return samples, posterior, metrics


def plot_truth_and_mask(seed_dir: Path, cfg: Config, data: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fields = (
        (data["latent_f"], "latent truth"),
        (data["rate"], "Poisson rate"),
        (data["y_full"], "count truth"),
        (data["obs_mask"], "observation mask"),
    )
    for ax, (values, title) in zip(axes, fields):
        field = np.asarray(values).reshape(cfg.grid_size, cfg.grid_size)
        im = ax.imshow(field, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(seed_dir / "truth_count_and_mask.png", dpi=180)
    plt.close(fig)


def available_outputs(seed_dir: Path) -> dict[str, tuple[dict, dict, dict]]:
    outputs = {}
    for model_name in ALL_MODELS:
        loaded = load_model_outputs(seed_dir / model_name)
        if loaded is not None:
            outputs[model_name] = loaded
    return outputs


def plot_seed_comparisons(seed_dir: Path, cfg: Config, data: dict) -> None:
    outputs = available_outputs(seed_dir)
    if not outputs:
        return
    names = list(outputs)
    fig, axes = plt.subplots(
        1, len(names), figsize=(4 * len(names), 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        mean = outputs[name][1]["obs"].mean(axis=0)
        field = np.log1p(np.asarray(mean)).reshape(cfg.grid_size, cfg.grid_size)
        im = ax.imshow(field, origin="lower", cmap="viridis")
        ax.set_title(name)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(seed_dir / "posterior_predictive_means.png", dpi=180)
    plt.close(fig)

    if "full_gp" in outputs and len(outputs) > 1:
        full_mean = outputs["full_gp"][1]["obs"].mean(axis=0)
        compare_names = [name for name in names if name != "full_gp"]
        fig, axes = plt.subplots(
            1,
            len(compare_names),
            figsize=(4 * len(compare_names), 4),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        max_abs = max(
            float(
                jnp.max(
                    jnp.abs(outputs[name][1]["obs"].mean(axis=0) - full_mean)
                )
            )
            for name in compare_names
        )
        for ax, name in zip(axes, compare_names):
            difference = (
                outputs[name][1]["obs"].mean(axis=0) - full_mean
            ).reshape(cfg.grid_size, cfg.grid_size)
            im = ax.imshow(
                np.asarray(difference),
                origin="lower",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            ax.set_title(f"{name} - full_gp")
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, shrink=0.75)
        fig.savefig(
            seed_dir / "posterior_mean_differences_vs_full_gp.png", dpi=180
        )
        plt.close(fig)

    for parameter, truth in (("ls", cfg.gt_ls), ("beta", cfg.beta_true)):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for name in names:
            values = np.asarray(outputs[name][0][parameter]).reshape(-1)
            ax.hist(values, bins=40, alpha=0.4, density=True, label=name)
        ax.axvline(truth, color="black", linestyle="--", label=f"true {parameter}")
        ax.set_xlabel(parameter)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        fig.savefig(seed_dir / f"{parameter}_posteriors.png", dpi=180)
        plt.close(fig)

    level = int(cfg.coverage_level * 100)
    metrics = [outputs[name][2] for name in names]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for offset, subset in zip(
        (-width, 0.0, width), ("all", "observed", "unobserved")
    ):
        values = [row[f"coverage_{level}_{subset}"] for row in metrics]
        ax.bar(x + offset, values, width=width, label=subset)
    ax.axhline(cfg.coverage_level, color="black", linestyle="--")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"{level}% predictive coverage")
    ax.legend()
    fig.savefig(seed_dir / "coverage_comparison.png", dpi=180)
    plt.close(fig)

    cost_rows = [
        row
        for row in metrics
        if row.get("posterior_mean_mse_vs_full_gp") is not None
        and row.get("gp_sample_generation_time") is not None
    ]
    if cost_rows:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for row in cost_rows:
            cost = row["gp_sample_generation_time"] + row.get(
                "neural_optimization_time", 0.0
            )
            fidelity = row["posterior_mean_mse_vs_full_gp"]
            ax.scatter(cost, fidelity, s=60)
            ax.annotate(row["model_name"], (cost, fidelity), fontsize=8)
        ax.set_xlabel("measured pretraining time (seconds)")
        ax.set_ylabel("posterior mean MSE vs full_gp")
        fig.savefig(seed_dir / "cost_fidelity_comparison.png", dpi=180)
        plt.close(fig)


def refresh_seed_metrics(seed_dir: Path) -> list[dict]:
    rows = []
    for model_name in ALL_MODELS:
        path = seed_dir / model_name / "metrics.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    write_csv(seed_dir / "metrics.csv", rows)
    return rows


def aggregate_results(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.glob("seed_*/*/metrics.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        return
    write_csv(run_dir / "combined_raw_metrics.csv", rows)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    summary = []
    for model_name in ALL_MODELS:
        model_rows = [row for row in rows if row["model_name"] == model_name]
        if not model_rows:
            continue
        summary_row = {"model_name": model_name, "num_seeds": len(model_rows)}
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in model_rows
                if row.get(key) is not None and np.isfinite(float(row[key]))
            ]
            if values:
                summary_row[f"{key}_mean"] = float(np.mean(values))
                summary_row[f"{key}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
        summary.append(summary_row)
    write_csv(run_dir / "aggregate_metrics_by_model.csv", summary)
    plot_aggregate_results(run_dir, rows)


def plot_aggregate_results(run_dir: Path, rows: list[dict]) -> None:
    metrics = (
        "posterior_mean_mse_vs_full_gp",
        "ls_wasserstein_vs_full_gp",
        "coverage_90_unobserved",
        "infer_time",
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        names, means, errors = [], [], []
        for model_name in ALL_MODELS:
            values = [
                float(row[metric])
                for row in rows
                if row["model_name"] == model_name
                and row.get(metric) is not None
            ]
            if values:
                names.append(model_name)
                means.append(float(np.mean(values)))
                errors.append(
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
        if names:
            x = np.arange(len(names))
            ax.errorbar(x, means, yerr=errors, fmt="o", capsize=4)
            ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_title(metric)
    fig.savefig(run_dir / "aggregate_metrics_with_error_bars.png", dpi=180)
    plt.close(fig)

    deep_rows = [
        row
        for row in rows
        if row["model_name"] != "full_gp"
        and row.get("posterior_mean_mse_vs_full_gp") is not None
        and row.get("gp_sample_generation_time") is not None
    ]
    if deep_rows:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for model_name in ALL_MODELS[1:]:
            selected = [row for row in deep_rows if row["model_name"] == model_name]
            if not selected:
                continue
            costs = np.asarray(
                [
                    row["gp_sample_generation_time"]
                    + row.get("neural_optimization_time", 0.0)
                    for row in selected
                ]
            )
            errors = np.asarray(
                [row["posterior_mean_mse_vs_full_gp"] for row in selected]
            )
            ax.errorbar(
                costs.mean(),
                errors.mean(),
                xerr=costs.std(ddof=1) if len(costs) > 1 else 0.0,
                yerr=errors.std(ddof=1) if len(errors) > 1 else 0.0,
                fmt="o",
                capsize=4,
                label=model_name,
            )
        ax.set_xlabel("measured pretraining time (seconds)")
        ax.set_ylabel("posterior mean MSE vs full_gp")
        ax.legend(fontsize=8)
        fig.savefig(run_dir / "aggregate_cost_fidelity.png", dpi=180)
        plt.close(fig)


def run_one_model(
    cfg: Config,
    seed_dir: Path,
    model_name: str,
    data: dict,
    priors: dict,
    infer_model: Callable,
) -> None:
    model_dir = seed_dir / model_name
    if cfg.force_rerun:
        reset_output(seed_dir, model_name)
    existing = load_model_outputs(model_dir)
    if existing is not None:
        print(f"{model_name}: complete output exists; skipping")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        model_dir / "config.json",
        {**asdict(cfg), "model_name": model_name},
    )
    (model_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    write_json(model_dir / "environment.json", environment_info())

    full_reference = None
    if model_name != "full_gp":
        loaded = load_model_outputs(seed_dir / "full_gp")
        if loaded is None:
            raise RuntimeError(
                "A completed full_gp reference is required before running "
                f"{model_name}. Include full_gp in --models or run it first."
            )
        full_reference = (loaded[0], loaded[1])

    sample_s = None
    surrogate_decoder = None
    training = None
    if model_name != "full_gp":
        sample_s = pretrain_locations(model_name, data["s"], cfg.domain_stop)
        print(
            f"{model_name}: {sample_s.shape[0]} pretraining locations in "
            f"[{float(sample_s.min())}, {float(sample_s.max())}]"
        )
        surrogate_decoder, training = train_deeprv(
            cfg,
            model_name,
            data["s"],
            sample_s,
            priors,
            model_dir,
        )

    hmc_key = random.fold_in(
        random.key(cfg.seed + 1_000_000), MODEL_IDS[model_name]
    )
    print(f"{model_name}: running MCMC")
    samples, posterior, infer_time, ess, diagnostics = run_hmc(
        cfg,
        hmc_key,
        infer_model,
        data["y_full"],
        data["obs_mask"],
        surrogate_decoder,
    )
    metrics = summarize_model(
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
    save_model_outputs(model_dir, samples, posterior, metrics)
    print(f"{model_name}: complete")


def main() -> None:
    cfg = parse_args()
    numpyro.set_host_device_count(cfg.num_chains)
    wandb.init(mode="disabled")
    run_dir, seed_dir = prepare_seed_dir(cfg)
    log_file = (seed_dir / "console.log").open("a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print("=" * 80)
    print("Started:", utc_now())
    print("Output:", seed_dir)
    print("Config:", json.dumps(asdict(cfg), indent=2))
    print("JAX devices:", jax.devices())

    s = make_grid(cfg.grid_size, 0.0, cfg.domain_stop)
    data = load_or_create_data(cfg, seed_dir, s)
    priors = {
        "ls": dist.LogNormal(loc=cfg.prior_loc, scale=cfg.prior_scale),
        "beta": dist.Normal(0.0, 1.0),
    }
    infer_model = inference_model(s, priors)

    for model_name in cfg.models:
        run_one_model(cfg, seed_dir, model_name, data, priors, infer_model)
        refresh_seed_metrics(seed_dir)
        plot_seed_comparisons(seed_dir, cfg, data)
        aggregate_results(run_dir)

    print("Finished:", utc_now())
    print("Seed metrics:", seed_dir / "metrics.csv")
    print("Combined metrics:", run_dir / "combined_raw_metrics.csv")
    print("Aggregate metrics:", run_dir / "aggregate_metrics_by_model.csv")


if __name__ == "__main__":
    main()
