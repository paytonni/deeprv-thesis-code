#!/usr/bin/env python3
"""64x64 synthetic Poisson GP comparison.

The experiment compares Full GP MCMC, Exact DeepRV and inducing constructions
with two chains, 1,000 warmup iterations and 4,000 retained draws per chain.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
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


MODELS = (
    "full_gp",
    "deeprv_exact",
    "deeprv_lowres_8x8",
    "deeprv_lowres_12x12",
    "deeprv_lowres_16x16",
    "deeprv_lowres_24x24",
    "deeprv_local_8x8",
    "deeprv_local_16x16",
    "deeprv_local_32x32",
)
BUDGETS = ("initial", "extended")
MODEL_IDS = {name: index for index, name in enumerate(MODELS)}
BUDGET_IDS = {name: index for index, name in enumerate(BUDGETS)}
PRETRAIN_GRID_SIZES = {
    "deeprv_lowres_8x8": 8,
    "deeprv_lowres_12x12": 12,
    "deeprv_lowres_16x16": 16,
    "deeprv_lowres_24x24": 24,
}
LOCAL_GRID_SIZES = {
    "deeprv_local_8x8": 8,
    "deeprv_local_16x16": 16,
    "deeprv_local_32x32": 32,
}


@dataclass
class Config:
    seed: int = 0
    grid_size: int = 32
    domain_stop: float = 100.0
    gt_ls: float = 30.0
    obs_ratio: float = 0.5
    obs_mask_type: str = "spatial"
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
    checkpoint_interval: int = 10_000
    initial_warmup: int = 1_000
    initial_samples: int = 4_000
    extended_warmup: int = 1_000
    extended_samples: int = 4_000
    max_rhat: float = 1.01
    min_bulk_ess: float = 400.0
    min_tail_ess: float = 400.0
    max_relative_mcse: float = 0.05
    local_region_width: float = 0.5
    output_root: str = "outputs/deeprv_32x32_poisson_gp_pilot"
    run_name: str = "poisson_gp_32x32_ls30_pilot"
    models: tuple[str, ...] = (
        "full_gp",
        "deeprv_exact",
        "deeprv_lowres_8x8",
    )
    auto_extend_failed: bool = True
    prepare_data_only: bool = False
    force_rerun: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        "--data-mask-seed",
        dest="seed",
        type=int,
        choices=(0, 1, 2),
        default=Config.seed,
        help=(
            "Data/mask replicate seed. This controls the synthetic target field, "
            "Poisson observations, observation mask, and posterior inference RNG."
        ),
    )
    parser.add_argument("--grid-size", type=int, default=Config.grid_size)
    parser.add_argument("--domain-stop", type=float, default=Config.domain_stop)
    parser.add_argument("--gt-ls", type=float, default=Config.gt_ls)
    parser.add_argument("--obs-ratio", type=float, default=Config.obs_ratio)
    parser.add_argument(
        "--obs-mask-type",
        choices=("spatial", "uniform"),
        default=Config.obs_mask_type,
        help="Observation mask design: spatial blob/ellipse mask or uniform random mask.",
    )
    parser.add_argument("--train-steps", type=int, default=Config.train_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--valid-steps", type=int, default=Config.valid_steps)
    parser.add_argument("--num-chains", type=int, default=Config.num_chains)
    parser.add_argument(
        "--initial-warmup", type=int, default=Config.initial_warmup
    )
    parser.add_argument(
        "--initial-samples", type=int, default=Config.initial_samples
    )
    parser.add_argument(
        "--extended-warmup", type=int, default=Config.extended_warmup
    )
    parser.add_argument(
        "--extended-samples", type=int, default=Config.extended_samples
    )
    parser.add_argument("--max-rhat", type=float, default=Config.max_rhat)
    parser.add_argument("--min-bulk-ess", type=float, default=Config.min_bulk_ess)
    parser.add_argument("--min-tail-ess", type=float, default=Config.min_tail_ess)
    parser.add_argument(
        "--max-relative-mcse", type=float, default=Config.max_relative_mcse
    )
    parser.add_argument(
        "--local-region-width", type=float, default=Config.local_region_width
    )
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
        "--models", nargs="+", choices=MODELS, default=list(Config.models)
    )
    parser.add_argument(
        "--no-auto-extend-failed",
        action="store_false",
        dest="auto_extend_failed",
    )
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    args.models = tuple(dict.fromkeys(args.models))
    args.mcmc_warmup = args.extended_warmup
    args.mcmc_samples = args.extended_samples
    cfg = Config(**vars(args))
    if cfg.grid_size != 32:
        raise ValueError("This pilot requires --grid-size 32.")
    if cfg.num_chains < 2:
        raise ValueError("Use at least two chains for R-hat diagnostics.")
    if min(
        cfg.train_steps,
        cfg.valid_steps,
        cfg.initial_warmup,
        cfg.initial_samples,
        cfg.extended_warmup,
        cfg.extended_samples,
        cfg.checkpoint_interval,
    ) < 1:
        raise ValueError("Training, MCMC, and checkpoint values must be positive.")
    if min(
        cfg.max_rhat,
        cfg.min_bulk_ess,
        cfg.min_tail_ess,
        cfg.max_relative_mcse,
    ) <= 0:
        raise ValueError("Diagnostic thresholds must be positive.")
    if not 0.0 < cfg.local_region_width <= 1.0:
        raise ValueError("--local-region-width must be in (0, 1].")
    if "full_gp" not in cfg.models:
        raise ValueError("Include full_gp as the posterior reference.")
    if not cfg.prepare_data_only and "deeprv_exact" not in cfg.models:
        raise ValueError("Include deeprv_exact to separate approximation error.")
    return cfg


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    temporary.replace(path)


def write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        pickle.dump(payload, f)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
    temporary.replace(path)


class Tee:
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, data):
        alive = []
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
                alive.append(stream)
            except OSError:
                # Colab Drive streams can disappear during long progress-bar
                # updates. Keep the run alive and continue logging elsewhere.
                continue
        self.streams = alive
        return len(data)

    def flush(self):
        alive = []
        for stream in self.streams:
            try:
                stream.flush()
                alive.append(stream)
            except OSError:
                continue
        self.streams = alive


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


def make_grid(grid_size: int, domain_start: float, domain_stop: float) -> Array:
    return build_grid(
        [{"start": domain_start, "stop": domain_stop, "num": grid_size}] * 2
    ).reshape(-1, 2)


def gen_spatial_obs_mask(
    rng: Array, grid_shape: tuple[int, int], obs_ratio: float
) -> Array:
    height, width = grid_shape
    total = height * width
    target = int(obs_ratio * total)
    mask = jnp.zeros((height, width), dtype=bool)
    yy, xx = jnp.meshgrid(
        jnp.arange(height), jnp.arange(width), indexing="ij"
    )
    tries = 0
    while int(mask.sum()) < target and tries < 200:
        tries += 1
        rng, rng_blob = random.split(rng)
        rngs = random.split(rng_blob, 4)
        center_x = random.randint(rngs[0], (), 0, height)
        center_y = random.randint(rngs[1], (), 0, width)
        radius_x = random.randint(
            rngs[2], (), max(1, height // 8), max(2, height // 3 + 1)
        )
        radius_y = random.randint(
            rngs[3], (), max(1, width // 8), max(2, width // 3 + 1)
        )
        ellipse = (
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
            <= 1.0
        )
        mask = jnp.logical_or(mask, ellipse)
    flat = mask.flatten()
    if int(flat.sum()) < target:
        missing = target - int(flat.sum())
        available = jnp.argwhere(~flat).reshape(-1)
        rng, rng_fill = random.split(rng)
        chosen = random.choice(
            rng_fill, available, shape=(missing,), replace=False
        )
        flat = flat.at[chosen].set(True)
    elif int(flat.sum()) > target:
        present = jnp.argwhere(flat).reshape(-1)
        rng, rng_trim = random.split(rng)
        chosen = random.choice(
            rng_trim, present, shape=(target,), replace=False
        )
        flat = jnp.zeros(total, dtype=bool).at[chosen].set(True)
    return flat


def gen_uniform_obs_mask(
    rng: Array, grid_shape: tuple[int, int], obs_ratio: float
) -> Array:
    height, width = grid_shape
    total = height * width
    target = int(obs_ratio * total)
    chosen = random.choice(rng, total, shape=(target,), replace=False)
    return jnp.zeros(total, dtype=bool).at[chosen].set(True)


def gen_obs_mask(
    rng: Array, grid_shape: tuple[int, int], obs_ratio: float, mask_type: str
) -> Array:
    if mask_type == "spatial":
        return gen_spatial_obs_mask(rng, grid_shape, obs_ratio)
    if mask_type == "uniform":
        return gen_uniform_obs_mask(rng, grid_shape, obs_ratio)
    raise ValueError(f"Unknown observation mask type: {mask_type}")


def bilinear_interpolation_matrix(source_s: Array, target_s: Array) -> Array:
    source = np.asarray(source_s)
    target = np.asarray(target_s)
    x_axis = np.unique(source[:, 0])
    y_axis = np.unique(source[:, 1])
    index_by_xy = {
        (float(x), float(y)): index for index, (x, y) in enumerate(source)
    }
    weights = np.zeros((target.shape[0], source.shape[0]), dtype=np.float32)
    for row, (x_raw, y_raw) in enumerate(target):
        x = float(np.clip(x_raw, x_axis[0], x_axis[-1]))
        y = float(np.clip(y_raw, y_axis[0], y_axis[-1]))
        ix0 = int(
            np.clip(
                np.searchsorted(x_axis, x, side="right") - 1,
                0,
                len(x_axis) - 2,
            )
        )
        iy0 = int(
            np.clip(
                np.searchsorted(y_axis, y, side="right") - 1,
                0,
                len(y_axis) - 2,
            )
        )
        ix1, iy1 = ix0 + 1, iy0 + 1
        x0, x1 = x_axis[ix0], x_axis[ix1]
        y0, y1 = y_axis[iy0], y_axis[iy1]
        wx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        wy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        for ix, iy, weight in (
            (ix0, iy0, (1.0 - wx) * (1.0 - wy)),
            (ix1, iy0, wx * (1.0 - wy)),
            (ix0, iy1, (1.0 - wx) * wy),
            (ix1, iy1, wx * wy),
        ):
            weights[row, index_by_xy[(float(x_axis[ix]), float(y_axis[iy]))]] += (
                weight
            )
    return jnp.asarray(weights)


def nearest_target_indices(target_s: Array, sample_s: Array) -> Array:
    squared_distance = jnp.sum(
        (sample_s[:, None, :] - target_s[None, :, :]) ** 2, axis=-1
    )
    return jnp.argmin(squared_distance, axis=1)


def gen_y_obs(rng: Array, s: Array, gt_ls: float, beta_true: float):
    rng_mu, rng_poisson = random.split(rng)
    kernel = matern_1_2(s, s, 1.0, gt_ls) + 5e-4 * jnp.eye(s.shape[0])
    latent_f = dist.MultivariateNormal(jnp.zeros(s.shape[0]), kernel).sample(
        rng_mu
    )
    rate = jnp.exp(beta_true + latent_f)
    y_full = dist.Poisson(rate=rate).sample(rng_poisson)
    return y_full, latent_f, rate


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
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(jnp.exp(beta + mu)), obs=y)

    return poisson


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"extra": rng},
    )
    return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}


def make_batch_generator(
    target_s: Array, sample_s: Array, priors: dict, batch_size: int
):
    jitter = 5e-4 * jnp.eye(sample_s.shape[0])
    interpolation = bilinear_interpolation_matrix(sample_s, target_s)
    latent_indices = nearest_target_indices(target_s, sample_s)

    @jit
    def generate(rng_data):
        rng_ls, rng_z = random.split(rng_data)
        ls = priors["ls"].sample(rng_ls)
        z_target = dist.Normal().sample(
            rng_z, sample_shape=(batch_size, target_s.shape[0])
        )
        z_sample = z_target[:, latent_indices]
        kernel = matern_1_2(sample_s, sample_s, 1.0, ls) + jitter
        f_sample = jnp.einsum(
            "ij,bj->bi", jnp.linalg.cholesky(kernel), z_sample
        )
        return {
            "s": target_s,
            "z": z_target,
            "conditionals": jnp.array([ls]),
            "f": jnp.einsum("ts,bs->bt", interpolation, f_sample),
        }

    return generate


def initialize_train_state(model, optimizer, batch, rng_init: Array) -> TrainState:
    rng_parameters, rng_extra = random.split(rng_init)
    variables = model.init(
        {"params": rng_parameters, "extra": rng_extra}, **batch
    )
    parameters = variables.pop("params")
    return TrainState.create(
        apply_fn=model.apply,
        params=parameters,
        kwargs=variables,
        tx=optimizer,
    )


def save_training_checkpoint(
    checkpoint_dir: Path,
    state: TrainState,
    best_params,
    best_kwargs,
    best_metric: float,
    train_time: float,
    gp_sample_generation_time: float,
    neural_optimization_time: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{int(state.step):08d}"
    if not path.exists():
        payload = {
            "params": state.params,
            "kwargs": state.kwargs,
            "opt_state": state.opt_state,
            "step": state.step,
            "best_params": best_params,
            "best_kwargs": best_kwargs,
            "best_metric": np.asarray(best_metric),
            "train_time": np.asarray(train_time),
            "gp_sample_generation_time": np.asarray(
                gp_sample_generation_time
            ),
            "neural_optimization_time": np.asarray(
                neural_optimization_time
            ),
        }
        checkpointer = PyTreeCheckpointer()
        checkpointer.save(
            path.absolute(),
            payload,
            save_args=orbax_utils.save_args_from_target(payload),
        )
    write_json(
        checkpoint_dir / "latest.json",
        {"step": int(state.step), "path": path.name, "saved_at_utc": utc_now()},
    )
    prune_training_checkpoints(checkpoint_dir, path.name)


def prune_training_checkpoints(checkpoint_dir: Path, latest_name: str) -> None:
    """Prune old Orbax checkpoints when explicitly enabled by the runner.

    Checkpoint pruning is enabled only through an explicit environment setting.
    """
    try:
        keep = int(os.environ.get("DEEPRV_CHECKPOINTS_TO_KEEP", "0"))
    except ValueError as error:
        raise ValueError("DEEPRV_CHECKPOINTS_TO_KEEP must be an integer.") from error
    if keep <= 0:
        return
    checkpoints = sorted(
        (
            path
            for path in checkpoint_dir.glob("step_*")
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    retained = {latest_name}
    for path in checkpoints:
        if len(retained) >= keep:
            break
        retained.add(path.name)
    for path in checkpoints:
        if path.name not in retained:
            shutil.rmtree(path)


def restore_container_structure(template, saved):
    """Rebuild tuple/named-tuple containers flattened by Orbax restore.

    Depending on the Orbax version, an Optax state saved as nested named
    tuples can be returned as lists, numerically keyed mappings, or mappings
    keyed by named-tuple field.  Array leaves are already restored correctly;
    only their Python container types need to follow the current optimizer
    template.
    """
    if isinstance(template, tuple):
        is_named_tuple = hasattr(template, "_fields")
        fields = getattr(template, "_fields", ())
        if len(template) == 0 and saved is None:
            return type(template)() if is_named_tuple else ()
        if isinstance(saved, Mapping):
            saved_values = []
            for index in range(len(template)):
                candidates = (str(index), index)
                if fields:
                    candidates = (*candidates, fields[index])
                for key in candidates:
                    if key in saved:
                        saved_values.append(saved[key])
                        break
                else:
                    raise KeyError(
                        f"Missing optimizer-state entry {index}; "
                        f"available keys are {tuple(saved.keys())}."
                    )
        elif isinstance(saved, (list, tuple)):
            saved_values = list(saved)
        else:
            raise TypeError(
                f"Expected a sequence or mapping for {type(template).__name__}, "
                f"got {type(saved).__name__}."
            )
        if len(saved_values) != len(template):
            raise ValueError(
                f"Optimizer-state length changed: expected {len(template)}, "
                f"got {len(saved_values)}."
            )
        values = [
            restore_container_structure(child, value)
            for child, value in zip(template, saved_values)
        ]
        if is_named_tuple:
            return type(template)(*values)
        return tuple(values)
    if isinstance(template, list):
        if isinstance(saved, Mapping):
            saved = [saved[str(index)] for index in range(len(template))]
        if not isinstance(saved, (list, tuple)) or len(saved) != len(template):
            raise ValueError("Saved optimizer list does not match its template.")
        return [
            restore_container_structure(child, value)
            for child, value in zip(template, saved)
        ]
    if isinstance(template, Mapping):
        if not isinstance(saved, Mapping):
            raise TypeError("Saved optimizer mapping does not match its template.")
        values = {
            key: restore_container_structure(template[key], saved[key])
            for key in template
        }
        try:
            return type(template)(values)
        except TypeError:
            return values
    return saved


def restore_training_checkpoint(checkpoint_dir: Path, model, optimizer):
    latest_path = checkpoint_dir / "latest.json"
    checkpoint_paths = []
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text())
            pointed = checkpoint_dir / latest["path"]
            if pointed.is_dir():
                checkpoint_paths.append(pointed)
            else:
                print(f"Checkpoint pointer target is missing: {pointed}")
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            print(f"Ignoring invalid checkpoint pointer {latest_path}: {error!r}")
    for path in sorted(checkpoint_dir.glob("step_*"), reverse=True):
        if path.is_dir() and path not in checkpoint_paths:
            checkpoint_paths.append(path)
    if not checkpoint_paths:
        return None
    failures = []
    payload = None
    restored_path = None
    for path in checkpoint_paths:
        try:
            payload = PyTreeCheckpointer().restore(path.absolute())
            restored_path = path
            break
        except Exception as error:  # Orbax errors vary by storage failure mode.
            failures.append(f"{path.name}: {error!r}")
            print(f"Could not restore {path}; trying an older checkpoint.")
    if payload is None or restored_path is None:
        raise RuntimeError(
            f"No valid checkpoint could be restored from {checkpoint_dir}: "
            + "; ".join(failures)
        )
    # The payload is in memory, so pruning older checkpoints is safe even when
    # Drive was too full to finish writing latest.json during the prior run.
    prune_training_checkpoints(checkpoint_dir, restored_path.name)
    write_json(
        latest_path,
        {
            "step": int(payload["step"]),
            "path": restored_path.name,
            "saved_at_utc": utc_now(),
        },
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=payload["params"],
        kwargs=payload["kwargs"],
        tx=optimizer,
    )
    try:
        restored_opt_state = restore_container_structure(
            state.opt_state,
            payload["opt_state"],
        )
    except Exception as error:
        raise RuntimeError(
            f"Checkpoint {restored_path} contains an optimizer state that "
            "cannot be mapped to the current Optax structure."
        ) from error
    state = state.replace(step=payload["step"], opt_state=restored_opt_state)
    return (
        state,
        payload["best_params"],
        payload["best_kwargs"],
        float(payload["best_metric"]),
        float(payload.get("train_time", 0.0)),
        float(payload.get("gp_sample_generation_time", 0.0)),
        float(payload.get("neural_optimization_time", 0.0)),
    )


def evaluate_model(
    cfg: Config, state: TrainState, generate_batch: Callable, base_key: Array
) -> float:
    def loader(_):
        for index in range(cfg.valid_steps):
            yield generate_batch(random.fold_in(base_key, index))

    return float(
        evaluate(base_key, state, valid_step, loader, cfg.valid_steps)["norm MSE"]
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
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.yogi(cosine_annealing_lr(cfg.train_steps, cfg.lr)),
    )
    generate_batch = make_batch_generator(
        target_s, sample_s, priors, cfg.batch_size
    )
    model_key = random.fold_in(random.key(cfg.seed), MODEL_IDS[model_name])
    init_key, train_key, valid_key = random.split(model_key, 3)
    init_batch = generate_batch(random.fold_in(train_key, 0))
    jax.block_until_ready(init_batch["f"])
    checkpoint_dir = model_dir / "checkpoints"
    result_path = model_dir / "training_result.json"
    restored = restore_training_checkpoint(checkpoint_dir, model, optimizer)
    if restored is None:
        state = initialize_train_state(model, optimizer, init_batch, init_key)
        best_params, best_kwargs = state.params, state.kwargs
        best_metric = float("inf")
        previous_time = 0.0
        gp_time = 0.0
        optimization_time = 0.0
        print(f"{model_name}: starting training from step 0")
    else:
        (
            state,
            best_params,
            best_kwargs,
            best_metric,
            previous_time,
            gp_time,
            optimization_time,
        ) = restored
        if result_path.exists():
            prior_result = json.loads(result_path.read_text())
            gp_time = max(
                gp_time,
                float(prior_result.get("gp_sample_generation_time", 0.0)),
            )
            optimization_time = max(
                optimization_time,
                float(prior_result.get("neural_optimization_time", 0.0)),
            )
        print(f"{model_name}: resumed training from step {int(state.step)}")
    if int(state.step) >= cfg.train_steps and result_path.exists():
        best_state = state.replace(params=best_params, kwargs=best_kwargs)
        return (
            generate_surrogate_decoder(best_state, model),
            json.loads(result_path.read_text()),
        )
    start = perf_counter()
    for step in range(int(state.step) + 1, cfg.train_steps + 1):
        gp_start = perf_counter()
        batch = generate_batch(random.fold_in(train_key, step))
        jax.block_until_ready(batch["f"])
        gp_time += perf_counter() - gp_start
        optimization_start = perf_counter()
        state, loss = deep_rv_train_step(
            random.fold_in(train_key, cfg.train_steps + step), state, batch
        )
        jax.block_until_ready((state.params, loss))
        optimization_time += perf_counter() - optimization_start
        if step % cfg.checkpoint_interval == 0 or step == cfg.train_steps:
            metric = evaluate_model(
                cfg, state, generate_batch, random.fold_in(valid_key, step)
            )
            if metric < best_metric:
                best_metric = metric
                best_params, best_kwargs = state.params, state.kwargs
            elapsed = previous_time + perf_counter() - start
            save_training_checkpoint(
                checkpoint_dir,
                state,
                best_params,
                best_kwargs,
                best_metric,
                elapsed,
                gp_time,
                optimization_time,
            )
            print(
                f"{model_name}: step={step} loss={float(loss):.6g} "
                f"valid={metric:.6g}"
            )
    result = {
        "train_time": previous_time + perf_counter() - start,
        "gp_sample_generation_time": gp_time,
        "neural_optimization_time": optimization_time,
        "train_norm_mse": best_metric,
        "training_steps_completed": int(state.step),
        "num_pretrain_locations": int(sample_s.shape[0]),
        "pretrain_domain_min": float(sample_s.min()),
        "pretrain_domain_max": float(sample_s.max()),
    }
    write_json(result_path, result)
    best_state = state.replace(params=best_params, kwargs=best_kwargs)
    return generate_surrogate_decoder(best_state, model), result


def mse_values(y: Array, y_hat: Array, mask: Array) -> dict:
    squared = (y - y_hat) ** 2
    log_squared = (jnp.log1p(y) - jnp.log1p(y_hat)) ** 2
    return {
        "mse_all": float(squared.mean()),
        "mse_observed": float(squared[mask].mean()),
        "mse_unobserved": float(squared[~mask].mean()),
        "log1p_mse_all": float(log_squared.mean()),
        "log1p_mse_observed": float(log_squared[mask].mean()),
        "log1p_mse_unobserved": float(log_squared[~mask].mean()),
    }


def uncertainty_values(
    y: Array, posterior_obs: Array, mask: Array, level: float
) -> dict:
    alpha = 1.0 - level
    lower = jnp.quantile(posterior_obs, alpha / 2.0, axis=0)
    upper = jnp.quantile(posterior_obs, 1.0 - alpha / 2.0, axis=0)
    covered = (y >= lower) & (y <= upper)
    prefix = int(level * 100)
    return {
        f"coverage_{prefix}_all": float(covered.mean()),
        f"coverage_{prefix}_observed": float(covered[mask].mean()),
        f"coverage_{prefix}_unobserved": float(covered[~mask].mean()),
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
        "posterior_mean_ls": float(samples["ls"].mean()),
        "posterior_mean_beta": float(samples["beta"].mean()),
        "ESS_ls": ess.get("ls"),
        "ESS_beta": ess.get("beta"),
        "infer_time": infer_time,
        "nuts_inference_time": infer_time,
        "posterior_inference_time": infer_time,
        **diagnostics,
    }
    row.update(mse_values(y_full, y_hat, obs_mask))
    row.update(
        uncertainty_values(y_full, posterior["obs"], obs_mask, cfg.coverage_level)
    )
    if training is not None:
        row.update(training)
        pretrain_total = float(training.get("train_time", 0.0))
        gp_generation = float(training.get("gp_sample_generation_time", 0.0))
        neural_optim = float(training.get("neural_optimization_time", 0.0))
        row.update(
            {
                "pretraining_total_time": pretrain_total,
                "pretraining_gp_field_generation_time": gp_generation,
                "pretraining_neural_optimization_time": neural_optim,
                "pretraining_other_time": pretrain_total
                - gp_generation
                - neural_optim,
                "total_time_with_pretraining": pretrain_total + infer_time,
            }
        )
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


def plot_truth_and_mask(seed_dir: Path, cfg: Config, data: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, (values, title) in zip(
        axes,
        (
            (data["latent_f"], "latent truth"),
            (data["rate"], "Poisson rate"),
            (data["y_full"], "count truth"),
            (data["obs_mask"], "observation mask"),
        ),
    ):
        image = ax.imshow(
            np.asarray(values).reshape(cfg.grid_size, cfg.grid_size),
            origin="lower",
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.savefig(seed_dir / "truth_count_and_mask.png", dpi=180)
    plt.close(fig)
    for values, title, filename, cmap in (
        (data["latent_f"], "32x32 latent field", "latent_field.png", "coolwarm"),
        (data["y_full"], "32x32 count field", "count_field.png", "viridis"),
        (
            data["obs_mask"],
            "32x32 observation mask",
            "observation_mask.png",
            "gray_r",
        ),
    ):
        fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
        image = ax.imshow(
            np.asarray(values).reshape(cfg.grid_size, cfg.grid_size),
            origin="lower",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("grid column")
        ax.set_ylabel("grid row")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.savefig(seed_dir / filename, dpi=180)
        plt.close(fig)


def normalized_config(cfg: Config) -> dict:
    values = asdict(cfg)
    for key in (
        "seed",
        "force_rerun",
        "output_root",
        "prepare_data_only",
        "models",
    ):
        values.pop(key, None)
    return json.loads(json.dumps(values))


def prepare_dirs(cfg: Config) -> tuple[Path, Path]:
    run_dir = Path(cfg.output_root).expanduser() / cfg.run_name
    seed_dir = run_dir / f"seed_{cfg.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "experiment_config.json"
    current = normalized_config(cfg)
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        previous.pop("models", None)
        previous.setdefault("local_region_width", cfg.local_region_width)
        if previous != current:
            raise ValueError(
                f"Configuration mismatch in {run_dir}; use a new run name."
            )
    write_json(manifest, current)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    environment_path = seed_dir / "environment.json"
    if environment_path.exists():
        environment_path = seed_dir / f"environment_{timestamp}.json"
    write_json(environment_path, environment_info())
    command = shlex.join(sys.argv) + "\n"
    command_path = seed_dir / "command.txt"
    if command_path.exists() and command_path.read_text() != command:
        command_path = seed_dir / f"command_{timestamp}.txt"
    if not command_path.exists():
        command_path.write_text(command)
    return run_dir, seed_dir


def reset_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def pretrain_locations(cfg: Config, target_s: Array, model_name: str) -> Array:
    if model_name == "deeprv_exact":
        return target_s
    if model_name in PRETRAIN_GRID_SIZES:
        return make_grid(
            PRETRAIN_GRID_SIZES[model_name], 0.0, cfg.domain_stop
        )
    if model_name in LOCAL_GRID_SIZES:
        center = cfg.domain_stop / 2.0
        half_width = cfg.domain_stop * cfg.local_region_width / 2.0
        return make_grid(
            LOCAL_GRID_SIZES[model_name],
            center - half_width,
            center + half_width,
        )
    raise ValueError(f"No pretraining locations for {model_name}.")


def load_or_create_data(cfg: Config, seed_dir: Path, target_s: Array) -> dict:
    path = seed_dir / "observed_data.pkl"
    if path.exists():
        with path.open("rb") as f:
            data = pickle.load(f)
        saved_mask_type = data.get("obs_mask_type", "spatial")
        if saved_mask_type != cfg.obs_mask_type:
            raise ValueError(
                f"{path} contains obs_mask_type={saved_mask_type!r}, but the "
                f"current config requests {cfg.obs_mask_type!r}. Use a new "
                "run name before rerunning."
            )
        return data
    rng_data, rng_mask = random.split(random.key(cfg.seed))
    y_full, latent_f, rate = gen_y_obs(
        rng_data, target_s, cfg.gt_ls, cfg.beta_true
    )
    obs_mask = gen_obs_mask(
        rng_mask, (cfg.grid_size, cfg.grid_size), cfg.obs_ratio, cfg.obs_mask_type
    )
    data = {
        "s": target_s,
        "latent_f": latent_f,
        "rate": rate,
        "y_full": y_full,
        "obs_mask": obs_mask,
        "obs_mask_type": cfg.obs_mask_type,
        "obs_ratio": cfg.obs_ratio,
        "gt_ls": cfg.gt_ls,
        "beta_true": cfg.beta_true,
        "kernel": "matern_1_2",
    }
    write_pickle(path, data)
    plot_truth_and_mask(seed_dir, cfg, data)
    return data


def train_models(
    cfg: Config,
    seed_dir: Path,
    target_s: Array,
    priors: dict,
) -> dict[str, tuple[Callable, dict, Array]]:
    trained = {}
    for model_name in cfg.models:
        if model_name == "full_gp":
            continue
        model_dir = seed_dir / "training" / model_name
        if cfg.force_rerun:
            reset_output(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        sample_s = pretrain_locations(cfg, target_s, model_name)
        model_config_path = model_dir / "config.json"
        if not model_config_path.exists():
            write_json(
                model_config_path,
                {
                    **asdict(cfg),
                    "model_name": model_name,
                    "num_pretrain_locations": int(sample_s.shape[0]),
                },
            )
        decoder, result = train_deeprv(
            cfg,
            model_name,
            target_s,
            sample_s,
            priors,
            model_dir,
        )
        trained[model_name] = (decoder, result, sample_s)
    return trained


def budget_values(cfg: Config, budget: str) -> tuple[int, int]:
    if budget == "initial":
        return cfg.initial_warmup, cfg.initial_samples
    if budget == "extended":
        return cfg.extended_warmup, cfg.extended_samples
    raise ValueError(f"Unknown budget: {budget}")


def scalar_stat(dataset, name: str, parameter: str) -> Optional[float]:
    try:
        return float(dataset[parameter].mean().item())
    except Exception:
        return None


def lag1_autocorrelation(values: np.ndarray) -> float:
    if values.size < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def identify_worst_chain(samples_chain: dict[str, Array]) -> dict:
    candidates = []
    for parameter in ("ls", "beta"):
        values = np.asarray(samples_chain[parameter])
        for chain in range(values.shape[0]):
            autocorr = lag1_autocorrelation(values[chain])
            if np.isfinite(autocorr):
                candidates.append(
                    {
                        "parameter": parameter,
                        "chain": chain,
                        "lag1_autocorrelation": autocorr,
                        "absolute_lag1_autocorrelation": abs(autocorr),
                    }
                )
    if not candidates:
        return {
            "parameter": None,
            "chain": 0,
            "lag1_autocorrelation": None,
            "absolute_lag1_autocorrelation": None,
        }
    return max(candidates, key=lambda item: item["absolute_lag1_autocorrelation"])


def plot_chain_diagnostics(
    output_dir: Path,
    samples_chain: dict[str, Array],
    worst: dict,
    model_name: str,
    budget: str,
) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for ax, parameter in zip(axes, ("ls", "beta")):
        values = np.asarray(samples_chain[parameter])
        for chain in range(values.shape[0]):
            is_worst = chain == worst["chain"] and parameter == worst["parameter"]
            ax.plot(
                values[chain],
                color="red" if is_worst else colors[chain % len(colors)],
                linewidth=1.0 if is_worst else 0.6,
                alpha=1.0 if is_worst else 0.75,
                label=f"chain {chain}" + (" (worst)" if is_worst else ""),
            )
        ax.set_ylabel(parameter)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("post-warmup draw")
    fig.suptitle(f"{model_name}, {budget} budget")
    fig.savefig(output_dir / "trace_all_chains.png", dpi=180)
    plt.close(fig)

    chain = int(worst["chain"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), constrained_layout=True)
    for ax, parameter in zip(axes, ("ls", "beta")):
        values = np.asarray(samples_chain[parameter])[chain]
        ax.plot(values, color="red", linewidth=0.7)
        ax.axhline(values.mean(), color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(parameter)
    axes[-1].set_xlabel("post-warmup draw")
    fig.suptitle(
        f"Worst chain {chain}: selected by max absolute lag-1 autocorrelation"
    )
    fig.savefig(output_dir / "worst_chain_trace.png", dpi=180)
    plt.close(fig)


def run_hmc_diagnostic(
    cfg: Config,
    budget: str,
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Array,
    surrogate_decoder: Optional[Callable],
) -> tuple[dict, dict, float, dict]:
    warmup, draws = budget_values(cfg, budget)
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(
        nuts,
        num_chains=cfg.num_chains,
        num_samples=draws,
        num_warmup=warmup,
        progress_bar=True,
    )
    rng_run, rng_pred = random.split(rng)
    start = perf_counter()
    mcmc.run(
        rng_run,
        surrogate_decoder=surrogate_decoder,
        obs_mask=obs_mask,
        y=y_obs,
        extra_fields=("diverging",),
    )
    infer_time = perf_counter() - start
    samples_chain = {
        key: value
        for key, value in mcmc.get_samples(group_by_chain=True).items()
        if key in ("ls", "beta")
    }
    samples_flat = {
        key: value.reshape((-1,) + value.shape[2:])
        for key, value in samples_chain.items()
    }
    predictive_start = perf_counter()
    posterior = Predictive(model, samples_flat)(
        rng_pred, surrogate_decoder=surrogate_decoder
    )
    jax.block_until_ready(posterior["obs"])
    posterior_predictive_time = perf_counter() - predictive_start
    inference_data = az.from_dict(
        posterior={
            key: np.asarray(value)
            for key, value in samples_chain.items()
        }
    )
    rhat = az.rhat(inference_data)
    ess_bulk = az.ess(inference_data, method="bulk")
    ess_tail = az.ess(inference_data, method="tail")
    mcse = az.mcse(inference_data, method="mean")
    extra = mcmc.get_extra_fields(group_by_chain=True)
    worst = identify_worst_chain(samples_chain)
    posterior_sd = {
        parameter: float(np.asarray(samples_chain[parameter]).std(ddof=1))
        for parameter in ("ls", "beta")
    }
    diagnostics = {
        "mcmc_warmup": warmup,
        "mcmc_samples_per_chain": draws,
        "num_chains": cfg.num_chains,
        "num_divergences": int(np.asarray(extra["diverging"]).sum()),
        "nuts_inference_time": infer_time,
        "posterior_predictive_time": posterior_predictive_time,
        "inference_total_time": infer_time + posterior_predictive_time,
        "rhat_ls": scalar_stat(rhat, "rhat", "ls"),
        "rhat_beta": scalar_stat(rhat, "rhat", "beta"),
        "ess_bulk_ls": scalar_stat(ess_bulk, "ess_bulk", "ls"),
        "ess_bulk_beta": scalar_stat(ess_bulk, "ess_bulk", "beta"),
        "ess_tail_ls": scalar_stat(ess_tail, "ess_tail", "ls"),
        "ess_tail_beta": scalar_stat(ess_tail, "ess_tail", "beta"),
        "mcse_mean_ls": scalar_stat(mcse, "mcse", "ls"),
        "mcse_mean_beta": scalar_stat(mcse, "mcse", "beta"),
        "posterior_sd_ls": posterior_sd["ls"],
        "posterior_sd_beta": posterior_sd["beta"],
        "worst_chain": worst,
    }
    diagnostics["relative_mcse_ls"] = (
        diagnostics["mcse_mean_ls"] / posterior_sd["ls"]
        if posterior_sd["ls"] > 0
        else None
    )
    diagnostics["relative_mcse_beta"] = (
        diagnostics["mcse_mean_beta"] / posterior_sd["beta"]
        if posterior_sd["beta"] > 0
        else None
    )
    mu_chain = np.asarray(posterior["mu"]).reshape(
        cfg.num_chains, draws, -1
    )
    mu_data = az.from_dict(posterior={"mu": mu_chain})
    mu_rhat = np.asarray(az.rhat(mu_data)["mu"]).reshape(-1)
    mu_bulk = np.asarray(
        az.ess(mu_data, method="bulk")["mu"]
    ).reshape(-1)
    mu_tail = np.asarray(
        az.ess(mu_data, method="tail")["mu"]
    ).reshape(-1)
    diagnostics.update(
        {
            "latent_mu_rhat_max": float(np.nanmax(mu_rhat)),
            "latent_mu_rhat_q99": float(np.nanquantile(mu_rhat, 0.99)),
            "latent_mu_rhat_over_threshold": int(
                np.sum(mu_rhat > cfg.max_rhat)
            ),
            "latent_mu_bulk_ess_min": float(np.nanmin(mu_bulk)),
            "latent_mu_bulk_ess_q01": float(
                np.nanquantile(mu_bulk, 0.01)
            ),
            "latent_mu_tail_ess_min": float(np.nanmin(mu_tail)),
            "latent_mu_tail_ess_q01": float(
                np.nanquantile(mu_tail, 0.01)
            ),
        }
    )
    posterior_small = {"obs": posterior["obs"], "mu": posterior.get("mu")}
    return samples_chain, posterior_small, infer_time, diagnostics


def assess_diagnostics(cfg: Config, diagnostics: dict) -> dict:
    failures = []
    for parameter in ("ls", "beta"):
        rhat = diagnostics[f"rhat_{parameter}"]
        bulk = diagnostics[f"ess_bulk_{parameter}"]
        tail = diagnostics[f"ess_tail_{parameter}"]
        relative_mcse = diagnostics[f"relative_mcse_{parameter}"]
        if rhat is None or not np.isfinite(rhat) or rhat > cfg.max_rhat:
            failures.append(f"{parameter}: rhat={rhat}")
        if bulk is None or not np.isfinite(bulk) or bulk < cfg.min_bulk_ess:
            failures.append(f"{parameter}: bulk_ess={bulk}")
        if tail is None or not np.isfinite(tail) or tail < cfg.min_tail_ess:
            failures.append(f"{parameter}: tail_ess={tail}")
        if (
            relative_mcse is None
            or not np.isfinite(relative_mcse)
            or relative_mcse > cfg.max_relative_mcse
        ):
            failures.append(f"{parameter}: relative_mcse={relative_mcse}")
    if diagnostics["num_divergences"] > 0:
        failures.append(
            f"divergences={diagnostics['num_divergences']}"
        )
    latent_rhat = diagnostics.get("latent_mu_rhat_max")
    latent_bulk = diagnostics.get("latent_mu_bulk_ess_min")
    latent_tail = diagnostics.get("latent_mu_tail_ess_min")
    if (
        latent_rhat is None
        or not np.isfinite(latent_rhat)
        or latent_rhat > cfg.max_rhat
    ):
        failures.append(
            f"latent_mu: rhat_max={latent_rhat}"
        )
    if (
        latent_bulk is None
        or not np.isfinite(latent_bulk)
        or latent_bulk < cfg.min_bulk_ess
    ):
        failures.append(
            f"latent_mu: bulk_ess_min={latent_bulk}"
        )
    if (
        latent_tail is None
        or not np.isfinite(latent_tail)
        or latent_tail < cfg.min_tail_ess
    ):
        failures.append(
            f"latent_mu: tail_ess_min={latent_tail}"
        )
    return {
        "diagnostics_passed": not failures,
        "diagnostic_failures": failures,
        "thresholds": {
            "max_rhat": cfg.max_rhat,
            "min_bulk_ess": cfg.min_bulk_ess,
            "min_tail_ess": cfg.min_tail_ess,
            "max_relative_mcse": cfg.max_relative_mcse,
            "max_divergences": 0,
        },
    }


def inference_dir(seed_dir: Path, budget: str, model_name: str) -> Path:
    return seed_dir / "inference" / budget / model_name


def load_inference(output_dir: Path) -> Optional[tuple[dict, dict, dict]]:
    required = (
        output_dir / "complete.json",
        output_dir / "posterior_samples_by_chain.pkl",
        output_dir / "posterior_predictive.npz",
        output_dir / "metrics.json",
    )
    if not all(path.exists() for path in required):
        return None
    with (output_dir / "posterior_samples_by_chain.pkl").open("rb") as f:
        samples = pickle.load(f)
    predictive = np.load(output_dir / "posterior_predictive.npz")
    posterior = {
        "obs": jnp.asarray(predictive["obs"]),
        "mu": None if predictive["mu"].size == 0 else jnp.asarray(predictive["mu"]),
    }
    return samples, posterior, json.loads((output_dir / "metrics.json").read_text())


def save_inference(
    output_dir: Path,
    samples_chain: dict,
    posterior: dict,
    metrics: dict,
    diagnostics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_pickle(output_dir / "posterior_samples_by_chain.pkl", samples_chain)
    np.savez_compressed(
        output_dir / "posterior_predictive.npz",
        obs=np.asarray(posterior["obs"]),
        mu=(
            np.asarray(posterior["mu"])
            if posterior.get("mu") is not None
            else np.empty((0,))
        ),
    )
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "metrics.csv", [metrics])
    write_json(
        output_dir / "complete.json",
        {"completed_at_utc": utc_now(), "expected_files_present": True},
    )


def run_one_inference(
    cfg: Config,
    seed_dir: Path,
    budget: str,
    model_name: str,
    data: dict,
    infer_model: Callable,
    decoder: Optional[Callable],
    training: Optional[dict],
    sample_s: Optional[Array],
    full_reference: Optional[tuple[dict, dict]],
) -> tuple[dict, dict, dict]:
    output_dir = inference_dir(seed_dir, budget, model_name)
    if cfg.force_rerun:
        reset_output(output_dir)
    existing = load_inference(output_dir)
    if existing is not None and not cfg.force_rerun:
        print(f"{budget}/{model_name}: complete output exists; skipping")
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    key = random.fold_in(
        random.key(cfg.seed + 2_000_000),
        BUDGET_IDS[budget] * 100 + MODEL_IDS[model_name],
    )
    samples_chain, posterior, infer_time, diagnostics = run_hmc_diagnostic(
        cfg,
        budget,
        key,
        infer_model,
        data["y_full"],
        data["obs_mask"],
        decoder,
    )
    assessment = assess_diagnostics(cfg, diagnostics)
    diagnostics.update(assessment)
    samples_flat = {
        key: value.reshape((-1,) + value.shape[2:])
        for key, value in samples_chain.items()
    }
    reference_flat = None
    if full_reference is not None:
        reference_samples, reference_posterior = full_reference
        reference_flat = {
            key: value.reshape((-1,) + value.shape[2:])
            for key, value in reference_samples.items()
        }
        full_reference_flat = (reference_flat, reference_posterior)
    else:
        full_reference_flat = None
    budget_cfg = replace(
        cfg,
        mcmc_warmup=diagnostics["mcmc_warmup"],
        mcmc_samples=diagnostics["mcmc_samples_per_chain"],
    )
    legacy_ess = {
        "ls": diagnostics["ess_bulk_ls"],
        "beta": diagnostics["ess_bulk_beta"],
    }
    metrics = summarize_model(
        budget_cfg,
        model_name,
        sample_s,
        data,
        samples_flat,
        posterior,
        infer_time,
        legacy_ess,
        {"num_divergences": diagnostics["num_divergences"]},
        training,
        full_reference_flat,
    )
    metrics.update(
        {
            "budget": budget,
            **{
                key: value
                for key, value in diagnostics.items()
                if key != "worst_chain"
            },
            "worst_chain_index": diagnostics["worst_chain"]["chain"],
            "worst_chain_parameter": diagnostics["worst_chain"]["parameter"],
            "worst_chain_lag1_autocorrelation": diagnostics["worst_chain"][
                "lag1_autocorrelation"
            ],
            "diagnostics_passed": diagnostics["diagnostics_passed"],
            "diagnostic_failures": "; ".join(
                diagnostics["diagnostic_failures"]
            ),
        }
    )
    save_inference(output_dir, samples_chain, posterior, metrics, diagnostics)
    plot_chain_diagnostics(
        output_dir,
        samples_chain,
        diagnostics["worst_chain"],
        model_name,
        budget,
    )
    return samples_chain, posterior, metrics


def write_budget_comparison(cfg: Config, run_dir: Path, seed_dir: Path) -> None:
    rows = []
    for model_name in cfg.models:
        initial = load_inference(
            inference_dir(seed_dir, "initial", model_name)
        )
        extended = load_inference(
            inference_dir(seed_dir, "extended", model_name)
        )
        if initial is None or extended is None:
            continue
        initial_samples, initial_posterior, initial_metrics = initial
        extended_samples, extended_posterior, extended_metrics = extended
        initial_mean = initial_posterior["obs"].mean(axis=0)
        extended_mean = extended_posterior["obs"].mean(axis=0)
        row = {
            "seed": cfg.seed,
            "model_name": model_name,
            "initial_infer_time": initial_metrics["infer_time"],
            "extended_infer_time": extended_metrics["infer_time"],
            "runtime_ratio_initial_to_extended": (
                initial_metrics["infer_time"] / extended_metrics["infer_time"]
            ),
            "posterior_mean_mse_initial_vs_extended": float(
                jnp.mean((initial_mean - extended_mean) ** 2)
            ),
            "posterior_mean_log1p_mse_initial_vs_extended": float(
                jnp.mean(
                    (
                        jnp.log1p(initial_mean)
                        - jnp.log1p(extended_mean)
                    )
                    ** 2
                )
            ),
            "ls_wasserstein_initial_vs_extended": float(
                wasserstein_distance(
                    np.asarray(initial_samples["ls"]).reshape(-1),
                    np.asarray(extended_samples["ls"]).reshape(-1),
                )
            ),
            "beta_wasserstein_initial_vs_extended": float(
                wasserstein_distance(
                    np.asarray(initial_samples["beta"]).reshape(-1),
                    np.asarray(extended_samples["beta"]).reshape(-1),
                )
            ),
            "initial_rhat_max": max(
                initial_metrics["rhat_ls"], initial_metrics["rhat_beta"]
            ),
            "extended_rhat_max": max(
                extended_metrics["rhat_ls"], extended_metrics["rhat_beta"]
            ),
            "initial_ess_bulk_min": min(
                initial_metrics["ess_bulk_ls"],
                initial_metrics["ess_bulk_beta"],
            ),
            "extended_ess_bulk_min": min(
                extended_metrics["ess_bulk_ls"],
                extended_metrics["ess_bulk_beta"],
            ),
        }
        rows.append(row)
    write_csv(seed_dir / "budget_comparison.csv", rows)
    aggregate_budget_comparisons(run_dir)


def aggregate_budget_comparisons(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.glob("seed_*/budget_comparison.csv")):
        import csv

        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return
    numeric_keys = [key for key in rows[0] if key not in ("model_name",)]
    typed = []
    for row in rows:
        typed.append(
            {
                key: (
                    row[key]
                    if key == "model_name"
                    else float(row[key])
                )
                for key in row
            }
        )
    write_csv(run_dir / "combined_budget_comparison.csv", typed)
    summary = []
    for model_name in MODELS:
        selected = [row for row in typed if row["model_name"] == model_name]
        if not selected:
            continue
        item = {"model_name": model_name, "num_seeds": len(selected)}
        for key in numeric_keys:
            if key == "seed":
                continue
            values = [row[key] for row in selected]
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summary.append(item)
    write_csv(run_dir / "aggregate_budget_comparison.csv", summary)


def refresh_metrics(run_dir: Path, seed_dir: Path) -> None:
    rows = []
    for path in sorted(seed_dir.glob("inference/*/*/metrics.json")):
        rows.append(json.loads(path.read_text()))
    write_csv(seed_dir / "metrics.csv", rows)

    combined = []
    for path in sorted(run_dir.glob("seed_*/inference/*/*/metrics.json")):
        combined.append(json.loads(path.read_text()))
    write_csv(run_dir / "combined_metrics.csv", combined)


def write_diagnostic_decision(seed_dir: Path, payload: dict) -> Path:
    latest = seed_dir / "initial_diagnostic_decision.json"
    write_json(latest, payload)
    return latest


def main() -> None:
    cfg = parse_args()
    numpyro.set_host_device_count(cfg.num_chains)
    wandb.init(mode="disabled")
    run_dir, seed_dir = prepare_dirs(cfg)
    log_file = (seed_dir / "console.log").open("a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print("=" * 80)
    print("Started:", utc_now())
    print("Output:", seed_dir)
    print("Config:", json.dumps(asdict(cfg), indent=2))

    target_s = make_grid(cfg.grid_size, 0.0, cfg.domain_stop)
    data = load_or_create_data(cfg, seed_dir, target_s)
    if cfg.prepare_data_only:
        print("Prepared data and figures only:", seed_dir)
        return
    priors = {
        "ls": dist.LogNormal(loc=cfg.prior_loc, scale=cfg.prior_scale),
        "beta": dist.Normal(0.0, 1.0),
    }
    infer_model = inference_model(target_s, priors)
    trained = train_models(cfg, seed_dir, target_s, priors)

    full_samples, full_posterior, full_metrics = run_one_inference(
        cfg,
        seed_dir,
        "initial",
        "full_gp",
        data,
        infer_model,
        None,
        None,
        None,
        None,
    )
    reference = (full_samples, full_posterior)
    initial_results = {"full_gp": full_metrics}
    for model_name in cfg.models:
        if model_name == "full_gp":
            continue
        decoder, training, sample_s = trained[model_name]
        _, _, metrics = run_one_inference(
            cfg,
            seed_dir,
            "initial",
            model_name,
            data,
            infer_model,
            decoder,
            training,
            sample_s,
            reference,
        )
        initial_results[model_name] = metrics

    failed_models = [
        model_name
        for model_name, metrics in initial_results.items()
        if not metrics["diagnostics_passed"]
    ]
    write_diagnostic_decision(
        seed_dir,
        {
            "evaluated_at_utc": utc_now(),
            "failed_models": failed_models,
            "evaluated_models": list(initial_results),
            "auto_extend_failed": cfg.auto_extend_failed,
        },
    )
    if cfg.auto_extend_failed and failed_models:
        print("Extending failed models:", failed_models)
        extended_full = None
        if "full_gp" in failed_models:
            extended_full = run_one_inference(
                cfg,
                seed_dir,
                "extended",
                "full_gp",
                data,
                infer_model,
                None,
                None,
                None,
                None,
            )
        extended_reference = (
            (extended_full[0], extended_full[1])
            if extended_full is not None
            else reference
        )
        for model_name in failed_models:
            if model_name == "full_gp":
                continue
            decoder, training, sample_s = trained[model_name]
            run_one_inference(
                cfg,
                seed_dir,
                "extended",
                model_name,
                data,
                infer_model,
                decoder,
                training,
                sample_s,
                extended_reference,
            )
        write_budget_comparison(cfg, run_dir, seed_dir)

    refresh_metrics(run_dir, seed_dir)
    print("Finished:", utc_now())
    print("Initial outputs:", seed_dir / "inference" / "initial")
    print("Diagnostic decision:", seed_dir / "initial_diagnostic_decision.json")


if __name__ == "__main__":
    main()
