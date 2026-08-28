"""Final Target-128 DeepRV training and posterior inference."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

TARGET_GRID_SIZE = 128
N_TARGET = TARGET_GRID_SIZE**2
DOMAIN_MIN = 0.0
DOMAIN_MAX = 100.0
TRUE_LENGTHSCALE = 30.0
TRUE_BETA = 1.0
GP_VARIANCE = 1.0
OBSERVATION_RATIO = 0.5
N_OBSERVED = N_TARGET // 2
N_UNOBSERVED = N_TARGET // 2
DECODER_SEED = 0
NUTS_SEED = 0
INDUCING_GRID_SIZES = (8, 16, 32, 64)
WEIGHTINGS = ("bilinear", "cubic", "dtc", "fitc")
ARCHITECTURE = {
    "class_name": "gMLPDeepRV",
    "num_blocks": 2,
    "embedding_dimension": 64,
    "hidden_expansion_dimension": 128,
    "activation": "GELU",
    "normalization": "LayerNorm",
    "residual_structure": "x = x + block(LayerNorm(x))",
    "conditioning": "ell appended as a per-location feature after [z, x, y]",
    "output_parameterization": "MLP([128, 1], GELU), unconstrained latent field",
    "initialization": "Flax Dense orthogonal kernels; spatial gate Lecun-uniform; gate bias one",
    "dtype": "float32",
    "input_features_per_location": 4,
    "latent_dimension": N_TARGET,
    "output_field_dimension": N_TARGET,
    "parameter_count_formula": "79745 + 2*N^2 + 2*N",
}

PUBLIC_SEEDS = (0, 1, 2)
FINAL_MODELS = ('Exact128', 'Bilinear64', 'Cubic64', 'DTC64', 'FITC64')
FULL_FACTORIAL_MODELS = tuple(
    f"{ {'bilinear': 'Bilinear', 'cubic': 'Cubic', 'dtc': 'DTC', 'fitc': 'FITC'}[weighting] }{grid}"
    for weighting in WEIGHTINGS
    for grid in INDUCING_GRID_SIZES
) + ("Exact128",)

@dataclass(frozen=True)
class Config:
    output_root: str = 'outputs/target128'
    target_grid_size: int = TARGET_GRID_SIZE
    domain_min: float = DOMAIN_MIN
    domain_max: float = DOMAIN_MAX
    true_lengthscale: float = TRUE_LENGTHSCALE
    true_beta: float = TRUE_BETA
    gp_variance: float = GP_VARIANCE
    kernel: str = 'matern_1_2'
    covariance_jitter: float = 5e-4
    observation_ratio: float = OBSERVATION_RATIO
    mask_type: str = 'uniform_random'
    decoder_seed: int = DECODER_SEED
    nuts_seed: int = NUTS_SEED
    inducing_grid_sizes: tuple[int, ...] = INDUCING_GRID_SIZES
    formal_train_steps: int = 300_000
    checkpoint_steps: tuple[int, ...] = (200_000, 250_000, 300_000)
    checkpoint_interval: int = 10_000
    microbatch_size: int = 16
    gradient_accumulation_steps: int = 1
    valid_steps: int = 500
    validation_interval: int = 10_000
    learning_rate: float = 5e-3
    gradient_clip_norm: float = 3.0
    train_runtime_probe_steps: int = 1_000
    runtime_probe_warmup: int = 500
    runtime_probe_samples: int = 500
    num_chains: int = 1
    num_warmup: int = 4_000
    num_samples: int = 6_000
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    init_to_median_num_samples: int = 10
    dense_mass: bool = False
    thinning: int = 1
    chain_method: str = 'parallel'
    progress_bar: bool = True
    predictive_batch_size: int = 16
    teacher_target_chunk_size: int = 512
    gpu_memory_safety_fraction: float = 0.80
    keep_periodic_checkpoints: int = 2
    single_chain_thresholds: Mapping[str, float] = field(default_factory=lambda: {
        'max_divergences': 0.0, 'max_depth_hit_fraction': 0.05,
        'min_bulk_ess_scalar': 400.0, 'min_tail_ess_scalar': 400.0,
        'max_relative_mcse_scalar': 0.05, 'max_block_mean_range_sd': 0.50,
        'max_half_wasserstein_sd': 0.25, 'max_running_mean_drift_sd': 0.25,
        'max_lag1_autocorrelation': 0.99,
    })

def load_raw_posterior_samples(path: Path, expected_samples: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        missing = {"ell", "beta", "z"}.difference(loaded.files)
        if missing:
            raise RuntimeError(f"Raw posterior is missing sites: {sorted(missing)}")
        samples = {key: np.asarray(loaded[key]) for key in ("ell", "beta", "z")}
    expected_shapes = {
        "ell": (1, expected_samples),
        "beta": (1, expected_samples),
        "z": (1, expected_samples, 1, N_TARGET),
    }
    for key, expected_shape in expected_shapes.items():
        value = samples[key]
        if value.shape != expected_shape:
            raise RuntimeError(
                f"Raw posterior {key} shape mismatch: {value.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"Raw posterior {key} contains non-finite values")
    return samples

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_bytes(path, b"")
        return
    keys = sorted({str(key) for row in rows for key in row})
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: jsonable(row.get(key)) for key in keys} for row in rows])
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)

def run_training(cfg: Config, name: str, *, probe: bool) -> dict[str, Any]:
    root = resolved_root(cfg)
    runtime = lazy_runtime_imports()
    runtime["numpyro"].set_host_device_count(1)
    model_root = training_dir(root, name, probe=probe)
    model_root.mkdir(parents=True, exist_ok=True)
    write_json(model_root / "status.json", {"status": "RUNNING", "updated_at_utc": utc_now()})
    target_s, generator, model, optimizer, state, train_key, valid_key = initialize_training(runtime, cfg, name)
    checkpoint_dir = model_root / "checkpoints"
    best_params, best_kwargs, best_metric = state.params, state.kwargs, math.inf
    if (checkpoint_dir / "latest.json").exists() and not probe:
        state, best_params, best_kwargs, best_metric, _ = restore_checkpoint(
            runtime, checkpoint_dir, model, optimizer
        )
    initial_step = int(state.step)
    total_steps = cfg.train_runtime_probe_steps if probe else cfg.formal_train_steps
    train_step = accumulated_train_step(runtime, cfg.gradient_accumulation_steps)
    history_path = model_root / "training_history.csv"
    history = []
    if history_path.exists():
        with history_path.open(newline="") as handle:
            history = list(csv.DictReader(handle))
    validation_history_path = model_root / "validation_history.csv"
    validation_history = []
    if validation_history_path.exists():
        with validation_history_path.open(newline="") as handle:
            validation_history = list(csv.DictReader(handle))
    compile_seconds = None
    first_100_seconds = 0.0
    stable_900_seconds = 0.0
    validation_seconds_total = 0.0
    checkpoint_seconds_total = 0.0
    started = perf_counter()
    for step_number in range(int(state.step) + 1, total_steps + 1):
        batch_started = perf_counter()
        batch = generator(runtime["random"].fold_in(train_key, step_number))
        runtime["jax"].block_until_ready(batch["f"])
        teacher_seconds = perf_counter() - batch_started
        update_started = perf_counter()
        state, loss = train_step(
            runtime["random"].fold_in(train_key, cfg.formal_train_steps + step_number), state, batch
        )
        runtime["jax"].block_until_ready((state.params, loss))
        update_seconds = perf_counter() - update_started
        step_seconds = teacher_seconds + update_seconds
        if step_number <= 100:
            first_100_seconds += step_seconds
        else:
            stable_900_seconds += step_seconds
        if step_number == 1:
            compile_seconds = teacher_seconds + update_seconds
        if step_number % 100 == 0 or step_number == total_steps:
            history.append(
                {
                    "step": step_number,
                    "loss": float(loss),
                    "teacher_seconds": teacher_seconds,
                    "optimizer_update_seconds": update_seconds,
                    "probe": probe,
                }
            )
            write_csv(history_path, history)
        if step_number % 1000 == 0 or step_number == total_steps:
            elapsed = perf_counter() - started
            rate = elapsed / max(step_number - initial_step, 1)
            remaining = max(total_steps - step_number, 0) * rate
            print(
                "TRAIN_PROGRESS"
                f" model={name} step={step_number}/{total_steps}"
                f" loss={float(loss):.8g}"
                f" elapsed_hours={elapsed / 3600:.3f}"
                f" estimated_remaining_hours={remaining / 3600:.3f}",
                flush=True,
            )
        should_validate = step_number % cfg.validation_interval == 0 or step_number == total_steps
        should_checkpoint = step_number % cfg.checkpoint_interval == 0 or step_number == total_steps
        validation_metric = None
        validation_seconds = 0.0
        if should_validate:
            valid_started = perf_counter()
            validation_metric = evaluate_training(runtime, state, generator, runtime["random"].fold_in(valid_key, step_number), cfg.valid_steps)
            validation_seconds = perf_counter() - valid_started
            validation_seconds_total += validation_seconds
            if validation_metric < best_metric:
                best_metric = validation_metric
                best_params, best_kwargs = state.params, state.kwargs
                write_json(model_root / "best_validation.json", {"step": step_number, "validation_loss": best_metric})
            validation_history.append(
                {
                    "step": step_number,
                    "validation_loss": validation_metric,
                    "validation_seconds": validation_seconds,
                    "probe": probe,
                }
            )
            write_csv(validation_history_path, validation_history)
        if should_checkpoint:
            keep = step_number in cfg.checkpoint_steps
            checkpoint_started = perf_counter()
            path = save_checkpoint(runtime, checkpoint_dir, state, best_params, best_kwargs, best_metric, cfg, keep=keep)
            checkpoint_seconds = perf_counter() - checkpoint_started
            checkpoint_seconds_total += checkpoint_seconds
            write_json(
                model_root / "latest_progress.json",
                {
                    "step": step_number,
                    "loss": float(loss),
                    "validation_loss": validation_metric,
                    "validation_seconds": validation_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                    "checkpoint_path": str(path),
                    "updated_at_utc": utc_now(),
                },
            )
    total_seconds = perf_counter() - started
    result = {
        "status": "RUNTIME_CALIBRATION_NOT_FOR_ANALYSIS" if probe else "PASS",
        "model": name,
        "probe": probe,
        "steps": total_steps,
        "completed_step": int(state.step),
        "compile_seconds": compile_seconds,
        "first_100_steps_seconds": first_100_seconds,
        "post_first_100_steps_seconds": stable_900_seconds,
        "later_900_steps_seconds": stable_900_seconds if probe and total_steps == 1000 else None,
        "validation_seconds_total": validation_seconds_total,
        "checkpoint_save_seconds_total": checkpoint_seconds_total,
        "total_seconds": total_seconds,
        "seconds_per_step_overall": total_seconds / max(total_steps, 1),
        "best_validation_loss": best_metric,
        "effective_batch_size": cfg.microbatch_size * cfg.gradient_accumulation_steps,
        "microbatch_size": cfg.microbatch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "model_signature": model_signature(),
        "gpu": gpu_memory(),
        "peak_gpu_memory": jax_peak_memory(runtime),
    }
    if probe:
        stable_rows = [row for row in history if int(row["step"]) > 100]
        stable = [float(row["teacher_seconds"]) + float(row["optimizer_update_seconds"]) for row in stable_rows]
        central = float(np.mean(stable)) if stable else result["seconds_per_step_overall"]
        optimistic = float(np.quantile(stable, 0.25)) if stable else central * 0.9
        conservative = float(np.quantile(stable, 0.90)) if stable else central * 1.25
        estimate = {
            **result,
            "seconds_per_stable_training_step": central,
            "optimistic_hours": {str(step): optimistic * step / 3600 for step in cfg.checkpoint_steps},
            "central_hours": {str(step): central * step / 3600 for step in cfg.checkpoint_steps},
            "conservative_hours": {str(step): conservative * step / 3600 for step in cfg.checkpoint_steps},
            "runtime_probe_does_not_affect_formal_checkpoint": True,
        }
        write_json(root / "runtime_estimates" / "training_runtime_estimate.json", estimate)
        result = estimate
    else:
        write_json(model_root / "training_result.json", result)
    write_json(model_root / "status.json", result)
    write_json(model_root / "complete.json", {"status": result["status"], "completed_at_utc": utc_now()})
    return result

def initialize_training(runtime: Mapping[str, Any], cfg: Config, name: str):
    jax, random = runtime["jax"], runtime["random"]
    target_s = make_grid(runtime, TARGET_GRID_SIZE)
    effective_batch = cfg.microbatch_size * cfg.gradient_accumulation_steps
    generator = make_teacher_generator(runtime, cfg, name, target_s, effective_batch)
    stable_id = FULL_FACTORIAL_MODELS.index(name)
    model_key = random.fold_in(random.key(cfg.decoder_seed), stable_id)
    init_key, train_key, valid_key = random.split(model_key, 3)
    init_batch = generator(random.fold_in(train_key, 0))
    jax.block_until_ready(init_batch["f"])
    model = runtime["gMLPDeepRV"](num_blks=2)
    optimizer = optimizer_for(runtime, cfg)
    rng_parameters, rng_extra = random.split(init_key)
    variables = model.init({"params": rng_parameters, "extra": rng_extra}, **init_batch)
    params = variables.pop("params")
    state = runtime["TrainState"].create(apply_fn=model.apply, params=params, kwargs=variables, tx=optimizer)
    return target_s, generator, model, optimizer, state, train_key, valid_key

def flatten_posterior_samples(
    samples_chain: Mapping[str, np.ndarray], expected_samples: int
) -> dict[str, np.ndarray]:
    flat = {
        key: np.asarray(value).reshape((-1,) + np.asarray(value).shape[2:])
        for key, value in samples_chain.items()
        if key in ("ell", "beta", "z")
    }
    for key in ("ell", "beta", "z"):
        if key not in flat or flat[key].shape[0] != expected_samples:
            raise RuntimeError(f"Flattened posterior {key} does not contain {expected_samples} draws")
    if flat["z"].shape[1:] != (1, N_TARGET):
        raise RuntimeError(f"Flattened posterior z shape is invalid: {flat['z'].shape}")
    return flat

def validate_seed_payload(payload: Mapping[str, np.ndarray]) -> None:
    expected = {
        "coordinates": (N_TARGET, 2),
        "latent_truth": (N_TARGET,),
        "log_rate": (N_TARGET,),
        "rate": (N_TARGET,),
        "counts": (N_TARGET,),
        "mask": (N_TARGET,),
        "observed_indices": (N_OBSERVED,),
        "unobserved_indices": (N_UNOBSERVED,),
    }
    for name, shape in expected.items():
        if name not in payload or payload[name].shape != shape:
            raise ValueError(f"{name} shape mismatch: {payload.get(name)}")
    if not np.isfinite(payload["latent_truth"]).all() or not np.isfinite(payload["rate"]).all():
        raise ValueError("NaN/Inf in seed payload")
    if int(payload["mask"].sum()) != N_OBSERVED:
        raise ValueError("Observed count mismatch")
    if np.intersect1d(payload["observed_indices"], payload["unobserved_indices"]).size:
        raise ValueError("Observed/unobserved indices overlap")
    if len(np.union1d(payload["observed_indices"], payload["unobserved_indices"])) != N_TARGET:
        raise ValueError("Observed/unobserved indices do not cover target")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def nearest_regular_target_indices(size: int) -> np.ndarray:
    inducing_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, size)
    target_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, TARGET_GRID_SIZE)
    nearest = np.abs(inducing_axis[:, None] - target_axis[None, :]).argmin(axis=1)
    if len(np.unique(nearest)) != size:
        raise AssertionError("Nearest regular-grid assignment is not injective")
    return (nearest[:, None] * TARGET_GRID_SIZE + nearest[None, :]).reshape(-1).astype(np.int32)

def architecture_report() -> dict[str, Any]:
    count_64 = 79_745 + 2 * (64**2) ** 2 + 2 * (64**2)
    count_128 = 79_745 + 2 * N_TARGET**2 + 2 * N_TARGET
    shapes = parameter_shapes()
    if sum(item["count"] for item in shapes) != count_128:
        raise AssertionError("Analytical parameter-shape ledger mismatch")
    return {
        **ARCHITECTURE,
        "target_grid_size": TARGET_GRID_SIZE,
        "target_locations": N_TARGET,
        "parameter_shapes": shapes,
        "parameter_count_64x64": count_64,
        "parameter_count_128x128": count_128,
        "parameter_count_ratio_128_to_64": count_128 / count_64,
        "parameter_dtype": "float32",
    }

def gib(value: float) -> float:
    return float(value) / 1024**3

def evaluate_training(runtime: Mapping[str, Any], state: Any, generator: Any, key: Any, steps: int) -> float:
    values = []
    for index in range(steps):
        batch = generator(runtime["random"].fold_in(key, index))
        output = state.apply_fn(
            {"params": state.params, **state.kwargs}, **batch, rngs={"extra": runtime["random"].fold_in(key, steps + index)}
        )
        values.append(output.mse(batch["f"]))
    return float(np.asarray(runtime["jax"].device_get(runtime["jnp"].mean(runtime["jnp"].stack(values)))))

def run_nuts(
    cfg: Config,
    name: str,
    seed: int,
    *,
    probe: bool,
) -> dict[str, Any]:
    root = resolved_root(cfg)
    runtime = lazy_runtime_imports()
    jax, jnp, random = runtime["jax"], runtime["jnp"], runtime["random"]
    runtime["numpyro"].set_host_device_count(1)
    decoder, checkpoint_info = load_trained_decoder(runtime, cfg, name)
    data = generate_public_dataset(cfg, seed)
    posterior_model = build_posterior_model(runtime, data)

    warmup = cfg.runtime_probe_warmup if probe else cfg.num_warmup
    samples = cfg.runtime_probe_samples if probe else cfg.num_samples
    if probe:
        output_dir = root / "runtime_estimates" / "nuts_probe" / name / f"seed_{seed}"
    else:
        output_dir = root / "formal_inference" / name / f"seed_{seed}"
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and not probe:
        completed = json.loads(complete_path.read_text())
        diagnostics_path = output_dir / "diagnostics.json"
        metrics_path = output_dir / "metrics.json"
        if diagnostics_path.exists():
            completed["diagnostics"] = json.loads(diagnostics_path.read_text())
        if metrics_path.exists():
            completed["metrics"] = json.loads(metrics_path.read_text())
        return completed
    posterior_marker = root / "manifests" / "posterior_execution_started.json"
    if not probe and not posterior_marker.exists():
        write_json(
            posterior_marker,
            {
                "created_at_utc": utc_now(),
                "first_mode": "runtime_probe" if probe else "formal_inference",
                "model": name,
                "data_seed": seed,
                "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                "checkpoint_integrity_policy": checkpoint_info.get(
                    "inference_checkpoint_integrity_policy", "strict_hash"
                ),
                "checkpoint_directory_hash_verified_for_inference": checkpoint_info.get(
                    "checkpoint_directory_hash_verified_for_inference", True
                ),
                "checkpoint_restore_validated": checkpoint_info.get(
                    "checkpoint_restore_validated", True
                ),
            },
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "status.json", {"status": "RUNNING", "updated_at_utc": utc_now()})
    kernel = runtime["NUTS"](
        posterior_model,
        init_strategy=runtime["init_to_median"](num_samples=cfg.init_to_median_num_samples),
        target_accept_prob=cfg.target_accept_prob,
        max_tree_depth=cfg.max_tree_depth,
        dense_mass=cfg.dense_mass,
    )
    extra_fields = ("diverging", "num_steps", "accept_prob", "potential_energy", "energy", "adapt_state.step_size")
    master = random.key(cfg.nuts_seed)
    compile_key, warmup_key, sample_key, predictive_key = random.split(master, 4)
    compile_mcmc = runtime["MCMC"](
        kernel, num_chains=1, num_warmup=1, num_samples=1, thinning=1,
        chain_method=cfg.chain_method, progress_bar=False,
    )
    compile_started = perf_counter()
    compile_mcmc.run(
        compile_key,
        surrogate_decoder=decoder,
        obs_mask=jnp.asarray(data["mask"]),
        y=jnp.asarray(data["counts"]),
        extra_fields=("diverging",),
    )
    one = compile_mcmc.get_samples()
    jax.block_until_ready(one["ell"])
    compile_seconds = perf_counter() - compile_started
    del compile_mcmc, one

    mcmc = runtime["MCMC"](
        kernel,
        num_chains=1,
        num_warmup=warmup,
        num_samples=samples,
        thinning=cfg.thinning,
        chain_method=cfg.chain_method,
        progress_bar=cfg.progress_bar,
    )
    warmup_started = perf_counter()
    mcmc.warmup(
        warmup_key,
        surrogate_decoder=decoder,
        obs_mask=jnp.asarray(data["mask"]),
        y=jnp.asarray(data["counts"]),
        extra_fields=extra_fields,
    )
    jax.block_until_ready(mcmc.post_warmup_state.z)
    warmup_seconds = perf_counter() - warmup_started
    sampling_started = perf_counter()
    mcmc.run(
        sample_key,
        surrogate_decoder=decoder,
        obs_mask=jnp.asarray(data["mask"]),
        y=jnp.asarray(data["counts"]),
        extra_fields=extra_fields,
    )
    samples_chain = mcmc.get_samples(group_by_chain=True)
    jax.block_until_ready(samples_chain["ell"])
    sampling_seconds = perf_counter() - sampling_started
    extra = mcmc.get_extra_fields(group_by_chain=True)
    raw_path = output_dir / "posterior_samples_by_chain.npz"
    write_npz(raw_path, **{key: np.asarray(value) for key, value in samples_chain.items() if key != "mu"})

    scalar = {key: np.asarray(samples_chain[key]).reshape(-1) for key in ("ell", "beta")}
    inference_data = runtime["az"].from_dict(posterior={key: values[None, :] for key, values in scalar.items()})
    bulk = runtime["az"].ess(inference_data, method="bulk")
    tail = runtime["az"].ess(inference_data, method="tail")
    mcse = runtime["az"].mcse(inference_data, method="mean")
    energy = np.asarray(extra.get("energy", extra.get("potential_energy"))).reshape(-1)
    num_steps = np.asarray(extra["num_steps"]).reshape(-1)
    accept = np.asarray(extra["accept_prob"]).reshape(-1)
    divergences = int(np.asarray(extra["diverging"]).sum())
    depth_hits = int(np.sum(num_steps >= (2**cfg.max_tree_depth - 1)))
    diagnostics = {
        "rhat": "unavailable_single_chain",
        "num_chains": 1,
        "warmup": warmup,
        "samples": samples,
        "divergences": divergences,
        "max_tree_depth_hits": depth_hits,
        "max_depth_hit_fraction": depth_hits / samples,
        "depth_warnings": depth_hits > 0,
        "num_steps_mean": float(num_steps.mean()),
        "num_steps_median": float(np.median(num_steps)),
        "num_steps_p90": float(np.quantile(num_steps, 0.90)),
        "num_steps_max": int(num_steps.max()),
        "acceptance_probability_mean": float(accept.mean()),
        "step_size": float(np.asarray(mcmc.last_state.adapt_state.step_size)),
        "energy_mean": float(energy.mean()),
        "energy_std": float(energy.std(ddof=1)),
        "bfmi": float(np.asarray(runtime["az"].bfmi(energy[None, :])).reshape(-1)[0]),
        "bulk_ess_ell": float(np.asarray(bulk["ell"])),
        "bulk_ess_beta": float(np.asarray(bulk["beta"])),
        "tail_ess_ell": float(np.asarray(tail["ell"])),
        "tail_ess_beta": float(np.asarray(tail["beta"])),
        "relative_mcse_ell": (
            float(np.asarray(mcse["ell"])) / float(scalar["ell"].std(ddof=1))
            if float(scalar["ell"].std(ddof=1)) > 0
            else math.inf
        ),
        "relative_mcse_beta": (
            float(np.asarray(mcse["beta"])) / float(scalar["beta"].std(ddof=1))
            if float(scalar["beta"].std(ddof=1)) > 0
            else math.inf
        ),
        "single_chain": {key: single_chain_parameter_diagnostics(value) for key, value in scalar.items()},
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "sampling_seconds": sampling_seconds,
        "seconds_per_sample": sampling_seconds / samples,
        "ess_per_second_ell": float(np.asarray(bulk["ell"])) / sampling_seconds,
        "ess_per_second_beta": float(np.asarray(bulk["beta"])) / sampling_seconds,
        "gpu": gpu_memory(),
    }
    status, failures = diagnostic_gate(cfg, diagnostics)
    diagnostics["diagnostic_status"] = status
    diagnostics["diagnostic_failures"] = failures
    write_json(output_dir / "diagnostics.json", diagnostics)
    return finish_nuts_postprocessing(
        cfg,
        name,
        seed,
        probe=probe,
        runtime=runtime,
        decoder=decoder,
        checkpoint_info=checkpoint_info,
        data=data,
        posterior_model=posterior_model,
        predictive_key=predictive_key,
        output_dir=output_dir,
        samples_chain=samples_chain,
        diagnostics=diagnostics,
        raw_path=raw_path,
    )

def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "unavailable_nonfinite"
    return value

def make_grid(runtime: Mapping[str, Any], size: int) -> Any:
    return runtime["build_grid"](
        [{"start": DOMAIN_MIN, "stop": DOMAIN_MAX, "num": size}] * 2
    ).reshape(-1, 2)

def posterior_metrics(
    data: Mapping[str, np.ndarray], mu: np.ndarray, obs: np.ndarray,
    scalar: Mapping[str, np.ndarray], diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    mask = data["mask"]
    mu_mean = mu.mean(axis=0)
    expected_count = np.exp(TRUE_BETA + data["latent_truth"])
    expected_mean = np.exp(scalar["beta"][:, None] + mu).mean(axis=0)
    obs_mean = obs.mean(axis=0)
    latent_lo, latent_hi = np.quantile(mu, [0.05, 0.95], axis=0)
    count_lo, count_hi = np.quantile(obs, [0.05, 0.95], axis=0)

    def mse(left: np.ndarray, right: np.ndarray, selection: np.ndarray) -> float:
        return float(np.mean((left[selection] - right[selection]) ** 2))

    metrics = {
        "posterior_latent_mean_mse_versus_simulated_truth": mse(mu_mean, data["latent_truth"], np.ones(N_TARGET, dtype=bool)),
        "posterior_latent_mean_mse_unobserved": mse(mu_mean, data["latent_truth"], ~mask),
        "latent_interval_90_coverage": float(np.mean((data["latent_truth"] >= latent_lo) & (data["latent_truth"] <= latent_hi))),
        "posterior_expected_count_mse_versus_simulated_expected_count": mse(expected_mean, expected_count, np.ones(N_TARGET, dtype=bool)),
        "posterior_expected_count_mse_observed": mse(expected_mean, expected_count, mask),
        "posterior_expected_count_mse_unobserved": mse(expected_mean, expected_count, ~mask),
        "posterior_predictive_count_mse_all": mse(obs_mean, data["counts"], np.ones(N_TARGET, dtype=bool)),
        "posterior_predictive_count_mse_observed": mse(obs_mean, data["counts"], mask),
        "posterior_predictive_count_mse_unobserved": mse(obs_mean, data["counts"], ~mask),
        "posterior_predictive_log1p_mse_all": mse(np.log1p(obs_mean), np.log1p(data["counts"]), np.ones(N_TARGET, dtype=bool)),
        "posterior_predictive_90_coverage_all": float(np.mean((data["counts"] >= count_lo) & (data["counts"] <= count_hi))),
        "posterior_predictive_90_coverage_observed": float(np.mean((data["counts"][mask] >= count_lo[mask]) & (data["counts"][mask] <= count_hi[mask]))),
        "posterior_predictive_90_coverage_unobserved": float(np.mean((data["counts"][~mask] >= count_lo[~mask]) & (data["counts"][~mask] <= count_hi[~mask]))),
        "ell_mean": float(scalar["ell"].mean()),
        "ell_median": float(np.median(scalar["ell"])),
        "ell_q05": float(np.quantile(scalar["ell"], 0.05)),
        "ell_q95": float(np.quantile(scalar["ell"], 0.95)),
        "ell_mean_distance_to_true_30": float(abs(scalar["ell"].mean() - TRUE_LENGTHSCALE)),
        "beta_mean": float(scalar["beta"].mean()),
        "beta_median": float(np.median(scalar["beta"])),
        "beta_q05": float(np.quantile(scalar["beta"], 0.05)),
        "beta_q95": float(np.quantile(scalar["beta"], 0.95)),
        "rhat": "unavailable_single_chain",
        "divergences": diagnostics["divergences"],
        "max_tree_depth_hits": diagnostics["max_tree_depth_hits"],
        "bfmi": diagnostics["bfmi"],
    }
    return metrics

def runtime_environment(runtime: Mapping[str, Any]) -> dict[str, Any]:
    jax = runtime["jax"]
    numpyro = runtime["numpyro"]
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "numpyro_version": numpyro.__version__,
        "gpu": gpu_memory(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or "missing",
    }

def training_dir(root: Path, name: str, *, probe: bool = False) -> Path:
    return (root / "runtime_estimates" / "training_probe" / name) if probe else (root / "training" / name)

def plot_posterior_maps(
    output_dir: Path, data: Mapping[str, np.ndarray], mu: np.ndarray, obs: np.ndarray
) -> None:
    import matplotlib.pyplot as plt

    panels = (
        (data["latent_truth"], "simulated latent truth"),
        (mu.mean(axis=0), "posterior latent mean"),
        (mu.std(axis=0), "posterior latent std"),
        (obs.mean(axis=0), "posterior predictive count mean"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, (values, title) in zip(axes, panels):
        image = axis.imshow(values.reshape(128, 128), origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(output_dir / "posterior_maps.png", dpi=160)
    plt.close(fig)

def save_checkpoint(
    runtime: Mapping[str, Any],
    checkpoint_dir: Path,
    state: Any,
    best_params: Any,
    best_kwargs: Any,
    best_metric: float,
    cfg: Config,
    *,
    keep: bool,
) -> Path:
    actual_parameter_count = sum(
        math.prod(leaf.shape) for leaf in runtime["jax"].tree.leaves(state.params)
    )
    expected_parameter_count = architecture_report()["parameter_count_128x128"]
    if actual_parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Refusing checkpoint with {actual_parameter_count} parameters; "
            f"target-128 requires {expected_parameter_count}."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{int(state.step):08d}"
    payload = {
        "params": state.params,
        "kwargs": state.kwargs,
        "opt_state": state.opt_state,
        "step": state.step,
        "best_params": best_params,
        "best_kwargs": best_kwargs,
        "best_metric": np.asarray(best_metric),
        "model_signature": model_signature(),
        "config": jsonable(cfg),
    }
    if not path.exists():
        local_staging_root = Path(tempfile.gettempdir()) / "target128_checkpoint_staging"
        local_staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=local_staging_root,
            prefix=f"{path.name}-",
        ) as temporary_root:
            local_path = Path(temporary_root) / path.name
            runtime["PyTreeCheckpointer"]().save(
                local_path.absolute(),
                payload,
                save_args=runtime["orbax_utils"].save_args_from_target(payload),
            )
            checkpoint_hash = publish_checkpoint_directory(local_path, path)
    else:
        latest_path = checkpoint_dir / "latest.json"
        if not latest_path.is_file():
            raise RuntimeError(
                f"Refusing unreferenced existing checkpoint directory: {path}"
            )
        latest = json.loads(latest_path.read_text())
        checkpoint_hash = directory_sha256(path)
        if latest.get("path") != path.name or latest.get("sha256") != checkpoint_hash:
            raise RuntimeError(
                f"Existing checkpoint is not the hash-verified latest checkpoint: {path}"
            )
    write_json(
        checkpoint_dir / "latest.json",
        {
            "step": int(state.step),
            "path": path.name,
            "saved_at_utc": utc_now(),
            "sha256": checkpoint_hash,
            "publication_protocol": "local_stage_copy_verify_atomic_rename_v1",
        },
    )
    if keep:
        write_json(checkpoint_dir / f"retain_{int(state.step):08d}.json", {"path": path.name})
    prune_checkpoints(checkpoint_dir, cfg)
    return path

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def accumulated_train_step(runtime: Mapping[str, Any], accumulation_steps: int):
    jax, jnp = runtime["jax"], runtime["jnp"]

    @jax.jit
    def step(key, state, batch):
        keys = runtime["random"].split(key, accumulation_steps)
        microbatch = batch["z"].shape[0] // accumulation_steps

        def one_loss(params, index, rng):
            sliced = {
                **batch,
                "z": jax.lax.dynamic_slice_in_dim(batch["z"], index * microbatch, microbatch),
                "f": jax.lax.dynamic_slice_in_dim(batch["f"], index * microbatch, microbatch),
            }
            output = state.apply_fn(
                {"params": params, **state.kwargs}, **sliced, rngs={"extra": rng}
            )
            return output.mse(sliced["f"])

        def body(carry, values):
            index, rng = values
            loss, grads = jax.value_and_grad(one_loss)(state.params, index, rng)
            carry = jax.tree.map(lambda total, value: total + value, carry, grads)
            return carry, loss

        zeros = jax.tree.map(jnp.zeros_like, state.params)
        grads, losses = jax.lax.scan(body, zeros, (jnp.arange(accumulation_steps), keys))
        grads = jax.tree.map(lambda value: value / accumulation_steps, grads)
        return state.apply_gradients(grads=grads), losses.mean()

    return step

def load_predictive_chunk(
    path: Path, expected_draws: int, target_locations: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        if not {"mu", "obs"}.issubset(loaded.files):
            raise RuntimeError(f"Predictive chunk is missing mu/obs: {path}")
        return validate_predictive_arrays(
            np.asarray(loaded["mu"]), np.asarray(loaded["obs"]), expected_draws, target_locations
        )

def nuts_runtime_estimate(
    cfg: Config, diagnostics: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    warmup_per = diagnostics["warmup_seconds"] / cfg.runtime_probe_warmup
    sampling_per = diagnostics["sampling_seconds"] / cfg.runtime_probe_samples
    central = diagnostics["compile_seconds"] + 4000 * warmup_per + 6000 * sampling_per
    now = datetime.now(timezone.utc)
    from datetime import timedelta

    return {
        "status": "RUNTIME_CALIBRATION_NOT_FOR_ANALYSIS",
        "created_at_utc": utc_now(),
        "optimistic_hours": central * 0.85 / 3600,
        "central_hours": central / 3600,
        "conservative_hours": central * 1.30 / 3600,
        "expected_finish_time": (now + timedelta(seconds=central)).isoformat(),
        "warmup_seconds_per_iteration": warmup_per,
        "sampling_seconds_per_iteration": sampling_per,
        "num_steps_distribution": {
            key: diagnostics[key]
            for key in ("num_steps_mean", "num_steps_median", "num_steps_p90", "num_steps_max")
        },
        "estimated_output_size_gib": gib(6000 * N_TARGET * 4 * 3),
        "estimated_peak_memory": gpu_memory(),
        "probe_metrics_excluded_from_formal_tables": True,
    }

def finish_nuts_postprocessing(
    cfg: Config,
    name: str,
    seed: int,
    *,
    probe: bool,
    runtime: Mapping[str, Any],
    decoder,
    checkpoint_info: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    posterior_model,
    predictive_key,
    output_dir: Path,
    samples_chain: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
    raw_path: Path,
) -> dict[str, Any]:
    expected_samples = cfg.runtime_probe_samples if probe else cfg.num_samples
    flat_samples = flatten_posterior_samples(samples_chain, expected_samples)
    scalar = {key: np.asarray(flat_samples[key]).reshape(-1) for key in ("ell", "beta")}

    extraction_started = perf_counter()
    posterior_summary = {
        key: {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
        for key, values in scalar.items()
    }
    extraction_seconds = perf_counter() - extraction_started
    summary_path = output_dir / "posterior_summary.json"
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text())
        if existing_summary.keys() != posterior_summary.keys():
            raise RuntimeError("Existing posterior summary has unexpected sites")
        for key in posterior_summary:
            for statistic, value in posterior_summary[key].items():
                if not np.isclose(existing_summary[key][statistic], value, rtol=1e-6, atol=1e-7):
                    raise RuntimeError(
                        f"Existing posterior summary differs from raw samples: {key}.{statistic}"
                    )
    else:
        write_json(summary_path, posterior_summary)
    if not (output_dir / "trace_and_running_mean.png").exists():
        plot_diagnostics(output_dir, scalar)

    predict = fixed_shape_predictor(runtime, posterior_model, decoder, predictive_key)
    mu, obs, predictive_details = generate_predictive_chunks(
        output_dir,
        flat_samples,
        expected_samples,
        cfg.predictive_batch_size,
        predict,
        {
            "created_at_utc": utc_now(),
            "mode": "nuts_postprocess",
            "model": name,
            "data_seed": seed,
            "raw_posterior_sha256": sha256_file(raw_path),
            "diagnostics_sha256": sha256_file(output_dir / "diagnostics.json"),
            "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
            "checkpoint_integrity_policy": checkpoint_info.get(
                "inference_checkpoint_integrity_policy", "strict_hash"
            ),
            "checkpoint_directory_hash_verified_for_inference": checkpoint_info.get(
                "checkpoint_directory_hash_verified_for_inference", True
            ),
            "checkpoint_restore_validated": checkpoint_info.get(
                "checkpoint_restore_validated", True
            ),
        },
    )
    predictive_seconds = predictive_details["session_seconds"]
    metrics = posterior_metrics(data, mu, obs, scalar, diagnostics)
    known_total = (
        float(diagnostics["compile_seconds"])
        + float(diagnostics["warmup_seconds"])
        + float(diagnostics["sampling_seconds"])
        + extraction_seconds
        + predictive_seconds
    )
    metrics.update(
        {
            "model": name,
            "data_seed": seed,
            "decoder_seed": cfg.decoder_seed,
            "nuts_seed": cfg.nuts_seed,
            "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
            "checkpoint_integrity_policy": checkpoint_info.get(
                "inference_checkpoint_integrity_policy", "strict_hash"
            ),
            "checkpoint_integrity_provenance": checkpoint_info.get(
                "inference_checkpoint_integrity_provenance", ""
            ),
            "checkpoint_directory_hash_verified_for_inference": checkpoint_info.get(
                "checkpoint_directory_hash_verified_for_inference", True
            ),
            "checkpoint_restore_validated": checkpoint_info.get(
                "checkpoint_restore_validated", True
            ),
            "checkpoint_model_signature_validated": checkpoint_info.get(
                "checkpoint_model_signature_validated", True
            ),
            "checkpoint_parameter_count_validated": checkpoint_info.get(
                "checkpoint_parameter_count_validated", True
            ),
            "posterior_extraction_seconds": extraction_seconds,
            "posterior_predictive_seconds": predictive_seconds,
            "posterior_predictive_seconds_scope": "complete_current_session",
            "posterior_predictive_generated_draws": predictive_details["generated_draws"],
            "posterior_predictive_complete_timing_available": True,
            "inference_total_seconds": known_total,
            "inference_total_seconds_is_lower_bound": False,
            "status": "RUNTIME_CALIBRATION_NOT_FOR_ANALYSIS" if probe else diagnostics["diagnostic_status"],
            "probe": probe,
        }
    )
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "metrics.csv", [metrics])
    plot_posterior_maps(output_dir, data, mu, obs)
    completion = {
        "status": metrics["status"],
        "completed_at_utc": utc_now(),
        "model": name,
        "data_seed": seed,
        "probe": probe,
        "data_hashes": {key: array_sha256(data[key]) for key in ("latent_truth", "counts", "mask")},
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "checkpoint_integrity_policy": checkpoint_info.get(
            "inference_checkpoint_integrity_policy", "strict_hash"
        ),
        "checkpoint_integrity_provenance": checkpoint_info.get(
            "inference_checkpoint_integrity_provenance", ""
        ),
        "checkpoint_directory_hash_verified_for_inference": checkpoint_info.get(
            "checkpoint_directory_hash_verified_for_inference", True
        ),
        "checkpoint_restore_validated": checkpoint_info.get(
            "checkpoint_restore_validated", True
        ),
        "raw_posterior_sha256": sha256_file(raw_path),
    }
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing completion marker: {complete_path}")
    write_json(complete_path, completion)
    write_json(
        output_dir / "status.json",
        {"status": metrics["status"], "updated_at_utc": utc_now()},
    )
    if probe:
        estimate = nuts_runtime_estimate(cfg, diagnostics, metrics)
        write_json(resolved_root(cfg) / "runtime_estimates" / "nuts_runtime_estimate.json", estimate)
    return {**completion, "diagnostics": diagnostics, "metrics": metrics}

def resolved_root(cfg: Config) -> Path:
    return Path(cfg.output_root).expanduser()

def parse_model(name: str) -> tuple[str, int]:
    for weighting in WEIGHTINGS:
        prefix = {"bilinear": "Bilinear", "cubic": "Cubic", "dtc": "DTC", "fitc": "FITC"}[weighting]
        if name.startswith(prefix):
            grid = int(name[len(prefix) :])
            if grid not in INDUCING_GRID_SIZES:
                break
            return weighting, grid
    raise ValueError(name)

def write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )

def validate_predictive_arrays(
    mu: np.ndarray, obs: np.ndarray, expected_draws: int, target_locations: int
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu)
    obs = np.asarray(obs)
    expected_shape = (expected_draws, target_locations)
    if mu.shape != expected_shape or obs.shape != expected_shape:
        raise RuntimeError(
            f"Predictive chunk shape mismatch: mu={mu.shape}, obs={obs.shape}, expected={expected_shape}"
        )
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(obs)):
        raise RuntimeError("Predictive chunk contains non-finite values")
    if np.any(obs < 0):
        raise RuntimeError("Predictive count chunk contains negative values")
    return mu, obs

def fixed_shape_predictor(
    runtime: Mapping[str, Any], posterior_model, decoder, predictive_key
):
    jax, jnp, random = runtime["jax"], runtime["jnp"], runtime["random"]

    @jax.jit
    def compiled(key, ell, beta, z):
        posterior_samples = {"ell": ell, "beta": beta, "z": z}
        return runtime["Predictive"](posterior_model, posterior_samples)(
            key, surrogate_decoder=decoder
        )

    def predict(start: int, padded: Mapping[str, np.ndarray]):
        prediction = compiled(
            random.fold_in(predictive_key, start),
            jnp.asarray(padded["ell"]),
            jnp.asarray(padded["beta"]),
            jnp.asarray(padded["z"]),
        )
        jax.block_until_ready(prediction["obs"])
        return {"mu": np.asarray(prediction["mu"]), "obs": np.asarray(prediction["obs"])}

    return predict

def resource_gate(cfg: Config, required_gib: float, stage: str, root: Path) -> bool:
    observation = gpu_memory()
    free_gib = 0.0
    if observation["available"]:
        free_gib = max(device["free_mib"] for device in observation["devices"]) / 1024.0
    passed = observation["available"] and required_gib <= free_gib * cfg.gpu_memory_safety_fraction
    payload = {
        "created_at_utc": utc_now(),
        "stage": stage,
        "required_gib": required_gib,
        "free_gib": free_gib,
        "safety_fraction": cfg.gpu_memory_safety_fraction,
        "gpu": observation,
        "status": "PASS" if passed else "NOT_RUN_RESOURCE_LIMIT",
    }
    write_json(root / "static_audit" / f"resource_gate_{stage}.json", payload)
    return passed

def plot_diagnostics(output_dir: Path, scalar: Mapping[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, name in enumerate(("ell", "beta")):
        values = scalar[name]
        axes[row, 0].plot(values, linewidth=0.5)
        axes[row, 0].set_title(f"{name} trace")
        axes[row, 1].plot(np.cumsum(values) / np.arange(1, len(values) + 1))
        axes[row, 1].set_title(f"{name} running mean")
    fig.savefig(output_dir / "trace_and_running_mean.png", dpi=160)
    plt.close(fig)

def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(available for available in path.rglob("*") if available.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()

def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)

def publish_checkpoint_directory(source: Path, destination: Path) -> str:
    """Copy a completed local checkpoint to persistent storage and verify it.

    Orbax writes only to ``source`` on the runtime-local filesystem.  The
    persistent copy remains under an explicit temporary name until its hash
    matches the local source, so a disconnect cannot make a partial directory
    visible through ``latest.json``.
    """

    if not source.is_dir():
        raise FileNotFoundError(f"Local checkpoint staging directory missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = directory_sha256(source)
    if destination.exists():
        destination_hash = directory_sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                f"Refusing to overwrite existing checkpoint with different contents: {destination}"
            )
        return destination_hash

    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.publish-",
            suffix=".orbax-checkpoint-tmp",
        )
    )
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        persistent_hash = directory_sha256(temporary)
        if persistent_hash != source_hash:
            raise RuntimeError(
                "Checkpoint copy verification failed: "
                f"local={source_hash} persistent={persistent_hash}"
            )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return source_hash

def jax_peak_memory(runtime: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for device in runtime["jax"].devices():
        try:
            stats = device.memory_stats() or {}
        except Exception:
            stats = {}
        rows.append(
            {
                "device": str(device),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                "bytes_in_use": stats.get("bytes_in_use"),
                "bytes_limit": stats.get("bytes_limit"),
            }
        )
    return {"devices": rows, "source": "jax_device_memory_stats"}

def optimizer_for(runtime: Mapping[str, Any], cfg: Config):
    schedule = runtime["cosine_annealing_lr"](cfg.formal_train_steps, cfg.learning_rate)
    return runtime["optax"].chain(
        runtime["optax"].clip_by_global_norm(cfg.gradient_clip_norm),
        runtime["optax"].yogi(schedule),
    )

def generate_predictive_chunks(
    output_dir: Path,
    flat_samples: Mapping[str, np.ndarray],
    total_samples: int,
    batch_size: int,
    predict_fixed_batch,
    manifest_metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("predictive_batch_size must be positive")
    target_locations = int(np.asarray(flat_samples["z"]).shape[-1])
    started = perf_counter()
    mu_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    for start in range(0, total_samples, batch_size):
        stop = min(total_samples, start + batch_size)
        draws = stop - start
        padded: dict[str, np.ndarray] = {}
        for key, values in flat_samples.items():
            chunk = np.asarray(values[start:stop])
            if chunk.shape[0] != draws or draws <= 0:
                raise RuntimeError(f"Invalid posterior slice for {key}: {chunk.shape}")
            if draws < batch_size:
                padding = np.repeat(chunk[-1:], batch_size - draws, axis=0)
                chunk = np.concatenate((chunk, padding), axis=0)
            padded[key] = chunk
        prediction = predict_fixed_batch(start, padded)
        mu_part, obs_part = validate_predictive_arrays(
            np.asarray(prediction["mu"])[:draws],
            np.asarray(prediction["obs"])[:draws],
            draws,
            target_locations,
        )
        del prediction, padded
        gc.collect()
        mu_parts.append(mu_part)
        obs_parts.append(obs_part)

    mu = np.concatenate(mu_parts, axis=0)
    obs = np.concatenate(obs_parts, axis=0)
    if mu.shape != (total_samples, target_locations) or obs.shape != mu.shape:
        raise RuntimeError("Aggregated posterior predictive shape mismatch")
    details = {
        "session_seconds": perf_counter() - started,
        "generated_draws": total_samples,
    }
    return mu, obs, details

def model_name(weighting: str, grid: int) -> str:
    label = {"bilinear": "Bilinear", "cubic": "Cubic", "dtc": "DTC", "fitc": "FITC"}[weighting]
    return f"{label}{grid}"

def model_signature() -> dict[str, Any]:
    report = architecture_report()
    core = {
        "target_grid_size": TARGET_GRID_SIZE,
        "class_name": report["class_name"],
        "num_blocks": report["num_blocks"],
        "parameter_count": report["parameter_count_128x128"],
        "parameter_dtype": report["parameter_dtype"],
        "latent_dimension": N_TARGET,
        "output_field_dimension": N_TARGET,
    }
    core["sha256"] = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    return core

def parse_validation_loss(value: Any) -> float:
    """Parse persisted validation values without reviving non-finite availables."""

    if value in (None, "", "unavailable_nonfinite"):
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan

def build_posterior_model(runtime: Mapping[str, Any], data: Mapping[str, np.ndarray]):
    jnp = runtime["jnp"]
    target_s = jnp.asarray(data["coordinates"])
    priors = {"ls": runtime["dist"].LogNormal(3.0, 0.4), "beta": runtime["dist"].Normal()}

    def posterior_model(surrogate_decoder=None, obs_mask=True, y=None):
        ell = runtime["numpyro"].sample("ell", priors["ls"])
        beta = runtime["numpyro"].sample("beta", priors["beta"])
        z = runtime["numpyro"].sample("z", runtime["dist"].Normal(), sample_shape=(1, N_TARGET))
        mu = runtime["numpyro"].deterministic(
            "mu", surrogate_decoder(z, jnp.asarray([ell]), s=target_s).squeeze()
        )
        with runtime["handlers"].mask(mask=obs_mask):
            runtime["numpyro"].sample("obs", runtime["dist"].Poisson(jnp.exp(beta + mu)), obs=y)

    return posterior_model

def restore_checkpoint(runtime: Mapping[str, Any], checkpoint_dir: Path, model: Any, optimizer: Any):
    latest = json.loads((checkpoint_dir / "latest.json").read_text())
    path = checkpoint_dir / latest["path"]
    if directory_sha256(path) != latest["sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {path}")
    payload = runtime["PyTreeCheckpointer"]().restore(path.absolute())
    actual_parameter_count = sum(
        math.prod(leaf.shape) for leaf in runtime["jax"].tree.leaves(payload["params"])
    )
    if actual_parameter_count != architecture_report()["parameter_count_128x128"]:
        raise RuntimeError(
            f"Checkpoint parameter count {actual_parameter_count} is not target-128."
        )
    signature = payload.get("model_signature")
    if signature and signature.get("sha256") != model_signature()["sha256"]:
        raise RuntimeError("Checkpoint model signature mismatch")
    if payload.get("config", {}).get("target_grid_size") == 64:
        raise RuntimeError("64x64 checkpoint rejected")
    state = runtime["TrainState"].create(
        apply_fn=model.apply, params=payload["params"], kwargs=payload["kwargs"], tx=optimizer
    )
    import grid32_common as base

    opt_state = base.restore_container_structure(state.opt_state, payload["opt_state"])
    state = state.replace(step=payload["step"], opt_state=opt_state)
    return state, payload["best_params"], payload["best_kwargs"], float(payload["best_metric"]), path

def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)

def single_chain_parameter_diagnostics(values: np.ndarray) -> dict[str, Any]:
    from scipy.stats import wasserstein_distance

    values = np.asarray(values, dtype=float).reshape(-1)
    blocks = np.array_split(values, 4)
    summaries = []
    for index, block in enumerate(blocks):
        summaries.append(
            {
                "block": index + 1,
                "mean": float(block.mean()),
                "median": float(np.median(block)),
                "std": float(block.std(ddof=1)),
                "q05": float(np.quantile(block, 0.05)),
                "q95": float(np.quantile(block, 0.95)),
            }
        )
    sd = float(values.std(ddof=1))
    half = values.size // 2
    running = np.cumsum(values) / np.arange(1, values.size + 1)
    autocorrelation = []
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    for lag in (1, 5, 10, 25, 50, 100):
        numerator = float(np.dot(centered[:-lag], centered[lag:])) if lag < len(values) else math.nan
        autocorrelation.append({"lag": lag, "value": numerator / denominator if denominator else math.nan})
    return {
        "blocks": summaries,
        "block_to_block_mean_differences": [summaries[index + 1]["mean"] - summaries[index]["mean"] for index in range(3)],
        "block_mean_range_sd": (max(row["mean"] for row in summaries) - min(row["mean"] for row in summaries)) / sd if sd else math.inf,
        "first_half_second_half_wasserstein": float(wasserstein_distance(values[:half], values[half:])),
        "first_half_second_half_wasserstein_sd": float(wasserstein_distance(values[:half], values[half:]) / sd) if sd else math.inf,
        "running_mean_drift": float(abs(running[-1] - running[half - 1])),
        "running_mean_drift_sd": float(abs(running[-1] - running[half - 1]) / sd) if sd else math.inf,
        "autocorrelation": autocorrelation,
        "lag1_autocorrelation": autocorrelation[0]["value"],
    }

def _make_inducing_teacher_generator(
    runtime: Mapping[str, Any], cfg: Config, name: str, target_s: Any, effective_batch: int
):
    jax, jnp, random, dist = runtime["jax"], runtime["jnp"], runtime["random"], runtime["dist"]
    weighting, size = parse_model(name)
    inducing_s = make_grid(runtime, size)
    m = size**2
    eye_u = jnp.eye(m, dtype=jnp.float32)
    latent_indices = jnp.asarray(nearest_regular_target_indices(size))
    stencil = None
    if weighting in ("bilinear", "cubic"):
        indices, weights, _ = interpolation_stencil(size, weighting)
        stencil = (jnp.asarray(indices), jnp.asarray(weights))
    target_blocks = target_s.reshape(-1, cfg.teacher_target_chunk_size, 2)
    if N_TARGET % cfg.teacher_target_chunk_size:
        raise ValueError("teacher_target_chunk_size must divide 16384")

    @jax.jit
    def generate(key):
        key_ls, key_z = random.split(key)
        ell = dist.LogNormal(3.0, 0.4).sample(key_ls)
        z_h = random.normal(key_z, (effective_batch, N_TARGET), dtype=jnp.float32)
        z_u = z_h[:, latent_indices]
        k_uu = runtime["matern_1_2"](inducing_s, inducing_s, 1.0, ell) + cfg.covariance_jitter * eye_u
        chol_uu = jnp.linalg.cholesky(k_uu)
        u = jnp.einsum("ij,bj->bi", chol_uu, z_u)
        if weighting in ("bilinear", "cubic"):
            indices, weights = stencil
            teacher = jnp.sum(u[:, indices] * weights[None, :, :], axis=-1)
        elif weighting == "dtc":
            whitened = jax.scipy.linalg.solve_triangular(chol_uu.T, z_u.T, lower=False)

            def block_value(block):
                k_hu = runtime["matern_1_2"](block, inducing_s, 1.0, ell)
                return (k_hu @ whitened).T

            teacher = jax.lax.map(block_value, target_blocks)
            teacher = jnp.transpose(teacher, (1, 0, 2)).reshape(effective_batch, N_TARGET)
        elif weighting == "fitc":
            # Use the n-dimensional map described for the FITC teacher.
            k_hh = runtime["matern_1_2"](target_s, target_s, 1.0, ell)
            k_hu = runtime["matern_1_2"](target_s, inducing_s, 1.0, ell)
            solved = jax.scipy.linalg.cho_solve((chol_uu, True), k_hu.T)
            q_hh = k_hu @ solved
            fitc_cov = q_hh + jnp.diag(jnp.diag(k_hh - q_hh))
            fitc_cov = fitc_cov + cfg.covariance_jitter * jnp.eye(N_TARGET, dtype=jnp.float32)
            teacher = jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(fitc_cov), z_h)
        else:
            raise ValueError(weighting)
        return {"s": target_s, "z": z_h, "conditionals": jnp.asarray([ell]), "f": teacher}

    return generate

def prune_checkpoints(checkpoint_dir: Path, cfg: Config) -> None:
    retained = set()
    for pointer in checkpoint_dir.glob("retain_*.json"):
        try:
            retained.add(json.loads(pointer.read_text())["path"])
        except Exception:
            continue
    latest = json.loads((checkpoint_dir / "latest.json").read_text())["path"]
    retained.add(latest)
    periodic = sorted(checkpoint_dir.glob("step_*"), reverse=True)
    retained.update(path.name for path in periodic[: cfg.keep_periodic_checkpoints])
    for path in periodic:
        if path.name not in retained:
            shutil.rmtree(path)

def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()

def generate_seed_payload(runtime: Mapping[str, Any], cfg: Config, factor: Any, seed: int) -> dict[str, np.ndarray]:
    jax, jnp, random = runtime["jax"], runtime["jnp"], runtime["random"]
    rng_data, rng_mask = random.split(random.key(seed))
    rng_mu, rng_poisson = random.split(rng_data)
    latent = factor @ random.normal(rng_mu, (N_TARGET,), dtype=jnp.float32)
    rate = jnp.exp(TRUE_BETA + latent)
    counts = random.poisson(rng_poisson, rate, shape=rate.shape)
    configured = random.choice(rng_mask, N_TARGET, shape=(N_OBSERVED,), replace=False)
    mask = jnp.zeros(N_TARGET, dtype=bool).at[configured].set(True)
    jax.block_until_ready((latent, rate, counts, mask))
    payload = {
        "coordinates": np.asarray(make_grid(runtime, TARGET_GRID_SIZE), dtype=np.float32),
        "latent_truth": np.asarray(latent, dtype=np.float32),
        "log_rate": np.asarray(TRUE_BETA + latent, dtype=np.float32),
        "rate": np.asarray(rate, dtype=np.float32),
        "counts": np.asarray(counts, dtype=np.int32),
        "mask": np.asarray(mask, dtype=np.bool_),
    }
    payload["observed_indices"] = np.flatnonzero(payload["mask"]).astype(np.int64)
    payload["unobserved_indices"] = np.flatnonzero(~payload["mask"]).astype(np.int64)
    validate_seed_payload(payload)
    return payload

def gpu_memory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except Exception:
        return {"available": False, "devices": []}
    devices = []
    for line in result.stdout.splitlines():
        name, total, free, used = [part.strip() for part in line.split(",")]
        devices.append(
            {"name": name, "total_mib": float(total), "free_mib": float(free), "used_mib": float(used)}
        )
    return {"available": bool(devices), "devices": devices}

def interpolation_stencil(size: int, weighting: str) -> tuple[np.ndarray, np.ndarray, int]:
    source_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, size, dtype=np.float64)
    target_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, TARGET_GRID_SIZE, dtype=np.float64)
    target = np.stack(np.meshgrid(target_axis, target_axis, indexing="ij"), axis=-1).reshape(-1, 2)
    index_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []

    def cubic(distance: float, a: float = -0.5) -> float:
        value = abs(distance)
        if value <= 1.0:
            return (a + 2.0) * value**3 - (a + 3.0) * value**2 + 1.0
        if value < 2.0:
            return a * value**3 - 5.0 * a * value**2 + 8.0 * a * value - 4.0 * a
        return 0.0

    for x_raw, y_raw in target:
        tx = (x_raw - source_axis[0]) / (source_axis[1] - source_axis[0])
        ty = (y_raw - source_axis[0]) / (source_axis[1] - source_axis[0])
        if weighting == "bilinear":
            bx = int(np.clip(math.floor(tx), 0, size - 2))
            by = int(np.clip(math.floor(ty), 0, size - 2))
            wx, wy = tx - bx, ty - by
            entries = (
                (bx, by, (1.0 - wx) * (1.0 - wy)),
                (bx + 1, by, wx * (1.0 - wy)),
                (bx, by + 1, (1.0 - wx) * wy),
                (bx + 1, by + 1, wx * wy),
            )
        elif weighting == "cubic":
            bx, by = math.floor(tx), math.floor(ty)
            entries = tuple(
                (
                    int(np.clip(ix, 0, size - 1)),
                    int(np.clip(iy, 0, size - 1)),
                    cubic(tx - ix) * cubic(ty - iy),
                )
                for ix in range(bx - 1, bx + 3)
                for iy in range(by - 1, by + 3)
            )
        else:
            raise ValueError(weighting)
        consolidated: dict[int, float] = {}
        for ix, iy, value in entries:
            index = ix * size + iy
            consolidated[index] = consolidated.get(index, 0.0) + value
        values = list(consolidated.items())
        total = sum(value for _, value in values)
        indices = [index for index, _ in values]
        weights = [value / total for _, value in values]
        width = 4 if weighting == "bilinear" else 16
        indices.extend([indices[-1]] * (width - len(indices)))
        weights.extend([0.0] * (width - len(weights)))
        index_rows.append(indices)
        weight_rows.append(weights)
    indices = np.asarray(index_rows, dtype=np.int32)
    weights = np.asarray(weight_rows, dtype=np.float32)
    nnz = int(np.count_nonzero(weights))
    if indices.shape != (N_TARGET, 4 if weighting == "bilinear" else 16):
        raise AssertionError(indices.shape)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("Interpolation rows do not sum to one")
    return indices, weights, nnz

def parameter_shapes(grid_size: int = TARGET_GRID_SIZE) -> list[dict[str, Any]]:
    n = grid_size**2
    shapes = [
        ("embed/Dense_0/kernel", (4, 64)),
        ("embed/Dense_0/bias", (64,)),
        ("embed/Dense_1/kernel", (64, 64)),
        ("embed/Dense_1/bias", (64,)),
    ]
    for layer_norm in range(3):
        shapes.extend(
            [
                (f"gMLP_0/LayerNorm_{layer_norm}/scale", (64,)),
                (f"gMLP_0/LayerNorm_{layer_norm}/bias", (64,)),
            ]
        )
    for block in range(2):
        prefix = f"gMLP_0/gMLPBlock_{block}"
        shapes.extend(
            [
                (f"{prefix}/SpatialGatingUnit_0/weights", (1, n, n)),
                (f"{prefix}/SpatialGatingUnit_0/bias", (1, 1, n, 1)),
                (f"{prefix}/proj_in/Dense_0/kernel", (64, 128)),
                (f"{prefix}/proj_in/Dense_0/bias", (128,)),
                (f"{prefix}/proj_in/Dense_1/kernel", (128, 128)),
                (f"{prefix}/proj_in/Dense_1/bias", (128,)),
                (f"{prefix}/proj_out/Dense_0/kernel", (64, 64)),
                (f"{prefix}/proj_out/Dense_0/bias", (64,)),
                (f"{prefix}/proj_out/Dense_1/kernel", (64, 64)),
                (f"{prefix}/proj_out/Dense_1/bias", (64,)),
            ]
        )
    shapes.extend(
        [
            ("gMLP_0/gMLPBlock_0/SpatialGatingUnit_0/norm/scale", (64,)),
            ("gMLP_0/gMLPBlock_0/SpatialGatingUnit_0/norm/bias", (64,)),
            ("head/Dense_0/kernel", (64, 128)),
            ("head/Dense_0/bias", (128,)),
            ("head/Dense_1/kernel", (128, 1)),
            ("head/Dense_1/bias", (1,)),
        ]
    )
    return [
        {"name": name, "shape": list(shape), "dtype": "float32", "count": math.prod(shape)}
        for name, shape in shapes
    ]

def lazy_runtime_imports() -> dict[str, Any]:
    import arviz as az
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    import optax
    from flax.training import orbax_utils
    from jax import random
    from numpyro import handlers
    from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
    from orbax.checkpoint import PyTreeCheckpointer

    from dl4bi.core.train import TrainState, cosine_annealing_lr
    from dl4bi.vae import gMLPDeepRV
    from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder
    from dl4bi_sps.kernels import matern_1_2
    from dl4bi_sps.utils import build_grid

    return locals()

def add_diagonal_jitter(jnp: Any, matrix: Any, jitter: float) -> Any:
    indices = jnp.arange(matrix.shape[0])
    return matrix.at[indices, indices].add(jnp.asarray(jitter, dtype=matrix.dtype))

def make_exact_teacher(runtime: Mapping[str, Any], target_s: Any, batch_size: int, jitter: float):
    jax, jnp, random, dist = runtime['jax'], runtime['jnp'], runtime['random'], runtime['dist']
    matern_1_2 = runtime['matern_1_2']
    @jax.jit
    def generate(key):
        key_lengthscale, key_latent = random.split(key)
        lengthscale = dist.LogNormal(3.0, 0.4).sample(key_lengthscale)
        latent = random.normal(key_latent, shape=(batch_size, target_s.shape[0]), dtype=jnp.float32)
        covariance = matern_1_2(target_s, target_s, 1.0, lengthscale)
        covariance = add_diagonal_jitter(jnp, covariance, jitter)
        teacher = jnp.einsum('ij,bj->bi', jnp.linalg.cholesky(covariance), latent)
        return {'s': target_s, 'z': latent, 'conditionals': jnp.asarray([lengthscale]), 'f': teacher}
    return generate

def make_teacher_generator(runtime, cfg, name, target_s, effective_batch):
    if name == 'Exact128':
        return make_exact_teacher(runtime, target_s, effective_batch, cfg.covariance_jitter)
    return _make_inducing_teacher_generator(runtime, cfg, name, target_s, effective_batch)

def generate_public_dataset(cfg: Config, seed: int) -> dict[str, np.ndarray]:
    if seed not in PUBLIC_SEEDS:
        raise ValueError(f'Expected one of {PUBLIC_SEEDS}, received {seed}')
    runtime = lazy_runtime_imports()
    target_s = make_grid(runtime, TARGET_GRID_SIZE)
    covariance = runtime['matern_1_2'](target_s, target_s, cfg.gp_variance, cfg.true_lengthscale)
    covariance = add_diagonal_jitter(runtime['jnp'], covariance, cfg.covariance_jitter)
    factor = runtime['jnp'].linalg.cholesky(covariance)
    return generate_seed_payload(runtime, cfg, factor, seed)

def load_trained_decoder(runtime: Mapping[str, Any], cfg: Config, name: str):
    checkpoint_dir = training_dir(resolved_root(cfg), name) / 'checkpoints'
    latest = json.loads((checkpoint_dir / 'latest.json').read_text())
    path = checkpoint_dir / latest['path']
    if directory_sha256(path) != latest['sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch: {path}')
    payload = runtime['PyTreeCheckpointer']().restore(path.absolute())
    model = runtime['gMLPDeepRV'](num_blks=2)
    state = runtime['TrainState'].create(apply_fn=model.apply, params=payload['best_params'], kwargs=payload['best_kwargs'], tx=optimizer_for(runtime, cfg))
    decoder = runtime['generate_surrogate_decoder'](state, model)
    return decoder, {'checkpoint_sha256': latest['sha256'], 'checkpoint_restore_validated': True}

def diagnostic_gate(cfg: Config, diagnostics: Mapping[str, Any]) -> tuple[str, list[str]]:
    threshold = cfg.single_chain_thresholds
    failures = []
    if diagnostics["divergences"] > threshold["max_divergences"]:
        failures.append(f"divergences={diagnostics['divergences']}")
    if diagnostics["max_depth_hit_fraction"] > threshold["max_depth_hit_fraction"]:
        failures.append(f"max_depth_hit_fraction={diagnostics['max_depth_hit_fraction']}")
    for parameter in ("ell", "beta"):
        if diagnostics[f"bulk_ess_{parameter}"] < threshold["min_bulk_ess_scalar"]:
            failures.append(f"bulk_ess_{parameter}={diagnostics[f'bulk_ess_{parameter}']}")
        if diagnostics[f"tail_ess_{parameter}"] < threshold["min_tail_ess_scalar"]:
            failures.append(f"tail_ess_{parameter}={diagnostics[f'tail_ess_{parameter}']}")
        if (
            not np.isfinite(diagnostics[f"relative_mcse_{parameter}"])
            or diagnostics[f"relative_mcse_{parameter}"] > threshold["max_relative_mcse_scalar"]
        ):
            failures.append(f"relative_mcse_{parameter}={diagnostics[f'relative_mcse_{parameter}']}")
        detail = diagnostics["single_chain"][parameter]
        for key, threshold_key in (
            ("block_mean_range_sd", "max_block_mean_range_sd"),
            ("first_half_second_half_wasserstein_sd", "max_half_wasserstein_sd"),
            ("running_mean_drift_sd", "max_running_mean_drift_sd"),
            ("lag1_autocorrelation", "max_lag1_autocorrelation"),
        ):
            if not np.isfinite(detail[key]) or detail[key] > threshold[threshold_key]:
                failures.append(f"{parameter}.{key}={detail[key]}")
    return ("PASS" if not failures else "FAILED_DIAGNOSTICS"), failures
