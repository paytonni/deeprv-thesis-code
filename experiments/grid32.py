"""Final 32x32 DeepRV comparison using full-domain inducing grids."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dl4bi-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dl4bi-cache")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import wandb
from jax import jit, random
from scipy.optimize import linear_sum_assignment

import grid32_common as base
from dl4bi_sps.kernels import matern_1_2
from model_checks import (
    PriorDiagnosticConfig,
    run_prior_diagnostics,
)


WEIGHTINGS = ("bilinear", "cubic", "dtc", "fitc")


@dataclass(frozen=True)
class Config:
    grid_size: int = 32
    inducing_grid_sizes: tuple[int, ...] = (4, 8, 16)
    weightings: tuple[str, ...] = WEIGHTINGS
    only_models: tuple[str, ...] = ()
    include_full_gp_reference: bool = True
    include_exact: bool = True
    seed: int = 0
    decoder_seed: int = 0
    domain_stop: float = 100.0
    gt_ls: float = 30.0
    beta_true: float = 1.0
    obs_ratio: float = 0.5
    obs_mask_type: str = "uniform"
    train_steps: int = 200_000
    batch_size: int = 32
    validation_interval: int = 500
    validation_batches: int = 500
    checkpoint_save_interval: int = 10_000
    lr: float = 5e-3
    prior_loc: float = 3.0
    prior_scale: float = 0.4
    coverage_level: float = 0.9
    num_chains: int = 2
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    init_to_median_num_samples: int = 10
    initial_warmup: int = 1_000
    initial_samples: int = 4_000
    extended_warmup: int = 1_000
    extended_samples: int = 4_000
    inference_budget: str = "formal"
    max_rhat: float = 1.01
    min_bulk_ess: float = 400.0
    min_tail_ess: float = 400.0
    max_relative_mcse: float = 0.05
    jitter: float = 5e-4
    output_root: str = "outputs/grid32"
    run_name: str = "grid32_thesis_comparison"
    force_rerun: bool = False
    run_truth_metrics: bool = False
    run_prior_diagnostics: bool = False
    prior_diagnostic_ells: tuple[float, ...] = (10.0, 30.0, 50.0)
    prior_diagnostic_samples: int = 32
    prior_diagnostic_z_seed: int = 0


def parse_args(default_grid_size: int | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=default_grid_size or 32)
    parser.add_argument("--inducing-grid-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--weightings", nargs="+", choices=WEIGHTINGS, default=list(WEIGHTINGS))
    parser.add_argument(
        "--only-models",
        nargs="+",
        default=(),
        help=(
            "Optional subset of model names to run after constructing the full "
            "factorial. Use this for targeted reruns while keeping the "
            "decoder-cache naming tied to the full pretraining settings."
        ),
    )
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument(
        "--no-full-gp-reference",
        action="store_false",
        dest="include_full_gp_reference",
        help=(
            "Skip the Full GP posterior reference for an explicitly requested "
            "resource-limited run; "
            "metrics versus Full GP will be absent."
        ),
    )
    parser.add_argument(
        "--seed",
        "--data-mask-seed",
        dest="seed",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Data/mask replicate seed. This controls the synthetic target field, "
            "Poisson observations, observation mask, and posterior inference RNG. "
            "Use --decoder-seed for reusable DeepRV pretraining randomness."
        ),
    )
    parser.add_argument(
        "--decoder-seed",
        type=int,
        default=0,
        help=(
            "Seed used for reusable DeepRV pretraining. Keep this fixed when "
            "rerunning inference with different observation masks."
        ),
    )
    parser.add_argument("--domain-stop", type=float, default=100.0)
    parser.add_argument("--gt-ls", type=float, default=30.0)
    parser.add_argument("--beta-true", type=float, default=1.0)
    parser.add_argument("--obs-ratio", type=float, default=0.5)
    parser.add_argument(
        "--obs-mask-type",
        choices=("spatial", "uniform"),
        default="uniform",
        help="Observation mask design used by the fixed downstream likelihood.",
    )
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=500)
    parser.add_argument("--checkpoint-save-interval", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--prior-loc", type=float, default=3.0)
    parser.add_argument("--prior-scale", type=float, default=0.4)
    parser.add_argument("--coverage-level", type=float, default=0.9)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--init-to-median-num-samples", type=int, default=10)
    parser.add_argument("--initial-warmup", type=int, default=1_000)
    parser.add_argument("--initial-samples", type=int, default=4_000)
    parser.add_argument("--extended-warmup", type=int, default=1_000)
    parser.add_argument("--extended-samples", type=int, default=4_000)
    parser.add_argument(
        "--inference-budget",
        choices=("formal", "manual-rerun"),
        default="formal",
        help=(
            "The default runs the single formal chain set. 'manual-rerun' keeps "
            "the formal output and writes a separate recovery run only after a "
            "failed diagnostic result."
        ),
    )
    parser.add_argument("--max-rhat", type=float, default=1.01)
    parser.add_argument("--min-bulk-ess", type=float, default=400.0)
    parser.add_argument("--min-tail-ess", type=float, default=400.0)
    parser.add_argument("--max-relative-mcse", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=5e-4)
    parser.add_argument("--output-root", default="outputs/grid32")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--run-truth-metrics",
        action="store_true",
        help="Write latent-field, rate, and count metrics against simulated truth.",
    )
    parser.add_argument(
        "--run-prior-diagnostics",
        action="store_true",
        help="Run frozen-decoder prior checks before posterior inference.",
    )
    parser.add_argument(
        "--prior-diagnostic-ells",
        nargs="+",
        type=float,
        default=[10.0, 30.0, 50.0],
    )
    parser.add_argument("--prior-diagnostic-samples", type=int, default=32)
    parser.add_argument("--prior-diagnostic-z-seed", type=int, default=0)
    args = parser.parse_args()
    inducing = args.inducing_grid_sizes
    if inducing is None:
        inducing = [4, 8, 16]
    args.inducing_grid_sizes = tuple(dict.fromkeys(inducing))
    args.weightings = tuple(dict.fromkeys(args.weightings))
    args.only_models = tuple(dict.fromkeys(args.only_models))
    args.prior_diagnostic_ells = tuple(args.prior_diagnostic_ells)
    args.include_exact = not args.no_exact
    del args.no_exact
    mask_label = "" if args.obs_mask_type == "spatial" else f"_{args.obs_mask_type}mask"
    args.run_name = args.run_name or (
        f"target{args.grid_size}{mask_label}_"
        f"inducing{'-'.join(map(str, args.inducing_grid_sizes))}_ls{args.gt_ls:g}"
    )
    cfg = Config(**vars(args))
    if cfg.grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    if any(size < 2 or size >= cfg.grid_size for size in cfg.inducing_grid_sizes):
        raise ValueError("Each inducing grid must be at least 2 and smaller than target.")
    if cfg.prior_diagnostic_samples < 2:
        raise ValueError("prior_diagnostic_samples must be at least 2.")
    return cfg


def cubic_kernel(distance: float, a: float = -0.5) -> float:
    value = abs(distance)
    if value <= 1.0:
        return (a + 2.0) * value**3 - (a + 3.0) * value**2 + 1.0
    if value < 2.0:
        return a * value**3 - 5.0 * a * value**2 + 8.0 * a * value - 4.0 * a
    return 0.0


def cubic_interpolation_matrix(source_s, target_s) -> jax.Array:
    source = np.asarray(source_s)
    target = np.asarray(target_s)
    x_axis = np.unique(source[:, 0])
    y_axis = np.unique(source[:, 1])
    index = {(float(x), float(y)): i for i, (x, y) in enumerate(source)}
    weights = np.zeros((len(target), len(source)), dtype=np.float32)
    dx, dy = x_axis[1] - x_axis[0], y_axis[1] - y_axis[0]
    for row, (x_raw, y_raw) in enumerate(target):
        x = float(np.clip(x_raw, x_axis[0], x_axis[-1]))
        y = float(np.clip(y_raw, y_axis[0], y_axis[-1]))
        tx, ty = (x - x_axis[0]) / dx, (y - y_axis[0]) / dy
        bx, by = int(np.floor(tx)), int(np.floor(ty))
        for raw_ix in range(bx - 1, bx + 3):
            wx = cubic_kernel(tx - raw_ix)
            ix = int(np.clip(raw_ix, 0, len(x_axis) - 1))
            for raw_iy in range(by - 1, by + 3):
                wy = cubic_kernel(ty - raw_iy)
                iy = int(np.clip(raw_iy, 0, len(y_axis) - 1))
                weights[row, index[(float(x_axis[ix]), float(y_axis[iy]))]] += wx * wy
        total = weights[row].sum()
        if not np.isclose(total, 1.0):
            weights[row] /= total
    return jnp.asarray(weights)


def independent_latent_indices(target_s, sample_s) -> jax.Array:
    distances = np.sum(
        (np.asarray(sample_s)[:, None, :] - np.asarray(target_s)[None, :, :]) ** 2,
        axis=-1,
    )
    rows, columns = linear_sum_assignment(distances)
    order = np.argsort(rows)
    return jnp.asarray(columns[order])


def make_teacher_generator(cfg, target_s, sample_s, weighting, priors, batch_size):
    n, m = target_s.shape[0], sample_s.shape[0]
    latent_indices = independent_latent_indices(target_s, sample_s)
    bilinear = base.bilinear_interpolation_matrix(sample_s, target_s)
    cubic = cubic_interpolation_matrix(sample_s, target_s)
    eye_u, eye_h = jnp.eye(m), jnp.eye(n)

    @jit
    def generate(rng_data):
        rng_ls, rng_z = random.split(rng_data)
        ls = priors["ls"].sample(rng_ls)
        z_h = dist.Normal().sample(rng_z, sample_shape=(batch_size, n))
        if weighting == "exact":
            k_hh = matern_1_2(target_s, target_s, 1.0, ls) + cfg.jitter * eye_h
            teacher = jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(k_hh), z_h)
        elif weighting in ("bilinear", "cubic"):
            z_u = z_h[:, latent_indices]
            k_uu = matern_1_2(sample_s, sample_s, 1.0, ls) + cfg.jitter * eye_u
            u = jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(k_uu), z_u)
            matrix = bilinear if weighting == "bilinear" else cubic
            teacher = jnp.einsum("nm,bm->bn", matrix, u)
        elif weighting == "dtc":
            z_u = z_h[:, latent_indices]
            k_uu = matern_1_2(sample_s, sample_s, 1.0, ls) + cfg.jitter * eye_u
            chol_uu = jnp.linalg.cholesky(k_uu)
            k_hu = matern_1_2(target_s, sample_s, 1.0, ls)
            whitened = jax.scipy.linalg.solve_triangular(
                chol_uu.T, z_u.T, lower=False
            )
            teacher = (k_hu @ whitened).T
        elif weighting == "fitc":
            k_hh = matern_1_2(target_s, target_s, 1.0, ls)
            k_uu = matern_1_2(sample_s, sample_s, 1.0, ls) + cfg.jitter * eye_u
            k_hu = matern_1_2(target_s, sample_s, 1.0, ls)
            solved = jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(k_uu), True), k_hu.T)
            q_hh = k_hu @ solved
            fitc = q_hh + jnp.diag(jnp.diag(k_hh - q_hh)) + cfg.jitter * eye_h
            teacher = jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(fitc), z_h)
        else:
            raise ValueError(weighting)
        return {
            "s": target_s,
            "z": z_h,
            "conditionals": jnp.array([ls]),
            "f": teacher,
        }

    return generate


def build_specs(cfg: Config):
    specs = []
    if cfg.include_exact:
        specs.append({"name": "deeprv_exact", "domain": "exact", "grid": cfg.grid_size, "weighting": "exact"})
    for size in cfg.inducing_grid_sizes:
        for weighting in cfg.weightings:
            specs.append(
                {
                    "name": f"deeprv_lowres_{weighting}_{size}x{size}",
                    "domain": "lowres",
                    "grid": size,
                    "weighting": weighting,
                }
            )
    if cfg.only_models:
        available = {spec["name"] for spec in specs}
        missing = sorted(set(cfg.only_models).difference(available))
        if missing:
            raise ValueError(
                "Requested --only-models entries are not in the constructed "
                f"factorial: {missing}. Available: {sorted(available)}"
            )
        wanted = set(cfg.only_models)
        specs = [spec for spec in specs if spec["name"] in wanted]
    return specs


def locations_for_spec(cfg: Config, target_s, spec):
    if spec["domain"] == "exact":
        return target_s
    return base.make_grid(spec["grid"], 0.0, cfg.domain_stop)


def as_base_config(cfg: Config, model_names) -> base.Config:
    models = (*(("full_gp",) if cfg.include_full_gp_reference else ()), *model_names)
    return base.Config(
        seed=cfg.seed,
        grid_size=cfg.grid_size,
        domain_stop=cfg.domain_stop,
        gt_ls=cfg.gt_ls,
        obs_ratio=cfg.obs_ratio,
        obs_mask_type=cfg.obs_mask_type,
        train_steps=cfg.train_steps,
        batch_size=cfg.batch_size,
        validation_interval=cfg.validation_interval,
        validation_batches=cfg.validation_batches,
        num_chains=cfg.num_chains,
        lr=cfg.lr,
        beta_true=cfg.beta_true,
        prior_loc=cfg.prior_loc,
        prior_scale=cfg.prior_scale,
        coverage_level=cfg.coverage_level,
        checkpoint_save_interval=cfg.checkpoint_save_interval,
        target_accept_prob=cfg.target_accept_prob,
        max_tree_depth=cfg.max_tree_depth,
        init_to_median_num_samples=cfg.init_to_median_num_samples,
        initial_warmup=cfg.initial_warmup,
        initial_samples=cfg.initial_samples,
        extended_warmup=cfg.extended_warmup,
        extended_samples=cfg.extended_samples,
        max_rhat=cfg.max_rhat,
        min_bulk_ess=cfg.min_bulk_ess,
        min_tail_ess=cfg.min_tail_ess,
        max_relative_mcse=cfg.max_relative_mcse,
        output_root=cfg.output_root,
        run_name=cfg.run_name,
        models=models,
        force_rerun=cfg.force_rerun,
        run_truth_metrics=cfg.run_truth_metrics,
    )


def run_spec_prior_diagnostics(cfg, seed_dir, target_s, sample_s, spec, decoder):
    """Run one explicitly requested diagnostic without changing checkpoints."""
    interpolation = None
    latent_indices = None
    if spec["weighting"] != "exact":
        latent_indices = independent_latent_indices(target_s, sample_s)
    if spec["weighting"] == "bilinear":
        interpolation = base.bilinear_interpolation_matrix(sample_s, target_s)
    elif spec["weighting"] == "cubic":
        interpolation = cubic_interpolation_matrix(sample_s, target_s)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        seed_dir
        / "prior_diagnostics"
        / spec["name"]
        / f"run_{timestamp}"
    )
    result = run_prior_diagnostics(
        decoder,
        target_s,
        output_dir,
        grid_size=cfg.grid_size,
        method=spec["weighting"],
        config=PriorDiagnosticConfig(
            ell_values=cfg.prior_diagnostic_ells,
            num_samples=cfg.prior_diagnostic_samples,
            z_seed=cfg.prior_diagnostic_z_seed,
            jitter=cfg.jitter,
        ),
        inducing_s=None if spec["weighting"] == "exact" else sample_s,
        interpolation_matrix=interpolation,
        latent_indices=latent_indices,
    )
    with (seed_dir / "prior_diagnostics_index.jsonl").open("a") as handle:
        handle.write(json.dumps(result, default=str, sort_keys=True) + "\n")
    return result


def write_progress(seed_dir, cfg, stage, current=None, completed=None, total=None, note=None, extra=None):
    payload = {
        "updated_at_utc": base.utc_now(),
        "seed": cfg.seed,
        "data_mask_seed": cfg.seed,
        "decoder_seed": cfg.decoder_seed,
        "run_name": cfg.run_name,
        "obs_mask_type": cfg.obs_mask_type,
        "stage": stage,
        "current": current,
        "completed": completed or [],
        "completed_count": len(completed or []),
        "total_count": total,
        "note": note,
    }
    if extra:
        payload.update(extra)
    base.write_json(seed_dir / "run_progress.json", payload)
    with (seed_dir / "progress_events.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(
        "[PROGRESS]",
        f"stage={stage}",
        f"current={current}",
        f"completed={len(completed or [])}/{total if total is not None else '?'}",
        note or "",
        flush=True,
    )


def audit_observation_mask(cfg, seed_dir, data):
    mask = jnp.asarray(data["obs_mask"])
    observed_indices = np.asarray(jnp.argwhere(mask).reshape(-1))
    audit = {
        "created_at_utc": base.utc_now(),
        "seed": cfg.seed,
        "data_mask_seed": cfg.seed,
        "decoder_seed": cfg.decoder_seed,
        "requested_obs_mask_type": cfg.obs_mask_type,
        "saved_obs_mask_type": data.get("obs_mask_type", "spatial"),
        "obs_ratio": cfg.obs_ratio,
        "observed_count": int(mask.sum()),
        "total_count": int(mask.size),
        "expected_observed_count": int(cfg.obs_ratio * mask.size),
        "first_observed_indices": observed_indices[:20].astype(int).tolist(),
    }
    if cfg.obs_mask_type == "uniform":
        _, rng_mask = random.split(random.key(cfg.seed))
        expected = base.gen_uniform_obs_mask(
            rng_mask, (cfg.grid_size, cfg.grid_size), cfg.obs_ratio
        )
        matches = bool(jnp.array_equal(mask, expected))
        audit["matches_regenerated_uniform_mask"] = matches
        if not matches:
            base.write_json(seed_dir / "mask_audit.json", audit)
            raise ValueError(
                "Saved observation mask does not match the deterministic "
                "uniform random mask for this seed/config."
            )
    base.write_json(seed_dir / "mask_audit.json", audit)
    print("Mask audit:", json.dumps(audit, indent=2))
    return audit


def train_spec(cfg, base_cfg, seed_dir, target_s, priors, spec):
    model_name = spec["name"]
    model_dir = decoder_model_dir(cfg, spec)
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_s = locations_for_spec(cfg, target_s, spec)
    train_base_cfg = replace(base_cfg, seed=cfg.decoder_seed)
    original = base.make_batch_generator
    setup_timing = {"teacher_generator_setup_time": 0.0}

    def timed_teacher_generator(target, sample, prior, batch):
        start = perf_counter()
        generator = make_teacher_generator(
            cfg, target, sample, spec["weighting"], prior, batch
        )
        setup_timing["teacher_generator_setup_time"] += perf_counter() - start
        return generator

    base.make_batch_generator = timed_teacher_generator
    try:
        decoder, result = base.train_deeprv(
            train_base_cfg, model_name, target_s, sample_s, priors, model_dir
        )
    finally:
        base.make_batch_generator = original
    result.update(
        {
            "decoder_seed": cfg.decoder_seed,
            "decoder_cache_dir": str(model_dir.parent),
            "teacher_weighting": spec["weighting"],
            "teacher_domain": spec["domain"],
            "teacher_grid_size": spec["grid"],
            "teacher_generator_setup_time": setup_timing[
                "teacher_generator_setup_time"
            ],
            "inducing_grid_definition": (
                "U = regular full-domain inducing grid"
                if spec["domain"] == "lowres"
                else None
            ),
            "local_latent_noise": None,
            "fitc_factorization": "dense_cholesky" if spec["weighting"] == "fitc" else None,
        }
    )
    base.write_json(model_dir / "training_result.json", result)
    return decoder, result, sample_s


def decoder_cache_dir(cfg: Config) -> Path:
    inducing = "-".join(map(str, cfg.inducing_grid_sizes))
    weightings = "-".join(cfg.weightings)
    name = (
        f"target{cfg.grid_size}_decoderseed{cfg.decoder_seed}_"
        f"inducing{inducing}_domainfull_weightings{weightings}_"
        f"ls{cfg.gt_ls:g}_steps{cfg.train_steps}"
    )
    return Path(cfg.output_root).expanduser() / "_decoder_cache" / name


def decoder_model_dir(cfg: Config, spec) -> Path:
    return decoder_cache_dir(cfg) / "training" / spec["name"]


def write_decoder_cache_manifest(cfg: Config, specs) -> None:
    path = decoder_cache_dir(cfg) / "decoder_cache_manifest.json"
    payload = {
        "updated_at_utc": base.utc_now(),
        "decoder_seed": cfg.decoder_seed,
        "grid_size": cfg.grid_size,
        "domain_stop": cfg.domain_stop,
        "gt_ls": cfg.gt_ls,
        "train_steps": cfg.train_steps,
        "batch_size": cfg.batch_size,
        "validation_interval": cfg.validation_interval,
        "validation_batches": cfg.validation_batches,
        "checkpoint_save_interval": cfg.checkpoint_save_interval,
        "prior_loc": cfg.prior_loc,
        "prior_scale": cfg.prior_scale,
        "jitter": cfg.jitter,
        "inducing_grid_sizes": list(cfg.inducing_grid_sizes),
        "domain": "full",
        "weightings": list(cfg.weightings),
        "only_models": list(cfg.only_models),
        "include_full_gp_reference": cfg.include_full_gp_reference,
        "include_exact": cfg.include_exact,
        "specs": specs,
        "reuse_note": (
            "These DeepRV decoder checkpoints are independent of the "
            "observation mask. Reuse them with a different run_name and "
            "obs_mask_type by keeping decoder_seed and pretraining settings fixed."
        ),
    }
    base.write_json(path, payload)


def main(default_grid_size: int | None = None):
    cfg = parse_args(default_grid_size)
    specs = build_specs(cfg)
    names = [spec["name"] for spec in specs]
    base_cfg = as_base_config(cfg, names)
    write_decoder_cache_manifest(cfg, specs)
    base.MODELS = base_cfg.models
    base.MODEL_IDS = {name: index for index, name in enumerate(base_cfg.models)}
    numpyro.set_host_device_count(cfg.num_chains)
    wandb.init(mode="disabled")
    run_dir, seed_dir = base.prepare_dirs(base_cfg)
    log_file = (seed_dir / "console.log").open("a", buffering=1)
    sys.stdout = base.Tee(sys.__stdout__, log_file)
    sys.stderr = base.Tee(sys.__stderr__, log_file)
    config_payload = asdict(cfg)
    config_payload["data_mask_seed"] = cfg.seed
    base.write_json(seed_dir / "weighting_config.json", config_payload)
    print("Weighting config:", json.dumps(config_payload, indent=2))
    print(
        "Seed roles: data_mask_seed="
        f"{cfg.seed} controls target data, uniform observation mask, and MCMC RNG; "
        f"decoder_seed={cfg.decoder_seed} controls reusable DeepRV pretraining."
    )
    print(
        "Lowres inducing design: U = regular full-domain inducing grid "
        "on [0, domain_stop]^2; U is fixed, not learned, and not selected from data."
    )
    print("Decoder cache:", decoder_cache_dir(cfg))
    print(
        "Reuse contract: train decoders once with decoder_seed="
        f"{cfg.decoder_seed}; different masks reuse the same decoder checkpoints "
        "and rerun inference only."
    )
    completed = []
    total_units = (1 if cfg.include_full_gp_reference else 0) + len(specs)
    write_progress(
        seed_dir,
        cfg,
        "started",
        total=total_units,
        note="Preparing target grid, data, decoder cache, and inference queue.",
        extra={"decoder_cache_dir": str(decoder_cache_dir(cfg))},
    )
    target_s = base.make_grid(cfg.grid_size, 0.0, cfg.domain_stop)
    data = base.load_or_create_data(base_cfg, seed_dir, target_s)
    audit_observation_mask(cfg, seed_dir, data)
    priors = {
        "ls": dist.LogNormal(cfg.prior_loc, cfg.prior_scale),
        "beta": dist.Normal(0.0, 1.0),
    }
    infer_model = base.inference_model(target_s, priors)

    reference = None
    if cfg.include_full_gp_reference:
        write_progress(
            seed_dir,
            cfg,
            "inference",
            current="full_gp initial",
            completed=completed,
            total=total_units,
            note="Running or restoring Full GP initial posterior inference.",
        )
        print("\n[INFER] full_gp initial")
        full = base.run_one_inference(
            base_cfg, seed_dir, "initial", "full_gp", data, infer_model,
            None, None, None, None
        )
        if (
            cfg.inference_budget == "manual-rerun"
            and not full[2]["diagnostics_passed"]
        ):
            write_progress(
                seed_dir,
                cfg,
                "inference",
                current="full_gp extended",
                completed=completed,
                total=total_units,
                note="User-requested recovery run after failed formal diagnostics.",
            )
            print("[INFER] full_gp extended")
            full = base.run_one_inference(
                base_cfg, seed_dir, "extended", "full_gp", data, infer_model,
                None, None, None, None
            )
        reference = (full[0], full[1])
        completed.append("full_gp")
        write_progress(
            seed_dir,
            cfg,
            "completed_model",
            current="full_gp",
            completed=completed,
            total=total_units,
            note="Full GP reference complete for this mask/data seed.",
        )
    else:
        print("\n[INFER] full_gp skipped: --no-full-gp-reference")

    for spec in specs:
        model_name = spec["name"]
        model_dir = decoder_model_dir(cfg, spec)
        train_result = model_dir / "training_result.json"
        latest_checkpoint = model_dir / "checkpoints" / "latest.json"
        reuse_status = (
            "reuse completed decoder"
            if train_result.exists() and latest_checkpoint.exists()
            else "train or resume decoder"
        )
        write_progress(
            seed_dir,
            cfg,
            "training",
            current=model_name,
            completed=completed,
            total=total_units,
            note=reuse_status,
            extra={"decoder_model_dir": str(model_dir)},
        )
        print(f"\n[TRAIN/RESTORE] {model_name}: {reuse_status}")
        decoder, training, sample_s = train_spec(
            cfg, base_cfg, seed_dir, target_s, priors, spec
        )
        if cfg.run_prior_diagnostics:
            write_progress(
                seed_dir,
                cfg,
                "prior_diagnostics",
                current=model_name,
                completed=completed,
                total=total_units,
                note="Running explicitly requested Phase-1 frozen-prior checks.",
            )
            prior_result = run_spec_prior_diagnostics(
                cfg, seed_dir, target_s, sample_s, spec, decoder
            )
            print(
                f"[PRIOR] {model_name}: {prior_result['status']} -> "
                f"{prior_result['output_dir']}"
            )
        write_progress(
            seed_dir,
            cfg,
            "inference",
            current=f"{model_name} formal",
            completed=completed,
            total=total_units,
            note="Running or restoring the formal posterior inference.",
            extra={"decoder_model_dir": str(model_dir)},
        )
        print(f"[INFER] {model_name} formal")
        result = base.run_one_inference(
            base_cfg, seed_dir, "initial", model_name, data, infer_model,
            decoder, training, sample_s, reference
        )
        if (
            cfg.inference_budget == "manual-rerun"
            and not result[2]["diagnostics_passed"]
        ):
            write_progress(
                seed_dir,
                cfg,
                "inference",
                current=f"{model_name} manual recovery",
                completed=completed,
                total=total_units,
                note="User-requested recovery run; the formal run is retained separately.",
                extra={"decoder_model_dir": str(model_dir)},
            )
            print(f"[INFER] {model_name} manual recovery")
            base.run_one_inference(
                base_cfg, seed_dir, "extended", model_name, data, infer_model,
                decoder, training, sample_s, reference
            )
        completed.append(model_name)
        write_progress(
            seed_dir,
            cfg,
            "completed_model",
            current=model_name,
            completed=completed,
            total=total_units,
            note="Model complete for this mask/data seed.",
            extra={"decoder_model_dir": str(model_dir)},
        )
        base.refresh_metrics(run_dir, seed_dir)
    base.refresh_metrics(run_dir, seed_dir)
    write_progress(
        seed_dir,
        cfg,
        "complete",
        completed=completed,
        total=total_units,
        note="All queued model inference completed for this seed.",
        extra={"combined_metrics": str(run_dir / "combined_metrics.csv")},
    )
    print("Complete:", run_dir / "combined_metrics.csv")


if __name__ == "__main__":
    main()
