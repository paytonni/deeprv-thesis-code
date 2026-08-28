"""Poisson-GP benchmark for SKI, DTC/SoR, and FITC priors.

Run from the repository root.  This script intentionally leaves the existing
DeepRV exact/lowres/local implementations unchanged.  It uses the same seeded
data and mask construction as ``paperlike_32x32_poisson_gp_pilot.py`` and adds
direct approximate-GP priors for comparison with the saved DeepRV runs.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import pickle
import platform
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dl4bi-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dl4bi-cache")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from scipy.special import gammaln, logsumexp

import grid32_common as base
from dl4bi_sps.kernels import matern_1_2


METHODS = (
    "full_gp",
    "kissgp_ski_bilinear",
    "kissgp_ski_cubic",
    "dtc_sor",
    "fitc",
)


class Tee:
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, text):
        alive = []
        for stream in self.streams:
            try:
                stream.write(text)
                stream.flush()
                alive.append(stream)
            except OSError:
                # Colab Drive-backed streams can disappear during long tqdm
                # updates. Drop the broken stream instead of aborting NUTS.
                continue
        self.streams = alive
        return len(text)

    def flush(self):
        alive = []
        for stream in self.streams:
            try:
                stream.flush()
                alive.append(stream)
            except OSError:
                continue
        self.streams = alive


def local_console_log_path(cfg: "Config") -> Path:
    log_root = Path(os.environ.get("DL4BI_LOCAL_LOG_DIR", "/tmp/dl4bi-directgp-logs"))
    log_root.mkdir(parents=True, exist_ok=True)
    safe_run = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in cfg.run_name
    )
    return log_root / f"{safe_run}_seed{cfg.seed}_{os.getpid()}.log"


def append_local_console_log(local_path: Path, seed_dir: Path) -> None:
    if not local_path.exists():
        return
    destination = seed_dir / "console.log"
    try:
        with local_path.open("r") as source, destination.open("a") as target:
            target.write("\n\n===== copied from local Colab log =====\n")
            target.write(source.read())
    except OSError as exc:
        print(f"Warning: could not copy console log to {destination}: {exc}")


@dataclass(frozen=True)
class Config:
    grid_size: int
    inducing_grid_size: int = 8
    seed: int = 0
    domain_stop: float = 100.0
    gt_ls: float = 30.0
    beta_true: float = 1.0
    obs_ratio: float = 0.5
    obs_mask_type: str = "spatial"
    prior_loc: float = 3.0
    prior_scale: float = 0.4
    coverage_level: float = 0.9
    num_chains: int = 2
    target_accept: float = 0.8
    max_tree_depth: int = 10
    initial_warmup: int = 1_000
    initial_samples: int = 4_000
    extended_warmup: int = 1_000
    extended_samples: int = 4_000
    max_rhat: float = 1.01
    min_bulk_ess: float = 400.0
    min_tail_ess: float = 400.0
    max_relative_mcse: float = 0.05
    covariance_jitter: float = 5e-4
    methods: tuple[str, ...] = METHODS
    require_full_reference: bool = True
    reference_run_name: str | None = None
    output_root: str = "outputs/poisson_gp_inducing_benchmark"
    run_name: str = "poisson_gp_inducing"
    force_rerun: bool = False
    run_truth_metrics: bool = False


def parse_args(default_grid_size: int | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=default_grid_size or 16)
    parser.add_argument("--inducing-grid-size", type=int, default=8)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--domain-stop", type=float, default=100.0)
    parser.add_argument("--gt-ls", type=float, default=30.0)
    parser.add_argument("--beta-true", type=float, default=1.0)
    parser.add_argument("--obs-ratio", type=float, default=0.5)
    parser.add_argument(
        "--obs-mask-type",
        choices=("spatial", "uniform"),
        default="spatial",
        help="Observation mask design used by the fixed likelihood.",
    )
    parser.add_argument("--prior-loc", type=float, default=3.0)
    parser.add_argument("--prior-scale", type=float, default=0.4)
    parser.add_argument("--coverage-level", type=float, default=0.9)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--initial-warmup", type=int, default=1_000)
    parser.add_argument("--initial-samples", type=int, default=4000)
    parser.add_argument("--extended-warmup", type=int, default=1000)
    parser.add_argument("--extended-samples", type=int, default=4000)
    parser.add_argument("--max-rhat", type=float, default=1.01)
    parser.add_argument("--min-bulk-ess", type=float, default=400.0)
    parser.add_argument("--min-tail-ess", type=float, default=400.0)
    parser.add_argument("--max-relative-mcse", type=float, default=0.05)
    parser.add_argument("--covariance-jitter", type=float, default=5e-4)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--reference-run-name",
        default=None,
        help=(
            "Optional run_name containing a completed full_gp reference for the "
            "same seed. This lets approximate methods reuse one full GP run "
            "instead of rerunning full_gp for every inducing grid."
        ),
    )
    parser.add_argument(
        "--allow-missing-full-reference",
        action="store_false",
        dest="require_full_reference",
        help=(
            "Allow approximate direct-GP methods to run without full_gp in the "
            "same run. Metrics versus Full GP will be absent. This is intended "
            "for 64x64/128x128 cost pilots."
        ),
    )
    parser.add_argument(
        "--output-root", default="outputs/poisson_gp_inducing_benchmark"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--run-truth-metrics",
        action="store_true",
        help="Write latent-field, rate, and count metrics against simulated truth.",
    )
    args = parser.parse_args()
    args.methods = tuple(dict.fromkeys(args.methods))
    args.run_name = args.run_name or (
        f"target{args.grid_size}_inducing{args.inducing_grid_size}_ls{args.gt_ls:g}"
    )
    cfg = Config(**vars(args))
    if cfg.grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    if not 2 <= cfg.inducing_grid_size < cfg.grid_size:
        raise ValueError("inducing_grid_size must be at least 2 and below grid_size.")
    if (
        cfg.require_full_reference
        and "full_gp" not in cfg.methods
        and cfg.reference_run_name is None
    ):
        raise ValueError(
            "Include full_gp, pass --reference-run-name, or pass "
            "--allow-missing-full-reference for large-grid cost pilots."
        )
    return cfg


def as_base_config(cfg: Config) -> base.Config:
    """Supply the fields needed by the existing data and metric utilities."""
    return base.Config(
        seed=cfg.seed,
        grid_size=cfg.grid_size,
        domain_stop=cfg.domain_stop,
        gt_ls=cfg.gt_ls,
        obs_ratio=cfg.obs_ratio,
        obs_mask_type=cfg.obs_mask_type,
        num_chains=cfg.num_chains,
        initial_warmup=cfg.initial_warmup,
        initial_samples=cfg.initial_samples,
        extended_warmup=cfg.extended_warmup,
        extended_samples=cfg.extended_samples,
        max_rhat=cfg.max_rhat,
        min_bulk_ess=cfg.min_bulk_ess,
        min_tail_ess=cfg.min_tail_ess,
        max_relative_mcse=cfg.max_relative_mcse,
        beta_true=cfg.beta_true,
        prior_loc=cfg.prior_loc,
        prior_scale=cfg.prior_scale,
        coverage_level=cfg.coverage_level,
        output_root=cfg.output_root,
        run_name=cfg.run_name,
        force_rerun=cfg.force_rerun,
        run_truth_metrics=cfg.run_truth_metrics,
    )


def make_direct_gp_model(
    method: str,
    target_s: jax.Array,
    inducing_s: jax.Array,
    interpolations: dict[str, jax.Array],
    cfg: Config,
):
    n = target_s.shape[0]
    m = inducing_s.shape[0]
    eye_h = jnp.eye(n)
    eye_u = jnp.eye(m)

    def model(surrogate_decoder=None, obs_mask=True, y=None):
        del surrogate_decoder
        ls = numpyro.sample("ls", dist.LogNormal(cfg.prior_loc, cfg.prior_scale))
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        if method == "full_gp":
            z_h = numpyro.sample("z_h", dist.Normal(0.0, 1.0).expand((n,)))
            k_hh = matern_1_2(target_s, target_s, 1.0, ls)
            mu = jnp.linalg.cholesky(k_hh + cfg.covariance_jitter * eye_h) @ z_h
        else:
            z_u = numpyro.sample("z_u", dist.Normal(0.0, 1.0).expand((m,)))
            k_uu = matern_1_2(inducing_s, inducing_s, 1.0, ls)
            chol_uu = jnp.linalg.cholesky(k_uu + cfg.covariance_jitter * eye_u)
            if method in ("kissgp_ski_bilinear", "kissgp_ski_cubic"):
                interpolation = interpolations[method]
                mu = interpolation @ (chol_uu @ z_u)
            else:
                k_hu = matern_1_2(target_s, inducing_s, 1.0, ls)
                conditional_component = k_hu @ jax.scipy.linalg.solve_triangular(
                    chol_uu.T, z_u, lower=False
                )
                if method == "dtc_sor":
                    mu = conditional_component
                elif method == "fitc":
                    z_residual = numpyro.sample(
                        "z_residual", dist.Normal(0.0, 1.0).expand((n,))
                    )
                    projected = jax.scipy.linalg.solve_triangular(
                        chol_uu, k_hu.T, lower=True
                    )
                    q_diag = jnp.sum(projected**2, axis=0)
                    residual_diag = jnp.clip(1.0 - q_diag, min=0.0)
                    mu = conditional_component + jnp.sqrt(
                        residual_diag + cfg.covariance_jitter
                    ) * z_residual
                else:
                    raise ValueError(f"Unknown method: {method}")
        mu = numpyro.deterministic("mu", mu)
        with handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(jnp.exp(beta + mu)), obs=y)

    return model


def cubic_kernel(distance: float, a: float = -0.5) -> float:
    value = abs(distance)
    if value <= 1.0:
        return (a + 2.0) * value**3 - (a + 3.0) * value**2 + 1.0
    if value < 2.0:
        return a * value**3 - 5.0 * a * value**2 + 8.0 * a * value - 4.0 * a
    return 0.0


def cubic_interpolation_matrix(source_s: jax.Array, target_s: jax.Array) -> jax.Array:
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


def budget_values(cfg: Config, budget: str) -> tuple[int, int]:
    if budget == "initial":
        return cfg.initial_warmup, cfg.initial_samples
    if budget == "extended":
        return cfg.extended_warmup, cfg.extended_samples
    raise ValueError(budget)


def poisson_lpd(
    samples_chain: dict,
    posterior: dict,
    y: jax.Array,
    mask: jax.Array,
) -> dict[str, float]:
    mu = np.asarray(posterior["mu"])
    beta = np.asarray(samples_chain["beta"]).reshape(-1, 1)
    rates = np.exp(beta + mu)
    y_np = np.asarray(y)[None, :]
    log_prob = y_np * np.log(rates) - rates - gammaln(y_np + 1.0)
    point_lpd = logsumexp(log_prob, axis=0) - np.log(log_prob.shape[0])
    mask_np = np.asarray(mask, dtype=bool)
    return {
        "mean_log_predictive_density_all": float(point_lpd.mean()),
        "mean_log_predictive_density_observed": float(point_lpd[mask_np].mean()),
        "mean_log_predictive_density_unobserved": float(point_lpd[~mask_np].mean()),
    }


def representation_bytes(method: str, n: int, m: int) -> int:
    float_bytes = 4  # JAX defaults to float32 in the experiment environment.
    elements = {
        "full_gp": n * n,
        "kissgp_ski_bilinear": n * m + m * m,
        "kissgp_ski_cubic": n * m + m * m,
        "dtc_sor": n * m + m * m,
        "fitc": n * m + m * m + n,
    }[method]
    return elements * float_bytes


def output_dir(seed_dir: Path, budget: str, method: str) -> Path:
    return seed_dir / "inference" / budget / method


def load_result(path: Path):
    required = (
        path / "complete.json",
        path / "posterior_samples_by_chain.pkl",
        path / "posterior_predictive.npz",
        path / "metrics.json",
    )
    if not all(item.exists() for item in required):
        return None
    with (path / "posterior_samples_by_chain.pkl").open("rb") as handle:
        samples = pickle.load(handle)
    predictive = np.load(path / "posterior_predictive.npz")
    posterior = {"obs": predictive["obs"], "mu": predictive["mu"]}
    metrics = json.loads((path / "metrics.json").read_text())
    diagnostics = json.loads((path / "diagnostics.json").read_text())
    return samples, posterior, metrics, diagnostics


def load_full_reference(seed_dir: Path):
    """Load the best available full-GP reference from a seed directory."""
    for budget in ("extended", "initial"):
        loaded = load_result(output_dir(seed_dir, budget, "full_gp"))
        if loaded is None:
            continue
        diagnostics = loaded[3]
        if budget == "extended" or diagnostics.get("diagnostics_passed", False):
            print(f"Loaded Full GP reference: {output_dir(seed_dir, budget, 'full_gp')}")
            return loaded
    fallback = load_result(output_dir(seed_dir, "initial", "full_gp"))
    if fallback is not None:
        print(
            "Loaded initial Full GP reference with failed or missing diagnostics: "
            f"{output_dir(seed_dir, 'initial', 'full_gp')}"
        )
    return fallback


def run_hmc_diagnostic(
    cfg: Config,
    budget: str,
    rng: jax.Array,
    model,
    data: dict,
    surrogate_decoder=None,
):
    """Run NUTS while retaining inducing/residual draws for prediction."""
    warmup, draws = budget_values(cfg, budget)
    mcmc = MCMC(
        NUTS(
            model,
            init_strategy=init_to_median(num_samples=10),
            target_accept_prob=cfg.target_accept,
            max_tree_depth=cfg.max_tree_depth,
        ),
        num_chains=cfg.num_chains,
        num_warmup=warmup,
        num_samples=draws,
        progress_bar=True,
    )
    rng_run, rng_predictive = random.split(rng)
    start = perf_counter()
    mcmc.run(
        rng_run,
        surrogate_decoder=surrogate_decoder,
        obs_mask=data["obs_mask"],
        y=data["y_full"],
        extra_fields=("diverging",),
    )
    infer_time = perf_counter() - start
    all_chain = mcmc.get_samples(group_by_chain=True)
    latent_chain = {
        name: value
        for name, value in all_chain.items()
        if name not in ("mu", "obs")
    }
    latent_flat = {
        name: value.reshape((-1,) + value.shape[2:])
        for name, value in latent_chain.items()
    }
    predictive_start = perf_counter()
    posterior = Predictive(model, latent_flat)(
        rng_predictive, surrogate_decoder=surrogate_decoder
    )
    jax.block_until_ready(posterior["obs"])
    posterior_predictive_time = perf_counter() - predictive_start
    samples_chain = {name: latent_chain[name] for name in ("ls", "beta")}
    inference_data = base.az.from_dict(
        posterior={name: np.asarray(value) for name, value in samples_chain.items()}
    )
    rhat = base.az.rhat(inference_data)
    ess_bulk = base.az.ess(inference_data, method="bulk")
    ess_tail = base.az.ess(inference_data, method="tail")
    mcse = base.az.mcse(inference_data, method="mean")
    posterior_sd = {
        name: float(np.asarray(value).std(ddof=1))
        for name, value in samples_chain.items()
    }
    worst = base.identify_worst_chain(samples_chain)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    diagnostics = {
        "mcmc_warmup": warmup,
        "mcmc_samples_per_chain": draws,
        "num_chains": cfg.num_chains,
        "num_divergences": int(np.asarray(extra["diverging"]).sum()),
        "nuts_inference_time": infer_time,
        "posterior_predictive_time": posterior_predictive_time,
        "inference_total_time": infer_time + posterior_predictive_time,
        "rhat_ls": base.scalar_stat(rhat, "rhat", "ls"),
        "rhat_beta": base.scalar_stat(rhat, "rhat", "beta"),
        "ess_bulk_ls": base.scalar_stat(ess_bulk, "ess_bulk", "ls"),
        "ess_bulk_beta": base.scalar_stat(ess_bulk, "ess_bulk", "beta"),
        "ess_tail_ls": base.scalar_stat(ess_tail, "ess_tail", "ls"),
        "ess_tail_beta": base.scalar_stat(ess_tail, "ess_tail", "beta"),
        "mcse_mean_ls": base.scalar_stat(mcse, "mcse", "ls"),
        "mcse_mean_beta": base.scalar_stat(mcse, "mcse", "beta"),
        "posterior_sd_ls": posterior_sd["ls"],
        "posterior_sd_beta": posterior_sd["beta"],
        "worst_chain": worst,
    }
    for name in ("ls", "beta"):
        sd = diagnostics[f"posterior_sd_{name}"]
        diagnostics[f"relative_mcse_{name}"] = (
            diagnostics[f"mcse_mean_{name}"] / sd if sd > 0 else None
        )
    mu_chain = np.asarray(posterior["mu"]).reshape(cfg.num_chains, draws, -1)
    mu_data = base.az.from_dict(posterior={"mu": mu_chain})
    mu_rhat = np.asarray(base.az.rhat(mu_data)["mu"]).reshape(-1)
    mu_bulk = np.asarray(base.az.ess(mu_data, method="bulk")["mu"]).reshape(-1)
    mu_tail = np.asarray(base.az.ess(mu_data, method="tail")["mu"]).reshape(-1)
    diagnostics.update(
        {
            "latent_mu_rhat_max": float(np.nanmax(mu_rhat)),
            "latent_mu_rhat_q99": float(np.nanquantile(mu_rhat, 0.99)),
            "latent_mu_rhat_over_threshold": int(np.sum(mu_rhat > cfg.max_rhat)),
            "latent_mu_bulk_ess_min": float(np.nanmin(mu_bulk)),
            "latent_mu_bulk_ess_q01": float(np.nanquantile(mu_bulk, 0.01)),
            "latent_mu_tail_ess_min": float(np.nanmin(mu_tail)),
            "latent_mu_tail_ess_q01": float(np.nanquantile(mu_tail, 0.01)),
        }
    )
    return samples_chain, {"obs": posterior["obs"], "mu": posterior["mu"]}, infer_time, diagnostics


def run_method(
    cfg: Config,
    base_cfg: base.Config,
    budget: str,
    method: str,
    model,
    data: dict,
    inducing_s: jax.Array,
    setup_timings: dict,
    full_reference,
):
    destination = output_dir(
        Path(cfg.output_root) / cfg.run_name / f"seed_{cfg.seed}", budget, method
    )
    existing = load_result(destination)
    if existing is not None and not cfg.force_rerun:
        print(f"{budget}/{method}: complete output exists; skipping")
        return existing
    if cfg.force_rerun and destination.exists():
        base.reset_output(destination)
    destination.mkdir(parents=True, exist_ok=True)
    key = random.fold_in(random.key(cfg.seed + 4_000_000), METHODS.index(method))
    if budget == "extended":
        key = random.fold_in(key, 1)
    samples_chain, posterior_small, infer_time, diagnostics = run_hmc_diagnostic(
        cfg, budget, key, model, data
    )
    diagnostics.update(base.assess_diagnostics(base_cfg, diagnostics))
    samples_flat = {
        name: value.reshape((-1,) + value.shape[2:])
        for name, value in samples_chain.items()
    }
    reference = None
    if full_reference is not None:
        ref_samples, ref_posterior = full_reference
        reference = (
            {
                name: value.reshape((-1,) + value.shape[2:])
                for name, value in ref_samples.items()
            },
            ref_posterior,
        )
    metric_cfg = base_cfg.__class__(
        **{
            **asdict(base_cfg),
            "mcmc_warmup": diagnostics["mcmc_warmup"],
            "mcmc_samples": diagnostics["mcmc_samples_per_chain"],
        }
    )
    ess = {
        "ls": diagnostics["ess_bulk_ls"],
        "beta": diagnostics["ess_bulk_beta"],
    }
    metrics = base.summarize_model(
        metric_cfg,
        method,
        None if method == "full_gp" else inducing_s,
        data,
        samples_flat,
        posterior_small,
        infer_time,
        ess,
        {"num_divergences": diagnostics["num_divergences"]},
        None,
        reference,
    )
    metrics.update(poisson_lpd(samples_chain, posterior_small, data["y_full"], data["obs_mask"]))
    metrics.update(
        {
            "budget": budget,
            "target_grid_size": cfg.grid_size,
            "inducing_grid_size": cfg.inducing_grid_size,
            "obs_mask_type": cfg.obs_mask_type,
            "nuts_inference_time": infer_time,
            "posterior_predictive_time": diagnostics.get(
                "posterior_predictive_time"
            ),
            "inference_total_time": diagnostics.get("inference_total_time"),
            "direct_gp_precompute_time": setup_timings["setup_total_time"],
            "a_matrix_setup_time": setup_timings.get(
                f"setup_{method}_interpolation_matrix_time"
            ),
            "setup_target_grid_time": setup_timings["setup_target_grid_time"],
            "setup_inducing_grid_time": setup_timings[
                "setup_inducing_grid_time"
            ],
            "setup_interpolation_matrix_time": setup_timings[
                "setup_interpolation_matrix_time"
            ],
            "setup_kissgp_ski_bilinear_interpolation_matrix_time": setup_timings[
                "setup_kissgp_ski_bilinear_interpolation_matrix_time"
            ],
            "setup_kissgp_ski_cubic_interpolation_matrix_time": setup_timings[
                "setup_kissgp_ski_cubic_interpolation_matrix_time"
            ],
            "setup_data_time": setup_timings["setup_data_time"],
            "setup_total_time": setup_timings["setup_total_time"],
            "total_time_with_precompute": (
                setup_timings["setup_total_time"]
                + diagnostics.get("inference_total_time", infer_time)
            ),
            "timing_note": (
                "setup_*_interpolation_matrix_time fields are explicit "
                "KISS/SKI A-matrix setup times. DTC/FITC kernel "
                "cross-covariance work is inside NUTS and is not separately "
                "isolated."
            ),
            "structured_representation_bytes": representation_bytes(
                method, cfg.grid_size**2, cfg.inducing_grid_size**2
            ),
            "diagnostics_passed": diagnostics["diagnostics_passed"],
            "diagnostic_failures": "; ".join(diagnostics["diagnostic_failures"]),
        }
    )
    base.save_inference(destination, samples_chain, posterior_small, metrics, diagnostics)
    base.plot_chain_diagnostics(
        destination,
        samples_chain,
        diagnostics["worst_chain"],
        method,
        budget,
    )
    return samples_chain, posterior_small, metrics, diagnostics


def prepare(cfg: Config, base_cfg: base.Config):
    setup_start = perf_counter()
    run_dir = Path(cfg.output_root) / cfg.run_name
    seed_dir = run_dir / f"seed_{cfg.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    config_path = seed_dir / "config.json"
    saved = asdict(cfg)
    saved["methods"] = list(cfg.methods)
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous.setdefault("run_truth_metrics", False)
        if previous != saved:
            raise ValueError(f"Configuration mismatch in {seed_dir}; use a new run name.")
    base.write_json(config_path, saved)
    base.write_json(seed_dir / "environment.json", base.environment_info(cfg.grid_size))
    (seed_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n")
    stage_start = perf_counter()
    target_s = base.make_grid(cfg.grid_size, 0.0, cfg.domain_stop)
    target_grid_time = perf_counter() - stage_start
    stage_start = perf_counter()
    inducing_s = base.make_grid(cfg.inducing_grid_size, 0.0, cfg.domain_stop)
    inducing_grid_time = perf_counter() - stage_start
    stage_start = perf_counter()
    bilinear_interpolation = base.bilinear_interpolation_matrix(inducing_s, target_s)
    bilinear_interpolation_time = perf_counter() - stage_start
    stage_start = perf_counter()
    cubic_interpolation = cubic_interpolation_matrix(inducing_s, target_s)
    cubic_interpolation_time = perf_counter() - stage_start
    interpolation = {
        "kissgp_ski_bilinear": bilinear_interpolation,
        "kissgp_ski_cubic": cubic_interpolation,
    }
    stage_start = perf_counter()
    data = base.load_or_create_data(base_cfg, seed_dir, target_s)
    data_time = perf_counter() - stage_start
    mask = np.asarray(data["obs_mask"], dtype=bool)
    base.write_json(
        seed_dir / "mask_audit.json",
        {
            "created_at_utc": base.utc_now(),
            "seed": cfg.seed,
            "requested_obs_mask_type": cfg.obs_mask_type,
            "saved_obs_mask_type": data.get("obs_mask_type", "spatial"),
            "obs_ratio": cfg.obs_ratio,
            "observed_count": int(mask.sum()),
            "total_count": int(mask.size),
            "expected_observed_count": int(cfg.obs_ratio * mask.size),
        },
    )
    setup_timings = {
        "setup_target_grid_time": target_grid_time,
        "setup_inducing_grid_time": inducing_grid_time,
        "setup_interpolation_matrix_time": (
            bilinear_interpolation_time + cubic_interpolation_time
        ),
        "setup_kissgp_ski_bilinear_interpolation_matrix_time": (
            bilinear_interpolation_time
        ),
        "setup_kissgp_ski_cubic_interpolation_matrix_time": (
            cubic_interpolation_time
        ),
        "setup_data_time": data_time,
        "setup_total_time": perf_counter() - setup_start,
    }
    base.write_json(seed_dir / "setup_timing.json", setup_timings)
    return run_dir, seed_dir, target_s, inducing_s, interpolation, data, setup_timings


def main(default_grid_size: int | None = None) -> None:
    cfg = parse_args(default_grid_size)
    base_cfg = as_base_config(cfg)
    (
        run_dir,
        seed_dir,
        target_s,
        inducing_s,
        interpolation,
        data,
        setup_timings,
    ) = prepare(cfg, base_cfg)
    local_log = local_console_log_path(cfg)
    try:
        with local_log.open("a") as log_file:
            with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(
                Tee(sys.stderr, log_file)
            ):
                model_start = perf_counter()
                models = {
                    method: make_direct_gp_model(
                        method, target_s, inducing_s, interpolation, cfg
                    )
                    for method in cfg.methods
                }
                model_construction_time = perf_counter() - model_start
                setup_timings = {
                    **setup_timings,
                    "model_construction_time": model_construction_time,
                    "setup_total_time": setup_timings["setup_total_time"]
                    + model_construction_time,
                }
                base.write_json(seed_dir / "setup_timing.json", setup_timings)
                results = {}
                order = ("full_gp",) + tuple(
                    method for method in cfg.methods if method != "full_gp"
                )
                for method in order:
                    if method not in models:
                        continue
                    reference = None
                    if method != "full_gp":
                        full = results.get("full_gp") or load_full_reference(seed_dir)
                        if full is None and cfg.reference_run_name is not None:
                            reference_seed_dir = (
                                Path(cfg.output_root)
                                / cfg.reference_run_name
                                / f"seed_{cfg.seed}"
                            )
                            full = load_full_reference(reference_seed_dir)
                        if full is None:
                            if cfg.require_full_reference:
                                raise RuntimeError(
                                    "Run full_gp in this benchmark before approximations, "
                                    "or pass --reference-run-name pointing to a completed "
                                    "Full GP reference run."
                                )
                            print(
                                f"{method}: running without Full GP reference; "
                                "metrics versus Full GP will be absent."
                            )
                        else:
                            reference = (full[0], full[1])
                    result = run_method(
                        cfg,
                        base_cfg,
                        "initial",
                        method,
                        models[method],
                        data,
                        inducing_s,
                        setup_timings,
                        reference,
                    )
                    results[method] = result
                    if not result[3]["diagnostics_passed"]:
                        print(
                            f"{method}: initial diagnostics failed; running extended budget"
                        )
                        results[method] = run_method(
                            cfg,
                            base_cfg,
                            "extended",
                            method,
                            models[method],
                            data,
                            inducing_s,
                            setup_timings,
                            reference,
                        )
                rows = [result[2] for result in results.values()]
                base.write_csv(seed_dir / "combined_metrics.csv", rows)
                print("Combined metrics:", seed_dir / "combined_metrics.csv")
                print("Run root:", run_dir)
    finally:
        append_local_console_log(local_log, seed_dir)


if __name__ == "__main__":
    main()
