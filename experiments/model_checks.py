"""Shared model diagnostics for the synthetic DeepRV experiments.

This module is deliberately independent of notebook state.  It records the
JAX/XLA backend, computes accuracy against the simulated latent/rate truth,
and compares a frozen DeepRV prior with its GP/teacher references.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

os.environ["MPLBACKEND"] = "Agg"

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import random
from scipy.special import gammaln, logsumexp

from dl4bi_sps.kernels import matern_1_2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TRUTH_METRIC_PREFIXES = (
    "mse_latent_truth_",
    "mae_latent_truth_",
    "coverage_latent_truth_",
    "mse_rate_truth_",
    "mae_rate_truth_",
    "coverage_rate_truth_",
    "count_mae_truth_",
    "count_log1p_mse_truth_",
    "count_predictive_log_density_",
    "count_coverage_",
)
FULL_GP_FIDELITY_KEYS = (
    "posterior_mean_mse_vs_full_gp",
    "posterior_mean_log1p_mse_vs_full_gp",
    "ls_wasserstein_vs_full_gp",
    "beta_wasserstein_vs_full_gp",
)


@dataclass(frozen=True)
class PriorDiagnosticConfig:
    ell_values: tuple[float, ...] = (10.0, 30.0, 50.0)
    num_samples: int = 32
    z_seed: int = 0
    jitter: float = 5e-4
    distance_bins: int = 12
    max_map_samples: int = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standard_gmlp_architecture(grid_size: int) -> dict[str, Any]:
    """Describe the exact gMLPDeepRV instantiated by the experiment runners."""
    locations = int(grid_size) ** 2
    fixed_parameters = 79_745
    spatial_parameters = 2 * locations * locations + 2 * locations
    return {
        "backbone": "gMLP",
        "class_name": "gMLPDeepRV",
        "num_blocks": 2,
        "embedding_dimension": 64,
        "hidden_expansion_dimension": 128,
        "input_features_per_location": 4,
        "input_feature_definition": ["z", "x_coordinate", "y_coordinate", "ell"],
        "output_features_per_location": 1,
        "latent_dimension": locations,
        "input_shape": ["batch", locations, 4],
        "output_shape": ["batch", locations, 1],
        "spatial_gating_shape_per_block": [locations, locations],
        "parameter_count": fixed_parameters + spatial_parameters,
        "parameter_count_method": (
            "analytical count for the standard two-block gMLPDeepRV; "
            "validated against Orbax checkpoint metadata"
        ),
        "grid_size": int(grid_size),
    }


def count_parameters(parameters: Any) -> int:
    leaves = jax.tree.leaves(parameters)
    # Read shape metadata only.  Calling np.asarray on a 64x64 checkpoint leaf
    # can copy hundreds of MB from an accelerator merely to count parameters.
    return int(sum(math.prod(leaf.shape) for leaf in leaves))


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi_gpu_names() -> list[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    result = subprocess.run(
        [executable, "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_environment_report(
    *,
    grid_size: Optional[int] = None,
    architecture: Optional[dict[str, Any]] = None,
    argv: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Collect explicit Python -> JAX -> XLA -> accelerator provenance."""
    devices = list(jax.devices())
    backend = None
    backend_error = None
    try:
        from jax.extend import backend as jax_backend

        backend = jax_backend.get_backend()
    except Exception as error:  # Backend APIs differ across supported JAX releases.
        backend_error = repr(error)
    device_rows = []
    for device in devices:
        client = getattr(device, "client", None)
        device_rows.append(
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
                "local_hardware_id": getattr(device, "local_hardware_id", None),
                "client_platform": getattr(client, "platform", None),
                "client_platform_version": getattr(client, "platform_version", None),
                "client_runtime_type": getattr(client, "runtime_type", None),
                "string": str(device),
            }
        )
    report = {
        "created_at_utc": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "argv": list(sys.argv if argv is None else argv),
        "versions": {
            "jax": jax.__version__,
            "jaxlib": _package_version("jaxlib"),
            "numpyro": _package_version("numpyro"),
            "flax": _package_version("flax"),
            "optax": _package_version("optax"),
            "orbax_checkpoint": _package_version("orbax-checkpoint"),
            "numpy": np.__version__,
        },
        "jax_default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_devices": device_rows,
        "gpu_names": _nvidia_smi_gpu_names(),
        "xla_backend": {
            "platform": getattr(backend, "platform", None),
            "platform_version": getattr(backend, "platform_version", None),
            "runtime_type": getattr(backend, "runtime_type", None),
            "error": backend_error,
        },
        "software_stack": [
            "NumPyro NUTS",
            "JAX autodiff and JIT",
            "XLA/PJRT",
            (
                "CUDA GPU backend"
                if any(row["platform"] in ("gpu", "cuda") for row in device_rows)
                else "CPU backend"
            ),
        ],
    }
    if architecture is None and grid_size is not None:
        architecture = standard_gmlp_architecture(grid_size)
    if architecture is not None:
        report["deeprv_architecture"] = architecture
    return report


def _draw_matrix(values: Any, locations: int) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("Posterior draw array is empty.")
    if array.shape[-1] != locations:
        raise ValueError(
            f"Expected final dimension {locations}, got posterior shape {array.shape}."
        )
    return array.reshape(-1, locations)


def _region_metrics(prefix: str, error: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_all": float(np.mean(error)),
        f"{prefix}_observed": float(np.mean(error[mask])),
        f"{prefix}_unobserved": float(np.mean(error[~mask])),
    }


def _coverage_metrics(
    prefix: str,
    draws: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    level: float,
) -> dict[str, float]:
    alpha = 1.0 - level
    lower = np.quantile(draws, alpha / 2.0, axis=0)
    upper = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    covered = (truth >= lower) & (truth <= upper)
    suffix = int(round(level * 100))
    return _region_metrics(f"{prefix}_{suffix}", covered.astype(float), mask)


def compute_truth_relative_metrics(
    data: dict[str, Any],
    posterior: dict[str, Any],
    samples: dict[str, Any],
    *,
    coverage_level: float = 0.9,
) -> dict[str, float]:
    """Compute latent-field, rate, and count accuracy against simulated truth."""
    true_f = np.asarray(data["latent_f"]).reshape(-1)
    true_rate = np.asarray(data["rate"]).reshape(-1)
    true_counts = np.asarray(data["y_full"]).reshape(-1)
    mask = np.asarray(data["obs_mask"], dtype=bool).reshape(-1)
    locations = true_f.size
    if not (true_rate.size == true_counts.size == mask.size == locations):
        raise ValueError("Truth arrays and observation mask must have the same size.")

    mu_draws = _draw_matrix(posterior["mu"], locations)
    beta_draws = np.asarray(samples["beta"]).reshape(-1)
    if beta_draws.size == 1:
        beta_draws = np.repeat(beta_draws, mu_draws.shape[0])
    if beta_draws.size != mu_draws.shape[0]:
        raise ValueError(
            "beta and latent mu must contain the same number of posterior draws: "
            f"{beta_draws.size} versus {mu_draws.shape[0]}."
        )
    rate_draws = np.exp(beta_draws[:, None] + mu_draws)

    posterior_mean_f = np.mean(mu_draws, axis=0)
    posterior_mean_rate = np.mean(rate_draws, axis=0)
    metrics: dict[str, float] = {}
    metrics.update(
        _region_metrics("mse_latent_truth", (posterior_mean_f - true_f) ** 2, mask)
    )
    metrics.update(
        _region_metrics("mae_latent_truth", np.abs(posterior_mean_f - true_f), mask)
    )
    metrics.update(
        _coverage_metrics(
            "coverage_latent_truth", mu_draws, true_f, mask, coverage_level
        )
    )
    metrics.update(
        _region_metrics("mse_rate_truth", (posterior_mean_rate - true_rate) ** 2, mask)
    )
    metrics.update(
        _region_metrics("mae_rate_truth", np.abs(posterior_mean_rate - true_rate), mask)
    )
    metrics.update(
        _coverage_metrics(
            "coverage_rate_truth", rate_draws, true_rate, mask, coverage_level
        )
    )

    if posterior.get("obs") is not None:
        count_draws = _draw_matrix(posterior["obs"], locations)
        posterior_mean_counts = np.mean(count_draws, axis=0)
        metrics.update(
            _region_metrics(
                "count_mae_truth", np.abs(posterior_mean_counts - true_counts), mask
            )
        )
        metrics.update(
            _region_metrics(
                "count_log1p_mse_truth",
                (np.log1p(posterior_mean_counts) - np.log1p(true_counts)) ** 2,
                mask,
            )
        )
        metrics.update(
            _coverage_metrics(
                "count_coverage", count_draws, true_counts, mask, coverage_level
            )
        )

    log_prob = (
        true_counts[None, :] * np.log(np.maximum(rate_draws, 1e-30))
        - rate_draws
        - gammaln(true_counts[None, :] + 1.0)
    )
    pointwise_lpd = logsumexp(log_prob, axis=0) - math.log(rate_draws.shape[0])
    metrics.update(
        _region_metrics("count_predictive_log_density", pointwise_lpd, mask)
    )
    metrics["truth_metric_draws"] = float(mu_draws.shape[0])
    return metrics


def write_metric_schema(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Write the truth-relative and Full-GP-fidelity metric views."""
    identity = {
        key: metrics.get(key)
        for key in ("seed", "model_name", "budget", "diagnostics_passed")
        if key in metrics
    }
    truth = {
        key: value
        for key, value in metrics.items()
        if key.startswith(TRUTH_METRIC_PREFIXES) or key == "truth_metric_draws"
    }
    fidelity = {
        key: metrics.get(key)
        for key in FULL_GP_FIDELITY_KEYS
    }
    write_csv(output_dir / "truth_relative_metrics.csv", [{**identity, **truth}])
    write_csv(output_dir / "full_gp_fidelity_metrics.csv", [{**identity, **fidelity}])


def _kernel(locations: jax.Array, ell: float, jitter: float) -> jax.Array:
    n = locations.shape[0]
    return matern_1_2(locations, locations, 1.0, ell) + jitter * jnp.eye(n)


def construct_teacher_prior_samples(
    method: str,
    target_s: jax.Array,
    ell: float,
    z_target: jax.Array,
    *,
    jitter: float = 5e-4,
    inducing_s: Optional[jax.Array] = None,
    interpolation_matrix: Optional[jax.Array] = None,
    latent_indices: Optional[jax.Array] = None,
    z_residual: Optional[jax.Array] = None,
) -> dict[str, Any]:
    """Construct teacher draws while keeping pairing semantics explicit."""
    method = method.lower()
    target_s = jnp.asarray(target_s)
    z_target = jnp.asarray(z_target)
    n = target_s.shape[0]
    if z_target.ndim != 2 or z_target.shape[1] != n:
        raise ValueError(f"z_target must have shape (samples, {n}).")
    k_hh = _kernel(target_s, ell, jitter)
    exact = jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(k_hh), z_target)
    if method == "exact":
        return {
            "teacher_samples": exact,
            "exact_gp_samples": exact,
            "teacher_covariance": k_hh,
            "exact_gp_covariance": k_hh,
            "conditional_component": exact,
            "residual_component": jnp.zeros_like(exact),
            "paired_exact_gp_valid": True,
            "pairing_basis": "same full target-grid z and exact GP Cholesky",
        }

    if inducing_s is None or latent_indices is None:
        raise ValueError(f"{method} requires inducing_s and latent_indices.")
    inducing_s = jnp.asarray(inducing_s)
    latent_indices = jnp.asarray(latent_indices)
    m = inducing_s.shape[0]
    if latent_indices.shape != (m,):
        raise ValueError(f"latent_indices must have shape ({m},).")
    z_u = z_target[:, latent_indices]
    k_uu = _kernel(inducing_s, ell, jitter)
    chol_uu = jnp.linalg.cholesky(k_uu)
    u = jnp.einsum("ij,bj->bi", chol_uu, z_u)

    if method in ("bilinear", "cubic"):
        if interpolation_matrix is None:
            raise ValueError(f"{method} requires interpolation_matrix.")
        matrix = jnp.asarray(interpolation_matrix)
        if matrix.shape != (n, m):
            raise ValueError(f"Interpolation matrix must have shape ({n}, {m}).")
        teacher = jnp.einsum("nm,bm->bn", matrix, u)
        teacher_covariance = matrix @ k_uu @ matrix.T
        return {
            "teacher_samples": teacher,
            "exact_gp_samples": exact,
            "teacher_covariance": teacher_covariance,
            "exact_gp_covariance": k_hh,
            "conditional_component": teacher,
            "residual_component": jnp.zeros_like(teacher),
            "paired_exact_gp_valid": True,
            "pairing_basis": (
                "same target-grid z with the declared inducing latent subset; "
                "pairing is construction-specific"
            ),
        }

    k_hu = matern_1_2(target_s, inducing_s, 1.0, ell)
    conditional = (
        k_hu
        @ jax.scipy.linalg.solve_triangular(chol_uu.T, z_u.T, lower=False)
    ).T
    solved = jax.scipy.linalg.cho_solve((chol_uu, True), k_hu.T)
    q_hh = k_hu @ solved
    if method == "dtc":
        return {
            "teacher_samples": conditional,
            "exact_gp_samples": exact,
            "teacher_covariance": q_hh,
            "exact_gp_covariance": k_hh,
            "conditional_component": conditional,
            "residual_component": jnp.zeros_like(conditional),
            "paired_exact_gp_valid": True,
            "pairing_basis": (
                "same target-grid z with the declared inducing latent subset; "
                "pairing is construction-specific"
            ),
        }
    if method != "fitc":
        raise ValueError(f"Unknown teacher method: {method}.")
    if z_residual is None:
        raise ValueError("FITC requires independent z_residual draws.")
    z_residual = jnp.asarray(z_residual)
    if z_residual.shape != z_target.shape:
        raise ValueError(f"FITC z_residual must have shape {z_target.shape}.")
    residual_diag = jnp.clip(jnp.diag(k_hh - q_hh), min=0.0)
    residual = jnp.sqrt(residual_diag)[None, :] * z_residual
    return {
        "teacher_samples": conditional + residual,
        "exact_gp_samples": exact,
        "teacher_covariance": q_hh + jnp.diag(residual_diag),
        "exact_gp_covariance": k_hh,
        "conditional_component": conditional,
        "residual_component": residual,
        "paired_exact_gp_valid": False,
        "pairing_basis": (
            "FITC conditional component uses shared inducing z, while the diagonal "
            "residual uses independent noise; no unique full-sample pairing"
        ),
    }


def _relative_frobenius(estimate: np.ndarray, target: np.ndarray) -> float:
    denominator = np.linalg.norm(target, ord="fro")
    return float(np.linalg.norm(estimate - target, ord="fro") / max(denominator, 1e-12))


def _sample_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _distance_curves(
    samples: np.ndarray,
    target_s: np.ndarray,
    bins: int,
) -> list[dict[str, float]]:
    centered = samples - samples.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(samples.shape[0] - 1, 1)
    diffs = samples[:, :, None] - samples[:, None, :]
    semivariance = 0.5 * np.mean(diffs**2, axis=0)
    distances = np.linalg.norm(target_s[:, None, :] - target_s[None, :, :], axis=-1)
    upper = np.triu_indices(target_s.shape[0], k=1)
    distance_values = distances[upper]
    edges = np.linspace(0.0, float(distance_values.max()) + 1e-12, bins + 1)
    rows = []
    for index in range(bins):
        selected = (distance_values >= edges[index]) & (distance_values < edges[index + 1])
        if not np.any(selected):
            continue
        rows.append(
            {
                "distance_bin": index,
                "distance_mean": float(np.mean(distance_values[selected])),
                "covariance_mean": float(np.mean(covariance[upper][selected])),
                "semivariance_mean": float(np.mean(semivariance[upper][selected])),
                "pair_count": int(np.sum(selected)),
            }
        )
    return rows


def _decoder_array(decoder_output: Any) -> np.ndarray:
    if hasattr(decoder_output, "f_hat"):
        decoder_output = decoder_output.f_hat
    values = np.asarray(decoder_output)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"DeepRV decoder must return (samples, locations), got {values.shape}.")
    return values


def _plot_sample_maps(
    output_path: Path,
    deep_samples: np.ndarray,
    exact_samples: np.ndarray,
    grid_size: int,
    paired: bool,
    count: int,
) -> None:
    columns = 3 if paired else 2
    count = min(count, deep_samples.shape[0])
    fig, axes = plt.subplots(count, columns, figsize=(4 * columns, 3.4 * count), squeeze=False)
    for row in range(count):
        panels = [(exact_samples[row], "Exact GP"), (deep_samples[row], "DeepRV")]
        if paired:
            panels.append((deep_samples[row] - exact_samples[row], "DeepRV - exact"))
        for column, (values, title) in enumerate(panels):
            image = axes[row, column].imshow(values.reshape(grid_size, grid_size), cmap="viridis")
            axes[row, column].set_title(f"sample {row}: {title}")
            fig.colorbar(image, ax=axes[row, column], shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_prior_diagnostics(
    decoder: Callable[..., Any],
    target_s: jax.Array,
    output_dir: Path,
    *,
    grid_size: int,
    method: str,
    config: PriorDiagnosticConfig = PriorDiagnosticConfig(),
    inducing_s: Optional[jax.Array] = None,
    interpolation_matrix: Optional[jax.Array] = None,
    latent_indices: Optional[jax.Array] = None,
) -> dict[str, Any]:
    """Run paired and distributional prior checks for one frozen decoder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    target_np = np.asarray(target_s)
    sample_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    smoothness_panels = []

    for ell_index, ell in enumerate(config.ell_values):
        key = random.fold_in(random.key(config.z_seed), ell_index)
        key_z, key_residual = random.split(key)
        z_target = random.normal(key_z, (config.num_samples, target_s.shape[0]))
        z_residual = random.normal(key_residual, z_target.shape)
        reference = construct_teacher_prior_samples(
            method,
            target_s,
            ell,
            z_target,
            jitter=config.jitter,
            inducing_s=inducing_s,
            interpolation_matrix=interpolation_matrix,
            latent_indices=latent_indices,
            z_residual=z_residual if method.lower() == "fitc" else None,
        )
        deep = _decoder_array(decoder(z_target, jnp.asarray([ell]), s=target_s))
        exact = np.asarray(reference["exact_gp_samples"])
        teacher = np.asarray(reference["teacher_samples"])
        conditional = np.asarray(reference["conditional_component"])
        paired = bool(reference["paired_exact_gp_valid"])
        if deep.shape != exact.shape:
            raise ValueError(f"DeepRV shape {deep.shape} does not match exact GP {exact.shape}.")

        for sample_index in range(config.num_samples):
            row: dict[str, Any] = {
                "ell": float(ell),
                "sample_index": sample_index,
                "z_seed": config.z_seed,
                "paired_exact_gp_valid": paired,
                "pairing_basis": reference["pairing_basis"],
                "conditional_component_mse": float(
                    np.mean((deep[sample_index] - conditional[sample_index]) ** 2)
                ),
                "conditional_component_correlation": _sample_correlation(
                    deep[sample_index], conditional[sample_index]
                ),
            }
            if paired:
                row["paired_mse"] = float(np.mean((deep[sample_index] - exact[sample_index]) ** 2))
                row["paired_correlation"] = _sample_correlation(
                    deep[sample_index], exact[sample_index]
                )
            else:
                row["paired_mse"] = None
                row["paired_correlation"] = None
            sample_rows.append(row)

        exact_covariance = np.asarray(reference["exact_gp_covariance"])
        teacher_covariance = np.asarray(reference["teacher_covariance"])
        deep_covariance = np.cov(deep, rowvar=False, ddof=1)
        deep_variance = np.var(deep, axis=0, ddof=1)
        exact_variance = np.diag(exact_covariance)
        selected_rows = [row for row in sample_rows if row["ell"] == float(ell)]
        paired_mses = [row["paired_mse"] for row in selected_rows if row["paired_mse"] is not None]
        paired_correlations = [
            row["paired_correlation"]
            for row in selected_rows
            if row["paired_correlation"] is not None and np.isfinite(row["paired_correlation"])
        ]
        aggregate_rows.append(
            {
                "grid_size": grid_size,
                "method": method,
                "ell": float(ell),
                "num_samples": config.num_samples,
                "paired_exact_gp_valid": paired,
                "pairing_basis": reference["pairing_basis"],
                "paired_mse_mean": float(np.mean(paired_mses)) if paired_mses else None,
                "paired_correlation_mean": (
                    float(np.mean(paired_correlations)) if paired_correlations else None
                ),
                "deeprv_marginal_mean_abs_mean": float(np.mean(np.abs(np.mean(deep, axis=0)))),
                "deeprv_marginal_variance_mean": float(np.mean(deep_variance)),
                "exact_marginal_variance_mean": float(np.mean(exact_variance)),
                "marginal_variance_mae": float(np.mean(np.abs(deep_variance - exact_variance))),
                "deeprv_empirical_covariance_relative_frobenius_error": _relative_frobenius(
                    deep_covariance, exact_covariance
                ),
                "teacher_theoretical_covariance_relative_frobenius_error": _relative_frobenius(
                    teacher_covariance, exact_covariance
                ),
                "conditional_component_mse_mean": float(
                    np.mean([row["conditional_component_mse"] for row in selected_rows])
                ),
            }
        )
        for source_name, source_samples in (
            ("deeprv", deep),
            ("exact_gp", exact),
            ("teacher", teacher),
        ):
            for curve in _distance_curves(source_samples, target_np, config.distance_bins):
                curve_rows.append(
                    {"ell": float(ell), "source": source_name, **curve}
                )

        ell_label = f"{ell:g}".replace(".", "p")
        _plot_sample_maps(
            output_dir / f"prior_sample_maps_ell_{ell_label}.png",
            deep,
            exact,
            grid_size,
            paired,
            config.max_map_samples,
        )
        archive[f"ell_{ell_label}_deeprv"] = deep
        archive[f"ell_{ell_label}_exact_gp"] = exact
        archive[f"ell_{ell_label}_teacher"] = teacher
        archive[f"ell_{ell_label}_conditional_component"] = conditional
        archive[f"ell_{ell_label}_residual_component"] = np.asarray(reference["residual_component"])
        smoothness_panels.append((float(ell), exact[0], deep[0]))

    np.savez_compressed(output_dir / "prior_samples.npz", **archive)
    write_csv(output_dir / "sample_metrics.csv", sample_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(output_dir / "distance_curves.csv", curve_rows)
    write_json(
        output_dir / "method_metadata.json",
        {
            "method": method,
            "grid_size": grid_size,
            "target_coordinates_shape": list(target_np.shape),
            "inducing_coordinates_shape": (
                None if inducing_s is None else list(np.asarray(inducing_s).shape)
            ),
            "config": asdict(config),
            "paired_exact_gp_valid": aggregate_rows[0]["paired_exact_gp_valid"],
            "pairing_basis": aggregate_rows[0]["pairing_basis"],
            "fitc_reporting_rule": (
                "Report the shared inducing-z conditional component separately; "
                "assess full FITC through marginal/distributional covariance because "
                "the residual noise is independent."
            ),
        },
    )

    fig, axes = plt.subplots(2, len(smoothness_panels), figsize=(4 * len(smoothness_panels), 7))
    if len(smoothness_panels) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (ell, exact, deep) in enumerate(smoothness_panels):
        for row, (values, title) in enumerate(((exact, "Exact GP"), (deep, "DeepRV"))):
            image = axes[row, column].imshow(values.reshape(grid_size, grid_size), cmap="viridis")
            axes[row, column].set_title(f"{title}, ell={ell:g}")
            fig.colorbar(image, ax=axes[row, column], shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_dir / "smoothness_across_ell.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ell in config.ell_values:
        for source, style in (("exact_gp", "-"), ("deeprv", "--")):
            selected = [
                row for row in curve_rows if row["ell"] == float(ell) and row["source"] == source
            ]
            axes[0].plot(
                [row["distance_mean"] for row in selected],
                [row["covariance_mean"] for row in selected],
                style,
                label=f"{source}, ell={ell:g}",
            )
            axes[1].plot(
                [row["distance_mean"] for row in selected],
                [row["semivariance_mean"] for row in selected],
                style,
                label=f"{source}, ell={ell:g}",
            )
    axes[0].set(xlabel="distance", ylabel="empirical covariance")
    axes[1].set(xlabel="distance", ylabel="empirical semivariance")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "covariance_variogram_vs_distance.png", dpi=180)
    plt.close(fig)

    covariance_errors = [
        row["deeprv_empirical_covariance_relative_frobenius_error"]
        for row in aggregate_rows
    ]
    paired_correlations = [
        row["paired_correlation_mean"]
        for row in aggregate_rows
        if row["paired_correlation_mean"] is not None
    ]
    clear_mismatch = bool(
        np.median(covariance_errors) > 0.75
        and (not paired_correlations or np.median(paired_correlations) < 0.5)
    )
    status = "CLEAR_PRIOR_MISMATCH" if clear_mismatch else "NO_CLEAR_PRIOR_FAILURE_IN_THIS_CHECK"
    report_lines = [
        "# DeepRV prior diagnostic report",
        "",
        f"- Status: `{status}`",
        f"- Grid: `{grid_size}x{grid_size}`",
        f"- Teacher: `{method}`",
        f"- z samples per ell: `{config.num_samples}`",
        f"- ell values: `{list(config.ell_values)}`",
        f"- Paired exact-GP comparison valid: `{aggregate_rows[0]['paired_exact_gp_valid']}`",
        f"- Pairing basis: {aggregate_rows[0]['pairing_basis']}",
        "",
        "This automatic label is a screening result, not a posterior-quality conclusion. "
        "If prior samples or covariance are clearly wrong, investigate the teacher, checkpoint, "
        "or network before NUTS. If the prior looks adequate while NUTS fails, inspect posterior "
        "geometry in the main comparison.",
    ]
    if method.lower() == "fitc":
        report_lines.extend(
            [
                "",
                "FITC note: the conditional component and independent diagonal residual are "
                "reported separately. No unique full-sample same-z pairing is asserted.",
            ]
        )
    (output_dir / "summary_report.md").write_text("\n".join(report_lines) + "\n")
    result = {
        "status": status,
        "output_dir": str(output_dir),
        "aggregate_rows": aggregate_rows,
        "paired_exact_gp_valid": aggregate_rows[0]["paired_exact_gp_valid"],
    }
    write_json(output_dir / "complete.json", {"completed_at_utc": utc_now(), **result})
    return result
