"""Measure exact Full-GP NUTS feasibility at the 128 x 128 target grid.

The short run records timing and device metadata and is not used as a
posterior result.
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

import target128_common as target


@dataclass(frozen=True)
class Config:
    seed: int = 0
    output_root: str = "outputs/full_gp128_probe"
    warmup: int = 10
    samples: int = 10
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    covariance_jitter: float = 5e-4


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=target.PUBLIC_SEEDS, default=0)
    parser.add_argument("--output-root", default=Config.output_root)
    parser.add_argument("--warmup", type=int, default=Config.warmup)
    parser.add_argument("--samples", type=int, default=Config.samples)
    parser.add_argument("--target-accept-prob", type=float, default=Config.target_accept_prob)
    parser.add_argument("--max-tree-depth", type=int, default=Config.max_tree_depth)
    parser.add_argument("--covariance-jitter", type=float, default=Config.covariance_jitter)
    cfg = Config(**vars(parser.parse_args()))
    if cfg.warmup < 1 or cfg.samples < 1:
        raise ValueError("warmup and samples must be positive")
    return cfg


def build_model(runtime, coordinates, counts, mask, jitter):
    jnp, numpyro, dist = runtime["jnp"], runtime["numpyro"], runtime["dist"]
    matern_1_2 = runtime["matern_1_2"]

    def model():
        ell = numpyro.sample("ell", dist.LogNormal(3.0, 0.4))
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        z = numpyro.sample(
            "z", dist.Normal(0.0, 1.0).expand((target.N_TARGET,)).to_event(1)
        )
        covariance = matern_1_2(coordinates, coordinates, 1.0, ell)
        # Add jitter for numerical stability in the Cholesky factorization.
        covariance = target.add_diagonal_jitter(jnp, covariance, jitter)
        latent = jnp.linalg.cholesky(covariance) @ z
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("counts", dist.Poisson(jnp.exp(beta + latent)), obs=counts)

    return model


def run(cfg: Config) -> dict:
    runtime = target.lazy_runtime_imports()
    jax, jnp = runtime["jax"], runtime["jnp"]
    dataset_cfg = target.Config(covariance_jitter=cfg.covariance_jitter)
    data = target.generate_public_dataset(dataset_cfg, cfg.seed)
    coordinates = jnp.asarray(data["coordinates"])
    counts = jnp.asarray(data["counts"])
    mask = jnp.asarray(data["mask"])
    model = build_model(runtime, coordinates, counts, mask, cfg.covariance_jitter)
    kernel = runtime["NUTS"](
        model,
        target_accept_prob=cfg.target_accept_prob,
        max_tree_depth=cfg.max_tree_depth,
    )
    sampler = runtime["MCMC"](
        kernel,
        num_warmup=cfg.warmup,
        num_samples=cfg.samples,
        num_chains=1,
        progress_bar=True,
    )
    started = perf_counter()
    sampler.run(runtime["random"].key(cfg.seed))
    jax.block_until_ready(sampler.get_samples()["ell"])
    elapsed = perf_counter() - started
    return {
        "status": "runtime_probe_only",
        "interpretation": "The short chain measures feasibility and is not a posterior result.",
        "config": asdict(cfg),
        "elapsed_seconds": elapsed,
        "seconds_per_retained_draw": elapsed / cfg.samples,
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "dataset_hashes": {
            key: target.array_sha256(np.asarray(data[key]))
            for key in ("latent_truth", "counts", "mask")
        },
    }


def main() -> None:
    cfg = parse_args()
    report = run(cfg)
    output = Path(cfg.output_root) / f"seed_{cfg.seed}" / "runtime_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
