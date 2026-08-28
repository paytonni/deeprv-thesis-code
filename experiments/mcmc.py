"""Helpers for consistent NumPyro MCMC configuration."""
from __future__ import annotations


PUBLIC_SEEDS = (0, 1, 2)
DEFAULT_WARMUP = 1_000
DEFAULT_SAMPLES = 4_000


def validate_public_seed(seed: int) -> int:
    if seed not in PUBLIC_SEEDS:
        raise ValueError(f"seed must be one of {PUBLIC_SEEDS}")
    return seed


def build_sampler(MCMC, NUTS, model, *, num_chains: int = 2, warmup: int = DEFAULT_WARMUP,
                  samples: int = DEFAULT_SAMPLES, target_accept_prob: float = 0.8,
                  max_tree_depth: int = 10, progress_bar: bool = True):
    kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    return MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=num_chains,
                progress_bar=progress_bar)
