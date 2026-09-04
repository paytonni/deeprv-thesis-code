from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from metric_normalization import normalize_analysis_metrics


class Target128NormalizationTest(unittest.TestCase):
    def test_realistic_target128_posterior_summary_aliases(self) -> None:
        row = normalize_analysis_metrics(
            {
                "model": "Exact128",
                "target_grid_size": 128,
                "data_seed": 0,
                "ell_mean": 29.75,
                "ell_median": 29.50,
                "ell_q05": 23.25,
                "ell_q95": 36.75,
                "beta_mean": 1.04,
                "beta_median": 1.02,
                "beta_q05": 0.71,
                "beta_q95": 1.39,
            }
        )

        self.assertEqual(row["posterior_lengthscale_mean"], 29.75)
        self.assertEqual(row["posterior_lengthscale_median"], 29.50)
        self.assertEqual(row["posterior_lengthscale_q05"], 23.25)
        self.assertEqual(row["posterior_lengthscale_q95"], 36.75)
        self.assertEqual(row["posterior_beta_mean"], 1.04)
        self.assertEqual(row["posterior_beta_median"], 1.02)
        self.assertEqual(row["posterior_beta_q05"], 0.71)
        self.assertEqual(row["posterior_beta_q95"], 1.39)
        self.assertNotIn("posterior_lengthscale_sd", row)
        self.assertNotIn("posterior_beta_sd", row)


if __name__ == "__main__":
    unittest.main()
