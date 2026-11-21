"""Smoke tests for release-only helpers; no external datasets are required."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_utils import score_cohort_robust_z, GPL11532_PATH

class ReleaseHelperSmokeTest(unittest.TestCase):
    def test_robust_score_is_finite_and_uses_positive_and_negative_weights(self):
        x = np.array([[1., 1.], [2., 2.], [3., 5.], [4., 8.]])
        result, usable = score_cohort_robust_z(x, [0, 1], np.array([1., -1.]))
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(usable.tolist(), [True, True])
        self.assertGreater(result[0], result[-1])

    def test_annotation_path_uses_configured_data_root(self):
        self.assertEqual(GPL11532_PATH.name, "GPL11532.annot.gz")
        self.assertEqual(GPL11532_PATH.parent.name, "reference")

if __name__ == "__main__":
    unittest.main()
