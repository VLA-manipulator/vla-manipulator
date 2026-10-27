"""Check the RTC mask against the paper's exponential decay equation."""

from __future__ import annotations

import unittest

import numpy as np

from openpi_ext.rtc_mask import prefix_weights


class RTCPrefixMaskTest(unittest.TestCase):
    def test_frozen_soft_and_new_regions(self):
        weights = prefix_weights(delay=2, overlap=6, horizon=10)
        np.testing.assert_array_equal(weights[:2], [1.0, 1.0])
        np.testing.assert_allclose(weights[2], 0.8 * np.expm1(0.8) / np.expm1(1.0))
        self.assertTrue(np.all(np.diff(weights[2:6]) < 0))
        np.testing.assert_array_equal(weights[6:], np.zeros(4))

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            prefix_weights(delay=6, overlap=5, horizon=10)


if __name__ == "__main__":
    unittest.main()
