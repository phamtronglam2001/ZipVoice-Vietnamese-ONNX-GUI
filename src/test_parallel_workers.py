"""Clamp rules for parallel chunk workers (CPU vs GPU)."""
from __future__ import annotations

import unittest

from chunk_synthesis import (
    clamp_parallel_workers,
    max_parallel_workers,
    ui_parallel_workers_max,
)


class TestParallelWorkers(unittest.TestCase):
    def test_gpu_clamps_high_request_to_one(self) -> None:
        self.assertEqual(max_parallel_workers(use_gpu=True), 1)
        self.assertEqual(clamp_parallel_workers(8, use_gpu=True), 1)

    def test_cpu_allows_requested_within_cap(self) -> None:
        cap = max_parallel_workers(use_gpu=False)
        requested = min(4, cap)
        self.assertEqual(clamp_parallel_workers(requested, use_gpu=False), requested)

    def test_minimum_is_one(self) -> None:
        self.assertEqual(clamp_parallel_workers(0, use_gpu=False), 1)
        self.assertEqual(clamp_parallel_workers(-3, use_gpu=True), 1)

    def test_ui_slider_max_at_least_two(self) -> None:
        self.assertEqual(ui_parallel_workers_max(use_gpu=True), 2)
        self.assertGreaterEqual(ui_parallel_workers_max(use_gpu=False), 2)


if __name__ == "__main__":
    unittest.main()
