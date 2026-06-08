"""Unit tests for inference performance helpers (no GPU required)."""
from __future__ import annotations

import unittest

import numpy as np

from config import inference_batch_size, ode_solver_default, onnx_num_threads
from onnx_engine import (
    ODE_SOLVERS,
    _normalize_ode_solver,
    _pad_token_batch,
    _run_ode_loop,
)
from onnx_session_opts import build_session_options


class _FakeFmModel:
    def __init__(self) -> None:
        self.calls = 0

    def run_fm_decoder(self, t, x, text_condition, speech_condition, guidance_scale):
        self.calls += 1
        return np.zeros_like(x)


class TestPerfConfig(unittest.TestCase):
    def test_onnx_num_threads_default_positive(self):
        self.assertGreaterEqual(onnx_num_threads(), 1)

    def test_inference_batch_default(self):
        self.assertGreaterEqual(inference_batch_size(), 1)

    def test_ode_solver_default(self):
        self.assertIn(ode_solver_default(), ODE_SOLVERS)


class TestOdeSolvers(unittest.TestCase):
    def test_normalize_solver_invalid(self):
        self.assertEqual(_normalize_ode_solver("bogus"), "euler")

    def test_euler_one_call_per_step(self):
        fake = _FakeFmModel()
        x = np.zeros((1, 4, 2), dtype=np.float32)
        tc = np.zeros((1, 4, 2), dtype=np.float32)
        sc = np.zeros((1, 4, 2), dtype=np.float32)
        g = np.array(1.0, dtype=np.float32)
        ts = np.linspace(0, 1, 4, dtype=np.float32)
        _run_ode_loop(
            fake,  # type: ignore[arg-type]
            x=x,
            text_condition=tc,
            speech_condition=sc,
            guidance_np=g,
            timesteps=ts,
            num_step=3,
            solver="euler",
        )
        self.assertEqual(fake.calls, 3)

    def test_heun_two_calls_per_step(self):
        fake = _FakeFmModel()
        x = np.zeros((1, 4, 2), dtype=np.float32)
        tc = np.zeros((1, 4, 2), dtype=np.float32)
        sc = np.zeros((1, 4, 2), dtype=np.float32)
        g = np.array(1.0, dtype=np.float32)
        ts = np.linspace(0, 1, 3, dtype=np.float32)
        _run_ode_loop(
            fake,  # type: ignore[arg-type]
            x=x,
            text_condition=tc,
            speech_condition=sc,
            guidance_np=g,
            timesteps=ts,
            num_step=2,
            solver="heun",
        )
        self.assertEqual(fake.calls, 4)


class TestTokenPadding(unittest.TestCase):
    def test_pad_token_batch(self):
        rows = [[1, 2], [3, 4, 5]]
        out = _pad_token_batch(rows)
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_array_equal(out[0], [1, 2, 0])
        np.testing.assert_array_equal(out[1], [3, 4, 5])


class TestSessionOptions(unittest.TestCase):
    def test_build_session_options(self):
        opts = build_session_options(num_threads=2)
        self.assertEqual(opts.intra_op_num_threads, 2)
        self.assertEqual(opts.inter_op_num_threads, 2)


if __name__ == "__main__":
    unittest.main()
