"""Pytest configuration for the MLX EP op-correctness suite.

The MLX execution-provider plugin is registered once per test session from the
``ONNXRUNTIME_MLX_EP_LIB`` environment variable (set by CTest to the built
``libonnxruntime_mlx_ep.dylib``). Running ``pytest`` without that variable skips
the suite rather than failing, so the tests are usable both under CTest and by
hand (``ONNXRUNTIME_MLX_EP_LIB=<dylib> pytest tests/ops``).
"""

from __future__ import annotations

import os

import onnxruntime as ort
import pytest

EP_NAME = "MLXExecutionProvider"


@pytest.fixture(scope="session", autouse=True)
def register_mlx_ep() -> None:
    """Register the MLX EP plugin exactly once for the whole session."""
    lib = os.environ.get("ONNXRUNTIME_MLX_EP_LIB")
    if not lib:
        pytest.skip(
            "ONNXRUNTIME_MLX_EP_LIB is not set (absolute path to libonnxruntime_mlx_ep.dylib)",
            allow_module_level=True,
        )
    lib = os.path.abspath(lib)
    if not os.path.isfile(lib):
        pytest.skip(f"MLX EP library not found at {lib}", allow_module_level=True)
    ort.register_execution_provider_library(EP_NAME, lib)
