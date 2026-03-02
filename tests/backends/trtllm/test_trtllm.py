# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT-LLM Backend Tests

Tests for TRT-LLM-specific functionality and performance data validation.
"""

import pytest
import os
from pathlib import Path


pytestmark = [
    pytest.mark.backend,
    pytest.mark.trtllm,
]


class TestTRTLLMBasic:
    """Basic TRT-LLM backend tests."""

    def test_trtllm_import(self):
        """Test that TensorRT-LLM can be imported."""
        try:
            import tensorrt_llm
            assert hasattr(tensorrt_llm, '__version__')
        except ImportError:
            pytest.skip("TensorRT-LLM not installed in this environment")

    def test_trtllm_version(self):
        """Test TRT-LLM version compatibility."""
        try:
            import tensorrt_llm
            version = tensorrt_llm.__version__
            # Check version format
            parts = version.split('.')
            assert len(parts) >= 2, f"Invalid version format: {version}"
        except Exception:
            pytest.skip("Could not determine TRT-LLM version")


class TestTRTLLMGEMM:
    """GEMM performance tests for TRT-LLM."""

    @pytest.fixture
    def perf_data_dir(self):
        """Get performance data directory."""
        base_dir = Path(__file__).parent.parent.parent.parent / "systems" / "data"
        return base_dir

    @pytest.mark.skipif(
        not os.environ.get("GPU_TYPE"),
        reason="No GPU_TYPE environment variable"
    )
    def test_gemm_perf_data_exists(self, perf_data_dir):
        """Test that GEMM performance data exists."""
        gpu_type = os.environ.get("GPU_TYPE", "h200")
        data_file = perf_data_dir / gpu_type / "gemm_perf.txt"
        
        if not data_file.exists():
            pytest.skip(f"No GEMM performance data for {gpu_type}")


# Placeholder for additional TRT-LLM tests
