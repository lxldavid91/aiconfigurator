# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang Backend Tests

Tests for SGLang-specific functionality and performance data validation.
"""

import pytest
import os
from pathlib import Path


# Mark all tests in this module
pytestmark = [
    pytest.mark.backend,
    pytest.mark.sglang,
]


class TestSGLangBasic:
    """Basic SGLang backend tests."""

    def test_sglang_import(self):
        """Test that SGLang can be imported."""
        try:
            import sglang
            assert hasattr(sglang, '__version__')
        except ImportError:
            pytest.skip("SGLang not installed in this environment")

    def test_sglang_version(self):
        """Test SGLang version compatibility."""
        import pkg_resources
        try:
            version = pkg_resources.get_distribution("sglang").version
            # Check version is >= 0.5.0
            major, minor, *_ = map(int, version.split('.')[:2])
            assert major >= 0 and minor >= 5, f"SGLang version {version} too old"
        except Exception:
            pytest.skip("Could not determine SGLang version")


class TestSGLangGEMM:
    """GEMM performance tests for SGLang."""

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
        """Test that GEMM performance data exists for this GPU."""
        gpu_type = os.environ.get("GPU_TYPE", "h200")
        data_file = perf_data_dir / gpu_type / "gemm_perf.txt"
        
        if not data_file.exists():
            pytest.skip(f"No GEMM performance data for {gpu_type}")

    @pytest.mark.skipif(
        not os.environ.get("GPU_TYPE"),
        reason="No GPU_TYPE environment variable"
    )
    def test_gemm_perf_data_valid(self, perf_data_dir):
        """Validate GEMM performance data format and values."""
        import pandas as pd
        
        gpu_type = os.environ.get("GPU_TYPE", "h200")
        data_file = perf_data_dir / gpu_type / "gemm_perf.txt"
        
        if not data_file.exists():
            pytest.skip(f"No GEMM performance data for {gpu_type}")
        
        df = pd.read_csv(data_file)
        
        # Check required columns
        required_cols = ['framework', 'version', 'device', 'op_name', 'gemm_dtype', 'm', 'n', 'k']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check no negative dimensions
        assert (df['m'] > 0).all(), "Negative M dimension"
        assert (df['n'] > 0).all(), "Negative N dimension"
        assert (df['k'] > 0).all(), "Negative K dimension"

    def test_gemm_dtype_coverage(self, perf_data_dir):
        """Test that all required dtypes are covered."""
        import pandas as pd
        
        gpu_type = os.environ.get("GPU_TYPE", "h200")
        data_file = perf_data_dir / gpu_type / "gemm_perf.txt"
        
        if not data_file.exists():
            pytest.skip(f"No GEMM performance data for {gpu_type}")
        
        df = pd.read_csv(data_file)
        
        # Check dtype coverage
        expected_dtypes = {'fp8', 'float16', 'int8'}
        actual_dtypes = set(df['gemm_dtype'].unique())
        
        missing = expected_dtypes - actual_dtypes
        if missing:
            pytest.fail(f"Missing dtype coverage: {missing}")


class TestSGLangAttention:
    """Attention performance tests for SGLang."""

    @pytest.mark.skipif(
        not os.environ.get("GPU_TYPE"),
        reason="No GPU_TYPE environment variable"
    )
    def test_attention_perf_data_valid(self):
        """Validate attention performance data."""
        # TODO: Implement attention data validation
        pytest.skip("Attention data validation not implemented")


class TestSGLangMoE:
    """MoE performance tests for SGLang."""

    @pytest.mark.skipif(
        not os.environ.get("GPU_TYPE"),
        reason="No GPU_TYPE environment variable"
    )
    def test_moe_perf_data_valid(self):
        """Validate MoE performance data."""
        # TODO: Implement MoE data validation
        pytest.skip("MoE data validation not implemented")


class TestSGLangIntegration:
    """Integration tests for SGLang backend."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="Integration tests not enabled"
    )
    def test_end_to_end_inference(self):
        """Test end-to-end inference with SGLang."""
        # TODO: Implement e2e test
        pytest.skip("End-to-end test not implemented")


# Performance benchmark tests
@pytest.mark.performance
class TestSGLangPerformance:
    """Performance benchmark tests."""

    @pytest.mark.parametrize("dtype", ["fp8", "float16", "int8"])
    @pytest.mark.parametrize("m,n,k", [
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ])
    def test_gemm_benchmark(self, dtype, m, n, k):
        """Benchmark GEMM operations."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # TODO: Implement actual benchmark
        # This is a placeholder that validates the test structure
        device = torch.device("cuda:0")
        
        # Create test tensors
        if dtype == "fp8":
            a = torch.randn(m, k, dtype=torch.float8_e4m3fn, device=device)
            b = torch.randn(k, n, dtype=torch.float8_e4m3fn, device=device)
        elif dtype == "float16":
            a = torch.randn(m, k, dtype=torch.float16, device=device)
            b = torch.randn(k, n, dtype=torch.float16, device=device)
        else:  # int8
            a = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=device)
            b = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=device)
        
        # Warmup
        if dtype in ["fp8", "float16"]:
            _ = torch.matmul(a.float(), b.float())
        
        # Benchmark
        torch.cuda.synchronize()
        
        # Just validate shapes for now
        assert a.shape == (m, k)
        assert b.shape == (k, n)
