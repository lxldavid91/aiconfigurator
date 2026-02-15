# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend test configuration
"""

import pytest
import os


def pytest_configure(config):
    """Configure custom markers for backend tests."""
    config.addinivalue_line(
        "markers", "backend: mark test as a backend test"
    )
    config.addinivalue_line(
        "markers", "sglang: mark test as SGLang-specific"
    )
    config.addinivalue_line(
        "markers", "trtllm: mark test as TensorRT-LLM-specific"
    )
    config.addinivalue_line(
        "markers", "vllm: mark test as vLLM-specific"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )


@pytest.fixture(scope="session")
def backend_type():
    """Get the backend type from environment."""
    return os.environ.get("BACKEND", "unknown")


@pytest.fixture(scope="session")
def gpu_type():
    """Get the GPU type from environment."""
    return os.environ.get("GPU_TYPE", "unknown")
