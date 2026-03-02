# Multi-Backend Test Matrix

This document describes the multi-backend test matrix CI workflow for AIConfigurator.

## Overview

The `backend-matrix.yml` workflow provides comprehensive testing across multiple GPU types and inference backends.

## Trigger Conditions

- **Push to main/release branches**
- **PRs that modify:**
  - `src/` - Core source code
  - `tests/backends/` - Backend test files
  - `pyproject.toml` - Dependencies
  - `.github/workflows/backend-matrix.yml` - This workflow
- **Manual trigger** via workflow_dispatch

## Test Matrix

### Supported Backends

| Backend | Container Image | Notes |
|---------|----------------|-------|
| SGLang | `lmsysorg/sglang:v0.5.6.post2-cu130` | Latest stable |
| TensorRT-LLM | `nvcr.io/nvidia/tensorrt-llm:v0.20.0` | NVIDIA optimized |
| vLLM | `vllm/vllm-openai:v0.14.0` | High-performance |

### Supported GPU Types

| GPU Type | Runner Label | Memory |
|----------|--------------|--------|
| H200 SXM | `self-hosted-h200` | 141GB |
| H100 SXM | `self-hosted-h100` | 80GB |
| H20 SXM | `self-hosted-h20` | 143GB |
| A100 SXM | `self-hosted-a100` | 80GB |

## Workflow Jobs

### 1. prepare-matrix
- Determines which backend/GPU combinations to test
- Skips if no relevant changes detected

### 2. backend-tests
- Runs tests for each backend/GPU combination
- **Timeout:** 60 minutes per job
- **Fail-fast:** Disabled (all combinations run to completion)

### 3. aggregate-results
- Merges JUnit test reports
- Generates summary report
- Fails workflow if any tests failed

### 4. validate-performance
- Validates performance data quality
- Checks for missing values, suspicious values
- Generates performance summary

### 5. notify-failure
- Creates failure summary on workflow failure

## Manual Trigger

```bash
# Trigger via GitHub CLI
gh workflow run backend-matrix.yml \
  -f backends=sglang,trtllm \
  -f gpu_types=h200,h100

# Or via GitHub UI:
# Actions → Multi-Backend Test Matrix → Run workflow
```

## Test Structure

```
tests/backends/
├── conftest.py           # Shared fixtures
├── __init__.py
├── sglang/
│   ├── __init__.py
│   └── test_sglang.py    # SGLang tests
├── trtllm/
│   ├── __init__.py
│   └── test_trtllm.py    # TRT-LLM tests
└── vllm/
    ├── __init__.py
    └── test_vllm.py      # vLLM tests
```

## Adding New Tests

### 1. Create Test File

```python
# tests/backends/sglang/test_my_feature.py

import pytest

pytestmark = [
    pytest.mark.backend,
    pytest.mark.sglang,
]

class TestMyFeature:
    def test_basic(self):
        """Basic test."""
        pass
    
    @pytest.mark.parametrize("size", [1024, 4096])
    def test_parametrized(self, size):
        """Parametrized test."""
        pass
```

### 2. Update pytest.ini

Ensure test markers are registered:

```ini
[pytest]
markers =
    backend: Backend tests
    sglang: SGLang-specific tests
    trtllm: TRT-LLM-specific tests
    vllm: vLLM-specific tests
```

## Performance Data Validation

Run validation manually:

```bash
python tools/sanity_check/validate_perf_data.py \
  --backend sglang \
  --gpu h200 \
  --data-dir systems/data
```

## Artifacts

| Artifact | Contents | Retention |
|----------|----------|-----------|
| `test-results-{backend}-{gpu}` | JUnit XML + HTML reports | 30 days |
| `perf-data-{backend}-{gpu}` | Performance data files | 30 days |
| `merged-test-report` | Combined JUnit report | 30 days |

## Self-Hosted Runner Setup

### Requirements

1. **GPU:** Matching GPU type (H200, H100, H20, or A100)
2. **Docker:** Installed and configured
3. **NVIDIA Container Toolkit:** For GPU access in containers
4. **Labels:** Set appropriate labels (e.g., `self-hosted-h200`)

### Runner Configuration

```yaml
# Example: config.yml for self-hosted runner
labels:
  - self-hosted
  - linux
  - x64
  - gpu
  - self-hosted-h200  # GPU-specific label
```

## Troubleshooting

### Container Pull Failures

```bash
# Manual pull to diagnose
docker pull lmsysorg/sglang:v0.5.6.post2-cu130-amd64-runtime
```

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Test Timeout

Adjust timeout in workflow:

```yaml
env:
  TEST_TIMEOUT_MINUTES: 120  # Increase to 2 hours
```

## Best Practices

1. **Use markers** to categorize tests
2. **Skip tests gracefully** when dependencies unavailable
3. **Parametrize tests** for better coverage
4. **Keep tests fast** - unit tests should complete quickly
5. **Document assumptions** about test environment

## Future Improvements

- [ ] Add performance regression detection
- [ ] Support multi-node testing
- [ ] Add custom test data injection
- [ ] Integrate with performance dashboard
- [ ] Add automatic PR comments with results
