#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance Data Validation Tool

Validates performance data files for consistency and correctness.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class PerformanceDataValidator:
    """Validator for performance data files."""
    
    # Expected columns for each data type
    EXPECTED_COLUMNS = {
        'gemm': [
            'framework', 'version', 'device', 'op_name', 'kernel_source',
            'gemm_dtype', 'm', 'n', 'k', 'latency'
        ],
        'attention': [
            'framework', 'version', 'device', 'op_name', 'batch_size',
            'seq_len', 'num_heads', 'head_dim', 'latency'
        ],
        'moe': [
            'framework', 'version', 'device', 'op_name', 'batch_size',
            'hidden_size', 'num_experts', 'top_k', 'latency'
        ],
    }
    
    # Reasonable latency bounds (in ms)
    LATENCY_BOUNDS = {
        'gemm': (0.001, 10000),  # 1 microsecond to 10 seconds
        'attention': (0.001, 5000),
        'moe': (0.001, 10000),
    }
    
    # Expected dtypes for GEMM
    EXPECTED_GEMM_DTYPES = {'fp8', 'fp8_block', 'float16', 'int8', 'int4'}
    
    def __init__(self, data_dir: Path, backend: str, gpu: str):
        self.data_dir = data_dir
        self.backend = backend
        self.gpu = gpu
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """Run all validations."""
        print(f"Validating performance data for {self.backend} on {self.gpu}")
        print(f"Data directory: {self.data_dir}")
        print("-" * 60)
        
        # Check directory exists
        if not self.data_dir.exists():
            self.errors.append(f"Data directory does not exist: {self.data_dir}")
            return False
        
        # Validate each data type
        self._validate_gemm_data()
        self._validate_attention_data()
        self._validate_moe_data()
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_gemm_data(self):
        """Validate GEMM performance data."""
        gemm_file = self.data_dir / "gemm_perf.txt"
        
        if not gemm_file.exists():
            self.warnings.append("No GEMM performance data found")
            return
        
        try:
            df = pd.read_csv(gemm_file)
        except Exception as e:
            self.errors.append(f"Failed to read GEMM data: {e}")
            return
        
        # Filter for current backend
        df_backend = df[df['framework'].str.lower().str.contains(self.backend.lower())]
        
        if len(df_backend) == 0:
            self.warnings.append(f"No GEMM data for backend: {self.backend}")
            return
        
        # Validate columns
        self._validate_columns(df_backend, 'gemm', gemm_file.name)
        
        # Validate data types
        self._validate_gemm_dtypes(df_backend, gemm_file.name)
        
        # Validate latency values
        self._validate_latency(df_backend, 'gemm', gemm_file.name)
        
        # Validate dimensions
        self._validate_dimensions(df_backend, gemm_file.name)
        
        print(f"  GEMM: {len(df_backend)} records")
    
    def _validate_attention_data(self):
        """Validate attention performance data."""
        # Check for various attention data files
        attention_files = list(self.data_dir.glob("*attention*.txt"))
        
        if not attention_files:
            self.warnings.append("No attention performance data found")
            return
        
        for file in attention_files:
            try:
                df = pd.read_csv(file)
                print(f"  Attention ({file.name}): {len(df)} records")
            except Exception as e:
                self.errors.append(f"Failed to read {file.name}: {e}")
    
    def _validate_moe_data(self):
        """Validate MoE performance data."""
        moe_files = list(self.data_dir.glob("*moe*.txt"))
        
        if not moe_files:
            self.warnings.append("No MoE performance data found")
            return
        
        for file in moe_files:
            try:
                df = pd.read_csv(file)
                print(f"  MoE ({file.name}): {len(df)} records")
            except Exception as e:
                self.errors.append(f"Failed to read {file.name}: {e}")
    
    def _validate_columns(self, df: pd.DataFrame, data_type: str, filename: str):
        """Validate that required columns are present."""
        expected = self.EXPECTED_COLUMNS.get(data_type, [])
        missing = set(expected) - set(df.columns)
        
        if missing:
            self.errors.append(f"{filename}: Missing columns: {missing}")
    
    def _validate_gemm_dtypes(self, df: pd.DataFrame, filename: str):
        """Validate GEMM data types."""
        if 'gemm_dtype' not in df.columns:
            return
        
        dtypes = set(df['gemm_dtype'].unique())
        unknown = dtypes - self.EXPECTED_GEMM_DTYPES
        
        if unknown:
            self.warnings.append(f"{filename}: Unknown GEMM dtypes: {unknown}")
    
    def _validate_latency(self, df: pd.DataFrame, data_type: str, filename: str):
        """Validate latency values are reasonable."""
        if 'latency' not in df.columns:
            return
        
        min_lat, max_lat = self.LATENCY_BOUNDS.get(data_type, (0, float('inf')))
        
        # Check for negative latencies
        negative = (df['latency'] < 0).sum()
        if negative > 0:
            self.errors.append(f"{filename}: {negative} negative latency values")
        
        # Check for suspiciously large latencies
        large = (df['latency'] > max_lat).sum()
        if large > 0:
            self.warnings.append(f"{filename}: {large} latency values > {max_lat}ms")
        
        # Check for zero latencies
        zero = (df['latency'] == 0).sum()
        if zero > 0:
            self.warnings.append(f"{filename}: {zero} zero latency values")
    
    def _validate_dimensions(self, df: pd.DataFrame, filename: str):
        """Validate dimension values are positive."""
        for dim in ['m', 'n', 'k', 'batch_size', 'seq_len', 'num_heads', 'head_dim']:
            if dim in df.columns:
                negative = (df[dim] <= 0).sum()
                if negative > 0:
                    self.errors.append(f"{filename}: {negative} non-positive {dim} values")
    
    def _print_results(self):
        """Print validation results."""
        print("-" * 60)
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")
        elif not self.errors:
            print(f"\n✅ Validation passed with {len(self.warnings)} warnings")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} errors")


def main():
    parser = argparse.ArgumentParser(description="Validate performance data")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["sglang", "trtllm", "vllm"],
        help="Backend to validate"
    )
    parser.add_argument(
        "--gpu",
        required=True,
        help="GPU type (e.g., h200, h100, h20)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("systems/data"),
        help="Base directory for performance data"
    )
    
    args = parser.parse_args()
    
    # Construct GPU-specific data directory
    data_dir = args.data_dir / args.gpu
    
    # Run validation
    validator = PerformanceDataValidator(data_dir, args.backend, args.gpu)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
