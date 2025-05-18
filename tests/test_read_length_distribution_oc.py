#!/usr/bin/env python
# filepath: /home/tgambin/workspace/polars-bio/tests/test_read_length_distribution_oc.py

import os
import polars as pl
import matplotlib.pyplot as plt
from polars_bio.qc import (
    read_length_distribution, 
    read_length_distribution_parallel, 
    read_length_distribution_rust
)


def test_read_length_distribution_parallel():
    """Test the parallel read length distribution function."""
    # Path to test FASTQ file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fastq_path = os.path.join(current_dir, "data", "example.fastq")
    
    # Get the read length distribution using the original function
    regular_df = read_length_distribution(fastq_path)
    
    # Get the read length distribution using the new parallel function
    parallel_df = read_length_distribution_parallel(fastq_path)
    
    # Test with streaming mode
    parallel_lazy = read_length_distribution_parallel(fastq_path, streaming=True)
    parallel_streaming_df = parallel_lazy.collect()
    
    # Test with different partition counts
    parallel_multi_part = read_length_distribution_parallel(fastq_path, target_partitions=4)
    
    # Test the Rust implementation
    rust_df = read_length_distribution_rust(fastq_path)
    rust_streaming_df = read_length_distribution_rust(fastq_path, streaming=True).collect()
    rust_multi_part = read_length_distribution_rust(fastq_path, target_partitions=4)
    
    # Verify that all methods produce the same result
    assert parallel_df.schema == regular_df.schema
    assert rust_df.schema == regular_df.schema
    
    # Sort both dataframes by length to ensure consistent comparison
    regular_df = regular_df.sort("length")
    parallel_df = parallel_df.sort("length")
    parallel_streaming_df = parallel_streaming_df.sort("length")
    parallel_multi_part = parallel_multi_part.sort("length")
    
    # Sort the Rust dataframes by length for consistent comparison
    rust_df = rust_df.sort("length")
    rust_streaming_df = rust_streaming_df.sort("length")
    rust_multi_part = rust_multi_part.sort("length")
    
    # Convert to dictionaries for easier comparison
    regular_dict = {row["length"]: row["count"] for row in regular_df.iter_rows(named=True)}
    parallel_dict = {row["length"]: row["count"] for row in parallel_df.iter_rows(named=True)}
    streaming_dict = {row["length"]: row["count"] for row in parallel_streaming_df.iter_rows(named=True)}
    multi_part_dict = {row["length"]: row["count"] for row in parallel_multi_part.iter_rows(named=True)}
    rust_dict = {row["length"]: row["count"] for row in rust_df.iter_rows(named=True)}
    rust_streaming_dict = {row["length"]: row["count"] for row in rust_streaming_df.iter_rows(named=True)}
    rust_multi_part_dict = {row["length"]: row["count"] for row in rust_multi_part.iter_rows(named=True)}
    
    # Compare results
    assert regular_dict == parallel_dict, "Parallel implementation gives different results than original"
    assert regular_dict == streaming_dict, "Streaming implementation gives different results than original"
    assert regular_dict == multi_part_dict, "Multi-partition implementation gives different results than original"
    assert regular_dict == rust_dict, "Rust implementation gives different results than original"
    assert regular_dict == rust_streaming_dict, "Rust streaming implementation gives different results than original"
    assert regular_dict == rust_multi_part_dict, "Rust multi-partition implementation gives different results than original"
    
    # Visualize results (optional)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.bar(regular_df["length"].to_numpy(), regular_df["count"].to_numpy(), 
            width=1.0, align='center', label="Original")
    ax1.set_xlabel('Read length')
    ax1.set_ylabel('Count')
    ax1.set_title('Original Implementation')
    ax1.legend()
    
    ax2.bar(parallel_df["length"].to_numpy(), parallel_df["count"].to_numpy(), 
            width=1.0, align='center', color='orange', label="Parallel")
    ax2.set_xlabel('Read length')
    ax2.set_ylabel('Count')
    ax2.set_title('Parallel Implementation')
    ax2.legend()
    
    ax3.bar(rust_df["length"].to_numpy(), rust_df["count"].to_numpy(), 
            width=1.0, align='center', color='green', label="Rust")
    ax3.set_xlabel('Read length')
    ax3.set_ylabel('Count')
    ax3.set_title('Rust Implementation')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "data", "read_length_distribution_comparison.png"))
    plt.close()


if __name__ == "__main__":
    test_read_length_distribution_parallel()