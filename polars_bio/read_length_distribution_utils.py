import polars as pl
import polars_bio
from typing import Iterator, Union
from datafusion import DataFrame, SessionContext
from polars.io.plugins import register_io_source
from tqdm.auto import tqdm
from .context import ctx
from .range_op_helpers import stream_wrapper
from .io import lazy_scan
from polars_bio.polars_bio import (
    InputFormat, 
    ReadOptions, 
    py_read_table, 
    py_register_table, 
    py_scan_table,
    read_length_distribution_native
)

def read_length_distribution(fastq_path: str) -> pl.DataFrame:
    """
    Python wrapper for the Rust read_length_distribution function.
    
    Parameters:
        fastq_path: Path to the FASTQ file.
        
    Returns:
        A DataFrame with read length distribution.
    """
    df = polars_bio.polars_bio.read_length_distribution(fastq_path)
    # Try to convert to Polars DataFrame if not already
    if not isinstance(df, pl.DataFrame):
        try:
            import pyarrow as pa
            if hasattr(df, "to_arrow_table"):
                arrow_table = df.to_arrow_table()
                return pl.from_arrow(arrow_table)
            elif hasattr(df, "to_pandas"):
                return pl.from_pandas(df.to_pandas())
        except Exception:
            pass
    return df

def read_length_distribution_parallel(
    fastq_path: str, 
    streaming: bool = False,
    target_partitions: int = 8
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Compute read length distribution from a FASTQ file with out-of-core processing
    and parallel execution.
    
    This function processes FASTQ files in a streaming fashion using the Polars lazy
    evaluation model. It counts the occurrences of each read length in the file.
    
    Parameters:
        fastq_path: Path to the FASTQ file.
        streaming: If True, returns a LazyFrame for streaming execution. Otherwise,
                  returns a materialized DataFrame.
        target_partitions: Number of partitions for parallel execution.
        
    Returns:
        A DataFrame or LazyFrame with columns 'length' and 'count'.
    """
    # Register the FASTQ file as a table
    df = read_file(fastq_path, InputFormat.Fastq, None)
    
    # Set the target partitions for parallel execution
    previous_partitions = ctx.get_option("datafusion.execution.target_partitions")
    ctx.set_option("datafusion.execution.target_partitions", str(target_partitions), False)
    
    # Create a LazyFrame with the read length distribution
    lf = lazy_scan(df).select([
        pl.col("sequence").map_elements(lambda s: len(s), return_dtype=pl.UInt32).alias("length")
    ]).group_by("length").agg([
        pl.len().alias("count").cast(pl.UInt64)  # Using len() instead of count() and casting to UInt64
    ]).sort("length")
    
    # Reset the target partitions
    ctx.set_option("datafusion.execution.target_partitions", previous_partitions, False)
    
    if streaming:
        return lf
    else:
        return lf.collect()

def read_file(
    path: str,
    input_format: InputFormat,
    read_options: ReadOptions,
    streaming: bool = False,
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Read a file into a DataFrame.
    
    Parameters:
        path: Path to the file.
        input_format: Input format of the file.
        read_options: Options for reading the file.
        streaming: If True, returns a streaming LazyFrame.
        
    Returns:
        A DataFrame or LazyFrame.
    """
    table = py_register_table(ctx, path, None, input_format, read_options)
    if streaming:
        return stream_wrapper(py_scan_table(ctx, table.name))
    else:
        return py_read_table(ctx, table.name)

def read_length_distribution_rust(
    fastq_path: str, 
    streaming: bool = False,
    target_partitions: int = 8
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Compute read length distribution from a FASTQ file with out-of-core processing
    using Rust for the data processing.
    
    This function registers the FASTQ file as a table in DataFusion and then uses 
    DataFusion's SQL engine to compute the distribution directly in Rust, which is
    more efficient than processing in Python.
    
    Parameters:
        fastq_path: Path to the FASTQ file.
        streaming: If True, returns a LazyFrame for streaming execution. Otherwise,
                  returns a materialized DataFrame.
        target_partitions: Number of partitions for parallel execution.
        
    Returns:
        A DataFrame or LazyFrame with columns 'length' and 'count'.
    """
    # Register the FASTQ file as a table
    table = py_register_table(ctx, fastq_path, None, InputFormat.Fastq, None)
    
    # Set the target partitions for parallel execution
    previous_partitions = ctx.get_option("datafusion.execution.target_partitions")
    ctx.set_option("datafusion.execution.target_partitions", str(target_partitions), False)
    
    # Process the data directly in Rust
    datafusion_df = read_length_distribution_native(ctx, table.name)
    
    # Reset the target partitions
    ctx.set_option("datafusion.execution.target_partitions", previous_partitions, False)
    
    # Convert to Polars DataFrame with correct schema
    try:
        # First attempt to convert to Arrow Table
        if hasattr(datafusion_df, "to_arrow"):
            arrow_table = datafusion_df.to_arrow()
            polars_df = pl.from_arrow(arrow_table)
        else:
            # Try to collect the DataFrame and create a Polars DataFrame from the records
            records = datafusion_df.collect()
            
            # Check if records is a list of dicts or tuples
            if isinstance(records, list) and len(records) > 0:
                first = records[0]
                if hasattr(first, "get") or (isinstance(first, (tuple, list)) and len(first) == 2):
                    # Extract length and count from the records
                    lengths = []
                    counts = []
                    for record in records:
                        if hasattr(record, "get"):
                            lengths.append(record.get("length"))
                            counts.append(record.get("count"))
                        else:
                            lengths.append(record[0])
                            counts.append(record[1])
                    polars_df = pl.DataFrame({
                        "length": pl.Series(lengths, dtype=pl.UInt32),
                        "count": pl.Series(counts, dtype=pl.UInt64)
                    })
                elif str(type(first)).endswith("pyarrow.lib.RecordBatch'>"):
                    # Handle pyarrow.RecordBatch
                    polars_df = pl.from_arrow(first)
                else:
                    # Unexpected structure: print for debugging and raise
                    raise ValueError(f"Unexpected record structure from DataFusion: {records}")
            else:
                # If records is not a list, try a more generic approach
                import pandas as pd
                polars_df = pl.from_pandas(pd.DataFrame(records))
        
        # Ensure correct column types
        polars_df = polars_df.with_columns([
            pl.col("length").cast(pl.UInt32),
            pl.col("count").cast(pl.UInt64)
        ])
    except Exception as e:
        raise ValueError(f"Failed to convert DataFusion DataFrame to Polars DataFrame: {e}")
    
    if streaming:
        # Convert to a LazyFrame for streaming
        return polars_df.lazy()
    else:
        # Return the DataFrame directly
        return polars_df

def plot_read_length_distribution(df: pl.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(df['length'].to_numpy(), df['count'].to_numpy(), width=1.0, align='center')
    ax.set_xlabel('Read length')
    ax.set_ylabel('Count')
    ax.set_title('Read Length Distribution')
    return ax
