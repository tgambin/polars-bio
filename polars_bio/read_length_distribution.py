import polars as pl
from polars_bio import read_length_distribution as rust_read_length_distribution

def read_length_distribution(fastq_path: str) -> pl.DataFrame:
    """
    Python wrapper for the Rust read_length_distribution function.
    """
    return rust_read_length_distribution(fastq_path)

def plot_read_length_distribution(df: pl.DataFrame, ax=None):
    """
    Plot read length distribution from a Polars DataFrame with columns 'length' and 'count'.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(df['length'].to_numpy(), df['count'].to_numpy(), width=1.0, align='center')
    ax.set_xlabel('Read length')
    ax.set_ylabel('Count')
    ax.set_title('Read Length Distribution')
    return ax
