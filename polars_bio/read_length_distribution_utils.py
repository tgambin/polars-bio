import polars as pl
import polars_bio

def read_length_distribution(fastq_path: str) -> pl.DataFrame:
    """
    Python wrapper for the Rust read_length_distribution function.
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

def plot_read_length_distribution(df: pl.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(df['length'].to_numpy(), df['count'].to_numpy(), width=1.0, align='center')
    ax.set_xlabel('Read length')
    ax.set_ylabel('Count')
    ax.set_title('Read Length Distribution')
    return ax
