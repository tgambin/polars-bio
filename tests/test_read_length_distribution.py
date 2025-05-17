import polars as pl
from polars_bio.qc import read_length_distribution, plot_read_length_distribution

def test_read_length_distribution():
    # Path to a small test FASTQ file (update path as needed)
    fastq_path = "tests/data/example.fastq"
    df = read_length_distribution(fastq_path)
    # Accept both polars.DataFrame and polars.dataframe.frame.DataFrame
    assert type(df).__name__ == "DataFrame"
    assert "length" in df.columns and "count" in df.columns
    # Optionally plot and save to file for CI/headless environments
    ax = plot_read_length_distribution(df)
    import matplotlib.pyplot as plt
    plt.savefig("tests/data/read_length_distribution_plot.png")

if __name__ == "__main__":
    test_read_length_distribution()
