// polars-bio/src/qc/read_length_distribution.rs

use polars::prelude::*;
use std::collections::HashMap;
use needletail::parse_fastx_file;
use std::path::Path;
use arrow::array::{UInt32Array, UInt64Array};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use rayon::prelude::*;
use needletail::FastxReader;

/// Helper: process a chunk of read lengths and return a HashMap of length -> count
fn process_lengths(lengths: &[u32]) -> HashMap<u32, u64> {
    let mut map = HashMap::new();
    for &len in lengths {
        *map.entry(len).or_insert(0u64) += 1;
    }
    map
}

/// Compute read length distribution from a FASTQ file.
use datafusion::prelude::*;

pub async fn read_length_distribution(
    ctx: &SessionContext,
    fastq_path: &str,
) -> datafusion::error::Result<datafusion::dataframe::DataFrame> {
    let mut reader = parse_fastx_file(&Path::new(fastq_path)).expect("Invalid path/file");
    let chunk_size = 100_000;
    let mut chunks: Vec<Vec<u32>> = Vec::new();
    let mut current_chunk = Vec::with_capacity(chunk_size);
    while let Some(record) = reader.next() {
        if let Ok(seqrec) = record {
            current_chunk.push(seqrec.seq().len() as u32);
            if current_chunk.len() == chunk_size {
                chunks.push(std::mem::take(&mut current_chunk));
            }
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    // Parallel processing of chunks of lengths
    let results: Vec<HashMap<u32, u64>> = chunks
        .into_par_iter()
        .map(|chunk| process_lengths(&chunk))
        .collect();
    // Merge all HashMaps
    let mut read_lengths = HashMap::new();
    for map in results {
        for (len, count) in map {
            *read_lengths.entry(len).or_insert(0u64) += count;
        }
    }
    let mut lengths: Vec<u32> = read_lengths.keys().cloned().collect();
    lengths.sort_unstable();
    let counts: Vec<u64> = lengths.iter().map(|l| read_lengths[l]).collect();

    let length_array = Arc::new(UInt32Array::from(lengths)) as _;
    let count_array = Arc::new(UInt64Array::from(counts)) as _;

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("length", arrow::datatypes::DataType::UInt32, false),
        arrow::datatypes::Field::new("count", arrow::datatypes::DataType::UInt64, false),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![length_array, count_array])?;

    let provider = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table("read_length_distribution", Arc::new(provider))?;

    ctx.table("read_length_distribution").await
}

// TODO: Register as DataFusion UDF and add Python bindings.
