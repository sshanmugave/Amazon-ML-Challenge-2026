"""
Script 03: Multi-View Normalization Builder
Amazon ML Challenge 2026 - Business Entity Resolution

Reads raw TSVs, applies the multi-view normalization engine, and writes
structured records directly to compact Parquet files in cache/normalized/
"""

import os
import sys
import time
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_resolution.normalization import normalize_record

DATASET_ROOT = r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset"
OUTPUT_DIR = r"cache\normalized"

PARQUET_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("source", pa.string()),
    ("name_raw", pa.string()),
    ("name_ascii", pa.string()),
    ("name_alnum", pa.string()),
    ("name_no_suffix", pa.string()),
    ("legal_suffix", pa.string()),
    ("addr_raw", pa.string()),
    ("addr_ascii", pa.string()),
    ("addr_alnum", pa.string()),
    ("house_number", pa.string()),
    ("postal_code", pa.string()),
    ("addr_numerics", pa.string()),
])


def normalize_file(input_path: str, output_path: str, batch_size: int = 100000):
    """Streams a TSV file, normalizes records in batches, and writes to Parquet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t0 = time.time()
    
    fn = os.path.basename(input_path)
    print(f"Normalizing {fn} -> {os.path.basename(output_path)} (batch_size={batch_size:,}) ...", flush=True)
    
    total_processed = 0
    writer = None
    
    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            addr_idx = header.index("business_address")
            country_idx = header.index("country")
            
            batch_records = []
            
            for line in f:
                parts = line.strip().split("\t")
                eid = parts[id_idx]
                name = parts[name_idx] if len(parts) > name_idx else ""
                addr = parts[addr_idx] if len(parts) > addr_idx else ""
                country = parts[country_idx] if len(parts) > country_idx else "UNKNOWN"
                
                rec = normalize_record(eid, name, addr, country)
                batch_records.append(rec.to_dict())
                
                if len(batch_records) >= batch_size:
                    batch_df = pa.Table.from_pylist(batch_records, schema=PARQUET_SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA, compression="zstd")
                    writer.write_table(batch_df)
                    total_processed += len(batch_records)
                    print(f"  Processed {total_processed:,} records ({time.time() - t0:.1f}s) ...", flush=True)
                    batch_records = []
                    
            if batch_records:
                batch_df = pa.Table.from_pylist(batch_records, schema=PARQUET_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA, compression="zstd")
                writer.write_table(batch_df)
                total_processed += len(batch_records)
                
    finally:
        if writer is not None:
            writer.close()
            
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
    elapsed = time.time() - t0
    rate = total_processed / elapsed if elapsed > 0 else 0
    print(f"  [DONE] {total_processed:,} records in {elapsed:.1f}s ({rate:,.0f} rec/s) | Output size: {out_size_mb:.1f} MB\n", flush=True)


DEFAULT_DATASET_ROOT = os.environ.get(
    "AMAZON_ML_DATASET_ROOT",
    r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset"
    if os.path.exists(r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)
DEFAULT_OUTPUT_DIR = r"cache\normalized"


def main():
    parser = argparse.ArgumentParser(description="Multi-View Normalization Builder")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Root folder of raw datasets")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output folder for normalized parquets")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train", help="Which dataset split to normalize")
    parser.add_argument("--batch-size", type=int, default=100000, help="Batch size for writing Parquet")
    args = parser.parse_args()
    
    files = []
    if args.split in ("train", "all"):
        files.extend([
            (os.path.join(args.dataset_root, "train", "train_source1.tsv"), os.path.join(args.output_dir, "train_source1.parquet")),
            (os.path.join(args.dataset_root, "train", "train_source2.tsv"), os.path.join(args.output_dir, "train_source2.parquet")),
            (os.path.join(args.dataset_root, "train", "train_source3.tsv"), os.path.join(args.output_dir, "train_source3.parquet")),
        ])
    if args.split in ("test", "all"):
        files.extend([
            (os.path.join(args.dataset_root, "test", "test_source1.tsv"), os.path.join(args.output_dir, "test_source1.parquet")),
            (os.path.join(args.dataset_root, "test", "test_source2.tsv"), os.path.join(args.output_dir, "test_source2.parquet")),
            (os.path.join(args.dataset_root, "test", "test_source3.tsv"), os.path.join(args.output_dir, "test_source3.parquet")),
        ])
        
    print("=" * 78)
    print(f"  STAGE 3.3: MULTI-VIEW NORMALIZATION PIPELINE ({args.split.upper()} SPLIT)")
    print("=" * 78)
    
    total_t0 = time.time()
    for in_path, out_path in files:
        normalize_file(in_path, out_path, batch_size=args.batch_size)
        
    print("=" * 78)
    print(f"All files in {args.split.upper()} normalized in {time.time() - total_t0:.1f} seconds.")
    print("=" * 78)

if __name__ == "__main__":
    main()
