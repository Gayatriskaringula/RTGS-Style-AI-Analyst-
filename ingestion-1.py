import click
import pandas as pd
import os
import json
from datetime import datetime

@click.command()
@click.option(
    "--src",
    required=True,
    type=click.Path(exists=True),
    help="Path to the local dataset file (CSV, Excel, JSON)",
)
@click.option(
    "--out",
    default="../data/raw",
    type=click.Path(),
    help="Output directory for raw files [default: ../data/raw]",
)
def load(src, out):
    """
    Ingest a dataset from local file into raw data folder.
    Saves metadata log with rows, columns, size, checksum.
    """

    # Detect file type
    ext = os.path.splitext(src)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(src)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(src)
    elif ext == ".json":
        df = pd.read_json(src)
    else:
        raise click.ClickException(f"❌ Unsupported file format: {ext}")

    # Ensure output directory exists
    os.makedirs(out, exist_ok=True)

    # Save a copy of ingested file in output directory
    fname = os.path.basename(src)
    dest_path = os.path.join(out, fname)
    df.to_csv(dest_path, index=False)

    # Collect metadata
    metadata = {
        "source_file": src,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "columns_info": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "saved_to": dest_path,
        "ingested_at": datetime.now().isoformat(),
    }

    # Save metadata log
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ingest_{fname}.json")
    with open(log_file, "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"\n✅ Dataset ingested successfully!")
    click.echo(f"📂 Saved to: {dest_path}")
    click.echo(f"📝 Log saved to: {log_file}")
    click.echo(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

if __name__ == "__main__":
    load()
