# pipeline/ingest-1.py

import click
import pandas as pd
from pathlib import Path
import hashlib
import json
from datetime import datetime


def compute_checksum(file_path):
    """Compute MD5 checksum for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@click.command()
@click.option("--src", required=True, help="Path to the local dataset file")
@click.option("--out", default="data/raw", help="Output directory for raw files")
def load(src, out):
    """
    Ingest a dataset from local file into raw data folder.
    Saves metadata log with rows, columns, size, checksum.
    """
    src_path = Path(src)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy file into raw directory
    dest_file = out_dir / src_path.name
    df = pd.read_csv(src_path)  # for now only CSV
    df.to_csv(dest_file, index=False)

    # Compute metadata
    metadata = {
        "source": str(src_path),
        "saved_as": str(dest_file),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "file_size_kb": round(src_path.stat().st_size / 1024, 2),
        "checksum": compute_checksum(src_path),
        "ingested_at": datetime.utcnow().isoformat(),
    }

    # Save metadata log
    log_file = out_dir / f"{src_path.stem}_ingest_log.json"
    with open(log_file, "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"[INGEST] Saved {dest_file}")
    click.echo(f"[LOG] Metadata written to {log_file}")


if __name__ == "__main__":
    load()
