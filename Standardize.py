
import os
import re
import json
import click
import pandas as pd
from datetime import datetime
import hashlib


def compute_checksum(file_path):
    """Compute MD5 checksum for integrity check."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case lowercase."""
    df.columns = [
        re.sub(r'[^a-z0-9]+', '_', str(col).strip().lower())
        for col in df.columns
    ]
    return df


def standardize_dates(df: pd.DataFrame, date_format=None) -> pd.DataFrame:
    """Try converting object columns to YYYY-MM-DD date format."""
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return df


def standardize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-like strings (with commas, ₹) into floats."""
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].str.replace(",", "", regex=True)
            cleaned = cleaned.str.replace("₹", "", regex=False)
            try:
                df[col] = pd.to_numeric(cleaned)
            except Exception:
                pass
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: drop very empty columns, fill others."""
    for col in df.columns:
        if df[col].isna().mean() > 0.3:
            df = df.drop(columns=[col])
        else:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")
    return df


@click.command()
@click.option("--src", required=True, help="Path to ingested file (from data/raw)")
@click.option("--out", required=False, default="../data/standardized", help="Output directory for standardized files")
@click.option("--date-format", required=False, default=None, help="Optional date format string, e.g. %d-%m-%Y")
def standardize(src, out, date_format):
    """
    Standardize an ingested dataset.
    Saves cleaned dataset in data/standardized and a metadata log in logs/.
    """
    os.makedirs(out, exist_ok=True)

    # Load ingested CSV
    df = pd.read_csv(src)

    # Apply transformations
    df = standardize_column_names(df)
    df = standardize_dates(df, date_format)
    df = standardize_numeric(df)
    df = handle_missing_values(df)

    # Save standardized CSV
    out_file = os.path.join(out, os.path.basename(src))
    df.to_csv(out_file, index=False)

    # Metadata
    meta = {
        "source_file": src,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "columns_info": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_summary": df.isna().sum().to_dict(),
        "saved_to": out_file,
        "checksum": compute_checksum(out_file),
        "standardized_at": datetime.now().isoformat()
    }

    # Save log into logs/ folder
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    base_name = os.path.basename(src).replace(".csv", "")
    log_file = os.path.join(log_dir, f"standardize_{base_name}.json")
    with open(log_file, "w") as f:
        json.dump(meta, f, indent=2)

    click.echo("\n✅ Standardization complete")
    click.echo(f"📂 Standardized file: {out_file}")
    click.echo(f"📝 Metadata log: {log_file}")


if __name__ == "__main__":
    standardize()
