#!/usr/bin/env python3
"""
RTGS Data Ingestion Module
Handles loading and validation of various data formats
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles data loading and initial validation."""
    
    def __init__(self, data_dir: str = "data", logs_dir: str = "logs"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        for dir_path in [self.raw_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, file_path: str, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load dataset from file and perform initial validation.
        
        Args:
            file_path: Path to the source file
            dataset_name: Name to assign to the dataset
            
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        logger.info(f"Starting ingestion for dataset: {dataset_name}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        # Detect and load based on file format
        df, load_metadata = self._load_by_format(file_path)
        
        # Perform initial validation
        validation_results = self._validate_dataset(df, dataset_name)
        
        # Generate file checksum for integrity
        checksum = self._compute_checksum(file_path)
        
        # Save raw copy
        raw_file_path = self._save_raw_copy(df, dataset_name)
        
        # Compile metadata
        metadata = {
            "dataset_name": dataset_name,
            "source_file": file_path,
            "ingestion_timestamp": datetime.now().isoformat(),
            "file_checksum": checksum,
            "raw_file_path": str(raw_file_path),
            "load_metadata": load_metadata,
            "validation_results": validation_results,
            "initial_shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Log ingestion
        self._log_ingestion(dataset_name, metadata)
        
        logger.info(f"Successfully ingested {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df, metadata
    
    def _load_by_format(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Load data based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        load_metadata = {
            "file_extension": ext,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / 1024 / 1024
        }
        
        try:
            if ext == ".csv":
                # Try different encodings and separators
                df = self._load_csv_robust(file_path, load_metadata)
            elif ext in [".xlsx", ".xls"]:
                df = self._load_excel_robust(file_path, load_metadata)
            elif ext == ".json":
                df = self._load_json_robust(file_path, load_metadata)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            load_metadata["status"] = "success"
            load_metadata["rows_loaded"] = len(df)
            load_metadata["columns_loaded"] = len(df.columns)
            
        except Exception as e:
            load_metadata["status"] = "error"
            load_metadata["error_message"] = str(e)
            raise Exception(f"Failed to load {file_path}: {str(e)}")
        
        return df, load_metadata
    
    def _load_csv_robust(self, file_path: str, metadata: Dict) -> pd.DataFrame:
        """Robust CSV loading with multiple fallback strategies."""
        # Common CSV loading strategies
        strategies = [
            {"encoding": "utf-8", "sep": ","},
            {"encoding": "utf-8", "sep": ";"},
            {"encoding": "latin-1", "sep": ","},
            {"encoding": "cp1252", "sep": ","},
            {"encoding": "utf-8", "sep": "\t"},
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                df = pd.read_csv(file_path, **strategy, low_memory=False)
                metadata["loading_strategy"] = f"Strategy {i+1}: {strategy}"
                logger.info(f"CSV loaded successfully with strategy: {strategy}")
                return df
            except Exception as e:
                if i == len(strategies) - 1:  # Last strategy failed
                    raise e
                continue
    
    def _load_excel_robust(self, file_path: str, metadata: Dict) -> pd.DataFrame:
        """Robust Excel loading with sheet detection."""
        try:
            # First, check available sheets
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            metadata["excel_sheets"] = sheets
            
            # Load the first sheet by default
            df = pd.read_excel(file_path, sheet_name=sheets[0])
            metadata["sheet_loaded"] = sheets[0]
            
            # If multiple sheets, warn user
            if len(sheets) > 1:
                metadata["warning"] = f"Multiple sheets found: {sheets}. Loaded first sheet: {sheets[0]}"
                logger.warning(f"Multiple Excel sheets detected: {sheets}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Excel loading failed: {str(e)}")
    
    def _load_json_robust(self, file_path: str, metadata: Dict) -> pd.DataFrame:
        """Robust JSON loading with format detection."""
        try:
            # Try different JSON orientations
            orientations = ['records', 'index', 'values', 'split', 'table']
            
            for orientation in orientations:
                try:
                    df = pd.read_json(file_path, orient=orientation)
                    metadata["json_orientation"] = orientation
                    return df
                except:
                    continue
            
            # If all orientations fail, try basic JSON loading
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            metadata["json_orientation"] = "custom_parsing"
            return df
            
        except Exception as e:
            raise Exception(f"JSON loading failed: {str(e)}")
    
    def _validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Perform initial dataset validation."""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "checks_performed": [],
            "warnings": [],
            "errors": []
        }
        
        # Check 1: Empty dataset
        if df.empty:
            validation["errors"].append("Dataset is empty")
            return validation
        validation["checks_performed"].append("empty_check")
        
        # Check 2: Minimum viable size
        if len(df) < 2:
            validation["warnings"].append(f"Very small dataset: only {len(df)} records")
        validation["checks_performed"].append("size_check")
        
        # Check 3: Column names validation
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            validation["errors"].append(f"Duplicate column names: {duplicate_cols}")
        validation["checks_performed"].append("column_names_check")
        
        # Check 4: Data quality assessment
        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()
        null_percentage = (null_cells / total_cells) * 100
        
        validation["data_quality"] = {
            "total_cells": total_cells,
            "null_cells": int(null_cells),
            "null_percentage": round(null_percentage, 2),
            "completeness": round(100 - null_percentage, 2)
        }
        
        if null_percentage > 50:
            validation["warnings"].append(f"High missing data: {null_percentage:.1f}%")
        validation["checks_performed"].append("data_quality_check")
        
        # Check 5: Memory usage warning
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 100:  # >100MB
            validation["warnings"].append(f"Large dataset: {memory_mb:.1f} MB")
        validation["checks_performed"].append("memory_check")
        
        # Check 6: Data type diversity
        dtype_counts = df.dtypes.value_counts().to_dict()
        validation["data_type_distribution"] = {str(k): v for k, v in dtype_counts.items()}
        validation["checks_performed"].append("dtype_check")
        
        # Check 7: Potential issues detection
        potential_issues = self._detect_potential_issues(df)
        validation["potential_issues"] = potential_issues
        validation["checks_performed"].append("issues_detection")
        
        logger.info(f"Validation completed: {len(validation['errors'])} errors, {len(validation['warnings'])} warnings")
        return validation
    
    def _detect_potential_issues(self, df: pd.DataFrame) -> Dict:
        """Detect common data quality issues."""
        issues = {
            "mixed_types": [],
            "suspicious_nulls": [],
            "encoding_issues": [],
            "numeric_as_text": []
        }
        
        for col in df.columns:
            # Check for mixed types in object columns
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    # Check for numeric values stored as strings
                    numeric_count = sum(str(val).replace('.', '').replace('-', '').replace(',', '').isdigit() 
                                      for val in sample if pd.notna(val))
                    if 0.3 < numeric_count / len(sample) < 0.9:
                        issues["mixed_types"].append(col)
                    elif numeric_count / len(sample) > 0.9:
                        issues["numeric_as_text"].append(col)
                    
                    # Check for encoding issues
                    text_sample = [str(val) for val in sample if pd.notna(val)]
                    for text in text_sample[:20]:  # Check first 20 text values
                        if any(ord(char) > 127 for char in text if isinstance(char, str)):
                            try:
                                text.encode('utf-8')
                            except UnicodeEncodeError:
                                if col not in issues["encoding_issues"]:
                                    issues["encoding_issues"].append(col)
                                break
            
            # Check for suspicious null patterns
            null_pct = df[col].isnull().mean()
            if 0.1 < null_pct < 0.9:  # Suspicious range
                issues["suspicious_nulls"].append({
                    "column": col,
                    "null_percentage": round(null_pct * 100, 1)
                })
        
        return issues
    
    def _compute_checksum(self, file_path: str) -> str:
        """Compute MD5 checksum for file integrity."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _save_raw_copy(self, df: pd.DataFrame, dataset_name: str) -> Path:
        """Save raw copy of loaded data."""
        raw_file = self.raw_dir / f"{dataset_name}_raw.csv"
        df.to_csv(raw_file, index=False)
        logger.info(f"Raw copy saved: {raw_file}")
        return raw_file
    
    def _log_ingestion(self, dataset_name: str, metadata: Dict):
        """Log ingestion process."""
        log_file = self.logs_dir / f"ingestion_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Ingestion logged: {log_file}")
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Retrieve information about an ingested dataset."""
        raw_file = self.raw_dir / f"{dataset_name}_raw.csv"
        
        if not raw_file.exists():
            return None
        
        try:
            df = pd.read_csv(raw_file)
            return {
                "dataset_name": dataset_name,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "file_size": raw_file.stat().st_size,
                "last_modified": datetime.fromtimestamp(raw_file.stat().st_mtime).isoformat()
            }
        except Exception as e:
            logger.error(f"Error reading dataset info: {e}")
            return None
    
    def list_ingested_datasets(self) -> list:
        """List all ingested datasets."""
        datasets = []
        for file_path in self.raw_dir.glob("*_raw.csv"):
            dataset_name = file_path.stem.replace("_raw", "")
            info = self.get_dataset_info(dataset_name)
            if info:
                datasets.append(info)
        return datasets

def main():
    """Example usage of DataIngestion class."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python ingestion.py <file_path> <dataset_name>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    dataset_name = sys.argv[2]
    
    # Initialize ingestion
    ingestion = DataIngestion()
    
    try:
        # Load dataset
        df, metadata = ingestion.load_dataset(file_path, dataset_name)
        
        print(f"\n✅ Successfully ingested dataset: {dataset_name}")
        print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory usage: {metadata['memory_usage_mb']:.2f} MB")
        print(f"🔍 Data quality: {metadata['validation_results']['data_quality']['completeness']:.1f}% complete")
        
        if metadata['validation_results']['warnings']:
            print(f"⚠️  Warnings: {len(metadata['validation_results']['warnings'])}")
            for warning in metadata['validation_results']['warnings']:
                print(f"   • {warning}")
        
        if metadata['validation_results']['errors']:
            print(f"❌ Errors: {len(metadata['validation_results']['errors'])}")
            for error in metadata['validation_results']['errors']:
                print(f"   • {error}")
    
    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()