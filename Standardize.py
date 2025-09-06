#!/usr/bin/env python3
"""
RTGS Data Standardization Module
Handles column naming, data type standardization, and format normalization
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class DataStandardization:
    """Handles data standardization and format normalization."""
    
    def __init__(self, data_dir: str = "data", logs_dir: str = "logs"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.standardized_dir = self.data_dir / "standardized"
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        for dir_path in [self.standardized_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def standardize_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize a dataset that was previously ingested.
        
        Args:
            dataset_name: Name of the dataset to standardize
            
        Returns:
            Tuple of (standardized_DataFrame, standardization_metadata)
        """
        logger.info(f"Starting standardization for dataset: {dataset_name}")
        
        # Load raw dataset
        df = self._load_raw_dataset(dataset_name)
        original_shape = df.shape
        original_columns = list(df.columns)
        original_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Track all standardization steps
        standardization_log = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "original_shape": original_shape,
            "original_columns": original_columns,
            "original_dtypes": original_dtypes,
            "steps": []
        }
        
        # Step 1: Standardize column names
        df, column_mapping = self._standardize_column_names(df, standardization_log)
        
        # Step 2: Standardize data types
        df, type_conversions = self._standardize_data_types(df, standardization_log)
        
        # Step 3: Standardize text data
        df, text_standardizations = self._standardize_text_data(df, standardization_log)
        
        # Step 4: Standardize numeric data
        df, numeric_standardizations = self._standardize_numeric_data(df, standardization_log)
        
        # Step 5: Standardize date/time data
        df, date_standardizations = self._standardize_datetime_data(df, standardization_log)
        
        # Step 6: Standardize categorical data
        df, categorical_standardizations = self._standardize_categorical_data(df, standardization_log)
        
        # Compile final metadata
        standardization_log.update({
            "final_shape": df.shape,
            "final_columns": list(df.columns),
            "final_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "column_mapping": column_mapping,
            "type_conversions": type_conversions,
            "text_standardizations": text_standardizations,
            "numeric_standardizations": numeric_standardizations,
            "date_standardizations": date_standardizations,
            "categorical_standardizations": categorical_standardizations
        })
        
        # Save standardized dataset
        self._save_standardized_dataset(df, dataset_name)
        
        # Log standardization
        self._log_standardization(dataset_name, standardization_log)
        
        logger.info(f"Standardization completed: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df, standardization_log
    
    def _load_raw_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load raw dataset from ingestion output."""
        raw_file = self.raw_dir / f"{dataset_name}_raw.csv"
        
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw dataset not found: {raw_file}")
        
        return pd.read_csv(raw_file)
    
    def _standardize_column_names(self, df: pd.DataFrame, log: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Standardize column names to snake_case format."""
        original_columns = list(df.columns)
        
        # Convert to snake_case
        standardized_columns = []
        for col in original_columns:
            # Convert to string and handle edge cases
            col_str = str(col).strip()
            
            # Replace special characters and spaces with underscores
            standardized = re.sub(r'[^a-zA-Z0-9]+', '_', col_str)
            
            # Convert to lowercase
            standardized = standardized.lower()
            
            # Remove leading/trailing underscores
            standardized = standardized.strip('_')
            
            # Handle empty column names
            if not standardized:
                standardized = f"column_{len(standardized_columns)}"
            
            # Handle duplicate column names
            base_name = standardized
            counter = 1
            while standardized in standardized_columns:
                standardized = f"{base_name}_{counter}"
                counter += 1
            
            standardized_columns.append(standardized)
        
        # Apply new column names
        column_mapping = dict(zip(original_columns, standardized_columns))
        df.columns = standardized_columns
        
        log["steps"].append({
            "step": "column_name_standardization",
            "description": "Converted column names to snake_case format",
            "columns_changed": sum(1 for orig, new in column_mapping.items() if orig != new)
        })
        
        logger.info(f"Standardized {len(column_mapping)} column names")
        return df, column_mapping
    
    def _standardize_data_types(self, df: pd.DataFrame, log: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Intelligently standardize data types."""
        type_conversions = {}
        
        for col in df.columns:
            original_type = str(df[col].dtype)
            converted = False
            
            # Skip if already proper numeric type
            if df[col].dtype in ['int64', 'float64']:
                continue
            
            # Try datetime conversion first
            if df[col].dtype == 'object':
                if self._looks_like_datetime(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                        type_conversions[col] = f"{original_type} → datetime64[ns]"
                        converted = True
                    except:
                        pass
            
            # Try numeric conversion if not converted to datetime
            if not converted and df[col].dtype == 'object':
                if self._looks_like_numeric(df[col]):
                    try:
                        # Clean numeric data first
                        cleaned_series = self._clean_numeric_strings(df[col])
                        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                        
                        # Only convert if most values are successfully converted
                        valid_conversion_rate = numeric_series.notna().sum() / len(numeric_series)
                        if valid_conversion_rate > 0.8:
                            df[col] = numeric_series
                            type_conversions[col] = f"{original_type} → {str(numeric_series.dtype)}"
                            converted = True
                    except:
                        pass
            
            # Try boolean conversion
            if not converted and df[col].dtype == 'object':
                if self._looks_like_boolean(df[col]):
                    try:
                        df[col] = df[col].map(self._convert_to_boolean)
                        type_conversions[col] = f"{original_type} → bool"
                        converted = True
                    except:
                        pass
        
        log["steps"].append({
            "step": "data_type_standardization",
            "description": "Converted data types based on content analysis",
            "types_converted": len(type_conversions)
        })
        
        logger.info(f"Converted data types for {len(type_conversions)} columns")
        return df, type_conversions
    
    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """Check if series contains datetime-like values."""
        sample = series.dropna().astype(str).head(100)
        if len(sample) == 0:
            return False
        
        # Common datetime patterns
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',           # MM/DD/YYYY or DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',           # MM-DD-YYYY or DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}',           # YYYY/MM/DD
            r'\d{1,2}-\w{3}-\d{4}',         # DD-Mon-YYYY
            r'\w{3}\s+\d{1,2},\s+\d{4}',    # Mon DD, YYYY
        ]
        
        for pattern in datetime_patterns:
            matches = sample.str.match(pattern, na=False).sum()
            if matches > len(sample) * 0.6:  # 60% match threshold
                return True
        
        return False
    
    def _looks_like_numeric(self, series: pd.Series) -> bool:
        """Check if series contains numeric-like values."""
        sample = series.dropna().astype(str).head(200)
        if len(sample) == 0:
            return False
        
        # Numeric patterns (including currency, percentages, etc.)
        numeric_patterns = [
            r'^[+-]?\d+$',                      # Integer
            r'^[+-]?\d+\.\d+$',                 # Decimal
            r'^[+-]?\d{1,3}(,\d{3})*(\.\d+)?$', # Comma-separated numbers
            r'^[₹$€£¥]\s*\d+(\.\d+)?$',        # Currency
            r'^\d+(\.\d+)?%$',                  # Percentage
            r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$'   # Scientific notation
        ]
        
        numeric_count = 0
        for value in sample:
            for pattern in numeric_patterns:
                if re.match(pattern, str(value)):
                    numeric_count += 1
                    break
        
        return numeric_count > len(sample) * 0.7  # 70% match threshold
    
    def _looks_like_boolean(self, series: pd.Series) -> bool:
        """Check if series contains boolean-like values."""
        unique_values = set(str(v).lower() for v in series.dropna().unique())
        
        boolean_sets = [
            {'true', 'false'},
            {'yes', 'no'},
            {'y', 'n'},
            {'1', '0'},
            {'on', 'off'},
            {'enabled', 'disabled'}
        ]
        
        for bool_set in boolean_sets:
            if unique_values.issubset(bool_set) and len(unique_values) == 2:
                return True
        
        return False
    
    def _clean_numeric_strings(self, series: pd.Series) -> pd.Series:
        """Clean numeric strings by removing currency symbols, commas, etc."""
        cleaned = series.astype(str)
        
        # Remove currency symbols
        cleaned = cleaned.str.replace(r'[₹$€£¥]', '', regex=True)
        
        # Remove commas in numbers
        cleaned = cleaned.str.replace(',', '')
        
        # Remove percentage signs
        cleaned = cleaned.str.replace('%', '')
        
        # Remove extra whitespace
        cleaned = cleaned.str.strip()
        
        # Replace empty strings with NaN
        cleaned = cleaned.replace('', np.nan)
        cleaned = cleaned.replace('nan', np.nan)
        
        return cleaned
    
    def _convert_to_boolean(self, value):
        """Convert value to boolean."""
        if pd.isna(value):
            return np.nan
        
        val_str = str(value).lower()
        
        if val_str in ['true', 'yes', 'y', '1', 'on', 'enabled']:
            return True
        elif val_str in ['false', 'no', 'n', '0', 'off', 'disabled']:
            return False
        else:
            return np.nan
    
    def _standardize_text_data(self, df: pd.DataFrame, log: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Standardize text data (encoding, whitespace, etc.)."""
        text_standardizations = {}
        
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            changes = []
            
            # Remove leading/trailing whitespace
            df[col] = df[col].str.strip()
            changes.append("whitespace_stripped")
            
            # Standardize empty strings to NaN
            df[col] = df[col].replace('', np.nan)
            changes.append("empty_strings_to_nan")
            
            # Fix common encoding issues
            if df[col].dtype == 'object':
                # Replace common problematic characters
                df[col] = df[col].str.replace(r'â€™', "'", regex=True)  # Smart quote
                df[col] = df[col].str.replace(r'â€œ|â€\x9d', '"', regex=True)  # Smart quotes
                changes.append("encoding_fixes")
            
            if changes:
                text_standardizations[col] = changes
        
        log["steps"].append({
            "step": "text_data_standardization",
            "description": "Standardized text data formatting",
            "columns_processed": len(text_standardizations)
        })
        
        return df, text_standardizations
    
    def _standardize_numeric_data(self, df: pd.DataFrame, log: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Standardize numeric data formats and ranges."""
        numeric_standardizations = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            changes = []
            original_stats = {
                "min": float(df[col].min()) if not df[col].empty else None,
                "max": float(df[col].max()) if not df[col].empty else None,
                "mean": float(df[col].mean()) if not df[col].empty else None
            }
            
            # Handle infinite values
            if np.isinf(df[col]).any():
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                changes.append("infinite_values_to_nan")
            
            # Round very long decimals
            if df[col].dtype == 'float64':
                # Check if values have excessive decimal places
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    decimal_places = sample.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                    if decimal_places.mean() > 6:  # More than 6 decimal places on average
                        df[col] = df[col].round(6)
                        changes.append("rounded_to_6_decimals")
            
            # Convert float columns to int if they're whole numbers
            if df[col].dtype == 'float64':
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0 and (non_null_values % 1 == 0).all():
                    df[col] = df[col].astype('Int64')  # Nullable integer
                    changes.append("converted_float_to_int")
            
            if changes:
                numeric_standardizations[col] = {
                    "changes": changes,
                    "original_stats": original_stats
                }
        
        log["steps"].append({
            "step": "numeric_data_standardization",
            "description": "Standardized numeric data formats",
            "columns_processed": len(numeric_standardizations)
        })
        
        return df, numeric_standardizations
    
    def _standardize_datetime_data(self, df: pd.DataFrame, log: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Standardize datetime data to consistent formats."""
        date_standardizations = {}
        
        # Find datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            changes = []
            
            # Standardize timezone (remove timezone info for consistency)
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
                changes.append("timezone_removed")
            
            # Add derived date components if useful
            date_range = df[col].max() - df[col].min()
            if pd.notna(date_range) and date_range.days > 365:  # More than 1 year range
                # Add year column if date spans multiple years
                year_col = f"{col}_year"
                if year_col not in df.columns:
                    df[year_col] = df[col].dt.year
                    changes.append(f"added_year_column_{year_col}")
                
                # Add month column if useful
                month_col = f"{col}_month"
                if month_col not in df.columns:
                    df[month_col] = df[col].dt.month
                    changes.append(f"added_month_column_{month_col}")
            
            if changes:
                date_standardizations[col] = changes
        
        log["steps"].append({
            "step": "datetime_standardization",
            "description": "Standardized datetime formats and added derived columns",
            "columns_processed": len(date_standardizations)
        })
        
        return df, date_standardizations
    
    def _standardize_categorical_data(self, df: pd.DataFrame, log: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Standardize categorical data for consistency."""
        categorical_standardizations = {}
        
        # Find categorical columns (object type with reasonable number of unique values)
        categorical_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            
            # Consider categorical if unique values are less than 50% of total
            if unique_count < total_count * 0.5 and unique_count < 100:
                categorical_columns.append(col)
        
        for col in categorical_columns:
            changes = []
            original_unique_count = df[col].nunique()
            
            # Standardize case (convert to title case for consistency)
            if df[col].dtype == 'object':
                # Only apply title case if most values are text (not codes/IDs)
                sample = df[col].dropna().head(50)
                text_values = sum(1 for val in sample if isinstance(val, str) and any(c.isalpha() for c in val))
                
                if text_values > len(sample) * 0.7:  # 70% are text values
                    df[col] = df[col].str.title()
                    changes.append("converted_to_title_case")
            
            # Fix common inconsistencies
            if df[col].dtype == 'object':
                # Standardize common variations
                standardization_map = {
                    # Boolean-like variations
                    'Y': 'Yes', 'N': 'No',
                    'T': 'True', 'F': 'False',
                    '1': 'Yes', '0': 'No',
                    
                    # Common abbreviations
                    'M': 'Male', 'F': 'Female',
                    'govt': 'Government', 'pvt': 'Private',
                    'std': 'Standard', 'qty': 'Quantity',
                    
                    # Status variations
                    'active': 'Active', 'inactive': 'Inactive',
                    'enabled': 'Enabled', 'disabled': 'Disabled'
                }
                
                for old_val, new_val in standardization_map.items():
                    if df[col].str.lower().str.contains(old_val, na=False).any():
                        df[col] = df[col].str.replace(old_val, new_val, case=False)
                        changes.append(f"standardized_{old_val}_to_{new_val}")
            
            # Remove extra spaces in categorical values
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            changes.append("normalized_spaces")
            
            final_unique_count = df[col].nunique()
            
            if changes:
                categorical_standardizations[col] = {
                    "changes": changes,
                    "original_unique_count": original_unique_count,
                    "final_unique_count": final_unique_count
                }
        
        log["steps"].append({
            "step": "categorical_data_standardization",
            "description": "Standardized categorical values for consistency",
            "columns_processed": len(categorical_standardizations)
        })
        
        return df, categorical_standardizations
    
    def _save_standardized_dataset(self, df: pd.DataFrame, dataset_name: str):
        """Save standardized dataset."""
        standardized_file = self.standardized_dir / f"{dataset_name}_standardized.csv"
        df.to_csv(standardized_file, index=False)
        logger.info(f"Standardized dataset saved: {standardized_file}")
    
    def _log_standardization(self, dataset_name: str, log_data: Dict):
        """Log standardization process."""
        log_file = self.logs_dir / f"standardization_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Standardization logged: {log_file}")
    
    def get_standardized_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a standardized dataset."""
        standardized_file = self.standardized_dir / f"{dataset_name}_standardized.csv"
        
        if not standardized_file.exists():
            raise FileNotFoundError(f"Standardized dataset not found: {standardized_file}")
        
        return pd.read_csv(standardized_file)
    
    def list_standardized_datasets(self) -> List[str]:
        """List all standardized datasets."""
        datasets = []
        for file_path in self.standardized_dir.glob("*_standardized.csv"):
            dataset_name = file_path.stem.replace("_standardized", "")
            datasets.append(dataset_name)
        return datasets

def main():
    """Example usage of DataStandardization class."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python standardize.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Initialize standardization
    standardizer = DataStandardization()
    
    try:
        # Standardize dataset
        df, log_data = standardizer.standardize_dataset(dataset_name)
        
        print(f"\n✅ Successfully standardized dataset: {dataset_name}")
        print(f"📊 Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Show key transformations
        steps_completed = len(log_data['steps'])
        print(f"🔧 Completed {steps_completed} standardization steps:")
        
        for step in log_data['steps']:
            print(f"   • {step['description']}")
        
        # Show column changes
        column_changes = sum(1 for orig, new in log_data['column_mapping'].items() if orig != new)
        if column_changes > 0:
            print(f"📝 Renamed {column_changes} columns to snake_case format")
        
        # Show type conversions
        type_changes = len(log_data['type_conversions'])
        if type_changes > 0:
            print(f"🔄 Converted data types for {type_changes} columns")
            for col, conversion in list(log_data['type_conversions'].items())[:5]:
                print(f"   • {col}: {conversion}")
        
        print(f"\n💾 Standardized data saved to: data/standardized/{dataset_name}_standardized.csv")
    
    except Exception as e:
        print(f"❌ Standardization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()