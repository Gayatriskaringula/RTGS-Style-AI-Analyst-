#!/usr/bin/env python3
"""
Simple Data Cleaning Module
Basic data cleaning operations: duplicates, missing values, outliers
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DataCleaning:
    """Simple data cleaning operations."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.standardized_dir = self.data_dir / "standardized"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.logs_dir = self.data_dir / "logs"
        
        # Create directories
        for dir_path in [self.cleaned_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def clean_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean a dataset with basic operations.
        
        Args:
            dataset_name: Name of the dataset to clean
            
        Returns:
            Tuple of (cleaned_DataFrame, cleaning_stats)
        """
        logger.info(f"Starting cleaning for dataset: {dataset_name}")
        
        # Load standardized dataset
        df = self._load_standardized_dataset(dataset_name)
        original_shape = df.shape
        
        # Initialize cleaning log
        cleaning_log = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "original_shape": original_shape,
            "steps": []
        }
        
        # Step 1: Remove duplicates
        df = self._remove_duplicates(df, cleaning_log)
        
        # Step 2: Handle missing values
        df = self._handle_missing_values(df, cleaning_log)
        
        # Step 3: Handle outliers
        df = self._handle_outliers(df, cleaning_log)
        
        # Final stats
        final_shape = df.shape
        cleaning_log["final_shape"] = final_shape
        cleaning_log["rows_removed"] = original_shape[0] - final_shape[0]
        cleaning_log["cols_removed"] = original_shape[1] - final_shape[1]
        
        # Save cleaned dataset
        self._save_cleaned_dataset(df, dataset_name)
        self._save_cleaning_log(dataset_name, cleaning_log)
        
        logger.info(f"Cleaning completed: {final_shape[0]} rows × {final_shape[1]} columns")
        return df, cleaning_log
    
    def _load_standardized_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load standardized dataset."""
        file_path = self.standardized_dir / f"{dataset_name}_standardized.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Standardized dataset not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def _remove_duplicates(self, df: pd.DataFrame, log: Dict) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        final_rows = len(df_clean)
        
        duplicates_removed = initial_rows - final_rows
        
        if duplicates_removed > 0:
            log["steps"].append(f"Removed {duplicates_removed} duplicate rows")
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame, log: Dict) -> pd.DataFrame:
        """Handle missing values with simple strategies."""
        
        # Drop columns with >70% missing data
        missing_threshold = 0.7
        cols_to_drop = []
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > missing_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            log["steps"].append(f"Dropped {len(cols_to_drop)} columns with >70% missing data")
            logger.info(f"Dropped columns: {cols_to_drop}")
        
        # Fill remaining missing values
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    log["steps"].append(f"Filled {col} missing values with median: {fill_value}")
                else:
                    # Fill categorical columns with mode or 'Unknown'
                    mode_values = df[col].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_value)
                    log["steps"].append(f"Filled {col} missing values with: {fill_value}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, log: Dict) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_handled = 0
        
        for col in numeric_cols:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers (don't remove them)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_handled += outliers
        
        if outliers_handled > 0:
            log["steps"].append(f"Capped {outliers_handled} outliers using IQR method")
            logger.info(f"Handled {outliers_handled} outliers")
        
        return df
    
    def _save_cleaned_dataset(self, df: pd.DataFrame, dataset_name: str):
        """Save cleaned dataset."""
        file_path = self.cleaned_dir / f"{dataset_name}_cleaned.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Cleaned dataset saved: {file_path}")
    
    def _save_cleaning_log(self, dataset_name: str, log: Dict):
        """Save cleaning log."""
        log_file = self.logs_dir / f"cleaning_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2, default=str)
        
        logger.info(f"Cleaning log saved: {log_file}")
    
    def get_cleaned_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a cleaned dataset."""
        file_path = self.cleaned_dir / f"{dataset_name}_cleaned.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")
        
        return pd.read_csv(file_path)

def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python clean.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Initialize cleaner
    cleaner = DataCleaning()
    
    try:
        # Clean dataset
        df, log = cleaner.clean_dataset(dataset_name)
        
        print(f"\n✅ Successfully cleaned dataset: {dataset_name}")
        print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"🔧 Operations performed: {len(log['steps'])}")
        
        for step in log['steps']:
            print(f"   • {step}")
        
        print(f"\n💾 Cleaned data saved to: data/cleaned/{dataset_name}_cleaned.csv")
    
    except Exception as e:
        print(f"❌ Cleaning failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()