#!/usr/bin/env python3
"""
Simple Data Transform Module
Basic data transformations: feature engineering, encoding, scaling
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

class DataTransform:
    """Simple data transformation operations."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cleaned_dir = self.data_dir / "cleaned"
        self.transformed_dir = self.data_dir / "transformed"
        self.logs_dir = self.data_dir / "logs"
        
        # Create directories
        for dir_path in [self.transformed_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def transform_dataset(self, dataset_name: str, transform_config: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Transform a cleaned dataset.
        
        Args:
            dataset_name: Name of the dataset to transform
            transform_config: Configuration for transformations
            
        Returns:
            Tuple of (transformed_DataFrame, transform_stats)
        """
        logger.info(f"Starting transformation for dataset: {dataset_name}")
        
        # Load cleaned dataset
        df = self._load_cleaned_dataset(dataset_name)
        original_shape = df.shape
        
        # Default configuration
        config = {
            "create_features": True,
            "encode_categorical": True,
            "scale_numeric": True,
            "handle_dates": True
        }
        if transform_config:
            config.update(transform_config)
        
        # Initialize transform log
        transform_log = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "original_shape": original_shape,
            "config": config,
            "steps": [],
            "new_features": [],
            "encoded_columns": [],
            "scaled_columns": []
        }
        
        # Step 1: Handle date columns
        if config["handle_dates"]:
            df = self._handle_date_columns(df, transform_log)
        
        # Step 2: Create new features
        if config["create_features"]:
            df = self._create_features(df, transform_log)
        
        # Step 3: Encode categorical variables
        if config["encode_categorical"]:
            df = self._encode_categorical(df, transform_log)
        
        # Step 4: Scale numeric variables
        if config["scale_numeric"]:
            df = self._scale_numeric(df, transform_log)
        
        # Final stats
        final_shape = df.shape
        transform_log["final_shape"] = final_shape
        transform_log["features_added"] = final_shape[1] - original_shape[1]
        
        # Save transformed dataset
        self._save_transformed_dataset(df, dataset_name)
        self._save_transform_log(dataset_name, transform_log)
        
        logger.info(f"Transformation completed: {final_shape[0]} rows × {final_shape[1]} columns")
        return df, transform_log
    
    def _load_cleaned_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load cleaned dataset."""
        file_path = self.cleaned_dir / f"{dataset_name}_cleaned.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def _handle_date_columns(self, df: pd.DataFrame, log: Dict) -> pd.DataFrame:
        """Handle date columns and extract features."""
        date_columns = []
        
        # Find date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_columns.append(col)
                except:
                    continue
        
        # Extract date features
        for col in date_columns:
            # Extract year, month, day
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            
            log["new_features"].extend([
                f"{col}_year", f"{col}_month", 
                f"{col}_day", f"{col}_weekday"
            ])
        
        if date_columns:
            log["steps"].append(f"Extracted features from {len(date_columns)} date columns")
            logger.info(f"Processed {len(date_columns)} date columns")
        
        return df
    
    def _create_features(self, df: pd.DataFrame, log: Dict) -> pd.DataFrame:
        """Create new features from existing data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        new_features = []
        
        # Create ratio features for numeric columns
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if (df[col2] != 0).all():  # Avoid division by zero
                    ratio_col = f"{col1}_to_{col2}_ratio"
                    df[ratio_col] = df[col1] / df[col2]
                    new_features.append(ratio_col)
                    
                    # Limit to prevent too many features
                    if len(new_features) >= 5:
                        break
            if len(new_features) >= 5:
                break
        
        # Create categorical frequency features
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 50:  # Only for low cardinality
                freq_col = f"{col}_frequency"
                df[freq_col] = df[col].map(df[col].value_counts())
                new_features.append(freq_col)
        
        if new_features:
            log["new_features"].extend(new_features)
            log["steps"].append(f"Created {len(new_features)} new features")
            logger.info(f"Created {len(new_features)} new features")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, log: Dict) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        encoded_cols = []
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            if unique_count <= 10:  # One-hot encode low cardinality
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                encoded_cols.append(f"{col} (one-hot)")
                
            elif unique_count <= 50:  # Label encode medium cardinality
                # Label encoding
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                encoded_cols.append(f"{col} (label)")
                
            else:  # Keep high cardinality as is or drop
                log["steps"].append(f"Kept {col} as-is (high cardinality: {unique_count})")
        
        if encoded_cols:
            log["encoded_columns"] = encoded_cols
            log["steps"].append(f"Encoded {len(encoded_cols)} categorical columns")
            logger.info(f"Encoded {len(encoded_cols)} categorical columns")
        
        return df
    
    def _scale_numeric(self, df: pd.DataFrame, log: Dict) -> pd.DataFrame:
        """Scale numeric variables."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaled_cols = []
        
        # Use StandardScaler for most columns
        scaler = StandardScaler()
        
        for col in numeric_cols:
            # Skip binary columns (0s and 1s only)
            unique_values = df[col].unique()
            if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                continue
                
            # Scale the column
            try:
                df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
                scaled_cols.append(col)
            except:
                log["steps"].append(f"Could not scale {col}")
        
        if scaled_cols:
            log["scaled_columns"] = scaled_cols
            log["steps"].append(f"Scaled {len(scaled_cols)} numeric columns")
            logger.info(f"Scaled {len(scaled_cols)} numeric columns")
        
        return df
    
    def _save_transformed_dataset(self, df: pd.DataFrame, dataset_name: str):
        """Save transformed dataset."""
        file_path = self.transformed_dir / f"{dataset_name}_transformed.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Transformed dataset saved: {file_path}")
    
    def _save_transform_log(self, dataset_name: str, log: Dict):
        """Save transform log."""
        log_file = self.logs_dir / f"transform_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2, default=str)
        
        logger.info(f"Transform log saved: {log_file}")
    
    def get_transformed_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a transformed dataset."""
        file_path = self.transformed_dir / f"{dataset_name}_transformed.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Transformed dataset not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def reverse_transform(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Simple reverse transformation (remove scaled/encoded columns)."""
        # This is a basic implementation - in practice you'd want to save
        # transformation parameters and properly reverse them
        
        cols_to_remove = []
        for col in df.columns:
            if '_scaled' in col or '_encoded' in col or '_ratio' in col:
                cols_to_remove.append(col)
        
        df_original = df.drop(columns=cols_to_remove)
        logger.info(f"Removed {len(cols_to_remove)} transformed columns")
        
        return df_original

def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python transform.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Initialize transformer
    transformer = DataTransform()
    
    try:
        # Transform dataset
        df, log = transformer.transform_dataset(dataset_name)
        
        print(f"\n✅ Successfully transformed dataset: {dataset_name}")
        print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"🔧 Operations performed: {len(log['steps'])}")
        print(f"✨ New features created: {log['features_added']}")
        
        for step in log['steps']:
            print(f"   • {step}")
        
        if log['new_features']:
            print(f"\n🆕 New features: {', '.join(log['new_features'][:5])}...")
        
        print(f"\n💾 Transformed data saved to: data/transformed/{dataset_name}_transformed.csv")
    
    except Exception as e:
        print(f"❌ Transformation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()