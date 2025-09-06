# Volume score (adequate sample size)
        if len(df) >= 1000:
            volume_score = 100
        elif len(df) >= 100:
            volume_score = 70 + (len(df) - 100) / 900 * 30
        elif len(df) >= 10:
            volume_score = 30 + (len(df) - 10) / 90 * 40
        else:
            volume_score = max(0, len(df) / 10 * 30)
        
        scores["volume"] = round(volume_score, 1)
        
        # Validity score (based on validation issues)
        validation_info = log.get("validation_info", {})
        issues_count = len(validation_info.get("issues_found", []))
        validity_score = max(0, 100 - (issues_count * 10))  # -10 points per issue
        scores["validity"] = round(validity_score, 1)
        
        # Overall quality score (weighted average)
        overall_score = (
            scores["completeness"] * 0.3 +
            scores["type_consistency"] * 0.2 +
            scores["volume"] * 0.2 +
            scores["validity"] * 0.3
        )
        scores["overall"] = round(overall_score, 1)
        
        # Quality rating
        if overall_score >= 90:
            quality_rating = "Excellent"
        elif overall_score >= 80:
            quality_rating = "Good"
        elif overall_score >= 70:
            quality_rating = "Acceptable"
        elif overall_score >= 60:
            quality_rating = "Poor"
        else:
            quality_rating = "Very Poor"
        
        scores["rating"] = quality_rating
        
        return scores
    
    def _save_cleaned_dataset(self, df: pd.DataFrame, dataset_name: str):
        """Save cleaned dataset."""
        cleaned_file = self.cleaned_dir / f"{dataset_name}_cleaned.csv"
        df.to_csv(cleaned_file, index=False)
        logger.info(f"Cleaned dataset saved: {cleaned_file}")
    
    def _log_cleaning(self, dataset_name: str, log_data: Dict):
        """Log cleaning process."""
        log_file = self.logs_dir / f"cleaning_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Cleaning logged: {log_file}")
    
    def get_cleaned_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a cleaned dataset."""
        cleaned_file = self.cleaned_dir / f"{dataset_name}_cleaned.csv"
        
        if not cleaned_file.exists():
            raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_file}")
        
        return pd.read_csv(cleaned_file)
    
    def list_cleaned_datasets(self) -> List[str]:
        """List all cleaned datasets."""
        datasets = []
        for file_path in self.cleaned_dir.glob("*_cleaned.csv"):
            dataset_name = file_path.stem.replace("_cleaned", "")
            datasets.append(dataset_name)
        return datasets
    
    def get_cleaning_summary(self, dataset_name: str) -> Dict:
        """Get summary of cleaning operations for a dataset."""
        # Find most recent cleaning log
        log_files = sorted(
            self.logs_dir.glob(f"cleaning_{dataset_name}_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not log_files:
            return {"error": "No cleaning log found"}
        
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        # Extract key summary information
        summary = {
            "dataset_name": dataset_name,
            "cleaning_timestamp": log_data["timestamp"],
            "original_shape": log_data["original_shape"],
            "final_shape": log_data["final_shape"],
            "records_removed": log_data["records_removed"],
            "quality_score": log_data["quality_score"],
            "major_improvements": []
        }
        
        # Summarize major improvements
        if log_data["duplicate_info"]["duplicates_removed"] > 0:
            summary["major_improvements"].append(
                f"Removed {log_data['duplicate_info']['duplicates_removed']} duplicates"
            )
        
        if log_data["missing_value_info"]["columns_dropped"]:
            summary["major_improvements"].append(
                f"Dropped {len(log_data['missing_value_info']['columns_dropped'])} low-quality columns"
            )
        
        if log_data["outlier_info"].get("outliers_handled", 0) > 0:
            summary["major_improvements"].append(
                f"Handled {log_data['outlier_i#!/usr/bin/env python3
"""
RTGS Data Cleaning Module
Handles missing values, duplicates, outliers, and data quality validation
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataCleaning:
    """Handles comprehensive data cleaning and quality assurance."""
    
    def __init__(self, data_dir: str = "data", logs_dir: str = "logs"):
        self.data_dir = Path(data_dir)
        self.standardized_dir = self.data_dir / "standardized"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        for dir_path in [self.cleaned_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def clean_dataset(self, dataset_name: str, cleaning_config: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean a standardized dataset comprehensively.
        
        Args:
            dataset_name: Name of the dataset to clean
            cleaning_config: Optional configuration for cleaning parameters
            
        Returns:
            Tuple of (cleaned_DataFrame, cleaning_metadata)
        """
        logger.info(f"Starting cleaning for dataset: {dataset_name}")
        
        # Load standardized dataset
        df = self._load_standardized_dataset(dataset_name)
        original_shape = df.shape
        
        # Set default cleaning configuration
        config = self._get_default_config()
        if cleaning_config:
            config.update(cleaning_config)
        
        # Initialize cleaning log
        cleaning_log = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "original_shape": original_shape,
            "cleaning_config": config,
            "steps": [],
            "quality_improvements": {}
        }
        
        # Step 1: Remove duplicate records
        df, duplicate_info = self._remove_duplicates(df, cleaning_log, config)
        
        # Step 2: Handle missing values
        df, missing_value_info = self._handle_missing_values(df, cleaning_log, config)
        
        # Step 3: Detect and handle outliers
        df, outlier_info = self._handle_outliers(df, cleaning_log, config)
        
        # Step 4: Data validation and consistency checks
        df, validation_info = self._validate_data_consistency(df, cleaning_log, config)
        
        # Step 5: Quality scoring and assessment
        quality_score = self._calculate_quality_score(df, cleaning_log)
        
        # Compile final metadata
        cleaning_log.update({
            "final_shape": df.shape,
            "records_removed": original_shape[0] - df.shape[0],
            "duplicate_info": duplicate_info,
            "missing_value_info": missing_value_info,
            "outlier_info": outlier_info,
            "validation_info": validation_info,
            "quality_score": quality_score,
            "final_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
        
        # Save cleaned dataset
        self._save_cleaned_dataset(df, dataset_name)
        
        # Log cleaning process
        self._log_cleaning(dataset_name, cleaning_log)
        
        logger.info(f"Cleaning completed: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df, cleaning_log
    
    def _get_default_config(self) -> Dict:
        """Get default cleaning configuration."""
        return {
            "duplicate_removal": True,
            "missing_value_threshold": 0.7,  # Drop columns with >70% missing
            "outlier_method": "iqr",  # 'iqr', 'zscore', or 'none'
            "outlier_threshold": 1.5,  # IQR multiplier or Z-score threshold
            "outlier_action": "cap",  # 'remove', 'cap', or 'flag'
            "min_records": 10,  # Minimum records to keep dataset
            "consistency_checks": True
        }
    
    def _load_standardized_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load standardized dataset."""
        standardized_file = self.standardized_dir / f"{dataset_name}_standardized.csv"
        
        if not standardized_file.exists():
            raise FileNotFoundError(f"Standardized dataset not found: {standardized_file}")
        
        return pd.read_csv(standardized_file)
    
    def _remove_duplicates(self, df: pd.DataFrame, log: Dict, config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate records."""
        if not config.get("duplicate_removal", True):
            return df, {"skipped": True}
        
        initial_count = len(df)
        
        # Find duplicates
        duplicate_mask = df.duplicated()
        duplicate_count = duplicate_mask.sum()
        
        # Get some example duplicates for logging
        duplicate_examples = []
        if duplicate_count > 0:
            duplicate_rows = df[duplicate_mask].head(5)
            for idx, row in duplicate_rows.iterrows():
                duplicate_examples.append({
                    "index": int(idx),
                    "sample_values": {col: str(val)[:50] for col, val in row.head(3).items()}
                })
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates()
        final_count = len(df_cleaned)
        
        duplicate_info = {
            "initial_records": initial_count,
            "duplicates_found": int(duplicate_count),
            "duplicates_removed": int(duplicate_count),
            "final_records": final_count,
            "duplicate_percentage": round((duplicate_count / initial_count) * 100, 2),
            "examples": duplicate_examples
        }
        
        log["steps"].append({
            "step": "duplicate_removal",
            "description": f"Removed {duplicate_count} duplicate records",
            "records_before": initial_count,
            "records_after": final_count
        })
        
        logger.info(f"Removed {duplicate_count} duplicate records")
        return df_cleaned, duplicate_info
    
    def _handle_missing_values(self, df: pd.DataFrame, log: Dict, config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values intelligently."""
        missing_threshold = config.get("missing_value_threshold", 0.7)
        
        missing_info = {
            "columns_analyzed": len(df.columns),
            "columns_dropped": [],
            "imputation_strategies": {},
            "missing_summary_before": df.isnull().sum().to_dict(),
            "missing_summary_after": {}
        }
        
        columns_to_drop = []
        
        # Analyze missing values per column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df)
            
            if missing_pct > missing_threshold:
                # Drop columns with too much missing data
                columns_to_drop.append(col)
                missing_info["columns_dropped"].append({
                    "column": col,
                    "missing_percentage": round(missing_pct * 100, 2),
                    "reason": f"Exceeds {missing_threshold*100}% missing threshold"
                })
            
            elif missing_pct > 0:
                # Apply appropriate imputation strategy
                strategy = self._get_imputation_strategy(df[col])
                df[col] = self._apply_imputation(df[col], strategy)
                
                missing_info["imputation_strategies"][col] = {
                    "strategy": strategy["method"],
                    "fill_value": strategy["value"],
                    "missing_count": int(missing_count),
                    "missing_percentage": round(missing_pct * 100, 2)
                }
        
        # Drop columns with excessive missing data
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Dropped {len(columns_to_drop)} columns with excessive missing data")
        
        # Update missing summary after cleaning
        missing_info["missing_summary_after"] = df.isnull().sum().to_dict()
        missing_info["final_completeness"] = round(
            (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2
        )
        
        log["steps"].append({
            "step": "missing_value_handling",
            "description": f"Handled missing values: dropped {len(columns_to_drop)} columns, imputed {len(missing_info['imputation_strategies'])} columns",
            "columns_dropped": len(columns_to_drop),
            "columns_imputed": len(missing_info["imputation_strategies"])
        })
        
        return df, missing_info
    
    def _get_imputation_strategy(self, series: pd.Series) -> Dict:
        """Determine the best imputation strategy for a column."""
        if series.dtype in ['int64', 'float64', 'Int64']:
            # Numeric data - use median (robust to outliers)
            fill_value = series.median()
            return {"method": "median", "value": fill_value}
        
        elif series.dtype == 'datetime64[ns]':
            # DateTime data - use mode (most common date)
            mode_value = series.mode()
            fill_value = mode_value.iloc[0] if len(mode_value) > 0 else series.min()
            return {"method": "mode", "value": fill_value}
        
        elif series.dtype == 'bool':
            # Boolean data - use mode
            mode_value = series.mode()
            fill_value = mode_value.iloc[0] if len(mode_value) > 0 else False
            return {"method": "mode", "value": fill_value}
        
        else:
            # Categorical/Object data - use mode
            mode_value = series.mode()
            if len(mode_value) > 0:
                fill_value = mode_value.iloc[0]
            else:
                fill_value = "Unknown"
            return {"method": "mode", "value": fill_value}
    
    def _apply_imputation(self, series: pd.Series, strategy: Dict) -> pd.Series:
        """Apply imputation strategy to a series."""
        return series.fillna(strategy["value"])
    
    def _handle_outliers(self, df: pd.DataFrame, log: Dict, config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers in numeric columns."""
        outlier_method = config.get("outlier_method", "iqr")
        outlier_threshold = config.get("outlier_threshold", 1.5)
        outlier_action = config.get("outlier_action", "cap")
        
        if outlier_method == "none":
            return df, {"skipped": True, "reason": "Outlier handling disabled"}
        
        outlier_info = {
            "method": outlier_method,
            "threshold": outlier_threshold,
            "action": outlier_action,
            "columns_processed": [],
            "outliers_detected": 0,
            "outliers_handled": 0
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().all():
                continue
                
            outliers_mask, bounds = self._detect_outliers(df[col], outlier_method, outlier_threshold)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                column_info = {
                    "column": col,
                    "outliers_detected": int(outlier_count),
                    "outlier_percentage": round((outlier_count / len(df)) * 100, 2),
                    "bounds": bounds
                }
                
                # Apply outlier handling action
                if outlier_action == "remove":
                    df = df[~outliers_mask]
                    column_info["action_taken"] = "removed_records"
                    
                elif outlier_action == "cap":
                    df.loc[outliers_mask, col] = self._cap_outliers(df[col], bounds)
                    column_info["action_taken"] = "capped_values"
                    
                elif outlier_action == "flag":
                    flag_col = f"{col}_outlier_flag"
                    df[flag_col] = outliers_mask
                    column_info["action_taken"] = f"flagged_in_column_{flag_col}"
                
                outlier_info["columns_processed"].append(column_info)
                outlier_info["outliers_detected"] += outlier_count
                outlier_info["outliers_handled"] += outlier_count
        
        log["steps"].append({
            "step": "outlier_handling",
            "description": f"Handled {outlier_info['outliers_detected']} outliers using {outlier_method} method",
            "columns_processed": len(outlier_info["columns_processed"]),
            "outliers_detected": outlier_info["outliers_detected"]
        })
        
        logger.info(f"Handled {outlier_info['outliers_detected']} outliers in {len(outlier_info['columns_processed'])} columns")
        return df, outlier_info
    
    def _detect_outliers(self, series: pd.Series, method: str, threshold: float) -> Tuple[np.ndarray, Dict]:
        """Detect outliers using specified method."""
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            bounds = {"lower": float(lower_bound), "upper": float(upper_bound)}
            
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series.dropna()))
            # Align z_scores with original series (account for NaN values)
            outliers_mask = pd.Series(False, index=series.index)
            outliers_mask[series.notna()] = z_scores > threshold
            
            bounds = {
                "threshold": threshold,
                "mean": float(series.mean()),
                "std": float(series.std())
            }
        
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        return outliers_mask.values, bounds
    
    def _cap_outliers(self, series: pd.Series, bounds: Dict) -> pd.Series:
        """Cap outliers to specified bounds."""
        if "lower" in bounds and "upper" in bounds:
            return series.clip(lower=bounds["lower"], upper=bounds["upper"])
        return series
    
    def _validate_data_consistency(self, df: pd.DataFrame, log: Dict, config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Validate data consistency and logical constraints."""
        if not config.get("consistency_checks", True):
            return df, {"skipped": True}
        
        validation_info = {
            "checks_performed": [],
            "issues_found": [],
            "corrections_applied": []
        }
        
        # Check 1: Negative values in columns that should be positive
        positive_keywords = ['count', 'amount', 'quantity', 'total', 'sum', 'revenue', 'income', 'age', 'size']
        for col in df.select_dtypes(include=[np.number]).columns:
            if any(keyword in col.lower() for keyword in positive_keywords):
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    validation_info["issues_found"].append({
                        "issue": f"Negative values in '{col}' column",
                        "count": int(negative_count),
                        "action": "flagged_for_review"
                    })
                validation_info["checks_performed"].append(f"negative_values_check_{col}")
        
        # Check 2: Date consistency (future dates where not expected)
        current_date = datetime.now()
        for col in df.select_dtypes(include=['datetime64']).columns:
            if 'birth' in col.lower() or 'created' in col.lower() or 'established' in col.lower():
                future_dates = (df[col] > current_date).sum()
                if future_dates > 0:
                    validation_info["issues_found"].append({
                        "issue": f"Future dates in '{col}' column",
                        "count": int(future_dates),
                        "action": "flagged_for_review"
                    })
                validation_info["checks_performed"].append(f"future_dates_check_{col}")
        
        # Check 3: Percentage columns should be between 0-100 or 0-1
        percentage_keywords = ['percent', 'rate', 'ratio']
        for col in df.select_dtypes(include=[np.number]).columns:
            if any(keyword in col.lower() for keyword in percentage_keywords):
                max_val = df[col].max()
                min_val = df[col].min()
                
                if max_val > 100 or min_val < 0:
                    validation_info["issues_found"].append({
                        "issue": f"Unusual percentage values in '{col}' column",
                        "range": f"{min_val:.2f} to {max_val:.2f}",
                        "action": "flagged_for_review"
                    })
                validation_info["checks_performed"].append(f"percentage_range_check_{col}")
        
        # Check 4: Ensure minimum record count
        min_records = config.get("min_records", 10)
        if len(df) < min_records:
            validation_info["issues_found"].append({
                "issue": f"Dataset has only {len(df)} records (minimum: {min_records})",
                "action": "warning_issued"
            })
        validation_info["checks_performed"].append("minimum_records_check")
        
        log["steps"].append({
            "step": "consistency_validation",
            "description": f"Performed {len(validation_info['checks_performed'])} consistency checks",
            "checks_performed": len(validation_info["checks_performed"]),
            "issues_found": len(validation_info["issues_found"])
        })
        
        return df, validation_info
    
    def _calculate_quality_score(self, df: pd.DataFrame, log: Dict) -> Dict:
        """Calculate overall data quality score."""
        scores = {}
        
        # Completeness score (0-100)
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        scores["completeness"] = round(completeness, 1)
        
        # Consistency score (based on data types and formats)
        type_consistency = len([col for col in df.columns if df[col].dtype != 'object']) / len(df.columns)
        scores["type_consistency"] = round(type_consistency * 100, 1)
        
        # Volume score (adequate sample size)
        if len(df) >= 1000:
            volume_score = 100
        elif len