#!/usr/bin/env python3
"""
Analyze Module with Llama
Analyze transformed data using Llama model and generate ASCII charts
"""

import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Analyze data using Llama model and generate insights with ASCII charts."""
    
    def __init__(self, data_dir: str = "data", llama_url: str = "http://localhost:11434"):
        self.data_dir = Path(data_dir)
        self.transformed_dir = self.data_dir / "transformed"
        self.analyzed_dir = self.data_dir / "analyzed"
        self.logs_dir = self.data_dir / "logs"
        self.llama_url = llama_url
        
        # Create directories
        for dir_path in [self.analyzed_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def analyze_dataset(self, dataset_name: str) -> Dict:
        """
        Analyze dataset and generate insights using Llama.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Analysis results with insights and ASCII charts
        """
        logger.info(f"Starting analysis for dataset: {dataset_name}")
        
        # Load transformed dataset
        df = self._load_transformed_dataset(dataset_name)
        
        # Generate basic statistics
        basic_stats = self._generate_basic_stats(df)
        
        # Create ASCII charts
        ascii_charts = self._create_ascii_charts(df)
        
        # Generate insights using Llama
        llama_insights = self._generate_llama_insights(df, basic_stats, dataset_name)
        
        # Compile analysis results
        analysis_results = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object']).columns)
            },
            "basic_statistics": basic_stats,
            "ascii_charts": ascii_charts,
            "llama_insights": llama_insights,
            "key_findings": self._extract_key_findings(basic_stats, llama_insights)
        }
        
        # Save analysis results
        self._save_analysis_results(analysis_results, dataset_name)
        
        # Display results
        self._display_analysis(analysis_results)
        
        logger.info(f"Analysis completed for {dataset_name}")
        return analysis_results
    
    def _load_transformed_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load transformed dataset."""
        file_path = self.transformed_dir / f"{dataset_name}_transformed.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Transformed dataset not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def _generate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistical analysis."""
        stats = {}
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            stats["numeric_summary"] = numeric_stats.to_dict()
            
            # Correlation matrix for top numeric columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                stats["correlations"] = corr_matrix.to_dict()
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        cat_stats = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(5)
            cat_stats[col] = {
                "unique_count": df[col].nunique(),
                "top_5_values": value_counts.to_dict()
            }
        stats["categorical_summary"] = cat_stats
        
        # Missing values
        missing_data = df.isnull().sum()
        stats["missing_values"] = missing_data[missing_data > 0].to_dict()
        
        return stats
    
    def _create_ascii_charts(self, df: pd.DataFrame) -> Dict:
        """Create ASCII charts for visualization."""
        charts = {}
        
        # Chart for numeric columns (histogram)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            charts[f"{col}_histogram"] = self._create_ascii_histogram(df[col], col)
        
        # Chart for categorical columns (bar chart)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            if df[col].nunique() <= 10:  # Only for low cardinality
                charts[f"{col}_bar_chart"] = self._create_ascii_bar_chart(df[col], col)
        
        # Correlation heatmap for numeric data
        if len(numeric_cols) >= 2:
            charts["correlation_heatmap"] = self._create_correlation_heatmap(df[numeric_cols])
        
        return charts
    
    def _create_ascii_histogram(self, series: pd.Series, title: str, width: int = 50) -> str:
        """Create ASCII histogram."""
        if series.empty or series.isna().all():
            return f"\n{title} Histogram:\nNo data available\n"
        
        # Create bins
        values = series.dropna()
        hist, bin_edges = np.histogram(values, bins=10)
        
        # Normalize to fit width
        max_count = max(hist) if max(hist) > 0 else 1
        
        chart = f"\n{title} Histogram:\n"
        chart += "=" * (width + 20) + "\n"
        
        for i, count in enumerate(hist):
            bar_length = int((count / max_count) * width)
            bar = "█" * bar_length
            range_str = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
            chart += f"{range_str:>12} |{bar:<{width}} {count}\n"
        
        chart += "=" * (width + 20) + "\n"
        return chart
    
    def _create_ascii_bar_chart(self, series: pd.Series, title: str, width: int = 40) -> str:
        """Create ASCII bar chart for categorical data."""
        if series.empty:
            return f"\n{title} Bar Chart:\nNo data available\n"
        
        value_counts = series.value_counts().head(8)  # Top 8 values
        max_count = max(value_counts) if len(value_counts) > 0 else 1
        
        chart = f"\n{title} Bar Chart:\n"
        chart += "=" * (width + 25) + "\n"
        
        for value, count in value_counts.items():
            bar_length = int((count / max_count) * width)
            bar = "█" * bar_length
            value_str = str(value)[:15]  # Truncate long values
            chart += f"{value_str:>15} |{bar:<{width}} {count}\n"
        
        chart += "=" * (width + 25) + "\n"
        return chart
    
    def _create_correlation_heatmap(self, df_numeric: pd.DataFrame) -> str:
        """Create ASCII correlation heatmap."""
        if len(df_numeric.columns) < 2:
            return "\nCorrelation Heatmap:\nInsufficient numeric columns\n"
        
        corr = df_numeric.corr()
        
        # Create ASCII heatmap
        chart = "\nCorrelation Heatmap:\n"
        chart += "=" * 60 + "\n"
        
        # Header
        col_names = [col[:8] for col in corr.columns[:6]]  # Limit columns and truncate names
        chart += f"{'':>10}"
        for name in col_names:
            chart += f"{name:>8}"
        chart += "\n"
        
        # Correlation values with symbols
        for i, row_name in enumerate(corr.index[:6]):
            chart += f"{row_name[:10]:>10}"
            for j, col_name in enumerate(corr.columns[:6]):
                value = corr.iloc[i, j]
                if abs(value) > 0.7:
                    symbol = "██"  # Strong correlation
                elif abs(value) > 0.4:
                    symbol = "▓▓"  # Moderate correlation
                elif abs(value) > 0.2:
                    symbol = "░░"  # Weak correlation
                else:
                    symbol = "  "  # No correlation
                
                if value < 0:
                    symbol = "▼▼" if abs(value) > 0.5 else "▽▽"
                
                chart += f"{symbol:>8}"
            chart += f"  {row_name[:10]}\n"
        
        chart += "\nLegend: ██ Strong(>0.7) ▓▓ Moderate(>0.4) ░░ Weak(>0.2) ▼▼ Negative\n"
        chart += "=" * 60 + "\n"
        
        return chart
    
    def _generate_llama_insights(self, df: pd.DataFrame, basic_stats: Dict, dataset_name: str) -> Dict:
        """Generate insights using Llama model."""
        try:
            # Prepare data summary for Llama
            data_summary = self._prepare_data_summary(df, basic_stats, dataset_name)
            
            # Generate insights prompt
            insights_prompt = self._create_insights_prompt(data_summary, dataset_name)
            
            # Call Llama API
            insights_response = self._call_llama_api(insights_prompt, "insights")
            
            # Generate governance recommendations
            governance_prompt = self._create_governance_prompt(data_summary, dataset_name)
            governance_response = self._call_llama_api(governance_prompt, "governance")
            
            return {
                "data_insights": insights_response,
                "governance_recommendations": governance_response,
                "analysis_quality": "success"
            }
            
        except Exception as e:
            logger.error(f"Llama analysis failed: {e}")
            return {
                "data_insights": "Llama analysis unavailable - using fallback analysis",
                "governance_recommendations": self._generate_fallback_insights(df, basic_stats),
                "analysis_quality": "fallback",
                "error": str(e)
            }
    
    def _prepare_data_summary(self, df: pd.DataFrame, basic_stats: Dict, dataset_name: str) -> str:
        """Prepare concise data summary for Llama."""
        summary = f"Dataset: {dataset_name}\n"
        summary += f"Rows: {len(df):,}, Columns: {len(df.columns)}\n\n"
        
        # Numeric summary
        if "numeric_summary" in basic_stats:
            summary += "Key Numeric Statistics:\n"
            numeric_stats = basic_stats["numeric_summary"]
            for col in list(numeric_stats.keys())[:5]:  # Top 5 columns
                stats = numeric_stats[col]
                summary += f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}\n"
        
        # Categorical summary
        if "categorical_summary" in basic_stats:
            summary += "\nKey Categorical Data:\n"
            cat_stats = basic_stats["categorical_summary"]
            for col, stats in list(cat_stats.items())[:3]:  # Top 3 columns
                summary += f"- {col}: {stats['unique_count']} unique values\n"
        
        return summary
    
    def _create_insights_prompt(self, data_summary: str, dataset_name: str) -> str:
        """Create prompt for general insights."""
        return f"""
Analyze this governance dataset and provide key insights:

{data_summary}

Provide insights in this format:
1. Key Patterns: [2-3 main patterns you observe]
2. Data Quality: [assessment of data completeness and reliability]  
3. Notable Trends: [significant trends or anomalies]
4. Statistical Insights: [important statistical findings]

Keep response under 200 words and focus on actionable insights for policymakers.
"""
    
    def _create_governance_prompt(self, data_summary: str, dataset_name: str) -> str:
        """Create prompt for governance recommendations."""
        return f"""
Based on this Telangana governance dataset, provide policy recommendations:

{data_summary}

Provide recommendations in this format:
1. Immediate Actions: [urgent policy actions needed]
2. Resource Allocation: [how to better allocate resources]
3. Performance Improvements: [ways to improve governance metrics]
4. Monitoring Priorities: [what metrics to track closely]

Focus on practical, implementable recommendations for government officials.
Keep under 200 words.
"""
    
    def _call_llama_api(self, prompt: str, analysis_type: str) -> str:
        """Call Llama API for analysis."""
        try:
            payload = {
                "model": "mistral",  # or "llama3" depending on your setup
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 150
                }
            }
            
            response = requests.post(
                f"{self.llama_url}/api/generate",
                json=payload,
                timeout=600
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                raise Exception(f"Llama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Llama API call failed: {e}")
            return f"Analysis unavailable: {str(e)}"
    
    def _generate_fallback_insights(self, df: pd.DataFrame, basic_stats: Dict) -> str:
        """Generate basic insights when Llama is unavailable."""
        insights = []
        
        # Data quality insight
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct < 5:
            insights.append("✓ High data quality: <5% missing values")
        elif missing_pct < 20:
            insights.append("⚠ Moderate data quality: some missing values present")
        else:
            insights.append("❌ Low data quality: significant missing values")
        
        # Numeric insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            high_variance_cols = []
            for col in numeric_cols:
                if df[col].std() > 2 * df[col].mean():
                    high_variance_cols.append(col)
            
            if high_variance_cols:
                insights.append(f"📊 High variance detected in: {', '.join(high_variance_cols[:3])}")
        
        # Categorical insights
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() == 1:
                insights.append(f"⚠ {col} has only one unique value")
        
        return "\n".join(insights) if insights else "Basic statistical analysis completed."
    
    def _extract_key_findings(self, basic_stats: Dict, llama_insights: Dict) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # From basic stats
        if "missing_values" in basic_stats and basic_stats["missing_values"]:
            findings.append(f"Missing data found in {len(basic_stats['missing_values'])} columns")
        
        if "correlations" in basic_stats:
            # Find strong correlations
            strong_corrs = 0
            for col1, corrs in basic_stats["correlations"].items():
                for col2, corr_val in corrs.items():
                    if col1 != col2 and abs(corr_val) > 0.7:
                        strong_corrs += 1
            if strong_corrs > 0:
                findings.append(f"Found {strong_corrs//2} strong correlations between variables")
        
        # Add Llama insights summary
        if llama_insights.get("analysis_quality") == "success":
            findings.append("AI-powered insights generated successfully")
        else:
            findings.append("Basic statistical analysis completed")
        
        return findings
    
    def _save_analysis_results(self, results: Dict, dataset_name: str):
        """Save analysis results."""
        # Save JSON results
        results_file = self.analyzed_dir / f"{dataset_name}_analysis.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save readable report
        report_file = self.analyzed_dir / f"{dataset_name}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._format_analysis_report(results))
        
        logger.info(f"Analysis saved: {results_file} and {report_file}")
    
    def _format_analysis_report(self, results: Dict) -> str:
        """Format analysis results as readable report."""
        report = f"RTGS Analysis Report - {results['dataset_name']}\n"
        report += "=" * 60 + "\n"
        report += f"Generated: {results['timestamp']}\n\n"
        
        # Dataset info
        info = results['dataset_info']
        report += f"Dataset Overview:\n"
        report += f"- Total Records: {info['rows']:,}\n"
        report += f"- Total Columns: {info['columns']}\n"
        report += f"- Numeric Columns: {info['numeric_columns']}\n"
        report += f"- Categorical Columns: {info['categorical_columns']}\n\n"
        
        # ASCII charts
        if results['ascii_charts']:
            report += "VISUALIZATIONS:\n"
            for chart_name, chart in results['ascii_charts'].items():
                report += chart + "\n"
        
        # Llama insights
        llama = results['llama_insights']
        report += "\nAI INSIGHTS:\n"
        report += "-" * 40 + "\n"
        report += f"Data Insights:\n{llama['data_insights']}\n\n"
        report += f"Governance Recommendations:\n{llama['governance_recommendations']}\n\n"
        
        # Key findings
        report += "KEY FINDINGS:\n"
        for finding in results['key_findings']:
            report += f"• {finding}\n"
        
        return report
    
    def _display_analysis(self, results: Dict):
        """Display analysis results to console."""
        print(f"\n{'='*60}")
        print(f"RTGS ANALYSIS REPORT - {results['dataset_name'].upper()}")
        print(f"{'='*60}")
        
        # Dataset overview
        info = results['dataset_info']
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Records: {info['rows']:,}")
        print(f"   Columns: {info['columns']} ({info['numeric_columns']} numeric, {info['categorical_columns']} categorical)")
        
        # ASCII Charts
        if results['ascii_charts']:
            print(f"\n📈 VISUALIZATIONS:")
            for chart_name, chart in list(results['ascii_charts'].items())[:2]:  # Show first 2 charts
                print(chart)
        
        # AI Insights
        llama = results['llama_insights']
        print(f"\n🤖 AI INSIGHTS:")
        print(f"   {llama['data_insights']}\n")
        
        print(f"🏛️ GOVERNANCE RECOMMENDATIONS:")
        print(f"   {llama['governance_recommendations']}\n")
        
        # Key findings
        print(f"🔍 KEY FINDINGS:")
        for finding in results['key_findings']:
            print(f"   • {finding}")
        
        print(f"\n💾 Full report saved to: data/analyzed/{results['dataset_name']}_report.txt")
        print(f"{'='*60}\n")

def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <dataset_name>")
        print("\nMake sure Llama is running: ollama serve")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    try:
        # Analyze dataset
        results = analyzer.analyze_dataset(dataset_name)
        
        print(f"✅ Analysis completed successfully!")
        print(f"📁 Results saved in data/analyzed/")
        
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("Make sure you've run transform.py first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Make sure Llama is running: ollama serve")
        sys.exit(1)

if __name__ == "__main__":
    main()