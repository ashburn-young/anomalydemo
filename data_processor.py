"""
Data Processing Module for Retail Anomaly Detection
Handles data ingestion, preprocessing, and statistical anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailDataProcessor:
    """Process and analyze retail data for anomalies"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        
    def load_and_validate_data(self, uploaded_file) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and validate uploaded retail data file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (DataFrame, list of validation messages)
        """
        validation_messages = []
        
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {uploaded_file.name}")
            
            logger.info(f"Successfully loaded file: {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns")
            
            # Basic validation
            validation_messages.extend(self._validate_retail_data(df))
            
            return df, validation_messages
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg)
            validation_messages.append(error_msg)
            return pd.DataFrame(), validation_messages
    
    def _validate_retail_data(self, df: pd.DataFrame) -> List[str]:
        """Validate retail data structure and content"""
        messages = []
        
        # Check for minimum required columns (flexible approach)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(numeric_cols) == 0:
            messages.append("Warning: No numeric columns found for analysis")
        
        # Check for missing values
        missing_percentage = (df.isnull().sum() / len(df) * 100)
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        
        if high_missing_cols:
            messages.append(f"Warning: High missing values (>50%) in columns: {high_missing_cols}")
        
        # Check data size
        if len(df) < 10:
            messages.append("Warning: Dataset is very small (< 10 rows), analysis may be limited")
        elif len(df) > 100000:
            messages.append("Info: Large dataset detected, processing may take longer")
        
        messages.append(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        return messages
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess retail data for anomaly detection
        
        Args:
            df: Raw retail data DataFrame
            
        Returns:
            Tuple of (processed DataFrame, preprocessing summary)
        """
        try:
            processed_df = df.copy()
            summary = {
                'original_shape': df.shape,
                'columns_processed': [],
                'transformations_applied': []
            }
            
            # Handle missing values
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            categorical_columns = processed_df.select_dtypes(include=['object']).columns
            
            # Fill missing values
            for col in numeric_columns:
                if processed_df[col].isnull().any():
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
                    summary['transformations_applied'].append(f"Filled missing values in {col} with median")
            
            for col in categorical_columns:
                if processed_df[col].isnull().any():
                    processed_df[col].fillna('Unknown', inplace=True)
                    summary['transformations_applied'].append(f"Filled missing values in {col} with 'Unknown'")
            
            # Convert date columns
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    # Try to convert to datetime
                    try:
                        processed_df[col] = pd.to_datetime(processed_df[col], errors='ignore')
                        if processed_df[col].dtype.name.startswith('datetime'):
                            summary['transformations_applied'].append(f"Converted {col} to datetime")
                    except:
                        pass
            
            # Create derived features for retail analysis
            processed_df = self._create_derived_features(processed_df, summary)
            
            summary['final_shape'] = processed_df.shape
            summary['columns_processed'] = list(processed_df.columns)
            
            logger.info(f"Data preprocessing completed: {summary['original_shape']} -> {summary['final_shape']}")
            
            return processed_df, summary
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            return df, {'error': str(e)}
    
    def _create_derived_features(self, df: pd.DataFrame, summary: Dict[str, Any]) -> pd.DataFrame:
        """Create derived features specific to retail analysis"""
        try:
            # Look for common retail column patterns
            amount_cols = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['amount', 'price', 'cost', 'value', 'total', 'sales'])]
            
            quantity_cols = [col for col in df.columns if any(keyword in col.lower() 
                            for keyword in ['quantity', 'qty', 'count', 'units'])]
            
            date_cols = df.select_dtypes(include=['datetime64']).columns
            
            # Create transaction value features if we have amount and quantity
            if amount_cols and quantity_cols:
                for amount_col in amount_cols[:1]:  # Use first amount column
                    for qty_col in quantity_cols[:1]:  # Use first quantity column
                        if df[amount_col].dtype in [np.float64, np.int64] and df[qty_col].dtype in [np.float64, np.int64]:
                            df['unit_price'] = df[amount_col] / (df[qty_col] + 1e-8)  # Avoid division by zero
                            summary['transformations_applied'].append("Created unit_price feature")
                            break
            
            # Create time-based features from date columns
            for date_col in date_cols:
                if df[date_col].dtype.name.startswith('datetime'):
                    df[f'{date_col}_hour'] = df[date_col].dt.hour
                    df[f'{date_col}_day_of_week'] = df[date_col].dt.dayofweek
                    df[f'{date_col}_month'] = df[date_col].dt.month
                    summary['transformations_applied'].append(f"Created time features from {date_col}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Could not create derived features: {str(e)}")
            return df
    
    def detect_statistical_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            results = {
                'statistical_anomalies': [],
                'methods_used': [],
                'summary_stats': {}
            }
            
            # Get numeric columns for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                return {
                    'statistical_anomalies': [],
                    'methods_used': ['No numeric columns available'],
                    'summary_stats': {}
                }
            
            # Don't drop rows, just replace infinite values with NaN for calculations
            df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # For each column, calculate statistics on non-null values only
            total_anomalies_found = 0
            
            # Z-score based anomalies (lowered threshold for more sensitivity)
            z_threshold = 2.5  # More sensitive than 3
            z_anomaly_count = 0
            
            for col in numeric_cols:
                if df_clean[col].notna().sum() > 1:  # Need at least 2 values
                    col_data = df_clean[col].dropna()
                    if col_data.std() > 0:  # Avoid division by zero
                        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                        col_z_anomalies = z_scores > z_threshold
                        z_anomaly_count += col_z_anomalies.sum()
            
            if z_anomaly_count > 0:
                results['statistical_anomalies'].append({
                    'method': 'Z-Score',
                    'anomaly_count': int(z_anomaly_count),
                    'percentage': float(z_anomaly_count / len(df) * 100),
                    'description': f'Data points with Z-score > {z_threshold} (more than {z_threshold} standard deviations from mean)'
                })
                results['methods_used'].append('Z-Score Analysis')
                total_anomalies_found += z_anomaly_count
            
            # IQR based anomalies
            iqr_anomaly_count = 0
            
            for col in numeric_cols:
                if df_clean[col].notna().sum() > 4:  # Need at least 5 values for quartiles
                    col_data = df_clean[col].dropna()
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:  # Valid IQR
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        col_anomalies = (col_data < lower_bound) | (col_data > upper_bound)
                        iqr_anomaly_count += col_anomalies.sum()
            
            if iqr_anomaly_count > 0:
                results['statistical_anomalies'].append({
                    'method': 'IQR',
                    'anomaly_count': int(iqr_anomaly_count),
                    'percentage': float(iqr_anomaly_count / len(df) * 100),
                    'description': 'Data points outside 1.5 * IQR from quartiles (traditional outlier detection)'
                })
                results['methods_used'].append('IQR Analysis')
                total_anomalies_found += iqr_anomaly_count
            
            # Isolation Forest with better error handling
            if len(df_clean.dropna()) >= 5:  # Need minimum samples
                try:
                    # Use only complete cases for isolation forest
                    df_complete = df_clean.dropna()
                    if len(df_complete) >= 5:
                        df_scaled = self.scaler.fit_transform(df_complete)
                        
                        # Use higher contamination rate for more sensitivity
                        contamination_rate = min(0.1, max(0.01, len(df_complete) * 0.05 / len(df_complete)))
                        self.isolation_forest.set_params(contamination=contamination_rate)
                        
                        isolation_scores = self.isolation_forest.fit_predict(df_scaled)
                        isolation_anomalies = isolation_scores == -1
                        isolation_count = isolation_anomalies.sum()
                        
                        if isolation_count > 0:
                            results['statistical_anomalies'].append({
                                'method': 'Isolation Forest',
                                'anomaly_count': int(isolation_count),
                                'percentage': float(isolation_count / len(df) * 100),
                                'description': 'Machine learning-based anomaly detection using isolation forest algorithm'
                            })
                            results['methods_used'].append('Isolation Forest')
                            total_anomalies_found += isolation_count
                except Exception as e:
                    logger.warning(f"Isolation Forest failed: {str(e)}")
            
            # Add modified Z-score method for additional sensitivity
            modified_z_anomaly_count = 0
            modified_z_threshold = 3.5
            
            for col in numeric_cols:
                if df_clean[col].notna().sum() > 1:
                    col_data = df_clean[col].dropna()
                    if len(col_data) > 0:
                        median = col_data.median()
                        mad = np.median(np.abs(col_data - median))  # Median Absolute Deviation
                        
                        if mad > 0:  # Avoid division by zero
                            modified_z_scores = 0.6745 * (col_data - median) / mad
                            col_modified_z_anomalies = np.abs(modified_z_scores) > modified_z_threshold
                            modified_z_anomaly_count += col_modified_z_anomalies.sum()
            
            if modified_z_anomaly_count > 0:
                results['statistical_anomalies'].append({
                    'method': 'Modified Z-Score',
                    'anomaly_count': int(modified_z_anomaly_count),
                    'percentage': float(modified_z_anomaly_count / len(df) * 100),
                    'description': 'Robust anomaly detection using Modified Z-Score with Median Absolute Deviation'
                })
                results['methods_used'].append('Modified Z-Score Analysis')
                total_anomalies_found += modified_z_anomaly_count
            
            # Generate summary statistics
            results['summary_stats'] = {
                'total_records': len(df),
                'numeric_columns': len(numeric_cols),
                'columns_analyzed': numeric_cols,
                'total_anomalies_found': total_anomalies_found,
                'overall_anomaly_rate': f"{(total_anomalies_found / len(df) * 100):.2f}%"
            }
            
            logger.info(f"Statistical anomaly detection completed using {len(results['methods_used'])} methods")
            logger.info(f"Found {total_anomalies_found} total anomalies across all methods")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {str(e)}")
            return {
                'statistical_anomalies': [],
                'methods_used': [],
                'summary_stats': {'error': str(e)}
            }
    
    def get_data_summary(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive data summary for AI analysis"""
        try:
            summary_parts = []
            
            # Basic info
            summary_parts.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            summary_parts.append(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}")
            summary_parts.append(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}")
            summary_parts.append(f"Date columns ({len(date_cols)}): {date_cols}")
            
            # Summary statistics for numeric columns
            if numeric_cols:
                summary_parts.append("\nNumeric Column Statistics:")
                for col in numeric_cols[:3]:  # Limit to first 3 columns
                    stats = df[col].describe()
                    summary_parts.append(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
            
            # Missing values
            missing_info = df.isnull().sum()
            missing_cols = missing_info[missing_info > 0]
            if len(missing_cols) > 0:
                summary_parts.append(f"\nMissing values: {dict(missing_cols)}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return f"Error generating summary: {str(e)}"