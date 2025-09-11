"""
Time Series Analysis Module
============================

Comprehensive time series analysis functions for:
- Seasonal normalization and adjustment
- Stationarity testing and transformation  
- Multi-series analysis by grouping variables
- Statistical decomposition and trend analysis

Dependencies: pandas, numpy, scipy, statsmodels, scikit-learn
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

# Core dependencies
try:
    from scipy import stats
    from scipy.stats import jarque_bera, normaltest
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - some statistical tests will be disabled")

# Statsmodels for advanced time series
try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.filters.hp_filter import hpfilter
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available - advanced time series analysis will be disabled")

# Scikit-learn for preprocessing
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available - some preprocessing functions will be disabled")


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis toolkit for commodity and economic data.
    
    Features:
    - Seasonal normalization and decomposition
    - Stationarity testing and transformations
    - Multi-series analysis by grouping variables
    - Statistical tests and diagnostics
    - Trend extraction and filtering
    """
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'date', 
                 value_columns: Optional[List[str]] = None):
        """
        Initialize the TimeSeriesAnalyzer.
        
        Args:
            data: DataFrame with time series data
            date_column: Name of date columns_col (or index if None)
            value_columns: List of numeric columns to analyze (default: all numeric)
        """
        self.data = data.copy()
        self.date_column = date_column
        
        # Set up date index
        if date_column and date_column in self.data.columns:
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            if not self.data.index.equals(self.data[date_column]):
                self.data.set_index(date_column, inplace=True)
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Identify value columns
        if value_columns is None:
            self.value_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.value_columns = value_columns
            
        # Store original data
        self.original_data = self.data.copy()
        
    def seasonal_normalize(self, column: str, method: str = 'decompose', 
                          seasonal_periods: Optional[int] = None,
                          group_by: Optional[str] = None) -> pd.Series:
        """
        Remove seasonal component from time series.
        
        Args:
            column: Column to normalize
            method: 'decompose', 'stl', 'detrend', or 'zscore'
            seasonal_periods: Number of periods per season (auto-detect if None)
            group_by: Column to group by for multi-series normalization
            
        Returns:
            Series with seasonal component removed
        """
        if not HAS_STATSMODELS and method in ['decompose', 'stl']:
            warnings.warn(f"Method {method} requires statsmodels. Using simple detrend.")
            method = 'detrend'
            
        if group_by:
            return self._group_seasonal_normalize(column, method, seasonal_periods, group_by)
        
        data = self.data[column].dropna()
        
        if seasonal_periods is None:
            seasonal_periods = self._detect_seasonality(data)
            
        if method == 'decompose' and HAS_STATSMODELS:
            decomposition = seasonal_decompose(data, model='additive', 
                                             period=seasonal_periods, extrapolate_trend='freq')
            normalized = data - decomposition.seasonal
            
        elif method == 'stl' and HAS_STATSMODELS:
            stl = STL(data, seasonal=seasonal_periods)
            decomposition = stl.fit()
            normalized = data - decomposition.seasonal
            
        elif method == 'detrend':
            # Simple seasonal detrending using moving average
            seasonal_ma = data.rolling(window=seasonal_periods, center=True).mean()
            normalized = data - seasonal_ma
            
        elif method == 'zscore':
            # Z-score normalization by season
            normalized = self._seasonal_zscore(data, seasonal_periods)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized.fillna(method='ffill')
    
    def _group_seasonal_normalize(self, column: str, method: str, 
                                 seasonal_periods: Optional[int], group_by: str) -> pd.Series:
        """Apply seasonal normalization grouped by a categorical variable."""
        results = []
        
        for group_name, group_data in self.data.groupby(group_by):
            if len(group_data) < 2 * (seasonal_periods or 12):
                # Skip groups with insufficient data
                normalized_group = group_data[column].fillna(0)
            else:
                analyzer = TimeSeriesAnalyzer(group_data.reset_index(), 
                                            date_column=group_data.index.name or 'date')
                normalized_group = analyzer.seasonal_normalize(column, method, seasonal_periods)
                normalized_group.index = group_data.index
            
            results.append(normalized_group)
        
        return pd.concat(results).sort_index()
    
    def _detect_seasonality(self, data: pd.Series) -> int:
        """Auto-detect seasonal periods in the data."""
        freq = pd.infer_freq(data.index)
        
        if freq:
            if 'D' in freq:
                return 365  # Daily data - yearly seasonality
            elif 'W' in freq:
                return 52   # Weekly data - yearly seasonality
            elif 'M' in freq:
                return 12   # Monthly data - yearly seasonality
            elif 'Q' in freq:
                return 4    # Quarterly data - yearly seasonality
        
        # Default fallback
        return min(52, len(data) // 4)
    
    def _seasonal_zscore(self, data: pd.Series, periods: int) -> pd.Series:
        """Z-score normalization within seasonal periods."""
        data_copy = data.copy()
        seasonal_idx = np.arange(len(data)) % periods
        
        for i in range(periods):
            mask = seasonal_idx == i
            seasonal_data = data_copy[mask]
            if len(seasonal_data) > 1:
                mean_val = seasonal_data.mean()
                std_val = seasonal_data.std()
                if std_val > 0:
                    data_copy[mask] = (seasonal_data - mean_val) / std_val
                    
        return data_copy
    
    def test_stationarity(self, column: str, group_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Test for stationarity using multiple methods.
        
        Args:
            column: Column to test
            group_by: Column to group by for multi-series testing
            
        Returns:
            Dictionary with test results
        """
        if group_by:
            return self._group_stationarity_test(column, group_by)
        
        data = self.data[column].dropna()
        results = {'series_length': len(data)}
        
        # Augmented Dickey-Fuller test
        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(data, autolag='AIC')
                results['adf'] = {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                }
            except Exception as e:
                results['adf'] = {'error': str(e)}
            
            # KPSS test
            try:
                kpss_result = kpss(data, regression='ct')
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05
                }
            except Exception as e:
                results['kpss'] = {'error': str(e)}
        
        # Simple variance test
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        if len(first_half) > 10 and len(second_half) > 10:
            var_ratio = np.var(second_half) / np.var(first_half)
            results['variance_test'] = {
                'ratio': var_ratio,
                'is_stable': 0.5 < var_ratio < 2.0
            }
        
        return results
    
    def _group_stationarity_test(self, column: str, group_by: str) -> Dict[str, Any]:
        """Apply stationarity tests grouped by a categorical variable."""
        results = {}
        
        for group_name, group_data in self.data.groupby(group_by):
            if len(group_data) > 20:  # Minimum data for meaningful test
                analyzer = TimeSeriesAnalyzer(group_data.reset_index(), 
                                            date_column=group_data.index.name or 'date')
                results[str(group_name)] = analyzer.test_stationarity(column)
            else:
                results[str(group_name)] = {'error': 'insufficient_data'}
        
        return results
    
    def make_stationary(self, column: str, method: str = 'diff',
                       group_by: Optional[str] = None) -> pd.Series:
        """
        Transform series to make it stationary.
        
        Args:
            column: Column to transform
            method: 'diff', 'log_diff', 'hp_filter', 'detrend', or 'standardize'
            group_by: Column to group by for multi-series transformation
            
        Returns:
            Transformed stationary series
        """
        if group_by:
            return self._group_make_stationary(column, method, group_by)
        
        data = self.data[column].dropna()
        
        if method == 'diff':
            result = data.diff()
            
        elif method == 'log_diff':
            # Ensure positive values for log transform
            if (data <= 0).any():
                data = data - data.min() + 1
            result = np.log(data).diff()
            
        elif method == 'hp_filter' and HAS_STATSMODELS:
            try:
                cycle, trend = hpfilter(data, lamb=1600)  # Standard lambda for quarterly data
                result = cycle
            except Exception:
                result = data.diff()  # Fallback to differencing
                
        elif method == 'detrend':
            # Linear detrending
            x = np.arange(len(data))
            if HAS_STATSMODELS:
                model = OLS(data, add_constant(x)).fit()
                result = data - model.predict()
            else:
                # Simple detrend using numpy
                trend = np.polyfit(x, data, 1)
                result = data - (trend[0] * x + trend[1])
                
        elif method == 'standardize':
            result = (data - data.mean()) / data.std()
            
        else:
            raise ValueError(f"Unknown stationarity method: {method}")
        
        return result
    
    def _group_make_stationary(self, column: str, method: str, group_by: str) -> pd.Series:
        """Apply stationarity transformation grouped by a categorical variable."""
        results = []
        
        for group_name, group_data in self.data.groupby(group_by):
            if len(group_data) > 10:
                analyzer = TimeSeriesAnalyzer(group_data.reset_index(), 
                                            date_column=group_data.index.name or 'date')
                transformed_group = analyzer.make_stationary(column, method)
                transformed_group.index = group_data.index
            else:
                # For small groups, just return original data
                transformed_group = group_data[column]
            
            results.append(transformed_group)
        
        return pd.concat(results).sort_index()
    
    def decompose_series(self, column: str, method: str = 'stl',
                        seasonal_periods: Optional[int] = None,
                        group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            column: Column to decompose
            method: 'stl', 'classical', or 'x13' (if available)
            seasonal_periods: Seasonal periods (auto-detect if None)
            group_by: Column to group by for multi-series decomposition
            
        Returns:
            DataFrame with trend, seasonal, residual components
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for time series decomposition")
        
        if group_by:
            return self._group_decompose(column, method, seasonal_periods, group_by)
        
        data = self.data[column].dropna()
        
        if seasonal_periods is None:
            seasonal_periods = self._detect_seasonality(data)
        
        if method == 'stl':
            stl = STL(data, seasonal=seasonal_periods)
            decomposition = stl.fit()
            
        elif method == 'classical':
            decomposition = seasonal_decompose(data, model='additive', 
                                             period=seasonal_periods, extrapolate_trend='freq')
        else:
            raise ValueError(f"Unknown decomposition method: {method}")
        
        result = pd.DataFrame({
            f'{column}_trend': decomposition.trend,
            f'{column}_seasonal': decomposition.seasonal,
            f'{column}_residual': decomposition.resid
        }, index=data.index)
        
        return result
    
    def _group_decompose(self, column: str, method: str, seasonal_periods: Optional[int], 
                        group_by: str) -> pd.DataFrame:
        """Apply decomposition grouped by a categorical variable."""
        results = []
        
        for group_name, group_data in self.data.groupby(group_by):
            if len(group_data) >= 2 * (seasonal_periods or 12):
                analyzer = TimeSeriesAnalyzer(group_data.reset_index(), 
                                            date_column=group_data.index.name or 'date')
                decomp = analyzer.decompose_series(column, method, seasonal_periods)
                decomp.index = group_data.index
                decomp[group_by] = group_name
            else:
                # For insufficient data, create empty decomposition
                decomp = pd.DataFrame({
                    f'{column}_trend': np.nan,
                    f'{column}_seasonal': np.nan, 
                    f'{column}_residual': np.nan,
                    group_by: group_name
                }, index=group_data.index)
            
            results.append(decomp)
        
        return pd.concat(results).sort_index()
    
    def create_grouped_timeseries(self, value_column: str, group_column: str,
                                 pivot: bool = True, agg_func: str = 'mean') -> pd.DataFrame:
        """
        Create time series organized by grouping variable.
        
        Args:
            value_column: Column containing values to track over time
            group_column: Column to group by (e.g., 'country', 'commodity')
            pivot: Whether to pivot groups into columns
            agg_func: Aggregation function for duplicate dates ('mean', 'sum', 'last')
            
        Returns:
            DataFrame with time series for each group
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Group columns_col '{group_column}' not found in data")
        
        if value_column not in self.data.columns:
            raise ValueError(f"Value columns_col '{value_column}' not found in data")
        
        # Aggregate by date and group
        grouped_data = (self.data.groupby([self.data.index, group_column])[value_column]
                       .agg(agg_func)
                       .reset_index())
        
        if pivot:
            # Pivot to have groups as columns
            result = grouped_data.pivot(index=grouped_data.columns[0], 
                                      columns=group_column, 
                                      values=value_column)
            result.columns.name = None
        else:
            # Keep in long format
            result = grouped_data.set_index(grouped_data.columns[0])
        
        return result
    
    def multi_series_analysis(self, value_columns: List[str], group_by: str,
                             normalize: bool = True, correlation_analysis: bool = True) -> Dict[str, Any]:
        """
        Analyze multiple time series grouped by a categorical variable.
        
        Args:
            value_columns: List of numeric columns to analyze
            group_by: Column to group by
            normalize: Whether to normalize series for comparison
            correlation_analysis: Whether to compute cross-correlations
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {
            'group_statistics': {},
            'series_data': {},
            'correlations': {},
            'stationarity_tests': {}
        }
        
        # Analyze each group
        for group_name, group_data in self.data.groupby(group_by):
            group_name = str(group_name)
            
            if len(group_data) < 10:
                results['group_statistics'][group_name] = {'error': 'insufficient_data'}
                continue
            
            analyzer = TimeSeriesAnalyzer(group_data.reset_index(), 
                                        date_column=group_data.index.name or 'date',
                                        value_columns=value_columns)
            
            # Basic statistics
            group_stats = {}
            series_data = {}
            
            for col in value_columns:
                if col in group_data.columns:
                    data = group_data[col].dropna()
                    
                    if len(data) > 0:
                        # Store series data (normalized if requested)
                        if normalize and len(data) > 1:
                            normalized = (data - data.mean()) / data.std()
                            series_data[f'{col}_normalized'] = normalized
                        series_data[col] = data
                        
                        # Basic statistics
                        group_stats[col] = {
                            'count': len(data),
                            'mean': data.mean(),
                            'std': data.std(),
                            'min': data.min(),
                            'max': data.max(),
                            'trend': self._calculate_trend(data),
                            'volatility': data.std() / data.mean() if data.mean() != 0 else np.inf
                        }
                        
                        # Stationarity test
                        if len(data) > 20:
                            stationarity = analyzer.test_stationarity(col)
                            results['stationarity_tests'][f'{group_name}_{col}'] = stationarity
            
            results['group_statistics'][group_name] = group_stats
            results['series_data'][group_name] = series_data
            
            # Correlation analysis within group
            if correlation_analysis and len(value_columns) > 1:
                numeric_data = group_data[value_columns].select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    results['correlations'][group_name] = corr_matrix.to_dict()
        
        # Cross-group correlation analysis
        if correlation_analysis:
            for col in value_columns:
                pivot_data = self.create_grouped_timeseries(col, group_by)
                if len(pivot_data.columns) > 1:
                    cross_corr = pivot_data.corr()
                    results['correlations'][f'cross_group_{col}'] = cross_corr.to_dict()
        
        return results
    
    def _calculate_trend(self, data: pd.Series) -> float:
        """Calculate simple linear trend coefficient."""
        if len(data) < 3:
            return 0.0
        
        x = np.arange(len(data))
        try:
            trend_coef = np.polyfit(x, data.values, 1)[0]
            return float(trend_coef)
        except:
            return 0.0
    
    def detect_outliers(self, column: str, method: str = 'iqr',
                       group_by: Optional[str] = None) -> pd.Series:
        """
        Detect outliers in time series data.
        
        Args:
            column: Column to analyze for outliers
            method: 'iqr', 'zscore', or 'modified_zscore'
            group_by: Column to group by for group-specific outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if group_by:
            return self._group_detect_outliers(column, method, group_by)
        
        data = self.data[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > 3
            
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > 3.5
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Align with original data index
        result = pd.Series(False, index=self.data.index)
        result[data.index] = outliers
        
        return result
    
    def _group_detect_outliers(self, column: str, method: str, group_by: str) -> pd.Series:
        """Detect outliers grouped by a categorical variable."""
        results = []
        
        for group_name, group_data in self.data.groupby(group_by):
            if len(group_data) > 10:
                analyzer = TimeSeriesAnalyzer(group_data.reset_index(), 
                                            date_column=group_data.index.name or 'date')
                group_outliers = analyzer.detect_outliers(column, method)
                group_outliers.index = group_data.index
            else:
                # For small groups, mark no outliers
                group_outliers = pd.Series(False, index=group_data.index)
            
            results.append(group_outliers)
        
        return pd.concat(results).sort_index()
    
    def summary_statistics(self, columns: Optional[List[str]] = None,
                          group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.
        
        Args:
            columns: Columns to summarize (default: all numeric columns)
            group_by: Column to group by for grouped statistics
            
        Returns:
            DataFrame with summary statistics
        """
        if columns is None:
            columns = self.value_columns
        
        if group_by:
            return self._group_summary_statistics(columns, group_by)
        
        results = []
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            data = self.data[col].dropna()
            
            if len(data) == 0:
                continue
            
            stats = {
                'columns_col': col,
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'cv': data.std() / data.mean() if data.mean() != 0 else np.inf,
                'trend': self._calculate_trend(data)
            }
            
            # Add normality tests if scipy available
            if HAS_SCIPY and len(data) > 8:
                try:
                    jb_stat, jb_p = jarque_bera(data)
                    stats['jarque_bera_p'] = jb_p
                    stats['is_normal'] = jb_p > 0.05
                except:
                    stats['jarque_bera_p'] = np.nan
                    stats['is_normal'] = False
            
            results.append(stats)
        
        return pd.DataFrame(results)
    
    def _group_summary_statistics(self, columns: List[str], group_by: str) -> pd.DataFrame:
        """Generate summary statistics grouped by a categorical variable."""
        results = []
        
        for group_name, group_data in self.data.groupby(group_by):
            analyzer = TimeSeriesAnalyzer(group_data.reset_index(), 
                                        date_column=group_data.index.name or 'date',
                                        value_columns=columns)
            
            group_stats = analyzer.summary_statistics(columns)
            group_stats[group_by] = str(group_name)
            results.append(group_stats)
        
        return pd.concat(results, ignore_index=True)


def example_usage():
    """Example usage of the TimeSeriesAnalyzer."""
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='W')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': dates,
        'commodity': np.random.choice(['wheat', 'corn', 'soybeans'], len(dates)),
        'country': np.random.choice(['USA', 'Brazil', 'Argentina'], len(dates)),
        'price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1) + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 52),
        'volume': 1000 + np.random.randn(len(dates)) * 100
    })
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(data, date_column='date', value_columns=['price', 'volume'])
    
    # Seasonal normalization
    normalized_price = analyzer.seasonal_normalize('price', method='stl')
    print("Seasonal normalization completed")
    
    # Stationarity test
    stationarity_results = analyzer.test_stationarity('price')
    print(f"Stationarity test results: {stationarity_results}")
    
    # Multi-series analysis by commodity
    multi_analysis = analyzer.multi_series_analysis(['price', 'volume'], 'commodity')
    print(f"Multi-series analysis completed for {len(multi_analysis['group_statistics'])} groups")
    
    # Outlier detection
    outliers = analyzer.detect_outliers('price', method='iqr')
    print(f"Found {outliers.sum()} outliers")
    
    return analyzer, data


if __name__ == "__main__":
    # Run example
    analyzer, sample_data = example_usage()
    print("TimeSeriesAnalyzer example completed successfully!")