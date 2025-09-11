"""
Seasonal analysis utilities.

Provides helper functions to compute seasonal indices from
multi-year time series data and to remove those indices from data.

Functions
---------
create_seasonal_index
    Calculate a seasonal index for monthly or weekly data.
seasonal_difference
    Remove a seasonal index from a series (additive or multiplicative differencing).
get_seasonal_ratio
    Get the seasonal index ratio for a specific date and marketing year.
"""

from typing import Union, Optional
from datetime import datetime
import warnings

import pandas as pd


# Marketing year utility functions
def _get_marketing_year_start(commodity_type: str) -> int:
    """Get marketing year start month for commodity type."""
    marketing_years = {
        'grains': 9,      # September (wheat, corn, etc.)
        'oilseeds': 9,    # September (soybeans, etc.)
        'livestock': 1    # January (cattle, pork, etc.)
    }
    return marketing_years.get(commodity_type.lower(), 9)


def _get_marketing_year_from_date(date: Union[pd.Timestamp, datetime], 
                                  marketing_year_start: int) -> int:
    """Calculate marketing year for a given date."""
    if isinstance(date, pd.Timestamp):
        date_obj = date.to_pydatetime()
    elif isinstance(date, str):
        date_obj = pd.to_datetime(date).to_pydatetime()
    else:
        date_obj = date
    
    if date_obj.month >= marketing_year_start:
        return date_obj.year
    else:
        return date_obj.year - 1


def _get_my_week_from_date(date: Union[pd.Timestamp, datetime], 
                           marketing_year_start: int) -> int:
    """Calculate week of marketing year for a given date."""
    if isinstance(date, pd.Timestamp):
        date_obj = date.to_pydatetime()
    elif isinstance(date, str):
        date_obj = pd.to_datetime(date).to_pydatetime()
    else:
        date_obj = date
    
    marketing_year = _get_marketing_year_from_date(date_obj, marketing_year_start)
    my_start = datetime(marketing_year, marketing_year_start, 1)
    delta = date_obj - my_start
    return max(1, delta.days // 7 + 1)


def create_seasonal_index(
    data: Union[pd.Series, pd.DataFrame],
    frequency: str = "M",
    scale: float = 1.0,
) -> pd.Series:
    """Compute a seasonal index from multiple years of data.

    Parameters
    ----------
    data:
        Time series values. Must have a :class:`~pandas.DatetimeIndex` or a
        ``date`` columns_col that can be parsed as dates. If ``data`` is a
        :class:`~pandas.DataFrame`, the first columns_col will be used.
    frequency:
        ``"M"`` for monthly seasonality or ``"W"`` for weekly seasonality.
    scale:
        Factor used to scale the resulting index. ``1`` returns ratios while
        ``100`` returns percentage style indices.

    Returns
    -------
    :class:`~pandas.Series`
        Seasonal factors indexed by period number (1-12 for monthly or
        1-52/53 for weekly).
    """
    if isinstance(data, pd.DataFrame):
        if "date" in data.columns:
            data = data.set_index("date")
        data = data.iloc[:, 0]

    series = data.dropna().copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    freq = frequency.upper()
    if freq == "M":
        period_numbers = series.index.month
    elif freq == "W":
        period_numbers = series.index.isocalendar().week
    else:
        raise ValueError("frequency must be 'M' or 'W'")

    grouped = series.groupby(period_numbers)
    period_means = grouped.mean()
    overall_mean = series.mean()

    index = (period_means / overall_mean) * scale
    # Ensure a complete index of periods
    num_periods = int(period_numbers.max())
    index = index.reindex(range(1, num_periods + 1))
    return index


def seasonal_difference(
    data: Union[pd.Series, pd.DataFrame],
    seasonal_index: pd.Series,
    frequency: str = "M",
    method: str = "multiplicative",
    commodity_type: str = "grains",
    marketing_year_start: Optional[int] = None,
) -> pd.Series:
    """Remove a seasonal index from data.

    The function can use either additive (subtraction) or multiplicative (division)
    seasonal adjustment. For percentage-style indices (scale=100), multiplicative
    is typically appropriate.

    Parameters
    ----------
    data:
        Series or DataFrame containing the observations to adjust.
    seasonal_index:
        Output from :func:`create_seasonal_index`.
    frequency:
        ``"M"`` for monthly seasonality or ``"W"`` for weekly seasonality.
    method:
        ``"additive"`` for subtraction or ``"multiplicative"`` for division.
        Use multiplicative when seasonal_index was created with scale=100.
    commodity_type:
        ``"grains"``, ``"oilseeds"``, or ``"livestock"`` for marketing year logic.
        Only used when frequency="W" and marketing year weeks are needed.
    marketing_year_start:
        Override the marketing year start month. If None, uses commodity_type default.

    Returns
    -------
    :class:`~pandas.Series`
        Deseasonalized series with the same index as ``data``.
    """
    if isinstance(data, pd.DataFrame):
        if "date" in data.columns:
            data = data.set_index("date")
        data = data.iloc[:, 0]

    series = data.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    freq = frequency.upper()
    if freq == "M":
        period_numbers = pd.Series(series.index.month, index=series.index)
    elif freq == "W":
        # Determine if we should use marketing year weeks or calendar weeks
        if marketing_year_start is None:
            marketing_year_start = _get_marketing_year_start(commodity_type)
        
        # For marketing year logic, calculate week within marketing year
        if marketing_year_start != 1:  # Not January start, use marketing year weeks
            period_numbers = pd.Series([
                _get_my_week_from_date(date, marketing_year_start) 
                for date in series.index
            ], index=series.index)
        else:
            # Use calendar weeks for January start (livestock)
            period_numbers = pd.Series(series.index.isocalendar().week, index=series.index)
    else:
        raise ValueError("frequency must be 'M' or 'W'")

    # Map periods to seasonal index values
    seasonal_values = period_numbers.map(seasonal_index)
    
    # Handle missing values in seasonal index
    seasonal_values = seasonal_values.fillna(seasonal_index.mean())
    
    # Apply seasonal adjustment based on method
    method_lower = method.lower()
    if method_lower == "additive":
        return series - seasonal_values
    elif method_lower == "multiplicative":
        # For percentage-style indices (scale=100), divide by (index/100)
        # This assumes seasonal_index values are like 110 for 10% above average
        # Check if values suggest percentage style (most values between 50-200)
        if seasonal_index.median() > 10:  # Likely percentage style
            seasonal_factors = seasonal_values / 100.0
        else:
            seasonal_factors = seasonal_values
        
        # Avoid division by zero or very small numbers
        seasonal_factors = seasonal_factors.replace(0, 1.0)
        seasonal_factors = seasonal_factors.fillna(1.0)
        
        # Ensure factors are reasonable (between 0.1 and 10)
        seasonal_factors = seasonal_factors.clip(0.1, 10.0)
        
        return series / seasonal_factors
    else:
        raise ValueError("method must be 'additive' or 'multiplicative'")


def get_seasonal_ratio(
    date: Union[pd.Timestamp, datetime, str],
    seasonal_index: pd.Series, 
    frequency: str = "M",
    commodity_type: str = "grains",
    marketing_year_start: Optional[int] = None
) -> float:
    """Get the seasonal index ratio for a specific date and marketing year.
    
    This function finds the appropriate seasonal adjustment factor for a given
    date based on the seasonal index and marketing year logic.
    
    Parameters
    ----------
    date:
        The date to get the seasonal ratio for.
    seasonal_index:
        Output from :func:`create_seasonal_index`.
    frequency:
        ``"M"`` for monthly seasonality or ``"W"`` for weekly seasonality.
    commodity_type:
        ``"grains"``, ``"oilseeds"``, or ``"livestock"`` for marketing year logic.
    marketing_year_start:
        Override the marketing year start month. If None, uses commodity_type default.
    
    Returns
    -------
    float
        The seasonal ratio for the given date. For percentage-style indices,
        this will be a value like 1.10 for 10% above seasonal average.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif isinstance(date, datetime):
        date = pd.Timestamp(date)
    
    if marketing_year_start is None:
        marketing_year_start = _get_marketing_year_start(commodity_type)
    
    freq = frequency.upper()
    if freq == "M":
        period_number = date.month
    elif freq == "W":
        if marketing_year_start != 1:  # Use marketing year weeks
            period_number = _get_my_week_from_date(date, marketing_year_start)
        else:
            period_number = date.isocalendar().week
    else:
        raise ValueError("frequency must be 'M' or 'W'")
    
    # Get the seasonal value for this period
    seasonal_value = seasonal_index.get(period_number)
    
    if seasonal_value is None:
        # If period not found, use the average
        seasonal_value = seasonal_index.mean()
    
    # Convert to ratio format
    if seasonal_index.median() > 10:  # Likely percentage style
        return seasonal_value / 100.0
    else:
        return seasonal_value
