"""
Seasonal analysis utilities.

Provides helper functions to compute seasonal indices from
multi-year time series data and to remove those indices from data.

Functions
---------
create_seasonal_index
    Calculate a seasonal index for monthly or weekly data.
seasonal_difference
    Remove a seasonal index from a series (additive differencing).
"""

from typing import Union

import pandas as pd


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
        ``date`` column that can be parsed as dates. If ``data`` is a
        :class:`~pandas.DataFrame`, the first column will be used.
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
) -> pd.Series:
    """Remove a seasonal index from data by subtraction.

    The function assumes an additive seasonal model and subtracts the
    period-specific seasonal index from the series.

    Parameters
    ----------
    data:
        Series or DataFrame containing the observations to adjust.
    seasonal_index:
        Output from :func:`create_seasonal_index`.
    frequency:
        ``"M"`` for monthly seasonality or ``"W"`` for weekly seasonality.

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
        period_numbers = pd.Series(series.index.isocalendar().week, index=series.index)
    else:
        raise ValueError("frequency must be 'M' or 'W'")

    seasonal_values = period_numbers.map(seasonal_index)
    return series - seasonal_values
