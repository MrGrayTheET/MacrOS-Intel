# ===== /pages/esr/esr_utils.py =====
"""
ESR utility functions and data handling
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


def get_sample_esr_data(commodity: str, year: int, countries: list) -> pd.DataFrame:
    """Generate sample ESR data for demonstration."""
    if not countries:
        countries = ['Korea, South', 'Japan', 'China']

    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=26)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')

    data_rows = []
    for country in countries:
        for date in dates:
            row = {
                'weekEndingDate': date,
                'country': country,
                'commodity': commodity,
                'marketing_year': year,
                'weeklyExports': np.random.randint(1000, 8000),
                'outstandingSales': np.random.randint(20000, 50000),
                'grossNewSales': np.random.randint(500, 15000),
                'currentMYNetSales': np.random.randint(5000, 25000),
                'currentMYTotalCommitment': np.random.randint(25000, 75000),
                'nextMYOutstandingSales': np.random.randint(0, 10000),
                'nextMYNetSales': np.random.randint(0, 5000)
            }
            data_rows.append(row)

    return pd.DataFrame(data_rows)


def get_multi_year_esr_data(commodity: str, country: str) -> pd.DataFrame:
    """Get multi-year sample data for country analysis."""
    current_year = pd.Timestamp.now().year
    years = list(range(current_year - 4, current_year + 1))

    all_data = []
    for year in years:
        year_data = get_sample_esr_data(commodity, year, [country])
        all_data.append(year_data)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def create_empty_figure(title: str) -> go.Figure:
    """Create empty figure with title."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(title=title, height=400)
    return fig
