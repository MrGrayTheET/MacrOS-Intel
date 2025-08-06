from datetime import datetime, timedelta

import pandas as pd
import re
import  numpy as np
from plotly import graph_objects as go

futures_pattern = re.compile(r'^[A-Za-z]{2}_\d+$')
def walk_dict(d, parent_key=()):
    """
    Recursively walk through nested dictionary and yield (key_path, value)
    where key_path is a tuple of keys leading to the value.
    """
    for k, v in d.items():
        full_key = parent_key + (k,)
        if isinstance(v, dict):
            yield from walk_dict(v, full_key)
        else:
            yield full_key, v

def key_to_name(table_key):
    key_parts = table_key.rsplit('/')
    name = f"{key_parts[-1]}_{key_parts[-2]}"
    return name

def generate_layout_keys(table_client, categories=[]):
    layout_menu_dict = {}

    for cat in categories:
        cat_dict = table_client.mapping[cat]
        value_dict = {}
        for k in cat_dict.keys():
            value_dict[k] = f'{cat}/{k}'
        layout_menu_dict.update({f'{cat}':value_dict})

    return layout_menu_dict

def get_returns(df:pd.DataFrame, close_col='Close', freq='1W'):
    close_data = df.resample(freq).apply({close_col:'last'})
    df['1wk_return'] = close_data.pct_change(-1)
    df['1mo_return'] = close_data.pct_change(-4)
    df['3mo_return'] = close_data.pct_change(-13)
    return df

def fetch_contract_keys(table_client):
    """Fetch contract keys matching pattern: XX_N (e.g., NG_1, CL_3)"""
    try:
        price_keys = table_client.mapping.get('prices', [])
        pattern = re.compile(r'^[A-Za-z]{2}_\d+$')

        # Filter and sort matching keys
        contract_keys = []
        for key in price_keys:
            key_name = key.split('/')[-1] if '/' in key else key
            if pattern.match(key_name):
                contract_keys.append(key)

        # Sort by commodity code and number
        contract_keys.sort(key=lambda k: (
            re.match(r'^.*?([A-Za-z]{2})_(\d+)$', k).groups()
            if re.match(r'^.*?([A-Za-z]{2})_(\d+)$', k) else (k, 0)
        ))
        return [f'prices/{k}' for k in contract_keys[:4]]  # Return first 4 contracts

    except Exception as e:
        print(f"Error fetching contract keys: {e}")
        return []
def load_contract_keys(table_client):
    key_names = fetch_contract_keys(table_client)
    ct_df = table_client.get_keys([f'prices/{k}' for k in key_names], use_simple_name=True)
    return ct_df

def calc_contract_spreads(df, second_month=False):
    contract_cols = sorted([k for k in df.columns if futures_pattern.match(k) or k.endswith('prices')], reverse=False)
    spread_df = df[contract_cols].diff(axis=1)
    spread_cols = []
    df = df.drop_duplicates(subset=contract_cols)
    for i in range(2, len(contract_cols)):
        df[f'spread_1_{i}'] = spread_df[f'NG_{i}_prices']
        spread_cols.append(f'spread_1_{i}')
        df[f'{spread_cols[-1]}_1mo_return'] = df[spread_cols[-1]].resample('1M').last().diff(-1)
        df[f'{spread_cols[-1]}_1wk_return'] = df[spread_cols[-1]].resample('1W').last().diff(-1)
        if second_month:
            if i >2 :
                df[f'spread_2_{i}'] = df[contract_cols[1]] - df[contract_cols]
                spread_cols.append(f'spread_2_{i}')
                df[f'{spread_cols[-1]}_1mo_return'] = df[spread_cols[-1]].resample('1M').last().diff(-1)
                df[f'{spread_cols[-1]}_1wk_return'] = df[spread_cols[-1]].resample('1W').last().diff(-1)

    return df

def store_to_df(store_data,
                index_column=['Partner','date'],
                          datetime_date_col=True):
    df = pd.DataFrame(store_data)
    df['date'] = pd.to_datetime(df['date']) if datetime_date_col else df['date']
    df.set_index(keys=index_column, inplace=True) if index_column else None

    return df

def df_to_store(df:pd.DataFrame, reset_index=True, new_data :dict=None):
    store_df = df.copy()
    if reset_index:
        store_df.reset_index(inplace=True)
    if new_data:
        for k, v in new_data.items():
            store_df[k] = v

    data_store = store_df.to_dict('records')
    return data_store


def get_sample_data(commodity: str, year: int, countries: list) -> pd.DataFrame:
    """Generate sample ESR data."""
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


def get_multi_year_sample_data(commodity: str, country: str) -> pd.DataFrame:
    """Get multi-year sample data for country analysis."""
    current_year = pd.Timestamp.now().year
    years = list(range(current_year - 4, current_year + 1))

    all_data = []
    for year in years:
        year_data = get_sample_data(commodity, year, [country])
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
