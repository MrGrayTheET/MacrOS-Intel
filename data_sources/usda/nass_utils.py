import pandas as pd

def calc_dates(df):
    """
    Given a DataFrame with USDA AgriStats schema, detect the
    appropriate date column (week_ending, or year+begin_code),
    build a datetime, set it as the index, and return the new DF.
    """
    df = df.copy()

    # 2) Monthly
    if 'freq_desc' in df.columns and df['freq_desc'].eq('MONTHLY').any():
        df['date'] = pd.to_datetime(dict(
            year = df['year'],
            month = df['begin_code'].astype(int),
            day = 1
        ))

    # 3) Quarterly
    elif 'freq_desc' in df.columns and df['freq_desc'].str.upper().eq('QUARTERLY').any():
        # assume begin_code is quarter number 1–4
        df['date'] = pd.to_datetime(dict(
            year  = df['year'],
            month = df['begin_code'].astype(int) * 3,
            day   = 1
        ))

    # 4) Annual
    elif 'freq_desc' in df.columns and df['freq_desc'].str.upper().eq('ANNUAL').any():
        df['date'] = pd.to_datetime(dict(
            year  = df['year'],
            month = 12,
            day   = 31
        ))

    # 5) Fallback: parse “MMM YYYY” from reference_period_desc + year
    elif 'reference_period_desc' in df.columns:
        # e.g. “JAN” + “2025” → “JAN 2025”
        combo = df['reference_period_desc'].str.strip() + ' ' + df['year'].astype(str)
        df['date'] = pd.to_datetime(combo, format='%b %Y', errors='coerce')

    else:
        raise ValueError("No recognizable date columns found.")

    return df['date']

def clean_numeric_column(series):
    """
    Cleans a pandas Series with values like '7,000,000+' and converts to numeric.
    """
    return (
        series.astype(str)
              .str.replace(',', '', regex=False)
              .str.replace('+', '', regex=False)
              .str.strip()
              .replace('', pd.NA)
              .astype(float)
    )