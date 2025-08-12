import pandas as pd
import numpy as np
import re
month_codes = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10,
               'NOV': 11, 'DEC': 12}
date_codes = {'START': 1, 'END': 28}

year_pattern = re.compile(r"^(19|20)\d{2}$")


def calc_dates(df):
    """
    Given a DataFrame with USDA AgriStats schema, detect the
    appropriate date column (week_ending, or year+begin_code),
    build a datetime, set it as the index, and return the new DF.
    """
    df = df.copy()

    if 'freq_desc' in df.columns and df.freq_desc.eq('WEEKLY').any():
        df['date'] = pd.to_datetime(df['week_ending'])

    elif 'freq_desc' in df.columns and df['freq_desc'].eq('POINT IN TIME').any():
        df['day'] = 28
        df['month'] = df['end_code']
        df['date'] = pd.to_datetime(dict(year=df['year'],month=df['month'], day=df['day']))


    # 2) Monthly
    elif 'freq_desc' in df.columns and df['freq_desc'].eq('MONTHLY').any():
        code_info = df['reference_period_desc'].str.rsplit(' ')
        if len(code_info[0]) == 1:
            df['day'] = 28
            df['month'] = df['begin_code']
        elif len(code_info[0]) >= 3:
            df['day'] = 1 if code_info[0][0] == 'START' else 28
            df['month'] = df['reference_period_desc'].apply(lambda x: month_codes[x.rsplit(' ')[-1]])

        df['date'] = pd.to_datetime(dict(
            year=df['year'],
            month=df['begin_code'].astype(int),
            day=df['day']
        ))

    # 3) Quarterly
    elif 'freq_desc' in df.columns and df['freq_desc'].str.upper().eq('QUARTERLY').any():
        # assume begin_code is quarter number 1–4
        df['date'] = pd.to_datetime(dict(
            year=df['year'],
            month=df['begin_code'].astype(int) * 3,
            day=1
        ))

    # 4) Annual
    elif 'freq_desc' in df.columns and df['freq_desc'].str.upper().eq('ANNUAL').all():
        df['date'] = pd.to_datetime(dict(
            year=df['year'],
            month=12,
            day=31
        ))

    # 5) Fallback: parse “MMM YYYY” from reference_period_desc + year
    elif 'reference_period_desc' in df.columns and 'date' not in df.columns:
        # e.g. “JAN” + “2025” → “JAN 2025”
        if len(df['reference_period_desc'].str.rsplit(' ')[0]) > 2:
            df['reference_period_desc_list'] = df['reference_period_desc'].str.rsplit(' ')
            df["combined"] = df.apply(lambda row: row["reference_period_desc_list"] + [str(row["year"])], axis=1)
            df['date'] = convert_date_series(df["combined"])
        else:
             raise ValueError("No recognizable date columns found.")

    return df['date']


from datetime import datetime
import pandas as pd



def convert_date_series(series_of_lists):
    """
    Converts a Series of lists like ["END", "OF", "AUG", "1917"]
    into a Series of datetime objects representing the start or end of the month.
    """
    def parse_entry(entry):
        try:
            entry = [e.upper() for e in entry]
            position = entry[0]
            month_str = entry[-2].capitalize()
            year_str = entry[-1]
            month_num = datetime.strptime(month_str, "%b").month
            base_date = pd.Timestamp(year=int(year_str), month=month_num, day=1)
            if position == "END":
                return base_date + pd.offsets.MonthEnd(0)
            elif position == "START":
                return base_date
            else:
                return pd.NaT  # Not a valid date
        except Exception as e:
            print(f"Error parsing entry {entry}: {e}")
            return pd.NaT

    return series_of_lists.apply(parse_entry)

def clean_numeric_columns(df):
    """
    Convert string numbers to integers and keep only numeric/datetime columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame

    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with only numeric and datetime columns
    """

    df_clean = df.copy()

    # Convert string columns to numeric where possible
    for col in df_clean.columns:
        if df_clean[col].data_type == 'object':
            # Try converting to numeric
            numeric_series = pd.to_numeric(df_clean[col].str.strip(','), errors='coerce')

            # If most values converted successfully, keep the conversion
            if numeric_series.notna().sum() > len(df_clean) * 0.5:
                df_clean[col] = numeric_series

    # Select only numeric and datetime columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    datetime_cols = df_clean.select_dtypes(include=['datetime64']).columns

    final_cols = list(numeric_cols) + list(datetime_cols)

    return df_clean[final_cols]

def clean_import_df(fp):
    columns = [f'Value.{n}' for n in range(1,12)]
    columns.extend(['Value', 'Year', 'Partner', 'Partner Code'])
    df = pd.read_csv(fp, skiprows=4)


import pandas as pd
from datetime import datetime
import calendar


def clean_fas(file_path, year_col="Year", skip_rows=4,
              month_mapping=None, value_col_name='value',
              additional_id_cols=None):
    """
    Reshape a wide format dataframe (months as columns) to long format with datetime index.

    Handles special "Value" pattern where:
    - 'Value' (no number) = January
    - 'Value.2' through 'Value.11' = February through November
    - 'Value.12' = Total (excluded from output)

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    year_col : int or str, default 0
        Column index or name containing years
    skip_rows : int, default 3
        Number of rows to skip from the top (data starts at row skip_rows+1)
    month_mapping : dict, optional
        Mapping of column names to month numbers if needed
        e.g., {'Jan': 1, 'Feb': 2, ...} or {'January': 1, 'February': 2, ...}
    value_col_name : str, default 'value'
        Name for the value column in the reshaped dataframe
    additional_id_cols : list, optional
        Additional columns to preserve (e.g., ['Partner', 'Partner Code'])
        If None, will automatically detect 'Partner' and 'Partner Code' columns

    Returns:
    --------
    pandas.DataFrame
        Dataframe with datetime index (last day of each month) and values
    """


    columns = ['Year', 'Partner', 'Partner Code', 'Value']
    columns.extend([f'Value.{n}' for n in range(1, 12)])
    # Read the CSV starting from the specified row
    df = pd.read_csv(file_path, skiprows=skip_rows, usecols=columns).dropna()
    df[year_col] = [n[0] for n in df[year_col].astype(str).str.split('-')]
    # Read the CSV starting from the specified row


    # Get year column
    if isinstance(year_col, int):
        year_column = df.iloc[:, year_col]
        year_col_name = df.columns[year_col]
    else:
        year_column = df[year_col]
        year_col_name = year_col

    # Identify additional ID columns to preserve
    if additional_id_cols is None:
        # Auto-detect Partner and Partner Code columns
        additional_id_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if 'partner' in col_lower and 'code' in col_lower:
                additional_id_cols.append(col)
            elif col_lower == 'partner':
                additional_id_cols.append(col)

    # Ensure additional_id_cols exist in dataframe
    additional_id_cols = [col for col in additional_id_cols if col in df.columns]

    # All ID columns (year + additional)
    all_id_cols = [year_col_name] + additional_id_cols

    # Get month columns (all columns except the ID columns)
    month_columns = [col for col in df.columns if col not in all_id_cols]

    # Create month mapping if not provided
    if month_mapping is None:
        month_mapping = {}
        for col in month_columns:
            col_str = str(col).strip()

            # Check for "Value" pattern (special case: Value = Jan, Value.2-11 = Feb-Nov, Value.12 = Total)
            if col_str.lower() == 'value':
                # 'Value' without number = January (month 1)
                month_mapping[col] = 1
                continue
            elif col_str.lower().startswith('value.'):
                try:
                    month_num = int(col_str.split('.')[1])
                    if 2 <= month_num <= 11:  # Only months 2-11, exclude 12 (total)
                        month_mapping[col] = month_num
                        continue
                    elif month_num == 12:
                        # Skip Value.12 as it's the total, not December
                        print(f"Skipping '{col}' as it appears to be a total column")
                        continue
                except (ValueError, IndexError):
                    pass

            # Check if it's a direct number (1-12)
            try:
                month_num = int(col_str)
                if 1 <= month_num <= 12:
                    month_mapping[col] = month_num
                    continue
            except ValueError:
                pass

            # Check common month abbreviations/names
            month_names = {
                'jan': 1, 'january': 1,
                'feb': 2, 'february': 2,
                'mar': 3, 'march': 3,
                'apr': 4, 'april': 4,
                'may': 5,
                'jun': 6, 'june': 6,
                'jul': 7, 'july': 7,
                'aug': 8, 'august': 8,
                'sep': 9, 'sept': 9, 'september': 9,
                'oct': 10, 'october': 10,
                'nov': 11, 'november': 11,
                'dec': 12, 'december': 12
            }

            col_lower = col_str.lower()
            if col_lower in month_names:
                month_mapping[col] = month_names[col_lower]
            else:
                # Try partial matches
                for name, num in month_names.items():
                    if col_lower.startswith(name) or name.startswith(col_lower):
                        month_mapping[col] = num
                        break

    # Melt the dataframe from wide to long format
    df_melted = df.melt(id_vars=all_id_cols,
                        value_vars=month_columns,
                        var_name='month_col',
                        value_name=value_col_name)

    # Map month columns to month numbers
    df_melted['month'] = df_melted['month_col'].map(month_mapping)

    # Remove rows where month mapping failed
    df_melted = df_melted.dropna(subset=['month'])
    df_melted['month'] = df_melted['month'].astype(int)

    # Create datetime index with last day of month
    datetime_index = []
    for _, row in df_melted.iterrows():
        year = int(row[year_col_name])
        month = int(row['month'])

        # Get the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        date = datetime(year, month, last_day)
        datetime_index.append(date)

    # Create final dataframe with additional columns
    result_data = {value_col_name: df_melted[value_col_name].values}

    # Add additional ID columns to the result
    for col in additional_id_cols:
        result_data[col] = df_melted[col].values

    result_data['date'] = datetime_index

    result_df = pd.DataFrame(result_data)
    result_df['value'] = result_df['value'].str.replace(',', '')
    result_df['value'] = result_df['value'].str.replace('-', '')
    result_df = result_df.loc[~(result_df['value'] == '')]
    result_df.set_index(['Partner', 'date'], inplace=True)
    result_df['value'] = result_df['value'].astype(float)


    # Sort by datetime index
    result_df = result_df.sort_index()


    return result_df


def reshape_dataframe_to_datetime_index(df, year_col=0, month_mapping=None, value_col_name='value',
                                        additional_id_cols=None):
    """
    Reshape an already loaded wide format dataframe to datetime index.

    Handles special "Value" pattern where:
    - 'Value' (no number) = January
    - 'Value.2' through 'Value.11' = February through November
    - 'Value.12' = Total (excluded from output)

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with years in first column and months as other columns
    year_col : int or str, default 0
        Column index or name containing years
    month_mapping : dict, optional
        Mapping of column names to month numbers
    value_col_name : str, default 'value'
        Name for the value column in the reshaped dataframe
    additional_id_cols : list, optional
        Additional columns to preserve (e.g., ['Partner', 'Partner Code'])
        If None, will automatically detect 'Partner' and 'Partner Code' columns

    Returns:
    --------
    pandas.DataFrame
        Dataframe with datetime index (last day of each month) and values
    """

    # Get year column
    if isinstance(year_col, int):
        year_column = df.iloc[:, year_col]
        year_col_name = df.columns[year_col]
    else:
        year_column = df[year_col]
        year_col_name = year_col

    # Identify additional ID columns to preserve
    if additional_id_cols is None:
        # Auto-detect Partner and Partner Code columns
        additional_id_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if 'partner' in col_lower and 'code' in col_lower:
                additional_id_cols.append(col)
            elif col_lower == 'partner':
                additional_id_cols.append(col)

    # Ensure additional_id_cols exist in dataframe
    additional_id_cols = [col for col in additional_id_cols if col in df.columns]

    # All ID columns (year + additional)
    all_id_cols = [year_col_name] + additional_id_cols

    # Get month columns (all columns except the ID columns)
    month_columns = [col for col in df.columns if col not in all_id_cols]

    # Create month mapping if not provided (same logic as above)
    if month_mapping is None:
        month_mapping = {}
        for col in month_columns:
            col_str = str(col).strip()

            # Check for "Value" pattern (special case: Value = Jan, Value.2-11 = Feb-Nov, Value.12 = Total)
            if col_str.lower() == 'value':
                # 'Value' without number = January (month 1)
                month_mapping[col] = 1
                continue
            elif col_str.lower().startswith('value.'):
                try:
                    month_num = int(col_str.split('.')[1])
                    if 2 <= month_num <= 11:  # Only months 2-11, exclude 12 (total)
                        month_mapping[col] = month_num
                        continue
                    elif month_num == 12:
                        # Skip Value.12 as it's the total, not December
                        continue
                except (ValueError, IndexError):
                    pass

            # Check if it's a direct number (1-12)
            try:
                month_num = int(col_str)
                if 1 <= month_num <= 12:
                    month_mapping[col] = month_num
                    continue
            except ValueError:
                pass

            month_names = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
                'mar': 3, 'march': 3, 'apr': 4, 'april': 4, 'may': 5,
                'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
                'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                'dec': 12, 'december': 12
            }

            col_lower = col_str.lower()
            if col_lower in month_names:
                month_mapping[col] = month_names[col_lower]
            else:
                for name, num in month_names.items():
                    if col_lower.startswith(name) or name.startswith(col_lower):
                        month_mapping[col] = num
                        break

    # Melt and process (same as above)
    df_melted = df.melt(id_vars=all_id_cols,
                        value_vars=month_columns,
                        var_name='month_col',
                        value_name=value_col_name)

    df_melted['month'] = df_melted['month_col'].map(month_mapping)
    df_melted = df_melted.dropna(subset=['month'])
    df_melted['month'] = df_melted['month'].astype(int)

    # Create datetime index with last day of month using vectorized approach
    df_melted['temp_date'] = pd.to_datetime(df_melted[[year_col_name, 'month']].assign(day=1))
    datetime_index = df_melted['temp_date'] + pd.offsets.MonthEnd(0)

    # Create final dataframe with additional columns
    result_data = {value_col_name: df_melted[value_col_name].values}

    # Add additional ID columns to the result
    for col in additional_id_cols:
        result_data[col] = df_melted[col].values

    result_df = pd.DataFrame(result_data, index=datetime_index)

    return result_df.sort_index()
# Example usage

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

def pivot_data(df, years):
    df_melted = df.melt(
        id_vars=["commodity", "attribute", "country"],
        value_vars=years,
        var_name="year",
        value_name="value"
    )

    # Convert year to integer
    df_melted["year"] = df_melted["year"].astype(int)

    # Now pivot: years as index, attributes as columns
    pivoted = df_melted.pivot_table(
        index="year",
        columns="attribute",
        values="value",
    )

    # Optional: flatten MultiIndex columns
    pivoted.columns.name = None
    return pivoted

def clean_livestock(livestock_df, end_month=12):
    year_cols = [col for col in livestock_df.columns if year_pattern.match(str(col))]
    suffix = f'-{end_month}-31'
    pivoted = pivot_data(livestock_df, year_cols)
    new_idx = pd.to_datetime(pivoted.index.astype(str) + suffix)
    pivoted['date'] = new_idx
    pivoted.set_index('date')

    return pivoted

def clean_grains(grain_df, end_month=7):
    df = grain_df.copy(deep=True)
    name_map = {}
    suffix = f'{end_month}-30'
    for i, col in enumerate(df.columns):
        if isinstance(col, str):
            if len(col.split('/')) > 1:
                name_map.update({str(col):int(col.split('/')[-1])})

    new_grains = df.rename(name_map, axis=1)

    pivoted = pivot_data(new_grains, name_map.values())

    return pivoted








