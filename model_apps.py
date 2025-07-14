from dotenv import load_dotenv

load_dotenv('../.env')
from data_sources.tables import TableClient, EIATable
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from dash import dcc, html, dash_table, callback, Input, Output, State, no_update, callback_context
import plotly.express as px
import re
from utils import key_to_name, generate_layout_keys, fetch_contract_keys, calc_contract_spreads
from plotly import graph_objects as go


class ModelApp:
    """
    Enhanced base class with flexible filtering capabilities
    """

    def __init__(self, table_client: TableClient, keys=None, app_id='model', enforce_dt=True):
        self.app_func = None
        self.table_client = table_client
        self.data = pd.DataFrame()
        self.enforce_datetime = enforce_dt
        # Dictionary to store filter configurations
        self.filter_configs = {}
        self.filtered_data = pd.DataFrame()  # Will hold filtered data
        self.div_layout_params = {}


        if keys:
            self.data = self.__add__(keys)

        if not self.data.empty:
            self.data.sort_index(ascending=False)

        self.app_id = app_id

    def configure_filters(self, filter_definitions):
        self.filter_configs = filter_definitions or {}
        # If selected_series is requested, append dynamically
        if 'signal_series' in self.filter_configs:
            self.filter_configs['signal_series']['options'] = self.data.columns.tolist()


    def get_filter_options(self, column_name):
        """Dynamically generate filter options based on data"""
        if column_name == 'signal_series':
            return  {'unique_values':self.data.columns.tolist(),
                     }
        if column_name not in self.data.columns:
            return None

        col_data = self.data[column_name].dropna()

        if pd.api.types.is_numeric_dtype(col_data):
            unique_vals = sorted(col_data.unique())
            unique_vals = [val.item() if isinstance(val, np.generic) else val for val in unique_vals]
            return {
                'min': col_data.min().item(),
                'max': col_data.max().item(),
                'unique_values': unique_vals if len(unique_vals) <= 20 else None,
                'default_range': [col_data.min().item(), col_data.max().item()]  # Add default range
            }
        else:
            unique_vals = sorted(col_data.unique())
            unique_vals = [val.item() if isinstance(val, np.generic) else val for val in unique_vals]
            return {
                'unique_values': unique_vals,
                'default_range': [unique_vals[0], unique_vals[-1]] if unique_vals else None
            }

    def generate_filter_menu(self, filter_columns=None, filter_configs=None, include_series_selector=True):
        if filter_columns is None:
            self.filter_configs.keys()

        if filter_configs:
            self.filter_configs = filter_configs

        if not filter_columns and not self.filter_configs:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            potential_filters = []
            for col in numeric_cols:
                if 'month' in col.lower() or 'year' in col.lower():
                    potential_filters.append(col)
                elif col.nunique() <= 20:
                    potential_filters.append(col)
            filter_columns = potential_filters[:4] + ['signal_series'] if include_series_selector else potential_filters[:4]
        filter_elements = []

        for col in self.filter_configs.keys():
            config = self.filter_configs.get(col, {})
            label = config.get('label', col.replace('_', ' ').title())
            filter_type = config.get('type', 'range')

            # If source = columns, it's a meta-selector like signal_series
            if config.get('source') == 'columns':
                match = config.get('match', {})
                all_cols = self.data.columns.tolist()

                if 'contains' in match:
                    matched_cols = [c for c in all_cols if match['contains'] in c]
                elif 'endswith' in match:
                    matched_cols = [c for c in all_cols if c.endswith(match['endswith'])]
                elif 'startswith' in match:
                    matched_cols = [c for c in all_cols if c.startswith(match['startswith'])]
                else:
                    matched_cols = all_cols

                if filter_type == 'single_select':
                    filter_element = html.Div([
                        html.Label(label, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id=f'{self.app_id}-{col}-dropdown',
                            options=[{'label': c, 'value': c} for c in matched_cols],
                            value=matched_cols[0] if matched_cols else None,
                            placeholder=f"Select {label.lower()}..."
                        )
                    ], style={'marginBottom': '20px'})

                elif filter_type == 'multi_select':
                    filter_element = html.Div([
                        html.Label(label, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Checklist(
                            id=f'{self.app_id}-{col}-checklist',
                            options=[{'label': c, 'value': c} for c in matched_cols],
                            value=matched_cols[:3],
                            style={'maxHeight': '200px', 'overflowY': 'auto'}
                        )
                    ], style={'marginBottom': '20px'})

                filter_elements.append(filter_element)
                continue  # skip to next filter_column

            # Otherwise this is a column-based filter
            if col not in self.data.columns:
                continue

            col_options = self.get_filter_options(col)
            if col_options is None:
                continue

            if filter_type == 'range':
                if col_options['unique_values'] and len(col_options['unique_values']) <= 20:
                    filter_element = html.Div([
                        html.Label(label, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.RangeSlider(
                            id=f'{self.app_id}-{col}-range-slider',
                            min=col_options['min'],
                            max=col_options['max'],
                            step=1,
                            value=[col_options['min'], col_options['max']],
                            marks={val: str(val) for val in
                                   col_options['unique_values'][::max(1, len(col_options['unique_values']) // 5)]},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'marginBottom': '20px'})
                else:
                    filter_element = html.Div([
                        html.Label(label, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        html.Div([
                            dcc.Input(
                                id=f'{self.app_id}-{col}-min-input',
                                type='number',
                                placeholder=f'Min ({col_options["min"]:.2f})',
                                value=col_options['min'],
                                style={'width': '45%', 'marginRight': '10%'}
                            ),
                            dcc.Input(
                                id=f'{self.app_id}-{col}-max-input',
                                type='number',
                                placeholder=f'Max ({col_options["max"]:.2f})',
                                value=col_options['max'],
                                style={'width': '45%'}
                            )
                        ], style={'display': 'flex'})
                    ], style={'marginBottom': '20px'})

            elif filter_type == 'categorical':
                options = [{'label': str(val), 'value': val} for val in col_options['unique_values']]
                filter_element = html.Div([
                    html.Label(label, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id=f'{self.app_id}-{col}-dropdown',
                        options=options,
                        value=col_options['unique_values'],
                        multi=True,
                        placeholder=f"Select {label.lower()}..."
                    )
                ], style={'marginBottom': '20px'})

            elif filter_type == 'multi_select':
                options = [{'label': str(val), 'value': val} for val in col_options['unique_values']]
                filter_element = html.Div([
                    html.Label(label, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Checklist(
                        id=f'{self.app_id}-{col}-checklist',
                        options=options,
                        value=col_options['unique_values'],
                        style={'maxHeight': '150px', 'overflowY': 'auto'}
                    )
                ], style={'marginBottom': '20px'})

            filter_elements.append(filter_element)

        # Control buttons
        filter_elements.append(html.Div([
            html.Button(
                "Apply Filters",
                id=f'{self.app_id}-apply-filters-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#28a745',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%'
                }
            ),
            html.Button(
                "Reset Filters",
                id=f'{self.app_id}-reset-filters-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#dc3545',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%',
                    'marginTop': '10px'
                }
            )
        ]))

        return html.Div([
            html.H3("Data Filters", style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div(filter_elements, style={'padding': '10px'})
        ], style={
            'width': '35%',
            'padding': '15px',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'backgroundColor': '#f8f9fa',
            'boxSizing': 'border-box'
        })

    def _create_series_selector(self):
        """
        Create a series selector component for choosing which data series to display
        """
        if self.data.empty:
            return None

        # Get all available columns
        all_columns = self.data.columns.tolist()

        # Try to categorize columns intelligently
        categories = self._categorize_columns(all_columns)

        series_elements = []

        # If we have categories, create collapsible sections
        if len(categories) > 1:
            for category, columns in categories.items():
                if not columns:
                    continue

                category_div = html.Details([
                    html.Summary(
                        f"{category} ({len(columns)})",
                        style={'fontWeight': 'bold', 'cursor': 'pointer', 'marginBottom': '10px'}
                    ),
                    dcc.Checklist(
                        id=f'{self.app_id}-series-{category.lower().replace(" ", "-")}',
                        options=[{'label': col, 'value': col} for col in columns],
                        value=columns[:2] if len(columns) > 2 else columns,  # Default: first 2 series
                        style={'paddingLeft': '20px', 'maxHeight': '150px', 'overflowY': 'auto'}
                    )
                ], open=(category == list(categories.keys())[0]))  # Open first category by default

                series_elements.append(category_div)
        else:
            # Single checklist if no clear categories
            series_elements.append(
                dcc.Checklist(
                    id=f'{self.app_id}-series-all',
                    options=[{'label': col, 'value': col} for col in all_columns],
                    value=all_columns[:3] if len(all_columns) > 3 else all_columns,  # Default: first 3
                    style={'maxHeight': '200px', 'overflowY': 'auto'}
                )
            )

        # Add select all/none buttons
        series_selector = html.Div([
            html.Label("Select Series to Display", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.Div([
                html.Button(
                    "Select All",
                    id=f'{self.app_id}-select-all-series',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#007bff',
                        'color': 'white',
                        'padding': '5px 10px',
                        'border': 'none',
                        'borderRadius': '3px',
                        'cursor': 'pointer',
                        'marginRight': '10px',
                        'fontSize': '12px'
                    }
                ),
                html.Button(
                    "Clear All",
                    id=f'{self.app_id}-clear-all-series',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#6c757d',
                        'color': 'white',
                        'padding': '5px 10px',
                        'border': 'none',
                        'borderRadius': '3px',
                        'cursor': 'pointer',
                        'fontSize': '12px'
                    }
                )
            ], style={'marginBottom': '10px'}),
            html.Div(series_elements)
        ], style={'marginBottom': '20px'})

        return series_selector

    def _categorize_columns(self, columns):
        """
        Intelligently categorize columns based on naming patterns and data types
        """
        categories = {
            'Contracts': [],
            'Spreads': [],
            'Changes/Returns': [],
            'Statistics': [],
            'Other': []
        }

        # Common patterns for categorization
        contract_patterns = [r'[A-Za-z]{2}_\d+', r'ct_\d+', r'contract']
        spread_patterns = ['spread', 'diff', 'basis']
        change_patterns = ['change', 'return', 'pct', 'growth']
        stat_patterns = ['mean', 'std', 'var', 'median', 'max', 'min', 'sum', 'count']

        for col in columns:
            col_lower = col.lower()
            categorized = False

            # Check contract patterns
            for pattern in contract_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    categories['Contracts'].append(col)
                    categorized = True
                    break

            if not categorized:
                # Check for spreads
                if any(pattern in col_lower for pattern in spread_patterns) and \
                        not any(pattern in col_lower for pattern in change_patterns):
                    categories['Spreads'].append(col)
                    categorized = True
                # Check for changes/returns
                elif any(pattern in col_lower for pattern in change_patterns):
                    categories['Changes/Returns'].append(col)
                    categorized = True
                # Check for statistics
                elif any(pattern in col_lower for pattern in stat_patterns):
                    categories['Statistics'].append(col)
                    categorized = True

            if not categorized:
                categories['Other'].append(col)

        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}

        # If only "Other" category exists, return empty dict (will use single list)
        if len(categories) == 1 and 'Other' in categories:
            return {}

        return categories

    def apply_filters(self, data, filter_values):
        """
        Apply filters to the data based on filter values

        Args:
            data (pd.DataFrame): Data to filter
            filter_values (dict): Dictionary of filter values from callbacks

        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = data.copy()

        for col, config in self.filter_configs.items():
            if col not in data.columns:
                continue

            filter_type = config.get('type', 'range' if pd.api.types.is_numeric_dtype(data[col]) else 'categorical')

            if filter_type == 'range':
                # Handle range filters
                if f'{col}_range' in filter_values and filter_values[f'{col}_range']:
                    min_val, max_val = filter_values[f'{col}_range']
                    filtered_data = filtered_data[
                        (filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)
                        ]
                elif f'{col}_min' in filter_values and f'{col}_max' in filter_values:
                    min_val = filter_values[f'{col}_min']
                    max_val = filter_values[f'{col}_max']
                    if min_val is not None and max_val is not None:
                        filtered_data = filtered_data[
                            (filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)
                            ]

            elif filter_type in ['categorical', 'multi_select']:
                # Handle categorical filters
                if f'{col}_values' in filter_values and filter_values[f'{col}_values']:
                    selected_values = filter_values[f'{col}_values']
                    filtered_data = filtered_data[filtered_data[col].isin(selected_values)]

        return filtered_data

    def run(self, **params):
        res = self.app_func(**params)
        return res

    def __add__(self, key):
        srs = None
        if isinstance(key, str):
            srs = self.table_client.get_key(key)
            srs.sort_index(ascending=True)
            srs = srs.rename(columns={srs.columns[0]: key_to_name(key)})

        if isinstance(key, list):
            srs = self.table_client.get_keys(key)
            col_names = []
            for i in key:
                col_names.append(key_to_name(i))
            srs = srs.rename(columns=dict(zip(srs.columns, col_names)))

        if self.enforce_datetime and srs is not None:
            if isinstance(srs.index, (pd.Index)): srs.index = pd.to_datetime(srs.index)

        self.data = pd.concat([self.data, srs], axis=1) if srs is not None else self.data

        return self.data

    def __delattr__(self, item):
        self.data.drop(columns=item, axis=1, inplace=True)
        return

    def layout_keys(self, categories=[]):
        return generate_layout_keys(self.table_client, categories)

    def generate_menu(self, categories=['storage', 'prices', 'production', 'demand'], height="10%", width="30%", multi_select_x=False):
        """Generate menu content with dropdowns for each category"""
        # Get layout keys for menu dropdowns
        layout_keys = self.layout_keys(categories)

        dropdown_options = [
            {'label': display_name, 'value': hdf_key}
            for display_name, hdf_key in layout_keys.items()
        ]


        # Create menu content with dropdowns for each category
        menu_content = html.Div([
            html.H3("Data Selection Menu", style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([dcc.Tabs(children=[
                dcc.Tab(children=[
                html.Div([
                    html.Label(f"{category.title()} Selection:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id=f'{self.app_id}-{category}-dropdown',
                        options=[
                            {'label': key, 'value': value}
                            for key, value in layout_keys[category].items()
                        ],
                        value=getattr(self, f'{category}_key', None) if hasattr(self, f'{category}_key') else
                        (layout_keys.get(category, [None])[0] if layout_keys.get(category) else None),
                        placeholder=f"Select {category}...",
                        style={'marginBottom': '15px'}
                    )
                ], style={'marginBottom': '20px'})
                for category in categories
            ]),
            dcc.Tab(
                html.Div([
                    html.H3('Column Settings', style={'textAlign': 'center', 'marginBottom': '20px'}),
                    html.Label("Dependent Variable (Y-axis):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(id=f'{self.app_id}-y-dd',
                                 options=[col for col in self.data.columns if ('return' in col) or ('spread' in col)],
                                 value='return_1mo' if 'return_1mo' in self.data.columns.str.lower() else None,
                                 placeholder=f'Select Y-axis',
                                 style={
                                     'marginBottom': '15px'}
                                 ),
                    html.Label("Independent Variable (X-axis)" ,style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(id=f'{self.app_id}-x-dd',
                                 options=[col for col in self.data.columns],
                                 value='total_lower_48_storage',
                                 placeholder='Select X-axis',
                                 style= {'marginBottom': '15px'})

                ]))
            ]) # Tab End
            ]), # Div


            html.Div([
                html.Label("Select Date:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.DatePickerSingle(
                    id=f'{self.app_id}-date-picker',
                    date=datetime.today().date(),
                    display_format='YYYY-MM-DD',
                    style={
                        'width': '100%'
                    }
                )
            ], ),

            html.Div([
                html.Button("Update", id=f"{self.app_id}-update-button", n_clicks=0,
                            style={
                                'backgroundColor': '#007bff',
                                'color': 'white',
                                'padding': '10px 20px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'cursor': 'pointer'
                            })
            ], style={'textAlign': 'center', 'marginTop': '20px'})
        ], style={
            'width': '20%',
            'height': '5%',
            'margin': 'auto',
            'marginBottom': '20px',
            'marginLeft': '100px',
            'padding': '10px',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'boxSizing': 'border-box'
        }, className="DataSelector")

        return menu_content


class StorageApp(ModelApp):
    """
    Enhanced StorageApp with flexible filtering capabilities
    """

    def __init__(self, storage_key='storage/total_lower_48', price_key='prices/NG_1', withdrawal_key ='consumption/net_withdrawals'):
        super().__init__(table_client=EIATable('NG'), keys=[storage_key, price_key, withdrawal_key])
        self.storage_key = storage_key
        self.prices_key = price_key
        self.withdrawal_key = withdrawal_key
        self.prices_col = key_to_name(price_key)
        self.storage_col = key_to_name(storage_key)
        self.withdrawal_col = key_to_name(withdrawal_key)
        self.data.ffill(inplace=True)
        self.data = self.data.resample('1W').apply({self.storage_col: 'last', self.prices_col: 'last', self.withdrawal_col:
                                                    'last'})
        # Calculate storage metrics
        self.app_func = self.seasonal_storage_data
        self.data = self.run()
        # Configure filters for this specific application
        self.configure_filters({
            'month': {
                'type': 'range',
                'label': 'Month',
                'min_val': 1,
                'max_val': 12,

            },
            'signal_strength_decile': {
                'type': 'range',
                'label': 'Signal Strength Decile',
                'min_val':0,
                'max_val':9
            },
        })

    def update_keys(self, storage_key=None, price_key=None):
        keys_to_add = []
        cols_to_drop = []
        if isinstance(self.data.index, pd.Index):
            self.data.index = pd.to_datetime(self.data.index)

        if storage_key:
            if storage_key != self.storage_key:
                cols_to_drop.append(self.storage_col)
                self.storage_key = storage_key
                self.storage_col = key_to_name(storage_key)
                keys_to_add.append(storage_key)

        if price_key:
            if self.prices_key != price_key:
                cols_to_drop.append(self.prices_col)
                self.prices_key = price_key
                self.prices_col = key_to_name(price_key)
                keys_to_add.append(price_key)

        # Preserve the index range before adding new data
        min_idx, max_idx = self.data.index.min(), self.data.index.max()

        if keys_to_add:
            self.__add__(keys_to_add)
            self.data = self.data.ffill().drop(columns=cols_to_drop)


            # Filter Non-datetimes from index
            self.data = self.data[[isinstance(idx, (pd.Timestamp, datetime)) for idx in self.data.index]]
            self.data.index = pd.to_datetime(self.data.index) if isinstance(self.data.index,
                                                                            pd.Index) else self.data.index

            # Resample to weekly and recalculate metrics
            self.data = self.data.resample('1W').apply({
                self.storage_col: 'last',
                self.prices_col: 'last',
                self.withdrawal_col: 'last'
            })

            self.data = self.seasonal_storage_data()

        return self.data

    def seasonal_storage_data(self):
        df = self.data.copy()
        df['date'] = df.index
        df = df.sort_values('date')

        # Add week-of-year, month, and year columns
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Calculate seasonal historical min/max (same week, past 5 years)
        df['hist_max'] = np.nan
        df['hist_min'] = np.nan
        df['hist_mean'] = np.nan

        for idx, row in df.iterrows():
            current_week = row['week_of_year']
            current_year = row['year']

            # Get historical data (same week, past 5 years including current year)
            historical_mask = (
                    (df['week_of_year'] == current_week) &
                    (df['year'] <= current_year) &
                    (df['year'] > current_year - 5) &
                    (df.index <= idx)
            )
            historical_data = df.loc[historical_mask, self.storage_col]

            if len(historical_data) > 0:
                df.loc[idx, 'hist_max'] = historical_data.max()
                df.loc[idx, 'hist_min'] = historical_data.min()
                df.loc[idx, 'hist_mean'] = historical_data.mean()

        # Calculate percentage position within historical range
        df['pct_above_min'] = ((df[self.storage_col] - df['hist_min']) /
                               (df['hist_max'] - df['hist_min'])) * 100

        df['dev_from_mean'] = ((df[self.storage_col] - df['hist_mean']) / df[self.storage_col].std())

        # Calculate future price change (4 weeks ahead)
        df['price_1mo'] = df[self.prices_col].shift(-4)
        df['price_3mo'] = df[self.prices_col].shift(-12)
        df['return_1mo'] = np.log(df.price_1mo) - np.log(df[self.prices_col])
        df['return_3mo'] = np.log(df.price_3mo) - np.log(df[self.prices_col])
        df['signal_strength_decile'] = pd.qcut(df['dev_from_mean'],10, labels=False)

        return df

    def generate_layout(self):
        """Generate the enhanced Dash layout with flexible filtering"""
        # Filter dates to exclude last 4 weeks to avoid incomplete forward-looking data
        valid_dates = self.data[self.data['date'] <= self.data['date'].max() - timedelta(weeks=4)]

        if 'date' in valid_dates.columns:
            min_date = valid_dates['date'].min()
            max_date = valid_dates['date'].max()
            default_date = max_date
        else:
            min_date = valid_dates.index.min()
            max_date = valid_dates.index.max()
            default_date = max_date

        # Generate menus using the enhanced base class functionality
        menu_div = self.generate_menu(categories=['prices', 'storage'])

        # Generate the flexible filter menu
        filter_menu = self.generate_filter_menu(
            filter_columns=['month', 'signal_strength_decile'],  # Only these 2
            filter_configs={  # Only include configs for these 2
                'month': self.filter_configs['month'],
                'signal_strength_decile': self.filter_configs['signal_strength_decile']
            },
            include_series_selector=False
        )

        layout = html.Div([
            html.H1("Natural Gas Storage-Price Analysis", style={'textAlign': 'center'}),
            html.Div(children=[
                menu_div,
                filter_menu,
                html.Div(id=f'{self.app_id}-scatter-plot-div', children=[
                    dcc.Graph(id=f'{self.app_id}-scatter-plot')],
                         style={
                             'height': '20%',
                             'width': '50%',
                             'marginTop': '5px'
                         })],
                style={'display': 'flex', 'height': '20%', 'width': '100%'}),

            html.Div(id=f'{self.app_id}-line-plot-div', children=[

                dcc.Tabs(children=[
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-storage-plot')
                    ]),
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-withdrawal-plot')
                    ])
                ]),
            ],
                     style={
                         'height': '60%',
                         'width': '100%',
                         'marginTop': '10px'
                     }),

            html.Div(id=f'{self.app_id}-metrics-container', className='metrics-grid', style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '15px',
                'margin': '20px'
            }),

            html.Div([
                html.H3("Analysis Summary"),
                html.Div(id=f'{self.app_id}-summary')
            ], style={'margin': '20px'})
        ])

        return layout

    def register_callbacks(self, app):
        @callback(
            [Output(f'{self.app_id}-storage-plot', 'figure'),
             Output(f'{self.app_id}-withdrawal-plot', 'figure'),
             Output(f'{self.app_id}-metrics-container', 'children'),
             Output(f'{self.app_id}-scatter-plot', 'figure'),
             Output(f'{self.app_id}-summary', 'children'),
             Output(f'{self.app_id}-month-range-slider', 'value'),
             Output(f'{self.app_id}-signal_strength_decile-range-slider', 'value')],
            [Input(f'{self.app_id}-date-picker', 'date'),
             Input(f'{self.app_id}-apply-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-reset-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-prices-dropdown', 'value'),
             Input(f'{self.app_id}-storage-dropdown', 'value'),
             Input(f'{self.app_id}-month-range-slider', 'value'),
             Input(f'{self.app_id}-signal_strength_decile-range-slider', 'value')],
            prevent_initial_call=False
        )
        def update_analysis(selected_date, apply_clicks, reset_clicks,
                            price_key, storage_key, month_range, ssq_values):
            # Get callback context to identify trigger
            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            selected_date = pd.to_datetime(selected_date)

            # Handle key updates first (affects both original and filtered data)
            if price_key or storage_key:
                self.update_keys(price_key=price_key, storage_key=storage_key)
                self.original_data = self.data.copy()  # Update original data
                self.filtered_data = pd.DataFrame()  # Reset filtered data

            # Use original data as base for processing
            df = self.original_data.copy().dropna(subset=[self.withdrawal_col, self.prices_col, self.storage_col])

            # Handle reset action
            if trigger_id == f'{self.app_id}-reset-filters-btn':
                self.filtered_data = pd.DataFrame()  # Clear filtered data
                # Get default values for filters
                month_options = self.get_filter_options('month')
                ssq_options = self.get_filter_options('signal_strength_decile')
                reset_month_range = [month_options['min'], month_options['max']]
                reset_ssq_values = [ssq_options['min'], ssq_options['max']]
                summary_text = "Filters reset successfully"

            # Handle apply action
            elif trigger_id == f'{self.app_id}-apply-filters-btn':
                filter_values = {
                    'month_range': month_range,
                    'signal_strength_decile_values': ssq_values
                }
                self.filtered_data = self.apply_filters(df, filter_values)
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = f"Applied filters: {len(self.filtered_data)}/{len(df)} records"

            # No filter action - use last state
            else:
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = "Using current data selection"

            # Determine which dataset to use for visualizations
            display_df = self.filtered_data if not self.filtered_data.empty else df
            # Get data for selected date (same logic as before)
            if 'date' in df.columns:
                selected_data = df[df['date'] == selected_date]
            else:
                selected_data = df[df.index == selected_date]

            if selected_data.empty:
                if 'date' in df.columns:
                    closest_date = df.loc[(df['date'] - selected_date).abs().idxmin(), 'date']
                    selected_data = df[df['date'] == closest_date]
                else:
                    delta_series = pd.Series(df.index - selected_date)
                    closest_idx = delta_series.abs().idxmin()
                    selected_data = df.iloc[[closest_idx]]

            selected_row = selected_data.iloc[0]

            # Create visualizations (same logic as before but with filtered data)
            storage_fig = go.Figure()

            # Add storage level
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df[self.storage_col],
                name='Current Storage',
                line=dict(color='blue', width=2)
            ))

            # Add 5-year max/min bands
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_max'],
                name='5-Year Max',
                line=dict(color='red', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_min'],
                name='5-Year Min',
                line=dict(color='green', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_mean'],
                name='5-Year Mean',
                line=dict(color='orange', dash='dot')
            ))

            storage_fig.update_layout(
                title="Natural Gas Storage Levels vs Historical Range",
                xaxis_title="Date",
                yaxis_title="Storage Volume (BCF)",
                hovermode='x unified'
            )

            withdrawal_fig = go.Figure(go.Scatter(
                x=df['date'],
                y=df[self.withdrawal_col],
                line=dict(color='blue')
                ),
                layout=dict(xaxis_title = "Date",
                yaxis_title= "Net Withdrawals (BCF)"))

            # Prepare metrics (same as before)
            metrics = [
                ("Current Storage", f"{selected_row[self.storage_col]:,.0f} BCF"),
                ("5-Year Max", f"{selected_row['hist_max']:,.0f} BCF"),
                ("5-Year Min", f"{selected_row['hist_min']:,.0f} BCF"),
                ("% Above 5Y Min", f"{selected_row['pct_above_min']:.1f}%"),
                ("Current Price", f"${selected_row[self.prices_col]:,.2f}"),
                ("Price 1Mo Later",
                 f"${selected_row['price_1mo']:,.2f}" if pd.notna(selected_row['price_1mo']) else "N/A"),
                ("1Mo Price Change",
                 f"{selected_row['return_1mo']:+.1f}%" if pd.notna(selected_row['return_1mo']) else "N/A"),
                ("3Mo Price Change",
                 f"{selected_row['return_3mo']:+.1f}%" if pd.notna(selected_row['return_3mo']) else "N/A")
            ]

            # Create metrics HTML
            metrics_html = [
                html.Div([
                    html.Div(metric[0], className='metric-label', style={'fontWeight': 'bold'}),
                    html.Div(metric[1], className='metric-value', style={'fontSize': '1.2em'})
                ], className='metric-item', style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'backgroundColor': '#f9f9f9'
                }) for metric in metrics
            ]

            # Create scatter plot showing historical relationship with filtered data
            valid_data = display_df.dropna(subset=['dev_from_mean', 'return_1mo'])

            scatter_fig = px.scatter(
                display_df,
                x='dev_from_mean',
                y='return_1mo',
                color='signal_strength_decile' if 'signal_strength_decile' in valid_data.columns else None,
                trendline='ols',
                title=f"Storage Position vs Future Price Change (1 Month) - {len(valid_data)} observations",
                labels={
                    'dev_from_mean': 'Deviation from 5-Year Mean',
                    'return_1mo': 'Price Change Next Month (%)',
                    'signal_strength_decile': 'Signal Strength'
                }
            )

            # Add current selection
            if pd.notna(selected_row['return_1mo']):
                scatter_fig.add_trace(go.Scatter(
                    x=[selected_row['dev_from_mean']],
                    y=[selected_row['return_1mo']],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Selected Date'
                ))

            # Analysis summary with filter information
            storage_position = "high" if selected_row['pct_above_min'] > 50 else "low"
            expected_direction = "decrease" if selected_row['pct_above_min'] > 50 else "increase"

            filter_info = ""
            if apply_clicks and not reset_clicks:
                filter_info = f"\n• Analysis based on {len(valid_data)} filtered observations"

            summary_text = f"""
            Based on the analysis for {selected_date.strftime('%Y-%m-%d')}:

            • Storage is at {selected_row['pct_above_min']:.1f}% above the 5-year minimum, indicating a {storage_position} storage position
            • Historical patterns suggest prices typically {expected_direction} when storage is at this level
            • Actual price change was {selected_row['return_1mo']:+.1f}% over the following month{filter_info}
            """

            summary_html = html.Pre(summary_text, style={'whiteSpace': 'pre-wrap'})

            return  storage_fig,withdrawal_fig,  metrics_html,  scatter_fig, summary_html,  reset_month_range,  reset_ssq_values


class StorageAppWSpreads(StorageApp):
    """
    Enhanced StorageApp with flexible filtering capabilities
    """

    def __init__(self, storage_key='storage/total_lower_48', price_key='prices/NG_1', withdrawal_key ='consumption/net_withdrawals'):
        super().__init__()
        self.data.drop(columns=[self.prices_col], inplace=True)
        self.futures_keys = fetch_contract_keys(self.table_client)
        self.__add__(self.futures_keys)
        self.futures_cols = [key_to_name(k) for k in self.futures_keys]
        resample_params = {self.storage_col: 'last', self.prices_col: 'last', self.withdrawal_col:
                                                    'last'}
        resample_params.update({self.futures_cols[i]:'last' for i in range(0, len(self.futures_cols))})
        self.data.ffill(inplace=True)
        self.data = self.data.resample('1W').last()
        # Calculate storage metrics
        self.app_func = self.seasonal_storage_data
        self.data = self.run()
        # Configure filters for this specific application
        self.configure_filters({
            'month': {
                'type': 'range',
                'label': 'Month',
                'min_val': 1,
                'max_val': 12
            },
            'signal_strength_decile': {
                'type': 'range',
                'label': 'Signal Strength Decile',
                'min_val':0,
                'max_val':9
            },
            'signal_series': {
                'type': 'single_select',
                'label': 'Signal Series',
                'source': 'columns',
                'match': {'contains': '1mo'}
            }
        })

    def update_keys(self, storage_key=None, price_key=None):
        keys_to_add = []
        cols_to_drop = []
        if isinstance(self.data.index, pd.Index):
            self.data.index = pd.to_datetime(self.data.index)

        if storage_key:
            if storage_key != self.storage_key:
                cols_to_drop.append(self.storage_col)
                self.storage_key = storage_key
                self.storage_col = key_to_name(storage_key)
                keys_to_add.append(storage_key)

        if price_key:
            if self.prices_key != price_key:
                cols_to_drop.append(self.prices_col)
                self.prices_key = price_key
                self.prices_col = key_to_name(price_key)
                keys_to_add.append(price_key)

        # Preserve the index range before adding new data
        min_idx, max_idx = self.data.index.min(), self.data.index.max()

        if keys_to_add:
            self.__add__(keys_to_add)
            self.data = self.data.ffill().drop(columns=cols_to_drop)


            # Filter Non-datetimes from index
            self.data = self.data[[isinstance(idx, (pd.Timestamp, datetime)) for idx in self.data.index]]
            self.data.index = pd.to_datetime(self.data.index) if isinstance(self.data.index,
                                                                            pd.Index) else self.data.index

            # Resample to weekly and recalculate metrics
            self.data = self.data.resample('1W').apply({
                self.storage_col: 'last',
                self.prices_col: 'last',
                self.withdrawal_col: 'last'
            })

            self.data = self.seasonal_storage_data()

        return self.data

    def seasonal_storage_data(self):
        df = self.data.copy()
        df['date'] = df.index
        df = df.sort_values('date')

        # Add week-of-year, month, and year columns
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Calculate seasonal historical min/max (same week, past 5 years)
        df['hist_max'] = np.nan
        df['hist_min'] = np.nan
        df['hist_mean'] = np.nan

        for idx, row in df.iterrows():
            current_week = row['week_of_year']
            current_year = row['year']

            # Get historical data (same week, past 5 years including current year)
            historical_mask = (
                    (df['week_of_year'] == current_week) &
                    (df['year'] <= current_year) &
                    (df['year'] > current_year - 5) &
                    (df.index <= idx)
            )
            historical_data = df.loc[historical_mask, self.storage_col]

            if len(historical_data) > 0:
                df.loc[idx, 'hist_max'] = historical_data.max()
                df.loc[idx, 'hist_min'] = historical_data.min()
                df.loc[idx, 'hist_mean'] = historical_data.mean()

        # Calculate percentage position within historical range
        df['pct_above_min'] = ((df[self.storage_col] - df['hist_min']) /
                               (df['hist_max'] - df['hist_min'])) * 100

        df['dev_from_mean'] = ((df[self.storage_col] - df['hist_mean']) / df['hist_mean'])

        # Calculate future price change (4 weeks ahead)
        df['price_1mo'] = df[self.prices_col].shift(-4)
        df['price_3mo'] = df[self.prices_col].shift(-12)
        df['return_1mo'] = np.log(df.price_1mo) - np.log(df[self.prices_col])
        df['return_3mo'] = np.log(df.price_3mo) - np.log(df[self.prices_col])
        df['signal_strength_decile'] = pd.qcut(df['dev_from_mean'],10, labels=False)

        df = calc_contract_spreads(df, second_month=False)
        for i in df.columns:
            if 'spread' in i:
                df[f'{i}_return_1mo'] = df[i].shift(-4) - df[i]
                df[f'{i}_return_1wk'] = df[i].shift(-1) - df[i]

        return df

    def generate_layout(self):
        """Generate the enhanced Dash layout with flexible filtering"""
        # Filter dates to exclude last 4 weeks to avoid incomplete forward-looking data
        valid_dates = self.data[self.data['date'] <= self.data['date'].max() - timedelta(weeks=4)]

        if 'date' in valid_dates.columns:
            min_date = valid_dates['date'].min()
            max_date = valid_dates['date'].max()
            default_date = max_date
        else:
            min_date = valid_dates.index.min()
            max_date = valid_dates.index.max()
            default_date = max_date

        # Generate menus using the enhanced base class functionality
        menu_div = self.generate_menu(categories=['prices', 'storage'])

        # Generate the flexible filter menu
        filter_menu = self.generate_filter_menu(
            filter_columns=['month', 'signal_strength_decile'],  # Only these 2
            filter_configs={  # Only include configs for these 2
                'month': self.filter_configs['month'],
                'signal_strength_decile': self.filter_configs['signal_strength_decile']
            },
            include_series_selector=True
        )

        layout = html.Div([
            html.H1("Natural Gas Storage-Price Analysis", style={'textAlign': 'center'}),
            html.Div(children=[
                menu_div,
                filter_menu,
                html.Div(id=f'{self.app_id}-scatter-plot-div', children=[
                    dcc.Graph(id=f'{self.app_id}-scatter-plot')],
                         style={
                             'height': '20%',
                             'width': '50%',
                             'marginTop': '5px'
                         })],
                style={'display': 'flex', 'height': '20%', 'width': '100%'}),

            html.Div(id=f'{self.app_id}-line-plot-div', children=[

                dcc.Tabs(children=[
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-storage-plot')
                    ]),
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-withdrawal-plot')
                    ])
                ]),
            ],
                     style={
                         'height': '60%',
                         'width': '100%',
                         'marginTop': '10px'
                     }),

            html.Div(id=f'{self.app_id}-metrics-container', className='metrics-grid', style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '15px',
                'margin': '20px'
            }),

            html.Div([
                html.H3("Analysis Summary"),
                html.Div(id=f'{self.app_id}-summary')
            ], style={'margin': '20px'})
        ])

        return layout

    def register_callbacks(self, app):
        @callback(
            [Output(f'{self.app_id}-storage-plot', 'figure'),
             Output(f'{self.app_id}-withdrawal-plot', 'figure'),
             Output(f'{self.app_id}-metrics-container', 'children'),
             Output(f'{self.app_id}-scatter-plot', 'figure'),
             Output(f'{self.app_id}-summary', 'children'),
             Output(f'{self.app_id}-month-range-slider', 'value'),
             Output(f'{self.app_id}-signal_strength_decile-range-slider', 'value')],
            [Input(f'{self.app_id}-date-picker', 'date'),
             Input(f'{self.app_id}-apply-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-reset-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-prices-dropdown', 'value'),
             Input(f'{self.app_id}-storage-dropdown', 'value'),
             Input(f'{self.app_id}-month-range-slider', 'value'),
             Input(f'{self.app_id}-signal_strength_decile-range-slider', 'value'),
             Input(f'{self.app_id}-x-dd', 'value'),  # New X dropdown input
             Input(f'{self.app_id}-y-dd', 'value')],  # New Y dropdown input
            prevent_initial_call=False
        )
        def update_analysis(selected_date, apply_clicks, reset_clicks,
                            price_key, storage_key, month_range, ssq_values,
                            x_variable, y_variable):  # New parameters
            # Get callback context to identify trigger
            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            selected_date = pd.to_datetime(selected_date)

            # Handle key updates first (affects both original and filtered data)
            if price_key or storage_key:
                self.update_keys(price_key=price_key, storage_key=storage_key)
                self.original_data = self.data.copy()  # Update original data
                self.filtered_data = pd.DataFrame()  # Reset filtered data

            # Use original data as base for processing
            df = self.original_data.copy().dropna(subset=[self.withdrawal_col, self.prices_col, self.storage_col])

            # Handle reset action
            if trigger_id == f'{self.app_id}-reset-filters-btn':
                self.filtered_data = pd.DataFrame()  # Clear filtered data
                # Get default values for filters
                month_options = self.get_filter_options('month')
                ssq_options = self.get_filter_options('signal_strength_decile')
                reset_month_range = [month_options['min'], month_options['max']]
                reset_ssq_values = [ssq_options['min'], ssq_options['max']]
                summary_text = "Filters reset successfully"

            # Handle apply action
            elif trigger_id == f'{self.app_id}-apply-filters-btn':
                filter_values = {
                    'month_range': month_range,
                    'signal_strength_decile_range': ssq_values  # Fixed key name
                }
                self.filtered_data = self.apply_filters(df, filter_values)
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = f"Applied filters: {len(self.filtered_data)}/{len(df)} records"

            # No filter action - use last state
            else:
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = "Using current data selection"

            # Determine which dataset to use for visualizations
            display_df = self.filtered_data if not self.filtered_data.empty else df

            # Get data for selected date
            if 'date' in df.columns:
                selected_data = df[df['date'] == selected_date]
            else:
                selected_data = df[df.index == selected_date]

            if selected_data.empty:
                if 'date' in df.columns:
                    closest_date = df.loc[(df['date'] - selected_date).abs().idxmin(), 'date']
                    selected_data = df[df['date'] == closest_date]
                else:
                    delta_series = pd.Series(df.index - selected_date)
                    closest_idx = delta_series.abs().idxmin()
                    selected_data = df.iloc[[closest_idx]]

            selected_row = selected_data.iloc[0]

            # Create storage visualization
            storage_fig = go.Figure()

            # Add storage level
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df[self.storage_col],
                name='Current Storage',
                line=dict(color='blue', width=2)
            ))

            # Add 5-year max/min bands
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_max'],
                name='5-Year Max',
                line=dict(color='red', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_min'],
                name='5-Year Min',
                line=dict(color='green', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_mean'],
                name='5-Year Mean',
                line=dict(color='orange', dash='dot')
            ))

            storage_fig.update_layout(
                title="Natural Gas Storage Levels vs Historical Range",
                xaxis_title="Date",
                yaxis_title="Storage Volume (BCF)",
                hovermode='x unified'
            )

            # Create withdrawal figure
            withdrawal_fig = go.Figure(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df[self.withdrawal_col],
                line=dict(color='blue')
            ))

            withdrawal_fig.update_layout(
                title="Net Withdrawals",
                xaxis_title="Date",
                yaxis_title="Net Withdrawals (BCF)"
            )

            # Prepare metrics
            metrics = [
                ("Current Storage", f"{selected_row[self.storage_col]:,.0f} BCF"),
                ("5-Year Max", f"{selected_row['hist_max']:,.0f} BCF"),
                ("5-Year Min", f"{selected_row['hist_min']:,.0f} BCF"),
                ("% Above 5Y Min", f"{selected_row['pct_above_min']:.1f}%"),
                ("Current Price", f"${selected_row[self.prices_col]:,.2f}"),
                ("Price 1Mo Later",
                 f"${selected_row['price_1mo']:,.2f}" if pd.notna(selected_row['price_1mo']) else "N/A"),
                ("1Mo Price Change",
                 f"{selected_row['return_1mo'] * 100:+.1f}%" if pd.notna(selected_row['return_1mo']) else "N/A"),
                ("3Mo Price Change",
                 f"{selected_row['return_3mo'] * 100:+.1f}%" if pd.notna(selected_row['return_3mo']) else "N/A")
            ]

            # Create metrics HTML
            metrics_html = [
                html.Div([
                    html.Div(metric[0], className='metric-label', style={'fontWeight': 'bold'}),
                    html.Div(metric[1], className='metric-value', style={'fontSize': '1.2em'})
                ], className='metric-item', style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'backgroundColor': '#f9f9f9'
                }) for metric in metrics
            ]

            # Create scatter plot using selected X and Y variables
            # Set defaults if not selected
            if not x_variable:
                x_variable = 'total_lower_48_storage' if 'total_lower_48_storage' in display_df.columns else \
                display_df.columns[0]
            if not y_variable:
                y_variable = 'return_1mo' if 'return_1mo' in display_df.columns else display_df.columns[1]

            # Ensure selected variables exist in the dataframe
            if x_variable not in display_df.columns or y_variable not in display_df.columns:
                scatter_fig = go.Figure()
                scatter_fig.add_annotation(
                    text="Selected variables not found in data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                scatter_fig.update_layout(
                    title="Scatter Plot - Invalid Variable Selection",
                    xaxis_title=x_variable,
                    yaxis_title=y_variable
                )
            else:
                # Create scatter plot with filtered data
                valid_data = display_df.dropna(subset=[x_variable, y_variable])

                # Determine color variable (use signal_strength_decile if available)
                color_var = 'signal_strength_decile' if 'signal_strength_decile' in valid_data.columns else None

                scatter_fig = px.scatter(
                    valid_data,
                    x=x_variable,
                    y=y_variable,
                    color=color_var,
                    trendline='ols',
                    title=f"{y_variable} vs {x_variable} - {len(valid_data)} observations",
                    labels={
                        x_variable: x_variable.replace('_', ' ').title(),
                        y_variable: y_variable.replace('_', ' ').title(),
                        'signal_strength_decile': 'Signal Strength'
                    }
                )

                # If y_variable is a return, multiply by 100 for percentage
                if 'return' in y_variable.lower():
                    scatter_fig.update_layout(yaxis=dict(tickformat='.1%'))

                # Add current selection point if both variables are available
                if pd.notna(selected_row.get(x_variable)) and pd.notna(selected_row.get(y_variable)):
                    scatter_fig.add_trace(go.Scatter(
                        x=[selected_row[x_variable]],
                        y=[selected_row[y_variable]],
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='star'),
                        name='Selected Date',
                        showlegend=True
                    ))

            # Analysis summary with dynamic variable information
            storage_position = "high" if selected_row['pct_above_min'] > 50 else "low"
            expected_direction = "decrease" if selected_row['pct_above_min'] > 50 else "increase"

            filter_info = ""
            if not self.filtered_data.empty:
                filter_info = f"\n• Analysis based on {len(self.filtered_data)} filtered observations"

            # Add scatter plot variable info
            scatter_info = f"\n• Scatter plot shows {y_variable} vs {x_variable}"

            summary_text = f"""
            Based on the analysis for {selected_date.strftime('%Y-%m-%d')}:

            • Storage is at {selected_row['pct_above_min']:.1f}% above the 5-year minimum, indicating a {storage_position} storage position
            • Historical patterns suggest prices typically {expected_direction} when storage is at this level
            • Actual price change was {selected_row.get('return_1mo', 0) * 100:+.1f}% over the following month{filter_info}
            {scatter_info}
            """

            summary_html = html.Pre(summary_text, style={'whiteSpace': 'pre-wrap'})

            return (storage_fig, withdrawal_fig, metrics_html, scatter_fig,
                    summary_html, reset_month_range, reset_ssq_values)
class FwdCurveApp(ModelApp):
    """
    Application for analyzing futures forward curves and spreads
    """

    def __init__(self, table_client, keys=None, app_id='futures_forward_curve', enforce_dt=True):
        super().__init__(table_client, keys, app_id, enforce_dt)
        self.futures_pattern = re.compile(r'^[A-Za-z]{2}_\d+$')

        # Store contract keys for reference
        self.contract_keys = keys if isinstance(keys, list) else self._fetch_contract_keys()

        # Calculate forward curves if data is available
        if not self.data.empty and (len(self.data.columns) >= 2):
            self.calculate_forward_curves()
        # Configure filters for the series
        self.configure_series_filters()

    def _fetch_contract_keys(self):
        """Fetch contract keys matching pattern: XX_N (e.g., NG_1, CL_3)"""
        try:
            price_keys = self.table_client.mapping.get('prices', [])

            # Filter and sort matching keys
            contract_keys = []
            for key in price_keys:
                key_name = key.split('/')[-1] if '/' in key else key
                if self.futures_pattern.match(key_name):
                    contract_keys.append(key)

            # Sort by commodity code and number
            contract_keys.sort(key=lambda k: (
                re.match(r'^.*?([A-Za-z]{2})_(\d+)$', k).groups()
                if re.match(r'^.*?([A-Za-z]{2})_(\d+)$', k) else (k, 0)
            ))
            return contract_keys[:4]  # Return first 4 contracts

        except Exception as e:
            print(f"Error fetching contract keys: {e}")
            return []


    def calculate_forward_curves(self):
        """Calculate forward curves (spreads) between futures contracts"""
        if len(self.data.columns) < 2:
            print("Need at least 2 contracts to calculate forward curves")
            return
        pattern = re.compile(r'^[A-Za-z]{2}_\d+$')

        # Get contract columns (assuming they follow pattern ct_1, ct_2, etc.)
        contract_cols = [col for col in self.data.columns if re.match(pattern, col)]

        if len(contract_cols) < 2:
            # If no specific pattern, use first 4 columns
            contract_cols = self.data.columns[:4].tolist()

        # Calculate spreads (forward curves)
        if len(contract_cols) >= 2:
            self.data['spread_1_2'] = self.data[contract_cols[0]] - self.data[contract_cols[1]]
            self.data['spread_1_2_change'] = self.data['spread_1_2'].diff()
            self.data['spread_1_2_log_return'] = np.log(self.data['spread_1_2'] / self.data['spread_1_2'].shift(1))

        if len(contract_cols) >= 3:
            self.data['spread_1_3'] = self.data[contract_cols[0]] - self.data[contract_cols[2]]
            self.data['spread_2_3'] = self.data[contract_cols[1]] - self.data[contract_cols[2]]

        if len(contract_cols) >= 4:
            self.data['spread_1_4'] = self.data[contract_cols[0]] - self.data[contract_cols[3]]
            self.data['spread_2_4'] = self.data[contract_cols[1]] - self.data[contract_cols[3]]
            self.data['spread_3_4'] = self.data[contract_cols[2]] - self.data[contract_cols[3]]


    def configure_series_filters(self):
        """Configure filters for selecting which series to display"""
        # Get all available columns for filtering
        available_series = {
            'contracts': [col for col in self.data.columns if self.futures_pattern.match(col)],
            'spreads': [col for col in self.data.columns if 'spread' in col.lower()],
            'all': self.data.columns.tolist()
        }

        self.filter_configs = {
            'series_selector': {
                'type': 'multi_select',
                'label': 'Select Series to Display',
                'options': available_series['all']
            }
        }

    def create_layout(self):
        """Create the dashboard layout"""
        return html.Div([
            html.H1("Futures Forward Curve Analysis",
                    style={'textAlign': 'center', 'marginBottom': '30px'}),

            # Control panel with filters
            html.Div([
                # Series filter menu
                self.generate_series_filter_menu(series),

                # Additional controls
                html.Div([
                    html.H3("Display Options", style={'marginBottom': '15px'}),

                    dcc.Checklist(
                        id=f'{self.app_id}-display-options',
                        options=[
                            {'label': 'Show Contracts', 'value': 'contracts'},
                            {'label': 'Show Spreads', 'value': 'spreads'},
                            {'label': 'Show Spread Changes', 'value': 'changes'},
                            {'label': 'Separate Charts', 'value': 'separate'}
                        ],
                        value=['spreads', 'changes', 'separate'],
                        style={'marginBottom': '20px'}
                    ),

                    html.Div([
                        html.Label("Chart Height (px):", style={'marginRight': '10px'}),
                        dcc.Input(
                            id=f'{self.app_id}-chart-height',
                            type='number',
                            value=400,
                            min=200,
                            max=800,
                            step=50,
                            style={'width': '100px'}
                        )
                    ])
                ], style={
                    'width': '30%',
                    'padding': '15px',
                    'border': '1px solid #ccc',
                    'borderRadius': '5px',
                    'backgroundColor': '#f8f9fa',
                    'marginLeft': '20px'
                })
            ], style={'display': 'flex', 'marginBottom': '30px'}),

            # Chart container
            html.Div(id=f'{self.app_id}-chart-container', style={'width': '100%'}),

            # Summary statistics
            html.Div(id=f'{self.app_id}-summary-stats', style={'marginTop': '30px'})
        ])

    def generate_series_filter_menu(self):
        """Generate a custom filter menu for series selection"""
        # Get all available series
        all_series = self.data.columns.tolist()

        # Categorize series
        contracts = [s for s in all_series if 'ct_' in s.lower() or 'contract' in s.lower()]
        spreads = [s for s in all_series if 'spread' in s.lower() and 'change' not in s.lower()]
        changes = [s for s in all_series if 'change' in s.lower() or 'return' in s.lower()]

        return html.Div([
            html.H3("Series Selection", style={'marginBottom': '15px'}),

            # Contract series
            html.Div([
                html.H4("Contracts", style={'fontSize': '14px', 'marginBottom': '10px'}),
                dcc.Checklist(
                    id=f'{self.app_id}-contract-series',
                    options=[{'label': s, 'value': s} for s in contracts],
                    value=[],  # Start with none selected
                    style={'maxHeight': '150px', 'overflowY': 'auto'}
                )
            ], style={'marginBottom': '15px'}),

            # Spread series
            html.Div([
                html.H4("Spreads", style={'fontSize': '14px', 'marginBottom': '10px'}),
                dcc.Checklist(
                    id=f'{self.app_id}-spread-series',
                    options=[{'label': s, 'value': s} for s in spreads],
                    value=['spread_1_2'] if 'spread_1_2' in spreads else spreads[:1],
                    style={'maxHeight': '150px', 'overflowY': 'auto'}
                )
            ], style={'marginBottom': '15px'}),

            # Change series
            html.Div([
                html.H4("Changes/Returns", style={'fontSize': '14px', 'marginBottom': '10px'}),
                dcc.Checklist(
                    id=f'{self.app_id}-change-series',
                    options=[{'label': s, 'value': s} for s in changes],
                    value=['spread_1_2_change'] if 'spread_1_2_change' in changes else changes[:1],
                    style={'maxHeight': '150px', 'overflowY': 'auto'}
                )
            ])
        ], style={
            'width': '35%',
            'padding': '15px',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'backgroundColor': '#f8f9fa'
        })

    def create_callbacks(self, app):
        """Create Dash callbacks for interactivity"""

        @app.callback(
            [Output(f'{self.app_id}-chart-container', 'children'),
             Output(f'{self.app_id}-summary-stats', 'children')],
            [Input(f'{self.app_id}-contract-series', 'value'),
             Input(f'{self.app_id}-spread-series', 'value'),
             Input(f'{self.app_id}-change-series', 'value'),
             Input(f'{self.app_id}-display-options', 'value'),
             Input(f'{self.app_id}-chart-height', 'value')]
        )
        def update_charts(contract_series, spread_series, change_series, display_options, chart_height):
            # Combine all selected series
            selected_series = (contract_series or []) + (spread_series or []) + (change_series or [])

            if not selected_series:
                return html.Div("Please select at least one series to display"), ""

            # Filter data to selected series
            plot_data = self.data[selected_series].dropna()

            # Create charts based on display options
            if 'separate' in display_options:
                # Create separate charts for spreads and changes
                charts = []

                # Main chart for contracts and spreads
                main_series = [s for s in selected_series if 'change' not in s.lower() and 'return' not in s.lower()]
                if main_series:
                    fig_main = go.Figure()
                    for series in main_series:
                        fig_main.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data[series],
                            mode='lines',
                            name=series,
                            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
                        ))

                    fig_main.update_layout(
                        title="Futures Contracts and Spreads",
                        xaxis_title="Date",
                        yaxis_title="Price/Spread",
                        height=chart_height,
                        hovermode='x unified'
                    )

                    charts.append(dcc.Graph(figure=fig_main))

                # Change chart
                change_series_list = [s for s in selected_series if 'change' in s.lower() or 'return' in s.lower()]
                if change_series_list:
                    fig_change = go.Figure()
                    for series in change_series_list:
                        fig_change.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data[series],
                            mode='lines',
                            name=series,
                            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
                        ))

                    fig_change.update_layout(
                        title="Spread Changes and Returns",
                        xaxis_title="Date",
                        yaxis_title="Change/Return",
                        height=chart_height,
                        hovermode='x unified'
                    )

                    charts.append(dcc.Graph(figure=fig_change))

                chart_div = html.Div(charts)

            else:
                # Single chart with all series
                fig = go.Figure()
                for series in selected_series:
                    fig.add_trace(go.Scatter(
                        x=plot_data.index,
                        y=plot_data[series],
                        mode='lines',
                        name=series,
                        hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
                    ))

                fig.update_layout(
                    title="Futures Forward Curve Analysis",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=chart_height,
                    hovermode='x unified'
                )

                chart_div = dcc.Graph(figure=fig)

            # Calculate summary statistics
            stats_data = []
            for series in selected_series:
                if series in plot_data.columns:
                    stats_data.append({
                        'Series': series,
                        'Mean': plot_data[series].mean(),
                        'Std Dev': plot_data[series].std(),
                        'Min': plot_data[series].min(),
                        'Max': plot_data[series].max(),
                        'Latest': plot_data[series].iloc[-1]
                    })

            stats_df = pd.DataFrame(stats_data)

            stats_table = html.Div([
                html.H3("Summary Statistics", style={'marginBottom': '15px'}),
                html.Table([
                    html.Thead([
                        html.Tr([html.Th(col) for col in stats_df.columns])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(stats_df.iloc[i][col] if col == 'Series'
                                    else f"{stats_df.iloc[i][col]:.4f}")
                            for col in stats_df.columns
                        ]) for i in range(len(stats_df))
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
            ])

            return chart_div, stats_table
