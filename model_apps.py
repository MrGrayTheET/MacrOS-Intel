from dotenv import load_dotenv

load_dotenv('../.env')
from data.data_tables import TableClient, NASSTable, FASTable
import pandas as pd
import numpy as np
from datetime import datetime
from dash import dcc, html, Input, Output
import re
from utils.data_tools import key_to_name, generate_layout_keys
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
            filter_type = config.get('key_type', 'range')

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

            filter_type = config.get('key_type', 'range' if pd.api.types.is_numeric_dtype(data[col]) else 'categorical')

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
                'key_type': 'multi_select',
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
                self.generate_series_filter_menu(),

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
