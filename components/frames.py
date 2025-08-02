from components.plotting.chart_components import FundamentalChart, MarketChart, COTPlotter as COTPlot
from assets.plotting.cot_plotter import COTPlotter as COTPlot
from dotenv import load_dotenv
import dash
from dash import dcc, html, dash_table, Input, Output, State
from sources.data_tables import MarketTable
import pandas as pd
from typing import List, Dict, Any

load_dotenv('C:\\Users\\nicho\PycharmProjects\macrOS-Int\.env')


class FundamentalFrame:
    """
    A class for generating organized div layouts for Dash applications.

    Manages FundamentalChart instances and tables with preset layouts, sizing, and data selection menus.
    Integrates with TableClient for data retrieval from HDF5 files.
    - Horizontal: charts stacked vertically with x-axes parallel
    - Vertical: charts arranged side by side with y-axes parallel
    """

    def __init__(self,
                 table_client,
                 chart_configs: List[Dict[str, Any]] = None,
                 tables: List[Dict[str, Any]] = None,
                 layout: str = "vertical",
                 div_prefix: str = "fundamental_frame",
                 width: str = "100%",
                 height: str = "600px",
                 data_options: Dict[str, Dict[str, str]] = None,
                 enable_settings: bool = True):
        """
        Initialize FundamentalFrame.

        Args:
            table_client: TableClient instance for data retrieval
            chart_configs: List of chart configuration dictionaries
            tables: List of table data dictionaries for dash_table.DataTable
            layout: Layout orientation ("vertical" or "horizontal")
            div_prefix: Prefix for component IDs to ensure uniqueness
            width: Width of the frame container
            height: Height of the frame container
            data_options: Dictionary with data selection options for menu
                         Format: {'Category': {'display_name': 'hdf_key'}}
            enable_settings: Whether to show chart settings panels and enable interactivity
        """
        self.table_client = table_client
        self.chart_configs = chart_configs or []
        self.charts = []
        self.tables = tables or []
        self.layout = layout.lower()
        self.div_prefix = div_prefix
        self.width = width
        self.height = height
        self.enable_settings = enable_settings

        # Default data options structure
        self.data_options = self.generate_data_options()
        if self.layout not in ["vertical", "horizontal"]:
            raise ValueError("Layout must be 'vertical' or 'horizontal'")
        # Initialize charts from configurations
        self._initialize_charts()

    def _initialize_charts(self):
        """Initialize FundamentalChart instances from configurations."""
        for i, config in enumerate(self.chart_configs):
            chart_id = f"{self.div_prefix}_chart_{i}"
            config['data'] = self.table_client[config.get('starting_key', None)]
            # Create FundamentalChart instance
            chart = FundamentalChart(
                chart_id=chart_id,
                config=config
            )

            self.charts.append(chart)

    def generate_data_options(self):
        keys = [k[1:].rsplit('/') for k in self.table_client.available_keys()]
        data_options = {}

        for k in keys:
            data_options.update({k[0]: {k[1]: f'{k[0]}/{k[1]}'}}) if k[0] not in data_options.keys() else data_options[
                k[0]].update({
                k[1]:
                    f'{k[0]}/{k[1]}'
            })

        self.data_options = data_options

        return self.data_options

    def add_chart_config(self, config: Dict[str, Any]):
        """Add a chart configuration and initialize the chart."""
        self.chart_configs.append(config)
        chart_id = f"{self.div_prefix}_chart_{len(self.charts)}"
        config['data'] = self.table_client[config.get('starting_key', None)]

        chart = FundamentalChart(
            chart_id=chart_id,
            config=config
        )

        self.charts.append(chart)

    def add_table(self, table_data: Dict[str, Any]):
        """Add a table to the frame."""
        self.tables.append(table_data)

    def set_layout(self, layout: str):
        """Set the layout orientation."""
        layout = layout.lower()
        if layout not in ["vertical", "horizontal"]:
            raise ValueError("Layout must be 'vertical' or 'horizontal'")
        self.layout = layout

    def set_size(self, width: str, height: str):
        """Set the frame dimensions."""
        self.width = width
        self.height = height

    def enable_chart_settings(self, enable: bool = True):
        """
        Enable or disable chart settings panels.

        Args:
            enable: Whether to enable settings panels
        """
        self.enable_settings = enable

    def disable_chart_settings(self):
        """Disable chart settings panels."""
        self.enable_settings = False

    def get_available_hdf_keys(self) -> List[str]:
        """Get available HDF5 keys from the table client."""
        try:
            with pd.HDFStore(self.table_client.table_db, mode='r') as store:
                return list(store.keys())
        except Exception as e:
            print(f"Error accessing HDF5 file: {e}")
            return []

    def get_columns_for_data_source(self, hdf_key: str) -> List[str]:
        """
        Get available columns for a specific data source.

        Args:
            hdf_key: The HDF5 key to get data and extract columns from

        Returns:
            List of column names available in the data source
        """
        try:
            data = self.table_client[hdf_key]
            if hasattr(data, 'columns'):
                return list(data.columns)
            elif hasattr(data, 'keys'):
                return list(data.keys())
            else:
                return []
        except Exception as e:
            print(f"Error getting columns for {hdf_key}: {e}")
            return []

    def generate_chart_settings_panel(self, chart_index: int) -> html.Div:
        """
        Generate the combined settings panel for a specific chart.
        Updated to include column selection.

        Args:
            chart_index: Index of the chart to generate settings for

        Returns:
            html.Div containing the combined settings panel
        """
        chart_id = f"{self.div_prefix}_chart_{chart_index}"

        # Color options for charts
        color_options = [
            {'label': 'Blue', 'value': '#1f77b4'},
            {'label': 'Green', 'value': '#2ca02c'},
            {'label': 'Red', 'value': '#d62728'},
            {'label': 'Orange', 'value': '#ff7f0e'},
            {'label': 'Purple', 'value': '#9467bd'},
            {'label': 'Brown', 'value': '#8c564b'},
            {'label': 'Pink', 'value': '#e377c2'},
            {'label': 'Gray', 'value': '#7f7f7f'},
            {'label': 'Olive', 'value': '#bcbd22'},
            {'label': 'Cyan', 'value': '#17becf'}
        ]

        # Chart key_type options
        chart_type_options = [
            {'label': 'Line Chart', 'value': 'line'},
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Area Chart', 'value': 'area'}
        ]

        # Create data source tabs
        data_source_tabs = []
        for category, options in self.data_options.items():
            dropdown_options = [
                {'label': display_name, 'value': hdf_key}
                for display_name, hdf_key in options.items()
            ]

            tab_content = html.Div([
                html.Label(f"Select {category}:",
                           style={'margin-bottom': '5px', 'font-weight': 'bold'}),
                dcc.Dropdown(
                    id=f"{chart_id}_select_{category.lower()}",
                    options=dropdown_options,
                    value=list(options.values())[0] if options else None,
                    style={'margin': '5px 0'}
                ),

                # Column selection dropdown
                html.Label("Select Column:",
                           style={'margin': '10px 0 5px 0', 'font-weight': 'bold'}),
                dcc.Dropdown(
                    id=f"{chart_id}_select_column_{category.lower()}",
                    options=[],  # Will be populated by callback
                    value=None,
                    placeholder="Select a column...",
                    style={'margin': '5px 0'}
                )
            ])

            data_source_tabs.append(
                dcc.Tab(
                    label=category,
                    value=category.lower(),
                    children=tab_content,
                    style={'padding': '10px'}
                )
            )

        settings_panel = html.Div([
            html.H5("Chart Settings",
                    style={'margin': '10px 0', 'font-size': '14px', 'color': '#333'}),

            # Chart Properties Section
            html.Div([
                html.H6("Chart Properties",
                        style={'margin-bottom': '10px', 'color': '#444'}),

                # Date Range
                html.Div([
                    html.Label("Date Range:",
                               style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.DatePickerRange(
                        id=f"{chart_id}_date_range",
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ], style={'margin-bottom': '10px'}),

                # Chart Type and Color Row
                html.Div([
                    html.Div([
                        html.Label("Chart Type:",
                                   style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id=f"{chart_id}_chart_type",
                            options=chart_type_options,
                            value=self.charts[chart_index].chart_type if chart_index < len(self.charts) else 'bar',
                            style={'width': '100%'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'margin-right': '4%'}),

                    html.Div([
                        html.Label("Chart Color:",
                                   style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id=f"{chart_id}_chart_color",
                            options=color_options,
                            value=self.charts[chart_index].line_color if chart_index < len(self.charts) else '#1f77b4',
                            style={'width': '100%'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'})
                ], style={'margin-bottom': '15px'})
            ], style={'border-bottom': '1px solid #eee', 'padding-bottom': '15px', 'margin-bottom': '15px'}),

            # Data Source Section
            html.Div([
                html.H6("Data Source",
                        style={'margin-bottom': '10px', 'color': '#444'}),
                dcc.Tabs(
                    id=f"{chart_id}_data_tabs",
                    value=list(self.data_options.keys())[0].lower(),
                    children=data_source_tabs,
                    style={'margin-bottom': '15px'}
                )
            ], style={'border-bottom': '1px solid #eee', 'padding-bottom': '15px', 'margin-bottom': '15px'}),

            # Update Button
            html.Div([
                html.Button(
                    "Update Chart",
                    id=f"{chart_id}_update_button",
                    style={
                        'background-color': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px',
                        'border-radius': '5px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%'
                    }
                )
            ], style={'text-align': 'center'})
        ],
            style={
                'padding': '15px',
                'background-color': '#f8f9fa',
                'border-radius': '5px',
                'height': '100%',
                'overflow-y': 'auto'
            })

        return settings_panel

    def generate_chart_divs(self) -> List[html.Div]:
        """
        Generate individual divs for each FundamentalChart with optional tabs for chart and settings.

        Returns:
            List of html.Div components containing chart elements with optional tabs
        """
        chart_divs = []

        # Calculate chart dimensions based on layout
        if self.layout == "horizontal":
            chart_width = "100%"
            chart_height = f"{(int(self.height.replace('px', '')) // max(len(self.charts), 1)) - 5}px"
        else:  # vertical
            chart_width = f"{(100 // max(len(self.charts), 1)) - 2}%"
            chart_height = self.height

        for i, chart in enumerate(self.charts):
            # Update chart dimensions
            chart.width = chart_width

            # Adjust height based on whether settings are enabled
            height_adjustment = 150 if self.enable_settings else 80  # Less adjustment when no settings tab
            chart.height = int(chart_height.replace('px', '')) - height_adjustment

            # Create chart tab content
            chart_tab_content = html.Div([
                dcc.Graph(
                    id=chart.chart_id,
                    figure=chart.get_chart_figure(),
                    style={
                        'width': '100%',
                        'height': f"{chart.height}px"
                    }
                )
            ])

            # Create tabs or simple container based on settings enablement
            if self.enable_settings:
                # Create settings tab content
                settings_tab_content = self.generate_chart_settings_panel(i)

                # Create tabs for chart and settings
                chart_content = dcc.Tabs(
                    id=f"{chart.chart_id}_tabs",
                    value="chart",
                    children=[
                        dcc.Tab(
                            label="Chart",
                            value="chart",
                            children=chart_tab_content,
                            style={'padding': '10px'}
                        ),
                        dcc.Tab(
                            label="Settings",
                            value="settings",
                            children=settings_tab_content,
                            style={'padding': '10px'}
                        )
                    ],
                    style={'height': '40px'}
                )
            else:
                # Just show the chart without tabs
                chart_content = chart_tab_content

            # Create div for the chart
            chart_div = html.Div([
                html.Div([
                    html.H5(chart.title, style={'margin': '0 0 5px 0', 'color': '#333'}),
                    html.P(f"ID: {chart.chart_id}",
                           style={'margin': '0', 'font-size': '11px', 'color': '#666', 'font-family': 'monospace'})
                ], style={'text-align': 'center', 'margin-bottom': '10px'}),

                chart_content
            ],
                id=f"{self.div_prefix}_chart_container_{i}",
                className="chart-container",
                style={
                    'width': chart_width,
                    'height': chart_height,
                    'padding': '10px',
                    'border': '2px solid #34495e',
                    'border-radius': '5px',
                    'margin': '5px',
                    'box-sizing': 'border-box',
                    'background-color': '#ffffff'
                })

            chart_divs.append(chart_div)

        return chart_divs

    def generate_table_divs(self) -> List[html.Div]:
        """
        Generate individual divs for each table with proper sizing.

        Returns:
            List of html.Div components containing dash_table.DataTable elements
        """
        table_divs = []

        if not self.tables:
            return table_divs

        # Calculate table dimensions based on layout
        if self.layout == "horizontal":
            table_width = "100%"
            table_height = f"{int(self.height.replace('px', '')) // max(len(self.tables), 1)}px"
        else:  # vertical
            table_width = f"{100 // max(len(self.tables), 1)}%"
            table_height = self.height

        for i, table_data in enumerate(self.tables):
            # Create unique ID for each table
            table_id = f"{self.div_prefix}_table_{i}"

            # Create div for the table
            table_div = html.Div([
                dash_table.DataTable(
                    id=table_id,
                    style_table={
                        'height': f"{int(table_height.replace('px', '')) - 50}px",
                        'overflowY': 'auto'
                    },
                    **table_data
                )
            ],
                id=f"{self.div_prefix}_table_container_{i}",
                className="table-container",
                style={
                    'width': table_width,
                    'height': table_height,
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'border-radius': '5px',
                    'margin': '5px',
                    'box-sizing': 'border-box'
                })

            table_divs.append(table_div)

        return table_divs

    def generate_layout_div(self) -> html.Div:
        """
        Generate the main layout div containing all components.

        Returns:
            html.Div with complete layout (no longer includes top-level settings)
        """
        # Generate content divs
        chart_divs = self.generate_chart_divs()
        table_divs = self.generate_table_divs()

        # Combine all components
        all_components = chart_divs + table_divs

        # Apply layout-specific styling for content area
        if self.layout == "horizontal":
            # Horizontal layout: components stacked vertically
            content_style = {
                'display': 'flex',
                'flex-direction': 'column',
                'width': '100%',
                'height': self.height,
                'gap': '10px'
            }
        else:  # vertical layout
            # Vertical layout: components side by side
            content_style = {
                'display': 'flex',
                'flex-direction': 'row',
                'width': '100%',
                'height': self.height,
                'gap': '10px',
                'flex-wrap': 'wrap'
            }

        # Create content container
        content_container = html.Div(
            children=all_components,
            id=f"{self.div_prefix}_content_container",
            style=content_style
        )

        # Create main container (simplified without top control row)
        main_div = html.Div([
            content_container
        ],
            id=f"{self.div_prefix}_container",
            className="fundamental-frame",
            style={
                'width': self.width,
                'padding': '20px',
                'border': '2px solid #333',
                'border-radius': '10px',
                'background-color': '#ffffff',
                'box-sizing': 'border-box'
            })

        return main_div

    def get_component_ids(self) -> Dict[str, Any]:
        """
        Get all component IDs for callback registration.
        Updated to include column selection IDs and conditional settings components.

        Returns:
            Dictionary containing all component IDs with prefix
        """
        chart_ids = {}
        for i, chart in enumerate(self.charts):
            chart_id = chart.chart_id
            chart_components = {
                'chart_id': chart_id,
            }

            # Only include settings-related IDs if settings are enabled
            if self.enable_settings:
                chart_components.update({
                    'tabs': f"{chart_id}_tabs",
                    'date_range': f"{chart_id}_date_range",
                    'chart_type': f"{chart_id}_chart_type",
                    'chart_color': f"{chart_id}_chart_color",
                    'data_tabs': f"{chart_id}_data_tabs",
                    'update_button': f"{chart_id}_update_button",
                    'select_dropdowns': {
                        category.lower(): f"{chart_id}_select_{category.lower()}"
                        for category in self.data_options.keys()
                    },
                    'column_dropdowns': {
                        category.lower(): f"{chart_id}_select_column_{category.lower()}"
                        for category in self.data_options.keys()
                    }
                })

            chart_ids[f'chart_{i}'] = chart_components

        return {
            'container': f"{self.div_prefix}_container",
            'charts': chart_ids,
            'tables': [f"{self.div_prefix}_table_{i}" for i in range(len(self.tables))],
            'chart_containers': [f"{self.div_prefix}_chart_container_{i}" for i in range(len(self.charts))],
            'table_containers': [f"{self.div_prefix}_table_container_{i}" for i in range(len(self.tables))]
        }

    def register_callbacks(self, app):
        """
        Register callbacks for each individual chart.
        Only registers callbacks if settings are enabled.

        Args:
            app: Dash application instance
        """
        # Only register callbacks if settings are enabled
        if not self.enable_settings:
            return

        component_ids = self.get_component_ids()

        # Register callbacks for each chart
        for chart_index, (chart_key, chart_components) in enumerate(component_ids['charts'].items()):
            chart_id = chart_components['chart_id']

            # Skip if this chart doesn't have settings components
            if 'select_dropdowns' not in chart_components:
                continue

            # Register column update callbacks for each data source category
            for category in self.data_options.keys():
                category_lower = category.lower()
                data_dropdown_id = chart_components['select_dropdowns'][category_lower]
                column_dropdown_id = chart_components['column_dropdowns'][category_lower]

                @app.callback(
                    Output(column_dropdown_id, 'options'),
                    Output(column_dropdown_id, 'value'),
                    [Input(data_dropdown_id, 'value')],
                    prevent_initial_call=True
                )
                def update_column_options(selected_hdf_key, category=category):
                    if not selected_hdf_key:
                        return [], None

                    columns = self.get_columns_for_data_source(selected_hdf_key)
                    column_options = [{'label': col, 'value': col} for col in columns]
                    default_value = columns[0] if columns else None

                    return column_options, default_value

            # Main chart update callback
            @app.callback(
                Output(chart_id, 'figure'),
                [Input(chart_components['update_button'], 'n_clicks')],
                [State(chart_components['chart_type'], 'value'),
                 State(chart_components['chart_color'], 'value'),
                 State(chart_components['data_tabs'], 'value'),
                 State(chart_components['date_range'], 'start_date'),
                 State(chart_components['date_range'], 'end_date')] +
                [State(dropdown_id, 'value') for dropdown_id in chart_components['select_dropdowns'].values()] +
                [State(dropdown_id, 'value') for dropdown_id in chart_components['column_dropdowns'].values()],
                prevent_initial_call=True
            )
            def update_chart_callback(n_clicks, chart_type, color, active_tab, start_date, end_date, *all_values):
                if n_clicks is None:
                    return dash.no_update

                # Find the chart by iterating through the charts
                target_chart = None
                for chart in self.charts:
                    if chart.chart_id == chart_id:
                        target_chart = chart
                        break

                if not target_chart:
                    return dash.no_update

                # Split the values into data source selections and column selections
                num_categories = len(self.data_options.keys())
                data_values = all_values[:num_categories]
                column_values = all_values[num_categories:]

                # Map dropdown values to categories
                data_dropdown_ids = list(chart_components['select_dropdowns'].keys())
                column_dropdown_ids = list(chart_components['column_dropdowns'].keys())

                selected_data_values = dict(zip(data_dropdown_ids, data_values))
                selected_column_values = dict(zip(column_dropdown_ids, column_values))

                # Update chart properties
                if chart_type:
                    target_chart.chart_type = chart_type
                if color:
                    target_chart.line_color = color

                # Update data source if active tab has a selection
                if active_tab in selected_data_values and selected_data_values[active_tab]:
                    new_hdf_key = selected_data_values[active_tab]
                    selected_column = selected_column_values.get(active_tab)

                    # Get data from table client
                    data = self.table_client[new_hdf_key]

                    # Pass both data and selected column to the chart
                    if hasattr(target_chart, 'update_data_source'):
                        if selected_column:
                            target_chart.update_data_source(data, selected_column)
                        else:
                            target_chart.update_data_source(data)

                # Apply date filtering if dates are provided
                if start_date and end_date:
                    if hasattr(target_chart, 'set_date_range'):
                        target_chart.set_date_range(start_date, end_date)

                return target_chart.get_chart_figure()

class MarketFrame:
    """Professional market data visualization framework."""

    def __init__(self, market_table: MarketTable, chart_configs: List[Dict[str, Any]] = None,
                 layout: str = "vertical", div_prefix: str = "market_frame",
                 width: str = "100%", height: str = "600px"):
        self.market_table = market_table
        self.chart_configs = chart_configs or []
        self.charts = []
        self.layout = layout.lower()
        self.div_prefix = div_prefix
        self.width = width
        self.height = height
        self.frame_df = pd.DataFrame()

        self.market_data_options = {
            'Price Data': {'OHLC Chart': 'candlestick', 'Close Price': 'close', 'Line Chart': 'line'},
            'Timeframe': {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'},
            'COT Data': {
                'Traditional COT (Net)': 'cot_traditional_net',
                'Disaggregated COT (Net)': 'cot_disaggregated_net',
                'Commercial Net': 'cot_commercial_net',
                'Non-Commercial Net': 'cot_noncommercial_net',
                'Money Manager Net': 'cot_money_manager_net'
            }
        }

        if self.layout not in ["vertical", "horizontal"]:
            raise ValueError("Layout must be 'vertical' or 'horizontal'")

        self._initialize_charts()

    def _initialize_charts(self):
        """Initialize charts from configurations."""
        for i, config in enumerate(self.chart_configs):
            chart_id = f"{self.div_prefix}_chart_{i}"

            chart = MarketChart(
                chart_id=chart_id,
                config=config
            )

            chart.load_ticker_data()

            self.charts.append(chart)

    def generate_chart_settings_panel(self, chart_index: int) -> html.Div:
        """Generate settings panel for chart."""
        chart_id = f"{self.div_prefix}_chart_{chart_index}"

        color_options = [{'label': name.title(), 'value': color} for name, color in
                         [('blue', '#1f77b4'), ('green', '#2ca02c'), ('red', '#d62728'),
                          ('orange', '#ff7f0e'), ('purple', '#9467bd')]]

        chart_type_options = [{'label': 'Line Chart', 'value': 'line'},
                              {'label': 'Candlestick', 'value': 'candlestick'}]

        data_source_tabs = []
        for category, options in self.market_data_options.items():
            dropdown_options = [{'label': k, 'value': v} for k, v in options.items()]

            tab_content = html.Div([
                html.Label(f"Select {category}:", style={'margin-bottom': '5px', 'font-weight': 'bold'}),
                dcc.Dropdown(id=f"{chart_id}_select_{category.lower().replace(' ', '_')}",
                             options=dropdown_options, value=list(options.values())[0],
                             style={'margin': '5px 0'})
            ])

            data_source_tabs.append(dcc.Tab(label=category, value=category.lower().replace(' ', '_'),
                                            children=tab_content, style={'padding': '10px'}))

        return html.Div([
            html.H5("Chart Settings", style={'margin': '10px 0', 'font-size': '14px', 'color': '#333'}),

            # Chart Info
            html.Div([
                html.H6("Chart Information", style={'margin-bottom': '10px', 'color': '#444'}),
                html.P(f"Ticker: {self.charts[chart_index].ticker if chart_index < len(self.charts) else 'N/A'}",
                       style={'margin': '0', 'padding': '8px', 'background-color': '#f5f5f5', 'border-radius': '4px'})
            ], style={'border-bottom': '1px solid #eee', 'padding-bottom': '15px', 'margin-bottom': '15px'}),

            # Chart Properties
            html.Div([
                html.H6("Chart Properties", style={'margin-bottom': '10px', 'color': '#444'}),
                dcc.DatePickerRange(id=f"{chart_id}_date_range", display_format='YYYY-MM-DD',
                                    style={'width': '100%', 'margin-bottom': '10px'}),
                html.Div([
                    html.Div([dcc.Dropdown(id=f"{chart_id}_chart_type", options=chart_type_options,
                                           value=self.charts[chart_index].chart_type if chart_index < len(
                                               self.charts) else 'line')],
                             style={'width': '48%', 'display': 'inline-block', 'margin-right': '4%'}),
                    html.Div([dcc.Dropdown(id=f"{chart_id}_chart_color", options=color_options,
                                           value=self.charts[chart_index].line_color if chart_index < len(
                                               self.charts) else '#1f77b4')],
                             style={'width': '48%', 'display': 'inline-block'})
                ])
            ], style={'border-bottom': '1px solid #eee', 'padding-bottom': '15px', 'margin-bottom': '15px'}),

            # Data Source
            html.Div([
                html.H6("Data Source", style={'margin-bottom': '10px', 'color': '#444'}),
                dcc.Tabs(id=f"{chart_id}_data_tabs", value='price_data', children=data_source_tabs)
            ], style={'border-bottom': '1px solid #eee', 'padding-bottom': '15px', 'margin-bottom': '15px'}),

            # Update Button
            html.Button("Update Chart", id=f"{chart_id}_update_button",
                        style={'background-color': '#2196F3', 'color': 'white', 'border': 'none',
                               'padding': '10px 20px', 'border-radius': '5px', 'width': '100%'})
        ], style={'padding': '15px', 'background-color': '#f0f8ff', 'border-radius': '5px',
                  'height': '100%', 'overflow-y': 'auto'})

    def generate_chart_divs(self) -> List[html.Div]:
        """Generate chart divs with tabs."""
        chart_divs = []

        # Calculate dimensions
        if self.layout == "horizontal":
            chart_width = "100%"
            chart_height = f"{int(self.height.replace('px', '')) // max(len(self.charts), 1)}px"
        else:
            chart_width = f"{100 // max(len(self.charts), 1)}%"
            chart_height = self.height

        for i, chart in enumerate(self.charts):
            chart.height = int(chart_height.replace('px', '')) - 150

            # Chart and settings tabs
            chart_tabs = dcc.Tabs(id=f"{chart.chart_id}_tabs", value="chart", children=[
                dcc.Tab(label="Chart", value="chart", children=[
                    dcc.Graph(id=chart.chart_id, figure=chart.get_chart_figure(),
                              style={'width': '100%', 'height': f"{chart.height}px"})
                ]),
                dcc.Tab(label="Settings", value="settings", children=[
                    self.generate_chart_settings_panel(i)
                ])
            ])

            chart_div = html.Div([
                html.Div([
                    html.H5(chart.title, style={'margin': '0 0 5px 0', 'color': '#333'}),
                    html.P(f"ID: {chart.chart_id} | Ticker: {chart.ticker or 'None'}",
                           style={'margin': '0', 'font-size': '11px', 'color': '#666'})
                ], style={'text-align': 'center', 'margin-bottom': '10px'}),
                chart_tabs
            ], id=f"{self.div_prefix}_chart_container_{i}",
                style={'width': chart_width, 'height': chart_height, 'padding': '10px',
                       'border': '2px solid #2196F3', 'border-radius': '5px', 'margin': '5px',
                       'background-color': '#ffffff'})

            chart_divs.append(chart_div)

        return chart_divs

    def generate_layout_div(self) -> html.Div:
        """Generate main layout."""
        chart_divs = self.generate_chart_divs()

        content_style = {
            'display': 'flex',
            'flex-direction': 'column' if self.layout == "horizontal" else 'row',
            'width': '100%', 'height': self.height, 'gap': '10px'
        }

        return html.Div([
            html.Div(children=chart_divs, style=content_style)
        ], style={'width': self.width, 'padding': '20px', 'border': '2px solid #2196F3',
                  'border-radius': '10px', 'background-color': '#ffffff'})

    def get_component_ids(self) -> Dict[str, Any]:
        """Get component IDs for callbacks."""
        chart_ids = {}
        for i, chart in enumerate(self.charts):
            chart_id = chart.chart_id
            chart_ids[f'chart_{i}'] = {
                'chart_id': chart_id,
                'update_button': f"{chart_id}_update_button",
                'chart_type': f"{chart_id}_chart_type",
                'chart_color': f"{chart_id}_chart_color",
                'date_range': f"{chart_id}_date_range",
                'data_tabs': f"{chart_id}_data_tabs",
                'select_dropdowns': {
                    category.lower().replace(' ', '_'): f"{chart_id}_select_{category.lower().replace(' ', '_')}"
                    for category in self.market_data_options.keys()
                }
            }

        return {'charts': chart_ids}

    def register_callbacks(self, app):
        """Register callbacks for charts."""
        component_ids = self.get_component_ids()

        for chart_index, (_, chart_components) in enumerate(component_ids['charts'].items()):
            chart_id = chart_components['chart_id']
            target_chart = self.charts[chart_index]

            @app.callback(
                Output(chart_id, 'figure'),
                [Input(chart_components['update_button'], 'n_clicks')],
                [State(chart_components['chart_type'], 'value'),
                 State(chart_components['chart_color'], 'value'),
                 State(chart_components['data_tabs'], 'value'),
                 State(chart_components['date_range'], 'start_date'),
                 State(chart_components['date_range'], 'end_date')] +
                [State(dropdown_id, 'value') for dropdown_id in chart_components['select_dropdowns'].values()],
                prevent_initial_call=True
            )
            def update_chart_callback(n_clicks, chart_type, color, active_tab, start_date, end_date, *dropdown_values):
                if n_clicks is None:
                    return dash.no_update

                # Update chart properties
                if chart_type:
                    target_chart.chart_type = chart_type
                if color:
                    target_chart.line_color = color

                # Handle data source selection
                dropdown_ids = list(chart_components['select_dropdowns'].keys())
                selected_values = dict(zip(dropdown_ids, dropdown_values))

                if active_tab in selected_values and selected_values[active_tab]:
                    data_type = selected_values[active_tab]

                    # Handle chart key_type changes
                    if data_type in ['candlestick', 'line']:
                        target_chart.chart_type = data_type

                    # Handle timeframe changes
                    elif data_type in ['D', 'W', 'M', 'Q', 'Y']:
                        target_chart.set_timeframe_config(resample_freq=data_type)

                    # Handle COT data
                    elif data_type.startswith('cot_') and target_chart.ticker:
                        cot_data = self.market_table.get_cot(target_chart.ticker)
                        if cot_data is not None and not cot_data.empty:
                            cot_plotter = COTPlot(cot_data)
                            target_chart.indicators = [ind for ind in target_chart.indicators
                                                       if not ind.get('name', '').startswith('COT')]

                            cot_mapping = {
                                'cot_traditional_net': (['producer_merchant_net', 'non_commercial_net'],
                                                        [cot_plotter.colors['commercials'],
                                                         cot_plotter.colors['non_commercials']]),
                                'cot_disaggregated_net': (
                                    ['producer_merchant_net', 'swap_dealer_net', 'money_manager_net'],
                                    [cot_plotter.colors['commercials'], cot_plotter.colors['swap_dealers'],
                                     cot_plotter.colors['money_managers']]),
                                'cot_commercial_net': (['producer_merchant_net'], [cot_plotter.colors['commercials']]),
                                'cot_noncommercial_net': (
                                    ['non_commercial_net'], [cot_plotter.colors['non_commercials']]),
                                'cot_money_manager_net': (['money_manager_net'], [cot_plotter.colors['money_managers']])
                            }

                            if data_type in cot_mapping:
                                columns, colors = cot_mapping[data_type]
                                available_columns = [col for col in columns if col in cot_plotter.df.columns]
                                if available_columns:
                                    target_chart.plot_indicator({
                                        'key_type': 'line', 'axis': 'separate', 'data': cot_plotter.df,
                                        'y_columns': available_columns,
                                        'name': f'COT {data_type.replace("cot_", "").title()}',
                                        'y_title': 'Net Positions', 'colors': colors[:len(available_columns)]
                                    })

                # Apply date filtering
                if start_date or end_date:
                    target_chart.set_timeframe_config(
                        resample_freq=target_chart.interval,
                        start_date=start_date, end_date=end_date
                    )

                return target_chart.get_chart_figure()

    def load_cot_data_for_chart(self, chart_index: int, commodity: str = None, cot_type: str = 'traditional_net'):
        """Load COT data for a specific chart."""
        if 0 <= chart_index < len(self.charts):
            chart = self.charts[chart_index]
            commodity = commodity or chart.ticker

            if commodity:
                cot_data = self.market_table.get_cot(commodity)
                if cot_data is not None and not cot_data.empty:
                    cot_plotter = COTPlot(cot_data)
                    chart.indicators = [ind for ind in chart.indicators if not ind.get('name', '').startswith('COT')]

                    if cot_type == 'traditional_net' and 'producer_merchant_net' in cot_plotter.df.columns:
                        chart.plot_indicator({
                            'key_type': 'line', 'axis': 'separate', 'data': cot_plotter.df,
                            'y_columns': ['producer_merchant_net', 'non_commercial_net'],
                            'name': 'COT Traditional Net', 'y_title': 'Net Positions',
                            'colors': [cot_plotter.colors['commercials'], cot_plotter.colors['non_commercials']]
                        })

                    return chart.get_chart_figure()
        return None
