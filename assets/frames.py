import dash
import pandas as pd
from typing import List, Union, Dict, Any, Optional, Tuple
from assets.plotting.chart_components import  FundamentalChart
from dash import dcc, html, dash_table, callback, Input, Output, State


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
                 data_options: Dict[str, Dict[str, str]] = None):
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
        """
        self.table_client = table_client
        self.chart_configs = chart_configs or []
        self.charts = []
        self.tables = tables or []
        self.layout = layout.lower()
        self.div_prefix = div_prefix
        self.width = width
        self.height = height

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

            # Create FundamentalChart instance
            chart = FundamentalChart(
                chart_id=chart_id,
                TableClient=self.table_client,
                starting_key=config.get('starting_key'),
                title=config.get('title', f"Chart {i + 1}"),
                x_column=config.get('x_column', 'Date'),
                chart_type=config.get('chart_type', 'bar'),
                width='100%',
                height=config.get('height', 300),
                line_color=config.get('line_color', '#1f77b4')
            )

            self.charts.append(chart)

    def generate_data_options(self):
        keys = [k[1:].rsplit('/') for k in self.table_client.available_keys()]
        data_options = {}

        for k in keys:
            data_options.update({k[0]:{k[1]:f'{k[1]}/{k[0]}'}}) if k[0] not in data_options.keys() else data_options[k[0]].update({
                k[1]:
                    f'{k[0]}/{k[1]}'
            })

        self.data_options = data_options

        return self.data_options

    def add_chart_config(self, config: Dict[str, Any]):
        """Add a chart configuration and initialize the chart."""
        self.chart_configs.append(config)
        chart_id = f"{self.div_prefix}_chart_{len(self.charts)}"

        chart = FundamentalChart(
            chart_id=chart_id,
            TableClient=self.table_client,
            starting_key=config.get('starting_key'),
            title=config.get('title', f"Chart {len(self.charts) + 1}"),
            x_column=config.get('x_column', 'Date'),
            chart_type=config.get('chart_type', 'bar'),
            width='100%',
            height=config.get('height', 300),
            line_color=config.get('line_color', '#1f77b4')
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

    def get_available_hdf_keys(self) -> List[str]:
        """Get available HDF5 keys from the table client."""
        try:
            with pd.HDFStore(self.table_client.table_db, mode='r') as store:
                return list(store.keys())
        except Exception as e:
            print(f"Error accessing HDF5 file: {e}")
            return []

    def generate_chart_settings_panel(self, chart_index: int) -> html.Div:
        """
        Generate the combined settings panel for a specific chart.

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

        # Chart type options
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
        Generate individual divs for each FundamentalChart with tabs for chart and settings.

        Returns:
            List of html.Div components containing chart elements with tabs
        """
        chart_divs = []

        # Calculate chart dimensions based on layout
        if self.layout == "horizontal":
            chart_width = "100%"
            chart_height = f"{int(self.height.replace('px', '')) // max(len(self.charts), 1)}px"
        else:  # vertical
            chart_width = f"{100 // max(len(self.charts), 1)}%"
            chart_height = self.height

        for i, chart in enumerate(self.charts):
            # Update chart dimensions
            chart.width = "100%"
            chart.height = int(chart_height.replace('px', '')) - 150  # Account for tabs and padding

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

            # Create settings tab content
            settings_tab_content = self.generate_chart_settings_panel(i)

            # Create tabs for chart and settings
            chart_tabs = dcc.Tabs(
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

            # Create div for the chart with tabs
            chart_div = html.Div([
                html.Div([
                    html.H5(chart.title, style={'margin': '0 0 5px 0', 'color': '#333'}),
                    html.P(f"ID: {chart.chart_id}",
                           style={'margin': '0', 'font-size': '11px', 'color': '#666', 'font-family': 'monospace'})
                ], style={'text-align': 'center', 'margin-bottom': '10px'}),

                chart_tabs
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

        Returns:
            Dictionary containing all component IDs with prefix
        """
        chart_ids = {}
        for i, chart in enumerate(self.charts):
            chart_id = chart.chart_id
            chart_ids[f'chart_{i}'] = {
                'chart_id': chart_id,
                'tabs': f"{chart_id}_tabs",
                'date_range': f"{chart_id}_date_range",
                'chart_type': f"{chart_id}_chart_type",
                'chart_color': f"{chart_id}_chart_color",
                'data_tabs': f"{chart_id}_data_tabs",
                'update_button': f"{chart_id}_update_button",
                'select_dropdowns': {
                    category.lower(): f"{chart_id}_select_{category.lower()}"
                    for category in self.data_options.keys()
                }
            }

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

        Args:
            app: Dash application instance
        """
        component_ids = self.get_component_ids()

        # Register a callback for each chart individually
        for chart_index, (chart_key, chart_components) in enumerate(component_ids['charts'].items()):
            chart_id = chart_components['chart_id']

            # Create callback for this specific chart
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

                # Find the chart by iterating through the charts
                target_chart = None
                for chart in self.charts:
                    if chart.chart_id == chart_id:
                        target_chart = chart
                        break

                if not target_chart:
                    return dash.no_update

                # Map dropdown values to categories
                dropdown_ids = list(chart_components['select_dropdowns'].keys())
                selected_values = dict(zip(dropdown_ids, dropdown_values))

                # Update chart properties
                if chart_type:
                    target_chart.chart_type = chart_type
                if color:
                    target_chart.line_color = color

                # Update data source if active tab has a selection
                if active_tab in selected_values and selected_values[active_tab]:
                    new_hdf_key = selected_values[active_tab]
                    target_chart.hdf_key = new_hdf_key

                    # Reload data when key changes
                    if hasattr(target_chart, '_load_data'):
                        target_chart._load_data()


                # Apply date filtering if dates are provided
                if start_date and end_date:
                    # This would need to be implemented in your FundamentalChart class
                    if hasattr(target_chart, 'set_date_range'):
                        target_chart.set_date_range(start_date, end_date)

                return target_chart.get_chart_figure()

class MarketFrame:

    def __init__(self,market_table,starting_keys= None,chart_configs=None, div_height="50%", div_width="65%", ):
        self.table_client = market_table
        self.height =div_height
        self.width = div_width

        return


# Placeholder for FundamentalChart class (would be imported from your module)


# Example usage
if __name__ == "__main__":
    # Mock TableClient for example
    class MockTableClient:
        def __init__(self):
            self.table_db = "mock_data.h5"
            self.prefix = "energy"

        def get_key(self, key, use_prefix=True):
            # Mock data return
            import pandas as pd
            return pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=12, freq='M'),
                'Value': [100, 110, 105, 120, 115, 130, 125, 140, 135, 150, 145, 160]
            })


    # Create mock table client
    table_client = MockTableClient()

    # Define chart configurations
    chart_configs = [
        {
            'starting_key': 'supply/production',
            'title': 'Production Data',
            'chart_type': 'bar',
            'y_column': 'Value',
            'height': 300
        },
        {
            'starting_key': 'demand/consumption',
            'title': 'Consumption Data',
            'chart_type': 'line',
            'y_column': 'Value',
            'height': 300
        }
    ]

    # Create data options for menu
    data_options = {
        'Supply': {
            'Production': 'supply/production',
            'Imports': 'supply/imports',
            'Inventory': 'supply/inventory'
        },
        'Demand': {
            'Consumption': 'demand/consumption',
            'Exports': 'demand/exports',
            'Industrial Use': 'demand/industrial'
        },
        'Producers': {
            'Domestic': 'producers/domestic',
            'International': 'producers/international'
        }
    }

    # Create FundamentalFrame instance
    frame = FundamentalFrame(
        table_client=table_client,
        chart_configs=chart_configs,
        layout="horizontal",
        div_prefix="energy_dashboard",
        width="1200px",
        height="800px",
        data_options=data_options
    )

    # Generate layout
    layout = frame.generate_layout_div()

    # Get component IDs for callbacks
    component_ids = frame.get_component_ids()
    print("Component IDs:", component_ids)