import sys

from dash.dcc import Tabs

from components.chart_components import FundamentalChart, MarketChart, COTPlotter as COTPlot
from dotenv import load_dotenv
import dash
from dash import dcc, html, dash_table, Input, Output, State
from data.data_tables import MarketTable, TableClient, FASTable
import pandas as pd
from typing import List, Dict, Any, Union, Optional



class FundamentalFrame:
    """
    Simplified FundamentalFrame focusing only on chart display.
    Configuration handled by FrameGrid.
    """

    def __init__(self,
                 table_client,
                 chart_configs: List[Dict[str, Any]] = None,
                 tables: List[Dict[str, Any]] = None,
                 layout: str = "vertical",
                 div_prefix: str = "fundamental_frame",
                 width: str = "100%",
                 height: str = "600px"):

        self.table_client = table_client
        self.chart_configs = chart_configs or []
        self.charts = []
        self.tables = tables or []
        self.layout = layout.lower()
        self.div_prefix = div_prefix
        self.width = width
        self.height = height

        if self.layout not in ["vertical", "horizontal"]:
            raise ValueError("Layout must be 'vertical' or 'horizontal'")

        self._initialize_charts()

    def _initialize_charts(self):
        """Initialize FundamentalChart instances from configurations."""
        for i, config in enumerate(self.chart_configs):
            chart_id = f"{self.div_prefix}_chart_{i}"
            config['data'] = self.table_client[config.get('starting_key', None)]

            chart = FundamentalChart(
                chart_id=chart_id,
                config=config
            )
            self.charts.append(chart)

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

    def generate_chart_divs(self) -> List[html.Div]:
        """Generate individual divs for each FundamentalChart."""
        chart_divs = []

        if self.layout == "horizontal":
            chart_width = "100%"
            chart_height = f"{(int(self.height.replace('px', '')) // max(len(self.charts), 1)) - 5}px"
        else:
            chart_width = f"{(100 // max(len(self.charts), 1)) - 2}%"
            chart_height = self.height

        for i, chart in enumerate(self.charts):
            chart.width = chart_width
            chart.height = int(chart_height.replace('px', '')) - 80

            chart_content = dcc.Graph(
                id=chart.chart_id,
                figure=chart.get_chart_figure(),
                style={'width': '100%', 'height': f"{chart.height}px"}
            )

            chart_div = html.Div([
                html.Div([
                    html.H5(chart.title, style={'margin': '0 0 5px 0', 'color': '#333'}),
                    html.P(f"ID: {chart.chart_id}",
                           style={'margin': '0', 'font-size': '11px', 'color': '#666'})
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
        """Generate individual divs for each table."""
        table_divs = []

        if not self.tables:
            return table_divs

        if self.layout == "horizontal":
            table_width = "100%"
            table_height = f"{int(self.height.replace('px', '')) // max(len(self.tables), 1)}px"
        else:
            table_width = f"{100 // max(len(self.tables), 1)}%"
            table_height = self.height

        for i, table_data in enumerate(self.tables):
            table_id = f"{self.div_prefix}_table_{i}"

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
        """Generate the main layout div containing all components."""
        chart_divs = self.generate_chart_divs()
        table_divs = self.generate_table_divs()
        all_components = chart_divs + table_divs

        if self.layout == "horizontal":
            content_style = {
                'display': 'flex',
                'flex-direction': 'column',
                'width': '100%',
                'height': self.height,
                'gap': '10px'
            }
        else:
            content_style = {
                'display': 'flex',
                'flex-direction': 'row',
                'width': '100%',
                'height': self.height,
                'gap': '10px',
                'flex-wrap': 'wrap'
            }

        main_div = html.Div([
            html.Div(
                children=all_components,
                id=f"{self.div_prefix}_content_container",
                style=content_style
            )
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
        """Get all component IDs for external use."""
        return {
            'container': f"{self.div_prefix}_container",
            'charts': {f'chart_{i}': chart.chart_id for i, chart in enumerate(self.charts)},
            'tables': [f"{self.div_prefix}_table_{i}" for i in range(len(self.tables))],
            'chart_containers': [f"{self.div_prefix}_chart_container_{i}" for i in range(len(self.charts))],
            'table_containers': [f"{self.div_prefix}_table_container_{i}" for i in range(len(self.tables))]
        }

class MarketFrame:
    """Simplified MarketFrame focusing only on chart display."""

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

    def generate_chart_divs(self) -> List[html.Div]:
        """Generate chart divs."""
        chart_divs = []

        if self.layout == "horizontal":
            chart_width = "100%"
            chart_height = f"{int(self.height.replace('px', '')) // max(len(self.charts), 1)}px"
        else:
            chart_width = f"{100 // max(len(self.charts), 1)}%"
            chart_height = self.height

        for i, chart in enumerate(self.charts):
            chart.height = int(chart_height.replace('px', '')) - 80

            chart_content = dcc.Graph(
                id=chart.chart_id,
                figure=chart.get_chart_figure(),
                style={'width': '100%', 'height': f"{chart.height}px"}
            )

            chart_div = html.Div([
                html.Div([
                    html.H5(chart.title, style={'margin': '0 0 5px 0', 'color': '#333'}),
                    html.P(f"ID: {chart.chart_id} | Ticker: {chart.ticker or 'None'}",
                           style={'margin': '0', 'font-size': '11px', 'color': '#666'})
                ], style={'text-align': 'center', 'margin-bottom': '10px'}),
                chart_content
            ],
                id=f"{self.div_prefix}_chart_container_{i}",
                style={
                    'width': chart_width,
                    'height': chart_height,
                    'padding': '10px',
                    'border': '2px solid #2196F3',
                    'border-radius': '5px',
                    'margin': '5px',
                    'background-color': '#ffffff'
                })

            chart_divs.append(chart_div)

        return chart_divs

    def generate_layout_div(self) -> html.Div:
        """Generate main layout."""
        chart_divs = self.generate_chart_divs()

        content_style = {
            'display': 'flex',
            'flex-direction': 'column' if self.layout == "horizontal" else 'row',
            'width': '100%',
            'height': self.height,
            'gap': '10px'
        }

        return html.Div([
            html.Div(children=chart_divs, style=content_style)
        ],
            style={
                'width': self.width,
                'padding': '20px',
                'border': '2px solid #2196F3',
                'border-radius': '10px',
                'background-color': '#ffffff'
            })

    def get_component_ids(self) -> Dict[str, Any]:
        """Get component IDs."""
        return {
            'charts': {f'chart_{i}': chart.chart_id for i, chart in enumerate(self.charts)}
        }

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
                            'key_type': 'line',
                            'axis': 'separate',
                            'data': cot_plotter.df,
                            'y_columns': ['producer_merchant_net', 'non_commercial_net'],
                            'name': 'COT Traditional Net',
                            'y_title': 'Net Positions',
                            'colors': [cot_plotter.colors['commercials'], cot_plotter.colors['non_commercials']]
                        })

                    return chart.get_chart_figure()
        return None


import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dash import html, dcc, Input, Output, State, dash, no_update


class FrameGrid:
    """Layout manager for FundamentalFrame and MarketFrame instances with configurable menu."""

    def __init__(self,
                 frames: List[Union['FundamentalFrame', 'MarketFrame']], config: Optional[Dict[str, Any]] = None,
                 style_config: Optional[Dict[str, Any]] = None,
                 menu_config: Optional[Dict[str, Any]] = None, container_id: str = "frame-grid-container"):

        self.frames = frames
        self.container_id = container_id

        # Default configurations
        default_grid_config = {
            'layout_type': 'auto',
            'rows': None,
            'cols': None,
            'gap': '20px',
            'responsive': True,
            'breakpoints': {
                'sm': {'cols': 1},
                'md': {'cols': 2},
                'lg': {'cols': 3},
                'xl': {'cols': 4}
            },
            'frame_positions': {}

        }

        default_style_config = {
            'container_width': '100%',
            'container_height': 'auto',
            'padding': '20px',
            'background_color': '#0f0f0f',
            'frame_background': '#1a1a2e',
            'border_color': '#333',
            'border_radius': '10px',
            'box_shadow': '0 4px 6px rgba(0, 0, 0, 0.3)',
            'title_color': '#fff',
            'frame_min_height': '400px',
            'responsive_padding': {
                'sm': '10px',
                'md': '15px',
                'lg': '20px'
            }

        }

        default_menu_config = {
            'enabled': True,
            'position': 'top',
            'size': {'width': '100%', 'height': '160px'},
            'alterable_frames': None,
            'data_sources': None,
            'categories': ['storage', 'prices', 'production'],
            'show_frame_selector': True,
            'show_apply_button': True,
            'menu_title': 'Data Control Panel',
            'compact_mode': False,
            'background_color': '#1a1a2e',
            'text_color': '#fff',
            'filter_criteria': None,
            'selection_mode': 'keys',
            'show_column_selector': True
        }

        self.grid_config = {**default_grid_config, **(grid_config or {})}
        self.style_config = {**default_style_config, **(style_config or {})}
        self.menu_config = {**default_menu_config, **(menu_config or {})}

        self._initialize_menu_data()

        if self.grid_config['layout_type'] == 'auto':
            self._calculate_auto_grid()

        return

    def get_menu_component_ids(self) -> Dict[str, str]:
        """Get all menu-related component IDs for callback registration."""
        menu_ids = {}

        if not self.menu_config['enabled']:
            return menu_ids

        menu_ids['apply_button'] = f"{self.container_id}_menu_apply_button"

        if self.menu_config['show_frame_selector']:
            menu_ids['frame_selector'] = f"{self.container_id}_menu_frame_selector"

        first_frame_idx = self.menu_config['alterable_frames'][0]
        frame_data_sources = self.menu_config['data_sources'].get(first_frame_idx, {})

        for category in frame_data_sources.keys():
            menu_ids[f'{category}_dropdown'] = f"{self.container_id}_menu_{category}_dropdown"
            if (self.menu_config['show_column_selector'] and
                    self.menu_config['selection_mode'] in ['columns', 'both']):
                menu_ids[f'{category}_column_dropdown'] = f"{self.container_id}_menu_{category}_column_dropdown"

        if len(frame_data_sources) > 1:
            menu_ids['data_tabs'] = f"{self.container_id}_menu_data_tabs"

        return menu_ids

    def register_menu_callbacks(self, app):
        """Register callbacks for menu interactions with keys/columns support."""
        if not self.menu_config['enabled']:
            return

        menu_ids = self.get_menu_component_ids()
        alterable_frames = self.menu_config['alterable_frames']

        if not menu_ids.get('apply_button'):
            return

        if (self.menu_config['show_column_selector'] and
                self.menu_config['selection_mode'] in ['columns', 'both']):
            self._register_column_update_callbacks(app, menu_ids)

        callback_inputs = [Input(menu_ids['apply_button'], 'n_clicks')]
        callback_states = []

        if 'frame_selector' in menu_ids:
            callback_states.append(State(menu_ids['frame_selector'], 'value'))

        first_frame_idx = alterable_frames[0]
        frame_data_sources = self.menu_config['data_sources'].get(first_frame_idx, {})

        for category in frame_data_sources.keys():
            dropdown_id = menu_ids[f'{category}_dropdown']
            callback_states.append(State(dropdown_id, 'value'))

            if f'{category}_column_dropdown' in menu_ids:
                callback_states.append(State(menu_ids[f'{category}_column_dropdown'], 'value'))

        if 'data_tabs' in menu_ids:
            callback_states.append(State(menu_ids['data_tabs'], 'value'))

        callback_outputs = []
        for frame_idx in alterable_frames:
            if (frame_idx < len(self.frames) and
                    hasattr(self.frames[frame_idx], 'charts') and
                    self.frames[frame_idx].charts):
                first_chart = self.frames[frame_idx].charts[0]
                callback_outputs.append(Output(first_chart.chart_id, 'figure'))

        if not callback_outputs:
            return

        @app.callback(
            callback_outputs,
            callback_inputs,
            callback_states,
            prevent_initial_call=True
        )
        def update_frames_from_menu(n_clicks, *state_values):
            if n_clicks is None or n_clicks == 0:
                return [no_update] * len(callback_outputs)

            state_idx = 0

            if 'frame_selector' in menu_ids:
                selected_frame_idx = state_values[state_idx] if state_values[state_idx] is not None else \
                alterable_frames[0]
                state_idx += 1
            else:
                selected_frame_idx = alterable_frames[0]

            selection_data = {}
            for category in frame_data_sources.keys():
                selection_data[category] = {
                    'main_value': state_values[state_idx],
                    'column_value': None
                }
                state_idx += 1

                if f'{category}_column_dropdown' in menu_ids:
                    selection_data[category]['column_value'] = state_values[state_idx]
                    state_idx += 1

            if 'data_tabs' in menu_ids:
                active_tab = state_values[state_idx]
            else:
                active_tab = list(frame_data_sources.keys())[0].lower()

            if selected_frame_idx in alterable_frames and active_tab in selection_data:
                return self._update_frame_with_selection(
                    selected_frame_idx, active_tab, selection_data[active_tab], alterable_frames
                )

            return [no_update] * len(callback_outputs)

    def _register_column_update_callbacks(self, app, menu_ids):
        """Register callbacks to update column dropdowns based on main selection."""
        first_frame_idx = self.menu_config['alterable_frames'][0]
        frame_data_sources = self.menu_config['data_sources'].get(first_frame_idx, {})

        for category in frame_data_sources.keys():
            if f'{category}_column_dropdown' in menu_ids:
                @app.callback(
                    Output(menu_ids[f'{category}_column_dropdown'], 'options'),
                    [Input(menu_ids[f'{category}_dropdown'], 'value')],
                    prevent_initial_call=True
                )
                def update_column_options(selected_value, category=category):
                    if not selected_value:
                        return []

                    options_data = frame_data_sources[category]
                    selected_option = None

                    for display_name, option_data in options_data.items():
                        if isinstance(option_data, dict):
                            if option_data.get('value') == selected_value:
                                selected_option = option_data
                                break
                        elif option_data == selected_value:
                            selected_option = {'type': 'key', 'value': selected_value}
                            break

                    if selected_option and selected_option.get('type') == 'key':
                        columns = selected_option.get('columns', [])
                        return [{'label': col, 'value': col} for col in columns]

                    return []

    def _update_frame_with_selection(self, frame_idx, active_category, selection_data, alterable_frames):
        """Update frame with selected data (key or column)."""
        selected_value = selection_data['main_value']
        selected_column = selection_data['column_value']

        if not selected_value or frame_idx >= len(self.frames):
            return [no_update] * len(alterable_frames)

        target_frame = self.frames[frame_idx]
        frame_data_sources = self.menu_config['data_sources'].get(frame_idx, {})
        category_options = frame_data_sources.get(active_category, {})

        selected_option_data = None
        for display_name, option_data in category_options.items():
            if isinstance(option_data, dict):
                if option_data.get('value') == selected_value:
                    selected_option_data = option_data
                    break
            elif option_data == selected_value:
                selected_option_data = {'type': 'key', 'value': selected_value}
                break

        if not selected_option_data:
            return [no_update] * len(alterable_frames)

        selection_type = selected_option_data.get('type', 'key')

        if selection_type == 'key':
            if hasattr(target_frame, 'table_client'):
                new_data = target_frame.table_client[selected_value]

                if hasattr(target_frame, 'charts') and target_frame.charts:
                    target_chart = target_frame.charts[0]
                    if hasattr(target_chart, 'update_data_source'):
                        target_chart.update_data_source(new_data, selected_column)

                        outputs = []
                        for alt_frame_idx in alterable_frames:
                            if alt_frame_idx == frame_idx:
                                outputs.append(target_chart.get_chart_figure())
                            else:
                                outputs.append(no_update)
                        return outputs

        elif selection_type == 'column':
            chart_index = selected_option_data.get('chart_index', 0)
            if (hasattr(target_frame, 'charts') and
                    chart_index < len(target_frame.charts)):
                target_chart = target_frame.charts[chart_index]

                if hasattr(target_chart, 'change_y_column'):
                    target_chart.change_y_column(selected_value)

                    outputs = []
                    for alt_frame_idx in alterable_frames:
                        if alt_frame_idx == frame_idx:
                            outputs.append(target_chart.get_chart_figure())
                        else:
                            outputs.append(no_update)
                    return outputs

        return [no_update] * len(alterable_frames)

    def register_all_callbacks(self, app):
        """Register callbacks for all frames in the grid plus menu callbacks."""
        for frame in self.frames:
            if hasattr(frame, 'register_callbacks'):
                frame.register_callbacks(app)

        self.register_menu_callbacks(app)

    def get_all_component_ids(self) -> Dict[str, Any]:
        """Get all component IDs from all frames for callback management."""
        all_ids = {
            'container': self.container_id,
            'frames': {},
            'menu': self.get_menu_component_ids()
        }

        for i, frame in enumerate(self.frames):
            frame_key = f'frame_{i}'
            if hasattr(frame, 'get_component_ids'):
                all_ids['frames'][frame_key] = frame.get_component_ids()
            else:
                all_ids['frames'][frame_key] = {'frame_id': f'{self.container_id}-frame-{i}'}

        return all_ids

    def update_frame(self, frame_index: int, new_frame):
        """Update a specific frame in the grid."""
        if 0 <= frame_index < len(self.frames):
            self.frames[frame_index] = new_frame
            self._initialize_menu_data()
        else:
            raise IndexError(f"Frame index {frame_index} out of range")

    def add_frame(self, frame, position: Optional[int] = None):
        """Add a new frame to the grid."""
        if position is None:
            self.frames.append(frame)
        else:
            self.frames.insert(position, frame)

        if self.grid_config['layout_type'] == 'auto':
            self._calculate_auto_grid()
        self._initialize_menu_data()

    def remove_frame(self, frame_index: int):
        """Remove a frame from the grid."""
        if 0 <= frame_index < len(self.frames):
            self.frames.pop(frame_index)

            if self.grid_config['layout_type'] == 'auto':
                self._calculate_auto_grid()
            self._initialize_menu_data()
        else:
            raise IndexError(f"Frame index {frame_index} out of range")

        self.frames = frames
        self.container_id = container_id

        # Default configurations
        default_grid_config = {
            'layout_type': 'auto',
            'rows': None,
            'cols': None,
            'gap': '20px',
            'responsive': True,
            'breakpoints': {
                'sm': {'cols': 1},
                'md': {'cols': 2},
                'lg': {'cols': 3},
                'xl': {'cols': 4}
            },
            'frame_positions': {}
        }

        default_style_config = {
            'container_width': '100%',
            'container_height': 'auto',
            'padding': '20px',
            'background_color': '#0f0f0f',
            'frame_background': '#1a1a2e',
            'border_color': '#333',
            'border_radius': '10px',
            'box_shadow': '0 4px 6px rgba(0, 0, 0, 0.3)',
            'title_color': '#fff',
            'frame_min_height': '400px',
            'responsive_padding': {
                'sm': '10px',
                'md': '15px',
                'lg': '20px'
            }
        }

        default_menu_config = {
            'enabled': True,
            'position': 'top',
            'size': {'width': '100%', 'height': '160px'},
            'alterable_frames': None,
            'data_sources': None,
            'categories': ['storage', 'prices', 'production'],
            'show_frame_selector': True,
            'show_apply_button': True,
            'menu_title': 'Data Control Panel',
            'compact_mode': False,
            'background_color': '#1a1a2e',
            'text_color': '#fff',
            'filter_criteria': None,
            'selection_mode': 'keys',
            'show_column_selector': True
        }

        self.grid_config = {**default_grid_config, **(grid_config or {})}
        self.style_config = {**default_style_config, **(style_config or {})}
        self.menu_config = {**default_menu_config, **(menu_config or {})}

        self._initialize_menu_data()

        if self.grid_config['layout_type'] == 'auto':
            self._calculate_auto_grid()

    def _initialize_menu_data(self):
        """Initialize menu data sources from frames with optional filtering."""
        if not self.menu_config['enabled']:
            return

        if self.menu_config['alterable_frames'] is None:
            self.menu_config['alterable_frames'] = list(range(len(self.frames)))

        if self.menu_config['data_sources'] is None:
            self.menu_config['data_sources'] = {}

            for i, frame in enumerate(self.frames):
                if hasattr(frame, 'table_client') and hasattr(frame.table_client, 'available_keys'):
                    frame_data = {}

                    if self.menu_config['selection_mode'] in ['keys', 'both']:
                        keys = frame.table_client.available_keys()

                        for category in self.menu_config['categories']:
                            category_keys = [k for k in keys if k.startswith(f'/{category}/')]

                            if self.menu_config.get('filter_criteria'):
                                category_keys = self._filter_keys_by_criteria(category_keys, frame, category)

                            if category_keys:
                                frame_data[category] = {}
                                for key in category_keys:
                                    display_name = key.split('/')[-1].replace('_', ' ').title()
                                    frame_data[category][display_name] = {
                                        'type': 'key',
                                        'value': key[1:],
                                        'columns': self._get_columns_for_key(key[1:], frame)
                                    }

                    if self.menu_config['selection_mode'] in ['columns', 'both']:
                        self._add_column_based_sources(frame_data, frame, i)

                    if frame_data:
                        self.menu_config['data_sources'][i] = frame_data

    def _get_columns_for_key(self, key: str, frame) -> List[str]:
        """Get available columns for a specific key."""
        try:
            data = frame.table_client[key]
            return list(data.columns) if hasattr(data, 'columns') else []
        except:
            return []

    def _add_column_based_sources(self, frame_data: Dict, frame, frame_idx: int):
        """Add column-based data sources to frame_data."""
        if hasattr(frame, 'charts') and frame.charts:
            for chart_idx, chart in enumerate(frame.charts):
                if hasattr(chart, 'data') and chart.data is not None:
                    data = chart.data
                    if hasattr(data, 'columns'):
                        category_name = f'chart_{chart_idx}_columns'
                        if category_name not in frame_data:
                            frame_data[category_name] = {}

                        for col in data.columns:
                            if pd.api.types.is_numeric_dtype(data[col]):
                                display_name = col.replace('_', ' ').title()
                                frame_data[category_name][display_name] = {
                                    'type': 'column',
                                    'value': col,
                                    'chart_index': chart_idx,
                                    'data_shape': data.shape
                                }

    def _filter_keys_by_criteria(self, keys: List[str], frame, category: str) -> List[str]:
        """Filter keys based on boolean criteria."""
        filter_criteria = self.menu_config['filter_criteria']
        if not filter_criteria:
            return keys

        filtered_keys = []

        for key in keys:
            key_name = key.split('/')[-1]
            should_include = True

            # Check global criteria
            if 'global' in filter_criteria:
                should_include = self._evaluate_criteria(key_name, filter_criteria['global'])

            # Check category-specific criteria
            if should_include and category in filter_criteria:
                should_include = self._evaluate_criteria(key_name, filter_criteria[category])

            # Check frame-specific criteria
            frame_idx = self.frames.index(frame)
            if should_include and f'frame_{frame_idx}' in filter_criteria:
                should_include = self._evaluate_criteria(key_name, filter_criteria[f'frame_{frame_idx}'])

            # Check data-based criteria
            if should_include and 'data_checks' in filter_criteria:
                should_include = self._evaluate_data_criteria(key, frame, filter_criteria['data_checks'])

            if should_include:
                filtered_keys.append(key)

        return filtered_keys

    def _evaluate_criteria(self, key_name: str, criteria: Dict[str, Any]) -> bool:
        """Evaluate string-based criteria for a key name."""
        # String pattern matching
        if 'startswith' in criteria:
            patterns = criteria['startswith'] if isinstance(criteria['startswith'], list) else [criteria['startswith']]
            if not any(key_name.startswith(pattern) for pattern in patterns):
                return False

        if 'endswith' in criteria:
            patterns = criteria['endswith'] if isinstance(criteria['endswith'], list) else [criteria['endswith']]
            if not any(key_name.endswith(pattern) for pattern in patterns):
                return False

        if 'contains' in criteria:
            patterns = criteria['contains'] if isinstance(criteria['contains'], list) else [criteria['contains']]
            if not any(pattern in key_name for pattern in patterns):
                return False

        if 'not_contains' in criteria:
            patterns = criteria['not_contains'] if isinstance(criteria['not_contains'], list) else [
                criteria['not_contains']]
            if any(pattern in key_name for pattern in patterns):
                return False

        if 'regex' in criteria:
            import re
            if not re.search(criteria['regex'], key_name):
                return False

        # Length criteria
        if 'min_length' in criteria and len(key_name) < criteria['min_length']:
            return False
        if 'max_length' in criteria and len(key_name) > criteria['max_length']:
            return False

        # Include/exclude lists
        if 'include_only' in criteria:
            include_list = criteria['include_only'] if isinstance(criteria['include_only'], list) else [
                criteria['include_only']]
            if key_name not in include_list:
                return False
        if 'exclude' in criteria:
            exclude_list = criteria['exclude'] if isinstance(criteria['exclude'], list) else [criteria['exclude']]
            if key_name in exclude_list:
                return False

        return True

    def _evaluate_data_criteria(self, key: str, frame, criteria: Dict[str, Any]) -> bool:
        """Evaluate data-based criteria by loading and checking the actual data."""
        try:
            data = frame.table_client[key[1:]]

            if data is None or (hasattr(data, 'empty') and data.empty):
                return criteria.get('allow_empty', True)

            # Check data type
            if 'data_type' in criteria:
                expected_type = criteria['data_type']
                if expected_type == 'DataFrame' and not isinstance(data, pd.DataFrame):
                    return False
                elif expected_type == 'Series' and not isinstance(data, pd.Series):
                    return False

            # Check data shape
            if hasattr(data, 'shape'):
                if 'min_rows' in criteria and data.shape[0] < criteria['min_rows']:
                    return False
                if 'max_rows' in criteria and data.shape[0] > criteria['max_rows']:
                    return False
                if 'min_cols' in criteria and len(data.shape) > 1 and data.shape[1] < criteria['min_cols']:
                    return False
                if 'max_cols' in criteria and len(data.shape) > 1 and data.shape[1] > criteria['max_cols']:
                    return False

            # Check column names
            if hasattr(data, 'columns') and 'required_columns' in criteria:
                required_cols = criteria['required_columns']
                if not all(col in data.columns for col in required_cols):
                    return False

            # Check data freshness
            if 'max_data_age_days' in criteria:
                from datetime import datetime
                max_age = criteria['max_data_age_days']

                if hasattr(data, 'index') and pd.api.types.is_datetime64_any_dtype(data.index):
                    latest_date = data.index.max()
                elif hasattr(data, 'columns'):
                    date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if date_cols:
                        latest_date = pd.to_datetime(data[date_cols[0]]).max()
                    else:
                        return True
                else:
                    return True

                if pd.notna(latest_date):
                    age_days = (datetime.now() - latest_date).days
                    if age_days > max_age:
                        return False

            # Check data completeness
            if 'min_completeness' in criteria and hasattr(data, 'notna'):
                completeness = data.notna().mean()
                if isinstance(completeness, pd.Series):
                    completeness = completeness.mean()
                if completeness < criteria['min_completeness']:
                    return False

            return True

        except Exception:
            return criteria.get('allow_errors', True)

    def _calculate_auto_grid(self):
        """Calculate optimal grid dimensions based on number of frames."""
        num_frames = len(self.frames)

        if num_frames <= 1:
            self.grid_config['rows'] = 1
            self.grid_config['cols'] = 1
        elif num_frames <= 2:
            self.grid_config['rows'] = 1
            self.grid_config['cols'] = 2
        elif num_frames <= 4:
            self.grid_config['rows'] = 2
            self.grid_config['cols'] = 2
        elif num_frames <= 6:
            self.grid_config['rows'] = 2
            self.grid_config['cols'] = 3
        elif num_frames <= 9:
            self.grid_config['rows'] = 3
            self.grid_config['cols'] = 3
        else:
            import math
            self.grid_config['cols'] = 4
            self.grid_config['rows'] = math.ceil(num_frames / 4)

    def generate_data_menu(self) -> html.Div:
        """Generate the data selection menu div."""
        if not self.menu_config['enabled']:
            return html.Div()

        alterable_frames = self.menu_config['alterable_frames']
        compact = self.menu_config['compact_mode']

        menu_content = []

        # Menu title
        if not compact:
            menu_content.append(
                html.H4(
                    self.menu_config['menu_title'],
                    style={
                        'textAlign': 'center',
                        'marginBottom': '15px',
                        'color': self.menu_config['text_color'],
                        'fontSize': '1.2rem' if compact else '1.5rem'
                    }
                )
            )

        # Frame selector
        if self.menu_config['show_frame_selector'] and len(alterable_frames) > 1:
            frame_options = []
            for frame_idx in alterable_frames:
                if frame_idx < len(self.frames):
                    frame_title = getattr(self.frames[frame_idx], 'title', f'Frame {frame_idx + 1}')
                    frame_options.append({'label': f'{frame_title}', 'value': frame_idx})

            selector_content = [
                html.Label("Target Frame:",
                           style={'fontWeight': 'bold', 'marginRight': '10px',
                                  'color': self.menu_config['text_color']}),
                dcc.Dropdown(
                    id=f"{self.container_id}_menu_frame_selector",
                    options=frame_options,
                    value=alterable_frames[0],
                    style={'width': '200px' if compact else '100%', 'display': 'inline-block' if compact else 'block'}
                )
            ]

            if compact:
                menu_content.append(
                    html.Div(selector_content,
                             style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'})
                )
            else:
                menu_content.append(
                    html.Div(selector_content, style={'marginBottom': '15px'})
                )

        # Data category selection
        if self.menu_config['data_sources']:
            if compact:
                category_content = self._generate_compact_data_selection()
            else:
                category_content = self._generate_full_data_selection()

            menu_content.append(category_content)

        # Apply button
        if self.menu_config['show_apply_button']:
            button_style = {
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'padding': '8px 16px' if compact else '12px 24px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }

            if compact:
                button_style.update({'marginLeft': '10px'})
            else:
                button_style.update({'width': '100%'})

            menu_content.append(
                html.Button(
                    "Apply" if compact else "Apply Selection",
                    id=f"{self.container_id}_menu_apply_button",
                    n_clicks=0,
                    style=button_style
                )
            )

        # Menu container styling
        menu_style = {
            'padding': '10px' if compact else '20px',
            'backgroundColor': self.menu_config['background_color'],
            'border': f"1px solid {self.style_config['border_color']}",
            'borderRadius': self.style_config['border_radius'],
            'boxSizing': 'border-box',
            'color': self.menu_config['text_color'],
            **self.menu_config['size']
        }

        if compact:
            menu_style.update({
                'display': 'flex',
                'alignItems': 'center',
                'flexWrap': 'wrap',
                'gap': '10px'
            })

        return html.Div(
            menu_content,
            id=f"{self.container_id}_menu_container",
            style=menu_style
        )

    def _generate_compact_data_selection(self) -> html.Div:
        """Generate compact horizontal data selection interface."""
        first_frame_idx = self.menu_config['alterable_frames'][0]
        frame_data_sources = self.menu_config['data_sources'].get(first_frame_idx, {})

        if not frame_data_sources:
            return html.Div()

        selection_elements = []

        for category, options in frame_data_sources.items():
            dropdown_options = []
            for display_name, option_data in options.items():
                if isinstance(option_data, dict):
                    option_type = option_data.get('type', 'key')
                    value = option_data.get('value', '')
                    label = f"{display_name} ({option_type})" if self.menu_config[
                                                                     'selection_mode'] == 'both' else display_name
                else:
                    value = option_data
                    label = display_name

                dropdown_options.append({'label': label, 'value': value})

            selection_elements.extend([
                html.Label(f"{category.title()}:",
                           style={'fontWeight': 'bold', 'marginRight': '5px', 'color': self.menu_config['text_color']}),
                dcc.Dropdown(
                    id=f"{self.container_id}_menu_{category}_dropdown",
                    options=dropdown_options,
                    value=dropdown_options[0]['value'] if dropdown_options else None,
                    style={'width': '200px', 'marginRight': '15px'}
                )
            ])

            if (self.menu_config['show_column_selector'] and
                    self.menu_config['selection_mode'] in ['columns', 'both']):
                selection_elements.extend([
                    html.Label("Column:",
                               style={'fontWeight': 'bold', 'marginRight': '5px',
                                      'color': self.menu_config['text_color']}),
                    dcc.Dropdown(
                        id=f"{self.container_id}_menu_{category}_column_dropdown",
                        options=[],
                        value=None,
                        placeholder="Auto",
                        style={'width': '150px', 'marginRight': '15px'}
                    )
                ])

        return html.Div(
            selection_elements,
            style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '10px'}
        )

    def _generate_full_data_selection(self) -> html.Div:
        """Generate full tabbed data selection interface."""
        first_frame_idx = self.menu_config['alterable_frames'][0]
        frame_data_sources = self.menu_config['data_sources'].get(first_frame_idx, {})

        if not frame_data_sources:
            return html.Div()

        if len(frame_data_sources) > 1:
            tab_children = []
            for category, options in frame_data_sources.items():
                tab_content = self._create_category_tab_content(category, options)
                tab_children.append(
                    dcc.Tab(
                        label=category.replace('_', ' ').title(),
                        value=category.lower(),
                        children=tab_content,
                        style={'padding': '15px', 'backgroundColor': self.menu_config['background_color']}
                    )
                )

            return dcc.Tabs(
                id=f"{self.container_id}_menu_data_tabs",
                value=list(frame_data_sources.keys())[0].lower(),
                children=tab_children,
                style={'marginBottom': '15px'}
            )
        else:
            category = list(frame_data_sources.keys())[0]
            options = frame_data_sources[category]
            return self._create_category_tab_content(category, options)

    def _create_category_tab_content(self, category: str, options: Dict) -> html.Div:
        """Create tab content for a data category."""
        dropdown_options = []
        for display_name, option_data in options.items():
            if isinstance(option_data, dict):
                option_type = option_data.get('type', 'key')
                value = option_data.get('value', '')
                label = f"{display_name} ({option_type})" if self.menu_config[
                                                                 'selection_mode'] == 'both' else display_name
            else:
                value = option_data
                label = display_name

            dropdown_options.append({'label': label, 'value': value})

        content_elements = [
            html.Label(f"Select {category.replace('_', ' ').title()}:",
                       style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': self.menu_config['text_color']}),
            dcc.Dropdown(
                id=f"{self.container_id}_menu_{category}_dropdown",
                options=dropdown_options,
                value=dropdown_options[0]['value'] if dropdown_options else None,
                style={'marginBottom': '10px'}
            )
        ]

        if (self.menu_config['show_column_selector'] and
                self.menu_config['selection_mode'] in ['columns', 'both']):
            content_elements.extend([
                html.Label("Select Column (optional):",
                           style={'fontWeight': 'bold', 'marginBottom': '5px',
                                  'color': self.menu_config['text_color']}),
                dcc.Dropdown(
                    id=f"{self.container_id}_menu_{category}_column_dropdown",
                    options=[],
                    value=None,
                    placeholder="Auto-detect or use default",
                    style={'marginBottom': '10px'}
                )
            ])

        return html.Div(content_elements)

    def _generate_grid_css(self) -> str:
        """Generate CSS for the grid layout including menu positioning."""
        gap = self.grid_config['gap']
        menu_enabled = self.menu_config['enabled']
        menu_position = self.menu_config['position']
        menu_size = self.menu_config['size']

        base_css = f"""
        .frame-grid-wrapper {{
            display: flex;
            width: {self.style_config['container_width']};
            height: {self.style_config['container_height']};
            background-color: {self.style_config['background_color']};
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        }}

        .frame-grid-container {{
            display: grid;
            gap: {gap};
            padding: {self.style_config['padding']};
            box-sizing: border-box;
            flex: 1;
        }}

        .frame-grid-item {{
            background: {self.style_config['frame_background']};
            border: 1px solid {self.style_config['border_color']};
            border-radius: {self.style_config['border_radius']};
            box-shadow: {self.style_config['box_shadow']};
            overflow: hidden;
            min-height: {self.style_config['frame_min_height']};
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .frame-grid-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }}

        .frame-title {{
            color: {self.style_config['title_color']};
            text-align: center;
            margin: 0;
            padding: 15px;
            border-bottom: 1px solid {self.style_config['border_color']};
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }}
        """

        # Add menu positioning styles
        if menu_enabled:
            if menu_position == 'top':
                base_css += f"""
                .frame-grid-wrapper {{ flex-direction: column; }}
                .frame-grid-container {{ height: calc(100% - {menu_size.get('height', '160px')} - 20px); }}
                """
            elif menu_position == 'bottom':
                base_css += f"""
                .frame-grid-wrapper {{ flex-direction: column-reverse; }}
                .frame-grid-container {{ height: calc(100% - {menu_size.get('height', '160px')} - 20px); }}
                """
            elif menu_position == 'left':
                base_css += f"""
                .frame-grid-wrapper {{ flex-direction: row; }}
                .frame-grid-container {{ width: calc(100% - {menu_size.get('width', '300px')} - 20px); }}
                """
            elif menu_position == 'right':
                base_css += f"""
                .frame-grid-wrapper {{ flex-direction: row-reverse; }}
                .frame-grid-container {{ width: calc(100% - {menu_size.get('width', '300px')} - 20px); }}
                """

        # Add responsive breakpoints
        if self.grid_config['responsive']:
            breakpoints = self.grid_config['breakpoints']

            if 'sm' in breakpoints:
                cols_sm = breakpoints['sm'].get('cols', 1)
                padding_sm = self.style_config['responsive_padding'].get('sm', '10px')
                base_css += f"""
                @media (max-width: 768px) {{
                    .frame-grid-container {{
                        grid-template-columns: repeat({cols_sm}, 1fr);
                        padding: {padding_sm};
                        gap: 10px;
                    }}
                }}
                """

            if 'md' in breakpoints:
                cols_md = breakpoints['md'].get('cols', 2)
                padding_md = self.style_config['responsive_padding'].get('md', '15px')
                base_css += f"""
                @media (min-width: 769px) and (max-width: 1024px) {{
                    .frame-grid-container {{
                        grid-template-columns: repeat({cols_md}, 1fr);
                        padding: {padding_md};
                    }}
                }}
                """

            if 'lg' in breakpoints:
                cols_lg = breakpoints['lg'].get('cols', 3)
                padding_lg = self.style_config['responsive_padding'].get('lg', '20px')
                base_css += f"""
                @media (min-width: 1025px) {{
                    .frame-grid-container {{
                        grid-template-columns: repeat({cols_lg}, 1fr);
                        padding: {padding_lg};
                    }}
                }}
                """
        else:
            rows = self.grid_config['rows'] or 'auto'
            cols = self.grid_config['cols'] or 'auto'
            base_css += f"""
            .frame-grid-container {{
                grid-template-rows: repeat({rows}, 1fr);
                grid-template-columns: repeat({cols}, 1fr);
            }}
            """

        return base_css

    def _create_frame_item(self, frame, index: int) -> html.Div:
        """Create a grid item containing a frame."""
        frame_layout = frame.generate_layout_div()

        item_style = {}
        if self.grid_config['layout_type'] == 'custom' and index in self.grid_config['frame_positions']:
            pos_config = self.grid_config['frame_positions'][index]
            if 'row' in pos_config:
                item_style['grid-row'] = f"{pos_config['row']}"
                if 'row_span' in pos_config:
                    item_style['grid-row'] += f" / span {pos_config['row_span']}"
            if 'col' in pos_config:
                item_style['grid-column'] = f"{pos_config['col']}"
                if 'col_span' in pos_config:
                    item_style['grid-column'] += f" / span {pos_config['col_span']}"

        frame_title = getattr(frame, 'title', f'Frame {index + 1}')

        return html.Div([
            html.H3(frame_title, className='frame-title'),
            html.Div(frame_layout, style={'padding': '0', 'height': '100%'})
        ],
            className='frame-grid-item',
            id=f'{self.container_id}-frame-{index}',
            style=item_style)

    def generate_layout(self, include_title: bool = True, title: str = "Dashboard") -> html.Div:
        """Generate the complete grid layout with optional menu."""
        grid_css = self._generate_grid_css()
        frame_items = [self._create_frame_item(frame, i) for i, frame in enumerate(self.frames)]

        layout_children = []
        if include_title:
            title_div = html.Div([
                html.H1(title, style={
                    'textAlign': 'center',
                    'color': self.style_config['title_color'],
                    'marginBottom': '30px',
                    'fontSize': '2.5rem',
                    'fontWeight': '300'
                })
            ], style={'padding': '20px 0'})

        layout_div = html.Div(children=[title_div],style=grid_css)


        wrapper_children = []

        if self.menu_config['enabled']:
            menu_div = self.generate_data_menu()
            wrapper_children.append(menu_div)

