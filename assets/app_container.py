from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
from typing import List, Dict, Any, Callable
import time
import dash


class UnifiedDashboard:
    """
    Enhanced unified dashboard that ensures proper data loading in callbacks.
    """

    def __init__(self,
                 app: dash.Dash,
                 theme: str = "dark",
                 title: str = "Unified Dashboard"):
        self.app = app
        self.theme = theme
        self.title = title
        self.frames = {}
        self.active_frame = None
        self.frame_loaded_states = {}  # Track which frames have been initialized

        # Dark theme styles (same as before)
        self.dark_theme = {
            'background': '#1a1a1a',
            'card_background': '#2d2d2d',
            'border_color': '#404040',
            'text_color': '#e0e0e0',
            'heading_color': '#ffffff',
            'accent_color': '#4CAF50',
            'secondary_accent': '#2196F3',
            'hover_background': '#3a3a3a',
            'input_background': '#363636',
            'tab_inactive': '#2d2d2d',
            'tab_active': '#404040',
            'tab_border': '#555555'
        }

        self.light_theme = {
            'background': '#f5f5f5',
            'card_background': '#ffffff',
            'border_color': '#dddddd',
            'text_color': '#333333',
            'heading_color': '#222222',
            'accent_color': '#4CAF50',
            'secondary_accent': '#2196F3',
            'hover_background': '#f0f0f0',
            'input_background': '#ffffff',
            'tab_inactive': '#f0f0f0',
            'tab_active': '#ffffff',
            'tab_border': '#cccccc'
        }

        self.current_theme = self.dark_theme if theme == "dark" else self.light_theme

    def add_frame(self, frame_id: str, frame_instance: Any, frame_type: str):
        """
        Add a frame instance to the dashboard with initialization tracking.
        """
        self.frames[frame_id] = {
            'instance': frame_instance,
            'key_type': frame_type,
            'callbacks_registered': False,
            'data_loaded': False
        }
        self.frame_loaded_states[frame_id] = False

    def apply_dark_theme_to_frame(self, frame_layout: html.Div) -> html.Div:
        """
        Apply dark theme styling to a frame layout while preserving functionality.
        """

        def apply_theme_recursive(element):
            if hasattr(element, 'style') and element.style:
                # Create a copy of the style to avoid modifying the original
                new_style = element.style.copy() if isinstance(element.style, dict) else {}

                # Update background colors
                if 'background-color' in new_style:
                    if new_style['background-color'] in ['#ffffff', '#fff', 'white']:
                        new_style['background-color'] = self.current_theme['card_background']
                    elif new_style['background-color'] in ['#f8f9fa', '#f5f5f5', '#f0f8ff']:
                        new_style['background-color'] = self.current_theme['input_background']

                # Update text colors
                if 'color' in new_style:
                    if new_style['color'] in ['#333', '#333333', '#444', '#666', '#666666']:
                        new_style['color'] = self.current_theme['text_color']
                    elif new_style['color'] in ['#000', '#000000', 'black']:
                        new_style['color'] = self.current_theme['heading_color']

                # Update border colors
                if 'border' in new_style:
                    new_style['border'] = new_style['border'].replace('#333', self.current_theme['border_color'])
                    new_style['border'] = new_style['border'].replace('#34495e', self.current_theme['border_color'])
                    new_style['border'] = new_style['border'].replace('#2196F3', self.current_theme['secondary_accent'])
                    new_style['border'] = new_style['border'].replace('#ddd', self.current_theme['border_color'])
                    new_style['border'] = new_style['border'].replace('#eee', self.current_theme['border_color'])

                if 'border-bottom' in new_style:
                    new_style['border-bottom'] = new_style['border-bottom'].replace('#eee',
                                                                                    self.current_theme['border_color'])

                element.style = new_style

            # Apply to children recursively
            if hasattr(element, 'children'):
                if isinstance(element.children, list):
                    for child in element.children:
                        if hasattr(child, 'style'):
                            apply_theme_recursive(child)
                elif hasattr(element.children, 'style'):
                    apply_theme_recursive(element.children)

            return element

        return apply_theme_recursive(frame_layout)

    def create_navigation_tabs(self) -> dcc.Tabs:
        """
        Create navigation tabs for switching between frames.
        """
        tabs = []
        for frame_id, frame_info in self.frames.items():
            tab_label = frame_id.replace('_', ' ').title()
            tabs.append(
                dcc.Tab(
                    label=tab_label,
                    value=frame_id,
                    style={
                        'background-color': self.current_theme['tab_inactive'],
                        'color': self.current_theme['text_color'],
                        'border': f"1px solid {self.current_theme['tab_border']}",
                        'border-radius': '4px 4px 0 0',
                        'padding': '12px 20px',
                        'margin-right': '5px',
                        'font-weight': '500'
                    },
                    selected_style={
                        'background-color': self.current_theme['tab_active'],
                        'color': self.current_theme['heading_color'],
                        'border': f"1px solid {self.current_theme['tab_border']}",
                        'border-bottom': 'none',
                        'border-radius': '4px 4px 0 0',
                        'padding': '12px 20px',
                        'margin-right': '5px',
                        'font-weight': '600'
                    }
                )
            )

        return dcc.Tabs(
            id='main-navigation-tabs',
            value=list(self.frames.keys())[0] if self.frames else None,
            children=tabs,
            style={
                'height': '50px',
                'margin-bottom': '0'
            }
        )

    def create_frame_container(self, frame_id: str) -> html.Div:
        """
        Create a container for a specific frame with initialization store.
        """
        frame_info = self.frames.get(frame_id)
        if not frame_info:
            return html.Div("Frame not found")

        frame_instance = frame_info['instance']

        # Get the frame's layout
        if hasattr(frame_instance, 'generate_layout_div'):
            frame_layout = frame_instance.generate_layout_div()
        elif hasattr(frame_instance, 'layout'):
            frame_layout = frame_instance.layout()
        else:
            frame_layout = html.Div("Frame layout method not found")

        # Apply dark theme
        themed_layout = self.apply_dark_theme_to_frame(frame_layout)

        # Wrap with initialization tracking
        return html.Div([
            # Hidden store to track frame initialization
            dcc.Store(id=f'{frame_id}-init-store', data={'initialized': False}),

            # Loading indicator
            dcc.Loading(
                id=f"{frame_id}-loading",
                type="default",
                children=[
                    html.Div(
                        id=f'{frame_id}-content-wrapper',
                        children=themed_layout,
                        style={'min-height': '400px'}
                    )
                ],
                style={'min-height': '400px'},
                color=self.current_theme['accent_color']
            )
        ],
            id=f'{frame_id}-container',
            style={
                'padding': '20px',
                'background-color': self.current_theme['background'],
                'min-height': '100vh',
                'display': 'block' if frame_id == list(self.frames.keys())[0] else 'none'
            }
        )

    def generate_layout(self) -> html.Div:
        """
        Generate the complete dashboard layout.
        """
        # Create containers for all frames
        frame_containers = []
        for frame_id in self.frames.keys():
            frame_containers.append(self.create_frame_container(frame_id))

        # Main layout
        layout = html.Div([
            # Hidden store for active frame tracking
            dcc.Store(id='active-frame-store', data={'active': list(self.frames.keys())[0] if self.frames else None}),

            # Header
            html.Div([
                html.H1(
                    self.title,
                    style={
                        'color': self.current_theme['heading_color'],
                        'margin': '0',
                        'padding': '20px',
                        'font-size': '28px',
                        'font-weight': '600'
                    }
                ),
                html.Hr(style={
                    'margin': '0',
                    'border': 'none',
                    'border-top': f"1px solid {self.current_theme['border_color']}"
                })
            ], style={
                'background-color': self.current_theme['card_background'],
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),

            # Navigation tabs
            html.Div([
                self.create_navigation_tabs()
            ], style={
                'background-color': self.current_theme['background'],
                'padding': '20px 20px 0 20px'
            }),

            # Frame content area
            html.Div(
                id='frame-content-area',
                children=frame_containers,
                style={
                    'background-color': self.current_theme['background'],
                    'min-height': 'calc(100vh - 150px)'
                }
            )
        ], style={
            'background-color': self.current_theme['background'],
            'min-height': '100vh',
            'margin': '0',
            'padding': '0',
            'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
        })

        return layout

    def register_unified_callbacks(self):
        """
        Register callbacks with proper initialization handling.
        """
        # First, register all frame callbacks immediately
        for frame_id, frame_info in self.frames.items():
            frame_instance = frame_info['instance']
            if hasattr(frame_instance, 'register_callbacks') and not frame_info['callbacks_registered']:
                try:
                    frame_instance.register_callbacks(self.app)
                    frame_info['callbacks_registered'] = True
                    print(f"Callbacks registered for {frame_id}")
                except Exception as e:
                    print(f"Error registering callbacks for {frame_id}: {e}")

        # Callback for switching between frames and tracking active state
        @self.app.callback(
            [Output(f'{frame_id}-container', 'style') for frame_id in self.frames.keys()] +
            [Output('active-frame-store', 'data')],
            [Input('main-navigation-tabs', 'value')],
            [State('active-frame-store', 'data')]
        )
        def switch_frame(active_tab, current_active_data):
            styles = []

            for frame_id in self.frames.keys():
                if frame_id == active_tab:
                    styles.append({
                        'padding': '20px',
                        'background-color': self.current_theme['background'],
                        'min-height': '100vh',
                        'display': 'block'
                    })
                else:
                    styles.append({
                        'padding': '20px',
                        'background-color': self.current_theme['background'],
                        'min-height': '100vh',
                        'display': 'none'
                    })

            return styles + [{'active': active_tab}]

        # Frame initialization callbacks
        for frame_id in self.frames.keys():
            @self.app.callback(
                Output(f'{frame_id}-init-store', 'data'),
                [Input('active-frame-store', 'data')],
                [State(f'{frame_id}-init-store', 'data')],
                prevent_initial_call=False
            )
            def initialize_frame(active_data, init_data, frame_id=frame_id):
                if active_data and active_data.get('active') == frame_id and not init_data.get('initialized'):
                    # Trigger any initialization needed for the frame
                    frame_instance = self.frames[frame_id]['instance']

                    # If the frame has an initialization method, call it
                    if hasattr(frame_instance, 'initialize') or hasattr(frame_instance, 'load_data'):
                        try:
                            if hasattr(frame_instance, 'initialize'):
                                frame_instance.initialize()
                            if hasattr(frame_instance, 'load_data'):
                                frame_instance.load_data()
                        except Exception as e:
                            print(f"Error initializing {frame_id}: {e}")

                    return {'initialized': True, 'timestamp': time.time()}

                return init_data


# Example usage
def create_unified_dashboard_app():
    """
    Example function showing how to create a unified dashboard with multiple frames.
    """
    # Initialize Dash app with dark theme CSS
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],  # Bootstrap dark theme
        suppress_callback_exceptions=True  # Important for dynamic callbacks
    )

    # Create unified dashboard instance
    dashboard = UnifiedDashboard(app, theme="dark", title="Multi-Frame Trading Dashboard")

    # Example: Add your frame instances
    # Assuming you have instances of your frames already created:

    # from your_module import StorageApp, FundamentalFrame, MarketFrame
    #
    # storage_app = StorageApp(...)
    # fundamental_frame = FundamentalFrame(
    #     table_client=your_table_client,
    #     chart_configs=your_chart_configs,
    #     layout="horizontal",
    #     div_prefix="fundamental",
    #     data_options=your_data_options
    # )
    # market_frame = MarketFrame(
    #     market_table=your_market_table,
    #     chart_configs=your_market_configs,
    #     layout="vertical",
    #     div_prefix="market"
    # )
    #
    # dashboard.add_frame('storage', storage_app, 'storage')
    # dashboard.add_frame('fundamentals', fundamental_frame, 'fundamental')
    # dashboard.add_frame('market', market_frame, 'market')

    # Set the app layout
    app.layout = dashboard.generate_layout()

    # Register unified callbacks
    dashboard.register_unified_callbacks()

    return app, dashboard


# CSS overrides for better dark theme support
DARK_THEME_CSS = """ /* assets/styles.css */"""

if __name__ == "__main__":
    # Create and run the app
    app, dashboard = create_unified_dashboard_app()

    # Add custom CSS
    app.index_string = f'''
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {DARK_THEME_CSS}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    '''

    app.run_server(debug=True)