"""
Advanced example showing Patch-based updates for ESR charts
This demonstrates how to use Patch() for efficient chart updates
"""

import dash
from dash import html, dcc, callback, Input, Output, State, Patch
import pandas as pd
import plotly.graph_objects as go
from data.data_tables import ESRTableClient
import json

class ESRPatchComponent:
    """
    ESR component demonstrating advanced Patch-based chart updates.
    Shows how to update specific chart properties without full re-render.
    """
    
    def __init__(self, component_id="esr-patch"):
        self.component_id = component_id
        self.esr_client = ESRTableClient()
        
    def create_layout(self):
        """Create layout with store and charts optimized for Patch updates"""
        return html.Div([
            # Data store
            dcc.Store(id=f'{self.component_id}-data-store'),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Commodity:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id=f'{self.component_id}-commodity',
                        options=[
                            {'label': 'Wheat', 'value': 'wheat'},
                            {'label': 'Corn', 'value': 'corn'},
                            {'label': 'Soybeans', 'value': 'soybeans'}
                        ],
                        value='wheat'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label("Metric:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id=f'{self.component_id}-metric',
                        options=[
                            {'label': 'Weekly Exports', 'value': 'weeklyExports'},
                            {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
                            {'label': 'Total Commitments', 'value': 'currentMYTotalCommitment'}
                        ],
                        value='weeklyExports'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label("Top N Countries:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id=f'{self.component_id}-top-n',
                        min=3,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(3, 11)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'})
            ], style={'marginBottom': '20px'}),
            
            # Chart that will be updated with Patch
            dcc.Graph(
                id=f'{self.component_id}-main-chart',
                figure=self._create_initial_figure()
            ),
            
            # Update buttons to demonstrate different Patch operations
            html.Div([
                html.Button("Update Colors", id=f'{self.component_id}-color-btn', 
                           className="btn btn-primary", style={'margin': '5px'}),
                html.Button("Update Title", id=f'{self.component_id}-title-btn',
                           className="btn btn-secondary", style={'margin': '5px'}),
                html.Button("Add Trendline", id=f'{self.component_id}-trend-btn',
                           className="btn btn-success", style={'margin': '5px'}),
                html.Button("Toggle Markers", id=f'{self.component_id}-marker-btn',
                           className="btn btn-info", style={'margin': '5px'})
            ], style={'textAlign': 'center', 'margin': '20px'}),
            
            # Status display
            html.Div(id=f'{self.component_id}-status', 
                    style={'margin': '20px', 'textAlign': 'center'})
        ])
    
    def _create_initial_figure(self):
        """Create initial empty figure"""
        fig = go.Figure()
        fig.update_layout(
            title="ESR Data - Select options to load",
            xaxis_title="Date",
            yaxis_title="Volume (Metric Tons)",
            height=500
        )
        return fig
    
    def register_callbacks(self):
        """Register callbacks with Patch-based updates"""
        
        # Load data into store when commodity changes
        @callback(
            Output(f'{self.component_id}-data-store', 'data'),
            Input(f'{self.component_id}-commodity', 'value')
        )
        def load_data(commodity):
            """Load merged data for commodity"""
            if not commodity:
                return None
                
            try:
                data = self.esr_client.get_merged_export_data(commodity)
                if data.empty:
                    return None
                return data.to_json(date_format='iso', orient='records')
            except Exception as e:
                print(f"Error loading {commodity}: {e}")
                return None
        
        # Main chart update with data filtering and top-N selection
        @callback(
            Output(f'{self.component_id}-main-chart', 'figure'),
            [Input(f'{self.component_id}-data-store', 'data'),
             Input(f'{self.component_id}-metric', 'value'),
             Input(f'{self.component_id}-top-n', 'value')]
        )
        def update_main_chart(data_json, metric, top_n):
            """Update main chart with new data"""
            if not data_json:
                return self._create_initial_figure()
            
            try:
                data = pd.read_json(data_json, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
                
                # Get top N countries by selected metric
                country_totals = data.groupby('country')[metric].sum().sort_values(ascending=False)
                top_countries = country_totals.head(top_n).index.tolist()
                
                # Filter data to top countries
                filtered_data = data[data['country'].isin(top_countries)]
                
                # Create figure
                fig = go.Figure()
                
                # Add traces for each country
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                for i, country in enumerate(top_countries):
                    country_data = filtered_data[filtered_data['country'] == country]
                    country_data = country_data.sort_values('weekEndingDate')
                    
                    fig.add_trace(go.Scatter(
                        x=country_data['weekEndingDate'],
                        y=country_data[metric],
                        mode='lines',
                        name=country,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    f'{metric}: %{{y:,.0f}} MT<br>' +
                                    '<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"Top {top_n} Countries - {metric.replace('weekly', 'Weekly ').replace('outstanding', 'Outstanding ').replace('current', 'Current ')}",
                    xaxis_title="Date",
                    yaxis_title=f"{metric} (Metric Tons)",
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                return fig
                
            except Exception as e:
                print(f"Error updating main chart: {e}")
                return self._create_initial_figure()
        
        # Patch-based color update
        @callback(
            Output(f'{self.component_id}-main-chart', 'figure', allow_duplicate=True),
            Input(f'{self.component_id}-color-btn', 'n_clicks'),
            State(f'{self.component_id}-main-chart', 'figure'),
            prevent_initial_call=True
        )
        def update_colors(n_clicks, current_figure):
            """Update line colors using Patch for efficiency"""
            if not n_clicks or not current_figure or not current_figure.get('data'):
                return dash.no_update
            
            patch_obj = Patch()
            
            # New color scheme
            new_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                         '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
            
            # Update each trace color
            for i, trace in enumerate(current_figure['data']):
                patch_obj['data'][i]['line']['color'] = new_colors[i % len(new_colors)]
            
            return patch_obj
        
        # Patch-based title update
        @callback(
            Output(f'{self.component_id}-main-chart', 'figure', allow_duplicate=True),
            Input(f'{self.component_id}-title-btn', 'n_clicks'),
            State(f'{self.component_id}-main-chart', 'figure'),
            prevent_initial_call=True
        )
        def update_title(n_clicks, current_figure):
            """Update chart title using Patch"""
            if not n_clicks:
                return dash.no_update
            
            patch_obj = Patch()
            patch_obj['layout']['title']['text'] = f"Updated Chart (Click #{n_clicks})"
            return patch_obj
        
        # Patch-based trendline addition
        @callback(
            Output(f'{self.component_id}-main-chart', 'figure', allow_duplicate=True),
            Input(f'{self.component_id}-trend-btn', 'n_clicks'),
            State(f'{self.component_id}-main-chart', 'figure'),
            State(f'{self.component_id}-data-store', 'data'),
            State(f'{self.component_id}-metric', 'value'),
            prevent_initial_call=True
        )
        def add_trendline(n_clicks, current_figure, data_json, metric):
            """Add trendline using Patch"""
            if not n_clicks or not data_json:
                return dash.no_update
            
            try:
                data = pd.read_json(data_json, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
                
                # Calculate overall trend
                agg_data = data.groupby('weekEndingDate')[metric].sum().reset_index()
                agg_data = agg_data.sort_values('weekEndingDate')
                
                # Simple linear trend
                import numpy as np
                x = np.arange(len(agg_data))
                y = agg_data[metric].values
                z = np.polyfit(x, y, 1)
                trend_y = np.polyval(z, x)
                
                patch_obj = Patch()
                
                # Add trendline trace
                trend_trace = {
                    'x': agg_data['weekEndingDate'].tolist(),
                    'y': trend_y.tolist(),
                    'mode': 'lines',
                    'name': 'Trend',
                    'line': {'color': 'red', 'width': 3, 'dash': 'dash'},
                    'hovertemplate': 'Trend: %{y:,.0f} MT<extra></extra>'
                }
                
                patch_obj['data'].append(trend_trace)
                return patch_obj
                
            except Exception as e:
                print(f"Error adding trendline: {e}")
                return dash.no_update
        
        # Patch-based marker toggle
        @callback(
            Output(f'{self.component_id}-main-chart', 'figure', allow_duplicate=True),
            Input(f'{self.component_id}-marker-btn', 'n_clicks'),
            State(f'{self.component_id}-main-chart', 'figure'),
            prevent_initial_call=True
        )
        def toggle_markers(n_clicks, current_figure):
            """Toggle markers on/off using Patch"""
            if not n_clicks or not current_figure or not current_figure.get('data'):
                return dash.no_update
            
            patch_obj = Patch()
            
            # Toggle between 'lines' and 'lines+markers'
            for i, trace in enumerate(current_figure['data']):
                if trace.get('name') == 'Trend':  # Skip trendline
                    continue
                    
                current_mode = trace.get('mode', 'lines')
                new_mode = 'lines+markers' if current_mode == 'lines' else 'lines'
                patch_obj['data'][i]['mode'] = new_mode
                
                if new_mode == 'lines+markers':
                    patch_obj['data'][i]['marker'] = {'size': 4}
            
            return patch_obj
        
        # Status updates
        @callback(
            Output(f'{self.component_id}-status', 'children'),
            [Input(f'{self.component_id}-color-btn', 'n_clicks'),
             Input(f'{self.component_id}-title-btn', 'n_clicks'),
             Input(f'{self.component_id}-trend-btn', 'n_clicks'),
             Input(f'{self.component_id}-marker-btn', 'n_clicks')]
        )
        def update_status(color_clicks, title_clicks, trend_clicks, marker_clicks):
            """Update status display"""
            operations = []
            if color_clicks:
                operations.append(f"Colors updated {color_clicks} time(s)")
            if title_clicks:
                operations.append(f"Title updated {title_clicks} time(s)")
            if trend_clicks:
                operations.append(f"Trendline added {trend_clicks} time(s)")
            if marker_clicks:
                operations.append(f"Markers toggled {marker_clicks} time(s)")
            
            if operations:
                return html.Div([
                    html.H6("Chart Modifications:", style={'marginBottom': '10px'}),
                    html.Ul([html.Li(op) for op in operations])
                ])
            else:
                return html.P("Use the buttons above to see Patch-based updates in action!")


def create_patch_demo_app():
    """Create demonstration app for Patch functionality"""
    app = dash.Dash(__name__)
    
    component = ESRPatchComponent()
    app.layout = html.Div([
        html.H1("ESR Patch-Based Updates Demo", 
               style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P("This demo shows how to use Dash Patch for efficient chart updates without full re-renders.",
               style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#666'}),
        component.create_layout()
    ])
    
    component.register_callbacks()
    return app


if __name__ == '__main__':
    app = create_patch_demo_app()
    print("Starting ESR Patch Demo...")
    print("Available at: http://127.0.0.1:8052")
    app.run_server(debug=True, port=8052)