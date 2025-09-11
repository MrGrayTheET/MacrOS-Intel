"""
Simple Natural Gas Analysis App using EIATable and available keys
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
from data.data_tables import EIATable

class SimpleNaturalGasApp:
    """Simple natural gas analysis using available EIA keys"""
    
    def __init__(self):
        self.eia_client = EIATable('NG')
        self.available_keys = self._get_available_keys()
        
    def _get_available_keys(self):
        """Get categorized available keys"""
        return {
            'storage': {
                'Total Lower 48': 'storage/total_lower_48',
                'East Region': 'storage/east_region',
                'Midwest Region': 'storage/midwest_region', 
                'South Central Region': 'storage/south_central_region',
                'Pacific Region': 'storage/pacific_region'
            },
            'prices': {
                'NG Front Month': 'prices/NG_1',
                'NG Second Month': 'prices/NG_2',
                'Henry Hub Daily': 'prices/henry_hub_daily',
                'Residential Monthly': 'prices/residential_monthly'
            },
            'consumption': {
                'Net Withdrawals': 'consumption/net_withdrawals',
                'Total Consumption': 'consumption/total',
                'Residential': 'consumption/residential',
                'Commercial': 'consumption/commercial',
                'Industrial': 'consumption/industrial',
                'Electric Power': 'consumption/electric_power'
            },
            'production': {
                'Dry Production Monthly': 'production/dry_production_monthly'
            },
            'trade': {
                'Total Exports': 'exports/total',
                'LNG Exports': 'exports/lng',
                'Canada Imports': 'imports/Canada',
                'Mexico Imports': 'imports/Mexico'
            }
        }
    
    def create_layout(self):
        """Create the app layout"""
        return html.Div([
            html.H1("Natural Gas Analysis Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            html.P("Select data series to analyze from available EIA natural gas data",
                   style={'textAlign': 'center', 'margin': '20px'}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Primary Series:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='primary-series',
                        options=self._create_dropdown_options(),
                        value='storage/total_lower_48',
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label("Secondary Series:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='secondary-series', 
                        options=self._create_dropdown_options(),
                        value='prices/NG_1',
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'margin': '10px'})
            ], style={'textAlign': 'center'}),
            
            # Date range picker
            html.Div([
                html.Label("Date Range:", style={'fontWeight': 'bold'}),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date='2020-01-01',
                    end_date='2025-06-01',
                    display_format='YYYY-MM-DD'
                )
            ], style={'textAlign': 'center', 'margin': '20px'}),
            
            # Charts
            html.Div([
                dcc.Graph(id='time-series-chart')
            ], style={'margin': '20px'}),
            
            html.Div([
                dcc.Graph(id='correlation-chart')
            ], style={'margin': '20px'}),
            
            # Data summary
            html.Div(id='data-summary', style={'margin': '20px', 'padding': '20px'})
        ])
    
    def _create_dropdown_options(self):
        """Create dropdown options from available keys"""
        options = []
        for category, keys in self.available_keys.items():
            for name, key in keys.items():
                options.append({
                    'label': f"{category.title()}: {name}",
                    'value': key
                })
        return options
    
    def register_callbacks(self, app):
        """Register callbacks"""
        
        @callback(
            [Output('time-series-chart', 'figure'),
             Output('correlation-chart', 'figure'),
             Output('data-summary', 'children')],
            [Input('primary-series', 'value'),
             Input('secondary-series', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_charts(primary_key, secondary_key, start_date, end_date):
            try:
                # Load data
                data = self.eia_client.get_keys([primary_key, secondary_key])
                
                if data.empty:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No data available", 
                                           xref="paper", yref="paper", x=0.5, y=0.5)
                    return empty_fig, empty_fig, "No data available"
                
                # Filter by date range
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                # Remove NaN values
                data = data.dropna()
                
                if data.empty:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No data in selected range", 
                                           xref="paper", yref="paper", x=0.5, y=0.5)
                    return empty_fig, empty_fig, "No data in selected date range"
                
                # Create time series chart
                ts_fig = go.Figure()
                
                # Primary series
                primary_col = data.columns[0]
                ts_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[primary_col],
                    name=primary_col,
                    line=dict(color='blue'),
                    yaxis='y'
                ))
                
                # Secondary series (on secondary y-axis)
                secondary_col = data.columns[1]
                ts_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[secondary_col],
                    name=secondary_col,
                    line=dict(color='red'),
                    yaxis='y2'
                ))
                
                ts_fig.update_layout(
                    title="Natural Gas Time Series Analysis",
                    xaxis_title="Date",
                    yaxis=dict(title=primary_col, side='left'),
                    yaxis2=dict(title=secondary_col, side='right', overlaying='y'),
                    hovermode='x unified'
                )
                
                # Create correlation scatter plot
                corr_fig = go.Figure()
                corr_fig.add_trace(go.Scatter(
                    x=data[primary_col],
                    y=data[secondary_col],
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='green', size=4)
                ))
                
                # Calculate correlation
                correlation = data[primary_col].corr(data[secondary_col])
                
                corr_fig.update_layout(
                    title=f"Correlation Analysis (r = {correlation:.3f})",
                    xaxis_title=primary_col,
                    yaxis_title=secondary_col
                )
                
                # Create summary
                summary = html.Div([
                    html.H4("Data Summary"),
                    html.P(f"Data Points: {len(data):,}"),
                    html.P(f"Date Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"),
                    html.P(f"Correlation: {correlation:.4f}"),
                    html.H5(f"{primary_col} Statistics:"),
                    html.P(f"Mean: {data[primary_col].mean():.2f}, Std: {data[primary_col].std():.2f}"),
                    html.H5(f"{secondary_col} Statistics:"),
                    html.P(f"Mean: {data[secondary_col].mean():.2f}, Std: {data[secondary_col].std():.2f}")
                ])
                
                return ts_fig, corr_fig, summary
                
            except Exception as e:
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", 
                                       xref="paper", yref="paper", x=0.5, y=0.5)
                error_summary = html.P(f"Error loading data: {str(e)}")
                return error_fig, error_fig, error_summary


# Create app instance
def create_app():
    """Create and return the Dash app"""
    app = dash.Dash(__name__)
    
    natgas_app = SimpleNaturalGasApp()
    app.layout = natgas_app.create_layout()
    natgas_app.register_callbacks(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("Starting Simple Natural Gas Analysis App...")
    print("Available at: http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)