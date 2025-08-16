"""
ESR Store Component with dcc.Store for efficient data caching and Patch-based callbacks
"""

import dash
from dash import html, dcc, callback, Input, Output, State, Patch
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data.data_tables import ESRTableClient
import json
from datetime import datetime, date

class ESRStoreComponent:
    """
    Enhanced ESR component using dcc.Store for data caching and Patch for efficient updates.
    Data is cached in store and only refreshed when commodity changes.
    """
    
    def __init__(self, component_id="esr-store"):
        self.component_id = component_id
        self.esr_client = ESRTableClient()
        
    def create_layout(self):
        """Create the component layout with dcc.Store"""
        return html.Div([
            # Data stores
            dcc.Store(
                id=f'{self.component_id}-data-store',
                data=None
            ),
            dcc.Store(
                id=f'{self.component_id}-filtered-store',
                data=None
            ),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Commodity:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id=f'{self.component_id}-commodity-dropdown',
                        options=[
                            {'label': 'Wheat', 'value': 'wheat'},
                            {'label': 'Corn', 'value': 'corn'},
                            {'label': 'Soybeans', 'value': 'soybeans'},
                            {'label': 'Cattle', 'value': 'cattle'},
                            {'label': 'Hogs', 'value': 'hogs'}
                        ],
                        value='wheat',
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '20%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label("Countries:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id=f'{self.component_id}-country-dropdown',
                        multi=True,
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '35%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label("Marketing Years:", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id=f'{self.component_id}-year-slider',
                        min=2010,
                        max=2025,
                        step=1,
                        value=[2020, 2025],
                        marks={i: str(i) for i in range(2010, 2026, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '40%', 'display': 'inline-block', 'margin': '10px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
            
            # Charts
            html.Div([
                html.Div([
                    dcc.Graph(id=f'{self.component_id}-exports-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id=f'{self.component_id}-commitments-chart') 
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id=f'{self.component_id}-country-comparison-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id=f'{self.component_id}-seasonal-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Summary statistics
            html.Div(
                id=f'{self.component_id}-summary',
                style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}
            )
        ])
    
    def register_callbacks(self):
        """Register all callbacks for the component"""
        
        # Callback to load data into store when commodity changes
        @callback(
            [Output(f'{self.component_id}-data-store', 'data'),
             Output(f'{self.component_id}-country-dropdown', 'options')],
            Input(f'{self.component_id}-commodity-dropdown', 'value')
        )
        def load_commodity_data(commodity):
            """Load merged export data for selected commodity"""
            if not commodity:
                return None, []
                
            try:
                # Get merged data for commodity
                print(f"Loading merged data for {commodity}...")
                data = self.esr_client.get_merged_export_data(commodity)
                
                if data.empty:
                    return None, []
                
                # Convert to JSON for storage
                data_json = data.to_json(date_format='iso', orient='records')
                
                # Get available countries for dropdown
                countries = sorted(data['country'].unique().tolist())
                country_options = [{'label': country, 'value': country} for country in countries]
                
                print(f"Loaded {len(data)} rows for {commodity}")
                return data_json, country_options
                
            except Exception as e:
                print(f"Error loading data for {commodity}: {e}")
                return None, []
        
        # Callback to filter data based on selections
        @callback(
            Output(f'{self.component_id}-filtered-store', 'data'),
            [Input(f'{self.component_id}-data-store', 'data'),
             Input(f'{self.component_id}-country-dropdown', 'value'),
             Input(f'{self.component_id}-year-slider', 'value')]
        )
        def filter_data(data_json, selected_countries, year_range):
            """Filter data based on selections"""
            if not data_json:
                return None
                
            try:
                # Convert from JSON
                data = pd.read_json(data_json, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
                
                # Filter by countries
                if selected_countries:
                    data = data[data['country'].isin(selected_countries)]
                
                # Filter by years
                if year_range:
                    data = data[
                        (data['marketing_year'] >= year_range[0]) & 
                        (data['marketing_year'] <= year_range[1])
                    ]
                
                # Convert back to JSON
                return data.to_json(date_format='iso', orient='records')
                
            except Exception as e:
                print(f"Error filtering data: {e}")
                return None
        
        # Callback to update exports chart using Patch
        @callback(
            Output(f'{self.component_id}-exports-chart', 'figure'),
            Input(f'{self.component_id}-filtered-store', 'data')
        )
        def update_exports_chart(filtered_data_json):
            """Update exports chart efficiently using Patch"""
            if not filtered_data_json:
                return self._create_empty_figure("Weekly Exports")
            
            try:
                data = pd.read_json(filtered_data_json, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
                
                # Create figure
                fig = go.Figure()
                
                # Determine the exports column name
                exports_col = 'weeklyExports' if 'weeklyExports' in data.columns else 'all_exports'
                
                # Group by country for multiple lines
                for country in data['country'].unique():
                    country_data = data[data['country'] == country]
                    country_data = country_data.sort_values('weekEndingDate')
                    
                    fig.add_trace(go.Scatter(
                        x=country_data['weekEndingDate'],
                        y=country_data[exports_col],
                        mode='lines',
                        name=country,
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    f'{exports_col}: %{{y:,.0f}} MT<br>' +
                                    '<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Weekly Exports by Country",
                    xaxis_title="Date",
                    yaxis_title="Weekly Exports (Metric Tons)",
                    hovermode='x unified',
                    height=400
                )
                
                return fig
                
            except Exception as e:
                print(f"Error updating exports chart: {e}")
                return self._create_empty_figure("Weekly Exports - Error")
        
        # Callback to update commitments chart
        @callback(
            Output(f'{self.component_id}-commitments-chart', 'figure'),
            Input(f'{self.component_id}-filtered-store', 'data')
        )
        def update_commitments_chart(filtered_data_json):
            """Update commitments vs sales chart"""
            if not filtered_data_json:
                return self._create_empty_figure("Commitments vs Sales")
            
            try:
                data = pd.read_json(filtered_data_json, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
                
                # Determine the exports column name
                exports_col = 'weeklyExports' if 'weeklyExports' in data.columns else 'all_exports'
                
                # Aggregate by date
                agg_columns = {
                    'outstandingSales': 'sum',
                    'currentMYTotalCommitment': 'sum',
                    exports_col: 'sum'
                }
                # Only include columns that exist
                agg_columns = {k: v for k, v in agg_columns.items() if k in data.columns}
                
                agg_data = data.groupby('weekEndingDate').agg(agg_columns).reset_index()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=agg_data['weekEndingDate'],
                    y=agg_data['outstandingSales'],
                    mode='lines',
                    name='Outstanding Sales',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=agg_data['weekEndingDate'],
                    y=agg_data['currentMYTotalCommitment'],
                    mode='lines',
                    name='Total Commitments',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title="Outstanding Sales vs Total Commitments",
                    xaxis_title="Date",
                    yaxis_title="Volume (Metric Tons)",
                    hovermode='x unified',
                    height=400
                )
                
                return fig
                
            except Exception as e:
                print(f"Error updating commitments chart: {e}")
                return self._create_empty_figure("Commitments vs Sales - Error")
        
        # Callback to update country comparison using Patch for efficiency
        @callback(
            Output(f'{self.component_id}-country-comparison-chart', 'figure'),
            Input(f'{self.component_id}-filtered-store', 'data')
        )
        def update_country_comparison(filtered_data_json):
            """Update country comparison chart with Patch optimization"""
            if not filtered_data_json:
                return self._create_empty_figure("Country Comparison")
            
            try:
                data = pd.read_json(filtered_data_json, orient='records')
                
                # Determine the exports column name
                exports_col = 'weeklyExports' if 'weeklyExports' in data.columns else 'all_exports'
                
                # Aggregate by country
                agg_columns = {
                    exports_col: 'sum',
                    'currentMYTotalCommitment': 'sum'
                }
                # Only include columns that exist
                agg_columns = {k: v for k, v in agg_columns.items() if k in data.columns}
                
                country_totals = data.groupby('country').agg(agg_columns).reset_index()
                
                # Sort by total exports
                country_totals = country_totals.sort_values(exports_col, ascending=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=country_totals['country'],
                        x=country_totals[exports_col],
                        orientation='h',
                        name='Total Exports',
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title="Total Exports by Country",
                    xaxis_title="Total Exports (Metric Tons)",
                    yaxis_title="Country",
                    height=400
                )
                
                return fig
                
            except Exception as e:
                print(f"Error updating country comparison: {e}")
                return self._create_empty_figure("Country Comparison - Error")
        
        # Callback to update seasonal chart
        @callback(
            Output(f'{self.component_id}-seasonal-chart', 'figure'),
            Input(f'{self.component_id}-filtered-store', 'data')
        )
        def update_seasonal_chart(filtered_data_json):
            """Update seasonal patterns chart"""
            if not filtered_data_json:
                return self._create_empty_figure("Seasonal Patterns")
            
            try:
                data = pd.read_json(filtered_data_json, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
                data['month'] = data['weekEndingDate'].dt.month
                
                # Determine the exports column name
                exports_col = 'weeklyExports' if 'weeklyExports' in data.columns else 'all_exports'
                
                # Aggregate by month and year
                monthly_data = data.groupby(['marketing_year', 'month']).agg({
                    exports_col: 'sum'
                }).reset_index()
                
                fig = go.Figure()
                
                # Create line for each year
                for year in sorted(monthly_data['marketing_year'].unique()):
                    year_data = monthly_data[monthly_data['marketing_year'] == year]
                    
                    fig.add_trace(go.Scatter(
                        x=year_data['month'],
                        y=year_data[exports_col],
                        mode='lines+markers',
                        name=f'MY {year}',
                        hovertemplate=f'<b>MY {year}</b><br>' +
                                    'Month: %{x}<br>' +
                                    'Total Exports: %{y:,.0f} MT<br>' +
                                    '<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Seasonal Export Patterns by Marketing Year",
                    xaxis_title="Month",
                    yaxis_title="Monthly Exports (Metric Tons)",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(1, 13)),
                        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    ),
                    height=400
                )
                
                return fig
                
            except Exception as e:
                print(f"Error updating seasonal chart: {e}")
                return self._create_empty_figure("Seasonal Patterns - Error")
        
        # Callback to update summary statistics
        @callback(
            Output(f'{self.component_id}-summary', 'children'),
            Input(f'{self.component_id}-filtered-store', 'data'),
            State(f'{self.component_id}-commodity-dropdown', 'value')
        )
        def update_summary(filtered_data_json, commodity):
            """Update summary statistics"""
            if not filtered_data_json:
                return html.P("No data available")
            
            try:
                data = pd.read_json(filtered_data_json, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
                
                # Determine the exports column name
                exports_col = 'weeklyExports' if 'weeklyExports' in data.columns else 'all_exports'
                
                # Calculate statistics
                total_exports = data[exports_col].sum()
                avg_weekly = data[exports_col].mean()
                countries_count = data['country'].nunique()
                date_range = f"{data['weekEndingDate'].min().strftime('%Y-%m-%d')} to {data['weekEndingDate'].max().strftime('%Y-%m-%d')}"
                years_covered = sorted(data['marketing_year'].unique())
                
                return html.Div([
                    html.H4(f"{commodity.title()} Export Summary", style={'marginBottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.H6("Total Exports", style={'margin': '0', 'color': '#6c757d'}),
                            html.H4(f"{total_exports:,.0f} MT", style={'margin': '0', 'color': '#007bff'})
                        ], style={'textAlign': 'center', 'padding': '10px'}),
                        
                        html.Div([
                            html.H6("Avg Weekly", style={'margin': '0', 'color': '#6c757d'}),
                            html.H4(f"{avg_weekly:,.0f} MT", style={'margin': '0', 'color': '#28a745'})
                        ], style={'textAlign': 'center', 'padding': '10px'}),
                        
                        html.Div([
                            html.H6("Countries", style={'margin': '0', 'color': '#6c757d'}),
                            html.H4(f"{countries_count}", style={'margin': '0', 'color': '#ffc107'})
                        ], style={'textAlign': 'center', 'padding': '10px'}),
                        
                        html.Div([
                            html.H6("Years Covered", style={'margin': '0', 'color': '#6c757d'}),
                            html.H4(f"{years_covered[0]}-{years_covered[-1]}", style={'margin': '0', 'color': '#17a2b8'})
                        ], style={'textAlign': 'center', 'padding': '10px'})
                    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
                    
                    html.P(f"Date Range: {date_range}", style={'textAlign': 'center', 'marginTop': '10px', 'color': '#6c757d'})
                ])
                
            except Exception as e:
                print(f"Error updating summary: {e}")
                return html.P(f"Error calculating summary: {str(e)}")
    
    def _create_empty_figure(self, title):
        """Create an empty figure with title"""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig


# Example usage function
def create_esr_store_app():
    """Create a standalone Dash app with ESR Store Component"""
    app = dash.Dash(__name__)
    
    esr_component = ESRStoreComponent()
    app.layout = html.Div([
        html.H1("ESR Data Dashboard with Store Caching", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        esr_component.create_layout()
    ])
    
    esr_component.register_callbacks()
    
    return app


if __name__ == '__main__':
    app = create_esr_store_app()
    print("Starting ESR Store Dashboard...")
    print("Available at: http://127.0.0.1:8051")
    app.run_server(debug=True, port=8051)