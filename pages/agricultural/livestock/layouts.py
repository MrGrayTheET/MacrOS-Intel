from dotenv import load_dotenv
load_dotenv('.env'); from sources.data_tables import NASSTable, FASTable
from typing import List
from assets.frames import FundamentalFrame as FFrame, MarketFrame as MFrame
from assets.plotting.chart_components import FundamentalChart as FChart
from assets.app_container import DARK_THEME_CSS, UnifiedDashboard as UDash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash
import dash_bootstrap_components as dbc
from pathlib import Path

class UnifiedChartLayout:

    def __init__(self, frame_windows):
        self.frames = frame_windows
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
        self.current_theme = self.dark_theme

        return

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

    def get_chart_ids(self):
        chart_ids = []
        for frame in self.frames:
            if hasattr(frame, 'charts'):
                for chart in frame.charts:
                    if hasattr(frame, 'chart_id'):
                        chart_ids.append(chart.chart_id)

        return chart_ids

    def chart_options(self):
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

        chart_menu = html.Div(children=[
            html.H4("Data Menu", style={'text-color':self.current_theme['text_color']}),
            html.Title("Chart Color", style={ 'text-color':self.current_theme['text_color']}),
            dcc.Dropdown(id='color-dd', options=color_options, value=color_options[0]['value'], style={'height':'10px','width':'50px'}),
            html.Br(),
            html.Title("Chart Type", style={'text-color':self.current_theme['text_color']}),dcc.Dropdown(id='chart-key_type-dd',options=chart_type_options, value=chart_type_options[1]['value'])
        ],
            style={
            'background-color':self.current_theme['background'],
            'border': f'1px solid {self.current_theme["border"]}',
            'color': self.current_theme['heading_color']

        })
        return chart_menu


import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, date

# Initialize the Dash app
app = dash.Dash(__name__)


# Sample data creation (replace with your actual data loading)
def create_sample_data():
    """Create sample data matching your structure for demonstration"""
    countries = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Mexico', 'Peru', 'Uruguay']
    country_codes = ['AR', 'BR', 'CL', 'CO', 'MX', 'PE', 'UY']

    # Create date range from 1996 to 2024
    dates = pd.date_range(start='1996-01-31', end='2024-12-31', freq='M')

    data = []
    for country, code in zip(countries, country_codes):
        for date_val in dates:
            # Generate realistic import values with some seasonality and trends
            base_value = np.random.randint(5000, 50000)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date_val.month / 12)
            trend_factor = 1 + 0.02 * (date_val.year - 1996)
            noise = np.random.normal(1, 0.1)

            value = int(base_value * seasonal_factor * trend_factor * noise)
            data.append({
                'Partner': country,
                'date': date_val,
                'value': max(value, 0),  # Ensure no negative values
                'Partner Code': code
            })

    df = pd.DataFrame(data)
    df = df.set_index(['Partner', 'date'])
    return df


# Load your data here - replace this with your actual data loading
# df = pd.read_csv('your_data.csv')
# df = df.set_index(['Partner', 'date'])
df = create_sample_data()

# Extract unique countries and years for dropdowns
countries = df.index.get_level_values('Partner').unique().tolist()
years = df.index.get_level_values('date').year.unique().tolist()
years.sort()

# Define custom colors for dark theme
colors = {
    'primary': '#64B5F6',
    'secondary': '#E57373',
    'background': '#121212',
    'card_background': '#1E1E1E',
    'text': '#FFFFFF',
    'text_secondary': '#B0B0B0',
    'accent': '#FFB74D',
    'grid': '#2A2A2A',
    'border': '#333333'
}

# Custom CSS styling
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Import Sources Analysis Dashboard",
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '10px'}),
        html.P("Analyze commodity import patterns by country and time period",
               style={'textAlign': 'center', 'color': colors['text_secondary'], 'fontSize': '16px'})
    ], style={'backgroundColor': colors['card_background'], 'padding': '20px', 'marginBottom': '20px',
              'border': f'1px solid {colors["border"]}'}),

    # Main content container
    html.Div([
        # First row: Time series analysis
        html.Div([
            html.Div([
                html.H3("Monthly Import Trends by Country",
                        style={'color': colors['text'], 'marginBottom': '20px'}),

                # Controls for time series
                html.Div([
                    html.Div([
                        html.Label("Select Country:",
                                   style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': colors['text']}),
                        dcc.Dropdown(
                            id='country-dropdown',
                            options=[{'label': country, 'value': country} for country in countries],
                            value=countries[0],
                            style={'marginBottom': '10px'},
                            className='dark-dropdown'
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.Label("Select Date Range:",
                                   style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': colors['text']}),
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            start_date=date(2020, 1, 1),
                            end_date=date(2024, 12, 31),
                            display_format='YYYY-MM-DD',
                            style={'width': '100%'},
                            className='dark-datepicker'
                        )
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),

                # Time series chart
                dcc.Graph(id='time-series-chart')

            ], style={'backgroundColor': colors['card_background'], 'padding': '20px', 'borderRadius': '10px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.3)', 'marginBottom': '20px',
                      'border': f'1px solid {colors["border"]}'})
        ]),

        # Second row: Import sources breakdown
        html.Div([
            html.Div([
                html.H3("Import Sources Breakdown by Year",
                        style={'color': colors['text'], 'marginBottom': '20px'}),

                # Controls for breakdown
                html.Div([
                    html.Div([
                        html.Label("Select Year:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='year-dropdown',
                            options=[{'label': str(year), 'value': year} for year in years],
                            value=years[-1],  # Default to most recent year
                            style={'marginBottom': '10px'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.Label("Chart Type:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.RadioItems(
                            id='chart-key_type-radio',
                            options=[
                                {'label': 'Pie Chart', 'value': 'pie'},
                                {'label': 'Bar Chart', 'value': 'bar'}
                            ],
                            value='pie',
                            inline=True,
                            style={'marginTop': '5px'}
                        )
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),

                # Breakdown chart
                dcc.Graph(id='breakdown-chart')

            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ]),

        # Third row: Summary statistics
        html.Div([
            html.H3("Summary Statistics",
                    style={'color': colors['text'], 'marginBottom': '20px'}),
            html.Div(id='summary-stats')
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginTop': '20px'})

    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 20px'})

], style={'backgroundColor': colors['background'], 'minHeight': '100vh'})


# Callback for time series chart
@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_time_series(selected_country, start_date, end_date):
    # Filter data for selected country and date range
    country_data = df.loc[selected_country].copy()

    # Convert date strings to datetime if needed
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        country_data = country_data[(country_data.index >= start_date) &
                                    (country_data.index <= end_date)]

    # Create time series plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=country_data.index,
        y=country_data['value'],
        mode='lines+markers',
        name=selected_country,
        line=dict(color=colors['primary'], width=3),
        marker=dict(size=6, color=colors['primary'])
    ))

    fig.update_layout(
        title=f"Monthly Import Values - {selected_country}",
        xaxis_title="Date",
        yaxis_title="Import Value",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=colors['text']),
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')

    return fig


# Callback for breakdown chart
@app.callback(
    Output('breakdown-chart', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('chart-key_type-radio', 'value')]
)
def update_breakdown(selected_year, chart_type):
    # Filter data for selected year
    year_data = df[df.index.get_level_values('date').year == selected_year].copy()

    # Group by country and sum values
    country_totals = year_data.groupby('Partner')['value'].sum().sort_values(ascending=False)

    if chart_type == 'pie':
        fig = px.pie(
            values=country_totals.values,
            names=country_totals.index,
            title=f"Import Sources Distribution - {selected_year}"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')

    else:  # bar chart
        fig = px.bar(
            x=country_totals.index,
            y=country_totals.values,
            title=f"Import Values by Country - {selected_year}",
            color=country_totals.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Total Import Value",
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=colors['text'])
    )

    return fig


# Callback for summary statistics
@app.callback(
    Output('summary-stats', 'children'),
    [Input('country-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_summary_stats(selected_country, selected_year):
    # Overall statistics
    total_imports = df['value'].sum()
    avg_monthly = df['value'].mean()

    # Country-specific statistics
    country_data = df.loc[selected_country]
    country_total = country_data['value'].sum()
    country_avg = country_data['value'].mean()

    # Year-specific statistics
    year_data = df[df.index.get_level_values('date').year == selected_year]
    year_total = year_data['value'].sum()
    top_country = year_data.groupby('Partner')['value'].sum().idxmax()

    return html.Div([
        html.Div([
            html.Div([
                html.H4("Overall Statistics", style={'color': colors['primary']}),
                html.P(f"Total Imports: ${total_imports:,.0f}"),
                html.P(f"Average Monthly: ${avg_monthly:,.0f}")
            ], className="four columns"),

            html.Div([
                html.H4(f"{selected_country} Statistics", style={'color': colors['secondary']}),
                html.P(f"Total Imports: ${country_total:,.0f}"),
                html.P(f"Average Monthly: ${country_avg:,.0f}"),
                html.P(f"Share of Total: {(country_total / total_imports) * 100:.1f}%")
            ], className="four columns"),

            html.Div([
                html.H4(f"{selected_year} Statistics", style={'color': colors['accent']}),
                html.P(f"Year Total: ${year_total:,.0f}"),
                html.P(f"Top Source: {top_country}"),
                html.P(f"Number of Sources: {len(year_data.index.get_level_values('Partner').unique())}")
            ], className="four columns")
        ], className="row")
    ])


# Add CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .four.columns { width: 30%; margin: 1.5%; }
            .row { display: flex; justify-content: space-around; }
            body { font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True)


