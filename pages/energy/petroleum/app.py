import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    # Navigation bar
    html.Div([
        html.H3("Petroleum Market Research Dashboard",
                style={'text-align': 'center', 'margin': '0', 'padding': '15px', 'color': 'white'})
    ], style={
        'background-color': '#2c3e50',
        'height': '60px',
        'border-bottom': '2px solid #34495e'
    }),

    # Main content area
    html.Div([
        # Left sidebar
        html.Div([
            # Current Page menu section
            html.Div([
                html.H4("Current Page menu and settings",
                        style={'text-align': 'center', 'margin': '10px 0', 'font-size': '14px'}),
                html.Hr(),
                # Add dropdown for commodity selection
                dcc.Dropdown(
                    id='commodity-dropdown',
                    options=[
                        {'label': 'Crude Oil', 'value': 'crude_oil'},
                        {'label': 'Natural Gas', 'value': 'natural_gas'},
                        {'label': 'Gasoline', 'value': 'gasoline'},
                        {'label': 'Diesel', 'value': 'diesel'}
                    ],
                    value='crude_oil',
                    style={'margin': '10px 0'}
                ),
                # Add date range picker
                dcc.DatePickerRange(
                    id='date-picker-range',
                    start_date='2023-01-01',
                    end_date='2024-01-01',
                    style={'margin': '10px 0'}
                ),
                # Add settings checkboxes
                dcc.Checklist(
                    id='settings-checklist',
                    options=[
                        {'label': 'Show trend lines', 'value': 'trend'},
                        {'label': 'Normalize data', 'value': 'normalize'},
                        {'label': 'Show volatility', 'value': 'volatility'}
                    ],
                    value=['trend'],
                    style={'margin': '10px 0'}
                )
            ], style={
                'height': '350px',
                'border': '2px solid #34495e',
                'padding': '10px',
                'background-color': '#ecf0f1'
            }),

            # Charts menu section
            html.Div([
                html.H4("Charts menu",
                        style={'text-align': 'center', 'margin': '10px 0', 'font-size': '14px'}),
                html.Hr(),
                # Add chart key_type selection
                dcc.RadioItems(
                    id='chart-key_type-radio',
                    options=[
                        {'label': 'Line Chart', 'value': 'line'},
                        {'label': 'Candlestick', 'value': 'candlestick'},
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Area Chart', 'value': 'area'}
                    ],
                    value='line',
                    style={'margin': '10px 0'}
                ),
                # Add time interval selection
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[
                        {'label': '1 Day', 'value': '1D'},
                        {'label': '1 Week', 'value': '1W'},
                        {'label': '1 Month', 'value': '1M'},
                        {'label': '3 Months', 'value': '3M'},
                        {'label': '1 Year', 'value': '1Y'}
                    ],
                    value='1M',
                    style={'margin': '10px 0'}
                )
            ], style={
                'height': '350px',
                'border': '2px solid #34495e',
                'padding': '10px',
                'background-color': '#ecf0f1',
                'margin-top': '10px'
            })
        ], style={
            'width': '18%',
            'float': 'left',
            'margin': '10px 5px'
        }),

        # Main content area with charts
        html.Div([
            # Top row - Market data chart and Custom charts
            html.Div([
                # Market data chart
                html.Div([
                    html.H4("Market data chart",
                            style={'text-align': 'center', 'margin': '10px 0'}),
                    dcc.Graph(
                        id='market-data-chart',
                        figure=go.Figure().add_trace(
                            go.Scatter(x=[], y=[], mode='lines', name='Market Data')
                        ).update_layout(
                            title="Market Data Placeholder",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=300
                        )
                    )
                ], style={
                    'width': '49%',
                    'float': 'left',
                    'border': '2px solid #34495e',
                    'padding': '10px',
                    'margin': '5px 1%'
                }),

                # Custom charts
                html.Div([
                    html.H4("Custom charts (market)",
                            style={'text-align': 'center', 'margin': '10px 0'}),
                    dcc.Graph(
                        id='custom-market-chart',
                        figure=go.Figure().add_trace(
                            go.Scatter(x=[], y=[], mode='lines', name='Custom Market')
                        ).update_layout(
                            title="Custom Market Chart Placeholder",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            height=300
                        )
                    )
                ], style={
                    'width': '49%',
                    'float': 'right',
                    'border': '2px solid #34495e',
                    'padding': '10px',
                    'margin': '5px 1%'
                })
            ], style={
                'height': '400px',
                'overflow': 'hidden'
            }),

            # Bottom row - Supply/Demand/Production charts
            html.Div([
                # Left Supply/Demand/Production chart
                html.Div([
                    html.H4("Supply/Demand/Production chart",
                            style={'text-align': 'center', 'margin': '10px 0'}),
                    dcc.Graph(
                        id='supply-demand-chart-1',
                        figure=go.Figure().add_trace(
                            go.Bar(x=[], y=[], name='Supply/Demand')
                        ).update_layout(
                            title="Supply/Demand/Production Placeholder",
                            xaxis_title="Period",
                            yaxis_title="Volume (barrels)",
                            height=300
                        )
                    )
                ], style={
                    'width': '49%',
                    'float': 'left',
                    'border': '2px solid #34495e',
                    'padding': '10px',
                    'margin': '5px 1%'
                }),

                # Right Supply/Demand/Production chart
                html.Div([
                    html.H4("Supply/Demand/Production chart",
                            style={'text-align': 'center', 'margin': '10px 0'}),
                    dcc.Graph(
                        id='supply-demand-chart-2',
                        figure=go.Figure().add_trace(
                            go.Bar(x=[], y=[], name='Production')
                        ).update_layout(
                            title="Supply/Demand/Production Placeholder",
                            xaxis_title="Period",
                            yaxis_title="Volume (barrels)",
                            height=300
                        )
                    )
                ], style={
                    'width': '49%',
                    'float': 'right',
                    'border': '2px solid #34495e',
                    'padding': '10px',
                    'margin': '5px 1%'
                })
            ], style={
                'height': '400px',
                'overflow': 'hidden',
                'clear': 'both'
            })
        ], style={
            'width': '52%',
            'float': 'left',
            'margin': '10px 5px'
        }),

        # Right sidebar - Infotable tabs
        html.Div([
            # Infotable tabs section
            html.Div([
                html.H4("Infotable tabs",
                        style={'text-align': 'center', 'margin': '10px 0', 'font-size': '14px'}),
                dcc.Tabs(
                    id='infotable-tabs',
                    value='seasonal-stats',
                    children=[
                        dcc.Tab(label='Seasonal Stats', value='seasonal-stats'),
                        dcc.Tab(label='Data Release Info', value='data-release'),
                        dcc.Tab(label='Infotables', value='infotables')
                    ]
                ),
                html.Div(id='infotable-content', style={'padding': '20px'})
            ], style={
                'height': '350px',
                'border': '2px solid #34495e',
                'padding': '10px',
                'background-color': '#ecf0f1'
            }),

            # Infotable #2 section
            html.Div([
                html.H4("Infotable #2",
                        style={'text-align': 'center', 'margin': '10px 0', 'font-size': '14px'}),
                html.Div([
                    html.P("Additional market insights and data tables will be displayed here.",
                           style={'text-align': 'center', 'color': '#7f8c8d'})
                ], style={'padding': '20px'})
            ], style={
                'height': '350px',
                'border': '2px solid #34495e',
                'padding': '10px',
                'background-color': '#ecf0f1',
                'margin-top': '10px'
            })
        ], style={
            'width': '25%',
            'float': 'right',
            'margin': '10px 5px'
        })
    ], style={
        'background-color': '#ffffff',
        'min-height': '100vh',
        'padding': '0',
        'overflow': 'hidden'
    }),

    # Clear floats
    html.Div(style={'clear': 'both'})
])


# Callback for infotable tabs content
@callback(
    Output('infotable-content', 'children'),
    Input('infotable-tabs', 'value')
)
def render_infotable_content(active_tab):
    if active_tab == 'seasonal-stats':
        return html.Div([
            html.H5("Seasonal Statistics"),
            html.P("• Q1 Average: $72.50/barrel"),
            html.P("• Q2 Average: $78.25/barrel"),
            html.P("• Q3 Average: $81.75/barrel"),
            html.P("• Q4 Average: $69.80/barrel"),
            html.P("• Peak Season: Summer"),
            html.P("• Low Season: Winter")
        ])
    elif active_tab == 'data-release':
        return html.Div([
            html.H5("Data Release Information"),
            html.P("• EIA Weekly Release: Wednesday 10:30 AM ET"),
            html.P("• IEA Monthly Report: 12th of each month"),
            html.P("• OPEC Monthly Report: 15th of each month"),
            html.P("• Baker Hughes Rig Count: Friday"),
            html.P("• API Inventory: Tuesday 4:30 PM ET")
        ])
    elif active_tab == 'infotables':
        return html.Div([
            html.H5("Market Infotables"),
            html.P("• Current WTI Price: $75.45"),
            html.P("• Brent Crude: $78.92"),
            html.P("• US Production: 12.8M bbl/day"),
            html.P("• Strategic Reserve: 368M barrels"),
            html.P("• Refinery Utilization: 89.2%")
        ])


# Callback for updating charts based on commodity selection
@callback(
    [Output('market-data-chart', 'figure'),
     Output('custom-market-chart', 'figure'),
     Output('supply-demand-chart-1', 'figure'),
     Output('supply-demand-chart-2', 'figure')],
    [Input('commodity-dropdown', 'value'),
     Input('chart-key_type-radio', 'value'),
     Input('timeframe-dropdown', 'value')]
)
def update_charts(commodity, chart_type, timeframe):
    # Placeholder data - replace with actual data later
    import pandas as pd
    import numpy as np

    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = np.cumsum(np.random.randn(100)) + 75

    # Market data chart
    market_fig = go.Figure()
    if chart_type == 'line':
        market_fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name=f'{commodity} Price'))
    elif chart_type == 'bar':
        market_fig.add_trace(go.Bar(x=dates, y=prices, name=f'{commodity} Price'))

    market_fig.update_layout(
        title=f"{commodity.replace('_', ' ').title()} Market Data ({timeframe})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=300
    )

    # Custom market chart
    custom_fig = go.Figure()
    custom_fig.add_trace(go.Scatter(x=dates, y=prices * 1.1, mode='lines', name='Custom Indicator'))
    custom_fig.update_layout(
        title="Custom Market Analysis",
        xaxis_title="Date",
        yaxis_title="Custom Value",
        height=300
    )

    # Supply/Demand charts
    supply_data = np.random.randint(80, 120, 12)
    demand_data = np.random.randint(85, 125, 12)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    supply_fig = go.Figure()
    supply_fig.add_trace(go.Bar(x=months, y=supply_data, name='Supply', marker_color='lightblue'))
    supply_fig.add_trace(go.Bar(x=months, y=demand_data, name='Demand', marker_color='lightcoral'))
    supply_fig.update_layout(
        title="Supply vs Demand",
        xaxis_title="Month",
        yaxis_title="Volume (Million barrels)",
        height=300,
        barmode='group'
    )

    production_data = np.random.randint(75, 115, 12)
    production_fig = go.Figure()
    production_fig.add_trace(go.Bar(x=months, y=production_data, name='Production', marker_color='lightgreen'))
    production_fig.update_layout(
        title="Production Levels",
        xaxis_title="Month",
        yaxis_title="Volume (Million barrels)",
        height=300
    )

    return market_fig, custom_fig, supply_fig, production_fig


if __name__ == '__main__':
    app.run_server(debug=True)