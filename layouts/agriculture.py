#layouts/agriculture.py
from datetime import date

# Fixed layout - ensure component IDs match callback expectations

from datetime import date
from dash import html, dcc
import dash_bootstrap_components as dbc


def import_export_layout(data_type: str = "imports", data_source=None):
    """Generate layout for different data types (imports/exports)"""

    from data.data_tables import FASTable
    table_client = FASTable()

    # Get initial data and available commodities
    available_keys = table_client.available_keys()
    data_keys = [key for key in available_keys if key.endswith(f'{data_type}')]
    available_commodities = [key.split('/')[0] for key in data_keys]

    if not available_commodities:
        available_commodities = ['wheat', 'corn', 'rice']

    default_commodity = available_commodities[0]

    # Use provided data_source or get default data
    if data_source is not None:
        df = data_source
    else:
        df = table_client[f'{default_commodity}/{data_type}']

    # Extract unique countries and years for dropdowns
    countries = df.index.get_level_values('Partner').unique().tolist()
    years = df.index.get_level_values('date').year.unique().tolist()
    years.sort()

    return html.Div([
        # Header
        html.Div([
            html.H1(f"{data_type.title()} Sources Analysis Dashboard",
                    className='dashboard-title'),
            html.P(f"Analyze commodity {data_type} patterns by country and time period",
                   className='dashboard-subtitle'),

            # Commodity selector
            html.Div([
                html.Label("Select Commodity:", className='form-label'),
                dcc.Dropdown(
                    id='commodity-dropdown',  # ✓ Matches callback
                    options=[{'label': commodity.title(), 'value': commodity}
                             for commodity in available_commodities],
                    value=available_commodities[0],
                    className='commodity-dropdown dark-dropdown'
                ),
            ], className='commodity-selector-container'),

            # Store components
            dcc.Store(id='data-key-type', data={'key-type': data_type}),  # ✓ Matches callback
            dcc.Store(id='trade-data'),  # ✓ Matches callback

        ], className='dashboard-header'),

        # Main content container
        html.Div([
            # First row: Time series analysis
            html.Div([
                html.Div([
                    html.H3(f"Monthly {data_type.title()} Trends by Country",
                            className='section-title'),

                    # Controls for time series
                    html.Div([
                        html.Div([
                            html.Label("Select Country:", className='form-label'),
                            dcc.Dropdown(
                                id='country-dropdown',  # ✓ Matches callback
                                options=[{'label': country, 'value': country}
                                         for country in countries],
                                value=countries[0] if countries else None,
                                className='control-dropdown dark-dropdown'
                            )
                        ], className='control-column-left'),

                        html.Div([
                            html.Label("Select Date Range:", className='form-label'),
                            dcc.DatePickerRange(
                                id='date-range-picker-trade',  # ✓ Fixed to match callback
                                start_date=date(2020, 1, 1),
                                end_date=date(2024, 12, 31),
                                display_format='YYYY-MM-DD',
                                className='date-picker dark-datepicker'
                            )
                        ], className='control-column-right')
                    ], className='controls-row'),

                    # Time series chart
                    dcc.Graph(id='time-series-chart')  # ✓ Matches callback

                ], className='chart-card')
            ]),

            # Second row: Data sources breakdown
            html.Div([
                html.Div([
                    html.H3(f"{data_type.title()} Sources Breakdown by Year",
                            className='section-title'),

                    # Controls for breakdown
                    html.Div([
                        html.Div([
                            html.Label("Select Year:", className='form-label'),
                            dcc.Dropdown(
                                id='year-dropdown',  # ✓ Matches callback
                                options=[{'label': str(year), 'value': year}
                                         for year in years],
                                value=years[-1] if years else None,
                                className='control-dropdown dark-dropdown'
                            )
                        ], className='control-column-left'),

                        html.Div([
                            html.Label("Chart Type:", className='form-label'),
                            dcc.RadioItems(
                                id='chart-type-radio',  # ✓ Fixed ID to match callback
                                options=[
                                    {'label': 'Pie Chart', 'value': 'pie'},
                                    {'label': 'Bar Chart', 'value': 'bar'}
                                ],
                                value='pie',
                                inline=True,
                                className='radio-group dark-radio'
                            )
                        ], className='control-column-right')
                    ], className='controls-row'),

                    # Breakdown chart
                    dcc.Graph(id='breakdown-chart')  # ✓ Matches callback

                ], className='chart-card')
            ]),

            # Third row: Summary statistics
            html.Div([
                html.H3("Summary Statistics", className='section-title'),
                html.Div(id='summary-stats')  # ✓ Matches callback
            ], className='stats-card')

        ], className='main-content'),

    ], className='dashboard-container')

def st_exports(commodity, year, trade_type='exports'):
    menu_config = {
        'enabled': True,
    }

    return
