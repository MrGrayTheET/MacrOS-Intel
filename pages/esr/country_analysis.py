
# ===== /pages/esr/esr_country_analysis.py =====
"""
ESR Country Analysis page layout
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from data.data_tables import ESRTableClient
from components.frames import FundamentalFrame,FlexibleMenu, EnhancedFrameGrid
import pandas as pd

table_client = ESRTableClient()

def create_country_analysis_layout():
    """Create the ESR Country Analysis page layout."""

    chart_configs = [
        {
            'title': 'Country Export Performance (5-Year)',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/2024',
            'y_columns': ['weeklyExports'],
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 450
        },
        {
            'title': 'Outstanding Sales Trend (5-Year)',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/2024',
            'y_columns': ['outstandingSales'],
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 450
        }
    ]

    country_frame = FundamentalFrame(
        table_client=table_client,
        chart_configs=chart_configs,
        layout="horizontal",
        div_prefix="esr_country_analysis",
        width="100%",
        height="900px"
    )

    # Create menu
    country_menu = FlexibleMenu('esr_country_analysis_menu', position='right', width='300px',
                                title='Country Analysis Controls')

    country_menu.add_dropdown('commodity', 'Commodity', [
        {'label': 'Cattle', 'value': 'cattle'},
        {'label': 'Corn', 'value': 'corn'},
        {'label': 'Wheat', 'value': 'wheat'},
        {'label': 'Soybeans', 'value': 'soybeans'}
    ], value='cattle')

    # Year range controls
    current_year = pd.Timestamp.now().year
    country_menu.add_dropdown('start_year', 'Start Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 1)
    ], value=current_year - 4)

    country_menu.add_dropdown('end_year', 'End Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 1)
    ], value=current_year)

    # Get all countries available in the data
    try:
        all_countries = table_client.get_available_countries('cattle')  # Get all countries
        countries_options = [{'label': country, 'value': country} for country in all_countries]
        default_country = all_countries[0] if all_countries else 'Korea, South'
    except:
        # Fallback if dynamic loading fails
        countries_options = [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Mexico', 'value': 'Mexico'},
            {'label': 'Canada', 'value': 'Canada'},
            {'label': 'Taiwan', 'value': 'Taiwan'}
        ]
        default_country = 'Korea, South'

    country_menu.add_dropdown('country', 'Select Country', countries_options, value=default_country)

    country_menu.add_button('apply', 'Apply Changes')

    # Create enhanced grid
    grid = EnhancedFrameGrid(frames=[country_frame], flexible_menu=country_menu)

    return grid, grid.generate_layout_with_menu(title="ESR Country Performance Analysis")

grid, children = create_country_analysis_layout()
layout = html.Div(id="country-analysis-page", children=[children])