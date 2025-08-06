# ===== /pages/esr/esr_sales_trends.py =====
"""
ESR Sales Trends page layout
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from components.frames import FundamentalFrame
from components.frames import FlexibleMenu, EnhancedFrameGrid
from data.data_tables import ESRTableClient
import pandas as pd

table_client = ESRTableClient()

def create_sales_trends_layout():
    """Create the ESR Sales Trends page layout."""

    # Chart configurations
    chart_configs = [
        {
            'title': 'Weekly Export Trends',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/2024',
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Outstanding Sales Analysis',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/2024',
            'y_column': 'outstandingSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Gross New Sales Trends',
            'chart_type': 'bar',
            'starting_key': 'cattle/exports/2024',
            'y_column': 'grossNewSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        }
    ]

    # Create frame
    sales_frame = FundamentalFrame(
        table_client=table_client,
        chart_configs=chart_configs,
        layout="horizontal",
        div_prefix="esr_sales_trends",
        width="100%",
        height="1200px"
    )

    # Create menu
    sales_menu = FlexibleMenu('esr_sales_trends_menu', position='right', width='300px', title='Sales Trends Controls')

    sales_menu.add_dropdown('commodity', 'Commodity', [
        {'label': 'Cattle', 'value': 'cattle'},
        {'label': 'Corn', 'value': 'corn'},
        {'label': 'Wheat', 'value': 'wheat'},
        {'label': 'Soybeans', 'value': 'soybeans'}
    ], value='cattle')

    # Year range controls
    current_year = pd.Timestamp.now().year
    sales_menu.add_dropdown('start_year', 'Start Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 1)
    ], value=current_year - 2)

    sales_menu.add_dropdown('end_year', 'End Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 1)
    ], value=current_year)

    # Get dynamic top countries for default commodity
    try:
        top_countries = table_client.get_top_countries('cattle', top_n=10)
        countries_options = [{'label': country, 'value': country} for country in top_countries]
        default_countries = top_countries[:3] if len(top_countries) >= 3 else top_countries
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
        default_countries = ['Korea, South', 'Japan', 'China']

    sales_menu.add_checklist('countries', 'Countries to Display', countries_options, value=default_countries)

    sales_menu.add_button('apply', 'Apply Changes')

    # Create enhanced grid
    grid = EnhancedFrameGrid(frames=[sales_frame], flexible_menu=sales_menu)


    return grid, grid.generate_layout_with_menu(title="ESR Sales Trends Analysis")
grid, children = create_sales_trends_layout()

# Page layout for use in routing
layout = html.Div(id="sales-trends-page", children=[children])
