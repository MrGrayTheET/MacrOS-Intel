import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from components.frames import FundamentalFrame,FlexibleMenu, EnhancedFrameGrid
from data.data_tables import ESRTableClient

table_client = ESRTableClient()

def create_comparative_analysis_layout():
    """Create the ESR Comparative Analysis page layout."""

    # Create two frames for comparison
    frame1_configs = [
        {
            'title': 'Commodity A - Export Metrics',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350
        }
    ]

    frame2_configs = [
        {
            'title': 'Commodity B - Export Metrics',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350
        }
    ]

    comparison_frame1 = FundamentalFrame(
        table_client=table_client,
        chart_configs=frame1_configs,
        div_prefix="esr_comparison_frame1"
    )

    comparison_frame2 = FundamentalFrame(
        table_client=table_client,
        chart_configs=frame2_configs,
        div_prefix="esr_comparison_frame2"
    )

    # Create menu
    comparative_menu = FlexibleMenu('esr_comparative_analysis_menu', position='right', width='320px',
                                    title='Comparative Analysis Controls')

    comparative_menu.add_dropdown('commodity_a', 'Commodity A', [
        {'label': 'Cattle', 'value': 'cattle'},
        {'label': 'Corn', 'value': 'corn'},
        {'label': 'Wheat', 'value': 'wheat'},
        {'label': 'Soybeans', 'value': 'soybeans'}
    ], value='cattle')

    comparative_menu.add_dropdown('commodity_b', 'Commodity B', [
        {'label': 'Cattle', 'value': 'cattle'},
        {'label': 'Corn', 'value': 'corn'},
        {'label': 'Wheat', 'value': 'wheat'},
        {'label': 'Soybeans', 'value': 'soybeans'}
    ], value='corn')

    # Year range controls (replacing single year)
    current_year = pd.Timestamp.now().year
    comparative_menu.add_dropdown('start_year', 'Start Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 1)
    ], value=current_year - 2)

    comparative_menu.add_dropdown('end_year', 'End Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 1)
    ], value=current_year)

    comparative_menu.add_dropdown('metric', 'Metric to Compare', [
        {'label': 'Weekly Exports', 'value': 'weeklyExports'},
        {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
        {'label': 'Gross New Sales', 'value': 'grossNewSales'},
        {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'}
    ], value='weeklyExports')

    # Get dynamic top countries for default commodity A
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

    comparative_menu.add_dropdown('countries', 'Countries to Display', countries_options, value=default_countries, multi=True)

    comparative_menu.add_button('apply', 'Apply Changes')

    # Create enhanced grid
    grid = EnhancedFrameGrid(frames=[comparison_frame1, comparison_frame2], flexible_menu=comparative_menu)

    return grid, grid.generate_layout_with_menu(title="ESR Comparative Analysis")

grid, children = create_comparative_analysis_layout()
# Page layout
layout = html.Div(id="comparative-analysis-page", children=[children])