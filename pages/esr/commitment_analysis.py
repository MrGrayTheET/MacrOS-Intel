import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from components.frames import FundamentalFrame,FlexibleMenu, EnhancedFrameGrid
from data.data_tables import ESRTableClient

table_client = ESRTableClient()

def create_commitment_analysis_layout():
    """Create the ESR Commitment Analysis page layout."""

    chart_configs = [
        {
            'title': 'Current MY Total Commitment',
            'chart_type': 'area',
            'starting_key': 'cattle/exports/2024',
            'y_column': 'currentMYTotalCommitment',
            'x_column': 'weekEndingDate',
            'width': '49%',
            'height': 400
        },
        {
            'title': 'Current MY Net Sales',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/2024',
            'y_column': 'currentMYNetSales',
            'x_column': 'weekEndingDate',
            'width': '49%',
            'height': 400
        },
        {
            'title': 'Next MY Outstanding Sales',
            'chart_type': 'bar',
            'starting_key': 'cattle/exports/2024',
            'y_column': 'nextMYOutstandingSales',
            'x_column': 'weekEndingDate',
            'width': '49%',
            'height': 400
        },
        {
            'title': 'Next MY Net Sales',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/2024',
            'y_column': 'nextMYNetSales',
            'x_column': 'weekEndingDate',
            'width': '49%',
            'height': 400
        }
    ]

    commitment_frame = FundamentalFrame(
        table_client=table_client,
        chart_configs=chart_configs,
        layout="vertical",
        div_prefix="esr_commitment_analysis",
        width="100%",
        height="850px"
    )

    # Create menu
    commitment_menu = FlexibleMenu('esr_commitment_analysis_menu', position='right', width='300px',
                                   title='Commitment Analysis Controls')

    commitment_menu.add_dropdown('commodity', 'Commodity', [
        {'label': 'Cattle', 'value': 'cattle'},
        {'label': 'Corn', 'value': 'corn'},
        {'label': 'Wheat', 'value': 'wheat'},
        {'label': 'Soybeans', 'value': 'soybeans'}
    ], value='cattle')

    # Year range controls (replacing single year)
    current_year = pd.Timestamp.now().year
    commitment_menu.add_dropdown('start_year', 'Start Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 1)
    ], value=current_year - 2)

    commitment_menu.add_dropdown('end_year', 'End Year', [
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

    commitment_menu.add_checklist('countries', 'Countries to Display', countries_options, value=default_countries)

    commitment_menu.add_button('apply', 'Apply Changes')

    # Create enhanced grid
    grid = EnhancedFrameGrid(frames=[commitment_frame], flexible_menu=commitment_menu)

    return grid, grid.generate_layout_with_menu(title="ESR Commitment Analysis")

grid, children = create_commitment_analysis_layout()
# Page layout
layout = html.Div(id="commitment-analysis-page", children=[children])