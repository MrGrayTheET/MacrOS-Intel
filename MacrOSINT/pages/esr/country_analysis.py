
# ===== /pages/esr/esr_country_analysis.py =====
"""
ESR Country Analysis page layout
"""
from dash import html
from data.data_tables import ESRTableClient
from MacrOSINT.components import FundamentalFrame,FlexibleMenu, EnhancedFrameGrid
import pandas as pd

table_client = ESRTableClient()

def create_country_analysis_layout():
    """Create the ESR Country Analysis page layout with dynamic country generation and market year overlays."""

    chart_configs = [
        {
            'title': 'Country Export Performance - Market Year Overlays',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 450
        },
        {
            'title': 'Country Analysis - Current Marketing Year',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'weeklyExports',
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

    # Create menu with enhanced country and market year functionality
    country_menu = FlexibleMenu('esr_country_analysis_menu', position='right', width='350px',
                                title='Country Analysis Controls')

    # Metric selection for analysis
    country_menu.add_dropdown('country_metric', 'Analysis Metric', [
        {'label': 'Weekly Exports', 'value': 'weeklyExports'},
        {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
        {'label': 'Gross New Sales', 'value': 'grossNewSales'},
        {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'},
        {'label': 'Current MY Total Commitment', 'value': 'currentMYTotalCommitment'}
    ], value='weeklyExports')

    # Country selection mode
    country_menu.add_dropdown('country_display_mode', 'Country Display', [
        {'label': 'Individual Countries', 'value': 'individual'},
        {'label': 'Sum Countries', 'value': 'sum'}
    ], value='individual')

    # Multiple countries selection - will be updated dynamically from store
    country_menu.add_dropdown('countries', 'Select Countries', [
        {'label': 'Korea, South', 'value': 'Korea, South'},
        {'label': 'Japan', 'value': 'Japan'},
        {'label': 'China', 'value': 'China'}
    ], value=['Korea, South', 'Japan'], multi=True)

    # Market year range controls for overlays
    current_year = pd.Timestamp.now().year
    country_menu.add_dropdown('start_year', 'Start Marketing Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 2)
    ], value=current_year - 3)

    country_menu.add_dropdown('end_year', 'End Marketing Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 2)
    ], value=current_year)

    # Date range selector
    country_menu.add_date_range_picker('date_range', 'Date Range', 
                                     start_date=None, end_date=None)

    country_menu.add_button('apply', 'Apply Changes')

    # Create enhanced grid with store data source
    grid = EnhancedFrameGrid(
        frames=[country_frame], 
        flexible_menu=country_menu,
        data_source='esr-df-store'  # Use the ESR store from home page
    )

    return grid, grid.generate_layout_with_menu(title="ESR Country Performance Analysis")

grid, children = create_country_analysis_layout()
layout = html.Div(id="country-analysis-page", children=[children])