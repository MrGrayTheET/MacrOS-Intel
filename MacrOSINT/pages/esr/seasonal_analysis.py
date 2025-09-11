from dash import html
import pandas as pd
from MacrOSINT.components import FundamentalFrame, FlexibleMenu, EnhancedFrameGrid
from data.data_tables import ESRTableClient

table_client = ESRTableClient()

def create_seasonal_analysis_layout(data_source=None):
    """Create the ESR Seasonal Analysis page layout.
    
    Args:
        data_source: Store ID to read data from (optional)
    """

    # Chart configurations for seasonal patterns - updated to use store data
    chart_configs = [
        {
            'title': 'Multi-Year Seasonal Overlay',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'weeklyExports',
            'x_column': 'my_week',
            'width': '100%',
            'height': 400,
            'use_analytics': True,
            'analytics_function': 'seasonal_overlay_analysis'
        },
        {
            'title': 'Detailed Seasonal Analysis',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'weeklyExports',
            'x_column': 'my_week',
            'width': '100%',
            'height': 400,
            'use_analytics': True,
            'analytics_function': 'seasonal_pattern_analysis'
        }
    ]

    # Create frame for charts
    seasonal_frame = FundamentalFrame(
        table_client=table_client,
        chart_configs=chart_configs,
        layout="horizontal",
        div_prefix="esr_seasonal_analysis",
        width="100%",
        height="850px"
    )

    # Create menu with enhanced controls using FlexibleMenu component configs
    menu_configs = [
        # Seasonal metric selection
        {
            'type': 'dropdown',
            'id': 'seasonal_metric',
            'label': 'Seasonal Metric',
            'options': [
                {'label': 'Weekly Exports', 'value': 'weeklyExports'},
                {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
                {'label': 'Gross New Sales', 'value': 'grossNewSales'},
                {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'},
                {'label': 'Current MY Total Commitment', 'value': 'currentMYTotalCommitment'}
            ],
            'value': 'weeklyExports'
        },
        # Country selection with multi-select dropdown
        {
            'type': 'dropdown',
            'id': 'countries',
            'label': 'Select Countries',
            'options': [
                {'label': 'Korea, South', 'value': 'Korea, South'},
                {'label': 'Japan', 'value': 'Japan'},
                {'label': 'China', 'value': 'China'},
                {'label': 'Mexico', 'value': 'Mexico'},
                {'label': 'Canada', 'value': 'Canada'},
                {'label': 'Taiwan', 'value': 'Taiwan'}
            ],
            'value': ['Korea, South', 'Japan', 'China'],
            'multi': True
        },
        # Country display mode
        {
            'type': 'radio_items',
            'id': 'country_display_mode',
            'label': 'Country Display Mode',
            'options': [
                {'label': 'Individual Countries', 'value': 'individual'},
                {'label': 'Sum All Selected', 'value': 'sum'}
            ],
            'value': 'individual'
        },
        # Selected market year for detailed analysis (bottom chart)
        {
            'type': 'dropdown',
            'id': 'selected_market_year',
            'label': 'Market Year for Analysis',
            'options': [
                {'label': str(year), 'value': year}
                for year in range(pd.Timestamp.now().year - 5, pd.Timestamp.now().year + 1)
            ],
            'value': pd.Timestamp.now().year
        },
        # Year range controls for seasonal analysis
        {
            'type': 'dropdown',
            'id': 'start_year',
            'label': 'Start Year',
            'options': [
                {'label': str(year), 'value': year}
                for year in range(pd.Timestamp.now().year - 10, pd.Timestamp.now().year + 2)
            ],
            'value': pd.Timestamp.now().year - 4
        },
        {
            'type': 'dropdown',
            'id': 'end_year',
            'label': 'End Year',
            'options': [
                {'label': str(year), 'value': year}
                for year in range(pd.Timestamp.now().year - 10, pd.Timestamp.now().year + 2)
            ],
            'value': pd.Timestamp.now().year
        },
        # Date range controls
        {
            'type': 'date_range_picker',
            'id': 'date_range',
            'label': 'Date Range',
            'start_date': (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d'),
            'end_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
    ]

    seasonal_menu = FlexibleMenu(
        'esr_seasonal_analysis_menu', 
        position='right', 
        width='320px',
        title='Seasonal Analysis Controls',
        component_configs=menu_configs
    )

    # Create enhanced grid with store data source
    grid = EnhancedFrameGrid(
        frames=[seasonal_frame], 
        flexible_menu=seasonal_menu,
        data_source='esr-df-store'  # Use the ESR store from home page
    )

    # Create layout without seasonal summary table
    main_layout = grid.generate_layout_with_menu(title="ESR Seasonal Analysis")

    return grid, main_layout

# Create default instances
grid, layout = create_seasonal_analysis_layout()

# Page layout for use in routing
page_layout = html.Div(id="seasonal-analysis-page", children=[layout])