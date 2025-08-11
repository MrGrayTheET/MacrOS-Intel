import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from components.frames import FundamentalFrame, FlexibleMenu, EnhancedFrameGrid
from data.data_tables import ESRTableClient
import dash_ag_grid as dag

table_client = ESRTableClient()

def create_seasonal_analysis_layout(data_source=None):
    """Create the ESR Seasonal Analysis page layout.
    
    Args:
        data_source: Store ID to read data from (optional)
    """

    # Chart configurations for seasonal patterns - updated to use store data
    chart_configs = [
        {
            'title': 'Seasonal Export Patterns',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Average Seasonal Pattern',
            'chart_type': 'area',
            'starting_key': None,  # Will use store data
            'y_column': 'weeklyExports',
            'x_column': 'week_of_year',
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
        # Country selection with multi-select capability
        {
            'type': 'checklist',
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
            'value': ['Korea, South', 'Japan', 'China']
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
        # Market year overlay selection
        {
            'type': 'dropdown',
            'id': 'overlay_market_year',
            'label': 'Overlay Market Year',
            'options': [
                {'label': str(year), 'value': year}
                for year in range(pd.Timestamp.now().year - 5, pd.Timestamp.now().year + 1)
            ],
            'value': pd.Timestamp.now().year
        },
        # Difference analysis selector
        {
            'type': 'dropdown',
            'id': 'difference_from_year',
            'label': 'Show Difference From Year',
            'options': [
                {'label': str(year), 'value': year}
                for year in range(pd.Timestamp.now().year - 5, pd.Timestamp.now().year + 1)
            ],
            'value': None
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

    # Create layout with seasonal summary table
    main_layout = grid.generate_layout_with_menu(title="ESR Seasonal Analysis")
    
    # Add seasonal summary table below the charts
    seasonal_table = dag.AgGrid(
        id="esr_seasonal_analysis_table_0",
        columnDefs=[
            {"headerName": "Commodity", "field": "commodity", "width": 120},
            {"headerName": "Marketing Year", "field": "my_start", "width": 100},
            {"headerName": "Peak Weeks", "field": "peak_weeks", "width": 120},
            {"headerName": "Low Weeks", "field": "low_weeks", "width": 120},
            {"headerName": "Seasonality Strength", "field": "seasonality", "width": 140, "type": "numericColumn"},
        ],
        rowData=[],
        style={"height": "200px"},
        className="ag-theme-alpine-dark"
    )

    # Combine main layout with table
    combined_layout = html.Div([
        main_layout,
        html.Hr(style={'margin': '30px 0', 'borderColor': '#444'}),
        html.H4("Seasonal Summary", style={'color': '#e8e8e8', 'textAlign': 'center', 'marginBottom': '20px'}),
        seasonal_table
    ])

    return grid, combined_layout

# Create default instances
grid, layout = create_seasonal_analysis_layout()

# Page layout for use in routing
page_layout = html.Div(id="seasonal-analysis-page", children=[layout])