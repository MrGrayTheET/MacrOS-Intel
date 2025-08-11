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

    # Chart configurations - updated to use store data
    chart_configs = [
        {
            'title': 'Weekly Export Trends',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Outstanding Sales Analysis',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'outstandingSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Gross New Sales Trends',
            'chart_type': 'bar',
            'starting_key': None,  # Will use store data
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

    # Create menu with enhanced controls
    menu_configs = [
        # Column selectors for each chart
        {
            'type': 'dropdown',
            'id': 'chart_0_column',
            'label': 'Chart 1 Metric',
            'options': [
                {'label': 'Weekly Exports', 'value': 'weeklyExports'},
                {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
                {'label': 'Gross New Sales', 'value': 'grossNewSales'},
                {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'},
                {'label': 'Current MY Total Commitment', 'value': 'currentMYTotalCommitment'}
            ],
            'value': 'weeklyExports'
        },
        {
            'type': 'dropdown',
            'id': 'chart_1_column',
            'label': 'Chart 2 Metric',
            'options': [
                {'label': 'Weekly Exports', 'value': 'weeklyExports'},
                {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
                {'label': 'Gross New Sales', 'value': 'grossNewSales'},
                {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'},
                {'label': 'Current MY Total Commitment', 'value': 'currentMYTotalCommitment'}
            ],
            'value': 'outstandingSales'
        },
        {
            'type': 'dropdown',
            'id': 'chart_2_column',
            'label': 'Chart 3 Metric',
            'options': [
                {'label': 'Weekly Exports', 'value': 'weeklyExports'},
                {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
                {'label': 'Gross New Sales', 'value': 'grossNewSales'},
                {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'},
                {'label': 'Current MY Total Commitment', 'value': 'currentMYTotalCommitment'}
            ],
            'value': 'grossNewSales'
        },
        # Country selection with display options
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
        # Date range controls
        {
            'type': 'date_range_picker',
            'id': 'date_range',
            'label': 'Date Range',
            'start_date': (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d'),
            'end_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
    ]

    sales_menu = FlexibleMenu(
        'esr_sales_trends_menu', 
        position='right', 
        width='320px', 
        title='Sales Trends Controls',
        component_configs=menu_configs
    )

    # Create enhanced grid with store data source
    grid = EnhancedFrameGrid(
        frames=[sales_frame], 
        flexible_menu=sales_menu, 
        data_source='esr-df-store'
    )


    return grid, grid.generate_layout_with_menu(title="ESR Sales Trends Analysis")
grid, children = create_sales_trends_layout()

# Page layout for use in routing
layout = html.Div(id="sales-trends-page", children=[children])
