from dash import html
from MacrOSINT.components import FundamentalFrame, FlexibleMenu, EnhancedFrameGrid
from data.data_tables import ESRTableClient

table_client = ESRTableClient()

def create_commitment_analysis_layout():
    """Create the ESR Commitment Analysis page with 2 frames for visual clarity."""

    # Frame 0: MY Commitments/Shipments and Sales Backlog (store-based)
    frame0_chart_configs = [
        {
            'title': 'MY Commitments/Shipments/Sales (Selectable)',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'currentMYTotalCommitment',  # Default, will be updated by menu
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350
        },
        {
            'title': 'Sales Backlog Analysis',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'sales_backlog',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350,
            'use_analytics': True,
            'analytics_function': 'commitment_vs_shipment_analysis'
        }
    ]

    # Frame 1: Commitment Utilization and Fulfillment Rate (analytics-based)
    frame1_chart_configs = [
        {
            'title': 'Commitment Utilization Rate',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'commitment_utilization',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350,
            'use_analytics': True,
            'analytics_function': 'commitment_vs_shipment_analysis'
        },
        {
            'title': 'Export Fulfillment Rate',
            'chart_type': 'line',
            'starting_key': None,  # Will use store data
            'y_column': 'fulfillment_rate',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350,
            'use_analytics': True,
            'analytics_function': 'commitment_vs_shipment_analysis'
        }
    ]

    # Create Frame 0 (MY Commitments/Shipments and Sales Backlog)
    commitment_frame0 = FundamentalFrame(
        table_client=table_client,
        chart_configs=frame0_chart_configs,
        layout="horizontal",
        div_prefix="esr_commitment_frame0",
        width="100%",
        height="750px"
    )

    # Create Frame 1 (Utilization and Fulfillment)
    commitment_frame1 = FundamentalFrame(
        table_client=table_client,
        chart_configs=frame1_chart_configs,
        layout="horizontal",
        div_prefix="esr_commitment_frame1",
        width="100%",
        height="750px"
    )

    # Create menu with store-based controls
    commitment_menu = FlexibleMenu('esr_commitment_analysis_menu', position='right', width='350px',
                                   title='Commitment Analysis Controls')

    # Column selection for Frame 0 Chart 0 (MY Commitments/Shipments/Sales)
    commitment_menu.add_dropdown('commitment_metric', 'Commitment Metric', [
        {'label': 'MY Total Commitment', 'value': 'currentMYTotalCommitment'},
        {'label': 'MY Net Sales', 'value': 'currentMYNetSales'},
        {'label': 'Weekly Exports', 'value': 'weeklyExports'},
        {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
        {'label': 'Gross New Sales', 'value': 'grossNewSales'},
        {'label': 'Next MY Outstanding Sales', 'value': 'nextMYOutstandingSales'}
    ], value='currentMYTotalCommitment')

    # Country selection mode
    commitment_menu.add_dropdown('country_display_mode', 'Country Display', [
        {'label': 'Individual Countries', 'value': 'individual'},
        {'label': 'Sum Countries', 'value': 'sum'}
    ], value='individual')

    # Multiple countries selection - will be updated dynamically from store
    commitment_menu.add_dropdown('countries', 'Select Countries', [
        {'label': 'Korea, South', 'value': 'Korea, South'},
        {'label': 'Japan', 'value': 'Japan'},
        {'label': 'China', 'value': 'China'}
    ], value=['Korea, South', 'Japan'], multi=True)

    # Date range selector
    commitment_menu.add_date_range_picker('date_range', 'Date Range', 
                                        start_date=None, end_date=None)

    commitment_menu.add_button('apply', 'Apply Changes')

    # Create enhanced grid with both frames and store data source
    grid = EnhancedFrameGrid(
        frames=[commitment_frame0, commitment_frame1],

        flexible_menu=commitment_menu,
        data_source='esr-df-store'  # Use the ESR store from home page
    )

    return grid, grid.generate_layout_with_menu(title="ESR Commitment Analysis")

grid, children = create_commitment_analysis_layout()
# Page layout
layout = html.Div(id="commitment-analysis-page", children=[children])