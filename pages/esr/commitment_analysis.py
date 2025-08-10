import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from components.frames import FundamentalFrame, FlexibleMenu, EnhancedFrameGrid
from data.data_tables import ESRTableClient

table_client = ESRTableClient()

def create_commitment_analysis_layout():
    """Create the ESR Commitment Analysis page with 2 frames for visual clarity."""

    # Frame 0: MY Commitments/Shipments and Sales Backlog
    frame0_chart_configs = [
        {
            'title': 'MY Commitments/Shipments/Sales (Selectable)',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
            'y_column': 'currentMYTotalCommitment',  # Default, will be updated by menu
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350
        },
        {
            'title': 'Sales Backlog Analysis',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
            'y_column': 'sales_backlog',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 350,
            'use_analytics': True,
            'analytics_function': 'commitment_vs_shipment_analysis'
        }
    ]

    # Frame 1: Commitment Utilization and Fulfillment Rate
    frame1_chart_configs = [
        {
            'title': 'Commitment Utilization Rate',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
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
            'starting_key': 'cattle/exports/all',
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

    # Create menu
    commitment_menu = FlexibleMenu('esr_commitment_analysis_menu', position='right', width='320px',
                                   title='Commitment Analysis Controls')

    commitment_menu.add_dropdown('commodity', 'Commodity', [
        {'label': 'Cattle', 'value': 'cattle'},
        {'label': 'Corn', 'value': 'corn'},
        {'label': 'Wheat', 'value': 'wheat'},
        {'label': 'Soybeans', 'value': 'soybeans'}
    ], value='cattle')

    # Column selection for Frame 0 Chart 0 (MY Commitments/Shipments/Sales)
    commitment_menu.add_dropdown('commitment_metric', 'Commitment Metric', [
        {'label': 'MY Total Commitment', 'value': 'currentMYTotalCommitment'},
        {'label': 'MY Net Sales', 'value': 'currentMYNetSales'},
        {'label': 'Weekly Exports', 'value': 'weeklyExports'},
        {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
        {'label': 'Gross New Sales', 'value': 'grossNewSales'},
        {'label': 'Next MY Outstanding Sales', 'value': 'nextMYOutstandingSales'}
    ], value='currentMYTotalCommitment')

    # Year range controls 
    current_year = pd.Timestamp.now().year
    commitment_menu.add_dropdown('start_year', 'Start Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 2)
    ], value=current_year - 2)

    commitment_menu.add_dropdown('end_year', 'End Year', [
        {'label': str(year), 'value': year}
        for year in range(current_year - 10, current_year + 2)
    ], value=current_year)

    # Country selection for analysis
    try:
        top_countries = table_client.get_top_countries('cattle', top_n=10)
        countries_options = [{'label': country, 'value': country} for country in top_countries]
        default_country = top_countries[0] if top_countries else 'Korea, South'
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

    # Single country selection for commitment analysis
    commitment_menu.add_dropdown('country_selection', 'Country Analysis', countries_options, value=default_country)

    # Multiple countries for aggregated view
    default_countries = [default_country]
    if len(countries_options) >= 3:
        default_countries = [opt['value'] for opt in countries_options[:3]]
    
    commitment_menu.add_checklist('countries', 'Countries (Aggregated)', countries_options, value=default_countries)

    commitment_menu.add_button('apply', 'Apply Changes')

    # Create enhanced grid with both frames
    grid = EnhancedFrameGrid(frames=[commitment_frame0, commitment_frame1], flexible_menu=commitment_menu)

    return grid, grid.generate_layout_with_menu(title="ESR Commitment Analysis")

grid, children = create_commitment_analysis_layout()
# Page layout
layout = html.Div(id="commitment-analysis-page", children=[children])