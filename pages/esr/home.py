import dash
from dash import html, Input, Output, State, dcc, callback
import dash_bootstrap_components as dbc
import pandas as pd
import json
from .sales_trends import layout as sales_trends_layout, grid as sales_grid
from .country_analysis import layout as country_analysis_layout, grid as country_grid
from .commitment_analysis import layout as commitment_analysis_layout, grid as commitment_grid
from .comparative_analysis import layout as comparative_analysis_layout, grid as comparative_grid
from .seasonal_analysis import layout as seasonal_analysis_layout, grid as seasonal_grid
from data.data_tables import ESRTableClient
from callbacks.esr import (
    sales_trends_chart_update,
    country_analysis_chart_update, 
    comparative_analysis_chart_update,
    unified_seasonal_analysis_update,
    commitment_metric_chart,
    commitment_analytics_chart
)

dash.register_page(__name__, path='/esr', path_template='/esr/<page_name>', name='ESR Dashboard')
app = dash.get_app()

layout_dict = {
    'sales_trends': {
        'layout': sales_trends_layout,
        'grid': sales_grid,
        'callback_fn': sales_trends_chart_update
    },
    'country_analysis': {
        'layout': country_analysis_layout,
        'grid': country_grid,
        'callback_fn': country_analysis_chart_update
    },
    'commitment_analysis': {
        'layout': commitment_analysis_layout,
        'grid': commitment_grid,
        'callback_fn': commitment_metric_chart
    },
    'comparative_analysis': {
        'layout': comparative_analysis_layout,
        'grid': comparative_grid,
        'callback_fn': comparative_analysis_chart_update
    },
    'seasonal_analysis': {
        'layout': seasonal_analysis_layout,
        'grid': seasonal_grid,
        'callback_fn': None  # Using individual callbacks instead
    }
}


def register_esr_callbacks():
    """Register all ESR callbacks for each page using new store-based system"""
    
    # Register store-based callbacks for sales trends
    sales_grid = layout_dict['sales_trends']['grid']
    sales_callback_fn = layout_dict['sales_trends']['callback_fn']
    
    # Get all chart IDs for sales trends
    chart_ids = sales_grid.get_chart_ids()
    
    # Register single multi-chart store callback for all sales trend charts
    sales_grid.register_chart_store_callback(
        app=app,
        chart_id=chart_ids,  # Pass all chart IDs as a list
        update_function=sales_callback_fn,
        menu_inputs=['countries', 'country_display_mode', 'date_range', 
                    'chart_0_column', 'chart_1_column', 'chart_2_column']
    )
    
    print(f"Registered single multi-chart store callback for {len(chart_ids)} sales trend charts")
    
    # Register individual callbacks for commitment analysis charts
    commitment_grid = layout_dict['commitment_analysis']['grid']
    
    # Get all chart IDs for commitment analysis
    commitment_chart_ids = commitment_grid.get_chart_ids()
    
    # Register callbacks: 1 for metric chart, 1 for 3 analytics charts
    metric_chart_id = 'esr_commitment_frame0_chart_0'
    analytics_chart_ids = [cid for cid in commitment_chart_ids if cid != metric_chart_id]
    
    # Register single metric chart callback
    commitment_grid.register_chart_store_callback(
        app=app,
        chart_id=metric_chart_id,
        update_function=commitment_metric_chart,
        menu_inputs=['commitment_metric', 'countries', 'country_display_mode', 'date_range']
    )
    
    # Register multi-chart analytics callback for the 3 analytics charts
    commitment_grid.register_chart_store_callback(
        app=app,
        chart_id=analytics_chart_ids,  # List of 3 chart IDs
        update_function=commitment_analytics_chart,
        menu_inputs=['countries', 'country_display_mode', 'date_range']
    )
    
    print(f"Registered individual store callbacks for {len(commitment_chart_ids)} commitment analysis charts")
    
    # Register store-based callbacks for country analysis
    country_grid = layout_dict['country_analysis']['grid']
    country_callback_fn = layout_dict['country_analysis']['callback_fn']
    
    # Get all chart IDs for country analysis
    country_chart_ids = country_grid.get_chart_ids()
    
    # Register single multi-chart store callback for all country analysis charts
    country_grid.register_chart_store_callback(
        app=app,
        chart_id=country_chart_ids,  # Pass all chart IDs as a list
        update_function=country_callback_fn,
        menu_inputs=['countries', 'country_display_mode', 'country_metric', 'start_year', 'end_year', 'date_range']
    )
    
    print(f"Registered single multi-chart store callback for {len(country_chart_ids)} country analysis charts")
    
    # Register individual callbacks for seasonal analysis charts
    seasonal_grid = layout_dict['seasonal_analysis']['grid']
    
    # Get all chart IDs for seasonal analysis
    seasonal_chart_ids = seasonal_grid.get_chart_ids()
    
    # Register unified callback for both seasonal analysis charts
    seasonal_grid.register_chart_store_callback(
        app=app,
        chart_id=seasonal_chart_ids,  # Pass all chart IDs for multi-chart callback
        update_function=unified_seasonal_analysis_update,
        menu_inputs=['seasonal_metric', 'countries', 'country_display_mode', 'selected_market_year', 'start_year', 'end_year', 'date_range']
    )
    
    print(f"Registered unified callback for {len(seasonal_chart_ids)} seasonal analysis charts")
    if len(seasonal_chart_ids) >= 2:
        print(f"  - Overlay chart: {seasonal_chart_ids[0]}")
        print(f"  - Differenced chart: {seasonal_chart_ids[1]}")
    else:
        print(f"  - Chart IDs: {seasonal_chart_ids}")
    
    print("Seasonal analysis table callback removed")
    
    # For other pages, keep using traditional callbacks for now
    print("Checking for remaining pages that need traditional callback registration...")
    for page_name, page_config in layout_dict.items():
        if page_name in ['sales_trends', 'commitment_analysis', 'country_analysis', 'seasonal_analysis']:
            print(f"  - Skipping {page_name} (already handled with store-based callbacks)")
            continue  # Already handled above
            
        callback_fn = page_config['callback_fn']
        if callback_fn is not None:
            print(f"  - Registering traditional callbacks for {page_name}")
            grid = page_config['grid']
            grid.create_menu_callbacks(app, callback_fn)
        else:
            print(f"  - Skipping {page_name} (no callback function defined)")
    
    print("Callback registration completed!")


def layout(page_name='sales_trends'):
    nav_style = {
        'backgroundColor': '#222222',
        'padding': '10px 20px',
        'marginBottom': '20px',
        'borderRadius': '5px'
    }
    button_style = {
        'backgroundColor': '#333333',
        'color': '#e8e8e8',
        'border': '1px solid #444444',
        'padding': '10px 20px',
        'margin': '0 10px',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'fontSize': '14px'
    }
    active_button_style = {
        **button_style,
        'backgroundColor': '#4CAF50',
        'borderColor': '#4CAF50'
    }
    
    navigation = html.Div([
        html.H1("ESR Analysis Dashboard",
                style={'color': '#e8e8e8', 'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            dbc.Button("Sales Trends", id="nav-sales-trends", href='/esr/sales_trends', 
                      style=active_button_style if page_name == 'sales_trends' else button_style),
            dbc.Button("Country Analysis", id="nav-country-analysis", href='/esr/country_analysis', 
                      style=active_button_style if page_name == 'country_analysis' else button_style),
            dbc.Button("Commitment Analysis", id="nav-commitment-analysis", href='/esr/commitment_analysis',
                      style=active_button_style if page_name == 'commitment_analysis' else button_style),
            dbc.Button("Comparative Analysis", id="nav-comparative-analysis", href='/esr/comparative_analysis',
                      style=active_button_style if page_name == 'comparative_analysis' else button_style),
            dbc.Button("Seasonal Analysis", id="nav-seasonal-analysis", href='/esr/seasonal_analysis',
                      style=active_button_style if page_name == 'seasonal_analysis' else button_style),
        ], style={'textAlign': 'center'})
    ], style=nav_style)
    
    # Page content container  
    page_content = html.Div(id="esr-page-content", children=layout_dict[page_name]['layout'])
    
    # ESR Data Stores and Controls
    esr_controls = html.Div([
        html.H3("Data Controls", style={'color': '#e8e8e8', 'marginBottom': '15px'}),
        
        html.Div([
            html.Label("Select Commodity:", style={'color': '#e8e8e8', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='esr-commodity-dd',
                options=[
                    {'label': 'Cattle', 'value': 'cattle'},
                    {'label': 'Corn', 'value': 'corn'},
                    {'label': 'Wheat', 'value': 'wheat'},
                    {'label': 'Soybeans', 'value': 'soybeans'},
                    {'label': 'Hogs', 'value': 'hogs'}
                ],
                value='cattle',
                style={'backgroundColor': '#333333', 'color': '#e8e8e8'}
            )
        ], style={'marginBottom': '20px', 'maxWidth': '300px'}),
        
        # Marketing Year Information Display
        html.Div([
            html.Label("Marketing Year Period:", style={'color': '#e8e8e8', 'marginBottom': '5px'}),
            html.Div(
                id='esr-marketing-year-display',
                children="September - August",
                style={
                    'color': '#e8e8e8', 
                    'backgroundColor': '#333333',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'border': '1px solid #555555'
                }
            )
        ], style={'marginBottom': '20px', 'maxWidth': '300px'})
    ], style={'backgroundColor': '#222222', 'padding': '15px', 'marginBottom': '20px', 'borderRadius': '5px'})
    
    # Main layout
    main_layout = html.Div([
        # ESR Data Stores
        dcc.Store(id="esr-df-store"),
        dcc.Store(id="esr-options-store"),
        
        # Store for current page
        dcc.Store(id="current-page", data=page_name),

        # ESR Controls
        esr_controls,

        # Navigation
        navigation,
        
        # Page content
        page_content,
    ], style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'})

    return main_layout


# ESR Store Callbacks
@callback(
    Output('esr-df-store', 'data'),
    Input('esr-commodity-dd', 'value'),
    prevent_initial_call=False
)
def update_esr_data_store(commodity):
    """Update ESR data store based on selected commodity"""
    if not commodity:
        return {}
    
    try:
        client = ESRTableClient()
        # Load ESR data for the selected commodity
        data_key = f"{commodity}/exports/all"
        df = client.get_key(data_key, use_simple_name=False)
        
        if df is not None and not df.empty:
            # Debug: Print available columns
            print(f"DEBUG - Available columns in {commodity} data: {list(df.columns)}")
            
            # Ensure date column is properly formatted
            if 'weekEndingDate' in df.columns:
                df['weekEndingDate'] = pd.to_datetime(df['weekEndingDate'])
                df['weekEndingDate'] = df['weekEndingDate'].dt.strftime('%Y-%m-%d')
            
            # Return data as dict with orient='records'
            return df.to_dict('records')
        else:
            return {}
    except Exception as e:
        print(f"Error loading ESR data for {commodity}: {e}")
        return {}

@callback(
    Output('esr-options-store', 'data'),
    Input('esr-df-store', 'data'),
    prevent_initial_call=False
)
def update_esr_options_store(esr_data):
    """Generate menu options based on loaded ESR data"""
    if not esr_data:
        return {}
    
    try:
        df = pd.DataFrame(esr_data)
        
        options_data = {}
        
        # Get unique countries
        if 'country' in df.columns:
            countries = sorted(df['country'].unique().tolist())
            options_data['countries'] = [
                {'label': country, 'value': country} 
                for country in countries if pd.notna(country)
            ]
        
        # Get available numeric columns for chart metrics
        numeric_columns = df.select_dtypes(include=[int, float]).columns.tolist()
        # Common ESR columns to prioritize
        priority_columns = ['weeklyExports', 'outstandingSales', 'grossNewSales', 
                          'currentMYNetSales', 'currentMYTotalCommitment']
        
        # Order columns with priority ones first
        ordered_columns = []
        for col in priority_columns:
            if col in numeric_columns:
                ordered_columns.append(col)
                numeric_columns.remove(col)
        ordered_columns.extend(numeric_columns)
        
        options_data['columns'] = [
            {'label': col.replace('weekly', 'Weekly ').replace('outstanding', 'Outstanding ')
                         .replace('gross', 'Gross ').replace('current', 'Current ')
                         .replace('MY', ' MY').replace('Net', ' Net')
                         .replace('Sales', ' Sales').replace('Exports', ' Exports')
                         .replace('Commitment', ' Commitment').title(), 
             'value': col}
            for col in ordered_columns
        ]
        
        # Get date range
        if 'weekEndingDate' in df.columns:
            dates = pd.to_datetime(df['weekEndingDate'])
            options_data['date_range'] = {
                'min_date': dates.min().strftime('%Y-%m-%d'),
                'max_date': dates.max().strftime('%Y-%m-%d')
            }
        
        return options_data
        
    except Exception as e:
        print(f"Error generating ESR options: {e}")
        return {}

# Dynamic country options update for commitment analysis
@callback(
    Output('esr_commitment_analysis_menu-countries', 'options'),
    Output('esr_commitment_analysis_menu-countries', 'value'),
    Input('esr-options-store', 'data'),
    prevent_initial_call=True
)
def update_commitment_countries_options(options_data):
    """Update commitment analysis countries options from store"""
    if not options_data or 'countries' not in options_data:
        # Return default options
        default_options = [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'}
        ]
        return default_options, ['Korea, South', 'Japan']
    
    countries_options = options_data['countries']
    # Set default to first 2-3 countries
    default_values = [opt['value'] for opt in countries_options[:3]]
    return countries_options, default_values

# Dynamic country options update for sales trends
@callback(
    Output('esr_sales_trends_menu-countries', 'options'),
    Output('esr_sales_trends_menu-countries', 'value'),
    Input('esr-options-store', 'data'),
    prevent_initial_call=True
)
def update_sales_trends_countries_options(options_data):
    """Update sales trends countries options from store"""
    if not options_data or 'countries' not in options_data:
        # Return default options
        default_options = [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Mexico', 'value': 'Mexico'},
            {'label': 'Canada', 'value': 'Canada'},
            {'label': 'Taiwan', 'value': 'Taiwan'}
        ]
        return default_options, ['Korea, South', 'Japan', 'China']

    countries_options = options_data['countries']
    # Set default to first 3 countries
    default_values = [opt['value'] for opt in countries_options[:3]]
    return countries_options, default_values

# Dynamic country options update for country analysis
@callback(
    Output('esr_country_analysis_menu-countries', 'options'),
    Output('esr_country_analysis_menu-countries', 'value'),
    Input('esr-options-store', 'data'),
    prevent_initial_call=True
)
def update_country_analysis_countries_options(options_data):
    """Update country analysis countries options from store"""
    if not options_data or 'countries' not in options_data:
        # Return default options
        default_options = [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'}
        ]
        return default_options, ['Korea, South', 'Japan']
    
    countries_options = options_data['countries']
    # Set default to first 2 countries
    default_values = [opt['value'] for opt in countries_options[:2]]
    return countries_options, default_values

# Dynamic country options update for seasonal analysis
@callback(
    Output('esr_seasonal_analysis_menu-countries', 'options'),
    Output('esr_seasonal_analysis_menu-countries', 'value'),
    Input('esr-options-store', 'data'),
    prevent_initial_call=True
)
def update_seasonal_analysis_countries_options(options_data):
    """Update seasonal analysis countries options from store"""
    if not options_data or 'countries' not in options_data:
        # Return default options
        default_options = [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Mexico', 'value': 'Mexico'},
            {'label': 'Canada', 'value': 'Canada'},
            {'label': 'Taiwan', 'value': 'Taiwan'}
        ]
        return default_options, ['Korea, South', 'Japan', 'China']
    
    countries_options = options_data['countries']
    # Set default to first 3 countries
    default_values = [opt['value'] for opt in countries_options[:3]]
    return countries_options, default_values

# Marketing Year Display Callback
@callback(
    Output('esr-marketing-year-display', 'children'),
    Input('esr-commodity-dd', 'value'),
    prevent_initial_call=False
)
def update_marketing_year_display(commodity):
    """Update marketing year display based on selected commodity"""
    if not commodity:
        return "September - August"
    
    # Define marketing year periods for different commodity types
    marketing_year_periods = {
        'cattle': 'January - December',
        'hogs': 'December - November', 
        'corn': 'September - August',
        'wheat': 'June - May',
        'soybeans': 'September - August'
    }
    
    return marketing_year_periods.get(commodity, 'September - August')

# Register callbacks when module is imported
register_esr_callbacks()