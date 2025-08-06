import dash
from dash import html, Input, Output, State, dcc
import dash_bootstrap_components as dbc
from .sales_trends import layout as sales_trends_layout, grid as sales_grid
from .country_analysis import layout as country_analysis_layout, grid as country_grid
from .commitment_analysis import layout as commitment_analysis_layout, grid as commitment_grid
from .comparative_analysis import layout as comparative_analysis_layout, grid as comparative_grid
from callbacks.esr import (
    sales_trends_chart_update,
    country_analysis_chart_update, 
    commitment_analysis_chart_update,
    comparative_analysis_chart_update
)

dash.register_page(__name__, path='/esr', path_template='/esr/<page_name>')
app = dash.get_app()
layout_dict = {
    'sales_trends':
        {
            'layout': sales_trends_layout,
            'grid': sales_grid,
            'callback_fn': sales_trends_chart_update
        },

    'country_analysis':
        {
            'layout': country_analysis_layout,
            'grid': country_grid,
            'callback_fn': country_analysis_chart_update
        },
    'commitment_analysis':
        {
            'layout': commitment_analysis_layout,
            'grid': commitment_grid,
            'callback_fn': commitment_analysis_chart_update
        },
    'comparative_analysis':
        {
            'layout': comparative_analysis_layout,
            'grid': comparative_grid,
            'callback_fn': comparative_analysis_chart_update
        }
}


def register_esr_callbacks():
    """Register all ESR callbacks for each page"""
    for page_name, page_config in layout_dict.items():
        # Create and register menu callbacks for each page
        grid = page_config['grid']
        callback_fn = page_config['callback_fn']
        grid.create_menu_callbacks(app, callback_fn)


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
            dbc.Button("Sales Trends", id="nav-sales-trends", href='/esr/sales_trends', style=active_button_style),
            dbc.Button("Country Analysis", id="nav-country-analysis", href='/esr/country_analysis', style=button_style),
            dbc.Button("Commitment Analysis", id="nav-commitment-analysis", href='/esr/commitment_analysis',
                       style=button_style),
            dbc.Button("Comparative Analysis", id="nav-comparative-analysis", href='/esr/comparative_analysis',
                       style=button_style),
        ], style={'textAlign': 'center'})
    ], style=nav_style)
    page_name = page_name
    # Page content container  
    page_content = html.Div(id="esr-page-content", children=layout_dict[page_name]['layout'])
    # Main layout
    main_layout = html.Div([
        # Store for current page
        dcc.Store(id="current-page", data=page_name),

        # Navigation
        navigation,

        # Page content
        page_content,

        # Global styles
        html.Div(id="global-styles")
    ], style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'})
    return main_layout

@app.callback(
    Output('esr-page-content', 'children'),
    Input('url', 'pathname'),
)
def update_esr_content(pathname):
    if 'esr' in pathname:
        return layout_dict[pathname.rsplit('/')[-1]]['layout']

# Register all ESR callbacks when module is imported
register_esr_callbacks()



