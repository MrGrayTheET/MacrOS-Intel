"""
Energy Landing Page - Main entry point for energy commodity analysis
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from dash import register_page

# Register this page
#register_page(__name__, path='/energy', name='Energy Commodities', title='Energy Commodities Dashboard')

def create_energy_card(title, description, link, icon_class, color):
    """Create a card for energy commodity sections"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"{icon_class} fa-3x mb-3", style={'color': color}),
                html.H4(title, className="card-title", style={'color': color}),
                html.P(description, className="card-text"),
                dbc.Button("Enter Analysis", color="primary", href=link, external_link=True,
                          style={'width': '100%'})
            ], style={'text-align': 'center'})
        ])
    ], style={'height': '300px', 'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'})

layout = dbc.Container([
    # Header
    html.Div([
        html.H1("Energy Commodities Dashboard", 
               className="text-center mb-4",
               style={'color': '#2c3e50', 'font-weight': 'bold'}),
        html.P("Comprehensive analysis platform for petroleum and natural gas markets",
               className="text-center mb-5",
               style={'font-size': '18px', 'color': '#7f8c8d'})
    ], style={'margin-bottom': '50px'}),
    
    # Main commodity selection cards
    dbc.Row([
        dbc.Col([
            create_energy_card(
                title="Natural Gas",
                description="Storage analysis, price movements, production data, and consumption patterns. Analyze seasonal storage metrics and price relationships.",
                link="/energy/natural-gas",
                icon_class="fas fa-fire",
                color="#e74c3c"
            )
        ], width=6),
        
        dbc.Col([
            create_energy_card(
                title="Petroleum",
                description="Refinery operations, inventory tracking, crude oil analysis, and refined products. Domain-specific analysis for gasoline, distillates, and crude.",
                link="/energy/petroleum", 
                icon_class="fas fa-oil-can",
                color="#f39c12"
            )
        ], width=6)
    ], className="mb-5"),
    
    # Quick stats or highlights section
    html.Div([
        html.H3("Market Overview", className="text-center mb-4", style={'color': '#34495e'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Natural Gas Storage", className="card-title"),
                        html.P("Real-time storage levels compared to 5-year historical range", 
                              className="card-text"),
                        html.Small("Updated Weekly", className="text-muted")
                    ])
                ], color="light")
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Refinery Operations", className="card-title"),
                        html.P("Utilization rates, crude input, and refined product output",
                              className="card-text"),
                        html.Small("Updated Weekly", className="text-muted")
                    ])
                ], color="light")
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Inventory Levels", className="card-title"),
                        html.P("Crude oil, gasoline, and distillate inventory tracking",
                              className="card-text"),
                        html.Small("Updated Weekly", className="text-muted")
                    ])
                ], color="light")
            ], width=4)
        ])
    ], style={'margin-top': '50px'}),
    
    # Footer with additional links
    html.Hr(style={'margin-top': '50px'}),
    html.Div([
        html.P([
            "Energy data sourced from the ",
            html.A("U.S. Energy Information Administration (EIA)", 
                  href="https://www.eia.gov", target="_blank"),
            " | Built with the Commodities Dashboard Framework"
        ], className="text-center", style={'color': '#95a5a6'})
    ], style={'margin-bottom': '30px'})
    
], fluid=True, style={'background-color': '#f8f9fa', 'min-height': '100vh', 'padding': '40px'})


# Add any page-specific callbacks here if needed

def handle_navigation(pathname):
    # This callback can be used for any page-specific navigation logic
    return pathname