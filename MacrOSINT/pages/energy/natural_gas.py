"""
Natural Gas Analysis Page - Sub-landing page for natural gas analysis tools
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from dash import register_page

# Register this page
#register_page(__name__, path='/energy/natural-gas', name='Natural Gas Analysis', title='Natural Gas Analysis')

def create_analysis_card(title, description, link, metrics, status="Available"):
    """Create a card for natural gas analysis tools"""
    status_color = "success" if status == "Available" else "warning" if status == "Coming Soon" else "secondary"
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5(title, className="mb-0", style={'display': 'inline-block'}),
                dbc.Badge(status, color=status_color, className="ms-2")
            ])
        ]),
        dbc.CardBody([
            html.P(description, className="card-text"),
            html.Hr(),
            html.H6("Key Metrics:", className="text-muted mb-2"),
            html.Ul([html.Li(metric) for metric in metrics], className="mb-3"),
            dbc.Button("Launch Analysis" if status == "Available" else "Coming Soon", 
                      color="primary" if status == "Available" else "secondary",
                      href=link if status == "Available" else "#",
                      disabled=status != "Available",
                      style={'width': '100%'})
        ])
    ], style={'height': '100%', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'})

layout = dbc.Container([
    # Breadcrumb navigation
    dbc.Breadcrumb(
        items=[
            {"label": "Energy", "href": "/energy", "external_link": True},
            {"label": "Natural Gas", "active": True}
        ],
        style={'background-color': 'transparent', 'padding': '0'}
    ),
    
    # Header
    html.Div([
        html.H1([
            html.I(className="fas fa-fire me-3", style={'color': '#e74c3c'}),
            "Natural Gas Analysis"
        ], className="mb-4", style={'color': '#2c3e50'}),
        html.P("Comprehensive analysis tools for natural gas markets including storage, prices, production, and consumption patterns.",
               style={'font-size': '16px', 'color': '#7f8c8d', 'margin-bottom': '40px'})
    ]),
    
    # Analysis tools grid
    dbc.Row([
        # Storage Analysis
        dbc.Col([
            create_analysis_card(
                title="Storage Analysis",
                description="Analyze natural gas storage levels against historical patterns. Track seasonal storage metrics and their relationship with price movements.",
                link="/energy/natural-gas/storage",
                metrics=[
                    "Current vs 5-year historical range",
                    "Storage percentile calculations", 
                    "Price correlation analysis",
                    "Seasonal pattern recognition"
                ],
                status="Available"
            )
        ], width=6, className="mb-4"),
        
        # Price Analysis  
        dbc.Col([
            create_analysis_card(
                title="Price Analysis", 
                description="Track natural gas futures prices, spot markets, and regional differentials. Analyze price volatility and seasonal patterns.",
                link="/energy/natural-gas/prices",
                metrics=[
                    "Futures curve analysis",
                    "Henry Hub spot prices",
                    "Regional price differentials", 
                    "Volatility metrics"
                ],
                status="Coming Soon"
            )
        ], width=6, className="mb-4"),
        
        # Production Analysis
        dbc.Col([
            create_analysis_card(
                title="Production Analysis",
                description="Monitor natural gas production trends, drilling activity, and supply forecasts across major production regions.",
                link="/energy/natural-gas/production",
                metrics=[
                    "Dry production levels",
                    "Regional production trends",
                    "Rig count correlations",
                    "Supply growth forecasts"
                ],
                status="Coming Soon"
            )
        ], width=6, className="mb-4"),
        
        # Consumption Analysis
        dbc.Col([
            create_analysis_card(
                title="Consumption Analysis",
                description="Analyze natural gas demand patterns across sectors including residential, commercial, industrial, and power generation.",
                link="/energy/natural-gas/consumption", 
                metrics=[
                    "Sectoral demand breakdown",
                    "Seasonal consumption patterns",
                    "Power burn analysis",
                    "Industrial demand trends"
                ],
                status="Coming Soon"
            )
        ], width=6, className="mb-4")
    ]),
    
    # Market insights section
    html.Div([
        html.H3("Market Insights", className="mb-4", style={'color': '#34495e'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-chart-line me-2", style={'color': '#3498db'}),
                            "Storage-Price Relationship"
                        ], className="card-title"),
                        html.P("Natural gas storage levels are a key driver of price volatility. When storage is below the 5-year average, prices tend to be more volatile and higher.", className="card-text"),
                        html.Small("Historical correlation: -0.65", className="text-muted")
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-thermometer-half me-2", style={'color': '#e67e22'}),
                            "Seasonal Patterns"
                        ], className="card-title"),
                        html.P("Natural gas follows predictable seasonal patterns with injection season (Apr-Oct) and withdrawal season (Nov-Mar) driving storage cycles.", className="card-text"),
                        html.Small("Peak storage typically occurs in October-November", className="text-muted")
                    ])
                ])
            ], width=6)
        ])
    ], style={'margin-top': '50px'}),
    
    # Data sources and methodology
    html.Div([
        html.Hr(style={'margin-top': '40px'}),
        html.H4("Data Sources & Methodology", style={'color': '#34495e'}),
        dbc.Row([
            dbc.Col([
                html.H6("Data Sources:", className="text-primary"),
                html.Ul([
                    html.Li("U.S. Energy Information Administration (EIA)"),
                    html.Li("Natural Gas Intelligence (NGI)"),
                    html.Li("NYMEX Futures Markets"),
                    html.Li("Regional Pipeline Data")
                ])
            ], width=6),
            
            dbc.Col([
                html.H6("Key Calculations:", className="text-primary"),
                html.Ul([
                    html.Li("Storage Percentile = (Current - 5Y Min) / (5Y Max - 5Y Min) × 100"),
                    html.Li("Deviation from Mean = (Current - 5Y Mean) / 5Y Mean"),
                    html.Li("Forward Returns = Log(Price[t+n] / Price[t])"),
                    html.Li("Signal Strength = Decile Ranking of Deviations")
                ])
            ], width=6)
        ])
    ], style={'margin-top': '30px', 'margin-bottom': '40px'})
    
], fluid=True, style={'background-color': '#f8f9fa', 'min-height': '100vh', 'padding': '20px'})


# Navigation callback

def handle_navigation(pathname):
    return pathname