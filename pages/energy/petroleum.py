"""
Petroleum Analysis Page - Domain-specific analysis for petroleum markets
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from dash import register_page

# Register this page
register_page(__name__, path='/energy/petroleum', name='Petroleum Analysis', title='Petroleum Analysis')

def create_domain_card(title, description, products, metrics, status="Available", color="#f39c12"):
    """Create a card for petroleum domain analysis"""
    status_color = "success" if status == "Available" else "warning" if status == "Coming Soon" else "secondary"
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="fas fa-oil-can me-2", style={'color': color}),
                html.H4(title, className="mb-0 d-inline", style={'color': color}),
                dbc.Badge(status, color=status_color, className="ms-auto")
            ], style={'display': 'flex', 'align-items': 'center'})
        ], style={'background-color': f'{color}15'}),
        dbc.CardBody([
            html.P(description, className="card-text mb-3"),
            
            # Products covered
            html.H6("Products Covered:", className="text-muted mb-2"),
            html.Div([
                dbc.Badge(product, color="light", text_color="dark", className="me-2 mb-1")
                for product in products
            ], className="mb-3"),
            
            # Key metrics
            html.H6("Key Metrics:", className="text-muted mb-2"), 
            html.Ul([html.Li(metric) for metric in metrics], className="mb-3"),
            
            dbc.Button([
                html.I(className="fas fa-chart-area me-2"),
                "Launch Analysis" if status == "Available" else "Coming Soon"
            ], color="primary" if status == "Available" else "secondary",
               href="#" if status != "Available" else f"/energy/petroleum/{title.lower().replace(' ', '-')}",
               disabled=status != "Available",
               style={'width': '100%'})
        ])
    ], style={'height': '100%', 'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'})

def create_sub_analysis_card(title, description, link, icon, status="Available"):
    """Create smaller cards for sub-analyses"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"{icon} fa-2x mb-3", style={'color': '#3498db'}),
                html.H6(title, className="card-title"),
                html.P(description, className="card-text text-muted", style={'font-size': '14px'}),
                dbc.Button("Analyze", size="sm", color="outline-primary",
                          href=link if status == "Available" else "#",
                          disabled=status != "Available")
            ], style={'text-align': 'center'})
        ])
    ], style={'height': '220px'})

layout = dbc.Container([
    # Breadcrumb navigation
    dbc.Breadcrumb(
        items=[
            {"label": "Energy", "href": "/energy", "external_link": True},
            {"label": "Petroleum", "active": True}
        ],
        style={'background-color': 'transparent', 'padding': '0'}
    ),
    
    # Header
    html.Div([
        html.H1([
            html.I(className="fas fa-oil-can me-3", style={'color': '#f39c12'}),
            "Petroleum Analysis"
        ], className="mb-4", style={'color': '#2c3e50'}),
        html.P("Domain-specific analysis platform for petroleum markets covering refinery operations, inventory management, and global supply dynamics.",
               style={'font-size': '16px', 'color': '#7f8c8d', 'margin-bottom': '40px'})
    ]),
    
    # Main domain categories
    dbc.Row([
        # Refinery Analysis
        dbc.Col([
            create_domain_card(
                title="Refinery Operations",
                description="Comprehensive analysis of refinery operations including utilization rates, crude input, refined product output, and operational efficiency metrics.",
                products=["Crude Oil Input", "Gasoline Production", "Distillate Production", "Utilization Rates"],
                metrics=[
                    "Refinery utilization rates by region",
                    "Crack spreads (gasoline, distillate)",
                    "Crude oil input patterns", 
                    "Refined product yields",
                    "Maintenance and capacity analysis"
                ],
                status="Coming Soon",
                color="#e74c3c"
            )
        ], width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        # Inventory Analysis
        dbc.Col([
            create_domain_card(
                title="Inventory Management",
                description="Track petroleum product inventories including crude oil, gasoline, and distillates. Monitor strategic petroleum reserve levels and commercial stock patterns.",
                products=["Crude Oil Stocks", "Gasoline Inventories", "Distillate Stocks", "SPR Levels"],
                metrics=[
                    "Commercial crude oil inventories",
                    "Strategic Petroleum Reserve levels",
                    "Gasoline and distillate stocks",
                    "Days of supply calculations",
                    "Inventory vs demand ratios"
                ],
                status="Coming Soon", 
                color="#9b59b6"
            )
        ], width=6, className="mb-4"),
        
        # OPEC Analysis
        dbc.Col([
            create_domain_card(
                title="OPEC+ Dynamics",
                description="Analysis of OPEC+ production decisions, compliance rates, spare capacity, and global supply-demand balance impacts.",
                products=["OPEC Production", "Spare Capacity", "Compliance Rates", "Price Targets"],
                metrics=[
                    "OPEC+ production quotas vs actual",
                    "Compliance rate tracking",
                    "Spare capacity estimates",
                    "Market share analysis",
                    "Geopolitical risk factors"
                ],
                status="Coming Soon",
                color="#1abc9c"
            )
        ], width=6, className="mb-4")
    ]),
    
    # Quick analysis tools
    html.Div([
        html.H3("Quick Analysis Tools", className="mb-4", style={'color': '#34495e'}),
        dbc.Row([
            dbc.Col([
                create_sub_analysis_card(
                    "Crack Spreads",
                    "Gasoline and distillate crack spread analysis",
                    "#",
                    "fas fa-chart-line",
                    "Coming Soon"
                )
            ], width=3),
            
            dbc.Col([
                create_sub_analysis_card(
                    "Price Correlations",
                    "WTI vs Brent price relationship analysis",
                    "#",
                    "fas fa-exchange-alt", 
                    "Coming Soon"
                )
            ], width=3),
            
            dbc.Col([
                create_sub_analysis_card(
                    "Futures Curves",
                    "Contango vs backwardation analysis",
                    "#",
                    "fas fa-chart-area",
                    "Coming Soon"
                )
            ], width=3),
            
            dbc.Col([
                create_sub_analysis_card(
                    "Import/Export",
                    "Trade flow analysis and dependencies",
                    "#",
                    "fas fa-ship",
                    "Coming Soon"
                )
            ], width=3)
        ])
    ], style={'margin-top': '40px'}),
    
    # Market overview section
    html.Div([
        html.H3("Petroleum Market Overview", className="mb-4", style={'color': '#34495e'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-industry me-2", style={'color': '#e74c3c'}),
                            "Refining Sector"
                        ], className="card-title"),
                        html.P("U.S. refineries process approximately 17-18 million barrels per day of crude oil, with utilization rates typically ranging from 85-95% during peak driving season.", className="card-text"),
                        html.Small("Capacity: ~18.1 million bpd", className="text-muted")
                    ])
                ])
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-database me-2", style={'color': '#9b59b6'}),
                            "Strategic Reserves"
                        ], className="card-title"),
                        html.P("The Strategic Petroleum Reserve maintains emergency crude oil supplies, while commercial inventories provide operational flexibility for refiners and marketers.", className="card-text"),
                        html.Small("SPR Capacity: 714 million barrels", className="text-muted")
                    ])
                ])
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="fas fa-globe me-2", style={'color': '#1abc9c'}),
                            "Global Markets"
                        ], className="card-title"),
                        html.P("OPEC+ decisions significantly impact global oil prices and market dynamics, with spare capacity serving as a key buffer against supply disruptions.", className="card-text"),
                        html.Small("OPEC+ Share: ~40% of global production", className="text-muted")
                    ])
                ])
            ], width=4)
        ])
    ], style={'margin-top': '40px'}),
    
    # Methodology section
    html.Div([
        html.Hr(style={'margin-top': '40px'}),
        html.H4("Analysis Framework", style={'color': '#34495e'}),
        dbc.Row([
            dbc.Col([
                html.H6("Refinery Metrics:", className="text-primary"),
                html.Ul([
                    html.Li("Utilization Rate = (Crude Input / Operable Capacity) × 100"),
                    html.Li("Crack Spread = Product Price - (Crude Price × Yield)"),
                    html.Li("Yield Analysis = Product Output / Crude Input"),
                    html.Li("Regional Capacity Analysis")
                ])
            ], width=4),
            
            dbc.Col([
                html.H6("Inventory Analysis:", className="text-primary"),
                html.Ul([
                    html.Li("Days of Supply = Current Stock / Daily Demand"),
                    html.Li("Stock-to-Use Ratio = Inventory / Annual Consumption"),
                    html.Li("Seasonal Adjustment Factors"),
                    html.Li("Commercial vs Strategic Reserve Tracking")
                ])
            ], width=4),
            
            dbc.Col([
                html.H6("OPEC+ Analysis:", className="text-primary"),
                html.Ul([
                    html.Li("Compliance Rate = Actual / Quota Production"),
                    html.Li("Spare Capacity = Sustainable Capacity - Current Production"),
                    html.Li("Market Share Analysis"),
                    html.Li("Geopolitical Risk Assessment")
                ])
            ], width=4)
        ])
    ], style={'margin-top': '30px', 'margin-bottom': '40px'})
    
], fluid=True, style={'background-color': '#f8f9fa', 'min-height': '100vh', 'padding': '20px'})