"""
Natural Gas Storage Analysis Page - Integrated with the framework
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from dash import register_page
from pages.energy.natgas.models import NaturalGasStorageApp

# Register this page
#register_page(__name__, path='/energy/natural-gas/storage', name='Natural Gas Storage Analysis', title='Natural Gas Storage Analysis')

# Initialize the Natural Gas Storage Application
try:
    natgas_app = NaturalGasStorageApp(app_id='natgas-storage-page')
    app_layout = natgas_app.generate_layout()
    has_data = True
    error_message = None
except Exception as e:
    has_data = False
    error_message = str(e)
    app_layout = html.Div("Error loading application")

def create_breadcrumb_and_header():
    """Create breadcrumb navigation and header"""
    return html.Div([
        dbc.Breadcrumb(
            items=[
                {"label": "Energy", "href": "/energy", "external_link": True},
                {"label": "Natural Gas", "href": "/energy/natural-gas", "external_link": True},
                {"label": "Storage Analysis", "active": True}
            ],
            style={'background-color': 'transparent', 'padding': '0', 'margin-bottom': '20px'}
        ),
        
        # Header with key info
        html.Div([
            html.H2([
                html.I(className="fas fa-database me-3", style={'color': '#e74c3c'}),
                "Natural Gas Storage Analysis"
            ], style={'color': '#2c3e50'}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H6([
                            html.I(className="fas fa-info-circle me-2"),
                            "About This Analysis"
                        ], className="alert-heading"),
                        html.P([
                            "This tool analyzes U.S. natural gas storage levels compared to historical patterns. ",
                            "The key metric is the ",
                            html.Strong("Storage Percentile"),
                            " = (Current Storage - 5Y Min) / (5Y Max - 5Y Min) × 100, ",
                            "which shows where current storage sits within the 5-year historical range."
                        ], className="mb-0")
                    ], color="info", style={'margin-bottom': '20px'})
                ], width=12)
            ])
        ], style={'margin-bottom': '30px'})
    ])

def create_error_layout():
    """Create error layout if app fails to load"""
    return dbc.Container([
        create_breadcrumb_and_header(),
        dbc.Alert([
            html.H4("Application Error", className="alert-heading"),
            html.P(f"Failed to load Natural Gas Storage Analysis: {error_message}"),
            html.Hr(),
            html.P("Please check the data connections and try again.", className="mb-0"),
            dbc.Button("Return to Natural Gas Overview", color="primary", 
                      href="/energy/natural-gas", external_link=True, className="mt-3")
        ], color="danger")
    ], fluid=True, style={'padding': '20px'})

def create_main_layout():
    """Create the main layout with the storage app"""
    return html.Div([
        # Navigation and header
        dbc.Container([
            create_breadcrumb_and_header()
        ], fluid=True, style={'padding': '0 20px'}),
        
        # Main application
        app_layout
    ], style={'background-color': '#f8f9fa', 'min-height': '100vh'})

# Set the layout based on whether the app loaded successfully
layout = create_main_layout() if has_data else create_error_layout()

# Register callbacks if the app loaded successfully
if has_data:
    # Register the natural gas app callbacks
    def register_storage_callbacks(app):
        """Register callbacks for the storage application"""
        try:
            natgas_app.register_callbacks(app)
        except Exception as e:
            print(f"Error registering storage app callbacks: {e}")
    
    # This will be called when the main app initializes
    # The callback registration will happen in the main app file