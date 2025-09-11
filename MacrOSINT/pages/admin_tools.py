"""
Admin Tools Page

Administrative tools for managing data sources and TableClient updates.
"""

from dash import html
from MacrOSINT.components import table_client_updater


def create_admin_tools_layout():
    """Create the admin tools page layout."""
    
    return html.Div([
        html.H1("Administrative Tools", className="mb-4"),
        html.P("Tools for managing data sources, updating TableClients, and system maintenance.", 
               className="lead mb-4"),
        
        # TableClient Updater Section
        html.Div([
            table_client_updater.get_layout()
        ], className="mb-5"),
        
        # Additional admin tools can be added here
        html.Hr(),
        html.Div([
            html.H3("Additional Tools", className="mb-3"),
            html.P("Future administrative tools will be added here:", className="text-muted"),
            html.Ul([
                html.Li("Database maintenance utilities"),
                html.Li("Data validation tools"), 
                html.Li("System health monitoring"),
                html.Li("Configuration management")
            ], className="text-muted")
        ])
        
    ], className="container-fluid p-4")


# Page layout
layout = create_admin_tools_layout()