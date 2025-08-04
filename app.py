# app.py - Main application entry point
import dash
from dash import html, dcc, page_container, Input, Output, State
import dash_bootstrap_components as dbc
import os
import sys
from  dotenv import load_dotenv
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv(Path(os.path.dirname(os.path.abspath(__file__)), '.env')
            )


# Initialize the Dash app with pages
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=True,
    pages_folder="pages"
)

app.title = "Commodities Dashboard"

# Define the navigation bar
navbar = dbc.NavbarSimple(
    brand="Commodities Dashboard",
    brand_href="/",
    color="dark",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink(id='psd-data', href="/energy/ng_storage")),
    ]
)


# Main layout
app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id='commodity-state'),
    navbar,
    html.Div(
        page_container,
        className="p-4"
    )
])


if __name__ == "__main__":
    app.run(debug=True)