import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Commodities Dashboard"

# Define the top navigation bar
navbar = dbc.NavbarSimple(
    brand="Commodities Dashboard",
    brand_href="/",
    color="dark",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Natural Gas", href="/natural-gas")),
        dbc.NavItem(dbc.NavLink("Crude Oil", href="/crude-oil")),
        dbc.NavItem(dbc.NavLink("Corn", href="/corn")),
        dbc.NavItem(dbc.NavLink("Wheat", href="/wheat"))
    ]
)

# Placeholder for main content area
def serve_layout():
    return html.Div([
        dcc.Location(id="url"),
        navbar,
        html.Div(id="page-content", className="p-4")
    ])

app.layout = serve_layout

# Simple page router
@app.callback(
    dash.dependencies.Output("page-content", "children"),
    [dash.dependencies.Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/natural-gas":
        return html.H3("Natural Gas Dashboard Coming Soon...")
    elif pathname == "/crude-oil":
        return html.H3("Crude Oil Dashboard Coming Soon...")
    elif pathname == "/corn":
        return html.H3("Corn Dashboard Coming Soon...")
    elif pathname == "/wheat":
        return html.H3("Wheat Dashboard Coming Soon...")
    else:
        return html.H3("Welcome to the Commodities Dashboard!")

if __name__ == "__main__":
    app.run_server(debug=True)
