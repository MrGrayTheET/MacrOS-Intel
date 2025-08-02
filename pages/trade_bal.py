"""
Flexible Trade Analysis Dashboard

This module provides a reusable Dash application for analyzing different types of trade data.
The layout function can be configured for imports, exports, or any other data key_type.

Usage Examples:
--------------
# Imports Dashboard:
app.layout = get_layout(data_type="imports")

# Exports Dashboard:
app.layout = get_layout(data_type="exports")

# Custom Data Dashboard:
custom_df = pd.read_csv('production_data.csv')
app.layout = get_layout(data_type="production", data_source=custom_df)

Features:
---------
- Dynamic commodity selection
- Time series analysis by country
- Breakdown charts (pie/bar) by year
- Summary statistics
- Dark theme UI
- Responsive design
"""

# pages/trade_bal.py
import dash
from dotenv import load_dotenv
from layouts.agricultural.layouts import import_export_layout
from callbacks.agricultural import import_export_callbacks
load_dotenv('.env')
from data.data_tables import FASTable
from  dash import html
import dash_bootstrap_components as dbc
# Initialize the Dash app
data_type = None
dash.register_page(__name__, path_template='trade/<data_type>/', title=f"{data_type}")

# Initialize TableClient
table_client = FASTable()

# Set the layout using the function (default to imports)
def layout(data_type=None, **kwargs):
    if not data_type:
        return html.Div('Invalid trade type (select imports or exports)')
    else:
        return import_export_layout(table_client, data_type)



