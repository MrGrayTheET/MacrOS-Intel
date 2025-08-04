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
from layouts.agriculture import import_export_layout
from callbacks.agricultural import import_export_callbacks
load_dotenv('.env')
from data.data_tables import FASTable
from  dash import html, Input, Output

# Initialize the Dash app
data_type = None
dash.register_page(__name__, path ='/trade_bal' ,path_template='/trade_bal/<data_type>', title=f"{data_type}")

# Initialize TableClient
table_client = FASTable()
cbs = import_export_callbacks(table_client)

# Set the layout using the function (default to imports)
def layout(data_type=None, **kwargs):
    if not data_type:
        return html.Div('Invalid trade type (select imports or exports)')
    else:

        return import_export_layout(data_type)


    




