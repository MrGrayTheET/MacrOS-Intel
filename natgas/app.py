from dotenv import load_dotenv; load_dotenv('.env')
load_dotenv('.env')
from assets.frames import FundamentalFrame as FFrame
import dash
from data_sources.tables import EIATable
from model_apps import StorageApp

import plotly.io as  pio

pio.renderers.default = "browser"

NGTable = EIATable(commodity="NG")
chart_configs = [
        {
            'starting_key': 'prices/NG_1',
            'title': 'In Storage',
            'chart_type': 'line',
            'height': 300
        },
        {
            'starting_key': 'consumption/net_withdrawals',
            'title': 'Net Withdrawals',
            'chart_type': 'line',
            'height': 300
        }
    ]
storage = StorageApp()
NGFrame = FFrame(NGTable, chart_configs, None, div_prefix='NG-div')
big_div = NGFrame.generate_layout_div()
app_div = storage.generate_layout()






app = dash.Dash(__name__)

app.layout = layout

NGFrame.register_callbacks(app)


if __name__ == "__main__":
    app.run_server(debug=True)

