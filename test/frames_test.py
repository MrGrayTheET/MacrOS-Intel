from dotenv import load_dotenv; load_dotenv('.env')
load_dotenv('.env')
from components.frames import MarketFrame as MFrame
import dash
from sources.data_tables import MarketTable

import plotly.io as  pio
pio.renderers.default = "browser"
ticker = "NG"
configs =[{'ticker':ticker,
         'title':'Natural Gas Front Month',
         'chart_type':'candlestick',
         'start_date':'2016-01-01'
         ,'end_date':'2024-01-01',
         'freq':'D'}]

test_div = MFrame(MarketTable(), chart_configs=configs)
test_div.generate_layout_div()

app = dash.Dash(__name__)

app.layout = test_div.generate_layout_div()
test_div.register_callbacks(app)
app.run(debug=True)
