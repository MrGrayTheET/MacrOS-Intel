import dash
from dotenv import load_dotenv; load_dotenv('.env')
from assets.frames import FundamentalFrame as FFrame, MarketFrame as MFrame
from assets.app_container import DARK_THEME_CSS, UnifiedDashboard as UDash
import dash_bootstrap_components as dbc
from sources.data_tables import EIATable, MarketTable
from energy.natgas.models import StorageAppWSpreads as SpreadApp

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],  # Bootstrap dark theme
    suppress_callback_exceptions=True  # Important for dynamic callbacks
)
dashapp = UDash(app, title='Natural Gas')
ticker = 'NG'
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

market_charts = [{'ticker':ticker,
                  'title':'Natural Gas Front Month intraday',
                  'chart_type':'candlestick',
                  'start_date':'2016-01-01',
                  'end_date':'2025-01-01',
                  'interval':'1d',
                  }]
storage = SpreadApp()
storage.app_id = 'storage'
tbl = MarketTable()
NGFFrame = FFrame(NGTable, chart_configs=chart_configs, div_prefix="fundamentals")
NGMFrame = MFrame(tbl,chart_configs=market_charts, layout='horizontal', div_prefix="market")
mkt_cot_div = NGMFrame.generate_layout_div()
dashapp.add_frame('fundamentals', NGFFrame, 'fundamental')
dashapp.add_frame('storage', storage, 'storage')
dashapp.register_unified_callbacks()
dashapp.generate_layout()
app.index_string = f'''
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {DARK_THEME_CSS}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    '''

app.layout = dashapp.generate_layout()
app.run_server(debug=True)

if __name__ == "__main__":
    app.run_server(debug=True)

