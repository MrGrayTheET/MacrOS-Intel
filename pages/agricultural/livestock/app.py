from dotenv import load_dotenv
load_dotenv('.env'); from sources.data_tables import FASTable
from assets.frames import FundamentalFrame as FFrame, MarketFrame as MFrame
from assets.app_container import DARK_THEME_CSS
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash
import dash_bootstrap_components as dbc
from pathlib import Path


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
fas_cli = FASTable('cattle')

import_export_cfg = [{
 'starting_key': 'psd/summary',
'title': 'Beef Exports',
'y_column': 'Exports',
'chart_type': 'bar',
'width': "45%",
'height':'100%'
}, {'starting_key':'psd/summary',
    'title': 'Beef Imports',
    'y_column':'Imports',
    'chart_type':'bar', 'width':"45%", 'height':'100%'}]

beginning_ending_cfg  = [{
    'starting_key':'psd/summary',
    'y_column': 'Beginning Stocks',
    'chart_type':'bar',
    'width':"45%",
    'height': "100%",
    'title': 'Total Stocks (start)'},
    {
    'starting_key':'psd/summary',
    'y_column': 'Ending Stocks',
    'chart_type':'bar',
    'width':"45%",
    'height': "100%",
    'title': 'Total Stocks (end)'
    }]

supply_demand =[{
      'starting_key': 'psd/summary',
      'y_column': 'Production',
      'title': "Domestic Production",
      'chart_type':'bar',
      'width': "45%",
      'height': "100%",
      },
    {'starting_key': 'psd/summary',
      'y_column': 'Domestic Consumption',
      'title': "Domestic Consumption",
      'chart_type':'bar',
      'width': "45%",
      'height': "100%",}]

market_cfgs = []

trade_data = FFrame(fas_cli, import_export_cfg, layout='horizontal', width="800px",div_prefix='intl-trade', enable_settings=False)
domestic_data = FFrame(fas_cli, supply_demand, layout='horizontal', width="800px", div_prefix='domestic-trade', enable_settings=False)
inventory_data = FFrame(fas_cli, beginning_ending_cfg, layout='horizontal', height="600px", div_prefix='inventory', enable_settings=False)

layout = [html.Div(children=[trade_data.generate_layout_div(),
                             domestic_data.generate_layout_div()],
                   style={'height':'800px',
                          'width':'1650px',
                          'display':'flex',
                          'background-color':'black'}),
          html.Div(children=[inventory_data.generate_layout_div()],
                             style={'width':'2000px', 'background-color':'black'})]
app.layout = layout
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

if __name__ == "__main__":
 app.run_server(debug=True)