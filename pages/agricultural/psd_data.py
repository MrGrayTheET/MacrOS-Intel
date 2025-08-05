from dotenv import load_dotenv
load_dotenv('.env'); from data.data_tables import FASTable
from components.frames import FundamentalFrame as FFrame, FrameGrid as FrameMgr
from components.chart_components import FundamentalChart, MarketChart
from dash import html, dcc, callback, Input, Output
import dash
from components.divs import create_dd_menu

dash.register_page(__name__, path='/psd_data', path_template='/psd_data/<commodity_table>')

commodity_table = None
load_dotenv('.env')
commodity_table = 'cattle' if not commodity_table else commodity_table
fas_cli = FASTable()
comm_alias = fas_cli.esr_codes['alias'][commodity_table] if commodity_table in fas_cli.esr_codes['alias'].keys() else commodity_table

import_export_cfg = [
    {
 'starting_key': f'{commodity_table}/psd/summary',
'title': f'{comm_alias} Exports',
'y_column': 'Exports',
'chart_type': 'bar',
'width': "45%",
'height':'100%'
},
    {'starting_key':f'{commodity_table}/psd/summary',
    'title': f'{comm_alias} Imports',
    'y_column':'Imports',
    'chart_type':'bar', 'width':"45%", 'height':'100%'}]

beginning_ending_cfg  = [
    {
    'starting_key':f'{commodity_table}/psd/summary',
    'y_column': 'Beginning Stocks',
    'chart_type':'bar',
    'width':"45%",
    'height': "100%",
    'title': 'Total Stocks (start)'},
    {
    'starting_key':f'{commodity_table}/psd/summary',
    'y_column': 'Ending Stocks',
    'chart_type':'bar',
    'width':"45%",
    'height': "100%",
    'title': 'Total Stocks (end)'
    }
]

supply_demand =[
    {
      'starting_key': f'{commodity_table}/psd/summary',
      'y_column': 'Production',
      'title': "Domestic Production",
      'chart_type':'bar',
      'width': "45%",
      'height': "100%",
      },
    {'starting_key': f'{commodity_table}/psd/summary',
      'y_column': 'Domestic Consumption',
      'title': "Domestic Consumption",
      'chart_type':'bar',
      'width': "45%",
      'height': "100%",}
]

market_cfgs = []

trade_data = FFrame(fas_cli, import_export_cfg, layout='vertical', width="90%",div_prefix='intl-trade')
domestic_data = FFrame(fas_cli, supply_demand, layout='vertical', width="90%", div_prefix='domestic-trade', )
inventory_data = FFrame(fas_cli, beginning_ending_cfg, layout='horizontal', height="600px", div_prefix='inventory')
dd_select_menu, comp_ids = create_dd_menu(FASTable(), header_text="Select Commodity")

grid_cfg = {'layout_type':'custom',
            'rows':3, 'cols':2,
            'frame_positions':{
                0:{'row':1, 'col':1, 'col_span':4},
                1:{'row':1, 'col':2,'col_span': 4},
                2:{'row':2, 'col':1, 'col_span': 1}
        }
            }
menu_config= {'enabled':False}


Fgrid = FrameMgr(frames=[trade_data,domestic_data,inventory_data], grid_config=grid_cfg, menu_config=False)

output_charts = []
app = dash.get_app()
for frame in Fgrid.frames:
    output_charts.extend(frame.charts)

def layout():
    return html.Div(children=[
    dd_select_menu,
    Fgrid.generate_layout(include_title="Production, Supply, Distribution")
])

@app.callback( *[Output(chart.chart_id, 'figure') for chart in output_charts],
                Input(comp_ids['dd'], 'value'),
                Input(comp_ids['load-btn'],'n_clicks')
            )


def get_psd_figures(key, n_clicks):
    commodity = key.split('/')[0]
    if n_clicks:
        df = fas_cli[key]
        update_columns = ['Exports', 'Imports', 'Production', 'Domestic Consumption', 'Beginning Stocks', 'Ending Stocks']
        charts = []
        for frame in Fgrid.frames:
            charts.extend(frame.charts)
        new_figures = []
        for n in range(len(charts)):
            if hasattr(charts[n], 'title'):
                charts[n].title = f'{fas_cli.esr_codes["alias"][commodity.lower()].capitalize()} {update_columns[n]}'

            new_figures.append(
                charts[n].update_data_source(df, y_column=update_columns[n])
                )

        return new_figures

