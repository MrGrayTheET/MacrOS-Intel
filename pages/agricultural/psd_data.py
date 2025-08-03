from dotenv import load_dotenv
load_dotenv('.env'); from data.data_tables import FASTable
from components.frames import FundamentalFrame as FFrame, FrameGrid as FrameMgr
from assets.app_container import DARK_THEME_CSS
from dash import html
import dash
import dash_bootstrap_components as dbc
commodity_table = None
dash.register_page(__name__, path='/agricultural/psd_data', path_template='/agricultural/psd_data/<commodity_table>')
load_dotenv('.env')
commodity_table = 'cattle' if not commodity_table else commodity_table
fas_cli = FASTable(commodity_table)
comm_alias = fas_cli.esr_codes['alias'][commodity_table] if commodity_table in fas_cli.esr_codes['alias'].keys() else commodity_table
import_export_cfg = [{
 'starting_key': 'psd/summary',
'title': f'{comm_alias} Exports',
'y_column': 'Exports',
'chart_type': 'bar',
'width': "45%",
'height':'100%'
}, {'starting_key':'psd/summary',
    'title': f'{comm_alias} Imports',
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

frame_cfgs = [import_export_cfg, beginning_ending_cfg, supply_demand]
market_cfgs = []

trade_data = FFrame(fas_cli, import_export_cfg, layout='horizontal', width="800px",div_prefix='intl-trade')
domestic_data = FFrame(fas_cli, supply_demand, layout='horizontal', width="800px", div_prefix='domestic-trade', )
inventory_data = FFrame(fas_cli, beginning_ending_cfg, layout='horizontal', height="600px", div_prefix='inventory')

grid_cfg = {'layout_type':'custom',
            'rows':3, 'cols':2,
            'frame_positions':{
                0:{'row':1, 'col':1, 'col_span': 2},
                1:{'row':2, 'col':1,'col_span': 2},
                2:{'row':3, 'col':1, 'col_span': 1}
        }
            }
from components.frames import create_enhanced_dashboard_configs

Fgrid = FrameMgr('psd-app')
for f in frame_cfgs:
    Fgrid.add_fundamental_frame(fas_cli, f)


def layout(**kwargs):
    return Fgrid.generate_layout(title='PSD (Production, Supply, Distribution)'), Fgrid.register_callbacks(dash.get_app())
