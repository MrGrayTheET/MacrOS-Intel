from dotenv import load_dotenv

from callbacks.agricultural import register_psd_callbacks
from layouts.agriculture import psd_layout
load_dotenv('.env'); from data.data_tables import FASTable
import dash

dash.register_page(__name__, path='/psd_data', path_template='/psd_data/<commodity_key>')
main_app = dash.get_app()
commodity_table = None
load_dotenv('.env')
commodity_table = 'cattle' if not commodity_table else commodity_table
fas_cli = FASTable()
comm_alias = fas_cli.esr_codes['alias'][commodity_table] if commodity_table in fas_cli.esr_codes['alias'].keys() else commodity_table

PSDgrid, page_layout, component_ids = psd_layout(commodity_table, fas_cli)

def layout():
    return page_layout

register_psd_callbacks(PSDgrid, component_ids, fas_cli)