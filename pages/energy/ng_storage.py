import dash
from dash import html, dcc, Input, Output
from pages.energy.natgas.models import StorageAppWSpreads

dash.register_page(__name__, path='/storage')

storageApp = StorageAppWSpreads()

def layout(**kwargs):
    storageApp.register_callbacks(dash.get_app())
    return storageApp.generate_layout_div()

