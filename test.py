from sources.data_tables import EIATable
from dotenv import load_dotenv
import plotly.io as  pio

load_dotenv('.env')
pio.renderers.default = "browser"


products = ['crude_oil', 'gasoline', 'distillates']
important_series = []

eia_data = EIATable(commodity='PET')


eia_data.update_all()





