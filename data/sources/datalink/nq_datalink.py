import requests
import pandas as pd
import os
from pathlib import Path

import toml
from dotenv import load_dotenv
import nasdaqdatalink as ndl


load_dotenv()
data_path = os.getenv('data_path')

if not os.path.exists(data_path):
    os.mkdir(data_path)

with open('data_sources/COT/futures_mappings.toml') as f:
    contract_map = toml.load(f)


# noinspection PyTypeChecker
class COT:

    table_folder = Path(data_path, 'COT')

    if not os.path.exists(table_folder):
        os.mkdir(table_folder)

    table_db = Path(table_folder, 'cot.h5')

    def __init__(self, datalink_api_key=None):

        return

    def initialize_from_file(self, master_csv_location):
        big_csv = pd.read_csv(master_csv_location, index_col=['contract_code'])
        for k, v in contract_map.items():
            try:
                ticker_cot = big_csv.loc[v]
            except Exception as e:
                print(f'Error Locating {k} contract. Code {v} is likely wrong')
            else:
                ticker_cot.to_hdf(self.table_db, f'{k}')
                print(f'{k} with code {v} has been saved to {self.table_db.parent}')

    def get_cot(self, commodity):

        with pd.HDFStore(self.table_db) as store:
            data_cot = store[commodity]

        return data_cot





