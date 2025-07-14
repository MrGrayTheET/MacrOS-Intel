import os
from copy import deepcopy
from pathlib import Path
import pandas as pd
import requests
import toml
from myeia import API
from pyncei import NCEIBot
from nasspython.nass_api import nass_param, nass_data
from data_sources.usda.nass_utils import calc_dates, clean_numeric_column
from utils import walk_dict, key_to_name

store_folder = os.getenv('data_path')
os.chdir('C:\\Users\\nicho\PycharmProjects\macrOS-Int\\')
with open('C:\\Users\\nicho\PycharmProjects\macrOS-Int\data_sources\eia\data_mapping.toml') as m:
    table_mapping = toml.load(m)

eia_client = API()


# noinspection PyTypeChecker
class TableClient:

    def __init__(self, client, data_folder, db_file_name, key_prefix=None, map_file=None):
        self.client = client
        self.data_folder = Path(data_folder)
        self.table_db = Path(self.data_folder, db_file_name)
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        if key_prefix is not None:
            self.prefix = key_prefix

        self.tables = self.available_keys()
        self.client_params = {}

        if map_file:
            with open(map_file) as m:
                self.table_map_all = toml.load(m)
            self.mapping = self.table_map_all[self.prefix] if self.prefix else self.table_map_all

        return

    def available_keys(self):
        with pd.HDFStore(self.table_db) as store:
            tables = [k[1+len(self.prefix):] for k in store.keys() if k[1:].startswith(self.prefix)] if self.prefix else [k[1:] for k in
                                                                                                         store.keys()]
        return tables

    def get_key(self, key, use_prefix=True):
        with pd.HDFStore(self.table_db) as store:
            key = f'{self.prefix}/{key}' if self.prefix and use_prefix else f'{key}'
            table = store[f'{key}']

        return table

    def get_keys(self, keys: list, use_prefix=True,use_simple_name=True):
        dfs = []
        with pd.HDFStore(self.table_db) as store:
            for k in keys:
                key = f'{self.prefix}/{k}' if self.prefix and use_prefix else f'{k}'
                table = store[f'{key}']
                if use_simple_name:
                    table = table.rename({table.columns[0]: key_to_name(k)}, axis=1)
                dfs.append(table)

        key_df = pd.concat(dfs, axis=1)
        return key_df

    def get_cat(self, category,endswith='', rename_columns=False):
        keys = [f'{category}/{k}' for k,v in self.mapping[category].items() if v.endswith(endswith)]
        return self.get_keys(keys)

    def update_request_params(self, param_dict):
        self.client_params.update(param_dict)

    def merge_and_update(self, keys, new_alias):

        return

    def update_from_dict(self, update_dict: dict, id_field='series_id',client_params={}, rename_column=False):
        unprocessed_data = {}
        failed_series = []
        for k, v in walk_dict(update_dict):
            self.client_params.update({id_field: v})
            try:
                data = self.client(**self.client_params)
            except Exception as e:
                print(f'Failed to get key {k}')
                failed_series.append(k)
            else:
                if not isinstance(k, str) and len(k > 1):
                    key = f'{self.prefix}/{k[-2]}/{k[-1]}'
                    new_name = f'{k[-1]}_{k[-2]}'
                else:
                    key = f'{self.prefix}/{k}' if self.prefix else f'{k}'
                    new_name = f'{k}'
                if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                    if rename_column:
                        (data.rename({data.columns[0]: new_name}, inplace=True) if isinstance(data,
                                                                                             pd.DataFrame)
                         else data.rename(
                            new_name, inplace=True))
                    data.to_hdf(self.table_db, key)
                    print('Data successfully saved to ' + key)
                else:
                    unprocessed_data.update({k: data})
        if unprocessed_data or failed_series:
            print(f'Failed to get following series:\n{failed_series}')
            if unprocessed_data:
                print(f'Data retrieved from client was not in dataframe format')

            return (unprocessed_data, failed_series)
        else:
            print('Successfully updated from dict')

            return

class EIATable(TableClient):
    client = API()

    def __init__(self, commodity):
        super().__init__(eia_client.get_series,
                         store_folder,
                         'eia_data.h5',
                         key_prefix=commodity,
                         map_file='C:\\Users\\nicho\PycharmProjects\macrOS-Int\data_sources\eia\data_mapping.toml')
        self.commodity = self.prefix = commodity
        return

    def update_db(self, series, alias=None, rename_column=False, start_date=None, end_date=None, commodity=None):
        params = dict(series_id=series)
        commodity = self.commodity if commodity is None else commodity
        if start_date is not None:
            params.update(dict(start_date=start_date, end_date=end_date))

        try:
            srs = self.client(**params)
            if rename_column:
                if isinstance(srs, pd.DataFrame) or isinstance(srs, pd.Series):
                    if rename_column:
                        names = alias.rsplit('/')
                        if len(names) > 1:
                            new_name = names[-2]+'_'+names[-1]
                        else:
                            new_name = names[-1]
                        if isinstance(srs,
                                      pd.DataFrame):
                            srs.rename({srs.columns[0]: new_name}, inplace=True)
                        else:
                            srs.rename(new_name,axis=1, inplace=True)

        except requests.exceptions.HTTPError as e:
            print(f'failed to get {series} ')
            return False

        if not alias:
            if series in self.mapping['aliases'].keys():
                key_name = f'{commodity}/{self.mapping["aliases"][series]}'
            else:
                return srs

        else:
            key_name = f'{commodity}/{alias}'

            srs.to_hdf(self.table_db, key_name)
            print(f'Data Successfully Updated to {self.table_db}\n Key:{key_name}')

        return True

    def update_all(self, rename_columns=True):
        keyvals = walk_dict(self.mapping)

        for k, v in keyvals:
            self.update_db(v, alias=f'{k[0]}/{k[-1]}', rename_column=rename_columns)
        return

    def update_from_mapping(self, map_key, sub_keys=False):
        items = self.mapping[map_key].items() if not sub_keys else walk_dict(self.mapping[map_key])
        missing_tables = []
        for k, v in items:
            update = self.update_db(v, alias=f'{map_key}/{k}')
            if not update:
                missing_tables.append(f'{map_key}/{k}')

        return

    def update_futures(self, contracts=(1, 4), product=None):
        codes = []
        futs_codes = walk_dict(table_mapping['futs']['aliases'])
        update_dict = {}
        for key, code in futs_codes:
            for i in range(contracts[0], contracts[1] + 1):
                if 'PET' not in key:
                    series_id = f'{key[-1]}.{code}{i}.D'
                    alias = f'futures/contract_{i}'
                    update_dict.update({key: series_id})
                    commodity = key[-1]
                else:
                    series_id = f'{key[-2]}.{code}{i}.D'
                    alias = f'{key[-1]}/futures/contract_{i}'
                    commodity = key[-2]

                self.update_db(series_id, alias, commodity=commodity)

        return



nass_key = os.getenv('NASS_KEY')

nass_client = lambda x: nass_data(os.getenv('NASS_KEY'), **x)

class NASSTable(TableClient):

    def __init__(self, source_desc,

                 sector_desc,
                 group_desc,
                 commodity_desc, freq_desc=None):
        super().__init__(nass_client, data_folder=store_folder, db_file_name="nass_agri_stats.hd5")

        self.table_params = {
            'api_key': nass_key,
            'source_desc': source_desc,
            'sector_desc': sector_desc,
            'group_desc': group_desc,
            'commodity': commodity_desc
        }

        self.table = commodity_desc
        if freq_desc is not None:
            self.table_params.update({'freq_desc': freq_desc})

        return

    def get_descs(self, param_value='short_desc'):

        return nass_param(param=param_value, **self.table_params)

    def download_table(self, short_desc, key=None):
        params = deepcopy(self.table_params)
        params.update({'short_desc': short_desc})
        data = nass_data(**params)['data']
        df = pd.DataFrame(data)
        try:
            df.index = calc_dates(df)
        except Exception as e:
            print('Error Trying to create datetimeindex')
            print(f'{e}')
        else:
            mask = df.apply(lambda col: col.astype(str).str.strip().eq('').all())
            df = df.loc[:, ~mask]

        df['Value'] = clean_numeric_column(df['Value'])
        df.sort_index(inplace=True)
        if key is not None:
            df.to_hdf(self.table_db, key=f'{self.table}/{key}')
            print(f'USDA Statistic {short_desc} saved to {Path(self.table_db)} in key /{self.table}/{key}')

        return df

class WeatherTable(TableClient):

    def __init__(self, client, data_folder, db_file_name, key_prefix=None, map_file=None ):

        return


