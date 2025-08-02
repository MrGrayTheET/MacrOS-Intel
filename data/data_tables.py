import datetime
import os
from dotenv import load_dotenv

load_dotenv('.env')
from copy import deepcopy
from pathlib import Path
import pandas as pd
import requests
import toml
from myeia import API
import datetime as dt
from nasspython.nass_api import nass_param, nass_data
from sources.usda.nass_utils import calc_dates, clean_numeric_column as clean_col, clean_fas
from sources.usda.api_wrappers.psd_api import PSD_API, CommodityCode, CountryCode, AttributeCode, comms_dict
from sources.usda.api_wrappers.esr_api import USDAESR
from utils import walk_dict, key_to_name
from typing import List

store_folder = os.getenv('data_path')
os.chdir('C:\\Users\\nicho\PycharmProjects\macrOS-Int\\')
with open('C:\\Users\\nicho\PycharmProjects\macrOS-Int\data_sources\eia\data_mapping.toml') as m:
    table_mapping = toml.load(m)


# noinspection PyTypeChecker
class TableClient:

    def __init__(self, client, data_folder, db_file_name, key_prefix=None, map_file=None, api_data_col=None,
                 rename_on_load=False):
        self.client = client
        self.data_folder = Path(data_folder)
        self.table_db = Path(self.data_folder, db_file_name)
        self.data_col = api_data_col
        self.rename = rename_on_load
        self.app_path = os.getenv('APP_PATH')

        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        if key_prefix is not None:
            self.prefix = key_prefix

        self.client_params = {}

        if map_file:
            with open(map_file) as m:
                self.table_map_all = toml.load(m)
            self.mapping = self.table_map_all[self.prefix] if self.prefix else self.table_map_all

        return

    def available_keys(self):
        with pd.HDFStore(self.table_db) as store:
            if self.prefix:
                tables = [k[1 + len(self.prefix):] for k in store.keys() if
                          k[1:].startswith(self.prefix)]
            else:
                tables = [k[1:] for k in store.keys()]
        return tables

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get_key(self, key, use_prefix=True, use_simple_name=True):
        with pd.HDFStore(self.table_db) as store:
            key = f'{self.prefix}/{key}' if self.prefix and use_prefix else f'{key}'
            table = store[f'{key}']
        col_value = table.columns[0] if self.data_col is None else self.data_col
        if use_simple_name:
            new_name = key_to_name(key)
            table = table.rename({col_value: new_name}, axis=1)

        return table

    def get_keys(self, keys: list, use_prefix=True, use_simple_name=True):
        dfs = []
        with pd.HDFStore(self.table_db) as store:
            for k in keys:
                key = f'{self.prefix}/{k}' if self.prefix and use_prefix else f'{k}'
                table = store[f'{key}']
                if isinstance(table, pd.DataFrame):

                    if use_simple_name:
                        col = key_to_name(k)
                        if col in table.columns:
                            data = table[col]
                        elif self.data_col in table.columns and col not in table.columns:
                            data = table[self.data_col]
                        elif len(table.columns) == 1:
                            data_col_value = table.columns[0] if self.data_col is None else self.data_col
                            data = table.rename({data_col_value: col}, axis=1)
                        else:
                            k_split = k.split('/')
                            data_col_value = self.mapping[k_split[1]][k_split[-1]]
                            data = table.rename({data_col_value: col}, axis=1)
                            data = data[col]
                    else:
                        data = table
                        pass

                elif isinstance(table, pd.Series):
                    table.rename(key_to_name(k), axis=1)

                dfs.append(data)

        key_df = pd.concat(dfs, axis=1)

        return key_df

    def get_cat(self, category, endswith='', rename_columns=False):
        keys = [f'{category}/{k}' for k, v in self.mapping[category].items() if v.endswith(endswith)]
        return self.get_keys(keys)

    def update_request_params(self, param_dict):
        self.client_params.update(param_dict)

    def local_update(self, update_file_path: str, new_key: str, load_func, use_prefix=False):
        df = load_func(update_file_path)
        new_key = f'{self.prefix}/{new_key}' if use_prefix else new_key
        df.to_hdf(self.table_db, f'{new_key}')

        return

    def update_from_dict(self, update_dict: dict, id_field='series_id', client_params={}, rename_column=False):
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

    def __getitem__(self, item):
        if isinstance(item, List):

            return self.get_keys(item, use_prefix=True, use_simple_name=self.rename)
        elif isinstance(item, str):
            return self.get_key(item, use_prefix=True, use_simple_name=self.rename)

market_fp = os.getenv('market_data_path')
cot_fp = os.getenv('cot_path')

# noinspection PyTypeChecker
class MarketTable:
    """Data access layer for market and COT data from HDF5 files."""

    def __init__(self, alias_file=None, initial_ticker=None, start_date=None, end_date=None, freq='1D'):
        self.market_table_db = Path(market_fp)
        self.cot_table_db = Path(cot_fp)
        self.ohlc_agg = {
            'Open': 'first',  # First open of the period
            'High': 'max',  # Highest high of the period
            'Low': 'min',  # Lowest low of the period
            'Close': 'last',  # Last close of the period
            'Volume': 'sum'  # Total volume of the period
        }
        if not alias_file:
            alias_file = 'C:\\Users\\nicho\PycharmProjects\macrOS-Int\\assets\plotting\chart_mappings.toml'
            try:
                with open(alias_file, 'r') as f:
                    self.ticker_map = toml.load(f)
            except FileNotFoundError:
                print(f"Warning: Alias file {alias_file} not found.")
                self.ticker_map = {'aliases': {}}
        else:
            with open(alias_file, 'r') as f:
                self.ticker_map = toml.load(f)

    def get_historical(self, ticker, start_date='2015-01-01', end_date=None, resample=False, interval='1D'):
        """Get market data for a ticker."""
        try:
            with pd.HDFStore(self.market_table_db, mode='r') as store:
                if ticker in self.ticker_map.keys():
                    key = self.ticker_map[ticker]['market_ticker']
                    data = store[key]
                elif ticker in [t[1:] for t in store.keys()]:
                    data = store[ticker]
                    data.index = pd.to_datetime(data.index) if not isinstance(data.index,
                                                                              pd.DatetimeIndex) else data.index

                else:
                    print(f'Ticker {ticker} not found. Available: {list(store.keys())}')
                    return None

            df = data.resample(interval).apply(self.ohlc_agg) if resample else data

            if not start_date and not end_date:
                return df
            elif start_date and not end_date:
                mask = data.index >= dt.datetime.strptime(start_date, '%y-%m-%d')
            elif end_date and not start_date:
                mask = data.index <= dt.datetime.strptime(end_date, '%y-%m-%d')
            else:
                return df.loc[start_date:end_date]

            return data.loc[mask]
        except Exception as e:
            print(f"Error accessing market data: {e}")
            return None

    def get_cot(self, commodity, filter_by_type='F_ALL', start_date=None, end_date=None):
        """Get COT data for a commodity."""
        try:
            with pd.HDFStore(self.cot_table_db, mode='r') as store:
                if commodity in self.ticker_map.keys():
                    key = self.ticker_map[commodity]['cot_name']
                    data = store[key]
                elif commodity in [t[1:] for t in store.keys()]:
                    data = store[commodity]
                else:
                    print(f'Commodity {commodity} not found. Available: {list(store.keys())}')
                    return None

            data['Date'] = pd.to_datetime(data['date'])
            data = data.loc[data['key_type'] == filter_by_type].sort_values(by='Date')

            if not start_date and not end_date:
                return data
            elif start_date and not end_date:
                mask = data.index >= dt.datetime.strptime(start_date)
            elif end_date and not start_date:
                mask = data.index <= dt.datetime.strptime(end_date)
            else:
                return data.loc[start_date:end_date]

            return data.loc[mask]

        except Exception as e:
            print(f"Error accessing COT data: {e}")
            return None


class EIATable(TableClient):

    def __init__(self, commodity, rename_key_cols=True):
        super().__init__(API().get_series,
                         store_folder,
                         'eia_data.h5',
                         key_prefix=commodity,
                         map_file='C:\\Users\\nicho\PycharmProjects\macrOS-Int\data_sources\eia\data_mapping.toml',
                         rename_on_load=rename_key_cols)
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
                            new_name = names[-2] + '_' + names[-1]
                        else:
                            new_name = names[-1]
                        if isinstance(srs,
                                      pd.DataFrame):
                            srs.rename({srs.columns[0]: new_name}, inplace=True)
                        else:
                            srs.rename(new_name, axis=1, inplace=True)

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


nass_info = {
    'key': os.getenv('NASS_TOKEN'),
    'client': lambda x: nass_data(os.getenv('NASS_TOKEN'), **x)}

# National Agricultural Stats
class NASSTable(TableClient):
    with open('C:\\Users\\nicho\PycharmProjects\macrOS-Int\data_sources\\usda\data_mapping.toml') as map_file:
        table_map = toml.load(map_file)

    def __init__(self, source_desc,

                 sector_desc,
                 group_desc,
                 commodity_desc, freq_desc=None, prefix=None):
        self.prefix = self.table = prefix
        super().__init__(nass_info['client'], data_folder=store_folder, db_file_name="nass_agri_stats.hd5",
                         api_data_col='Value', key_prefix=prefix, rename_on_load=True)
        self.client_params = {
            'source_desc': source_desc,
            'sector_desc': sector_desc,
            'group_desc': group_desc,
            'commodity_desc': commodity_desc,
            'agg_level_desc': 'NATIONAL',
            'domain_desc': 'TOTAL',
        }

        self.mapping = self.table_map[prefix]

        if freq_desc is not None:
            self.client_params.update({'freq_desc': freq_desc})

        return

    def get_descs(self, param_value='short_desc'):

        return nass_param(param=param_value, **self.client_params)

    def update_table(self, short_desc, freq_desc_preference='MONTHLY', key=None, agg_level=None, state=None,
                     county=None, ):
        agg_level = 'NATIONAL' if agg_level is None else agg_level

        params = {
            'short_desc': short_desc,
            'agg_level_desc': agg_level
        }

        if agg_level != 'NATIONAL':
            params.update({
                'state_name': state,
                'county_name': county
            })

        try:
            data = self.client(params)['data']

        except Exception as e:
            print(
                f'Series {short_desc} not able to be downloaded\n Exception {e}'
            )
            return
        else:
            df = pd.DataFrame(data)
            if 'freq_desc' in df.columns:
                freqs = df['freq_desc'].unique().tolist()
                if len(freqs) > 1:
                    if freq_desc_preference in freqs:
                        df = df.loc[df['freq_desc'].str.upper() == freq_desc_preference.upper()]
                    else:
                        # Save the frequency with most datapoints
                        biggest_len = 0
                        largest_fq = None
                        for fq in freqs:
                            fq_len = len(df.loc[df['freq_desc'] == fq])
                            biggest_len, largest_fq = (fq_len, fq) if fq_len > biggest_len else (
                                biggest_len, largest_fq)
                        if largest_fq == 'POINT IN TIME':
                            # Renamed for locating values
                            df = df.loc[df['freq_desc'] == largest_fq]
                            largest_fq = 'MONTHLY'
                            df.loc[:, 'freq_desc'] = largest_fq

                try:
                    df.index = calc_dates(df)
                except Exception as e:
                    print(
                        f'Series {short_desc} failed at calculating datetimes step\nError Calculating datetimes, the following columns are in the dataframe:\n{df.columns}\n '
                        f'The following was the cause of the exception \n{e}\n')
                else:
                    if key is None: key = f'{short_desc.lower()}'
                    df['date'] = df.index

            else:
                # 28 used as date placeholder since the release date is not located in report
                df = df[(1 < df['end_code'].astype(int) < 12)]
                df['date'] = df['year'].astype(str) + '-' + df['end_code'].astype(str) + '-28'
                df.index = pd.to_datetime(df['date'])

            if key is not None and 'date' in df.columns:
                df.sort_index(inplace=True)
                df[key_to_name(key)] = clean_col(df['Value'])
                df[['date', 'commodity_desc', key_to_name(key), 'year', 'end_code']].to_hdf(self.table_db,
                                                                                            key=f'{self.prefix}/{key}')
                print(f'USDA Statistic {short_desc} saved to {Path(self.table_db)} in key {self.prefix}/{key}')

            return df

    def update_all(self):
        failed = {}
        for k, v in walk_dict(self.mapping):
            n_keys = len(k)
            key = f'{k[0]}/{k[-1]}' if n_keys >= 2 else f'{k[-1]}'
            try:
                self.update_table(short_desc=v, key=key)

            except Exception as e:
                print(f"Exception Raised with the following error: {e}")
                failed.update({key: v})

            else:
                print(self[f'{key}'])

        if failed:
            print(f'The following keys failed to update: {failed}')
            return failed
        else:
            print('All keys were successfully updated')

        return

    def __str__(self):
        params = deepcopy(self.client_params)
        params.update({'api_key': os.getenv('NASS_TOKEN')})
        return str(params)

# Foreign Agricultural Service
# Can also pull from NASSTable without calling any prefix or arguments
class FASTable(TableClient):
    psd = PSD_API()

    comms, attrs, countries = CommodityCode, AttributeCode, CountryCode

    # Shares the same table as NASSTable for convenience’s sake as they link to the same commodities and agency
    def __init__(self, commodity=None):
        super().__init__(client=self.psd, data_folder=os.getenv('DATA_PATH'), db_file_name='nass_agri_stats.hd5',
                         key_prefix=commodity, rename_on_load=False)
        self.esr = USDAESR(os.getenv('FAS_TOKEN'))
        self.prefix = commodity if commodity else None
        self.code = comms_dict[commodity] if commodity else None
        self.country = CountryCode.UNITED_STATES
        self.type = 'livestock' if (self.prefix == 'hog' or self.prefix == 'cattle') else 'grain'
        with open(Path(self.app_path, 'data_sources', 'usda', "esr_map.toml")) as fp:
            data_map = toml.load(fp)
            self.aliases = data_map['aliases']
            self.esr_codes = data_map['esr']
        self.rename = False

    def psd_summary(self, commodity=None, start_year=1982, end_year=2025):
        years = list(range(start_year, end_year))
        sd_summary = self.client.get_supply_demand_summary(commodity=self.code, country=self.country, years=years)
        sd_summary.to_hdf(self.table_db, key=f'{self.prefix}/psd/summary')

        return sd_summary

    # noinspection PyTypeChecker
    def update_table_local(self, data_folder, key_type="imports"):
        from sources.usda.nass_utils import clean_import_df
        aliases = self.aliases
        loading_func = lambda x: clean_fas(x)
        files = os.listdir(data_folder)
        tables_updated = []
        file_aliases = aliases.get(key_type, {})
        for f in files:
            split_str = f.split('_')
            label, split_key = split_str[0], split_str[1]
            split_key = split_key.replace('.csv', '')
            if file_aliases.get(label, False) and split_key == key_type:
                prefix = aliases[key_type][label]
                # noinspection PyTypeChecker
                self.local_update(Path(data_folder, f), new_key=f'{prefix}/{key_type}', use_prefix=False,
                                  load_func=clean_fas)
                tables_updated.append(prefix)
        print(f"Tables Updated: \n{[table for table in tables_updated]}")

        return

    def get_top_sources_by_year(self, commodity, data_type='exports', selected_year=2024):
        if commodity in self.aliases.keys():
            commodity_cat = self.aliases[commodity]
        else:
            commodity_cat = commodity

        df_sources = self[f'{commodity_cat}/{data_type}']
        year_data = df_sources[df_sources.index.get_level_values('date').year == selected_year].copy()
        country_totals = year_data.groupby('Partner')['value'].sum().sort_values(ascending=False)

        return country_totals

    def ESR_top_sources_update(self, commodity, market_year, top_n=10, return_key=True, use_prev_mkt_year=False):
        src_year = market_year - 1 if use_prev_mkt_year or market_year == datetime.datetime.today().year else market_year
        top_res = self.get_top_sources_by_year(commodity, selected_year=market_year)
        top_destinations = top_res.index.to_list()

        if commodity.lower() in self.esr_codes['alias'].keys():
            commodity_code = self.esr_codes['commodities'][self.esr_codes['alias'][commodity]]
            table_name = commodity.lower()
        else:
            commodity_code = self.esr_codes[commodity]
            table_name = self.aliases[commodity]

        dest_dfs = []
        failed_countries = []

        for t in top_destinations[:top_n]:
            t = t[:-3] if t.endswith('(*)') else t[:]
            if self.esr_codes['countries'].get(t.upper(), False):
                country_code = self.esr_codes['countries'][t.upper()]
                country_df = self.esr.get_export_sales(commodity_code=commodity_code, country_code=country_code,
                                                       market_year=market_year)
                try:
                    country_df.index = pd.to_datetime(country_df['weekEndingDate'])
                except KeyError:
                    print(f"Failed to get country {t}")
                    if country_df.empty:
                        print('Failed to get data from server')
                    failed_countries.append(t)
                else:
                    country_df['country'] = t
                    country_df['commodity'] = commodity
                    dest_dfs.append(country_df)

            else:
                print(f'Failed to get country {t}')
                failed_countries.append(t)

        if dest_dfs:
            export_year_df = pd.concat(dest_dfs, axis=0)
            export_year_df.to_hdf(self.table_db, f'{table_name}/exports/{market_year}')
            return_data  = export_year_df if return_key else None
            return return_data

        else:
            print(f'Failed to get data for year {market_year} ')


class WeatherTable(TableClient):

    def __init__(self, client, data_folder, db_file_name, key_prefix=None, map_file=None):
        return
