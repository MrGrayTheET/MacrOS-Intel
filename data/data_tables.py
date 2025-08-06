import datetime
import os
from pathlib import Path
import pandas as pd
import requests
import toml
from copy import deepcopy
import numpy as np
from enum import Enum
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple


# Import from config instead of using dotenv directly
try:
    from config import (
        DATA_PATH, MARKET_DATA_PATH, COT_PATH,
        NASS_TOKEN, FAS_TOKEN, EIA_KEY,
        load_mapping, PROJECT_ROOT
    )
except ImportError:
    # Fallback for when config.py doesn't exist
    from dotenv import load_dotenv

    load_dotenv()
    DATA_PATH = os.getenv('data_path', './data')
    MARKET_DATA_PATH = os.getenv('market_data_path', './data/market')
    COT_PATH = os.getenv('cot_path', './data/cot')
    NASS_TOKEN = os.getenv('NASS_TOKEN', '')
    PROJECT_ROOT = Path(__file__).parent.parent

# Import external APIs
try:
    from myeia import API
    from nasspython.nass_api import nass_param, nass_data
except ImportError as e:
    print(f"Warning: Could not import external APIs: {e}")
    API = None
    nass_param = None
    nass_data = None

# Import local modules with proper paths
try:
    # Update import paths to match your structure
    from data.sources.usda.nass_utils import calc_dates, clean_numeric_column as clean_col, clean_fas
    from data.sources.usda.api_wrappers.psd_api import PSD_API, CommodityCode, CountryCode, AttributeCode, comms_dict, rev_lookup, valid_codes, code_lookup
    from data.sources.usda.api_wrappers.esr_api import USDAESR
except ImportError as e:
    print(f"Warning: Could not import USDA modules: {e}")
    # Provide None or mock objects
    calc_dates = None
    clean_col = None
    clean_fas = None
    PSD_API = None
    USDAESR = None

# Import utilities
try:
    from utils import walk_dict, key_to_name
except ImportError:
    # Provide simple fallback implementations
    def walk_dict(d, parent_key=()):
        for k, v in d.items():
            full_key = parent_key + (k,)
            if isinstance(v, dict):
                yield from walk_dict(v, full_key)
            else:
                yield full_key, v


    def key_to_name(table_key):
        key_parts = table_key.rsplit('/')
        name = f"{key_parts[-1]}_{key_parts[-2]}"
        return name

# Store folder
store_folder = Path(DATA_PATH)
store_folder.mkdir(parents=True, exist_ok=True)

# Load table mapping with proper path
try:
    mapping_path = PROJECT_ROOT / 'data' / 'sources' / 'eia' / 'data_mapping.toml'
    if mapping_path.exists():
        with open(mapping_path) as m:
            table_mapping = toml.load(m)
    else:
        print(f"Warning: EIA mapping file not found at {mapping_path}")
        table_mapping = {}
except Exception as e:
    print(f"Error loading table mapping: {e}")
    table_mapping = {}


class TableClient:
    """Base class for table data access"""

    def __init__(self, client, data_folder, db_file_name, key_prefix=None, map_file=None, api_data_col=None,
                 rename_on_load=False):
        self.client = client
        self.data_folder = Path(data_folder)
        self.table_db = Path(self.data_folder, db_file_name)
        self.data_col = api_data_col
        self.rename = rename_on_load
        self.app_path = PROJECT_ROOT

        # Create data folder if it doesn't exist
        self.data_folder.mkdir(parents=True, exist_ok=True)

        if key_prefix is not None:
            self.prefix = key_prefix

        self.client_params = {}

        if map_file:
            try:
                with open(map_file) as m:
                    self.table_map_all = toml.load(m)
                self.mapping = self.table_map_all[self.prefix] if self.prefix else self.table_map_all
            except Exception as e:
                print(f"Error loading map file {map_file}: {e}")
                self.mapping = {}
        else:
            self.mapping = {}

    def available_keys(self):
        """Get available keys from HDF5 store"""
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                all_keys = list(store.keys())
                print(f"Raw store keys: {all_keys}")  # Debug logging
                
                if hasattr(self, 'prefix') and self.prefix:
                    # Filter keys that start with the prefix
                    prefix_with_slash = f"/{self.prefix}/"
                    tables = []
                    for k in all_keys:
                        if k.startswith(prefix_with_slash):
                            # Remove the prefix and leading slash to get the relative key
                            relative_key = k[len(prefix_with_slash):]
                            tables.append(relative_key)
                    print(f"Filtered keys for prefix '{self.prefix}': {tables}")  # Debug logging
                else:
                    # Remove leading slash from all keys
                    tables = [k[1:] if k.startswith('/') else k for k in all_keys]
                    print(f"All keys (no prefix filter): {tables}")  # Debug logging
                    
            return tables
        except Exception as e:
            print(f"Error accessing HDF5 store: {e}")
            return []

    def get_key(self, key, use_prefix=True, use_simple_name=True):
        """Get data for a specific key"""
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                # Construct the full key path
                if hasattr(self, 'prefix') and self.prefix and use_prefix:
                    full_key = f'{self.prefix}/{key}'
                else:
                    full_key = key
                
                # Ensure key starts with '/' for HDF5 compatibility
                if not full_key.startswith('/'):
                    full_key = f'/{full_key}'
                
                print(f"Attempting to access key: {full_key}")  # Debug logging
                print(f"Available keys: {list(store.keys())}")  # Debug logging
                
                table = store[full_key]

            col_value = table.columns[0] if self.data_col is None else self.data_col
            if use_simple_name:
                new_name = key_to_name(key)
                table = table.rename({col_value: new_name}, axis=1)

            return table
        except Exception as e:
            print(f"Error getting key {key} (full_key: {full_key if 'full_key' in locals() else 'unknown'}): {e}")
            return pd.DataFrame()

    def get_keys(self, keys: list, use_prefix=True, use_simple_name=True):
        """Get data for multiple keys"""
        dfs = []
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                print(f"Available keys in store: {list(store.keys())}")  # Debug logging
                
                for k in keys:
                    # Construct the full key path
                    if hasattr(self, 'prefix') and self.prefix and use_prefix:
                        full_key = f'{self.prefix}/{k}'
                    else:
                        full_key = k
                    
                    # Ensure key starts with '/' for HDF5 compatibility
                    if not full_key.startswith('/'):
                        full_key = f'/{full_key}'
                    
                    try:
                        print(f"Attempting to access key: {full_key}")  # Debug logging
                        table = store[full_key]
                        if isinstance(table, pd.DataFrame):
                            if use_simple_name:
                                col = key_to_name(k)
                                # Handle column renaming logic
                                if len(table.columns) == 1:
                                    data = table.rename({table.columns[0]: col}, axis=1)
                                else:
                                    data = table
                        elif isinstance(table, pd.Series):
                            table.rename(key_to_name(k), axis=1)
                            data = table
                        else:
                            data = table

                        dfs.append(data)
                    except KeyError:
                        print(f"Key {full_key} not found in store")
                        continue

            if dfs:
                key_df = pd.concat(dfs, axis=1)
                return key_df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Error getting keys: {e}")
            return pd.DataFrame()

    def rename_table_keys(self, old_pattern: str, new_pattern: str, dry_run: bool = True) -> Dict[str, str]:
        """
        Rename tables in HDF5 store by pattern matching.
        
        Args:
            old_pattern: Pattern to match in existing keys (e.g., 'soybean')
            new_pattern: Replacement pattern (e.g., 'soybeans') 
            dry_run: If True, only show what would be renamed without making changes
            
        Returns:
            Dict mapping old keys to new keys
        """
        rename_map = {}
        
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                all_keys = list(store.keys())
                
                # Find keys that match the pattern
                matching_keys = [k for k in all_keys if old_pattern in k]
                
                if not matching_keys:
                    print(f"No keys found matching pattern '{old_pattern}'")
                    return rename_map
                
                print(f"Found {len(matching_keys)} keys matching pattern '{old_pattern}':")
                for key in matching_keys:
                    new_key = key.replace(old_pattern, new_pattern)
                    rename_map[key] = new_key
                    print(f"  {key} -> {new_key}")
                
                if dry_run:
                    print("\nDry run mode - no changes made. Set dry_run=False to apply changes.")
                    return rename_map
            
            # Actually perform the rename operation
            print(f"\nRenaming {len(rename_map)} tables...")
            
            with pd.HDFStore(self.table_db, mode='a') as store:
                for old_key, new_key in rename_map.items():
                    try:
                        # Read data from old key
                        data = store[old_key]
                        
                        # Write to new key
                        store[new_key] = data
                        print(f"✓ Copied {old_key} -> {new_key}")
                        
                        # Remove old key
                        del store[old_key]
                        print(f"✓ Removed {old_key}")
                        
                    except Exception as e:
                        print(f"✗ Error renaming {old_key} -> {new_key}: {e}")
                        # Try to clean up if new key was created but old key couldn't be deleted
                        try:
                            if new_key in store:
                                del store[new_key]
                        except:
                            pass
            
            print(f"\nCompleted renaming {len(rename_map)} tables")
            
        except Exception as e:
            print(f"Error during rename operation: {e}")
            
        return rename_map

    def list_all_tables(self) -> List[str]:
        """List all tables in the HDF5 store with their sizes."""
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                all_keys = list(store.keys())
                
                print(f"HDF5 Store: {self.table_db}")
                print(f"Total tables: {len(all_keys)}")
                print("-" * 60)
                
                for key in sorted(all_keys):
                    try:
                        data = store[key]
                        if hasattr(data, 'shape'):
                            shape_info = f"{data.shape[0]} rows x {data.shape[1]} cols"
                        else:
                            shape_info = f"{len(data)} items"
                        print(f"{key:<40} {shape_info}")
                    except Exception as e:
                        print(f"{key:<40} Error: {e}")
                        
                return all_keys
                
        except Exception as e:
            print(f"Error listing tables: {e}")
            return []

    def __getitem__(self, item):
        """Allow dict-like access"""
        if isinstance(item, list):
            return self.get_keys(item, use_prefix=True, use_simple_name=self.rename)
        elif isinstance(item, str):
            return self.get_key(item, use_prefix=True, use_simple_name=self.rename)


class MarketTable:
    """Data access layer for market and COT data from HDF5 files."""

    def __init__(self, alias_file=None, initial_ticker=None, start_date=None, end_date=None, freq='1D'):
        self.market_table_db = Path(MARKET_DATA_PATH) / 'market_data.h5'
        self.cot_table_db = Path(COT_PATH) / 'cot_data.h5'
        self.ohlc_agg = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        # Load ticker mappings
        if not alias_file:
            alias_file = PROJECT_ROOT / 'components' / 'plotting' / 'chart_mappings.toml'

        try:
            if Path(alias_file).exists():
                with open(alias_file, 'r') as f:
                    self.ticker_map = toml.load(f)
            else:
                print(f"Warning: Alias file {alias_file} not found.")
                self.ticker_map = {}
        except Exception as e:
            print(f"Error loading alias file: {e}")
            self.ticker_map = {}

    def get_historical(self, ticker, start_date='2015-01-01', end_date=None, resample=False, interval='1D'):
        """Get market data for a ticker."""
        try:
            if not self.market_table_db.exists():
                print(f"Market data file not found: {self.market_table_db}")
                return None

            with pd.HDFStore(self.market_table_db, mode='r') as store:
                if ticker in self.ticker_map:
                    key = self.ticker_map[ticker].get('market_ticker', ticker)
                else:
                    key = ticker

                if key in store:
                    data = store[key]
                elif f'/{key}' in store:
                    data = store[f'/{key}']
                else:
                    print(f'Ticker {ticker} not found. Available: {list(store.keys())}')
                    return None

            # Process data
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Apply date filtering
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]

            # Resample if requested
            if resample and interval:
                data = data.resample(interval).agg(self.ohlc_agg)

            return data

        except Exception as e:
            print(f"Error accessing market data: {e}")
            return None

    def get_cot(self, commodity, filter_by_type='F_ALL', start_date=None, end_date=None):
        """Get COT data for a commodity."""
        try:
            if not self.cot_table_db.exists():
                print(f"COT data file not found: {self.cot_table_db}")
                return None

            with pd.HDFStore(self.cot_table_db, mode='r') as store:
                if commodity in self.ticker_map:
                    key = self.ticker_map[commodity].get('cot_name', commodity)
                else:
                    key = commodity

                if key in store:
                    data = store[key]
                elif f'/{key}' in store:
                    data = store[f'/{key}']
                else:
                    print(f'Commodity {commodity} not found. Available: {list(store.keys())}')
                    return None

            # Process COT data
            if 'date' in data.columns:
                data['Date'] = pd.to_datetime(data['date'])

            if 'key_type' in data.columns:
                data = data[data['key_type'] == filter_by_type]

            data = data.sort_values(by='Date' if 'Date' in data.columns else data.index)

            # Apply date filtering
            if start_date:
                data = data[data['Date'] >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data['Date'] <= pd.to_datetime(end_date)]

            return data

        except Exception as e:
            print(f"Error accessing COT data: {e}")
            return None


class EIATable(TableClient):
    """EIA data table client"""

    def __init__(self, commodity, rename_key_cols=True):
        # Use the loaded mapping
        map_file = PROJECT_ROOT / 'data' / 'sources' / 'eia' / 'data_mapping.toml'

        if API is None:
            print("Warning: EIA API not available. Using mock client.")
            client = lambda **kwargs: pd.DataFrame()  # Mock client
        else:
            client = API().get_series

        super().__init__(
            client,
            store_folder,
            'eia_data.h5',
            key_prefix=commodity,
            map_file=str(map_file),
            rename_on_load=rename_key_cols
        )
        self.commodity = self.prefix = commodity


nass_info = {
    'key': os.getenv('NASS_TOKEN'),
    'client': lambda x: nass_data(os.getenv('NASS_TOKEN'), **x)}

# National Agricultural Stats
class NASSTable(TableClient):
    with open(Path(PROJECT_ROOT, 'data', 'sources','usda' ,'data_mapping.toml')) as map_file:
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
        super().__init__(client=self.psd, data_folder=DATA_PATH, db_file_name='nass_agri_stats.hd5',
                         key_prefix=commodity, rename_on_load=False)
        self.esr = USDAESR()
        self.prefix = commodity if commodity else None
        self.code = comms_dict[commodity] if commodity else None
        self.country = CountryCode.UNITED_STATES
        self.type = 'livestock' if (self.prefix == 'hog' or self.prefix == 'cattle') else 'grain'
        with open(Path(self.app_path, 'data', 'sources/usda', "esr_map.toml")) as fp:
            data_map = toml.load(fp)
            self.aliases = data_map['aliases']
            self.esr_codes = data_map['esr']
        self.rename = False

    def update_psd(self, commodity=None, start_year=1982, end_year=2025):
        if isinstance(commodity,Enum):
            if rev_lookup.get(commodity, False):
                table_key = rev_lookup[commodity]
                code = commodity.value
            else:
                table_key = commodity.name.lower()
                code = commodity.value
        else:
            if comms_dict.get(commodity, False):
                table_key = commodity
                code = comms_dict.get(commodity)
            elif commodity in valid_codes:
                table_key = code_lookup[commodity].lower()
                code = commodity
            else:
                return print(f'Unable to find valid Commodity/Code for {commodity}')

        years = list(range(start_year, end_year))
        sd_summary = self.client.get_supply_demand_summary(commodity=code, country=self.country, years=years)
        sd_summary.to_hdf(self.table_db, key=f'{table_key}/psd/summary')

        return sd_summary

    # noinspection PyTypeChecker
    def update_table_local(self, data_folder, key_type="imports"):
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

        # Translate commodity table name to ESR commodity code
        # Step 1: commodity (e.g., "cattle") -> commodity_name (e.g., "beef") via esr.alias 
        # Step 2: commodity_name (e.g., "beef") -> commodity_code (e.g., "1701") via esr.commodities
        
        if commodity.lower() in self.esr_codes['alias'].keys():
            # Get the commodity name from the alias (cattle -> beef, hogs -> pork, etc.)
            commodity_name = self.esr_codes['alias'][commodity.lower()]
            # Get the commodity code from the commodity name (beef -> 1701, pork -> 1702, etc.)
            commodity_code = self.esr_codes['commodities'][commodity_name]
            table_name = commodity.lower()
        else:
            # Fallback: assume commodity is already the commodity name (beef, pork, etc.)
            commodity_code = self.esr_codes['commodities'].get(commodity.lower())
            if not commodity_code:
                raise ValueError(f"Unknown commodity: {commodity}. Available: {list(self.esr_codes['alias'].keys())}")
            table_name = commodity.lower()

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


class ESRTableClient(FASTable):
    """
    Extended TableClient specifically for ESR data handling.
    Includes methods for ESR-specific data processing and filtering.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_esr_data(self, commodity, year=None, country=None,
                     start_date=None, end_date=None):
        """
        Get ESR data with optional filtering.

        Args:
            commodity: Commodity name (e.g., 'wheat', 'corn', 'soybeans')
            year: Marketing year (optional)
            country: Country filter (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            pd.DataFrame: Filtered ESR data
        """
        # Construct key based on ESR format: {commodity}/exports/{year}
        if year:
            key = f"{commodity}/exports/{year}"
        else:
            # Get most recent year available
            available_keys = self.available_keys()
            commodity_keys = [k for k in available_keys if k.startswith(f"/{commodity}/exports/")]
            if not commodity_keys:
                return pd.DataFrame()

            # Get latest year
            years = [int(k.split('/')[-1]) for k in commodity_keys if k.split('/')[-1].isdigit()]
            if not years:
                return pd.DataFrame()

            latest_year = max(years)
            key = f"{commodity}/exports/{latest_year}"

        # Get base data
        data = self.get_key(key)

        if data is None or data.empty:
            return pd.DataFrame()

        # Apply filters
        if country:
            data = data[data['country'].str.contains(country, case=False, na=False)]

        if start_date:
            data = data[pd.to_datetime(data['weekEndingDate']) >= pd.to_datetime(start_date)]

        if end_date:
            data = data[pd.to_datetime(data['weekEndingDate']) <= pd.to_datetime(end_date)]

        return data

    def get_multi_year_esr_data(self, commodity, years=None, country=None, 
                               start_year=None, end_year=None):
        """
        Get ESR data concatenated from multiple years.

        Args:
            commodity: Commodity name
            years: List of years (if None, uses start_year/end_year or last 5 years)
            country: Optional country filter
            start_year: Start year for range (alternative to years list)
            end_year: End year for range (alternative to years list)

        Returns:
            pd.DataFrame: Concatenated ESR data
        """
        if years is None:
            if start_year and end_year:
                years = list(range(start_year, end_year + 1))
            else:
                current_year = pd.Timestamp.now().year
                years = list(range(current_year - 4, current_year + 1))

        all_data = []

        for year in years:
            year_data = self.get_esr_data(commodity, year, country)
            if not year_data.empty:
                year_data['marketing_year'] = year
                all_data.append(year_data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['marketing_year', 'weekEndingDate'])
            return combined_data

        return pd.DataFrame()

    def get_available_commodities(self):
        """Get list of available commodities in ESR data."""
        available_keys = self.available_keys()
        commodities = set()

        for key in available_keys:
            if '/exports/' in key:
                parts = key.split('/')
                if len(parts) >= 2:
                    commodity = parts[1]  # Remove leading /
                    commodities.add(commodity)

        return sorted(list(commodities))

    def get_available_years(self, commodity):
        """Get available years for a specific commodity."""
        available_keys = self.available_keys()
        years = set()

        for key in available_keys:
            if f"/{commodity}/exports/" in key:
                parts = key.split('/')
                if len(parts) >= 4 and parts[-1].isdigit():
                    years.add(int(parts[-1]))

        return sorted(list(years))

    def get_available_countries(self, commodity, year=None):
        """Get available countries for a commodity."""
        data = self.get_esr_data(commodity, year)
        if not data.empty and 'country' in data.columns:
            return sorted(data['country'].unique().tolist())
        return []
    
    def get_top_countries(self, commodity, metric='weeklyExports', top_n=10, 
                         start_year=None, end_year=None):
        """
        Get top N countries by export metric for dynamic menu population.
        
        Args:
            commodity: Commodity name
            metric: Metric to rank by ('weeklyExports', 'outstandingSales', etc.)
            top_n: Number of top countries to return
            start_year: Start year for analysis (optional)
            end_year: End year for analysis (optional)
            
        Returns:
            List of country names sorted by metric (descending)
        """
        try:
            # Get multi-year data for ranking
            data = self.get_multi_year_esr_data(commodity, start_year=start_year, end_year=end_year)
            
            if data.empty or metric not in data.columns:
                return []
            
            # Aggregate by country
            country_totals = data.groupby('country')[metric].sum().sort_values(ascending=False)
            
            # Return top N countries
            top_countries = country_totals.head(top_n).index.tolist()
            
            return top_countries
            
        except Exception as e:
            print(f"Error getting top countries: {e}")
            return []

    def aggregate_esr_data(self, data, group_by='country', time_period='weekly'):
        """
        Aggregate ESR data by different dimensions.

        Args:
            data: ESR DataFrame
            group_by: 'country', 'commodity', or 'month'
            time_period: 'weekly', 'monthly', 'quarterly'

        Returns:
            pd.DataFrame: Aggregated data
        """
        if data.empty:
            return pd.DataFrame()

        # Ensure weekEndingDate is datetime
        data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])

        # Create time grouping column
        if time_period == 'monthly':
            data['time_group'] = data['weekEndingDate'].dt.to_period('M')
        elif time_period == 'quarterly':
            data['time_group'] = data['weekEndingDate'].dt.to_period('Q')
        else:  # weekly
            data['time_group'] = data['weekEndingDate']

        # Define aggregation functions
        agg_functions = {
            'weeklyExports': 'sum',
            'accumulatedExports': 'last',
            'outstandingSales': 'last',
            'grossNewSales': 'sum',
            'currentMYNetSales': 'sum',
            'currentMYTotalCommitment': 'last',
            'nextMYOutstandingSales': 'last',
            'nextMYNetSales': 'sum'
        }

        # Group and aggregate
        if group_by == 'country':
            grouped = data.groupby(['time_group', 'country']).agg(agg_functions).reset_index()
        elif group_by == 'commodity':
            grouped = data.groupby(['time_group', 'commodity']).agg(agg_functions).reset_index()
        else:  # total
            grouped = data.groupby('time_group').agg(agg_functions).reset_index()

        return grouped

    async def ESR_multi_year_update(self, commodity: str, start_year: int, end_year: int, 
                                   top_n: int = 10, max_concurrent: int = 3) -> Dict[str, any]:
        """
        Update ESR data for multiple years using async/await for parallelization.
        Uses asyncio instead of threading because:
        - I/O-bound operations (API calls) benefit more from async
        - Better resource efficiency (single thread + event loop)
        - Natural error propagation and handling
        - Built-in concurrency control
        
        Args:
            commodity: Commodity name
            start_year: Starting marketing year
            end_year: Ending marketing year  
            top_n: Number of top countries to update (default: 10)
            max_concurrent: Maximum concurrent requests (default: 3)
            
        Returns:
            Dict containing success/failure results and summary statistics
        """
        years = list(range(start_year, end_year + 1))
        results = {
            'successful_years': [],
            'failed_years': [],
            'errors': {},
            'updated_tables': [],
            'summary': {}
        }
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def update_single_year(year: int) -> Tuple[int, bool, Optional[str]]:
            """Update a single year with error handling."""
            async with semaphore:  # Limit concurrent requests
                try:
                    print(f"Starting update for {commodity} {year}...")
                    
                    # Use existing synchronous method but wrap in executor if needed
                    # For now, we'll call the sync method in a thread pool
                    loop = asyncio.get_event_loop()
                    
                    # Call the existing ESR_top_sources_update method
                    result = await loop.run_in_executor(
                        None, 
                        lambda: self.ESR_top_sources_update(
                            commodity=commodity, 
                            market_year=year, 
                            top_n=top_n,
                            return_key=True
                        )
                    )
                    
                    if result is not None and not result.empty:
                        print(f"✓ Successfully updated {commodity} {year} - {len(result)} records")
                        return year, True, None
                    else:
                        error_msg = f"No data returned for {commodity} {year}"
                        print(f"⚠ Warning: {error_msg}")
                        return year, False, error_msg
                        
                except Exception as e:
                    error_msg = f"Failed to update {commodity} {year}: {str(e)}"
                    print(f"✗ Error: {error_msg}")
                    return year, False, error_msg
        
        # Execute updates concurrently
        print(f"Updating {commodity} data for years {start_year}-{end_year} (max {max_concurrent} concurrent)")
        
        try:
            # Create tasks for all years
            tasks = [update_single_year(year) for year in years]
            
            # Execute with progress tracking
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in completed_results:
                if isinstance(result, Exception):
                    error_year = "unknown"
                    results['failed_years'].append(error_year)
                    results['errors'][error_year] = str(result)
                else:
                    year, success, error_msg = result
                    if success:
                        results['successful_years'].append(year)
                        results['updated_tables'].append(f"{commodity}/exports/{year}")
                    else:
                        results['failed_years'].append(year)
                        if error_msg:
                            results['errors'][year] = error_msg
            
            # Generate summary
            total_years = len(years)
            successful_count = len(results['successful_years'])
            failed_count = len(results['failed_years'])
            
            results['summary'] = {
                'commodity': commodity,
                'year_range': f"{start_year}-{end_year}",
                'total_years_requested': total_years,
                'successful_updates': successful_count,
                'failed_updates': failed_count,
                'success_rate': f"{successful_count/total_years*100:.1f}%" if total_years > 0 else "0%",
                'top_n_countries': top_n,
                'max_concurrent_requests': max_concurrent
            }
            
            print(f"\n📊 Update Summary for {commodity}:")
            print(f"   Years requested: {total_years}")
            print(f"   Successful: {successful_count}")
            print(f"   Failed: {failed_count}")
            print(f"   Success rate: {results['summary']['success_rate']}")
            
            if results['failed_years']:
                print(f"   Failed years: {results['failed_years']}")
            
        except Exception as e:
            error_msg = f"Critical error during multi-year update: {str(e)}"
            print(f"💥 {error_msg}")
            results['errors']['critical'] = error_msg
            
        return results

    def update_esr_multi_year_sync(self, commodity: str, start_year: int, end_year: int, 
                                  top_n: int = 10, max_concurrent: int = 3) -> Dict[str, any]:
        """
        Synchronous wrapper for the async multi-year update method.
        
        Args:
            commodity: Commodity name
            start_year: Starting marketing year
            end_year: Ending marketing year
            top_n: Number of top countries to update (default: 10)
            max_concurrent: Maximum concurrent requests (default: 3)
            
        Returns:
            Dict containing update results
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, need to create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.ESR_multi_year_update(commodity, start_year, end_year, top_n, max_concurrent)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.ESR_multi_year_update(commodity, start_year, end_year, top_n, max_concurrent)
            )

    def update_all_ESR(self, start_year, end_year, top_n=10, max_concurrent=3):
        return


