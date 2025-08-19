import os
from datetime import date

import requests
from dateutil.relativedelta import relativedelta
import pandas as pd
from myeia import API
from dotenv import load_dotenv
from config import DOT_ENV
from data.sources.eia.EIA_API import EIAClient
# -----------------------------
# Config
# -----------------------------
load_dotenv(DOT_ENV)
eia = API()  # reads EIA_TOKEN from .env
  # API v2 route for natural gas summary (lsum)
FREQ = "monthly"

# State list (50 + DC)
STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY",
    "LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH",
    "OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

SECTORS = {
    "end_use_total": "N3060",   # Delivered to consumers (total)
    "residential":   "N3010",
    "commercial":    "N3020",
    "industrial":    "N3035",
    "electric_power":"N3045",
}

# Optional: Census regions for aggregation
CENSUS_REGIONS = {
    "Northeast": ["CT","ME","MA","NH","RI","VT","NJ","NY","PA"],
    "Midwest":   ["IL","IN","MI","OH","WI","IA","KS","MN","MO","NE","ND","SD"],
    "South":     ["DE","DC","FL","GA","MD","NC","SC","VA","WV","AL","KY","MS","TN","AR","LA","OK","TX"],
    "West":      ["AZ","CO","ID","MT","NV","NM","UT","WY","AK","CA","HI","OR","WA"],
}


API_KEY = os.getenv("EIA_TOKEN")  # or hardcode
BASE = "https://api.eia.gov/v2/natural-gas/sum/lsum/data/"

consumption_processes = ["VC0", "VIN", "VCS", "VRS", "VEU", "VGT"]

states = ["AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN",
          "KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ",
          "NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA",
          "WI","WV","WY"]

states_plus_usa = ["NUS"] + [f"S{st}" for st in states]
duoareas_dict = {'states_plus': states_plus_usa, 'offshore': ['R3FM'] }

def aggregate_regions(wide_states_df, region_map=CENSUS_REGIONS):
    """
    Sum states -> regions (keeps monthly frequency).
    df_states_tidy must be (date, state, mmcf).
    """
    # Wide by state, then sum per region
    reg_frames = []
    for region, st_list in region_map.items():
        reg = wide_states_df[st_list].sum(axis=1).to_frame(name=region)
        reg_frames.append(reg)
    return pd.concat(reg_frames, axis=1)


class NatGasHelper:

    def __init__(self):
        self.routes = {'summary':("sum", "lsum"), 'offshore_production': ('prod', 'off'), 'production': ('prod')}
        self.top_producers = {
            "STX": "Texas",
            "SPA": "Pennsylvania",
            "SLA": "Louisiana",
            "SWV": "West Virginia",
            "SNM": "New Mexico",
            "SOK": "Oklahoma",
            "SCO": "Colorado",
            "SOH": "Ohio",
            "SWY": "Wyoming",
            "SND": "North Dakota",
            "OFF": "Federal Offshore - Gulf of Mexico"
        }

        self.client = EIAClient(os.getenv('EIA_TOKEN')).natural_gas
        return

    def consumption_breakdown(self):
        all_rows = self.client.get_data_by_route(*self.routes['summary'], ['value'], facets={"process":consumption_processes, "duoarea":states_plus_usa}, get_all=True)

        # -> DataFrame (optional)
        resp_df = pd.DataFrame(all_rows)
        # Optional: clean up + helper columns
        proc_map = {"VC0": "Total", "VIN": "Industrial", "VCS": "Commercial",
                    "VRS": "Residential", "VEU": "Electric Power", "VGT": "Delivered"}

        if not resp_df.empty:
            # period comes as 'YYYY-MM'; keep as string or convert to month-end
            resp_df["sector"] = resp_df["process"].map(proc_map)
            # duoarea: NUS=US, SXX=state
            resp_df["state"] = resp_df["duoarea"].str[1:]
            resp_df.loc[resp_df["duoarea"] == "NUS", "state"] = "US"
            resp_df["value"] = pd.to_numeric(resp_df["value"], errors="coerce")

        resp_df["period"] = pd.to_datetime(resp_df["period"] + "-01").dt.to_period("M").dt.to_timestamp("M")
        resp_df["value"] = pd.to_numeric(resp_df["value"], errors="coerce")
        sector_order = ["Total","Delivered", "Industrial", "Commercial", "Residential", "Electric Power"]
        wide = (
            resp_df.groupby(["period", "state", "sector"], as_index=False)["value"].sum()
            .assign(sector=lambda x: pd.Categorical(x["sector"], categories=sector_order, ordered=True))
            .pivot(index=["period", "state"], columns="sector", values="value")
            .reindex(columns=sector_order)
            .reset_index()
            .sort_values(["period", "state"])
        )
        return wide



class PetroleumHelper:

    def __init__(self):

        return
