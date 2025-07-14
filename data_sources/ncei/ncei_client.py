import requests
import pandas as pd
import toml
from typing import Optional, List
class NOAACDOClient:
    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({'token': self.token})

    def get_data(
            self,
            datasetid: str = "NCLIMGRID",
            datatypeid: Optional[List[str]] = None,
            locationid: Optional[str] = None,
            startdate: str = None,
            enddate: str = None,
            limit: int = 1000,
            offset: int = 1,
            units: str = "metric",
            include_attributes: bool = False
    ) -> pd.DataFrame:
        """
        Fetch daily climate summaries for specified region and time range.

        Parameters:
        - datasetid: "NCLIMGRID" (default)
        - datatypeid: List of types, e.g. ["TMAX", "TMIN", "PRCP"]
        - locationid: e.g. "CLIM_REG:101"
        - startdate, enddate: e.g. "2020-01-01"
        - limit: Number of results to fetch (max 1000)
        - offset: Pagination offset
        - units: "standard" or "metric"
        - include_attributes: Whether to include metadata

        Returns:
        - DataFrame of results
        """
        url = f"{self.BASE_URL}/data"
        params = {
            "datasetid": datasetid,
            "startdate": startdate,
            "enddate": enddate,
            "limit": limit,
            "offset": offset,
            "units": units,
            "includeAttributes": str(include_attributes).lower()
        }

        if locationid:
            params["locationid"] = locationid
        if datatypeid:
            params["datatypeid"] = datatypeid if isinstance(datatypeid, str) else ','.join(datatypeid)

        response = self.session.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        data = response.json().get("results", [])
        return pd.DataFrame(data)