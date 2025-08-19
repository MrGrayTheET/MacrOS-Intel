"""
EIA OpenData API v2 Python Wrapper
A comprehensive Python client for the U.S. Energy Information Administration API
with automatic pagination support for large datasets
"""

import requests
from config import EIA_KEY, DOT_ENV
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union, Any, Generator
from datetime import datetime
from urllib.parse import urlencode
import json
import time
import os


load_dotenv(DOT_ENV)

class EIAAPIError(Exception):
    """Custom exception for EIA API errors"""
    pass


class EIAClient:
    """
    Main client for interacting with the EIA OpenData API v2

    Usage:
        client = EIAClient(api_key="your_api_key")

        # Get petroleum stocks by state with automatic pagination
        data = client.petroleum.sum_mkt.get_all_data(
            frequency="monthly",
            data_columns=["value"],
            facets={
                "process": ["STR"],
                "duoarea": ["NUS", "SAK", "SAL", "SAR", "SAZ", "SCA", "SCO"]
            }
        )
    """

    BASE_URL = "https://api.eia.gov/v2"

    def __init__(self, api_key: str = None, timeout: int = 60):
        """
        Initialize the EIA API client

        Args:
            api_key: Your EIA API key (register at https://www.eia.gov/opendata/)
            timeout: Request timeout in seconds (default: 60)
        """
        self.api_key = api_key or os.getenv('EIA_TOKEN', EIA_KEY)
        self.timeout = timeout
        self.session = requests.Session()

        # Initialize sub-clients for different data categories
        self.petroleum = PetroleumClient(self)
        self.natural_gas = NaturalGasClient(self)
        self.electricity = ElectricityClient(self)
        self.coal = CoalClient(self)

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the EIA API

        Args:
            endpoint: API endpoint path (without base URL)
            params: Query parameters

        Returns:
            API response as dictionary

        Raises:
            EIAAPIError: If the API returns an error
        """
        if params is None:
            params = {}

        # Always include API key
        params['api_key'] = self.api_key

        # Build full URL
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Check for API errors in response
            if 'error' in data:
                raise EIAAPIError(f"API Error: {data.get('error', 'Unknown error')}")

            return data

        except requests.exceptions.RequestException as e:
            raise EIAAPIError(f"Request failed: {str(e)}")

    def build_params_dict(
            self,
            data_columns: Optional[List[str]] = None,
            facets: Optional[Dict[str, Union[str, List[str]]]] = None,
            frequency: Optional[str] = None,
            start: Optional[str] = None,
            end: Optional[str] = None,
            sort: Optional[List[Dict[str, str]]] = None,
            length: Optional[int] = None,
            offset: Optional[int] = None
    ) -> Dict:
        """
        Build parameters dictionary for API request

        Args:
            data_columns: List of data columns to return
            facets: Dictionary of facets to filter by
            frequency: Data frequency (annual, monthly, etc.)
            start: Start date (YYYY-MM format for monthly)
            end: End date (YYYY-MM format for monthly)
            sort: List of sort specifications
            length: Number of rows to return (max 5000)
            offset: Number of rows to skip

        Returns:
            Dictionary of parameters ready for requests
        """
        params = {
            "api_key": self.api_key
        }

        # Add data columns
        if data_columns:
            for i, col in enumerate(data_columns):
                params[f"data[{i}]"] = col

        # Add facets - using the same key with [] for list values
        if facets:
            for facet_name, facet_values in facets.items():
                if isinstance(facet_values, str):
                    facet_values = [facet_values]
                # Use the same key format as the example
                key = f"facets[{facet_name}][]"
                params[key] = facet_values

        # Add other parameters
        if frequency:
            params["frequency"] = frequency
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if length:
            params["length"] = length
        if offset is not None:
            params["offset"] = offset

        # Add sort parameters
        if sort:
            for i, sort_spec in enumerate(sort):
                if 'column' in sort_spec:
                    params[f"sort[{i}][column]"] = sort_spec['column']
                if 'direction' in sort_spec:
                    params[f"sort[{i}][direction]"] = sort_spec['direction']

        return params

    def get_metadata(self, route: str) -> Dict:
        """
        Get metadata for a specific route

        Args:
            route: API route path (e.g., "/petroleum/sum/mkt")

        Returns:
            Metadata dictionary including available facets, data columns, etc.
        """
        return self._make_request(route)

    def get_facet_options(self, route: str, facet_id: str) -> Dict:
        """
        Get available options for a specific facet

        Args:
            route: API route path
            facet_id: Facet identifier (e.g., "duoarea", "process", "product")

        Returns:
            Dictionary with facet options
        """
        return self._make_request(f"{route}/facet/{facet_id}")


class BaseDataClient:
    """Base class for data category clients"""

    def __init__(self, client: EIAClient, base_route: str):
        self.client = client
        self.base_route = base_route

    def get_data(
            self,
            route_suffix: str = "",
            data_columns: Optional[List[str]] = None,
            facets: Optional[Dict[str, Union[str, List[str]]]] = None,
            frequency: Optional[str] = None,
            start: Optional[str] = None,
            end: Optional[str] = None,
            sort: Optional[List[Dict[str, str]]] = None,
            length: Optional[int] = 5000,
            offset: Optional[int] = 0
    ) -> Dict:
        """
        Get a single page of data from the API

        Args:
            route_suffix: Additional route path after base route
            data_columns: List of data columns to return
            facets: Dictionary of facets to filter by
            frequency: Data frequency
            start: Start date (YYYY-MM format for monthly)
            end: End date (YYYY-MM format for monthly)
            sort: Sort specifications
            length: Number of rows to return (max 5000)
            offset: Number of rows to skip

        Returns:
            API response data for a single page
        """
        route = f"{self.base_route}{route_suffix}/data"
        params = self.client.build_params_dict(
            data_columns=data_columns,
            facets=facets,
            frequency=frequency,
            start=start,
            end=end,
            sort=sort,
            length=length,
            offset=offset
        )

        # Make request
        url = f"{self.client.BASE_URL}{route}"

        try:
            response = self.client.session.get(url, params=params, timeout=self.client.timeout)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                raise EIAAPIError(f"API Error: {data.get('error', 'Unknown error')}")

            return data

        except requests.exceptions.RequestException as e:
            raise EIAAPIError(f"Request failed: {str(e)}")

    def get_all_data(
            self,
            route_suffix: str = "",
            data_columns: Optional[List[str]] = None,
            facets: Optional[Dict[str, Union[str, List[str]]]] = None,
            frequency: Optional[str] = None,
            start: Optional[str] = None,
            end: Optional[str] = None,
            sort: Optional[List[Dict[str, str]]] = None,
            page_size: int = 5000,
            max_pages: Optional[int] = None,
            delay_between_requests: float = 0.1
    ) -> List[Dict]:
        """
        Get all data from the API with automatic pagination

        Args:
            route_suffix: Additional route path after base route
            data_columns: List of data columns to return
            facets: Dictionary of facets to filter by
            frequency: Data frequency
            start: Start date (YYYY-MM format for monthly)
            end: End date (YYYY-MM format for monthly)
            sort: Sort specifications
            page_size: Number of rows per page (max 5000)
            max_pages: Maximum number of pages to fetch (None for all)
            delay_between_requests: Delay in seconds between requests

        Returns:
            List of all data records combined from all pages
        """
        all_rows = []
        offset = 0
        page_count = 0

        while True:
            # Check if we've reached max pages
            if max_pages and page_count >= max_pages:
                break

            # Get a page of data
            response = self.get_data(
                route_suffix=route_suffix,
                data_columns=data_columns,
                facets=facets,
                frequency=frequency,
                start=start,
                end=end,
                sort=sort,
                length=page_size,
                offset=offset
            )

            # Extract data rows
            rows = response.get("response", {}).get("data", [])

            # If no rows, we're done
            if not rows:
                break

            # Add rows to our collection
            all_rows.extend(rows)

            # If we got fewer rows than requested, we've reached the end
            if len(rows) < page_size:
                break

            # Update offset for next page
            offset += page_size
            page_count += 1

            # Small delay to be nice to the API
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

        return all_rows

    def iter_data(
            self,
            route_suffix: str = "",
            data_columns: Optional[List[str]] = None,
            facets: Optional[Dict[str, Union[str, List[str]]]] = None,
            frequency: Optional[str] = None,
            start: Optional[str] = None,
            end: Optional[str] = None,
            sort: Optional[List[Dict[str, str]]] = None,
            page_size: int = 5000,
            delay_between_requests: float = 0.1
    ) -> Generator[Dict, None, None]:
        """
        Iterate over all data from the API with automatic pagination

        This is memory-efficient for very large datasets as it yields
        one record at a time instead of loading all into memory.

        Args:
            route_suffix: Additional route path after base route
            data_columns: List of data columns to return
            facets: Dictionary of facets to filter by
            frequency: Data frequency
            start: Start date
            end: End date
            sort: Sort specifications
            page_size: Number of rows per page (max 5000)
            delay_between_requests: Delay in seconds between requests

        Yields:
            Individual data records
        """
        offset = 0

        while True:
            # Get a page of data
            response = self.get_data(
                route_suffix=route_suffix,
                data_columns=data_columns,
                facets=facets,
                frequency=frequency,
                start=start,
                end=end,
                sort=sort,
                length=page_size,
                offset=offset
            )

            # Extract data rows
            rows = response.get("response", {}).get("data", [])

            # If no rows, we're done
            if not rows:
                break

            # Yield each row
            for row in rows:
                yield row

            # If we got fewer rows than requested, we've reached the end
            if len(rows) < page_size:
                break

            # Update offset for next page
            offset += page_size

            # Small delay to be nice to the API
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

    def get_metadata(self, route_suffix: str = "") -> Dict:
        """Get metadata for this data category"""
        route = f"{self.base_route}{route_suffix}"
        return self.client.get_metadata(route)

    def get_facet_options(self, facet_id: str, route_suffix: str = "") -> Dict:
        """Get available options for a specific facet"""
        route = f"{self.base_route}{route_suffix}"
        return self.client.get_facet_options(route, facet_id)


class PetroleumClient(BaseDataClient):
    """Client for petroleum data endpoints"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/petroleum")

        # Initialize sub-routes
        self.sum_mkt = PetroleumSumMktClient(client)
        self.pri = PetroleumPriClient(client)
        self.cons = PetroleumConsClient(client)
        self.move = PetroleumMoveClient(client)
        self.stoc = PetroleumStocClient(client)

    def get_stocks_by_state(
            self,
            states: List[str],
            frequency: str = "monthly",
            start: Optional[str] = None,
            end: Optional[str] = None,
            get_all: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Convenience method to get petroleum stocks by state

        Args:
            states: List of state codes (e.g., ["SAK", "SAL", "SAR"])
            frequency: Data frequency (default: monthly)
            start: Start date (YYYY-MM format)
            end: End date (YYYY-MM format)
            get_all: If True, get all pages; if False, get first page only

        Returns:
            Petroleum stocks data (list if get_all=True, dict if False)
        """
        if get_all:
            return self.sum_mkt.get_all_data(
                data_columns=["value"],
                facets={
                    "process": ["STR"],
                    "duoarea": states
                },
                frequency=frequency,
                start=start,
                end=end
            )
        else:
            return self.sum_mkt.get_data(
                data_columns=["value"],
                facets={
                    "process": ["STR"],
                    "duoarea": states
                },
                frequency=frequency,
                start=start,
                end=end
            )


class PetroleumSumMktClient(BaseDataClient):
    """Client for petroleum summary market data"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/petroleum/sum/mkt")


class PetroleumPriClient(BaseDataClient):
    """Client for petroleum price data"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/petroleum/pri")


class PetroleumConsClient(BaseDataClient):
    """Client for petroleum consumption data"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/petroleum/cons")


class PetroleumMoveClient(BaseDataClient):
    """Client for petroleum movement data"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/petroleum/move")


class PetroleumStocClient(BaseDataClient):
    """Client for petroleum stock data"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/petroleum/stoc")


class NaturalGasClient(BaseDataClient):
    """Client for natural gas data endpoints"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/natural-gas")

    def get_data_by_route(
            self,
            route1: str,
            route2: str,
            data_columns: Optional[List[str]] = None,
            facets: Optional[Dict[str, Union[str, List[str]]]] = None,
            get_all: bool = False,
            **kwargs
     ) -> Union[Dict, List[Dict]]:
        """
        Get natural gas data by specific route

        Args:
            route1: First route segment (e.g., "sum")
            route2: Second route segment (e.g., "lsum")
            data_columns: Data columns to return
            facets: Facets to filter by
            get_all: If True, get all pages; if False, get first page only
            **kwargs: Additional parameters

        Returns:
            Natural gas data
        """
        route_suffix = f"/{route1}/{route2}"

        if get_all:
            return self.get_all_data(
                route_suffix=route_suffix,
                data_columns=data_columns,
                facets=facets,
                **kwargs
            )
        else:
            return self.get_data(
                route_suffix=route_suffix,
                data_columns=data_columns,
                facets=facets,
                **kwargs
            )

class NaturalGasProdClient(BaseDataClient):

    def __init__(self,client:EIAClient):

        super().__init__(client=client, base_route='/natural-gas/prod')

        return



class ElectricityClient(BaseDataClient):
    """Client for electricity data endpoints"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/electricity")

        # Initialize sub-routes
        self.retail_sales = ElectricityRetailSalesClient(client)
        self.electric_power_operational = ElectricityOperationalClient(client)


class ElectricityRetailSalesClient(BaseDataClient):
    """Client for electricity retail sales data"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/electricity/retail-sales")


class ElectricityOperationalClient(BaseDataClient):
    """Client for electricity operational data"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/electricity/electric-power-operational-data")


class CoalClient(BaseDataClient):
    """Client for coal data endpoints"""

    def __init__(self, client: EIAClient):
        super().__init__(client, "/coal")