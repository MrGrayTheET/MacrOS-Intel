"""
Data Access Layer
=================

Unified interfaces for commodity data sources including USDA, EIA, and market data.
Provides async-capable table clients with built-in caching and FIPS preservation.

Classes:
    TableClient: Base class for all data sources
    NASSTable: USDA National Agricultural Statistics Service
    FASTable: USDA Foreign Agricultural Service  
    EIATable: Energy Information Administration
    ESRTableClient: Enhanced Export Sales Reporting
    WeatherTable: NCEI weather data integration
    MarketTable: Market and COT data access
"""

from .data_tables import (
    TableClient,
    NASSTable,
    FASTable, 
    EIATable,
    ESRTableClient,
    WeatherTable,
    MarketTable
)

try:
    from .google_drive_client import GoogleDriveTableClient, create_gdrive_config_template
    HAS_GDRIVE = True
except ImportError:
    HAS_GDRIVE = False

__all__ = [
    'TableClient', 'NASSTable', 'FASTable', 'EIATable', 
    'ESRTableClient', 'WeatherTable', 'MarketTable'
]

if HAS_GDRIVE:
    __all__.extend(['GoogleDriveTableClient', 'create_gdrive_config_template'])