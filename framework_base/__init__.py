"""
Commodities Dashboard Framework (comm_dash)
==========================================

A comprehensive Python framework for building commodity data analysis dashboards 
with Plotly Dash. Provides reusable components, data access layers, and 
visualization tools for agricultural, energy, and financial market data.

Quick Start
-----------
```python
from comm_dash.data.data_tables import NASSTable
from comm_dash.components.frames import FundamentalFrame

# Initialize data client
nass = NASSTable()

# Create chart configuration
chart_config = {
    'title': 'Corn Production',
    'chart_type': 'line',
    'starting_key': 'corn/production',
    'y_column': 'value',
    'x_column': 'date'
}

# Create dashboard frame
frame = FundamentalFrame(
    table_client=nass,
    chart_configs=[chart_config]
)
```

Main Components
---------------
- **data**: Data access layer with unified API clients
- **components**: Reusable UI components and layout managers  
- **models**: Analytics and modeling utilities
- **utils**: Utility functions and diagnostics

Data Sources
------------
- USDA NASS (National Agricultural Statistics Service)
- USDA FAS (Foreign Agricultural Service)
- EIA (Energy Information Administration)
- COT (Commitment of Traders)
- NCEI (National Centers for Environmental Information)

Key Features
------------
- Async multi-year data updates with concurrency control
- Automatic FIPS code preservation for geographic data
- Configuration-driven development with TOML files
- Store-based callback patterns for performance
- Modular and extensible architecture
"""

__version__ = "1.0.0"
__author__ = "Commodities Analytics Team"
__email__ = "analytics@commodities-dashboard.com"
__license__ = "MIT"

# Import main classes for convenience
try:
    from .data.data_tables import (
        TableClient,
        NASSTable, 
        EIATable,
        FASTable,
        ESRTableClient,
        WeatherTable,
        MarketTable
    )
    
    from .components.frames import (
        FundamentalFrame,
        FlexibleMenu,
        EnhancedFrameGrid
    )
    
    from .models.commodity_analytics import (
        CommodityAnalyzer,
        ESRAnalyzer,
        CommodityPriceAnalyzer
    )
    
    from .models.timeseries_analysis import TimeSeriesAnalyzer
    
    __all__ = [
        # Data access
        'TableClient', 'NASSTable', 'EIATable', 'FASTable', 
        'ESRTableClient', 'WeatherTable', 'MarketTable',
        
        # UI Components
        'FundamentalFrame', 'FlexibleMenu', 'EnhancedFrameGrid',
        
        # Analytics
        'CommodityAnalyzer', 'ESRAnalyzer', 'CommodityPriceAnalyzer',
        'TimeSeriesAnalyzer',
        
        # Package info
        '__version__', '__author__', '__email__', '__license__'
    ]
    
except ImportError as e:
    # Handle optional dependencies gracefully
    import warnings
    warnings.warn(f"Some components could not be imported: {e}")
    
    __all__ = ['__version__', '__author__', '__email__', '__license__']

# Package metadata
def get_version():
    """Return the package version."""
    return __version__

def get_info():
    """Return package information."""
    return {
        'name': 'comm_dash',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'description': 'Comprehensive Python framework for commodity data analysis dashboards'
    }

# Configuration defaults
DEFAULT_CONFIG = {
    'data_path': './data',
    'cache_enabled': True,
    'async_timeout': 30,
    'max_concurrent_requests': 5,
    'chart_template': 'plotly_white',
    'date_format': '%Y-%m-%d'
}

def configure(**kwargs):
    """
    Configure package defaults.
    
    Args:
        **kwargs: Configuration options to override defaults
        
    Example:
        configure(
            data_path='/custom/data/path',
            max_concurrent_requests=10
        )
    """
    DEFAULT_CONFIG.update(kwargs)
    return DEFAULT_CONFIG