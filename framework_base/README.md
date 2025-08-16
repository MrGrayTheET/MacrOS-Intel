# Commodities Dashboard Framework (comm_dash)

A comprehensive Python framework for building commodity data analysis dashboards with Plotly Dash. Provides reusable components, data access layers, and visualization tools for agricultural, energy, and financial market data.

## Overview

The Commodities Dashboard Framework (`comm_dash`) is a modular framework designed to accelerate the development of commodity data analysis applications. It provides:

- **Data Access Layer**: Unified interfaces for USDA, EIA, COT, and other commodity data sources
- **Visualization Components**: Reusable chart components and layout managers
- **Async Data Processing**: High-performance multi-year data updates with concurrency control
- **Geographic Data**: Automatic preservation of FIPS codes and geographic identifiers
- **Configuration Management**: TOML-based configuration for data mappings and API endpoints

## Quick Start

### Installation

```bash
pip install comm_dash
```

### Basic Usage

```python
from comm_dash.data.data_tables import NASSTable, EIATable
from comm_dash.components.frames import FundamentalFrame, FlexibleMenu

# Initialize data clients
nass_client = NASSTable()
eia_client = EIATable()

# Create chart configurations
chart_configs = [{
    'title': 'Corn Production by State',
    'chart_type': 'line',
    'starting_key': 'corn/production/acres_planted',
    'y_column': 'value',
    'x_column': 'date',
    'width': '100%',
    'height': 400
}]

# Create dashboard frame
frame = FundamentalFrame(
    table_client=nass_client,
    chart_configs=chart_configs,
    layout="horizontal"
)
```

## Core Components

### Data Access Layer (`data/`)

#### NASSTable - USDA Agricultural Statistics
```python
from comm_dash.data.data_tables import NASSTable

# Initialize client
nass = NASSTable()

# Single year update
nass.api_update('CORN - ACRES PLANTED', commodity_desc='CORN', year='2024')

# Multi-year async update with FIPS preservation
results = nass.api_update_multi_year_sync(
    short_desc='CORN - ACRES PLANTED',
    start_year=2015,
    end_year=2024,
    agg_level='COUNTY',  # Preserves county_code, county_ansi, state_fips_code
    max_concurrent=3
)
```

#### EIATable - Energy Information Administration
```python
from comm_dash.data.data_tables import EIATable

eia = EIATable()
# Access petroleum, natural gas, electricity data
```

#### FASTable - Foreign Agricultural Service
```python
from comm_dash.data.data_tables import FASTable

fas = FASTable()
# Access international trade and market data
```

### Visualization Components (`components/`)

#### FundamentalFrame - Advanced Chart Container
```python
from comm_dash.components.frames import FundamentalFrame

frame = FundamentalFrame(
    table_client=data_client,
    chart_configs=[{
        'title': 'Price Analysis',
        'chart_type': 'line',  # line, bar, area, scatter
        'starting_key': 'commodity/price',
        'y_column': 'price',
        'x_column': 'date',
        'color_column': 'region',  # Optional grouping
        'width': '100%',
        'height': 500
    }],
    layout="horizontal"  # or "vertical"
)
```

#### FlexibleMenu - Dynamic Control Panel
```python
from comm_dash.components.frames import FlexibleMenu

menu = FlexibleMenu(
    menu_id='controls',
    title='Data Filters',
    component_configs=[
        {
            'type': 'dropdown',
            'id': 'commodity',
            'label': 'Commodity',
            'options': [
                {'label': 'Corn', 'value': 'CORN'},
                {'label': 'Soybeans', 'value': 'SOYBEANS'}
            ],
            'value': 'CORN'
        },
        {
            'type': 'range_slider',
            'id': 'year_range',
            'label': 'Year Range',
            'min': 2010,
            'max': 2024,
            'value': [2020, 2024]
        }
    ]
)
```

#### EnhancedFrameGrid - Advanced Layout Manager
```python
from comm_dash.components.frames import EnhancedFrameGrid

grid = EnhancedFrameGrid(
    frames=[frame1, frame2, frame3],
    flexible_menu=menu,
    data_source='shared_data_store'  # Optional store-based caching
)

# Register individual chart callbacks
grid.register_chart_store_callback(
    app=app,
    chart_id='frame1_chart_0',
    update_function=my_update_function,
    menu_inputs=['commodity', 'year_range']
)
```

### Models and Analytics (`models/`)

#### Commodity Analytics
```python
from comm_dash.models.commodity_analytics import CommodityAnalyzer

analyzer = CommodityAnalyzer()
seasonal_patterns = analyzer.calculate_seasonal_index(data)
price_trends = analyzer.trend_analysis(price_data)
```

#### Time Series Analysis
```python
from comm_dash.models.timeseries_analysis import TimeSeriesAnalyzer

ts_analyzer = TimeSeriesAnalyzer()
forecasts = ts_analyzer.forecast(data, periods=12)
volatility = ts_analyzer.calculate_volatility(price_data)
```

## Advanced Features

### Async Multi-Year Data Updates

The framework supports high-performance concurrent data retrieval:

```python
import asyncio
from comm_dash.data.data_tables import NASSTable

async def bulk_update():
    nass = NASSTable()
    
    # Update multiple commodities concurrently
    commodities = ['CORN', 'SOYBEANS', 'WHEAT']
    tasks = []
    
    for commodity in commodities:
        task = nass.api_update_multi_year_async(
            short_desc=f'{commodity} - ACRES PLANTED',
            start_year=2010,
            end_year=2024,
            agg_level='STATE',
            max_concurrent=3
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# Run async update
results = asyncio.run(bulk_update())
```

### Geographic Data Preservation

Automatically preserves essential FIPS codes and geographic identifiers:

**State-Level Data:**
- `state_fips_code`: 2-digit FIPS state code
- `state_ansi`: ANSI state code
- `state_alpha`: 2-letter state abbreviation
- `state_name`: Full state name

**County-Level Data:**
- `county_code`: 3-digit county FIPS code
- `county_ansi`: ANSI county code
- `county_name`: County name
- `location_desc`: Full location description

### Configuration Management

#### Data Source Mappings (TOML)
```toml
# data/sources/usda/data_mapping.toml
[nass.endpoints]
base_url = "https://quickstats.nass.usda.gov/api"
api_get = "/api_GET/"
get_counts = "/get_counts/"

[nass.default_params]
source_desc = "SURVEY"
domain_desc = "TOTAL"
```

#### Chart Configuration Templates
```toml
# components/plotting/chart_mappings.toml
[chart_templates.price_analysis]
chart_type = "line"
height = 400
showlegend = true
template = "plotly_white"

[chart_templates.production_bar]
chart_type = "bar"
height = 350
orientation = "v"
```

## Data Sources Supported

### USDA (United States Department of Agriculture)
- **NASS**: National Agricultural Statistics Service
- **FAS**: Foreign Agricultural Service  
- **ERS**: Economic Research Service

### EIA (Energy Information Administration)
- Petroleum data
- Natural gas statistics
- Electricity generation and consumption
- Renewable energy data

### Market Data
- **COT**: Commitment of Traders reports
- **CFTC**: Commodity Futures Trading Commission data
- Price and volume data from various exchanges

### Weather Data
- **NCEI**: National Centers for Environmental Information
- Climate data integration for agricultural analysis

## Architecture

### Modular Design
```
comm_dash/
├── components/          # Reusable UI components
│   ├── frames.py       # Layout and chart containers
│   ├── callback_utils.py # Callback management utilities
│   └── plotting/       # Chart configuration and templates
├── data/               # Data access layer
│   ├── data_tables.py  # Core table client classes
│   ├── sources/        # API wrappers by data provider
│   └── google_drive_client.py # Cloud storage integration
├── models/             # Analytics and modeling
│   ├── commodity_analytics.py
│   ├── seasonal.py
│   └── timeseries_analysis.py
└── utils/              # Utility functions
    ├── data_tools.py
    └── diagnostics.py
```

### Design Patterns

#### Table Client Pattern
All data sources implement a consistent interface:
- `get_key(key)`: Retrieve cached data
- `api_update(params)`: Update data from API
- `update_table(key)`: Refresh specific dataset

#### Configuration-Driven Development
- TOML files for data mappings and API configurations
- Template-based chart generation
- Environment variable management for API keys

#### Async-First Architecture
- Non-blocking data retrieval
- Concurrency control with semaphores
- Progress tracking for long-running operations

## Performance Optimization

### Data Caching
- HDF5 storage for large datasets
- Store-based callback patterns for shared data
- Intelligent cache invalidation

### Concurrent Processing
- Async/await throughout data layer
- Configurable concurrency limits
- Background task execution

### Memory Management
- Lazy loading of large datasets
- Columnar storage optimization
- Automatic data compression

## Development Patterns

### Creating New Data Sources
```python
from comm_dash.data.data_tables import TableClient

class MyDataSource(TableClient):
    def __init__(self):
        super().__init__(
            table_db="my_data.h5",
            mapping_file="my_mapping.toml"
        )
    
    def api_update(self, **params):
        # Implement data retrieval logic
        data = self.fetch_from_api(params)
        return self.process_and_store(data)
```

### Adding Chart Types
```python
from comm_dash.components.frames import FundamentalFrame

# Extend chart configurations
custom_config = {
    'title': 'Custom Visualization',
    'chart_type': 'heatmap',  # New chart type
    'custom_params': {
        'colorscale': 'Viridis',
        'showscale': True
    }
}
```

### Custom Analytics
```python
from comm_dash.models.commodity_analytics import CommodityAnalyzer

class MyAnalyzer(CommodityAnalyzer):
    def custom_indicator(self, data):
        # Implement custom analysis
        return processed_data
```

## Configuration

### Environment Variables
```bash
# API Keys
export NASS_TOKEN="your_nass_api_key"
export EIA_API_KEY="your_eia_api_key"
export FAS_TOKEN="your_fas_api_key"

# Data Paths
export DATA_PATH="/path/to/data"
export MARKET_DATA_PATH="/path/to/market/data"
export COT_PATH="/path/to/cot/data"
```

### Configuration Files
```python
# config.py
DATA_PATH = "/data"
MARKET_DATA_PATH = "/data/market"
ENABLE_CACHE = True
DEFAULT_TIMEOUT = 30
MAX_CONCURRENT_REQUESTS = 5
```

## Dependencies

See `requirements.txt` for complete package requirements. Key dependencies:
- Dash for web UI framework
- Plotly for charting
- Pandas for data manipulation
- Requests for API calls
- aiohttp for async HTTP requests
- TOML for configuration management