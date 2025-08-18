# Commodities Dashboard Framework Base

This folder contains the core, reusable framework modules for building commodity analytics dashboards. These modules are site-agnostic and can be used across different dashboard implementations.

## Framework Components

### Core Modules
- **components/**: Reusable UI components
  - `frames.py`: FundamentalFrame, FlexibleMenu, EnhancedFrameGrid classes
  - `chart_components.py`: Chart generation utilities
  - `plotting/`: Chart configuration and plotting utilities
  
- **data/**: Data access and processing layer
  - `data_tables.py`: TableClient, FASTable, ESRTableClient classes
  - `sources/`: API wrappers for external data sources
    - `usda/`: USDA NASS and FAS API integrations
    - `eia/`: Energy Information Administration APIs
    - `ncei/`: Weather data integration
    - `COT/`: Commodity futures data

- **models/**: Analytics and modeling utilities
  - `commodity_analytics.py`: ESRAnalyzer and CommodityPriceAnalyzer
  - `timeseries_analysis.py`: Time series analysis utilities
  - `seasonal.py`: Seasonal pattern analysis

- **utils/**: Utility functions
  - `data_tools.py`: Data processing utilities
  - `diagnostics.py`: System diagnostics
  - `rename_tables.py`: Table management utilities

### Configuration
- `config.py`: Environment and API configuration management
- `requirements.txt`: Python package dependencies

## Architecture Patterns

### Creating Charts
Use FundamentalFrame with chart configurations:
```python
chart_configs = [{
    'title': 'Chart Title',
    'chart_type': 'line',  # line, bar, area
    'starting_key': 'data/key/path',
    'y_column': 'metric_name',
    'x_column': 'date_column',
    'width': '100%',
    'height': 400
}]

frame = FundamentalFrame(
    table_client=client,
    chart_configs=chart_configs,
    div_prefix="unique_prefix"
)
```

### Data Access
```python
from data.data_tables import TableClient, FASTable

# For general data
client = TableClient()
data = client.get_key('data_key')

# For FAS/ESR data
fas_client = FASTable()
esr_data = fas_client.get_esr_data('commodity', year=2024)
```

### Analytics
```python
from models.commodity_analytics import ESRAnalyzer

analyzer = ESRAnalyzer(data, commodity_type='grains')
seasonal_analysis = analyzer.analyze_seasonal_patterns('weeklyExports')
commitment_analysis = analyzer.commitment_vs_shipment_analysis(country='Korea, South')
```

## Usage

This framework is designed to be imported and used in site-specific dashboard applications. The modules provide:

1. **Reusable UI Components**: Chart containers, menus, layout grids
2. **Data Layer Abstraction**: Unified API access across multiple data sources  
3. **Analytics Engine**: Commodity-specific time series and seasonal analysis
4. **Configuration Management**: Environment-based API key and settings management

## Dependencies

See `requirements.txt` for complete package requirements. Key dependencies:
- Dash for web UI framework
- Plotly for charting
- Pandas for data manipulation
- Requests for API calls