# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Commodities Dashboard** - a multi-page Dash application for analyzing agricultural and energy commodity data including:
- Export Sales Reporting (ESR) data
- USDA agricultural statistics  
- EIA energy data
- Commitment of Traders (COT) data
- Trade balance and import data

## Application Architecture

### Main Application Structure
- **app.py**: Main Dash application entry point with multi-page routing
- **pages/**: Page modules using Dash's built-in page system
  - `esr/`: Export Sales Reporting analysis modules
  - `agricultural/psd_data.py`: Production, Supply & Distribution data
  - `trade_bal.py`: Trade balance analysis
  - `energy/`: Energy commodity pages (natural gas, petroleum)

### Core Framework Components
- **components/frames.py**: Core framework classes
  - `FundamentalFrame`: Reusable chart container with data binding
  - `FlexibleMenu`: Dynamic control panel system with component configuration support
  - `EnhancedFrameGrid`: Advanced layout management with individual chart callback registration
- **callbacks/**: Dynamic callback system with registry pattern
  - `callback_registry.py`: Centralized callback management
  - Module-specific callbacks (agricultural.py, esr.py, global_callbacks.py)

### Framework Base
The `framework_base/` directory contains clean, reusable framework modules:
- Core UI components without site-specific customizations
- Data access layer and API wrappers
- Analytics and modeling utilities
- Configuration and utility functions

### Data Layer
- **data/sources/**: API wrappers and data mappings
  - `usda/`: NASS and FAS API integrations with ESR staging API
    - `api_wrappers/usda_quickstats.py`: QuickStatsClient for USDA/NASS Quick Stats API
  - `eia/`: Energy Information Administration APIs
  - `ncei/`: Weather data integration
  - `COT/`: Commodity futures data
- **data/data_tables.py**: Core data access layer with `TableClient`, `FASTable`, and `ESRTableClient` classes
- **config.py**: Environment configuration and API keys management

## Key Development Commands

### Running the Application
```bash
python app.py              # Main multi-page dashboard
```

### Testing Framework Components
```bash
python test_individual_chart_callbacks.py    # Test individual chart registration
python test_multi_chart_callbacks.py         # Test multi-chart registration
```

### Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create `.env` file with:
```
NASS_TOKEN=your_nass_api_key
FAS_TOKEN=your_fas_api_key  
EIA_API_KEY=your_eia_api_key
data_path=F:\Macro\OSINT
```

## Configuration System

### TOML-based Mappings
- **data/sources/*/data_mapping.toml**: API endpoint and data field mappings
- **components/plotting/chart_mappings.toml**: Chart configuration templates
- **data/sources/COT/futures_mappings.toml**: COT commodity symbol mappings
- **data/sources/usda/esr_map.toml**: ESR data mapping configurations

### Data Paths (configurable via environment)
- `DATA_DIR`: Base data directory (default: F:\Macro\OSINT)
- `MARKET_DATA_PATH`: Market data storage
- `COT_PATH`: Commitment of Traders data

## Framework Patterns

### Creating Charts with FundamentalFrame
Use `FundamentalFrame` with chart configurations:
```python
from components.frames import FundamentalFrame

chart_configs = [{
    'title': 'Chart Title',
    'chart_type': 'line',  # line, bar, area, scatter
    'starting_key': 'data/key/path',
    'y_column': 'metric_name',
    'x_column': 'date_column',
    'width': '100%',
    'height': 400
}]

frame = FundamentalFrame(
    table_client=client,
    chart_configs=chart_configs,
    layout="horizontal",  # 'horizontal' stacks charts, 'vertical' places side by side
    div_prefix="unique_prefix"
)
```

### Creating Dynamic Menus with FlexibleMenu
```python
from components.frames import FlexibleMenu

menu_configs = [
    {
        'type': 'dropdown',
        'id': 'commodity',
        'label': 'Commodity',
        'options': [
            {'label': 'Cattle', 'value': 'cattle'},
            {'label': 'Corn', 'value': 'corn'}
        ],
        'value': 'cattle'
    },
    {
        'type': 'checklist',
        'id': 'countries',
        'label': 'Countries',
        'options': [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'}
        ],
        'value': ['Korea, South']
    },
    {
        'type': 'button',
        'id': 'apply_btn',
        'label': 'Apply Changes',
        'color': '#4CAF50'
    }
]

menu = FlexibleMenu(
    menu_id='control_menu',
    title='Data Controls',
    component_configs=menu_configs
)
```

### Advanced Grid System with EnhancedFrameGrid
```python
from components.frames import EnhancedFrameGrid

# Create grid with optional store data source for performance
grid = EnhancedFrameGrid(
    frames=[frame1, frame2],
    flexible_menu=menu,
    data_source='data_store_id'  # Optional: for store-based callbacks
)
```

## Individual Chart Callback Registration

The `EnhancedFrameGrid` supports advanced callback registration patterns:

### Single Chart Registration
```python
# Store-based callback (uses dcc.Store for data caching)
grid.register_chart_store_callback(
    app=app,
    chart_id='frame_prefix_chart_0',
    update_function=my_update_function,
    menu_inputs=['commodity', 'countries']
)

# Function-based callback (traditional Dash callback)
grid.register_chart_function_callback(
    app=app,
    chart_id='frame_prefix_chart_1',
    update_function=my_update_function,
    input_components=['commodity', 'year', 'apply_btn'],
    trigger_component='apply_btn'
)
```

### Multi-Chart Registration
```python
# Register multiple charts with single update function
chart_ids = ['frame_prefix_chart_0', 'frame_prefix_chart_1']

grid.register_chart_store_callback(
    app=app,
    chart_id=chart_ids,  # List of chart IDs
    update_function=multi_chart_update_function,
    menu_inputs=['commodity', 'countries']
)

# Update function can return single figure (duplicated) or list of figures
def multi_chart_update_function(chart_ids, store_data=None, **menu_values):
    # Return list of figures for each chart
    return [create_figure_1(), create_figure_2()]
    # Or return single figure (automatically duplicated)
    # return create_single_figure()
```

### Chart Registry Management
```python
# Get chart information
chart_ids = grid.get_chart_ids()
store_chart_ids = grid.get_store_chart_ids()
function_chart_ids = grid.get_function_chart_ids()
unregistered_ids = grid.get_unregistered_chart_ids()

# Check multi-chart status
multi_groups = grid.get_multi_chart_groups()
is_multi = grid.is_multi_chart_callback('chart_id')

# Print comprehensive status
grid.print_chart_registry_summary()
```

## Data Access Patterns

### TableClient System
```python
from data.data_tables import TableClient, FASTable, ESRTableClient, NASSTable

# General data access
client = TableClient()
data = client.get_key('data_key')

# NASS agricultural statistics data access
nass_client = NASSTable('CORN', prefix='corn')
corn_data = nass_client.api_update('CORN - ACRES PLANTED')

# FAS/ESR data access
fas_client = FASTable()
esr_data = fas_client.get_esr_data('commodity', year=2024)

# Enhanced ESR client with additional functionality
esr_client = ESRTableClient()
processed_data = esr_client.process_esr_data('cattle')
```

### Store-Based Data Caching
```python
# Use dcc.Store for performance optimization
import dash
from dash import dcc

app.layout = html.Div([
    dcc.Store(id='data_store'),
    # ... other components
])

# Store callback updates cache, chart callbacks read from cache
@app.callback(Output('data_store', 'data'), [Input('update_trigger', 'value')])
def update_store(trigger):
    return expensive_data_processing()
```

## Data Structure Conventions

### ESR Data Format
Expected columns: `weekEndingDate`, `country`, `commodity`, `weeklyExports`, `outstandingSales`, `grossNewSales`, `currentMYNetSales`, `currentMYTotalCommitment`

### Date Handling
- Use pandas datetime conversion: `pd.to_datetime()`
- Standard date column: `weekEndingDate` for ESR, `date` for general data
- Marketing years follow USDA conventions (Oct-Sep for most commodities)

### Chart Update Function Patterns
```python
# Store-based update function signature
def store_update_function(chart_id, store_data=None, **menu_values):
    # Process store_data and menu_values
    # Return plotly figure or list of figures for multi-chart

# Function-based update function signature  
def function_update_function(chart_id, **input_values):
    # Process input_values from menu components
    # Return plotly figure or list of figures for multi-chart
```

## Testing
- Use only UTF-8 and common characters when creating tests. Emojis and unicode result in charmap errors
### Framework Tests
- `test_individual_chart_callbacks.py`: Tests individual chart callback registration
- `test_multi_chart_callbacks.py`: Tests multi-chart callback registration with various scenarios

### Running Tests
```bash
python test_individual_chart_callbacks.py
python test_multi_chart_callbacks.py
```

## Development Patterns

### Multi-Application Architecture
- Single standalone applications: Individual page modules can run independently
- Multi-page integrated dashboard: `app.py` coordinates multiple pages
- Framework base: Reusable components in `framework_base/` for clean separation

### Adding New Pages
1. Create page module in `pages/`
2. Import and register in `app.py`
3. Add navigation link in navbar
4. Use `EnhancedFrameGrid` and `FlexibleMenu` for consistency

### API Integration
1. Create API wrapper in appropriate `data/sources/` subdirectory
2. Add data mapping TOML file for endpoint configuration
3. Integrate with `TableClient` system for unified access
4. Use ESR staging API pattern for complex data processing

### USDA QuickStats API Integration
The application uses a custom QuickStatsClient wrapper (`data/sources/usda/api_wrappers/usda_quickstats.py`) that replaces the deprecated nasspython library:

```python
from data.sources.usda.api_wrappers.usda_quickstats import QuickStatsClient

# Direct API usage
client = QuickStatsClient()
data = client.query_df_numeric(commodity_desc="CORN", year="2024", short_desc="CORN - ACRES PLANTED")

# Get available parameter values
commodities = client.get_param_values("commodity_desc")
years = client.get_param_values("year", commodity_desc="CORN")

# Large queries with automatic chunking
all_data = client.fetch_all(commodity_desc="CORN", agg_level_desc="COUNTY")
```

**Key Features:**
- Automatic numeric conversion with `query_df_numeric()` method
- Parameter validation and error handling  
- Automatic chunking for large queries via `fetch_all()`
- Environment variable support for API keys (NASS_TOKEN)
- Built-in retry logic and request throttling
- Returns pandas DataFrames directly
- Essential location columns preserved: state_fips_code, state_ansi, county_code, county_ansi, county_name
- Async methods for concurrent multi-year requests

**NASSTable Integration:**
The NASSTable class automatically uses QuickStatsClient through a lambda wrapper:
```python
# NASSTable client is configured as:
nass_info['client'] = lambda x: _quickstats_client.query_df_numeric(**x)
```

**Async Multi-Year Updates:**
For improved performance when fetching multiple years of data:
```python
# Async multi-year update (faster for large requests)
nass_client = NASSTable(prefix='corn')
result = nass_client.api_update_multi_year_sync(
    short_desc='CORN - ACRES PLANTED',
    start_year=2010,
    end_year=2024,
    agg_level='COUNTY',
    max_concurrent=3
)

# Check results
print(f"Updated {result['summary']['successful_updates']} out of {result['summary']['total_years_requested']} years")
print(f"Success rate: {result['summary']['success_rate']}")
```

**Direct Async QuickStats Usage:**
```python
from data.sources.usda.api_wrappers.usda_quickstats import QuickStatsClient
import asyncio

async def fetch_multiple_years():
    client = QuickStatsClient()
    
    # Use async methods for concurrent requests
    data = await client.fetch_all_async(
        commodity_desc="CORN",
        short_desc="CORN - ACRES PLANTED", 
        agg_level_desc="COUNTY",
        year__GE=2010,
        year__LE=2024,
        max_concurrent=3
    )
    return data

# Run async function
data = asyncio.run(fetch_multiple_years())
```

## Essential Geographic Columns

When retrieving NASS data at STATE or COUNTY levels, the system automatically preserves essential geographic identifier columns for data analysis and mapping:

**State-Level Identifiers:**
- `state_alpha`: 2-letter state code (e.g., "IA", "IL")
- `state_name`: Full state name (e.g., "IOWA", "ILLINOIS")  
- `state_fips_code`: 2-digit FIPS state code (e.g., "19", "17")
- `state_ansi`: ANSI state code for standardized identification

**County-Level Identifiers:**
- `county_name`: County name (e.g., "STORY", "POLK")
- `county_code`: 3-digit county FIPS code within state (e.g., "169", "153")
- `county_ansi`: ANSI county code for standardized identification
- `location_desc`: Full location description combining state and county

These identifiers enable:
- Geographic mapping and visualization
- Cross-referencing with other FIPS-based datasets
- Standardized location matching across data sources
- Integration with census and demographic data

### Callback Architecture Best Practices
- Use store-based callbacks for heavy data processing and shared data
- Use function-based callbacks for simple interactions and triggers
- Register charts individually for fine-grained control
- Use multi-chart registration for coordinated updates
- Leverage chart registry for status tracking and debugging
- Next we need to update all figures to return as a list. The schema, for some reason, expects a list every output (which is fine, just remember it when creating callbacks)
- Only use `utf-8` when rendering characters. Some unicode characters are not included in the termina;