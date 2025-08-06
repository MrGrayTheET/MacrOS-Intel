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
  - `esr_app.py`: Comprehensive ESR analysis dashboard
  - `agricultural/psd_data.py`: Production, Supply & Distribution data
  - `trade_bal.py`: Trade balance analysis
  - `energy/`: Energy commodity pages (natural gas, petroleum)

### Core Framework Components
- **components/frames.py**: Core framework classes
  - `FundamentalFrame`: Reusable chart container with data binding
  - `FlexibleMenu`: Dynamic control panel system
  - `EnhancedFrameGrid`: Layout management for multiple frames
- **callbacks/**: Dynamic callback system with registry pattern
  - `callback_registry.py`: Centralized callback management
  - Module-specific callbacks (agricultural.py, esr.py, global_callbacks.py)

### Data Layer
- **data/sources/**: API wrappers and data mappings
  - `usda/`: NASS and FAS API integrations
  - `eia/`: Energy Information Administration APIs
  - `ncei/`: Weather data integration
  - `COT/`: Commodity futures data
- **data/data_tables.py**: Core data access layer with `TableClient` and `FASTable` classes
- **config.py**: Environment configuration and API keys management

## Key Development Commands

### Running the Application
```bash
python app.py              # Main multi-page dashboard
python pages/esr_app.py    # Standalone ESR application
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
- **data_mapping.toml**: API endpoint and data field mappings
- **chart_mappings.toml**: Chart configuration templates
- **futures_mappings.toml**: COT commodity symbol mappings

### Data Paths (configurable via environment)
- `DATA_DIR`: Base data directory (default: F:\Macro\OSINT)
- `MARKET_DATA_PATH`: Market data storage
- `COT_PATH`: Commitment of Traders data

## Framework Patterns

### Creating New Charts
Use `FundamentalFrame` with chart configurations:
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

### Adding New Pages
1. Create page module in `pages/`
2. Import and register in `app.py`
3. Add navigation link in navbar
4. Use existing frame/menu components for consistency

### Data Access Pattern
```python
from data.data_tables import TableClient, FASTable

# For general data
client = TableClient()
data = client.get_key('data_key')

# For FAS/ESR data
fas_client = FASTable()
esr_data = fas_client.get_esr_data('commodity', year=2024)
```

### Callback Registration
Use the callback registry for organized callback management:
```python
from callbacks.callback_registry import CallbackRegistry
registry = CallbackRegistry()

@registry.register(
    name='callback_name',
    outputs=[('component-id', 'property')],
    inputs=[('trigger-id', 'property')]
)
def callback_function(trigger_value):
    return processed_value
```

## Data Structure Conventions

### ESR Data Format
Expected columns: `weekEndingDate`, `country`, `commodity`, `weeklyExports`, `outstandingSales`, `grossNewSales`, `currentMYNetSales`, `currentMYTotalCommitment`

### Date Handling
- Use pandas datetime conversion: `pd.to_datetime()`
- Standard date column: `weekEndingDate` for ESR, `date` for general data
- Marketing years follow USDA conventions (Oct-Sep for most commodities)

## Testing
Test files located in `test/` directory. Run individual tests:
```bash
python test/frames_test.py
```

## Development Notes

### Multi-Application Architecture
This codebase supports both:
- Single standalone applications (like `esr_app.py`)
- Multi-page integrated dashboard (`app.py`)

### API Integration
All external APIs use wrapper classes in `data/sources/`. Add new data sources by:
1. Creating API wrapper in appropriate subdirectory
2. Adding data mapping TOML file
3. Integrating with `TableClient` system

### Chart Responsiveness
Charts use percentage widths and fixed heights. For mobile responsiveness, adjust in chart_configs or use CSS media queries in `assets/styles.css`.