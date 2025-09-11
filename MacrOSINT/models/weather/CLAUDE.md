# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive agricultural climate data analysis project that retrieves and analyzes weather/climate data from NOAA's NCEI (National Centers for Environmental Information) datasets. The project focuses on agricultural applications, particularly analyzing climate conditions for major crop-producing regions in the United States.

## Key Components

### Core Modules

1. **agclimate.py** - Main agricultural climate data module (1245 lines)
   - `AgClimateAPI` class for accessing GHCND (daily) and GSOM (monthly) datasets
   - `CropClimateAnalyzer` for crop-specific climate analysis
   - `DataQualityProcessor` for data validation and cleaning
   - `ModelDataConverter` for exporting to agricultural modeling formats (APSIM, DSSAT)

2. **crop_workflow.py** - Extended climate analysis workflow (831 lines)
   - `NCEInClimDivAPI` for nClimDiv dataset access
   - `NCEI_GHCND_API` for GHCND station management
   - `CropClimateAnalyzer` for crop suitability analysis
   - Report generation and visualization tools

3. **config.py** - Configuration management
   - Loads environment variables from `.env` file
   - Manages NCEI API token (`NCEI_TOKEN`)

### Data Sources & APIs

The project integrates with multiple NOAA datasets:
- **GHCND** (Global Historical Climatology Network - Daily)
- **GSOM** (Global Summary of the Month)
- **nClimDiv** (Climate Divisional Dataset)

### Agricultural Focus Areas

Pre-configured agricultural regions:
- **Corn Belt**: Iowa, Illinois, Indiana, Ohio, Missouri, Nebraska, Minnesota, etc.
- **Wheat Belt**: Kansas, North Dakota, Montana, Washington, Oklahoma, etc.
- **Cotton Belt**: Texas, Georgia, Mississippi, Alabama, etc.
- **California Central Valley**: Intensive agriculture region
- **Pacific Northwest**: Wheat, potatoes, apples

### Crop Parameters

Built-in crop-specific parameters for:
- Corn/Maize, Soybeans, Winter/Spring Wheat, Cotton, Rice
- Growing degree days (GDD) calculations
- Optimal temperature and precipitation ranges
- Critical growth periods and stress indices

## Development Workflow
- Only use standard ASCII characters, no unicode/emojis

### Required Dependencies

The project uses these key Python packages:
- `pyncei` - NCEI API client
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- 
- `scipy` - Statistical analysis
- `matplotlib`, `seaborn` - Visualization
- `geopandas` (optional) - Spatial analysis
- `python-dotenv` - Environment variable management

### API Token Setup

1. Obtain NCEI API token from: https://www.ncdc.noaa.gov/cdo-web/token
2. Create `.env` file with: `NCEI_TOKEN=your_token_here`
3. Token is loaded via `config.py`

### Running Analysis

Main analysis workflows are defined in both modules:
- Run `python agclimate.py` for comprehensive GHCND/GSOM analysis
- Run `python crop_workflow.py` for nClimDiv regional analysis
- Both modules include complete example workflows when run as main

### Data Processing Pipeline

1. **Station Selection**: Geographic filtering with data coverage requirements
2. **Data Retrieval**: Year-by-year chunking for large datasets (GHCND) or bulk download (GSOM/nClimDiv)
3. **Quality Control**: Flag-based filtering, extreme value detection, logical consistency checks
4. **Gap Filling**: Interpolation, climatology, or forward-fill methods
5. **Agricultural Analysis**: Growing season statistics, GDD calculations, stress indices
6. **Report Generation**: Markdown format with trends, risks, and suitability assessments

### Output Formats

The project generates multiple output types:
- **CSV exports**: Raw and processed climate data
- **Markdown reports**: Comprehensive analysis summaries
- **Agricultural model formats**: APSIM, DSSAT weather files
- **Visualizations**: Time series plots with trend analysis

## Common Tasks

### Retrieving Climate Data

```python
from agclimate import AgClimateAPI
api = AgClimateAPI()  # Uses NCEI_TOKEN from config

# Get daily data for Corn Belt
daily_data = api.get_agricultural_data(
    dataset='GHCND',
    region='corn_belt',
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    max_stations=10
)
```

### Running Crop Analysis

```python
from agclimate import CropClimateAnalyzer
analyzer = CropClimateAnalyzer()
corn_analysis = analyzer.analyze_crop_climate(daily_data, 'corn')
```

### Generating Reports

```python
from agclimate import create_climate_report
report = create_climate_report(corn_analysis, 'output_report.md')
```

## Important Notes

- **Rate Limiting**: API calls are throttled (0.2s delays) to respect NCEI limits
- **Data Caching**: Uses SQLite caching (`agclimate_cache.sqlite`, `agricultural_cache.sqlite`)
- **Error Handling**: Comprehensive exception handling for network and data issues
- **Memory Management**: Year-by-year chunking for large GHCND datasets
- **Quality Flags**: Extensive data quality assessment using NCEI quality flags

## File Structure

- Core analysis modules are the main entry points
- Generated reports and CSV files are outputs
- Cache databases store retrieved data for performance
- Configuration is centralized in `config.py`
- No formal test suite or build system currently implemented