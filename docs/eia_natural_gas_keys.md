# EIA Natural Gas Data Keys

Based on the EIATable analysis, here are all available natural gas data keys:

## Storage Data (7 keys)
- `/NG/storage/east_region` - Natural gas storage in East region
- `/NG/storage/midwest_region` - Natural gas storage in Midwest region  
- `/NG/storage/mountain_region` - Natural gas storage in Mountain region
- `/NG/storage/pacific_region` - Natural gas storage in Pacific region
- `/NG/storage/south_central_region` - Natural gas storage in South Central region
- `/NG/storage/total_lower_48` - **Total Lower 48 storage (primary key)**
- `/NG/storage/total_underground` - Total underground storage

## Price Data (9 keys)
- `/NG/prices/NG_1` - **Natural Gas Futures (Front Month)**
- `/NG/prices/NG_2` - Natural Gas Futures (2nd Month)
- `/NG/prices/NG_3` - Natural Gas Futures (3rd Month)
- `/NG/prices/NG_4` - Natural Gas Futures (4th Month)
- `/NG/prices/canada_import` - Canada import prices
- `/NG/prices/citygate_monthly` - City gate monthly prices
- `/NG/prices/henry_hub_daily` - Henry Hub daily spot prices
- `/NG/prices/mexico_import` - Mexico import prices
- `/NG/prices/residential_monthly` - Residential monthly prices

## Consumption Data (6 keys)
- `/NG/consumption/commercial` - Commercial consumption
- `/NG/consumption/electric_power` - Electric power consumption
- `/NG/consumption/gross_withdrawals` - Gross withdrawals
- `/NG/consumption/industrial` - Industrial consumption
- `/NG/consumption/net_withdrawals` - **Net withdrawals (primary key)**
- `/NG/consumption/residential` - Residential consumption
- `/NG/consumption/total` - Total consumption

## Production Data (1 key)
- `/NG/production/dry_production_monthly` - **Dry production monthly**

## Import Data (3 keys)
- `/NG/imports/Canada` - Imports from Canada
- `/NG/imports/Mexico` - Imports from Mexico
- `/NG/imports/pipeline` - Pipeline imports

## Export Data (3 keys)
- `/NG/exports/lng` - LNG exports
- `/NG/exports/pipeline_mexico` - Pipeline exports to Mexico
- `/NG/exports/total` - Total exports

## Futures Data (4 keys)
- `/NG/futures/contract_1` - Futures contract 1
- `/NG/futures/contract_2` - Futures contract 2
- `/NG/futures/contract_3` - Futures contract 3
- `/NG/futures/contract_4` - Futures contract 4

## Key Analysis Applications

### Primary Storage Analysis (Current Implementation)
- **Storage**: `storage/total_lower_48`
- **Price**: `prices/NG_1` 
- **Consumption**: `consumption/net_withdrawals`

### Regional Storage Analysis
- Compare storage levels across regions:
  - East: `storage/east_region`
  - Midwest: `storage/midwest_region`
  - Mountain: `storage/mountain_region`
  - Pacific: `storage/pacific_region`
  - South Central: `storage/south_central_region`

### Price Analysis Applications
- **Futures Curve**: `prices/NG_1`, `prices/NG_2`, `prices/NG_3`, `prices/NG_4`
- **Spot vs Futures**: `prices/henry_hub_daily` vs `prices/NG_1`
- **Regional Pricing**: Compare import prices from Canada and Mexico

### Supply/Demand Analysis
- **Supply**: `production/dry_production_monthly` + `imports/pipeline` + `imports/Canada` + `imports/Mexico`
- **Demand**: `consumption/total` or sector-specific consumption
- **Balance**: Net storage changes via `consumption/net_withdrawals`

### Seasonal Analysis
- **Injection Season**: April-October (storage builds)
- **Withdrawal Season**: November-March (storage draws)
- Key relationship: Storage levels vs heating/cooling demand

## Data Quality Notes
- Data range: 1973-2025 (5,558 total rows)
- Daily/weekly/monthly frequency depending on series
- Some series may have NaN values in early periods
- Most current data available through June 2025

## Recommended Analysis Combinations

### 1. Storage-Price Relationship
```python
keys = ['storage/total_lower_48', 'prices/NG_1', 'consumption/net_withdrawals']
```

### 2. Regional Storage Analysis  
```python
keys = ['storage/east_region', 'storage/midwest_region', 'storage/south_central_region', 'storage/pacific_region']
```

### 3. Futures Curve Analysis
```python
keys = ['prices/NG_1', 'prices/NG_2', 'prices/NG_3', 'prices/NG_4']
```

### 4. Supply/Demand Balance
```python  
keys = ['production/dry_production_monthly', 'consumption/total', 'imports/pipeline', 'exports/total']
```

### 5. Sectoral Demand Analysis
```python
keys = ['consumption/residential', 'consumption/commercial', 'consumption/industrial', 'consumption/electric_power']
```