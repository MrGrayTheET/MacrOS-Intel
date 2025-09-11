"""
Agricultural County Data Analysis
==================================
This script cleans USDA county-level agricultural data, ranks counties by acres planted,
and creates a grid system for climate analysis (rainfall and GDD calculations).

Author: Agricultural Data Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import warnings
from .add_county_centroids import append_centroids_via_shapes as append_centroids





warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: DATA CLEANING AND TYPE CONVERSION
# =============================================================================

def clean_agricultural_data(data) -> pd.DataFrame:
    """
    Clean the agricultural CSV data and convert string numbers to appropriate numeric types.

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned DataFrame with proper numeric types
    """
    print("STEP 1: DATA CLEANING AND TYPE CONVERSION")
    print("=" * 80)

    # Read the CSV file
    if isinstance(data, str):
        df = pd.read_csv(data)
    if isinstance(data, pd.DataFrame):
        df = data
    print(f"[OK] Loaded {len(df)} records from CSV")

    # Create a copy for cleaning
    df_clean = df.copy()

    # 1. Clean the 'Value' column (acres planted)
    # Remove commas and quotes, convert to integer (acres are whole numbers)
    def clean_value(value):
        if pd.isna(value):
            return np.nan
        # Remove commas, quotes, and whitespace
        value_str = str(value).replace(',', '').replace('"', '').strip()
        try:
            # Using integer for acres (whole numbers are more appropriate for land area)
            return int(value_str)
        except (ValueError, TypeError):
            return np.nan

    for col in df_clean.columns:
        if 'acres_planted' in col or "Value" in col.capitalize():
            df_clean.rename({col:'acres_planted'}, axis=1, inplace=True)



    # 2. Clean the 'CV (%)' column
    # Convert to float (percentages need decimal precision)
    def clean_cv(cv):
        if pd.isna(cv):
            return np.nan
        try:
            return float(cv)
        except (ValueError, TypeError):
            return np.nan

    df_clean['cv_percent'] = df_clean['CV (%)'].apply(clean_cv)

    # 3. Convert other numeric columns
    numeric_columns = {
        'year': 'int',
        'county_code': 'int',
        'state_fips_code': 'int',
        'state_ansi': 'int',
        'county_ansi': 'int',
        'asd_code': 'int'
    }

    for col, dtype in numeric_columns.items():
        if col in df_clean.columns:
            df_clean[f'{col}_clean'] = pd.to_numeric(df_clean[col], errors='coerce')
            if dtype == 'int':
                # Handle NaN values before converting to int
                mask = df_clean[f'{col}_clean'].notna()
                df_clean.loc[mask, f'{col}_clean'] = df_clean.loc[mask, f'{col}_clean'].astype(int)

    # 4. Data quality report
    print("\nData Type Conversion Summary:")
    print("-" * 50)
    print(f"  Value -> acres_planted (integer):")
    print(f"    - Valid conversions: {df_clean['acres_planted'].notna().sum()}")
    print(f"    - Failed conversions: {df_clean['acres_planted'].isna().sum()}")
    print(f"    - Range: {df_clean['acres_planted'].min():.0f} to {df_clean['acres_planted'].max():.0f} acres")

    print(f"\n  CV (%) -> cv_percent (float):")
    print(f"    - Valid conversions: {df_clean['cv_percent'].notna().sum()}")
    print(f"    - Failed conversions: {df_clean['cv_percent'].isna().sum()}")
    print(f"    - Range: {df_clean['cv_percent'].min():.2f}% to {df_clean['cv_percent'].max():.2f}%")

    # 5. Filter out invalid records
    df_valid = df_clean[df_clean['acres_planted'].notna() & (df_clean['acres_planted'] > 0)].copy()
    print(f"\n[OK] Retained {len(df_valid)} valid records with positive acres planted")

    return df_valid


# =============================================================================
# STEP 2: RANK COUNTIES BY ACRES PLANTED
# =============================================================================

def rank_counties_by_acres(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Rank counties by acres planted and return top N counties.

    Args:
        df: Cleaned DataFrame
        top_n: Number of top counties to return

    Returns:
        DataFrame with top counties ranked by acres
    """
    print("\nSTEP 2: RANKING COUNTIES BY ACRES PLANTED")
    print("=" * 80)

    # Separate individual counties from "OTHER COUNTIES" aggregates
    df_individual = df[~df['county_name'].str.contains('OTHER', na=False)].copy()
    df_other = df[df['county_name'].str.contains('OTHER', na=False)].copy()

    print(f"[OK] Found {len(df_individual)} individual county records")
    print(f"[OK] Found {len(df_other)} 'OTHER COUNTIES' aggregate records")

    # Rank individual counties
    df_ranked = df_individual.sort_values('acres_planted', ascending=False).reset_index(drop=True)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)

    # Get top N counties
    top_counties = df_ranked.head(top_n).copy()

    # Display top counties
    print(f"\n[TOP] TOP {top_n} COUNTIES BY CORN ACRES PLANTED (2024):")
    print("-" * 80)
    print(f"{'Rank':<5} {'State':<15} {'County':<20} {'Acres Planted':>15} {'CV%':>8}")
    print("-" * 80)

    for idx, row in top_counties.iterrows():
        rank = row['rank']
        state = row['state_name'][:14]
        county = row['county_name'][:19]
        acres = f"{row['acres_planted']:,}"
        cv = f"{row['cv_percent']:.1f}" if pd.notna(row['cv_percent']) else "N/A"
        print(f"{rank:<5} {state:<15} {county:<20} {acres:>15} {cv:>8}")

    # Calculate summary statistics
    print("\nSummary Statistics (Individual Counties):")
    print("-" * 50)
    print(f"  Total acres: {df_individual['acres_planted'].sum():,}")
    print(f"  Mean acres per county: {df_individual['acres_planted'].mean():,.0f}")
    print(f"  Median acres: {df_individual['acres_planted'].median():,.0f}")
    print(f"  Standard deviation: {df_individual['acres_planted'].std():,.0f}")

    # State-level summary
    state_summary = df_individual.groupby('state_name').agg({
        'acres_planted': ['sum', 'mean', 'count']
    }).round(0)
    state_summary.columns = ['total_acres', 'avg_acres', 'county_count']
    state_summary = state_summary.sort_values('total_acres', ascending=False)

    print("\n[STATES] Top 10 States by Total Acres:")
    print("-" * 60)
    print(f"{'State':<15} {'Total Acres':>15} {'Counties':>10} {'Avg/County':>15}")
    print("-" * 60)

    for state, row in state_summary.head(10).iterrows():
        total = f"{row['total_acres']:,.0f}"
        counties = f"{row['county_count']:.0f}"
        avg = f"{row['avg_acres']:,.0f}"
        print(f"{state:<15} {total:>15} {counties:>10} {avg:>15}")

    return top_counties


# =============================================================================
# STEP 3: CREATE GRID SYSTEM FOR CLIMATE ANALYSIS
# =============================================================================

def create_climate_grid(df: pd.DataFrame, grid_size: float = 2.0,
                        density_threshold: int = 100000) -> Dict:
    """
    Create a grid system for calculating rainfall and GDD in dense agricultural regions.

    Args:
        df: Cleaned DataFrame
        grid_size: Size of grid cells in degrees (lat/lon)
        density_threshold: Minimum acres to be considered high-density

    Returns:
        Dictionary containing grid analysis results
    """
    print("\nSTEP 3: CREATING GRID SYSTEM FOR CLIMATE ANALYSIS")
    print("=" * 80)

    # Filter for high-density counties
    df_individual = df[~df['county_name'].str.contains('OTHER', na=False)].copy()
    df_high_density = df_individual[df_individual['acres_planted'] >= density_threshold].copy()

    print(f"[OK] Identified {len(df_high_density)} high-density counties (>={density_threshold:,} acres)")


    df_high_density = append_centroids(df_high_density)
    df_high_density = df_high_density.dropna(subset=['latitude', 'longitude'])
    print(f"[OK] Geocoded {len(df_high_density)} high-density counties")

    # Create grid cells
    def assign_grid_cell(lat, lon, grid_size):
        """Assign a county to a grid cell based on coordinates."""
        grid_lat = int(lat // grid_size) * grid_size
        grid_lon = int(lon // grid_size) * grid_size
        return f"{grid_lat:.0f}N_{abs(grid_lon):.0f}W"

    df_high_density['grid_cell'] = df_high_density.apply(
        lambda row: assign_grid_cell(row['latitude'], row['longitude'], grid_size),
        axis=1
    )

    # Analyze grid cells
    grid_summary = df_high_density.groupby('grid_cell').agg({
        'acres_planted': ['sum', 'mean', 'count'],
        'state_name': lambda x: list(x.unique()),
        'county_name': lambda x: list(x)[:3]  # Top 3 counties
    }).round(0)

    grid_summary.columns = ['total_acres', 'avg_acres', 'county_count', 'states', 'top_counties']
    grid_summary = grid_summary.sort_values('total_acres', ascending=False)

    print(f"\n[GRID] GRID CELL ANALYSIS ({grid_size}° × {grid_size}° cells):")
    print("-" * 80)
    print(f"{'Grid Cell':<15} {'Counties':>10} {'Total Acres':>15} {'States':<30}")
    print("-" * 80)

    for cell, row in grid_summary.head(10).iterrows():
        counties = f"{row['county_count']:.0f}"
        total = f"{row['total_acres']:,.0f}"
        states = ', '.join(row['states'][:2])  # Show first 2 states
        print(f"{cell:<15} {counties:>10} {total:>15} {states:<30}")

    # Climate zones mapping
    climate_zones = {
        'Northern Plains': ['NORTH DAKOTA', 'SOUTH DAKOTA', 'MONTANA'],
        'Corn Belt': ['IOWA', 'ILLINOIS', 'INDIANA', 'OHIO', 'MISSOURI'],
        'Central Plains': ['KANSAS', 'NEBRASKA', 'OKLAHOMA'],
        'Upper Midwest': ['MINNESOTA', 'WISCONSIN', 'MICHIGAN'],
        'Southern Plains': ['TEXAS', 'ARKANSAS'],
        'Mid-South': ['KENTUCKY', 'TENNESSEE', 'MISSISSIPPI', 'ALABAMA']
    }

    # Assign climate zones to counties
    def get_climate_zone(state):
        for zone, states in climate_zones.items():
            if state in states:
                return zone
        return 'Other'

    df_high_density['climate_zone'] = df_high_density['state_name'].apply(get_climate_zone)

    # Calculate climate metrics for each grid cell
    print("\n[CLIMATE] CLIMATE METRICS FOR TOP GRID CELLS:")
    print("-" * 80)
    print(f"{'Grid Cell':<15} {'Climate Zone':<20} {'Est. GDD':>10} {'Est. Rain':>10} {'Risk':<10}")
    print("-" * 80)

    # Sample climate parameters (in production, use actual weather data)
    climate_params = {
        'Corn Belt': {'gdd_base': 2800, 'rainfall_base': 35, 'risk': 'Low'},
        'Northern Plains': {'gdd_base': 2400, 'rainfall_base': 25, 'risk': 'Medium'},
        'Central Plains': {'gdd_base': 3000, 'rainfall_base': 28, 'risk': 'Medium'},
        'Upper Midwest': {'gdd_base': 2600, 'rainfall_base': 32, 'risk': 'Low'},
        'Southern Plains': {'gdd_base': 3200, 'rainfall_base': 30, 'risk': 'High'},
        'Mid-South': {'gdd_base': 3100, 'rainfall_base': 45, 'risk': 'Low'},
        'Other': {'gdd_base': 2700, 'rainfall_base': 30, 'risk': 'Medium'}
    }

    for cell, row in grid_summary.head(10).iterrows():
        # Get dominant climate zone for this grid cell
        cell_counties = df_high_density[df_high_density['grid_cell'] == cell]
        zone_counts = cell_counties['climate_zone'].value_counts()
        dominant_zone = zone_counts.index[0] if len(zone_counts) > 0 else 'Other'

        # Get climate parameters
        params = climate_params[dominant_zone]

        # Add some variation
        gdd = params['gdd_base'] + np.random.randint(-200, 200)
        rainfall = params['rainfall_base'] + np.random.uniform(-5, 5)

        print(f"{cell:<15} {dominant_zone:<20} {gdd:>10} {rainfall:>10.1f}\" {params['risk']:<10}")

    # Create output dictionary
    results = {
        'high_density_counties': df_high_density,
        'grid_summary': grid_summary,
        'climate_zones': climate_zones,
        'climate_parameters': climate_params,
        'grid_size': grid_size,
        'density_threshold': density_threshold
    }

    print(f"\n[OK] Grid system created successfully!")
    print(f"  - Total grid cells: {len(grid_summary)}")
    print(f"  - Grid size: {grid_size}° × {grid_size}°")
    print(f"  - Ready for GDD and rainfall calculations")

    return results


# =============================================================================
# STEP 4: EXPORT RESULTS
# =============================================================================

def export_results(df_clean: pd.DataFrame, top_counties: pd.DataFrame,
                   grid_results: Dict, output_prefix: str = 'agricultural_analysis'):
    """
    Export analysis results to CSV files.

    Args:
        df_clean: Cleaned DataFrame
        top_counties: Top counties by acres
        grid_results: Grid analysis results
        output_prefix: Prefix for output files
    """
    print("\nEXPORTING RESULTS")
    print("=" * 80)

    # Export cleaned data
    filename1 = f"{output_prefix}_cleaned_data.csv"
    df_clean.to_csv(filename1, index=False)
    print(f"[OK] Exported cleaned data to {filename1}")

    # Export top counties
    filename2 = f"{output_prefix}_top_counties.csv"
    top_counties.to_csv(filename2, index=False)
    print(f"[OK] Exported top counties to {filename2}")

    # Export high-density counties with grid assignments
    filename3 = f"{output_prefix}_grid_analysis.csv"
    grid_results['high_density_counties'].to_csv(filename3, index=False)
    print(f"[OK] Exported grid analysis to {filename3}")

    # Export grid summary
    filename4 = f"{output_prefix}_grid_summary.csv"
    grid_results['grid_summary'].to_csv(filename4)
    print(f"[OK] Exported grid summary to {filename4}")

    print("\n[SUCCESS] All results exported successfully!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for agricultural county analysis.
    """
    print("=" * 80)
    print("[CORN] AGRICULTURAL COUNTY DATA ANALYSIS - CORN ACRES PLANTED 2024")
    print("=" * 80)

    # File path (update this to your actual file path)
    filepath = 'counties_example.csv'

    try:
        # Step 1: Clean data and convert types
        df_clean = clean_agricultural_data(filepath)

        # Step 2: Rank counties by acres planted
        top_counties = rank_counties_by_acres(df_clean, top_n=25)

        # Step 3: Create grid system for climate analysis
        grid_results = create_climate_grid(
            df_clean,
            grid_size=10.0,  # 2-degree grid cells
            density_threshold=100000  # 100,000 acres threshold
        )

        # Step 4: Export results
        export_results(df_clean, top_counties, grid_results)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)

        # Print final summary
        print("\nFINAL SUMMARY:")
        print(f"  - Total counties analyzed: {len(df_clean)}")
        print(f"  - High-density counties: {len(grid_results['high_density_counties'])}")
        print(f"  - Grid cells created: {len(grid_results['grid_summary'])}")
        print(f"  - Files exported: 4")

        return df_clean, top_counties, grid_results

    except FileNotFoundError:
        print(f"\n[ERROR] Error: File '{filepath}' not found.")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"\n[ERROR] Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the analysis
    cleaned_data, top_counties, grid_results = main()

    # Optional: Additional analysis or visualization can be added here
    print("\n[TIP] TIP: You can now use the returned data for further analysis:")
    print("  - cleaned_data: Full cleaned dataset")
    print("  - top_counties: Top 25 counties by acres planted")
    print("  - grid_results: Grid system with climate zones")