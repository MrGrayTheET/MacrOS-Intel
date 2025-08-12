"""
ESR CSV Data Processor

This module provides functionality to clean and process raw ESR CSV downloads
into the format expected by the ESRTableClient data storage system.

The processor converts raw USDA ESR CSV exports to the standardized format
used in the application's data tables.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import toml


class ESRCSVProcessor:
    """
    Processes raw ESR CSV files into the standardized format for data storage.
    
    Handles the conversion from raw USDA ESR export format to the cleaned format
    used by ESRTableClient for storage in the {commodity}/exports/{year} keys.
    """
    
    def __init__(self):
        """Initialize the ESR CSV processor."""
        self.esr_map = self._load_esr_mappings()
        
    def _load_esr_mappings(self) -> Dict:
        """Load ESR commodity and country mappings."""
        try:
            esr_map_path = Path(__file__).parent / "esr_map.toml"
            with open(esr_map_path) as f:
                return toml.load(f)
        except Exception as e:
            print(f"Warning: Could not load esr_map.toml: {e}")
            return {}
    
    def clean_csv_data(self, csv_file_path: str, commodity: str = None) -> pd.DataFrame:
        """
        Clean raw ESR CSV data into standardized format.
        
        Args:
            csv_file_path: Path to the raw ESR CSV file
            commodity: Commodity name (will be auto-detected if not provided)
            
        Returns:
            pd.DataFrame: Cleaned data in the standardized format
        """
        # Read the raw CSV
        raw_df = pd.read_csv(csv_file_path)
        
        # Detect commodity if not provided
        if commodity is None:
            commodity = self._detect_commodity(raw_df)
            
        print(f"Processing ESR CSV for commodity: {commodity}")
        print(f"Raw data shape: {raw_df.shape}")
        
        # Clean and transform the data
        cleaned_df = self._transform_raw_data(raw_df, commodity)
        
        print(f"Cleaned data shape: {cleaned_df.shape}")
        return cleaned_df
    
    def _detect_commodity(self, df: pd.DataFrame) -> str:
        """
        Auto-detect commodity from the raw CSV data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            str: Detected commodity name
        """
        if 'Commodity' in df.columns:
            commodity_values = df['Commodity'].unique()
            
            # Map common USDA commodity names to our internal names
            commodity_mapping = {
                'ALL WHEAT': 'wheat',
                'WHEAT': 'wheat', 
                'CORN': 'corn',
                'SOYBEANS': 'soybeans',
                'BEEF': 'cattle',
                'CATTLE': 'cattle',
                'PORK': 'hogs',
                'HOGS': 'hogs'
            }
            
            for raw_commodity in commodity_values:
                if raw_commodity in commodity_mapping:
                    return commodity_mapping[raw_commodity]
                    
            # If not found in mapping, use the first commodity value (lowercase)
            if len(commodity_values) > 0:
                return commodity_values[0].lower()
                
        return 'unknown'
    
    def _transform_raw_data(self, raw_df: pd.DataFrame, commodity: str) -> pd.DataFrame:
        """
        Transform raw ESR data into the standardized format.
        
        Raw format columns:
        - Commodity, WeekEndingDate(WeekNumber), ReportingMarketingYearName, 
          CountryName(CountryCode), WeekendingDate, WeekNumber, CommodityCode, 
          CountryName, CountryCode2, GrossNewSales, NetSales, WeeklyExports,
          OutstandingSales, AccumulatedExports, TotalCommitment, NextMYNetSales,
          NextMYOutstandingSales, etc.
          
        Target format columns:
        - weekEndingDate, commodityCode, countryCode, weeklyExports, 
          accumulatedExports, outstandingSales, grossNewSales, currentMYNetSales,
          currentMYTotalCommitment, nextMYOutstandingSales, nextMYNetSales,
          unitId, country, commodity
        """
        
        # Create the standardized DataFrame
        cleaned_data = []
        
        for _, row in raw_df.iterrows():
            
            # Parse the date - handle multiple possible date formats
            week_ending_date = self._parse_date(row.get('WeekendingDate', row.get('WeekEndingDate')))
            
            # Clean numeric values (remove commas and convert to float)
            def clean_numeric(value):
                if pd.isna(value):
                    return 0.0
                if isinstance(value, str):
                    # Remove commas and quotes
                    value = value.replace(',', '').replace('"', '')
                    try:
                        return float(value)
                    except:
                        return 0.0
                return float(value) if value is not None else 0.0
            
            # Map country code and name
            country_code = row.get('CountryCode2', row.get('CountryCode', ''))
            country_name = self._clean_country_name(row.get('CountryName', ''))
            
            # Get commodity code from our mapping
            commodity_code = self._get_commodity_code(commodity)
            
            # Build the cleaned record
            cleaned_record = {
                'weekEndingDate': week_ending_date,
                'commodityCode': commodity_code,
                'countryCode': int(country_code) if str(country_code).isdigit() else 0,
                'weeklyExports': clean_numeric(row.get('WeeklyExports', 0)),
                'accumulatedExports': clean_numeric(row.get('AccumulatedExports', 0)),
                'outstandingSales': clean_numeric(row.get('OutstandingSales', 0)),
                'grossNewSales': clean_numeric(row.get('GrossNewSales', 0)),
                'currentMYNetSales': clean_numeric(row.get('NetSales', 0)),
                'currentMYTotalCommitment': clean_numeric(row.get('TotalCommitment', 0)),
                'nextMYOutstandingSales': clean_numeric(row.get('NextMYOutstandingSales', 0)),
                'nextMYNetSales': clean_numeric(row.get('NextMYNetSales', 0)),
                'unitId': 1,  # Standard unit ID
                'weekEndingDate': week_ending_date,  # Duplicate for compatibility
                'country': country_name,
                'commodity': commodity
            }
            
            cleaned_data.append(cleaned_record)
        
        # Create DataFrame and sort by date and country
        df = pd.DataFrame(cleaned_data)
        
        # Convert date column to datetime
        df['weekEndingDate'] = pd.to_datetime(df['weekEndingDate'])
        
        # Sort by date and country for consistency
        df = df.sort_values(['weekEndingDate', 'country']).reset_index(drop=True)
        
        return df
    
    def _parse_date(self, date_value) -> str:
        """
        Parse date from various formats to YYYY-MM-DD string.
        
        Args:
            date_value: Date value in various formats
            
        Returns:
            str: Standardized date string (YYYY-MM-DD)
        """
        if pd.isna(date_value):
            return datetime.now().strftime('%Y-%m-%d')
            
        # Handle string dates
        if isinstance(date_value, str):
            # Try common formats
            formats = ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_value, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
                    
        # Handle datetime objects
        if hasattr(date_value, 'strftime'):
            return date_value.strftime('%Y-%m-%d')
            
        # Default to current date if parsing fails
        print(f"Warning: Could not parse date '{date_value}', using current date")
        return datetime.now().strftime('%Y-%m-%d')
    
    def _clean_country_name(self, country_name: str) -> str:
        """
        Clean and standardize country names.
        
        Args:
            country_name: Raw country name from CSV
            
        Returns:
            str: Cleaned country name
        """
        if pd.isna(country_name):
            return 'Unknown'
            
        # Remove extra whitespace and standardize
        cleaned = str(country_name).strip()
        
        # Common country name mappings for standardization
        country_mappings = {
            'CHINA, PEOPLES REPUBLIC OF': 'China',
            'KOREA, SOUTH': 'Korea, South',
            'KOREA, REPUBLIC OF': 'Korea, South',
            'HONG KONG': 'Hong Kong',
            'TAIWAN': 'Taiwan',
            'UNITED STATES': 'United States',
            'USA': 'United States',
            'CANADA': 'Canada',
            'MEXICO': 'Mexico',
            'JAPAN': 'Japan',
            'NETHERLANDS': 'Netherlands',
            'BELGIUM-LUXEMBOURG': 'Belgium',
            'DOMINICAN REPUBLIC': 'Dominican Republic'
        }
        
        # Apply mapping if available
        cleaned_upper = cleaned.upper()
        if cleaned_upper in country_mappings:
            return country_mappings[cleaned_upper]
            
        # Title case for consistency
        return cleaned.title()
    
    def _get_commodity_code(self, commodity: str) -> int:
        """
        Get commodity code for the given commodity.
        
        Args:
            commodity: Commodity name
            
        Returns:
            int: Commodity code
        """
        if 'esr' in self.esr_map and 'commodities' in self.esr_map['esr']:
            commodities = self.esr_map['esr']['commodities']
            
            # Check aliases first
            if 'alias' in self.esr_map['esr']:
                aliases = self.esr_map['esr']['alias']
                if commodity.lower() in aliases:
                    commodity = aliases[commodity.lower()]
            
            # Get commodity code
            if commodity.lower() in commodities:
                return int(commodities[commodity.lower()])
        
        # Default commodity codes if mapping fails
        default_codes = {
            'wheat': 107,
            'corn': 104,
            'soybeans': 105,
            'cattle': 1701,
            'hogs': 1702,
            'beef': 1701,
            'pork': 1702
        }
        
        return default_codes.get(commodity.lower(), 999)
    
    def extract_year_from_data(self, df: pd.DataFrame) -> int:
        """
        Extract the marketing year from the cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            int: Marketing year
        """
        if 'weekEndingDate' in df.columns and not df.empty:
            # Get the most common year from the data
            dates = pd.to_datetime(df['weekEndingDate'])
            years = dates.dt.year
            return int(years.mode()[0])
            
        return datetime.now().year
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the cleaned data for quality and completeness.
        
        Args:
            df: Cleaned DataFrame to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required_cols = [
            'weekEndingDate', 'commodityCode', 'countryCode', 'weeklyExports',
            'accumulatedExports', 'outstandingSales', 'country', 'commodity'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty data
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check date format
        try:
            pd.to_datetime(df['weekEndingDate'])
        except Exception as e:
            issues.append(f"Invalid date format in weekEndingDate: {e}")
        
        # Check for negative values in key metrics (shouldn't normally be negative)
        numeric_cols = ['weeklyExports', 'accumulatedExports', 'outstandingSales']
        for col in numeric_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > len(df) * 0.1:  # More than 10% negative values
                    issues.append(f"High number of negative values in {col}: {negative_count}")
        
        # Check for missing country names
        missing_countries = df['country'].isna().sum()
        if missing_countries > 0:
            issues.append(f"Missing country names: {missing_countries} records")
        
        is_valid = len(issues) == 0
        return is_valid, issues


def process_esr_csv(csv_file_path: str, output_file_path: str = None, 
                   commodity: str = None) -> Tuple[pd.DataFrame, int]:
    """
    Convenience function to process an ESR CSV file.
    
    Args:
        csv_file_path: Path to the raw CSV file
        output_file_path: Path to save cleaned CSV (optional)
        commodity: Commodity name (auto-detected if None)
        
    Returns:
        Tuple[pd.DataFrame, int]: (cleaned_dataframe, marketing_year)
    """
    processor = ESRCSVProcessor()
    
    # Clean the data
    cleaned_df = processor.clean_csv_data(csv_file_path, commodity)
    
    # Extract marketing year
    marketing_year = processor.extract_year_from_data(cleaned_df)
    
    # Validate the data
    is_valid, issues = processor.validate_data(cleaned_df)
    
    if not is_valid:
        print("⚠️ Data validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Data validation passed")
    
    # Save cleaned data if output path provided
    if output_file_path:
        cleaned_df.to_csv(output_file_path, index=False)
        print(f"Cleaned data saved to: {output_file_path}")
    
    print(f"Processing complete:")
    print(f"  - Records: {len(cleaned_df)}")
    print(f"  - Countries: {cleaned_df['country'].nunique()}")
    print(f"  - Marketing Year: {marketing_year}")
    print(f"  - Date Range: {cleaned_df['weekEndingDate'].min()} to {cleaned_df['weekEndingDate'].max()}")
    
    return cleaned_df, marketing_year


# Example usage
if __name__ == "__main__":
    # Example processing
    csv_path = "esr_2025.csv"  # Path to raw CSV
    output_path = "cleaned_esr_2025.csv"  # Where to save cleaned data
    
    try:
        cleaned_data, year = process_esr_csv(csv_path, output_path, commodity="wheat")
        print(f"\n✅ Successfully processed {csv_path}")
        print(f"Marketing year: {year}")
        print(f"Sample data:")
        print(cleaned_data.head())
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")