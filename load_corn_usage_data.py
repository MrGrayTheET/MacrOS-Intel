#!/usr/bin/env python3
"""
Script to load corn usage data files from F:\Macro\usage into storage keys
- corn_fsi.csv -> corn/usage/fsi
- corn_quarterly_use.csv -> corn/usage/quarterly
"""

import pandas as pd
from data.data_tables import TableClient

def load_corn_usage_files():
    """Load corn usage data files into the storage system"""
    try:
        print("Initializing TableClient...")
        client = TableClient()
        
        # File paths and corresponding storage keys
        files_to_load = [
            {
                'file_path': 'F:\\Macro\\usage\\corn_fsi.csv',
                'storage_key': 'corn/usage/fsi',
                'description': 'Corn FSI Usage Data'
            },
            {
                'file_path': 'F:\\Macro\\usage\\corn_quarterly_use.csv', 
                'storage_key': 'corn/usage/quarterly',
                'description': 'Corn Quarterly Usage Data'
            }
        ]
        
        for file_info in files_to_load:
            print(f"\nLoading {file_info['description']}...")
            print(f"  Source: {file_info['file_path']}")
            print(f"  Target: {file_info['storage_key']}")
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_info['file_path'])
                print(f"  ✅ Successfully read {len(df)} rows from CSV")
                
                # Convert date column to datetime if present
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    print(f"  ✅ Converted date column to datetime")
                
                # Store the data using the client
                client.put_key(file_info['storage_key'], df)
                print(f"  ✅ Successfully stored data to key: {file_info['storage_key']}")
                
                # Verify the data was stored
                test_data = client.get_key(file_info['storage_key'])
                if test_data is not None and not test_data.empty:
                    print(f"  ✅ Verification successful: Retrieved {len(test_data)} rows")
                else:
                    print(f"  ⚠️ Warning: Retrieved data appears to be empty")
                    
            except Exception as e:
                print(f"  ❌ Error processing {file_info['description']}: {e}")
                continue
                
        print(f"\n🎉 Corn usage data loading completed!")
        
        # Display summary of what was loaded
        print("\nSummary of loaded data:")
        for file_info in files_to_load:
            try:
                data = client.get_key(file_info['storage_key'])
                if data is not None:
                    print(f"  - {file_info['storage_key']}: {len(data)} rows, columns: {list(data.columns)}")
                else:
                    print(f"  - {file_info['storage_key']}: No data found")
            except Exception as e:
                print(f"  - {file_info['storage_key']}: Error retrieving data: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Fatal error in load_corn_usage_files: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_corn_usage_files()
    if success:
        print("\n✨ All corn usage data successfully loaded into storage!")
    else:
        print("\n💥 Failed to load corn usage data - check errors above")