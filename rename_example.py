"""
Quick example script to rename soybean -> soybeans tables
"""

from data.data_tables import ESRTableClient

def main():
    # Initialize the ESR client
    client = ESRTableClient()
    
    print("Current tables:")
    client.list_all_tables()
    
    print("\n" + "="*50)
    print("Renaming soybean -> soybeans tables")
    print("="*50)
    
    # First do a dry run to see what would be renamed
    print("\nDry run results:")
    dry_results = client.rename_table_keys('soybean', 'soybeans', dry_run=True)
    
    if dry_results:
        print(f"\nFound {len(dry_results)} tables to rename.")
        print("Proceeding with actual rename...")
        
        # Actually perform the rename
        results = client.rename_table_keys('soybean', 'soybeans', dry_run=False)
        
        print(f"\nRename completed! {len(results)} tables processed.")
        
        print("\nUpdated tables:")
        client.list_all_tables()
    else:
        print("No tables found matching 'soybean' pattern.")

if __name__ == "__main__":
    main()