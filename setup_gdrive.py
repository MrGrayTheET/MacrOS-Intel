#!/usr/bin/env python3
"""
Google Drive Setup Script for TableClient Integration

This script helps set up Google Drive integration for the TableClient class.
It provides utilities to:
- Test authentication
- List available files
- Create configuration files
- Test data access

Usage:
    python setup_gdrive.py --help
"""

import argparse
import json
import toml
from pathlib import Path
from data.google_drive_client import GoogleDriveClient, setup_google_drive_auth, create_gdrive_config_template


def test_authentication(credentials_file='credentials.json'):
    """Test Google Drive authentication."""
    print("Testing Google Drive authentication...")
    
    try:
        client = GoogleDriveClient(credentials_file=credentials_file)
        print("✅ Authentication successful!")
        
        # Test listing files
        files = client.list_files()
        print(f"✅ Found {len(files)} .h5 files in your Google Drive")
        
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def list_h5_files(credentials_file='credentials.json', folder_id=None):
    """List all .h5 files in Google Drive."""
    try:
        client = GoogleDriveClient(credentials_file=credentials_file)
        files = client.list_files(folder_id=folder_id)
        
        if not files:
            print("No .h5 files found")
            return
        
        print(f"\nFound {len(files)} .h5 files:")
        print("-" * 80)
        print(f"{'Name':<40} {'Size (MB)':<12} {'Modified':<20} {'File ID'}")
        print("-" * 80)
        
        for file_info in files:
            name = file_info['name']
            size_mb = int(file_info.get('size', 0)) / (1024 * 1024)
            modified = file_info.get('modifiedTime', 'Unknown')[:10]  # Just date part
            file_id = file_info['id']
            
            print(f"{name:<40} {size_mb:>8.1f} MB   {modified:<20} {file_id}")
        
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def create_config_file(output_file='config/gdrive_config.toml'):
    """Create a configuration file template."""
    config = create_gdrive_config_template()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to TOML format
    toml_config = {
        'google_drive': {
            'folder_id': config['folder_id'],
            'credentials_file': config['credentials_file'],
            'cache_dir': config['cache_dir'],
            'file_mappings': config['file_mappings']
        }
    }
    
    with open(output_path, 'w') as f:
        toml.dump(toml_config, f)
    
    print(f"✅ Created configuration template: {output_path}")
    print("Please edit this file with your actual Google Drive folder ID and file mappings.")


def create_file_mappings(credentials_file='credentials.json', folder_id=None, output_file=None):
    """Create file mappings from available files."""
    files = list_h5_files(credentials_file, folder_id)
    
    if not files:
        print("No files found to create mappings")
        return
    
    mappings = {}
    print("\nCreating file mappings...")
    print("Suggested logical names for your files:")
    
    for file_info in files:
        name = file_info['name']
        file_id = file_info['id']
        
        # Generate logical name from filename
        logical_name = name.lower().replace('.h5', '').replace(' ', '_').replace('-', '_')
        mappings[logical_name] = file_id
        
        print(f"  {logical_name} = '{file_id}'  # {name}")
    
    if output_file:
        config = {
            'google_drive': {
                'folder_id': folder_id or 'your_folder_id_here',
                'credentials_file': credentials_file,
                'cache_dir': './gdrive_cache',
                'file_mappings': mappings
            }
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            toml.dump(config, f)
        
        print(f"\n✅ Created configuration file: {output_path}")
    
    return mappings


def test_data_access(config_file='config/gdrive_config.toml'):
    """Test data access with created configuration."""
    try:
        # Load config
        with open(config_file, 'r') as f:
            config = toml.load(f)
        
        gdrive_config = config['google_drive']
        
        # Test client
        from data.google_drive_client import GoogleDriveTableClient
        client = GoogleDriveTableClient(gdrive_config)
        
        print("Testing data access...")
        
        # List files
        files = client.get_available_files()
        print(f"✅ Can access {len(files)} files")
        
        # Test loading keys from first file
        if files and gdrive_config['file_mappings']:
            first_mapping = next(iter(gdrive_config['file_mappings'].items()))
            logical_name, file_id = first_mapping
            
            print(f"Testing keys from {logical_name}...")
            keys = client.get_h5_keys(logical_name)
            print(f"✅ Found {len(keys)} keys in {logical_name}")
            
            if keys:
                print("Sample keys:")
                for key in keys[:5]:  # Show first 5 keys
                    print(f"  - {key}")
                if len(keys) > 5:
                    print(f"  ... and {len(keys) - 5} more")
        
        print("✅ Google Drive integration is working!")
        
    except Exception as e:
        print(f"❌ Error testing data access: {e}")


def main():
    parser = argparse.ArgumentParser(description='Set up Google Drive integration for TableClient')
    parser.add_argument('--credentials', default='credentials.json', 
                       help='Path to Google OAuth2 credentials file')
    parser.add_argument('--folder-id', help='Google Drive folder ID')
    parser.add_argument('--config-file', default='config/gdrive_config.toml',
                       help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Complete setup process')
    
    # Auth test command
    auth_parser = subparsers.add_parser('test-auth', help='Test authentication only')
    
    # List files command
    list_parser = subparsers.add_parser('list-files', help='List available .h5 files')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create configuration file')
    
    # Test data access command
    test_parser = subparsers.add_parser('test-access', help='Test data access')
    
    # Instructions command
    instructions_parser = subparsers.add_parser('instructions', help='Show setup instructions')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        print("🚀 Setting up Google Drive integration...\n")
        
        # Step 1: Test authentication
        if not test_authentication(args.credentials):
            print("\n❌ Setup failed at authentication step")
            print("Please check your credentials file and try again")
            return
        
        # Step 2: List files and create mappings
        print("\n📁 Listing available files...")
        create_file_mappings(args.credentials, args.folder_id, args.config_file)
        
        # Step 3: Test data access
        print("\n🧪 Testing data access...")
        test_data_access(args.config_file)
        
        print("\n✅ Setup complete!")
        print(f"Edit {args.config_file} to customize your configuration")
        
    elif args.command == 'test-auth':
        test_authentication(args.credentials)
        
    elif args.command == 'list-files':
        list_h5_files(args.credentials, args.folder_id)
        
    elif args.command == 'create-config':
        create_config_file(args.config_file)
        
    elif args.command == 'test-access':
        test_data_access(args.config_file)
        
    elif args.command == 'instructions':
        print(setup_google_drive_auth())
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()