# Google Drive Integration for TableClient

This document explains how to set up and use Google Drive integration with the TableClient class to remotely access .h5 files.

## Features

- **Remote Access**: Access .h5 files stored on Google Drive without downloading them permanently
- **Intelligent Caching**: Files are cached locally and only re-downloaded when they change
- **Seamless Fallback**: Automatically falls back to local files if Google Drive is unavailable
- **File Mapping**: Use logical names instead of Google Drive file IDs
- **Batch Operations**: Sync multiple files at once
- **Easy Setup**: Guided setup process with authentication handling

## Installation

1. Install required Google API libraries:
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Setup Process

### Step 1: Google Cloud Console Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API for your project
4. Create OAuth 2.0 credentials for a desktop application
5. Download the credentials JSON file and save it as `credentials.json` in your project root

### Step 2: Run Setup Script

Use the included setup script to configure your integration:

```bash
# Get setup instructions
python setup_gdrive.py instructions

# Complete automated setup
python setup_gdrive.py setup

# Or step by step:
python setup_gdrive.py test-auth
python setup_gdrive.py list-files
python setup_gdrive.py create-config
```

### Step 3: Configure File Mappings

Edit the generated `config/gdrive_config.toml` file:

```toml
[google_drive]
folder_id = "your_google_drive_folder_id_here"
credentials_file = "credentials.json"
cache_dir = "./gdrive_cache"

[google_drive.file_mappings]
nass_data = "1ABC123_your_nass_file_id"
esr_data = "1XYZ789_your_esr_file_id"
market_data = "1DEF456_your_market_file_id"
```

## Usage

### Basic Setup

```python
from data.data_tables import TableClient
import toml

# Load configuration
with open('config/gdrive_config.toml', 'r') as f:
    config = toml.load(f)

# Create TableClient with Google Drive integration
client = TableClient(
    client=None,
    data_folder='./data',
    db_file_name='nass_data.h5',
    key_prefix='nass',
    gdrive_config=config['google_drive']
)
```

### Core Operations

#### Sync Files from Google Drive
```python
# Sync all mapped files
client.sync_from_gdrive()

# Sync specific file
client.sync_from_gdrive('nass_data', force_download=True)
```

#### Access Data Directly from Google Drive
```python
# Get data from Google Drive (with caching)
data = client.get_key_from_gdrive('cattle/production', 'nass_data')

# Falls back to local if Google Drive unavailable
data = client.get_key_from_gdrive('cattle/production')
```

#### File and Key Management
```python
# List available files on Google Drive
files = client.list_gdrive_files()

# Get available keys from a Google Drive file
keys = client.get_gdrive_keys('nass_data')

# Check Google Drive integration status
status = client.get_gdrive_status()
print(status)
```

#### Cache Management
```python
# Clear old cache files (older than 7 days)
client.clear_gdrive_cache(older_than_days=7)

# Clear all cache files
client.clear_gdrive_cache()
```

### Advanced Usage

#### Custom File Mappings
```python
gdrive_config = {
    'folder_id': 'your_folder_id',
    'credentials_file': 'credentials.json',
    'cache_dir': './custom_cache',
    'file_mappings': {
        'production_data': 'file_id_1',
        'historical_data': 'file_id_2',
        'market_analysis': 'file_id_3'
    }
}
```

#### Handling Multiple Data Sources
```python
# Different clients for different data types
nass_client = TableClient(
    client=None,
    data_folder='./data/nass',
    db_file_name='nass_data.h5',
    key_prefix='nass',
    gdrive_config=nass_gdrive_config
)

esr_client = TableClient(
    client=None,
    data_folder='./data/esr',
    db_file_name='esr_data.h5', 
    key_prefix='esr',
    gdrive_config=esr_gdrive_config
)
```

## Configuration Options

### Google Drive Configuration
- `folder_id`: Google Drive folder ID containing your .h5 files
- `credentials_file`: Path to OAuth2 credentials JSON file
- `cache_dir`: Local directory for cached files
- `file_mappings`: Dictionary mapping logical names to Google Drive file IDs

### TableClient Integration
The Google Drive config is passed to the TableClient constructor:
```python
TableClient(..., gdrive_config=config)
```

## File Management

### Cache Behavior
- Files are downloaded to `cache_dir` on first access
- Cached files are used if they match the Google Drive version (by modification time and checksum)
- Files are automatically re-downloaded if the Google Drive version is newer
- Cache can be manually cleared or managed by age

### File Versioning
The system uses Google Drive's modification time and MD5 checksums to detect changes:
- If local cache matches Google Drive metadata, cached version is used
- If Google Drive file is newer or has different checksum, file is re-downloaded
- Force download option bypasses cache checks

## Error Handling and Fallbacks

### Graceful Degradation
- If Google API libraries are not installed, integration is automatically disabled
- If authentication fails, operations fall back to local files
- If network is unavailable, cached files are used
- If specific Google Drive file is unavailable, local database is used

### Error Messages
The system provides clear error messages for common issues:
- Missing credentials file
- Authentication failures
- Network connectivity issues
- File not found on Google Drive
- Invalid configuration

## Security Considerations

### Authentication
- Uses OAuth 2.0 for secure authentication
- Tokens are stored locally and refreshed automatically
- Only requires read-only access to Google Drive

### Data Privacy
- Files are cached locally in specified directory
- No data is sent to external services except Google Drive API calls
- Cache can be encrypted using system-level encryption

## Performance Considerations

### Caching Strategy
- First access downloads and caches file locally
- Subsequent access uses cached version if current
- Cache checks are fast (metadata comparison)
- Large files benefit significantly from caching

### Network Usage
- Only downloads files when necessary (new or changed)
- Supports partial downloads for updated files
- Batch operations for multiple file sync
- Progress indicators for large downloads

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure credentials.json is in project root
   - Check Google Cloud Console project settings
   - Verify Google Drive API is enabled

2. **File Not Found**
   - Verify file IDs in configuration
   - Check folder permissions in Google Drive
   - Ensure files are actually .h5 format

3. **Cache Issues**
   - Clear cache and force re-download
   - Check cache directory permissions
   - Verify sufficient disk space

4. **Network Issues**
   - Check internet connectivity
   - Verify firewall settings
   - Use cached files if available

### Debug Information
Enable debug logging to see detailed operation information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Complete Integration Example
```python
#!/usr/bin/env python3
"""Complete example of Google Drive integration with TableClient."""

import toml
from data.data_tables import TableClient

def main():
    # Load configuration
    try:
        with open('config/gdrive_config.toml', 'r') as f:
            config = toml.load(f)
        gdrive_config = config['google_drive']
    except FileNotFoundError:
        print("Configuration file not found. Run setup_gdrive.py first.")
        return
    
    # Create client with Google Drive integration
    client = TableClient(
        client=None,
        data_folder='./data',
        db_file_name='nass_data.h5',
        key_prefix='nass',
        gdrive_config=gdrive_config
    )
    
    # Check status
    status = client.get_gdrive_status()
    print(f"Google Drive enabled: {status['enabled']}")
    
    if status['enabled']:
        # List available files
        print("Available Google Drive files:")
        files = client.list_gdrive_files()
        
        # Sync data from Google Drive
        print("Syncing data from Google Drive...")
        client.sync_from_gdrive()
        
        # Access data
        print("Available data keys:")
        keys = client.available_keys()
        for key in keys:
            data = client.get_key(key)
            print(f"  {key}: {data.shape}")
    
    else:
        print("Google Drive integration not available, using local files only")
        keys = client.available_keys()
        for key in keys:
            data = client.get_key(key)
            print(f"  {key}: {data.shape}")

if __name__ == '__main__':
    main()
```

## API Reference

### GoogleDriveClient Class
- `__init__(credentials_file, token_file, cache_dir)`: Initialize client
- `list_files(folder_id, file_extension)`: List files in Google Drive
- `download_file(file_id, local_path, force_download)`: Download file
- `clear_cache(older_than_days)`: Clear cached files

### GoogleDriveTableClient Class  
- `__init__(gdrive_config)`: Initialize with configuration
- `download_h5_file(file_id, force_download)`: Download .h5 file
- `get_available_files()`: List available .h5 files
- `load_h5_data(file_identifier, key, force_download)`: Load data from file
- `get_h5_keys(file_identifier)`: Get available keys from file

### TableClient Extensions
- `sync_from_gdrive(file_identifier, force_download)`: Sync files
- `get_key_from_gdrive(key, file_identifier, use_simple_name, force_download)`: Get data from Google Drive
- `list_gdrive_files()`: List Google Drive files
- `get_gdrive_keys(file_identifier)`: Get keys from Google Drive file
- `clear_gdrive_cache(older_than_days)`: Clear cache
- `get_gdrive_status()`: Get integration status

## License and Support

This Google Drive integration is part of the commodities dashboard project and follows the same licensing terms.

For support or issues:
1. Check the troubleshooting section above
2. Review error messages and logs
3. Verify configuration and authentication setup
4. Test with the included test scripts