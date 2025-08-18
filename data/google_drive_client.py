"""
Google Drive integration for downloading and caching .h5 files remotely.

This module provides functionality to:
- Authenticate with Google Drive API
- Download .h5 files from Google Drive
- Cache files locally for performance
- Handle file versioning and updates
"""

import os
import io
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd

try:
    # Google Drive API imports
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    HAS_GOOGLE_API = True
except ImportError:
    print("Google API libraries not installed. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    HAS_GOOGLE_API = False

# If modifying these scopes, delete the token file
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class GoogleDriveClient:
    """Client for accessing .h5 files from Google Drive."""
    
    def __init__(self, credentials_file: str = 'credentials.json', 
                 token_file: str = 'token.json',
                 cache_dir: str = './gdrive_cache'):
        """
        Initialize Google Drive client.
        
        Args:
            credentials_file: Path to Google OAuth2 credentials JSON file
            token_file: Path to store authentication token
            cache_dir: Directory to cache downloaded files
        """
        if not HAS_GOOGLE_API:
            raise ImportError("Google API libraries required. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.service = None
        self.file_cache = {}  # Cache file metadata
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API."""
        creds = None
        
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Google Drive credentials file not found: {self.credentials_file}. "
                        "Download from Google Cloud Console and place in project root."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
    
    def list_files(self, folder_id: Optional[str] = None, 
                   file_extension: str = '.h5') -> List[Dict]:
        """
        List .h5 files in Google Drive folder.
        
        Args:
            folder_id: Google Drive folder ID (None for root)
            file_extension: File extension to filter by
            
        Returns:
            List of file metadata dictionaries
        """
        if not self.service:
            raise RuntimeError("Not authenticated with Google Drive")
        
        # Build query
        query = f"name contains '{file_extension}' and trashed = false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        
        results = self.service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, size, modifiedTime, md5Checksum)"
        ).execute()
        
        files = results.get('files', [])
        
        # Update file cache
        for file_info in files:
            self.file_cache[file_info['id']] = file_info
        
        return files
    
    def download_file(self, file_id: str, local_path: Optional[str] = None,
                     force_download: bool = False) -> str:
        """
        Download file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            local_path: Local path to save file (None for auto-generated)
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded file
        """
        if not self.service:
            raise RuntimeError("Not authenticated with Google Drive")
        
        # Get file metadata
        file_metadata = self.service.files().get(
            fileId=file_id,
            fields="id, name, size, modifiedTime, md5Checksum"
        ).execute()
        
        # Determine local path
        if local_path is None:
            local_path = self.cache_dir / file_metadata['name']
        else:
            local_path = Path(local_path)
        
        # Check if file needs to be downloaded
        if not force_download and self._is_file_cached(file_metadata, local_path):
            print(f"Using cached file: {local_path}")
            return str(local_path)
        
        print(f"Downloading {file_metadata['name']} ({file_metadata.get('size', 'unknown')} bytes)...")
        
        # Download file
        request = self.service.files().get_media(fileId=file_id)
        
        with io.BytesIO() as file_buffer:
            downloader = MediaIoBaseDownload(file_buffer, request)
            done = False
            
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download progress: {int(status.progress() * 100)}%")
            
            # Write to local file
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(file_buffer.getvalue())
        
        # Update cache metadata
        self._update_cache_metadata(file_metadata, local_path)
        
        print(f"Downloaded to: {local_path}")
        return str(local_path)
    
    def _is_file_cached(self, file_metadata: Dict, local_path: Path) -> bool:
        """Check if file is already cached and up-to-date."""
        if not local_path.exists():
            return False
        
        # Check if we have cached metadata
        metadata_path = self._get_metadata_path(local_path)
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                cached_metadata = json.load(f)
            
            # Compare modification times and checksums
            return (
                cached_metadata.get('modifiedTime') == file_metadata.get('modifiedTime') and
                cached_metadata.get('md5Checksum') == file_metadata.get('md5Checksum')
            )
        except Exception as e:
            print(f"Error reading cache metadata: {e}")
            return False
    
    def _update_cache_metadata(self, file_metadata: Dict, local_path: Path):
        """Update cache metadata file."""
        metadata_path = self._get_metadata_path(local_path)
        
        cache_info = {
            'file_id': file_metadata['id'],
            'name': file_metadata['name'],
            'modifiedTime': file_metadata.get('modifiedTime'),
            'md5Checksum': file_metadata.get('md5Checksum'),
            'local_path': str(local_path),
            'cached_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(cache_info, f, indent=2)
    
    def _get_metadata_path(self, local_path: Path) -> Path:
        """Get path for metadata file."""
        return local_path.parent / f".{local_path.name}.metadata"
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached files.
        
        Args:
            older_than_days: Only clear files older than this many days (None for all)
        """
        if older_than_days is None:
            # Remove all cache files
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
            print("Cleared all cached files")
        else:
            # Remove files older than specified days
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_path.unlink()
                        print(f"Removed old cached file: {file_path}")
    
    def get_file_url(self, file_id: str) -> str:
        """Get shareable URL for a file."""
        return f"https://drive.google.com/file/d/{file_id}/view"


class GoogleDriveTableClient:
    """Extended TableClient with Google Drive integration."""
    
    def __init__(self, gdrive_config: Dict):
        """
        Initialize Google Drive Table Client.
        
        Args:
            gdrive_config: Configuration dictionary with:
                - folder_id: Google Drive folder ID
                - credentials_file: Path to credentials JSON
                - cache_dir: Local cache directory
                - file_mappings: Dict mapping logical names to file IDs
        """
        self.config = gdrive_config
        self.gdrive_client = GoogleDriveClient(
            credentials_file=gdrive_config.get('credentials_file', 'credentials.json'),
            cache_dir=gdrive_config.get('cache_dir', './gdrive_cache')
        )
        self.file_mappings = gdrive_config.get('file_mappings', {})
        
    def download_h5_file(self, file_id: str, force_download: bool = False) -> str:
        """
        Download .h5 file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            force_download: Force re-download even if cached
            
        Returns:
            Local path to downloaded file
        """
        return self.gdrive_client.download_file(file_id, force_download=force_download)
    
    def get_available_files(self) -> List[Dict]:
        """Get list of available .h5 files from Google Drive."""
        folder_id = self.config.get('folder_id')
        return self.gdrive_client.list_files(folder_id=folder_id, file_extension='.h5')
    
    def load_h5_data(self, file_identifier: str, key: str, force_download: bool = False) -> pd.DataFrame:
        """
        Load data from .h5 file on Google Drive.
        
        Args:
            file_identifier: File ID or logical name from mappings
            key: HDF5 key to load from file
            force_download: Force re-download even if cached
            
        Returns:
            DataFrame with requested data
        """
        # Resolve file ID from identifier
        if file_identifier in self.file_mappings:
            file_id = self.file_mappings[file_identifier]
        else:
            file_id = file_identifier
        
        # Download file
        local_path = self.download_h5_file(file_id, force_download=force_download)
        
        # Load data from HDF5 file
        try:
            with pd.HDFStore(local_path, mode='r') as store:
                if key not in store.keys():
                    available_keys = list(store.keys())
                    raise KeyError(f"Key '{key}' not found in file. Available keys: {available_keys}")
                
                return store[key]
        except Exception as e:
            print(f"Error loading data from {local_path}, key {key}: {e}")
            return pd.DataFrame()
    
    def get_h5_keys(self, file_identifier: str) -> List[str]:
        """
        Get available keys from .h5 file on Google Drive.
        
        Args:
            file_identifier: File ID or logical name from mappings
            
        Returns:
            List of available HDF5 keys
        """
        # Resolve file ID from identifier
        if file_identifier in self.file_mappings:
            file_id = self.file_mappings[file_identifier]
        else:
            file_id = file_identifier
        
        # Download file
        local_path = self.download_h5_file(file_id)
        
        # Get keys from HDF5 file
        try:
            with pd.HDFStore(local_path, mode='r') as store:
                return list(store.keys())
        except Exception as e:
            print(f"Error accessing {local_path}: {e}")
            return []


# Utility functions for integration with existing TableClient
def create_gdrive_config_template() -> Dict:
    """Create a template configuration for Google Drive integration."""
    return {
        'folder_id': 'your_google_drive_folder_id_here',
        'credentials_file': 'credentials.json',
        'cache_dir': './gdrive_cache',
        'file_mappings': {
            # Example mappings from logical names to Google Drive file IDs
            'nass_data': 'google_drive_file_id_1',
            'esr_data': 'google_drive_file_id_2',
            'market_data': 'google_drive_file_id_3'
        }
    }


def setup_google_drive_auth():
    """
    Instructions for setting up Google Drive authentication.
    
    Returns:
        Instructions as a string
    """
    return """
    To set up Google Drive integration:
    
    1. Go to Google Cloud Console: https://console.cloud.google.com/
    2. Create a new project or select existing one
    3. Enable the Google Drive API
    4. Create credentials (OAuth 2.0 Client IDs) for a desktop application
    5. Download the credentials JSON file and save as 'credentials.json' in your project root
    6. Run the application - it will open a browser for OAuth consent
    7. Grant permissions and the token will be saved automatically
    
    Example credentials.json structure:
    {
      "installed": {
        "client_id": "your_client_id.apps.googleusercontent.com",
        "client_secret": "your_client_secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
      }
    }
    """