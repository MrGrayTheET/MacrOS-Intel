# diagnose.py - Run this to check your setup
import sys
import os
from pathlib import Path

print("=== Dash Project Diagnostic Tool ===\n")

# Check Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}\n")

# Check current directory
print(f"Current directory: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}\n")

# Check for required packages
required_packages = [
    'dash', 'plotly', 'pandas', 'numpy', 'dash_bootstrap_components',
    'python-dotenv', 'requests', 'toml'
]

print("Checking installed packages:")
missing_packages = []
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\nInstall missing packages with:")
    print(f"pip install {' '.join(missing_packages)}")

# Check project structure
print("\n=== Checking Project Structure ===")
required_dirs = [
    'data', 'components', 'layouts', 'pages', 'assets',
    'data/sources', 'data/sources/eia', 'data/sources/usda',
    'components/plotting', 'pages/energy', 'pages/agricultural'
]

missing_dirs = []
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"✓ {dir_path}/")
    else:
        print(f"✗ {dir_path}/ - MISSING")
        missing_dirs.append(dir_path)

# Check for __init__.py files
print("\n=== Checking __init__.py Files ===")
init_locations = [
    'data', 'components', 'layouts', 'pages',
    'components/plotting', 'data/sources'
]

for location in init_locations:
    init_file = Path(location) / '__init__.py'
    if init_file.exists():
        print(f"✓ {init_file}")
    else:
        print(f"✗ {init_file} - MISSING")

# Check environment variables
print("\n=== Checking Environment Variables ===")
from dotenv import load_dotenv
load_dotenv()

env_vars = ['data_path', 'market_data_path', 'cot_path', 'APP_PATH',
            'NASS_TOKEN', 'FAS_TOKEN', 'EIA_API_KEY']

for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"✓ {var} = {value[:20]}..." if len(str(value)) > 20 else f"✓ {var} = {value}")
    else:
        print(f"✗ {var} - NOT SET")

# Check for data files
print("\n=== Checking Data Files ===")
data_files = [
    'data/sources/eia/data_mapping.toml',
    'data/sources/usda/data_mapping.toml',
    'components/plotting/chart_mappings.toml'
]

for file_path in data_files:
    if Path(file_path).exists():
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path} - MISSING")

# Try importing main modules
print("\n=== Testing Imports ===")
test_imports = [
    ('data.data_tables', 'TableClient'),
    ('components.frames', 'FundamentalFrame'),
    ('utils', 'key_to_name')
]

for module_name, class_name in test_imports:
    try:
        module = __import__(module_name, fromlist=[class_name])
        if hasattr(module, class_name):
            print(f"✓ Can import {class_name} from {module_name}")
        else:
            print(f"✗ {class_name} not found in {module_name}")
    except Exception as e:
        print(f"✗ Cannot import from {module_name}: {str(e)}")

print("\n=== Diagnostic Complete ===")
print("\nNext steps:")
print("1. Fix any missing directories or files")
print("2. Install missing packages")
print("3. Create missing __init__.py files")
print("4. Update import statements to match your structure")
print("5. Set required environment variables in .env file")