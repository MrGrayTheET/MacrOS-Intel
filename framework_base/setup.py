"""
Setup configuration for the Commodities Dashboard Framework (comm_dash).
"""
from setuptools import setup, find_packages
import os

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as req_file:
        return [line.strip() for line in req_file if line.strip() and not line.startswith("#")]

setup(
    name="comm_dash",
    version="1.0.0",
    author="Commodities Analytics Team",
    author_email="analytics@commodities-dashboard.com",
    description="A comprehensive Python framework for building commodity data analysis dashboards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/comm_dash",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.950",
            "flake8>=4.0",
            "pre-commit>=2.17",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
        ],
        "google": [
            "google-api-python-client>=2.0",
            "google-auth-httplib2>=0.1",
            "google-auth-oauthlib>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "comm-dash=comm_dash.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "comm_dash": [
            "data/sources/*/data_mapping.toml",
            "data/sources/*/routes.toml",
            "data/sources/*/nass_queries.toml",
            "data/sources/*/esr_map.toml",
            "data/sources/COT/futures_mappings.toml",
            "components/plotting/chart_mappings.toml",
            "*.toml",
            "*.md",
        ],
    },
    keywords=[
        "commodities", "dashboard", "agriculture", "energy", "trading", 
        "analytics", "data-visualization", "plotly", "dash", "usda", 
        "eia", "market-data", "time-series", "async", "geospatial"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/comm_dash/issues",
        "Source": "https://github.com/your-org/comm_dash",
        "Documentation": "https://comm-dash.readthedocs.io",
        "Funding": "https://github.com/sponsors/your-org",
    },
)