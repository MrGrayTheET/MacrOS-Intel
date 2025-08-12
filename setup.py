#!/usr/bin/env python3
"""Project setup utility.

Installs dependencies, configures paths and downloads required data files."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


def install_dependencies() -> None:
    """Install Python packages from requirements and gdown for downloads."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def setup_paths() -> None:
    """Create data directories and write path variables to .env."""
    data_dir = PROJECT_ROOT / "data"
    market_dir = data_dir / "market"
    cot_dir = data_dir / "cot"

    data_dir.mkdir(parents=True, exist_ok=True)
    market_dir.mkdir(parents=True, exist_ok=True)
    cot_dir.mkdir(parents=True, exist_ok=True)

    env_lines = [
        f"data_path={data_dir}",
        f"market_data_path={market_dir}",
        f"cot_path={cot_dir}",
        f"APP_PATH={PROJECT_ROOT}",
    ]
    ENV_FILE.write_text("\n".join(env_lines) + "\n")


def download_h5() -> None:
    """Download .h5 files from the project Drive folder into the data path."""
    import gdown

    url = "https://drive.google.com/drive/folders/1PM5dv-Acy7fgVPLQvOsmDxcRis7somiC?usp=drive_link"
    output = PROJECT_ROOT / "data"
    gdown.download_folder(url, output=str(output), quiet=False, use_cookies=False)


def main() -> None:
    install_dependencies()
    setup_paths()
    download_h5()


if __name__ == "__main__":
    main()
