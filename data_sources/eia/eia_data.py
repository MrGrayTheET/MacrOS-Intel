from pathlib import Path
import requests
import toml
import zipfile
from io import BytesIO
import os
from data_sources.tables import store_folder


class EIABulk:

    data_folder = Path(store_folder, 'EIABulk')
    table_db = Path(data_folder, 'eia_data.hd5')


    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    def __init__(self):
        datasets = requests.get('https://www.eia.gov/opendata/bulk/manifest.txt').json()['dataset']
        urls = {}
        for k, v in datasets.items():
            urls[k]=v['accessURL']
        print('EIA Bulk Download')
        print(f'Available Datasets : {urls.keys()}')
        self.urls = urls
        return
    def download_dataset(self, key):
        print(f'Downloading Dataset {key}')
        destination = Path(self.data_folder, key)

        if not os.path.exists(destination):
            os.mkdir(destination)

        response = requests.get(url=self.urls[key], stream=True)

        try :
            response.raise_for_status()
        except Exception as e:
            print(f'Failed to get dataset\n\nHTTP Error {e}')
            return
        else:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(destination)
        files = os.listdir(destination)
        print(f'Dataset {key} Successfully loaded to {destination}\n')
        print(f'Files available: {files}')

        return
















