import os
from dotenv import load_dotenv

load_dotenv()
store_folder = os.getenv('data_path')

if not os.path.exists(store_folder):
    os.mkdir(store_folder)



