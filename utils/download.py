import os

import requests
import pathlib


def download_data(data_dir: str, uri: str, file_name: str):
    data_path = pathlib.Path(f'Data')
    data_dir_path = data_path / data_dir
    if (data_dir_path / file_name).is_file() and data_dir_path.is_dir():
        print(f'{file_name} already downloaded')

    else:
        data_dir_path.mkdir(exist_ok=True, parents=True)
        request = requests.get(uri)
        with open(data_dir_path / file_name, 'wb') as down_file:
            down_file.write(request.content)


def download_file(uri: str, file_name: str):
    file_name = pathlib.Path(f'{file_name}')
    if not file_name.is_file():
        request = requests.get(uri)
        with open(file_name, 'wb') as down_file:
            down_file.write(request.content)
    else:
        print('file already exists')
