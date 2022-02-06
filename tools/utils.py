import json
from os import path
from typing import List

CONFIG_PATH = 'config'
CLASSES_PATH = 'tools/video/classes/coco.names'


def read_config(filename: str = 'config.json'):
    try:
        with open(path.join(CONFIG_PATH, filename)) as f:
            config = json.load(f)
        return config
    except:
        return None


def save_config(config: List[dict], filename: str) -> None:
    try:
        with open(path.join(CONFIG_PATH, filename), 'w') as f:
            json.dump(config, f)
    except:
        print(f'Error saving config file:{path.join(CONFIG_PATH, filename)}')


CONFIG = read_config()


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


CLASSES = read_class_names(CLASSES_PATH)