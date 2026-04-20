import os


def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)