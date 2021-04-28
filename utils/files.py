import pickle
from pathlib import Path
from typing import Union


def get_files(path: Union[str, Path], extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    # print(list(path.rglob(f'*{extension}')))
    return list(path.rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]):
    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]):
    with open(str(file), 'rb') as f:
        return pickle.load(f)
