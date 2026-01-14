import pprint
from pathlib import Path
from ..utils import io


def func(args):
    config_path = Path(args.file).resolve()

    config_data = io.load_yaml(config_path)

    pprint.pprint(config_data, indent=2, sort_dicts=False)