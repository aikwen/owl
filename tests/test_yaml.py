import pathlib
import pprint

from owl.utils import io


if __name__ == '__main__':
    p = pathlib.Path(__file__).parent / "train.yaml"
    print(p.exists())
    y = io.load_yaml(p)
    pprint.pprint(y)