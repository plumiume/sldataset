from pathlib import Path
import importlib
import argparse
from types import ModuleType

parser = argparse.ArgumentParser()
class Namespace:
    module: str = parser.add_argument('module', choices=[p.stem for p in Path(__file__).parent.iterdir()])
    root: Path = parser.add_argument('root', type=Path)
    target: str = parser.add_argument('target')
    dst: Path = parser.add_argument('dst', type=Path)
    pattern: str = parser.add_argument('-p', '--pattern', default='**/*')
    annotation: str = parser.add_argument('-a', '--annotation', default=None)
args: Namespace = parser.parse_args()

class Module(ModuleType):
    def main(): ...

module: Module = importlib.import_module(f'sldataset.{args.module}')
module.main(args)
