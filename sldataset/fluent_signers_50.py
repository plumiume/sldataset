from pathlib import Path
import argparse
from typing import TypedDict
import csv
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import parse
import numpy as np
import torch
from torch import Tensor

import sldataset

def main():

    parser = argparse.ArgumentParser()
    class Namespace:
        root: Path = parser.add_argument('root', type=Path)
        target: str = parser.add_argument('target')
        annotation: str = parser.add_argument('annotation')
        intermediate: Path | None = parser.add_argument('-i', '--intermediate', default=None, type=Path)
        pattern: str = parser.add_argument('-p', '--pattern', default='**/*')
    args: Namespace = parser.parse_args()


    if not args.root.exists():
        raise FileNotFoundError(f'root directory ({args.root}) is not exists')
    if not args.root.is_dir():
        raise FileNotFoundError(f'root ({args.root}) must be directory')
    annotation_csv = args.root / args.annotation
    if not annotation_csv.exists():
        raise FileNotFoundError(f'annotation csv file ({annotation_csv}) is not exists')
    target_dir = args.root / args.target
    if not target_dir.exists():
        raise FileNotFoundError(f'target directory ({target_dir}) is not exists')

    sentence_map = {
        int(record['ID']): str(record['Gloss']).split()
        for record in csv.DictReader(open(annotation_csv, encoding='utf8'))
    }

    if target_dir.is_file():
        match target_dir.suffix:
            case '.pkl':
                formatted_dataset = sldataset.FormattedDataset.load(target_dir)
            case _:
                raise TypeError(f'invalid file type ({target_dir.suffix})')
    else:

        raw_dataset = sldataset.RawDataset([], [])

        def fn(file: Path) -> tuple[Tensor, list[str], int, int]:

            if file.is_dir():
                return

            parse_result: parse.Result | None = parse.parse('P{:d}_S{:d}_{:d}', file.stem)
            if not isinstance(parse_result, parse.Result):
                return

            person_id, sentence_id, valiation = parse_result.fixed
            
            match file.suffix:
                case '.pt':
                    x: Tensor = torch.load(file)
                case '.npy':
                    x = torch.from_numpy(np.load(file))
                case '.csv':
                    x = torch.from_numpy(np.genfromtxt(file, dtype=np.float32, delimiter=',', encoding='utf8'))
                case _:
                    raise TypeError(f'invalid file type ({file})')

            return x, sentence_map[sentence_id], person_id, valiation

        people = list[int]()
        valiations = list[int]()

        with ThreadPoolExecutor(8) as pool:
            input_gloss_tuple_list = pool.map(fn, target_dir.glob(args.pattern))
        for igpv in input_gloss_tuple_list:
            if igpv is None: continue
            i, g, p, v = igpv
            raw_dataset.inputs.append(i)
            raw_dataset.glosses.append(g)
            people.append(p)
            valiations.append(v)

        standard_scaler = sldataset.standard_scale(raw_dataset.inputs)
        labels, label_encoder = sldataset.label_encode(
            raw_dataset.glosses,
            list(chain.from_iterable(sentence_map.values()))
        )

        formatted_dataset = sldataset.FormattedDataset(
            raw_dataset.inputs, standard_scaler,
            labels, label_encoder,
            {
                'people': people,
                'valiations': valiations
            }
        )

        if args.intermediate is not None:
            formatted_dataset.save(args.intermediate)

        return formatted_dataset

@dataclass
class FS50DatasetAnnotations:
    people: list[int]
    valiations: list[int]
    def split(self, train_size: float = 0.8, test_size: float = 0.1, val_size: float = 0.1):
        num_people = len(set(self.people))
        order = torch.rand((num_people,)).argsort(0) / num_people
        return (
            order[order < (r := train_size)],
            order[r <= order < (r := r + test_size)],
            order[r <= order < (r + val_size)]
        )

@dataclass
class FS50FormattedDataest(sldataset.FormattedDataset, FS50DatasetAnnotations):
    pass

@dataclass
class FS50ReadyDataset(sldataset.ReadyDataset, FS50DatasetAnnotations):
    pass

if __name__ == '__main__':
    main()
