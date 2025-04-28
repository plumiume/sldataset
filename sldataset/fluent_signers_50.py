from pathlib import Path
import argparse
from typing import TypedDict
import csv
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

import parse
import numpy as np
import torch
from torch import Tensor

import sldataset

class Namespace:
    module: str
    root: Path
    target: str
    dst: Path
    pattern: str
    annotation: str | None

def main(args: Namespace):

    if args.annotation is None:
        raise ValueError

    if not args.root.exists():
        raise FileNotFoundError(f'root directory ({args.root}) is not exists')
    if not args.root.is_dir():
        raise FileNotFoundError(f'root ({args.root}) must be directory')
    annotation_csv = args.root / args.annotation
    logging.info(f'annotation csv file: {annotation_csv}')
    
    if not annotation_csv.exists():
        raise FileNotFoundError(f'annotation csv file ({annotation_csv}) is not exists')
    target_dir = args.root / args.target
    logging.info(f'target directory: {target_dir}')
    if not target_dir.exists():
        raise FileNotFoundError(f'target directory ({target_dir}) is not exists')

    sentence_map = {
        int(record['ID']): str(record['Gloss']).split()
        for record in csv.DictReader(open(annotation_csv, encoding='utf8'))
    }
    logging.info(f'sentence map: {sentence_map}')

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

            logging.info(f'loaded file: {file} ({file.suffix}), shape: {x.shape}, dtype: {x.dtype}')

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

        sldataset.inf_to_nan(raw_dataset.inputs)

        logging.info("Starting standard scaling of raw dataset inputs.")
        standard_scaler = sldataset.standard_scale(raw_dataset.inputs)
        logging.info("Finished standard scaling of raw dataset inputs.")

        logging.info("Starting standard scaling of raw dataset inputs.")
        labels, label_encoder = sldataset.label_encode(
            raw_dataset.glosses,
            list(chain.from_iterable(sentence_map.values()))
        )
        logging.info("Finished standard scaling of raw dataset inputs.")

        formatted_dataset = FS50FormattedDataest(
            raw_dataset.inputs, standard_scaler,
            labels, label_encoder,
            people, valiations
        )

        formatted_dataset.save(args.dst)
        logging.info(f'saved formatted dataset: {args.dst}')
        return

@dataclass
class FS50DatasetAnnotations:
    people: list[int]
    valiations: list[int]
    def split(
        self,
        train_test_threashould: float = 0.8,
        test_val_threshould: float = 0.9
    ):
        num_people = len(set(self.people))
        order = torch.rand([num_people]).argsort() / num_people

        return (
            torch.tensor([
                i for i, p in enumerate(self.people)
                if order[p] < train_test_threashould
            ]),
            torch.tensor([
                i for i, p in enumerate(self.people)
                if train_test_threashould <= order[p] < test_val_threshould
            ]),
            torch.tensor([
                i for i, p in enumerate(self.people)
                if test_val_threshould <= order[p]
            ])
        )
    def validation(
        self,
        num_split: int = 5
        ):

        num_people = len(set(self.people))
        group = torch.rand([num_people]).argsort() % num_split

        return (
            (
                torch.tensor([
                    i for i, p in enumerate(self.people) if group[p] != idx
                ]),
                torch.tensor([
                    i for i, p in enumerate(self.people) if group[p] == idx
                ])
            )
            for idx in torch.arange(num_split)
        )


@dataclass
class FS50FormattedDataest(FS50DatasetAnnotations, sldataset.FormattedDataset):
    pass

@dataclass
class FS50ReadyDataset(FS50DatasetAnnotations, sldataset.ReadyDataset):
    pass

if __name__ == '__main__':
    main()
