# Copyright 2024 plumiume.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from itertools import chain
from typing import Sized, Iterable, NamedTuple
if sys.version_info.minor < 11:
    from typing_extensions import Self
else:
    from typing import Self
from dataclasses import dataclass
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from lightning import LightningDataModule

def inf_to_nan(x: list[Tensor]):
    for idx in range(len(x)):
        x[idx] = torch.where(torch.isfinite(x[idx]), x[idx], torch.nan)

def standard_scale(x: list[Tensor]) -> StandardScaler:
    ss = StandardScaler().fit(
        torch.cat(x, dim=0)
    )
    for idx in range(len(x)):
        x[idx] = torch.from_numpy(ss.transform(x[idx]))
    return ss

def label_encode(y: list[list[str]], clss: list[str] | None = None, blank: int = 0) -> tuple[list[Tensor], LabelEncoder]:
    le = LabelEncoder().fit(
        [blank] + list(set(chain.from_iterable(y)))
        if clss is None else
        clss 
    )
    new = [
        torch.from_numpy(le.transform(yi))
        for yi in y
    ]
    return new, le

def maxlen(x: Iterable[Sized], default: int = 0):
    return max((len(xi) for xi in x), default=default)

def pad_data(x: list[Tensor], maxlen: int | None = None, dim: int = 0, padding_value: torch.NumberType = 0) -> tuple[Tensor, Tensor]:

    xlen = torch.tensor([len(xi) for xi in x])

    if maxlen is None:
        maxlen = xlen.max().item()

    x = torch.stack([
        torch.nn.functional.pad(
            input=xi,
            pad=(0, 0) * (xi.ndim - dim - 1) + (0, max(0, maxlen - xleni)),
            mode='constant',
            value=padding_value
        )
        for xi, xleni in zip(x, xlen)
    ], dim=0)

    return x, xlen

def remove_nan(x: Tensor, value: torch.NumberType = 0) -> Tensor:
    return torch.masked_fill(x, torch.isnan(x), value)

class _SizedAndIterableDataset(Sized, Iterable):
    file: str | None = None
    inputs: Sized
    def __len__(self) -> int:
        return len(self.inputs)
    def __iter__(self):
        for k in self.__dict__.keys():
            yield getattr(self, k)
    def save(self, file: str):
        pickle.dump(self, open(file, 'wb'))
    @classmethod
    def load(cls, file: str) -> Self:
        self: Self = pickle.load(open(file, 'rb'))
        self.file = file
        return self

@dataclass
class RawDataset(_SizedAndIterableDataset):
    inputs: list[Tensor]
    glosses: list[list[str]]

@dataclass
class FormattedDataset(_SizedAndIterableDataset):
    inputs: list[Tensor]
    standard_scaler: StandardScaler
    labels: list[Tensor]
    label_encoder: LabelEncoder
    @property
    def num_features(self):
        return self.standard_scaler.n_features_in_
    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)

@dataclass
class ReadyDataset(_SizedAndIterableDataset):
    inputs: Tensor
    input_lengths: Tensor
    standard_scaler: StandardScaler
    labels: Tensor
    label_lengths: Tensor
    label_encoder: LabelEncoder
    @property
    def num_features(self):
        return self.standard_scaler.n_features_in_
    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)

class ReadyBatch(NamedTuple):
    inputs: Tensor
    input_lengths: Tensor
    labels: Tensor
    label_lengths: Tensor
    def to(self, device: torch.device | int | str | None = None, dtype: torch.dtype | str | None = None, non_blocking: bool = False, copy: bool = False):
        return ReadyBatch(*(field.to(device, dtype, non_blocking, copy) for field in self))

class DataModule(LightningDataModule):
    def __init__(
        self,
        formatted_dataset: FormattedDataset,
        train_indices: list[int] | Tensor | None = None,
        test_indices: list[int] | Tensor = [],
        val_indices: list[int] | Tensor = [],
        batch_size: int = 1
        ):
        super().__init__()

        self.formatted_dataset = formatted_dataset
        self.input_maxlen = maxlen(formatted_dataset.inputs)
        self.label_maxlen = maxlen(formatted_dataset.labels)

        self.train_indices = (
            torch.arange(len(self.formatted_dataset))
            if train_indices is None else
            train_indices
            if isinstance(train_indices, Tensor) else
            torch.tensor(train_indices)
        )
        self.test_indices = (
            test_indices
            if isinstance(test_indices, Tensor) else
            torch.tensor(test_indices, dtype=self.train_indices.dtype)
        )
        self.val_indices = (
            val_indices
            if isinstance(val_indices, Tensor) else
            torch.tensor(val_indices, dtype=self.train_indices.dtype)
        )
        self.batch_size = batch_size

    def collate_fn(self, indices: Tensor):
        inputs, input_lengths = pad_data(
            [self.formatted_dataset.inputs[idx] for idx in indices],
            maxlen=self.input_maxlen
        )
        labels, label_lengths = pad_data(
            [self.formatted_dataset.labels[idx] for idx in indices],
            maxlen=self.label_maxlen
        )
        return ReadyBatch(
            inputs=inputs,
            input_lengths=input_lengths,
            labels=labels,
            label_lengths=label_lengths,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, ReadyBatch):
            return batch.to(device=device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def train_dataloader(self):
        return DataLoader[ReadyBatch](
            self.train_indices, batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True, drop_last=True
        )
    def test_dataloader(self):
        return DataLoader[ReadyBatch](
            self.test_indices, batch_size=self.batch_size, 
            collate_fn=self.collate_fn
        )
    def val_dataloader(self):
        return DataLoader[ReadyBatch](
            self.val_indices, batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
