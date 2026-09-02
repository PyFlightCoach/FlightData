from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .labelgroup import LabelGroup


@dataclass
class Slicer:
    group: str
    labels: LabelGroup
    data: Table

    def __getattr__(self, name: str) -> Table:
        return self.extract(name)

    def __getitem__(self, name):
        return self.extract(name)

    def __contains__(self, name):
        return name in self.labels

    def extract(self, name: str | Iterable[str]):
        """
        TODO check contents of iterable are contiguous and handle non-contiguous case
        """
        if hasattr(name, "__iter__") and not isinstance(name, str):
            name = list(name)
            start = self.labels[name[0]].start
            stop = self.labels[name[-1]].stop
        else:
            start = self.labels[name].start
            stop = self.labels[name].stop

        return self.data[start:stop]

    @property
    def value(self):
        return self.labels.active(self.data.t[0])

    def __iter__(self):
        for k in self.labels:
            yield self[k]

    def items(self):
        for k in self.labels:
            yield k, self[k]


from .table import Table
