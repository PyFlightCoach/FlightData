from __future__ import annotations
from dataclasses import dataclass
from .label import Label
from .labelgroup import LabelGroup
from typing import Iterable

@dataclass
class Slicer:
    group: str
    labels: LabelGroup
    data: Table

    def __getattr__(self, name: str) -> Table:
        return self.extract(name)

    def __getitem__(self, name):
        return self.extract(name)

    def extract(self, name: str | Iterable[str]):
        if hasattr(name, "__iter__") and not isinstance(name, str):
            return self.data.__class__.stack([self[n] for n in name])

        label = self.labels[name]
        start = label.start if isinstance(label, Label) else label[0].start
        stop = label.stop if isinstance(label, Label) else label[1].stop
        res = self.data[start : stop]

        if isinstance(label, Label) and len(label.sublabels) > 0:
            res = res.label(label.sublabels)  

        return res
#        return self.data.__class__.stack({n: self[n] for n in names}, self.group)

    @property
    def value(self):
        return self.labels.active(self.data.t[0])
    
    def __iter__(self):
        for k in self.labels.keys():
            yield self[k]

    def items(self):
        for k in self.labels.keys():
            yield k, self[k]

from .table import Table  # noqa: E402
