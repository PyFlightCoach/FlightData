from __future__ import annotations
from dataclasses import dataclass
from .labelgroup import LabelGroup


@dataclass
class Slicer:
    labels: LabelGroup
    data: Table

    def __getattr__(self, name):
        label = self.labels[name]
        res = self.data[label.start : label.stop]
        if len(label.sublabels) > 0:
            res = res.label(label.sublabels)
        return res

    def __getitem__(self, name):
        return self.__getattr__(name)

    @property
    def value(self):
        return self.labels.active(self.data.t[0])
    
    def __iter__(self):
        for k in self.labels.keys():
            yield self[k]


from .table import Table