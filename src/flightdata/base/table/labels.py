from __future__ import annotations
import numpy as np
import pandas as pd
import numpy.typing as npt
from geometry import Time
from numbers import Number
from dataclasses import field, dataclass
from typing import Annotated, Literal
from geometry.utils import get_index, get_value


@dataclass
class Label:
    start: float | None = None
    stop: float | None = None

    def intersects(self, t: Table) -> bool:
        """Check if this label intersects the table"""
        start_before_last = self.start is None or self.start <= t.t[-1]
        stop_after_first = self.stop is None or self.stop > t.t[0]
        return start_before_last and stop_after_first

    def contains(self, t: npt.NDArray | Number | list[Number]) -> npt.NDArray:
        if isinstance(t, Number):
            t = [t]
        t = np.array(t)
        res = np.full(t.shape, True)
        if self.start is not None:
            res[t < self.start] = False
        if self.stop is not None:
            res[t >= self.stop] = False
        return res
    
    def to_iloc(self, t: npt.NDArray):
        return Label(
            None if self.start is None else get_index(t, self.start),
            None if self.stop is None else get_index(t, self.stop),
        )

    def to_t(self, t: npt.NDArray):
        return Label(
            None if self.start is None else get_value(t, self.start),
            None if self.stop is None else get_value(t, self.stop),
        )

    def transfer(
        self,
        a: npt.NDArray,
        b: npt.NDArray,
        path: Annotated[npt.NDArray[np.integer], Literal["N", 2]],
    ):
        # get the location in a
        a_iloc = self.to_iloc(a)

        # get the location in the path array  
        path_iloc = a_iloc.to_iloc(path[:,0])

        #get the location in b
        b_iloc = path_iloc.to_t(path[:,1])
        
        # get the time in b
        b_t = b_iloc.to_t(b)

        return b_t

    def __eq__(self, other: Label):
        return self.start == other.start and self.stop == other.stop

@dataclass
class LabelGroup:
    labels: dict[str, Label] = field(default_factory=lambda: {})

    def __iter__(self):
        for v in self.labels.values():
            yield v

    def __len__(self):
        return len(self.labels)

    def __getattr__(self, name):
        return self.labels[name]

    def __getitem__(self, k):
        return self.labels[k]

    def intersect(self, t: Table):
        return LabelGroup({k: v for k, v in self.labels.items() if v.intersects(t)})

    @property
    def empty(self):
        return len(self) == 0

    def __repr__(self):
        return f"LabelGroup({list(self.labels.keys())})"

    @staticmethod
    def read_series(data: pd.Series):
        labels = {}
        for label_name in data.unique():
            start = data.where(data == label_name).first_valid_index()
            stop = data.where(data == label_name).last_valid_index()
            if stop == data.index[-1]:
                stop = None
            else:
                stop = data.index[data.index.get_loc(stop) + 1]

            labels[label_name] = Label(start, stop)
        return LabelGroup(labels)

    @staticmethod
    def read_array(t: Time, labels: npt.NDArray):
        if len(labels.shape) > 1:
            raise ValueError("Label data must be 1D")
        return LabelGroup.read_series(pd.Series(labels, index=t.t))

    def is_tesselated(self, t: Table | None = None):
        """Check if the labels are tesselated and ordered.
        If data is passed then also check that it is covered"""
        if not np.array(set(self.labels.keys())) == np.array(self.labels.keys()):
            return False
        lvs = list(self.labels.values())

        if t:
            if not (lvs[0].start is None or lvs[0].start <= t.t[0]):
                return False
            if not (lvs[-1].stop is None or lvs[-1].stop >= t.t[-1] + t.dt[-1]):
                return False

        for v0, v1 in zip(lvs[:-1], lvs[1:]):
            if v0.stop != v1.start:
                return False
        return True

    def active(self, t: float):
        return {k: v.contains(t) for k, v in self.labels.items()}

    def dump_array(self, data: Table):
        assert self.is_tesselated(data)
        return np.concatenate(
            [np.full(sum(v.contains(data.t)), k) for k, v in self.labels.items()]
        )

    def scale(self, factor: float):
        return LabelGroup(
            {k: Label(v.start * factor, v.stop * factor) for k, v in self.labels.items()}
        )
    
    def offset(self, offset: float):
        return LabelGroup(
            {k: Label(v.start + offset, v.stop + offset) for k, v in self.labels.items()}
        )

    def transfer(
        self,
        a: Table,
        b: Table,
        path: Annotated[npt.NDArray[np.integer], Literal["N", 2]] | None,
    ):
        if path is None:
            return self.offset(-a.t[0]).scale(b.duration / a.duration).offset(b.t[0])
        
        return LabelGroup(
            {k: v.transfer(a, b, path) for k, v in self.labels.items()}
        )


@dataclass
class Slicer:
    labels: LabelGroup
    data: Table

    def __getattr__(self, name):
        label = self.labels[name]
        return self.data[label.start : label.stop]

    def __getitem__(self, name):
        return self.__getattr__(name)

    @property
    def value(self):
        return self.labels.active(self.data.t[0])


from .table import Table
