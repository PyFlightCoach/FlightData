from __future__ import annotations
import numpy as np
import numpy.typing as npt
from geometry import Time
from numbers import Number
from dataclasses import field, dataclass
from typing import Annotated, Literal, Callable
from geometry.utils import get_index, get_value


@dataclass
class Label:
    start: float
    stop: float
    sublabels: LabelGroups = field(default_factory=lambda: LabelGroups({}))

    def intersects(self, tstart: float, tstop) -> bool:
        """Check if this label intersects the table"""
        return self.start <= tstop and self.stop > tstart

    def contains(self, t: npt.NDArray | Number | list[Number]) -> npt.NDArray:
        if isinstance(t, Number):
            t = [t]
        t = np.array(t)
        res = np.full(t.shape, True)
        res[t < self.start] = False
        res[t >= self.stop] = False
        return res

    def to_iloc(self, t: npt.NDArray):
        return Label(get_index(t, self.start)[0], get_index(t, self.stop)[0])

    def to_t(self, t: npt.NDArray):
        return Label(get_value(t, self.start), get_value(t, self.stop))

    def slice(self, tstart: float, tstop: float):
        return Label(
            max(self.start, tstart),
            min(self.stop, tstop),
            self.sublabels.slice(tstart, tstop),
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
        path_iloc = a_iloc.to_iloc(path[:, 0])

        # get the location in b
        b_iloc = path_iloc.to_t(path[:, 1])

        # get the time in b
        b_t = b_iloc.to_t(b)

        return Label(b_t.start, b_t.stop, self.sublabels.transfer(a, b, path))

    def __eq__(self, other: Label):
        return self.start == other.start and self.stop == other.stop

    @property
    def is_valid(self):
        return self.start < self.stop


@dataclass
class LabelGroup:
    labels: dict[str, Label] = field(default_factory=lambda: {})

    def __eq__(self, other: LabelGroup):
        return all([v == other[k] for k, v in self.labels.items()])

    def __dict__(self):
        return self.labels

    def __iter__(self):
        return self.values()

    def items(self):
        return self.labels.items()

    def values(self):
        self.labels.values()

    def keys(self):
        return self.labels.keys()

    def update(self, fun: Callable[[Label], Label]):
        return LabelGroup({k: fun(v) for k, v in self.labels.items()})

    def __repr__(self):
        return f"LabelGroup({','.join([str(l) for l in self.labels.keys()])})"

    def filter(self, fun: Callable[[Label], bool]):
        return LabelGroup({k: v for k, v in self.labels.items() if fun(v)})

    def __len__(self):
        return len(self.labels)

    def __getattr__(self, name):
        return self.labels[name]

    def __getitem__(self, name: str | int):
        if isinstance(name, str):
            return self.labels[name]
        elif isinstance(name, int):
            return list(self.labels.values())[name]
        else:
            raise ValueError(f"Can only index labelgroup with int or str, got {name.__class__.__name__}")

    @property
    def empty(self):
        return len(self) == 0

    @staticmethod
    def read_array(t: Time, labels: npt.NDArray):
        if len(labels.shape) > 1:
            raise ValueError("Label data must be 1D")

        data = {}
        for label_name in np.unique(labels):
            indeces = np.argwhere(labels == label_name)
            tstart = t[indeces[0]]
            tstop = t[indeces[-1]]
            data[label_name] = Label(tstart.t[0], tstop.t[0] + tstop.dt[0])

        return LabelGroup(data)

    @staticmethod
    def concat(*args: list[LabelGroup]):
        new_labels = {}
        for lg in args:
            for k, v in lg.items():
                if k in new_labels:
                    if new_labels[k].stop == v.start:
                        new_labels[k].stop = v.stop
                    else:
                        raise ValueError(f"Labels {k} are not contiguous")
                else:
                    new_labels[k] = v

        return LabelGroup(new_labels)

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

    def intersect(self, t: Table):
        """Return a subset of the labels that intersect the table"""
        return self.filter(lambda v: v.intersects(t.t[0], t.t[-1] + t.dt[-1]))

    def slice(self, tstart: float, tstop: float):
        return (
            self.filter(lambda v: v.intersects(tstart, tstop))
            .update(lambda v: v.slice(tstart, tstop))
            .filter(lambda v: v.is_valid)
        )

    def to_array(self, data: Table):
        assert self.is_tesselated(data)
        return np.concatenate(
            [np.full(sum(v.contains(data.t)), k) for k, v in self.labels.items()]
        )

    def scale(self, factor: float):
        return self.update(
            lambda v: Label(
                v.start * factor, v.stop * factor, v.sublabels.scale(factor)
            )
        )

    def offset(self, offset: float):
        return self.update(
            lambda v: Label(
                v.start + offset, v.stop + offset, v.sublabels.offset(offset)
            )
        )

    def transfer(
        self,
        a: npt.NDArray,
        b: npt.NDArray,
        path: Annotated[npt.NDArray[np.integer], Literal["N", 2]] | None,
    ):
        if path is None:
            return (
                self.offset(-a.t[0]).scale((b[-1] - b[0]) / (a[-1] / a[0])).offset(b[0])
            )
        else:
            return self.update(lambda v: v.transfer(a, b, path))


@dataclass
class LabelGroups:
    lgs: dict[str, LabelGroup] = field(default_factory=lambda: {})

    def __eq__(self, other: LabelGroup):
        return all([v == other[k] for k, v in self.lgs.items()])

    def __dict__(self):
        return self.lgs

    def __iter__(self):
        return self.values()

    def items(self):
        return self.lgs.items()

    def values(self):
        return self.lgs.values()

    def keys(self):
        return self.lgs.keys()

    def __getitem__(self, name: str | int):
        if isinstance(name, str):
            return self.lgs[name]
        elif isinstance(name, int):
            return list(self.lgs.values())[name]
        else:
            raise ValueError(f"Can only index labelgroups with int or str, got {name.__class__.__name__}")


    def update(self, fun: Callable[[LabelGroup], LabelGroup]):
        return LabelGroups({k: fun(v) for k, v in self.items()})

    def __repr__(self):
        return f"LabelGroups({','.join([str(k) for k in self.keys()])})"

    def filter(self, fun: Callable[[LabelGroup], bool]):
        return LabelGroups({k: v for k, v in self.items() if fun(v)})

    def filter_keys(self, fun: Callable[[str], bool]):
        return LabelGroups({k: v for k, v in self.items() if fun(k)})

    def __len__(self):
        return len(self.lgs)

    def __getattr__(self, name):
        return self.lgs[name]

    def intersect(self, t: Table):
        """Return a subset of the labels that intersect the table"""
        return self.update(lambda v: v.intersect(t.t[0], t.t[-1] + t.dt[-1]))

    def slice(self, tstart: float, tstop: float):
        return self.update(lambda v: v.slice(tstart, tstop))

    def scale(self, factor: float):
        return self.update(lambda l: l.scale(factor))

    def offset(self, offset: float | npt.NDArray):
        return self.update(lambda v: v.offset(offset))

    def transfer(
        self,
        a: npt.NDArray,
        b: npt.NDArray,
        path: Annotated[npt.NDArray[np.integer], Literal["N", 2]] | None,
    ):
        return self.update(lambda v: v.transfer(a, b, path))

    @staticmethod
    def concat(*args: list[LabelGroups]):
        newlgs: dict[str, list[LabelGroup]] = {}
        for lgs in args:
            for k, v in lgs.items():
                if k not in newlgs:
                    newlgs[k] = []
                newlgs[k].append(v)
        return LabelGroups({k: LabelGroup.concat(*v) for k, v in newlgs.items()})


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


from .table import Table
