from __future__ import annotations
import numpy as np
import pandas as pd
import numpy.typing as npt
from geometry import Time
from dataclasses import field, dataclass
from typing import Annotated, Literal, Callable
from .label import Label


@dataclass
class LabelGroup:
    """Contains a dict of labels
    They should be tesselated, so the stop of one label is the start of the next
    """
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
        return self.labels.values()

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
            raise ValueError(
                f"Can only index labelgroup with int or str, got {name.__class__.__name__}"
            )

    @property
    def empty(self):
        return len(self) == 0

    @staticmethod
    def read_array(t: npt.NDArray, labels: npt.NDArray):
        if len(labels.shape) > 1:
            raise ValueError("Label data must be 1D")
        if len(labels) == len(t):
            labels= labels[:-1]
        assert len(labels) == len(t) - 1
        labnames = pd.unique(labels)
        data = {}
        for i, label_name in enumerate(labnames):
            indeces = np.argwhere(labels == label_name)
            tstart = t[indeces[0]]
            tstop = t[indeces[-1] + 1]
            data[label_name] = Label(tstart[0], tstop[0])

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

    def is_tesselated(self, t: npt.NDArray | None = None):
        """Check if the labels are tesselated and ordered.
        If data is passed then also check that it is covered"""
        if not np.array(set(self.labels.keys())) == np.array(self.labels.keys()):
            return False
        lvs = list(self.labels.values())

        if t is not None:
            if not (lvs[0].start is None or lvs[0].start <= t[0]):
                return False
            if not (lvs[-1].stop is None or lvs[-1].stop >= t[-1]):
                return False

        for v0, v1 in zip(lvs[:-1], lvs[1:]):
            if v0.stop != v1.start:
                return False
        return True

    def active(self, t: float):
        return {k: v.contains(t) for k, v in self.labels.items()}

    def intersect(self, time: Time):
        """Return a subset of the labels that intersect the table"""
        return self.filter(lambda v: v.intersects(time.t[0], time.t[-1] + time.dt[-1]))

    def slice(self, tstart: float, tstop: float):
        return (
            self.filter(lambda v: v.intersects(tstart, tstop))
            .update(lambda v: v.slice(tstart, tstop))
            .filter(lambda v: v.is_valid)
        )

    def to_array(self, t: npt.NDArray):
        assert self.is_tesselated(t)
        return np.concatenate(
            [np.full(sum(v.contains(t, i == len(self.labels) - 1)), k) for i, (k, v) in enumerate(self.labels.items())]
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
                self.offset(-a[0]).scale((b[-1] - b[0]) / (a[-1] - a[0])).offset(b[0])
            )
        else:
            
            mans = (
                pd.DataFrame(path, columns=["a", "b"])
                .set_index("a")
                .join(pd.Series(self.to_array(a), name="a"))
                .groupby(["b"])
                .last()
                .reset_index()
                .set_index("b")
            )
            
            #st: Self = flown.__class__(flown.data).label(**mans.to_dict(orient="list"))
            return LabelGroup.read_array(b, mans.a)
#            return self.update(lambda v: v.transfer(a, b, path))

    @property
    def boundaries(self) -> dict[str, float]:
        return [v.stop for v in self.values()]

    def set_boundaries(self, stops: list[float]):
        """set the start and stop times of the labels, assumes tesselated"""
        newlabels = {}
        for i, (k, v) in enumerate(self.items()):
            start = stops[i - 1] if i > 0 else v.start
            stop = stops[i] if i < len(stops) else v.stop
            newlabels[k] = Label(start, stop)
        return LabelGroup(newlabels)

    def set_boundary(self, key: str | int, value: float, min_duration: int=0):
        """Set the stop time of a label, and the start of the next label"""
        index = list(self.keys()).index(key) if isinstance(key, str) else key
        if (
            index <= len(self) - 1 - min_duration
            and self[index].start + min_duration < value
            and self[index + 1].stop - min_duration >= value
        ):
            boundaries = self.boundaries
            boundaries[index] = value
            return self.set_boundaries(boundaries)
        else:
            raise ValueError(f"Cannot set boundary to {value} for label {key}")

    def step_boundary(self, key: str | int, steps: int, t: npt.NDArray, min_len: int):
        """Step the stop time of a label, and the start of the next label by steps timesteps"""
        ilg = self.to_iloc(t)
        iboundaries = [0] + ilg.boundaries
        lengths = [b1-b0 for b0, b1 in zip(iboundaries[:-1], iboundaries[1:])]
        index = list(self.keys()).index(key) if isinstance(key, str) else key
#        new_iloc = np.where(t==self[index].stop)[0][0] + steps
        
        
        if lengths[index] >= -steps + min_len -1  and lengths[index+1] >= steps + min_len - 1:
            
            ilg[index].stop = ilg[index].stop + steps
            if index < len(ilg) - 1:
                ilg[index+1].start = ilg[index+1].start + steps
            return ilg.to_t(t)
#            return self.set_boundary(key, t[new_iloc], 0)
        else:
            raise ValueError(f"Cannot step boundary for label {key}")

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.labels.items()}

    @staticmethod
    def from_dict(data: dict[str, dict]):
        return LabelGroup({k: Label.from_dict(v) for k, v in data.items()})

    def to_iloc(self, t: npt.NDArray):
        return self.update(lambda v: v.to_iloc(t))

    def to_t(self, t: npt.NDArray):
        return self.update(lambda v: v.to_t(t))

    def split_label(self, key: str, new_k: str, prop: float, pos: Literal["start", "end"], t: npt.NDArray, minl: int):
        """Split a label at the propotion prop (between 0 and 1). give the new label the key new_k. 
            if prop ==0, the original label will have length minl
            if prop ==1, the new label will have length minl.
            TODO handle sublabels
        """
        new_labs = {}
        for k, v in self.items():
            if k == key:
                old_ilable = self[key].to_iloc(t)
                if pos=="start":
                    new_labs[new_k] = Label(old_ilable.start, old_ilable.start+minl).to_t(t)
                    new_labs[key] = Label(old_ilable.start+minl, old_ilable.stop).to_t(t)
                elif pos=="end":
                    new_labs[key] = Label(old_ilable.start, old_ilable.stop - minl).to_t(t)
                    new_labs[new_k] = Label(old_ilable.stop-minl, old_ilable.stop).to_t(t)
            else:
                new_labs[k]=v
        return LabelGroup(new_labs)

    def insert(self, loc: int, names: list[str], t: npt.NDArray, minl: int) -> LabelGroup:
        """Insert one or more keys at the given location.
        key will be inserted into the longer of the preceding and following labels.
        raise Exception if both preceding and following labels are too short to accept new label.
        function is called recursively if more than one name provided 
        """

        if len(names) > 1:
            return self.insert(names[:-1], loc, t, minl)

        name = names[0]

        lab0 = self[loc-1] if loc>0 else None
        l0 = lab0.to_iloc(t).width if lab0 else 0
        lab1 = self[loc] if loc<len(self) else None
        l1 = lab1.to_iloc(t).width if lab1 else 0
        
        if l0 > l1 and l0 > 2*minl:
            return self.split_label(list(self.keys())[loc-1], name, 1, "end", t, minl)
        elif l1 > l0 and l1 > 2*minl:
            return self.split_label(list(self.keys())[loc], name, 0, "start", t, minl)
        else:
            raise Exception(f"Cannot insert missing label {name} here")

    def unsquash(self, keys: list[str], t: npt.NDArray, minl: int=1):
        """update self based on the provided list of keys. 
        Where missing keys are found they will be inserted with minl timesteps.
        bounding labels will be shifted to accomodate."""
        labs = list(self.keys())
    
        il = 0
        ii = 0
        inserts = []
        new_labs = {}
        while ii < len(keys):
            if labs[il] == keys[ii]:
                if len(inserts):
                    new_labs[il] = inserts                
                inserts = []
                il += 1
                ii += 1
            else:
                inserts.append(keys[ii])
                ii += 1
        for loc in list(new_labs.keys())[::-1]:
            self=self.insert(loc, new_labs[loc], t, minl)
        return self
