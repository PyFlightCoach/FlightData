from __future__ import annotations
import numpy as np
import pandas as pd
import numpy.typing as npt
from geometry import Time
from dataclasses import field, dataclass
from typing import Annotated, Literal, Callable
from .labelgroup import LabelGroup



@dataclass
class LabelGroups:
    lgs: dict[str, LabelGroup] = field(default_factory=lambda: {})

    def __eq__(self, other: LabelGroup):
        return all([v == other[k] for k, v in self.lgs.items()])

    def __dict__(self):
        return self.lgs

    def __iter__(self):
        for k, v in self.lgs.items():
            yield k, v

    def items(self):
        for k, v in self.lgs.items():
            yield k, v

    def values(self):
        for v in self.lgs.values():
            yield v

    def keys(self):
        for k in self.lgs.keys():
            yield k

    def __getitem__(self, name: str | int):
        if isinstance(name, str):
            return self.lgs[name]
        elif isinstance(name, int):
            return list(self.lgs.values())[name]
        else:
            raise ValueError(
                f"Can only index labelgroups with int or str, got {name.__class__.__name__}"
            )

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

    def intersect(self, t: Time):
        """Return a subset of the labels that intersect the table"""
        return self.update(lambda v: v.intersect(t.t[0], t.t[-1] + t.dt[-1]))

    def slice(self, tstart: float, tstop: float):
        return self.update(lambda v: v.slice(tstart, tstop))

    def scale(self, factor: float):
        return self.update(lambda l: l.scale(factor))

    def offset(self, offset: float | npt.NDArray):
        return self.update(lambda v: v.offset(offset))

    def copy(self): 
        return LabelGroups({k: v.copy() for k, v in self.items()})

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

    def to_df(self, t: npt.NDArray):
        return pd.DataFrame(
            {k: v.to_array(t) for k, v in self.items()}, index=t
        )

    def step_boundary(self, group: str, key: str, steps: int, t: npt.NDArray, min_len: int):
        return LabelGroups({
            k: v.step_boundary(key, steps, t, min_len) if k == group else v for k, v in self.items()
        })

    def set_boundary(self, group: str, key: str, new_t: float, min_duration: float):
        return LabelGroups({
            k: v.set_boundary(key, new_t, min_duration) if k == group else v for k, v in self.items()
        })

    def set_boundaries(self, group: str, boundaries: npt.NDArray):
        return LabelGroups({
            k: v.set_boundaries(boundaries) if k == group else v for k, v in self.items()
        })

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.lgs.items()}
    
    @staticmethod
    def from_dict(data: dict[str, dict]):
        return LabelGroups({k: LabelGroup.from_dict(v) for k, v in data.items()})

    def whole(self, time: Time):
        """Return only labels that span more than one timestep expanded to cover the removed labels"""
        return LabelGroups({k: LabelGroup.read_array(time, v.to_array(time.t)) for k, v in self.items()})

    
