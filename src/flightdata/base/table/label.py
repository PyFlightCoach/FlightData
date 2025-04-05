from __future__ import annotations
import numpy as np
import numpy.typing as npt
from numbers import Number
from dataclasses import field, dataclass
from typing import Annotated, Literal
from geometry.utils import get_index, get_value


@dataclass
class Label:
    """Indicates the range where a label is active.
    The range is inclusive
    """
    start: float
    stop: float
    sublabels: LabelGroups = field(default_factory=lambda: LabelGroups({}))

    @property
    def width(self):
        return self.stop - self.start

    def intersects(self, tstart: float, tstop) -> bool:
        """Check if this label intersects the table"""
        return self.start <= tstop and self.stop > tstart

    def contains(self, t: npt.NDArray | Number | list[Number], inclusive: bool=True) -> npt.NDArray:
        if isinstance(t, Number):
            t = [t]
        t = np.array(t)
        res = np.full(t.shape, True)
        res[t < self.start] = False
        if inclusive:
            res[t > self.stop] = False
        else:
            res[t >= self.stop] = False
        return res

    def to_iloc(self, t: npt.NDArray):
        return Label(get_index(t, self.start), get_index(t, self.stop, direction="backward"))

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

    def to_dict(self):
        return {"start": self.start, "stop": self.stop, "sublabels": self.sublabels.to_dict()}

    @staticmethod
    def from_dict(data):
        return Label(data['start'], data['stop'], LabelGroups.from_dict(data['sublabels']))

    def shift(self, t: npt.NDArray, steps: int):
        """shift the end point by steps timesteps"""
        ilab = self.to_iloc(t)
        ilab.stop += steps
        return ilab.to_t(t)
    
    def copy(self):
        return Label(self.start, self.stop, self.sublabels.copy())

from .labelgroups import LabelGroups