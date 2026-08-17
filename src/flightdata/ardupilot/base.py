from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import geometry as g
import numpy as np
import numpy.typing as npt

type BinFunc[Out] = Callable[[float | npt.NDArray[np.float64]], Out]


T = TypeVar("T", g.Point, g.Quaternion)


@dataclass
class Field[T]:
    t: npt.NDArray[np.float64]
    data: T

    def slice(self, start: float, end: float) -> Field[T]:
        istart = np.searchsorted(self.t, start, side="left")
        iend = np.searchsorted(self.t, end, side="right")
        return Field(self.t[istart:iend], self.data[istart:iend])

    def to_dict(self) -> dict[str, npt.NDArray[np.float64]]:
        return {"t": self.t, "data": self.data.to_dict()}
