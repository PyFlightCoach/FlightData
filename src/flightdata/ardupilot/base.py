from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import geometry as g
import numpy as np
import numpy.typing as npt

type BinFunc[Out] = Callable[[float | npt.NDArray[np.float64]], Out]


@dataclass
class Field[T: (g.Point, g.Quaternion)]:
    t: npt.NDArray[np.float64]
    data: T

    def slice(self, start: float, end: float) -> Field[T]:
        istart = max(0, np.searchsorted(self.t, start, side="left") - 1)
        iend = min(len(self.t), np.searchsorted(self.t, end, side="right") + 1)
        return Field(self.t[istart:iend], self.data[istart:iend])

    def to_dict(self) -> dict[str, dict]:
        return {"t": self.t, "data": self.data.to_dict()}

    @cached_property
    def freq(self) -> float:
        """Estimate the frequency of the data in Hz"""
        dt = np.diff(self.t)
        dt = dt[dt > 0]

        if len(dt) == 0:
            return np.nan

        return 1 / np.median(dt)