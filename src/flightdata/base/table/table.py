from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property, partial
from numbers import Number
from time import time
from typing import Annotated, ClassVar, Literal, Self
from xmlrpc.client import boolean

import geometry as g
import numpy as np
import numpy.typing as npt
import pandas as pd
from geometry.utils import get_value

from .label import Label
from .labelgroup import LabelGroup
from .labelgroups import LabelGroups
from .slicer import Slicer

default_interpolators = {
    "Time": "linterp_recaclulate_dt",
    "Point": "linterp",
    "Quaternion": "slerp",
    "Air": "linterp",
    "Attack": "linterp",
}


class TableError(Exception):
    """Base exception for table-related errors."""


@dataclass
class Construct:
    name: str
    type: g.GBase
    cols: list[str]


@dataclass
class Table:
    """Base data structure, wraps around a pandas dataframe.
    All the columns are defined in the constructs class variable.
    A dictionary of labels is included, keys are label group names, values are instances of LabelGroup.
    """

    _construct_freq: ClassVar[float] = 25
    _constructs: ClassVar[list[Construct]] = [Construct("time", g.Time, ["t", "dt"])]
    labels: LabelGroups = field(default_factory=lambda: LabelGroups(), kw_only=True)
    time: g.Time

    @cached_property
    def columns(self) -> list[str]:
        return [col for con in self._constructs for col in con.cols]

    @cached_property
    def column_getters(self) -> dict[str, g.GBase]:
        return {
            col: partial(
                lambda name, column: getattr(self, name).data[:, column], con.name, i
            )
            for con in self._constructs
            for i, col in enumerate(con.cols)
        }

    @cached_property
    def construct_names(self) -> list[str]:
        return [con.name for con in self._constructs]

    @cached_property
    def constructs(self) -> dict[str, g.GBase]:
        return {con.name: con.type for con in self._constructs}

    @cached_property
    def t0(self) -> float:
        return self.t - self.t[0]

    def __getattr__(self, name: str):
        if name in self.construct_names:
            return getattr(self, f"{name}")
        if name in self.columns:
            return self.column_getters[name]()
        if name in self.labels.lgs:
            value = Slicer(name, self.labels[name], self)
            self.__dict__[name] = value
            return value

        raise AttributeError(f"Unknown column or construct {name}")

    def to_dataframe(self, labels: bool = False) -> pd.DataFrame:
        df = pd.DataFrame(
            np.column_stack([getattr(self, con).data for con in self.construct_names]),
            columns=self.columns,
            index=self.t,
        )
        if labels:
            df = pd.concat([df, self.labels.to_df(self.t)], axis=1)
        return df

    def to_dict(self, legacy: boolean = False) -> dict[str, dict]:
        if legacy:
            df: pd.DataFrame = pd.concat([self.data, self.labels.to_df(self.t)], axis=1)
            return df.to_dict(orient="records")
        else:
            return {
                "data": self.data.to_dict(orient="list"),
                "labels": self.labels.to_dict(),
            }

    @classmethod
    def from_dict(Cls, data: dict | list[dict]) -> Self:
        raise NotImplementedError("from_dict is not implemented yet")

    def __len__(self):
        return len(self.time)

    @cached_property
    def duration(self):
        return self.t[-1] - self.t[0]

    def _get_interpolator(self, con: str):
        """caching for the interpoaltors"""
        key = f"_interp_{con}"
        if key not in self.__dict__:
            geom: g.Base = getattr(self, con)
            self.__dict__[key] = geom.interpolate(
                self.t, default_interpolators[geom.__class__.__name__]
            )
        return self.__dict__[key]

    def interpolate(self, t: float | npt.NDArray) -> Self:

        return self.__class__(
            *[self._get_interpolator(con)(t) for con in self.construct_names],
            labels=self.labels.copy(),
        )

    def resample(self, dt: float = 1 / 25, sli: slice | None = None):
        if sli is None or sli.start is None:
            start = self.t[0]
        else:
            start = sli.start
        if sli is None or sli.stop is None:
            stop = self.t[-1]
        else:
            stop = sli.stop

        return self.interpolate(np.linspace(start, stop, int((stop - start) / dt)))

    @cached_property
    def _t_np(self):
        return self.data.index.to_numpy()

    @cached_property
    def _dt_np(self):
        return self.data["dt"].to_numpy()

    def __getitem__(self, sli):
        if isinstance(sli, Number | np.ndarray):
            return self.interpolate(sli)

        if isinstance(sli, slice):
            start = sli.start if sli.start is not None else self.t[0]
            istart = np.searchsorted(self.t, start, "left")
            stop = sli.stop if sli.stop is not None else self.t[-1]
            istop = np.searchsorted(self.t, stop, "right") - 1

            frames = []

            if start >= self.t[0] and start <= self.t[-1] and self.t[istart] != start:
                frames.append(self.interpolate(start))

            frames.append(
                self.__class__(
                    *[
                        getattr(self, con)[istart : istop + 1]
                        for con in self.construct_names
                    ]
                )
            )

            if stop >= self.t[0] and stop <= self.t[-1] and self.t[istop] != stop:
                frames.append(self.interpolate(stop))

            return self.__class__.concatenate(frames).label(
                self.labels.slice(start, stop)
            )

        if hasattr(sli, "__len__"):
            return self.interpolate(np.array(sli)).label(self.labels)

        raise TypeError(f"Expected Number, slice or array, got {type(sli).__name__}")

    @property
    def iloc(self):
        return _ILocer(self)

    def __iter__(self):
        for t in self.t:
            yield self[t]

    def __eq__(self, other: Self):
        return self.data.equals(other.data) and self.labels == other.labels

    def __repr__(self):
        return f"{self.__class__.__name__}({','.join([str(l) for l in self.labels.lgs])},duration={self.duration})"

    def copy(self) -> Self:
        return self.__class__(
            *[getattr(self, con).copy() for con in self.construct_names],
            labels=self.labels.copy(),
        )

    def append(self, other, timeoption: str = "dt"):
        if timeoption in ["now", "t"]:
            t = np.array([time()]) if timeoption == "now" else other.t
            dt = other.dt
            dt[0] = t[0] - self.t[-1]
            new_time = g.Time(t, dt)
        elif timeoption == "dt":
            new_time = g.Time(other.t + self[-1].t - other[0].t + other[0].dt, other.dt)

        return self.__class__(
            pd.concat(
                [self.data, other.copy(new_time).data], axis=0, ignore_index=True
            ).set_index("t", drop=False)
        )

    def zero_index(self):
        return self.shift_time(-self.data.index[0])

    def shift_time(self, offset: float):
        """Shift the time of the table by offset seconds"""
        return (
            replace(self, time=self.time + offset)
            .remove_labels()
            .label(self.labels.offset(offset))
        )
        # data = self.copy(time=self.time + offset).label(self.labels.offset(offset))

        # return data

    @classmethod
    def stack(
        Cls,
        sts: list[Table] | dict[str, Table],
        label_title: str | None = None,
        label_values: list[str] | None = None,
        overlap: Literal[0, 1] = 1,
    ) -> Self:
        """Stack a list of Tables on top of each other.
        The overlap is the number of rows to overlap between each st.
        Existing labels will be moved to sublabels if label_title is not None
        otherwise they will be concatenated.
        """
        if isinstance(sts, dict):
            label_values = list(sts.keys())
            sts = list(sts.values())

        if label_title:
            assert len(label_values) == len(sts)
            sts[0] = sts[0].over_label(label_title, label_values[0])

        newst = sts[0]
        if len(sts) > 1:
            for i, st in enumerate(sts[1:], 1):
                if overlap > 0:
                    next_t = newst.t[-overlap]
                    newst = Cls(newst.data.iloc[:-overlap, :]).label(newst.labels)
                else:
                    next_t = newst.t[-1] + newst.dt[-1]

                if label_title:
                    st = st.over_label(label_title, label_values[i])

                newst = Cls.concatenate(
                    [
                        newst,
                        st.shift_time(next_t - st.data.index[0]),
                    ]
                )

        return newst

    def recalculate_dt(self):
        t = g.Time.from_t(self.data.t.to_numpy())
        _ndf = self.data.assign(
            t=t.t,
            dt=t.dt,
        )
        assert _ndf.index.is_monotonic_increasing
        return self.__class__(_ndf, self.labels)

    @classmethod
    def concatenate(Cls, sts: list[Table] | dict[Table]) -> Self:
        """Concatenate a list of Tables and recalculate the timesteps
        The times are exprected to be correct in the input tables and the tables
        must be passed in order. No checks are performed to ensure the resulting time is monotonic
        """

        return Cls(
            *[
                GCls.concatenate([getattr(st, con) for st in sts])
                for con, GCls in sts[0].constructs.items()
            ]
        ).label(LabelGroups.concat(*[st.labels for st in sts]))

    @classmethod
    def splice(Cls, sts: list[Table]):
        """
        Splice a list of Tables together,
        the time of the first table is preserved and the time of the last table is preserved
        if indeces are repeated the first table with that index is used, the others are ignored
        """

        newdf = (
            pd.concat(
                [st.data for st in sts],
                axis=0,
            )
            .drop_duplicates(subset="t")
            .sort_index()
            .reset_index(drop=True)
            .set_index("t", drop=False)
        )

        return (
            Cls(newdf)
            .recalculate_dt()
            .label(LabelGroups.concat(*[st.labels for st in sts]))
        )

    def label(
        self,
        lgs: LabelGroups = None,
        inplace=False,
        **kwargs: dict[str, LabelGroup | str | npt.NDArray],
    ) -> Self:
        labelgroups: dict[str, LabelGroup] = {} if lgs is None else lgs.lgs
        for key, value in kwargs.items():
            newlg: LabelGroup = None
            if isinstance(value, str):
                newlg = LabelGroup({value: Label(self.t[0], self.t[-1])})
            elif isinstance(value, LabelGroup):
                newlg = value
            elif pd.api.types.is_list_like(value):
                newlg = LabelGroup.read_array(self.t, np.array(value))
            else:
                raise ValueError(f"Unknown type for label {key}")
            newlg = newlg.intersect(self.time)
            if not newlg.empty:
                if key in labelgroups:
                    raise ValueError(f"Label {key} already exists")
                labelgroups[key] = newlg
        new_lgs = LabelGroups(labelgroups)
        if inplace:
            self.labels = new_lgs
        return self.__class__(
            *[getattr(self, con) for con in self.construct_names],
            labels=new_lgs,
        )

    def shift_labels(self, lg: str, boundaries: list[float]) -> Self:
        new_lg = self.labels[lg].set_boundaries(boundaries)
        _newlgs = self.labels.copy()
        del _newlgs.lgs[lg]
        _newlgs.lgs[lg] = new_lg
        return self.__class__(self.data, _newlgs)

    def nest_labels(self, **kwargs: dict[str, npt.NDArray]) -> Self:
        first_key = next(iter(kwargs.keys()))
        first_values = next(iter(kwargs.values()))
        newst = self.label(**{first_key: first_values})

        if len(kwargs) == 1:
            return newst
        else:
            sts = []
            for name, label in newst.labels[first_key].items():
                iloc = label.to_iloc(newst.t)
                sublabels = {
                    k: v[iloc.start : iloc.stop]
                    for k, v in kwargs.items()
                    if k != first_key
                }
                sts.append(getattr(newst, first_key)[name].nest_labels(**sublabels))

            return self.__class__.stack(
                sts, first_key, pd.unique(np.array(first_values))
            )

    def over_label(
        self, title: str, value: str, child_groups: list[str] | None = None
    ) -> Self:
        """label with the value, make existing labels sublabels of the new label
        if child_groups is not None, only the child groups are made sublabels"""
        child_groups = (
            list(self.labels.keys()) if child_groups is None else child_groups
        )
        labels = self.labels.filter_keys(lambda k: k in child_groups)
        newlg = LabelGroup({value: Label(self.t[0], self.t[-1], labels)})

        return self.label(
            LabelGroups({title: newlg}),
            **self.labels.filter_keys(lambda k: k not in child_groups).lgs,
        )

    def remove_labels(self) -> Self:
        return self.__class__(*[getattr(self, con) for con in self.construct_names])

    @staticmethod
    def labselect(
        data: pd.DataFrame, test: str | None = None, offset=False, **kwargs
    ) -> pd.DataFrame:
        """Select rows from a dataframe based on the values in the kwargs
        in kwargs, keys are column names and values are the values to select
        if test is not None, it is a string that is a pandas string method .
        if offset is True the row after the last selected row for each kwarg is included.
        """
        sel = np.full(len(data), True)
        for k, v in kwargs.items():
            if test:
                sel = getattr(data[k].str, test)(v)
            else:
                sel = sel & (data[k] == v)
        if offset:
            return data.loc[sel + (sel.astype(int).diff() == -1)]
        else:
            return data.loc[sel]

    @staticmethod
    def copy_labels(
        template: Table,
        flown: Table,
        path: Annotated[npt.NDArray[np.integer], Literal["N", 2]] = None,
        min_len=None,
    ) -> Self:
        """Copy the labels from template to flown along the index warping path
        If path is None, the labels are copied directly from the template to the flown
        TODO - min_len prevents the labels from being shortened to less than min_len rows,
        even if the label dows not exist in the warping path the order of labels in template
        will be preserved.
        """

        newtab = flown.label(
            **{
                k: v.transfer(template.t, flown.t, path)
                for k, v in template.labels.items()
            }
        )

        if min_len is not None:
            newtab = newtab.remove_labels().label(
                LabelGroups(
                    {
                        k: v.to_iloc(flown.t)
                        .insert_list(list(template.labels[k].keys()))
                        .expand(min_len)
                        .to_t(flown.t)
                        for k, v in newtab.labels.items()
                    }
                )
            )
        return newtab

    def step_label(
        self,
        group: str,
        name: str,
        steps: int | Literal["left_limit", "right_limit"],
        min_len: int = 3,
    ) -> Self:
        """Shift the label by steps rows"""
        labels = self.labels[group].to_iloc(self.t)

        label_index = list(labels.keys()).index(name)

        if steps == "left_limit":
            steps = -labels[name].width + min_len
        elif steps == "right_limit":
            next_label = list(labels.keys())[min(len(labels) - 1, label_index + 1)]
            steps = labels[next_label].width - min_len - 1
        return self.__class__(self.data).label(
            self.labels.step_boundary(group, name, steps, self.t, min_len)
        )

    def move_label(
        self, group: str, name: str, t: float, min_duration: float = 0
    ) -> Self:
        return self.__class__(self.data).label(
            self.labels.set_boundary(group, name, t, min_duration)
        )

    def set_boundaries(self, group: str, boundaries: npt.NDArray) -> Self:
        return self.__class__(self.data).label(
            self.labels.set_boundaries(group, boundaries)
        )


@dataclass
class _ILocer:
    table: Table

    def __getitem__(self, sli) -> Table:

        return self.table[get_value(self.table.t, sli)]
