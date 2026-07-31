from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from numbers import Number
from time import time
from typing import Annotated, ClassVar, Literal, Self, overload
from xmlrpc.client import boolean

import geometry as g
import numpy as np
import numpy.typing as npt
import pandas as pd
from geometry.utils import get_value

from flightdata.base.table.constructs import Constructs, SVar

from .label import Label
from .labelgroup import LabelGroup
from .labelgroups import LabelGroups
from .slicer import Slicer

default_interpolators = {
    "Time": "linterp",
    "Point": "linterp",
    "Quaternion": "slerp",
    "Air": "linterp",
    "Attack": "linterp",
}


class TableError(Exception):
    """Base exception for table-related errors."""


@dataclass
class Table:
    """Base data structure, wraps around a pandas dataframe.
    All the columns are defined in the constructs class variable.
    A dictionary of labels is included, keys are label group names, values are instances of LabelGroup.
    """

    constructs: ClassVar[Constructs] = Constructs(
        [SVar("time", g.Time, ["t", "dt"], lambda tab: g.Time.from_t(tab.t))]
    )
    data: pd.DataFrame
    labels: LabelGroups = field(default_factory=lambda: LabelGroups())

    @overload
    def __getattr__(self, name: Literal["time"]) -> g.Time: ...

    @cached_property
    def index(self):
        return self.data.index

    @cached_property
    def t_end(self):
        return self.t + self.dt

    @cached_property
    def t0(self):
        return self.t - self.t[0]

    @classmethod
    def build(
        Cls,
        data: pd.DataFrame | dict | pd.Series,
        labels: LabelGroups = None,
        fill=True,
        min_len=1,
    ):
        labels = LabelGroups() if labels is None else labels
        if isinstance(data, dict):
            data = pd.Series(data)
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data).T

        if len(data) < min_len:
            raise TableError(
                f"Table constructor length check failed, data length = {len(data)}, min_len = {min_len}"
            )

        base_cols = [c for c in data.columns if c in Cls.constructs.cols()]
        lab_cols = [c for c in data.columns if c not in base_cols]
        labdf = data.loc[:, lab_cols]

        if data.loc[:, base_cols].isnull().values.any():
            raise ValueError("nan values in data")

        instance = (
            Cls(data.loc[:, base_cols], labels).populate()
            if fill
            else Cls(data.loc[:, base_cols], labels)
        )

        if len(labdf.columns) and not len(labels):
            return instance.nest_labels(**labdf.to_dict(orient="list"))
        else:
            return instance

    def populate(self):
        missing = self.__class__.constructs.missing(self.data.columns)
        if not missing:
            return self.__class__(self.data, self.labels)
        new_frames = [
            svar.builder(self)
            .to_pandas(columns=svar.keys, index=self.data.index)
            .loc[:, [k for k in svar.keys if k not in self.data.columns]]
            for svar in missing
        ]
        return self.__class__(pd.concat([self.data, *new_frames], axis=1), self.labels)


    def __getattr__(self, name):
        if name in self.data.columns:
            value = self.data[name].to_numpy()
            self.__dict__[name] = value
            return value

        if name in self.__class__.constructs.data:
            con = self.__class__.constructs[name]
            value = con.obj(self.data[con.keys].values)
            self.__dict__[name] = value
            return value

        if name in self.labels.lgs:
            value = Slicer(name, self.labels[name], self)
            self.__dict__[name] = value
            return value

        raise AttributeError(f"Unknown column or construct {name}")
    
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
        if not isinstance(data, list):
            df = pd.DataFrame.from_dict(data["data"]).set_index("t", drop=False)
            labels = LabelGroups.from_dict(data["labels"])
            return Cls.build(df, labels, True)
        else:
            if "data" in data:
                data = data["data"]
            df = pd.DataFrame.from_dict(data).set_index("t", drop=False)
            return Cls.build(df)

    def to_parquet(self, path: str):
        df: pd.DataFrame = pd.concat([self.data, self.labels.to_df(self.t)], axis=1)
        df.to_parquet(path, index=False)

    @classmethod
    def from_parquet(Cls, path: str) -> Self:
        df = pd.read_parquet(path)
        return Cls.build(df)

    def to_pybytes(self, ascii: bool = False) -> bytes:
        import base64

        import pyarrow as pa

        table = pa.Table.from_pandas(
            pd.concat([self.data, self.labels.to_df(self.t)], axis=1)
        )

        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

        payload = sink.getvalue().to_pybytes()

        return base64.b64encode(payload).decode("ascii") if ascii else payload

    @classmethod
    def from_pybytes(Cls, payload: bytes | str) -> Self:
        import base64

        import pyarrow as pa

        if isinstance(payload, str):
            payload = base64.b64decode(payload)

        reader = pa.ipc.open_stream(pa.BufferReader(payload))
        table = reader.read_all()
        df = table.to_pandas().set_index("t", drop=False)
        return Cls.build(df)

    def __len__(self):
        return len(self.data)

    @property
    def duration(self):
        return self.t[-1] - self.t[0]

    def _get_interpolator(self, con):
        key = f"_interp_{con.name}"
        if key not in self.__dict__:
            geom = getattr(self, con.name)
            self.__dict__[key] = geom.interpolate(
                self.t, default_interpolators[con.obj.__name__]
            )
        return self.__dict__[key]

    def interpolate(self, t: npt.NDArray | float):
        if isinstance(t, Number):
            t = np.array([t])

        
        new_table = self.__class__.from_constructs(
            *[
                self._get_interpolator(con)(t)
                for con in self.constructs
            ]
        )

        return new_table.label(self.labels)


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
        t = self._t_np

        if isinstance(sli, Number):
            if sli == 0:
                return self.__class__(self.data.iloc[[0], :]).label(
                    self.labels.slice(t[0], t[0])
                )

            if sli < t[0] or sli > t[-1]:
                idx = np.searchsorted(t, sli)
                idx = max(0, min(idx, len(t)-1))
                return self.__class__(self.data.iloc[[idx], :]).label(
                    self.labels.slice(t[idx], t[idx])
                )

            idx = np.searchsorted(t, sli)
            if idx < len(t) and t[idx] == sli:
                return self.__class__(self.data.iloc[[idx], :]).label(
                    self.labels.slice(sli, sli)
                )

            return self.interpolate(sli)

        if isinstance(sli, slice):
            start = sli.start if sli.start is not None else t[0]
            stop  = sli.stop  if sli.stop  is not None else t[-1] + self._dt_np[-1]

            i0 = np.searchsorted(t, start, side="left")
            i1 = np.searchsorted(t, stop,  side="right")

            frames = []

            if start >= t[0] and start <= t[-1] and (i0 == len(t) or t[i0] != start):
                frames.append(self.interpolate(start).data)

            frames.append(self.data.iloc[i0:i1])

            if stop >= t[0] and stop <= t[-1] and (i1 == 0 or t[i1-1] != stop):
                    frames.append(self.interpolate(stop).data)

            out = pd.concat(frames, axis=0)

            res = self.__class__(out)
            res_t = res.t
            return res.label(self.labels.slice(res_t[0], res_t[-1]))

        if pd.api.types.is_list_like(sli):
            return self.concatenate([self[s] for s in sli])

        raise TypeError(f"Expected Number or slice, got {type(sli).__name__}")


    @property
    def iloc(self):
        return _ILocer(self)
    
    def __iter__(self):
        for t in self.t:
            yield self[t]

    def __eq__(self, other: Self):
        return self.data.equals(other.data) and self.labels == other.labels

    @classmethod
    def from_constructs(Cls, *args, **kwargs) -> Self:
        kwargs = dict(
            **{list(Cls.constructs.data.keys())[i]: arg for i, arg in enumerate(args)},
            **kwargs,
        )

        df = pd.concat(
            [
                x.to_pandas(columns=Cls.constructs[key].keys, index=kwargs["time"].t)
                for key, x in kwargs.items()
                if x is not None
            ],
            axis=1,
        )

        return Cls.build(df)

    def __repr__(self):
        return f"{self.__class__.__name__}({','.join([str(l) for l in self.labels.lgs])},duration={self.duration})"

    def copy(self, *args, **kwargs) -> Self:
        if not args and not kwargs:
            return self.__class__(self.data.copy(), self.labels)
        kwargs = dict(
            kwargs,
            **{list(self.constructs.data.keys())[i]: arg for i, arg in enumerate(args)},
        )  # add the args to the kwargs
        old_constructs = {
            key: self.__getattr__(key)
            for key in self.constructs.existing(self.data.columns).data
            if key not in kwargs
        }
        new_constructs = {
            key: value
            for key, value in list(kwargs.items()) + list(old_constructs.items())
        }
        return self.__class__.from_constructs(**new_constructs).label(self.labels)

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
        data = self.copy(time=self.time + offset).label(self.labels.offset(offset))

        return data

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
        """Concatenate a list of Tables and recalculate the timesteps"""
        df = pd.concat(
            [st.data for st in (sts if isinstance(sts, list) else sts.values())], axis=0
        )
        t = g.Time.from_t(df.t.to_numpy())
        df.t = t.t
        df.dt = t.dt
        assert df.index.is_monotonic_increasing
        return Cls(df).label(LabelGroups.concat(*[st.labels for st in sts]))

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
        
        return Cls(newdf).recalculate_dt().label(LabelGroups.concat(*[st.labels for st in sts]))

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
        return self.__class__(self.data, new_lgs)

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
        return self.__class__(self.data)

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
        self, group: str, name: str, steps: int | Literal["left_limit", "right_limit"], min_len: int=3
    ) -> Self:
        """Shift the label by steps rows"""
        labels = self.labels[group].to_iloc(self.t)

        label_index = list(labels.keys()).index(name)

        if steps=="left_limit":
            steps = -labels[name].width + min_len
        elif steps=="right_limit":
            next_label = list(labels.keys())[min(len(labels)-1, label_index+1)]
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