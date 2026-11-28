from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, cached_property, partial
from inspect import signature
from itertools import chain
from numbers import Number
from time import time
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, Self

import geometry as g
import numpy as np
import numpy.typing as npt
import pandas as pd

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


def _interp_column(
    ts: npt.NDArray, values: npt.NDArray, t: float | npt.NDArray
) -> npt.NDArray:
    """Linear interpolation of an extra column (rows of values at times ts) at t."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if values.ndim == 1:
        return np.interp(t, ts, values)
    return np.column_stack([np.interp(t, ts, values[:, i]) for i in range(values.shape[1])])


def _common_extra(tables: list[Table]) -> list[str]:
    """Names of the extra columns that every table has."""
    if not tables or not tables[0].extra:
        return []
    return [k for k in tables[0].extra if all(k in tb.extra for tb in tables[1:])]


@cache
def _init_misc_fields(cls: type) -> tuple[str, ...]:
    """__init__ parameters after the constructs and labels, which replace() carries over."""
    return tuple(
        list(signature(cls.__init__).parameters.keys())[len(cls._constructs) + 2 :]
    )


@dataclass
class Construct:
    name: str
    type: g.GBase
    cols: list[str]
    lazy: bool = False

    @cached_property
    def raw_name(self) -> str:
        return ("_" if self.lazy else "") + self.name


class Table:
    """Base data structure, wraps around a pandas dataframe.
    All the columns are defined in the constructs class variable.
    A dictionary of labels is included, keys are label group names, values are instances of LabelGroup.

    Extra columns are named arrays with one row per sample, for values derived from the
    table that the code using it needs to carry along, for example a rate smoothed over a
    whole manoeuvre that its slices must not recompute. Subclasses opt in by accepting an
    ``extra`` argument in __init__ (State does). Slicing, interpolating, stacking and
    splicing carry them like constructs, interpolating linearly at new times. Replacing a
    construct other than time drops them, because the data they were derived from has
    changed; pass ``extra`` explicitly to keep them. They are not serialised.
    """

    _construct_freq: ClassVar[float] = 25
    _constructs: ClassVar[list[Construct]] = [Construct("time", g.Time, ["t", "dt"])]
    # instance attribute set in __init__; the read-only default covers instances
    # unpickled from before extra columns existed
    extra: Mapping[str, npt.NDArray] = MappingProxyType({})

    def __init__(
        self,
        time: g.Time,
        labels: LabelGroups = None,
        extra: dict[str, npt.NDArray] | None = None,
    ):
        if labels is None:
            labels = LabelGroups()
        self.time = time
        self.labels = labels
        self.extra = {} if extra is None else dict(extra)

        self._validate()

    def _validate(self):
        assert isinstance(self.time, g.Time), "time must be an instance of g.Time"

        for con in self._constructs[1:]:
            _con = getattr(self, con.raw_name)

            if _con is not None:
                if not isinstance(_con, con.type):
                    raise TableError(
                        f"Construct {con.name} must be of type {con.type.__name__}"
                    )
                if len(_con) != len(self.time):
                    raise TableError(
                        f"Construct {con.name} must have the same length as time"
                    )

        for name, values in self.extra.items():
            if len(values) != len(self.time):
                raise TableError(
                    f"Extra column {name} must have the same length as time"
                )

    @classmethod
    def supports_extra(Cls) -> bool:
        return "extra" in _init_misc_fields(Cls)

    def _extra_kwargs(self, extra: dict[str, npt.NDArray]) -> dict:
        """The extra argument for a new instance, omitted when there is nothing to carry
        (so classes that do not accept extra columns are built as before)."""
        return {"extra": extra} if extra else {}

    def _interp_extra(self, t: float | npt.NDArray) -> dict[str, npt.NDArray]:
        return {k: _interp_column(self.t, v, t) for k, v in self.extra.items()}

    def with_extra(self, **columns: npt.ArrayLike) -> Self:
        """A copy with these extra columns added or replaced (one row per sample)."""
        if not self.supports_extra():
            raise TableError(f"{self.__class__.__name__} does not take extra columns")
        return self.replace(
            extra={**self.extra, **{k: np.asarray(v, dtype=float) for k, v in columns.items()}}
        )

    def copy(self, **kwargs) -> Self:
        return self.replace(copy=True, **kwargs)

    def replace(self, copy: bool = False, **kwargs) -> Self:
        """Return a new instance of the table with the specified attributes replaced."""
        def _copy(x):
            return x.copy() if copy and hasattr(x, "copy") else x

        _misc_fields = _init_misc_fields(self.__class__)

        misc = {k: kwargs.get(k, getattr(self, k)) for k in _misc_fields if k != "extra"}
        if "extra" in _misc_fields:
            if "extra" in kwargs:
                misc["extra"] = kwargs["extra"]
            elif not any(con.name in kwargs for con in self._constructs[1:]):
                # extra columns are derived from the constructs, so keep them only
                # while the constructs are unchanged
                misc["extra"] = (
                    {k: v.copy() for k, v in self.extra.items()} if copy else self.extra
                )

        return self.__class__(
            *[
                kwargs.get(con.name, _copy(getattr(self, con.raw_name)))
                for con in self._constructs
            ],
            labels=kwargs.get("labels", self.labels),
            **misc,
        )

    @classmethod
    def get_columns(Cls) -> list[str]:
        return [col for con in Cls._constructs for col in con.cols]

    @cached_property
    def columns(self) -> list[str]:
        return self.__class__.get_columns()

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
    def raw_construct_names(self) -> list[str]:
        return [con.raw_name for con in self._constructs]

    @cached_property
    def constructs(self) -> dict[str, g.GBase]:
        return {con.name: con.type for con in self._constructs}

    @cached_property
    def construct_dict(self) -> dict[str, Construct]:
        return {con.name: con for con in self._constructs}

    @property
    def raw_constructs(self) -> dict[str, g.GBase | None]:
        return {con.name: getattr(self, con.raw_name) for con in self._constructs}

    @property
    def loaded_constructs(self) -> dict[str, g.GBase]:
        return {
            con.name: getattr(self, con.raw_name)
            for con in self._constructs
            if getattr(self, con.raw_name) is not None
        }

    @cached_property
    def t0(self) -> float:
        return self.t - self.t[0]

    def to_numpy(self, generate: bool = False) -> npt.NDArray:
        """Return the data as a numpy array, if generate is True
        the lazy attributes are generated and included in the output
        otherwise the space is filled with NaNs
        """
        _condata = []
        for con in self._constructs:
            if con.lazy and (not generate) and (getattr(self, con.raw_name) is None):
                _condata.append(np.full((len(self), len(con.cols)), np.nan))
            else:
                _condata.append(getattr(self, con.name).data)
        return np.column_stack(_condata)

    @classmethod
    def from_numpy(
        Cls,
        data: npt.NDArray,
        labels: LabelGroups = None,
        extra: dict[str, npt.NDArray] | None = None,
    ) -> Self:
        if labels is None:
            labels = LabelGroups()

        _col = 0
        _cons: list[g.GBase | None] = []
        for con in Cls._constructs:
            _data = data[:, _col : _col + len(con.cols)]
            if np.any(np.isnan(_data)):
                if not con.lazy:
                    raise TableError(
                        f"Construct {con.name} is not lazy but has NaN values in the data"
                    )
                _cons.append(None)
            else:
                _cons.append(con.type(data[:, _col : _col + len(con.cols)]))
            _col += len(con.cols)

        return Cls(
            *_cons,
            labels=labels,
            **({"extra": extra} if extra else {}),
        )

    def __getattr__(self, name: str):
        if name in self.columns:
            return self.column_getters[name]()
        if self.labels is not None and name in self.labels.lgs:
            value = Slicer(name, self.labels[name], self)
            self.__dict__[name] = value
            return value

        raise AttributeError(f"Unknown column or construct {name}")

    def to_dataframe(self, labels: bool = True, con_subset: list[str]|None=None) -> pd.DataFrame:
        if con_subset is None:
           cons = [con for con in self._constructs if getattr(self, con.raw_name) is not None]
        else:
            cons = [self.construct_dict[con] for con in con_subset]
        df = pd.DataFrame(
            np.column_stack([getattr(self, con.name).data for con in cons]),
            columns=list(chain(*[con.cols for con in cons])),
            index=self.t,
        ).dropna(axis=1)
        if labels:
            df = pd.concat([df, self.labels.to_df(self.t)], axis=1)
        return df

    @property
    def df(self) -> pd.DataFrame:
        return self.to_dataframe()

    def to_dict(self) -> dict[str, dict]:
        return self.df.to_dict(orient="records")
        
    @classmethod
    def from_df(Cls, data: pd.DataFrame, lgs: LabelGroups = None) -> Self:
        cons = []
        for con in Cls._constructs:
            if all(col in data.columns for col in con.cols):
                cons.append(con.type(data[con.cols]))
            else:
                if not con.lazy:
                    raise TableError(
                        f"Construct {con.name} is not lazy but is missing from the dataframe"
                    )
                cons.append(None)

        lab_cols = [c for c in data.columns if c not in Cls.get_columns()]

        if lgs is None and len(lab_cols) > 0:
            labdf = data.loc[:, lab_cols]

            instance = Cls(*cons)
            return instance.nest_labels(**labdf.to_dict(orient="list"))

        else:
            return Cls(*cons, labels=lgs)

    @classmethod
    def from_dict(Cls, data: dict | list[dict]) -> Self:
        if isinstance(data, list):
            df = pd.DataFrame.from_dict(data).set_index("t", drop=False)
            return Cls.from_df(df)
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data["data"]).set_index("t", drop=False)
            df = df.drop(columns=["manoeuvre", "element"], errors="ignore")
            labels = LabelGroups.from_dict(data["labels"])
            return Cls.from_df(df, labels)
        raise NotImplementedError("from_dict is not implemented yet")

    def __len__(self):
        return len(self.time)

    @cached_property
    def duration(self):
        return self.time.duration

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
            *[
                self._get_interpolator(con.name)(t)
                if getattr(self, con.raw_name) is not None
                else None
                for con in self._constructs
            ],
            labels=self.labels.copy(),
            **self._extra_kwargs(self._interp_extra(t)),
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

        return self[np.linspace(start, stop, int((stop - start) / dt))]

    @cached_property
    def _t_np(self):
        return self.data.index.to_numpy()

    @cached_property
    def _dt_np(self):
        return self.data["dt"].to_numpy()

    def _row_slice_supported(self) -> bool:
        """Whether interpolate() is this class's construct-wise interpolation, so
        _row_slice can stand in for it."""
        return type(self).interpolate is Table.interpolate

    def _row_slice(
        self, t: npt.NDArray, istart: int, istop: int, head: bool, tail: bool
    ) -> Self:
        """self.interpolate(t) for t = [start?] + self.t[istart:istop+1] + [stop?].

        Interpolating at a recorded sample returns that sample unchanged, so the
        recorded rows are copied and only the inserted end rows are interpolated.
        Time is still interpolated, as it recalculates dt for the new rows.
        """
        constructs = []
        for con in self._constructs:
            raw = getattr(self, con.raw_name)
            if raw is None:
                constructs.append(None)
            elif con.name == "time":
                constructs.append(self._get_interpolator("time")(t))
            else:
                parts = []
                if head:
                    parts.append(self._get_interpolator(con.name)(t[:1]))
                parts.append(con.type(raw.data[istart : istop + 1].copy()))
                if tail:
                    parts.append(self._get_interpolator(con.name)(t[-1:]))
                constructs.append(
                    parts[0] if len(parts) == 1 else con.type.concatenate(parts)
                )
        extra = {}
        for name, values in self.extra.items():
            parts = []
            if head:
                parts.append(_interp_column(self.t, values, t[:1]))
            parts.append(values[istart : istop + 1].copy())
            if tail:
                parts.append(_interp_column(self.t, values, t[-1:]))
            extra[name] = np.concatenate(parts)
        return self.__class__(
            *constructs, labels=self.labels.copy(), **self._extra_kwargs(extra)
        )

    def __getitem__(self, sli: Number | slice | npt.ArrayLike) -> Self:

        if isinstance(sli, Number) and (sli == 0 or sli == -1):
            return self.iloc[sli]

        if isinstance(sli, slice):
            start = sli.start if sli.start is not None else self.t[0]
            istart = np.searchsorted(self.t, start, "left")
            stop = sli.stop if sli.stop is not None else self.t[-1]
            istop = np.searchsorted(self.t, stop, "right") - 1

            sli = []

            head = start >= self.t[0] and start <= self.t[-1] and self.t[istart] != start
            if head:
                sli.append(self._get_interpolator("time")(start))

            sli.append(self.time[istart : istop + 1])

            tail = stop >= self.t[0] and stop <= self.t[-1] and self.t[istop] != stop
            if tail:
                sli.append(self._get_interpolator("time")(stop))
            sli = g.Time.concatenate(sli).t

            if (
                0 <= istart <= istop < len(self)
                and self._row_slice_supported()
                and sli[0] >= 0
            ):
                return self._row_slice(sli, istart, istop, head, tail).label(
                    self.labels.slice(np.min(sli), np.max(sli))
                )

        if isinstance(sli, Number | np.ndarray | list):
            sli = np.array(sli)
            sli = np.where(sli < 0, len(self) + sli, sli)

            _label_start = np.min(sli)
            _label_stop = np.max(sli)
            return self.interpolate(sli).label(
                self.labels.slice(_label_start, _label_stop)
            )

        raise TypeError(f"Expected Number, slice or array, got {type(sli).__name__}")

    @property
    def iloc(self):
        return _ILocer(self)

    def __iter__(self):
        for t in self.t:
            yield self[t]

    def almost_equal(self, other: Self, tol: float = 1e-6) -> bool:
        return (
            np.all(
                [
                    con.almost_equal(other.raw_constructs[name], tol)
                    for name, con in self.raw_constructs.items()
                ]
            )
            and self.labels == other.labels
        )

    def __eq__(self, other: Self):
        return (
            all(
                getattr(self, con) == getattr(other, con)
                for con in self.raw_construct_names
            )
            and self.labels == other.labels
        )

    def __str__(self):
        return f"{self.__class__.__name__}({','.join([str(l) for l in self.labels.lgs.keys()] if self.labels is not None else [])},duration={self.duration})"

    def __repr__(self):
        return str(self)

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
        return self.shift_time(-self.t[0])

    def shift_time(self, offset: float):
        """Shift the time of the table by offset seconds"""
        return self.replace(time=self.time + offset, labels=self.labels.offset(offset))

        # data = self.copy(time=self.time + offset).label(self.labels.offset(offset))

        # return data

    @classmethod
    def stack(
        Cls,
        sts: list[Table] | dict[str, Table],
        label_title: str | None = None,
        label_values: list[str] | None = None,
        overlap: bool = True,
    ) -> Self:
        """Stack a list of Tables on top of each other and sort out the times.
        if overlap is True the last row of the previous table is removed.
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
                _t_offset = newst.t[-1] - st.t[0] + (0 if overlap else st.dt[-1])

                _st = st.replace(
                    time=st.time + _t_offset, labels=st.labels.offset(_t_offset)
                )

                if label_title:
                    _st = _st.over_label(label_title, label_values[i])

                if len(newst) > 1 or not overlap:
                    if overlap:
                        newst = Cls(
                            *[
                                getattr(newst, con)[:-1]
                                for con in newst.construct_names
                            ],
                            labels=newst.labels,
                            **newst._extra_kwargs(
                                {k: v[:-1] for k, v in newst.extra.items()}
                            ),
                        )

                    newst = Cls.concatenate(
                        [
                            newst,
                            _st,
                        ]
                    )
                else:
                    newst = _st

        return newst

    def recalculate_dt(self):
        newt = g.Time.from_t(self.time.t)
        return self.replace(time=newt, labels=self.labels)

    @classmethod
    def concatenate(Cls, sts: list[Table] | dict[Table]) -> Self:
        """Concatenate a list of Tables and recalculate the timesteps
        The times are exprected to be correct in the input tables and the tables
        must be passed in order. No checks are performed to ensure the resulting time is monotonic
        """

        sts = list(sts.values()) if isinstance(sts, dict) else list(sts)
        extra = {k: np.concatenate([st.extra[k] for st in sts]) for k in _common_extra(sts)}
        return Cls(
            *[
                GCls.concatenate([getattr(st, con) for st in sts])
                for con, GCls in sts[0].constructs.items()
            ],
            **({"extra": extra} if extra else {}),
        ).label(LabelGroups.concat(*[st.labels for st in sts]))

    def remove_duplicate_ts(self) -> Self:
        """Remove duplicate time steps from the table"""
        _, idx = np.unique(self.t, return_index=True)
        return self.iloc[idx]

    @classmethod
    def splice(Cls, sts: list[Table]):
        """
        Splice a list of Tables together,
        the time of the first table is preserved and the time of the last table is preserved
        if indeces are repeated the first table with that index is used, the others are ignored
        """
        
        data = np.vstack([st.to_numpy(False) for st in sts])
        order = data[:, 0].argsort()
        extra = {
            k: np.concatenate([st.extra[k] for st in sts])[order]
            for k in _common_extra(sts)
        }

        return (
            Cls.from_numpy(
                data[order, :],
                LabelGroups.concat(*[st.labels for st in sts]),
                extra,
            )
            .remove_duplicate_ts()
            .recalculate_dt()
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
        return self.replace(
            labels=new_lgs,
        )

    def shift_labels(self, lg: str, boundaries: list[float]) -> Self:
        new_lg = self.labels[lg].set_boundaries(boundaries)
        _newlgs = self.labels.copy()
        del _newlgs.lgs[lg]
        _newlgs.lgs[lg] = new_lg
        return self.replace(labels=_newlgs)

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
        return self.replace(labels=LabelGroups())

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
        name: str | int,
        steps: int | Literal["left_limit", "right_limit"],
        min_len: int = 3,
    ) -> Self:
        """Shift the label by steps rows"""
        labels = self.labels[group].to_iloc(self.t)

        label_index = list(labels.keys()).index(name) if isinstance(name, str) else name

        if steps == "left_limit":
            steps = -labels[name].width + min_len
        elif steps == "right_limit":
            next_label = list(labels.keys())[min(len(labels) - 1, label_index + 1)]
            steps = labels[next_label].width - min_len - 1
        return self.replace(
            labels=self.labels.step_boundary(group, name, steps, self.t, min_len)
        )

    def move_label(
        self, group: str, name: str, t: float, min_duration: float = 0
    ) -> Self:
        return self.set_boundary(group, name, t, min_duration)

    def set_boundary(self, group: str, name_or_id: str | int, t: float, min_duration: float = 0) -> Self:

        return self.replace(labels=self.labels.set_boundary(group, name_or_id, t, min_duration))

    def set_boundaries(self, group: str, boundaries: npt.NDArray) -> Self:
        return self.replace(labels=self.labels.set_boundaries(group, boundaries))


@dataclass
class _ILocer:
    table: Table

    def int_items(self, sli: slice | list[int] | npt.NDArray) -> dict[str, g.GBase]:
        """Return a dictionary of the constructs sliced by the given slice or list of indices"""
        if isinstance(sli, slice):
            sli = slice(sli.start, sli.stop + 1 if sli.stop else None)
            start = sli.start if sli.start is not None else 0
            stop = sli.stop if sli.stop is not None else len(self.table)

        elif isinstance(sli, (list, np.ndarray)):
            sli = np.array(sli).astype(int)
            start = np.min(sli)
            stop = np.max(sli) + 1
        else:
            sli = int(sli)
            start = sli
            stop = sli + 1

        return self.table.replace(
            **{
                con.name: getattr(self.table, con.raw_name)[sli]
                if getattr(self.table, con.raw_name) is not None
                else None
                for con in self.table._constructs
            },
            labels=self.table.labels.slice(
                self.table.t[start],
                self.table.t[stop - 1] + self.table.dt[stop - 1],
            ),
            **self.table._extra_kwargs(
                {
                    k: v[sli : sli + 1] if isinstance(sli, int) else v[sli]
                    for k, v in self.table.extra.items()
                }
            ),
        )

    def __getitem__(self, sli: Number | slice | npt.ArrayLike) -> Table:

        if isinstance(sli, Number | np.ndarray | list):
            if np.all(np.array(sli).astype(int) == np.array(sli)):
                return self.int_items(sli)
            else:
                return self.table[self.table.time.get_value(np.array(sli))]
        elif isinstance(sli, slice):
            assert sli.step is None, "Slicing with step is not supported"
            if (sli.start is None or int(sli.start) == sli.start) and (
                sli.stop is None or int(sli.stop) == sli.stop
            ):
                return self.int_items(sli)
            else:
                return self.table[
                    slice(
                        self.table.time.get_value(sli.start)
                        if sli.start is not None
                        else None,
                        self.table.time.get_value(sli.stop)
                        if sli.stop is not None
                        else None,
                        None,
                    )
                ]

        return self.table[sli]
