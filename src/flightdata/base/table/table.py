from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number
from time import time
from typing import Annotated, ClassVar, Literal, Self, Tuple, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from geometry import Base, Time, Point

from flightdata.base.table.constructs import Constructs, SVar

from .labels import Label, LabelGroup, Slicer, LabelGroups


@dataclass
class Table:
    """Base data structure, wraps around a pandas dataframe.
    All the columns are defined in the constructs class variable.
    A dictionary of labels is included, keys are label group names, values are instances of LabelGroup.
    """

    constructs: ClassVar[Constructs] = Constructs(
        [SVar("time", Time, ["t", "dt"], lambda tab: Time.from_t(tab.t))]
    )
    data: pd.DataFrame
    labels: LabelGroups = field(default_factory=lambda: LabelGroups())

    @overload
    def __getattr__(self, name: Literal["time"]) -> Time: ...

    @property
    def t_end(self):
        return self.t + self.dt

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
            raise Exception(
                f"Table constructor length check failed, data length = {len(data)}, min_len = {min_len}"
            )

        base_cols = [c for c in data.columns if c in Cls.constructs.cols()]
        # label_cols = [c for c in data.columns if c not in Cls.constructs.cols()]
        
        data = data.drop(columns=set(data.columns) - set(base_cols), errors="ignore")

        bcs = base_cols
        if data.loc[:, bcs].isnull().values.any():
            raise ValueError("nan values in data")
        
        return Cls(data, labels).populate() if fill else Cls(data, labels)

    def populate(self):
        newtab = self.__class__(self.data.copy(), self.labels)
        missing = self.__class__.constructs.missing(self.data.columns)
        for svar in missing:
            newdata = (
                svar.builder(newtab)
                .to_pandas(columns=svar.keys, index=newtab.data.index)
                .loc[:, [key for key in svar.keys if key not in newtab.data.columns]]
            )
            newtab = self.__class__(pd.concat([newtab.data, newdata], axis=1), self.labels)

        return newtab

    def __getattr__(self, name: str) -> npt.NDArray | Base:
        if name in self.data.columns:
            return self.data[name].to_numpy()
        elif name in self.__class__.constructs.data.keys():
            con: SVar = self.__class__.constructs[name]
            return con.obj(self.data.loc[:, con.keys])
        elif name in self.labels.lgs:
            return Slicer(self.labels[name], self)
        else:
            raise AttributeError(f"Unknown column or construct {name}")

    def to_dict(self):
        # TODO - add labels
        return self.data.to_dict(orient="records")

    @classmethod
    def from_dict(Cls, data):
        if "data" in data:
            data = data["data"]
        
        return Cls.build(data)

    def __len__(self):
        return len(self.data)

    @property
    def duration(self):
        return self.t[-1] - self.t[0]

    def interpolate(self, t: float):
        interpolators = dict(
            Time="linterp",
            Point="linterp",
            Quaternion="slerp",
            Air="linterp",
            Attack="linterp",
        )

        i0 = self.data.index.get_indexer([t], method="ffill")[0]
        i1 = self.data.index.get_indexer([t], method="bfill")[0]
        if i0 == i1:
            return self.iloc[i0]
        if i0 == -1 or i1 == -1:
            raise ValueError(f"Interpolation time {t} is outside the table range")
        t0 = self.data.index[i0]
        t1 = self.data.index[i1]
        loc = i0 + (t - t0) / (t1 - t0)
        new_table = self.__class__.from_constructs(
            *[
                getattr(self, con.name).interpolate(
                    loc, interpolators[con.obj.__name__]
                )
                for con in self.constructs
            ]
        )

        return new_table.label(self.labels)

    def __getitem__(self, sli: Number | slice) -> Self:
        if isinstance(sli, slice):
            middle = self.data.loc[
                slice(
                    self.t[0] if sli.start is None else sli.start,
                    self.t[-1] + self.dt[-1] if sli.stop is None else sli.stop,
                    sli.step,
                )
            ]
            if sli.start is None or sli.start < self.data.index[0]:
                first = None
            else:
                istart = self.data.index.get_indexer([sli.start])[0]
                first = self.interpolate(sli.start) if istart == -1 else None

            if sli.stop is None or sli.stop > self.data.index[-1]:
                last = None
            else:
                iend = self.data.index.get_indexer([sli.stop])[0]
                last = self.interpolate(sli.stop) if iend == -1 else None

            if first is not None:
                middle = pd.concat([first.data, middle], axis=0)
            if last is not None:
                middle.loc[middle.iloc[-1].name, "dt"] = (
                    last.data.t - middle.iloc[-1].t
                ).item()
                middle = pd.concat([middle, last.data], axis=0)

            res = self.__class__(middle)
        elif isinstance(sli, Number):
            if sli <= 0:
                return self.__class__(self.data.iloc[[int(sli)], :])
            i = self.data.index.get_indexer([sli])[0]
            if i == -1:
                res = self.interpolate(sli)
            else:
                res = self.__class__(pd.DataFrame(self.data.iloc[i, :]).T)
        else:
            raise TypeError(f"Expected Number or slice, got {sli.__class__.__name__}")

        return res.label(self.labels.slice(res.t[0], res.t[-1]))

    @property
    def iloc(self):
        @dataclass
        class ILocer:
            def __getitem__(_, sli: Number | slice) -> Table:
                df = self.data.iloc[sli]
                if isinstance(df, pd.Series):
                    df = pd.DataFrame(df).T
                new_table = self.__class__(df)
                return new_table.label(
                    self.labels.slice(new_table.t[0], new_table.t[-1])
                )

        return ILocer()

    def __iter__(self):
        for t in list(self.data.index):
            yield self[t]

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
        return f"{self.__class__.__name__}({','.join([str(l) for l in self.labels.lgs.keys()])},duration={self.duration})"

    def copy(self, *args, **kwargs) -> Self:
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
            new_time = Time(t, dt)
        elif timeoption == "dt":
            new_time = Time(other.t + self[-1].t - other[0].t + other[0].dt, other.dt)

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
        sts: list[Table],
        label_title: str = None,
        label_values: list[str] = None,
        overlap: Literal[0, 1] = 1,
    ) -> Self:
        """Stack a list of Tables on top of each other.
        The overlap is the number of rows to overlap between each st.
        Existing labels will be moved to sublabels if label_title is not None
        otherwise they will be concatenated.
        """
        if len(sts) == 1:
            return sts[0]

        if label_title:
            assert len(label_values) == len(sts)
            sts[0] = sts[0].over_label(label_title, label_values[0])

        newst = sts[0]
        for i, st in enumerate(sts[1:], 1):
            if overlap > 0:
                newst = Cls(newst.data.iloc[:-overlap, :]).label(newst.labels)

            if label_title:
                st = st.over_label(label_title, label_values[i])

            newst = Cls.concatenate(
                [
                    newst,
                    st.shift_time(newst.t_end[-1] - st.data.index[0]),
                ]
            )

        return newst

    @classmethod
    def concatenate(Cls, sts: list[Table]) -> Self:
        """Concatenate a list of Tables and recalculate the timesteps"""
        df = pd.concat([st.data for st in sts], axis=0)
        t = Time.from_t(df.t.to_numpy())
        df.t = t.t
        df.dt = t.dt
        assert df.index.is_monotonic_increasing
        return Cls(df).label(LabelGroups.concat(*[st.labels for st in sts]))

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
                newlg = LabelGroup.read_array(self.time, np.array(value))
            else:
                raise ValueError(f"Unknown type for label {key}")
            newlg = newlg.intersect(self)
            if not newlg.empty:
                if key in labelgroups:
                    raise ValueError(f"Label {key} already exists")
                labelgroups[key] = newlg
        new_lgs = LabelGroups(labelgroups)
        if inplace:
            self.labels = new_lgs
        return self.__class__(self.data, new_lgs)
        

    def over_label(
        self, title: str, value: str, child_groups: list[str] = None
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
        data: pd.DataFrame, test: str = None, offset=False, **kwargs
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
        min_len=0,
    ) -> Self:
        """Copy the labels from template to flown along the index warping path
        If path is None, the labels are copied directly from the template to the flown
        TODO - min_len prevents the labels from being shortened to less than min_len rows,
        even if the label dows not exist in the warping path the order of labels in template
        will be preserved.
        """

        return flown.label(
            **{k: v.transfer(template.t, flown.t, path) for k, v in template.labels.items()}
        )
