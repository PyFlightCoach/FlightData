from __future__ import annotations
import numpy as np
import pandas as pd
import numpy.typing as npt
from geometry import Base, Time
from typing import ClassVar, Self, Tuple, Annotated, Literal
from flightdata.base.table.constructs import SVar, Constructs
from numbers import Number
from time import time
from dataclasses import field, dataclass
from .labels import Label, LabelGroup, Slicer


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
    labels: dict[str, LabelGroup] = field(default_factory=lambda: {})

    @property
    def t_end(self):
        return self.data.t + self.data.dt

    @classmethod
    def build(
        Cls,
        data: pd.DataFrame | dict | pd.Series,
        labels: dict[str, LabelGroup] = None,
        fill=True,
        min_len=1,
    ):
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

        bcs = base_cols
        if data.loc[:, bcs].isnull().values.any():
            raise ValueError("nan values in data")

        return Cls(data, labels or {}).populate() if fill else Cls(data, labels or {})

    def populate(self):
        data = self.data.copy()
        missing = self.__class__.constructs.missing(self.data.columns)
        for svar in missing:
            newdata = (
                svar.builder(self)
                .to_pandas(columns=svar.keys, index=self.data.index)
                .loc[:, [key for key in svar.keys if key not in self.data.columns]]
            )
            data = pd.concat([data, newdata], axis=1)

        return self.__class__(data, self.labels)

    def __getattr__(self, name: str) -> npt.NDArray | Base:
        if name in self.data.columns:
            return self.data[name].to_numpy()
        elif name in self.__class__.constructs.data.keys():
            con: SVar = self.__class__.constructs[name]
            return con.obj(self.data.loc[:, con.keys])
        elif name in self.labels:
            return Slicer(self.labels[name], self)
        else:
            raise AttributeError(f"Unknown column or construct {name}")

    def to_csv(self, filename):
        self.data.to_csv(filename, index=False)
        return filename

    def to_dict(self):
        return self.data.to_dict(orient="records")

    @classmethod
    def from_dict(Cls, data):
        if "data" in data:
            data = data["data"]
        return Cls.build(pd.DataFrame.from_dict(data).set_index("t", drop=False))

    def __len__(self):
        return len(self.data)

    @property
    def duration(self):
        return self.data.index[-1] - self.data.index[0]

    @property
    def iloc(self):
        @dataclass
        class ILocer:
            data: Table

            def __getitem__(inner, sli):
                return self.__class__(inner.data.iloc[sli], self.labels)

        return ILocer(self)

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
            return self.iloc(i0)
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

        return new_table.label(
            **(self.labels or {})
        )

    def __getitem__(self, sli):
        if isinstance(sli, slice):
            middle = self.data.loc[
                slice(
                    sli.start if sli.start else self.data.index[0],
                    sli.stop if sli.stop else self.data.index[-1],
                    sli.step,
                )
            ]
            istart = self.data.index.get_indexer([sli.start])[0]
            iend = self.data.index.get_indexer([sli.stop])[0]

            first = self.interpolate(sli.start) if istart == -1 else None
            last = self.interpolate(sli.stop) if iend == -1 else None
            if first is not None:
                middle = pd.concat([first.data, middle], axis=0)
            if last is not None:
                middle.loc[middle.iloc[-1].name, "dt"] = (
                    last.data.t - middle.iloc[-1].t
                ).item()
                middle = pd.concat([middle, last.data], axis=0)

            return self.__class__(middle).label(**(self.labels or {}))
        elif isinstance(sli, Number):
            if sli <= 0:
                return self.__class__(self.data.iloc[[int(sli)], :])
            i = self.data.index.get_indexer([sli])[0]
            if i == -1:
                return self.interpolate(sli)
            else:
                return self.__class__(pd.DataFrame(self.data.iloc[i, :]).T).label(
                    **(self.labels or {})
                )
        else:
            raise TypeError(f"Expected Number or slice, got {sli.__class__.__name__}")

    def __iter__(self):
        for ind in list(self.data.index):
            yield self[ind - self.data.index[0]]

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
        return f"{self.__class__.__name__} Table(duration = {self.duration})"

    def copy(self, *args, **kwargs):
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
        return self.__class__.from_constructs(**new_constructs).label(
            **(self.labels or {})
        )

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
        data = self.data.copy()
        return self.__class__(data.set_index(data.index - data.index[0]))

    @classmethod
    def stack(Cls, sts: list, overlap: int = 1) -> Self:
        """Stack a list of Tables on top of each other.
        The overlap is the number of rows to overlap between each st
        """
        t0 = sts[0].data.index[0]
        sts = [st.zero_index() for st in sts]
        if overlap > 0:
            offsets = np.cumsum([0] + [s0.data.index[-overlap] for s0 in sts[:-1]])
            dfs = [st.data.iloc[:-overlap] for st in sts[:-1]] + [sts[-1].data]
        elif overlap == 0:
            offsets = np.cumsum([0] + [sec.duration + sec.dt[-1] for sec in sts[:-1]])
            dfs = [st.data for st in sts]
        else:
            raise AttributeError("Overlap must be >= 0")

        for df, offset in zip(dfs, offsets):
            df.index = np.array(df.index) - df.index[0] + offset
        combo = pd.concat(dfs)
        combo.index.name = "t"
        combo.index = combo.index + t0
        combo["t"] = combo.index

        return Cls(combo)

    @classmethod
    def concatenate(Cls, sts: list) -> Self:
        """Concatenate a list of Tables"""
        df = pd.concat([st.data for st in sts], axis=0)
        t = Time.from_t(df.t.to_numpy())
        df.t = t.t
        df.dt = t.dt
        return Cls(df)

    def label(self, **kwargs: dict[str, LabelGroup | str | npt.NDArray]) -> Self:
        labelgroups: dict[str, LabelGroup] = {}
        for key, value in kwargs.items():
            newlg: LabelGroup = None
            if isinstance(value, str):
                newlg = LabelGroup({value: Label(self.t[0], self.t_end.iloc[-1])})
            elif isinstance(value, LabelGroup):
                newlg = value
            elif pd.api.types.is_list_like(value):
                newlg = LabelGroup.read_array(self, np.array(value))
            else:
                raise ValueError(f"Unknown type for label {key}")
            newlg = newlg.intersect(self)
            if not newlg.empty:
                labelgroups[key] = newlg

        return self.__class__(self.data, {k: v for k, v in labelgroups.items()})

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

    def get_label_id(self, test: str = None, **kwargs) -> int | float:
        dfo = Table.labselect(self.unique_labels(), test, **kwargs)
        return dfo.index[0] if len(dfo) > 0 else list(dfo.index)

    def get_subset_df(
        self, test: str | None = None, offset: bool = True, **kwargs
    ) -> pd.DataFrame:
        return Table.labselect(self.data, test, offset, **kwargs)

    def get_label_subset(self, min_len=1, test: str | None = None, **kwargs) -> Self:
        return self.__class__(self.get_subset_df(test, **kwargs), min_len=min_len)

    def get_label_len(self, test: str = None, offset=False, **kwargs) -> int:
        try:
            return len(self.get_subset_df(test, offset, **kwargs))
        except Exception:
            return 0

    def unique_labels(self, cols=None) -> pd.DataFrame:
        if cols is None:
            cols = self.label_cols
        elif isinstance(cols, str):
            cols = [cols]
        return (
            self.data.loc[:, cols]
            .reset_index(drop=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def shift_label(self, offset: int, min_len=None, test=None, **kwargs) -> Self:
        """Shift the end of a label forwards or backwards by offset rows
        Do not allow a label to be reduced to less than min_len"""
        if min_len is None:
            min_len = 1
        ranges = self.label_ranges()

        i = self.get_label_id(test, **kwargs)
        labels: pd.DataFrame = self.labels.copy()
        labcols = [labels.columns.get_loc(c) for c in kwargs.keys()]
        if offset > 0 and i < len(ranges):
            offset = min(offset, ranges.iloc[i + 1, -1] - min_len)
            if offset > 0:
                labels.iloc[
                    ranges.iloc[i + 1].start : ranges.iloc[i + 1].start + offset,
                    labcols,
                ] = pd.Series(kwargs)
        elif offset < 0:
            offset = max(offset, -ranges.iloc[i, -1] + min_len)
            if offset < 0:
                labels.iloc[
                    ranges.iloc[i].end + offset : ranges.iloc[i].end + 1, labcols
                ] = ranges.iloc[i + 1].loc[kwargs.keys()]
        return self.label(**labels.to_dict(orient="list"))

    @classmethod
    def shift_multi(
        Cls, steps: int, tb1: Self, tb2: Self, min_len=1
    ) -> Tuple[Self, Self]:
        """Take datapoints off the start of tb2 and add to the end tb1"""
        # if (steps>0 and len(tb2)-min_len<steps) or (steps<0 and min_len - len(tb1) > steps):
        #    raise ValueError(f'Cannot Squash a Table to less than {min_len} datapoints')
        tj = Cls.stack([tb1, tb2]).shift_label(
            steps, min_len, **dict(tb1.labels.iloc[0])
        )

        return Cls(tj.get_subset_df(**dict(tb1.labels.iloc[0]))), Cls(
            tj.get_subset_df(**dict(tb2.labels.iloc[0]))
        )

    def shift_label_ratio(self, ratio: float, min_len=None, **kwargs) -> Self:
        """shift a label within its allowable bounds, with a ratio of
        1 representing the maximum allowabe movement forwards or backwards
        without squashing a label"""
        ranges = self.label_ranges()
        i = self.get_label_id(**kwargs)
        if ratio > 0:
            limit = ranges.iloc[i + 1, -1] - 2
        else:
            limit = ranges.iloc[i, -1] - 2

        return self.shift_label(int(limit * ratio), min_len, **kwargs)

    def shift_labels_ratios(self, ratios: list[float], min_len: int) -> Self:
        assert len(ratios) == len(self.unique_labels()) - 1
        res = self
        for lab, ratio in zip(
            [r[1] for r in self.unique_labels()[:-1].iterrows()], ratios
        ):
            res = res.shift_label_ratio(ratio, min_len, **lab)
        return res

    def label_range(self, t=False, **kwargs) -> tuple[int]:
        """Get the first and last index of a label.
        If t is True this gives the time, if False it gives the index"""
        labs = self.get_subset_df(**kwargs)
        if not t:
            return self.data.index.get_indexer([labs.index[0]])[
                0
            ], self.data.index.get_indexer([labs.index[-1]])[0]
        else:
            return labs.index[0], labs.index[-1]

    def label_ranges(self, cols: list[str] = None, t=False) -> pd.DataFrame:
        """get the first and last index for each unique label"""
        if cols is None:
            cols = self.label_cols
        df: pd.DataFrame = self.unique_labels(cols)
        res = []
        for row in df.iterrows():
            res.append(list(self.label_range(t=t, **row[1].to_dict())))
        df = pd.concat([df, pd.DataFrame(res, columns=["start", "end"])], axis=1)
        df["length"] = df.end - df.start
        return df

    def single_labels(self) -> list[str]:
        return ["_".join(r[1]) for r in self.data.loc[:, self.label_cols].iterrows()]

    def label_lens(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.split_labels().items()}

    def extract_single_label(self, lab) -> Self:
        labs = np.array(self.single_labels())
        return self.__class__(self.data[labs == lab])

    def split_labels(self, cols: list[str] | str = None) -> dict[str, Self]:
        """Split into multiple tables based on the labels"""
        res = {}
        for label in self.unique_labels(cols).iterrows():
            ld = label[1].to_dict()
            res["_".join(ld.values())] = self.get_label_subset(**ld)
        return res

    def cumulative_labels(self, *cols) -> Self:
        """Return a string concatenation of the requested labels. append an indexer to the end
        of the string for repeat descrete groups of the same label."""
        cols = self.label_cols if len(cols) == 0 else cols
        labs = self.data.loc[:, cols].stack().groupby(level=0).apply("_".join)

        changes = labs.shift() != labs
        new_labels = labs.loc[changes]
        uls = []
        for i, nl in enumerate(new_labels):
            uls.append(sum(new_labels.iloc[:i] == nl))

        df = pd.DataFrame(labs).assign(indexer=np.array(uls)[changes.cumsum() - 1])
        strdf = df.copy()
        strdf["indexer"] = strdf["indexer"].astype(int).astype(str)
        strdf = strdf.stack().groupby(level=0).apply("_".join)
        return strdf.values

    def str_replace_label(self, **kwargs: dict[str, npt.NDArray[np.str_]]) -> Self:
        """perform a string replace for labels"""
        dfo = self.data.copy()
        for k, v in kwargs.items():
            for rep in v:
                dfo[k] = dfo[k].str.replace(*rep)
        return self.__class__(dfo)

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

        return template.label(
            **{k: v.transfer(flown, template, path) for k, v in flown.labels.items()}
        )
