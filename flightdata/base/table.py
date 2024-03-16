from __future__ import annotations
import numpy as np
import pandas as pd
import numpy.typing as npt
from geometry import Base, Time
from typing import Union, Self, Tuple
from .constructs import SVar, Constructs
from numbers import Number
from time import time


def make_time(tab):
    return Time.from_t(tab.t)
    
    
class Table:
    constructs = Constructs([
        SVar("time", Time,        ["t", "dt"]               , make_time )
    ])

    def __init__(self, data: pd.DataFrame, fill=True, min_len=1):
        if len(data) < min_len:
            raise Exception(f"State constructor length check failed, data length = {len(data)}, min_len = {min_len}")
        self.base_cols = [c for c in data.columns if c in self.constructs.cols()]
        self.label_cols = [c for c in data.columns if not c in self.constructs.cols()]
    
        self.data = data

        self.data.index = self.data.index - self.data.index[0]
        
        if fill:
            missing = self.constructs.missing(self.data.columns)
            for svar in missing:
                
                newdata = svar.builder(self).to_pandas(
                    columns=svar.keys, 
                    index=self.data.index
                ).loc[:, [key for key in svar.keys if key not in self.data.columns]]
                
                self.data = pd.concat([self.data, newdata], axis=1)
            bcs = self.constructs.cols()
        else:
            bcs = self.base_cols
        if np.any(np.isnan(self.data.loc[:,bcs])):
            raise ValueError("nan values in data")
        

    def __getattr__(self, name: str) -> Union[pd.DataFrame, Base]:
        if name in self.data.columns:
            return self.data[name].to_numpy()
        elif name in self.constructs.data.keys():
            con = self.constructs.data[name]
            return con.obj(self.data.loc[:, con.keys])
        else:
            raise AttributeError(f"Unknown column or construct {name}")

    def to_csv(self, filename):
        self.data.to_csv(filename)
        return filename

    def to_dict(self):
        return self.data.to_dict(orient="records")
    
    @classmethod
    def from_dict(Cls, data):
        if 'data' in data:
            data = data['data']
        return Cls(pd.DataFrame.from_dict(data).set_index("t", drop=False))

    def __len__(self):
        return len(self.data)
    
    @property
    def duration(self):
        return self.data.index[-1] - self.data.index[0]

    def __getitem__(self, sli):
        if isinstance(sli, Number):
            if sli<0:
                return self.__class__(self.data.iloc[[int(sli)], :])

            return self.__class__(
                self.data.iloc[self.data.index.get_indexer([sli], method="nearest"), :]
            )
        
        return self.__class__(self.data.loc[sli])

    def slice_raw_t(self, sli):
        inds = self.data.reset_index(names="t2").set_index("t").loc[sli].t2.to_numpy()#set_index("t", drop=False).columns

        return self.__class__(self.data.loc[inds])

    @classmethod
    def stack(Cls, sections: list, overlap: int=1) -> Self:
        """Stack a list of Tables on top of each other. 
        The overlap is the number of rows to overlap between each section
        """
        # first build list of index offsets, to be added to each dataframe
        if overlap > 0:
            offsets = np.cumsum([0] + [sec.data.index[-overlap] for sec in sections[:-1]])
            dfs = [section.data.iloc[:-overlap] for section in sections[:-1]] + [sections[-1].data]
        elif overlap == 0:
            offsets = np.cumsum([0] + [sec.duration + sec.dt[-1] for sec in sections[:-1]])
            dfs = [section.data for section in sections]
        else:
            raise AttributeError("Overlap must be >= 0")

        for df, offset in zip(dfs, offsets):
            df.index = np.array(df.index) - df.index[0] + offset
        combo = pd.concat(dfs)
        combo.index.name = "t"

        combo["t"] = combo.index

        return Cls(combo)

    def __iter__(self):
        for ind in list(self.data.index):
            yield self[ind]

    @classmethod
    def from_constructs(cls, *args,**kwargs) -> Self:
        kwargs = dict(
            **{list(cls.constructs.data.keys())[i]: arg for i, arg in enumerate(args)},
            **kwargs
        )

        df = pd.concat(
            [
                x.to_pandas(
                    columns=cls.constructs[key].keys, 
                    index=kwargs["time"].t
                ) for key, x in kwargs.items() if x is not None
            ],
            axis=1
        )

        return cls(df)

    def __repr__(self):
        return f"{self.__class__.__name__} Table(duration = {self.duration})"

    def copy(self, *args,**kwargs):
        kwargs = dict(kwargs, **{list(self.constructs.data.keys())[i]: arg for i, arg in enumerate(args)}) # add the args to the kwargs
        old_constructs = {key: self.__getattr__(key) for key in self.constructs.existing(self.data.columns).data if not key in kwargs}       
        new_constructs = {key: value for key, value in list(kwargs.items()) + list(old_constructs.items())}
        return self.__class__.from_constructs(**new_constructs).label(**self.labels.to_dict(orient='list'))

    def append(self, other, timeoption:str="dt"):
        if timeoption in ["now", "t"]:
            t = np.array([time()]) if timeoption == "now" else other.t
            dt = other.dt
            dt[0] = t[0] - self.t[-1]
            new_time = Time(t, dt)
        elif timeoption == "dt":
            new_time = Time(other.t + self[-1].t - other[0].t + other[0].dt, other.dt)

        return self.__class__(pd.concat(
            [
                self.data, 
                other.copy(new_time).data
            ], 
            axis=0, 
            ignore_index=True
        ).set_index("t", drop=False))
      
    def label(self, **kwargs) -> Self:
        return self.__class__(self.data.assign(**kwargs))

    @property
    def label_keys(self):
        return self.label_cols
    
    @property
    def labels(self) -> dict[str, npt.NDArray]:
        return self.data.loc[:, self.label_cols]

    def remove_labels(self) -> Self:
        return self.__class__(
            self.data.drop(
                self.label_keys, 
                axis=1, 
                errors="ignore"
            )
        )
    
    def get_subset_df(self, **kwargs) -> pd.DataFrame:
        dfo = self.data
        for k, v in kwargs.items():
            dfo = dfo.loc[dfo[k] == v, :]            
        return dfo

    def get_label_subset(self, min_len=1, **kwargs) -> Self:
        return self.__class__(self.get_subset_df(**kwargs), min_len=min_len)

    def get_label_len(self, **kwargs) -> int:
        try:
            return len(self.get_subset_df(**kwargs))
        except Exception:
            return 0

    def unique_labels(self, cols = None) -> pd.DataFrame:
        '''TODO Fix This'''
        if cols is None:
            cols = self.label_cols
        return self.data.loc[:, cols].reset_index(drop=True).drop_duplicates().reset_index(drop=True)

    def shift_labels(self, col, elname, offset, allow_label_loss=True) -> Self:
        """Move the end of a label forwards or backwards by offset seconds.
        TODO this should be part of shift_label. current implementation doesn't allow
        min label length, and only allows indexing by element"""
        new_t = self.label_ts("element")[elname][1] +  offset
        
        odata = self.data.copy()

        elnames = list(odata[col].unique())
        elid = elnames.index(elname)
        elt = odata.loc[odata[col] == elname].index.to_numpy()       
        nelt = self.data.loc[odata[col] == elnames[elid+1]].index.to_numpy()
        
        if allow_label_loss:
            new_t = max(new_t, self.data.index[0])
            new_t = min(new_t, self.data.index[-1])
        else:
            new_t = max(new_t, elt[0])
            new_t = min(new_t, nelt[-1])

        if elt[-1] > new_t:    
            odata.loc[new_t:nelt[-1], col] = elnames[elid+1]
        else:
            odata.loc[elt[0]:new_t, col] = elname
        
        return self.__class__(odata)

    def shift_label(self, offset: int, min_len=None, **kwargs) -> Self:
        '''Shift the end of a label forwards or backwards by offset rows
        Do not allow a label to be reduced to less than min_len'''
        if min_len is None:
            min_len=1
        ranges = self.label_ranges()
        i = self.get_label_id(**kwargs)
        labels: pd.DataFrame = self.labels.copy()
        labcols = [labels.columns.get_loc(c) for c in kwargs.keys()]
        if offset > 0 and i < len(ranges):
            offset = min(offset, ranges.iloc[i+1, -1] - min_len) 
            if offset > 0:
                labels.iloc[ranges.iloc[i+1].start:ranges.iloc[i+1].start+offset, labcols] = pd.Series(kwargs)
        elif offset < 0:
            offset = max(offset, -ranges.iloc[i, -1] + min_len)
            if offset < 0:
                labels.iloc[ranges.iloc[i].end+offset:ranges.iloc[i].end+1, labcols] = ranges.iloc[i+1].loc[kwargs.keys()]
        return self.label(**labels.to_dict(orient='list'))
    
    @classmethod
    def shift_multi(Cls, steps: int, tb1: Self, tb2: Self, min_len=2) -> Tuple(Self, Self):
        '''Take datapoints off the start of tb2 and add to the end tb1'''
        if (steps>0 and len(tb2)-min_len<steps) or (steps<0 and len(tb1)-2<min_len):
            raise ValueError(f'Cannot Squash a Table to less than {min_len} datapoints')
        dfj = Cls.stack([tb1, tb2], overlap=0).data
        return Cls(dfj.iloc[:len(tb1)+steps, :]), \
               Cls(dfj.iloc[len(tb1)+steps:, :])
    
    def shift_label_ratio(self, ratio: float, min_len=None, **kwargs) -> Self:
        '''shift a label within its allowable bounds, with a ratio of
        1 representing the maximum allowabe movement forwards or backwards
        without squashing a label'''
        ranges = self.label_ranges()
        i = self.get_label_id(**kwargs)
        if ratio > 0:
            limit = ranges.iloc[i+1, -1] - 2
        else:    
            limit = ranges.iloc[i, -1] -2
        
        return self.shift_label(int(limit * ratio), min_len, **kwargs)
    
    def shift_labels_ratios(self, ratios: list[float], min_len: int) -> Self:
        assert len(ratios) == len(self.unique_labels())-1
        res = self
        for lab, ratio in zip([r[1] for r in self.unique_labels()[:-1].iterrows()], ratios):
            res = res.shift_label_ratio(ratio, min_len, **lab)
        return res
    
    def get_label_id(self, **kwargs) -> Union[int, float]:
        dfo = self.unique_labels()
        for k, v in kwargs.items():
            dfo = dfo.loc[dfo[k] == v, :]            
        return dfo.index[0]
    
    def label_range(self, t=False, **kwargs) -> tuple[int]:
        '''Get the first and last index of a label. 
            If t is True this gives the time, if False it gives the index'''
        labs = self.get_subset_df(**kwargs)
        if not t:
            return self.data.index.get_indexer([labs.index[0]])[0], self.data.index.get_indexer([labs.index[-1]])[0]
        else:
            return labs.index[0], labs.index[-1]

    def label_ranges(self, cols: list[str] = None, t=False) -> pd.DataFrame:
        '''get the first and last index for each unique label'''
        if cols is None:
            cols = self.label_cols
        df: pd.DataFrame = self.unique_labels(cols)
        res = []
        for row in df.iterrows():
            res.append(list(self.label_range(t=t,**row[1].to_dict())))
        df =  pd.concat([df, pd.DataFrame(res, columns=['start', 'end'])], axis=1)
        df['length'] = df.end - df.start
        return df
        
    def single_labels(self) -> list[str]:
        return ['_'.join(r[1]) for r in self.data.loc[:, self.label_cols].iterrows()]

    def label_lens(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.split_labels().items()}

    def extract_single_label(self, lab) -> Self:
        labs = np.array(self.single_labels())
        return self.__class__(self.data[labs == lab])

    def split_labels(self, cols=None) -> dict[str, Self]:
        '''Split into multiple tables based on the labels'''
        res = {}
        for label in self.unique_labels(cols).iterrows():
            ld = label[1].to_dict()
            res['_'.join(ld.values())] = self.get_label_subset(**ld)
        return res

    def cumulative_labels(self, *cols) -> Self:
        '''Return a string concatenation of the requested labels. append an indexer to the end
        of the string for repeat descrete groups of the same label.'''
        cols = self.label_cols if len(cols)==0 else cols
        labs = self.data.loc[:, cols].stack().groupby(level=0).apply('_'.join)
        
        changes = labs.shift() != labs
        new_labels = labs.loc[changes]
        uls = []
        for i, nl in enumerate(new_labels):
            uls.append(sum(new_labels.iloc[:i] == nl))
        
        df = pd.DataFrame(labs).assign(indexer = np.array(uls)[changes.cumsum() - 1])
        strdf = df.copy()
        strdf['indexer'] = strdf['indexer'].astype(int).astype(str)
        strdf  = strdf.stack().groupby(level=0).apply('_'.join)
        strdf.loc[df.indexer==0] = df[0]
        return strdf.values

    @staticmethod
    def copy_labels(template: Self, flown: Self, path=None, min_len=0) -> Self:
        """Copy the labels from along the index warping path"""

        flown = flown.remove_labels()

        if path is None:
            return flown.__class__(
                pd.concat(
                    [flown.data.reset_index(drop=True), template.data.loc[:,template.label_cols].reset_index(drop=True)], 
                    axis=1
                ).set_index("t", drop=False)
            )
        else:
            mans = pd.DataFrame(path, columns=["template", "flight"]).set_index("template").join(
                    template.data.reset_index(drop=True).loc[:, template.label_cols]
                ).groupby(['flight']).last().reset_index().set_index("flight")

            st: Self = flown.__class__(flown.data).label(**mans.to_dict(orient='list'))

            if min_len > 0:
                unique_labels = template.unique_labels()
                
                for i, row in unique_labels.iterrows():
                    lens = st.label_lens()
                    labels = st.labels.copy()

                    def get_len(_i: int):
                        key = '_'.join(list(unique_labels.iloc[_i].to_dict().values()))
                        return lens[key] if key in lens else 0
                    
                    def get_range(_i: int):
                        return st.label_range(**unique_labels.iloc[_i].to_dict())

                    if get_len(i) < min_len:

                        max_bck = get_len(i-1) - min_len if i > 0 else 0
                        max_fwd = get_len(i+1) - min_len if i < len(unique_labels)-1 else 0

                        if max_bck + max_fwd + get_len(i) < min_len:
                            raise Exception(f'{row.iloc[0]},{row.iloc[1]} too short and cannot shorten adjacent labels further')
                        else:
                            _extend = (min_len - get_len(i)) / 2
                            ebck = min(max_bck, int(np.floor(_extend)))
                            efwd = min(max_fwd, int(np.floor(_extend))+1)

                            if ebck > 0:
                                rng = get_range(i-1)
                                labels.iloc[rng[1]-ebck:rng[1]+1] = unique_labels.iloc[i]
                            if efwd > 0:
                                rng = get_range(i+1)
                                labels.iloc[rng[0]:rng[0]+efwd+1] = unique_labels.iloc[i]
                            st = st.label(**labels.to_dict(orient='list'))
            return st
    
    