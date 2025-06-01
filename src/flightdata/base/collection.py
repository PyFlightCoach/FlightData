from typing import Union, Any, Self, TypeVar, Iterable, Callable
import pandas as pd


T = TypeVar('T')


class Collection:
    VType: T = None

    uid = "uid"
    def __init__(self, data: Union[dict[str, T], list[str,T]]=None, check_types=True):
        
        self.data: dict[str, T] = {}
        if isinstance(data, dict):
            self.data = data
        elif isinstance(data, self.__class__):
            self.data = data.data
        elif isinstance(data, self.__class__.VType):
            self.data = {getattr(data, self.__class__.uid): data}
        elif data is None:
            pass
        else:
            self.data = {getattr(d, self.__class__.uid): d for d in data}

        assert all([hasattr(v, self.__class__.uid) for v in self.data.values()])
        if check_types:
            assert all(isinstance(v, self.__class__.VType) for v in self.data.values())

    def __getattr__(self, name) -> T:
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"{name} not found in {self.__class__}")

    def __getitem__(self, key: Union[int, str, slice]) -> Union[Self, T]:
        if isinstance(key, int): 
            return list(self.data.values())[key]
        elif isinstance(key, slice):
            return self.__class__(list(self.data.values())[key])
        elif isinstance(key, str):
            return self.data[key]
        elif isinstance(key, self.__class__.VType):
            return self.data[getattr(key, self.__class__.uid)]
        raise ValueError(f"Invalid Key or Indexer {key}")

    def subset(self, keys: list[str]) -> Self:
        return self.__class__([getattr(self, k) for k in keys])

    def keys(self):
        return self.data.keys()
    
    def items(self):
        return self.data.items()
    
    def values(self):
        return self.data.values()

    def __iter__(self) -> Iterable[T]:
        for v in self.data.values():
            yield v

    def update(self, fun: Callable[[T], T]):
        return self.__class__({k: fun(v) for k, v in self.items()})

    def filter_values(self, fun: Callable[[T], bool]):
        return self.__class__({k: v for k, v in self.items() if fun(v)})

    def filter_keys(self, fun: Callable[[str], bool]):
        return self.__class__({k: v for k, v in self.items() if fun(k)})

    def filter_items(self, fun: Callable[[str, T], bool]):
        return self.__class__({k: v for k, v in self.items() if fun(k, v)})
    
    def index(self, key: str) -> int:
        if key in self.data:
            return list(self.data.keys()).index(key)
        raise KeyError(f"{key} not found in {self.__class__}")

    def to_list(self) -> list[T]:
        return list(self.values())

    def to_dicts(self, *args, **kwargs) -> list[dict[str, Any]]:
        return [v.to_dict(*args, **kwargs) for v in self.data.values()]

    def to_dict(self, *args, **kwargs) -> dict[str, dict[str, Any]]:
        return {k: v.to_dict(*args, **kwargs) for k, v in self.data.items()}

    @classmethod
    def from_dicts(cls, vals: list[dict[str: Any]]) -> Self:
        return cls([cls.VType.from_dict(**v) for v in vals])    

    @classmethod
    def from_dict(cls, vals: dict[str, dict[str, Any]]) -> Self:
        return cls([cls.VType.from_dict(v) for v in vals.values()])
    
    def add(self, v: T | Self, inplace=True) -> Self:
        odata = self.data.copy()
        if isinstance(v, self.VType):
            odata[getattr(v, self.uid)] = v
        elif isinstance(v, self.__class__):
            odata = odata | v.data #dict(**odata, **v.data)
        elif isinstance(v, list):
            odata = odata | {getattr(d, self.uid): d for d in v}
        if inplace:
            self.data = odata
            return v
        else:
            return self.__class__(odata) 
    
    @classmethod
    def concat(Cls, vs: list) -> Self:
        coll = Cls([])
        for v in vs:
            coll.add(v)
        return coll
    
    @classmethod
    def merge(Cls, vs: list[Self]) -> Self:
        coll = vs[0].data
        for v in vs[1:]:
            coll = coll | v.data
        return Cls(coll)

    def add_start(self, v: T | Self, inplace=True) -> Self:
        ocol = self.copy() if not inplace else self
        if isinstance(v, self.VType):
            ocol.data.update({getattr(v, ocol.uid): v})
        elif isinstance(v, self.__class__):
            ocol.data = dict(**v.data, **ocol.data)
        else:
            raise TypeError(f"Expected a {self.__class__.__name__} or a {self.VType}")
        return ocol
    
    def next_free_name(self, prefix: str) -> str:
        i=0
        while f"{prefix}{i}" in self.data:
            i+=1
        else:
            return f"{prefix}{i}"

    def copy(self, deep=True) -> Self:
        return self.__class__([v.copy() for v in self] if deep else self.data.copy())
    
    def __str__(self) -> str:
        return str(pd.Series({k: str(v) for k, v in self.data.items()}))
    
    def __repr__(self) -> str:
        return str(pd.Series({k: repr(v) for k, v in self.data.items()}))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def remove(self, key_or_id: str | int) -> Self:
        if isinstance(key_or_id, int):
            key_or_id = list(self.data.keys())[key_or_id]
        return self.__class__({k: v for k, v in self.data.items() if k != key_or_id})

    def replace(self, key_or_id: str | int, v: T, inplace=False) -> Self:
        if isinstance(key_or_id, int):
            key_or_id = list(self.data.keys())[key_or_id]
        if not hasattr(v, self.uid):
            raise ValueError(f"Value {v} does not have a {self.uid} attribute")
        if not inplace:
            new_data = self.data.copy()
            new_data[key_or_id] = v
            return self.__class__(new_data)
        else:
            if key_or_id in self.data:
                self.data[key_or_id] = v
            else:
                raise KeyError(f"{key_or_id} not found in {self.__class__}")
            return self

    @property
    def uids(self) -> list[str]:
        return list(self.data.keys())