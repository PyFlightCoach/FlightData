from .table import Table, Constructs, SVar
from .collection import Collection
from .numpy_encoder import NumpyEncoder
from numbers import Number

def to_list(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif obj is None:
        return None
    else:
        return list(obj)
