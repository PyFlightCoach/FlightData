from .table import Table, Constructs, SVar
from .collection import Collection
from .numpy_encoder import NumpyEncoder

def to_list(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return list(obj)
