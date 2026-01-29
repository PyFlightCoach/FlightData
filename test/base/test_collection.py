from pytest import fixture
from flightdata.base.collection import Collection
from dataclasses import dataclass

@dataclass
class P:
    uid: str


class Col(Collection):
    VType=P


@fixture()
def col():
    return Col([P("a"), P("b")])


def test_getattr(col: Col):
    assert P("a") == col.a
    assert P("b") == col.b