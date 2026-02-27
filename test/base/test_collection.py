from pytest import fixture
from flightdata.base.collection import Collection
from dataclasses import dataclass

@dataclass
class P:
    uid: str

    def to_dict(self):
        return {"uid": self.uid}

    @staticmethod
    def from_dict(d):
        return P(d["uid"])

class Col(Collection):
    VType=P


@fixture()
def col():
    return Col([P("a"), P("b")])


def test_getattr(col: Col):
    assert P("a") == col.a
    assert P("b") == col.b


def test_to_from_dict(col: Col):
    # given a collection
    # when we convert it to a dict and back
    d = col.to_dict()
    col2 = Col.from_dict(d)
    # then we get the same collection back
    
    for k, p in col.items():
        assert p.uid == col2[k].uid
