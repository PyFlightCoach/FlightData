from flightdata import State
from geometry import Point, Quaternion, Transformation, PX, Time


def test_from_constructs():
    st = State.from_constructs(
        time=Time(5, 1 / 30),
        pos=Point.zeros(),
        att=Quaternion.from_euler(Point.zeros()),
    )
    assert st.pos == Point.zeros()


def test_from_transform():
    st = State.from_transform(Transformation())
    assert st.vel.x == 0

    st = State.from_transform(Transformation(), vel=PX(20))
    assert st.vel.x == 20


