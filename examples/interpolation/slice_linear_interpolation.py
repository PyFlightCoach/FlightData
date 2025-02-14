from flightdata import State
import geometry as g
from plotting import plotsec
import numpy as np
import plotly.graph_objects as go

st = (
    State.from_transform(g.Transformation(), vel=g.PX(30), rvel=g.PY(10))
    .extrapolate(0.2)
    .superimpose_roll(np.radians(180))
)
# st = State.from_dict(load(Path("examples/interpolation/st.json").open()))[425:429]

pass

plotsec(
    dict(base=st, new=st[0.025:0.175].move(g.Transformation(g.PY(1)))),
    nmodels=20,
    scale=0.2,
).show()

st = (
    State.from_transform(g.Transformation(), vel=g.PX(30), rvel=g.PY(10))
    .extrapolate(0.2)
)

st1s = []
st2s = []
for t in np.linspace(st.t[0], st.t[-1], 20):
    st1s.append(st.interpolate(t))
    st2s.append(st.interpolate_kinematic(t))
st1 = State.concatenate(st1s).move(g.Transformation(g.PY(0.5)))
st2 = State.concatenate(st2s).move(g.Transformation(g.PY(1.0)))
fig = plotsec(
    dict(
        original=st,
        linear=st1,
        kinematic=st2,
    ),
    nmodels=20,
    scale=0.1,
)

fig.show()


st3 = State.from_constructs(st2.time, st2.pos, st2.att)

accfig = go.Figure()
for k, v in dict(base=st, linear=st1, kinematic=st2, derived=st3).items():
    accfig.add_trace(go.Scatter(x=v.t, y=v.dw, name=k, mode="lines"))
# px.scatter(st.data).show()
accfig.show()