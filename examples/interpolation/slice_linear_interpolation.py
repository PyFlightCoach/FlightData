from flightdata import State
import geometry as g
from plotting import plotsec
import numpy as np
import plotly.express as px
from pathlib import Path
from json import load


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

# fig = plotsec(st, nmodels=20, scale=0.5)
# for t in np.linspace(st.t[0], st.t[-1], 20):
#
#    fig = plotsec(st.interpolate(t), nmodels=1, scale=0.5, fig=fig, color="red")
#    fig = plotsec(st.interpolate_kinematic(t), nmodels=1, scale=0.5, fig=fig, color="green")
# fig.show()

# px.scatter(st.data).show()
