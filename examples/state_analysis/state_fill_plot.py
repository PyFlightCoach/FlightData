from flightdata import State
from geometry import Transformation, P0, Euldeg, PX, Point
from flightplotting import plotsec
from flightplotting.traces import vectors
import numpy as np


judging = State.from_transform(
    Transformation(P0(), Euldeg(180,0,0)), 
    vel=PX(20), rvel=np.pi * Point(0.25, 0.25, 0)
).extrapolate(6, 3)


fig = plotsec(judging, scale=2, nmodels=5)

fig.add_traces(vectors(20, judging, 0.5*judging.body_to_world(judging.acc, True)))

fig.add_traces(vectors(20, judging, 5*judging.body_to_world(judging.rvel, True), line=dict(color="green")))
fig.show()


