from flightdata import State, Environment
import geometry as g
from plotting import plotsec
import numpy as np
from flightdata import Coefficients, Environment, Flow
from flightdata.constants import Constants, F3APlane
import plotly.express as px

track = State.from_transform(
    g.Transformation(g.Euldeg(180,0,0)), 
    vel=g.PX(20), rvel=np.pi * g.Point(0.25, 0.25, 0)
).extrapolate(6, 3)

env = Environment.from_constructs(g.Time.now(), wind=g.PY(5))

wind = track.track_to_wind(env)

flow = Flow.from_state(wind, env)
coeffs = Coefficients.from_state(wind, flow.q, F3APlane)

flow = flow.rotate(coeffs, 10, 5)

flow.flow.plot().show()

px.line(np.degrees(flow.flow.to_pandas().iloc[:,:-1])).show()

body = wind.wind_to_body(flow)


fig = plotsec([track, wind, body], nmodels=10, scale=3)
fig.show()
