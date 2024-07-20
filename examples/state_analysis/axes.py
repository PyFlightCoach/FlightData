from flightdata import State, Environment
from geometry import Transformation, P0, Euldeg, PY, PX, Point, Time
from flightplotting import plotsec
import numpy as np
from flightdata import Coefficients, Environment, Flow
from flightdata.model import cold_draft as constants
import plotly.express as px


track = State.from_transform(
    Transformation(P0(), Euldeg(180,0,0)), 
    vel=PX(20), rvel=np.pi * Point(0.25, 0.25, 0)
).extrapolate(6, 3)

env = Environment.from_constructs(Time.now(), wind=PY(5))

wind = track.track_to_wind(env)



flow = Flow.build(wind, env)
coeffs = Coefficients.build(wind, flow.q, constants)
flow = flow.rotate(coeffs, 10, 5)
px.line(np.degrees(flow.flow.to_pandas().iloc[:,:-1])).show()

body = wind.wind_to_body(flow)


fig = plotsec([track, wind, body], nmodels=10, scale=3)
fig.show()
