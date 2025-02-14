from flightdata import State
import geometry as g
import numpy as np

radius=20 #m
duration = 10 #s
u = np.pi * radius / duration # rad/s, half a loop in 10 seconds

q = np.pi / duration

st = State.from_transform(
    g.Transformation(g.Euler(np.pi, 0, 0)), vel=g.PX(u), rvel=g.PY(q)
)


for i in [2, 3, 10, 100]:
    stn = st.fill(g.Time.from_t(np.linspace(0, duration, i)))
    stn.plot().show()