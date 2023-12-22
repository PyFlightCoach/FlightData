

from flightdata import *
import geometry as g
from flightplotting import plotsec
import numpy as np

fl = Flight.from_json('examples/flight_dynamics/00000150.json').flying_only()
origin = Origin.from_f3a_zone('examples/flight_dynamics/box.f3a')

body = State.from_flight(fl, origin)

controls = fl.rcin.iloc[:,[0,1,3,4]]
controls.columns = list('taer')


S = 0.5
c=0.3

mass = g.Mass.cuboid(5, 0.5, 0.3, 0.2)

def test_model(parms):
    a= SimplePropeller.nparms
    b=SimpleAerodynamic.nparms
    
    thrustmodel = SimplePropeller(*parms[:a])
    aeromodel = SimpleAerodynamic(*parms[a:a+b])

    env = Environment.from_constructs(
        body.time,
        Air.iso_sea_level(len(body)),
        g.Point(parms[a+b], parms[a+b+1], 0).tile(len(body))
    )

    flow = Flow.build(body, env)

    thrust = thrustmodel(flow.aspd.x, controls.t)
    coeff = aeromodel(body.time, flow.alpha, flow.beta, controls.a, controls.e, controls.r)
    
    aero_forces = coeff.force / (flow.q * S)
    aero_moments = coeff.moment / (flow.q * S * g.Point(1, c, 1))
    
    measured_forces = body.F_inertia(mass) + body.F_gravity(mass)
    measured_moments = body.M_inertia(mass)
    
    cost = sum(abs(measured_forces - aero_forces)) + sum(abs(measured_moments - aero_moments))
    print(cost)
    return cost

initial = SimplePropeller.initial + SimpleAerodynamic.initial + [0.0, 0.0]
print(test_model(initial))

from scipy.optimize import minimize

res = minimize(test_model, np.zeros(SimplePropeller.nparms + SimpleAerodynamic.nparms + 2 ))
print(res.x)