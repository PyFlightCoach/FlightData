from dataclasses import dataclass

import geometry as g


@dataclass
class Constants:
    S: float # wing area in m2
    b: float # wing span in m
    c: float # MAC in m
    Cd0: float 
    M: g.Mass # based on flight dynamics reference coord origin at nose.


F3APlane = Constants(
    0.6,
    1.7,
    0.4,
    0.02,
    g.Mass.cuboid(5, 0.5, 0.3, 0.05).offset(g.PX(50)),
)