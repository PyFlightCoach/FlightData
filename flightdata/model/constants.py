from geometry import Point, Mass
from dataclasses import dataclass

@dataclass
class ACConstants:
    s: float
    c: float
    b: float
    mass: Mass
    cg: Point


cold_draft = ACConstants(
    0.569124, 
    0.31211, 
    1.8594, 
    Mass.cuboid(4.5, 800, 400, 100), 
    Point(0.6192,0.0,0.0)
)
