import geometry as g


def interpolate(t0: float, t1: float, p0: g.Point, p1: g.Point, v0: g.Point, v1: g.Point,):
    dt1 = t1 - t0
    m = (p1 - p0 - v0 * dt1 - 0.5 * (v1 - v0)) / \
        ((dt1**3)/6 - (dt1**2)/4)
    b = (v1 - v0) / (dt1**2) - m / 2
    
    def interp(t):
        dt = t - t0
        a = b + m * dt
        v = v0 + b * dt + 0.5 * m * dt**2
        p = p0 + v0 * dt + 0.5 * b * dt**2 + 1/6 * m * dt**3

        return p, v, a
    
    return interp