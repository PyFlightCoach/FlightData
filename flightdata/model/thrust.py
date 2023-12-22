from . import Model

class SimplePropeller(Model):
    nparms = 2
    def __call__(self, u, pin):
        c = self.parms
        return c[0] + c[1] * pin / u


    initial = [0.0, -1.0]