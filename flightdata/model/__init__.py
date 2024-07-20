

class Model:
    def __init__(self, *parms):
        self.parms = parms
        
    def __call__(self, *args):
        raise NotImplementedError("Model is an abstract class")


from .aerodynamic import SimpleAerodynamic
from .thrust import SimplePropeller

