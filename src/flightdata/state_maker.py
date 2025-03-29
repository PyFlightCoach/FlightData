from .bindata import BinData
from dataclasses import dataclass
import numpy.typing as npt
import geometry as g

@dataclass
class StateMaker(BinData):
    pass

    def t(self, msg: str="ATT"):
        """Return a consistent timestamp, based on a message. """
        tflight = self.dfs[msg].time_flight

        pass

    def create_state(self, t: npt.NDArray):
        
        att = g.Euldeg(self.ATT.Roll, self.ATT.Pitch, self.ATT.Yaw).slerp(self.ATT.timestamp)(t)
        pos = g.GPS(self.POS.Lat, self.POS.Lng, self.POS.Alt).bspline(self.POS.timestamp)(t)
        
        

