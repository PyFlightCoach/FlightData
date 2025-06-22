import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Parameters:
    parms: dict[str, pd.DataFrame]

    def __getattr(self, name: str):
        return self.parms[name]
    
