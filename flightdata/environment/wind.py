import numpy as np
import pandas as pd
from geometry import Point
from typing import List, Tuple


class WindModel:
    def __init__(self, func, kind, args):
        self.func = func
        self.kind = kind
        self.args = args

    def __call__(self, height, time=None):
        return self.func(height, time if not time is None else np.zeros(len(height)))

    @staticmethod
    def zero():
        return WindModel(lambda h, t=0: 0, "zero", [0,0])

class WindModelBuilder:
    def __init__(self, builder, defaults: List[float], bounds: List[Tuple[float]]):
        self.builder = builder
        self.defaults = defaults
        self.bounds = bounds

    def __call__(self, params):
        return self.builder(params)
    
    @staticmethod
    def uniform(minwind=0.1, maxwind=20.0):

        def uniform_wind_builder(args):
            """generates a wind function for constant wind
            
            Args:
                args ([float]): [heading, speed]

            Returns:
                function: function to get wind vector for given altitude and time. 
            """
            assert len(args) == 2
            return WindModel(
                lambda height, time: WindModelBuilder.wind_vector(
                    lambda h: np.full(len(h), args[1]),
                    height, args[0]
                ),
                "uniform",
                args
            )

        return WindModelBuilder(
            uniform_wind_builder,
            [0.0, 3.0],
            [(0.0, 4 * np.pi), (minwind, maxwind)]
        )

    @staticmethod
    def power_law(minwind=0.1, maxwind=20.0):

        def wind_power_law_builder(args):
            """generates a wind function based on a standard wind altitude power law model
            
            Args:
                args ([float]): [heading, speed, exponent]

            Returns:
                function: function to get wind vector for given altitude and time. 
            """
            assert len(args) == 3
            return WindModel(
                lambda height, time: WindModelBuilder.wind_vector(
                    lambda h: args[1] * (h/300)**args[2],
                    height, args[0]
                ),
                "power_law",
                args
            )

        return WindModelBuilder(
                wind_power_law_builder,
                [0.0, 3.0, 0.2],
                [(-np.pi, 3 * np.pi), (minwind, maxwind), (0.01, 0.6)]
            )
        

    @staticmethod
    def fit(minwind=0.1, maxwind=20.0, minh=0, maxh=500, npoints=10, **kwargs ):
        from scipy.interpolate import interp1d
        def wind_fit_builder(args):
            """generates a wind function based on a fit through arbitrary number of points. 

            Args:
                args ([float]): first index heading, rest are speeds up to 1000m
                kind (str): see scipy docs for kind: https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
                            linear, nearest, nearest-up, zero, slinear, quadratic, cubic, previous, or next. zero, slinear, quadratic and cubic 

            Returns:
                function: function to get wind vector for given altitude and time.
            """
            model = interp1d(
                x=np.linspace(minh,np.sqrt(maxh), len(args)-1) ** 2,
                y=args[1:], 
                **kwargs
            ) 
            return WindModel(
                lambda height, time: WindModelBuilder.wind_vector(model, height, args[0]),
                "fit",
                args
            )
            
        return WindModelBuilder(
            wind_fit_builder,
            [3.0 for _ in range(npoints)],
            [(minwind, maxwind) for _ in range(npoints)]
        )


    @staticmethod
    def wind_vector(wind_speed_model, height, heading):
        """create a Point or Points representing the wind vector based on a wind speed model"""
        speed = wind_speed_model(height)
        direc =  Point(np.cos(heading), np.sin(heading), 0.0)
        
        if type(height) in [list, np.ndarray]:
            return direc.tile(len(speed)) * speed
        else:
            return direc * float(speed)

