from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal
import geometry as g
from .state import State


type AlignRadiusOption = Literal["full"] | int


@dataclass
class Alignment:
    dist: float
    path: Annotated[npt.NDArray[np.integer], Literal["N", 2]]
    aligned: State

    def plot(self, template: State, step: float, offset: g.Point):
        from plotting import plot_regions
        import plotly.graph_objects as go

        if offset is None:
            offset = g.PY(30)

        tp = template.move(g.Transformation(offset))
        fig = plot_regions(tp, "element")
        fig = plot_regions(self.aligned, "element", fig=fig)

        a = self.aligned.iloc[self.path[::step, 1]].pos + offset / 10
        b = tp.iloc[self.path[::step, 0]].pos - offset / 10
        text = [f"{p[0]},{i},{p[1]}" for i, p in enumerate(self.path[::step])]

        fig.add_traces(
            [
                go.Scatter3d(
                    x=[a.x[i], b.x[i]],
                    y=[a.y[i], b.y[i]],
                    z=[a.z[i], b.z[i]],
                    name=text[i],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="black"),
                )
                for i in range(0, len(self.path), step)
            ]
        )
        return fig

    @staticmethod
    def norm(tp: npt.NDArray, fl: npt.NDArray) -> npt.NDArray[np.floating]:
        
        # remove columns that are constant in the template
        drop_cols = np.isclose(tp,tp[0, :]).all(axis=0)
        tp = tp[:, ~drop_cols]
        fl = fl[:, ~drop_cols]

        #normalise by standard deviation

        #make mean 0
        
        
        return tp, fl

    @staticmethod
    def align(
        flown: State,
        template: State,
        radius: AlignRadiusOption = "full",
        mirror=True,
        weights: g.Point = None,
        tp_weights: g.Point = None,
        norm: bool = False,  # TODO This is not Z norm
    ) -> Alignment:
        """Perform a temporal alignment between two states. return the flown state with labels
        copied from the template along the warped path.
        """
        from fastdtw.fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        weights = weights or g.Point(1, 1.2, 0.5)
        tp_weights = tp_weights or g.Point(0.6, 0.6, 0.6)

        fl = flown.rvel
        tp = template.rvel

                
        def get_brv(brv):
            if mirror:
                brv = g.Point(
                    np.abs(brv.x), brv.y, np.abs(brv.z)
                )
            return brv * weights

        fl = get_brv(flown.rvel)

        tp = get_brv(template.rvel * tp_weights)

        distance, path = fastdtw(
            *Alignment.norm(tp.data, fl.data),            
            radius=len(flown) and len(template) if radius == "full" else radius,
            dist=euclidean,
        )
        path = np.array(path)
        return Alignment(distance, path, State.copy_labels(template, flown, path, 3))
