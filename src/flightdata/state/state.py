from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal

import geometry as g
import numpy as np
import numpy.typing as npt
import pandas as pd
from ardupilot_log_reader import Ardupilot
from schemas import fcj

from flightdata import Environment, Flight, Flow, Origin
from flightdata.ardupilot import Field, StateData
from flightdata.base.table import Construct, Label, LabelGroups, Table
from flightdata.state.spline_interpolation import SplineState


class State(Table):
    _constructs: ClassVar[list[Construct]] = Table._constructs + [
        Construct("pos", g.Point, ["x", "y", "z"], lazy=True),
        Construct("att", g.Quaternion, ["rw", "rx", "ry", "rz"], lazy=True),
        Construct("vel", g.Point, ["u", "v", "w"], lazy=True),
        Construct("rvel", g.Point, ["p", "q", "r"], lazy=True),
        Construct("acc", g.Point, ["du", "dv", "dw"], lazy=True),
    ]

    def __init__(
        self,
        time: g.Time,
        pos: g.Point | None = None,
        att: g.Quaternion | None = None,
        vel: g.Point | None = None,
        rvel: g.Point | None = None,
        acc: g.Point | None = None,
        labels: LabelGroups | None = None,
        splines: SplineState | None = None,
    ):
        assert isinstance(time, g.Time), "time must be a g.Time object"
        assert (att is not None and pos is not None) or (splines is not None), (
            "Must have either att and pos or splines"
        )
        self._pos = pos
        self._att = att
        self._vel = vel
        self._rvel = rvel
        self._acc = acc
        self.splines = splines
        super().__init__(time, labels=labels)

    @property
    def pos(self) -> g.Point:
        if self._pos is None:
            if self.splines is not None:
                self._pos = self.splines.pos(self.time.t)
            else:
                self._pos = self.att.inverse().transform_point(
                    g.Point.concatenate([g.P0(), self.vel.diff(self.time.dt)])
                )
        return self._pos

    @property
    def att(self) -> g.Quaternion:
        if self._att is None:
            if self.splines is not None:
                self._att = self.splines.att(self.time.t)
            else:
                self._att = g.Quaternion.concatenate(
                    [g.Q0(), g.Quaternion.from_axis_angle(self.rvel * self.dt)]
                )
        return self._att

    @property
    def vel(self) -> g.Point:
        if self._vel is None:
            if self.splines is not None:
                self._vel = self.splines.vel(self.time.t)
            else:
                self._vel = self.att.inverse().transform_point(
                    self.pos.diff(self.time.dt)
                )
        return self._vel

    @property
    def rvel(self) -> g.Point:
        if self._rvel is None:
            if self.splines is not None:
                self._rvel = self.splines.rvel(self.time.t)
            else:
                self._rvel = self.att.body_diff(self.dt)
        return self._rvel

    @property
    def acc(self) -> g.Point:
        if self._acc is None:
            if self.splines is not None:
                self._acc = self.splines.acc(self.time.t)
            else:
                self._acc = self.att.inverse().transform_point(
                    self.att.transform_point(self.vel).diff(self.dt) + g.PZ(9.81)
                )
        return self._acc

    @cached_property
    def transform(self):
        return g.Transformation.build(self.pos, self.att)

    @cached_property
    def back_transform(self):
        return g.Transformation(-self.pos, self.att.inverse())

    @cached_property
    def wvel(self) -> g.Point:
        return self.att.transform_point(self.vel)

    @cached_property
    def wacc(self) -> g.Point:
        return self.att.transform_point(self.acc)

    @cached_property
    def wacc_zero_g(self) -> g.Point:
        return self.wacc - g.PZ(9.81)

    @cached_property
    def acc_zero_g(self) -> g.Point:
        return self.att.inverse().transform_point(self.wacc_zero_g)

    @cached_property
    def wrvel(self) -> g.Point:
        return self.att.transform_point(self.rvel)

    @cached_property
    def wx(self) -> g.Point:
        return self.att.transform_point(g.PX())

    @cached_property
    def wy(self) -> g.Point:
        return self.att.transform_point(g.PY())

    @cached_property
    def wz(self) -> g.Point:
        return self.att.transform_point(g.PZ())

    @classmethod
    def splice(Cls, sts: list[State]):
        if len(sts) == 1:
            return sts[0]
        if all(st.splines is not None for st in sts):
            sps = sts[0].splines
            assert all(st.splines is sps for st in sts[1:]), (
                "All tables must have the same spline object to splice"
            )

            t = np.unique(np.sort(np.concatenate([st.t for st in sts])))
            return Cls(
                g.Time.from_t(t),
                labels=LabelGroups.concat(*[st.labels for st in sts]),
                splines=sps,
            )
        else:
            return super().splice(sts)

    @staticmethod
    def from_transform(transform: g.Transformation = None, **kwargs) -> State:
        if transform is None:
            transform = g.Transformation()
        if "time" not in kwargs:
            kwargs["time"] = g.Time.from_t(
                np.linspace(0, State._construct_freq * len(transform), len(transform))
            )
        return State(pos=transform.p, att=transform.q, **kwargs)

    def interpolate(self, t: float | npt.NDArray) -> State:
        if self.splines is not None:
            return self.__class__(
                self._get_interpolator("time")(t),
                splines=self.splines,
                labels=self.labels.copy(),
            )
        else:
            return self.__class__(
                *[
                    self._get_interpolator(con.name)(t)
                    if getattr(self, con.raw_name) is not None
                    else None
                    for con in self._constructs
                ],
                labels=self.labels.copy(),
            )

    def create_splines(
        self,
        auto_s: bool = True,
        auto_s_cutoff_freq: float = 5.0,
        replace: bool = False,
    ) -> State:
        if self.splines is not None and not replace:
            return self
        return State(
            self.time,
            splines=SplineState.build(
                StateData(
                    Field(self.time.t, self.att),
                    Field(self.time.t, self.rvel),
                    Field(self.time.t, self.pos),
                    Field(self.time.t, self.vel),
                    Field(self.time.t, self.acc),
                ),
                auto_s,
                auto_s_cutoff_freq,
            ),
            labels=self.labels.copy(),
        )

    def body_to_world(self, pin: g.Point, rotation_only=False) -> g.Point:
        """Rotate a g.Point in the body frame to a g.Point in the data frame

        Args:
            pin (g.Point): g.Point on the aircraft

        Returns:
            g.Point: g.Point in the world
        """
        if rotation_only:
            return self.transform.rotate(pin)
        else:
            return self.transform.point(pin)

    def world_to_body(self, pin: g.Point, rotation_only=False) -> g.Point:
        if rotation_only:
            self.back_transform.rotate(pin)
        else:
            return self.back_transform.point(pin)

    def fill(self, time: g.Time) -> State:
        """Project forward through time assuming uniform circular motion"""
        st = self.iloc[-1]
        t = time.t - time.t[0]

        vel: g.Point = st.vel.tile(len(time))
        rvel = (
            g.point.vector_rejection(self.rvel, self.vel).tile(len(time))
            if self.vel != 0
            else self.rvel.tile(len(time))
        )
        att: g.Quaternion = st.att.body_rotate(rvel * t)
        wvel = att.transform_point(self.vel)
        wrvel = att.transform_point(rvel)

        if self.rvel != 0 and self.vel != 0:
            theta = wrvel * t

            r0 = (
                g.point.cross(wrvel[0], wvel[0]).unit()
                * abs(wvel[0])[0]
                / abs(g.point.vector_rejection(wrvel[0], wvel[0]))[0]
            )

            center = self.pos + r0
            radius = g.Quaternion.from_axis_angle(theta).transform_point(-r0)
            acc = -radius.unit() * abs(wvel[0]) ** 2 / abs(radius) + g.PZ(
                9.81, len(time)
            )
            pos = center + radius
        else:
            pos = self.pos + wvel[0] * t
            acc = g.PZ(9.81, len(time))

        return State(time, pos, att, vel, rvel, att.inverse().transform_point(acc))

    def plot(self, **kwargs):
        from plotting import plotsec

        return plotsec(self, **({"nmodels": 10, "ribb": True} | kwargs))

    def plotlabels(self, label: str, **kwargs):
        from plotting import plot_regions

        return plot_regions(self, label, **kwargs)

    def extrapolate(self, duration: float, min_len=3, freq: float = 25.0) -> State:
        """Extrapolate the input state assuming uniform circular motion and small angles"""
        npoints = np.max([int(np.ceil(duration * freq)), min_len])
        time = g.Time.from_t(np.linspace(0, duration, npoints))
        return self.fill(time)

    def to_dict(
        self, legacy: bool = False, include_data: bool = False
    ) -> dict[str, list[float]] | list[dict[str, float]]:
        if legacy:
            return super().to_dict()
        else:
            data = {
                "contents": "statedata",
                "t": self.time.t.tolist(),
                "labels": self.labels.to_dict()
            }
            if include_data:
                data["data"] = pd.concat(
                    [self.to_dataframe(True, ["time", "pos", "att"])]
                ).to_dict("records")
            
            if self.splines is not None:
                data["splines"] = self.splines.to_dict()
            else:
                data["pos"] = self.pos.to_dict()
                data["att"] = self.att.to_dict()
                data["vel"] = self.vel.to_dict()
                data["rvel"] = self.rvel.to_dict()
                data["acc"] = self.acc.to_dict()
            
            return data

    @classmethod
    def from_dict(Cls, data: list[dict[str, float | str]] | dict[str, dict]) -> State:
        def checkcol(name: str, type, func: Callable):
            _vals = data.get(name, None)
            if _vals is not None:
                return func(_vals)

        if isinstance(data, list):
            data = {"data": data}

        if data.get("splines") is not None:
            return State(
                g.Time.from_t(np.array(data["t"])),
                splines=checkcol("splines", SplineState, SplineState.from_dict),
                labels=checkcol("labels", LabelGroups, LabelGroups.from_dict),
            )
        elif data.get("pos") and data.get("att"):
            return State(
                g.Time.from_t(np.array(data["t"])),
                pos=g.Point.from_dict(data["pos"]),
                att=g.Quaternion.from_dict(data["att"]),
                vel=checkcol("vel", g.Point, g.Point.from_dict),
                rvel=checkcol("rvel", g.Point, g.Point.from_dict),
                acc=checkcol("acc", g.Point, g.Point.from_dict),
                labels=checkcol("labels", LabelGroups, LabelGroups.from_dict),
            )
        else:
            # need to reorder the label columns to make sure the labels are nested correctly
            df = pd.DataFrame.from_dict(data.get("data", data)).set_index(
                "t", drop=False
            )
            if "manoeuvre" in df.columns and "element" in df.columns:
                iman = df.columns.get_loc("manoeuvre")
                iel = df.columns.get_loc("element")
                if iman != -1 and iel != -1:
                    cols = df.columns.to_list()
                    cols[min(iman, iel)] = "manoeuvre"
                    cols[max(iman, iel)] = "element"
                    df = df.reindex(cols, axis=1)
                    df = df.to_dict("records")
            return State.from_df(df)

    #            return super().from_dict(data)

    @staticmethod
    def from_csv(filename) -> State:
        df = pd.read_csv(filename)

        if (
            "time_index" in df.columns
        ):  # This is for back compatability with old csv files where time column was labelled differently
            if "t" in df.columns:
                df.drop("time_index", axis=1)
            else:
                df = df.rename({"time_index": "t"}, axis=1)
        return State(df.set_index("t", drop=False))

    @staticmethod
    def from_flight(
        flight: Flight, origin: Origin | str | None = None, smoothing: bool = False
    ) -> State:
        if isinstance(origin, str):
            extension = Path(origin).split()[1]
            if extension == "f3a":
                origin = Origin.from_f3a_zone(origin)
            elif extension == "json":
                origin = Origin.from_json(origin)
        elif origin is None:
            origin = flight.origin

        if flight.primary_pos_source.startswith("pos"):
            pos = origin.rotation.transform_point(
                g.GPS(flight.pos.ffill().bfill()) - origin.pos
            )
        else:
            pos = origin.rotation.transform_point(
                g.Point(flight.position.ffill().bfill())
            )

        att = origin.rotation * g.Euler(flight.attitude.ffill().bfill())
        time = g.Time.from_t(np.array(flight.data.time_flight))
        if smoothing:
            return State(
                time,
                splines=SplineState.build(
                    StateData(
                        time,
                        Field(time, att),
                    )
                ),
            )
        else:
            return State(
                time,
                pos,
                att,
                (
                    att.inverse().transform_point(
                        origin.rotation.transform_point(
                            g.Point(flight.velocity.ffill().bfill())
                        )
                    )
                    if all(flight.contains("velocity"))
                    else None
                ),
                (
                    g.Point(flight.axisrate.ffill().bfill())
                    if all(flight.contains("axisrate"))
                    else None
                ),
                (
                    g.Point(flight.acceleration.ffill().bfill())
                    if all(flight.contains("acceleration"))
                    else None
                ),
            )

    @staticmethod
    def read_bin(
        binfile: Path | str | Ardupilot | StateData,
        origin: Origin | None = None,
        freq: float = 25.0,
        **kwargs,
    ) -> State:
        """Create a State from a bin file"""

        _data = (
            binfile
            if isinstance(binfile, StateData)
            else StateData.parse_bin(binfile, origin)
        )
        return State(
            g.Time.from_t(
                np.linspace(_data.t0, _data.t1, int((_data.t1 - _data.t0) * freq))
            ),
            splines=SplineState.build(_data, **kwargs),
        )

    def splitter_labels(
        self: State,
        mans: list[fcj.Man],
        better_names: list[str] | None = None,
        target_col="manoeuvre",
        t0=0,
    ) -> State:
        """label the manoeuvres in a State based on the flight coach splitter information

        TODO this assumes the state only contains the dataset contained in the json

        Args:
            mans (list): the mans field of a flight coach json
            better_names: names to replace the splitter names with. does not include takeoff or landing.

        Returns:
            State: State with labelled manoeuvres
        """
        i0 = self.data.index.get_indexer([t0], "nearest")[0]

        takeoff = self.data.iloc[0 : int(mans[0].stop) + i0 + 1]

        labels = [mans[0].name]
        labelled = [State(takeoff).label(**{target_col: labels[0]})]
        if better_names:
            better_names.append("land")

        for i, split_man in enumerate(mans[1:]):
            while split_man.name in labels:
                split_man.name = split_man.name + "2"

            name = better_names[i] if better_names else split_man.name

            labelled.append(
                State(
                    self.data.iloc[
                        int(split_man.start) + i0 : int(split_man.stop) + i0 + 1
                    ]
                ).label(**{target_col: name})
            )
            labels.append(split_man.name)

        return State.stack(labelled)

    def body_rotate(self: State, r: g.Point) -> State:
        """Rotate by an axis angle"""
        att = self.att.body_rotate(r)
        q = att.inverse() * self.att

        # TODO axis rates need to update, not just rotate
        rvel = q.transform_point(self.rvel)
        if len(r) == len(self):
            rvel = rvel + g.Point.concatenate([g.P0(), r.diff(self.dt, "diff")[:-1]])

        return self.replace(
            pos=self.pos,
            att=att,
            vel=q.transform_point(self.vel),
            rvel=rvel,
            acc=q.transform_point(self.acc),
            splines=None,
        )

    def scale(self: State, factor: float) -> State:
        return self.replace(
            pos=self.pos * factor,
            vel=self.vel * factor,
            acc=self.acc * factor,
            splines=None,
        )

    def mirror_zy(self: State) -> State:
        att = g.Quaternion.from_euler(
            (self.att.to_euler() + g.Point(0, 0, np.pi)) * g.Point(-1, 1, -1)
        )
        return self.replace(pos=self.pos * g.Point(-1, 1, 1), att=att, splines=None)

    def to_track(self: State) -> State:
        """This rotates the body so the x axis is in the velocity vector"""
        return self.body_to_wind()

    def body_to_stability(self: State, flow: Flow = None) -> State:
        if not flow:
            env = Environment(self.time)
            flow = Flow.from_state(self, env)
        return self.body_rotate(-g.Point(0, 1, 0) * flow.alpha)

    def stability_to_wind(self: State, flow: Flow = None) -> State:
        if not flow:
            env = Environment(self.time)
            flow = Flow.from_state(self, env)
        return self.body_rotate(g.Point(0, 0, 1) * flow.beta)

    def body_to_wind(self: State, flow: Flow = None) -> State:
        return self.body_to_stability(flow).stability_to_wind(flow)

    def track_to_wind(self: State, env: Environment) -> State:
        """I think the correct way to go from track axis to wind axis is to do a yaw rotation then a pitch
        rotation, as this keeps the wing vector in the track axis XY plane.

        Args:
            self (State): the track axis data
            env (Environment): the environment

        Returns:
            State: the wind axis data
        """
        # the local wind vector in the track frame:
        jwind = self.att.inverse().transform_point(env.wind)

        # the yaw rotation required to align the xz plane with the local wind vector:
        yaw_rotation = (jwind + self.vel).angles(g.PX()) * g.Point(0, 0, 1)

        # transform the data by this yaw rotation:
        int_axis = self.body_rotate(yaw_rotation)

        # the local wind vector in the intermediate frame:
        intwind = int_axis.att.inverse().transform_point(env.wind)

        # the pitch rotation required to align the xy plane with the local wind vector:
        pitch_rotation = (intwind + int_axis.vel).angles(g.PX()) * g.Point(0, 1, 0)

        # transform by this pitch rotation to get the wind axis state
        return int_axis.body_rotate(pitch_rotation)

    def wind_to_body(self: State, flow: Flow) -> State:
        stability_axis = self.body_rotate(-g.Point(0, 0, 1) * flow.beta)
        body_axis = stability_axis.body_rotate(g.Point(0, 1, 0) * flow.alpha)

        return body_axis

    def _create_json_data(self: State) -> list[fcj.Data]:
        wvels = self.transform.rotate(self.vel)

        transform = g.Transformation.from_coords(
            g.Coord.from_xy(g.Point(0, 0, 0), g.Point(1, 0, 0), g.Point(0, 1, 0)),
            g.Coord.from_xy(g.Point(0, 0, 0), g.Point(1, 0, 0), g.Point(0, -1, 0)),
        )
        eul = transform.rotate(self.att).to_euler()

        fcd = pd.DataFrame(
            data={
                "time": (self.t * 1e6).astype(int),
                "N": self.x,
                "E": -self.y,
                "D": -self.z,
                "VN": wvels.x,
                "VE": -wvels.y,
                "VD": -wvels.z,
                "r": np.degrees(eul.x),
                "p": np.degrees(eul.y),
                "yw": np.degrees(eul.z),
                "wN": np.zeros(len(self)),
                "wE": np.zeros(len(self)),
                "roll": eul.x,
                "pitch": eul.y,
                "yaw": eul.z,
            },
        )

        return [fcj.Data(**row) for row in fcd.to_dict("records")]

    def _create_json_mans(self: State, kfactors: list[int]) -> list[fcj.Man]:
        mans = []
        for i, (k, label) in enumerate(self.labels.manoeuvre.items()):
            lab: Label = label.to_iloc(self.t)
            mans.append(
                fcj.Man(
                    name=k,
                    k=kfactors[i],
                    id=f"sp_{i}",
                    sp=i,
                    wd=100 * label.width / self.duration,
                    start=lab.start,
                    stop=lab.stop,
                    sel=False,
                    background="",
                )
            )
        return mans

    def create_fc_json(
        self: State,
        kfactors: list[int],
        schedule_name: str,
        schedule_category: str = "F3A",
        scores: list[float] | None = None,
    ) -> fcj.FCJ:
        fcmans = self._create_json_mans(kfactors)
        return fcj.FCJ(
            version="1.3",
            comments="DO NOT EDIT\n",
            name=schedule_name,
            view=fcj.View(
                position={"x": -120, "y": 130.5, "z": 265.0},
                target={"x": -22, "y": 160, "z": -204},
            ),
            parameters=fcj.Parameters(
                rotation=-1.5707963267948966,
                start=int(fcmans[1].start),
                stop=int(fcmans[1].stop),
                moveEast=0.0,
                moveNorth=0.0,
                wingspan=3,
                modelwingspan=25,
                elevate=0,
                originLat=0.0,
                originLng=0.0,
                originAlt=0.0,
                pilotLat="0.0",
                pilotLng="0.0",
                pilotAlt="0.00",
                centerLat="0.0",
                centerLng="-0.1",
                centerAlt="0.00",
                schedule=[schedule_category, schedule_name],
            ),
            scored=scores is not None,
            scores=[
                0,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                600,
            ]
            if scores is None
            else scores,
            mans=fcmans,
            data=self._create_json_data(),
        )

    def direction(self):
        """returns 1 for going right, -1 for going left"""
        return np.sign(self.att.transform_point(g.Point(1, 0, 0)).x)

    def cross_direction(self):
        """returns 1 for going out, -1 for coming in"""
        return np.sign(self.att.transform_point(g.Point(1, 0, 0)).y)

    def inverted(self) -> npt.NDArray:
        """returns true if the aircraft is inverted, false if upright"""
        return self.att.transform_point(g.Point(0, 0, 1)).z > 0

    def upright(self):
        return not self.inverted()

    def judging_itrans(self: State, template_itrans: g.Transformation):
        """The judging initial transform has its X axis in the states velocity vector and
        its wings aligned with the template"""
        return g.Transformation(
            self.pos[0],
            g.Quaternion.from_rotation_matrix(
                g.Coord.from_xy(
                    g.P0(),
                    self.att[0].transform_point(self.vel[0]),
                    template_itrans.att.transform_point(g.PY()),
                ).rotation_matrix()
            ).inverse(),
        )

    def move(self: State, transform: g.Transformation) -> State:
        """Move the state by a transformation"""
        return State(
            time=self.time,
            pos=transform.point(self.pos),
            att=transform.rotate(self.att),
            vel=self.vel,
            rvel=self.rvel,
            acc=self.acc,
        ).label(**self.labels)

    def move_back(self: State, transform: g.Transformation) -> State:
        _self = self.move(g.Transformation(-transform.pos, g.Q0()))
        return _self.move(g.Transformation(g.P0(), transform.att.inverse()))

    def relocate(self: State, start_pos: g.Point) -> State:
        offset = start_pos - self.pos[0]
        return self.move(g.Transformation(offset, g.Q0()))

    def superimpose_angles(
        self: State, angles: g.Point, reference: str = "body"
    ) -> State:
        assert reference in ["body", "world"]

        if reference == "body":
            rot = g.Quaternion.from_axis_angle(angles).inverse()
            return State(
                self.time,
                self.pos,
                self.att.body_rotate(angles),
                vel=rot.transform_point(self.vel),
                rvel=rot.transform_point(self.rvel)
                + angles.diff(self.dt),  # need to differentiate angles and add here
                acc=rot.transform_point(self.acc),
            ).label(**self.labels)
        else:
            att = self.att.rotate(angles)
            inv = att.inverse()
            return State(
                self.time,
                self.pos,
                att,
                inv.transform_point(self.att.transform_point(self.vel)),
                rvel=inv.transform_point(
                    self.att.transform_point(self.rvel) + angles.diff(self.dt)
                ),
                acc=inv.transform_point(self.att.transform_point(self.acc)),
            ).label(**self.labels)

    def superimpose_rotation(
        self: State, axis: g.Point, angle: float, reference: str = "body"
    ) -> State:
        """Generate a new section, identical to self, but with a continous rotation integrated"""

        superimposed_rotation = (
            (self.t - self.t[0])
            * angle
            / (self.duration if self.duration > 0 else self.dt[0])
        )

        angles = axis.unit().tile(len(self)) * superimposed_rotation

        return self.superimpose_angles(angles, reference)

    def superimpose_roll(self: State, angle: float) -> State:
        """Generate a new section, identical to self, but with a continous roll integrated

        Args:
            angle (float): the amount of roll to integrate
        """
        return self.superimpose_rotation(g.PX(), angle)

    def smooth_rotation(
        self: State,
        axis: g.Point,
        angle: float,
        reference: str = "body",
        w: float = 0.25,
        w2=0.1,
    ):
        """Accelerate for acc_prop * t, flat rate for middle, slow down for acc_prop * t.

        Args:
            axis (g.Point): Axis to rotate around.
            angle (float): angle to rotate.
            reference (selfr, optional): rotate about body or world. Defaults to "body".
            acc_prop (float, optional): proportion of total rotation to be accelerating for. Defaults to 0.1.
        """

        t = self.time.t - self.time.t[0]

        T = t[-1]

        V = angle / (T * (1 - 0.5 * w - 0.5 * w2))  # The maximum rate

        # between t=0 and t=wT
        x = t[t <= w * T]
        angles_0 = (V * x**2) / (2 * w * T)

        # between t=wT and t=T(1-w)
        y = t[(t > w * T) & (t <= (T - w2 * T))]
        angles_1 = V * y - V * w * T / 2

        # between t=T(1-w2) and t=T
        z = t[t > (T - w2 * T)] - T + w2 * T
        angles_2 = (
            V * z - V * z**2 / (2 * w2 * T) + V * T - V * w2 * T - 0.5 * V * w * T
        )

        angles = g.Point.full(axis.unit(), len(t)) * np.concatenate(
            [angles_0, angles_1, angles_2]
        )

        return self.superimpose_angles(angles, reference)

    def arc_centre(self) -> g.Point:
        acc = g.point.vector_rejection(self.acc_zero_g, self.vel)
        with np.errstate(invalid="ignore"):
            return acc.unit() * abs(self.vel) ** 2 / abs(acc)

    def curvature(self, axis: g.Point) -> g.Point:
        """Returns the curvature of the path in 1/m, axis is the desired axial direction
        in the world frame"""
        trfl = self.to_track()
        body_curvature = g.point.cross((-trfl.acc_zero_g / trfl.u**2), g.PX())
        world_curvature = trfl.att.transform_point(body_curvature)

        return g.point.scalar_projection(world_curvature, axis)

    #        po.data[-1, :] = np.nan
    #        return po.ffill()

    def F_gravity(self, mass: g.Mass):
        """Returns the gravitational force in N"""
        return mass.m * self.att.inverse().transform_point(g.PZ(-9.81))

    def F_inertia(self, mass: g.Mass):
        """Returns the inertial force in N"""
        return mass.m * (self.acc_zero_g + g.point.cross(self.rvel, self.vel))

    def M_inertia(self, mass: g.Mass):
        """return the inertial moment in Nm"""
        h = mass.angular_momentum(self.rvel)
        return h.diff(self.dt) + g.point.cross(self.rvel, h)

    def get_rotation(self):
        return np.pad(
            np.cumsum(g.point.scalar_projection(self.rvel, self.vel) * self.dt), (1, 0)
        )[:-1]

    def roll_rotation(self):
        return np.pad(np.cumsum(self.rvel.x * self.dt), (1, 0))[:-1]

    def estimate_wind(self):
        """This is a very rough estimate of the wind, it is based on the assumption that
        on average the lateral component of alpha and beta is caused by the wind.
        It will only work for flights with lots of different orientations.
        It also assumes the wind is constant in time and space
        """
        body_slip = g.point.vector_rejection(self.vel, g.PX())
        world_slip = self.att.transform_point(body_slip)
        confidence = abs(g.point.vector_rejection(world_slip, g.PZ())) / abs(self.vel)
        confidence = np.nan_to_num(confidence)
        wind = (world_slip * confidence).sum() / (np.mean(confidence) * len(self))
        return g.point.vector_rejection(wind, g.PZ()), np.mean(confidence)

    def boundary_measure(
        self,
        metric: Literal[
            "measure_rate",
            "measure_speed",
            "measure_length",
            "measure_duration",
            "measure_radius",
        ],
        t0: float,
        t1: float,
        iatt: g.Quaternion,
        axis: g.Point = None,
        delta: float = 0.001,
    ) -> tuple[float, float, float]:

        m = getattr(self[t0:t1], metric)(iatt, axis)
        m0 = getattr(self[t0 - delta : t1], metric)(iatt, axis)
        m1 = getattr(self[t0 : t1 + delta], metric)(iatt, axis)

        return m, (m - m0) / delta, (m1 - m) / delta

    def measure_rate(
        self,
        iatt: g.Quaternion,
        axis: g.Point = None,
    ) -> float | tuple[float, float, float]:
        """measure the roll rate, asuming it started at t0 and finished at t1."""
        return np.sum(self.p * self.dt) / self.duration

    def measure_speed(
        self,
        iatt: g.Quaternion,
        axis: g.Point = None,
    ) -> float:
        """measure the speed of the flown line assuming it started at t0 and finished at t1.
        Linearly interpolate between datapoints
        """
        return np.sum(self.u * self.dt) / self.duration

    def measure_length(self, iatt: g.Quaternion, axis: g.Point = None) -> float:
        """measure the length of the flown line assuming it started at t0 and finished at t1.
        Linearly interpolate between datapoints
        """

        return np.sum(abs(self.vel) * self.dt)

    def measure_duration(self, iatt: g.Quaternion, axis: g.Point = None) -> float:
        """measure the duration of the flown line assuming it started at t0 and finished at t1.
        Linearly interpolate between datapoints
        """
        return self.duration

    def measure_radius(
        self,
        iatt: g.Quaternion,
        axis: g.Point = None,
    ) -> float:

        centre = self.arc_centre()

        bvec = self.att.inverse().transform_point(axis)
        rads = abs(g.point.vector_rejection(centre, bvec))

        angles = np.arctan(abs(self.vel) * self.dt / rads)
        keep = ~np.isnan(rads * angles)

        return np.sum((rads * angles)[keep][:-1]) / np.sum(angles[keep][:-1])

    def measure_roll(self, *args, **kwargs) -> float:
        return self.roll_rotation()[-1]

    def measure_turns(self, *args, **kwargs) -> float:

        return self.roll_rotation()[-1]
