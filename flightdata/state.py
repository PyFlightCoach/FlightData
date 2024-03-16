from __future__ import annotations
from typing import Union, List, Tuple, Self
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
import geometry as g
from flightdata import Table, Constructs, SVar, Origin, Flow, Environment


class State(Table):
    constructs = Table.constructs + Constructs([
        SVar("pos", g.Point,       ["x", "y", "z"]           , lambda self: g.P0(len(self))), 
        SVar("att", g.Quaternion,  ["rw", "rx", "ry", "rz"]  , lambda self : g.Q0(len(self))),
        SVar("vel", g.Point,       ["u", "v", "w"]           , lambda st: g.P0() if len(st)==1 else st.att.inverse().transform_point(st.pos.diff(st.dt))),
        SVar("rvel", g.Point,       ["p", "q", "r"]           , lambda st: g.P0() if len(st)==1 else st.att.body_diff(st.dt)),
        SVar("acc", g.Point,       ["du", "dv", "dw"]        , lambda st : g.P0() if len(st)==1 else st.att.inverse().transform_point(st.att.transform_point(st.vel).diff(st.dt) + g.PZ(9.81, len(st)))),
        SVar("racc", g.Point,       ["dp", "dq", "dr"]        , lambda st: g.P0() if len(st)==1 else st.rvel.diff(st.dt)),
    ])
    _construct_freq = 30

    @property
    def transform(self):
        return g.Transformation.build(self.pos, self.att)
    
    @property
    def back_transform(self):
        return g.Transformation(-self.pos, self.att.inverse())

    @staticmethod
    def from_transform(transform: g.Transformation=None, **kwargs) -> State:
        if transform is None:
            transform = g.Transformation()
        if "time" not in kwargs: 
            kwargs["time"] = g.Time.from_t(np.linspace(0, State._construct_freq*len(transform), len(transform)))
        return State.from_constructs(pos=transform.p, att=transform.q, **kwargs)

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
            return self.back_transform.g.Point(pin)

    def fill(self, time: g.Time) -> State:
        '''Project forward through time assuming small angles and uniform circular motion'''
        st = self[-1]
        vel = st.vel.tile(len(time))   
        rvel = st.rvel.tile(len(time))
        att = st.att.body_rotate(rvel * time.t)
        pos = g.Point.concatenate([
            g.P0(), 
            (att.transform_point(vel) * time.dt).cumsum()[:-1]
        ]) + st.pos
        return State.from_constructs(time,pos, att, vel, rvel)
    

    def extrapolate(self, duration: float, min_len=3) -> State:
        """Extrapolate the input state assuming uniform circular motion and small angles
        """
        npoints = np.max([int(np.ceil(duration / self.dt[0])), min_len])
        time = g.Time.from_t(np.linspace(0,duration, npoints))
        return self.fill(time)

    @staticmethod
    def from_csv(filename) -> State:
        df = pd.read_csv(filename)

        if "time_index" in df.columns: # This is for back compatability with old csv files where time column was labelled differently
            if "t" in df.columns:
                df.drop("time_index", axis=1)
            else:
                df = df.rename({"time_index":"t"}, axis=1)
        return State(df.set_index("t", drop=False))


    @staticmethod
    def from_flight(flight, origin: Union[Origin, str] = None) -> State:
        """Read position and attitude directly from the log(after transforming to flightline)"""
        #flight = flight.remove_time_flutter()
        if isinstance(origin, str):
            extension = Path(origin).split()[1]
            if extension == "f3a":
                origin = Origin.from_f3a_zone(origin)
            elif extension == "json":
                origin = Origin.from_json(origin)
        elif origin is None:
            origin = flight.origin

        time = g.Time.from_t(np.array(flight.data.time_flight))

        if all(flight.contains('gps')) and flight.primary_pos_source == 'gps':
            pos = origin.rotation.transform_point(g.GPS(flight.gps) - origin.pos[0])
        else: 
            pos = origin.rotation.transform_point(
                flight.origin.pos.offset(g.Point(flight.position)) - origin.pos[0]
            )
        
        att = origin.rotation * g.Euler(flight.attitude) 
        vel =  att.inverse().transform_point(origin.rotation.transform_point(g.Point(flight.velocity))) if all(flight.contains('velocity')) else None
        rvel = g.Point(flight.axisrate) if all(flight.contains('axisrate')) else None
        acc = g.Point(flight.acceleration) if all(flight.contains('acceleration')) else None
        
        return State.from_constructs(time, pos, att, vel, rvel, acc)

    @staticmethod
    def align(
        flown: State, 
        template: State, 
        radius=5, mirror=True,
        weights = g.Point(1,1.2,0.5),
        tp_weights = g.Point(0.6,0.6,0.6),
    ) -> Tuple(float, Self):
        """Perform a temporal alignment between two sections. return the flown section with labels 
        copied from the template along the warped path. 
        """
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        def get_brv(brv):
            if mirror:
                brv = brv.abs() * g.Point(1, 0, 1) + brv * g.Point(0, 1, 0 )
            return brv * weights

        fl = get_brv(flown.rvel)

        tp = get_brv(template.rvel * tp_weights)

        distance, path = fastdtw(
            tp.data,
            fl.data,
            radius=radius,
            dist=euclidean
        )

        return distance, State.copy_labels(template, flown, path, 3)

    
    
    def splitter_labels(self: State, mans: List[dict], better_names: List[str]=None) -> State:
            """label the manoeuvres in a State based on the flight coach splitter information

            TODO this assumes the state only contains the dataset contained in the json

            Args:
                mans (list): the mans field of a flight coach json
                better_names: names to replace the splitter names with. does not include takeoff or landing.

            Returns:
                State: State with labelled manoeuvres
            """

                
            takeoff = self.data.iloc[0:int(mans[0]["stop"])+1]

            labels = [mans[0]["name"]]
            labelled = [State(takeoff).label(manoeuvre=labels[0])]
            if better_names:
                better_names.append('land')

            for i, split_man in enumerate(mans[1:]):
                
                while split_man["name"] in labels:
                    split_man["name"] = split_man["name"] + "2"


                name = better_names[i] if better_names else split_man["name"]

                labelled.append(
                    State(
                        self.data.iloc[int(split_man["start"]):int(split_man["stop"])+1]
                    ).label(manoeuvre=name)
                )
                labels.append(split_man["name"])

            return State.stack(labelled)

    
    def get_subset(self: State, mans: Union[list, slice], col="manoeuvre", min_len=1) -> Self:
        selectors = self.data.loc[:,col].unique()
        if isinstance(mans, slice):
            mans = selectors[mans]

        if not is_list_like(mans):
            mans = [mans]
        
        if not all(isinstance(m, str) for m in mans):
            mans = [m.uid if m.__class__.__name__ == "Manoeuvre" else m for m in mans]
            mans = [m.uid if m.__class__.__bases__[0].__name__ == "El" else m for m in mans]    
            mans = [selectors[m] if isinstance(m, int) else m for m in mans]
            
        assert all(isinstance(m, str) for m in mans)

        return State(self.data.loc[self.data.loc[:, col].isin(mans)], False, min_len)

    def get_manoeuvre(self: State, manoeuvre: Union[str, list, int]) -> Self:
        return self.get_subset(manoeuvre, "manoeuvre")

    def get_element(self: State, element: Union[str, list, int]) -> Self:
        return self.get_subset(element, "element") 
        
    def get_man_or_el(self: State, el: str) -> Self:
        if el in self.data.element.unique():
            return self.get_element([el])
        elif el in self.data.manoeuvre.unique():
            return self.get_manoeuvre([el])
        
    def get_meid(self: State, manid: int, elid: int=None):
        st = self.get_manoeuvre(manid)
        if not elid is None:
            return st.get_element(elid)
        else:
            return st

    def convert_state(self: State, r: g.Point) -> State:
        """Rotate body axis by an axis angle"""
        att = self.att.body_rotate(r)
        q =  att.inverse() * self.att
        return State.copy_labels(self, State.from_constructs(
            time=self.time,
            pos=self.pos,
            att=att,
            vel=q.transform_point(self.vel),
            rvel=q.transform_point(self.rvel),
            acc=q.transform_point(self.acc),
            racc=q.transform_point(self.racc),
        ))

    def scale(self: State, factor: float) -> State:
        return State.copy_labels(self, State.from_constructs(
            time=self.time,
            pos=self.pos * factor,
            att=self.att,
            vel=self.vel * factor,
            rvel=self.rvel,
            acc=self.acc * factor,
            racc=self.racc,
        ))

    def mirror_zy(self: State) -> State:
        att = g.Quaternion.from_euler((self.att.to_euler() + g.Point(0, 0, np.pi)) * g.Point(-1, 1, -1))
        return State.copy_labels(self, State.from_constructs(
            time=self.time,
            pos=self.pos * g.Point(-1, 1, 1),
            att=att,#g.Quaternion(self.att.w, self.att.x, -self.att.y, -self.att.z),
            vel=self.vel
        ))

    def to_track(self: State) -> State:
        """This rotates the body so the x axis is in the velocity vector"""
        return self.body_to_wind()

    def body_to_stability(self: State, flow: Flow=None) -> State:
        if not flow:
            env = Environment.from_constructs(self.time)
            flow = Flow.build(self, env)
        return self.convert_state(-g.Point(0,1,0) * flow.alpha)    

    def stability_to_wind(self: State, flow: Flow=None) -> State:
        if not flow:
            env = Environment.from_constructs(self.time)
            flow = Flow.build(self, env)
        return self.convert_state(g.Point(0,0,1) * flow.beta)

    def body_to_wind(self: State, flow: Flow=None) -> State:
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
        yaw_rotation = (jwind + self.vel).angles(
            g.PX()
        ) * g.Point(0,0,1) 

        #transform the data by this yaw rotation:
        int_axis = self.convert_state(yaw_rotation)

        #the local wind vector in the intermediate frame:
        intwind = int_axis.att.inverse().transform_point(env.wind) 

        #the pitch rotation required to align the xy plane with the local wind vector:
        pitch_rotation = (intwind + int_axis.vel).angles(g.PX()) * g.Point(0,1,0)

        #transform by this pitch rotation to get the wind axis state
        return int_axis.convert_state(pitch_rotation)

    def wind_to_body(self: State, flow: Flow) -> State:

        stability_axis = self.convert_state(-g.Point(0,0,1) * flow.beta)
        body_axis = stability_axis.convert_state(g.Point(0,1,0) * flow.alpha)

        return body_axis

    def _create_json_data(self: State) -> pd.DataFrame:

        wvels = self.transform.rotate(self.vel)

        transform = g.Transformation.from_coords(
            g.Coord.from_xy(g.Point(0, 0, 0), g.Point(1, 0, 0), g.Point(0, 1, 0)),
            g.Coord.from_xy(g.Point(0, 0, 0), g.Point(1, 0, 0), g.Point(0, -1, 0))
        )
        eul = transform.rotate(self.att).to_euler()

        fcd = pd.DataFrame(
            data = dict(
                time = self.t * 1e6,
                N=self.x, 
                E=-self.y, 
                D=-self.z, 
                VN=wvels.x, 
                VE=-wvels.y, 
                VD=-wvels.z, 
                r=np.degrees(eul.x), 
                p=np.degrees(eul.y), 
                yw=np.degrees(eul.z), 
                wN=np.zeros(len(self)), 
                wE=np.zeros(len(self)), 
                roll=eul.x, 
                pitch=eul.y, 
                yaw=eul.z
            ),
        )

        return fcd

    def _create_json_mans(self: State, kfactors: list[int]) -> pd.DataFrame:
        
        mans = pd.DataFrame(columns=["name", "id", "sp", "wd", "start", "stop", "sel", "background", "k"])
        mnames = self.data.manoeuvre.unique()
        mans["name"] = mnames
        mans["k"] = kfactors
        mans["id"] = ["sp_{}".format(i) for i in range(len(mnames))]

        mans["sp"] = list(range(len(mnames)))
        
        itsecs = [self.get_manoeuvre(mn) for mn in mnames] 

        mans["wd"] = [100 * st.duration / self.duration for st in itsecs]
        
        dat = self.data.reset_index(drop=True)

        mans["start"] = [dat.loc[dat.manoeuvre==mn].index[0] for mn in mnames]

        mans["stop"] = [dat.loc[dat.manoeuvre==mn].index[-1] + 1 for mn in mnames]
            
        mans["sel"] = np.full(len(mnames.data), False)
        mans.loc[1,"sel"] = True
        mans["background"] = np.full(len(mnames), "")

        return mans

    def create_fc_json(self: State, kfactors: list[int], schedule_name: str, schedule_category: str="F3A"):
        fcdata = self._create_json_data()
        fcmans = self._create_json_mans(kfactors)
        return {
            "version": "1.3",
            "comments": "DO NOT EDIT\n",
            "name": schedule_name,
            "view": {
                "position": {
                    "x": -120,
                    "y": 130.50000000000003,
                    "z": 264.99999999999983
                },
                "target": {
                    "x": -22,
                    "y": 160,
                    "z": -204
                }
            },
            "parameters": {
                "rotation": -1.5707963267948966,
                "start": int(fcmans.iloc[1].start),
                "stop": int(fcmans.iloc[1].stop),
                "moveEast": 0.0,
                "moveNorth": 0.0,
                "wingspan": 3,
                "modelwingspan": 25,
                "elevate": 0,
                "originLat": 0.0,
                "originLng": 0.0,
                "originAlt": 0.0,
                "pilotLat": "0.0",
                "pilotLng": "0.0",
                "pilotAlt": "0.00",
                "centerLat": "0.0",
                "centerLng": "-0.1",
                "centerAlt": "0.00",
                "schedule": [schedule_category, schedule_name]
            },
            "scored": False,
            "scores": [0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,600],
            "mans": fcmans.to_dict("records"),
            "data": fcdata.to_dict("records")
        }


    def direction(self):
        """returns 1 for going right, -1 for going left"""
        return np.sign(self.att.transform_point(g.Point(1, 0, 0)).x)
    
    def cross_direction(self):
        """returns 1 for going out, -1 for coming in"""
        return np.sign(self.att.transform_point(g.Point(1, 0, 0)).y)
    

    def inverted(self):
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
                    template_itrans.att.transform_point(g.PY()) 
                ).rotation_matrix()
            ).inverse()
        )


    def move(self: State, transform: g.Transformation) -> State:
        return State.copy_labels(self, State.from_constructs(
            time=self.time,
            pos=transform.point(self.pos),
            att=transform.rotate(self.att),
            vel=self.vel,
            rvel=self.rvel,
            acc=self.acc,
            racc=self.racc,
        ))

    def move_back(self:State, transform:g.Transformation) -> State:
        self = self.move(g.Transformation(-transform.pos, g.Q0()))
        return self.move(g.Transformation(g.P0(), transform.att.inverse()))

    def relocate(self:State, start_pos: g.Point) -> State:
        offset = start_pos - self.pos[0]
        return self.move(g.Transformation(offset, g.Q0()))

    def superimpose_angles(self: State, angles: g.Point, reference:str="body"): 
        assert reference in ["body", "world"]

        if reference == "body":
            rot = g.Quaternion.from_axis_angle(angles).inverse()
            return State.copy_labels(
                self, 
                State.from_constructs(
                    self.time,
                    self.pos,
                    self.att.body_rotate(angles),
                    vel = rot.transform_point(self.vel),
                    #rvel = rot.transform_point(self.rvel), # need to differentiate angles and add here
                    acc = rot.transform_point(self.acc),
                )
            ) 
        else:
            return State.copy_labels(
                self, 
                State.from_constructs(
                    self.time,
                    self.pos,
                    self.att.rotate(angles),
                )
            )

    def superimpose_rotation(self: State, axis: g.Point, angle: float, reference:str="body"):
        """Generate a new section, identical to self, but with a continous rotation integrated
        """
        t = self.time.t - self.time.t[0]
        rate = angle / self.time.t[-1]
        superimposed_rotation = t * rate

        angles = axis.unit().tile(len(t)) * superimposed_rotation

        return self.superimpose_angles(angles, reference)


    def superimpose_roll(self: State, angle: float) -> State:
        """Generate a new section, identical to self, but with a continous roll integrated

        Args:
            angle (float): the amount of roll to integrate
        """
        return self.superimpose_rotation(g.PX(), angle)

    def smooth_rotation(self: State, axis: g.Point, angle: float, reference:str="body", w: float=0.25, w2=0.1):
        """Accelerate for acc_prop * t, flat rate for middle, slow down for acc_prop * t.

        Args:
            axis (g.Point): Axis to rotate around.
            angle (float): angle to rotate.
            reference (selfr, optional): rotate about body or world. Defaults to "body".
            acc_prop (float, optional): proportion of total rotation to be accelerating for. Defaults to 0.1.
        """

        t = self.time.t - self.time.t[0]

        T = t[-1]

        V = angle / (T*(1-0.5*w-0.5*w2))  # The maximum rate

        #between t=0 and t=wT
        x = t[t<=w*T]
        angles_0 = (V * x**2) / (2 * w * T)    

        #between t=wT and t=T(1-w)
        y=t[(t>w*T) & (t<=(T-w2*T))]
        angles_1 = V * y - V * w * T / 2
        
        #between t=T(1-w2) and t=T
        z = t[t>(T-w2*T)] - T + w2*T
        angles_2 = V*z - V * z **2 / (2*w2*T) + V*T - V * w2 * T  - 0.5*V*w*T

        angles = g.Point.full(axis.unit(), len(t)) * np.concatenate([angles_0, angles_1, angles_2])

        return self.superimpose_angles(angles, reference)
    
    def zero_g_acc(self):
        return self.att.inverse().transform_point(g.PZ(-9.81)) + self.acc

    def arc_centre(self) -> g.Point:
        acc = g.Point.vector_rejection(self.zero_g_acc(), self.vel)
        with np.errstate(invalid='ignore'):
            return acc.unit() * abs(self.vel) ** 2 / abs(acc)
        
    def F_gravity(self, mass: g.Mass):
        '''Returns the gravitational force in N'''
        return mass.m * self.att.inverse().transform_point(g.PZ(-9.81))
    
    def F_inertia(self, mass: g.Mass):
        '''Returns the inertial force in N'''
        return mass.m * (self.zero_g_acc() + g.Point.cross(self.rvel, self.vel))
    
    def M_inertia(self, mass: g.Mass):
        '''return the inertial moment in N'''
        h = mass.angular_momentum(self.rvel)
        return h.diff(self.dt) + g.Point.cross(self.rvel, h)
    
    