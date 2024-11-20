"""
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations
from typing import Self, Union, IO
import numpy as np
import pandas as pd
from .fields import fields, Field
from geometry import GPS, Point, P0
from geometry.checks import assert_almost_equal
from time import time
from json import load, dump
from flightdata.base.numpy_encoder import NumpyEncoder
from .ardupilot import flightmodes
from flightdata import Origin, fcj
from numbers import Number
from scipy.signal import filtfilt, butter
from datetime import datetime


def filter(data, cutoff=25, order=5, fs=25):
    return filtfilt(
        *butter(order, cutoff, fs=fs, btype="low", analog=False),
        data,
        padlen=len(data) - 1,
    )


class Flight:
    ardupilot_types = [
        "XKF1",
        "XKF2",
        "NKF1",
        "NKF2",
        "POS",
        "ATT",
        "RATE",
        "ACC",
        "GYRO",
        "IMU",
        "ARSP",
        "GPS",
        "RCIN",
        "RCOU",
        "BARO",
        "MODE",
        "RPM",
        "MAG",
        "BAT",
        "BAT2",
        "VEL",
        "ORGN",
        "ESC",
        "CURRENT",
    ]

    def __init__(
        self,
        data: pd.DataFrame,
        parameters: pd.DataFrame = None,
        origin: Origin = None,
        primary_pos_source="pos_c0",
    ):
        self.data = data
        self.parameters = parameters
        self.origin = origin
        self.primary_pos_source = primary_pos_source

    def __getattr__(self, name):
        if self.parameters is not None and 'parameter' in self.parameters.columns:
            if name in self.parameters.parameter.unique():
                df = self.parameters.loc[self.parameters.parameter == name]
                return df.loc[df.value != df.value.shift()]
        cols = getattr(fields, name)
        if cols is None:
            cols = [f for f in self.data.columns if f.startswith(name)]
            if len(cols) > 0:
                return self.data[cols]
        else:
            try:
                if isinstance(cols, Field):
                    return self.data[cols.col]
                else:
                    return self.data.loc[
                        :, [f.col for f in cols if f.col in self.data.columns]
                    ]
            except KeyError:
                if isinstance(cols, Field):
                    return pd.Series(np.full(len(self), np.nan), name=cols.col)
                else:
                    return pd.DataFrame(
                        data=np.full((len(self), len(cols)), np.nan), columns=[f.col for f in cols]
                    )
        raise AttributeError(f"'Flight' object has no attribute '{name}'")

    def make_param_labels(
        self, pname: str, prefix: str = None, suffix: str = None, unknown=""
    ):
        """Make a series with the parameter values at the correct times."""
        ser = pd.Series(np.nan, index=self.data.index, name=pname)
        param = getattr(self, pname)
        ser.iloc[ser.index.get_indexer(param.index, "nearest")] = param.value
        ser = ser.ffill()

        if prefix or suffix:
            sout = pd.Series(unknown, index=self.data.index, name=pname)
            sout[~np.isnan(ser)] = (
                (prefix or "") + ser[~np.isnan(ser)].astype(str) + (suffix or "")
            )
            return sout
        else:
            return ser

    def make_param_df(self, pnames: list[str]):
        """Make a dataframe of parameter values"""
        return pd.DataFrame([self.make_param_labels(p) for p in pnames]).T

    def contains(self, name: Union[str, list[str]]):
        cols = getattr(fields, name)
        if isinstance(cols, Field):
            return name in self.data.columns
        else:
            return [f.column in self.data.columns for f in cols]

    def __getitem__(self, sli: Number | slice) -> Flight:
        if isinstance(sli, Number):
            if sli < 0:
                return self.data.iloc[sli]
            else:
                gl = self.data.index.get_indexer([sli], method="nearest")
                return Flight(
                    self.data.iloc[gl],
                    self.parameters[:sli] if self.parameters else None,
                    self.origin,
                    self.primary_pos_source,
                )
        elif isinstance(sli, slice):
            return Flight(
                self.data.loc[slice(sli.start, sli.stop, sli.step)],
                self.parameters[:sli.stop] if self.parameters else None,
                self.origin,
                self.primary_pos_source,
            )
        else:
            raise TypeError(
                f"Expected a number or a slice, got a {sli.__class__.__name__}"
            )

    def __len__(self):
        return len(self.data)

    def slice_raw_t(self, sli: Number | slice) -> Flight:
        def opp(df: pd.DataFrame, indexer: Number | slice):
            return (
                df.reset_index(drop=True)
                .set_index("time_actual", drop=False)
                .loc[indexer]
                .set_index("time_flight", drop=False)
            )

        return Flight(
            opp(self.data, sli),
            opp(
                self.parameters,
                slice(None, sli if isinstance(sli, Number) else sli.stop, None),
            ),
            self.origin,
            self.primary_pos_source,
        )

    def slice_time_flight(self, sli) -> Flight:
        return Flight(
            self.data.loc[sli],
            self.parameters.loc[: sli if isinstance(sli, Number) else sli.stop],
            self.origin,
            self.primary_pos_source,
        )

    def copy(self, **kwargs) -> Flight:
        return Flight(
            kwargs["data"] if "data" in kwargs else self.data.copy(),
            kwargs["parameters"]
            if "parameters" in kwargs
            else self.parameters.copy()
            if self.parameters is not None
            else None,
            kwargs["origin"] if "origin" in kwargs else self.origin.copy(),
            kwargs["primary_pos_source"]
            if "primary_pos_source" in kwargs
            else self.primary_pos_source,
        )

    def to_dict(self):
        return {
            "data": self.data.to_dict("list"),
            "parameters": self.parameters.to_dict("list"),
            "origin": self.origin.to_dict(),
            "primary_pos_source": self.primary_pos_source,
        }

    @staticmethod
    def from_dict(data: dict) -> Flight:
        return Flight(
            data=pd.DataFrame.from_dict(data["data"]).set_index(
                "time_flight", drop=False
            ),
            parameters=pd.DataFrame.from_dict(data["parameters"]).set_index(
                "time_flight", drop=False
            ),
            origin=Origin.from_dict(data["origin"]),
            primary_pos_source=data["primary_pos_source"],
        )

    def to_json(self, file: str) -> str:
        with open(file, "w") as f:
            dump(self.to_dict(), f, cls=NumpyEncoder, indent=2)
        return file

    @staticmethod
    def from_json(file: str) -> Self:
        return Flight.from_dict(load(open(file, "r")))

    @staticmethod
    def build_cols(**kwargs) -> pd.DataFrame:
        df = pd.DataFrame(columns=list(fields.data.keys()))
        for k, v in kwargs.items():
            df[k] = v
        return df.dropna(axis=1, how="all")

    @staticmethod
    def synchronise(fls: list[Self]) -> list[Self]:
        """Take a list of overlapping flights and return a list of flights with
        identical time indexes. All Indexes will be equal to the portion of the first
        flights index that overlaps all the other flights.
        """
        start_t = max([fl.time_actual.iloc[0] for fl in fls])
        end_t = min([fl.time_actual.iloc[-1] for fl in fls])
        if end_t < start_t:
            raise Exception("These flights do not overlap")
        otf = fls[0].slice_raw_t(slice(start_t, end_t, None)).time_actual

        flos = []
        for fl in fls:
            flos.append(
                fl.copy(
                    data=pd.merge_asof(
                        otf, fl.data.reset_index(), on="time_actual"
                    ).set_index("time_flight", drop=False)
                )
            )

        return flos

    def split_modes(self):
        """Split the flight into segments of the same flight mode.

        Returns:
            list[Flight]: list of flights
        """
        modechanges = (
            (self.flightmode_a.diff().fillna(value=0) != 0).astype(int).cumsum()
        )

        flights = {flightmodes[m]: [] for m in self.flightmode_a.unique()}

        for mode in modechanges.unique():
            _fl = self.data.loc[modechanges == mode, :]
            flights[flightmodes[_fl.flightmode_a.iloc[0]]].append(
                Flight(_fl, self.parameters, self.origin, self.primary_pos_source)
            )
        return flights

    @property
    def duration(self):
        return self.data.iloc[-1].name - self.data.iloc[0].name

    def flying_only(self, minalt=5, minv=10):
        vs = abs(Point(self.velocity))
        above_ground = self.data.loc[(self.gps_altitude >= minalt) & (vs > minv)]

        return self[above_ground.index[0] : above_ground.index[-1]]

    def unique_identifier(self) -> str:
        """Return a string to identify this flight that is very unlikely to be the same as a different flight

        Returns:
            str: flight identifier
        """
        _ftemp = Flight(self.data.loc[self.data.position_z < -10])
        return "{}_{:.8f}_{:.6f}_{:.6f}".format(
            len(_ftemp.data), _ftemp.duration, *self.origin.data[0]
        )

    def __eq__(self, other):
        try:
            pd.testing.assert_frame_equal(self.data, other.data)
            assert_almost_equal(self.origin.pos, other.origin.pos)
            assert self.origin.heading == other.origin.heading
            pd.testing.assert_frame_equal(self.parameters, other.parameters)
            return True
        except Exception:
            return False

    def boot_time(self):
        timestamp = self.time_actual.iloc[0]
        return datetime.fromtimestamp(timestamp) if not np.isnan(timestamp) else None

    @staticmethod
    def from_log(
        log: str,
        extra_types: list[str] = None,
        ppsource: str = "pos",
        imu_instance=0,
        **kwargs,
    ) -> Flight:
        """Constructor from an ardupilot bin file.
        ppsource = xkf or pos
        """
        parser = log
        if not hasattr(log, "dfs"):
            from ardupilot_log_reader.reader import Ardupilot

            parser = Ardupilot.parse(
                str(log),
                types=list(
                    set(
                        Flight.ardupilot_types + []
                        if extra_types is None
                        else extra_types
                    )
                ),
            )

        params = Flight.build_cols(
            time_actual=parser.PARM.timestamp,
            time_flight=parser.PARM.TimeUS / 1e6,
            parameter=parser.PARM.Name,
            value=parser.PARM.Value,
        ).set_index("time_flight", drop=False)

        if params.loc[params.parameter == "AHRS_EKF_TYPE"].iloc[0].value == 2:
            ekf1 = "NKF1"
            ekf2 = "NKF2"
        else:
            ekf1 = "XKF1"
            ekf2 = "XKF2"

        ekf1 = parser.dfs[ekf1] if ekf1 in parser.dfs else None
        ekf2 = parser.dfs[ekf2] if ekf2 in parser.dfs else None

        dfs = []

        if "ATT" in parser.dfs:
            att = parser.ATT.iloc[1:, :]
            dfs.append(
                Flight.build_cols(
                    time_actual=att.timestamp,
                    time_flight=att.TimeUS / 1e6,
                    attitude_roll=np.radians(att.Roll),
                    attitude_pitch=np.radians(att.Pitch),
                    attitude_yaw=np.radians(att.Yaw),
                    attdes_roll=np.radians(att.DesRoll),
                    attdes_pitch=np.radians(att.DesPitch),
                    attdes_yaw=np.radians(att.DesYaw),
                )
            )

        if "POS" in parser.dfs:
            dfs.append(
                Flight.build_cols(
                    time_actual=parser.POS.timestamp,
                    pos_latitude=parser.POS.Lat,
                    pos_longitude=parser.POS.Lng,
                    pos_altitude=parser.POS.Alt,
                )
            )
        else:
            ppsource = ekf1.lower()[:-1]

        ppsource = f"{ppsource}_c{imu_instance}"

        if ekf1 is not None:
            newdfs = Flight.parse_instances(
                ekf1,
                dict(
                    position_N="PN",
                    position_E="PE",
                    position_D="PD",
                    velocity_N="VN",
                    velocity_E="VE",
                    velocity_D="VD",
                ),
                "C",
            )

            for i, df in enumerate(newdfs):
                ekffs = 1 / np.mean(np.diff(df.time_actual))
                ps = "" if i == 0 else f"_{i}"
                df[f"velocity_N{ps}"] = filter(df[f"velocity_N{ps}"], 5, 5, ekffs)
                df[f"velocity_E{ps}"] = filter(df[f"velocity_E{ps}"], 5, 5, ekffs)
                df[f"velocity_D{ps}"] = filter(df[f"velocity_D{ps}"], 5, 5, ekffs)
                dfs.append(df)

        if ekf2 is not None:
            dfs = dfs + Flight.parse_instances(
                ekf2,
                {
                    "wind_N": "VWN",
                    "wind_E": "VWE",
                },
                "C",
            )
        if "RATE" in parser.dfs:
            dfs.append(
                Flight.build_cols(
                    time_actual=parser.RATE.timestamp,
                    axisrate_roll=parser.RATE.R,
                    axisrate_pitch=parser.RATE.P,
                    axisrate_yaw=parser.RATE.Y,
                    desrate_roll=parser.RATE.RDes,
                    desrate_pitch=parser.RATE.PDes,
                    desrate_yaw=parser.RATE.YDes,
                )
            )

        if "IMU" in parser.dfs:
            imu = parser.IMU
            if "I" in imu:
                imu = imu.loc[imu.I == imu_instance, :]

            if ekf1 is not None:
                if "C" in ekf1.columns:
                    imu = pd.merge_asof(
                        imu,
                        ekf1.loc[ekf1.C == imu_instance],
                        on="timestamp",
                        direction="nearest",
                    )
                else:
                    imu = pd.merge_asof(imu, ekf1, on="timestamp", direction="nearest")

                if all([v in imu.columns for v in ["GX", "GY", "GZ"]]):
                    imu["GyrX"] = imu.GyrX + np.radians(imu.GX) / 100
                    imu["GyrY"] = imu.GyrY + np.radians(imu.GY) / 100
                    imu["GyrZ"] = imu.GyrZ + np.radians(imu.GZ) / 100

            if ekf2 is not None:
                if "C" in ekf2.columns:
                    imu = pd.merge_asof(
                        imu,
                        ekf2.loc[ekf2.C == imu_instance],
                        on="timestamp",
                        direction="nearest",
                    )
                else:
                    imu = pd.merge_asof(imu, ekf2, on="timestamp", direction="nearest")
                if all([v in imu.columns for v in ["AX", "AY", "AZ"]]):
                    imu["AccX"] = imu.AccX + imu.AX / 100
                    imu["AccY"] = imu.AccY + imu.AY / 100
                    imu["AccZ"] = imu.AccZ + imu.AZ / 100

            _imufs = 1 / np.mean(np.diff(imu.timestamp))

            dfs.append(
                Flight.build_cols(
                    time_actual=imu.timestamp,
                    acceleration_x=filter(imu.AccX, 10, 5, _imufs),
                    acceleration_y=filter(imu.AccY, 10, 5, _imufs),
                    acceleration_z=filter(imu.AccZ, 10, 5, _imufs),
                    axisrate_roll=filter(imu.GyrX, 10, 5, _imufs),
                    axisrate_pitch=filter(imu.GyrY, 10, 5, _imufs),
                    axisrate_yaw=filter(imu.GyrZ, 10, 5, _imufs),
                )
            )

        if "GPS" in parser.dfs:
            dfs.append(
                Flight.build_cols(
                    time_actual=parser.GPS.timestamp,
                    gps_latitude=parser.GPS.Lat,
                    gps_longitude=parser.GPS.Lng,
                    gps_altitude=parser.GPS.Alt,
                    gps_satellites=parser.GPS.NSats,
                    gps_hdop=parser.GPS.HDop,
                )
            )

        if "MAG" in parser.dfs:
            dfs.append(
                Flight.build_cols(
                    time_actual=parser.MAG.timestamp,
                    magnetometer_x=parser.MAG.MagX,
                    magnetometer_y=parser.MAG.MagY,
                    magnetometer_z=parser.MAG.MagZ,
                )
            )

        if "BARO" in parser.dfs:
            dfs = dfs + Flight.parse_instances(
                parser.BARO,
                dict(air_pressure="Press", air_temperature="Temp", air_altitude="Alt"),
                "I",
            )

        if "ARSP" in parser.dfs:
            dfs = dfs + Flight.parse_instances(
                parser.ARSP, dict(air_speed="Airspeed"), "I"
            )

        if "RCIN" in parser.dfs:
            dfs.append(
                Flight.build_cols(
                    time_actual=parser.RCIN.timestamp,
                    **{
                        f"rcin_c{i}": parser.RCIN[f"C{i}"]
                        for i in range(20)
                        if f"C{i}" in parser.RCIN.columns
                    },
                )
            )

        if "RCOU" in parser.dfs:
            dfs.append(
                Flight.build_cols(
                    time_actual=parser.RCOU.timestamp,
                    **{
                        f"rcout_c{i}": parser.RCOU[f"C{i}"]
                        for i in range(20)
                        if f"C{i}" in parser.RCOU.columns
                    },
                )
            )

        if "MODE" in parser.dfs:
            df = Flight.build_cols(
                time_actual=parser.MODE.timestamp,
                flightmode_a=parser.MODE.Mode,
                flightmode_b=parser.MODE.ModeNum,
                flightmode_c=parser.MODE.Rsn,
            )

            # direction backward sets all the nans to the previous one, fillna backward sets the first bunch of nans to the subsequent
            dfs.append(
                pd.merge_asof(
                    dfs[0].loc[:, "time_actual"],
                    df,
                    on="time_actual",
                    direction="backward",
                ).bfill()
            )

        if "BAT" in parser.dfs:
            dfs = dfs + Flight.parse_instances(
                parser.BAT,
                dict(
                    battery_voltage="Volt",
                    battery_current="Curr",
                    battery_totalcurrent="CurrTot",
                    battery_totalenergy="EnrgTot",
                ),
            )

        if "ESC" in parser.dfs:
            dfs = dfs + Flight.parse_instances(
                parser.ESC,
                dict(motor_voltage="Volt", motor_current="Curr", motor_rpm="RPM"),
            )
        elif "RPM" in parser.dfs:
            dfs.append(
                Flight.build_cols(
                    time_actual=parser.RPM.timestamp,
                    **{
                        f"motor_rpm_{i}": parser.RPM[f"rpm{i}"]
                        for i in range(2)
                        if f"rpm{i}" in parser.RPM.columns
                    },
                )
            )

        for k, v in kwargs.items():
            if k in parser.dfs:
                dfs = dfs + Flight.parse_instances(parser.dfs[k], v)

        dfout = dfs[0]
        dt = dfout.time_actual.diff().max()
        for df in dfs[1:]:
            dfout = pd.merge_asof(
                dfout,
                df,
                on="time_actual",
                direction="nearest",
                tolerance=min(max(dt, df.time_actual.diff().max()), 0.1),
            )

        origin = Origin("ekf_origin", GPS(parser.ORGN.iloc[:, -3:]), 0)

        return Flight(
            dfout.set_index("time_flight", drop=False), params, origin, ppsource
        )

    @staticmethod
    def parse_instances(
        indf: pd.DataFrame, colmap: dict[str, str], instancecol="Instance"
    ):
        """Where an instance column exists in an input df split the values into two columns"""
        instances = (
            sorted(indf[instancecol].unique()) if instancecol in indf.columns else [0]
        )
        dfs = []
        for i in instances:
            _subdf = (
                indf.loc[indf[instancecol] == i, :]
                if instancecol in indf.columns
                else indf
            )

            dfs.append(
                Flight.build_cols(
                    time_actual=_subdf.timestamp,
                    **{
                        f'{k}{f"_{i}" if i > 0 else ""}': _subdf[v]
                        for k, v in colmap.items()
                    },
                )
            )
        return dfs

    @staticmethod
    def from_fc_json(fc_json: fcj.FCJ) -> Flight:
        if fc_json.parameters:
            origin = Origin.from_fcjson_parameters(fc_json.parameters)
            shift = origin.rotation.transform_point(
                Point(
                    fc_json.parameters.moveEast,
                    -fc_json.parameters.moveNorth,
                    0,
                )
            )
        else:
            origin = Origin("dummy_origin", GPS(0, 0, 0), -np.pi / 2)
            shift = P0()

        df = pd.DataFrame([d.__dict__ for d in fc_json.data], dtype=float)

        df = Flight.build_cols(
            time_flight=df["time"] / 1e6,
            attitude_roll=np.radians(df["r"]),
            attitude_pitch=np.radians(df["p"]),
            attitude_yaw=np.radians(df["yw"]),
            position_N=df["N"],
            position_E=df["E"],
            position_D=df["D"],
            velocity_N=df["VN"],
            velocity_E=df["VE"],
            velocity_D=df["VD"],
            wind_N=df["wN"] if "wN" in df.columns else None,
            wind_E=df["wE"] if "wE" in df.columns else None,
        )
        shift = P0() if shift is None else shift
        df["position_N"] = df["position_N"] + shift.x
        df["position_E"] = df["position_E"] + shift.y
        df["position_D"] = df["position_D"] + shift.z
        return Flight(df.set_index("time_flight", drop=False), None, origin, "xkf_c0")

    def remove_time_flutter(self):
        # I think the best option is just to take the average of the timestep.
        # FC loop rate seems consistent but there is noise in the time that is recorded
        # to write, probably because different things are being written at different rates
        time_cols = fields.get_cols(["time"])
        avcols = []
        for col in time_cols:
            _col = self.data.loc[:, col]
            avcols.append(np.linspace(_col.iloc[0], _col.iloc[-1], len(_col)))

        return self.copy(
            data=pd.concat(
                [
                    pd.DataFrame(
                        np.array(avcols).T, columns=time_cols, index=self.data.index
                    ),
                    self.data.loc[
                        :, [c for c in self.data.columns if c not in time_cols]
                    ],
                ],
                axis=1,
            )
            .reset_index(drop=True)
            .set_index("time_flight", drop=False)
        )

    def filter(self, b, a):
        from scipy.signal import filtfilt

        dont_filter = [
            c
            for c in fields.get_cols(["time", "flightmode", "rcin", "rcout"])
            if c in self.data.columns
        ]
        unwrap_cols = [
            c for c in fields.get_cols(["attitude"]) if c in self.data.columns
        ]

        filter_cols = [
            c for c in self.data.columns if c not in dont_filter + unwrap_cols
        ]

        filtdf = filtfilt(b, a, self.data.loc[:, filter_cols], axis=0)
        unwfiltdf = filtfilt(
            b, a, np.unwrap(self.data.loc[:, unwrap_cols], axis=0), axis=0
        )

        return self.copy(
            data=pd.concat(
                [
                    self.data.loc[:, dont_filter],
                    pd.DataFrame(unwfiltdf, columns=unwrap_cols, index=self.data.index)
                    % (2 * np.pi),
                    pd.DataFrame(filtdf, columns=filter_cols, index=self.data.index),
                ],
                axis=1,
            )
        )

    def butter_filter(self, cutoff, order=5):
        from scipy.signal import butter

        ts = self.time_flight.to_numpy()
        N = len(self)
        T = (ts[-1] - ts[0]) / N
        fs = 1 / T
        return self.filter(*butter(order, cutoff, fs=fs, btype="low", analog=False))
