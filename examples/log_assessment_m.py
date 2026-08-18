import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    from ardupilot_log_reader import Ardupilot

    import flightdata.ardupilot.messages as msgs
    from flightdata import Origin
    from flightdata.ardupilot import Field
    from flightdata.ardupilot import StateData
    from flightdata.state.spline_interpolation import SplineState
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go


    bin = Ardupilot.parse("../test/data/p23.BIN", cache_file=True)

    origin = Origin.from_f3a_zone("../test/data/p23_box.f3a")

    fields = bin.dfs

    active_core = msgs.primary_core_at_time(fields)
    imu_msg = msgs.IMU.load(fields, active_core)
    xkf1_msg = msgs.XKF1.load(fields, active_core, origin)
    xkf2_msg = msgs.XKF2.load(fields, active_core, origin)
    pos_msg = msgs.Pos.load(fields, origin)
    att_msg = msgs.Att.load_att(fields, origin)

    att = Field(att_msg.t, att_msg.att)
    gyro = Field(imu_msg.t, imu_msg.gyro)
    gyro_bias = Field(xkf1_msg.t, xkf1_msg.gyro_bias)
    rvel = Field(
        gyro.t, gyro.data - gyro_bias.data.linterp(gyro_bias.t, "nearest")(gyro.t)
    )

    pos = Field(pos_msg.t, pos_msg.pos)
    vel = Field(xkf1_msg.t, xkf1_msg.vel)
    accelerometer = Field(imu_msg.t, imu_msg.acc)
    accelerometer_bias = Field(xkf2_msg.t, xkf2_msg.acc_bias)
    acc = Field(
        accelerometer.t,
        accelerometer.data
        - accelerometer_bias.data.linterp(accelerometer_bias.t, "nearest")(
            accelerometer.t
        ),
    )
    return (
        SplineState,
        StateData,
        acc,
        accelerometer,
        att,
        att_msg,
        bin,
        fields,
        gyro,
        imu_msg,
        make_subplots,
        np,
        origin,
        pos,
        pos_msg,
        px,
        rvel,
        vel,
        xkf1_msg,
        xkf2_msg,
    )


@app.cell
def _(bin, px):
    px.line(bin.XKF1, x="TimeUS", y=["VN", "VE", "VD"], title="XKF1 Velocities")
    return


@app.cell
def _(acc, att, att_msg, imu_msg, pos, pos_msg, rvel, vel, xkf1_msg, xkf2_msg):
    print("Message Frequencies:")
    for k, v in {
        "pos": pos_msg,
        "att": att_msg,
        "xkf1": xkf1_msg,
        "xkf2": xkf2_msg,
        "imu": imu_msg,
    }.items():
        print(f"{k}: {v.t.shape}, {len(v.t) / (v.t[-1] - v.t[0]):.2f} Hz")

    print("\nField Frequencies:")
    for k, v in {
        "att": att,
        "rvel": rvel,
        "pos": pos,
        "vel": vel,
        "acc": acc,
    }.items():
        print(f"{k}: {v.t.shape}, {len(v.t) / (v.t[-1] - v.t[0]):.2f} Hz")
    return


@app.cell
def _(SplineState, StateData, fields, origin):

    t0, t1 = 246, 272
    sd = StateData.parse_fields(fields, origin).slice(t0, t1)
    rs, ts = 0.5, 100
    ss = SplineState.build(sd, auto_s=True, auto_s_cutoff_freq=5)

    return ss, t0, t1


@app.cell
def _(acc, accelerometer, gyro, make_subplots, np, rvel, ss, t0, t1, vel):


    _gyro = gyro.slice(t0, t1)
    _rvel = rvel.slice(t0, t1)
    _vel = vel.slice(t0, t1)
    _accelerometer = accelerometer.slice(t0, t1)
    _acc = acc.slice(t0, t1)


    _f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=["RVel", "Vel", "Acc"])

    _x = np.linspace(t0, t1, 5000)
    _f.add_traces(_rvel.data.plot(index=_rvel.t, restart_color=True, showlegend=False, dash="dash").data, rows=1, cols=1)
    _f.add_traces(ss.rvel(_x).plot(index=_x, restart_color=True, showlegend=False, dash="solid").data, rows=1, cols=1)

    _f.add_traces(_vel.data.plot(index=_vel.t, restart_color=True, showlegend=False, dash="dash").data, rows=2, cols=1)
    _f.add_traces(ss.vel(_x).plot(index=_x, restart_color=True, showlegend=False, dash="solid").data, rows=2, cols=1)

    _f.add_traces(_acc.data.plot(index=_acc.t, restart_color=True, showlegend=False, dash="dash").data, rows=3, cols=1)
    _x = np.linspace(t0, t1, 5000)
    _f.add_traces(ss.acc(_x).plot(index=_x, restart_color=True, showlegend=False, dash="solid").data, rows=3, cols=1)

    _f.update_layout(template="simple_white", margin=dict(t=30, b=0, l=0, r=0))
    return


if __name__ == "__main__":
    app.run()
