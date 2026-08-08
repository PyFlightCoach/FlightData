import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    from flightdata import Flight, Origin, State
    from ardupilot_log_reader import Ardupilot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
    import geometry as g


    log = Ardupilot.parse("../test/data/p23.BIN", types=Flight.ardupilot_types)

    fl = Flight.from_log(log)

    origin = Origin.from_f3a_zone("../test/data/p23_box.f3a")

    st = State.from_flight(fl, origin)

    return g, go, log, make_subplots, np, px, st


@app.cell
def _(st):
    st.plot()
    return


@app.cell
def _(log):
    import pandas as pd
    imu = log.IMU
    xkf1 = log.XKF1
    xkf2 = log.XKF2
    dfout = log.IMU
    for df in [log.XKF1, log.XKF2]:
        dfout = pd.merge_asof(dfout, df, on="TimeUS", direction="nearest")

    dfout = dfout.assign(TimeUS=dfout.TimeUS / 1e6).set_index("TimeUS")
    dfout.columns
    return (dfout,)


@app.cell
def _(g, go, make_subplots, st):

    subst = st[221:230] # [167:168]

    ts = subst.t

    axis = "z"
    mode = "world"
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[f"world pos {axis}", f"{mode} vel {axis}", f"{mode} acc {axis}"],
        shared_xaxes=True,
        vertical_spacing=0.2,
    )

    for row, (kind, obj) in enumerate(
        zip(["pos", f"{mode} vel {axis}", f"{mode} acc {axis}"], [subst.pos,subst.wvel if mode == "world" else subst.vel, subst.wacc if mode == "world" else subst.acc]), start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=ts, y=getattr(obj, axis), mode="lines", line=dict(color="black"), showlegend=False
            ),
            row=row,
            col=1,
        )


    wv_diff = subst.pos.diff(subst.dt, "gradient")
    v_diff = subst.att.inverse().transform_point(wv_diff)

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=getattr(wv_diff if mode == "world" else v_diff, axis),
            showlegend=False,
        ),
        row=2,
        col=1
    )

    wa_diff = subst.wvel.diff(subst.dt, "gradient") + g.PZ(9.81)
    a_diff = subst.att.inverse().transform_point(wa_diff )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=getattr(wa_diff if mode=="world" else a_diff, axis),
            showlegend=False,
        ),
        row=3,
        col=1
    )


    fig.update_layout(
        template="simple_white",
        #title=name,
        height=400,
        width=600,
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis_title="H (m)",
        yaxis2_title="V (m/s)",
        yaxis3_title="A (m/s²)",
        xaxis3_title="Time",
        showlegend=True,
    )


    return subst, ts


@app.cell
def _(subst):
    subst.pos.z[-1] - subst.pos.z[0]
    return


@app.cell
def _(dfout, go, np, ts):

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=dfout.loc[ts[0] : ts[-1]].index,
            y=dfout.loc[ts[0] : ts[-1]].GyrX ,
            mode="lines",
            line=dict(color="blue"),
            name="GyrX",  # rad/s
        ),
    )

    fig2.add_trace(
        go.Scatter(
            x=dfout.loc[ts[0] : ts[-1]].index,
            y=dfout.loc[ts[0] : ts[-1]].GyrX - np.radians(dfout.loc[ts[0] : ts[-1]].GX) ,
            mode="lines",
            line=dict(color="red"),
            name="GyrX - bias", # deg/s
        ),
    )

    return


@app.cell
def _(dfout, np, px):
    px.line(dfout.GX).show()
    px.line(np.degrees(dfout.GyrX)).show()
    return


if __name__ == "__main__":
    app.run()
