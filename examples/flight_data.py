from flightdata import Flight, State
import numpy as np
from flightplotting import plotsec
fl = Flight.from_fc_json('examples/data/manual_F3A_P23_22_05_31_00000350.json')
flf = Flight.from_json('test/data/p23_flight.json').remove_time_flutter()
flf = flf.butter_filter(5,5)

st = State.from_flight(fl)

plotsec(st).show()
stf = State.from_flight(flf)

import plotly.express as px
df = st.time.fft(st.t)
px.line(df, x=df.index, y='dt').update_yaxes(range=[0,0.0005]).show()

df = st.pos.fft(st.t)
px.line(df, x=df.index, y='z').update_yaxes(range=[0,0.0005]).show()

df = stf.time.fft(stf.t)
px.line(df, x=df.index, y='dt').update_yaxes(range=[0,0.0005]).show()



#import plotly.graph_objects as go
#
#fig=go.Figure()
#
#fig.add_trace(go.Scatter(x=fl.time_flight, y=np.gradient(fl.time_flight), name='original'))
#fig.add_trace(go.Scatter(x=flf.time_flight, y=np.gradient(flf.time_flight), name='filtered'))
#
#fig.add_trace(go.Scatter(x=fl.time_flight, y=np.gradient(st.pos.z), name='posz', yaxis='y2'))
#
#fig.update_layout(yaxis2=dict(anchor='x', overlaying='y', side='right'))
#
#fig.show()