from flightdata import Flight, State

fl = Flight.from_fc_json('examples/data/manual_F3A_P23_22_05_31_00000350.json')

flf = Flight.from_json('test/data/p23_flight.json').remove_time_flutter()
flf = flf.butter_filter(5,5)

st = State.from_flight(fl)

stf = State.from_flight(flf)

