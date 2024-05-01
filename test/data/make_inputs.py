from flightdata import Flight


fl = Flight.from_log('test/data/p23.BIN')
fl.to_json('test/data/p23.json')
fl.to_json('test/data/p23_flight.json')
Flight.from_log('test/data/vtol_hover.bin').to_json('test/data/vtol_hover.json')
