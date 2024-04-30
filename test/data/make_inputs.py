from flightdata import Flight


Flight.from_log('test/data/p23.BIN').to_json('test/data/p23.json')
Flight.from_log('test/data/vtol_hover.bin').to_json('test/data/vtol_hover.json')
