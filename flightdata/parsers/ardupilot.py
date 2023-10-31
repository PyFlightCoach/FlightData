from ardupilot_log_reader import Ardupilot


def parse_bin(binfile):
    _parser = Ardupilot(
        binfile, 
        types=[
            'XKF1', 'XKQ1', 'NKF1', 'NKQ1', 'NKF2', 
            'XKF2', 'POS', 'ATT', 'ACC', 'GYRO', 'IMU', 
            'ARSP', 'GPS', 'RCIN', 'RCOU', 'BARO', 'MODE', 
            'RPM', 'MAG', 'BAT', 'BAT2'
        ]
    )
