from .ardupilot_ekfv2 import ardupilot_ekfv2_io_info
from .ardupilot_ekfv3 import ardupilot_ekfv3_io_info


def get_ardupilot_mapping(ekfv):
    if ekfv == 2:
        return ardupilot_ekfv2_io_info
    elif ekfv == 3:
        return ardupilot_ekfv3_io_info
    else:
        raise IOError('unknown EKF type')
