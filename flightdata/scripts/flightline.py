from flightdata import Flight, Origin
from geometry import GPS
from pathlib import Path
import argparse


def box_from_log(log: Flight, channel: int):
    c6on = Flight(log.data.loc[log.data[f'rcin_c{channel}']>=1500])
    groups = (c6on.time_flight.diff() > 1).cumsum()
    pilot = Flight(c6on.data.loc[groups==0])
    centre = Flight(c6on.data.loc[groups==1])

    return Origin.from_points("new", GPS(pilot.gps)[-1], GPS(centre.gps)[-1])

def box_from_logs(pilot: Flight, centre: Flight):
    return Origin.from_points("new", GPS(*pilot.gps.iloc[-1]), GPS(*centre.gps.iloc[-1]))


def main():
    parser = argparse.ArgumentParser(description='A tool for creating a flightline .f3a file from bin logs')

    parser.add_argument('-l', '--logdir', default='', help='folder to look for logs in')
    parser.add_argument('-p', '--pilot', default=None, help='flight log bin file to use, None for first')
    parser.add_argument('-c', '--centre', default=None, help='centre position bin file to use if input==None')
    parser.add_argument('-i', '--input', default=6, help='channel used to indicate pilot or centre postions (pwm>=1500), None for two files')

    args = parser.parse_args()

    print(args)
    
    logs = sorted(list(Path(args.logdir).glob("*.BIN")))
    logids = [int(log.stem) for log in logs]

    if args.pilot in logs:
        plog = args.pilot
    elif args.pilot is None:
        plog=logs[0]
    elif args.pilot.isdigit():
        plog = logs[logids.index(int(args.pilot))]

    print(f'Pilot position log: {plog}')

    if args.centre in logs:
        clog = args.centre
    elif args.centre is None:
        clog=None
    elif args.centre.isdigit():
        clog = logs[logids.index(int(args.centre))]
    
    print(f'Centre position log: {clog}')

    if args.centre:
        box = box_from_logs(Flight.from_log(plog), Flight.from_log(clog))
    else:
        box = box_from_log(Flight.from_log(plog), args.input)

    box.to_f3a_zone(Path(args.logdir) / f'box_{plog.stem}.f3a')

if __name__ == '__main__':
    main()
