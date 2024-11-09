from flightdata import Flight, Origin
from geometry import GPS
from pathlib import Path
import argparse



def get_con_groups(log: Flight, channel: int):
    c6on = log.data.loc[log.data[f'rcin_c{channel}']>=1500]
    groups = (c6on.time_flight.diff() > 1).cumsum()
    return [Flight(c6on.loc[groups==grp]) for grp in groups.unique()]

def box_from_log(log: Flight, channel: int):
    grps = get_con_groups(log, channel)
    pilot = grps[0]
    centre = grps[1]
#    c6on = Flight(log.data.loc[log.data[f'rcin_c{channel}']>=1500])
#    groups = (c6on.time_flight.diff() > 1).cumsum()
#    pilot = Flight(c6on.data.loc[groups==0])
#    centre = Flight(c6on.data.loc[groups==1])

    return Origin.from_points("new", GPS(pilot.pos)[-1], GPS(centre.pos)[-1])

def box_from_logs(pilot: Flight, centre: Flight):
    return Origin.from_points("new", GPS(*pilot.pos.iloc[-1]), GPS(*centre.pos.iloc[-1]))


def main():
    parser = argparse.ArgumentParser(description='A tool for creating a flightline .f3a file from bin logs')

    parser.add_argument('-l', '--logdir', default='', help='folder to look for logs in')
    parser.add_argument('-p', '--pilot', default=None, help='flight log bin file to use, None for first')
    parser.add_argument('-c', '--centre', default=None, help='centre position bin file to use if input==None')
    parser.add_argument('-d', '--direction', default=None, help='heading of the box, if this is specified only pilot will be read')
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

    pilot = Flight.from_log(plog)

    print(f'Pilot position log: {plog}')

    if args.centre in logs:
        clog = args.centre
    elif args.centre is None:
        clog=None
    elif args.centre.isdigit():
        clog = logs[logids.index(int(args.centre))]
    if clog:
        centre = Flight.from_log(clog)
    print(f'Centre position log: {clog}')

    if args.centre:
        box = Origin.from_points("new", GPS(*pilot.pos.iloc[-1]), GPS(*centre.pos.iloc[-1]))
    else:
        groups = get_con_groups(pilot, args.input)
        if args.direction:
            box = Origin("new", GPS(groups[0].pos)[-1], float(args.direction))
        else:
            box = Origin.from_points("new", GPS(groups[0].pos)[-1], GPS(groups[1].pos)[-1])
            
    box.to_f3a_zone(Path(args.logdir) / f'box_{plog.stem}.f3a')

if __name__ == '__main__':
    main()
