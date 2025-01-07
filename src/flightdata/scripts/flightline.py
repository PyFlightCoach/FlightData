from ast import In
import sys
from flightdata import Flight, Origin
from geometry import GPS
from pathlib import Path
import argparse
import re


def get_con_groups(log: Flight, channel: int):
    c6on = log.data.loc[log.data[f'rcin_c{channel}']>=1500]
    groups = (c6on.time_flight.diff() > 1).cumsum()
    return [Flight(c6on.loc[groups==grp]) for grp in groups.unique()]

def box_from_log(log: Flight, channel: int):
    grps = get_con_groups(log, channel)
    pilot = grps[0]
    centre = grps[1]

    return Origin.from_points("new", GPS(pilot.pos)[-1], GPS(centre.pos)[-1])

def box_from_logs(pilot: Flight, centre: Flight):
    return Origin.from_points("new", GPS(*pilot.pos.iloc[-1]), GPS(*centre.pos.iloc[-1]))

re_logid = re.compile(r'[\d+]$')
def get_bin_from_number(folder: Path, number: str):
    try:
        return list(folder.glob(f"*{number}.BIN"))[0]
    except IndexError:
        return list(folder.glob(f"*{number}.bin"))[0]



def create_flightline(args) -> argparse.Namespace:
    logdir = Path(args.logdir)

    plog = logdir / args.pilot
    if not plog.exists():
        if args.pilot is None:
            plog=sorted(list(logdir.glob("*.BIN")))[0]
        elif args.pilot.isdigit():
            plog = get_bin_from_number(logdir, args.pilot)
        else:
            raise FileNotFoundError(f'Could not find pilot log file: {args.pilot}')
    
    pilot = Flight.from_log(plog)

    print(f'Pilot position log: {plog}')

    if args.centre:
        clog = logdir / args.centre
        if not clog.exists():
            if args.centre is None:
                clog=None
            elif args.centre.isdigit():
                clog = get_bin_from_number(logdir, args.centre)
        
        print(f'Centre position log: {clog}' if clog else "No centre position log")
        centre = Flight.from_log(clog) if clog else None
        
        box = Origin.from_points("new", GPS(*pilot.pos.iloc[-1]), GPS(*centre.pos.iloc[-1]))
    else:
        if args.direction:
            box = Origin("new", GPS(plog.pos)[-1], float(args.direction))
        else:
            groups = get_con_groups(pilot, args.input)
            box = Origin.from_points("new", GPS(groups[0].pos)[-1], GPS(groups[1].pos)[-1])
            
    return box, plog.stem


def parse_args(args):
    parser = argparse.ArgumentParser(description='A tool for creating a flightline .f3a file from bin logs')

    parser.add_argument('-l', '--logdir', default='', help='folder to look for logs in')
    parser.add_argument('-p', '--pilot', default=None, help='flight log bin file to use, None for first')
    parser.add_argument('-c', '--centre', default=None, help='centre position bin file to use if input==None')
    parser.add_argument('-d', '--direction', default=None, help='heading of the box, if this is specified only pilot will be read')
    parser.add_argument('-i', '--input', default=6, help='channel used to indicate pilot or centre postions (pwm>=1500), None for two files')

    args = parser.parse_args(args)

    return args


def main():
    args = parse_args(sys.argv[1:])
    box, name = create_flightline(args)
    box.to_f3a_zone(Path(args.logdir) / f'box_{name}.f3a')
    

if __name__ == '__main__':
    main()
