from pathlib import Path
import os
from datetime import datetime
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description='A tool to collect logs from a memory card')
    parser.add_argument('-f', '--folder', default='/media/', help='folder to get logs from, defaults to /media/')
    parser.add_argument('-s', '--subfolder', default=True, action=argparse.BooleanOptionalAction, help='Create subfolder for logs, defaults to True')

    args = parser.parse_args()

    source_folder = Path(args.folder) 
    
    all_logs = sorted(list(source_folder.rglob('*.BIN')))

    for file in all_logs:

        date = datetime.fromtimestamp(os.path.getmtime(file))
        folder = Path(date.strftime('%Y_%m_%d'))
        folder.mkdir(exist_ok=True)
        
        if not (folder / file.name).exists():
            print(f'copying {file} to {folder / file.name}')
            shutil.copyfile(file, folder / file.name )
        

if __name__ == '__main__':
    main()
