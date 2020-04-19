import numpy as np
from pathlib import Path
import shutil
import os
import re
import matplotlib.pyplot as plt
import sys
from random import seed
from random import random

# parameters
seed(1)
train_valid_split = 0.8  # percentage to split train and validation set randomly
root_dir = Path('data')
in_files = root_dir / 'downloaded_files'
out_files = root_dir / 'out'
chosen_aerofoil_x = 'NACA_24112.csv'  # use x coordinates of this file for all other files

# make folders
shutil.rmtree(out_files)  # delete all previous outputs
out_files.mkdir(exist_ok=True)
train_set = out_files / 'train'
test_set = out_files / 'test'
train_set.mkdir(exist_ok=True)
test_set.mkdir(exist_ok=True)


def start_code():
    strings2print = [f" ",
                     f"Chosen aerofoil dictates x coordinates of all other aerofoils"]
    spacing = 80

    print("#"*spacing)
    print(f"#{'Make spacing of all aerofoils equal':^{spacing-2}}#")
    for string in strings2print:
        print(f"# {string:<{spacing-4}} #")
    print("#"*spacing)
    print()

    # TODO: make this check more robust
    if len(os.listdir(train_set)) > 0:
        print("TRAIN SET DIRECTORY NOT EMPTY!!!! FILES WILL BE OVERWRITTEN OR ADDED IF SEED HAS CHANGED!")
    if len(os.listdir(test_set)) > 0:
        print("VALIDATION SET DIRECTORY NOT EMPTY!!!! FILES WILL BE OVERWRITTEN OR ADDED IF SEED HAS CHANGED!")


def do_code():
    # make and read files & folders
    aerofoils = [file for file in os.listdir(in_files)
                 if re.search(r"(.csv)$", file)
                 if os.path.isfile(in_files / file)]

    # get x coordinates of chosen file
    x_target = []
    with open(in_files / chosen_aerofoil_x) as f:
        for i, line in enumerate(f):
            if i <= 1:
                continue  # skip first two lines of file
            else:
                xy = re.findall(r'[+-]?\d*[.]?\d*', line)
                xy = [num for num in xy if num != '']
                x_target.append(float(xy[0]))
    x_target_half = x_target[len(x_target) // 2:]  # x target is symmetrical top and bottom

    # make all aerofoils same size
    for aerofoil in aerofoils:
        if aerofoils == chosen_aerofoil_x:
            continue  # no need to interpolate chosen aerofoil
        try:
            x_coord = []
            y_coord = []
            with open(in_files / aerofoil) as f:
                for i, line in enumerate(f):
                    if i <= 1:
                        continue  # skip first two lines of file
                    else:
                        xy = re.findall(r'[+-]?\d*[.]?\d*', line)
                        xy = [num for num in xy if num != '']
                        x_coord.append(float(xy[0]))
                        y_coord.append(float(xy[1]))

            x_top = np.append(x_coord[len(y_coord) // 2:0:-1], x_coord[0])
            x_bottom = x_coord[len(y_coord) // 2:]
            y_top = np.append(y_coord[len(y_coord) // 2:0:-1], y_coord[0])
            y_bottom = y_coord[len(y_coord) // 2:]

            y_bottom_target = np.interp(x_target_half, x_bottom, y_bottom)
            y_top_target = np.interp(x_target_half, x_top, y_top)
            y_target = np.append(y_top_target[:0:-1], y_bottom_target)  # remove one of the two zeros

            with open(in_files / aerofoil) as f:
                found_line = False
                for line in f:
                    if 'ClCd' in line:
                        max_ClCd_angle = line
                        found_line = True
                        break
                if not found_line:
                    raise Exception("Max ClCd & angle not found in file")

            # print file
            if random() > train_valid_split:
                # move file to validation set
                out_dest = test_set
            else:
                # move file to train set
                out_dest = train_set

            with open(out_dest / aerofoil, 'w') as f:
                f.write(f"{max_ClCd_angle}")
                for x, y in zip(x_target, y_target):
                    f.write(f"{x:.4f} {y:.4f}\n")

            # plt.plot(x_coord, y_coord, 'r-')
            # plt.plot(x_target, y_target, 'bo')
            # plt.show()

        except Exception as exc:
            print(f"Error in file {aerofoil}. File ignored.\n"
                  f"Error: {exc}.\n")

    print(f"Code finished. Output folder: {out_files}.\n"
          f"Number of coordinates in every aerofoil file: {len(y_target)}")


start_code()
do_code()
