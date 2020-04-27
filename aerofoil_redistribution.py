import numpy as np
from pathlib import Path
import shutil
import os
import re
import matplotlib.pyplot as plt
import sys
import TitleSequence as Title
from random import seed, random
from tqdm import tqdm

# parameters
seed(42)
train_valid_test_split = [0.77, 0.2, 0.03]  # percentage to split train, validation and test sets randomly
root_dir = Path('data')
in_files = root_dir / 'auto_downloaded_files'
out_files = root_dir / 'out'
chosen_aerofoil_x = 'du06.csv'  # use x coordinates of this file for all other files

# make folders
shutil.rmtree(out_files)  # delete all previous outputs
out_files.mkdir(exist_ok=True)
train_set = out_files / 'train'
valid_set = out_files / 'valid'
test_set = out_files / 'test'
train_set.mkdir(exist_ok=True)
valid_set.mkdir(exist_ok=True)
test_set.mkdir(exist_ok=True)

Title.print_title([" ", "Chosen aerofoil dictates x coordinates of all other aerofoils"], spacing=80)

# make and read files & folders
aerofoils = [file for file in os.listdir(in_files)
             if re.search(r"(.csv)$", file)]

# get x coordinates of chosen file
x_target = []
with open(in_files / chosen_aerofoil_x) as f:
    for i, line in enumerate(f):
        if i <= 1:
            continue  # skip first two lines of file
        else:
            xy = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
            x_target.append(float(xy[0]))
x_target_half = x_target[len(x_target) // 2:]  # x target is symmetrical top and bottom

# mean of ClCd and angle
# ClCd_list = []
# angle_list = []
# for aerofoil in aerofoils:
#     with open(in_files / aerofoil) as f:
#         for line in f:
#             if 'ClCd' in line:
#                 outputs = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
#                 ClCd_list.append(float(outputs[0]))
#                 angle_list.append(float(outputs[1]))
#                 break
# ClCd_mean = sum(ClCd_list) / len(aerofoils)
# angle_mean = sum(angle_list) / len(aerofoils)

# # standard deviation of ClCd and angle
# ClCd_list_SD = [(ClCd - ClCd_mean)**2 for ClCd in ClCd_list]
# angle_list_SD = [(angle - angle_mean)**2 for angle in angle_list]
# ClCd_SD = np.sqrt(1/len(aerofoils) * sum(ClCd_list_SD))
# angle_SD = np.sqrt(1/len(aerofoils) * sum(angle_list_SD))

# make all aerofoils same size

for aerofoil in tqdm(aerofoils):
    try:
        x_coord = []
        y_coord = []
        with open(in_files / aerofoil) as f:
            for i, line in enumerate(f):
                if "NACA" in line:
                    continue
                if 'ClCd' in line:
                    max_ClCd_angle = line
                    # outputs = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
                    # ClCd = (float(outputs[0]) - ClCd_mean) / ClCd_SD
                    # angle = (float(outputs[1]) - angle_mean) / angle_SD
                else:
                    xy = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
                    x_coord.append(float(xy[0]))
                    y_coord.append(float(xy[1]))

        x_top = np.append(x_coord[len(y_coord) // 2:0:-1], x_coord[0])
        x_bottom = x_coord[len(y_coord) // 2:]
        y_top = np.append(y_coord[len(y_coord) // 2:0:-1], y_coord[0])
        y_bottom = y_coord[len(y_coord) // 2:]

        y_bottom_target = np.interp(x_target_half, x_bottom, y_bottom)
        y_top_target = np.interp(x_target_half, x_top, y_top)
        y_target = np.append(y_top_target[:0:-1], y_bottom_target)  # remove one of the two zeros

        # print file
        random_num = random()
        if random_num <= train_valid_test_split[0]:  # move file to train set
            out_dest = train_set
        elif random_num <= (train_valid_test_split[0] + train_valid_test_split[1]):  # move file to validation set
            out_dest = valid_set
        else:  # move file to test set
            out_dest = test_set

        with open(out_dest / aerofoil, 'w') as f:
            # f.write(f"Max ClCd {ClCd:.4f} at {angle:.4f}deg\n")
            f.write(max_ClCd_angle)
            for x, y in zip(x_target, y_target):
                f.write(f"{x:.4f} {y:.4f}\n")

        # plt.plot(x_coord, y_coord, 'r-')
        # plt.plot(x_target, y_target, 'bo')
        # plt.show()

    except Exception as exc:
        print(f"Error in file {aerofoil}. File ignored.\n"
              f"Error: {exc}.\n")

print(f"Code finished. Output folder: {out_files}.\n"
      f"Number of coordinates in every aerofoil file: {len(x_target)}.\n")
      # f"Mean angle = {angle_mean:.2f}, standard deviation angle = {angle_SD:.2f}.\n"
      # f"Mean ClCd = {ClCd_mean:.2f}, standard deviation ClCd = {ClCd_SD:.2f}.")

# with open(out_files / "Normalising_values.txt", 'w') as f:
#     f.write(f"Mean angle = {angle_mean:.2f}, standard deviation angle = {angle_SD:.2f}.\n"
#             f"Mean ClCd = {ClCd_mean:.2f}, standard deviation ClCd = {ClCd_SD:.2f}.")
