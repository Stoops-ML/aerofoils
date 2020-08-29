import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree
from pathlib import Path
import shutil
import os
import re
from random import seed, random
from bs4 import BeautifulSoup
import requests
import sys


def split_data(in_files, out_path, train_valid_test_split=None):
    """
    split data into train, valid and test set
    'train_valid_test_split' is the percentage to split train, validation and test sets respectively
    """

    # define train, valid, test split
    if not train_valid_test_split:
        train_valid_test_split = [0.77, 0.2, 0.03]
    else:
        assert isinstance(train_valid_test_split, list) and \
               len(train_valid_test_split) == 3 and \
               sum(train_valid_test_split) == 1, "'train_valid_test_split' must be a list with three numbers that sum" \
                                                 "to 1."

    # folders
    in_files = Path(in_files)
    out_path = Path(out_path)
    if out_path.is_dir():
        shutil.rmtree(out_path)  # delete all previous outputs
    out_path.mkdir(exist_ok=True)
    (out_path / 'train').mkdir(exist_ok=True)
    (out_path / 'valid').mkdir(exist_ok=True)
    (out_path / 'test').mkdir(exist_ok=True)

    # move aerofoils to train, valid, or test set
    aerofoils = [file for file in os.listdir(in_files) if re.search(r"(.csv)$", file)]
    for aerofoil in aerofoils:
        # choose whether aerofoil is in test, valid or train set
        random_num = random()
        if random_num <= train_valid_test_split[0]:  # move file to train set
            shutil.copyfile(in_files / aerofoil, out_path / 'train' / aerofoil)
        elif random_num <= (train_valid_test_split[0] + train_valid_test_split[1]):  # move file to validation set
            shutil.copyfile(in_files / aerofoil, out_path / 'valid' / aerofoil)
        else:  # move file to test set
            shutil.copyfile(in_files / aerofoil, out_path / 'test' / aerofoil)


def aerofoil_redistribution(base_aerofoil, in_files, out_path):
    """
    Redistribute coordinates of all aerofoils such that all aerofoils have the same number of coordinates
    All aerofoils in 'in_files' will have the same x coordinates as 'base_aerofoil'. The y coordinates of all aerofoils
    will be interpolated at their new x coordinates. This is used to remove the x coordinates from the model inputs.
    dataset=True if aerofoil file is to be part of train, valid or test set. If aerofoil is used for prediction:
    dataset=False.

    NOTE:
        from numpy documentation on np.interp:
        "The x-coordinate sequence is expected to be increasing, but this is not explicitly enforced. However, if the
        sequence xp is non-increasing, interpolation results are meaningless."
        The bottom aerofoil surface has decreasing x coordinates, however the interpolation method still works.
        There is an error where for even coordinates (if the input is odd, and vice-versa for the other way round)
        there will be a NaN in the output of the interpolation method at the leading edge of the aerofoil.
    """
    # parameters
    seed(42)

    # folders
    in_files = Path(in_files)
    out_path = Path(out_path)
    if out_path.is_dir():
        shutil.rmtree(out_path)  # delete all previous outputs
    out_path.mkdir(exist_ok=True)

    # get x coordinates of base file
    coordinates = np.loadtxt(base_aerofoil, delimiter=" ", dtype=np.float32, skiprows=1)
    min_ind = np.argmin(coordinates[:, 0])  # index of coordinate that separates top from bottom surface
    num_mins = np.where(coordinates[:, 0] == coordinates[min_ind, 0])[0]  # number of minimum x coordinates
    if np.shape(num_mins)[0] == 2:  # if two minimum x coordinates are present then change one of them
        coordinates[num_mins[1], 0] += 0.00000001
    x_top_target = coordinates[:min_ind, 0]  # x target coordinates of top surface
    x_bottom_target = coordinates[min_ind:, 0]  # x target coordinates of top surface
    # all aerofoil files in directory
    aerofoils = [file for file in os.listdir(in_files) if re.search(r"(.csv)$", file)]

    # transform y coordinates of all aerofoils to those of the base aerofoil's x coordinates
    for aerofoil in aerofoils:
        with open(in_files / aerofoil) as f:  # get aerodynamics of aerofoil from file
            aerodynamics = f.readline().strip()
        coordinates = np.loadtxt(in_files / aerofoil, delimiter=" ", dtype=np.float32, skiprows=1)  # xy coordinates
        min_ind = np.argmin(coordinates[:, 0])
        num_mins = np.where(coordinates[:, 0] == coordinates[min_ind, 0])[0]
        if np.shape(num_mins)[0] > 1:
            for i, minimum_num in enumerate(num_mins):
                if i == 0:
                    continue
                coordinates[num_mins[1], 0] += 0.00000001  # todo fix this for more than two minimums

        # top aerofoil surface
        x_top = coordinates[:min_ind, 0]
        # x_top = x_top[::-1]
        y_top = coordinates[:min_ind, 1]
        # assert np.all(np.diff(x_top) > 0), 'interpolation wont work because x values must be increasing'
        f_interp = interpolate.interp1d(x_top, y_top, fill_value='extrapolate')
        y_top_target = f_interp(x_top_target)
        # y_top_target = y_top_target[::-1]

        # bottom aerofoil surface
        x_bottom = coordinates[min_ind:, 0]
        # print(x_bottom)
        # assert np.all(np.diff(x_bottom) > 0), 'interpolation wont work because x values must be increasing'
        y_bottom = coordinates[min_ind:, 1]
        f_interp = interpolate.interp1d(x_bottom, y_bottom, fill_value='extrapolate')
        y_bottom_target = f_interp(x_bottom_target)

        # combine top and bottom surfaces and print file
        x_target = np.append(x_top_target, x_bottom_target)
        y_target = np.append(y_top_target, y_bottom_target)

        # fix files
        y_target[np.isnan(y_target)] = 0.  # change NaN elements to 0.0
        y_target[np.isinf(y_target)] = 0.  # change inf elements to 0.0
        # todo THIS IS A QUICK FIX TO REMOVE NaN FROM y_target. SEE DOC STRING FOR MORE DETAILS ON THIS.

        np.savetxt(out_path / aerofoil, np.transpose([x_target, y_target]), header=aerodynamics,
                   fmt='%.6f', comments='')

    print(f"Number of coordinates in every aerofoil file: {len(x_target)}.\n")

    return len(x_target)


def download_aerofoils(out_path, num_outputs=1, overwrite=True):
    """download all 2D aeroofoils from airfoiltools.com"""

    # links
    aerofoils_link = 'http://airfoiltools.com/search/airfoils'
    coordinates_link = 'http://airfoiltools.com/airfoil/seligdatfile?airfoil='
    aerodynamics_link = 'http://airfoiltools.com/airfoil/details?airfoil='
    out_path = Path(out_path)

    page = requests.get(aerofoils_link)
    aerofoils_soup = BeautifulSoup(page.content, 'lxml')
    aerofoil_links = aerofoils_soup.find_all("a", href=lambda href: href and "/airfoil/details?airfoil=" in href)
    aerofoil_names = [name['href'].split('=')[-1] for name in aerofoil_links]

    for aerofoil_name in aerofoil_names:
        # check if aerofoil file exists
        aerofoil_file = ''.join((aerofoil_name, ".csv"))
        if (out_path / aerofoil_file).is_file() and not overwrite:
            continue

        # aerofoil coordinates
        coordinates_page = ''.join((coordinates_link, aerofoil_name))
        coordinates = requests.get(coordinates_page)
        x, y = [], []
        for line in coordinates.iter_lines():
            xy = re.findall(r'^\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)', str(line, 'utf-8'))
            if xy:
                [(x_coordinate, y_coordinate)] = xy
                x.append(float(x_coordinate))
                y.append(float(y_coordinate))

        # get maximum Cl/Cd at angle
        aerodynamics_page = ''.join((aerodynamics_link, aerofoil_name))
        aerodynamics_page = requests.get(aerodynamics_page)
        aerodynamics_soup = BeautifulSoup(aerodynamics_page.content, 'lxml')
        aerodynamics = [item.get_text() for item in aerodynamics_soup.select('td.cell4')
                        if 'α' in item.get_text()]
        max_ClCd_list, angle_list = [], []
        for aerodynamic in aerodynamics:
            output = re.findall(r'^([+-]?\d+\.\d+)\D+([+-]?\d+\.\d+)', aerodynamic)
            if output:
                [(max_ClCd, angle)] = output
                max_ClCd_list.append(float(max_ClCd))
                angle_list.append(float(angle))

        # check if aerodynamic data collected
        if len(max_ClCd_list) == 0:
            continue  # don't print file if no aerodynamic data collected

        ClCd = max_ClCd_list[:num_outputs]
        angle = angle_list[:num_outputs]

        # print aerofoil file
        with open(out_path / aerofoil_file, 'w') as f:
            f.write(f"Max ClCd {ClCd} at {angle}deg\n")
            for x_target, y_target in zip(x, y):
                f.write(f"{x_target:.4f} {y_target:.4f}\n")

    print(f"Number of aerofoils downloaded = {len(os.listdir(out_path))}")
