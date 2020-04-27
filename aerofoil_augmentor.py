import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from pathlib import Path
import os
import TitleSequence as title
import re
import sys
import random


def do_checks(file):
    """perform checks on data"""
    # check file coordinates
    x, _, _, _ = get_coordinates(file)
    if (len(x) - 1) % 2 != 0:
        raise ValueError("xy coordinates must be an odd number. Redistribute aerofoil coordinates.")


def get_coordinates(file):
    """get x,y coordinates of areofoil"""
    xy = np.loadtxt(file, delimiter=" ", dtype=np.float32, skiprows=1)  # xy coordinates
    x = xy[:, 0]
    y = xy[:, 1]
    num_coords = len(y) - 1  # -1 for coordinate at leading edge (x=0)
    with open(file) as f:
        target = f.readline()
    return x, y, num_coords, target


def convert_system(y, num_coords):
    """move from x,y coordinates to camber & thickness along x"""
    camber = []
    thickness = []
    for i in range(num_coords // 2):  # two y points for one camber point
        camber.append((y[i] + y[-i-1]) / 2)
        thickness.append(y[i] - y[-i-1])
    return camber, thickness


def modify_aerofoil(camber, thickness, Bpoints_camber, Bpoints_thick, n):
    """modify camber and camber according to Bezier points"""
    new_camber = []
    for y in camber:
        new_point = 0.
        for i, control_point in enumerate(Bpoints_camber):
            binom_coefficient = scipy.special.binom(n, i)  # binomial coefficient
            bern_poly = binom_coefficient * y ** i * (1-y) ** (n-i)  # Bernstein basis polynomials
            new_point += bern_poly * control_point
        new_camber.append(new_point)

    new_thickness = []
    for y in thickness:
        new_point = 0.
        for i, control_point in enumerate(Bpoints_thick):
            binom_coefficient = scipy.special.binom(n, i)  # binomial coefficient
            bern_poly = binom_coefficient * y ** i * (1-y) ** (n-i)  # Bernstein basis polynomials
            new_point += bern_poly * control_point
        new_thickness.append(new_point)

    return new_camber, new_thickness


def return_system(camber, thickness, y_orig, num_coords):
    """convert system from camber along x to x,y coordinates"""
    new_y = [0] * len(y_orig)  # this also sorts out coordinate at leading edge
    for i in range(num_coords//2):
        new_y[i] = camber[i] + thickness[i] / 2
        new_y[-i-1] = camber[i] - thickness[i] / 2
    return new_y


def bezier_cp(smallest_modification):
    """determine Bezier control points"""
    modification = smallest_modification * random.randint(1, 60)
    factor_camber = 1 if random.random() > 0.5 else -1  # decide whether aerofoil expands of contracts
    factor_thickness = 1 if random.random() > 0.5 else -1  # decide whether aerofoil expands of contracts
    modification_camber = factor_camber * abs(modification / 3) if factor_camber == -1 \
        else factor_camber * abs(modification)  # camber modification too strong without /3
    modification_thickness = factor_thickness * abs(modification)

    Bpoints_camber = [0]  # control points of camber Bezier curve
    Bpoints_thickness = [0]  # control points of thickness Bezier curve

    for i in range(num_control_points - 2):
        cp_camber = random.uniform(modification_camber / 2, modification_camber)
        Bpoints_camber.append(cp_camber)
        modification_camber = cp_camber / 2 if factor_camber == 1 else cp_camber * 1.3

        cp_thickness = random.uniform(modification_thickness / 2, modification_thickness)
        Bpoints_thickness.append(cp_thickness)
        modification_thickness = cp_thickness / 2 if factor_thickness == 1 else cp_thickness * 2
    Bpoints_camber.append(0)
    Bpoints_thickness.append(0)

    degree_poly = len(Bpoints_thickness)  # degree of polynomials (camber and thickness Bezier curves are same size)

    return Bpoints_camber, Bpoints_thickness, degree_poly


def print_aerofoil(x, new_y, dest, target):
    np.savetxt(dest, np.vstack(zip(x, new_y[::-1])), fmt='%.4f', delimiter=' ', header=target, comments='')


if __name__ == "__main__":
    title.print_title(["", "Augment aerofoils by changing camber and camber"])

    # parameters
    print_plots = False  # use this to print a before and after of one aerofoil due to Bezier curve
    num_control_points = 6  # number of control points for camber and thickness (includes 0's at end)
    smallest_modification = 0.01  # absolute value taken
    num_new_aerofoils = 2  # build n new aerofoils with the Bezier control points from one original aerofoil

    # paths
    root_dir = Path('data')
    in_files = root_dir / 'out' / 'test'  # read in aerofoils
    out_files = root_dir / 'augmented_aerofoils'  # print directory
    out_files.mkdir(exist_ok=True)

    # calculations
    aerofoils = [file for file in os.listdir(in_files) if re.search(r"(.csv)$", file)]

    do_checks(in_files / aerofoils[0])

    for aerofoil in aerofoils:
        for i in range(num_new_aerofoils):
            Bpoints_camber, Bpoints_thickness, n = bezier_cp(smallest_modification)
            x, y, num_coords, targets = get_coordinates(in_files / aerofoil)
            camber, thickness = convert_system(y, num_coords)
            new_camber, new_thickness = modify_aerofoil(camber, thickness, Bpoints_camber, Bpoints_thickness, n)
            new_y = return_system(new_camber, new_thickness, y, num_coords)
            if not print_plots:
                print_aerofoil(x, new_y, out_files / (str(i) + aerofoil), targets)

            if print_plots:
                plt.plot(x, y, 'b-', label='original aerofoil')
                plt.plot(x, new_y, 'r-', label='modified aerofoil')
                plt.plot(x[:num_coords // 2], camber, 'b-.', label='original camber')
                plt.plot(x[:num_coords // 2], new_camber, 'r-.', label='modified camber')
                # plt.plot(np.linspace(x[num_coords // 2], x[-1], n), Bpoints_camber, "ko",
                #          label='camber control points')
                # plt.plot(np.linspace(x[num_coords // 2], x[-1], n), Bpoints_thickness, "g*",
                #          label='thickness control points')
                plt.legend(loc="upper right")
                plt.show()

        if print_plots:
            sys.exit("Sample plot printed. Code terminated")
