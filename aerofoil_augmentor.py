import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from pathlib import Path
import os
import TitleSequence as title
import re
import sys


def do_checks(file, Bpoints_camber, Bpoints_thick):
    """perform checks on data"""
    # check file coordinates
    x, _, _ = get_coordinates(file)
    if (len(x) - 1) % 2 != 0:
        raise ValueError("xy coordinates must be an odd number. Redistribute aerofoil coordinates.")
    # if x[0] != 1. or x[-1] != 1.:
    #     raise ValueError("x coordinates of aerofoil must go from 0 to 1 at both leading and trailing edges.")

    # check control points of Bezier curves
    if Bpoints_camber[0] != 0 or Bpoints_camber[-1] != 0:
        raise ValueError("Camber control points must start and end in 0")
    if Bpoints_thick[0] != 0 or Bpoints_thick[-1] != 0:
        raise ValueError("Thickness control points must start and end in 0")


def get_coordinates(file):
    """get x,y coordinates of areofoil"""
    xy = np.loadtxt(file, delimiter=" ", dtype=np.float32, skiprows=1)  # xy coordinates
    x = np.array(xy[:, 0], dtype=np.float32)
    y = np.array(xy[:, 1], dtype=np.float32)
    num_coords = len(y) - 1  # -1 for coordinate at leading edge (x=0)
    return x, y, num_coords


def convert_system(y, num_coords):
    """move from x,y coordinates to camber & thickness along x"""
    camber = []
    thickness = []
    for i in range(num_coords // 2):  # two y points for one camber point
        camber.append((y[i] + y[-i-1]) / 2)
        thickness.append(y[i] - y[-i-1])
    return camber, thickness


def modify_aerofoil(camber, thickness, Bpoints_camber, Bpoints_thick, n_camber, n_thickness):
    """modify camber and camber according to Bezier points"""
    new_camber = []
    for y in camber:
        new_point = 0.
        for i, control_point in enumerate(Bpoints_camber):
            binom_coefficient = scipy.special.binom(n_camber, i)  # binomial coefficient
            bern_poly = binom_coefficient * y ** i * (1-y) ** (n_camber-i)  # Bernstein basis polynomials
            new_point += bern_poly * control_point
        new_camber.append(new_point)

    new_thickness = []
    for y in thickness:
        new_point = 0.
        for i, control_point in enumerate(Bpoints_thick):
            binom_coefficient = scipy.special.binom(n_thickness, i)  # binomial coefficient
            bern_poly = binom_coefficient * y ** i * (1-y) ** (n_thickness-i)  # Bernstein basis polynomials
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


if __name__ == "__main__":
    title.print_title(["", "Augment aerofoils by changing camber and camber"])

    # parameters
    print_plots = True  # use this to print a before and after of one aerofoil due to Bezier curve
    Bpoints_camber = [0, 0.3, 0]  # control points of the Bezier curve
    Bpoints_thickness = [0, -0.3, 0]  # control points of the Bezier curve

    # paths
    root_dir = Path('data')
    in_files = root_dir / 'out' / 'test'
    out_files = root_dir / 'augmented aerofoils'

    # calculations
    n_camber = len(Bpoints_camber)  # degree of Bernstein basis polynomial
    n_thickness = len(Bpoints_thickness)  # degree of Bernstein basis polynomial
    aerofoils = [file for file in os.listdir(in_files) if re.search(r"(.csv)$", file)]

    do_checks(in_files / aerofoils[0], Bpoints_camber, Bpoints_thickness)

    for aerofoil in aerofoils:
        x, y, num_coords = get_coordinates(in_files / aerofoil)
        camber, thickness = convert_system(y, num_coords)
        new_camber, new_thickness = modify_aerofoil(camber, thickness, Bpoints_camber, Bpoints_thickness, n_camber,
                                                    n_thickness)
        new_y = return_system(new_camber, new_thickness, y, num_coords)

        if print_plots:
            plt.plot(x, y, 'b-', label='original aerofoil')
            plt.plot(x, new_y, 'r-', label='modified aerofoil')
            plt.plot(x[:num_coords // 2], camber, 'b-.', label='original camber')
            plt.plot(x[:num_coords // 2], new_camber, 'r-.', label='modified camber')
            plt.plot(np.linspace(x[num_coords // 2], x[-1], n_camber), Bpoints_camber, "ko",
                     label='camber control points')
            plt.plot(np.linspace(x[num_coords // 2], x[-1], n_thickness), Bpoints_thickness, "g*",
                     label='thickness control points')
            plt.legend(loc="upper right")
            plt.show()
            sys.exit("Sample plot printed. Code terminated")
