import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt


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


def do_code():
    # parameters
    root_dir = Path('data')
    print_data = root_dir / 'out'
    chosen_aerofoil_x = 'NACA_0009.csv'  # use x coordinates of this file for all other files

    # make and read files & folders
    print_data.mkdir(exist_ok=True)
    aerofoils = [file for file in os.listdir(root_dir) if 'csv' in file] # add if os.path.isfile(file)

    # get x coordinates of chosen file
    coordinates = np.loadtxt(root_dir / chosen_aerofoil_x, delimiter=' ', dtype=np.float32, skiprows=1)  # output is np array
    x_target = coordinates[:, 0]
    x_target_half = x_target[len(x_target) // 2:]  # x target is symmetrical top and bottom

    # make all aerofoils same size
    for aerofoil in aerofoils:
        if aerofoils == chosen_aerofoil_x:
            continue  # no need to interpolate chosen aerofoil
        try:
            coordinates = np.loadtxt(root_dir / aerofoil, delimiter=' ', dtype=np.float32, skiprows=1)
            y_coord = coordinates[:, 1]
            x_coord = coordinates[:, 0]
            x_top = np.append(x_coord[len(y_coord) // 2:0:-1], x_coord[0])
            x_bottom = x_coord[len(y_coord) // 2:]
            y_top = np.append(y_coord[len(y_coord) // 2:0:-1], y_coord[0])
            y_bottom = y_coord[len(y_coord) // 2:]

            y_bottom_target = np.interp(x_target_half, x_bottom, y_bottom)
            y_top_target = np.interp(x_target_half, x_top, y_top)
            y_target = np.append(y_top_target[:0:-1], y_bottom_target)  # remove one of the two zeros

            np.savetxt(print_data / aerofoil, np.c_[x_target, y_target], fmt='%.4f')

            # plt.plot(x_coord, y_coord, 'r-')
            # plt.plot(x_target, y_target, 'bo')
            # plt.show()

        except Exception:  # as exc
            print(f"Error in file {aerofoil}. File ignored.")

    print(f"Code finished. Output folder: {print_data}.\n"
          f"Number of coordinates in every aerofoil file: {len(y_target)}")


start_code()
do_code()
