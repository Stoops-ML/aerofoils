from urllib import request
import sys
import re
from pathlib import Path
import os

# folders
root_dir = Path('data')
out_dir = root_dir / 'auto_downloaded_files'
out_dir.mkdir(exist_ok=True)

aerofoils_ignored = 0
response = request.urlopen('http://airfoiltools.com/search/airfoils')
for line in response:
    line_str = str(line, 'utf-8')  # convert bytes to string
    line_with_aerofoils = re.findall(r'">.{,15} - ', line_str)  # find all aerofoils on one line of source code

    if line_with_aerofoils:
        for aerofoil_found in line_with_aerofoils:
            url_find = re.search(r'^(">)(.+) - ', aerofoil_found)  # should this be (\.+) instead?
            url_identifier = url_find.group(2)
            aerofoil_name_find = re.search(r'[\w\d]+', url_identifier)
            aerofoil_name = aerofoil_name_find.group()

            if Path.exists(out_dir / (aerofoil_name + ".csv")):
                continue

            try:
                # get coordinates
                aerofoil_coords = request.urlopen('http://airfoiltools.com/airfoil/seligdatfile?airfoil=' +
                                                  url_identifier)
                x = []
                y = []
                for coords in aerofoil_coords:
                    coords_str = str(coords, 'utf-8').strip()  # convert bytes to string
                    xy = re.search(r'^([\+-]?\d*\.?\d*)\s+([\+-]?\d*\.?\d*)$', coords_str)
                    if xy:
                        x.append(float(xy.group(1)))
                        y.append(float(xy.group(2)))
                # aerofoils[aerofoil_name] = x, y

                # get maximum Cl/Cd at angle
                data = request.urlopen('http://airfoiltools.com/airfoil/details?airfoil=' + url_identifier)
                for line_data in data:
                    line_data_str = str(line_data, 'utf-8')  # convert bytes to string
                    line_with_data = re.search(r'(">).{,40}(deg;)', line_data_str)
                    if line_with_data:
                        ClCd_angle = [float(num) for num in re.findall(r'([+-]?\d*[.]?\d*)', line_with_data.group()) if num != '']
                        break  # we want the first values, which occur at Re = 50,000, Ncrit = 9
                max_ClCd = ClCd_angle[0]
                angle = ClCd_angle[1]

                # print aerofoil file
                with open(out_dir / (aerofoil_name + ".csv"), 'w') as f:
                    # f.write(aerofoil_name + '\n')
                    f.write(f"Max ClCd {max_ClCd} at {angle}deg\n")
                    for x_target, y_target in zip(x, y):
                        f.write(f"{x_target:.4f} {y_target:.4f}\n")

            except Exception as excp:
                aerofoils_ignored += 1
                print(f"Error with of aerofoil {aerofoil_name}\n"
                      f"URL identifier: {url_identifier}\n"
                      f"Error: {excp}\n")

print(f"Number of aerofoils downloaded = {len(os.listdir(out_dir))}")
print(f"Number of aerofoils ignored = {aerofoils_ignored}")
