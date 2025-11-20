import uproot
import matplotlib.pyplot as plt
import numpy as np


# Gaussian function definition
def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-((x - mean)**2) / (2 * std_dev**2))

# Plot PSD histogram with hardcoded Gaussian curves
def fit_and_plot_psd(psd_data, route_number):
    x = np.linspace(0, 4096, 4096)

    if route_number == 4:
        y1 = gaussian(x, 3411, 309, 24.36)
        y2 = gaussian(x, 253, 497, 45.29)

    elif route_number == 8:
        y1 = gaussian(x, 90, 494, 16.92)
        y2 = gaussian(x, 19, 625, 35.45)

    plt.plot(x, y1, 'b', label='Gaussian 1')
    plt.plot(x, y2, 'm', label='Gaussian 2')
    plt.title(f"PSD with Gaussian Fit (Route {route_number})")
    plt.xlabel("PSD")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid(True)
    plt.show()

# Extracts parameters from the root file and determines possible routes
def load_data(root_file_path):
    with uproot.open(root_file_path) as file:
        tree = file["T;1"]
        ea_values = tree["ea"].array()
        psd_values = tree["psd"].array()
        route_values = tree["route"].array()
        tof_values = tree["tof"].array()
        unique_routes = [4, 8]  # Known routes for this specific file
        return ea_values, psd_values, route_values, tof_values, unique_routes

# Energy calibration and redefinition function
def ecal(ea_array, route_number):
    if route_number == 4:  # NE-213
        ea_array = ea_array * 0.001455 - 0.20
        return ea_array

    elif route_number == 8:  # Stilbene
        ea_array = ea_array * 0.004372 - 0.20
        return ea_array
    
def tcal(tof_array, route_number):
    if route_number == 4:  # NE-213
        tof_array = tof_array * -0.22 + 932.53
        return tof_array
    
    elif route_number == 8:  # Stilbene
        tof_array = tof_array * -0.22 + 932.53
        return tof_array


# Plots any graph given a user selected route and histogram type
def plot_histograms_for_route(route_number, histogram_type, ea_values, psd_values, route_values, tof_values):
    mask_route = route_values == route_number
    ea_route = ea_values[mask_route]
    psd_route = psd_values[mask_route]
    tof_route = tof_values[mask_route]
    route_names_dict = {4: "NE-213", 8: "Stilbene"}

    if histogram_type in ['ea', 'all']:
        ea_route = ecal(ea_route, route_number) # Energy calibration function
        plt.figure(figsize=(10, 6))
        plt.hist(ea_route, bins = 4096, histtype='step', color='green')
        plt.title(f"Energy vs Counts ({route_names_dict[route_number]})")
        plt.xlabel("Energy (MeVee)")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.show()

    if histogram_type in ['tof', 'all']:
        tof_route = tcal(tof_route, route_number) # TOF calibration function
        plt.figure(figsize=(10, 6))
        plt.hist(tof_route, bins=4096, histtype='step', color='green')
        plt.title(f"Tof vs Counts ({route_names_dict[route_number]})")
        plt.xlabel("Tof (ns)")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.show()

    if histogram_type in ['psd', 'all']:
        plt.figure(figsize=(10, 6))
        plt.hist(psd_route, bins=4096, range=(0, 4096), histtype='step', color='green')
        plt.title(f"PSD vs Counts ({route_names_dict[route_number]})")
        plt.xlabel("PSD")
        plt.ylabel("Counts")
        plt.grid(True)
        fit_and_plot_psd(psd_route, route_number)
        plt.show()

    if histogram_type in ['epsd', 'all']:
        ea_route = ecal(ea_route, route_number)
        plt.figure(figsize=(10, 6))
        plt.scatter(ea_route, psd_route, s=1, color='green', alpha=0.5)
        plt.title(f"EA vs PSD ({route_names_dict[route_number]})")
        plt.xlabel("Energy (MeVee)")
        plt.ylabel("PSD")
        plt.grid(True)
        plt.show()

# ---- Main function ----
# Specify the path to your ROOT file
root_file_path = r"C:\Users\jackb\OneDrive - Ohio University\Desktop\VS Code\Root\INPP Fall 2025\run3200.root"

# Load data from the ROOT file
ea_values, psd_values, route_values, tof_values, unique_routes = load_data(root_file_path)

# User input for route selection and histogram type
while True:
    print(f"\nAvailable routes: {unique_routes}")
    try:
        selected_route = int(input("Select a route to filter: ").strip())
        if selected_route not in unique_routes:
            print(f"Invalid route. Please choose from: {unique_routes.tolist()}")
        else:
            break
    except ValueError:
        print("Route must be an integer. Try again.")

# User input for histogram type
while True:
    print("\nAvailable histogram types: ea, psd, epsd, tof, all")
    histogram_type = input("Enter histogram type to plot: ").strip().lower()
    if histogram_type in ['ea', 'psd', 'tof', 'epsd', 'all']:
        break
    else:
        print("Invalid histogram type. Please choose from: ea, psd, epsd, tof, all")

plot_histograms_for_route(selected_route, histogram_type, ea_values, psd_values, route_values, tof_values)


# ---- TO DO ----
#If psd is within a certain range (min, max), plot tof to determine neutron peak (Sum of gaussian fits curve fitting for finding neutron peak in tof)
#2D FOM for different energy slices (see where FOM goes below 1)
#Look at higher energy neutrons for Joespeh detector (check MeVee to MeV conversion is scaling correctly)
#Look at 5 MeV neutrons for Eljen-3  

# ---- DONE ----
# 1) Made general program for any root file with ea, psd, route, tof parameters
# 2) Now making one specific to run3200.root with known routes and detectors for ease of use 
#    - Energy calibration function for both detectors
#    - Hardcoded gaussian fits for both detectors in psd histogram

# ---- General Notes ----
#4 NE-213, 8 STILBENE
#2 MeV alphas - mike and cade