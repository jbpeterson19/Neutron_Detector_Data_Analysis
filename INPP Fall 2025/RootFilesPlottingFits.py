import uproot
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Gaussian model: sum of two Gaussians
def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
            a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))


# Fit and plot PSD histogram with fixed range and fixed initial guess
def fit_and_plot_psd(psd_data, route_number):

    #Get user input for initial guesses of fit parameters used in popt, _ = curve_fit
    max1 = float(input("Enter initial guess for maximum of Gaussian 1: ").strip())
    cent1 = float(input("Enter initial guess for centorid of Gaussian 1: ").strip())
    sigma1 = float(input("Enter initial guess for sigma of Gaussian 1: ").strip())
    max2 = float(input("Enter initial guess for maximum of Gaussian 2: ").strip())
    cent2 = float(input("Enter initial guess for centroid of Gaussian 2: ").strip())
    sigma2 = float(input("Enter initial guess for sigma of Gaussian 2: ").strip())
    psd_range = input("Enter range for fitting (Ex 200 800): ").strip()
    psd_range = [int(x) for x in psd_range.split()]

    #Get PSD data already found for route number and apply range filter
    psd_data = np.asarray(psd_data)
    psd_data = psd_data[(psd_data >= float(psd_range[0])) & (psd_data <= float(psd_range[1]))]

    #Extract counts and bin arrays for fitting
    counts, bin_edges = np.histogram(psd_data, bins=4096, range=(0, 4096))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fixed initial guess for all detectors
    initial_guess = [
        max1, cent1, sigma1,   # First peak: amplitude, mean, sigma
        max2, cent2, sigma2  # Second peak: amplitude, mean, sigma
    ]

    try:
        #Use curve fit to fit double gaussian to psd data and only save popt
        popt, _ = curve_fit(double_gaussian, bin_centers, counts, p0=initial_guess)

        # Print parameters
        print(f"\nFit parameters (Route {route_number}) [PSD 200–800]:")
        print(f"Gaussian 1: amplitude={popt[0]:.2f}, mean={popt[1]:.2f}, sigma={popt[2]:.2f}")
        print(f"Gaussian 2: amplitude={popt[3]:.2f}, mean={popt[4]:.2f}, sigma={popt[5]:.2f}")

        # Plot histogram and fit
        plt.figure(figsize=(10, 6))
        plt.hist(psd_data, bins=4096, range=(0, 4096), histtype='step', color='black', label='PSD Data')
        x_fit = np.linspace(200, 800, 1000)
        
        # Individual Gaussians
        g1 = popt[0] * np.exp(-(x_fit - popt[1])**2 / (2 * popt[2]**2))
        g2 = popt[3] * np.exp(-(x_fit - popt[4])**2 / (2 * popt[5]**2))
        plt.plot(x_fit, g1, 'b', label='Gaussian 1')
        plt.plot(x_fit, g2, 'm', label='Gaussian 2')

        plt.title(f"PSD with Gaussian Fit (Route {route_number})")
        plt.xlabel("PSD")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(True)
        plt.show()

    except RuntimeError:
        print("Fit failed. Try adjusting initial guesses or binning.")


# Gaussian model: sum of two Gaussians
def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (
        a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
        a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    )


# Fit and plot EA vs PSD with fixed range and fixed initial guess
def fit_and_plot_psd_ea(psd_data, ea_data, route_number):
    # Get user input for initial guesses
    #max1 = float(input("Enter initial guess for maximum of Gaussian 1: ").strip())
    #cent1 = float(input("Enter initial guess for centroid of Gaussian 1 (PSD): ").strip())
    #sigma1 = float(input("Enter initial guess for sigma of Gaussian 1: ").strip())
    #max2 = float(input("Enter initial guess for maximum of Gaussian 2: ").strip())
    #cent2 = float(input("Enter initial guess for centroid of Gaussian 2 (PSD): ").strip())
    #sigma2 = float(input("Enter initial guess for sigma of Gaussian 2: ").strip())
    #ea_range = input("Enter EA range for fitting (e.g., 200 800): ").strip()
    #ea_range = [float(x) for x in ea_range.split()]    

    #Initial parameters
    max1 = 2650
    cent1 = 350
    sigma1 = 10
    max2 = 1835
    cent2 = 500
    sigma2 = 30
    psd_range = [float(x) for x in '150 700'.split()]

    # Filter PSD and EA data together
    psd_data = np.asarray(psd_data)
    ea_data = np.asarray(ea_data)
    mask = (psd_data >= psd_range[0]) & (psd_data <= psd_range[1])
    psd_data_filtered = psd_data[mask]
    ea_data_filtered = ea_data[mask]


    # Fit EA as a function of PSD
    initial_guess = [max1, cent1, sigma1, max2, cent2, sigma2]

    try:
        popt, _ = curve_fit(double_gaussian, psd_data_filtered, ea_data_filtered, p0=initial_guess)

        print(f"\nFit parameters (Route {route_number}) [EAA {psd_range[0]}–{psd_range[1]}]:")
        print(f"Gaussian 1: amplitude={popt[0]:.2f}, mean={popt[1]:.2f}, sigma={popt[2]:.2f}")
        print(f"Gaussian 2: amplitude={popt[3]:.2f}, mean={popt[4]:.2f}, sigma={popt[5]:.2f}")

        # Plot EA vs PSD with fitted curve
        plt.figure(figsize=(10, 6))
        plt.scatter(psd_data_filtered, ea_data_filtered, s=10, color='black', label='EA vs PSD')
        x_fit = np.linspace(min(psd_data_filtered), max(psd_data_filtered), 1000)
        y_fit = double_gaussian(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r', label='Double Gaussian Fit')

        plt.title(f"EA vs PSD with Gaussian Fit (Route {route_number})")
        plt.xlabel("PSD")
        plt.ylabel("EA (Energy Axis)")
        plt.legend()
        plt.grid(True)
        plt.show()

    except RuntimeError:
        print("Fit failed. Try adjusting initial guesses or data range.")


# Extracts parameters from the root file and determines possible routes
def load_data(root_file_path):
    with uproot.open(root_file_path) as file:
        tree = file["T;1"]
        ea_values = tree["ea"].array()
        psd_values = tree["psd"].array()
        route_values = tree["route"].array()
        tof_values = tree["tof"].array()
        unique_routes = np.unique(route_values)
        return ea_values, psd_values, route_values, tof_values, unique_routes


# Plots any graph given a user selected route and histogram type
def plot_histograms_for_route(route_number, histogram_type, ea_values, psd_values, route_values, tof_values):
    mask_route = route_values == route_number
    ea_route = ea_values[mask_route]
    psd_route = psd_values[mask_route]
    tof_route = tof_values[mask_route]

    if histogram_type in ['ea', 'all']:
        plt.figure(figsize=(10, 6))
        plt.hist(ea_route, bins=4096,  range=(0, 4096), histtype='step', color='green')
        plt.title(f"EA vs Counts (route {route_number})")
        plt.xlabel("EA")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.show()

    if histogram_type in ['tof', 'all']:
        plt.figure(figsize=(10, 6))
        plt.hist(tof_route, bins=4096, range=(0, 4096), histtype='step', color='green')
        plt.title(f"TOF vs Counts (route {route_number})")
        plt.xlabel("TOF")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.show()

    if histogram_type in ['psd', 'all']:
        plt.figure(figsize=(10, 6))
        plt.hist(psd_route, bins=4096, range=(0, 4096), histtype='step', color='green')
        plt.title(f"PSD vs Counts (route {route_number})")
        plt.xlabel("PSD")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.show()

        Y_N = input("Would you like to fit the PSD histogram? (Y/N): ").strip().upper()
        if Y_N in ['Y', "y"]:
            fit_and_plot_psd(psd_route, route_number)
    

    if histogram_type in ['epsd', 'all']:

        # Define range
        min_val = 86
        max_val = 2500

        # Create a mask for the desired range
        mask = (ea_route >= min_val) & (ea_route <= max_val)

        # Apply the mask to both arrays
        filtered_ea = ea_route[mask]
        filtered_psd = psd_route[mask]


        #Plot EA vs PSD with filtered data
        plt.figure(figsize=(10, 6))

        # Plot full dataset in light gray
        plt.scatter(psd_route, ea_route, s=1, color='lightgray', alpha=0.3, label='Full Data')

        # Overlay filtered dataset in green
        plt.scatter(filtered_psd, filtered_ea, s=1, color='green', alpha=0.5, label='Filtered Data')

        plt.title(f"PSD vs EA (route {route_number})")
        plt.xlabel("EA")
        plt.ylabel("PSD")
        plt.grid(True)
        plt.legend()
        plt.show()

        Y_N = input("Would you like to fit the PSD histogram? (Y/N): ").strip().upper()
        if Y_N in ['Y', "y"]:
            fit_and_plot_psd_ea(filtered_psd, filtered_ea, route_number)


# Main function
root_file_path = (r'C:\Users\jackb\OneDrive - Ohio University\Desktop\VS Code\Root\INPP Fall 2025\run3200.root')


#Get parameter values from root file
ea_values, psd_values, route_values, tof_values, unique_routes = load_data(root_file_path)


#User selects route 
while True:
    print(f"\nAvailable routes: {unique_routes.tolist()}")
    try:
        selected_route = int(input("Select a route to filter: ").strip())
        if selected_route not in unique_routes:
            print(f"Invalid route. Please choose from: {unique_routes.tolist()}")
        else:
            break
    except ValueError:
        print("Route must be an integer. Try again.")

#User selects histogram type
while True:
    print("\nAvailable histogram types: ea, psd, epsd, tof, all")
    histogram_type = input("Enter histogram type to plot: ").strip().lower()
    if histogram_type in ['ea', 'psd', 'tof', 'epsd', 'all']:
        break
    else:
        print("Invalid histogram type. Please choose from: ea, psd, epsd, tof, all")

# Plot selected histogram
plot_histograms_for_route(selected_route, histogram_type, ea_values, psd_values, route_values, tof_values)


#If psd is within a certain range (min, max), plot tof to determine neutron peak (Sum of gaussian fits curve fitting for finding neutron peak in tof)
#Double check tof to make sure only plotting one detector with xy files
#Use energy cal for NE-213 x-axis
#2D FOM for different energy slices (see where FOM goes below 1)


#4 NE-213, 8 STILBENE
#2 MeV alphas - mike and cade