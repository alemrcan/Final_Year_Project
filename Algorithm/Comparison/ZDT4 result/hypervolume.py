import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

# Function to compute the hypervolume
def compute_hypervolume(data, ref_point):
    hv = HV(ref_point=ref_point)
    return hv.do(data)

# Load the data from Excel files
nsga2_file = "NSGA2_ZDT4_results.xlsx"
modbo_file = "MODBO_ZDT4_results.xlsx"
spea2_file = "SPEA2_ZDT4_results.xlsx"

nsga2_data = pd.read_excel(nsga2_file, sheet_name=None)
modbo_data = pd.read_excel(modbo_file, sheet_name=None)
spea2_data = pd.read_excel(spea2_file, sheet_name=None)

# Extract the data from each Excel file
nsga2_values = nsga2_data['Sheet1'].values
modbo_values = modbo_data['Sheet1'].values
spea2_values = spea2_data['Sheet1'].values

# Define the reference point for hypervolume calculation (should dominate all points)
max_nsga2 = np.max(nsga2_values, axis=0)
max_modbo = np.max(modbo_values, axis=0)
max_spea2 = np.max(spea2_values, axis=0)

# Determine the global maximum across all datasets
global_max = np.maximum.reduce([max_nsga2, max_modbo, max_spea2])
ref_point = global_max + 0.1  # or add a small constant, like 0.1, depending on the data scale

# Compute hypervolumes for NSGA2, SPEA2, and MODBO
hv_nsga2 = compute_hypervolume(nsga2_values, ref_point)
hv_modbo = compute_hypervolume(modbo_values, ref_point)
hv_spea2 = compute_hypervolume(spea2_values, ref_point)

# Plotting the hypervolume comparison
algorithms = ['NSGA2', 'MODBO', 'SPEA2']
hypervolumes = [hv_nsga2,hv_modbo, hv_spea2]

plt.bar(algorithms, hypervolumes, color=['blue', 'green', 'orange'])
plt.xlabel('Algorithm')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Comparison: NSGA2 vs MODBO vs SPEA2')
plt.show()

# Display the computed hypervolume values
print(f"NSGA2 Hypervolume: {hv_nsga2}")
print(f"MODBO Hypervolume: {hv_modbo}")
print(f"SPEA2 Hypervolume: {hv_spea2}")