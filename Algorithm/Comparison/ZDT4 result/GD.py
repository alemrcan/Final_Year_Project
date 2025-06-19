import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances

# Function to compute generational distance
def compute_gd(data, pareto_front):
    distances = euclidean_distances(data, pareto_front)
    min_distances = np.min(distances, axis=1)  # Minimum distance for each solution to the Pareto front
    gd = np.sqrt(np.sum(min_distances**2)) / len(min_distances)  # Root mean square of the distances
    return gd

# Load the data from Excel files
nsga2_file = "NSGA2_ZDT4_results.xlsx"
spea2_file = "SPEA2_ZDT4_results.xlsx"
modbo_file = "MODBO_ZDT4_results.xlsx"

nsga2_data = pd.read_excel(nsga2_file, sheet_name=None)
spea2_data = pd.read_excel(spea2_file, sheet_name=None)
modbo_data = pd.read_excel(modbo_file, sheet_name=None)

# Extract the data from each Excel file
nsga2_values = nsga2_data['Sheet1'].values
spea2_values = spea2_data['Sheet1'].values
modbo_values = modbo_data['Sheet1'].values

# Define the true Pareto front for ZDT1 (for simplicity, a known approximation of the Pareto front)
pareto_front = np.array([[i/100.0, 1.0 - np.sqrt(i/100.0)] for i in range(101)])

# Compute GD for NSGA2, SPEA2, and MODBO
gd_nsga2 = compute_gd(nsga2_values, pareto_front)
gd_spea2 = compute_gd(spea2_values, pareto_front)
gd_modbo = compute_gd(modbo_values, pareto_front)

# Plotting the generational distance comparison
algorithms = ['NSGA2', 'SPEA2', 'MODBO']
generational_distances = [gd_nsga2, gd_spea2, gd_modbo]

plt.bar(algorithms, generational_distances, color=['blue', 'green', 'orange'])
plt.xlabel('Algorithm')
plt.ylabel('Generational Distance (GD)')
plt.title('GD Comparison: NSGA2 vs SPEA2 vs MODBO')
plt.show()

# Display the computed GD values
print(f"NSGA2 Generational Distance: {gd_nsga2}")
print(f"SPEA2 Generational Distance: {gd_spea2}")
print(f"MODBO Generational Distance: {gd_modbo}")