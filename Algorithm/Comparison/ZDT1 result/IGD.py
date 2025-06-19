import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances

# Function to compute inverted generational distance (IGD)
def compute_igd(pareto_front, data):
    distances = euclidean_distances(pareto_front, data)
    min_distances = np.min(distances, axis=1)  # Minimum distance for each Pareto front point to the solution set
    igd = np.sqrt(np.sum(min_distances**2)) / len(min_distances)  # Root mean square of the distances
    return igd

# Load the data from Excel files
nsga2_file = "NSGA2_ZDT1_results.xlsx"
spea2_file = "SPEA2_ZDT1_results.xlsx"
modbo_file = "MODBO_ZDT1_results.xlsx"

nsga2_data = pd.read_excel(nsga2_file, sheet_name=None)
spea2_data = pd.read_excel(spea2_file, sheet_name=None)
modbo_data = pd.read_excel(modbo_file, sheet_name=None)

# Extract the data from each Excel file
nsga2_values = nsga2_data['Sheet1'].values
spea2_values = spea2_data['Sheet1'].values
modbo_values = modbo_data['Sheet1'].values

# Define the true Pareto front for ZDT1 (for simplicity, a known approximation of the Pareto front)
pareto_front = np.array([[i/100.0, 1.0 - np.sqrt(i/100.0)] for i in range(101)])

# Compute IGD for NSGA2, SPEA2, and MODBO
igd_nsga2 = compute_igd(pareto_front, nsga2_values)
igd_spea2 = compute_igd(pareto_front, spea2_values)
igd_modbo = compute_igd(pareto_front, modbo_values)

# Plotting the IGD comparison
algorithms = ['NSGA2', 'SPEA2', 'MODBO']
inverted_generational_distances = [igd_nsga2, igd_spea2, igd_modbo]

plt.bar(algorithms, inverted_generational_distances, color=['blue', 'green', 'orange'])
plt.xlabel('Algorithm')
plt.ylabel('Inverted Generational Distance (IGD)')
plt.title('IGD Comparison: NSGA2 vs SPEA2 vs MODBO')
plt.show()

# Display the computed IGD values
print(f"NSGA2 Inverted Generational Distance: {igd_nsga2}")
print(f"SPEA2 Inverted Generational Distance: {igd_spea2}")
print(f"MODBO Inverted Generational Distance: {igd_modbo}")