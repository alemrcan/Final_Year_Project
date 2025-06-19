import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean(file_path):
    df = pd.read_excel(file_path)
    
    # Convert to numeric; invalid strings like '▲' or '●' become NaN
    df['Function 1'] = pd.to_numeric(df['Function 1'], errors='coerce')
    df['Function 2'] = pd.to_numeric(df['Function 2'], errors='coerce')
    
    # Drop rows with NaNs (i.e., corrupted or non-numeric entries)
    df = df.dropna(subset=['Function 1', 'Function 2'])
    
    # Drop exact duplicates in (Function 1, Function 2)
    df = df.drop_duplicates(subset=['Function 1', 'Function 2'])
    
    return df


# Load data
nsga2_df = load_and_clean('NSGA2_ZDT3_results.xlsx')
spea2_df = load_and_clean('SPEA2_ZDT3_results.xlsx')
new_df = load_and_clean('MODBO_ZDT3_results.xlsx')



# Plot the data
plt.figure(figsize=(10, 6))

plt.scatter(nsga2_df['Function 1'], nsga2_df['Function 2'], label='NSGA-II', color='blue',marker='^')
plt.scatter(spea2_df['Function 1'], spea2_df['Function 2'], label='SPEA2', color='red', marker='x')
plt.scatter(new_df['Function 1'], new_df['Function 2'], label='MODBO', color='green', marker='o')

# Labeling
plt.title('Comparison of NSGA-II, SPEA2, and MODBO on ZDT3')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
