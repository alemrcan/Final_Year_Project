from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import pandas as pd
import math
# Define f1(x)
def f1_zdt2(x):
    return x[0]

# Define g(x)
def g_zdt2(x):
    n = len(x)
    s = sum(x[i] for i in range(1, n))  # Sum from x2 to xn
    return 1 + 9 * s / (n - 1)

# Define f2(x)
def f2_zdt2(x):
    gx = g_zdt2(x)
    return gx * (1 - (x[0] / gx) ** 2)

# Set the problem based on ZDT2
problem_zdt2 = Problem(num_of_variables=30, objectives=[f1_zdt2, f2_zdt2], variables_range=[(0, 1)], same_range=True, expand=False)

# Evolutionary algorithm setup
evo_zdt2 = Evolution(problem_zdt2, mutation_param=20)
final_population = evo_zdt2.evolve()

# Extracting objectives for plotting
func_zdt2 = [ind.objectives for ind in final_population]
function1_zdt2 = [i[0] for i in func_zdt2]
function2_zdt2 = [i[1] for i in func_zdt2]

# Creating a DataFrame from function1 and function2
df = pd.DataFrame({'Function 1': function1_zdt2,'Function 2': function2_zdt2})

# Saving the DataFrame to an Excel file
excel_filename = 'NSGA2_ZDT2_results.xlsx'
df.to_excel(excel_filename, index=False)

# Plotting the results
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1_zdt2, function2_zdt2)
plt.title('ZDT2 Function')
plt.show()

print(f'Data has been saved to {excel_filename}')