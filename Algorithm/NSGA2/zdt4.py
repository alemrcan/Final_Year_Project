from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import pandas as pd
import math

# Define f1(x) for ZDT4
def f1_zdt4(x):
    return x[0]

# Define g(x) for ZDT4
def g_zdt4(x):
    n = len(x)
    s = sum([x[i]**2 - 10 * math.cos(4 * math.pi * x[i]) for i in range(1, n)])
    return 1 + 10 * (n - 1) + s

# Define f2(x) for ZDT4
def f2_zdt4(x):
    gx = g_zdt4(x)
    return gx * (1 - math.sqrt(x[0] / gx))

# Set the problem based on ZDT4
problem = Problem(num_of_variables=10, objectives=[f1_zdt4, f2_zdt4], variables_range=[(0, 1), (-5, 5)], same_range=False, expand=False)

# Evolutionary algorithm setup
evo_zdt4 = Evolution(problem, mutation_param=20)
final_population = evo_zdt4.evolve()

# Extracting objectives for plotting
func_zdt4 = [ind.objectives for ind in final_population]
function1_zdt4 = [i[0] for i in func_zdt4]
function2_zdt4 = [i[1] for i in func_zdt4]

# Creating a DataFrame from function1 and function2
df = pd.DataFrame({'Function 1': function1_zdt4,'Function 2': function2_zdt4})

# Saving the DataFrame to an Excel file
excel_filename = 'NSGA2_ZDT4_results.xlsx'
df.to_excel(excel_filename, index=False)

# Plotting the results
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1_zdt4, function2_zdt4)
plt.title('ZDT4 Function')
plt.show()

print(f'Data has been saved to {excel_filename}')