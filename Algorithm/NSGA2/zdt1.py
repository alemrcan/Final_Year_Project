import pandas as pd
from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import math

# Objective functions for ZDT1
def f1_zdt1(x):
    return x[0]

def g_zdt1(x):
    return 1 + 9 * sum(x[1:]) / (len(x) - 1)

def f2_zdt1(x):
    f1 = f1_zdt1(x)
    g = g_zdt1(x)
    return g * (1 - math.sqrt(f1 / g))

# Defining the ZDT1 problem
problem_zdt1 = Problem(num_of_variables=30, objectives=[f1_zdt1, f2_zdt1], variables_range=[(0, 1)], same_range=True, expand=False)

# Evolving the population
evo_zdt1 = Evolution(problem_zdt1, mutation_param=20)
final_population = evo_zdt1.evolve()

# Extracting objectives for plotting
func_zdt1 = [ind.objectives for ind in final_population]
function1_zdt1 = [obj[0] for obj in func_zdt1]
function2_zdt1 = [obj[1] for obj in func_zdt1]

# Creating a DataFrame to store only the ZDT1 function outputs (Function 1 and Function 2)
df = pd.DataFrame({'Function 1': function1_zdt1,'Function 2': function2_zdt1})

# Saving the DataFrame to an Excel file
excel_filename = 'NSGA2_ZDT1_results.xlsx'
df.to_excel(excel_filename, index=False)

# Plotting the results
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1_zdt1, function2_zdt1)
plt.title('ZDT1 Function')
plt.show()

print(f'Data has been saved to {excel_filename}')