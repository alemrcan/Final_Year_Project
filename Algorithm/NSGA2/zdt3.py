from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
import pandas as pd
import math

# Objective functions for ZDT3
def f1_zdt3(x):
    return x[0]

def g_zdt3(x):
    return 1 + 9 * sum(x[1:]) / (len(x) - 1)

def f2_zdt3(x):
    f1 = f1_zdt3(x)
    g = g_zdt3(x)
    return g * (1 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10 * math.pi * f1))

# Defining the ZDT3 problem
problem_zdt3 = Problem(num_of_variables=30, objectives=[f1_zdt3, f2_zdt3], variables_range=[(0, 1)], same_range=True, expand=False)

# Evolving the population
evo_zdt3 = Evolution(problem_zdt3, mutation_param=20)
final_population = evo_zdt3.evolve()

# Extracting objectives for plotting
func_zdt3 = [ind.objectives for ind in final_population]
function1_zdt3 = [i[0] for i in func_zdt3]
function2_zdt3 = [i[1] for i in func_zdt3]

# Creating a DataFrame from function1 and function2
df = pd.DataFrame({'Function 1': function1_zdt3,'Function 2': function2_zdt3})

# Saving the DataFrame to an Excel file
excel_filename = 'NSGA2_ZDT3_results.xlsx'
df.to_excel(excel_filename, index=False)

# Plotting the results
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1_zdt3, function2_zdt3)
plt.title('ZDT3 Function')
plt.show()

print(f'Data has been saved to {excel_filename}')