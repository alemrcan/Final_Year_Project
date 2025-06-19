from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem.multiobjective.zdt import ZDT4
from jmetal.util.termination_criterion import StoppingByEvaluations

import pandas as pd

# Define the problem
problem = ZDT4()

# Define the algorithm (fixed: removed 'archive')
algorithm = SPEA2(
    problem=problem,
    population_size=100,
    offspring_population_size=200,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
    crossover=SBXCrossover(probability=0.9, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=25000)
)

# Run the algorithm
algorithm.run()
result = algorithm.solutions

# Extract and clean the result
objectives = [solution.objectives for solution in result]
df = pd.DataFrame(objectives, columns=['Function 1', 'Function 2'])
df = df.drop_duplicates()

# Save to Excel
df.to_excel('SPEA2_ZDT4_results1.xlsx', index=False)
print("✅ Optimization complete. Results saved to 'SPEA2_ZDT4_results1.xlsx'.")
