import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Configuration for ZDT1 Problem
class Problem:
    def __init__(self, num_of_variables, objectives, variables_range, same_range=True, expand=False):
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.variables_range = variables_range
        self.same_range = same_range
        self.expand = expand

# Define ZDT1 objective functions
def f1_zdt1(x):
    return x[0]

def f2_zdt1(x):
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h = 1 - (x[0] / g) ** 0.5
    return g * h

# Initialize the problem
problem_zdt1 = Problem(
    num_of_variables=30, 
    objectives=[f1_zdt1, f2_zdt1], 
    variables_range=[(0, 1)], 
    same_range=True, 
    expand=False
)

# Parameters
population_size = 200
max_generations = 1000
archive_size = 200
crossover_prob = 0.7
mutation_prob = 0.02
num_objectives = 2

# Fitness Function for ZDT1 Problem
def evaluate_fitness(individual):
    f1 = problem_zdt1.objectives[0](individual)
    f2 = problem_zdt1.objectives[1](individual)
    return np.array([f1, f2])

# Initialize Population based on Problem Configuration
def initialize_population(problem):
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(
            problem.variables_range[0][0], 
            problem.variables_range[0][1], 
            problem.num_of_variables
        )
        f1 = problem.objectives[0](individual)
        f2 = problem.objectives[1](individual)
        fitness = np.array([f1, f2])
        population.append({'position': individual, 'fitness': fitness})
    return population

# Non-Dominated Sorting (NSGA-II)
def non_dominated_sorting(population):
    fronts = [[]]
    for i, p in enumerate(population):
        p['dominated_by'] = []
        p['dominates'] = 0
        for j, q in enumerate(population):
            if dominates(p['fitness'], q['fitness']):
                p['dominated_by'].append(j)
            elif dominates(q['fitness'], p['fitness']):
                p['dominates'] += 1
        if p['dominates'] == 0:
            fronts[0].append(i)
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for idx in fronts[current_front]:
            for dominated_idx in population[idx]['dominated_by']:
                population[dominated_idx]['dominates'] -= 1
                if population[dominated_idx]['dominates'] == 0:
                    next_front.append(dominated_idx)
        current_front += 1
        fronts.append(next_front)
    return fronts[:-1]

# Dominance Check
def dominates(individual1, individual2):
    return all(ind1 <= ind2 for ind1, ind2 in zip(individual1, individual2)) and any(
        ind1 < ind2 for ind1, ind2 in zip(individual1, individual2))

# Crowding Distance Calculation
def crowding_distance(front, population):
    if not front:
        return
    for p in front:
        population[p]['distance'] = 0
    for obj in range(num_objectives):
        front.sort(key=lambda x: population[x]['fitness'][obj])
        min_f = population[front[0]]['fitness'][obj]
        max_f = population[front[-1]]['fitness'][obj]
        population[front[0]]['distance'] = population[front[-1]]['distance'] = float('inf')
        for i in range(1, len(front) - 1):
            if max_f != min_f:
                population[front[i]]['distance'] += (population[front[i + 1]]['fitness'][obj] - population[front[i - 1]]['fitness'][obj]) / (max_f - min_f)

# Archive Maintenance (SPEA-2)
def update_archive(population, archive):
    combined_population = population + archive
    fronts = non_dominated_sorting(combined_population)
    new_archive = []
    for front in fronts:
        if len(new_archive) + len(front) > archive_size:
            crowding_distance(front, combined_population)
            front.sort(key=lambda x: combined_population[x]['distance'], reverse=True)
        new_archive.extend(front)
        if len(new_archive) >= archive_size:
            break
    return [combined_population[i] for i in new_archive[:archive_size]]

# Update Positions Based on Dung Beetle Behavior
def update_position(beetle, population, archive, iteration):
    if beetle['type'] == 'ball-rolling':
        worst = max(population, key=lambda x: np.sum(x['fitness']))
        delta_x = abs(beetle['position'] - worst['position'])
        beetle['position'] += np.random.uniform() * delta_x
    elif beetle['type'] == 'breeding':
        best = min(archive, key=lambda x: np.sum(x['fitness']))
        beetle['position'] += np.random.uniform() * (beetle['position'] - best['position'])
    elif beetle['type'] == 'small':
        best = min(archive, key=lambda x: np.sum(x['fitness']))
        worst = max(archive, key=lambda x: np.sum(x['fitness']))
        beetle['position'] += np.random.uniform() * (best['position'] - worst['position'])
    elif beetle['type'] == 'thieving':
        gbest = min(archive, key=lambda x: np.sum(x['fitness']))
        beetle['position'] += np.random.uniform() * (abs(beetle['position'] - gbest['position']))
    beetle['position'] = np.clip(beetle['position'], problem_zdt1.variables_range[0][0], problem_zdt1.variables_range[0][1])
    beetle['fitness'] = evaluate_fitness(beetle['position'])

# Selection, Crossover, and Mutation (NSGA-II)
def selection(population):
    selected = []
    fronts = non_dominated_sorting(population)
    for front in fronts:
        crowding_distance(front, population)
        selected.extend(front)
        if len(selected) >= population_size:
            break
    return [population[i] for i in selected[:population_size]]

def crossover(parent1, parent2):
    if np.random.rand() < crossover_prob:
        point = np.random.randint(1, problem_zdt1.num_of_variables)
        child1 = np.concatenate((parent1['position'][:point], parent2['position'][point:]))
        child2 = np.concatenate((parent2['position'][:point], parent1['position'][point:]))
        return [{'position': child1, 'fitness': evaluate_fitness(child1)}, {'position': child2, 'fitness': evaluate_fitness(child2)}]
    return [parent1, parent2]

def mutate(individual):
    if np.random.rand() < mutation_prob:
        index = np.random.randint(0, problem_zdt1.num_of_variables)
        individual['position'][index] += np.random.uniform(-0.1, 0.1)
        individual['position'] = np.clip(individual['position'], problem_zdt1.variables_range[0][0], problem_zdt1.variables_range[0][1])
        individual['fitness'] = evaluate_fitness(individual['position'])

# Main Algorithm
def multi_objective_dung_beetle_optimization():
    population = initialize_population(problem_zdt1)
    archive = []

    for generation in tqdm(range(max_generations), desc="Generations Progress"):
        for beetle in population:
            beetle['fitness'] = evaluate_fitness(beetle['position'])
            beetle['type'] = random.choice(['ball-rolling', 'breeding', 'small', 'thieving'])

        archive = update_archive(population, archive)

        for beetle in population:
            update_position(beetle, population, archive, generation)

        next_generation = []
        while len(next_generation) < population_size:
            parents = random.sample(population, 2)
            offspring = crossover(parents[0], parents[1])
            for child in offspring:
                mutate(child)
                next_generation.append(child)
        population = selection(next_generation)

    return archive

# Run the algorithm
final_pareto_front = multi_objective_dung_beetle_optimization()

# Extracting the fitness values for exporting
f1_values = [solution['fitness'][0] for solution in final_pareto_front]
f2_values = [solution['fitness'][1] for solution in final_pareto_front]

# Creating a DataFrame to store only the ZDT1 function outputs (Function 1 and Function 2)
df = pd.DataFrame({'Function 1': f1_values,'Function 2': f2_values})

# Saving the DataFrame to an Excel file
excel_filename = 'MODBO_ZDT1_results.xlsx'
df.to_excel(excel_filename, index=False)

# Plotting the Pareto front
plt.scatter(f1_values, f2_values, color="blue", s=10)
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('MODBO ZDT1 Function')
#plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.show()

print(f'Data has been saved to {excel_filename}')