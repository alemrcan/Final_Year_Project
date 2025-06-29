import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import pandas as pd

# Configuration for ZDT4 Problem
class Problem:
    def __init__(self, num_of_variables, objectives, variables_range, same_range=False, expand=False):
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.variables_range = variables_range
        self.same_range = same_range
        self.expand = expand

# Define ZDT4 objective functions
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

# Initialize the problem
problem_zdt4 = Problem(
    num_of_variables=10, 
    objectives=[f1_zdt4, f2_zdt4], 
    variables_range=[(0, 1)] + [(-5, 5)],  # First variable is [0,1], others are [-5,5]
    same_range=False, 
    expand=False
)

# Parameters
population_size = 100
max_generations = 1000
archive_size = 100
crossover_prob = 0.7
mutation_prob = 0.02
num_objectives = 2

# Fitness Function for ZDT4 Problem
def evaluate_fitness(individual):
    f1 = problem_zdt4.objectives[0](individual)
    f2 = problem_zdt4.objectives[1](individual)
    return np.array([f1, f2])

# Initialize Population based on Problem Configuration
def initialize_population(problem):
    population = []
    for _ in range(population_size):
        individual = np.array([np.random.uniform(low, high) for low, high in problem.variables_range])
        fitness = evaluate_fitness(individual)
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
def update_position(beetle, population, archive):
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
    beetle['position'] = np.clip(beetle['position'], [r[0] for r in problem_zdt4.variables_range], [r[1] for r in problem_zdt4.variables_range])
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
        point = np.random.randint(1, problem_zdt4.num_of_variables)
        child1 = np.concatenate((parent1['position'][:point], parent2['position'][point:]))
        child2 = np.concatenate((parent2['position'][:point], parent1['position'][point:]))
        return [{'position': child1, 'fitness': evaluate_fitness(child1)}, 
                {'position': child2, 'fitness': evaluate_fitness(child2)}]
    return [parent1, parent2]

def mutate(individual):
    if np.random.rand() < mutation_prob:
        # Choose an index based on the current size of the individual
        index = np.random.randint(0, individual['position'].size)
        # Mutate the chosen index
        individual['position'][index] += np.random.uniform(-0.1, 0.1)
        # Ensure the position is within bounds
        individual['position'] = np.clip(individual['position'], 
                                          [r[0] for r in problem_zdt4.variables_range], 
                                          [r[1] for r in problem_zdt4.variables_range])
        # Update fitness after mutation
        individual['fitness'] = evaluate_fitness(individual['position'])

# Main Algorithm
def multi_objective_dung_beetle_optimization():
    population = initialize_population(problem_zdt4)
    archive = []

    for generation in tqdm(range(max_generations), desc="Generations Progress"):
        for beetle in population:
            beetle['fitness'] = evaluate_fitness(beetle['position'])
            beetle['type'] = random.choice(['ball-rolling', 'breeding', 'small', 'thieving'])

        archive = update_archive(population, archive)

        for beetle in population:
            update_position(beetle, population, archive)

        next_generation = []
        while len(next_generation) < population_size:
            parents = random.sample(population, 2)
            offspring = crossover(parents[0], parents[1])
            for child in offspring:
                mutate(child)
                next_generation.append(child)
        population = selection(next_generation)

    return archive

# Run the algorithm and visualize the results
final_pareto_front = multi_objective_dung_beetle_optimization()

# Extracting the fitness values for plotting
fitness_values = np.array([ind['fitness'] for ind in final_pareto_front])

# Convert the fitness values to a DataFrame
df = pd.DataFrame(fitness_values, columns=['f1', 'f2'])

# Saving the DataFrame to an Excel file
excel_filename = 'MODBO_ZDT4_results.xlsx'
df.to_excel(excel_filename, index=False)

# Plotting
plt.scatter(fitness_values[:, 0], fitness_values[:, 1], c='blue', marker='o')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('MODBO ZDT4 Function')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

print(f'Data has been saved to {excel_filename}')