import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 

# Function: Initialize Variables
def initial_population(population_size=100, min_values=[0]*30, max_values=[1]*30, list_of_functions=[None, None]):
    population = np.zeros((population_size, len(min_values) + len(list_of_functions)))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
            population[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range(1, len(list_of_functions) + 1):
            population[i,-k] = list_of_functions[-k](list(population[i,0:population.shape[1]-len(list_of_functions)]))
    return population

# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions=2):
    count = 0
    for k in range(1, number_of_functions + 1):
        if solution_1[-k] <= solution_2[-k]:
            count += 1
    return count == number_of_functions

# Function: Raw Fitness
def raw_fitness_function(population, number_of_functions=2):    
    strength = np.zeros((population.shape[0], 1))
    raw_fitness = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j and dominance_function(population[i,:], population[j,:], number_of_functions):
                strength[i,0] += 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j and dominance_function(population[i,:], population[j,:], number_of_functions):
                raw_fitness[j,0] += strength[i,0]
    return raw_fitness

# Function: Euclidean Distance
def euclidean_distance(x, y):       
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

# Function: Fitness Calculation
def fitness_calculation(population, raw_fitness, number_of_functions=2):
    k = int(len(population)**(1/2)) - 1
    fitness  = np.zeros((population.shape[0], 1))
    distance = np.zeros((population.shape[0], population.shape[0]))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j:
                x = population[i, -number_of_functions:]
                y = population[j, -number_of_functions:]
                distance[i,j] = euclidean_distance(x, y)
    for i in range(0, fitness.shape[0]):
        distance_ordered = np.sort(distance[i,:])
        fitness[i,0] = raw_fitness[i,0] + 1 / (distance_ordered[k] + 2)
    return fitness

# Function: Sort Population by Fitness
def sort_population_by_fitness(population, fitness):
    idx = np.argsort(fitness[:, -1])
    fitness_new = fitness[idx]
    population_new = population[idx]
    return population_new, fitness_new

# Function: Roulette Wheel Selection
def roulette_wheel(fitness_new): 
    fitness = np.zeros((fitness_new.shape[0], 2))
    fitness[:, 0] = 1 / (1 + fitness_new[:, 0] + abs(fitness_new[:, 0].min()))
    fitness[:, 1] = np.cumsum(fitness[:, 0])
    fit_sum = fitness[:, 1].max()
    fitness[:, 1] /= fit_sum
    random_value = random.random()
    return np.searchsorted(fitness[:, 1], random_value)

# Function: Offspring Creation
def breeding(population, fitness, min_values=[0]*30, max_values=[1]*30, mu=1, list_of_functions=[None, None]):
    offspring = np.copy(population)
    for i in range(0, offspring.shape[0], 2):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            beta = np.power(2*random.random(), 1/(mu+1))
            offspring[i,j] = np.clip(((1 + beta)*population[parent_1, j] + (1 - beta)*population[parent_2, j]) / 2, min_values[j], max_values[j])
            if i+1 < offspring.shape[0]:
                offspring[i+1,j] = np.clip(((1 - beta)*population[parent_1, j] + (1 + beta)*population[parent_2, j]) / 2, min_values[j], max_values[j])
        for k in range(1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring

# Function: Mutation
def mutation(offspring, mutation_rate=0.02, eta=1, min_values=[0]*30, max_values=[1]*30, list_of_functions=[None, None]):
    for i in range(0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            if random.random() < mutation_rate:
                delta = np.power(2 * random.random(), 1/(eta + 1)) - 1
                offspring[i,j] = np.clip(offspring[i,j] + delta, min_values[j], max_values[j])
        for k in range(1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring

# SPEA-2 Function
def strength_pareto_evolutionary_algorithm_2(population_size=100, archive_size=100, mutation_rate=0.02, min_values=[0]*30, max_values=[1]*30, list_of_functions=[None, None], generations=1000, mu=1, eta=1):        
    population = initial_population(population_size=population_size, min_values=min_values, max_values=max_values, list_of_functions=list_of_functions) 
    archive = initial_population(population_size=archive_size, min_values=min_values, max_values=max_values, list_of_functions=list_of_functions)     

    for count in tqdm(range(generations), desc="Generation Progress"):
        population = np.vstack([population, archive])
        raw_fitness = raw_fitness_function(population, number_of_functions=len(list_of_functions))
        fitness = fitness_calculation(population, raw_fitness, number_of_functions=len(list_of_functions))        
        population, fitness = sort_population_by_fitness(population, fitness)
        population, archive, fitness = population[0:population_size,:], population[0:archive_size,:], fitness[0:archive_size,:]
        population = breeding(population, fitness, mu=mu, min_values=min_values, max_values=max_values, list_of_functions=list_of_functions)
        population = mutation(population, mutation_rate=mutation_rate, eta=eta, min_values=min_values, max_values=max_values, list_of_functions=list_of_functions)             
    return archive            

# ZDT1 Function 1
def f1_zdt1(x):
    return x[0]

# ZDT1 Function 2
def f2_zdt1(x):
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    return g * (1 - math.sqrt(x[0] / g))

# Running SPEA-2 for ZDT1
spea_2_zdt1 = strength_pareto_evolutionary_algorithm_2(population_size=100, archive_size=100, mutation_rate=0.02, min_values=[0]*30, max_values=[1]*30, list_of_functions=[f1_zdt1, f2_zdt1], generations=1000, mu=1, eta=1)

# Store Function 1 and 2 values
func_1_values = spea_2_zdt1[:,-2]
func_2_values = spea_2_zdt1[:,-1]

# Save data to a DataFrame and export to an Excel file
df = pd.DataFrame({'Function 1': func_1_values, 'Function 2': func_2_values})
file_path = "SPEA2_ZDT1_results.xlsx"
df.to_excel(file_path, index=False)

print(f"Data has been stored in: {file_path}")

# Plotting the ZDT1 Pareto Front
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(func_1_values, func_2_values, c='red', s=25, marker='o', label='SPEA-2')
plt.title('ZDT1 Function')
plt.show()