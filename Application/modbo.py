from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
from jmetal.util.solution import get_non_dominated_solutions
from typing import List
import numpy as np
import random
import copy

class MODBO(EvolutionaryAlgorithm):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 max_evaluations: int,
                 crossover: Crossover,
                 mutation: Mutation,
                 selection: Selection,
                 convergence_threshold: float = 1e-6):

        super().__init__(problem, population_size, offspring_population_size=population_size)

        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.crossover = crossover
        self.mutation = mutation
        self.selection_operator = selection
        self.archive = CrowdingDistanceArchive(population_size)

        self.evaluations = 0
        self.previous_population = []
        self.convergence_threshold = convergence_threshold

    def get_name(self) -> str:
        return "MODBO"

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solutions: List[FloatSolution]) -> List[FloatSolution]:
        for solution in solutions:
            self.problem.evaluate(solution)
        return solutions

    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        return [self.selection_operator.execute(population) for _ in range(self.population_size)]

    def reproduction(self, mating_pool: List[FloatSolution]) -> List[FloatSolution]:
        offspring = []
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1 if i + 1 < len(mating_pool) else i]
            children = self.crossover.execute([parent1, parent2])
            for child in children:
                offspring.append(self.mutation.execute(child))
        return offspring

    def fast_non_dominated_sort(self, solutions: List[FloatSolution]) -> List[List[FloatSolution]]:
        fronts = [[]]
        domination_counts = {}
        domination_sets = {}

        for p in solutions:
            domination_sets[p] = []
            domination_counts[p] = 0
            for q in solutions:
                if p == q:
                    continue
                if self.dominates(p, q):
                    domination_sets[p].append(q)
                elif self.dominates(q, p):
                    domination_counts[p] += 1
            if domination_counts[p] == 0:
                p.attributes['rank'] = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in domination_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        q.attributes['rank'] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts

    def dominates(self, s1: FloatSolution, s2: FloatSolution) -> bool:
        and_condition = all(x <= y for x, y in zip(s1.objectives, s2.objectives))
        or_condition = any(x < y for x, y in zip(s1.objectives, s2.objectives))
        return and_condition and or_condition

    def _compute_crowding_distance(self, front: List[FloatSolution]) -> None:
        if not front:
            return
        num_objectives = len(front[0].objectives)
        for solution in front:
            solution.attributes['crowding_distance'] = 0.0

        for m in range(num_objectives):
            front.sort(key=lambda x: x.objectives[m])
            front[0].attributes['crowding_distance'] = float('inf')
            front[-1].attributes['crowding_distance'] = float('inf')
            m_values = [s.objectives[m] for s in front]
            min_value = min(m_values)
            max_value = max(m_values)
            if max_value - min_value == 0:
                continue
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / (max_value - min_value)
                front[i].attributes['crowding_distance'] += distance

    def replacement(self, population: List[FloatSolution], offspring: List[FloatSolution]) -> List[FloatSolution]:
        combined = population + offspring
        fronts = self.fast_non_dominated_sort(combined)

        new_population = []
        for front in fronts:
            self._compute_crowding_distance(front)
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                front.sort(key=lambda s: s.attributes['crowding_distance'], reverse=True)
                new_population.extend(front[:self.population_size - len(new_population)])
                break

        for s in new_population:
            self.archive.add(s)
        return new_population

    def result(self) -> List[FloatSolution]:
        return self.archive.solution_list

    def init_progress(self) -> None:
        self.evaluations = 0
        self.solutions = self.create_initial_solutions()
        self.evaluate(self.solutions)
        for s in self.solutions:
            self.archive.add(s)
        self.previous_population = [copy.deepcopy(s) for s in self.solutions]

    def update_progress(self) -> None:
        self.evaluations += self.population_size

    def stopping_condition_is_met(self) -> bool:
        if self.evaluations >= self.max_evaluations:
            return True
        avg_dist = np.mean([
            np.linalg.norm(np.array(self.solutions[i].objectives) - np.array(self.previous_population[i].objectives))
            for i in range(self.population_size)
        ])
        return avg_dist < self.convergence_threshold

    def run(self) -> None:
        self.init_progress()

        while not self.stopping_condition_is_met():
            mating_pool = self.selection(self.solutions)
            offspring = self.reproduction(mating_pool)
            self.evaluate(offspring)

            best = min(self.archive.solution_list, key=RankingAndCrowdingDistanceComparator())
            self.solutions = self.update_beetle_positions(self.solutions, best)
            self.evaluate(self.solutions)

            self.solutions = self.replacement(self.solutions, offspring)
            self.previous_population = [copy.deepcopy(s) for s in self.solutions]
            self.update_progress()

        self.solutions = self.archive.solution_list

    def update_beetle_positions(self, population: List[FloatSolution], best: FloatSolution) -> List[FloatSolution]:
        updated = []
        for x in population:
            new_x = copy.deepcopy(x)
            for i in range(len(x.variables)):
                alpha = random.uniform(0.5, 1.0)
                beta = random.uniform(0.0, 0.1)
                direction = alpha * (best.variables[i] - x.variables[i]) + beta * random.uniform(-1, 1)
                new_x.variables[i] = min(max(x.variables[i] + direction, self.problem.lower_bound[i]), self.problem.upper_bound[i])
            updated.append(new_x)
        return updated
