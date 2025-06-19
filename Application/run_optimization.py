import matplotlib.pyplot as plt
import pandas as pd
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import BinaryTournamentSelection, PolynomialMutation, SBXCrossover
from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.core.solution import FloatSolution  
from doodptw import DOODPTW
from modbo import MODBO

def filter_dominated(stc_list, cdsd_list):
    """Remove dominated solutions based on STC and CDSD."""
    solutions = [(stc, cdsd) for stc, cdsd in zip(stc_list, cdsd_list)]
    # Create dummy FloatSolution objects for get_non_dominated_solutions
    dummy_solutions = []
    for stc, cdsd in solutions:
        sol = FloatSolution([0], [1], 2, 0)  #Dummy solution with one variable
        sol.objectives = [stc, cdsd]
        dummy_solutions.append(sol)
    # Get non-dominated solutions
    non_dominated = get_non_dominated_solutions(dummy_solutions)
    # Extract STC and CDSD values
    return ([sol.objectives[0] for sol in non_dominated],
            [sol.objectives[1] for sol in non_dominated])

def run_optimization():
    # Define the problem
    problem = DOODPTW(num_orders=60, num_vehicles=15)

    # Common parameters for all algorithms
    population_size = 100
    max_evaluations = 1000
    crossover = SBXCrossover(probability=0.9, distribution_index=20.0)
    mutation = PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0)
    selection = BinaryTournamentSelection(comparator=DominanceComparator())
    termination_criterion = StoppingByEvaluations(max_evaluations=max_evaluations)

    # Run MODBO
    print("Running MODBO...")
    modbo_algorithm = MODBO(
        problem=problem,
        population_size=population_size,
        max_evaluations=max_evaluations,
        crossover=crossover,
        mutation=mutation,
        selection=selection
    )
    modbo_algorithm.run()
    modbo_solutions = modbo_algorithm.result()
    modbo_STC = [s.objectives[0] for s in modbo_solutions]
    modbo_CDSD = [s.objectives[1] for s in modbo_solutions]
    # Filter dominated solutions
    modbo_STC, modbo_CDSD = filter_dominated(modbo_STC, modbo_CDSD)

    # Run NSGA-II
    print("Running NSGA-II...")
    nsgaii_algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=mutation,
        crossover=crossover,
        selection=selection,
        termination_criterion=termination_criterion
    )
    nsgaii_algorithm.run()
    nsgaii_solutions = nsgaii_algorithm.result()
    nsgaii_STC = [s.objectives[0] for s in nsgaii_solutions]
    nsgaii_CDSD = [s.objectives[1] for s in nsgaii_solutions]
    # Filter dominated solutions
    nsgaii_STC, nsgaii_CDSD = filter_dominated(nsgaii_STC, nsgaii_CDSD)

    # Run SPEA2
    print("Running SPEA2...")
    spea2_algorithm = SPEA2(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=termination_criterion
    )
    spea2_algorithm.run()
    spea2_solutions = spea2_algorithm.result()
    spea2_STC = [s.objectives[0] for s in spea2_solutions]
    spea2_CDSD = [s.objectives[1] for s in spea2_solutions]
    # Filter dominated solutions
    spea2_STC, spea2_CDSD = filter_dominated(spea2_STC, spea2_CDSD)

    # Print values to terminal
    print("\nMODBO Results:")
    for stc, cdsd in zip(modbo_STC, modbo_CDSD):
        print(f"STC: {stc:.2f}, CDSD: {cdsd:.2f}")

    print("\nNSGA-II Results:")
    for stc, cdsd in zip(nsgaii_STC, nsgaii_CDSD):
        print(f"STC: {stc:.2f}, CDSD: {cdsd:.2f}")

    print("\nSPEA2 Results:")
    for stc, cdsd in zip(spea2_STC, spea2_CDSD):
        print(f"STC: {stc:.2f}, CDSD: {cdsd:.2f}")

    # Save results to Excel
    results = {
        "Algorithm": ["MODBO"] * len(modbo_STC) + ["NSGA-II"] * len(nsgaii_STC) + ["SPEA2"] * len(spea2_STC),
        "STC": modbo_STC + nsgaii_STC + spea2_STC,
        "CDSD": modbo_CDSD + nsgaii_CDSD + spea2_CDSD
    }
    df = pd.DataFrame(results)
    df.to_excel("optimization_results.xlsx", index=False)
    print("\nResults saved to 'optimization_results.xlsx'")

    # Plot the Pareto fronts with sorted points and connecting lines
    plt.figure(figsize=(10, 6))

    # Sort MODBO points by STC and connect with lines
    modbo_points = sorted(zip(modbo_STC, modbo_CDSD), key=lambda x: x[0])
    modbo_STC_sorted, modbo_CDSD_sorted = zip(*modbo_points) if modbo_points else ([], [])
    if modbo_points:
        plt.scatter(modbo_STC_sorted, modbo_CDSD_sorted, c='blue', label='MODBO', alpha=0.6, marker='o')
        plt.plot(modbo_STC_sorted, modbo_CDSD_sorted, c='blue', alpha=0.3)

    # Sort NSGA-II points by STC and connect with lines
    nsgaii_points = sorted(zip(nsgaii_STC, nsgaii_CDSD), key=lambda x: x[0])
    nsgaii_STC_sorted, nsgaii_CDSD_sorted = zip(*nsgaii_points) if nsgaii_points else ([], [])
    if nsgaii_points:
        plt.scatter(nsgaii_STC_sorted, nsgaii_CDSD_sorted, c='red', label='NSGA-II', alpha=0.6, marker='^')
        plt.plot(nsgaii_STC_sorted, nsgaii_CDSD_sorted, c='red', alpha=0.3)

    # Sort SPEA2 points by STC and connect with lines
    spea2_points = sorted(zip(spea2_STC, spea2_CDSD), key=lambda x: x[0])
    spea2_STC_sorted, spea2_CDSD_sorted = zip(*spea2_points) if spea2_points else ([], [])
    if spea2_points:
        plt.scatter(spea2_STC_sorted, spea2_CDSD_sorted, c='green', label='SPEA2', alpha=0.6, marker='s')
        plt.plot(spea2_STC_sorted, spea2_CDSD_sorted, c='green', alpha=0.3)

    plt.xlabel('System Transportation Cost (STC)')
    plt.ylabel('Customer Dissatisfaction Degree (CDSD)')
    plt.title(f'Pareto Front Comparison for DOODPTW\n')
    plt.legend()
    plt.grid(True)
    plt.savefig('pareto_front_comparison.png')
    print("Pareto front comparison plot saved as 'pareto_front_comparison.png'")

if __name__ == "__main__":
    run_optimization()