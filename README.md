# Multi-Objective Dung Beetle Optimization (MODBO) Algorithm for Order Dispatching

## Project Title & Description
This repository contains the implementation of the Multi-Objective Dung Beetle Optimization (MODBO) algorithm, a novel bio-inspired optimization technique applied to the Dynamic Order Dispatching Problem with Time Windows (DOODPTW). The project optimizes two objectives: Total Service Cost (STC) and Customer Dissatisfaction Degree (CDSD) for a fleet of vehicles dispatching orders. The code includes the MODBO algorithm, a problem definition (`MODPTW.py`), and a runner script (`run_optimization.py`) to generate and compare Pareto fronts with NSGA-II and SPEA2.

[<image-card alt="Build Status" src="https://img.shields.io/badge/build-passing-green" ></image-card>](https://github.com/alemrcan/Final_Year_Project)

## Installation
To set up the project environment, follow these steps:

- Ensure you have Python 3.8 or higher installed.
- Install required dependencies using the provided `requirements.txt`:
  ```bash
  pip install -r requirements.txt

## Usage
To run the optimization and generate results:

Navigate to the project directory containing run_optimization.py:
```bash
  cd Final_Year_Project/application
```
Execute the script with the following command:
```bash
  python run_optimization.py
```
The script will:
Run MODBO, NSGA-II, and SPEA2 algorithms.
Save results to optimization_results.xlsx.
Generate a Pareto front comparison plot as pareto_front_comparison.png.


## Results
<image-card alt="Pareto Front" src="Application/Outputs/pareto_front_comparison_M50_N15.png" ></image-card>
<image-card alt="Pareto Front" src="Application/Outputs/Picture1.jpg" ></image-card>
<image-card alt="Pareto Front" src="Application/Outputs/3a.png" ></image-card>
<image-card alt="Pareto Front" src="Application/Outputs/3b.png" ></image-card>


## Acknowledgments
- Special thanks to Dr. Harinandan Tunga and Dr. Samarjit Kar for guidance.
- Inspired by bio-mimetic optimization techniques and the JMetal framework.
- Gratitude to team members for collaborative efforts during the semester.

