import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from typing import List
import random
import math

class DOODPTW(FloatProblem):
    def __init__(self, num_orders: int = 25, num_vehicles: int = 10):
        super().__init__()
        self._num_orders = num_orders
        self._num_vehicles = num_vehicles
        self._number_of_variables = num_orders * num_vehicles  # x_ijl: assignment of orders to vehicles
        self._number_of_objectives = 2  # STC and CDSD
        self._number_of_constraints = 7  # All seven constraints

        # Problem parameters (adjusted for realistic STC)
        self.Q = 15  # Vehicle capacity
        self.V_0 = 5  # Increased vehicle speed to 5 km/h for shorter travel times
        self.T_max = 18  # Max time (18:00)
        self.working_time = [8, 18]  # [8:00, 18:00]

        # Generate positions in a 10km x 10km area 
        self.positions = [(random.uniform(0, 25), random.uniform(0, 25)) for _ in range(num_orders + 1)]  # +1 for depot
        self.depot = (0, 0)  # Depot position
        self.positions[0] = self.depot

        # Order demands and time windows
        self.demands = [1 for _ in range(num_orders)]  # All demands are 1
        self.time_windows = []
        for i in range(num_orders):
            early = random.uniform(8, 16)
            late = min(18, early + random.uniform(1, 2))
            self.time_windows.append([early, late])
        self.occurrence_times = [8 + i * 0.1 for i in range(num_orders)]

        # Vehicle positions (limited to reduce initial distance)
        self.vehicle_positions = [(0, 0)] + [(random.uniform(0, 5), random.uniform(0, 5)) for _ in range(num_vehicles - 1)]

        # Costs (reduced to keep STC under 2000)
        self.fixed_cost = 5  # Lowered fixed cost per vehicle
        self.driving_cost = 2  # Reduced cost per km

        # Variable bounds
        self.lower_bound = [0.0] * self._number_of_variables
        self.upper_bound = [1.0] * self._number_of_variables

    @property
    def number_of_variables(self) -> int:
        return self._number_of_variables

    @property
    def number_of_objectives(self) -> int:
        return self._number_of_objectives

    @property
    def number_of_constraints(self) -> int:
        return self._number_of_constraints

    @property
    def name(self) -> str:
        return "DOODPTW"

    def euclidean_distance(self, pos1: tuple, pos2: tuple) -> float:
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def satisfaction_function(self, service_time: float, a_j: float, b_j: float) -> float:
        """
        Compute satisfaction degree F(s_j^l) with corrected piecewise logic.
        """
        if service_time <= a_j - 1:
            return 0
        elif a_j - 1 < service_time < a_j:
            return (service_time - (a_j - 1)) / (a_j - (a_j - 1))
        elif a_j <= service_time <= b_j:
            return 1
        elif b_j < service_time < b_j + 1:
            return (b_j + 1 - service_time) / (b_j + 1 - b_j)
        else:  # service_time >= b_j + 1
            return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables).reshape((self._num_orders, self._num_vehicles))

        # Initialize service times
        s = np.zeros((self._num_orders + 1, self._num_vehicles))
        for l in range(self._num_vehicles):
            s[0][l] = 8  # Start at 8:00 from depot

        # Constraint 1: Each order to one vehicle
        constraint_1 = np.sum(x, axis=1)
        penalty_1 = np.sum(np.abs(constraint_1 - 1)) * 10  # Moderate penalty

        # Constraint 2: Loading capacity
        penalty_2 = 0
        for l in range(self._num_vehicles):
            total_demand = np.sum([self.demands[j] * x[j][l] for j in range(self._num_orders)])
            if total_demand > self.Q:
                penalty_2 += (total_demand - self.Q) * 10

        # Constraint 3: Vehicle allocation (no penalty, handled by x=0)
        penalty_3 = 0

        # Constraint 4: Route start/end (implicit in STC)
        penalty_4 = 0

        # Constraint 5: Positive service time
        penalty_5 = 0
        for j in range(self._num_orders):
            for l in range(self._num_vehicles):
                if x[j][l] > 0:
                    prev_node = 0 if j == 0 else j - 1  # Corrected sequence
                    dist = self.euclidean_distance(self.positions[prev_node + 1], self.positions[j + 1])
                    s[j + 1][l] = s[prev_node + 1][l] + (dist / self.V_0)
                    if s[j + 1][l] <= 0:
                        penalty_5 += -s[j + 1][l] * 10

        # Constraint 6: Service precedence
        penalty_6 = 0
        for j in range(1, self._num_orders):
            for l in range(self._num_vehicles):
                if x[j][l] > 0:
                    prev_node = j - 1
                    dist = self.euclidean_distance(self.positions[prev_node + 1], self.positions[j + 1])
                    expected_s = s[prev_node + 1][l] + (dist / self.V_0)
                    if s[j + 1][l] < expected_s:
                        penalty_6 += (expected_s - s[j + 1][l]) * 10

        # Constraint 7: Maximum time
        penalty_7 = 0
        for j in range(self._num_orders):
            for l in range(self._num_vehicles):
                if x[j][l] > 0 and s[j + 1][l] > self.T_max:
                    penalty_7 += (s[j + 1][l] - self.T_max) * 10

        # Objective 1: STC (Total Service Cost)
        STC = 0
        used_vehicles = set()
        for l in range(self._num_vehicles):
            if np.sum(x[:, l]) > 0:
                used_vehicles.add(l)
                STC += self.fixed_cost
                prev_pos = self.vehicle_positions[l]
                for j in range(self._num_orders):
                    if x[j][l] > 0:
                        dist = self.euclidean_distance(prev_pos, self.positions[j + 1])
                        STC += self.driving_cost * dist
                        prev_pos = self.positions[j + 1]
        STC = min(round(STC), 2000)  # Cap STC at 2000

        # Objective 2: CDSD
        satisfaction_sum = 0
        valid_orders = 0
        for j in range(self._num_orders):
            for l in range(self._num_vehicles):
                if x[j][l] > 0:
                    a_j, b_j = self.time_windows[j]
                    service_time = s[j + 1][l]
                    F_jl = self.satisfaction_function(service_time, a_j, b_j)
                    satisfaction_sum += F_jl
                    valid_orders += 1
                    break

        CDSD = 1.0 - (satisfaction_sum / self._num_orders) if self._num_orders > 0 else 0

        # Apply penalties
        total_penalty = penalty_1 + penalty_2 + penalty_5 + penalty_6 + penalty_7
        STC += total_penalty
        CDSD += total_penalty  # Scaled penalty for CDSD

        solution.objectives[0] = STC
        solution.objectives[1] = self.sigmoid_scale_to_1_5(CDSD)
        return solution

    def sigmoid_scale_to_1_5(self, value):
        normalized = 1 / (1 + math.exp(-value))
        scaled = 1 + normalized * 4
        return scaled

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives, self.number_of_constraints
        )
        new_solution.variables = [0.0] * self.number_of_variables
        for j in range(self._num_orders):
            l = random.randint(0, min(5, self._num_vehicles - 1))  # Limit to 5 vehicles initially
            idx = j * self._num_vehicles + l
            new_solution.variables[idx] = 1.0
        return new_solution