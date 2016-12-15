# Greedy

import data_reader
import numpy as np


depot_position_x = 40.5
depot_position_y = -74.0

# Data reading
n, td, xp, yp, xd, yd = data_reader.read_data()

N_drivers = 100

C_assignment = 40000000
speed_limit = 0.001

# current drivers state
class Solution():
    def __init__(self):
        self.cur_driver_pos_x = []
        self.cur_driver_pos_y = []
        self.cur_driver_time_finish = []
        self.cur_driver_path = []
        self.cur_driver_cost = []
        self.cur_driver_assigned = []

        # initializing current drivers states
        for i in range(N_drivers):
            self.cur_driver_pos_x.append(depot_position_x)
            self.cur_driver_pos_y.append(depot_position_y)
            self.cur_driver_time_finish.append(0)
            self.cur_driver_path.append([])
            self.cur_driver_cost.append(0)
            self.cur_driver_assigned.append(0)


greedy_solution = Solution()


def assignment_cost(driver_ind):
    return C_assignment


def get_positions(client_ind, driver_ind):
    client_pos = np.zeros(2)
    driver_pos = np.zeros(2)
    driver_pos[0] = greedy_solution.cur_driver_pos_x[driver_ind]
    driver_pos[1] = greedy_solution.cur_driver_pos_y[driver_ind]
    client_pos[0] = xp[client_ind]
    client_pos[1] = yp[client_ind]
    return client_pos, driver_pos


def fuel_costs(client_ind, driver_ind, depot=False):
    # warning! if changed, please change at return_all_to_depot
    client_pos, driver_pos = get_positions(client_ind, driver_ind)
    return np.sqrt(np.dot(driver_pos, client_pos))


def waiting_costs_and_time_arrival(client_ind, driver_ind):
    client_pos, driver_pos = get_positions(client_ind, driver_ind)
    distance = np.sqrt(np.dot(driver_pos - client_pos, driver_pos - client_pos))
    time_to_travel = 1.0 * distance / speed_limit
    time_arrival = greedy_solution.cur_driver_time_finish[driver_ind] + time_to_travel
    waiting_costs = max(0, np.sign(time_arrival - td[client_ind])*(time_arrival - td[client_ind]) ** 2)
    return waiting_costs, time_arrival


def one_client_route_time(client_ind):
    client_start_pos = np.zeros(2)
    client_finish_pos = np.zeros(2)

    client_start_pos[0] = xp[client_ind]
    client_start_pos[1] = yp[client_ind]
    client_finish_pos[0] = xd[client_ind]
    client_finish_pos[1] = yd[client_ind]
    distance_to_travel = np.sqrt(np.dot(client_start_pos - client_finish_pos, client_start_pos - client_finish_pos))
    time_to_travel = 1.0 * distance_to_travel / speed_limit
    return time_to_travel


def calculate_costs_driver(client_ind, driver_ind):
    costs_driver = 0

    # calculating assignment costs
    assignment = 0
    if not greedy_solution.cur_driver_assigned[driver_ind]:
        assignment = assignment_cost(driver_ind)

    # calculating fuel costs
    fuel = fuel_costs(client_ind, driver_ind)

    # calculating waiting costs
    waiting = waiting_costs_and_time_arrival(client_ind, driver_ind)[0]
    costs_driver = assignment + fuel + waiting
    return costs_driver


def calculate_costs_total_plus_driver(client_ind, driver_ind):
    # calculating previous total costs
    total_costs = 0
    for i in range(N_drivers):
        total_costs += greedy_solution.cur_driver_cost[i]
    total_costs += calculate_costs_driver(client_ind, driver_ind)
    return total_costs


def assign_client(client_ind, driver_ind):
    costs = calculate_costs_driver(client_ind, driver_ind)
    time_arrival = waiting_costs_and_time_arrival(client_ind, driver_ind)[1]
    time_to_travel = one_client_route_time(client_ind)

    # assignment
    greedy_solution.cur_driver_pos_x[driver_ind] = xd[client_ind]
    greedy_solution.cur_driver_pos_y[driver_ind] = yd[client_ind]
    greedy_solution.cur_driver_time_finish[driver_ind] = max(time_arrival, td[client_ind]) + time_to_travel
    greedy_solution.cur_driver_path[driver_ind].append(client_ind)
    greedy_solution.cur_driver_cost[driver_ind] += costs
    greedy_solution.cur_driver_assigned[driver_ind] = 1
    pass


def total_costs():
    costs = 0
    for i in range(N_drivers):
        costs += greedy_solution.cur_driver_cost[i]
    return costs


def return_all_to_depot():
    for i in range(N_drivers):
        if greedy_solution.cur_driver_assigned[i]:
            cur_pos = np.zeros(2)
            cur_pos[0] = greedy_solution.cur_driver_pos_x[i]
            cur_pos[1] = greedy_solution.cur_driver_pos_y[i]

            depot_pos = np.asarray([depot_position_x, depot_position_y])
            distance_to_depot = np.sqrt(np.dot(cur_pos - depot_pos, cur_pos - depot_pos))
            time_to_depot = 1.0 * distance_to_depot / speed_limit

            greedy_solution.cur_driver_pos_x[i] = depot_position_x
            greedy_solution.cur_driver_pos_y[i] = depot_position_y
            greedy_solution.cur_driver_time_finish[i] += time_to_depot
            greedy_solution.cur_driver_path[i].append(-1)
            greedy_solution.cur_driver_cost[i] += distance_to_depot
    pass


for i in range(n):
    min_costs = np.inf
    min_costs_driver = -1
    for j in range(N_drivers):

        cur_costs = calculate_costs_total_plus_driver(i, j)
        if cur_costs < min_costs:
            min_costs_driver = j
            min_costs = cur_costs

    # assigning client to driver
    assert min_costs_driver != -1
    assign_client(i, min_costs_driver)

return_all_to_depot()

costs = total_costs()
print 'Costs = ', costs

assigned_drivers = 0
for i in range(N_drivers):
    if greedy_solution.cur_driver_assigned[i]:
        assigned_drivers += 1

print (assigned_drivers)