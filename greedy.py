# Data preparations

import data_reader
import pandas as pd
import numpy as np
import copy
import random
import math
import time
from dateutil.parser import parse

# Data reading
n, td, xp, yp, xd, yd = data_reader.read_data(in_times=True)

depot_position_x = -1
depot_position_y = -1

N_drivers = 200
N_clients = n

C_assignment = 40

#speed in km/min
speed_limit = 1.0


# current drivers state
class Solution():
    def __init__(self):
        self.cur_driver_pos_x = []
        self.cur_driver_pos_y = []
        self.cur_driver_time_finish = []
        self.cur_driver_path = []
        self.cur_driver_cost = []
        self.cur_driver_assigned = []
        self.total_costs = 0

        # initializing current drivers states
        for i in range(N_drivers):
            self.cur_driver_pos_x.append(depot_position_x)
            self.cur_driver_pos_y.append(depot_position_y)
            self.cur_driver_time_finish.append(0)
            self.cur_driver_path.append([])
            self.cur_driver_cost.append(0)
            self.cur_driver_assigned.append(0)


def assignment_cost(driver_ind):
    return C_assignment


def get_positions(client_ind, driver_ind, solution):
    client_pos = np.zeros(2)
    driver_pos = np.zeros(2)
    driver_pos[0] = solution.cur_driver_pos_x[driver_ind]
    driver_pos[1] = solution.cur_driver_pos_y[driver_ind]
    client_pos[0] = xp[client_ind]
    client_pos[1] = yp[client_ind]
    return client_pos, driver_pos


def fuel_costs(client_ind, driver_ind, solution):
    # warning! if changed, please change at return_all_to_depot
    client_pos, driver_pos = get_positions(client_ind, driver_ind, solution)
    return np.sqrt(np.dot(driver_pos - client_pos, driver_pos - client_pos))


def waiting_costs_and_time_arrival(client_ind, driver_ind, solution):
    client_pos, driver_pos = get_positions(client_ind, driver_ind, solution)
    distance = np.sqrt(np.dot(driver_pos - client_pos, driver_pos - client_pos))
    time_to_travel = 1.0 * distance / speed_limit
    time_arrival = solution.cur_driver_time_finish[driver_ind] + time_to_travel
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


def calculate_costs_driver(client_ind, driver_ind, solution):
    costs_driver = 0

    # calculating assignment costs
    assignment = 0
    if not solution.cur_driver_assigned[driver_ind]:
        assignment = assignment_cost(driver_ind)

    # calculating fuel costs
    fuel = fuel_costs(client_ind, driver_ind, solution)

    # calculating waiting costs
    waiting = waiting_costs_and_time_arrival(client_ind, driver_ind, solution)[0]
    costs_driver = assignment + fuel + waiting
    return costs_driver


def calculate_costs_total_plus_driver(client_ind, driver_ind, solution):
    # calculating previous total costs
    total_costs = 0
    for i in range(N_drivers):
        total_costs += solution.cur_driver_cost[i]
    total_costs += calculate_costs_driver(client_ind, driver_ind, solution)
    return total_costs


def assign_client(client_ind, driver_ind, solution):
    costs = calculate_costs_driver(client_ind, driver_ind, solution)
    time_arrival = waiting_costs_and_time_arrival(client_ind, driver_ind, solution)[1]
    time_to_travel = one_client_route_time(client_ind)

    # assignment
    solution.cur_driver_pos_x[driver_ind] = xd[client_ind]
    solution.cur_driver_pos_y[driver_ind] = yd[client_ind]
    solution.cur_driver_time_finish[driver_ind] = max(time_arrival, td[client_ind]) + time_to_travel
    solution.cur_driver_path[driver_ind].append(client_ind)
    solution.cur_driver_cost[driver_ind] += costs
    solution.cur_driver_assigned[driver_ind] = 1
    pass


def total_costs(solution):
    costs = 0
    for i in range(N_drivers):
        costs += solution.cur_driver_cost[i]
    return costs


def return_all_to_depot(solution):
    for i in range(N_drivers):
        if solution.cur_driver_assigned[i]:
            cur_pos = np.zeros(2)
            cur_pos[0] = solution.cur_driver_pos_x[i]
            cur_pos[1] = solution.cur_driver_pos_y[i]

            depot_pos = np.asarray([depot_position_x, depot_position_y])
            distance_to_depot = np.sqrt(np.dot(cur_pos - depot_pos, cur_pos - depot_pos))
            time_to_depot = 1.0 * distance_to_depot / speed_limit

            solution.cur_driver_pos_x[i] = depot_position_x
            solution.cur_driver_pos_y[i] = depot_position_y
            solution.cur_driver_time_finish[i] += time_to_depot
            solution.cur_driver_path[i].append(-1)
            solution.cur_driver_cost[i] += distance_to_depot
    pass


def get_greedy():
    greedy_solution = Solution()

    for i in range(N_clients):
        #if int(100.0 * i / N_clients) % 100 == 0:
        #print(100.0 * i / N_clients)
        min_costs = np.inf
        min_costs_driver = -1
        for j in range(N_drivers):

            cur_costs = calculate_costs_total_plus_driver(i, j, greedy_solution)
            if cur_costs < min_costs:
                min_costs_driver = j
                min_costs = cur_costs

        # assigning client to driver
        assert min_costs_driver != -1
        assign_client(i, min_costs_driver, greedy_solution)

    return_all_to_depot(greedy_solution)

    costs = total_costs(greedy_solution)
    greedy_solution.total_costs = costs
    print 'Greedy'
    print 'Costs = ', costs

    assigned_drivers = 0
    for i in range(N_drivers):
        if greedy_solution.cur_driver_assigned[i]:
            assigned_drivers += 1

    print (assigned_drivers)
    return greedy_solution


#LOCAL OPT

def get_random():
    solution=Solution()
    clients = list(range(N_clients))
    random.shuffle(clients)
    for client_ind in clients:
        driver = np.random.randint(N_drivers)
        assign_client(client_ind, driver, solution)
    return_all_to_depot(solution)
    solution.total_costs = total_costs(solution)
    return solution


def get_neighbour(solution):
    new_solution = copy.deepcopy(solution)

    assert len(new_solution.cur_driver_path) == N_drivers
    from_driver = np.random.randint(0, N_drivers)
    while len(new_solution.cur_driver_path[from_driver]) == 0:
        from_driver = np.random.randint(0, N_drivers)
    to_driver = np.random.randint(0, N_drivers)


    #defeting returning to depot
    if from_driver != to_driver:
        del new_solution.cur_driver_path[from_driver][-1]
        if len(new_solution.cur_driver_path[to_driver]) != 0:
            del new_solution.cur_driver_path[to_driver][-1]
    else:
        del new_solution.cur_driver_path[from_driver][-1]

    client_to_move_ind_in_from_driver = np.random.randint(len(new_solution.cur_driver_path[from_driver]))
    client_to_move = new_solution.cur_driver_path[from_driver][client_to_move_ind_in_from_driver]
    del new_solution.cur_driver_path[from_driver][client_to_move_ind_in_from_driver]

    new_list=[]
    if len(new_solution.cur_driver_path[to_driver]) == 0:
        new_list=[client_to_move]
    else:
        insert_after_index_in_to_driver = np.random.randint(len(new_solution.cur_driver_path[to_driver]))
        new_list = []
        new_list.extend(new_solution.cur_driver_path[to_driver][:insert_after_index_in_to_driver])
        new_list.append(client_to_move)
        new_list.extend(new_solution.cur_driver_path[to_driver][insert_after_index_in_to_driver:])
    new_solution.cur_driver_path[to_driver] = new_list

    #removing all info about to_driver and from_driver
    new_solution.total_costs = 0
    new_solution.cur_driver_pos_x[to_driver] = depot_position_x
    new_solution.cur_driver_pos_y[to_driver] = depot_position_y
    new_solution.cur_driver_pos_x[from_driver] = depot_position_x
    new_solution.cur_driver_pos_y[from_driver] = depot_position_y
    new_solution.cur_driver_time_finish[to_driver] = 0
    new_solution.cur_driver_time_finish[from_driver] = 0
    new_solution.cur_driver_assigned[to_driver] = 0
    new_solution.cur_driver_assigned[from_driver] = 0
    new_solution.cur_driver_cost[to_driver] = 0
    new_solution.cur_driver_cost[from_driver] = 0
    path_to_driver = new_solution.cur_driver_path[to_driver]
    path_from_driver = new_solution.cur_driver_path[from_driver]
    new_solution.cur_driver_path[to_driver] = []
    new_solution.cur_driver_path[from_driver] = []

    for client_ind in path_to_driver:
        assign_client(client_ind, to_driver, new_solution)

    if to_driver != from_driver:
        for client_ind in path_from_driver:
            assign_client(client_ind, from_driver, new_solution)

    if to_driver != from_driver:
        drivers_array=[to_driver, from_driver]
    else:
        drivers_array = [to_driver]
    #returning to depot
    for i in drivers_array:
        if new_solution.cur_driver_assigned[i]:
            cur_pos = np.zeros(2)
            cur_pos[0] = new_solution.cur_driver_pos_x[i]
            cur_pos[1] = new_solution.cur_driver_pos_y[i]

            depot_pos = np.asarray([depot_position_x, depot_position_y])
            distance_to_depot = np.sqrt(np.dot(cur_pos - depot_pos, cur_pos - depot_pos))
            time_to_depot = 1.0 * distance_to_depot / speed_limit

            new_solution.cur_driver_pos_x[i] = depot_position_x
            new_solution.cur_driver_pos_y[i] = depot_position_y
            new_solution.cur_driver_time_finish[i] += time_to_depot
            new_solution.cur_driver_path[i].append(-1)
            new_solution.cur_driver_cost[i] += distance_to_depot

    new_solution.total_costs = total_costs(new_solution)

    return new_solution


def get_local(n_iter = 15000):
    cur_solution = get_random()
    NUM_ITER = n_iter
    min_cost = cur_solution.total_costs
    values = np.zeros(n_iter)
    for i in range(NUM_ITER):
        new_solution = get_neighbour(cur_solution)
        #print(new_solution.total_costs)
        if new_solution.total_costs < min_cost:
            #print('!!!!!!!!!!!!!!!!!!!!!!')
            #print(new_solution.total_costs)
            min_cost = new_solution.total_costs
            cur_solution = new_solution
        values[i] = new_solution.total_costs

    print ("Finished")
    print ('Costs = ', cur_solution.total_costs)

    assigned_drivers = 0
    for i in range(N_drivers):
        if cur_solution.cur_driver_assigned[i]:
            assigned_drivers += 1

    print (assigned_drivers)
    return cur_solution, values


def get_sa(n_iter=25000, low_temperature=0.1, step=0.1, iter_step=500):
    values = []
    cur_solution = get_greedy()
    print cur_solution.total_costs
    temperature = 1.0  # init temperature

    for i in range(n_iter):
        temp_solution = get_neighbour(cur_solution)
        cost_delta = (temp_solution.total_costs - cur_solution.total_costs) / cur_solution.total_costs

        if cost_delta <= 0 or random.random() < math.exp(-cost_delta / temperature):
            if temperature != 0.1 or cost_delta <= 0:
                cur_solution = temp_solution
                values.append(cur_solution.total_costs)

        if i % iter_step == 0 and temperature > low_temperature + 1e-10:
            temperature -= step
            print temperature
    return cur_solution, np.asarray(values)

get_greedy()
start_time = time.time()
sa, values = get_sa()
end_time = time.time()
print sa.total_costs
np.savetxt('sa_random.txt', values)
print 'Delta time: ',end_time-start_time