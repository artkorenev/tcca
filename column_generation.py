import numpy as np
import cvxpy as cvx
import data_reader
import espprc

# Data reading
n, td, xp, yp, xd, yd = data_reader.get_easy_data()

segm = data_reader.get_segments(xp, yp, xd, yd)
speed = 1.0


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2 ) ** 2 + (y1 - y2) ** 2)


def get_adj_graph(xp, yp, xd, yd, depot_x, depot_y):
    B = np.fromfunction(lambda i, j: np.sqrt((xd[i]-xp[j])**2+(yd[i]-yp[j])**2), (xp.size, xp.size), dtype=int)

    A = np.zeros((B.shape[0] + 2, B.shape[1] + 2))
    A[1:B.shape[0] + 1, 1:B.shape[1] + 1] = B

    for i in range(1, B.shape[0] + 1):
        A[0, i] = np.sqrt((xp[i - 1] - depot_x) ** 2 + (yp[i - 1] - depot_y) ** 2)
        A[i, -1] = np.sqrt((xd[i - 1] - depot_x) ** 2 + (yd[i - 1] - depot_y) ** 2)

    for i in range(A.shape[0]):
        A[i, i] = -1

    A[:, 0] = -1
    A[-1] = -1
    A[0, -1] = -1
    return A


def prepare_espprc_data(xp, yp, xd, yd, td, depot_x, depot_y):
    A = get_adj_graph(xp, yp, xd, yd, depot_x, depot_y)

    segm = data_reader.get_segments(xp, yp, xd, yd)
    route = np.zeros(2 + segm.shape[0])
    route[1:segm.shape[0] + 1] = segm

    tc = np.zeros(2 + td.shape[0])
    tc[0] = 0.0
    tc[-1] = 1000000000000.0  # infinity for this task
    tc[1:td.shape[0] + 1] = td

    return A, route, tc


def compute_the_penalty(route):
    curr_x = 0.0
    curr_y = 0.0
    current_time = 0.0

    lateness_penalty = 0.0
    distance_between = 0.0
    for ind in route:
        to_i = distance(xp[ind], yp[ind], curr_x, curr_y)
        distance_between += to_i

        current_time = np.maximum(td[ind], to_i / speed + current_time)
        lateness = current_time - td[ind]
        lateness_penalty += lateness ** 2

        curr_x = xd[ind]
        curr_y = yd[ind]

        current_time += segm[ind] / speed
    distance_between += distance(xd[route[-1]], yd[route[-1]], 0.0, 0.0)

    return lateness_penalty + np.sum(segm[route]) + distance_between


def binary_route(route):
    cost = compute_the_penalty(route)

    binary_route = np.zeros(n)
    binary_route[route] = 1.0

    return binary_route, cost

def find_initial_routes():
    '''Creates route matrix A, covering all clients for a given number of taxis'''

    route = []
    for i in range(n):
        route.append(i)

    return binary_route(route)

def prepare_initial_data():
    print 'Expected solution is: two routes: [0, 1, 2, 3] and [4, 5, 6, 7]'
    print 'or in binary solution is [1, 1, 1, 1, 0, 0, 0, 0] and [0, 0, 0, 0, 1, 1, 1, 1]'
    print 'Expected: optimal solution: 68.0886170382\n\n'

    routes = [
        [0, 1, 2],
        [3, 4, 7],
        [0, 6, 4, 2],
        [1, 3, 5, 7],
        [5, 6, 7]
    ]

    binary_routes = []
    cost_routes = []

    for i in range(len(routes)):
        bin_route, cost = binary_route(routes[i])
        binary_routes.append(bin_route)
        cost_routes.append(cost)

    patterns = np.zeros((n, len(routes)))

    costs = np.zeros(len(cost_routes))
    for i in range(len(cost_routes)):
        costs[i] = cost_routes[i]
        patterns[:, i] = binary_routes[i]

    return patterns, costs, routes


def solve_dual(patterns, costs, cars):
    y = cvx.Variable(n)
    z = cvx.Variable()

    obj = cvx.Maximize(cvx.sum_entries(y) + cars * z)

    constraints = [
        patterns.T * y + z <= costs,
        y >= 0,
        z <= 0
    ]

    prob = cvx.Problem(obj, constraints)
    res = prob.solve()

    # print 'Dual:', y.value
    # print 'Dual value:', res
    return res, y.value


def solve_primary(patterns, costs, cars):
    patt_n = patterns.shape[1]

    x = cvx.Variable(patt_n)

    obj = cvx.Minimize(costs * x)

    constraints = [
        patterns * x >= 1,
        np.ones(patt_n) * x <= cars,
        x >= 0
    ]

    prob = cvx.Problem(obj, constraints)
    res = prob.solve()

    # print 'Primary:', x.value
    # print 'Primary value:', res

    return res, x.value


def solve_cg():
    '''Finds a solution of the given problem using column generation
       Initial A - identity matrix of size CxC
       Uses a limit of iterations to get a solution in reasonable time
       C - number of clients, N - number of taxi cars'''
    patterns, costs, routes = prepare_initial_data()

    cars = 2
    iterations = 8

    x = None
    res_prim = np.inf

    for z in range(iterations):

        res_prim, x = solve_primary(patterns, costs, cars)

        print
        print 'Primary value: ', res_prim
        print 'Primary solution', x
        res_dual, y = solve_dual(patterns, costs, cars)

        # solving the ESPPRC problem
        A, route, tc = prepare_espprc_data(xp, yp, xd, yd, td, 0.0, 0.0)
        labels = espprc.espprc(0, A, route, tc, y)[-1]

        lmbd = np.zeros(n)
        for i in range(n):
            lmbd[i] = y[i]

        min_i = 0
        min_v = np.inf

        for i in range(len(labels)):
            if labels[i][0][-1] < min_v:
                is_a_copy = False
                for j in range(patterns.T.shape[0]):
                    if np.linalg.norm(patterns.T[j] - labels[i][0][3:-2]) == 0:
                        is_a_copy = True
                        break

                if not is_a_copy:
                    min_v = labels[i][0][-1]
                    min_i = i


        new_pattern = labels[min_i][0][3:-2]
        new_path = labels[min_i][1][1:-1]
        routes.append(new_path)

        new_bin_route, new_cost = binary_route(new_path)

        costs = np.append(costs, new_cost)
        patterns = np.c_[patterns, new_bin_route]

        print 'Added route:', new_path

    print '\nFinal solution: {} primary problem value'.format(res_prim)
    for i in range(x.shape[0]):
        if np.around(x[i], 2) != 0:
            print 'Path: {} with weight {}'.format(routes[i], float(np.around(x[i], 2)))


solve_cg()

