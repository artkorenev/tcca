"""
Greedy dynamic programming solution.

Here we provide an assumption that clients are delivered
at the same order as their pickup time.

Since it is clearly can have counter examples to be an
optimum solution, we understand that considering some
reorderings of clients will take an enormous time to
compute, because it would be nothing more than the brute
force search for the optimal solution. This would clearly
give us an optimal answer, but would take too much time
to compute.
"""

# Greedy dynamic programming solutions here

import numpy as np

import data_reader


# Simple distance function
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2 ) ** 2 + (y1 - y2) ** 2)


# Data reading
n, td, xp, yp, xd, yd = data_reader.read_data(in_km=True)
segm = data_reader.get_segments(xp, yp, xd, yd)
speed = 1.0  # km in minutes
driver_opening_cost = 40.0

# number of drivers
m = 200

# first and the last point of the drivers (taxi depot)
x_driver_init = -1.0
y_driver_init = -1.0

# so we initialize the first positions of the drivers
d_pos_x = np.ones(m) * x_driver_init
d_pos_y = np.ones(m) * y_driver_init

# driver time
d_time = np.zeros(m)

# driver costs
d_cost = np.zeros(m)

# binary variables if drivers are assigned or not
drivers_on = np.zeros(m)

table = np.zeros((n, m))

for i in range(n):
    # index of the best driver to assign the customer to
    min_cost = np.inf
    min_dr = 0
    time_min_dr = 0

    for j in range(m):
        # distance to the client
        to_i = distance(d_pos_x[j], d_pos_y[j], xp[i], yp[i])

        # time when we arrive to the client
        time_on_i = np.maximum(td[i], to_i / speed + d_time[j])
        # how we are late
        lateness = time_on_i - td[i]

        # computing the penalty (cost)
        penalty = lateness ** 2 + to_i + segm[i]

        # if driver is not 'opened' yet, we provide additional penalty (cost)
        if drivers_on[j] == 0:
            penalty += driver_opening_cost

        # updating the minimum
        if min_cost > penalty:
            min_cost = penalty
            min_dr = j
            time_min_dr = time_on_i
        table[i, j] = min_cost

    d_cost[min_dr] += min_cost
    drivers_on[min_dr] = 1.0
    d_pos_x[min_dr] = xd[i]
    d_pos_y[min_dr] = yd[i]
    d_time[min_dr] = time_min_dr + segm[i] / speed


print 'Final cost: {}'.format(np.sum(d_cost) + # cumulative costs of drivers
                       np.sum(np.sqrt((d_pos_x - x_driver_init) ** 2 + (d_pos_y - y_driver_init) ** 2)) - # cost of travellign back to the depot
                       np.sum(segm))  # sum of all orders (their are treated as a constant since we deliver all clients)
print 'Number of opened drivers: {}'.format(np.count_nonzero(drivers_on))