# Dynamic programming solutions here

import numpy as np
import time

import data_reader

# SIMPLE IMPLEMENTATION OF THE DYNAMIC PROGRAMMING:
# Assumptions:
# 1. Clients are sorted by the pickup desired time
# 2. Clients are picked up in the order of their pickup desired time

# Data reading
n, td, xp, yp, xd, yd = data_reader.read_data()

# drivers amount
m = 2
speed = 0.001
assignment_penalty = 40000

# driver's initial positions
dr_x = np.ones(m) * 40.5
dr_y = np.ones(m) * -74

table = np.zeros((m, n + 1))

# positions for a driver at each point of the solution
d_x_positions = np.zeros((m, n + 1))
d_y_positions = np.zeros((m, n + 1))
d_x_positions[:, 0] = dr_x
d_y_positions[:, 0] = dr_y

d_times = np.zeros((m, n + 1))
penalties = np.zeros(n)
assignments = np.zeros(n) - 1

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


start = time.time()

for i_c in range(1, n + 1):  # iterating by clients
    d_x_positions[:, i_c] = d_x_positions[:, i_c - 1]
    d_y_positions[:, i_c] = d_y_positions[:, i_c - 1]
    d_times[:, i_c] = d_times[:, i_c - 1]

    # desired time of current client
    desired_time = td[i_c - 1]

    # point of pickup of current client
    pickup_x = xp[i_c - 1]
    pickup_y = yp[i_c - 1]

    # min driver
    i_min = 0
    min_penalty = np.inf
    for i_d in range(m):  # iterating by drivers
        distance_to_pickup = distance(d_x_positions[i_d, i_c - 1], d_y_positions[i_d, i_c - 1], pickup_x, pickup_y)
        lateness = np.maximum(0.0, d_times[i_d, i_c] + speed * distance_to_pickup - desired_time)

        total_penalty = distance_to_pickup + lateness ** 2

        if min_penalty > total_penalty:
            min_penalty = total_penalty
            i_min = i_d

    # assignment index of current client
    assignments[i_c - 1] = i_min
    penalties[i_c - 1] += min_penalty
    # updating the position of the assigned driver to the destination point of the client
    d_x_positions[i_min, i_c] = xd[i_c - 1]
    d_y_positions[i_min, i_c] = yd[i_c - 1]

    # updating time of the driver after delivering the client
    # it is max from desired time or time of pick up + time that is spent on delivering
    d_times[i_min, i_c] = \
        np.maximum(td[i_c - 1],
                   d_times[i_min, i_c - 1] + speed * distance(d_x_positions[i_min, i_c - 1], d_y_positions[i_min, i_c - 1],
                                                      pickup_x, pickup_y)) + \
        speed * distance(pickup_x, pickup_y, xd[i_c - 1], yd[i_c - 1])

end = time.time()

print 'Processed: {} clients with {} drivers for {} seconds'.format(n, m, (end - start))
print "Assignments of the drivers for clients: {}".format(assignments)
# print 'Times of taxis: {}'.format(d_times)
print 'Total penalty: {}'.format(np.count_nonzero(assignments + 1) * assignment_penalty + np.sum(penalties))