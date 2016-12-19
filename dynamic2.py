# Dynamic programming solutions here

import numpy as np
import time

import data_reader

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2 ) ** 2 + (y1 - y2) ** 2)

# Data reading
n, td, xp, yp, xd, yd = data_reader.read_data(in_times=True)
segm = data_reader.get_segments(xp, yp, xd, yd)
speed = 40.0
driver_opening_cost = 40.0

# drivers
m = 200

d_pos_x = np.ones(m) * -1.0
d_pos_y = np.ones(m) * -1.0

# driver time
d_time = np.zeros(m)

# driver costs
d_cost = np.zeros(m)

drivers_on = np.zeros(m)

table = np.zeros((n, m))

for i in range(n):

    min_cost = np.inf
    min_dr = 0
    time_min_dr = 0

    for j in range(m):
        to_i = distance(d_pos_x[j], d_pos_y[j], xp[i], yp[i])

        time_on_i = np.maximum(td[i], to_i / speed + d_time[j])
        lateness = time_on_i - td[i]

        penalty = lateness ** 2 + to_i + segm[i]

        if drivers_on[j] == 0:
            penalty += driver_opening_cost

        if min_cost > penalty:
            min_cost = penalty
            min_dr = j
            time_min_dr = time_on_i

    d_cost[min_dr] += min_cost
    table[i, j] = min_cost
    drivers_on[min_dr] = 1.0
    d_pos_x[min_dr] = xd[i]
    d_pos_y[min_dr] = yd[i]
    d_time[min_dr] = time_min_dr + segm[i] / speed


print 'Cost {}'.format(np.sum(d_cost) + np.sum(np.sqrt(np.power(d_pos_x, 2) + np.power(d_pos_y, 2))) - np.sum(segm))