# Data preparations

import pandas as pd
import numpy as np
import time

def read_data(filename='data/04.12.2014.csv', verbose=False):
    df = pd.read_csv(filename)

    # sorting by time
    df = df.sort_values(['Hour', 'Minute'])

    if verbose:
        print df.head()

    # Computing times with ms
    times = df.as_matrix(columns=['Hour', 'Minute'])
    # times = 60 * 1000 * (60 * times[:, 0] + times[:, 1])
    times = (60 * times[:, 0] + times[:, 1])  # TODO: Change to MS (the line above)

    # Vectors of X and Y positions
    x_pos = df.as_matrix(columns=['Lat']).reshape(-1)
    y_pos = df.as_matrix(columns=['Lon']).reshape(-1)

    # number of clients
    n = times.shape[0] / 2

    # Desired times
    td = times[:n]

    # Pickup XY coordinates
    xp = x_pos[:n]
    yp = y_pos[:n]

    # Destination XY coordinates
    xd = x_pos[n:]
    yd = y_pos[n:]


    return n, td, xp, yp, xd, yd

def get_easy_data():
    # number of clients
    n = 8

    # Desired times
    td = np.asarray([1.0, 6.0, 11.0, 17.0, 4.0, 10.0, 20.0, 26.0])

    # Pickup XY coordinates
    xp = np.asarray([2.0, 7.0, 8.0, 4.0, 1.0, 5.0, 4.0, 1.0])
    yp = np.asarray([1.0, 1.0, 5.0, 5.0, 4.0, 8.0, 17.0, 11.0])

    # Destination XY coordinates
    xd = np.asarray([6.0, 9.0, 5.0, 1.0, 5.0, 5.0, 1.0, 1.0])
    yd = np.asarray([1.0, 5.0, 5.0, 1.0, 7.0, 17.0, 12.0, 6.0])

    return n, td, xp, yp, xd, yd


if __name__ == "__main__":
    n, td, xp, yp, xd, yd = read_data(filename='data/04.12.2014.csv', verbose=True)

    print
    print 'Total clients: {}'.format(n)
    print 'Desired Times: {}, shape: {}'.format(td, td.shape)
    print 'Pickup X-coordinate: {}, shape: {}'.format(xp, xp.shape)
    print 'Pickup Y-coordinate: {}, shape: {}'.format(yp, yp.shape)
    print 'Destination X-coordinate: {}, shape: {}'.format(xd, xd.shape)
    print 'Distanation Y-coordinate: {}, shape: {}'.format(yd, yd.shape)


