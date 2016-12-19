# Data preparations

import pandas as pd
import numpy as np
import time


def read_data(in_times=False, filename='data/04.12.2014.csv', verbose=False):
    df = pd.read_csv(filename)

    # sorting by time
    df = df.sort(['Hour', 'Minute'])

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

    if in_times:
        xp, yp, xd, yd = latlon_to_km(xp, yp, xd, yd)

    return n, td, xp, yp, xd, yd


def latlon_to_minute_data(xp, yp, xd, yd, av_speed=40, time=True, m_s=False, m=True):
    """
   :param dataA: pickups lat/lon, np.ndarray of N x 2
   :param dataB: drops lat/lon, np.ndarray of N x 2
   :return: minutes to travel from every A to every B (need A_i -> B_i and B_j -> A_i), np.ndarray N x N
   """
    r = xp.shape[0]
    # kms = latlon_to_metres_data(np.hstack((xp[:r].reshape(-1, 1), yp[:r].reshape(-1, 1))),
    #                             np.hstack((xd[:r].reshape(-1, 1), yd[:r].reshape(-1, 1))))
    dataA = np.hstack((xp[:r].reshape(-1, 1), yp[:r].reshape(-1, 1)))
    dataB = np.hstack((xd[:r].reshape(-1, 1), yd[:r].reshape(-1, 1)))
    N = dataB.shape[0]
    n = 100
    res = np.empty((N, N))
    R = 6378.137  # Radius of earth in KM var
    for i in range(np.round(N / n)):
        start = i * n
        end = (i + 1) * n + (i + 1 == np.round(N / n)) * (N % n - 1)
        # (B-A)_ijk = (dataB_i-dataA_j)_k, k = lat/lon
        dLatLonBA = dataB[start: end, np.newaxis, :] * np.pi / 180 - \
                    dataA[np.newaxis, :, :] * np.pi / 180
        dLatBA = dLatLonBA[:, :, 0]
        dLonBA = dLatLonBA[:, :, 1]
        a = np.sin(dLatBA / 2) * np.sin(dLatBA / 2) + \
            np.cos(dataA[np.newaxis, :, 0] * np.pi / 180) * \
            np.cos(dataB[start: end, 0, np.newaxis] * \
                   np.pi / 180) * np.sin(dLonBA / 2) * np.sin(dLonBA / 2)
        res[start:end] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * \
                         (av_speed / (1 * (1 - m_s) + m_s * (3600 / 1000.))) ** (-1) * (1 * (1 - m) + m * 60)
    return res  # minutes


def latlon_to_metres_data(dataA, dataB):
    """
   :param dataA: pickups lat/lon, np.ndarray of N x 2
   :param dataB: drops lat/lon, np.ndarray of N x 2
   :return: metres from every A to every B (need A_i -> B_i and B_j -> A_i), np.ndarray N x N
   """
    N = dataB.shape[0]
    n = 100
    res = np.empty((N, N))
    R = 6378.137  # Radius of earth in KM var
    for i in range(np.round(N / n)):
        start = i * n
        end = (i + 1) * n + (i + 1 == np.round(N / n)) * (N % n - 1)
        # (B-A)_ijk = (dataB_i-dataA_j)_k, k = lat/lon
        dLatLonBA = dataB[start: end, np.newaxis, :] * np.pi / 180 - \
                    dataA[np.newaxis, :, :] * np.pi / 180
        dLatBA = dLatLonBA[:, :, 0]
        dLonBA = dLatLonBA[:, :, 1]
        a = np.sin(dLatBA / 2) * np.sin(dLatBA / 2) + \
            np.cos(dataA[np.newaxis, :, 0] * np.pi / 180) * \
            np.cos(dataB[start: end, 0, np.newaxis] * \
                   np.pi / 180) * np.sin(dLonBA / 2) * np.sin(dLonBA / 2)
        res[start: end] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # print start, end
    return res  # meters


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


def get_segments(xp, yp, xd, yd):
    return np.sqrt(np.power(xp - xd, 2) + np.power(yp - yd, 2))


def latlon_to_km(xp, yp, xd, yd):
    x0 = xp[0]
    y0 = yp[0]

    return (xp - x0) * 110.574, \
           (yp - y0) * 111.320 * np.cos(xp - x0), \
           (xd - x0) * 110.574, \
           (yp - y0) * 111.320 * np.cos(xp - x0)


def get_adj_graph(xp, yp, xd, yd):
    A = np.fromfunction(lambda i,j: np.sqrt((xd[i]-xp[j])**2+(yd[i]-yp[j])**2), (xp.size, xp.size), dtype=int)
    for i in range(A.shape[0]):
        A[i, i] = -1
    return A


if __name__ == "__main__":
    n, td, xp, yp, xd, yd = read_data(filename='data/04.12.2014.csv')  # , verbose=True)

    print
    print 'Total clients: {}'.format(n)
    print 'Desired Times: {}, shape: {}'.format(td, td.shape)
    print 'Pickup X-coordinate: {}, shape: {}'.format(xp, xp.shape)
    print 'Pickup Y-coordinate: {}, shape: {}'.format(yp, yp.shape)
    print 'Destination X-coordinate: {}, shape: {}'.format(xd, xd.shape)
    print 'Distanation Y-coordinate: {}, shape: {}'.format(yd, yd.shape)


    #minAB = latlon_to_minute_data(xp, yp, xd, yd, av_speed=40, m_s=False, m=True)
    # min_xd_yd = latlon_to_minute_data(np.array([0]), np.array([0]), xd, yd, av_speed=40, m_s=False, m=True)

    #print np.mean(minAB)
