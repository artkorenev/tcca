import numpy as np

import data_reader

# Data reading
n, td, xp, yp, xd, yd = data_reader.read_data()

def get_adj_graph(xp, yp, xd, yd):
    A = np.fromfunction(lambda i,j: np.sqrt((xd[i]-xp[j])**2+(yd[i]-yp[j])**2), (xp.size, xp.size), dtype=int)
    for i in range(A.shape[0]):
        A[i, i] = -1
    return A

n = 500
A = get_adj_graph(xp[:n], yp[:n], xd[:n], yd[:n])
route = np.sqrt(np.power(xp - xd, 2) + np.power(yp - yd, 2))
route = route[:n]
tc = td[:n]

A[:, 0] = -1
A[-1] = -1
A[0, -1] = -1

# n = 4
#
# A = np.asarray([
#     [-1, 1.0, 1.0, -1],
#     [-1, -1, np.sqrt(5), 2.0 ],
#     [-1, np.sqrt(5), -1, 2.0 ],
#     [-1, -1, -1, -1 ]
# ])
#
# route = np.asarray(
#     [0.0, 1.0, 1.0, 0.0]
# )
#
# # Time constraint
# tc = np.array(
#     [0.0, 1.0, 2.0, 0.0]
# )


def empty_label():
    return np.zeros(1 + 1 + n + 1) # 1 is time constraint,
                                   # 1 is number of elements in path,
                                   # n is binary variables,
                                   # 1 is cost


def extend_label(label, fr, to):
    dist = A[fr, to]

    time_so_far = label[0]

    new_time = max(time_so_far + dist, tc[to]) + route[to]

    if new_time > tc[to] + 100.0:
        return None

    new_label = np.copy(label)
    new_label[0] = new_time
    new_label[2 + to] = 1
    new_label[1] = label[1] + 1
    new_label[-1] += dist + route[to]
    return new_label


def left_dominant(labelleft, labelright):
    return np.all(labelleft >= labelright)


def remove_dominant_labels(old_labels, new_labels):
    new_removed = np.zeros(len(new_labels))

    for i in range(len(new_labels)):
        for j in range(len(new_labels)):
            if i != j and left_dominant(new_labels[i], new_labels[j]):
                new_removed[i] = 1
                break

    for i in range(len(new_labels) - 1, -1):
        if new_removed[i] == 1:
            new_labels.pop(i)

    new_removed = np.zeros(len(new_labels))
    old_removed = np.zeros(len(old_labels))

    for i in range(len(new_labels)):
        for j in range(len(old_labels)):
            if left_dominant(new_labels[i], old_labels[j]):
                new_removed[i] = 1

            if left_dominant(old_labels[j], new_labels[i]):
                old_removed[i] = 1


    changed = np.any(old_removed == 1) or np.any(new_removed == 0)
    result = []
    for i in range(len(old_labels)):
        if not old_removed[i] == 1:
            result.append(old_labels[i])

    for i in range(len(new_labels)):
        if not new_removed[i] == 1:
            result.append(new_labels[i])

    return result, changed


def espprc(p):
    labels = []
    for i in xrange(n):
        labels.append([])

    labels[p] = [empty_label()]

    queue = [p]

    while len(queue) > 0:
        v_i = queue.pop(0)  # the element to discover

        for v_j in range(A.shape[0]):  # iterating for all elements for v_i
            if A[v_i, v_j] != -1:  # if v_i and v_j are connected

                new_labels = []
                old_labels = labels[v_j]

                for label in labels[v_i]:
                    if label[2 + v_j] == 0:  # if for v_i v_j is not discovered in this label yet
                        new_label = extend_label(label, v_i, v_j)
                        if new_label is not None:
                            new_labels.append(new_label)

                new_labels_removed, changed = remove_dominant_labels(old_labels, new_labels)
                labels[v_j] = new_labels_removed
                if changed:
                    queue.append(v_j)

    return labels

res = espprc(0)

# for i in range(len(res)):
#     print res[i], "\n"
