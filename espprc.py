"""
Here is an ESPPRC algorithm implementation.
The paper that it is based on:
http://onlinelibrary.wiley.com/doi/10.1002/net.20033/abstract

Unfortunately, due to extremely limitied amount of the time,
the performance of the algorithm is extremely poor.
Therefore, we could not handle any big dataset to solve the
problem using the algorithm, so we provided a really tiny
demo for the algorithm in column_generation.py file.

So the algorithm is basically an shortest-path algorithm that
also considers the resource constraints (in our case it is time).
We see, what paths 'dominate' other paths (i.e. some paths are
clearly more effective than others) and eliminate worse such
paths.
This algorithm can be viewed as a Bellman's algorithm for
finding the shortest-paths.

In our problem, we find a shortest-path from our depot to itself
but prohibiting travelling to itself directly (so we need to find
paths through the clients).
"""


import numpy as np
import copy

def empty_label(n):
    return [
        np.zeros(1 + 1 + n + 1),    # 1 is time constraint,
                                    # 1 is number of elements in path,
                                    # n is binary variables,
                                    # 1 is cost
        [-1]  # list to recover the route, values are added during the algorithm
    ]


# Extending the given label with adjacent vertex.
# Basically, we 'extend' the path to the given vertex.
def extend_label(label, fr, to, A, route, tc, lmbd):
    dist = A[fr, to]

    time_so_far = label[0][0]

    new_time = max(time_so_far + dist, tc[to])

    if new_time > tc[to] + 10.0:
        return None

    new_label = [np.copy(label[0])]
    new_label[0][0] = new_time + route[to]
    new_label[0][2 + to] = 1
    new_label[0][1] = label[0][1] + 1
    # recalculating the cost
    new_label[0][-1] += dist + route[to] - lmbd[to] \
                        + (new_time - tc[to]) ** 2

    new_label.append(copy.copy(label[1]))
    new_label[1].append(to - 1)  # Adding the given point to the route

    # Optimization step with euristic, not workable in our case, since our windows are not fixed
    # for i in range(A.shape[0]):
    #     if A[to, i] != -1:
    #         dist_to_i = A[to, i]
    #         new_time_i = new_time + dist_to_i
    #         if new_time_i > tc[i] + 10.0:
    #             new_label[0][2 + i] = 1
    #             new_label[0][1] += 1

    return new_label


# if left label is left dominant than the right
# (i.e. all values are bigger or equal than in the right label)
def left_dominant(labelleft, labelright):
    return np.all(labelleft[0] >= labelright[0])


# Given two sets of labels merging to the one without any
# dominating labels between each other
# (i know, that it is an awful piece of code in terms of
# performance)
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
                old_removed[j] = 1

    changed = np.any(old_removed == 1) or np.any(new_removed == 0)
    result = []
    for i in range(len(old_labels)):
        if not old_removed[i] == 1:
            result.append(old_labels[i])

    for i in range(len(new_labels)):
        if not new_removed[i] == 1:
            result.append(new_labels[i])

    return result, changed


# Launching the algorithm from the point p (p is an index)
# lmbd is a dual problem solution that we need to use during the
# solving this problem
def espprc(p, A, route, tc, lmbd=None):
    lmbd_extended = np.zeros(A.shape[0])

    if lmbd is not None:
        for i in range(1, lmbd.shape[0] + 1):
            lmbd_extended[i] = lmbd[i - 1]

    n = A.shape[0]

    labels = []
    for i in xrange(n):
        labels.append([])

    labels[p] = [empty_label(n)] # starting with the empty label

    queue = [p]

    while len(queue) > 0:
        v_i = queue.pop(0)  # the element to discover

        for v_j in range(A.shape[0]):  # iterating for all elements for v_i
            if A[v_i, v_j] != -1:  # if v_i and v_j are connected

                new_labels = []
                old_labels = labels[v_j]

                for label in labels[v_i]:
                    if label[0][2 + v_j] == 0:  # if for v_i v_j is not discovered in this label yet
                        new_label = extend_label(label, v_i, v_j, A, route, tc, lmbd_extended)
                        if new_label is not None:
                            new_labels.append(new_label)

                new_labels_removed, changed = remove_dominant_labels(old_labels, new_labels)
                labels[v_j] = new_labels_removed
                if changed:
                    queue.append(v_j)

    return labels