import numpy as np
import copy

def empty_label(n):
    return [
        np.zeros(1 + 1 + n + 1),
        [-1]
    ]
    # 1 is time constraint,
    # 1 is number of elements in path,
    # n is binary variables,
    # 1 is cost


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
    new_label[0][-1] += dist + route[to] - lmbd[to] \
                        + (new_time - tc[to]) ** 2

    new_label.append(copy.copy(label[1]))
    new_label[1].append(to - 1)

    # Optimization step with euristic, not workable in our case, since our windows are not fixed
    # for i in range(A.shape[0]):
    #     if A[to, i] != -1:
    #         dist_to_i = A[to, i]
    #         new_time_i = new_time + dist_to_i
    #         if new_time_i > tc[i] + 10.0:
    #             new_label[0][2 + i] = 1
    #             new_label[0][1] += 1

    return new_label


def left_dominant(labelleft, labelright):
    return np.all(labelleft[0] >= labelright[0])


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


def espprc(p, A, route, tc, lmbd=None):
    lmbd_extended = np.zeros(A.shape[0])

    if lmbd is not None:
        for i in range(1, lmbd.shape[0] + 1):
            lmbd_extended[i] = lmbd[i - 1]

    n = A.shape[0]

    labels = []
    for i in xrange(n):
        labels.append([])

    labels[p] = [empty_label(n)]

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


if __name__ == "__main__":
    A = np.asarray([
        [-1, 1, -1, 0.5, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, -1, 1, -1],
        [-1, -1, -1, -1, 1],
        [-1, -1, -1, -1, -1]
    ])

    route = np.zeros(5)
    tc = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0])

    print espprc(0, A, route, tc)