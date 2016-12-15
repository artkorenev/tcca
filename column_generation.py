import numpy as np
import cvxpy as cvx

def compute_cost(route):
    pass

def get_new_route(y):
    pass

def find_initial_routes(C, N):
    '''Creates route matrix A, covering all clients for a given number of taxis'''
    if N >= C:
        return np.identity(C)
    else:
        A = np.zeros((C, N))
        mult = int(np.ceil(1.0*C/N))
        for j in range(N):
            for i in range(mult):
                row = mult*j + i
                if row > C-1:
                    return A
                else:
                    A[row, j] = 1
    return A

def solve_cg(C, N, maxiters = 100000, verbose = False):
    '''Finds a solution of the given problem using column generation
       Initial A - identity matrix of size CxC
       Uses a limit of iterations to get a solution in reasonable time
       C - number of clients, N - number of taxi cars'''
    num_routes = C
    A = find_initial_routes(C, N)
    costs = np.zeros(num_routes)
    for i in range(num_routes):
        costs[i] = compute_cost(A[:,i])

    #Forming constraints via dual problem:
    for i in range(maxiters):
        if verbose:
            print "Iteration: {}".format(i)
        y = cvx.Variable(C)
        z = cvx.Variable(1)

        obj = cvx.Maximize(cvx.sum_entries(y) + N*z)

        constraints = [
            (A.T)*y + z <= costs,
            y >= 0,
            z <= 0
        ]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        new_route = get_new_route(y.value)
        #TODO: Fix route dimensions
        if new_route in (A.T).tolist():
            break
        else:
            num_routes += 1
            costs = np.append(costs, compute_cost(new_route))
            A = np.concatenate((A, new_route), axis = 1)

    #Solving initial problem:
    x = cvx.Variable(num_routes)

    obj = cvx.Minimize(costs.T*x)

    constraints = [
        A*x >= 1,
        cvx.sum_entries(x) <= N,
        x >= 0
    ]

    prob = cvx.Problem(obj, constraints)
    prob.solve()

    return x.value