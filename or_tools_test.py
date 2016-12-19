"""
THIS FILE IS PURELY FOR TESTING THE PERFORMANCE OF THE OR TOOLS.
This is not going to work if you have no ortools installed.
We also test here just the amount of data that can be processed
by the library, so we didn't intend to receive a really
meaningful result to our data.

We shrunk the data by the 3 times (out of 9000), but were not
manage to successfully compute any result on this dataset in a
couple of hours.

We also seen that computing time increases in WAY-nonlinear
time, so the full dataset surely will not compute for a
reasonable time.

The problem for smaller datasets (up to 400 points) were actually
computable.

NOTE: the code is a slightly modified version of the Google's
demo of the OR tools, that solves the vehicle routing problem,
but we changed the data passing to it (at the bottom of the file).
The full version of the demo is here:
https://developers.google.com/optimization/routing/tsp/vehicle_routing
 """

import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def distance(x1, y1, x2, y2):
    # Manhattan distance
    dist = abs(x1 - x2) + abs(y1 - y2)

    return dist

# Distance callback

class CreateDistanceCallback(object):
  """Create callback to calculate distances between points."""

  def __init__(self, locations):
    """Initialize distance array."""
    size = len(locations)
    self.matrix = {}

    for from_node in xrange(size):
      self.matrix[from_node] = {}
      for to_node in xrange(size):
        if from_node == to_node:
          self.matrix[from_node][to_node] = 0
        else:
          x1 = locations[from_node][0]
          y1 = locations[from_node][1]
          x2 = locations[to_node][0]
          y2 = locations[to_node][1]
          self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]

# Demand callback
class CreateDemandCallback(object):
  """Create callback to get demands at each location."""

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]

def main():
  # Create the data.
  data = create_data_array()
  locations = data[0]
  demands = data[1]
  num_locations = len(locations)
  depot = 0
  num_vehicles = 100

  # Create routing model.
  if num_locations > 0:

    # The number of nodes of the VRP is num_locations.
    # Nodes are indexed from 0 to num_locations - 1. By default the start of
    # a route is node 0.
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # The 'PATH_CHEAPEST_ARC' method does the following:
    # Starting from a route "start" node, connect it to the node which produces the
    # cheapest route segment, then extend the route by iterating on the last
    # node added to the route.

    # Put a callback to the distance function here. The callback takes two
    # arguments (the from and to node indices) and returns the distance between
    # these nodes.

    dist_between_locations = CreateDistanceCallback(locations)
    dist_callback = dist_between_locations.Distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    # Put a callback to the demands.
    demands_at_locations = CreateDemandCallback(demands)
    demands_callback = demands_at_locations.Demand

    # Adding capacity dimension constraints.
    vehicle_capacity = 1000
    null_capacity_slack = 0
    fix_start_cumul_to_zero = True
    capacity = "Capacity"
    routing.AddDimension(demands_callback, null_capacity_slack, vehicle_capacity,
                         fix_start_cumul_to_zero, capacity)

    # Solve, displays a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
      # Display solution.
      # Solution cost.
      print "Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\n"

      for vehicle_nbr in range(num_vehicles):
        index = routing.Start(vehicle_nbr)
        index_next = assignment.Value(routing.NextVar(index))
        route = ''
        route_dist = 0
        route_demand = 0

        while not routing.IsEnd(index_next):
          node_index = routing.IndexToNode(index)
          node_index_next = routing.IndexToNode(index_next)
          route += str(node_index) + " -> "
          # Add the distance to the next node.
          route_dist += dist_callback(node_index, node_index_next)
          # Add demand.
          route_demand += demands[node_index_next]
          index = index_next
          index_next = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        node_index_next = routing.IndexToNode(index_next)
        route += str(node_index) + " -> " + str(node_index_next)
        route_dist += dist_callback(node_index, node_index_next)
        print "Route for vehicle " + str(vehicle_nbr) + ":\n\n" + route + "\n"
        print "Distance of route " + str(vehicle_nbr) + ": " + str(route_dist)
        print "Demand met by vehicle " + str(vehicle_nbr) + ": " + str(route_demand) + "\n"
    else:
      print 'No solution found.'
  else:
    print 'Specify an instance greater than 0.'

import random

def create_data_array():

  # # Works for a reasonable time
  # n = 200

  # Not.
  n = 3000
  locations = []
  demands = []

  for i in range(n):
    locations.append([random.randint(0, 1000), random.randint(0, 1000)])
    demands.append(random.randint(0, 3))

  data = [locations, demands]
  return data
if __name__ == '__main__':
  main()