import networkx as nx
import matplotlib.pyplot as plt
% matplotlib inline


def graph_solution(greedy_solution, N_drivers, N_clients, depot_position_x, depot_position_y, xp, yp, xd, yd):
    plt.figure(figsize=(15, 15))

    G = nx.DiGraph()

    pos_nodes = {}
    pos_nodes["s"] = (depot_position_x, depot_position_y)
    pos_labels = {}
    labels = {}
    c_edges = []
    for i in range(N_clients):
        pos_nodes["{}p".format(i)] = (xp[i], yp[i])
        pos_nodes["{}d".format(i)] = (xd[i], yd[i])
        pos_labels["{}p".format(i)] = (xp[i] + 0.005, yp[i] + 0.005)
        pos_labels["{}d".format(i)] = (xd[i] + 0.005, yd[i] + 0.005)
        pos_labels["s"] = (depot_position_x + 0.005, depot_position_y + 0.005)
        labels["{}p".format(i)] = "{}p".format(i)
        labels["{}d".format(i)] = "{}d".format(i)
        labels["s"] = "s"

    G.add_nodes_from(pos_nodes, color="red")

    for i in range(N_clients):
        c_edges.append(("{}p".format(i), "{}d".format(i)))

    for i in range(N_drivers):
        driver_edges = []
        if greedy_solution.cur_driver_path[i]:
            for c in range(1, len(greedy_solution.cur_driver_path[i]) - 1):
                cur_c = greedy_solution.cur_driver_path[i][c]
                prev_c = greedy_solution.cur_driver_path[i][c - 1]
                driver_edges.append(("{}d".format(prev_c), "{}p".format(cur_c)))
            driver_edges.append(
                ("s", "{}p".format(greedy_solution.cur_driver_path[i][0])))  # Add edge from start to first pickup
            driver_edges.append(
                ("{}d".format(greedy_solution.cur_driver_path[i][-2]), "s"))  # Add edge from start to first pickup
        color = "{0:#0{1}x}".format(i * 20000 + 10000, 8)[2:8]
        color = "#" + color
        nx.draw_networkx_edges(G, pos_nodes, edgelist=driver_edges, edge_color=color, width=1.2)

    nx.draw_networkx_labels(G, pos_labels, labels, font_size=10)
    nx.draw_networkx_edges(G, pos_nodes, edgelist=c_edges, edge_color='black', width=0.5)

    nx.draw_networkx_nodes(G, pos=pos_nodes, node_size=50)
