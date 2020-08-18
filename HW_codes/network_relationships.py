import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import bernoulli
def er_graph(N,p):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p = p):
                G.add_edge(node1, node2)
    return G
def plot_degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")

A1 = np.loadtxt("village_relationships.txt", delimiter = ",")
A2 = np.loadtxt("village_relationships2.txt", delimiter = ",")
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)
def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    degree_sequence = [d for n, d in G.degree()]
    print("Average degree: %.2f" % np.mean(degree_sequence))
def connected_component_subgraphs(G): 
    for c in nx.connected_components(G): 
        yield G.subgraph(c)
gen = nx.connected_components(G1)
g = gen.__next__()
G1_LCC = max(nx.connected_components(G1), key = len)
G2_LCC = max(nx.connected_components(G2), key = len)
# plt.figure()
# nx.draw(G1_LCC, node_color = "red", edge_color = "gray", node_size = "20")
# plt.show()
