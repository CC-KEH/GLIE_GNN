import networkx as nx
import random
import numpy as np
import pandas as pd

def convert_to_weighted(input_filename, output_filename, weight_method='degree'):
    """Converts an undirected graph to a directed, weighted graph.

    Args:
        input_filename: Path to the .txt file containing the undirected graph.
        output_filename: Path to save the directed, weighted graph.
        weight_method: The method to assign weights to edges. Options are:
            - 'random': Assigns random weights between 0 and 1.
            - 'degree': Assigns weights based on node degrees.
            - 'custom': Allows you to provide a custom weight assignment function.
    """

    g = nx.read_edgelist(input_filename, nodetype=int)  # Load the undirected graph

    if weight_method == 'random':
        weights = {}
        for node in g.nodes():
            weights[node] = random.uniform(0, 1)
    elif weight_method == 'degree':
        degree = nx.degree(g)
        maxDegree = max(dict(degree).values())
        weights = {}
        for node in g.nodes():
            weights[node] = degree[node] / maxDegree
    elif weight_method == 'degree_noise':
        degree = nx.degree(g)
        mu = np.mean(list(dict(degree).values()))
        std = np.std(list(dict(degree).values()))
        weights = {}
        for node in g.nodes():
            episilon = np.random.normal(mu, std, 1)[0]
            weights[node] = 0.5 * degree[node] + episilon
            if weights[node] < 0.0:
                weights[node] = -weights[node]
        maxDegree = max(weights.values())
        for node in g.nodes():
            weights[node] = weights[node] / maxDegree

    # Add weights to edges
    for u, v in g.edges():
        g[u][v]['weight'] = weights[u] * weights[v]

    # Save the graph
    nx.write_weighted_edgelist(g, output_filename)


def convert_to_directed(output_file):
    G = pd.read_csv(output_file,header=None,sep=" ")
    G.columns = ["node1","node2","w"]
    del G["w"]
    # make undirected directed
    tmp = G.copy()
    G = pd.DataFrame(np.concatenate([G.values, tmp[["node2","node1"]].values]),columns=G.columns)

    G.columns = ["source","target"]

    outdegree = G.groupby("target").agg('count').reset_index()
    outdegree.columns = ["target","weight"]

    outdegree["weight"] = 1/outdegree["weight"]
    outdegree["weight"] = outdegree["weight"].apply(lambda x:float('%s' % float('%.6f' % x)))
    G = G.merge(outdegree, on="target")
    G.to_csv(output_file,sep=" ",header=None,index=False)

# def directed(path):
#     g = nx.read_edgelist(path, nodetype=int)  # Load the undirected graph
#     directed_g = g.to_directed()  # Convert to directed graph
#     nx.write_weighted_edgelist(directed_g, path)
    
# Example usage
input_file = "crime.txt"
output_file = "crime_processed.txt"
convert_to_weighted(input_file, output_file, weight_method='degree')  # Example using 'degree' weighting
convert_to_directed(output_file=output_file)