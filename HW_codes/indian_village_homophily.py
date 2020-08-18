from collections import Counter
import numpy as np
import networkx as nx

def marginal_prob(chars):
    frequencies = dict(Counter(chars.values()))
    sum_frequencies = sum(frequencies.values())
    return {char: freq / sum_frequencies for char, freq in frequencies.items()}
                
def chance_homophily(chars):
    marginal_probs = marginal_prob(chars)
    return np.sum(np.square(list(marginal_probs.values())))

favorite_colors = {
    "ankit":  "red",
    "xiaoyu": "blue",
    "mary":   "blue"
}

color_homophily = chance_homophily(favorite_colors)

import pandas as pd

df  = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@individual_characteristics.csv", low_memory=False, index_col=0)
df1 = df.loc[df['village'] == 1]
df2 = df.loc[df['village'] == 2]


sex1 = pd.Series(df1.resp_gend.values, index=df1.pid).to_dict()
caste1 = pd.Series(df1.caste.values, index=df1.pid).to_dict()
religion1 = pd.Series(df1.religion.values, index=df1.pid).to_dict()
sex2 = pd.Series(df2.resp_gend.values, index=df2.pid).to_dict()
caste2 = pd.Series(df2.caste.values, index=df2.pid).to_dict()
religion2 = pd.Series(df2.religion.values, index=df2.pid).to_dict()

def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties = 0
    num_ties = 0
    for n1, n2 in G.edges():
        if IDs[n1] in chars and IDs[n2] in chars:
            if G.has_edge(n1, n2):
                num_ties += 1
                if chars[IDs[n1]] == chars[IDs[n2]]:
                    num_same_ties += 1
    return (num_same_ties / num_ties)

data_filepath1 = "https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@key_vilno_1.csv"
data_filepath2 = "https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@key_vilno_2.csv"
pid1 = pd.read_csv(data_filepath1, index_col= 0)
pid2 = pd.read_csv(data_filepath2, index_col= 1)


A1 = np.array(pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@adj_allVillageRelationships_vilno1.csv", index_col=0))
A2 = np.array(pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@adj_allVillageRelationships_vilno2.csv", index_col=0))
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

pid1 = pd.read_csv(data_filepath1, dtype=int)['0'].to_dict()
pid2 = pd.read_csv(data_filepath2, dtype=int)['0'].to_dict()

# print(homophily(G1, sex1, pid1))
# print(homophily(G1, caste1, pid1))
# print(homophily(G1, religion1, pid1))
# print(homophily(G2, sex2, pid2))
# print(homophily(G2, caste2, pid2))
# print(homophily(G2, religion2, pid2))

# print(chance_homophily(sex1))
# print(chance_homophily(caste1))
# print(chance_homophily(religion1))
# print(chance_homophily(sex2))
# print(chance_homophily(caste2))
# print(chance_homophily(religion2))


