import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import networkx as nx
from graph_generator import graph_generator


directory = '/Users/arbas/Documents/Projects/Major/learn_im-main/learn_im-main/data/sim_graphs'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

generator = graph_generator()
generator.gen_new_graphs(300,500,30)
generator.save_graphs(directory)

os.chdir(directory)


# make them directed
for g in tqdm(glob.glob("*")):
    G = pd.read_csv(g,header=None,sep=" ")
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
    G.to_csv(g,sep=" ",header=None,index=False)