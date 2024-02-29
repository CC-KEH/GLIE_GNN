import random
import os
from tqdm import tqdm
import numpy as np 
import glob
import pandas as pd
import torch
import scipy.sparse as sp 
import torch.nn as nn
import os
import time

from diffuse import IC
from heapdict import heapdict

random.seed(1)


class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2*n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2*n_hidden_1, n_hidden_2)
        self.fc4 = nn.Linear(n_feat+n_hidden_1+n_hidden_2, 1)#+n_hidden_3
        
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)

        
    def forward(self, adj,x_in,idx):
        lst = list()

        # 1st message passing layer
        lst.append(x_in)
        
        x = self.relu(self.fc1( torch.cat( (x_in, torch.mm(adj, x_in)),1 ) ) )
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        # 2nd message passing layer
        x = self.relu(self.fc2( torch.cat( (x, torch.mm(adj, x)),1) ))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        # output layer
        x = torch.cat(lst, dim=1)
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1 , x.size(1)).to(x_in.device)
        x = out.scatter_add_(0, idx, x)
        
        #print(out.size())
        x = self.relu(self.fc4(x))

        return x

def gnn_eval(model,A,tmp,feature,idx,device):
    
    feature[tmp,:] = 1
    
    output = model(A,feature,idx).squeeze()
    return output.cpu().detach().numpy().item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


random.seed(1)
torch.manual_seed(1) 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    feat_d = 50
    dropout = 0.4
    hidden=64
    model = GNN_skip_small(feat_d,hidden,int(hidden/2),int(hidden/4),dropout).to(device)
    checkpoint = torch.load('models/model_g.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    seed_size = 100
    
    fw = open("celf++_glie_results.csv","a")
    fw.write("graph,nodes,infl20,time20,infl100,time100\n")

    
    for g in tqdm(["CR"]):
    
        print(g)
        if "l" in g:
            path = "data/sim_graphs/train/"+g+".txt"
        else:
            if g=="CR":
                path = "data/real_processsed/crime_processed.txt"    
    
        start = time.time()
        
        # Remove nodes based on degree
        G = pd.read_csv(path,header=None,sep=" ")
        nodes = set(G[0].unique()).union(set(G[1].unique()))
        adj = sp.coo_matrix((G[2], (G[1], G[0]) ), shape=(len(nodes), len(nodes)))
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        G.columns = ["source","target","weight"]
    
        outdegree = G.groupby("source").agg('target').count().reset_index()
        if g!="YT":
            deg_thres = np.histogram(outdegree.target,20)#,30) #np.histogram(outdegree.target)
            deg_thres  = deg_thres[1][1] #-deg_thres[1][0])/2  #deg_thres[1][1]
        else:
            deg_thres = np.histogram(outdegree.target,30)#,30) #np.histogram(outdegree.target)
            deg_thres  = deg_thres[1][1] #-deg_thres[1][0])/2  #deg_thres[1][1]
            
        nodes = outdegree.source[outdegree.target>deg_thres].values
        idx = torch.LongTensor(np.array([0]*adj.shape[0])).to(device)
        
        feature = torch.FloatTensor(np.zeros([adj.shape[0],feat_d])).to(device)
        
        #--- Celf++
        
        S = []
        Q = heapdict() 
        
        last_seed = None
        cur_best = None
        node_data_list = []
        # node_data = None 
        with torch.no_grad():
            for u in nodes:
                mg1 = gnn_eval(model,adj,[u],feature.clone(),idx,device)
                node_data = {
                  "node": u,
                  "mg1": mg1,
                  "prev_best": cur_best,
                  "mg2": gnn_eval(model,adj,[u,cur_best["node"]],feature.clone(),idx,device) if cur_best else mg1,
                  "flag": 0,
                  "list_index": len(node_data_list),
                }
                cur_best = cur_best if cur_best and cur_best["mg1"] > node_data["mg1"] else node_data
                # nodes[u]["node_data"] = node_data
                node_data_list.append(node_data)
                Q[len(node_data_list) - 1] = - node_data["mg1"]
                
        while len(S) < seed_size :
            node_idx, _ = Q.peekitem()
            node_data = node_data_list[node_idx]
            if node_data["flag"] == len(S):
              S.append(node_data["node"])
              del Q[node_idx]
              last_seed = node_data
              continue
          
            elif node_data["prev_best"] == last_seed:
              node_data["mg1"] = node_data["mg2"]
              
            else:
              before = gnn_eval(model,adj,S,feature.clone(),idx,device)
              S.append(node_data["node"])
              after = gnn_eval(model,adj,S,feature.clone(),idx,device)
              S.remove(node_data["node"])
              node_data["mg1"] = after - before
              node_data["prev_best"] = cur_best
              S.append(cur_best["node"])
              before = gnn_eval(model,adj,S,feature.clone(),idx,device)
              S.append(node_data["node"])
              after = gnn_eval(model,adj,S,feature.clone(),idx,device)
              S.remove(cur_best["node"])
              if node_data["node"] != cur_best["node"]: S.remove(node_data["node"])
              node_data["mg2"] = after - before
        
            if cur_best and cur_best["mg1"] < node_data["mg1"]:
              cur_best = node_data

            if len(S)==20:
                x20 = time.time()-start
            
            node_data["flag"] = len(S)
            Q[node_idx] = - node_data["mg1"]
            
            
        x100 = time.time()-start
        print("Done, now evaluating..") 
        
        x_ic100 = IC(G,S[:100])
        x_ic20 = IC(G,S[:20])
        
        
        fw.write(g.replace("\n","")+',"'+",".join([str(i) for i in S])+'",'+
                 str(x20)+","+str(x_ic20)+","+str(x100)+","+str(x_ic100)+"\n")     
        fw.flush()
    fw.close()


if __name__ == "__main__":
    main()