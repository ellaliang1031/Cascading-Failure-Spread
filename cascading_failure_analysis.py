# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:41:51 2019

@author: Ella
"""

import networkx as nx
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
G = nx.DiGraph()
G.add_nodes_from(range(28))
#def the position of nodes to draw a pretty graph
poslist = [(4,5),(4,7),(1,11),(5,11),(6,11),(1,13),(4,12),(6,12),(4,16),(5,14),
           (6,17),(9,10),(8,13),(10,13),(8,16),(9,16),(8,18),(11,18),(9,20),
           (15,11),(14,13),(17,13),(14,15),(17,15),(15,17),(18,15),(18,17),(13,19)]

edgelist = [(1,0),(2,1),(3,1),(4,1),(5,2),(6,2),(7,3),(8,5),(8,9),(9,6),(9,7),(6,1),
            (10,9),(11,10),(12,11),(13,11),(14,12),(15,13),(17,13),(16,15),(18,16),
            (18,17),(16,14),(17,27),(27,24),(24,22),(24,23),(22,20),(23,21),(20,19),
            (21,19),(25,21),(26,25)]
G.add_edges_from(edgelist)
#nx.draw(G,poslist,with_labels = True) 
#plt.xlim(0,20)
#plt.ylim(0,22)
#plt.show()


x = []
y = {11:[],17:[]}
for beta in np.arange(0,1.5,0.1):
    alpha = 0.4
    for failnode in [11,17]:
        #set the state
        state_init={}
        for node in G.nodes():
            state_init[node]='S'
        state_init[failnode]='F'
        nx.set_node_attributes(G, state_init, 'state')
        #node_weight set, could use random as well
        weightlist = [1,4,4,3.5,3,2,1.5,2,3,0.5,2.5,1,2,2,3,1,0.5,3.5,2,2,1,2,3,2,2,1,1,3.5]
        node_weight = {}
        for i in range(28):
            node_weight[i] = weightlist[i]
        nx.set_node_attributes(G, node_weight, 'weight')
        
        #edge weight
        edge_weight = {}
        for edge in G.edges():
            edge_weight[edge] = (G.node[edge[0]]['weight'] + G.node[edge[1]]['weight'])/2
        nx.set_edge_attributes(G, edge_weight, 'weight')
        
        #initial load
        initial_load = {}
        for node in G.nodes():
            initial_load[node] = G.degree(node, weight = 'weight')**alpha
        nx.set_node_attributes(G, initial_load, 'initial')
           
        #initial capacity
        node_capacity = {}
        for node in G.nodes():
            node_capacity[node] = (1+beta)*G.node[node]['initial']
        nx.set_node_attributes(G, node_capacity, 'capacity')  
        #compute the node with highest degree, closeness and betweenness in project2
        #sorted(nx.degree_centrality(G).items(),key=lambda item:item[1]) #17
        #sorted(nx.betweenness_centrality(G,weight = 'weight').items(),key=lambda item:item[1]) #11
        
        #algorithm
        A = []
        A.append(failnode)
        B = []
        t = 0
        NS = 0
        while len(A) != 0:
            for node in A:
                t = t + 1
                for neb in nx.neighbors(G, node):
                    if (G.node[node]['initial'] * (G.edges[node, neb]['weight']
                        /G.out_degree(node, weight='weight')) +  G.node[neb]['initial']
                        > G.node[neb]['capacity']):
                        if G.node[neb]['state'] == 'S':
                            G.node[neb]['state'] = 'F'
                            A.append(neb)
                if t == 1:
                    NS = len(A)
            B.append(node)
            A.remove(node)
        NC = len(B)
        CF = NS/(NC * 27)
        y[failnode].append(CF)
    x.append(beta)
    nx.draw(G,poslist,node_color='g',node_size=400,with_labels=True,alpha=0.6)
    nx.draw_networkx_nodes(G,poslist,nodelist=B,node_size=400,node_color='r')
    plt.xlim(0,20)
    plt.ylim(0,20)
    #plt.show()   

plt.plot(x, y[17])
plt.plot(x, y[11])
plt.xlim(0, 1.5) 
plt.ylim(0, 0.05)
plt.plot(x, y[17], marker='*', mec='r', mfc='w',label = 'alpha = 1.5, node17')
plt.plot(x, y[11], marker='o', ms=10, label = 'alpha = 1.5,node11') 
plt.xlabel('Beta')
plt.ylabel("CF")
plt.legend() 
plt.show()