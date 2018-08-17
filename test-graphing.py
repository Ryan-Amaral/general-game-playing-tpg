import networkx as nx
import pickle
import matplotlib.pyplot as plt
from tpg.extensions import getRootTeamGraph
from tpg.extensions import getFullGraph

with open('saved-model-sgp.pkl', 'rb') as f:
    trainer = pickle.load(f)
    
#nodes, edges = getFullGraph(trainer)
nodes, edges = getRootTeamGraph(trainer.getBestAgent(tasks=['Assault-v0-18000']).team)
        
graph = nx.MultiDiGraph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

nodeColors = []
edgeColors = []
edgeWeights = []
labels = {}
for node in nodes:
    if node[0] == 'R':
        nodeColors.append('red')
        labels[node] = ''
    elif node[0] == 'T':
        nodeColors.append('lightcoral')
        labels[node] = ''
    else:
        nodeColors.append('yellow')
        labels[node] = node.split(':')[1]
        
for edge in edges:
    if edge[1][0] == 'T':
        edgeWeights.append(1.5)
    else:
        edgeWeights.append(1)
    edgeColors.append('grey')

pos = nx.spring_layout(graph, k=.5, iterations=20)
nx.draw(graph, pos, node_color=nodeColors, 
        edge_color=edgeColors, width=edgeWeights)
nx.draw_networkx_labels(graph, pos, labels)
plt.show()

