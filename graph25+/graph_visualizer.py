import networkx as nx
import matplotlib.pyplot as plt 
import math

GRAPH_FILE = 'gen_graph.csv'

G=nx.DiGraph()

# The graph file is read
file_data = None
with open(GRAPH_FILE) as graph:
	file_data = graph.read()
file_data = file_data.split('\n')

labels_data = {} # For each node a custom label is made (indexed with the node name)

# Used to count how many nodes have a CPU or GPU only implementations or both
CPU_only_c = 0
GPU_only_c = 0
DOUBLE_c = 0

# All edges are added
for line in file_data:
	values = line.split(',')
	node_name = values[0]
	if len(values) > 5:
		for i in range(5, len(values), 2):
			G.add_edge(values[i], node_name)
			
			# The custom text consists in node name + cpu time (if present) + gpu time (if present)
			text = node_name + '\n'
			CPU = False
			GPU = False
			if float(values[2]) != -1:
				text += 'CPU: '+str(float(values[2]))+'\n'
				CPU = True
			if float(values[3]) != -1:
				text += 'GPU: '+str(float(values[3]))+'\n'
				GPU = True
			labels_data[node_name] = text
			
			# Counts the number of CPU and GPU implementations
			if CPU and not GPU:
				CPU_only_c += 1
			elif not CPU and GPU:
				GPU_only_c += 1
			elif CPU and GPU:
				DOUBLE_c += 1

# The graph is shown
pos = nx.shell_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1000,alpha=0.9,node_shape='s')
nx.draw_networkx_labels(G, pos, labels=labels_data,font_color='black',font_size=7)
nx.draw_networkx_edges(G, pos, node_size=1000)
plt.show()

print('CPU:\t', CPU_only_c)
print('GPU:\t', GPU_only_c)
print('DOUBLE:\t', DOUBLE_c)