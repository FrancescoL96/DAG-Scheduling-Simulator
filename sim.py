from decimal import Decimal, ROUND_HALF_EVEN
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

n_cpu = 2
n_gpu = 1
PROCESSORS = float(n_cpu+n_gpu)	# Used in the formula to calculate priority points for G-FL
DEADLINE = False # If set to true it will schedule using EDD, otherwise G-FL

"""
There is a class Node, which rappresents a single computing node, such as Fast, or Scale, it has various fields, which are used to create the dependencies' graph
and to verify the scheduling
The scheduling is created, verified and shown by the class Schedule, which taken a list of nodes, once it has created the scheduling, verifies that all temporal restrictions are respected
There are a couple functions (EDD_auto, EDD_auto_rec, create_dependencies_graph_rec, calculate_priority_points) which are used to compute some additional data on the nodes, specifically, EDD deadlines and dependencies in the reverse order. The last function calculates priority points for G-FL
"""
class Node:
	def __init__(self, name, level, time, requirements, execution, time_gpu = -1):
		self.name = name					# Name of the node that needs to scheduled
		self.level = level					# Level of the node (0 to 7)
		self.time = time					# Runtime
		self.requirements = requirements	# List of nodes which this node depends on
		self.time_gpu = time_gpu			# If it has a GPU implementation set the time here
		self.execution = execution			# This is a numeric value used to indicate to which graph this nodes belong to
		self.copy_time = 0					# Used to account for copy time between CPU/GPU ----- It is in the code, it is used, but it is never set
		self.is_gpu = False					# If this value is True then the GPU time is used
		# This values are set automatically
		self.delay = 0						# How long does this node need to wait before it can start
		self.scheduled_time = -1			# At what time is this node scheduled
		self.required_by = []				# Which nodes require this be completed (sed to calculate deadlines)
		self.deadline = 1.0					# EDD deadline, calculated using recursive formula: EDD(n) = [for each successor s] - min(EDD(s) - WCET - Cpy)
		self.priority_point = 0.0			# Proprity value for G-FL scheduling, calculated for each node using: Y = Deadline - ((Procs - 1)/Procs) * WCET - min [for each job j] (Deadline_j - ((Procs - 1)/Procs) * WCET_j)
		# Currently using the relative priority point formula: Y = Deadline - ((Procs - 1)/Procs) * WCET
	
	# If this function is called, is_gpu is set to True, allowing the code to use time_gpu for execution time intead of time (which should be used for cpu time)
	def set_gpu(self):
		if (self.time_gpu != -1):
			self.is_gpu = True
	
	# If a node cannot start right away and dependencies do not allow the node to start right away, set this value to make the node wait before it starts
	def set_delay(self, delay):
		if (delay >= 0):
			self.delay = delay
	
	# Overrides the string function, printing the node name
	# Or other stuff actually, just print what you need :)
	def __str__(self):
		"""
		cpu_time = '\nfinishes at (cpu): ' + str(self.scheduled_time + self.time)
		gpu_time = '\nfinishes at (gpu): ' + str(self.scheduled_time + self.time_gpu)
		concat = gpu_time if (self.is_gpu == True) else cpu_time
		return 'Node name: '+self.name+' - is scheduled at time: '+str(self.scheduled_time) + concat + '\nDelay: ' + str(self.delay) +'\n---------------------------'
		"""
		return node.name
	
# This class creates and verifies the scheduling, prints the scheduling only if it is verified
class Schedule:
	# Takes as parameter a list of all nodes ordered by priority
	def __init__(self, node_list, n_cpu, n_gpu):
		self.node_list = node_list
		self.verified = False
		self.max_time = 0
		self.n_cpu = n_cpu
		self.n_gpu = n_gpu
	
	# Creates the scheduling
	def create_schedule(self):
		# For each CPU and GPU creates a list of nodes
		execution_elements_cpu = [[] for i in range(0, self.n_cpu)]
		execution_elements_gpu = [[] for i in range(0, self.n_gpu)]
		# Creates the same lists to save times (at what time is the computation) for each processor
		times_cpu = [0 for i in range(0, self.n_cpu)]
		times_gpu = [0 for i in range(0, self.n_gpu)]
		available = [] 
		available.append(self.node_list[0])
		# Run while there are nodes that can run
		while available:
			node = available[0] # Takes the highest priority node that can execute
			self.set_minimum_start_time(node) # If this node depends on other nodes that have not completed yet we might need to wait
			if (node.is_gpu):
				# Adds this node to the GPU with the smallest current time
				execution_elements_gpu[times_gpu.index(min(times_gpu))].append(node)
				# If the current time is less than the minimum time for this specific node, a delay is added to make sure all dependencies are met
				node.delay = node.scheduled_time - times_gpu[times_gpu.index(min(times_gpu))] if node.scheduled_time - times_gpu[times_gpu.index(min(times_gpu))] > 0 else 0
				# The node's scheduled time is updated
				node.scheduled_time = times_gpu[times_gpu.index(min(times_gpu))] + node.delay
				# The current time for this processor is updated
				times_gpu[times_gpu.index(min(times_gpu))] += node.time_gpu + node.delay + node.copy_time
			else:
				# Behaves like above, but for CPU cores
				execution_elements_cpu[times_cpu.index(min(times_cpu))].append(node)
				node.delay = node.scheduled_time - times_cpu[times_cpu.index(min(times_cpu))] if node.scheduled_time - times_cpu[times_cpu.index(min(times_cpu))] > 0 else 0
				node.scheduled_time = times_cpu[times_cpu.index(min(times_cpu))] + node.delay
				times_cpu[times_cpu.index(min(times_cpu))] += node.time + node.delay + node.copy_time
			for next in node.required_by:			
				if (next.scheduled_time == -1 and next not in available):
					available.append(next)
			available.remove(node)
			if (DEADLINE):
				available.sort(key=lambda x: x.deadline, reverse=False)
			else:
				available.sort(key=lambda x: x.priority_point, reverse=False)

		# Trasforms the separeted lists in a sigle one: [CPU_0 list[node_a, node_b], CPU_1 list[node_c, node_d], ..., GPU_0 list[node_e, node_f], ...]
		self.node_list = execution_elements_cpu + execution_elements_gpu
		times = times_cpu + times_gpu

		self.verified = False # If there is no violation of dependencies in the scheduling this variable is set to True
		# Verifies the scheduling
		for computing_element in self.node_list:
			for node in computing_element:
				node_scheduled_time = Decimal(node.scheduled_time).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)
				# Verify that all node requirements are scheduled and completed
				for required in node.requirements:
					# If a required node is not scheduled, it is not a valid scheduling
					if (required.scheduled_time == -1):
						self.error_function(node, required, node_scheduled_time, self.node_list.index(computing_element))
						
					# If a required node is scheduled, but it has not yet completed (if it's running on a different CE), then it is not a valid scheduling
					# If the node is run on the GPU we need to use the GPU time, otherwise we use the normal cpu time
					if (required.is_gpu == True):
						required_gpu_time = Decimal(required.scheduled_time + required.time_gpu).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)
						if (node_scheduled_time.compare(required_gpu_time) == -1):
							self.error_function(node, required, node_scheduled_time, self.node_list.index(computing_element))
					else:
						required_cpu_time = Decimal(required.scheduled_time + required.time).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)
						if (node_scheduled_time.compare(required_cpu_time) == -1):
							self.error_function(node, required, node_scheduled_time, self.node_list.index(computing_element))
		self.verified = True
		self.max_time = round(max(times), 2)
		return self.max_time
		
	""" 
	error_from: which node has a dependency not met
	caused_by: which node is the dependency not met
	current_time: at what time was "error_from" scheduled
	execution_unit: on which execution unit was the error generated
	"""
	def error_function(self, error_from, caused_by, current_time, execution_unit):
		print('Current time: '+str(current_time))
		print('Executing on unit: '+str(execution_unit))
		print('Error from: \n'+str(error_from))
		print('Caused by: ')
		print(caused_by)
		exit()
	
	def set_minimum_start_time(self, node):
		# the minimum time possible for this node is the maximum time (scheduled + runtime) obtained from all its dependencies, if this node has no requirements then it can start at time 0
		node.scheduled_time = max((req.scheduled_time + req.copy_time + (req.time_gpu if req.is_gpu else req.time) for req in node.requirements)) if node.requirements else 0
			
	# This function prints the Grantt graph of the scheduled nodes
	# Supports any number of processors (more or less, zooming in and out might be required)
	def create_bar_graph(self, labels):
		# If the scheduling is not verified it cannot be shown
		if (not self.verified):
			return

		fig, gnt = plt.subplots(figsize=(16,9), num='Scheduling')
		fig.set_tight_layout(0.1)
		
		# Legend is created here color vectors are created below with the same rules
		scale_legend = mpatches.Patch(color='steelblue', label='scale')
		orb_legend = mpatches.Patch(color='limegreen', label='orb')
		gauss_legend = mpatches.Patch(color='black', label='gauss')
		grid_legend = mpatches.Patch(color='gold', label='grid')
		fast_legend = mpatches.Patch(color='orange', label='fast')
		dl_legend = mpatches.Patch(color='gray', label='dl')
		name_legend = mpatches.Patch(color='white', label='initial_level_execution')
		plt.legend(handles=[scale_legend, orb_legend, gauss_legend, grid_legend, fast_legend, dl_legend, name_legend])
		
		gnt.set_ylim(0, 50) 
		gnt.set_xlim(0, self.max_time*1.05) 
		gnt.set_xlabel('seconds since start') 
		gnt.set_ylabel('Processor')

		gnt.set_yticks([15+i*10 for i in range(0, len(labels))])
		gnt.set_yticklabels(labels) 
		
		gnt.grid(True) 
		bars = []
		colors = []
		names = []
		# Basically what is done is that for each computing element a broken bar is created (bar), for each segment is picked a color based (color) on the node name, which is written on top of the segment as well (name)
		# (bars, colors and names contain all a number of lists based on the number of computing elements)
		for computing_element in self.node_list:
			bar = []
			name = []
			color = []
			for node in computing_element:
				name.append(node.name[0]+node.name[-2:]+'_'+str(node.execution))
				if (node.is_gpu): 
					bar.append((node.scheduled_time, node.time_gpu))
				else:
					bar.append((node.scheduled_time, node.time))
					
				# Creating the actual color vectors
				if ('scale' in node.name):
					color.append('steelblue')
				elif ('orb' in node.name):
					color.append('limegreen')
				elif ('gauss' in node.name):
					color.append('black')
				elif ('grid' in node.name):
					color.append('gold')
				elif ('fast' in node.name):
					color.append('orange')
				else:
					color.append('gray')
			bars.append(bar)
			names.append(name)
			colors.append(color)
		# Creates the broken bars, using the bars and color vectors
		for i in range(0, len(bars)):				
			gnt.broken_barh(bars[i], (10*len(bars)-i*10, 6), facecolors=tuple(colors[i]), edgecolor='white')
		# Writes the names of the nodes separetly
		for i in range(0, len(self.node_list)):
			for j in range(0, len(self.node_list[i])):
					gnt.text(x=bars[i][j][0]+bars[i][j][1]/2, y=7+10*len(bars)-i*10+(j%3), s=names[i][j], ha='center', va='center', color='black',)
		
		mng = plt.get_current_fig_manager()
		mng.window.state('zoomed')
		plt.title('Makespan: '+str(self.max_time))
		plt.show(block=True)
		
# This function, starting from the nodes that are not dependencies of any other, creates recursively the opposite connections, meaning which nodes need the node under exam to finish before they can start
def create_dependencies_graph_rec(end_points):
	for end_point in end_points:
		if (len(end_point.requirements) != 0):
			create_dependencies_graph_rec(end_point.requirements)
			for required in end_point.requirements:
				if end_point not in required.required_by:
					required.required_by.append(end_point)
		
# Starting from an array of leaves (nodes that do not depend on any other) it calculates for each leaf in leaves the deadlines for EDD, using EDD_auto_rec
def EDD_auto(leaves):
	for leaf in leaves:
		EDD_auto_rec(leaf, [])

def EDD_auto_rec(leaf, visited):
	# If this node is not required by anyone else
	if (not leaf.required_by):
		leaf.deadline = 1000
	else:
		deadline_candidates = []
		# For each node (next) that requires this leaf we compute the EDD deadline, once we have explored all the nodes, we take the lowest
		for next in leaf.required_by:
			visited.append(leaf) if leaf not in visited else None
			EDD_auto_rec(next, visited)
			if (next.is_gpu == True):
				deadline_candidates.append((next.deadline - next.time_gpu - next.copy_time))
			else:
				deadline_candidates.append((next.deadline - next.time - next.copy_time))

		leaf.deadline = min(leaf.deadline, min(deadline_candidates))

# Normalizes deadlines and calculates priority points
def calculate_priority_points(all_nodes):
	min_deadline = - 1.0 * min(all_nodes, key=lambda x: x.deadline).deadline
	for node in all_nodes:
		node.deadline += min_deadline
		if (node.is_gpu == True):
			node.priority_point = float(node.deadline - ((PROCESSORS - 1)/PROCESSORS)*node.time_gpu)
		else:
			node.priority_point = float(node.deadline - ((PROCESSORS - 1)/PROCESSORS)*node.time)
		
def main():
	# Timings and dependencies should be imported from a file, for now they are hardcoded
	execution = 0 # This variable is a place holder to introduce pipelining

	# Creates all the nodes, this should be put in iterable structures to allow for pipelining
	scale_0 = Node('scale_0', 0, 2.5, [], execution, 1.25)
	scale_1 = Node('scale_1', 1, 2, [scale_0], execution, 1)
	scale_2 = Node('scale_2', 2, 1.6, [scale_1], execution, 0.8)
	scale_3 = Node('scale_3', 3, 1.28, [scale_2], execution, 0.64)
	scale_4 = Node('scale_4', 4, 1.02, [scale_3], execution, 0.51)
	scale_5 = Node('scale_5', 5, 0.82, [scale_4], execution, 0.41)
	scale_6 = Node('scale_6', 6, 0.66, [scale_5], execution, 0.33)
	scale_7 = Node('scale_7', 7, 0.53, [scale_6], execution, 0.26)

	fast_0 = Node('fast_0', 0, 25, [scale_0], execution, 2)
	fast_1 = Node('fast_1', 1, 20, [scale_1], execution, 1.6)
	fast_2 = Node('fast_2', 2, 16, [scale_2], execution, 1.28)
	fast_3 = Node('fast_3', 3, 12.8, [scale_3], execution, 1.02)
	fast_4 = Node('fast_4', 4, 10.24, [scale_4], execution, 0.82)
	fast_5 = Node('fast_5', 5, 8.19, [scale_5], execution, 0.66)
	fast_6 = Node('fast_6', 6, 6.55, [scale_6], execution, 0.53)
	fast_7 = Node('fast_7', 7, 5.24, [scale_7], execution, 0.42)

	grid_tree_angle_0 = Node('grid_tree_angle_0', 0, 7, [fast_0], execution)
	grid_tree_angle_1 = Node('grid_tree_angle_1', 1, 5.6, [fast_1], execution)
	grid_tree_angle_2 = Node('grid_tree_angle_2', 2, 4.48, [fast_2], execution)
	grid_tree_angle_3 = Node('grid_tree_angle_3', 3, 3.58, [fast_3], execution)
	grid_tree_angle_4 = Node('grid_tree_angle_4', 4, 2.86, [fast_4], execution)
	grid_tree_angle_5 = Node('grid_tree_angle_5', 5, 2.29, [fast_5], execution)
	grid_tree_angle_6 = Node('grid_tree_angle_6', 6, 1.83, [fast_6], execution)
	grid_tree_angle_7 = Node('grid_tree_angle_7', 7, 1.46, [fast_7], execution)

	gauss_0 = Node('gauss_0', 0, 2.8, [scale_0], execution, 0.5)
	gauss_1 = Node('gauss_1', 1, 2.24, [scale_1], execution, 0.4)
	gauss_2 = Node('gauss_2', 2, 1.79, [scale_2], execution, 0.32)
	gauss_3 = Node('gauss_3', 3, 1.43, [scale_3], execution, 0.26)
	gauss_4 = Node('gauss_4', 4, 1.14, [scale_4], execution, 0.21)
	gauss_5 = Node('gauss_5', 5, 0.91, [scale_5], execution, 0.17)
	gauss_6 = Node('gauss_6', 6, 0.73, [scale_6], execution, 0.14)
	gauss_7 = Node('gauss_7', 7, 0.58, [scale_7], execution, 0.11)
		
	orb_0 = Node('orb_0', 0, 3.6, [gauss_0, grid_tree_angle_0], execution, 2.2)
	orb_1 = Node('orb_1', 1, 2.88, [gauss_1, grid_tree_angle_1], execution, 1.76)
	orb_2 = Node('orb_2', 2, 2.3, [gauss_2, grid_tree_angle_2], execution, 1.41)
	orb_3 = Node('orb_3', 3, 1.84, [gauss_3, grid_tree_angle_3], execution, 1.13)
	orb_4 = Node('orb_4', 4, 1.47, [gauss_4, grid_tree_angle_4], execution, 0.9)
	orb_5 = Node('orb_5', 5, 1.18, [gauss_5, grid_tree_angle_5], execution, 0.72)
	orb_6 = Node('orb_6', 6, 0.94, [gauss_6, grid_tree_angle_6], execution, 0.58)
	orb_7 = Node('orb_7', 7, 0.75, [gauss_7, grid_tree_angle_7], execution, 0.46)

	deep_learning = Node('deep_learning', 0, -1, [scale_0], execution, 15)

	# GPU NODES ----------------------------------------------------------------
	scale_0.set_gpu()
	scale_1.set_gpu()
	scale_2.set_gpu()
	scale_3.set_gpu()
	scale_4.set_gpu()
	scale_5.set_gpu()
	scale_6.set_gpu()
	scale_7.set_gpu()

	fast_0.set_gpu()
	fast_1.set_gpu()
	fast_2.set_gpu()
	fast_3.set_gpu()
	fast_4.set_gpu()
	fast_5.set_gpu()
	fast_6.set_gpu()
	fast_7.set_gpu()

	gauss_0.set_gpu()
	gauss_1.set_gpu()
	gauss_2.set_gpu()
	gauss_3.set_gpu()
	gauss_4.set_gpu()
	gauss_5.set_gpu()
	gauss_6.set_gpu()
	gauss_7.set_gpu()

	orb_0.set_gpu()
	orb_1.set_gpu()
	orb_2.set_gpu()
	orb_3.set_gpu()
	orb_4.set_gpu()
	orb_5.set_gpu()
	orb_6.set_gpu()
	orb_7.set_gpu()

	deep_learning.set_gpu()

	end_points = [deep_learning, orb_0, orb_1, orb_2, orb_3, orb_4, orb_5, orb_6, orb_7] # This are the nodes that no other node depends on
	create_dependencies_graph_rec(end_points)

	EDD_auto([scale_0]) # Creates deadlines

	all_nodes = [deep_learning, scale_0, scale_1, scale_2, scale_3, scale_4, scale_5, scale_6, scale_7, fast_0, fast_1, fast_2, fast_3, fast_4, fast_5, fast_6, fast_7, gauss_0, gauss_1, gauss_2, gauss_3, gauss_4, gauss_5, gauss_6, gauss_7, orb_0, orb_1, orb_2, orb_3, orb_4, orb_5, orb_6, orb_7, grid_tree_angle_0, grid_tree_angle_1, grid_tree_angle_2, grid_tree_angle_3, grid_tree_angle_4, grid_tree_angle_5, grid_tree_angle_6, grid_tree_angle_7]

	calculate_priority_points(all_nodes)
			
	# All the nodes are ordered by their priority point (as G-FL demands) or deadline
	if (DEADLINE):
		all_nodes.sort(key=lambda x: x.deadline, reverse=False)
	else:
		all_nodes.sort(key=lambda x: x.priority_point, reverse=False)	

	GFL = Schedule(all_nodes, n_cpu, n_gpu)
	print(('Makespan EDD: ' if DEADLINE else 'Makespan G-FL: ')+str(GFL.create_schedule()))

	GPU_labels = ['GPU '+str(i) for i in range(0, n_gpu)]
	CPU_labels = ['CPU '+str(i) for i in range(0, n_cpu)]
	GFL.create_bar_graph(labels=GPU_labels + CPU_labels)


if __name__ == '__main__':
	main()