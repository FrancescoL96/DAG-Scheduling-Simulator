from decimal import Decimal, ROUND_HALF_EVEN
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

n_cpu = 2
n_gpu = 1
PROCESSORS = float(n_cpu+n_gpu)	# Used in the formula to calculate priority points for G-FL
DEADLINE = False # If set to true it will schedule using EDD, otherwise G-FL
FRAMES = 3

"""
There is a class Node, which rappresents a single computing node, such as Fast, or Scale, it has various fields, which are used to create the dependencies' graph
and to verify the scheduling
The scheduling is created, verified and shown by the class Schedule, which taken a list of nodes, once it has created the scheduling, verifies that all temporal restrictions are respected
There are a couple functions (EDD_auto, EDD_auto_rec, create_dependencies_graph_rec, calculate_priority_points) which are used to compute some additional data on the nodes, specifically, EDD deadlines and dependencies in the reverse order. The last function calculates priority points for G-FL
"""
class Node:
	def __init__(self, name, level, time, requirements, execution, time_gpu = -1, copy_time = 0):
		self.name = name					# Name of the node that needs to scheduled
		self.level = level					# Level of the node (0 to 7)
		self.time = time					# Runtime
		self.requirements = requirements	# List of nodes which this node depends on
		self.time_gpu = time_gpu			# If it has a GPU implementation set the time here
		self.execution = execution			# This is a numeric value used to indicate to which graph this nodes belong to
		self.copy_time = copy_time			# Used to account for copy time between CPU/GPU ----- It is in the code, it is used, but it is never set
		self.is_gpu = False					# If this value is True then the GPU time is used
		# This values are set automatically
		self.delay = 0						# How long does this node need to wait before it can start
		self.scheduled_time = -1			# At what time is this node scheduled
		self.required_by = []				# Which nodes require this be completed (sed to calculate deadlines)
		self.deadline = 1.0					# EDD deadline, calculated using recursive formula: EDD(n) = [for each successor s] - min(EDD(s) - WCET - Cpy)
		self.priority_point = 0.0			# Proprity value for G-FL scheduling, calculated for each node using: Y = Deadline - ((Procs - 1)/Procs) * WCET - min [for each job j] (Deadline_j - ((Procs - 1)/Procs) * WCET_j)
		# Currently using the relative priority point formula: Y = Deadline - ((Procs - 1)/Procs) * WCET
	
	# If this function is called, is_gpu is set to True, allowing the code to use time_gpu for execution time intead of time (which should be used for cpu time)
	# It has a parameter in case it is needed to remove a node from the GPU, otherwise just sets to True
	def set_gpu(self, value=True):
		if (self.time_gpu != -1):
			self.is_gpu = value
	
	# If a node cannot start right away and dependencies do not allow the node to start right away, set this value to make the node wait before it starts
	def set_delay(self, delay):
		if (delay >= 0):
			self.delay = delay
	
	# Overrides the string function, printing the node name
	# Or other stuff actually, just print what you need :)
	def __str__(self):
		cpu_time = '\nfinishes at (cpu): ' + str(round(self.scheduled_time + self.time, 2))
		gpu_time = '\nfinishes at (gpu): ' + str(round(self.scheduled_time + self.time_gpu, 2))
		concat = gpu_time if (self.is_gpu == True) else cpu_time
		later = '\nis scheduled at time: '+str(round(self.scheduled_time, 2)) + concat + '\nPriority: ' + str(round(self.priority_point, 2))
		requirements_str = ''
		for node in self.requirements:
			requirements_str += node.name + '_' + str(node.level) + '_' + str(node.execution) + ' '
		return '---------------------------\n' + self.name + '_' + str(self.level) + '_' + str(self.execution) + str(' GPU' if self.is_gpu else ' CPU') + '\nDelay: ' + str(round(self.delay, 2)) + '\nRequires: ' + requirements_str + later + '\n---------------------------'
	
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
			node = available[0] # The next node to schedule is the highest priority (it might get ovverridden)
			candidates = []
			# Takes all nodes that have requirements met, and calculates the minimum starting times and how much each node will take to run (if they are on the same processor compared to highest priority task)
			for n in available:
				self.set_minimum_start_time(n)
				delay = self.does_this_node_need_to_wait(n, times_gpu, times_cpu) + n.time_gpu if n.is_gpu else n.time
				if self.they_share_the_same_processor(n, available[0]):
					candidates.append([n, delay])
					
			# Calculates how much the highest priority node needs to wait before it can run
			priority_candidate_delay = self.does_this_node_need_to_wait(available[0], times_gpu, times_cpu)
			
			# If there are any candidates that can run (other than the highest priority node), it takes the one that takes the lowest time, and checks if it can be slotted in the delay of the highest priority task (basically no deadline gets changed, but some task might get run earlier)
			found = False
			while candidates and not found:
				minimum_time_needed = min(candidates, key=lambda x: x[1])	
				if priority_candidate_delay > minimum_time_needed[1] and minimum_time_needed[0].scheduled_time + (minimum_time_needed[0].time_gpu if minimum_time_needed[0].is_gpu else minimum_time_needed[0].time) < available[0].scheduled_time:					
					node = minimum_time_needed[0]
					found = True
				else:
					candidates.remove(minimum_time_needed)
			
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
				# the length check on requirmenets is place so that if a node that was needed by the one just executed depends on more than one node we can verify all dependencies before we add it to those that can be scheduled
				if (next.scheduled_time == -1 and next not in available and len(next.requirements) <= 1):
					available.append(next)
				else:
					# If any one the dependencies is not met (one is met for sure: 'node' has just been executed, and it is a dependency for next), we cannot add it to the available nodes 
					append = True
					for check in next.requirements:
						if check.scheduled_time == -1:
							append = False
					if (append and next not in available):
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
		
	# If the node cannot start but needs to wait, return True
	def does_this_node_need_to_wait(self, node, times_gpu, times_cpu):
		if (node.is_gpu):
			return node.scheduled_time - times_gpu[times_gpu.index(min(times_gpu))] if node.scheduled_time - times_gpu[times_gpu.index(min(times_gpu))] > 0 else 0
		else:
			return node.scheduled_time - times_cpu[times_cpu.index(min(times_cpu))] if node.scheduled_time - times_cpu[times_cpu.index(min(times_cpu))] > 0 else 0
	
	def they_share_the_same_processor(self, node_0, node_1):
		return node_0.is_gpu == node_1.is_gpu
			
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
				name.append(node.name[0]+'_'+str(node.level)+'_'+str(node.execution))
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
	levels = 8
	
	scale = [[] for i in range(0, FRAMES)]
	fast = [[] for i in range(0, FRAMES)]
	grid = [[] for i in range(0, FRAMES)]
	gauss = [[] for i in range(0, FRAMES)]
	orb = [[] for i in range(0, FRAMES)]
	deep_learning = [[] for i in range(0, FRAMES)]
	
	# Creates all the nodes, this should be put in iterable structures to allow for pipelining
	# Only Grid and Orb have copy times because they are run on the CPU and they need data from the GPU
	# Scale needs data from Scale, but they are all run on the GPU, no data copy needed (same for Fast and Gauss)
	
	scale_timings_cpu = [2.5, 2.0, 1.6, 1.28, 1.02, 0.82, 0.66, 0.53]
	scale_timings_gpu = [1.25, 1.0, 0.8, 0.64, 0.51, 0.41, 0.33, 0.26]
	
	fast_timings_cpu = [25, 20, 16, 12.8, 10.24, 8.19, 6.55, 5.24]
	fast_timings_gpu = [2.0, 1.6, 1.28, 1.02, 0.82, 0.66, 0.53, 0.42]
	
	grid_timings_cpu = [7.0, 5.6, 4.48, 3.58, 2.86, 2.29, 1.83, 1.46]
	grid_timings_cpu_gpu_copy = [0.5, 0.4, 0.32, 0.26, 0.21, 0.17, 0.14, 0.11]
	
	gauss_timings_cpu = [2.8, 2.24, 1.78, 1.43, 1.14, 0.91, 0.73, 0.58]
	gauss_timings_gpu = [0.5, 0.4, 0.32, 0.26, 0.21, 0.17, 0.14, 0.11]
	
	orb_timings_cpu = [3.6, 2.88, 2.3, 1.84, 1.47, 1.18, 0.94, 0.75]
	# Gpu run times for orb are synthetic and do not reflect real world performance, they are used to fabricate a worst case scenario
	orb_timings_gpu = [2.2, 1.76, 1.41, 1.13, 0.9, 0.72, 0.58, 0.46]
	orb_timings_cpu_gpu_copy = [0.5, 0.4, 0.32, 0.26, 0.21, 0.17, 0.14, 0.11]
	
	deep_learning_timings_gpu = [15.0]
	
	# All scale_0 nodes, as they are the first to execute of each frame, have special dependencies, and are set before
	# Node(name, level, time, requirements, execution, time_gpu = -1, copy_time = 0):
	scale[0].append(Node('scale', 0, scale_timings_cpu[0], [], 0, scale_timings_gpu[0]))
	scale[0][0].set_gpu()
	
	for execution in range(0, FRAMES):
		# First node of each execution has special treatment due to special constraints 
		if execution > 0:
			scale[execution].append(Node('scale', 0, scale_timings_cpu[0], [scale[execution-1][0]], execution, scale_timings_gpu[0]))
			scale[execution][0].set_gpu()
		# As scale_0 are set outside, we need to skip them here, where we initialize all the other nodes
		for level in range(1, levels):
			scale[execution].append(Node('scale', level, scale_timings_cpu[level], [scale[execution][level-1]], execution, scale_timings_gpu[level]))
			scale[execution][level].set_gpu()
			
		for level in range(0, levels):
			fast[execution].append(Node('fast', level, fast_timings_cpu[level], [scale[execution][level]], execution, fast_timings_gpu[level]))
			fast[execution][level].set_gpu()
			
			grid[execution].append(Node('grid', level, grid_timings_cpu[level], [fast[execution][level]], execution, copy_time=grid_timings_cpu_gpu_copy[level]))
			
			gauss[execution].append(Node('gauss', level, gauss_timings_cpu[level], [scale[execution][level]], execution, gauss_timings_gpu[level]))
			gauss[execution][level].set_gpu()
			
			orb[execution].append(Node('orb', level, orb_timings_cpu[level], [gauss[execution][level], grid[execution][level]], execution, orb_timings_gpu[level]))
			#orb[execution][level].set_gpu() # Remove the comment on this line to create a worst case scenario for G-FL
			
		deep_learning[execution].append(Node('deep_learning', 0, -1, [scale[execution][0]], execution, deep_learning_timings_gpu[0]))
		deep_learning[execution][0].set_gpu()
		

	all_nodes = []
	max_queue_value = 0 # this value is used to normalize deadlines or priorities based on the execution number
	for execution in range(0, FRAMES):
		end_points = [orb[execution][i] for i in range (0, levels)]	# This are the nodes that no other node depends on
		end_points.append(deep_learning[execution][0])
		create_dependencies_graph_rec(end_points)

		EDD_auto([scale[execution][0]]) # Creates deadlines
		all_nodes_exec = [deep_learning[execution][0]]
		for i in range(0, levels):
			# These are the nodes for the execution under exam
			all_nodes_exec.append(scale[execution][i])
			all_nodes_exec.append(fast[execution][i])
			all_nodes_exec.append(grid[execution][i])
			all_nodes_exec.append(gauss[execution][i])
			all_nodes_exec.append(orb[execution][i])
			# These are ALL the nodes used for the scheduling
			all_nodes.append(scale[execution][i])
			all_nodes.append(fast[execution][i])
			all_nodes.append(grid[execution][i])
			all_nodes.append(gauss[execution][i])
			all_nodes.append(orb[execution][i])
			
		calculate_priority_points(all_nodes_exec)
		max_queue_value = max(node.deadline for node in all_nodes_exec) if DEADLINE else max(node.priority_point for node in all_nodes_exec)
		
		all_nodes.append(deep_learning[execution][0])
		
	# Normalizes priorities based on the execution number, giving more priority to earlier frames
	
	for node in all_nodes:
		if (DEADLINE):
			node.deadline += max_queue_value * node.execution # There is no change in deadline or priority if its the first frame
		else:
			node.priority_point += max_queue_value * node.execution
	
	
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