import sys
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from decimal import Decimal, ROUND_HALF_EVEN

FILENAME = 'timings.csv'
n_cpu = 2
n_gpu = 1
PROCESSORS = float(n_cpu+n_gpu)	# Used in the formula to calculate priority points for G-FL
DEADLINE = False # If set to true it will schedule using EDD, otherwise G-FL (CPU only)
FRAMES = 3
"""
Structure
	Classes 	(2)
	Functions	(5)
	Main		(1)
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
		later = '\nis scheduled at time: '+str(round(self.scheduled_time, 2)) + concat + '\nPriority: ' + str(round(self.priority_point, 2)) + ' Deadline: '+str(round(self.deadline, 2))
		requirements_str = ''
		for node in self.requirements:
			requirements_str += node.name + '_' + str(node.level) + '_' + str(node.execution) + (' S ' if node.scheduled_time != -1 else ' NS ')
		return '---------------------------\n' + self.name + '_' + str(self.level) + '_' + str(self.execution) + str(' GPU' if self.is_gpu else ' CPU') + '\nDelay: ' + str(round(self.delay, 2)) + '\nRequires: ' + requirements_str + later + '\n---------------------------'
	
# This class creates and verifies the scheduling, prints the scheduling only if it is verified
class Schedule:
	# Takes as parameter a list of all nodes ordered by priority
	def __init__(self, node_list_cpu, node_list_gpu, n_cpu, n_gpu):
		self.node_list_cpu = node_list_cpu
		self.node_list_gpu = node_list_gpu
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
		times_cpu = [[(0.0,0.0)] for i in range(0, self.n_cpu)]
		times_gpu = [[(0.0,0.0)] for i in range(0, self.n_gpu)]
		
		available_cpu = []
		available_gpu = []
		
		available_cpu.append(self.node_list_cpu[0])
		available_gpu.append(self.node_list_gpu[0])

		# Run while there are nodes that can run
		while available_gpu or available_cpu:
			# If there are nodes available for a GPU
			if (available_gpu):
				node = available_gpu[0]
				# Check whats the minimum starting time for this node
				self.set_minimum_start_time(node)
				# It requirements are not met the minimum start time is not set and we go on to a CPU node below
				if (node.scheduled_time != -1):
					# Adds this node to the GPU with the smallest current time
					min_time_gpu = [10000, -1]
					for gpu in times_gpu:
						if min_time_gpu[0] > gpu[-1][1]:
							min_time_gpu[0] = gpu[-1][1]
							min_time_gpu[1] = times_gpu.index(gpu)
					execution_elements_gpu[min_time_gpu[1]].append(node)
					# If the current time is less than the minimum time for this specific node, a delay is added to make sure all dependencies are met
					node.delay = node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] if node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] > 0 else 0
					# The node's scheduled time is updated
					node.scheduled_time = times_gpu[min_time_gpu[1]][-1][1] + node.delay
					# The current time for this processor is updated
					times_gpu[min_time_gpu[1]].append((node.scheduled_time, node.scheduled_time + node.time_gpu + node.copy_time))
					
					for next in node.required_by:
					# the length check on requirmenets is placed so that if a node that was needed by the one just executed depends on more than one node we can verify all dependencies before we add it to those that can be scheduled
						append = True
						for check in next.requirements:
							if check.scheduled_time == -1:
								append = False
						if (next.is_gpu):
							if (append and next not in available_gpu and next.scheduled_time == -1):
								available_gpu.append(next)
						else:
							if (append and next not in available_cpu and next.scheduled_time == -1):
								available_cpu.append(next)
					available_gpu.remove(node)
			
			# Behaves like above, but for CPU cores
			if (available_cpu):
				node = available_cpu[0]
				self.set_minimum_start_time(node)
				if (node.scheduled_time != -1):
					min_time_cpu = [10000, -1]
					for cpu in times_cpu:
						if min_time_cpu[0] > cpu[-1][1]:
							min_time_cpu[0] = cpu[-1][1]
							min_time_cpu[1] = times_cpu.index(cpu)
					execution_elements_cpu[min_time_cpu[1]].append(node)
					node.delay = node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] if node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0
					node.scheduled_time = times_cpu[min_time_cpu[1]][-1][1] + node.delay
					times_cpu[min_time_cpu[1]].append((node.scheduled_time, node.scheduled_time + node.time + node.copy_time))
					for next in node.required_by:
						append = True
						for check in next.requirements:
							if check.scheduled_time == -1:
								append = False
						if (next.is_gpu):
							if (append and next not in available_gpu and next.scheduled_time == -1):
								available_gpu.append(next)
						else:
							if (append and next not in available_cpu and next.scheduled_time == -1):
								available_cpu.append(next)
					available_cpu.remove(node)
			
			if (DEADLINE):
				available_cpu.sort(key=lambda x: x.deadline, reverse=False)
				available_gpu.sort(key=lambda x: x.deadline, reverse=False)
			else:
				available_cpu.sort(key=lambda x: x.priority_point, reverse=False)
				available_gpu.sort(key=lambda x: x.deadline, reverse=False)
		
		# TODO
		# Add can_it_be_scheduled_earlier
		
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
		self.max_time = round(max(times, key=lambda x: x[-1][1])[-1][1], 2)
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
		for req in node.requirements:
			if req.scheduled_time == -1:
				return -1
		node.scheduled_time = max((req.scheduled_time + req.copy_time + (req.time_gpu if req.is_gpu else req.time) for req in node.requirements)) if node.requirements else 0	
		
	"""
	def can_it_be_scheduled_earlier()
			candidates = []
			# Takes all nodes that have requirements met, and calculates the minimum starting times and how much each node will take to run (if they are on the same processor compared to highest priority task)
			for n in available:
				self.set_minimum_start_time(n)
				delay = self.does_this_node_need_to_wait(n, times_gpu, times_cpu) + n.time_gpu if n.is_gpu else n.time
				if self.they_share_the_same_processor(n, available[0]):
					candidates.append([n, delay])
					
			# Calculates how much the highest priority node needs to wait before it can run
			priority_candidate_delay = self.does_this_node_need_to_wait(available[0], times_gpu, times_cpu)
			
			# If there are any candidates that can run (other than the highest priority node), it tries from the one that takes the lowest time, and checks if it can be slotted in the delay of the highest priority task (basically no deadline gets changed, but some tasks might get run earlier)
			found = False
			while candidates and not found:
				minimum_time_needed = min(candidates, key=lambda x: x[1])
				if priority_candidate_delay > minimum_time_needed[1] and minimum_time_needed[0].scheduled_time + (minimum_time_needed[0].time_gpu if minimum_time_needed[0].is_gpu else minimum_time_needed[0].time) < available[0].scheduled_time:					
					node = minimum_time_needed[0]
					found = True
				else:
					candidates.remove(minimum_time_needed)
			"""
			
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
		plt.title('Makespan: '+str(self.max_time) + ', Average Frame Time: '+str(round(float(self.max_time/FRAMES), 2)))
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
		leaf.deadline = 0
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
		
# Normalizes priorities by summing the lowest value (as they are all negativs) and then, based on the execution number, giving more priority to earlier frames	
def normalize_deadlines(all_nodes_exec):
		# Normalizes from earliest deadline negative, to earliest deadline zero
		min_deadline =  - 1.0 * min(node.deadline for node in all_nodes_exec)
		for node in all_nodes_exec:
			node.deadline += min_deadline
		
		# Increases deadline for all nodes based on the execution number
		max_queue_value = max(node.deadline for node in all_nodes_exec)
		for node in all_nodes_exec:
			node.deadline += max_queue_value * (node.execution) # There is no change in deadline or priority if its the first frame (0)
		
# Normalizes deadlines and calculates priority points
def calculate_priority_points(all_nodes):
	for node in all_nodes:
		if (node.is_gpu == True):
			node.priority_point = float(node.deadline - ((PROCESSORS - 1)/PROCESSORS)*node.time_gpu)
		else:
			node.priority_point = float(node.deadline - ((PROCESSORS - 1)/PROCESSORS)*node.time)
	# Get the normalized priority point (obtaining same max lateness for all nodes)
	min_priority_point =  min(all_nodes, key = lambda x: x.priority_point).priority_point
	for node in all_nodes:
		node.priority_point -= min_priority_point
		
def main():
	# Dependencies should be imported from a file, for now they are hardcoded
	levels = 8
	
	scale = [[] for i in range(0, FRAMES)]
	fast = [[] for i in range(0, FRAMES)]
	grid = [[] for i in range(0, FRAMES)]
	gauss = [[] for i in range(0, FRAMES)]
	orb = [[] for i in range(0, FRAMES)]
	deep_learning = [[] for i in range(0, FRAMES)]
	
	# Creates all the nodes
	# Only Grid and Orb have copy times because they are run on the CPU and they need data from the GPU
	# Scale needs data from Scale, but they are all run on the GPU, no data copy needed (same for Fast and Gauss)
	
	timings = {}
	with open(FILENAME) as timings_file:
		for row in timings_file:
			elements = row.split(',')
			values = [float(elements[i]) for i in range(1, len(elements))]
			timings[elements[0]] = values
	
	# All scale_0 nodes, as they are the first to execute of each frame, have special dependencies, and are set before
	# Node(name, level, time, requirements, execution, time_gpu = -1, copy_time = 0):
	scale[0].append(Node('scale', 0, timings['scale_cpu'][0], [], execution=0, time_gpu=timings['scale_gpu'][0]))
	scale[0][0].set_gpu()
	
	for execution in range(0, FRAMES):
		# First node of each execution has special treatment due to special constraints
		if execution > 0:
			scale[execution].append(Node('scale', 0, timings['scale_cpu'][0], [scale[execution-1][0]], execution, timings['scale_gpu'][0]))
			scale[execution][0].set_gpu()
		# As scale_0 are set outside, we need to skip them here, where we initialize all the other nodes
		for level in range(1, levels):
			scale[execution].append(Node('scale', level, timings['scale_cpu'][level], [scale[execution][level-1]], execution, timings['scale_gpu'][level]))
			scale[execution][level].set_gpu()
			
		for level in range(0, levels):
			fast[execution].append(Node('fast', level, timings['fast_cpu'][level], [scale[execution][level]], execution, timings['fast_gpu'][level]))
			fast[execution][level].set_gpu()
			
			grid[execution].append(Node('grid', level, timings['grid_cpu'][level], [fast[execution][level]], execution, copy_time=timings['grid_cpu_gpu_copy'][level]))
			
			gauss[execution].append(Node('gauss', level, timings['gauss_cpu'][level], [scale[execution][level]], execution, timings['gauss_gpu'][level]))
			gauss[execution][level].set_gpu()
			
			if (level == 7 and execution > 0):
				# A Frame cannot complete computing before an earlier one
				orb[execution].append(Node('orb', level, timings['orb_cpu'][level], [gauss[execution][level], grid[execution][level], orb[execution-1][7]], execution, timings['orb_gpu'][level], copy_time=timings['orb_cpu_gpu_copy'][level]))
			else:
				orb[execution].append(Node('orb', level, timings['orb_cpu'][level], [gauss[execution][level], grid[execution][level]], execution, timings['orb_gpu'][level], copy_time=timings['orb_cpu_gpu_copy'][level]))
			#orb[execution][level].set_gpu() # Remove the comment on this line to create a worst case scenario for G-FL
			
		deep_learning[execution].append(Node('deep_learning', 0, timings['deep_learning_cpu'][0], [scale[execution][0]], execution, timings['deep_learning_gpu'][0]))
		deep_learning[execution][0].set_gpu()
		
		#scale[execution][1].set_gpu(False)
		#gauss[execution][0].set_gpu(False)
		
	all_nodes_cpu = []
	all_nodes_gpu = []
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
			# They are added to the respective queue if they are a GPU node or not
			# Scale
			if scale[execution][i].is_gpu:
				all_nodes_gpu.append(scale[execution][i])
			else:
				all_nodes_cpu.append(scale[execution][i])
			# Fast
			if fast[execution][i].is_gpu:
				all_nodes_gpu.append(fast[execution][i])
			else:
				all_nodes_cpu.append(fast[execution][i])
			# Grid
			if grid[execution][i].is_gpu:
				all_nodes_gpu.append(grid[execution][i])
			else:
				all_nodes_cpu.append(grid[execution][i])
			# Gauss
			if gauss[execution][i].is_gpu:
				all_nodes_gpu.append(gauss[execution][i])
			else:
				all_nodes_cpu.append(gauss[execution][i])
			# Orb
			if orb[execution][i].is_gpu:
				all_nodes_gpu.append(orb[execution][i])
			else:
				all_nodes_cpu.append(orb[execution][i])

		
		normalize_deadlines(all_nodes_exec)
		calculate_priority_points(all_nodes_exec)
		# Deep learning is GPU only and has no levels so its added outside the for cycle
		all_nodes_gpu.append(deep_learning[execution][0])
	
	# All the nodes are ordered by their priority point (as G-FL demands) or deadline
	if (DEADLINE):
		all_nodes_cpu.sort(key=lambda x: x.deadline, reverse=False)
		all_nodes_gpu.sort(key=lambda x: x.deadline, reverse=False)
	else:
		all_nodes_cpu.sort(key=lambda x: x.priority_point, reverse=False)		
		all_nodes_gpu.sort(key=lambda x: x.deadline, reverse=False)
	
	# Sort all nodes using a stable algorithm by execution (if they have the same priority the earlier executions go first)
	all_nodes_cpu.sort(key=lambda x: x.execution, reverse=False)
	all_nodes_gpu.sort(key=lambda x: x.execution, reverse=False)
		
	GFL = Schedule(all_nodes_cpu, all_nodes_gpu, n_cpu, n_gpu)
	max_time = GFL.create_schedule()
	print(('Makespan EDD: ' if DEADLINE else 'Makespan G-FL: ')+str(max_time)+ ', Average Frame Time: '+str(round(float(max_time/FRAMES), 2)))

	GPU_labels = ['GPU '+str(i) for i in range(0, n_gpu)]
	CPU_labels = ['CPU '+str(i) for i in range(0, n_cpu)]
	GFL.create_bar_graph(labels=GPU_labels + CPU_labels)
	
	max_lateness = 0
	n_nodes = 0
	avg_lateness = 0
	for node in all_nodes_cpu:
		n_nodes += 1
		avg_lateness += (node.scheduled_time + node.time) - node.deadline
		max_lateness = max(max_lateness, (node.scheduled_time + node.time) - node.deadline)
	for node in all_nodes_gpu:
		n_nodes += 1
		avg_lateness += (node.scheduled_time + node.time_gpu) - node.deadline
		max_lateness = max(max_lateness, (node.scheduled_time + node.time_gpu) - node.deadline)
	print('avg lateness: '+str(round(float(avg_lateness/n_nodes), 2)))
	print('max lateness: '+str(round(max_lateness, 2)))

if __name__ == '__main__':
	if (len(sys.argv) == 4):
		try:
			FILENAME = sys.argv[1]
			DEADLINE = bool(int(sys.argv[2]))
			FRAMES = int(sys.argv[3])
		except:
			print('Usage: \n sim.py timings_file.csv DEADLINE(0,1) FRAMES(n)\nExample: open timings.csv, simulate three frames and  use EDD\n\tsim.py timings.csv 1 3', file=sys.stderr)
			exit()
	elif (len(sys.argv) == 3):
		try:
			DEADLINE = bool(int(sys.argv[1]))
			FRAMES = int(sys.argv[2])
		except:
			print('Usage: \n sim.py [timings_file.csv] DEADLINE(0,1) FRAMES(n)\nExample: open timings.csv, simulate three frames and  use EDD\n\tsim.py timings.csv 1 3', file=sys.stderr)
			exit()
	elif (len(sys.argv) == 2):
		if (sys.argv[1][0]) == '?':
			print('Usage: \n sim.py FILENAME[csv] DEADLINE(0,1) FRAMES(n)\nExample: open timings.csv, simulate three frames and  use EDD\n\tsim.py timings.csv 1 3\nRunning with default settings: \n\tFile ' +str(FILENAME) + ' Frames: '+str(FRAMES) + ' Deadline: '+str(DEADLINE))
	main()