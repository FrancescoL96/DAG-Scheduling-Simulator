import sys
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from decimal import Decimal, ROUND_HALF_EVEN

GRAPH_FILE = 'graph_file.csv' # File to import the graph							!!! (can be overridden with program parameters)
n_cpu = 2
n_gpu = 1
PROCESSORS = float(n_cpu+n_gpu)	# Used in the formula to calculate priority points for G-FL
DEADLINE = False # If set to true it will schedule using EDD 						!!! (can be overridden with program parameters)
GFL = False # If set to true it will schedule using G-FL (CPU only) 				!!! (can be overridden with program parameters)
HEFT = False # If set to true it will schedule using HEFT 							!!! (can be overridden with program parameters)
PIPELINING = False # If set to true it will enable pipelining						!!! (can be overridden with program parameters)
FRAMES = 3 # Default value is 3 													!!! (can be overridden with program parameters)
MAXIMUM_PIPELINING_ATTEMPTS = 100
MINIMUM_FRAME_DELAY = 33.33 # The minimum time between frames, 16ms is 60 fps, 33ms is 30 fps, 22ms is 45 fps

# If set to True it will display a graph with the scheduling
SHOW_GRAPH = True
"""
Structure
	Classes 	(2)
	Functions	(11)
	Main		(1)
"""
class Node:
	'''
	__init__
	set_gpu
	time
	avg_time
	__str__
	'''
	def __init__(self, name, level, time_cpu, requirements, execution, time_gpu = -1, copy_time = 0):
		self.name = name					# Name of the node that needs to scheduled
		self.level = level					# Level of the node (0 to 7)
		self.time_cpu = time_cpu			# Runtime
		self.requirements = requirements	# List of nodes which this node depends on
		self.time_gpu = time_gpu			# If it has a GPU implementation set the time here
		self.execution = execution			# This is a numeric value used to indicate to which graph this nodes belong to
		self.is_gpu = False					# If this value is True then the GPU time is used
		self.copy_time = copy_time			# Copy time between processors memories to start this node
		# This values are set automatically
		self.delay = 0						# How long does this node need to wait before it can start
		self.scheduled_time = -1			# At what time is this node scheduled
		self.required_by = []				# Which nodes require this be completed (sed to calculate deadlines)
		self.deadline = 1.0					# EDD deadline, calculated using recursive formula: EDD(n) = [for each successor s] - min(EDD(s) - WCET - Cpy)
		self.priority_point = 0.0			# Proprity value for G-FL scheduling, calculated for each node using: Y = Deadline - ((Procs - 1)/Procs) * WCET - min [for each job j] (Deadline_j - ((Procs - 1)/Procs) * WCET_j)
		self.heft_rank = 0.0				# Rank value for HEFT scheduling
		# This two variables are used to know on which processor the node is scheduled
		self.scheduled_on_cpu = -1
		self.scheduled_on_gpu = -1
	
	# If this function is called, is_gpu is set to True, allowing the code to use time_gpu for execution time intead of time (which should be used for cpu time)
	# It has a parameter in case it is needed to remove a node from the GPU, otherwise just sets to True
	def set_gpu(self, value=True):
		if (self.time_gpu != -1):
			self.is_gpu = value
	
	# Returns the computation time required by this node, taking into account if the node is run on CPU or GPU
	def time(self):
		return (self.time_gpu if self.is_gpu else self.time_cpu) + self.copy_time
		
	def avg_time(self):
		if (self.time_cpu != -1 and self.time_gpu != -1):
			return self.time_gpu + self.time_cpu / 2.0
		elif (self.time_cpu != -1):
			return self.time_cpu
		else:
			return self.time_gpu
			
	# If any data of a required node is on a different processor we need a copy
	def check_copy(self):
		# The copy time is used in the code by accessing this variable, at this point it is still unknown wheter we will use this value or not, so a copy is made to restore it in case it's needed
		keep_copy = self.copy_time
		self.copy_time = 0
		for req in self.requirements:
			if req.is_gpu != self.is_gpu:
				self.copy_time = keep_copy
	
	# Overrides the string function, printing the node name
	# Or other stuff actually, just print what you need :)
	def __str__(self):
		cpu_time = '\nfinishes at (cpu): ' + str(round(self.scheduled_time + self.time_cpu + self.copy_time, 2))
		gpu_time = '\nfinishes at (gpu): ' + str(round(self.scheduled_time + self.time_gpu + self.copy_time, 2))
		concat = (gpu_time if (self.is_gpu) else cpu_time) + ' (' + str(self.copy_time) + ')'
		later = '\nis scheduled at time: '+str(round(self.scheduled_time, 2)) + concat + '\nPriority: ' + str(round(self.priority_point, 2)) + ' Deadline: '+str(round(self.deadline, 2))
		requirements_str = ''
		for node in self.requirements:
			requirements_str += node.name + '_' + str(node.level) + '_' + str(node.execution) + (' S ' if node.scheduled_time != -1 else ' NS ')
		return '---------------------------\n' + self.name + '_' + str(self.level) + '_' + str(self.execution) + str(' GPU' if self.is_gpu else ' CPU') + '\nDelay: ' + str(round(self.delay, 2)) + (' Scheduled on CPU: '+str(self.scheduled_on_cpu) if self.scheduled_on_cpu != -1 else ' Scheduled on GPU: '+str(self.scheduled_on_gpu)) + '\nRequires: ' + requirements_str + later + '\n---------------------------'
	
# This class creates and verifies the scheduling, prints the scheduling only if it is verified
class Schedule:
	'''
	__init__
	set_starting_points
	create_schedule
	create_schedule_HEFT
	verify_scheduling
	pipeline
	error_function
	update_frame_counter
	set_minimum_start_time
	can_any_node_be_run_earlier
	calculate_idle_time
	get_first_time_gap
	create_bar_graph
	'''
	# Takes as parameter a list of all nodes ordered by priority and execution, separeted between CPU and GPU nodes
	def __init__(self, node_list_cpu, node_list_gpu, n_cpu, n_gpu):
		self.node_list_cpu = node_list_cpu
		self.node_list_gpu = node_list_gpu
		self.verified = False # A schedule is verified if the nodes in the current schedule meet all dependencies
		self.max_time = 0 # This is the longest time for any processor
		self.max_time_pipelining = 0 # This value is used as a temporal value to calculate the pipelining improvement
		self.n_cpu = n_cpu
		self.n_gpu = n_gpu
		# Ready queues for CPU and GPU tasks of the current frame
		self.available_cpu = [[] for i in range(0, FRAMES)]
		self.available_gpu = [[] for i in range(0, FRAMES)]
		# Always starts from frame zero
		self.current_frame = 0
	
	# The nodes which have no dependencies are added to their respective ready queues
	def set_starting_points(self, starting_points):
		for node in starting_points:
			for frame in range(0, FRAMES):
				if node.execution == frame:
					# If we are working with HEFT all the nodes are added to the CPU queue, as we need just one with HEFT (mapping is done automatically)
					if node.is_gpu and not HEFT:
						self.available_gpu[frame].append(node)
					else:
						self.available_cpu[frame].append(node)
	
	# Creates the scheduling
	def create_schedule(self):
		# For each CPU and GPU creates a list of nodes
		execution_elements_cpu = [[] for i in range(0, self.n_cpu)]
		execution_elements_gpu = [[] for i in range(0, self.n_gpu)]
		# Creates the same lists to save times (at what time is the computation) for each processor
		times_cpu = [[(0.0,0.0)] for i in range(0, self.n_cpu)]
		times_gpu = [[(0.0,0.0)] for i in range(0, self.n_gpu)]
		
		# Run while there are nodes that can run
		# This while takes advantage of lazy expression evaluation to avoid index out of bounds
		while self.current_frame < FRAMES and (self.available_gpu[self.current_frame] or self.available_cpu[self.current_frame]):
			# The processor with the earliest time goes first			
			if ((min(times_gpu, key=lambda x: x[-1][1])[-1][1]+(self.available_gpu[self.current_frame][0].time_gpu if self.available_gpu[self.current_frame] else 0) <= min(times_cpu, key=lambda x: x[-1][1])[-1][1]+(self.available_cpu[self.current_frame][0].time_cpu if self.available_cpu[self.current_frame] else 0) and self.available_gpu[self.current_frame]) or not self.available_cpu[self.current_frame]):
				# If there are nodes available for a GPU
				if (self.available_gpu[self.current_frame]):
					node = self.available_gpu[self.current_frame][0]
					# Check whats the minimum starting time for this node
					self.set_minimum_start_time(node)
					# If requirements are not met the minimum start time is not set and we go on to a CPU node below
					# We take the last run required node that was run on the same processor
					required = None
					if node.requirements:
						required = node.requirements[0]
						for req in node.requirements:
							if req.is_gpu and node.is_gpu:
								if req.scheduled_time > required.scheduled_time:
									required = req
					if (node.scheduled_time != -1):
						min_time_gpu = [50000, -1]
						# If the requirements is the last run node, we schedule the current node on the same processor (instead of creating a delay)
						if required != None and required.is_gpu and execution_elements_gpu[required.scheduled_on_gpu][-1] == required:
							min_time_gpu[0] = [times_gpu[required.scheduled_on_gpu]][-1][1]
							min_time_gpu[1] = required.scheduled_on_gpu
						# Adds this node to the GPU with the smallest current time
						else: 
							for gpu in times_gpu:
								if min_time_gpu[0] > gpu[-1][1]:
									min_time_gpu[0] = gpu[-1][1]
									min_time_gpu[1] = times_gpu.index(gpu)
						
						# If this node has to wait before it can start, we try a different node that maybe doesn't have to wait, without delaying this one
						if node.scheduled_time - (times_gpu[min_time_gpu[1]][-1][1] if node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] > 0 else 0) > 0:
							# We skip the first node, as we already considered it
							for i in range(1, len(self.available_gpu[self.current_frame])):
								# We take one of the available to run nodes
								attempt_early_start_node = self.available_gpu[self.current_frame][i]
								# We check if it can start (it should be able to, as it is in the ready queue, all dependencies should be already scheduled)
								self.set_minimum_start_time(attempt_early_start_node)
								# We calculate how long it would have to wait in the current situation before it can run (even if all dependencies have been scheduled, they may have not finished running)
								delay_early_node = attempt_early_start_node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] if attempt_early_start_node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] > 0 else 0
								# We check if running this node before the highest priority one would delay it, if not, we run this node
								if times_gpu[min_time_gpu[1]][-1][1] + delay_early_node + attempt_early_start_node.time_gpu + attempt_early_start_node.copy_time < node.scheduled_time:
									# This node is not currently scheduled so its time is reset
									node.scheduled_time = -1
									# The new node to schedule is this new one, that doesn't delay the current one
									node = attempt_early_start_node
									break
								else:
									# If it does delay, we reset the time and keep searching
									attempt_early_start_node.scheduled_time = -1
					
						execution_elements_gpu[min_time_gpu[1]].append(node)
						# If the current time is less than the minimum time for this specific node, a delay is added to make sure all dependencies are met
						node.delay = node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] if node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] > 0 else 0
						# The node's scheduled time is updated
						node.scheduled_time = round(times_gpu[min_time_gpu[1]][-1][1] + node.delay, 2)
						# The current time for this processor is updated
						times_gpu[min_time_gpu[1]].append((round(node.scheduled_time, 2), round(node.scheduled_time + node.time_gpu + node.copy_time, 2)))
						node.scheduled_on_gpu = min_time_gpu[1]
						
						# Now that this node is complete we can put in the ready queue all the nodes that have all their dependencies met(we check only the ones connected to the one just executed as the other will not have changed their status)
						for next in node.required_by:
							append = True # We use this variable to know if all dependencies are met and we can append the node to the ready queue
							for check in next.requirements:
								if check.scheduled_time == -1:
									append = False # If at least one node is not scheduled we cannot put in the ready queue this one
							# If it is a GPU node we append it to the ready queue for the GPU, otherwise CPU queue
							if (next.is_gpu):
								if (append and next not in self.available_gpu[next.execution] and next.scheduled_time == -1):
									self.available_gpu[next.execution].append(next)
							else:
								if (append and next not in self.available_cpu[next.execution] and next.scheduled_time == -1):
									self.available_cpu[next.execution].append(next)
						self.available_gpu[self.current_frame].remove(node)
						node.delay = 0 # Once the node has been scheduled we can reset this value, as it has been considered in the scheduled time
			else:
				# Behaves like above, but for CPU cores
				if (self.available_cpu[self.current_frame]):
					node = self.available_cpu[self.current_frame][0]
					self.set_minimum_start_time(node)
					required = None
					if node.requirements:
						required = node.requirements[0]
						for req in node.requirements:
							if not req.is_gpu and not node.is_gpu:
								if req.scheduled_time > required.scheduled_time:
									required = req
					if (node.scheduled_time != -1):
						min_time_cpu = [50000, -1]
						if required != None and not required.is_gpu and execution_elements_cpu[required.scheduled_on_cpu][-1] == required:
							min_time_cpu[0] = [times_cpu[required.scheduled_on_cpu]][-1][1]
							min_time_cpu[1] = required.scheduled_on_cpu
						else:
							for cpu in times_cpu:
								if min_time_cpu[0] > cpu[-1][1]:
									min_time_cpu[0] = cpu[-1][1]
									min_time_cpu[1] = times_cpu.index(cpu)
									
						# For fake Pipelining improvements, remove from here ->
						if node.scheduled_time - (times_cpu[min_time_cpu[1]][-1][1] if node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0) > 0:
							for i in range(1, len(self.available_cpu[self.current_frame])):
								attempt_early_start_node = self.available_cpu[self.current_frame][i]
								self.set_minimum_start_time(attempt_early_start_node)
								delay_early_node = attempt_early_start_node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] if attempt_early_start_node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0
								if times_cpu[min_time_cpu[1]][-1][1] + delay_early_node + attempt_early_start_node.time_cpu + attempt_early_start_node.copy_time < node.scheduled_time:
									node.scheduled_time = -1
									node = attempt_early_start_node
									break
								else:
									attempt_early_start_node.scheduled_time = -1
						# <- to here

						execution_elements_cpu[min_time_cpu[1]].append(node)
						node.delay = node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] if node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0
						node.scheduled_time = round(times_cpu[min_time_cpu[1]][-1][1] + node.delay, 2)
						times_cpu[min_time_cpu[1]].append((round(node.scheduled_time, 2), round(node.scheduled_time + node.time_cpu + node.copy_time, 2)))
						node.scheduled_on_cpu = min_time_cpu[1]
						for next in node.required_by:
							append = True
							for check in next.requirements:
								if check.scheduled_time == -1:
									append = False
							if (next.is_gpu):
								if (append and next not in self.available_gpu[next.execution] and next.scheduled_time == -1):
										self.available_gpu[next.execution].append(next)
							else:
								if (append and next not in self.available_cpu[next.execution] and next.scheduled_time == -1):
									self.available_cpu[next.execution].append(next)
						self.available_cpu[self.current_frame].remove(node)
						node.delay = 0
						
			# The two queues are sorted using EDD and G-FL (cpu only)
			# It is not needed to sort them by execution, as it is a stable sort
			if (DEADLINE):
				self.available_cpu[self.current_frame].sort(key=lambda x: x.deadline, reverse=False)
				self.available_gpu[self.current_frame].sort(key=lambda x: x.deadline, reverse=False)
			elif (GFL):
				self.available_cpu[self.current_frame].sort(key=lambda x: x.priority_point, reverse=False)
				self.available_gpu[self.current_frame].sort(key=lambda x: x.deadline, reverse=False)
			
			self.update_frame_counter()
		
		# Trasforms the separeted lists in a sigle one: [CPU_0 list[node_a, node_b], CPU_1 list[node_c, node_d], ..., GPU_0 list[node_e, node_f], ...]
		self.node_list = execution_elements_cpu + execution_elements_gpu
		for ce in self.node_list:
			ce.sort(key=lambda x: x.scheduled_time)
		times = times_cpu + times_gpu
		
		self.verify_scheduling(times)
		self.pipeline(execution_elements_cpu, execution_elements_gpu, times_cpu, times_gpu)
		
		self.node_list = execution_elements_cpu + execution_elements_gpu
		for ce in self.node_list:
			ce.sort(key=lambda x: x.scheduled_time)

		return self.verify_scheduling(times)
	
	# This function creates the scheduling for HEFT, in this case self.node_list_cpu refers to all nodes, and not just the CPU ones (as it moves them between processors it self, the separated queues are not needed)
	# It behaves very similarly to create_schedule, but is not as optimized. The optimizations are not required as the choice for the processor is done by the algorithm and not by my code
	def create_schedule_HEFT(self):	
		execution_elements_cpu = [[] for i in range(0, self.n_cpu)]
		execution_elements_gpu = [[] for i in range(0, self.n_gpu)]
		
		times_cpu = [[(0.0,0.0)] for i in range(0, self.n_cpu)]
		times_gpu = [[(0.0,0.0)] for i in range(0, self.n_gpu)]
			
		# While there are still frames to process and there are nodes to compute we keep iterating
		while (self.current_frame < FRAMES and self.available_cpu[self.current_frame]):
			node = self.available_cpu[self.current_frame][0]
			self.set_minimum_start_time(node)
			
			# We take the last run required node that was run on the same processor
			required_cpu = None
			required_gpu = None
			for req in node.requirements:
				if (required_gpu == None and req.is_gpu) or (req.is_gpu and req.scheduled_time > required_gpu.scheduled_time):
					required_gpu = req
				elif (required_cpu == None and not req.is_gpu) or (not req.is_gpu and req.scheduled_time > required_cpu.scheduled_time):
					required_cpu = req

			# If the selected node has a minimum start time set by "set_minimum_start_time" we continue (it should always be set, it is still checked)
			if (node.scheduled_time != -1):
				min_time_cpu = [50000, -1]
				min_time_gpu = [50000, -1]
				# If the node has a CPU time, we take the processor with the earliest start time
				if (node.time_cpu != -1):
					for cpu in times_cpu:
						if min_time_cpu[0] > cpu[-1][1]:
							min_time_cpu[0] = cpu[-1][1]
							min_time_cpu[1] = times_cpu.index(cpu)
				# If the node has a GPU time, we take the GPU processor with the earliest start time
				if (node.time_gpu != -1):
					for gpu in times_gpu:
						if min_time_gpu[0] > gpu[-1][1]:
							min_time_gpu[0] = gpu[-1][1]
							min_time_gpu[1] = times_gpu.index(gpu)
				
				# If the node finishes on the GPU before it can finish on the CPU we run it there (having a GPU time or not is deciced through the min_time set to 50000)				
				if min_time_cpu[0] + node.time_cpu > min_time_gpu[0] + node.time_gpu:
					
					# If this node has to wait before it can start, we try a different node that maybe doesn't have to wait, without delaying this one
					if node.scheduled_time - (times_gpu[min_time_gpu[1]][-1][1] if node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] > 0 else 0) > 0:
						# We skip the first node, as we already considered it
						for i in range(1, len(self.available_cpu[self.current_frame])):
							# We take one of the available to run nodes
							attempt_early_start_node = self.available_cpu[self.current_frame][i]
							# We check if it can start (it should be able to, as it is in the ready queue, all dependencies should be already scheduled)
							self.set_minimum_start_time(attempt_early_start_node)
							# We calculate how long it would have to wait in the current situation before it can run (even if all dependencies have been scheduled, they may have not finished running)
							delay_early_node = attempt_early_start_node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] if attempt_early_start_node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] > 0 else 0
							# We check if running this node before the highest priority one would delay it, if not, we run this node
							if times_gpu[min_time_gpu[1]][-1][1] + delay_early_node + attempt_early_start_node.time_gpu + attempt_early_start_node.copy_time < node.scheduled_time and attempt_early_start_node.time_gpu != -1:
								# This node is not currently scheduled so its time is reset
								node.scheduled_time = -1
								# The new node to schedule is this new one, that doesn't delay the current one
								node = attempt_early_start_node
								break
							else:
								# If it does delay, we reset the time and keep searching
								attempt_early_start_node.scheduled_time = -1
				
					# All the values of the node are updated
					node.delay = node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] if node.scheduled_time - times_gpu[min_time_gpu[1]][-1][1] > 0 else 0
					node.scheduled_time = round(times_gpu[min_time_gpu[1]][-1][1] + node.delay, 2)
					node.scheduled_on_gpu = min_time_gpu[1]
					node.delay = 0
					node.set_gpu(True)
					# We check if the last run node is one of this node's dependencies, if we were going to wait to run this node waiting on that dependency and we are currently planning on running this node on a different CPU, it gets moved if possible
					if required_gpu != None and node.scheduled_time == required_gpu.scheduled_time + required_gpu.time() and node.scheduled_on_gpu != required_gpu.scheduled_on_gpu:
						if times_gpu[required_gpu.scheduled_on_gpu][-1][0] == required_gpu.scheduled_time:
							node.scheduled_on_gpu = required_gpu.scheduled_on_gpu
					# All dependencies are scheduled and we know where we will run this node, we can check if we need to add copy times or not
					node.check_copy()
					# Scheduled node list times are updated with the new node times
					times_gpu[node.scheduled_on_gpu].append((round(node.scheduled_time, 2), round(node.scheduled_time + node.time(), 2)))
					# Scheduled node list is updated with this node
					execution_elements_gpu[node.scheduled_on_gpu].append(node)
				else:
					# Behaves the same as above, but if running on the CPU is the better choice
					if node.scheduled_time - (times_cpu[min_time_cpu[1]][-1][1] if node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0) > 0:
						for i in range(1, len(self.available_cpu[self.current_frame])):
							attempt_early_start_node = self.available_cpu[self.current_frame][i]
							self.set_minimum_start_time(attempt_early_start_node)
							delay_early_node = attempt_early_start_node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] if attempt_early_start_node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0
							if times_cpu[min_time_cpu[1]][-1][1] + delay_early_node + attempt_early_start_node.time_cpu + attempt_early_start_node.copy_time < node.scheduled_time and attempt_early_start_node.time_cpu != -1:
								node.scheduled_time = -1
								node = attempt_early_start_node
								break
							else:
								attempt_early_start_node.scheduled_time = -1
					
					node.delay = node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] if node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0
					node.scheduled_time = round(times_cpu[min_time_cpu[1]][-1][1] + node.delay, 2)
					node.scheduled_on_cpu = min_time_cpu[1]
					node.delay = 0
					node.set_gpu(False)
					if required_cpu != None and node.scheduled_time == required_cpu.scheduled_time + required_cpu.time() and node.scheduled_on_cpu != required_cpu.scheduled_on_cpu:
						if times_cpu[required_cpu.scheduled_on_cpu][-1][0] == required_cpu.scheduled_time:
							node.scheduled_on_cpu = required_cpu.scheduled_on_cpu
					node.check_copy()
					times_cpu[node.scheduled_on_cpu].append((round(node.scheduled_time, 2), round(node.scheduled_time + node.time(), 2)))
					execution_elements_cpu[node.scheduled_on_cpu].append(node)
					
				# Updates the available nodes list (it says CPU but it is for all nodes, it is recycled)
				for next in node.required_by:
					append = True
					for check in next.requirements:
						if check.scheduled_time == -1:
							append = False
					if (append and next not in self.available_cpu[next.execution] and next.scheduled_time == -1):
						self.available_cpu[next.execution].append(next)
				self.available_cpu[self.current_frame].remove(node)
				node.delay = 0
			# Sorts the available nodes list by HEFT rank and checks if the current frame is completed
			self.available_cpu[self.current_frame].sort(key=lambda x: x.heft_rank, reverse=True)
			self.update_frame_counter()
			
		# Trasforms the separeted lists in a sigle one: [CPU_0 list[node_a, node_b], CPU_1 list[node_c, node_d], ..., GPU_0 list[node_e, node_f], ...]
		self.node_list = execution_elements_cpu + execution_elements_gpu
		for ce in self.node_list:
			ce.sort(key=lambda x: x.scheduled_time)
		times = times_cpu + times_gpu
		
		self.verify_scheduling(times)
		self.pipeline(execution_elements_cpu, execution_elements_gpu, times_cpu, times_gpu)
		
		self.node_list = execution_elements_cpu + execution_elements_gpu
		for ce in self.node_list:
			ce.sort(key=lambda x: x.scheduled_time)
		
		return self.verify_scheduling(times)
		
	# This function verifies if all dependencies in the scheduling are respected (done after the scheduling and pipelining)
	def verify_scheduling(self, times):
		self.verified = True # If there is no violation of dependencies in the scheduling this variable is set to True
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
						required_cpu_time = Decimal(required.scheduled_time + required.time_cpu).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)
						if (node_scheduled_time.compare(required_cpu_time) == -1):
							self.error_function(node, required, node_scheduled_time, self.node_list.index(computing_element))
		# If there are errors in the scheduling the program exits
		if (not self.verified):
			exit()
		self.max_time = round(max(times, key=lambda x: x[-1][1])[-1][1], 2)
		return self.max_time
	
	# Pipelines the scheduling, moving nodes earlier than previously scheduled
	# It basically just fills holes, no node is ever delayed
	def pipeline(self, execution_elements_cpu, execution_elements_gpu, times_cpu, times_gpu):
		# Calculates how much time was spent idle for all processors
		self.max_time_pipelining = round(max(times_cpu+times_gpu, key=lambda x: x[-1][1])[-1][1], 2)
		print('Makespan pre-pipelining: ' + str(self.max_time_pipelining))
		print('Idle times before pipelining: ')
		total_time = self.calculate_idle_time(times_cpu)
		total_time += self.calculate_idle_time(times_gpu, is_gpu=True)
		print('Total idle before: '+str(round(total_time, 2)))
		print('---')
		
		# Tries to move nodes as early as it can without delaying any other node (fills holes basically)
		# Super greedy approach
		# Keeps trying until there are no more changes (or MAXIMUM_PIPELINING_ATTEMPTS is reached)
		attempts = 0
		# If pipelining is enabled it will try to reorganize the task with some overlap between them (as long as frame times are respected)
		while (PIPELINING): # This should be a do-while
			moved_cpu = self.can_any_node_be_run_earlier(execution_elements_cpu, times_cpu)
			moved_gpu = self.can_any_node_be_run_earlier(execution_elements_gpu, times_gpu)
			attempts += 1
			if ((not moved_cpu and not moved_gpu) or attempts > MAXIMUM_PIPELINING_ATTEMPTS):
				break

		# Calculates idle times again to see what kind of improvement is obtained from pipelining
		total_time_old = total_time
		if (PIPELINING):
			print('Makespan post-pipelining: ' + str(round(max(times_cpu+times_gpu, key=lambda x: x[-1][1])[-1][1], 2)) + ' ('+str(round((self.max_time_pipelining/round(max(times_cpu+times_gpu, key=lambda x: x[-1][1])[-1][1], 2)-1)*100, 2))+'%)')
			self.max_time_pipelining = round(max(times_cpu+times_gpu, key=lambda x: x[-1][1])[-1][1], 2)
			print('Idle times after pipelining: ')
			total_time = self.calculate_idle_time(times_cpu)
			total_time += self.calculate_idle_time(times_gpu, is_gpu=True)
			print('Total idle after: '+str(round(total_time, 2)) + ' (-'+str(round(100.0 - total_time/total_time_old*100,2))+'%)')
			print('---')
		
	""" 
	error_from: which node has a dependency not met
	caused_by: which node is the dependency not met
	current_time: at what time was "error_from" scheduled
	execution_unit: on which execution unit was the error generated
	It also sets the "verified" value for the scheduling to false
	"""
	def error_function(self, error_from, caused_by, current_time, execution_unit):
		print('Current time: '+str(current_time))
		print('Executing on unit: '+str(execution_unit))
		print('Error from: \n'+str(error_from))
		print('Caused by: ')
		print(caused_by)
		self.verified = False
		
	def update_frame_counter(self):
		for node in self.node_list_cpu:
			if node.execution == self.current_frame and node.scheduled_time == -1:
				return
		for node in self.node_list_gpu:
			if node.execution == self.current_frame and node.scheduled_time == -1:
				return
		self.current_frame += 1
		
	
	def set_minimum_start_time(self, node):
		# If any node needed by this one has not been scheduled, this node cannot start
		for req in node.requirements:
			if req.scheduled_time == -1:
				return -1
		# the minimum time possible for this node is the maximum time (scheduled + runtime) obtained from all its dependencies, if this node has no requirements then it can start at time 0
		node.scheduled_time = round(max((req.scheduled_time + req.time() for req in node.requirements)) if node.requirements else 0, 2)

		# If requirements are met (or there are none), it is still needed to check if this task has been released (under the assumption that there is a new frame every MINIMUM_FRAME_DELAY, if we are before this deadline, the task is delayed
		if node.scheduled_time < node.execution * MINIMUM_FRAME_DELAY:
			node.scheduled_time = node.execution * MINIMUM_FRAME_DELAY
		
	def can_any_node_be_run_earlier(self, execution_elements, times):
		# If there is a hole in the times (this processor is not doing anything in this time)
		# We try fill this space
		moved_cpu = False
		moved_gpu = False
		for processor in execution_elements:
			for node in processor:
				# We search for the first gap that can fit our node (super greedy)
				# Also the first gap after this task has been released (meaining greater than execution*MINIMUM_FRAME_DELAY)
				# This gap also has to satisfy requirements timings (not only release timings)
				gap = self.get_first_time_gap(times, node.time(), node)
				# If the earliest gap doesn't exist or it is after our task then we continue to the next node
				if (not gap or gap[0] >= node.scheduled_time):
					continue
				# The node delay is reset because it will be recalculated
				node.delay = 0
				are_requirements_met = True
				# All requirements are checked again
				for req in node.requirements:
					if gap[0] < req.scheduled_time + req.time():
						# If a requirement is not met, an attempt for a delay is made
						attempt_delay = round(req.scheduled_time + req.time() - gap[0], 2)
						# We then check if this delay moves our task outside of the free slice of time
						# and also we check if this delay causes our task to move to a time after it is already scheduled
						if (gap[0] + attempt_delay > gap[1] or gap[0] + attempt_delay >= node.scheduled_time or gap[1] - attempt_delay - gap[0] < node.time()):
							are_requirements_met = False
							# If the requirements for this node and delay are not met, the delay is reset 
							node.delay = 0
						else:
							# If not the delay improves the situation while still staying inside the free time gap
							node.delay = max(node.delay, attempt_delay)
				if (are_requirements_met and gap[0] + node.delay < node.scheduled_time):
					# The new start time is locked in a variable
					new_start_time = gap[0] + node.delay
					# The node is removed from the processor where it is scheduled
					processor.remove(node)
					# The nodes is also removed from the timings table
					times[node.scheduled_on_gpu if node.is_gpu else node.scheduled_on_cpu].remove((round(node.scheduled_time, 2), round(node.scheduled_time + node.time(), 2)))
					# The node is added to the new processor
					execution_elements[gap[2]].append(node)
					# We add the node times and the delay to the timings list
					times[gap[2]].append((round(new_start_time, 2), round(new_start_time + node.time(), 2)))
					# The node is updated with the new start time
					node.scheduled_time = round(new_start_time, 2)
					# The node is also updated with the processor that is running it
					if (node.is_gpu):
						node.scheduled_on_gpu = gap[2]
						moved_gpu = True
					else:
						node.scheduled_on_cpu = gap[2]
						moved_cpu = True
					# As times have been removed and added, we need to sort the times again (they will be out of order due to remove in the middle and add at the end)
					for time in times:
						time.sort(key=lambda x: x[0])
		return moved_cpu if moved_cpu else moved_gpu
	
	# This function calculates idle times for each processing unit
	def calculate_idle_time(self, times, is_gpu=False):
		total_time = 0
		for processor_times in times:
			idle_time = 0
			for i in range(1, len(processor_times)):
				if (processor_times[i][0] - processor_times[i-1][1]) > 0.0:
					idle_time += processor_times[i][0] - processor_times[i-1][1]
			idle_time += (self.max_time_pipelining - processor_times[-1][1])
			total_time += idle_time
			if (not is_gpu):
				print('CPU '+str(times.index(processor_times))+': '+str(round(idle_time, 2))+' ('+str(round(idle_time/self.max_time_pipelining * 100, 2))+'%)')
			else:
				print('GPU '+str(times.index(processor_times))+': '+str(round(idle_time, 2))+' ('+str(round(idle_time/self.max_time_pipelining * 100, 2))+'%)')
		return round(total_time, 2)
				

	# Given the times for all the processors, searches for a gap at least the size of min_gap
	# Returns the start of the gap, the end of the gap and the processor in which it has been found
	# The release time of a task is also taken into account (if the frame to compute exists or not)
	def get_first_time_gap(self, times, min_gap, node):
		# max_time_pipelining is used to compute the improvement from pipelining, here we take advatange of it to know what is the maximum possible value for the earliest gap (so that our search can decrease it)
		earliest_gap = [self.max_time_pipelining+1, self.max_time_pipelining+2, 'e', 'n'] # The e is positioned so that we know if no gap was found, it is used in the return value
		for processor_times in times:
			for i in range(1, len(processor_times)):
				# If there is a gap before the current task, we can simply move it up
				if (processor_times[i][0] - processor_times[i-1][1]) > 0 and node.scheduled_time == processor_times[i][0] and (node.scheduled_on_gpu if node.is_gpu else node.scheduled_on_cpu) == times.index(processor_times) and processor_times[i-1][1] > node.execution*MINIMUM_FRAME_DELAY:
					if (processor_times[i-1][1] < earliest_gap[0]):
						earliest_gap = [processor_times[i-1][1], processor_times[i][1], times.index(processor_times), 'a']
						
				# If the gap is wide enough and it's not earlier than the task release time (which is the frame time)
				elif (processor_times[i][0] - processor_times[i-1][1] > min_gap and processor_times[i-1][1] >= node.execution*MINIMUM_FRAME_DELAY ):
					return_value = True
					# If moving the task to this gap would make it not respect a dependency then a delay is added
					dependency_delay = 0.0
					for req in node.requirements:
						if req.scheduled_time + req.time() > processor_times[i-1][1]:
							return_value = False
							# All dependencies are explored, the worst one is saved, so that we can attempt a delay
							if (dependency_delay < req.scheduled_time + req.time() - processor_times[i-1][1]):
								dependency_delay = req.scheduled_time + req.time() - processor_times[i-1][1]
					if (return_value):
						if (processor_times[i-1][1] < earliest_gap[0]):
							earliest_gap = [processor_times[i-1][1], processor_times[i][0], times.index(processor_times), 'b']
					else:
						# If a gap is not valid because of a dependency, a delay is added
						new_gap = processor_times[i][0] - (processor_times[i-1][1] + dependency_delay)
						if (processor_times[i-1][1] + dependency_delay < processor_times[i][0] and min_gap < new_gap):
							earliest_gap = [processor_times[i-1][1] + dependency_delay, processor_times[i][0], times.index(processor_times), 'c']
			
				# If the gap is wide enough and it's earlier than the task release time there is an attempt to delay it inside the gap
				# as long as the gap is big enough to host the task and await the start of the frame
				elif (processor_times[i][0] - processor_times[i-1][1] > min_gap and processor_times[i-1][1] < node.execution*MINIMUM_FRAME_DELAY):
					release_delay = node.execution*MINIMUM_FRAME_DELAY - processor_times[i-1][1]
					new_gap = processor_times[i][0] - (processor_times[i-1][1] + release_delay)
					if (processor_times[i-1][1] + release_delay < processor_times[i][0] and min_gap < new_gap):
						return_value = True
						for req in node.requirements:
							if req.scheduled_time > processor_times[i-1][1] + release_delay:
								return_value = False
						if (return_value):
							if (processor_times[i-1][1] < earliest_gap[0]):
								earliest_gap = [processor_times[i-1][1]+release_delay, processor_times[i][0], times.index(processor_times), 'd']
		if (earliest_gap[2] == 'e'):
			return []
		return earliest_gap
			
	# This function prints the Grantt graph of the scheduled nodes
	# Supports any number of processors (more or less, zooming in and out might be required)
	def create_bar_graph(self, labels):
		distance = 13 - PROCESSORS
	
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
		super_legend = mpatches.Patch(color='red', label='Super')
		name_legend = mpatches.Patch(color='white', label='initial_level_execution')
		
		frame_lines, = plt.Line2D([0], [0], linestyle='-.', color='gray', alpha=0.5, lw=1, label='Frame times'),
		finish_line, = plt.plot([self.max_time, self.max_time], [0,100], color='black', alpha=0.7, label='Finish time', linewidth=1)
		
		plt.legend(handles=[scale_legend, orb_legend, gauss_legend, grid_legend, fast_legend, dl_legend, super_legend, frame_lines, finish_line, name_legend], ncol=2)
		
		gnt.set_ylim(0, 50) 
		gnt.set_xlim(0, self.max_time*1.05) 
		gnt.set_xlabel('Milliseconds since start') 
		gnt.set_ylabel('Processor')

		gnt.set_yticks([distance+3+i*distance for i in range(0, len(labels))])
		gnt.set_yticklabels(labels) 
		
		gnt.yaxis.grid(True)
		
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
				if (node.execution >= 0):
					#name.append(node.name[0]+'_'+str(node.level)+'_'+str(node.execution)+(' ('+str(node.priority_point)+')' if not node.is_gpu else ''))
					name.append(node.name[0]+node.name[-1]+'_'+str(node.level)+'_'+str(node.execution))
					bar.append((node.scheduled_time, node.time()))
						
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
					elif ('super' in node.name):
						color.append('red')
					else:
						color.append('gray')
			bars.append(bar)
			names.append(name)
			colors.append(color)
		# Creates the broken bars, using the bars and color vectors
		for i in range(0, len(bars)):				
			gnt.broken_barh(bars[i], (distance*len(bars)-i*distance, 3), facecolors=tuple(colors[i]), edgecolor='white')
		# Writes the names of the nodes separetly
		for i in range(0, len(bars)):	
			for j in range(0, len(bars[i])):
				gnt.text(x=bars[i][j][0]+bars[i][j][1]/2, y=(4)+distance*len(bars)-i*distance+(j%3), s=names[i][j], ha='center', va='center', color='black',)
		# Adds the frame times line (skips frame zero, as that's the start of the graph)
		for i in range(1, FRAMES):
			plt.plot([i*MINIMUM_FRAME_DELAY,i*MINIMUM_FRAME_DELAY], [0,100], '-.', color='gray', alpha=0.5)
		
		mng = plt.get_current_fig_manager()
		mng.window.state('zoomed')
		plt.title('Makespan: '+str(self.max_time) + ', Average Frame Time: '+str(round(float(self.max_time/FRAMES), 2))+' (Max FPS: '+str(round(1000/(self.max_time/FRAMES),2))+', Camera: '+str(round(1000/MINIMUM_FRAME_DELAY, 2))+')')
		plt.show(block=True)
		
'''
create_dependencies_graph
EDD_auto
EDD_auto_rec
normalize_deadlines
calculate_priority_points
HEFT_auto
HEFT_auto_rec
import_graph
calculate_end_points
calculate_start_points
add_cross_dependencies_between_frames
main
'''		

# Completes the field "required_by" of each node, meaning that it compiles the son's list of each node
def create_dependencies_graph(all_nodes):
	for key in all_nodes:
		for req in all_nodes[key].requirements:
			if (all_nodes[key] not in req.required_by):
				req.required_by.append(all_nodes[key])

# Starting from an array of leaves (nodes that do not depend on any other) it calculates for each leaf in leaves the deadlines for EDD, using EDD_auto_rec
def EDD_auto(leaves):
	for leaf in leaves:
		EDD_auto_rec(leaf)

def EDD_auto_rec(leaf):
	# If this node is not required by anyone else
	if (not leaf.required_by):
		leaf.deadline = 0
	else:
		deadline_candidates = []
		# For each node (next) that requires this leaf we compute the EDD deadline, once we have explored all the nodes, we take the lowest
		for next in leaf.required_by:
			EDD_auto_rec(next)
			deadline_candidates.append(next.deadline - next.time())
		leaf.deadline = round(min(leaf.deadline, min(deadline_candidates)), 2)
		
# Normalizes priorities by summing the lowest value (as they are all negativs)
def normalize_deadlines(all_nodes_exec):
		# Normalizes from earliest deadline negative, to earliest deadline zero
		min_deadline = MINIMUM_FRAME_DELAY
		for node in all_nodes_exec:
			node.deadline += min_deadline
			
# Calculates priority points
def calculate_priority_points(all_nodes):
	for node in all_nodes:
		node.priority_point = round(float(node.deadline - ((PROCESSORS - 1)/PROCESSORS)*node.time()), 2)
		
# Calculates the HEFT rank for each node (behaves like EDD_auto)
def HEFT_auto(leaves):
	for leaf in leaves:
		HEFT_auto_rec(leaf)

def HEFT_auto_rec(leaf):
	if (not leaf.required_by):
		leaf.heft_rank = leaf.avg_time()
	else:
		rank_candidates = []
		for next in leaf.required_by:
			HEFT_auto_rec(next)
			rank_candidates.append(next.copy_time + next.heft_rank)
		leaf.heft_rank = round(max(leaf.heft_rank, leaf.avg_time() + max(rank_candidates)), 2)

# Liv name(0) - Levels(1) - CPU times(2+levels) - GPU times(2+levels*2) - Copy times(2+levels*3) - (Dependencies, Level), ...
# Imports the DAG from a file with this structure ^ (level for the dependencies is an offset, eg.: i'm node level X and i depend on X + Level from Dependency)
# Also applies the mapping from the global variable "nodes_is_gpu"
def import_graph():
	# This dict will contain all nodes with the following key = ['node name', level, frame number]
	node_list = {}
	# The data is read and put into this variable split into rows to be able to run through it multiple times
	file_data = None
	with open(GRAPH_FILE) as graph:
		file_data = graph.read()
	file_data = file_data.split('\n')
	node_max_level = {}
	# For each frame we add all the nodes of the file to the node list
	for frame in range(0, FRAMES):
		dependencies = {}
		node_max_level = {}
		for row in file_data:
			line = row.split(',')
			# All the node data os acquired
			node_name = line[0]
			node_levels = int(line[1])
			node_cpu_times = [float(line[2+i]) for i in range(0, node_levels)]
			node_gpu_times = [float(line[2+node_levels+i]) for i in range(0, node_levels)]
			node_copy_times = [float(line[2+node_levels*2+i]) for i in range(0, node_levels)]
			node_max_level[node_name] = node_levels - 1
			# Dependencies are coded as a tuple, so after copy times we have ('dependency name', level) repeated for as many dependencies
			for i in range (2+node_levels*3, len(line), 2):
				if node_name not in dependencies.keys():
					dependencies[node_name] = [(line[i], int(line[i+1]))]
				else:
					dependencies[node_name].append((line[i], int(line[i+1])))
			# We create and add all the nodes to the dict
			for level in range(0, node_levels):
				node_list[node_name, level, frame] = Node(node_name, level, node_cpu_times[level], [], frame, node_gpu_times[level], node_copy_times[level])
		# We use the dict just created and the dependencies to update all the nodes with their respective dependencies
		# This bit of code is fairly complicated (aka garbage)
		# We run through all the nodes we just created, but we do not want all of them, just the current frame
		# Instead of just using key, we use key[0] for the node name and key[1] for the level, and frame for the frame (instead of key[2])
		# Now we need all the dependencies for that node, which are store in dependencies['node name'], and the node name is key_node[0]
		# At this point we want that the offset of the level plus the level of the node is greater than zero and that the sum of the two is not greater than the maximum level of the dependency (which might happend eg. if a node level 3 is dependent on a node with no levels)
		# Then we add the dependency to the node key_node[0], key_node[1], frame (name, level, frame), taking the node from the node's list
		for key_node in node_list:
			if key_node[0] in dependencies:
				for key_dep in dependencies[key_node[0]]:
					if (key_node[1]+key_dep[1] >= 0 and key_node[1]+key_dep[1] <= node_max_level[key_dep[0]] and node_list[key_dep[0], key_node[1]+key_dep[1], frame] not in node_list[key_node[0], key_node[1], frame].requirements):
						node_list[key_node[0], key_node[1], frame].requirements.append(node_list[key_dep[0], key_node[1]+key_dep[1], frame])
					elif (key_node[1]+key_dep[1] >= 0 and key_node[1]+key_dep[1] > node_max_level[key_dep[0]] and node_list[key_dep[0], node_max_level[key_dep[0]], frame] not in node_list[key_node[0], key_node[1], frame].requirements):
						node_list[key_node[0], key_node[1], frame].requirements.append(node_list[key_dep[0], node_max_level[key_dep[0]], frame])
	# Following the VisionWork policy, we always use the fastest implementation for the mapping (even if it creates imbalance between processors)
	for key in node_list:
		if (node_list[key].time_gpu != -1 and node_list[key].time_cpu != -1):
			if (node_list[key].time_gpu < node_list[key].time_cpu):
				node_list[key].set_gpu(True)
		# If there is only a GPU time, then it is a GPU node
		elif (node_list[key].time_gpu != -1 and node_list[key].time_cpu == -1):
			node_list[key].set_gpu(True)
		# If there is only a CPU time, then it is not a GPU node
		elif (node_list[key].time_gpu == -1 and node_list[key].time_cpu != -1):	
			node_list[key].set_gpu(False)
	return node_list
	
# Calculates end points, which are the nodes that are not required by any other node
def calculate_end_points(node_list):
	end_points = []
	for key_candidates in node_list:
		candidate = node_list[key_candidates]
		valid = True
		# For every node it checks if it is required by any other
		for key_comparison in node_list:
			if candidate in node_list[key_comparison].requirements:
				# If so, this node is not a valid candidate
				valid = False
		if (valid and candidate not in end_points):
			end_points.append(candidate)
	return end_points
			
# Calculates start points, which are the nodes that do not require any other node to run
def calculate_start_points(node_list):
	start_points = []
	for key in node_list:
		if not node_list[key].requirements and node_list[key] not in start_points:
			start_points.append(node_list[key])
	return start_points
	
# Adds dependencies to start nodes so that no frame can start before the earlier frame
def add_cross_dependencies_between_frames(node_list, start_points, end_points):
	start_points.sort(key = lambda x: x.execution)
	end_points.sort(key = lambda x: x.execution)
	for frame in range(1, FRAMES):
		for i in range(1, len(start_points)):
			if start_points[i-1] not in start_points[i].requirements:
				start_points[i].requirements.append(start_points[i-1])
		
def main():
	node_list = import_graph()
	if (not HEFT):
		for key in node_list:
			node_list[key].check_copy()

	end_points = calculate_end_points(node_list)
	start_points = calculate_start_points(node_list)
	add_cross_dependencies_between_frames(node_list, start_points, end_points)
	create_dependencies_graph(node_list)

	if (HEFT):
		all_nodes = []
	all_nodes_cpu = [] # CPU queue
	all_nodes_gpu = [] # GPU queue
	
	EDD_auto(start_points)
	HEFT_auto(start_points)
		
	for frame in range(0, FRAMES):
		all_nodes_exec = []
		for key in node_list:
			if (node_list[key].execution == frame):
				# These are the nodes for the execution under exam
				all_nodes_exec.append(node_list[key[0], key[1], frame])
				# These are ALL the nodes used for the scheduling
				# They are added to the respective queue if they are a GPU node or not
				if (not HEFT): # As HEFT decides the processor to use too, unlike EDD and G-FL, the queues are created differently
					if node_list[key[0], key[1], frame].is_gpu:
						all_nodes_gpu.append(node_list[key[0], key[1], frame])
					else:
						all_nodes_cpu.append(node_list[key[0], key[1], frame])
				else:
					all_nodes.append(node_list[key[0], key[1], frame])
		normalize_deadlines(all_nodes_exec)
		calculate_priority_points(all_nodes_exec)
	#print(node_list['scale2', 1, 0])
		
	# All the nodes are ordered by their priority point (as G-FL demands) or deadline
	if (DEADLINE):
		all_nodes_cpu.sort(key=lambda x: x.deadline, reverse=False)
		all_nodes_gpu.sort(key=lambda x: x.deadline, reverse=False)
	elif (GFL):
		all_nodes_cpu.sort(key=lambda x: x.priority_point, reverse=False)
		all_nodes_gpu.sort(key=lambda x: x.deadline, reverse=False)
	else:
		all_nodes.sort(key=lambda x: x.heft_rank, reverse=True)	
	
	if (HEFT):
		all_nodes.sort(key=lambda x: x.execution, reverse=False)
		scheduler = Schedule(all_nodes, [], n_cpu, n_gpu)
		scheduler.set_starting_points(start_points)
		max_time = scheduler.create_schedule_HEFT()
	else:
		# Sort all nodes using a stable algorithm by execution (if they have the same priority the earlier executions go first)
		all_nodes_cpu.sort(key=lambda x: x.execution, reverse=False)
		all_nodes_gpu.sort(key=lambda x: x.execution, reverse=False)
		scheduler = Schedule(all_nodes_cpu, all_nodes_gpu, n_cpu, n_gpu)
		scheduler.set_starting_points(start_points)
		max_time = scheduler.create_schedule()
	
	if (DEADLINE):
		print('Makespan EDD: ', end=' ')
	elif (GFL):
		print('Makespan G-FL: ', end=' ')
	else:
		print('Makespan HEFT: ', end=' ')
	print(str(max_time) + ' ('+str(round(max_time/FRAMES, 2))+')')

	GPU_labels = ['GPU '+str(i) for i in range(n_gpu-1, -1, -1)]
	CPU_labels = ['CPU '+str(i) for i in range(n_cpu-1, -1, -1)]
	
	if (not HEFT):
		all_nodes = all_nodes_cpu + all_nodes_gpu
	
	# Here at the end are calculated for all tasks average lateness, max lateness, and also exact frame time
	max_lateness = 0
	avg_lateness = 0
	for node in all_nodes:
		lateness = node.scheduled_time + node.time() - (node.deadline + MINIMUM_FRAME_DELAY * node.execution)
		avg_lateness += lateness
		max_lateness = max(max_lateness, lateness)
	print('Average lateness: '+str(round(float(avg_lateness/len(all_nodes)), 2)))
	print('Maximum lateness: '+str(round(max_lateness, 2)))
	print('---')
	
	# To calculate frame times, for each frame, we take the first task that starts execution and the last to complete and then the difference is taken as frame time
	min_time = [10000 for i in range(0, FRAMES)]
	max_time = [0 for i in range(0, FRAMES)]
	for node in all_nodes:
		for execution in range(0, FRAMES):
			if node.execution == execution:
				min_time[execution] = round(min(min_time[execution], node.scheduled_time), 2)
				max_time[execution] = round(max(max_time[execution], node.scheduled_time+node.time()), 2)
	print('Exact frame times: ')
	sum_frame_times = 0
	for i in range(0, FRAMES):
		sum_frame_times += max_time[i]-min_time[i]
		print(round(max_time[i]-min_time[i], 2), end=' ')	
	print('\nAverage frame time: '+str(round(sum_frame_times/FRAMES, 2)))
	
	if (SHOW_GRAPH):
		scheduler.create_bar_graph(labels=GPU_labels + CPU_labels)
	
if __name__ == '__main__':
	if (len(sys.argv) == 5):
		try:
			GRAPH_FILE = sys.argv[1]
			if int(sys.argv[2]) == 1:
				DEADLINE = True
			elif int(sys.argv[2]) == 0:
				GFL = True
			else:
				HEFT = True
			FRAMES = int(sys.argv[3])
			PIPELINING = bool(int(sys.argv[4]))
		except:
			print('Usage: \n sim.py FILENAME[csv] SCHEDULING(0,1,2) FRAMES(n) PIPELINE(0,1)\nExample: open graph.csv, simulate three frames and use EDD with Pipeline\n\tsim.py graph.csv 1 3 1', file=sys.stderr)
			exit()
	elif (len(sys.argv) == 4):
		try:
			if int(sys.argv[1]) == 1:
				DEADLINE = True
			elif int(sys.argv[1]) == 0:
				GFL = True
			else:
				HEFT = True
			FRAMES = int(sys.argv[2])
			PIPELINING = bool(int(sys.argv[3]))
		except:
			print('Usage: \n sim.py FILENAME[csv] SCHEDULING(0,1,2) FRAMES(n) PIPELINE(0,1)\nExample: open graph_file.csv, simulate three frames and use EDD with pipeline\n\tsim.py 1 3 1', file=sys.stderr)
			exit()
	elif (len(sys.argv) == 3):
		try:
			if int(sys.argv[1]) == 1:
				DEADLINE = True
			elif int(sys.argv[1]) == 0:
				GFL = True
			else:
				HEFT = True
			FRAMES = int(sys.argv[2])
		except:
			print('Usage: \n sim.py FILENAME[csv] SCHEDULING(0,1,2) FRAMES(n) PIPELINE(0,1)\nExample: open graph_file.csv, simulate three frames and use EDD without pipeline\n\tsim.py 1 3', file=sys.stderr)
			exit()
	elif (len(sys.argv) == 2):
		if (sys.argv[1][0]) == '?':
			GFL = True
			print('Usage: \n sim.py FILENAME[csv] DEADLINE(0,1) FRAMES(n)\nExample: open timings.csv, simulate three frames and use EDD without Pipeline\n\tsim.py graph_file.csv 1 3 0\nRunning with default settings: \n\tFile ' +str(GRAPH_FILE) + ' Frames: '+str(FRAMES) + ' Scheduling: GFL = '+str(GFL) + ' Pipeline: '+str(PIPELINING))
	else:
		GFL = True
	main()