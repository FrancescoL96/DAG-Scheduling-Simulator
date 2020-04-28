import sys
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from decimal import Decimal, ROUND_HALF_EVEN

FILENAME = 'timings.csv'
n_cpu = 2
n_gpu = 1
PROCESSORS = float(n_cpu+n_gpu)	# Used in the formula to calculate priority points for G-FL
DEADLINE = False # If set to true it will schedule using EDD, otherwise G-FL (CPU only) !!! (can be overridden with program parameters)
PIPELINING = True # If set to true it will enable pipelining
FRAMES = 3 # Default value is 3 !!! (can be overridden with program parameters)
MAXIMUM_PIPELINING_ATTEMPTS = 100
MINIMUM_FRAME_DELAY = 33 # The minimum time between frames, 16ms is 60 fps, 33ms is 30 fps, 22ms is 45 fps
# If set to True, those group of nodes will be scheduled on the GPU
nodes_is_gpu = {'scale': True, 'fast': True, 'gauss': False, 'orb': False}
# If set to True it will display a graph with the scheduling
SHOW_GRAPH = True
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
		self.is_gpu = False					# If this value is True then the GPU time is used
		self.copy_time = copy_time			# Copy time between processors memories to start this node
		# This values are set automatically
		self.delay = 0						# How long does this node need to wait before it can start
		self.scheduled_time = -1			# At what time is this node scheduled
		self.required_by = []				# Which nodes require this be completed (sed to calculate deadlines)
		self.deadline = 1.0					# EDD deadline, calculated using recursive formula: EDD(n) = [for each successor s] - min(EDD(s) - WCET - Cpy)
		self.priority_point = 0.0			# Proprity value for G-FL scheduling, calculated for each node using: Y = Deadline - ((Procs - 1)/Procs) * WCET - min [for each job j] (Deadline_j - ((Procs - 1)/Procs) * WCET_j)
		# This two variables are used to know on which processor the node is scheduled
		self.scheduled_on_cpu = -1
		self.scheduled_on_gpu = -1
	
	# If this function is called, is_gpu is set to True, allowing the code to use time_gpu for execution time intead of time (which should be used for cpu time)
	# It has a parameter in case it is needed to remove a node from the GPU, otherwise just sets to True
	def set_gpu(self, value=True):
		if (self.time_gpu != -1):
			self.is_gpu = value
			
	# If any data of a required node is on a different processor we need a copy
	def check_copy(self):
		# The copy time is used in the code by accessing this variable, at this point it is still unknown wheter we will use this value or not, so a copy is made to restore it in case it's needed
		keep_copy = self.copy_time
		self.copy_time = 0
		for req in self.requirements:
			if req.is_gpu != self.is_gpu:
				self.copy_time = keep_copy
		
	
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
		return '---------------------------\n' + self.name + '_' + str(self.level) + '_' + str(self.execution) + str(' GPU' if self.is_gpu else ' CPU') + '\nDelay: ' + str(round(self.delay, 2)) + (' Scheduled on CPU: '+str(self.scheduled_on_cpu) if self.scheduled_on_cpu != -1 else ' Scheduled on GPU: '+str(self.scheduled_on_gpu)) + '\nRequires: ' + requirements_str + later + '\n---------------------------'
	
# This class creates and verifies the scheduling, prints the scheduling only if it is verified
class Schedule:
	# Takes as parameter a list of all nodes ordered by priority and execution, separeted between CPU and GPU nodes
	def __init__(self, node_list_cpu, node_list_gpu, n_cpu, n_gpu):
		self.node_list_cpu = node_list_cpu
		self.node_list_gpu = node_list_gpu
		self.verified = False
		self.max_time = 0
		self.n_cpu = n_cpu
		self.n_gpu = n_gpu
		# Ready queues for CPU and GPU tasks of the current frame
		self.available_cpu = [[] for i in range(0, FRAMES)]
		self.available_gpu = [[] for i in range(0, FRAMES)]
		# Always starts from frame zero
		self.current_frame = 0
	
	# Creates the scheduling
	def create_schedule(self):
		# For each CPU and GPU creates a list of nodes
		execution_elements_cpu = [[] for i in range(0, self.n_cpu)]
		execution_elements_gpu = [[] for i in range(0, self.n_gpu)]
		# Creates the same lists to save times (at what time is the computation) for each processor
		times_cpu = [[(0.0,0.0)] for i in range(0, self.n_cpu)]
		times_gpu = [[(0.0,0.0)] for i in range(0, self.n_gpu)]
		
		# We add the highest priority node to each queue if they have 0 requirements
		if not self.node_list_cpu[0].requirements:
			self.available_cpu[self.current_frame].append(self.node_list_cpu[0])
			
		if not self.node_list_gpu[0].requirements:
			self.available_gpu[self.current_frame].append(self.node_list_gpu[0])

		# Run while there are nodes that can run
		# This while takes advantage of lazy expression evaluation to avoid index out of bounds
		while self.current_frame < FRAMES and (self.available_gpu[self.current_frame] or self.available_cpu[self.current_frame]):
			# The processor with the earliest time goes first			
			if ((min(times_gpu, key=lambda x: x[-1][1])[-1][1]+(self.available_gpu[self.current_frame][0].time_gpu if self.available_gpu[self.current_frame] else 0) <= min(times_cpu, key=lambda x: x[-1][1])[-1][1]+(self.available_cpu[self.current_frame][0].time if self.available_cpu[self.current_frame] else 0) and self.available_gpu[self.current_frame]) or not self.available_cpu[self.current_frame]):
				# If there are nodes available for a GPU
				if (self.available_gpu[self.current_frame]):
					node = self.available_gpu[self.current_frame][0]
					# Check whats the minimum starting time for this node
					self.set_minimum_start_time(node)
					# It requirements are not met the minimum start time is not set and we go on to a CPU node below
					# We take the last run required node that was run on the same processor
					required = None
					if node.requirements:
						required = node.requirements[0]
						for req in node.requirements:
							if req.is_gpu and node.is_gpu:
								if req.scheduled_time > required.scheduled_time:
									required = req
					if (node.scheduled_time != -1):
						min_time_gpu = [10000, -1]
						# If the requirements is the last run node, we schedule the current node on the same processor (instead of creating a delay)
						if required != None and required.is_gpu and execution_elements_gpu[required.scheduled_on_gpu][-1] == required:
							min_time_gpu[0] = [times_gpu[required.scheduled_on_gpu]][-1][1]
							min_time_gpu[1] = required.scheduled_on_gpu
						# Adds this node to the GPU with the smallest current time
						else: 
							min_time_gpu = [10000, -1]
							for gpu in times_gpu:
								if min_time_gpu[0] > gpu[-1][1]:
									min_time_gpu[0] = gpu[-1][1]
									min_time_gpu[1] = times_gpu.index(gpu)
					
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
						min_time_cpu = [10000, -1]
						if required != None and not required.is_gpu and execution_elements_cpu[required.scheduled_on_cpu][-1] == required:
							min_time_cpu[0] = [times_cpu[required.scheduled_on_cpu]][-1][1]
							min_time_cpu[1] = required.scheduled_on_cpu
						else:
							for cpu in times_cpu:
								if min_time_cpu[0] > cpu[-1][1]:
									min_time_cpu[0] = cpu[-1][1]
									min_time_cpu[1] = times_cpu.index(cpu)
						execution_elements_cpu[min_time_cpu[1]].append(node)
						node.delay = node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] if node.scheduled_time - times_cpu[min_time_cpu[1]][-1][1] > 0 else 0
						node.scheduled_time = round(times_cpu[min_time_cpu[1]][-1][1] + node.delay, 2)
						times_cpu[min_time_cpu[1]].append((round(node.scheduled_time, 2), round(node.scheduled_time + node.time + node.copy_time, 2)))
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
			else:
				self.available_cpu[self.current_frame].sort(key=lambda x: x.priority_point, reverse=False)
				self.available_gpu[self.current_frame].sort(key=lambda x: x.deadline, reverse=False)
			
			self.update_frame_counter()
		
		# Calculates how much time was spent idle for all processors
		print('Idle times before pipelining: ')
		total_time = self.calculate_idle_time(times_cpu)
		total_time += self.calculate_idle_time(times_gpu, is_gpu=True)
		print('Total idle before: '+str(round(total_time, 2)))
		
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
		if (PIPELINING):
			print('Idle times after pipelining: ')
			total_time = self.calculate_idle_time(times_cpu)
			total_time += self.calculate_idle_time(times_gpu, is_gpu=True)
			print('Total idle after: '+str(round(total_time, 2)))
		
		# Trasforms the separeted lists in a sigle one: [CPU_0 list[node_a, node_b], CPU_1 list[node_c, node_d], ..., GPU_0 list[node_e, node_f], ...]
		self.node_list = execution_elements_cpu + execution_elements_gpu
		for ce in self.node_list:
			ce.sort(key=lambda x: x.scheduled_time)
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
		node.scheduled_time = round(max((req.scheduled_time + req.copy_time + (req.time_gpu if req.is_gpu else req.time) for req in node.requirements)) if node.requirements else 0, 2)
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
				gap = self.get_first_time_gap(times, node.copy_time + (node.time_gpu if node.is_gpu else node.time), node.execution, node)
				# If the earliest gap doesn't exist or it is after our task then we continue to the next node
				if (not gap or gap[0] > node.scheduled_time):
					continue
				are_requirements_met = True
				# All requirements are checked again
				for req in node.requirements:
					if gap[0] < req.scheduled_time + req.copy_time + (req.time_gpu if req.is_gpu else req.time):
						# If a requirement is not met, an attempt for a delay is made
						attempt_delay = req.scheduled_time + req.copy_time + (req.time_gpu if req.is_gpu else req.time) - gap[0]
						# We then check if this delay moves our task outside of the free slice of time
						# and also we check if this delay causes our task to move to a time after it is already scheduled
						if (gap[0] + attempt_delay > gap[1] or gap[0] + attempt_delay + node.copy_time + (node.time_gpu if node.is_gpu else node.time) > node.scheduled_time or gap[0] + attempt_delay - gap[1] < node.copy_time + (node.time_gpu if node.is_gpu else node.time)):
							are_requirements_met = False
						else:
							# If not the delay improves the situation while still staying inside the free time gap
							node.delay = max(node.delay, attempt_delay)
				if (are_requirements_met):
					# The new start time is locked in a variable
					new_start_time = gap[0] + node.delay
					# After being used the delay is reset
					node.delay = 0
					# The node is removed from the processor where it is scheduled
					processor.remove(node)
					# The nodes is also removed from the timings table
					times[node.scheduled_on_gpu if node.is_gpu else node.scheduled_on_cpu].remove((round(node.scheduled_time, 2), round(node.scheduled_time + node.copy_time + (node.time_gpu if node.is_gpu else node.time), 2)))
					# The node is added to the new processor
					execution_elements[gap[2]].append(node)
					# We add the node times and the delay to the timings list
					times[gap[2]].append((round(new_start_time, 2), round(new_start_time + node.copy_time + (node.time_gpu if node.is_gpu else node.time), 2)))
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
				if (processor_times[i][0] - processor_times[i-1][1]) > 0:
					idle_time += processor_times[i][0] - processor_times[i-1][1]
			total_time += idle_time
			if (not is_gpu):
				print('CPU '+str(times.index(processor_times))+': '+str(round(idle_time, 2))+' ('+str(round(idle_time/processor_times[-1][1], 3))+'%)')
			else:
				print('GPU '+str(times.index(processor_times))+': '+str(round(idle_time, 2))+' ('+str(round(idle_time/processor_times[-1][1], 3))+'%)')
		return round(total_time, 2)
				

	# Given the times for all the processors, searches for a gap at least the size of min_gap
	# Returns the start of the gap, the end of the gap, the duration and the processor in which it has been found
	# The release time of a task is also taken into account (if the frame to compute exists or not)
	def get_first_time_gap(self, times, min_gap, frame_number, node):
		for processor_times in times:
			for i in range(1, len(processor_times)):
				# If there is a gap before the current task, we can simply move it up
				if (processor_times[i][0] - processor_times[i-1][1]) > 0 and node.scheduled_time == processor_times[i][0] and (node.scheduled_on_gpu if node.is_gpu else node.scheduled_on_cpu) == times.index(processor_times) and processor_times[i-1][1] > frame_number*MINIMUM_FRAME_DELAY:
					return [processor_times[i-1][1], processor_times[i][1], times.index(processor_times)]
				# If the gap is wide enough and it's not earlier than the task release time (which is the frame time)
				if (processor_times[i][0] - processor_times[i-1][1] > min_gap and processor_times[i-1][1] > frame_number*MINIMUM_FRAME_DELAY ):
					return_value = True
					# If moving the task to this gap would make it not respect a dependency then it skips this gap for this task
					for req in node.requirements:
						if req.scheduled_time > processor_times[i-1][1]:
							return_value = False
					if (return_value):
						return [processor_times[i-1][1], processor_times[i][0], times.index(processor_times)]
				# If the gap is wide enough and it's earlier than the task release time there is an attempt to delay it inside the gap
				# as long as the gap is big enough to host the task and await the start of the frame
				elif (processor_times[i][0] - processor_times[i-1][1] > min_gap and processor_times[i-1][1] < frame_number*MINIMUM_FRAME_DELAY):
					release_delay = frame_number*MINIMUM_FRAME_DELAY - processor_times[i-1][1]
					new_gap = processor_times[i][0] - (processor_times[i-1][1] + release_delay)
					if (processor_times[i-1][1] + release_delay < processor_times[i][0] and min_gap < new_gap):	
						return_value = True
						for req in node.requirements:
							if req.scheduled_time > processor_times[i-1][1] + release_delay:
								return_value = False
						if (return_value):
							return [processor_times[i-1][1]+release_delay, processor_times[i][0], times.index(processor_times)]
		return []
			
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
				if (node.execution >= 0):
					#name.append(node.name[0]+'_'+str(node.level)+'_'+str(node.execution)+(' ('+str(node.priority_point)+')' if not node.is_gpu else ''))
					name.append(node.name[0]+'_'+str(node.level)+'_'+str(node.execution))
					if (node.is_gpu): 
						bar.append((node.scheduled_time, node.time_gpu + node.copy_time))
					else:
						bar.append((node.scheduled_time, node.time + node.copy_time))
						
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
		for i in range(0, len(bars)):	
			for j in range(0, len(bars[i])):	
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
		leaf.deadline = round(min(leaf.deadline, min(deadline_candidates)), 2)
		
# Normalizes priorities by summing the lowest value (as they are all negativs)
def normalize_deadlines(all_nodes_exec):
		# Normalizes from earliest deadline negative, to earliest deadline zero
		#min_deadline =  - 1.0 * min(node.deadline for node in all_nodes_exec)
		min_deadline = MINIMUM_FRAME_DELAY
		for node in all_nodes_exec:
			node.deadline += min_deadline
		
# calculates priority points
def calculate_priority_points(all_nodes):
	for node in all_nodes:
		if (node.is_gpu == True):
			node.priority_point = round(float(node.deadline - ((PROCESSORS - 1)/PROCESSORS)*node.time_gpu), 2)
		else:
			node.priority_point = round(float(node.deadline - ((PROCESSORS - 1)/PROCESSORS)*node.time), 2)

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
	scale[0][0].set_gpu(nodes_is_gpu['scale'])
	
	for execution in range(0, FRAMES):
		# First node of each execution has special treatment due to special constraints
		if execution > 0:
			scale[execution].append(Node('scale', 0, timings['scale_cpu'][0], [scale[execution-1][0]], execution, timings['scale_gpu'][0]))
			scale[execution][0].set_gpu(nodes_is_gpu['scale'])
		# As scale_0 are set outside, we need to skip them here, where we initialize all the other nodes
		for level in range(1, levels):
			scale[execution].append(Node('scale', level, timings['scale_cpu'][level], [scale[execution][level-1]], execution, timings['scale_gpu'][level]))
			scale[execution][level].set_gpu(nodes_is_gpu['scale'])
			
		for level in range(0, levels):
			fast[execution].append(Node('fast', level, timings['fast_cpu'][level], [scale[execution][level]], execution, timings['fast_gpu'][level], copy_time=timings['orb_cpu_gpu_copy'][level]))
			fast[execution][level].set_gpu(nodes_is_gpu['fast'])
			
			grid[execution].append(Node('grid', level, timings['grid_cpu'][level], [fast[execution][level]], execution, copy_time=timings['grid_cpu_gpu_copy'][level]))
			
			gauss[execution].append(Node('gauss', level, timings['gauss_cpu'][level], [scale[execution][level]], execution, timings['gauss_gpu'][level]))
			gauss[execution][level].set_gpu(nodes_is_gpu['gauss'])
			
			if (level == 7 and execution > 0):
				# A Frame cannot complete computing before an earlier one
				orb[execution].append(Node('orb', level, timings['orb_cpu'][level], [gauss[execution][level], grid[execution][level], orb[execution-1][7]], execution, timings['orb_gpu'][level], copy_time=timings['orb_cpu_gpu_copy'][level]))
			else:
				orb[execution].append(Node('orb', level, timings['orb_cpu'][level], [gauss[execution][level], grid[execution][level]], execution, timings['orb_gpu'][level], copy_time=timings['orb_cpu_gpu_copy'][level]))
			orb[execution][level].set_gpu(nodes_is_gpu['orb'])
			
		deep_learning[execution].append(Node('deep_learning', 0, timings['deep_learning_cpu'][0], [scale[execution][0]], execution, timings['deep_learning_gpu'][0]))
		deep_learning[execution][0].set_gpu()
		# Once all the nodes have been created and the processor is set (with is_gpu()), the copy times are updated
		for node in gauss[execution]+fast[execution]+orb[execution]:
			node.check_copy()
	
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
	print(('Makespan EDD: ' if DEADLINE else 'Makespan G-FL: ')+str(max_time))

	GPU_labels = ['GPU '+str(i) for i in range(0, n_gpu)]
	CPU_labels = ['CPU '+str(i) for i in range(0, n_cpu)]
	
	# Here at the end are calculated for all tasks average lateness, max lateness, and also exact frame time
	max_lateness = 0
	avg_lateness = 0
	for node in all_nodes_cpu+all_nodes_gpu:
		lateness = node.scheduled_time + (node.time_gpu if node.is_gpu else node.time) + node.copy_time - (node.deadline + MINIMUM_FRAME_DELAY * node.execution)
		avg_lateness += lateness
		max_lateness = max(max_lateness, lateness)
	
	print('Average lateness: '+str(round(float(avg_lateness/len(all_nodes_cpu+all_nodes_gpu)), 2)))
	print('Maximum lateness: '+str(round(max_lateness, 2)))
	
	# To calculate frame times, for each frame, we take the first task that starts execution and the last to complete and then the difference is taken as frame time
	min_time = [10000 for i in range(0, FRAMES)]
	max_time = [0 for i in range(0, FRAMES)]
	for node in all_nodes_cpu+all_nodes_gpu:
		for execution in range(0, FRAMES):
			if node.execution == execution:
				min_time[execution] = round(min(min_time[execution], node.scheduled_time), 2)
				max_time[execution] = round(max(max_time[execution], node.scheduled_time+node.copy_time+(node.time_gpu if node.is_gpu else node.time)), 2)
	print('Exact frame times: ')
	sum_frame_times = 0
	for i in range(0, FRAMES):
		sum_frame_times += max_time[i]-min_time[i]
		print(round(max_time[i]-min_time[i], 2), end=' ')	
	print('\nAverage frame time: '+str(round(sum_frame_times/FRAMES, 2)))
		
	if (SHOW_GRAPH):
		GFL.create_bar_graph(labels=GPU_labels + CPU_labels)

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