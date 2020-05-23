import random

# Minimum and Maximum run time for a Node that is CPU only or double implementation
MINIMUM_CPU_TIME = [10.0, 10.0, 50.0]
MAXIMUM_CPU_TIME = [90.0, 50.0, 90.0]
# Minimum and Maximum run time for a Node that is GPU only
MINIMUM_GPU_TIME = [10, 10, 50]
MAXIMUM_GPU_TIME = [90, 50, 90]
# If a node has a double implementation the GPU time is based on a speedup range
MINIMUM_GPU_SPEEDUP = [1, 4, 0.5]
MAXIMUM_GPU_SPEEDUP = [4, 9, 2]

PROBABILITY_SINGLE = [0.2, 0.5, 0.8]	# Probability for a node to be a single implementation
PROBABILITY_CPU_GPU = [0.2, 0.5, 0.8]	# Probability for a single implementation node to be CPU

# As all the timing and probabilities are sets, we need a way to identify which to use
SET = 0

# Use a sequence of letters as node name
next_node_name = 'b'
# This is the string that will be written to the file that the simulator will read, the node 'a' is the first node with no requirements, that every other node will depend on
graph = 'a,1,0.0001,-1,0'
last_generated_node_tree = 'a'

first_node_last_row = 'a'
last_generated_node_linear = 'a'

def node_generator():
	global next_node_name
	node_name = next_node_name
	res = node_name+','
	next_char = ''
	# If the last character of the node name is not 'z' we update it to the next letter, otherwise we can get the next one (it is the last letter of the alphabet)
	if next_node_name[-1] != 'z':
		next_char += chr(ord(next_node_name[-1])+1)
		next_node_name = next_node_name[0 : len(next_node_name)-1]
		next_node_name += next_char
	else:
		# If it is the last letter we append an 'a' to the end and start over
		next_node_name += 'a'
	# The node only has one level (this is a remnant of the structure required for Orb-SLAM)	
	res += '1,'
	time_cpu = -1
	time_gpu = -1
	copy_time = 0
	# Double or single implementation?
	if round(random.randrange(0, 100) * 0.01, 2) < PROBABILITY_SINGLE[SET]:
		# It is single
		# Is it CPU or GPU?
		if round(random.randrange(0, 100) * 0.01, 2) < PROBABILITY_CPU_GPU[SET]:
			# It is CPU only
			time_cpu = random.randrange(MINIMUM_CPU_TIME[SET], MAXIMUM_CPU_TIME[SET])
		else:
			# It is GPU only
			time_gpu = random.randrange(MINIMUM_GPU_TIME[SET], MAXIMUM_GPU_TIME[SET])
	else:
		# It is double, the CPU time is random and the GPU time depends on the speedup
		time_cpu = random.randrange(MINIMUM_CPU_TIME[SET], MAXIMUM_CPU_TIME[SET])
		speedup = random.randrange(MINIMUM_GPU_SPEEDUP[SET], MAXIMUM_GPU_SPEEDUP[SET])
		time_gpu = round(time_cpu/speedup, 2)
	res += str(time_cpu) + ',' + str(time_gpu) + ',' + str(copy_time)
	return node_name, res
		
	
def linear_generator(height, depth):
	global last_generated_node_linear, first_node_last_row, graph
	for i in range(0, height):
		for i in range(0, depth):
			cur_node_name, cur_node = node_generator()
			# If it is the first node it depends on the first node of the earlier row
			if i == 0:
				cur_node += ','+str(first_node_last_row)+','+str(0)
				# The node just created is both the last created node for the row, and the first dependency for the next row
				first_node_last_row = cur_node_name
				last_generated_node_linear = cur_node_name
			else:
				cur_node += ','+str(last_generated_node_linear)+','+str(0)
				last_generated_node_linear = cur_node_name
			graph += '\n'+cur_node
		

#def tree_generator(depth):


with open('test.csv', 'w') as file:
	linear_generator(4,4)
	file.write(graph)

	