import sim as simulator
import generator

SETS = 3
SIZES_LINEAR = [4, 7, 10]
DEPTH_TREE = [4,5,6,7]

# 0 for G-FL, 1 for EDD and 2 for HEFT
SCHEDULER = 0

avg_res = {}

RUNS = 10

for run in range(RUNS):
	simulator.enable_print()
	print(run)
	simulator.disable_print()
	for set in range(SETS):
		simulator.enable_print()
		print('SET: '+str(set))
		simulator.disable_print()
		for DEPTH in range(len(SIZES_LINEAR)):
			for HEIGHT in range(len(SIZES_LINEAR)):
				generator.main([set, SIZES_LINEAR[HEIGHT], SIZES_LINEAR[HEIGHT]])
				simulator.enable_print()
				print('----------------------------------')
				simulator.disable_print()
				temp_sum = 0
				temp_count = 0
				for CPU_cores in range (2, 6):
					for FRAMES in range(1, 6, 2):
						output_line = 'Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')\t'
						output_line += 'CPU cores: '+str(CPU_cores)+' '
						output_line += 'FRAMES: '+str(FRAMES)+' '
						time, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores])
						output_line += 'Makespan (Frame): '+ output + ' '
						time_pipe, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores])
						output_line += '\tPIPE (%): '+ str(round(((time/time_pipe)-1.0)*100, 2)) + '%\t' + output
						simulator.enable_print()
						print(output_line)
						simulator.disable_print()
						temp_sum += round(((time/time_pipe)-1.0)*100, 2)
						temp_count += 1
				if 'Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')' in avg_res.keys():
					avg_res['Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')'] += round(temp_sum/temp_count, 2)
				else:
					avg_res['Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')'] = round(temp_sum/temp_count, 2)
		for DEPTH in range(len(DEPTH_TREE)):
			generator.main([set, DEPTH_TREE[DEPTH]])
			simulator.enable_print()
			print('----------------------------------')
			simulator.disable_print()
			temp_sum = 0
			temp_count = 0
			for CPU_cores in range (2, 6):
				for FRAMES in range(1, 6, 2):				
					output_line = 'Tree('+str(DEPTH_TREE[DEPTH])+')\t'
					output_line += 'CPU cores: '+str(CPU_cores)+' '
					output_line += 'FRAMES: '+str(FRAMES)+' '
					time, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores])
					output_line += 'Makespan (Frame): '+ output + ' '
					time_pipe, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores])
					output_line += '\tPIPE (%): '+ str(round(((time/time_pipe)-1.0)*100, 2)) + '%t' + output
					simulator.enable_print()
					print(output_line)
					simulator.disable_print()
					temp_sum += round(((time/time_pipe)-1.0)*100, 2)
					temp_count += 1
			if 'Tree('+str(DEPTH_TREE[DEPTH])+')' in avg_res.keys():
				avg_res['Tree('+str(DEPTH_TREE[DEPTH])+')'] += round(temp_sum/temp_count, 2)
			else:
				avg_res['Tree('+str(DEPTH_TREE[DEPTH])+')'] = round(temp_sum/temp_count, 2)
		
for key in avg_res:
	print(key, round(avg_res[avg]/RUNS, 2))