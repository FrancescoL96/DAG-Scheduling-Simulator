import sim as simulator
import generator
from shutil import copyfile

SETS = 3
SIZES_LINEAR = [4, 7]
DEPTH_TREE = [4]

# 0 for G-FL, 1 for EDD, 2 for HEFT and 3 for G-FL_C
SCHEDULER = 3
SCHEDULER_COMP = 2

avg_res = {}
max_res = 0.0

counter = 0

RUNS = 1

for run in range(RUNS):
	simulator.enable_print()
	print('RUN', run)
	simulator.disable_print()
	for set in range(2, SETS):
		simulator.enable_print()
		print('SET', set)
		simulator.disable_print()
		for DEPTH in range(len(SIZES_LINEAR)):
			for HEIGHT in range(len(SIZES_LINEAR)):
				generator.main([set, SIZES_LINEAR[HEIGHT], SIZES_LINEAR[HEIGHT]])
				simulator.enable_print()
				print('----------------------------------')
				simulator.disable_print()
				temp_sum = 0
				temp_sum_comp = 0
				temp_count = 0
				for CPU_cores in range (2, 6):
					for FRAMES in range(1, 6, 2):
						comp_time, output = simulator.main(['gen_graph.csv', SCHEDULER_COMP, FRAMES, 0, CPU_cores])
						
						output_line = 'Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')\t'
						output_line += 'CPU cores: '+str(CPU_cores)+' '
						output_line += 'FRAMES: '+str(FRAMES)+' '
						time, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores])
						
						output_line += 'Makespan (Frame): '+ output + ' '
						if (FRAMES != 1):
							time_pipe, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores])
							output_line += '\tPIPE (%): '+ str(round(((time/time_pipe)-1.0)*100, 2)) + '%\t' + output
							temp_sum += round(((time/time_pipe)-1.0)*100, 2)
							if round(((time/time_pipe)-1.0)*100, 2) >= 25.0:
								copyfile('gen_graph.csv', './graph25+/gen_graph'+str(counter)+'.csv')
								counter += 1
							temp_sum_comp += round(((comp_time/time_pipe)-1.0)*100, 2)
							temp_count += 1
							if max_res < round(((time/time_pipe)-1.0)*100, 2):
								max_res = round(((time/time_pipe)-1.0)*100, 2)
						simulator.enable_print()
						print(output_line)
						simulator.disable_print()						
				if 'Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')' in avg_res.keys():
					avg_res['Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')'] = (avg_res['Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')'][0] + round(temp_sum/temp_count, 2), avg_res['Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')'][1] + round(temp_sum_comp/temp_count, 2))
				else:
					avg_res['Linear('+str(SIZES_LINEAR[HEIGHT])+', '+str(SIZES_LINEAR[DEPTH])+')'] = (round(temp_sum/temp_count, 2), round(temp_sum_comp/temp_count, 2))

		for DEPTH in range(len(DEPTH_TREE)):
			generator.main([set, DEPTH_TREE[DEPTH]])
			simulator.enable_print()
			print('----------------------------------')
			simulator.disable_print()
			temp_sum = 0
			temp_sum_comp = 0
			temp_count = 0
			for CPU_cores in range (2, 6):
				for FRAMES in range(1, 6, 2):				
					comp_time, output = simulator.main(['gen_graph.csv', SCHEDULER_COMP, FRAMES, 0, CPU_cores])
					
					output_line = 'Tree('+str(DEPTH_TREE[DEPTH])+')\t'
					output_line += 'CPU cores: '+str(CPU_cores)+' '
					output_line += 'FRAMES: '+str(FRAMES)+' '
					time, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores])
					output_line += 'Makespan (Frame): '+ output + ' '
					if (FRAMES != 1):
						time_pipe, output = simulator.main(['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores])
						output_line += '\tPIPE (%): '+ str(round(((time/time_pipe)-1.0)*100, 2)) + '%\t' + output
						temp_sum += round(((time/time_pipe)-1.0)*100, 2)
						if round(((time/time_pipe)-1.0)*100, 2) >= 25.0:
							copyfile('gen_graph.csv', './graph25+/gen_graph'+str(counter)+'.csv')
							counter += 1
						temp_sum_comp += round(((comp_time/time_pipe)-1.0)*100, 2)
						temp_count += 1
						if max_res < round(((time/time_pipe)-1.0)*100, 2):
							max_res = round(((time/time_pipe)-1.0)*100, 2)
					simulator.enable_print()
					print(output_line)
					simulator.disable_print()
			if 'Tree('+str(DEPTH_TREE[DEPTH])+')' in avg_res.keys():
				avg_res['Tree('+str(DEPTH_TREE[DEPTH])+')'] = (avg_res['Tree('+str(DEPTH_TREE[DEPTH])+')'][0] + round(temp_sum/temp_count, 2), avg_res['Tree('+str(DEPTH_TREE[DEPTH])+')'][1] + round(temp_sum_comp/temp_count, 2))
			else:
				avg_res['Tree('+str(DEPTH_TREE[DEPTH])+')'] = (round(temp_sum/temp_count, 2), round(temp_sum_comp/temp_count, 2))
		
for key in avg_res:
	simulator.enable_print()
	print(str(key), round(avg_res[key]/RUNS, 2))

print('Max', max_res)