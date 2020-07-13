import os
import sim as simulator
import generator
import multiprocessing
from shutil import copyfile

SETS = 3
'''SIZES_LINEAR = [4, 7, 10]
DEPTH_TREE = [4, 5, 6]'''
SIZES_LINEAR = [4]
DEPTH_TREE = []

# 0 for G-FL, 1 for EDD, 2 for HEFT, 3 for G-FL_C and 4 for XEFT

RUNS = 1

def auto_sim():
	counter = -1
	output_csv = 'ID_graph, scheduler, params, sizes, frame, cpu_cores, makespan, makespan_pipe, improvement\n'
	UNIQUE_RUN_ID = 0 # Used to store the schedule
	for run in range(RUNS):
		for SET in range(0, SETS):
			simulator.enable_print()
			print('RUN', run, 'SET', SET)
			simulator.disable_print()
			for DEPTH in range(len(SIZES_LINEAR)):
				for HEIGHT in range(len(SIZES_LINEAR)):
					counter += 1
					generator.main([SET, SIZES_LINEAR[HEIGHT], SIZES_LINEAR[DEPTH]])
					copyfile('gen_graph.csv', './graphs/'+str(counter)+'_gen_graph.csv')
					simulator.enable_print()
					print('----------------------------------')
					simulator.disable_print()
					for CPU_cores in range (2, 6):
						for FRAMES in range(1, 6, 2):
							procs = []
							# GFL
							SCHEDULER = 0
							time_0 = multiprocessing.Value("d", 0.0, lock=False)
							t_0 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_0, UNIQUE_RUN_ID]])
							t_0.start()
							UNIQUE_RUN_ID_0 = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_0)
							
							# HEFT
							SCHEDULER = 2
							time_2 = multiprocessing.Value("d", 0.0, lock=False)
							t_2 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_2, UNIQUE_RUN_ID]])
							t_2.start()
							UNIQUE_RUN_ID_2 = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_2)
								
							# GFL_c
							SCHEDULER = 3
							time_3 = multiprocessing.Value("d", 0.0, lock=False)
							t_3 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_3, UNIQUE_RUN_ID]])
							t_3.start()
							UNIQUE_RUN_ID_3 = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_3)
								
							# XEFT
							SCHEDULER = 4
							time_4 = multiprocessing.Value("d", 0.0, lock=False)
							t_4 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_4, UNIQUE_RUN_ID]])
							t_4.start()
							UNIQUE_RUN_ID_4 = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_4)
							
							# Pipeline for all schedulers (if it is one frame, we skip this)
							if (FRAMES != 1):
								SCHEDULER = 0
								time_pipe_0 = multiprocessing.Value("d", 0.0, lock=False)
								t_0_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_0, UNIQUE_RUN_ID]])
								t_0_p.start()
								UNIQUE_RUN_ID_0_P = UNIQUE_RUN_ID
								UNIQUE_RUN_ID += 1
								procs.append(t_0_p)
								
								SCHEDULER = 2
								time_pipe_2 = multiprocessing.Value("d", 0.0, lock=False)
								t_2_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_2, UNIQUE_RUN_ID]])
								t_2_p.start()
								UNIQUE_RUN_ID_2_P = UNIQUE_RUN_ID
								UNIQUE_RUN_ID += 1
								procs.append(t_2_p)
								
								SCHEDULER = 3
								time_pipe_3 = multiprocessing.Value("d", 0.0, lock=False)
								t_3_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_3, UNIQUE_RUN_ID]])
								t_3_p.start()
								UNIQUE_RUN_ID_3_P = UNIQUE_RUN_ID
								UNIQUE_RUN_ID += 1
								procs.append(t_3_p)
								
								SCHEDULER = 4
								time_pipe_4 = multiprocessing.Value("d", 0.0, lock=False)
								t_4_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_4, UNIQUE_RUN_ID]])
								t_4_p.start()
								UNIQUE_RUN_ID_4_P = UNIQUE_RUN_ID
								UNIQUE_RUN_ID += 1
								procs.append(t_4_p)
							else:
								time_pipe_0 = time_0
								time_pipe_2 = time_2
								time_pipe_3 = time_3
								time_pipe_4 = time_4
							
							for t in procs:
								t.join()
								
							time_0 = time_0.value
							time_2 = time_2.value
							time_3 = time_3.value
							time_4 = time_4.value
							time_pipe_0 = time_pipe_0.value
							time_pipe_2 = time_pipe_2.value
							time_pipe_3 = time_pipe_3.value
							time_pipe_4 = time_pipe_4.value
								
							
							output_csv += (str(UNIQUE_RUN_ID_0) if FRAMES == 1 else str(UNIQUE_RUN_ID_0)+'.'+str(UNIQUE_RUN_ID_0_P))+','+str(counter)+','+str(0)+','+'L_'+str(SET)+','+str(SIZES_LINEAR[HEIGHT])+'.'+str(SIZES_LINEAR[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_0, 2))+','+str(round(time_pipe_0, 2))+','+str(round(((time_0/time_pipe_0)-1.0)*100, 2))+'\n'
							
							output_csv += (str(UNIQUE_RUN_ID_2) if FRAMES == 1 else str(UNIQUE_RUN_ID_2)+'.'+str(UNIQUE_RUN_ID_2_P))+','+str(counter)+','+str(2)+','+'L_'+str(SET)+','+str(SIZES_LINEAR[HEIGHT])+'.'+str(SIZES_LINEAR[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_2, 2))+','+str(round(time_pipe_2, 2))+','+str(round(((time_2/time_pipe_2)-1.0)*100, 2))+'\n'
							
							output_csv += (str(UNIQUE_RUN_ID_3) if FRAMES == 1 else str(UNIQUE_RUN_ID_3)+'.'+str(UNIQUE_RUN_ID_3_P))+','+str(counter)+','+str(3)+','+'L_'+str(SET)+','+str(SIZES_LINEAR[HEIGHT])+'.'+str(SIZES_LINEAR[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_3, 2))+','+str(round(time_pipe_3, 2))+','+str(round(((time_3/time_pipe_3)-1.0)*100, 2))+'\n'
							
							output_csv += (str(UNIQUE_RUN_ID_4) if FRAMES == 1 else str(UNIQUE_RUN_ID_4)+'.'+str(UNIQUE_RUN_ID_4_P))+','+str(counter)+','+str(4)+','+'L_'+str(SET)+','+str(SIZES_LINEAR[HEIGHT])+'.'+str(SIZES_LINEAR[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_4, 2))+','+str(round(time_pipe_4, 2))+','+str(round(((time_4/time_pipe_4)-1.0)*100, 2))+'\n'
							
							simulator.enable_print()
							print(counter, 'L', str(SIZES_LINEAR[HEIGHT])+','+str(SIZES_LINEAR[DEPTH]), 'Frames', FRAMES, 'Cpu cores', CPU_cores, 'DONE')
							simulator.disable_print()

			for DEPTH in range(len(DEPTH_TREE)):
				counter += 1
				generator.main([SET, DEPTH_TREE[DEPTH]])
				copyfile('gen_graph.csv', './graphs/'+str(counter)+'_gen_graph.csv')
				simulator.enable_print()
				print('----------------------------------')
				simulator.disable_print()
				for CPU_cores in range (2, 6):
					for FRAMES in range(1, 6, 2):
						procs = []
						# GFL
						SCHEDULER = 0
						time_0 = multiprocessing.Value("d", 0.0, lock=False)
						t_0 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_0, UNIQUE_RUN_ID]])
						t_0.start()
						UNIQUE_RUN_ID_0 = UNIQUE_RUN_ID
						UNIQUE_RUN_ID += 1
						procs.append(t_0)
						
						# HEFT
						SCHEDULER = 2
						time_2 = multiprocessing.Value("d", 0.0, lock=False)
						t_2 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_2, UNIQUE_RUN_ID]])
						t_2.start()
						UNIQUE_RUN_ID_2 = UNIQUE_RUN_ID
						UNIQUE_RUN_ID += 1
						procs.append(t_2)
							
						# GFL_c
						SCHEDULER = 3
						time_3 = multiprocessing.Value("d", 0.0, lock=False)
						t_3 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_3, UNIQUE_RUN_ID]])
						t_3.start()
						UNIQUE_RUN_ID_3 = UNIQUE_RUN_ID
						UNIQUE_RUN_ID += 1
						procs.append(t_3)
							
						# XEFT
						SCHEDULER = 4
						time_4 = multiprocessing.Value("d", 0.0, lock=False)
						t_4 = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 0, CPU_cores, time_4, UNIQUE_RUN_ID]])
						t_4.start()
						UNIQUE_RUN_ID_4 = UNIQUE_RUN_ID
						UNIQUE_RUN_ID += 1
						procs.append(t_4)
						
						# Pipeline for all schedulers (if it is one frame, we skip this)
						if (FRAMES != 1):
							SCHEDULER = 0
							time_pipe_0 = multiprocessing.Value("d", 0.0, lock=False)
							t_0_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_0, UNIQUE_RUN_ID]])
							t_0_p.start()
							UNIQUE_RUN_ID_0_P = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_0_p)
							
							SCHEDULER = 2
							time_pipe_2 = multiprocessing.Value("d", 0.0, lock=False)
							t_2_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_2, UNIQUE_RUN_ID]])
							t_2_p.start()
							UNIQUE_RUN_ID_2_P = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_2_p)
							
							SCHEDULER = 3
							time_pipe_3 = multiprocessing.Value("d", 0.0, lock=False)
							t_3_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_3, UNIQUE_RUN_ID]])
							t_3_p.start()
							UNIQUE_RUN_ID_3_P = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_3_p)
							
							SCHEDULER = 4
							time_pipe_4 = multiprocessing.Value("d", 0.0, lock=False)
							t_4_p = multiprocessing.Process(target=simulator.main, args=[['gen_graph.csv', SCHEDULER, FRAMES, 1, CPU_cores, time_pipe_4, UNIQUE_RUN_ID]])
							t_4_p.start()
							UNIQUE_RUN_ID_4_P = UNIQUE_RUN_ID
							UNIQUE_RUN_ID += 1
							procs.append(t_4_p)
						else:
							time_pipe_0 = time_0
							time_pipe_2 = time_2
							time_pipe_3 = time_3
							time_pipe_4 = time_4
						
						for t in procs:
							t.join()
						
						time_0 = time_0.value
						time_2 = time_2.value
						time_3 = time_3.value
						time_4 = time_4.value
						time_pipe_0 = time_pipe_0.value
						time_pipe_2 = time_pipe_2.value
						time_pipe_3 = time_pipe_3.value
						time_pipe_4 = time_pipe_4.value
						
						output_csv += (str(UNIQUE_RUN_ID_0) if FRAMES == 1 else str(UNIQUE_RUN_ID_0)+'.'+str(UNIQUE_RUN_ID_0_P))+','+str(counter)+','+str(0)+','+'T_'+str(SET)+','+str(DEPTH_TREE[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_0, 2))+','+str(round(time_pipe_0, 2))+','+str(round(((time_0/time_pipe_0)-1.0)*100, 2))+'\n'
						
						output_csv += (str(UNIQUE_RUN_ID_2) if FRAMES == 1 else str(UNIQUE_RUN_ID_2)+'.'+str(UNIQUE_RUN_ID_2_P))+','+str(counter)+','+str(2)+','+'T_'+str(SET)+','+str(DEPTH_TREE[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_2, 2))+','+str(round(time_pipe_2, 2))+','+str(round(((time_2/time_pipe_2)-1.0)*100, 2))+'\n'
						
						output_csv += (str(UNIQUE_RUN_ID_3) if FRAMES == 1 else str(UNIQUE_RUN_ID_3)+'.'+str(UNIQUE_RUN_ID_3_P))+','+str(counter)+','+str(3)+','+'T_'+str(SET)+','+str(DEPTH_TREE[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_3, 2))+','+str(round(time_pipe_3, 2))+','+str(round(((time_3/time_pipe_3)-1.0)*100, 2))+'\n'
						
						output_csv += (str(UNIQUE_RUN_ID_4) if FRAMES == 1 else str(UNIQUE_RUN_ID_4)+'.'+str(UNIQUE_RUN_ID_4_P))+','+str(counter)+','+str(4)+','+'T_'+str(SET)+','+str(DEPTH_TREE[DEPTH])+','+str(FRAMES)+','+str(CPU_cores)+','+str(round(time_4, 2))+','+str(round(time_pipe_4, 2))+','+str(round(((time_4/time_pipe_4)-1.0)*100, 2))+'\n'
						
						simulator.enable_print()
						print(counter, 'T', DEPTH_TREE[DEPTH], 'Frames', FRAMES, 'Cpu cores', CPU_cores, 'DONE')
						simulator.disable_print()

	with open('output.csv', 'w+') as output:
		output.write(output_csv)

if __name__ == "__main__":
	auto_sim()