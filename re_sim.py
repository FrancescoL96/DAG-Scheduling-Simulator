import sim as simulator
from time import process_time 

output = ""
t0 = process_time()
for i in range(0, 288000):
	if (i % 3000 == 0):
		t1 = process_time()
		simulator.enable_print()
		print(round((i/288000)*100, 2), '- Estimated time left (minutes): ', round(((100.0-round((i/288000)*100, 2))*(t1-t0))/60, 2))
		simulator.disable_print()
		t0 = process_time()
	output += str(i)+','
	output += simulator.main([str(i)+'_schedule.pkl'])

with open('result.csv', 'a') as result:
	result.write(output)