from csv import reader

with open('output.csv', 'r') as data, open('result.csv', 'r') as merge, open('new.csv', 'w') as new:
	csv_data = reader(data)
	csv_merge = reader(merge)
	output_text = ''
	for line in csv_data:
		append = ''
		if '.' in line[0]:
			id = line[0].split('.')
			for line_merge in csv_merge:
				if (id[1] == line_merge[0]):
					append = str(line_merge[1])
					break
		for word in line:
			output_text += word + ','
		output_text += append + '\n'
	new.write(output_text)
		