import sys

def write_data(data, filename):
	f = open(filename, "w+")
	for item in data:
		for line in item:
			f.write(line)
		f.write("\n")
	f.close()

f = open(sys.argv[1])

data = []
ins = []
for line in f:
	if line.strip():
		ins.append(line)
	else:
		data.append(ins)
		ins = []

num = len(data)
print "data num :  "+str(num)

train = data[:num/20*15]
dev = data[num/20*15:num/20*17]
test = data[num/20*17:]

write_data(train, "traindata")
write_data(dev, "devdata")
write_data(test, "testdata")

		
	
