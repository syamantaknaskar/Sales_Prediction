import matplotlib.pyplot as plt
import csv
import numpy as np
import datetime
x=[]
y=[]
mean = np.zeros(24)
qt_t = np.zeros(24)
_hrs = np.arange(0,24,1)
max_quantity = 0
with open('../training_data/bag_train.csv','r') as myfile:
	rfile = csv.reader(myfile,delimiter = ',')
	next(rfile)
	for row in rfile:
		datestr, timestr=row[4].split()
		t = timestr.split(':')
		month, day, year = (int(x) for x in datestr.split('/'))
		dt = datetime.date(year, month, day).weekday()
		qty = int(row[3])
		if qty > max_quantity:
			max_quantity = qty
		if month == 10 and day == 7:
			x.append(int(t[0]))
			y.append(qty)
			mean[int(t[0])] = (mean[int(t[0])]*qt_t[int(t[0])]+qty)/(qt_t[int(t[0])]+1)
			qt_t[int(t[0])] = qt_t[int(t[0])]+1
		
		

print(max_quantity)
plt.xlabel("time for a perticular day... >")
plt.ylabel("quantity --- >")
plt.xlim(0,25)
plt.ylim(-1,100)
# plt.axis([0,int(max_price+1),0,max_quantity])
plt.xticks(np.arange(0,24,1))
plt.plot(_hrs, mean, 'ro')
plt.plot(x, y, '.')
plt.savefig('qty_time.png')
#plt.show()
plt.close()
