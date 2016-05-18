import matplotlib.pyplot as plt
import csv
import numpy as np
import datetime
x=[]
y=[]
mean = np.zeros(7)
qt_t = np.zeros(7)
_day = np.arange(0,7,1)
max_quantity = 0
with open('../training_data/bag_train.csv','r') as myfile:
	rfile = csv.reader(myfile,delimiter = ',')
	next(rfile)
	for row in rfile:
		datestr, timestr=row[4].split()
		month, day, year = (int(x) for x in datestr.split('/'))
		dt = datetime.date(year, month, day).weekday()
		qty = int(row[3])
		if qty > max_quantity:
			max_quantity = qty
		mean[dt] = (mean[dt]*qt_t[dt]+qty)/(qt_t[dt]+1)
		qt_t[dt] = qt_t[dt]+1
		x.append(dt)
		y.append(qty)

print(_day,mean)
print(max_quantity)
plt.xlabel("day... >")
plt.ylabel("quantity --- >")
plt.xlim(-1,7)
plt.ylim(-1,100)
# plt.axis([0,int(max_price+1),0,max_quantity])
#plt.xticks(np.arange(0,12,1))
plt.plot(_day,mean,'ro')
plt.plot(x, y, '.')
plt.savefig('qty_day.png')
#plt.show()
plt.close()
