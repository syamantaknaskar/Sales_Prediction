import matplotlib.pyplot as plt
import csv
import numpy as np
import datetime
x=[]
y=[]
mean = np.zeros(12)
qt_t = np.zeros(12)
mth = np.arange(1,13,1)
max_quantity = 0
with open('../training_data/bag_train.csv','r') as myfile:
	rfile = csv.reader(myfile,delimiter = ',')
	next(rfile)
	for row in rfile:
		datestr, timestr=row[4].split()
		month, day, year = (int(x) for x in datestr.split('/'))
		qty = int(row[3])
		if qty > max_quantity:
			max_quantity = qty
		x.append(month)
		y.append(qty)
		mean[month-1] = (mean[month-1]*qt_t[month-1] + qty)/(qt_t[month-1]+1)
		qt_t[month-1] = qt_t[month-1] +1

print(max_quantity)
print(mean)
plt.xlabel("month... >")
plt.ylabel("quantity --- >")
plt.xlim(0,13)
# plt.axis([0,int(max_price+1),0,max_quantity])
plt.xticks(np.arange(0,12,1))
plt.plot(mth,mean,'r-')
plt.plot(x, y, '.')
plt.savefig('qty_month.png')
#plt.show()
plt.close()
