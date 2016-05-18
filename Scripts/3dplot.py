import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import datetime

x = []
y = []
z = []
max_quantity = 0
with open('../training_data/bag_train.csv','r') as myfile:
	rfile = csv.reader(myfile,delimiter = ',')
	next(rfile)
	for row in rfile:
		datestr, timestr=row[4].split()
		month, day, year = (int(x) for x in datestr.split('/'))
		dt = datetime.date(year, month, day).weekday()
		ti,tmp = (int(x) for x in timestr.split(':'))
		qty = int(row[3])
		if qty > max_quantity:
			max_quantity = qty
		x.append(dt)
		y.append(ti)
		z.append(qty)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100

ax.scatter(x, y, z, c='r', marker='.')

ax.set_xlabel('Day -- >')
ax.set_ylabel('Time Interval -- >')
ax.set_zlabel('Quantity -- >')
fig.savefig('3dplot.jpg')

plt.show()