import matplotlib.pyplot as plt
import csv
import numpy as np
x=[]
y=[]
max_price = 0
max_quantity = 0
with open('../training_data/bag_train.csv','r') as myfile:
	rfile = csv.reader(myfile,delimiter = ',')
	next(rfile)
	for row in rfile:
		price = float(row[5])
		qty = int(row[3])
		if  price > max_price:
			max_price = price
		if qty > max_quantity:
			max_quantity = qty
		x.append(price)
		y.append(qty)

print(max_quantity)

plt.xlabel("price ... >")
plt.ylabel("quantity --- >")
plt.xlim(0,30)
# plt.axis([0,int(max_price+1),0,max_quantity])
plt.xticks(np.arange(0,int(max_price+1)/2,1))
plt.plot(x, y, '.')
plt.savefig('qty_price.png')
#plt.show()
plt.close()
