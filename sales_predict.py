#import libraries required for the project
import sys as sys
import csv as csv			# for manipulating csv files
import numpy as npy		# for matrices and heavy math computations	
import datetime as dtm	# for extractiong day of the week
import pandas as pds		# for manipulting csv and other data files

# for MACHINE LEARNING models
from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

import random as rnd		# for random numbers
import datetime			# for time and date format
import re				# for regular expression
import math	as mat		# for math function like sqrt
import matplotlib.pyplot as plt

# return a list k direfferent random numbers btw a and b spaced by c
def randlist(a,b,c):
	ret = []
	while a<=b:
		ret.append(rnd.randint(a,min(a+c,b)))
		a+=c
	return ret

def tmp_fun():
	# devide the data into training and testing data
	for i in range(0,18):
		pdt = input()
		tpdt = pdt+".csv"
		map_date = dict()				# map all products in day with day
		header = ["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"]
		# open the data file 'online_retail_m.csv'
		with open(tpdt,'r',newline ='', encoding='mac_roman') as a:
			p=csv.reader(a,delimiter=',')
			next(p)
			for row in p:
				datestr , timestr = row[4].split()
				month, day, year = (int(x) for x in datestr.split('/'))
				if datestr in map_date:
					map_date[datestr].append(row)
				else :
					map_date[datestr]=[row]
		
		#create train_data.csv and test_data.csv
		tpdt = pdt+"_train.csv"
		ttpdt = pdt+"_test.csv"
		with open(tpdt,'w+',newline = '') as a, open(ttpdt,'w+',newline = '') as b:
			awrite = csv.writer(a,delimiter=',')
			bwrite = csv.writer(b,delimiter=',')
			# write header
			awrite.writerow(header)
			bwrite.writerow(header)
			# for each day take one out of five as test data and remaining as train data 
			for x in map_date:
				ntrain = len(map_date[x])
				rlist = randlist(0,ntrain-1,5)			# get list of random numbers for random selection of data
				for i in range(0,ntrain):
					if i in rlist:
						bwrite.writerow(map_date[x][i])		# test data	
					else :
						awrite.writerow(map_date[x][i])		# train data

# list of selected features
selected_products = ["clock","paper napkins","postage","bird ornament","teacup and saucer","cakestand","bunting","jelly moulds","chalkboard",
				"paper chain kit","night light","picture frame","jam making set","cake tins","spotty bunting","recipe box","bag","cake"]

# create a map from products to its selected features
pdt_feature_map={
	'clock':["ALARM","BAKELIKE","PINK","RED","GREEN","DINER","WALL","BLUE","IVORY","ORANGE","BLACK","CHOCOLATE","WHITE","TRAVEL","GRASS","HOPPER","WOODEN","STITCHED","SQUARE","TABLE","SAVOY","DECO","CHERRY","BLOSSOM","FLOWER","PURPLE","SUCKER","BAROQUE","CARRIAGE","MAGNET","KITCHEN","BULL","DOG","BOTTLE","REFECTORY","MINT","ROUND","PEONY","ARTICULATED","RETRO"],
	'paper napkins':["RED","RETROSPOT","PAPER","NAPKINS","PACK","ENGLISH","ROSE","SKULL","STRAWBERRY","FRUIT","SALAD"],
	'postage':["DOTCOM"],
	'bag':["CHARLOTTE","GIRL","DOLLY","JUMBO","PINK","POLKADOT","BAROQUE","BLACK","WHITE","CHARLIE","LOLA","STRAWBERRY","LUNCH","RED","RETROSPOT","STORAGE","SUKI","VINTAGE","PAISLEY","SKULL","RECYCLING","SPACEBOY","WOODLAND","BLUE","SWIRLY","GLASS","TASSLE","CHARM","OWLS","SHOPPER","TOTE","LOVE","WOODEN","SKITTLES","COTTON","SILVER","BAUBLES","CIRCULAR","RUBBERS","PARTY","STICKER","CHAMPION","SMALL","STRIPES","CHOCOLATE","GIFT","HANDBAG","BOXES","SKULLS","ANIMALS","COSMETIC","ROSE","LOLITA","SHOPPING","SOFT","TOY","SCANDINAVIAN","WASH","BROWN","FRUIT","TURQUOISE","GREEN","WASHBAG","BUTTERFLY","BOBBLES","GEMS","FLORAL","FELTCRAFT","COSMETICS","FUNKY","MONKEY","MEDIUM","DRAWSTRING","KIDS","CABIN","CAROUSEL","LARGE","TAHITI","BEACH","GOLD","PRINT","PAPER","YELLOW","FLOWERS","PEG","APPLES","GREY","PSYCHEDELIC","WEEKEND","DINOSAUR","BEAD","COASTERS","GAUZE","SUMMER","DAISIES","BUTTERFLIES","PICNIC","STONES","RABBIT","FLOWER","EMBROIDERY","FOOD","TURQ","BUM","POCKET","SPOT","PAISELY","BOTTLE","POSY","CANDY","STAR","AMBER","PURPLE","JACK","BIRTHDAY","AQUA","BERTIE","SCOTTIES","DISCO","HAND","MONTE","CARLO","DAISY","METALIC","LEAVES","CHARMS","HEART","KING","RETRO","BIG","APPLE","SPOTTY","BOYS","GIRLS","RIVIERA","ALPHABET","LEAF","HANDLE","HAYNES","CAMPER","PEARS","DOILEY","BLUE","ORANGE","CHRISTMAS","BRASS","IVORY","METAL","DOILY"],
	'bird ornament':["ASSORTED","COLOUR"],
	'teacup and saucer':["ROSES","REGENCY","GREEN","PINK"],
	'cakestand':["REGENCY","TIER","SWEETHEART","LOVEHEART"],
	'bunting':["VINTAGE","JACK","PAPER","WHITE","LACE","COLOURED","RETROSPOT","RED","BABY","WOODEN","PINK","TEA","PARTY","BLUE","HAPPY","BIRTHDAY","PAISLEY","REGATTA","SPOTTY","CHRISTMAS","PARK"],
	'jelly moulds':["PANTRY"],
	'chalkboard':["NATURAL","SLATE","HEART","RECTANGLE","KITCHEN","LARGE","CHRISTMAS","STAR"],
	'cake':["RETROSPOT","DINOSAUR","PINK","PAISLEY","FAIRY","CERAMIC","CHERRY","MONEY","STRAWBERRY","GREEN","CREAM","RED","FLANNEL","SKULL","SPACEBOY","SWEETHEART","MAGIC","VICTORIAN","FILIGREE","MED","VINTAGE","MUSHROOM","DOLLY","HEART","FOIL","REGENCY","CAKESTAND","STAR","LOVEBIRD","WHITE","MINI","JIGSAW","BIRTHDAY","CANDLE","BOWS","GIFT","TAPE","SMALL","HANGING","CHILDRENS","APRON","METAL","SINGLE","HOOK","NOTEBOOK","ROUND","TINS","PLACEMATS","SPOTTED","LARGE","CHOCOLATE","SPOTS","NOVELTY","BISCUITS","IVORY","CUP","LOVEHEART","FRIDGE","MAGNETS","LIGHTS","WEDDING","WRAP","CHOPSTICKS","UMBRELLA","HEARTS","STRAWBERY","DECORATION","RABBITS","STICKER","CUSHION","CUTLERY","BAKING","MOULD","CUPCAKES","GARDEN","PANTRY","SKETCHBOOK","BLACK","MEDIUM","GLASS","LAVENDER","SCENT","PANCAKE","GOLD","SILVER","FORK","SLICE","LOVE","LEAF","FRILL","HAPPY","TEDDY","DECORATIONS","SWEET","CARRIER","CHRISTMAS"],
	'paper chain kit':["CHRISTMAS","RETROSPOT","VINTAGE","LONDON","EMPIRE","SKULLS"],
	'night light':["RED","TOADSTOOL","LED","SUNJAR","RABBIT","FAIRY","TALE","COTTAGE",],
	'picture frame':["WOODEN","WHITE","FAMILY","ALBUM","WOOD","TRIPLE","PORTRAIT","STICKERS","MEDIUM","PARLOUR","LARGE","SMALL"],
	'jam making set':["JARS","PRINTED"],
	'cake tins':["RED","RETROSPOT","ROUND","PANTRY","SKETCHBOOK","REGENCY"],
	'spotty bunting':[],
	'recipe box':["METAL","HEART","RETROSPOT","PANTRY","YELLOW","BLUE","SKETCHBOOK"]
	}
				
# object for each product
class product_model():
	def __init__(self):
		self.svr_model = svm.SVR(gamma=0.1, C=500, kernel='rbf')
		self.lr_model = linear_model.Ridge(alpha=1)

# function to extract the data in the required format from raw data 
def data_format(file_name):
	XValues = []
	YValues = []
	with open(file_name,newline='',encoding='mac_roman') as data:
		dreader = csv.reader(data,delimiter=',')
		next(dreader)
		for row in dreader:
			# pdt_price and pdt_cancel
			#~ price = mat.exp(-float(row[5])) + float(row[5])
			#~ price = 1/float(row[5])**3 + 1/float(row[5])**2 + 1/float(row[5]) + float(row[5]) + 10 
			price = (float(row[5])**3)/6 + (float(row[5])**2)/2 + float(row[5]) + 10 
			xrow=[price, int('C' in row[0])]
			
			datestr, timestr=row[4].split()								# extract time and date
			month, day, year = (int(x) for x in datestr.split('/'))				# split date into day, month and year
			dt = datetime.date(year, month, day).weekday()				# get the day of the week
			timestr = timestr.split(":")									# extract hour from timestamp hh:mm
			
			drow=[0]*7												# temparory list for 7days of the week
			trow = [0]*24												# temparory list for 24hrs
			mrow=[0]*12												# temporory list for 12 months
			drow[dt]=1												# set to true if the sale is on dt_th day
			trow[int(timestr[0])]=1										# set to true at which sale happened
			mrow[month-1]=1											# set to true the in which sales happened
		
			tmp_feature = re.findall(r"[\w']+",row[2])						# extract the features from the product discriptions
			frow=[0]*(len(pdt_feature_map[selected_products[itr]]))		# temparory feature vector and all vectors are set to false
			for x in tmp_feature:
				if x in pdt_feature_map[selected_products[itr]]:			# if the feature x is in considered features then set the corresponding feature in feature vector
					frow[pdt_feature_map[selected_products[itr]].index(x)]=1
			
			# concat all features into one single feature vector
			xrow = xrow + drow + mrow + trow + frow
			XValues.append(xrow)							# add training set to data matrix
			YValues.append(int(row[3]))					# ouput values
	return [XValues,YValues]
	
# create a dictionary/map from product_name to model
pdt_model_map = dict()

print("RSME")

# for each product train the data and create model
for itr in range(0,len(selected_products)):
	# create a product model for each product
	if selected_products[itr] not in pdt_model_map:
		pdt_model_map[selected_products[itr]] = product_model()
	
	file_name = "training_data/"+selected_products[itr]+"_train.csv"
	
	# data matrix
	data = data_format(file_name)
	XValues = data[0]
	YValues = data[1]	
	
	# 5-fold corss validation
	Xtrain, Xtest,Ytrain, Ytest = cross_validation.train_test_split(XValues,YValues,test_size=0.2,random_state=0)
	
	#train and fit the data and predict
	pdt_model_map[selected_products[itr]].svr_model.fit(XValues,YValues)
	
	file_name = "test_data/"+selected_products[itr]+"_test.csv"
	# data matrix
	data = data_format(file_name)
	XValues = data[0]
	YValues = data[1]
	
	#train and fit the data and predict
	ypredict = pdt_model_map[selected_products[itr]].svr_model.predict(XValues)
	
	print(mat.sqrt(metrics.mean_squared_error(YValues,ypredict)))
	
	#plt.hold('on')
	#plt.plot(XValues, ypredict, c='r')
	#plt.figure()
	#plt.show()
