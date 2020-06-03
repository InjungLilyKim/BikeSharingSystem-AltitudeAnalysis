# Bike sharing system analysis by python statsmodels
# % Poisson Regression model with prediction, validation
# Author: Injung Kim
# Last modified: 6/1/2020

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels
from statsmodels.genmod.families import Poisson
from statsmodels.tools.eval_measures import rmse

from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression 

from patsy import dmatrices
import warnings
warnings.filterwarnings("ignore")

statsmodels.__version__

# poisson regression validation 
def pr_calc_err(predictions, x, y):
	#rmse
	err = np.sqrt(metrics.mean_squared_error(y, predictions.predict(x)))
	#print(err)
	print(round(err, 4))

# draw prediction plot 
def pr_plot(predictions, weather, model, x_test, y_test):
	y_pred = predictions.predict(x_test)

	#print(x_test)
	#print(y_test)
	y_test.sort_index(inplace=True)
	y_pred.sort_index(inplace=True)
	plt.plot(y_test, c="blue", label="actual", linewidth=2)
	plt.plot(y_pred, c="red", label="predicted", linewidth=2)
	plt.savefig(weather + str(model) + "_model.png")
	plt.clf()
	plt.cla()
	plt.close()

# run model and prediction
def runModelandPrediction(formula, w, train_set, valid_set, test_set):
	y_train, x_train = dmatrices(formula, train_set, return_type='dataframe')
	y_valid, x_valid = dmatrices(formula, valid_set, return_type='dataframe')
	y_test, x_test = dmatrices(formula, test_set, return_type='dataframe')

	# get rmse
	predictions = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
	print("PR train rmse: in-sample error")
	train_err = pr_calc_err(predictions, x_train, y_train)
	print("PR valid rmse: out-of-sample error")
	valid_err = pr_calc_err(predictions, x_valid, y_valid)
	print("PR test rmse")
	test_err = pr_calc_err(predictions, x_test, y_test)

	# draw plot
	pr_plot(predictions, w, m, x_test, y_test)
	print("\--------------------------------------")

#main
weather = ["summer", "winter"]

#model numbers
#summerWinterModels = [1, 3, 4, 5, 6, 8, 9, 10]
summerWinterModels = [9]
#allSeasonsModels = [2, 7]

# dataframe for all seasons
summerTrainDF = []
summerValidDF = []
summerTestDF = []
allSeasonsTrainDF = []
allSeasonsValidDF = []
allSeasonsTestDF = []

# run summer/winter seasons model
for w in weather:
	print("\ 1)season:")
	print(w)

	df = pd.read_csv('./modelling_9_' + w + '.csv')
		
	# split data 70:20:10
	train_set, valid_set, test_set = np.split(df.sample(frac=1), [int(.7*len(df)), int(.9*len(df))])

	# save the dataset for allSeasonsModels
	if w == "summer":
		summerTrainDF = train_set.copy()
		summerValidDF = valid_set.copy()
		summerTestDF = test_set.copy()
	elif w == "winter":
		allSeasonsTrainDF = pd.concat([summerTrainDF, train_set], ignore_index=True)
		allSeasonsValidDF = pd.concat([summerValidDF, valid_set], ignore_index=True)
		allSeasonsTestDF = pd.concat([summerTestDF, test_set], ignore_index=True)

	# create a formula for each model
	for m in summerWinterModels:
		if m == 1:
			print("\ 2)altitude and distance are range value")
			print("\ 3)include same station rental/return")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+C(dis_range)*C(alti_range)"""
		elif m == 3:
			print("\ 2)altitude and distance are real value")
			print("\ 3)include same station rental/return")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+distance_real*alti_diff"""
		elif m == 4:
			print("\ 2)altitude range and distance real value")
			print("\ 3)include same station rental/return")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+distance_real*C(alti_range)"""
		elif m == 5:
			print("\ 2)altitude real and distance range value")
			print("\ 3)include same station rental/return")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+C(dis_range)*alti_diff"""

		elif m == 6:
			print("\ 2)both are range value")
			print("\ 3)include a loop feature")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+C(dis_range)*C(alti_range)+C(loop)"""
		elif m == 8:
			print("\ 2)both are real value")
			print("\ 3)include a loop feature")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+distance_real*alti_diff+C(loop)"""
		elif m == 9:
			print("\ 2)altitude range and distance real value")
			print("\ 3)include a loop feature")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+distance_real*C(alti_range)+C(loop)"""
			y, x = dmatrices(formula, df, return_type='dataframe')

			pm = sm.GLM(y, x, family=sm.families.Poisson()).fit()
			print(pm.summary().as_latex())
			#print("poisson regression's rmse value")
			#print(sm.tools.eval_measures.rmse(response, pm.fittedvalues, axis=0))
		elif m == 10:
			print("\ 2)altitude real and distance range value")
			print("\ 3)include a loop feature")
			formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+C(dis_range)*alti_diff+C(loop)"""

		# run model and prediction
		#runModelandPrediction(formula, w, train_set, valid_set, test_set)

# run two all seasons model
print("\ 1)season: all")

# create a formula for each model
for m in allSeasonsModels:
	if m == 2:
		print("\ 2)altitude and distance are range value")
		print("\ 3)include same station rental/return")
		formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+C(dis_range)*C(alti_range)"""
	elif m == 8:
		print("\ 2)altitude and distance are range value")
		print("\ 3)include a loop feature")
		formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+C(dis_range)*C(alti_range)+C(loop)"""

	# run model and prediction
	#runModelandPrediction(formula, "all", allSeasonsTrainDF, allSeasonsValidDF, allSeasonsTestDF)
