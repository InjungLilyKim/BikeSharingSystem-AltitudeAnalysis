# Bike sharing system analysis by python statsmodels
# % Poisson Regression model with prediction
# Author: Injung Kim
# Last modified: 5/26/2020

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

# poisson regression
def pr(weather):
	# create a poisson regression model
	print("\--------------------------------------")
	print("\poisson regression with constant")
	print("\ 1)altitude and distance are real value")
	print("\ 2)season:")
	print(weather)
	print("\ 3)include same station rental/return")
	print("\--------------------------------------")
	df = pd.read_csv('./modelling_9_' + weather + '.csv')
	formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+distance_real*alti_diff"""
	response, predictors = dmatrices(formula, df, return_type='dataframe')
	pm = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
	print(pm.summary().as_latex())
	#print("poisson regression's rmse value")
	#print(sm.tools.eval_measures.rmse(response, pm.fittedvalues, axis=0))

	#print("poisson regression's rmse value")
	#print(sm.tools.eval_measures.rmse(response, pm.fittedvalues, axis=0))

	pr_predict(weather, predictors, response)

# poisson regression prediction
def pr_predict(weather, x, y):
	#############################
	#here is for train/test ratio 80:20 
	size = 0.2

	#train r2, rmse
	print("PR train r2 and rmse")
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = size)
	pm_train = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
	print(np.sqrt(metrics.mean_squared_error(y_train, pm_train.predict(x_train))))
	#test r2, rmse
	print("PR test r2 and rmse")
	pm_test = sm.Poisson(y_train, x_train).fit()
	print(np.sqrt(metrics.mean_squared_error(y_test, pm_test.predict(x_test))))
	print("\n********************************")
	y_pred = pm_test.predict(x_test)

	#print(y_test)
	#print(y_pred)
	y_test.sort_index(inplace=True)
	y_pred.sort_index(inplace=True)
	#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	#df1 = df.head(10000)
	#df1.sort_index(inplace=True)
	#print(df1['Actual'])
	#print(df1['Predicted'])
	#plt.plot(df1['Actual'], c="blue", label="actual", linewidth=2)
	#plt.plot(df1['Predicted'], c="red", label="predicted", linewidth=2)
	plt.plot(y_test, c="blue", label="actual", linewidth=2)
	plt.plot(y_pred, c="red", label="predicted", linewidth=2)
	plt.savefig(weather + "_pr_model3.png")
	plt.clf()
	plt.cla()
	plt.close()

pr("summer")
pr("winter")

