# Bike sharing system analysis by python statsmodels
# % Poisson Regression model with prediction
# Author: Injung Kim
# Last modified: 5/22/2020

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

import warnings
warnings.filterwarnings("ignore")

statsmodels.__version__

# poisson regression
def pr(weather):
	# create a poisson regression model
	print("\--------------------------------------")
	print("\poisson regression with constant")
	print("\--------------------------------------")
	#pd.set_option('display.max_columns', None)
	df = pd.read_csv('./modelling_' + weather + '.csv')
	df = pd.concat((df, pd.get_dummies(df['dis_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['alti_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['alti_dis_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['dayofweek'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['week_wknd'])), axis=1)
	#pd.options.display.max_rows=10
	y = df['daily_usage']
	x = df[[-300, -200, -100, 100, 200, 300, 2, 4, 6, 7, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'weekday', 'weekend', 'high', 'low', 'perci', 'f_station_rack', 't_station_rack', '-300 2', '-200 2', '-100 2', '100 2', '200 2', '300 2', '-300 4', '-200 4', '-100 4', '100 4', '200 4', '300 4', '-300 6', '-200 6', '-100 6', '100 6', '200 6', '300 6', '-300 7', '-200 7', '-100 7', '100 7', '200 7', '300 7']]
	x = sm.add_constant(x)
	pm = sm.GLM(y, x, family=sm.families.Poisson()).fit()
	print(pm.summary().as_latex())
	print("poisson regression's rmse value")
	print(sm.tools.eval_measures.rmse(y, pm.fittedvalues, axis=0))

	pr_predict(weather, x, y)

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
	df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	df1 = df.head(100000)
	df1.sort_index(inplace=True)
	#print(df1['Actual'])
	#print(df1['Predicted'])
	plt.plot(df1['Actual'], c="blue", label="actual", linewidth=2)
	plt.plot(df1['Predicted'], c="red", label="predicted", linewidth=2)
	plt.savefig(weather + "_pr_line.png")
	plt.clf()
	plt.cla()
	plt.close()


pr("summer")
pr("winter") 
