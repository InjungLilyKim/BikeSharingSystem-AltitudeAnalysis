# Bike sharing system analysis by python statsmodels
# % Multiple Linear Regression model with prediction 
# % Poisson Regression model with prediction
# Author: Injung Kim
# Last modified: 5/19/2020

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

#from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import KFold 
from sklearn import metrics 
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression 
#from sklearn.model_selection import KFold 

import warnings
warnings.filterwarnings("ignore")

statsmodels.__version__

# multiple linear regression
def mlr(weather):
	# Load dataset
	df = pd.read_csv('./modelling_' + weather + '.csv')
	y = df['daily_usage']
	df = pd.concat((df, pd.get_dummies(df['dis_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['alti_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['alti_dis_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['dayofweek'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['week_wknd'])), axis=1)
	#pd.set_option('display.max_columns', None)
	#pd.options.display.max_rows=10
	#print list(df.columns.values)
	#print (df)

	print("--------------------------------------")
	print("daily usage with distance/altitude range")
	# create a linear regression model 
	print("--------------------------------------")
	print("linear regression with a constant")
	print("--------------------------------------")
	x = df[[-300, -200, -100, 100, 200, 300, 2, 4, 6, 7, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'weekday', 'weekend', 'high', 'low', 'perci', 'f_station_rack', 't_station_rack', '-300 2', '-300 4', '-300 6', '-300 7', '-200 2', '-200 4', '-200 6', '-200 7', '-100 2', '-100 4', '-100 6', '-100 7', '100 2', '100 4', '100 6', '100 7', '200 2', '200 4', '200 6', '200 7', '300 2', '300 4', '300 6', '300 7']]
	#lm = smf.ols("daily_usage ~ C(dis_range) + C(alti_range) + C(alti_dis_range) + high + low + perci + C(dayofweek) + C(week_wknd) + f_station_rack + t_station_rack", data=df).fit()
	x = sm.add_constant(x)
	lm = sm.OLS(y, x).fit()
	print(lm.summary().as_latex())
	#print(lm.params)
	print("********************************")
	print(sm.tools.eval_measures.rmse(y, lm.fittedvalues, axis=0))
	print("linear regression's rsquared value")
	print (lm.rsquared)
	print("********************************")
	df.plot(kind='scatter', x='index', y='daily_usage')
	plt.plot(y, c='green', linewidth=2)
	plt.title('Multiple Linear Regression')
	plt.savefig(weather + '_mlr_100.png')
	plt.clf()
	plt.cla()
	plt.close()
	print("********************************")

	mlr_predict(weather, 91, x, y)
	mlr_predict(weather, 82, x, y)
	mlr_predict(weather, 73, x, y)

# multiple linear regression prediction
def mlr_predict(weather, ratio, x, y):
	##################################
	#here is for train/test ration 
	size = 0
	if ratio == 91:
		size = 0.1
	elif ratio == 82:
		size = 0.2
	else:
		size = 0.3
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = size)
	lr = LinearRegression(fit_intercept=True)
	#train r2, rmse
	print("MLR " + str(ratio) + " train r2 and rmse")
	lm_train = lr.fit(x_train, y_train)
	print(r2_score(y_train, lm_train.predict(x_train)))
	print(np.sqrt(metrics.mean_squared_error(y_train, lm_train.predict(x_train))))
	#test r2, rmse
	print("MLR " + str(ratio) + " test r2 and rmse")
	lm_test = lr.fit(x_train, y_train)
	print(r2_score(y_test, lm_test.predict(x_test)))
	print(np.sqrt(metrics.mean_squared_error(y_test, lm_test.predict(x_test))))
	print("********************************")
	#draw the plot - bar, line types
	y_pred = lm_test.predict(x_test)
	df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	df1 = df.head(50)
	df1.sort_index(inplace=True)
	df1.plot(kind='bar',figsize=(16,10))
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	#plt.savefig(weather + "_" + str(ratio) + '_mlr_bar.png')
	#print(df1)
	#print(df1.index)
	#print(y_pred)
	plt.clf()
	plt.cla()
	plt.close()
	df1 = df.head(300)
	df1.sort_index(inplace=True)
	plt.plot(df1['Actual'], c="blue", label="actual", linewidth=2)
	plt.plot(df1['Predicted'], c="red", label="predicted", linewidth=2)
	plt.savefig(weather + "_" + str(ratio) + '_mlr_line.png')
	plt.clf()
	plt.cla()
	plt.close()
	print("********************************")

# poisson regression
def pr(weather, pr_type):
	# create a poisson regression model
	print("\--------------------------------------")
	print("\poisson regression with constant")
	print("\--------------------------------------")
	#pd.set_option('display.max_columns', None)
	#df = pd.read_csv('./modelling_' + weather + '.csv')
	df = pd.read_csv('./modelling_all_season.csv')
	df = pd.concat((df, pd.get_dummies(df['dis_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['alti_range'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['month'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['alti_dis_month'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['dayofweek'])), axis=1)
	df = pd.concat((df, pd.get_dummies(df['week_wknd'])), axis=1)
	#pd.options.display.max_rows=10
	y = df['daily_usage']
	if pr_type == 1:
		x = df[[-300, -200, -100, 100, 200, 300, 2, 4, 6, 7, 'Jun', 'Feb', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'weekday', 'weekend', 'high', 'low', 'perci', 'f_station_rack', 't_station_rack']]
		print("\pr type: range and other values")
	elif pr_type == 2:
		x = df[['high', 'low', 'perci', 'f_station_rack', 't_station_rack', '-300 2 Jun', '-200 2 Jun', '-100 2 Jun', '100 2 Jun', '200 2 Jun', '300 2 Jun', '-300 4 Jun', '-200 4 Jun', '-100 4 Jun', '100 4 Jun', '200 4 Jun', '300 4 Jun', '-300 6 Jun', '-200 6 Jun', '-100 6 Jun', '100 6 Jun', '200 6 Jun', '300 6 Jun', '-300 7 Jun', '-200 7 Jun', '-100 7 Jun', '100 7 Jun', '200 7 Jun', '300 7 Jun']]
		print("\pr type: weather and interaction")
	else:
		x = df[['high', 'low', 'perci', 'f_station_rack', 't_station_rack', '-300 2 Feb', '-200 2 Feb', '-100 2 Feb', '100 2 Feb', '200 2 Feb', '300 2 Feb', '-300 4 Feb', '-200 4 Feb', '-100 4 Feb', '100 4 Feb', '200 4 Feb', '300 4 Feb', '-300 6 Feb', '-200 6 Feb', '-100 6 Feb', '100 6 Feb', '200 6 Feb', '300 6 Feb', '-300 7 Feb', '-200 7 Feb', '-100 7 Feb', '100 7 Feb', '200 7 Feb', '300 7 Feb']]
	x = sm.add_constant(x)
	pm = sm.GLM(y, x, family=sm.families.Poisson()).fit()
	print(pm.summary().as_latex())
	print("poisson regression's rmse value")
	print(sm.tools.eval_measures.rmse(y, pm.fittedvalues, axis=0))
	"""
	print("\n********************************")
	df.plot(kind='scatter', x='index', y='daily_usage')
	plt.plot(df['index'], pm.predict(x), c='green', linewidth=2)
	plt.title('Poisson Regression')
	plt.savefig(weather + "_" + str(pr_type) + '_pr_100.png')
	plt.clf()
	plt.cla()
	plt.close()
	df1 = df.head(300)
	df1.sort_index(inplace=True)
	print("\n********************************")
	"""

	pr_predict(weather, pr_type, 91, x, y)
	pr_predict(weather, pr_type, 82, x, y)
	pr_predict(weather, pr_type, 73, x, y)

# poisson regression prediction
def pr_predict(weather, pr_type, ratio, x, y):
	#############################
	#here is for train/test ratio 
	size = 0
	if ratio == 91:
		size = 0.1
	elif ratio == 82:
		size = 0.2
	else:
		size = 0.3
	#train r2, rmse
	print("PR " + str(ratio) + " train r2 and rmse")
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = size)
	pm_train = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
	print(np.sqrt(metrics.mean_squared_error(y_train, pm_train.predict(x_train))))
	#test r2, rmse
	print("PR " + str(ratio) + " test r2 and rmse")
	pm_test = sm.Poisson(y_train, x_train).fit()
	print(np.sqrt(metrics.mean_squared_error(y_test, pm_test.predict(x_test))))
	print("\n********************************")
	y_pred = pm_test.predict(x_test)
	df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	"""
	df1 = df.head(50)
	df1.sort_index(inplace=True)
	df1.plot(kind='bar',figsize=(16,10))
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.savefig(weather + "_" + str(pr_type) + "_" + str(ratio) + '_pr_bar.png')
	print(df1)
	#print(df1.index)
	print(y_pred)
	plt.clf()
	plt.cla()
	plt.close()
	"""
	df1 = df.head(100000)
	df1.sort_index(inplace=True)
	#print(df1['Actual'])
	#print(df1['Predicted'])
	plt.plot(df1['Actual'], c="blue", label="actual", linewidth=2)
	plt.plot(df1['Predicted'], c="red", label="predicted", linewidth=2)
	#plt.savefig(weather + "_" + str(pr_type) + "_" + str(ratio) + '_pr_line.png')
	plt.savefig("all_season_" + str(pr_type) + "_" + str(ratio) + '_pr_line.png')
	plt.clf()
	plt.cla()
	plt.close()

def kford_c_vali():
	##################################
	#here is for train/test and K-fold & cross validation
	df = pd.concat([x, y], axis=1)
	scores = np.zeros(5)
	cv = KFold(5, shuffle=True, random_state=0)
	for i, (idx_train, idx_test) in enumerate(cv.split(df)):
		df_train = df.iloc[idx_train]
		df_test = df.iloc[idx_test]
		model = sm.OLS("MEDV ~ ", data=df_train)
		result = model.fit()
		pred = result.predict(df_test)

		print("\n********************************")
		print(result.rsquared)
		print("\n********************************")

#mlr("summer")
#pr("summer", 1)
#pr("summer", 2)
#mlr("winter")
#pr("winter", 1)
#pr("winter", 2)
pr("winter", 3) 
