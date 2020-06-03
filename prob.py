# Bike sharing system analysis by python statsmodels
# % Poisson Regression model to get actual distribution for the predicted number of trips 
# Author: Injung Kim
# Last modified: 6/2/2020

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

from scipy import stats

from patsy import dmatrices
import warnings
warnings.filterwarnings("ignore")

statsmodels.__version__

# draw prediction plot 
def prob_plot(d):
	#n, bins, patches = plt.hist(x=d, bins='auto', color="blue")
	print(len(d))
	"""
	plt.hist(x=d, bins='auto', color="blue")
	#plt.grid(axis='y')
	plt.xlabel('The number of trips')
	plt.ylabel('PMF')
	
	plt.savefig("model_prob.png")
	plt.clf()
	plt.cla()
	plt.close()
	"""

#main
weather = ["summer", "winter"]

#model numbers
summerWinterModels = [9]

# run summer/winter seasons model
for w in weather:
	print("\ 1)season:")
	print(w)
	print("\ 2)altitude range and distance real value")
	print("\ 3)include a loop feature")

	df = pd.read_csv('./modelling_9_' + w + '.csv')
		
	# create a formula for each model
	formula ="""daily_usage~C(dayofweek)+high_real+low_real+perci_real+f_station_rack+t_station_rack+distance_real*C(alti_range)+C(loop)"""
	y, x = dmatrices(formula, df, return_type='dataframe')

	pm = sm.GLM(y, x, family=sm.families.Poisson()).fit()
	#pm = sm.Poisson(y, x).fit()
	#predicted = pm.predict()[:,None]
	predicted = pm.predict(x)

	#counts = np.atleast_2d(np.arange(0, np.max(pm.model.endog)+1))

	if w == "summer":
		for i in range(32):
			prob_arr = stats.poisson.pmf(i, predicted)
			print("the number of trips ", i)
			#print("avg cdf value is ", stats.poisson.cdf(i,predicted).mean())
			print("avg pmf value is ", round(prob_arr.mean(), 4))
			print("max pmf value is ", round(np.max(prob_arr), 4))
	elif w == "winter":
		for i in range(10):
			prob_arr = stats.poisson.pmf(i, predicted)
			print("the number of trips ", i)
			#print("avg cdf value is ", stats.poisson.cdf(i,predicted).mean())
			print("avg pmf value is ", round(prob_arr.mean(), 4))
			print("max pmf value is ", round(np.max(prob_arr), 4))

	#prob_plot(prob_arr)

