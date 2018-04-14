import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('error.csv')

def getError():
	A = np.nan_to_num(df.error)
	A[A > 0.02] = 0
	A = A[np.where(A > 0)]

	plt.hist(A, bins=150)
	plt.show()

def getErrorRough():
	A = np.nan_to_num(df.error)
	A[A > 0.2] = 0
	A = A[np.where(A > 0)]

	plt.hist(A, bins=150)
	plt.show()

def getTop():
	dfSorted = df.sort_values('error').head(16)
	print(dfSorted)

def groupInput():
	# A = df.replace([np.inf, -np.inf], np.nan)
	# A = A.fillna(1000000)
	A = df.groupby('input scaling')['error'].mean()
	print(A)

groupInput()