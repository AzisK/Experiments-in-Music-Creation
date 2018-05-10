import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv('gridsearch.csv')

def getHead():
	dfSorted = df.sort_values('rmse').head(20)
	print(dfSorted)

def getTail():
	dfSorted = df.sort_values('rmse').tail(20)
	print(dfSorted)

getHead()
getTail()