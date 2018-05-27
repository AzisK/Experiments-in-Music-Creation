import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path

# df = pd.read_csv('gridsearch.edge0.csv')
# df120 = pd.read_csv('gridsearch120.csv')
# df = pd.read_csv('gridsearch.edge.02-.2.csv')
# df = pd.read_csv('gridsearch.edge1.csv')
# df = pd.read_csv('gridsearch.edge2.csv')
# df = pd.read_csv('gridsearch.edge3.csv')
# df = pd.read_csv('gridsearch.edge4.csv')
# df = pd.read_csv('gridsearch.edge5.csv')
# df = pd.read_csv('gridsearch.edge6.csv')
# df = pd.read_csv('gridsearch.edge7.csv')
# df = pd.read_csv('gridsearch.edge8.csv')
# df = pd.read_csv('gridsearch.edge10.csv')
# df = pd.read_csv('gridsearch.edge11.csv')
# df = pd.read_csv('gridsearch.sigm.1.csv')
# df = pd.read_csv('gridsearch.sigm.2.csv')
# df = pd.read_csv('gridsearch.sigm.3.csv')
# df = pd.read_csv('gridsearch.sigm.4.csv')
# df = pd.read_csv('gridsearch.sigm.5.csv')

def getCsvs():
  # Set the path
  path = Path('.')
  # Return all midis in the path
  return list(path.glob('gridsearch.edge*.csv'))
  # return list(path.glob('gridsearch.sigm*.csv'))

def loadCsvs():
	frames = []
	for csv in getCsvs():
		frames.append(pd.read_csv(csv, index_col=False))
	return pd.concat(frames)

# df = loadCsvs()

def getHead():
	dfSorted = df.sort_values('rmse').head(10)
	print(dfSorted)

def getTail():
	Df = df.loc[df['reg'] <= 10]
	dfSorted = Df.sort_values('rmse').tail(10)
	print(dfSorted)

def getRows(minp=0, maxp=40):
	minp = 260
	maxp = 300
	print('{0}-{1}'.format(minp, maxp))
	print(df.sort_values('rmse')[minp:maxp])

def groupSr():
	print('Grouped by sr')
	A = df.groupby('sr')['rmse'].min()
	print(A)

def groupIns():
	print('Grouped by ins')
	A = df.groupby('ins')['rmse'].min()
	print(A)

def groupLr():
	print('Grouped by lr')
	A = df.groupby('lr')['rmse'].min()
	print(A)

def groupReg():
	print('Grouped by reg')
	A = df.groupby('reg')['rmse'].min()
	print(A)

def groupInputAndRadius():
	A = df.groupby(['ins', 'sr'])['rmse'].min()
	print(A)

def plot3d():
	Df = df
	# Df = Df.loc[Df['lr'] == 0.025]
	Df = Df.groupby(['ins', 'sr'])['ins', 'sr' ,'rmse'].min()
	fig = plt.figure()
	ax = Axes3D(fig)
	X = Df.iloc[:,0].values
	Y = Df.iloc[:,1].values
	Z = Df.iloc[:,2].values
	surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=1, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel('Input scaling')
	ax.set_ylabel('Spectral radius')
	ax.set_zlabel('RMSE')
	plt.show()

def plotLr():
	A = df.groupby(['lr'])['lr' ,'rmse'].min()
	X = A.iloc[:,0].values
	Y = A.iloc[:,1].values
	mpl_fig = plt.figure()
	ax = mpl_fig.add_subplot(111)
	ax.plot(X, Y)
	ax.set_xlabel('Leaking rate')
	ax.set_ylabel('RMSE')
	plt.show()

def plotReg():
	A = df.groupby(['reg'])['reg' ,'rmse'].min()
	X = A.iloc[:,0].values
	Y = A.iloc[:,1].values
	mpl_fig = plt.figure()
	ax = mpl_fig.add_subplot(111)
	ax.plot(X, Y)
	ax.set_xlabel('Regularization')
	ax.set_ylabel('RMSE')
	ax.set_xscale('log')
	plt.show()

def plotIns():
	A = df.groupby(['ins'])['ins' ,'rmse'].min()
	X = A.iloc[:,0].values
	Y = A.iloc[:,1].values
	mpl_fig = plt.figure()
	ax = mpl_fig.add_subplot(111)
	ax.plot(X, Y)
	ax.set_xlabel('Input scaling')
	ax.set_ylabel('RMSE')
	plt.show()

def plotSr():
	A = df.groupby(['sr'])['sr' ,'rmse'].min()
	X = A.iloc[:,0].values
	Y = A.iloc[:,1].values
	mpl_fig = plt.figure()
	ax = mpl_fig.add_subplot(111)
	ax.plot(X, Y)
	ax.set_xlabel('Spectral radius')
	ax.set_ylabel('RMSE')
	plt.show()

def getLrRange():
	A = df.groupby(['lr'])['lr' ,'rmse'].min()
	print(A)

def getSrRange():
	A = df.groupby(['sr'])['sr' ,'rmse'].min()
	print(A)

def getInsRange():
	A = df.groupby(['ins'])['ins' ,'rmse'].min()
	print(A)

def getRegRange():
	A = df.groupby(['reg'])['reg' ,'rmse'].min()
	print(A)	


# print('Quant 60')
getHead()
getTail()
# groupIns()
# groupSr()
# groupLr()
# groupReg()
# groupInputAndRadius()
# plot3d()
# plotLr()
# plotReg()
# plotIns()
# plotSr()
# getLrRange()
# getSrRange()
# getInsRange()
# getRegRange()

# print('Quant 120')
# getHead(df120)
# getTail(df120)