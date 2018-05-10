import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

df = pd.read_csv('error.csv')

def getError():
	A = np.nan_to_num(df.error)
	A[A > 0.02] = 0
	A = A[np.where(A > 0)]

	plt.hist(A, bins=150)
	plt.xlabel('Error')
	plt.ylabel('Times')
	plt.show()

def getErrorRough():
	A = np.nan_to_num(df.error)
	A[A > 0.2] = 0
	A = A[np.where(A > 0)]

	plt.hist(A, bins=150)
	plt.xlabel('Error')
	plt.ylabel('Times')
	plt.show()

def getTop():
	dfSorted = df.sort_values('error').head(16)
	print(dfSorted)

def groupInput():
	A = df.groupby('input scaling')['error'].mean()
	A.to_csv('errorGroupByInput.csv')

def groupInputAndRadius():
	A = df.groupby(['input scaling', 'spectral radius'])['error'].mean()
	A.to_csv('errorGroupByInputAndRadius.csv')

def plot3d(csv):
	df = pd.read_csv(csv)
	df.loc[df['error'] > 0.1, 'error'] = 0.1
	fig = plt.figure()
	ax = Axes3D(fig)
	X = df.iloc[:,0].values
	Y = df.iloc[:,1].values
	Z = df.iloc[:,2].values
	surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=1, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel('Input scaling')
	ax.set_ylabel('Spectral radius')
	ax.set_zlabel('Mean error')
	plt.show()

# getError()
# getErrorRough()
# groupInput()
# groupInputAndRadius()
plot3d('errorGroupByInputAndRadius.csv')
