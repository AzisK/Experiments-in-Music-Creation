# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.linalg as lin
import midiUtils as mu

def predictNotes(y):
    for index, value in enumerate(y):
        if value > np.random.rand():
            y[index] = 1
        else:
            y[index] = 0

def generateNotes(Y, t, y, noteLength):
    degree = 0.8

    if value > np.random.rand() ** degree:
        y[index] = 1
    elif t > 0 and Y[index, t-1] == 1:
        noteLength += 1
        addition = (1 - degree) / noteLength + degree

        if value > np.random.rand() ** addition:
            y[index] = 1
    else:
        y[index] = 0
        noteLength = 0

def predict(lr = 0.8, ins = 1, sr = 1.25, reg=1):
    inSize = outSize = 128
    resSize = 1000

    np.random.seed(123)

    trainLen = round(len(data) * 0.8)
    testLen = len(data) - trainLen
    initLen = 120

    Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * ins
    W = np.random.rand(resSize, resSize) - 0.5 

    print('Computing spectral radius...')
    rhoW = max(abs(lin.eig(W)[0]))

    W *= sr / rhoW 
    print('Done! Spectral radius / rhoW: {0}'.format(sr / rhoW))

    # Allocated memory for the design (collected states) matrix
    X = np.zeros((1 + inSize + resSize, trainLen - initLen))

    # Set the corresponding target matrix directly
    Yt = data[initLen + 1 : trainLen + 1]

    # Run the reservoir with the data and collect X
    x = np.zeros((resSize, 1))

    print('Starting training...')
    for t in range(trainLen):
        u = data[t][:].reshape(-1, 1)
        x = (1 - lr) * x + a * np.tanh( np.dot( Win, np.vstack((1, u)) ) + np.dot( W, x ) )
        if t >= initLen:
            X[:, t - initLen] = np.vstack((1, u, x))[:, 0]

    # Train the output
    X_T = X.T
    Wout = np.dot( np.dot(Yt.T, X_T), lin.inv( np.dot(X, X_T) + reg * np.eye(1 + inSize + resSize)))
    print('Training done!')

    # Generate output matrix
    Y = np.zeros((outSize, testLen))

    u = data[trainLen].reshape(-1, 1)

    # noteLength = 0

    print('Starting testing!')
    for t in range(testLen):
        x = (1 - a) * x + a * np.tanh(np.dot( Win, np.vstack((1, u)) ) + np.dot(W, x))
        y = np.dot( Wout, np.vstack((1, u, x)) )

        # generateNotes(Y, t, y, noteLength)

        Y[:,t] = y.T
        # Generative mode:
        # u = y
        # This would be a predictive mode:
        u = data[trainLen + t].reshape(-1, 1)
    print('Testing done!')

    # Compute MSE for the first errorLen time steps
    se = np.square(data[trainLen: trainLen + testLen] - Y[:, 0 : testLen].T)
    rmse = np.mean(np.sqrt(se))
    print("Error: {}".format(rmse))
    return rmse
    # return Y.T

df = mu.loadPieces()
data = mu.toStateMatrix(df)

def gridSearch():
    print('Starting grid search for optimal values for the echo network...')
    gs = []
    for lr in np.linspace(0.25, 1, 4, endpoint=True):
        for ins in np.linspace(0.2, 2, 4, endpoint=True):
            for sr in np.linspace(0.2, 2, 4, endpoint=True):
                for reg in np.logspace(-2, -2, 5, endpoint=True):
                    print('*')
                    print('Calculating the error for lr={0}, ins={1}, sr={2}'.format(lr, ins, sr, reg))
                    gs.append((lr, ins, sr, reg, predict(lr, ins, sr, reg)))
                    gsd = pd.DataFrame(gs, columns=['lr', 'ins', 'sr', 'reg', 'error'])

    print('Grid search done!')
    gsd.to_csv('gridsearch.csv')

gridSearch()
# error = predict()
# print(error)