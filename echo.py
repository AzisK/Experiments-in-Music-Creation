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

def predict(minp=0, maxp=127, lr = 0.5, ins = 0.65, sr = 0.75, reg=1, size=1000, noises=False):
    inSize = outSize = maxp + 1 - minp
    resSize = int(size)
    print('Reservoir size is: {}'.format(resSize))

    print('Noise is: {}'.format(noises))

    # Pick only the necessary elements
    # data = data[:, minp : maxp + 1]

    # np.random.seed(123)

    lengths = np.shape(data)[1]
    train = round(lengths * 0.8)
    test = lengths - train
    init = 300

    Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * ins
    W = np.random.rand(resSize, resSize) - 0.5

    print('Computing spectral radius...')

    rhoW = max(abs(np.linalg.eigvals(W)))
    print('Maximum eigen-value: {}'.format(rhoW))

    W *= sr / rhoW
    print('Done! Spectral radius / rhoW: {0}'.format(sr / rhoW))

    # Allocated memory for the design (collected states) matrix
    X = np.zeros((train - init, 1 + inSize + resSize))

    # Set the corresponding target matrix directly
    Yt = data[:, init + 1 : train + 1]

    # Run the reservoir with the data and collect X
    x = np.zeros((resSize, 1))

    print('Starting training...')
    for t in range(train):
        u = np.array([data[:, t]]).transpose()
        inps = np.dot(Win, np.vstack((1, u)))
        x = (1 - lr) * x + lr * np.tanh(inps + np.dot(W, x))

        if noises:
            noise = np.random.rand(resSize, 1)
            x += noise

        if t >= init:
            X[t - init, :] = np.vstack((1, u, x)).transpose()

    print('Iterations done :)')

    # Train the output
    inv = lin.inv(np.dot(X.T, X) + reg * np.eye(1 + inSize + resSize))
    fdback = np.dot(Yt, X)

    print('Computing Wout...')
    Wout = np.dot(fdback, inv)

    print('Training done!')

    # Generate output matrix
    Y = np.zeros((outSize, test))

    u = np.array([data[:, train]]).transpose()

    # noteLength = 0

    print('Starting testing!')
    for t in range(test):
        inps = np.dot(Win, np.vstack((1, u)))
        x = (1 - lr) * x + lr * np.tanh(inps + np.dot(W, x))

        if noises:
            noise = np.random.rand(resSize, 1)
            x += noise

        y = np.dot(Wout, np.vstack((1, u, x)))

        # generateNotes(Y, t, y, noteLength)

        Y[:, t] = y.transpose()

        # GENERATIVE:
        # u = y

        # PREDICTIVE:
        u = np.array([data[:, train + t]]).transpose()

    print('Testing done!')

    # Compute MEAN, RMSE & STANDARD DEVATION
    mean = Y.mean()

    diff = data[:, train: train + test + 1] - Y[:, 0 : test]
    se = np.square(diff)
    rmse = np.mean(np.sqrt(se))

    coldiff = Y - np.array([Y.mean(1)]).transpose()
    std = np.mean(np.sqrt(np.square(coldiff)))
    print("MEAN: {0}, RMSE: {1}, STD: {2}".format(mean, rmse, std))
    return mean, rmse, std

    # OUTPUT
    # return Y.T

df = mu.loadPieces()
mu.quantizeDf(df)
data = mu.toStateMatrix(df, 29, 91)

def gridSearch():
    print('Starting grid search for optimal values for the echo network...')
    gs = []
    for lr in np.linspace(0.25, 1, 4, endpoint=True):
        for ins in np.linspace(0.2, 2, 4, endpoint=True):
            for sr in np.linspace(0.2, 2, 4, endpoint=True):
                for reg in np.logspace(-2, -2, 5, endpoint=True):
                    for size in np.linspace(1000, 2000, 2, endpoint=True):
                        for noises in [True, False]:
                            print('*')
                            print('Calculating the error for:')
                            print('*** lr={0}, ins={1}, sr={2}, reg={3}, size={4}, noises={5}'.format(lr, ins, sr, reg, size, noises))

                            gs.append((lr, ins, sr, reg, size, noises, predict(29, 91, lr, ins, sr, reg, size, noises)))
                            gsd = pd.DataFrame(gs, columns=['lr', 'ins', 'sr', 'reg', 'size', 'noises', 'error'])

    print('Grid search done!')
    gsd.to_csv('gridsearch.csv')

gridSearch()
# mean, rmse, std = predict(29, 91)