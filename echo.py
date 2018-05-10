# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.linalg as lin
import midiUtils as mu
from pathlib import Path

def predictNotes(y, deg):
    for index, value in enumerate(y):
        if coldiff[index] > np.power(value, deg):
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

def predict(minp=0, maxp=127, lr = 0.75, ins = 0.2, sr = 0.2, regs=[100], degs=(1, 1, 1)):
    inSize = outSize = maxp + 1 - minp
    resSize = 1000
    print('Reservoir size is: {}'.format(resSize))

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

        if t >= init:
            X[t - init, :] = np.vstack((1, u, x)).transpose()

    print('Iterations done :)')

    z = x

    # Train the output
    selfie = np.dot(X.T, X)
    
    rers = []

    for r in regs:
        print('Starting testing for regularization={}'.format(r))

        inv = lin.inv(selfie + r * np.eye(1 + inSize + resSize))
        fdback = np.dot(Yt, X)

        print('Computing Wout...')
        Wout = np.dot(fdback, inv)

        # Generate output matrix
        Y = np.zeros((outSize, test))

        for deg in np.linspace(degs[0], degs[1], degs[2], endpoint=True):
            print('Quantizing the notes with the degree of: {}'.format(deg))

            u = np.array([data[:, train]]).transpose()
            # noteLength = 0

            for t in range(test):
                inps = np.dot(Win, np.vstack((1, u)))

                if t == 0:
                    x = z

                x = (1 - lr) * x + lr * np.tanh(inps + np.dot(W, x))

                y = np.dot(Wout, np.vstack((1, u, x)))

                predictNotes(y, deg)

                Y[:, t] = y.transpose()

                # GENERATIVE:
                u = y

                # PREDICTIVE:
                # u = np.array([data[:, train + t]]).transpose()

            # Compute MEAN, RMSE & STANDARD DEVATION
            mean = Y.mean()

            diff = data[:, train: train + test + 1] - Y[:, 0 : test]
            se = np.square(diff)
            rmse = np.mean(np.sqrt(se))

            coldiff = Y - np.array([Y.mean(1)]).transpose()
            std = np.mean(np.sqrt(np.square(coldiff)))
            print("MEAN: {0}, RMSE: {1}, STD: {2}, reg: {3}, deg: {4}".format(mean, rmse, std, r, deg))
            rers.append((r, mean, rmse, std, deg))

            # OUTPUT
            # return Y

    return rers

def gridSearch():
    print('Starting grid search for optimal values for the echo network...')
    gs = []
    for lr in np.linspace(0.25, 1, 4, endpoint=True):
        for ins in np.linspace(0.2, 2, 4, endpoint=True):
            for sr in np.linspace(0.2, 2, 4, endpoint=True):
                print('*')
                print('Calculating the error for:')
                print('*** lr={0}, ins={1}, sr={2}'.format(lr, ins, sr))
                regs = [0.01, 0.1, 1, 10, 100]
                rers = predict(29, 91, lr, ins, sr, regs)
                for rer in rers:
                    reg, mean, rmse, std, deg = rer
                    gs.append((lr, ins, sr, reg, mean, rmse, std, deg))
                    gsd = pd.DataFrame(gs, columns=['lr', 'ins', 'sr', 'reg', 'mean', 'rmse', 'std', 'deg'])

    print('Grid search done!')
    gsd.to_csv('gridsearch.csv')

def getColdiff():
    my_file = Path('coldiff.npy')
    if not my_file.is_file():
        coldiff = np.array([data.mean(1)]).transpose()
        np.save('coldiff.npy', coldiff)
    else:
        coldiff = np.load('coldiff.npy')
    return coldiff

df = mu.loadPieces()
mu.quantizeDf(df)
data = mu.toStateMatrix(df, 29, 91)
coldiff = getColdiff()

# gridSearch()
# out = predict(29, 91, degs=(1, 1.5, 5))
out = predict(29, 91, regs=[1, 100], degs=(1, 2, 3))
# tuples = mu.state2Tuples(out, 29)
# mu.tuples2Midi(tuples)