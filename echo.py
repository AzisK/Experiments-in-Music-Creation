# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.linalg as lin
import midiUtils as mu
from pathlib import Path
from operator import itemgetter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predictNotes(y, deg, noteLengths, outSize):
    mnotes = []
    pnotes = []

    for index, value in enumerate(y):
        if noteLengths[index] > 1:
            diff = np.power(abs(value), 1 + (deg - 1) / ((noteLengths[index] % 8) + 1)) - np.random.rand()
        elif noteLengths[index]:
            diff = np.power(abs(value), 1 + (deg - 1) / 2) - np.random.rand()
        else:
            diff = np.power(abs(value), deg) - np.random.rand()

        if diff > 0:
            if value > 0:
                pnotes.append((index, diff))
            elif value < 0:
                mnotes.append((index, diff))
        # if np.power(abs(value), deg) > np.random.rand():
        # if abs(value) > np.random.rand():

        #     if value > 0:
        #         y[index] = 1
        #     else:
        #         y[index] = -1
        # else:
        #     y[index] = 0

    y[:] = 0
    newNoteLengths = np.zeros((outSize))

    pnotes.sort(key=itemgetter(1), reverse=True)
    mnotes.sort(key=itemgetter(1), reverse=True)

    high = []
    if pnotes:
        for note in pnotes:
            if not high:
                high.append(note[0])
            if high:
                for h in high:
                    chord = abs(h - note[0])
                    if chord == 3 or chord == 4:
                        high.append(note[0])
            if len(high) == 6:
                break

    for note in high:
        y[note] = 1
        newNoteLengths[note] = 1
        noteLengths += 1
            
    high = []
    if mnotes:
        for note in mnotes:
            if not high:
                high.append(note[0])
            if high:
                for h in high:
                    chord = abs(h - note[0])
                    if chord == 3 or chord == 4:
                        high.append(note[0])
            if len(high) == 6:
                break        

    for note in high:
        y[note] = -1
        newNoteLengths[note] = 1
        noteLengths += 1

    for i, n in enumerate(newNoteLengths):
        if not n:
            noteLengths[i] = 0

def predict(minp=0, maxp=127, lr = 0.02, ins = 0.00023, sr = 0.02, regs=[0.1], degs=(1, 1, 1)):
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

    # for i in range(len(W)):
    #     for j in range(len(W[i])):
    #         if np.random.rand() < 0.1:
    #             W[i, j] = 0

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
            u = np.array([data[:, train]]).transpose()
            noteLengths = np.zeros((outSize))

            for t in range(test):
                inps = np.dot(Win, np.vstack((1, u)))

                if t == 0:
                    x = z

                x = (1 - lr) * x + lr * np.tanh(inps + np.dot(W, x))

                y = np.dot(Wout, np.vstack((1, u, x)))

                predictNotes(y, deg, noteLengths, outSize)

                Y[:, t] = y.transpose()

                # GENERATIVE:
                u = y

                # PREDICTIVE:
                # u = np.array([data[:, train + t]]).transpose()

            # Compute MEAN, RMSE & STANDARD DEVATION
            mean = Y.mean()

            diff = data[:, train: train + test + 1] - Y[:, 0 : test]
            # diff = data[:, train: train + test + 1]
            se = np.square(diff)
            rmse = np.mean(np.sqrt(se))

            coldiff = Y - np.array([Y.mean(1)]).transpose()
            std = np.mean(np.sqrt(np.square(coldiff)))
            print("MEAN: {0}, RMSE: {1}, STD: {2}, reg: {3}, deg: {4}".format(mean, rmse, std, r, deg))
            rers.append((r, mean, rmse, std, deg))

            # OUTPUT
            return Y

    # return rers

def gridSearch():
    print('Starting grid search for optimal values for the echo network...')
    gs = []
    for lr in np.linspace(0.02, 0.02, 1, endpoint=True):
    # for lr in [0.025]:
        for ins in np.linspace(0.000223, 0.00025, 3, endpoint=True):
        # for ins in [0.00002]:
            for sr in np.linspace(0.02, 0.02, 1, endpoint=True):
            # for sr in [0.1]:
                print('*')
                print('Calculating the error for:')
                print('*** lr={0}, ins={1}, sr={2}'.format(lr, ins, sr))
                regs = [1, 0.1]
                rers = predict(29, 91, lr, ins, sr, regs)
                for rer in rers:
                    reg, mean, rmse, std, deg = rer
                    gs.append((lr, ins, sr, reg, mean, rmse, std, deg))
                    # gs.append((lr, ins, sr, reg, mean, rmse, std))

                    gsd = pd.DataFrame(gs, columns=['lr', 'ins', 'sr', 'reg', 'mean', 'rmse', 'std', 'deg'])
                    # gsd = pd.DataFrame(gs, columns=['lr', 'ins', 'sr', 'reg', 'mean', 'rmse', 'std'])

    print('Grid search done!')
    gsd.to_csv('gridsearch.hand.3.csv', index=False)

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

# coldiff = getColdiff()
# print(coldiff)

# gridSearch()
out = predict(29, 91, degs=(1, 1, 1))
# print(out)
# out = predict(29, 91, regs=[1, 100], degs=(1, 1.4, 5))
# out = predict(29, 91, regs=[100], degs=(1.125, 1.125, 1))
tuples = mu.state2Tuples(out, 29, 60)
# mu.tuples2Midi(tuples, 'Midi1.05.mid')