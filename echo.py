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


class EchoNetwork:
    def __init__(self, data, resSize=1000):
        self.data = data
        self.resSize = resSize
        print('Initiating Echo State Network...)')
        print(f'Reservoir size is: {resSize}')

    def predict(self, minp=0, maxp=127, lr=0.02, ins=0.00023, sr=0.02, regs=[0.1], degs=(1, 1, 1)):
        inSize = outSize = maxp + 1 - minp

        # Pick only the necessary elements
        # data = data[:, minp : maxp + 1]

        # np.random.seed(123)

        lengths = np.shape(self.data)[1]
        train = round(lengths * 0.9)
        test = lengths - train
        init = 300

        Win = (np.random.rand(self.resSize, 1 + inSize) - 0.5) * ins
        W = np.random.rand(self.resSize, self.resSize) - 0.5

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
        X = np.zeros((train - init, 1 + inSize + self.resSize))

        # Set the corresponding target matrix directly
        Yt = self.data[:, init + 1: train + 1]

        # Run the reservoir with the data and collect X
        x = np.zeros((self.resSize, 1))

        print('Starting training...')
        for t in range(train):
            u = np.array([self.data[:, t]]).transpose()
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

            inv = lin.inv(selfie + r * np.eye(1 + inSize + self.resSize))
            fdback = np.dot(Yt, X)

            print('Computing Wout...')
            Wout = np.dot(fdback, inv)

            # Generate output matrix
            Y = np.zeros((outSize, test))

            for deg in np.linspace(degs[0], degs[1], degs[2], endpoint=True):
                u = np.array([self.data[:, train]]).transpose()
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

                diff = self.data[:, train: train + test + 1] - Y[:, 0:test]
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

    def getColdiff(self):
        my_file = Path('coldiff.npy')
        if not my_file.is_file():
            coldiff = np.array([self.data.mean(1)]).transpose()
            np.save('coldiff.npy', coldiff)
        else:
            coldiff = np.load('coldiff.npy')
        return coldiff

    def gridSearch(self):
        print('Starting grid search for optimal values for the echo network...')
        gs = []
        for lr in np.linspace(0.02, 0.02, 1, endpoint=True):
            for ins in np.linspace(0.000223, 0.00025, 3, endpoint=True):
                for sr in np.linspace(0.02, 0.02, 1, endpoint=True):
                    print('*')
                    print('Calculating the error for:')
                    print('*** lr={0}, ins={1}, sr={2}'.format(lr, ins, sr))
                    regs = [1, 0.1]
                    rers = self.predict(29, 91, lr, ins, sr, regs)
                    for rer in rers:
                        reg, mean, rmse, std, deg = rer
                        gs.append((lr, ins, sr, reg, mean, rmse, std, deg))

                        gsd = pd.DataFrame(gs, columns=['lr', 'ins', 'sr', 'reg', 'mean', 'rmse', 'std', 'deg'])

        print('Grid search done!')
        gsd.to_csv('gridsearch.hand.3.csv', index=False)


def load_midi_state(quant=60, force=False) -> np.ndarray:
    file_name = 'MidiStateMatrix.npy'
    if force:
        df = mu.loadPieces(force=force)
        mu.quantizeDf(df, quant=quant)
        data = mu.toStateMatrix(df, 29, 91, quant=quant)
        np.save(file_name, data)
        return data
    else:
        return np.load(file_name)


if __name__ == '__main__':
    QUANT = 60

    midi_state = load_midi_state(quant=QUANT, force=True)

    echo_network = EchoNetwork(midi_state)

    generated_state = echo_network.predict(29, 91, degs=(1, 1, 1))

    tuples = mu.state2Tuples(generated_state, 29, quant=QUANT)
    mu.tuples2Midi(tuples, 'MidiOriginal.mid')
