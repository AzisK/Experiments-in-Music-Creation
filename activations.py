import matplotlib.pyplot as plt
import numpy as np

def plotSigmoid():
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    x = np.linspace(-10,10,100)
    plt.plot(x, sigmoid(x), 'b')
    plt.grid()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Sigmoid Function')
    plt.suptitle('Sigmoid')

    plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)

    plt.subplots_adjust(bottom=0.22)

    plt.show()

def plotTanh():
    x = np.linspace(-10,10,100)
    plt.plot(x, np.tanh(x), 'b')
    plt.grid()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Hyperbolic Tangent Function')
    plt.suptitle('Hyperbolic Tangent')

    plt.text(4, 0.8, r'$y=tanh(x)$', fontsize=15)

    plt.subplots_adjust(bottom=0.22)

    plt.show()

# plotSigmoid()
plotTanh()