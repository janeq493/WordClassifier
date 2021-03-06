import numpy as np


class Layer:
    def __init__(self, size, lastLayerSize, activationFunc='tanh', costFunction="no"):
        np.random.seed(9010)
        self.weights = np.random.random((size, lastLayerSize)) -0.5
        self.weightsDelta = np.zeros((size, lastLayerSize))
        self.error = np.zeros((size))
        self.input = np.zeros((lastLayerSize))
        self.values = np.zeros((size))
        self.valuesActivated = np.zeros((size))
        self.bias = np.ones((size)) -0.5
        self.biasDelta = np.zeros((size))

        self.activationFunc = activationFunc
        self.costFunction = costFunction

        self.lastCost = 0

    def feedForward(self, input):
        self.input = input
        self.values = np.dot(self.weights, self.input) + self.bias
        self.valuesActivated = self.activate(self.values)
        return self.valuesActivated

    def backProp(self, prevLayerError=None, label=None):
        if self.costFunction == "no":
            self.error = prevLayerError * self.activate(self.values, True)
        else:
            self.error = self.cost(self.valuesActivated, label, True) * \
                self.activate(self.values, True)
            self.updateCost(label)
            
        self.weightsDelta += np.outer(self.error, self.input)
        self.biasDelta += self.error
        return np.dot(self.weights.T, self.error)

    def update(self, batchSize, learningRate):
        self.weights -= self.weightsDelta * learningRate / batchSize
        self.bias -= self.biasDelta * learningRate / batchSize
        self.weightsDelta = self.weightsDelta *0.0
        self.biasDelta = self.biasDelta *0.0

    def activate(self, x, derv=False):
        if derv:
            if self.activationFunc == 'tanh':
                return 1 - np.tanh(x)**2
            elif self.activationFunc == 'sig' or self.activationFunc == 'sigmoid':
                return (np.tanh(x / 2) / 2 + 0.5) * (1 - np.tanh(x / 2) / 2 + 0.5)
            elif self.activationFunc.lower() == 'relu':
                x[x < 0] = 1/50
                x[x >= 0] = 1
                return x
        else:
            if self.activationFunc == 'tanh':
                return np.tanh(x)
            elif self.activationFunc == 'sig' or self.activationFunc == 'sigmoid':
                return np.tanh(x / 2) / 2 + 0.5
            elif self.activationFunc.lower() == 'relu':
                x[x<0] = x[x<0]/50
                return x

    def cost(self, x, label, derv=False):
        if derv:
            if self.costFunction.lower() == 'mse' or self.costFunction.lower() == 'mean squared error' or self.costFunction.lower() == 'meansquarederror':
                return (x - label)
        else:
            if self.costFunction.lower() == 'mse' or self.costFunction.lower() == 'mean squared error' or self.costFunction.lower() == 'meansquarederror':
                return 1 / 2 * (x - label)**2

    def updateCost(self,label):        
        self.lastCost+=np.linalg.norm(self.cost(self.valuesActivated, label)**2)
