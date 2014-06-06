'''
Tenzin Rigden and Christopher Winter

Uses the backwards propogation algorithm to build a neural network. This particular code is designed to be used for optical character recognition.
'''

import math

class inputNode:
    def __init__(self,inputValue):
        self.value = inputValue

    def setValue(self,value):
        self.value = value

    def getValue(self):
        return self.value

'''
Still need to implement the calculate output and calculate input functions.
'''
class hiddenNode:
    def __init__(self,numInputs):
        self.weights = [0] * numInputs
        self.weights[0] = 1
        self.inputs = [0] * numInputs
        self.inputs[0] = 1
        self.output = 0
<<<<<<< HEAD
        self.delta = 0

    def setDelta(self, newDelta):
        self.delta = newDelta

    def getDelta(self):
        return self.delta
=======
        self.weightedInput = 0
>>>>>>> FETCH_HEAD

    def setWeight(self,weight,wieghtIndex):
        self.weights[weightIndex] = weight

    def setInput(self,input,inputIndex):
        self.inputs[inputIndex] = input

    def batchSetWeight(self,weights):
        for i in range(len(self.weights)):
            self.setWeight(weights[i],i)
        return self.weights

    def batchSetInputs(self,inputs):
        for i range(len(self.weights)):
            self.setInput(inputs[i],i)

    def calculateWeightedInputs(self):
        weightedInput = 0 # This is the inj notation in the book
        for i in range(len(self.weights)):
            weightedInput = weightedInput + (self.weights[i] * self.inputs[i])
        self.weightedInput = weightedInput
        return weightedInput

    def getWeightedInput(self):
        return self.weightedInput

    def calculateOutput(self):
        return 1

    def setOutput(self,output):
        self.output = output

    def getWeights(self):
        return self.weights

    def getInputs(self):
        return self.inputs

'''
Still need to implement the calculate output and calculate input functions.
'''
class outputNode:
    def __init__(self,numHiddenNodes):
        self.weights = [0] * numHiddenNodes
        self.weight[0] = 1
        self.inputs = [0] * numHiddenNodes
        self.input[0] = 1
        self.output = 0
<<<<<<< HEAD
        self.delta = 0

    def setDelta(self, newDelta):
        self.delta = newDelta

    def getDelta(self):
        return self.delta
=======
        self.weightedInput = 0
>>>>>>> FETCH_HEAD

    def setWeight(self,weight,wieghtIndex):
        self.weights[weightIndex] = weight

    def batchSetWeight(self,weights):
        for i in range(len(self.weights)):
            self.setWeight(weights[i],i)
        return self.weights

    def batchSetInputs(self,inputs):
        for i range(len(self.weights)):
            self.setInput(inputs[i],i)

    def setInput(self,input,inputIndex):
        self.inputs[inputIndex] = input

    def calculateWeightedInputs(self):
        weightedInput = 0 # This is the inj notation in the book
        for i in range(len(self.weights)):
            weightedInput = weightedInput + (self.weights[i] * self.inputs[i])
        self.weightedInput = weightedInput
        return weightedInput

    def getWeightedInput(self):
        return self.weightedInput

    def calculateOutput(self):
        return 1

    def setOutput(self,output):
        self.output = output

    def getWeights(self):
        return self.weights

    def getInputs(self):
        return self.inputs


def sigmoidFunction(x):
    return 1 / (1 + math.exp(x))

def sigmoidDerivativeFunction(x):
    return sigmoidFunction(x) * (1 - sigmoidFunction(x))

def setInitialWeights(network):
    intialWeight = 0.5
    initialWeightsHidden = [initialWeight] * len(hidden)
    initialWeightsOutputs = [intialWeight] * len(outputs)
    for node in network[1]:
        node.batchSetWeights(initialWeightsHidden)
    for node in network[2]:
        node.batchSetWeights(initialWeightsOutputs)

    return None

def

'''
Main algorithm.
Examples is a list of dictionaries, each dictionary is one example. The key is the pixel, and the value is the value of that pixel.
'''
def backPropLearning(examples,network):
    inputs = examples[0]
    hidden = examples[1]
    outputs = examples[2]

    condition = True
    while condition: # This is the repeat in the pseudocode that I'm not really sure about

        # Sets the weights to all be some small initial value
        setInitialWeights(network)

        for example in examples:
            # set the inputs
            for i in range(len(example[0].values())):
                inputs[i].setValue(example[0][i])

            # Set up all the hidden nodes

            for node in hidden:
                for inputNumber in range(len(inputs)):
                    node.setInput(inputs[inputNumber],inputNumber+1) # The "+1" is because of the dummy input
                weightedInputs = node.calculateWeightedInputs()
                output = sigmoidFunction(weightedInputs)
                node.setOutput(output)






    return True

def main():
    numHiddenNodes = 96
    numOutputs = 10

    outputVector = [0] * numOutputs

    ''' Examples list will be a list of tuples, where the first item in the tuple is the dictionary for the example, and the second item in the tuple is the actual value. This code takes that list, and converts the output value into an output vector.'''
    examplesList = []
    for example in examplesList:
        value = example[1]
        example[1] = list(outputVector)
        example[1][value] = 1

    numberOfInputs = len(examplesList)
    inputs = [0] * numberOfInputs
    for i in range(numberOfInputs):
        inputs[i] = inputNode(0)

    hidden = [0] * numberOfInputs + 1
    for node in range(numHiddenNodes):
        hidden[node] = hiddenNode(numberOfInputs)

    outputs = [0] * numHiddenNodes + 1
    for i in range(numOutputs):
        outputs[i] = outputNode(numHiddenNodes)

    network = [inputs, hidden, outputs]

    neuralNetwork = backPropLearning(examplesList,network)
