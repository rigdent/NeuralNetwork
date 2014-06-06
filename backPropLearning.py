'''
Tenzin Rigden and Christopher Winter

Uses the backwards propogation algorithm to build a neural network. This particular code is designed to be used for optical character recognition.
'''


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
        self.weights = [0] * numWeights
        self.inputs = [0] * numWeights
        self.output = 0

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
        return 1

    def calculateOutput(self):
        return 1

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
        self.inputs = [0] * numHiddenNodes
        self.output = 0

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
        return 1

    def calculateOutput(self):
        return 1

    def getWeights(self):
        return self.weights

    def getInputs(self):
        return self.inputs


def getWeights(network):
    

    return None

'''
Main algorithm.
Examples is a list of dictionaries, each dictionary is one example. The key is the pixel, and the value is the value of that pixel.
'''
def backPropLearning(examples,network):
    condition = True
    while condition: # This is the repeat in the pseudocode that I'm not really sure about
        for weight in getWeights(network): # How to get the weights


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

    hidden = [0] * numberOfInputs
    for node in range(numHiddenNodes):
        hidden[node] = hiddenNode(numberOfInputs) # Where this list will become (input,wieght)

    outputs = [0] * numHiddenNodes
    for i in range(numOutputs):
        outputs[i] = outputNode(numHiddenNodes)

    network = [inputs, hidden, outputs]

    neuralNetwork = backPropLearning(examplesList,network)
