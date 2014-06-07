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
    def __init__(self,numInputsAndDummy):
        self.weights = [0] * (numInputsAndDummy)
        self.weights[0] = 1
        self.inputs = [0] * (numInputsAndDummy)
        self.inputs[0] = 1
        self.output = 0
        self.delta = 0
        self.weightedInput = 0

    def setDelta(self, newDelta):
        self.delta = newDelta

    def getDelta(self):
        return self.delta

    '''calculates delta I. It takes the following parameters: the current example, the index of the
    hidden node we are calculating the delta for (we use it to find the appropriate weight from the
    hidden node to the output node which is stored as weights in output node), deltaJList is used to
    get delta J of the outputNodes.
    '''
    def calculateDeltaI(self, example, index, deltaJList):
        sumI = 0
        sumI+=outputs[index].getWeights()[0]*1
        for i in range(0,len(outputs[index].getWeights())):
            sumI+= outputs[index].getWeights()[i+1]*deltaJList[i]
        self.delta = sigmoidDerivativeFunction(self.weightedInput)*sumI

    def setWeight(self,weight,wieghtIndex):
        self.weights[weightIndex] = weight

    def setInput(self,input,inputIndex):
        self.inputs[inputIndex] = input

    def batchSetWeight(self,weights):
        for i in range(len(self.weights)):
            self.setWeight(weights[i],i)
        return self.weights

    def batchSetInputs(self,inputs):
        for i in range(len(self.weights)):
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
    def __init__(self,numHiddenNodesAndDummy):
        self.weights = [0] * (numHiddenNodesAndDummy)
        self.weights[0] = 1
        self.inputs = [0] * (numHiddenNodesAndDummy)
        self.inputs[0] = 1
        self.output = 0
        self.delta = 0
        self.weightedInput = 0

    def setDelta(self, newDelta):
        self.delta = newDelta

    def getDelta(self):
        return self.delta

    def calculateDeltaJ(self, example):
        self.delta = sigmoidDerivativeFunction(self.getWeightedInputs)*(example.index(1)-self.output)

    def setWeight(self,weight,weightIndex):
        self.weights[weightIndex] = weight

    def batchSetWeight(self,weights):
        for i in range(len(self.weights)):
            self.setWeight(weights[i],i)
        return self.weights

    def batchSetInputs(self,inputs):
        for i in range(len(self.weights)):
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
    initialWeight = 0.5
    initialWeightsHidden = [initialWeight] * len(network[1])+1
    initialWeightsOutputs = [initialWeight] * len(network[-1])+1
    for node in network[1]:
        node.batchSetWeights(initialWeightsHidden)
    for node in network[2]:
        node.batchSetWeights(initialWeightsOutputs)

    return None

def setCondition(numIterations):
    epoch = 50000 # Maximum number of iterations
    if numIterations >= epoc:
        return False
    else:
        return True


'''
Main algorithm.
Examples is a list of dictionaries, each dictionary is one example. The key is the pixel, and the value is the value of that pixel.
'''
def backPropLearning(examples,network):

    #changed examples list to network list
    inputs = network[0]
    hidden = network[1]
    outputs = network[2]

    # Sets the weights to all be some small initial value
    setInitialWeights(network)

    alpha = 1

    iteration = 0
    condition = setCondition(iteration)
    while condition: # This is the repeat in the pseudocode that I'm not really sure about

        for example in examples:
            # set the inputs
            for i in range(len(example[0].values())):
                inputs[i].setValue(example[0][i])

            # Set up rest of the layers

            for layer in network[1:]:

                for node in layer:
                    for inputNumber in range(len(network.index(layer)-1)):
                        node.setInput(inputs[inputNumber],inputNumber+1) # The "+1" is because of the dummy input
                    weightedInputs = node.calculateWeightedInputs()
                    output = sigmoidFunction(weightedInputs)
                    node.setOutput(output)

            for node in outputs:
                node.setDelta(node.calculateDeltaJ(example))
            for hiddenNode in hidden:
                deltaJList = []
                for outputNode in outputs:
                    deltaJList.append(outputNode.getDelta)
                hiddenNode.calculateDeltaI(example,hidden.index(hiddenNode), deltaJList)

            #this is where we update weights
            for layerIndex in range(1, len(network)+1):
                for node in network[layerIndex]:
                    weights = node.getWeights()
                    inputs = node.getInputs()
                    for i in range(len(weights)):
                        newWeight = weights[i] + decreaseAlpha(alpha)*inputs[i]*node.getDelta()
                        node.setWeight(newWeight,i)



        iteration += 1
        condition = setCondition(iteration)

    return True

def decreaseAlpha(alpha):
    return float(alpha)

def main():
    data = open('smallData.txt', 'r')
    data = data.readlines()
    length = len(data)
    counter = 1
    examplesList = []
    dict = {}
    yVal = 0
    for i in range(0,length):
        if counter <17:
            for j in range(0,16):
                dict[yVal*16+j] = data[i][j]
            yVal+=1
        elif counter == 17:
            examplesList.append([dict, int(data[i][0])])
            counter = 0
            yVal = 0
            dict = {}
        counter+= 1





    numHiddenNodes = 96
    numOutputs = 10

    outputVector = [0] * numOutputs

    ''' Examples list will be a list of tuples, where the first item in the tuple is the dictionary for the example, and the second item in the tuple is the actual value. This code takes that list, and converts the output value into an output vector.'''
    for example in examplesList:
        value = example[1]
        example[1] = list(outputVector)
        example[1][value] = 1

    numberOfInputs = len(examplesList[0][0]) # The length of one of the dictionaries of inputs
    inputs = [0] * numberOfInputs
    for i in range(numberOfInputs):
        inputs[i] = inputNode(0)



    hidden = [0] * (numHiddenNodes)
    for node in range(numHiddenNodes):
        hidden[node] = hiddenNode(numberOfInputs+1)

    outputs = [0] * (numOutputs)
    for i in range(numOutputs):
        outputs[i] = outputNode(numHiddenNodes+1)

    print len(hidden[0].getInputs())

    network = [inputs, hidden, outputs]

    print len(network[1])

    neuralNetwork = backPropLearning(examplesList,network)

main()
