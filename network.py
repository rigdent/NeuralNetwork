'''By Chris and Tenzin
'''
import random
import math

class inputNode:
    def __init__(self,index,inputValue,dummy = False):
        self.value = inputValue
        self.index = index
        self.dummy = True

    def setValue(self,newValue):
        self.value = newValue

    def getOutput(self):
        return self.value

    def getIndex(self):
        return self.index

    def getDummy(self):
        return self.dummy

class hiddenNode:
    def __init__(self,index):
        self.index = index # Index will start at 1 because of dummy input? Nevermind
        self.edges = [[0,inputNode(0,1)]] # Start with dummy input
        self.input = 0
        self.output = 0
        self.deltaI = 0
        self.previousWeightUpdates = [0]

    def setEdge(self,weight,node):
        self.edges.append([weight,node])
        self.previousWeightUpdates.append(0)

    def updateEdgeWeight(self,index,weight):
        self.edges[index][0] = weight

    def updateEdgeWeightMomentum(self,index,weight):
        #print self.previousWeightUpdates[index]
        self.edges[index][0] = weight + (self.previousWeightUpdates[index]) * 0.5

    def getEdge(self,index):
        return self.edges[index]

    def getEdges(self):
        return self.edges

    def setInput(self,input):
        self.input = input

    def getInput(self):
        return self.input

    def setWeightedInput(self):
        weightedInput = 0
        for edge in self.edges:
            weightedInput += edge[0] * edge[1].getOutput()
        self.input = weightedInput
        return self.input

    def setOutput(self):
        self.output = sigmoidFunction(self.getInput())
        return self.output

    def getOutput(self):
        return self.output

    def calculateDeltaI(self,example,previousLayer):
        sumI = 0
        for node in previousLayer:
            sumI += node.getEdges()[self.index+1][0] * node.getDelta() # The plus 1 is to account for dummy variable
        self.deltaI = sumI * sigmoidDerivativeFunction(self.input)
        return self.deltaI

    def getDelta(self):
        return self.deltaI

    def setPreviousWeightUpdate(self,index,previousWeight):
        self.previousWeightUpdates[index] = previousWeight

class outputNode:
    def __init__(self,index):
        self.index = index
        self.edges = [[0,inputNode(0,1)]] # Start with dummy input
        self.input = 0
        self.output = 0
        self.deltaJ = 0
        self.previousWeightUpdates = [0]
        self.error = 0


    def setEdge(self,weight,node):
        self.previousWeightUpdates.append(0)
        self.edges.append([weight,node])

    def updateEdgeWeight(self,index,weight):
        self.edges[index][0] = weight

    def updateEdgeWeightMomentum(self,index,weight):
        #print self.previousWeightUpdates[index]
        self.edges[index][0] = (self.edges[index][0] * 0.9) + ((self.previousWeightUpdates[index]) * 0.5)

    def getEdge(self,index):
        return self.edges[index]

    def getEdges(self):
        return self.edges

    def setInput(self,input):
        self.input = input

    def getInput(self):
        return self.input

    def setWeightedInput(self):
        weightedInput = 0
        for edge in self.edges:
            weightedInput += edge[0] * edge[1].getOutput()
        self.input = weightedInput
        return self.input

    def setOutput(self):
        self.output = sigmoidFunction(self.getInput())
        return self.output

    def getOutput(self):
        if self.output >= 0.9:
            return 1
        elif self.output <= 0.1:
            return 0
        else:
            return self.output

    def setDeltaJ(self,deltaJ):
        self.deltaJ = deltaJ

    def calculateDeltaJ(self,example):
        self.deltaJ = sigmoidDerivativeFunction(self.input) * (example[1][self.index] - self.output)
        self.error = example[1][self.index] - self.output
        return self.deltaJ

    def getDelta(self):
        return self.deltaJ

    def setPreviousWeightUpdate(self,index,previousWeight):
        self.previousWeightUpdates[index] = previousWeight
        return None

    def getError(self):
        return self.error

'''
This method returns the result of the sigmoid function.
'''
def sigmoidFunction(x):
    return 1 / (1 + math.exp(-x))

'''
This method returns the result of the derivative of the sigmoid function.
'''
def sigmoidDerivativeFunction(x):
    return sigmoidFunction(x) * (1 - sigmoidFunction(x))

'''
This was a method used in testing to print the edges of the network.
'''
def printEdges(network):
    for layer in network[1:]:
        for node in layer:
            print node.getEdges()

    return None

'''
This was a method used in testing to print the inputs of the network.
'''
def printInputs(network):
    for node in network[0]:
        print node.getIndex(),
        print ", ",
        print node.getValue()

    return None

'''
This function decreases alpha by the appropriate amount each time.
'''
def decreaseAlpha(alpha):
    return float(alpha)*0.999

'''
This function sets all of the weights to be a random intial value between 0 and 0.1.
'''
def setInitialWeights(network):
    random.seed(10)

    for layer in network[1:]:
        for node in layer:
            for edge in range(len(node.getEdges())):
                node.updateEdgeWeight(edge,random.random()/10)

    return None

'''
This is the function used to actually test a number.
'''
def testNumber(network,example):

    # Set the inputs
    for input in network[0]:
        input.setValue(example[0][input.getIndex()])

    # propogate forward

    for layer in network[1:]:
        for node in layer:
            node.setWeightedInput()
            node.setOutput()


    vector = []

    for node in network[-1]:
        vector.append(node.getOutput())

    return vector



'''
This function performs k-fold cross validation on the network.
'''
def kFoldCrossV(network,examplesList,numToCheck):
    #pick from list is used to select from examples randomly and keep track of what's been tested
    pickFromList = range(len(examplesList))
    sucesses = 0
    #loops one less times than it needs to. i.e. if examples is 100 long and numToCheck is 30, it loops 3 times
    #this is because the last test case only checks 10 examples instead of the 30 the other test cases check.
    for i in range(len(examplesList)/numToCheck):
        testData = []
        trainingData = list(examplesList)
        #add all the ones you want to check into a test list and pop them from the examples list
        for j in range(numToCheck):
            numberToTest = random.choice(pickFromList)
            testIndex = pickFromList.index(numberToTest)
            pickFromList.pop(testIndex)
            testData.append(trainingData.pop(testIndex))
        #make network using this shortened examples list
        testNetwork = backPropLearning(trainingData,network)
        #then test the data you kept out with it.
        for testExample in testData:
            networkOutputVector = testNumber(testNetwork, testExample)
            if 1 in networkOutputVector:
                vectorIndex = networkOutputVector.index(1)
                if vectorIndex == testExample[1].index(1):
                    sucesses += 1
            print "sucesses: ",sucesses

    #this is the last loop that checks the leftover examples in one last case.
    testData = []
    trainingData = list(examplesList)
    for j in range((len(examplesList)/numToCheck)*numToCheck, len(examplesList)):
        numberToTest = random.choice(pickFromList)
        testIndex = pickFromList.index(numberToTest)
        pickFromList.pop(testIndex)
        testData.append(trainingData.pop(testIndex))
    testNetwork = backPropLearning(trainingData,network)
    for testExample in testData:
        networkOutputVector = testNumber(testNetwork, testExample)
        if 1 in networkOutputVector:
            vectorIndex = networkOutputVector.index(1)
            if vectorIndex == testExample[1].index(1):
                sucesses += 1
        print "sucesses: ",sucesses
    #return the percentage of sucesses.
    return float(sucesses)/len(examplesList)



'''
This function does leave-one-out cross validation on the network. Takes a very long time!
'''
def leaveOneOut(network,examplesList):
    sucesses = 0
    attempts = 0
    for example in examplesList:
        print attempts
        trainingData = list(examplesList)
        testData = trainingData.pop(examplesList.index(example))
        testNetwork = backPropLearning(trainingData,network)
        networkOutputVector = testNumber(testNetwork,testData)
        if 1 in networkOutputVector: # If the networkOutputVector does not contain a 1, assume failure
            vectorIndex = networkOutputVector.index(1)
            if vectorIndex == testData[1].index(1):
                sucesses += 1
        attempts += 1
        print sucesses
    print float(sucesses) / len(examplesList)

'''
The main function that trains the network.
'''
def backPropLearning(examples,network):

    setInitialWeights(network)

    #printEdges(network[1:])


    alpha = 1 # Start learning rate at 1

    iteration = 0
    epoch = 100
    condition = True
    meanSquaredError = 0
    while condition: # For now only iterate 5 times

        #print iteration

        for example in examples:
            # set the inputs
            for input in network[0]:
                input.setValue(example[0][input.getIndex()])

            # Propogate forward

            for layer in network[1:]:
                for node in layer:
                    node.setWeightedInput()
                    node.getInput()
                    node.setOutput()

            # calculate delta J for output layer
            error = 0
            for node in network[-1]:
                node.calculateDeltaJ(example)
                error += node.getError()
            # calculate delta I for all hidden layers
            previousMeanSquaredError = meanSquaredError
            meanSquaredError = error**2 / len(network[-1])
            #print meanSquaredError

            for layer in reversed(network[1:-1]):
                previousLayer = network.index(layer) + 1 # This is the index of the previous layer
                for node in layer:
                    # Calculate the delta I here
                    node.calculateDeltaI(example,network[previousLayer])

            #printEdges(network[1:])

            # update all edges in network

            #printEdges(network[1:])
            #print '\n'

            for layer in network[1:]:
                for node in layer:
                    for edgeIndex in range(len(node.getEdges())):
                        weightUpdate = node.getEdge(edgeIndex)[0] + ( alpha * node.getEdge(edgeIndex)[1].getOutput() * node.getDelta() )

                        #node.setPreviousWeightUpdate(edgeIndex,weightUpdate)
                        node.updateEdgeWeight(edgeIndex,weightUpdate)
            #printEdges(network[1:])
            #print '\n'
        alpha = decreaseAlpha(alpha)
        iteration += 1

        if iteration >= epoch:
            condition = False
        elif meanSquaredError < 0.000001:
            condition = False
    print "finished"
    #print meanSquaredError
    print "Iterations: ",iteration
    #for node in network[-1]:
    # print node.getOutput()
    #printEdges(network[1:])

    return network

def main():
    # reads in the examples
    data = open('smallData.txt', 'r')
    data = data.readlines()
    length = len(data) #/9 + 17*6
    counter = 1
    examplesList = []
    dict = {}
    yVal = 0
    for i in range(0,length):
        if counter <17:
            for j in range(0,16):
                dict[yVal*16+j] = int(data[i][j])
            yVal+=1
        elif counter == 17:
            examplesList.append([dict, int(data[i][0])])
            counter = 0
            yVal = 0
            dict = {}
        counter+= 1


    # our output is in the form of a vector. The number zero would be [1,0,0,0,0,0,0,0,0,0], etc

    outputVector = [0] * 10

    ''' Examples list will be a list of tuples, where the first item in the tuple is the dictionary for the example, and the second item in the tuple is the actual value. This code takes that list, and converts the output value into an output vector.'''
    for example in examplesList:
        value = example[1]
        example[1] = list(outputVector)
        example[1][value] = 1

    '''
This section constructs the network. It consists of n input nodes, where n is the number of attributes for each example. There are then an arbitrary number of hidden nodes and output nodes.

The input nodes are stored in a list, and are initialized to have a value of 0. They also have an index attribute.

The hidden and output nodes are initialized just with an index. They are also set to have the connections from the layer that they receive inputs from as the edges.
'''

    numInputs = len(examplesList[0][0])
    numHidden = 50
    numOutputs = 10

    inputs = []
    for i in range(numInputs):
        node = inputNode(i,0)
        inputs.append(node)

    hidden = []
    for i in range(numHidden):
        hidden.append(hiddenNode(i))
        for input in inputs:
            hidden[i].setEdge(0,input)

    outputs = []
    for i in range(numOutputs):
        outputs.append(outputNode(i))
        for nodeInHiddenLayer in hidden:
            outputs[i].setEdge(0,nodeInHiddenLayer)

    network = [inputs,hidden,outputs]


    #leaveOneOut(network,examplesList)

    print kFoldCrossV(network,examplesList,30)
    
main()
