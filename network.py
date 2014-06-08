class inputNode:
    def __init__(self,index,inputValue):
        self.value = inputValue
        self.index = index

    def setValue(self,newValue):
        self.value = newValue

    def getValue(self):
        return self.value

    def getIndex(self):
        return self.index

class hiddenNode:
    def __init__(self,index):
        self.index = index
        self.edges = []

    def setEdge(self,weight,node):
        self.edges.append([weight,node])

    def updateEdgeWeight(self,index,weight):
        self.edges[index][0] = weight

    def getEdge(self,index):
        return self.edges[index]

    def getEdges(self):
        return self.edges

class outputNode:
    def __init__(self,index):
        self.index = index
        self.edges = []

    def setEdge(self,weight,node):
        self.edges.append([weight,node])

    def updateEdgeWeight(self,index,weight):
        self.edges[index][0] = weight

def main():

    node = hiddenNode(0)

    numInputs = 2
    numHidden = 2
    numOutputs = 2

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

    for node in network[1]:
        print node.getEdges()

main()
