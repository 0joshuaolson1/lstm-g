import math
import random
class LSTM_g:
    def logisticFunc(self, value, mode):
        def logValue(value):
            return 1. / (1 + math.exp(-value))
        def logDeriv(value):
            result = logValue(value)
            return result * (1 - result)
        if mode == self.VALUE_MODE:
            return logValue(value)
        return logDeriv(value)
    def getFuncs(self):
        return [self.logisticFunc]
    def getNodes(self):
        return self.net[0]
    def getConnections(self):
        return self.net[1]
    def getEpsilonKs(self):
        return self.net[2]

    def initNet(self):
        self.net = [{}, {}, {}]
    def initNode(self, j):
        self.net[0][j] = [0, 0, 0, [], 0, 0, 0]
    def initConnection(self, j, i):
        self.net[1][j, i] = [0, 0, 0]

    def getState(self, j):
        return self.net[0][j][0]
    def setState(self, j, s):
        self.net[0][j][0] = s
    def getAct(self, j):
        return self.net[0][j][1]
    def setAct(self, j, y):
        self.net[0][j][1] = y
    def getFuncIndex(self, j):
        return self.net[0][j][2]
    def setFuncIndex(self, j, f):
        self.net[0][j][2] = f
    def getGatedArray(self, j):
        return self.net[0][j][3]
    def setGatedArray(self, j, gated):
        self.net[0][j][3] = gated
    def getDelta(self, j):
        return self.net[0][j][4]
    def setDelta(self, j, delta):
        self.net[0][j][4] = delta
    def getDeltaP(self, j):
        return self.net[0][j][5]
    def setDeltaP(self, j, deltaP):
        self.net[0][j][5] = deltaP
    def getDeltaG(self, j):
        return self.net[0][j][6]
    def setDeltaG(self, j, deltaG):
        self.net[0][j][6] = deltaG
    def getWeight(self, j, i):
        return self.net[1][j, i][0]
    def setWeight(self, j, i, w):
        self.net[1][j, i][0] = w
    def getGater(self, j, i):
        return self.net[1][j, i][1]
    def setGater(self, j, i, gater):
        self.net[1][j, i][1] = gater
    def getEpsilon(self, j, i):
        return self.net[1][j, i][2]
    def setEpsilon(self, j, i, epsilon):
        self.net[1][j, i][2] = epsilon
    def getEpsilonK(self, j, i, k):
        return self.net[2][j, i, k]
    def setEpsilonK(self, j, i, k, epsilonK):
        self.net[2][j, i, k] = epsilonK

    def gain(self, j, i):
        if self.getGater(j, i) < 0:
            return 1
        return self.getAct(self.getGater(j, i))
    def calcState(self, j):
        if (j, j) in self.getConnections():
            self.setState(j, self.getState(j) * self.gain(j, j) * self.getWeight(j, j))
        else:
            self.setState(j, 0)
        for i in self.getNodes():
            if j != i and (j, i) in self.getConnections():
                self.setState(j, self.getState(j) + self.gain(j, i) * self.getWeight(j, i) * self.getAct(i))
    def calcEpsilon(self, j, i):
        if (j, j) in self.getConnections():
            self.setEpsilon(j, i, self.getEpsilon(j, i) * self.gain(j, j) * self.getWeight(j, j))
        else:
            self.setEpsilon(j, i, 0)
        self.setEpsilon(j, i, self.getEpsilon(j, i) + self.gain(j, i) * self.getAct(i))
    def calcEpsilonK(self, j, i, k):
        if (k, k) in self.getConnections():
            self.setEpsilonK(j, i, k, self.getEpsilonK(j, i, k) * self.gain(k, k) * self.getWeight(k, k))
            if (k, k) in self.getGatedArray(j):
                self.setEpsilonK(j, i, k, self.getEpsilonK(j, i, k) + self.getFuncs()[self.getFuncIndex(j)](self.getState(j), self.DERIV_MODE) * self.getEpsilon(j, i) * self.getWeight(k, k) * self.getState(k))
        else:
            self.setEpsilonK(j, i, k, 0)
        for p, a in self.getGatedArray(j):
            if p == k and a != k:
                self.setEpsilonK(j, i, k, self.getEpsilonK(j, i, k) + self.getFuncs()[self.getFuncIndex(j)](self.getState(j), self.DERIV_MODE) * self.getEpsilon(j, i) * self.getWeight(k, a) * self.getAct(a))
    def updateTraces(self, j):
        for i in self.getNodes():
            if (j, i) in self.getConnections():
                self.calcEpsilon(j, i)
                m = -1
                for k, l in self.getGatedArray(j):
                    if m != k:
                        m = k
                        self.calcEpsilonK(j, i, k)
    def calcDeltaP(self, j):
        self.setDeltaP(j, 0)
        for k in range(j + 1, len(self.getNodes())):
            if (k, j) in self.getConnections():
                self.setDeltaP(j, self.getDeltaP(j) + self.getDelta(k) * self.gain(k, j) * self.getWeight(k, j))
        self.setDeltaP(j, self.getDeltaP(j) * self.getFuncs()[self.getFuncIndex(j)](self.getState(j), self.DERIV_MODE))
    def calcDeltaG(self, j):
        self.setDeltaG(j, 0)
        m = -1
        for k, l in self.getGatedArray(j):
            if m != k and k > j:
                m = k
                if (k, k) in self.getConnections():
                    self.setDeltaG(j, self.getDeltaG(j) + self.getDelta(k) * self.getWeight(k, k) * self.getState(k))
                for p, a in self.getGatedArray(j):
                    if p == k and a != k:
                        self.setDeltaG(j, self.getDeltaG(j) + self.getDelta(k) * self.getWeight(k, a) * self.getAct(a))
        self.setDeltaG(j, self.getDeltaG(j) * self.getFuncs()[self.getFuncIndex(j)](self.getState(j), self.DERIV_MODE))
    def updateWeights(self, learnRate):
        for j, i in self.getConnections():
            self.setWeight(j, i, self.getWeight(j, i) + learnRate * self.getDeltaP(j) * self.getEpsilon(j, i))
            m = -1
            for k, l in self.getGatedArray(j):
                if m != k and k > j:
                    m = k
                    self.setWeight(j, i, self.getWeight(j, i) + learnRate * self.getDelta(k) * self.getEpsilonK(j, i, k))

    def initialize(self, netSpec):
        self.VALUE_MODE = 0
        self.DERIV_MODE = 1
        self.TRAIN_MODE = 0
        self.TEST_MODE = 1
        self.initNet()
<<<<<<< HEAD
        listType = 0
        for line in netSpec:
            if line == "":
                listType += 1
                continue
            args = line.split(" ")
            if listType < 1:
=======
        for line in netSpec:
            args = line.split(" ")
            if len(args) < 4:
>>>>>>> d3512ae7f96751d9bb4c5bdd46a1c97c22841808
                self.initNode(int(args[0]))
                self.setState(int(args[0]), float(args[1]))
                self.setAct(int(args[0]), self.getFuncs()[int(args[2])](float(args[1]), self.VALUE_MODE))
                self.setFuncIndex(int(args[0]), int(args[2]))
<<<<<<< HEAD
            elif listType < 2:
=======
            elif len(args) > 4:
>>>>>>> d3512ae7f96751d9bb4c5bdd46a1c97c22841808
                self.initConnection(int(args[0]), int(args[1]))
                self.setWeight(int(args[0]), int(args[1]), float(args[2]))
                self.setGater(int(args[0]), int(args[1]), int(args[3]))
                if int(args[3]) > -1:
                    gatedArray = self.getGatedArray(int(args[3]))
                    gatedArray.append(int(args[0]), int(args[1]))
                    self.setGatedArray(int(args[3]), gatedArray)
                self.setEpsilon(int(args[0]), int(args[1]), float(args[4]))
            else:
                self.setEpsilonK(int(args[0]), int(args[1]), int(args[2]), float(args[3]))
<<<<<<< HEAD
    def autoBuild(self, numInputs, numOutputs, inputToOutput, biasToOutput, layerFirsts, layerSizes, inputToBlock, biasToBlock, blockToBlock, blockToOutput):
        def connectionStr(j, i, gater):
            return "\n" + str(j) + " " + str(i) + " " + str(random.random() / 5 - .1) + " " + str(gater) + " 0"
        def getBlockIndices(blockOffset, layerFirsts, layerSizes, blockNum):
            layerIndex = -1
            leftBound = 0
            rightBound = len(layerFirsts)
            while leftBound < rightBound:
                middle = (leftBound + rightBound) / 2
                if layerFirsts[middle] > blockNum:
                    rightBound = middle
                elif layerFirsts[middle] + layerSizes[middle] > blockNum:
                    leftBound = middle
                else:
                    layerIndex = middle
                    break
            blockIndices = []
            for index in range(blockOffset, blockOffset + 4):
                if layerIndex < 0:
                    blockIndices.append(blockNum * 4 + index)
                else:
                    blockIndices.append(layerFirsts[layerIndex] * 3 + blockNum + layerSizes[layerIndex] * index)
            return blockIndices[0], blockIndices[1], blockIndices[2], blockIndices[3]
        def addEpsilonK(epsilonKs, j, i, k):
            if j not in epsilonKs:
                epsilonKs[j] = [set(), set()]
            if i > -1:
                epsilonKs[j][0].add(i)
            if k > -1:
                epsilonKs[j][1].add(k)
        useBias = 0
        if biasToOutput > 1 or len(biasToBlock) > 0:
            useBias = 1
        blockOffset = numInputs + useBias
        numBlocks = max(blockToOutput)
        for connection in blockToBlock:
            numBlocks = max(numBlocks, connection[0])
        outputOffset = blockOffset + numBlocks * 4
=======
    def autoBuild(self, numInputs, numOutputs, inputToOutput, useOutputBias, hiddenLayerSizes, useBiasFlags, peepholeFlags, peepplusFlags, numHiddenLayers):
        def nodeString(j):
            return "\n" + j + " 0 0"
        netSpec = ""
        useBiases = useOutputBias
        for biasFlag in useBiasFlags:
            if biasFlag > 0:
                useBiases = 1
        layerOffset = numInputs + useBiases
        for j in range(layerOffset):
            netSpec += nodeString(j)
        for hiddenLayer in range(numHiddenLayers):
            for memNode in range(memCellCounts[hiddenLayer]):
                for nodeOffset in range(4):
                    netSpec += nodeString(layerOffset + memNode * 4 + nodeOffset)
            layerOffset += memCellCounts[hiddenLayer] * 4
            for normNode in range(normNodeCounts[hiddenLayer]):
                netSpec += nodeString(layerOffset + normNode)
            layerOffset += normNodeCounts[hiddenLayer]
        for j in range(numOutputs):
            netSpec += nodeString(layerOffset + j)
        if inputToOutput > 0:
            for i in range(numInputs + useBiases):
                for j in range(numOutputs):
                    netSpec += nodeString(layerOffset + j)
        if useBiases > 0:
            pass
        return netSpec[1:]
#peepplus[0,1,2=ungated]
#j s fnx_index
#j i w gater epsilon
#j i k epsilon_k
    def __init__(self, netSpec):
        lines = netSpec.split("\n")
        if len(lines) > 1:
            self.initialize(lines)
            return
        archSpec = lines[0].split(" ")
        hiddenLayerSizes = []
        useBiasFlags = []
        peepholeFlags = []
        peepplusFlags = []
        for index in range(4, len(archSpec), 4):
            hiddenLayerSizes.append(int(archSpec[index]))
            useBiasFlags.append(int(archSpec[index + 1]))
            peepholeFlags.append(int(archSpec[index + 2]))
            peepplusFlags.append(int(archSpec[index + 3]))
        self.initialize(self.autoBuild(int(archSpec[0]), int(archSpec[1]), int(archSpec[2]), int(archSpec[3]), hiddenLayerSizes, useBiasFlags, peepholeFlags, peepplusFlags, len(hiddenLayerSizes)).split("\n"))
    def toString(self):
>>>>>>> d3512ae7f96751d9bb4c5bdd46a1c97c22841808
        netSpec = ""
        for nodeNum in range(outputOffset + numOutputs):
            netSpec += "\n" + str(nodeNum) + " 0 0"
        netSpec += "\n"
        if inputToOutput > 0:
            for inputNum in range(numInputs):
                for outputNum in range(outputOffset, outputOffset + numOutputs):
                    netSpec += connectionStr(outputNum, inputNum, -1)
        if biasToOutput > 0:
            for outputNum in range(outputOffset, outputOffset + numOutputs):
                netSpec += connectionStr(outputNum, numInputs, -1)
        epsilonKs = {}
        for blockNum in range(numBlocks):
            inputGateIndex, forgetGateIndex, memoryCellIndex, outputGateIndex = getBlockIndices(blockNum)
            netSpec += "\n" + str(memoryCellIndex) + " " + str(memoryCellIndex) + " 1 " + str(forgetGateIndex) + " 0"
            addEpsilonK(epsilonKs, forgetGateIndex, -1, memoryCellIndex)
        for blockNum in inputToBlock:
            inputGateIndex, forgetGateIndex, memoryCellIndex, outputGateIndex = getBlockIndices(blockNum)
            for inputNum in range(numInputs):
                netSpec += connectionStr(inputGateIndex, inputNum, -1)
                netSpec += connectionStr(forgetGateIndex, inputNum, -1)
                netSpec += connectionStr(memoryCellIndex, inputNum, inputGateIndex)
                netSpec += connectionStr(outputGateIndex, inputNum, -1)
                addEpsilonK(epsilonKs, inputGateIndex, inputNum, memoryCellIndex)
                addEpsilonK(epsilonKs, forgetGateIndex, inputNum, -1)
                addEpsilonK(epsilonKs, outputGateIndex, inputNum, -1)
        for blockNum in biasToBlock:
            inputGateIndex, forgetGateIndex, memoryCellIndex, outputGateIndex = getBlockIndices(blockNum)
            netSpec += connectionStr(inputGateIndex, numInputs, -1)
            netSpec += connectionStr(forgetGateIndex, numInputs, -1)
            netSpec += connectionStr(memoryCellIndex, numInputs, inputGateIndex)
            netSpec += connectionStr(outputGateIndex, numInputs, -1)
            addEpsilonK(epsilonKs, inputGateIndex, numInputs, -1)
            addEpsilonK(epsilonKs, forgetGateIndex, numInputs, -1)
            addEpsilonK(epsilonKs, outputGateIndex, numInputs, -1)
        for connection in blockToBlock:
            inputGateIndexTo, forgetGateIndexTo, memoryCellIndexTo, outputGateIndexTo = getBlockIndices(connection[0])
            inputGateIndexFrom, forgetGateIndexFrom, memoryCellIndexFrom, outputGateIndexFrom = getBlockIndices(connection[1])
            if connection[2] < 1:
                netSpec += connectionStr(inputGateIndexTo, memoryCellIndexFrom, -1)
                netSpec += connectionStr(forgetGateIndexTo, memoryCellIndexFrom, -1)
                netSpec += connectionStr(memoryCellIndexTo, memoryCellIndexFrom, inputGateIndexTo)
                netSpec += connectionStr(outputGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, inputGateIndexTo, memoryCellIndexFrom, memoryCellIndexTo)
                addEpsilonK(epsilonKs, forgetGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, outputGateIndexTo, memoryCellIndexFrom, -1)
            elif connection[2] < 2:
                netSpec += connectionStr(inputGateIndexTo, memoryCellIndexFrom, -1)
                netSpec += connectionStr(forgetGateIndexTo, memoryCellIndexFrom, -1)
                netSpec += connectionStr(outputGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, inputGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, forgetGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, outputGateIndexTo, memoryCellIndexFrom, -1)
            else:
                netSpec += connectionStr(inputGateIndexTo, memoryCellIndexFrom, outputGateIndexFrom)
                netSpec += connectionStr(forgetGateIndexTo, memoryCellIndexFrom, outputGateIndexFrom)
                netSpec += connectionStr(outputGateIndexTo, memoryCellIndexFrom, outputGateIndexFrom)
                addEpsilonK(epsilonKs, inputGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, forgetGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, outputGateIndexTo, memoryCellIndexFrom, -1)
                addEpsilonK(epsilonKs, outputGateIndexFrom, -1, inputGateIndexTo)
                addEpsilonK(epsilonKs, outputGateIndexFrom, -1, forgetGateIndexTo)
                addEpsilonK(epsilonKs, outputGateIndexFrom, -1, outputGateIndexTo)
        for blockNum in blockToOutput:
            inputGateIndex, forgetGateIndex, memoryCellIndex, outputGateIndex = getBlockIndices(blockNum)
            for outputNum in range(outputOffset, outputOffset + numOutputs):
                netSpec += connectionStr(outputNum, memoryCellIndex, outputGateIndex)
                addEpsilonK(epsilonKs, outputGateIndex, -1, outputNum)
        netSpec += "\n"
        for j in epsilonKs:
            for i in epsilonKs[j][0]:
                for k in epsilonKs[j][1]:
                    netSpec += "\n" + str(j) + " " + str(i) + " " + str(k) + " 0"
        return netSpec[1:]
#0
#j s fnx_index
#j i w gater epsilon
#j i k epsilon_k
# OR
#1
#numInputs[1+] numOutputs[1+] i-o[0/1] Bias-o[0/1](not list)
#i-c[0+]
#b-c[0+]
#firstInLayer[0+] numInLayer[2+]
#c-c{down,peephole,gated}[0+,0+,0-2]
#c-o[0+]

#see 3.3.(2,5), 4.2.1, 5.5.1, 6.6.2, 7.2.1 of thesis for bias ideas
#see papers, tanh, magic number for activation function ideas
#to do: list random weight ideas
    def __init__(self, netSpec):
        lines = netSpec.split("\n")
        if lines[0] == "0":
            self.initialize(lines[2:])
            return
        settings = lines[2].split(" ")
        layerFirsts = layerSizes = inputToBlock = biasToBlock = blockToBlock = blockToOutput = []
        listType = 0
        for line in lines[4:]:
            if line == "":
                listType += 1
                continue
            if listType < 1:
                inputToBlock.append(int(line))
            elif listType < 2:
                biasToBlock.append(int(line))
            elif listType < 3:
                args = line.split(" ")
                layerFirsts.append(int(args[0]))
                layerSizes.append(int(args[1]))
            elif listType < 4:
                args = line.split(" ")
                blockToBlock.append([int(args[0]), int(args[1]), int(args[2])])
            else:
                blockToOutput.append(int(line))
        self.initialize(self.autoBuild(int(settings[0]), int(settings[1]), int(settings[2]), int(settings[3]), layerFirsts, layerSizes, inputToBlock, biasToBlock, blockToBlock, blockToOutput).split("\n"))
    def toString(self):
        netSpec = "0\n"
        for j in self.getNodes():
            netSpec += "\n" + str(j) + " " + str(self.getState(j)) + " " + str(self.getFuncIndex(j))
        netSpec += "\n"
        for i in self.getNodes():
            for j in self.getNodes():
                if (j, i) in self.getConnections():
                    netSpec += "\n" + str(j) + " " + str(i) + " " + str(self.getWeight(j, i)) + " " + str(self.getGater(j, i)) + " " + str(self.getEpsilon(j, i))
<<<<<<< HEAD
        netSpec += "\n"
        for j, i in self.getConnections():
            m = -1
=======
        m = -1
        for j, i in self.getConnections():
>>>>>>> d3512ae7f96751d9bb4c5bdd46a1c97c22841808
            for k, l in self.getGatedArray(j):
                if m != k:
                    m = k
                    netSpec += "\n" + str(j) + " " + str(i) + " " + str(k) + " " + str(self.getEpsilonK(j, i, k))
<<<<<<< HEAD
        return netSpec
=======
        return netSpec[1:]
>>>>>>> d3512ae7f96751d9bb4c5bdd46a1c97c22841808
    def step(self, inputs, mode):
        for j in range(len(inputs)):
            self.setAct(j, inputs[j])
        for j in range(len(inputs), len(self.getNodes()):
            self.calcState(j)
            self.setAct(j, self.getFuncs()[self.getFuncIndex(j)](self.getState(j), self.VALUE_MODE))
            if mode == self.TRAIN_MODE:
                self.updateTraces(j)
    def getOutput(self, length):
        result = []
        for j in range(len(self.getNodes()) - length, len(self.getNodes())):
            result.append(self.getAct(j))
        return result
    def adjust(self, target, learnRate):
        for j in range(len(self.getNodes()) - len(target), len(self.getNodes())):
            self.setDelta(j, target[j] - self.getAct(j))
        for j in reversed(range(len(self.getNodes()) - len(target))):
            self.calcDeltaP(j)
            self.calcDeltaG(j)
            self.setDelta(j, self.getDeltaP(j) + self.getDeltaG(j))
        self.updateWeights(learnRate)
