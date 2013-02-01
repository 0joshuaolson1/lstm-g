#175 lines not including comments, which are coming soon
#no line is longer than 99 characters - sorry if you wanted 80
import math, random, os
class LSTM_g:
    def actFunc(self, s, derivative, bias=0):
        value = 1 / (1 + math.exp(bias - s))
        if derivative:
            value *= 1 - value
        return value
    def gain(self, j, i):
        if (j, i) in self.gater:
            return self.activation[self.gater[j, i]]
        if (j, i) in self.weight:
            return 1
        return 0
    def theTerm(self, j, k, state, activation):
            term = 0
            if (k, k) in self.gater and j == self.gater[k, k]:
                term = state[k]
            for l, a in self.gater:
                if l == k and a != k and j == self.gater[k, a] and activation is self.activation:
                    term += self.weight[k, a] * activation[a]
                elif l == k and a != k and j == self.gater[k, a]:
                    term += self.weight[k, a] * activation[k, a]
            return term
    def build(self, specData):
        self.state, self.activation, self.weight, self.gater = {}, {}, {}, {}
        self.trace, self.extendedTrace = {}, {}
        self.numInputs, self.numOutputs = int(specData[0][0]), int(specData[0][1])
        newNetwork = True
        for args in specData[1:]:
            if len(args) < 3:
                newNetwork = False
                self.state[int(args[0])] = float(args[1])
                self.activation[int(args[0])] = self.actFunc(self.state[int(args[0])], False)
            if newNetwork:
                self.weight[int(args[0]), int(args[1])] = float(args[2])
                if args[3] != "-1":
                    self.gater[int(args[0]), int(args[1])] = int(args[3])
                if args[0] != args[1]:
                    self.trace[int(args[0]), int(args[1])] = 0
            elif len(args) > 3:
                self.extendedTrace[int(args[0]), int(args[1]), int(args[2])] = float(args[3])
            elif len(args) > 2:
                self.trace[int(args[0]), int(args[1])] = float(args[2])
        if newNetwork:
            for j, i in self.trace:
                self.state[j] = self.activation[j] = 0
                for k, a in self.gater:
                    if j < k and j == self.gater[k, a]:
                        self.extendedTrace[j, i, k] = 0
        self.numUnits = max(self.state) + 1
    def toLowLevel(self, specData):
        def addConnection(j, i, g=-1):
            specData.append([str(j), str(i), repr(random.uniform(-.1, .1)), str(g)])
        def unitsInBlock(blockNum):
            for firstBlockInLayer, layerSize in layerData:
                if 0 <= blockNum - firstBlockInLayer < layerSize:
                    offset = numInputs + 3 * firstBlockInLayer + blockNum
                    return range(offset, offset + 4 * layerSize, layerSize)
            return range(numInputs + 4 * blockNum, numInputs + 4 * blockNum + 4, 4)
        numInputs, numOutputs = int(specData[0][0]), int(specData[0][1])
        inputToOutput, biasOutput = int(specData[0][2]), int(specData[0][3])
        lastInput = numInputs - int(biasOutput)
        blockData, connections, layerData = [], [], []
        for args in specData[1:]:
            if len(args) > 3:
                blockData.append([int(args[0]), args[1], args[2], args[3]])
                if args[3] == "1":
                    lastInput = numInputs - 1
            elif len(args) > 2:
                connections.append([int(args[0]), int(args[1]), args[2]])
            else:
                layerData.append([int(args[0]), int(args[1])])
        firstOutput = numInputs + 4 * max(blockData)[0] + 4
        specData[:] = [specData[0][:2]]
        for outputUnit in range(firstOutput, firstOutput + numOutputs):
            if inputToOutput == "1":
                for inputUnit in range(lastInput):
                    addConnection(outputUnit, inputUnit)
            if biasOutput == "1":
                addConnection(outputUnit, lastInput)
        for memoryBlock, receiveInput, sendToOutput, biased in blockData:
            inputGate, forgetGate, memoryCell, outputGate = unitsInBlock(memoryBlock)
            if receiveInput == "1":
                for inputUnit in range(lastInput):
                    for unit in [inputGate, forgetGate, outputGate]:
                        addConnection(unit, inputUnit)
                    addConnection(memoryCell, inputUnit, inputGate)
            if sendToOutput == "1":
                for outputUnit in range(firstOutput, firstOutput + numOutputs):
                    addConnection(outputUnit, memoryCell, outputGate)
            if biased == "1":
                for unit in [inputGate, memoryCell, forgetGate, outputGate]:
                    addConnection(unit, lastInput)
            specData.append([str(memoryCell), str(memoryCell), "1", str(forgetGate)])
        for toBlock, fromBlock, connectionType in connections:
            toIGate, toFGate, toMCell, toOGate = unitsInBlock(toBlock)
            fromIGate, fromFGate, fromMCell, fromOGate = unitsInBlock(fromBlock)
            for toUnit in [toIGate, toFGate, toOGate]:
                if connectionType == "0":
                    fromOGate = -1
                addConnection(toUnit, fromMCell, fromOGate)
            if connectionType == "2":
                addConnection(toMCell, fromMCell, toIGate)
    def __init__(self, specString):
        specData = []
        for line in specString.splitlines():
            if line.strip() != "":
                specData.append([arg.strip() for arg in line.split(",")])
        if len(specData[0]) > 2:
            self.toLowLevel(specData)
        self.build(specData)
    def toString(self, newNetwork, newline=os.linesep):
        specString = str(self.numInputs) + ", " + str(self.numOutputs)
        for (j, i), w in sorted(self.weight.items()):
            specString += newline + str(j) + ", " + str(i) + ", " + repr(w) + ", -1"
            if (j, i) in self.gater:
                specString = specString[:-2] + str(self.gater[j, i])
        if not newNetwork:
            for j, s in sorted(self.state.items()):
                specString += newline + str(j) + ", " + repr(s)
            for (j, i), t in sorted(self.trace.items()):
                specString += newline + str(j) + ", " + str(i) + ", " + repr(t)
            for (j, i, k), e in sorted(self.extendedTrace.items()):
                specString += newline + str(j) + ", " + str(i) + ", " + str(k) + ", " + repr(e)
        return specString
    def step(self, inputs):
        for j in range(self.numInputs):
            self.activation[j] = inputs[j]
        oldState, oldActivation, oldGain = self.state.copy(), {}, {}
        for j in sorted(self.state):
            oldGain[j] = self.gain(j, j)
            self.state[j] *= oldGain[j]
            bias = 0
            for (l, i), t in self.trace.items():
                if l == j:
                    oldActivation[j, i], oldGain[j, i] = self.activation[i], self.gain(j, i)
                    if i >= self.numInputs or (j, i) in self.gater:
                        self.state[j] += oldGain[j, i] * self.weight[j, i] * self.activation[i]
                        self.trace[j, i] = oldGain[j] * t + oldGain[j, i] * self.activation[i]
                    else:
                        bias = self.trace[j, i] = self.activation[i]
            self.activation[j] = self.actFunc(self.state[j], False, bias)
        for (j, i, k), e in self.extendedTrace.items():
            terms = self.trace[j, i] * self.theTerm(j, k, oldState, oldActivation)
            self.extendedTrace[j, i, k] = oldGain[k] * e + self.actFunc(oldState[j], True) * terms
        return [self.activation[j] for j in range(self.numUnits - self.numOutputs, self.numUnits)]
    def getError(self, targets):
        error = 0
        for j in range(self.numUnits - self.numOutputs, self.numUnits):
            t, y = targets[j + self.numOutputs - self.numUnits], self.activation[j]
            error += t * math.log(y, 2) + (1 - t) * math.log(1 - y, 2)
        return error
    def learn(self, targets, learningRate=.1):
        errorProj, errorResp = {}, {}
        for j in range(self.numUnits - self.numOutputs, self.numUnits):
            errorResp[j] = targets[j + self.numOutputs - self.numUnits] - self.activation[j]
        for j in reversed(range(self.numInputs, self.numUnits - self.numOutputs)):
            errorProj[j] = errorResp[j] = 0
            for k, l in self.trace:
                if l == j and j < k:
                    errorProj[j] += errorResp[k] * self.gain(k, j) * self.weight[k, j]
            errorProj[j] *= self.actFunc(self.state[j], True)
            lastK = 0
            for k, a in sorted(self.gater):
                if lastK < k and j < k and j == self.gater[k, a]:
                    lastK = k
                    errorResp[j] += errorResp[k] * self.theTerm(j, k, self.state, self.activation)
            errorResp[j] = errorProj[j] + self.actFunc(self.state[j], True) * errorResp[j]
        for (j, i), t in self.trace.items():
            self.weight[j, i] += learningRate * errorResp[j] * t
            if j < self.numUnits - self.numOutputs:
                self.weight[j, i] += learningRate * (errorProj[j] - errorResp[j]) * t
                for (l, m, k), e in self.extendedTrace.items():
                    if l == j and m == i:
                        self.weight[j, i] += learningRate * errorResp[k] * e
