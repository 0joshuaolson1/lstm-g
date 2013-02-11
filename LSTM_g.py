#comments assume unfamiliarity with Python 2.7.3, and can obviously be removed

import math, random, os
class LSTM_g:

#the activation function used is the logistic sigmoid, with range (0, 1) and a neat derivative
#the bias (0 if omitted) is for memory cell biases, which are not added to the state directly
#Python requires "self" as the first parameter of class methods
    def actFunc(self, s, derivative, bias=0):
        value = 1 / (1 + math.exp(-s - bias))
        if derivative:
            value *= 1 - value
        return value

#Eq. 14
    def gain(self, j, i):

#"in" checks if (j, i) is a key in the network's dictionary (associative array) named gater
        if (j, i) in self.gater:
            return self.activation[self.gater[j, i]]

        if (j, i) in self.weight:
            return 1

#returns 0 if there is no w_ji to account for nonexistent self-connections (Eqs. 15, 17, and 18)
        return 0

#the part of Eqs. 18 and 22 in parentheses
#cached states and activations are used for extended traces, and current ones for responsibilities
    def theTerm(self, j, k, state, activation):
        term = 0
        if (k, k) in self.gater and j == self.gater[k, k]:
            term = state[k]

#for all dictionary keys meeting the properties checked for below
#this is a common technique in the code, which is slow but clear
        for l, a in self.gater:

#checks if the given activations are the current ones, because the cached ones take two parameters
            if l == k and a != k and j == self.gater[k, a] and activation is self.activation:
                term += self.weight[k, a] * activation[a]
            elif l == k and a != k and j == self.gater[k, a]:
                term += self.weight[k, a] * activation[k, a]

        return term

#manual building takes a list of lists of string parameters, otherwise in the format in the readme
    def build(self, specData):

#assignments with commas like this are respective, and {} is an empty dictionary
        self.state, self.activation, self.weight, self.gater = {}, {}, {}, {}
        self.trace, self.extendedTrace = {}, {}

#the first two parameters in the first line of specData, converted to integers
        self.numInputs, self.numOutputs = int(specData[0][0]), int(specData[0][1])

#assumes the states etc. are not included until they are seen
#this is necessary because there are two types of lines with four parameters
        newNetwork = True

#args is the parameter list from each line after the first (the first is excluded by [1:])
        for args in specData[1:]:

#if a two-parameter line (for states) is seen, the other optional lines are now expected
            if len(args) < 3:
                newNetwork = False
                self.state[int(args[0])] = float(args[1])
                self.activation[int(args[0])] = self.actFunc(self.state[int(args[0])], False)

#args[0] and args[1] are j and i respectively
            if newNetwork:
                self.weight[int(args[0]), int(args[1])] = float(args[2])

#gater associates a connection with its gating unit, or gater (but -1 means there is none)
                if args[3] != "-1":
                    self.gater[int(args[0]), int(args[1])] = int(args[3])

#traces are unneeded for self-connections, and are initially 0
#these will be overwritten if traces are specified
                if args[0] != args[1]:
                    self.trace[int(args[0]), int(args[1])] = 0

            elif len(args) > 3:
                self.extendedTrace[int(args[0]), int(args[1]), int(args[2])] = float(args[3])
            elif len(args) > 2:
                self.trace[int(args[0]), int(args[1])] = float(args[2])

#if the optional lines were omitted, states, activations, and extended traces are initially 0
        if newNetwork:
            for j, i in self.trace:
                self.state[j] = self.activation[j] = 0

#the combinations of j, i, and k that extended traces are defined for are deduced from connections
                for k, a in self.gater:
                    if j < k and j == self.gater[k, a]:
                        self.extendedTrace[j, i, k] = 0

#units start from 0, so the number of units is one more than the state dictionary's largest key
        self.numUnits = max(self.state) + 1

#automatic building changes the high-level list of parameter lists into what build can use
    def toLowLevel(self, specData):

#adds a list of connection information to the end of specData
#repr is used so that the weight (in [-.1, .1]) converts back to the same float in manual building
        def addConnection(j, i, g=-1):
            specData.append([str(j), str(i), repr(random.uniform(-.1, .1)), str(g)])

#gives the indices of units in a memory block, checking if blockNum falls in a layer grouping
        def unitsInBlock(blockNum):
            for firstBlockInLayer, layerSize in layerData:

#Python requires both inequalities to be true
                if 0 <= blockNum - firstBlockInLayer < layerSize:

                    offset = numInputs + 3 * firstBlockInLayer + blockNum

#range gives an ordered list of numbers from the first parameter to just less than the second
#the third parameter determines the step size between entries
                    return range(offset, offset + 4 * layerSize, layerSize)
            return range(numInputs + 4 * blockNum, numInputs + 4 * blockNum + 4, 4)

        numInputs, numOutputs = int(specData[0][0]), int(specData[0][1])

#inputToOutput is only used once, so it can stay as a string
        inputToOutput, biasOutput = specData[0][2], int(specData[0][3])

#lastInput is the number of true input units, and also the index of the bias unit if there is one
#if biasOutput is 1, a bias will be needed, so lastInput is one less than numInputs
        lastInput = numInputs - biasOutput

#these store automatic building information that will be lost when specData is overwritten
#[] is an empty list
        blockData, connections, layerData = [], [], []
        for args in specData[1:]:
            if len(args) > 3:
                blockData.append([int(args[0]), args[1], args[2], args[3]])

#if any memory block is biased, a bias unit will be needed
                if args[3] == "1":
                    lastInput = numInputs - 1

            elif len(args) > 2:
                connections.append([int(args[0]), int(args[1]), int(args[2])])
            else:
                layerData.append([int(args[0]), int(args[1])])

#blockData is a list of lists, but max gives the list with the largest first entry
        numUnits = numInputs + 4 * max(blockData)[0] + 4 + numOutputs

#the first two parameters in the high-level list's first line make the low-level list's first line
#[:] prevents the creation of a local copy instead
        specData[:] = [specData[0][:2]]

#connections to output units
        for outputUnit in range(numUnits - numOutputs, numUnits):

#connections from input units
            if inputToOutput == "1":

#this range has one parameter, giving a list of values from 0 to lastInput - 1 in steps of 1
                for inputUnit in range(lastInput):

                    addConnection(outputUnit, inputUnit)

#connection from the bias unit
            if biasOutput > 0:
                addConnection(outputUnit, lastInput)

#connections between memory blocks and non-hidden units
        for memoryBlock, receiveInput, sendToOutput, biased in blockData:
            inputGate, forgetGate, memoryCell, outputGate = unitsInBlock(memoryBlock)

#connections from input units, with the input gate gating the memory cell input
            if receiveInput == "1":
                for inputUnit in range(lastInput):
                    for unit in [inputGate, forgetGate, outputGate]:
                        addConnection(unit, inputUnit)
                    addConnection(memoryCell, inputUnit, inputGate)

#connections to output units, all gated by the output gate
            if sendToOutput == "1":
                for outputUnit in range(numUnits - numOutputs, numUnits):
                    addConnection(outputUnit, memoryCell, outputGate)

#biases for memory block units
            if biased == "1":
                for unit in unitsInBlock(memoryBlock):
                    addConnection(unit, lastInput)

#the memory cell's self-connection has a weight of 1 and is gated by the forget gate
            specData.append([str(memoryCell), str(memoryCell), "1", str(forgetGate)])

#connections between and within memory blocks
        for toBlock, fromBlock, connectionType in connections:
            toIGate, toFGate, toMCell, toOGate = unitsInBlock(toBlock)
            fromIGate, fromFGate, fromMCell, fromOGate = unitsInBlock(fromBlock)

#all connection types have the following kind of connection for non-memory-cell receiving units
            for toUnit in [toIGate, toFGate, toOGate]:

#the following connection is ungated if type 0, as seen in Fig. 4
                if connectionType < 1:
                    fromOGate = -1
                addConnection(toUnit, fromMCell, fromOGate)

#Fig. 6 uses toIGate instead of fromOGate as the gater in this part of type 2 connections
            if connectionType > 1:
                addConnection(toMCell, fromMCell, toIGate)

#class constructor
    def __init__(self, specString):
        specData = []

#fills specData with specString's non-empty lines as lines of parameter lists
        for line in specString.splitlines():
            if line.strip() != "":

#the appended list is created using Python's list comprehension, like set-builder notation in math
                specData.append([arg.strip() for arg in line.split(",")])

#if specData has a high-level network specification, toLowLevel readies it for build
        if len(specData[0]) > 2:
            self.toLowLevel(specData)
        self.build(specData)

#returns a low-level string, but only with weights and gaters if newNetwork is True
#the newline used is the operating system's default if omitted
#note that repr is used for floats so that they can be exactly recovered by the constructor
    def toString(self, newNetwork, newline=os.linesep):
        specString = str(self.numInputs) + ", " + str(self.numOutputs)

#loops through the keys and values together, in ascending order (sorted by j, then by i...)
        for (j, i), w in sorted(self.weight.items()):

            specString += newline + str(j) + ", " + str(i) + ", " + repr(w) + ", -1"

#replaces the last two characters in the string, "-1", with the gater, since it exists
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

#given a list of input values, does a forward pass and returns a list of output unit activations
#calculates states (and activations from those) in the order of activation, while caching values
#cached values are used in trace and extended trace updates
    def step(self, inputs):

#input unit activations come directly from inputs
        for j in range(self.numInputs):
            self.activation[j] = inputs[j]

#cached states are just the states copied from the last time step, but the others are cached below
        oldState, oldActivation, oldGain = self.state.copy(), {}, {}

#units are indexed by order of activation
        for j in sorted(self.state):

#cached self-connection gain
            oldGain[j] = self.gain(j, j)

            for l, i in self.trace:
                if l == j:

#oldActivation's first parameter is the unit denoting the "time" the value is cached
                    oldActivation[j, i], oldGain[j, i] = self.activation[i], self.gain(j, i)

#first term in Eq. 15
            self.state[j] *= self.gain(j, j)

#will be used if j is a self-connected unit with an ungated connection from an input unit
            bias = 0

#loops through units with connections to j (i != j for traces) for the second term in Eq. 15
            for (l, i), t in self.trace.items():

#if j does not receive a bias connection from i (j is not self-connected if the gain on j, j is 0)
                if l == j and (oldGain[j] == 0 or i >= self.numInputs or (j, i) in self.gater):

#the inner part of the second term in Eq. 15
                    self.state[j] += self.gain(j, i) * self.weight[j, i] * self.activation[i]

#Eq. 17 (might as well update traces in the same loop)
                    self.trace[j, i] = oldGain[j] * t + oldGain[j, i] * oldActivation[j, i]

#else the bias term in the activation function is used and the trace is specially defined
                elif l == j:
                    bias = self.trace[j, i] = self.activation[i]

#Eq. 16
            self.activation[j] = self.actFunc(self.state[j], False, bias)

#Eq. 18 (the trace used in any extended trace calculation must be current)
        for (j, i, k), e in self.extendedTrace.items():
            terms = self.trace[j, i] * self.theTerm(j, k, oldState, oldActivation)

#True means the derivative of the activation function is used
            self.extendedTrace[j, i, k] = oldGain[k] * e + self.actFunc(oldState[j], True) * terms

#returns network output
        return [self.activation[j] for j in range(self.numUnits - self.numOutputs, self.numUnits)]

#cross-entropy gives an error difference between target values and the output units' activations
    def getError(self, targets):
        error = 0
        for j in range(self.numUnits - self.numOutputs, self.numUnits):
            t, y = targets[j + self.numOutputs - self.numUnits], self.activation[j]

#base 2 logarithms
            error += t * math.log(y, 2) + (1 - t) * math.log(1 - y, 2)

        return error

#given a list of target values, does a backward pass and adjusts weights with the learning rate
    def learn(self, targets, learningRate=.1):

#projected responsibility and (full) responsibility respectively
        errorProj, errorResp = {}, {}

#Eq. 10
        for j in range(self.numUnits - self.numOutputs, self.numUnits):
            errorResp[j] = targets[j + self.numOutputs - self.numUnits] - self.activation[j]

#error responsibilities are calculated in the reverse order of activation
        for j in reversed(range(self.numInputs, self.numUnits - self.numOutputs)):

#preparation for their sums in Eqs. 21 and 22
#gating responsibility will be temporarily stored in errorResp, since it is only used in Eq. 23
            errorProj[j] = errorResp[j] = 0

#summation in Eq. 21, looping through P_j (Eq. 19)
            for k, l in self.trace:
                if l == j and j < k:
                    errorProj[j] += errorResp[k] * self.gain(k, j) * self.weight[k, j]

            errorProj[j] *= self.actFunc(self.state[j], True)

#summation in Eq. 22, looping through G_j (Eq. 20) and making sure no k is repeated
            lastK = 0
            for k, a in sorted(self.gater):
                if lastK < k and j < k and j == self.gater[k, a]:
                    lastK = k
                    errorResp[j] += errorResp[k] * self.theTerm(j, k, self.state, self.activation)

#Eq. 23, but the gating responsibility had not yet been multipled by the activation derivative
            errorResp[j] = errorProj[j] + self.actFunc(self.state[j], True) * errorResp[j]

#Eq. 24 (weight changes can be in any order once the responsibilities used are known)
        for (j, i), t in self.trace.items():

#if j is not an output unit
            if j < self.numUnits - self.numOutputs:

#first term in Eq. 24
                self.weight[j, i] += learningRate * errorProj[j] * t

#second term in Eq. 24 (this is another way to loop through G_j, when i is fixed)
                for (l, m, k), e in self.extendedTrace.items():
                    if l == j and m == i:
                        self.weight[j, i] += learningRate * errorResp[k] * e

#else Eq. 13, since j is an output unit
            else:
                self.weight[j, i] += learningRate * errorResp[j] * t
