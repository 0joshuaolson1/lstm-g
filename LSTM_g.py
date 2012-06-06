import math
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
        for line in netSpec:
            args = line.split(" ")
            if len(args) < 4:
                self.initNode(int(args[0]))
                self.setState(int(args[0]), float(args[1]))
                self.setAct(int(args[0]), self.getFuncs()[int(args[2])](float(args[1]), self.VALUE_MODE))
                self.setFuncIndex(int(args[0]), int(args[2]))
            elif len(args) > 4:
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

    def __init__(self, netSpec):
        lines = netSpec.split("\n")
        if len(lines) > 1:
            self.initialize(lines)
            return
        archSpec = lines[0].split(" ")
        useBiases = archSpec[0]
        inputToOutput = archSpec[1]
        numInputs = archSpec[2]
        numOutputs = archSpec[3]
        peepholeFlags = []
        peepplusFlags = []
        memCellCounts = []
        normCellCounts = []
        for index in range(4, len(archSpec), 4):
            peepholeFlags.append(int(archSpec[index]))
            peepplusFlags.append(int(archSpec[index + 1]))
            memCellCounts.append(int(archSpec[index + 2]))
            normCellCounts.append(int(archSpec[index + 3]))
        newSpec = ""
        
# useBiases inputToOutput numInputs numOutputs( peephole peepplus[0,1,2=ungated] numMemCells numNormCells)+
    def toString(self):
        netSpec = ""
        for j in self.getNodes():
            netSpec += "\n" + str(j) + " " + str(self.getState(j)) + " " + str(self.getFuncIndex(j))
        for i in self.getNodes():
            for j in self.getNodes():
                if (j, i) in self.getConnections():
                    netSpec += "\n" + str(j) + " " + str(i) + " " + str(self.getWeight(j, i)) + " " + str(self.getGater(j, i)) + " " + str(self.getEpsilon(j, i))
        m = -1
        for j, i in self.getConnections():
            for k, l in self.getGatedArray(j):
                if m != k:
                    m = k
                    netSpec += "\n" + str(j) + " " + str(i) + " " + str(k) + " " + str(self.getEpsilonK(j, i, k))
        return netSpec[1:]
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