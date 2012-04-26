import math
class LSTM_g:
    VALUE_MODE = 0
    DERIV_MODE = 1
    TRAIN_MODE = 0
    TEST_MODE = 1
    net = [{}, {}, {}]
    def constFunc(value, mode):
        return value * (1 - mode)
    def logisticFunc(value, mode):
        def logValue(value):
            return 1. / (1 + math.exp(-value))
        def logDeriv(value):
            result = logValue(value)
            return result * (1 - result)
        if mode == VALUE_MODE:
            return logValue(value)
        return logDeriv(value)

    def getFuncs():
        return [constFunc, logisticFunc]
    def getNodes():
        return net[0]
    def getConnections():
        return net[1]
    def getEpsilonKs():
        return net[2]

    def getState(j):
        return net[0][j][0]
    def setState(j, s):
        net[0][j][0] = s
    def getAct(j):
        return net[0][j][1]
    def setAct(j, y):
        net[0][j][1] = y
    def getFuncIndex(j):
        return net[0][j][2]
    def setFuncIndex(j, f):
        net[0][j][2] = f
    def getGatedArray(j):
        return net[0][j][3]
    def setGatedArray(j, gated):
        net[0][j][3] = gated
    def getDelta(j):
        return net[0][j][4]
    def setDelta(j, delta):
        net[0][j][4] = delta
    def getDeltaP(j):
        return net[0][j][5]
    def setDeltaP(j, deltaP):
        net[0][j][5] = deltaP
    def getDeltaG(j):
        return net[0][j][6]
    def setDeltaG(j, deltaG):
        net[0][j][6] = deltaG
    def getWeight(j, i):
        return net[1][j, i][0]
    def setWeight(j, i, w):
        net[1][j, i][0] = w
    def getGater(j, i):
        return net[1][j, i][1]
    def setGater(j, i, gater):
        net[1][j, i][1] = gater
    def getEpsilon(j, i):
        return net[1][j, i][2]
    def setEpsilon(j, i, epsilon):
        net[1][j, i][2] = epsilon
    def getEpsilonK(j, i, k):
        return net[2][j, i, k]
    def setEpsilonK(j, i, k, epsilonK):
        net[2][j, i, k] = epsilonK

    def gain(j, i):
        if getGater(j, i) < 0:
            return 1
        return getAct(getGater(j, i))
    def calcState(j):
        if (j, j) in getConnections():
            setState(j, getState(j) * gain(j, j) * getWeight(j, j))
        else:
            setState(j, 0)
        for i in getNodes():
            if j != i and (j, i) in getConnections():
                setState(j, getState(j) + gain(j, i) * getWeight(j, i) * getAct(i))
    def calcEpsilon(j, i):
        if (j, j) in getConnections():
            setEpsilon(j, i, getEpsilon(j, i) * gain(j, j) * getWeight(j, j))
        else:
            setEpsilon(j, i, 0)
        setEpsilon(j, i, getEpsilon(j, i) + gain(j, i) * getAct(i))
    def calcEpsilonK(j, i, k):
        if (k, k) in getConnections():
            setEpsilonK(j, i, k, getEpsilonK(j, i, k) * gain(k, k) * getWeight(k, k))
            if (k, k) in getGatedArray(j):
                setEpsilonK(j, i, k, getEpsilonK(j, i, k) + getFuncs()[getFuncindex(j)](getState(j), DERIV_MODE) * getEpsilon(j, i) * getWeight(k, k) * getState(k))
        else:
            setEpsilonK(j, i, k, 0)
        for p, a in getGatedArray(j):
            if p == k and a != k:
                setEpsilonK(j, i, k, getEpsilonK(j, i, k) + getFuncs()[getFuncIndex(j)](getState(j), DERIV_MODE) * getEpsilon(j, i) * getWeight(k, a) * getAct(a))
    def updateTraces(j):
        for i in getNodes():
            if (j, i) in getConnections():
                calcEpsilon(j, i)
                m = -1
                for k, l in getGatedArray(j):
                    if m != k:
                        m = k
                        calcEpsilonK(j, i, k)
    def calcDeltaP(j):
        setDeltaP(j, 0)
        for k in range(j + 1, len(getNodes())):
            if (k, j) in getConnections():
                setDeltaP(j, getDeltaP(j) + getDelta(k) * gain(k, j) * getWeight(k, j))
        setDeltaP(j, getDeltaP(j) * getFuncs()[getFuncIndex(j)](getState(j), DERIV_MODE))
    def calcDeltaG(j):
        setDeltaG(j, 0)
        m = -1
        for k, l in getGatedArray(j):
            if m != k and k > j:
                m = k
                if (k, k) in getConnections():
                    setDeltaG(j, getDeltaG(j) + getDelta(k) * getWeight(k, k) * getState(k))
                for p, a in getGatedArray(j):
                    if p == k and a != k:
                        setDeltaG(j, getDeltaG(j) + getDelta(k) * getWeight(k, a) * getAct(a))
        setDeltaG(j, getDeltaG(j) * getFuncs()[getFuncIndex(j)](getState(j), DERIV_MODE))
    def updateWeights(learnRate):
        for j, i in getConnections():
            setWeight(j, i, getWeight(j, i) + learnRate * getDeltaP(j) * getEpsilon(j, i))
            m = -1
            for k, l in getGatedArray(j):
                if m != k and k > j:
                    setWeight(j, i, getWeight(j, i) + learnRate * getDelta(k) * getEpsilonK(j, i, k))

    def __init__(self, netSpec):
        net = [{}, {}, {}]
        for line in netSpec.split("\n"):
            args = line.split(" ")
            if len(args) < 4:
                setState(args[0], args[1])
                setAct(args[0], getFuncs()[args[2]](args[1], VALUE_MODE))
                setFuncIndex(args[0], args[2])
                setGatedArray(args[0], [])
                setDelta(args[0], 0)
                setDeltaP(args[0], 0)
                setDeltaG(args[0], 0)
            else:
                setWeight(args[0], args[1], args[2])
                setGater(args[0], args[1], args[3])
                setEpsilon(args[0], args[1], 0)
                if args[3] > -1:
                    gatedArray = getGatedArray(args[3])
                    gatedArray.append((args[0], args[1]))
                    setGatedArray(args[3], gatedArray)
        for j, i in getConnections():
            for k, l in getGatedArray(j):
                setEpsilonK(j, i, k, 0)
    def toString():
        netSpec = ""
        for j in getNodes():
            netSpec += "\n" + j + " " + getState(j) + " " + getFuncIndex(j)
        for i in getNodes():
            for j in getNodes():
                if (j, i) in getConnections():
                    netSpec += "\n" + j + " " + i + " " + getWeight(j, i) + " " + getGater(j, i)
        return netSpec[1:]
    def step(input, mode):
        for j in input:
            setAct(j, input[j])
        for j in range(len(input), len(input[1])):
            calcState(j)
            setAct(j, getFuncs()[getFuncIndex(j)](getState(j), VALUE_MODE))
            if mode == TRAIN_MODE:
                updateTraces(j)
    def getOutput(length):
        result = []
        for j in range(len(getNodes()) - length, len(getNodes())):
            result.append(getAct(j))
        return result
    def adjust(target, learnRate):
        for j in range(len(getNodes()) - len(target), len(getNodes())):
            setDelta(j, target[j] - getAct(j))
        for j in reversed(range(len(getNodes()) - len(target))):
            calcDeltaP(j)
            calcDeltaG(j)
            setDelta(j, getDeltaP(j) + getDeltaG(j))
        updateWeights(learnRate)