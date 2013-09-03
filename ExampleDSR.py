import LSTM_g, random
#the network architecture used in the first distracted sequence recall experiment
#numInputs is 11 instead of 10 because the last "input unit" is a bias unit
specString = "11, 4, 1, 1"
for memoryBlock in range(8):
    specString += "\n" + str(memoryBlock) + ", 1, 1, 1"
for memoryBlock in range(8):
    specString += "\n" + str(memoryBlock) + ", " + str(memoryBlock) + ", 0"
specString += "\n0, 8"
net = LSTM_g.LSTM_g(specString)
learnRate = .1#.05
targetSymbols = [2, 4, 7, 8]
distractorSymbols = [3, 5, 6, 9]
promptSymbols = [0, 1]
trial = 0
while True:
    trial += 1
    sequence = [random.choice(distractorSymbols) for i in range(22)]
    targetIndices = [random.randint(0, 3), random.randint(0, 3)]
    targetPositions = sorted(random.sample(range(22), 2))#Kevin forgot to sort targetIdx's
    for i in range(2):
        sequence[targetPositions[i]] = targetSymbols[targetIndices[i]]
    sequence.extend(promptSymbols)
    distractorsCorrect = targetsCorrect = 0
    for i in range(24):
        inputs = [0] * 11
        inputs[sequence[i]] = 1
        inputs[10] = 1#bias
        target = [0] * 4
        if i == 22:
            target[targetIndices[0]] = 1
        if i == 23:
            target[targetIndices[1]] = 1
        if [round(output) for output in net.step(inputs, True)] == target:#clearValues
            if i >= 22:
                targetsCorrect += 1
            else:
                distractorsCorrect += 1
        if i >= 22:
            net.learn(target, learnRate)
    if trial % 1 == 0:
        print str(trial) + ": " + str(distractorsCorrect) + "/22 " + str(targetsCorrect) + "/2"
