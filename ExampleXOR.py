import LSTM_g, random
specString = "3, 1, 1, 1"
for memoryBlock in range(4):
    specString += "\n" + str(memoryBlock) + ", 1, 1, 1"
for memoryBlock in range(4):
    specString += "\n" + str(memoryBlock) + ", " + str(memoryBlock) + ", 0"
specString += "\n0, 4"
net = LSTM_g.LSTM_g(specString)
numErrors = 0
for trial in range(15000):
    inputs = [random.randint(0, 1), random.randint(0, 1), 1]
    targets = [inputs[0] ^ inputs[1]]
    #net.clear()
    outputs = net.step(inputs)
    if targets[0] != round(outputs[0]):
        numErrors += 1
    net.learn(targets, .1)
    if trial > 0 and trial % 100 == 0:
        print float(numErrors) / trial
    #if(trial % 10000 == 0):
    #    net.jiggle(raw_input("connection jiggle: "))#read network.JiggleConnections() and implement?
