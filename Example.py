import LSTM_g
#the network architecture used in the first Distracted Sequence Recall experiment
#numInputs is 11 instead of 10 because the last "input unit" is a bias unit
specString = "11, 4, 1, 1"
for memoryBlock in range(8):
    specString += "\n" + str(memoryBlock) + ", 1, 1, 1"
for memoryBlock in range(8):
    specString += "\n" + str(memoryBlock) + ", " + str(memoryBlock) + ", 0"
specString += "\n0, 8"
net = LSTM_g.LSTM_g(specString)
print net.toString(True)
