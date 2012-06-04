Introduction

Among artificial neural networks, the recurrent Long Short-Term Memory (LSTM) architecture and algorithm has a large amount of empirical support for its ability to efficiently learn and generalize despite noise in and time lags between relevant inputs, and on several problems for which other network types lack the power. See http://www.felixgers.de/papers/phd.pdf for a comprehensive overview of the state-of-the-art model. In 2010, LSTM was generalized and simplified in "A generalized LSTM-like training algorithm for second-order recurrent neural networks" to require only one type of node while preserving the space and time locality of LSTM's back-propagation-like learning algorithm.

Code

This library, initially only in Python, is an implementation of LSTM-g from Derek Monner's paper. See http://www.cs.umd.edu/~dmonner/papers/lstmg.pdf for the full source, in addition to experiments and explanations of network set-ups used in them. Currently, code to automatically build the four architectures in the paper is in the works, and will be used to test for identical weight changes to those in D. Monner's Java implementation (https://bitbucket.org/dmonner/xlbp/src).

Usage

The class constructor LSTM_g(netSpec) takes a string, which defines the current states and activation functions for each node and the weights and optional gating nodes for each connection, and additionally stores the internal ds and epsilons related to training. It is of the format:

j s fnx_index
...
j i w gater epsilon
...
j i k epsilon_k
...

In the first sub-list, j is the integer index of a node, s is the real state, and fnx_index refers to one of currently two possible activation functions: 0 is the constant function and 1 is the classic logistic sigmoid. A node's activation follows from its state and function type and thus is never specified. It is assumed that j starts at 0 and increases by one for each new line; nodes are activated in the same ascending order as their indices.

In the second sub-list (no blank line before it), a connection from node i to node j is initialized with real weight w and with an optional gating node whose index is gater. If no gating node is used, gater is -1. The values of the epsilons (and epsilon_ks, addressed below) are important only if the network will be trained in the future. If so, they should be zero if the network is new, or else they should have the values from a previous network's toString() method (explained hereafter). The order in which connections are defined does not matter.

In the third sub-list, again immediately following, an epsilon_k must be provided for each combination of nodes j, i, and k, where there is a connection from i to j and j gates a connection into k.

The learning algorithm assumes that the first n nodes (whose activations will come directly from an input data array of length n) have no incoming connections, that no node gates its self-connection, and that the weight of any self-connection is 1.

toString() returns a string of the same format, specifying the current state, function choices, epsilons, and topology of the network.

step(input, mode) takes an array of input data and propagates the input nodes' activations through the network for a single time step. It is the user's responsibility to ensure that the input array's length is consistent with the intended number of input nodes. If the mode is TRAIN_MODE, then eligibility traces that are used in the adjust method (explained below) are updated. Such calculations are skipped if the mode is TEST_MODE.

getOutput(length) takes the intended number of output data elements and returns an array of the most recent activations of that number of nodes.

adjust(target, learnRate) modifies the network's weights according to the current target output array and the real learning rate.

The learning algorithm assumes that all data points for output are in the range [0, 1]. To incorporate bias inputs (one suggestion is to bias all non-input units), always include an input data point of value 1 and dedicate an input node to share that value with other nodes.

Discussion

While the point of LSTM-g is to allow LSTM-like architectures, the choice of how to connect nodes and gate those connections is left to the user. Until the functions for automatically building certain architectures are complete, a reading of the LSTM-g paper is pretty much necessary in order to understand how to properly build a high-performance network; note that D. Monner reinterpreted gating in the standard LSTM model from being on activations to being on connections.

It is possible to perform multiple input-to-output forward passes between backward passes, since information used to adjust weights is updated every forward pass.

There are currently no functions to calculate error; there are different ways to do so, but none of them are necessary to implement the learning algorithm. Neither are there functions for batch training, momentum, alpha stepping, bagging, boosting, or any of that. There's a lot more that could be done...

One long-term goal of this library is efficiency. The current Python code is intended to be functional, but it is also meant to be directly illustrative of the definition of Generalized LSTM as found in the paper referenced above. Strictly speaking, the fastest platform-independent library implementation would probably be a program that generates a C/C++ header file with a hard-coded network and optimized activation function approximations. However, an almost equally efficient but more flexible solution would run as an application with a API through TCP sockets.

Idea

Perhaps LSTM-g can be efficiently implemented in hardware, like an FPGA?