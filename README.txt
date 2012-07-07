Introduction

Among artificial neural networks, the recurrent Long Short-Term Memory (LSTM) architecture and algorithm has a large amount of empirical support for its ability to efficiently learn and generalize despite noise in and time lags between relevant inputs, and on several problems for which other network types lack the power. See http://www.felixgers.de/papers/phd.pdf for a comprehensive overview of the state-of-the-art model, or http://etd.uwc.ac.za/usrfiles/modules/etd/docs/etd_init_3937_1174040706.pdf for a terser explanation. In 2010, LSTM was generalized and simplified in "A generalized LSTM-like training algorithm for second-order recurrent neural networks" to require only one type of node while preserving the space and time locality of LSTM's back-propagation-like learning algorithm.

Code

This library, initially only in Python, is an implementation of LSTM-g from Derek Monner's paper. See http://www.cs.umd.edu/~dmonner/papers/lstmg.pdf for the full source, in addition to experiments and explanations of network set-ups used in them. Or, given Science Direct access, http://www.sciencedirect.com/science/article/pii/S0893608011002036 has nicer typesetting. This library includes code to automatically build architectures, which will be used to test for identical weight changes to those in D. Monner's Java implementation (https://bitbucket.org/dmonner/xlbp/src).

Usage - Manual Building

The class constructor LSTM_g(netSpec) takes a string, which defines the current states and activation functions for each node and the weights and optional gating nodes for each connection, and additionally stores the internal epsilons related to training. It is of the format:

0

j s fnx_index
...

j i w gater epsilon
...

j i k epsilon_k
...

In the first sub-list (which follows a zero and a blank line to specify manual building), j is the integer index of a node, s is the real state, and fnx_index refers to an activation function; 0 is the classic logistic sigmoid, the only option for now. A node's activation follows from its state and function type and thus is never specified. It is assumed that j starts at 0 and increases by one for each new line; nodes are activated in the same ascending order as their indices.

In the second sub-list (separated from the previous sub-list by a blank line), a connection from node i to node j is initialized with real weight w and with an optional gating node whose index is gater. If no gating node is used, gater is -1. The values of the epsilons (and epsilon_ks, addressed below) are important only if the network will be trained in the future. If so, they should be zero if the network is new, or else they should have the values from a previous network's toString() method (explained hereafter). The order in which connections are defined does not matter.

In the third sub-list (again preceded by a blank line), an epsilon_k must be provided for each combination of nodes j, i, and k, where there is a connection from i to j and j gates a connection into k.

The learning algorithm assumes that the first n nodes (whose activations will come directly from an input data array of length n) have no incoming connections, that no node gates its self-connection, and that the weight of any self-connection is 1.

Usage - Automatic Building

The class constructor LSTM_g(netSpec) takes a string, which defines which types of connections are present between the input nodes, optional bias node, memory blocks, and output nodes. In addition, layer groupings for memory blocks can be specified in order to change the order of activation of blocks' nodes. The string is of the format:

1

numberOfInputs numberOfOutputs inputToOutputConnections biasToOutputConnections

inputReceivingBlock
...

biasReceivingBlock
...

firstBlockInLayer layerSize
...

sendingBlock receivingBlock connectionType
...

outputSendingBlock
...

Following the one and blank line to specify automatic building, the first four arguments are integers specifying respectively the number of input nodes, the number of output nodes, whether all input nodes should project connections to all output nodes (0 for no, 1 for yes), and whether all output nodes should be biased (also 0 or 1). All other sections are sub-lists, each separated from the previous section by a blank line, and can be empty (i.e. if one or more lists other than the last are empty, there will be two or more blank lines), although omitting certain sub-lists makes no sense. The lines in each sub-list may appear in any order. Wherever a memory block index is required, n blocks are expected to be numbered from 0 to n - 1, where a higher index means that that block will be activated later in a forward time step than all blocks with lower indices (excluding blocks in the same layer).

The first sub-list is all blocks that will receive connections from all input nodes.

The second sub-list is all blocks that will receive connections from a bias node.

Each line in the third sub-list specifies an optional layer grouping for any number of blocks (hybrid, partially layered networks are possible). By default, the input gate, forget gate, memory cell, and output gate of a memory block are activated one right after another, after all previous blocks and before all blocks afterward. However, if two or more blocks are specified as being in the same layer (where firstBlockInLayer is the index of the first block and the next layerSize - 1 blocks make up the remainder of the layer), their input gates are activated one right after another, then their forget gates together, then their memory cells, then their output gates. This is the order of activation of nodes in the original LSTM architecture according to Monner's paper. Blocks in a layer will still have to have their connections to each other specified, if there are any, whether all-to-all or otherwise.

The fourth sub-list allows three different types of connections from sendingBlock to receivingBlock. If connectionType is 0, it will be a downstream connection (useful for creating the two-layer network in the LSTM-g paper, for example). If the type is 1, it will be a gated peephole connection such as those used in the gated recurrence architecture in the paper. If the type is 2, it will be an ungated peephole connection, needed to create the paper's ungated recurrence architecture or to give a memory block its own peephole connections. Note that, unlike with manual building, the source of a connection is listed before the destination; this is hopefully more intuitive, while the reverse stays consistent with the paper's j-i notation.

The fifth sub-list is all blocks that will project connections to all output nodes.

Currently, all connections except for the constant-weight-of-one memory cell self-connections are given randomized weights in the range [-0.1, 0.1), just like in the paper.

Usage - API

toString() returns a string of the same format as that described in the manual building section, specifying the current state, function choices, epsilons, and topology of the network.

step(input, mode) takes an array of input data and propagates the input nodes' activations through the network for a single time step. It is the user's responsibility to ensure that the input array's length is consistent with the intended number of input nodes (and optional bias node, elaborated further down). If the mode is TRAIN_MODE, eligibility traces that are used in the adjust method (explained below) are updated. Such calculations are skipped if the mode is TEST_MODE.

getOutput(length) takes the intended number of output data elements and returns an array of the most recent activations of that number of nodes.

adjust(target, learnRate) modifies the network's weights according to the current target output array and the real learning rate.

The learning algorithm assumes that all data points for output are in the [0, 1] range. To incorporate bias inputs (one suggestion is to bias all non-input units) in manual building, always include an input data point of value 1 and dedicate an input node to share that value with other nodes. In automatic building, if either biasToOutputConnections is 1 or biasReceivingBlock is non-empty, the user is likewise expected to include a constant input of one, appended at the end of any input array (so it would have length numberOfInputs + 1).

Discussion

While the point of LSTM-g is to allow LSTM-like architectures, the choice of how to connect nodes and gate those connections is left to the user when manually building. Unless the autmatic building functionality is used, a reading of the LSTM-g paper is pretty much necessary in order to understand how to properly build a high-performance network; note that D. Monner reinterpreted gating in the standard LSTM model from being on activations to being on connections. Even still, the types of connections currently allowed by automatic building are dictated by the graphics in the paper.

It is possible to perform multiple input-to-output forward passes between backward passes, since information used to adjust weights is updated every forward pass.

There are currently no functions to calculate error; there are different ways to do so, but none of them are necessary to implement the learning algorithm. Neither are there functions for batch training, momentum, alpha stepping, bagging, boosting, or any of that. There's a lot more that could be done...

One long-term goal of this library is efficiency. The current Python code is intended to be functional, but it is also meant to be directly illustrative of the definition of Generalized LSTM as found in Monner's paper. Strictly speaking, the fastest platform-independent library implementation would probably be a program that generates a C/C++ header file with a hard-coded network and optimized activation function approximations. However, an almost equally efficient but more flexible solution would either be loadable as a dynamic library (for local applications) or run as an application with a API through UDP or TCP sockets (for remote use).

Idea

Perhaps LSTM-g can be efficiently implemented in hardware, like an FPGA?
