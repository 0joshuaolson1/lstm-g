Introduction

Among artificial neural networks, the recurrent Long Short-Term Memory (LSTM) architecture and algorithm has a large amount of empirical support for its ability to efficiently learn and generalize despite noise in and time lags between relevant inputs, and on several problems for which other network types lack the power. See http://www.felixgers.de/papers/phd.pdf for a comprehensive overview of the state-of-the-art model. In 2010, LSTM was generalized and simplified in "A generalized LSTM-like training algorithm for second-order recurrent neural networks" to require only one type of node while preserving the space and time locality of LSTM's back-propagation-like learning algorithm.

Code

This library, initially only in Python, is an implementation of LSTM-g from Derek Monner's paper. See http://www.cs.umd.edu/~dmonner/papers/lstmg.pdf for the full source, in addition to experiments and explanations of network set-ups used in them. Some algorithmic details are still vague, so communication with him will be ongoing before this is ready for debugging (and no longer so rough :) ), but all of the details known so far should be in Algorithm.png and Notes.txt.

Usage

LSTM-g.py is currently stateless. lsm(fmt) takes a string and outputs an array, while str(arr) does the opposite. A string defines the activation functions for each node and the weights and optional gating nodes for each connection, and is of the format:

j fnx_index
...
j i w gater
...

In the first sub-list, j is the index of a node and fnx_index refers to one of currently two possible activation functions: 0 is the constant function and 1 is the classic logistic sigmoid. It is assumed that j starts at 0 and increases by one for each new line; nodes are activated in the same ascending order as their indices. In the second sub-list (no blank line before it), a connection from node i to node j is initialized with real weight w and with an optional gating node whose index is gater. If no gating node is used, gater is -1. The order in which connections are defined does not matter. The learning algorithm assumes that the first n nodes (whose activations will come directly from an input data array of length n) have no incoming connections, that no node gates its self-connection, and that the weight of any self-connection is 1.

An array contains the operable and more complete state of the network in the following format:

arr[0] = {j, i: [w, gater, epsilon]}
arr[1] = {j: [s, y, f, gated = [], sigma, sigma_p, sigma_g]}
arr[2] = {j, i, k: epsilon_k}

arr[0][j, i] for tuple i, j gives (an array with) the weight, gater (-1 if none), and value of epsilon (for the learning algorithm) for the given connection from node i to node j. arr[1][j] gives the state, activation, function index, array of nodes it gates (possibly empty, usually no larger than length 1), and sigma, sigma_p, and sigma_g for the learning algorithm. arr[2][j, i, k] gives the learning algorithm's epsilon_k for when node i connects to node j and j gates a connection to k.

fwd(ar2, dat) takes a network array and returns the array after one time step given an array of input data, dat. bwd(ar2, dat, lrn) takes a network array and returns the array after one backwards-pass weight modification given the target output data array, dat, and the learning rate, lrn. For dat's length, n, the activations of the last n nodes are treated as the network's output for a time step. The learning algorithm assumes that all data points for both input and output are in the range [0, 1].

Discussion

While the point of LSTM-g is to allow LSTM-like architectures, the choice of how to connect nodes and gate those connections is left to the user. Currently, a reading of the LSTM-g paper is pretty much necessary in order to understand how to properly build a high-performance network; there are no functions yet to build a standard network automatically, and Derek Monner reinterpreted gating in the standard LSTM model from being on activations to being on connections.

Don't take this to court, but it might be fine to perform multiple input-to-output forward passes between backward passes, since information used to adjust weights is updated every forward pass. On the other hand, error information is currently updated whether in training or testing. Also related to error, there are currently no functions to calculate error; there are different ways to do so, but none of them were necessary to implement the learning algorithm. Neither are there functions for momentum, alpha stepping, bagging, boosting, or any of that. There's a lot more that could be done...

Code-level documentation and efficiency are hopefully in the future, as well as ports to Java and C. Any others? (languages or contributors, either way :) )

Idea

Perhaps LSTM-g can be efficiently implemented in hardware, like an FPGA?