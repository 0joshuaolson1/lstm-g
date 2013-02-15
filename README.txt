Introduction

LSTM-g stands for Generalized Long Short-Term Memory. LSTM is a class of recurrent neural network architectures (and their associated learning algorithm) with a large amount of empirical support for its ability to efficiently learn and generalize despite noise in and arbitrarily long time lags between relevant inputs, and on several problems for which other network types fail. See "Long Short-Term Memory in Recurrent Neural Networks" (http://www.felixgers.de/papers/phd.pdf) for a comprehensive overview of the state-of-the-art model. For a terser explanation, see chapters 3 and 4 of "Data Mining, Fraud Detection and Mobile Telecommunications: Call Pattern Analysis with Unsupervised Neural Networks" (http://etd.uwc.ac.za/usrfiles/modules/etd/docs/etd_init_3937_1174040706.pdf).

In 2012, Derek Monner published "A generalized LSTM-like training algorithm for second-order recurrent neural networks" (http://www.cs.umd.edu/~dmonner/papers/nn2012.pdf), reinterpreting gating from being on activations to being on connections. This broadens the class of LSTM architectures that can be trained by the learning algorithm, while giving equivalent behavior to LSTM networks with forget gates. However, LSTM-g networks with forget gates and peepholes use an additional source of error that has never been taken into account before. This additional error, plus architectural flexibility, gives LSTM-g the potential to greatly outperform LSTM (see the experiments in the paper), at least in terms of learning speed and performance-to-computation ratio.

An added bonus of the learning algorithm is that it remains spacially and temporally local like traditional LSTM: unlike Back-Propagation Through Time, Real-Time Recurrent Learning, Decoupled Extended Kalman Filters, Evolino, and other alternative LSTM training methods, weight changes only depend on information in the spacial neighborhood of each connection and since the previous time step. This locality gives the same or better complexity than any known alternatives (O(S_j) per time step and weight, where S_j is the number of memory cells in memory block j). It also gives a measure of physical plausibility, for those who may be interested in trying to draw conclusions about biological brains from LSTM experiments.

Library

The code in this library is written in Python version 2, the old, stable, unchanging branch of one of the easiest and most readable programming languages in wide use. It is intended to be a more accessible reference than Monner's Java implementation, XLBP (https://bitbucket.org/dmonner/xlbp/src), and as such has been optimized for readability and brevity, not speed. This also means that the library's methods do not check for erroneous/nonsensical parameters or usage, so read this readme carefully. Note that this has not yet been tested for identical weight changes to complex XLBP networks; no correctness is promised until then. Past communication with Monner has revealed several details that are not in or obvious from the paper.

Porting this to other languages is highly encouraged. Monner and I would love to hear about uses anyone finds for the documentation and/or code, and feel free to ask questions! Remember to cite the LSTM-g paper if this is used in research:

D. Monner and J.A. Reggia (2012). A generalized LSTM-like training algorithm for second-order recurrent neural networks. Neural Networks, 25, pp 70-83.

Note that some languages may have more than one way to convert floating point numbers to strings. Some ways may not retain enough precision to give the same value when turned back into a number. In this library's code, the repr method is used instead of the str method when randomizing weights for automatic building (see Usage - Automatic Building) and in the toString method (see Usage - API).

I am working on a C program (https://github.com/MrMormon/lstm-g-hardcoder) that generates C/C++ headers with hardcoded networks for maximum platform-independent efficiency. The computational complexity is being improved over the loop-heavy methods used here, and calculated values are reused where possible. Maybe someone can try parallelizing it or utilizing cache memory or GPUs (although possible speedup is architecture-dependent). Or try other ideas, such as using tables of precomputed, interpolated activation function approximations, extending LSTM-g in ways that LSTM has been modified, or even getting a network working on an FPGA or neurocomputer...

Usage - Manual Building

The class constructor LSTM_g(specString) takes a string in comma-separated values (csv) format, defining the number of input and output units, the connections between input, output, and hidden units, and the current weights and gating units of those connections. For a network that has been previously built and run, the states, eligibility traces, and extended eligibility traces can also be defined (normally taken from strings from the toString method - see Usage - API):

numInputs, numOutputs
j, i, w, g
[j, s
j, i, t
j, i, k, e]

Each line after the first represents any number of unordered consecutive lines of the same format. Blank lines are allowed anywhere, and any amount of non-line-separator whitespace is allowed before and after any comma or line (space after a comma is not required). j, i, and k refer to units, which are numbered by order of activation from 0 to (number of units - 1). The input and output units are 0 through (numInputs - 1) and (number of units - numOutputs) through (number of units - 1) respectively.

The first line is self-explanatory.

In the second line, w is the weight of the connection from i to j. Unit g may gate this connection; use -1 if it does not have a gater.

In the third line, s is j's state. Neither activations nor input unit states are specified, since j's activation follows from its state, and input unit activations come directly from network input.

In the fourth line, t is the trace for the connection from i to j, where i does not equal j.

In the fifth line, e is the extended trace for the combination of j, i, and k such that there is a connection from i to j, i does not equal j, j gates a connection from some unit to k, and k is activated after j.

Usage - Automatic Building

The constructor also takes a csv string of the following high-level format, defining a new network with connections between input units, an optional bias unit, memory blocks, and output units. In addition, layer groupings can be specified in order to change the order of activation of units in adjacent memory blocks to match that of LSTM:

numInputs, numOutputs, inputToOutput, biasOutput
memoryBlock, receiveInput, sendToOutput, biased
[toBlock, fromBlock, connectionType]
[firstBlockInLayer, layerSize]

Each line after the first represents any number of unordered consecutive lines of the same format, and the same freedoms of blank lines and whitespace as in manual building apply here. Memory blocks are numbered from 0 to (number of blocks - 1) by the order that their units are activated (see the fourth line's explanation for the exception to this). All connections from units to themselves have weights of 1, and non-self-connections have randomized weights in the range [-0.1, 0.1] (the paper uses [-0.1, 0.1) for its experiments, but Python's pseudorandom floats are documented as inclusive).

In the first line, numInputs is the number of input units (unless there is a bias unit - see the last paragraph of this section), and numOutputs is the number of output units. inputToOutput (and other binary options) is either 0 or 1; if it is 1, all input units are connected to all output units. If biasOutput is 1, all output units are biased.

In the second line, memoryBlock is defined with none or any combination of the following three properties. If receiveInput is 1, all input units are connected to memoryBlock. If sendToOutput is 1, memoryBlock is connected to all output units. If biased is 1, all units in memoryBlock are biased.

In the third line, the connection from fromBlock to toBlock has type connectionType, which is either 0, 1, or 2. Type 0 is a peephole connection when fromBlock is equal to toBlock, and is used in the paper's ungated recurrence architecture. Type 1 is from the gated recurrence architecture. Type 2 is a downstream connection from the two-stage network architecture. See Figs. 2, 4, and 6 in the paper. Note that all-to-all connections must be specified individually.

In the fourth line, firstBlockInLayer and the next (layerSize - 1) memory blocks have a different order of activation than the default: instead of activating the input gate, forget gate, memory cell, and output gate (in that order) of one block before activating the units in the next block, the input gates of blocks in the same layer are activated in block order before activating the layer's forget gates in block order, etc. This is LSTM's order of activation, but LSTM-g's architectural flexibility means that layers need to be specified. Hybrid, partially layered networks are possible.

If any memory blocks or the output units are biased, numInputs is not technically the number of input units. The last "input unit" is a bias unit. It receives its activation from the last entry of input lists (see the step method in Usage - API).

Usage - API

toString(newNetwork[, newline]) returns a string of the format used in manual building, except there are no blank lines, the only non-line-separator whitespace is a single space after each comma, and consecutive lines of the same format are sorted in ascending order, first by j, then by i, then by k (where applicable). If newNetwork is True, the states, traces, and extended traces are not included. Unless the newline string is given, the operating system's default line separator is used.

step(inputs) takes a list of input data with a length equal to numInputs as specified in the string given to the class constructor (see the previous two sections). Input unit activations come directly from this, and then all other units are activated in order for a single time step. A list of the output units' activations is returned.

getError(targets) returns the difference between the most recent output unit activations and the given list of target data. This cross-entropy error function is from Eq. 9 in the paper.

learn(targets[, learningRate]) adjusts the network's weights using the most recent output unit activations and the given list of target data. If the learning rate is not specified, 0.1 is used as in the experiments in the paper.

Call the step method at least once before calling the learn method; input unit activations are not provided by the class constructor (see Usage - Manual Building) and are undefined until they are given input data. Training works by calling the step method one or more times (possible because the information used to calculate weight changes is updated every time step) and then calling the learn method exactly once. There is no reason to call the learn method twice in a row, and the second time would use the wrong weights in its calculations.

Architectures

LSTM-g is a mathematically exact generalization of LSTM with forget gates. This means that despite the complete flexibility of connectivity and gating that is now possible, the methods of activation propagation and weight adjustment are designed for LSTM-like architectures. Self-connected units, their biases, and input and output units have special treatment (see Algorithm). The learning algorithm assumes that no unit gates its own self-connection and that self-connection weights are a constant 1.

Still, the network specification format in Usage - Automatic Building allows only a subset of LSTM-like architectures, flexible enough that all networks used in the experiments in the paper can be built. Connections involving input units, output units, or biases do not have to be one-to-all, all-to-all, etc. Output units can project or gate connections. There can be hidden units outside of memory blocks. Forget gates are not required (although overwhelming evidence to their usefulness has made them standard). Memory blocks in the same layer do not have to be adjacent, and in fact there is a lot of choice for the order that parallel, independent hidden units can be activated. However, connections to input units are unused (see Usage - Manual Building).

Of course, one is free to try less conventional architectures (or modifications to the algorithm); what is important in practice is learning speed, success at generalization from experience, robustness to noisy input and long time lags, and the performance-to-computation ratio. LSTM-g can also train classic recurrent networks, and other concepts from artificial neural networks, such as bagging and boosting, are not dependent on the type of network.

Algorithm

Algorithm.png contains most of the paper's details about how LSTM-g networks work, but neither the image nor the paper mention everything. This section is supplementary and hopefully understandable to those unfamiliar with neural networks; much of this is restating the paper's equations in words.

A network consists of connected units: one or more input units, zero or more hidden units, and one or more output units. Each unit is referred to or indexed by a number, and the units have a fixed order of activation (see the next paragraph) over those numbers; input units are activated first, and output units are activated last. Each unit has a state, an activation, and a nonlinear, everywhere-differentiable function from states to activations called an activation function, with the exception of input units, which only need an activation. A connection is directed; that is, it is an output or outgoing connection for the sending unit and an input or incoming connection for the receiving unit. A connection has a weight, a gain, and may additionally be gated by a unit associated with it called the gating unit or gater. A connection's gain is a constant 1 if it is ungated, and equal to the most recent gater activation otherwise (Eq. 14 in the paper).

There are two operations defined for a network: a forward pass and a backward pass. In a forward pass, the activations of the input units are first set equal to a vector, array, list, etc. of input values that the network must learn to produce correct output values from (see the backward pass's explanation). Then the states and activations (both initially 0) of all other units are calculated in their predefined order. If a unit has a connection to itself, its state is first set equal to the product of the gain on this self-connection and the previous state. Otherwise, its state is set equal to 0. Next, for each of the unit's other incoming connections, the product of the connection gain, connection weight, and the sending unit's activation is added to the state (Eq. 15). Finally, the state is passed through the unit's activation function to get its activation (Eq. 16). The activations of the output units after a forward pass define the network output for that time step.

Additional values are stored from and updated immediately after a forward pass for use in a future backward pass. For each non-input unit, its state, sending units' activations, self-connection gain (if it is self-connected), and other incoming connections' gains (all values used in its state calculation) are cached. Note that these values are used in place of the current values in the calculation of eligibility traces and extended eligibility traces. An eligibility trace (tracked for every non-self-connection and initially 0) is the sum of the product of the gain of the receiving unit's self-connection and the previous trace (if the receiving unit has a self-connection; otherwise this term is 0), and the product of the connection's gain and the sending unit's activation (Eq. 17).

An extended eligibility trace (initially 0) is tracked for every combination of non-self-connection and "gated unit" such that the receiving unit gates one or more of the gated unit's incoming connections and the gated unit is activated after the receiving unit. After the trace for the same connection is calculated, the extended trace is first set equal to the product of the gain on the receiving unit's self-connection and the previous extended trace (if the receiving unit has a self-connection; otherwise this term is 0). Then the product of the derivative of the receiving unit's activation function at the receiving unit's state, the just-updated trace, and the Term is added to the extended trace (Eq. 18). The Term is first set equal to the state of the gated unit if the receiving unit gates the gated unit's self-connection, and 0 otherwise. Then for all non-self-connections that the gated unit receives and that are gated by the receiving unit, the product of that connection's weight and that connection's sending unit's activation is added to the Term.

In a backward pass, a list of target output values is given. Connection weights are adjusted so that the error difference between the target and the network output approaches zero (from below by gradient ascent, technically - see Eq. 9, although this cross-entropy function is not used directly). To determine the adjustments, an error responsibility for each non-input unit is calculated in the reverse of the units' order of activation. Unlike with the traces and extended traces, responsibility calculations use the current states, activations, and gains. First, the output unit responsibilities are set equal to the difference between each unit's corresponding target value and its most recent activation (Eq. 10). The other responsibilities are the sum of two components of error: projection error responsibility and gating error responsibility (Eq. 23).

Projection responsibility is the product of the derivative of the unit's activation function at the unit's state and a sum over all units receiving connections from this unit that are activated after it (Eq. 21). Each term in the sum is the product of the receiving unit's responsibility, the connection gain, and the connection weight. Gating responsibility is the product of the derivative of the unit's activation function at the unit's state and a sum over all units receiving connections gated by this unit that are activated after it (Eq. 22); each of these can be called a "gated unit" as when extended traces were described. Each term in the sum is the product of the gated unit's responsibility and the Term. The Term is as described for extended traces, except the current states and activations are used and the sending unit is instead the unit for which the gating responsibility is being calculated. Ignore the hat over the s denoting the previous state.

Lastly, for all connections such that the receiving unit is an output unit, the weight change is equal to the product of the learning rate for the backward pass, the output unit's responsibility, and the trace for the connection (Eq. 13). For all other non-self-connections, the change is the product of this pass's learning rate and the sum of two terms (Eq. 24). The first term is the product of the receiving unit's projection responsibility and the connection's trace. The second term is a sum over all units such that there is an extended trace for the combination of the connection and the "gated unit". Each term in the sum is the product of the gated unit's responsibility and the associated extended trace.

Note that while the paper says that target values and output unit activations must always be within [0, 1], the allowed range is actually (0, 1) for output unit activations. However, floating point numbers may round to 0 or 1 in practice. Input values and the activation functions of non-output units may have other ranges.

Biases, specifically bias connections from bias units to other units, are treated like connections from input units with one exception. Like input units, bias states are unneeded. Like other biases, memory cell biases are ungated. The exception is that bias activations are not included in the state calculations of self-connected units. Instead, the unit's activation function is passed the sum of the current bias value and the unit's state. Because the bias is applied later, the connection's trace is specially defined as the bias unit's most recent activation. Note that this library's code assumes that an ungated connection from an input unit to a self-connected unit is a memory cell bias.

Monner has found that alternatives to the basic weight updating function, such as momentum, are not useful; LSTM seems to be pretty optimal on its own. Momentum (and batch training/offine learning for that matter) may not even make sense in the context of LSTM. However, using different learning rates for different backward passes as in learning rate decay or alpha stepping does make sense.

Missing Features

The only activation function used in the code is the classic logistic sigmoid, with a range of (0, 1). Units do not have to share the same activation function, but the activation functions of output units must still have a range of (0, 1). Furthermore, LSTM memory cells actually have two activation functions: the input and output squashing functions. The paper takes all input squashing functions to be the identity and simply refers to output squashing functions as activation functions, but they can be manually added to any network in the following way. Insert a unit (with the desired activation function) between the memory cell and what used to be its direct inputs (excluding the self-connection, which remains unchanged). The memory cell's input gate gates the single connection between the input squashing unit and the memory cell, instead of gating the memory cell's former inputs.

One could instead wrap the second term of the state formula (the summation in Eq. 15) with the desired function. While not equivalent to LSTM, it might work (paraphrased from Monner).

LSTM networks can have multiple memory cells per memory block. Nothing in the LSTM-g formalism needs to be changed for architectures like this to work.

Updating traces and extended traces is optional and unnecessary if the network's weights will never be adjusted again. Other optimizations are possible (see Library).

(paraphrased from Monner) XLBP's PiLayer allows connections to be gated by multiple units. The normal case is for PiLayer to have two inputs: the source of the connection and its gater. PiLayer avoids this distinction, since the source and gater activations are mathematically interchangeable. A PiLayer with three inputs (for example) is like having a source and two gates. You can look at how it does the math for calculating the error responsibilities and work backward from there to get the general equation. In the original case you have a connection from unit S gated by G1, and your error responsibility for each unit changes as follows (note the primes denoting derivatives):

error(S) *= f'(S) * f(G1)
error(G1) *= f(S) * f'(G1)

Expanding to the two-gate case:

error(S) *= f'(S) * f(G1) * f(G2)
error(G1) *= f(S) * f'(G1) * f(G2)
error(G1) *= f(S) * f(G1) * f'(G2)

So every error responsibility for each unit gets multiplied by the product of all the other units, except itself, where it instead gets multiplied by the derivative.

Other Omissions from the Paper

All non-input units in the experiments in the paper were biased with a constant value of 1, and these bias connections were not included in the reported numbers of weights (nor were self-connections, for that matter).

The cross-entropy error function in the paper uses base-2 logarithms. However, using a different base would only scale the function by a linear factor, so it does not really matter. This error measure was added to the library as a demonstration, but other functions work as well.

LSTM with BPTT does use the full gradient (see http://www.cs.toronto.edu/~graves/nn_2005.pdf, for example), and this seems to be the norm in more recent LSTM research. Thanks to Alex Graves for pointing this out.
