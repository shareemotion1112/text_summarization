A Synaptic Neural Network (SynaNN) consists of synapses and neurons.

Inspired by the synapse research of neuroscience, we built a synapse model with a nonlinear synapse function of excitatory and inhibitory channel probabilities.

Introduced the concept of surprisal space and constructed a commutative diagram, we proved that the inhibitory probability function -log(1-exp(-x)) in surprisal space is the topologically conjugate function of the inhibitory complementary probability 1-x in probability space.

Furthermore, we found that the derivative of the synapse over the parameter in the surprisal space is equal to the negative Bose-Einstein distribution.

In addition, we constructed a fully connected synapse graph (tensor) as a synapse block of a synaptic neural network.

Moreover, we proved the gradient formula of a cross-entropy loss function over parameters, so synapse learning can work with the gradient descent and backpropagation algorithms.

In the proof-of-concept experiment, we performed an MNIST training and testing on the MLP model with synapse network as hidden layers.

Synapses play an important role in biological neural networks BID11 ).

They are joint points of neurons' connection with the capability of learning and memory in neural networks.

Based on the analysis of excitatory and inhibitory channels of synapses BID11 ), we proposed a probability model BID6 for probability introduction) of the synapse together with a non-linear function of excitatory and inhibitory probabilities BID17 (synapse function)).

Inspired by the concept of surprisal from (Jones (1979)(self-information), BID15 , BID2 (surprisal analysis), BID16 (surprisal theory in language)) or negative logarithmic space BID21 ), we proposed the concept of surprisal space and represented the synapse function as the addition of the excitatory function and inhibitory function in the surprisal space.

By applying a commutative diagram, we figured out the fine structure of inhibitory function and proved that it was the topologically conjugate function of an inhibitory function.

Moreover, we discovered (rediscovered) that the derivative of the inhibitory function over parameter was equal to the negative Bose-Einstein distribution BID22 ).

Furthermore, we constructed a fully connected synapse graph and figured out its synapse tensor expression.

From synapse tensor and a cross-entropy loss function, we found and proved its gradient formula that was the basis for gradient descent learning and using backpropagation algorithm.

In surprisal space, the parameter (weight) updating for learning was the addition of the value of the negative Bose-Einstein distribution.

Finally, we designed the program to implement a Multiple Layer Perceptrons (MLP) BID20 ) for MNIST BID14 ) and tested it to achieve the near equal accuracy of standard MLP in the same setting.

Hodgkin and Huxley presented a physiological neuron model that described the electronic potential of the membrane between a neuron and a synapse with a differential equation BID9 ).

Later, neuron scientists have found that a synapse might have a complicated channel structure with rich chemical and electronic properties BID19 (biological synapse), BID4 (computing synaptic conductances), BID1 (synaptic plasticity)).

Other synapse models based on differential equations had been proposed and been simulated by analogy circuits like Spiking Neural Network (SNN) BID13 (differential equations), Lin et al. (2018) (Intel's SNN Loihi) ).

In these approaches, synapses acted as linear amplifiers with adjustable coefficients.

An example was the analog circuit implementation of Hopfield neural network BID10 (analog neural circuits)).In this paper, we proposed a simple synapse model represented by the joint opening probability of excitatory and inhibitory channels in a synapse.

It was described as a non-linear computable synapse function.

This neuroscience-inspired model was motivated on our unpublished research to solve optimization problems by neural networks.

To do learning by gradient descent and backpropagation algorithm BID8 (book on deep learning)), because of the differentiable of the synapse function in the synaptic neural network, we could compute Jacobian matrix explicitly and compute the gradient of the cross-entropy loss function over parameters.

Therefore, we provided a detailed proof of the formula of gradients in Appendix AIn the process of analyzing Jacobian matrix, we found that the derivative of the inhibitory function log(1 − e −x ) was equal to the 1/(e x − 1) which was the formula of Bose-Einstein distribution BID5 (quantum ideal gas)).

In statistical physics and thermodynamics, Bose-Einstein distribution had been concluded from the geometric series of the Bose statistics.

A dual space analysis was an efficient scientific method.

After successful expressing fully-connected synapse network in a logarithmic matrix, we started to consider log function and log space.

The concept of surprisal (where was the first definition of surprisal?), which was the measurement of surprise from Information Theory BID25 ), gave us hints.

Original surprisal was defined on the random variable, however, it was convenient to consider the probability itself as a variable.

So we introduced the surprisal space with a mapping function -log(p).

The motivation was to transform any points from probability space to surprisal space and in reverse.

In surprisal space, a synapse function was the addition of an excitatory identity function and an inhibitory function.

Although we had figured out the inhibitory function being −log(1 − e −x ), we wanted to know its structure and what class it belonged to.

This was a procedure that we rediscovered the way to construct a commutative diagram for synapse inhibitory function Diagram (2.2.3).

In 1903, Mathematician Bertrand Russell presented the first commutative diagram in his book BID24 ) before the category theory.

You can find a good introduction of applied category theory by BID3 ).

In this paper, we did not require to know category theory.

The basic idea was to given two spaces and two points in source space which have corresponding points in target space by a continuous and inverse mapping function from source space to target space, plus, a function that maps start point to the endpoint in the same source space.

Our question is to find the function that maps the corresponding start point to the corresponding endpoint in the target space (refer to diagram 2.2.3).

There are two paths from source start point to target endpoint: one is from top-left, go right and down to bottom-right; another is from top-left, go down and right to bottom-right.

The solution is to solve the equation that has the same target endpoint.

We found that the synapse inhibitory function −log(1 − e −x ) was also a topologically conjugate function.

Therefore, the synaptic neural network has the same dynamical behavior in both probability space and surprisal space.

To convince that the synaptic neural network can work for learning and using the backpropagation algorithm, we proved the gradients of loss function by applying basic calculus.

In surprisal space, the negative Bose-Einstein distribution was applied to the updating of parameters in the learning of synaptic neural network.

Finally, we implemented a MNIST experiment of MLP to be the proof-of-concept.

1) present a neuroscience-inspired synapse model and a synapse function based on the opening probability of channels.

2) defined surprisal space to link information theory to the synaptic neural network.

3) figure out function G(x) = −log(1 − e −x ) as the inhibitory part of a synapse.

4) find the derivative of G(x) to be the formula of negative Bose-Einstein distribution.

5) discover G(x) to be a topologically conjugate function of the complementary probability.

6) represent fully-connected synapse as a synapse tensor.

7) express synapse learning of gradient descent as a negative Bose-Einstein distribution.

A Synaptic Neural Network (SynaNN) contains non-linear synapse networks that connect to neurons.

A synapse consists of an input from the excitatory-channel, an input from the inhibitory-channel, and an output channel which sends a value to other synapses or neurons.

Synapses may form a graph to receive inputs from neurons and send outputs to other neurons.

In advance, many synapse graphs can connect to neurons to construct a neuron graph.

In traditional neural network, its synapse graph is simply the wight matrix or tensor.

Changes in neurons and synaptic membranes (i.e. potential gate control channel and chemical gate control channel show selectivity and threshold) explain the interactions between neurons and synapses BID26 ).

The process of the chemical tokens (neurotransmitters) affecting the control channel of the chemical gate is accomplished by a random process of mixing tokens of the small bulbs on the membrane.

Because of the randomness, a probabilistic model does make sense for the computational model of the biological synapse BID11 ).In a synapse, the Na+ channel illustrates the effect of an excitatory-channel.

The Na+ channels allow the Na+ ions flow in the membrane and make the conductivity increase, then produce excitatory post-synapse potential.

The K+ channels illustrate the effects of inhibitory channels.

The K+ channel that lets the K+ ions flow out of the membrane shows the inhibition.

This makes the control channel of potential gate closing and generates inhibitory post-potential of the synapse.

Other kinds of channels (i.e. Ca channel) have more complicated effects.

Biological experiments show that there are only two types of channels in a synapse while a neuron may have more types of channels on the membrane.

Experiments illustrate that while a neuron is firing, it generates a series of spiking pulse where the spiking rate (frequency) reflects the strength of stimulation.

From neuroscience, there are many types of chemical channels in the membrane of a synapse.

They have the following properties: 1) the opening properties of ion channels reflect the activity of synapses.

2) the excitatory and inhibitory channels are two key types of ion channels.

3) the random properties of channels release the statistical behavior of synapses.

From the basic properties of synapses, we proposed the synapse model below:1) The open probability x of the excitatory channel (α-channel) is equal to the number of open excitatory channels divided by the total number of excitatory channels of a synapse.2) The open probability y of the inhibitory channel (β-channel) is equal to the number of open inhibitory channels divided by the total number of inhibitory channels of a synapse.3) The joint probability of a synapse that affects the activation of the connected output neuron is the product of the probability of excitatory channel and the complementary probability of the inhibitory channel.

4) There are two parameters to control excitatory channel and inhibitory channel respectively.

Given two random variables (X, Y ), their probabilities (x, y), and two parameters (α, β), the joint probability distribution function S(x, y; α, β) for X, Y (the joint probability of a synapse that activates the connected neuron) is defined as S(x, y; α, β) = αx(1 − βy)where x ∈ (0, 1) is the open probability of all excitatory channels and α > 0 is the parameter of the excitatory channels; y ∈ (0, 1) is the open probability of all inhibitory channels and β ∈ (0, 1) is the parameter of the inhibitory channels.

The symbol semicolon ";" separates the variables and parameters in the definition of function S.

Surprisal (self-information) is a measure of the surprise in the unit of bit, nat, or hartley when a random variable is sampled.

Surprisal is a fundamental concept of information theory and other basic concepts such as entropy can be represented as the function of surprisal.

The concept of surprisal has been successfully used in molecular chemistry and natural language research.

Given a random variable X with value x, the probability of occurrence of x is p(x).

The standard definitions of Surprisal I p (x) is the measure of the surprise in the unit of a bit (base 2), a nat (base e), or a hartley (base 10) when the random variable X is sampled at x. Surprisal is the negative logarithmic probability of x such that I p (x) = −log(p(x)).

Ignored random variable X, we can consider p(x) as a variable in Probability Range Space or simply called Probability Space in the context of this paper which is the open interval (0,1) of real numbers.

Surprisal Function is defined as I : (0, 1) → (0, ∞) and I(x) = −log(x) where x ∈ (0, 1) is an open interval in R + .

Its inverse function is I −1 (u) = e −u where u ∈ R + .

Since surprisal function I(x) is bijective, exists inverse and is continuous, I(x) is a homeomorphism.

Surprisal Space S is the mapping space of the Probability Space P with the negative logarithmic function which is a bijective mapping from the open interval (0, 1) of real numbers to the real open interval (0, ∞) = R + .

DISPLAYFORM0 The probability space P and the surprisal space S are topological spaces of real open interval (0,1) and positive real numbers R + that inherit the topology of real line respectively.

Given variables u, v ∈ S and parameters θ, γ ∈ S which are equal to variables −log(x), −log(y) and parameters −log(α), −log(β) respectively.

The Surprisal Synapse LS(u, v; θ, γ) ∈ S is defined as, LS(u, v; θ, γ) = −log(S(x, y; α, β))Expanding the right side, there is LS(u, v; θ, γ) = (−log(αx)) + (−log(1 − βy)).

The first part is an identity mapping plus a parameter.

To understand the second part more, we need to figure out its structure and class.

Theorem 1 (Topologically conjugate function).

Given y = F(x) where F(x) = 1 − x; x, y ∈ P, (u, v) = I(x, y) where u, v ∈ S, and the homeomorphism I(x) = −log(x) from P to S, then function DISPLAYFORM0 Proof.

Building a commutative diagram with the homeomorphism I(x) below, DISPLAYFORM1 The proof is to figure out the equivalent of two paths from x to v. One path is from top x, go right to y and go down to bottom so v = I(F(x)).

Another path is from top x, go down to u and go right to bottom so v = G • I, thus, I(F(x)) = G(I(x)).

Let • be the composition of functions, the previous equation is I • F = G • I.

Applying I −1 on both right sides and compute G on given functions, we proved Eq.(4).Given two topological spaces P and S, continuous function F : P → P and G : S → S as well as homeomorphism I : P → S, if I • F = G • I, then G is called the topologically conjugated function of the function F in the standard definition.

From Theorem 1, specially G(u) = −log(1 − e −u ) is the topologically conjugate function of the complementary probability function 1 − x. Features:i) The iterated function F and its topologically conjugate function G have the same dynamics.

ii) They have the same mapped fixed point where F : x = 1/2 and G : u = −log(1/2).

iii) I(x) = −log(x) is a infinite differentiable and continuous function in real open interval (0,1).Let parametric function be D(u; θ) = u + θ, the surprisal synapse is DISPLAYFORM2 From Eq. FORMULA5 , the universal function of a surprisal synapse is the addition of the excitatory function and the topologically conjugate inhibitory function in surprisal space.

By constructed a commutative diagram, we figured out the elegant structure and topological conjugacy of the function −log(1−e −u ), which is a new example of the commutative diagram and the topological conjugate function from synaptic neural network.

A bridge has been built to connect the synaptic neural network to the category theory, the topology, and the dynamical system.

It is interesting to find the connection between the surprisal synapse and the topologically conjugate function.

Furthermore, we are going to figure out the connection between the surprisal synapse and the Bose-Einstein distribution.

The Bose-Einstein distribution (BED) is represented as the formula DISPLAYFORM0 where f(E) is the probability that a particle has the energy E in temperature T. k is Boltzmann constant, A is the coefficient (Nave FORMULA0 ).Theorem 2.

The BED function is defined as BED(v; γ) = 1 e γ+v −1 where variable v ∈ S, parameter γ ∈ S, and v + γ ≥ ln(2), so that 0 ≤ BED(v; γ) ≤ 1, then there is DISPLAYFORM1 Proof.

Proved by computing the derivative of the function on left side.

Recall that D(v; γ) = v +γ, the derivative of the topologically conjugate function G over parameter γ is equal to the negative Bose-Einstein distribution.

The gradient of the surprisal synapse LS(u, v; θ, γ) is DISPLAYFORM2 This is a connection between surprisal synapse and statistical physics.

In physics, BED(v; γ) can be thought of as the probability that boson particles remain in v energy level with an initial value γ.

Generally, a biological neuron consists of a soma, an axon, and dendrites.

Synapses are distributed on dendritic trees and the axon connects to other neurons in the longer distance.

A synapse graph is the set of synapses on dendritic trees of a neuron.

A synapse can connect its output to an input of a neuron or to an input of another synapse.

A synapse has two inputs: one is excitatory input and another is inhibitory input.

Typically neurons receive signals via the synapses on dendrites and send out spiking plus to an axon BID11 ).Assume that the total number of input of the synapse graph equals the total number of outputs, the fully-connected synapse graph is defined as DISPLAYFORM0 where x = (x 1 , · · · , x n ), x i ∈ (0, 1) and y = (y 1 , · · · , y n ) are row vectors of probability distribution; β β β i = (β i1 , · · · , β in ), 0 < β ij < 1 are row vectors of parameters; β β β = matrix{β ij } is the matrix of all parameters.

α α α = 1 is assigned to Eq.1 to simplify the computing.

An output y i of the fully-connected synapse graph is constructed by linking the output of a synapse to the excitatory input of another synapse in a chain while the inhibitory input of each synapse is the output of neuron x i in series.

In the case of the diagonal value β ii is zero, there is no self-correlated factor in the ith item.

This fully-connected synapse graph represents that only neuron itself acts as excitation all its connected synapses act as inhibition.

This follows the observation of neuroscience that most synapses act as inhibition.

Theorem 3 (Synapse tensor formula).

The following synapse tensor formula Eq.9 is equivalent to fully-connected synapse graph defined in the Eq.8 DISPLAYFORM1 or I(y) = I(x) + 1 |x| * I(1 |β| − diag(x) * β β β T ) where x, y, and β β β are distribution vectors and parameter matrix.

β β β T is the transpose of the matrix β β β.

1 |x| is the row vector of all real number ones and 1 |β| is the matrix of all real number ones that have the same size and dimension of x and β β β respectively.

Moreover, the * is the matrix multiplication, diag(x) is the diagonal matrix of the row vector x, and the log is the logarithm of the tensor (matrix).Proof.

Applying the log on both sides of the definition Eq. FORMULA9 and completes the matrix multiplications in the fully-connected synapse graph, we proved the formula Eq.(9).

Furthermore, by the definition of I(x), we have the expression of surprisal synapse.

To prove that synapse learning of synaptic neural network is compatible with the standard backpropagation algorithm, we are going to apply cross-entropy as the loss function and use gradient descent to minimize that loss function.

The basic idea of deep learning is to apply gradient descent optimization algorithm to update the parameters of the deep neural network and achieve a global minimization of the loss function BID8 ).

DISPLAYFORM0 )/∂β ij whereô ô o is the target vector and o o o is the output vector and the fully-connected synapse graph outputs through a softmax activation function that is o j = sof tmax(y j ).Proof.

The proof is given in the Appendix A.

Considering the surprisal space, let (u k , v k , γ ki ) = −log(x k , y k , β ki ), the fully-connected synapse graph is denoted as v k = u k + i (−log(1 − e −(γ ki +ui) )) .

Compute the gradient over parameters DISPLAYFORM0 because only when k = p and i = q, two δ are 1, so DISPLAYFORM1 1−e −(γpq +uq ) .

Replacing the indexes and reformulating, we have DISPLAYFORM2 The right side of Eq. FORMULA0 is the negative Bose-Einstein Distribution in the surprisal space.

To compute the loss function in surprisal space, we convert the target vectorô ô o and output vector o o o to surprisal space as (ô ô o, o o o), so the new loss function is L(t t t, t t t) = kt k * t k .

The log function has been removed in L(t t t, t t t) because log is implied in the surprisal space.

Without using an activation function, there is t k = v k .

By Eq. FORMULA0 , DISPLAYFORM3 We can apply error back-propagation to implement gradient descent for synapse learning.

DISPLAYFORM4 where η is the learning rate.

The equation Eq. FORMULA0 illustrates that the learning of synaptic neural network follows the Bose-Einstein statistics in the surprisal space.

This paper "Memory as an equilibrium Bose gas" by BID7 , Pascual-Leone (1970)) showed that memory maybe possible to be represented as the equilibrium of Bose gas.

We are going to illustrate a Synaptic Neural Network implementation SynaMLP with the connection of Multiple Layer Perceptrons (MLP) BID20 ).

SynaMLP has an input layer for down-sampling and an output layer for classification.

The hidden layer is a block of fully-connected synapse tensor.

The inputs of the block are neurons from the input layer and the outputs of the block are neurons to the output layer.

The block is the implementation of synapse tensor in Eq.(9).

The activation functions are connected synapse tensor to the output layer.

Moreover, the input of the block is a probability distribution.

The block can be thought of the replacement of the weight layer in a standard neural network.

To proof-of-concept, we implemented the SynaMLP with MNIST.

Hand-written digital MNIST data sets are used for training and testing in machine learning.

It is split into three parts: 60,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation) BID14 The MNIST SynaMLP training and testing is implemented by Python, Keras and Tensorflow BID0 ) from the revision of the example of mnist_mlp.py in Keras distribution.

The synapse tensor is designed to be a class to replace Dense in Keras.

The layer sequence is as below, In the comparison experiment, SynaNN MLP and traditional MLP generated the similar test accuracy of around 98%.

We applied a softmax activation function in front of the input of synapse to avoid the error of NAN (computing value out of the domain).

In fact, synaptic neural network handles a probability distribution (vector from neurons).

In this paper, we presented and analyzed a Synaptic Neural Network (SynaNN).

We found the fine structure of synapse and the construction of synapse network as well as the BE distribution in the gradient descent learning.

In surprisal space, the input of a neuron is the addition of the identity function and the sum of topologically conjugate functions of inhibitory synapses which is the sum of bits of information.

The formula of surprisal synapse function is defined as LS(u, v; θ, γ) = (θ + u) + (I • F • I −1 )(γ + v))The non-linear synaptic neural network may be implemented by physical or chemical components.

Instead of using a simple linear synapse function, more synapse functions maybe found in the researches and applications of neural network.

<|TLDR|>

@highlight

A synaptic neural network with synapse graph and learning that has the feature of topological conjugation and Bose-Einstein distribution in surprisal space.  

@highlight

The authors propose a hybrid neural nework composed of a synapse graph that can be embedded into a standard neural network

@highlight

Presents a biologically-inspired neural network model based on the excitatory and inhibitory ion channels in the membranes of real cells