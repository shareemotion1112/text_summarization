We describe Kernel RNN Learning (KeRNL), a reduced-rank, temporal eligibility trace-based approximation to backpropagation through time (BPTT) for training recurrent neural networks (RNNs) that gives competitive performance to BPTT on long time-dependence tasks.

The approximation replaces a rank-4 gradient learning tensor, which describes how past hidden unit activations affect the current state, by a simple reduced-rank product of a sensitivity weight and a temporal eligibility trace.

In this structured approximation motivated by node perturbation, the sensitivity weights and eligibility kernel time scales are themselves learned by applying perturbations.

The rule represents another step toward biologically plausible or neurally inspired ML, with lower complexity in terms of relaxed architectural requirements (no symmetric return weights), a smaller memory demand (no unfolding and storage of states over time), and a shorter feedback time.

Animals and humans excel at learning tasks that involve long-term temporal dependencies.

A key challenge of learning such tasks is the problem of spatiotemporal credit assignment: the learner must find which of many past neural states is causally connected to the currently observed error, then allocate credit across neurons in the brain.

When the time-dependencies between network states and errors are long, learning becomes difficult.

In machine learning, the current standard for training recurrent architectures is Backpropagation Through Time (BPTT, Rumelhart et al. (1985) , BID18 ).

BPTT assigns temporal credit or blame by unfolding a recurrent neural network in time up to a horizon length T , processing the input in a forward pass, and then backpropagating the error back in time in a backward pass (see FIG0 ).From a biological perspective, BPTT -like backpropagation in feedforward neural networks -is implausible for many reasons.

For each weight update, BPTT requires using the transpose of the recurrent weights to transmit errors backwards in time and assign credit for how past activity affected present performance.

Running the network with transposed weights requires that the network either has two-way synapses, or uses a symmetric copy of the feedforward weights to backpropagate error.

In either case, the network must alternatingly gate its dynamical process to run forward or backward, and switch from nonlinear to linear dynamics, depending on whether activity or errors are being sent through the network.

From both biological and engineering perspectives, there is a heavy memory demand: the complete network states, going T timesteps back in time, must be stored.

The time-complexity of computation of the gradient in BPTT scales like T , making each iteration slow when training tasks with long time scale dependencies.

Although T should match the length of the task or the maximum temporal lag between network states and errors for unbiased gradient learning, in practice T is often truncated to mitigate these computational costs, introducing a bias.

The present work is another step in the direction of providing heuristics and relaxed approximations to backpropagation-based gradient learning for recurrent networks.

KeRNL confronts the problems of efficiency and biological plausibility.

It replaces the lengthy, linearized backward-flowing backpropagation phase with a product of a forward-flowing temporal eligibility trace and a spatial sensitivity weight.

Instead of storing all details of past states in memory, synapses integrate their past activity during the forward pass (see FIG0 ).

The network does not have to recompute the entire gradient at each update, as the time-scales of the eligibility trace and the sensitivity weight are learned over time.

In recent years, much work has been devoted to implementing backpropagation algorithms in a more biologically plausible way, partly in the hope that more plausible implementations might also be simpler.

The symmetry requirement between the forwards and backwards weights can be alleviated by using random return weights BID9 and BID10 ), however, learning still requires a separate backward pass through a network with linearized dynamics.

Neurons may be able to extract error information in the time derivative of their firing rates using an STDP-like learning rule BID0 ), with error backpropagation computed as a relaxation to equilibrium BID14 ), at least for learning fixed points.

Other work has focused on replacing batch learning with online learning.

Typically, BPTT is implemented in a setting where data is prepared into batches of fixed sequence length T and used to perform learning in a T -step unrolled graph; however, online learning, with a constant stream of data error signals, is a more natural description of how the world supplies a learning system with data.

BPTT without truncation struggles with online learning, as it must repeatedly backpropagate the error all the way through a continuously expanding graph.

Since computation of the unbiased gradient scales with the length of the graph, gradient computation increases linearly with time.

For a task with T timesteps, the total computation of the gradients scales like T 2 .Real Time Recurrent Learning (RTRL, Williams & Zipser (1989) ) and Unbiased Online Gradient Optimization (UORO, BID15 , BID11 ) deal with this issue by keeping track of how the synaptic weights affect the hidden state in a feedforward way.

Decoupled Neural Interfaces (DNI BID6 ) estimates the truncated part of the gradient by continually predicting the future loss with respect to the hidden state.

KeRNL offers this same advantage, in addition to other benefits.

RTRL requires that the network keep track of an unwieldy rank-3 tensor, which could not be stored by any known biological entities.

UORO factorizes this into rank-2 objects but still requires non-local computations like vector norm operations.

Finally, DNI requires an entire separate network to keep track of the synthetic gradient.

KeRNL is distinguished by its simplicity, requiring only rank-2 tensors.

All computations are local, and synapses need to integrate over only a few relevant quantities.

Consider a single-layer RNN in discrete time (indexed by t) with readout, input, and hidden layer activations given by y t , x t , and h t , respectively (boldface represents vectors, with vector entries denoting the activity of individual units).

The dynamics of the recurrently connected hidden units are given by: DISPLAYFORM0 where W rec , W in are the recurrent and input weights, b are the hidden biases, σ is a general pointwise non-linearity, and g t represents the summed inputs (pre-nonlinearity) to the neurons at time t.

The readout is given by DISPLAYFORM1 whereŷ T is the target output, in the case where error feedback is received at the end of an episode of length T , and C = T t=0 C t when errors DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 is the gradient of the cost with respect to the current hidden state, β ij is a set of learned sensitivity weights, and DISPLAYFORM5 is a local eligibility trace (Williams (1992)) consisting of a temporally filtered version of the product of presynaptic activation and a postsynaptic activity factor.

The temporal filter or kernel, K, in the eligibility has learnable time-scales; in this manuscript we use the simplest version of a lowpass temporal filter, a decaying exponential with a single time-constant γ j per neuron: K(τ, γ j ) = exp(−γ j τ ), though one can imagine many other function choices with multiple timescales.

The role of the eligibility is to specify how strongly a synapse W jk should be held responsible for any errors in neuron j at the present time, on the basis on how far in the past the presynaptic neuron k was active.

Here s DISPLAYFORM6 } stands in for the activation of the neuron presynaptic to the synapse being updated.1 .

Since the eligibility trace can be computed during the forward pass, KeRNL does not require backpropagating the error through time.

Furthermore, KeRNL only uses at most rank-2 tensors, so neurons and synapses could plausibly do all of the required computation.

The contrast between BPTT and KeRNL is depicted in FIG0 ,b. KeRNL emerges from the following Ansatz: DISPLAYFORM7 We call ∂h DISPLAYFORM8 , a key term in the computation of the gradient, the sensitivity tensor in an extension of the usage in BID1 ).

This sensitivity describes how the activity of neuron j at a previous time t − τ affects the activity of neuron i at the current time t. While the true sensitivity is a 4-index tensor summarizing many interactions based on the many paths through which activity propagates forward in a recurrent network, we approximate it with a product of a (learnable) rank-2 sensitivity weight matrix β and a temporal kernel K with (learnable) inversetime coefficients γ.

The sensitivity weights β ij describe how strongly neuron j affects neuron i on average, while the temporal kernel describes how far into the future the activity of a neuron affects the other neurons for learning.

We describe how to learn these parameters (β, γ) in the next section.

We arrive at KeRNL by using our Ansatz for the sensitivity (4) in the computation of a gradientbased weight update, instead of using the true sensitivity.

First, we write down the full gradient rule for a recurrent network.

If the parameters W ij are treated as functions that can vary over time during a trial, then the derivative can be written as a functional derivative: DISPLAYFORM9 This is simply mathematical notation for the "unfolding-in-time" trick, in which the network and weights are assumed to be replicated for each time-step of the dynamics of a recurrent network, and separate gradients are computed for each time-replica of the weights; the actual weight updates are simply the average of the separate weight variations for each time-replica.

We next apply the sensitivity lemma BID1 to express gradients with respect to weights as gradients with respect to input activations, times the presynaptic activity: DISPLAYFORM10 DISPLAYFORM11 By replacing the sensitivity δh i (t)/δh j (t − τ ) with our Ansatz (4), we arrive at our learning rule, KeRNL(2).The time-dependent part of the computation-a leaky integral of the product of the presynaptic activity multiplied by the instantaneous change in the postsynaptic activity-can be computed during the forwards pass, without any backpropagation of activity or error signals.

For our Ansatz to align as well as possible with the gradient, we allow the sensitivity weights β and inverse-timescales γ to be learned.

We learn these parameters by tracking the effect of small i.i.d.

hidden perturbations ξ during the forward pass.

In order to do so our hidden neurons must store two values, the true hidden state h, and a perturbed hidden stateh, which is generated by applying noise to the neurons during the forward pass: DISPLAYFORM0 The effect of previous noise on the current hidden state can be computed using the sensitivitỹ DISPLAYFORM1 We train γ, β to predict the network's response to these noisy perturbations.

We take gradients with respect to the objective function.

DISPLAYFORM2 which we have generated by substituting our Ansatz (4) into (9).

Taking gradients with respect to this objective function gives us the following update rule for the sensitivity weights and inverse-timescales.

DISPLAYFORM3 represents the error in reconstructing the effect of the perturbation via the sensitivity weights and DISPLAYFORM4 are integrals that neuron h j performs over the applied perturbation ξ.

In our implementation, we update these parameters immediately before we compute the gradient using (2).

The full update rule is described in the pseudocode table.2 If we don't care about the size of the gradients and only the direction, we can use the cost function DISPLAYFORM5 where DISPLAYFORM6 .

This cost function trains the parameters to predict the correct direction of the perturbed hidden state minus the hidden state and works for algorithms where the gradient is divided by a running average of its magnitude (RMSProp, Adam).

DISPLAYFORM7

We test KeRNL on several benchmark tasks that require memory and computation over time, showing that it is competitive with BPTT across these tasks.

We implemented batch learning with KeRNL and BPTT on two tasks: the adding problem BID4 ; BID5 ) and pixel-by-pixel MNIST BID8 ).

We implemented an online version of KeRNL with an LSTM network on the A n , B n task BID3 ) to compare with results from the UORO algorithm BID15 ).

The tuned hyperparameters for BPTT and KeRNL were the learning rate, η, and the gradient clipping parameter, gc BID12 ).

For KeRNL, we additionally permitted a shared learning rate parameter for the sensitivity weights and kernels, η m .

In practice, the same hyperparameter settings η, gc tended to work well for both BPTT and KeRNL.

The additional hyperparameter for KeRNL, η m , did not need to be find tuned, and often worked well across a broad range (across several orders of magnitude, so long as it not too small but smaller than η).We implemented both the RMSprop BID16 ) and Adam (Kingma & Ba FORMULA0 ) optimizers and reported the best result.

In the adding problem, the network receives two input streams, one a sequence of random numbers in [0, 1], and the second a mask vector of zeros, with two entries set randomly to one in each trial.

The network's task is to sum the input from the first stream whenever there is a non-zero entry in the second.

This task requires remembering sparse pieces of information over long time scales and ignoring long sequences of noise, which is difficult for RNNs when the sequences are long.

We tested the performance of two networks on a variety of sequence lengths, up to 400, using both BPTT and KeRNL, Table 2 .

The networks were an IRNN, which is an RNN with a ReLU non-linearity where the recurrent weight matrix is initialized to identity, and a RNN with tanh nonlinearity.

The implementation details are described in Appendix A.Untruncated BPTT applied to an IRNN performed very well on this task, but less so on the RNN with tanh nonlinearity.

KeRNL was somewhat unstable on the IRNN, but it outperformed BPTT with the tanh nonlinearity FIG3 .We believe that KeRNL outperforms BPTT on the tanh nonlinearity because our Ansatz allows the sensitivity tanh nonlinearity.

By applying gradients generated by our Ansatz (instead of the true gradients)

we push our network toward a solution with longer time scales via a feedback alignment-like mechanism BID9 BID10 ), as schematized in FIG0 .To investigate the importance of learning the kernel timescales, we implemented KeRNL without training the sensitivity weights (β) or the inverse timescales (γ).

When these parameters are not learned, KeRNL is still able to perform the task for the shorter 200-length sequence (Table 2) implying that a feedback-alignment-like mechanism BID9 BID10 ) may be enabling learning even when the error signals are not delivered along the instantaneous gradients.

For longer sequences, however, learning the sensitivity and timescale parameters is important.

Surprisingly, learning the inverse timescales is even more important than learning the sensitivity weights.

We hypothesize that as long as the timescales over which error is correlated with outcome are appropriate, sensitivity weights are relatively less important because of feedback-alignment-like mechanisms.

We show an example of how the timescales may change in FIG5 .

Our second task is pixel-by-pixel MNIST BID8 .

Here the RNN is given a stream of pixels left-to-right, top-to-bottom for a given handwritten digit from the MNIST data set.

At the end of the sequence, the network is tasked with identifying the digit it was shown.

This problem is difficult, as the RNN must remember an long sequence of 784 singly-presented pixels.

We tuned over the same hyperparameters as in the adding problem, looking at performance after 100, 000 minibatches.

Neither KeRNL nor BPTT worked well with a tanh nonlinearity, but both performed relatively well on an IRNN, FIG5 .

KeRNL preferred a slightly lower learning rate η than BPTT.

While the KeRNL algorithm is able to learn almost as quickly on pixel-by-pixel MNIST, it does not reach as high an asymptotic performance.

Still, it performs reasonably well relative to BPTT on the task.

Table 2 : Learning of KeRNL parameters.

Left: Histogram of inverse time coefficients before training (blue) and after 7 × 10 4 minibatches (orange) on the adding problem FORMULA3 : the network learns the relative importance of certain time-scales.

Right: Examining the relative importance of learnable parameters in KeRNL: Performance on BPTT and various versions of KeRNL using a tanh RNN after 7 × 10 4 minibatches: fixing the sensitivities, β, while learning the inverse timescales, γ, is better than doing the reverse.

While KeRNL is comparable in speed to BPTT for batch learning, we expect it to be significantly faster for online learning when the time-dependencies are of length T .

Untruncated BPTT requires information sent back T steps in time for each weight update, thus the wallclock speed of computation of the gradients at each weight update in online learning scales as T , and the total scaling is thus of order T 2 .

If BPTT updates are truncated S < T steps back in time, the scaling is ST .

KeRNL requires no backward unrolling in time, thus online KeRNL requires only O(1) time per weight update, for a total scaling of T .

As a result, optimized-speed online-KeRNL should run faster than truncated online BPTT by a factor T when the trunctation time is similar to the total time-dependencies in the problem.

We tested the performance of online KeRNL against UORO, another online learning algorithm, and online BPTT on the A n , B n task, where the network must predict the next character in a stream of letters.

Each stream consists first of a sequence of n As followed by a sequence of n Bs.

The length, n, of the sequences is randomly generated in some range.

The network cannot solve this task perfectly, as it can not predict the number of As before it has seen the sequence, but can do well by matching the number of Bs to the number of As.

We generated n ∈ {1, 32}. The minimum achievable average bit-loss for this task is 0.14.To compare with results in the literature, we implemented KeRNL in an LSTM layer, with h representing a concatenation of the hidden and cell states (Details in Appendix B).

Instead of optimizing common hyperparameters, we simply used the values from BID15 , which included decaying the learning rate in time as η t = η/(1 + α √ t).

However, we varied the learning rate η m , with η DISPLAYFORM0 Results other than those for KeRNL are from BID15 Table 5 : Average cross-entropy bit-loss (over 10 4 minibatches) on the online A n , B n task after 10 6 minibatches entropy.

Although 17-step BPTT and UORO outperformed KeRNL, we expect speed-optimized versions of KeRNL to be much faster (wall clock speed) in direct comparisons.

To test how computation time for truncated-BPTT and KeRNL compare in the online setting, we implemented a dummy RNN, where the required tensor operations were performed using a random vector for both the input data and the error signal TAB5 , both algorithms were implemented in Python for uniformity; Details in Appendix A).

KeRNL is faster than truncated BPTT beyond very short truncation lengths.

Step BPTT 3Step BPTT 10Step BPTT 20Step BPTT CPU Time 14.1 4.23 7.22 17.8 30.9 In this paper we show that KeRNL, a reduced-rank and forward-running approximation to backpropagation in RNNs, is able to perform roughly comparably to BPTT on a range of hard RNN tasks with long time-dependencies.

One may view KeRNL as imposing a strong prior on the way in which neural activity from the past should be assigned credit for current performance, through the choice of the temporal kernels K in the eligibility trace, and the choice of the sensitivity weights β.

This product of two rank-2 tensors in KeRNL (replacing the rank-4 sensitivity tensor for backpropagation in RNNs), assumes that the strength of influence of a neuron on another at fixed time-delay can be summarized by a simple sensitivity weight matrix, β ki , 4 , and a decay due to the time difference given by K. This strong simplifying assumption is augmented or mitigated by the ability to (meta)learn the parameters of the sensitivity weights and kernels in the eligibility trace, giving the rule simultaneous simplicity and flexibility.

The form of the KeRNL ansatz or prior, if wellsuited to learning problems in recurrent networks, serves as a regularizer on the types of solutions the network can find, and could even, for good choices of kernel K, provide better solutions than BPTT.

We present limited evidence that KeRNL may combat the vanishing gradient problem with tanh units by imposing a prior of long time-dependencies through the eligibility.

Finally, we show that KeRNL can be implemented online, where it has a shorter computation cycle than BPTT.KeRNL is a step toward biologically plausible learning.

It eschews the segmented two phase backpropagation algorithm for a computation that is largely feedforward.

It does not require the segmentation and storage of all past states, instead using an integrated activity or eligibility trace, and it gives rise to a naturally asymmetric structure that is more similar to the brain.

While we show empirically that KeRNL performs hill-climbing, there is no guarantee that the gradients computed by KeRNL are unbiased.

In the future, we hope to show empirically that KeRNL is able to perform well on more realistic tasks, and obtain some analytical guarantees on the performance of KeRNL.

We hope the present contribution inspires more work on training RNNs with shorter, more plausible feedback paths.

More generally, we hope that the present work shows how, with the use of reduced-rank tensor products and eligibility traces, to construct entire nested families of relaxed approximations to gradient learning in RNNs.

For the adding problem and pixel-by-pixel MNIST, we tested performance by varying η and gc over several orders of magnitude: η = {1e − 03, 1e − 04, 1e − 05, 1e − 06, 1e − 07}, gc = {1, 10, 100}, using both Adam BID7 ) and RMSProp BID16 ).

We then varied ηm = {1e − 03, 1e − 04, 1e − 05, 1e − 06, 1e − 07} on KeRNL.

We found that KeRNL was relatively robust across ηm.

For all sequence lengths, we used the hyperparameters that performed best on the task with sequence length 400.

Besides the recurrent weights of the IRNN, all other weight matrices were initialized using Xavier initialization BID2 ).

We initialized β with Xavier initialization for the tanh RNN, and to the identity for the IRNN.

This choice was motivated by the initial sensitivity of the IRNN ( we used the alternative cost function described in footnote 2.

We trained on both of these tasks using the Python numpy (Walt et al. FORMULA0 ) package.

For the dummy RNN, we used the Python numpy package BID17 ) to perform matrix algebra on a RNN with 100 hidden nodes, 100 input nodes and a tanh nonlinearity.

We called "matmul" for matrix multiplication and "einsum" for other tensor operations.

We used the "tanh" and "cosh" functions to compute the nonlinearity and its derivatives.

In this section we describe how to implement KeRNL on an LSTM Hochreiter & Schmidhuber (1997) in more detail.

The dynamics of the LSTM (without peepholes) are as follows DISPLAYFORM0 where net t j represents the presynaptic input to f t j .

The other gradients can be calculated in an analogous manner.

In the interest of full disclosure, we note that KeRNL did not perform well on next word prediction on the PennTreebank dataset.

We tested an LSTM network across a wide variety of learning rates and gradient clippings and were not able to achieve near state of the art performance using KeRNL.

<|TLDR|>

@highlight

A biologically plausible learning rule for training recurrent neural networks