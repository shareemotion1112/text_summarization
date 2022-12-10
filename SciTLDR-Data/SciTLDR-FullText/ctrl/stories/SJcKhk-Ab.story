Successful recurrent models such as long short-term memories (LSTMs) and gated recurrent units (GRUs) use \emph{ad hoc} gating mechanisms.

Empirically these models have been found to improve the learning of medium to long term temporal dependencies and to help with vanishing gradient issues.

We prove that learnable gates in a recurrent model formally provide \emph{quasi-invariance to general time transformations} in the input data.

We recover part of the LSTM architecture from a simple axiomatic approach.

This result leads to a new way of initializing gate biases in LSTMs and GRUs.

Experimentally, this new \emph{chrono initialization} is shown to greatly improve learning of long term dependencies, with minimal implementation effort.

Handling long term dependencies in temporal data has been a classical issue in the learning of recurrent networks.

Indeed, stability of a dynamical system comes at the price of exponential decay of the gradient signals used for learning, a dilemma known as the vanishing gradient problem BID21 BID9 BID1 .

This has led to the introduction of recurrent models specifically engineered to help with such phenomena.

Use of feedback connections BID10 and control of feedback weights through gating mechanisms BID5 partly alleviate the vanishing gradient problem.

The resulting architectures, namely long short-term memories (LSTMs BID10 BID5 ) and gated recurrent units (GRUs BID2 ) have become a standard for treating sequential data.

Using orthogonal weight matrices is another proposed solution to the vanishing gradient problem, thoroughly studied in BID22 BID15 BID0 BID24 BID8 .

This comes with either computational overhead, or limitation in representational power.

Furthermore, restricting the weight matrices to the set of orthogonal matrices makes forgetting of useless information difficult.

The contribution of this paper is threefold:• We show that postulating invariance to time transformations in the data (taking invariance to time warping as an axiom) necessarily leads to a gate-like mechanism in recurrent models (Section 1).

This provides a clean derivation of part of the popular LSTM and GRU architectures from first principles.

In this framework, gate values appear as time contraction or time dilation coefficients, similar in spirit to the notion of time constant introduced in BID20 .•

From these insights, we provide precise prescriptions on how to initialize gate biases (Section 2) depending on the range of time dependencies to be captured.

It has previously been advocated that setting the bias of the forget gate of LSTMs to 1 or 2 provides overall good performance BID4 BID12 .

The viewpoint here explains why this is reasonable in most cases, when facing medium term dependencies, but fails when facing long to very long term dependencies.• We test the empirical benefits of the new initialization on both synthetic and real world data (Section 3).

We observe substantial improvement with long-term dependencies, and slight gains or no change when short-term dependencies dominate.

When tackling sequential learning problems, being resilient to a change in time scale is crucial.

Lack of resilience to time rescaling implies that we can make a problem arbitrarily difficult simply by changing the unit of measurement of time.

Ordinary recurrent neural networks are highly nonresilient to time rescaling: a task can be rendered impossible for an ordinary recurrent neural network to learn, simply by inserting a fixed, small number of zeros or whitespaces between all elements of the input sequence.

An explanation is that, with a given number of recurrent units, the class of functions representable by an ordinary recurrent network is not invariant to time rescaling.

Ideally, one would like a recurrent model to be able to learn from time-warped input data ( ( )) as easily as it learns from data ( ), at least if the time warping ( ) is not overly complex.

The change of time may represent not only time rescalings, but, for instance, accelerations or decelerations of the phenomena in the input data.

We call a class of models invariant to time warping, if for any model in the class with input data ( ), and for any time warping ( ), there is another (or the same) model in the class that behaves on data ( ( )) in the same way the original model behaves on ( ).

(In practice, this will only be possible if the warping is not too complex.)

We will show that this is deeply linked to having gating mechanisms in the model.

Let us first discuss the simpler case of a linear time rescaling.

Formally, this is a linear transformation of time, that is : DISPLAYFORM0 with > 0.

For instance, receiving a new input character every 10 time steps only, would correspond to = 0.1.Studying time transformations is easier in the continuous-time setting.

The discrete time equation of a basic recurrent network with hidden state ℎ , DISPLAYFORM1 can be seen as a time-discretized version of the continuous-time equation DISPLAYFORM2 namely, (2) is the Taylor expansion ℎ( DISPLAYFORM3 with discretization step = 1.Now imagine that we want to describe time-rescaled data ( ) with a model from the same class.

Substituting ← ( ) = , ( ) ← ( ) and ℎ( ) ← ℎ( ) and rewriting (3) in terms of the new variables, the time-rescaled model satisfies DISPLAYFORM4 However, when translated back to a discrete-time model, this no longer describes an ordinary RNN but a leaky RNN (Jaeger, 2002, §8.1) .

Indeed, taking the Taylor expansion of ℎ( + ) with = 1 in (4) yields the recurrent model DISPLAYFORM5 1 We will use indices ℎ for discrete time and brackets ℎ( ) for continuous time. .

Then rename to ℎ, to and to to match the original notation.

Thus, a straightforward way to ensure that a class of (continuous-time) models is able to represent input data ( ) in the same way that it can represent input data ( ), is to take a leaky model in which > 0 is a learnable parameter, corresponding to the coefficient of the time rescaling.

Namely, the class of ordinary recurrent networks is not invariant to time rescaling, while the class of leaky RNNs (5) is.

Learning amounts to learning the global characteristic timescale of the problem at hand.

More precisely, 1/ ought to be interpreted as the characteristic forgetting time of the neural network.

Invariance to time warpings In all generality, we would like recurrent networks to be resilient not only to time rescaling, but to all sorts of time transformations of the inputs, such as variable accelerations or decelerations.

An eligible time transformation, or time warping, is any increasing differentiable function from R + to R + .

This amounts to facing input data ( ( )) instead of ( ).

Applying a time warping ← ( ) to the model and data in equation FORMULA2 and reasoning as above yields DISPLAYFORM0 Ideally, one would like a model to be able to learn from input data ( ( )) as easily as it learns from data ( ), at least if the time warping ( ) is not overly complex.

To be invariant to time warpings, a class of (continuous-time) models has to be able to represent Equation (6) for any time warping ( ).

Moreover, the time warping is unknown a priori, so would have to be learned.

Ordinary recurrent networks do not constitute a model class that is invariant to time rescalings, as seen above.

A fortiori, this model class is not invariant to time warpings either.

For time warping invariance, one has to introduce a learnable function that will represent the derivative 4 of the time warping, DISPLAYFORM1 in (6).

For instance may be a recurrent neural network taking the 's as input.5 Thus we get a class of recurrent networks defined by the equation DISPLAYFORM2 where belongs to a large class (universal approximator) of functions of the inputs.

The class of recurrent models FORMULA8 is quasi-invariant to time warpings.

The quality of the invariance will depend on the learning power of the learnable function : a function that can represent any function of the data would define a class of recurrent models that is perfectly invariant to time warpings; however, a specific model for (e.g., neural networks of a given size) can only represent a specific, albeit large, class of time warpings, and so will only provide quasi-invariance.

Heuristically, ( ) acts as a time-dependent version of the fixed in (4).

Just like 1/ above, 1/ ( 0 ) represents the local forgetting time of the network at time 0 : the network will effectively retain information about the inputs at 0 for a duration of the order of magnitude of 1/ ( 0 ) (assuming ( ) does not change too much around 0 ).Let us translate back this equation to the more computationally realistic case of discrete time, using a Taylor expansion with step size = 1, so that DISPLAYFORM3 3 Namely, in the "free" regime if inputs stop after a certain time 0, ( ) = 0 for > 0, with = 0 and ℎ = 0, the solution of FORMULA4 is ℎ( ) = − ( − 0 ) ℎ( 0), and so the network retains information from the past < 0 during a time proportional to 1/ .4 It is, of course, algebraically equivalent to introduce a function that learns the derivative of , or to introduce a function that learns .

However, only the derivative of appears in (6).

Therefore the choice to work with DISPLAYFORM4 is more convenient.

Moreover, it may also make learning easier, because the simplest case of a time warping is a time rescaling, for which DISPLAYFORM5 = is a constant.

Time warpings are increasing by definition: this translates as > 0.5 The time warping has to be learned only based on the data seen so far.where itself is a function of the inputs.

This model is the simplest extension of the RNN model that provides invariance to time warpings.

It is a basic gated recurrent network, with input gating and forget gating (1 − ).Here has to be able to learn an arbitrary function of the past inputs ; for instance, take for the output of a recurrent network with hidden state ℎ : DISPLAYFORM0 with sigmoid activation function (more on the choice of sigmoid below).

Current architectures just reuse for ℎ the states ℎ of the main network (or, equivalently, relabel ℎ ← (ℎ, ℎ ) to be the union of both recurrent networks and do not make the distinction).The model (8) provides invariance to global time warpings, making all units face the same dilation/contraction of time.

One might, instead, endow every unit with its own local contraction/dilation function .

This offers more flexibility (gates have been introduced for several reasons beyond time warpings BID9 ), especially if several unknown timescales coexist in the signal: for instance, in a multilayer model, each layer may have its own characteristic timescales corresponding to different levels of abstraction from the signal.

This yields a model DISPLAYFORM1 with ℎ and ( , ℎ , ) being respectively the activation and the incoming parameters of unit , and with each a function of both inputs and units.

Equation 10 defines a simple form of gated recurrent network, that closely resembles the evolution equation of cell units in LSTMs, and of hidden units in GRUs.

In (10), the forget gate is tied to the input gate ( and 1 − ).

Such a setting has been successfully used before (e.g. BID14 ) and saves some parameters, but we are not aware of systematic comparisons.

Below, we initialize LSTMs this way but do not enforce the constraint throughout training.

Of course, the analogy between continuous and discrete time breaks down if the Taylor expansion is not valid.

The Taylor expansion is valid when the derivative of the time warping is not too large, say, when 1 or 1 (then FORMULA9 and FORMULA8 are close).

Intuitively, for continuous-time data, the physical time increment corresponding to each time step → + 1 of the discrete-time recurrent model should be smaller than the speed at which the data changes, otherwise the situation is hopeless.

So discrete-time gated models are invariant to time warpings that stretch time (such as interspersing the data with blanks or having long-term dependencies), but obviously not to those that make things happen too fast for the model.

Besides, since time warpings are monotonous, we have DISPLAYFORM0 > 0, i.e., > 0.

The two constraints > 0 and < 1 square nicely with the use of a sigmoid for the gate function .

If we happen to know that the sequential data we are facing have temporal dependencies in an approximate range [ min , max ], it seems reasonable to use a model with memory (forgetting time) lying approximately in the same temporal range.

As mentioned in Section 1, this amounts to having values of in the range DISPLAYFORM0 The biases of the gates greatly impact the order of magnitude of the values of ( ) over time.

If the values of both inputs and hidden layers are centered over time, ( ) will typically take values centered around ( ).

Values of ( ) in the desired range [︁ For LSTMs, using a variant of BID6 ): DISPLAYFORM1 the correspondence between between the gates in (10) and those in (13) is as follows: 1 − corresponds to , and to .

To obtain a time range around for unit , we must both ensure that lies around 1 − 1/ , and that lies around 1/ .

When facing time dependencies with largest time range max , this suggests to initialize LSTM gate biases to DISPLAYFORM2 with the uniform distribution and max the expected range of long-term dependencies to be captured.

Hereafter, we refer to this as the chrono initialization.

First, we test the theoretical arguments by explicitly introducing random time warpings in some data, and comparing the robustness of gated and ungated architectures.

Next, the chrono LSTM initialization is tested against the standard initialization on a variety of both synthetic and real world problems.

It heavily outperforms standard LSTM initialization on all synthetic tasks, and outperforms or competes with it on real world problems.

The synthetic tasks are taken from previous test suites for RNNs, specifically designed to test the efficiency of learning when faced with long term dependencies BID10 BID15 BID7 BID18 BID0 ).In addition (Appendix A), we test the chrono initialization on next character prediction on the Text8 (Mahoney, 2011) dataset, and on next word prediction on the Penn Treebank dataset BID21 .

Single layer LSTMs with various layer sizes are used for all experiments, except for the word level prediction, where we use the best model from (Zilly et al., 2016), a 10 layer deep recurrent highway network (RHN).Pure warpings and paddings.

To test the theoretical relationship between gating and robustness to time warpings, various recurrent architectures are compared on a task where the only challenge comes from warping.

The unwarped task is simple: remember the previous character of a random sequence of characters.

Without time warping or padding, this is an extremely easy task and all recurrent architectures are successful.

The only difficulty will come from warping; this way, we explicitly test the robustness of various architectures to time warping and nothing else.

Uniformly time-warped tasks are produced by repeating each character maximum_warping times both in the input and output sequence, for some fixed number maximum_warping.

Variably time-warped tasks are produced similarly, but each character is repeated a random number of times uniformly drawn between 1 and maximum_warping.

The same warping is used for the input and output sequence (so that the desired output is indeed a function of the input).

This exactly corresponds to transforming input ( ) into ( ( )) with a random, piecewise affine time warping.

FIG0 gives an illustration.

For each value of maximum_warping, the train dataset consists of 50, 000 length-500 randomly warped random sequences, with either uniform or variable time warpings.

The alphabet is of size 10 (including a dummy symbol).

Contiguous characters are enforced to be different.

After warping, each sequence is truncated to length 500.

Test datasets of 10, 000 sequences are generated similarily.

The criterion to be minimized is the cross entropy in predicting the next character of the output sequence.

Note that each sample in the dataset uses a new random sequence from a fixed alphabet, and (for variable warpings) a new random warping.

A similar, slightly more difficult task uses padded sequences instead of warped sequences, obtained by padding each element in the input sequence with a fixed or variable number of 0's (in continuoustime, this amounts to a time warping of a continuous-time input sequence that is nonzero at certain points in time only).

Each time the input is nonzero, the network has to output the previous nonzero character seen.

We compare three recurrent architectures: RNNs (Eq. (2), a simple, ungated recurrent network), leaky RNNs (Eq. FORMULA5 , where each unit has a constant learnable "gate" between 0 and 1) and gated RNNs, with one gate per unit, described by (10).

All networks contain 64 recurrent units.

The point of using gated RNNs (10) ("LSTM-lite" with tied input and forget gates), rather than full LSTMs, is to explicitly test the relevance of the arguments in Section 1 for time warpings.

Indeed these LSTM-lite already exhibit perfect robustness to warpings in these tasks.

RMSprop with an parameter of 0.9 and a batch size of 32 is used.

For faster convergence, learning rates are divided by 2 each time the evaluation loss has not decreased after 100 batches.

All architectures are trained for 3 full passes through the dataset, and their evaluation losses are compared.

Each setup is run 5 times, and mean, maximum and minimum results among the five trials are reported.

Results on the test set are summarized in Fig. 1 .Gated architectures significantly outperform RNNs as soon as moderate warping coefficients are involved.

As expected from theory, leaky RNNs perfectly solve uniform time warpings, but fail to achieve optimal behavior with variable warpings, to which they are not invariant.

Gated RNNs, which are quasi invariant to general time warpings, achieve perfect performance in both setups for all values of maximum_warping.

Synthetic tasks.

For synthetic tasks, optimization is performed using RMSprop BID23 ) with a learning rate of 10 −3 and a moving average parameter of 0.9.

No gradient clipping is performed; this results in a few short-lived spikes in the plots below, which do not affect final performance.

COPY TASKS.

The copy task checks whether a model is able to remember information for arbitrarily long durations.

We use the setup from BID10 BID0 , which we summarize here.

Consider an alphabet of 10 characters.

The ninth character is a dummy character and the tenth character is a signal character.

For a given , input sequences consist of + 20 characters.

The first 10 characters are drawn uniformly randomly from the first 8 letters of the alphabet.

These first characters are followed by − 1 dummy characters, a signal character, whose aim is to signal the network that it has to provide its outputs, and the last 10 characters are dummy characters.

The target sequence consists of + 10 dummy characters, followed by the first 10 characters of the input.

This dataset is thus about remembering an input sequence for exactly timesteps.

We also provide results for the variable copy task setup presented in BID8 , where the number of characters between the end of the sequence to copy and the signal character is drawn at random between 1 and .The best that a memoryless model can do on the copy task is to predict at random from among possible characters, yielding a loss of 10 log(8) +20 BID0 .

On those tasks we use LSTMs with 128 units.

For the standard initialization (baseline), the forget gate biases are set to 1.

For the new initialization, the forget gate and input gate biases are chosen according to the chrono initialization (16), with max = 3 2 for the copy task, thus a bit larger than input length, and max = for the variable copy task.

The results are provided in Figure 3 .

Importantly, our LSTM baseline (with standard initialization) already performs better than the LSTM baseline of BID0 , which did not outperform random prediction.

This is presumably due to slightly larger network size, increased training time, and our using the bias initialization from BID4 .On the copy task, for all the selected 's, chrono initialization largely outperforms the standard initialization.

Notably, it does not plateau at the memoryless optimum.

On the variable copy task, chrono initialization is even with standard initialization for = 500, but largely outperforms it for = 1000.ADDING TASK.

The adding task also follows a setup from BID10 BID0 .

Each training example consists of two input sequences of length .

The first one is a sequence of numbers drawn from ([0, 1]), the second is a sequence containing zeros everywhere, except for two locations, one in the first half and another in the second half of the sequence.

The target is a single number, which is the sum of the numbers contained in the first sequence at the positions marked in the second sequence.

The best a memoryless model can do on this task is to predict the mean of 2 × ([0, 1]), namely 1 BID0 .

Such a model reaches a mean squared error of 0.167.LSTMs with 128 hidden units are used.

The baseline (standard initialization) initializes the forget biases to 1.

The chrono initialization uses max = .

Results are provided in Figure 4 .

For all 's, chrono initialization significantly speeds up learning.

Notably it converges 7 times faster for = 750.

The self loop feedback gating mechanism of recurrent networks has been derived from first principles via a postulate of invariance to time warpings.

Gated connections appear to regulate the local time constants in recurrent models.

With this in mind, the chrono initialization, a principled way of initializing gate biases in LSTMs, has been introduced.

Experimentally, chrono initialization is shown to bring notable benefits when facing long term dependencies.

A ADDITIONAL EXPERIMENTS On the generalization capacity of recurrent architectures.

We proceeded to test the generalization properties of RNNs, leaky RNNs and chrono RNNs on the pure warping experiments presented in Section 3.

For each of the architectures, a recurrent network with 64 recurrent units is trained for 3 epochs on a variable warping task with warps between 1 and 50.

Each network is then tested on warped sequences, with warps between 100 and an increasingly big maximum warping.

Results are summarized in Figure 5 .All networks display reasonably good, but not perfect, generalization.

Even with warps 10 times longer than the training set warps, the networks still have decent accuracy, decreasing from 100% to around 75%.Interestingly, plain RNNs and gated RNNs display a different pattern: overall, gated RNNs perform better but their generalization performance decreases faster with warps eight to ten times longer than those seen during training, while plain RNN never have perfect accuracy, below 80% even within the training set range, but have a flatter performance when going beyond the training set warp range.

Pixel level classification: MNIST and pMNIST.

This task, introduced in BID15 , consists in classifying images using a recurrent model.

The model is fed pixels one by one, from top to bottom, left to right, and has to output a probability distribution for the class of the object in the image.

We evaluate standard and chrono initialization on two image datasets: MNIST (LeCun et al., 1999) and permuted MNIST, that is, MNIST where all images have undergone the same pixel permutation.

LSTMs with 512 hidden units are used.

Once again, standard initialization sets forget biases to 1, and the chrono initialization parameter is set to the length of the input sequences, max = 784.

Results on the validation set are provided in Figure 6 .

On non-permuted MNIST, there is no clear difference, even though the best validation error is obtained with chrono initialization.

On permuted MNIST, chrono initialization performs better, with a best validation result of 96.3%, while standard initialization obtains a best validation result of 95.4%.Next character prediction on text8.

Chrono initialization is benchmarked against standard initialization on the character level text8 dataset BID17 .

Text8 is a 100M character formatted text sample from Wikipedia.

BID21 's train-valid-test split is used: the first 90M characters are used as training set, the next 5M as validation set and the last 5M as test set.

The exact same setup as in BID3 ) is used, with the code directly taken from there.

Namely: LSTMs with 2000 units, trained with Adam BID13 with learning rate 10 −3 , batches of size 128 made of non-overlapping sequences of length 180, and gradient clipping at 1.0.

Weights are orthogonally initialized, and recurrent batch normalization BID3 ) is used.

Chrono initialization with max = 8 is compared to standard = 1 initialization.

Results are presented in FIG4 .

On the validation set, chrono initialization uniformly outperforms standard initialization by a small margin.

On the test set, the compression rate is 1.37 with chrono initialization, versus 1.38 for standard initialization.8 This same slight difference is observed on two independent runs.

Our guess is that, on next character prediction, with moderately sized networks, short term dependencies dominate, making the difference between standard and chrono initialization relatively small.

Next word prediction on Penn Treebank.

To attest for the resilience of chrono initialization to more complex models than simple LSTMs, we train on word level Penn Treebank BID21 using the best deep RHN network from BID25 .

All hyperparameters are taken from of BID25 .

For the chrono bias initialization, a single bias vector is sampled according to ∼ log( (1, max )), the carry gate bias vectors of all layers are initialized to − , and the transform gate biases to .

max is chosen to be 11 (because this gives an average bias initialization close to the value 2 from (Zilly et al., 2016)).9 .

Without further hyperparameter search and with a single run, we obtain test results similar to BID25 , with a test perplexity of 6.54.

<|TLDR|>

@highlight

Proves that gating mechanisms provide invariance to time transformations. Introduces and tests a new initialization for LSTMs from this insight.

@highlight

Paper links recurrent network deisgn and its effect on how the network reacts to time transformations, and uses this to develop a simple bias initialization scheme.