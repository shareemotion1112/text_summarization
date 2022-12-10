Developing effective biologically plausible learning rules for deep neural networks is important for advancing connections between deep learning and neuroscience.

To date, local synaptic learning rules like those employed by the brain have failed to match the performance of backpropagation in deep networks.

In this work, we employ meta-learning to discover networks that learn using feedback connections and local, biologically motivated learning rules.

Importantly, the feedback connections are not tied to the feedforward weights, avoiding any biologically implausible weight transport.

It can be shown mathematically that this approach has sufficient expressivity to approximate any online learning algorithm.

Our experiments show that the meta-trained networks effectively use feedback connections to perform online credit assignment in multi-layer architectures.

Moreover, we demonstrate empirically that this model outperforms a state-of-the-art gradient-based meta-learning algorithm for continual learning on regression and classification benchmarks.

This approach represents a step toward biologically plausible learning mechanisms that can not only match gradient descent-based learning, but also overcome its limitations.

Deep learning has achieved impressive success in solving complex tasks, and in some cases its learned representations have been shown to match those in the brain [19, 10] .

However, there is much debate over how well the learning algorithm commonly used in deep learning, backpropagation, resembles biological learning algorithms.

Causes for skepticism include the facts that (1) backpropagation ignores the nonlinearities imposed by neurons in the backward pass and assumes instead that derivatives of the forward-pass nonlinearities can be applied, (2) in backpropagation, feedback path weights are exactly tied to feedforward weights, even as weights are updated with learning, and (3) backpropagation assumes alternating forward and backward passes [12] .

The question of how so-called credit assignment -appropriate propagation of learning signals to non-output neurons -can be performed in biologically plausible fashion in deep neural networks remains open.

We propose a new learning paradigm that aims to solve the credit assignment problem in more biologically plausible fashion.

Our approach is as follows: (1) endow a deep neural network with feedback connections that propagate information about target outputs to neurons at all layers, (2) apply local plasticity rules (e.g. Hebbian or neuromodulated plasticity) to update feedforward synaptic weights following feedback projections, and (3) employ meta-learning to optimize for the initialization of feedforward weights, the setting of feedback weights, and synaptic plasticity levels.

On a set of online regression and classification learning tasks, we find that meta-learned deep networks can successfully perform useful weight updates in early layers, and that feedback with local learning rules can in fact outperform gradient descent as an inner-loop learning algorithm on challenging few-shot and continual learning tasks.

Some research has investigated alternative algorithms to backpropagation that relax or eliminate the requirement of weight symmetry.

A surprising set of results [14, 17] , show that random feedback weights are sufficient to induce learning for simple tasks.

Another family of methods, known as target propagation, use a reconstruction loss to learn a feedback pathway that approximates the inverse of the feedforward pathway [3] .

However, both of these approaches have been found not to scale well to difficult tasks such as ImageNet classification [2] .

To some extent, performance can be recovered by permitting sign-symmetry in forward and backward weights [13] , but this partially re-introduces the weight symmetry issue and fails to address concerns (1) and (3) above.

Backpropagation-based deep learning notably falls short of human and animal learning in several key respects.

In particular, it has difficulty learning from few examples, and learning in online fashion from a stream of data and on multiple tasks.

One approach to addressing these issues is meta-learning, in which a network's learning procedure itself is learned in an "outer loop" of optimization.

A popular class of such methods is gradient-based meta-learning [4] , in which the network initialization is meta-optimized so that batch gradient descent will learn quickly from few examples of a new task.

In the batch (i.e. not online) learning case, this approach has the expressive power to implement any batch learning algorithm [5] .

This method has been extended to the continual learning case, in which the "inner loop" optimization consists of many online gradient steps on a potentially nonstationary data distribution [8] .

Building on the meta-learning paradigm, another line of research has explored the approach of performing inner-loop updates according to biologically motivated Hebbian learning rules rather than by gradient descent [1, 15, 16] .

However, none of these methods fully address the credit assignment problem, in that they either restrict plasticity to output weights or allow plasticity to proceed without any dependence on supervised error signals.

Recent work has also considered meta-learning algorithm for learning feedback weights [11] .

Their methods, based on node perturbation and RL algorithms, differ substantially from ours, but a comparison or synthesis could prove fruitful.

See Figure 1 for a schematic comparing our model to standard backpropagation and direct feedback alignment [17] .

In our model, a network propagates an input x forward through a neural network f (·; θ), receives a target signal y from the environment, and propagates a function g(y) back to its neurons.

The output of g is an update to the activations of the network.

Subsequently, the network undergoes synaptic plasticity according to a local learning rule that adjusts a synaptic weight w based on the previous weight value, the presynaptic activity a, and the postsynaptic activity b resulting from feedback.

In some experiments we allow plasticity only in the final N network layers (varying N ).

We may take a to be the pre or post-feedback presynaptic activations.

The post-feedback case corresponds to a model in which neural activations are updated directly with feedback and Hebbianstyle plasticity ensues.

The pre-feedback case requires error signals to be propagated without affecting the neural activations used in feedforward computation.

This approach has the advantage of avoiding possible disruption to the feedforward computation, though it may be more difficult to implement.

Possible biological implementations include a segregated dendrites model (see [6] ), or feedback through neuromodulatory signals at postsynaptic sites, with weight updates that are proportional to presynaptic and neuromodulatory activity, but not postsynaptic activity (see [7] ).

In our simulations we use Oja's learning rule:

We use linear feedback connections with one ReLU nonlinearity applied to enforce positive-valued feedback activations.

Concretely, the activation x i at the output of layer W i with corresponding feedback matrix G i is set to ReLU

, where β i controls the "strength" of feedback.

Note that β i = 0 corresponds to pure unsupervised Hebbian learning in layer i, while β i = 1 corresponds to supervised learning.

The description above specifies how a network in our model learns in its "lifetime."

However, to create a network that effectively learns using the above procedure, we employ meta-learning.

In particular, for each of our benchmark tasks (described below) we simulate an entire learning episode and test input, evaluate the performance on the test input, and backpropagate through the entire learning procedure (see [4, 8] ).

The meta-learned parameters are the initial weights θ and feedback Red quantities indicate plastic weights that change during a network's lifetime, while green quantities indicate meta-learned quantities optimized over many lifetimes.

In backpropagation, learning signals propagate through a feedback pathway involving transposes of the feedforward weights and the derivative of the neuron activation function.

Direct feedback alignment replaces the transpose matrices with random feedback pathways.

In the proposed method, feedforward weights evolve according to Hebbian plasticity during a lifetime, while feedback pathways and initial feedforward weights are meta-optimized over lifetimes.

Additionally, error signals are injected into layers directly, without any derivative computations.

function g, as well as the plasticity coefficients for each plastic weight and each layer's β coefficient, which controls the balance of supervised vs. unsupervised learning occurring in that layer.

We are able prove that sufficiently wide and deep neural networks using the above learning procedure can approximate any learning algorithm.

A learning algorithm, for our purposes, maps a set of training examples {(x, y) k } and a test input x to a predicted outputŷ * .

Theorem.

For any learning rule f target ({(x, y) k }, x ), there exists a deep ReLU network feedforward functionf (·; θ) and feedback function g(y) such thatf

, where ∆ θ (y, x) is computed following feedback according to a local learning rule at each synapse, either Hebb's rule or Oja's rule.

Proof.

See Appendix A.

It borrows techniques from [5] , which proved a similar universality result for gradient-based meta-learning in the non-online batch learning case (in which the entire dataset is available at once).

The feedforward network initialization and feedback weights are chosen so that the weight updates losslessly encode the training data in early layers of the network in such a way that it can be processed in an arbitrary way (i.e. to simulate f target ) by downstream layers.

The proof deviates from [5] in at least one major respect: in the online, continual case, the ability to choose the feedback weights is essential.

Indeed, one can indeed show that there are some reasonable f target which gradient-based learning (where feedback weights are tied to feedforward weights) cannot approximate.

We build off the experimental protocol of [8] , evaluating our approach on a regression task and a classification task, all requiring online continual learning.

We use the same architectures for the regression and classification tasks as [8] (in short, a nine-layer fully connected network for regression and an six convolutional layer + two fully connected layer for classification).

The regression problem is as follows: in each training episode, ten sine functions are sampled randomly, parameterized by amplitude in [0.

1, 5] and phase in [0, π].

In each episode, forty size-32 batches of (x, y) pairs from the first function are presented, then forty from the second, and so on.

The inputx contains both the function input x and the index k of the function being used.

The network is tasked with outputing y for a newx.

Evaluation occurs on new episodes with sine functions not used in meta-training.

Meta-training is performed for 20,000 episodes.

The dataset is split into meta-training and meta-testing classes.

During an episode, k examples from one class are presented, followed by k from the next, up to a total of N classes.

The model is tested on unseen examples from the classes in the episode.

We choose k = 1, N = 20 to consider the one-shot continual learning case.

Feedback to output activations is clamped to their target values, but feedback weights to earlier layers are meta-learned.

Evaluation occurs on episodes with classes not used in meta-training.

Meta-training is performed for 40,000 episodes.

Experimental Protocol: We evaluate our method in a number of ways: (1) We compare its performance to a gradient-based meta-learner with the same architecture (we also allow its plasticity coefficients to be meta-learned to permit fair comparison).

(2) We vary the number of plastic layers in the network.

In particular, the case in which only the output weights are plastic serves as a control to indicate whether the feedback propagation to earlier layers is indeed helping learning.

(3) We perform ablation experiments to discern the significance of the learned feedback weights.

In particular, we experiment with disallowing feedback altogether but maintaining Hebbian plasticity throughout the network, and with clamping all β parameters to 1 to prevent the network from performing Hebbian unsupervised learning along with feedback-modulated updates.

(4) We compare using pre or postfeedback presynaptic activations for plasticity updates, corresponding to the two scenarios (with or without dendritically segragated or neuromodulator-carried learning signals) described above.

Experimental outcomes are shown in Figure 2 .

We find that the architecture with meta-learned feedback and local plasticity significantly outperforms an architecturally equivalent gradient-based meta-learner on both the regression and classification tasks.

Ablation experiments show that feedback in addition to local plasticity is necessary to enable learning, and that feedback to earlier layers aids performance beyond what can be achieved with feedback only to output layers.

Interestingly, we also find that networks invariably learn β values between 0 and 1, and that networks with all β fixed at 1 perform worse.

This result indicates that a mix of unsupervised Hebbian and supervised feedback-modulated learning is beneficial.

We additionally examined the correlation between weight updates in the feedback network and updates that would be computed by gradient descent.

We find that the average correlation between the two increases from early to late layers but remains weak (< 0.1) throughout, and is even negative at some stages in the learning process on the regression task.

This phenomenon suggests that the meta-learned feedback network learns in a manner that is qualitatively different from gradient-based learners.

This work demonstrates that meta-learning procedures can optimize for neural networks that learn online using local plasticity rules and feedback connections.

Several follow-up directions could be pursued.

First, meta-learning of this kind is computationally expensive, as the meta-learner must backpropagate through the network's entire training procedure.

In order to scale this approach, it will be important to find ways to meta-train networks that generalize to longer lifetimes than were used during meta-training, or to explore alternatives to backprop-based meta-training (e.g. evolutionary algorithms).

The present work focused on the case of online learning, but the case of learning from repeated exposure to large datasets is also of interest, and scaling the method in this fashion will be crucial to exploring this regime.

Future work could also increase the biological plausibility of the method.

For instance, in the present implementation the feedforward and feedback + update passes occur sequentially.

However, a natural extension would enable them to run in parallel.

This requires ensuring (through appropriate meta-learning and/or a segregated dendrites model [6] ) that feedforward and feedback information do not interfere destructively.

Third, the meta-learning procedure in this work optimizes for a precise feedforward and feedback weight initialization.

Optimizing instead for a distribution of weight initializations or connectivity patterns would better reflect the stochasticity in synapse development.

Another direction is to apply meta-learning to understand biological learning systems (see [9] for an example of such an effort).

Well-constrained biological learning models meta-optimized in this manner might show emergence of learning circuits used in biology and even suggest new ones.

[

We prove that sufficiently wide and deep neural networks with supervised feedback and local learning rules can approximate any learning algorithm.

We borrow some of the notation and proof techniques from [5] .

We suppose the network propagates an input x forward, receives a target signal y from a supervisor, propagates a function of y back to its neural activations (feedback), and undergoes synaptic plasticity according to a local learning role dependent on these activations.

We let {(x k , y k )} denote the training data, observed in that order, and x denote the test input.

We want to construct a network architecture with feedforward functionf (·; θ) and feedback function g(y) such thatf (x ; θ ) ≈ f target ({(x, y) k }, x ), where θ = θ k , θ 0 = θ, and

The update ∆ θ (y,f (x; θ)) is assumed to proceed according to a local learning rule that adjust a synaptic weight w based on the previous weight value, the presynaptic activity a, and the postsynaptic activity b, where the values of a and b are taken following feedback propagation.

We will consider Hebb's learning rule: w ← w + α(ab) and Oja's learning rule: w ← w + α(ab − b 2 w).

We letf be a deep neural network with 2N + 2 layers and ReLU nonlinearities.

We will ensure nonnegativity of the activations of the intermediate 2N layers, allowing us to treat them as linear.

This simplification allows us to write the model as follows:

where φ(·; θ ft ) is an initial neural network with parameters θ ft .

i is a product of 2N square linear weight matrices, and f out (·; θ out ) is an output neural network with parameters θ out .

We adopt corresponding notation of G We will ensure nonnegativity of the projection so that we may ignore the ReLU.

The weights of layer W j i are then updated according to one of the following rules:

wherex j i refers to the activations at the layer preceding layer x j i , and diag(x) denotes a square diagonal matrix with x along the diagonal.

We will conduct the proofs for Hebb's rule and Oja's rule in parallel, using as an indicator variable -a value of 1 indicates we are using Oja's rule, and 0 corresponds to Hebb's rule.

Hence we may write the learning rule compactly as follows:

We set all W 2 i to be identity matrices, all β i to be a constant α (assumed in the rest of the proof to be sufficiently small).

These choices specify an architecture consisting of feedforward layers coming in groups of two.

The first layer in each group consists of a general feedforward matrix W 1 i , which we will henceforth write simply as W i .

The matrix W i will undergo plasticity at rate α induced by the feedforward activations at its input and feedback-induced activations at its output from feedback matrix G 1 i (which we will now write simply as G i ).

The second layer is a nonplastic identity transformation which effectively "shields" W i−1 from undergoing plasticity induced by the feedback projection G i .

We assume no feedback propagation to and no plasticity in the feature extractor φ or output network f out .

Thus feedforward propagation is affected only by the W i , and plasticity updates following feedback propagation will only modify the W i matrices.

Now we expandf (x ; θ ).

We let z k = N i=1 W i φ(x k ).

After one step, each W i is updated as follows:

and up to terms of O(α 2 ), the update is of the same form for all steps k = 1, 2, ..., K. We let α be small enough that higher-order terms in α can be ignored.

Now

Thus we can expand

This expansion allows us to derive the form of z for input x :

−αL

Note that appropriate choice of W i and G i allows us to simplify the form of z in Equation 3 into the following:

where the B i = N i+1 W i can be set to arbitrary invertible square matrices.

Now, our goal is to choose B i , G i , ϕ, and φ to ensure that the expression above contains a complete description of the values of {(x, y) k } (up to permuting the order of the examples) and x .

Since f out can approximate any function to arbitrary precision,f (x ; θ ) = f out (z ) can approximate any function of {(x, y) k } and x .

We set ϕ(y) = discr(y), yielding a one-hot d-dimensional vector indicating the value of y up to arbitrary precision.

We let φ (recall φ is a universal function approximator) have the following form:

where discr(x) is a one-hot J-dimensional vector indicating the value of x up to a discretization of arbitrary precision, and 0 J 2 is a zero vector of dimension J 2 .

Note that φ satisfies the requirement that all its outputs are nonnegative.

We furthermore let N = J 2 and rewrite the layer index i as a double index (j, l) where j and l each range from 1 through J. For future reference let us denote the dimensionality of y as d. B j,l and G j,l are defined as follows:

whereB j,l is a 1 × J matrix containing ones in the j and l positions and zeroes elsewhere, the I is included to ensure the invertibility of B j,l , andG j,l maps ϕ(y) to a vector consisting of a stack of J 2 d-dimensional vectors, all of which are zero except the vector in the slot corresponding to (j, l), which is ϕ(y).

That is,G

with ϕ(y) appearing in the J * j + l position.

Now we observe that:

if discr(x) ∈ {e j , e l } 0 otherwise B jl φ(x ) ≈ e j if discr(x ) ∈ {e j , e l } 0 otherwise where the approximation in the equalities is due to the terms included to ensure invertibility.

As a result, we have:

wherez k ≈ v(discr(y k ), {j + J * l, l + J * j}) if discr(x ) = e j = e l = discr(x k ) v(discr(y k ), {j + J * i|1 ≤ i ≤ J} ∪ {i + J * j|1 ≤ i ≤ J}) if discr(x ) = e j = discr(x k ) with v(a, A) defined as the J 2 d-dimensional vector consisting of J 2 stacked d-dimensional vectors, all of which are zero except those located in the slots specified by the set A, which are set to a. Now we claim that {(x, y) k } and x can be decoded with arbitrary accuracy from z .

Indeed, note that B 0 = N i=1 contains an identity matrix in its last J-dimensional block, meaning that B 0 φ(x ), and hence z , contains an unaltered copy of discr(x ) in its last J dimensions, from which x can be decoded to arbitrary accuracy.

Given the value of x we may also subtract B 0 φ(x ) from z and multiply by 1 α to obtain an unaltered version of K k=1z k .

Next, we may decode K k=1z k in the following fashion.

First, we can infer whether, and if so how many, of the x k have the same discretization as x by checking if any of the J d-dimensional vectors in slot j + J * j is nonzero, and if so, what its value is.

If slot j + J * j has nonzero value c, we subtract c from all slots with index j + J * i and i + J * j for any i. Given discr(x ) = e l the resulting vector, which we may call z k , This leaves us with a vector which in each slot j + J * l and l + J * j indicates (by summing the d components of the slot) how many times an x has been observed with discr(x) = e j and (by looking at the nonzero components in the slot) counts of how many times every possible discr(y) value was observed to correspond with that discr(x).

Thus, the set {(x, y) k } as well as x can be decoded to arbitrary accuracy from z .

Since f out is a universal function approximator, we let f out (z ) be the function that performs the decoding procedure above and then uses the inferred values of {(x, y) k } and x to approximate f target ({(x, y) k }, x ) to arbitrary precision.

<|TLDR|>

@highlight

Networks that learn with feedback connections and local plasticity rules can be optimized for using meta learning.