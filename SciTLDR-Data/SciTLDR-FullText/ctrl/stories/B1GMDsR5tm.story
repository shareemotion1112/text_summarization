Deep neural networks are almost universally trained with reverse-mode automatic differentiation (a.k.a.

backpropagation).

Biological networks, on the other hand, appear to lack any mechanism for sending gradients back to their input neurons, and thus cannot be learning in this way.

In response to this, Scellier & Bengio (2017) proposed Equilibrium Propagation - a method for gradient-based train- ing of neural networks which uses only local learning rules and, crucially, does not rely on neurons having a mechanism for back-propagating an error gradient.

Equilibrium propagation, however, has a major practical limitation: inference involves doing an iterative optimization of neural activations to find a fixed-point, and the number of steps required to closely approximate this fixed point scales poorly with the depth of the network.

In response to this problem, we propose Initialized Equilibrium Propagation, which trains a feedforward network to initialize the iterative inference procedure for Equilibrium propagation.

This feed-forward network learns to approximate the state of the fixed-point using a local learning rule.

After training, we can simply use this initializing network for inference, resulting in a learned feedforward network.

Our experiments show that this network appears to work as well or better than the original version of Equilibrium propagation.

This shows how we might go about training deep networks without using backpropagation.

Deep neural networks are almost always trained with gradient descent, and gradients are almost always computed with backpropagation.

For those interested in understanding the working of the brain in the context of machine learning, it is therefore distressing that biological neurons appear not to send signals backwards.

Biological neurons communicate by sending a sequence of pulses to downstream neurons along a one-way signaling pathway called an "axon".

If neurons were doing backpropagation, one would expect a secondary signalling pathway wherein gradient signals travel backwards along axons.

This appears not to exist, so it seems that biological neurons cannot be doing backpropagation.

Moreover, backpropagation may not be the ideal learning algorithm for efficient implementation in hardware, because it involves buffering activations for each layer until an error gradient returns.

This requirement becomes especially onerous when we wish to backpropagate through many steps of time, or through many layers of depth.

For these reasons, researchers are looking into other means of neural credit assignment -mechanisms for generating useful learning signals without doing backpropagation.

Recently, BID19 proposed a novel algorithm called Equilibrium Propagation, which enables the computation of parameter gradients in a deep neural network without backpropagation.

Equilibrium Propagation defines a neural network as a dynamical system, whose dynamics follow the negative-gradient of an energy function.

The "prediction" of this network is the fixedpoint of the dynamics -the point at which the system settles to a local minimum energy given the input, and ceases to change.

Because of this inference scheme, Equilibrium Propagation is impractically slow for large networks -the network has to iteratively converge to a fixed point at every training iteration.

In this work, we take inspiration from BID6 and distill knowledge from a slow, energy based equilibrating network into a fast feedforward network by training the feedforward network to predict the fixed-points of the equilibrating network with a local loss.

At the end of training, we can then discard the equilibrating network and simply use our feedforward network for testtime inference.

We thus have a way to train a feedforward network without backpropagation.

The resulting architecture loosely resembles a Conditional Generative Adversarial Network BID17 , where the feedforward network produces a network state which is evaluated by the energy-based equilibrating network.

To aid the reader, this paper contains a glossary of symbols in Appendix A.

Equilibrium Propagation BID19 ) is a method for training an Energy-Based Model Hopfield (1984) for classification.

The network performs inference by iteratively converging to a fixed-point, conditioned on the input data, and taking the state of the output neurons at the fixed point to be the output of the network.

The network's dynamics are defined by an energy function over neuron states s and parameters θ = (w, b): DISPLAYFORM0 Where I is the set of input neuron indices, S is the set of non-input neuron indices; s ∈ R |S| is the vector of neuron states; where α j ⊂ {I ∪ S} is the set of neurons connected to neuron j; x denotes the input vector; and ρ is some nonlinearity; w is a weight matrix with a symmetric constraint: w ij = w ji 1 , and entries only defined for {w ij : i ∈ α j } The state-dynamics for non-input neurons, derived from Equation 1, are: DISPLAYFORM1 The network is trained using a two-phase procedure, with a negative and then a positive phase.

In the negative phase, the network is allowed to settle to an energy minimum s − := arg min s E θ (s, x) conditioned on a minibatch of input data x. In the positive phase, a target y is introduced, and the energy function is augmented to "perturb" the fixed-point of the state towards the target with a "clamping factor" β: E β θ (s, x, y) = E θ (s, x) + βC(s O , y), where β is a small scalar and C(s O , y) is a loss defined between the output neurons in the network and the target y (we use C(s O , y) = s O − y 2 2 ).

The network is allowed to settle to the perturbed state s + := arg min s (E β (s, x, y)).Finally, the parameters of the network are learned based on a contrastive loss between the negativephase and positive-phase energy, which can be shown to be proportional to the gradient of the output loss in the limit of β → 0: DISPLAYFORM2 Where η is some learning rate; O ⊆ S is the subset of output neurons.

This results in a local learning rule, where parameter changes only depend on the activities of the pre-and post-synaptic neurons: DISPLAYFORM3 DISPLAYFORM4 Intuitively, the algorithm works by adjusting θ to pull arg min s E θ (s, x) closer to arg min s E β θ (s, x, y) so that the network will gradually learn to naturally minimize the output loss.

Because inference involves an iterative settling process, it is an undesirably slow process in Equilibrium propagation.

In their experiments, BID19 indicate that the number of settling steps required scales super-linearly with the number of layers.

This points to an obvious need for a fast inference scheme.

We propose training a feedforward network f φ (x) → s f ∈ R |S| to predict the fixed-point of the equilibrating network.

This allows the feedforward network to achieve two things: First, it initializes the state of the equilibrating network, so that the settling process starts in the right regime.

Second, the feedforward network can be used to perform inference at test-time, since it learns to approximate the minimal-energy state of the equilibrating network, which corresponds to the prediction.

f φ (x) is defined as follows: DISPLAYFORM0 Where DISPLAYFORM1 is the set of feedforward connections to neuron j (which is a subset of α j -the full set of connections to neuron j from the equilibrium network from Equation 1); φ = (ω, c) is the set of parameters of the feedforward network.

This feedforward network produces the initial state of the negative phase of equilibrium propagation network, given the input data -i.e., instead of starting at a zero-state, the equilibrium-propagation network is initialized in a state s f := f φ (x).

We train the parameters φ to approximate the minimal energy state s − of the equilibrating network 2 .

In other words, we seek: DISPLAYFORM2 DISPLAYFORM3 The derivative of the forward parameters of the i'th neuron, φ i = (ω αi,i , c i ), can be expanded as: 2 We could also minimize the distance with s + , but found experimentally that this actually works slightly worse than s − .

We believe that this is because equilibrium propagation depends on s − being very close to a true minimum of the energy function, and so initializing the negative phase to s f ≈ s − will lead to better gradient computations than when we initialize the negative phase to DISPLAYFORM4 The distant term is problematic, because computing DISPLAYFORM5 would require backpropagation, and the entire purpose of this exercise is to train a neural network without backpropagation.

However, we find that only optimizing the local term ∂Li ∂φi does not noticeably harm performance.

In Section 2.4 we go into more detail on why it appears to be sufficient to minimize local losses.

Over the course of training, parameters φ will learn until our feedforward network is a good predictor of the minimal-energy state of the equilibrating network.

This feedforward network can then be used to do inference: we simply take the state of the output neurons to be our prediction of the target data.

The full training procedure is outlined in Algorithm 1.

At the end of training, inference can be done either by taking the output activations from the forward pass of the inference network f φ (Algorithm 2), or by initializing with a forward pass and then iteratively minimizing the energy (Algorithm 3).

Experiments in Section 3 indicate that the forward pass performs just as well as the full energy minimization.

DISPLAYFORM6 for t ∈ 1..

T − do # Neg.

Phase 8: DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9

The fixed point s − of the equilibrating network is a nonlinear function of x, whose value is computed by iterative bottom-up and top-down inference using all of the parameters θ.

The initial state s f , by contrast, is generated in a single forward pass, meaning that the function relating s f j to its direct inputs s f αj ∈ R |αj | is constrained to the form of Equation 6.

Because of this, the computation resulting in s − may be more flexible than that of the forward pass, so it is possible for the equilibrating network to create targets that are not achievable by the neurons in the feedforward network.

This is similar to the notion of an "amortization gap" in variational inference, and we discuss this connection more in Section 4.2.Neurons in the feedforward network simply learn a linear mapping from the previous layer's activations to the targets provided by the equilibrating network.

In order to encourage the equilibrating network to stay in the regime that is reachable by the forward network, we can add a loss encouraging the fixed points to stay in the regime of the forward pass.

DISPLAYFORM0 Where λ is a hyperparameter which brings the fixed-points of the equilibrating network closer to the states of the forward pass, and encourages the network to optimize the energy landscape in the region reachable by the forward network.

Of course this may reduce the effective capacity of the equilibrating network, but if our goal is only to train the feedforward network, this does not matter.

This trick has a secondary benefit: It allows faster convergence in the negative phase by pulling the minimum of E λ θ (s, x) closer to the feedforward prediction, so we can learn with fewer convergence steps.

It can however, cause instabilities when set too high.

We investigate the effect of different values of λ with experiments in Appendix D.

In Equation 9 we decompose the loss-gradient of parameters φ into a local and a global component.

Empirically (see Figures 1, 3 ), we find that using the local loss and simply ignoring the global loss led to equally good convergence.

To understand why this is the case, let use consider a problem where we learn the mapping from an input x to a set of neuron-wise targets: s * .

Assume these targets are generated by some (unknown) set of ideal parameters φ * , so that s * = f φ * (x).

To illustrate, we consider a two layer network with φ = (w 1 , w 2 ) and φ * = (w * 1 , w * 2 ): DISPLAYFORM0 It may come as a surprise that when φ is in the neighbourhood of the ideal parameters φ * , the cosine similarity between the local and distant gradients: DISPLAYFORM1 is almost always positive, i.e. the local and distant gradients tend to be aligned.

This is a pleasant surprise because it means the local loss will tend to guide us in the right direction.

The reason becomes apparent when we define ∆w := w − w * , and write out the expression for the gradient in the limit of ∆w → 0 (see Appendix B for derivation) DISPLAYFORM2 When the term w 2 ρ (s 1 · w 2 ) 2 · w T 2 is proportional to an identity matrix, we can see that DISPLAYFORM3 and G 1 are perfectly aligned.

This will be the case when w 2 is orthogonal and layer 2 has a linear activation.

However, even for randomly sampled parameters and a nonlinear activation, w 2 ρ (s 1 · w 2 ) 2 · w T 2 tends to have a strong diagonal component and the terms thus tend to be positively aligned.

Figure 1 demonstrates that this gradient-alignment tends to increase as then network trains to approximate a set of targets (i.e. as φ → φ * ).

Note that the alignment of the local loss-gradient with the global loss-gradient is at least as high as with the distant loss-gradient, because DISPLAYFORM4 This explains the empirical observation in Figures 1 and 3 that optimizing the local, as opposed to the global, loss for the feedforward network does not appear to slow down convergence: Later layers do not have to "wait" for earlier layers to converge before they themselves converge -earlier layers Figure 1: We train a 6-layer network with parameters φ to predict layerwise targets generated by another network with random parameters φ * .

Left: We compare the convergence of the global loss of two training runs starting from the same initial conditions and identical (untuned) hyperparamters: A network with parameters φ local trained using only local losses and a network with parameters φ global trained directly on the global loss.

We note that the locally trained network converges significantly faster, suggesting that optimization is easier in the absence of the "confusing" distant-gradient signals from the not-yet-converged higher layers.

Right: We plot the cosine-similarity of local and distant components of the gradient of φ local as training progresses.

We see that as we approach convergence (as φ local → φ * ), the local and distant gradients tend to align.optimize the loss of later layers right from the beginning of training.

As shown in Figure 1 , it may in fact speed up convergence since each layer's optimizer is solving a simpler problem (albeit with changing input representations for layers > 1).When local targets s − are provided by the equilibrating network, it is not in general true that there exists some φ * such that s − = s * .

In our experiments, we observed that this did not prevent the forward network from learning to classify just as well as the equilibrating network.

However, this may not hold for more complex datasets.

As mentioned in Section 2.3, this could be resolved in future work with a scheme for annealing λ up to infinity while maintaining stable training.

We base our experiments off of those of BID19 : We use the hard sigmoid ρ(x) = max(0, min(1, x)) as our nonlinearity.

We clip the state of s i to the range (0, 1) because, since ρ (x) = 0 : x < 0 ∨ x > 1, if the system in Equation 2 were run in continuous time, it should never reach states outside this range.

Borrowing a trick from BID19 to avoid instability problems arising from incomplete negative-phase convergence, we randomly sample β ∼ U({−β base , +β base }), where β base is a small positive number, for each minibatch and use this for both the positive phase and for multiplying the learning rate in Equation 3 (for simplicity, this is not shown in Algorithm 1).3 .

Unlike BID19 , we do not use the trick of caching and reusing converged states for each data point between epochs.

In order to avoid "dead gradient" zones, we modify the activation function of our feedforward network (described in Equation 6) to ρ mod (x) = ρ(x) + 0.01x, where the 0.01 "leak" is added to prevent the feed-forward neurons from getting stuck due to zero-gradients in the saturated regions.

We use λ = 0.1 as the regularizing parameter from Equation 10, having scanned for values in Appendix D.

We verify that the our algorithm works on the MNIST dataset.

The learning curves can be seen in FIG3 .

We find, somewhat surprisingly, that the forward pass of our network performs almost indistinguishably from the performance of the negative-phase of Equilibrium Propagation.

This encouraging result shows that this approach for training a feedforward network without backprop does indeed work.

We also see from from the top-two panels of FIG3 that our approach can stabilize Equilibrium-Prop learning when we run the network for fewer steps than are needed for 3 When β is negative, the positive-state s + is pushed away from the targets, but gradients still point in the correct direction because the learning rate is scaled by −1/β.

This trick avoids an instability when, due to incomplete negative-phase convergence, the network continues approaching the true minimum of E(s, x) in the positive phase, and thus on every iteration contues to push down the energy of this "true" negative minimum [784-500-500-500-10], 50-step Training with a small-number of negative-phase steps (4 for the shallow network, 20 for the deeper) shows that feedfoward initialization makes training more stable by providing a good starting point for the negative phase optimization.

The Eq Prop s − lines on the upper plots are shortened because we terminate training when the network fails to converge.

Bottom Row: Training with more negative-phase steps shows that when the baseline Equilibrium Propagation network is given sufficient time to converge, it performs comparably with our feedforward network (Note that the y-axis scale differs from the top).

We compare the performance of Initialized Equilibrium Propagation when the feedforward network is trained using only local losses vs the global loss (i.e. using backpropagation).

s f denotes the forward pass and s − denotes the state at the end of the negative phase.

Note that we observe no disadvantage when we only use local losses.

Right: We observe the same effect as for our toy problem (see Figure 1) .

Early on in training, the local error gradients tend to align with gradients coming from higher layers.

full convergence.

By initializing the negative phase in a close-to-optimal regime, the network is able to learn when the number of steps is so low that plain Equilibrium Propagation cannot converge.

Moreover we note that as the number of steps is enough for convergence, there is not much advantage to using more negative-phase iterations -the longer negative phase does not improve our error.

In Figure 3 we demonstrate that using only local losses to update the feedforward network comes with no apparent disadvantage.

In line with our results from Section 2.4, we see that local loss gradients become aligned with the loss gradients from higher layers, explaining why it appears to be sufficient to only use the local gradients.

The most closely related work to ours is by .

There, the authors examine the idea of initializing an iterative settling process with a forward pass.

They propose using the parameters of the Equilibriating network to do a forward pass, and describe the conditions under which this provides a good approximation of the energy-minimizing state.

Their conclusion is that this criterion is met when consecutive layers of the energy-based model form a good autoencoder.

Their model differs from ours in that the parameters of the forward model are tied to the parameters of the energy-based model.

The effects of this assumption are unclear, and the authors do not demonstrate a training algorithm using this idea.

Our work was loosely inspired by BID6 , who proposed "distilling" the knowledge of a large neural network or ensemble into a smaller network which is designed to run efficiently at inference time.

In this work, we distill knowledge from a slow, equilibrating network in to a fast feedforward network.

Several authors BID9 , BID5 , BID22 have pointed out the connection between Energy Based Models and Generative Adversarial Networks (GANs).

In these works, a feedforward generator network proposes synthetic samples to be evaluated by an energybased discriminator, which learns to push down the energy of real samples and push up the energy of synthetic ones.

In these models, both the generator/sample proposer and the discriminator/energybased-model are deep feedforward networks trained with backpropagation.

In our approach, we have a similar scenario.

The inference network f φ can be thought of as a conditional generator which produces a network state s f given a randomly sampled input datum DISPLAYFORM0 .

Parameters φ are trained to approximate the minimal-energy states of the energy function: min φ f φ (x) − arg min s E θ (s, x) .

However, in our model, the Energy-Based network E θ (s, x) does not directly evaluate the energy of the generated data s f , but of the minimizing state s − = arg min s E θ (s, x) which is produced by performing T − energy-minimization steps on s f (see Algorithm 1).

Like a discriminator, the energy-based model parameters θ learn based on a contrastive loss which pushes up the energy of the "synthetic" network state s − while pushing down the energy of the "real" state s + .

In variational inference, we aim to estimate a posterior distribution p(z|x) over a latent variable z given data x, using an approximate posterior q(z).

Algorithms such as Expectation Maximization BID4 iteratively update a set of posterior parameters µ per-data point, so that z n ∼ q(z|µ n ).

In amortized inference, we instead learn a global set of parameters φ which can map a sample x n to a posterior estimate z n ∼ q φ (z|x n ).

BID3 proposed using a "recognition' network" as this amortized predictor, and BID11 showed that you can train this recognition network efficiently using the reparameterization trick.

However, this comes at the expense of an "amortization gap" BID2 -where the posterior estimate suffers due to the sharing of posterior estimation parameters across data samples.

Several recent works BID16 , BID14 , BID10 , have proposed various versions of a "teacher-student" framework, in which an amortized network q θ (z|x) provides an initial guess for the posterior, which is then refined by a slow, non-amortized network which refines q(z) in several steps into a better posterior estimate.

The "student" then learns to refine its posterior estimate using the final result of the iterative inference.

In the context of training Deep Boltzmann Machines, BID18 trained a feedforward network with backpropagation to initialize variational parameters which are then optimized to estimate the posterior over latent variables.

Initialized Equilibrium Propagation is a zero-temperature analog of amortized variational inference.

In the zero-temperature limit, the mean-field updates of variational inference reduce to coordinate ascent on the variational parameters.

The function of the amortized student network q φ (z|x) is then analogous to the function of our initializing network f φ (x), and the negative phase corresponds to the iterative optimization of varational parameters from the starting point provided by f φ (x).

Another interesting approach to shortening the inference phase in Equilibrium propagation was proposed by BID12 .

The authors propose a model that is almost a feedforward network, except that the output layer projects back to the input layer.

The negative phase consists of making several feedforward passes through the network, reprojecting the output back to the input with each pass.

Although the resulting inference model is not a feedforward network, the authors claim that this approach allows them to dramatically shorten convergence time of the negative phase.

There is also a notable similarity between Initialized Equilibrium Propagation and Method of Auxiliary Coordinates BID1 .

In that paper, the authors propose a scheme for optimizing a layered feedforward network which consists of alternating optimization of the neural activations (which can be parallelized across samples) and parameters (which can be parallelized across layers).

In order to ensure that the layer activations z k remain close to what a feedforward network can compute, the objective includes a layerwise cost DISPLAYFORM0 2 , where z k is layer k's activation, f k is layer k's function, and µ is a the strength of the layerwise cost (

as they anneal µ → ∞ this cost becomes a constraint).

This is identical in form and function to our DISPLAYFORM1 .

However, they differ from our method in that their neurons backpropagate their gradients back to input neurons (albeit only across one layer).

BID21 do something similar with using the Alternating Direction Method of Multipliers (ADMM), where Lagrange multipliers enforce the "layer matching" constraints exactly.

Both methods, unlike Equilibrium Prop, are full-batch methods.

More broadly, other approaches to backprop-free credit assignment have been tried.

DifferenceTarget propagation BID13 proposes a mechanism to send back targets to each layer, such that locally optimizing targets also optimizes the true objective.

Feedback-Alignment BID15 shows that, surprisingly, it is possible to train while using random weights for the backwards pass in backpropagation, because the forward pass parameters tends to "align" to the backwards-pass parameters so that the pseudogradients tend to be within 90• of the true gradients.

A similar phenomenon was observed in Equilibrium Propagation by BID20 , who showed that when one removed the constraint of symmetric weight in Equilibrium propagation, the weights would evolve towards symmetry through training.

Finally, BID8 used a very different approach -rather than create local targets, each layer predicts its own "pseudogradient".

The gradient prediction parameters are then trained either by the true gradients (which no longer need to arrive before a parameter update takes place) or by backpropagated versions of pseudogradients from higher layers.

In this paper we describe how to use a recurrent, energy-based model to provide layerwise targets with which to train a feedforward network without backpropagation.

This work helps us understand how the brain might be training fast inference networks.

In this view, neurons in the inference network learn to predict local targets, which correspond to the minimal energy states, which are found by the iterative settling of a separate, recurrently connected equilibrating network.

More immediately perhaps, this could lead towards efficient analog neural network designs in hardware.

As pointed out by BID19 , it is much easier to design an analog circuit to minimize some (possibly unknown) energy function than it is to design a feedforward circuit and a parallel backwards circuit which exactly computes its gradients.

However it is very undesirable for the function of a network to depend on peculiarities of a particular piece of analog hardware, because then the network cannot be easily replicated.

We could imagine using a hybrid circuit to train a digital, copy-able feedforward network, which is updated by gradients computed in the analog hardware.

Without the constraint of having to backpropagate through the feedforward network, designs could be simplified, for example to do away with the need for differentiable activation functions or to use feedforward architectures which would otherwise suffer from vanishing/exploding gradient effects.

Here we we have a reference of symbols used in the paper, in (Greek, Latin) alphabetical order.

α i ⊂ {j : j ∈ S, j = i}:

The set of neurons in the Equilibrating Network that connect to neuron i α f i = {j : j ∈ α i , j < i}: The set of neurons in the Feedforward Network connected to neuron i. β ∈ R: The perturbation factor, which modulates how much the augmented energy E DISPLAYFORM0 The minimizing state of the Energy function.

DISPLAYFORM1 The minimizing state of the augmented energy function.

DISPLAYFORM2 The state output by the feedforward network.

DISPLAYFORM3 DISPLAYFORM4 Where: DISPLAYFORM5 Now we will compute the gradient of each of the local losses with respect to ∆w 1 , in the limit where ∆w 1 is small.

DISPLAYFORM6 Why do we observe gradient alignment even at random initialization?

Let us start with the same 2-layer network defined in Appendix B Then our second term can be rewritten as: ρ(ρ(xw 1 )w 2 ) ρ (ρ(xw 1 )w 2 )w T 2 = ρ(xw 1 )w 2 w T 2 .

Given a random matrix w 2 , the matrix w 2 w T 2 will tend to have a strong diagonal component, causing this term to be aligned with ρ(xw 1 ) D EFFECT OF λ PARAMETER In Equation 10 we introduce a new parameter λ which encourages the state of the equilibrating network to state close to that of the forward pass.

Here we perform a sweep of parameter λ to evaluate its effect.

Here we scan the λ parameter and plot the final score at the end of training.

Each point in each plot corresponds to the final score of a network with parameter λ fixed at the given value throughout training.

The top row of plots is a for a small network with one hidden layer of 500 hidden units.

The bottom is for a large network with 3 layers of [500, 500, 500] hidden units.

Each column is for a different number of steps of negative-phase convergence.

DISPLAYFORM7 We see in FIG8 that when the number of steps of negative-phase convergence is small, introducing λ can allow for more stable training.

This makes sense -if the minimizing state of the equilibating network is "pulled" towards the state at the forward pass, it should take fewer steps of iteration to reach this state when initialized at the state of the forward pass.

However, we also see that training fails when λ is too high.

We believe this is because the simple iterative settling scheme (Euler integration) used in this paper, as well as the original Equilibrium Prop by BID19 , can become unstable when optimizing a loss surface with sharp, steep, minima (as are induced with large λ).

This could be addressed in later work by either using an adaptive λ term or an adaptive Euler-integration step-size.

<|TLDR|>

@highlight

We train a feedforward network without backprop by using an energy-based model to provide local targets

@highlight

This paper aims at quickening the iterative inference procedure in energy-based models trained with Equilibrium Propagation (EP), by proposing to train a feedforward network to predict a fixed point of the "equilibrating network". 

@highlight

Training a separate network to initialize recurrent networks trained using equilibrium propagation 