The biological plausibility of the backpropagation algorithm has long been doubted by neuroscientists.

Two major reasons are that neurons would need to send two different types of signal in the forward and backward phases, and that pairs of neurons would need to communicate through symmetric bidirectional connections.

We present a simple two-phase learning procedure for fixed point recurrent networks that addresses both these issues.

In our model, neurons perform leaky integration and synaptic weights are updated through a local mechanism.

Our learning method extends the framework of Equilibrium Propagation to general dynamics, relaxing the requirement of an energy function.

As a consequence of this generalization, the algorithm does not compute the true gradient of the objective function, but rather approximates it at a precision which is proven to be directly related to the degree of symmetry of the feedforward and feedback weights.

We show experimentally that the intrinsic properties of the system lead to alignment of the feedforward and feedback weights, and that our algorithm optimizes the objective function.

Deep learning BID18 is the de-facto standard in areas such as computer vision BID17 , speech recognition and machine translation BID3 .

These applications deal with different types of data and share little in common at first glance.

Remarkably, all these models typically rely on the same basic principle: optimization of objective functions using the backpropagation algorithm.

Hence the question: does the cortex in the brain implement a mechanism similar to backpropagation, which optimizes objective functions?The backpropagation algorithm used to train neural networks requires a side network for the propagation of error derivatives, which is vastly seen as biologically implausible BID7 .

One hypothesis, first formulated by Hinton & McClelland (1988) , is that error signals in biological networks could be encoded in the temporal derivatives of the neural activity and propagated through the network via the neuronal dynamics itself, without the need for a side network.

Neural computation would correspond to both inference and error back-propagation.

This work also explores this idea.

The framework of Equilibrium Propagation BID29 requires the network dynamics to be derived from an energy function, enabling computation of an exact gradient of an objective function.

However, in terms of biological realism, the requirement of symmetric weights between neurons arising from the energy function is not desirable.

The work presented here extends this framework to general dynamics, without the need for energy functions, gradient dynamics, or symmetric connections.

Our approach is the following.

We start from classical models in neuroscience for the dynamics of the neuron's membrane voltage and for the synaptic plasticity (section 3).

Unlike in the Hopfield model BID16 , we do not assume pairs of neurons to have symmetric connections.

We then describe an algorithm for supervised learning based on these models (section 4) with minimal extra assumptions.

Our model is based on two phases: at prediction time, no synaptic changes occur, whereas a local update rule becomes effective when the targets are observed.

The proposed update mechanism is compatible with spike-timing-dependent plasticity , which supposedly governs synaptic changes in biological neural systems.

Finally, we show that the proposed algorithm has the desirable machine learning property of optimizing an objective function (section 5).

We show this experimentally ( Figure 3 ) and we provide the beginning for a theoretical explanation.

Historically, models based on energy functions and/or gradient dynamics have represented a key subject of neural network research.

Their mathematical properties often allow for a simplified analysis, in the sense that there often exists an elegant formula or algorithm for computing the gradient of the objective function BID0 BID24 BID29 .

However, we argue in this section that 1. due to the energy function, such models are very restrictive in terms of dynamics they can model -for instance the Hopfield model requires symmetric weights, 2.

machine learning algorithms do not require computation of the gradient of the objective function, as shown in this work and the work of BID19 .In this work, we propose a simple learning algorithm based on few assumptions.

To this end, we relax the requirement of the energy function and, at the same time, we give up on computing the gradient of the objective function.

We believe that, in order to make progress in biologically plausible machine learning, dynamics more general than gradient dynamics should be studied.

As discussed in section 6, another motivation for studying more general dynamics is the possible implementation of machine learning algorithms, such as our model, on analog hardware: analog circuits implement differential equations, which do not generally correspond to gradient dynamics.

Most dynamical systems observed in nature cannot be described by gradient dynamics.

A gradient field is a very special kind of vector field, precisely because it derives from a primitive scalar function.

The existence of a primitive function considerably limits the "number of degrees of freedom" of the vector field and implies important restrictions on the dynamics.

In general, a vector field does not derive from a primitive function.

In particular, the dynamics of the leaky integrator neuron model studied in this work (Eq. 1) is not a gradient dynamics, unless extra (biologically implausible) assumptions are made, such as exact symmetry of synaptic weights (W ij = W ji ) in the case of the Hopfield model.

Machine learning relies on the basic principle of optimizing objective functions.

Most of the work done in deep learning has focused on optimizing objective functions by gradient descent in the weight space (thanks to backpropagation).

Although it is very well known that following the gradient is not necessarily the best option -many optimization methods based on adaptive learning rates for individual parameters have been proposed such as RMSprop Tieleman & Hinton (2012) and Adagrad BID9 -almost all proposed optimization methods rely on computing the gradient, even if they do not follow the gradient.

In the field of deep learning, "computing the gradient" has almost become synonymous with "optimizing".In fact, in order to optimize a given objective function, not only following the gradient unnecessary, but one does not even need to compute the gradient of that objective function.

A weaker sufficient condition is to compute a direction in the parameter space whose scalar product with the gradient is negative, without computing the gradient itself.

A major step forward was achieved by BID19 .

One of the contributions of their work was to dispel the long-held assumption that a learning algorithm should compute the gradient of an objective function in order to be sound.

Their algorithm computes a direction in the parameter space that has at first sight little to do with the gradient of the objective function.

Yet, their algorithm "learns" in the sense that it optimizes the objective function.

By giving up on the idea of computing the gradient of the objective function, a key aspect rendering backpropagation biologically implausible could be fixed, namely the weight transport problem.

The work presented here is along the same lines.

We give up on the idea of computing the gradient of the objective function, and by doing so, we get rid of the biologically implausible symmetric connections required in the Hopfield model.

In this sense, the "weight transport" problem in the backpropagation algorithm appears to be similar, at a high level, to the requirement of symmetric connections in the Hopfield model.

We suggest that in order to make progress in biologically plausible machine learning, it might be necessary to move away from computing the true gradients in the weight space.

An important theoretical effort to be made is to understand and characterize the dynamics in the weight space that optimize objective functions.

The set of such dynamics is of course much larger than the tiny subset of gradient dynamics.

We denote by s i the averaged membrane voltage of neuron i across time, which is continuous-valued and plays the role of a state variable for neuron i. We also denote by ρ(s i ) the firing rate of neuron i.

We suppose that ρ is a deterministic function (nonlinear activation) that maps the averaged voltage s i to the firing rate ρ(s i ).

The synaptic strength from neuron j to neuron i is denoted by W ij .

In biological neurons a classical model for the time evolution of the membrane voltage s i is the rate-based leaky integrator neuron model, in which neurons are seen as performing leaky temporal integration of their past inputs BID8 : DISPLAYFORM0 Unlike energy-based models such as the Hopfield model BID16 that assume symmetric connections between neurons, in the model studied here the connections between neurons are not tied.

Thus, our model is described by a directed graph, whereas the Hopfield model is best regarded as an undirected graph ( Figure 1 ).(a) The network model studied here is best represented by a directed graph.(b) The Hopfield model is best represented by an undirected graph.

Figure 1: From the point of view of biological plausibility, the symmetry of connections in the Hopfield model is a major drawback (1b).

The model that we study here is, like a biological neural network, a directed graph (1a).3.2 SPIKE-TIMING DEPENDENT PLASTICITY Spike-Timing Dependent Plasticity (STDP) is considered a key mechanism of synaptic change in biological neurons BID21 BID11 BID22 .

STDP is often conceived of as a spike-based process which relates the change in the synaptic weight W ij to the timing difference between postsynaptic spikes (in neuron i) and presynaptic spikes (in neuron j) BID5 .

In fact, both experimental and computational work suggest that postsynaptic voltage, not postsynaptic spiking, is more important for driving LTP (Long Term Potentiation) and LTD (Long Term Depression) BID6 BID20 .Similarly, have shown in simulations that a simplified Hebbian update rule based on pre-and post-synaptic activity can functionally reproduce STDP: DISPLAYFORM1 Throughout this paper we will refer to this update rule (Eq. 2) as "STDP-compatible weight change" and propose a machine learning justification for such an update rule.

Let s = (s 1 , s 2 , . . .) be the global state variable and parameter W the matrix of connection weights W ij .

We write µ(W, s) the vector whose components are defined as DISPLAYFORM0 defining a vector field over the neurons state space, indicating in which direction each neuron's activity changes: DISPLAYFORM1 Since ρ(s j ) = ∂µi ∂Wij (W, s), the weight change Eq. 2 can also be expressed in terms of µ in the form dW ij ∝ ∂µi ∂Wij (W, s)ds i .

Note that for all i = i we have ∂µ i ∂Wij = 0 since to each synapse W ij corresponds a unique post-synaptic neuron s i .

Hence dW ij ∝ ∂µ ∂Wij (W, s) · ds.

We rewrite the STDP-compatible weight change in the more concise form DISPLAYFORM2

The framework and the algorithm in their general forms are described in Appendix A.To illustrate our algorithm, we consider here the supervised setting in which we want to predict an output y given an input x. We describe a simple two-phase learning procedure based on the dynamics Eq. 4 and Eq. 5 for the state and the parameter variables.

This algorithm is similar to the one proposed by BID29 , but here we do not assume symmetric weights between neurons.

Note that similar algorithms have also been proposed by O'Reilly (1996); BID13 or more recently by BID23 .

Our contribution in this work are theoretical insights into why the proposed algorithm works.

In the supervised setting studied here, the units of the network are split in two sets: the inputs x whose values are always clamped, and the dynamically evolving units h (the neurons activity, indicating the state of the network), which themselves include the hidden layers (h 1 and h 2 here) and an output layer (h 0 here), as in Figure 2 .

In this context the vector field µ is defined by its components µ 0 , µ 1 and µ 2 on h 0 , h 1 and h 2 respectively, as follows: DISPLAYFORM0 Here the scalar function ρ is applied elementwise to the components of the vectors.

The neurons h follow the dynamics DISPLAYFORM1 In this section and the next we use the notation h rather than s for the state variable.

The layer h 0 plays the role of the output layer where the prediction is read.

The target outputs, denoted by y, have the same dimension as the output layer h 0 .

The discrepancy between the output units h 0 and the targets y is measured by the quadratic cost function DISPLAYFORM2 Unlike in the continuous Hopfield model, here the feed-forward and feedback weights are not tied, and in general the state dynamics Eq. 9 is not guaranteed to converge to a fixed point.

However we observe experimentally that the dynamics almost always converges.

We will see in section 5 that, for a whole set of values of the weight matrix W .

the dynamics of the neurons h converges.

Assuming this condition to hold, the dynamics of the neurons converge to a fixed point which we denote by h 0 (beware not to confuse with the notation for the output units h 0 ).

The prediction h 0 0 is then read out on the output layer and compared to the actual target y. The objective function (for a single training case (x, y)) that we aim to minimize is the cost at the fixed point h 0 , which we write DISPLAYFORM3 Note that this objective function is the same as the one proposed by BID1 BID28 .

Their method to optimize J is to compute the gradient of J thanks to an algorithm which they call "Recurrent Backpropagation".

Other methods related to Recurrent Backpropagation exist to compute the gradient of J -in particular the "adjoint method", "implicit differentiation" and "Backprop Through Time".

These methods are biologically implausible, as argued in Appendix B.Here our approach to optimize J is to give up on computing the true gradient of J and, instead, we propose a simple algorithm based only on the leaky integrator dynamics (Eq. 4) and the STDPcompatible weight change (Eq. 5).

We will show in section 5 that our algorithm computes a proxy for the gradient of J. Also, note that in its general formulation, our algorithm applies to any vector field µ and cost function C (Appendix A)

The idea of Equilibrium Propagation BID29 is to see the cost function C (Eq. 10) as an external potential energy for the output units h 0 , which can drive them towards their target y.

Following the same idea we define the "extended vector field" µ β as DISPLAYFORM0 and we redefine the dynamics of the state variable h as DISPLAYFORM1 The real-valued scalar β ≥ 0 controls whether the output h 0 is pushed towards the target y or not, and by how much.

We call β the "influence parameter" or "clamping factor".The differential equation of motion Eq. 13 can be seen as a sum of two "forces" that act on the temporal derivative of the state variable h. Apart from the vector field µ that models the interactions between neurons within the network, an "external force" −β ∂C ∂h is induced by the external potential βC and acts on the output neurons: DISPLAYFORM2 DISPLAYFORM3 The form of Eq. 14 suggests that when β = 0, the output units h 0 are not sensitive to the targets y from the outside world.

In this case we say that the network is in the free phase (or first phase).

When β > 0, the "external force" drives the output units h 0 towards the target y. When β 0 (small positive value), we say that the network is in the weakly clamped phase (or second phase).

Also, note that the case β → ∞, not studied here, would correspond to fully clamped outputs.

We propose a simple two-phase learning procedure, similar to the one proposed by BID29 .

In the first phase of training, the inputs are set (clamped) to the input values.

The state variable (all the other neurons) follows the dynamics Eq. 9 (or equivalently Eq. 13 with β = 0) and the output units are free.

We call this phase the free phase, as the system relaxes freely towards the free fixed point h 0 without any external constraints on his output neurons.

During this phase, the synaptic weights are unchanged.

Figure 2: Input x is clamped.

Neurons h include "hidden layers" h 2 and h 1 , and "output layer" h 0 that corresponds to the layer where the prediction is read.

Target y has the same dimension as h 0 .

The clamping factor β scales the "external force" −β ∂C ∂h that attracts the output h 0 towards the target y.

In the second phase, the influence parameter β takes on a small positive value β 0.

The state variable follows the dynamics Eq. 13 for that new value of β, and the synaptic weights follow the STDP-compatible weight change Eq. 5.

This phase is referred to as the weakly clamped phase.

The novel "external force" −β ∂C ∂h in the dynamics Eq. 13 acts on the output units and drives them towards their targets (Eq. 14).

This force models the observation of y: it nudges the output units h 0 from their free fixed point value in the direction of their targets.

Since this force only acts on the output layer h 0 , the other hidden layers (h i with i > 0) are initially at equilibrium at the beginning of the weakly clamped phase.

The perturbation caused at the output layer will then propagate backwards along the layers of the network, giving rise to "back-propagating" error signals.

The network eventually settles to a new nearby fixed point, corresponding to the new value β 0, termed weakly clamped fixed point and denoted h β .

Our model assumes that the STDP-compatible weight change (Eq. 5) occurs during the second phase of training (weakly clamped phase) when the network's state moves from the free fixed point h 0 to the weakly clamped fixed point h β .

Normalizing by a factor β and letting β → 0, we get the update rule ∆W ∝ ν(W ) for the weights, where ν(W ) is the vector defined as DISPLAYFORM0 The vector ν(W ) has the same dimension as W .

Formally ν is a vector field in the weight space.

It is shown in section 5 that ν(W ) is a proxy to the gradient ∂J ∂W .

The effectiveness of the proposed method is demonstrated through experimental studies (Figure 3 ).

In this section, we attempt to understand why the proposed algorithm is experimentally found to optimize the objective function J (Figure 3) .

We say that W is a "good parameter" if:1.

for any initial state for the neurons, the state dynamics dh dt = µ (W, x, h) converges to a fixed point -a condition required for the algorithm to be correctly defined, 2.

the scalar product ∂J ∂W · ν(W ) at the point W is negative -a desirable condition for the algorithm to optimize the objective function J.Experiments show that the dynamics of h (almost) always converges to a fixed point and that J consistently decreases (Figure 3) .

This means that, during training, as the parameter W follows the update rule ∆W ∝ ν(W ), all values of W that the network takes are "good parameters".

In this section we attempt to explain why.

∂J ∂W AND ν Theorem 1.

The gradient of J can be expressed in terms of µ and C as DISPLAYFORM0 Similarly, the vector field ν (Eq. 16) is equal to DISPLAYFORM1 In these expressions, all terms are evaluated at the fixed point h 0 .

and that the angle between these two vectors is directly linked to the "degree of symmetry" of the Jacobian of µ.An important particular case is the setting of Equilibrium Propagation BID29 , in which the vector field µ is a gradient field µ = − ∂E ∂h , meaning that it derives from an energy function E. In this case the Jacobian of µ is symmetric since it is the Hessian of E. Indeed DISPLAYFORM0 .

Therefore, Theorem 1 shows that ν is also a gradient field, namely the gradient of the objective function J, that is ν = − ∂J ∂W .

Note that in this setting the set of "good parameters" is the entire weight space -for all W , the dynamics dh dt = − ∂E ∂h (W, h) converges to an energy minimum, and W converges to a minimum of J since ∆W ∝ − ∂J ∂W .

We argue that the set of "good parameters" covers a large proportion of the weight space and that they contain the matrices W that present a form of symmetry or "alignment".

In the next subsection, we discuss how this form of symmetry may arise from the learning procedure itself.

Figure 3: Example system trained on the MNIST dataset, as described in Appendix C. The objective function is optimized: the training error decreases to 0.00% in around 70 epochs.

The generalization error is about 2%.

Right: A form of symmetry or alignment arises between feedforward and feedback weights W k,k+1 and W k+1,k in the sense that tr(W k,k+1 · W k+1,k ) > 0.

This architecture uses 3 hidden layers each of dimension 512.

Experiments show that a form of symmetry between feedforward and feedback weights arises from the learning procedure itself (Figure 3 ).

Although the causes for this phenomenon aren't understood very well yet, it is worth pointing out that similar observations have been made in previous work and different settings.

A striking example is the following one.

A major argument against the plausibility of backpropagation in feedforward nets is the weight transport problem: the signals sent forward in the network and those sent backward use the same connections.

BID19 have observed that, in the backward pass, (back)propagating the error signals through fixed random feedback weights (rather than the transpose of the feedforward weights) does not harm learning.

Moreover, the learned feedforward weights W k,k+1 tend to 'align' with the fixed random feedback weights W k+1,k in the sense that the trace of W k,k+1 · W k+1,k is positive.

Denoising autoencoders without tied weights constitute another example of learning algorithms where a form of symmetry in the weights has been observed as learning goes on BID31 .The theoretical result from BID2 also shows that, in a deep generative model, the transpose of the generative weights perform approximate inference.

They show that the symmetric solution minimizes the autoencoder reconstruction error between two successive layers of rectifying linear units.

Our approach provides a basis for implementing machine learning models in continuous-time systems, while requirements regarding the actual dynamics are reduced to a minimum.

This means that the model applies to a large class of physical realizations of vectorfield dynamics, including analog electronic circuits.

Implementations of recurrent networks based on analog electronics have been proposed in the past, e.g. BID13 , however, these models typically required circuits and associated dynamics to adhere to an exact theoretical model.

With our framework, we provide a way of implementing a learning system on a physical substrate without even knowing the exact dynamics or microscopic mechanisms that give rise to it.

Thus, this approach can be used to analog electronic system end-to-end, without having to worry about exact device parameters and inaccuracies, which inevitably exist in any physical system.

Instead of approximately implementing idealized computations, the actual analog circuit, with all its individual device variations, is trained to perform the task of interest.

Thereby, the more direct implementation of the dynamics might result in advantages in terms of speed, power, and scalability, as compared to digital approaches.

Our model demonstrates that biologically plausible learning in neural networks can be achieved with relatively few assumptions.

As a key contribution, in contrast to energy-based approaches such as the Hopfield model, we do not impose any symmetry constraints on the neural connections.

Our algorithm assumes two phases, the difference between them being whether synaptic changes occur or not.

Although this assumption begs for an explanation, neurophysiological findings suggest that phase-dependent mechanisms are involved in learning and memory consolidation in biological systems.

Theta waves, for instance, generate neural oscillatory patterns that can modulate the learning rule or the computation carried out by the network BID26 .

Furthermore, synaptic plasticity, and neural dynamics in general, are known to be modulated by inhibitory neurons and dopamine release, depending on the presence or absence of a target signal.

BID10 ; BID27 .In its general formulation (Appendix A), the work presented in this paper is an extension of the framework of BID29 to general dynamics.

This is achieved by relaxing the requirement of an energy function.

This generalization comes at the cost of not being able to compute the (true) gradient of the objective function but, rather a direction in the weight space which is related to it.

Thereby, precision of the approximation of the gradient is directly related to the "alignment" between feedforward and feedback weights.

Even though the exact underlying mechanism is not fully understood yet, we observe experimentally that during training the weights symmetrize to some extent, as has been observed previously in a variety of other settings BID19 BID31 BID2 .

Our work shows that optimization of an objective function can be achieved without ever computing the (true) gradient.

More thorough theoretical analysis needs to be carried out to understand and characterize the dynamics in the weight space that optimize objective functions.

Naturally, the set of all such dynamics is much larger than the tiny subset of gradient-based dynamics.

Our framework provides a means of implementing learning in a variety of physical substrates, whose precise dynamics might not even be known exactly, but which simply have to be in the set of sup-ported dynamics.

In particular, this applies to analog electronic circuits, potentially leading to faster, more efficient, and more compact implementations.

In this Appendix, we present the framework and the algorithm in their general formulations and we prove the theoretical results.

We consider a physical system specified by a state variable s and a parameter variable θ.

The system is also influenced by an external input v, e.g. in the supervised setting v = (x, y) where y is the target that the system wants to predict given x.

Let s → µ(θ, v, s) be a vector field in the state space and C(θ, v, s) a cost function.

We assume that the state dynamics induced by µ converges to a stable fixed point s 0 θ,v , corresponding to the "prediction" from the model and characterized by DISPLAYFORM0 The objective function that we want to optimize is the cost at the fixed point DISPLAYFORM1 Note the distinction between J and C: the cost function is defined for any state s whereas the objective function is the cost at the fixed point.

The training objective (for a single data sample v) is find arg min DISPLAYFORM2 Equivalently, the training objective can be reformulated as a constrained optimization problem: DISPLAYFORM3 where the constraint µ (θ, v, s) = 0 is the fixed point condition.

All traditional methods to compute the gradient of J (adjoint method, implicit differentiation, Recurrent Backpropagation and Backpropagation Through Time or BPTT) are thought to be biologically implausible.

Our approach is to give up on computing the gradient of J and let the parameter variable θ follow a vector field ν in the parameter space which is "close" to the gradient of J.Before defining ν we first introduce the "extended vector field" DISPLAYFORM4 where β is a real-valued scalar called "influence parameter".

Then we extend the notion of fixed point for any value of β.

For any β we define the β-fixed point s β θ,v such that DISPLAYFORM5 Under mild regularity conditions on µ and C, the implicit function theorem ensures that, for a fixed data sample v, the funtion (θ, β) → s β θ,v is differentiable.

Now for every θ and v we define the vector ν(θ, v) in the parameter space as DISPLAYFORM6 As shown in section 4, the second term on the right hand side can be estimated in a biologically realistic way thanks to a two-phase training procedure.

and ∂s DISPLAYFORM7 Proof of Lemma 2.

First we differentiate the fixed point equation Eq. 27 with respect to θ: DISPLAYFORM8 Rearranging the terms we get Eq. 28.

Similarly we differentiate the fixed point equation Eq. 27 with respect to β: DISPLAYFORM9 Rearranging the terms we get Eq. 29.Theorem 3.

The gradient of the objective function is equal to DISPLAYFORM10 and the vector field ν is equal to DISPLAYFORM11 All the factors on the right-hand sides of Eq. 32 and Eq. 33 are evaluated at the fixed point s 0 θ .Proof of Theorem 3.

Let us compute the gradient of the objective function with respect to θ.

Using the chain rule of differentiation we get DISPLAYFORM12 Hence Eq. 32 follows from Eq. 28 evaluated at β = 0.

Similarly, the expression for the vector field ν (Eq. 33) follows from its definition (Eq. 26), the identity Eq. 29 evaluated at β = 0 and the fact that We finish by stating and proving a last result.

Consider the setting introduced in section 4 with the quadratic cost function C = 1 2 y − h 0 2 .

In the weakly clamped phase, the "external influence" −β (y − h 0 ) added to the vector field µ (with β 0) slightly attracts the output state h 0 to the target y.

It is intuitively clear that the weakly clamped fixed point is better than the free fixed point in terms of prediction error.

Proposition 5 below generalizes this property to any vector field µ and any cost function C.Proposition 4.

Let s 0 be a stable fixed point of the vector field s → µ(s), in the sense that s − s 0 · µ (s) < 0 for s in the neighborhood of s 0 (i.e. the vector field at s points towards s 0 ).

Then the Jacobian of µ at the fixed point ∂µ ∂s s 0 is negative, in the sense that DISPLAYFORM13 Proof.

Let v be a vector in the state space, α > 0 a positive scalar and let s := s 0 + αv.

For α small enough, the vector s is in the region of stability of s 0 .

Using a first order Taylor expansion and the fixed point condition µ s 0 = 0 we get DISPLAYFORM14 as α → 0.

Hence the result.

The following proposition shows that, unless the free fixed point s 0 θ,v is already optimal in terms of cost value, for β > 0 small enough, the nudged fixed point s β θ,v achieves lower cost value than the free fixed point.

Thus, a small perturbation due to a small increment of β nudges the network towards a configuration that reduces the cost value.

The inequality holds because B ADJOINT METHOD AND RELATED ALGORITHMS Earlier work have proposed various methods to compute the gradient of the objective function J (Eq. 20).

One common method is the "adjoint method".

In the context of fixed point recurrent neural networks studied here, the adjoint method leads to Backpropagation Through Time (BPTT) and "Recurrent Backpropagation" BID1 BID28 .

BPTT is the method of choice today for deep learning but its biological implausibility is obvious -it requires the network to store all its past states for the propagation of error derivatives in the second phase.

Recurrent Backpropagation corresponds to a special case of BPTT where the network is initialized at the fixed point.

This algorithm does not need to store the past states of the network (the state at the fixed point suffices) but it still requires neurons to send a different kind of signals through a different computational path in the second phase, which seems therefore less biologically plausible than our algorithm.

Our approach is to give up on the idea of computing the gradient of the objective function.

Instead we show that the STDP-compatible weight change computes a proxy to the gradient in a more biologically plausible way.

For completeness, we state and prove a continuous-time version of Backpropagation Through Time and Recurrent Backpropagation.

The formulas for the propagation of error derivatives (Theorem 6 and Corollary 7) will make it obvious that our algorithm is more biologically plausible than both of these algorithms.

To keep notations simple, we omit to write dependences in the data sample v.

We consider the dynamics ds dt = µ(θ, s) and denote by s t the state of the system at time t ≥ 0 when it starts from an initial state s 0 at time t = 0.

Note that s t converges to the fixed point s 0 θ as t → ∞. We then define a family of objective functions L(θ, s 0 , T ) := C (θ, s T ) (42) for every couple (s 0 , T ) of initial state s 0 and duration T ≥ 0.

This is the cost of the state at time t = T when the network starts from the state s 0 at time t = 0.

Note that L(θ, s 0 , T ) tends to J(θ) as T → ∞ (Eq. 20).We want to compute the gradient ∂L ∂θ (θ, s 0 , T ) as T → ∞. For that purpose, we fix T to a large value and we consider the following quantity ∂L ∂s T −t := ∂L ∂s (θ, s T −t , t) ,which represents the "partial derivative of the cost with respect to the state at time T − t".

In other words this is the "partial derivative of the cost with respect to the (T − t)-th hidden layer" if one regards the network as unfolded in time (though time is continuous here).

The formulas in Theorem 6 below correspond to a continuous-time version of BPTT for the propagation of the partial derivatives Computing ∂L ∂s T −t h i cannot reach values above 1.

As a consequence h i must remain in the domain 0 ≤ h i ≤ 1.

Therefore, rather than the standard gradient descent (Eq. 58), we will use a slightly different update rule for the state variable h: DISPLAYFORM0 This little implementation detail turns out to be very important: if the i-th hidden unit was in some state h i < 0, then Eq. 58 would give the update rule h i ← (1 − )h i , which would imply again h i < 0 at the next time step (assuming < 1).

As a consequence h i would remain in the negative range forever.

We use different learning rates for the different layers in our experiments.

We do not have a clear explanation for why this improves performance, but we believe that this is due to the finite precision with which we approach the fixed points.

The hyperparameters chosen for each model are shown in Table 1 and the results are shown in Figure 3 .

We initialize the weights according to the Glorot-Bengio initialization BID12 .

Table 1 : Hyperparameters.

for both the 2 and 3 layered MNIST.

Example system trained on the MNIST dataset, as described in Appendix C. The objective function is optimized: the training error decreases to 0.00%.

The generalization error lies between 2% and 3% depending on the architecture.

The learning rate is used for iterative inference (Eq. 59).

β is the value of the clamping factor in the second phase.

α k is the learning rate for updating the parameters in layer k.

We were also able to train on MNIST using a Convolutional Neural Network (CNN).

We got around 2% generalization error.

The hyperparameters chosen to train this Convolutional Neural Network are shown in TAB0 .

<|TLDR|>

@highlight

We describe a biologically plausible learning algorithm for fixed point recurrent networks without tied weights