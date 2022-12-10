To which extent can successful machine learning inform our understanding of biological learning?

One popular avenue of inquiry in recent years has been to directly map such algorithms into a realistic circuit implementation.

Here we focus on learning in recurrent networks and investigate a range of learning algorithms.

Our approach decomposes them into their computational building blocks and discusses their abstract potential as biological operations.

This alternative strategy provides a “lazy” but principled way of evaluating ML ideas in terms of their biological plausibility

One could take each algorithm individually and try to model in detail a biophysical implementation, à la [1, 2, 3, 4, 5] .

However, it's unlikely that any single ML solution maps one-to-one onto neural circuitry.

Instead, a more useful exercise would be to identify core computational building blocks that are strictly necessary for solving temporal credit assignment, which are more likely to have a direct biological analogue.

To this end, we put forward a principled framework for evaluating biological plausibility in terms of the mathematical operations required-hence our "lazy" analysis.

We examine several online algorithms within this framework, identifying potential issues common across algorithms, for example the need to physically represent the Jacobian of the network dynamics.

We propose some novel solutions to this and other issues and in the process articulate biological mechanisms that could facilitate these solutions.

Finally, we empirically validate that these biologically realistic approximations still solve temporal credit assignment, in two simple synthetic tasks.

Plausibility criteria for recurrent learning.

Consider a recurrent network of n units, with voltages v (t) = Wr (t−1) , wherer (t) is the concatenation of recurrent and external inputs, with an additional constant input for the bias term,r

For a closer match to neural circuits, the firing rates update in continuous time, via r (t) = (1 − α)r (t−1) + αφ(v (t) ), using a point-wise neural activation function φ : R n → R n (e.g. tanh) and the network's inverse time constant α ∈ (0, 1].

The network output y (t) = softmax(W out r (t) + b out ) ∈ R nout is computed by output weights/bias W out ∈ R nout×n , b out ∈ R nout and compared with the training label y * (t) to produce an instantaneous loss L (t) .

BPTT and RTRL each provide a method for calculating the gradient of each instantaneous loss ∂L (t) /∂W ij , to be used for gradient descent.

BPTT unrolls the network over time and performs backpropagation as if on a feedforward network:

33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

Tensor(s) Update equations Notes UORO [9] A

where c * ( Table 1 : A summary of several new online algorithms' tensor structure and update equations.

where

n is the immediate credit assignment vector and

is the network Jacobian, with

(1) explicitly references activity at all time points, RTRL instead recursively updates the "influence tensor" M

, preserving the first-order long-term dependencies in the network as it runs forward.

The actual gradient is then calculated as

Unlike BPTT, every computation in RTRL involves only current time t or t − 1.

In general, an online algorithm has some tensor structure for summarizing the inter-temporal dependencies in the network, to avoid having to explicitly unroll the network.

These tensor(s) must update at each time step as new data come in.

RTRL uses an order-3 tensor, resulting in an O(n 3 ) memory requirement that is neither efficient nor biologically plausible.

However, all of the new online algorithms we discuss are only O(n 2 ) in memory.

In Table 1 we show the tensor structure and update equations for each of these algorithms in order to discuss the mathematical operations needed for each and whether a neural circuit could implement them.

How these updates lead to sensible learning is outside our scope, and we refer the reader to either the original papers [9, 10, 12, 11, 13] or the review [14] .

In a purely artificial setting, these tensor updates from Table 1 are straightforward to implement, but biologically, one has to consider how these tensors are physically represented and the mechanism for performing the updates.

We present a list of mathematical operations and comment on how a biological neural network might or might not be able to perform it in parallel with the forward pass: i A vector can be encoded as a firing rate, voltage, or any other intracellular variable.

ii A matrix must be encoded as the strengths of a set of synapses; if individual entries change, they must do so time-continuously and via a (local) synaptic plasticity rule.

iii Matrix-vector multiplication can be implemented by neural transmission, but input vectors must represent firing rates, as opposed to voltages or other intracellular variables.

iv Matrix-matrix multiplication is at face value not possible, as it requires O(n 3 ) multiplications, and there is no biological structure to support this.

v Independent additive noise is feasible; biological neural networks are naturally noisy in ways that can be leveraged for computation.

vi At face value, it is not possible to maintain a "noisy" copy of the network to estimate perturbation effects, e.g. KeRNL (Table 1) or [15] .

However, there may be workarounds.

How do different algorithms do?

RFLO is sufficiently simple to pass all of these tests, but it arguably doesn't actually solve temporal credit assignment and merely regresses natural memory traces to task labels (see Section 5.5 of [14] ), which limits its performance ceiling.

Every other algorithm fails at least one of our criteria, at least at first glance.

KF-RTRL and R-KF are out because of the matrix-matrix products in their updates.

Although the eligibility-trace-like update in KeRNL for B (t) ij is straightforward, learning the A (t) ki matrix requires a perturbed network-on the surface unlikely biologically (vi).

While UORO uses only matrix-vector products, the time-continuity requirement (ii) is awkward, because if we choose the constants ρ 0 , ρ 1 to make one update equation smooth in time (e.g. ρ 0 = 1 − , ρ 1 = , for 0 < 1), the other update becomes unstable due to the appearance of ρ

Can we fix any of these issues?

While each algorithm poses its own challenges, the Jacobian is a recurring problem for anything that meaningfully solves credit assignment.

Therefore we propose a general solution, to instead use an approximate Jacobian, whose entries we refer to as J ij , which updates at each step according to a perceptron-like learning rule:

Biologically, this would correspond to having an additional set of synapses (possibly spatially segregated from W) with their own plasticity rules [16] .

Computationally, this approximation brings no traditional speed benefits, but it offers a plausible mechanism by which a neural circuit can access its own Jacobian for learning purposes.

As for other challenges, the matrix-matrix-vector product appearing in DNI can be implemented by the circuit itself in a phase of computation separate from the forward pass.

For the intermediate result to pass through the second matrix, it must be represented as a firing rate (iii), which already requires altering the original equations to m φ l r

A l m is a voltage.

This would naively interfere with the forward pass, since v (t) = Wr (t−1) already uses the network firing rates and somatic voltages.

However, we could imagine the A synapses feeding into an electrically isolated neural compartment (say the apical dendrites) to define a separate voltage u (t+1) m , which is allowed to drive neural firing to φ(u (t+1) m ) in specific "update" phases.

We already know that branch-specific gating (by interneurons) can filter which information makes it to the soma to drive spiking [17] .

Do these fixes work empirically?

Given our criteria and novel workarounds, RFLO and DNI(b), our altered version of DNI (with the approximate Jacobian), remain as viable candidates for neural learning.

To ensure our additional approximations do not ruin performance, we empirically evaluate DNI(b), along with the original DNI and RFLO.

As upper and lower bounds on performance, respectively, we also include exact credit assignment methods (BPTT and RTRL) and a "fixed-W" algorithm that only trains the output weights.

We use two synthetic tasks, each of which requires solving temporal credit assignment and has clear markers for success.

One task ("Add") requires mapping a stream of i.i.d.

Bernoulli inputs x (t) to an output y * (t) = 0.5 + 0.5x (t−t1) − 0.25x (t−t2) [18] , with time rescaled to match α.

The label depends on the inputs via lags t 1 , t 2 that can be adjusted to modulate task difficulty.

The other task ("Mimic") requires reproducing the response of a separate RNN with the same architecture and fixed weights to a shared Bernoulli input stream.

We find that training loss for RFLO and DNI is worse than the optimal solutions (BPTT and RTRL), but both beat the fixed-W performance lower bound.

DNI(b) performs worse than original DNI, unsurprising because it involves further approximations, but still much better than the fixed-W baseline.

This demonstrates that solving temporal credit assignment is possible within biological constraints.

It is still unclear how neural circuits achieve sophisticated learning, in particular solving temporal credit assignment.

Here we approached the problem by looking for biologically sensible approximations to RTRL and BPTT.

Although we have empirical results to prove that our solutions can solve temporal credit assignment for simple tasks, the substance of our contribution is conceptual, in articulating what computations are abstractly feasible and which are not.

In particular, we have shown that accessing the Jacobian for learning is possible by using a set of synapses trained to linearly approximate the network's own dynamics.

Along the way, we have identified some key lessons.

The main one is that neural circuits need additional infrastructure specifically to support learning.

This could be extra neurons, extra compartments within neurons, separate coordinated phases of computation, input gating by inhibition, etc.

While we all know that biology is a lot more complicated than traditional models of circuit learning would suggest, it has proved difficult to identify the functional role of these details in a bottom-up way.

On the other hand, drawing a link between ML algorithms and biology can hint at precise computational roles for not well understood circuit features.

Another lesson is that implementing even fairly simple learning equations in parallel to the forward pass is nontrivial, since it already uses up so much neural hardware.

Even a simple matrix-vector product requires an entirely separate phase of network dynamics in order to not interfere with the forward pass of computation.

While it may be tempting to outsource some of these update equations to separate neurons, the results would not be locally available to drive synaptic plasticity.

Of course, we acknowledge that any particular solution, whether RFLO or DNI, is a highly contrived, specific, and likely incorrect guess at how neural circuits learn, but we believe the exercise has big-picture implications for how to think about biological learning.

Beyond the particular topic of online learning in recurrent networks, our work provides a general blueprint for abstractly evaluating computational models as mechanistic explanations for biological neural networks.

Knowing what computational building blocks are at our disposal and what biological details are needed to implement them is an important foundation for studying ML algorithms in a biological context.

<|TLDR|>

@highlight

We evaluate new ML learning algorithms' biological plausibility in the abstract based on mathematical operations needed