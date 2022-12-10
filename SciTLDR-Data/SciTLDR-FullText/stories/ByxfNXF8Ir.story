Backpropagation is driving today's artificial neural networks.

However, despite extensive research, it remains unclear if the brain implements this algorithm.

Among neuroscientists, reinforcement learning (RL) algorithms are often seen as a realistic alternative.

However, the convergence rate of such learning scales poorly with the number of involved neurons.

Here we propose a hybrid learning approach, in which each neuron uses an RL-type strategy to learn how to approximate the gradients that backpropagation would provide.

We show that our approach learns to approximate the gradient, and can match the performance of gradient-based learning on fully connected and convolutional networks.

Learning feedback weights provides a biologically plausible mechanism of achieving good performance, without the need for precise, pre-specified learning rules.

It is unknown how the brain solves the credit assignment problem when learning: how does each neuron know its role in a positive (or negative) outcome, and thus know how to change its activity to perform better next time?

Biologically plausible solutions to credit assignment include those based on reinforcement learning (RL) algorithms [4] .

In these approaches a globally distributed reward signal provides feedback to all neurons in a network.

However these methods have not been demonstrated to operate at scale.

For instance, variance in the REINFORCE estimator scales with the number of units in the network.

This drives the hypothesis that learning in the brain must rely on additional structures beyond a global reward signal.

In artificial neural networks, credit assignment is performed with gradient-based methods computed through backpropagation.

This is significantly more efficient than RL-based algorithms.

However there are well known problems with implementing backpropagation in biologically realistic neural networks.

For instance backpropagation requires a feedback structure with the same weights as the feedforward network to communicate gradients (so-called weight transport).

Yet such structures are not observed in neural circuits.

Despite this, backpropagation is the only method known to solve learning problems at scale.

Thus modifications or approximations to backpropagation that are more plausible have been the focus of significant recent attention [8, 3] .

Notably, it turns out that weight transport can be avoided by using fixed, random feedback weights, through a phenomenon called feedback alignment [8] .

However feedback alignment does not work in larger, more complicated network architectures (such as convolutional networks).

Here we propose to use an RL algorithm to train a feedback system to enable learning.

We propose to use a REINFORCE-style perturbation approach to train a feedback signal to approximate what would have been provided by backpropagation.

We demonstrate that our model learns as well as regular backpropagation in small models, overcomes the limitations of fixed random feedback weights ("feedback alignment") on more complicated feedforward networks, and can be utilized in convolutional networks.

Our method illustrates a biologically realistic way the brain could perform gradient descent-like learning.

Let an N hidden-layer network be given byŷ = f (x) ∈ R p , composed of a set of layer-wise summation and non-linear activations

, for hidden layer states h i ∈ R ni , non-linearity σ and with input h 0 = x and output h N +1 =ŷ.

Define L as the loss function L(x, y), where the data (x, y) ∈ D are drawn from a distribution ρ.

Our aim is then to minimize: E ρ [L(x, y)] .

Backpropagation computes the error signalẽ i in a top-down fashion:

Let the loss gradient term be denoted as

Here we replace λ i with an approximation, with its own parameters to be learned:

, for parameters B.

We will useẽ i to denote the gradient signal backpropagated through the synthetic gradients, and e i for the true gradients.

To estimate B we use stochasticity inherent to biological neural networks.

For each input each unit produces a noisy response:

) with standard deviation c h > 0.

This then generates a noisy lossL(x, y, ξ) and a baseline loss L(x, y) =L(x, y, 0).

We will use the noisy response to estimate gradients, that then allow us to optimize the baseline L. This is achieved by linearizing the loss:

To demonstrate the method can be used to solve simple supervised learning problems we use node perturbation with a four-layer network and MSE loss to solve MNIST (Fig. 1) .

We approximate loss gradients as follows:

Tẽi+1 .

The feedback parameters B i+1 are estimated by solving the least squares problem:

, whereλ is the perturbationbased estimator derived above.

B is updated with each mini-batch using stochastic gradient-descent to minimize this loss.

1 Updates to W i are made using the synthetic gradients ∆W i = ηẽ i h i−1 , for learning rate η.

We observed that the system is able to provide a close correspondence between the feedforward and feedback matrices in both layers of the network (Fig. 1a) .

The relative error between B i and W i is lower than what is observed for feedback alignment, suggesting that this co-adaptation of W i and B i is indeed beneficial.

We observe that the alignment (the angle between the estimated gradient and the true gradient, proportional to e T W B

) is lower for node perturbation than for feedback alignment (Fig. 1b) .

Recent studies have shown that sign congruence of the feedforward and feedback matrices is all that is required to achieve good performance [10] .

Here the sign congruence is also higher in node perturbation (Fig. 1c) .

Finally, the learning performance of node perturbation is comparable to backpropagation (Fig. 1d ) -achieving close to 3% test error.

These suggest node perturbation for learning feedback weights can be used in deep networks.

Hyperparameters found through random search.

A known shortcoming of feedback alignment is in auto-encoding networks with tight bottleneck layers [8] .

To see if our method has the same shortcoming we examine a simple auto-encoding network with MNIST input data (size 784-200-2-200-784, MSE loss).

We also compare the method to the 'matching' learning rule [9] , in which updates to B match updates to W .

As expected, feedback alignment performs poorly.

Node perturbation actually performs better than backpropagation, and comparable to ADAM (Fig. 2a) .

In fact ADAM begins to overfit, while node perturbation does not.

The matched learning rule performs similarly to backpropagation.

These results are surprising at first glance.

Perhaps, similar to feedback alignment, learning feedback weights strikes the right balance between providing a useful signal to learn, and constraining the updates to be sufficiently aligned with B, acting as a type of regularization [8] .

The noise added when estimating the feedback weights may also serve to regularize the latent representation, as, indeed, the latent space learnt by node perturbation shows a more evenly distributed separation of digits.

While, in contrast, the representations learnt by backprop and ADAM show more structure, and feedback alignment does not learn a useful representation at all (Fig. 2b,c) .

These results show that node perturbation is able to successfully communicate error signals through thin layers of a network as needed.

Finally we test the method on a convolutional neural network (CNN) solving CIFAR10.

The CNN has the architecture Conv(3x3, 1x1, 32), MaxPool(3x3, 2x2), Conv(5x5, 1x1, 128), MaxPool(3x3, 2x2), Conv(5x5, 1x1, 256), MaxPool(3x3, 2x2), FC 2048, FC 2048, Softmax(10), with hyperparameters found through random search.

For this network we learn feedback weights direct from the output layer to each earlier layer:

TẽN (similar to 'direct feedback alignment').

Here this was solved by gradient-descent.

We obtain a test accuracy of 75.2%.

When compared with fixed feedback weights (test accuracy of 72.5%) and backpropagation (test accuracy of 77.2%), we see it is advantageous to learn feedback weights.

This shows the method can be used in a CNN, and can solve challenging computer vision problems without weight transport.

Here we implement a perturbation-based synthetic gradient method to train neural networks.

We show that this hybrid approach can be used in both fully connected and convolutional networks.

By removing both the symmetric feedforward, feedback weight requirement imposed by backpropagation this approach is a step towards more biologically-plausible deep learning.

In contrast to many perturbation-based methods, this hybrid approach can solve large-scale problems.

We thus believe this approach can provide powerful and biologically plausible learning algorithms.

While previous research has provided some insight and theory for how feedback alignment works [8, 3, 2] the effect remains somewhat mysterious, and not applicable in some network architectures.

Recent studies have shown that some of these weaknesses can be addressed by instead imposing sign congruent feedforward and feedback matrices [10] .

Yet what mechanism may produce congruence in biological networks is unknown.

Here we show that the shortcomings of feedback alignment can be addressed in another way: the system can learn to adjust weights as needed to provide a useful error signal.

Our work is closely related to Akrout et al 2019 [1] , which also uses perturbations to learn feedback weights.

However our approach does not divide learning into two phases, and training of the feedback weights does not occur in a layer-wise fashion.

Here we tested our method in an idealized setting, however it is consistent with neurobiology in two important ways.

First, it involves the separate learning of feedforward and feedback weights.

This is possible in cortical networks where complex feedback connections exist between layers, and where pyramidal cells have apical and basal compartments that allow for separate integration of feedback and feedforward signals [5] .

Second, noisy perturbations are common in neural learning models.

There are many mechanisms by which noise can be measured or approximated [4, 7] , or neurons could use a learning rule that does not require knowing the noise [6] .

While our model involves the subtraction of a baseline loss to reduce the variance of the estimator, this does not affect the expected value of the estimator; technically the baseline could be removed or approximated [7] .

Thus we believe our approach could be implemented in neural circuits.

There is a large space of plausible learning rules that can learn feedback signals in order to more efficiently learn.

These promise to inform both models of learning in the brain and learning algorithms in artificial networks.

Here we take an early step in this direction.

We review the key components of the model.

Data (x, y) ∈ D are drawn from a distribution ρ.

The loss function is linearized:

such that

with expectation taken over the noise distribution ν(ξ).

This suggests a good estimator of the loss gradient iŝ

Letẽ i be the error signal computed by backpropagating the synthetic gradients:

Then parameters B i+1 are estimated by solving the least squares problem:

Under what conditions can we show thatB i+1 → W i+1 (with enough data)?

One way to find an answer is to define the synthetic gradient in terms of the system without noise added.

Then B Tẽ is deterministic with respect to x, y and, assumingL has a convergent power series around ξ = 0, we can write

Taken together these suggest we can proveB i+1 → W i+1 in the same way we prove consistency of the linear least squares estimator.

For this to work we must show the expectation of the Taylor series approximation (2) is well behaved.

That is, we must show the expected remainder term of the expansion:

is finite and goes to zero as c h → 0.

This requires some additional assumptions on the problem.

We make the following assumptions:

• A1:

the noise ξ is subgaussian,

• A3: the error matricesẽ n (ẽ n ) T are full rank, for 1 ≤ n ≤ N + 1,

• A4: the mean of the remainder and error terms is bounded:

Consider first convergence of the final layer feedback matrix, B N +1 .

In the final layer it is true that e N +1 = e N +1 .

, then the least squares estimator

solves (4) and converges to the true feedback matrix, in the sense that:

.

We first show that, under A1-2, the conditional expectation of the estimator (5) converges to the gradient L

Taking a conditional expectation gives:

We must show the remainder term

goes to zero as c h → 0.

This is true provided each moment E((ξ N j ) m |x, y) is sufficiently well-behaved.

Using Jensen's inequality and the triangle inequality in the first line, we have that

With this in place, we have that the problem (4) is close to a linear least squares problem, sincê

with residual η

This follows since e N +1 is defined in relation to the baseline loss, not the stochastic loss, meaning it is measurable with respect to (x, y) and can be moved into the conditional expectation.

From (7) and A3, we have that the least squares estimator (5) satisfies

Thus, using the continuous mapping theorem

Then we have: lim

We can use Theorem 1 to establish convergence over the rest of the layers of the network when the activation function is the identity.

and σ(x) = x, the least squares estimator

solves (4) and converges to the true feedback matrix, in the sense that:

Proof.

DefineW n (c) := plim T →∞B n , assuming this limit exists.

From Theorem 1 the top layer estimateB N +1 converges in probability toW N +1 (c).

We can then use induction to establish thatB j in the remaining layers also converges in probability toW j (c).

That is, assume thatB j converge in probability toW j (c) in higher layers N + 1 ≥ j >

n.

Then we must establish thatB n also converges in probability.

To proceed it is useful to also definẽ

as the error signal backpropagated through the converged (but biased) weight matricesW (c).

Again it is true thatẽ N +1 = e N +1 .

As in Theorem 1, the least squares estimator has the form:

Thus, again by the continuous mapping theorem:

In this case continuity again allows us to separate convergence of each term in the product:

using the weak law of large numbers in the first term, and the induction assumption for the remaining terms.

In the same way

Note that the induction assumption also implies limc→0ẽ n (c) = e n .

Thus, putting it together, by A3, A4 and the same reasoning as in Theorem 1 we have the result:

solves (4) and converges to the true feedback matrix, in the sense that:

Proof.

For a deep linear network notice that the node perturbation estimator can be expressed as:

where the first term represents the true gradient, given by the simple linear backpropagation, the second and third terms are the remainder and a noise term, as in Theorem 1.

Define

Wj.

Then following the same reasoning as the proof of Theorem 1, we have:

Then we have: lim

It is worth making the following points on each of the assumptions:

• A1.

In the paper we assume ξ is Gaussian.

Here we prove the more general result of convergence for any subgaussian random variable.

• A2.

In practice this may be a fairly restrictive assumption, since it precludes using relu non-linearities.

Other common choices, such as hyperbolic tangent and sigmoid non-linearities with an analytic cost function do satisfy this assumption, however.

• A3.

It is hard to establish general conditions under whichẽ n (ẽ n ) T will be full rank.

While it may be a reasonable assumption in some cases.

Extensions of Theorem 2 to a non-linear network may be possible.

However, the method of proof used here is not immediately applicable because the continuous mapping theorem can not be applied in such a straightforward fashion as in Equation (10) .

In the non-linear case the resulting sums over all observations are neither independent or identically distributed, which makes applying any law of large numbers complicated.

@highlight

Perturbations can be used to learn feedback weights on large fully connected and convolutional networks.