Backpropagation is driving today's artificial neural networks (ANNs).

However, despite extensive research, it remains unclear if the brain implements this algorithm.

Among neuroscientists, reinforcement learning (RL) algorithms are often seen as a realistic alternative: neurons can randomly introduce change, and use unspecific feedback signals to observe their effect on the cost and thus approximate their gradient.

However, the convergence rate of such learning scales poorly with the number of involved neurons.

Here we propose a hybrid learning approach.

Each neuron uses an RL-type strategy to learn how to approximate the gradients that backpropagation would provide.

We provide proof that our approach converges to the true gradient for certain classes of networks.

In both feedforward and convolutional networks, we empirically show that our approach learns to approximate the gradient, and can match the performance of gradient-based learning.

Learning feedback weights provides a biologically plausible mechanism of achieving good performance, without the need for precise, pre-specified learning rules.

It is unknown how the brain solves the credit assignment problem when learning: how does each neuron know its role in a positive (or negative) outcome, and thus know how to change its activity to perform better next time?

This is a challenge for models of learning in the brain.

Biologically plausible solutions to credit assignment include those based on reinforcement learning (RL) algorithms and reward-modulated STDP (Bouvier et al., 2016; Fiete et al., 2007; Legenstein et al., 2010; Miconi, 2017) .

In these approaches a globally distributed reward signal provides feedback to all neurons in a network.

Essentially, changes in rewards from a baseline, or expected, level are correlated with noise in neural activity, allowing a stochastic approximation of the gradient to be computed.

However these methods have not been demonstrated to operate at scale.

For instance, variance in the REINFORCE estimator (Williams, 1992) scales with the number of units in the network (Rezende et al., 2014) .

This drives the hypothesis that learning in the brain must rely on additional structures beyond a global reward signal.

In artificial neural networks (ANNs), credit assignment is performed with gradient-based methods computed through backpropagation (Rumelhart et al., 1986; Werbos, 1982; Linnainmaa, 1976) .

This is significantly more efficient than RL-based algorithms, with ANNs now matching or surpassing human-level performance in a number of domains (Mnih et al., 2015; Silver et al., 2017; LeCun et al., 2015; He et al., 2015; Haenssle et al., 2018; Russakovsky et al., 2015) .

However there are well known problems with implementing backpropagation in biologically realistic neural networks.

One problem is known as weight transport (Grossberg, 1987) : an exact implementation of backpropagation requires a feedback structure with the same weights as the feedforward network to communicate gradients.

Such a symmetric feedback structure has not been observed in biological neural circuits.

Despite such issues, backpropagation is the only method known to solve supervised and reinforcement learning problems at scale.

Thus modifications or approximations to backpropagation that are more plausible have been the focus of significant recent attention (Scellier & Bengio, 2016; Lillicrap et al., 2016; Lee et al., 2015; Lansdell & Kording, 2018) .

These efforts do show some ways forward.

Synthetic gradients demonstrate that learning can be based on approximate gradients, and need not be temporally locked (Jaderberg et al., 2016; Czar-necki et al., 2017b) .

In small feedforward networks, somewhat surprisingly, fixed random feedback matrices in fact suffice for learning (Lillicrap et al., 2016 ) (a phenomenon known as feedback alignment).

But still issues remain: feedback alignment does not work in CNNs, very deep networks, or networks with tight bottleneck layers.

Regardless, these results show that rough approximations of a gradient signal can be used to learn; even relatively inefficient methods of approximating the gradient may be good enough.

On this basis, here we propose an RL algorithm to train a feedback system to enable learning.

Recent work has explored similar ideas, but not with the explicit goal of approximating backpropagation (Miconi, 2017; Miconi et al., 2018; Song et al., 2017) .

RL-based methods like REINFORCE may be inefficient when used as a base learner, but they may be sufficient when used to train a system that itself instructs a base learner.

We propose to use REINFORCE-style perturbation approach to train feedback signals to approximate what would have been provided by backpropagation.

This sort of two-learner system, where one network helps the other learn more efficiently, may in fact align well with cortical neuron physiology.

For instance, the dendritic trees of pyramidal neurons consist of an apical and basal component.

Such a setup has been shown to support supervised learning in feedforward networks (Guergiuev et al., 2017; Kording & Konig, 2001) .

Similarly, climbing fibers and Purkinje cells may define a learner/teacher system in the cerebellum (Marr, 1969) .

These components allow for independent integration of two different signals, and may thus provide a realistic solution to the credit assignment problem.

Thus we implement a network that learns to use feedback signals trained with reinforcement learning via a global reward signal.

We mathematically analyze the model, and compare its capabilities to other methods for learning in ANNs.

We prove consistency of the estimator in particular cases, extending the theory of synthetic gradient-like approaches (Jaderberg et al., 2016; Czarnecki et al., 2017b; Werbos, 1992; Schmidhuber, 1990) .

We demonstrate that our model learns as well as regular backpropagation in small models, overcomes the limitations of feedback alignment on more complicated feedforward networks, and can be used in convolutional networks.

Thus, by combining local and global feedback signals, this method points to more plausible ways the brain could solve the credit assignment problem.

We use the following notation.

Let x ∈ R m represent an input vector.

Let an N hidden-layer network be given byŷ = f (x) ∈ R p .

This is composed of a set of layer-wise summation and non-linear activations

for hidden layer states h i ∈ R ni , non-linearity σ, weight matrices W i ∈ R ni×ni−1 and denoting h 0 = x and h N +1 =ŷ.

Some loss function L is defined in terms of the network output: L(y,ŷ).

Let L denote the loss as a function of (x, y):

Backpropagation relies on the error signal e i , computed in a top-down fashion:

where • denotes element-wise multiplication.

Let the loss gradient term be denoted as

In this work we replace λ i with an approximation with its own parameters to be learned (known as a synthetic gradient, or conspiring network, (Jaderberg et al., 2016; Czarnecki et al., 2017b) , or error critic (Werbos, 1992) ): Node perturbation introduces noise in each layer, ξ i , that perturbs that layer's output and resulting loss function.

The perturbed loss function,L, is correlated with the noise to give an estimate of the error current.

This estimate is used to update feedback matrices B i to better approximate the error signal.

for parameters θ.

Note that we must distinguish the true loss gradients from their synthetic estimates.

Letẽ i be loss gradients computed by backpropagating the synthetic gradients

For the final layer the synthetic gradient matches the true gradient: e N +1 =ẽ N +1 .

This setup can accommodate both top-down and bottom-up information, and encompasses a number of published models (Jaderberg et al., 2016; Czarnecki et al., 2017b; Lillicrap et al., 2016; Nøkland, 2016; Liao et al., 2016; Xiao et al., 2018) .

To learn a synthetic gradient we utilze the stochasticity inherent to biological neural networks.

A number of biologically plausible learning rules exploit random perturbations in neural activity (Xie & Seung, 2004; Seung, 2003; Fiete et al., 2007; Song et al., 2017) .

Here, at each time each unit produces a noisy response:

) and standard deviation c h > 0.

This generates a noisy lossL(x, y, ξ) and a baseline loss L(x, y) =L(x, y, 0).

We will use the noisy response to estimate gradients that then allow us to optimize the baseline L -the gradients used for weight updates are computed using the deterministic baseline.

For Gaussian white noise, the well-known REINFORCE algorithm (Williams, 1992) coincides with the node perturbation method Fiete et al., 2007) .

Node perturbation works by linearizing the loss:L

such that

with expectation taken over the noise distribution ν(ξ).

This provides an estimator of the loss gradi-

This approximation is made more precise in Theorem 1 (Supplementary material).

There are many possible sensible choices of g(·).

For example, taking g as simply a function of each layer's activations:

) is in fact sufficient parameterization to express the true gradient function (Jaderberg et al., 2016) .

We may expect, however, that the gradient estimation problem be simpler if each layer is provided with some error information obtained from the loss function and propagated in a top-down fashion.

Symmetric feedback weights may not be biologically plausible, and random fixed weights may only solve certain problems of limited size or complexity (Lillicrap et al., 2016) .

However, a system that can learn to appropriate feedback weights B may be able to align the feedforward and feedback weights as much as is needed to successfully learn.

We investigate various choices of g(h i ,ẽ i+1 ; B i+1 ) outlined in the applications below.

Parameters B i+1 are estimated by solving the least squares problem:

Unless otherwise noted this was solved by gradient-descent, updating parameters once with each minibatch.

Refer to the supplementary material for additional experimental descriptions and parameters.

We can prove the estimator (3) is consistent as the noise variance c h → 0, in some particular cases.

We state the results informally here, and give the exact details in the supplementary materials.

Consider first convergence of the final layer feedback matrix, B N +1 .

, then the least squares estimator

solves (3) and converges to the true feedback matrix, in the sense that:

, where plim indicates convergence in probability.

Theorem 1 thus establishes convergence of B in a shallow (1 hidden layer) non-linear network.

In a deep, linear network we can also use Theorem 1 to establish convergence over the rest of the layers.

solves (3) and converges to the true feedback matrix, in the sense that:

Given these results we can establish consistency for the 'direct feedback alignment' (DFA; Nøkland (2016)) estimator:

.

Theorem 1 applies trivially since for the final layer, the two approximations have the same form:

Theorem 2 can be easily extended according to the following:

solves (3) and converges to the true feedback matrix, in the sense that: Thus for a non-linear shallow network or a deep linear network, for both g F A and g DF A , we have the result that, for sufficiently small c h , if we fix the network weights W and train B through node perturbation then we converge to W .

Validation that the method learns to approximate W , for fixed W , is provided in the supplementary material.

In practice, we update B and W simultaneously.

Some convergence theory is established for this case in (Jaderberg et al., 2016; Czarnecki et al., 2017b) .

Tẽi+1 , which describes a non-symmetric feedback network ( Figure 1 ).

To demonstrate the method can be used to solve simple supervised learning problems we use node perturbation with a four-layer network and MSE loss to solve MNIST ( Figure  2 ).

Updates to W i are made using the synthetic gradients ∆W i = ηẽ i h i−1 , for learning rate η.

The feedback network needs to co-adapt with the feedforward network in order to continue to provide a useful error signal.

We observed that the system is able to adjust to provide a close correspondence between the feedforward and feedback matrices in both layers of the network ( Figure 2A ).

The relative error between B i and W i is lower than what is observed for feedback alignment, suggesting that this co-adaptation of both W i and B i is indeed beneficial.

The relative error depends on the amount of noise used in node perturbation -lower variance doesn't necessarily imply the lowest error between W and B, suggesting there is an optimal noise level that balances bias in the estimate and the ability to co-adapt to the changing feedforward weights.

Consistent with the low relative error in both layers, we observe that the alignment (the angle between the estimated gradient and the true gradient -proportional to e T W B

) is low in each layer -much lower for node perturbation than for feedback alignment, again suggesting that the method is much better at communicating error signals between layers ( Figure 2B ).

In fact, recent studies have shown that sign congruence of the feedforward and feedback matrices is all that is required to achieve good performance (Liao et al., 2016; Xiao et al., 2018) .

Here the sign congruence is also higher in node perturbation, again depending somewhat the variance.

The amount of congruence is comparable between layers ( Figure 2C ).

Finally, the learning performance of node perturbation is comparable to backpropagation ( Figure 2D ), and better than feedback alignment in this case, though not by much.

These results instead highlight the qualitative differences between the methods, and suggest that node perturbation for learning feedback weights can be used to approximate gradients in deep networks.

The above results demonstrate node perturbation provides error signals closely aligned with the true gradients.

However, performance-wise they do not demonstrate any clear advantage over feedback alignment or backpropagation.

A known shortcoming of feedback alignment is in very deep networks and in autoencoding networks with tight bottleneck layers (Lillicrap et al., 2016) .

To see if node perturbation has the same shortcoming, we test performance of a g(

Tẽi+1 model on a simple auto-encoding network with MNIST input data (size 784-200-2-200-784).

In this more challenging case we also compare the method to the 'matching' learning rule (Rombouts et al., 2015; Martinolli et al., 2018) , in which updates to B match updates to W and weight decay is added, a denoising autoencoder (DAE) (Vincent et al., 2008) , and the ADAM (Kingma & Ba, 2015) optimizer (with backprop gradients).

As expected, feedback alignment performs poorly, while node perturbation performs better than backpropagation ( Figure 3A) .

The increased performance relative to backpropagation may seem surprising.

A possible reason is the addition of noise in our method encourages learning of more robust latent factors (Alain & Bengio, 2015) .

The DAE also improves the loss over vanilla backpropagation ( Figure 3A) .

And, in line with these ideas, the latent space learnt by node perturbation shows a more uniform separation between the digits, compared to the networks trained by backpropagation.

Feedback alignment, in contrast, does not learn to separate digits in the bottleneck layer at all ( Figure 3B ), resulting in scrambled output ( Figure 3C ).

The matched learning rule performs similarly to backpropagation.

These possible explanations are investigated more below.

Regardless, these results show that node perturbation is able to successfully communicate error signals through thin layers of a network as needed.

Convolutional networks are another known shortcoming of feedback alignment.

Here we test the method on a convolutional neural network (CNN) solving CIFAR (Krizhevsky, 2009 ).

Refer to the supplementary material for architecture and parameter details.

For this network we learn feedback weights direct from the output layer to each earlier layer:

TẽN +1 (similar to 'direct feedback alignment' (Nøkland, 2016) ).

Here this was solved by gradient-descent.

On CIFAR10 we obtain a test accuracy of 75%.

When compared with fixed feedback weights and backpropagation, we see it is advantageous to learn feedback weights on CIFAR10 and marginally advantageous on CIFAR100 (Table 1) .

This shows the method can be used in a CNN, and can solve challenging computer vision problems without weight transport.

To solve the credit assignment problem, our method utilizes two well-explored strategies in deep learning: adding noise (generally used to regularize (Bengio et al., 2013; Gulcehre et al., 2016; Neelakantan et al., 2015; Bishop, 1995) ), and approximating the true gradients (Jaderberg et al., Table 1 :

Mean test accuracy of CNN over 5 runs trained with backpropagation, node perturbation and direct feedback alignment (DFA) (Nøkland, 2016; Crafton et al., 2019 .

To determine which of these features are responsible for the improvement in performance over fixed weights, in the autoencoding and CIFAR10 cases, we study the performance while varying where noise is added to the models (Table 2 ).

Noise can be added to the activations (BP and FA w. noise, Table 2 ), or to the inputs, as in a denoising autoencoder (DAE, Table 2 ).

Or, noise can be used only in obtaining an estimator of the true gradients (as in our method; NP, Table 2 ).

For comparison, a noiseless version of our method must instead assume access to the true gradients, and use this to learn feedback weights (i.e. synthetic gradients (Jaderberg et al., 2016) ; SG, Table 2 ).

Each of these models is tested on the autoencoding and CIFAR10 tasks, allowing us to better understand the performance of the node perturbation method.

In the autoencoding task, both noise (either in the inputs or the activations) and using an approximator to the gradient improve performance (Table 2 , left).

Noise benefits performance for both SGD optimization and ADAM (Kingma & Ba, 2015) .

In fact in this task, the combination of both of these factors (i.e. our method) results in better performance over either alone.

Yet, the addition of noise to the activations does not help feedback alignment.

This suggests that our method is indeed learning useful approximations of the error signals, and is not merely improving due to the addition of noise to the system.

In the CIFAR10 task (Table 2 , right), the addition of noise to the activations has minimal effect on performance, while having access to the true gradients (SG) does result in improved performance over fixed feedback weights.

Thus in these tasks it appears that noise does not always help, but using a less-based gradient estimator does, and noisy activations are one way of obtaining an unbiased gradient estimator.

Our method also is the best performing method that does not require either weight transport or access to the true gradients as a supervisory signal.

Here we implement a perturbation-based synthetic gradient method to train neural networks.

We show that this hybrid approach can be used in both fully connected and convolutional networks.

By removing the symmetric feedforward/feedback weight requirement imposed by backpropagation, this approach is a step towards more biologically-plausible deep learning.

By reaching comparable performance to backpropagation on MNIST, the method is able to solve larger problems than perturbation-only methods (Xie & Seung, 2004; Fiete et al., 2007; Werfel et al., 2005) .

By working in cases that feedback alignment fails, the method can provide learning without weight transport in a more diverse set of network architectures.

We thus believe the idea of integrating both local and global feedback signals is a promising direction towards biologically plausible learning algorithms.

Of course, the method does not solve all issues with implementing gradient-based learning in a biologically plausible manner.

For instance, in the current implementation, the forward and the backwards passes are locked.

Here we just focus on the weight transport problem.

A current drawback is that the method does not reach state-of-the-art performance on more challenging datasets like CIFAR.

We focused on demonstrating that it is advantageous to learn feedback weights, when compared with fixed weights, and successfully did so in a number of cases.

However, we did not use any additional data augmentation and regularization methods often employed to reach state-of-theart performance.

Thus fully characterizing the performance of this method remains important future work.

However the method does has a number of computational advantages.

First, without weight transport the method has better data-movement performance (Crafton et al., 2019; Akrout et al., 2019) , meaning it may be more efficiently implemented than backpropagation on specialized hardware.

Second, by relying on random perturbations to measure gradients, the method does not rely on the environment to provide gradients (compared with e.g. Czarnecki et al. (2017a) ; Jaderberg et al. (2016) ).

Our theoretical results are somewhat similar to that of Alain & Bengio (2015) , who demonstrate that a denoising autoencoder converges to the unperturbed solution as Gaussian noise goes to zero.

However our results apply to subgaussian noise more generally.

While previous research has provided some insight and theory for how feedback alignment works (Lillicrap et al., 2016; Ororbia et al., 2018; Moskovitz et al., 2018; Bartunov et al., 2018; Baldi et al., 2018 ) the effect remains somewhat mysterious, and not applicable in some network architectures.

Recent studies have shown that some of these weaknesses can be addressed by instead imposing sign congruent feedforward and feedback matrices (Xiao et al., 2018 ).

Yet what mechanism may produce congruence in biological networks is unknown.

Here we show that the shortcomings of feedback alignment can be addressed in another way: the system can learn to adjust weights as needed to provide a useful error signal.

Our work is closely related to Akrout et al. (2019) , which also uses perturbations to learn feedback weights.

However our approach does not divide learning into two phases, and training of the feedback weights does not occur in a layer-wise fashion, assuming only one layer is noisy at a time, which is a strong assumption.

Here instead we focus on combining global and local learning signals.

Here we tested our method in an idealized setting.

However the method is consistent with neurobiology in two important ways.

First, it involves separate learning of feedforward and feedback weights.

This is possible in cortical networks, where complex feedback connections exist between layers (Lacefield et al., 2019; Richards & Lillicrap, 2019) and pyramidal cells have apical and basal compartments that allow for separate integration of feedback and feedforward signals (Guerguiev et al., 2017; Körding & König, 2001) .

A recent finding that apical dendrites receive reward information is particularly interesting (Lacefield et al., 2019) .

Models like Guerguiev et al. (2017) show how the ideas in this paper may be implemented in spiking neural networks.

We believe such models can be augmented with a perturbation-based rule like ours to provide a better learning system.

The second feature is that perturbations are used to learn the feedback weights.

How can a neuron measure these perturbations?

There are many plausible mechanisms (Seung, 2003; Xie & Seung, 2004; Fiete et al., 2007) .

For instance, birdsong learning uses empiric synapses from area LMAN (Fiete et al., 2007) , others proposed it is approximated (Legenstein et al., 2010; Hoerzer et al., 2014) , or neurons could use a learning rule that does not require knowing the noise (Lansdell & Kording, 2018) .

Further, our model involves the subtraction of a baseline loss to reduce the variance of the estimator.

This does not affect the expected value of the estimator -technically the baseline could be removed or replaced with an approximation (Legenstein et al., 2010; Loewenstein & Seung, 2006) .

Thus both separation of feedforward and feedback systems and perturbation-based estimators can be implemented by neurons.

As RL-based methods do not scale by themselves, and exact gradient signals are infeasible, the brain may well use a feedback system trained through reinforcement signals to usefully approximate gradients.

There is a large space of plausible learning rules that can learn to use feedback signals in order to more efficiently learn, and these promise to inform both models of learning in the brain and learning algorithms in artificial networks.

Here we take an early step in this direction.

We review the key components of the model.

Data (x, y) ∈ D are drawn from a distribution ρ.

The loss function is linearized:

such that

with expectation taken over the noise distribution ν(ξ).

This suggests a good estimator of the loss gradient isλ

Letẽ i be the error signal computed by backpropagating the synthetic gradients:

Then parameters B i+1 are estimated by solving the least squares problem:

Note that the matrix-vector form of backpropagation given here is setup so that we can think of each term as either a vector for a single input, or as matrices corresponding to a set of T inputs.

Here we focus on the question, under what conditions can we show thatB

One way to find an answer is to define the synthetic gradient in terms of the system without noise added.

Then B Tẽ is deterministic with respect to x, y and, assumingL has a convergent power series around ξ = 0, we can write

Taken together these suggest we can proveB i+1 → W i+1 in the same way we prove consistency of the linear least squares estimator.

For this to work we must show the expectation of the Taylor series approximation (1) is well behaved.

That is, we must show the expected remainder term of the expansion:

is finite and goes to zero as c h → 0.

This requires some additional assumptions on the problem.

We make the following assumptions:

• A1: the noise ξ is subgaussian, • A2: the loss function L(x, y) is analytic on D,

• A3:

the error matricesẽ i (ẽ i ) T are full rank, for 1 ≤ i ≤ N + 1, with probability 1,

• A4: the mean of the remainder and error terms is bounded:

Consider first convergence of the final layer feedback matrix, B N +1 .

In the final layer it is true that e N +1 =ẽ N +1 .

, then the least squares estimator

solves (3) and converges to the true feedback matrix, in the sense that:

.

We first show that, under A1-2, the conditional expectation of the estimator (2) converges to the gradient L

Taking a conditional expectation gives:

We must show the remainder term

goes to zero as c h → 0.

This is true provided each moment E((ξ N j ) m |x, y) is sufficiently wellbehaved.

Using Jensen's inequality and the triangle inequality in the first line, we have that

With this in place, we have that the problem (9) is close to a linear least squares problem, sincê

This follows since e N +1 is defined in relation to the baseline loss, not the stochastic loss, meaning it is measurable with respect to (x, y) and can be moved into the conditional expectation.

From (12) and A3, we have that the least squares estimator (10) satisfies

Thus, using the continuous mapping theorem

Then we have: lim

We can use Theorem 1 to establish convergence over the rest of the layers of the network when the activation function is the identity.

and σ(x) = x, the least squares estimator

solves (9) and converges to the true feedback matrix, in the sense that:

Proof.

DefineW i (c) := plim

assuming this limit exists.

From Theorem 1 the top layer estimateB N +1 converges in probability toW N +1 (c).

We can then use induction to establish thatB j in the remaining layers also converges in probability toW j (c).

That is, assume thatB j converge in probability toW j (c) in higher layers N + 1 ≥ j >

i.

Then we must establish thatB i also converges in probability.

To proceed it is useful to also definẽ

as the error signal backpropagated through the converged (but biased) weight matricesW (c).

Again it is true thatẽ N +1 = e N +1 .

As in Theorem 1, the least squares estimator has the form:

Thus, again by the continuous mapping theorem:

In this case continuity again allows us to separate convergence of each term in the product:

using the weak law of large numbers in the first term, and the induction assumption for the remaining terms.

In the same way

Note that the induction assumption also implies lim c→0ẽ i (c) = e i .

Thus, putting it together, by A3, A4 and the same reasoning as in Theorem 1 we have the result:

+1 and σ(x) = x, the least squares estimator

solves (3) and converges to the true feedback matrix, in the sense that:

Proof.

For a deep linear network notice that the node perturbation estimator can be expressed as:

where the first term represents the true gradient, given by the simple linear backpropagation, the second and third terms are the remainder and a noise term, as in Theorem 1.

Define

Then following the same reasoning as the proof of Theorem 1, we have:

Then we have: lim

It is worth making the following points on each of the assumptions:

• A1.

In the paper we assume ξ is Gaussian.

Here we prove the more general result of convergence for any subgaussian random variable.

• A2.

In practice this may be a fairly restrictive assumption, since it precludes using relu nonlinearities.

Other common choices, such as hyperbolic tangent and sigmoid non-linearities with an analytic cost function do satisfy this assumption, however.

• A3.

It is hard to establish general conditions under whichẽ i (ẽ i ) T will be full rank.

While it may be a reasonable assumption in some cases.

Extensions of Theorem 2 to a non-linear network may be possible.

However, the method of proof used here is not immediately applicable because the continuous mapping theorem can not be applied in such a straightforward fashion as in Equation (15).

In the non-linear case the resulting sums over all observations are neither independent or identically distributed, which makes applying any law of large numbers complicated.

We demonstrate the method's convergence in a small non-linear network solving MNIST for different noise levels, c h , and layer widths ( Figure 4) .

As basic validation of the method, in this experiment the feedback matrices are updated while the feedforward weights W i are held fixed.

We should expect the feedback matrices B i to converge to the feedforward matrices W i .

Here different noise variance does results equally accurate estimators ( Figure 4A ).

The estimator correctly estimates the true feedback matrix W 2 to a relative error of 0.8%.

The convergence is layer dependent, with the second hidden layer matrix, W 2 , being accurately estimated, and the convergence of the first hidden layer matrix, W 1 , being less accurately estimated.

Despite this, the angles between the estimated gradient and the true gradient (proportional to e T W B

) are very close to zero for both layers ( Figure 4B ) (less than 90 degrees corresponds to a descent direction).

Thus the estimated gradients strongly align with true gradients in both layers.

Recent studies have shown that sign congruence of the feedforward and feedback matrices is all that is required to achieve good performance Liao et al. (2016) ; Xiao et al. (2018) .

Here significant sign congruence is achieved in both layers ( Figure 4C ), despite the matrices themselves being quite different in the first layer.

The number of neurons has an effect on both the relative error in each layer and the extent of alignment between true and synthetic gradient ( Figure 4D,E) .

The method provides useful error signals for a variety of sized networks, and can provide useful error information to layers through a deep network.

Details of each task and parameters are provided here.

All code is implemented in TensorFlow.

Networks are 784-50-20-10 with an MSE loss function.

A sigmoid non-linearity is used.

A batch size of 32 is used.

B is updated using synthetic gradient updates with learning rate η = 0.0005, W is updated with learning rate 0.0004, standard deviation of noise is 0.01.

Same step size is used for feedback alignment, backpropagation and node perturbation.

An initial warm-up period of 1000 iterations is used, in which the feedforward weights are frozen but the feedback weights are adjusted.

Network has dimensions 784-200-2-200-784.

Activation functions are, in order: tanh, identity, tanh, relu.

MNIST input data with MSE reconstruction loss is used.

A batch size of 32 was used.

In this case stochastic gradient descent was used to update B. Values for W step size, noise variance and B step size were found by random hyperparameter search for each method.

The denoising autoencoder used Gaussian noise with zero mean and standard deviation σ = 0.3 added to the input training data.

Networks are 784-50-20-10 (noise variance) or 784-N-50-10 (number of neurons) solving MNIST with an MSE loss function.

A sigmoid non-linearity is used.

A batch size of 32 is used.

Here W is fixed, and B is updated according to an online ridge regression least-squares solution.

This was used becase it converges faster than the gradient-descent based optimization used for learning B throughout the rest of the text, so is a better test of consistency.

A regularization parameter of γ = 0.1 was used for the ridge regression.

That is, for each update, B i was set to the exact solution of the following:B i+1 = arg min

Code and CNN architecture are based on the direct feedback alignment implementation of Crafton et al. (2019) .

Specifically, for both CIFAR10 and CIFAR100, the CNN has the architecture Conv(3x3, 1x1, 32), MaxPool(3x3, 2x2), Conv(5x5, 1x1, 128), MaxPool(3x3, 2x2), Conv(5x5, 1x1, 256), MaxPool(3x3, 2x2), FC 2048 , FC 2048 .

Hyperparameters (learning rate, feedback learning rate, and perturbation noise level) were found through random search.

All other parameters are the same as Crafton et al. (2019) .

In particular, ADAM optimizer was used, and dropout with probability 0.5 was used.

The methods listed in Table 2 are implemented as follows.

For the autoencoding task: Through hyperparameter search, a noise standard deviation of c * h = 0.02 was found to give optimal performance for our method.

For BP(SGD), BP(ADAM), FA, the 'noise' results in the Table are obtained by adding zero-mean Gaussian noise to the activations with the same standard deviation, c * h .

For the DAE, a noise standard deviation of c i = 0.3 was added to the inputs of the network.

Implementation of the synthetic gradient method here takes the same form as our method: g(h, e, y; B) = Be (this contrasts with the form used in Jaderberg et al. (2016) : g(h, e, y; B, c) = B T h + c).

But the matrices B are trained by providing true gradients λ, instead of noisy estimators based on node perturbation.

This is not biologically plausible, but provides a useful baseline to determine the source of good performance.

The other co-adapting baseline we investigate is the 'matching' rule (similar to (Akrout et al., 2019; Rombouts et al., 2015; Martinolli et al., 2018) ): the updates to B match those of W , and weight decay is used to drive the feedforward and feedback matrices to be similar.

For the CIFAR10 results, our hyperparameter search identified a noise standard deviation of c h = 0.067 to be optimal.

This was added to the activations .

The synthetic gradients took the same form as above.

@highlight

Perturbations can be used to train feedback weights to learn in fully connected and convolutional neural networks

@highlight

This paper proposes a method that addresses the "weight transport" problem by estimating the weights for the backward pass using a noise-based estimator 