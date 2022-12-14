Artificial neural networks revolutionized many areas of computer science in recent years since they provide solutions to a number of previously unsolved problems.

On the other hand, for many problems, classic algorithms exist, which typically exceed the accuracy and stability of neural networks.

To combine these two concepts, we present a new kind of neural networks—algorithmic neural networks (AlgoNets).

These networks integrate smooth versions of classic algorithms into the topology of neural networks.

Our novel reconstructive adversarial network (RAN) enables solving inverse problems without or with only weak supervision.

Artificial Neural Networks are employed to solve numerous problems, not only in computer science but also in all other natural sciences.

Yet, the reasoning for the topologies of neural networks seldom reaches beyond empirically-based decisions.

In this work, we present a novel approach to designing neural networks-algorithmic neural networks (short: AlgoNet).

Such networks integrate algorithms as algorithmic layers into the topology of neural networks.

However, propagating gradients through such algorithms is problematic, because crisp decisions (conditions, maximum, etc.) introduce discontinuities into the loss function.

If one passes from one side of a crisp decision to the other, the loss function may change in a non-smooth fashion-it may "jump."

That is, the loss function suddenly improves (or worsens, depending on the direction) without these changes being locally noticeable anywhere but exactly at these "jumps."

Hence, a gradient descent based training, regardless of the concrete optimizer, cannot approach these "jumps" in a systematic fashion, since neither the loss function nor the gradient provides any information about these "jumps" in any place other than exactly the location at which they occur.

Therefore, a smoothing is necessary, such that information about the direction of improvement becomes exploitable by gradient descent also in the area surrounding the "jump."

That is, by smoothing, e.g., an if, one can smoothly, by gradient descent, undergo a transition between the two crisp cases using only local gradient information.

Generally, for end-to-end trainable neural network systems, all components should at least be C 0 smooth, i.e., continuous, to avoid "jumps."

However, having C k smooth, i.e., k times differentiable and then still continuous components with k ≥ 1 is favorable.

This property of higher smoothness allows for higher-order derivatives and thus prevents unexpected behavior of the gradients.

Hence, we designed smooth approximations to basic algorithms where the functions representing the algorithms are ideally C ∞ smooth.

That is, we designed pre-programmed neural networks (restricted to smooth components) with the structure of given algorithms.

Related work [1] - [3] in neural networks focused on dealing with crisp decisions by passing through gradients for the alternatives of the decisions.

There is no smooth transition between the alternatives, which introduces discontinuities in the loss function that hinder learning, which of the alternatives should be chosen.

TensorFlow contains a sorting layer (tf.sort) as well as a while loop construct (tf.while_loop).

Since the sorting layer only performs a crisp relocation of the gradients and the while loop has a crisp exit condition, there is no gradient with respect to the conditions in these layers.

Concurrently, we developed a smooth sorting layer and a smooth while loop.

Theoretical work by DeMillo et al. [4] proved that any program could be modeled by a smooth function.

Consecutive works [5] - [7] provided approaches for smoothing programs using, i.a., Gaussian smoothing [6] , [7] .

Algorithmic layers, i.e., smooth approximations, exist for any Turing computable algorithm [8] .

To design a smooth algorithmic layer, all discrete cases (e.g., conditions of if statements or loops) have to be replaced by continuous or smooth functions.

The essential property is that the implementation is differentiable with respect to all internal choices and does not-as in previous work-only carry the gradients through the algorithm [1] .

For example, an if statement can be replaced by a sigmoid-weighted sum of both cases.

By using a smooth sigmoid function, the statement is smoothly interpreted.

Hence, the gradient descent method can influence the condition to hold if the content of the then case reduces the loss and influence the condition to fail if the loss is lower when the else case is executed.

Thus, the partial derivative with respect to a neuron is computed because the neuron is used in the if statement.

In contrast, when propagating back the gradient of the then or the else case depending on the value of the condition, there is a discontinuity at the points where the value of the condition changes and the partial derivative of the neuron in the condition equals zero.

The logistic sigmoid function (Eq. 1) is a C ∞ smooth replacement for the Heaviside sigmoid function (Eq. 2), which is equivalent to the if statement.

Alternatively, one could use other sigmoid functions, e.g., the C 1 smooth step function x 2 − 2 · x 3 for x ∈ [0, 1], and 0 and 1 for all values before and after the given range, respectively.

After designing an algorithmic layer, we can use it to solve for its inverse by using the Reconstructive Adversarial Neural Network (RAN).

Reconstructive Adversarial Networks (RAN) use an algorithm that solves for the inverse of a given problem.

For example, they use a smooth renderer for 3D-reconstruction, a smooth iterated function system (IFS) for solving the inverse-problem of IFS, and a smooth text-to-speech synthesizer for speech recognition.

While RANs could be used in supervised settings, they are designed for unsupervised or weakly supervised solving of inverse-problems.

Their concept is the following:

This structure is similar to auto-encoders and the encoder-renderer architecture presented by Che et al. [2] .

Such an architecture, however, cannot directly be trained since there is a domain shift between the input domain A and the smooth output domain B. Thus, we introduce domain translators (a2b and b2a) to translate between these two domains.

Since training is extremely hard with three consecutive components, of which the middle one is highly restrictive, we use the RAN as a novel training schema for these components.

For that, we also include a discriminator to allow for adversarial training of the components a2b and b2a.

Of our five components four are trainable (the reconstructor R, the domain translators a2b and b2a, and the discriminator D), and one is non-trainable (the smooth inverse Inv).

Figure 1 : RAN System overview.

The reconstructor receives an object from the input domain A and predicts the corresponding reconstruction.

The reconstruction, then, is validated through our smooth inverse.

The latter produces objects in a different domain, B, which are translated back to the input domain A for training purposes (b2a).

Unlike in traditional GAN systems, the purpose of our discriminator D is mainly to indicate whether the two inputs match in content, not in style.

Our novel training scheme trains the whole network via five different data paths, including two which require another domain translator, a2b.

Since, initially, neither the reconstructor nor the domain translators are trained, we are confronted with a causality dilemma.

A typical approach for solving such causality dilemmas is to solve the two components coevolutionarily by iteratively applying various influences towards a common solution.

Fig. 1 depicts the structure of the RAN, which allows for such a coevolutionary training scheme.

The discriminator receives two inputs, one from space A and one from space B. One of these inputs (either A or B) receives two values, a real and a fake value; the task of the discriminator is to distinguish between these two, given the other input.

For training, the discriminator is trained to distinguish between the different path combinations for the generation of inputs.

Consecutively, the generator modules are trained to fool the discriminator.

This adversarial game allows training the RAN.

In the following, we will present this process, as well as its involved losses, in detail.

Our optimization of R, a2b, b2a, and D involves adversarial losses, cycle-consistency losses, and regularization losses.

Specifically, we solve the following optimization:

where α i is a weight in [0, 1] and L, and L i shall be defined below.

L reg denotes the (optional) regularization losses imposed on the reconstruction output.

We define b ′ , b ′′ ∈ B and a ′ , a ′′ ∈ A in dependency of a ∈ A according to Fig. 1 as

With that, our losses are (without hyper-parameter weights)

We alternately train the different sections of our network in the following order:

1.

The discriminator D 2.

The translation from B to A (b2a) 3.

The components that perform a translation from A to B (R+Inv, a2b)

For each of these sections, we separately train the five losses

, and L 5 .

In our experiments, we used one Adam optimizer [9] for each trainable component (R, a2b, b2a, and D).

For our experiments we developed an unsupervised 3D reconstruction method using a C ∞ smooth 3D mesh renderer [11] .

Our reconstructor is a network mapping from one or multiple images to a set of 3D coordinates, and our smooth inverse is a smooth 3D mesh renderer.

As domain translators, we used the pix2pix network as well as a convolutional and deconvolutional ResNet [12] .

Describing the smooth renderer in great detail would exceed the scope of this paper.

The main differences to common ray tracers are that our renderer performs a smooth rasterization and a smooth occlusion handling / z-buffer.

Training the RAN with the scheme described in Sec. 4, we achieved first results on unsupervised 3D reconstruction trained only on camera-captured images.

Some qualitative results for that are presented in Fig. 2 .

Other inverse problems that we experimented on are speech recognition as well as the inverse problem of iterated function systems.

We presented AlgoNets as a new kind of layers for neural networks and RANs as a novel technique for solving ill-posed inverse problems.

Concurrent with their benefits, AlgoNets, such as the aforementioned rendering layer, can get computationally very expensive.

On the other hand, the rendering layer is very powerful since it allows training a 3D reconstruction without 3D supervision using the RAN.

Since the RAN is a very complex architecture that requires a very specific training paradigm, it can also take relatively long to train it.

To accommodate this issue, we found that by increasing some loss weights and introducing a probability of whether the computation is executed, the training time can be reduced by a factor of two or more.

The AlgoNet can also be used in such a way that algorithmic layers solve sub-problems of a given problem to assist a neural network in solving a larger problem.

This principle could also be used in the realm of explainable artificial intelligence [13] by adding residual algorithmic layers into neural networks and then analyzing the neurons of the trained AlgoNet.

For that, network activation and/or network sensitivity can indicate the relevance of the residual algorithmic layer.

To compute the network sensitivity of an algorithmic layer, the gradient with respect to additional weights (constant equal to one) in the algorithmic layer could be computed.

By that, similarities between classic algorithms and the behavior of neural networks could be inferred.

An alternative approach would be to gradually replace parts of trained neural networks with algorithmic layers and analyzing the effect on the new model accuracy.

In the future, we will develop a high-level smooth programming language to improve smooth representations of higher-level programming concepts.

Adding trainable weights to the algorithmic layers to improve the accuracy of smooth algorithms and/or allow the rest of the network to influence the behavior of the algorithmic layer is subject to future research.

Another future objective is the exploration of neural networks not with a fixed but instead a smooth topology.

@highlight

Solving inverse problems by using smooth approximations of the forward algorithms to train the inverse models.