Unsupervised and semi-supervised learning are important problems that are especially challenging with complex data like natural images.

Progress on these problems would accelerate if we had access to appropriate generative models under which to pose the associated inference tasks.

Inspired by the success of Convolutional Neural Networks (CNNs) for supervised prediction in images, we design the Neural Rendering Model (NRM), a new hierarchical probabilistic generative model whose inference calculations correspond to those in a CNN.

The NRM introduces a small set of latent variables at each level of the model and enforces dependencies among all the latent variables via a conjugate prior distribution.

The conjugate prior yields a new regularizer for learning based on the paths rendered in the generative model for training CNNs–the Rendering Path Normalization (RPN).

We demonstrate that this regularizer improves generalization both in theory and in practice.

Likelihood estimation in the NRM yields the new Max-Min cross entropy training loss, which suggests a new deep network architecture–the Max- Min network–which exceeds or matches the state-of-art for semi-supervised and supervised learning on SVHN, CIFAR10, and CIFAR100.

@highlight

We develop a new deep generative model for semi-supervised learning and propose a new Max-Min cross-entropy for training CNNs.