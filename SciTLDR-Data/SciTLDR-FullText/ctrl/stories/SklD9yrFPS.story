Neural Tangents is a library for working with infinite-width neural networks.

It provides a high-level API for specifying complex and hierarchical neural network architectures.

These networks can then be trained and evaluated either at finite-width as usual, or in their infinite-width limit.

For the infinite-width networks, Neural Tangents performs exact inference either via Bayes' rule or gradient descent, and generates the corresponding Neural Network Gaussian Process and Neural Tangent kernels.

Additionally, Neural Tangents provides tools to study gradient descent training dynamics of wide but finite networks.



The entire library runs out-of-the-box on CPU, GPU, or TPU.

All computations can be automatically distributed over multiple accelerators with near-linear scaling in the number of devices.



In addition to the repository below, we provide an accompanying interactive Colab notebook at https://colab.sandbox.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb

<|TLDR|>

@highlight

Keras for infinite neural networks.