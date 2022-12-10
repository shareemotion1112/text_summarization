Neural Tangents is a library designed to enable research into infinite-width neural networks.

It provides a high-level API for specifying complex and hierarchical neural network architectures.

These networks can then be trained and evaluated either at finite-width as usual or in their infinite-width limit.

Infinite-width networks can be trained analytically using exact Bayesian inference or using gradient descent via the Neural Tangent Kernel.

Additionally, Neural Tangents provides tools to study gradient descent training dynamics of wide but finite networks in either function space or weight space.



The entire library runs out-of-the-box on CPU, GPU, or TPU.

All computations can be automatically distributed over multiple accelerators with near-linear scaling in the number of devices.

Neural Tangents is available at https://www.github.com/google/neural-tangents



We also provide an accompanying interactive Colab notebook at https://colab.sandbox.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb

@highlight

Keras for infinite neural networks.