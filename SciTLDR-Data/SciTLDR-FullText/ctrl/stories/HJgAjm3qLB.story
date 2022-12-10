We develop end-to-end learned reconstructions for lensless mask-based cameras, including an experimental system for capturing aligned lensless and lensed images for training.

Various reconstruction methods are explored, on a scale from classic iterative approaches (based on the physical imaging model) to deep learned methods with many learned parameters.

In the middle ground, we present several variations of unrolled alternating direction method of multipliers (ADMM) with varying numbers of learned parameters.

The network structure combines knowledge of the physical imaging model with learned parameters updated from the data, which compensate for artifacts caused by physical approximations.

Our unrolled approach is 20X faster than classic methods and produces better reconstruction quality than both the classic and deep methods on our experimental system.

<|TLDR|>

@highlight

We improve the reconstruction time and quality on an experimental mask-based lensless imager using an end-to-end learning approach which incorporates knowledge of the imaging model.