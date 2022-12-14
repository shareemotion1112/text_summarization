Binarized Neural networks (BNNs) have been shown to be effective in improving network efficiency during the inference phase, after the network has been trained.

However, BNNs only binarize the model parameters and activations during propagations.

Therefore, BNNs do not offer significant efficiency improvements during training, since the gradients are still propagated and used with high precision.

We show there is no inherent difficulty in training BNNs using "Binarized BackPropagation" (BBP), in which we also binarize the gradients.

To avoid significant degradation in test accuracy, we simply increase the number of filter maps in a each convolution layer.

Using BBP on dedicated hardware can potentially significantly improve the execution efficiency (\emph{e.g.}, reduce dynamic memory footprint, memory bandwidth and computational energy) and speed up the training process with an appropriate hardware support, even after such an increase in network size.

Moreover, our method is ideal for distributed learning as it reduces the communication costs significantly (e.g., by ~32).

Using this method, we demonstrate a minimal loss in classification accuracy on several datasets and topologies.

<|TLDR|>

@highlight

Binarized Back-Propagation all you need for completely binarized training is to is to inflate the size of the network