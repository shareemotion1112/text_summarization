We study the use of knowledge distillation to compress the U-net architecture.

We show that, while standard distillation is not sufficient to reliably train a compressed U-net, introducing other regularization methods, such as batch normalization and class re-weighting, in knowledge distillation significantly improves the training process.

This allows us to compress a U-net by over 1000x, i.e., to 0.1% of its original number of parameters, at a negligible decrease in performance.

where a y t ij indicates the label of sample t at location (i, j), andŷ t is the probability map predicted 41 by the network.

In the standard setting, this probability map is obtained via the softmax function.

to generate probabilitiesŷ t * for each sample x t in a validation set.

These probabilities are then H(y t ,ŷ t ), for mixed distillation.

After being trained at T > 1, the softmax temperature of the 48 student network is reduced back to 1.

Improving Distillation.

As evidenced by our results in Section 3, using standard distillation, whether 50 vanilla or mixed, did not prove sufficient to distill a standard U-net into a very small one.

To overcome 51 this, we therefore propose two modifications of the original strategy.

First, as indicated in FIG0

<|TLDR|>

@highlight

We present additional techniques to use knowledge distillation to compress U-net by over 1000x.

@highlight

The authors introduced a modified distillation strategy to compress a U-net architecture by over 1000x while retaining an accuracy close to the original U-net.