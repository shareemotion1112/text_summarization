Interpreting generative adversarial network (GAN) training as approximate divergence minimization has been theoretically insightful, has spurred discussion, and has lead to theoretically and practically interesting extensions such as f-GANs and Wasserstein GANs.

For both classic GANs and f-GANs, there is an original variant of training and a "non-saturating" variant which uses an alternative form of generator update.

The original variant is theoretically easier to study, but the alternative variant frequently performs better and is recommended for use in practice.

The alternative generator update is often regarded as a simple modification to deal with optimization issues, and it appears to be a common misconception that the two variants minimize the same divergence.

In this short note we derive the divergences approximately minimized by the original and alternative variants of GAN and f-GAN training.

This highlights important differences between the two variants.

For example, we show that the alternative variant of KL-GAN training actually minimizes the reverse KL divergence, and that the alternative variant of conventional GAN training minimizes a "softened" version of the reverse KL.

We hope these results may help to clarify some of the theoretical discussion surrounding the divergence minimization view of GAN training.

<|TLDR|>

@highlight

Typical GAN training doesn't optimize Jensen-Shannon, but something like a reverse KL divergence.