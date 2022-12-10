Recent advances in deep learning techniques has shown the usefulness of the deep neural networks in extracting features required to perform the task at hand.

However, these features learnt are in particular helpful only for the initial task.

This is due to the fact that the features learnt are very task specific and does not capture the most general and task agnostic features of the input.

In fact the way humans are seen to learn is by disentangling features which task agnostic.

This indicates that leaning task agnostic features by disentangling only the most informative features from the input data.

Recently Variational Auto-Encoders (VAEs) have shown to be the de-facto models to capture the latent variables in a generative sense.

As these latent features can be represented as continuous and/or discrete variables, this indicates us to use VAE with a mixture of continuous and discrete variables for the latent space.

We achieve this by performing our experiments using a modified version of joint-vae to learn the disentangled features.

Feature learning is one of the most fundamental task in machine learning and recently deep learning has made revolutionary advanced in this.

What ever the machine learning task at hand, deep neural networks are excellent models for feature extraction from a raw data.

But the features extracted or learned are very task specific as one use particular loss functions that are suited for task at hand.

For example, cross entropy loss used for multiclass classification problems.

This way of learning performs well only for the particular trained task leading to what is called as a narrow or weak artificial intelligence.

However, to achieve the ultimate goal of true or general artificial intelligence, one needs to learn representations in a task agnostic manner.

These task agnostic features should be enough to capture all the required information of the given entity.

One such effort made in recent times is towards learning disentangled representations.

As BID0 defines, disentangled representations are the representations where a change in a single unit of the representation corresponds to a change in a single factor of the BID2 .In this work, we experiment with JointVAE BID2 to explore the disentangled representation for the given dataset Gondal (2019).

In the next sections, we discuss our experimental setup and results.

There are several state-of-the-art variants of VAEs have been reported to extract disentangled representations BID4 BID1 BID2 .

A common assumption in these models is that latent variables follow Gaussian distribution.

The reason for this is, (a) this is the most common and simplest form of distribution observed over multiple naturally occurring datasets, and (b) the assumption of Gaussian distribution helps in simplifying the sampling of latent variables using reparametrization trick.

This assumption of Gaussian distribution holds only when the data follows linear interpolation in both the input feature space and latent variable space.

In the case of discrete variables, where we cannot observe the linear interpolation, it is necessary to directly represent the latent feature as a discrete multinominal variables.

This is achieved in Joint-VAE Dupont (2018) by using a mixture of continuous and discrete latent variables to represent the disentangled features.

In this model, the continuous variables are assumed to be of Gaussian distribution and the discrete variables are assumed to be multinominal distribution.

For continuous latent variables, one can use the normal reparameterization trick for sampling from latent variable.

However, in the case of discrete multinominal variable, the sampling is done using Gumbel Softmax trick BID6 .

This allows us to represent the disentangled features in terms of both continuous and discrete variables without any assumptions, thus utilizing the best of both worlds.

<|TLDR|>

@highlight

Mixture Model for Neural Disentanglement