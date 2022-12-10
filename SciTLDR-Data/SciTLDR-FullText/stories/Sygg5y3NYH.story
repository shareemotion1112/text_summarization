Variational Auto Encoders (VAE) are capable of generating realistic images, sounds and video sequences.

From practitioners point of view, we are usually interested in solving problems where tasks are learned sequentially, in a way that avoids revisiting all previous data at each stage.

We address this problem by introducing a conceptually simple and scalable end-to-end approach of incorporating past knowledge by learning prior directly from the data.

We consider scalable boosting-like approximation for intractable theoretical optimal prior.

We provide empirical studies on two commonly used benchmarks, namely  MNIST and Fashion MNIST on disjoint sequential image generation tasks.

For each dataset proposed method delivers the best results among comparable approaches,  avoiding catastrophic forgetting in a fully automatic way with a fixed model architecture.

Since most of the real-world datasets are unlabeled, unsupervised learning is an essential part of the machine learning field.

Generative models allow us to obtain samples from observed empirical distributions of the complicated high-dimensional objects such as images, sounds or texts.

This work is mostly devoted to VAEs with the focus on incremental learning setting.

It was observed that VAEs ignore dimensions of latent variables and produce blurred reconstructions (Burda et al., 2015; Sønderby et al., 2016) .

There are several approaches to address these issues, including amortization gap reduction (Kim et al., 2018) , KL-term annealing (Cremer et al., 2018) and alternative optimization objectives introduction (Rezende and Viola, 2018) .

In all cases, it was observed that the choice of the prior distribution is highly important and use of default Gaussian prior overregularizes the encoder.

In this work, we address the problem of constructing the optimal prior for VAE.

The form of optimal prior (Tomczak and Welling, 2018) was obtained by maximizing a lower bound of the marginal likelihood (ELBO) as the aggregated posterior over the whole training dataset.

To construct reasonable approximation, we consider a method of greedy KL-divergence projection.

Applying the maximum entropy approach allows us to formulate a feasible optimization problem and avoid overfitting.

The greedy manner in which components are added to prior reveals a high potential of the method in an incremental learning setting since we expect prior components to store information about previously learned tasks and overcome catastrophic forgetting (McCloskey and Cohen, 1989; Goodfellow et al., 2013; Nguyen et al., 2017) .

From practitioners point of view, it is essential to be able to store one model, capable of solving several tasks arriving sequentially.

Hence, we propose the algorithm with one pair of encoderdecoder and update only the prior.

We validate our method on the disjoint sequential image generation tasks.

We consider MNIST and Fashion-MNIST datasets.

VAEs consider two-step generative process by a prior over latent space p(z) and a conditional generative distribution p θ (x|z), parametrized by a deep neural network.

Given empirical data distribution p e (x) = 1 N N n=1 δ(x − x n ) we aim at maximizing expected marginal log-likelihood.

Following the variational auto-encoder architecture, amortized inference is proposed by choosing variational c A. Authors.

posterior distribution q φ (z|x) to be parametrized by DNN, resulting in a following objective:

(1)

To obtain a richer prior distribution one may combine variational and empirical bayes approaches and optimize the objective (1) over the prior distribution.

For a given empirical density, optimal prior is a mixture of posterior distributions in all points of the training dataset.

Clearly, such prior leads to overfitting.

Hence, keeping the same functional form, truncated version with K presudoinputs was proposed (Tomczak and Welling, 2018) as VampPrior.

In the present paper, we address two crucial drawbacks of the VampPrior.

Firstly, for large values of K variational inference will be very computationally expensive.

Even for the MNIST dataset Tomczak and Welling (2018) used the mixture of 500 components.

Secondly, it is not clear how to choose K for the particular dataset, as we have a trade-off between prior capacity and overfitting.

For this purpose, we adapt maxentropy variational inference framework (Egorov et al., 2019) .

We add components to the prior during training in a greedy manner and show that in such setting, fewer components are needed for the comparable performance.

Assume, that we want to approximate complex distribution p * by a mixture of simple components p i Each component of the mixture is learned greedily.

At the first stage we initialize mixture by a standard normal distribution.

Afterwards, we add new component h from some family of distributions Q, with the weight α one by one p t = αh + (1 − α)p t−1 , α ∈ (0, 1) in two stages:

1.

Find optimal h ∈ Q. We apply Maximum Entropy approach to minimize KL-divergence between mixture and target distribution.

This task can be reformulated as a following KL-

2.

Choose α, corresponding to the optimal h: α * = arg min

In this work, we suggest combining boosting algorithm for density distributions with the idea of VampPrior to deal with the problem of catastrophic forgetting in the incremental learning setting.

Proposed algorithm for training VAE consists of two steps.

On the first one we optimize evidence lower bound (1) w.r.t.

the parameters of the encoder and decoder.

On the second stage, we learn new component h for the prior distribution, and its weight α, keeping parameters of the encoder and decoder fixed.

We learn each component to be posterior given the learnable artificial input u: q φ (z|u) with target density being mixture of posteriors in all points from random subset M of the whoel training dataset D. Parameters of the first component u 0 are obtained by ELBO maximization simultaneously with the network parameters, as shown in the Algorithm 1.

In the incremental learning setting, we do not have access to the whole dataset.

Instead, subsets D 1 , . . .

D T arrive sequentially and may come from different domains.

With the first task D 1 we Algorithm 1: BooVAE algorithm

, λ, Maximal number of components K Output: p K , θ * , φ * Choose random subset M ⊂ D and initialize prior p 0 = q φ (z|u 0 ) and k = 1; θ * , φ * , u 0 = arg max L(p 0 , θ, φ);

follow Algorithm 1 to obtain prior p (1) and optimal values of the network parameters.

Starting from t > 1, we add regularization to ELBO, which ensures that the model keeps encoding and decoding learned prior components in the same manner (see Appendix B).

Since we do not have access to the whole dataset anymore, the form of optimal prior also changes.

We use prior of the previous step as a proxy for an optimal one (see Appendix C) p * (t) ≈ t−1

We perform experiments on MNIST dataset (LeCun et al., 2010) , containing ten hand-written digits and on fashion MNIST (Xiao et al., 2017) , which also has ten classes of different clothing items.

Therefore, we have ten tasks in each dataset for sequential learning.

To evaluate the performance of the VAE approach, we estimate a negative log-likelihood (NLL) on the test set, calculated by importance sampling method, with 5000 samples for each observation.

In an offline setting, we aim at comparing our method to VampPrior and Mixture of Gaussians prior.

In the first case, each component is a posterior distribution given learnable pseudo-input, while in the second we learn each component as a Gaussian with diagonal covariance matrix in the latent space.

For both priors, all the component are learned simultaneously.

Results in the tables above demonstrate that BooVAE manages to overcome catastrophic forgetting better than pure EWC regularization with standard normal prior.

Unstable NLL values for VAE with standard prior and EWC can be explained by the fact that some classes in the dataset are quite similar and even though model forgets old class, knowledge about a new class let her reconstruct it relatively good, resulting in deceptively better results.

In Appendix D we provide results for each task separately to illustrate this effect further.

Figure 1: Samples from prior after training on 10 tasks incrementally.

Our approach is capable of sampling images from different tasks, while other methods either stick to the latest seen class, or are able to reproduce images from simplest classes

On Figure 1 , we provide samples from the prior illustrating generation ability of all the methods after training on all of the ten tasks sequentially.

Appendix E and F provide detailed qualitative and quantitative (Figure 3 ) evaluation of the samples diversity for different models.

In this work, we propose a method for learning a data-driven prior, using a MM algorithm which allows us to reduce the number of components in the prior distribution without the loss of performance.

Based on this method, we suggest an efficient algorithm for incremental VAE learning which has single encoder-decoder pair for all the tasks and drastically reduces catastrophic forgetting.

To make sure that model keeps encoding and decoding prior components as it did during the training of the corresponding task, we add a regularization term to ELBO.

For that purpose at the end of each task we compute reconstructions of the components' mean value p θ j (x|µ i,j ) = p ij (x).

All thing considered, objective for the maximization step is the following:

When training a new component and its' weight, we want optimal prior that we approximate to be close to the mixture of variational posteriors at all the training points, seen by the model.

Since we don't have access to the data from the previous tasks, we suggest using trained prior as a proxy for the corresponding part of the mixture.

Therefore, optimal prior for tasks 1 : t can be expressed, using optimal prior from the previous step and training dataset from the current stage:

During training, we approximate optimal prior by the mixture p (t) , using random subset containing n observations of the given dataset M t ⊂ D t .

Therefore, the optimal prior can be approximated in the following way:

To evaluate diversity of the generated images, we calculate KL-divergence between Bernoulli distribution with equal probability for each class and empirical distribution of classes generated by the model.

Since we want to assign classes to generated images automatically, we train classification network, which can classify images with high probability (more than 90%), use it to label generated objects and calculated the empirical distribution over 10000 generated samples.

Figure 3 depicts proposed metric to evaluate diversity.

We want this value to stay as close as possible to 0 as the number of tasks grows since it will mean that model keeps generating diverse images.

We can see a drastic difference between boosting and other approaches: samples from prior, estimated by the boosting approach are very close to uniform in contrast to all the comparable methods.

@highlight

Novel algorithm for Incremental learning of VAE with fixed architecture