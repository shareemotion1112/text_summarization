The Variational Auto Encoder (VAE) is a popular generative  latent variable model that is often  applied for representation learning.

Standard VAEs assume continuous valued  latent variables and are trained by maximization of the evidence lower bound (ELBO).

Conventional methods obtain a  differentiable estimate of the ELBO with reparametrized sampling and optimize it with Stochastic Gradient Descend (SGD).

However, this is not possible if  we want to train VAEs with discrete valued latent variables,  since reparametrized sampling is not possible.

Till now, there exist no simple solutions to circumvent this problem.

In this paper, we propose an easy method to train VAEs  with binary or categorically valued latent representations.

Therefore, we use a differentiable estimator for the ELBO which is based on importance sampling.

In experiments, we verify the approach and train two different VAEs architectures with Bernoulli and  Categorically distributed latent representations on two different benchmark datasets.

The variational auto encoder (VAE) is a generative model which it is trained to approximate the true data generating distribution p(x) of an observed random vector x from a given training set D = {x 1 , ..., x N } BID5 ; BID6 ).

It is an especially suited model if x is high dimensional or has highly nonlinear dependent elements.

Therefore, the VAE is oftenly used for tasks like density estimation, data generation, data interpolation BID10 ), outlier and anomaly detection BID0 ; BID12 ) or clustering BID4 ; BID3 ).As shown in FIG0 , the VAE is an easy latent variable model, where the observations x ∼ p(x|z) are dependent on latent variables z ∼ p(z).

During training, the VAE maximizes the probability p(x) to observe the data x. Therefore, the negative evidence lower bound (ELBO) L(θ) = −E q(z|x) [ln p(x|z) ] + D KL (q(z|x)||p(z))(1) ≥ − ln p(x) + D KL (q(z|x)||p(z|x))is minimized, where p(z|x) = p(x|z)p(z)/ p(x|z)p(z)dz is the true but intractable posterior distribution the model assigns to z, q(z|x) is the corresponding tractable variational approximation and D KL (q(z|x)||p(z|x)) is the Kullback-Leibler (KL) divergence between p(z|x) and q(z|x).

Because D KL (q(z|x)||p(z|x)) > 0, minimizing L(θ) means to maximize the probability p(x) the model assigns to observations x.

Therefore, D KL (q(z|x)||p(z|x)) must be as as close as possible to 0, meaning that after training q(z|x) is a very good approximation of the true posterior p(z|x).

BID5 proposed to minimize L(θ), using stochastic gradient descent on a training data set, which they called Stochastic Gradient Variational Bayes (SGVB).The VAE uses parametric distributions that are parametrized by an encoder network with parameters θ E and a decoder network with parameters θ D for both q(z|x) and p(x|z), respectively.

This leads to the well known encoder-decoder structure in Fig. 2 .

The data likelihood is a distribution with mean x, that is the output of the decoder network.

Further, we assume in this paper, that the variational posterior q(z|x) is a distribution from the exponential family DISPLAYFORM0 with natural parameters η(x; θ E ), sufficient statistic T (z) and log partition function A(η(x; θ E )).

This gives us the flexibility to study training with different q(z|x) in the same mathematical framework.

As shown in Fig. 2 , the natural parameters η are the output of the encoder network, where we drop the arguments x, θ E for shorter notations in the remainder of the paper.

DISPLAYFORM1 Figure 2: The encoder-decoder structure of a VAE.

The encoder parametrizes q(z|x) as an exponential family distribution with natural parameters η and the decoder parametrizes p(x|z) with mean x.

The conventional VAE proposed in BID5 ; BID6 ) learns continuous latent representations z ∈ R c .

It uses i.i.d.

Gaussian distributed z, meaning that DISPLAYFORM2 T and A(η) is chosen such that q(z|x) integrates to one.

The likelihood is also Gaussian, with p(x|z) ∼ N (x, 1).

But in many applications learning discrete rather than continuous representations is advantageous.

Binary representations z ∈ {0, 1} c can for example be used very efficiently for hashing, what is a powerful method for large-scale visual search BID8 ).

Learning Categorical representations z ∈ {e 1 , ..., e c } is interesting, because this naturally lead to clustering of the data x, as shown in the experiments.

Further, for both binary and categorical z it is easy to find entropy based heuristics to choose the size of the latent space, because the entropy is bounded for discrete z.

However, training VAEs with discrete latent representations is problematic, since standard SGVB can not be applied for optimization.

Because SGVB is a gradient based method, we need to calculate the derivative of the two cost terms with respect to the encoder and decoder parameters ∂ ∂θ DISPLAYFORM3 where L KL (θ) only depends on the encoder parameters and the expected log likelihood term L L (θ) depends on both encoder and decoder parameters.

For a suited choice of p(z) and q(z|x), L KL (θ) can be calculated in closed form.

However, L L (θ) contains an expectation over z ∼ q(z|x) that has to be estimated during training.

A good estimatorL L (θ) for L D (θ) that is unbiased, differentiable with respect to θ and that has low variance is the key to train VAEs.

SGVB uses an estimatorL R L (θ) that is based on reparametrization of q(z|x) and sampling BID5 ).

However, as described in section 2, this method places many restrictions on the form of q(z|x) and fails if q(z|x) can not be reparametrized.

This is the case if z is discrete, for example.

In this paper, we propose a simple and differentiable estimatorL DISPLAYFORM4 that is based on importance sampling.

Because no reparametrization is needed, it can be used to train VAEs with binary or categorical latent representations.

Compared to previously proposed methods like the Vector Quantised-Variational Auto Encoder (VQ-VAE) (van den Oord et al. FORMULA0 ), which is based on a straight-through estimator for the gradient of L L (θ) BID1 ), our proposed estimator has two advantages.

It is unbiased and its variance approaches zero the closer we are to the optimum.

where is a random variable with the distribution p( ), m are samples from this distribution and DISPLAYFORM0 .

This estimator can be used to train VAEs with SGVB if two conditions are fulfilled: I) There exists a distribution p( ) and a reparametrization function DISPLAYFORM1 II) The derivative of Eq. 5 must exist.

With Eq. 8, we obtain DISPLAYFORM2 meaning that both the reparametrization function f ( , θ) and ln p(x|z) must be differentiable with respect to z and θ, respectively, to allow direct backpropagation of the gradient through the reparametrized sampling operator.

If these conditions are fulfilled, the gradient can flow directly from the output to the input layer of the VAE, as shown in FIG1 .

Distributions over discrete latent representations z can not be reparametrized this way.

Therefore, this estimator can not be used to train VAEs with such representations.

DISPLAYFORM3

We propose an estimatorL I L (θ) which is based on importance sampling and can also be used to train VAEs with binary or categorical latent representations z. Expanding Eq. 4 leads to DISPLAYFORM0 where q I (z) is an arbitrary distribution that is of the same form as q(z|x) which is independent from the parameters θ.

z m ∼ q I (z) are samples from this distribution.

The estimator computes a weighted sum of the log likelihood ln p(x|z m ) with the weighting q(z m |x)/q I (z).The benefit is that the log likelihood ln p(x|z m ) depends on the decoder parameters θ D only and not on the encoder parameters θ E whereas the weighting q(z m |x)/q I (z) depends only on θ D and not on θ E .

Therefore, calculation of the gradient ofL DISPLAYFORM1 with DISPLAYFORM2 As shown in FIG3 , gradient backpropagation is split into two separate parts.

DISPLAYFORM3

Assume the latent representation z has i.i.d.

components that are Bernoulli distributed, i.e. both the variational posterior distribution q(z|x) and q I (z) have the form DISPLAYFORM0 DISPLAYFORM1 where z ∈ {0, 1} c , η = [ln(q 1 /(1 − q 1 )), ..., ln(q c /(1 − q c ))] is the output vector of the encoder that contains the logits of the independent Bernoulli distributions and A(η) = 1 T ln(1 + e η ) are the corresponding log-partition functions.

Hence, Eq. 17 is DISPLAYFORM2 1+e −η contains the probabilities q(z i = 1|x).

The variance of the estimatorL I L (θ) heavily depends on the choice of the natural parameters ξ of the distribution q I (z).

We choose ξ = η, leading to a gradient of the very simple form DISPLAYFORM3 This estimator of the gradient has two desirable properties for training, which can be easily seen in the one dimensional case with z ∈ {0, 1}. The mean of the estimator is DISPLAYFORM4 meaning that the estimator is unbiased.

Further, the variance of ∂ ∂θ EL I L (θ) reduces to zero, the closer q is to 0 or 1, because q(1 − q) → 0 and hence the variance of the estimator approaches 0.That is desirable, since there are only three interesting cases during training that are shown in This means, that the only candidate points that maximize the log likelihood are q = 0 or q = 1 lie near q(z = 1|x) = 0/1.

Therefore, the longer we train, the more accurate the gradient estimate will be.

For Categorically distributed z both the variational posterior distribution q(z|x) and q I (z), again have the form DISPLAYFORM0 but now z ∈ {e 1 , ..., e c } can assume only c different values.

The vector of natural parameters is η = [ln(p 1 /p c ), ..., ln(p c−1 /p c ), 0)] and the log partition function is A(η) = ln 1 T e η .With the formulas above, we arrive at the same easy form of the expected gradient of the log likelihood DISPLAYFORM1 but now with q = softmax(η) that consists of the probabilities q i = q(z = e i ), where DISPLAYFORM2

In the following section, we show our preliminary experiments on the MNIST and Fashion MNIST datasets LeCun & Cortes FORMULA0 ; BID11 .

Two different kinds of VAEs have been evaluated:1.

The BVAE with Bernoulli distributed z ∈ {0, 1} c .2.

The CVAE with Categorically distributed z ∈ {e 1 , ..., e c }.To train both architectures, the estimatorL I L (θ) derived in Sec. 5 is used.

Both BVAE and CVAE are tested with two different architectures given in Tab.

1.

The fully connected architecture has 2 dense encoder and decoder layers.

The encoder and decoder networks of the convolutional architecture consist of 4 convolutional layers and one dense layer each.

In our first experiment we train a FC BVAE with c = 50, i.e. z ∈ {0, 1} 50 and a FC CVAE with c = 100, i.e. z ∈ {z 1 , ..., z 100 }.

We train them for 300 epochs on the MNIST dataset, using DISPLAYFORM0 SGVB with our proposed estimatorL I L (θ), to estimate the expected log likelihood, and ADAM as optimizer.

FIG6 shows the convergence of the loss, the log likelihood the VAEs assign to the training data ln p(x|z) and the variance of the estimatorL I L (θ), for a learning rate of 1e − 3 and a batch size of 2048.

During training, the loss decreases steadily without oscillation.

We observe that the variance of the estimatorL I L (θ) decreases the longer we train and the closer we get to the optimum.

This is consistent with our theoretically considerations in Sec. 4.

The results of the corresponding simulations with the CNN BVAE and the CNN CVAE are shown in Fig. 9 , in the appendix.

The performance of the FC CVAE is worse than the performance of the FC BVAE.

Training converges to a lower log likelihood ln p(x|z), because the maximal information content H CV AE (z) ≤ ln(100) of the latent variables of the FC CVAE is much less than the maximal information content H BV AE (z) ≤ c ln(2) of the latent variables of the FC BVAE.

The FC CVAE can at maximum learn to generate 100 different handwritten digits, what is a small number compared to the 2 50 different images that the FC CVAE can learn to generate.

FIG7 shows handwritten digits that are generated by the FC BVAE and the FC CVAE if we sample z from the variational posterior q(z|x).

To draw samples from q(z|x), we feed test data which has not been seen during training to the encoders.

The test data is shown in FIG7 and FIG7 .

The corresponding reconstructions generated by the decoders are shown in FIG7 and FIG7 .

Both input and reconstructed images are very similar in case of the FC BVAE, meaning that it can approximate the data generating distribution p(x) well.

However, in case of the FC CVAE, the generated are blurry and look very different than the input of the encoder.

In some cases, the class of the generated digit is even flipped.

This happens because of the the very limited model capacity.

Similar results for the CNN BVAE and CNN CVAE are shown in FIG0 in the appendix.

FIG9 shows generated images of the FC BVAE if we sample z ∼ p(z) from the prior distribution.

A few generated images look like templates of handwritten digits and the remaining generated images seem to resemble mixtures of different digits.

This is similar to the behaviour of a VAE with continuous latent variables, where we can interpolate between or generate mixtures of different Since the FC CVAE can only learn to generate 100 different images, its decoder learns to generate template images that fit well to all the training images.

We observe that some latent representations are decoded to meaningless patterns that just fit well to the data in avarage.

However, the decoder also learned to generate at least one template image for each class of handwritten digits.

Hence, the categorical latent representation can be interpreted as the cluster affiliation and the encoder of the FC CVAE automatically learns to cluster the data.

Similar results for the CNN BVAE and CNN CVAE are shown in FIG0 in the appendix.

A major drawback of the FC CVAE is, that the latent space of the FC CVAE can encode only very little information and thus its generative capabilites are poor.

However, we think that they can be increased considerably if we allow a hybrid latent space with some continuous latent variables, as proposed in BID2 .

This could lead to a powerfull model for nonlinear clustering.

In this paper, we derived an easy estimator for the ELBO, which does not rely on reparametrized sampling and therefore can be used to obtain differentiable estimates, even if reparametrization is not possible, e.g. if the latent variables z are Bernoulli or Categorically distributed.

We have shown theoretically and in experiments, close to the optimal parameter configuration, the variance of the estimator approaches zero.

This is a very desirable property for training.

As shown in FIG0 , the gradient variance approaches 0, the closer we get to the optimum.

This is the same behaviour as for the MNIST dataset.

As shown in FIG0 , the FC BVAE can correctly reconstruct the shape of the given clothes with high accuracy.

However, details like texture are lost.

This is due to the limited model capacity, i.e. the latent representation z ∈ {0, 1} 50 of the given VAE can at most encode 50Bits of information.

@highlight

We propose an easy method to train Variational Auto Encoders (VAE) with discrete latent representations, using importance sampling

@highlight

Introducting an importance sampling distribution and using samples from distribution to compute importance-weighted estimate of the gradient

@highlight

This paper proposes to use important sampling to optimize VAE with discrete latent variables.