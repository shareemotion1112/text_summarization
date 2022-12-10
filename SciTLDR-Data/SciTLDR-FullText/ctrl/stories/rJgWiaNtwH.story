One of the challenges in training generative models such as the variational auto encoder (VAE) is avoiding posterior collapse.

When the generator has too much capacity, it is prone to ignoring latent code.

This problem is exacerbated when the dataset is small, and the latent dimension is high.

The root of the problem is the ELBO objective, specifically the Kullback–Leibler (KL) divergence term in objective function.

This paper proposes a new  objective function to replace the KL term with one that emulates the maximum mean discrepancy (MMD) objective.

It also introduces a new technique, named latent clipping, that is used to control distance between samples in latent space.

A probabilistic autoencoder model, named $\mu$-VAE, is designed and trained on MNIST and MNIST Fashion datasets, using the new objective function and is shown to outperform models trained with ELBO and $\beta$-VAE objective.

The $\mu$-VAE is less prone to posterior collapse, and can generate reconstructions and new samples in good quality.

Latent representations learned by $\mu$-VAE are shown to be good and can be used for downstream tasks such as classification.

Autoencoders(AEs) are used to learn low-dimensional representation of data.

They can be turned into generative models by using adversarial, or variational training.

In the adversarial approach, one can directly shape the posterior distribution over the latent variables by either using an additional network called a Discriminator (Makhzani et al., 2015) , or using the encoder itself as a discriminator (Huang et al., 2018) .

AEs trained with variational methods are called Variational Autoencoders (VAEs) (Kingma & Ba, 2014; Rezende et al., 2014) .

Their objective maximizes the variational lower bound (or evidence lower bound, ELBO) of p θ (x).

Similar to AEs, VAEs contain two networks:

Encoder -Approximate inference network: In the context of VAEs, the encoder is a recognition model q φ (z|x) 1 , which is an approximation to the true posterior distribution over the latent variables, p θ (z|x).

The encoder tries to map high-level representations of the input x onto latent variables such that the salient features of x are encoded on z.

Decoder -Generative network: The decoder learns a conditional distribution p θ (x|z) and has two tasks: i) For the task of reconstruction of input, it solves an inverse problem by taking mapped latent z computed using output of encoder and predicts what the original input is (i.e. reconstruction x ≈ x).

ii) For generation of new data, it samples new data x , given the latent variables z.

During training, encoder learns to map the data distribution p d (x) to a simple distribution such as Gaussian while the decoder learns to map it back to data distribution p(x) 2 .

VAE's objective function has two terms: log-likelihood term (reconstruction term of AE objective function) and a prior regularization term 3 .

Hence, VAEs add an extra term to AE objective function, and approximately maximizes the log-likelihood of the data, log p(x), by maximizing the evidence lower bound (ELBO):

Maximizing ELBO does two things:

• Increase the probability of generating each observed data x.

• Decrease distance between estimated posterior q(z|x) and prior distribution p(z), pushing KL term to zero.

Smaller KL term leads to less informative latent variable.

Pushing KL terms to zero encourages the model to ignore latent variable.

This is especially true when the decoder has a high capacity.

This leads to a phenomenon called posterior collapse in literature (Razavi et al., 2019; Dieng et al., 2018; van den Oord et al., 2017; Bowman et al., 2015; Sønderby et al., 2016; Zhao et al., 2017) .

This work proposes a new method to mitigate posterior collapse.

The main idea is to modify the KL term of the ELBO such that it emulates the MMD objective (Gretton et al., 2007; Zhao et al., 2019) .

In ELBO objective, minimizing KL divergence term pushes mean and variance parameters of each sample at the output of encoder towards zero and one respectively.

This , in turn, brings samples closer, making them indistinguishable.

The proposed method replaces the KL term in the ELBO in order to encourage samples from latent variable to spread out while keeping the aggregate mean of samples close to zero.

This enables the model to learn a latent representation that is amenable to clustering samples which are similar.

As shown in later sections, the proposed method enables learning good generative models as well as good representations of data.

The details of the proposal are discussed in Section 4.

In the last few years, there have been multiple proposals on how to mitigate posterior collapse.

These proposals are concentrated around i) modifying the ELBO objective, ii) imposing a constraint on the VAE architecture, iii) using complex priors, iv) changing distributions used for the prior and the posterior v) or some combinations of these approaches.

Modifications of the ELBO objective can be done through annealing the KL term (Sønderby et al., 2016; Bowman et al., 2015) , lower-bounding the KL term to prevent it from getting pushed to zero (Razavi et al., 2019) , controlling KL capacity by upper bounding it to a pre-determined value (Burgess et al., 2018) or lower-bounding the mutual information by adding skip connections between the latent layer and the layers of the decoder (Dieng et al., 2018) .

Proposals that constrain the structure of the model do so by reducing the capacity of the decoder (Bowman et al., 2015; Yang et al., 2017; Gulrajani et al., 2016) , by adding skip connections to each layer of the decoder (Dieng et al., 2018) , or by imposing constraints on encoder structure (Razavi et al., 2019) .

Taking a different approach, Tomczak & Welling (2017) and van den Oord et al. (2017) replace simple Gaussian priors with more complex ones such as a mixture of Gaussians.

The most recent of these proposals are δ-VAE (Razavi et al., 2019) and SKIP-VAE (Dieng et al., 2018 ).

δ-VAE imposes a lower bound on KL term to prevent it from getting pushed to zero.

One of the drawbacks of this approach is the fact that it introduces yet another hyper-parameter to tune carefully.

Also, the model uses dropout to regularize the decoder, reducing the effective capacity of the decoder during training.

It is not clear how effective the proposed method is when training more powerful decoders without such regularization.

Moreover, the proposal includes an additional constraint on encoder structure, named the anti-causal encoder.

SKIP-VAE , on the other hand, proposes to lower bound mutual information by adding skip connections from latent layers to each layer of decoder.

One drawback of this approach is that it introduces additional non-linear layer per each hidden layer, resulting in more parameters to optimize.

Moreover, its advantage is not clear in cases, where one can increase capacity of decoder by increasing number of units in each layer (or number of channels in CNN-based decoders) rather than adding more layers.

When we train a VAE model, we ideally want to end up with a model that can reconstruct a given input well and can generate new samples in high quality.

Good reconstruction requires extracting the most salient features of data and storing them on latent variable ('Encoder + Latent layer' part of the model).

Generating good samples requires a generative model ('Latent layer + Decoder' part) with a model distribution that is a good approximation to actual data distribution.

However, there tends to be a trade-off between reconstruction quality of a given input, and quality of new samples.

To understand why we have such a trade-off, we can start by looking at ELBO objective function 4 :

Maximizing this objective function increases p θ (x), the probability of generating each observed data x while decreasing distance between q(z|x) and prior p(z).

Pushing q(z|x) closer to p(z) makes latent code less informative i.e. z is influenced less by input data x.

The reason why the KL term can be problematic becomes more clear when we look at the KL loss term typically modelled with log of variance during optimization:

where D is the dimension of latent variable, and i refers to i th sample.

Noting that the mean is in L2 norm, minimizing the KL term leads to pushing the each dimension of the mean,µ

d , to zero while pushing σ 2 towards 1.

This makes estimated posterior less informative and less dependent on input data.

The problem gets worse when dimension of latent variable, D, increases, or when the KL term is multiplied with a coefficient β > 1 (Higgins et al., 2017) .

Ideally, we want to be able to distinguish between different input samples.

This can be achieved by having distinctive means and variances for clusters of samples.

This is where MMD might have advantage over the KL divergence.

Matching distributions using MMD can match their sample means although their variance might still differ.

We can emulate behaviour of MMD by modifying the KL term.

We do so by changing L2 norm of mean,

Re-writing it for B samples, we have:

It is important to note that we are taking absolute value of sum of sample means.

This new formulation results in aggregate mean of samples to be zero (i.e. same mean as that of prior distribution) while allowing samples to spread out and enabling model to encode information about input data onto z. It should be noted that this new L1 norm of µ can push individual mean estimates to very high values if it is not constrained.

To avoid that, L2 norm of means for each sample is clipped by a pre-determined value during optimization.

Based on experiments, it is found that clipping L2 norm of sample means by three times square root of latent dimension works well in general although bigger values might help improve results in tasks such as classification:

This method will be referred as latent clipping for the rest of this work.

In addition, the remaining terms in the KL loss can be kept as is, i.e. exp (log σ 2 ) − log σ 2 − 1 , or we can just optimize for subset of it by using either "log σ 2 ", or " exp (log σ 2 ) − 1 " term since each method will push log σ 2 towards zero (i.e. variance towards one).

log σ 2 is chosen in this work since it is simpler.

Finally, the µ-VAE objective function can be formulated as follows:

where first term is reconstruction loss, B refers to batch size since aggregated mean is computed over batch samples, J refers to dimension of data, D refers to dimension of latent variable, x is original input, and x is reconstructions.

To visualize the implications of the latent clipping, a toy VAE model shown in Table 2 in Appendix A is used.

Figure 1 compares three cases, in which a VAE model is trained on MNIST dataset using ReLu, Tanh, and Leaky ReLu activation functions for each case, and the latent layer is visualized to observe clustering of digits.

Those three cases are: i) Standard VAE objective with the KL divergence, ii) Standard VAE objective with the KL divergence + latent clipping, and iii) µ-VAE objective function + latent clipping.

Two observations can be made:

1.

Latent clipping might help improve smoothness of latent space, even in the case of standard VAE objective, ELBO.

2. µ-VAE objective function seems to work well.

To test the effectiveness of µ-VAE objective, a CNN-based VAE model is designed and trained on MNIST and MNIST Fashion using same hyper-parameters for both datasets.

Centered isotropic Gaussian prior, p(z) ∼ N (0, 1.0), is used and the true posterior is approximated as Gaussian with an approximately diagonal co-variance.

No regularization methods such as dropout, or techniques such as batch-normalization is used to avoid having any extra influence on the performance, and to show the advantages of the new objective function.

The model is trained with four different objective functions: i) VAE (ELBO objective), ii) β-VAE with β = 4, iii) µ-VAE#1 s.t.

µ sample ≤ 3 * √ z dim and iv) µ-VAE#2 s.t.

µ sample ≤ 6 * √ z dim , where z dim = 10.

Details of architecture, objective functions, hyper-parameters, and training are described in Appendix B.

During training of the models, a simple three layer fully connected classifier is also trained over 10 dimensional latent variable to learn to classify data using features encoded on latent variable.

Classifier parameters are updated when encoder and decoder parameters are frozen and vice versa so that classifier has no impact on how information is encoded on the latent variable.

Evaluation of the generative model is done qualitatively in the form of inspecting quality, and diversity of samples.

Posterior collapse is assessed by comparing reconstructions of input data to observe whether the decoder ignores latent code encoded by input data and by comparing the KL divergences obtained for each model.

For all three objective functions, the KL divergence is measured using standard KL formula in Equation 3.

Moreover, the accuracy of the classifier trained on latent variable is used as a measure of how well the latent variable represents data (Dieng et al., 2018).

Higher classification accuracy reflects a better representation of data and opens doors to use latent representation for downstream tasks such as classification.

Figure 2 shows training curves for MNIST Fashion dataset (MNIST results can be seen in Appendix C).

The new objective function results in lower reconstruction loss, higher KL divergence, and higher classification accuracy.

Higher KL divergence and classification accuracy can be interpreted as a sign of learning a more informative latent code.

β-VAE performs the worst across all metrics as expected.

The reason is that β factor encourages latent code to be less informative, and is known to result in worse reconstruction quality (Higgins et al., 2017) .

of samples obtained using test data for both datasets.

VAE seems to able to distinguish all ten digits, but performs worse in MNIST Fashion.

β-VAE pushes samples closer as expected, which explains why its performance is low in classification task.

µ-VAE, on the other hand, is able to cluster similar samples together in both datasets.

Moreover, when upper-bound on µ sample is increased, it spreads out clusters of samples, making them easier to distinguish.

Hence, upper-bound used in latent clipping can be a knob to control distance between samples.

Also, we should note that we can achieve similar clustering results to the Table 1 lists classification accuracy obtained using test datasets.

µ-VAE performs the best as expected since it is able to push the clusters apart.

Higher upper bound on µ sample results in a higher classification accuracy.

Also, it should be noted that reported accuracy numbers can be improved, but the purpose of this test was to show that new objective function can reliably be used in downstream tasks such as classification.

Figure 4 compares sample distributions obtained at each dimension of latent variable using test dataset of MNIST Fashion for each objective function.

β-VAE samples follow N (0, 1) prior very closely, and hence resulting in the smallest KL divergence term.

Sample distributions from the most dimensions of VAE are also close to prior distribution although some of them show a multi-modal behavior.

Sample distributions from both µ-VAE#1 & #2 result in zero mean, but they are more spread out as expected.

Spread is controlled by upper-bound on µ sample .

Similar to VAE, some sample distributions show a multi-modal behavior.

Figure 5 shows reconstruction of input using test dataset.

β-VAE reconstructions are either blurrier, or wrong, the latter of which is a sign of posterior collapse.

VAE performs better, and both versions of µ-VAE gives the best reconstruction quality.

Figure 6 shows images generated using random samples drawn from multivariate Gaussian, N(0, σ), where σ = 1 is for VAE and β-VAE while it is 3 for µ-VAE since their samples are more spread out (MNIST results can be seen in Appendix C).

We can observe that some samples generated from µ-VAE models have dark spots.

This is because the model is trying to generate texture on these samples.

This can also be observed in samples of VAE model, but it is less pronounced.

However, samples from β-VAE do not show any such phenomena since the model perhaps learns global structure of shapes while ignoring local features.

Failing to capture local structures is a known problem in latent variable models (Larsen et al., 2015; Razavi et al., 2019) .

N(0, σ) .From left to right, model (σ): VAE (σ=1), β-VAE (σ=1), µ-VAE#1 (σ=3), and µ-VAE#2 (σ=3).

Higger σ is used for µ-VAE models since their samples are more spread out.

that most dimensions of latent variable are not very informative.

VAE is slightly better.

However, both µ-VAE models learn diverse classes of objects across different dimensions.

Moreover, they learn different classes on opposite sides of same dimension.

This is encouraging since it shows its power to learn rich representations.

In this work, a new objective function is proposed to mitigate posterior collapse observed in VAEs.

It is shown to give better reconstruction quality and to learn good representations of data in the form of more informative latent codes.

A method, named latent clipping, is introduced as a knob to control distance between samples.

Samples can be pushed apart for tasks such as classification, or brought closer for smoother transition between different clusters of samples.

Unlike prior work, the proposed method is robust to parameter tuning, and does not constraint encoder, or decoder structure.

It can be used as a direct replacement for ELBO objective.

Moreover, the proposed method is demonstrated to learn representations of data that can work well in downstream tasks such as classification.

Applications of µ-VAE objective with more powerful decoders in various settings can be considered as a future work.

Optimization: In all experiments, learning rate of 1e-4 and batch size of 64 are used.

Adam algorithm with high momentum (β1 = 0.9, β2 = 0.999) is used as optimizer.

High momentum is chosen mainly to let most of previous training samples influence the current update step.

For reconstruction loss, mean square error, x−x 2 , is used for all cases.

As for initialization, since the model consists of convolutional layers with Leaky ReLu in both encoder and decoder, Xavier initialization is used Glorot & Bengio (2010) .

Thus, initial weights are drawn from a Gaussian distribution with standard deviation (stdev) of 2/N , where N is number of nodes from previous layer.

For example, for a kernel size of 3x3 with 32 channels, N = 288, which results in stdev of 0.083.

Objective functions are shown in Table 3 , where µ-VAE objective is written explicitly to avoid any ambiguity in terms of how batch statistics are computed.

Table 4 shows model architecture as well as classifier used for all experiments.

It consists of CNNbased encoder and decoder while classifier is three layer fully connected neural network.

They all use Leaky Relu activation and learning rate of 1e-4.

<|TLDR|>

@highlight

This paper proposes a new  objective function to replace KL term with one that emulates maximum mean discrepancy (MMD) objective. 