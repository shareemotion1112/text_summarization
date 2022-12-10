Variational autoencoders (VAEs) have been successful at learning a low-dimensional manifold from high-dimensional data with complex dependencies.

At their core, they consist of a powerful Bayesian probabilistic inference model, to capture the salient features of the data.

In training, they exploit the power of variational inference, by optimizing a lower bound on the model evidence.

The latent representation and the performance of VAEs are heavily influenced by the type of bound used as a cost function.

Significant research work has been carried out into the development of tighter bounds than the original ELBO, to more accurately approximate the true log-likelihood.

By leveraging the q-deformed logarithm in the traditional lower bounds, ELBO and IWAE, and the upper bound CUBO, we bring contributions to this direction of research.

In this proof-of-concept study, we explore different ways of creating these q-deformed bounds that are tighter than the classical ones and we show improvements in the performance of such VAEs on the binarized MNIST dataset.

Variational autoencoders (VAEs) BID10 , BID4 ) are powerful Bayesian probabilistic models, which combine the advantages of neural networks with those of Bayesian inference.

They consist of an encoder created with a neural network architecture, which maps the high-dimensional input data, x, to a low-dimensional latent representation, z, through the posterior probability distribution, p(z|x).

Then, samples from this latent distribution are decoded back to a high-dimensional signal, through another neural network architecture and the probability distribution p(x|z).

Integration performed with these probability distributions from the Bayesian framework of VAEs is intractable.

As a solution, variational inference is employed to perform learning in these models, whereby a tractable bound on the model evidence is optimized instead of the intractable model evidence itself BID3 .

By design, the output model is set as p(x|z), usually a Bernoulli or a Gaussian probability distribution, depending on whether the target is discrete or continuous, and the prior distribution of the latent space as p(z).

However, the true posterior distribution, p(z|x), remains unknown and is intractable.

To solve this issue, an approximate posterior distribution, q(z|x), is learnt by means of a lower bound on the model evidence, termed the ELBO.

For one data point, x (i) , writing out the Kullback-Leibler divergence between the true and approximate posterior distributions and using its positivity property yields this bound: DISPLAYFORM0 The lower bound on the model evidence, the ELBO, now becomes the cost function used during the training phase of the VAEs.

Over time, the first term shows how the reconstruction loss changes and the second term how far the approximate posterior is to the prior distribution.

The result of inference and the performance of VAEs on reconstructing and generating images heavily depend on the type of bound employed in training.

A significant body of work has been carried out to replace the ELBO with tighter bounds on the model evidence.

On the one hand, starting from an unbiased estimator of the true log-likelihood, the authors of BID0 derive an importance sampling estimate of the model evidence, the IWAE.

This represents one of the tightest bounds of VAEs and has only recently been improved on in BID8 , BID11 .

Increasing the number of importance samples in the IWAE objective, decreases the signal-to-noise-ratio of the gradients, which makes the learning more difficult, as the gradients suffer from a larger level of noise BID8 .

Several strategies are able to correct this issue.

In the first algorithm, MIWAE, the outer expectation of the IWAE objective is approximated with more than one sample, as is the case in the IWAE.

The second algorithm, CIWAE, represents a convex combination of the ELBO and the IWAE bounds and the third algorithm, PIWAE, separately trains the encoder and the decoder networks with different IWAE objectives.

On the other hand, leveraging different divergences between the true and the approximate posterior distributions has lead to diverse bounds on the model evidence.

Starting from the Rényi α-divergence BID9 between such distributions, a family of lower and upper bounds are obtained, parameterized by α BID6 .

However, these lower bounds become competitive with the IWAE, only in the limit α → −∞. In addition, the upper bounds suffer from approximation errors and bias and the means to select the best value of the hyperparameter α is unknown.

Through an importance sampling scheme similar to the one found in the IWAE, these Rényi α bounds are tightened in BID15 .

If the Rényi α-divergence is replaced with the χ 2 divergence, the bound on the model evidence becomes the upper bound CUBO BID1 .

The Rényi α-family of bounds and others lose their interpretability as a reconstruction loss and a Kullback-Leibler divergence term that measures how close the approximate posterior is to the prior distribution.

They remain just a cost function optimized during training.

With different compositions of convex and concave functions, the approaches described above are unified in the K-sample generalized evidence lower bound, GLBO BID11 .

This study generalizes the concept of maximizing the logarithm of the model evidence to maximizing the φ-evidence score, where φ(u) is a concave function that replaces the logarithm.

It allows for great flexibility in the choice of training objectives in VAEs.

One particular setting provides a lower bound, the CLBO, which surpasses the IWAE objective.

The aim of this work is to leverage the theory of q-deformed functions introduced in BID12 , BID13 , BID14 , to derive tighter lower bounds on the model evidence in VAEs.

To this end, our contributions are three-fold: firstly, we derive two novel lower bounds, by replacing the logarithm function in the classical ELBO, BID10 , BID4 , and IWAE bounds, BID0 , BID7 , respectively, with the q-deformed logarithm function.

Values of q < 1.0 yield upper bounds of varying tightness on the classical logarithm function, as illustrated in FIG0 .Secondly, we combine the information given by the upper bound CUBO, BID1 , with the information given by the ELBO and the IWAE, respectively, to obtain a lower bound that is placed between the two.

By the means of their construction, we hypothesize these q-deformed bounds to be closer to the true log-likelihood.

We are able to confirm it in our experiments.

We term our novel lower bounds the qELBO and the qIWAE.Thirdly, the tighteness of the gap between the classical logarithm function and the q-deformed one depends on the value of q, as seen in FIG0 .

Thus, q becomes a hyperparameter of our algorithm.

Since q is a number, we can optimize it efficiently and accurately, using standard optimization algorithms.

By solving for the best q for each data batch, we make q a data-driven hyperparameter, tuned in an adaptive way during training.

With the q-entropy, introduced in BID12 , the author developed the field of nonextensive statistical mechanics, as a generalization of traditional statistical mechanics, centered around the Boltzmann-Gibbs distribution.

The S q entropy provides a generalization of this distribution, which can more accurately explain the phenomena of anomalous physical systems, characterized by rare events.

In the following definitions, the original quantities can be recovered in the limit q → 1.

If k > 0 is a constant, W ∈ N is the total number of possible states of a system and p i the corresponding probabilities, ∀i = 1 : W , then: DISPLAYFORM0 The generalized logarithmic function, termed the q-logarithm, is introduced in BID13 as: DISPLAYFORM1 The Kullback-Leibler divergence is generalized in BID14 to the form DISPLAYFORM2 In order to derive our q-deformed bounds, we replace the logarithm function from the ELBO and IWAE bounds, with its q-deformed version.

By appropriately optimizing the hyperparameter q, we will obtain an upper bound on the ELBO and IWAE, respectively: DISPLAYFORM3 DISPLAYFORM4 Optimization algorithm for q. We train a variational autoencoder with our novel qELBO and qIWAE bounds.

The training procedure and the optimization method for q are identical for both types of q-deformed bounds.

We will describe them in the case of the qELBO.We start the training procedure with an initial value of q = 1.0 − 10 −6 .

For one batch of images, we compute the qELBO lower bound and the CUBO upper bound BID1 , averaged over the batch.

In order to obtain a tighter lower bound, qELBO * , we set a desired value of the cost function at qELBO * = qELBO +τ · (CUBO − qELBO), where, in our experiments, τ ∈ {0.5, 0.75}.By means of the L-BFGS-B optimization method, we find the optimal value q * , such that DISPLAYFORM5 For this task, we employ the scipy optimization package in python.

We apply the gradient descent step on our new, improved, cost function, qELBO * , computed with this optimal value, q * .

We save this value of q for the next batch of images and we repeat the optimization steps described above, for all training batches.

For the experiments conducted on the MNIST dataset BID5 , we use the one-stochastic layer architecture employed in BID0 and in BID6 .

The encoder and the decoder are composed of two deterministic layers, each with 200 nodes, and of a stochastic layer with 50 nodes.

The dimension of the latent space is equal to 50 and the activation functions are the softplus function.

The approximate posterior is modeled as a Gaussian distribution, with a diagonal covariance matrix.

The output model is a Bernoulli distribution for each pixel.

We use the binarized MNIST dataset provided by tensorflow, with 55000 training images and 10000 test images.

The learning rate is fixed at 0.005 and there is no updating schedule.

To implement and test our new algorithms, we modify publicly available code 1 BID6 .

On the benchmark binary MNIST dataset BID5 , we compare our newly derived q-deformed bounds with the ELBO and the IWAE and we show several improvements that we obtained.

On the test set, we report the bounds computed with K number of samples and the true log-likelihood estimated with 5000 importance samples, logp x .

The expectations involved in all of the bounds are estimated with Monte Carlo sampling.

For the ELBO and the qELBO bounds, the expectation is approximated with K number of samples.

The expectation in the standard IWAE is approximated with one sample.

Thus, we will compute the expectation in the qIWAE with one sample, as well.

Here, K refers to the number of importance samples used in the computation of the bound.

In addition, we illustrate the performance of our algorithms on reconstructed binary MNIST test images and on randomly generated ones.

After 3000 epochs of training, the qIWAE(τ =0.5) Figure 3 : Method: qVAE(τ = 0.5) with K=50 samples.

From left to right: original binary MNIST test images, reconstructed and randomly generated ones.algorithm, with the bound estimated with K=50 samples, gives the best result on the importance sampling estimate of the true log-likelihood, very close to the one given by the standard IWAE.

Moreover, the q-deformed bound is much closer to the estimated true value, than is the IWAE bound.

We observe this behaviour for all the q-deformed bounds.

This implies that, during training, optimizing the q-deformed bounds provides a cost function that is a more accurate approximation of the model evidence.

Although the q-deformed ELBO does not outperform the standard IWAE, we can see significant improvements over the traditional ELBO, in all the test cases.

A large decrease in the value of the bound is present for all the qELBO variants, more pronounced in the large sample regime.

We addressed the challenging task of deriving tighter bounds on the model evidence of VAEs.

Significant research effort has gone in this direction, with several major contributions having been developed so far, which we reviewed in the introduction.

We leveraged the q-deformed logarithm function, to explore other ways of tightening the lower bounds.

As well as improvements in the estimated true log-likelihood, we found that the q-deformed bounds are much closer to the estimated true log-likelihood, than the classical bounds are.

Thus, training with our novel bounds as the cost function may increase the learning ability of VAEs.

Through the preliminary experiments we have conducted so far, we have achieved our goal.

They show that our approach has merit and that this direction of research is worth pursuing in more depth, to produce more accurate bounds and to study their impact on the performance of VAEs.

As future work, similarly to BID8 , we plan to investigate how the tightening the ELBO and the IWAE influences the learning process and affects the gradients and the structure of the latent space, compared with the classical case.

In addition, we plan to explore different optimization strategies for q and to study its role in achieving tighter bounds.

We will also apply our q-deformed bounds, to investigate the disentanglement problem in VAEs, see for example BID2 .

The research question addressed here is how different bounds change the structure of the latent space, to provide better or worse disentanglement scores.

Finally, we would also like to test our novel bounds on all the major benchmark datasets used for assessing the performance of VAEs and compare them with other state-of-the-art bounds on the model evidence.

<|TLDR|>

@highlight

Using the q-deformed logarithm, we derive tighter bounds than IWAE, to train variational autoencoders.