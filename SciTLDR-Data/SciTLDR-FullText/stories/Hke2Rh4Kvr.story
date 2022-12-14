A deep generative model is a powerful method of learning a data distribution, which has achieved tremendous success in numerous scenarios.

However, it is nontrivial for a single generative model to faithfully capture the distributions of the complex data such as images with complicate structures.

In this paper, we propose a novel approach of cascaded boosting for boosting generative models, where meta-models (i.e., weak learners) are cascaded together to produce a stronger model.

Any hidden variable meta-model can be leveraged as long as it can support the likelihood evaluation.

We derive a decomposable variational lower bound of the boosted model, which allows each meta-model to be trained separately and greedily.

We can further improve the learning power of the generative models by combing our cascaded boosting framework with the multiplicative boosting framework.

The past decade has witnessed tremendous success in the field of deep generative models (DGMs) in both unsupervised learning (Goodfellow et al., 2014; Kingma & Welling, 2013; Radford et al., 2015) and semi-supervised learning (Abbasnejad et al., 2017; Kingma et al., 2014; Li et al., 2018) paradigms.

DGMs learn the data distribution by combining the scalability of deep learning with the generality of probabilistic reasoning.

However, it is not easy for a single parametric model to learn a complex distribution, since the upper limit of a model's ability is determined by its fixed structure.

If a model with low capacity was adopted, the model would be likely to have a poor performance.

Straightforwardly increasing the model capacity (e.g., including more layers or more neurons) is likely to cause serious challenges, such as vanishing gradient problem (Hochreiter et al., 2001 ) and exploding gradient problem (Grosse, 2017 ).

An alternative approach is to integrate multiple weak models to achieve a strong one.

The early success was made on mixture models (Dempster et al., 1977; Figueiredo & Jain, 2002; Xu & Jordan, 1996) and product-of-experts (Hinton, 1999; .

However, the weak models in such work are typically shallow models with very limited capacity.

Recent success has been made on boosting generative models, where a set of meta-models (i.e., weak learners) are combined to construct a stronger model.

In particular, Grover & Ermon (2018) propose a method of multiplicative boosting, which takes the geometric average of the meta-model distributions, with each assigned an exponentiated weight.

This boosting method improves performance on density estimation and sample generation, compared to a single meta-model.

However, the boosted model has an explicit partition function, which requires importance sampling (Rubinstein & Kroese, 2016) for an estimation.

In general, sampling from the boosted model is conducted based on Markov chain Monte Carlo (MCMC) method (Hastings, 1970) .

As a result, it requires a high time complexity of likelihood evaluation and sample generation.

Rosset & Segal (2003) propose another method of additive boosting, which takes the weighted arithmetic mean of meta-models' distributions.

This method can sample fast, but the improvement of performance on density estimation is not comparable to the multiplicative boosting, since additive boosting requires that the expected log-likelihood and likelihood of the current meta-model are better-or-equal than those of the previous boosted model (Grover & Ermon, 2018) , which is difficult to satisfy.

In summary, it is nontrivial for both of the previous boosting methods to balance well between improving the learning power and keeping the efficiency of sampling and density estimation.

To address the aforementioned issues, we propose a novel boosting framework, called cascaded boosting, where meta-models are connected in cascade.

The framework is inspired by the greedy layer-wise training algorithm of DBNs (Deep Belief Networks) (Bengio et al., 2007; Hinton et al., 2006) , where an ensemble of RBMs (Restricted Boltzmann Machines) (Smolensky, 1986) are converted to a stronger model.

We propose a decomposable variational lower bound, which reveals the principle behind the greedy layer-wise training algorithm.

The decomposition allows us to incorporate any hidden variable meta-model, as long as it supports likelihood evaluation, and train these meta-models separately and greedily, yielding a deep boosted model.

Finally, We demonstrate that our boosting framework can be integrated with the multiplicative boosting framework (Grover & Ermon, 2018) , yielding a hybrid boosting with an improved learning power of generative models.

To summary, we make the following contributions:

??? We propose a boosting framework to boost generative models, where meta-models are cascaded together to produce a stronger model.

??? We give a decomposable variational lower bound of the boosted model, which reveals the principle behind the greedy layer-wise training algorithm.

??? We finally demonstrate that our boosting framework can be extended to a hybrid model by integrating it with the multiplicative boosting models, which further improves the learning power of generative models.

In subsection 2.1, we review the current multiplicative boosting (Grover & Ermon, 2018) .

Then, we present our cascaded boosting.

We first figure out how to connect meta-models, and then propose our boosting framework with its theoretical analysis.

Afterwards, we discuss the convergence of our cascaded boosting.

Grover & Ermon (2018) introduced multiplicative boosting, which takes the geometric average of meta-models' distributions, with each assigned an exponentiated weight ?? i as

where M i (x) (0 ??? i ??? n) are distributions of meta-models, which are required to support likelihood evaluation, P n (x) is the distribution of the boosted model and Z n is the partition function.

The first meta-model M 0 is trained on the empirical data distribution p D which is defined to be uniform over the dataset D. The other meta-models

where

with ?? i ??? [0, 1] being the hypermeter.

Grover & Ermon (2018) show that the expected log-likelihood of the boosted model P i over the dataset D will not decrease (i.e.,

The multiplicative boosting succeeds in improving the learning power of generative models.

Compared to an individual meta-model, it has better performance on density estimation and generates samples of higher quality.

However, importance sampling and MCMC are required to evaluate the partition function Z n and generate samples respectively, which limits its application in occasions requiring fast density estimation and sampling.

To overcome these shortcomings, we propose our cascaded boosting framework.

In multiplicative boosting, distributions of meta-models are connected by multiplication, leading to the troublesome partition function.

To overcome this problem, we connect meta-models in cascade.

Suppose we have n meta-models

, where x i is the visible variable, h i is the hidden variable and m i (x i , h i ) is their joint distribution.

These meta-models can belong to different families (e.g. RBMs and VAEs (Variational Autoencoders) (Kingma & Welling, 2013) ), as long as they have hidden variables and support likelihood evaluation.

We replace x i with h i???1 and connect metamodels in a top-down style to construct a boosted model as

where h 0 = x and 1 ??? k ??? n. This formulation avoids the troublesome partition function and we can sample from the boosted model in a top-down style using simple ancestral sampling.

The boosted model allows us to generate samples hereby.

Then, we build the approximation of the posterior distribution, which allows us to do inference.

We connect meta-models in a bottom-up style to construct the approximation of the posterior distribution as

where

when j < k, we can omit the subscript, thereby writing q k as q. The approximation of the posterior distribution makes an assumption of conditional independence:

, thereby leading to the advantage that we don't need to re-infer the whole boosted model after incorporating a new meta-model m k : we only need to infer

Supposing D is the training set, we give a decomposable variational lower bound

, and q(h 1 , ?? ?? ?? , h k???1 |x) be the approximate posterior constructed from

, then we have:

where

Proof: see Appendix A.

is the difference between the marginal likelihood of the observable variable of m i and the hidden variable of m i???1 .

When k = 1, there is only one meta-model and the lower bound is exactly equal to the marginal likelihood of the boosted model.

So the lower bound is tight when k = 1.

Based on the initially tight lower bound, we can further promote it by optimizing these decomposed terms sequentially, yielding the greedy layer-wise training algorithm, as discussed in subsection 2.4.

The difference between

Algorithm 1 Cascaded Boosting

To ensure the lower bound grows with m k incorporated, we only need to ensure l k is positive.

When we train the meta-model m k , we first fix rest meta-models

is constant, and then train m k by maximizing

As a result, we can train each meta-model separately and greedily, as outlined in Alg.

1.

are arbitrarily powerful learners, we can derive the non-decreasing property of the decomposable lower bound, as given in Theorem 2.

Theorem 2.

When {m i } n i=2 are arbitrarily powerful learners (i.e., m i is able to model any distribution), we have L 1 ??? L 2 ??? ?? ?? ?? ??? L n during the greedy layer-wise training.

Proof.

During the kth (2 ??? k ??? n) round of Alg.

1,

In practice, l k (m 1 , ?? ?? ?? , m k ) is likely to be negative under the following three cases:

??? m k is not well trained.

In this case, l k (m 1 , ?? ?? ?? , m k ) is very negative, which indicates us to tune hyperparameters of m k and retrain this meta-model.

will be close to zero, and we can either keep training by incorporating more powerful meta-models or just stop training.

??? The lower bound converges.

In this case, the lower bound will stop growing even if more meta-models are incorporated, which will be further discussed in subsection 2.5

For models with m k (h k???1 ) initialized from m k???1 (h k???1 ), such as DBNs (Hinton et al., 2006) ,

It's impossible for the decomposable lower bound to grow infinitely.

After training m k , if

] is maximized, then the lower bound will stop growing even if we keep incorporating more meta-models.

We call this phenomenon convergence of the boosted model, which is formally described in Theorem 3.

Proof: see Appendix B.

It indicates that it's unnecessary to incorporate meta-models as much as possible.

To help judge whether the boosted model has converged, a necessary condition is given in Theorem 4.

Proof: see Appendix B.

We can use c k to help us judge whether the boosted model has converged after training m k .

For meta-models such as VAEs, m k (h k ) is the standard normal distribution and

is analytically solvable, leading to a simple estimation of c k .

We can further consider a hybrid boosting by integrating our cascaded boosting with the multiplicative boosting.

It is not difficult to implement: we can think of the boosted model produced by our method as the meta-model for multiplicative boosting.

An open problem for hybrid boosting is to determine what kind of meta-models to use and how meta-models are connected, which is closely related to the specific dataset and task.

Here we introduce some strategies for this problem.

For cascaded connection, if the dataset can be divided to several categories, it is appropriate to use a GMM (Gaussian Mixture Model) (Smolensky, 1986) as the top-most meta-model.

Other metamodels can be selected as VAEs (Kingma & Welling, 2013) or their variants (Burda et al., 2015; S??nderby et al., 2016) .

There are three reasons for this strategy: (1) the posterior of VAE is much simpler than the dataset distribution, making a GMM enough to learn the posterior; (2) the posterior of a VAE is likely to consist of several components, with each corresponding to one category, making a GMM which also consists of several components suitable; (3) Since m k???1 (h k???1 ) is a standard Gaussian distribution when m k???1 is a VAE, when m k (h k???1 ) is a GMM, which covers the standard Gaussian distribution as a special case, we can make sure that Equation 7 will not be negative after training m k .

For multiplicative connection, each meta-model should have enough learning power for the dataset, since each meta-model is required to learn the distribution of the dataset or the reweighted dataset.

If any meta-model fails to learn the distribution, the performance of the boosted model will be harmed.

In subsection 4.5, we give a negative example, where a VAE and a GMM are connected by multiplication and the overall performance is extremely bad.

We now present experiments to verify the effectiveness of our method.

We first validate that the non-decreasing property of the decomposable lower bound holds in practice.

Next, we give results of boosting advanced models to show that our method can be used as a technique to further promote the performance of state-of-the-art models.

Then, we compare our method with naively increasing model capacity.

Finally, we make a comparison between different generative boosting methods.

We do experiments on static binarized mnist (LeCun & Cortes, 2010) , which contains 60000 training data and 10000 testing data, as well as the more complex celebA dataset (Liu et al., 2015) , which contains 202599 face images, with each first resized to 64 ?? 64.

The meta-models we use include RBMs (Smolensky, 1986) , GMMs (Reynolds, 2015) , VAEs (Kingma & Welling, 2013) , ConvVAEs (i.e., VAEs with convolutional layers), IWAEs (Burda et al., 2015) , and LVAEs (S??nderby et al., 2016) , with their architectures given in Appendix C. The marginal likelihoods of RBMs are estimated using importance sampling, and the marginal likelihoods of VAEs and their variants are estimated using the variational lower bound (Kingma & Welling, 2013) .

All experiments are conducted on one 2.60GHz CPU and one GeForce GTX TITAN X GPU.

The non-decreasing property (Theorem 2) of decomposable lower bound (Equation 5 ) is the theoretical guarantee of the greedy layer-wise training algorithm (subsection 2.4).

We validate that the non-decreasing property also holds in practice by using RBMs and VAEs as meta-models.

We evaluate the decomposable lower bound on 4 combinations of RBMs and VAEs on static binarized mnist.

Since the stochastic variables in RBMs are discrete, we put RBMs at bottom and put VAEs at top.

For each combination, we evaluate the lower bound (Equation 5) at different k (1 ??? k ??? 6) on both training and testing dataset.

As shown in Figure 1 , both the training and testing curves of the decomposable lower bound present the non-decreasing property.

We also notice a slight drop at the end of these curves when incorporating VAEs, which can be explained by the convergence (subsection 2.5): if E D E q(h1,?????? ,h k???1 |x) [logm k (h k???1 )] is maximized after training m k , then the lower bound will stop growing even if we keep incorporating more meta-models.

The first four meta-models are RBMs and the rest are VAEs.

The lower bound grows as the first two RBMs are incorporated, while the incorporation of next two RBMs doesn't help promote the lower bound.

We further improve the lower bound by adding two VAEs.

(4): All meta-models are RBMs.

After incorporating two RBMs, the lower bound becomes stable.

Besides, the quality of generated samples of the boosted model also has a non-decreasing property, which is consistent with the non-decreasing property of the decomposable lower bound, as shown in Table 1 .

Furthermore, we can get some evidence about the convergence of the boosted model from c k (Theorem 4).

We see that c k is the smallest when k = 3, which indicates that the boosted model is likely to converge after incorporating 3 VAEs and the last VAE is redundant.

Table 1 : Samples generated from boosted models consisting of k VAEs (1 ??? k ??? 4).

The lower bounds (Equation 5) of these boosted models are also given.

c k (Theorem 4) is an indicator to help judge whether a boosted model has converged, where a small one supports convergence.

For both mnist and celebA, the quality of generated samples has a non-decreasing property.

Besides, c k is the smallest when k = 3, which indicates that the boosted model is likely to converge after incorporating 3 VAEs and the last VAE is redundant.

We show that our cascaded boosting can be used as a technique to further promote the performance of state-of-the-art models.

We choose ConvVAE (i.e., VAE with convolutional layers), LVAE (S??nderby et al., 2016) , IWAE (Burda et al., 2015) as advanced models, which represent current state-of-art methods.

We use one advanced model and one GMM (Reynolds, 2015) to construct a boosted model, with the advanced model at the bottom and the GMM at the top.

The result is given in Table 2 .

We see that the lower bound of each advanced model is further promoted by incorporating a GMM, at the cost of a few seconds.

The performance improvement by incorporating a GMM is theoretically guaranteed: since m 1 (h 1 ) is a standard Gaussian distribution in above four cases considered in Table 2 , when m 2 (h 1 ) is a GMM, which covers the standard Gaussian distribution as a special case, we can ensure that l 2 (Equation 7) will not be negative after training m 2 .

Besides, the dimension of hidden variable h 1 is much smaller than the dimension of observable variable h 0 for VAEs and their variants, and thus the training of m 2 only requires very little time.

We compare our cascaded boosting with the method of naively increasing model capacity.

The conventional method of increasing model capacity is either to add more deterministic hidden layers or to increase the dimension of deterministic hidden layers, so we compare our boosted model (Boosted VAEs) with a deeper model (Deeper VAE) and a wider model (Wider VAE).

The Deeper VAE has ten 500-dimensional deterministic hidden layers; the Wider VAE has two 2500-dimensional deterministic hidden layers; the Boosted VAEs is composed of 5 base VAEs, each of them has two 500-dimensional deterministic hidden layers.

As a result, all the three models above have 5000 deterministic hidden units.

Figure 2 shows the results.

Wider VAE has the highest lower bound, but its generated digits are usually undistinguishable.

Meanwhile, the Deeper VAE is able to generate distinguishable digits, but some digits are rather blurred and its lower bound is the lowest one.

Only the digits generated by Boosted VAEs are both distinguishable and sharp.

Since straightforwardly increasing the model capacity is likely to cause serious challenges, such as vanishing gradient problem (Hochreiter et al., 2001 ) and exploding gradient problem (Grosse, 2017) , it often fails to achieve the desired results on improving models' learning power.

Our boosting method avoids these challenges by leveraging the greedy layer-wise training algorithm.

We make a comparison between our cascaded boosting, multiplicative boosting and hybrid boosting.

The result is given in Table 3 .

The hybrid boosting produces the strongest models, but the time cost of density estimation and sampling is high, due to the troublesome partition function.

Our cascaded boosting allows quick density estimation and sampling, but its boosted models are not as strong as the hybrid boosting.

It is also worth note that the multiplicative connection of one VAE and one GMM produces a bad model, since the learning power of a GMM is too weak for directly learning the distribution of mnist dataset and the training time of a GMM is long for high dimensional data.

Table 3 : Comparison between different boosting methods on mnist.

The '+' represents the cascaded connection and the ' ' represents the multiplicative connection.

The density log p(x) is estimated on the test set, using Equation 5 for cascaded connection and importance sampling for multiplicative connection respectively.

The sampling time is the time cost for sampling 10000 samples.

Deep Belief Networks.

Our work is inspired by DBNs (Hinton et al., 2006) .

A DBN has a multilayer structure, whose basic components are RBMs (Smolensky, 1986) .

During training, each RBM is learned separately, and stacked to the top of current structure.

It is a classical example of our cascaded boosting, since a group of RBMs are cascaded to produce a stronger model.

Our decomposable variational lower bound reveals the principle behind the training algorithm of DBNs: since

, we can make sure that l k ??? 0, assuring the non-decreasing property of the decomposable lower bound (Equation 5).

Deep Latent Gaussian Models.

DLGMs (Deep Latent Gaussian Models) are deep directed graphical models with multiple layers of hidden variables (Burda et al., 2015; .

The distribution of hidden variables in layer k conditioned on hidden variables in layer k+1 is a Gaussian distribution.

introduce an approximate posterior distribution which factorises across layers.

Burda et al. (2015) introduce an approximate posterior distribution which is a directed chain.

Our work reveals that the variational lower bound of Burda et al. (2015) can be further decomposed and optimized greedily and layer-wise.

Other methods of boosting generative models.

Methods of boosting generative models have been explored.

Previous work can be divided into two categories: sum-of-experts (Figueiredo & Jain, 2002; Rosset & Segal, 2003; Tolstikhin et al., 2017) , which takes the arithmetic average of meta-models' distributions, and product-of-experts (Hinton, 2002; Grover & Ermon, 2018) , which takes the geometric average of meta-models' distributions.

We propose a framework for boosting generative models by cascading meta-models.

Any hidden variable meta-model can be incorporated, as long as it supports likelihood evaluation.

The decomposable lower bound allows us to train meta-models separately and greedily.

Our cascaded boosting can be integrated with the multiplicative boosting.

In our experiments, we first validate that the non-decreasing property of the decomposable variational lower bound (Equation 5) holds in practice, and next further promote the performance of some advanced models, which represent state-ofthe-art methods.

Then, we show that our cascaded boosting has better performance of improving models' learning power, compared with naively increasing model capacity.

Finally, we compare different generative boosting methods, validating the ability of the hybrid boosting in further improving learning power of generative models.

A PROOF OF THEOREM 1

Proof.

Using q(h 1 , ?? ?? ?? , h k???1 |x) as the approximate posterior, we have a variational lower bound

Thus, the lower bound is equal to:

Thus,

Take the expection with respect to dataset D, we have

B PROOF OF THEOREM 3 AND THEOREM 4

where n is the number of meta-models.

Since

we can omit the subscript, thereby writing q i as q.

For any j ??? [k + 1, n]

??? Z and any m k+1 , m k+2 , ?? ?? ?? , m j , given i (k + 1 ??? i ??? j), we have Let q(h k , ?? ?? ?? , h j???1 |h k???1 ) be the approximate posterior of p j (h k???1 , ?? ?? ?? , h j???1 ), according to Theorem 1, we have

we have

The architectures of VAEs, ConvVAEs, IWAEs and LVAEs are given in this part.

All VAEs have two deterministic hidden layers for both generation, and inference and we add batch normalization layers (Ioffe & Szegedy, 2015; S??nderby et al., 2016 ) after deterministic hidden layers.

The dimension of deterministic hidden layers is set to 500 and 2500, and the dimension of stochastic hidden variables is set to 20 and 100, for experiments on mnist and celebA respectively.

The ConvVAE has one 500-dimensional deterministic hidden layer and one 50-dimensional stochastic hidden variable, with four additional convolutional layers LeCun et al. (1998) .

All convolutional layers have a kernel size of 4 ?? 4 and a stride of 2.

Their channels are 32, 64, 128 and 256 respectively.

We add batch normalization layers after deterministic hidden layers.

The IWAE has two 500-dimensional deterministic hidden layers and one 50-dimensional stochastic hidden variable.

The number of importance sampling is set to 5 and 10.

The LVAE has four 1000-dimensional deterministic hidden layers and two 30-dimensional stochastic hidden variables.

We add batch normalization layers after deterministic hidden layers.

@highlight

Propose an approach for boosting generative models by cascading hidden variable models

@highlight

This paper proposed a novel approach of cascaded boosting for boosting generative models which allows each each meta-model to be trained separately and greedily.