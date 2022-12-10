We investigate a variant of variational autoencoders where there is a superstructure of discrete latent variables on top of the latent features.

In general, our superstructure is a tree structure of multiple super latent variables and it is automatically learned from data.

When there is only one latent variable in the superstructure, our model reduces to one that assumes the latent features to be generated from a Gaussian mixture model.

We call our model the latent tree variational autoencoder (LTVAE).

Whereas previous deep learning methods for clustering produce only one partition of data, LTVAE produces multiple partitions of data, each being given by one super latent variable.

This is desirable because high dimensional data usually have many different natural facets and can be meaningfully partitioned in multiple ways.

Clustering is a fundamental task in unsupervised machine learning, and it is central to many datadriven application domains.

Cluster analysis partitions all the data into disjoint groups, and one can understand the structure of the data by examining examples in each group.

Many clustering methods have been proposed in the literature BID0 , such as k-means BID18 , Gaussian mixture models BID5 and spectral clustering BID30 .

Conventional clustering methods are generally applied directly on the original data space.

However, it is challenging to perform cluster analysis on high dimensional and unstructured data BID26 , such as images.

It is not only because the dimensionality is high, but also because the original data space is too complex to interpret, e.g. there are semantic gaps between pixel values and objects in images.

Recently, deep learning based clustering methods have been proposed that simultanously learn nonlinear embeddings through deep neural networks and perform cluster analysis on the embedding space.

The representation learning process learns effective high-level representations from high dimensional data and helps the cluster analysis.

This is typically achieved by unsupervised deep learning methods, such as restricted Boltzmann machine (RBM) BID11 , autoencoders (AE) BID28 , variational autoencoders (VAE) BID16 , etc.

Previous deep learning based clustering methods BID33 BID10 BID14 BID34 ) assume one single partition over the data and that all attributes define that partition.

In real-world applications, however, the assumptions are usually not true.

High-dimensional data are often multifaceted and can be meaningfully partitioned in multiple ways based on subsets of attributes BID4 .

For example, a student population can be clustered in one way based on course grades and in another way based on extracurricular activities.

Movie reviews can be clustered based on both sentiment (positive or negative) and genre (comedy, action, war, etc.) .

It is challenging to discover the multi-facet structures of data, especially for high-dimensional data.

To resolve the above issues, we propose an unsupervised learning method, latent tree variational autoencoder (LTVAE) to learn latent superstructures in variational autoencoders, and simultaneously perform representation learning and structure learning.

LTVAE is a generative model, where the data is assumed to be generated from latent features through neural networks, while the latent features themselves are generated from tree-structured Bayesian networks with another level of latent variables as shown in Fig. 1 .

Each of those latent variables defines a facet of clustering.

The proposed method automatically selects subsets of latent features for each facet, and learns the dependency structure among different facets.

This is achieved through systematic structure learning.

Consequently, LTVAE is able to discover complex structures of data rather than one partition.

We also propose efficient learning algorithms for LTVAE with gradient descent and Stepwise EM through message passing.

The rest of the paper is organized as follows.

The related works are reviewed in Section 2.

We introduce the proposed method and learning algorithms in Section 3.

In Section 4, we present the empirical results.

The conclusion is given in Section 5.

Clustering has been extensively studied in the literature in many aspects BID0 .

More complex clustering methods related to structure learning using Bayesian nonparametrics have been proposed, like Dirichlet Process , Hierarchical Dirichlet Process (HDP) .

However, those are with conventional clustering methods that apply on raw data.

Recently, deep learning based clustering methods have drawn more and more attention.

A simple two-stage approach is to first learn low-dimensional embeddings using unsupervised feature learning methods, and then perform cluster analysis on the embeddings.

However, without any supervision, the representation learning do not necessarily reveal the true cluster structure of the data.

DEC BID33 ) is a method that simultaneously learns feature representations and cluster assignments through deep autoencoders.

It gradually improves the clustering by driving the deep network to learn a better mapping.

Improved Deep Embedded Clustering BID10 improves DEC by keeping the decoder network and adding reconstruction loss to the original clustering loss in DEC.

Variational deep embedding BID14 ) is a generative method that models the data generative process using a Gaussian mixture model combined with a VAE, and also performs joint learning of representations and clustering.

Similarly, GMVAE BID7 performs joint learning of a GMM and a VAE, but instead generates the mixture components through neural networks.

Deep clustering network (DCN) BID34 is another one that jointly learns an autoencoder and performs k-means clustering.

These joint learning methods consistently achieve better clustering results than conventional ones.

The method proposed in BID35 uses convolutional neural networks and jointly learns the representations and clustering in a recurrent framework.

All these methods assume flat partitions over the data, and do not attempt the structure learning issue.

An exception is hierarchical nonparametric variational autoencoders proposed in BID9 .

It uses nCRP as the prior for VAE to allow infinitely deep and branching tree hierarchy structure and focuses on learning hierarchy of concepts.

However, it is still one partition over the data, only that the partitions in upper levels are more general partitions, while those in lower levels more fine-grained.

Different from it, our work focuses on multifacets of clustering, for example, the model could make one partition based on identity of subjects, while another partition based on pose.

In this section, we present the proposed latent tree variational autoencoder and the learning algorithms for joint representation learning and structure learning for multidimensional clustering.

Deep generative models assume that data x is generated from latent continuous variable z through some random process.

The process consists of two steps: (1) a value z is generated from some prior distribution p(z); (2) the observation x is generated from the conditional distribution p θ (x|z), which is parameterized through deep neural networks.

Thus, it defines the joint distribution between

Generation Network DISPLAYFORM0 Figure 2: Inference and gradient through message passing.

Solid-arrows denote collecting message, and dashed-arrows denote distributing message.observation x and latent variable z: DISPLAYFORM1 This process is hidden from our view, and we learn this process by maximizing the marginal loglikelihood p(x) over the parameters θ and latent variable z from data.

After the learning, the latent variable z can be regarded as the deep representations of x since it captures the most relevant information of x. Thus, the learning process is also called representation learning.

In order to learn the latent structure of z, for example multidimensional cluster structure, we in- .

This essentially forms a Bayesian network.

And if we restrict the network to be treestructured, the z and Y together form a latent tree model BID37 BID21 BID19 BID20 BID38 with z being the observed variables and Y being the latent variables.

For multidimensional clustering, each latent variable Y is taken to be a discrete variable, where each discrete state y of Y defines a cluster.

Each latent variable Y thus defines a facet partition over the data based on subset of attributes and multiple Y 's define multiple facets.

Given a value y of Y , z b follows a conditional Gaussian distribution P (z b |y) = N (µ y , Σ y ) with mean vector µ y and covariance matrix Σ y .

Thus, each z b and its parent constitute a Gaussian mixture model (GMM).

Suppose the parent of a node is denoted as π(·), the maginal distribution of z is defined as follows DISPLAYFORM2 which sums over all possible combinations of Y states.

As a matter of fact, a GMM is a Gaussian LTM that has only one latent variable connecting to all observed variables.

Let the latent structure of Y be S, defining the number of latent variables in Y, the number of discrete states in each variable Y and the connectivity structure among all variables in z and Y. And let the parameters for all conditional probabilities in the latent structure be Θ. Both the latent structure S and the latent parameters Θ are unknown.

We aim to jointly learn data representations and the latent structure.

The proposed LTVAE model is shown in Fig. 1 .

The latent structure S are automatically learned from data and will be discussed in a later section.

Due to the existence of the generation network, the inference of the model is intratable.

Instead, we do amortized variational inference for the latent variable z by introducing an inference network BID16 and define an approximate posterior q φ (z|x).

The evidence lower bound (ELBO) L ELBO of the marginal loglikelihood of the data given (S, Θ) is: DISPLAYFORM3 where log y p S (z, y; Θ) is the marginal loglikelihood of the latent variable z under the latent tree model, and H[·] is the entropy.

The conditional generative distribution p θ (x|z) could be a Gaussian distribution if the input data is real-valued, or a Bernoulli distribution if binary, parameterized by the generation network.

Using Monte Carlo sampling, the ELBO can be asymptotically estimated by DISPLAYFORM4 where DISPLAYFORM5 can be computed analytically if we choose the form of DISPLAYFORM6 , where J is the dimensionality of z.

Furthermore, the marginal loglikelihood log y p S (z (i) , y; Θ) can be computed efficiently through message passing.

Message passing is an efficient algorithm for inference in Bayesian networks BID17 BID22 .

In message passing, we first build a clique tree using the factors in the defined probability density.

Because of the tree structure, each z b along with its parent form a clique with the potential ψ(z b , y) being the corresponding conditional distribution.

This is illustrated in Fig. 2 .

With the sampled z (i) , we can compute the message ψ (y) by absorbing the evidence from z. During collecting message phase, the message ψ (y) are sent towards the pivot.

After receiving all messages, the pivot distributes back messages towards all z. Both the posterior of Y and the marginal loglikelihood of z (i) thus can be computed in the final normalization step.

In this section, we propose efficient learning algorithms for LTVAE through gradient descent and stepwise EM with message passing.

Given the latent tree model (S, Θ), the parameters of neural networks can be efficiently optimized through stochastic gradient descent (SGD).

However, in order to learn the model, it is important to efficiently compute the gradient of the marginal loglikelihood log p S (z; Θ) from the latent tree model, the third term in Eq. 4.

Here, we propose an efficient method to compute gradient through message passing.

Let z b be the variables that we want to compute gradient with respect to, and let Y b be the parent node.

The marginal loglikelihood of full z can be written as DISPLAYFORM0 where f (y b ) is the collection of all the rest of the terms not containing z b .

The gradient g z b of the marginal loglikelihood log p S (z; Θ) w.r.t z b thus can be computed as DISPLAYFORM1 where p(y b |z) is the posterior probability of y b and can be computed efficiently with message passing as described in the previous section.

The detailed derivation is in Appendix E. DISPLAYFORM2 With the efficient computation of the third term in Eq. 4 and its gradient w.r.t z through message passing, the parameters of inference network and generation network can be efficiently optimized through SGD.In order to jointly learn the parameters of the latent tree Θ, we propose Stepwise EM algorithm based on mini-batch of data.

Specifically, we maximize the third term in Eq. 4, i.e. the marginal loglikelihood of z under the latent tree.

In the Stepwise E-step, we compute the distributions P (y, y |z, θ (t−1) ) and P (y|z, θ (t−1) ) for each latent node Y and its parent Y .

In the Stepwise M-step, we estimate the new parameter θ (t) .

Let s(z, y) be a vector the sufficient statistics for a single data case.

Lets = E p S (y|z;Θ) [s(z, y)] be the expected sufficient statistics for the data case, where the expectation is w.r.t the posterior distribution of y with current parameter.

And let µ = N i=1s i be the sum of the expected sufficient statistics.

The update of the parameter Θ is performed as follows: DISPLAYFORM3 where η is the learning rate and l is the complete data loglikelihood.

Each iteration of update of LTVAE thus is composed of one iteration of gradient descent update for the neural network parameters and one iteration of Stepwise EM update for the latent tree model parameters with a mini-batch of data.

For the latent structure S, there are four aspects need to determine: the number of latent variables, the cardinalities of latent variables, the connectivities among variables.

We aim at finding the model m * that maximizes the BIC score BID25 BID17 : DISPLAYFORM0 where θ * is the MLE of the parameters and d(m) is the number of independent parameters.

The first term is known as the likelihood term.

It favors models that fit data well.

The second term is known as the penalty term.

It discourages complex models.

Hence, the BIC score provides a tradeoff between model fit and model complexity.

To this end, we perform systematic searching to find a structure with a high BIC score.

We use the hill-climing algorithm to search for m * as in BID21 , and define 7 search operators: node introduction (NI) and node deletion (ND) to introduce new latent nodes and delete existing nodes, state introduction (SI) and state deletion (SD) to add a new state and delete a state for existing nodes, node relocation (NR) to change links of existing nodes, pouching (PO) and unpouching (UP) operators to combine nodes into a single node and separate variables from a node..

The structure search operators are shown in FIG1 .

Each operator produces a set of candidates from existing structure, and the best candidate is picked if it improves the previous one.

To reduce the number of possible search candidates, we first perform SI, NI and PO to expand the structure and pick the best model.

Then we perform NR to adjust the best model.

Finally, we perform UP, ND and SD to simplify the current best structure and pick the best one.

Acceleration techniques BID22 are adopted that make the algorithm efficient enough.

The structure learning is performed iteratively together with the parameter learning of neural networks.

The overall learning algorithm is illustrated in Algorithm 1.

Starting from a pretrained model, we iteratively improve the structure and parameters of latent tree model while learning the representations of data through neural network in a greedy manner.

Using current structure S t as the initial structure, Published as a conference paper at ICLR 2019

Compute log pS (z; Θ) and ∂ log p S (z) ∂z from Eq. 5 and 7 Compute ELBO from Eq. 4 θ, φ ← Back-propagation and SGD step DISPLAYFORM0 we search for a better model.

With new latent tree model, we optimize for a better representation until convergence.

We first demonstrate the effectiveness of the proposed method through synthetic data.

Assume that the data points have two facets Y 1 and Y 2 , where each facet controlls a subset of attributes (e.g. two-dimensional domain) and defines one partition over the data.

This four-dimensional domain z = {z 1 , z 2 , z 3 , z 4 } is a latent representation which we do not observe.

What we observe is x ∈ R 100 that is obtained via the following non-linear transformation: DISPLAYFORM0 where W ∈ R 10×4 and U ∈ R 100×10 are matrices whose entries follow the zero-mean unit-variance i.i.d.

Gaussian distribution, σ(·) is a sigmoid function to introduce nonlinearity.

The generative model is shown in FIG2 .

We define two clusters in facet Y 1 and two clusters in facet Y 2 , and generate 5,000 samples of x. Under the above generative model, recovering the two facets Y 1 and Y 2 structure and the latent z domain from the observation of x seems very challenging.

All previous DNN-based methods (AE+GMM, DEC, DCN, etc.) are only able to discover one-facet of clustering (i.e. one partition over the data), and none of these is applicable to solve such a multidimensional clustering problem.

FIG2 shows the results of the proposed method.

As one can see, the LTVAE successfully discovers the true superstructure of Y 1 and Y 2 .

The 2-d plot of z 1 and z 2 shows the separable latent space clusters under facet Y 1 , and it matches the ground-truth cluster assignments.

Additionally, the 2-d plot of z 3 and z 4 shows another separable clusters under facet Y 2 , and it also matches the ground-truth cluster assignments well in the other facet.

We evaluate the proposed LTVAE model on two image datasets and two other datasets, and compare it against other deep learning based clustering algorithms, including two-stage methods, AE+GMM and VAE+GMM, which first learn AE/VAE BID16 ) models then construct a GMM on top of them, and joint learning methods, DEC BID33 and DCN BID34 .

The datasets include MNIST, STL-10, Reuters BID33 BID14 and the Heterogeneity Human Activity Recognition (HHAR) dataset.

When evaluating the clustering performance, for fair of comparison, we follow previous works BID33 BID34 and use the network structures of d−500−500−2000−10 for the encoder network and 10−2000− 500 − 500 − d for the decoder network for all datasets, where d is the data-space dimension, which varies among datasets.

All layers are fully-connected.

We follow the pretraining procedure as in BID33 .

We first perform greedy layer-wise pretraining in denoising autoencoder manner, then stack all layers to form deep autoencoder.

The deep autoencoder is further finetuned to minimize the reconstruction loss.

The weights of the deep autoencoder are used to intialize the weights of encoder and decoder networks of above methods.

After the pretraining, we optimze the objectives of those methods.

For DEC and DCN, we use the same hyperparameter settings as the original papers.

When initializing the cluster centroids for DEC and DCN, we perform 10 random restarts and pick the results with the best objective value for k-means/GMM.

For the proposed LTVAE, we use Adam optimzer BID15 with initial learning rate of 0.001 and mini-batch size of 128.

For Stepwise EM, we set the learning rate to be 0.01.

As in Algorithm 1, we set E = 5, i.e. we update the latent tree model every 5 epochs.

When optimizing the candidate models during structure search, we perform 10 random restarts and train with EM for 200 iterations.

We first show that, by using the marginal loglikelihood defined by the latent tree model as the prior, LTVAE better fits the data than conventional VAE and importance weighted autoencoders (IWAE) BID3 .

While alternative quantitative criteria have been proposed BID2 BID13 BID24 for generative models, log-likelihood of held-out test data remains one of the most important measures of a generative model's performance BID16 BID3 BID32 BID9 .

For comparison, we approximate true loglikelihood L 5000 using importance sampling BID3 : DISPLAYFORM0 , where z (i) ∼ q φ (z|x).

The results for all datasets are shown in TAB0 .

The proposed LTVAE obtains a higher test data loglikelihood and ELBO, implying that it can better model the underlying complex data distribution embedded in the image data.

The most important features of the proposed model are that it can perform variable selection for model-based clustering, leading to multiple facets clustering.

We use the standard unsupervised evaluation metric and protocols for evaluations and comparisons to other algorithms BID36 .

For baseline algorithms we set the number of clusters to the number of ground-truth categories.

While for LTVAE, it automatically determines the number of facets and latent superstructure through structure learning.

We evaluate performance with unsupervised clustering accuracy (ACC): DISPLAYFORM0 where l i is the groundtruth label, c i is the cluster assignment produced by the algorithm, and m ranges over all possible mappings between clusters and labels.

TAB1 show the quantitative clustering results compared with previous works.

With z dimension of small value like 10, LTVAE usually discovers only one facet.

It can be seen the, for MNIST dataset LTVAE achieves clustering accuracy of 86.32%, better than the results of other methods.

This is also the case for STL-10, Reuters and HHAR.More importantly, the proposed LTVAE does not just give one partition over the data.

Instead, it explains the data in multi-faceted ways.

Unlike previous clustering experiments, for this experiment, we choose the z dimension to be 20.

FIG3 shows the two facet clustering results for MNIST.

It can be seen that facet 1 gives quite clean clustering over the identity of the digits and the ten digits are well separated.

On the other hand, facet 2 gives a more grand partition based on the shape and pose.

Note how up-right "4" and "9" are similar, and how tilted "4","7" and "9" are similar.

The facet meanings are more evident in FIG3 .

FIG4 shows four facets discovered for the STL-10 The digits generated by the proposed model.

Digits in the same row come from the same latent code of the latent tree.dataset.

Although it is hard to characterize precisely how the facets differ from each other, there are visible patterns.

For example, the cats, monkeys and birds in facet 2 have clearly visible eyes, while this is not always true in facet 1.

The deers in facet 2 are all showing their antlers/ears, while this is not true in facet 3.

In facet 2 we see frontal views of cars, while in facets 1 and 3 we see side view of cars.

In facet 1, each cluster consists of the same types of objects/animal.

In facet 3/4, images in the same cluster do not necessarily show the same type of objects/animals.

However, they have similar overall feel.

Since the structure of the data in latent space is automatically learned through the latent tree, we can sample the data in a more structured way.

One way is through ancestral sampling, where we first sample the root of the latent tree and then hierarchically sample the children variables to get z, from which the images can be generated through generation network.

The other way is to pick one component from the Gaussian mixture and sample z from that component.

This produces samples from a particular cluster.

FIG5 shows the samples generated in this way.

As it can be seen, digits sampled from each component has clear semantic meaning and belong to the same category.

Whereas, the samples generated by VAE does not have such structure.

Conditional image generation can also be performed to alter the attributes of the same digit as shown in Appendix B.

LTVAE learns the dependencies among latent variables Y. In general, latent variables are often correlated.

For example, the social skills and academic skills of a student are generally correlated.

Therefore, its better to model this relationship to better fit the data.

Experiments show that removing such dependencies in LTVAE models results in inferior data loglikelihood.

In this paper, for the inference network, we simply use mean-field inference network with same structure as the generative network BID16 .

However, the limited expressiveness of the mean-field inference network could restrict the learning in the generative network and the quality of the learned model BID31 BID6 .

Using a faithful inference network structure as in BID31 to incorporate the dependencies among latent variables in the posterior, for example one parameterized with masked autoencoder distribution estimator (MADE) model BID8 , could have a significant improvement in learning.

We leave it for future investigation.

In this paper, we propose an unsupervised learning method, latent tree variational autoencoder (LT-VAE), which simultaneously performs representation learning and multidimensional clustering.

Different from previous deep learning based clustering methods, LTVAE learns latent embeddings from data and discovers multi-facet clustering structure based on subsets of latent features rather than one partition over data.

Experiments show that the proposed method achieves state-of-the-art clustering performance and reals reasonable multifacet structures of the data.

For the MNIST dataset, the conditional probability between identity facet Y 1 (x-axis) and pose facet Y 2 (y-axis) is shown in Fig. 8 .

It can be seen that a cluster in Y 1 facet could correspond to multiple clusters in Y 2 facet due to the conditional probability, e.g. cluster 0, 4, 5, 11 and 12.

However, not all clusters in Y 2 facet are possible for a given cluster in Y 1 facet.

Figure 8: Conditional probability of Y 1 and Y 2 for the two facets of MNIST discovered by LTVAE.

Here we show more results on conditional image generation.

Interestingly, with LTVAE, we can change the original images by fixing variables in some facet and sampling in other facets.

For example, in MNIST we can fix the variables in identity facet and change the pose of the digit by sampling in the pose facet.

FIG6 shows the samples generated in this way.

As it can be seen, the pose of the input digits are changed in the samples generated by the proposed method.

C COMPUTATIONAL TIMEWe compare the computational time of the proposed LTVAE w/ structure learning and that w/ fixed structure.

For LTVAE with fixed structure, we fixed the structure of the latent tree model to be a single Y connecting to all zs, in which each z node consists of single z variable.

@highlight

We investigate a variant of variational autoencoders where there is a superstructure of discrete latent variables on top of the latent features.