The joint optimization of representation learning and clustering in the embedding space has experienced a breakthrough in recent years.

In spite of the advance, clustering with representation learning has been limited to flat-level categories, which oftentimes involves cohesive clustering with a focus on instance relations.

To overcome the limitations of flat clustering, we introduce hierarchically clustered representation learning (HCRL), which simultaneously optimizes representation learning and hierarchical clustering in the embedding space.

Specifically, we place a nonparametric Bayesian prior on embeddings to handle dynamic mixture hierarchies under the variational autoencoder framework, and to adopt the generative process of a hierarchical-versioned Gaussian mixture model.

Compared with a few prior works focusing on unifying representation learning and hierarchical clustering, HCRL is the first model to consider a generation of deep embeddings from every component of the hierarchy, not just leaf components.

This generation process enables more meaningful separations and mergers of clusters via branches in a hierarchy.

In addition to obtaining hierarchically clustered embeddings, we can reconstruct data by the various abstraction levels, infer the intrinsic hierarchical structure, and learn the level-proportion features.

We conducted evaluations with image and text domains, and our quantitative analyses showed competent likelihoods and the best accuracies compared with the baselines.

Clustering is one of the most traditional and frequently used machine learning tasks.

Clustering models are designed to represent intrinsic data structures, such as latent Dirichlet allocation BID2 .

The recent development of representation learning has contributed to generalizing model feature engineering, which also enhances data representation BID1 .

Therefore, representation learning has been merged into the clustering models, e.g., variational deep embedding (VaDE) (Jiang et al., 2017) .

Besides merging representation learning and clustering, another critical line of research is structuring the clustering result, e.g., hierarchical clustering.

This paper introduces a unified model enabling nonparametric Bayesian hierarchical clustering with neural-network-based representation learning.

Autoencoder (Rumelhart et al., 1985) is a typical neural network for unsupervised representation learning and achieves a non-linear mapping from a high-dimensional input space to a lowdimensional embedding space by minimizing reconstruction errors.

To turn the low-dimensional embeddings into random variables, a variational autoencoder (VAE) (Kingma & Welling, 2014) places a Gaussian prior on the embeddings.

The autoencoder, whether it is probabilistic or not, has a limitation in reflecting the intrinsic hierarchical structure of data.

For instance, VAE assuming a single Gaussian prior needs to be expanded to suggest an elaborate clustering structure.

Due to the limitations of modeling the cluster structure with autoencoders, prior works combine the autoencoder and the clustering algorithm.

While some early cases pipeline just two models, e.g., Huang et al. (2014) , a typical merging approach is to model an additional loss, such as a clustering loss, in the autoencoders (Xie et al., 2016; Guo et al., 2017; Yang et al., 2017; Nalisnick et al., 2016; BID4 Jiang et al., 2017) .

These suggestions exhibit gains from unifying the encoding and the clustering, yet they remain at the parametric and flat-structured clustering.

A more recent development releases the previous constraints by using the nonparametric Bayesian approach.

Figure 1: Example of hierarchically clustered embeddings on MNIST with three levels of hierarchy, the reconstructed digits from the hierarchical Gaussian mixture components, and the extracted level proportion features.

We marked the mean of a Gaussian mixture component with the colored square, and the digit written inside the square refers to the unique index of the mixture component.

For example, the infinite mixture of VAEs (IMVAE) BID0 explores the infinite space for VAE mixtures by looking for an adequate embedding space through sampling, such as the Chinese restaurant process (CRP).

Whereas IMVAE remains at the flat-structured clustering, VAEnested CRP (VAE-nCRP) (Goyal et al., 2017) captures a more complex structure, i.e., a hierarchical structure of the data, by adopting the nested Chinese restaurant process (nCRP) prior (Griffiths et al., 2004) into the cluster assignment of the Gaussian mixture model.

This paper proposes hierarchically clustered representation learning (HCRL) that is a joint model of 1) nonparametric Bayesian hierarchical clustering, and 2) representation learning with neural networks.

HCRL extends a previous work on merging flat clustering and representation learning, i.e., VaDE, by incorporating inter-cluster relation modelings.

Unlike a previous work of VAE-nCRP, HCRL learns the full spectrum of hierarchical clusterings, such as the level assignment and the level proportion of generating a component hierarchy.

These level assignments and proportions were not modeled in VAE-nCRP, so each data instance cannot be analyzed from the perspective of generalization and specialization in a hierarchy.

On the contrary, by adding level assignment and proportion modeling, a data instance can be generated from an internal component of the hierarchy, which is limited to the leaf component in VAE-nCRP.

Hierarchical mixture density estimation (Vasconcelos & Lippman, 1999) , where all internal and leaf components are directly modeled to generate data, is a flexible framework for hierarchical mixture modeling, such as hierarchical topic modeling (Mimno et al., 2007; Griffiths et al., 2004) , with regard to the learning of the internal components.

Specifically, HCRL jointly optimizes soft-divisive hierarchical clustering in an embedding space from VAE via two mechanisms.

First, HCRL includes a hierarchical-versioned Gaussian mixture model (HGMM) with a mixture of hierarchically organized Gaussian distributions.

Then, HCRL sets the prior of embeddings by adopting the generative processes of HGMM.

Second, to handle a dynamic hierarchy structure dealing with the clusters of unequal sizes, we explore the infinite hierarchy space by exploiting an nCRP prior.

These mechanisms are fused as a unified objective function; this is done rather than concatenating the two distinct models of clustering and autoencoding.

The quantitative evaluations focus on density estimation quality and hierarchical clustering accuracy, which shows that HCRL has competent likelihoods and the best accuracies compared with the baselines.

When we observe our results qualitatively, we visualize 1) the hierarchical clusterings, 2) the embeddings under the hierarchy modeling, and 3) the reconstructed images from each Gaussian mixture component, as shown in FIG3 .

These experiments were conducted by crossing the data domains of texts and images, so our benchmark datasets include MNIST, CIFAR-100, RCV1 v2, and 20Newsgroups.

2.1 VARIATIONAL DEEP EMBEDDING mixture components, respectively, are declared outside of the neural network 1 .

VaDE trains model parameters to maximize the lower bound of marginal log likelihoods via the mean-field variational inference (Jordan et al., 1999) .

VaDE uses the Gaussian mixture model (GMM) as the prior, whereas VAE assumes a single standard Gaussian distribution on embeddings.

Following the generative process of GMM, VaDE assumes that 1) the embedding draws a cluster assignment, and 2) the embedding is generated from the selected Gaussian mixture component.

VaDE uses an amortized inference as VAE, with a generative and inference networks; L(x) in Equation 1 denotes the evidence lower bound (ELBO), which is the lower bound on the log likelihood.

It should be noted that VaDE merges the ELBO of VAE with the likelihood of GMM.

DISPLAYFORM0 2.2 VARIATIONAL AUTOENCODER NESTED CHINESE RESTAURANT PROCESS VAE-nCRP uses the nonparametric Bayesian prior for learning tree-based hierarchies, the nCRP (Griffiths et al., 2004) , so the representation could be hierarchically organized.

The nCRP prior defines the distributions over children components for each parent component, recursively in a topdown way.

The variational inference of the nCRP can be formalized by the nested stick-breaking construction (Wang & Blei, 2009) , which is also kept in the VAE setting.

The distribution over paths on the hierarchy is defined as being proportional to the product of weights corresponding to the nodes lying in each path.

The weight, π i , for the i-th node follows the Griffiths-Engen-McCloskey (GEM) distribution (Pitman et al., 2002) , where π i is constructed as DISPLAYFORM1 by a stick-breaking process.

Since the nCRP provides the ELBO with the nested stick-breaking process, VAE-nCRP has a unified ELBO of VAE and the nCRP in Equation 2.

DISPLAYFORM2 Given the ELBO of VAE-nCRP, we recognized a number of potential improvements.

First, term (3.1) is for modeling the hierarchical relationship among clusters, i.e., each child is generated from its parent.

VAE-nCRP trade-off is the direct dependency modeling among clusters against the meanfield variational approach.

This modeling may reveal that the higher clusters in the hierarchy are more difficult to train.

Second, in term (3.2), leaf mixture components generate embeddings, which implies that only leaf clusters have direct summarization ability for sub-populations.

Additionally, in term (3.2), variance parameter σ 2 D is modeled as the hyperparameter shared by all clusters.

In other words, only with J-dimensional parameters, α, for the leaf mixture components, the local density modeling without variance parameters has a critical disadvantage.

For all of these weaknesses, we were able to compensate with the level proportion modeling and HGMM prior.

The level assignment generated from the level proportion allows a data instance to select among all mixture components.

We do not need direct dependency modeling between the parents and their children because all internal mixture components also generate embeddings.

The generative process of HCRL resembles the generative process of hierarchical clusterings, such as the hierarchical latent Dirichlet allocation (Griffiths et al., 2004) .

In detail, the generative process departs from selecting a path ζ, from the nCRP prior (phase 1).

Then, we sample a level proportion (phase 2) and a level, l (phase 3), from the sampled level proportion to find the mixture component in the path, and this component of ζ l provides the Gaussian distribution for the latent representation (phase 4).

Finally, the latent representation is exploited to generate an observed datapoint (phase 5).

The below formulas are the generative process with its density functions.

In addition, FIG2 illustrates a graphical representation corresponding to the described generative process.

The generative process also presents our formalization of corresponding prior distributions, denoted as p(·), and variational distributions, denoted as q(·), by generation phases.

The variational distributions are used in our inference methods called mean-field variational inference (MFVI) (Jordan et al., 1999) as detailed in Section 3.3.

where DISPLAYFORM0 The neural architecture of HCRL consists of two probabilistic encoders on z and η, and one probabilistic decoder on z as shown in the right part of FIG2 .

This unbalanced architecture originates from our modeling assumption of p(x|z), not p(x|z, η).

The reconstruction design of x depending on the two stochastic variables of z and η may lead to a large variance of the reconstruction on x. Additionally, we cannot guarantee that both z and η contribute to the the reconstruction on x BID3 .

Although the decoding structure of η is not included explicitly in the neural network architecture of HCRL, we provide the formalization of p(η|z) in TAB0 according to our generative assumptions.

We call this reconstruction process, which is inherently a generative process of the traditional probabilistic graphical model (PGM), PGM reconstruction (see the decoding neural network part of FIG2 ).

DISPLAYFORM1 The formal specification can be a factorized probabilistic model as Equation 3, where Φ = {v, ζ, η, l, z} denotes the set of latent variables, and M T denotes the set of all nodes in tree T .

DISPLAYFORM2 The proportion and assignment on the mixture components for the n-th data instance are modeled by ζ n as a path assignment; η n as a level proportion; and l n as a level assignment.

v is a Beta draw used in the stick-breaking construction.

The latent variables are inferred through MFVI, and therefore we assume the variational distributions are as Equation 4 : DISPLAYFORM3 where q φη (η n |x n ) and q φz (z n |x n ) should be noted because these two variational distributions follow the amortized inference of VAE.

q(ζ|x) ∝ S ζ ζ∈child(ζ) S ζ is the variational distribution over path ζ, where child(ζ) means the set of all full paths that are not in T but include ζ as a sub path.

Because we specified both generative and variational distributions, we define the ELBO of HCRL, L = E q log p(Φ,x) q(Φ|x) , in Equation 5.

Appendix F enumerates the full derivation in detail.

We report that the Laplace approximation with the logistic normal distribution is applied to model the prior, α, of the level proportion, η.

We choose a conjugate prior of a multinomial, so p(η n |α) follows the Dirichlet distribution.

To configure the inference network on the Dirichlet prior, the Laplace approximation is used (MacKay, 1998; Srivastava & Sutton, 2017; Hennig et al., 2012) .

DISPLAYFORM4

This model is formalized according to the stick-breaking process scheme.

Unlike the CRP, the stickbreaking process does not represent the direct sampling of the mixture component at the data instance level.

Therefore, it is necessary to devise a heuristic algorithm for operations, such as GROW, PRUNE, and MERGE, to refine the hierarchy structure.

Appendix C provides details about each operation, together with the overall training algorithm of HCRL.

In the below description, an inner path and a full path refer to the path ending with an internal node and a leaf node, respectively.• GROW expands the hierarchy by creating a new branch under the heavily weighted internal node.

Compared with the work of Wang & Blei (2009), we modified GROW to first sample a path, ζ * , proportional to n q(ζ n = ζ * ), and then to grow the path if the sampled path is an inner path.• PRUNE cuts a randomly sampled minor full path, ζ * , satisfying DISPLAYFORM0 n,ζ q(ζn=ζ) < δ, where δ is the pre-defined threshold.

If the removed leaf node of the full path is the last child of the parent node, we also recursively remove the parent node.• MERGE combines two full paths, ζ (i) and ζ (j) , with similar posterior probabilities, measured DISPLAYFORM1

Datasets: We used various hierarchically organized benchmark datasets as well as MNIST.• MNIST (LeCun et al., 1998) : 28x28x1 handwritten image data, with 60,000 train images and 10,000 test images.

We reshaped the data to 784-d in one dimension.• CIFAR-100 (Krizhevsky & Hinton, 2009): 32x32x3 colored images with 20 coarse and 100 fine classes.

We used 3,072-d flattened data with 50,000 training and 10,000 testing.

DISPLAYFORM0 The preprocessed text of the Reuters Corpus Volume.

We preprocessed the text by selecting the top 2,000 tf-idf words.

We used the hierarchical labels up to the 4-level, and the multi-labeled documents were removed.

The final preprocessed corpus consists of 11,370 training and 10,000 testing documents randomly sampled from the original test corpus.• 20Newsgroups (Lang, 1995) : The benchmark text data extracted from 20 newsgroups, consisting 11,314 training and 7,532 testing documents.

We also labeled by 4-level following the annotated hierarchical structure.

We preprocessed the data through the same process as that of RCV1 v2.Baselines: We completed our evaluation in two aspects: 1) optimizing the density estimation, and 2) clustering the hierarchical categories.

First, we evaluated HCRL from the density estimation perspective by comparing it with diverse flat clustered representation learning models, and VAE-nCRP.

Second, we tested HCRL from the accuracy perspective by comparing it with multiple divisive hierarchical clusterings.

The below is the list of baselines.

We also added the two-stage pipeline approaches, where we trained features from VaDE first and then applied the hierarchical clusterings.

We reused the open source codes 3 provided by the authors for several baselines, such as IDEC, DCN, VAE-nCRP, and SSC-OMP.

We used two measures to evaluate the learned representations in terms of the density estimations: 1) negative log likelihood (NLL), and 2) reconstruction errors (REs).

Autoencoder models, such as IDEC and DCN, were tested only for the REs.

The NLL is estimated with 100 samples.

TAB1 indicates that HCRL is best in the NLL and is competent in the REs which means that the hierarchically clustered embeddings preserve the intrinsic raw data structure.

VaDE generally performed better than VAE did, whereas other flat clustered representation learning models tended to be slightly different for each dataset.

HCRL showed overall competent performance and better results with a deeper hierarchy of level four than of level three, which implies that capturing the deeper hierarchical structure is likely to be useful for the density estimation.

Additionally, we evaluated hierarchical clustering accuracies by following Xie et al. FORMULA33 , except for MNIST that is flat structured.

TAB2 points out that HCRL has significantly better microaveraged F-scores compared with every baseline.

HCRL is able to reproduce the ground truth hierarchical structure of the data, and this trend is consistent when HCRL compared with the pipelined model, such as VaDE with a clustering model.

The result of the comparisons with the clustering models, such as HKM, MOHG, RGMM, and RSSCOMP, is interesting because it experimentally proves that the joint optimization of hierarchical clustering in the embedding space improves hierarchical clustering accuracies.

HCRL also presented better hierarchical accuracies than VAE-nCRP.

We conjecture the reasons for the modeling aspect of VAE-nCRP: 1) the simplified prior modeling on the variance of the mixture component as just constants, and 2) the non-flexible learning of the internal components.

MNIST: In FIG3 , the digits {4, 7, 9} and the digits {3, 8} are grouped together with a clear hierarchy, which was consistent between HCRL and VaDE.

Also, some digits {0, 4, 2} in a round form are grouped, together, in HCRL.

In addition, among the reconstructed digits from the hierarchical mixture components, the digits generated from the root have blended shapes from 0 to 9, which is natural considering the root position.

FIG4 shows the hierarchical clustering results on CIFAR-100.

Given that there were no semantic inputs from the data, the color was dominantly reflected in the clustering criteria.

However, if one observes the second hierarchy, the scene images of the same sub-hierarchy are semantically consistent, although the background colors are slightly different.

RCV1 v2: FIG5 shows the embedding of RCV1 v2.

VAE and VaDE show no hierarchy, and close sub-hierarchies are distantly embedded.

VAE-nCRP guides the internal mixture components to be agglomerated at the center, and the cause of agglomeration is the generative process of VAEnCRP, where the parameter of the internal components are inferred without direct information from data.

HCRL shows a clear separation between the sub-hierarchy without the agglomeration.

20Newsgroups: FIG6 shows the example sub-hierarchies on 20Newsgroups.

We enumerated topic words from documents with top-five likelihoods for each cluster, and we filtered the words by tf-idf values.

We observe relatively more general contents in the internal clusters than in the leaf clusters of each internal cluster.

In this paper, we have introduced a hierarchically clustered representation learning framework for the hierarchical mixture density estimation on deep embeddings.

HCRL aims at encoding the relations among clusters as well as among instances to preserve the internal hierarchical structure of data.

The main differentiated features of HCRL are 1) the crucial assumption regarding the internal mixture components for having the ability to generate data directly, and 2) the unbalanced autoencoding neural architecture for the level proportion modeling as the encoding structure, and the probabilistic model as the decoding structure.

From the modeling and the evaluation, we found that HCRL enables the improvements due to the high flexibility modeling compared with the baselines.

We created a synthetic dataset that has a hierarchical structure and is sampled from the 50-dimensional Gaussian distributions, presented in FIG7 .

The hierarchy, which has a branch factor of two and a depth of four, has a total of eight leaf clusters.

FIG7 shows the raw synthetic dataset in the input space of R 50 , and after running HCRL, we plot the hierarchically clustered embeddings in the latent space in FIG7 .

In addition to the embeddings, we also present a confidence ellipse with dashed lines for each learned Gaussian mixture component.

Because the root component is involved in generating all of the data, it forms a large ellipse, while the leaf component summarizes the local density, so the small ellipse is learned.

We show how the above embeddings learned to be hierarchically clustered in the latent space during training in FIG8 .

In the learning mechanism of HCRL, we can observe the hierarchically clustered embeddings from a major deviation to a minor deviation in the data over iterations.

We conducted experiments for all autoencoder-based models with a neural architecture whose encoder network was set as fully connected layers with dimensions D-2000-2000-500-J for z, and D-10-10-L for η, and the decoder network is a mirror of the encoder network for z. The hyperparameters of HCRL given by users, γ and α, was set to 1.0, and a vector of all entries 1 sized of L, respectively.

We used the Adam optimizer (Kingma & Ba, 2014) with an init learning rate of 0.001 for MNIST dataset and 0.0001 for other datasets.

Meanwhile, VAE-nCRP is targeted for grouped data.

For experiments with our non-grouped datasets, we treated the group instance as a group instance having a single data instance.

For parametric hierarchical clustering models, we gave the branch factor as the input parameter, [1, 20, 5] , [1, 4, 7, 9] , and [1, 6, 4, 3], for CIFAR-100, RCV1 v2, and 20Newsgroups, respectively.

For VaDE, we set the number of clusters to the number of leaf clusters; 100 for CIFAR-100, 252 for RCV1 v2, and 72 for 20Newsgroups.

Algorithm 1 summarizes the overall algorithm for HCRL.

The tree-based hierarhcy T is defined as (N, P), where N and P denote a set of nodes and paths, respectively.

We refer to the node at level l lying on path ζ, as N(ζ 1:l ) ∈ N. The defined paths, P, consist of full paths (ending at a leaf node), P full , and inner paths (ending at an internal node), P inner , as a union set.

Algorithm 1 selects an operation out of three operations: GROW, PRUNE, and MERGE.

The GROW algorithm is executed for every specific iteration period, t G .

After ellapsing t b iterations since performing the GROW operation, we begin to check whether the PRUNE or MERGE operation should be performed.

We prioritize the PRUNE operation first, and if the condition of performing PRUNE is not satisfied, we check for the MERGE operation next.

After performing any operation, we initialize n b to 0, which is for locking the changed hierarchy during minimum t b iterations to be fitted to the training data.

Input: Training examples x; the tree-based hierarchy depth, L; period of performing GROW, t grow ; minimum number of epochs locking the hierarchy, t lock ; operation-related thresholds δ prune , δ merge ; a queue whose element is the set of changed paths, Q; the number of training epochs, E; maximum length of Q, Q max ; grow scale, s grow Output: φ z , φ η , θ ← Update the network weight parameters using gradients ∇ φz,φη,θ L(x) 6: DISPLAYFORM0 DISPLAYFORM1 Update other variational parameters using gradients ∇L(x) DISPLAYFORM2 if mod(e, t grow ) = 0 then 9: if T (e) = T (e−1) then n lock ← 0 else n lock ← n lock + 1 16: end for DISPLAYFORM3 DISPLAYFORM4 Sample a path ζ * with probability DISPLAYFORM5 Q ← φ // Temporary set of changed paths in this epoch 5:if ζ * ∈ P inner and ζ * / ∈ Q s.t.

Q ∈ Q then 6: DISPLAYFORM6 j 0 ← Maximum index for the child node whose parent path is ζ * 1:l 9: DISPLAYFORM7 11: DISPLAYFORM8 if l < L − 1 then 13: DISPLAYFORM9 else 15: enqueue Q to Q 20: DISPLAYFORM10 DISPLAYFORM11 T ← (N, P) The PRUNE operation cuts a minor path, which is sampled according to N n=1 q(ζ n = ζ) among the full paths satisfying N n=1 q(ζ n = ζ) < δ, where δ is the pre-defined threshold parameter.

If the removed leaf node of the full path is the last child of the parent node, we also recursively remove the parent node as shown in the upper case of FIG3 .

DISPLAYFORM12 Randomly sample a full path ζ * ∼ Ω

if |P full | > 1 and ζ * / ∈ Q s.t.

Q ∈ Q then 6: DISPLAYFORM0 for l = L − 1, · · · , 1 do 8: DISPLAYFORM1 if l < L − 1 then 10: DISPLAYFORM2 end if

n c ← Number of the children nodes whose parent path is ζ DISPLAYFORM0 T ← (N, P) The MERGE operation combines two full paths with similar posterior probabilities, measured by DISPLAYFORM1 Gaussian components by following Ueda et al. (1999) .

The specific meaning of combining the two paths is merging the paired two Gaussian distributions lying on the two paths by level, if the two Gaussian distribtions are different.

The estimation of merged Gaussian parameters, µ and σ, is the weighted summation of two subject Gaussian parameters.

The propbability of the node at level l lying on a path ζ given x, p(ζ l |x), is proportional to n {q(l n = l) · ζ∈Λ q(ζ n = ζ)}, where Λ = {ζ |ζ l = ζ l and ζ ∈ P full }.

DISPLAYFORM2 Randomly sample a pair of paths (ζ DISPLAYFORM3 Q ← φ // Temporary set of changed paths in this epoch DISPLAYFORM4 l ← Maximum level of nodes shared by ζ (1) , ζ8: DISPLAYFORM5 for l = l, · · · , L − 1 do 10: DISPLAYFORM6 11: DISPLAYFORM7 w (1) +w FORMULA33 13: j 0 ← Maximum index for the child node whose parent path is ζ * 1:l 14: DISPLAYFORM8 if l < L − 1 then 18: DISPLAYFORM9 1:l +1 }

else 20: while Q max < |Q| do dequeue Q DISPLAYFORM0

P ← P full ∪ P inner 28:T ← (N, P)

return T, Q 30: end function

The following TAB8 lists the notations used throughout this paper.

A encoder network parametrized by * , whose input is x f θ (z)A decoder network parametrized by θ, whose input is z θThe variational parameters and weights of the decoder network f θ µz,σ 2 zThe variational mean and variance for Gaussian distribution q φz (z|x) µx, σ 2 xThe prior parameters, mean and variance, for Gaussian distribution DISPLAYFORM0 The variational parameters and weights of the encoder network g φ µ,σ 2The variational mean and variance for Gaussian distribution q φ (z|x)

The number of datapoints xn=1,··· ,N n-th observed datapoint zn=1,··· ,N n-th latent representation corresponding to xn VAE-nCRP & HCRL

The height of the tree-based hierarchy VaDE K The number of (finite) clusters cn=1,··· ,N The cluster assignment of zn, ∈ {1, ..., K} κThe prior parameter for multinomial distribution p(c) µc, σ 2 cThe prior parameters, mean and variance, for Gaussian distribution of c-th cluster, p(z)

The number of sequences Nm=1,··· ,MThe number datapoints in m-th sequence xm,n=1,··· ,N n-th observed datapoint in m-th sequence zm,n=1,··· ,N n-th latent representation corresponding to xmn vmp The Beta draws of m-th sequence on node p, for the tree-based stickbreaking construction γ * The prior parameter for Beta distribution p(vmp)

γ DISPLAYFORM0 The variational parameters, for Beta distribution q(vmp|xm)

ζmnThe path assignment of zmn S * mnThe variational parameter for multinomial distribution q(ζmn|xmn) α par(p)The J-dimensional parameter vector for the parent node of p α * The prior parameter for Gaussian distribution p(αp) for the root node µ par(p) , σ 2 par(p)The variational mean and variance for Gaussian distribution q(α par(p) |x) αpThe J-dimensional parameter vector for node p σ The prior parameter, variance, for Gaussian distribution p(zmn|ζmn, αp)

The variational parameters and weights of the encoder network g φz φηThe variational parameters and weights of the encoder network g φη µz,σ 2 zThe variational mean and variance for Gaussian distribution q φz (z|x) µη,σ 2 ηThe variational mean and variance for logistic normal distribution q φη (η|x) αThe variational parameter for Dirichlet distribution q φη (η|x) viThe Beta draws for the tree-based stick-breaking construction of node i γThe prior parameter for Beta distribution p(vi) ai, biThe variational parameters, for Beta distribution q(vi|x) ζnThe path assignment of zn SnThe variational parameter for multinomial distribution q(ζn|xn) ηnThe level proportion of zn αThe prior parameter for Dirichlet distribution p(ηn) lnThe level assignment of zn, ∈ {1, ..., L} ωnThe variational parameter for multinomial distribution q(ln|xn) µi, σ E GENERATIVE AND INFERENCE MODEL FOR HCRL HCRL assumes the generative process as described in Section 3.1.

Section E.1 describes the joint probability distribution, and Section E.2 presents the corresponding variational distributions.

We adopt the much notation-related conventions from Wang & Blei (2009), especially on paths.

DISPLAYFORM0 • p θ (x n |z n ) : Probabilistic decoding of x n parametrized by θ, whose input is z n • Tree-based stick-breaking construction -We will denote all Beta draws as v, each of which is an independent draw from Beta(v|1, γ) (except for root v 1 = 1) DISPLAYFORM1 As VAE, we infer the random variables via the mean-field approximation, where the variational distribution, q φη,φz (v, ζ, η, l, z|x) , approximates the intractable posterior.

We model the variational distributions as follows: DISPLAYFORM2 The level of the mixture component i• q(ζ n |x n ) ∝ S nζ ζ∈child(ζ) S nζ -ζ: a path in the truncated tree T , either an inner path (a path ending at an internal node) or a full path (a path ending at a leaf node)-child(ζ): the set of all full paths that are not in T but include ζ as a sub path * As a special case, if ζ is a full path, child(ζ) just contains itself -In the case of a full path, DISPLAYFORM3 : maximum index for the child node whose parent path is ζ DISPLAYFORM4

In this section, we present the detailed derivation of the ELBO in Equation 6, which is the objective function for learning HCRL.

v, ζ, η, l, z|x) = E q log i∈M T p(v i |γ) N n=1 p(ζ n |v)p(η n |α)p(l n |η n )p(z n |ζ n , l n )p θ (x n |z n ) i∈M T q(v i |a i , b i ) N n=1 q(ζ n |x n )q φη (η n |x n )q(l n |ω n , x n )q φz (z n |x n ) DISPLAYFORM0 DISPLAYFORM1 E q [log p(ζ n |v) + log p(η n |α) + log p(l n |η n ) + log p(z n |ζ n , l n ) DISPLAYFORM2 E q [log q(ζ n |x n ) + log q φη (η n |x n ) + log q(l n |ω n , x n ) + log q φz (z n |x n )]

The followings are additional notations used for the detailed derivation:• ψ : The digamma function DISPLAYFORM0 Under review as a conference paper at ICLR 2019 DISPLAYFORM1 DISPLAYFORM2 q(l n = l )q(z n = z) log p(η n = η |α)dη dz dv = η q(η n = η ) log p(η n = η |α)dη = η Dir(η | α n ) · log Dir(η |α)dη DISPLAYFORM3 q(l n = l )q(z n = z ) log p(l n = l |η n )dηdz dv = η l q(η n = η )q(l n = l) log p(l n = l|η n = η )dη = l q(l n = l ) η q(η n = η ) log Mult(l |η )dη DISPLAYFORM4 q(l n = l )q(z n = z) log p θ (x n |z n = z )dη dz dv = z q(z n = z) log p θ (x n |z n = z )dz DISPLAYFORM5 log p θ (x n |z DISPLAYFORM6 q(l n = l )q(z n = z) log q(ζ n = ζ)dη dz dv = ζ q(ζ n = ζ) log q(ζ n = ζ) DISPLAYFORM7 q(l n = l )q(z n = z) log q(η n = η )dη dz dv = η q(η n = η ) log q(η n = η )dη = η Dir(η | α n ) · log Dir(η | α n )dη DISPLAYFORM8 q(l n = l )q(z n = z) log q(l n = l )dη dz dv DISPLAYFORM9 q(l n = l )q(z n = z) log q(z n = z)dη dz dv (1 + log σ 2 znj )

<|TLDR|>

@highlight

We introduce hierarchically clustered representation learning (HCRL), which simultaneously optimizes representation learning and hierarchical clustering in the embedding space.

@highlight

The paper proposes using the nested CRP as a clustering model rather than a topic model

@highlight

Presents a novel hierarchical clustering method over an embedding space where both the embedding space and the heirarchical clustering are simultaneously learnt