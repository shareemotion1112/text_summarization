Clustering high-dimensional datasets is hard because interpoint distances become less informative in high-dimensional spaces.

We present a clustering algorithm that performs nonlinear dimensionality reduction and clustering jointly.

The data is embedded into a lower-dimensional space by a deep autoencoder.

The autoencoder is optimized as part of the clustering process.

The resulting network produces clustered data.

The presented approach does not rely on prior knowledge of the number of ground-truth clusters.

Joint nonlinear dimensionality reduction and clustering are formulated as optimization of a global continuous objective.

We thus avoid discrete reconfigurations of the objective that characterize prior clustering algorithms.

Experiments on datasets from multiple domains demonstrate that the presented algorithm outperforms state-of-the-art clustering schemes, including recent methods that use deep networks.

Clustering is a fundamental procedure in machine learning and data analysis.

Well-known approaches include center-based methods and their generalizations BID2 BID26 , and spectral methods BID20 BID34 .

Despite decades of progress, reliable clustering of noisy high-dimensional datasets remains an open problem.

High dimensionality poses a particular challenge because assumptions made by many algorithms break down in high-dimensional spaces BID1 BID3 BID24 .There are techniques that reduce the dimensionality of data by embedding it in a lower-dimensional space .

Such general techniques, based on preserving variance or dissimilarity, may not be optimal when the goal is to discover cluster structure.

Dedicated algorithms have been developed that combine dimensionality reduction and clustering by fitting low-dimensional subspaces BID14 BID31 .

Such algorithms can achieve better results than pipelines that first apply generic dimensionality reduction and then cluster in the reduced space.

However, frameworks such as subspace clustering and projected clustering operate on linear subspaces and are therefore limited in their ability to handle datasets that lie on nonlinear manifolds.

Recent approaches have sought to overcome this limitation by constructing a nonlinear embedding of the data into a low-dimensional space in which it is clustered BID7 BID36 BID38 .

Ultimately, the goal is to perform nonlinear embedding and clustering jointly, such that the embedding is optimized to bring out the latent cluster structure.

These works have achieved impressive results.

Nevertheless, they are based on classic center-based, divergencebased, or hierarchical clustering formulations and thus inherit some limitations from these classic methods.

In particular, these algorithms require setting the number of clusters a priori.

And the optimization procedures they employ involve discrete reconfigurations of the objective, such as discrete reassignments of datapoints to centroids or merging of putative clusters in an agglomerative procedure.

Thus it is challenging to integrate them with an optimization procedure that modifies the embedding of the data itself.

We seek a procedure for joint nonlinear embedding and clustering that overcomes some of the limitations of prior formulations.

There are a number of characteristics we consider desirable.

First, we wish to express the joint problem as optimization of a single continuous objective.

Second, this optimization should be amenable to scalable gradient-based solvers such as modern variants of SGD.

Third, the formulation should not require setting the number of clusters a priori, since this number is often not known in advance.

While any one of these desiderata can be fulfilled by some existing approaches, the combination is challenging.

For example, it has long been known that the k-means objective can be optimized by SGD BID5 .

But this family of formulations requires positing the number of clusters k in advance.

Furthermore, the optimization is punctuated by discrete reassignments of datapoints to centroids, and is thus hard to integrate with continuous embedding of the data.

In this paper, we present a formulation for joint nonlinear embedding and clustering that possesses all of the aforementioned desirable characteristics.

Our approach is rooted in Robust Continuous Clustering (RCC), a recent formulation of clustering as continuous optimization of a robust objective BID22 .

The basic RCC formulation has the characteristics we seek, such as a clear continuous objective and no prior knowledge of the number of clusters.

However, integrating it with deep nonlinear embedding is still a challenge.

For example, Shah & Koltun (2017) presented a formulation for joint linear embedding and clustering (RCC-DR), but this formulation relies on a complex alternating optimization scheme with linear least-squares subproblems, and does not apply to nonlinear embeddings.

We present an integration of the RCC objective with dimensionality reduction that is simpler and more direct than RCC-DR, while naturally handling deep nonlinear embeddings.

Our formulation avoids alternating optimization and the introduction of auxiliary dual variables.

A deep nonlinear embedding of the data into a low-dimensional space is optimized while the data is clustered in the reduced space.

The optimization is expressed by a global continuous objective and conducted by standard gradient-based solvers.

The presented algorithm is evaluated on high-dimensional datasets of images and documents.

Experiments demonstrate that our formulation performs on par or better than state-of-the-art clustering algorithms across all datasets.

This includes recent approaches that utilize deep networks and rely on prior knowledge of the number of ground-truth clusters.

Controlled experiments confirm that joint dimensionality reduction and clustering is more effective than a stagewise approach, and that the high accuracy achieved by the presented algorithm is stable across different dimensionalities of the latent space.

Let X = [x 1 , . . .

, x N ] be a set of points in R D that must be clustered.

Generic clustering algorithms that operate directly on X rely strongly on interpoint distances.

When D is high, these distances become less informative BID1 BID3 .

Hence most clustering algorithms do not operate effectively in high-dimensional spaces.

To overcome this problem, we embed the data into a lower-dimensional space R d .

The embedding of the dataset into R d is denoted by Y = [y 1 , . . .

, y N ].

The function that performs the embedding is denoted by f θ : DISPLAYFORM0 Our goal is to cluster the embedded dataset Y and to optimize the parameters θ of the embedding as part of the clustering process.

This formulation presents an obvious difficulty: if the embedding f θ can be manipulated to assist the clustering of the embedded dataset Y, there is nothing that prevents f θ from distorting the dataset such that Y no longer respects the structure of the original data.

We must therefore introduce a regularizer on θ that constrains the low-dimensional image Y with respect to the original high-dimensional dataset X. To this end, we also consider a reverse mapping DISPLAYFORM1 To constrain f θ to construct a faithful embedding of the original data, we require that the original data be reproducible from its low-dimensional image BID11 : DISPLAYFORM2 Here DISPLAYFORM3 , and · F denotes the Frobenius norm.

Next, we must decide how the low-dimensional embedding Y will be clustered.

A natural solution is to choose a classic clustering framework: a center-based method such as k-means, a divergence-based formulation, or an agglomerative approach.

These are the paths taken in recent work on combining nonlinear dimensionality reduction and clustering BID7 BID36 BID38 .

However, the classic clustering algorithms have a discrete structure: associations between centroids and datapoints need to be recomputed or putative clusters need to be merged.

In either case, the optimization process is punctuated by discrete reconfigurations.

This makes it difficult to coordinate the clustering of Y with the optimization of the embedding parameters Ω that modify the dataset Y itself.

Since we must conduct clustering in tandem with continuous optimization of the embedding, we seek a clustering algorithm that is inherently continuous and performs clustering by optimizing a continuous objective that does not need to be updated during the optimization.

The recent RCC formulation provides a suitable starting point BID22 .

The key idea of RCC is to introduce a set of representatives Z ∈ R d×N and optimize the following nonconvex objective: DISPLAYFORM4 where ρ is a redescending M-estimator, E is a graph connecting the datapoints, {w i,j } are appropriately defined weights, and λ is a coefficient that balances the two objective terms.

The first term in objective (2) constrains the representatives to remain near the corresponding datapoints.

The second term pulls the representatives to each other, encouraging them to merge.

This formulation has a number of advantages.

First, it reduces clustering to optimization of a fixed continuous objective.

Second, each datapoint has its own representative in Z and no prior knowledge of the number of clusters is needed.

Third, the nonconvex robust estimator ρ limits the influence of outliers.

To perform nonlinear embedding and clustering jointly, we wish to integrate the reconstruction objective (1) and the RCC objective (2).

This idea is developed in the next section.

The Deep Continuous Clustering (DCC) algorithm optimizes the following objective: DISPLAYFORM0 This formulation bears some similarity to RCC-DR (Shah & Koltun, 2017), but differs in three major respects.

First, RCC-DR only operates on a linear embedding defined by a sparse dictionary, while DCC optimizes a more expressive nonlinear embedding parameterized by Ω. Second, RCC-DR alternates between optimizing dictionary atoms, sparse codes, representatives Z, and dual line process variables; in contrast, DCC avoids duality altogether and optimizes the global objective directly.

Third, DCC does not rely on closed-form or linear least-squares solutions to subproblems; rather, the joint objective is optimized by modern gradient-based solvers, which are commonly used for deep representation learning and are highly scalable.

We now discuss objective (3) and its optimization in more detail.

The mappings F θ and G ω are performed by an autoencoder with fully-connected or convolutional layers and rectified linear units after each affine projection BID11 BID18 .

The graph E is constructed on X using the mutual kNN criterion BID6 , augmented by the minimum spanning tree of the kNN graph to ensure connectivity to all datapoints.

The role of M-estimators ρ 1 and ρ 2 is to pull the representatives of a true underlying cluster into a single point, while disregarding spurious connections across clusters.

For both estimators, we use scaled Geman-McClure functions BID9 : DISPLAYFORM1 The parameters µ 1 and µ 2 control the radii of the convex basins of the estimators.

The weights w i,j are set to balance the contribution of each datapoint to the pairwise loss: DISPLAYFORM2 Here n i is the degree of z i in the graph E. The numerator is simply the average degree.

The parameter λ balances the relative strength of the data loss and the pairwise loss.

To balance the different terms, DISPLAYFORM3 , where A = (i,j)∈E w i,j (e i − e j )(e i − e j ) and · 2 denotes the spectral norm.

In contrast to RCC-DR, the parameter λ need not be updated during the optimization.

Objective (3) can be optimized using scalable modern forms of stochastic gradient descent (SGD).

Note that each z i is updated only via its corresponding loss and pairwise terms.

On the other hand, the autoencoder parameters Ω are updated via all data samples.

Thus in a single epoch, there is bound to be a difference between the update rates for Z and Ω. To deal with this imbalance, an adaptive solver such as Adam should be used BID13 .Another difficulty is that the graph E connects all datapoints such that a randomly sampled minibatch is likely to be connected by pairwise terms to datapoints outside the minibatch.

In other words, the objective (3), and more specifically the pairwise loss, does not trivially decompose over datapoints.

This requires some care in the construction of minibatches.

Instead of sampling datapoints, we sample subsets of edges from E. The corresponding minibatch B is defined by all nodes incident to the sampled edges.

However, if we simply restrict the objective (3) to the minibatch and take a gradient step, the reconstruction and data terms will be given additional weight since the same datapoint can participate in different minibatches, once for each incident edge.

To maintain balance between the terms, we must weigh the contribution of each datapoint in the minibatch.

The rebalanced minibatch loss is given by DISPLAYFORM0 where DISPLAYFORM1 Here DISPLAYFORM2 ni , where n B i is the number of edges connected to the i th node in the subgraph E B .The gradients of L B with respect to the low-dimensional embedding Y and the representatives Z are given by DISPLAYFORM3 These gradients are propagated to the parameters Ω.

Initialization.

The embedding parameters Ω are initialized using the stacked denoising autoencoder (SDAE) framework BID32 .

Each pair of corresponding encoding and decoding layers is pretrained in turn.

Noise is introduced during pretraining by adding dropout to the input of each affine projection BID23 .

Encoder-decoder layer pairs are pretrained sequentially, from the outer to the inner.

After all layer pairs are pretrained, the entire SDAE is fine-tuned end-to-end using the reconstruction loss.

This completes the initialization of the embedding parameters Ω. These parameters are used to initialize the representatives Z, which are set to Z = Y = F θ (X).Continuation.

The price of robustness is the nonconvexity of the estimators ρ 1 and ρ 2 .

One way to alleviate the dangers of nonconvexity is to use a continuation scheme that gradually sharpens the estimator BID4 BID17 .

Following Shah & Koltun (2017), we initially set µ i to a high value that makes the estimator ρ i effectively convex in the relevant range.

The value of µ i is decreased on a regular schedule until a threshold δi 2 is reached.

We set δ 1 to the mean of the distance of each y i to the mean of Y, and δ 2 to the mean of the bottom 1% of the pairwise distances in E at initialization.

Every iteration, construct a minibatch B defined by a sample of edges E B .

Update {z i } i∈B and Ω.

Every M epochs, update µ i = max µi 2 , δi 2 .

10: end while 11: Construct graph G = (V, F) with f i,j = 1 if z * i − z * j 2 < δ 2 .

12: Output clusters given by the connected components of G.Stopping criterion.

Once the continuation scheme is completed, DCC monitors the computed clustering.

At the end of every epoch, a graph G = (V, F) is constructed such that f i,j = 1 if z i − z j < δ 2 .

The cluster assignment is given by the connected components of G. DCC compares this cluster assignment to the one produced at the end of the preceding epoch.

If less than 0.1% of the edges in E changed from intercluster to intracluster or vice versa, DCC outputs the computed clustering and terminates.

Complete algorithm.

The complete algorithm is summarized in Algorithm 1.

We conduct experiments on six high-dimensional datasets, which cover domains such as handwritten digits, objects, faces, and text.

We used datasets from Shah & Koltun (2017) that had dimensionality above 100.

The datasets are further described in the appendix.

All features are normalized to the range [0, 1].Note that DCC is an unsupervised learning algorithm.

Unlabelled data is embedded and clustered with no supervision.

There is thus no train/test split.

The presented DCC algorithm is compared to 12 baselines, which include both classic and deep clustering algorithms.

The baselines include k-means++ BID0 , DBSCAN BID8 , two variants of agglomerative clustering: Ward (AC-W) and graph degree linkage (GDL) BID40 , two variants of spectral clustering: spectral embedded clustering (SEC) BID21 and local discriminant models and global integration (LDMGI) BID39 , and two variant of robust continuous clustering: RCC and RCC-DR (Shah & Koltun, 2017).The deep clustering baselines include four recent approaches that share our basic motivation and use deep networks for clustering: deep embedded clustering (DEC) BID36 , joint unsupervised learning (JULE) BID38 , the deep clustering network (DCN) BID37 , and deep embedded regularized clustering (DEPICT) BID7 .

These are strong baselines that use deep autoencoders, the same network structure as our approach (DCC).

The key difference is in the loss function and the consequent optimization procedure.

The prior formulations are built on KLdivergence clustering, agglomerative clustering, and k-means, which involve discrete reconfiguration of the objective during the optimization and rely on knowledge of the number of ground-truth clusters either in the design of network architecture, during the embedding optimization, or in post-processing.

In contrast, DCC optimizes a robust continuous loss and does not rely on prior knowledge of the number of clusters.

We report experimental results for two different autoencoder architectures: one with only fullyconnected layers and one with convolutional layers.

This is motivated by prior deep clustering algorithms, some of which used fully-connected architectures and some convolutional.

For fully-connected autoencoders, we use the same autoencoder architecture as DEC BID36 .

Specifically, for all experiments on all datasets, we use an autoencoder with the following dimensions: D-500-500-2000-d-2000-500-500-D. This autoencoder architecture follows parametric t-SNE (van der Maaten, 2009).For convolutional autoencoders, the network architecture is modeled on JULE BID38 ).

The architecture is specified in the appendix.

As in BID38 , the number of layers depends on image resolution in the dataset and it is set such that the output resolution of the encoder is about 4×4.In both architectures and for all datasets, the dimensionality of the reduced space is set to d = 10. (It is only varied for controlled experiments that analyze stability with respect to d.) No dataset-specific hyperparameter tuning was done.

For autoencoder initialization, a minibatch size of 256 and dropout probability of 0.2 are used.

SDAE pretraining and finetuning start with a learning rate of 0.1, which is decreased by a factor of 10 every 80 epochs.

Each layer is pretrained for 200 epochs.

Finetuning of the whole SDAE is performed for 400 epochs.

For the fully-connected SDAE, the learning rates are scaled in accordance with the dimensionality of the dataset.

For m-kNN graph construction, the nearest-neighbor parameter k is set to 10 and the cosine distance metric is used.

The Adam solver is used with its default learning rate of 0.001 and momentum 0.99.

Minibatches are constructed by sampling 128 edges.

DCC was implemented using the PyTorch library.

For the baselines, we use publicly available implementations.

For k-means++, DBSCAN and AC-W, we use the implementations in the SciPy library and report the best results across ten random restarts.

For a number of baselines, we performed hyperparameter search to maximize their reported performance.

For DBSCAN, we searched over values of Eps, for LDMGI we searched over values of the regularization constant λ, for SEC we searched over values of the parameter µ, and for GDL we tuned the graph construction parameter a.

The DCN approach uses a different network architecture for each dataset.

Wherever possible, we report results using their dataset-specific architecture.

For YTF, Coil100, and YaleB, we use their reference architecture for MNIST.

Common measures of clustering accuracy include normalized mutual information (NMI) BID25 and clustering accuracy (ACC).

However, NMI is known to be biased in favor of fine-grained partitions and ACC is also biased on imbalanced datasets BID33 .

To overcome these biases, we use adjusted mutual information (AMI) BID33 , defined as DISPLAYFORM0 Here H(·) is the entropy, MI(·, ·) is the mutual information, and c andĉ are the two partitions being compared.

AMI lies in a range [0, 1].

Higher is better.

For completeness, results according to ACC are reported in the appendix.

The results are summarized in Table 1 .

Among deep clustering methods that use fully-connected networks, DCN and DEC are not as accurate as fully-connected DCC and are also less consistent: the performance of DEC drops on the high-dimensional image datasets, while DCN is far behind on MNIST and YaleB. Among deep clustering methods that use convolutional networks, the performance of DEPICT drops on COIL100 and YTF, while JULE is far behind on YTF.

The GDL algorithm failed to scale to the full MNIST dataset and the corresponding measurement is marked as 'n/a'.

MNIST Coil100 YTF YaleB Reuters RCV1 Table 1 : Clustering accuracy of DCC and 12 baselines, measured by AMI.

Higher is better.

Methods that do no use deep networks are listed first, followed by deep clustering algorithms that use fullyconnected autoencoders (including the fully-connected configuration of DCC) and deep clustering algorithms that use convolutional autoencoders (including the convolutional configuration of DCC).Results that are within 1% of the highest accuracy achieved by any method are highlighted in bold.

DCC performs on par or better than prior deep clustering formulations, without relying on a priori knowledge of the number of ground-truth clusters.

Importance of joint optimization.

We now analyze the importance of performing dimensionality reduction and clustering jointly, versus performing dimensionality reduction and then clustering the embedded data.

To this end, we use the same SDAE architecture and training procedure as fully-connected DCC.

We optimize the autoencoder but do not optimize the full DCC objective.

This yields a standard nonlinear embedding, using the same autoencoder that is used by DCC, into a space with the same reduced dimensionality d. In this space, we apply a number of clustering algorithms: k-means++, AC-W, DBSCAN, SEC, LDMGI, GDL, and RCC.

The results are shown in TAB3 (top).These results should be compared to results reported in Table 1 .

The comparison shows that the accuracy of the baseline algorithms benefits from dimensionality reduction.

However, in all cases their accuracy is still lower than that attained by DCC using joint optimization.

Furthermore, although RCC and DCC share the same underlying nearest-neighbor graph construction and a similar clustering loss, the performance of DCC far surpasses that achieved by stagewise SDAE embedding followed by RCC.

Note also that the relative performance of most baselines drops on Coil100 and YaleB. We hypothesize that the fully-connected SDAE is limited in its ability to discover a good low-dimensional embedding for very high-dimensional image datasets (tens of thousands of dimensions for Coil100 and YaleB).Next, we show the performance of the same clustering algorithms when they are applied in the reduced space produced by DCC.

These results are reported in TAB3 (bottom).

In comparison to TAB3 (top), the performance of all algorithms improves significantly and some results are now on par or better than the results of DCC as reported in Table 1 .

The improvement for k-means++, Ward, and DBSCAN is particularly striking.

This indicates that the performance of many clustering algorithms can be improved by first optimizing a low-dimensional embedding using DCC and then clustering in the learned embedding space.

The embedding is performed using the same autoencoder architecture as used by fully-connected DCC, into the same target space.

However, dimensionality reduction and clustering are performed separately.

Clustering accuracy is much lower than the accuracy achieved by DCC.

Bottom: Here clustering is performed in the reduced space discovered by DCC.

The performance of all clustering algorithms improves significantly.

Visualization.

A visualization is provided in FIG0 .

Here we used Barnes-Hut t-SNE (van der BID29 BID28 to visualize a randomly sampled subset of 10K datapoints from the MNIST dataset.

We show the original dataset, the dataset embedded by the SDAE into R d (optimized for dimensionality reduction), and the embedding into R d produced by DCC.

As shown in the figure, the embedding produced by DCC is characterized by well-defined, clearly separated clusters.

The clusters strongly correspond to the ground-truth classes (coded by color in the figure), but were discovered with no supervision.

Robustness to dimensionality of the latent space.

Next we study the robustness of DCC to the dimensionality d of the latent space.

For this experiment, we consider fully-connected DCC.

We vary d between 5 and 60 and measure AMI on the MNIST and Reuters datasets.

For comparison, we report the performance of DEC, which uses the same autoencoder architecture, as well as the accuracy attained by running k-means++ on the output of the SDAE, optimized for dimensionality reduction.

The results are shown in FIG1 .

The results yield two conclusions.

First, the accuracy of DCC, DEC, and SDAE+k-means gradually decreases as the dimensionality d increases.

This supports the common view that clustering becomes progressively harder as the dimensionality of the data increases.

Second, the results demonstrate that DCC is more robust to increased dimensionality than DEC and SDAE.

For example, on MNIST, as the dimensionality d changes from 5 to 60, the accuracy of DEC and SDAE drops by 28% and 35%, respectively, while the accuracy of DCC decreases by only 9%.

When d = 60, the accuracy attained by DCC is higher than the accuracy attained by DEC and SDAE by 27% and 40%, respectively.

We have presented a clustering algorithm that combines nonlinear dimensionality reduction and clustering.

Dimensionality reduction is performed by a deep network that embeds the data into a lower-dimensional space.

The embedding is optimized as part of the clustering process and the resulting network produces clustered data.

The presented algorithm does not rely on a priori knowledge of the number of ground-truth clusters.

Nonlinear dimensionality reduction and clustering are performed by optimizing a global continuous objective using scalable gradient-based solvers.

B CONVOLUTIONAL NETWORK ARCHITECTURE TAB4 summarizes the architecture of the convolutional encoder used for the convolutional configuration of DCC.

Convolutional kernels are applied with a stride of two.

The encoder is followed by a fully-connected layer with output dimension d and a convolutional decoder with kernel size that matches the output dimension of conv5.

The decoder architecture mirrors the encoder and the output from each layer is appropriately zero-padded to match the input size of the corresponding encoding layer.

All convolutional and transposed convolutional layers are followed by batch normalization and rectified linear units BID12 BID18 .

C HYPERPARAMETERS DCC uses three hyperparameters: the nearest neighbor graph (mkNN) parameter k, the embedding dimensionality d, and the update period M for graduated nonconvexity.

For fair comparison to RCC and RCC-DR, we fix k = 10 (the setting used in BID22 ).

The other two hyperparameters were set to d = 10 and M = 20 based on grid search on MNIST.

The hyperparameters are fixed at these values across all datasets.

No dataset-specific tuning is done.

However, note that the hyperparameter M is architecture-specific.

We set M = 10 for convolutional autoencoders and it is varied for varying dimensionality d during the controlled experiment reported in FIG1 .

The other hyperparameters such as λ, δ i , µ i are set automatically as described in Sections 3.2 and 3.3 and in BID22 .

DISPLAYFORM0

For completeness, we report results according to the ACC measure.

TAB5 provides the ACC counterpart to Table 1 .

FIG2 provides the ACC counterpart to FIG1 .

We also report results according to the NMI measure.

<|TLDR|>

@highlight

A clustering algorithm that performs joint nonlinear dimensionality reduction and clustering by optimizing a global continuous objective.

@highlight

Presents a clustering algorithm by jointly solving deep autoencoder and clustering as a global continuous objective, showing better results than state-of-the-art clustering schemas.

@highlight

Deep Continuous Clustering is a clustering method that integrates the autoencoder objective with the clustering objective then train using SGD.