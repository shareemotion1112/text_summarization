Training generative models like Generative Adversarial Network (GAN)  is challenging for noisy data.

A novel curriculum learning algorithm pertaining to clustering is proposed to address this issue in this paper.

The curriculum construction is based on the centrality of underlying clusters in data points.

The data points of high centrality takes priority of being fed into generative models during training.

To make our algorithm scalable to large-scale data, the active set is devised, in the sense that every round of training proceeds only on an active subset containing a small fraction of already trained data and the incremental data of lower centrality.

Moreover, the geometric analysis is presented to interpret the necessity of cluster curriculum for generative models.

The experiments on cat and human-face data validate that our algorithm is able to learn the optimal generative models (e.g. ProGAN) with respect to specified quality metrics for noisy data.

An interesting finding is that the optimal cluster curriculum is closely related to the critical point of the geometric percolation process formulated in the paper.

Deep generative models have piqued researchers' interest in the past decade.

The fruitful progress has been achieved on this topic, such as auto-encoder (Hinton & Salakhutdinov, 2006) and variational auto-encoder (VAE) (Kingma & Welling, 2013; Rezende et al., 2014) , generative adversarial network (GAN) (Goodfellow et al., 2014; , normalizing flow (Rezende & Mohamed, 2015; Dinh et al., 2015; Kingma & Dhariwal, 2018) , and autoregressive models (van den Oord et al., 2016b; a; .

However, it is non-trivial to train a deep generative model that can converge to a proper minimum of the associated optimization.

For example, GAN suffers non-stability, mode collapse, and generative distortion during training.

Many insightful algorithms have been proposed to circumvent those issues, including feature engineering (Salimans et al., 2016) , various discrimination metrics (Mao et al., 2016; Berthelot et al., 2017) , distinctive gradient penalties (Gulrajani et al., 2017; Mescheder et al., 2018) , spectral normalization to discriminator (Miyato et al., 2018) , and orthogonal regularization to generator (Brock et al., 2019) .

What is particularly of interest is that the breakthrough for GANs has been made with a simple technique of progressively growing neural networks of generators and discriminators from low-resolution images to high-resolution counterparts (Karras et al., 2018a) .

This kind of progressive growing also helps push the state of the arts to a new level by enabling StyleGAN to produce photo-realistic and detail-sharp results (Karras et al., 2018b) , shedding new light on wide applications of GANs in solving real problems.

This idea of progressive learning is actually a general manner of cognition process (Elman, 1993; Oudeyer et al., 2007) , which has been formally named curriculum learning in machine learning (Bengio et al., 2009) .

The central topic of this paper is to explore a new curriculum for training deep generative models.

To facilitate robust training of deep generative models with noisy data, we propose curriculum learning with clustering.

The key contributions are listed as follows:

??? We first summarize four representative curricula for generative models, i.e. architecture (generation capacity), semantics (data content), dimension (data space), and cluster (data structure).

Among these curricula, cluster curriculum is newly proposed in this paper.

??? Cluster curriculum is to treat data according to the centrality of each data point, which is pictorially illustrated and explained in detail.

To foster large-scale learning, we devise the active set algorithm that only needs an active data subset of small fixed size for training.

??? The geometric principle is formulated to analyze hardness of noisy data and advantage of cluster curriculum.

The geometry pertains to counting a small sphere packed in an ellipsoid, on which is based the percolation theory we use.

The research on curriculum learning is diverse.

Our work focuses on curricula that are closely related to data attributes, beyond which is not the scope we concern in this paper.

Curriculum learning has been a basic learning approach to promoting performance of algorithms in machine learning.

We quote the original words from the seminal paper (Bengio et al., 2009 ) as its definition: Curriculum learning.

"

The basic idea is to start small, learn easier aspects of the task or easier sub-tasks, and then gradually increase the difficulty level" according to pre-defined or self-learned curricula.

From cognitive perspective, curriculum learning is common for human and animal learning when they interact with environments (Elman, 1993) , which is the reason why it is natural as a learning rule for machine intelligence.

The learning process of cognitive development is gradual and progressive (Oudeyer et al., 2007) .

In practice, the design of curricula is task-dependent and data-dependent.

Here we summarize the representative curricula that are developed for generative models.

Architecture curriculum.

The deep neural architecture itself can be viewed as a curriculum from the viewpoint of learning concepts (Hinton & Salakhutdinov, 2006; Bengio et al., 2006) or disentangling representations (Lee et al., 2011) .

For example, the different layers decompose distinctive features of objects for recognition (Lee et al., 2011; Zeiler & Fergus, 2014; Zhou et al., 2016) and generation (Bau et al., 2018) .

Besides, Progressive growing of neural architectures is successfully exploited in GANs (Karras et al., 2018a; Heljakka et al., 2018; Korkinof et al., 2018; Karras et al., 2018b) .

Semantics curriculum.

The most intuitive content for each datum is the semantic information that the datum conveys.

The hardness of semantics determines the difficulty of learning knowledge from data.

Therefore, the semantics can be a common curriculum.

For instance, the environment for a game in deep reinforcement learning (Justesen et al., 2018) and the number sense of learning cognitive concepts with neural networks (Zou & McClelland, 2013) can be such curricula.

Dimension curriculum.

The high dimension usually poses the difficulty of machine learning due to the curse of dimensionality (Donoho, 2000) , in the sense that the amount of data points for learning grows exponentially with dimension of variables (Vershynin, 2018) .

Therefore, the algorithms are expected to be beneficial from growing dimensions.

The effectiveness of dimension curriculum is evident from recent progress on deep generative models, such as ProGANs (Karras et al., 2018a; b) by gradually enlarging image resolution and language generation from short sequences to long sequences of more complexity (Rajeswar et al., 2017; Press et al., 2017) .

For fitting distributions, dense data points are generally easier to handle than sparse data or outliers.

To train generative models robustly, therefore, it is plausible to raise cluster curriculum, meaning that generative algorithms first learn from data points close to cluster centers and then with more data progressively approaching cluster boundaries.

Thus the stream of feeding data points to models for curriculum learning is the process of clustering data points according to cluster centrality that will be explained in section 3.2.

The toy example in Figure 1 illustrates how to form cluster curriculum.

The importance of clusters for data points is actually obvious from geometric point of view.

The data sparsity in high-dimensional spaces causes the difficulty of fitting the underlying distribution of n = 100 n = 200 n = 300 n = 400

Figure 1: Cluster Curriculum.

From magenta color to black color, the centrality of data points reduces.

The value n is the number of data points taken with centrality order.

data points (Vershynin, 2018) .

So generative algorithms may be beneficial when proceeding from the local spaces where data points are relatively dense.

Such data points form clusters that are generally informative subsets with respect to the entire dataset.

In addition, clusters contain common regular patterns of data points, where generative models are easier to converge.

What is most important is that noisy data points deteriorate performance of algorithms.

For classification, the effectiveness of curriculum learning is theoretically proven to circumvent the negative influence of noisy data (Gong et al., 2016) .

We will analyze this aspect for generative models with geometric facts.

With cluster curriculum, we are allowed to gradually learn generative models from dense clusters to cluster boundaries and finally to all data points.

In this way, generative algorithms are capable of avoiding the direct harm of noise or outliers.

To this end, we first need a measure called centrality that is the terminology in graph-based clustering.

It quantifies the compactness of a cluster in data points or a community in complex networks (Newman, 2010) .

A large centrality implies that the associated data point is close to one of cluster centers.

For easy reference, we provide the algorithm of the centrality we use in Appendix.

For experiments in this paper, all the cluster curricula are constructed by the centrality of stationary probability distribution, i.e. the eigenvector corresponding to the largest eigenvalue of the transition probability matrix drawn from the data.

To be specific, let c ??? R m denote the centrality vector of m data points.

Namely, the i-th entry c i of c is the centrality of data point x i .

Sorting c in descending order and adjusting the order of original data points accordingly give data points arranged by cluster centrality.

Let

. .

, ??? ??? X l } signify the set of centrality-sorted data points, where ??? ??? X 0 is the base set that contains sufficient data to attain convergent generative models, and the rest of ??? ??? X is evenly divided into l subsets according to centrality order.

In general, the number of data points in ??? ??? X 0 is much less than m and determined according to X .

Such division of ??? ??? X 0 serves to efficiency of training, because we do not need to train models from a very small dataset.

The cluster curriculum learning is carried out by incrementally feeding subsets in ??? ??? X into generative algorithms.

In other words, algorithms are successively trained on

, meaning that the curriculum for each round of training is accumulated with ??? ??? X i .

In order to determine the optimal curriculum

we need the aid of quality metric of generative models, such as Fr??chet inception distance (FID) or sliced Wasserstein distance (SWD) (Borji, 2018) .

For generative models trained with each curriculum, we calculate the associated score s i via the specified quality metric.

The optimal curriculum for effective training can be identified by the minimal value for all s i , where i = 1, . . .

, l + 1.

The interesting phenomenon of this score curve will be illustrated in the experiment.

The minimum of score s is apparently metric-dependent.

One can refer to (Borji, 2018) for the review of evaluation metrics.

In practice, we can opt one of reliable metrics to use or multiple metrics for decision-making of the optimal model.

There are two ways of using the incremental subset ??? ??? X i during training.

One is that the parameters of models are re-randomized when the new data are used, the procedure of which is given in Algorithm 1 in Appendix.

The other is that the parameters are fine-tuned based on pre-training of the previous model, which will be presented with a fast learning algorithm in the following section.

To obtain the precise minimum of s, the cardinality of ??? ??? X i needs to be set much smaller than m, meaning that l will be large even for a dataset of moderate scale.

The training of many loops will be time-consuming.

Here we propose the active set to address the issue, in the sense that for each loop of cluster curriculum, the generative models are always trained with a subset of a small fixed size instead of ??? ??? X 0 ??? ??? ??? X 0 ??? ??? ???

X i whose size becomes incrementally large.

(a)

Figure 2: Schematic illustration of active set for cluster curriculum.

The cardinality |A| of the active set A is 200.

When ??? ??? X 2 is taken for training, we need to randomly sample another 100 (i.e.

??? ??? X 2|) data points from the history data

Then the complete active set is composed by

We can see that data points in

become less dense after sampling.

To form the active set A, the subset ??? ??? A 0 of data points are randomly sampled from ??? ??? X 0 to combine with ??? ??? X i for the next loop, where

For easy understanding, we illustrate the active set with toy example in Figure 2 .

In this scenario, progressive pre-training must be applied, meaning that the update of model parameters for the current training is based on parameters of previous loop.

The procedure of cluster curriculum with active set is detailed in Algorithm 2 in Appendix.

The active set allows us to train generative models with a small dataset that is actively adapted, thereby significantly reducing the training time for large-scale data.

Cluster curriculum bears the interesting relation to high-dimensional geometry, which can provide geometric understanding of our algorithm.

Without loss of generality, we work on a cluster obeying the normal distribution.

The characteristic of the cluster can be extended into other clusters of the same distribution.

For easy analysis, let us begin with a toy example.

As Figure 3(a) shows, the confidence ellipse E 2 fitted from the subset of centrality-ranked data points is nearly conformal to E 1 of all data points, which allows us to put the relation of these two ellipses by virtue of the confidence-level equation.

Let N (0, ??) signify the center and covariance matrix of the cluster C of interest, where C = {x i |x i ??? R d , i = 1, . . .

, n}. To make it formal, we can write the equation by The annulus formed by of removing the inner ellipse from the outer one.

where ?? To analyze the hardness of training generative models, a fundamental aspect is to examine the number n(E) of given data points falling in a geometric entity E 1 and the number N (E) of lattice points in it.

The less n(E) is compared to N (E), the harder the problem will be.

However, the enumeration of lattice points is computationally prohibitive for high dimensions.

Inspired by the information theory of encoding data of normal distributions (Roman, 1996; Ma et al., 2007) , we count the number of small spheres S ?? of radius ?? packed in the ellipsoid E instead.

Thus we can use this number to replace the role of N (E) as long as the radius of the sphere S ?? is set properly.

With a little abuse of normal d=2 d=20 d=50 d=100 d=500 d=1000 lattice d=2 d=20 d=50 d=100 d=500 d=1000 Figure 4 : Comparison between the number n(A) of data points sampled from the isotropic normal distributions and N (A) of spheres (lattice) packed in the annulus A with respect to the Chi quantile ?? ??2 .

d is the dimension of data points.

For each dimension, we sample 70,000 data points from N (0, I).

The scales of y-axis and x-axis are normalized by 10,000 and ?? ??1 , respectively.

notation, we still use N (E) to denote the packing number in the following context.

Theorem 1 gives the exact form of N (E).

Theorem 1.

For a set C = {x i |x i ??? R d } of n data points drawn from normal distribution N (0, ??), the ellipsoid E ?? of confidence 1 ??? ?? is defined as x ?? ???1 x ??? ?? 2 ?? , where ?? has no zero eigenvalues and ?? ??? [0, 1].

Let N (E ?? ) be the number of spheres of radius ?? packed in the ellipsoid E ?? .

Then we can establish

We can see that N (E ?? ) admits a tidy form with Mahalanobis distance ?? ?? , dimension d, and sphere radius ?? as variables.

The proof is provided in Appendix.

The geometric region of interest for cluster curriculum is the annulus A formed by removing the ellipsoid 2 E ??2 from the ellipsoid E ??1 , as Figure 3 (b) displays.

We investigate the varying law between n(A) and N (A) in the annulus A when the inner ellipse E ??2 grows with cluster curriculum.

For this purpose, we need the following two corollaries that immediately follows from Theorem 1.

Corollary 1.

Let N (A) be the number of spheres of radius ?? packed in the annulus A that is formed by removing the ellipsoid E ??1 from the ellipsoid E ??1 , where ?? 1 ??? ?? 2 .

Then we have

It is obvious that N (A) goes infinite when d ??? ??? under the conditions that ?? ??1 > ?? ??2 and ?? is bounded.

Besides, when E ??2 (cluster) grows, N (A) reduces with exponent d if E ??1 is fixed.

In light of Corollary 1, we can now demonstrate the functional law between n(A) and N (A).

First, we determine ?? ??1 as follows

which means that E ??1 is the ellipsoid of minimal Mahalanobis distance to the center that contains all the data points in the cluster.

In addition, we need to estimate a suitable sphere radius ??, such that n(E ??1 ) and N (E ??1 ) have comparable scales in order to make n(A) and N (A) comparable in scale.

To achieve this, we define an oracle ellipse E where n(E) = N (E).

For simplicity, we let E ??1 be the oracle ellipse.

Thus we can determine ?? with Corollary 3.

Corollary 3.

If we let E ??1 be the oracle ellipse such that n(E ??1 ) = N (E ??1 ), then the free parameter ?? can be computed with ?? = ?? ??1 det(??)/n(E ??1 )

To make the demonstration amenable to handle, data points we use for simulation are assumed to obey the isotropic normal distribution, meaning that data points are generated with nearly equal Figure 5 : Examples of LSUN cat dataset and CelebA face dataset.

The samples in the first row are of high centrality and the samples of low centrality in the second row are noisy data or outliers that we call in the context.

variance along each dimension.

Figure 4 shows that n(A) gradually exhibits the critical phenomena of percolation processes 3 when the dimension d goes large, implying that the data points in the annulus A are significantly reduced when E ??2 grows a little bigger near the critical point.

In contrast, the number N (A) of lattice points is still large and varies negligibly until E ??2 approaches the boundary.

This discrepancy indicates clearly that fitting data points in the annulus is pretty hard and guaranteeing the precision is nearly impossible when crossing the critical point of n(A) even for a moderate dimension (e.g. d = 500).

Therefore, the plausibility of cluster curriculum can be drawn naturally from this geometric fact.

The generative model that we use for experiments are Progressive growing of GAN (ProGAN) (Karras et al., 2018a ).

This algorithm is chosen because ProGAN is the state-of-the-arts algorithm of GANs with official open sources available.

According to convention, we opt the Fr??chet inception distance (FID) (Borji, 2018) for ProGAN as the quality metric.

We randomly sample the 200,000 cat images from the LSUN dataset (Yu et al., 2015) .

These cat images are captured in the wild.

So their styles vary significantly.

Figure 5 shows the cat examples of high and low centralities.

We can see that the noisy cat images differ much from the clean ones.

There actually contain the images of very few informative cat features, which are the outliers we refer to.

The curriculum parameters are set as | ??? ??? X 0 | = 20, 000 and | ??? ???

X i | = 10, 000, which means that the algorithms are trained with 20,000 images first and after the initial training, another 10,000 images according to centrality order are merged into the current training data for further re-training.

For active set, its size is fixed to be 30, 000.

The CelebA dataset is a large-scale face attribute dataset (Liu et al., 2015) .

We use the cropped and well-aligned faces with a bit of image backgrounds preserved for generation task.

For clustercurriculum learning, we randomly sample 70,000 faces as the training set.

The face examples of different centralities are shown in Figure 5 .

The curriculum parameters are set as | ??? ??? X 0 | = 10, 000 and | ??? ???

X i | = 5, 000.

We bypass the experiment of the active set on faces because it is used for the large-scale data.

Each image in two databases is resized to be 64 ?? 64.

To form cluster curricula, we exploit ResNet34 (He et al., 2016) pre-trained on ImageNet (Russakovsky et al., 2015) to extract 512-dimensional features for each face and cat images.

The directed graphs are built with these feature vectors.

We determine the parameter ?? of edge weights by enforcing the geometric mean of weights to be 0.8.

The robustness of varying the value was validated in (Zhao & Tang, 2008) for clustering.

The number of nearest neighbors is set to be K = 4 log m. The centrality is the stationary probability distribution.

All codes are written with TensorFlow.

: FID curves of cluster-curriculum learning for ProGAN on the cat dataset and CelebA face dataset.

The centrality and the FID share the x-axis due to that they have the same order of data points.

The same colors of the y-axis labels and the curves denote the figurative correspondence.

The network parameters for "normal training" are randomly re-initialized for each re-training.

The active set is based on progressive pre-training of the fixed small dataset.

The scale of the x-axis is normalized by 10,000.

From Figure 6a , we can see that the FID curves are all nearly V-shaped, indicating that the global minima exist amid the training process.

This is the clear evidence that the noisy data and outliers deteriorate the quality of generative models during training.

From the optimal curricula found by two algorithms (i.e. curricula at 110,000 and 100,000), we can see that the curriculum of the active set differs from that of normal training only by one-step data increment, implying that the active set is reliable for fast cluster-curriculum learning.

The performance of the active set measured by FID is much worse than that of normal training, especially when more noisy data are fed into generative models.

However, this does not change the whole V-shape of the accuracy curve.

Namely, it is applicable as long as the active set admits the metric minimum corresponding to the appropriate curriculum.

The V-shape of the centrality-FID curve on the cat data is due to that the noisy data of low centrality contains little effective information to characterize the cats, as already displayed in Figure 5 .

However, it is different for the CelebA face dataset where the face images of low centrality also convey the part of face features.

As evident by Figure 6b , ProGAN keeps being optimized by the majority of the data until the curriculum of size 55, 000.

To highlight the meaning of this nearly negligible minimum, we also conduct the exactly same experiment on the FFHQ face dataset containing 70, 000 face images of high-quality (Karras et al., 2018b) .

For FFHQ data, the noisy face data can be ignored.

The gray curve of normal training in Figure 6b indicates that the FID of ProGAN is monotonically decreased for all curricula.

This gentle difference of the FID curves at the ends between CelebA and FFHQ clearly demonstrates the difficulty of noisy data to generative algorithms.

To understand cluster curriculum deeply, we employ the geometric method formulated in section 4 to analyze the cat and face data.

The percolation processes are both conducted with 512-dimensional features from ResNet34.

Figure 7 displays the curve of n(A) that is the variable of interest in this scenario.

As expected, the critical point in the percolation process occurs for both cases, as shown by blue curves.

An obvious fact is that the optimal curricula (red strips) both fall into the (feasible) domains of percolation processes after the critical points, as indicated by gray color.

This is a desirable property because data become rather sparse in the annuli when crossing the critical points.

Then noisy data play the non-negligible role on tuning the parameters of generative models.

Therefore, a fast learning strategy can be derived from the percolation process.

Training may begin from the curriculum specified by the critical point, thus significantly accelerating cluster-curriculum learning.

The pink strips are intervals of optimal curricula derived by generative models.

For example, the value 9 of the pink interval in (a) is obtained by 9 = 20 ??? 11, where 11 is one of the minima (i.e. 110,000) in Figure 6a .

The others are derived in the same way.

The subtraction transforms the data number in the cluster to be the one in the annulus.

The critical points are determined by searching the maxima of the absolute discrete difference of the associated curves.

The scales of y-axes are normalized by 10,000.

Another intriguing phenomenon is that the more noisy the data, the closer the optimal interval (red strip) is to the critical point.

We can see that the optimal interval of the cat data is much closer to the critical point than that of the face data.

What surprises us here is that the optimal interval of cluster curricula associated with the cat data nearly coincides with the critical point of the percolation process in the annulus!

This means that the optimal curriculum may be found at the intervals close to the critical point of of n(A) percolation for heavily noisy data, thus affording great convenience to learning an appropriate generative model for such datasets.

Cluster curriculum is proposed for robust training of generative models.

The active set of cluster curriculum is devised to facilitate scalable learning.

The geometric principle behind cluster curriculum is analyzed in detail as well.

The experimental results on the LSUN cat dataset and CelebA face dataset demonstrate that the generative models trained with cluster curriculum is capable of learning the optimal parameters with respect to the specified quality metric such as Fr??chet inception distance and sliced Wasserstein distance.

Geometric analysis indicates that the optimal curricula obtained from generative models are closely related to the critical points of the associated percolation processes established in this paper.

This intriguing geometric phenomenon is worth being explored deeply in terms of the theoretical connection between generative models and high-dimensional geometry.

It is worth emphasizing that the meaning of model optimality refers to the global minimum of the centrality-FID curve.

As we already noted, the optimality is metric-dependent.

We are able to obtain the optimal model with cluster curriculum, which does not mean that the algorithm only serves to this purpose.

We know that more informative data can help learn a more powerful model covering the large data diversity.

Here a trade-off arises, i.e. the robustness against noise and the capacity of fitting more data.

The centrality-FID curve provides a visual tool to monitor the state of model training, thus aiding us in understanding the learning process and selecting suitable models according to noisy degree of given data.

For instance, we can pick the trained model close to the optimal curriculum for heavily noisy data or the one near the end of the centrality-FID curve for datasets of little noise.

In fact, this may be the most common way of using cluster curriculum.

In this paper, we do not investigate the cluster-curriculum learning for the multi-class case, e.g. the ImageNet dataset with BigGAN (Brock et al., 2019) .

The cluster-curriculum learning of multiple classes is more complex than that we have already analyzed on the face and cat data.

We leave this study for future work.

The centrality or clustering coefficient pertaining to a cluster in data points or a community in a complex network is a well-studied traditional topic in machine learning and complex systems.

Here we introduce the graph-theoretic centrality for the utilization of cluster curriculum.

Firstly, we construct a directed graph (digraph) with K nearest neighbors by the method in (Zhao & Tang, 2008) .

The weighted adjacency matrix W of the digraph can be formed in this way:

2 ) if x j is one of the nearest neighbors of x i and 0 otherwise, where d ij is the distance between x i and x j and ?? is a free parameter.

The density of data points can be quantified with the stationary probability distribution of a Markov chain.

For a digraph built from data, the transition probability matrix can be derived by row normalization, say, P ij = W ij / j W ij .

Then the stationary probability u can be obtained by solving an eigenvalue problem P u = u,

where denotes the matrix transpose.

It is straightforward to know that u is the eigen-vector of P corresponding to the largest eigenvalue (i.e. 1).

u is also defined as a kind of PageRank in many scenarios.

For density-based cluster curriculum, the centrality c coincides with the stationary probability u. Figure 1 in the main context shows the plausibility of using the stationary probability distribution to quantify the data density.

Then we derive the length of semi-axis with respect to ?? ?? , i.e.

For a d-dimensional ellipsoid E, the volume of E is

where r i the leng of semi-axis of E and ??(??) is the Gamma function.

Substituting (10) into the above equation, we obtain the final formula of volume

Using the volume formula in (11), it is straightforward to get the volume of the packing sphere S ??

By the definition of N (E ?? ), we can write

We conclude the proof of the theorem.

@highlight

A novel cluster-based algorithm of curriculum learning is proposed to solve the robust training of generative models.