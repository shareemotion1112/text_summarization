Clustering is the central task in unsupervised learning and data mining.

k-means is one of the most widely used clustering algorithms.

Unfortunately, it is generally non-trivial to extend k-means to cluster data points beyond Gaussian distribution, particularly, the clusters with non-convex shapes (Beliakov & King, 2006).

To this end, we, for the first time, introduce Extreme Value Theory (EVT) to improve the clustering ability of k-means.

Particularly, the Euclidean space was transformed into a novel probability space denoted as extreme value space by EVT.

We thus propose a novel algorithm called Extreme Value k-means (EV k-means), including GEV k-means and GPD k-means.

In addition, we also introduce the tricks to accelerate Euclidean distance computation in improving the computational efficiency of classical k-means.

Furthermore, our EV k-means is extended to an online version, i.e., online Extreme Value k-means, in utilizing the Mini Batch k-means to cluster streaming data.

Extensive experiments are conducted to validate our EV k-means and online EV k-means on synthetic datasets and real datasets.

Experimental results show that our algorithms significantly outperform competitors in most cases.

Clustering is a fundamental and important task in the unsupervised learning (Jain, 2010; Rui Xu & Wunsch, 2005) .

It aims at clustering data samples of high similarity into the same cluster.

The most well-known clustering algorithm is the k-means, whose objective is to minimize the sum of squared distances to their closest centroids.

k-means has been extensively studied in the literature, and some heuristics have been proposed to approximate it (Jain, 2010; Dubes & Jain, 1988) .

The most famous one is Lloyd's algorithm (Lloyd, 1982) .

The k-means algorithm is widely used due to its simplicity, ease of use, geometric intuition (Bottesch et al., 2016) .

Unfortunately, its bottleneck is that computational complexity reaches O(nkd) (Rui Xu & Wunsch, 2005) , since it requires computing the Euclidean distances between all samples and all centroids.

The data is embedded in the Euclidean space (Stemmer & Kaplan, 2018) , which causes the failure on clustering non-convex clusters (Beliakov & King, 2006) .

Even worse, k-means is highly sensitive to the initial centroids, which usually are randomly initialized.

Thus, it is quite possible that the objective of k-means converges to a local minimum, which causes the instability of k-means, and is less desirable in practice.

Despite a stable version -k-means++ (Arthur & Vassilvitskii, 2007) gives a more stable initialization, fundamentally it is still non-trivial to extend k-means in clustering data samples of non-convex shape.

To solve these problems, this paper improves the clustering ability of k-means by measuring the similarity between samples and centroids by EVT (Coles et al., 2001 ).

In particular, we consider the generalized extreme value (GEV) (Jenkinson, 1955 ) distribution or generalized Pareto distribution (GPD) (Pickands III et al., 1975; DuMouchel, 1975) to transform the Euclidean space into a probability space defined as, extreme value space.

GEV and GPD are employed to model the maximum distance and output the probability that a distance is an extreme value, which indicates the similarity of a sample to a centroid.

Further, we adopt the Block Maxima Method (BMM) (Gumbel, 2012) to choose the maximal distance for helping GEV fit the data.

The Peaks-Over-Thresh (POT) method (Leadbetter, 1991 ) is utilized to model the excess of distance exceeding the threshold, and thus very useful in fitting the data for GPD.

Formally, since both GEV and GPD can measure the similarity of samples and centroids, they can be directly utilized in k-means, i.e., GEV k-means and GPD k-means, which are uniformly called Extreme Value k-means (EV k-means) algorithm.

In contrast to k-means, EV k-means is a probability-based clustering algorithm that clusters samples according to the probability output from GEV or GPD.

Furthermore, to accelerate the computation of Euclidean distance, We expand the samples and the centroids into two tensors of the same shape, and then accelerate with the high performance parallel computing of GPU.

For clustering steaming data, we propose online Extreme Value k-means based on Mini Batch kmeans (Sculley, 2010) .

When fit the GEV distribution, we use mini batch data as a block.

For the fitting of GPD, we dynamically update the threshold.

The parameters of GEV or GPD are learned by stochastic gradient descent (SGD) (LeCun et al., 1998) .

The main contributions are described as follows.

(1) This paper utilizes EVT to improve k-means in addressing the problem of clustering data of non-convex shape.

We thus propose the novel Extreme Value k-means, including GEV k-means and GPD k-means.

A method for accelerating Euclidean distance computation has also been proposed to solve the bottleneck of k-means.

(2) Under the strong theoretical support provided by EVT, we use GEV and GPD to transform Euclidean space into extreme value space, and measure the similarity between samples and centroids.

(3) Based on Mini Batch k-means, We propose online Extreme value k-means for clustering streaming data, which can learn the parameters of GEV and GPD online.

We corroborate the effectiveness of EV k-means and online EV k-means by conducting experiments on synthetic datasets and real datasets.

Experimental results show that EV k-means and online EV k-means significantly outperform compared algorithms consistently across all experimented datasets.

k-means and EVT have been extensively studied in the literature in many aspects (Jain, 2010; Rui Xu & Wunsch, 2005; De Haan & Ferreira, 2007) .

Previous work on k-means focused on the following aspects, such as determining the optimal k, initializing the centroids, and accelerating k-means.

Bandyopadhyay & Maulik (2002) ; Lin et al. (2005) ; Van der Merwe & Engelbrecht (2003) ; Omran et al. (2005) propose to select the optimal k value based on the genetic algorithm.

Initializing the centroids is a hot issue in k-means (Celebi et al., 2013) .

k-means++ (Arthur & Vassilvitskii, 2007) is the most popular initialization scheme.

Katsavounidis et al. (1994) ; Khan & Ahmad (2004) ; Redmond & Heneghan (2007) proposed density-based initial centroid selection method, that is, selecting the initial cluster center according to the density distribution of the samples.

Recently, Bachem et al. (2016) propose using Markov chain Monte Carlo to accelerate k-means++ sampling.

There is also a lot of work focused on solving the computational complexity of k-means.

Hamerly (2010) argued that using triangle inequality can accelerate k-means.

Sinha (2018) showed that randomly sparse the original data matrix can significantly speed up the computation of Euclidean distance.

EVT is widely used in many area, such as natural phenomena, finance, and traffic prediction.

In recent years, EVT has many applications in the field of machine learning.

However, far too little attention has been paid to the combination of k-means and EVT.

Li et al. (2012) proposes using generalized extreme value distribution for feature learning based on k-means.

However, our method is significant different from this method.

First, they compute the squared distance from a point to the nearest centroid and form a GEV regarding to each point, while we compute the distance from a centroid all data points and then fit the GEV or GPD regarding to each centroid.

Second, their algorithm adds the likelihood function as a penalty term into the objective function of k-means, but our algorithm changes the objective function by fitting the GEV or GPD for each centroid and assign the data point to the one with the highest probabilities they belong to.

Finally, this paper also presents GPD k-means and online Extreme Value k-means, which is not stated in this paper.

The sum squared error is defined as

(1) Eq. (1) indicates that the smaller J is, the higher degree of closeness between the samples and their centroids in the clusters, so the similarity of the samples in the clusters is higher.

To find the global minimum of Eq. 1, we need to compute all possible cluster partitions, so k-means is an NP-hard problem (Aloise et al., 2009 ).

Lloyd's algorithm (Lloyd, 1982) uses a greedy strategy to approximate the Eq. (1) by iteratively optimizing between assigning cluster labels and updating centroids.

Specifically, in assigning cluster labels, a cluster label is assigned to each sample according to the closest centroid.

When the centroids are being updated, each centroid is updated to the mean of all samples in the cluster.

These two steps loop iteratively until the centroids no longer change.

In this subsection, we first introduce the statistical aspects of a sample maximum in Extreme Value Theory (EVT), which is a branch of statistics dealing with the stochastic behavior of extreme events found in the tail of probability distribution.

Let X 1 , X 2 , . . .

, X n be a sample of independent copy of X with distribution F .

It is theoretically interesting to consider the asymptotic behavior of sample maximum and upper order statistics.

More specifically, denote M n = max 1≤i≤n X i as the sample maximum, whose distribution is

On the other hand, the upper order statistics of the sample is related to the survival function over a threshold u, which is

EVT considers the non-degenerated limit when n → ∞ in Eq.(2) and u ↑ x * in Eq.(3) by re-scaling the objects, which is presented as the conditions of the maximum domain of attraction for F .

Theorem 3.1 (Fisher-Tippett Theorem (Fisher & Tippett, 1928) ) A distribution function F satisfis the condition of a maximum domain of attraction: if there exists a constant ξ ∈ R and sequences a n > 0, b n , n ∈ N such that

The shape parameter ξ is called the extreme value index.

Theorem 3.1 motivates the Block Maxima Method (BMM) (Gumbel, 2012) : for the block size s ∈ {1, 2, . . .

, n}, divide the sample into m = n/s blocks of length s. Since the data is independent, each block maxima has distribution F s and can be approximated by a three-parametric generalized extreme value distribution (GEV) G GEV (·; µ, σ, ξ) when the block size s is large enough and the number of blocks m is sufficient.

The class of GEV distributions is defined as

We treat the case of ξ = 0 as the limit of ξ → 0.

An equivalent representation of the maximum domain of attraction condition is as follows:

Theorem 3.2 (Pickands-Balkema-de Haan Theorem (Balkema & De Haan, 1974) ) A distribution function F satisfies the condition of maximum domain of attraction: if there exists a constant ξ ∈ R and a positive function σ(t) such that

where x * denotes the right end-point of the support of F .

The clustering results of GPD k-means in three isotropic Gaussian blobs.

The color of the surface and contour in the figures represent the probability density of GPD.

The closer to yellow, the greater the probability density.

The closer to blue, the smaller the probability density.

The upper order statistics of a sample usually provides useful information about the tail of the distribution F .

Then Theorem 3.2 gives rise to an alternative peak-over-threshold (POT) approach (Pickands III et al., 1975) : given sufficient large threshold u in Eq. (7), we have that, for any X i > u, its conditional distribution can be approximated by a two-parametric generalized Pareto distribution (GPD) G GP D (·; σ, ξ), which is defined as

Similarly, we treat the case of ξ = 0 as the limit of ξ → 0.

The POT approach focuses on the excess over the threshold u to fit the GPD and asymptotically characterize the tail features of the distribution, while the BMM only approximates the GEV distribution when m is large enough.

The BMM only uses a very small amount of dataset, and there may be cases where the submaximal value of one block is larger than the maximum value of the other block, which cannot be utilized.

In contrast, POT method uses all data beyond the threshold to fit the GPD, making full use of the extreme data.

However, there is no winner in theory.

4 THE EXTREME VALUE k-MEANS ALGORITHM 4.1 MEASURING SIMILARITY BY EXTREME VALUE THEORY Measuring similarity with Euclidean distance is the core step of k-means clustering.

Similarly, for all clustering algorithms, how to measure the distance (dissimilarity) or similarity between the samples and the centroids is a critical issue (Rui Xu & Wunsch, 2005) as it determines the performance of the algorithm.

However, due to the properties of Euclidean distance, k-means fails for clustering non-convex clusters.

Therefore, this paper proposes to use the EVT to transform the Euclidean space into a probability space called the extreme value space.

Fig. 1(a) demonstrates measuring similarity by GEV or GPD.

The Euclidean distances from µ 1 and µ 3 to x t is much larger than the Euclidean distances from µ 1 and µ 3 to the most of surrounding samples, i.e. x t − µ 1 2 x i − µ 1 2 , x i ∈ C 1 and x t − µ 3 2 x i − µ 3 2 , x i ∈ C 3 .

Therefore, x t − µ 1 2 and x t − µ 3 2 are maximums concerning x i − µ 1 2 , x i ∈ C 1 and x i − µ 3 2 , x i ∈ C 3 with different degree.

We want a distribution that can be used to model maximum distance and reflect the probability that a distance is belong to a cluster, which equivalent to the similarity between the sample and the centroid.

Obviously, the EVT is a good choice.

As described in Section 3.2, the BMM can be applied to fit the GEV distribution.

In order to fit a GEV distribution for each cluster, we first compute the Euclidean distance d ij between Θ = {µ 1 , µ 2 , . . .

, µ k } and sample x i ∈ X , i.e., d ij = x i − µ j 2 , i ∈ {1, 2, . . .

, n}, j ∈ {1, 2, . . .

, k}. For the centroid µ j , its distances to all samples is denoted by d j = {d 1j , d 2j , . . .

, d nj }.

Then we divided them equally into m blocks of size s = n m (possibly the last block with no sufficient observations can be discarded), and then the maximum value of each block is taken to obtain the block maximum sequence M j .

We use M j to estimate the parameters of GEV distributions for cluster C j .

We assume the location parameter is zero for the reason that the position of centroids change small in the later stage of clustering.

The most commonly used estimating method, maximum likelihood estimation (MLE), is implemented to estimate the two parameters of the GEV.

The log likelihood function of GEV is derived from Eq. (5),

σj > 0 when ξ j = 0.

We get the estimated valueσ j andξ j of σ j and ξ j by maximizing L GEV .

Alternatively, we use the POT method to model the excess of Euclidean distance d j exceeding threshold u j for centroid µ j and fit the GPD.

We first compute the excess that is defined as

where k j is the total number of observations greater than the threshold u j .

Then we also implement MLE to estimate the parameters of the GPD.

The log likelihood function of GPD can be derived from Eq. (7),

0 when ξ j > 0 and 0 y j i − σj ξj when ξ j < 0.

We get the estimated valueσ j andξ j of σ j and ξ j by maximizing the L GP D .

Finally, we can obtain the probability that x i belong to cluster C j through the GEV and GPD:

The traditional k-means clusters samples in view of the closeness to the centroids of clusters.

As described in Section 4.1, we can model the distribution classes of GEV and GPD to measure the similarity between the samples and the centroids.

Thus we propose GEV k-means and GPD k-means, which are uniformly called the Extreme Value k-means (EV k-means) algorithm.

In contrast to k-means, the proposed EV k-means is a probability-based clustering algorithm as it instead clusters samples by the probability output from GEV or GPD.

The larger the block size s and the threshold u of BMM and POT, the smaller the deviation of MLE, but the larger the variance of MLE.

Conversely, the smaller the block size s and the threshold u, the larger the deviation of the MLE, but the smaller the variance of the MLE.

How to choose these two hyperparameters has not yet had a standard method, and it is necessary to comprehensively balance the relationship between deviation and variance in practical applications.

Therefore, we set the block size by grid search and set threshold adaptively.

Specifically, we first set the hyperparameter α to indicate the percentage of excess for all samples.

Then we sort d j from big to small, and the u is set to the αn-th of sorted d j .

The algorithm of GEV k-means has three steps: Given the dataset X , block size s and k initial centroids (obtained randomly or using k-means++ algorithm).

During the step of fitting a GEV distribution, we firstly use BMM to select the maximal sample data M j for µ j .

Then, we estimate the GEV parametersσ j andξ j by MLE using M j for µ j .

So each cluster has its own independent GEV distribution.

In the assigning labels step, each sample is assigned a cluster label based on the maximum probability, i.e., λ i = arg max j∈{1,2,...,k} P ij .

In the updating centroid step, each centroid is updated to the mean of all samples in the cluster, i.e., µ i = 1 |Ci| x∈Ci x. There three steps are iterated until the centroids no longer change.

The algorithm of GPD k-means is very similar to GEV k-means, except the fitting GPD distribution step.

GPD k-means use the POT to model the excess y j of Euclidean distance d j exceeding threshold u j and fit the GPD.

Fig. 1(b) and Fig. 1(c) show clustering results of GPD k-means in three isotropic Gaussian blobs and show that the closer to the centroids, the greater the probability density.

The main bottleneck of the k-means is the computation of Euclidean distances for the reason that the Euclidean distances between all samples and all centroids need to be computed.

In naïve implementation, double-layer nested for loop is often used to perform operations on the CPU, which is very slow.

This paper proposes an accelerated computation method to solve this bottleneck.

Firstly, let matrix X ∈ R n×d represents samples consisting of n d-dimensional samples, and matrix C ∈ R k×d represents centroids consisting of k d-dimensional centroids.

Secondly, insert a dimension between the two dimensions of matrix X and copy X along the new dimension to tensor X with shape of [n, k, d] .

A similar operation for matrix C, adding a new dimension before the first dimension and copy C along the new dimension to tensor C with shape of [n, k, d] .

Finally, the Euclidean distances between all samples and all centroids are D i,j = X − C 2 , i ∈ {1, 2, . . .

, n}, j ∈ {1, 2, . . .

, k} that can be accelerate with the advantages of GPU parallel computing.

The overall Extreme Value k-means algorithm is illustrated in Algorithm 1.

In the era of Big Data, data is no longer stored in memory, but in the form of streams (Bugdary & Maymon, 2019) .

Therefore, clustering streaming data is a significant and challenging problem.

It is indispensable to design an Extreme Value k-menas algorithm that can learn online for clustering streaming data.

This paper proposes the online Extreme Value k-means for clustering streaming data based on Mini Batch k-means (Sculley, 2010) .

When fit the GEV distribution, we use mini batch data as a block and choose the maximum value for learning the parameters of GEV online.

For the fitting of the GPD, the online EV k-means can dynamically update the threshold u and learn the parameters of GPD online.

The Online Extreme Value k-means algorithm is illustrated in Algorithm 2.

The algorithm randomly choose a mini batch contains b samples from the data stream each iteration.

On the first iteration, it initializes the parameters of each GEV or GPD, and initializes centroid C on the mini batch.

Then compute the Euclidean distances D using the accelerated computation method we proposed, update u j to tαn-th of sorted h, and compute the maximum M j and excess y j .

Because the GEV and GPD parameters have not been updated at the first iteration, so P ij cannot be computed.

Therefore, from the second iteration, the algorithm clusters the mini batch samples based on Mini Batch kmeans.

Finally, the negative log-likelihood function of all GEVs or GPDs is summed to obtain L s , and the L s is minimized by SGD to update the parameters of GEV or GPD, which is equivalent to maximizing

We evaluate the performance of the clustering algorithm by three widely used metrics, unsupervised clustering accuracy (ACC) (Cai et al., 2010) , normalized mutual information (NMI) (Vinh et al., 2010) , and adjusted rand index (ARI) (Vinh et al., 2010) .

Note that the values of ACC and NMI are

Input: samples X ∈ R n×d , number of cluster k, block size s for GEV k-means, the percentage of excess α for GPD k-means Output: clusters C Initialize centroid C ∈ R k×d ; repeat Cj = ∅, 1 j k; Perform transformation on X and C to obtain X and C, and then compute D = X − C 2; for j = 1, 2, . . .

, k do // GEV k-means Obtain M j from d:.j by BMM;

Estimate theσj,ξj by MLE on M j ; // GPD k-means Obtain y j from D:,j by POT;

Estimate theσj,ξj by MLE on y j ; end for i = 1, 2, . . .

, n do λi = arg max j∈{1,2,...,k} Pij;

x; end until centroids no longer change; return clusters C; LGEV (M j ; σj, ξj) // online GEV k-means

LGP D (y j ; σj, ξj) // online GPD k-means Compute the gradient Ls and then update the parameters of the GEV or GPD; end return centroid C ; Figure 2 : Visualization of six synthetic datasets shows the result of our Extreme Value k-means compared to k-means, k-means++, k-medoid, bidecting k-means and spectral clustering.

The results from top to down are the clustering results on the datasets D1, D2, D3, D4, D5 and D6, respectively.

The nine algorithms from the first column to the ninth column are respectively k-means, k-means++, k-medoid, bidecting k-means, spectral clustering, GEV k-means (RM), GEV k-means (++), GPD k-means (RM), GPD k-means (++).

in the range of 0 to 1, with 1 indicating the best clustering and 0 indicating the worst clustering.

The value of ARI is in range of -1 to 1, -1 indicates the worst clustering, and 1 indicates the best clustering.

We demonstrate our algorithm compared to other algorithms on six two-dimensional synthetic datasets.

As illustrated in Fig. 2 , there are the clustering results of the datasets D1, D2, D3, D4, D5 and D6 from top to down.

D1 consists of 5 isotropic Gaussian clusters, each of which has 100 samples.

D2 consists of 15 isotropic Gaussian clusters, each of which has 4000 samples.

D3 consists of two 'C'-shaped clusters in the same direction, each of which has 250 samples.

D4 consists of two clusters, each of which has 500 samples including a Gaussian blob and a 'C'-shaped region.

D5 consists of a Gaussian cluster having 500 samples and a 'C'-shaped cluster having 250 samples.

The difference between D5 and D4 is that the lower cluster in D5 has no 'C'-shaped region, and Gaussian blobs has a larger variance.

In D5, the upper cluster has 500 samples and the lower cluster has 250 samples.

The centroids of GEV k-means and GPD k-means can be initialized randomly or using k-means++.

Let 'RM' and '++' denote randomly and using k-means++ initialize centroids, respectively.

Therefore, there are nine algorithms in this experiment, k-means, k-means++, k-medoid (Kaufman & Rousseeuw, 2009) , bisecting k-means (Steinbach et al., 2000) , spectral clustering (Ng et al., 2002) , GEV k-means (RM), GEV k-means (++), GPD k-means (RM), GPD k-means (++), respectively.

From the clustering results of the nine algorithms on six different synthetic datasets in Fig. 2 , it can be seen that our four algorithms can successfully cluster convex and non-convex clusters, but the clustering results of k-means and k-means++ on non-convex but visibly well-separated clusters are completely unsuccessful.

In addition, the clustering results k-medoid, bisecting k-means, spectral clustering on D3, D4, D5 is worse than our four algorithms.

We evaluated the proposed EV k-means on nine real datasets: iris (n = 150, d = 4, k = 3), breast cancer (n = 683, d = 10, k = 2), live disorders (n = 145, d = 5, k = 2), heart (n = 270, d = 13, k = 2), diabetes (n = 768, d = 8, k = 2), glass (n = 214, d = 9, k = 6), vehicle (n = 846, d = 18, k = 4), MNIST and CIFAR10.

The first seven datasets are available from LIBSVM Data website 1 .

MNIST is a dataset comprises 60,000 training gray-scale images and 10,000 gray-scale images of handwritten digits 0 to 9.

Each of the training images is represented by an 84-dimensional vector obtained by LeNet (LeCun et al., 1998) .

So the MNIST dataset we use has 60,000 samples with 84 features belonging to 10 classes, i.e., n = 60, 000, d = 84, k = 10.

CIFAR10 is a dataset containing 50,000 taining and 10,000 test color images with 32 × 32 pixels, grouped into 10 different classes of equal size, representing 10 different objects.

Each of the training images is represented by a 512-dimensional vector extracted by a ResNet-18 (He et al., 2016) .

Therefore, the CIFAR10 we use in the experiment has 50,000 samples with 512 features grouped in 10 classes, i.e., n = 50, 000, d = 84, k = 10.

We repeat each experiment 10 times with different random seeds and took the mean of the results of 10 times experiments as the final result.

In each of the experiments, all algorithms that initialize centroids randomly or by using k-means++ start from the same initial centroids.

The results of EV k-means on real datasets are shown in Tab.

1.

As shown in Tab.

1, our proposed EV k-means on some datasets are comparable to other algorithms, and outperform other algorithms on MNIST and CIFAR10.

We compare online EV k-means with k-means, k-means++, Mini Batch k-means (RM) and Mini Batch kmeans (++) on MNIST and CIFAR10.

As illustrated in Tab.

2, the values of the three metrics of online EV k-means are slightly smaller than the values of EV k-means.

However, the values of the three metrics of Mini Batch k-means are much smaller than the values of k-means.

For example, the values of the three metrics of Mini Batch k-means on MNIST are 10%, 17%, 8% smaller than the values of k-means.

However, the values

<|TLDR|>

@highlight

This paper introduces Extreme Value Theory into k-means to measure similarity and proposes a novel algorithm called Extreme Value k-means for clustering.