We propose a neural clustering model that jointly learns both latent features and how they cluster.

Unlike similar methods our model does not require a predefined number of clusters.

Using a supervised approach, we agglomerate latent features towards randomly sampled targets within the same space whilst progressively removing the targets until we are left with only targets which represent cluster centroids.

To show the behavior of our model across different modalities we apply our model on both text and image data and very competitive results on MNIST.

Finally, we also provide results against baseline models for fashion-MNIST, the 20 newsgroups dataset, and a Twitter dataset we ourselves create.

Clustering is one of the fundamental problems of unsupervised learning.

It involves the grouping of items into clusters such that items within the same cluster are more similar than items in different clusters.

Crucially, the ability to do this often hinges upon learning latent features in the input data which can be used to differentiate items from each other in some feature space.

Two key questions thus arise:

How do we decide upon cluster membership?

and How do we learn good representations of data in feature space?Spurred initially by studies into the division of animals into taxa BID31 , cluster analysis matured as a field in the subsequent decades with the advent of various models.

These included distribution-based models, such as Gaussian mixture models BID9 ; densitybased models, such as DBSCAN BID11 ; centroid-based models, such as k-means.2 and hierarchical models, including agglomerative BID29 and divisive models BID13 .While the cluster analysis community has focused on the unsupervised learning of cluster membership, the deep learning community has a long history of unsupervised representation learning, yielding models such as variational autoencoders BID21 , generative adversarial networks BID12 , and vector space word models BID28 .In this paper, we propose using noise as targets for agglomerative clustering (or NATAC).

As in BID1 we begin by sampling points in features space called noise targets which we match with latent features.

During training we progressively remove targets and thus agglomerate latent features around fewer and fewer target centroids using a simple heuristic.

To tackle the instability of such training we augment our objective with an auxiliary loss which prevents the model from collapsing and helps it learn better representations.

We explore the performance of our model across different modalities in Section 3.Recently, there have been several attempts at jointly learning both cluster membership and good representations using end-to-end differentiable methods.

Similarly to us, BID37 use a policy to agglomerate points at each training step but they require a given number of clusters to stop agglomerating at.

BID23 propose a form of supervised neural clustering which can then be used to cluster new data containing different categories.

BID25 propose jointly learning representations and clusters by using a k-means style objective.

BID36 introduce deep embedding clustering (DEC) which learns a mapping from data space to a lower-dimensional feature space in which it iteratively optimizes a clustering objective (however, as opposed to the hard assignment we use, they optimize based on soft assignment).Additionally, there have been unsupervised clustering methods using nonnegative low-rank approximations BID40 which perform competitively to current neural methods on datasets such as MNIST.Unlike all of the above papers, our method does not require a predefined number of clusters.

We begin by discussing the use of noise as targets (NAT) introduced in BID1 which is crucial to the understanding of our model.

We then describe the intuition behind our approach, then proceed to describe the mechanism itself.

Bojanowski & Joulin (2017) proposed a new form of unsupervised learning called "Noise as Targets" (NAT), aimed at extracting useful features from a set of objects.3 Roughly speaking, this approach selects a set of points, referred to as "targets" uniformly at random from the unit sphere.

It then aims to find a mapping from the raw representations of objects to points on the unit sphere, so that these points are close to the corresponding targets; the correspondence (matching) between mapped representations and the target points is done so as to minimize the sum of the distances between the mapped points and the targets.

The intuition behind the NAT approach is that the model learns to map raw inputs to the latent space in a way that both covers the entire latent space well, and that places "similar" inputs in neighborhoods of similar targets.

More formally, the NAT task is to learn an encoder function f ?? : X ??? Z from input space to a latent representation space.

The objective is to minimize the L 2 loss between representations z i ??? Z, where points in Z are unit normalized, and corresponding targets y k ??? Y where the targets are uniformly sampled from the L 2 unit sphere (thus inhabiting the same space as the z i ).Instead of being tied to corresponding representations as in classic regression, during the model fitting, inputs are considered in batches.

Each batch consists of inputs and targets; when processing a batch the targets in the batch are permuted so as to minimize the batch-wise loss.

To compute the optimal (loss-minimizing) one-to-one assignment of latent representations and targets of a batch, the Hungarian Method is used BID22 .

This target re-assignment pushes the representations z i of similar inputs x i ??? X to neighborhoods of similar targets y k .Importantly, every example must be paired to a single noise target.

Minimizing the L 2 loss requires each latent representation to be close to its assigned target.

Therefore, the model learns a mapping to latent space that very closely matches the distribution of the noise targets.

The motivation behind using NAT was to learn unsupervised features from input data.

The authors show their method performs on par with state-of-the-art unsupervised representation learning methods.

Viewed from a clustering perspective we can think of the targets y i as cluster centroids to which the latent representations z i are (one-to-one) assigned.

Note that although the method introduced in BID1 brings the representations of similar x i closer together (by matching and moving them closer to neighborhoods of similar targets) it cannot produce many-to-one matchings or match multiple similar z i with a single centroid thus forming a cluster.

Simply changing the re-assignment policy to allow for many-to-one matchings is difficult because it causes the model to In this case, The delete-and-copy policy removes the target on the top-left of the sphere and copies the target at the top-right.

This leads to an agglomeration of the two latent representations at the top of the sphere into the same cluster.collapse in on a single target.

In this paper we use the above to propose a new form of neural clustering where we progressively delete targets over time and re-assign representations to other nearby targets, all while keeping the model stable using an auxiliary objective.

Delete-and-copy To be able to cluster latent representations we need a way of assigning them to cluster centroids.

We propose doing this via an additional delete-and-copy step which allows for many-to-one matchings.

Similarly to the NAT method, we first assign representations z to targets y using the Hungarian method.

In some cases, the optimally assigned target y opt is not the nearest target to a latent representation z. In this case, we remove the assigned target and reassign a copy of the nearest target for that z with some probability ?? as in Algorithm 1 (see also FIG1 ).

This has the effect of not only reassigning targets so as to minimize the distance between matched pairs, but also to encourage the model to allow similar examples to be assigned to the same target.

The new assignments are denoted as y new i .

The loss is then defined as: DISPLAYFORM0 Auxiliary objective To prevent the model from collapsing to a single point, we introduce an auxiliary objective in addition to the loss between representations and targets.

In our case we set the auxiliary objective L aux to be the reconstruction loss i x i ??? f dec (z i ) 2 where f dec is some decoder network.

The final objective L is then a weighted sum of the NAT loss L NAT (which in our case is the L 2 loss) and the auxiliary objective L aux : DISPLAYFORM1 Importantly, the auxiliary objective not only prevents the model from collapsing, it also informs how the model clusters.

For example, when clustering images the reconstruction loss encourages the model to cluster on similar pixel values.

Alternative forms of auxiliary objectives could allow for a discriminator loss or a classification loss for tackling semi-supervised settings.

As our goal is unsupervised clustering, we only consider the reconstruction loss.

Algorithm 1: The delete-and-copy policy used in our experiments.

Input : A batch of latent representations z, the optimal NAT assignment y opt , and a probability of copying ?? Output: A new NAT assignment y new = y Model definition During initialization each example x i in the dataset is paired with a random target y i , uniformly sampled from a d-dimensional sphere.

The NATAC model is then trained using minibatches from the dataset.

Each training step can be broken down as follows: DISPLAYFORM2 The examples x from a random batch of example-target pairs DISPLAYFORM3 2.

Re-assignment step: Using the Hungarian method from BID22 , the representations z are optimally one-to-one matched with targets y so as to minimize the total sum of distances between matched pairs in the batch.

The newly assigned example-target pair- DISPLAYFORM4 has the permutation of labels within the batch to minimize the batch-wise loss.3.

Delete-and-copy (DnC): With a probability ??, delete the optimal target y

The L 2 loss between targets and latent representations is taken and combined with the auxiliary loss.

Gradients w.r.t.

?? are then taken and back-propagated along.

Notice that although the (re)-assignment step follows a non-differentiable policy, the model is still end-to-end differentiable.

Finally, the new example-target assignments are kept after the training step, and persist into the next training step where they are reassigned again.

Stopping criterion During training the number of unique targets is tracked.

We stop training when the number of unique targets stops decreasing after an epoch of training.

Multi-stage training We found that an initial period of training where the auxiliary objective is prioritized (i.e. the NAT loss is multiplied by a very small coefficient) and the DnC policy is not used, improved overall performance.

Transitioning to a higher NAT loss and turning on the deleteand-copy policy later on in training increased the stability of our model.

We therefore propose training NATAC models as follows:1.

Warm-Up stage: ?? = 0 and ?? is very small.2.

Transition stage: ?? increases gradually from 0 to 1, ?? also increases gradually to a larger value (approximately 100?? larger than its initial value).3.

Clustering stage: ?? = 1, ?? is large.

Table 1 : NMI scores from varying the dimensionality of the latent space in the NATAC and baseline models.

The baselines use k-means with the same number of clusters as the repsective NATAC model converged to.

We include NATAC models with a latent dimensionality of d = 3, whose latent representations can be viewed without dimensionality reduction.

Appendix A contains links to the visualizations hosted on the TensorFlow embedding projector.

DISPLAYFORM0 Dimensionality of the latent space In all of our experiments, we found that the best performing models tend to have a latent space dimensionality between 4 and 12.At dimensionalities much larger than this, the model collapses to very few points during the transition stage of training, possibly due to the high expressiveness of the latent space.

On the other hand, using a low dimensional representation results in an information bottleneck too small to sufficiently learn the auxiliary objective.

For example, when clustering tweets from our Twitter dataset, a latent space of two dimensions was too small for the decoder to reliably reconstruct tweets from latent vectors.

With an auxiliary objective that cannot be effectively learned, centroids collapse to a single point.

We now describe the datasets and evaluation metrics used in our experiments followed by the presentation and analysis of our results in comparison to others.

The full details regarding the hyperparameters used in our experiments can be found in appendix D.Datasets We evaluate our models on four different datasets -two image and two text datasets.

For images we use MNIST BID24 and Fashion-MNIST BID35 .

For text we use 20 Newsgroups BID19 and a Twitter dataset which we gather ourselves.

Our key evaluation metric is the normalized mutual information (NMI) score BID33 which measures the information gained from knowing cluster membership whilst taking the number of clusters into account.

Values of NMI range from 0, where clustering gives no information, to 1, where clustering gives perfect information i.e. cluster membership is identical to class membership.

In our experiments, we train models on the concatenation of the train and validation sets and evaluate them on the test set.

This is done by computing the latent representations of test examples and then assigning them to their nearest respective centroids, then computing the NMI score.

Additionally, we provide classification error scores on the MNIST dataset to compare ourselves to other related methods.

We also compare our model to clustering methods trained on the 20 Newsgroups dataset.

The guiding motivation behind our experiments is to analyze how well our models learns cluster membership and latent representations.

Introduced by BID24 , MNIST is canonical dataset for numerous machine learning tasks including clustering.

MNIST has ten classes, corresponding to the ten digits 0-9, and contains 60,000 train and 10,000 test examples.

We train our model with an auxiliary reconstruction loss and use small convolutional architectures (see Figure 5 ) for our encoder and decoder networks.

As points of comparison we also provide results of using k-means on the latent representations learned by our model (NATAC-k in Table 1 ) and k-means on representations learned by a simple autoencoder with the same encoder and decoder architecture (AE-k in Table 1 ).

Table 1 shows our model's performance is best when d = 10 and worse for much lower or higher values of d. This indicates that the dimensionality of the latent space impacts our model's performance (see Section 2.3 for further discussion).The ability of our model to cluster MNIST examples well is shown by two keys results.

First, it beats both NATAC-k and AE-k (Table 1) .

Second, it achieves very competitive results when compared to other methods (see NMI column in Table 2 ).

To re-iterate, other clustering techniques cited in Table 2 required a predefined number of clusters (in the case of MNIST k=10).We note that NATAC-k beats AE-k indicating that our model learns representations that suit k-means clustering more than a simple autoencoder.

However, we note that this is not consistent across all modalities in this paper (see results in section 3.4).Finally, we discuss the number of centroids our model converges on (see Table 1 ).

We show that our model is successfully capable of finding centroids that represent different digits, as shown in the top row of FIG4 .

However, the model also learns centroids that contain very few examples for which the decoded images do not represent any handwritten digits, as shown in the second row of FIG4 .

Even with these "dead centroids", the model still performs well.

Indeed, the twelve most dense centroids contain 98% of all of the examples in MNIST (out of a total of 61).Interestingly, the model also differentiates between ones with different slopes.

This suggests that the latent representations of these digits are sufficiently far apart to warrant splitting them into different clusters.

Introduced in BID35 , Fashion-MNIST is a convenient swap-in dataset for MNIST.

Instead of digits the dataset consists of ten different types of clothes.

There are 60,000 train and 10,000 test examples just like in MNIST.

Fashion-MNIST is generally considered more difficult than MNIST, with classifiers scoring consistently lower on it.

The model and analysis from the previous section carry over for fashion-MNIST with a few additional important points.

First, the differences between NATAC-k and AE-k are less pronounced (see Table 1 ) in fashion-MNIST which indicates that the representations learned by NATAC in comparison to a simple autoencoder are not as important for k-means clustering.

Interestingly, our model still outperforms both NATAC-k and AE-k, with one exception being when d = 12.Qualitatively, FIG4 shows that the model separates garments into slightly different categories than the labels in the dataset.

For example, the most dense cluster seems to be a merging of both "pullovers" and "shirts", suggesting that the model finds it difficult to separate the two different garments.

We ourselves find it difficult to discriminate between the two categories, as the lowresolution images do not easily show whether or not the garment has buttons.

Additionally, the "sandal" class has been split into two separate clusters: flip-flops and high-heeled shoes with straps.

This indicates that our model has found an important distinction between these two type of shoes, that the original Fashion-MNIST labels ignore.

Similarly to MNIST, our model also learns "dead clusters", which the model does not decode into any discernible garment.

Further visualizations of these experiments can be found appendix section A.

Introduced in BID19 the 20 Newsgroups dataset is a collection of 18,846 documents pertaining to twenty different news categories.

We use the commonly used 60:40 temporal train-test split.

Interestingly, because of the temporal split in the data, the test set contains documents which differ considerably from the train set.

We calculate NMI on the news categories in the dataset.

BID27 -4.10 Deep Gaussian Mixture VAE BID6 -3.08 IMSAT Hu et al. (2017) -1.6 Autencoder based clustering BID32 0.669 -Task-specific Clustering With Deep Model BID34 0.651 -Agglomerative Clustering Using Average Linkage BID18 0.686 -Large-Scale Spectral Clustering BID4 0 Table 2 : Comparison of our best performing NATAC model (with d = 10) on the entire MNIST dataset.

NMI and classification error are calculated from the entire data set.

We report the evaluation metric used by the authors of each respective model.

Precision of values are the same as those reported by the original paper.

Note that many of the best-performing methods (DCD, IMSAT, Adversarial Autoencoders) also assume a uniform class distribution along with a pre-set number of clusters.

We use an auxiliary reconstruction loss and a two layer fully connected network for both the encoder and decoder, both with hidden layer sizes of 256 and ReLU nonlinearities.

We represent each article as an L2 normalized term-frequency-inverse-document-frequency (TF-IDF) vector of the 5000 most occurring words in the train set.

BID15 0.08 LSD BID0 0.44 MTV BID2 0.13 PLSI BID16 0.47 SSC BID10 0.29 LSC BID5 0.48 PNMF BID38 0.37 Ncut BID30 0.52 ONMF BID7 0.38 NSC BID8 0.52 k-means BID26 0.44 DCD BID40 0.54 NATAC Autoencoder (425 clusters) 0.479 Table 4 : Comparison of our best performing NATAC model (with d = 4) on the entire 20 Newsgroups dataset.

NMI is calculated from the entire data set.

Figures for other methods taken from BID40 Along with NATAC-k and AE-k comparisons, we also use a spherical k-means model.

Spherical k-means is a commonly used technique of unsupervised document clustering, a good description of it can be found in BID3 .

TAB2 shows how the performance of the each model varies with different dimensionalities of the latent space.

The best NATAC models with a latent dimensionality of 3 to 6 centroids outperform a spherical k-means model with 1000 clusters, far more clusters than any of the NATAC models.

Although we could not find any neural clustering techniques which report performance on 20 Newsgroups, many non-neural methods report NMI on the whole dataset.

Table 4 shows the NATAC model performs competitively to these methods.

However, our method does converge on a higher number of clusters (other methods are trained with a pre-defined number of 20 clusters).

To further explore the performance of our model on text data we build a dataset of 38,309 ASCIIonly tweets of English speaking users containing exactly one hashtag.

The dataset has 647 different hashtags, with each hashtag having at least ten tweets containing it.

We use 10, 000 of the tweets as the test set.

As a preprocessing step, URLs and hashtags are replaced with special characters.

We calculate NMI on the hashtag used in each tweet.

We train a character-based Sequence-to-Sequence autoencoder on the Twitter dataset.

Just as before we use an auxiliary reconstruction loss.

We set the encoder to be a bidirectional GRU with a hidden size of 64 followed by a fully connected layer which takes the GRU's final output and maps it to the latent space.

The decoder uses a fully connected layer to map the latent vectors to a 128 dimensional vector which is then used as the initial hidden state for the decoder GRU.Similarly to section 3.4, we compare our approach to using spherical k-means along with the NATAC-k and AE-k baselines.

As shown in table 3.5, we see that NATAC-k outperforms NATAC and AE-k models on all of the reported latent dimensionalities.

This suggests that the latent mapping learned by the NATAC models does improve on a vanilla autoencoder, but the centroid assignment of a trained NATAC model is less effective than using k-means.

Finally, all of the neural models outperform the spherical k-means baseline.

However, this baseline is much more competitive to the neural methods reported in this experiment than those reported in the 20 Newsgroups experiments.

In this section, we explore how sensitive the NATAC framework is to changes in our hyperparameters.

For these experiments, we use the model and dataset from the 20 Newsgroups experiments.

We take the best performing hyperparameters from these experiments (d = 4, see TAB2 ) and observe how the end performance of the model is affected by changing the hyperparameters.

We show that our method is reasonably robust to differing values of hyperparameters, although extreme changes, such as skipping pre-training, do adversely affect performance.

The NATAC training method contains four sources of randomness: (1) parameter initialization (2) the delete-and-copy policy (3) random batching (4) the sampling of noise as targets.

We train 50 NATAC models with the same hyperparameters (but different random seeds) and measure the variation in NMI and the number of converged clusters.

Figure 3 shows the variability of NMI and the converged number of clusters from training the best performing model on the 20 Newsgroups dataset.

We observe that the NMI varies with a small standard deviation of 0.007 (mean 0.465) regardless of how many clusters the model converged to.

In contrast, we observe a higher relative standard deviation of 24 with the converged number of clusters (mean 420).

Qualitatively, we observe that the variance in the number of converged clusters is mostly due to dead centroids.

The value of ?? varies throughout the training of a NATAC model.

In our experiments, we initially set ?? to zero for a period of pre-training, after which we incrementally increase the value to 1 over several epochs.

TAB5 : Mean NMI and converged number of centroids when training a NATAC model with varying amounts of pre-training.

Mean taken from 5 consecutive runs using the same hyperparameters.

Models trained on the train set of 20 Newsgroups and evaluated on the test set.??final \??inital 10 but after 100 epochs of pre-training, the model does not significantly benefit from any more pretraining.

Interestingly, the longer the period of pre-training, the more clusters the model converges to.

We believe that models which have longer to pre-train before clustering learn a more uniform mapping to latent space.

When the clustering phase of training occurs, the latent representations are more uniformly spread across latent space, and thus agglomerate less readily.

Alongside changing the value of ??, the coefficient for the NAT loss in NATAC models is also varied.

Similarly to ??, we set ?? to a small value for warm-up stage of training, and then progressively increase ?? to a larger value afterwards.

In the other experiments involving 20 Newsgroups, warm-up training uses ?? inital = 10 ???4 and a final value of ?? final = 10 ???2 .

The transition happens at the same time as the change in ?? during training.

TAB6 shows how the final NMI of NATAC models vary with differing values for ?? inital and ?? final .

We notice that the value of ?? final does not seem to greatly impact the number of clusters or NMI, and that the smaller values of ?? inital have very similar NMI scores and number of clusters.

Interestingly, when trained with large ?? inital of 10 ???3 , the model scores higher in NMI than models with smaller ?? inital and also converges to more clusters.

Here, we believe that a larger ?? inital forces the model to learn a more uniform mapping to latent space (to minimize the NAT loss) during the warm-up stage of training.

Similar to increasing the length of pre-training, this causes the model to agglomerate less readily.

In this paper, we present a novel neural clustering method which does not depend on a predefined number of clusters.

Our empirical evaluation shows that our model works well across modalities.

We show that NATAC has competitive performance to other methods which require a pre-defined number of clusters.

Further, it outperforms powerful baselines on Fashion-MNIST and text datasets (20 Newsgroups and a Twitter hashtag dataset).

However, NATAC does require some hyperparameters to be tuned, namely the dimensionality of the latent space, the length of warm-up training and the values for the loss coefficient ??.

However, our experiments indicate that NATAC models are fairly robust to hyperparameter changes.

Future work Several avenues of investigation could flow from this work.

Firstly, the effectiveness of this method in a semi-supervised setting could be explored using a joint reconstruction and classi-fication auxiliary objective.

Another interesting avenue to explore would be different agglomerative policies other than delete-and-copy.

Different geometries of the latent space could also be considered other than a unit normalized hypersphere.

To remove the need of setting hyperparameters by hand, work into automatically controlling the coefficients (e.g. using proportional control) could be studied.

Finally, it would be interesting to see whether clustering jointly across different feature spaces would help with learning better representations.

B EXAMPLES FROM THE FASHION-MNIST DATASET.

We experimented with using polar coordinates early on in our experiments.

Rather than using euclidean coordinates as the latent representation, z is considered a list of angles ?? 1 , ?? 2 ?? ?? ?? ?? n where ?? 1 ?? ?? ?? ?? n???1 ??? [0, ??] and ?? n ??? [0, 2??].

However, we found that the models using polar geometry performed significantly worse than those with euclidean geometry.

Additionally, we also experimented with not L2 normalizing the output of the encoder network.

We hypothesized that the model would learn a better representation of the latent space by also "learning" the geometry of the noise targets.

Unfortunately, the unnormalized representation caused the noise targets to quickly collapse to a single point.

Although each different modality (monochrome images, bag-of-words, sequence of characters) uses a different set of hyperparameters, we follow a similar recipe for determining the values for each one:??? We use a large batch size of 100.

This is so each batch has a representative sample of the targets to reassign in each training step.??? The warm-up period is calculated by observing when the auxiliary objective starts to converge.??? The final value for ?? in training is set so that the NAT loss was approximately 1% of the total loss.

??? The initial value for ?? is set as approximately 1% of the final value of ??.??? The transition phase typically lasts 100 epochs of training.??? During the transition phase, the value of ?? is incrementally increased from 0 to 1.We now explicitly list the hyperparameters used for each experiment:

??? A batch size of 100.??? A warm-up period of 10 ?? d epochs, during which ?? = 0.001.??? A transition period lasts for 250 epochs, where ?? is incrementally increased to 0.25, and ?? is incremented from 0 to 1.

??? The ADAM optimizer BID20 ) with a learning rate (?? = 10 ???4 ) Figure 5 : Architecture of the encoder (left) and decoder (right) used for the MNIST experiments.

Between each subsampling layer in the encoder, a single convolution layer is applied with a filter shape of 3 ?? 3 with border padding to keep the same shape before and after the convolution.

Similarly, in the decoder one transpose convolutional layer is applied between each upsampling layer, 3 ?? 3 filter shape and shape-preserving padding.

??? A batch size of 100.??? A warm-up period of 1, 000 epochs, during which ?? = 10 ???4 .???

A transition period lasts for 100 epochs, where ?? is incrementally increased to 0.01, and ?? is incremented from 0 to 1.??? The ADAM optimizer BID20 ) with a learning rate (?? = 10 ???5 ).???

Dropout with a keep-probability of 0.95 in the hidden layers of the encoder and decoder.

??? A batch size of 100.??? A warm-up period of 100 epochs, during which ?? = 0.01.??? A transition period lasts for 100 epochs, where ?? is incrementally increased to 1, and ?? is incremented from 0 to 1.??? The ADAM optimizer BID20 ) with a learning rate (?? = 10 ???3 ).

<|TLDR|>

@highlight

Neural clustering without needing a number of clusters