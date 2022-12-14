In this paper we propose a Deep Autoencoder Mixture Clustering (DAMIC) algorithm.

It is based on a mixture of deep autoencoders where each cluster is represented by an autoencoder.

A clustering network transforms the data into another space and then selects one of the clusters.

Next, the autoencoder associated with this cluster is used to reconstruct the data-point.

The clustering algorithm jointly learns the nonlinear data representation and the set of autoencoders.

The optimal clustering is found by minimizing the reconstruction loss of the mixture of autoencoder network.

Unlike other deep clustering algorithms, no regularization term is needed to avoid data collapsing to a single point.

Our experimental evaluations on image and text corpora show significant improvement over state-of-the-art methods.

Effective automatic grouping of objects into clusters is one of the fundamental problems in machine learning and data analysis.

In many approaches, the first step toward clustering a dataset is extracting a feature vector from each object.

This reduces the problem to the aggregation of groups of vectors in a feature space.

A commonly used clustering algorithm in this case is the k-means.

Clustering high-dimensional datasets is, however, hard because inter-point distances become less informative in high-dimensional spaces.

As a result, representation learning has been used to map the input data into a low-dimensional feature space.

In recent years, motivated by the success of deep neural network in supervised learning, there are many attempts to apply unsupervised deep learning approaches for clustering.

Most methods are focused on clustering over a low-dimensional feature space of an autoencoder or a variational autoencoder.

Recent good overviews of deep clustering methods can be found in BID1 and BID14 .Using deep neural networks, it is possible to learn nonlinear mappings allowing to transform the data into more clustering-friendly representations.

A deep version of k-means is based on learning a data representation and applying k-means in the embedded space.

A straightforward implementation of the deep k-means algorithm would lead, however, to a trivial solution where the features are collapsed to a single point in the embedded space and the centroids are collapsed into a single entity.

The objective function of most deep clustering algorithms is, therefore, composed of a clustering term computed in the embedded space and a regularization term in the form of reconstruction error to avoid data collapsing.

Deep Embedded Clustering (DEC) BID16 ) is first pre-trained using an autoencoder reconstruction loss and then optimizes cluster centroids in the embedded space through a Kullback-Leibeler divergence loss.

Deep Clustering network (DCN) BID17 ) is another autoencoder-based method that uses k-means for clustering.

Similar to DEC, in the first phase, the network is pre-trained using the autoencoder reconstruction loss.

In the second phase, in contrast to DEC, the network is jointly trained using a mathematical combination of the autoencoder reconstruction loss and the k-means clustering loss function.

Thus, due to the fact that strict cluster assignments were used during the training (instead of probabilities such as in DEC) the method requires an alternation process between the network training and the cluster updates.

In this paper we propose an algorithm to perform unsupervised clustering within the mixture-ofexperts framework BID8 ).

Each cluster is represented by an autoencoder neuralnetwork and the clustering itself is performed in a low-dimensional embedded space by a softmax classification layer that directs the input data to the most suitable autoencoder.

Unlike most deep clustering algorithms the proposed algorithm is deep in nature and not a deep variant of a classical clustering algorithm.

The proposed algorithm does not suffer from the clustering collapsing problem and therefore there is no need for regularization terms that have to be tuned separately for each dataset.

Note that parameter tuning in clustering is problematic since it is based, either explicitly or implicitly, on the data labels which are not supposed to be available in the clustering process.

Another major difference of the proposed method from previous approaches is the learning method of the embedding latent space where the actual clustering operation is taking place.

In most previous methods, the embedded space is controlled by an autoencoder.

Thus, in order to gain a good reconstruction, it requires to encode into the embedded space information that can be entirely irrelevant to the clustering process.

In contrast, in our algorithm no decoding is applied to the clustering embedded space and the only goal of the embedded space is to find a good organization of the data into separated clusters.

We validate the method on standard real datasets including various document and image corpora.

Evidently, visible improvement from the respective state-of-art is observed for all the tested datasets.

The contribution of this paper is twofold: (i) a novel deep learning clustering method that unlike deep variants of k-means, does not require a tuned regularization term to avoid clustering collapsing to a single point; and (ii) state-of-the-art performance on standard datasets.

Consider the problem of clustering a set of n points x 1 , . . .

, x n ??? R d into k clusters.

The k-means algorithm represents each cluster by a centroid.

In our approach, rather than representing a cluster by a centroid, we represent each cluster by an autoencoder that is specialized in reconstructing objects belonging to that cluster.

The clustering itself is carried out by directing the input object to the most suitable autoencoder.

We next formally describe the proposed clustering algorithm.

The algorithm is based on a (soft) clustering network that produces a distribution over the k clusters: DISPLAYFORM0 such that ?? c is the parameter set of the clustering network, h(x) is a nonlinear representation of a point x computed by the clustering network and w 1 , . . .

, w k , b 1 , . . . , b k ??? ?? c are the parameters of the softmax output layer.

The (hard) cluster assignment of a point x is thus: DISPLAYFORM1 The clustering task is, by definition, unsupervised and therefore we cannot directly train the clustering network.

Instead, we use the clustering results to obtain a more accurate reconstruction of the network input.

We represent each cluster by an autoencoder that is specializing in reconstructing instances of that cluster.

If the dataset is properly clustered, we expect that all the points assigned to the same cluster are similar and hence the task of a cluster-specialized autoencoder should be relatively easy compared to a single autoencoder for the entire data.

Hence, we expect that a good clustering should results in a small reconstruction error.

Denote the autoencoder associated with cluster i by f i (x; ?? i ) where ?? i is the parameter-set of the network autoencoder.

We can view the reconstructed object f i (x; ?? i ) ??? R d as a data-driven centroid of the cluster i that is tuned for the input x.

The goal of the training procedure is to find a clustering of the data such that the error of the cluster-based reconstruction is minimized.

The clustering is thus computed by minimizing the following loss function: DISPLAYFORM2 is the reconstruction error of the i-th autoencoder.

In our implementation we set d( DISPLAYFORM3 In the minimization of (3) we simultaneously perform data clustering in the embedded space h(x) and learn a 'centroid' representation for each cluster in the form of an autoencoder.

Unlike most of previously proposed deep clustering methods, there is no risk of collapsing to a trivial solution where all the data points are transformed to the same vector, even though the clustering is carried out in the embedded space.

Collapsing all the data points into a single vector in the embedded space will result in directing all the points to the same autoencoder for reconstruction.

As our clustering goal is to minimize the reconstruction error, this situation is, of course, worse than using k different autoencoder for reconstruction.

Hence, there is no need for adding regularization terms to the loss function (that might influence the clustering accuracy) to prevent data collapsing.

Specifically, there is no need to add a decoder to the embedded space, where the clustering is actually performed, to prevent data collapsing.

The back-propagation equation for the parameter set of the clustering network is: DISPLAYFORM4 such that DISPLAYFORM5 is a soft assignment of x t into the i-th cluster based on the current parameters-set.

In other words, the reconstruction error of the autoencoders is used to obtain soft labels that are used for training the clustering network.

In the last few years network pre-training has been largely obsoleted for supervised tasks due to availability of large labeled training datasets.

However, for handling hard optimization problems paused by unsupervised clustering tasks, like that in (1), initialization is crucial.

To initialize the parameters of the network, we first train a single autoencoder and use the layer-wise pre-training method as in BID2 ) for training autoencoders.

After training the autoencoder, we carry out a k-means clustering on the output of the bottleneck layer to obtain initial clustering values.

The k-means assigns a label for each data point.

Note, that in the pre-training procedure a single autoencoder is trained with all the database.

We use these labels as a supervision to pre-train the clustering network (1).

The points that were assigned by the k-means algorithm to cluster i are next used to pre-train the i-th autoencoder f i (x; ?? i ).

Once all the network parameter are initialized by this pre-training procedure, the network parameters are jointly trained to minimize the autoencoding reconstruction error defined by the loss function (3).

We dub the proposed algorithm Deep Autoen-codr MIxture Clustering (DAMIC).

The architecture of the network that is trained by the DAMIC algorithm is shown in FIG0 and the clustering algorithm is summarized in TAB0 .The DAMIC algorithm can be viewed as an extension of the k-means algorithm.

Assume we replace each autoencoder in our network by a constant function f i (x t , ?? i ) ??? ?? i ??? R d and we replace the clustering network by a hard decision based on the reconstruction error, then we obtain exactly the classical k-means algorithm.

The DAMIC algorithm replaces the constant centroid with a data driven representation of the input computed by an autoencoder.

The probabilistic modeling used by the DAMIC clustering algorithm can be viewed as an instance of mixture-of-experts (MoE) model introduced by BID8 and BID9 .

The MoE model is comprised of several expert models and a gate model.

Each of the experts provides a decision and the gate is a latent variable that selects the relevant expert based on the input data.

In spite of the huge success of deep learning, there are only a few studies that have explicitly utilized and analyzed MoEs as an architecture component of a neural network BID5 ; BID15 ).

MoE has been mostly applied to supervised tasks such as classification and regression.

In our clustering algorithm the clustering network is the equivalent of the MoE gating function.

The experts here are autoencoders in which each autoencoder expertise is reconstructing a sample from the associated cluster.

Our clustering cost function (3) is following the training strategy proposed in BID8 , which prefers error function that encourages expert specialization instead of cooperation.

Goal: clustering x1, . . .

, xn ??? R d into k clusters.

Network components:??? A nonlinear representation: x ??? h(x; ??c) DISPLAYFORM6 ??? A set of autoencoders (one for each cluster): fi(xt; ??i), i = 1, . . .

, k

??? Train a single autoencoder for the entire dataset.??? Apply k-means algorithm in the embedded space.??? Use the k-means clustering to initialize the network parameters.

Clustering is obtained by minimizing the reconstruction error: DISPLAYFORM0 The final (hard) clustering is: DISPLAYFORM1 p(ct = i|xt; ??c), t = 1, . . .

, n.

In this section we evaluate the clustering results of our approach.

We carried out experiments on different datasets and compared the proposed method to state-of-the-art standard and k-means related deep clustering algorithms.

The datasets used in the experiments are standard clustering benchmark collections.

We considered both image and text datasets to demonstrate the general applicability of our approach.

Image datasets consist of MNIST (70,000 images, 28 ?? 28 pixels, 10 classes) which contains hand-written digit images.

We reshaped the images to one dimensional vectors and normalized the pixel intensity levels (between 0 and 1).

The text collections we considered are the 20 Newsgroups dataset (hereafter dubbed, 20NEWS) and the RCV1-v2 dataset (hereafter dubbed, RCV1) BID12 ).

For 20NEWS, the entire dataset comprising 18,846 documents labeled into 20 different news-groups was used.

For the RCV1, similar to BID17 ) we used a subset of the database containing 365,968 documents, each of which pertains to only one of 20 topics.

Because of the text datasets sparsity, and as proposed in BID16 ) and BID17 ), we selected the 2000 words with the highest tf-idf values to represent each document.

The clustering performance of the evaluated methods is evaluated with respect to the following three standard measures: normalized mutual information (NMI) BID3 ), clustering accuracy (ACC) BID3 ), and adjusted Rand index (ARI) BID18 ).

NMI is an information-theoretic measure based on the mutual information of the ground-truth classes and the obtained clusters, normalized using the entropy of each.

ACC measures the proportion of data points for which the obtained clusters can be correctly mapped to ground-truth classes, where the matching is based on the Hungarian algorithm BID11 ).

Finally ARI is a variant the Rand index that is adjusted for the chance grouping of elements.

Note that NMI and ACC lie in the range of 0 to 1 with one being the perfect clustering result and zero the worst.

ARI is a value between minus one to (plus) one, with 1 being the best clustering performance and ???1 the opposite.

The proposed DAMIC algorithm is compared with the following methods: DISPLAYFORM0 The classic k-means BID13 ).

This algorithm is carried out in two steps.

First, a DAE is applied.

Next, KM is applied to the embedded layer of the DAE.

This algorithm is also used as an initialization step for the proposed algorithm.

The algorithm performs joint reconstruction and k-means clustering at the same time.

The loss comprises penalties on both the reconstruction and the clustering losses BID17 ).

The algorithm performs joint embedding and clustering in the embedded space.

The loss function contains only a clustering loss term BID16 ).

The proposed method was implemented with the deep learning toolbox Tensorflow BID0 ).

All datasets were normalized between 0 and 1.

All neurons in the proposed architecture except the output layer were using rectified linear unit (ReLU) as the transfer function.

The output layer in all DAEs was the sigmoid function, while the clustering network output layer was a softmax layer.

Batch normalization BID7 ) was utilized to all layers, and the ADAM optimizer BID10 ) was used for both the pre-training as well as the training phase.

In the pre-training phase, the DAE networks were trained with the binary cross-entropy loss function.

We set the number of ephocs for the training phases to be 50 ephocs.

Yet, early stopping was used to prevent mis-convergence of the loss.

The mini-batch size is 256.It is worth noting that for simplicity and to show the robustness of the proposed method, the architectures of the proposed DAMIC in all the following experiments are with a similar shape, i.e. for each of the DAEs we use 5-layers DNN with the following input size: 1024, 256, k, 256, 1024, ReLUs, respectively, and for the clustering network we used 512, 512, k, ReLUs, respectively, where k is the number of clusters.

We emphasize that there was no need for hyperparameter tuning for the experiments on the different datasets.3.5 RESULTS

As mentioned before, the MNIST database has 70000 hand written gray-scale digit images.

Each image size is 28 ?? 28 pixels.

Note that we work on the raw data of the dataset (without preprocessing).

For simplicity, the architecture of each one of the DAE is identical.

Specifically, for the MNIST dataset we used 5-layers network with number of neurons 1024, 256, 10, 256, 1024, respectively.

The output layer of each DAE was set to be the sigmoid function.

For the clustering network we used simpler network with 3-layers 512, 512, 10 neurons, respectively.

The output transfer layer of the clustering network is the softmax function.

TAB1 presents the results of the NMI, the ACC and the ARI of the proposed DAMIC method and several standard baselines.

It is clear that the DAMIC outperforms the other methods in the NMI and ARI measures.

The DEC method get the highest ACC result.

To demonstrate the expertise of each one of the DAE we conducted the following experiment.

After the clustering algorithm converged on the MNIST dataset, we synthetically created a new image in which all the pixels are set to be '1' (Fig. 2a) .

The image reconstruction of all the 10 DAEs is shown in Fig. 2 .

It is evident that each DAE assumes a different pattern of the input.

Specifically, each DAE is responsible for a different digit.

The clustering task is unsupervised and we sorted the autoencoders in Fig. 2a ) by their corresponding digits from '0' to '9' just for visualization.

Best reconstruction wins To further understand the behavior of the gate we carried out a different test.

An image of the digit '4' was fed to the network (Fig. 3a) .

The outputs of the different DAEs are depicted in Fig. 3 .

Since each DAE specializes in a different digit, it is expected that the respective DAE will obtain the lowest reconstruction error.

This is also reflected by decision of the clustering network p(c = 4|x; ?? c ) = 0.99.

Note, that the other DAEs reshaped the reconstruction to be close to their digit specialty.

The 20Newsgroup corpus consists of 18,846 documents from 20 news groups.

As in BID17 we also use the tf-idf representation of the documents and pick the 2,000 most frequently used words as the features.

The architecture used in each one of the DAEs for this experiment also consists of 5-layers with number of neurons 1024, 256, 20, 256, 1024 , respectively.

The clustering network here consists of 512, 512, 20 neurons.

TAB2 shows the results of the NMI, ARI and ACC measures.

It is evident that the proposed clustering method outperforms the compared baselines.

DISPLAYFORM0 Figure 2: The outputs of the different DAEs with a vector of all-ones input.

DISPLAYFORM1 Figure 3: An example of the outputs of the different DAEs with the digit '4' as the input.

The dataset was used in this experiment is a subset of the RCV-1-v2 with 365, 968 documents each containing one of 20 topics.

As in BID17 the 2,000 most frequently used words (in the tf-idf form) are used as the features of each documents.

In contrast with the previous databases, in the RCV1 dataset, the size of each class in not equal.

Therefore, KM-based approaches might be not sufficient in this case.

In our architecture we used 1024, 256, 20, 256, 1024 ReLU neurons in all DAEs, respectively, and in the clustering network we used 512, 512, 20 ReLU neurons.

In TAB3 we present the 3 objective measurements for the RCV1 experiment.

The proposed method outperforms the competing methods in NMI and ARI measures but obtains lower result in the ACC measure.

To further illustrate the capabilities of the DAMIC algorithm we generated a synthetic data as in BID17 .

The 2D latent domain contains 4000 samples from four Gaussian distributed clusters as shown in FIG1 .

The observed signal is x t = (??(W ?? v t )) 2 t = 1, ?? ?? ?? , n, where ?? is the sigmoid function, W ??? R 100??2 and v t is the t-th point in the latent domain.

We first applied the DAE+KM algorithm for initialization.

The architecture of the DAE consists of 4-layers-encoder with 100, 50, 10, 2 neurons respectively.

The decoder is a mirrored version of the forward network.

FIG1 depicts the 2D embedded space of the DAE.

It is not sufficiently separated in the embedded space.

The proposed DAMIC algorithm was then applied.

The architecture of each autoencoder consists of 5-layers 1024, 256, 4, 256, 1024 neurons as in the previous experiments.

The clustering network is also similar with 512, 512, 2 neurons, respectively.

FIG1 depicts the 2D embedded space of the clustering network h(x t ).

It is easy to see that the embedded space is much more separable.

TAB4 summarizes the results of the k-means, the DAE+KM and the DAMIC algorithms on the synthetic generated data.

It is easy to verify that the DAMIC algorithm outperforms the two competing algorithms in both NMI and ARI measures.

In this study we presented a clustering technique which leverages the strength of deep neural network.

Our technique has two major properties: first, unlike most previous methods, the clusters are represented by an autoencoder network instead of a single centroid vector in the embedded space.

This enables a much richer representation of each cluster.

Second, The algorithm does not cause a data collapsing problem.

Hence, there is no need for regularization terms that have to be tuned for each dataset separately.

Experiments on a variety of real datasets showed the improved performance of the proposed algorithm.

<|TLDR|>

@highlight

We propose a deep clustering method where instead of a centroid each cluster is represented by an autoencoder

@highlight

Presents deep clustering based on a mixture of autoencoders, where data points are allocated to a cluster based the representation error if the autoencoder network were used to represent it.

@highlight

A deep clustering approach that uses an autoencoder framework to learn a low-dimensional embedding of the data simultaneously while clustering data using a deep neural network.

@highlight

A deep clustering method which represents each cluster with different auto-encoders, works in an end-to-end manner, and also can be used to cluster new incoming data without redoing the whole clustering procedure.