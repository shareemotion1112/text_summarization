Using class labels to represent class similarity is a typical approach to training deep hashing systems for retrieval; samples from the same or different classes take binary 1 or 0 similarity values.

This similarity does not model the full rich knowledge of semantic relations that may be present between data points.

In this work we build upon the idea of using semantic hierarchies to form distance metrics between all available sample labels; for example cat to dog has a smaller distance than cat to guitar.

We combine this type of semantic distance into a loss function to promote similar distances between the deep neural network embeddings.

We also introduce an empirical Kullback-Leibler divergence loss term to promote binarization and uniformity of the embeddings.

We test the resulting SHREWD method and demonstrate improvements in hierarchical retrieval scores using compact, binary hash codes instead of real valued ones, and show that in a weakly supervised hashing setting we are able to learn competitively without explicitly relying on class labels, but instead on similarities between labels.

Content-Based Image Retrieval (CBIR) on very large datasets typically relies on hashing for efficient approximate nearest neighbor search; see e.g. BID12 for a review.

Early methods such as (LSH) BID5 were data-independent, but Data-dependent methods (either supervised or unsupervised) have shown better performance.

Recently, Deep hashing methods using CNNs have had much success over traditional methods, see e.g. Hashnet BID1 , DADH .

Most supervised hashing techniques rely on a pairwise binary similarity matrix S = {s ij }, whereby s ij = 1 for images i and j taken from the same class, and 0 otherwise.

A richer set of affinity is possible using semantic relations, for example in the form of class hierarchies.

BID13 consider the semantic hierarchy for non-deep hashing, minimizing inner product distance of hash codes from the distance in the semantic hierarchy.

In the SHDH method , the pairwise similarity matrix is defined from such a hierarchy according to a weighted sum of weighted Hamming distances.

In Unsupervised Semantic Deep Hashing (USDH, Jin (2018)), semantic relations are obtained by looking at embeddings on a pre-trained VGG model on Imagenet.

The goal of the semantic loss here is simply to minimize the distance between binarized hash codes and their pre-trained embeddings, i.e. neighbors in hashing space are neighbors in pre-trained feature space.

This is somewhat similar to our notion of semantic similarity except for using a pre-trained embedding instead of a pre-labeled semantic hierarchy of relations.

BID14 consider class-wise Deep hashing, in which a clustering-like operation is used to form a loss between samples both from the same class and different levels from the hierarchy.

Recently BID0 explored image retrieval using semantic hierarchies to design an embedding space, in a two step process.

Firstly they directly find embedding vectors of the class labels on a unit hypersphere, using a linear algebra based approach, such that the distances of these embeddings are similar to the supplied hierarchical similarity.

In the second stage, they train a standard CNN encoder model to regress images towards these embedding vectors.

They do not consider hashing in their work.

We also make use of hierarchical relational distances in a similar way to constrain our embeddings.

However compared to our work, BID0 consider continuous representations and require the embedding dimension to equal the number of classes, whereas we learn compact quantized hash codes of arbitrary length, which are more practical for real world retrieval performance.

Moreover, we do not directly find fixed target embeddings for the classes, but instead require that the neural network embeddings will be learned in conjunction with the network weights, to best match the similarities derived from the labels.

And unlike BID14 , in our weakly supervised SHREWD method, we do not require explicit class membership, only relative semantic distances to be supplied.

Let (x, y) denote a training example pair consisting of an image and some (possibly weakly) supervised target y, which can be a label, tags, captions etc.

The embeddings are defined as??? = f ?? (x) for a deep neural network f parameterized by weights ??.

Instead of learning to predict the target y, we assume that there exists an estimate of similarity between targets, d(y, y ).

The task of the network is then to learn this similarity by attempting to match ??? ?????? with d(y, y ) under some predefined norm in the embedding space.

While in this work we use class hierarchies to implicitly inform our loss function via the similarity metric d, in general our formulation is weakly supervised in the sense that these labels themselves are not directly required as targets.

We could equally well replace this target metric space with any other metric based on for instance web-mined noisy tag distances in a word embedding space such as GloVe or word2vec, as in BID4 , or ranked image similarities according to recorded user preferences.

In addition to learning similarities between images, it is important to try to fully utilize the available hashing space in order to facilitate efficient retrieval by using the Hamming distance to rank most similar images to a given query image.

Consider for example a perfect ImageNet classifier.

We could trivially map all 1000 class predictions to a 10-bit hash code, which would yield a perfect mAP score.

The retrieval performance of such a "mAP-miner" model would however be poor, because the model is unable to rank examples both within a given class and between different classes BID3 .

We therefore introduce an empirical Kullback-Leibler (KL) divergence term between the embedding distribution and a (near-)binary target distribution, which we add as an additional loss term.

The KL loss serves an additional purpose in driving the embeddings close to binary values in order to reduce the information loss due to binarizing the embeddings.

We next describe the loss function, L(??), that we minimize in order to train our CNN model.

We break down our approach into the following 3 parts: DISPLAYFORM0 L cls represents a traditional categorical cross-entropy loss on top of a linear layer with softmax placed on the non-binarized latent codes.

The meaning and use of each of the other two terms are described in more detail below.

Similar to BID0 we consider variants with and without the L cls , giving variants of the algorithm we term SHREWD (weakly supervised, no explicit class labels needed) and SHRED (fully supervised).

In order to weakly supervise using a semantic similarity metric, we seek to find affinity between the normalized distances in the learned embedding space and normalized distances in the semantic space.

Therefore we define DISPLAYFORM0 where B is a minibatch size, . . .

M denotes Manhattan distance (because in the end we will measure similarity in the binary space by Hamming distance), d (y b , y b ) is the given ground truth similarity and w bb is an additional weight, which is used to give more weight to similar example pairs (e.g. cat-dog) than distant ones (e.g. cat-moon).

?? z and ?? y are normalizing scale factors estimated Note that while L cls performs best on supervised classification, L sim allows for better retrieval performance, however this is degraded unless L KL is also included to regularize towards binary embeddings.

For measuring classification accuracy on methods that don't include L cls , we measure using a linear classifier with the same structure as in L cls trained on the output of the first network.from the current batch.

We use a slowly decaying form for the weight, DISPLAYFORM1 with parameter values ?? = 0.1 and ?? = 2.

Our empirical loss for minimizing the KL divergence KL(p||q) .

= dzp(z) log(p(z)/q(z)) between the sample embedding distribution p(z) and a target distribution q(z) is based on the Kozachenko-Leonenko estimator of entropy BID7 , and can be defined as DISPLAYFORM0 where ??(??? b ; z) denotes the distance of??? b to a nearest vector z b , where z is a sample (of e.g. size B) of vectors from a target distribution.

We employ the beta distribution with parameters ?? = ?? = 0.1 as this target distribution, which is thus moderately concentrated to binary values in the embedding space.

The result is that our embedding vectors will be regularized towards uniform binary values, whilst still enabling continuous backpropagation though the network and giving some flexibility in allowing the distance matching loss to perform its job.

When quantized, the resulting embeddings are likely to be similar to their continuous values, meaning that the binary codes will have distances more similar to their corresponding semantic distances.

Metrics As discussed in section 2, the mAP score can be a misleading metric for retrieval performance when using class information only.

Similarly to other works such as BID2 BID0 , we focus on measuring the retrieval performance taking semantic hierarchical relations into account by the mean Average Hierarchical Precision (mAHP).

However more in line with other hashing works, we use the hamming distance of the binary codes for ranking the retrieved results.

We first test on CIFAR-100 BID8 using the same semantic hierarchy and Resnet-110w architecture as in BID0 , where only the top fully connected layer is replaced to return embeddings at the size of the desired hash length.

See TAB2 for comparisons with previous methods, an ablation study, and effects of hash code length.

ILSVRC 2012 We also evaluate on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 dataset.

For similarity labels, we use the same tree-structured WordNet hierarchy as in BID0 .

We use a standard Resnet-50 architecture with a fully connected hashing layer as before.

Retrieval results are summarized in TAB3 .

We compare the resulting Hierarchical Precision scores with and without L KL , for binarized and continuous values in FIG1 .

We see that our results improve on the previously reported hierarchical retrieval results whilst using quantized embeddings, enabling efficient retrieval.

We see a substantial drop in the precision after binarization when not using the KL loss.

Also binarization does not cause as severe a drop in precision when using the KL loss.

We approached Deep Hashing for retrieval, introducing novel combined loss functions that balance code binarization with equivalent distance matching from hierarchical semantic relations.

We have demonstrated new state of the art results for semantic hierarchy based image retrieval (mAHP scores) on CIFAR and ImageNet with both our fully supervised (SHRED) and weakly-supervised (SHREWD) methods.

@highlight

We propose a new method for training deep hashing for image retrieval using only a relational distance metric between samples