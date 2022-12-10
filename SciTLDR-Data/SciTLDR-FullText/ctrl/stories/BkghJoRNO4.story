Most approaches in generalized zero-shot learning rely on cross-modal mapping between an image feature space and a class embedding space or on generating artificial image features.

However, learning a shared cross-modal embedding by aligning the latent spaces of modality-specific autoencoders is shown to be promising in (generalized) zero-shot learning.

While following the same direction, we also take artificial feature generation one step further and propose a model where a shared latent space of image features and class embeddings is learned by aligned variational autoencoders, for the purpose of generating latent features to train a softmax classifier.

We evaluate our learned latent features on conventional benchmark datasets and establish a new state of the art on generalized zero-shot as well as on few-shot learning.

Moreover, our results on ImageNet with various zero-shot splits show that our latent features generalize well in large-scale settings.

Generalized zero-shot learning (GZSL) is a classification task where no labeled training examples are available from some of the classes.

Many approaches learn a mapping between images and their class embeddings BID11 BID19 BID0 .

For instance, ALE maps CNN features of images to a perclass attribute space.

An orthogonal approach to GZSL is to augment data by generating artificial image features, such as BID21 who proposed to generate image features via a conditional WGAN.

As a third approach, BID16 proposed to learn a latent space embedding by transforming both modalities to the latent spaces of autoencoders and match the corresponding distributions by minimizing the Maximum Mean Discrepancy (MMD).

Learning such cross-modal embeddings can be beneficial for potential downstream tasks that require multimodal fusion.

In this regard, BID13 recently used a cross-modal autoencoder to extend visual question answering to previously unseen objects.

Although recent cross-modal autoencoder architectures represent class prototypes in a latent space BID10 BID16 , better generalization can be achieved if the shared representation space is more amenable to interpolation between different classes.

Variational Autoencoders (VAEs) are known for their capability in accurate interpolation between representations in their latent space, i.e. as demonstrated for sentence interpolation BID2 and image interpolation BID6 .

Hence, in this work, we train VAEs to encode and decode features from different modalities, and align their latent spaces by matching the parametrized latent distributions and by enforcing a cross-modal reconstruction criterion.

Since we learn representations that are oblivious to their origin, a zero-shot visual classifier can be trained using latent space features from semantic data.

Our contributions in this work are as follows.

FORMULA1 Generalized Zero-shot Learning Let S = {(x, y, c(y))| x ∈ X, y ∈ Y S , c(y) ∈ C} be a set of training examples, consisting of image-features x, e.g. extracted by a CNN, class labels y available during training and class-embeddings c(y).

Typical class-embeddings are vectors of continuous attributes or Word2Vec features .

In addition, an set U = {(u, c(u))| u ∈ Y u , c(u) ∈ C} is used, where u denote unseen classes from a set Y u , which is disjoint from Y S .

Here, C(U ) = {c(u 1 ), ..., c(u L )} is the set of class-embeddings of unseen classes.

In the legacy challenge of ZSL, the task is to learn a classifier f ZSL : X → Y U .

However, in this work, we focus on the more realistic and challenging setup of generalized zero-shot learning (GZSL) where the aim is to learn a classifier DISPLAYFORM0 The Objective Function CADA-VAE is trained with pairs of image features and attribute vectors of seen classes.

The data of each pair has to belong to the same class.

In this process, an image feature encoder and an attribute encoder learn to transform the training data into the shared latent space.

The encoders belong to two VAEs with a common latent space.

Once the VAEs are trained, a softmax classifier is trained on both seen image data and unseen attributes, after they are transformed into the latent representation.

As the VAE encoding is non-deterministic, many latent features are sampled for each datapoint.

Since we only have one attribute vector per class, we oversample latent-space encoded features of unseen classes.

To test the classifier, the visual test data is first transformed into the latent space, using only the predicted means µ of the latent representation.

The Objective function for training the VAEs is derived as follows.

For every modality i (image features, attributes), a VAE is trained.

The basic VAE loss for a feature x of modality i ∈ 1, 2, ..M is: DISPLAYFORM1 where D KL represents the Kullback-Leibler Divergence, β is a weight, q(z|x (i) ) = N (µ, Σ) is the VAE encoder consisting of a multilayer perceptron, and p(z) is a Gaussian prior.

Additionally, each encoded datapoint is decoded into every available modality, e.g. encoded image features are decoded into attributes and vice versa.

Consequently, we minimize the L1 cross-reconstruction loss: DISPLAYFORM2 where γ is a weight.

The L1 loss empirically proved to provide slighthly better results than L2.

Furthermore, the 2-Wasserstein W distance between the multivariate Gaussian latent distribution of image features and attributes is minimized: DISPLAYFORM3 The VAE is trained using the final objective L = L basic +L CA +L DA .

We refer to the Cross-Aligned and Distribution-Aligned VAE as CADA-VAE.

In addition, we test the variant L = L basic + L CA , termed CA-VAE, and the variant L = L basic + L DA , referred to as DA-VAE.

A latent size of 64 is used for all experiments, except 128 for ImageNet.

We evaluate our framework on zero-shot learning benchmark datasets CUB-200-2011 BID18 , SUN attribute BID12 BID7 BID20 for the GZSL setting.

All image features used for training the VAEs are extracted from the 2048-dimensional final pooling layer of a ResNet-101.

To avoid violating the zero-shot assumption, i.e. test classes need to be disjoint from the classes that ResNet-101 was trained with, we use the proposed training splits in BID20 .

As class embeddings, attribute vectors were utilized if available.

For ImageNet we used Word2Vec embeddings provided by BID3 .

All hyperparameters were chosen on a validation set provided by BID20 .

We report the harmonic mean (H) between seen (S) and unseen (U) average per-class accuracy, i.e. the Top-1 accuracy is averaged on a per-class basis.

, AwA1 and 2 (Generalized Zero-Shot Learning We compare our model with 11 state-of-the-art models.

Among those, CVAE , SE (Verma et al., 2017), and f-CLSWGAN BID21 learn to generate artificial visual data and thereby treat the zero-shot problem as a data-augmentation problem.

On the other hand, the classic ZSL methods DeViSE , SJE BID0 , ALE , EZSL BID14 and LATEM BID19 use a linear compatibility function or other similarity metrics to compare embedded visual and semantic features; CMT BID15 and LATEM BID19 utilize multiple neural networks to learn a non-linear embedding; and SYNC BID3 learns by aligning a class embedding space and a weighted bipartite graph.

ReViSE BID16 proposes a shared latent manifold learning using an autoencoder between the image features and class attributes.

The results in TAB3 show that our CADA-VAE outperforms all other methods on all datasets.

Moreover, our model achieves significant improvements over feature generating models most notably on CUB.

Compared to the classic ZSL methods, our method leads to at least 100% improvement in harmonic mean accuracies.

In the legacy challenge of ZSL setting, which is hardly realistic, our CADA-VAE provides competitive performance, i.e. 60.4 on CUB, 61.8 on SUN, 62.3 on AWA1, 64.0 on AWA2.

However, in this work, we focus on the more practical and challenging GZSL setting.

We believe the obtained increase in performance by our model can be explained as follows.

CADA-VAE learns a shared representation in a weakly supervised fashion, through a crossreconstruction objective.

Since the latent features have to be decoded into every involved modality, and since every modality encodes complementary information, the model is encouraged to learn an encoding that retains the information contained in all used modalities.

In doing so, our method is less biased towards learning the distribution of the seen class image features, which is known as the projection domain shift problem BID5 .

As we generate a certain number of latent features per class using non-deterministic encoders, our method is also akin to data-generating approaches.

However, the learned representations lie in a lower dimensional space, i.e. only 64, and therefore, are less prone to bias towards the training set of image features.

In effect, our training is more stable than the adversarial training schemes used for data generation BID21 .

BID20 several evaluation splits were proposed with increasing granularity and size both in terms of the number of classes and the number of images.

Note that since all the images of 1K classes are used to train ResNet-101, measuring seen class accuracies would be biased.

However, we can still evaluate the accuracy of unseen class images in the GZSL search space that contains both seen and unseen classes.

Hence, at test time the 1K seen classes BID15 49.8 7.2 12.6 21.8 8.1 11.8 87.6 0.9 1.8 90.0 0.5 1.0 SJE BID0 59 BID14 63.8 12.6 21.0 27.9 11.0 15.8 75.6 6.6 12.1 77.8 5.9 11.0 SYNC BID3 70.9 11.5 19.8 43.3 7.9 13.4 87.3 8.9 16.2 90.5 10.0 18.0 DeViSE 53.0 23.8 32.8 27.4 16.9 20.9 68.7 13.4 22.4 74.7 17.1 27.8 f- CLSWGAN Xian et al. (2018b) 57 act as distractors.

For ImageNet, as attributes are not available, we use Word2Vec features as class embeddings provided by BID3 .

We compare our model with f-CLSWGAN BID21 , i.e. an image feature generating framework which currently achieves the state of the art on ImageNet.

We use the same evaluation protocol on all the splits.

Among the splits, 2H and 3H are the classes 2 or 3 hops away from the 1K seen training classes of ImageNet according to the ImageNet hierarchy.

M 500, M 1K and M 5K are the 500, 1000 and 5000 most populated classes, while L500, L1K and L5K are the 500, 1000 and 5000 least populated classes that come from the rest of the 21K classes.

Finally, 'All' denotes the remaining 20K classes of ImageNet.

As shown in FIG1 , our model significantly improves the state of the art in all the available splits.

Note that the test time search space in the 'All' split is 22K dimensional.

Hence even a small improvement in accuracy on this split is considered to be compelling.

The achieved substantial increase in performance by CADA-VAE shows that our 128-dim latent feature space constitutes a robust generalizable representation, surpassing the current state-of-the-art image feature generating framework f-CLSWGAN.

In this work, we propose CADA-VAE, a cross-modal embedding framework for generalized zeroshot learning in which the modality-specific latent distributions are aligned by minimizing their Wasserstein distance and by using cross-reconstruction.

This procedure leaves us with encoders that can encode features from different modalities into one cross-modal embedding space, in which a linear softmax classifier can be trained.

We present different variants of cross-aligned and distribution aligned VAEs and establish new state-of-the-art results in generalized zero-shot learning for four medium-scale benchmark datasets as well as the large-scale ImageNet.

We further show that a cross-modal embedding model for generalized zero-shot learning achieves better performance than data-generating methods, establishing the new state of the art.

<|TLDR|>

@highlight

We use VAEs to learn a shared latent space embedding between image features and attributes and thereby achieve state-of-the-art results in generalized zero-shot learning.