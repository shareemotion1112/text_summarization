Identifying analogies across domains without supervision is a key task for artificial intelligence.

Recent advances in cross domain image mapping have concentrated on translating images across domains.

Although the progress made is impressive, the visual fidelity many times does not suffice for identifying the matching sample from the other domain.

In this paper, we tackle this very task of finding exact analogies between datasets i.e. for every image from domain A find an analogous image in domain B. We present a matching-by-synthesis approach: AN-GAN, and show that it outperforms current techniques.

We further show that the cross-domain mapping task can be broken into two parts: domain alignment and learning the mapping function.

The tasks can be iteratively solved, and as the alignment is improved, the unsupervised translation function reaches quality comparable to full supervision.

Humans are remarkable in their ability to enter an unseen domain and make analogies to the previously seen domain without prior supervision ("This dinosaur looks just like my dog Fluffy").

This ability is important for using previous knowledge in order to obtain strong priors on the new situation, which makes identifying analogies between multiple domains an important problem for Artificial Intelligence.

Much of the recent success of AI has been in supervised problems, i.e., when explicit correspondences between the input and output were specified on a training set.

Analogy identification is different in that no explicit example analogies are given in advance, as the new domain is unseen.

Recently several approaches were proposed for unsupervised mapping between domains.

The approaches take as input sets of images from two different domains A and B without explicit correspondences between the images in each set, e.g. Domain A: a set of aerial photos and Domain B: a set of Google-Maps images.

The methods learn a mapping function T AB that takes an image in one domain and maps it to its likely appearance in the other domain, e.g. map an aerial photo to a Google-Maps image.

This is achieved by utilizing two constraints: (i) Distributional constraint: the distributions of mapped A domain images (T AB (x)) and images of the target domain B must be indistinguishable, and (ii) Cycle constraint: an image mapped to the other domain and back must be unchanged, i.e., T BA (T AB (x)) = x.

In this paper the task of analogy identification refers to finding pairs of examples in the two domains that are related by a fixed non-linear transformation.

Although the two above constraints have been found effective for training a mapping function that is able to translate between the domains, the translated images are often not of high enough visual fidelity to be able to perform exact matching.

We hypothesize that it is caused due to not having exemplar-based constraints but rather constraints on the distributions and the inversion property.

In this work we tackle the problem of analogy identification.

We find that although current methods are not designed for this task, it is possible to add exemplar-based constraints in order to recover high performance in visual analogy identification.

We show that our method is effective also when only some of the sample images in A and B have exact analogies whereas the rest do not have exact analogies in the sample sets.

We also show that it is able to find correspondences between sets when no exact correspondences are available at all.

In the latter case, since the method retrieves rather than maps examples, it naturally yields far better visual quality than the mapping function.

Using the domain alignment described above, it is now possible to perform a two step approach for training a domain mapping function, which is more accurate than the results provided by previous unsupervised mapping approaches:1.

Find the analogies between the A and B domain, using our method.2.

Once the domains are aligned, fit a translation function T AB between the domains y mi = T AB (x i ) using a fully supervised method.

For the supervised network, larger architectures and non-adversarial loss functions can be used.

This paper aims to identify analogies between datasets without supervision.

Analogy identification as formulated in this paper is highly related to image matching methods.

As we perform matching by synthesis across domains, our method is related to unsupervised style-transfer and image-to-image mapping.

In this section we give a brief overview of the most closely related works.

Image Matching Image matching is a long-standing computer vision task.

Many approaches have been proposed for image matching, most notably pixel-and feature-point based matching (e.g. SIFT BID11 ).

Recently supervised deep neural networks have been used for matching between datasets BID18 , and generic visual features for matching when no supervision is available (e.g. BID5 ).

As our scenario is unsupervised, generic visual feature matching is of particular relevance.

We show in our experiments however that as the domains are very different, standard visual features (multi-layer VGG-16 BID15 ) are not able to achieve good analogies between the domains.

Generative Adversarial Networks GAN (Goodfellow et al., 2014) technology presents a major breakthrough in image synthesis (and other domains).

The success of previous attempts to generate random images in a class of a given set of images, was limited to very specific domains such as texture synthesis.

Therefore, it is not surprising that most of the image to image translation work reported below employ GANs in order to produce realistically looking images.

GAN (Goodfellow et al., 2014) methods train a generator network G that synthesizes samples from a target distribution, given noise vectors, by jointly training a second network D. The specific generative architecture we and others employ is based on the architecture of BID12 .

In image mapping, the created image is based on an input image and not on random noise BID9 BID21 BID10 BID16 .Unsupervised Mapping Unsupervised mapping does not employ supervision apart from sets of sample images from the two domains.

This was done very recently BID16 BID9 BID21 for image to image translation and slightly earlier for translating between natural languages BID19 .

The above mapping methods however are focused on generating a mapped version of the sample in the other domain rather than retrieving the best matching sample in the new domain.

Supervised Mapping When provided with matching pairs of (input image, output image) the mapping can be trained directly.

An example of such method that also uses GANs is , where the discriminator D receives a pair of images where one image is the source image and the other is either the matching target image ("real" pair) or a generated image ("fake" pair); The link between the source and the target image is further strengthened by employing the U-net architecture of BID14 .

We do not use supervision in this work, however by the successful completion of our algorithm, correspondences are generated between the domains, and supervised mapping methods can be used on the inferred matches.

Recently, BID2 demonstrated improved mapping results, in the supervised settings, when employing the perceptual loss and without the use of GANs.

In this section we detail our method for analogy identification.

We are given two sets of images in domains A and B respectively.

The set of images in domain A are denoted x i where i ??? I and the set image in domain B are denoted y j where j ??? J. Let m i denote the index of the B domain image y mi that is analogous to x i .

Our goal is to find the matching indexes m i for i ??? I in order to be able to match every A domain image x i with a B domain image y mi , if such a match exists.

We present an iterative approach for finding matches between two domains.

Our approach maps images from the source domain to the target domain, and searches for matches in the target domain.

A GAN-based distribution approach has recently emerged for mapping images across domains.

Let x be an image in domain A and y be an image in domain B. A mapping function T AB is trained to map x to T AB (x) so that it appears as if it came from domain B. More generally, the distribution of T AB (x) is optimized to appear identical to that of y. The distributional alignment is enforced by training a discriminator D to discriminate between samples from p(T AB (x)) and samples from p(y), where we use p(x) to denote the distribution of x and p(T AB (x)) to denote the distribution of T AB (x) when x ??? p(x).

At the same time T AB is optimized so that the discriminator will have a difficult task of discriminating between the distributions.

The loss function for training T and D are therefore: DISPLAYFORM0 is a binary cross-entropy loss.

The networks L D and L T are trained iteratively (as they act in opposite directions).In many datasets, the distribution-constraint alone was found to be insufficient.

Additional constraints have been effectively added such as circularity (cycle) BID9 and distance invariance BID0 .

The popular cycle approach trains one-sided GANs in both the A ??? B and B ??? A directions, and then ensures that an A image domain translated to B (T AB (x)) and back to A (T BA (T BA (x))) recovers the original x.

Let L 1 denote the L 1 loss.

The complete two-sided cycle loss function is given by: DISPLAYFORM1 DISPLAYFORM2 The above two-sided approach yields mapping function from A to B and back.

This method provides matching between every sample and a synthetic image in the target domain (which generally does not correspond to an actual target domain sample), it therefore does not provide exact correspondences between the A and B domain images.

In the previous section, we described a distributional approach for mapping A domain image x to an image T AB (x) that appears to come from the B domain.

In this section we provide a method for providing exact matches between domains.

Let us assume that for every A domain image x i there exists an analogous B domain image y mi .

Our task is find the set of indices m i .

Once the exact matching is recovered, we can also train a fully supervised mapping function T AB , and thus obtain a mapping function of the quality provided by supervised method.

Let ?? i,j be the proposed match matrix between B domain image y j and A domain image x i , i.e., every x i matches a mixture of all samples in B, using weights ?? i,: , and similarly for y j for a weighing using ?? :,j of the training samples from A. Ideally, we should like a binary matrix with ?? i,j = 1 for the proposed match and 0 for the rest.

This task is formally written as: DISPLAYFORM0 where L p is a "perceptual loss", which is based on some norm, a predefined image representation, a Laplacian pyramid, or otherwise.

See Sec. 3.4.The optimization is continuous over T AB and binary programming over ?? i,j .

Since this is computationally hard, we replace the binary constraint on ?? by the following relaxed version: DISPLAYFORM1 In order to enforce sparsity, we add an entropy constraint encouraging sparse solutions.

DISPLAYFORM2 The final optimization objective becomes: DISPLAYFORM3 The positivity ?? ??? 0 and i ?? i,j = 1 constraints are enforced by using an auxiliary variable ?? and passing it through a Sof tmax function.

DISPLAYFORM4 The relaxed formulation can be optimized using SGD.

By increasing the significance of the entropy term (increasing k entropy ), the solutions can converge to the original correspondence problem and exact correspondences are recovered at the limit.

Since ?? is multiplied with all mapped examples T AB (x), it might appear that mapping must be performed on all x samples at every batch update.

We have however found that iteratively updating T AB for N epochs, and then updating ?? for N epochs (N = 10) achieves excellent results.

Denote the ?? (and ??) updates-?? iterations and the updates of T AB -T iterations.

The above training scheme requires the full mapping to be performed only once at the beginning of the ?? iteration (so once in 2N epochs).

Although the examplar-based method in Sec. 3.2 is in principle able to achieve good matching, the optimization problem is quite hard.

We have found that a good initialization of T AB is essential for obtaining good performance.

We therefore present AN-GAN -a cross domain matching method that uses both exemplar and distribution based constraints.

The AN-GAN loss function consists of three separate constraint types:1.

Distributional loss L T dist : The distributions of T AB (x) matches y and T BA (y) matches x (Eq. 3).

2.

Cycle loss L T cycle : An image when mapped to the other domain and back should be unchanged (Eq. 4).

3.

Exemplar loss L T exemplar : Each image should have a corresponding image in the other domain to which it is mapped (Eq. 11).The AN-GAN optimization problem is given by: min DISPLAYFORM0 The optimization also adversarially trains the discriminators D A and D B as in equation Eq. 6.

Initially ?? are all set to 0 giving all matches equal likelihood.

We use an initial burn-in period of 200 epochs, during which ?? = 0 to ensure that T AB and T BA align the distribution before aligning individual images.

We then optimize the examplar-loss for one ??-iteration of 22 epochs, one T -iteration of 10 epochs and another ??-iteration of 10 epochs (joint training of all losses did not yield improvements).

The initial learning rate for the exemplar loss is 1e ??? 3 and it is decayed after 20 epochs by a factor of 2.

We use the same architecture and hyper-parameters as CycleGAN unless noted otherwise.

In all experiments the ?? parameters are shared between the two mapping directions, to let the two directions inform each other as to likelihood of matches.

All hyper-parameters were fixed across all experiments.

In the previous sections we assumed a "good" loss function for determining similarity between actual and synthesized examples.

In our experiments we found that Euclidean or L 1 loss functions were typically not perceptual enough to provide good supervision.

Using the Laplacian pyramid loss as in GLO BID1 does provide some improvement.

The best performance was however achieved by using a perceptual loss function.

This was also found in several prior works BID4 , BID8 , BID2 .For a pair of images I 1 and I 2 , our loss function first extracts VGG features for each image, with the number of feature maps used depending on the image resolution.

We use the features extracted by the the second convolutional layer in each block, 4 layers in total for 64X64 resolution images and five layers for 256X256 resolution images.

We additionally also use the L 1 loss on the pixels to ensure that the colors are taken into account.

Let us define the feature maps for images I 1 and I 2 as ?? m 1 and ?? m 2 (m is an index running over the feature maps).

Our perceptual loss function is: DISPLAYFORM0 Where N P is the number of pixels and N m is the number of features in layer m. We argue that using this loss, our method is still considered to be unsupervised matching, since the features are available off-the-shelf and are not tailored to our specific domains.

Similar features have been extracted using completely unsupervised methods (see e.g. BID3

To evaluate our approach we conducted matching experiments on multiple public datasets.

We have evaluated several scenarios: (i) Exact matches: Datasets on which all A and B domain images have We compare our method against a set of other methods exploring the state of existing solutions to cross-domain matching:U nmapped ??? P ixel: Finding the nearest neighbor of the source image in the target domain using L 1 loss on the raw pixels.

U nmapped ??? V GG: Finding the nearest neighbor of the source image in the target domain using VGG feature loss (as described in Sec. 3.4.

Note that this method is computationally quite heavy due to the size of each feature.

We therefore randomly subsampled every feature map to 32000 values, we believe this is a good estimate of the performance of the method.

CycleGAN ??? P ixel: Train Eqs. 5, 6 using the authors' CycleGAN code.

Then use L 1 to compute the nearest neighbor in the target set.

CycleGAN ??? V GG: Train Eqs. 5, 6 using the authors' CycleGAN code.

Then use VGG loss to compute the nearest neighbor in the target set.

The VGG features were subsampled as before due to the heavy computational cost.?? iterations only: Train AN ??? GAN as described in Sec. 3.3 but with ?? iterations only, without iterating over T XY .AN ??? GAN : Train AN ??? GAN as described in Sec. 3.3 with both ?? and T XY iterations.

We evaluate our method on 4 public exact match datasets:Facades: 400 images of building facades aligned with segmentation maps of the buildings (Radim Tyle??ek, 2013).Maps: The Maps dataset was scraped from Google Maps by .

It consists of aligned Maps and corresponding satellite images.

We use the 1096 images in the training set.

The original dataset contains around 50K images of shoes from the Zappos50K dataset BID23 , (Yu & Grauman) .

The edge images were automatically detected by using HED ( BID20 ).

The original dataset contains around 137k images of Amazon handbags ( BID24 ).

The edge images were automatically detected using HED by .For both E2S and E2H the datasets were randomly down-sampled to 2k images each to accommodate the memory complexity of our method.

This shows that our method works also for moderately sized dataset.

In this set of experiments, we compared our method with the five methods described above on the task of exact correspondence identification.

For each evaluation, both A and B images are shuffled prior to training.

The objective is recovering the full match function m i so that x i is matched to y mi .

The performance metric is the percentage of images for which we have found the exact match in the other domain.

This is calculated separately for A ??? B and B ??? A.The results are presented in Table.

1.

Several observations are apparent from the results: matching between the domains using pixels or deep features cannot solve this task.

The domains used in our experiments are different enough such that generic features are not easily able to match between them.

Simple mapping using CycleGAN and matching using pixel-losses does improve matching performance in most cases.

CycleGAN performance with simple matching however leaves much space for improvement.

The next baseline method matched perceptual features between the mapped source images and the target images.

Perceptual features have generally been found to improve performance for image retrieval tasks.

In this case we use VGG features as perceptual features as described in Sec. 3.4.

We found exhaustive search too computationally expensive (either in memory or runtime) for our datasets, and this required subsampling the features.

Perceptual features performed better than pixel matching.

We also run the ?? iterations step on mapped source domain images and target images.

This method matched linear combinations of mapped images rather than a single image (the largest ?? component was selected as the match).

This method is less sensitive to outliers and uses the same ?? parameters for both sides of the match (A ??? B and B ??? A) to improve identification.

The performance of this method presented significant improvements.

The exemplar loss alone should in principle recover a plausible solution for the matches between the domains and the mapping function.

However, the optimization problem is in practice hard and did not converge.

We therefore use a distributional auxiliary loss to aid optimization.

When optimized with the auxiliary losses, the exemplar loss was able to converge through ?? ??? T iterations.

This shows that the distribution and cycle auxiliary losses are essential for successful analogy finding.

Our full-method AN-GAN uses the full exemplar-based loss and can therefore optimize the mapping function so that each source sample matches the nearest target sample.

It therefore obtained significantly better performance for all datasets and for both matching directions.

In this set of experiments we used the same datasets as above but with M % of the matches being unavailable This was done by randomly removing images from the A and B domain datasets.

In this scenario M % of the domain A samples do not have a match in the sample set in the B domain and similarly M % of the B images do not have a match in the A domain.(1 ??? M )% of A and B images contain exact matches in the opposite domain.

The task is identification of the correct matches for all the samples that possess matches in the other domain.

The evaluation metric is the percentage of images for which we found exact matches out of the total numbers of images that have an exact match.

Apart from the removal of the samples resulting in M % of non-matching pairs, the protocol is identical to Sec. 4.1.1.The results for partial exact matching are shown in Table.

2.

It can be clearly observed that our method is able to deal with scenarios in which not all examples have matches.

When 10% of samples do not have matches, results are comparable to the clean case.

The results are not significantly lower for most datasets containing 25% of samples without exact matches.

Although in the general case a low exact match ratio lowers the quality of mapping function and decreases the quality of matching, we have observed that for several datasets (notably Facades), AN-GAN has achieved around 90% match rate with as much as 75% of samples not having matches.

Although the main objective of this paper is identifying exact analogies, it is interesting to test our approach on scenarios in which no exact analogies are available.

In this experiment, we qualitatively evaluate our method on finding similar matches in the case where an exact match is not available.

We evaluate on the Shoes2Handbags scenario from BID9 .

As the CycleGAN architecture is not effective at non-localized mapping we used the DiscoGAN architecture BID9 for the mapping function (and all of the relevant hyper-parameters from that paper).In FIG2 we can observe several analogies made for the Shoes2Handbags dataset.

The top example shows that when DiscoGAN is able to map correctly, matching works well for all methods.

However in the bottom two rows, we can see examples that the quality of the DiscoGAN mapping is lower.

In this case both the DiscoGAN map and DiscoGAN + ?? iterations present poor matches.

On the other hand AN ??? GAN forced the match to be more relevant and therefore the analogies found by AN ??? GAN are better.

We have shown that our method is able to align datasets with high accuracy.

We therefore suggested a two-step approach for training a mapping function between two datasets that contain exact matches but are unaligned: (i) Find analogies using AN ??? GAN , and (ii) Train a standard mapping function using the self-supervision from stage (i).For the Facades dataset, we were able to obtain 97% alignment accuracy.

We used the alignment to train a fully self-supervised mapping function using Pix2Pix .

We evaluate on the facade photos to segmentations task as it allows for quantitative evaluation.

In Fig. 3 we show two facade photos from the test set mapped by: CycleGAN, Pix2Pix trained on AN-GAN matches DISPLAYFORM0 Figure 3: Supervised vs unsupervised image mapping: The supervised mapping is far more accurate than unsupervised mapping, which is often unable to match the correct colors (segmentation labels).

Our method is able to find correspondences between the domains and therefore makes the unsupervised problem, effectively supervised.

and a fully-supervised Pix2Pix approach.

We can see that the images mapped by our method are of higher quality than CycleGAN and are about the fully-supervised quality.

In Table.

3 we present a quantitative comparison on the task.

As can be seen, our self-supervised method performs similarly to the fully supervised method, and much better than CycleGAN.We also show results for the edges2shoes and edges2handbags datasets.

The supervised stage uses a Pix2Pix architecture, but only L 1 loss (rather than the combination with cGAN as in the paper -L 1 only works better for this task).

The test set L 1 error is shown in Tab.

4.

It is evident that the use of an appropriate loss and larger architecture enabled by the ANGAN-supervision yields improved performance over CycleGAN and is competitive with full-supervision.

We have also evaluated our method on point cloud matching in order to test our method in low dimensional settings as well as when there are close but not exact correspondences between the samples in the two domains.

Point cloud matching consists of finding the rigid 3D transformation between a set of points sampled from the reference and target 3D objects.

The target 3D object is a We ran the experiments using the Bunny benchmark, using the same setting as in BID17 .

In this benchmark, the object is rotated by a random 3D rotation, and we tested the success rate of our model in achieving alignment for various ranges of rotation angles.

For both CycleGAN and our method, the following architecture was used.

D is a fully connected network with 2 hidden layers, each of 2048 hidden units, followed by BatchNorm and with Leaky ReLU activations.

The mapping function is a linear affine matrix of size 3X3 with a bias term.

Since in this problem, the transformation is restricted to be a rotation matrix, in both methods we added a loss term that encourages orthonormality of the weights of the mapper.

Namely, W W T ??? I , where W are the weights of our mapping function.

Tab.

5 depicts the success rate for the two methods, for each rotation angle bin, where success is defined in this benchmark as achieving an RMSE alignment accuracy of 0.05.Our results significantly outperform the baseline results reported in BID17 at large angles.

Their results are given in graph form, therefore the exact numbers could not be presented in Tab.

5.

Inspection of the middle column of Fig.3 in BID17 will verify that our method performs the best for large transformations.

We therefore conclude that our method is effective also for low dimensional transformations and well as settings in which exact matches do not exist.

We presented an algorithm for performing cross domain matching in an unsupervised way.

Previous work focused on mapping between images across domains, often resulting in mapped images that were too inaccurate to find their exact matches.

In this work we introduced the exemplar constraint, specifically designed to improve match performance.

Our method was evaluated on several public datasets for full and partial exact matching and has significantly outperformed baseline methods.

It has been shown to work well even in cases where exact matches are not available.

This paper presents an alternative view of domain translation.

Instead of performing the full operation end-toend it is possible to (i) align the domains, and (ii) train a fully supervised mapping function between the aligned domains.

Future work is needed to explore matching between different modalities such as images, speech and text.

As current distribution matching algorithms are insufficient for this challenging scenario, new ones would need to be developed in order to achieve this goal.

<|TLDR|>

@highlight

Finding correspondences between domains by performing matching/mapping iterations