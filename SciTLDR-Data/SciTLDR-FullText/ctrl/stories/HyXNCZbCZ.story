We propose a novel hierarchical generative model with a simple Markovian structure and a corresponding inference model.

Both the generative and inference model are trained using the adversarial learning paradigm.

We demonstrate that the hierarchical structure supports the learning of progressively more abstract representations as well as providing semantically meaningful reconstructions with different levels of fidelity.

Furthermore, we show that minimizing the Jensen-Shanon divergence between the generative and inference network is enough to minimize the reconstruction error.

The resulting semantically meaningful hierarchical latent structure discovery is exemplified on the CelebA dataset.

There, we show that the features learned by our model in an unsupervised way outperform the best handcrafted features.

Furthermore, the extracted features remain competitive when compared to several recent deep supervised approaches on an attribute prediction task on CelebA. Finally, we leverage the model's inference network to achieve state-of-the-art performance on a semi-supervised variant of the MNIST digit classification task.

Deep generative models represent powerful approaches to modeling highly complex high-dimensional data.

There has been a lot of recent research geared towards the advancement of deep generative modeling strategies, including Variational Autoencoders (VAE) BID16 , autoregressive models BID32 b) and hybrid models BID9 BID31 .

However, Generative Adversarial Networks (GANs) BID8 have emerged as the learning paradigm of choice across a varied range of tasks, especially in computer vision BID47 , simulation and robotics BID7 BID41 .

GANs cast the learning of a generative network in the form of a game between the generative and discriminator networks.

While the discriminator is trained to distinguish between the true and generated examples, the generative model is trained to fool the discriminator.

Using a discriminator network in GANs avoids the need for an explicit reconstruction-based loss function.

This allows this model class to generate visually sharper images than VAEs while simultaneously enjoying faster sampling than autoregressive models.

Recent work, known as either ALI or BiGAN , has shown that the adversarial learning paradigm can be extended to incorporate the learning of an inference network.

While the inference network, or encoder, maps training examples x to a latent space variable z, the decoder plays the role of the standard GAN generator mapping from space of the latent variables (that is typically sampled from some factorial distribution) into the data space.

In ALI, the discriminator is trained to distinguish between the encoder and the decoder, while the encoder and decoder are trained to conspire together to fool the discriminator.

Unlike some approaches that hybridize VAE-style inference with GAN-style generative learning (e.g. BID20 , ), the encoder and decoder in ALI use a purely adversarial approach.

One big advantage of adopting an adversarial-only formalism is demonstrated by the high-quality of the generated samples.

Additionally, we are given a mechanism to infer the latent code associated with a true data example.

One interesting feature highlighted in the original ALI work is that even though the encoder and decoder models are never explicitly trained to perform reconstruction, this can nevertheless be easily done by projecting data samples via the encoder into the latent space, copying these values across to the latent variable layer of the decoder and projecting them back to the data space.

Doing this yields reconstructions that often preserve some semantic features of the original input data, but are perceptually relatively different from the original samples.

These observations naturally lead to the question of the source of the discrepancy between the data samples and their ALI reconstructions.

Is the discrepancy due to a failure of the adversarial training paradigm, or is it due to the more standard challenge of compressing the information from the data into a rather restrictive latent feature vector?

BID44 show that an improvement in reconstructions is achievable when additional terms which explicitly minimize reconstruction error in the data space are added to the training objective.

BID23 palliates to the non-identifiability issues pertaining to bidirectional adversarial training by augmenting the generator's loss with an adversarial cycle consistency loss.

In this paper we explore issues surrounding the representation of complex, richly-structured data, such as natural images, in the context of a novel, hierarchical generative model, Hierarchical Adversarially Learned Inference (HALI), which represents a hierarchical extension of ALI.

We show that within a purely adversarial training paradigm, and by exploiting the model's hierarchical structure, we can modulate the perceptual fidelity of the reconstructions.

We provide theoretical arguments for why HALI's adversarial game should be sufficient to minimize the reconstruction cost and show empirical evidence supporting this perspective.

Finally, we evaluate the usefulness of the learned representations on a semi-supervised task on MNIST and an attribution prediction task on the CelebA dataset.

Our work fits into the general trend of hybrid approaches to generative modeling that combine aspects of VAEs and GANs.

For example, Adversarial Autoencoders BID28 replace the Kullback-Leibler divergence that appears in the training objective for VAEs with an adversarial discriminator that learns to distinguish between samples from the approximate posterior and the prior.

A second line of research has been directed towards replacing the reconstruction penalty from the VAE objective with GANs or other kinds of auxiliary losses.

Examples of this include BID20 that combines the GAN generator and the VAE decoder into one network and that uses the loss of a pre-trained classifier as an additional reconstruction loss in the VAE objective.

Another research direction has been focused on augmenting GANs with inference machinery.

One particular approach is given by ; , where, like in our approach, there is a separate inference network that is jointly trained with the usual GAN discriminator and generator.

BID15 presents a theoretical framework to jointly train inference networks and generators defined on directed acyclic graphs by leverage multiple discriminators defined nodes and their parents.

Another related work is that of BID12 which takes advantage of the representational information coming from a pre-trained discriminator.

Their model decomposes the data generating task into multiple subtasks, where each level outputs an intermediate representation conditioned on the representations from higher level.

A stack of discriminators is employed to provide signals for these intermediate representations.

The idea of stacking discriminator can be traced back to BID4 which used used a succession of convolutional networks within a Laplacian pyramid framework to progressively increase the resolution of the generated images.

The goal of generative modeling is to capture the data-generating process with a probabilistic model.

Most real-world data is highly complex and thus, the exact modeling of the underlying probability density function is usually computationally intractable.

Motivated by this fact, GANs BID8 model the data-generating distribution as a transformation of some fixed distribution over latent variables.

In particular, the adversarial loss, through a discriminator network, forces the generator network to produce samples that are close to those of the data-generating distribution.

While GANs are flexible and provide good approximations to the true data-generating mechanism, their original formulation does not permit inference on the latent variables.

In order to mitigate this, Adversarially Learned Inference (ALI) extends the GAN framework to include an inference network that encodes the data into the latent space.

The discriminator is then trained to discriminate between the joint distribution of the data and latent causes coming from the generator and inference network.

Thus, the ALI objective encourages a matching of the two joint distributions, which also results in all the marginals and conditional distributions being matched.

This enables inference on the latent variables.

We endeavor to improve on ALI in two aspects.

First, as reconstructions from ALI only loosely match the input on a perceptual level, we want to achieve better perceptual matching in the reconstructions.

Second, we wish to be able to compress the observables, x, using a sequence of composed features maps, leading to a distilled hierarchy of stochastic latent representations, denoted by z 1 to z L .

Note that, as a consequence of the data processing inequality BID2 , latent representations higher up in the hierarchy cannot contain more information than those situated lower in the hierarchy.

In information-theoretic terms, the conditional entropy of the observables given a latent variable is non-increasing as we ascend the hierarchy.

This loss of information can be seen as responsible for the perceptual discrepancy observed in ALI's reconstructions.

Thus, the question we seek to answer becomes: How can we achieve high perceptual fidelity of the data reconstructions while also having a compressed latent space that is strongly coupled with the observables?

In this paper, we propose to answer this using a novel model, Hierarchical Adversarially Learned Inference (HALI), that uses a simple hierarchical Markovian inference network that is matched through adversarial training to a similarly constructed generator network.

Furthermore, we discuss the hierarchy of reconstructions induced by the HALI's hierarchical inference network and show that the resulting reconstruction errors are implicitly minimized during adversarial training.

Also, we leverage HALI's hierarchial inference network to offer a novel approach to semi-supervised learning in generative adversarial models.

Denote by P(S) the set of all probability measures on some set S. Let T Z|X be a Markov kernel associating to each element x ∈ X a probability measure P Z|X=x ∈ P(Z).

Given two Markov kernels T W |V and T V |U , a further Markov kernel can be defined by composing these two and then marginalizing over V , i.e. T W |V • T V |U : U → P(W ).

Consider a set of random variables x, z 1 , . . .

, z L .

Using the composition operation, we can construct a hierarchy of Markov kernels or feature transitions as DISPLAYFORM0 A desirable property for these feature transitions is to have some form of inverses.

Motivated by this, we define the adjoint feature transition as DISPLAYFORM1 This can be interpreted as the generative mechanism of the latent variables given the data being the "inverse" of the data-generating mechanism given the latent variables.

Let q(x) denote the distribution of the data and p(z L ) be the prior on the latent variables.

Typically the prior will be a simple distribution, e.g. DISPLAYFORM2 The composition of Markov kernels in Eq. 1, mapping data samples x to samples of the latent variables z L using z 1 , . . .

, z L−1 constitutes the encoder.

Similarly, the composition of kernels in Eq. 2 mapping prior samples of z L to data samples x through z L−1 , . . .

, z 1 constitutes the decoder.

Thus, the joint distribution of the encoder can be written as DISPLAYFORM3 while the joint distribution of the decoder is given by DISPLAYFORM4 Algorithm 1 HALI training procedure.

DISPLAYFORM5 Sample from the prior for l ∈ {1, . . .

, L} dô z DISPLAYFORM6 Sample from each level in the encoder's hierarchy end for for l ∈ {L . . .

1} do z DISPLAYFORM7 Sample from each level in the decoder's hierarchy end for ρ DISPLAYFORM8 Get discriminator predictions on decoder's distribution end for DISPLAYFORM9 Compute discriminator loss DISPLAYFORM10 Compute generator loss DISPLAYFORM11 Gradient update on generator networks until convergenceThe encoder and decoder distributions can be visualized graphically as DISPLAYFORM12 Having constructed the joint distributions of the encoder and decoder, we can now match these distributions through adversarial training.

It can be shown that, under an ideal (non-parametric) discriminator, this is equivalent to minimizing the Jensen-Shanon divergence between the joint Eq. 3 and Eq. 4, see .

Algorithm 1 details the training procedure.

The Markovian character of both the encoder and decoder implies a hierarchy of reconstructions in the decoder.

In particular, for a given observation x ∼ p(x), the model yields L different reconstructionŝ x l ∼ T x|z l • T z l |x for l ∈ {1, . . .

, L} withx l the reconstruction of the x at the l-th level of the hierarchy.

Here, we can think of T z l |x as projecting x to the l-th intermediate representation and T x|z l as projecting it back to the input space.

Then, the reconstruction error for a given input x at the l-th hierarchical level is given by DISPLAYFORM0 Contrary to models that try to merge autoencoders and adversarial models, e.g. BID36 BID20 , HALI does not require any additional terms in its loss function in order to minimize the above reconstruction error.

Indeed, the reconstruction errors at the different levels of HALI are minimized down to the amount of information about x that a given level of the hierarchy is able to encode as training proceeds.

Furthermore, under an optimal discriminator, training in HALI minimizes the Jensen-Shanon divergence between q(x, z 1 , . . .

, z L ) and p(x, z 1 , . . .

, z L ) as formalized in Proposition 1 below.

Furthermore, the interaction between the reconstruction error and training dynamics is captured in Proposition 1.

Proposition 1.

Assuming q(x, z l ) is bounded away for zero for all l ∈ {1, . . .

, L}, we have that DISPLAYFORM1 where H(x | z l ) is computed under the encoder's distribution and K is as defined in Lemma 2 in the appendix.

On the other hand, proposition 2 below relates the intermediate representations in the hierarchy to the corresponding induced reconstruction error.

Proposition 2.

For any given latent variable z l , DISPLAYFORM2 i.e. the reconstruction error is an upper bound on H(x | z l ).In summary, Propositions 1 and 2 establish the dynamics between the hierarchical representation learned by the inference network, the reconstruction errors and the adversarial matching of the joint distributions Eq. 3 and Eq. 4.

The proofs on the two propositions above are deferred to the appendix.

Having theoretically established the interplay between layer-wise reconstructions and the training mechanics, we now move to the empirical evaluation of HALI.

We designed our experiments with the objective of addressing the following questions: Is HALI successful in improving the fidelity perceptual reconstructions?

Does HALI induces a semantically meaningful representation of the observed data?

Are the learned representations useful for downstream classification tasks?

All of these questions are considered in turn in the following sections.

We evaluated HALI on four datasets, CIFAR10 BID18 , SVHN BID30 , ImageNet 128x128 BID37 and CelebA BID25 .

We used two conditional hierarchies in all experiments with the Markov kernels parametrized by conditional isotropic Gaussians.

For SVHN, CIFAR10 and CelebA the resolutions of two level latent variables are z 1 ∈ R 64×16×16 and z 2 ∈ R 256 .

For ImageNet, the resolutions is z 1 ∈ R 64×32×32 and z 2 ∈ R 256 .For both the encoder and decoder, we use residual blocks BID10 with skip connections between the blocks in conjunction with batch normalization BID13 .

We use convolution with stride 2 for downsampling in the encoder and bilinear upsampling in the decoder.

In the discriminator, we use consecutive stride 1 and stride 2 convolutions and weight normalization BID38 .

To regularize the discriminator, we apply dropout every 3 layers with a probability of retention of 0.2.

We also add Gaussian noise with standard deviation of 0.2 at the inputs of the discriminator and the encoder.

One of the desired objectives of a generative model is to reconstruct the input images from the latent representation.

We show that HALI offers improved perceptual reconstructions relative to the (non-hierarchical) ALI model.

First, we present reconstructions obtained on ImageNet.

Reconstructions from SVHN and CIFAR10 can be seen in FIG7 in the appendix.

Fig. 1 highlights HALI's ability to reconstruct the input samples with high fidelity.

We observe that reconstructions from the first level of the hierarchy exhibit local differences in the natural images, while reconstructions from the second level of the hierarchy displays global change.

Higher conditional reconstructions are more often than not reconstructed as a different member of the same class.

Moreover, we show in Fig. 2 that this increase in reconstruction fidelity does not impact the quality of the generative samples from HALI's decoder.

We further investigate the quality of the reconstructions with a quantitative assessment of the preservation of perceptual features in the input sample.

For this evaluation task, we use the CelebA dataset where each image comes with a 40 dimensional binary attributes vector.

A VGG-16 classifier BID42 was trained on the CelebA training set to classify the individual attributes.

This trained model is then used to classify the attributes of the reconstructions from the validation set.

We consider a reconstruction as being good if it preserves -as measured by the trained classifier -the attributes possessed by the original sample.

We report a summary of the statistics of the classifier's accuracies in Table 1 .

We do this for three different models, VAE, ALI and HALI.

An inspection of the table reveals that the proportion of attributes where HALI's reconstructions outperforms the other models is clearly dominant.

Therefore, the encoder-decoder relationship of HALI better preserves the identifiable attributes compared to other models leveraging such relationships.

Please refer to Table 5 in the appendix for the full table of attributes score.

In the same spirit as Larsen et al. FORMULA0 , we construct a metric by computing the Euclidean distance between the input images and their various reconstructions in the discriminator's feature space.

More precisely, let · →D(·) be the embedding of the input to the pen-ultimate layer of the discriminator.

We compute the discriminator embedded distance DISPLAYFORM0 where · → · 2 is the Euclidean norm.

We then compute the average distances d c (x,x 1 ) and d c (x,x 2 ) over the ImageNet validation set.

Fig. 3a shows that under d c , the average reconstruction errors for bothx 1 andx 2 decrease steadily as training advances.

Furthermore, the reconstruction error under d c of the reconstructions from the first level of the hierarchy are uniformly bounded by above by those of the second.

We note that while the VAEGAN model of BID20 explicitly minimizes the perceptual reconstruction error by adding this term to their loss function, HALI implicitly minimizes it during adversarial training, as shown in subsection 3.2.(a) (b) Figure 3 : Comparison of average reconstruction error over the validation set for each level of reconstructions using the Euclidean (a) and discriminator embedded (b) distances.

Using both distances, reconstructions errors for x ∼ T x|z 1 are uniformly below those for x ∼ T x|z 2 .

The reconstruction error using the Euclidean distance eventually stalls showing that the Euclidean metric poorly approximates the manifold of natural images.

We now move on to assessing the quality of our learned representation through inpainting, visualizing the hierarchy and innovation vectors.

Inpainting is the task of reconstructing the missing or lost parts of an image.

It is a challenging task since sufficient prior information is needed to meaningfully replace the missing parts of an image.

While it is common to incorporate inpainting-specific training BID45 ; BID35 ; BID34 , in our case we simply use the standard HALI adversarial loss during training and reconstruct incomplete images during inference time.

We first predict the missing portions from the higher level reconstructions followed by iteratively using the lower level reconstructions that are pixel-wise closer to the original image.

FIG2 shows the inpaintings on center-cropped SVHN, CelebA and MS-COCO BID24 datasets without any blending post-processing or explicit supervision.

The effectiveness of our model at this task is due the hierarchy -we can extract semantically consistent reconstructions from the higher levels of the hierarchy, then leverage pixel-wise reconstructions from the lower levels.

To qualitatively show that higher levels of the hierarchy encode increasingly abstract representation of the data, we individually vary the latent variables and observe the effect.

The process is as follow: we sample a latent code from the prior distribution z 2 .

We then multiply individual components of the vector by scalars ranging from −3 to 3.

For z 1 , we fix z 2 and multiply each feature map independently by scalars ranging from −3 to 3.

In all cases these modified latent vectors are then decoded back to input data space.

FIG4 (a) and (b) exhibit some of those decodings for z 2 , while (c) and (d) do the same for the lower conditional z 1 .

The last column contain the decodings obtained from the originally sampled latent codes.

We see that the representations learned in the z 2 conditional are responsible for high level variations like gender, while z 1 codes imply local/pixel-wise changes such as saturation or lip color.

We sample a set of z2 vectors from the prior.

We repeatedly replace a single relevant entry in each vector by a scalar ranging from −3 to 3 and decode.

(c) and (d) follows the same process using the z1 latent space.

With HALI, we can exploit the jointly learned hierarchical inference mechanism to modify actual data samples by manipulating their latent codes.

We refer to these sorts of manipulations as latent semantic innovations.

Consider a given instance from a dataset x ∼ q(x).

Encoding x yieldsẑ 1 andẑ 2 .

We modifyẑ 2 by multiplying a specific entry by a scalar α.

We denote the resulting vector byẑ α 2 .

We decode the latter and getz α 1 ∼ T z1|z2 .

We decode the unmodified encoding vector and getz 1 ∼ T z1|ẑ2 .

We then form the innovation tensor η α =z 1 −z α 1 .

Finally, we subtract the innovation vector from the initial encoding, thus gettingẑ α 1 =ẑ 1 − η α , and samplex α ∼ T x|ẑ α

.

This method provides explicit control and allows us to carry out these variations on real samples in a completely unsupervised way.

The results are shown in FIG3 .

These were done on the CelebA validation set and were not used for training.

We evaluate the usefulness of our learned representation for downstream tasks by quantifying the performance of HALI on attribute classification in CelebA and on a semi-supervised variant of the MNIST digit classification task.

Following the protocol established by BID0 ; BID25 , we train 40 linear SVMs on HALI encoder representations (i.e. we utilize the inference network) on the CelebA validation set and subsequently measure performance on the test set.

As in BID0 BID11 ; BID14 , we report the balanced accuracy in order to evaluate the attribute prediction performance.

We emphasize that, for this experiment, the HALI encoder and decoder were trained in on entirely unsupervised data.

Attribute labels were only used to train the linear SVM classifiers.

A summary of the results are reported in TAB2 .

HALI's unsupervised features surpass those of VAE and ALI, but more remarkably, they outperform the best handcrafted features by a wide margin BID46 .

Furthermore, our approach outperforms a number of supervised BID11 and deeply supervised BID25 features.

Table 6 in the appendix arrays the results per attribute.

Std # Best Triplet-kNN BID40 71.55 12.61 0 PANDA BID46 76.95 13.33 0 Anet BID25 79.56 12.17 0 LMLE-kNN BID11 MNIST (# errors) VAE (M1+M2) BID17 233 ± 14 VAT BID29 136 CatGAN BID43 191 ± 10 Adversarial Autoencoder BID28 190 ± 10 PixelGAN BID27 108 ± 15 ADGM BID26 96 ± 2 Feature-Matching GAN 93 ± 6.5 Triple GAN BID22 91 ± 58 GSSLTRABG BID3 79.5 ± 9.8 HALI (ours) 73 Table 3 : Comparison on semi-supervised learning with state-of-the-art methods on MNIST with 100 labels instance per class.

Only methods without data augmentation are included.

The HALI hierarchy can also be used in a more integrated semi-supervised setting, where the encoder also receives a training signal from the supervised objective.

The currently most successful approach to semi-supervised in adversarially trained generative models are built on the approach introduced by .

This formalism relies on exploiting the discriminator's feature to differentiate between the individual classes present in the labeled data as well as the generated samples.

Taking inspiration from BID28 BID27 , we adopt a different approach that leverages the Markovian hierarchical inference network made available by HALI, DISPLAYFORM0 Where z = enc(x + σ ), with ∼ N (0, I), and y is a categorical random variable.

In practice, we characterize the conditional distribution of y given z by a softmax.

The cost of the generator is then augmented by a supervised cost.

Let us write D sup as the set of pairs all labeled instance along with their label, the supervised cost reads DISPLAYFORM1 We showcased this approach on a semi-supervised variant of MNIST(LeCun et al., 1998) digit classification task with 100 labeled examples evenly distributed across classes.

Table 3 shows that HALI achieves a new state-of-the-art result for this setting.

Note that unlike BID3 , HALI uses no additional regularization.

In this paper, we introduced HALI, a novel adversarially trained generative model.

HALI learns a hierarchy of latent variables with a simple Markovian structure in both the generator and inference networks.

We have shown both theoretically and empirically the advantages gained by extending the ALI framework to a hierarchy.

While there are many potential applications of HALI, one important future direction of research is to explore ways to render the training process more stable and straightforward.

GANs are well-known to be challenging to train and the introduction of a hierarchy of latent variables only adds to this.

Operation Kernel Strides Feature maps BN/WN?

Dropout Nonlinearity DISPLAYFORM0 DISPLAYFORM1 Concatenate D(x, z 1 ) and z 2 along the channel axis B PROOFS Lemma 1.

Let f be a valid f-divergence generator.

Let p and q be joint distributions over a random vector x. Let x A be any strict subset of x and x −A its complement, then DISPLAYFORM2 DISPLAYFORM3 Proof.

By definition, we have DISPLAYFORM4 Using that f is convex, Jensen's inequality yields DISPLAYFORM5 Simplifying the inner expectation on the right hand side, we conclude that DISPLAYFORM6 Lemma 2 (Kullback-Leibler's upper bound by Jensen-Shannon).

Assume that p and q are two probability distribution absolutely continuous with respect to each other.

Moreover, assume that q is bounded away from zero.

Then, there exist a positive scalar K such that DISPLAYFORM7 Proof.

We start by bounding the Kullblack-Leibler divergence by the χ 2 -distance.

We have DISPLAYFORM8 The first inequality follows by Jensen's inequality.

The third inequality follows by the Taylor expansion.

Recall that both the χ 2 -distance and the Jensen-Shanon divergences are f-divergences with generators given by f χ 2 (t) = (t − 1) 2 and f JS (t) = u log( 2t t+1 ) + log( 2t t+1 ), respectively.

We form the function t → h(t) = f χ 2 (t) f JS (t) .

h is strictly increasing on [0, ∞).

Since we are assuming q to be bounded away from zero, we know that there is a constant c 1 such that q(x) > c 1 for all x. Subsequently for all x, we have that q(x) ).

Intergrating with respect to q, we conclude DISPLAYFORM9 Proposition 3.

Assuming q(x, z l ) and p(x, z l ) are positive for any l ∈ {1, . . .

, L}. We have DISPLAYFORM10 Where H(x | z l ) is computed under the encoder's distribution q(x, z l )Proof.

By elementary manipulations we have.

DISPLAYFORM11 Where the conditional entropy H(x l | z l ) is computed q(x, z l ).

By the non-negativity of the KL-divergence we obtain DISPLAYFORM12 Using lemma 2, we have DISPLAYFORM13 The Jensen-Shanon divergence being f-divergence, using Lemma 1, we conclude DISPLAYFORM14 Proposition 4.

For any given latent variable z l , the reconstruction likelihood E x∼qx [E z∼T z l |x [− log p(x | z l )]] is an upper bound on H(x | z l ).Proof.

By the non-negativity of the Kullback-Leibler divergence, we have that DISPLAYFORM15 .

Integrating over the marginal and applying Fubini's theorem yields DISPLAYFORM16 where the conditional entropy H(x | z l ) is computed under the encoder distribution.

<|TLDR|>

@highlight

Adversarially trained hierarchical generative model with robust and semantically learned latent representation.