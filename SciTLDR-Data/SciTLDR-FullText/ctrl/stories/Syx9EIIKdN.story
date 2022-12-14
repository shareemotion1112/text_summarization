In this paper, we explore new approaches to combining information encoded within the learned representations of autoencoders.

We explore models that are capable of combining the attributes of multiple inputs such that a resynthesised output is trained to fool an adversarial discriminator for real versus synthesised data.

Furthermore, we explore the use of such an architecture in the context of semi-supervised learning, where we learn a mixing function whose objective is to produce interpolations of hidden states, or masked combinations of latent representations that are consistent with a conditioned class label.

We show quantitative and qualitative evidence that such a formulation is an interesting avenue of research.

The autoencoder is a fundamental building block in unsupervised learning.

Autoencoders are trained to reconstruct their inputs after being processed by two neural networks: an encoder which encodes the input to a high-level representation or bottleneck, and a decoder which performs the reconstruction using the representation as input.

One primary goal of the autoencoder is to learn representations of the input data which are useful BID1 , which may help in downstream tasks such as classification BID27 BID9 or reinforcement learning BID20 BID5 .

The representations of autoencoders can be encouraged to contain more 'useful' information by restricting the size of the bottleneck, through the use of input noise (e.g., in denoising autoencoders, BID23 , through regularisation of the encoder function BID17 , or by introducing a prior BID11 .

Another goal is in learning interpretable representations BID3 BID10 .

In unsupervised learning, learning often involves qualitative objectives on the representation itself, such as disentanglement of latent variables BID12 or maximisation of mutual information BID3 BID0 BID8 .Mixup BID26 and manifold mixup BID21 are regularisation techniques that encourage deep neural networks to behave linearly between two data samples.

These methods artificially augment the training set by producing random convex combinations between pairs of examples and their corresponding labels and training the network on these combinations.

This has the effect of creating smoother decision boundaries, which can have a positive effect on generalisation performance.

In BID21 , the random convex combinations are computed in the hidden space of the network.

This procedure can be viewed as using the high-level representation of the network to produce novel training examples and provides improvements over strong baselines in the supervised learning.

Furthermore, BID22 propose a simple and efficient method for semi-supervised classification based on random convex combinations between unlabeled samples and their predicted labels.

In this paper we explore the use of a wider class of mixing functions for unsupervised learning, mixing in the bottleneck layer of an autoencoder.

These mixing functions could consist of continuous interpolations between latent vectors such as in BID21 , to binary masking operations to even a deep neural network which learns the mixing operation.

In order to ensure that the output of the decoder given the mixed representation resembles the data distribution at the pixel level, we leverage adversarial learning BID4 , where here we train a discriminator to distinguish between decoded mixed and unmixed representations.

This technique affords a model the ability to simulate novel data points (such as those corresponding to combinations of annotations not present in the training set).

Furthermore, we explore our approach in the context of semi-supervised learning, where we learn a mixing function whose objective is to produce interpolations of hidden states consistent with a conditioned class label.

Our method can be thought of as an extension of autoencoders that allows for sampling through mixing operations, such as continuous interpolations and masking operations.

Variational autoencoders (VAEs, Kingma & Welling, 2013) can also be thought of as a similar extension of autoencoders, using the outputs of the encoder as parameters for an approximate posterior q(z|x) which is matched to a prior distribution p(z) through the evidence lower bound objective (ELBO).

At test time, new data points are sampled by passing samples from the prior, z ??? p(z), through the decoder.

In contrast, our we sample a random mixup operation between the representations of two inputs from the encoder.

The Adversarially Constrained Autoencoder Interpolation (ACAI) method is another approach which involves sampling interpolations as part of an unsupervised objective BID2 .

ACAI uses a discriminator network to predict the mixing coefficient from the decoder output of the mixed representation, and the autoencoder tries to 'fool' the discriminator, making interpolated points indistinguishable from real ones.

The GAIA algorithm BID18 ) uses a BEGAN framework with an additional interpolation-based adversarial objective.

What primarily differentiates our work from theirs is that we perform an exploration into different kinds of mixing functions, including a semi-supervised variant which uses an MLP to produce mixes consistent with a class label.

Let us consider an autoencoder model F (??), with the encoder part denoted as f (??) and the decoder g(??).

In an autoencoder we wish to minimise the reconstruction, which is simply: DISPLAYFORM0 Because autoencoders trained by input-reconstruction loss tend to produce images which are slightly blurry, one can train an adversarial autoencoder BID14 , but instead of putting the adversary on the bottleneck, we put it on the reconstruction, and the discriminator (denoted D) tries to distinguish between real and reconstructed x, and the autoencoder (which is analogous to the generator) tries to construct 'realistic' reconstructions so as to fool the discriminator.

Because of this, we coin the term 'ARAE' (adversarial reconstruction autoencoder).

This can be written as: DISPLAYFORM1 where GAN is a GAN-specific loss function.

In our case, GAN is the binary cross-entropy loss, which corresponds to the Jenson-Shannon GAN BID4 .One way to use the autoencoder to generate novel samples would be to encode two inputs h 1 = f (x 1 ) and h 2 = f (x 2 ) into their latent representation, perform some combination between them, and then run the result through the decoder g(??).

There are many ways one could combine the two latent representations, and we denote this function Mix(h 1 , h 2 ).

Manifold mixup BID21 implements mixing in the hidden space through convex combinations: DISPLAYFORM2 where ?? ??? [0, 1] (bs,) is sampled from a Uniform(0, 1) distribution and bs denotes the minibatch size.

In contrast, here we explore a strategy in which we randomly retain some components of the hidden representation from h 1 and use the rest from h 2 , and in this case we would randomly sample a binary mask m ??? {0, 1} (bs??f ) (where f denotes the number of feature maps) and perform the following operation: DISPLAYFORM3 where m is sampled from a Bernoulli(p) distribution (p can simply be sampled uniformly).With this in mind, we propose the adversarial mixup resynthesiser (AMR), where part of the autoencoder's objective is to produce mixes which, when decoded, are indistinguishable from real images.

The generator and the discriminator of AMR are trained by the following mixture of loss components: DISPLAYFORM4 fool D with reconstruction DISPLAYFORM5 fool D with mixes DISPLAYFORM6 label reconstruction as fake DISPLAYFORM7 label mixes as fake .Note that the mixing consistency loss is simply the reconstruction between the mixh mix = Mix(f (x), f (x )) and the re-encoding of it f (g(h mix )), where x and x are two randomly sampled images from the training set.

This may be necessary as without it the decoder may simply output an image which is not semantically consistent with the two images which were mixed (refer to Section 5.2 for an in-depth explanation and analysis of this loss).

Both the generator and discriminator are trained by the decoded image of the mix g(Mix(f (x), f (x ))).

The discriminator D is trained to label it as a fake image by minimising its probability and the generator F is trained to fool the discriminator by maximising its probability.

Note that the coefficient ?? controls the reconstruction and the coefficient ?? controls the mixing consistency in the generator.

See Figure 1 for a visualisation of the AMR model.

While it is interesting to generate new examples via random mixing strategies in the hidden states, we also explore a supervised mixing formulation in which we learn a mixing function that can produce mixes between two examples such that they are consistent with a particular class label.

We make this possible by backpropagating through a classifier network p(y|x) which branches off the end of the discriminator, i.e., an auxiliary classifier GAN BID16 .Let us assume that for some image x, we have a set of binary attributes y associated with it, where y ??? {0, 1} k (and k ??? 1).

We introduce a mixing function Mix sup (h 1 , h 2 , y), which is an MLP that maps y to Bernoulli parameters p ??? [0, 1] bs??f .

These parameters are used to sample a Bernoulli mask m ??? Bernoulli(p) to produce a new combinationh mix = mh 1 + (1 ??? m)h 2 , which is consistent with the class label y. Note that the conditioning class label should be semantically meaningful with respect to both of the conditioned hidden states.

For example, if we're producing mixes based on the gender attribute and both h 1 and h 2 are male, it would not make sense to condition on the 'female' label.

To enforce this constraint, we simply make the conditioning label a convex combination??? mix = ??y 1 + (1 ??? ??)y 2 as well, using ?? ??? Uniform(0, 1).

DISPLAYFORM0 The unsupervised version of the adversarial mixup resynthesiser (AMR).

In addition to the autoencoder loss functions, we have a mixing function Mix which creates some combination between the latent variables h 1 and h 2 , which is subsequently decoded into an image intended to be realistic-looking and semantically consistent with the two constituent images.

This is achieved through the consistency loss (weighted by ??) and the discriminator.

To make this more concrete, the autoencoder and discriminator, in addition to their losses described in Equation 5, try to minimise the following losses: DISPLAYFORM1 label mixes as fake DISPLAYFORM2 Note that for the consistency loss the same coefficient ?? is used.

See Figure 2 for a visualisation of the supervised AMR model.

We use ResNets BID6 for both the generator and discriminator.

The precise architectures for generator and discriminator can be found here.1 The datasets evaluated on are:??? UT Zappos50K BID24 : a large dataset comprising 50k images of shoes, sandals, slippers, and boots.

Each shoe is centered on a white background and in the same orientation, which makes it convenient for generative modelling purposes.??? CelebA BID13 : a large-scale and highly diverse face dataset consisting of 200K images.

We use the aligned and cropped version of the dataset downscaled to 64px, and only consider (via the use of a keypoint-based heuristic) frontal faces.

It is worth noting that despite this, there is still quite a bit of variation in terms of the size and position of the faces, which can make mixing between faces a more difficult task since the faces are not completely aligned.mixing with labels mixing without labels Figure 2 : The supervised version of the adversarial mixup resynthesiser (AMR).

The mixer function, denoted in this figure as Mix sup , takes h 1 , h 2 and a convex combination of y 1 and y 2 (denoted??? mix ) and internally produces a Bernoulli mask which is then used to produce an output combinatio?? h mix = mh 1 + (1 ??? m)h 2 .h mix is then passed to the generator to generatex mix .

In addition to fooling the discriminator usingx mix , the generator also has to make sure the class prediction by the auxiliary classifier is consistent with the mixed class??? mix .

Note that in this formulation, we still perform the kind of mixing which was shown in Figure 1 , and this is shown in the diagram with the component noted 'mixing without labels'.

BID2 and (d) the adversarial mixup resynthesiser (AMR).

For more images, consult the appendix section.

As seen in FIG0 , all of the mixup variants produce more realistic-looking interpolations than in pixel space.

Due to background details in CelebA however, it is slightly harder to distinguish the quality between the different methods.

Though this may not be the most ideal metric to use in our case (see discussion at end of this section)

we use the Frechet Inception Distance (FID) by BID7 , which is based on features extracted from a pre-trained CelebA classifier, to compute the distance between samples from the dataset and ones from our autoencoders 2 .

Concretely, we compute (on the validation set) two scores: the FID between validation samples and their reconstructions (denoted in the table as FID(data, reconstruction)), and the FID between validation samples and randomly sampled interpolations (denoted in the table as FID(data, mix)).

In the latter case, we repeat this five times (over five different sets of randomly sampled interpolations) for three different random seeds, resulting in 5 ?? 3 = 15 FID scores from which we compute the mean and standard deviation.

These results are shown in TAB0 for the mixup and Bernoulli mixup formulations, respectively.

Lower FID is usually considered to be better.

However, FID may not be the most appropriate metric to use in our case.

Because the FID is a measure of distance between two distributions, one can simply obtain a very low FID by simplying autoencoding the data, as shown in TAB0 .

In the case of mixing, one situation which may favour a lower FID is if g(??f ( DISPLAYFORM0 ; in other words, the supposed mix simply decodes into one of the original examples x 1 or x 2 , which clearly lie on the data manifold.

To avoid having the mixed features ??x 1 + (1 ??? ??)x 2 being decoded back into samples which lie on the data manifold, we leverage the consistency loss, which is tuned by coefficient ??.

The lower the coefficient, the more likely that decoded mixes are projected back onto the manifold, but if this constraint is too weak then it may not necessarily be desirable if one wants to create novel data points. (For more details, see Section 5.2 in the appendix.)Despite potential shortcomings of using FID, it seems reasonable to use such a metric to compare against baselines without any mixing losses, such as the adversarial reconstruction autoencoder (ARAE), which we indeed outperform for both mixup and Bernoulli mixup.

For Bernoulli mixup, the FID scores appear to be higher than those in the mixup case TAB0 because the sampled Bernoulli mask m is also across the channel axis, i.e., it is of the shape (bs, f ), whereas in mixup ?? has the shape (bs, ).

Because the mixing is performed on an extra axis, this produces a greater degree of variability in the mixes, and we have observed similar FID scores to the Bernoulli mixup case by evaluating on a variant of mixup where the ?? has the shape (bs, f ) instead of (bs, ).

We present some qualitative results with the supervised formulation.

We train our supervised AMR variant using a subset of the attributes in CelebA ('is male', 'is wearing heavy makeup', and 'is wearing lipstick').

We consider pairs of examples {(x 1 , y 1 ), (x 2 , y 2 )} (where one example is male and the other female) and produce random convex combinations of the attributes??? mix = ??y 1 + (1 ??? ??)y 2 and decode their resulting mixes Mix sup (f (x 1 ), f (x 2 ),??? mix ).

This can be seen in FIG2 .

: Interpolations produced by the class mixer function for the set of binary attributes {male, heavy makeup, lipstick}. For each image, the left-most face is x 1 and the right-most face x 2 , with faces in between consisting of mixes Mix sup (f (x 1 ), f (x 2 ),??? mix ) of a particular attribute mix??? mix , shown below each column (where red denotes 'off' and green denotes 'on').We can see that for the most part, the class mixer function has been able to produce decent mixes between the two faces consistent with the desired attributes.

There are some issues -namely, the model does not seem to disentangle the lipstick and makeup attributes well -but this may be due to the strong correlation between lipstick and makeup (lipstick is makeup!), or be in part due to the classification performance of the auxiliary classifier part of the discriminator (while its accuracy on both training and validation was as high as 95%, there may still be room for improvement).

We also achieved better results by simply having the embedding function produce a mask m ??? [0, 1] rather than {0, 1}, most likely because such a formulation allows a greater degree of flexibility when it comes to mixing.

Indeed, one priority is to conduct further hyperparameter tuning in order to improve these results.

For a visualisation of the Bernoulli parameters output by the embedding function, see Section 5.3 in the appendix.

In this paper, we proposed the adversarial mixup resynthesiser and showed that it can be used to produce realistic-looking combinations of examples by performing mixing in the bottleneck of an autoencoder.

We proposed several mixing functions, including one based on sampling from a uniform distribution and the other a Bernoulli distribution.

Furthermore, we presented a semisupervised version of the Bernoulli variant in which one can leverage class labels to learn a mixing function which can determine what parts of the latent code should be mixed to produce an image consistent with a desired class label.

While our technique can be used to leverage an autoencoder as a generative model, we conjecture that our technique may have positive effects on the latent representation and therefore downstream tasks, though this is yet to be substantiated.

Future work will involve more comparisons to existing literature and experiments to determine the effects of mixing on the latent space itself and downstream tasks.

We will provide a summary of our experimental setup here, though we also provide links to (and encourage viewers to look at) various parts of the code such as the networks used for the generator and discriminator and the optimiser hyperparameters.

We use a residual network for both the generator and discriminator.

The discriminator uses spectral normalisation BID15 , with five discriminator updates being performed for each generator update.

We use ADAM for our optimiser with ?? = 2e ???4 , ?? 1 = 0.0 and ?? 2 = 0.99.

In order to examine the effect of the consistency loss, we explore a simple two-dimensional spiral dataset, where points along the spiral are deemed to be part of the data distribution and points outside it are not.

With the mixup loss enabled and ?? = 10, we try values of ?? ??? {0, 0.1, 10, 100}. After 100 epochs of training, we produce decoded random mixes and plot them over the data distribution, which are shown as orange points (overlaid on top of real samples, shown in blue).

This is shown in FIG3 .As we can see, the lower ?? is, the more likely interpolated points will lie within the data manifold (i.e. the spiral).

This is because the consistency loss competes with the discriminator loss -as ?? is decreased, there is a relatively greater incentive for the autoencoder to try and fool the discriminator with interpolations, forcing it to decode interpolated points such that they lie in the spiral.

Ideally however we would want a bit of both: we want high consistency so that interpolations in hidden states are semantically meaningful (and do not decode into some other random data point), while also having those decoded interpolations look realistic.

Interpolations are defined as DISPLAYFORM0 We also compare our formulation to ACAI BID2 , which does not explicitly have a consistency loss term.

Instead, the discriminator tries to predict what the mixing coefficient ?? is, and the autoencoder tries to fool it into thinking interpolations have a coefficient of 0.

In FIG5 we compare this to our formulation in which ?? = 0.

This is shown in FIG5 (right figure) .

It appears that ACAI also prefers to place points in the spiral, although not as strongly as AMR with ?? = 0 (though this may be because ACAI needs to trained for longer -ACAI and AMR were trained for the same number of epochs).

In FIG5 we can see that over the course of training the consistency losses for both ACAI and AMR gradually rise, indicating both models' preference for moving interpolated points closer to the data manifold.

Note that here we are only observing the consistency loss during training, and it is not used in the generator's loss.

Lastly, in Figure 9 we show some side-by-side comparisons of our model interpolating between faces when ?? = 50 and ?? = 0.

We can see that when ?? = 0 interpolations between faces are not as smooth in terms of colour and lighting.

This somewhat slight discontinuity in the interpolation may be explained by the decoder pushing these interpolated points closer to the data manifold, since there is no consistency loss enforced.(a) Left: AMR with ?? = 10, ?? = 0; right: ACAI with ?? = 10 (?? = 0 since ACAI does not enforce a consistency loss).

AMR was trained for 200 epochs and ACAI for 300 epochs, since ACAI takes longer to converge.

DISPLAYFORM1 and ?? ??? U (0, 1) for randomly sampled {x 1 , x 2 }) over the course of training.

DISPLAYFORM2 Figure 9: Interpolations using AMR {?? = 50, ?? = 50} and {?? = 50, ?? = 0}.

To recap, the class mixer in the supervised formulation internally maps from a label??? mix to Bernoulli parameters p ??? [0, 1] K , from which a Bernoulli mask m ??? Bernoulli(p) is sampled.

The resulting Bernoulli parameters p are shown in Figure 10 , where each row denotes some combination of attributes y ??? {000, 001, 010, . . . } and the columns denote the index of p (spread out across four images, such that the first image denotes p 1:128 , second image p 128:256 , etc.).

We can see that each attribute combination spells out a binary combination of feature maps, which allows one to easily glean which feature maps contribute to which attributes.

Figure 10: Visualisation of Bernoulli parameters p internally produced by the class mixer function.

Rows denote attribute combinations y and columns denote the index of p.

In this section we show additional samples of the AMR model (using mixup and Bernoulli mixup variants) on Zappos and CelebA datasets.

We compare AMR against linear interpolation in pixel space (pixel), adversarial reconstruction autoencoder (ARAE), and adversarialy contrained autoencoder interpolation (ACAI).

As can be observed in the following images, the interpolations of pixel and ARAE are less realistic and suffer from more artifacts.

AMR and ACAI produce more realisticlooking results, while AMR generates a smoother transition between the two samples.??? Figure

<|TLDR|>

@highlight

We leverage deterministic autoencoders as generative models by proposing mixing functions which combine hidden states from pairs of images. These mixes are made to look realistic through an adversarial framework.