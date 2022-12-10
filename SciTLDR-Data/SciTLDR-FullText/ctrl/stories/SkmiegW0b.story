We study the problem of building models that disentangle independent factors of variation.

Such models encode features that can efficiently be used for classification and to transfer attributes between different images in image synthesis.

As data we use a weakly labeled training set, where labels indicate what single factor has changed between two data samples, although the relative value of the change is unknown.

This labeling is of particular interest as it may be readily available without annotation costs.

We introduce an autoencoder model and train it through constraints on image pairs and triplets.

We show the role of feature dimensionality and adversarial training theoretically and experimentally.

We formally prove the existence of the reference ambiguity, which is inherently present in the disentangling task when weakly labeled data is used.

The numerical value of a factor has different meaning in different reference frames.

When the reference depends on other factors, transferring that factor becomes ambiguous.

We demonstrate experimentally that the proposed model can successfully transfer attributes on several datasets, but show also cases when the reference ambiguity occurs.

One way to simplify the problem of classifying or regressing attributes of interest from data is to build an intermediate representation, a feature, where the information about the attributes is better separated than in the input data.

Better separation means that some entries of the feature vary only with respect to one and only one attribute.

In this way, classifiers and regressors would not need to build invariance to many nuisance attributes.

Instead, they could devote more capacity to discriminating the attributes of interest, and possibly achieve better performance.

We call this task disentangling factors of variation, and we identify attributes with the factors.

In addition to facilitating classification and regression, this task is beneficial to image synthesis.

One could build a model to render images, where each input varies only one attribute of the output, and to transfer attributes between images.

When labeling is possible and available, supervised learning can be used to solve this task.

In general, however, some attributes may not be easily quantifiable (e.g., style).

Therefore, we consider using weak labeling, where we only know what attribute has changed between two images, although we do not know by how much.

This type of labeling may be readily available in many cases without manual annotation.

For example, image pairs from a stereo system are automatically labeled with a viewpoint change, albeit unknown.

A practical model that can learn from these labels is an encoder-decoder pair subject to a reconstruction constraint.

In this model the weak labels can be used to define similarities between subsets of the feature obtained from two input images.

We introduce a novel adversarial training of autoencoders to solve the disentangling task when only weak labels are available.

Compared to previous methods, our discriminator is not conditioned on class labels, but takes image pairs as inputs.

This way the number of parameters can be kept constant.

We describe the shortcut problem, where all the the information is encoded only in one part of the feature, while other part is completely ignored, as FIG0 illustrates.

We prove our method solves this problem and demonstrate it experimentally.

We formally prove existence of the reference ambiguity, that is inherently present in the disentangling task when weak labels are used.

Thus no algorithm can provably learn disentangling.

As FIG0 shows, the reference ambiguity means that a factor (for example viewpoint) can have different meaning when using a different reference frame that depends on another factor (for example car type).

We show experimentally that this ambiguity rarely arise, we can observe it only when the data is complex.

Autoencoders.

Autoencoders in BID1 , BID7 , BID0 learn to reconstruct the input data as x = Dec(Enc(x)), where Enc(x) is the internal image representation (the encoder) and Dec (the decoder) reconstructs the input of the encoder.

Variational autoencoders in Kingma & Welling (2014) use a generative model; p(x, z) = p(x|z)p(z), where x is the observed data (images), and z are latent variables.

The encoder estimates the parameters of the posterior, Enc(x) = p(z|x), and the decoder estimates the conditional likelihood, Dec(z) = p(x|z).

In BID8 autoencoders are trained with transformed image input pairs.

The relative transformation parameters are also fed to the network.

Because the internal representation explicitly represents the objects presence and location, the network can learn their absolute position.

One important aspect of the autoencoders is that they encourage latent representations to keep as much information about the input as possible.

GAN.

Generative Adversarial Nets BID6 learn to sample realistic images with two competing neural networks.

The generator Dec creates images x = Dec(z) from a random noise sample z and tries to fool a discriminator Dsc, which has to decide whether the image is sampled from the generator p g or from real images p real .

After a successful training the discriminator cannot distinguish the real from the generated samples.

Adversarial training is often used to enforce constraints on random variables.

BIGAN, BID5 learns a feature representation with adversarial nets by training an encoder Enc, such that Enc(x) is Gaussian, when x ∼ p real .

CoGAN, Liu & Tuzel (2016) learns the joint distribution of multi-domain images by having generators and discriminators in each domain, and sharing their weights.

They can transform images between domains without being given correspondences.

InfoGan, BID3 learns a subset of factors of variation by reproducing parts of the input vector with the discriminator.

Disentangling and independence.

Many recent methods use neural networks for disentangling features, with various degrees of supervision.

In Xi Peng (2017) multi-task learning is used with full supervision for pose invariant face recognition.

Using both identity and pose labels BID19 can learn pose invariant features and synthesize frontalized faces from any pose.

In BID22 autoencoders are used to generate novel viewpoints of objects.

They disentangle the object category factor from the viewpoint factor by using as explicit supervision signals: the relative viewpoint transformations between image pairs.

In BID4 the output of the encoder is split in two parts: one represents the class label and the other represents the nuisance factors.

Their objective function has a penalty term for misclassification and a cross-covariance cost to disentangle class from nuisance factors.

Hierarchical Boltzmann Machines are used in BID15 for disentangling.

A subset of hidden units are trained to be sensitive to a specific factor of variation, while being invariant to others.

Variational Fair Autoencoders BID12 learn a representation that is invariant to specific nuisance factors, while retaining as much information as possible.

Autoencoders can also be used for visual analogy .

GAN is used for disentangling intrinsic image factors (albedo and normal map) in BID17 without using ground truth labeling.

They achieve this by explicitly modeling the physics of the image formation in their network.

The work most related to ours is BID13 , where an autoencoder restores an image from another by swapping parts of the internal image representation.

Their main improvement over is the use of adversarial training, which allows for learning with image pairs instead of image triplets.

Therefore, expensive labels like viewpoint alignment between different car types are no longer needed.

One of the differences between this method and ours is that it trains a discriminator for each of the given labels.

A benefit of this approach is the higher selectivity of the discriminator, but a drawback is that the number of model parameters grows linearly with the number of labels.

In contrast, we work with image pairs and use a single discriminator so that our method is uninfluenced by the number of labels.

Moreover, we show formally and experimentally the difficulties of disentangling factors of variation.

We are interested in the design and training of two models.

One should map a data sample (e.g., an image) to a feature that is explicitly partitioned into subvectors, each associated to a specific factor of variation.

The other model should map this feature back to an image.

We call the first model the encoder and the second model the decoder.

For example, given the image of a car we would like the encoder to yield a feature with two subvectors: one related to the car viewpoint, and the other related to the car type.

The subvectors of the feature obtained from the encoder should be useful for classification or regression of the corresponding factor that they depend on (the car viewpoint and type in the example).

This separation would also be very useful to the decoder.

It would enable advanced editing of images, for example, the transfer of the viewpoint or car types from an image to another, by swapping the corresponding subvectors.

Next, we introduce our model of the data and formal definitions of our encoder and decoder.

Data model.

We assume that our observed data x is generated through some unknown deterministic invertible and smooth process f that depends on the factors v and c, so that x = f (v, c).

In our earlier example, x is an image, v is a viewpoint, c is a car type, and f is the rendering engine.

It is reasonable to assume that f is invertible, as for most cases the factors are readily apparent form the image.

We assume f is smooth, because a small change in the factors should only result in a small change in the image and vice versa.

We denote the inverse of the rendering engine as DISPLAYFORM0 c ], where the subscript refers to the recovered factor.

Weak labeling.

In the training we are given pairs of images x 1 and x 2 , where they differ in v (varying factor), but they have the same c (common factor).

We also assume that the two varying factors and the common factor are sampled independently, v 1 ∼ p v , v 2 ∼ p v and c ∼ p c .

The images are generated as x 1 = f (v 1 , c) and DISPLAYFORM1 We call this labeling weak, because we do not know the absolute values of either the v or c factors or even relative changes between v 1 and v 2 .

All we know is that the image pairs share the same common factor.

The encoder.

Let Enc be the encoder mapping images to features.

For simplicity, we consider features split into only two column subvectors, N v and N c , one associated to the varying factor v and the other associated to the common factor c. Then, we have that Enc(x) = [N v (x), N c (x)].

Ideally, we would like to find the inverse of the image formation function, [N v , N c ] = f −1 , which separates and recovers the factors v and c from data samples x, i.e., DISPLAYFORM2 In practice, this is not possible because any bijective transformation of v and c could be undone by f and produce the same output x. Therefore, we aim for N v and N c that satisfy the following feature disentangling properties DISPLAYFORM3 for all v, c, and for some bijective functions R v and R c , so that N v is invariant to c and N c is invariant to v.

The decoder.

Let Dec be the decoder mapping features to images.

The sequence encoder-decoder is constrained to form an autoencoder, so DISPLAYFORM4 To use the decoder for image synthesis, so that each input subvector affects only one factor in the rendered image, the ideal decoder should satisfy the data disentangling property DISPLAYFORM5 for any v 1 , v 2 , c 1 , and c 2 .

The equation above describes the transfer of the varying factor v 1 of x 1 and the common factor c 2 of x 2 to a new image DISPLAYFORM6 In the next section we describe our training method for disentangling.

We introduce a novel adversarial term, that does not need to be conditioned on the common factor, rather it uses only image pairs, that keeps the model parameters constant.

Then we address the two main challenges of disentangling, the shortcut problem and the reference ambiguity.

We discuss which disentanglement properties can be (provably) achieved by our (or any) method.

In our training procedure we use two terms in the objective function: an autoencoder loss and an adversarial loss.

We describe these losses in functional form, however the components are implemented using neural networks.

In all our terms we use the following sampling of independent factors DISPLAYFORM0 The images are formed as DISPLAYFORM1 The images x 1 and x 2 share the same common factor, and x 1 and x 3 are independent.

In our objective functions, we use either pairs or triplets of the above images.

Autoencoder loss.

In this term, we use images x 1 and x 2 with the same common factor c 1 .

We feed both images to the encoder.

Since both images share the same c 1 , we impose that the decoder should reconstruct x 1 from the encoder subvector N v (x 1 ) and the encoder subvector N c (x 2 ), and similarly for the reconstruction of x 2 .

The autoencoder objective is thus defined as DISPLAYFORM2 Adversarial loss.

We introduce an adversarial training where the generator is our encoder-decoder pair and the discriminator Dsc is a neural network, which takes image pairs as input.

The discriminator learns to distinguish between real image pairs [x 1 , x 2 ] and fake ones [ DISPLAYFORM3 .

If the encoder were ideal, the image x 3⊕1 would be the result of taking the common factor from x 1 and the varying factor from x 3 .

The generator learns to fool the discriminator, so that x 3⊕1 looks like the random variable x 2 (the common factor is c 1 and the varying factor is independent of v 1 ).

To this purpose, the decoder must make use of N c (x 1 ), since x 3 does not carry any information about c 1 .

The objective function is thus defined as DISPLAYFORM4 Composite loss.

Finally, we optimize the weighted sum of the two losses DISPLAYFORM5 where λ regulates the relative importance of the two losses.

Ideally, at the global minimum of L AE , N v relates only to the factor v and N c only to c. However, the encoder may map a complete description of its input into N v and the decoder may completely ignore N c .

We call this challenge the shortcut problem.

When the shortcut problem occurs, the decoder is invariant to its second input, so it does not transfer the c factor correctly, DISPLAYFORM0 The shortcut problem can be addressed by reducing the dimensionality of N v , so it cannot build a complete representation of all input images.

This also forces the encoder and decoder to make use of N c for the common factor.

However, this strategy may not be convenient as it leads to a time consuming trial-and-error procedure to find the correct dimensionality.

A better way to address the shortcut problem is to use adversarial training (7) (8).

Proposition 1.

Let x 1 , x 2 and x 3 data samples generated according to (5), where the factors c 1 , c 3 , v 1 , v 2 , v 3 are jointly independent, and x 3⊕1 .

= Dec(N v (x 3 ), N c (x 1 )).

When the global optimum of the composite loss (8) is reached, the c factor is transferred to x 3⊕1 , i.e. f −1 c (x 3⊕1 ) = c 1 .Proof.

When the global optimum of FORMULA12 is reached, the distribution of real [x 1 , x 2 ] and fake [x 1 , x 3⊕1 ] image pairs are identical.

We compute statistics of the inverse of the rendering engine of the common factor f −1 c on the data.

For the images x 1 and x 2 we obtain DISPLAYFORM1 by construction (of x 1 and x 2 ).

For the images x 1 and x 3⊕1 we obtain DISPLAYFORM2 where DISPLAYFORM3 c (x 3⊕1 ).

We achieve equality if and only if c 1 = c 3⊕1 everywhere.

Let us consider the ideal case where we observe the space of all images.

When weak labels are made available to us, we also know what images x 1 and x 2 share the same c factor (for example, which images have the same car).

This labeling is equivalent to defining the probability density function p c and the joint conditional p x1,x2|c , where DISPLAYFORM0 Firstly, we show that the labeling allows us to satisfy the feature disentangling property for c (2).

For any [x 1 , x 2 ] ∼ p x1,x2|c we impose N c (x 1 ) = N c (x 2 ).

In particular, this equation is true for pairs when one of the two images is held fixed.

Thus, a function C(c) = N c (x 1 ) can be defined, where the C only depends on c, because N c is invariant to v. Lastly, images with the same v, but different c must also result in different features, DISPLAYFORM1 , otherwise the autoencoder constraint (3) cannot be satisfied.

Then, there exists a bijective function R c = C −1 such that property (2) is satisfied for c. Unfortunately the other disentangling properties can not provably be satisfied.

Definition 1.

A function g reproduces the data distribution, when it generates samples y 1 = g(v 1 , c) and y 2 = g(v 2 , c) that have the same distribution as the data.

Formally, [y 1 , y 2 ] ∼ p x1,x2 , where the latent factors are independent, v 1 ∼ p v , v 2 ∼ p v and c ∼ p c .The reference ambiguity occurs, when a decoder reproduces the data without satisfying the disentangling properties.

Proposition 2.

Let p v assign the same probability value to at least two different instances of v. Then, we can find encoders that reproduce the data distribution, but do not satisfy the disentangling properties for v in (2) and (4).Proof.

We already saw that N c satisfies (2), so we can choose N c = f −1 c , the inverse of the rendering engine.

Now we look at defining N v and the decoder.

The iso-probability property of p v implies that there exists a mapping T (v, c), such that T (v, c) ∼ p v and T (v, c 1 ) = T (v, c 2 ) for some v and c 1 = c 2 .

For example, let us denote with v 1 = v 2 two varying components such that DISPLAYFORM2 and C is a subset of the domain of c, where C p c (c)dc = 1/2.

Now, let us define the encoder as v, c) .

By using the autoencoder constraint, the decoder satisfies DISPLAYFORM3 DISPLAYFORM4 Even though T (v, c) depends on c functionally, they are statistically independent.

Because T (v, c) ∼ p v and c ∼ p c by construction, our encoder-decoder pair defines a data distribution identical to that given as training set DISPLAYFORM5 The feature disentanglement property is not satisfied because c 2 ) ), when c 1 ∈ C and c 2 ∈ C. Similarly, the data disentanglement property does not hold, because Dec( DISPLAYFORM6 DISPLAYFORM7 The above proposition implies that we cannot provably disentangle all the factors of variation from weakly labeled data, even if we had access to all the data and knew the distributions p v and p c .To better understand it, let us consider a practical example.

Let v ∼ U[−π, π] be the (continuous) viewpoint (the azimuth angle) and c ∼ B(0.5) the car type, where U denotes the uniform distribution and B(0.5) the Bernoulli distribution with probability p c (c = 0) = p c (c = 1) = 0.5 (i.e., there are only 2 car types).

In this case, every instance of v is iso-probable in p v so we have the worst scenario for the reference ambiguity.

We can define the function T (v, c) = v(2c − 1) so that the mapping of v is mirrored as we change the car type.

By construction T (v, c) ∼ U[−π, π] for any c and T (v, c 1 ) = T (v, c 2 ) for v = 0 and c 1 = c 2 .

So we cannot tell the difference between T and the ideal correct mapping to the viewpoint factor.

This is equivalent to an encoder DISPLAYFORM8 that reverses the ordering of the azimuth of car 1 with respect to car 0.

Each car has its own reference system, and thus it is not possible to transfer the viewpoint from one system to the other, as it is illustrated in FIG0 .

In our implementation we use convolutional neural networks for all the models.

We denote with θ the parameters associated to each network.

Then, the optimization of the composite loss can be written aŝ DISPLAYFORM0 We choose λ = 1 and also add regularization to the adversarial loss so that each logarithm has a minimum value.

We define log Dsc(x 1 , x 2 ) = log( + Dsc(x 1 , x 2 )) (and similarly for the other logarithmic term) and use = 10 −12 .

The main components of our neural network are shown in Fig. 2 .

The architecture of the encoder and the decoder were taken from DCGAN Radford et al. (2015) , with slight modifications.

We added fully connected layers at the output of the encoder and to the input of the decoder.

For the discriminator we used a simplified version of the VGG Simonyan & Zisserman (2014) network.

As the input to the discriminator is an image pair, we concatenate them along the color channels.

Normalization.

In our architecture both the encoder and the decoder networks use blocks with a convolutional layer, a nonlinear activation function (ReLU/leaky ReLU) and a normalization layer, typically, batch normalization (BN).

As an alternative to BN we consider the recently introduced instance normalization (IN) BID20 .

The main difference between BN and IN is that the latter just computes the mean and standard deviation across the spatial domain of the input and not along the batch dimension.

Thus, the shift and scaling for the output of each layer is the same at every iteration for the same input image.

In practice, we find that IN improves the performance.

We tested our method on the MNIST, Sprites and ShapeNet datasets.

We performed ablation studies on the shortcut problem using ShapeNet cars.

We focused on the effect of the feature dimensionality and having the adversarial term (L AE + L GAN ) or not (L AE ).

We also show that in most cases the reference ambiguity does not arise in practice (MNIST, Sprites, ShapeNet cars), we can only observe it when the data is more complex (ShapeNet chairs).

DISPLAYFORM0 Figure 2: Learning to disentangle factors of variation.

The scheme above shows how the encoder (Enc), the decoder (Dec) and the discriminator (Dsc) are trained with input triplets.

The components with the same name share weights.

ShapeNet cars.

The ShapeNet dataset BID2 contains 3D objects than we can render from different viewpoints.

We consider only one category (cars) for a set of fixed viewpoints.

Cars have high intraclass variability and they do not have rotational symmetries.

We used approximately 3K car types for training and 300 for testing.

We rendered 24 possible viewpoints around each object in a full circle, resulting in 80K images in total.

The elevation was fixed to 15 degrees and azimuth angles were spaced 15 degrees apart.

We normalized the size of the objects to fit in a 100 × 100 pixel bounding box, and placed it in the middle of a 128 × 128 pixel image.

In FIG2 we visualize the t-SNE embeddings of the N v features for several models using different feature sizes.

For the 2D case, we do not modify the data.

We can see that both L AE with 2 dimensions and L AE + L GAN with 128 separate the viewpoints well, but L AE with 128 dimensions does not due to the shortcut problem.

We investigate the effect of dimensionality of the N v features on the nearest neighbor classification task.

The performance is measured by the mean average precision.

For N v we use the viewpoint as ground truth.

DISPLAYFORM0 Mean average precision curves for the viewpoint prediction from the viewpoint feature using different models and dimensions for N v .

DISPLAYFORM1 We compare the different normalization choices in TAB0 .

We evaluate the case when batch, instance and no normalization are used and compute the performance on the nearest neighbor classification task.

We fixed the feature dimensions at 1024 for both N v and N c features in all normalization cases.

We can see that both batch and instance normalization perform equally well on viewpoint classification and no normalization is slightly worse.

For the car type classification instance normalization is clearly better.

MNIST.

The MNIST dataset BID10 contains handwritten grayscale digits of size 28 × 28 pixel.

There are 60K images of 10 classes for training and 10K for testing.

The common factor is the digit class and the varying factor is the intraclass variation.

We take image pairs that have the same digit for training, and use our full model L AE + L GAN with dimensions 64 for N v and 64 for N c .

In FIG4 Sprites.

The Sprites dataset contains 60 pixel color images of animated characters (sprites).

There are 672 sprites, 500 for training, 100 for testing and 72 for validation.

Each sprite has 20 animations and 178 images, so the full dataset has 120K images in total.

There are many changes in the appearance of the sprites, they differ in their body shape, gender, hair, armour, arm type, greaves, and weapon.

We consider character identity as the common factor and the pose as the varying factor.

We train our system using image pairs of the same sprite and do not exploit labels on their pose.

We train the L AE + L GAN model with dimensions 64 for N v and 448 for N c .

ShapeNet chairs.

We render the ShapeNet chairs with the same settings (viewpoints, image size) as the cars.

There are 3500 chair types for training and 3200 for testing, so the dataset contains 160K images.

We trained L AE + L GAN , and set the feature dimensions to 1024 for both N v and N c .

In Fig. 6 we show results on attribute transfer and compare it with ShapeNet cars.

We found that the reference ambiguity does not emerge for cars, but it does for chairs, possibly due to the higher complexity, as cars have much less variability than chairs.

In this paper we studied the challenges of disentangling factors of variation, mainly the shortcut problem and the reference ambiguity.

The shortcut problem occurs when all information is stored in only one feature chunk, while the other is ignored.

The reference ambiguity means that the reference in which a factor is interpreted, may depend on other factors.

This makes the attribute transfer ambiguous.

We introduced a novel training of autoencoders to solve disentangling using image triplets.

We showed theoretically and experimentally how to keep the shortcut problem under control through adversarial training, and enable to use large feature dimensions.

We proved that the reference ambiguity is inherently present in the disentangling task when weak labels are used.

Most importantly this can be stated independently of the learning algorithm.

We demonstrated that training and transfer of factors of variation may not be guaranteed.

However, in practice we observe that our trained model works well on many datasets and exhibits good generalization capabilities.

<|TLDR|>

@highlight

It is a mostly theoretical paper that describes the challenges in disentangling factors of variation, using autoencoders and GAN.

@highlight

This paper considers disentangling factors of variation in images, shows that in general, without further assumptions, one cannot tell apart two different variation factors, and suggests a novel AE+GAN architecture to try and disentangle variation factors.

@highlight

This paper studies the challenges of disentangling independent factors of variation under weakly labeled data and introduces the term reference ambiguity for data point mapping.