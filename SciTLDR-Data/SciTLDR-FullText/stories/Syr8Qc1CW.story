Disentangling factors of variation has always been a challenging problem in representation learning.

Existing algorithms suffer from many limitations, such as unpredictable disentangling factors, bad quality of generated images from encodings, lack of identity information, etc.

In this paper, we proposed a supervised algorithm called DNA-GAN trying to disentangle different attributes of images.

The latent representations of images are DNA-like, in which each individual piece represents an independent factor of variation.

By annihilating the recessive piece and swapping a certain piece of two latent representations, we obtain another two different representations which could be decoded into images.

In order to obtain realistic images and also disentangled representations, we introduced the discriminator for adversarial training.

Experiments on Multi-PIE and CelebA datasets demonstrate the effectiveness of our method and the advantage of overcoming limitations existing in other methods.

The success of machine learning algorithms depends on data representation, because different representations can entangle different explanatory factors of variation behind the data.

Although prior knowledge can help us design representations, the vast demand of AI algorithms in various domains cannot be met, since feature engineering is labor-intensive and needs domain expert knowledge.

Therefore, algorithms that can automatically learn good representations of data will definitely make it easier for people to extract useful information when building classifiers or predictors.

Of all criteria of learning good representations as discussed in BID1 , disentangling factors of variation is an important one that helps separate various explanatory factors.

For example, given a human-face image, we can obtain various information about the person, including gender, hair style, facial expression, with/without eyeglasses and so on.

All of these information are entangled in a single image, which renders the difficulty of training a single classifier to handle different facial attributes.

If we could obtain a disentangled representation of the face image, we may build up only one classifier for multiple attributes.

In this paper, we propose a supervised method called DNA-GAN to obtain disentangled representations of images.

The idea of DNA-GAN is motivated by the DNA double helix structure, in which different kinds of traits are encoded in different DNA pieces.

We make a similar assumption that different visual attributes in an image are controlled by different pieces of encodings in its latent representations.

In DNA-GAN, an encoder is used to encode an image to the attribute-relevant part and the attribute-irrelevant part, where different pieces in the attribute-relevant part encode information of different attributes, and the attribute-irrelevant part encodes other information.

For example, given a facial image, we are trying to obtain a latent representation that each individual part controls different attributes, such as hairstyles, genders, expressions and so on.

Though annihilating recessive pieces and swapping certain pieces, we can obtain novel crossbreeds that can be decoded into new images.

By the adversarial discriminator loss and the reconstruction loss, DNA-GAN can reconstruct the input images and generate new images with new attributes.

Each attribute is disentangled from others gradually though iterative training.

Finally, we are able to obtain disentangled representations in the latent representations.

The summary of contributions of our work is as follows:1.

We propose a supervised algorithm called DNA-GAN, that is able to disentangle multiple attributes as demonstrated by the experiments of interpolating multiple attributes on Multi-PIE BID5 and CelebA BID12 datasets.2.

We introduce the annihilating operation that prevents from trivial solutions: the attributerelevant part encodes information of the whole image instead of a certain attribute.3.

We employ iterative training to address the problem of unbalanced multi-attribute image data, which was theoretically proved to be more efficient than random image pairs.

Traditional representation learning algorithms focus on (1) probabilistic graphical models, characterized by Restricted Boltzmann Machine (RBM) BID19 , Autoencoder (AE) and their variants; (2) manifold learning and geometrical approaches, such as Principal Components Analysis (PCA) BID15 , Locally Linear Embedding (LLE) BID17 , Local Coordinate Coding (LCC) BID23 , etc.

However, recent research has actively focused on developing deep probabilistic models that learn to represent the distribution of data.

BID9 employs an explicit model distribution and uses variational inference to learn its parameters.

As the generative adversarial networks (GAN) BID4 has been invented, many implicit models are developed.

In the semi-supervised setting, BID18 learns a disentangled representations by using an auxiliary variable.

BID2 proposes the ML-VAE that can learn disentangled representations from a set of grouped observations.

In the unsupervised setting, InfoGAN BID3 tries to maximize mutual information between a small subset of latent variables and observations by introducing an auxiliary network to approximate the posterior.

However, it relies much on the a-priori choice of distributions and suffered from unstable training.

Another popular unsupervised method β-VAE BID6 , adapted from VAE, lays great stress on the KL distance between the approximate posterior and the prior.

However, unsupervised approaches do not anchor a specific meaning into the disentanglement.

More closely with our method, supervised methods take the advantage of labeled data and try to disentangle the factors as expected.

DC-IGN BID10 asks the active attribute to explain certain factor of variation by feeding the other attributes by the average in a mini-batch.

TD-GAN BID22 ) uses a tag mapping net to boost the quality of disentangled representations, which are consistent with the representations extracted from images through the disentangling network.

Besides, the quality of generated images is improved by implementing the adversarial training strategy.

However, the identity information should be labeled so as to preserve the id information when swapping attributes, which renders the limitation of applying it into many other datasets without id labels.

IcGAN BID16 ) is a multi-stage training algorithm that first takes the advantage of cGAN BID14 to learn a map from latent representations and conditional information to real images, and then learn its inverse map from images to the latent representations and conditions in a supervised manner.

The overall effect depends on each training stage, therefore it is hard to obtain satisfying images.

Unlike these models, our model requires neither explicit id information in labels nor multi-stage training.

Many works have studied the image-to-image translation between unpaired image data using GANbased architectures, see BID8 , BID21 , BID25 , BID11 and BID24 .

Interestingly, these models require a form of 0/1 weak supervision that is similar to our setting.

However, they are circumscribed in two image domains which are opposite to each other with respect to a single attribute.

Our model differs from theirs as we generalize to the case of multi-attribute image data.

Specifically, we employ the strategy of iterative training to overcome the difficulty of training on unbalanced multi-attribute image datasets.

In this section, we formally outline our method.

A set X of multi-labeled images and a set of labels Y are considered in our setting.

Let DISPLAYFORM0 } denote the whole training dataset, where X i ∈ X is the i-th image with its label Y i ∈ Y .

The small letter m denotes the number of samples in set X and n denotes the number of attributes.

The label DISPLAYFORM1 is a n-dimensional vector where each element represents whether X i has certain attribute or not.

For example, in the case of labels with three candidates [Bangs, Eyeglasses, Smiling], the facial image X i whose label is Y i = (1, 0, 1) should depict a smiling face with bangs and no eyeglasses.

As shown in FIG0 , DNA-GAN is mainly composed of three parts: an encoder (Enc), a decoder (Dec) and a discriminator (D).

The encoder maps the real-world images A and B into two latent disentangled representations DISPLAYFORM0 where [a 1 , . . .

, a i , . . .

, a n ] is called the attribute-relevant part, and z a is called the attribute-irrelevant part.

a i is supposed to be a DNA piece that controls y i , the i-th attribute in the label, and z a is for keeping other silent factors which do not appear in the attribute list as well as image identity information.

The same thing applies for Enc(B).

We focus on one attribute each time in our framework.

Let's say we are at i-th attribute.

A and B are required to have different labels, i.e. (y DISPLAYFORM1 , respectively.

In our convention, A is always for the dominant pattern, while B is for the recessive pattern.

We copy Enc(A) directly as the latent representation of A 1 , and annihilate b i in the copy of Enc(B) as the latent representation of B 1 .

The annihilating operation means replacing all elements with zeros, and plays a key role in disentangling the attribute, which we will discuss in detail in Section 3.3.

By swapping a i and 0 i , we obtain two new latent representations [a 1 , . . .

, 0 i , . . .

, a n , z a ] and [b 1 , . . .

, a i , . . . , b n , z b ] that are supposed to be decoded into A 2 and B 2 , respectively.

Though a decoder Dec, we can get four newly generated images A 1 , B 1 , A 2 and B 2 .

DISPLAYFORM2 Out of these four children, A 1 and B 1 are reconstructions of A and B, while A 2 and B 2 are novel crossbreeds.

The reconstruction losses between A and A 1 , B and B 1 ensure the quality of reconstructed samples.

Besides, using an adversarial discriminator D that helps make generated samples A 2 indistinguishable from B, and B 2 indistinguishable from A, we can enforce attribute-related information to be encoded in a i .

n ) which are different at the i-th position, the data flow can be summarized by (1) and (2).

We force the i-th latent encoding of B to be zero in order to prevent from trivial solutions as we will discuss in Section 3.3.The encoder and decoder receive two types of losses: (1) the reconstruction loss, DISPLAYFORM0 which measures the reconstruction quality after a sequence of encoding and decoding; (2) the standard GAN loss, DISPLAYFORM1 which measures how realistic the generated images are.

The discriminator takes the generated image and the i-th element of its label as inputs, and outputs a number which indicates how realistic the input image is.

The larger the number is, the more realistic the image is.

Omitting the coefficient, the loss function for the encoder and decoder is DISPLAYFORM2 The discriminator D receives the standard GAN discriminator loss DISPLAYFORM3 DISPLAYFORM4 where L D1 drives D to tell A from B 2 , and L D0 drives D to tell B from A 2 .

Through experiments, we observe that there exist trivial solutions to our model without the annihilating operation.

We just take the single-attribute case as an example.

Suppose that Enc(A) = [a, z a ] and Enc(B) = [b, z b ], we can get four children without annihilating operation DISPLAYFORM0 The reconstruction loss makes it invertible between the latent encoding space and image space.

The adversarial discriminator D is supposed to disentangle the attribute from other information by telling whether A 2 looks as real as B and B 2 looks as real as A or not.

As we know that the generative adversarial networks give the best solution when achieving the Nash equilibrium.

But without the annihilating operation, information of the whole image could be encoded into the attribute-relevant part, which means DISPLAYFORM1 Therefore, we obtain the following four children DISPLAYFORM2 In this situation, the discriminator D cannot discriminate A 2 from B, since they share the same latent encodings.

By reconstruction loss, A 2 and B are exactly the same image, which is against our expectation that A 2 should depict the person from A with the attribute borrowed from B. The same thing happens to B 2 and A as well.

To prevent from learning trivial solutions, we adopt the annihilating operation by replacing the recessive pattern b with a zero tensor of the same size 1 .

If information of the whole image were encoded into the attribute-relevant part, the four children in this case are DISPLAYFORM3 The encodings of B 1 and A 2 contain no information at all, thus neither the person in B 1 nor A 2 who is supposed to be the same as in B can be reconstructed by Dec.

This forces the attribute-irrelevant part to encode some information of images.

To reduce the difficulty of disentangling multiple attributes, we take the strategy of iterative training: we update our model using a pair of images with opposite labels at a certain position each time.

Suppose that we are at the i-th position, the label of image A is (y Compared with training with random pairs of images, iterative training is proved to be more effective.

Random pairs of images means randomly selecting pairs of images each time without label constraints.

A pair of images with different labels is called a useful pair.

We theoretically show that our iterative training is much more efficient than random image pairs especially when the dataset is unbalanced.

All proofs can be found in the Appendix.

DISPLAYFORM0 } denote the whole multi-attribute image dataset, where X i is a multi-attribute image and its label Y i = (y i 1 , . . .

, y i n ) is an n-dimensional vector.

There are totally 2 n kinds of labels, denoted by L = {l 1 , . . .

, l 2 n }.

The number of images with label l i is m i , and DISPLAYFORM1 To select all useful pairs at least once, the expected numbers of iterations needed for randomly selecting pairs and for iterative training are denoted by E 1 and E 2 respectively.

Then, DISPLAYFORM2 ..,n i∈Is,j∈Js DISPLAYFORM3 where I s represents the indices of labels where the s-th element is 1, and J s represents the indices of labels where the s-th element is 0.

Definition 1. (Balancedness) Define the balancedness of a dataset X described above with respect to the s-th attribute as follows: DISPLAYFORM4 where I s represents the indices of labels where the s-th element is 1, and J s represents the indices of labels where the s-th element is 0.

Theorem 2.

We have E 2 ≤ E 1 , when DISPLAYFORM5 Specifically, E 2 ≤ E 1 holds true for all n ≤ 2.The property of the function (ρ + 1) 2 /(2ρ) suits well with the definition of balancedness, because it attains the same value for ρ and 1/ρ, which is invariant to different labeling methods.

Its value gets larger as the dataset becomes more unbalanced.

The minimum is obtained at ρ = 1, which is the case of a balanced dataset.

Theorem 2 demonstrates that the iterative training mechanism is always more efficient than random pairs of images when the number of attributes met the criterion (16).

As the dataset becomes more unbalanced, (ρ s + 1) 2 /(2ρ s ) goes larger, which means (16) can be more easily satisfied.

More importantly, iterative training helps stabilize the training process on unbalanced datasets.

For example, given a two-attribute dataset, the number of data of each kind is as follows: If m 1 is a very large number, then it is highly likely that we will select a pair of images whose labels are (1, 0) and (1, 1) each time by randomly selecting pairs.

We ignore the pair of images DISPLAYFORM6 Figure 2: Manipulating illumination factors on the Multi-PIE dataset.

From left to right, the six images in a row are: original images A with light illumination and B with the dark illumination, newly generated images A 2 and B 2 by swapping the illumination-relevant piece in disentangled representations, and reconstructed images A 1 and B 1 .whose labels are (1, 0) and (1, 0) or (1, 1) and (1, 1), though these two cases have equal probabilities of being chosen.

Because they are not useful pairs, thus do not participated in training.

In this case, most of the time the model is trained with respect to the second attribute, which will cause the final learnt model less effective to the first attribute.

However, iterative training can prevent this from happening, since we update our model evenly with respect to two attributes.

In this section, we perform different kinds of experiments on two real-world datasets to validate the effectiveness of our methods.

We use the RMSProp BID20 optimization method initialized by a learning rate of 5e-5 and momentum 0.

All neural networks are equipped with Batch Normalization BID7 after convolutions or deconvolutions.

We used Leaky Relu BID13 as the activation function in the encoder.

Besides, we adopt strategies mentioned in Wasserstein GAN BID0 for stable training.

More details will be available online.

We divide all images into training images and test images according to the ratio of 9:1.

All of the following results are from test images without cherry-picking.

The Multi-PIE BID5 face database contains over 750,000 images of 337 subjects captured under 15 view points and 19 illumination conditions.

We collecte all front faces images of different illuminations and align them based on 5-point landmarks on eyes, nose and mouth.

All aligned images are resized into 128 × 128 as inputs in our experiments.

We label the light illumination face images by 1 and the dark illumination face images by 0.

As shown in Figure 2 , the illumination on one face is successfully transferred into the other face without modifying any other information in the images.

This demonstrates that DNA-GAN can effectively disentangle the illumination factor from other factors in the latent space.

CelebA BID12 is a dataset composed of 202599 face images and 40 attribute binary vectors and 5 landmark locations.

We use the aligned and cropped version and scaled all images down to 64 × 64.

To better demonstrate the advantage of our method, we choose TD-GAN BID22 and IcGAN BID16 for comparisons.

As we mentioned before, TD-GAN requires the explicit id information in the label, thus cannot be applied to the CelebA dataset directly.

To overcome this limitation, we use some channels to encode the id information in its latent representations.

In our experiments, the id information is preserved when swapping the attribute information in the latent encodings.

We also compared the experimental results of IcGAN with ours in the celebA dataset.

The following results are obtained using the the official code and pre-trained celebA model provided by the author 2 .(a) TD-GAN (b) IcGAN Figure 3 : The experimental results of TD-GAN and IcGAN on CelebA dataset.

Three rows indicates the swapping attributes of Bangs, Eyeglasses and Smiling.

For each model, the four images in a row are: two original images, and two newly generated images by swapping the attributes.

The third image is generated by adding the attribute to the first one, and the fourth image is generated by removing the attribute from the second one.

As displayed in Figure 3a , modified TD-GAN encounters the problem of trivial solutions.

Without id information explicitly contained in the label, TD-GAN encodes the information of the whole image into the attribute-related part in the latent representations.

As a result, two faces are swapped directly.

Whereas in Figure 3b , the quality of images generated by IcGAN are very bad, which is probably due to the multi-stage training process of IcGAN.

Since the overall effect of the model relies much on the each stage.

DNA-GAN is able to disentangle multiple attributes in the latent representations as shown in FIG3 .

Since different attributes are encoded in different DNA pieces in our latent representations, we are able to interpolate the attribute subspaces by linear combination of disentangled encodings.

Figure 4a, 4b and 4c present disentangled attribute subspaces spanned by any two attributes of Bangs, Eyeglasses and Smiling.

They demonstrate that our model is effective in learning disentangled representations.

FIG3 shows the hairstyle transfer process among different Bangs styles.

It is worth mentioning that the top-left image in FIG3 is outside the CelebA dataset, which further validate the generalization potential of our model on unseen data.

Please refer to FIG5 in the Appendix for more results.

FIG3 shows the attribute subspaces spanned by several Bangs feature vectors.

Besides, the top-left image in FIG3 is outside the CelebA dataset.

In this paper, we propose a supervised algorithm called DNA-GAN that can learn disentangled representations from multi-attribute images.

The latent representations of images are DNA-like, consisting of attribute-relevant and attribute-irrelevant parts.

By the annihilating operation and attribute hybridization, we are able to create new latent representations which could be decoded into novel images with designed attributes.

The iterative training strategy effectively overcomes the difficulty of training on unbalanced datasets and helps disentangle multiple attributes in the latent space.

The experimental results not only demonstrate that DNA-GAN is effective in learning disentangled representations and image editing, but also point out its potential in interpretable deep learning, image understanding and transfer learning.

There also exist some limitations of our model.

Without strong guidance on the attribute-irrelevant parts, some background information is encoded into the attribute-relevant part.

As we can see in FIG3 , the background color gets changed when swapping attributes.

Besides, our model may fail when several attributes are highly correlated with each other.

For example, Male and Mustache are statistically dependent, which are hard to disentangle in the latent representations.

These are left as our future work.

To prove Theorem 1, we need the following lemma.

Lemma 1.

A set S = {s 1 , . . .

, s m } has m different elements, from which elements are being selected equally likely with replacement.

The expected number of trials needed to collect a subset R = {s 1 , . . .

, s n } of n(1 ≤ n ≤ m) elements is m · 1 1 + 1 2 + · · · + 1 n .Proof.

Let T be the time to collect all n elements in the subset R, and let t i be the time to collect the i-th new elements after i − 1 elements in R have been collected.

Observe that the probability of collecting a new element is p i = (n − (i − 1))/m.

Therefore, t i is a geometrically distributed random variable with expectation 1/p i .

By the linearity of expectations, we have: DISPLAYFORM0

We first consider the case of randomly selecting pairs.

All possible image pairs are actually in the product space X × X , whose cardinality is m 2 .

If we take the order of two images in a pair into consideration, the number of possible pairs is m 2 .

Recall that the useful pair denotes a pair of image of different labels.

Therefore, the number of all useful pairs is i =j m i m j .

By Lemma 1, the expected number of iterations for randomly selecting pairs to select all useful pairs at least once is Now we consider the case of iterative training.

We always select a pair of images of different labels each time.

Suppose we are selecting images with opposite labels at the s-th position.

Let I s denote the indices of all labels with the s-th element 1, and J s denote the indices of all labels with the s-th element 0, where |I s | = |J s | = 2 n−1 .

Then we consider the subproblem by neglecting the first position in data labels, the number of all possible pairs is 2 i∈Is,j∈Js m i m j (regarding of order),

@highlight

We proposed a supervised algorithm, DNA-GAN, to disentangle multiple attributes of images.

@highlight

This paper investigates the problem of attribute-conditioned image generation using generative adversarial networks, and proposes to generate images from attribute and latent code as high-level representation.

@highlight

This paper proposed a new method to disentangle different attributes of images using a novel DNA structure GAN