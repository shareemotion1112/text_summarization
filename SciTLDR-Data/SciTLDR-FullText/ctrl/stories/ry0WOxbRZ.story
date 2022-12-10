Generative adversarial networks (GANs) are a powerful framework for generative tasks.

However, they are difficult to train and tend to miss modes of the true data generation process.

Although GANs can learn a rich representation of the covered modes of the data in their latent space, the framework misses an inverse mapping from data to this latent space.

We propose Invariant Encoding Generative Adversarial Networks (IVE-GANs), a novel GAN framework that introduces such a mapping for individual samples from the data by utilizing features in the data which are invariant to certain transformations.

Since the model maps individual samples to the latent space, it naturally encourages the generator to cover all modes.

We demonstrate the effectiveness of our approach in terms of generative performance and learning rich representations on several datasets including common benchmark image generation tasks.

Generative Adversarial Networks (GANs) BID4 emerged as a powerful framework for training generative models in the recent years.

GANs consist of two competing (adversarial) networks: a generative model that tries to capture the distribution of a given dataset to map from an arbitrary latent space (usually drawn from a multi-variate Gaussian) to new synthetic data points, and a discriminative model that tries to distinguish between samples from the generator and the true data.

Iterative training of both models ideally results in a discriminator capturing features from the true data that the generator does not synthesize, while the generator learns to include these features in the generative process, until real and synthesized data are no longer distinguishable.

Experiments by BID10 showed that a GAN can learn rich representation of the data in the latent space in which interpolations produce semantic variations and shifts in certain directions correspond to variations of specific features of the generated data.

However, due to the lack of an inverse mapping from data to the latent space, GANs cannot be used to encode individual data points in the latent space BID2 .

Moreover, although GANs show promising results in various tasks, such as the generation of realistic looking images BID9 BID0 BID11 or 3D objects BID13 , training a GAN in the aforementioned ideal way is difficult to set up and sensitive to hyper-parameter selection BID10 .

Additionally, GANs tend to restrict themselves on generating only a few major modes of the true data distribution, since such a so-called mode collapse is not penalized in the GAN objective, while resulting in more realistic samples from these modes BID1 .

Hence, the majority of the latent space only maps to a few regions in the target space resulting in poor representation of the true data.

We propose a novel GAN framework, Invariant-Encoding Generative Adversarial Networks (IVE-GAN), which extends the classical GAN architecture by an additional encoding unit E to map samples from the true data x to the latent space z (compare FIG0 ).

To encourage the encoder to learn a rich representation of the data in the latent space, the discriminator D is asked to distinguish between different predefined transformations T(x) of the input sample and generated samples G(E(x)) by taking the the original input as condition into account.

While the discriminator has to learn what the different variations have in common with the original input, the encoder is forced to encode the necessary information in the latent space so that the generator can fool the discriminator by generating samples which are similar to the original samples.

Since the discriminator is invariant to the predefined transformations, the encoder can ignore these variations in the input space and learn a rich and to such transformations invariant representation of the data.

The variations of the generated samples are modeled by an additional latent vector z (drawn from a multi-variate Gaussian).

Thus, the encoded samples condition the generator G(z , E(x)).

Moreover, since the discriminator learns to distinguish between generated samples and variations of the original for each individual sample x in the data, the latent space can not collapse to a few modes of the true data distribution since it will be easy for the discriminator to distinguish generated samples from original ones if the mode is not covered by the generator.

Thus, the proposed IVE-GAN learns a rich and to certain transformations invariant representation of a dataset and naturally encourages the generator to cover all modes of the data distribution.

To generate novel samples, the generator G(z) can be fed with an arbitrary latent representation z ∼ P noise .In summary, we make the following contributions:• We derive a novel GAN framework for learning rich and transformation invariant representation of the data in the latent space.• We show that our GANs reproduce samples from a data distribution without mode collapsing issues.• We demonstrate robust GAN training across various data sets and showcase that our GAN produces very realistic samples.

Generative Adversarial Networks BID4 (GANs) are a framework for training generative models.

It is based on a min-max-game of two competing (adversarial) networks.

The generator G tries to map an arbitrary latent space z ∼ P Z (z) (usually drawn from a multi-variate Gaussian) to new synthetic data points by training a generator distribution P G (x) that matches the true data distribution P data (x).

The training is performed by letting the generator compete against the second network, the discriminator D. The discriminator aims at distinguishing between samples from the generator distribution P G (x) and real data points from P data (x) by assigning a probability DISPLAYFORM0 The formal definition of the objective of this min-max-game is given by: DISPLAYFORM1 However, training a GAN on this objective usually results in a generator distribution P G (x) where large volumes of probability mass collapse onto a few major modes of the true data generation distribution P data (x) BID1 .

This issue, often called mode collapsing, has been subject of several recent publications proposing new adjusted objectives to reward a model for higher variety in data generation.

A straightforward approach to control the generated modes of a GAN is to condition it with additional information.

Conditional Generative Adversarial Nets BID8 utilize additional information such as class-labels to direct the data generation process.

The conditioning is done by additionally feeding the information y into both generator and discriminator.

The objective function Eq. (1) becomes: DISPLAYFORM2 Obviously, such a conditioning is only possible if additional information y is provided.

Che et al. FORMULA1 proposed a framework which adds two regularizers to the classical GAN.

The metric regularizer trains in addition to the generator G(z) : Z → X an encoder E(x) : X → Z and includes the objective DISPLAYFORM3 ] in the training.

This additional objective forces the generated modes closer to modes of the true data.

As distance measure d they proposed e.g. the pixel-wise L 2 distance or the distance of learned features by the discriminator BID3 .

To encourage the generator to also target minor modes in the proximity of major modes the objective is extended by a mode regulizer DISPLAYFORM4 Another proposed approach addressing the mode collapsing issue are unrolled GANs BID7 .

In practice GANs are trained by simultaneously updating V (D, G) in Eq. (1), since explicitly updating G for the optimal D for every step is computational infeasible.

This leads to an update for the generator which basically ignores the max-operation for the calculation of the gradients and ultimately encourages mode collapsing.

The idea behind unrolled GANs is to update the generator by backpropagating through the gradient updates of the discriminator for a fixed number of steps.

This leads to a better approximation of Eq. (1) and reduces mode collapsing.

Another way to avoid mode collapse is the Coulomb GAN, which models the GAN learning problem as a potential field with a unique, globally optimal nash equlibrium BID12 .Work has also been done aiming at introducing an inverse mapping from data x to latent space z. Bidirectional Generative Adversarial Networks (BiGANs) BID2 and Adversarially Learning Inference BID3 are two frameworks based on the same idea of extending the GAN framework by an additional encoder E and to train the discriminator on distinguishing joint samples (x, E(x)) from (G(z), z) in the data space (x versus G(z)) as well as in the latent space (E(x) versus z)).

Also the inverse mapping (E(G(z)) is never explicitly computed, the authors proved that in an ideal case the encoder learns to invert the generator almost everywhere (E = G −1 ) to fool the discriminator.

However, upon visual inspection of their reported results BID3 , it appears that the similarity of the original x to the reconstructions G(E(x) is rather vague, especially in the case of relative complex data such as CelebA (compare appendix A).

It seems that the encoder concentrates mostly on prominent features such as gender, age, hair color, but misses the more subtle traits of the face.

Consider a subset S of the domain D that is setwise invariant under a transformation T : D → D so that x ∈ S ⇒ T (x) ∈ S. We can utilize different elements x ∈ S to train a discriminator on learning the underlying concept of S by discriminating samples x ∈ S and samples x ∈ S. In an adversarial procedure we can then train a generator on producing samples x ∈ S. S could be e.g. a set of higher-level features that are invariant under certain transformation.

An example for a dataset with such features are natural images of faces.

High-level features like facial parts that are critical to classify and distinguish between different faces are invariant e.g. to small local shifts or rotations of the image.

We propose an approach to learn a mapping from the data to the latent space by utilizing such invariant features S of the data.

In contrast to the previous described related methods, we learn the mapping from the data to the latent space not by discriminating the representations E(x) and z but by discriminating generated samples conditioned on encoded original samples G(z , E(x) and transformations of an original sample T (x) by taking the original sample as additional information into account.

In order to fool the discriminator, the encoder has to extract enough features from the original sample so that the generator can reconstruct samples which are similar to the original one, apart from variations T .

The discriminator, on the other hand, has to learn which features samples T (x) have in common with the original sample x to discriminate variations from the original samples and generated samples.

To fool a perfect discriminator the encoder has to extract all individual features from the original sample so that the generator can produce perfect variants of the original.

To generate novel samples, we can draw samples z ∼ P Z as latent space.

To learn a smooth representation of the data, we also include such generated samples and train an additional discriminator D on discriminating them from true data as in the classical GAN objective Eq. (1).Hence, the objective for the IVE-GAN is defined as a min-max-game: DISPLAYFORM0 One thing to consider here is that by introducing an invariance in the discriminator with respect to transformations T , the generator is no longer trying to exactly match the true data generating distribution but rather the distribution of the transformed true data.

Thus, one has to carefully decide which transformation T to chose, ideally only such describing already present variations in the data and which are not affecting features in data that are of interest for the representation.

To evaluate the IVE-GAN with respect to quality of the generated samples and learned representations we perform experiments on 3 datasets: a synthetic dataset, the MNIST dataset and the CelebA dataset.

To evaluate how well a generative model can reproduce samples from a data distribution without missing modes, a synthetic dataset of known distribution is a good way to check if a the model suffers from mode collapsing BID7 .

Following BID7 , we evaluate our model on a synthetic dataset of a 2D mixture of eight Gaussians x ∼ N (µ, Σ) with covariance matrix Σ, and means µ k arranged on a ring.

As invariant transformation T (x) we define small shifts in both dimensions: DISPLAYFORM0 so that the discriminator becomes invariant to the exact position within the eight Gaussians.

FIG1 shows the distribution of the generated samples of the model over time compared to the true data generating distribution.

The IVE-GAN learns a generator distribution which converges to all modes of the true data generating distribution while distributing its probability mass equally over all modes.

As a next step to increase complexity we evaluate our model on the MNIST dataset.

As invariant transformations T (x) we define small random shifts (up to 4 pixels) in both width-and heightdimension and small random rotations up to 20• .

FIG2 shows novel generated samples from the IVE-GAN trained on the MNIST dataset as a result of randomly sampling the latent representation from a uniform distribution z ∼ P noise .

FIG3 shows for different samples x from the MNIST dataset the generated reconstructions G(z , E(x)) for a model with a 16-dimensional as well as a 3-dimensional latent space z. As one might expect, the model with higher capacity produces images of more similar style to the originals and makes less errors in reproducing digits of unusual style.

However, 3 dimensions still provide enough capacity for the IVE-GAN to store enough information of the original image to reproduce the right digit class in most cases.

Thus, the IVE-GAN is able to learn a rich representation of the MNIST dataset in 3 dimensions by utilizing class-invariant transformations.

Fig. 5 shows the learned representation of the MNIST dataset in 3 dimensions without using further dimensionality reduction methods.

We observe distinct clusters for the different digit classes.

As a last experiment we evaluate the proposed method on the more complex CelebA dataset BID5 , centrally cropped to 128×128 pixel.

As in the case of MNIST, we define invariant transformation T (x) as small random shifts (here up to 20 pixel) in both width-and height-dimension as well as random rotations up to 20• .

Additionally, T (x) performs random horizontal flips and small random variations in brightness, contrast and hue.

Fig. 6 shows for different images x from the CelebA dataset some of the random transformation T (x) and some of the generated reconstructed images G(z , E(x)) with random noise z .

The reconstructed images show clear similarity to the original images not only in prominent features but also subtle facial traits.

FIG5 shows novel generated samples from the IVE-GAN trained on the CelebA dataset as an result of randomly sampling the latent representation from a uniform distribution z ∼ P noise .

To illustrate the influence of the noise component z , the generation was performed with the same five noise components for each image respectively.

We observe that altering the noise component induces a similar relative transformation in each image.

To visualize the learned representation of the trained IVE-GAN we encode 10.000 samples from the CelebA dataset into the 1024-dimensional latent space and projected it into two dimensions using t-Distributed Stochastic Neighbor Embedding (t-SNE) BID6 .

FIG6 shows this projection of the latent space with example images for some high density regions.

Since the CelebA dataset comes with labels, we can evaluate the representation with respect to its ability to clusters images of same features.

FIG6 shows the t-SNE embedding of both the latent representation and the original images for a selection of features.

Observing the We also evaluate whether smooth interpolation in the learned feature space can be performed.

Since we have a method at hand that can map arbitrary images from the dataset into the latent space, we can also interpolate between arbitrary images of this dataset.

This is in contrast to BID10 , who could only show such interpolations between generated ones.

Fig. 9 shows generated images based on interpolation in the latent representation between two original images.

The intermediate images, generated from the interpolated latent representations are visually appealing and show a smooth transformation between the images.

This finding indicates that the IVE-GAN learns a smooth latent space and can generate new realistic images not only from latent representations of training samples.

With this work we proposed a novel GAN framework that includes a encoding unit that maps data to a latent representation by utilizing features in the data which are invariant to certain transformations.

We evaluate the proposed model on three different dataset and show the IVE-GAN can generate visually appealing images of high variance while learning a rich representation of the dataset also covering subtle features.

B NETWORK ARCHITECTURE AND HYPERPARAMETERS

<|TLDR|>

@highlight

A noval GAN framework that utilizes transformation-invariant features to learn rich representations and strong generators.

@highlight

Proposes a modified GAN objective consisting of a classic GAN term and an invariant encoding term.

@highlight

This paper presents the IVE-GAN, a model that introduces en encoder to the Generative Adversarial Network framework.