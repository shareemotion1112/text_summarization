Disentangling underlying generative factors of a data distribution is important for interpretability and generalizable representations.

In this paper,  we introduce two novel disentangling methods.

Our first method, Unlabeled Disentangling GAN (UD-GAN, unsupervised), decomposes the latent noise by generating similar/dissimilar image pairs and it learns a distance metric on these pairs with siamese networks and a contrastive loss.

This pairwise approach provides consistent representations for similar data points.

Our second method (UD-GAN-G, weakly supervised) modifies the UD-GAN with user-defined guidance functions, which restrict the information that goes into the siamese networks.

This constraint helps UD-GAN-G to focus on the desired semantic variations in the data.

We  show  that  both  our  methods  outperform  existing  unsupervised approaches in quantitative metrics that measure semantic accuracy of the learned representations.

In addition, we illustrate that simple guidance functions we use in UD-GAN-G allow us to directly capture the desired variations in the data.

Generative Adversarial Networks (GANs) are generative model estimators, where two neural networks (generator and discriminator) are trained in an adversarial setting, so that likelihood-based probabilistic modeling is not necessary.

This works particularly well for sampling from a complex probability distribution, such as images.

Although GANs yield realistic looking images BID18 , the original formulation in only allows for randomly sampling from the data distribution without disentangled structural or semantic control over the generated data points.

One way to disentangle the generation process is to use conditional GANs BID16 BID17 .

These models modify the generator by conditioning it with supervised labels.

Then, they either take the same labels as input in the discriminator BID16 and measure the image-label compatibility, or classify the correct label at the output, given the generated image BID17 .

Conditional GANs rely on a dataset with labels, which might not always be available or might be time-consuming to collect.

In this paper, we propose two GAN-based methods that learns disentangled representations without using labeled data.

Our first method, Unlabeled Disentangling GAN (UD-GAN), generates image pairs, then embeds them with Siamese Networks BID2 , and finally learns a distance metric on a disentangled representation space.

Whereas our second method, UD-GAN-G, uses guidance functions to restrict the input to our siamese networks, so that they capture desired semantic variations.

There have been many studies on learning disentangled representations in generative models, which can be grouped into the level of supervision/labeled data they require.

In BID28 BID26 , the identity and the viewpoint of an object are disentangled via reconstructing the same object from a different viewpoint and minimizing a reconstruction loss.

Whereas in BID15 , the style and category of an object is separated via autoencoders, where an encoder embeds the style of an input image to a latent representation, and a decoder takes the category and style input to reconstruct the input image.

In BID24 , autoencoders and GANs are combined to decompose identity and attribute of an object, where the disentangled representation is obtained at the encoder outputs, and image labels are used at the output of the discriminator.

Disentangled representations (semi-supervised).

In BID20 , they clamp the hidden units for a pair of images with the same identity but with different pose or expression to have the same identity representation.

Whereas in BID11 , synthesized images are used to disentangle pose, light, and shape of an object by passing a batch of images where only one attribute varies and the rest of the representation is clamped to be the same.

These techniques only require a batch of samples with one attribute different at a time.

Disentangled representations (unsupervised).

InfoGAN BID1 is an unsupervised technique that discovers categorical and continuous factors by maximizing the mutual information between a GAN's noise variables and the generated image.

β-VAE BID7 and DIP-VAE BID12 are unsupervised autoencoder-based techniques that disentangle different factors in the latent representation of an encoded image.

In β-VAE, the KL-divergence between the latent and a prior distribution is weighted with a factor β > 1 to encourage disentanglement in the posterior latent distributions.

Wheres in DIP-VAE, the covariance matrix of the latent distribution is encouraged to be an identity matrix, thus leading to uncorrelated latent representations.

For all of the unsupervised methods, after a model is trained, a human needs to investigate which factors map to which semantic property.

In addition, as the methods are unsupervised, not all desirable factors might be represented.

In contrast, our method builds on existing approaches with two important modifications: (i) We operate on pairs of similar/dissimilar image pairs. (ii) We compute the image embeddings using separate networks, which allows us to guide the disentangling process with information restriction.

In GANs, the generator, G(.), maps a latent variable z, which has an easy-to-sample distribution, into a more complex and unknown distribution, such as images.

On the other hand, the discriminator D(.) tries to distinguish real images from the ones that are generated by G. In , the training is performed as a minimax game as follows: DISPLAYFORM0 where P R and P Z are the probability distributions of real images and the latent variable z, respectively.

We train our GAN by using the loss in equation 1.

In order to increase stability, we modify the generator loss by maximzing log(D(G(z))), instead of minimizing the second term in equation 1.

In a standard GAN setting, all of the variation in the distribution of real images is captured by the latent variable z. However, a single dimension or a slice of z does not necessarily have a semantic meaning.

In this paper, our target is to slice the latent variable into multiple vectors, where each vector controls a different semantic variation.

Our network architecture is visualized in Figure 1 .

In our method, the latent vector DISPLAYFORM0 , which represent different attributes we aim to disentangle.

One can add a final variable that captures the variation (and the noise) that is not picked up by the knobs.

In our experiments, this additional variable did not have a notable effect.

In our notation, q i refers to all of the knobs, except q i .

In order to train our model, first, for each q i , we sample two different vectors, q The image pairs that are generated with the same q i vectors, {x 11 , x 12 } or {x 21 , x 22 }, should have the same i th attribute, regardless of the values of q i .

We can ensure this via embedding the generated image pairs into a representation space with Siamese Networks BID2 , which are denoted as φ i (.), and then learning a distance metric on the embedding vectors by employing Contrastive Loss BID6 ).

An optional guidance function is used to restrict the information that goes into a siamese network, thus letting us approximate a desired representation space.

The guidance is disabled for our unsupervised UD-GAN approach.

Whereas for UD-GAN-G, the guidance is a simple, user-defined function, which is discussed in Section 3.3.We use a Contrastive Loss function to pull similar image pairs together, and push dissimilar pairs apart as follows: DISPLAYFORM1 where, L φ i is the Contrastive Loss for the i th Siamese Network φ i (.), the function DISPLAYFORM2 ) 2 is a shorthand for embedding distance between x ni1 and x ni2 , and γ DISPLAYFORM3 is an adaptive margin of the form γ DISPLAYFORM4 .

Using an adaptive margin makes the distance between two latent samples semantically meaningful and we empirically found that it improves the training stability.

The discriminator network D is not modified and is trained to separate real and generated image distributions.

BID3 use a similar latent variable slicing for capturing illumination and pose variations of a face with a fixed identity.

Their discriminator needs image pairs, which must be labeled for real images, to judge the quality and identity of the faces.

Our method does not require any labels for the real images.

Instead, we create similar and dissimilar image pairs via concatenating latent variables and generating image batches.

Our final loss function is: DISPLAYFORM5 where, L W GAN is the GAN loss described in equation 1, λ φ i is the weight of the embedding loss, and the sampling of the latent variables depends on i and is performed as described above.

A guidance function reduces the information content that flows into a siamese network and causes the corresponding embedding space to capture only the variations in the restricted input.

For example, consider we want to capture the hair-related attributes in the CelebA dataset BID14 , which contains aligned images of human faces.

By cropping every region but the top part of a generated image, we are able to guide φ top (.) to learn only the variations in the "Hair Color" as shown in the first row of FIG1 .

Note that, the knob q top (that corresponds to φ top (.)) changes the hair color not only at the cropped part of the image but as a whole.

This is due to the interplay between the adversarial part of our loss (see equation 3), which enforces global realism in images, and the contrastive loss, which administers disentangled representations.

As shown in FIG1 , different guidance functions leads to capturing different variations in the CelebA dataset.

Crop Guidance

We can gain a probabilistic interpretation of our method on a toy example.

Let us assume a problem, where we want to generate images of colored polygons (see Figure 1 ), where there are two independent factors of variation: shape and color, which we want to capture using two knobs q i and q j , respectively.

When we set q j to a certain value and vary q i , we want to generate polygons with the same color, but different shapes, and vice versa.

Let P be the probability distribution of colored polygons.

For each attribute, P can be decomposed into a mixture distribution as follows: DISPLAYFORM0 is a mixture component and π DISPLAYFORM1 is its corresponding probability of choosing it, and N i is the number of different values an attribute (in our example, i corresponds to shape) can take.

A similar explanation can be made for attribute j, i.e. color.

For the sake of this analysis, we accept that for each attribute, P can be decomposed into different discrete mixture distributions as shown in Figure 3 .

For this specific case, Q(1) i and Q (2) i are the distributions of colored squares and colored diamonds, respectively.

For the color attribute, which is indexed by j, each Q (k) j corresponds to a distribution of polygons with a single color (i.e., green polygons).Our contrastive loss in equation 2 has two terms.

The first term is minimizing the spread of each mixture component Q (k) i .

This spread is inversely related to disentanglement.

If all samples from DISPLAYFORM2 Figure 3: Illustration of the embedding spaces and separated probability distributions after training our model.

DISPLAYFORM3 are mapped to the same embedding vector, the effect of j (and any other attribute) on the representation φ i (.) disappears and disentangling is achieved.

During training, we stochastically go through all embedding spaces and minimize their spread, thus resulting in a disentangled representation in TAB6 in Appendix G.The second term in equation 2 separates all Q (k) i from each other using an adaptive margin γ (1,2) i .

This margin depends on the difference between input latent pairs, so that the resulting embedding space is smooth.

In other words, we separate rectangles, circles, and ovals from each other, but circles should be closer to ovals than squares, due to their relative similarity.

In the following, we focus on the shape attribute that is represented by i, however, derivations carry over to the color attribute j.

In order to separate the probability distributions over image embeddings, one can maximize a divergence between all pairs from Q (k) i .

One way to measure the distance between these distributions is to use the unbiased estimator of the energy distance BID23 : DISPLAYFORM4 The energy distance in equation 5 can be interpreted as an instance of Maximum Mean Discrepancy BID0 and resembles the Contrastive Loss BID6 .

We can rewrite equation 5 using the Contrastive Loss in equation 2 as follows: DISPLAYFORM5 Each element in the second sum is quadratic function and has its minimum at DISPLAYFORM6 /2 and the value of the minimum is γ (1,2) i 2 /2.

So, we can rewrite equation 6 as follows: DISPLAYFORM7 Therefore, as the margin γ (1,2) i depends only on the input latent variables and is not trainable, minimizing our embedding loss L φ i maximizes the lower bound for the energy distance D E .

This corresponds to learning a Siamese Network φ i (.) that separates two probability distributions Q

We perform our experiments on a server with Intel Xeon Gold 6134 CPU, 256GB system memory, and an NVIDIA V100 GPU with 16GB of graphics memory.

Our generator and discriminator architectures are outlined in our Appendix A. Each knob is a 1-dimensional slice of the latent variable and is sampled from Unif (−1, 1) .

We use ADAM BID9 as an optimizer for our training with the following parameters: learning rate=0.0002 and β 1 = 0.5.

We will release our code after the review process.

Datasets.

We evaluate our method on two image datasets: (i) the CelebA dataset BID14 , which consists of over 200,000 images of aligned faces.

We cropped the images to 64 × 64 pixels in size. (ii) the 2D Shapes BID7 , which is a dataset that is synthetically created with different properties, such as shape, scale, orientation, and x-y locations.

Both datasets are divided into training and test sets with a 90%-10% ratio.

The weight values for the contrastive loss is λ φ = 1 for the CelebA dataset and λ φ = 5 for the 2D shapes dataset.

We use a 32 and 10-dimensional latent variables for the CelebA and the 2D Shapes datasets, respectively.

Baselines.

We have two versions of our algorithm.

UD-GAN refers to the results that are obtained without any guidance at the input of our siamese networks, whereas UD-GAN-G represents a guided training.

We compare our method against β-VAE (Higgins et al., 2017), DIP-VAE BID12 , and InfoGAN BID1 to compare against both autoencoder and GAN-based approaches.

We get the quantitative and visual results for β-VAE and DIP-VAE from BID7 and BID12 , and use our own implementation of InfoGAN for training and testing.

The same generator/discriminator architecture is used for InfoGAN and our method.

Guidance.

For the CelebA dataset, the first 28 of 32 latent knobs are unguided and therefore are processed by the same siamese network that outputs a 28-dimensional embedding vector 1 .

Whereas the remaining four knobs correspond to four siamese networks (φ top , φ miu , φ mil , φ bot ) that are guided with cropped images in FIG1 .

For the 2D shapes dataset, we have 10 knobs, where the first 7 dimensions are unguided.

In order to guide the remaining three networks, we estimate the center of mass (M x ,M y ) and the sizeŜ of the generated object and feed them to our siamese networks, φ X (M x ), φ Y (M y ), and φ S (Ŝ).

More information for this computation can be found in Appendix D.

Disentanglement Metric.

This metric was proposed by BID7 and measures whether learned disentangled representations can capture separate semantic variations in a dataset.

In β-VAE and DIP-VAE, this representation is the output of the encoder, i.e., the inferred latent variable.

For InfoGAN, we use the representation learned by the discriminator.

In our method, we use the concatenated outputs of our siamese networks, which we denote as φ(.).The disentanglement metric scores for different methods are illustrated in TAB0 .

Here, we can see that both of our methods outperforms the baseline on the CelebA dataset.

All of the baseline approaches relate the latent variables to generated images on per-image basis.

Whereas our approach attempts to relate similarities/differences of latent variable pairs to image pairs, which provides a discriminative image embedding, where each dimension is invariant to unwanted factors BID6 .

For both datasets, our guided network (UD-GAN-G) performs better than our unguided approach, especially on the CelebA dataset.

This might be due to the correlations between irrelevant attributes.

For example the correlation coefficient between "Wearing Lipstick" and "Wavy Hair" attributes is 0.36, although they are not necessarily dependent.

One of our guided networks receive the cropped image around the mouth of a person, which prevents cluttering it with hairstyle.

Therefore, this guidance provides better disentanglement and results in an improved score as shown in TAB0 .

Due to containing simple synthetic images, our disentanglement scores for the 2D shapes dataset are very high.

The reason we get 100.0 score on our guided method is because of the guidances we choose, which are highly correlated with the ground truth labels, as shown in TAB4 in Appendix D. TAB1 , we compare our method against baseline approaches on CelebA attribute classification accuracy using the aforementioned projection vector.

Similar to the results in TAB0 , our guided approach slightly outperforms our unguided method and the other completely unsupervised techniques.

This is because some attributes in the CelebA dataset can be spatially isolated via cropping, which leads to a better classification performance.

For example, the attributes that are related to hair (Black Hair, Blond Hair, Wavy Hair) and mouth (Mouth Slightly Open, Wearing Lipstick) are captured better by the guided approach, because our top and bottom crops (see FIG1 ) are detaching the effects of other variations and are making attributes less correlated.

The accuracy on the attribute "Bangs" is worse on the guided approach.

This might be due to heuristic cropping we perform that divides the relevant image region into two slits.

Table 3 , we illustrate images generated by different methods on the CelebA dataset.

Each of the three rows capture the change in a semantic property: smile, azimuth, and hair color, respectively.

Within each image group, a latent dimension is varied (from top to bottom) to visualize the semantic change in that property.

Compared to adversarial methods, such as InfoGAN and UD-GAN-G, the DIP-VAE method generates blurrier images, due to the data likelihood term in VAE-based approaches, which is usually implemented as a pixel-wise image reconstruction loss.

In GAN-based approaches, this is handled via a learnable discriminator in an adversarial setting.

In TAB0 and 2, we quantitatively show the advantage of using our guided approach.

Another advantage is to have better control over the captured attributes.

For example, in all unsupervised approaches (including UD-GAN), we need to check which latent dimension represents corresponds to which visual attribute.

In some cases, a semantic attribute might not be captured due to the correlated nature of a dataset.

Whereas, in UD-GAN-G, we directly obtain the variations in smile, azimuth, and hair color through cropping the bottom, middle, and top part of our images, respectively.

Thanks to our guidance in FIG1 , we can directly manipulate these three attributes using the knobs q bot , q mil , and q top as shown in Table 3 .The same trend is true for the 2D Shapes dataset results in Table 4 .

Although the X and Y positions and the scale of the synthetic object is captured by both our unsupervised and guided approaches, the guidance we choose directly captures the desired feature on in advance chosen knobs q X , q Y , and q S , respectively.

Table 3 : Images generated by varying a latent dimension, which corresponds to a semantic property.

Table 4 : Generated images for the 2D Shapes dataset by varying a latent dimension, which corresponds to a semantic property (first row: UD-GAN, second row: UD-GAN-G).

DISPLAYFORM0 DISPLAYFORM1 In completely unsupervised approaches, there is no guarantee to capture all of the desired semantic variations.

The main premise behind UD-GAN-G is to find very simple, yet effective ways to capture some of the variation in the data.

This weak supervision helps us to obtain proxies to certain semantic properties, so that we get the desired features without training the model multiple times with different hyperparameters or initializations.

In the aligned the CelebA dataset, each face is roughly centered around the nose.

This reduces the variation and simplifies the problem of guidance design, as we show in FIG1 .

In more complex scenarios, where the objects can appear in a large variety of scales, translations, and viewpoints, one can use a pre-trained object detection and localization method, such as YOLO BID19 , as a guidance network.

This enables us to use the knowledge obtained from a labeled dataset, such as ImageNet BID21 to disentangle a new unlabeled dataset.

Note that backpropagating the gradients of a deep network into an image might cause adversarial samples BID22 .

However, the discriminator can alleviate this by rejecting problematic images.

In order to backpropagate the gradients from the siamese networks to the generator, the guidance function we use needs to be differentiable.

This might pose a limitation to our method; however, differentiable relaxations can instead be used to guide our network.

For example, one can employ differentiable relaxation of the superpixel segmentation in BID8 ) to disentangle a low-level image segmentation.

Our latent variables are sampled from a uniform distribution.

In addition, image similarity is measured by using L2-distance between a pair of image embeddings.

We experimented with modeling some latent dimensions as categorical variables.

However, we encountered training stability issues, due to computing the softmax loss between two learnable categorical image embeddings, instead of one embedding and one fixed label vector as it is usually done.

We plan to tackle that problem in our future work.

In this paper we introduced UD-GAN and UD-GAN-G, novel GAN formulations which employ Siamese networks with contrastive losses in order to make slices of the latent noise space disentangled and more semantically meaningful.

Our experiments encompassed guided and unguided approaches for the embedding networks, and illustrated how our methods can be used for semantically meaningful image manipulation.

Our qualitative and quantiative results confirm that our method can adjust well to the intrinsic factors of variation of the data and outperform the current state-of-the-art methods on the CelebA and 2D Shapes datasets.

In future work, we plan to investigate more powerful forms of embedders, e.g. extracting information from pre-trained networks for semantic segmentation and landmark detection.

This allows for even more powerful novel image manipulation techniques.

In TAB2 , we show the neural network layers we use in our generator for different datasets.

Our discriminator and siamese network architectures are the inverted version of our generator.

Each fully connected and Conv2D layer is followed by a Leaky ReLU non-linearity, except the last layer.

DISPLAYFORM0

The Siamese Networks φ i are desired to map images into embedding spaces, where they can be grouped within a distinct semantic context.

For the example shown in FIG3 , where we disentangle the shape and the color, this might not be directly achievable in a completely unsupervised setting, because the separation in equation 4 is not unique.

However, we can still benefit from the disentangling capability of our method via small assumptions and domain knowledge, without collecting labeled data.

Consider the toy example, where we extend the MNIST dataset BID13 to have a random color, sampled from a uniform RGB color distribution.

We define our problem to independently capture the shape of a digit with q 1 and its color with q 2 .In FIG3 (a), we show images created by a generator, which is trained along with two networks, φ 1 and φ 2 , without any guidance in an unsupervised setting.

We can see that the knobs, q 1 and q 2 , capture the variations in the data, however, these variations are coupled with multiple semantic properties.

Each knob modifies a complicated combination of shape and color.

However, if we design a network architecture in a slightly smarter way, we should be able to separate the shape and the color attributes.

This is exemplified in FIG3 (b), where instead of feeding the whole image to φ 2 , we feed the average color of some randomly sampled pixels from a generated image.

This choice prevents φ 2 to capture the spatial structure of the generated digit and to focus only on color.

After the training our method with a modified φ 2 , the first network captures shape of a digit, and the second one captures the color variations.

This can also be observed in FIG3 (c) and 4(d), where we use t-SNE (van der BID25 to visualize embedding spaces for shape and color, respectively.

In order to show the effect of the guided siamese networks, we perform three experiments on the MS-Celeb dataset BID5 by using different guiding proxies.

In the first experiment, only one of the two networks is guided with an edge detector at the input.

Results of this experiment are shown in TAB3 .

We can see that the first knob, which is connected to edges, captures the overall outline and roughly controls the identity of the generated face.

On the other hand, the unguided second knob modifies the image with minimal changes to image edges.

This change, in this case, corresponds to the lighting of the face.

DISPLAYFORM0 We perform a second experiment with the edge detector, where in this case, the second knob is guided with the average color of the generated image.

In TAB3 , we can observe the results of our disentangled image manipulation.

The first knob with the edge detector again captures the outline of the face, and the second average color knob modifies a combination of the light and the skin color, similar to the results in TAB3 .In our third experiment, we employ the cropped guidance networks.

The two knobs receive the cropped top and bottom part of the image for training.

Although these image crops are not independent, we still get acceptable results that are shown in TAB3 .

Adjusting the first knob only modifies the upper part of the face; the hair and the eyes.

Similarly, the second knob is responsible for determining the chin and mouth shape.

In order to guide our siamese networks for the 2D shapes dataset, we estimate the center of mass of the generated image, and the size of the generated object as follows: DISPLAYFORM1 where, x is a generated image, x[c x , c y ] is the pixel intensity at image coordinates [c x , c y ], (M x ,M y ) are the coordinates of the center of mass of x, andŜ is the size estimate for the generated object.

As the 2D shapes dataset is relatively simple and contain only one object, these guidances are highly correlated with the ground truth attributes as shown in TAB4 .

FIG5 , we illustrate additional semantic properties that are captured by UD-GAN-G.

In TAB5 , we compare the classification perfromance of our method to InfoGAN on all attributes in the CelebA dataset.

G ATTRIBUTE CORRELATIONS.In TAB6 , we compare the correlation between different embedding (or latent) dimensions and the correlation between embedding dimensions and the CelebA attributes.

Although DIP-VAE encodes a more un-correlated representation, due to the correlated nature of CelebA attributes, it does not necessarily transfer to a disentangled semantic representation, as illustrated by the quantitative results in TAB0 and 2.

@highlight

We use Siamese Networks to guide and disentangle the generation process in GANs without labeled data.