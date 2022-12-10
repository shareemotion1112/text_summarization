Learning disentangled representations of  data is one of the central themes in unsupervised learning in general and generative modelling in particular.

In this work,  we tackle a slightly more intricate scenario where the observations are generated from a conditional distribution of some known control variate and some latent noise variate.

To this end, we present a hierarchical model and a training method (CZ-GEM) that leverages some of the recent developments in likelihood-based and likelihood-free generative models.

We show that by formulation, CZ-GEM introduces the right inductive biases that ensure the disentanglement of the control from the noise variables, while also keeping the components of the control variate disentangled.

This is achieved without compromising on the quality of the generated samples.

Our approach is simple, general, and can be applied both in supervised and unsupervised settings.

Consider the following scenario: a hunter-gatherer walking in the African Savannah some 50,000 years ago notices a lioness sprinting out of the bush towards her.

In a split second, billions of photons reaching her retinas carrying an enormous amount of information: the shade of the lioness' fur, the angle of its tail, the appearance of every bush in her field of view, the mountains in the background and the clouds in the sky.

Yet at this point there is a very small number of attributes which are of importance: the type of the charging animal, its approximate velocity and its location.

The rest are just details.

The significance of the concept that the world, despite its complexity, can be described by a few explanatory factors of variation, while ignoring the small details, cannot be overestimated.

In machine learning there is a large body of work aiming to extract low-dimensional, interpretable representations of complex, often visual, data.

Interestingly, many of the works in this area are associated with developing generative models.

The intuition is that if a model can generate a good approximation of the data then it must have learned something about its underlying representation.

This representation can then be extracted either by directly inverting the generative process (Srivastava et al., 2019b) or by extracting intermediate representations of the model itself (Kingma & Welling, 2014; Higgins et al., 2017) .

Clearly, just learning a representation, even if it is low-dimensional, is not enough.

The reason is that while there could be many ways to compress the information captured in the data, allowing good enough approximations, there is no reason to a priori assume that such a representation is interpretable and disentangled in the sense that by manipulating certain dimensions of the representation one can control attributes of choice, say the pose of a face, while keeping other attributes unchanged.

The large body of work on learning disentangled representations tackles this problem in several settings; fully supervised, weakly supervised and unsupervised, depending on the available data (Tran et al., 2018; Reed et al., 2014; Jha et al., 2018; Mathieu et al., 2016; Higgins et al., 2017; Kim & Mnih, 2018; Nguyen-Phuoc et al., 2019; Narayanaswamy et al., 2017) .

Ideally, we would like to come up with an unsupervised generative model that can generate samples which approximate the data to a high level of accuracy while also giving rise to a disentangled and interpretable representation.

In the last decade two main approaches have captured most of the attention; Generative Adversarial Networks (GANs) and Variational Auto-Encoders (VAEs).

In their original versions, both GANs (Goodfellow et al., 2014) and VAEs (Kingma & Welling, 2014) were trained in an unsupervised manner and (a) Chair rotation generated by CGAN (b) Chair rotation generated by CZ-GEM Figure 1 : Changing the azimuth of chairs in CGAN and CZ-GEM while holding Z constant.

Unlike CZ-GEM, C and Z are clearly entangled in CGAN as changing C also changes the type of chair even though Z is held constant.

gave rise to entangled representations.

Over the years, many methods to improve the quality of the generated data as well as the disentanglement of the representations have been suggested (Brock et al., 2018; Kingma & Dhariwal, 2018; Nguyen-Phuoc et al., 2019; Jeon et al., 2018) .

By and large, GANs are better than VAEs in the quality of the generated data while VAEs learn better disentangled representations, in particular in the unsupervised setting.

In this paper, we present a framework for disentangling a small number of control variables from the rest of the latent space which accounts for all the additional details, while maintaining a high quality of the generated data.

We do that by combining VAE and GAN approaches thus enjoying the best of both worlds.

The framework is general and works in both the supervised and unsupervised settings.

Let us start with the supervised case.

We are provided with paired examples (x, c) where x is the observation and c is a control variate.

Crucially, there exists a one-to-many map from c to the space of observations, and there are other unobserved attributes z (or noise) that together completely define x. For instance, if x were an image of a single object, c controls the orientation of the object relative to the camera and z could represent object identity, texture or background.

Our goal is to learn a generative model p θ (x|c, z) that fulfills two criteria:

If we were learning models of images, we would like the generated images to look realistic and match the true conditional distribution p(x|c).

The posterior is factorized p(c, z|x; θ) = p(c|x; θ)p(z|x; θ): We would like the control variate to be disentangled from the noise.

For example, changing the orientation of the object should not change the identity under our model.

This problem setup can occur under many situations such as learning approximate models of simulators, 3D reconstructions, speaker recognition (from speech), and even real-world data processing in the human brain as in the hunter-gatherer example above.

We argue that a naive implementation of a graphical model as shown in Figure 2 (left), e.g. by a conditional GAN (Mirza & Osindero, 2014) , does not satisfy Criterion 2.

In this model, when we condition on x, due to d-separation, c and z could become dependent, unless additional constraints are posed on the model.

This effect is demonstrated in Figure 1 (a).

To overcome this we split the generative process into two stages by replacing C with a subgraph (C → Y ) as shown in Figure  2 (center).

First, we generate a crude approximation y of the data which only takes c into account.

The result is a blurry average of the data points conditioned on c, see Figure 2 (right).

We then feed this crude approximation into a GAN-based generative model which adds the rest of the details conditioned on z. We call this framework CZ-GEM.

The conditioning on z in the second stage must be done carefully to make sure that it does not get entangled with y. To that end we rely on architectural choices and normalization techniques from the style transfer literature (Huang & Belongie, 2017) 2 .

The result is a model which generates images of high quality while disentangling c and z as can be clearly seen in Figure 1(b) .

Additionally, in the unsupervised setting, when the labels c are not available, (C → Y ) can be realized by β-VAE, a regularized version of VAE which has been shown to learn a disentangled representation of its latent variables (Higgins et al., 2017; Burgess et al., 2018) .

In Section 3 we provide implementation details for both the supervised and unsupervised versions.

We summarize our two main contributions: Figure 2 : On the left, a conditional GAN (CGAN) model.

CZ-GEM in the middle replaces node C with a subgraph (C → Y ) that is trained independently of the rest of the model.

This subgraph learns to only partially render the observation.

As such, Z comes at a later stage of the rendering pipeline to add details to Y .

As an example, consider the rightmost graph where the observation is made up of different types of chairs in different poses.

Let the pose be controlled by C and the type (Identity) be explained by Z. Then in step one of CZ-GEM we learn the pose relationship between C and X via the subgraph, giving rise to a blurry chair in the correct pose.

Once the pose is learned, in the second step, the approximate rendering Y is transformed into X by allowing Z to add identity related details to the blurry image.

We break down the architecture to model an intermediate representation that lends itself to interpretability and disentanglement, and then (carefully) use a GAN based approach to add the rest of the details, thus enjoying a superior image generation quality compared to VAEs.

We show that our model can be combined easily with common methods for discovering disentangled representations such as β-VAE to extract c and treat them as labels to generate images that do not compromise on generative quality.

Generative adversarial networks (GAN) (Goodfellow et al., 2014) represent the current state of the art in likelihood-free generative modeling.

In GANs, a generator network G θ is trained to produce samples that can fool a discriminator network D ω that is in turn trained to distinguish samples from the true data distribution p(x) and the generated samples

Here, p z is usually a low dimensional easy-to-sample distribution like standard Gaussian.

A variety of tricks and techniques need to be employed to solve this min-max optimization problem.

For our models, we employ architectural constraints proposed by DC-GAN (Radford et al., 2015) that have been widely successful in ensuring training stability and improving generated image quality.

Conditional GANs (CGAN) (Mirza & Osindero, 2014) adapt the GAN framework for generating class conditional samples by jointly modeling the observations with their class labels.

In CGAN, the generator network G θ is fed class labels c to produce fake conditional samples and the discriminator D ω is trained to discriminate between the samples from the joint distribution of true conditional and true labels p(x|c)p(c) and the fake conditional and true labels p θ (x|c)p(c).

While not the main focus of this paper, we present a novel information theoretic perspective on CGANs.

Specifically, we show that CGAN is trained to maximize a lower-bound to the mutual information between the observation and its label while simultaneously minimizing an upper-bound to it.

We state this formally:

by training a discriminator D ω to approximate the log-ratio of the true and generated data densities i.e. D ω ≈ log p(x, c)/p θ (x, c) in turn minimizing the following

where I g,θ (x, c) is the generative mutual information and q(c|x, θ) is the posterior under the learned model.

The detailed derivation is provided in Appendix A.1.

Notice that at the limit, the model learns exactly the marginal distribution of x and the posterior q(c|x) and the KL terms vanish.

Variational autoencoders (VAE) represent a class of likelihood-based deep generative models that have recently been extensively studied and used in representation learning tasks (Higgins et al., 2017; Burgess et al., 2018; .

Consider a latent variable model where observation X is assumed to be generated from some underlying low-dimensional latent feature space Z. VAE models learn the conditional distribution p(x|z) using a deep neural network (parameterized by θ) called decoder network.

It uses another deep neural network (parameterized by φ), called encoder to model the posterior distribution p(z|x).

The encoder and decoder networks are trained using amortized variational inference (Kingma & Welling, 2014) to maximizes a variational lower-bound to the evidence likelihood (ELBO).

Recently, Higgins et al. (2017) showed that by regularizing the variational posterior approximation of p(z|x) to be close to the prior distribution p(z) in KLdivergence, the model is encouraged to learn disentangled representations.

I.e. the model learns a posterior distribution that is factorized over the dimensions.

They call their model β-VAE.

We note that information bottleneck based methods for disentangled representation learning, such as β-VAE, severely compromise the generative quality.

Batch-Normalization (BN) (Ioffe & Szegedy, 2015) plays a crucial role in ensuring the stability of GAN training Radford et al. (2015) .

However, as we discuss in Section 3, it is not suitable for our purposes.

Recently, it has been shown that Instance Normalization (IN) Ulyanov et al. (2016) and its variant Adaptive Instance Normalization (AdaIN) Huang & Belongie (2017) can be particularly useful for image generation and stylization.

IN normalizes each convolutional channel per training sample, while AdaIN modifies this normalization to be a function of an additional variable z (usually style in style transfer).

The final transformation applied by AdaIN is:

where µ(x) = 1 HW h,w x nhwc and σ(x) = 1 HW h,w (x nhwc − µ(x)) 2 + .

γ(z) and β(z) are learned functions of z that could be parameterized by a neural network, usually a fully connected layer.

In Section 1, we provided a high level description of our approach.

We will now provide a detailed description of how the two components of CZ-GEM, subgraph C → Y and the conditional generative models (Y, Z) → X are implemented and trained in practice.

Figure 3 provides an implementation schematic of our proposed framework.

If C is known a priori then learning the subgraph C → Y reduces to the regression problem that minimizes ||x c − y c || 2 .

In practice, since our observations are images, this subgraph is realized using a deep transposed-convolution based decoder network and is trained to learn the map between C and Y .

This is similar to the recent work of Srivastava et al. (2019b) .

We emphasize that this network is trained independently of the rest of the model.

to discover disentangled generative control factors.

In our implementation we use β-VAE (see Section 2.2).

One drawback of these information bottleneck based methods is that they compromise on the generative quality.

This is where the GAN likelihood-free approach in the second stage comes into play.

In fact, even if the output of the first stage (i.e. the intermediate image Y in Figure 2 ) is of very low generative quality, the final image is of high quality since the second stage explicitly adds details using a state-of-the-art GAN method.

In Section 5 we show how a simple VAE with a very narrow information bottleneck (2-6 dimensions) can be used within CZ-GEM to discover C in an unsupervised fashion without compromising on generation quality.

Vanilla GANs can only model the marginal data distribution i.e. they learn p θ to match p x and in doing so they use the input to the generator (G θ ) only as a source of stochasticity.

Therefore we start with a conditional GAN model instead, to preserve the correspondence between Y and X. As shown in section 2.1, this framework trains G θ such that the observation X is maximally explained by the conditioning variable Y .

One major deviation from the original model is that the conditioning variable in our case is the same type and dimensionality as the observation.

That is, it is an image, albeit a blurry one.

This setup has previously been used by Isola et al. (2017) in the context of image-to-image translation.

Incorporating Z requires careful implementation due to two challenges.

First, trivially adding Z to the input along with Y invokes d-separation and as a result Y and Z can get entangled.

Intuitively, Z is adding high level details to the intermediate representation Y .

We leverage this insight as an inductive bias, by incorporating Z at higher layers of the network rather than just feeding it as an input to the bottom layer.

A straightforward implementation of this idea does not work tough.

The reason is that BatchNorm uses batch-level statistics to normalize the incoming activations of the previous layer to speed up learning.

In practice, mini-batch statistics is used to approximate batch statistics.

This adds internal stochasticity to the generator causing it to ignore any externally added noise, such as Z. An elegant solution to resolve this second challenge comes in the form of adaptive instance normalization (see Section 2.3).

It not only removes any dependency on the batch-statistics but also allows for the incorporation of Z in the normalization process itself.

For this reason, it has previously been used in style transfer tasks (Huang & Belongie, 2017) .

We replace all instances of BatchNorm in the generator with Adaptive InstanceNorm.

We then introduce Z to the generative process using equation 1.

γ(z) and β(z) are parameterized as a simple feed-forward network and are applied to each layer of AdaIN in the generator.

Disentangled representation learning has been widely studied in recent years, both in the supervised and unsupervised settings.

In supervised cases, works such as Tran et al. (2018) Recently, Locatello et al. (2018) has emphasized the use of inductive biases and weak supervision instead of fully unsupervised methods for disentangled representation learning.

Nguyen-Phuoc et al. (2019) and Sitzmann et al. (2019) have successfully shown that including inductive biases, respectively an explicit 3D representation, leads to better performance.

Their inductive bias comes in the form of learned 3D transformation pipeline.

In comparison, CZ-GEM is much simpler and smaller in design and applies to the general setting where the data is determined by control and noise variables.

In addition, it can be used in both supervised and unsupervised setting and does not rely on the knowledge of 3D transformations.

Manually disentangled generative models like the 3D morphable model (Blanz & Vetter, 1999) have been built for faces.

They are powerful in terms of generalization but there is a big gap between those synthetic images and real-world face images.

In addition, those models are built highly supervised from 3D scans and the approach is limited by the correspondence assumption which does not scale to more complex objects like chairs .

We use a 3D morphable model to generate our synthetic face dataset and show that we can disentangle pose variation from synthetic and real 2D images.

In this section, we provide a comprehensive set of quantitative and qualitative results to demonstrate how CZ-GEM is clearly able to not only disentangle C from Z in both supervised and unsupervised settings but also ensure that independent components of C stay disentangled after training.

Additionally, we show how in unsupervised settings CZ-GEM can be used to discover disentangled latent factors when C is not explicitly provided.

We evaluate CZ-GEM on a variety of image generation tasks which naturally involve observed attributes C and unobserved attributes Z. To that end, we generate three 3D image datasets of faces, chairs, and cars with explicit control variables.

Chairs and cars datasets are derived from ShapeNet (Chang et al., 2015) .

We sample 100k images from the full yaw variation and a pitch variation of 90 degrees.

We used the straight chair subcategory with 1968 different chairs and the sedan subcategory with 559 different cars.

We used Blender to render the ShapeNet meshes scripted with the Stanford ShapeNet renderer.

For faces, we generated 100k images from the Basel Face Model 2017 (Gerig et al., 2018) .

We sample shape and color (first 50 coefficients), expressions (first 5 coefficients), pose (yaw -90 to 90 degrees uniformly, pitch and roll according to a Gaussian with variance of 5 degrees) and the illumination from the Basel Illumination Prior .

For the generation of the faces dataset, we use the software provided by .

For the stated datasets we have complete access to C, but we also include unsupervised results on celebA (Liu et al., 2015) with unconstrained real images.

All our datasets are built from publicly available data and tools.

We use the DCGAN architecture (Radford et al., 2015) for all neural networks involved in all the experiments in this work and provide a reference implementation with exact architecture and hyperparameter settings at https://github.com/AnonymousAuthors000/CZ-GEM.

In the supervised setting we compare CZ-GEM to CGAN.

We quantitatively compare the two methods to ensure that independent components of C stay disentangled post learning.

Furthermore, we qualitatively compare their abilities to disentangle C and Z. And finally, we compare the quality of the samples that the models generate.

For chairs and cars, C contains only the pose variables and all other variations are explained by Z. For faces, C contains in addition to pose the first 4 principal directions of shape variations.

three datasets.

We also include the training error (i.e. the MSE of the regressor on the real data) for comparison.

The results show that CGAN and CZ-GEM are comparable in preserving the label information in the generated data, but as we show below, only CZ-GEM does that while ensuring that C and Z remain disentangled.

To qualitatively evaluate the level of disentanglement between C and Z, we vary each individual dimension of C over its range while holding Z constant.

We plot the generated images for both models on car and chair datasets in Figure 4 .

Notice that CZ-GEM allows us to vary the control variates without changing the identity of the object, whereas CGAN does not.

In addition, we find that for CGAN, the noise Z provides little to no control over the identity of the chairs.

This is potentially due to the internal stochasticity introduced by the BatchNorm.

The last rows for the CZ-GEM figures provide the visualization of Y .

It can be seen how Y is clearly preserving C (pose information) but averaging the identity related details.

We also qualitatively evaluate CZ-GEM on the more challenging faces dataset that includes 10 control variates.

As shown in Figure 9 in the appendix, CZ-GEM is not only able to model the common pose factors such as rotation and azimuth but also accurately captures the principal shape component of Basel face model that approximates the width of the forehead, the width of jaw etc.

Compared to CGAN, CZ-GEM does a qualitatively better job at keeping the identity constant.

Finally, in order to ensure that our method does not compromise the generative quality, we evaluate the Inception score (Salimans et al., 2016) on all three datasets.

Inception score has been widely used to measure the diversity and the generative quality of GANs.

As shown in Table 2 , unlike CGAN, CZ-GEM does not degrade the image quality.

We now test the performance of CZ-GEM in the unsupervised setting, where disentangled components of C needs to be discovered, using β-VAE, as part of learning the mapping C → Y .

For our purpose, we use a simple version of the original β-VAE method with a very narrow bottleneck (6D for faces and 2D for cars and chairs) to extract C.

The latent traversals for the faces dataset are presented in Figure 5 .

Unsupervised discovery is able to recover rotation as well as translation variation present in the dataset.

For comparison, we evaluate InfoGAN and present the results in Figure 6 where it is evident that CZ-GEM clearly outperforms InfoGAN on both disentanglement and generative quality.

More traversal results are provided in the appendix.

We further test our method on the CelebA dataset (Liu et al., 2015) , where pose information is not available.

This traversal plot is shown in Figure 7 .

Traversal plots for cars and chairs dataset are provided in the Appendix Figure 12 and Figure 13 .

We present a simple yet effective method of learning representations in deep generative models in the setting where the observation is determined by control variate C and noise variate Z. Our method ensures that in the learned representation both C and Z are disentangled as well as the components of C themselves.

This is done without compromising the quality of the generated samples.

In future work, we would like to explore how this method can be applied to input with multiple objects.

Apart from the MSE-based estimator reported in 1, we report and additional evaluation measure.

We use the same regressor f (x) trained for 1, but we report the Pearson correlation co-efficient (r) between the predicted label and the true label r(c, f (G θ (c, z))) for each dimension of C. Comparison of CZ-GEM and CGAN on face dataset is shown in Figure 9 .

CGAN not only produces blurry faces but also shows more undesired identity changes.

In order to show the shape variation clearly, we provide a zoomed-in view in Figure 10 .

We provide additional results for supervised and unsupervised results on the chair dataset from Aubry et al. (2014) in Figure 11 and Figure 12 respectively.

The observation is the same with the previous one.

CZ-GEM varies the control variables without changing the shape of chairs.

In the first row in Figure 11 , the leg of the chairs are visually indistinguishable showing an excellent disentanglement between C and Z. For the results in unsupervised setting showing in Figure 12 , CZ-GEM is able to disentangle the rotation of chairs without any label.

Additional results of latent traversal of CZ-GEM in the unsupervised setting is provided in Figure 13 .

The model is able capture the rotation but the translation is not very smooth.

<|TLDR|>

@highlight

Hierarchical generative model (hybrid of VAE and GAN) that learns a disentangled representation of data without compromising the generative quality.