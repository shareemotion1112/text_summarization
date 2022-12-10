We propose a novel unsupervised generative model, Elastic-InfoGAN, that learns to disentangle object identity from other low-level aspects in class-imbalanced datasets.

We first investigate the issues surrounding the assumptions about uniformity made by InfoGAN, and demonstrate its ineffectiveness to properly disentangle object identity in imbalanced data.

Our key idea is to make the discovery of the discrete latent factor of variation invariant to identity-preserving transformations in real images, and use that as the signal to learn the latent distribution's parameters.

Experiments on both artificial (MNIST) and real-world (YouTube-Faces) datasets demonstrate the effectiveness of our approach in imbalanced data by: (i) better disentanglement of object identity as a latent factor of variation; and (ii) better approximation of class imbalance in the data, as reflected in the learned parameters of the latent distribution.

Generative models aim to model the true data distribution, so that fake samples that seemingly belong to the modeled distribution can be generated (Ackley et al. (1985) ; Rabiner (1989) ; Blei et al. (2003) ).

Recent deep neural network based models such as Generative Adversarial Networks (Goodfellow et al. (2014) ; Salimans et al. (2016) ; ) and Variational Autoencoders (Kingma & Welling (2014) ; Higgins et al. (2017) ) have led to promising results in generating realistic samples for high-dimensional and complex data such as images.

More advanced models show how to discover disentangled representations ; Chen et al. (2016) ; Tran et al. (2017) ; Hu et al. (2018) ; Singh et al. (2019) ), in which different latent dimensions can be made to represent independent factors of variation (e.g., pose, identity) in the data (e.g., human faces).

InfoGAN ) in particular, tries to learn an unsupervised disentangled representation by maximizing the mutual information between the discrete or continuous latent variables and the corresponding generated samples.

For discrete latent factors (e.g., digit identities), it assumes that they are uniformly distributed in the data, and approximates them accordingly using a fixed uniform categorical distribution.

Although this assumption holds true for many existing benchmark datasets (e.g., MNIST LeCun (1998)), real-word data often follows a long-tailed distribution and rarely exhibits perfect balance between the categories.

Indeed, applying InfoGAN on imbalanced data can result in incoherent groupings, since it is forced to discover potentially non-existent factors that are uniformly distributed in the data; see Fig. 1 .

In this work, we augment InfoGAN to discover disentangled categorical representations from imbalanced data.

Our model, Elastic-InfoGAN, makes two modifications to InfoGAN which are simple and intuitive.

First, we remodel the way the latent distribution is used to fetch the latent variables; we lift the assumption of any knowledge about class imbalance, where instead of deciding and fixing them beforehand, we treat the class probabilities as learnable parameters of the optimization process.

To enable the flow of gradients back to the class probabilities, we employ the Gumbel-Softmax distribution (Jang et al. (2017) ; Maddison et al. (2017) ), which acts as a proxy for the categorical distribution, generating differentiable samples having properties similar to that of categorical samples.

Second, we enforce our network to assign the same latent category for an image I and its transformed image I , which induces the discovered latent factors to be invariant to identity-preserving transformations like illumination, translation, rotation, and scale changes.

Although there are multiple meaningful ways to partition unlabeled data-e.g.

, with digits, one partitioning could be based Samples generated with an InfoGAN model learned with a fixed uniform categorical distribution Cat(K = 10, p = 0.1) on balanced and imbalanced data, respectively.

Each row corresponds to a different learned latent category. (Right): Samples generated with Elastic-InfoGAN using its automatically learned latent categorical distribution.

Although InfoGAN discovers digit identities in the balanced data, it produces redundant/incoherent groupings in the imbalanced data.

In contrast, our model is able to discover digit identities in the imbalanced data.

on identity, whereas another could be based on stroke width-we aim to discover the partitioning that groups objects according to a high-level factor like identity while being invariant to low-level "nuisance" factors like lighting, pose, and scale changes.

Such partitionings focusing on object identity are more likely to be useful for downstream visual recognition applications (e.g., semi-supervised object recognition).

In sum, our modifications to InfoGAN lead to better disentanglement and categorical grouping of the data (Fig. 1) , while at the same time enabling the discovery of the original imbalance through the learned probability parameters of the Gumbel softmax distribution.

Importantly, these modifications do not impede InfoGAN's ability to jointly model both continuous and discrete factors in either balanced or imbalanced data scenarios.

Our contributions can be summarized as follows: (1) To our knowledge, our work is the first to tackle the problem of unsupervised generative modeling of categorical disentangled representations in imbalanced data.

We show qualitatively and quantitatively our superiority in comparison to Info-GAN and other relevant baselines.

(2) Our work takes a step forward in the direction of modeling real data distributions, by not only explaining what modes of a factor of variation are present in the data, but also discovering their respective proportions.

Disentangled representation learning Learning disentangled representations of the data has a vast literature (Hinton et al. (2011); Bengio et al. (2013) Singh et al. (2019) ).

InfoGAN ) is one of the most popular unsupervised GAN based disentanglement methods, which learns disentanglement by maximizing the mutual information between the latent codes and generated images.

It has shown promising results for discovering meaningful latent factors in balanced datasets like MNIST (LeCun (1998)), CelebA (Liu et al. (2015) ), and SVHN (Netzer et al. (2011) ).

The recent method of JointVAE (Dupont (2018) ) extends beta-VAE (Higgins et al. (2017) ) by jointly modeling both continuous and discrete factors, using Gumbel-Softmax sampling.

However, both InfoGAN and JointVAE assume uniformly distributed data, and hence fail to be equally effective in imbalanced data, evident by Fig. 1 and our experiments.

Our work proposes modifications to InfoGAN to enable it to discover meaningful latent factors in imbalanced data.

Learning from imbalanced data Real world data have a long-tailed distribution (Guo et al. (2016) ; Van Horn et al. (2018) ), which can impede learning, since the model can get biased towards the dominant categories.

To alleviate this issue, researchers have proposed re-sampling (Chawla et al. (2002) ; He et al. (2008) ; Shen et al. (2016) ; Buda et al. (2018) ; Zou et al. (2018) ) and class reweighting techniques (Ting (2000) ; ; Dong et al. (2017) ; Mahajan et al. (2018)) to oversample rare classes and down-weight dominant classes.

These methods have shown to be effective for the supervised setting, in which the class distributions are known a priori.

There are Figure 2: Elastic-InfoGAN takes a sampled categorical code from a Gumbel-Softmax distribution and a noise vector to generate fake samples.

Apart from the original InfoGAN ) loss functions, we have two additional constraints: (1) We take real images x and create a transformed version x using identity-preserving operations (e.g., small rotation), and force their inferred latent code distributions to be close; (2) We also constrain their entropy to be low.

The use of differentiable latent variables from the Gumbel-Softmax enables gradients to flow back to the class probabilities to update them.

also unsupervised clustering methods that deal with imbalanced data in unknown class distributions (e.g., Nguwi & Cho (2010); You et al. (2018) ).

Our model works in the same unsupervised setting; however, unlike these methods, we propose an unsupervised generative model method that learns to disentangle latent categorical factors in imbalanced data.

Leveraging data augmentation for unsupervised image grouping Some works (Hui (2013) ; Dosovitskiy et al. (2015) ; Hu et al. (2017) ; Ji et al. (2019) ) use data augmentation for image transformation invariant unsupervised clustering or representation learning.

The main idea is to maximize the mutual information or similarity between the features of an image and its corresponding transformed image.

However, unlike our approach, these methods do not target imbalanced data and do not perform generative modeling.

Let X = {x 1 , x 2 , . . .

, x N } be a dataset of N unlabeled images from k different classes.

No knowledge about the nature of class imbalance is known beforehand.

Our goal is twofold: (i) learn a generative model G which can learn to disentangle object category from other aspects (e.g., digits in MNIST (LeCun (1998)), face identity in YouTube-Faces (Wolf et al. (2011) )); (ii) recover the unknown true class imbalance distribution via the generative modeling process.

In the following, we first briefly discuss InfoGAN ), which addressed this problem for the balanced setting.

We then explain how InfoGAN can be extended to the scenario of imbalanced data.

Learning disentangled representations using the GAN (Goodfellow et al. (2014) ) framework was introduced in InfoGAN ).

The intuition is for generated samples to retain the information about latent variables, and consequently for latent variables to gain control over certain aspects of the generated image.

In this way, different types of latent variables (e.g., discrete categorical vs. continuous) can control properties like discrete (e.g., digit identity) or continuous (e.g., digit rotation) variations in the generated images.

Formally, InfoGAN does this by maximizing the mutual information between the latent code c and the generated samples G(z, c), where z ∼ P noise (z) and G is the generator network.

The mutual information I(c, G(c, z)) can then be used as a regularizer in the standard GAN training objective.

Computing I(c, G(c, z)) however, requires P (c|x), which is intractable and hard to compute.

The authors circumvent this by using a lower bound of I(c, G(c, z)), which can approximate P (c|x) via a neural network based auxiliary distribution Q(c|x).

The training objective hence becomes:

Figure 3: Different ways for unsupervised learning based methods to group unlabeled data; based on rotation (left) vs. digit identity (right).

Here, we show two different groups for each grouping.

where D is the discriminator network, and H(c) is the entropy of the latent code distribution.

Training with this objective results in latent codes c having control over the different factors of variation in the generated images G(z, c).

To model discrete variations in the data, InfoGAN employs nondifferentiable samples from a uniform categorical distribution with fixed class probabilities; i.e., c ∼ Cat(K = k, p = 1/k) where k is the number of discrete categories to be discovered.

As shown in Fig. 1 , applying InfoGAN to an imbalanced dataset results in suboptimal disentanglement, since the uniform prior assumption does not match the actual ground-truth data distribution of the discrete factor (e.g., digit identity).

To address this, we propose two augmentations to InfoGAN.

The first is to enable learning of the latent distribution's parameters (class probabilities), which requires gradients to be backpropagated through latent code samples c, and the second is to enforce identity-preserving transformation invariance in the learned latent variables so that the resulting disentanglement favors groups that coincide with object identities.

Learning the prior distribution To learn the prior distribution, we replace the fixed categorical distribution in InfoGAN with the Gumbel-Softmax distribution (Jang et al. (2017) ; Maddison et al. (2017) ), which enables sampling of differentiable samples.

The continuous Gumbel-Softmax distribution can be smoothly annealed into a categorical distribution.

Specifically, if p 1 , p 2 ..., p k are the class probabilities, then sampling of a k-dimensional vector c can be done in a differentiable way:

Here g i , g j are samples drawn from Gumbel(0, 1), and τ (softmax temperature) controls the degree to which samples from Gumbel-Softmax resemble the categorical distribution.

Low values of τ make the samples possess properties close to that of a one-hot sample.

In theory, InfoGAN's behavior in the class balanced setting (Fig. 1 left) can be replicated in the imbalanced case (where grouping becomes incoherent, Fig. 1 center) , by simply replacing the fixed uniform categorical distribution with Gumbel-Softmax with learnable class probabilities p i 's; i.e. gradients can flow back to update the class probabilities (which are uniformly initialized) to match the true class imbalance.

And once the true imbalance gets reflected in the class probabilities, the possibility of proper categorical disentanglement ( Fig. 1 right) becomes feasible.

Empirically, however, this ideal behavior is not observed in a consistent manner.

As shown in Fig. 3 (left), unsupervised grouping can focus on non-categorical attributes such as rotation of the digit.

Although this is one valid way to group unlabeled data, our goal in this work is to prefer groupings that correspond to class identity as in Fig. 3 (right).

Learning object identities To capture object identity as the factor of variation, we make another modification to InfoGAN.

Specifically, to make the model focus on high level object identity and be invariant to low level factors like rotation, thickness, illumination, etc., we explicitly create these identity-preserving transformations on real images, and enforce the latent prediction Q(c|x) to be invariant to these transformations.

Note that such transformations (aka data augmentations) are standard for learning invariant representations for visual recognition tasks.

Formally, for any real image x ∼ P data (x), we apply a set of transformations δ to obtain a transformed image x = δ(x).

It is important to point out that these transformations are not learned over the optimization process.

Instead we use fixed simple transformations which guarantee that the human defined object identity label for the original image x and the transformed image x image remain the same.

For example, the digit identity of a 'one' from MNIST will remain the same if a transformation of rotation (±10 degree) is applied.

Similarly, a face identity will remain the same upon horizontal flipping.

We hence formulate our transformation constraint loss function:

where d(·) is a distance metric (e.g., cosine distance), and Q(c x |x), Q(c x |x ), are the latent code predictions for real image x and transformed image x , respectively.

Note that ideally Q(c|x), for either x ∼ P data (x) or x ∼ P g (G), should have low entropy (peaky class distribution) for proper inference about the latent object category.

Eq. 2 automatically enforces a peaky class distribution for Q(c|x) for x ∼ P g (G), because the sampled input latent code c from Gumbel-Softmax is peaky.

For x ∼ P data (x) though, Eq. 4 alone isn't sufficient as it can be optimized in a sub-optimal manner (e.g., if c x ≈ c x , but both have high entropy).

We hence add an additional entropy loss which forces c x and c x to have low entropy (s) class distributions:

The losses L trans and L ent , along with Gumble-Softmax, constitute our overall training objective:

V Inf oGAN plays the role of generating realistic images and associating the latent variables to correspond to some factor of variation in the data, while the addition of L trans will push the discovered factor of variation to be close to object identity.

Finally, L ent 's objective is to ensure Q behaves similarly for real and fake image distributions.

The latent codes sampled from Gumbel-softmax, generated fake images, and losses operating on fake images are all functions of class probabilities p i 's too.

Thus, during the minimization phase of Eqn.

6, the gradients are used to optimize the class probabilities along with G and Q in the backward pass.

In this section, we perform quantitative and qualitative analyses to demonstrate the advantage of Elastic-InfoGAN in discovering categorical disentanglement for imbalanced datasets.

We use: (1) MNIST (LeCun (1998)) and (2) YouTube-Faces (Wolf et al. (2011)) .

MNIST is by default a balanced dataset with 70k images, with a similar number of training samples for each of 10 classes.

We artificially introduce imbalance over 50 random splits (max imbalance ratio 10:1 between the largest and smallest class).

YouTube-Faces is a real world imbalanced video dataset with varying number of training samples (frames) for the 40 face identity classes (as used in Shah & Koltun (2018)).

The smallest/largest class has 53/695 images, with a total of 10,066 tightly-cropped face images.

All results are reported over the average of: (i) 50 runs (over 50 random imbalances) for MNIST, (ii) 5 runs over the same imbalanced dataset for YouTube-Faces.

We use MNIST to provide a proof-of-concept of our approach.

For example, one of the ways in which different 'ones' in MNIST vary is rotation, which can be used as a factor (as opposed to object identity) to group data in imbalanced cases (recall Fig. 3 left) .

Thus, using rotation as a transformation in L trans should alleviate this problem.

We ultimately care most about the YouTube-Faces results since it is more representative of real world data, both in terms of challenging visual variations (e.g., facial pose, scale, expression, and lighting changes) a well as inherent class imbalance.

For this reason, the effect of augmentations in L trans will be more reflective of how well our model can work in real world data.

We design different baselines to show the importance of having learnable priors for different latent variables and applying our transformation constraints.

• Uniform InfoGAN ): This is the original InfoGAN with fixed and uniform categorical distribution.

• Ground-truth InfoGAN: This is InfoGAN with a fixed, but imbalanced categorical distribution where the class probabilities reflect the ground-truth class imbalance.

• Ground-truth InfoGAN + Transformation constraint: Similar to the previous baseline but with our data transformation constraint (L trans ).

• Gumbel-softmax: In this case, InfoGAN does not have a fixed prior for the latent variables.

Instead, the priors are learned using the Gumbel-softmax technique (Jang et al. (2017) ).

• Gumbel-softmax + Transformation constraint:

Apart from having a learnable prior we also apply our transformation constraint (L trans ).

This is a variant of our final approach that does not have the entropy loss (L ent ).

• Gumbel-softmax + Transformation constraint + Entropy Loss (Elastic-InfoGAN): This is our final model with all the losses, L trans and L ent , in addition to V Inf oGAN (D, G, Q).

• JointVAE (Dupont (2018)): We also include this VAE based baseline, which performs joint modeling of disentangled discrete and continuous factors.

Our evaluation should capture: (1) how well we learn class-specific disentanglement for the imbalanced dataset, and (2) recover the ground-truth class distribution of the imbalanced dataset.

To capture these aspects, we apply three evaluation metrics:

• Average Entropy (ENT): Evaluates two properties: (i) whether the images generated for a given categorical code belong to the same ground-truth class i.e., whether the ground-truth class histogram for images generated for each categorical code has a low entropy; (ii) whether each ground-truth class is associated with a single unique categorical code.

We generate 1000 images for each of the k latent categorical codes, compute class histograms using a pre-trained classifier 2 to get a k × k matrix (where rows index latent categories and columns index ground-truth categories).

We report the average entropy across the rows (tests (i)) and columns (tests (ii)).

• Normalized Mutual Information (NMI) (Xu et al. (2003) ): We treat our latent category assignments of the fake images (we generate 1000 fake images for each categorical code) as one clustering, and the category assignments of the fake images by the pre-trained classifier as another clustering.

NMI measures the correlation between the two clusterings.

The value of NMI will vary between 0 to 1; higher the NMI, stronger the correlation.

• Root Mean Square Error (RMSE) between predicted and actual class distributions: measures the accuracy of approximating the true class distribution of the imbalanced dataset.

Since the learned latent distribution may not be aligned to the ground-truth distribution (e.g., the first dimension for the learned distribution might capture 9's in MNIST whereas the first dimension for the groundtruth distribution may be for 0's), we need a way to align the two.

For this, we use the pre-trained classifier to classify the generated images for a latent variable and assign the variable to the most frequent class.

If more than one latent variable is assigned to the same class, then their priors are added before computing its distance with the known prior of the ground-truth class.

We first evaluate disentanglement quality as measured by NMI and average entropy (ENT); see

Figure 4: Representative image generations on a random imbalanced MNIST split.

Each row corresponds to a learned latent variable.

Our approach generates inconsistent images in only row 2 whereas Uniform InfoGAN does so in rows 1,2,6,8 and JointVAE does so in rows 3,5,6,7,9,10.

particular, our full model obtains significant boosts of 0.101 and 0.104 in NMI, and -0.222 and -0.305 in ENT compared to the Uniform InfoGAN baseline for MNIST and YouTube-Faces, respectively.

The boost is even more significant when compared to JointVAE: 0.1977, 0.3380 in NMI, and -0.4658, -0.9963 in ENT for MNIST and YouTube-Faces, respectively.

This again is a result of the assumption of a uniform categorical prior by JointVAE, along with poorer quality generations.

We see that our transformation constraint generally improves the performance for both when the ground-truth prior is known (Ground-truth InfoGAN vs. Ground-truth InfoGAN + Transformation constraint) as well as when the prior is learned (Gumbel-softmax vs. Gumbel-softmax + Transformation constraint).

This shows that enforcing the network to learn groupings that are invariant to identity-preserving transformations helps it to learn a disentangled representation in which the latent dimensions correspond more closely to identity-based classes.

Also, learning the prior using the Gumbel-softmax leads to better categorical disentanglement than fixed uniform priors, which demonstrates the importance of learning the prior distribution in imbalanced data.

Overall, our approach using Gumbel-softmax to learn the latent prior distribution together with our transformation constraint works better than applying them individually, which demonstrates their complementarity.

Interestingly, using a fixed ground-truth prior (Ground-truth InfoGAN) does not result in better disentanglement than learning the prior (Gumbel-softmax).

This requires further investigation, but we hypothesis that having a rigid prior makes optimization more difficult compared to allowing the network to converge to a distribution on its own, as there are multiple losses that need to be simultaneously optimized.

Finally, in Table 2 , we evaluate how well the Gumbel-softmax can recover the ground-truth prior distribution.

For this, we compute the RMSE between the learned prior distribution and ground- truth prior distribution.

Our full model (transformation constraint + entropy loss) produces the best estimate of the true class imbalance for both datasets, as evident through lowest RMSE.

Our improvement over the Gumbel-Softmax baseline indicates the importance of our tranformation L trans and entropy L ent losses in approximating the class imbalance.

We next qualitatively evaluate the disentanglement achieved by our approach .

Figs. 4 , 5, and 7 show results for MNIST and YouTube-Faces.

Overall, Elastic-InfoGAN generates more consistent images for each latent code compared to Uniform InfoGAN and JointVAE.

For example, in Fig. 4 , ElasticInfoGAN only generates inconsistent images in the second row whereas the baseline approaches generate inconsistent images in several rows.

Similarly, in Fig. 7 , Elastic-InfoGAN generates faces of the same person corresponding to a latent variable more consistently than the baselines.

Both Uniform InfoGAN and JointVAE on the other hand tend to mix up identities within the same categorical code because they incorrectly assume a prior uniform distribution.

Finally, we demonstrate that Elastic-InfoGAN does not impede modeling of continuous factors in the imbalanced setting.

Specifically, one can augment the input with continuous latent codes (e.g. r1, r2 ∼ Unif(-1, 1)) along with the existing categorical and noise vectors.

In Fig. 6 , we show the results of continuous code interpolation; we can see that each of the two continuous codes largely captures a particular continuous factor (stroke width on left, and digit rotation on the right).

In this work, we proposed a new unsupervised generative model that learns categorical disentanglement in imbalanced data.

Our model learns the class distribution of the imbalanced data and enforces invariance to be learned in the discrete latent variables.

Our results demonstrate superior performance over alternative baselines.

We hope this work will motivate other researchers to pursue this interesting research direction in generative modeling of imbalanced data.

For MNIST, we operate on the original 28x28 image size, with 10-dimensional categorical code to represent 10 digit categories.

For YouTube-Faces, we crop the faces using bounding box annotations provided, and then resize them to 64x64 resolution, and use a 40-dimensional categorical code to represent 40 face identities (first 40 categories sorted in alphabetical manner), as done in Shah & Koltun (2018) .

Pre-trained classification architecture used for evaluation for MNIST: 2 Conv + 2 FC layers, with max pool and ReLU after every convolutional layer.

For YouTube-Faces classification, we fine-tune a ResNet-50 network pretrained on VGGFace2, for face recognition.

We set λ 1 = 1 (for L 1 ), λ 2 = 10 (for L trans ), and λ 3 = 1 (for L ent ).

These hyperparameters were chosen to balance the magnitude of the different loss terms.

Finally, one behavior we observe is that if the random initialization of class probabilities is too skewed (only few classes have high probability values), then it becomes very difficult for them to get optimized to the ideal state.

We hence initialize them with the uniform distribution, which makes training much more stable.

We follow the exact architecture as described in InfoGAN ): The generator network G takes as input a 64 dimensional noise vector z ∼ N (0, 1) and 10 dimensional samples from Gumbel-Softmax distribution.

The discriminator D and the latent code prediction network Q share most of the layers except the final fully connected layers.

Elastic-InfoGAN architecture for YouTube Faces We operate on cropped face images resized to 64x64 resolution.

Our architecture is based on the one proposed in StackGANv2 (Zhang et al. (2018) ), where we use its 2-stage version for generating 64x64 resolution images.

The input is a 100 dimensional noise vector z ∼ N (0, 1) and 40 dimensional samples (c) from the Gumbel-Softmax distribution.

There is an initial fully connected layer which maps the input (concatenation of z and c) to an intermediate feature representation.

A series of a combination of upsampling + convolutional (interleaved with batch normalization and Gated Linear Units) increase the spatial resolution of the feature representation, starting from 1024 (feature size: 4 x 4 x 1024) channels to 64 (feature size: 64 x 64 x 64) channels.

For the first stage, a convolutional network transforms the feature representation into a 3 channel output, while maintaining the spatial resolution; this serves as the fake image from the first stage.

The next stage uses the 64 x 64 x 64 resolution features, forwards it through a network containing residual blocks and convolutional layers, while again maintaining the spatial resolution of 64 x 64.

For the second stage, again a convolutional layer maps the resulting feature into a 64 x 64 resolution fake image, which is the one used by the model for evaluation purposes.

The discriminator networks are identical at both stages.

It consists of 4 convolutional layers interleaved with batch normalization and leaky ReLU layers, which serve as the common layers for both the D and Q networks.

After that, D has one non-shared convolutional layer which maps the feature representation into a scalar value reflecting the real/fake score.

For Q, we have a pair of non-shared convolutional layers which map the feature representation into a 40 dimensional latent code prediction.

We employ a similar way of training the generative and discriminative modules as described in Chen et al. (2016) .

We first update the discriminator based on the real/fake adversarial loss.

In the next step, after computing the remaining losses (mutual information + L trans + L ent ), we update the generator (G) + latent code predictor (Q) + latent distribution parameters at once.

Our optimization process alternates between these two phases.

For MNIST, we train all baselines for 200 epochs, with a batch size of 64.

For YouTube-Faces, we train until convergence, as measured via qualitative realism of the generated images.

We use a batch size of 50.

τ = 0.1 when used for sampling from Gumbel-Softmax, which results in samples having very low entropy (very close to one hot vectors from a categorical distribution).

Here we describe the exact class imbalance used in our experiments.

For MNIST, we include below the 50 random imbalances created.

For YouTube-Faces, we include the true ground truth class imbalance in the first 40 categories.

The imbalances reflect the class frequency.

A.2.1 MNIST 147, 0.037, 0.033, 0.143, 0.136, 0.114, 0.057, 0.112, 0.143, 0.078 • 0.061, 0.152, 0.025, 0.19, 0.12, 0.036, 0.092, 0.185, 0.075, 0.064 • 0.173, 0.09, 0.109, 0.145, 0.056, 0.114, 0.075, 0.03, 0.093, 0.116 • 0.079, 0.061, 0.033, 0.139, 0.145, 0.135, 0.057, 0.062, 0.169, 0.121 • 0.053, 0.028, 0.111, 0.142, 0.13, 0.121, 0.107, 0.066, 0.125, 0.118 • 0.072, 0.148, 0.092, 0.081, 0.119, 0.172, 0.05, 0.109, 0.085, 0.073 • 0.084, 0.143, 0.07, 0.082, 0.059, 0.163, 0.156, 0.063, 0.074, 0.105 • 0.062, 0.073, 0.065, 0.183, 0.099, 0.08, 0.05, 0.16, 0.052, 0.177 • 0.139, 0.113, 0.074, 0.06, 0.068, 0.133, 0.142, 0.13, 0.112, 0.03 • 0.046, 0.128, 0.059, 0.112, 0.135, 0.164, 0.142, 0.125, 0.051, 0.037 • 0.107, 0.057, 0.154, 0.122, 0.05, 0.111, 0.032, 0.044, 0.136, 0.187 • 0.129, 0.1, 0.039, 0.112, 0.119, 0.095, 0.047, 0.14, 0.156, 0.064 • 0.146, 0.08, 0.06, 0.072, 0.051, 0.119, 0.176, 0.11, 0.158, 0.028 A.3 DISCUSSION ABOUT EVALUATING PREDICTED CLASS IMBALANCE IN SEC.

4.2

To measure the ability of a generative model to approximate the class imbalance present in the data, we derive a metric in Section 4.2 of the main paper, the results of which are presented in Table  2 .

Even though we do get better results as measured by RMSE between the approximated and the original imbalance distribution, we would like to discuss certain flaws associated with this metric.

In its current form, we compute the class histogram (using the pre-trained classifier, which classifies each fake image into one of the ground-truth categories) for a latent code and associate the latent code to the most frequent class.

If multiple latent codes get associated to the same ground-truth class, there will be ground-truth classes for which the predicted class probability will be zero.

This is rarely an issue for MNIST, as it only has 10 ground-truth classes, and thus in most cases both our method and the baselines assign each latent code to a unique ground-truth class.

However, for YouTube-Faces, after associating latent codes to the ground truth categories in this manner, roughly 10-13 ground-truth classes (out of 40) get associated with 0 probability for both our approach and the baselines (due to multiple latent codes being associated to the same majority ground-truth class).

Our metric therefore may be too strict, especially for difficult settings with many confusing groundtruth categories.

The tricky part about evaluating how well the model is approximating the class imbalance is that there are two key aspects that need to be simultaneously measured.

Specifically, not only should (i) the raw probability values discovered match the ground-truth class imbalance distribution, but (ii) the class probabilities approximated by the latent codes must correspond to the correct ground-truth classes.

For example, if the original data had 80% samples from class A and 20% from class B, the generative model should not only estimate the imbalance as 80%-20%, but the model must associate 80% to class A and 20% to class B (instead of 80% to class B and 20% to class A).

Another way to evaluate whether a model is capturing the ground-truth class imbalance could be the FID score, but it's worth noting that a method can still have a good FID score without disentangling the different factors of variations.

Given the limitation with our metric on YouTube-Faces, we have also measured the min/max of predicted prior values.

For YouTube-Faces, the min/max of predicted and ground-truth priors are: Gumbel-Softmax: Min 2.76748415e-05, Max: 0.0819286481; Ours without L ent : Min 0.00211485, Max: 0.06152404; Ours complete: Min 0.00336615, Max: 0.06798439; and GroundTruth: Min 0.005265, Max: 0.069044.

Our full method's min/max more closely matches that of the ground-truth, and the overall ordering of the methods follows that of Table 2 using our RMSE based metric.

In sum, we have made an effort to evaluate accurate class imbalance prediction in multiple ways, but it is important to note that this is an area which calls for better metrics to evaluate the model's ability to approximate the class imbalance distribution.

@highlight

Elastic-InfoGAN is a modification of InfoGAN that learns, without any supervision, disentangled representations in class imbalanced data