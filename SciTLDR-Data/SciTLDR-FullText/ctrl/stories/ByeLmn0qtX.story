This paper proposes variational domain adaptation, a unified, scalable, simple framework for learning multiple distributions through variational inference.

Unlike the existing methods on domain transfer through deep generative models, such as StarGAN (Choi et al., 2017) and UFDN (Liu et al., 2018), the variational domain adaptation has three advantages.

Firstly, the  samples from the target are not required.

Instead, the framework requires one known source as a prior $p(x)$ and binary discriminators, $p(\mathcal{D}_i|x)$, discriminating the target domain $\mathcal{D}_i$ from others.

Consequently, the framework regards a target as a posterior that can be explicitly formulated through the Bayesian inference, $p(x|\mathcal{D}_i) \propto p(\mathcal{D}_i|x)p(x)$, as exhibited by a further proposed model of dual variational autoencoder (DualVAE).

Secondly, the framework is scablable to large-scale domains.

As well as VAE encodes a sample $x$ as a mode on a latent space: $\mu(x) \in \mathcal{Z}$, DualVAE encodes a domain $\mathcal{D}_i$ as a mode on the dual latent space $\mu^*(\mathcal{D}_i) \in \mathcal{Z}^*$, named domain embedding.

It reformulates the posterior with a natural paring $\langle, \rangle: \mathcal{Z} \times \mathcal{Z}^* \rightarrow \Real$, which can be expanded to uncountable infinite domains such as continuous domains as well as interpolation.

Thirdly, DualVAE fastly converges without sophisticated automatic/manual hyperparameter search in comparison to GANs as it requires only one additional parameter to VAE.

Through the numerical experiment, we demonstrate the three benefits with multi-domain image generation task on CelebA with up to 60 domains, and exhibits that DualVAE records the state-of-the-art performance outperforming StarGAN and UFDN.

"...we hold that all the loveliness of this world comes by communion in Ideal-Form.

All shapelessness whose kind admits of pattern and form, as long as it remains outside of Reason and Idea, is ugly from that very isolation from the Divine-Thought.

Agents that interact in various environments have to handle multiple observation distributions .

Domain adaptation BID0 ) is a methodology employed to exploit deep generative models, such as adversarial learning BID2 and variational inference BID8 , that can handle distributions that vary with environments and other agents.

Further, multi-task learning and domain transfer are examples of how domain adaptation methodology is used.

We focus on domain transfer involving transfers across a distribution between domains.

For instance, pix2pix BID5 ) outputs a sample from the target domain that corresponds to the input sample from the source domain.

This can be achieved by learning the pair relation of samples from the source and target domains.

CycleGAN BID21 transfers the samples between two domains using samples obtained from both domains.

Similarly, UNIT BID12 , DiscoGAN , and DTN BID20 have been proposed in previous studies.

However, the aforementioned method requires samples that are obtained from the target domains, and because of this requirement, it cannot be applied to domains for which direct sampling is expensive or often impossible.

For example, the desired, continuous, high-dimensional action in the environment, intrinsic reward (e.g., preference and curiosity) and the policy of interacting agents other than itself cannot be sampled from inside, and they can only discriminate the proposed input.

Even for ourselves, the concept of beauty or interest in our conscious is subjective, complex, and difficult to be sampled from the inside, although it is easy to discriminate on the outside.

The key concept of variational domain adaptation.

a) Given the proposal drawn from the prior, the discriminator discriminates the target domain from the others.

Each domain is posterior for the prior N (z|0, 1); further, the distribution in the latent space is observed to be a normal distribution using the conjugate likelihood.

b) Domain transfer is represented by the mean shift in the latent space.

c) Domain embedding: After training, all the domains can only be represented by vectors µ i .In this study, we propose variational domain adaptation, which is a framework for targets that pose challenges with respect to direct sampling.

One solution is multi-domain semi-supervision, which converts the problem to semi-supervised learning, thereby making is possible to perform variational inference.

In this supervision, a source domain is regarded as a prior p(x) and a target domain is considered to be a posterior p(x|D i ) by referring to the label given by a supervised discriminator p(D i |x) that distinguishes the target domain from others.

Our model imitates the behavior of the discriminator and models the target domain using a simple conclusion of the Bayesian theorem, p θ (x|D i ) ∝ p θ (D i |x)p θ (x).

The end-to-end learning framework also makes it possible to learn good prior p θ (x) with respect to all the domains.

After the training was completed, the posterior p θ (x|D i ) succeeded in deceiving the discriminator p(D i |x).

This concept is similar to rejection sampling in the Monte Carlo methods.

Further, variational domain adaptation is the first important contribution from this study.

The second contribution from this study is a model of dual variational autoencoder (DualVAE), which is a simple extension of the conditional VAE BID9 ), employed to demonstrate our concept of multi-domain semi-supervision.

DualVAE learns multiple domains in one network by maximizing the variational lower bound of the total negative KL-divergence between the target domain and the model.

DualVAE uses VAE to model the prior p(x) and an abstract representation for the discriminator p(D i |x).

The major feature of DualVAE is domain embedding that states that all the posteriors are modeled as a normal distribution N (z|µ i , σ 2 ) in the same latent space Z using the conjecture distribution of the prior.

Here, µ i is the domain embedding that represents the domain D i .

This enables us to sample from p θ (x|D i ).

Our major finding was that the discriminator of DualVAE was a simple inner product between the two means of domain embedding and the VAE output: DISPLAYFORM0 that acts as a natural paring between the sample and the domain.

The probabilistic end-to-end model learns multiple domains in a single network, making it possible to determine the effect of transfer learning and to learn data that multi-domains cannot observe from sparse feedback.

Domain embedding is a powerful tool and allows us to use VAEs instead of GANs.

The third contribution of this study is that DualVAE was validated for use in a recommendation task using celebA BID13 .

In the experiment, using celebA and face imaging data obtained based on evaluations by 60 users, an image was generated based on the prediction of user evaluation and an ideal image that was determined to be good by multiple users.

We demonstrated that the image could be modified to improve the evaluation by interpolating the image, and the image was evaluated using the domain inception score (DIS), which is the score of the model that has learned the preference of each user.

We present the beauty inside each evaluator by simply sampling p θ (x|D i ).

The DIS of DualVAE is higher than that of a single domain, and the dataset and code are available online.

Under review as a conference paper at ICLR 2019

The existing literature related to the domain transfer is based on the assumption that the samples are obtained from the target domain.

For example, pix2pix BID5 can output the samples from the target domain that corresponds to the input samples from the source domain by learning the pair relation between the samples of the source and target domains.

CycleGAN BID21 , which differs from pix2pix, does not require sample pairs from both domains.

Similarly, UNIT BID12 , DiscoGAN , and DTN BID20 also do not require sample pairs.

Furthermore, because there are few cases in which samples from the source and target domains form a one-to-one pair in real world research after being extended to the conversion of one-to-many relationships, including BicycleGAN BID22 and MUNIT BID3 .Several studies were conducted to model multiple distributions in a semi-supervised manner.

Star-GAN BID1 , UFDN , and RegCGAN BID14 are extensions of the aforementioned models and are frameworks that can convert the source domain samples into samples for various target domains with a single-network structure.

However, the problem with these methods is associated with hyperparameter tuning, which arises from the characteristics of adversarial learning.

DualVAE is a simple extension of a conditional VAE in a multi-domain situation.

Conditional VAEs utilizes VAE for semi-supervised learning.

Although the model is quite simple, it is powerful and scalable making it possible to learn multiple distributions with domain embedding.

In fact, we demonstrated that DualVAE quickly converged for more than 30 domains without sophisticated hyperparameter tuning.

In the experiment conducted in this study, E ω [J(θ|ω)] was evaluated instead of J(θ|ω) to demonstrate that our method required less hyperparameter tuning.

With regard to n domains D 1 , . . .

, D n , and a sample x on an observation space X , the objective of unsupervised domain adaptation is to minimize the KL-divergence between the target distribution and the model, DISPLAYFORM0 , over all the domains D i .

From the perspective of optimizing θ, minimizing the KL divergence is equivalent to maximizing the cross-entropy.

As DISPLAYFORM1 , the unsupervised domain adaptation can be formulated as a maximizing problem for the weighted average of cross-entropy over the domains: DISPLAYFORM2 where DISPLAYFORM3 for all the i's, the objective function is simply the mean, and if γ i = 0 for certain i's, the domain D i is ignored.

The difficulty arises from the fact that it is not possible to directly sample x from p (i) x can be directly sampled from the likelihood p(D i |x).

This challenge was the motivation for considering multi-domain semi-supervision.

Multi-domain semi-supervision assumes a prior p(x) and models each the domain as a posterior p (i) = p(x|D i ).

As the Bayesian inference, we reformulate the cross-entropy E x∼p (i) [log p θ (x|D i )] in Eq. (1) as follows: DISPLAYFORM0 where DISPLAYFORM1 , the objective is identical to: DISPLAYFORM2 where [n] is a uniform distribution over {1, . . .

, n} and DISPLAYFORM3 The first term is the likelihood from the discriminator; the second term is the prior learned by a generative model, including VAE; and the last term is the regularizer.

Because the equation is intractable, we use Monte Carlo sampling to estimate the function.

During the estimation, we initially sample x 1 , . . . , x m from the prior p(x) and subsequently obtain the binary labels y ij ∈ {0, 1} from each discriminator y ij ∼ p(D i |x j ).

Since the number of labels from supervises is nm, the situation that the sparse labels: k << nm is considered.

Further, some discriminators only provide parts of the labels.

In the situation, the missing values are 0-padded: DISPLAYFORM4 where ≈ indicates Monte Carlo estimation andȳ = n i=1 m j=1 y ij /k.

In the limit of n → ∞, the right side of the equation is identical to the left side.

We extended the VAE for multi-domain transfer to demonstrate our concept of multi-domain semisupervision.

Our proposed model, dual variational autoencoder (DualVAE), models each domain p i (x) as a posterior distribution p(x|D i ) that is similar to that observed in a conditional VAE.

FIG1 depicts the VAE and DualVAE graphical models.

The major feature of DualVAE is domain embedding, where all the domains and the prior share the same latent space Z. For the prior distribution, p(z) = N (z|0, I) and p(z|D i ) = N (z|µ i , σ 2 I), where µ i ∈ Z is an embedding and I is a unit matrix in Z. In the following, we denote σ 2 I = σ 2 without loss of generality.

The domain D i is characterized only by its embedding µ i .

Here, µ 0 is the embedding of the prior that can be assumed to be µ 0 = 0.Training DualVAE is virtually equivalent to simultaneously training (n + 1) VAEs which share a parameter, including the prior.

Using conjecture distribution for the prior p(z), the posterior distribution is observed to be a normal distribution.

Therefore, all the posteriors are VAEs.

The joint distribution can be given as follows: DISPLAYFORM0 A VAE BID8 ) is used to model the prior p(x), a deep generative model that employs an autoencoder to model the hidden variable as random variable.

The benefit of a VAE is that it can be used to model each distribution as a normal distribution in Z, achieved by maximizing the variational lower bound of log p(x) as follows: DISPLAYFORM1 where φ, w ∈ θ is a parameter of the encoder and the decoder, respectively.

The objective is to learn a pair of the encoder p w (x|z) and the decoder q φ (z|x) to maximize L(x).

z acts as a prior DISPLAYFORM2 The lower bound L θ (x) is derived using the reconstruction error and penalty term as the KL divergence between the model and the prior p(z).

Further, the gradient of the reconstruction term can be calculated using the Monte Carlo method, and because the construction term is the KL divergence between two normal distributions, it can be analytically calculated.

Right: The network structure of DualVAE.

The label is structured as the inner product of latent z θ and domain embedding z i .

DISPLAYFORM3 Using the definition and the Bayesian theorem, log f θ (D i |x) can be written as follows: DISPLAYFORM4 The equation above indicates log f θ (D i |x) can be written simply as the inner product between µ i and µ φ (x), and the objective can be written as follows: DISPLAYFORM5 where U = (µ 1 , . . .

, µ n ) T , µ * U = y T U/n and α = σ −2 .

Interestingly, it only requires one additional parameter U except a hyperparameter α.

U is named as a domain embedding matrix, representing the set of the domain prototypes.

Domain embedding makes it possible to extend our method to infinite domains such as a continuous domain.

In fact, µ * U (y) ∈ Z * represents a prototype of mixed domains indicated by y in a domain latent space Z * , a dual space of Z. Note that dim Z = dim Z * .

The overall parameters of DualVAE is θ = (w, φ, U ), where w is the encoder's , parameterφ is the decoders's parameter, and U is the domain embedding matrix.

While a typical VAE does not assume any distribution of w, φ, p(U ) is set as an exponential distribution with an additional hyperparameter β ∈ (0, ∞) to obtain sparse representation: DISPLAYFORM0 As the terms except for the first are independent of θ, we ignore them later as constants.

By putting together the prior, the discriminator, and the regularizer, the variational lower bound of the point-wise objective of DualVAE J(θ|x, y) can be written as a surprisingly simple form: DISPLAYFORM0 where u, v = v T u. Consequently, a DualVAE maximizes a duality paring ·, · : Z × Z * → R between the sample latent space Z = Z φ (X ) and the domain latent space DISPLAYFORM1 n .

Note that the objective requires only two additional hyperparameters in addition to the VAE.

If α, β → 0, it is equivalent to a single VAE.

Intuitively, 1/α and 1/β control variance and bias of the domain embeddings, respectively.

The training algorithm of the DualVAE is shown in Algorithm 1.

Require: observations (x j ) m j=1 , batch size M , VAE/encoder optimisers: g, g e , hyperparameters α, β, and the label matrix Y = (y j ) m j=1 .

Initialize encoder, decoder and domain embedding parameters: φ, w, U repeat DISPLAYFORM0

Based on an original numerical experiment in domain adaptation, we confirmed that the DualVAE learns multiple distributions both qualitatively and quantitatively.

Similar to the case of the existing methods, domain adaptation was confirmed via an image-generation task in this study.

First, we performed A facial image recommendation task, which is a content-based recommendation task for generating the preferences of users.

Second, we performed the standard domain transfer task with 40 domains in CelebA BID13 and we showed that DualVAE outperformed two state-ofthe-art methods through GAN and VAE.The objective of the first task was to generate an image that was preferred by a specific user.

We set the input space X as the raw image, the prior p(x) as faces, and the domain D i as a user.

We used the dataset of CelebA and SCUT-FBP5500 as the samples from the prior.

The objective of the task was to generate samples from p θ (x|D i ), exhibiting the images that were preferred by a user.

We used label y i ∼ p(D i |x) as the existing dataset of SCUT-FBP5500 with 5,500 faces and 60 users for the content-based recommendation.

The purpose of the second task was to transfer samples from p(x) into samples from p θ (x|D i ).

We set the prior p(x) as face images and the posterior p θ (x|D i ) as face images with certain attributes of CelebA. We used label y i ∼ p(D i |x) as the attribute of CelebA.The results revealed that the DualVAE successfully learned the model of the target distribution p θ (x|D i ) both quantitatively and qualitatively.

Quantitatively, we confirmed that the discriminator learned the distribution by evaluating the negative log-likelihood loss, − log p θ (D i |x).

We evaluated the samples using the domain inception score (DIS), which is the score for evaluating the transformation of images into multiple target domains.

Notably, the DIS of the DualVAE was higher than several models.

Qualitatively, we demonstrated that the image could be transferred to improve the evaluation by interpolating the image.

We further exhibited several beautiful facial images that the users were conscious of by decoding each domain embedding µ i , which can be considered as the projection of the ideal from inside the users.

In addition, 40 domain-transferred images using the dataset of CelebA by the proposed method was better than the images by other models.

CelebA CelebA BID13 comprises approximately 200,000 images of faces of celebrities with 40 attributes.

SCUT-FBP5500 SCUT-FBP5500 BID10 comprises 5500 face images and employs a 5-point scale evaluation by 60 people in terms of beauty preference.

The face images can be categorized as Asian male, Asian female, Caucasian male, and Caucasian female, with 2000, 2000, 750, 750 images, respectively.

The quantitative result of the experiment can be demonstrated by evaluating the generated images by several models using a Domain Inception Score (DIS).

Although the Inception Score BID18 ) is a score for measuring generated images, it can only measure the diversity of the images, and it is not for evaluating domain transfer of the images.

Therefore, we proposed using a DIS, which is a score for evaluating the transformation of images into multiple target domains.

The DIS is a scalar value using the output of Inceptionv3 BID19 pretrained to output the domain label, and it is evaluated by the sum of two elements.

The first is whether the domain transfer of the original image has been successful (transfer score), and the second is whether the features other than the transferred domain are retained (reconstruction score).

A more detailed explanation of the DIS is provided in the appendix.

Comparison of a DualVAE and a single-domain VAE A DualVAE can transform the image of the source domain into images of multiple target domains with one model.

However, considering a simpler method, it is also possible to transfer the image of the source domain to the images of the multiple target domains by creating multiple models.

We will call each of these models a Single Domain VAE (SD-VAE).

Since an SD-VAE is a model that converts the image of one source domain to the image of one target domain, models corresponding to the number of target domains are required, and thus, 60 models required training.

We demonstrated that the DualVAE performance was equal to or higher than that of the SD-VAE using the DIS.

With respect to the output images of these two models, the one with a higher DIS value was considered to be capable of outputting ideal images.

We calculated the DIS of 200 test images transferred by these two model.

The DIS of the DualVAE was -0.0185, whereas that of the SD-VAE was -0.0282.

Thus, the DIS of the DualVAE was 0.01 higher than that of SD-VAE.Comparison of DualVAE and several models The DualVAE was compared with several models capable of performing image-to-image translations for multiple domains using a single model.

In this experiment, only the celebA dataset and the attributes of the dataset were used as the domain.

Also, the input image was resized to 128 × 128.

In each model, the dimension of the latent variable and the learning rate were randomly changed, the DIS was calculated several times, and the average and the standard deviation were obtained.

The DualVAE obtained a higher DIS than the other models.

We transferred the images by interpolating between the original and the target domain images.

We calculated the following vector w i : DISPLAYFORM0 Here, w i was constrained by giving it the same norm as z to retain as much of the original features as possible.

By changing λ and decoding w i , five images were determined to represent unideal to ideal reconstructions for each of the three sample users (i = 14, 18, and 32), and interpolation was performed to approach the ideal image x i in FIG2 .

In addition, we have visualized transferred images of the 40 attributes by the proposed method and other models in FIG3 .3.

Although StarGAN and UFDN retained the characteristics of the original image considerably, it was qualitatively understood that domain transfer was not good especially when the number of domains was large like 40 attributes.

Variational domain adaptation, which is a unified framework for learning multiple distributions in a single network, is proposed in this study.

Our framework uses one known source as a prior p(x) and binary discriminator p(D i |x), thereby discriminating the target domain D i from the others; this is in contrast with the existing frameworks in which samples undergo domain transfer through deep generative models.

Consequently, our framework regards the target as a posterior that is characterized through Bayesian inference, p(x|D i ) ∝ p(D i |x)p(x).

This was exhibited by the proposed DualVAE.

The major feature of the DualVAE is domain embedding, which is a powerful tool that encodes all the domains and the samples obtained from the prior into normal distributions in the same latent space as that learned by a unified network through variational inference.

In the experiment, we applied our framework and model to a multi-domain image generation task.

celebA and face image data that were obtained based on evaluation by 60 users were used, and the result revealed that the DualVAE method outperformed StarGAN and UFDN.Several directions should be considered for future research.

First, we intend to expand DualVAEs for learning in complex domains, such as high-resolution images with several models, for example, glow BID7 .

Second, we will perform an experiment to consider wider domains with respect to beauty.

We expect that our proposed method will contribute to society in a number of ways and will help to deal with the paradigm of multiple contexts-multimodal, multi-task, and multi-agent contexts.

We visualized the latent space Z of VAE and DualVAE.

VAE differs from DualVAE methodology because evaluation regression is not conducted during training.

For each model, we can achieve 5500 latent vectors of 63 dimensions by encoding 5500 images from SCUT-FBP5500.

We obtained a scatter plot after using UMAP BID15 to reduce the number of dimensions to two.

The average score is indicated by colors ranging from red to blue.

As can be observed from the UMAP of DualVAE, the gradient of the score is learned, and it represents the user vector(domain embedding vector) in FIG4 .

Although the Inception Score BID18 ) is a score for measuring generated images, it can only measure the diversity of the images, and it is not for evaluating domain transfer of the images.

Therefore, we proposed using a DIS, which is a score for evaluating the transformation of images into multiple target domains.

DIS is a scalar value, and it is evaluated by the sum of two elements.

The first is whether the domain transfer of the original image has been successful (transfer score), and the second is whether the features other than the transferred domain are retained (reconstruction score).We calculated the DIS using Algorithm 2.

First, we assumed that there were N domains and we knew which domain each image belongs to.

We fine-tuned Inceptionv3 BID19 using images X as inputs and domains as outputs.

To enable the model to classify the images as the domains, we replaced the last layer of the model in a new layer which had N outputs.

Second, we transferred test images into N domains using Equation 10 and loaded the transferred images into the Inceptionv3 pretrained above.

Through this process we got N × N matrix for every original image, because one image was transferred into N domains and each domain image was mapped to N-dim vector.

We then mapped the original image into N-dim vector using Inceptionv3, and subtracted this vector from each row of the abobe N × N matrix.

We named this matrix M. The key points are (1) the diagonal elements of M should be large because we transferred the original image into the diagonal domains, and (2) the off-diagonal elements of M should be small because the transferred images should preserve original features as possible.

In a later subsection, we will directly visualize these two elements and evaluate models.

Require: observation x ∈ X , Inceptionv3 f , domain transfer model m. DISPLAYFORM0 In the Algorithm, abs denotes taking the absolute value, diag denotes taking the diagonal elements of the matrix, notdiag denotes taking the non-diagonal elements, avg denotes taking the mean of multiple values.

This section shows further results of TAB0 , the experimental result for domain adaptation over 40 domains made from CelebA. In the experimental setting above, we use attributes in CelebA as a domain, the setting is used by several studies with domain adaptation BID1 .

The result shows DualVAE only learns 40 domains in one network, which indicates DualVAE is an easy way to learn over 10 domains.

Next, we show several experimental results when we change the parameters of the models.

Because StarGAN uses GAN, the learning rate parameter is not robust, thus the learning is not conducted well.

Moreover, celebA has 40 domains which are too many for StarGAN, and this can also be considered as one of the reasons that learning is not conducted well.

Because reconstruction is conducted well, rs in Algorithm 2 becomes larger than that of DualVAE.

On the other hand, domain transfer is not conducted properly, ts in Algorithm 2 becomes extremely small compares to that of DualVAE.

Therefore, as we can see from TAB0 , DIS becomes a very small value.

Next, we conduct domain transfer experiments using the MNIST dataset.

In this experiment, we demonstrated that it is possible to transfer the image into another label (domain), while not compromising the style of the original image.

We also plotted the relation with DIS when labels are sparse.

Moreover, we showed in subsection I.1 it is possible to transfer to another domain step by step.

DISPLAYFORM0

By reducing the dimensions of the 60 domain embedding vectors from 63 to 2 using UMAP BID15 , the domain embedding vectors were visualized by means of a scatter plot.

Furthermore, x i was visualized by decoding samples from the domain distribution.

Figure 9 : Scatter plot of the domain embedding vectors, and several decoded images of the samples from each domain.

Six z i from the target domain distribution and output x i were decoded.

Furthermore, z 0 from the source domain data distribution and output x 0 was also decoded.

In this chapter, we show it is possible to conduct arithmetic operations among domains.

For example, suppose we learned the embedding vector of a charming image domain for each single person.

We can output the charming image for the group of people as an entity without learning simply by taking the average value of the domain embedding vectors.

Denote Community preference as f I , personal evaluation model as DISPLAYFORM0 where,μ = (1/|I|) i∈I µ i , which is the average of domain embedding vectors.

Moreover, i is the index denoting the domain (person), I is the number of domains, and z(x)

is the latent vector of image x.

, since the domain embedding vectors are linearly functional, by taking the inner product of the average of these vectorsμ and the latent vector z, the average of personal evaluation (evaluation of the community) can be obtained.

Therefore, by substituting µ i forμ in Equation 10, we can reconstruct the face images with high a high degree of community evaluation.

We reconstructed for higher (and lower) evaluation using 10 face images from both genders.

Each image enjoys higher evaluation to the right.

We can see that gradually the caving becomes deep, the beard disappears, the eyes become bigger and the outline becomes sharp FIG0 .

The section tells the proposed method, DualVAE, is a natural generalization from probabilistic Matrix Factorization (PMF) BID17 , proposed in ten years ago.

PMF is used in several application area, mainly collaborative filtering algorithm, which are typical recommendation algorithms.

PMF learns the user matrix U ∈ R K×N and the item matrix V ∈ R K×J that can restore the evaluation matrix.

Here, r ij is the evaluation value of item j by user i, the evaluation matrix is denoted as R ∈ R I×J .

Moreover, the column vector of the user matrix U and the item matrix V are denoted as u i ,v j respectively.

K is the dimension of these vectors, N is the number of users, J is the number of items.

I ij is the indicator function that takes the value 1 when evaluation r ij exists and 0 otherwise.

The log likelihood of PMF is DISPLAYFORM0 Our objective is to find the u i , v j that maximizes the above.

Relationship to DualVAE DualVAE is an end-to-end coupling of VAE and PMF.

We could see DualVAE as PMF extended to a generative model.

u i in Equation 12 corresponds to the domain embedding vector in DVAE, v j corresponds to the latent vector in DVAE, r ij corresponds to the likelihood that item j belongs to domain i.

We experimentally show that the DualVAE outperformed the non-end-to-end coupling.

We compared two models.

One is the model trained to regress evaluation of the image end-to-end by calculating inner product of hidden representation of VAE and domain embedding (DVAE).

The other is the model which learns hidden representation of VAE followed by learning to regress evaluation by inner product like above (VAE-PMF).

We used SCUTFBP-5500 FIG0 dataset, and validated it into 5000 images with 60 evaluators and 500 test images with 60 evaluators.

We quantitatively compared these two models in terms of Root Mean Square Error (RMSE) of model prediction and reconstruction error of test images.

The result suggests that DualVAE achieved a much smaller RMSE.

Moreover, though DualVAE constrained its hidden representation to regress evaluation, the reconstruction error was almost the same as VAE-PMF.

This suggests that DualVAE can generate as clear images as vanilla VAE.

Figure 11: RMSE and Reconstruction loss.

DualVAE is far superior to VAE in classification accuracy, and there is almost no difference in reconstruction error between them.

In addition to generalization capability, another benefit from PMF is robustness to sparsity as PMF is robust to a matrix with many missing values.

We will experimentally demonstrate that DualVAE is also robust with respect to sparse labels.

We calculate the rs and ts when applying Algorithm 2 on 160 celebA test images, and plot the below figure when we change the missing ratio of celeA's domain labels and the λ in Equation 10.

From FIG0 , keeping the characteristic of the upper right plots, it is possible to conduct domain transfer at the same time.

Moreover, the method is strong on the sparseness of domain labels, and DIS does not drop even when 90 of the labels are missing.

On the other hand, we show that StarGAN is not as robust as DualVAE with respect to sparseness.

When 90 of domain labels are missing, StarGAN cannot learn at all and generates identical images.

Under review as a conference paper at ICLR 2019 (b) s = 0.9.

All identical images are generated, and domain transfer is not properly conducted.

We conducted a comparison experiment with the existing methods when changing α(= σ −2 ) in Equation 9.

Here, the number of domains was set to 40.

As you can see from the results below, it turns out that the performance of DualVAE is robust to α.

The section shows three models used in tasks of domain adaptation over three types of domains: environment, attribute and class.

Environment First, we describe the experimental setting for domain transfer to the ideal image of each individual.

We assumed that the beauty criterion required for evaluating the facial images de-pends on the gender of a person in the target image.

Therefore, we added the gender information to the images.

For this purpose, we applied CGAN BID16 to VAE.

We normalized the scoring in [−1, 1] to accelerate the learning.

Subsequently, we considered the specific model structure of DualVAE.

Both the input and output images were RGB images, x ∈ R 256×256×3 .

We used convolution networks for the encoder and stride 2 for convolution and no pooling.

Convolution, batch normalization BID4 , and LeakyReLU were repeated four times and were subsequently connected to fully connected layers.

Further, after batch normalization and LeakyReLU layers, a 63-dimensional latent variable was obtained.

The decoder exhibited a completely symmetric shape with deconvolution layers instead of convolution layers.

Furthermore, as the gender attribute, we set 0 as female and 1 as male.

We added an image x ∈ R 256×256×1 comprising 0 or 1 data as the input of the encoder and a scalar of 0 or 1 for gender to the latent variable, which was the input to the decoder.

The detailed structure is in Structure A of TAB2 .

We optimized DualVAE on SCUT-FBP5500.

Because there were no face evaluation data in celebA, we only used it to optimize VAE.

Learning was alternatively realized using these two datasets.

We show the image example of SCUT-FBP5500 BID10 .

From FIG0 , we can see the evaluation value depends on each person.

Attribute Next, in comparative experiment with several models, domain transfer was performed with only celebA data and domain number of 40, 20, 10, and 5.

We experimented with several parameters of the models.

In particular, the dimensions of the latent variable and the learning rates were randomly selected.

Both the input and output images were RGB images, x ∈ R 128×128×3 .

The detailed structure is in Structure B of TAB2 .Class Finally, we describe the experimental setting of domain transfer in the MNIST dataset.

This experimental result is stated in the subsection C.2.

Both the input and output images were gray images, x ∈ R 28×28×1 .

The detailed structure is in Structure C of TAB2 .

The results below shows result from domain adaptation performed by DualVAE by randomlysampled images from two datasets: MNIST and CelebA.

Figure 18: DualVAE stably transfers samples across 10 domains while domain-irrelevant features (e.g., style) are kept.

<|TLDR|>

@highlight

This paper proposes variational domain adaptation, a uniﬁed, scalable, simple framework for learning multiple distributions through variational inference