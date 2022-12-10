We propose an algorithm, guided variational autoencoder (Guided-VAE), that is able to learn a controllable generative model by performing latent representation disentanglement learning.

The learning objective is achieved by providing signal to the latent encoding/embedding in VAE without changing its main backbone architecture, hence retaining the desirable properties of the VAE.

We design an unsupervised and a supervised strategy in Guided-VAE and observe enhanced modeling and controlling capability over the vanilla VAE.

In the unsupervised strategy, we guide the VAE learning by introducing a lightweight decoder that learns latent geometric transformation and principal components; in the supervised strategy, we use an adversarial excitation and inhibition mechanism to encourage the disentanglement of the latent variables.

Guided-VAE enjoys its transparency and simplicity for the general representation learning task, as well as disentanglement learning.

On a number of experiments for representation learning, improved synthesis/sampling, better disentanglement for classification, and reduced classification errors in meta learning have been observed.

The resurgence of autoencoders (AE) (LeCun, 1987; Bourlard & Kamp, 1988; Hinton & Zemel, 1994) is an important component in the rapid development of modern deep learning .

Autoencoders have been widely adopted for modeling signals and images (Poultney et al., 2007; Vincent et al., 2010) .

Its statistical counterpart, the variational autoencoder (VAE) (Kingma & Welling, 2014) , has led to a recent wave of development in generative modeling due to its two-in-one capability, both representation and statistical learning in a single framework.

Another exploding direction in generative modeling includes generative adversarial networks (GAN) Goodfellow et al. (2014) , but GANs focus on the generation process and are not aimed at representation learning (without an encoder at least in its vanilla version).

Compared with classical dimensionality reduction methods like principal component analysis (PCA) (Hotelling, 1933; Jolliffe, 2011) and Laplacian eigenmaps (Belkin & Niyogi, 2003) , VAEs have demonstrated their unprecedented power in modeling high dimensional data of real-world complexity.

However, there is still a large room to improve for VAEs to achieve a high quality reconstruction/synthesis.

Additionally, it is desirable to make the VAE representation learning more transparent, interpretable, and controllable.

In this paper, we attempt to learn a transparent representation by introducing guidance to the latent variables in a VAE.

We design two strategies for our Guided-VAE, an unsupervised version ( Fig. 1 .a) and a supervised version ( Fig. 1.b) .

The main motivation behind Guided-VAE is to encourage the latent representation to be semantically interpretable, while maintaining the integrity of the basic VAE architecture.

Guided-VAE is learned in a multi-task learning fashion.

The objective is achieved by taking advantage of the modeling flexibility and the large solution space of the VAE under a lightweight target.

Thus the two tasks, learning a good VAE and making the latent variables controllable, become companions rather than conflicts.

In unsupervised Guided-VAE, in addition to the standard VAE backbone, we also explicitly force the latent variables to go through a lightweight encoder that learns a deformable PCA.

As seen in Fig. 1 .a, two decoders exist, both trying to reconstruct the input data x: Dec main .

The main decoder, denoted as Dec main , functions regularly as in the standard VAE (Kingma & Welling, 2014) ; the secondary decoder, denoted as Dec sub , explicitly learns a geometric deformation together with a linear sub-space.

In supervised Guided-VAE, we introduce a subtask for the VAE by forcing one latent variable to be discriminative (minimizing the classification error) while making the rest of the latent variable to be adversarially discriminative (maximizing the minimal classification error).

This subtask is achieved using an adversarial excitation and inhibition formulation.

Similar to the unsupervised Guided-VAE, the training process is carried out in an end-to-end multi-task learning manner.

The result is a regular generative model that keeps the original VAE properties intact, while having the specified latent variable semantically meaningful and capable of controlling/synthesizing a specific attribute.

We apply Guided-VAE to the data modeling and few-shot learning problems and show favorable results on the MNIST, CelebA, and Omniglot datasets.

The contributions of our work can be summarized as follows:

• We propose a new generative model disentanglement learning method by introducing latent variable guidance to variational autoencoders (VAE).

Both unsupervised and supervised versions of Guided-VAE have been developed.

• In unsupervised Guided-VAE, we introduce deformable PCA as a subtask to guide the general VAE learning process, making the latent variables interpretable and controllable.

• In supervised Guided-VAE, we use an adversarial excitation and inhibition mechanism to encourage the disentanglement, informativeness, and controllability of the latent variables.

Guided-VAE is able to keep the attractive properties of the VAE and it is easy to implement.

It can be trained in an end-to-end fashion.

It significantly improves the controllability of the vanilla VAE and is applicable to a range of problems for generative modeling and representation learning.

Related work can be discussed along several directions.

Generative model families such as generative adversarial networks (GAN) (Goodfellow et al., 2014; Arjovsky et al., 2017) and variational autoencoder (VAE) (Kingma & Welling, 2014) have received a tremendous amount of attention lately.

Although GAN produces higher quality synthesis than VAE, GAN is missing the encoder part and hence is not directly suited for representation learning.

Here, we focus on disentanglement learning by making VAE more controllable and transparent.

Disentanglement learning (Mathieu et al., 2016; Achille & Soatto, 2018; Gonzalez-Garcia et al., 2018; Jha et al., 2018) recently becomes a popular topic in representation learning.

Adversarial training has been adopted in approaches such as (Mathieu et al., 2016; .

Various methods (Peng et al., 2017; Kim & Mnih, 2018; Lin et al., 2019) have imposed constraints/regularizations/supervisions to the latent variables but these existing approaches often involve an architectural change to the VAE backbone and the additional components in these approaches are not provided as secondary decoder for guiding the main encoder.

A closely related work is the β-VAE (Higgins et al., 2017) approach in which a balancing term β is introduced to control the capacity and the independence prior.

β-TCVAE (Chen et al., 2018) further extends β-VAE by introducing a total correlation term.

From a different angle, principal component analysis (PCA) family (Hotelling, 1933; Jolliffe, 2011; Candès et al., 2011) can also be viewed as representation learning.

Connections between robust PCA (Candès et al., 2011) and VAE (Kingma & Welling, 2014) have been observed (Dai et al., 2018) .

Although being a widely adopted method, PCA nevertheless has limited modeling capability due to its linear subspace assumption.

To alleviate the strong requirement for the input data being prealigned, RASL (Peng et al., 2012 ) deals with unaligned data by estimating a hidden transformation to each input.

Here, we take the advantage of the transparency of PCA and the modeling power of VAE by developing a sub-encoder (see Fig. 1 .a), deformable PCA, that guides the VAE training process in an integrated end-to-end manner.

After training, the sub-encoder can be removed by keeping the main VAE backbone only.

To achieve disentanglement learning in supervised Guided-VAE, we encourage one latent variable to directly correspond to an attribute while making the rest of the variables uncorrelated.

This is analogous to the excitation-inhibition mechanism (Yizhar et al., 2011) or the explaining-away (Wellman & Henrion, 1993) phenomena.

Existing approaches (Liu et al., 2018; Lin et al., 2019) impose supervision as a conditional model for an image translation task, whereas our supervised Guided-VAE model targets the generic generative modeling task by using an adversarial excitation and inhibition formulation.

This is achieved by minimizing the discriminative loss for the desired latent variable while maximizing the minimal classification error for the rest of the variables.

Our formulation has connection to the domain-adversarial neural networks (DANN) (Ganin et al., 2016) but the two methods differ in purpose and classification formulation.

Supervised Guided-VAE is also related to the adversarial autoencoder approach Makhzani et al. (2016) but the two methods differ in objective, formulation, network structure, and task domain.

In (Ilse et al., 2019) , the domain invariant variational autoencoders method (DIVA) differs from ours by enforcing disjoint sectors to explain certain attributes.

Our model also has connections to the deeply-supervised nets (DSN) (Lee et al., 2015) where intermediate supervision is added to a standard CNN classifier.

There are also approaches (Engel et al., 2018; Bojanowski et al., 2018) in which latent variables constraints are added but they have different formulations and objectives than Guided-VAE.

Recent efforts in fairness disentanglement learning (Creager et al., 2019; Song et al., 2018 ) also bear some similarity but there is still with a large difference in formulation.

In this section, we present the main formulations of our Guided-VAE models.

The unsupervised Guided-VAE version is presented first, followed by introduction of the supervised version.

Following the standard definition in variational autoencoder (VAE) (Kingma & Welling, 2014) , a set of input data is denoted as X = (x 1 , ..., x n ) where n denotes the number of total input samples.

The latent variables are denoted by vector z. The encoder network includes network and variational parameters φ that produces variational probability model q φ (z|x).

The decoder network is parameterized by θ to reconstruct samplex = f θ (z).

The log likelihood log p(x) estimation is achieved by maximizing the Evidence Lower BOund (ELBO) (Kingma & Welling, 2014) :

The first term in eq.

(1) corresponds to a reconstruction loss q φ (z|x) × ||x − f θ (z)|| 2 dz (the first term is the negative of reconstruction loss between input x and reconstructionx) under Gaussian parameterization of the output.

The second term in eq. (1) refers to the KL divergence between the variational distribution q φ (z|x) and the prior distribution p(z).

The training process thus tries to find the optimal (θ, φ) * such that:

In our unsupervised Guided-VAE, we introduce a deformable PCA as a secondary decoder to guide the VAE training.

An illustration can be seen in Fig. 1 .a.

This secondary decoder is called Dec sub .

Without loss of generality, we let z = (z def , z cont ).

z def decides a deformation/transformation field, e.g. an affine transformation denoted as τ (z def ).

z cont determines the content of a sample image for transformation.

The PCA model consists of K basis B = (b 1 , ..., b K ).

We define a deformable PCA loss as:

where • defines a transformation (affine in our experiments) operator decided by τ (z def ) and

2 can be optionally added to force the basis to be unit vectors.

We follow the spirit of the PCA optimization and a general formulation for learning PCA, which can be found in (Candès et al., 2011) .

To keep the simplicity of the method we learn a fixed basis function B and one can also adopt a probabilistic PCA model (Tipping & Bishop, 1999) .

Thus, learning unsupervised Guided-VAE becomes:

For training data X = (x 1 , ..., x n ), suppose there exists a total of T attributes with ground-truth labels.

The t-th attribute, let z = (z t , z rst t ) where z t defines a scalar variable deciding to decide the t-th attribute and z rst t represents remaining latent variables.

Let y t (x i ) be the ground-truth label for the t−th attribute of sample x i ; y t (x i ) ∈ {−1, +1}. For each attribute, we use an adversarial excitation and inhibition method with term:

which is a hinge term.

This is an excitation process since we want latent variable z t to directly correspond to the attribute label.

Notice the − sign before the summation since this term will be combined with eq. (1) for maximization.

where C t (z rst t ) refers to classifier making a prediction for the t-th attribute using the remaining latent variables z rst t .

− log p Ct (y = y(x)|z rst t ) is a cross-entropy term for minimizing the classification error in eq. (6).

This is an inhibition process since we want the remaining variables z rst t as independent as possible to the attribute label.

Note that the term L Inhibition (φ, t) within eq. (7) for maximization is an adversarial term to make z rst t as uninformative to attribute t as possible, by making the best possible classifier C t to be undiscriminative.

The formulation of eq. (7) bears certain similarity to that in domain-adversarial neural networks (Ganin et al., 2016) in which the label classification is minimized with the domain classifier being adversarially maximized.

Here, however, we respectively encourage and discourage different parts of the features to make the same type of classification.

In this section, we first present qualitative results demonstrating our proposed unsupervised Guided-VAE (Figure 1a ) capable of disentangling latent embedding in a more favourable way than VAE and previous disentangle methods (Higgins et al., 2017; Dupont, 2018) on MNIST dataset (LeCun et al., 2010) .

We also show that our learned latent representation can be later used to improve classification performance.

Next, we extend this idea to a supervised guidance approach in an adversarial excitation and inhibition fashion, where a discriminative objective for certain image properties is given (Figure 1b) on the CelebA dataset (Yang et al., 2015) .

Further, we show that our method can be applied to the few-shot classification tasks, which achieves competitive performance on Omniglot dataset proposed by Vinyals et al. (2016) .

4.1 UNSUPERVISED GUIDED-VAE 4.1.1 QUALITATIVE EVALUATION We present qualitative results on MNIST dataset by traversing latent variables received affine transformation guiding signal.

Here, we applied the Guided-VAE with the bottleneck size of 10 (i.e. the latent variables z ∈ R 10 ).

The first latent variable z 1 represents the rotation information and the second latent variable z 2 represents the scaling information.

The rest of the latent variables z 3:10 represent the content information.

Thus, the latent variables z ∈ R 10 are represented by z = (z def , z cont ) = (z 1:2 , z 3:10 ). , β-VAE with controlled capacity increase (CCβ-VAE), Joint-VAE (Dupont, 2018) and our Guided-VAE on the MNIST dataset.

z1 and z2 in Guided-VAE are controlled.

Figure 3: PCA basis learned by the secondary decoder in unsupervised Guided-VAE.

In Figure 2 , we show traversal results of all latent variables on MNIST dataset for vanilla VAE (Kingma & Welling, 2014), β-VAE (Higgins et al., 2017) , JointVAE (Dupont, 2018) and our guided VAE (β-VAE, JointVAE results are adopted from (Dupont, 2018) ).

While β-VAE cannot generate meaningful disentangled representations, even with controlled capacity increased, JointVAE is able to disentangle class type from continuous factors.

Different from previous methods, our Guided-VAE disentangles geometry properties (z 1 and z 2 ) like rotation angle and stroke thickness from the rest content information z 3:10 .

In Figure 3 , we visualize the basis B = (b 1 , ..., b 8 ) in the PCA part of Dec sub .

The basis primarily capture the content information.

For a quantitative evaluation, we first compare the reconstruction error among different models on the MNIST dataset.

In this experiment, we set the bottleneck size to 8 in Guided-VAE and use three settings for the deformation/transformation: Rotation, scaling, and both.

In Guided-VAE (Rotation) or Guided-VAE (Scaling), we take the first latent variable z 1 to represent the rotation or the scaling information.

In Guided-VAE (Rotation and Scaling), we use the first and second latent variables (z 1 and z 2 ) to represent rotation and scaling respectively.

As Table 1 shows, our reconstruction loss is on par with vanilla VAE, whereas the previous disentangling method (β-VAE) has higher loss.

Our proposed method is able to achieve added disentanglement while not sacrificing reconstruction capability over vanilla VAE.

In addition, we perform classification tasks on latent embeddings of different models.

Specifically, for each data point (x, y), we use the pre-trained VAE model to obtain the value of latent variable z given input image x.

Here z is a d z -dim vector.

We then train a linear classifier f (·) on the embedding-label pairs {(z, y)} in order to predict the class of digits.

For the Guided-VAE, we disentangle the latent variables z into deformation variables z def and content variables z cont with same dimensions (i.e. d z def = d zcont ) and use affine transformation as τ (z def ).

We compare the classification errors of different models under multiple choices of dimensions of the latent variables in Table 2 .

It shows that generally higher dimensional latent variables result in lower classification errors.

Our Guided-VAE method compares favourably over vanilla VAE and β-VAE.

Moreover, we attempt to validate the effectiveness of disentanglement in Guided-VAE.

We follow the same classification tasks above but use different parts of latent variables as input features for the classifier f (·): We may choose the deformation variables z def , the content variables z cont , or the whole latent variables z as the input feature vector.

To reach a fair comparison, we keep the same dimensions for the deformation variables z def and the content variables z cont .

Table 3 shows that the classification errors on z cont are significantly lower than the ones on z def , which indicates the success of disentanglement since the content variables should determine the class of digits while the deformation variables should be invariant to the class.

In addition, when the dimensions of latent variables z are higher, the classification errors on z def increase while the ones on z cont decrease, indicating a better disentanglement between deformation and content.

We first present qualitative results on the CelebA dataset by traversing latent variables of attributes.

We select three labeled attributes (emotion, gender and color) in the CelebA dataset as supervised guidance objectives.

The bottleneck size is set to 16.

We use the first three latent variables z 1 , z 2 , z 3 to represent the attribute information and the rest z 4:16 to represent the content information.

During evaluation, we choose z t ∈ {z 1 , z 2 , z 3 } while keeping the remaining latent variables z rst t fixed.

Then we obtain a set of images through traversing from the image with t-th attribute to the image without t-th attribute (e.g. smiling to non-smiling) and compare them over methods.

Figure 4 shows the traversal results for β-VAE and our Guided-VAE.

β-VAE performs decently for the controlled attribute change, but the individual z in β-VAE is not fully entangled or disentangled with the attribute.

Guided-VAE has a better disentanglement for latent variables and is able to better isolate the attributes w.r.t.

the corresponding latent variables.

In supervised Guided-VAE, we train a classifier to predict the attributes by using the disentangled attribute latent variable z t or the rest of latent variables z rst t as input features.

We perform adversarial excitation and inhibition by encouraging the target latent variable to best predict the corresponding t-th attribute and discouraging the rest of the variables for the prediction of that attribute. (left) shows that the classification errors on z t is significantly lower than the ones on z rst t , which indicates the effectiveness of disentanglement during the training procedure.

fiers for attribute classification) prediction for being negatives on the generated images.

We traverse z1 (gender) and z2 (smile) separately to generate images for the classification test.

Each latent z is traversed from −3.0 to 3.0 with 0.1 as the stride length.

Furthermore, we attempt to validate that the generated images from the supervised Guided-VAE can be actually controlled by the disentangled attribute variables.

Thus, we pre-train an external binary classifier for t-th attribute on the CelebA training set and then use this classifier to test the generated images from Guided-VAE.

Each test includes 10, 000 generated images randomly sampled on all latent variables except for the particular latent variable z t we decide to control.

As Figure 5 (right) shows, we can draw the confidence-z curves of the t-th attribute where z = z t ∈ [−3.0, 3.0].

For the gender and the smile attributes, it can be seen that the corresponding z t is able to enable (z t < −1) and disable (z t > 1) the attribute of the generated image.

Besides, for all the attributes, the probability monotonically decreases when z t increases, which shows the controlling ability of the t-th attribute by tuning the corresponding latent variable z t .

Previously, we have shown that Guided-VAE can generate images and be used as representation to perform classification task.

In this section, we will apply the proposed method to few-shot classification problem.

Specifically, we use our adversarial excitation and inhibition method in the Neural Statistician (Edwards & Storkey, 2017) by adding a supervised guidance network after the statistic network.

The supervised guidance signal is the label of each input.

We also apply the Mixup method (Zhang et al., 2018) in the supervised guidance network.

However, we couldn't reproduce exact reported results in the Neural Statistician, which is also indicated in Korshunova et al. (2018) .

For comparison, we mainly consider the Matching Nets (Vinyals et al., 2016) and Bruno (Korshunova et al., 2018 ).

Yet it cannot outperform Matching Nets, our proposed Guided-VAE reaches equivalent performance as Bruno (discriminative), where a discriminative objective is fine-tuned to maximize the likelihood of correct labels.

We conduct a series of ablation experiments to validate our proposed Guided-VAE model.

In this part, we conduct an experiment by excluding the geometry-guided part from the unsupervised Guided-VAE.

In this way, the nudging decoder is just a PCA-like decoder but not a deformable PCA.

The setting of this experiment is exactly same as described in the unsupervised Guided-VAE section.

The bottleneck size of our model is set to 10 of which the first two latent variables z 1 , z 2 represent the rotation and scaling information separately.

In the ablation part, we drop off the geometry-guided part so all 10 latent variables are controlled by the PCA-like light decoder.

In this part, we conduct an experiment of using the adversarial excitation method.

We design the experiment using the exact same setting described in the supervised Guided-VAE part.

As Figure 7 shows, though the traversal results still show the traversed results on some latent variables.

The results from the adversarial excitation method outperforms the results from the discriminative method.

While traversing the latent variable controlling the smiling information, the left part (a) also changes in the smiling status but it's controlled by another latent variable. shows the traversed images from the supervised Guided-VAE without adversarial inhibition.

The right part shows the traversed images from the supervised Guided-VAE using adversarial excitation and inhibition.

Both images are traversed on the latent variable that is supposed to control the gender information.

In this paper we have presented a new representation learning method, guided variational autoencoder (Guided-VAE), for disentanglement learning.

Both versions of Guided-VAE utilize lightweight guidance to the latent variables to achieve better controllability and transparency.

Improvements on disentanglement, image traversal, and meta-learning over the competing methods are observed.

Guided-VAE maintains the backbone of VAE and can be applied to other generative modeling applications.

A.1 PERCENTAGE OF DATA PARTICIPATING IN THE GUIDED SUB-NETWORK In this part, we design an experiment to show how the percentage of data participating in the guided sub-network can influence the final prediction.

We conduct this ablation study on MNIST using unsupervised Guided-VAE.

We change the percentage of data participating in the guided sub-network and then present the classification accuracy using the first half latent variables (represent geometry information) and the second half latent variables (represent content information) separately.

From Figure 8 , we observe consistent improvement for the last half latent variables when adding more samples to guide sub-network.

This indicates adding more samples can improve disentanglement, which causes that more content information is represented in the second half latent variables.

Similarity, the improvement of disentanglement leads the first half latent variables can represent more geometry information, which is indiscriminative for classes.

We also observe accuracy improvement when large amount of samples are used to train sub-network.

We hypothesize this is because geometry information is still partially affected by classes.

<|TLDR|>

@highlight

Learning a controllable generative model by performing latent representation disentanglement learning.