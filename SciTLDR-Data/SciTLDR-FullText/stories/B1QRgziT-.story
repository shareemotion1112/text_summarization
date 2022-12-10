One of the challenges in the study of generative adversarial networks is the instability of its training.

In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator.

Our new normalization technique is computationally light and easy to incorporate into existing implementations.

We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques.

Generative adversarial networks (GANs) BID11 have been enjoying considerable success as a framework of generative models in recent years, and it has been applied to numerous types of tasks and datasets BID15 .

In a nutshell, GANs are a framework to produce a model distribution that mimics a given target distribution, and it consists of a generator that produces the model distribution and a discriminator that distinguishes the model distribution from the target.

The concept is to consecutively train the model distribution and the discriminator in turn, with the goal of reducing the difference between the model distribution and the target distribution measured by the best discriminator possible at each step of the training.

GANs have been drawing attention in the machine learning community not only for its ability to learn highly structured probability distribution but also for its theoretically interesting aspects.

For example, BID26 BID37 BID24 revealed that the training of the discriminator amounts to the training of a good estimator for the density ratio between the model distribution and the target.

This is a perspective that opens the door to the methods of implicit models BID24 BID36 that can be used to carry out variational optimization without the direct knowledge of the density function.

A persisting challenge in the training of GANs is the performance control of the discriminator.

In high dimensional spaces, the density ratio estimation by the discriminator is often inaccurate and unstable during the training, and generator networks fail to learn the multimodal structure of the target distribution.

Even worse, when the support of the model distribution and the support of the target distribution are disjoint, there exists a discriminator that can perfectly distinguish the model distribution from the target .

Once such discriminator is produced in this situation, the training of the generator comes to complete stop, because the derivative of the so-produced discriminator with respect to the input turns out to be 0.

This motivates us to introduce some form of restriction to the choice of the discriminator.

In this paper, we propose a novel weight normalization method called spectral normalization that can stabilize the training of discriminator networks.

Our normalization enjoys following favorable properties.• Lipschitz constant is the only hyper-parameter to be tuned, and the algorithm does not require intensive tuning of the only hyper-parameter for satisfactory performance.• Implementation is simple and the additional computational cost is small.

In fact, our normalization method also functioned well even without tuning Lipschitz constant, which is the only hyper parameter.

In this study, we provide explanations of the effectiveness of spectral normalization for GANs against other regularization techniques, such as weight normalization BID31 , weight clipping , and gradient penalty BID12 .

We also show that, in the absence of complimentary regularization techniques (e.g., batch normalization, weight decay and feature matching on the discriminator), spectral normalization can improve the sheer quality of the generated images better than weight normalization and gradient penalty.

In this section, we will lay the theoretical groundwork for our proposed method.

Let us consider a simple discriminator made of a neural network of the following form, with the input x: DISPLAYFORM0 where θ := {W 1 , . . .

, W L , W L+1 } is the learning parameters set, DISPLAYFORM1 1×d L , and a l is an element-wise non-linear activation function.

We omit the bias term of each layer for simplicity.

The final output of the discriminator is given by DISPLAYFORM2 where A is an activation function corresponding to the divergence of distance measure of the user's choice.

The standard formulation of GANs is given by DISPLAYFORM3 where min and max of G and D are taken over the set of generator and discriminator functions, respectively.

The conventional form of V (G, D) BID11 is given by E x∼q data [log D(x)] + E x ∼p G [log(1 − D(x ))], where q data is the data distribution and p G is the (model) generator distribution to be learned through the adversarial min-max optimization.

The activation function A that is used in the D of this expression is some continuous function with range [0, 1] (e.g, sigmoid function).

It is known that, for a fixed generator G, the optimal discriminator for this form of V (G, D) is given by D * G (x) := q data (x)/(q data (x) + p G (x)).

The machine learning community has been pointing out recently that the function space from which the discriminators are selected crucially affects the performance of GANs.

A number of works BID37 Qi, 2017; BID12 advocate the importance of Lipschitz continuity in assuring the boundedness of statistics.

For example, the optimal discriminator of GANs on the above standard formulation takes the form DISPLAYFORM4 and its derivative DISPLAYFORM5 can be unbounded or even incomputable.

This prompts us to introduce some regularity condition to the derivative of f (x).A particularly successful works in this array are (Qi, 2017; BID12 , which proposed methods to control the Lipschitz constant of the discriminator by adding regularization terms defined on input examples x.

We would follow their footsteps and search for the discriminator D from the set of K-Lipschitz continuous functions, that is, DISPLAYFORM6 where we mean by f Lip the smallest value M such that f (x) − f (x ) / x − x ≤ M for any x, x , with the norm being the 2 norm.

While input based regularizations allow for relatively easy formulations based on samples, they also suffer from the fact that, they cannot impose regularization on the space outside of the supports of the generator and data distributions without introducing somewhat heuristic means.

A method we would introduce in this paper, called spectral normalization, is a method that aims to skirt this issue by normalizing the weight matrices using the technique devised by BID41 .

Our spectral normalization controls the Lipschitz constant of the discriminator function f by literally constraining the spectral norm of each layer g : h in → h out .

By definition, Lipschitz norm g Lip is equal to sup h σ(∇g(h)), where σ(A) is the spectral norm of the matrix A (L 2 matrix norm of A) DISPLAYFORM0 which is equivalent to the largest singular value of A. Therefore, for a linear layer g(h) = W h, the norm is given by DISPLAYFORM1 If the Lipschitz norm of the activation function a l Lip is equal to 1 1 , we can use the inequality g 1 •g 2 Lip ≤ g 1 Lip · g 2 Lip to observe the following bound on f Lip : DISPLAYFORM2 Our spectral normalization normalizes the spectral norm of the weight matrix W so that it satisfies the Lipschitz constraint σ(W ) = 1:W DISPLAYFORM3 If we normalize each W l using (8), we can appeal to the inequality (7) and the fact that σ W SN (W ) = 1 to see that f Lip is bounded from above by 1.Here, we would like to emphasize the difference between our spectral normalization and spectral norm "regularization" introduced by BID41 .

Unlike our method, spectral norm "regularization" penalizes the spectral norm by adding explicit regularization term to the objective function.

Their method is fundamentally different from our method in that they do not make an attempt to 'set' the spectral norm to a designated value.

Moreover, when we reorganize the derivative of our normalized cost function and rewrite our objective function (12), we see that our method is augmenting the cost function with a sample data dependent regularization function.

Spectral norm regularization, on the other hand, imposes sample data independent regularization on the cost function, just like L2 regularization and Lasso.

As we mentioned above, the spectral norm σ(W ) that we use to regularize each layer of the discriminator is the largest singular value of W .

If we naively apply singular value decomposition to compute the σ(W ) at each round of the algorithm, the algorithm can become computationally heavy.

Instead, we can use the power iteration method to estimate σ(W ) (Golub & BID10 BID41 .

With power iteration method, we can estimate the spectral norm with very small additional computational time relative to the full computational cost of the vanilla GANs.

Please see Appendix A for the detail method and Algorithm 1 for the summary of the actual spectral normalization algorithm.

The gradient 2 ofW SN (W ) with respect to W ij is: DISPLAYFORM0 where E ij is the matrix whose (i, j)-th entry is 1 and zero everywhere else, and u 1 and v 1 are respectively the first left and right singular vectors of W .

If h is the hidden layer in the network to be transformed byW SN , the derivative of the V (G, D) calculated over the mini-batch with respect to W of the discriminator D is given by: DISPLAYFORM1 DISPLAYFORM2 represents empirical expectation over the mini-batch.

DISPLAYFORM3 We would like to comment on the implication of (12).

The first termÊ δh T is the same as the derivative of the weights without normalization.

In this light, the second term in the expression can be seen as the regularization term penalizing the first singular components with the adaptive regularization coefficient λ.

λ is positive when δ andW SN h are pointing in similar direction, and this prevents the column space of W from concentrating into one particular direction in the course of the training.

In other words, spectral normalization prevents the transformation of each layer from becoming to sensitive in one direction.

We can also use spectral normalization to devise a new parametrization for the model.

Namely, we can split the layer map into two separate trainable components: spectrally normalized map and the spectral norm constant.

As it turns out, this parametrization has its merit on its own and promotes the performance of GANs (See Appendix E).

The weight normalization introduced by BID31 is a method that normalizes the 2 norm of each row vector in the weight matrix.

Mathematically, this is equivalent to requiring the weight by the weight normalizationW WN : DISPLAYFORM0 where σ t (A) is a t-th singular value of matrix A. Therefore, up to a scaler, this is same as the Frobenius normalization, which requires the sum of the squared singular values to be 1.

These normalizations, however, inadvertently impose much stronger constraint on the matrix than intended.

IfW WN is the weight normalized matrix of dimension DISPLAYFORM1 . .

, T , which means thatW WN is of rank one.

Similar thing can be said to the Frobenius normalization (See the appendix for more details).

Using suchW WN corresponds to using only one feature to discriminate the model probability distribution from the target.

In order to retain as much norm of the input as possible and hence to make the discriminator more sensitive, one would hope to make the norm ofW WN h large.

For weight normalization, however, this comes at the cost of reducing the rank and hence the number of features to be used for the discriminator.

Thus, there is a conflict of interests between weight normalization and our desire to use as many features as possible to distinguish the generator distribution from the target distribution.

The former interest often reigns over the other in many cases, inadvertently diminishing the number of features to be used by the discriminators.

Consequently, the algorithm would produce a rather arbitrary model distribution that matches the target distribution only at select few features.

Weight clipping ) also suffers from same pitfall.

Our spectral normalization, on the other hand, do not suffer from such a conflict in interest.

Note that the Lipschitz constant of a linear operator is determined only by the maximum singular value.

In other words, the spectral norm is independent of rank.

Thus, unlike the weight normalization, our spectral normalization allows the parameter matrix to use as many features as possible while satisfying local 1-Lipschitz constraint.

Our spectral normalization leaves more freedom in choosing the number of singular components (features) to feed to the next layer of the discriminator.

BID4 introduced orthonormal regularization on each weight to stabilize the training of GANs.

In their work, BID4 augmented the adversarial objective function by adding the following term: DISPLAYFORM2 While this seems to serve the same purpose as spectral normalization, orthonormal regularization are mathematically quite different from our spectral normalization because the orthonormal regularization destroys the information about the spectrum by setting all the singular values to one.

On the other hand, spectral normalization only scales the spectrum so that the its maximum will be one.

BID12 used Gradient penalty method in combination with WGAN.

In their work, they placed K-Lipschitz constant on the discriminator by augmenting the objective function with the regularizer that rewards the function for having local 1-Lipschitz constant (i.e. ∇xf 2 = 1) at discrete sets of points of the formx := x + (1 − )x generated by interpolating a samplex from generative distribution and a sample x from the data distribution.

While this rather straightforward approach does not suffer from the problems we mentioned above regarding the effective dimension of the feature space, the approach has an obvious weakness of being heavily dependent on the support of the current generative distribution.

As a matter of course, the generative distribution and its support gradually changes in the course of the training, and this can destabilize the effect of such regularization.

In fact, we empirically observed that a high learning rate can destabilize the performance of WGAN-GP.

On the contrary, our spectral normalization regularizes the function the operator space, and the effect of the regularization is more stable with respect to the choice of the batch.

Training with our spectral normalization does not easily destabilize with aggressive learning rate.

Moreover, WGAN-GP requires more computational cost than our spectral normalization with single-step power iteration, because the computation of ∇xf 2 requires one whole round of forward and backward propagation.

In the appendix section, we compare the computational cost of the two methods for the same number of updates.

In order to evaluate the efficacy of our approach and investigate the reason behind its efficacy, we conducted a set of extensive experiments of unsupervised image generation on CIFAR-10 BID35 and STL-10 (Coates et al., 2011) , and compared our method against other normalization techniques.

To see how our method fares against large dataset, we also applied our method on ILSVRC2012 dataset (ImageNet) BID29 as well.

This section is structured as follows.

First, we will discuss the objective functions we used to train the architecture, and then we will describe the optimization settings we used in the experiments.

We will then explain two performance measures on the images to evaluate the images produced by the trained generators.

Finally, we will summarize our results on CIFAR-10, STL-10, and ImageNet.

As for the architecture of the discriminator and generator, we used convolutional neural networks.

Also, for the evaluation of the spectral norm for the convolutional weight W ∈ R dout×din×h×w , we treated the operator as a 2-D matrix of dimension DISPLAYFORM0 3 .

We trained the parameters of the generator with batch normalization (Ioffe & Szegedy, 2015) .

We refer the readers to Table 3 in the appendix section for more details of the architectures.

For all methods other than WGAN-GP, we used the following standard objective function for the adversarial loss: DISPLAYFORM1 where z ∈ R dz is a latent variable, p(z) is the standard normal distribution N (0, I), and G : DISPLAYFORM2 is a deterministic generator function.

We set d z to 128 for all of our experiments.

For the updates of G, we used the alternate cost proposed by BID11 BID11 and BID38 .

For the updates of D, we used the original cost defined in (15).

We also tested the performance of the algorithm with the so-called hinge loss, which is given by DISPLAYFORM3 DISPLAYFORM4 respectively for the discriminator and the generator.

Optimizing these objectives is equivalent to minimizing the so-called reverse KL divergence : DISPLAYFORM5 .

This type of loss has been already proposed and used in BID20 ; BID36 .

The algorithm based on the hinge loss also showed good performance when evaluated with inception score and FID.

For Wasserstein GANs with gradient penalty (WGAN-GP) BID12 , we used the following objective function: DISPLAYFORM6 , where the regularization term is the one we introduced in the appendix section D.4.For quantitative assessment of generated examples, we used inception score and Fréchet inception distance (FID) BID14 .

Please see Appendix B.1 for the details of each score.

In this section, we report the accuracy of the spectral normalization (we use the abbreviation: SN-GAN for the spectrally normalized GANs) during the training, and the dependence of the algorithm's performance on the hyperparmeters of the optimizer.

We also compare the performance quality of the algorithm against those of other regularization/normalization techniques for the discriminator networks, including: Weight clipping , WGAN-GP BID12 , batch-normalization (BN) (Ioffe & Szegedy, 2015) , layer normalization (LN) BID3 , weight normalization (WN) BID31 and orthonormal regularization (orthonormal) BID4 .

In order to evaluate the stand-alone efficacy of the gradient penalty, we also applied the gradient penalty term to the standard adversarial loss of GANs (15).

We would refer to this method as 'GAN-GP'.

For weight clipping, we followed the original work and set the clipping constant c at 0.01 for the convolutional weight of each layer.

For gradient penalty, we set λ to 10, as suggested in BID12 .

For orthonormal, we initialized the each weight of D with a randomly selected orthonormal operator and trained GANs with the objective function augmented with the regularization term used in BID4 .

For all comparative studies throughout, we excluded the multiplier parameter γ in the weight normalization method, as well as in batch normalization and layer normalization method.

This was done in order to prevent the methods from overtly violating the Lipschitz condition.

When we experimented with different multiplier parameter, we were in fact not able to achieve any improvement.

For optimization, we used the Adam optimizer BID18 in all of our experiments.

We tested with 6 settings for (1) n dis , the number of updates of the discriminator per one update of the generator and (2) learning rate α and the first and second order momentum parameters (β 1 , β 2 ) of Adam.

We list the details of these settings in Table 1 in the appendix section.

Out of these 6 settings, A, B, and C are the settings used in previous representative works.

The purpose of the settings D, E, and F is to the evaluate the performance of the algorithms implemented with more aggressive learning rates.

For the details of the architectures of convolutional networks deployed in the generator and the discriminator, we refer the readers to Table 3 in the appendix section.

The number of updates for GAN generator were 100K for all experiments, unless otherwise noted.

Firstly, we inspected the spectral norm of each layer during the training to make sure that our spectral normalization procedure is indeed serving its purpose.

As we can see in the Figure 9 in the C.1, Table 1 : Hyper-parameter settings we tested in our experiments.

†, ‡ and are the hyperparameter settings following BID12 , BID38 and , respectively.

the spectral norms of these layers floats around 1-1.05 region throughout the training.

Please see Appendix C.1 for more details.

In FIG0 we show the inception scores of each method with the settings A-F.

We can see that spectral normalization is relatively robust with aggressive learning rates and momentum parameters.

WGAN-GP fails to train good GANs at high learning rates and high momentum parameters on both CIFAR-10 and STL-10.

Orthonormal regularization performed poorly for the setting E on the STL-10, but performed slightly better than our method with the optimal setting.

These results suggests that our method is more robust than other methods with respect to the change in the setting of the training.

Also, the optimal performance of weight normalization was inferior to both WGAN-GP and spectral normalization on STL-10, which consists of more diverse examples than CIFAR-10.

Best scores of spectral normalization are better than almost all other methods on both CIFAR-10 and STL-10.In TAB0 show the inception scores of the different methods with optimal settings on CIFAR-10 and STL-10 dataset.

We see that SN-GANs performed better than almost all contemporaries on the optimal settings.

SN-GANs performed even better with hinge loss (17).4 .

For the training with same number of iterations, SN-GANs fell behind orthonormal regularization for STL-10.

For more detailed comparison between orthonormal regularization and spectral normalization, please see section 4.1.2.In FIG5 we show the images produced by the generators trained with WGAN-GP, weight normalization, and spectral normalization.

SN-GANs were consistently better than GANs with weight normalization in terms of the quality of generated images.

To be more precise, as we mentioned in Section 3, the set of images generated by spectral normalization was clearer and more diverse than the images produced by the weight normalization.

We can also see that WGAN-GP failed to train good GANs with high learning rates and high momentums (D,E and F).

The generated images with GAN-GP, batch normalization, and layer normalization is shown in FIG0 in the appendix section.

We also compared our algorithm against multiple benchmark methods ans summarized the results on the bottom half of the TAB0 .

We also tested the performance of our method on ResNet based GANs used in BID12 .

Please note that all methods listed thereof are all different in both optimization methods and the architecture of the model.

Please see Table 4 and 5 in the appendix section for the detail network architectures.

Our implementation of our algorithm was able to perform better than almost all the predecessors in the performance.

Singular values analysis on the weights of the discriminator D In FIG2 , we show the squared singular values of the weight matrices in the final discriminator D produced by each method using the parameter that yielded the best inception score.

As we predicted in Section 3, the singular values of the first to fifth layers trained with weight clipping and weight normalization concentrate on a few components.

That is, the weight matrices of these layers tend to be rank deficit.

On the other hand, the singular values of the weight matrices in those layers trained with spectral normalization is more broadly distributed.

When the goal is to distinguish a pair of probability distributions on the low-dimensional nonlinear data manifold embedded in a high dimensional space, rank deficiencies in lower layers can be especially fatal.

Outputs of lower layers have gone through only a few sets of rectified linear transformations, which means that they tend to lie on the space that is linear in most parts.

Marginalizing out many features of the input distribution in such space can result in oversimplified discriminator.

We can actually confirm the effect of this phenomenon on the generated images especially in FIG5 .

The images generated with spectral normalization is more diverse and complex than those generated with weight normalization.

Training time On CIFAR-10, SN-GANs is slightly slower than weight normalization (about 110 ∼ 120% computational time), but significantly faster than WGAN-GP.

As we mentioned in Section 3, WGAN-GP is slower than other methods because WGAN-GP needs to calculate the gradient of gradient norm ∇ x D 2 .

For STL-10, the computational time of SN-GANs is almost the same as vanilla GANs, because the relative computational cost of the power iteration (18) is negligible when compared to the cost of forward and backward propagation on CIFAR-10 (images size of STL-10 is larger (48 × 48)).

Please see FIG0 in the appendix section for the actual computational time.

In order to highlight the difference between our spectral normalization and orthonormal regularization, we conducted an additional set of experiments.

As we explained in Section 3, orthonormal regularization is different from our method in that it destroys the spectral information and puts equal emphasis on all feature dimensions, including the ones that 'shall' be weeded out in the training process.

To see the extent of its possibly detrimental effect, we experimented by increasing the di- mension of the feature space 6 , especially at the final layer (7th conv) for which the training with our spectral normalization prefers relatively small feature space (dimension < 100; see FIG2 ).

As for the setting of the training, we selected the parameters for which the orthonormal regularization performed optimally.

The FIG3 shows the result of our experiments.

As we predicted, the performance of the orthonormal regularization deteriorates as we increase the dimension of the feature maps at the final layer.

Our SN-GANs, on the other hand, does not falter with this modification of the architecture.

Thus, at least in this perspective, we may such that our method is more robust with respect to the change of the network architecture.

To show that our method remains effective on a large high dimensional dataset, we also applied our method to the training of conditional GANs on ILRSVRC2012 dataset with 1000 classes, each consisting of approximately 1300 images, which we compressed to 128 × 128 pixels.

Regarding the adversarial loss for conditional GANs, we used practically the same formulation used in BID22 , except that we replaced the standard GANs loss with hinge loss (17).

Please see Appendix B.3 for the details of experimental settings.

GANs without normalization and GANs with layer normalization collapsed in the beginning of training and failed to produce any meaningful images.

GANs with orthonormal normalization BID4 and our spectral normalization, on the other hand, was able to produce images.

The inception score of the orthonormal normalization however plateaued around 20Kth iterations, while SN kept improving even afterward ( FIG4 .) To our knowledge, our research is the first of its kind in succeeding to produce decent images from ImageNet dataset with a single pair of a discriminator and a generator FIG6 .

To measure the degree of mode-collapse, we followed the footstep of BID27 and computed the intra MS-SSIM BID27 for pairs of independently generated GANs images of each class.

We see that our SN-GANs ((intra MS-SSIM)=0.101) is suffering less from the mode-collapse than AC-GANs ((intra MS-SSIM)∼0.25).To ensure that the superiority of our method is not limited within our specific setting, we also compared the performance of SN-GANs against orthonormal regularization on conditional GANs with projection discriminator BID23 as well as the standard (unconditional) GANs.

In our experiments, SN-GANs achieved better performance than orthonormal regularization for the both settings (See FIG0 in the appendix section).

This paper proposes spectral normalization as a stabilizer of training of GANs.

When we apply spectral normalization to the GANs on image generation tasks, the generated examples are more diverse than the conventional weight normalization and achieve better or comparative inception scores relative to previous studies.

The method imposes global regularization on the discriminator as opposed to local regularization introduced by WGAN-GP, and can possibly used in combinations.

In the future work, we would like to further investigate where our methods stand amongst other methods on more theoretical basis, and experiment our algorithm on larger and more complex datasets.

Let us describe the shortcut in Section 2.1 in more detail.

We begin with vectorsũ that is randomly initialized for each weight.

If there is no multiplicity in the dominant singular values and ifũ is not orthogonal to the first left singular vectors 7 , we can appeal to the principle of the power method and produce the first left and right singular vectors through the following update rule: DISPLAYFORM0 We can then approximate the spectral norm of W with the pair of so-approximated singular vectors: DISPLAYFORM1 If we use SGD for updating W , the change in W at each update would be small, and hence the change in its largest singular value.

In our implementation, we took advantage of this fact and reused theũ computed at each step of the algorithm as the initial vector in the subsequent step.

In fact, with this 'recycle' procedure, one round of power iteration was sufficient in the actual experiment to achieve satisfactory performance.

Algorithm 1 in the appendix summarizes the computation of the spectrally normalized weight matrixW with this approximation.

Note that this procedure is very computationally cheap even in comparison to the calculation of the forward and backward propagations on neural networks.

Please see FIG0 for actual computational time with and without spectral normalization.

Algorithm 1 SGD with spectral normalization DISPLAYFORM2

• For each update and each layer l:1.

Apply power iteration method to a unnormalized weight W l : DISPLAYFORM0 2.

CalculateW SN with the spectral norm: DISPLAYFORM1 3.

Update W l with SGD on mini-batch dataset D M with a learning rate α: DISPLAYFORM2 B EXPERIMENTAL SETTINGS

Inception score is introduced originally by : DISPLAYFORM0 , where p(y) is approximated by 1 N N n=1 p(y|x n ) and p(y|x) is the trained Inception convolutional neural network BID33 , which we would refer to Inception model for short.

In their work, reported that this score is strongly correlated with subjective human judgment of image quality.

Following the procedure in ; BID38 , we calculated the score for randomly generated 5000 examples from each trained generator to evaluate its ability to generate natural images.

We repeated each experiment 10 times and reported the average and the standard deviation of the inception scores.

Fréchet inception distance BID14 is another measure for the quality of the generated examples that uses 2nd order information of the final layer of the inception model applied to the examples.

On its own, the Frećhet distance BID7 is 2-Wasserstein distance between two distribution p 1 and p 2 assuming they are both multivariate Gaussian distributions: DISPLAYFORM1 where {µ p1 , C p1 }, {µ p2 , C p2 } are the mean and covariance of samples from q and p, respectively.

If f is the output of the final layer of the inception model before the softmax, the Fréchet inception distance (FID) between two distributions p 1 and p 2 on the images is the distance between f •p 1 and f • p 2 .

We computed the Fréchet inception distance between the true distribution and the generated distribution empirically over 10000 and 5000 samples.

Multiple repetition of the experiments did not exhibit any notable variations on this score.

For the comparative study, we experimented with the recent ResNet architecture of BID12 as well as the standard CNN.

For this additional set of experiments, we used Adam again for the optimization and used the very hyper parameter used in BID12 (α = 0.0002, β 1 = 0, β 2 = 0.9, n dis = 5).

For our SN-GANs, we doubled the feature map in the generator from the original, because this modification achieved better results.

Note that when we doubled the dimension of the feature map for the WGAN-GP experiment, however, the performance deteriorated.

The images used in this set of experiments were resized to 128 × 128 pixels.

The details of the architecture are given in Table 6 .

For the generator network of conditional GANs, we used conditional batch normalization (CBN) BID6 .

Namely we replaced the standard batch normalization layer with the CBN conditional to the label information y ∈ {1, . . .

, 1000}. For the optimization, we used Adam with the same hyperparameters we used for ResNet on CIFAR-10 and STL-10 dataset.

We trained the networks with 450K generator updates, and applied linear decay for the learning rate after 400K iterations so that the rate would be 0 at the end.

Table 3 : Standard CNN models for CIFAR-10 and STL-10 used in our experiments on image Generation.

The slopes of all lReLU functions in the networks are set to 0.1.

For the discriminator we removed BN layers in ResBlock.

Table 4 : ResNet architectures for CIFAR10 dataset.

We use similar architectures to the ones used in BID12 .

Table 6 : ResNet architectures for image generation on ImageNet dataset.

For the generator of conditional GANs, we replaced the usual batch normalization layer in the ResBlock with the conditional batch normalization layer.

As for the model of the projection discriminator, we used the same architecture used in BID23 .

Please see the paper for the details.

RGB image x ∈ R

ResBlock down 64ResBlock down 128ResBlock down 256Concat(Embed(y), h)

ResBlock down 1024ResBlock 1024

Global sum pooling dense → 1 (c) Discriminator for conditional GANs.

For computational ease, we embedded the integer label y ∈ {0, . . .

, 1000} into 128 dimension before concatenating the vector to the output of the intermediate layer.

C APPENDIX RESULTS C.1 ACCURACY OF SPECTRAL NORMALIZATION Figure 9 shows the spectral norm of each layer in the discriminator over the course of the training.

The setting of the optimizer is C in Table 1 throughout the training.

In fact, they do not deviate by more than 0.05 for the most part.

As an exception, 6 and 7-th convolutional layers with largest rank deviate by more than 0.1 in the beginning of the training, but the norm of this layer too stabilizes around 1 after some iterations.

FIG0 shows the effect of n dis on the performance of weight normalization and spectral normalization.

All results shown in FIG0 follows setting D, except for the value of n dis .

For WN, the performance deteriorates with larger n dis , which amounts to computing minimax with better accuracy.

Our SN does not suffer from this unintended effect.

This section is dedicated to the comparative study of spectral normalization and other regularization methods for discriminators.

In particular, we will show that contemporary regularizations including weight normalization and weight clipping implicitly impose constraints on weight matrices that places unnecessary restriction on the search space of the discriminator.

More specifically, we will show that weight normalization and weight clipping unwittingly favor low-rank weight matrices.

This can force the trained discriminator to be largely dependent on select few features, rendering the algorithm to be able to match the model distribution with the target distribution only on very low dimensional feature space.

The weight normalization introduced by BID31 is a method that normalizes the 2 norm of each row vector in the weight matrix 8 : DISPLAYFORM0 wherew i and w i are the ith row vector ofW WN and W , respectively.

Still another technique to regularize the weight matrix is to use the Frobenius norm: DISPLAYFORM1 where DISPLAYFORM2 Originally, these regularization techniques were invented with the goal of improving the generalization performance of supervised training BID31 BID2 .

However, recent works in the field of GANs BID39 found their another raison d'etat as a regularizer of discriminators, and succeeded in improving the performance of the original.

These methods in fact can render the trained discriminator D to be K-Lipschitz for a some prescribed K and achieve the desired effect to a certain extent.

However, weight normalization (25) imposes the following implicit restriction on the choice ofW WN : DISPLAYFORM3 where σ t (A) is a t-th singular value of matrix A. The above equation holds because Here, we see a critical problem in these two regularization methods.

In order to retain as much norm of the input as possible and hence to make the discriminator more sensitive, one would hope to make the norm ofW WN h large.

For weight normalization, however, this comes at the cost of reducing the rank and hence the number of features to be used for the discriminator.

Thus, there is a conflict of interests between weight normalization and our desire to use as many features as possible to distinguish the generator distribution from the target distribution.

The former interest often reigns over the other in many cases, inadvertently diminishing the number of features to be used by the discriminators.

Consequently, the algorithm would produce a rather arbitrary model distribution that matches the target distribution only at select few features.

DISPLAYFORM4 Our spectral normalization, on the other hand, do not suffer from such a conflict in interest.

Note that the Lipschitz constant of a linear operator is determined only by the maximum singular value.

In other words, the spectral norm is independent of rank.

Thus, unlike the weight normalization, our spectral normalization allows the parameter matrix to use as many features as possible while satisfying local 1-Lipschitz constraint.

Our spectral normalization leaves more freedom in choosing the number of singular components (features) to feed to the next layer of the discriminator.

To see this more visually, we refer the reader to Figure (14) .

Note that spectral normalization allows for a wider range of choices than weight normalization.

For the set of singular values permitted under the spectral normalization condition, we scaledW WN by 1/ √ d o so that its spectral norm is exactly 1.

By the definition of the weight normalization, the area under the blue curves are all bound to be 1.

Note that the range of choice for the weight normalization is small.

In summary, weight normalization and Frobenius normalization favor skewed distributions of singular values, making the column spaces of the weight matrices lie in (approximately) low dimensional vector spaces.

On the other hand, our spectral normalization does not compromise the number of feature dimensions used by the discriminator.

In fact, we will experimentally show that GANs trained with our spectral normalization can generate a synthetic dataset with wider variety and higher inception score than the GANs trained with other two regularization methods.

Still another regularization technique is weight clipping introduced by in their training of Wasserstein GANs.

Weight clipping simply truncates each element of weight matrices so that its absolute value is bounded above by a prescribed constant c ∈ R + .

Unfortunately, weight clipping suffers from the same problem as weight normalization and Frobenius normalization.

With weight clipping with the truncation value c, the value W x 2 for a fixed unit vector x is maximized when the rank of W is again one, and the training will again favor the discriminators that use only select few features.

BID12 refers to this problem as capacity underuse problem.

They also reported that the training of WGAN with weight clipping is slower than that of the original DCGAN .

One direct and straightforward way of controlling the spectral norm is to clip the singular values BID30 , BID17 .

This approach, however, is computationally heavy because one needs to implement singular value decomposition in order to compute all the singular values.

A similar but less obvious approach is to parametrize W ∈ R do×di as follows from the get-go and train the discriminators with this constrained parametrization: DISPLAYFORM0 where U ∈ R do×P , V ∈ R di×P , and S ∈ R P ×P is a diagonal matrix.

However, it is not a simple task to train this model while remaining absolutely faithful to this parametrization constraint.

Our spectral normalization, on the other hand, can carry out the updates with relatively low computational cost without compromising the normalization constraint.

Recently, BID12 introduced a technique to enhance the stability of the training of Wasserstein GANs .

In their work, they endeavored to place K-Lipschitz constraint (5) on the discriminator by augmenting the adversarial loss function with the following regularizer function: DISPLAYFORM0 where λ > 0 is a balancing coefficient andx is: DISPLAYFORM1 Using this augmented objective function, BID12 succeeded in training a GAN based on ResNet BID13 with an impressive performance.

The advantage of their method in comparison to spectral normalization is that they can impose local 1-Lipschitz constraint directly on the discriminator function without a rather round-about layer-wise normalization.

This suggest that their method is less likely to underuse the capacity of the network structure.

At the same time, this type of method that penalizes the gradients at sample pointsx suffers from the obvious problem of not being able to regularize the function at the points outside of the support of the current generative distribution.

In fact, the generative distribution and its support gradually changes in the course of the training, and this can destabilize the effect of the regularization itself.

On the contrary, our spectral normalization regularizes the function itself, and the effect of the regularization is more stable with respect to the choice of the batch.

In fact, we observed in the experiment that a high learning rate can destabilize the performance of WGAN-GP.

Training with our spectral normalization does not falter with aggressive learning rate.

Moreover, WGAN-GP requires more computational cost than our spectral normalization with single-step power iteration, because the computation of ∇ x D 2 requires one whole round of forward and backward propagation.

In FIG0 , we compare the computational cost of the two methods for the same number of updates.

Having said that, one shall not rule out the possibility that the gradient penalty can compliment spectral normalization and vice versa.

Because these two methods regularizes discriminators by completely different means, and in the experiment section, we actually confirmed that combination of WGAN-GP and reparametrization with spectral normalization improves the quality of the generated examples over the baseline (WGAN-GP only).

We can take advantage of the regularization effect of the spectral normalization we saw above to develop another algorithm.

Let us consider another parametrization of the weight matrix of the discriminator given by:W DISPLAYFORM0 where γ is a scalar variable to be learned.

This parametrization compromises the 1-Lipschitz constraint at the layer of interest, but gives more freedom to the model while keeping the model from becoming degenerate.

For this reparametrization, we need to control the Lipschitz condition by other means, such as the gradient penalty BID12 .

Indeed, we can think of analogous versions of reparametrization by replacingW SN in (32) with W normalized by other criterions.

The extension of this form is not new.

In BID31 , they originally introduced weight normalization in order to derive the reparametrization of the form (32) withW SN replaced (32) by W WN and vectorized γ.

In this part of the addendum, we experimentally compare the reparametrizations derived from two different normalization methods (weight normalization and spectral normalization).

We tested the reprametrization methods for the training of the discriminator of WGAN-GP.

For the architecture of the network in WGAN-GP, we used the same CNN we used in the previous section.

For the ResNet-based CNN, we used the same architecture provided by BID12 9 .Tables 7, 8 summarize the result.

We see that our method significantly improves the inception score from the baseline on the regular CNN, and slightly improves the score on the ResNet based CNN.

FIG0 shows the learning curves of (a) critic losses, on train and validation sets and (b) the inception scores with different reparametrization methods.

We can see the beneficial effect of spectral normalization in the learning curve of the discriminator as well.

We can verify in the figure 15a that the discriminator with spectral normalization overfits less to the training dataset than the discriminator without reparametrization and with weight normalization, The effect of overfitting can be observed on inception score as well, and the final score with spectral normalization is better than the others.

As for the best inception score achieved in the course of the training, spectral normalization achieved 7.28, whereas the spectral normalization and vanilla normalization achieved 7.04 and 6.69, respectively.

@highlight

We propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator of GANs.

@highlight

This paper uses spectral regularization to normalize GAN objectives, and the ensuing GAN, called SN-GAN, essentially ensures the Lipschitz property of the discriminator.

@highlight

This paper proposes"spectral normalization", moving a nice step forward in improving the training of GANs.