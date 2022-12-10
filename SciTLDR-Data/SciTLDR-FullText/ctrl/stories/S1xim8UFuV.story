Generative Adversarial Networks (GAN) can achieve promising performance on learning complex data distributions on different types of data.

In this paper, we first show that a straightforward extension of an existing GAN algorithm is not applicable to point clouds, because the constraint required for discriminators is undefined for set data.

We propose a two fold modification to a GAN algorithm to be able to generate point clouds (PC-GAN).

First, we combine ideas from hierarchical Bayesian modeling and implicit generative models by learning a hierarchical and interpretable sampling process.

A key component of our method is that we train a posterior inference network for the hidden variables.

Second, PC-GAN defines a generic framework that can incorporate many existing GAN algorithms.

We further propose a sandwiching objective, which results in a tighter Wasserstein distance estimate than the commonly used dual form in WGAN.

We validate our claims on the ModelNet40 benchmark dataset and observe that PC- GAN trained by the sandwiching objective achieves better results on test data than existing methods.

We also conduct studies on several tasks, including generalization on unseen point clouds, latent space interpolation, classification, and image to point clouds transformation, to demonstrate the versatility of the proposed PC-GAN algorithm.

A fundamental problem in machine learning is that given a data set, learn a generative model that can efficiently generate arbitrary many new sample points from the domain of the underlying distribution BID5 .

Deep generative models use deep neural networks as a tool for learning complex data distributions BID31 BID43 BID18 .

Especially, Generative Adversarial Networks (GAN) BID18 has drawn attention because of its success in many applications.

Compelling results have been demonstrated on different types of data, including text, images, and videos BID28 Vondrick et al., 2016) .

Their wide range of applicability was also shown in many important problems, including data augmentation (Salimans et al., 2016) , image style transformation BID24 , image captioning BID10 , and art creations BID27 .Recently, capturing 3D information is garnering attention.

There are many different data types for 3D information, such as CAD, 3D meshes, and point clouds.

3D point clouds are getting popular since these store more information than 2D images and sensors capable of collecting point clouds have become more accessible.

These include Lidar on self-driving cars, Kinect for Xbox, and face identification sensor on phones.

Compared to other formats, point clouds can be easily represented as a set of points, which has several advantages, such as permutation invariance of the set members.

The algorithms which can effectively learn from this type of data is an emerging field (Qi et al., 2017a; Zaheer et al., 2017; BID26 BID15 .

However, compared to supervised learning, unsupervised generative models for 3D data are still under explored BID0 BID42 .Extending existing GAN frameworks to point clouds or more generally set data is not straightforward.

In this paper, we begin by formally defining the problem and discussing its difficulty (Section 2).

Circumventing the challenges, we propose a deep generative adversarial network (PC-GAN) with a hierarchical sampling and inference network for point clouds.

The proposed architecture learns a stochastic procedure which can generate new point clouds and draw samples from the generated point clouds without explicitly modeling the underlying density function (Section 3).

The proposed PC-GAN is a generic algorithm which can incorporate many existing GAN variants.

By utilizing the property of point clouds, we further propose a sandwiching objective by considering both upper and lower bounds of Wasserstein distance estimate, which can lead to tighter approximation (Section 3.1).

Evaluation on ModelNet40 shows excellent generalization capability of PC-GAN.

We first demonstrate that we can sample from the learned model to generate new point clouds and the latent representations learned by the inference network provide meaningful interpolations between point clouds.

Then we show the conditional generation results on unseen classes of objects, which demonstrates the superior generalization ability of PC-GAN.

Lastly, we also provide several interesting studies, such as classification and point clouds generation from images (Section 5).

A point cloud for an object θ is a set of n low dimensional vectors X = {x 1 , ..., x n } with x i ∈ R d , where d is usually 3 and n can be infinite.

M different objects can be described as a collection of point clouds X(1) , ..., X (M ) .

A generative model for sets should be able to: (1) Sample entirely new sets according to p(X), and (2) sample arbitrarily many more points from the distribution of given set, i.e. x ∼ p(x|X).Based on the De-Finetti theorem, we could factor the probability with some suitably defined θ, such as object representation of point clouds, as p(X) = θ n i=1 p(x i |θ)p(θ)dθ.

In this view, the factoring can be understood as follows:

Given an object, θ, the points x i in the point cloud can be considered as i.i.d.

samples from p(x|θ), an unknown latent distribution representing object θ.

Joint likelihood can be expressed as: DISPLAYFORM0 One approach can be used to model the distribution of the point cloud set together, i.e., {{x DISPLAYFORM1 }.

In this setting, a naíve application of traditional GAN is possible through treating the point cloud as finite dimensional vector by fixing the number and order of the points (reducing the problem to instances in R n×3 ) with DeepSets (Zaheer et al., 2017) classifier as the discriminator to distinguish real sets from fake sets.

However, this approach would not work in practice because the integral probability metric (IPM) guarantees behind the traditional GAN no longer hold (e.g. in case of , nor are 1-Lipschitz functions over sets welldefined).

The probabilistic divergence approximated by a DeepSets classifier might be ill-defined.

Counter examples for breaking IPM guarantees can be easily found as we show next.

Counter Example Consider a simple GAN BID18 ) with a DeepSets classifier as the discriminator.

In order to generate coherent sets of variable size, we consider a generator G having two noise sources: u and z i .

To generate a set, u is sampled once and z i is sampled for i = 1, 2, ..., n to produce n points in the generated set.

Intuitively, fixing the first noise source u selects a set and ensures the points generated by repeated sampling of z i are coherent and belong to the same set.

The setup is depicted in FIG0 .

In this setup, the GAN minimax problem would be: min DISPLAYFORM2 Now consider the case, when there exists an 'oracle' mapping T which maps each sample point deterministically to the object it originated from, i.e. ∃T : T ({x i }) = θ.

A valid example is when different θ leads to conditional distribution p(x|θ) with non-overlapping support.

Let D = D • T and G ignore z, then the optimization task becomes as follows: min DISPLAYFORM3 Published as a workshop paper at ICLR 2019 Thus, we can achieve the lower bound − log(4) by only matching the p(θ) component, while the conditional p(x|θ) is allowed to remain arbitrary.

So simply using DeepSets classifier without any constraints in simple GAN in order to handle sets does not lead to a valid generative model.

Figure 2: Overview of PC-GAN.

As described in Section 2, directly learning point cloud generation under GAN formulation is difficult.

However, given θ, learning p(x|θ) is a simpler task of learning a 3-dimensional distribution.

Given two point clouds, one popular heuristic distance between them is the Chamfer distance BID0 .

On the other hand, if we treat each point cloud as a 3-dimensional distribution, we can adopt a broader class of probabilistic divergences for comparing them.

Instead of learning explicit densities BID25 Strom et al., 2010; BID13 , we are interested in implicit generative models with a GAN-like objective BID18 , which has been demonstrated to learn complicated distributions.

Formally, given a θ, we train a generator DISPLAYFORM0 and p(x|θ), which is denoted as P. The full objective can be written as DISPLAYFORM1 Inference Although GANs have been extended to learn conditional distributions BID38 BID24 , they require conditioning variables to be observed, such as the one-hot label or a given image.

Our θ, instead, is an unobserved latent variable for modeling different objects, which we need to infer during training.

The proposed algorithm has to concurrently learn the inference network Q(X) ≈ θ while we learn p(x|θ).

Since X is a set of points, we can adopt Qi et al. (2017a); Zaheer et al. (2017) for modeling Q.

We provide more discussion on this topic in the Appendix A.1.

Hierarchical Sampling After training G x and Q, we use the trained Q to collect the inferred Q(X) and train the generator G θ (u) ∼ p(θ) for higher hierarchical sampling.

Here u ∼ p(u) is the other noise source independent of z. In addition to layer-wise training, a joint training could further boost performance.

The full generative process for sampling one point cloud could be represented as DISPLAYFORM2 , where z 1 , . . .

, z n ∼ p(z), and u ∼ p(u).

The overview of proposed algorithm for point cloud generation (PC-GAN) is shown in Figure 2 .

To train the generator G x using a GAN-like objective for point clouds, we need a discriminator f (·) to distinguishes generated samples and true samples conditioned on θ.

Combining with the inference network Q(X) discussed aforementioned, the objecitve with IPM-based GANs can be written as DISPLAYFORM0 where Ω f is the constraint for different probabilistic distances, such as 1-Lipschitz , L 2 ball or Sobolev ball .

In our setting, each point x i in the point cloud can be considered to correspond to single images when we train GANs over images.

An example is illustrated in FIG1 where samples from MMD-GAN BID34 ) trained on CelebA consists of both good and bad faces.

In case of images, when quality is evaluated, it primarily focuses on coherence individual images and the few bad ones are usually left out.

Whereas in case of point cloud, to get representation of an object we need many sampled points together and presence of outlier points degrades the quality of the object.

Thus, when training a generative model for point cloud, we need to ensure a much lower distance D(P G) between true distribution P and generator distribution G than would be needed in case of images.

We begin by noting that the popular Wasserstein GAN , aims to optimize G by min w(P, G), where w(P, G) is the Wasserstein distance w(P, G) between the truth P and generated distribution G of G. Many GAN works (e.g. ) approximate w(P, G) in dual form (a maximization problem), such as (4), by neural networks.

The resulting estimate W L (P, G) is a lower bound of the true Wasserstein distance, as neural networks can only recover a subset of 1-Lipschitz functions BID2 required in the dual form.

However, finding a lower bound W L (P, G) for w(P, G) may not be an ideal surrogate for solving a minimization problem min w(P, G).

In optimal transport literature, Wassertein distance is usually estimated by approximate matching cost, W U (P, G), which gives us an upper bound of the true Wasserstein distance.

We propose to combine, in general, a lower bound W L and upper bound estimate W U by sandwiching the solution between the two, i.e. we solve the following minimization problem: DISPLAYFORM0 The problem can be simplified and solved using method of lagrange multipliers as follows: DISPLAYFORM1 By solving the new sandwiched problem (6), we show that under certain conditions we obtain a better estimate of Wasserstein distance in the following lemma: Lemma 1.

Suppose we have two approximators to Wasserstein distance: an upper bound W U and a lower W L , such that ∀P, G : DISPLAYFORM2 Then, using the sandwiched estimator W s from (6), we can achieve tighter estimate of the Wasserstein distance than using either one estimator, i.e. DISPLAYFORM3

For W L , we can adopt many GAN variants BID21 .

For W U , we use BID4 , which results in a fast approximation of the Wasserstein distance estimate in primal form without solving non-trivial linear programming.

We remark estimating Wasserstein distance w(P, G) with finite samples via its primal is only favorable to low dimensional data, such as point clouds.

The error of empirical estimate in primal is & Bach, 2017) .

When the dimension d is large (e.g. images), we cannot accurately estimate w(P, G) in primal as well as its upper bound with a small minibatch.

For detailed discussion of finding lower and upper bound, please refer to Appendix A.2 and A.3.

DISPLAYFORM0

Generative Adversarial Network BID18 aims to learn a generator that can sample data followed by the data distribution.

Compelling results on learning complex data distributions with GAN have been shown on images BID28 , speech ), text (Yu et al., 2016; BID23 ), vedio (Vondrick et al., 2016 ) and 3D voxels (Wu et al., 2016 .

However, the GAN algorithm on 3D point cloud is still under explored BID0 .

Many alternative objectives for training GANs have been studied.

Most of them are the dual form of f -divergence BID18 BID37 BID41 , integral probability metrics (IPMs) (Zhao et al., 2016; BID34 BID21 or IPM extensions .

BID16 learn the generative model by the approximated primal form of Wasserstein distance BID9 .Instead of training a generative model on the data space directly, one popular approach is combining with autoencoder (AE), which is called adversarial autoencoder (AAE) BID36 .

AAE constrain the encoded data to follow normal distribution via GAN loss, which is similar to VAE BID31 by replacing the KL-divergence on latent space via any GAN loss.

Tolstikhin et al. (2017) provide a theoretical explanation for AAE by connecting it with the primal form of Wasserstein distance.

The other variant of AAE is training the other generative model to learn the distribution of the encoded data instead of enforcing it to be similar to a known distribution BID14 BID30 .

BID0 explore a AAE variant for point cloud.

They use a specially-designed encoder network (Qi et al., 2017a) for learning a compressed representation for point clouds before training GAN on the latent space.

However, their decoder is restricted to be a MLP which generates m fixed number of points, where m has to be pre-defined.

That is, the output of their decoder is fixed to be 3m for 3D point clouds, while the output of the proposed G x is only 3 dimensional and G x can generate arbitrarily many points by sampling different random noise z as input.

Yang et al. FORMULA2 ; BID20 propose similar decoders to G x with fixed grids to break the limitation of BID0 aforementioned, but they use heuristic Chamfer distance without any theoretical guarantee and do not exploit generative models for point clouds.

The proposed PC-GAN can also be interpreted as an encoder-decoder formulation.

However, the underlying interpretation is different.

We start from De-Finetti theorem to learn both p(X|θ) and p(θ) with inference network interpretation of Q, while BID0 focus on learning p(θ) without modeling p(X|θ).Lastly, GAN for learning conditional distribution (conditional GAN) has been studied in images with single conditioning BID38 Pathak et al., 2016; BID24 BID6 or multiple conditioning (Wang & Gupta, 2016) .

The case on point cloud is still under explored.

Also, most of the works assume the conditioning is given (e.g. labels and base images) without learning the inference during the training.

Training GAN with inference is studied by BID11 ; Dumoulin et al. FORMULA2 ; BID35 ; however, their goal is to infer the random noise z of generators and match the semantic latent variable to be similar to z. Li et al. FORMULA2 is a parallel work aiming to learn GAN and unseen latent variable simultaneously, but they only study image and video datasets.

In this section we demonstrate the point cloud generation capabilities of PC-GAN.

As discussed in Section 4, we refer BID0 as AAE as it could be treated as an AAE extension to point clouds and we use the implementation provided by the authors for experiments.

The sandwitching objective W s for PC-GAN combines W L and W U with the mixture 1:20 without tunning for all experiment.

W L is a GAN loss by combining and (technical details are in Appendix A.3) and we adopt BID4 for W U .

We parametrize Q in PC-GAN by DeepSets (Zaheer et al., 2017) .

The review of DeepSets is in Appendix E. Other detailed configurations of each experiment can be found in Appendix F.

We generate 2D circle point clouds.

The center of circles follows a mixture of Gaussians N ({±16} × {±16}, 16I) with equal mixture weights.

The radius of the circles was drawn from a uniform distribution Unif(1.6, 6.4).

One sampled circile is shown in FIG2 .

For AAE, the output size of the decoder is 500 × 2 for 500 points, and the output size of the encoder (latent code) is 20.

The total number of parameters are 24K.

For PC-GAN, the inference network output size is 15.

The total nuumber of parameters of PC-GAN is only 12K.

We evaluated the conditional distributions on the 10, 000 testing circles.

We measured the empirical distributions of the centers and the radius of the generated circles conditioning on the testing data as shown in FIG2 .From FIG2 , both AAE and PC-GAN can successfully recover the center distribution, but AAE does not learn the radius distribution well even with larger latent code (20) and more parameters (24K).

The gap of memory usage could be larger if we configure AAE to generate more points, while the model size required for PC-GAN is independent of the number of points.

The reason is MLP decoder adopted by BID0 wastes parameters for nearby points.

Using the much larger model (more parameters) could boost the performance.

However, it is still restricted to generate a fixed number of points for each object as we discussed in Section 4.

We consider ModelNet40 (Wu et al., 2015) benchmark, which contains 40 classes of objects.

There are 9, 843 training and 2, 468 testing instances.

We follow BID0 to consider two settings.

One is training on single class of objects.

The other is training on all 9, 843 objects in the training set.

BID0 set the latent code size of AAE to be 128 and 256 for these two settings, with the total number of parameters to be 15M and 15.2M , respectively.

Similarly, we set the output dimension of Q in PC-GAN to be 128 and 256 for single-class and all-classes.

The total number of parameters are 1M and 3M , respectively.

Firstly, we are interested in whether the learned G x and Q can model the distribution of unseen test data.

For each test point cloud, we infer the latent variable Q(X), then use G x to generate points.

We then compare the distribution between the input point cloud and the conditionally generated point clouds.

There are many finite sample estimation for f -divergence and IPM can be used for evaluation.

However, those estimators with finite samples are either biased or with high variance (Peyré et al., 2017; Wang et al., 2009; Póczos et al., 2012; Weed & Bach, 2017) .

Also, it is impossible to use these estimators with infinitely many samples if they are accessible.

For ModelNet40, the meshes of each object are available.

In many statistically guaranteed distance estimates, the adopted statistics are commonly based on distance between nearest neighbors (Wang et al., 2009; Póczos et al., 2012) .

Therefore, we propose to measure the performance with the following criteria.

Given a point cloud {x i } n i=1 and a mesh, which is a collection of faces {F j } m j=1 , we measure the distance to face (D2F) as DISPLAYFORM0 where D(x i , F j ) is the Euclidean distance from x i to the face F j .

This distance is similar to Chamfer distance, which is commonly used for measuring images and point clouds BID0 BID15 , with infinitely samples from true distributions (meshes).Nevertheless, the algorithm can have low or zero D2F by only focusing a small portion of the point clouds (mode collapse).

Therefore, we are also interested in whether the generated points recover enough supports of the distribution.

We compute the Coverage ratio as follows.

For each point, we find the its nearest face, we then treat this face is covered 1 .

We then compute the ratio of number of faces of a mesh is covered.

A sampled mesh is showed in FIG3 , where the details have more faces (non-uniform).

Thus, it is difficult to get high coverage for AAE or PC-GAN trained by limited number of sampled points.

However, the coverage ratio, on the other hand, serve as an indicator about how much details the model recovers.

The results are reported in BID3 , which results in better coverage (support) than W U .Theoretically, the proposed sandwiching W s results in a tighter Wasserstein distance estimation than W U and W L (Lemma 1).

Based on above discussion, it can also be understood as balancing both D2F and coverage by combining both W U and W L to get a desirable middle ground.

Empirically, we even observe that W s results in better coverage than W L , and competitive D2F with W U .

The intuitive explanation is that some discriminative tasks are off to W U objective, so the discriminator can focus more on learning distribution supports.

We argue that this difference is crucial for capturing the object details.

Some reconstructed point clouds of testing data are shown in FIG4 .

For aeroplane examples, W U are failed to capture aeroplane tires and W s has better tire than W L .

For Chair example, W s recovers better legs than W U and better seat cushion than W L .

Lastly, we highlight W s outperforms others more significantly when training data is larger (ModelNet10 and ModelNet40) in TAB1 .Comparison between PC-GAN and AAE In most of cases, PC-GAN with W s has lower D2F in TAB1 with less number of parameters aforementioned.

Similar to the argument in Section 5.1, although AAE use larger networks, the decoder wastes parameters for nearby points.

AAE only outperforms PC-GAN (W s ) in Guitar and Sofa in terms of D2F, since the variety of these two classes are low.

It is easier for MLP to learn the shared template (basis) of the point clouds.

On the other hand, due to the limitation of the fixed number of output points and Chamfer distance objective, AAE has worse coverage than PC-GAN, It can be supported by FIG4 , where AAE is also failed to recover aeroplane tire.

Hierarchical Sampling In Section 3, we propose a hierarchical sampling process for sampling point clouds.

In the first hierarchy, the generator G θ , samples a object (θ = G θ (u), u ∼ P(u)), while the second generator G x samples points based on θ to form the point cloud.

The randomly sampled results without given any data as input are shown in FIG5 .

More results can be found in Appendix C.

The point clouds are all smooth, structured and almost symmetric.

It shows PC-GAN captures inherent symmetries and patterns in all the randomly sampled objects, even if overall object is not perfectly formed.

This highlights that learning point-wise generation scheme encourages learning basic building blocks of objects.

Interpolation of Learned Manifold We study whether the interpolation between two objects on the latent space results in smooth change.

We interpolate the inferred representations of two objects by Q, and use the generator G x to sample points based on the interpolation.

The inter-class result is shown in FIG6 .

More studies about interpolation between rotations can be found in Appendix D.1.

Generalization on Unseen Classes In above, we studied the reconstruction of unseen testing objects, while PC-GAN still saw the point clouds from the same class during training.

Here we study the more challenging task.

We train PC-GAN on first 30 (Alphabetic order) class, and test on the other fully unseen 10 classes.

Some reconstructed (conditionally generated) point clouds are shown in Figure 9 .

More (larger) results can be found in Appendix C. For the object from the unseen classes, the conditionally generated point clouds still recovers main shape and reasonable geometry structure, which confirms the advantage of the proposed PC-GAN: by enforcing the point-wise transformation, the model is forced to learn the underlying geometry structure and the shared building blocks, instead of naively copying the input from the conditioning.

The rsulted D2F and coverage are 57.4 and 0.36, which are only slightly worse than 48.4 and 0.38 by training on whole 40 classes in TAB1 (ModelNet40), which also supports the claims of the good generalization ability of PC-GAN.

Figure 9: The reconstructed objects from unseen classes (even in training).

In each plot, LHS is true data while RHS is PC-GAN.

PC-GAN generalizes well as it can match patterns and symmetries from classes seen in the past to new unseen classes.

More Studies We also condct other studies to make experiments complete, including interpolation between different rotations, classification and image to point clouds.

Due to space limit, all of the results can be found in Appendix D.

In this paper, we first showed a straightforward extension of existing GAN algorithm is not applicable to point clouds.

We then proposed a GAN modification (PC-GAN) that is capable of learning to generate point clouds by using ideas both from hierarchical Bayesian modeling and implicit generative models.

We further propose a sandwiching objective which results in a tighter Wasserstein distance estimate theoretically and better performance empirically.

In contrast to some existing methods BID0 , PC-GAN can generate arbitrary as many i.i.d.

points as we need to form a point clouds without pre-specification.

Quantitatively, PC-GAN achieves competitive or better results using smaller network than existing methods.

We also demonstrated that PC-GAN can capture delicate details of point clouds and generalize well even on unseen data.

Our method learns "point-wise" transformations which encourage the model to learn the building components of the objects, instead of just naively copying the whole object.

We also demonstrate other interesting results, including point cloud interpolation and image to point clouds.

Although we only focused on 3D applications in this paper, our framework can be naturally generalized to higher dimensions.

In the future we would like to explore higher dimensional applications, where each 3D point can have other attributes, such as RGB colors and 3D velocity vectors.

Our solution comprises of a generator G x (z, ψ) which takes in a noise source z ∈ R d1 and a descriptor ψ ∈ R d2 encoding information about distribution of θ.

For a given θ 0 , the descriptor ψ would encode information about the distribution δ(θ − θ 0 ) and samples generated as x = G x (z, ψ) would follow the distribution p(x|θ 0 ).

More generally, ψ can be used to encode more complicated distributions regarding θ as well.

In particular, it could be used to encode the posterior p(θ|X) for a given sample set X, such that x = G x (z, ψ) follows the posterior predictive distribution: DISPLAYFORM0 A major hurdle in taking this path is that X is a set of points, which can vary in size and permutation of elements.

Thus, making design of Q complicated as traditional neural network can not handle this and possibly is the reason for absence of such framework in the literature despite being a natural solution for the important problem of generative modeling of point clouds.

However, we can overcome this challenge and we propose to construct the inference network by utilizing the permutation equivariant layers from Deep Sets (Zaheer et al., 2017) .

This allows it handle variable number of inputs points in arbitrary order, yet yielding a consistent descriptor ψ.

After training G x and the inference network Q, we use trained Q to collect inferred Q(X) and train the generator G θ (u) ∼ p(θ) for higher hierarchical sampling, where u is the other noise source independent of z. In addition to the layer-wise training, a joint training may further boost the performance.

The full generative process for sampling one point cloud could be represented as DISPLAYFORM1 , where z 1 , . . .

, z n ∼ p(z) and u ∼ p(u).

We call the proposed GAN framework for learning to generative point clouds as PC-GAN as shown in Figure 2 .

The conditional distribution matching with a learned inference in PC-GAN can also be interpreted as an encoder-decoder formulation BID31 .

The difference between it and the point cloud autoencoder BID0 Yang et al., 2018) will be discussed in Section 4.

The primal form of Wasserstein distance is defined as DISPLAYFORM0 where γ is the coupling of P and G. The Wasserstein distance is also known as optimal transport (OT) or earth moving distance (EMD).

As the name suggests, when w(P, G) is estimated with finite number of samples X = x 1 , . . .

, x n and Y = y 1 , . . .

, y n , we find the one-to-one matching between X and Y such that the total pairwise distance is minimal.

The resulting minimal total (average) pairwise distance is w(X, Y ).

In practice, finding the exact matching efficiently is non-trivial and still an open research problem (Peyré et al., 2017) .

Instead, we consider an approximation provided by BID4 .

It is an iterative algorithm where each iteration operates like an auction whereby unassigned points x ∈ X bid simultaneously for closest points y ∈ Y , thereby raising their prices.

Once all bids are in, points are awarded to the highest bidder.

The crux of the algorithm lies in designing a non-greedy bidding strategy.

One can see by construction the algorithm is embarrassingly parallelizable, which is favourable for GPU implementation.

One can show that algorithm terminates with a valid matching and the resulting matching cost W U (X, Y ) is an -approximation of w(X, Y ).

Thus, the estimate can serve as an upper bound, i.e. DISPLAYFORM1 We remark estimating Wasserstein distance w(P, G) with finite sample via primal form is only favorable in low dimensional data, such as point clouds.

The error between w(P, G) and Weed & Bach, 2017) .

Therefore, for high dimensional data, such as images, we cannot accurately estimate wasserstein distance in primal and its upper bound with a small minibatch.

DISPLAYFORM2 Finding a modified primal form with low sample complexity is also an open research problem BID9 BID16 , and combining those into the proposed sandwiching objective for high dimensional data is left for future works.

The dual form of Wasserstein distance is defined as DISPLAYFORM0 where L k is the set of k-Lipschitz functions whose Lipschitz constant is no larger than k. In practice, deep neural networks parameterized by φ with constraints f φ ∈ Ω φ , result in a distance approximation DISPLAYFORM1 If there exists propose a weight clipping constraint Ω c , which constrains every weight to be in [−c, c] and guarantees that Ω c ⊆ L k for some k. However, choosing clipping range c is non-trivial in practice.

Small ranges limit the capacity of networks, while large ranges result in numerical issues during the training.

On the other hand, in addition to weight clipping, several constraints (regularization) have bee proposed with better empirical performance, such as gradient penalty BID21 and L 2 ball .

However, there is no guarantee the resulted functions are still Lipschitz or the resulted distances are lower bounds of Wasserstein distance.

To take the advantage of those regularization with the Lipschitz guarantee, we propose a simple variation by combining weight clipping, which always ensures Lipschitz functions.

DISPLAYFORM2 Lemma 2.

There exists k > 0 such that DISPLAYFORM3 Note that, if c → ∞, then Ω c ∩ Ω φ = Ω φ .

Therefore, from Proposition 2, for any regularization of discriminator BID21 , we can always combine it with a weight clipping constraint Ω c to ensure a valid lower bound estimate of Wasserstein distance and enjoy the advantage that it is numerically stable when we use large c compared with original weight-clipping WGAN .In practice, we found combing L 2 ball constraint and weight-clipping leads to satisfactory performance.

We also studied popular WGAN-GP BID21 with weight clipping to ensure Lipschitz continuity of discriminator, but we found L 2 ball with weight clipping is faster and more numerically stable to train.

Lemma 1.

Suppose we have two approximators to Wasserstein distance: an upper bound W U and a lower W L , such that ∀P, G : (1 + 1 )w(P, G) ≤ W U (P, G) ≤ (1 + 2 )w(P, G) and ∀P, G : (1 − 2 )w(P, G) ≤ W L (P, G) ≤ (1 − 1)w(P, G) respectively, for some 2 > 1 > 0 and 1 > 2 /3.

Then, using the sandwiched estimator W s from (6), we can achieve tighter estimate of the Wasserstein distance than using either one estimator, i.e. DISPLAYFORM0 Proof.

We prove the claim by show that LHS is at most 1 , which is the lower bound for RHS.

DISPLAYFORM1 Without loss of generality we can assume λ < 0.5, which brings us to DISPLAYFORM2 Now if we chose DISPLAYFORM3 Lemma 2.

There exists k > 0 such that DISPLAYFORM4 Proof.

Since there exists k such that DISPLAYFORM5

The larger and more hierarchical sampling discussed in Section 5.2 can be found in FIG0 .

The reconstruction results on unseen classes are shown in FIG0 .

It is also popular to show intra-class interpolation.

In addition show simple intra-class interpolations, where the objects are almost aligned, we present an interesting study on interpolations between rotations.

During the training, we only rotate data with 8 possible angles for augmentation, here we show it generalizes to other unseen rotations as shown in FIG0 .However, if we linearly interpolate the code, the resulted change is scattered and not smooth as shown in FIG0 .

Instead of using linear interpolation, We train a 2-layer MLP with limited hidden layer size to be 16, where the input is the angle, output is the corresponding latent representation of rotated object.

We then generate the code for rotated planes with this trained MLP.

It suggests although the transformation path of rotation on the latent space is not linear, it follows a smooth trajectory 2 .

It may also suggest the geodesic path of the learned manifold may not be nearly linear between rotations.

Finding the geodesic path with a principal method (Shao et al., 2017) and Understanding the geometry of the manifold for point cloud worth more deeper study as future work.

We evaluate the quality of the representation acquired from the learned inference network Q. We train the inference network Q and the generator G x on the training split of ModelNet40 with data FIG0 : Randomly sampled objects and corresponding point cloud from the hierarchical sampling.

Even if there are some defects, the objects are smooth, symmetric and structured.

It suggests PC-GAN captures inherent patterns and learns basic building blocks of objects.

augmentation as mentioned above for learning generative models without label information.

We then extract the latent representation Q(X) for each point clouds and train linear SVM on the that with its label.

We apply the same setting to a linear classifier on the latent code of BID0 .We only sample 1000 as input for our inference network Q. Benefited by the Deep Sets architecture for the inference network, which is invariant to number of points.

Therefore, we are allowed to sample different number of points as input to the trained inference network for evaluation.

Because of the randomness of sampling points for extracting latent representation, we repeat the experiments 20 times and report the average accuracy and standard deviation on the testing split in Table 2 .

By using 1000 points, we are already better than BID0 with 2048 points, and competitive with the supervised learning algorithm Deep Sets.

We also follow the same protocol as BID0 Wu et al. (2016) that we train on ShapeNet55 and test the accuracy on ModelNet40.

Compared with existing unsupervised learning algorithms, PC-GAN has the best performance as shown in TAB3 .

# points Accuracy PC-GAN 1000 87.5 ± .6% PC-GAN 2048 87.8 ± .2% AAE BID0 2048 85.5 ± .3% Deep Sets (Zaheer et al., 2017)

1000 87 ± 1% Deep Sets (Zaheer et al., 2017) 5000 90 ±

.3% Table 2 : Classification accuracy results.

We note that Yang et al. (2018) using additional geometry features by appending pre-calculated features with 3-dimensional coordinate as input or using more advanced grouping structure to achieve better performance.

Those techniques are all applicable to PC-GAN and leave it for future works by leveraging geometry information into the proposed PC-GAN framework.

Here we demonstrate a potential extension of the proposed PC-GAN for images to point cloud applications.

After training Q as described in 3 and Appendix A.1, instead of learning G θ for hierarchical sampling, we train a regressor R, where the input is the different views of the point cloud X, and the output is Q(X).

In this proof of concept experiment, we use the 12 view data and the Res18 architecture in Su et al. FORMULA2 , while we change the output size to be 256.

Some example results on reconstructing testing data is shown in FIG0 .

A straightforward extension is using end-to-end training instead of two-staged approached adopted here.

Also, after aligning objects and take representative view along with traditional ICP techniques, we can also do single view to point cloud transformation as Choy et al. FORMULA2 ; BID15 ; BID22 ; BID19 , which is not the main focus of this paper and we leave it for future work.

Method Accuracy SPH BID29 68.2% T-L Network BID17 74.4% LFD BID7 75.5% VConv-DAE (Sharma et al., 2016) 75.5% 3D GAN (Wu et al., 2016) 83.3% AAE BID0 84.5% PC-GAN 86.9% where σ can be any functions (e.g. parametrized by neural networks) and X = x 1 , . . .

, x n is an input set.

Also, the mox pooling operation can be replaced with mean pooling.

We note that PointNetQi et al. (2017a) is a special case of using Permutation Equivariance Layer by properly defining σ(·).In our experiments, we follow Zaheer et al. (2017) to set σ to be a linear layer with output size h followed by any nonlinear activation function.

The batch size is fixed to be 64.

We sampled 10,000 samples for training and testing.

For the inference network, we stack 3 mean Permutation Equivariance Layer (Zaheer et al., 2017) , where the hidden layer size (the output of the first two layers ) is 30 and the final output size is 15.

The activation function are used SoftPlus.

For the generater is a 5 layer MLP, where the hidden layer size is set to be 30.

The discirminator is 4 layer MLP with hidden layer size to be 30.

For BID0 , we change their implementation by replcing the number of filters for encoder to be [30, 30, 30, 30, 15] , while the hidden layer width for decoder is 10 or 20 except for the output layer.

The decoder is increased from 3 to 4 layers to have more capacity.

We follow Zaheer et al. (2017) to do pre-processing.

For each object, we sampled 10, 000 points from the mesh representation and normalize it to have zero mean (for each axis) and unit (global) variance.

During the training, we augment the data by uniformly rotating 0, π/8, . . .

, 7π/8 rad on the x-y plane.

The random noise z 2 of PC-GAN is fixed to be 10 dimensional for all experiments.

For Q of single class model, we stack 3 max Permutation Equivariance Layer with output size to be 128 for every layer.

On the top of the satck, we have a 2 layer MLP with the same width and the output .

The generator G x is a 4 layer MLP where the hidden layer size is 128 and output size is 3.

The discirminator is 4 layer MLP with hidden layer size to be 128.

The random source u and z are set to be 64 and 10 dimensional and sampled from standard normal distributions.

For training whole ModelNet40 training set, we increae the width to be 256.

The generator G x is a 5 layer MLP where the hidden layer size is 256 and output size is 3.

The discirminator is 5 layer MLP with hidden layer size to be 256.

For hirarchical sampling, the top generator G θ and discriminator are all 5-layer MLP with hidden layer size to be 256.For AAE, we follow every setting used in BID0 , where the latent code size is 128 and 256 for single class model and whole ModelNet40 models.

<|TLDR|>

@highlight

We propose a GAN variant which learns to generate point clouds. Different studies have been explores, including tighter Wasserstein distance estimate,  conditional generation, generalization to unseen point clouds and image to point cloud.

@highlight

This paper proposes using GAN to generate 3D point cloud and introduces a sandwiching objective, averaging the upper and lower bound of Wasserstein distance between distributions.

@highlight

This paper proposes a new generative model for unordered data, with a particular application to point clouds, which includes an inference method and a novel objective function. 