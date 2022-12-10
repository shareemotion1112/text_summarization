Learning rich representation from data is an important task for deep generative models such as variational auto-encoder (VAE).

However, by extracting high-level abstractions in the bottom-up inference process, the goal of preserving all factors of variations for top-down generation is compromised.

Motivated by the concept of “starting small”, we present a strategy to progressively learn independent hierarchical representations from high- to low-levels of abstractions.

The model starts with learning the most abstract representation, and then progressively grow the network architecture to introduce new  representations at different levels of abstraction.

We quantitatively demonstrate the ability of the presented model to improve disentanglement in comparison to existing works on two benchmark datasets using three disentanglement metrics, including a new metric we proposed to complement the previously-presented metric of mutual information gap.

We further present both qualitative and quantitative evidence on how the progression of learning improves disentangling of hierarchical representations.

By drawing on the respective advantage of hierarchical representation learning and progressive learning, this is to our knowledge the first attempt to improve disentanglement by progressively growing the capacity of VAE to learn hierarchical representations.

Variational auto-encoder (VAE), a popular deep generative model (DGM), has shown great promise in learning interpretable and semantically meaningful representations of data ; Chen et al. (2018) ; Kim & Mnih (2018) ).

However, VAE has not been able to fully utilize the depth of neural networks like its supervised counterparts, for which a fundamental cause lies in the inherent conflict between the bottom-up inference and top-down generation process (Zhao et al. (2017) ; Li et al. (2016) ): while the bottom-up abstraction is able to extract high-level representations helpful for discriminative tasks, the goal of generation requires the preservation of all generative factors that are likely at different abstraction levels.

This issue was addressed in recent works by allowing VAEs to generate from details added at different depths of the network, using either memory modules between top-down generation layers (Li et al. (2016) ), or hierarchical latent representations extracted at different depths via a variational ladder autoencoder (VLAE, Zhao et al. (2017) ).

However, it is difficult to learn to extract and disentangle all generative factors at once, especially at different abstraction levels.

Inspired by human cognition system, Elman (1993) suggested the importance of "starting small" in two aspects of the learning process of neural networks: incremental input in which a network is trained with data and tasks of increasing complexity, and incremental memory in which the network capacity undergoes developmental changes given fixed external data and tasks -both pointing to an incremental learning strategy for simplifying a complex final task.

Indeed, the former concept of incremental input has underpinned the success of curriculum learning (Bengio et al. (2015) ).

In the context of DGMs, various stacked versions of generative adversarial networks (GANs) have been proposed to decompose the final task of high-resolution image generation into progressive sub-tasks of generating small to large images (Denton et al. (2015) ; Zhang et al. (2018) ).

The latter aspect of "starting small" with incremental growth of network capacity is less explored, although recent works have demonstrated the advantage of progressively growing the depth of GANs for generating high-resolution images (Karras et al. (2018) ; ).

These works, so far, have focused on progressive learning as a strategy to improve image generation.

We are motivated to investigate the possibility to use progressive learning strategies to improve learning and disentangling of hierarchical representations.

At a high level, the idea of progressively or sequentially learning latent representations has been previously considered in VAE.

In Gregor et al. (2015) , the network learned to sequentially refine generated images through recurrent networks.

In Lezama (2019) , a teacher-student training strategy was used to progressively increase the number of latent dimensions in VAE to improve the generation of images while preserving the disentangling ability of the teacher model.

However, these works primarily focus on progressively growing the capacity of VAE to generate, rather than to extract and disentangle hierarchical representations.

In comparison, in this work, we focus on 1) progressively growing the capacity of the network to extract hierarchical representations, and 2) these hierarchical representations are extracted and used in generation from different abstraction levels.

We present a simple progressive training strategy that grows the hierarchical latent representations from different depths of the inference and generation model, learning from high-to low-levels of abstractions as the capacity of the model architecture grows.

Because it can be viewed as a progressive strategy to train the VLAE presented in Zhao et al. (2017) , we term the presented model pro-VLAE.

We quantitatively demonstrate the ability of pro-VLAE to improve disentanglement on two benchmark data sets using three disentanglement metrics, including a new metric we proposed to complement the metric of mutual information gap (MIG) previously presented in Chen et al. (2018) .

These quantitative studies include comprehensive comparisons to β-VAE ), VLAE (Zhao et al. (2017) ), and the teacher-student strategy as presented in (Lezama (2019) ) at different values of the hyperparameter β.

We further present both qualitative and quantitative evidence that pro-VLAE is able to first learn the most abstract representations and then progressively disentangle existing factors or learn new factors at lower levels of abstraction, improving disentangling of hierarhical representations in the process.

A hierarchy of feature maps can be naturally formed in stacked discriminative models (Zeiler & Fergus (2014) ).

Similarly, in DGM, many works have proposed stacked-VAEs as a common way to learn a hierarchy of latent variables and thereby improve image generation (Sønderby et al. (2016); Bachman (2016) ; Kingma et al. (2016) ).

However, this stacked hierarchy is not only difficult to train as the depths increases (Sønderby et al. (2016); Bachman (2016) ), but also has an unclear benefit for learning either hierarchical or disentangled representations: as shown in Zhao et al. (2017) , when fully optimized, it is equivalent to a model with a single layer of latent variables.

Alternatively, instead of a hierarchy of latent variables, independent hierarchical representations at different abstraction levels can be extracted and used in generation from different depths of the network (Rezende et al. (2014) ; Zhao et al. (2017) ).

A similar idea was presented in Li et al. (2016) to generate lost details from memory and attention modules at different depths of the top-down generation process.

The presented work aligns with existing works (Rezende et al. (2014) ; Zhao et al. (2017) ) in learning independent hierarchical representation from different levels of abstraction, and we look to facilitate this learning by progressively learning the representations from high-to low-levels.

Progressive learning has been successful for high-quality image generation, mostly in the setting of GANs.

Following the seminar work of Elman (1993), these progressive strategies can be loosely grouped into two categories.

Mostly, in line with incremental input, several works have proposed to divide the final task of image generation into progressive tasks of generating low-resolution to high-resolution images with multi-scale supervision (Denton et al. (2015) ; Zhang et al. (2018) ).

Alternatively, in line with incremental memory, a small number of works have demonstrated the ability to simply grow the architecture of GANs from a shallow network with limited capacity for generating low-resolution images, to a deep network capable of generating super-resolution images (Karras et al. (2018); ).

This approach was also shown to be time-efficient since the early-stage small networks require less time to converge comparing to training a full network from the beginning.

This latter group of works provided compelling evidence for the benefit of progressively growing the capacity of a network to generate images, although its extension for growing the capacity of a network to learn hierarchical representations has not been explored.

Limited work has considered incremental learning of representations in VAE.

In Gregor et al. (2015) , recurrent networks with attention mechanisms were used to sequentially refines the details in gen-erated images.

It however focused on the generation performance of VAE without considering the learned representations.

In Lezama (2019) , a teacher-student strategy was used to progressively grow the dimension of the latent representations in VAE.

Its fundamental motivation was that, given a teacher model that has learned to effectively disentangle major factors of variations, progressively learning additional nuisance variables will improve generation without compromising the disentangling ability of the teacher -the latter accomplished via a newly-proposed Jacobian supervision.

The capacity of this model to grow, thus, is by design limited to the extraction of nuisance variables.

In comparison, we are interested in a more significant growth of the VAE capacity to progressively learn and improve disentangling of important factors of variations which, as we will later demonstrate, is not what the model in Lezama (2019) is intended for.

In addition, neither of these works considered learning different levels of abstractions at different depths of the network, and the presented pro-VLAE provides a simpler training strategy to achieve progressive representation learning.

Learning disentangled representation is a primary motivation of our work, and an important topic in VAE.

Existing works mainly tackle this by promoting the independence among the learned latent factors in VAE ; Kim & Mnih (2018) ; Chen et al. (2018)).

The presented progressive learning strategy provides a novel approach to improve disentangling that is different to these existing methods and a possibility to augment them in the future.

We assume a generative model p(x, z) = p(x|z)p(z) for observed x and its latent variable z. To learn hierarchical representations of x, we decompose z into {z 1 , z 2 , ..., z L } with z l (l = 1, 2, 3, ..., L) from different abstraction levels that are loosely guided by the depth of neural network as in Zhao et al. (2017) .

We define the hierarchical generative model p θ as:

(1)

Note that there is no hierarchical dependence among the latent variables as in common hierarchical latent variable models.

Rather, similar to that in Rezende et al. (2014) and Zhao et al. (2017) , z l 's are independent and each represents generative factors at an abstraction level not captured in other levels.

We then define an inference model q φ to approximate the posterior as:

where h l (x) represents a particular level of bottom-up abstraction of x. We parameterize p θ and q φ with an encoding-decoding structure and, as in Zhao et al. (2017) , we approximate the abstraction level with the network depth.

The full model is illustrated in Fig. 1 (c), with a final goal to maximize a modified evidence lower bound (ELBO) of the marginal likelihood of data x:

where KL denotes the Kullback-Leibler divergence, prior p(z) is set to isotropic Gaussian N (0, I) according to standard practice, and β is a hyperparameter introduced in to promote disentangling, defaulting to the standrd ELBO objective when β = 1.

We present a progressive learning strategy, as illustrated in Fig. 1 , to achieve the final goal in equation (3) by learning the latent variables z l progressively from the highest (l = L) to the lowest l = 1) level of abstractions.

We start by learning the most abstraction representations at layer L as show in Fig. 1(a) .

In this case, our model degenerates to a vanilla VAE with latent variables z L at the deepest layer.

We keep the dimension of z L small to start small in terms of the capacity to learn latent representations, where we define the inference model at progressive step s = 0 as:

where f e l , µ L , and σ L are parts of the encoder architecture, f d l are parts of the decoder architecture, and D is the distribution of x parametrized by f d 0 (g 0 ), which can be either Bernoulli or Gaussian depending on the data.

Next, as shown in Fig. 1 , we progressively grow the model to learn z L−1 , ..., z 2 , z 1 from high to low abstraction levels.

At each progressive step s = 1, 2, ..., L − 1, we move down one abstraction level, and grow the inference model by introducing new latent code:

Simultaneously, we grow the decoder such that it can generate with the new latent code as:

where m l includes transposed convolution layers outputting a feature map in the same shape as g l+1 , and [·; ·] denotes a concatenation operation.

The training objective at progressive step s is then:

By replacing the full objective in equation (3) with a sequence of the objectives in equation (8) as the training progresses, we incrementally learn to extract and generate with hierarchical latent representations z l 's from high to low levels of abstractions.

Once trained, the full model as shown in Fig. 1 (c) will be used for inference and generation, and progressive processes are no loner needed.

Two important strategies are utilized to implement the proposed progressive representation learning.

First, directly adding new components to a trained network often introduce a sudden shock to the gradient: in VAEs, this often leads to the explosion of the variance in the latent distributions.

To avoid this shock, we adopt the popular method of "fade-in" (Karras et al. (2018) ) to smoothly blend the new and existing network components.

In specific, we introduce a "fade-in" coefficient α to equations (6) and (7) when growing new components in the encoder and the decoder:

where α increases from 0 to 1 within a certain number of iterations (5000 in our experiments) since the addition of the new network components µ l ,σ l , and m l .

Second, we further stabilize the training by weakly constraining the distribution of z l 's before they are added to the network.

This can be achieved by a applying a KL penalty, modulated by a small coefficient γ, to all latent variables that have not been used in the generation at progressive step s:

where γ is set to 0.5 in our experiments.

The final training objective at step s then becomes:

Note that the latent variables at the hierarchy lower than L − s are neither meaningfully inferred nor used in generation at progressive step s, and L pre−trained merely intends to regularize the distribution of these latent variables before they are added to the network.

In the experiments below, we use both "fade-in" and L pre−trained when implementing the progressive training strategy.

Various quantitative metrics for measuring disentanglement have been proposed ; Kim & Mnih (2018) ; Chen et al. (2018)).

For instance, the recently proposed MIG metrics (Chen et al. (2018) ) measures the gap of mutual information between the top two latent dimensions that have the highest mutual information with a given generative factor.

A low MIG score, therefore, suggests an undesired outcome that the same factor is split into multiple dimensions.

However, if different generative factors are entangled into the same latent dimension, the MIG score will not be affected.

Therefore, we propose a new disentanglement metric to supplement MIG by recognizing the entanglement of multiple generative factors into the same latent dimension.

We define MIG-sup as:

where z is the latent variables and v is the ground truth factors,

J is the number of meaningful latent dimensions, and I norm (z j ; v k ) is normalized mutual information I(z j ; v k )/H(v k ).

Considering MIG and MIG-sup together will provide a more complete measure of disentanglement, accounting for both the splitting of one factor into multiple dimensions and the encoding of multiple factors into the same dimension.

In an ideal disentanglement, both MIG and MIG-sup should be 1, recognizing a one-to-one relationship between a generative factor and a latent dimension.

This would have a similar effect to the metric that was proposed in Eastwood & Williams (2018) , although MIG-based metrics do not rely on training extra classifiers or regressors and are unbiased for hyperparameter settings.

The factor metric (Kim & Mnih (2018) ) also has similar properties with MIG-sup, although MIG-sup is stricter on penalizing any amount of other minor factors in the same dimension.

We tested the presented pro-VLAE on four benchmark data sets: dSprites ), 3DShapes (Kim & Mnih (2018) ), MNIST (LeCun et al. (1998) ), and CelebA (Liu et al. (2015) ), where the first two include ground-truth generative factors that allow us to carry out comprehensive quantitative comparisons of disentangling metrics with existing models.

In the following, we first quantitatively compare the disentangling ability of pro-VLAE in comparison to three existing models using three disentanglement metrics.

We then analyze pro-VLAE from the aspects of how it learns progressively, its ability to disentangle, and its ability to learn abstractions at different levels.

Comparisons in quantitative disentanglement metrics: For quantitative comparisons, we considered the factor metric in Kim & Mnih (2018) , the MIG in Chen et al. (2018) , and the MIG-sup presented in this work.

We compared pro-VLAE (changing β) with beta-VAE ), VLAE (Zhao et al. (2017) ) as a hierarchical baseline without progressive training, and the teacherstudent model (Lezama (2019) ) as the most related progressive VAE without hierarchical representations.

All models were considered at different values of β except the teacher-student model: the comparison of β-VAE, VLAE, and the presented pro-VLAE thus also provides an ablation study on the effect of learning hierarchical representations and doing so in a progressive manner.

For fair comparisons, we strictly required all models to have the same number of latent variables and the same number of training iterations.

For instance, if a hierarchical model has three layers that each has three latent dimensions, a non-hierarchical model will have nine latent dimensions; if a progressive method has three progressive steps with 15 epochs of training each, a non-progressive method will be trained for 45 epochs.

Three to five experiments were conducted for each model at each β value, and the average of the top three is used for reporting the quantitative results in Fig. 2 .

As shown, for MIG and MIG-sup, VLAE generally outperformed β-VAE at most β values, while pro-VLAE showed a clear margin of improvement over both methods.

With the factor metric, pro-VLAE was still among the top performers, although with a smaller margin and a larger overlap with VLAE on 3DShapes, and with β-VAE (β = 10) on dSprites.

The teacher-student strategy with Jacobian supervision in general had a low to moderate disentangling score, especially on 3DShapes.

This is consistent with the original motivation of the method for progressively learning nuisance variables after the teacher learns to disentangle effectively, rather than progressively disentangling hierarchical factors of variations as intended by pro-VLAE.

Note that pro-VLAE in general performed better with a smaller value of β (β < 20), suggesting that progressive learning already had an effect of promoting disentangling and a high value of β may over-promote disentangling at the expense of reconstruction quality.

Fig. 3 shows MIG vs. MIG-sup scores among the tested models.

As shown, results from pro-VLAE were well separated from the other three models at the right top quadrant of the plots, obtaining simultaneously high MIG and MIG-sup scores as a clear evidence for improved disentangling ability.

Fig. 4 provides images generated by traversing each latent dimension using the best pro-VLAE (β = 8), the best VLAE (β = 10), and the teacher-student model on 3DShapes data.

As shown, pro-VLAE learned to disentangle the object, wall, and floor color in the deepest layer; the following hierarchy of representations then disentangled objective scale, orientation, and shape, while the lowest-level of abstractions ran out of meaningful generative factors to learn.

In comparison, the VLAE distributed six generative factors over the nine latent dimensions, where color was split across At each progression and for each z l , the row of images are generated by randomly sampling from its prior distributions while fixing the other latent variables (this is NOT traversing).

The green bar at each row tracks the mutual information I(x; z l ), while the total mutual information I(x; z) is labeled on top.

the hierarchy and sometimes entangled with the object scale (in z 2 ).

The teacher-student model was much less disentangled, which we will delve into further in the following section.

To further understand what happened during progressive learning, we use mutual information I(x, z l ) as a surrogate to track the amount of information learned in each hierarchy of latent variables z l during the progressive learning.

We adopted the approach in Chen et al. (2018) to empirically estimate the mutual information by stratified sampling.

Fig. 5 shows an example from 3DShapes.

At progressive step 0, pro-VAE was only learning the deepest latent variables in z 3 , discovering most of the generative factors including color, objective shape, and orientation entangled within z 3 .

At progressive step 1, interestingly, the model was able to "drag" out shape and rotation factors from z 3 and disentangle them into z 2 along with a new scale factor.

Thus I(x; z3) decreased from 10.59 to 6.94 while I(x; z2) increased from 0.02 to 5.98 in this progression, while the total mutual information I(x; z) increased from 10.61 to 12.84, suggesting the overall learning of more detailed information.

Since 3DShapes only has 6 factors, the lowest-level representation z 1 had nothing to learn in progressive step 2, and the allocation of mutual information remained nearly unchanged.

Note that the sum of I(x, z l )'s does not equal to I(x, z) and I over = L 1 I(x, z l ) − I(x, z) suggests the amount of information that is entangled.

In comparison, the teacher-student model was less effective in progressively dragging entangled representations to newly added latent dimensions, as suggested by the slowing changing of I(x, z l )'s Figure 6 : Visualization of hierarchical features learnt for MNIST data.

Each sub-figure is generated by randomly sampling from the prior distribution of z l at one abstraction level while fixing the others.

The original latent code is inferred from a image with digit "0".

From left to right: z 3 encodes the highest abstraction: digit identity; z 2 encodes stroke width; and z 1 encodes other digit styles.

Figure 7: Visualization of hierarchical features learnt for CelebA data.

Each subfigure is generated by traversing along a selected latent dimension in each row within each hierarchy of z l 's. From left to right: latent variables z 4 to z 1 progressively learn major (e.g., gender in z 4 and smile in z 3 ) to minor representations (e.g. wavy-hair in z 2 and eye-shadow in z 1 ) in a disentangled manner.

during progression and the larger value of I over .

This suggests that, since the teacher-student model was motivated for progressively learning nuisance variables, the extent to which its capacity can grow for learning new representations is limited by two fundamental causes: 1) because it increases the dimension of the same latent vectors at the same depth, the growth of the network capacity is limited in comparison to pro-VLAE, and 2) the Jacobian supervision further restricts the student model to maintain the same disentangling ability of the teacher model.

We also qualitatively examined pro-VLAE on data with both relatively simple (MNIST) and complex (CelebA) factors of variations, all done in unsupervised training.

On MNIST (Figure 6 ), while the deepest latent representations encoded the highest-level features in terms of digit identity, the representations learned at shallower levels encoded changes in writing styles.

In Figure 7 , we show the latent representation progressively learned in CelebA from the highest to lowest levels of abstractions, along with disentangling within each level demonstrated by traversing one selected dimension at a time.

These dimensions are selected as examples associated with clear semantic meanings.

As shown, while the deepest latent representation z 4 learned to disentangle high-level features such as gender and race, the shallowest representation z 1 learned to disentangle low-level features such as eye-shadow.

Moreover, the number of distinct representations learned decreased from deep to shallow layers.

While demonstrating disentangling by traversing each individual latent dimension or by hierarchically-learned representations has been separately reported in previous works ; Zhao et al. (2017) ), to our knowledge this is the first time the ability of a model to disentangle individual latent factors in a hierarchical manner has been demonstrated.

This provides evidence that the presented progressive strategy of learning can improve the disentangling of first the most abstract representations followed by progressively lower levels of abstractions.

In this work, we present a progressive strategy for learning and disentangling hierarchical representations.

Starting from a simple VAE, the model first learn the most abstract representation.

Next, the model learn independent representations from high-to low-levels of abstraction by progressively growing the capacity of the VAE deep to shallow.

Experiments on several benchmark data sets demonstrated the advantages of the presented method.

An immediate future work is to include stronger guidance for allocating information across the hierarchy of abstraction levels, either through external multi-scale image supervision or internal information-theoretic regularization strategies.

Figure 8: An example of one factor being encoded in multiple dimensions.

Each row is a traverse for one dimension (dimension order adjusted for better visualization).

Notice that both dim1 and dim2 are encoding floor-color, both dim3 and dim4 are encoding wall-color, and both dim5 and dim6 are encoding object color.

Therefore, the MIG is very low since it penalizes splitting one factor to multiple dimensions.

On the other hand, the MIG-sup and factor-metric is not too bad since one dimension mainly encodes one factor, even though there are some entanglement of color-vs-shape and color-vs-scale.

Figure 9 : An example of one dimension containing multiple factors.

Each row is a traverse for one dimension (dimension order adjusted for better visualization).

Notice that both models achieve high and similar MIG because all 6 factors are encoded and no splitting to multiple dimensions.

However, the right-hand side model has much lower MIG-sup and factor-metric than the left-hand side model.

Because both scale and shape are encoded in dim5, while dim6 has no factor.

Both MIG-sup and factor-metric penalize encoding multiple factors in one dimension.

Besides, our MIG-sup is lower and drops more than factor-metric because MIG-sup is stricter in this case.

Figure 5 of (Zhao et al. (2017) ).

The network has 3 layers and 2 dimensional latent code at each layer.

Each image is generated by traversing each of the two-dimensional latent code in one layer, while randomly sampling from the other layers.

From left to right: The top layer z 3 encodes the digit identity and tilt; z 2 encodes digit width (digits around top-left are thicker than digits around bottom-right); and the bottom layer z 1 encodes stroke width.

Compared to VLAE, the representation learnt in the presented method suggests smoother traversing on digits and similar results for digit width and stroke width.

Table 1 : Mutual information I(x; z l ) between data x and latent codes z l at each l-th depth of the network, corresponding to the qualitative results presented in Fig. 4 and Fig. 6 on 3Dshapes and MNIST data sets.

Both VLAE and the presented pro-VLAE models have the same hierarchical architecture with 3 layers and 3 latent dimensions for each layer.

Compared to VLAE, the presented method allocates information in a more clear descending order owing to the progressive learning.

3DShapes I(x; z 3 ) I(x; z 2 ) I(x; z 1 ) total I(x; z)

In this section, we present additional quantitative results on how information flow among the latent variables during progressive training.

We conducted experiments on both 3DShapes and MNIST data sets, considering different hierarchical architectures including a combination of different number of latent layers L and different number of latent dimensions z dim for each layer.

Each experiment was repeated three times with random initializations, from which the mean and the standard deviation of mutual information I(x; z l ) were computed.

As shown in Tables 3-8 , for all hierarchical architectures, the information amount in each layer is captured in a clear descending order, which aligns with the motivation of the presented progressive learning strategy.

Generally, the information also tends to flow from previous layers to new layers, suggesting a disentanglement of latent factors as new latent layers are added.

This is especially obvious for 3DShapes data where the generative factors are better defined.

In addition, models with small latent codes (z dim = 1) are not able to learn the same amount of information (total I(x, z)) as those with larger latent codes (z dim = 3).

The variance of information in each layer in the former also appears to be high.

We reason that it may be because that the model is trying to squeeze too much information into a small code, resulting in large vibrations during progressive learning.

On the other hand, while a model has large latent codes (L = 4, z dim = 3), the information flow becomes less clear after the addition of certain layers.

Overall, assuming there are K generative factors and there are D dimensions in total available in model, ideally we would like to design the model such that D = K. However, since K is unknown in most data, L and z dim become hyperparameters that need to be tuned for different data sets.

Table 3 : 3DShapes, L = 2, z dim = 3 progressive step I(x; z 2 ) I(x; z 1 ) total I(x; z) 0 10.68 ± 0.19 10.68 ± 0.19 1 7.22 ± 0.30 5.94 ± 0.26 12.88 ± 0.20 Table 4 : 3DShapes, L = 3, z dim = 2 progressive step I(x; z 3 ) I(x; z 2 ) I(x; z 1 ) total I(x; z) 0 10.16 ± 0.13 10.16 ± 0.13 1 9.76 ± 0.05 7.36 ± 0.10 13.00 ± 0.02 2 6.83 ± 1.37 6.66 ± 0.17 5.80 ± 0.41 13.07 ± 0.02

@highlight

We proposed a progressive learning method to improve learning and disentangling latent representations at different levels of abstraction.