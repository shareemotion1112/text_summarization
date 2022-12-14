Deep generative modeling using flows has gained popularity owing to the tractable exact log-likelihood estimation with efficient training and synthesis process.

However, flow models suffer from the challenge of having high dimensional latent space, same in dimension as the input space.

An effective solution to the above challenge as proposed by Dinh et al. (2016) is a multi-scale architecture, which is based on iterative early factorization of a part of the total dimensions at regular intervals.

Prior works on generative flows involving a multi-scale architecture perform the dimension factorization based on a static masking.

We propose a novel multi-scale architecture that performs data dependent factorization to decide which dimensions should pass through more flow layers.

To facilitate the same, we introduce a heuristic based on the contribution of each dimension to the total log-likelihood which encodes the importance of the dimensions.

Our proposed heuristic is readily obtained as part of the flow training process, enabling versatile implementation of our likelihood contribution based multi-scale architecture for generic flow models.

We present such an implementation for the original flow introduced in Dinh et al. (2016), and demonstrate improvements in log-likelihood score and sampling quality on standard image benchmarks.

We also conduct ablation studies to compare proposed method with other options for dimension factorization.

Deep Generative Modeling aims to learn the embedded distributions and representations in input (especially unlabelled) data, requiring no/minimal human labelling effort.

Learning without knowledge of labels (unsupervised learning) is of increasing importance because of the abundance of unlabelled data and the rich inherent patterns they posses.

The representations learnt can then be utilized in a number of downstream tasks such as semi-supervised learning Odena, 2016) , synthetic data augmentation and adversarial training (Cisse et al., 2017) , text analysis and model based control etc.

The repository of deep generative modeling majorly includes Likelihood based models such as autoregressive models (Oord et al., 2016b; Graves, 2013) , latent variable models (Kingma & Welling, 2013) , flow based models (Dinh et al., 2014; 2016; Kingma & Dhariwal, 2018) and implicit models such as generative adversarial networks (GANs) (Goodfellow et al., 2014) .

Autoregressive models (Salimans et al., 2017; Oord et al., 2016b; a; achieve exceptional log-likelihood score on many standard datasets, indicative of their power to model the inherent distribution.

But, they suffer from slow sampling process, making them unacceptable to adopt in real world applications.

Latent variable models such as variational autoencoders (Kingma & Welling, 2013) tend to better capture the global feature representation in data, but do not offer an exact density estimate.

Implicit generative models such as GANs which optimize a generator and a discriminator in a min-max fashion have recently become popular for their ability to synthesize realistic data (Karras et al., 2018; Engel et al., 2019) .

But, GANs do not offer a latent space suitable for further downstream tasks, nor do they perform density estimation.

Flow based generative models (Dinh et al., 2016; Kingma & Dhariwal, 2018) perform exact density estimation with fast inference and sampling, due to their parallelizability.

They also provide an information rich latent space suitable for many applications.

However, the dimension of latent space for flow based generative models is same as the high-dimensional input space, by virtue of bijectivity nature of flows.

This poses a bottleneck for flow models to scale with increasing input dimensions due to computational complexity.

An effective solution to the above challenge is a multi-scale architecture, introduced by Dinh et al. (2016) , which performs iterative early gaussianization of a part of the total dimensions at regular intervals of flow layers.

This not only makes the model computational and memory efficient but also aids in distributing the loss function throughout the network for better training.

Many prior works including Kingma & Dhariwal (2018) ; Atanov et al. (2019) ; Durkan et al. (2019) ; implement multi-scale architecture in their flow models, but use static masking methods for factorization of dimensions.

We propose a multi-scale architecture which performs data dependent factorization to decide which dimensions should pass through more flow layers.

For the decision making, we introduce a heuristic based on the amount of total log-likelihood contributed by each dimension, which in turn signifies their individual importance.

We lay the ground rules for quantitative estimation and qualitative sampling to be satisfied by an ideal factorization method for a multi-scale architecture.

Since in the proposed architecture, the heuristic is obtained as part of the flow training process, it can be universally applied to generic flow models.

We present such implementations for flow models based on affine/additive coupling and ordinary differential equation (ODE) and achieve quantitative and qualitative improvements.

We also perform ablation studies to confirm the novelty of our method.

Summing up, the contributions of our research are,

1.

A log-determinant based heuristic which entails the contribution by each dimensions towards the total log-likelihood in a multi-scale architecture.

2.

A multi-scale architecture based on the above heuristic performing data-dependent splitting of dimensions, implemented for several classes of flow models.

3.

Quantitative and qualitative analysis of above implementations and an ablation study To the best of our knowledge, we are the first to propose a data-dependent splitting of dimensions in a multi-scale architecture.

In this section, we illustrate the functioning of flow based generative models and the multiscale architecture as introduced by Dinh et al. (2016) .

Let x be a high-dimensional random vector with unknown true distribution p(x).

The following formulation is directly applicable to continous data, and with some pre-processing steps such as dequantization (Uria et al., 2013; Salimans et al., 2017; Ho et al., 2019) to discrete data.

Let z be the latent variable with a known standard distribution p(z), such as a standard multivariate gaussian.

Using an i.i.d.

dataset D, the target is to model p ?? (x) with parameters ??.

A flow, f ?? is defined to be an invertible transformation that maps observed data x to the latent variable z. A flow is invertible, so the inverse function T maps z to x, i.e.

The log-likelihood can be expressed as,

where ???f ?? (x) ???x T is the Jacobian of f ?? at x. The invertibile nature of flow allows it to be capable of being composed of other flows of compatible dimensions.

In practice, flows are constructed by composing a series of component flows.

Let the flow f ?? be composed of K component flows, i.e.

and the intermediate variables be denoted by y K , y K???1 , ?? ?? ?? , y 0 = x.

Then the log-likelihood of the composed flow is,

Log-latent density

which follows from the fact that det(A ?? B) = det(A) ?? det(B).

In our work, we refer the first term in Equation 4 as log-latent-density and the second term as log-determinant (log-det).

The reverse path, from z to x can be written as a composition of inverse flows,

Confirming with the properties of a flow as mentioned above, different types of flows can be constructed (Kingma & Dhariwal, 2018; Dinh et al., 2016; Behrmann et al., 2018) .

Multi-scale architecture is a design choice for latent space dimensionality reduction of flow models, in which part of the dimensions are factored out/early gaussianized at regular intervals, and the other part is exposed to more flow layers.

The process is called dimension factorization.

In the problem setting as introduced in Section 2.1, the factoring operation can be mathematically expressed as,

The factoring of dimensions at early layers has the benefit of distributing the loss function throughout the network (Dinh et al., 2016) and optimizing the amount of computation and memory used by the model.

We consider the multi-scale architecture for flow based generative models as introduced by Dinh et al. (2016) (and later used by state-of-the-art flow models such as Glow (Kingma & Dhariwal, 2018) ) as the base of our research work.

In a multi-scale architecture, it is apparent that the network will better learn the distribution of variables getting exposed to more layers of flow as compared to the ones which get factored at a finer scale (earlier layer).

The method of dimension splitting proposed by prior works such as (Dinh et al., 2016; Kingma & Dhariwal, 2018; Behrmann et al., 2018) are static in nature and do not distinguish between importance of different dimensions.

In this section, we introduce a heuristic to estimate the contribution of each dimension towards the total log-likelihood, and introduce a method which can use the heuristic to decide the dimensions to be factored at an earlier layer, eventually achieving preferrential splitting in multiscale architecture.

Our approach builds an efficient multiscale architecture which factors the dimensions at each flow layer in a way such that the local variance in the input space is well captured as the flow progresses and the log-likelihood is maximized.

We also describe how our multi-scale architecture can be implemented over several standard flow models.

Recall from Equation 4 that the log-likelihood is composed of two terms, the log-latent-density term and the log-det term.

The log-latent-density term depends on the choice of latent distribution whereas the log-det term depends on the modeling of the flow layers.

So, careful design of flow layers can lead to maximized log-determinant, eventually maximizing the likelihood.

The total log-det term is nothing but the sum of log-det terms contributed by each dimension.

Let the dimension of the input space x be s ?? s ?? c, where s is the image height/width and c is the number of channels for image inputs.

For the following formulation, let us assume no dimensions were gaussianized early so that we have access to log-det term for all dimensions at each flow layer, and the dimension at all intermediate layer remains same (i.e. s ?? s ?? c).

We apply a flow (f ?? ) with K component flows to

The intermediate variables are denoted by y K , y K???1 , ?? ?? ?? , y 0 with y K = z (since no early gaussianization was done) and y 0 = x. The log-det term at layer l, L

The log-det of the jacobian term encompasses contribution by all the s ?? s ?? c dimensions.

We decompose it to obtain the individual contribution by variables (dimensions) towards the total log-det (??? total log-likelihood).

The log-det term can be viewed (with slight abuse of notations) as a s ?? s ?? c tensor corresponding to each of the dimensions, summed over the flow layers till l,

]

s??s??c , where ??, ?? ??? {0, ?? ?? ?? , s} and ?? ??? {0, ?? ?? ?? , c} (10) s.t.

The entries in [L

d ] s??s??c having higher value correspond to the variables which contribute more towards the total log-likelihood, hence are more valuable for better flow formulation.

So, we can use the likelihood contribution (in the form of log-det term) by each dimension as a heuristic for deciding which variables should be gaussianized early in a multi-scale architecture.

Ideally, at each flow layer, the variables with more log-det term should be exposed to more layer of flow and the ones having less log-det term should be factored out.

In this manner, selectively more power can be provided to variables which capture meaningful representation (and are more valuable from log-det perspective) to be expressive by being exposed to multiple flow layers.

This formulation leads to enhanced density estimation performance.

Additionally, for many datasets such as images, the spatial nature should be taken into account while deciding dimensions for early gaussianization.

Summarily, at every flow layer, an ideal factorization method should, 1. (Quantitative) For efficient density estimation: Early gaussianize the variables having less log-det and expose the ones having more log-det to more flow layers 2. (Qualitative) For qualitative reconstruction: Capture the local variance over the flow layers, i.e. the dimensions being exposed to more flow layers should contain representative pixel variables from throughout the whole image.

Keeping the above requirements in mind, variants of hybrid techniques for factorization can be implemented for different types of flow models which involve a multi-scale architecture, to improve their density estimation and qualitative performance.

The key requirement is availability of log-det contributions per dimension, which can be fulfilled by decomposition of the log-det of the jacobian.

We refer to the method as Likelihood Contribution based Multi-scale Architecture (LCMA).

The steps of LCMA implementation for flow models is summarized in Algorithm 1.

Note that in step 2 of dimension factorization phase in algorithm 1, we group the dimensions having more/less log-det locally and then perform splitting.

This preserves the local spatial variation of the image in both parts of the factorization, leveraging both enhanced density estimation as well as qualitative reconstruction.

Another important observation is since the factorization of dimensions does not occur during the training time, and before the actual training starts, the decision of dimensions which get factored at each flow layer is fixed, the change of variables formula can be applied.

This allows the use of non-invertible operations (e.g. max and min pooling) for efficient factorization with log-det heuristic.

Step 1 of dimension factorization phase requires computation of individual contribution of dimensions ([L (l) d ] s??s??c ) towards the total log-likelihood, which can vary depending on the original design of flow Algorithm 1: LCMA implementation for generative flow models Pre-Training Phase: Pre-train a network with no multiscale architecture (no dimensionality reduction) to obtain the log-det term at every flow layer.

Dimension Factorization: In this phase, the dimensions to be factored at each flow layer is decided based on the log-det term at that layer

is computed specifically for corresponding flow models (Refer Section 3.1 and Section 3.2).

shaped tensor using local max and min-pooling (= ???max-pooling(???input)) operations (Figure 1) at each flow layer.

3.

Among the 4c channels, one half contains the dimensions having more log-det term compared with its neighbourhood pixel (Black marked in Fig. 1 ), while the other half contains the dimensions having less log-det (White marked in Fig. 1 ).

4.

Split the tensor along the channel dimension to two parts.

5.

Forward the corresponding dimensions contributing more towards likelihood into more flow layers and early gaussianize the ones contributing less.

6.

Repeat steps 1-5 for all the layers with dimensions passed to that layer till the latent space.

Training Phase:

The decision of dimensions to be factored at each layer as performed in previous step remains fixed.

Finally, the flow model with proposed LCMA is trained.

models.

Some flow models offer direct decomposition of jacobian into per-dimension components, whereas for others, an indirect estimation method has to be adopted.

We now describe such methods to obtain such individual likelihood contribution of dimensions for flow models based on affine coupling (RealNVP (Dinh et al., 2016) and Glow (Kingma & Dhariwal, 2018)), and flow models involving ordinary differential equation (ODE) based density estimators (i-ResNet (Behrmann et al., 2018) ), all of which involve a multiscale architecture.

RealNVP (Dinh et al., 2016) :

For RealNVP with afffine coupling layers, the logarithm of individual diagonal elements of jacobian, summed over layers till layer l provides the per-dimensional likelihood contribution components at layer l.

Glow (Kingma & Dhariwal, 2018) :

Unlike RealNVP where the log-det terms for each dimension can be expressed as log of corresponding diagonal element of jacobian, Glow contains 1 ?? 1 convolution blocks having non-diagonal log-det term for channel dimensions, for a s ?? s ?? c tensor h given by,

It remains to decompose the log | det(W)| to individual contribution by each channel.

As a suitable candidate, singular values of W correspond to the contribution from each channel dimension, so their log value is the individual log-det contribution.

So the individual log-det term for channels are obtained by,

where ?? i (W) are the singular values of the weight matrix W. For affine blocks in Glow, same method as RealNVP is adopted.

Recent works on flow models such as Behrmann et al. (2018) ; employ variants of ODE based density estimators.

We introduce method to find perdimensional likelihood contribution for i-ResNet (Behrmann et al., 2018) , which is a residual network with invertibility and efficient jacobian computation properties.

i-ResNet is modelled as a flow F (x), such that z = F (x) = (I + g)(x), where g(x) is the forward propagation function.

The log-likelihood expression is written with the log-det of the jacobian is expressed as a power series,

where tr denotes the trace.

Due to computational constraints, the power series is computed up to a finite number of iterations with the tr(J

) is the vector-jacobian product which is multiplied again with v. The individual components which are summed when (v T J k g ) is multiplied with v correspond to the diagonal terms in jacobian, over the expectation E p(v) .

So those terms are the contribution by the individual dimensions, to the log-likelihood and are expressed as [L

Multi-scale architecture and variants have been successful in a number of prior works in deep generative modeling.

For invertible neural networks, use a keepChannel for selective feed forward of channels analogous to multi-scaling.

In the spectrum of generative flow models, multi-scale architecture has been utilized to achieve the dimensionality reduction and enhanced training because of the distribution of loss function in the network (Dinh et al., 2016; Kingma & Dhariwal, 2018) .

A variant of multiscale architecture has been utilized to capture local variations in auto-regressive models (Reed et al., 2017) .

Among GAN (Goodfellow et al., 2014) models, Denton et al. (2015) use a multiscale variant to generate images in a coarse-to-fine manner.

For multi-scale architectures in generative flow models, our proposed method performs factorization of dimensions based on their likelihood contribution, which in another sense translates to determining which dimensions are important from density estimation and qualitative reconstruction point of view.

Keeping this in mind, we discuss prior works on generative flow models which involve multi-scaling and/or incorporate permutation among dimensions to capture their interactions. (2018) introduce an 1 ?? 1 convolution layer in between the actnorm and affine coupling layer in their flow architecture.

The 1 ?? 1 convolution is a generalization of permutation operation which ensures that each dimension can affect every other dimension.

This can be interpreted as redistributing the contribution of dimensions to total likelihood among the whole space of dimensions.

So Kingma & Dhariwal (2018) treat the dimensions as equiprobable for factorization in their implementation of multi-scale architecture, and split the tensor at each flow layer evenly along the channel dimension.

We, on the other hand, take the next step and focus on the individuality of dimensions and their importance from the amount they contribute towards the total log-likelihood.

The log-det score is available via direct/indirect decomposition of the jacobian obtained as part of computations in a flow training, so we essentially have a heuristic for free.

Since our method focuses individually on the dimensions using a heuristic which is always available, it can prove to be have more versatility in being compatible with generic multi-scale architectures.

Hoogeboom et al. (2019) extend the concept of 1 ?? 1 convolutions to invertible d ?? d convolutions, but do not discuss about multi-scaling.

Dinh et al. (2016) also include a type of permutation which is equivalent to reversing the ordering of the channels, but is more restrictive and fixed.

Flow models such as Behrmann et al. (2018) ; ; involve ODE based density estimators.

They also implement a multi-scale architecture, but the splitting operation is a static channel wise splitting without considering importance of individual dimensions or any permutations. ; Durkan et al. (2019) ; ; Atanov et al. (2019) use multi-scale architecture in their flow models, coherent with Dinh et al. (2016) ; Kingma & Dhariwal (2018) , but perform the factorization of dimensions without any consideration of the individual contribution of the dimension towards the total log-likelihood.

For qualitative sampling along with efficient density estimation, we also propose that factorization methods should preserve spatiality of the image in the two splits, motivated by the spatial nature of splitting methods in Kingma & Dhariwal (2018) (channel-wise splitting) and Dinh et al. (2016) (checkerboard and channel-wise splitting).

In Section 3, we established that our proposed likelihood contribution based factorization of dimensions can be implemented for flow models involving a multi-scale architecture, in order to improve their density estimation and qualitative performance.

In this section we present the detailed results of proposed LCMA adopted for the flow model of RealNVP (Dinh et al., 2016) and quantitative comparisons with Glow (Kingma & Dhariwal, 2018) and i-ResNet (Behrmann et al., 2018) .

For direct comparison, all the experimental settings such as data pre-processing, optimizer parameters as well as flow architectural details (coupling layers, residual blocks) are kept the same, except that the factorization of dimensions at each flow layer is performed according to the methods described in Section 3.

For ease of access, we also have summarized the experimental details in Appendix A.

For RealNVP, we perform experiments on four benchmarked image datasets: CIFAR-10 (Krizhevsky, 2009), Imagenet (Russakovsky et al., 2014 ) (downsampled to 32 ?? 32 and 64 ?? 64), and CelebFaces Attributes (CelebA) (Liu et al., 2015) .

The scaling in LCMA is performed once for CIFAR-10, thrice for Imagenet 32 ?? 32 and 4 times for Imagenet 64 ?? 64 and CelebA. We compare LCMA with conventional RealNVP and report the quantitative and qualitative results.

For Glow and i-ResNet with LCMA, we perform experiments on CIFAR-10 and present improvements over baseline bits/dim.

We also perform an ablation studies for LCMA vs. other possible dimension splitting options.

The bits/dim scores of RealNVP with conventional multi-scale architecture (as introduced in Dinh et al. (2016) ) and RealNVP with LCMA are given in Table 1 .

It can be observed that the density estimation results using LCMA is in all cases better in comparison to the baseline.

We observed that the improvement for CelebA is relatively high as compared to natural image datasets.

This observation was expected as facial features often contain high redundancy and the flow model learns to put more importance (reflected in terms of high log-det) on selected dimensions that define the facial features.

Our proposed LCMA exposes such dimensions to more flow layers, making them more expressive and hence the significant improvement in code length (bits/dim) is observed.

The improvement in bits/dim is less for natural image datasets because of the high variance among features defining them, which has been the challenge with image compression algorithms.

Note that the improvement in density estimation is always relative to the original flow architecture (RealNVP in our case) over which we use our proposed LCMA, as we do not alter any architecture other than the dimension factorization method.

The quantitative results of LCMA implementation for RealNVP, Glow and i-ResNet with CIFAR-10 dataset is summarized in Table 2 .

The density estimation scores for flows with LCMA outperform the same flow with conventional multi-scale architectures.

ImageNet 64x64

RealNVP (Dinh et al., 2016)

An ideal dimension factorization method should capture the local variance over series of flow layers, which helps in qualitative sampling.

For LCMA implementation, we introduced local max and min pooling operations on log-det heuristic to decide which dimensions to be gaussianized early (Section 3).

We performed ablation studies to compare LCMA with other methods for dimension factorization in a multi-scale architecture.

We consider 4 variants for our study, namely fixed random permutation (Case 1), multiscale architecture with early gaussianization of high log-det dimensions (Case 2), factorization method with checker-board and channel splitting as introduced in RealNVP (Case 3) and multiscale architecture with early gaussianization of low log-det dimensions, which is our proposed LCMA (Case 4).

In fixed random permutation, we randomly partition the tensor into two halves, with no regard to the spatiality or log-det score.

In case 2, we do the reverse of LCMA, and gaussianize the high log-det variables early.

The bits/dim score and generated samples for each of the method are given in Table 3 .

As expected from an information theoretic perspective, gaussianizing high log-det variables early provides the worst density estimation, as the model could not capture the high amount of important information.

Comparing the same with fixed random permutation, the latter has better score as the probability of a high log-det variable being gaussianized early reduces to half, and it gets further reduced with RealNVP due to channel-wise and checkerboard splitting.

LCMA has the best score among all methods, as the variables carrying more information are exposed to more flow layers.

Fixed random permutation has the worst quality of sampled images, as the spatiality is lost during factorization.

The sample quality improves for Case 2 and RealNVP.

The sampled images are perceptually best for LCMA.

Summarizing, LCMA outperforms multi-scale architectures based on other factorization methods, as it improves density estimation and generates qualitative samples.

We proposed a novel multi-scale architecture for generative flows which employs a data-dependent splitting based the individual contribution of dimensions to the total log-likelihood.

Implementations of the proposed method for several state-of-the-art flow models such as RealNVP (Dinh et al., 2016) , Glow(Kingma & Dhariwal, 2018) and i-ResNet (Behrmann et al., 2018) were presented.

Empirical studies conducted on benchmark image datasets validate the strength of our proposed method, which improves log-likelihood scores and is able to generate qualitative samples.

Ablation study results confirm the power of LCMA over other options for dimension factorization.

For direct comparison with Dinh et al. (2016) , data pre-processing, optimizer parameters as well as flow architectural details (coupling layers, residual blocks) are kept the same, except that the factorization of dimensions at each flow layer is performed according to the method described in Section 3.

In this section, for the ease of access, we summarize the experimental settings.

We perform experiments on four benchmarked image datasets: CIFAR-10 (Krizhevsky, 2009), Imagenet (Russakovsky et al., 2014 ) (downsampled to 32 ?? 32 and 64 ?? 64), and CelebFaces Attributes (CelebA) (Liu et al., 2015) .

Pre-processing:

For CelebA, we take a central crop of 148 ?? 148 then resize it to 64 ?? 64.

Flow model architecture: We use affine coupling layers as introduced (Dinh et al., 2016) .

A layer of flow is defined as 3 coupling layers with checkerboard splits at s ?? s resolution, 3 coupling layers with channel splits at s/2 ?? s/2 resolution, where s is the resolution at the input of that layer.

For datasets having resolution 32, we use 3 such layers and for those having resolution 64, we use 4 layers.

The cascade connection of the layers is followed by 4 coupling layers with checkerboard splits at the final resolution, marking the end of flow composition.

For CIFAR-10, each coupling layer uses 8 residual blocks.

Other datasets having images of size 32 ?? 32 use 4 residual blocks whereas 64 ?? 64 ones use 2 residual blocks.

More details on architectures will be given in a source code release.

@highlight

Data-dependent factorization of dimensions in a multi-scale architecture based on contribution to the total log-likelihood