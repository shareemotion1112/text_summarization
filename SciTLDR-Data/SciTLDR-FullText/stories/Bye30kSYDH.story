Generative Adversarial Networks (GANs) have become the gold standard when it comes to learning generative models for high-dimensional distributions.

Since their advent, numerous variations of GANs have been introduced in the literature, primarily focusing on utilization of novel loss functions, optimization/regularization strategies and network architectures.

In this paper, we turn our attention to the generator and investigate the use of high-order polynomials as an alternative class of universal function approximators.

Concretely, we propose PolyGAN, where we model the data generator by means of a high-order polynomial whose unknown parameters are naturally represented by high-order tensors.

We introduce two tensor decompositions that significantly reduce the number of parameters and show how they can be efficiently implemented by hierarchical neural networks that only employ linear/convolutional blocks.

We exhibit for the first time that by using our approach a GAN generator can approximate the data distribution without using any activation functions.

Thorough experimental evaluation on both synthetic and real data (images and 3D point clouds) demonstrates the merits of PolyGAN against the state of the art.

Generative Adversarial Networks (GANs) are currently one of the most popular lines of research in machine learning.

Research on GANs mainly revolves around: (a) how to achieve faster and/or more accurate convergence (e.g., by studying different loss functions (Nowozin et al., 2016; Arjovsky & Bottou, 2017; Mao et al., 2017) or regularization schemes (Odena et al., 2018; Miyato et al., 2018; Gulrajani et al., 2017) ), and (b) how to design different hierarchical neural networks architectures composed of linear and non-linear operators that can effectively model high-dimensional distributions (e.g., by progressively training large networks (Karras et al., 2018) or by utilizing deep ResNet type of networks as generators (Brock et al., 2019) ).

Even though hierarchical deep networks are efficient universal approximators for the class of continuous compositional functions (Mhaskar et al., 2016) , the non-linear activation functions pose difficulties in their theoretical analysis, understanding, and interpretation.

For instance, as illustrated in Arora et al. (2019) , element-wise non-linearities pose a challenge on proving convergence, especially in an adversarial learning setting (Ji & Liang, 2018) .

Consequently, several methods, e.g., Saxe et al. (2014) ; Hardt & Ma (2017) ; Laurent & Brecht (2018) ; Lampinen & Ganguli (2019) , focus only on linear models (with respect to the weights) in order to be able to rigorously analyze the neural network dynamics, the residual design principle, local extrema and generalization error, respectively.

Moreover, as stated in the recent in-depth comparison of many different GAN training schemes (Lucic et al., 2018) , the improvements may mainly arise from a higher computational budget and tuning and not from fundamental architectural choices.

In this paper, we depart from the choice of hierarchical neural networks that involve activation functions and investigate for the first time in the literature of GANs the use of high-order polynomials as an alternative class of universal function approximators for data generator functions.

This choice is motivated by the strong evidence provided by the Stone-Weierstrass theorem (Stone, 1948) , which states that every continuous function defined on a closed interval can be uniformly approximated as closely as desired by a polynomial function.

Hence, we propose to model the vector-valued generator function Gpzq : R d ?? R o by a high-order multivariate polynomial of the latent vector z, whose unknown parameters are naturally represented by high-order tensors.

However, the number of parameters required to accommodate all higher-order correlations of the latent vector explodes with the desired order of the polynomial and the dimension of the latent vector.

To alleviate this issue and at the same time capture interactions of parameters across different orders of approximation in a hierarchical manner, we cast polynomial parameters estimation as a coupled tensor factorization (Papalexakis et al., 2016; Sidiropoulos et al., 2017) that jointly factorizes all the polynomial parameters tensors.

To this end, we introduce two specifically tailored coupled canonical polyadic (CP)-type of decompositions with shared factors.

The proposed coupled decompositions of the parameters tensors result into two different hierarchical structures (i.e., architectures of neural network decoders) that do not involve any activation function, providing an intuitive way of generating samples with an increasing level of detail.

This is pictorially shown in Figure 1 .

The result of the proposed PolyGAN using a fourth-order polynomial approximator is shown in Figure 1 (a), while Figure 1 (b) shows the corresponding generation when removing the fourth-order power from the generator.

Our contributions are summarized as follows:

??? We model the data generator with a high-order polynomial.

Core to our approach is to cast polynomial parameters estimation as a coupled tensor factorization with shared factors.

To this end, we develop two coupled tensor decompositions and demonstrate how those two derivations result in different neural network architectures involving only linear (e.g., convolution) units.

This approach reveals links between high-order polynomials, coupled tensor decompositions and network architectures.

??? We experimentally verify that the resulting networks can learn to approximate functions with analytic expressions.

??? We show how the proposed networks can be used with linear blocks, i.e., without utilizing activation functions, to synthesize high-order intricate signals, such as images.

??? We demonstrate that by incorporating activation functions to the derived polynomial-based architectures, PolyGAN improves upon three different GAN architectures, namely DC-GAN (Radford et al., 2015) , SNGAN (Miyato et al., 2018) and SAGAN (Zhang et al., 2019) .

(a) (b) Figure 1: Generated samples by an instance of the proposed PolyGAN.

(a) Generated samples using a fourth-order polynomial and (b) the corresponding generated samples when removing the terms that correspond to the fourth-order.

As evidenced, by extending the polynomial terms, PolyGAN generates samples with an increasing level of detail.

In this Section, we investigate the use of a polynomial expansion as a function approximator for the data generator in the context of GANs.

To begin with, we introduce the notation in Section 2.1.

In Section 2.2, we introduce two different polynomials models along with specifically tailored coupled tensor factorizations for the efficient estimation of their parameters.

Matrices (vectors) are denoted by uppercase (lowercase) boldface letters e.g., X, (x).

Tensors are denoted by calligraphic letters, e.g., X .

The order of a tensor is the number of indices needed to address its elements.

Consequently, each element of an M th-order tensor X is addressed by M indices, i.e., pX q i1,i2,...,i M .

The mode-m unfolding of a tensor X P R I1??I2??????????I M maps X to a matrix X pmq P R Im????m with

I k such that the tensor element x i1,i2,...,i M is mapped to the matrix element x im,j where j " 1`??

n"1 n???m I n .

The mode-m vector product of X with a vector u P R Im , denoted by X??n u P R I1??I2??????????In??1??In`1??????????I N , results in a tensor of order M??1:

(1)

The Khatri-Rao product (i.e., column-wise Kronecker product) of matrices A P R I??N and B P R J??N is denoted by A d B and yields a matrix of dimensions pIJq??N .

The Hadamard product of A P R I??N and B P R I??N is defined as A??B and is equal to A pi,jq B pi,jq for the pi, jq element.

The CP decomposition (Kolda & Bader, 2009; Sidiropoulos et al., 2017) factorizes a tensor into a sum of component rank-one tensors.

An M th-order tensor X P R I1??I2??????????I M has rank-1, when it is decomposed as the outer product of M vectors tu

, where??denotes for the vector outer product.

Consequently, the rank-R CP decomposition of an M th-order tensor X is written as:

where the factor matrices U rms " ru

collect the vectors from the rank-one components.

By considering the mode-1 unfolding of X , the CP decomposition can be written in matrix form as (Kolda & Bader, 2009 ):

More details on tensors and multilinear operators can be found in Kolda & Bader (2009); Sidiropoulos et al. (2017) .

GANs typically consist of two deep networks, namely a generator G and a discriminator D. G is a decoder (i.e., a function approximator of the sampler of the target distribution) which receives as input a random noise vector z P R d and outputs a sample x " Gpzq P R o .

D receives as input both Gpzq and real samples and tries to differentiate the fake and the real samples.

During training, both G and D compete against each other till they reach an "equilibrium" (Goodfellow et al., 2014) .

In practice, both the generator and the discriminator are modeled as deep neural networks, involving composition of linear and non-linear operators (Radford et al., 2015) .

In this paper, we focus on the generator.

Instead of modeling the generator as a composition of linear and non-linear functions, we assume that each generated pixel x i " pGpzqq i may be expanded as a N th order polynomial 1 in z. That is,

1 With an N th order polynomial we can approximate any smooth function (Stone, 1948) .

Under review as a conference paper at ICLR 2020

where the scalar ?? i , and the set of tensors W

are the parameters of the polynomial expansion associated to each output of the generator, e.g., pixel.

Clearly, when n " 1, the weights are d-dimensional vectors; when n " 2, the weights, i.e., W r2s i , form a d??d matrix.

For higher orders of approximation, i.e., when n ?? 3, the weights are n th order tensors.

By stacking the parameters for all pixels, we define the parameters ?? .

.

Consequently, the vector-valued generator function is expressed as:

Intuitively, (5) is an expansion which allows the N th order interactions between the elements of the noise latent vector z. Furthermore, (5) resembles the functional form of a truncated Maclaurin expansion of vector-valued functions.

In the case of a Maclaurin expansion, W rns represent the n th order partial derivatives of a known function.

However, in our case the generator function is unknown and hence all the parameters need to be estimated from training samples.

The number of the unknown parameters in (5) is pd

, which grows exponentially with the order of the approximation.

Consequently, the model of (5) is prone to overfitting and its training is computationally demanding.

A natural approach to reduce the number of parameters is to assume that the weights exhibit redundancy and hence the parameter tensors are of low-rank.

To this end, several low-rank tensor decompositions can be employed (Kolda & Bader, 2009; Sidiropoulos et al., 2017) .

For instance, let the parameter tensors W rns admit a CP decompostion (Kolda & Bader, 2009 ) of mutilinear rank-k, namely, tW rns " rrU rns,1 , U rns,2 , . . .

, U rns,pn`1q ssu N n"1 , with U rns,1 P R o??k , and U rns,m P R d??k , for m " 2, . . .

, n`1.

Then, (5) is expressed as

which has significantly less parameters than (5), especially when k !

d. However, a set of different factor matrices for each level of approximation are required in equation 6, and hence the hierarchical nature of images is not taken into account.

To promote compositional structures and capture interactions among parameters in different orders of approximation we introduce next two coupled CP decompositions with shared factors.

Model 1: Coupled CP decomposition:

Instead of factorizing each parameters tensor individually we propose to jointly factorize all the parameter tensors using a coupled CP decomposition with a specific pattern of factor sharing.

To illustrate the factorization, we assume a third order approximation (N " 3), however in the appendix a generalization to N -th order approximation is provided.

Let us assume that the parameters tensors admit the following coupled CP decomposition with the factors corresponding to lower-order levels of approximation being shared across all parameters tensors.

That is:

??? Let W r1s " CU T r1s , be the parameters for first level of approximation.

??? Let assume W r2s being a superposition of of two weights tensors, namely W r2s " W r2s 1:2??? r2s 1:3 , with W r2s i:j denoting parameters associated with the second order interactions across the i-th and j-th order of approximation.

By enforcing the CP decomposition of the above tensors to share the factor with tensors corresponding to lower-order of approximation we obtain in matrix form: ??? Similarly, we enforce the third-order parameters tensor to admit the following CP decomposition (in matrix form)

Note that all but the U r3s factor matrices are shared in the factorization of tensors capturing polynomial parameters for the first and second order of approximation.

The parameters are C P R o??k , U rms P R d??k for m " 1, 2, 3.

Then, (6) for N " 3 is written as:

The third order approximation of (7) can be implemented as a neural network with the structure of Figure 2 (proved in section B, Claim 1 of the appendix).

It is worth noting that the structure of the proposed network allows for incremental network growth.

Model 2: Coupled nested CP decomposition: Instead of explicitly separating the interactions between layers, we can utilize a joint hierarchical decomposition on the polynomial parameters.

Let us first introduce learnable hyper-parameters b rns P R ?? ( N n"1 , which act as scaling factors for each parameter tensor.

Therefore, we modify (5) to:

with

.

For illustration purposes, we consider a third order function approximation (N " 3).

That is,

To estimate its parameters we jointly factorize all parameters tensors by employing nested CP detecomposion with parameter sharing as follows (in matrix form)

??? First order parameters :

W r1s p1q " CpA r3s d B r3s q T .

??? Second order parametes:

??? Third order parameters:

)

??S r3s ???* T with C P R o??k , A rns P R d??k , S rns P R k??k , B rns P R ????k for n " 1, . . .

, N .

Altogether, (9) is written as:

As we prove in the appendix (section B, Claim 3), (10) can be implemented in a hierarchical manner with a three-layer neural network as shown in Figure 3 .

Comparison between the two models: Both models are based on the polynomial expansion, however there are few differences between those.

The Coupled CP decomposition has a simpler expression, however the Coupled nested CP decomposition relates to standard architectures using hierarchical composition that has recently yielded promising results in GANs (see Section 3).

In the remainder of the paper, we use the Coupled nested CP decomposition by default; in Section G, we include an experimental comparison of the two models.

The experimental comparison demonstrates that neither model outperforms the other in all datasets; they perform similarly.

Figure 3: Schematic illustration of the Coupled nested CP decomposition (for third order approximation).

Symbol??refers to the Hadamard product.

The literature on GANs is vast; we focus only on the works most closely related to ours.

The interested reader can find further information in a recent survey (Creswell et al., 2018) .

Despite the propagation of the noise z to successive layers, the aforementioned works have substantial differences from ours.

We introduce a well-motivated and mathematically elaborate method to achieve a more precise approximation with a polynomial expansion.

In contrast to the previously mentioned works, we also do not concatenate the noise with the feature representations, but rather perform multiplication of the noise with the feature representations, which we mathematically justify.

The work that is most closely related to ours is the recently proposed StyleGAN (Karras et al., 2019) , which is an improvement over the Progressive Growing of GANs (ProGAN) (Karras et al., 2018) .

As ProGAN, StyleGAN is a highly-engineered network that achieves compelling results on synthesized 2D images.

In order to provide an explanation on the improvements of StyleGAN over ProGAN, the authors adopt arguments from the style transfer literature (Huang & Belongie, 2017) .

Nevertheless, the idea of style transfer proposes to use features from images for conditional image translation, which is very different to unsupervised samples (image) generation.

We believe that these improvements can be better explained under the light of our proposed polynomial function approximation.

That is, as we show in Figure 1 , the Hadamard products build a hierachical decomposition with increasing level of detail (rather than different styles).

In addition, the improvements in StyleGAN (Karras et al., 2019) are demonstrated by using a well-tuned model.

In this paper we showcase that without any complicated engineering process the polynomial generation can be applied into several architectures (or any other type of decoders) and consistently improves the performance.

A sequence of experiments in both synthetic data (2D and 3D data manifolds) and higher-dimensional signals are conducted to assess the empirical performance of the proposed polynomial expansion.

The first experiments are conducted on a 2D manifolds that are analytically known (Section 4.1).

Further experiments on three 3D manifolds are deferred to the appendix (Section D).

In Section 4.2, the polynomial expansion is used for synthesizing digits.

Experiments on images beyond digits are conducted in Section E; more specifically, we experiment with images of faces and natural scenes.

The experiments with such images demonstrate how polynomial expansion can be used for learning highly complex distributions by using a single activation function in the generator.

Lastly, we augment our polynomial-based generator with non-linearities and show that this generator is at least as powerful as contemporary architectures.

Apart from the polynomial-based generators, we implemented two variations that are considered baselines: (a) 'Concat': we replace the Hadamard operator with concatenation (used frequently in recent methods, such as in Brock et al. (2019)), (b) '

Orig': the Hadamard products are ditched, while use b r1s ?? z, i.e., there is a composition of linear layers that transform the noise z.

Sinusoidal: We assess the polynomial-based generator on a sinusoidal function in the bounded domain r0, 2??s.

Only linear blocks, i.e., no activation functions, are used in the generator.

That is, all the element-wise non-linearities (such as ReLU's, tanh) are ditched.

The distribution we want to match is a sin x signal.

The input to the generator is z P R and the output is rx, sin xs with x P r0, 2??s.

We assume a 12 th order approximation where each S ris , A ris is a fully-connected layer and B ris is an identity matrix.

Each fully-connected layer has width 15.

In Figure 4 , 2, 000 random samples are synthesized.

We indeed verify that in low-dimensional distributions, such as the univariate sinusoidal, PolyGAN indeed approximates the data distribution quite accurately without using any non-linear activation functions.

The linear generator of the previous section is extended to greyscale images, in which an analytic expression of the ground-truth distribution remains elusive.

To our knowledge, there has not been a generation of greyscale images based on polynomial expansion in the past.

We capitalize on the expressivity of the recent resnet-based generator (Miyato et al., 2018; Brock et al., 2019) , to devise a new polynomial generator Gpzq : R 128 ?? R 32x32 .

We consider a fourth-order approximation (as derived in (5)) where B ris is the identity matrix, S ris is a residual block with two convolutions for i " 1, . . .

, 4.

We emphasize that the residual block as well as all layers are linear, i.e., there are no activation functions.

We only add a tanh in the output of the generator for normalization purposes.

The discriminator and the optimization procedure are the same as in SNGAN; the only difference is that we run one discriminator step per generator step (n dis " 1).

Note that the 'Orig' resnet-based generator resembles the generator of Miyato et al. (2018) in this case.

We perform digit generation (trained on MNIST (LeCun et al., 1998) ).

In Figure 5 , random samples are visualized for the three compared methods.

Note that the two baselines have practically collapsed into a single number each, whereas PolyGAN does synthesize plausible digits.

To further assist the generation process, we utilize the labels and train a conditional GAN.

That is, the class labels are used for conditional batch normalization.

As illustrated in Figure 6 , the samples synthesized are improved over the unsupervised setting. '

Orig' and 'Concat' still suffer from severe mode collapse, while PolyGAN synthesizes digits that have different thickness (e.g. 9), style (e.g. 2) and rotation (e.g. 1).

Figure 6: Conditional digit generation.

Note that both 'Orig' and 'Concat' suffer from severe mode collapse (details in section 4.2).

On the contrary, PolyGAN synthesizes digits that have different thickness (e.g. 9), style (e.g. 2) and rotation (e.g. 1).

We express data generation as a polynomial expansion task.

We model the high-order polynomials with tensorial factors.

We introduce two tailored coupled decompositions and show how the polynomial parameters can be implemented by hierarchical neural networks, e.g. as generators in a GAN setting.

We exhibit how such polynomial-based generators can be used to synthesize images by utilizing only linear blocks.

In addition, we empirically demonstrate that our polynomial expansion can be used with non-linear activation functions to improve the performance of standard state-of-the-art architectures.

Finally, it is worth mentioning that our approach reveals links between high-order polynomials, coupled tensor decompositions and network architectures.

Algorithm 1: PolyGAN (model 1).

% Perform the Hadamard product for the n th layer.

Algorithm 2: PolyGAN (model 2).

for n=2:N do 6 % Multiply with the current layer weight S rns and perform the Hadamard product.

?? "??S rns ??`pB rns q T b rns??????p A rns q T v7 end 8 x " ??`C??.

The appendix is organized as:

??? Section B provides the Lemmas and their proofs required for our derivations.

??? Section C generalizes the Coupled CP decomposition for N th order expansion.

??? Section D extends the experiments to 3D manifolds.

??? In Section E, additional experiments on image generation with linear blocks are conducted.

??? Comparisons with popular GAN architectures are conducted in Section F. Specifically, we utilize three popular generator architectures and devise their polynomial equivalent and perform comparisons on image generation.

We also conduct an ablation study indicating how standard engineering techniques affect the image generation of the polynomial generator.

??? In Section G, a comparison between the two proposed decompositions is conducted on data distributions from the previous Sections.

For a set of matrices tX m P R Im??N u N m"1 the Khatri-Rao product is denoted by:

In this section, we prove the following identity connecting the sets of matrices tA ?? R I????K u N ??"1 and

To demonstrate the simple case with two matrices, we prove first the special case with N " 2.

Lemma 1.

It holds that

Proof.

Initially, both sides of the equation have dimensions of K??L, i.e., they match.

The pi, jq element of the matrix product of pA

Then the pi, jq element of the right hand side (rhs) of (13) is:

A 2,pk2,iq B 2,pk2,jq q "

pA 1,pk1,iq A 2,pk2,iq qpB p1,k1,jq B 2,pk2,jq q

From the definition of Khatri-Rao, it is straightforward to obtain the p??, iq element with ?? " pk 1??1 qI 2`k2 , (i.e. ?? P r1, I 1 I 2 s) of A 1 d A 2 as A 1,pk1,iq A 2,pk2,iq .

Similarly, the p??, jq element of B 1 d B 2 is B 1,pk1,jq B 2,pk2,jq .

The respective pi, jq element of the left hand side (lhs) of (13) is:

In the last equation, we replace the sum in ?? (?? P r1, I 1 I 2 s) with the equivalent sums in k 1 , k 2 .

In a similar manner, we generalize the identity to the case of N ?? 2 terms below.

Lemma 2.

It holds that

Proof.

The rhs includes the Hadamard products of the matrices A T ????B?? .

Each matrix multiplication (A T ????B?? ) results in a matrix of K??L dimensions.

Thus, the rhs is a matrix of K??L dimensions.

The lhs is a matrix multiplication of two Khatri-Rao products.

The first Khatri-Rao product has dimensions K??p ?? ?? I ?? q, while the second p ?? ?? I ?? q??L. Altogether, the lhs has K??L dimensions.

Similarly to the previous Lemma, the pi, jq element of the rhs is:

To proceed with the lhs, it is straightforward to derive that

where s 1 " i and s ?? is a recursive function of the s ????1 .

However, the recursive definition of s ?? is summed in the multiplication and we obtain: Below, we prove that (7) (main paper) is equivalent to the three-layer neural network as shown in Figure 2 .

z.

Then, the form of (7) is equal to:

Proof.

Applying Lemma 2 on (7), we obtain:

The last equation is the same as (21).

In Claim 2 and Claim 3, we prove that (10) (main paper) is equivalent to the three-layer neural network as shown in Figure 3 .

Claim 2.

Let

Proof.

We will prove the equivalence starting from (23) and transform it into (24).

From (23):

where in the last equation, we have applied Lemma 1.

Applying the Lemma once more in the last term of (25), we obtain (24).

with ?? as in Claim 2.

Then, it holds for Gpzq of (10) that Gpzq " ??.

Proof.

Transforming (26) into (10):

To simplify the notation, we define M 1 "

"

??A r1s dB r1s??Sr2s ?? * and M 2 "??A r2s dB r2s??.

The last term of (27) becomes:

Replacing (28) into (27), we obtain (10).

Note that the ?? in Claim 3 is the equation behind Figure 3 .

By proving the claim, we have illustrated how the polynomial generator can be transformed into a network architecture for third-order approximation.

In this Section, we will show how the Coupled CP decomposition generalizes to the N th order approximation.

It suffices to find the decomposition that converts the N th order polynomial into a network structure (see Alg.

1).

As done in Section 2.2, we capture the n th order interactions by decomposing the parameter tensor W rns (with 2 ?? n ?? N ) as:

. . .

The term W rns 1:jn??1:...

:j1 denotes the interactions across the layers 1, j n??1 , . . .

, j 1 .

The N th order approximation becomes:

. . .

By considering the mode-1 unfoding of Coupled CP decomposition (like in Section 2.2), we obtain:

. . .

where we use x N as an abbreviation of the sums.

In the last equation, we have used Lemma 2 (Section B).

Claim 4.

The N th order approximation of (30) can be implemented with a neural network as described in Alg.

1.

Proof.

We will use induction to prove the Claim.

For N " 2, it trivially holds, while the proof for N " 3 is provided in Claim 1.

Suppose it holds for N th order approximation; we prove below that it holds for N`1 th order approximation.

Let us denote the approximation of (30) as G N pzq.

The pN`1q th order approximation from (30) is:

. . . . . . . . .

In the last equation, the first term in the sums is x N ; for the rest two terms we apply Lemma 2:

. . .

The term ?? is equal to the ?? " pn??1q th order of (31), while there is only a single term for n " N .

Therefore, (33) is transformed into:

which is exactly the form described by Alg.

1.

This concludes the induction proof.

Astroid: We implement a superellipse with parametric expression r?? cos 3 t, ?? sin 3 ts for t P r????, ??s.

This has a more complex distribution and four sharp edges.

The random samples are visualized in Figure 7 .

PolyGAN models the data distribution accurately in contrast to the two baselines.

We conduct three experiments in which the data distribution is analytically derived.

The experiments are: Sin3D: The data manifold is an extension over the 2D manifold of the sinusoidal experiment (Section 4.1).

The function we want to learn is Gpzq : R 2 ?? R 3 with the data manifold described by the vector rx, y, sinp10??ax 2`y2 qs for x, y P r??0.5, 0.5s.

In Figure 8 , 20, 000 samples are sampled from the generators and visualized.

PolyGAN captures the data distribution, while 'Orig' and 'Concat' fail.

Swiss roll: The three dimensional vector rt??sin t, y, t??cos ts`0.05??s for t, y P r0, 1s and s " N p0, 1q forms the data manifold 2 .

In Figure 9 , 20, 000 samples are visualized.

Gabriel's Horn: The three dimensional vector rx, ????c os t x , ????s in t x s for t P r0, 160??s and x P r1, 4s forms the data manifold.

The dependence on both sinusoidal and the function 1 x makes this curve challenging for a polynomial expansion.

In Figure 10 , the synthesized samples are plotted.

PolyGAN learns how to generate samples on the manifold despite the fraction in the parametric form.

Apart from digit generation (Section 4.2), we conduct two experiments on image generation of face and natural scenes.

Since both distributions are harder than the digits one, we extend the approximation followed on Section 4.2 by one order, i.e., we assume a fifth-order approximation.

We emphasize that each block is a residual block with no activation functions.

Faces:

In the experiment with faces, we utilize as the training samples the YaleB (Georghiades et al., 2001) dataset.

The dataset includes greyscale images of faces under extreme illuminations.

We rescale all of the images into 64??64 for our analysis.

Random samples are illustrated in Figure 11 .

Our method generates diverse images and captures the case of illuminating either half part of the face, while 'Orig' and 'Concat' generate images that have a dark side only on the left and right side, respectively.

The difference becomes profound in the finer details of the face (please zoom in), where both baselines fail to synthesize realistic semantic parts of the face.

et al., 2001) ) for a generator with linear blocks and a single activation function only on the output (i.e., tan h).

Notice that our method can illuminate either the left or right part of the face, in contrast to 'Orig' (and 'Concat') which generate images that have a dark side only on the left (respectively right) side.

In addition, both 'Orig' and 'Concat' fail to capture the fine details of the facial structure (please zoom in for the details).

Natural scenes: We further evaluate the generation of natural images, specifically by training on CIFAR10 (Krizhevsky et al., 2014) .

CIFAR10 includes 50, 000 training images of 32??32??3 resolution.

In Table 3 , we evaluate the standard metrics of Inception Score (IS) and Frechet Inception Distance (FID) (see more details for the metrics in section F).

Our model outperforms both 'Orig' and 'Concat' by a considerable margin.

In Figure 12 , some random synthesized samples are presented.

To demonstrate the flexibility of the PolyGAN, we utilize three different popular generators.

The three acrhitectures chosen are DCGAN (Radford et al., 2015) , SNGAN (Miyato et al., 2018) , and SAGAN (Zhang et al., 2019) .

Each original generator is converted into a polynomial expansion, while we use the non-linearities to boost the performance of the polynomial generator.

The hyperparameters are kept the same as the corresponding baseline.

Algorithms 3 and 4 succinctly present the key differences of our approach compared to the traditional one (in the case of SNGAN, similarly for other architectures).

In addition to the baseline, we implement the most closely related alternative to our framework, namely instead of using the Hadamard operator as in Figure 3 , we concatenate the noise with the feature representations at that block.

The latter approach is frequently used in the literature (Berthelot et al., 2017; Brock et al., 2019 ) (referred as "Concat" in the paper).

The number of the trainable parameters of the generators are reported in Table 13 .

Our method has only a minimal increase of the parameters, while the concatenation increases the number of parameters substantially.

To reduce the variance often observed during GAN training (Lucic et al., 2018; Odena et al., 2018) , each reported score is averaged over 10 runs utilizing different seeds.

The metrics we utilize are Inception Score (IS) (Salimans et al., 2016) and Frechet Inception Distance (FID) (Heusel et al., 2017) .

Below, we perform an ablation study on Section F.2, and then present the experiments on unsupervised (Section F.3) and conditional image generation (Section F.4) respectively.

Datasets:

We use CIFAR10 (Krizhevsky et al., 2014) and Imagenet (Russakovsky et al., 2015) as the two most widely used baselines for GANs:

??? CIFAR10 (Krizhevsky et al., 2014) includes 60, 000 images of 32??32 resolution.

We use 50, 000 images for training and the rest for testing.

??? Imagenet (Russakovsky et al., 2015) is a large scale dataset that includes over one million training images and 50, 000 validation images.

We reshape the images to 128??128 resolution.

Baseline architectures: The architectures employed are:

??? DCGAN (Radford et al., 2015) , as implemented in https://github.com/pytorch/ examples/tree/master/dcgan.

This is a widely used baseline.

??? SNGAN (Miyato et al., 2018) , as implemented in https://github.com/ pfnet-research/sngan_projection.

SNGAN is a strong performing GAN that introduced a spectral normalization in the discriminator.

??? SAGAN (Zhang et al., 2019) , as implemented in https://github.com/voletiv/ self-attention-GAN-pytorch.

This is a recent network architecture that utilizes the notion of self-attention (Wang et al., 2018) in a GAN setting, achieving impressive results on Imagenet (Russakovsky et al., 2015) .

The default hyper-parameters are left unchanged.

The aforementioned codes are used for reporting the results of both the baseline and our method to avoid any discrepancies, e.g. different frameworks resulting in unfair comparisons.

The source code will be released to enable the reproduction of our results.

Evaluation metrics: The popular Inception Score (IS) (Salimans et al., 2016) and Frechet Inception Distance (FID) (Heusel et al., 2017) are used for the quantitative evaluation.

Both scores extract feature representations from a pretrained classifier (in practice the Inception network (Szegedy et al., 2015) ).

Despite their shortcomings, IS and FID are widely used (Lucic et al., 2018; Creswell et al., 2018) , since alternative metrics fail for generative models (Theis et al., 2016) .

The Inception Score is defined as

where x is a generated sample and ppy|xq is the conditional distribution for labels y. The distribution ppyq over the labels is approximated by 1 M ?? M n"1 ppy|x n q for x n generated samples.

Following the methods in the literature (Miyato et al., 2018) , we compute the inception score for M " 5, 000 generated samples per run (10 splits for each run).

The Frechet Inception Distance (FID) utilizes feature representations from a pretrained network (Szegedy et al., 2015) and assumes that the distributions of these representations are Gaussian.

Denoting the representations of real images as N p?? r , C r q and the generated (fake) as N p?? f , C f q, FID is:

In the experiments, we use M " 10, 000 to compute the mean and covariance of the real images and M " 10, 000 synthesized samples for ?? f , C r .

For both scores the original tensorflow inception network weights are used; the routines of tensorflow.contrib.gan.eval are called for the metric evaluation.

We experimentally define that a (series of) affine transformation(s) on the input noise z are beneficial before using the transformed z for the Hadamard products.

3 These affine transformations are henceforth mentioned as global transformations on z.

The implementation details for each network are the following:

??? DCGAN:

We use a global transformation followed by a RELU non-linearity.

WThe rest details remain the same as the baseline model.

??? SNGAN: Similarly to DCGAN, we use a global transformation with a RELU non-linearity.

We consider each residual block as one order of approximation and compute the Hadamard product after each block (see algorithm 4).

We conduct an ablation study based on SNGAN architecture (or our variant of SNGAN-poly), since most recent methods are based on similar generators Zhang et al. (2019); Brock et al. (2019) .

Unless explicitly mentioned otherwise, the SNGAN is trained on CIFAR10 for unsupervised image generation.

We add a global transformation on z, i.e. a fully-connected layer and use the transformed noise as input to the generator.

In the first experiment, we evaluate whether to add a non-linear activation to the global transformation.

The two alternatives are: i) with linear global transformation ('Ours-linear-global'), i.e. no non-linearity, and ii) with global transformation followed by a RELU non-linearity ('Ours-RELU-global').

The first two results in Table 5 demonstrate that both metrics marginally improve when using a non-linear activation function.

We add this global transformation with RELU on the original SNGAN.

The results are reported in the last two rows of Table 5 (where the original is mentioned as 'Orig', while the alternative of adding a global transformation as 'Original-RELU-global').

Split z into chunks: The recent BigGAN of (Brock et al., 2019) performs hierarchical synthesis of images by splitting the latent vector z into one chunk per resolution (block).

Each chunk is then concatenated into the respective resolution.

We scrutinize this splitting against our method; we split the noise z into pk`1q non-overlapping chunks of equal size for performing k injections.

The injection with splitting is mentioned as 'Injectsplit' below.

Our splitting deteriorates the scores on the task as reported in Table 6 .

It is possible that more elaborate splitting techniques, such as those in Brock et al. (2019) We scrutinize a feature normalization on the baseline of 'Ours-RELU-global'.

For each layer i we divide the A ris z vector with its standard deviation.

The variant with global transformation followed by RELU and normalization before the Hadamard product is called 'Ours-norm'.

The results in Table 7 illustrate that normalization improves the metrics.

In Table 8 , we use 'Ours-RELU-global' as baseline against the model with the skip connection ('Ours-skip').

Since we use SNGAN both for unsupervised/conditional image generation, we verify the aforementioned results in the conditional setting, i.e. when the class information is also provided to the generator and the discriminator.

Normalization before Hadamard product: Similarly to the experiment above, for each layer i we divide the A ris z vector with its standard deviation.

The quantitative results in Table 9 improve the IS score, but the FID deteriorates.

Skip the Hadamard product: Similarly to the aforementioned unsupervised case, we assess the performance if we add a skip connection in the Hadamard.

In Table 10 , the quantitative results comparing the baseline and the skip case are presented.

In this experiment, we study the image generation problem without any labels or class information for the images.

The architectures of DCGAN and resnet-based SNGAN are used for image generation in CIFAR10 (Krizhevsky et al., 2014) .

Table 11 summarizes the results of the IS/FID scores of the compared methods.

In all of the experiments, PolyGAN outperforms the compared methods.

Frequently class information is available.

We can utilize the labels, e.g. use conditional batch normalization or class embeddings, to synthesize images conditioned on a class.

We train two networks, et al., 2015) .

SAGAN uses self-attention blocks (Wang et al., 2018) to improve the resnet-based generator.

Despite our best efforts to show that our method is both architecture and database agnostic, the recent methods are run for hundreds of thousands or even million iterations till "convergence".

In SAGAN the authors report that for each training multiple GPUs need to be utilized for weeks to reach the final reported Inception Score.

We report the metrics for networks that are run with batch size 64 (i.e., four times less than the original 256) to fit in a single 16GB NVIDIA V100 GPU.

Following the current practice in ML, due to the lack of computational budget (Hoogeboom et al., 2019), we run SAGAN for 400, 000 iterations (see Figure 3 of the original paper for the IS during training) 4 .

Each such experiment takes roughly 6 days to train.

The FID/IS scores of our approach compared against the baseline method can be found in Table 12 .

In both cases, our proposed method yields a higher Inception Score and a lower FID.

An experimental comparison of the two models described in Section 2 is conducted below.

Unless explicitly mentioned otherwise, the networks used below do not include any non-linear activation functions, they are polynomial expansions with linear blocks.

We use the following four experiments:

Sinusoidal on 2D: The data distribution is described by rx, sinpxqs with x P r0, 2??s (see Section 4.1 for further details).

We assume 8 th order approximation for Coupled CP decomposition and 12 th order for Coupled nested CP decomposition.

Both have width 15 units.

The comparison between the two models in Figure 13 demonstrates that they can both capture the data manifold.

Impressively, the Coupled CP decomposition does not synthesize a single point that is outside of the manifold.

Astroid: The data distribution is described on Section D. The samples comparing the two models are visualized in Figure 15 .

Sin3D: The data distribution is described on Section D. In Figure 15 the samples from the two models are illustrated.

Swiss roll: The data distribution is described on Section D. In Figure 16 the samples from the two models are illustrated.

We conduct an experiment on images to verify that both architectures can learn higher-dimensional distributions.

We select the digit images as described in Section 4.2.

In this case, Coupled CP decomposition is implemented as follows: each U ris is a series of linear convolutions with stride 2 for i " 1, . . .

, 4, while C is a linear residual block.

We emphasize that in both models all the activation functions are removed and there is a single tanh in the output of the generator for normalization purposes.

@highlight

We model the data generator (in GAN) by means of a high-order polynomial represented by high-order tensors.