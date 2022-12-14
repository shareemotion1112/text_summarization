We study the problem of learning to map, in an unsupervised way, between domains $A$ and $B$, such that the samples $

\vb \in B$ contain all the information that exists in samples $\va\in A$ and some additional information.

For example, ignoring occlusions, $B$ can be people with glasses, $A$ people without, and the glasses, would be the added information.

When mapping a sample $\va$ from the first domain to the other domain, the missing information is replicated from an independent reference sample $\vb\in B$. Thus, in the above example, we can create, for every person without glasses a version with the glasses observed in any face image.



Our solution employs a single two-pathway encoder and a single decoder for both domains.

The common part of the two domains and the separate part are encoded as two vectors, and the separate part is fixed at zero for domain $A$.

The loss terms are minimal and involve reconstruction losses for the two domains and a domain confusion term.

Our analysis shows that under mild assumptions, this architecture, which is much simpler than the literature guided-translation methods, is enough to ensure disentanglement between the two domains.

We present convincing results in a few visual domains, such as no-glasses to glasses, adding facial hair based on a reference image, etc.

In the problem of unsupervised domain translation, the algorithm receives two sets of samples, one from each domain, and learns a function that maps between a sample in one domain to the analogous sample in the other domain BID37 BID5 BID28 BID29 BID10 BID11 BID38 b; BID26 .

The term unsupervised means, in this context, that the two sets are unpaired.

In this paper, we consider the problem of domain B, which contains a type of content that is not present at A. As a running example, we consider the problem of mapping between a face without eyewear (domain A) to a face with glasses (domain B).

While most methods would map to a person with any glasses, our solution is guided and we attach to an image a ∈ A, the glasses that are present in a reference image b ∈ B.In comparison to other guided image to image translation methods, our method is considerably simpler.

It relies on having a latent space with two parts: (i) a shared part that is common to both A and B, and (ii) a specific part that encodes the added content in B. By setting the second part to be the zero vector for all samples in A, a disentanglement emerges.

Our analysis shows that this Table 1 : A comparison to other unsupervised guided image to image translation methods.† k = 5 is the number of pre-segmented face parts.‡ Used for domain confusion, not on the output.

MUNIT EG-UNIT BID30 DRIT BID27 PairedCycleGAN (Chang' domains.

The networks are of four types: encoders, which map images to a latent space, generators (also known as decoders), which generate images from a latent representation, discriminators that are used as part of an adversarial loss, and other, less-standard, networks.

It is apparent that our method is considerably simpler than the literature methods.

The main reason is that our method is based on the emergence of disentanglement, as detailed in Sec. 4.

This allows us to to train with many less parameters and without the need to apply excessive tuning, in order to balance or calibrate the various components of the compound loss.

The MUNIT architecture by , like our architecture, employs a shared latent space, in addition to a domain specific latent space.

Their architecture is not limited to two domains 1 and unlike ours, employs separate encoders and decoders for the various domains.

The type of guiding that is obtained from the target domain in MUNIT is referred to as style, while in our case, the guidance provides content.

Therefore, MUNIT, as can be seen in our experiments, cannot add specific glasses, when shifting from the no-glasses domain to the faces with eyewear domain.

The EG-UNIT architecture by BID30 presents a few novelties, including an adaptive method of masking-out a varying set of the features in the shared latent space.

In our latent representation of domain A, some of the features are constantly zero, which is much simpler.

This method also focuses on guiding for style and not for content, as is apparent form their experiments.

The very recent DRIT work by BID27 learns to map between two domains using a disentangled representation.

Unlike our work, this work seems to focus on style rather than content.

The proposed solution differs from us in many ways: (1) it relies on two-way mapping, while we only map from A to B. (2) it relies on shared weights in order to ensure that the common representation is shared.

(3) it adds a VAE-like BID24 statistical characterization of the latent space, which results in the ability to sample random attributes.

As can be seen in Tab.

1, the solution of BID27 is considerably more involved than our solution.

DRIT (and also MUNIT) employ two different types of encoders that enforce a separation of the latent space representations to either style or content vectors.

For example, the style encoder, unlike the content encoder, employs spatial pooling and it also results in a smaller representation than the content one.

This is important, in the context of these methods, in order to ensure that the two representations encode different aspects of the image.

If DRIT or MUNIT were to use the same type of encoder twice, then one encoder could capture all the information, and the image-based guiding (mixing representations from two images) would become mute.

In contrast, our method (i) does not separate style and content, and (ii) has a representation that is geared toward capturing the additional content.

The work most similar to us in its goal, but not in method, is the PairedCycleGAN work by BID7 .

This work explores the single application of applying the makeup of a reference face to a source face image.

Unfortunately, the method was only demonstrated on a proprietary unshared dataset and the code is also not publicly available, making a direct comparison impossible at this time.

The method itself is completely different from ours and does not employ disentanglement.

Instead, a generator with two image inputs is used to produce an output image, where the makeup is transfered between the input images, and a second generator is trained to remove makeup.

The generation is done separately to k = 5 pre-segmented facial regions, and the generators do not employ an encoder-decoder architecture.

Lastly, there are guided methods, which are trained in the supervised domain, i.e., when there are matches between domain A and B. Unlike the earlier one-to-one work, such as pix2pix BID22 , these methods produce multiple outputs based on a reference image in the target domain.

Examples include the Bicycle GAN by , who also applied, as baseline in their experiments, the methods of BID2 BID15 .Other Disentanglement Work InfoGAN BID9 learns a representation in which, due to the statistical properties of the representations, specific classes are encoded as a one-hot encoding of part of the latent vector.

In the work of ; BID16 , the representation is disentangled by reducing the class based information within it.

The separate class based information is different in nature from our multi-dimensional added content.

BID6 , which builds upon BID16 , performs guided image to image translation, but assumes the availability of class based information, which we do not.

We consider a setting with two domains A = (X A , D A ) and B = (X B , D B ).

Here, X A , X B ⊂ R M and D A , D B are distributions over them (resp.).

The algorithm is provided with two independent datasets S A = {a i } m1 i=1 and S B = {b j } m2 j=1 of samples from the two domains that were sampled in the following manner: DISPLAYFORM0 We assume a generative model, in which b is specified by a sample a and a specification c from a third unknown domain C = (X C , D C ) of specifications where D C is a distribution over the metric space X C ⊂ R N .

Formally, there is an invertible function u(b) = (u 1 (b), u 2 (b)) ∈ X A ×X C that takes a sample b ∈ X B and returns the content u 1 (b) of b and the specification u 2 (b) of b. The goal is to learn a target function y : X A × X B → X B such that: DISPLAYFORM1 Informally, the function y takes two samples a and b and returns the analog of a in B that has the specification of b. For example, A is the domain of images of persons, B is the domain of images of persons with sunglasses and C is the domain of images of sunglasses.

The function y takes an image of a person and an image of a person with sunglasses and returns an image of the first person with the specified sunglasses.

For simplicity, we assume that the target function is extended to inputs DISPLAYFORM2 In other words, b 1 and b 2 are mapped to a third b that has the content of b 1 and the specification of b 2 .

In particular, DISPLAYFORM3 Note that within Eq. 2, there is an assumption on the underlying distributions D A and D B .

Using the concrete example, our framework assumes that the distribution of images of persons with sunglasses and the distribution of images of persons without them is the same, except for the sunglasses.

Otherwise, the distribution of the samples generated by y when resampling a would not be the same as D B .

Note that we do not enforce this assumption on the data, and only employ it for our theoretical results, to avoid additional terms.

For two functions f 1 , f 2 : X → R and a distribution D over X, we define the generalization risk between f 1 and f 2 as follows: DISPLAYFORM4 For a loss function : DISPLAYFORM5 losses.

The goal of the algorithm is to return a hypothesis h ∈ H, such that h : DISPLAYFORM6 This quantity measures the expected loss of h in mapping two samples a ∼ D A and b ∼ D B to the analog y(a, b) of a, that has the specification u 2 (b) of b. The main challenge is that the algorithm does not observes paired examples of the form ((a, b), y(a, b)) as a direct supervision for learning the mapping y : X A × X B → X B .

In order to learn the mapping y, we only use an encoder-decoder architecture, in which the encoder receives two input samples and the decoder produces a single output sample that borrows from both input samples.

As we discuss in Sec. 2, the goal of the algorithm is to learn a mapping h = g •f ∈ H such that: g • f (a, b) ≈ y(a, b).

Here, f serves as an encoder and g as a decoder.

The encoder f in our framework is a member of a set of encoders F, each decomposable into two parts and takes the following form: where e 1 : R M → R E1 serves as an encoder of shared content and e 2 : R M → R E2 serves as an encoder of specific content.

Here, E 1 and E 2 are the dimensions of the encodings.

The decoder, g is a member of a set of decoders M. Each member of M is a function g : DISPLAYFORM0 DISPLAYFORM1 In order to learn the functions f and g, we apply the following min-max optimization: DISPLAYFORM2 for some weight parameter λ > 0, of the following training losses (see Fig. 1 ): DISPLAYFORM3 where 0 E2 is the vector of zeros of length E 2 , d is a discriminator network, and l(p, q) = −(q log(p) + (1 − q) log(1 − p))

is the binary cross entropy loss for p ∈ [0, 1] and q ∈ {0, 1}. The discriminator d is a member of a set of discriminators C that locates functions d : DISPLAYFORM4 The discriminator d is trained to minimize L D and Eq. 9 is a domain confusion term BID14

In this section, we provide a theoretical analysis for the success of the proposed method.

For this purpose, we recall a few technical notations (Cover & BID13 : the expectation and probability operators symbols E, P, the Shannon entropy (discrete or continuous) DISPLAYFORM0 , and the total correlation T C(z) : DISPLAYFORM1 is the marginal distribution of the i'th component of z. In particular, T C(z) is zero if and only if the components of z are independent, in which case we say that z is disentangled.

For two distributions D 1 and D 2 , we define the C-discrepancy between them to be disc DISPLAYFORM2 The discrepancy behaves as an adversarial distance measure between two distributions, where d(x) = (c 1 (x), c 2 (x)) is the discriminator that tries to differentiate between D 1 and D 2 , for c 1 , c 2 ∈ C. This quantity is being employed in Chazelle FORMULA1

Thm.

1 upper bounds the generalization risk, based on terms that can be minimized during training, as well as on approximation terms.

It is similar in fashion to the classic domain adaptation bounds proposed by BID4 ; BID31 .

Theorem 1.

Assume that the loss function is symmetric and obeys the triangle inequality.

Then, for any autoencoder h = g • f ∈ H, such that f (x 1 , x 2 ) = (e 1 (x 1 ), e 2 (x 2 )) ∈ F is an encoder and g ∈ M is a decoder, the following holds, DISPLAYFORM0 where DISPLAYFORM1 (The proofs can be found in the appendix.)

Thm.

1 provides an upper bound on the generalization risk R D A,B [h, y] , which is the argument that we would like to minimize.

The upper bound is decomposed of three terms: a reconstruction error, an approximation error and a discrepancy term.

The first term, DISPLAYFORM2 is the reconstruction error for samples b ∼ D B .Since we do not have full access to D B , we minimize its empirical version (see Eq. 8).The second term, While one can minimize the discrepancy term explicitly, by minimizing it with respect to e 1 and e 2 , using a discriminator, we found empirically that this confusion term, which involves both parts of the embedding, is highly unstable.

Instead, we show theoretically and empirically that there is a high likelihood for a disentangled representation (where e 1 (b) and e 2 (b) are independent) to emerge, and the discrepancy term can be replaced with the following discrepancy disc M (e 1 • D A , e 1 • D B ), which measures the closeness between the distributions of e 1 (a) and of e 1 (b) for a ∼ D A and b ∼ D B , as is done in Eq. 9.

Here, M is a set of discriminators that are similar in complexity to the ones in M. This discrepancy is simpler than the one in Eq. 10, since it does not involve a comparison of e 2 between two distributions, nor the interaction between e 1 and e 2 .

DISPLAYFORM3 In Lem.

1, we show that if e 1 (b) and e 2 (b) are independent, then, disc(f DISPLAYFORM4 Lemma 1.

Let M be the set of neural networks of the form: DISPLAYFORM5 k and a non-linear activation function φ 1 : R → R. Let M be the same as M with DISPLAYFORM6 , e 2 (x)) be an encoder and assume that: e 1 (b) |= e 2 (b).

Then, DISPLAYFORM7

The following results are very technical and inherit many of the assumptions used by previous work.

We therefore state the results informally here and leave the complete exposition to the appendix.

First, we extend Proposition 5.2 of BID0 from the case of multiclass classification to the case of autoencoders.

In their work, the aim is to show the conditions in which a mid-level representation f of a multi-class classification neural network h = c • f is both disentangled, i.e., T C(f (b)) is small, and is minimal, i.e., I(f (b); b) is small, where b is an input random variable.

In their Proposition 5.2, they focus on a linear representation, i.e., f = W is a linear transformation, and they introduce a tight upper bound on the sum T C(W b) + I(W b; b).In the general case, their goal is to show that for a neural network h = c•f , both quantities T C(f (b)) and I(f (b); b) are small, when f that is a high level representation of the input.

Unfortunately, they were unable to show that both terms are small simultaneously.

Therefore, in their Cor.

5.3, they extend the bound of their Proposition 5.2 to show that only the mutual information I(f (b); b) is small and assume that the components of each mid-level representation of b in the layers of f are uncorrelated, which is a very restrictive assumption.

In our Lem.

2, we provide an upper bound for T C(f (b)) that is similar in fashion to their bounds.

The main differentiating factor is that we deal with an autoencoder h = g •f .

In this case, the mutual information I ( .

Under some assumptions on the weights of the encoder, there is a monotonically decreasing function q(α) for α > 0 such that: DISPLAYFORM0 Eq. 13 bounds the total correlation of f (b), which measures the amount of dependence between the components of the encoder on samples in B. The bounds has three terms: DISPLAYFORM1 In this formulation, α denotes the amount of regularization in the weights of f .

In addition, q(α) is monotonically increasing as α tends to zero.

The term I(h(b); b) measures the mutual information between the input b and output h(b) of the autoencoder h. Since the mutual information is subtracted in the right hand side, the larger it is, the smaller T C(f (b)) should be.

The last term, O(d 1 /d 2 ) measures the ratio between the dimension of the output of f and the dimension of the previous layer of f .

Thus, this quantity is small whenever there is a significance reduction in the dimension in the application of the last layer of f .Therefore, there is a tradeoff between the amount of regularization in the weights of f and the mutual information I(h(b); b).

If there is small regularization, then, the autoencoder is able to produce better reconstruction h(b) ≈ b, and therefore, a larger value of I(h(b); b).

On the other hand, small regularization leads to a higher value of q(α).The bound relies on the mutual information between the inputs and outputs of the autoencoder to be large.

The following lemma provides an argument why this is the case when the expected reconstruction error of the autoencoder is small.

Lemma 3 (Informal).

Let b ∼ D B be a distribution over a discrete set X B and h = g • f an autoencoder.

Assume that ∀x 1 = x 2 ∈ X B : x 1 − x 2 1 > ∆. Then, DISPLAYFORM2 The above lemma asserts that if the samples in D B are well separated, whenever the autoencoder has a small expected reconstruction error, DISPLAYFORM3 is at least a large portion of H(b).

Therefore, we conclude that if the autoencoder generalizes well, then, it also maximizes the mutual information I(h(b); b).

To conclude the analysis: for a small enough reconstruction error, when training the autoencoder, the mutual information between the autoencoder's input and output is high (Lem.

10), which implies that the individual coordinates of the representation layer are almost independent of each other (Lem.

9).

When using part of the representation to encode the information that exists in domain A (the shared part), the other part would contain coordinates that are weakly dependent of the features encoded in A. In such a case, we can train with a GAN that involves only the shared representation (Lem.

1).

That way, we can upper bound the generalization error expressed in Thm.

1, using relatively simple loss terms, as is done in Sec. 3.

We evaluate our method on three additive facial attributes: eyewear, facial hair, and smile.

Images from the celebA face image dataset by BID36 were used, since these are conveniently annotated as having the attribute or not.

The images without the attribute (no glasses, or no facialhair, or no smile) were used as domain A in each of the experiments.

Note that three different A domains were used.

As the second domain B, we used the images labeled as having glasses, having facial hair, or smiling, according to the experiment.

Our underlying network architecture adapts the architecture used by , which is based on , where we use Instance Normalization BID35 instead of Batch Normalization BID20 , and without dropout.

Let C k denote a ConvolutionInstanceNorm-ReLU layer with k filters, where a kernel size of 4 × 4, with a stride of 2, and a padding of 1 is used.

The activations of the encoders e 1 , e 2 are leaky-ReLUs with a slope of 0.2 and the deocder g employs ReLUs.

e 1 has the following layers C 32 , C 64 , C 128 , C 256 , C 512 , C 512−d ; e 2 has a slightly lower capacity C 32 , C 64 , C 128 , C 128 , C 128 , C d , where d = 25.

The input images have a size of 128 × 128, and the encoding is of size 512 × 2 × 2 (split between the e 1 and e 2 ).

g is symmetric to the encoders and employs transposed convolutions for the upsampling.

In the first set of experiments, we add the relevant content from a random image b ∈ B into an image from a. The results are given in FIG4 , and appendix Fig. 7 and 8.

We compare with two guided image translation baselines: MUNIT and DRIT BID27 .

We used the published code for each method and despite our best effort, these methods fail on the task of content addition.

In almost all cases, the baseline methods apply the style of the guide and not the added content.

It should be noted that the simplicity of our approach directly translates to more efficient training than the two baselines methods.

Our method has one weighting hyperparameter, which is fixed throughout the experiments.

MUNIT and DRIT each has several weighting hyperparameters (since they use more loss terms) and these require attention and, need to change between the experiments, both in our runs and in the authors' own experiments.

In addition, our method has a lower memory footprint and a much shorter duration of each iteration.

The statistics are reported in Tab.

2 and as can be seen, the runtime and memory footprint of our method are much closer to the Fader network by , which cannot perform guided mapping, than to MUNIT and DRIT.Since the performance of the baselines is clearly inferior in the current setting, we did not hold a user study comparing different algorithms.

Instead, we compare the output of our method directly with real images.

Two experiments are conducted: (i) can users tell the difference between an image from domain B and an image from domain A that was translated to domain B, and (ii) can users tell the difference between an image from domain B and the same image, after replacing the attribute's content (glasses, smile, or facial-hair) with that of another image from B. The experiment was performed with n = 30 users, who observed 10 pairs of images each, for each of the tests.

The results are reported in Tab.

3.

As can be seen, users are able to detect the real image over the generated one, in most of the cases.

However, the success ratio varies between the three image translation tasks and between the two types of comparisons.

The most successful experiments, i.e., those where the users were confused the most, were in the facial hair ("beard") category.

In contrast, when replacing a person's glasses with those of a random person, the users were able to tell the real image 74% of the time.

FIG5 and appendix FIG13 and 10 show the type of images shown in the experiment where users were asked to tell an image from domain B from an hybrid image that contains a face of one image from this domain, and the attribute content from another image from it.

As can be seen, most mixand-match combinations seem natural.

However, going over the rows, which should have a fixed attribute (e.g., the same glasses), one observes some variation.

This unwanted variation arises from the need to fit the content to the new face.

The method does have, as can be expected, difficulty dealing with low quality inputs.

Examples are shown in FIG6 , including a misaligned source or guide image.

Also shown is an example in which the added content in the target domain is very subtle.

These challenges result in a lower quality output.

However, the output in each case does indicate some ability to overcome the challenging input.

To evaluate the linearity of the latent representation e 2 (b), we performed interpolation experiments.

The results are presented in FIG7 .

As can be seen, the change is gradual as we interpolate linearly between the e 2 encoding of the two guide images shown on the left and on the right.

In the supplementary appendix, we provide many more translation examples, see FIG9 , 12, and 13.

Table 3 : User study results.

In each cell is the ratio of images, were users selected a real image as more natural than a generated one.

Closer to 50% is better for the method.

While the previous experiments focused on the guided addition of content (mapping from A to B), our method can also be applied in the other direction, from B to A. This way, the specific content in image b ∈ B is removed.

In our method, this is achieved simply by decoding a representation of the form (e 1 (b), 0).The advantage of mapping in this direction is the availability of additional literature methods to compare with, since no guiding is necessary.

In Fig. 6 , we compare the results we obtain for removing a feature with the Fader network method of .

As can be seen, the removal process of our method results in less residuals.

To verify that we obtain a better quality in comparison with that of the published implementation of Fader networks, we have applied both an automatic classifier and a user study.

The classifier is trained on the training set of domain A and B, using the same architecture that is used by to perform model selection.

Tab.

4 presents the mean probability of class B provided by the classifier for the output of both Fader network and our method.

As can be seen, the probability to belong to the class of the image before the transformation is, as desired, low for both methods.

It is somewhat lower on average in our method, despite the fact that our method does not use such a network during training.

The user study was conducted on n = 20 users, each examining 20 random test set triplets from each experiment.

Each triplet showed the original image (with the feature) and the results of the two algorithms, where the feature is removed.

The users preferred our method over fader 92% of the time for glasses removal, 89% of the time for facial hair removal, and 91% of the time for the removal of a smile.

When converting between two domains, there is an inherent ambiguity that arises from the domainspecific information in the target domain.

In guided translation, the reference image in the target domain provides the missing information.

Previous work has focused on the missing information that is highly tied to the texture of the image.

For example, when translating between paintings and photos, DRIT adds considerable content from the reference photo.

However, this is unstructured content, which is not well localized and is highly related to subsets of the image patches that exist in the target domain.

In addition, the content from the reference photo that is out of the domain of paintings is not guaranteed to be fully present in the output.

Our work focuses on transformations in which the domain specific content is well structured, and guarantees to replicate all of the domain specific information from the reference image.

This is done using a small number of networks and a surprisingly simple set of loss terms, which, due to the emergence of a disentangled representation, solves the problem convincingly.

In this section we provide notations and terminology that are were not introduced in Sec. 4 but are necessary for the proofs of the claims in this section.

We say that three random variables (discrete or continuous) X 1 , X 2 , X 3 form a Markov chain, indicated with DISPLAYFORM0 The Data Processing Inequality (DPI) for a Markov chain X 1 → X 2 → X 3 ensures that I(X 1 ; X 3 ) ≤ min (I(X 1 ; X 2 ), I(X 2 ; X 3 )).

In particular, it holds for X 2 = f (X 1 ) and X 3 = g(X 2 ), where f, g are deterministic processes.

We denote by x ∼ log N (µ, σ 2 ) a random variable that is distributed by a log-normal distribution, i.e., log x ∼ N (µ, σ 2 ).

We consider that the mean and variance of a log-normal distribution log N (µ, σ 2 ) are exp(µ + σ 2 /2) and (exp(σ 2 ) − 1) exp(2µ + σ 2 ) respectively.

We denote by W U := (W k,j · U k,j ) k≤m,j≤m the Hadamard product of two matrices W , U ∈ R m×n .

For a given vector x ∈ R m , we denote dim(x) := m and for a matrix W ∈ R m×n , we denote dim(W ) := mn.

In addition, we denote x 2 = x x = (x 2 1 , . . .

, x 2 m ) and DISPLAYFORM1

In this section, we provide useful lemmas that aid in the proofs of our main results.

Lemma 4.

Let x = (x 1 , . . .

, x n ) ∈ R n be a random vector.

Let µ 1 , . . .

, µ n : R → R be continuous invertible functions and we denote µ(x) := (µ 1 (x 1 ), . . .

, µ n (x n )).

Then, T C(x) = T C(µ(x)).Proof.

First, we consider that: DISPLAYFORM0 where,x := (x 1 , . . .

,x n ) is a vector of independent random variables, such thatx i is distributed, according to the marginal distribution of x i .KL-divergence is invariant to applying continuous invertible transformations, i.e., D KL (X Y ) = D KL (µ(X) µ(Y )) for µ that is continuous and invertible.

Therefore, DISPLAYFORM1 Proof.

See (https://math.stackexchange.com/users/44121/jack daurizio).The following lemma is a modification of Claim 2.1 in BID33 .Lemma 6.

Let X and Y be two random variables.

Assume that there is a function (i.e., a deterministic process) DISPLAYFORM2 Proof.

By the data processing inequality, DISPLAYFORM3 Since conditioning does not increase entropy, DISPLAYFORM4 Therefore, we conclude that, I(X; Y ) ≥ qH(X) − H(q).

DISPLAYFORM5 Therefore, we have: DISPLAYFORM6 By Lem.

6, for X :← b, Y :← h v (b) and F :← F , we have: DISPLAYFORM7 The following lemma is an example of three uncorrelated variables X, Y, Z, such that there is a dimensionality reducing linear transformation over them that preserves all of their information.

Lemma 8.

Let X and Y be two independent uniform distributions over [−1, 1] and DISPLAYFORM8 Proof.

Since X and Y are independent, their covariance is zero.

By the definition of X and Y , we have: DISPLAYFORM9 Finally, we consider that T is a homeomorphic transformation T : (x, y, (x + y) 2 ) → (x, y) (between the manifolds {(x, y, z) | x, y ∈ [−1, 1], z = (x + y) 2 } and [−1, 1] 2 ) and mutual information is invariant to applications of homeomorphic transformations, i.e., I(X; Y ) = I(µ(X); ν(Y )) for homeomorphisms µ and ν over the sample spaces of X and Y (resp.).

Therefore, I(X, Y, Z; T (X, Y, Z)) = I(X, Y, Z; X, Y, Z) = H(X, Y, Z).

Theorem 1.

Assume that the loss function is symmetric and obeys the triangle inequality.

Then, for any autoencoder h = g • f ∈ H, such that f (x 1 , x 2 ) = (e 1 (x 1 ), e 2 (x 2 )) ∈ F is an encoder and g ∈ M is a decoder, the following holds, DISPLAYFORM0 where DISPLAYFORM1 .

Since the loss obeys the triangle inequality, DISPLAYFORM2

In this section, we employ the theory of BID0 in order to show the emergence of disentangled representations, when an autoencoder h = g • f generalizes well.

Our analysis shows that by learning an autoencoder such that the encoder f has log-normal regularization, there is a high likelihood of learning disentangled representations.

We note that until now, we treated the autoencoder h as a function of two variables (a, b or b, b) .

x 1 ) , . . .

, φ 1 (x k )) is a non-linear activation function φ 1 : R → R extended for all m ∈ N and (x 1 , . . .

, x k ) ∈ R k .

We assume that φ 1 : R → R is a homeomorphism (i.e., φ 1 is invertible, continuous and φ −1 1 is also continuous).

The encoder f w is the composition of the first t layers and the decoder g u is composed of the last t layers of h v .

Following BID0 , we assume that the posterior distribution p(W k i,j |S B ) is defined as a Gaussian dropout, DISPLAYFORM0 where i ∈ R di+1×di and k i,j ∼ log N (−α/2, α), for k ∈ {1, . . .

, t}. We consider that the mean and variance of log N (−α/2, α) are 1 and exp(α) − 1 respectively.

Here,Ŵ k is a learned mean of the k'th layer of the encoder.

We denote the output of the k'th layer of f w by z k , i.e., z k = φ(W k φ(W k−1 . . .

φ(W 1 x))).

The following lemma is a corollary of Proposition 5.2 in BID0 for our model.

DISPLAYFORM0 Proof.

By Proposition 5.2 in BID0 , we have: DISPLAYFORM1 By Lem.

4, we have, T C(y k ) = T C(z k ).

In addition, dim(y k ) = dim(z k ).

Therefore, DISPLAYFORM2 Finally, by the data processing inequality, for X 1 := b, X 2 := z k−1 , X 3 := y k and X 4 := h v (b), we have, I(h v (b); b) ≤ I(y k ; z k−1 ).

The bound follows from Eq. 30 and the last observation.

Lem.

9 provides an upper bound on the total correlation of the k'th layer of the autoencoder h v (b).

This bound assumes that the marginal distributions of y k and y k |z k−1 are Gaussians.

This is a reasonable assumption if dim(z k−1 ) is large by the central limit theorem.

We also assume that there are no pair-wise linear correlations between the components of z k−1 .

In some sense, this assumption can be viewed as minimality of z k−1 .

Informally, if there is a strong linear correlation between two components of z k−1 , then, we can throw away one of them and keep most of the information.

On the other hand, if the components of z k−1 are uncorrelated, the existence of a dimensionality

@highlight

An image to image translation method which adds to one image the content of another thereby creating a new image.

@highlight

This paper tackles the task of content transfer, with the novalty being on the loss.