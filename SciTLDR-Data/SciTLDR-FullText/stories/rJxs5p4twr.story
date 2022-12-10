In this paper, we ask for the main factors that determine a classifier's decision making and uncover such factors by studying latent codes produced by auto-encoding frameworks.

To deliver an explanation of a classifier's behaviour, we propose a method that provides series of examples highlighting semantic differences between the classifier's decisions.

We generate these examples through interpolations in latent space.

We introduce and formalize the notion of a semantic stochastic path, as a suitable stochastic process defined in feature space via latent code interpolations.

We then introduce the concept of semantic Lagrangians as a way to incorporate the desired classifier's behaviour and find that the solution of the associated variational problem allows for highlighting differences in the classifier decision.

Very importantly, within our framework the classifier is used as a black-box, and only its evaluation is required.

A considerable drawback of the deep classification paradigm is its inability to provide explanations as to why a particular model arrives at a decision.

This black-box nature of deep systems is one of the main reasons why practitioners often hesitate to incorporate deep learning solutions in application areas, where legal or regulatory requirements demand decision-making processes to be transparent.

A state-of-the-art approach to explain misclassification is saliency maps, which can reveal the sensitivity of a classifier to its inputs.

Recent work (Adebayo et al., 2018) , however, indicates that such methods can be misleading since their results are at times independent of the model, and therefore do not provide explanations for its decisions.

The failure to correctly provide explanations by some of these methods lies in their sensibility to feature space changes, i.e. saliency maps do not leverage higher semantic representations of the data.

This motivates us to provide explanations that exploit the semantic content of the data and its relationship with the classifier.

Thus we are concerned with the question: can one find semantic differences which characterize a classifier's decision?

In this work we propose a formalism that differs from saliency maps.

Instead of characterizing particular data points, we aim at generating a set of examples which highlight differences in the decision of a black-box model.

Let us consider the task of image classification and assume a misclassification has taken place.

Imagine, for example, that a female individual was mistakenly classified as male, or a smiling face was classified as not smiling.

Our main idea is to articulate explanations for such misclassifications through sets of semantically-connected examples which link the misclassified image with a correctly classified one.

In other words, starting with the misclassified point, we change its features in a suitable way until we arrive at the correctly classified image.

Tracking the black-box output probability while changing these features can help articulate the reasons why the misclassification happened in the first place.

Now, how does one generate such a set of semantically-connected examples?

Here we propose a solution based on a variational auto-encoder framework.

We use interpolations in latent space to generate a set of examples in feature space connecting the misclassified and the correctly classified points.

We then condition the resulting feature-space paths on the black-box classifier's decisions via a user-defined functional.

Optimizing the latter over the space of paths allows us to find paths which highlight classification differences, e.g. paths along which the classifier's decision changes only once and as fast as possible.

A basic outline of our approach is given in Fig. 1 .

In what follows we introduce and formalize the notion of stochastic semantic paths -stochastic processes on feature (data) space created by decoding latent code interpolations.

We formulate the corresponding path integral formalism which allows for a Lagrangian formulation of the problem, viz.

how to condition stochastic semantic paths on the output Figure 1: Auto-Encoding Examples Setup:

Given a misclassified point x 0 and representatives x −T , x T , we construct suitable interpolations (stochastic processes) by means of an Auto-Encoder.

Sampling points along the interpolations produces a set of examples highlighting the classifier's decision making.

probabilities of black-box models, and introduce an example Lagrangian which tracks the classifier's decision along the paths.

We show the explanatory power of our approach on the MNIST and CelebA datasets.

We are concerned with the problem of explaining a particular decision of a black-box model.

Many recent works discuss the roll and provide definitions of explanations in the machine learning context (Doshi-Velez et al., 2017; Gilpin et al., 2018; Abdul et al., 2018; Mittelstadt et al., 2019 ).

Here we follow Ribeiro et al. (2016) and, in broad terms, to explain we mean to provide textual or visual artifacts that provide qualitative understanding of the relationship between the data points and the model prediction.

Attempts to clarify such a broad notion of explanation require the answers to questions such as (1) what were the main factors in a decision?, as well as (2) would changing a certain factor have changed the decision? (Doshi-Velez et al., 2017) .

To provide an answer to such questions, one must be able to define a clear notion of factors.

One can think of factors as the minimal set of coordinates that allows us to describe the data points.

This definition mirrors the behavior and purpose of the variational auto-encoder (VAE) code -by training an auto-encoder one can find a code which describes a particular data point.

Our role here is to provide a connection between these latent codes and the classifier's decision.

Changes on the code should change the classification decision in a user-defined way.

Defining such a code will allow us to formalize the framework required to provide an answer to question (1) and (2) above.

Following Ribeiro et al. (2016) we require explanations to be model-agnostic, i.e independent of the classifier's inner workings, interpretable, and expressing local fidelity.

Following the discussion above, we use the variational auto-encoder (VAE) formalism (Kingma & Welling, 2013) to introduce a notion of semantics useful to qualitatively explain the decisions of a black-box classifier.

Let us denote the feature (data) space by X and the latent linear space of codes (describing the data) by Z, where usually dim(Z) dim(X ).

We consider a latent variable generative model whose distribution P θ (X) on X is defined implicitly through a two-step generation process: one first samples a code Z from a fixed prior distribution P (Z) on Z and then (stochastically) maps Z to feature space through a (decoder) distribution P θ (X|Z), the latter being parametrized by neural networks with parameters θ.

This class of models are generically train by minimizing specific distances between the empirical data distribution P D (X) and the model distribution P θ (X).

VAE approaches this problem by introducing an encoder distribution Q φ (Z|X), parametrized by neural networks with parameters φ, which approximates the true posterior distribution P θ (Z|X) and minimizing a variational upper bound on the Kullback-Leibler divergence D KL between P θ (X) and P D (X).

This bound reads

where p θ (x|z) denotes the decoder's density and yields the likelihood function of the data given the code 1 .

Once the model is trained one can think of the inferred latent code as containing some highlevel description of the input data.

Below we will use such inferred code to modify in a controlled way the features of a given input data point.

We define a defendant black-box model b(l, x) as a classifier which yields the probability that the data point x ∈ X in feature (data) space belongs to the class l ∈ L, where L is a set of classes.

Assume the model b(l, x) is expected to perform by its users or clients, in a dataset D = {(l i , x i )}, where x i ∈ X and l i ∈ L is the label that x i belongs to.

2 Suppose now that the following litigation case emerges.

The black-box model b has assigned the data point x 0 to the class l 0 .

Accordingly, a plaintiff presents a complaint as the point x 0 should have been classified as l t .

Furthermore, assume we are given two additional representative data points x −T , x T which have been correctly classified by the black-box model to the classes l −T , l T , respectively -as expected by e.g. the plaintiff, the defendant (if agreed), or the institution upon which the complain or litigation case is presented (say, the court).

With this set-up in mind, we propose that an explanation why x 0 was misclassified can be articulated through an example set E = {x −T , . . .

, x 0 , . . .

, x T }, where x t ∼ P θ (X|Z = z t ).

Here P θ (X|Z = z t ) is a given decoder distribution and the index t runs over semantic changes (properly defined below) that highlight classification decisions.

This example set constitutes the context revealing how factor changes impact the classification decision (see Section 2).

One expects that human oriented explanations are semantic in character.

One can understand the expression bigger eyes will change the classification.

As opposed to changes in some specific pixels 3 .

The index t would run over these changes e.g. would make the eyes bigger.

In this section we first formalize the notion of semantic change by introducing the concept of (stochastic) semantic interpolations in feature space X .

This will allow us to generate examples which provide local fidelity, as the examples are smooth modifications of the latent code associated to the plaintiff data point x 0 .

We then define a collection of probability measures over semantic paths in X .

These measures will be used later in Section 6 to constrain the paths to be explanatory with respect to the classifier's decision.

One of the main motivations behind the VAE formalism is the ability of the inferred latent code z to provide semantic high-level information over the data set.

If one is to generate examples which have characteristics common to two different data points, say x 0 and x T from the litigation case, one can perform interpolations between the latent codes of these points, that is z 0 and z T , and then decode the points along the interpolation.

A main observation is that these interpolations in latent space can be used to induce certain interpolating stochastic processes on feature space 4 X .

We refer to these as stochastic semantic processes.

In what follows, we first focus on linear latent interpolations, i.e.

and construct an interpolating stochastic semantic process X t on X by using the decoder distribution P θ (X|Z = z(t)).

In practice, the generation process of such stochastic interpolations consists then of three steps: (i) sample Q φ (Z|X) at the end points x 0 and x T using the reparametrization trick (Kingma & Welling, 2013) , (ii) choose a set of points z t along the line connecting z 0 and z T and (iii) decode the z t by sampling P θ (X|Z = z t ).

A formal description of this procedure is given below, in subsection 5.2, and an impression of the stochastic process thus constructed is presented in Fig. 1b .

We observe that for every sequence of points {t i } n i=0 there is a natural measure on piecewise linear paths starting at x 0 ∈ X and terminating at x T ∈ X .

More precisely, we define the probability of a piecewise linear path x(t) with nodes x 1 , x 2 . . .

, x n ∈ X as dP t0,...,tn (x(t)) :

where q φ , p θ label the densities of Q φ , P θ , respectively, and where z(t) is defined by eq. (2) 5 .

In other words, for every pair of points x 0 and x T in feature space, and its corresponding code

, the decoder P θ (X|Z) induces a measure over the space of paths {x(t)|x(0) = x 0 , x(T ) = x T }.

Formally speaking, the collection of measures dP t0,...,tn given by different choices of points {t i } n i=0 in (3) defines a family of consistent measures (cf.

Definition 2 in the Appendix, Subsection D.1).

This implies that these different measures are assembled into a stochastic process on feature space X over the continuous interval [0, T ]: Proposition 1.

The collection of measures prescribed by (3) induces a corresponding continuous-time stochastic process.

Moreover, under appropriate reconstruction assumptions on the auto-encoder mappings P θ , Q φ , the sample paths are interpolations, that is, start and terminate respectively at x 0 , x T almost surely.

The statement goes along the lines of classical results on existence of product measures.

For the sake of completeness we provide all the necessary technical details in the Appendix, Subsection D. Another important remark is that the stochastic semantic process construction in Proposition 1 is just one way to define such a process -there are other natural options, e.g. in terms of explicit transition kernels or Itô processes.

Having described a procedure to sample stochastic semantic processes in X , we need to discover autoencoding mappings (P θ , Q φ ) that give rise to reasonable and interesting stochastic paths.

Specifically, to generate examples which are able to explain the defendant black-box model b(l, x) in the current litigation case (Section 4), one needs to ensure that semantic paths between the data points x 0 and x T highlight classification differences, i.e. classifications of the model along this path are far apart in the plaintiff pair of labels.

Thus, to design auto-encoding mappings P θ , Q φ accordingly, we propose an optimization problem of the form min

where X t is a stochastic semantic process and S P θ ,Q φ is an appropriately selected functional that extracts certain features of the black-box model b(l, x).

The minimization problem (4) can be seen in the context of Lagrangian mechanics.

For a given stochastic semantic process X t , and given initial and final feature "states" x 0 and x T , we introduce the following function, named the model-b semantic Lagrangian

which gives rise to the semantic model action:

In mechanics, the optimization given by suitable Lagrangians delivers physically meaningful paths, e.g. those specified by the equations of motion (Landau & Lifshitz, 2013) .

In our case, a guiding intuition is that the semantic Lagrangian should reflect how the black-box model takes decisions along the path 6 X t , starting at x 0 and ending at x T .

In this way, the minimization of the semantic action (i.e. finding minimizing paths X t ) should make such classification aspects prominent along the example set.

Our problem, viz. to find encoding mappings P θ , Q φ which yield explainable semantic paths with respect to a black-box model, is then a constrain optimization problem whose total objective function we write as

where L VAE is given by eq. (1), S[x(t)] corresponds to the Lagrangian action and λ is an hyper parameter controlling the action' scale.

The average over the paths (Majumdar, 2007; Feynman & Hibbs, 1965 ) is taken with respect to the stochastic paths and the corresponding measure dP [x(t)] from Proposition 1, that is, the path integral

where x k t labels the tth point along the the kth path, sampled as described in Section 5, n is the number of points on each path, K is the total number of paths, and the estimator on the right hand side corresponds to an explicit average over paths 7 .

Algorithm 1: PATH Auto-Encoder

1 while φ and θ not converged do 2 Draw {x 1 , ..., x n } from the training set

Generate Latent Interpolations

Sample k Paths in Feature Space

Evaluate Semantic Action for each path k 12 and average over k

Update P θ and Q φ by descending:

In practice, both L VAE and the action term are optimized simultaneously.

Note that the VAE loss function L VAE is trained on the entire data set on which the black-box performs.

The action term, in contrast, only sees the x 0 and x T points.

This can be seen explicitly in Algorithm 1, which shows an overview of the auto-encoder pair training algorithm.

Let us finally note that, drawing analogies with the adversarial formalism (Goodfellow et al., 2015) , the defendant black-box model plays the role of a fixed discriminator, not guiding the example generation, but the interpolations among these examples.

There are plenty of options for Lagrangian functionals that provide reasonable (stochastic) example-paths -roughly speaking, we attempt to define an objective value for a certain subjective notion of explanations.

In what follows we illustrate one particular such Lagrangian 8 MINIMUM HESITANT PATH We want to find an example path such that the classifier's decisions along it changes as quickly as possible, as to highly certain regions in X .

In

Figure 2: Probability Paths for the litigation case l 0 = 2, l T = 7.

Y axis corresponds to classification probability and x axis corresponds to interpolation index.

Interpolation images for a specific paths are presented below the x axis.

other words, the path is forced to stay in regions where the black-box produces decisions with maximum/minimum probability.

An intuitive way to enforce this is via the simple Lagrangian

where l 0 , l T are the labels of the litigation case in question.

Roughly speaking, given the appropriate initial conditions, the paths that minimize the action associated to L 1 are paths that attempt to keep L 1 close to 1 over almost the entire interpolation interval.

Additionally we require b(l T , x(t)) to be a monotonous function along the interpolating path x(t).

Furthermore, in accordance with Proposition 1 we require certain level of reconstruction at the end points.

To enforce these conditions we introduce the regularizers r m , r e which are described in detail in subsection D.4 of the Appendix.

The total objective function is therefore

where λ, λ m , λ e are hyper-parameters and S 1 is the action associated to the minimum hesitant Lagrangian L 1 in eq. (9).

We evaluate our method in two real-world data sets: MNIST, consisting of 70k Handwriting digits, (LeCun, 1998) and CelebA (Liu et al., 2015) , with roughly 203k images.

We use a vanilla version of the VAE (Kingma & Welling, 2013) with Euclidean latent spaces Z = R dz and an isotropic Gaussian as a prior distribution P (Z) = N (Z|0, I dz ).

We used Gaussian encoders, i.e. Q φ (Z|X) = N (Z|µ φ (X), Σ φ (X)), where µ φ , σ φ are approximated with neural networks of parameters φ, and Bernoulli decoders P θ (X|Z).

We compare the standard VAE, VAE-EDGE (VAE augmented with the edge loss r e ) and PATH-VAE (our full model, eq. (10)).

The black-box classifier b(l, x) is defined as a deep network with convolutional layers and a final soft-max output layer for the labels.

Details of the specifics of the architectures as well as training procedure are left to the Appendix.

For MNIST we studied a litigation case wherein l −T , l T = 2, 7 and l 0 = 2, whereas its true label (i.e. that of x 0 ) is l t = 7 (see Section 4).

The results are presented in Fig. 2 .

VAE delivers interpolations which provide uninformative examples, i.e. the changes in the output probability b(l 0 , x) cannot be associated with changes in feature space.

In stark contrast, PATH-VAE causes the output probability to change abruptly.

This fact, together with the corresponding generated examples, allows us to propose explanations of the form: what makes the black-box model classify an image in the path as two or seven, is the shifting up of the lower stroke in the digit two as to coincide with the middle bar of the digit seven.

Similarly, the upper bar of the digit seven (especially the upper left part) has a significant decision weight.

In order to provide a more quantitative analysis we demonstrate the capability of our methodology to control the path action while retaining the reconstruction capacity.

Hereby, we use not only the VAE as the underlying generative model, but also Wasserstein Auto-Encoder (WAE) (Bousquet et al., 2017) and Adversarial Auto-Encoder (AAE) (Goodfellow et al., 2015) , i.e. we simply change L VAE in eq. (7) with the loss of WAE or AAE.

The theoretical details and corresponding architectures are presented in the Appendix.

We present, in Fig. 3 , the action values defined over random litigation end pairs (x −T , x T ).

The PATH version of the model indeed yields lower action values.

Furthermore, these models tend to reduce the variance within the different paths.

This is expected since there is one path that minimizes the action, hence, the distribution will try to arrive at this specific path for all samples.

In order to compare with other explanation models, we define a saliency map with the interpolations obtained in our methodology.

We defined the interpolation saliency as the sum over the differences between interpolation images weighted with the probability change of the black-box classifier through the interpolation path.

We see in Fig. 4 the comparisons among different methods.

While the standard methods only show local contributions to a classification probability, our saliency maps show the minimum changes that one is to apply to the dubious image in order to change the decision to the desired label.

Our approach reveals that the curvature of the lower bar is decisive to be classified as a two, while the style of the upper bar is important to be classified as a seven.

Further, we provide a sanity check analysis (Adebayo et al., 2018) by studying the rank correlation between original saliency map and the one obtained for a randomized layers of the black-box classifier, shown in Fig. 5 .

As desired, our proposed saliency map decorrelates with the randomized version.

For the CelebA dataset we use a black-box classifier based on the ResNet18 architecture (He et al., 2016) .

We investigate two specific misclassifications.

In the first case, a smile was not detected (Fig. 6 a) .

Here we only interpolate between the misclassified image (left) and a correctly classified one (right), of the same person.

Interpolations obtained by VAE are not informative: specific changes in feature space corresponding to changes in the probability cannot be detected since the latter changes rather slowly over the example path.

This observation also holds true for the VAE-EDGE model, except that the examples are sharper.

Finally, our PATH-VAE model yields a sharp change in the probability along with a change of the visible teeth (compare the third and fifth picture in the example path), revealing that this feature (i.e. teeth visibility) could constitute, from a human standpoint, a decisive factor in the probability of detecting a smile for the given black-box model.

It is important to note that these observations represent one of many possible path changes which could change the classifier decision.

This is constrained by the current realization and representative end points.

The important result is that our methods are able to shape the behavior of the classifier along the path.

Further experimental examples are provided in Section C of the Appendix.

The bulk of the explanation literature for deep/black-box models relies on input dependent methodologies.

Gradient Based approaches (Simonyan et al., 2013; Erhan et al., 2009) Figure 6 : Probability Paths for the case of detecting a smile in images of celebrities.

Y axis corresponds to classification probability and x axis corresponds to interpolation index.

Interpolation images for a specific paths are presented below the x axis.

The images are vertically aligned with a corresponding tick in the x-axis determining the interpolation index of the image score for a given input example and class label by computing the gradient of the classifier with respect to each input dimension.

Generalizations of this approach address gradient saturation by incorporating gradients' values in the saliency map (Shrikumar et al., 2017) or integrating scaled versions of the input (Sundararajan et al., 2017) .

Ad hoc modifications of the gradient explanation via selection of the required value (Springenberg et al., 2015) , (Zeiler & Fergus, 2014) , as well as direct studies of final layers of the convolutions units of the classifiers (Selvaraju et al., 2016) , are also provided.

In contrast to gradient based approaches, other categories of explanatory models rely on reference based approaches which modify certain inputs with uninformative reference values (Shrikumar et al., 2017) .

Bayesian approaches treat inputs as hidden variables and marginalize over the distribution to obtain the saliency of the input (Zintgraf et al., 2017) .

More recent generalizations exploit a variational Bernoulli distribution over the pixels values (Chang et al., 2018) .

Other successful methodologies include substitution of black-box model with locally interpretable linear classifiers.

This is further extended to select examples from the data points in such a way that the latter reflect the most informative components in the linear explanations, (Ribeiro et al., 2016) .

Studies of auto-encoder interpolations seek to guarantee reconstruction quality.

In (Arvanitidis et al., 2018) the authors characterize latent space distortions compared to the input space through a stochastic Riemannian metric.

Other solutions include adversarial cost on the interpolations such as to improve interpolation quality compared to the reconstructions, (Berthelot et al., 2018) .

Examples which are able to deceive the classifier's decisions have been widely studied in the framework of adversarial examples (Goodfellow et al., 2015) .

These methodologies, however, do not provide interpretable explanations or highlight any semantic differences that lead to the classifier's decisions.

Finally, the Auto-Encoder framework can also naturally be seen as a tool for dimensionality reduction.

Geometrically speaking, assuming that the data set approximately lies along a manifold embedded in feature space X , one can interpret the encoder, decoder as the coordinate map (chart) and its inverse.

From this point of view, our approach above translates to finding coordinate charts with additional constraints on mapping the segments from z 0 to z T to appropriate (stochastic) curves between x 0 and x T .

In the present work we provide a novel framework to explain black-box classifiers through examples obtained from deep generative models.

To summarize, our formalism extends the auto-encoder framework by focusing on the interpolation paths in feature space.

We train the auto-encoder, not only by guaranteeing reconstruction quality, but by imposing conditions on its interpolations.

These conditions are such that information about the classification decisions of the model B is encoded in the example paths.

Beyond the specific problem of generating explanatory examples, our work formalizes the notion of a stochastic process induced in feature space by latent code interpolations, as well as quantitative characterization of the interpolation through the semantic Lagrangian's and actions.

Our methodology is not constrained to a specific Auto-Encoder framework provided that mild regularity conditions are guaranteed for the auto-encoder.

There was no preprocessing on the 28x28 MNIST images.

The models were trained with up to 100 epochs with mini-batches of size 32 -we remark that in most cases, however, acceptable convergence occurs much faster, e.g. requiring up to 15 epochs of training.

Our choice of optimizer is Adam with learning rate α = 10 −3 .

The weight of the KL term of the VAE is λ kl = 1, the path loss weight is λ p = 10 3 and the edge loss weight is λ e = 10 −1 .

We estimate the path and edge loss during training by sampling 5 paths, each of those has 20 steps.

Encoder Architecture

Both the encoder and decoder used fully convolutional architectures with 3x3 convolutional filters with stride 2.

Conv k denotes the convolution with k filters, FSConv k the fractional strides convolution with k filters (the first two of them doubling the resolution, the third one keeping it constant), BN denotes batch normalization, and as above ReLU the rectified linear units, FC k the fully connected layer to R k .

The pre-processing of the CelebA images was done by first taking a 140x140 center crop and then resizing the image to 64x64.

The models are trained with up to 100 epochs and with mini-batches of size 128.

Our choice of optimizer is Adam with learning rate α = 10 −3 .

The weight of the KL term of the VAE is λ kl = 0.5, the path loss weight is λ p = 0.5 and the edge loss weight is λ e = 10 − 3.

We estimate the path and edge loss during training by sampling 10 paths, each of those has 10 steps.

Encoder Architecture

Decoder Architecture

Both the encoder and decoder used fully convolutional architectures with 3x3 convolutional filters with stride 2.

Conv k denotes the convolution with k filters, FSConv k the fractional strides convolution with k filters (the first two of them doubling the resolution, the third one keeping it constant), BN denotes batch normalization, and as above ReLU the rectified linear units, FC k the fully connected layer to R k .

C FURTHER RESULTS

Interpolation between 2 and 7.

It is seen that the Path-VAE interpolation optimizes both probabilities (P(2) and P (7)) according to the chosen Lagrangian -in this case the minimum hesitant L 1 .

Briefly put, the construction we utilize makes use of the well-known notion of consistent measures, which are finite-dimensional projections that enjoy certain restriction compatibility; afterwards, we show existence by employing the central extension result of Kolmogorov-Daniell.

We start with a couple of notational remarks.

Definition 1.

Let S, F be two arbitrary sets.

We denote

that is, the set of all maps F → S. Definition 2.

Let (S, B) be a measurable space and let G ⊆ F ⊆ [0, T ] for some positive number T .

We define the restriction projections π F,G by

Moreover, for each F ⊆ [0, T ] the restriction projections induce the σ-algebra B F which is the smallest σ-algebra on S F so that all projections

are measurable.

In particular, the projections π F,G are measurable with respect to B F , B G .

} is called consistent if it is push-forward compatible with respect to the restriction projection mappings, i.e.

be an arbitrary finite set.

The mapping

defines a consistent collection of finite measures.

Proof.

Let us fix

Without loss of generality, it suffices to check consistency for the pair (F 1 , F 2 ).

We have

where we have used L 1 -finiteness and integrated out the s variable via Fubini's theorem.

Note also, that by the definitions above χ π

for any fixed s ∈ X .

We briefly recall the following classical result due to Kolmogorov and Daniell: Theorem 1 (Theorem 2.11, Bär & Pfäffle (2012) ).

Let (S, B(S)) be a measurable space with S being compact and metrizable and let I be an index set.

Assume that for each J ∈ Fin(I) there exists a measure µ J on S J , B J , such that the following compatibility conditions hold:

Here π J1 : S J2 → S J1 denotes the canonical projection (obtained by restriction).

Then, there exists a unique measure µ on (S I , B I ) such that for all J ∈ Fin(I) one has

We recall that a well-known way to construct the classical Wiener measure and Brownian motion is precisely via the aid of Theorem 1 (Taylor (2011) ).

We are now in a position to construct the following stochastic process.

Proposition 3.

There exists a continuous-time stochastic process

Moreover, for small positive numbers , δ we have X 0 ∈ B δ (x 0 ) with probability at least (1 − ), provided the reconstruction error of encoding/decoding process is sufficiently small.

In particular, if x 0 stays fixed after the application of encoder followed by decoder, then X 0 = x 0 almost surely.

A similar statement holds also for the terminal point X t and x T respectively.

Proof.

By applying Theorem 1 to the collection of consistent finite measures prescribed by Proposition 2 we obtain a measure µ on the measurable space (S [0,T ] , B [0,T ] ).

Considering the probability space (S [0,T ] , B [0,T ] , µ) we define stochastic process

It follows from the construction and the Theorem of Kolmogorov-Daniell that P ((X t1 , X t2 , . . .

, X tn ) ∈ A) is expressed in the required way.

This shows the first claim of the statement.

Now, considering a small ball B δ (x 0 ) we have

Here, the function R(x * , U ) measures the probability that the input x * is decoded in the set U .

Thus, if the reconstruction error gets smaller, R converges to 1.

This implies the second statement.

Finally, if we assume that the auto-encoder fixes x 0 in the sense above, we similarly get

D.2 CONCERNING THE REGULARITY OF SAMPLE PATHS An important remark related to the the variational problem (4) is the following: one could develop plenty of meaningful functionals S P θ ,Q φ that involve taking velocities or higher derivatives -thus one is supposed to work over spaces of curves with certain regularity assumptions.

However, as stated above we are working over stochastic paths X t whose regularity is, in general, difficult to guarantee.

A straightforward way to alleviate this issue is to consider a "smooth" version of the curve X t -e.g.

by sampling X t through a decoder with controllable or negligible variance or by means of an appropriate smoothing.

Furthermore, one could also approach such stochastic variational analysis via Malliavin calculus -however, we do not pursue this direction in the present work.

We now briefly discuss a few remarks about the regularity of the stochastic semantic process from Proposition 1.

First, we state a well-known result of Kolmogorov and Chentsov: Theorem 2 (Theorem 2.17, Bär & Pfäffle (2012) ).

Let (M, ρ) be a metric measure space and let X t , t ∈ [0, T ] be a stochastic process.

Suppose that there exists positive numbers a, b, C, with the property

, ∀s, t, |s − t| < (38) Then, there exists a version Y t , t ∈ [0, T ] of the stochastic process X t whose paths are α-Hölder continuous for any α ∈ (0, b/a).

Thus, roughly speaking, an estimate on E [ρ(X s , X t ) a ] can be regarded as a measure of the extent to which Theorem 2 fails.

To give an intuitive perspective, let us consider the stochastic process given by Proposition 1 and, considering only the points X s , X s+δ for a small positive number δ, let us write the expectation in (38) as:

where we have used the standard Euclidean distance.

To estimate the integral further, let us for simplicity assume that the encoder is deterministic and the decoder is defined via a Gaussian Ansatz of the type µ(z) + σ(z) ⊗ for a normal Gaussian variable .

Thus the last integral can be written as:

where we denote the covariance matrix at time s by Σ s .

Now, if Σ s+δ becomes sufficiently small as δ converges to 0, then the exponential factor will dominate and thus (38) holds.

In other words, Hölder regularity of the process is verified provided that p θ (x|z) becomes localized in x and converges to a Dirac measure (similarly to the case of the heat kernel propagator and Brownian motion).

From this point of view, the variance of the decoder can be considered as an indicator of how far the stochastic process is from being Hölder continuous.

Below we discuss two other stochastic process constructions, one of which is built upon Itô diffusion processes and enjoys further path-regularity properties.

We briefly recall that, among other aspects, Lagrangian theory suggests a framework for optimization of functionals (Lagrangians) defined over appropriate function spaces.

Critical points of Lagrangians are identified by means of the corresponding Euler-Lagrange equations (Landau & Lifshitz (2013) ).

To obtain the Euler-Lagrange equations for the Lagrangians in (9, 45, 50) we compute in a straightforward manner the first variation

where φ : [0, T ] → T X is a compactly supported deformation 9 .

This produces the following conditions:

Assuming the Hessian is not identically vanishing along the curve, the critical points of the variational problem are given by the condition (∇B) (l T |x(t)) = αẋ(t).

In addition to following the geometry of the black box B, one could also impose a natural condition that the stochastic paths minimize distances on the manifold in feature space that the auto-encoder pair induces.

We recall from basic differential geometry that the image of the decoder as a subset of the feature space is a submanifold with a Riemannian metric g induced by the ambient Euclidean metric in the standard way (for background we refer to do Carmo (1976) ).

In the simple case of a deterministic auto-encoder, one can think of g as the matrix J T J where J denotes the Jacobian of the decoder -thus g gives rise to scalar product g(X, Y ) := XJ T JY .

In the stochastic case, one can use suitable approximations to obtain g in a similar manner -e.g.

in Arvanitidis et al. (2018) the authors decompose the decoder into a deterministic and a stochastic part, whose Jacobians J 1 , J 2 are summed as J T 1 J 1 + J T 2 J 2 to obtain the matrix g. Now, having Riemannian structure (i.e. the notion of a distance) on the data submanifold, geodesic curves naturally arise as minimizers of a suitable distance functional, namely:

where the norm · g is computed with respect to the Riemannian metric g, that is g(·, ·).

We note that the utilization of geodesics for suitable latent space interpolations was thoroughly discussed in Arvanitidis et al. (2018) .

As mentioned already, we would like that classifier's probabilities change in a monotonous fashion along the paths -these paths are preferable in the sense that they provide examples following a particular trend along the disputed labels.

We enforce such monotonic behaviour along the paths with the term r m := 1 K(n − 1)

with n the number of points along the path and K the number of paths.

Further, and in accordance with Proposition 1, one can also require that the auto-encoder reconstructs the endpoints with sufficiently large accuracy.

We enforce this requirement with the edge term r e := i (|b(l i , x i )) − b(l i ,x i )| + c(x i ,x i )) , i = 0, T, −T,where c measures the reconstruction error 10 andx i ∼ P θ (X|Z = z i ), with z i ∼ Q φ (Z|X = x i ) and x i the data points at i = 0, T, −T .

In contrast to VAE, within the WAE framework Tolstikhin et al. (2018) one only needs to be able to sample from Q φ (Z|X) and P θ (X|Z) -i.e.

their density is not needed.

WAE is trained by minimizing a (penalized) optimal transport divergence Bousquet et al. (2017) -the Wasserstein distance, between the input data distribution P D (X) and the implicit latent variable model P θ (X).

As in VAE, the latter is defined by first sampling Z from P (Z) and then mapping Z to X through the decoder P θ (X|Z).

The loss function of WAE is given by

where c is a distance function and D Z is an arbitrary divergence between the prior P (Z) and the agregate posterior Q φ (Z) = E P D (X) [Q φ (Z|X)], weighted by a positive hyperparameter λ.

Minimizing Equation 56 corresponds to minimizing the Wasserstein distance if the decoder is deterministic (i.e. P θ (X|Z = z) = δ g θ (z) ∀z ∈ Z, with the map g θ : Z → X ) and the distance term is optimized.

If the decoder is stochastic Equation 56 yields an upper bound on the Wasserstein

@highlight

We generate examples to explain a classifier desicion via interpolations in latent space. The variational auto encoder cost is extended with a functional of the classifier over the generated example path in data space.