It is clear that users should own and control their data and privacy.

Utility providers are also becoming more interested in guaranteeing data privacy.

Therefore, users and providers can and should collaborate in privacy protecting challenges, and this paper addresses this new paradigm.

We propose a framework where the user controls what characteristics of the data they want to share (utility) and what they want to keep private (secret), without necessarily asking the utility provider to change its existing machine learning algorithms.

We first analyze the space of privacy-preserving representations and derive natural information-theoretic bounds on the utility-privacy trade-off when disclosing a sanitized version of the data X. We present explicit learning architectures to learn privacy-preserving representations that approach this bound in a data-driven fashion.

We describe important use-case scenarios where the utility providers are willing to collaborate with the sanitization process.

We study space-preserving transformations where the utility provider can use the same algorithm on original and sanitized data, a critical and novel attribute to help service providers accommodate varying privacy requirements with a single set of utility algorithms.

We illustrate this framework through  the implementation of three use cases; subject-within-subject, where we tackle the problem of having a face identity detector that works only on a consenting subset of users, an important application, for example, for mobile devices activated by face recognition; gender-and-subject, where we preserve facial verification while hiding the gender attribute for users who choose to do so; and emotion-and-gender, where we hide independent variables, as is the case of hiding gender while preserving emotion detection.

will make such devices not to understand the data until the visual or sound trigger is detected, with the capability to do this at the sensor level and without modifying the existing recognition system.

This new paradigm of collaborative privacy environment is critical since it has also been shown that algorithmic or data augmentation and unpredictable correlations can break privacy BID7 ; BID14 ; Oh et al. (2016) ; BID18 .

The impossibility of universal privacy protection has been studied extensively in the domain of differential privacy BID4 , where a number of authors have shown that assumptions about the data or the adversary must be made in order to be able to provide utility BID5 ; BID6 ; BID10 .

We can, however, minimize the amount of privacy we are willing to sacrifice for a given level of utility.

Other recent data-driven privacy approaches like Wu et al. (2018) have also explored this notion, but do not integrate the additional collaborative constraints.

Therefore, it is important to design collaborative systems where each user shares a sanitized version of their data with the service provider in such a way that user-defined non-sensitive tasks can be performed but user-defined sensitive ones cannot, without the service provider requiring to change any data processing pipeline otherwise.

Contributions-We consider a scenario where a user wants to share a sanitized representation of data X in a way that a latent variable U can be inferred, but a sensitive latent variable S remains hidden.

We formalize this notion using privacy and transparency definitions.

We derive an informationtheoretic bound on privacy-preserving representations.

The metrics induced by this bound are used to learn such a representation directly from data, without prior knowledge of the joint distribution of the observed data X and the latent variables U and S. This process can accommodate for several user-specific privacy requirements, and can be modified to incorporate constraints about the service provider's existing utility inference algorithms enabling several privacy constraints to be satisfied in parallel for a given utility task.

We apply this framework to challenging use cases such as hiding gender information from a facial image (a relatively easy task) while preserving subject verification (a much harder task), or designing a sanitization function that preserves subject identification on a consenting subset of users, while disallowing it on the general population.

Blocking a simpler task while preserving a harder one and blocking a device from constantly listening out-of-sample data are new applications in this work, here addressed with theoretical foundations and respecting the provider's existing algorithms, which can simultaneously handle sanitized and non-sanitized data.

The problem statement is detailed in Section 2, and the information-theoretic bounds are derived in Section 3.

Section 4 defines a trainable adversarial game that directly attempts to achieve this bound; the section also discusses how service-provider specific requirements can be incorporated.

Examples of this framework are shown in Section 5.

The paper is concluded in Section 6.

Complementary information and proofs are presented in the Supplementary Material .

We describe a scenario in which we have access to possibly high-dimensional data X ∈ X , this data depends on two special latent variables U and S. U is called the utility latent variable, and is a variable we want to communicate, while S is called the secret, and is a variable we want to protect.

We consider two agents, a service provider that wants to estimate U from X, and an actor that wants to infer S from X.We define a third agent, the privatizer, that wants to learn a space-preserving stochastic mapping Q : X → Q ⊃ X in such a way that Q(X) provides information about the latent variable U , but provides relatively little information of S. In other words, we want to find a data representation that is private with respect to S and transparent with respect to U .

1 We first recall the definition of privacy presented in BID9 DISPLAYFORM0 Definition 2.1.

Privacy:

Let δ s be a measure of distance between probability distributions, b s ∈ R + a positive real number, and P (S) the marginal distribution of the sensitive attribute S. The stochastic mapping Q(X) is (δ s , b s )-private with respect to S if δ s (P (S), P (S|Q(X))) < b s .We can define transparency in the same fashion: Definition 2.2.

Transparency: Let δ u be a measure of distance between probability distributions, b u ∈ R + a positive real number, and P (U |X) the posterior conditional distribution of the utility variable U after observing X. The stochastic mapping Q(X) is (δ u , b u )-transparent with respect to U if δ u (P (U |X), P (U |Q(X))) < b u .Both definitions depend on the learned mapping Q; in the following section, we derive an information-theoretic bound between privacy and transparency, and show that this bound infers a particular choice of metrics δ u , δ s .

We then show that this inferred metric can be directly implemented as a loss function to learn privatization transformations from data using standard machine learning tools.

A similar analysis of these bounds for the special case where we directly observe the utility variable U (X = U ) was analyzed in BID1 in the context of the Privacy Funnel.

Here, we extend this to the more general case where U is observed indirectly.

More importantly, these bounds are used to design a data-driven implementation for learning privacy-preserving mappings.

Consider the utility and secret variables U and S defined over discrete alphabets U, S,and the observed data variable X, defined over X , with joint distribution P X,U,S .

FIG0 illustrates this set-up, and shows the fundamental relationship of their entropies H(·) and mutual information.

We analyze the properties of any mapping Q : X → Q, and measure the resulting mutual information between the transformed variable Q(X) and our quantities of interest.

Our goal is to find Q such that the information leakage from our sanitized data I(S; Q(X)) is minimized, while maximizing the shared information of the utility variable I(U ; Q(X)).

We will later relate these quantities and the bounds here developed with the privacy/utility definitions presented in the previous section.

Maximizing I(U ; Q(X)) is equivalent to minimizing I(U ; X | Q(X)), since I(U ; X|Q(X)) = I(U ; X) − I(U ; Q(X)).

The quantity I(U ; X | Q(X)) is the information X contains about U that is censured by the sanitization mapping Q. FIG0 illustrates I(S; Q(X)) and I(U ; X|Q(X)).

One can see that there exists a trade-off area, I(U, S) − I(U, S|X), that is always included in the union of I(S; Q(X)) and I(U ; X|Q(X)).

The lower we make I(S; Q(X)), the higher we make the censored information I(U ; X|Q(X)), and vice versa.

This induces a lower bound over the performance of the best possible mappings Q(X) that is formalized in the following lemma.

Lemma 3.1.

Let X, U, S be three discrete random variables with joint probability distribution P X,U,S .

For any stochastic mapping Q : X → Q we have DISPLAYFORM0 Proof of this lemma is shown in Supplementary Material.

To show that this bound is reachable in some instances, consider the following example.

Let U and S be independent discrete random variables, and X = (U, S).

The sanitization mapping Q(X) = U satisfies this bound with equality.

We can also prove, trivially, an upper bound for these quantities.

Lemma 3.2.

Let X, U, S be three discrete random variables with joint probability distribution P X,U,S .

For any stochastic mapping Q : X → Q we have: DISPLAYFORM1 That simply states that the information leakage about the secret and the censured information on the utility variable cannot exceed the total information present in the original observed variable X.

We relate the terms I(S; Q(X)) and I(U ; X | Q(X)) in Eq.1 back to our definitions of privacy and transparency.

DISPLAYFORM0 Here we used the fact that U is conditionally independent of Q given X. We then observe that Eq. Similarly, we can analyze I(S; Q) to get, DISPLAYFORM1 We can see from Eq.4 that the natural induced metric for measuring privacy δ s in Def,2.1 is the reverse Kullback-Leibler divergence RD KL .We can thus rewrite our fundamental tradeoff equation as DISPLAYFORM2 We show next how this bound can be used to define a trainable loss metric, allowing the privatizer to select different points in the transparency-privacy trade-off space.

Assume that for any given stochastic transformation mapping Q ∼ Q(X), we have access to the posterior conditional probability distributions P (S | Q), P (U | Q) , and P (U | X).

Assume we also have access to the prior distribution of P (S).

Inspired by the bounds from the previous section, the proposed privatizer loss is DISPLAYFORM0 where α ∈ [0, 1] is a tradeoff constant.

A low α value implies a high degree of transparency, while a high value of α implies a high degree of privacy.

Using Eq.5 we have a lower bound on how private or transparent the privatizer can be for any given α value, as detailed next.

Theorem 3.3.

For any α ∈ [0, 1], and stochastic mapping Q : X → Q the solution to Eq.6 guarantees the following bounds, DISPLAYFORM1 The proof is shown in Supplementary Material.

To recap, we proposed a privatizer loss Eq.6 with a controllable trade-off parameter α, and showed bounds on how transparent and private our data can be for any given value of α.

Next we show how to optimize this utility-privacy formulation.

Even if the joint distribution of P (U, S, X) is not known, the privatizer can attempt to directly implement Eq.6 in a data-driven architecture to find the optimal Q. Assume the privatizer has access to a dataset {(x, s, u)}, where s and u are the ground truth secret and utility values of observation x. Under these conditions, the privatizer searches for a parametric mapping q = Q θ (x, z), where z is an independent random variable, and attempts to predict the best possible attack by learning P η (s | q), an estimator of P (s | q).

The privatizer also needs P ψ (u|q) and P φ (u|x), estimators of P (u | q) and P (u | x) respectively, to measure how much information about the utility variable is censored with the proposed mapping.

Under this setup Q θ (x, z) is obtained by optimizing the following adversarial game: DISPLAYFORM0 Here the first three terms are crossentropy loss terms to ensure our estimators P η (s|q), P ψ (u|q), and P φ (u|x) are a good approximation to the true posterior distributions.

The final loss term attempts to find the best possible sampling function Q θ (x, z) such that (1 − α)I 2 (U ; X | Q) + αI 2 (S; Q) is minimized.

Details on the algorithmic implementation are given in Section 7.3.1.

Performance on simulated datasets is shown in Section 7.2.

The proposed framework naturally provides a means to achieve collaboration from the utility provider.

In this scenario, the utility provider wishes to respect the user's desired privacy, but is unwilling to change their estimation algorithm Pφ(u | x), and expects the privatizer to find a mapping that minimally affects its current performance.2 This is a more challenging scenario, with worse tradeoff characteristics, in which Q θ (x, z) is obtained by optimizinĝ DISPLAYFORM0 2 Recall that the utility provider wants to use the same algorithm for sanitized and non-sanitized data, a unique aspect of the proposed framework and critical to accept its collaboration.

A final scenario addressed by the proposed framework arises when the utility provider is the sole agent to access the sanitized data, and it has estimation algorithms for both the utility and the privacy variable Pφ(u | x), Pη(s | x), that it is unwilling to modify.

The service provider wishes to reassure the users that they are unable to infer the secret attribute from the sanitized data, if and when the user decides so.

Under these conditions, we optimize for DISPLAYFORM0 (10)

The following examples are based on the framework presented in FIG8 .

Here we have the three key agents mentioned before: (1) the utility algorithm that is used by the provider to estimate the information of interest.

This algorithm can take the raw data (X) or the mapped data (Q(X)) and be able to infer the utility; (2) the secret algorithm that is able to operate on the mapped data to infer the secret; (3) the privatizer that learns a space preserving mapping Q that allows the provider to learn the utility but prevents the secret algorithm to infer the secret.

The utility algorithm is trained to perform well on raw data, the secret algorithm is adversarially trained to infer the secret variable after sanitization.

In the next examples we show how the proposed framework performs under different scenarios, the privatizer architecture is kept unchanged across all experiments to show that the same architecture can achieve very different objectives using the proposed framework, the detailed architectures are shown in Section 7.3.2.

Extra experiments under known conditions are shown in 7.2.Figure 2: Three components of the collaborative privacy framework.

Raw data can be directly fed into the secret and utility inferring algorithm.

Since the privatization mapping is space preserving, the privatized data can also be directly fed to both tasks without any need for further adaptations.

We begin by analyzing the subject-within-subject problem.

Imagine a subset of users wish to unlock their phone using facial identification, while others opt out of the feature; we wish the face identification service to work only on the consenting subset of users.

We additionally assume that the utility provider wishes to comply with the user's wishes, so we can apply the framework described in Section 4.2.

Note that in this problem, the utility and secrecy variables are mutually exclusive.

We solve this problem by training a space-preserving stochastic mapping Q on facial image data X, where the utility and secret variable U and S are categorical variables over consenting and nonconsenting users respectively.

We test this over the FaceScrub dataset BID8 ), using VGGFace2 Cao et al. (2017 as the utility and secrecy inferring algorithm.

The stochastic mapping was implemented using a stochastic adaptation of the UNET Ronneberger et al. (2015) , architecture details are provided in Section 7.3.2.

Table 1 shows the top-5 categorical accuracy of the utility network over the sanitized data at various α points in the privacy-utility trade-off.

Figure 3 show some representantive images on how images are sanitized.

It also shows that the sanitization function is able to preserve information about the utility variable while effectively censoring the secret variable, even for unobserved images.

A phone equipped with this filter at the sensor level would be effectively incapable of collecting information on nonconsenting users.

Figure 3 : Left and center figures show images of consenting and nonconsenting (private) users respectively, along with their sanitized counterparts.

The identity of consenting users is still easily verified, while the identity of nonconsenting users is effectively censored.

Table on the right shows Top-5 accuracy performance of the subject detector after sanitization across several sanitation levels α.

Performance is shown across 3 subsets, consenting users (CU) are users that decided to be detected by the utility algorithm, observed private users (OPU) are those that explicitly decided to protect their privacy, while unobserved private users (UPU) are users that decided to protect their privacy but where not available during training.

Consenting users are still recognized by the system, while nonconsenting users are not.

For example, for α = 0.4, we significantly block OPU and UPU while preserving CU

Here we continue to work on facial image data X, where utility variable U is gender recognition, and the secret variable S is emotion (smiling/non-smiling).

In this scenario, variables U and S are independent.

We implement this over the CelebA dataset BID13 , using Xception networks Chollet (2017) as our utility and privacy estimators.

Table.

2 shows the distribution of the utility and secrecy estimators over the sanitized data.

FIG3 shows example sanitized images.

It is visually possible to identify the gender of the subject but not their emotion.

Most importantly, the existing gender detection algorithm still performs correctly over the sanitized images.

Table 2 : Gender and emotion detection on users on raw and sanitized data.

In this setup, we want to find a mapping Q that hides the gender attribute but allows subject verification.

The mapping Q should prevent a standard gender detection algorithm from performing its task, while allowing a standard subject detector algorithm to still perform subject verification.

This is the only experiment in this section where the secret inference algorithm is fixed.

The mapping that incorporates a pretrained FaderNet was chosen as the baseline for the stochastic mapping function since this network is already trained to defeat a gender discriminator in its encoding space.

This proves a suitable baseline comparison and starting point for a mapping function that needs to fool a gender discriminator in image space while simultaneously preserving subject verification performance.

We show the performance of using only the pretrained gender FaderNet and demonstrate how we can improve its performance by training a posterior processing mapping (UNET) using the loss proposed in Eq.10.We tested this framework on the FaceScrub dataset BID15 .

FIG5 shows how the output probabilities of the gender classification model approach the prior distribution of the dataset as α increases.

We see that sanitized images produce output gender probabilities close to the dataset prior even for relatively low α values.

Last column of FIG5 shows how the top-5 categorical accuracy of the subject verification task varies across different α values.

These results suggest that under these conditions we can achieve almost perfect privacy while maintaining reasonable utility performance.

DISPLAYFORM0

Inspired by information-theory bounds on the privacy-utility trade-off, we introduced a new paradigm where users and entities collaborate to achieve both utility and privacy per a user's specific requirements.

One salient feature of this paradigm is that it can be completely transparentinvolving only the use of a simple user-specific privacy filter applied to user data -in the sense that it requires otherwise no modifications to the system infrastructure, including the service provider algorithmic capability, in order to achieve both utility and privacy.

Representative architectures and results suggest that a collaborative user-controlled privacy approach can be achieved.

While the results here presented clearly show the potential of this approach, much has yet to be done, of particular note is extending this approach to continuous utility and privacy variables.

While the underlying framework still holds, reliably measuring information between continuous variables is a more challenging task to perform and optimize for.

The proposed framework provides privacy metrics and bounds in expectation; we are currently investigating how the privacy tails concentrate as data is acquired and if there is a need to use information theory metrics with worst-case scenario guarantees.

Modifying the information theory metrics to match some of the theoretical results in (local) differential privacy is also the subject of future research.

Privacy is closely related to fairness, transparency, and explainability, both in goals and in some of the underlying mathematics.

A unified theory of these topics will be a great contribution to the ML community.

Consider the equality I(U ; S) − I(U ; S|X) = I(S; Q(X)) − I(S; Q(X)|U ) + I(U ; X|Q(X)) − I(U ; X|Q(X), S).We know that DISPLAYFORM0 so we can guarantee DISPLAYFORM1 Proof.

Theorem 3.3 DISPLAYFORM2 and α ∈ [0, 1].Minimizing Eq.6 respecting Eq.5 and Eq.12 is equivalent to solving: DISPLAYFORM3 Consider the following relaxation of Eq.15 DISPLAYFORM4 where A and B are positive real values.

Eq.16 is a relaxation of Eq.15 because the space of possible tuples (A Q , B Q ) is included in the space of possible values of R

Suppose Q * is the solution to Eq.15, with corresponding values (A Q * , B Q * ), and suppose (A * , B * ) is the solution to Eq. 16.

We know DISPLAYFORM0 Assume DISPLAYFORM1 However, A Q * > 0 and A Q * + B * ≤ A * + B * ≤ K. Therefore, (A Q * , B * ) is a valid solution to Eq.16, and is smaller than the lower bound (A * , B * ).This contradiction arises from assuming A * > A Q * , we thus conclude that DISPLAYFORM2 Similarly for B * and B Q * we get B * ≤ B Q * .Additionally, Eq.16 is easily solvable and has solutions DISPLAYFORM3 Consequently, we proved DISPLAYFORM4

The following experiments attempt to show how close to the theoretical bound shown in Eq. 5 we can get by following Algorithm 1 under known conditions.

Consider the following scenario: Utility variable U and secret variable S are two Bernoulli variables with the following joint distribution: DISPLAYFORM0 where the marginal probabilities are P (U = 1) = ρ, P (S = 1) = β, and k is a parameter that controls the dependence between and S, k ∈ [0, min{ρ DISPLAYFORM1 For these experiments, we make both marginals equal to 0.5 (ρ = β = 0.5).

Note that when k = 1, U and S are independent (I(U ; S) = 0) and when k = 0 or k = 2 they reach maximum mutual information (I(U ; S) = H(U ) = H(S) = H b (0.5) = ln(1) nats).Our observations X will be taken in the extreme case, where X contains almost perfect information about the values of U and S. We do this by assuming that X ∈ R 2 is a Gaussian Mixture Model X with the following conditional distribution: DISPLAYFORM2 We choose a low σ = 0.05; this makes it so that every pair (u, s) is mapped to almost entirely disjoint regions, therefore knowing X gives nearly perfect information about (u, s) (I(U ; X) H(U ), I(S; X) H(S), I(U ; S|X) 0).

For added simplicity, the privacy filter is linear: Figure 6 shows how the raw and sanitized data are distributed for varying levels of codependence k and tradeoff α for linear sanitization functions.

Figure 7 shows that privacy filters optimized using Algorithm 1 learn effective privacy-preserving mappings close to the theoretical bounds, even for a simple filtering architecture.

They do so without any explicit modelling of the underlying data-generating distributions, and we can achieve different tradeoff points by simply modifing the parameter α.

Note that when variables U and S are perfectly independent or codependent, the linear filter is perfectly able to reach any point in the optimal bound.

For intermediate cases, the linear filter was capable of reaching the bound in the region where no utility is compromised, but was not capable of following the optimal tradeoff line for higher levels of privacy.

Figure 7: Top row shows the best privacy-utility loss so far on the validation set for different levels of codependence I(U ; S) and trade-off parameter α.

Middle row shows how the sum of the estimated informations approximate the information bound for the best privacy-utility loss so far on the validation set.

Finally, bottom row illustrates the trajectory in information space of the privacy-utility filters as they are being trained.

DISPLAYFORM3

Here, we explicitly elaborate on how we optimize the data-driven loss functions shown in equations 8, 9, and 10 using an adversarial approach.

We first detail the exact adversarial training setup that was used to perform the experiments in Section 5, and then provide the concrete network architectures used for all shown results.

Minimizing the objective functions in equations 8, 9, and 10, is a challenging problem in general.

By focusing our attention on Eq. 8, we see that each of the four loss terms have a distinct purpose: DISPLAYFORM0 The first three loss terms minimize a crossentropy objective for functions P η (s | q), P ψ (u | q), and P φ (u | x); this ensures that these functions are good estimators of the unknown true distributions of P (s | q), P (u | q), and P (u | x), where samples q are drawn from the learned sanitization mapping q = Q θ (x, z).

The final loss term attempts to find the best possible sampling function DISPLAYFORM1 We can approximately solve this problem by applying iterative Stochastic Gradient Descent to each of the individual loss terms with respect to their relevant parameters, this is similar to the procedure used to train Generative Adversarial Networks.

The algorithm we used to solve Eq. 8 is shown in Algorithm 1, similarly, algorithms to solve Eq. 9 and Eq. 10 are shown in Algorithm 2 and Algorithm 3 respectively.

DISPLAYFORM2 Evaluate crossentropy loss on raw utility inference 5: DISPLAYFORM3 Evaluate crossentropy loss on filtered utility inference 7: DISPLAYFORM4 Evaluate crossentropy loss on secret inference 9: DISPLAYFORM5 10: DISPLAYFORM6 Evaluate sanitation loss 11: DISPLAYFORM7 Stochastic gradient descent step on Q θ (x, z)12: until convergence

We now describe the exact architecture used to implement the privacy filter on all experiments shown in Section 5.

Figure 8 shows the network diagram.

The architecture presented in Figure 8 is fully convolutional, so the same network definition could be used across all three experiments by varying the input layer.

To speed up convergence to a good filtering solution, filters were initially trained to copy the image (under RMSE loss), and optionally infer some meaningful attribute from the input (in subject-within-subject, this attribute was a simple class label on whether the subject wished their privacy preserved).

We stress that this was only done for initialization, final training of the network was done exactly as described in Algorithm 1.

η ← η − lr∇ η H(η) Stochastic gradient descent step on P η (s | q)6: DISPLAYFORM0 Evaluate sanitation loss 7: DISPLAYFORM1 Stochastic gradient descent step on Q θ (x, z) Θ(θ) = (1 − α) DISPLAYFORM2 Evaluate sanitation loss 5: DISPLAYFORM3 Stochastic gradient descent step on Q θ (x, z) 6: until convergence Figure 8 : Architecture of privacy filter, based on UNET.

There is a single noise layer (shown in yellow) where standard Gaussian noise is injected into the network to make the resulting filtered image stochastic in nature.

The other notable component is the auxiliary label softmax, used for the subject-within-subject experiment.

This extra layer was trained only to initialize the network, but was not preserved during the final training stage.

Input image sizes are shown for the subject-within-subject experiment.

The architecture of the networks used to infer the utility and secret attribute in the emotion vs. gender experiment are identical, and are shown in Figure 9 .Networks used for the experiments in Section 7.2 are shown in FIG0 .All other networks used in the results section are implemented as described in their respective papers.

@highlight

Learning privacy-preserving transformations from data. A collaborative approach