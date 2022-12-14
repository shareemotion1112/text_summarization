Formal verification techniques that compute provable guarantees on properties of machine learning models, like robustness to norm-bounded adversarial perturbations, have yielded impressive results.

Although most techniques developed so far requires knowledge of the architecture of the machine learning model and remains hard to scale to complex prediction pipelines, the method of randomized smoothing has been shown to overcome many of these obstacles.

By requiring only black-box access to the underlying model, randomized smoothing scales to large architectures and is agnostic to the internals of the network.

However, past work on randomized smoothing has focused on restricted classes of smoothing measures or perturbations (like Gaussian or discrete) and has only been able to prove robustness with respect to simple norm bounds.

In this paper we introduce a general framework for proving robustness properties of smoothed machine learning models in the black-box setting.

Specifically, we extend randomized smoothing procedures to handle arbitrary smoothing measures and prove robustness of the smoothed classifier by using $f$-divergences.

Our methodology achieves state-of-the-art}certified robustness on MNIST, CIFAR-10 and ImageNet and also audio classification task, Librispeech, with respect to several classes of adversarial perturbations.

Predictors obtained from machine learning algorithms have been shown to be vulnerable to making errors when the inputs are perturbed by carefully chosen small but imperceptible amounts (Szegedy et al., 2014; Biggio et al., 2013) .

This has motivated significant amount of research in improving adversarial robustness of a machine learning model (see, e.g. Goodfellow et al., 2015; Madry et al., 2018) .

While significant advances have been made, it has been shown that models that were estimated to be robust have later been broken by stronger attacks (Athalye et al., 2018; Uesato et al., 2018) .

This has led to the need for methods that offer provable guarantees that the predictor cannot be forced to misclassify an example by any attack algorithm restricted to produce perturbations within a certain set (for example, within an p norm ball).

While progress has been made leading to methods that are able to compute provable guarantees for several image and text classification tasks (Wong & Kolter, 2018; Wong et al., 2018; Raghunathan et al., 2018; Dvijotham et al., 2018; Katz et al., 2017; Huang et al., 2019; Jia et al., 2019) , these methods require extensive knowledge of the architecture of the predictor and are not easy to extend to new models or architectures, requiring specialized algorithms for each new class of models.

Further, the computational complexity of these methods grows significantly with input dimension and model size.

Consequently, to deal with these obstacles, recent work has proposed the randomized smoothing strategy for verifying the robustness of classifiers.

Specifically, Lecuyer et al. (2019) ; Cohen et al. (2019) have shown that robustness properties can be more easily verified for the smoothed version of a base classifier h: h s (x) = arg max y???Y P X?????(x) [h(X) = y] ,

where the labels returned by the smoothed classifier h s are obtained by taking a "majority vote" over the predictions of the original classifier h on random inputs drawn from a probability distribution ??(x), called the smoothing measure (here Y denotes the set of classes in the problem).

Lecuyer et al. (2019) showed that verifying the robustness of this smoothed classifier is significantly simpler than verifying the original classifier h and only requires estimating the distribution of outputs of the classifier under random perturbations of the input, but does not require access to the internals of the classifier h. We refer to this as black-box verification.

In this work, we develop a general framework for black-box verification that recovers prior work as special cases, and improves upon previous results in various ways.

Contributions Our contributions are summarized as follows:

1.

We formulate the general problem of black-box verification via a generalized randomized smoothing procedure, which extends existing approaches to allow for arbitrary smoothing measures.

Specifically, we show that robustness certificates for smoothed classifiers can be obtained by solving a small convex optimization problem when allowed adversarial perturbations can be characterized via divergence-based bounds on the smoothing measure.

2.

We prove that our certificates generalize previous results obtained in related work (Lecuyer et al., 2019; Cohen et al., 2019; Li et al., 2019) , and vastly extend the class of perturbations and smoothing measures that can be used while still allowing certifiable guarantees.

3.

We introduce the notion of full-information and information-limited settings, and show that the information-limited setting that has been the main focus of prior work leads to weaker certificates for smoothed probabilistic classifiers, and can be improved by using additional information (the distribution of label scores under randomized smoothing).

4.

We evaluate our framework experimentally on image and classification tasks, obtaining robustness certificates that improve upon other black-box methods either in terms of certificate tightness or computation time on robustness to 0 , 1 or 2 perturbations on MNIST, CIFAR-10 and ImageNet.

2 perturbations result from worst-case realizations of white noise that is common in many image, speech and video processing.

0 perturbations can model missing data (missing pixels in an image, or samples in a time-domain audio signal) while 1 perturbations can be used to model convex combinations of discrete perturbations in text classification (Jia et al., 2019) .

We also obtain the first, to the best of our knowledge, certifiably robust model for an audio classification task, Librispeech (Panayotov et al., 2015) , with variable-length inputs.

Consider a binary classifier h : X ??? {??1} given to us as a black box, so we can only access the inputs and outputs of h but not its internals.

We are interested in investigating the robustness of the smoothed classifier h s (defined in Eq. 1) against adversarial perturbations of size at most with respect to a given norm ?? .

To determine whether a norm-bounded adversarial attack on a fixed input x ??? X with h s (x) = +1 could be successful, we can solve the optimization problem

and check whether the minimum value can be smaller than 1 2 .

This is a non-convex optimization problem for which we may not even be able to compute gradients since we only have black-box access to h. While techniques have been developed to address this problem, obtaining provable guarantees on whether these algorithms actually find the worst-case adversarial perturbation is difficult since we do not know anything about the nature of h.

Motivated by this difficulty, we take a different approach: Rather than studying the adversarial attack in the input space X , we study it in the space of probability measures over inputs, denoted by P(X ).

Formally, this amounts to rewriting Eq. 2 as

This is an infinite dimensional optimization problem over the space of probability measures ?? ??? P(X ) subject to the constraint ?? ??? D = {??(x ) : x ??? x ??? }.

While this set is still intractable to deal with, we can consider relaxations of this set defined by divergence constraints between ?? and ?? = ??(x), i.e., D ??? {?? : D(?? ??) ??? D } where D denotes some divergence between probability distributions.

We will show in Section 3 that for several commonly used divergences (in fact, for any f -divergence; cf.

Ali & Silvey, 1966) , the relaxed problem can be solved efficiently.

To formulate the general verification problem, consider a specification ?? : X ??? Z ??? R: a generic function over the input space (that typically is a function of the classifier output) that we want to verify has certain properties.

Unless otherwise specified, we will assume that X ??? R d (we work in a d dimensional input space).

Our framework also involves a reference measure ?? (in the above example we would take ?? = ??(x)) and a collection of perturbed distributions D (in the above example we would take D = D x, = {??(x ) : x ??? x ??? }).

Verifying that a given specification ?? is robustly certified is equivalent to checking whether the optimal value of the optimization problem

is non-negative.

Solving problems of this form is the key workhorse of our general framework for black-box certification of adversarial robustness for smoothed classifiers.

Using these ingredients we introduce two closely related certification problems: information-limited robust certification and full-information robust certification.

In the former case, we assume that we are given only given access to

In the latter case, we are given full-access to specification ??.

The definitions are below.

Definition 2.1 (Information-limited robust certification).

Given reference distribution ?? ??? P(X ), probabilities ?? a , ?? b that satisfy ?? a , ?? b ??? 0, ?? a + ?? b ??? 1 and collection of perturbed distributions D ??? P(X ) containing ??, define the class of specifications S as

We say that S is information-limited robustly certified at ?? with respect to D if the following condition holds:

Note since we don't have access to ??, we need to prove that E X????? [??(X)] ??? 0 ????? ??? D is satisfied for all specifications in set S. Although the information-limited case may seem challenging because we need to provide guarantees that hold simultaneously over a whole class of specifications, it turns out that, for perturbation sets D specified by an f -divergence bound, this certification task can be solved efficiently using convex optimization.

Definition 2.2 (Full-information robust certification).

Given a reference distribution ?? ??? P(X ), a specification ?? : X ??? Z ??? R and a collection of perturbed distributions D ??? P(X ) containing ??, we say that ?? is full-information robustly certified at ?? with respect to D if the following condition holds:

Most often we are dealing with the case where we have full access to the specification ??, thus we should be able to certify using full-information robust certification.

However, prior works, Cohen et al. (2019) and Lecuyer et al. (2019) , have only provided solutions to certify with respect to the information-limited case where we cannot use all of the information about ??.

The framework we develop is a more general method that can be used in both information-limited and full-information scenarios.

We will demonstrate that our framework recovers certificates provided by Cohen et al. (2019 ), Li et al. (2019 and dominates Lecuyer et al. (2019) in the information-limited setting (see section 5).

Further, it can utilize full-information about the specification ?? to provide tighter certificates for smoothed probabilistic classifiers (see section 6).

We first note that the definitions above are sufficient to capture the standard usage of randomized smoothing as it has been used in past work (e.g. Lecuyer et al., 2019; Cohen et al., 2019) to verify the robustness of smoothed multi-class classifiers.

Specifically, consider smoothing a classifier h : X ??? Y with a finite set of labels Y using a smoothing measure ?? : X ??? P(X ).

The resulting randomly smoothed classifier h s is defined in Eq. 1.

Our goal is to certify that the prediction h s (x) is robust to perturbations of size at most measured by distance function

To pose this question within our framework, we choose the reference distribution ?? = ??(x), the set of perturbed distributions D x, = {??(x ) : d(x, x ) ??? }, and the following specifications.

Let c = h s (x).

For every c ??? Y \ {c}, we define the specification ?? c,c : X ??? {???1, 0, +1} as follows:

Then, Eq. 5 holds if and only if every ?? c,c , c = c, is robustly certified at ??(x) with respect to D x, (see Appendix A.1).

Dealing with the set D x, directly is difficult due to its possibly non-convex geometry.

In this section, we discuss specific relaxations of this set, i.e., choices for sets D such that D x, ??? D that are easier to optimize over.

In particular, we focus on a general family of constraint sets defined in terms of f -divergences.

These divergences satisfy a number of useful properties and include many well-known instances (e.g. relative entropy, total variation); see Appendix A.2 for details.

Definition 2.3. (f -divergence constraint set).

Given ??, ?? ??? P(X ), their f -divergence is defined as

where f : R + ??? R is a convex function with f (1) = 0.

Given a reference distribution ??, an f -divergence D f and a bound f ??? 0, we define the f -divergence constraint set to be:

Technically, this definition depends on the Radon-Nikodym derivative of ?? with respect to ??, but we ignore measure-theoretic issues in this paper for simplicity of exposition.

For continuous distributions, ?? and ?? should be treated as densities, and for discrete distributions as probability mass functions.

Relaxations using f -divergence This construction immediately allows us to obtain relaxations of D x, .

For example, by choosing f (u) = u log(u), we have the KL-divergence.

Using KL-divergence yields the following relaxation between norm-based and divergence-based constraint sets for Gaussian smoothing measures, i.e. ??(x) = N (x, ?? 2 I): Tighter relaxations can be constructed by combining multiple divergence-based constraints.

In particular, suppose F is a collection of convex functions each defining an f -divergence, and assume each f ??? F has a bound f associated with it.

Then we can define the constraint set containing perturbed distributions where all the bounds hold simultaneously ( Fig. 1 ):

In this paper, we work with the following divergences:

(1) R??nyi

1 d is an arbitrary distance function (not necessarily a metric e.g. 0).

where f (x) = x ?? ??? 1 (for ?? ??? 1) and

with f (x) = 1 ??? x ?? (for 0 ??? ?? ??? 1).

The limit ?? ??? ??? yields the infinite order R??nyi divergence

It turns out that the R??nyi and KL divergences are computationally attractive for a broad class of smoothing measures, while the Hockey-Stick divergences are theoretically attractive as they lead to optimal certificates in the information-limited setting.

However, Hockey-Stick divergences are harder to estimate in general, so we only use them for Gaussian smoothing measures.

In general, our framework can be used with any family of smoothing measures and any family of f divergences such that an upper bound on max ?????Dx, D f (?? ??) can be estimated efficiently.

We describe how f -divergence bounds can be obtained for several classes of smoothing measures:

Product measures Product measures are of the form ??(

X i and ?? i is a smoothing measure on X i .

We note that the discrete smoothing measure used in (Lee et al., 2019) , the Gaussian measure used in (Cohen et al., 2019) and the Laplacian measure used in (Li et al., 2019) are all of this form.

For such measures, one can construct bounds on R??nyi-divergences subject to any p norm constraint using a Lagrangian relaxation of the optimization problem max x : x???x p ??? R ?? (??(x ) ??(x)) (see Appendix A.3 for details).

Norm-based smoothing measures Appendix A.9.1 also shows how we can obtain bounds on the infinite-order R??nyi divergence R ??? , as well as on several classes of f -divergences, for norm-based smoothing measures of the form ??(x)[X]

??? exp(??? X ??? x ).

We now show how to reduce the problems of full-information and information-limited robust blackbox certification to simple convex optimization problems for general constraint sets D defined in terms of f -divergences.

This allows us, by extension, to solve the problem for related divergences like R??nyi divergences.

The following two theorems provide the main foundation for the verification procedures in the paper.

Theorem 1 (Verifying full-information robust certification).

Let D F be the constraint set defined by

and denote its convex conjugate 2 by f * ?? .

The specification ?? is robustly certified at ?? with respect to D F (cf.

Definition 2.2) if and only if the optimal value of the following convex optimization problem is non-negative:

The proof of Theorem 1, given in Appendix A.4, uses standard duality results to show that the dual of the verification optimization problem has the desired form.

We note that the special case where M = 1 reduces to Proposition 1 of Duchi & Namkoong (2018) , although the result is used in a completely different context in that work.

To build a practical certification algorithm from Theorem 1, we must do two things: 1) compute the optimal values of ?? and ??; and 2) estimate the expectation in Eq. 6.

Since the estimation of Algorithm 1 Full information certification (see appendix A.9 for details of subroutines) Inputs: Query access to specification ?? : X ??? [a, b], sampling access to reference distribution ??, divergences f i and bounds i , sample sizes N ,?? , confidence level ??.

the expectation cannot be done in closed form (due to the black-box nature of ??), we must rely on sampling.

In step 1 of Algorithm 1, we use N samples taken independently from ?? to estimate the expectation and solve the "sampled" optimization problem using an off-the-shelf solver (Diamond & Boyd, 2016) .

This gives us ?? * , ?? * , the estimated optimal values of ?? and ??, respectively.

Then we take these values and compute a high-confidence lower bound on the objective function of Eq. 6, which is then used to verify robustness.

In particular, in step 2, we compute a high-confidence upper bound E ub on the expectation term in the objective such that

] with probability at least ??; this computation involves taking?? independent samples from ?? and finding a confidence interval around the resulting empirical estimate of the expectation (for details, see Eq. 25 in Appendix A.9.1).

Plugging in this estimate back into Eq. 6 gives the desired high-confidence lower bound in step 3.

Details of both subroutines ESTIMATEOPT and UPPERCONFIDENCEBOUND used in Algorithm 1 are given in Algorithm 3 in Appendix A.9.2.

Our next theorem concerns the specialization of this verification procedure to the information-limited setting.

Theorem 2 (Verifying information-limited robust certification).

Let D F be as in Theorem 1, and S and ?? a , ?? b be as in Definition 2.1.

The class of specifications S is information-limited robustly certified at ?? with respect to D F (cf.

Definition 2.1) if and only if the optimal value of the following convex optimization problem is non-negative:

where ?? = (?? a , ?? b , 1 ??? ?? a ??? ?? b ) and ?? = (?? a , ?? b , ?? c ) are interpreted as probability distributions.

The proof of Theorem 2 is presented in Appendix A.5.

It is based on the fact that in the informationlimited setting, it is possible to directly compute the expectation in Eq. 6, and in fact this expectation only depends on ?? via the probabilities ?? a and ?? b .

Theorem 2 naturally leads to a certification algorithm, presented in Algorithm 2.

It simply uses the same procedure as Cohen et al. (2019) to compute a high-confidence lower bound ?? a on the probability of the correct class under randomized smoothing and then solves the convex optimization problem Eq. 7.

Again, we can use an off-the-shelf solver CVXPY (Diamond & Boyd, 2016) in step 2 for the general M > 1 case, but closed-form solutions are also available for M = 1; these are given in Table 4 in Appendix A.6.

Inputs: Query access to classifier h, correct label y, sampling access to reference distribution ??, divergences f i and bounds i , sample sizes N ,?? , confidence level ??.

1: Use the method in Section 3.2 of Cohen et al. (2019) to determine whether y is the most likely label produced for h(X) with X ??? ?? (using N samples) and if so, obtain a lower bound ?? a on the probability that h outputs the correct class with confidence ?? (using?? samples).

2: Obtain o * by solving Eq. 7 with ?? a ??? ?? a and

We now present theoretical results characterizing our certification methods and show the following:

1 For smoothed probabilistic classifiers, the full-information certificate dominates the information-limited one.

2 In the information-limited setting, if we define the f -divergence relaxation D F using HockeyStick divergences with specific parameters, then the computed certificate is provably tight.

Consider a soft binary classifier H : X ??? [0, 1] that outputs the probability of label +1 and consider a point x ??? X with H(x) > 1/2.

We define the specification ??(x) = H(x) ??? 1 2 .

Then, the smoothed classifier H s (x) = E X?????(x) [H(X)] predicts label +1 for all x with x ??? x ??? if and only if ?? is full-information robustly certified at ??(x) with respect to D x, = {??(x ) : x ??? x ??? }.

Note that the optimization in Theorem 1 depends on the full distribution of

On the other hand, to certify this robustness in the information-limited setting is equivalent to taking the specification ??(

To compare the two approaches, consider the objective of Eq. 6 with a single f -divergence constraint

where the third line follows from Jensen's inequality.

The proof of Theorem 2 shows that maximizing the final expression above with respect to ??, ?? is equivalent to the dual of the information-limited certification problem Eq. 7.

Thus, the information-limited setting computes a weaker certificate than the full-information setting for soft classifiers: Corollary 3.

The optimization problem of Eq. 6 with the specification ?? defined above has an optimal value that is greater than or equal to that of the optimization problem defined in Eq. 7.

Ideally, we would like to certify robustness of specifications with respect to sets of the form

The following result shows that the gap between the ideal D x, and the tractable constraint sets D F can be closed in the context of information-limited robust certification provided that we can measure hockey-stick divergences of every non-negative order ?? ??? 0.

The proof is given in Appendix A.7.

Define the constraint set

Then, S is information-limited robustly certified at ?? with respect to D if and only if S is informationlimited robustly certified at ?? with respect to D HS .

Thus, the optimal information-limited certificate in this case can be obtained by applying theorem 2 to D HS .

Table 1 summarizes the differences between our work and prior work in terms of the set of smoothing measures admitted, the offline computation cost of the certification procedure (which needs to be performed once for every possible perturbation size and choice of smoothing measure), the perturbations considered, whether they can use information beyond ?? a , ?? b to improve the certificates and whether they compute optimal certificates for a given smoothing measure in the informationlimited setting.

Cohen et al. (2019) study the problem of verifying hard classifiers smoothed by Gaussian noise, and derive optimal certificates with respect to 2 perturbations of the input.

Their results can be recovered as a special case of our framework when applied to sets defined via constraints on hockey-stick divergences.

Theorem 4 shows that the optimal certificate in the information-limited setting can be computed by applying theorem 2 to a constraint set with two hockey-stick divergences.

For the Gaussian measure ??(x) = N x, ?? 2 I , the HS divergence D HS,?? (??(x) ??(x )) can be computed in closed form and is purely a function of the 2 distance x ??? x 2 .

This enables us to efficienctly compute the ?? * a , ?? * b in theorem 4.

Thus, we obtain the following result (see Appendix A.7.2 for a proof):

Let D HS be defined as in theorem 4.

Then, applying theorem 2 to the constraint set D HS gives the following condition for robust certification:

where ?? g is the CDF of a standard normal random variable N (0, 1).

With straightforward algebra (worked out in appendix A.7.2) , this can be shown to be equivalent to

which is the certificate from Theorem 1 of Cohen et al. (2019) .

Lee et al. (2019) derive optimal certificates in the information-limited setting under the assumption that the likelihood ratio between measures

can only take values from a finite set.

This is a restrictive assumption that prevents the authors from accommodating natural smoothing measures like Gaussian or Laplacian measures.

Further, the complexity of computing the certificates in their framework is significant: O(d 3 ) computation (where d is the input dimension) is needed to certify smoothness to 0 perturbations.

The authors also derive tighter certificates for the special case of certain classes of decision trees by exploiting the tree structure.

In contrast, our framework can derive tighter certificates in the full-information setting for arbitrary classifiers.

Li et al. (2019) use properties of R??nyi divergences to derive robustness certificates for classifiers smoothed by Gaussian (resp.

Laplacian) noise under 2 (resp.

1 ) perturbations.

Their results can be obtained as special cases of ours; in particular, the R??nyi divergence certificates in Table 4 (in Appendix A.6) recover the results of Lemma 1 of Li et al. (2019) , but the latter are only applicable for Gaussian and Laplacian smoothing measures.

Lecuyer et al. (2019) introduce the notion of pixel differential privacy (pixelDP) and show that smoothing measures ?? satisfying pixelDP with respect to a certain type of perturbations lead to adversarially robust classifiers.

We can show that pixelDP can be viewed as a special instance of our certification framework with two specific hockey-stick divergences, and that the certificates derived from the pixelDP are provably dominated by the certificates from our framework (Theorem 1) with the same choice of divergences (see Corollary 7 in Appendix A.7.3).

To compare full-information certificates with limited-information certificates, we trained a ResNet-152 model on ImageNet with data augmentation by adding noise via sampling from a zero-mean Gaussian with variance 0.5 for each coordinate; during certification we sample from the same distribution to estimate lower bounds on the probability of the top predicted class.

For the fullinformation certificate, we use two hockey-stick divergences for the certificate and tune the parameters ?? to obtain the highest value in the optimization problem in step 2 of Algorithm 1.

For the infromationlimited certificate, our approach reduces to that of (Cohen et al., 2019) and we follow the same certification procedure.

We use N = 1000,?? = 1000000, ?? = .99 for both certification procedures.

The dashed line represents equal certificates and every point below the dashed line has a stronger certificate from the full information verification setting.

We run the comparison on 50 randomly selected examples from the validation set.

Each blue dot in Figure 2 corresponds to one test point, with its x coordinate representing the radius for full information certificate (from Algorithm 1) and y coordinate the information-limited certificate (which is equivalent to the certification procedure of Cohen et al., 2019) .

The running time of the full-information certification procedure is .2s per example (excluding the sampling cost) while the limited-information certification takes .002s per example.

Both procedures incur the same sampling cost as they use the same number of samples.

Figure 2 shows the difference between the two certificates.

The certificate provided by the fullinformation method is always stronger than the one given by the information-limited method.

The difference is often substantia -for one of the test samples, the full-information setting can certify robustness to 2 perturbations of radius = 9.42 in the full-information case while the limitedinformation certificate can only be provided for perturbation radius = 2.69.

In this section we consider 0 perturbations for both ImageNet and Binary MNIST (that is, we consider the number of pixels that can be perturbed without changing the prediction).

To test for scalability and tightness trade-offs of our framework, we compare our methodology to that of Lee et al. (2019) , as their work obtains the optimal bound for 0 .

We computed certificates for a single model for each classification task; for Binary MNIST we used the same model and training procedure as Lee et al. (2019) and for ImageNet, we used the model released in the Github code accompanying the paper of Lee et al. (2019) .

We use the discrete smoothing measure (appendix A.10) with parameter p = 0.8 for Binary MNIST certification, and p = 0.2 for ImageNet certification.

In our experiments we ran the certification procedures on all test examples from the Binary MNIST dataset, while for ImageNet, following prior work (Lee et al., 2019; Cohen et al., 2019) , on every 100th example from validation set.

The proportion of the examples for which accuracy can be certified are reported in Table 2 , 2019) .

However, building audio classifiers that are provably robust to adversarial attacks has been hard due to the complexity of audio processing architectures.

We take a step towards provably robust audio classifiers by showing that our approach can certify robustness of a classifier trained for speaker recognition on a state-of-the-art model for this task.

We focus on 0 perturbations that zero out a fraction of the audio sample, as they correspond to missing data in an audio signal.

Missing data can occur due to errors in recording audio or packets dropped while transmitting an audio signal over a network and is a common issue (Turner, 2010; Smaragdis et al., 2009 ).

In principle, the method of Lee et al. (2019) is applicable to compute robustness certificates, but at an impractically large computational cost, since the computation needs to be repeated whenever an input of a new length (for which a certificate has not previously been computed) arrives.

Concretely, this constitutes an O(d 3 ) computation for the length d ranging from 38 to 522,320 (the set of audio sequence lengths observed in the Librispeech test dataset (Panayotov et al., 2015) ).

The results are shown in Table 3 .

To the best of our knowledge, these are the first results showing certified robustness of an audio classifier.

We believe this is a significant advance towards certification of classifiers in audio and classifiers operating on variable-length inputs more generally. (Panayotov et al., 2015) .

From the Librispeech dataset, we created a corpus of sentence utterances from ten different speakers.

The classification task is, given an audio sample, to predict whom is speaking.

The test set consisted of 30 audio samples for each of the ten speakers.

We use a DeepSpeaker architecture (Li et al., 2017) , trained with the Adam optimizer (?? 1 = 0.9, ?? 2 = 0.5) for 50,000 steps with a learning rate of 0.0001.

The architecture is the same as that of Li et al. (2017) , except for changing the number of neurons in the final layer for speaker identification with ten classes.

Three models were trained with smoothing values of p = 0.5, p = 0.7, and p = 0.9, respectively, and we used the same values for certification.

Certification was performed using N = 1000,?? = 1000000, ?? = .99 using M = 1 R??nyi divergence, with ?? tuned to obtain the best certificate.

The proportion of samples with certified robustness for different accuracy values are reported, computed on 300 test set samples.

We have introduced a general framework for black-box verification using f -divergence constraints.

The framework improves upon state-of-the-art results on both image classification and audio tasks by a significant margin in terms of robustness certificates or computation time.

We believe that our framework can potentially enable scalable computation of robustness verification for more complex predictors and structured perturbations that can be modeled using f-divergence constraints.

Therefore, E X????? [?? c,c (X)] ??? 0 for all c ??? Y \ {c} is equivalent to c ??? arg max y???Y P X????? [h(X) = y].

For ?? = ??(x ), this means that h s (x ) = c (assuming the argmax is unique).

In other words, E X????? [?? c,c (X)] ??? 0 for all c ??? Y \ {c} and all ??(x ) ??? D x, if and only if h s (x ) = c for all x such that d(x, x ) ??? , proving the required robustness certificate.

Consider a soft classifier H : X ??? P(Y) that for each input x returns a probability distribution H(x) over the set of potential labels Y (e.g. H might represent the outputs of the soft-max layer of a neural network).

As in the case of hard classifiers, our methodology can be used to provide robustness guarantees for smoothed soft classifiers obtained by applying a smoothing measure ??(x) to the input.

In this case, the smoothed classifier is again a soft classifier given by

Let x be a fixed input point and write p = H s (x) ??? P(Y) to denote the distribution over labels.

A number of robustness properties about the soft classifier H s at x can be phrased in terms of Definition 2.2.

For example, let Y = {1, . . .

, K} and suppose that p 1 ??? p 2 ??? ?? ?? ?? ??? p K so that {1, . . .

, k} are the top k labels at x.

Then we can verify that the set of top k labels will not change when moving the input from x to x with x ??? x ??? by defining the specifications

and showing that all of these ?? i,j are robustly certified at ??(x) with respect to the set D x, defined above.

The case k = 1 corresponds to robustness of the standard classification rule outputting the label with the largest score.

Another example is robustness of classifiers which are allowed to abstain.

For example, suppose we build a hard classifierh out of H s which returns the label with the maximum score as long as the gap between this score and the score of any other label is at least ??; otherwise it produces no output.

Then we can certify thath will not abstain and return the label c = arg max y???Y p y at any point close to x by showing that every ?? c (z) = H(z) c ??? H(z) c ??? ??, c = c, is robustly certified at ??(x) with respect to D x, .

A number of well-known properties about f -divergences are used throughout the paper, both explicitly and implicitly.

Here we review such properties for the readers' convenience.

Proofs and further details can be found in, e.g., (Csisz??r et al., 2004; Liese & Vajda, 2006) .

Recall that the f -divergences can be defined for any convex function f : R + ??? R such that f (1) = 0.

We note that this requirement holds without loss of generality as the map x ??? f (x) ??? f (1) is convex whenever f is convex.

Any f -divergence D f satisfies the following:

2.

D f (?? ??) = 0, and D f (?? ??) = 0 implies ?? = ?? whenever f is strictly convex at 1.

for any function F , where F * (??) is the push-forward of ??.

u is again convex withf (1) = 0.

We being with the optimization problem

The constraint can be rewritten as

Forming the Lagrangian relaxation, we obtain

where the constraint |x i ??? x i | ??? is implied by x ??? x p ??? .

We can maximize separately over each x i to obtain

By weak duality, for any ?? ??? 0, this is an upper bound on Eq. 10.

We can minimize this bound over ?? ??? 0 to obtain the tightest bound.

The minimization over x i for each i can be solved in closed-form or via as simple 1-dimensional minimization problem for most smoothing measures.

For simplicity of exposition (and to avoid measure theoretic issues), we focus on the case where ??, ?? have well defined densities ??(x), ??(x) such that ??(x) > 0 whenever ??(x) > 0.

We begin by rewriting the optimization problem in terms of the likelihood ratio r(X) =

where the first two equalities follow directly by plugging in ??(X) = ??(X)r(X) and the third is obtained using the fact that ?? is a probability measure.

Using these relations, the optimization over ?? can be rewritten as

where r ??? 0 denotes that r(x) ??? 0 ???x ??? X .

The optimization over r is a convex optimization problem and can be solved using Lagrangian duality as follows -we first dualize the constraints on r to obtain

By strong duality, it holds that maximizing the final expression with respect to ?? ??? 0, ?? achieves the optimal value in Eq. 11.

Thus, if the optimal value is smaller than 0, the specification is not robustly certified and if it is larger than 0, the specification is robustly certified.

Finally, since we are ultimately interested in proving that the objective is non-negative, we can restrict ourselves to ?? ??? 0 such that i ?? i = 1 (since if the optimal ?? added up to something larger, we could simply rescale the values to add up to 1 and multiply ?? by the same scaling factor without changing the sign of the objective function).

This concludes the proof of correctness of the certificate Eq. 6.

For the next result, we observe that when ?? is ternary valued, the optimization over ??, ?? above can be written as max

where

Writing out the expression for f * , we obtain

where the second inequality follows from strong duality.

The inner maximization is unbounded unless y???{a,b,c}

One thing to note is that, we can rewrite these constraints in terms of ?? = ?? ??, i.e. ?? y = ?? y ?? y for y ??? {a, b, c}. These constraints ensure that ?? is a probability distribution over {+1, 0, ???1} and furthermore

Thus, the second constraint above is equivalent to D fi (?? ??) ??? i .

Writing the optimization problem in terms of ??, we obtain min

In this section we present closed-form certificates for the information-limited setting which can be derived from Theorem 2 for M = 1.

The results are summarized in Table 4 .

In the next subsections we present the derivation of the certificates for Hockey-Stick and R??nyi divergences.

The certificates for the KL and infinite R??nyi divergence can be derived by taking limits of the R??nyi certificate (as ?? ??? 1, ??? respectively).

The function f (u) = max(u ??? ??, 0) ??? max(1 ??? ??, 0) is a convex function with f (1) = 0.

Then, we have Table 4 : Certificates for various f -divergences for the information-limited setting.

Note that the R??nyi divergences are not proper f -divergences, but are defined as R ?? (?? ??) = 1 ?????1 log(1 + D f (?? ??)).

The infinite R??nyi divergence, defined as sup x log(??(x)/??(x)), is obtained by taking the limit ?? ??? ???. All certificates depend on the gap between ?? a and ?? b .

Notation:

The certificate given by Eq. 6 in Theorem 1 for this divergence in the case of a smoothed hard classifier takes the form

where the specification takes the values

Plugging in the expression for f * the objective function above takes the form

where we use the notation [u] + = max(u, 0) and assumed the constraints ?? ??? ?? ??? 1 since the objective is ?????? otherwise.

If ?? ??? 1, the objective is increasing monotonically in ??, so the optimal value is to set ?? to its upper bound ?? ??? 1.

Plugging this in, the possible values of the derivative with respect to ?? are

Thus, if ??? ???? a , the maximum is attained at 2, if ???? a ??? ??? ??(1 ??? ?? b ), the maximum is attained at 1, else the maximum is attained at 0, leading to the certificate:

Thus, the certificate is non-negative only if

The case ?? ??? 1 can be worked out similarly, leading to

The two cases can be combined as

We consider the cases ?? ??? 1 and ?? ??? 1 separately.

Suppose we have a bound on the R??nyi divergence

Then the certificate Eq. 6 simplifies to (after some algebra)

Setting the derivative with respect to ?? to 0 and solving for ??, we obtain

, and the optimal certificate reduces to

For this number to be positive, we need that ?? ??? 0 and

The LHS above evaluates to

where ?? = 1 ?? ??? 0.

Maximizing this expression with respect to ??, we obtain

, so that the certificate reduces to

Taking logarithms now gives the result.

is a convex function with f (1) = 0.

Then, we have

Then the certificate from Eq. 6 reduces to

with the constraint ?? ??? ???1 (otherwise the certificate is ??????).

Setting the derivative with respect to ?? to 0 and solving for ??, we obtain

Plugging this back into the certificate and setting ?? = ?? 1????? , we obtain

For this number to be positive, we require that

The LHS of the above expression evaluates to

where ?? = ??? 1 ?? .

Maximizing this expression over ?? ??? [0, 1], we obtain the final certificate to be

Taking logarithms, we obtain

A.7.1 PROOF OF THEOREM 4

At a high level, the proof shows that, in the information-limited case, to achieve robust certification under an arbitrary set of constraints D it suffices to know the "envelope" of D with respect to all hockey-stick divergences of order ?? ??? 0, i.e. the function ?? ??? max ?????D D HS,?? (?? ??) captures all the necessary information to provide information-limited robust certification with respect to D.

We start by considering the following optimization problem:

In the information-limited setting, this problem attains the minimum expected value over ?? ??? S.

Here 1[??(X) = 1] denotes the indicator function.

It will be convenient to write this in a slightly different form: Rather than looking at the outputs of ?? as the +1, 0, ???1, we look at them as vectors in R 3 : Then, we can write the optimization problem Eq. 12 equivalently as

We first consider the minimization over ?? for a fixed value of ??.

We begin by observing that since the objective is linear, the optimization over ?? can be replaced with the optimization over the convex hull of the set of ?? that satisfy the constraints (Bubeck, 2013) .

Since each input x ??? X can be mapped independently of the rest, the convex hull is simply the cross product of the convex hull at every x, to obtain the constraint set

Therefore, the optimization problem reduces to

This is a convex optimization problem in ??. Denote

Considering the dual of this optimization problem with respect to the optimization variable ??, we obtain

Since we can choose ??(x) independently for each x ??? X , we can minimize each term in the expectation independently to obtain

This implies that the Lagrangian evaluates to

We now consider two cases:

Case 1 (?? a ??? ?? b ??? 0) In this case, we can see that

Then, the Lagrangian reduces to

Case 2 (?? b ??? ?? a ??? 0) In this case, we can see that

Then, the Lagrangian reduces to

We know that 1 ??? ?? a ??? ?? b and ?? b ??? ?? a .

If ?? b > ?? a , by choosing ?? a = ?? a + ?? and ?? b = ?? b ??? ?? for some small ?? > 0, we know that the the sum of the first three terms would reduce while the final term would remain unchanged.

Thus, at the the optimum in this case, we can assume ?? a = ?? b and we obtain

Final analysis of the Lagrangian Combining the two cases we can write the dual problem as

By strong duality, the optimal value of the above problem precisely matches the optimal value of Eq. 14 (and hence Eq. 12).

Thus, information limited robust certification with respect to D holds if and only if Eq. 15 has a non-negative optimal value for each ?? ??? D. Since we have that

information-limited robust certification holds if and only if the optimal value of

is non-negative.

Further, since the optimal value only depended on the value of D HS,?? (?? ??) for ?? ??? 0, it is equivalent to information-limited robust certification with respect to D HS .

The above argument also shows that in this case, information-limited robust certification with respect to D is equivalent to requiring that the following convex optimization problem has a non-negative optimal value:

Let ?? * a , ?? * b be the optimal values attained.

Since this certificate depends only on the value of two Hockey-stick divergences at ?? * a , ?? * b , it must coincide with the application of theorem 2 to the constraint set D HS defined by constraints on these hockey-stick divergences (as we know that 2 computes the optimal certificate for any constraint set defined only by a set of f-divergences).

This observation completes the proof.

Theorem 4 gives us the optimal limited-information certificate problem provided that we can compute

for each ?? ??? 0.

In particular, when ?? is a Gaussian measure ??(x) = N x, ?? 2 I , we can leverage the following result from Balle & Wang (2018) .

Lemma 6.

Let ?? g be the CDF of a standard normal random variable N (0, 1).

For any ?? ??? 0 and x ??? R d we have

Applying Eq. 17 to the expression in Lemma 6 proves Corollary 5.

Proof of Corollary 5.

With the notation from Theorem 4 we have, for ?? ??? 0,

Plugging this expression into Eq. 17 allows us to verify information-limited robust certification of N x , ?? 2 I with respect to D x, = {N x , ?? 2 I : x ??? x 2 ??? } by solving

Eq. 9 then follows from setting the derivatives of this expression to 0 with respect to ?? a , ?? b and imposing the condition that the optimal solution is non-negative.

To check that Corollary 5 is equivalent to the optimal certification in (Cohen et al., 2019 , Theorem 1) we first recall that, in our notation, their result can be stated as: the class of specifications S in Definition 2.1 is information-limited robustly certified at ?? = N x, ?? 2 I with respect to

The equivalence between Eq. 9 and Eq. 18 now follows from the identity 1 ??? ?? g (??) = ?? g (?????) and the monotonicity of ?? g :

Pixel differential privacy (pixelDP) was introduced in Lecuyer et al. (2019) using the same similarity measure between distributions used in differential privacy: a distribution-valued function G : R d ??? P(Z) satisfies (??, ?? )-pixelDP with respect to p perturbations if for any x ??? x p ??? 1 it holds that D DP,e ?? (G(x) G(x )) ??? ?? , where

and the supremum is over all (measurable) subsets E of Z. In particular, Lecuyer et al. show that using a smoothing measure ?? satisfying pixelDP with respect to p leads to adversarially robust classifiers against p perturbations.

To show that their result fits as a particular instance of our framework, take ?? = ??(x) and fix ?? ??? 0 and ?? ??? [0, 1].

Due to the symmetry of the constraint x ??? x p ??? 1, if ?? satisfies (??, ?? )-pixelDP with respect to p perturbations, then we have the relaxation condition {??(x ) :

Now we recall that Barthe & Olmedo (2013) noticed that D DP,e ?? is equivalent to the hockey-stick divergence D HS,?? of order ?? = e ?? .

Thus, since f -divergences are closed under reversal (property 4 in Appendix A.2), we see that the constraint set D ??,?? can be directly written in the form D F (cf.

Section 2.2).

The main result in Lecuyer et al. (2019) is a limited-information black-box certification method for smoothed classifiers.

The resulting certificate for, which provides certification with respect to D ??,?? , is given by

For comparison, the certificate we obtain for the relaxation {?? : D DP,e ?? (?? ??) ??? ?? } of D ??,?? (HS certificate in Table 4 ) already improves on the certificate by Lecuyer et al. whenever ?? a ??? ?? b ??? (?? ??? 1)(1 ??? ?? a ??? ?? b ), which, e.g., always holds in the binary classification case.

Furthermore, since Theorem 2 provides optimal certificates for D, we have the following result.

Corollary 7.

The optimal certificates for the constraint set D (cf.

Eq. 20) obtained from Theorem 2 are stronger than those obtained from Eq. 21.

Lemma 8.

The smoothing measure ?? : X ??? P(X ) with density ??(

if x is any norm.

Further, if f is convex function with f (1) = 0 such that f 1 u is convex and monotonically increasing in u, then

Proof.

By the triangle inequality, we have

Similarly, for f that satisfy the conditions of the theorem, it can be shown that D f (??(x ) ??(x)) is convex in x so that its maximum over the convex set x ??? x ??? is attained on the boundary.

For several norms, the optimization problem in Eq. 22 can be solved in closed form.

These include 1 , 2 , ??? norms and the matrix spectral norm and nuclear norm (the final two are relevant when X is a space of matrices).

The results are documented in Table 5 .

Thus, every f -divergence that meets the conditions of Lemma 8 can be estimated efficiently for these norms.

In particular, the divergences that are induced by the functionsf (u ????? ) for any monotonic convex functionf and ?? ??? 0 satisfy this constraint.

This gives us a very flexible class of f -divergences that can be efficiently estimated for these norm-based smoothing measures.

Table 5 : Bounds on f -divergences: e 0 is the vector with 1 in the first coordinate and zeros in all other coordinates and 1 is the vector with all coordinates equal to 1.

?? p refers to the smoothing measure induced by the p norm, U(S) refers to the uniform measure over the set S, O is the set of orthogonal matrices and B p = { z p ??? 1} is the unit ball in the p norm.

Efficient sampling The only other requirement for obtaining a certificate computationally is to be able to sample from ??(x) to estimate ?? a , ?? b .

Since ??(x) is log-concave, there are general purpose polynomial time algorithms for sampling from this measure.

However, for most norms, more efficient methods exist, as outlined below.

The random variable X ??? ??(x) can be obtained as X = x + Z with Z ??? ??(0).

Thus, to sample from ??(x) for any x it is enough to be able to sample from ??(0).

For ?? 1 , this reduces to sampling from a Laplace distribution which can be done easily.

For ?? ??? , (Steinke & Ullman, 2016) give the following efficient sampling procedure: first sample r from a Gamma distribution with shape d + 1 and mean d + 1, i.e. r ??? ??(d + 1, 1), and then sample each

Theorem 9 gives a short proof of correctness for this procedure.

Theorem 10 also has a similar result for the case of ?? 2 and Table 5 lists the sampling procedures for several norms.

Proof.

We first compute the normalization constant for a density of the form ??? e ??? z ??? as follows:

Next we show the density of Z satisfies p Z (z) = e 2 .

Applying Bernstein's inequality to the sum and the sum of the squares of these random variables, we get the empirical Bernstein bound (Audibert et al., 2009) , which states that with probability at least 1 ??? ??, |Z ??? m| ??? 2?? 2 log(3/??) N + 3R log(3/??) t .

The main benefit of the above inequality is that as long as the variance of the sample Z 1 , . . .

, Z N is small, the convergence rate becomes essentially O(1/N ) instead of the standard O(1/ ??? N ).

Also, since Eq. 23 only contains empirical quantities apart from the range R, it can be used to obtain computable bounds for the expectation ??: with probability at least 1 ??? ??, Z ??? 2?? 2 log(3/??) N ??? 3R log(3/??) N ??? m ???Z + 2?? 2 log(3/??) N + 3R log(3/??) N .

A.9.2 SUBROUTINES FOR ALGORITHM 1

The bound in Eq. 24 can be applied to approximate the expectation in Eq. 6 with high probability for given values of ?? and ??.

More specifically, if the function f * ?? (?? ??? ??(??)) is bounded with range R, then taking N samples X 1 , . . .

, X N independently from ??, and defining Z i = f * ?? (?? ??? ??(X i )), and Z and?? 2 as above, Eq. 24 implies that with probability at least 1 ??? ??,

Plugging in this bound to Eq. 6 gives a high-probability lower bound for the function to be maximized for any given ?? and ??.

Details of the above procedures are given in Algorithm 3.

We use an off-the-shelf convex optimization solver (Diamond & Boyd, 2016) in the ESTIMETEOPT subroutine.

Algorithm 3 return ?? * , ?? * .

end function function UPPERCONFIDENCEBOUND(??, ??,?? ,

, a, b, ??, ??, ??) Sample X 1 , . . .

, X?? ??? ?? and compute

.

Set E ub ???Z + 2?? 2 log(3/??) N + 3R log(3/??) N .

return E ub .

end function A.10 0 SMOOTHING MEASURE We can also handle discrete perturbations in our framework.

A natural case to consider is 0 perturbations.

In this case, we assume that X = A d where A = {1, . . .

, K} is a discrete set.

Then, we can choose

where p + q = 1, p ??? q ??? 0, and p denotes the probability that the measure retains the value of x and q K???1 denotes a uniform probability of switching it to a different value.

In this case, it can be shown that for every ?? > 0 that This can be extended to structured discrete perturbations by introducing coupling terms between the perturbations:

This would correlate perturbations between adjacent features (which for example may be useful to model correlated perturbations for time series data).

Since this can be viewed as a Markov Chain, R??nyi divergences between ??(x), ??(x ) are still easy to compute.

Here we compare our certificates to Lecuyer et al. (2019) on the MNIST, CIFAR-10 and ImageNet datasets.

The smoothing distribution is as described in A.8; a zero mean Laplacian distribution with smoothing value defined by the scale of the distribution.

We first describe the hyperparameters used in training and certification for each of the datasets.

For all datasets, images were normalized into a [0,1] range.

MNIST hyperparameters: We trained a standard three layer CNN ReLU classifier for 50,000 steps with a batch size of 128 and a learning rate of 0.001.

The smoothing value during training was set to 1.0.

For certification we use N = 1K,?? = 10M, ?? = .99, and sweep over a range of smoothing values between 0.5 and 1.5 and report the best certificate found.

Certified accuracy is reported on 1,000 MNIST test set images.

We trained a Wide ResNet classifier for 50,000 training steps with a batch size of 32 and a learning rate of 0.001.

The smoothing value during training was set to 0.2.

For certification we use N = 1K,?? = 1M, ?? = .99, and sweep over a range of smoothing values between 0.1 and 0.5 and report the best certificate found.

Certified accuracy is reported on 1,000 CIFAR-10 test set images.

We trained a ResNet-152 classifier for 1 million training steps with a batch size of 16 and an initial learning rate of 0.1 that is decayed by a factor of ten every 25,000 steps.

The smoothing value during training was set to 0.1.

For certification we use N = 1K,?? = 100K, ?? = .99, and sweep over a range of smoothing values between 0.05 and 0.25 and report the best certificate found.

Certified accuracy is reported on 500 ImageNet validation set images.

Certified Accuracy Results for MNIST can be seen in table 6, CIFAR-10 and ImageNet results are shown in figure 3.

We significantly outperform Lecuyer et al. (2019) on all three datasets.

<|TLDR|>

@highlight

Develop a general framework to establish certified robustness of ML models against various classes of adversarial perturbations