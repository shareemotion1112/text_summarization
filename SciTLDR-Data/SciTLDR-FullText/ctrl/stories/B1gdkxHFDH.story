We propose an approach to training machine learning models that are fair in the sense that their performance is invariant under certain perturbations to the features.

For example, the performance of a resume screening system should be invariant under changes to the name of the applicant.

We formalize this intuitive notion of fairness by connecting it to the original  notion of individual fairness put forth by Dwork et al and show that the proposed approach achieves this notion of fairness.

We also demonstrate the effectiveness of the approach on two machine learning tasks that are susceptible to gender and racial biases.

As AI systems permeate our world, the problem of implicit biases in these systems have become more serious.

AI systems are routinely used to make decisions or support the decision-making process in credit, hiring, criminal justice, and education, all of which are domains protected by anti-discrimination law.

Although AI systems appear to eliminate the biases of a human decision maker, they may perpetuate or even exacerbate biases in the training data (Barocas & Selbst, 2016) .

Such biases are especially objectionable when it adversely affects underprivileged groups of users (Barocas & Selbst, 2016) .

In response, the scientific community has proposed many formal definitions of algorithmic fairness and approaches to ensure AI systems remain fair.

Unfortunately, this abundance of definitions, many of which are incompatible (Kleinberg et al., 2016; Chouldechova, 2017) , has hindered the adoption of this work by practitioners (Corbett-Davies & Goel, 2018) .

There are two types of formal definitions of algorithmic fairness: group fairness and individual fairness.

Most recent work on algorithmic fairness considers group fairness because it is more amenable to statistical analysis (Jiang et al., 2019) .

Despite their prevalence, group notions of algorithmic fairness suffer from certain shortcomings.

One of the most troubling is there are many scenarios in which an algorithm satisfies group fairness, but its output is blatantly unfair from the point of view of individual users (Dwork et al., 2011) .

In this paper, we consider individual fairness instead of group fairness.

At a high-level, an individually fair ML model treats similar users similarly.

Formally, we consider an ML model as a map h : X ??? Y, where X and Y are the input and output spaces.

The leading notion of individual fairness is metric fairness (Dwork et al., 2011) ; it requires d y (h(x 1 ), h(x 2 ))

??? Ld x (x 1 , x 2 ) for all x 1 , x 2 ??? X , (1.1)

where d x and d y are metrics on the input and output spaces and L ??? R + .

The fair metric d x encodes our intuition of which samples should be treated similarly by the ML model.

We emphasize that d x (x 1 , x 2 ) being small does NOT imply x 1 and x 2 are similar in all respects.

Even if d x (x 1 , x 2 ) is small, x 1 and x 2 may differ in certain attributes that are irrelevant to the ML task at hand, e.g. protected attributes.

This is why we refer to pairs of samples x 1 and x 2 such that d x (x 1 , x 2 ) is small as comparable instead of similar.

Despite its benefits, individual fairness is considered impractical because the choices of d x and d y are ambiguous.

Unfortunately, in application areas where there is disagreement over the choice of d x and/or d y , this ambiguity negates most of the benefits of a formal definition of fairness.

Dwork et al. (2011) consider randomized ML algorithms, so h(x) is generally a random variable.

They suggest probability metrics (e.g. total variation distance) as d y and defer the choice of d x to regulatory bodies or civil rights organizations, but we are unaware of commonly accepted choices of d x .

In this paper, we consider two data-driven choices of the fair metric: one for problems in which the sensitive attribute is reliably observed, and another for problems in which the sensitive attribute is unobserved.

Due to space constraints, we defer the details to the supplement (see Appendix B).

In this paper, we consider an adversarial approach to training individually fair ML models: we show that individual fairness is a restricted form of robustness: robustness to certain sensitive perturbations to the inputs of an ML model.

This connection allows us to leverage recent advances in adversarial training (Madry et al., 2017) to train individually fair ML models.

The paper is organized into four main sections.

In Section 2 we develop a method to investigate algorithmic bias/unfairness in ML models.

This leads to a training method that trains ML models to pass such investigations.

Our algorithmic developments are followed by a theoretical investigation (see Section 3) and an empirical study (see Section 3) of the efficacy of the proposed approach on two ML tasks.

To motivate our approach, imagine an investigator auditing an AI system for unfairness.

The investigator collects a set of audit data and compares the output of the AI system on comparable samples in the audit data.

For example, to investigate whether a resume screening system is fair, the investigator may collect a stack of resumes and change the names on the resumes of Caucasian applicants to names more common among the African-American population.

If the system performs worse on the edited resumes, the investigator concludes the system treats resumes from African-American applicants unfairly.

This is the premise of Bertrand and Mullainathan's celebrated investigation of racial discrimination in the labor market (Bertrand & Mullainathan, 2004) .

Abstractly, the investigator looks for inputs that are comparable to the training examples (the unedited resumes in the preceding example) but on which the predictor performs poorly.

In the rest of this section, we formulate an optimization problem to find such inputs.

Setup Recall X and Y are the spaces of inputs and outputs of the ML algorithm.

To keep things simple, we assume Y is discrete.

We also assume for now that we have a fair metric d x of the form

where ?? ??? S d??d ++ is a covariance matrix for the ML task at hand.

For example, suppose we are given a set of K "sensitive" directions A ??? R d??K .

These directions may be provided by a subject expert or estimated from data -we elaborate further in the experiments section and Appendix B. We wish to have a metric insensitive to any perturbation of features within the sensitive subspace ran(A).

To achieve this we set ?? = I ??? P ran(A) , i.e. the orthogonal complement projector of ran(A).

We equip X with such metric.

Let Z = X ?? Y, and equip it with the metric

We consider d 2 z as a transportation cost function on Z. This cost function encodes our intuition of which samples are comparable.

We equip ???(Z), the set of probability distributions on Z, with the Wasserstein distance

where C(P, Q) is the set of couplings between P and Q and the transportation cost function is

2 .

This fair Wasserstein distance inherits our intuition of which samples are comparable through the cost function; i.e. the fair Wasserstein distance between two probability distributions is small if they are supported on comparable areas of the sample space.

To investigate whether an AI system h performs disparately on comparable samples, the investigator collects a set of audit data {(x i , y i )} n i=1 that is independent of the training data and picks a nonnegative loss function : Z ?? H ??? R + to measure the performance of the AI system.

The investigator solves the optimization problem

where P n is the empirical distribution of the audit data and > 0 is a budget parameter.

We interpret as a moving budget that the investigator may expend to discover discrepancies in the performance of the AI system.

This budget forces the investigator to avoid moving samples to incomparable areas of the sample space.

We emphasize that equation 2.1 detects aggregate violations of individual fairness.

In other words, although the violations that the investigator's problem detects are individual in nature, the investigator's problem is only able to detect collective violations.

The implicit notion of fairness in equation 2.1 is

2) where ?? > 0 is a small tolerance for discrepancies in the performance of the ML model on comparable inputs.

Although equation 2.1 is an infinite-dimensional optimization problem, it is convex, so it is possible to exploit duality to solve it exactly.

It is known ) that the dual of equation 2.1 is

( 2.3)

The function c ?? is called the c-transform of .

This is a univariate optimization problem, and it is amenable to stochastic optimization (see Algorithm 1).

To optimize equation 2.3, it is imperative that the investigator is able to evaluate ??? x ((x, y), h).

Require: starting point?? 1 , step sizes ?? t > 0 1: repeat

It is known that the optimal point of equation 2.1 is the discrete measure

We call T ?? an unfair map because it reveals unfairness in the AI system by mapping samples in the audit data to comparable areas of the sample space that the system performs poorly on.

We note that T ?? may map samples in the audit data to areas of the sample space that are not represented in the audit data, thereby revealing disparate treatment in the AI system not visible from the audit data alone.

We emphasize that T ?? more than reveals disparate treatment in the AI system; it localizes the unfairness to certain areas of the sample space.

We present a simple example to illustrating fairness through robustness (a similar example appeared in Hashimoto et al. (2018)).

Consider the binary classification dataset shown in Figure 1 .

There are two subgroups of observations in this dataset, and (sub)group membership is the protected attribute (e.g. the smaller group contains observations from a minority subgroup).

In the figure, the horizontal axis displays (the value of) the protected attribute, while the vertical axis displays the discriminative attribute.

In Figure 1a we see the decision heatmap of a vanilla logistic regression, which performs poorly on the blue minority subgroup.

A tenable fair metric in this instance is a metric that downweights differences in the horizontal direction.

Figure 1b shows that such classifier is unfair with respect to the aforementioned fair metric, i.e. the unfair map equation 2.4 leads to significant loss increase by transporting mass along the horizontal direction with very minor change of the vertical coordinate.

Comparison with Dwork et al. (2011) Before moving on to training individually fair ML models, we compare our notion of fairness equation 2.2 with the definition of individual fairness in Dwork et al. (2011) .

Although we concentrate on the differences between the two definitions here, they are more similar than different: both formalize the intuition that the outputs of a fair ML model should perform similarly on similar inputs.

That said, there are two main differences between the two definitions.

First, instead of requiring the output of the ML model to be similar on all inputs comparable to a training example, we require the output to be similar to the training label.

The two definitions have similar implications.

Let (x i , y i ) be a training example and x i be an input point such that

On a single training example, we consider a given predictor h fair if sup

Comparing equation 2.5 and equation 2.6, we see that the two implications are similar, with in equation 2.6 playing the role of d y in equation 2.5 and 2?? in equation 2.6 playing the part of L equation 2.5.

The main benefit of our definition is it encodes not only (individual) fairness but also accuracy.

Second, we consider differences between datasets instead of samples by replacing Dwork et al's fair metric on inputs with the Wasserstein distance (on input distributions) induced by the fair metric.

The main benefits of this modifications are (i) it is possible to optimize equation 2.1 efficiently, (ii) we can show this modified notion of individual fairness generalizes.

We cast the fair training problem as training supervised learning systems that are robust to sensitive perturbations.

We propose solving the minimax problem

where c ?? is defined in equation 2.3.

This is an instance of a distributionally robust optimization (DRO) problem, and it inherits some of the statistical properties of DRO.

To see why equation 2.7 encourages individual fairness, recall the loss function is a measure of the performance of the AI system.

By assessing the performance of an AI system by its worse-case performance on hypothetical populations of users with perturbed sensitive attributes, minimizing equation 2.7 ensures the system performs well on all such populations.

In our toy example, minimizing equation 2.7 implies learning a classifier that is insensitive to perturbations along the horizontal (i.e. sensitive) direction.

In Figure 1c this is achieved by the algorithm we describe next.

To keep things simple, we assume the hypothesis class is parametrized by ?? ??? ?? ??? R d and replace the minimization with respect to H by minimization with respect to ??.

In light of the similarities between the DRO objective function and adversarial training, we borrow algorithms for adversarial training (Madry et al., 2017) to solve equation 2.7 (see Algorithm 2).

Require: starting point?? 1 , step sizes ?? t , ?? t > 0 1: repeat 2:

sample mini-batch (x 1 , y 1 ), . . . , (x B , y B ) ??? P n 3:

Related work Our approach to fair training is an instance of distributionally robust optimization (DRO), which minimize objectives of the form sup P ???U E P (Z, ??) , where U is a (data dependent) uncertainty set of probability distributions.

Other instances of DRO consider uncertainty sets defined by moment or support constraints (Chen et al., 2007; Delage & Ye, 2010; Goh & Sim, 2010) as well as distances between probability distributions, such as f -divergences (Ben-Tal et al., 2012; Lam & Zhou, 2015; Miyato et al., 2015; and Wasserstein distances (Shafieezadeh-Abadeh et al., 2015; Lee & Raginsky, 2017; Sinha et al., 2017; Hashimoto et al., 2018) .

Most similar to our work is Hashimoto et al. (2018): they show that DRO with a ?? 2 -neighborhood of the training data prevents representation disparity, i.e. minority groups tend to suffer higher losses because the training algorithm ignores them.

One advantage of picking a Wasserstein uncertainty set is the set depends on the geometry of the sample space.

This allows us to encode the correct notion of individual fairness for the ML task at hand in the Wasserstein distance.

Our approach to fair training is also similar to adversarial training (Madry et al., 2017) , which hardens AI systems against adversarial attacks by minimizing adversarial losses of the form sup u???U (z + u, ??), where U is a set of allowable perturbations (Szegedy et al., 2013; Goodfellow et al., 2014; Papernot et al., 2015; Carlini & Wagner, 2016; Kurakin et al., 2016) .

Typically, U is a scaled p -norm ball: U = {u : u p ??? }.

Most similar to our work is Sinha et al. (2017) : they consider an uncertainty set that is a Wasserstein neighborhood of the training data.

There are a few papers that consider adversarial approaches to algorithmic fairness.

Zhang et al. (2018) propose an adversarial learning method that enforces equalized odds in which the adversary learns to predict the protected attribute from the output of the classifier.

Edwards & Storkey (2015) propose an adversarial method for learning classifiers that satisfy demographic parity.

Madras et al. (2018) generalize their method to learn classifiers that satisfy other (group) notions of algorithmic fairness.

Garg et al. (2019) propose to use adversarial logit pairing ) to achieve fairness in text classification using a pre-specified list of counterfactual tokens.

One of the main benefits of our approach is it provably trains individually fair ML models.

Further, it is possible for the learner to certify that an ML model is individually fair a posteriori.

As we shall see, both are consequences of uniform convergence results for the DR loss class.

More concretely, we study how quickly the uniform convergence error

where W * is the Wasserstein distance on ???(Z) with a transportation cost function c * that is possibly different from c, vanishes.

We permit some discrepancy in the (transportation) cost function to study the effect of a data-driven choice of c. In the rest of this section, we regard c * as the exact cost function and c as a cost function learned from human supervision.

We start by stating our assumptions on the ML task:

(A1)

the feature space X is bounded: D max{diam(X ), diam * (X )} < ???; (A2) the functions in the loss class L = { (??, ??) : ?? ??? ??} are non-negative and bounded: 0 ??? (z, ??) ??? M for all z ??? Z and ?? ??? ??, and L-Lipschitz with respect to d x :

(A3) the discrepancy in the (transportation) cost function is uniformly bounded:

Assumptions A1 and A2 are standard (see (Lee & Raginsky, 2017 , Assumptions 1, 2, 3)), but A3 deserves further comment.

Under A1, A3 is mild.

For example, if the exact fair metric is

??min(?? * ) , We see that the error in the transportation cost function vanishes in the large-sample limit as long as ?? is a consistent estimator of ?? * .

We state the uniform convergence result in terms of the entropy integral of the loss class:

as the r-covering number of the loss class in the uniform metric.

The entropy integral is a measure of the complexity of the loss class.

Proposition 3.1 (uniform convergence).

Under Assumptions A1-A3, equation 3.1 satisfies

with probability at least 1 ??? t.

We note that Proposition 3.1 is similar to the generalization error bounds by Lee & Raginsky (2017) .

The main novelty in Proposition 3.1 is allowing error in the transportation cost function.

We see that the discrepancy in the transportation cost function may affect the rate at which the uniform convergence error vanishes: it affects the rate if ?? c is ?? P (

A consequence of uniform convergence is SenSR trains individually fair classifiers (if there are such classifiers in the hypothesis class).

By individually fair ML model, we mean an ML model that has a small gap sup P :W * (P,P * )

3) The gap is the difference between the optimal value of the investigator's optimization problem equation 2.1 and the (non-robust) risk.

A small gap implies the investigator cannot significantly increase the loss by moving samples from P * to comparable samples.

Proposition 3.2.

Under the assumptions A1-A3, as long as there is?? ??? ?? such that sup P :W * (P,P * )??? E P (Z,??) ??? ?? * (3.4)

for some ?? * > 0,?? ??? arg min ??????? sup P :W (P,Pn)??? E P (Z, h) satisfies

where ?? n is the uniform convergence error equation 3.1.

Another consequence of uniform convergence is equation 3.3 is close to its empirical counterpart sup P :W (P,Pn)??? E P (Z, ??) ??? E Pn (Z, ??) .

(3.5) In other words, the gap generalizes.

This implies equation 3.5 is a certificate of individual fairness; i.e. it is possible for practitioners to check whether an ML model is individually fair by evaluating equation 3.5.

Proposition 3.3.

Under the assumptions A1-A3, for any > 0,

??? 2?? n w.p.

at least 1 ??? t.

In this section, we present results from using SenSR to train individually fair ML models for two tasks: sentiment analysis and income prediction.

We pick these two tasks to demonstrate the efficacy of SenSR on problems with structured (income prediction) and unstructured (sentiment analysis) inputs and in which the sensitive attribute (income prediction) is observed and unobserved (sentiment analysis).

We refer to Appendix C and D for the implementation details.

Problem formulation We study the problem of classifying the sentiment of words using positive (e.g. 'smart') and negative (e.g. 'anxiety') words compiled by Hu & Liu (2004) .

We embed words using 300-dimensional GloVe (Pennington et al., 2014) and train a one layer neural network with 1000 hidden units.

Such classifier achieves 95% test accuracy, however it entails major individual fairness violation.

Consider an application of this sentiment classifier to summarizing customer reviews, tweets or news articles.

Human names are typical in such texts and should not affect the sentiment score, hence we consider fair metric between any pair of names to be 0.

Then sentiment score for all names should be the same to satisfy the individual fairness.

To make connection to group fairness, following the study of Caliskan et al. (2017) that reveals the biases in word embeddings, we evaluate the fairness of our sentiment classifier using male and female names typical for Caucasian and African-American ethnic groups.

We emphasize that to satisfy individual fairness, sentiment of any name should be the same.

Comparison metrics To evaluate the gap between two groups of names, N 0 for Caucasian (or female) and N 1 for African-American (or male), we report

, where h(n) k is logits for class k of name n (k = 1 is the positive class).

We use list of names provided in Caliskan et al. (2017) , which consists of 49 Caucasian and 45 African-American names, among those 48 are female and 46 are male.

The gap between African-American and Caucasian names is reported as Race gap, while the gap between male and female names is reported as Gend.

gap in Table 1 .

As in Speer (2017) we also compare sentiment difference of two sentences: "Let's go get Italian food" and "Let's go get Mexican food", i.e. cuisine gap (abbreviated Cuis.

gap in Table 1 ), as a test of generalization beyond names.

To embed sentences we average their word embeddings.

Sensitive subspace We consider 94 names that we use for evaluation as sensitive directions, which may be regarded as utilizing the expert knowledge, i.e. these names form a list of words that an expert believes should be treated equally.

Following procedure described in Appendix B.2 and Algorithm 3, embeddings of these names define sensitive subspace inducing the fair metric.

When expert knowledge is not available, or we wish to achieve general fairness for names, we utilize a side dataset of popular baby names in New York City.

1 The dataset has 11k names, however only 32 overlap with the list of names used for evaluation.

Embeddings of these names define a single group of comparable samples, hence factor analysis in Algorithm 3 reduces to SVD.

We use top 50 singular vectors to form the sensitive subspace.

It is worth noting that, unlike many existing approaches in Results From box-plots in Figure 2 we see that both race and gender gaps are significant when using baseline neural network classifier.

It tends to predict Caucasian names as "positive", while the median for African-American names is negative; the median sentiment for female names is higher than that for male names.

We considered three other approaches to this problem: algorithm of Bolukbasi et al. (2016) for pre-processing word embeddings; pre-processing via projecting out sensitive subspace that we used for training SenSR (this is analogous to Prost et al. (2019) ); training a distributionally robust classifier with Euclidean distance cost (Sinha et al., 2017) .

All approaches improved upon the baseline, however only SenSR can be considered individually fair.

Our algorithm practically eliminates gender and racial gaps and achieves the notion of individual fairness as can be seen from almost equal predicted sentiment score for all names.

We remark that using expert knowledge (i.e. evaluation names) allowed SenSR-E (E for expert) to further improve both group and individual fairness.

However we warn practitioners that if the expert knowledge is too specific, generalization outside of the expert knowledge may not be very good.

In Table 1 we report results averaged across 10 repetitions with 90%/10% train/test splits, where we also verify that accuracy trade-off with the baseline is minor.

In the right column we present the generalization check, i.e. comparing a pair of sentences unrelated to names.

Utilizing expert knowledge led to a fairness overfitting effect, however we still see improvement over other methods.

When utilizing SVD of a larger dataset of names we observe better generalization.

Our generalization check suggests that fairness over-fitting is possible, therefore datasets and procedure for verifying fairness generalization are needed.

Problem formulation Demonstrating the broad applicability of SenSR outside of natural language processing tasks, we apply SenSR to a classification task on the Adult (Dua & Graff, 2017 ) data set to predict whether an individual makes at least $50k based on features like gender and occupation for approximately 45,000 individuals.

Models that predict income without fairness considerations can contribute to the problem of differences in pay between genders or races for the same work.

Throughout this section, gender (male or female) and race (Caucasian or non-Caucasian) are binary.

Comparison metrics Arguably a classifier is individually unfair if the classifications for two data points that are the same on all features except demographic features are different.

Therefore, to assess individual fairness, we report spouse consistency (S-Con.) and gender and race consistency (GR-Con.), which are measures of how often classifications change only because of differences in demographic features.

For S-Con (resp.

GR-con), we make 2 (resp.

4) copies of every data point where the only difference is that one is a husband and the other is a wife (resp.

difference is in gender and race).

S-Con (resp.

GR-Con) is the fraction of corresponding pairs (resp.

quadruples) that have the same classification.

We also report various group fairness measures proposed by De-Arteaga et al. (2019) with respect to race or gender based on true positive rates, i.e. the ability of a classifier to correctly identify a given class.

See the Supplement for the definitions.

We report Gap RMS R , Gap RMS G , Gap max R , and Gap max G where R refers to race, and G refers to gender.

We use balanced accuracy (B-acc) instead of accuracy 2 to measure predictive ability since only 25% of individuals make at least $50k.

be the set of features x i ??? R D of the data except the coordinate for gender is zeroed and where x gi indicates the gender of individual i. For ?? > 0,

) + ?? w 2 , i.e. w g is the learned hyperplane that classifies gender given by logistic regression.

Let e g ??? R D (resp.

e r ) be the vector that is 1 in the gender (resp.

race) coordinate and 0 elsewhere.

We use the approach from Appendix B.1 to learn the fair metric.

In particular, the sensitive subspace is spanned by A = [w g , e g , e r ] and the fair metric is given by equation B.3.

Results See Table 2 for the average 3 of each metric on the test sets over ten 80%/20% train/test splits for Baseline, Project (projecting features onto the orthogonal complement of the sensitive subspace before training), CoCL (De-Arteaga et al., 2019 ), Adversarial Debiasing (Zhang et al., 2018 , and SenSR.

With the exception of CoCL (De-Arteaga et al., 2019), each classifier is a 100 unit single hidden layer neural network.

The Baseline clearly exhibits individual and group fairness violations.

While SenSR has the lowest B-acc, SenSR is the best by a large margin for S-Con. and has the best group fairness measures.

We expect SenSR to do well on GR-consistency since the sensitive subspace includes the race and gender directions.

However, SenSR's individually fair performance generalizes: the sensitive directions do not directly use the husband and wife directions, yet SenSR performs well on S-Con.

Furthermore, SenSR outperforms Project on S-Con and group fairness measures illustrating that SenSR does much more than just ignoring the sensitive subspace.

CoCL only barely improves group fairness compared to the baseline with a significant drop in Bacc and while Adversarial Debiasing also improves group fairness, it is worse than the baseline on individual fairness measures illustrating that group fairness does not imply individual fairness.

We consider the task of training ML systems that are fair in the sense that their performance is invariant under certain perturbations in a sensitive subspace.

This notion of fairness is closely related to the notion of individual fairness proposed by Dwork et al. (2011) .

One of the main barriers to the adoption of individual fairness is the lack of consensus on the fair metric in the formal definition of individual fairness.

To circumvent this issue, we consider two approaches to learning a fair metric from data: one for problems in which the sensitive attribute is observed, and another for problems in which the sensitive attribute is unobserved.

Given a data-driven choice of fair metric, we provide an algorithm that provably trains individually fair ML models.

A.1 PROOF OF PROPOSITION 3.1 By the duality result of , for any > 0, sup P :W * (P,P * )??? E P (Z, ??) ??? sup

where ?? n ??? arg min ?????0 ?? + E Pn c ?? (Z, ??) .

By assumption A3,

This bound is crude; it is possible to obtain sharper bounds under additional assumptions on the loss and transportation cost functions.

We avoid this here to keep the result as general as possible.

Similarly, sup

where ?? * ??? arg min ?????0 {?? + E P * c * ?? (Z, ??) .

Lemma A.1 (Lee & Raginsky (2017) ).

Let?? ??? arg min ?????0 ?? + E P c ?? (Z, ??) .

As long as the function in the loss class are L-Lipschitz with respect to

Proof.

By the optimality of??,

for any ?? ??? 0.

By Assumption A2, the right side is at most

Under review as a conference paper at ICLR 2020 By Lemma A.1, we have

We combine the preceding bounds to obtain

where

, ?? ??? ??} is the DR loss class.

In the rest of the proof, we bound sup f ???L c * Z f (z)d(P * ??? P n )(z) with standard techniques from statistical learning theory.

Assumption A2 implies the functions in F are bounded:

This implies has bounded differences, so ?? n concentrates sharply around its expectation.

By the bounded-differences inequality and a symmetrization argument,

WP at least 1 ??? t, where R n (F) is the Rademacher complexity of F:

Lemma A.2.

The Rademacher complexity of the DR loss class is at most

Proof.

To study the Rademacher complexity of L c , we first show that the L c -indexed

where we recalled equation A.1 in the second step.

We evalaute the integral on the right side to arrive at the stated bound:

which implies

WP at least 1 ??? t.

A.2 PROOFS OF PROPOSITIONS 3.2 AND 3.3

Proof of Proposition 3.2.

It is enough to show sup P :W * (P,P * )??? E P (Z,??) ??? ?? * + 2?? n because the loss function is non-negative.

We have

Proof of Proposition 3.3.

The loss function is bounded, so it is possible to bound E P * (Z, ??) ??? E Pn (Z, ??) by standard uniform convergence results on bounded loss classes.

Here we assume the sensitive attribute is discrete and is observed for a small subset of the training data.

Formally, we assume this subset of the training data has the form {(X i , K i , Y i )}, where K i is the sensitive attribute of the i-th subject.

To learn the sensitive subspace, we fit a softmax regression model to the data

and take the span of A = [a 1 . . .

a k ] as the sensitive subspace to define the fair metric as

This approach readily generalizes to sensitive attributes that are not discrete-valued: replace the softmax model by an appropriate generalized linear model.

In many applications, the sensitive attribute is part of a user's demographic information, so it may not be available due to privacy restrictions.

This does not preclude the proposed approach because the sensitive attribute is only needed to learn the fair metric and is neither needed to train the classifier nor at test time.

The main barrier to wider adoption of individual fairness is disagreement over the fair metric.

In this section, we consider the task of learning a fair metric from human supervision.

To keep things simple, we focus on learning a generalized Mahalanobis distance

where ??(x) : X ??? R d is a known feature map and ?? ??? S d??d + is a covariance matrix, from groups of comparable samples.

This type of feedback is common in the literature on debiasing learned representations.

For example, method of Bolukbasi et al. (2016) for removing gender bias in word embeddings relies on sets of words whose embeddings mainly vary in a gender subspace (e.g. (king, queen)).

Our approach is based on a factor model

is the sensitive/irrelevant (resp.

relevant) attributes of x i to the task at hand, and i is an error term.

For example, in Bolukbasi et al. (2016) , the learned representations are the embeddings of words in the vocabulary, and the sensitive attribute is the gender bias of the words.

The sensitive and relevant attributes are generally unobserved.

Recall our goal is to obtain ?? so that equation B.2 is small whenever v 1 ??? v 2 .

One possible choice of ?? is the projection matrix onto the orthogonal complement of ran(A), which we denote by P ran(A) .

Indeed,

which is small whenever v 1 ??? v 2 .

Although ran(A) is unknown, it is possible to estimate it from the learned representations and groups of comparable samples by factor analysis.

The factor model attributes variation in the learned representations to variation in the sensitive and relevant attributes.

We consider two samples comparable if their relevant attributes are similar.

In other words, if I ??? [n] is (the indices of) a group of comparable samples, then T + E E T I HE I , This suggests estimating ran(A) from the learned representations and groups of comparable samples by factor analysis.

We summarize our approach in Algorithm 3.

Algorithm 3 estimating ?? for the fair metric

This section is to accompany the implementation of the SenSR algorithm and is best understood by reading it along with the code implemented using TensorFlow.

We discuss choices of learning rates and few specifics of the code.

Words in italics correspond to variables in the code and following notation in parentheses defines corresponding name in Table 3 , where we summarize all hyperparameter choices.

Handling class imbalance Datasets we study have imbalanced classes.

To handle it, on every epoch(E) (i.e. number of epochs) we subsample a batch size(B) training samples enforcing equal number of observations per class.

This procedure can be understood as data augmentation.

Perturbations specifics Our implementation of SenSR algorithm has two inner optimization problems -subspace perturbation and full perturbation (when > 0).

Subspace perturbation can be viewed as an initialization procedure for the attack.

We implement both using Adam optimizer (Kingma & Ba, 2014) inside the computation graph for better efficiency, i.e. defining corresponding perturbation parameters as Variables and re-setting them to zeros after every epoch.

This is in contrast with a more common strategy in the adversarial robustness implementations, where perturbations (i.e. attacks) are implemented using tf.gradients with respect to the input data defined as a Placeholder.

Learning rates As mentioned above, in addition to regular Adam optimizer for learning the parameters we invoke two more for the inner optimization problems of SenSR.

We use same learning rate of 0.001 for the parameters optimizer, however different learning rates across datasets for subspace step(s) and full step(f ).

Two other related parameters are number of steps of the inner optimizations: subspace epoch(se) and full epoch(f e).

We observed that setting subspace perturbation learning rate too small may prevent our algorithm from reducing unfairness, however setting it big does not seem to hurt.

On the other hand, learning rate for full perturbation should not be set too big as it may prevent algorithm from solving the original task.

Note that full perturbation learning rate should be smaller than perturbation budget eps( ) -we always use /10.

In general, malfunctioning behaviors are immediately noticeable during training and can be easily corrected, therefore we did not need to use any hyperparameter optimization tools.

This data is imbalanced: 25% make at least $50k per year.

Furthermore, there is demographic imbalance with respect to race and gender as well as class imbalance on the outcome when conditioning on race or gender: 86% of individuals are white of which 26% make at least $50k a year; 67% of individuals are male of which 31% make at least $50k a year; 11% of females make at least $50k a year; and 15% of non-whites make at least $50k a year.

See Tables 4 and 5 for the full experiment results.

The tables reports the average and the standard error for each metric of 10 train and test splits.

To learn the hyperplane that classifies females and males, we use our implementation of regularized logistic regression with a batch size of 5k, 5k epochs, and .1 2 regularization.

For each model, we use the same 10 train/test splits where use 80% of the data for training.

Because of the class imbalance, each minibatch is sampled so that there are an equal number of training points from both the "income at least $50k class" and the "income below $50k class."

See Table 3 for the hyperparameters we used on the Baseline, Project, and SenSR where the hyperparameters are defined in Section C.

We used Zhang et al. (2018)'s adversarial debiasing implementation in IBM's AIF360 package (Bellamy et al., 2018) where the source code was modified so that each mini-batch is balanced with respect to the binary labels just as we did with our experiments and dropout was not used.

Hyperparameters are the following: adversary loss weight = .001, num epochs = 500, batch size = 1000, and privileged groups are defined by binary gender and binary race.

Let C be a set of classes, A be a binary protected attribute and Y,?? ??? C be the true class label and the predicted class label.

Then for a ??? {0, 1} and c ??? C define TPR a,c = P(?? = c|A = a, Y = c); G where C is composed of the two classes that correspond to whether someone made at least $50k, R refers to race, and G refers to gender.

<|TLDR|>

@highlight

Algorithm for training individually fair classifier using adversarial robustness

@highlight

This paper proposes a new definition of algorithmic fairness and an algorithm to provably find an ML model that satisfies the fairness contraint.