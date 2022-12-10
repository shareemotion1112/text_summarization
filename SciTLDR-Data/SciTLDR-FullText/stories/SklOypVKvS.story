Measuring Mutual Information (MI) between high-dimensional, continuous, random variables from observed samples has wide theoretical and practical applications.

Recent works have developed accurate MI estimators through provably low-bias approximations and tight variational lower bounds assuming abundant supply of samples, but require an unrealistic number of samples to guarantee statistical significance of the estimation.

In this work, we focus on improving data efficiency and propose a Data-Efficient MINE Estimator (DEMINE) that can provide a tight lower confident interval of MI under limited data, through adding cross-validation to the MINE lower bound (Belghazi et al., 2018).

Hyperparameter search is employed and a novel meta-learning approach with task augmentation is developed to increase robustness to hyperparamters, reduce overfitting and improve accuracy.

With improved data-efficiency, our DEMINE estimator enables statistical testing of dependency at practical dataset sizes.

We demonstrate the effectiveness of DEMINE on synthetic benchmarks and a real world fMRI dataset, with application of inter-subject correlation analysis.

Mutual Information (MI) is an important, theoretically grounded measure of similarity between random variables.

MI captures general, non-linear, statistical dependencies between random variables.

MI estimators that estimate MI from samples are important tools widely used in not only subjects such as physics and neuroscience, but also machine learning ranging from feature selection and representation learning to explaining decisions and analyzing generalization of neural networks.

Existing studies on MI estimation between general random variables focus on deriving asymptotic lower bounds and approximations to MI under infinite data, and techniques for reducing estimator bias such as bias correction, improved signal modeling with neural networks and tighter lower bounds.

Widely used approaches include the k-NN-based KSG estimator (Kraskov et al., 2004) and the variational lower-bound-based Mutual Information Neural Estimator (MINE) family (Belghazi et al., 2018; Poole et al., 2018) .

Despite the empirical and asymptotic bias improvements, MI estimation has not seen wide adoption.

The challenges are two-fold.

First, the analysis of dependencies among variables -let alone any MI analyses for scientific studies -requires not only an MI estimate, but also confidence intervals (Holmes & Nemenman, 2019) around the estimate to quantify uncertainty and statistical significance.

Existing MI estimators, however, do not provide confidence intervals.

As low probability events may still carry a significant amount of information, the MI estimates could vary greatly given additional observations (Poole et al., 2018) .

Towards providing upper and lower bounds of true MI under limited number of observations, existing MI lower bound techniques assume infinite data and would need further relaxations when a limited number of observations are provided.

Closest to our work, Belghazi et al. (2018) studied the lower bound of the MINE estimator under limited data, but it involves bounds on generalization error of the signal model and would not yield useful confidence intervals for realistic datasets.

Second, practical MI estimators should be insensitive to the choice of hyperparameters.

An estimator should return a single MI estimate with its confidence interval irrespective of the type of the data and the number of observations.

For learning-based approaches, this means that the model design and optimization hyperparameters need to not only be determined automatically but also taken into account when computing the confidence interval.

Towards addressing these challenges, our estimator, DEMINE, introduces a predictive MI lower bound for limited samples that enables statistical dependency testing under practical dataset sizes.

Our estimator builds on top of the MINE estimator family, but performs cross-validation to remove the need to bound generalization error.

This yields a much tighter lower bound agnostic to hyperparameter search.

We automatically selected hyperparameters through hyperparameter search, and a new cross-validation meta-learning approach is developed, based upon few-shot meta-learning, to automatically decide initialization of model parameters.

Meta-overfitting is strongly controlled through task augmentation, a new task generation approach for meta-learning.

With these improvements, we show that DEMINE enables practical statistical testing of dependency for not only synthetic datasets but also for real world functional Magnetic Resonance Imaging (fMRI) data analysis capturing nonlinear and higher-order brain-to-brain coupling.

Our contributions are summarized as follows: 1) A data-efficient Mutual Information Neural Estimator (DEMINE) for statistical dependency testing; 2) A new formulation of meta-learning using Task Augmentation (Meta-DEMINE); 3) Application to real life, data-scarce applications (fMRI).

A widely used approach for estimating MI from samples is using k-NN estimates, notably the KSG estimator (Kraskov et al., 2004) .

Gao et al. (2017) provided a comprehensive review and studied the consistency and of asymptotic confidence bound of the KSG estimator (Gao et al., 2018) .

MI estimation can also be achieved by estimating individual entropy terms through kernel density estimation (Ahmad & Lin, 1976) or cross-entropy (McAllester & Statos, 2018) .

Despite their good performance on random variables with few dimensions, MI estimation on high-dimensional random variables remains challenging for commonly used Gaussian kernels.

Fundamentally, estimating MI requires accurately modeling the random variables, where high-capacity neural networks have shown excellent performance on complex high-dimensional signals such as text, image and audio.

Recent works on MI estimation have focused on developing tight asymptotic variational MI lower bounds where neural networks are used for signal modeling.

The IM algorithm (Agakov, 2004) introduces a variational MI lower bound, where a neural network q(z|x) is learned as a variational approximation to the conditional distribution P (Z|X).

The IM algorithm requires the entropy, H(Z), and E XZ log q(z|x) to be tractable, which applies to latent codes of Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) as well as categorical variables.

Belghazi et al. (2018) introduces MI lower bounds MINE and MINE-f which allow the modeling of general random variables and shows improved accuracy for high-dimensional random variables, with application to improving generative models.

Poole et al. (2018) introduces a spectrum of energy-based MI estimators based on MINE and MINE-f lower bounds and a new TCPC estimator inspired by Contrastive Predictive Coding for the case when multiple samples from P (Z|X) can be drawn.

Our work introduces cross-validation to the MINE-f estimator.

We derive the lower bound of MINE-f under limited number of samples, and introduce meta-learning and hyperparameter search to enable practical statistical dependency testing.

Existing works in general statistical dependency testing (Bach & Jordan, 2002; Gretton et al., 2005a; Berrett & Samworth, 2019) have developed non-parametric independent criterions based on correlation and mutual information estimators equivalent to testing I(X; Z) = 0, followed by detailed bias and variance analyses.

Our approach for independent testing suggest a different direction by harnessing the generalization power of neural networks and may improve test performance on complex signals.

The p-values provided by our test do not involve approximated distributions and hold for small number of examples and arbitrary number of signal dimensions.

As different statistical dependency testing approaches have explicit or implicit assumptions and biases that make them suitable in different situations, a fair comparison across different approaches is a challenging task.

Instead, we focus on a self-contained presentation of our dependency test, and provide preliminary comparisons with a widely studied Hilbert-Schmidt independence criterion (HSIC) (Gretton et al., 2005a) in the appendix.

Meta-learning, or "learning to learn", seeks to improve the generalization capability of neural networks by searching for better hyperparameters (Maclaurin et al., 2015) , network architectures (Pham et al., 2018) , initialization (Finn et al., 2017a; 2018; Kim et al., 2018) and distance metrics Snell et al., 2017) .

Meta-learning approaches have shown significant performance improvements in applications such as automatic neural architecture search (Pham et al., 2018) , few-shot image recognition (Finn et al., 2017a) and imitation learning (Finn et al., 2017b) .

In particular, our estimator benefits from the Model-Agnostic Meta-Learning (MAML) (Finn et al., 2017a) framework which is designed to improve few-shot learning performance.

A network initialization is learned to maximize its performance when fine-tuned on few-shot learning tasks.

Applications include few-shot image classification and navigation.

We leverage the model-agnostic nature of MAML for MI estimation between generic random variable and adopt MAML for maximizing MI lower bounds.

To construct a collection of diverse tasks for MAML learning from limited samples, inspired by MI's invariance to invertible transformations, we propose a task-augmentation protocol to automatically construct tasks by sampling random transformations to transform the samples.

Results show reduced overfitting and improved generalization.

In this section, we will provide the background necessary to understand our approach 1 .

We define X and Z to be two random variables, P (X, Z) is the joint distribution, and P (X) and P (Z) are the marginal distributions over X and Z respectively.

Our goal is to estimate MI, I(X; Z) given independent and identically distributed (i.i.d.) sample pairs (x i , z i ), i = 1, 2 . . .

n from P (X, Z).

Let F = {T θ (x, z)} θ∈Θ be a class of scalar functions, where θ is the set of model parameters.

Let x,z) .

Results from previous works (Belghazi et al., 2018; Poole et al., 2018) show that the following energy-based family of lower bounds of MI hold for any θ:

where, E is the expectation over the given distribution.

Based on I MINE , the MINE estimator I(X, Z) n is defined as in Eq.2.

Estimators for I EB1 , I MINE-f and I EB can be defined similarly.

With infinite samples to approximate expectation, Eq.2 converges to the lower bound I(X, Z) ∞ = sup θ∈Θ I MINE .

Note that the number of samples n needs to be substantially more than the number of model parameters d = |θ| to guarantee that T θ (X, Y ) does not overfit to the samples (x i , z i ), i = 1, 2 . . .

n and overestimate MI.

Formally, the sample complexity of MINE is defined as the minimum number of samples n in order to achieve Eq.3,

Specifically, MINE proves that under the following assumptions: 1)

. .

, d}, the sample complexity of MINE is given by Eq.4.

For example, a neural network with dimension d = 10, 000, M = 1, K = 0.1 and L = 1, achieving a confidence interval of = 0.1 with 95% confidence (δ = 0.05) would require n ≥ 18, 756, 256 samples.

This is achievable for synthetic example generated by GANs like that studied in Belghazi et al. (2018) .

For real data, however, the cost of data acquisition for reaching statistically significant estimation can be prohibitively expensive.

Our approach instead uses the MI lower bounds specified in Eq.1 from a prediction perspective, inspired by cross-validation.

Our estimator, DEMINE, improves sample complexity by disentangling data for lower bound estimation from data for learning a generalizable T θ (X, Z).

DEMINE enables high-confidence MI estimation on small datasets.

4 APPROACH Section 4.1 specifies DEMINE for predictive MI estimation and derives the confidence interval; Section 4.2 formulates Meta-DEMINE, explains task augmentation, and defines the optimization algorithms.

In DEMINE, we interpret the estimation of MINE-f lower bound 2 Eq.1 as a learning problem.

The goal is given a limited number of samples, infer the optimal network T θ * (X, Z) with parameters θ * defined as follows:

Specifically, samples from P (X, Z) are subdivided into a training set {(

The training set is used for learning a networkθ as an approximation to θ * whereas the validation set is used for computing the DEMINE estimation I(X, Z) n,θ defined as in Eq.5.

We propose an approach to learnθ, DEMINE.

DEMINE learnsθ by maximizing the MI lower bound on the training set as follows:

The DEMINE algorithm is shown in Algorithm 2 in appendix.

Sample complexity analysis.

Becauseθ is learned independently of validation samples {(x i , z i ) val , i = 1, . . .

, n}, the sample complexity of the DEMINE estimator does not involve the model class F and the sample complexity is greatly reduced compared to MINE-f.

DEMINE estimates I(X, Z) ∞,θ when infinite number of samples are provided, defined as:

We now derive the sample complexity of DEMINE defined as the number of samples n required for I(X, Z) n,θ to be a good approximation to I(X, Z) ∞,θ in Theorem 1.

given any accuracy and confidence δ, we have:

when the number of validation samples n satisfies:

2 MINE lower bound can also be interpreted in the predictive way, but will result in a higher sample complexity than MINE-f lower bound.

We choose MINE-f in favor of a lower sample complexity over bound tightness.

Proof.

Since Tθ(X, Z) is bounded by [L, U ], applying the Hoeffding inequality to the first half of Eq.5 yields:

, applying the Hoeffding inequality twice to the second half of Eq.5:

Combining the above bounds results in:

By solving ξ to minimize n according to Eq.8 we have:

Theorem 1 also implies the following MI lower confidence interval under limited number of samples

Compared to MINE, as per the example shown in Section 3, for M = 1 (i.e. L = −1 and U = 1), δ = 0.05, = 0.1, our estimator requires n = 10, 742 compared to MINE requiring n = 18, 756, 256 i.i.d validation samples to estimate a lower bound, which makes MI-based dependency analysis feasible for domains where data collection is prohibitively expensive, e.g. fMRI scans.

In practice, sample complexity can be further optimized by optimizing hyperparameters U and L.

Note that unlike Eq.3, Theorem 1 bounds the closeness of the DEMINE estimate, I(X, Z) n,θ , not towards the MI lower bound sup θ∈Θ I MINE-f , but towards the MI lower bound I(X, Z) ∞,θ .

Therefore, the sample complexity of DEMINE as in Eq.8 makes fair comparison with the sample complexity of MINE as in Eq.4.

MINE's higher sample complexity stems from the need to bound the generalization error of T θ (X, Z) on unseen {(x, z)}. Existing generalization bounds are known to be overly loose, as over-parameterized neural networks have been shown to generalize well in classification and regression tasks (Zhang et al., 2016) .

By using a learning-based formulation, DEMINE not only avoids the need to bound generalization error, but also allows further generalization improvements by learningθ through meta-learning.

In the following section, we present a meta-learning formulation, Meta-DEMINE, that learnsθ for generalization given the same model class and training samples.

Given training data {(x i , z i ) train , i = 1, . . .

m}, Meta-DEMINE first generates MI estimation tasks each consisting of a meta-training split A and a meta-val split B through a novel task augmentation process.

And then a parameter initialization θ init is then learned to maximize MI estimation performance on the generated tasks using initialization θ init as shown in Eq.9.

Here

is the meta-training process of starting from an initialization θ (0) and applying Stochastic Gradient Descent (SGD) 3 over t steps to learn θ where in every meta training iteration we have:

Finally,θ is learned using the entire training set {(x i , z i ) train , i = 1, . . .

, m} with θ init as initialization:θ = MetaTrain (x, z) train , θ init .

Task Augmentation: Meta-DEMINE adapts MAML (Finn et al., 2017a) for MI lower bound maximization.

MAML has been shown to improve generalization performance in N -class K-shot image classification.

MI estimation, however, does not come with predefined classes and tasks.

A naive approach to produce tasks would be through cross-validation -partitioning training data into meta-training and meta-validation splits.

However, merely using cross-validation tasks is prone to overfitting -a θ init , which memorizes all training samples would as a result have memorized all metavalidation splits.

Instead, Meta-DEMINE generates tasks by augmenting the cross-validation tasks through task augmentation.

Training samples are first split into meta-training and meta-validation splits, and then transformed using the same random invertible transformation to increase task diversity.

Meta-DEMINE generates invertible transformation by sequentially composing the following functions:

Since the MI between two random variables is invariant to invertible transformations on each variable, MetaTrain(·, ·) is expected to arrive at the same MI lower bound estimation regardless of the transformation applied.

At the same time, memorization is greatly suppressed, as the same pair

More sophisticated invertible transformations (affine, piece-wise linear) can also be added.

Task augmentation is an orthogonal approach to data augmentation.

Using image classification as an example, data augmentation generates variations of the image, translated, or rotated images assuming that they are valid examples of the class.

Task augmentation on the other hand, does not make such an assumption.

Task augmentation requires the initial parameters θ init to be capable of recognizing the same class in a world where all images are translated and/or rotated, with the assumption that the optimal initialization should easily adapt to both the upright world and the translated and/or rotated world.

Optimization: Solving θ init using the meta-learning formulation Eq.9 poses a challenging optimization problem.

The commonly used approach is back propagation through time (BPTT) which computes second order gradients and directly back propagates gradients from MetaTrain((x, z) A , θ (0) ) to θ init .

BPTT is very effective for a small number of optimization steps, but is vulnerable to exploding gradients and is memory intensive.

In addition to BPTT, we find that stochastic finite difference algorithms such as Evolution Strategies (ES) (Salimans et al., 2017) and Parameter-Exploring Policy Gradients (PEPG) (Sehnke et al., 2010) can sometimes improve optimization robustness.

In practice, we switch betwen BPTT and PEPG depending on the number of meta-training iterations.

Meta-DEMINE algorithm is specified in Algorithm 1.

Dataset.

We evaluate our approaches DEMINE and Meta-DEMINE against baselines and state-ofthe-art approaches on 3 synthetic datasets: 1D Gaussian, 20D Gaussian and sine wave.

For 1D and 20D Gaussian datasets, following Belghazi et al. (2018) , we define two k-dimensional multivariate Gaussian random variables X and Z which have component-wise correlation corr(X i , Z j ) = δ ij ρ, where ρ ∈ (−1, 1) and δ ij is Kronecker's delta.

Mutual information I(X; Z) has a closed form solution I(X; Z) = −k ln(1 − ρ 2 ).

For the sine wave dataset, we define two random variables X and Z, where X ∼ U(−1, 1), Z = sin(aX + π 2 ) + 0.05 , and ∼ N (0, 1).

Estimating mutual information accurately given few pairs of (X, Z) requires the ability to extrapolate the sine wave given few examples.

Ground truth MI for sine wave dataset is approximated by running the the KSG Estimator (Kraskov et al., 2004) on 1, 000, 000 samples.

Implementation.

We compare our estimators, DEMINE and Meta-DEMINE, against the KSG estimator (Kraskov et al., 2004) MI-KSG and MINE-f (Belghazi et al., 2018) .

For both DEMINE and Meta-DEMINE, we study variance reduction mode, referred to as -vr, where hyperparameters are selected by optimizing 95% confident estimation mean (µ − 2σ µ ) and statistical significance mode, referred to as -sig, where hyperparameters are selected by optimizing 95% confident MI Transformation R x for x, R x (·) = m(P(O(G(·)))) 6:

Sample a batch of (x, z) B ∼ (x, z) A 10:

Compute ∇ θ0 L meta -gradient to θ init using BPTT Sample a batch of (x, z) B ∼ (x, z) train 22:

Compute gradient ∇ θ L

Update θ using Adam with η 25: end for

lower bound (µ − ).

Samples (x, z) are split 50%-50% into (x, z) train and (x, z) val .

We use a separable network architecture T θ (x, z) = M tanh(w cos f (x), g(z) + b) − t .

f and g are MLP encoders that embed signals x and z into vector embeddings.

Hyperparameters t ∈ [−1, 1] and M control upper and lower bounds T θ (x, z) ∈ [−M (1 + t), M (1 − t)].

Parameters w and b are learnable parameters.

MLP design and optimization hyperparameters are selected using Bayesian hyperparameter optimization (Bergstra et al., 2013) described below.

Hyperparameter search on DEMINE-vr and DEMINE-sig was conducted using the hyperopt pack- .

Mean µ and sample standard deviation σ of MI estiamte computed over 3-fold cross-validation on (x, z) train .

DEMINE-vr maximizes two sigma low µ−2σ µ where σ µ = 1 √ 3 σ due to 3-fold cross-validation.

DEMINE-sig maximizes statistical significance µ − where is two-sided 95% confidence interval of MI.

Meta-DEMINE-vr and Meta-DEMINEsig subsequently reuse these hyperparameters as DEMINE-vr and DEMINE-sig.

Meta-learning hyperparameters are chosen as outer loop N M = 3, 000 iterations, task augmentation N T = 1 iterations, r = 0.8, η meta = η 3 , with task augmentation mode m(P (O(·))).

N O was capped at 30 iterations for 1D and 20D Gaussian datasets due to memory limit.

For the sine wave datasets with large N O , we used PEPG (Sehnke et al., 2010) rather than BPTT.

For MI-KSG, we use off-the-shelf implementation by Gao et al. (2017) with default number of nearest neighbors k = 3.

MI-KSG does not provide any confidence interval.

For MINE-f, we use the same network architecture same as DEMINE-vr.

we implement both the original formulation which optimizes T θ on (x, z) till convergence (10k iters), as well as our own implementation MINE-f-ES with early stopping, where optimization is stopped after the same number of iterations as DEMINEvr to control overfitting.

Results.

Figure 1(a) shows MI estimation performance on 20D Gaussian datasets with varying ρ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5} using N = 300 samples.

Results are averaged over 5 runs to compare estimator bias, variance and confidence.

Note that Meta-DEMINE-sig detects the highest p < 0.05 confidence MI, outperforming DEMINE-sig which is a close second.

Both detect p < 0.05 statistically significant dependency starting ρ = 0.3, whereas estimations of all other approaches are low confidence.

It shows that in contrary to common belief, estimating the variational lower bounds with high confidence can be challenging under limited data.

MINE-f estimates MI > 3.0 and MINE-f-ES estimates positive MI when ρ = 0, both due to overfitting, despite MINE-f-ES having the lowest empirical bias.

DEMINE variants have relatively high empirical bias but low variance due to tight upper and lower bound control, which provides a different angle to understand bias-variance trade off in MI estimation (Poole et al., 2018) .

Figure 1(b,c,d ) shows MI estimation performance on 1D, 20D Gaussian and sine wave datasets with fixed ρ = 0.8, 0.3 and a = 8π respectively, with varying N ∈ {30, 100, 300, 1000, 3000} number of samples.

More samples asymptotically improves empirical bias across all estimators.

As opposed to 1D Gaussian datasets which are well solved by N = 300 samples, higher-dimensional 20D Gaussian and higher-complexity sine wave datasets are much more challenging and are not solved using N = 3000 samples with a signal-agnostic MLP architecture.

DEMINE-sig and Meta-DEMINE-sig detect p < 0.05 statistically significant dependency on not only 1D and 20D Gaussian datasets where x and z have non-zero correlation, but also on the sine wave datasets where correlation between x and z is 0.

This means that DEMINE-sig and Meta-DEMINE-sig can be used for nonlinear dependency testing to complement linear correlation testing.

We study the effect of cross-validation meta-learning and task augmentation on 20D Gaussian with ρ = 0.3 and N = 300.

Figure 2 plots performance of Meta-DEMINE-vr over N M = 3000 meta iterations under combinations of task augmentations modes and number of adaptation iterations N O ∈ {0, 20}. Overall, task augmentation modes which involve axis flipping m(·) and permutation P (·) are the most successful.

With N O = 20 steps of adaptation, task augmentation modes P (·), m(P (·)) and m(P (O(·))) prevent overfitting and improves performance.

The performance improvements of task augmentation is not simply from change in batch size, learning rate or number of optimization iterations, because meta-learning without task augmentation for both N O = 0 and 20 could not outperform baseline.

Meta-learning without task augmentation and with task augmentation but using only O(·) or G(·) result in overfitting.

Task augmentation with m(·) or m(P (O(G(·)))) prevent overfitting, but do not provide performance benefits, possibly because their complexity is insufficient or excessive for 20 adaptation steps.

Further more, task augmentation with no adaptation (N O = 0) falls back to data augmentation, where samples from transformed distributions are directly used to learn T θ (x, z).

Data augmentation with O(·) outperforms no augmentation, but is unable to outperform baseline and suffers from overfitting.

It shows that task augmentation provides improvements orthogonal to data augmentation.

Humans use language to effectively transmit brain representations among conspecifics.

For example, after witnessing an event in the world, a speaker may use verbal communication to evoke neural representations reflecting that event in a listener's brain (Hasson et al., 2012) .

The efficacy of this transmission, in terms of listener comprehension, is predicted by speaker-listener neural synchrony and synchrony among listeners (Stephens et al., 2010) .

To date, most work has measured brainto-brain synchrony by locating statistically significant inter-subject correlation (ISC); quantified as the Pearson product-moment correlation coefficient between response time series for corresponding voxels or regions of interest (ROIs) across individuals (Hasson et al., 2004; Schippers et al., 2010; Silbert et al., 2014; Nastase et al., 2019) .

Using DEMINE and Meta-DEMINE for statistical dependency testing, we can extend ISC analysis to capture nonlinear and higher-order interactions in continuous fMRI responses.

Specifically, given synchronized fMRI response frames in two brain regions X and Z across K subjects X i , Z i , i = 1, . . . , K as random variables.

We model the conditional mutual information I(X i ; Z j |i = j) as the MI form of pair-wise ISC analysis.

By definition, I(X i ; Z j |i = j) first computes MI between activations X i and Z j from subjects i and j respectively, and then average across pairs of subjects i = j. It can be lower bounded using Eq. 7 by learning a T θ (x, z) shared across all subject pairs.

Dataset.

We study MI-based and correlation-based ISC on a fMRI story comprehension dataset by Nastase et al. (2019) with 40 participants listening to four spoken stories.

Average story duration is 11 minutes.

An fMRI frame with full brain coverage is captured at repetition time 1 TR =1.5 seconds with 2.5mm isotropic spatial resolution.

We restricted our analysis to subsets of voxels defined using independent data from previous studies: functionally-defined masks of high ISC voxels (ISC; 3,800 voxels) and dorsal Default-Mode Network voxels (dDMN; 3,940 voxels) from Simony et al. (2016) as well as 180 HCP-MMP1 multimodal cortex parcels from Glasser et al. (2016) .

All masks were defined in MNI space.

Implementation.

We compare MI-based ISC using DEMINE and Meta-DEMINE with correlationbased ISC using Pearson's correlation.

DEMINE and Meta-DEMINE setup follows Section 5.

The fMRI data were partitioned by subject into a train set of 20 subjects and a validation set of 20 different subjects.

Residual 1D CNN is used instead of MLP as the encoder for studying temporal dependency.

For Pearson's correlation, high-dimensional signals are reshaped to 1D for correlation analysis.

Effective sample size for confidence interval calculation is the number of unique nonoverlapping fMRI samples.

Results.

We first examine, for the fine grained HCM-MMP1 brain regions, which have p < 0.05 statistically significant MI and Pearson's correlation.

Table 1 shows the result.

Overall, more regions have statistically significant correlation than dependency.

This is expected because correlation requires less data to detect.

But Meta-DEMINE is able to find 6 brain regions that have statistically significant dependency but lacks significant correlation.

This shows that MI analysis can be used to complement correlation-based ISC analysis.

By considering temporal ISC over time, fMRI signals can be modeled with improved accuracy.

In Table 2 we apply DEMINE and Meta-DEMINE with L = 10TRs (15s) sliding windows as random variables to study amount of information that can be extracted from ISC and dDMN masks.

We use between-subject time-segment classification (BSC) for evaluation (Haxby et al., 2011; Guntupalli et al., 2016) .

Each fMRI scan is divided into K non-overlapping L = 10 TRs time segments.

The BSC task is one versus rest retrieval: retrieve the corresponding time segment z of an individual given a group of time segments x excluding that individual, measured by top-1 accuracy.

For retrieval score, T θ (X, Z) is used for DEMINE and Meta-DEMINE and ρ(X, Z) is used for Pearson's correlation as a simple baseline.

With CNN as encoder, DEMINE and Meta-DEMINE model the signal better and achieve higher accuracy.

Also.

Meta-DEMINE is able to extract 0.75 nats of MI from the ISC mask over 10 TRs or 15s, which could potentially be improved by more samples.

We illustrated that a predictive view of the MI lower bounds coupled with meta-learning results in data-efficient variational MI estimators, DEMINE and Meta-DEMINE, that are capable of performing statistical test of dependency.

We also showed that our proposed task augmentation reduces overfitting and improves generalization in meta-learning.

We successfully applied MI estimation to real world, data-scarce, fMRI datasets.

Our results suggest a greater avenue of using neural networks and meta-learning to improve MI analysis and applying neural network-based information theory tools to enhance the analysis of information processing in the brain.

Model-agnostic, high-confidence, MI lower bound estimation approaches -including MINE, DEMINE and Meta-DEMINE-are limited to estimating small MI lower bounds up to O(log N ) as pointed out in (McAllester & Statos, 2018) , where N is the number of samples.

In real fMRI datasets, however, strong dependency is rare and existing MI estimation tools are limited more by their ability to accurately characterize the dependency.

Nevertheless, when quantitatively measuring strong dependency, cross-entropy (McAllester & Statos, 2018) Sample a batch of (

Update θ (i) using Adam (Kingma & Ba, 2014) with η 7: end for

The dataset we used contains 40 participants (mean age = 23.3 years, standard deviation = 8.9, range: 1853; 27 female) recruited to listen to four spoken stories 56 .

The stories were renditions of "Pie Man" and "Running from the Bronx" by Jim OGrady (O'Grady, 2018b;a), "The Man Who Forgot Ray Bradbury" by Neil Gaiman (Gaiman, 2018) , and "I Knew You Were Black" by Carol Daniel (Daniel, 2018) ; story durations were 7, 9, 14, and 13 minutes, respectively.

After scanning, participants completed a questionnaire comprising 25-30 questions per story intended to measure narrative comprehension.

The questionnaires included multiple choice, True/False, and fill-in-theblank questions, as well as four additional subjective ratings per story.

Functional and structural images were acquired using a 3T Siemens Prisma with a 64-channel head coil.

Briefly, functional images were acquired in an interleaved fashion using gradient-echo echo-planar imaging with a multiband acceleration factor of 3 (TR/TE = 1500/31 ms where TE stands for "echo time", resolution = 2.5 mm isotropic voxels, full brain coverage).

All fMRI data were formatted according to the Brain Imaging Data Structure (BIDS) standard (Gorgolewski et al., 2016) and preprocessed using the fMRIPrep library (Esteban et al., 2018) .

Functional data were corrected for slice timing, head motion, and susceptibility distortion, and normalized to MNI space using nonlinear registration.

Nuisance variables comprising head motion parameters, framewise displacement, linear and quadratic trends, sine/cosine bases for high-pass filtering (0.007 Hz), and six principal component time series from cerebrospinal fluid (CSF) and white matter (WM) were regressed out of the signal using the Analysis of Functional NeuroImages (AFNI) software suite (Cox, 1996) .

The fMRI data comprise X ∈ R Vi×T for each subject, where V i represents the flattened and masked voxel space and T represents the number of samples (in TRs) during auditory stimulus presentation.

Additional Details on Dataset Collection Functional and structural images were acquired using a 3T Siemens Magnetom Prisma with a 64-channel head coil.

Functional, blood-oxygenation-leveldependent (BOLD) images were acquired in an interleaved fashion using gradient-echo echo-planar imaging with pre-scan normalization, fat suppression, a multiband acceleration factor of 3, and no in-plane acceleration: TR/TE = 1500/31 ms, flip angle = 67

• , bandwidth = 2480 hz per pixel, resolution = 2.5 mm 3 isotropic voxels, matrix size = 96 x 96, Field of view (FoV) = 240 x 240 mm, 48 axial slices with roughly full brain coverage and no gap, anteriorposterior phase encoding.

At the beginning of each scanning session, a T1-weighted structural scan (where T1 stands for "longitudinal relaxation time"), was acquired using a high-resolution single-shot MagnetizationPrepared 180 degrees radio-frequency pulses and RApid Gradient-Echo (MPRAGE) sequence with an in-plane acceleration factor of 2 using GeneRalized Autocalibrating Partial Parallel Acquisition (GRAPPA): TR/TE/TI = 2530/3.3/1100 ms where TI stands for inversion time, flip angle = 7

• , resolution = 1.0 x 1.0 x 1.0 mm voxels, matrix size = 256 x 256, FoV = 256 x 256 x 176 mm, 176 sagittal slices, ascending acquisition, anteriorposterior phase encoding, no fat suppression, 5 min 53 s total acquisition time.

At the end of each scanning session a T2-weighted (where T2 stands for "transverse relaxation time") structural scan was acquired using the same acquisition parameters and geometry as the T1-weighted structural image: TR/TE = 3200/428 ms, 4 minutes 40 seconds total acquisition time.

A field map was acquired at the beginning of each scanning session, but was not used in subsequent analyses.

Additional Details on Dataset Preprocessing Preprocessing was performed using the fMRIPrep library 7 Esteban et al. (2018) , a Nipype library 8 (Gorgolewski et al., 2011) based tool.

T1-weighted images were corrected for intensity non-uniformity using the N4 bias field correction algorithm (Tustison et al., 2010) and skull-stripped using Advanced Normalization Tools (ANTs) (Avants et al., 2008) .

Nonlinear spatial normalization to the International Consortium for Brain Mapping (ICBM) 152 Nonlinear Asymmetrical template version 2009c (Fonov et al., 2009 ) was performed using ANTs.

Brain tissue segmentation cerebrospinal fluid, white matter, and gray matter was was performed using FSL library's 9 FAST tool Zhang et al. (2001) .

Functional images were slice timing corrected using AFNI software's 3dTshift (Cox, 1996) and corrected for head motion using FSL library's MCFLIRT tool (Jenkinson et al., 2002) .

"Fieldmap-less" distortion correction was performed by co-registering each subject's functional image to that subject's intensity-inverted T1-weighted image (Wang et al., 2017) constrained with an average field map template (Treiber et al., 2016) .

This was followed by co-registration to the corresponding T1-weighted image using FreeSurfer software's 10 boundary-based registration (Greve & Fischl, 2009 ) with 9 degrees of freedom.

Motion correcting transformations, field distortion correcting warp, BOLD-to-T1 transformation and T1-to-template (MNI) warp were concatenated and applied in a single step with Lanczos interpolation using ANTs.

Physiological noise regressors were extracted applying "a Component Based Noise Correction Method" aCompCor (Behzadi et al., 2007) .

Six principal component time series were calculated within the intersection of the subcortical mask and the union of CSF and WM masks calculated in T1w (T1 weighted) space, after their projection to the native space of each functional run.

Framewise displacement (Power et al., 2014) was calculated for each functional run.

Functional images were downsampled to 3 mm resolution.

Nuisance variables comprising six head motion parameters (and their derivatives), framewise displacement, linear and quadratic trends, sine/cosine bases for high-pass filtering (0.007 Hz cutoff), and six principal component time series from an anatomically-defined mask of cerebrospinal fluid and white matter were regressed out of the signal using AFNI's 3dTproject (Cox, 1996) .

Functional response time series were z-scored for each voxel.

We first review the Hilbert-Schmidt independence criterion (HSIC), a widely-studied correlationbased independence criterion and discuss its connections with the MINE family of mutual information lower bound methods, and then study DEMINE and a spectral HSIC implementation on the synthetic datasets.

The HSIC approach (Gretton et al., 2005b; a) is based on a necessary and sufficient condition of independence: two random variables X and Z are independent if and only if for all bounded or positive functions f and g

A proof can be constructed by showing equivalence to the definition of independence, P (X, Z) = P (X)P (Z).

To construct an independence test, existing approaches (Gretton et al., 2005b; a) use Reproducing Kernel Hilbert Spaces (RKHS) for f and g, a function space that not only covers all functions between [0, 1], but also allows computationally efficient estimation or bounding of COCO(X, Z) = sup f,g E XZ f (X)g(Z) − E X f (X)E Z g(Z)

given samples, and test COCO(X, Z) = 0.

Confidence intervals are derived through McDiarmid's inequality, or using closed-form distributions to approximate the test statistics to a certain order of moments, and compute the confidence interval from the closed-form distribution.

The COCO(X, Z) used by HSIC estimators bears great resemblance to the MINE family of mutual information estimators.

In fact, it can be shown that COCO(X, Z) = sup f,g E (x,z)∼P XZ f (x)g(z) − Ex∼P X f (x)EZ∼P Z g(z)

= sup f,g E (x,z)∼P XZ f (x)g(z) − Ex∼P X ,z∼P Z log e f (x)g(z)

≥ sup f,g E (x,z)∼P XZ f (x)g(z) − Ex∼P X log Ez∼P Z e f (x)g(z) ≈ IEB1 ≥ sup f,g E (x,z)∼P XZ f (x)g(z) − log Ex∼P X ,z∼P Z e f (x)g(z) ≈ IMINE ≥ sup f,g E (x,z)∼P XZ f (x)g(z) − Ex∼P X ,z∼P Z e f (x)g(z) + 1 ≈ IMINE-f,IEB

It means that within a family of decomposable functions where T θ (X, Z) = f (X)g(Z), COCO(X,Z) is an upperbound to the MINE estimates.

In addition, the equivalence of COCO(X,Z) = 0 and I(X, Z) = 0 seems to suggest a form of mutual information bound.

On the other hand, MINE allows the use of non-decomposable T θ (X, Z).

Existing results on MINE (Poole et al., 2018) seem to suggest that a non-decomposable T θ (X, Z) gives superior empirical mutual information estimation performance over a decomposable T θ (X, Z).

The necessity of non-decomposable T θ (X, Z) designs and mutual information lower bounds under decomposable designs of T θ (X, Z) may be subjects of further research.

Similar to the MINE estimators, HSIC-based estimators tend to have loose confidence intervals due to the need to bound generalization error of kernels f and g on unseen data points.

We expect a cross-validation-based approach like DEMINE to also improve the performance of the HSIC-based estimators.

Comparison between DEMINE and HSIC on synthetic benchmarks.

We compare Canonical Correlation Analysis (CCA), DEMINE, DEMINE-meta and HSIC for independent testing on our 4 synthetic Gaussian and sine wave benchmarks presented in Section 5.

Results for a single random seed is reported for a compact presentation, but we have ran experiments using multiple random seeds and find the result of a single random seed representative enough.

For CCA, we compute p-value using the χ 2 test.

For HSIC, we report p-value using a publicly available implementation for a spectral HSIC test (Zhang et al., 2018) 11 .

The default kernel is used.

Hyperparameters are set to recommended setting when available.

For DEMINE and DEMINE-meta, the setup is identical to Section 5.

A 2-sided 95% confidence interval is reported, but showing only the lower side.

Experiment results are compiled in Table 3 .

Statistically significant dependence detections with p < 0.05 are bolded.

Results show that spectral HSIC requires less data to test dependency for the simple Gaussians dataset.

But on the more challenging sine wave dataset, DEMINE-sig and DEMINEmeta-sig perform better.

Overall, we find DEMINE more complementary to linear correlations for dependency testing on complex signals.

Note that Gaussian kernels are used for spectral HSIC.

More complex kernels have potential to improve results.

We performed sanity check of our approach, as well as several statistical dependency testing implementations that we compare against.

We run different statistical dependency testing implementations on our 1D Gaussian ρ = 0.0, N = 30 samples dataset where X and Z are independent.

A large number of runs with different random seeds are performed.

False positive rate of p < 0.05 statistical

@highlight

A new & practical statistical test of dependency using neural networks, benchmarked on synthetic and a real fMRI datasets.

@highlight

Proposes a neural-network-based estimation of mutal information which can reliably work with small datasets, reducing the sample complexity by decoupling the network learning problem and the estimation problem.