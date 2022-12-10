Detecting the emergence of abrupt property changes in time series is a challenging problem.

Kernel two-sample test has been studied for this task which makes fewer assumptions on the distributions than traditional parametric approaches.

However, selecting kernels is non-trivial in practice.

Although kernel selection for the two-sample test has been studied, the insufficient samples in change point detection problem hinder the success of those developed kernel selection algorithms.

In this paper, we propose KL-CPD, a novel kernel learning framework for time series CPD that optimizes a lower bound of test power via an auxiliary generative model.

With deep kernel parameterization, KL-CPD endows kernel two-sample test with the data-driven kernel to detect different types of change-points in real-world applications.

The proposed approach significantly outperformed other state-of-the-art methods in our comparative evaluation of benchmark datasets and simulation studies.

Detecting changes in the temporal evolution of a system (biological, physical, mechanical, etc.) in time series analysis has attracted considerable attention in machine learning and data mining for decades BID3 BID7 .

This task, commonly referred to as change-point detection (CPD) or anomaly detection in the literature, aims to predict significant changing points in a temporal sequence of observations.

CPD has a broad range of real-world applications such as medical diagnostics BID12 , industrial quality control BID4 , financial market analysis BID31 , video anomaly detection ) and more.

Figure 1: A sliding window over the time series input with two intervals: the past and the current, where w l , w r are the size of the past and current interval, respectively.

X (l) , Xconsists of the data in the past and current interval, respectively.

As shown in Fig. 1 , we focus on the retrospective CPD BID36 BID23 , which allows a flexible time window to react on the change-points.

Retrospective CPD not only enjoys more robust detection BID9 ) but embraces many applications such as climate change detection BID32 , genetic sequence analysis BID37 , networks intrusion detection BID41 , to name just a few.

Various methods have been developed BID17 , and many of them are parametric with strong assumptions on the distributions BID3 BID16 , including auto-regressive models BID40 and state-space models BID20 for tracking changes in the mean, the variance, and the spectrum.

Ideally, the detection algorithm should be free of distributional assumptions to have robust performance as neither true data distributions nor anomaly types are known a priori.

Thus the parametric assumptions in many works are unavoidably a limiting factor in practice.

As an alternative, nonparametric and kernel approaches are free of distributional assumptions and hence enjoy the advantage to produce more robust performance over a broader class of data distributions.

Kernel two-sample test has been applied to time series CPD with some success.

For example, BID18 presented a test statistic based upon the maximum kernel fisher discriminant ratio for hypothesis testing and BID23 proposed a computational efficient test statistic based on maximum mean discrepancy with block sampling techniques.

The performance of kernel methods, nevertheless, relies heavily on the choice of kernels.

BID13 BID14 conducted kernel selection for RBF kernel bandwidths via median heuristic.

While this is certainly straightforward, it has no guarantees of optimality regarding to the statistical test power of hypothesis testing.

BID15 show explicitly optimizing the test power leads to better kernel choice for hypothesis testing under mild conditions.

Kernel selection by optimizing the test power, however, is not directly applicable for time series CPD due to insufficient samples, as we discuss in Section 3.In this paper, we propose KL-CPD, a kernel learning framework for time series CPD.

Our main contributions are three folds.• In Section 3, we first observe the inaptness of existing kernel learning approaches in a simulated example.

We then propose to optimize a lower bound of the test power via an auxiliary generative model, which aims at serving as a surrogate of the abnormal events.• In Section 4, we present a deep kernel parametrization of our framework, which endows a data-driven kernel for the kernel two-sample test.

KL-CPD induces composition kernels by combining RNNs and RBF kernels that are suitable for the time series applications.• In Section 5, we conduct extensive benchmark evaluation showing the outstanding performance of KL-CPD in real-world CPD applications.

With simulation-based analysis in Section 6, in addition, we can see the proposed method not only boosts the kernel power but also evades the performance degradation as data dimensionality of time series increases.

Finally, our experiment code and datasets are available at https://github.com/ OctoberChang/klcpd_code.

Given a sequence of d-dimensional observations {x 1 , . . .

, x t , . . .}, x i ∈ R d , our goal is to detect the existence of a change-point 1 such that before the change-point, samples are i.i.d from a distribution P, while after the change-point, samples are i.i.d from a different distribution Q. Suppose at current time t and the window size w, denote the past window segment X (l) = {x t−w , . . .

, x t−1 } and the current window segment X (r) = {x t , . . . , x t+w−1 }, We compute the maximum mean discrepancy (MMD) between X (l) and X (r) , and use it as the plausibility of change-points:

The higher the distribution discrepancy, the more likely the point is a change-point.

Notice that there are multiple settings for change point detection (CPD) where samples could be piecewise iid, non-iid autoregressive, and more.

It is truly difficult to come up with a generic framework to tackle all these different settings.

In this paper, following the previous CPD works ( BID18 BID20 BID27 BID23 , we stay with the piecewise iid assumption of the time series samples.

Extending the current model to other settings is interesting and we leave it for future work.

We review maximum mean discrepancy (MMD) and its use to two-sample test, which are two cornerstones in this work.

Let k be the kernel of a reproducing kernel Hilbert space (RKHS) H k of functions on a set X .

We assume that k is measurable and bounded, sup x∈X k(x, x) < ∞. MMD is a nonparametric probabilistic distance commonly used in two-sample-test BID13 BID14 .

Given a kernel k, the MMD distance between two distributions P and Q is defined as DISPLAYFORM0 where DISPLAYFORM1 are the kernel mean embedding for P and Q, respectively.

In practice we use finite samples from distributions to estimate MMD distance.

Given X = {x 1 , . . .

, x m } ∼ P and Y = {y 1 , . . . , y m } ∼ Q, one unbiased estimator of M k (P, Q) iŝ DISPLAYFORM2 which has nearly minimal variance among unbiased estimators (Gretton et al., 2012a, Lemma 6) .For any characteristic kernel k, M k (P, Q) is non-negative and in particular M k (P, Q) = 0 iff P = Q. However, the estimatorM k (X, X ) may not be 0 even though X, X ∼ P due to finite sample size.

Hypothesis test instead offers thorough statistical guarantees of whether two finite sample sets are the same distribution.

Following BID14 , the hypothesis test is defined by the null hypothesis H 0 : P = Q and alternative H 1 : P = Q, using test statistic mM k (X, Y ).

For a given allowable false rejection probability α (i.e., false positive rate or Type I error), we choose a test threshold c α and reject DISPLAYFORM3 We now describe the objective to choose the kernel k for maximizing the test power BID15 BID34 .

First, note that, under the alternative DISPLAYFORM4 where V m (P, Q) denotes the asymptotic variance of theM k estimator.

The test power is then DISPLAYFORM5 where Φ is the CDF of the standard normal distribution.

Given a set of kernels K, We aim to choose a kernel k ∈ K to maximize the test power, which is equivalent to maximizing the argument of Φ.

In time series CPD, we denote P as the distribution of usual events and Q as the distribution for the event when change-points happen.

The difficulty of choosing kernels via optimizing test power in Eq. (2) is that we have very limited samples from the abnormal distribution Q. Kernel learning in this case may easily overfit, leading to sub-optimal performance in time series CPD.

To demonstrate how limited samples of Q would affect optimizing test power, we consider kernel selection for Gaussian RBF kernels on the Blobs dataset BID15 BID34 , which is considered hard for kernel two-sample test.

P is a 5 × 5 grid of two-dimensional standard normals, with spacing 15 between the centers.

Q is laid out identically, but with covariance q −1 q +1 between the coordinates (so the ratio of eigenvalues in the variance is q ).

Left panel of Fig. 2 shows X ∼ P (red samples), Y ∼ Q (blue dense samples),Ỹ ∼ Q (blue sparse samples) with q = 6.

Note that when q = 1, P = Q.For q ∈ {4, 6, 8, 10, 12, 14}, we take 10000 samples for X, Y and 200 samples forỸ .

We consider two objectives for choosing kernels: 1) median heuristic; 2) max-ratio DISPLAYFORM0 ; among 20 kernel bandwidths.

We repeat this process 1000 times and report the test power under false rejection rate α = 0.05.

As shown in the right panel of Fig. 2 , optimizing kernels using limited samplesỸ significantly decreases the test power compared to Y (blue curve down to the cyan curve).

This result not only verifies our claim on the inaptness of existing kernel learning objectives for CPD task, but also stimulates us with the following question, How to optimize kernels with very limited samples from Q, even none in an extreme?Figure 2: Left: 5 × 5 Gaussian grid, samples from P, Q and G. We discuss two cases of Q, one of sufficient samples, the other of insufficient samples.

Right: Test power of kernel selection versus q .

Choosing kernels by γ k * (X, Z) using a surrogate distribution G is advantageous when we do not have sufficient samples from Q, which is typically the case in time series CPD task.

We first assume there exist a surrogate distribution G that we can easily draw samples from (Z ∼ G, |Z| |Ỹ |), and also satisfies the following property: DISPLAYFORM0 Besides, we assume dealing with non trivial case of P and Q where a lower bound DISPLAYFORM1 Just for now in the blob toy experiment, we artifact this distribution G by mimicking Q with the covariance g = q − 2.

We defer the discussion on how to find G in the later subsection 3.3.

Choosing kernels via γ k * (X, Z) using surrogate samples Z ∼ G, as represented by the green curve in Fig. 2 , substantially boosts the test power compared to η k * (X,Ỹ ) with sparse samplesỸ ∼ Q.

This toy example not only suggesets that optimizing kernel with surrogate distribution G leads to better test power when samples from Q are insufficient, but also demonstrates that the effectiveness of our kernel selection objective holds without introducing any autoregressive/RNN modeling to control the Type-I error.

Test Threshold Approximation Under H 0 : P = Q, mM k (X, Y ) converges asymptotically to a distribution that depends on the unknown data distribution P (Gretton et al., 2012a, Theorem 12); we thus cannot evaluate the test threshold c α in closed form.

Common ways of estimating threshold includes the permutation test and a estimated null distribution based on approximating the eigenspectrum of the kernel.

Nonetheless, both are still computational demanding in practice.

Even with the estimated threshold, it is difficult to optimize c α because it is a function of k and P.For X, X ∼ P, we know that c α is a function of the empirical estimatorM k (X, X ) that controls the Type I error.

BoundingM k (X, X ) could be an approximation of bounding c α .

Therefore, we propose the following objective that maximizing a lower bound of test power DISPLAYFORM2 where λ is a hyper-parameter to control the trade-off between Type-I and Type-II errors, as well as absorbing the constants m, v l , v u in variance approximation.

Note that in experiment, the optimization of Eq. FORMULA10 is solved using the unbiased estimator of M k (P, G) with empirical samples.

The remaining question is how to construct the surrogate distribution G without any sample from Q.Injecting random noise to P is a simple way to construct G. While straightforward, it may result in a sub-optimal G because of sensitivity to the level of injected random noise.

As no prior knowledge of Q, to ensure (3) hold for any possible Q (e.g. Q = P but Q ≈ P), intuitively, we have to make G as closed to P as possible.

We propose to learn an auxiliary generative model G θ parameterized by θ such thatM DISPLAYFORM0 To ensure the first inequality hold, we set early stopping criterion when solving G θ in practice.

Also, if P is sophisticate, which is common in time series cases, limited capacity of parametrization of G θ with finite size model (e.g. neural networks) BID2 and finite samples of P also hinder us to fully recover P. Therefore, we result in a min-max formulation to consider all possible k ∈ K when we learn G, min DISPLAYFORM1 and solve the kernel for the hypothesis test in the mean time.

In experiment, we use simple alternative (stochastic) gradient descent to solve each other.

Lastly, we remark that although the resulted objective (6) is similar to BID22 , the motivation and explanation are different.

One major difference is we aim to find k with highest test power while their goal is finding G θ to approximate P. A more detailed discussion can be found in Appendix A.

In this section, we present a realization of the kernel learning framework for time series CPD.Compositional Kernels To have a more expressive kernel for complex time series, we consider compositional kernelsk = k • f that combines RBF kernels k with injective functions f φ : DISPLAYFORM0 The resulted kernelk is still characteristic if f is an injective function and k is characteristic BID14 .

This ensures the MMD endowed byk is still a valid probabilistic distance.

One example function class is {f φ |f φ (x) = φx, φ > 0}, equivalent to the kernel bandwidth tuning.

Inspired by the recent success of combining deep neural networks into kernels BID38 BID0 BID22 , we parameterize the injective functions f φ by recurrent neural networks (RNNs) to capture the temporal dynamics of time series.

For an injective function f , there exists a function F such that F (f (x)) = x, ∀x ∈ X , which can be approximated by an auto-encoder via sequence-to-sequence architecture for time series.

One practical realization of f would be a RNN encoder parametrized by φ while the function F is a RNN decoder parametrized by ψ trained to minimize the reconstruction loss.

Thus, our final objective is DISPLAYFORM1 Practical Implementation In practice, we consider two consecutive windows in mini-batch to estimateM f φ X, X in an online fashion for the sake of efficiency.

Specifically, the sample X ∼ P is divided into the left window segment X (l) = {x t−w , . . .

, x t−1 } and the right window segment X (r) = {x t , . . . , x t+w−1 } such that X = {X (l) , X (r) }.

We now reveal implementation details of the auxiliary generative model and the deep kernel.

Generator g θ Instead of modeling the explicit density G θ , we model a generator g θ where we can draw samples from.

The goal of g θ is to generate plausibly counterfeit but natural samples based on historical X ∼ P, which is similar to the conditional GANs BID28 BID19 .

We use sequence-to-sequence (Seq2Seq) architectures BID35 where g θe encodes time series into hidden states, and g θ d decodes it with the distributional autoregressive process to approximate the surrogate sample Z: DISPLAYFORM2 where ω ∼ P(W ) is a d h -dimensional random noise sampled from a base distribution P(W ) (e.g., uniform, Gaussian).

H = [h t−w , . . . , h t−1 ] ∈ R d h ×w is a sequence of hidden states of the generator's encoder.

X (r) 1 = {0, x t , x t+1 , . . . , x t+w−2 } denotes right shift one unit operator over X (r) .Deep Kernel Parametrization We aim to maximize a lower bound of test power via backpropagation on φ using the deep kernel formk = k • f φ .

On the other hand, we can also view the deep kernel parametrization as an embedding learning on the injective function f φ (x) that can be distinguished by MMD.

Similar to the design of generator, the deep kernel is a Seq2Seq framework with one GRU layer of the follow form: DISPLAYFORM3 where ν ∼ P ∪ G θ are from either the time series data X or the generated sample Z ∼ g θ (ω|X).We present an realization of KL-CPD in Algorithm 1 with the weight-clipping technique.

The stopping condition is based on a maximum number of epochs or the detecting power of kernel MMD M f φ P, G θ ) ≤ .

This ensure the surrogate G θ is not too close to P, as motivated in Sec. 3.2.Algorithm 1: KL-CPD, our proposed algorithm.

input : α the learning rate, c the clipping parameter, w the window size, n c the number of iterations of deep kernels training per generator update.

DISPLAYFORM4 t }, and ω ∼ P(Ω) DISPLAYFORM5

The section presents a comparative evaluation of the proposed KL-CPD and seven representative baselines on benchmark datasets from real-world applications of CPD, including the domains of biology, environmental science, human activity sensing, and network traffic loads.

The data statistics are summarized in BID25 , the datasets are split into the training set (60%), validation set (20%) and test set (20%) in chronological order.

Note that training is fully unsupervised for all methods while labels in the validation set are used for hyperparameters tuning.

For quantitative evaluation, we consider receiver operating characteristic (ROC) curves of anomaly detection results, and measure the area-under-the-curve (AUC) as the evaluation metric.

AUC is commonly used in CPD literature BID23 BID25 BID39 Table 2 : AUC on four real-world datasets.

KL-CPD has the best AUC on three out of four datasets.

In Table 2 , the first four rows present the real-time CPD methods, followed by three retrospective-CPD models, and the last is our proposed method.

KL-CPD shows significant improvement over the other methods on all the datasets, except being in a second place on the Yahoo dataset, with 2% lower AUC compared to the leading ARGP.

This confirms the importance of data-driven kernel selection and effectiveness of our kernel learning framework.

Notice that OPT-MMD performs not so good compared to KL-CPD, which again verifies our simulated example in Sec. 3 that directly applying existing kernel learning approaches with insufficient samples may not be suitable for realworld CPD task.

Distribution matching approaches like RDR-KCPD and Mstats-KCPD are not as competitive as KL-CPD, and often inferior to real-time CPD methods.

One explanation is both RDR-KCPD and Mstats-KCPD measure the distribution distance in the original data space with simple kernel selection using the median heuristic.

The change-points may be hard to detect without the latent embedding learned by neural networks.

KL-CPD, instead, leverages RNN to extract useful contexts and encodes time series in a discriminative embedding (latent space) on which kernel two-sample test is used to detection changing points.

This also explains the inferior performance of Mstats-KCPD which uses kernel MMD with a fix RBF kernel.

That is, using a fixed kernel to detect versatile types of change points is likely to fail.

Finally, the non-iid temporal structure in real-world applications may raise readers concern that the improvement coming from adopting RNN and controlling type-I error for model selection (kernel selection).

Indeed, using RNN parameterized kernels (trained by minimizing reconstruction loss) buys us some gain compared to directly conduct kernel two-sample test on the original time series samples FIG0 cyan bar rises to blue bar).

Nevertheless, we still have to do model selection to decide the parameters of RNN.

In Table 2 , we studied a kernel learning baseline, OPT-MMD, that optimizing an RNN parameterized kernel by controlling type-I error without the surrogate distribution.

OPT-MMD is inferior to the KL-CPD that introduce the surrogate distribution with an auxiliary generator.

On the other hand, from Table 2 , we can also observe KL-CPD is better than other RNN alternatives, such as LSTNet.

Those performance gaps between KL-CPD, OPT-MMD (regularizing type-I only) and other RNN works indicate the proposed maximizing testing power framework via an auxiliary distribution serves as a good surrogate for kernel (model) selection.

We further examine how different encoders f φ affects KL-CPD.

For MMD-dataspace, f φ is an identity map, equivalent to kernel selection with median heuristic in data space.

For MMDcodespace, {f φ , F ψ } is a Seq2Seq autoencoder minimizing reconstruction loss without optimizing test power.

For MMD-negsample, the same objective as KL-CPD except for replacing the auxiliary generator with injecting Gaussian noise to P. The results are shown in FIG0 .

We first notice the mild improvement of MMD-codespace over MMD-dataspace, showing that using MMD on the induced latent space is effective for discovering beneficial kernels for time series CPD.

Next, we see MMD-negsample outperforms MMDcodespace, showing the advantages of injecting a random perturbation to the current interval to approximate g θ (z|X (l) ).

This also justify the validity of the proposed lower bound approach by optimizing M k (P, G), which is effective even if we adopt simple perturbed P as G. Finally, KL-CPD models the G with an auxiliary generator g θ to obtain conditional samples that are more complex and subtle than the perturbed samples in MMD-negsample, resulting in even better performance.

In FIG1 , we also demonstrate how the tolerance of delay w r influences the performance.

Due to space limit, results other than Bee-Dance dataset are omitted, given they share similar trends.

KL-CPD shows competitive AUC mostly, only slightly decreases when w r = 5.

MMD-dataspace and MMD-codespace, in contrast, AUC degradation is much severe under low tolerance of delay (w r = {5, 10}).

The conditional generated samples from KL-CPD can be found in Appendix B.5.

To further explore the performance of KL-CPD with controlled experiments, we follow other time series CPD papers BID36 BID25 BID27 to create three simulated datasets each with a representative change-point characteristic: jumping mean, scaling variance, and alternating between two mixtures of Gaussian (Gaussian-Mixtures).

More description of the generated process see Appendix B.2.

Jumping-Mean Scaling-Variance Gaussian-Mixtures The results are summarized in TAB2 .

KL-CPD achieves the best in all cases.

Interestingly, retrospective-CPD (ARGP-BOCPD, RDR-KCPD, Mstats-KCPD) have better results compared to real-time CPD (ARMA, ARGP, RNN,LSTNet), which is not the case in real-world datasets.

This suggests low reconstruction error does not necessarily lead to good CPD accuracies.

As for why Mstats-KCPD does not have comparable performance as KL-CPD, given that both of them use MMD as distribution distance?

Notice that Mstats-KCPD assumes the reference time series (training data) follows the same distribution as the current interval.

However, if the reference time series is highly non-stationary, it is more accurate to compute the distribution distance between the latest past window and the current window, which is the essence of KL-CPD.

We study how different encoders f φ would affect the power of MMD versus the dimensionality of data.

We generate an simulated time series dataset by sampling between two multivariate Gaussian N (0, σ FIG4 plots the one-dimension data and AUC results.

We see that all methods remain equally strong in low dimensions (d ≤ 10), while MMD-dataspace decreases significantly as data dimensionality increases (d ≥ 12).

An explanation is non-parametric statistical models require the sample size to grow exponentially with the dimensionality of data, which limits the performance of MMDdataspace because of the fixed sample size.

On the other hand, MMD-codespace and KL-CPD are conducting kernel two-sample test on a learned low dimension codespace, which moderately alleviates this issue.

Also, KL-CPD finds a better kernel (embedding) than MMD-codespace by optimizing the lower bound of the test power.

We propose KL-CPD, a new kernel learning framework for two-sample test by optimizing a lower bound of test power with a auxiliary generator, to resolve the issue of insufficient samples in changepoints detection.

The deep kernel parametrization of KL-CPD combines the latent space of RNNs with RBF kernels that effectively detect a variety of change-points from different real-world applications.

Extensive evaluation of our new approach along with strong baseline methods on benchmark datasets shows the outstanding performance of the proposed method in retrospective CPD.

With simulation analysis in addition we can see that the new method not only boosts the kernel power but also evades the performance degradation as data dimensionality increases.

A CONNECTION TO MMD GAN Although our proposed method KL-CPD has a similar objective function as appeared in MMD GAN BID22 , we would like to point out the underlying interpretation and motivations are radically different, as summarized below.

The first difference is the interpretation of inner maximization problem max k M k (P, G).

MMD GANs BID22 treat whole maximization problem max k M k (P, G) as a new probabilistic distance, which can also be viewed as an extension of integral probability metric (IPM).

The properties of the distance is also studied in BID22 ; .

A follow-up work by combining BID29 push max k M k (P, G) further to be a scaled distance with gradient norm.

However, the maximization problem (4) of this paper defines the lower bound of the test power, which also takes the variance of the empirical estimate into account, instead of the distance.

Regarding the goals, MMD GAN aims to learn a generative model that approximates the underlying data distribution P of interests.

All the works BID11 BID24 BID34 BID22 use MMD or max k M k (P, G) to define distance, then try to optimize G to be as closed to P as possible.

However, that is not the goal of this paper, where G is just an auxiliary generative model which needs to satisfies Eq. (3).

Instead, we aim to find the most powerful k for conducting hypothesis test.

In practice, we still optimize G toward P because we usually have no prior knowledge (sufficient samples) about Q, and we want to ensure the lower bound still hold for many possible Q (e.g. Q can be also similar to P).

However, even with this reason, we still adopt early stopping to prevent the auxiliary G from being exactly the same as P.

• Bee-Dance 2 records the pixel locations in x and y dimensions and angle differences of bee movements.

Ethologists are interested in the three-stages bee waggle dance and aim at identifying the change point from one stage to another, where different stages serve as the communication with other honey bees about the location of pollen and water.• Fishkiller 3 records water level from a dam in Canada.

When the dam not functions normally, the water level oscillates quickly in a particular pattern, causing trouble for the fish.

The beginning and end of every water oscillation (fish kills) are treated as change points.• HASC 4 is a subset of the Human Activity Sensing Consortium (HASC) challenge 2011 dataset, which provides human activity information collected by portable three-axis accelerometers.

The task of change point detection is to segment the time series data according to the 6 behaviors: stay, walk, jog, skip, stair up, and stair down.• Yahoo 5 contains time series representing the metrics of various Yahoo services (e.g. CPU utilization, memory, network traffic, etc) with manually labeled anomalies.

We select 15 out of 68 representative time series sequences after removing some sequences with duplicate patterns in anomalies.

• Jumping-Mean: Consider the 1-dimensional auto-regressive model to generate 5000 samples y(t) = 0.6y(t − 1) − 0.5y(t − 2) + t , where y(1) = y(2) = 0, t ∼ N (µ, 1.5) is a Gaussian noise with mean µ and standard deviation 1.5.

A change point is inserted at every DISPLAYFORM0 where τ ∼ N (0, 10) and n is a natural number such that 100(n − 1) + 1 ≤ t ≤ 100n.• Scaling-Variance: Same auto-regressive generative model as Jumping-Mean, but a change point is inserted at every 100 + τ time stamps by setting the noise standard deviation of t at time t as σ n = 1 n = 1, 3, . . .

, 49, ln(e + n 4 ) n = 2, 4, . . . , 48, where τ ∼ N (0, 10) and n is a natural number such that 100(n − 1) + 1 ≤ t ≤ 100n.• Gaussian-Mixtures: Time series data are sampled alternatively between two mixtures of Gaussian 0.5N (−1, 0.5 2 ) + 0.5N (1, 0.5 2 ) and 0.8N (−1, 1.0 2 ) + 0.2N (1, 0.1 2 ) for every 100 time stamps, which is defined as the change points.

We include the following representative baselines in the literature of time series forecasting and change-point detection for evaluations:• Autoregressive Moving Average (ARMA) BID6 is the classic statistical model that predicts the future time series based on an Autoregressive (AR) and a moving average (MA), where AR involves linear regression, while MA models the error term as a linear combination of errors in the past.• Autoregressive Gaussian Process (ARGP) BID8 ) is a Gaussian Process for time series forecasting.

In an ARGP of order p, x t−p:t−1 are taken as the GP input while the output is x t .

ARGP can be viewed as a non-linear version of AR model.• Recurrent Neural Networks (RNN) BID10 are powerful neural networks for learning non-linear temporal dynamical systems.

We consider gated recurrent units (GRU) in our implementation.• LSTNet BID21 ) is a recent state-of-the-art deep neural network fore time series forecasting.

LSTNet combines different architectures including CNN, RNN, residual networks, and highway networks.• ARGP-BOCPD BID33 is an extension of the Bayesian online change point detection (BOCPD) which uses ARGP instead of AR in underlying predictive models of BOCPD framework.• RDR-KCPD BID25 ) considers f-divergence as the dissimilarity measure.

The f-divergence is estimated by relative density ratio technique, which involves solving an unconstrained least-squares importance fitting problem.• Mstats-KCPD BID23 consider kernel maximum mean discrepancy (MMD) on data space as dissimilarity measure.

Specifically, It samples B block of segments from the past time series, and computes B times MMD distance between the past block with the current segment and takes the average as the dissimilarity measure.

For hyper-parameter tuning in ARMA, the time lag p, q are chosen from {1, 2, 3, 4, 5}. For ARGP and ARGP-BOCPD the time lag order p is set to the same as ARMA and the hyperparameter of kernel is learned by maximizing the marginalized likelihood.

For RDR-KCPD, the window size w are chosen from {25, 50}, sub-dim k = 5, α = {0.01, 0.1, 1}. For Mstats-KCPD and KL-CPD, the window size w = 25, and we use RBF kernel with median heuristic setting the kernel bandwidth.

The hidden dimension of GRU is d h = 10 for MMD-codespace, MMD-negsample and KL-CPD.

For KL-CPD, λ is chosen from {0.1, 1, 10} and β is chosen from {10 −3 , 10 −1 , 1, 10}.

@highlight

In this paper, we propose KL-CPD, a novel kernel learning framework for time series CPD that optimizes a lower bound of test power via an auxiliary generative model as a surrogate to the abnormal distribution. 

@highlight

Describes a novel approach to optimising the choice of kernel towards increased testing power and shown to offer improvements over alternatives.