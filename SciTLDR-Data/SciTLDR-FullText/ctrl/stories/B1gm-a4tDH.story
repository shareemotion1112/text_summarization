Ability to quantify and predict progression of a disease is fundamental for selecting an appropriate treatment.

Many clinical metrics cannot be acquired frequently either because of their cost (e.g. MRI, gait analysis) or because they are inconvenient or harmful to a patient (e.g. biopsy, x-ray).

In such scenarios, in order to estimate individual trajectories of disease progression, it is advantageous to leverage similarities between patients, i.e. the covariance of trajectories, and find a latent representation of progression.

Most of existing methods for estimating trajectories do not account for events in-between observations, what dramatically decreases their adequacy for clinical practice.

In this study, we develop a machine learning framework named Coordinatewise-Soft-Impute (CSI) for analyzing disease progression from sparse observations in the presence of confounding events.

CSI is guaranteed to converge to the global minimum of the corresponding optimization problem.

Experimental results also demonstrates the effectiveness of CSI using both simulated and real dataset.

The course of disease progression in individual patients is one of the biggest uncertainties in medical practice.

In an ideal world, accurate, continuous assessment of a patient's condition helps with prevention and treatment.

However, many medical tests are either harmful or inconvenient to perform frequently, and practitioners have to infer the development of disease from sparse, noisy observations.

In its simplest form, the problem of modeling disease progressions is to fit the curve of y(t), t ∈ [t min , t max ] for each patient, given sparse observations y := (ỹ(t 1 ), . . . ,ỹ(t n )).

Due to the highdimensional nature of longitudinal data, existing results usually restrict solutions to subspace of functions and utilize similarities between patients via enforcing low-rank structures.

One popular approach is the mixed effect models, including Gaussian process approaches (Verbeke, 1997; Zeger et al., 1988) and functional principal components (James et al., 2000) .

While generative models are commonly used and have nice theoretical properties, their result could be sensitive to the underlying distributional assumptions of observed data and hard to adapt to different applications.

Another line of research is to pose the problem of disease progression estimation as an optimization problem.

Kidzinski and Hastie.

Kidziński & Hastie (2018) proposed a framework which formulates the problem as a matrix completion problem and solve it using matrix factorization techniques.

This method is distribution-free and flexible to possible extensions.

Meanwhile, both types of solutions model the natural progression of disease using observations of the targeted variables only.

They fail to incorporate the existence and effect of human interference: medications, therapies, surgeries, etc.

Two patients with similar symptoms initially may have different futures if they choose different treatments.

Without that information, predictions can be way-off.

To the best of our knowledge, existing literature talks little about modeling treatment effect on disease progression.

In Kidziński & Hastie (2018) , authors use concurrent observations of auxillary variables (e.g. oxygen consumption to motor functions) to help estimate the target one, under the assumption that both variables reflect the intrinsic latent feature of the disease and are thus correlated.

Treatments of various types, however, rely on human decisions and to some extent, an exogenous variable to the development of disease.

Thus they need to modeled differently.

In this work, we propose a model for tracking disease progression that includes the effects of treatments.

We introduce the Coordinatewise-Soft-Impute (CSI) algorithm for fitting the model and investigate its theoretical and practical properties.

The contribution of our work is threefold: First, we propose a model and an algorithm CSI, to estimate the progression of disease which incorporates the effect of treatment events.

The framework is flexible, distribution-free, simple to implement and generalizable.

Second, we prove that CSI converges to the global solution regardless of the initialization.

Third, we compare the performance of CSI with various other existing methods on both simulated data and a dataset of Gillette Children's Hospital with patients diagnosed with Cerebral Palsy, and demonstrate the superior performances of CSI.

The rest of the paper is organized as follows.

In Section 2 we state the problem and review existing methods.

Next, in Section 3 we describe the model and the algorithm.

Theoretic properties of the algorithm are derived in Section 4.

Finally, in Section 5 and 6 we provides empirical results of CSI on the simulated and the real datesets respectively.

We discuss some future directions in Section 7.

Let y(t) be the trajectory of our objective variable, such as the size of tumor, over fixed time range t ∈ [t min , t max ], and N be the number of patients.

For each patient 1 ≤ i ≤ N , we measure its trajectory y i (t) at n i irregularly time points t i = [t i,1 , t i,2 , ..., t i,ni ] and denote the results as

.

We are primarily interested in estimating the disease progression trajectories

To fit a continuous curve based on discrete observations, we restrict our estimations to a finitedimensional space of functions.

Let {b i , i ∈ N} be a fixed basis of L 2 ([t min , t max ]) (e.g. splines, Fourier basis) and b = {b i : 1 ≤ i ≤ K} be first K dimensions of it.

The problem of estimating y i (t) can then be reduced to the problem of estimating the coefficients

Though intuitive, the above method has two main drawbacks.

First, when the number of observations per patient is less than or equal to the number of basis functions K, we can perfectly fit any curve without error, leading to overfitting.

Moreover, this direct approach ignores the similarities between curves.

Different patients may share similar trend of the trajectories which could potentially imporve the prediction.

Below we describe two main lines of research improving on this, the mixed-effect model and the matrix completion model.

In mixed-effect models, every trajectory y i (t) is assumed to be composed of two parts: the fixed effect µ(t) = m b(t) for some m ∈ R K that remains the same among all patients and a random effect w i ∈ R K that differs for each i ∈ {1, . . .

, N }.

In its simplest form, we assume

where Σ is the K × K covariance matrix, σ is the standard deviation and

are functions µ(t) and b(t) evaluated at the times t i , respectively.

Estimations of model parameters µ, Σ can be made via expectation maximization (EM) algorithm (Laird & Ware, 1982) .

Individual coefficients w i can be estimated using the best unbiased linear predictor (BLUP) (Henderson, 1975) .

In linear mixed-effect model, each trajectory is estimated with |w i | = K degrees of freedom, which can still be too complex when observations are sparse.

One typical solution is to assume a low-rank structure of the covariance matrix Σ by introducing a contraction mapping A from the functional basis to a low-dimensional latent space.

More specifically, one may rewrite the LMM model as

where A is a K × q matrix with q < K andw i ∈ R q is the new, shorter random effect to be estimated.

Methods based on low-rank approximations are widely adopted and applied in practice and different algorithms on fitting the model have been proposed (James et al., 2000; Lawrence, 2004; Schulam & Arora, 2016) .

In the later sections, we will compare our algorithm with one specific implementation named functional-Principle-Component-Analysis (fPCA) (James et al., 2000) , which uses EM algorithm for estimating model parameters and latent variables w i .

While the probabilistic approach of mixed-effect models offers many theoretical advantages including convergence rates and inference testing, it is often sensitive to the assumptions on distributions, some of which are hard to verify in practice.

To avoid the potential bias of distributional assumptions in mixed-effect models, Kidzinski and Hastie (Kidziński & Hastie, 2018) formulate the problem as a sparse matrix completion problem.

We will review this approach in the current section.

To reduce the continuous-time trajectories into matrices, we discretize the time range

T ×K be the projection of the K-truncated basis b onto grid G.

by rounding the time t i,j of every observation y i (t i,j ) to the nearest time grid and regarding all other entries as missing values.

Due to sparsity, we assume that no two observation y i (t i,j )'s are mapped to the same entry of Y .

Let Ω denote the set of all observed entries of Y .

For any matrix A, let P Ω (A) be the projection of A onto Ω, i.e. P Ω (A) = M where M i,j = A i,j for (i, j) ∈ Ω and M i,j = 0 otherwise.

Similarly, we define P ⊥ Ω (A) = A − P Ω (A) to be the projection on the complement of Ω. Under this setting, the trajectory prediction problem is reduced to the problem of fitting a N × K matrix W such that W B ≈ Y on observed indices Ω.

The direct way of estimating W is to solve the optimization problem

where · F is the Fröbenius norm.

Again, if K is larger than the number of observations for some subject we will overfit.

To avoid this problem we need some additional constraints on W .

A typical approach in the matrix completion community is to introduce a nuclear norm penalty-a relaxed version of the rank penalty while preserving convexity (Rennie & Srebro, 2005; Candès & Recht, 2009) .

The optimization problem with the nuclear norm penalty takes form

where λ > 0 is the regularization parameter, · F is the Fröbenius norm, and · * is the nuclear norm, i.e. the sum of singular values.

In Kidziński & Hastie (2018) , a Soft-Longitudinal-Impute (SLI) algorithm is proposed to solve (2.2) efficiently.

We refer the readers to Kidziński & Hastie (2018) for detailed description of SLI while noting that it is also a special case of our algorithm 1 defined in the next section with µ fixed to be 0.

In this section, we introduce our model on effect of treatments in disease progression.

A wide variety of treatments with different effects and durations exist in medical practice and it is impossible to build a single model to encompass them all.

In this study we take the simplified approach and regard treatment, with the example of one-time surgery in mind, as a non-recurring event with an additive effect on the targeted variable afterward.

Due to the flexibility of formulation of optimization problem (2.1), we build our model based on matrix completion framework of Section 2.2.

More specifically, let s(i) ∈ G be the time of treatment of the i'th patient, rounded to the closest τ k ∈ G (s(i) = ∞ if no treatment is performed).

We encode the treatment information as a N × T zero-one matrix I S , where (I S ) i,j = 1 if and only τ j ≥ s(i), i.e. patient i has already taken the treatment by time τ j .

Each row of I S takes the form of (0, · · · , 0, 1, · · · , 1).

Let µ denote the average additive effect of treatment among all patients.

In practice, we have access to the sparse observation matrix Y and surgery matrix I S and aim to estimate the treatment effect µ and individual coefficient matrix W based on Y, I S and the fixed basis matrix B such that W B + µI S ≈ Y .

Again, to avoid overfitting and exploit the similarities between individuals, we add a penalty term on the nuclear norm of W .

The optimization problem is thus expressed as:

for some λ > 0.

Though the optimization problem (3.1) above does not admit an explicit analytical solution, it is not hard to solve for one of µ or W given the other one.

For fixed µ, the problem reduces to the optimization problem (2.2) withỸ = Y − µI S and can be solved iteratively by the SLI algorithm Kidziński & Hastie (2018) , which we will also specify later in Algorithm 1.

For fixed W , we have arg min

where Ω S is the set of non-zero indices of I S .

Optimization problem (3.2) can be solved by taking derivative with respect to µ directly, which yieldŝ

The clean formulation of (3.3) motivates us to the following Coordinatewise-Soft-Impute (CSI) algorithm (Algorithm In the definition, we define operator S λ as for any matrix X,

).

Note that if we set µ ≡ 0 throughout the updates, then we get back to our base model SLI without treatment effect.

In this section we study the convergence properties of Algorithm 1.

Fix the regularization parameter

λ ) be the value of (µ, W ) in the k'th iteration of the algorithm, the exact definition of which is provided below in (4.4).

We prove that Algorithm 1 reduces the loss function at each iteration and eventually converges to the global minimizer.

λ ) converges to a limit point (μ λ ,Ŵ λ ) which solves the optimization problem:

Moreover, (μ λ ,Ŵ λ ) satisfies that

The proof of Theorem 1 relies on five technique Lemmas stated below.

The detailed proofs of the lemmas and the proof to Theorem 1 are provided in Appendix A. The first two lemmas are on properties of the nuclear norm shrinkage operator S λ defined in Section 3.1.

Lemma 1.

Let W be an N × K matrix and B is an orthogonal T × K matrix of rank K. The solution to the optimization problem min W

is defined in Section 3.1.

Lemma 2.

Operator S λ (·) satisfies the following inequality for any two matrices W 1 , W 2 with matching dimensions:

Lemma 1 shows that in the k-th step of Algorithm 1,

is the minimizer for function

The next lemma proves the sequence of loss functions

is monotonically decreasing at each iteration.

Lemma 3.

For every fixed λ ≥ 0, the k'th step of the algorithm (µ

Then with any starting point (µ

The next lemma proves that differences

F both converge to 0.

Lemma 4.

For any positive integer k, we have W

Finally we show that if the sequence {(µ

λ )} k , it has to converge to a solution of (4.1).

In this section we illustrate properties of our Coordinatewise-Soft-Impute (CSI) algorithm via simulation study.

The simulated data are generated from a mixed-effect model with low-rank covariance structure on W :

for which the specific construction is deferred to Appendix B. Below we discuss the evaluation methods as well as the results from simulation study.

We compare the Coordinatewise-Soft-Impute (CSI) algorithm specified in Algorithm 1 with the vanilla algorithm SLI (corresponding toμ = 0 in our notation) defined in Kidziński & Hastie (2018) and the fPCA algorithm defined in James et al. (2000) based on mixed-effect model.

We train all three algorithms on the same set of basis functions and choose the tuning parameters λ (for CSI and SLI) and R (for fPCA) using a 5-fold cross-validation.

Each model is then re-trained using the whole training set and tested on a held-out test set Ω test consisting 10% of all data.

The performance is evaluated in two aspects.

First, for different combinations of the treatment effect µ and observation density ρ, we train each of the three algorithms on the simulated data set, and compute the relative squared error between the ground truth µ and estimationμ., i.e., RSE(μ) = (μ − µ) 2 /µ 2 .

Meanwhile, for different algorithms applied to the same data set, we compare the mean square error between observation Y and estimationŶ over test set Ω test , namely,

We train our algorithms with all combinations of treatment effect µ ∈ {0, 0.2, 0.4, · · · , 5}, observation rate ρ ∈ {0.1, 0.3, 0.5}, and thresholding parameter λ ∈ {0, 1, · · · , 4} (for CSI or SLI) or rank R ∈ {2, 3, · · · , 6} (for fPCA).

For each fixed combination of parameters, we implemented each algorithm 10 times and average the test error.

The results are presented in Table 1 and Figure 1 .

From Table 1 and the left plot of Figure 1 , we have the following findings:

1.

CSI achieves better performance than SLI and fPCA, regardless of the treatment effect µ and observation rate ρ.

Meanwhile SLI performs better than fPCA.

2.

All three methods give comparable errors for smaller values of µ. In particular, our introduction of treatment effect µ does not over-fit the model in the case of µ = 0.

3.

As the treatment effect µ increases, the performance of CSI remains the same whereas the performances of SLI and fPCA deteriorate rapidly.

As a result, CSI outperforms SLI and fPCA by a significant margin for large values of µ. For example, when ρ = 0.1, the MSE(Ŷ ) of CSI decreases from 72.3% of SLI and 59.6% of fPCA at µ = 1 to 12.4% of SLI and 5.8% of fPCA at µ = 5.

4.

All three algorithms suffer a higher MSE(Ŷ ) with smaller observation rate ρ.

The biggest decay comes from SLI with an average 118% increase in test error from ρ = 0.5 to ρ = 0.1.

The performances of fPCA and CSI remains comparatively stable among different observation rate with a 6% and 12% increase respectively.

This implies that our algorithm is tolerant to low observation rate.

To further investigate CSI's ability to estimate µ, we plot the relative squared error ofμ using CSI with different observation rate in the right plot of Figure 1 .

As shown in Figure 1 , regardless of the choice of observation rate ρ and treatment effect µ, RSE(μ) is always smaller than 1% and most of the estimations achieves error less than 0.1%.

Therefore we could conclude that, even for sparse matrix Y , the CSI algorithm could still give very accurate estimate of the treatment effect µ.

In this section, we apply our methods to real dataset on the progression of motor impairment and gait pathology among children with Cerebral Palsy (CP) and evaluate the effect of orthopaedic surgeries.

Cerebral palsy is a group of permanent movement disorders that appear in early childhood.

Orthopaedic surgery plays a major role in minimizing gait impairments related to CP (McGinley et al., 2012).

However, it could be hard to correctly evaluate the outcome of a surgery.

For example, the seemingly positive outcome of a surgery may actually due to the natural improvement during puberty.

Our objective is to single out the effect of surgeries from the natural progression of disease and use that extra piece of information for better predictions.

We analyze a data set of Gillette Children's Hospital patients, visiting the clinic between 1994 and 2014, age ranging between 4 and 19 years, mostly diagnosed with Cerebral Palsy.

The data set contains 84 visits of 36 patients without gait disorders and 6066 visits of 2898 patients with gait pathologies.

Gait Deviation Index (GDI), one of the most commonly adopted metrics for gait functionalities (Schwartz & Rozumalski, 2008) , was measured and recorded at each clinic visit along with other data such as birthday, subtype of CP, date and type of previous surgery and other medical results.

Our main objective is to model individual disease progression quantified as GDI values.

Due to insufficiency of data, we model surgeries of different types and multiple surgeries as a single additive effect on GDI measurements following the methodology from Section 3.

We test the same three methods CSI, SLI and fPCA as in Section 5, and compare them to two benchmarks-the population mean of all patients (pMean) and the average GDI from previous visits of the same patient (rMean).

All three algorithms was trained on the spline basis of K = 9 dimensions evaluated at a grid of T = 51 points, with regularization parameters λ ∈ {20, 25, ..., 40} for CSI and SLI and rank constraints r ∈ {2, . . .

, 6} for fPCA.

To ensure sufficient observations for training, we cross validate and test our models on patients with at least 4 visits and use the rest of the data as a common training set.

The effective size of 2-fold validation sets and test set are 5% each.

We compare the result of each method/combination of parameters using the mean square error of GDI estimations on held-out entries as defined in (5.1).

We run all five methods on the same training/validation/test set for 40 times and compare the mean and sd of test-errors.

The results are presented in Table 2 and Figure 2 .

Compared with the null model pMean (Column 2 of Table 2 ), fPCA gives roughly the same order of error; CSI, SLI and rowMean provide better predictions, achieving 62%, 66% and 73% of the test errors respectively.

In particular, our algorithm CSI improves the result of vanilla model SLI by 7%, it also provide a stable estimation with the smallest sd across multiple selections of test sets.

We take a closer look at the low-rank decomposition of disease progression curves provided by algorithms.

Fix one run of algorithm CSI with λ = 30, there are 6 non-zero singular value vectors, which we will refer as principal components.

We illustrate the top 3 PCs scaled with corresponding singular values in Figure 3a .

An example of predicted curve from patient ID 5416 is illustrated in Figure 3b , where the blue curve represents the prediction without estimated treatment effectμ = 4.33, green curve the final prediction and red dots actual observations.

It can be seen that the additive treatment effect helps to model the sharp difference between the exam before exam (first observation) and later exams.

In this paper, we propose a new framework in modeling the effect of treatment events in disease progression and prove a corresponding algorithm CSI.

To the best of our knowledge, it's the first comprehensive model that explicitly incorporates the effect of treatment events.

We would also like to mention that, although we focus on the case of disease progression in this paper, our framework is quite general and can be used to analyze data in any disciplines with sparse observations as well as external effects.

There are several potential extensions to our current framework.

Firstly, our framework could be extended to more complicated settings.

In our model, treatments have been characterized as the binary matrix I S with a single parameter µ. In practice, each individual may take different types of surgeries for one or multiple times.

Secondly, the treatment effect may be correlated with the latent variables of disease type, and can be estimated together with the random effect w i .

Finally, our framework could be used to evaluate the true effect of a surgery.

A natural question is: does surgery really help?

CSI provides estimate of the surgery effect µ, it would be interesting to design certain statistical hypothesis testing/casual inference procedure to answer the proposed question.

Though we are convinced that our work will not be the last word in estimating the disease progression, we hope our idea is useful for further research and we hope the readers could help to take it further.

Proof of Lemma 1.

Note that the solution of the optimization problem

is given byÂ = S λ (Z) (see Cai et al. (2010) for a proof).

Therefore it suffices to show the minimizer of the optimization problem (A.1) is the same as the minimizer of the following problem:

Using the fact that A 2 F = Tr(AA ) and B B = I K , we have

On the other hand

as desired.

Proof of Lemma 2.

We refer the readers to the proof in Mazumder et al. (2010, Section 4, Lemma 3) .

Proof of Lemma 3.

First we argue that µ

λ , µ) and the first inequality immediately follows.

We have

Taking derivative with respect to µ directly gives µ

λ , µ), as desired.

For the rest two inequalities, notice that

Here the (A.2) holds because we have

(A.3) follows from the fact that W

Proof of Lemma 4.

First we analyze the behavior of {µ

Meanwhile, the sequence

is decreasing and lower bounded by 0 and therefore converge to a non-negative number, yielding the differences

as desired.

The sequence {W (k) λ } is slightly more complicated, direct calculation gives

2 F , (A.6) where (A.5) follows from Lemma 2, (A.6) can be derived pairing the 4 terms according to P Ω and P ⊥ Ω .

By definition of µ

where (A.7) follows from the Cauchy-Schwartz inequality.

Combining (A.6) with (A.7), we get

Now we are left to prove that the difference sequence {W

} converges to zero.

Combining (A.4) and (A.7) it suffices to prove that P

and the left hand side converges to 0 because

which completes the proof.

Taking limits on both sides gives us the desire result.

Proof of Theorem 1.

Let (μ λ ,Ŵ λ ) be one limit point then we have:

here (A.8) uses Lemma 5 and (A.9) uses Lemma 2.

Meanwhile, Mazumder et al. (2010) guarantees 0 ∈ ∂ W f λ (Ŵ λ ,μ).

By taking derivative directly we have 0 = ∂ µ f λ (Ŵ λ ,μ).

Therefore (Ŵ λ ,μ) is a stationary point for f λ (W, µ).

Notice that the loss function f λ (W, µ) is a convex function with respect to (W, µ).

Thus we have proved that the limit point (Ŵ λ ,μ) minimizes the function f λ (W, µ).

Let G be the grid of T equidistributed points and let B be the basis of K spline functions evaluated on grid G. We will simulate the N × K observation matrix Y with three parts Y = W B + µI S + E, where W follows a mixture-Gaussian distribution with low rank structure, I S is the treatment matrix with uniformly distributed starting time and E represents the i.i.d.

measurement error.

The specific procedures is described below.

1.

Generating W given parameters κ ∈ (0, 1), r 1 , r 2 ∈ R, s 1 , s 2 ∈ R K ≥0 : (a) Sample two K × K orthogonal matrices V 1 , V 2 via singular-value-decomposing two random matrix.

where diag[s] is the diagonal matrix with diagonal elements s, "·" represents coordinatewise multiplication, and we are recycling t, 1 − t and r i γ i to match the dimension.

2.

Generating I S given parameter p tr ∈ (0, 1).

(a) For each k = 1, . . .

, N , sample T k uniformly at random from {1, . . .

, T /p tr }.

(b) Set I S ← (1{j ≥ T i }) 1≤i≤N,1≤j≤T .

3.

Given parameter ∈ R ≥0 , E is drawn from from i.i.d.

Normal(0, 2 ) samples.

4.

Given parameter µ ∈ R, let Y 0 ← W B + µI S + E.

5.

Given parameter ρ ∈ (0, 1), drawn 0-1 matrix I Ω from i.i.d.

Bernoulli(ρ) samples.

Let Ω denote the set of non-zero entries of I Ω , namely, the set of observed data.

Set

, where Y ij = (Y 0 ) ij if (I Ω ) ij = 1 NA otherwise .

In actual simulation, we fix the auxiliary parameters as follows, The remaining parameters are treatment effect µ and observation rate ρ, which we allow to vary across different trials.

<|TLDR|>

@highlight

A novel matrix completion based algorithm to model disease progression with events