In this paper, we study the learned iterative shrinkage thresholding algorithm (LISTA) for solving sparse coding problems.

Following assumptions made by prior works, we first discover that the code components in its estimations may be lower than expected, i.e., require gains, and to address this problem, a gated mechanism amenable to theoretical analysis is then introduced.

Specific design of the gates is inspired by convergence analyses of the mechanism and hence its effectiveness can be formally guaranteed.

In addition to the gain gates, we further introduce overshoot gates for compensating insufficient step size in LISTA.

Extensive empirical results confirm our theoretical findings and verify the effectiveness of our method.

Sparse coding serves as the foundation of many machine learning applications, e.g., the direction-ofarrival estimation (Xu et al., 2012) , signal denoising (Elad & Aharon, 2006) , and super resolution imaging (Yang et al., 2010) .

In general, it aims to recover an inherently sparse vector x s ∈ R n from an observation y ∈ R m corrupted by a noise vector ε ∈ R m .

That is,

in which A ∈ R m×n is an over-complete basis matrix.

The problem of recovering x s , however, is a challenging task, in which the main difficulties are to incorporate the sparse constraint which is nonconvex and to further determine the indices of its non-zero elements, i.e., the support of the vector.

A reasonable solution to the problem is to use convex functions as surrogates to relax the constraint of sparsity, among which the most classical one probably is the l 1 -norm penalty.

Such a problem is carefully studied in Lasso (Tibshirani, 1996) , and it can be solved via least angle regression (Efron et al., 2004) , the iterative shrinkage and thresholding algorithm (ISTA) (Daubechies et al., 2004) , etc.

Despite the simplicity, these conventional solvers suffer from critical shortcomings.

Taking ISTA as an example, we know that 1) it converges very slowly with only a sublinear rate (Beck & Teboulle, 2009) , 2) the correlation between each of the two columns of A should be relatively low.

In recent years, deep learning (LeCun et al., 2015) methods have achieved remarkable successes.

Deep neural networks (DNNs) have been proven both effective and efficient in dealing with many tasks, including image classification (He et al., 2016) , object detection (Girshick, 2015) , speech recognition (Hinton et al., 2012) , and also sparse coding (Gregor & LeCun, 2010; Borgerding et al., 2017; He et al., 2017; Zhang & Ghanem, 2018; Chen et al., 2018; Liu et al., 2019; Sulam et al., 2019) .

The core idea behind deep learning-based sparse coding is to train DNNs to approximate the optimal sparse code.

For instance, an initial work of Gregor and LeCun's (2010) takes the inspiration from ISTA and develops an approximator named learned ISTA (LISTA), which is structurally similar to a recurrent neural network (RNN).

It has been demonstrated both empirically and theoretically that LISTA is superior to ISTA Moreau & Bruna, 2017; Giryes et al., 2018; Chen et al., 2018) .

Nevertheless, it is also uncontroversial that there exists much room for further enhancing it.

In this paper, we delve deeply into the foundation of (L)ISTA and discover possible weaknesses of LISTA.

First and foremost, we know from prior arts (Chen et al., 2018; Liu et al., 2019) that LISTA tends to learn large enough biases to achieve no "false positive" in the support of generated codes and further ensure linear convergence, and we prove that this tendency, however, also makes the magnitude of the code components being lower than that of the ground-truth.

That said, there probably exists a requirement of gains in the code estimations.

Second, regarding the optimization procedure of ISTA as to minimize an upper bound of its objective function at each step, we conjecture that the element-wise update of (L)ISTA normally "lags behind" the optimal solution, which suggests that it requires overshoots to reach the optimum, just like what has been suggested in fast ISTA (FISTA) (Beck & Teboulle, 2009 ) and learned FISTA (LFISTA) (Moreau & Bruna, 2017) .

In this paper, our main contributions are summarized as follows:

• We discover weaknesses of LISTA by theoretically analyzing its optimization procedure, for mitigating which we introduce gain gates and overshoot gates, akin to update gate and reset gate mechanisms in the gated recurrent unit (GRU) Cho et al. (2014) .

• We provide convergence analyses for LISTA (with or without gates), which further give rise to conditions on which the performance of our method with gain gates can be guaranteed.

A practical case is considered, where the assumption of no "false positive" is relaxed.

• Insightful expressions for the gates are presented.

In comparison with state-of-the-art sparse coding networks (not limited to previous extensions to LISTA), our method achieves superior performance.

It also applies to variants of LISTA, e.g., LFSITA (Moreau & Bruna, 2017) and ALISTA (Liu et al., 2019) .

Notations:

In this paper, unless otherwise clarified, vectors and matrices are denoted by lowercase and uppercase characters, respectively.

For vectors/matrices originally introduced without any subscript, adding a subscript (e.g., i) indicates its element/column at the corresponding position.

For instance, for x ∈ R n , x i represents the i-th element of the vector, and W :,i and W i,: denote the i-th column and row of a matrix W respectively.

While for vectors introduced with subscripts already, e.g., x s , we use (x s ) i to denote its i-th element.

The operator is used to indicate element-wise multiplication of two vectors.

The support of a vector is denoted as supp(x) := {i|x i = 0}. We use sup xs as the simplified form of sup xs∈X (B,s,0) , see Assumption 1 for the definition of X (B, s, 0).

In general, sparse coding solves the problem that can be formulated as

in which f (x, y) calculates the residual of approximating y using a linear combination of column-wise features in A. The function f (x, y) is convex with respect to x in general.

In particular, if ε is a Gaussian vector, then it should be f (x, y) = Ax − y 2 2 .

The term λr(x) serves as a regularizer for sparsity and we have r(x) = x 1 in Lasso.

As mentioned, a variety of algorithms can be applied to solve the problem and our focus in the paper is (L)ISTA.

We first revisit the optimization procedure of ISTA, which is the foundation of LISTA as well.

Given y, let us introduce a scalar γ > 0 that fulfills γI − ∇ 2 x f (x, y) 0, ∀x, then it can be considered as optimizing an upper bound of the objective function obtained via Taylor expansion.

To be more specific, for any presumed x (t) , we have

By substituting r(x) with x 1 and optimizing the bound in an element-wise manner, we can easily get the one-step update rule that zeros the gradient based on x (t) .

It is, x (0) = 0 and

in which s b (x) := sign(x)(|x| − b) + is a shrinking function and (·) + is a rectified linear unit (ReLU) calculating max{0, ·}.

For Gaussian noises, the formulation reduces to

The update as shown in Eq. (4) and (5) can be performed iteratively until convergence.

However, the convergence of ISTA (along with some other conventional solvers) is known to be slow, and it has been shown that DNNs can be utilized to accelerate the procedure.

Many researchers have explored the idea since the initial work of Gregor and LeCun's (i.e., LISTA) .

For LISTA, they design deep architectures following the main procedure of ISTA yet to learn parameters in an end-to-end manner from data (Gregor & LeCun, 2010; Hershey et al., 2014) .

The inference process of LISTA is similar to that of an RNN and can be formulated as x (0) = 0 and

where Θ = {U (t) , W (t) , b (t) } t=0,1,...,d−1 , is learnable parameters set.

Some works (Xin et al., 2016; Chen et al., 2018) have proved that W (t) and U (t) should satisfy the constraint

The parameters in Θ are normally learned from a set of training samples by minimizing the difference between the final code estimations and ground-truth.

In this paper, our main assumption for theoretical analyses follows those of prior works (Chen et al., 2018; Liu et al., 2019) in a noiseless case, and noisy cases will be considered in the experiments.

Assumption 1.

The sparse vector x s and noise vector ε are sampled from a set X (B, s, 0) fulfilling:

In this section, we will introduce the advocated gain gates and overshoot gates.

Along with thorough discussions for the motivations, their formulations are provided in Section 3.1 and 3.2, respectively.

Figure 1 summarizes the inference process of the standard LISTA and two evolved versions with our gates incorporated.

Proofs of all our theoretical results are deferred to the appendix. (Chen et al., 2018; Liu et al., 2019) .

In order to guarantee the convergence, it is also demonstrated that the value of bias terms should be large enough to eliminate all "false positive" in the support of the generated codes.

However, this may lead to an issue that the magnitude of the generated code components in LISTA must be smaller than or at most equal to those of the ground-truth.

Our result in Proposition 1 makes this formal.

For clarity of the result, we would like to introduce the following definition first.

Definition 1. (Liu et al., 2019 ) Given a matrix A ∈ R m×n , its generalized mutual coherence is:

We let W(A) denote a set of all matrices that can achieve the generalized mutual coherence µ(A), which means:

Proposition 1. (Requirement of gains).

With U (t) ∈ W(A) and

(t) − x s 1 is achieved in LISTA to guarantee no "false positive" (i.e., supp(x (t) ) ⊂ supp(x s )) and further linear convergence (i.e., x (t) − x s 2 ≤ sB exp(ct), in which c = log((2s − 1)µ(A))), then we have for the estimation |x The generated code estimation can be more accurate if we enforce gains on its components.

Provided Proposition 1 as the evidence of a potential weakness of LISTA, we believe that if the code components can be enlarged appropriately, then the estimation at each step would be closer to x s , and the convergence of LISTA will be further improved, which inspires us to design a gate to enlarge the generated code components.

Such a gate is named as a gain gate and it acts on the input to the current estimation, akin to a reset gate in GRU (Cho et al., 2014) , which is

in which the gate function g t (·, ·|Λ

) outputs an n-dimensions vector, and

g is the set of its learnable parameters.

In the original implementation of LISTA, the output of each layer is obtained by calculating Eq. (4) iteratively.

It has been proven that the estimation x (t) ultimately converges to the ground-truth x s (as t → ∞), only if the condition of (W (t) − (I − U (t) A)) → 0 holds.

That said, it is suggested that U (t) and W (t) are entangled to the end.

Yet, with our gated mechanism, the update rule in neural networks has been modified into Eq. (10), making it unclear whether the convergence is guaranteed similarly or not.

To figure it out, we perform theoretical analyses in depth, which will further provide guidance for the gate design.

We are going to explore: whether the learnable matrices are still entangled as in LISTA, and to encourage fast convergence, what properties should the gate function satisfy?

Theorem 1 and 2 give some answers to these questions and they are based on the same assumptions as for Proposition 1.

Theorem 1.

If the s-th principal minor of W (t) have full rank, then for the gate function bounded from both above and below, we have x s as the fixed point of Eq. (10) only if

in which D is an n × n constant diagonal matrix and the function diag(·) creates a diagonal matrix with the elements of its input on the main diagonal.

From Theorem 1 we can equivalently have (

means the learnable matrices are similarly entangled as in the standard LISTA.

Besides, we know that as the number of layers increases, each introduced gain gate should ultimately converge to a constant (diagonal) matrix D to guarantee performance.

Then if W (t) → I − U (t) A, the gain gate function converges to an identical mapping as t → ∞, and vice versa.

This inspires us to "split" the gate function into an identical one and a residual one, and we thus advocate, for each index i of the vector, the i-th element of gain gate is

in which κ t (x (t) , y|Λ

g ), and it should decrease as t increases, in order to guarantee convergence in Eq. (11).

Let us further study the convergence rate of "LISTA" equipped with such gain gates.

For clarity, we introduce another condition for the function before moving to more details:

We present theoretical results as follows on the basis of Proposition 1, i.e., we still have

)

1 is achieved, following the update rule in Eq. (10), if the conditions in Eq. (12) and (13) hold for the gate function, there will be

in which c = log((2s − 1)µ(A)), c i = c if i ≤ log( sB xs 1 )/ log( 1 (2s−1)µ(A) ) , and c i < c otherwise.

Theorem 2 presents an upper bound of x (t) − x s 2 for LISTA with gain gates, and it shows that so long as the gates satisfying conditions in Eq. (12) and (13) By consolidating all these theoretical cues, we further give principled expressions for the gate function.

One may expect to endow the gates some learning capacities, thus we let

in which µ t ∈ R is a parameter to be learned, b (t−1) is the threshold parameter of the (t − 1)-th layer, and f t (x (t) |ν t ) is a newly introduced function constrained not to be greater than 2/|x (t) |.

We are going to evaluate different choices for the function f t (x (t) ) in experiments, e.g., the piece-wise linear function:

the inverse proportional function:

in which ν t ∈ R is a parameter to be learned, and is a tiny positive scalar introduced to avoid zero being divided.

All the learnable parameters in a gain gate are thus collected as Λ (t) g = {µ t , ν t }.

Our previous theoretical results show that the performance of LISTA can be improved by using a gain gate, as long as the gate function satisfies conditions in Eq. (12) and (13), and no "false positive" is encountered.

However, it is not always true in practice.

Our experimental results also show that when the inverse proportional function is adopted as gain gates in lower layer for LISTA, the performance of our gated LISTA may even degrade.

We conjecture that such contradiction to the theoretical results may be owing to impractical assumptions.

In this subsection, we try to relax the assumption about no "false positive", and we further found that a tighter bound can be achieved with a more reasonable assumption instead.

Through theoretical analyses as follows, we also demonstrate that the inverse proportional gain function should better be only adopted in higher layers.

For clarity of the results, we would like to introduce the following definition first.

Definition 2.

Given a model with Θ, in which

g ) − x s 1 , we introduce ω t+1 (k|Θ) to characterize its relationship with the false positive rate, which is

, and k t+1 ≥ 0 is the desired maximal number of "false positive" of x (t+1) .

The above definition applies to both the standard LISTA and LISTA with gain gates (we can let the gate function be an identity function to achieve a standard LISTA).

We first analyze the convergence of LISTA without gates.

We present theoretical results as follows on the basis of similar assumptions (including Assumption 1, U (t) ∈ W(A), and W (t) = I − U (t) A), but with a different requirement for b (t) from Proposition 1.

0 ), then there exists "false positive" with 0 < k t < s and

It can be seen that when we relax the assumption about no "false positive" and further reduce the value of the threshold b (t) , the error bound of LISTA becomes even lower.

Obviously, the previous bound of LISTA with gain gates in Theorem 2 is not necessarily lower than the tighter bound of a standard LISTA in Theorem 3, which well explains the contradiction of theoretical and empirical results.

Here we re-deduce the error bound of our gated LISTA with the inverse proportional function in the following theorem.

Note that we still have U (t) ∈ W(A), W (t) = I − U (t) A and Assumption 1.

We can conclude from Theorem 4 that, a) a gain gate expressed by the inverse proportional function should be applied to deeper layers in LISTA, rather than lower layers, b) when using the function, there indeed exists no "false positive" (i.e., k i = 0) in deeper layer.

We follow such guidelines in the implementation of our gated LISTA.

In addition, we observe that unlike the inverse proportional function, other considered functions show consistent performance gains on both lower and higher layers, hence we attempt to utilize them on lower layers in alliance with the inverse proportional function powered gain gates on the other layers.

In practice, we choose the ReLU-based piece-wise linear function, and it is uniformly applied to the first 10 layers.

We will empirically compare different choices between the gain gate functions in Section 4.1.

Unlike the gain gates that are incorporated before performing estimation at each step, the overshoot gates act more like adjustments to the outputs, which can be viewed as learnable boosts:

The gate function o t (·, ·|Λ

o collects all the trainable parameters in the function, akin to a dedicated update GRU gate (Cho et al., 2014) .

Our motivation comes from analyses of ISTA, whose update can be viewed as

, in which η = 1 is a constant step size.

We argue that η = 1 may not be the most suitable choice and the following proposition makes this formal.

We have it to theoretically analyze the update rule of ISTA and η

is convex with respect to x and γI − ∇ 2 x f (x) 0 holds for all x, if the update rule in Eq. (4) is adopted, then we have η

See also Figure 3 for an illustration of the issue with η = 1 as concerned.

Since the optimization procedure of ISTA inspires the network architecture in LISTA, the theoretical result in Proposition 2 that requires a boost in η for superior performance also inspires us to design specific overshoot gates for LISTA.

Having noticed that an essential principle we have obtained is to let η ≥ 1 (or η > 1), we may expect the output of the gate function to be greater than or at least equal to 1.

To achieve the goal, we can try different expressions for it, e.g., the sigmoid-based function:

Figure 3: The derivative function (illustrated in blue) of f (x, y) + λr(x), in which r(x) = x 1 , is monotonic owing to the convexity of f (x, y) and r(x), and its output should be consistently smaller than the derivative (illustrated in orange) of the upper bound in absolute value.

Let x * be the optimal solution to the problem, then we know from the figure that the estimation with a standard ISTA update (i.e., η = 1) normally "lags behind".

with σ(·) being the sigmoid function, Λ

o = {a o } for the two types of functions respectively, and being a tiny positive constant introduced to avoid zero being divided.

The principle of our overshoot gate is similar to that of some momentum-based methods, e.g., FISTA (Beck & Teboulle, 2009 ) and LFISTA (Moreau & Bruna, 2017) .

The fundamental difference between these methods and ours is that, (L)FISTA considers that the scaling factor in a momentum term should be independent of the current inputs (including the previous estimation and y), i.e., being time or at least input invariant, while the output of the overshoot gate is a function of both the previous estimation and y, hence being time-and-input-varying.

The design of our overshoot gate may endow the sparse coding network higher capability to learn from its inputs.

Experimental comparisons in Section 7 in the Appendix confirm the superiority of our method.

We also note that our convergence analyses in Section 3.1 generalize to η > 1 cases with a constant η, i.e., linear convergence can still be guaranteed, but the asymptotic behavior with learnable and adaptive overshoots should be further explored in future studies.

In this section, we perform experiments to confirm our theoretical results and evaluate the performance of our gated sparse coding networks.

Validations of our theoretical results are performed on synthetic data, and the performance of our method in sparse coding is tested on both synthetic and real data.

We set m = 250, n = 500, and sample the elements of the dictionary matrix A randomly from a standard Gaussian distribution in simulations.

The position of non-zero elements of the sparse vector x s is determined by a Bernoulli sampling with a probability of 0.1 (which means approximately 90% of the elements are set to be zero).

Different noise levels and condition numbers are considered in the sparse coding simulations.

We randomly synthesize in-stream x s and ε to obtain y for training, and we let two extra sets consisting of 1000 samples each as the validation and test sets, just like in prior works. (Chen et al., 2018; Liu et al., 2019; Borgerding et al., 2017) .

For the proposed gated LISTA and other deep learning-based methods, we set d = 16 and let {b (t) } not be shared between different layers under all circumstances.

The weight matrices {W (t) , U (t) } are not shared either in our method and the coupled constraints W (t) = I − U (t) A, ∀t, are imposed.

For all gates, ν t is initialized as 1.0, and then we let the initial value of µ t in the inverse proportional function powered gain gate be 1.0 too, since Eq. (12) and (13) indicate 0 ≤ µ t ≤ 2.

Other learnable parameters in our gates are uniformly initialized as 5.0 according to their suggested range of the gates.

The training batch size is 64.

We use Adam (Cho et al., 2014) and let β 1 = 0.9 and β 2 = 0.999.

The hyper-parameters are tuned on the validation set and fixed for all our experiments in the sequel.

Our training follows it of Chen et al.'s (2018) .

That said, the sparse coding network is trained progressively to update more layers, and we cut the learning rate for currently optimized layers when no decrease in the validation loss can be observed for 4000 iterations, with a base learning rate of 0.0005.

Training on current layers stops when the validation loss does not decrease any more with the learning rate being cut to 0.00001.

More details are explained in Section 8 in the appendix. , since the two metrics (i.e., false positive rate and ratio of generated non-zero code components that require gains) do not make much sense with an initial code estimation (i.e., 0).

is the set of all learnable parameters in the sparse coding network that generates x (d) given y. Note that in comparison with the parameter set in a standard LISTA, it also contains the parameters in gate functions.

In practice, we are given a set of training samples and opt to minimize an empirical loss instead of the one in Eq. (19).

Our evaluation metric for sparse coding is the normalized MSE (NMSE) (Chen et al., 2018) :

4.1 SIMULATION EXPERIMENTS

We first confirm Proposition 1.

In order to ensure that LISTA fulfills the assumption about no "false positive", we introduce an auxiliary loss into the learning object as:

We formally introduce the false positive rate (FPR) as FPR =

and try to approach no "false positive" (i.e., LISTA-nfp) by setting λ = 5.0 in the experiment.

1 Check Figure 4 for an illustrative comparison between different models, we see LISTA-nfp achieves almost no "false positive" in practice in Figure 4 (a), but its convergence is slower as demonstrated in Figure 4 (c), which is consistent with our result in Theorem 3.

In addition, we also see in Figure 4 (b) that without "false positive", the code components in LISTA estimations are almost always less than those of the ground-truth, which confirms our Proposition 1.

Validation of Theorem 1: We aim to calculate W (t) D − (I − U (t) A) 2 using a gated LISTA with the introduced ReLU-based piece-wise linear gain gate function 2 .

To accomplish this task, we need to first evaluate the output of our gate function, which is expect to converge to 1 as shown in the theorem.

We show such a trend indeed exists in Figure 5(a) .

Consequently, the matrix D is supposed to be an identity matrix in the end and we can calculate W (t) − (I − U (t) A) 2 as a surrogate.

In Figure 5 (b), it converges to zero in the end and the results confirm the theorem.

We apply three kinds of gated LISTA with an alliance of gain gate functions (i.e., what has been introduced in Section 3.1.1), the exponential function, and the inverse proportional function respectively to verify our theoretical results.

They were named as GLISTA (which is the abbreviation of gated LISTA), GLISTA-exp, GLISTA-inv, respectively.

From Figure 4(c) , we see that when the models with such gain gates has no "false positive", all of them are superior to the standard LISTA without "false positive" as well, which is consistent with the conclusion of Theorem 2.

In addition, from Figure 4 (a), we can also see that there actually exist "false positives" in lower layers of GLISTA, but even without the auxiliary loss term, the evaluated FPR of our GLISTA and it variants approach zero in higher layers, which is in good agreement with Theorem 4.

Empirical analyses for the gate functions: It should be interesting to compare the performance of our gates with different expressions.

We test LISTA with different overshoot gate functions introduced in Section 3.2 in Figure 6 (a).

Both of them are incorporated with their learnable parameters being shared among layers.

It can be seen from Figure 6 (a) that the accelerations in convergence and gain in final performance are obvious, just as expected.

For LISTA with gain gates, one can check Figure 6 (b).

It can be seen that the performance degrades a lot if either the bias term or the µ t term is removed.

We also try different f t (·) functions, including the ReLU-based piece-wise linear one and some possibly more nonlinear ones as mentioned in Section 3.1.

We confirm that gate functions whose outputs are relatively closer to the boundary condition may perform better.

Yet, it is worth noting that when the outputs of inverse proportional function reach that boundary condition and being applied uniformly to all layers, the performance degrades (see LISTA-inv-in Figure 6 (b)).

These results suggest an alliance of gain gate functions in practice.

We further test a combination of gain gates and overshoot gates, despite the mechanism with solely gain gates is already good enough.

See Figure 6 (c), when overshoots are further incorporated, the convergence on lower layers becomes faster while the overall convergence is not affected much, leading to similar final performance when the model is very deep and superior performance when the model is relatively shallow.

Compared with other state-of-the-art methods: We consider four state-of-the-arts: LISTA with support selections (namely LISTA-C-S and LISTA-S, with and without the coupled constraint) (Chen et al., 2018) , analytic LISTA with support selections (ALISTA-S) (Liu et al., 2019) , and learned AMP (LAMP) (Borgerding et al., 2017) for comparison, and their official implementations are directly used.

The hyper-parameters are set following the papers (Borgerding et al., 2017; Chen et al., 2018) .

We compare our GLISTA with these competitive methods under different levels of noises (including the signal-to-noise ratios (SNRs) being equal to 40dB, 20dB, and 10dB) and different condition numbers (including 3, 30, and 100, with SNR=40dB).

See Figure 7 for comparisons between LISTA, LAMP, LISTA-S, LISTA-C-S, ALISTA-S, and our GLISTA in some of the settings.

Obviously, the introduced gates facilitate LISTA significantly, and the concerned NMSE diminishes the fastest using GLISTA.

See our Appendix for comparisons of final performance after multiple runs and the results in other settings (i.e., SNR: 20dB, 40dB, and condition number: 3).

We know from these results that using the gain gates solely can already outperforms existing state-of-the-arts, while incorporating the overshoot gates additionally may further boost the performance, as testified.

Applying our method to variants of LISTA: We also try adopting the introduced gates into some variants of LISTA to verify their "generalization ability".

Specifically, we incorporate the gain gates to LFISTA (Moreau & Bruna, 2017) and ALISTA (Liu et al., 2019 ) to obtain GFLISTA and AGLISTA, respectively.

Since ALISTA is suggested to be implemented with support set selection in the original paper, i.e. ALISTA-S, we also compare with it.

The experiment is performed under different levels of noises (40dB, 20dB, and 10dB).

As can be seen in Table 1 in which average results along with their standard deviations calculated over five runs are reported, models with our gain gates perform significantly better, which verifies that our method generalizes well.

We now test on a more practical task, i.e., photometric stereo analysis, using sparse coding.

For a 3D object with Lambertian surface, if there are q different light conditions, a camera or some other kinds of sensors can obtain q different observations, all with noises caused by shadows and specularities.

The observations can be represented as a vector o ∈ R q for estimating the norm vector v ∈ R 3 at any position on the surface.

It is generally formulated as o = ρLv + e, in which L ∈ R q×3 represents the normalized light directions (q directions), e ∈ R q is a noise which is often sparse, ρ ∈ R represents the albedo reflectivity.

Our task is to obtain v from o and L which is also known.

The estimation of e can be considered as a sparse coding problem, and one can use L † (o − e) to recover v given the estimation.

More detailed descriptions of the task can be found in Xin et al.'s paper (2016).

In the sparse coding problem, we have Q ∈ R (q−3)×q (the orthogonal complement of L) as the dictionary matrix (i.e., A in Eq. (1)), e as the sparse code to be estimated, and Qo as the observation (i.e., y in Eq. (1)).

We mainly follow settings in Xin et al.'s work, e.g. the vectors of L are randomly selected from the hemispherical surface, except that we test with q = 15, 25, 35, and let 40% of the elements of e be non-zero.

We use GLISTA here to estimate e and the final result for v is calculated as L † (o − e * ), where L † ∈ R 3×q is the pseudo-inverse of L and e * is the estimation.

Our method is compared with LISTA and two traditional methods where no explicit training is introduced, i.e. the original least square (LS) and least L1, in Table 2 .

Our evaluation metric is the mean (± standard deviation) error in degree and it is calculated using the bunny picture (Xin et al., 2016) .

In this paper, we study LISTA for solving sparse coding problems.

We discover its potential weaknesses and introduce gated mechanisms to address them accordingly.

In particular, we theoretically prove that LISTA with gain gates can achieve faster convergence than the standard LISTA.

We also discover that LISTA (with or without gates) can obtain lower reconstruction errors under a weaker assumption of "false positive" in its code estimations.

It helps us improve the convergence analyses to achieve more solid theoretical results, which have been perfectly confirmed in simulation experiments.

The effectiveness of our introduced gates is verified in a variety of sparse coding experiments and the state-of-the-art performance is achieved.

In the future, we aim to extend the method to convolutional neural networks to deal with more complex tasks.

Before we delve deeply into the proof, we first give some importance notations.

We define S as the support of the vector x s , i.e. S = supp(x s ), and let |S| denote the number of elements in the set S. For a vector that shares the same size with x s , say z, we denote by z S ∈ R |S| a vector that keeps the elements with indices of z in S and removes the others.

If the vectors have been introduced with subscripts already, e.g. x s , we use (x s ) S to denote vectors obtained in such a manner.

For a square matrix with the same number of row and column as the size of x s , say M , M (S, S) is its principal minor with the index set formed by removing rows and columns whose indices are not in S. Assume a vector x with no zero elements, sign(·) is defined as (sign(x)) i = x i /|x i |, i.e. (sign(x)) i = 1 when x i > 0, and (sign(x)) i = −1 when x i < 0.

Recall that the update rule of LISTA is x (0) = 0 and

Proof.

Recall the definition of S is S = supp(x s ).

For the shrinking function z = s

We use Mathematical Induction to prove supp(x (t) ) ⊂ S, ∀t = 0, 1, . . . , d − 1.

We assume

).

).

Multiply the two sides of the Eq. (23)

the inequality holds for a · b ≤ a ∞ b 1 and

From Eq. (24), we know |x = 0.

Therefore, the x (t+1) i = 0, when i / ∈ S, i.e., supp(x (t+1) ) ⊂ S. As x (0) = 0 ⊂ S, the supp(x (t) ) ⊂ S, ∀t.

The no "false positive" property has been proved.

According to Eq. (23), as support set of x s and x (t) are the subsets of S, there is

As supp(x (t+1) ) ⊂ S, accumulate all |x (25) with i ∈ S, there is

The second equation is because of U (t) ∈ W(A), so that |W i,: A :,j | ≤ µ(A) when i = j and (26), and take the supremum of Eq. (26).

As |S| ≤ s there is

Let c = log((2s − 1)µ(A)), the l 2 error bound of t-th layer in LISTA should be calculated as

where the last inequality is deduced since (x s ) i ≤ B, and x s 0 ≤ s. The linear convergence has been proved.

Refer to the Eq. (25), as x

In conclusion, we can obtain |x

Recall that the update rule of LISTA with gain gates is x (0) = 0 and

Proof.

We assume that b (t) is a vector, i.e., b (t) ∈ R n , in our proof to make it more general.

According to definition of the shrinking function s b (t) (·) and y = Ax s , Eq. (10) is

Concentrate on the situation of t → ∞.

In the main body of Theorem 1, ∀x s satisfying x s 0 ≤ s is the fixed point of Eq. (10) when t → ∞. Eq. (31) is

(32) The equation group of the indices in S in Eq. (32) is

) are bounded, the right hand side of Eq. (33) is also tend to 0, which is b

As the S can be selected arbitrarily as long as |S| ≤ s, b (t) also satisfies

Substitute the b

where the W (t) (S, S) is defined at start of this section.

Eq. (36) is

where

.

The i-th row and j-th column element in M is denoted as m ij .

From Eq. (37)

where this equation should hold for all x s in Assumption 1.

Assume (x s ) S → 0, for g κ (x s ) is bounded, we can conclude that m ij = 0, if i = j. From Eq. (38), the final form of g κ ((x s ) S ) is formulated as

From Eq. (38), we can conclude that g κ (x s ) i is a constant if i ∈ S, as the S could be arbitrary subset of {1, · · · , n} as long as |S| ≤ s. We could deduce that g κ (x s ) i is constant ∀i ∈ {1, . . .

, n} and g κ (x s ) must be constant vector, i.e.

where D is an n × n constant diagonal matrix.

The first part of conclusion of Theorem 1 has been proved.

Substitute b (t) in Eq. (34) and diag(g κ (x s )) in Eq. (39) into Eq. (32), Eq. (32) is.

where

. .

, Z n ] and the Z i is the i-th column of Z.

Give a x s satisfying only the i-th element of x s is non-zero and all the other elements are equal to zero, i.e., x s = [0, 0, . . .

, ω, . . . , 0] T = ωe i , in which e i is basis vector with only the i-th element being 1 and ω = 0.

Substitute the x s = ωe i into Eq. (40), there is

As the Eq. (41) should hold for ∀ω = 0, we can deduce that Z i = e i .

As the i is selected arbitrarily,

, e 2 , . . .

, e n ] = I. Thus we have completed the proof and get

6.3 PROOF OF THEOREM 2

Recall that the update rule of LISTA with gain gates is x (0) = 0 and

Proof.

We simplify the g t (x (t) , y|Λ

According to the definition of gain gate in Eq. (43), we have

Simplify the x

.

For the i-th equation in Eq. (44), and i / ∈ S, give the value of

With almost the same proof process in Proposition 1, we could deduce that

which is the no "false poistive" property.

Recall the Eq. (44) and substitute the 1 + κ t+1 (x (t+1) ) = g t+1 (x (t+1) ) into it:

We shall calculate the non-zero |∆ g x (t+1) i | with the index i.

The i could be seperated to two parts.

One is i ∈ S but i / ∈ supp(x (t+1) ), another one part is i ∈ S and i ∈ supp(x (t+1) ).

Two kinds of i are discussed respectively.

)

≤ 1.

Select the i-th equation in Eq. (47), there is

For i ∈ S and i ∈ supp(x (t+1) ), there must be x (t+1) = 0 and h(x (t+1) ) = sign(x (t+1) ).

Select the i-th equation in Eq. (47), there is

According to the condition in Eq. (12) and (13), the 0 < κ t (x) |x| < 2b

.

Substituting it to Eq. (49), there is

Accumulate all the |∆ g x (t+1) i | with all i ∈ S, and define s (t) = |supp(x (t) )| as the number of non-zeros elements in x (t) there is

Take the supremum of Eq. (51), let s (t) * denote the infimum of s (t) with all of the x s ∈ X (B, s, 0),

Eq. (52) gives the upper bound of sup xs ∆ g x

1 , next we shall deduce the relationship between x (t) − x s 1 and it.

For the last layer (t-th layer), from Eq. (44), we have

Using almost the same process in Eq. (26) and Eq. (27), we could deduce Eq. (53) that

1 ,

where the third inequality sign holds because of Eq. (52) and the last equation holds because of c = log((2s − 1)µ(A)).

As at least c t ≤ c satisfies, there will be sup xs

In conclusion, from Eq. (54), there is

where c = log((2s − 1)µ(A)), c i = c when i ≤ t 0 , and c i < c when i > t 0 .

Proof.

For the t-th layer of the LISTA, according to the Eq. (23), we have

Take the supremum of left part of (63), there is

where c * t+1 = sup xs log((

, so that the number of false positive is less or equal than k t+1 , i.e. |S (t+1) | ≤ k t+1 .

According to the previous proof, when b (t) = µ(A) sup xs x (t) − x s 1 , the number of false positive satisfies k t+1 = 0.

That means that ω(0|Θ) ≤ 1 and ω(k|Θ) ≤ 1 when k > 0 3 .

The c * t+1 should be

As assumption of ω t+1 (·|Θ), ∃k

0 , we substitute it to Eq. (65), there will be

Recall the Eq. (64), we have

The l 2 error bound of the t-th layer of LISTA is

where c * i < log((2s − 1)µ(A)).

Proof.

For the t-th layer given in Eq. (10), according to Eq. (47),

and

As the no false positive is not fit for

S. We still define S (t) as ∀i ∈ S (t) satisfies i ∈ supp(x (t) i ) but i / ∈ S and define S (t) as ∀i ∈ S (t) satisfies i ∈ S and i ∈ supp(x (t) ).

In order to calculate the non-zero

, we divide the i into three situations: i ∈ S (t+1) , i / ∈ S (t+1) but i ∈ S, and i ∈ S (t+1) .

For i ∈ S (t+1) , there must be x (t+1) i = 0 , and (x s ) i = 0.

Substitute the form of κ t into i-th equation of Eq. (70):

we have assume µ t ≤ 1

) is the same as that of

Take the supremum of the left part of Eq. (77), the sup xs ∆ g x

After the upper bound of

.

According to main process in the proof Theorem 3, let

As the minimal absolute value of x s is less or equal than σ, S (t) = S. Select the b (i) so that k i = 0, ω i (k i |Θ) ≤ 1.

Recall the form in Eq. (78), As sup xs |S i | = k i = 0, and c i is

As

i.e., c i < c * i .

As ω i (·|Θ) is the monotone decreasing function, the second "<" in Eq. (89)holds since k i 0 < s and the last "<" holds since k i 0 > 0.

|S| ≤ s, and |S (t) | ≤ k t .

According to the assumption of ω t (·|Θ),

According to the similar derivation in Theorem 3, c * in Eq. (85) should satisfies c * ≤ log((s + k t + (s − k t )ω t (k t |Θ))µ(A)) < log((2s − 1)µ(A)).

All of the conclusions in Theorem 4 have been proven.

Validation of Proposition 2: Some more experimental results are given here due to the length limit of the main body of our paper.

One might also be interested in our Proposition 2, hence we first conduct an experiment to confirm it.

We adopt ISTA with an adaptive overshoot and compare it with the standard ISTA for sparse coding.

The adaptation is obtained via enlarging the step size from 1.0 through backtracking line search (see section 8 for more details).

Figure 8 demonstrates that our overshoot mechanism facilitates ISTA optimization, and such a result confirms Proposition 2.

Comparison with similar methods: As mentioned in the main body of the paper, the overshoot gates is proposed do address insufficient step size, which is similar to the motivation of (L)FISTA.

LIHT and support select can also be considered as special cases of our gain gates (by letting µ t = 1 in the inverse proportional function).

We compare these similar methods with our overshoot and gain gates in Figure 9 .

It can be seen that when compared with LISTA, LFISTA converges faster in lower layers, and our overshoot gates also show such advantage.

When applying to deeper layers, LFISTA converges quite slow while the overshoot gates still perform well, which indicates that the time-varying property is beneficial in practice.

LISTA with our gain gates is obviously better than LIHT as shown in Figure 9 (b), and sufficient experimental results in the paper also prove that the gain gate outperforms support select (e.g., in LISTA-C-S and LISTA-S).

Comparison under less challenging settings: Now we also give sparse coding results under the described less challenging settings on the noise level and the condition number in Figure 10 .

Compared with LISTA-CP, LAMP, LISTA-SS, and LISTA-CP-SS, our gated LISTA (GLISTA) performs remarkably better with less ill-posed dictionary matrices and less noises.

Table 3 and 4 report the statistical means and standard deviations of five runs using different methods.

It can be seen that the improvement achieved by our GLISTA is significant.

Chen et al.'s (2018) , and some key steps are listed here: 1) The model is trained progressively to include more layers during the training phase.

At the very beginning, only learnable parameters in the first layer is considered, and parameters in the second layer is only included once training on the first update converges, so as the third and higher layers.

2) Training after including the t-th layer is split into three stages, with an initial learning rate of 0.0005 to optimize its own learnable parameters first, and learning rates of 0.0001 and 0.00001 to jointly optimize all learnable parameters from the 0-th to t-th layers in the second and third stages, respectively.

We move to the next stage once no performance gain is observed on the validation set for 4000 iterations.

3) With the three stages done on the t-th layer, training moves to include the (t + 1)-th and the same three stages of training are performed.

We perform an adaptive overshoot in the experiment to confirm Proposition 2.

The algorithm is summarized in Algorithm 1.

Most of input variables are introduced in the main body of our paper and τ is given as the step size for performing line search.

The whole algorithm procedure is very similar to the famous backtracking line search.

The step size η for sparse coding is updated by τ until the objective function f (x, y) + λr(x) does not decrease any more.

Algorithm 1 ISTA with adaptive overshoot.

Input: The dictionary matrix A, an observation y, an initial step size η 0 = 1.0 for sparse coding, a step size τ = 1.05 for line search, and a maximal number of iteration.

Output: output result

<|TLDR|>

@highlight

We propose gated mechanisms to enhance learned ISTA for sparse coding, with theoretical guarantees on the superiority of the method. 

@highlight

Proposes extensions to LISTA which address underestimation by introducing "gain gates" and including momentum with "overshoot gates", showing improved convergence rates.

@highlight

This paper is focused on solving sparse coding problems using LISTA-type networks by proposing a "gain gating function" to mitigate the weakness of the "no false positive" assumption.