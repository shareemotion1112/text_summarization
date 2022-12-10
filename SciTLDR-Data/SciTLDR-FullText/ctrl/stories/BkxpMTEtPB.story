Recovering sparse conditional independence graphs from data is a fundamental problem in machine learning with wide applications.

A popular formulation of the problem is an $\ell_1$ regularized maximum likelihood estimation.

Many convex optimization algorithms have been designed to solve this formulation to recover the graph structure.

Recently, there is a surge of interest to learn algorithms directly based on data, and in this case, learn to map empirical covariance to the sparse precision matrix.

However, it is a challenging task in this case, since the symmetric positive definiteness (SPD) and sparsity of the matrix are not easy to enforce in learned algorithms, and a direct mapping from data to precision matrix may contain many parameters.

We propose a deep learning architecture, GLAD, which uses an Alternating Minimization (AM) algorithm as our model inductive bias, and learns the model parameters via supervised learning.

We show that GLAD learns a very compact and effective model for recovering sparse graphs from data.

Recovering sparse conditional independence graphs from data is a fundamental problem in high dimensional statistics and time series analysis, and it has found applications in diverse areas.

In computational biology, a sparse graph structure between gene expression data may be used to understand gene regulatory networks; in finance, a sparse graph structure between financial timeseries may be used to understand the relationship between different financial assets.

A popular formulation of the problem is an 1 regularization log-determinant estimation of the precision matrix.

Based on this convex formulation, many algorithms have been designed to solve this problem efficiently, and one can formally prove that under a list of conditions, the solution of the optimization problem is guaranteed to recover the graph structure with high probability.

However, convex optimization based approaches have their own limitations.

The hyperparameters, such as the regularization parameters and learning rate, may depend on unknown constants, and need to be tuned carefully to achieve the recovery results.

Furthermore, the formulation uses a single regularization parameter for all entries in the precision matrix, which may not be optimal.

It is intuitive that one may obtain better recovery results by allowing the regularization parameters to vary across the entries in the precision matrix.

However, such flexibility will lead to a quadratic increase in the number of hyperparameters, but it is hard for traditional approaches to search over a large number of hyperparameters.

Thus, a new paradigm may be needed for designing more effective sparse recovery algorithms.

Recently, there has been a surge of interest in a new paradigm of algorithm design, where algorithms are augmented with learning modules trained directly with data, rather than prescribing every step of the algorithms.

This is meaningful because very often a family of optimization problems needs to be solved again and again, similar in structures but different in data.

A data-driven algorithm may be able to leverage this distribution of problem instances, and learn an algorithm which performs better than traditional convex formulation.

In our case, the sparse graph recovery problem may also need to be solved again and again, where the underlying graphs are different but have similar degree distribution, the magnitude of the precision matrix entries, etc.

For instance, gene regulatory networks may be rewiring depending on the time and conditions, and we want to estimate them from gene

In our experiments, we show that the AM architecture provides very good inductive bias, allowing the model to learn very effective sparse graph recovery algorithm with a small amount of training data.

In all cases, the learned algorithm can recover sparse graph structures with much fewer data points from a new problem, and it also works well in recovering gene regulatory networks based on realistic gene expression data generators.

Related works.

Belilovsky et al. (2017) considers CNN based architecture that directly maps empirical covariance matrices to estimated graph structures.

Previous works have parameterized optimization algorithms as recurrent neural networks or policies in reinforcement learning.

For instance, Andrychowicz et al. (2016) considered directly parameterizing optimization algorithm as an RNN based framework for learning to learn.

Li & Malik (2016) approach the problem of automating algorithm design from reinforcement learning perspective and represent any particular optimization algorithm as a policy.

Khalil et al. (2017) learn combinatorial optimzation over graph via deep Q-learning.

These works did not consider the structures of our sparse graph recovery problem.

Another interesting line of approach is to develop deep neural networks based on unfolding an iterative algorithm Gregor & LeCun (2010) ; ; .

developed ALISTA which is based on unrolling the Iterative Shrinkage Thresholding Algorithm (ISTA).

Sun et al. (2016) developed 'ADMM-Net', which is also developed for compressive sensing of MRI data.

Though these seminal works were primarily developed for compressive sensing applications, they alluded to the general theme of using unrolled algorithms as inductive biases.

We thus identify a suitable unrolled algorithm and leverage its inductive bias to solve the sparse graph recovery problem.

Given m observations of a d-dimensional multivariate Gaussian random variable X = [X 1 , . . .

, X d ] , the sparse graph recovery problem aims to estimate its covariance matrix Σ * and precision matrix Θ * = (Σ * ) −1 .

The ij-th component of Θ * is zero if and only if X i and X j are conditionally independent given the other variables {X k } k =i,j .

Therefore, it is popular to impose an 1 regularization for the estimation of Θ * to increase its sparsity and lead to easily interpretable models.

Following Banerjee et al. (2008) , the problem is formulated as the 1 -regularized maximum likelihood estimation Θ = arg min Θ∈S d ++ − log(det Θ) + tr( ΣΘ) + ρ Θ 1,off ,

where Σ is the empirical covariance matrix based on m samples, S d ++ is the space of d × d symmetric positive definite matrices (SPD), and Θ 1,off = i =j |Θ ij | is the off-diagonal 1 regularizer with regularization parameter ρ.

This estimator is sensible even for non-Gaussian X, since it is minimizing an 1 -penalized log-determinant Bregman divergence Ravikumar et al. (2011) .

The sparse precision matrix estimation problem in Eq. (1) is a convex optimization problem which can be solved by many algorithms.

We give a few canonical and advanced examples which are compared in our experiments: G-ISTA.

G-ISTA is a proximal gradient method, and it updates the precision matrix iteratively

(2) The step sizes ξ k is determined by line search such that Θ k+1 is SPD matrix Rolfs et al. (2012) .

ADMM.

Alternating direction methods of multiplier (Boyd et al., 2011) transform the problem into an equivalent constrained form, decouple the log-determinant term and the 1 regularization term, and result in the following augmented Lagrangian form with a penalty parameter λ:

(3) Taking U := β/λ as the scaled dual variable, the update rules for the ADMM algorithm are

(5) BCD.

Block-coordinate decent methods Friedman et al. (2008) updates each column (and the corresponding row) of the precision matrix iteratively by solving a sequence of lasso problems.

The algorithm is very efficient for large scale problems involving thousands of variables.

Apart from various algorithms, rigorous statistical analysis has also been provided for the optimal solution of the convex formulation in Eq. (1).

Ravikumar et al. (2011) established consistency of the estimator Θ in Eq. (1) in terms of both Frobenius and spectral norms, at rate scaling roughly as

with high probability, where s is the number of nonzero entries in Θ * .

This statistical analysis also reveal certain limitations of the convex formulation:

The established consistency is based on a set of carefully chosen conditions, including the lower bound of sample size, the sparsity level of Θ * , the degree of the graph, the magnitude of the entries in the covariance matrix, and the strength of interaction between edge and non-edge in the precision matrix (or mutual incoherence on the Hessian Γ * := Σ * ⊗ Σ * ) .

In practice, it may be hard to a problem to satisfy these recovery conditions.

Therefore, it seems that there is still room for improving the above convex optimization algorithms for recovering the true graph structure.

Prior to the data-driven paradigm for sparse recovery, since the target parameter Θ * is unknown, the best precision matrix recovery method is to resort to a surrogate objective function (for instance, equation 1).

Optimally tuning the unknown parameter ρ is a very challenging problem in practice.

Instead, we can leverage the large amount of simulation or real data and design a learning algorithm that directly optimizes the loss in equation 9.

Furthermore, since the log-determinant estimator in Eq. (1) is NOT directly optimizing the recovery objective Θ − Θ * 2 F , there is also a mismatch in the optimization objective and the final evaluation objective (refer to the first experiment in section 5.1).

This increase the hope one may improve the results by directly optimizing the recovery objective with the algorithms learned from data.

In the remainder of the paper, we will present a data-driven method to learn an algorithm for precision matrix estimation, and we call the resulting algorithm GLAD (stands for Graph recovery Learning Algorithm using Data-driven training).

We ask the question of Given a family of precision matrices, is it possible to improve recovery results for sparse graphs by learning a data-driven algorithm?

More formally, suppose we are given n precision matrices {Θ * (i) } n i=1 from a family G of graphs and m samples {x (i,j) } m j=1 associated with each Θ * (i) .

These samples can be used to form n sample

.

We are interested in learning an algorithm for precision matrix estimation by solving a supervised learning problem,

, where f is a set of parameters in GLAD(·) and the output of GLAD f ( Σ (i) ) is expected to be a good estimation of Θ * (i) in terms of an interested evaluation metric L. The benefit is that it can directly optimize the final evaluation metric which is related to the desired structure or graph properties of a family of problems.

However, it is a challenging task to design a good parameterization of GLAD f for this graph recovery problem.

We will explain the challenges below and then present our solution.

In the literature on learning data-driven algorithms, most models are designed using traditional deep learning architectures, such as fully connected DNN or recurrent neural networks.

But, for graph recovery problems, directly using these architectures does not work well due to the following reasons.

First, using a fully connected neural network is not practical.

Since both the input and the output of graph recovery problems are matrices, the number of parameters scales at least quadratically in d. Such a large number of parameters will need many input-output training pairs to provide a decent estimation.

Thus some structures need to be imposed in the network to reduce the size of parameters and sample complexity.

Second, structured models such as convolution neural networks (CNNs) have been applied to learn a mapping from Σ to Θ * (Belilovsky et al., 2017) .

Due to the structure of CNNs, the number of parameters can be much smaller than fully connected networks.

However, a recovered graph should be permutation invariant with respect to the matrix rows/columns, and this constraint is very hard to be learned by CNNs, unless there are lots of samples.

Also, the structure of CNN is a bias imposed on the model, and there is no guarantee why this structure may work.

Third, the intermediate results produced by both fully connected networks and CNNs are not interpretable, making it hard to diagnose the learned procedures and progressively output increasingly improved precision matrix estimators.

Fourth, the SPD constraint is hard to impose in traditional deep learning architectures.

Although, the above limitations do suggest a list of desiderata when designing learning models: Small model size; Minimalist learning; Interpretable architecture; Progressive improvement; and SPD output.

These desiderata will motivate the design of our deep architecture using unrolled algorithms.

To take into account the above desiderata, we will use an unrolled algorithm as the template for the architecture design of GLAD.

The unrolled algorithm already incorporates some problem structures, such as permutation invariance and interpretable intermediate results; but this unrolled algorithm does not traditionally have a learning component, and is typically not directly suitable for gradient-based approaches.

We will leverage this inductive bias in our architecture design and augment the unrolled algorithm with suitable and flexible learning components, and then train these embedded models with stochastic gradient descent.

GLAD model is based on a reformulation of the original optimization problem in Eq. (1) with a squared penalty term, and an alternating minimization (AM) algorithm for it.

More specifically, we consider a modified optimization with a quadratic penalty parameter λ:

and the alternating minimization (AM) method for solving it:

where η ρ/λ (θ) := sign(θ) max(|θ| − ρ/λ, 0).

The derivation of these steps are given in Appendix A. We replace the penalty constants (ρ, λ) by problem dependent neural networks, ρ nn and Λ nn .

These neural networks are minimalist in terms of the number of parameters as the input dimensions are mere {3, 2} for {ρ nn , Λ nn } and outputs a single value.

Algorithm 1 summarizes the update equations for our unrolled AM based model, GLAD.

Except for the parameters in ρ nn and Λ nn , the constant t for initialization is also a learnable scalar parameter.

This unrolled algorithm with neural network augmentation can be viewed as a highly structured recurrent architecture as illustrated in Figure 1 .

There are many traditional algorithms for solving graph recovery problems.

We choose AM as our basis because: First, empirically, we tried models built upon other algorithms including G-ISTA, ADMM, etc, but AM-based model gives consistently better performances.

Appendix C.10 & C.11 discusses different parameterizations tried.

Second, and more importantly, the AM-based architecture has a nice property of maintaining Θ k+1 as a SPD matrix throughout the iterations as long as λ k < ∞. Third, as we prove later in Section 4, the AM algorithm has linear convergence rate, allowing us to use a fixed small number of iterations and still achieve small error margins.

Algorithm 1: GLAD

To learn the parameters in GLAD architecture, we will directly optimize the recovery objective function rather than using log-determinant objective.

A nice property of our deep learning architecture is that each iteration of our model will output a valid precision matrix estimation.

This allows us to add auxiliary losses to regularize the intermediate results of our GLAD architecture, guiding it to learn parameters which can generate a smooth solution trajectory.

Specifically, we will use Frobenius norm in our experiments, and design an objective which has some resemblance to the discounted cumulative reward in reinforcement learning:

where (Θ

is the output of the recurrent unit GLADcell at k-th iteration, K is number of unrolled iterations, and γ ≤ 1 is a discounting factor.

We will use stochastic gradient descent algorithm to train the parameters f in the GLADcell.

A key step in the gradient computation is to propagate gradient through the matrix square root in the GLADcell.

To do this efficiently, we make use of the property of SPD matrix that X = X 1/2 X 1/2 , and the product rule of derivatives to obtain

The above equation is a Sylvester's equation for d(X 1/2 ).

Since the derivative dX for X is easy to obtain, then the derivative of d(X 1/2 ) can be obtained by solving the Sylvester's equation in (10).

The objective function in equation 9 should be understood in a similar way as in Gregor & LeCun (2010) ; Belilovsky et al. (2017); where deep architectures are designed to directly produce the sparse outputs.

For GLAD architecture, a collection of input covariance matrix and ground truth sparse precision matrix pairs are available during training, either coming from simulated or real data.

Thus the objective function in equation 9 is formed to directly compare the output of GLAD with the ground truth precision matrix.

The goal is to train the deep architecture which can perform well for a family/distribution of input covariance matrix and ground truth sparse precision matrix pairs.

The average in the objective function is over different input covariance and precision matrix pairs such that the learned architecture is able to perform well over a family of problem instances.

Furthermore, each layer of our deep architecture outputs an intermediate prediction of the sparse precision matrix.

The objective function takes into account all these intermediate outputs, weights the loss according to the layer of the deep architecture, and tries to progressively bring these intermediate layer outputs closer and closer to the target ground truth.

We note that the designed architecture, is more flexible than just learning the regularization parameters.

The component in GLAD architecture corresponding to the regularization parameters are entry-wise and also adaptive to the input covariance matrix and the intermediate outputs.

GLAD architecture can adaptively choose a matrix of regularization parameters.

This task will be very challenging if the matrix of regularization parameters are tuned manually using cross-validation.

A recent theoretical work Sun et al. (2018) also validates the choice of GLAD's design.

Since GLAD architecture is obtained by augmenting an unrolled optimization algorithm by learnable components, the question is what kind of guarantees can be provided for such learned algorithm, and whether learning can bring benefits to the recovery of the precision matrix.

In this section, we will first analyze the statistical guarantee of running the AM algorithm in Eq. (7) and Eq. (8) for k steps with a fixed quadratic penalty parameter λ, and then interpret its implication for the learned algorithm.

First, we need some standard assumptions about the true model from the literature Rothman et al. (2008):

The assumption 2 guarantees that Θ * exists.

Assumption 1 just upper bounds the sparsity of Θ * and does not stipulate anything in particular about s. These assumptions characterize the fundamental limitation of the sparse graph recovery problem, beyond which recovery is not possible.

Under these assumptions, we prove the linear convergence of AM algorithm (proof is in Appendix B).

m , where ρ is the l 1 penalty, d is the dimension of problem and m is the number of samples, the Alternate Minimization algorithm has linear convergence rate for optimization objective defined in (6).

The k th iteration of the AM algorithm satisfies,

where 0 < C λ < 1 is a constant depending on λ.

From the theorem, one can see that by optimizing the quadratic penalty parameter λ, one can adjust the C λ in the bound.

We observe that at each stage k, an optimal penalty parameter λ k can be chosen depending on the most updated value C λ .

An adaptive sequence of penalty parameters (λ 1 , . . .

, λ K ) should achieve a better error bound compared to a fixed λ.

Since C λ is a very complicated function of λ, the optimal λ k is hard to choose manually.

Besides, the linear convergence guarantee in this theorem is based on the sparse regularity parameter ρ log d m .

However, choosing a good ρ value in practice is tedious task as shown in our experiments.

In summary, the implications of this theorem are:

• An adaptive sequence (λ 1 , . . . , λ K ) should lead to an algorithm with better convergence than a fixed λ, but the sequence may not be easy to choose manually.

• Both ρ and the optimal λ k depend on the corresponding error Θ AM − Θ λ F , which make these parameters hard to prescribe manually.

• Since, the AM algorithm has a fast linear convergence rate, we can run it for a fixed number of iterations K and still converge with a reasonable error margin.

Our learning augmented deep architecture, GLAD, can tune these sequence of λ k and ρ parameters jointly using gradient descent.

Moreover, we refer to a recent work by Sun et al. (2018) where they considered minimizing the graphical lasso objective with a general nonconvex penalty.

They showed that by iteratively solving a sequence of adaptive convex programs one can achieve even better error margins (refer their Algorithm 1 & Theorem 3.5).

In every iteration they chose an adaptive regularization matrix based on the most recent solution and the choice of nonconvex penalty.

We thus hypothesize that we can further improve our error margin if we make the penalty parameter ρ nonconvex and problem dependent function.

We choose ρ as a function depending on the most up-todate solution (Θ k , Σ, Z k ), and allow different regularizations for different entries of the precision matrix.

Such flexibility potentially improves the ability of GLAD model to recover the sparse graph.

In this section, we report several experiments to compare GLAD with traditional algorithms and other data-driven algorithms.

The results validate the list of desiderata mentioned previously.

Especially, it shows the potential of pushing the boundary of traditional graph recovery algorithms by utilizing data.

Python implementation (tested on P100 GPU) is available 1 .

Exact experimental settings details are covered in Appendix C. Evaluation metric.

We use normalized mean square error (NMSE) and probability of success (PS) to evaluate the algorithm performance.

NMSE is 10 log 10 (E Θ p − Θ * 2 F /E Θ * 2 F ) and PS is the probability of correct signed edge-set recovery, i.e., P sign(

, where E(Θ * ) is the true edge set.

Notation.

In all reported results, D stands for dimension d of the random variable, M stands for sample size and N stands for the number of graphs (precision matrices) that is used for training.

Inconsistent optimization objective.

Traditional algorithms are typically designed to optimize the 1 -penalized log likelihood.

Since it is a convex optimization, convergence to optimal solution is usually guaranteed.

However, this optimization objective is different from the true error.

Taking ADMM as an example, it is revealed in Figure 2 that, although the optimization objective always converges, errors of recovering true precision matrices measured by NMSE have very different behaviors given different regularity parameter ρ, which indicates the necessity of directly optimizing NMSE and hyperparameter tuning.

Expensive hyperparameter tuning.

Although hyperparameters of traditional algorithms can be tuned if the true precision matrices are provided as a validation dataset, we want to emphasize that hyperparamter tuning by grid search is a tedious and hard task.

Table 1 shows that the NMSE values are very sensitive to both ρ and the quadratic penalty λ of ADMM method.

For instance, the optimal NMSE in this table is −9.61 when λ = 0.1 and ρ = 0.03.

However, it will increase by a large amount to −2.06 if ρ is only changed slightly to 0.01.

There are many other similar observations in this table, where slight changes in parameters can lead to significant NMSE differences, which in turns makes grid-search very expensive.

G-ISTA and BCD follow similar trends.

For a fair comparison against GLAD which is data-driven, in all following experiments, all hyperparameters in traditional algorithms are fine-tuned using validation datasets, for which we spent extensive efforts (See more details in Appendix C.3, C.6).

In contrast, the gradient-based training of GLAD turns out to be much easier.

We follow the experimental setting in (Rolfs et al., 2012; Mazumder & Agarwal, 2011; Lu, 2010) to generate data and perform synthetic experiments on multivariate Gaussians.

Each offdiagonal entry of the precision matrix is drawn from a uniform distribution, i.e., Θ * ij ∼ U(−1, 1), and then set to zero with probability p = 1 − s, where s means the sparsity level.

Finally, an appropriate multiple of the identity matrix was added to the current matrix, so that the resulting matrix had the smallest eigenvalue as 1 (refer to Appendix C.1).

We use 30 unrolled steps for GLAD (Figure 3) and compare it to G-ISTA, ADMM and BCD.

All algorithms are trained/finetuned using 10 randomly generated graphs and tested over 100 graphs.

Convergence results and average runtime of different algorithms on Nvidia's P100 GPUs are shown in Figure 4 and Table 2 respectively.

GLAD consistently converges faster and gives lower NMSE.

Although the fine-tuned G-ISTA also has decent performance, the computation time in each iteration is much longer than GLAD because it requires line search steps.

Besides, we could also see a progressive improvement of GLAD across its iterations.

As analyzed by Ravikumar et al. (2011) , the recovery guarantee (such as in terms of Frobenius norm) of the 1 regularized log-determinant optimization significantly depends on the sample size and other conditions.

Our GLAD directly optimizes the recovery objective based on data, and it has the potential of pushing the sample complexity limit.

We experimented with this and found the results positive.

We follow Ravikumar et al. (2011) to conduct experiments on GRID graphs, which satisfy the conditions required in (Ravikumar et al., 2011) .

Furthermore, we conduct a more challenging task of recovering restricted but randomly constructed graphs (see Appendix C.7 for more details).

The probability of success (PS) is non-zero only if the algorithm recovers all the edges with correct signs, plotted in Figure 5 .

GLAD consistently outperforms traditional methods in terms of sample complexity as it recovers the true edges with considerably fewer number of samples.

Having a good inductive bias makes GLAD's architecture quite data-efficient compared to other deep learning models.

For instance, the state-of-the-art 'DeepGraph' by Belilovsky et al. (2017) is based on CNNs.

It contains orders of magnitude more parameters than GLAD.

Furthermore, it takes roughly 100, 000 samples, and several hours for training their DG-39 model.

In contrast, GLAD learns well with less than 25 parameters, within 100 training samples, and notably less training time.

Table 3 also shows that GLAD significantly outperforms DG-39 model in terms of AUC (Area under the ROC curve) by just using 100 training graphs, typically the case for real world settings.

Fully connected DL models are unable to learn from such small data and hence are skipped in the comparison.

Figure 6 shows that GLAD performs favourably for structure recovery in terms of NMSE on the gene expression data.

As the governing equations of the underlying distribution of the SynTReN are unknown, these experiments also emphasize the ability of GLAD to handle non-Gaussian data.

Figure 7 visualizes the edge-recovery performance of GLAD models trained on a sub-network of true Ecoli bacteria data.

We denote, TPR: True Positive Rate, FPR: False Positive Rate, FDR: False Discovery Rate.

The number of simulated training/validation graphs were set to 20/20.

One batch of M samples were taken per graph (details in Appendix C.9).

Although, GLAD was trained on graphs with D = 25, it was able to robustly recover a higher dimensional graph D = 43 structure.

Appendix C.12 contains details of the experiments done on real E.Coli data.

The GLAD model was trained using the SynTReN simulator.

Appendix C.13 explains our proposed approach to scale for larger problem sizes.

We presented a novel neural network, GLAD, for the sparse graph recovery problem based on an unrolled Alternating Minimization algorithm.

We theoretically prove the linear convergence of AM algorithm as well as empirically show that learning can further improve the sparse graph recovery.

The learned GLAD model is able to push the sample complexity limits thereby highlighting the potential of using algorithms as inductive biases for deep learning architectures.

Further development of theory is needed to fully understand and realize the potential of this new direction.

Alternating Minimization is performing

Taking the gradient of the objective function with respect to Θ to be zero, we have

Taking the gradient of the objective function with respect to Z to be zero, we have

where

Solving the above two equations, we obtain:

where

B LINEAR CONVERGENCE RATE ANALYSIS m , where ρ is the l 1 penalty, d is the dimension of problem and m is the number of samples, the Alternate Minimization algorithm has linear convergence rate for optimization objective defined in (6).

The k th iteration of the AM algorithm satisfies,

where 0 < C λ < 1 is a constant depending on λ.

We will reuse the following notations in the appendix:

The update rules for Alternating Minimization are:

Assumptions: With reference to the theory developed in Rothman et al. (2008), we make the following assumptions about the true model.

(O P (·) is used to denote bounded in probability.)

We now proceed towards the proof: Lemma 2.

For any x, y, k ∈ R, k > 0, x = y,

Proof.

where

is the largest eigenvalue of X in absolute value.

Proof.

First we factorize X using eigen decomposition, X = Q X D X Q X , where Q X and D X are orthogonal matrix and diagonal matrix, respectively.

Then we have,

Similarly, the above equation holds for Y .

Therefore,

where we define Q := Q Y Q X .

Similarly, we have,

Then the i-th entry on the diagonal of

ji .

Using the fact that D X and D Y are diagonal, we have,

The last step makes use of

Similarly, using (42), we have,

Assuming X − Y F > 0 (otherwise (37) trivially holds), using (52) and (50), we have,

Using lemma (2), we have,

Therefore,

Lemma 4.

Under assumption (2), the output of the k-th and

where 0 < C λ < 1 is a constant depending on λ.

Proof.

The first part is easy to show, if we observe that in the second update step of AM (8), η ρ/λ is a contraction under metric d(X, Y ) = X − Y F .

Therefore we have,

Next we will prove the second part.

To simplify notation, we let A(X) = X X + 4 λ I. Using the first update step of AM (7), we have,

where

The last derivation step makes use of the triangle inequality.

Using lemma (3), we have,

Therefore

where

Λ max (X) is the largest eigenvalue of X in absolute value.

The rest is to show that both Λ max (Y λ ) and Λ max (Y k+1 ) are bounded using assumption (2).

For Λ max (Y k+1 ), we have,

Combining (62) and (68), we have,

Therefore,

Continuing with (73), we have,

Since Z λ is the minimizer of a strongly convex function, its norm is bounded.

And we also have

Therefore both Λ max (Y λ ) and Λ max (Y k+1 ) are bounded in (70), i.e. 0 < C λ < 1 is a constant only depending on λ.

m , where ρ is the l 1 penalty, d is the dimension of problem and m is the number of samples, the Alternate Minimization algorithm has linear convergence rate for optimization objective defined in (6).

The k th iteration of the AM algorithm satisfies,

where 0 < C λ < 1 is a constant depending on λ.

Proof.

(1) Error between Θ λ and Θ G Combining the following two equations:

Note that by the optimality condition, ∇ z f ( Θ λ , Z λ , ρ, λ) = 0, we have the fixed point equation

λ and we have:

Since G is σ G -strongly convex, where σ G is independent of the sample covariance matrix Σ * as the hessian of G is independent of Σ * .

Therefore,

Proof.

(2) Error between Θ G and Θ * Corollary 5 (Theorem 1. of Rothman et al. (2008)).

Let Θ G be the minimizer for the optimization

C EXPERIMENTAL DETAILS This section contains the detailed settings used in the experimental evaluation section.

For sections 5.1 and 5.2, the synthetic data was generated based on the procedure described in Rolfs et al. (2012) .

A d dimensional precision matrix Θ was generated by initializing a d × d matrix with its off-diagonal entries sampled i.i.d.

from a uniform distribution Θ ij ∼ U(−1, 1).

These entries were then set to zero based on the sparsity pattern of the corresponding Erdos-Renyi random graph with a certain probability p.

Finally, an appropriate multiple of the identity matrix was added to the current matrix, so that the resulting matrix had the smallest eigenvalue as 1.

In this way, Θ was ensured to be [20, 100, 500] .

The top row has the sparsity probability p = 0.5 for the Erdos-Renyi random graph, whereas for the bottom row plots, the sparsity probabilities are uniformly sampled from ∼ U(0.05, 0.15).

For finetuning the traditional algorithms, a validation dataset of 10 graphs was used.

For the GLAD algorithm, 10 training graphs were randomly chosen and the same validation set was used.

C.5 GLAD: ARCHITECTURE DETAILS FOR SECTION(5.2) GLAD parameter settings: ρ nn was a 4 layer neural network and Λ nn was a 2 layer neural network.

Both used 3 hidden units in each layer.

The non-linearity used for hidden layers was tanh, while the final layer had sigmoid (σ) as the non-linearity for both, ρ nn and Λ nn (refer Figure 3) .

The learnable offset parameter of initial Θ 0 was set to t = 1.

It was unrolled for L = 30 iterations.

The learning rates were chosen to be around [0.01, 0.1] and multi-step LR scheduler was used.

The optimizer used was 'adam'.

The best nmse model was selected based on the validation data performance.

Figure( Figure 9: We attempt to illustrate how the traditional methods are very sensitive to the hyperparameters and it is a tedious exercise to finetune them.

The problem setting is same as described in section(5.3).

For all the 3 methods shown above, we have already tuned the algorithm specific parameters to a reasonable setting.

Now, we vary the L 1 penalty term ρ and can observe that how sensitive the probability of success is with even slight change of ρ values.

values are very sensitive to the choice of t as well.

These parameter values changes substantially for a new problem setting.

G-ISTA and BCD follow similar trends.

Additional plots highlighting the hyperparameter sensitivity of the traditional methods for model selection consistency experiments.

Refer figure(9).

Details for experiments in figure(5).

Two different graph types were chosen for this experiment which were inspired from Ravikumar et al. (2011) .

In the 'grid' graph setting, the edge weight for different precision matrices were uniformly sampled from w ∼ U(0.12, 0.25).

The edges within a graph carried equal weights.

The other setting was more general, where the graph was a random Erdos-Renyi graph with probability of an edge was p = 0.05.

The off-diagonal entries of the precision matrix were sampled uniformly from ∼ U[0.1, 0.4].

The parameter settings for GLAD were the same as described in Appendix C.5.

The model with the best PS performance on the validation dataset was selected.

train/valid/test=10/10/100 graphs were used with 10 sample batches per graph.

C.8 GLAD: COMPARISON WITH OTHER DEEP LEARNING BASED METHODS Table( 3) shows AUC (with std-err) comparisons with the DeepGraph model.

For experiment settings, refer Table 1 of Belilovsky et al. (2017) .

Gaussian Random graphs with sparsity p = 0.05 were chosen and edge values sampled from ∼ U(−1, 1).

GLAD was trained on only 10 graphs with 5 sample batches per graph.

The dimension of the problem is D = 39.

The architecture parameter choices of GLAD were the same as described in Appendix C.5 and it performs consistently better along all the settings by a significant AUC margin.

The SynTReN Van den Bulcke et al. (2006) is a synthetic gene expression data generator specifically designed for analyzing the structure learning algorithms.

The topological characteristics of the synthetically generated networks closely resemble the characteristics of real transcriptional networks.

The generator models different types of biological interactions and produces biologically plausible synthetic gene expression data enabling the development of data-driven approaches to recover the underlying network.

The SynTReN simulator details for section(5.5).

For performance evaluation, a connected ErdosRenyi graph was generated with probability as p = 0.05.

The precision matrix entries were sampled from Θ ij ∼ U(0.1, 0.2) and the minimum eigenvalue was adjusted to 1 by adding an appropriate multiple of identity matrix.

The SynTReN simulator then generated samples from these graphs by incorporating biological noises, correlation noises and other input noises.

All these noise levels were sampled uniformly from ∼ U(0.01, 0.1).

The figure(6) shows the NMSE comparisons for a fixed dimension D = 25 and varying number of samples M = [10, 25, 100] .

The number of training/validation graphs were set to 20/20 and the results are reported on 100 test graphs.

In these experiments, only 1 batch of M samples were taken per graph to better mimic the real world setting.

Figure (7) Unrolled model for ADMM: Algorithm 2 describes the unrolled model ADMMu updates.

ρ nn was a 4 layer neural network and Λ nn was a 2 layer neural network.

Both used 3 hidden units in each layer.

The non-linearity used for hidden layers was tanh, while the final layer had sigmoid (σ) as the non-linearity for both ,ρ nn and Λ nn .

The learnable offset parameter of initial Θ 0 was set to t = 1.

It was unrolled for L = 30 iterations.

The learning rates were chosen to be around [0.01, 0.1] and multi-step LR scheduler was used.

The optimizer used was 'adam'.

Figure 10 compares GLAD with ADMMu on the convergence performance with respect to synthetically generated data.

The settings were kept same as described in Figure 4 .

As evident from the plots, we see that GLAD consistently performs better than ADMMu.

We had similar observations for other set of experiments as well.

Hence, we chose AM based unrolled algorithm over ADMM's as it works better empirically and has less parameters.

Although, we are not entirely confident but we hypothesize the reason for above observations as follows.

In the ADMM update equations (4 & 5), both the Lagrangian term and the penalty term are intuitively working together as a 'function' to update the entries Θ ij , Z ij .

Observe that U k can be absorbed into Z k and/or Θ k and we expect our neural networks to capture this relation.

We thus expect GLAD to work at least as good as ADMMu.

In our formulation of unrolled ADMMu (Algorithm 2) the update step of U is not controlled by neural networks (as the number of parameters needed will be substantially larger) which might be the reason of it not performing as well as GLAD.

Our empirical evaluations corroborate this logic that just by using the penalty term we can maintain all the desired properties and learn the problem dependent 'functions' with a small neural network.

We tried multiple unrolled parameterizations of the optimization techniques used for solving the graphical lasso problem which worked to varying levels of success.

We list here a few, in interest for helping researchers to further pursue this recent and novel approach of data-driven algorithm designing.

1. ADMM + ALISTA parameterization: The threshold update for Z AM k+1 can be replaced by ALISTA network .

The stage I of ALISTA is determining W, which is trivial in our case as D = I. So, we get W = I. Thus, combining ALISTA updates along with AM's we get an interesting unrolled algorithm for our optimization problem.

All the settings are same as the fixed sparsity case described in Figure 4 .

We see that the AM based parameterization 'GLAD' consistently performs better than the ADMM based unrolled architecture 'ADMMu'.

2.

G-ISTA parameterization: We parameterized the line search hyperparameter c as well as replaced the next step size determination step by a problem dependent neural network of Algorithm (1) in Rolfs et al. (2012) .

The main challenge with this parameterization is to main the PSD property of the intermediate matrices obtained.

Learning appropriate parameterization of line search hyperparameter such that PSD condition is maintained remains an interesting aspect to investigate.

3.

Mirror Descent Net: We get a similar set of update equations for the graphical lasso optimization.

We identify some learnable parameters, use neural networks to make them problem dependent and train them end-to-end.

4.

For all these methods we also tried unrolling the neural network as well.

In our experience we found that the performance does not improve much but the convergence becomes unstable.

We use the real data from the 'DREAM 5 Network Inference challenge' (Marbach et al., 2012) .

This dataset contains 3 compendia that were obtained from microorganisms, some of which are pathogens of clinical relevance.

Each compendium consists of hundreds of microarray experiments, which include a wide range of genetic, drug, and environmental perturbations.

We test our method for recovering the true E.coli network from the gene expression values recorded by doing actual microarray experiments.

The E.coli dataset contains 4511 genes and 805 associated microarray experiments.

The true underlying network has 2066 discovered edges and 150214 pairs of nodes do not have an edge between them.

There is no data about the remaining edges.

For our experiments, we only consider the discovered edges as the ground truth, following the challenge data settings.

We remove the genes that have zero degree and then we get a subset of 1081 genes.

For our predictions, we ignore the direction of the edges and only consider retrieving the connections between genes.

We train the GLAD model using the SynTReN simulator on the similar settings as described in Appendix C.9.

Briefly, GLAD model was trained on D=50 node graphs sampled from Erdos-Renyi graph with sparsity probability ∼ U (0.01, 0.1), noise levels of SynTReN simulator sampled from ∼ U (0.01, 0.1) and Θ ij ∼ U (0.1, 0.2)).

The model was unrolled for 15 iterations.

This experiment also evaluates GLAD's ability to generalize to different distribution from training as well as scaling ability to more number of nodes.

We report the AUC scores for E.coli network in Table 4 .

We can see that GLAD improves over the other competing methods in terms of Area Under the ROC curve (AUC).

We understand that it is challenging to model real datasets due to the presence of many unknown latent extrinsic factors, but we do observe an advantage of using data-driven parameterized algorithm approaches.

Methods BCD GISTA GLAD AUC 0.548 0.541 0.572 We have shown in our experiments that we can train GLAD on smaller number of nodes and get reasonable results for recovering graph structure with considerably larger nodes (AppendixC.12).

Thus, in this section, we focus on scaling up on the inference/test part.

With the current GPU implementation, we can can handle around 10,000 nodes for inference.

For problem sizes with more than 100,000 nodes, we propose to use the randomized algorithm techniques given in Kannan & Vempala (2017) .

Kindly note that scaling up GLAD is our ongoing work and we just present here one of the directions that we are exploring.

The approach presented below is to give some rough idea and may contain loose ends.

Randomized algorithms techniques are explained elaborately in Kannan & Vempala (2017) .

Specifically, we will use some of their key results

• P1. (Theorem 2.1) We will use the length-squared sampling technique to come up with low-rank approximations • P2. (Theorem 2.5) For any large matrix A ∈ R m×n , we can use approximate it as A ≈ CU R , where C ∈ R m×r , U ∈ R s×r , R ∈ R r×m .

• P3. (Section 2.3) For any large matrix A ∈ R m×n , we can get its approximate SVD by using the property E(R T R) = A T A where R is a matrix obtained by length-squared sampling of the rows of matrix A.

The steps for doing approximate AM updates, i.e. of equations (7, 8) .

Using property P3, we can approximate

where V is the right singular vectors of R. Thus, we can combine this approximation with the sketch matrix approximation of Y ≈ CU R to calculate the update in equation (7).

Equation (8) is just a thresholding operation and can be done efficiently with careful implementation.

We are looking in to the experimental as well as theoretical aspects of this approach.

We are also exploring an efficient distributed algorithm for GLAD.

We are investigating into parallel MPI based algorithms for this task (https://stanford.edu/~boyd/admm.html is a good reference point).

We leverage the fact that the size of learned neural networks are very small, so that we can duplicate them over all the processors.

This is also an interesting future research direction.

<|TLDR|>

@highlight

A data-driven learning algorithm based on unrolling the Alternating Minimization optimization for sparse graph recovery.