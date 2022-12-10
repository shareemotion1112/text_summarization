This paper addresses the problem of representing a system's belief using multi-variate normal distributions (MND) where the underlying model is based on a deep neural network (DNN).

The major challenge with DNNs is the computational complexity that is needed to obtain model uncertainty using MNDs.

To achieve a scalable method, we propose a novel approach that expresses the parameter posterior in sparse information form.

Our inference algorithm is based on a novel Laplace Approximation scheme, which involves a diagonal correction of the Kronecker-factored eigenbasis.

As this makes the inversion of the information matrix intractable - an operation that is required for full Bayesian analysis, we devise a low-rank   approximation of this eigenbasis and a memory-efficient sampling scheme.

We provide both a theoretical analysis and an empirical evaluation on various benchmark data sets, showing the superiority of our approach over existing methods.

Whenever machine learning methods are used for safety-critical applications such as medical image analysis or autonomous driving, it is crucial to provide a precise estimation of the failure probability of the learned predictor.

Therefore, most of the current learning approaches return distributions rather than single, most-likely predictions.

For example, DNNs trained for classification usually use the softmax function to provide a distribution over predicted class labels.

Unfortunately, this method tends to severely underestimate the true failure probability, leading to overconfident predictions (Guo et al., 2017) .

The main reason for this is that neural networks are typically trained with a principle of maximum likelihood, neglecting their epistemic or model uncertainty with the point estimates.

A widely known work by Gal (2016) shows that this can be mitigated by using dropout at test time.

This so-called Monte-Carlo dropout (MC-dropout) has the advantage that it is relatively easy to use and therefore very popular in practice.

However, MC-dropout also has significant drawbacks.

First, it requires a specific stochastic regularization during training.

This limits its use on already well trained architectures, because current networks are often trained with other regularization techniques such as batch normalization.

Moreover, it uses a Bernoulli distribution to represent the complex model uncertainty, which in return, leads to an underestimation of the predictive uncertainty.

Several strong alternatives exist without these drawbacks.

Variational inference Kingma et al., 2015; Graves, 2011) and expectation propagation (Herandez-Lobato & Adams, 2015) are such examples.

Yet, these methods use a diagonal covariance matrix which limits their applicability as the model parameters are often highly correlated.

Building upon these, Sun et al. (2017) ; Louizos & Welling (2016) ; Zhang et al. (2018) ; Ritter et al. (2018a) show that the correlations between the parameters can also be computed efficiently by decomposing the covariance matrix of MND into Kronecker products of smaller matrices.

However, not all matrices can be Kronecker decomposed and thus, these simplifications usually induce crude approximations (Bae et al., 2018) .

As the dimensionality of statistical manifolds are prohibitively too large in DNNs, more expressive, efficient but still easy to use ways of representing such high dimensional distributions are required.

To tackle this challenge, we propose to represent the model uncertainty in sparse information form of MND.

As a first step, we devise a new Laplace Approximation (LA) for DNNs, in which we improve the state-of-the-art Kronecker factored approximations of the Hessian (George et al., 2018) by correcting the diagonal variance in parameter space.

We show that these can be computed efficiently, and that the information matrix of the resulting parameter posterior is more accurate in terms of the Frobenius norm.

In this way the model uncertainty is approximated in information form of the MND.

counts [-] Figure 1: Main idea.

(a) Covariance matrix Σ for DNNs is intractable to infer, store and sample (an example taken from our MNIST experiments).

(b) Our main insight is that the spectrum (eigenvalues) of information matrix (inverse of covariance) tend to be sparse.

(c) Exploiting this insight a Laplace Approximation scheme is devised which applies a spectral sparsification (LRA) while keeping the diagonals exact.

With this formulation, the complexity becomes tractable for sampling while producing more accurate estimates.

Here, the diagonal elements (nodes in graphical interpretation) corresponds to information content in a parameter whereas the corrections (links) are the off-diagonals.

As this results in intractable inverse operation for sampling, we further propose a novel low-rank representation of the resulting Kronecker factorization, which paves the way to applications on large network structures trained on realistically sized data sets.

To realize such sparsification, we propose a novel algorithm that enables a low-rank approximation of the Kronecker factored eigenvalue decomposition, and we demonstrate an associated sampling computations.

Our experiments demonstrate that our approach is effective in providing more accurate uncertainty estimates and calibration on considered benchmark data sets.

A detailed theoretical analysis is also provided for further insights.

We summarize our main contributions below.

• A novel Laplace Approximation scheme with a diagonal correction to the eigenvalue rescaled approximations of the Hessian, as a practical inference tool (section 2.2).

• A novel low-rank representation of Kronecker factored eigendecomposition that preserves Kronecker structure (section 2.3).

This results in a sparse information form of MND.

• A novel algorithm to enable a low rank approximation (LRA) for the given representation of MND (algorithm 1) and derivation of a memory-wise tractable sampler (section B.2).

• Both theoretical (section C) and experimental results (section 4) showing the applicability of our approach.

In our experiments, we showcase the state-of-the-art performance within the class of Bayesian Neural Networks that are scalable and training-free.

To our knowledge we explore a sparse information form to represent the model uncertainty of DNNs for the first time.

Figure 1 depicts our main idea which we provide more rigorous formulation next.

We model a neural network as a parameterized function f θ : R N 1 → R N l where θ ∈ R N θ are the weights and N θ = N 1 + · · · + N l .

This function f θ is in fact a concatenation of l layers, where each layer i ∈ {1, ..., l} computes h i = W i a i−1 and a i = φ(h i−1 ).

Here, φ is a nonlinear function, a i are activations, h i linear pre-activations, and W i are weight matrices.

The bias terms are absorbed into W i by appending 1 to each a i .

Thus, θ = vec(W 1 )

where vec is the operator that stacks the columns of a matrix to a vector.

Let g i = δh i , the gradient of h i w.r.t θ.

Using LA the posterior is approximated with a Gaussian.

The mean is then given by the MAP estimate θ MAP and the covariance by the Hessian of the log-likelihood (H + τI) −1 assuming a Gaussian prior with precision τ.

Using loss functions such as MSE or cross entropy and piece-wise linear activation a i Figure 2 : Sparse Information Matrix.

We perform a low rank approximation on Kronecker factored eigendecomposition that preserves Kronecker structure in eigenvectors for two reasons: (a) reducing directly (U A ⊗ U G ) 1:L is memory-wise infeasible, and (b) sampling scheme then only involves matrix multiplications of smaller matrices U A 1:a and U G 1:g .

Notations on indicing rules are also depicted.

mn×mn is a Kronecker product with row elements v i, j (see definition 1 below).

It follows from the properties of the Kronecker product that i = m(α − 1) + γ.

The derivation is shown in section B. Note that in this given form, the Kronecker products are never directly evaluated but the diagonal matrix D can be computed recursively, making it computationally feasible.

Definition 1: For U A ∈ R n×n and U G ∈ R m×m , the Kronecker product of V = U A ⊗ U G ∈ R mn×mn is given by v i, j = U a α,β U b γ,ζ , with the indices i = m(α − 1) + γ and j = m(β − 1) + ζ.

Here, the indices of the matrices U A and U G are α ∈ {1, · · · , n}, β ∈ {1, · · · , n}, γ ∈ {1, · · · , m} and ζ ∈ {1, · · · , m}.

Unfortunately, in the current form, it involves a matrix inversion with size N by N when sampling.

For some layers in modern architectures, this is not be feasible.

This problem is tackled next.

Sampling from the posterior is crucial.

For example, an important use-case of the parameter posterior is estimating the predict uncertainty for test data (x * ,y * ) by a full Bayesian analysis with K mc samples (equation 7).

The herein approximation step is so-called Monte-carlo integration (Gal, 2016) .

However, directly sampling from equation 6 is non-trivial as explained in an example below.

Example 1: Consider the architecture from figure 1 where the covariance matrix Σ 3 ∈ R N 3 ×N 3 for N 3 = 3211264.

With equation 6, the sampling requires O(N 3 3 ) complexity (the cost of inversion and finding a symmetrical factor) and obviously, this operation is computationally infeasible.

Consequently, we next describe a sparse formulation of equation 6 that ensures tractability.

To tackle this challenge, we propose the low rank form in equation 8 2 as a first step.

Here, Λ 1:L ∈ R L×L , U A 1:a ∈ R m×a and U G 1:g ∈ R n×g denote low rank form of corresponding eigenvalues and vectors (depicted in figure 2 ).

Naturally, it follows that L = ag, N = mn and furthermore, the persevered rank L corresponds to preserving top K and additional J eigenvalues (resulting in L ≥ K, L = ag = K + J).

Figure 3: Illustration of algorithm 1.

A low rank approximation on Kronecker factored eigendecomposition that preserves Kronecker structure in eigenvectors constitutes steps 1 to 5.

Note the difference to preserving top L eigenvalues and corresponding eigenvectors (Bishop, 2006) for LRA.

In our case, this results in intractable (U A ⊗ U G ) 1:L which defies the purpose.

Therefore, as seen in equation 8, the Kronecker structure in eigenvectors as (U A 1:a ⊗ U G 1:g ) is preserved.

Consequently, due to the Kronecker product operation, preserving top K eigenvalues results in L = K + J eigenvalues.

Example 2: Let matrix E decomposed as E = U 1:6 Λ 1:6 U T 1:6 ∈ R 6×6 with U 1:

in a descending order.

In this toy example, the LRA with top 3 eigenvalues result in E 1:3 = U 1:3 Λ 1:3 U T 1:3 ∈ R 6×6 (see notation to above).

Instead, consider now the matrix E kron = (U A 1:3 ⊗U G 1:2 )Λ 1:6 (U A 1:3 ⊗U G 1:2 )

T ∈ R 6×6 .

Again, say we want to preserve top 3 the eigenvalues Λ 1:3 and corresponding eigenvectors (U A 1:3 ⊗U G 1:2 ) 1:3 , However, as

, preserving the eigenvectors with the Kronecker structure results in having to store U A 1:2 = u A 1 u A 2 and U G 1:2 = u G 1 u G 2 .

Consequently, additional eigenvalue Λ 4 has to be saved in order to fulfill the definition of a Kronecker product E kron 1:

T ∈ R 6×6 .

In summary, preserving top K eigenvalues results in other J eigenvalues, which ensures the memory-wise tractability when performing LRA on large matrices.

Then, how do we compute a low rank approximation that preserves Kronecker structures in eigenvectors?

For this computation we propose algorithm 1 as an algorithmic contribution (also illustrated in figure 3 ).

Let us start with a definition on indexing rules of Kronecker factored diagonal matrices.

Definition 2: For diagonal matrices S A ∈ R n×n and S G ∈ R m×m , the Kronecker product of Λ = S A ⊗ S G ∈ R mn×mn is given by Λ i = s αβ s γζ , where the indices i = m(β − 1) + ζ with β ∈ {1, · · · , m} and ζ ∈ {1, · · · , n}. Then, given i and m, β = int( i m ) + 1 and given β, m, and i, ζ = i − m(β − 1).

Here, int(·) is an operator that maps its input to lower number integer.

Notations in algorithm 1 are also depicted in figure 2.

Now we explain with a toy example below.

Example 3: For explaining algorithm 1, the toy example can be revisited.

Firstly, as we preserve top 3 eigenvalues, i ∈ {1, 2, 3} which are indices of eigenvalues Λ 1:3 (line 1).

Then, using line 2, β ∈ {1, 2} and ζ ∈ {1, 2} can be computed using definition 2.

This relation holds as Λ is computed from S A ⊗ S G , and thus, U A and U G are their corresponding eigenvectors respectively.

In line 3, we keep U A 1:2 and U G 1:2 using β and ζ.

Again, in order to fulfill the Kronecker product operation, we use line 4 to find the eigenvalues j ∈ {1, 2, 3, 4}, and then preserve Λ 1:4 .

As explained, this has resulted in saving top 3 and additional 1 eigenvalues.

Algorithm 1 provides the generalization of this and even if eigendecomposition does not come with a descending order, the same logic trivially applies.

The incorporation of prior or regularization terms also follows without any additional approximation.

Sampling: A key benefit of the proposed LRA is that now, sampling from the given covariance (equation 6 with the low rank form in equation 8; equation 9 with an incorporation of priors) only involves the inversion of a L × L matrix (in offline settings) and matrix multiplications of smaller Kronecker factored matrices or diagonal matrices during a full Bayesian analysis.

To this end, we derive the analytical form of the sampler in section B.2 which makes the sampling computations feasible.

This enables us to bound the intractable complexity of figure 1 where we first show that IM of DNNs tend to be sparse in its spectrum (similar to the findings of Sagun et al. (2018) ).

With this insight we propose to represent the parameter posterior in a sparse information form which is visualized with its graphical interpretations.

From IM of EFB, we apply our LRA that weakens the strengths of weak nodes (diagonals of IM) and links (off-diagonals) in a preserving fashion.

Then, a diagonal correction can be added to keep the information of each nodes exact.

A key benefit is that the sampling computations can be achieved in a memory-wise feasible way.

Algorithm 2 shows the overall procedures.

Further note that, as IM is estimated after training, our method can be applied to existing architectures.

EFB is also computed in a different way to George et al. (2018) so that our EFB does not require batch assumption for taking expectations, and the scheme is cheaper since eigenvalue decomposition of A i−1 and G i are computed only once.

Computing diagonal correction term also does not involve data.

As a result our approach yields a sparse information form of MND where the IM has a low rank eigendecomposition plus diagonal structure that preserves Kronecker structure in eigenvectors (shown above; prior and scaling terms are omitted to keep the notation uncluttered;Ŵ IV MAP is an information vector associated to the proposed IM).

Since this formulation of model uncertainty has not bee studied before, we provide theoretical results in section C for further insights and justifications.

Sparse Information Filters: A similar idea of sparsifying the information matrix while keeping the diagonals accurate can be found in sparse information filters.

Here, Bayesian tracking is realized in information form of MND instead of canonical counterparts (Kalman Filters).

As this leads to inefficiency in marginalization, sparsity is introduced while keeping the diagonals accurate (Thrun et al., 2004) .

A main difference, however, is that DNNs typically have higher dimensions and a sparse structure in the spectrum (eigenvalues) in contrast to spaces of parameters in SLAM problems.

Thus, we propose to explore Kronecker factorization and induce spectral sparsity or LRA respectively.

Approximation of the Hessian: The Hessian of DNNs is prohibitively too large as its size is quadratic to the parameter space.

For this problem an efficient approximation is a layer-wise Kronecker factorization (Martens & Grosse, 2015; Botev et al., 2017) which have demonstrated a notable scalability (Ba et al., 2017) .

In a recent extension of (George et al., 2018 ) the eigenvalues of the Kronecker factored matrices are re-scaled so that the diagonal variance in its eigenbasis is exact.

The work demonstrates a provable method of achieving higher accuracy.

Yet, as this is harmed by inaccurate estimates of eigenvectors, we further correct the diagonals in the parameter space.

Laplace Approximation: Instead of methods rooted in variational inference (Hinton & van Camp, 1993) and sampling (Neal, 1996) , we build upon LA (MacKay, 1992) as a practical inference framework.

Recently, diagonal (Becker & Lecun, 1989) and Kronecker-factored approximations (Botev et al., 2017) to the Hessian have been applied to LA by Ritter et al. (2018a) .

The authors have further proposed to use LA in continual learning (Ritter et al., 2018b) , and demonstrate a competitive results by significantly outperforming its benchmarks (Kirkpatrick et al., 2017; Zenke et al., 2017) .

Building upon Ritter et al. (2018a) for approximate inference, we propose to use more expressive posterior distribution than matrix normal distribution.

In the context of variational inference, SLANG (Mishkin et al., 2018) share similar spirit to ours in using a low-rank plus diagonal form of covariance where the authors show the benefits of low-rank approximation in detail.

Yet, SLANG is different to ours as they do not explore Kronecker structures and requires changes in the training procedure.

Dimensionality Reduction: A vast literature is available for dimensionality reduction beyond principal component analysis (Wold et al., 1987) and singular value decomposition (Golub & Reinsch, 1971; Van Der Maaten et al., 2009 ).

To our knowledge though, dimensionality reduction in Kronecker factored eigendecomposition that maintains Kronecker structure of eigenvectors has not been studied before.

Thus, we propose algorithm 1 and further provide its theoretical properties in section C.

An empirical study is presented with a toy regression and classification tasks across MNIST (Lecun et al., 1998) , notMNIST (Bulatov, 2011) , CIFAR10 (Krizhevsky, 2009 ) and SHVN (Netzer et al., 2011) data-sets.

The experiments are designed to demonstrate the quality of predictive uncertainty, effects of varying LRA, the quality of approximate Hessian, and gains in reduction of computational complexity due to LRA.

All experiments are implemented using Tensorflow (Abadi et al., 2016) .

Predictive Uncertainty: Firstly, an evaluation on toy regression data-set is presented.

This experiment has an advantage that we can not only evaluate the quality of predictive uncertainty, but also directly compare various approximations to the Hessian.

For this a single-layered fully connected network with seven units in the first layer is considered.

We have used 100 uniformly distributed points x ∼ U(−4, 4) and samples y ∼ N(x 3 , 3 2 ).

Visualization of predictive uncertainty is shown in figure 4 .

HMC (Neal, 1996) , BBB (Blundell et al., 2015) , diagonal and KFAC Laplace Ritter et al. All the methods show higher uncertainty in the regimes far away from training data where BBB showing the most difference to HMC.

Furthermore, Diag, KFAC and EFB Laplace predicts rather high uncertainty even within the regions that are covered by the training data.

DEF variants slightly underestimate the uncertainty but produces the most comparable fit to the FB Laplace and HMC (our ground truths).

We believe this is the direct effect of modelling the Hessian more accurately 3 .

Moreover, since the only difference between EFB and DEF Laplace is a diagonal correction term, this empirical results suggest that keeping diagonals of IM exact results in accurate predictive uncertainty.

Effects of Low Rank Approximation: Next, we quantitatively study the effects of LRA by directly evaluating on the approximations of IM.

This is because uncertainty estimation, despite being a crucial entity, are confounded from the problem itself and may not reveal the algorithmic insights to its full potential.

For this, we revisit the toy regression problem and provide a direct evaluation of IM with measure on normalized Frobenius norm of error err NF in the first layer of the network.

The results are shown in figure 5.

Here, the reduced dimension is not proportional to the ranks (e.g. many zero or close to eigenvalues).

Figure 5 (a) depicts that DEF results in accurate estimates on I ii regardless of the chosen dimensions L while EFB has the more approximation error, which we believe is due to inaccurate estimates of eigenvectors.

KFAC on the other hand, produces the most errors on diagonal elements, which indicate that its assumption of Kronecker factorization induces crude approximation in this experiment.

Regarding the off-diagonal errors EFB also outperforms KFAC and Diag estimates.

Furthermore, error profile of off-diagonal error I i j also explains the principles of the LRA that as we decrease the ranks, the error increases but in a preserving manner.

These results can also be explained by Lemma 1 and 4 of section C which reflects the design principles of the method.

Predictive Uncertainty: Next, we evaluate predictive uncertainty on classification tasks in which the proposed low-rank representation is strictly necessary.

Furthermore, our goal is not to achieve the highest accuracy but evaluate predictive uncertainty.

To this end, we choose classification tasks with known and unknown classes, e.g. a network is not only trained and evaluated on MNIST but also tested using notMNIST.

Note that under such tests, any probabilistic methods should report their evaluations on both known and unknown classes with the same hyperparameter settings.

This is because a Bayesian Neural Network to be always highly uncertain, which may seem to work well on out-of-distribution samples but are always overestimating uncertainty, even for the correctly classified samples within the distribution similar to the train data.

For evaluating predictive uncertainty on known classes, Expectation Calibration Error (ECE) has been used.

As we found it more intuitive, normalized entropy is reported for evaluating predictive uncertainty on unknown classes.

On MNIST-notMNIST experiments, we compare to MC-dropout (Gal, 2016) , ensemble (Lakshminarayanan et al., 2017) of size 15, Diag and KFAC Laplace (Ritter et al., 2018a) .

These methods are state-of-the-art baselines that have a merit of requiring no changes in the training procedure.

The later is crucial for a fair comparison as we can use the same experiment settings (Mukhoti et al., 2018) .

Regarding the architectures, LeNet with RELU and a L2 coefficient of 1e-8 has been the choice.

In particular, this typically makes a neural network overconfident, and we can see the effects of model uncertainty.

This architecture validates our claim as it has the parameters of size θ 3 ∈ R 3137×1024 in the 3 rd layer.

Obviously, its covariance is intractable as it is quadratic in size (see figure 1) .

The results can be found in table 1.

Firstly all the methods improved significantly over the deterministic one (denoted NN).

Furthermore, DEF Laplace achieved here the lowest ECE, at the same time, predicted with the highest mean entropy on out-of-distribution samples.

Figure 6 shows this result where our method separates between wrong and correct predictions which stems from the domain change.

Further tests were performed on CIFAR10 (known) and SVHN (unknown) to see the generalization under batch normalization and data augmentation.

For this, we trained a 5 layer architecture with 2 CNN and 3 FC layers.

The results are also reported in table 1.

Similar to MNIST experiments, our method resulted in a better calibration performance and out-of-distribution detection overall.

Note that for Diag, KFAC and EFB Laplace, grid searches on hyperparameters were rather non-trivial here.

Increasing τI had the tendency to reduce ECE on CIFAR10, but in return resulted in underestimating the uncertainty on SVHN and vice versa.

DEF Laplace instead, required smallest regularization hyperparameters to strike a good balance between these two objectives.

We omitted dropout as using it as a stochastic regularization instead of batch normalization would result in a different network and thus, comparison would be not meaningful.

More implementation details are provided in section E.

The proposed LRA has been imposed as a means to tackle the challenges of computational intractability of MND.

To empirically access the reduction in complexity, we depict the parameter and low rank dimensions N and L respectively in table 2.

As demonstrated, our LRA based sampling computations reduce the computational complexity significantly.

Furthermore, this explains the necessity of LRA -certain layers (e.g. FC-1 of both MNIST and CIFAR experiments) are computationally intractable to store, infer and sample.

As a result, we demonstrate an alternative representation for DNNS without resorting to fully factorized and matrix normal distribution.

Discussion and Limitations: Importantly, we demonstrate that when projected to different success criteria, no inference methods largely win uniformly.

Yet these experiments also show empirical evidence that our method works in principle and compares well to the state-of-the-art.

Representing layer-wise MND in a sparse information form, and demonstrating a low rank inverse/sampling computations, we show an alternative approach of designing scalable and practical inference framework.

Finally, these results also indicate that keeping the diagonals of IM accurate while sparsifying the off-diagonals can lead to outstanding performance in terms of predictive uncertainty and generalizes well to various data, models and even measures.

For future works, we share the view that comparing different approximations to the true posterior is quite challenging for DNNs.

Consequently, better metrics and benchmarks that show the benefits of model uncertainty can be an important direction.

On the other hand, we also address a key limitation of our work which stems from two hypothesis: (a) when represented in information form, the spectrum of IM should be sparse, and (b) keeping the diagonals exact while sparsifying the off-diagonals should result in a better estimates of model uncertainty (equivalently keeping the information content of a node exact while sparsifying the weak links between the nodes from a graphical interpretation of information matrix).

While empirical evidence from prior works (Sagun et al., 2018; Thrun et al., 2004; Bailey & Durrant-Whyte, 2006) along with our experiments validate these to some extent, there exists no theoretic guarantees to our knowledge.

Consequently, theoretical studies that connect information geometry (Amari, 2016) of DNNs and Bayesian Neural Networks can be an exciting venue of future research.

Nevertheless, similar to sparse Gaussian Processes (Snelson & Ghahramani, 2006) , we believe our work can be a stepping stone for sparse Bayesian Neural Networks that goes beyond approximate inference alone.

We address an effective approach of representing model uncertainty in deep neural networks using Multivariate Normal Distribution, which has been thought computationally intractable so far.

This is achieved by designing its novel sparse information form.

With one of the most expressive representation of model uncertainty in current Bayesian deep learning literature, we show that uncertainty can be estimated more accurately than existing methods.

For future works, we plan to demonstrate a real world application of this approach, pushing beyond the validity of concepts.

The matrix normal distribution is a probability density function for the random variable X ∈ R n×m in matrix form.

It can be parameterized with mean W MAP ∈ R n×m , scales U A ∈ R n×n and U G ∈ R m×m .

It is essentially a multivariate Gaussian distribution with mean vec(W MAP ) and covariance U ⊗ V.

In section B, we denote this distribution with MN parameterized by W MAP , U A and U G so that p(X|W MAP , U A , U G ) = MN(W MAP , U A , U G ).

Here, tr stands for trace operation and we omitted layer indicing i for better clarity.

Refer to Gupta & Nagar (1999) for more details.

Information form of Multivariate Normal Distribution (MND) is a dual representation for the well known canonical form.

Lets denotex = vec(X) ∈ R nm , µ = vec(W MAP )∈ R nm and Σ = I −1 ∈ R mn×nm as a random variable, mean and covariance respectively for N = mn.

Then, equation 12 defines the canonical form.

Now we denote its Information form in equation 13.

Here,x ∈ R mn represent the random variable as well.

W IV MAP = Σ −1 µ ∈ R mn and I = Σ −1 ∈ R mn×mn are information vector (denoted IV in the main text with superscript) and matrix respectively.

We denote the information form as N −1 which is completely described by an information vector and matrix.

Information matrix is also widely known as precision matrix.

Thrun et al. (2004) in Simultaneous Localization and Mapping (SLAM) literature provides a good overview and explanations.

Directly evaluating U A ⊗ U G may not be computationally feasible for modern DNNs.

Therefore, we derive the analytical form of the diagonal elements for (

T without having to fully evaluate it.

Let U A ∈ R n×n and U G ∈ R m×m be the square matrices.

Λ ∈ R mn×mn is a diagonal matrix by construction.

V = U A ⊗ U G ∈ R mn×mn is a Kronecker product with elements v i, j with i = m(α − 1) + γ and j = m(β − 1) + ζ (from definition of Kronecker product).

Then, the diagonal entries of (U A ⊗ U G )Λ(U A ⊗ U G )

T can be computed as follows:

Derivation: As a first step of the derivation, we express (A ⊗ B)Λ(A ⊗ B) T in the following form:

Then, diag(UU

being again a diagonal matrix.

Therefore, u i j = v i, j Λ j due to the multiplication with a diagonal matrix from a right hand side.

Substituting back these results in (

2 which completes the derivation.

Formulating equation 14 for the non-square matrices (which results after a low rank approximation) such as U A 1:a ∈ R n×a and U G 1:g ∈ R m×g and paralleling this operation are rather trivial and hence, they are omitted.

For a full Bayesian analysis which is approximated by a Monte Carlo integration, sampling is a crucial operation (see equation 7) for computing predictive uncertainty.

We start by stating the problem.

Problem statement: Consider drawing samples vec(W s ) ∈ R nm from our sparse information form:

Typically, drawing such samples vec(W s ) from a canonical form of MND requires finding a symmetrical factor of the covariance matrix (e.g. Chloesky decomposition) which is cubic in cost O(N 3 ).

Even worse, when represented in an information form as in equation 16, it requires first an inversion of information matrix and then the computation of a symmetrical factor which overall constitutes two operations of cost O(N 3 ).

Clearly, if N lies in a high dimension such as 1 million, even storing is obvious not feasible, let alone the sampling computations.

Therefore, we need a sampling computation that (a) keeps the Kronecker structure while sampling so that first, the storage is memory-wise feasible, and then (b) the operations that require cubic cost such as inversion, must be performed in the dimensions of low rank L instead of full parameter dimensions N. We provide the solution below.

Analytical solution: Let us define X l ∈ R mn and X s ∈ R m×n as the samples from a standard Multivariate Normal Distribution in equation 17 where we denote the followings: 0 nm ∈ R nm , I mn ∈ R mn×mn , 0 n×m ∈ R n×m , I n ∈ R n×n and I m ∈ R m×m .

Note that these sampling operations are cheap.

Furthermore, we denote W l = vec(W s ) ∈ R mn , θ MAP = vec(W MAP ) ∈ R mn as a sample from equation 16 and its mean as a vector respectively.

We also note that Λ 1:L ∈ R L×L and D ∈ R mn×mn are the low ranked form of the re-scaled eigen-values and the diagonal correction term as previously defined.

U A 1:a ∈ R m×a and U G 1:g ∈ R n×g are the eigenvectors of low ranked eigen-basis so that m ≥ a, n ≥ g and L = ag.

Then, the samples of 16 can be computed analytically as 4 :

Firstly, the symmetrical factor F c ∈ R mn×mn in equation 18 is a function of matrices that are feasible to store as they involve diagonal matrices or small matrices in a Kronecker structure.

Furthermore, (20) 4 We show how the Kronecker structure of F c can be exploited to compute F c X l in the derivation only.

Consequently, the matrices in equation 18 are defined as C ∈ R L×L , (C −1 + V T V) ∈ R L×L and I L ∈ R L×L .

In this way, the two operations namely Cholesky decomposition and inversion that are cubic in cost O(N 3 ) are reduced to the low rank dimension L with complexity O(L 3 ).

Derivation: Firstly, note that sampling from a standard multivariate Gaussian for X l or X s is computationally cheap (see equation 17).

Given a symmetrical factor for the covariance Σ = F c F c T (e.g. by Cholesky decomposition), samples can be drawn via θ MAP + F c X l as depicted in equation 18.

Our derivation involves finding such symmetrical factor for the given form of covariance matrix while exploring the Kronecker structure for the sampling computations to bound the complexity as O(L 3 ).

Let us first reformulate the covariance (inverse of information matrix) as follows.

Here, we define:

1:L .

Now, a symmetrical factor for Σ = F c F c T can be found by exploiting the above structure.

We let W c be a symmetrical factor for VV T + I nm so that

is the symmetrical factor of Σ. Following the work of Ambikasaran & O'Neil (2014) the symmetrical factor W c can be found using equations below.

Note that A and B are Cholesky decomposed matrices of V T V ∈ R L×L and V T V + I L ∈ R L×L respectively.

As a first result, this operation is bounded by complexity O(L 3 ) instead of the full parameter dimension N. Now the symmetrical factor for Σ can be expressed as follows.

Woodbury's Identity is used here.

Now, it follows simply by substitution:

This completes the derivation of equation 18.

As a result, the inversion operation is bounded by complexity O(L 3 ).

Furthermore, the derivation constitutes smaller matrices U A 1:a and U G 1:g or diagonal matrices D and I mn which can be stored as vectors.

In short the complexity has significantly reduced.

Now we further derive computations that exploits rules of Kronecker products.

Consider:

Then, it follows by defining inverted matrix L c = (

We further reduce this by evaluating D

L×L .

We note that this multiplication operation is memory-wise feasible.

Now, we map X l D to matrix normal distribution by an unvec(·) operation so that

.

Using a widely known relation for Kronecker product that is -

Note that matrix multiplication is performed with small matrices.

Repeating a similar procedure as above we obtain the equation below for

This completes the derivation.

Lastly, we provide a remark below to summarize the main points.

Remark: We herein presented derivation is to sample from equation 16, a low-rank and information formulation of MND.

This analytical solution ensures (a)

for a matrix inversion, (c) storage of small matrices U G 1:g , U A 1:a , a diagonal matrix D and identity matrices and finally (d) matrix multiplications that only involve these matrices.

This is a direct benefit of our proposed LRA that preserves Kronecker structure in eigenvectors.

Some of the interesting theoretical properties are as follows with proofs provided in section D.

A theoretical result of adding a diagonal correction term is captured below.

This relates to the work presented in section 2.2 where a diagonal correction term is added to EFB estimates of IM.

Lemma 1:

Let I ∈ R N×N be the real Fisher information matrix, and let I def ∈ R N×N and I efb ∈ R N×N be the DEF and EFB estimates of it respectively.

It is guaranteed to have I −

I efb F ≥ I −

I def F .

Corollary 1:

Let I kfac ∈ R N×N and I def ∈ R N×N be KFAC and our estimates of real Fisher information matrix I ∈ R N×N respectively.

Then, it is guaranteed to have I − I kfac F ≥

I −

I def F .

Remark:

For interested readers, find the proof I − I k f ac F ≥ I −

I efb F in George et al. (2018) .

or vice versa.

Yet, our proposed approximation yield better estimates than KFAC in the information form of MND.

To our knowledge, the proposed sparse IM have not been studied before.

Therefore, we theoretically motivate its design and validity for better insights.

The analysis can be found below.

Firstly, we study the effects of preserving Kronecker structure in eigenvectors.

We define:

as a low rank EFB estimates of true Fisher that preserves top K eigenvalues.

Similarly,Î top 1:L can be defined which preserves top L eigenvalues.

In contract, our proposal to preserve Kronecker structure in eigenvectorsÎ 1:L is denoted as shown below.

Now, we provide our analysis with Lemma 2.

Lemma 2:

Let I ∈ R N×N be the real Fisher information matrix, and letÎ

andÎ 1:L ∈ R N×N be the low rank estimates of I of EFB obtained by preserving top K, L and top K plus additional J resulting in L eigenvalues.

Here, we define K < L.

Then, the approximation error of I 1:L is bounded as follows:

Remark: This bound provides an insight that if preserving top L eigenvalues result in prohibitively too large covariance matrix, our LRA provides an alternative to preserving top K eigenvalues given that K < L. In practise, note thatÎ 1:L is a memory-wise feasible option as we formulatê

T which preserves the Kronecker structure in eigenvectors.

This can be a case where evaluating (U A 1:a ⊗ U G 1:g ) or (U A 1:a ⊗ U G 1:g ) 1:K is not feasible to store.

∈ R N×N is a nondegenerate covariance matrix if the diagonal correction matrix D and

T are both symmetric and positive definite.

This condition is satisfied if

i for all i ∈ {1, 2, · · · , d} and with Λ 1:L 0.

Remark: This Lemma comments on validity of resulting parameter posterior and proves that sparsifying the matrix can lead to a valid non-degenerate covariance if two conditions are met.

As non-degenerate covariance can have a uniquely defined inverse, it is important to check these two conditions.

We note that searching the rank can be automated with off-line computations that does not involve any data.

Thus, it does not introduce significant overhead.

In case D does not turn out to be, there are still several techniques that can deal with it.

We recommend eigen-value clipping (Chen et al., 2018) or finding nearest positive semi-definite matrices (Higham, 1988) .

Lastly, D −1 does not get numerically unstable when we add a prior precision term and a scaling factor (ND + τI) −1 .

Let I ∈ R N×N be the real Fisher information matrix, and letÎ def ∈ R N×N , I efb ∈ R N×N and I kfac ∈ R N×N be the low rank DEF, EFB and KFAC estimates of it respectively.

Then, it is guaranteed to have diag

Furthermore, if the eigenvalues of I def contains all non-zero eigenvalues of I def , it follows:

Remark: Lemma 4 shows the optimally in capturing the diagonal variance while indicating that our approach also becomes effective in estimating off-diagonal entries if IM contains many close to zero eigenvalues.

Validity of this assumption has been studied by Sagun et al. (2018) where it is shown that the Hessian of overparameterized DNNs tend to have many close-to-zero eigenvalues.

Intuitively, from a graphical interpretation of IM, diagonal entries indicate information present in each nodes and off-diagonal entries are links of these nodes (depicted in figure 1 ).

Our sparsification scheme reduces the strength of the weak links (their numerical values) while keeping the diagonal variance exact.

This is a result of the diagonal correction after LRA which exploits spectrum sparsity of IM.

D.1 Diagonal correction leads to more accurate estimation of Information matrix Proposition 1: Let I ∈ R N×N be the real Fisher information matrix, and let I def ∈ R N×N and I def ∈ R N×N be our estimates of it with rank d and k such that k < d. Their diagonal entries are equal that is I ii = I def ii =Î def ii for all i = 1, 2, . . .

, N.

proof: The proof trivially follows from the definitions of I ∈ R N×N , I def ∈ R N×N andÎ def ∈ R N×N .

As the exact Fisher is an expectation on outer products of back-propagated gradients, its diagonal entries equal I ii = E δθ 2 i for all i = 1, 2, . . .

, N.

In the case of full ranked I de f , substituting

T ii results in equation 32 for all i = 1, 2, . . .

, N.

T ii which results in equation 33 for all i = 1, 2, . . .

, N.

Therefore, we have I ii = I def ii =Î def ii for all i = 1, 2, . . .

, N.

Lemma 1:

Let I ∈ R N×N be the real Fisher information matrix, and let I def ∈ R N×N and I efb ∈ R N×N be the DEF and EFB estimates of it respectively.

It is guaranteed to have I −

I efb F ≥ I −

I def F .

proof: Let e 2 = A − B 2 F define a squared Frobenius norm of error between the two matrices A ∈ R N×N and B ∈ R N×N .

Now, e 2 can be formulated as,

The first term of equation 34 belongs to errors of diagonal entries in B wrt A whilst the second term is due to the off-diagonal entries.

Now, it follows that,

since by definition, I efb and I def have the same off-diagonal terms.

Corollary 1:

Let I k f ac ∈ R N×N and I def ∈ R N×N be KFAC and our estimates of real Fisher Information matrix I ∈ R N×N respectively.

Then, it is guaranteed to have I − I k f ac F ≥ I −

I def F .

Find the proof I − I k f ac F ≥ I −

I efb F in George et al. (2018) .

Lemma 2:

Let I ∈ R N×N be the real Fisher information matrix, and letÎ

andÎ 1:L ∈ R N×N be the low rank estimates of I of EFB obtained by preserving top K, L and top K plus additional J resulting in L eigenvalues.

Here, we define K < L.

Then, the approximation error of I 1:L is bounded as follows:

For the second part of the proof, lets recap that Lemma 2 (Wely's idea on eigenvalue perturbation) that removing zero eigenvalues does not affect the approximation error in terms of Frobenius norm.

This then implies that off-diagonal elements ofÎ def and I efb are equivalent.

Then,:

2 ii = 0 according to proposition 1 for all the elements i.

KFAC library from Tensorflow 5 was used to implement the Fisher estimator (Martens & Grosse, 2015) for our methods and the works of Ritter et al. (2018a) .

Note that empirical Fisher usually is not a good estimates as it is typically biased (Martens & Grosse, 2015) and therefore, we did not use it.

KFAC library offers several estimation modes for both fully connected and convolutional layers.

We have used the gradients mode for KFAC Fisher estimation (which is also crucial for our pipelines) whereas the exact mode was used for diagonal approximations.

We did not use the exponential averaging for all our experiments as well as the inversion scheme in the library.

However, when using it in practice, it might be useful especially if there are too many layers that one cannot access convergence of the Fisher estimation.

We have used NVIDIA Tesla for grid searching the parameters of Diag and KFAC Laplace, and 1080Ti for all other experiments.

Apart from the architecture choices discussed in section 4, the training details are as follows.

A gradient descent optimizer from tensorflow has been used with a learning rate of 0.001 with zero prior precision or L2 regularization coefficient (τ = 0.2 for KFAC, τ = 0.45 for Diag, N = 1 and τ = 0 for both FB and DEF have been used).

Mean squared error (MSE) has been used as its loss function.

Interestingly, the exact block-wise Hessian and their approximations for the given experimental setup contained zero values on its diagonals.

This can be interpreted as zero variance in information matrix, meaning no information, resulting in information matrix being degenerate for the likelihood term.

In such cases, the covariance may not be uniquely defined (Thrun et al., 2004) .

Therefore, we treated these variances deterministic, making the information matrix non-degenerate (motivated from Lemma 3).

Similar findings are interestingly reported by MacKay (1992) .

More importantly, we present a detailed analysis to avoid misunderstanding about our toy dataset experiments.

As a starting remark, a main advantage of this toy regression problem is that it simplifies the understandings of on-going process, in lieu of sophisticated networks with a large number of parameters.

Typically, as of Herandez-Lobato & Adams (2015) , Ritter et al. (2018a) , Gal (2016) , or even originating back to Gaussian processes literature, this example has been used to check the predictive uncertainty by qualitatively evaluating on whether the method predicts high uncertainty in the regimes of no training data.

However, a drawback exists: no quantitative analysis has been reported to our knowledge other than qualitatively comparing it to community wide accepted ground truth such as Hamiltonian Monte Carlo Sampling (Neal, 1996) , and LA using KFAC and Diag seem to be sensitive to hyperparameters in this dataset which makes the comparison difficult.

This is illustrated in figure 7 where we additional introduce Random which is just a user-set τI for covariance estimation in order to demonstrate this.

Qualitatively analyzing from the first look, all the methods look very similar in delivering high uncertainty estimates in the regimes of no training data.

Here, we note that the same hyperparameter settings have been used for Diag, KFAC and FB Laplace whereas the user-set τ = 7 has been found for Random.

This agrees to the discussions of Ritter et al. (2018a) as KFAC resulted in less τ when compared to Diag Laplace.

However, we also observed that without the approximation step of equation 3 (denoted OKF), using the same hyper parameter as above resulted in visible over-prediction of uncertainty and inaccurate estimates on the prediction.

This is shown in figure 8 .

Again, tuning the parameter to a higher precision τ, similar behavior to figure 7 can be reproduced.

This can be analyzed by visualizing the covariance of KFAC and OKF.

As it can be seen, in this experiment settings, figure 8 shows that equation 3 damps the magnitude of estimated covariance matrix.

A possible explanation is that if the approximate Hessian is degenerate, then small τI places a big mass on areas of low posterior probabilities for some network parameters with no information (zero variance and correlations in the approximate Hessian).

This can be seen in figure 8 part (a) where the approximate Hessian contains 3 parameters with exactly zero diagonal elements and zeros in its off-diagonal elements.

If one tries to add a small τ = 0.001 here, then the covariance of these parameters get close to its inverse τ −1 = 1000 as shown in figure 8 part (c).

This would in return result in over prediction of uncertainty and inaccurate predictions which explains figure 8 part (a).

Another interesting experiments are studying the effects of dataset size to number of parameters.

For this, we have increased the dataset size to 100 in oppose to 20.

Again, we now compare the approximate Hessian by visualizing them.

Notably, at using 100 data points resulted in more number of zero diagonal entries and corresponding rows and columns.

This is due to over parameterization of the model which results in under determined Hessian.

These insights hint for the followings.

Accurately estimating the Hessian while forcing its estimates non-degeneracy via not considering zero eigenvalues for this data and model can lead to less sensitivity to its hyperparameters or τ in particular.

Secondly, further increasing or decreasing the ratio of data points to number of parameters change the approximate Hessian (similarly found for estimates of Fisher) changes its structure, and can lead to under-determined approximation (therefore, changing its loss landscape).

Finally, if the Hessian is under-determined, hyperparameters τ affects the resulting predictive uncertainty (or covariance) if its magnitude significantly differs (and in case of KFAC).

However, as more detailed experimental analysis is outside the scope of the paper, can be an interesting future work to further analyze the relation between the hyperparameters, their probabilistic interpretation and resulting loss landscape of neural network.

We have used Numpyro (Phan et al., 2019) for the implementations of HMC.

We have used 50000 MC samples to generate the results in order to ensure the convergence.

For the implementation of Bayes By Backprop we have used an open-source implementation https://github.com/ThirstyScholar/ bayes-by-backprop for which a similar experiment settings are implemented where the Gaussian noise is sampled in a batch initially, and a symmetric sampling technique is deployed.

We note that the number of data samples and network architectures are different.

Furthermore, we have used 10000 iterations to ensure convergence of the network.

Most of the implementations for MNIST and CIFAR10 experiments were taken from Tensorflow tutorials 6 including the network architectures and training pipelines if otherwise stated in the main text.

This is in line of argument that our method can be directly applied to existing, and well trained neural networks.

For MNIST experiments, the architecture choices are the followings.

Firstly, no down-scaling has been performed to its inputs.

The architecture constitutes 2 convolutional layers followed by 2 fully connected layer (each convolutional layer is followed by a pooling layer of size 2 by 2, and a stride 2).

For flattening from the second convolutional layer to the first fully connected layer, a pooling operation of 49 by 64 has been naturally used.

RELU activation have been used for all the layers except the last layer which computes the softmax output.

Dropout has been applied to the fully connected layer with dropout rate of 0.6 after a grid search (explained in section E.3.1).

Regarding the loss functions, cross entropy loss has been used with ADAM as its optimizer and learning rate of 0.001.

An important information is the size of each layers.

The first layer constitutes 32 filters with 5 by 5 kernel, followed by the second layer with 64 filters and 5 by 5 kernel.

The first fully connected layer then constitutes 1024 units and the last one ends with 10 units.

We note that, this validates our method on memory efficiency as the third layer has a large number of parameters, and its covariance, being quadratic in its size, cannot be stored in our utilized GPUs.

Regarding the architecture selection of CI-FAR10 experiments, no down-scaling of the inputs has be done.

The chosen architecture is composed of 2 convolutional layers followed by 3 fully connected layers.

Pooling layers of size 3 by 3 with strides 2 have been applied to outputs of the convolutional layers.

Obviously, the third convolutional layer is pooled to match the input size of the following fully connected layers.

Batch normalization has been applied to each outputs of convolutional layer before pooling, with bias 1, α of 0.001/9.0 and β of 0.75 (notations are different to the main text, and this follows that of tensorflow library).

A weight decay factor of 0.004 has been used, and trained again with cross entropy loss, now with a stochastic gradient descent.

Learning rate of 0.001 has been used.

Again, the most relevant settings are: the first layer constitutes 5 by 5 kernel with 64 filters.

This is then again followed by the same (but as input to CIFAR10 is RGB, the second layer naturally has more number of parameters).

Units of 384, 192, and 10 have been used for the fully connected layers in an ascending order.

Lastly, random cropping, flipping, brightness changes and contract have been applied as the data augmentation scheme.

Similar to MNIST experiments, the necessity of LRA is capture in CIFAR10 as well.

Unlike Ritter et al. (2018a) we did not artificially augment the data for MNIST experiments because the usual training pipeline did not require it.

For our low rank approximation, we always have used the maximum rank we could fit, after removing all the zero eigenvalues and checking the conditions from Lemma 3.

Lastly we have used 1000 Monte-Carlo samples for MNIST, and 100 samples for CIFAR10 and toy regression experiments.

Implementation of deep ensemble (Lakshminarayanan et al., 2017) was kept rather simple by not using the adversarial training, but we combined 15 networks that were trained with different initialization.

The same architecture and training procedure were used for all.

Note that CIFAR10 experiments with similar convolutional architectures were not present in the works of (Lakshminarayanan et al., 2017) to the best of our knowledge.

On MNIST, Louizos & Welling (2017) found similar results to ours that deep ensemble performed similar to the MC-dropout (Gal, 2016) .

For dropout, we have tried a grid search of dropout probabilities of 0.5 and 0.8, and have reported the best results.

For the methods based on Laplace approximation, we have performed grid search on hyperparameters N of (1, 50000, 100000) and 100 values of τ were tried using known class validation set.

Note that for every method, and different data-sets, each method required different values of τI to give a reasonable accuracy.

The starting point of the grid-search were determined based on if the mean values of their predictions were obtained similar accuracy to the deterministic counter parts.

The figure below are the examples on MNIST where minimum ece points were selected and reported.

@highlight

An approximate inference algorithm for deep learning