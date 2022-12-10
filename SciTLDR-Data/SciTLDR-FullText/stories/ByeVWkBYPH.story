In this paper, we propose a new loss function for performing principal component analysis (PCA) using linear autoencoders (LAEs).

Optimizing the standard L2 loss results in a decoder matrix that spans the principal subspace of the sample covariance of the data, but fails to identify the exact eigenvectors.

This downside originates from an invariance that cancels out in the global map.

Here, we prove that our loss function eliminates this issue, i.e. the decoder converges to the exact ordered unnormalized eigenvectors of the sample covariance matrix.

For this new loss, we establish that all local minima are global optima and also show that computing the new loss (and also its gradients) has the same order of complexity as the classical loss.

We report numerical results on both synthetic simulations, and a real-data PCA experiment on MNIST (i.e., a 60,000 x784 matrix), demonstrating our approach to be practically applicable and rectify previous LAEs' downsides.

Ranking among the most widely-used and valuable statistical tools, Principal Component Analysis (PCA) represents a given set of data within a new orthogonal coordinate system in which the data are uncorrelated and the variance of the data along each orthogonal axis is successively ordered from the highest to lowest.

The projection of data along each axis gives what are called principal components.

Theoretically, eigendecomposition of the covariance matrix provides exactly such a transformation.

For large data sets, however, classical decomposition techniques are infeasible and other numerical methods, such as least squares approximation schemes, are practically employed.

An especially notable instance is the problem of dimensionality reduction, where only the largest principal components-as the best representative of the data-are desired.

Linear autoencoders (LAEs) are one such scheme for dimensionality reduction that is applicable to large data sets.

An LAE with a single fully-connected and linear hidden layer, and Mean Squared Error (MSE) loss function can discover the linear subspace spanned by the principal components.

This subspace is the same as the one spanned by the weights of the decoder.

However, it failure to identify the exact principal directions.

This is due to the fact that, when the encoder is transformed by some matrix, transforming the decoder by the inverse of that matrix will yield no change in the loss.

In other words, the loss possesses a symmetry under the action of a group of invertible matrices, so that directions (and orderings/permutations thereto) will not be discriminated.

The early work of Bourlard & Kamp (1988) and Baldi & Hornik (1989) connected LAEs and PCA and demonstrated the lack of identifiability of principal components.

Several methods for neural networks compute the exact eigenvectors (Rubner & Tavan, 1989; Xu, 1993; Kung & Diamantaras, 1990; Oja et al., 1992) , but they depend on either particular network structures or special optimization methods.

It was recently observed (Plaut, 2018; Kunin et al., 2019 ) that regularization causes the left singular vectors of the decoder to become the exact eigenvectors, but recovering them still requires an extra decomposition step.

As Plaut (2018) point out, no existent method recovers the eigenvectors from an LAE in an optimization-independent way on a standard network -this work fills that void.

Moreover, analyzing the loss surface for various architectures of linear/non-linear neural networks is a highly active and prominent area of research (e.g. Baldi & Hornik (1989) ; Kunin et al. (2019) ; Pretorius et al. (2018) ; Frye et al. (2019) ).

Most of these works extend the results of Baldi & Hornik (1989) for shallow LAEs to more complex networks.

However, most retain the original MSE loss, and they prove the same critical point characterization for their specific architecture of interest.

Most notably Zhou & Liang (2018) extends the results of Baldi & Hornik (1989) to deep linear networks and shallow RELU networks.

In contrast in this work we are going after a loss with better loss surface properties.

We propose a new loss function for performing PCA using LAEs.

We show that with the proposed loss function, the decoder converges to the exact ordered unnormalized eigenvectors of the sample covariance matrix.

The idea is simple: for identifying p principal directions we build up a total loss function as a sum of p squared error losses, where the i th loss function identifies only the first i principal directions.

This approach breaks the symmetry since minimizing the first loss results in the first principal direction, which forces the second loss to find the first and the second.

This constraint is propagated through the rest of the losses, resulting in all p principal components being identified.

For the new loss we prove that all local minima are global minima.

Consequently, the proposed loss function has both theoretical and practical implications.

Theoretically, it provides better understanding of the loss surface.

Specifically, we show that any critical point of our loss L is a critical point of the original MSE loss but not vice versa, and conclude that L eliminates those undesirable global minima of the original loss (i.e., exactly those which suffer from the invariance).

Given that the set of critical points of L is a subset of critical points of MSE loss, many of the previous work on loss surfaces of more complex networks likely extend.

In light of the removal of undesirable global minima through L, examining more complex networks is certainly a very promising direction.

As for practical consequences, we show that the loss and its gradients can be compactly vectorized so that their computational complexity is no different from the MSE loss.

Therefore, the loss L can be used to perform PCA/SVD on large datasets using any method of optimization such as Stochastic Gradient Descent (SGD).

Chief among the compellingly reasons to perform PCA/SVD using this method is that, in recent years, there has been unprecedented gains in the performance of very large SGD optimizations, with autoencoders in particular successfully handling larger numbers of high-dimensional training data (e.g., images).

The loss function we offer is attractive in terms of parallelizability and distributability, and does not prescribe any single specific algorithm or implementation, so stands to continue to benefit from the arms race between SGD and its competitors.

More importantly, this single loss function (without an additional post hoc processing step) fits seamlessly into optimization pipelines (where SGD is but one instance).

The result is that the loss allows for PCA/SVD computation as single optimization layer, akin to an instance of a fully differentiable building block in a NN pipeline Amos & Kolter (2017) , potentially as part of a much larger network.

Let X ∈ R n×m and Y ∈ R n×m be the input and output matrices, where m centered sample points, each n-dimensional, are stacked column-wise.

Let x j ∈ R n and y j ∈ R n be the j th sample input and output (i.e. the

where ·, · F and · F are the Frobenius inner product and norm, I i;p is a p × p matrix with all elements zero except the first i diagonal elements being one. (Or, equivalently, the matrix obtained by setting the last p − i diagonal elements of a p × p identity matrix to zero, e.g. I 2;3 = 1 0 0 0 1 0 0 0 0

.)

In what follows, we shall denote the transpose of matrix M by M .

Moreover, the matrices A ∈ R n×p , and B ∈ R p×n can be viewed as the weights of the decoder and encoder parts of an LAE.

The results are based on the following standard assumptions that hold generically: Assumption 1.

For an input X and an output Y , let Σ xx := XX , Σ xy := XY , Σ yx := Σ xy and Σ yy = Y Y be their sample covariance matrices.

We assume

• The input and output data are centered (zero mean).

• Σ xx , Σ xy , Σ yx and Σ yy are positive definite (of full rank and invertible).

• The covariance matrix Σ := Σ yx Σ −1 xx Σ xy is of full rank with n distinct eigenvalues λ 1 > λ 2 > · · · > λ n .

•

The decoder matrix A has no zero columns.

Claim.

The main result of this work proved in Theorem 2 is as follows: If the above assumptions hold then all the local minima of L(A, B) are achieved iff A and B are of the form

where the i th column of U 1:p is the unit eigenvector of Σ := Σ yx Σ −1 xx Σ xy corresponding to the i th largest eigenvalue and D p is a diagonal matrix with nonzero diagonal elements.

In other words, A contains ordered unnormalized eigenvectors of Σ corresponding to the p largest eigenvalues.

Moreover, all the local minima are global minima with the value of the loss function at those global minima being

where λ i is the i th largest eigenvalue of Σ := Σ yx Σ −1 xx Σ xy .

In the case of autoencoder (Y = X): Σ = Σ xx .

Finally, while L(A, B) in the given form contains O(p) matrix products, we will show that it can be evaluated with constant (less than 7) matrix products independent of the value p.

In this paper, the underlying field is always R, and positive semidefinite matrices are symmetric by definition.

The following constant matrices are used extensively throughout.

The matrices T p ∈ R p×p and S p ∈ R p×p are defined as

Another matrix that will appear in the formulation isŜ p := T −1

p .

Clearly, the diagonal matrix T p is positive definite.

As shown in Lemma 2, S p andŜ p are positive definite as well.

The general strategy to prove the above claim is as follows.

First the analytical gradients of the loss is derived in a matrix form in Propositions 1 and 2.

We compare the gradients with that of the original Minimum Square Error (MSE) loss.

Next, we analyze the loss surface by solving the gradient equations which yields the general structure of critical points based on the rank of the decoder matrix A. Next, we delineate several interesting properties of the critical points, notably, any critical point of the loss is also a critical point for the MSE loss but not the other way around.

Finally, by performing second order analysis on the loss in Theorem 2 the exact equations for local minima are derived which is shown to be as claimed.

LetL(A, B) and L(A, B) be the original loss, and the proposed loss function, respectively, i.e.,

The first step is to calculate the gradients with respect to A and B and set them to zero to derive the implicit expressions for the critical points.

In order to do so, first, in Lemma 5, for a fixed A, we derive the directional (Gateaux) derivative of the loss with respect to B along an arbitrary direction

As shown in the proof of the lemma, d B L(A, B)W is derived by writing the norm in the loss as an inner product, opening it up using linearity of inner product, dismiss second order terms in W (i.e. O( W 2 )) and rearrange the result as the inner product between the gradient with respect to B, and the direction W , which yields

where, • is the Hadamard product and the constant matrices T p and S p , were defined in the beginning.

Second, the same process is done in Lemma 6, to derive d A L(A, B)V ; the derivative of L with respect to A in an arbitrary direction V ∈ R n×p , for a fixed B, which is then set to zero to derive the implicit expressions for the critical points.

The results are formally stated in the two following propositions.

Proposition 1.

For any fixed matrix A ∈ R n×p the function L(A, B) is convex in the coefficients of B and attains its minimum for any B satisfying the equation

where • is the Hadamard (element-wise) product operator, and S p and T p are constant matrices defined in the previous section.

Further, if A has no zero column, then L(A, B) is strictly convex in B and has a unique minimum when the critical B is

and in the autoencoder case it becomes

The proof is given in appendix A.2.

Remark 1.

Note that as long as A has no zero column, S p • (A A) is nonsingular (we will explain the reason soon).

In practice, A with zero columns can always be avoided by nudging the zero columns of A during the gradient decent process.

Proposition 2.

For any fixed matrix B ∈ R p×n the function L(A, B) is a convex function in A. Moreover, for a fixed B, the matrix A that satisfies

is a critical point of L(A, B).

The proof is given in appendix A.3.

V zero for any pair of directions (V , W ).

Therefore, the implicit equations for critical points are given below, next to the ones derived by Baldi & Hornik (1989) forL(A, B).

For L(A, B):

Remark 2.

Notice the similarity, and the difference only being the presence of Hadamard product by S p in the left and by diagonal T p in the right.

Therefore, practically, the added computational cost of evaluating the gradients is negligible compare to that of MSE loss.

The next step is to determine the structure of (A, B) that satisfies the above equations, and find the subset of those solutions that account for local minima.

For the original loss, the first expression (A ABΣ xx = A Σ yx ) is used to solve for B and put it in the second to derive an expression solely based on A. Obviously, in order to solve the first expression for B, two cases are considered separately: the case where A is of full rank p, so A A is invertible, and the case of A being of rank r < p.

Here, we assume A of any rank r ≤ p has no zero column (since this can be easily avoided in practice) and consider S p • (A A) to be always invertible.

Therefore, (A, B) define a critical point of lossesL and L if ForL(A, B) and full rank A:

For L(A, B) and no zero column A:

Before, we state the main theorem we need the following definitions.

First, a rectangular permutation matrix Π r ∈ R r×p is a matrix that each column consists of at most one nonzero element with the value 1.

If the rank of Π r is r with r < p then clearly, Π r has p − r zero columns.

Also, by taking away those zero columns the resultant r × r submatrix of Π r is a standard square permutation matrix.

Second, under the conditions provided in Assumption 1, the matrix Σ := Σ yx Σ −1 xx Σ xy has an eigenvalue decomposition Σ = U ΛU , where the i th column of U , denoted as u i , is an eigenvector of Σ corresponding to the i th largest eigenvalue of Σ, denoted as λ i .

Also, Λ = diag(λ 1 , · · · , λ n ) is the diagonal vector of ordered eigenvalues of Σ, with λ 1 > λ 2 > · · · > λ n > 0.

We use the following notation to organize a subset of eigenvectors of Σ into a rectangular matrix.

Let for any r ≤ p,

That is the columns of U Ir are the ordered orthonormal eigenvectors of Σ associated with eigenvalues λ i1 < · · · < λ ir .

Clearly, when r = p, we have

n×p and B ∈ R p×n such that A is of rank r ≤ p.

Under the conditions provided in Assumption 1 and the above notation, The matrices A and B define a critical point of L(A, B) if and only if for any given r−index set I r , and a nonsingular diagonal matrix D ∈ R r×r , A and B are of the form

where, C ∈ R r×p is of of full rank r with nonzero and normalized columns such that

−1 T p C is a rectangular permutation matrix of rank r and CΠ C = I r .

For all 1 ≤ r ≤ p, such C always exists.

In particular, if matrix A is of full rank p, i.e. r = p, the two given conditions on Π C are satisfied iff the invertible matrix C is any squared p × p permutation matrix Π. In this case (A, B) define a critical point of L(A, B) iff they are of the form

The proof is given in appendix A.4.

Remark 3.

The above theorem provides explicit equations for the critical points of the loss surface in terms of the rank of the decoder matrix A and the eigenvectors of Σ. This explicit structure allows us to further analyze the loss surface and its local/global minima.

Here, we provide a proof sketch for the above theorem to make the claims more clear.

Again as a reminder, the EVD of Σ := Σ yx Σ −1 xx Σ xy is Σ = U ΛU .

For bothL and L, the correspondinĝ B(A) is replaced by B on the RHS of critical point equations.

For the loss L(A, B), as shown in the proof of the theorem, results in the following identity

where (12) is symmetric so the RHS is symmetric too, so Λ∆ = (Λ∆) = ∆ Λ = ∆Λ. Therefore, ∆ commutes with the diagonal matrix of eigenvalues Λ. Since eigenvalues are assumed to be distinct, ∆ has to be diagonal as well.

By Lemma 2 T p (S p • (A A)) −1 T p is positive definite and U is an orthogonal matrix.

Therefore, r = rank(A) = rank(∆) = rank(U ∆U ), which implies that the diagonal matrix ∆, has r nonzero and positive diagonal entries.

There exists an r−index set I r corresponding to the nonzero diagonal elements of ∆. Forming a diagonal matrix ∆ Ir ∈ R r×r by filling its diagonal entries (in order) by the nonzero diagonal elements of ∆, we have

which indicates that the matrix A has the same column space as U Ir .

Therefore, there exists a full rank matrixC ∈ R r×p such that A = U IrC .

Since A has no zero column,C has no zero column.

Further, by normalizing the columns ofC we can write A = U Ir CD, where D ∈ R p×p is diagonal that contains the norms of columns ofC.

Baldi & Hornik (1989) did something similar for full rank A for the lossL to derive (AL = U IpC ).

But theirC can be any invertible p × p matrix.

However, in our case, the matrix C ∈ R r×p corresponding to rank r ≤ p matrix A, has to satisfy eq. (13) by replacing A by U Ir CD and eq. (12) by replacingB(A) byB(U Ir CD).

In the case of Baldi & Hornik (1989) , for the original lossL, equations similar to eq. (13) and eq. (12) appear but they are are satisfied trivially by any invertible matrixC. Simplifying those equations by using A = U Ir CD after some algebraic manipulation results in the following two conditions for C:

As detailed in proof of Theorem 1, solving for C leads to its specific structure as laid out in the theorem.

Remark 4.

Note that when A is of rank r < p with no zero columns then the invariant matrix C is not necessarily a rectangular permutation matrix but

permutation matrix with CΠ C = I r .

It is only when r = p that the invariant matrix C becomes a permutation matrix.

Nevertheless, as we show in the following corollary, the global map is always ∀r ≤ p :

xx .

It is possible to find further structure (in terms of block matrices) for the invariant matrix C when r < p.

However, this is not necessary as we soon show that all rank deficient matrix As are saddle points for the loss and ideally should be passed by during the gradient decent process.

Based on some numerical results our conjecture is that when r < p the matrix C can only start with a r × k rectangular permutation matrix of rank r with r ≤ k ≤ p and the rest of p − k columns of C is arbitrary as long as none of the columns are identically zero.

Corollary 1.

Let (A, B) be a critical point of L(A, B) under the conditions provided in Assumption 1 and rankA = r ≤ p.

Then the following are true 2.

For all 1 ≤ r ≤ p, for any critical pair (A, B), the global map G := AB becomes

The proof is given in appendix A.5.

Remark 5.

The above corollary implies that L(A, B) not only does not add any extra critical point compare to the original lossL(A, B), it provides the same global map G := AB.

It only limits the structure of the invariance matrix C as described in Theorem 1 so that the decoder matrix A can recover the exact eigenvectors of Σ.

(17) The above identity shows that the number of matrix operations required for computing the loss L(A, B) is constant and thereby independent of the value of p.

The proof is given in appendix A.6.

Theorem 2.

Let A * ∈ R n×p and B * ∈ R p×n such that A * is of rank r ≤ p.

Under the conditions provided in Assumption 1, (A * , B * ) define a local minima of the proposed loss function iff they are of the form

where the i th column of U 1:p is a unit eigenvector of Σ := Σ yx Σ −1 xx Σ xy corresponding the i th largest eigenvalue and D p is a diagonal matrix with nonzero diagonal elements.

In other words, A * contains ordered unnormalized eigenvectors of Σ corresponding to the p largest eigenvalues.

Moreover, all the local minima are global minima with the value of the loss function at those global minima being

where λ i is the i th largest eigenvalue of Σ.

The proof is given in appendix A.7.

Remark 6.

Finally, the second and third assumptions we made in the beginning in Assumption 1 can be relaxed by requiring only Σ xx to be full rank.

The output data can have a different dimension than the input.

That is Y ∈ R n×m and X ∈ R n ×m , where n = n .

The reason is that the given loss function structurally is very similar to MSE loss and can be represented as a Frobenius norm on the space of n × m matrices.

In this case the covariance matrix Σ := Σ yx Σ −1 xx Σ xy is still n × n. Clearly, for under-constrained systems with n < n the full rank assumption of Σ holds.

For the overdetermined case, where n > n the second and third assumptions in Assumption 1 can be relaxed: we only require Σ xx to be full rank since this is the only matrix that is inverted in the theorems.

Note that if p > min(n , n) then Λ Ip : the p × p diagonal matrix of eigenvalues of Σ for a p-index-set I p bounds to have some zeros and will be say rank r < p, which in turn, results in the encoder A with rank r. However, the Theorem 1 is proved for encoder of any rank r ≤ p. Finally, following theorem 2 then the first r columns of the encoder A converges to ordered eigenvectors of Σ while the p − r remaining columns span the kernel space of Σ. Moreover, Σ need not to have distinct eigenvectors.

In this case ∆ Ir becomes a block diagonal matrix, where the blocks correspond to identical eigenvalues Σ Ir .

In this case, the corresponding eigenvectors in A * are not unique but they span the respective eigenspace. , where Y = X is also applied in our experiments.

The weights of networks are initialized to random numbers with a small enough standard deviation (10 −7 in our case).

We choose to use the Adam optimizer with a scheduled learning rate (starting from 10 −3 and ending with 10 −6 in our case), which empirically benefits the optimization process.

The two training processes are stopped at the same iteration at which one of the models firstly finds all of the principal directions.

As a side note, we feed all data samples to the network at one time with batch size equal to m, although mini-batch implementations are apparently amendable.

We use the classical PCA approach to get the ground truth principal direction matrix A * ∈ R n×p , by conducting Eigen Value Decomposition (EVD) to XX ∈ R n×n or Singular Value Decomposition (SVD) to X ∈ R n×m .

As a reminder, A ∈ R n×p stands for the decoder weight matrix of an trained LAE given a loss function L. To measure the distance between A * and A, we propose an absolute cosine similarity (ACS) matrix inspired by mutual coherence (Donoho et al., 2005) , which is defined as:

where A * i ∈ R n×1 denotes the i th ground truth principal direction, and A j ∈ R n×1 denotes the j th column of the decoder A, i, j = 1, 2, . . .

, p. The elements of ACS ∈ R p×p in eq. (21) take values between [0,1], measuring pair-wise similarity across two sets of vectors.

The absolute value absorbs the sign ambiguity of principal directions.

The performances of LAEs are evaluated by defining the following metrics:

where I is the indicator function and is a manual tolerance threshold ( = 0.01 in our case).

If two vectors have absolute cosine similarity over 1 − , they are deemed equal.

Considering some columns of decoder may be correct principal directions but not in the right order, we introduce Ratio T P and Ratio F P in eqs. (22) and (23) to check the ratio of correct in-place and out-of-place principal directions respectively.

Then Ratio T otal in eq. (24) measures the total ratio of the correctly obtained principal directions by the LAE regardless of the order.

Datasets As a proof-of-concept, both synthetic data and real data are used.

For the synthetic data, 2000 zero-centered data samples are generated from a 1000-dimension zero mean multivariate normal distribution with the covariance matrix being diag(N p ).

For the real data, we choose to use MNIST dataset (LeCun et al., 1998) , which includes 60,000 grayscale handwritten digits images, each of dimension 28 × 28 = 784.

Synthetic Data Experiments In our experiment, p, the number of desired principal components (PCs), is set to 100, i.e. the dimension is to be reduced from 1000 to 100.

Figures 1 and 2 demonstrate a few conclusions.

First, during the training process, the loss ratio of both losses continuously decreases to 1, i.e. they both converge to the optimal loss value.

However, when both get close enough, L require more iterations since the optimizer is forced to find the right directions: it fully converges only after it has found all the principal directions in the right order.

Second, using the loss L results in finding more and more correct principal directions, with Ratio T P continuously rising; and ultimately affords all correct and ordered principal directions, Loss Ratio

Convergence of L andL to their corresponding optimum loss

Performance of finding the principal directions for both L andL Figure 2 : Performance of both losses L andL in finding the principal directions at the columns of their respective decoders.

with Ratio T P ending with 100%.

Notice that occasionally and temporarily, some of the principal directions is found but not at their correct position, which is indicated by the rise of Ratio F P in the figure.

However, as optimization continues they are shifted to the right column, which results in Ratio F P going back to zero, and Ratio T P reaching one.

As forL, it fails to identify any principal directions; both Ratio T P and Ratio F P forL stay at 0, which indicates that none of the columns of the decoderÃ, aligns with any principal direction.

Third, as shown in the figure, while the optimizer finds almost all the principal directions rather quickly, it requires much more iterations to find some final ones.

This is because some eigenvalues in the empirical covariance matrix of the finite 2000 samples become very close (the difference becomes less than 1).

Therefore, the loss has to get very close to the optimal loss, making the gradient of the loss hard to distinguish between the two.

We set the number of principal components (PCs) as 100, i.e., the dimension is to be reduced from 784 to 100.

We also try to reconstruct with the top-10 columns found in this case.

As in Fig. 3 , the reconstruction performance of L is consistently better thanL. That also reflects thatL does not identify PCs, while L is directly applicable to performing PCA without bells and whistles.

In this paper, we have introduced a loss function for performing principal component analysis and linear regression using linear autoencoders.

We have proved that the optimizing with the given loss results in the decoder matrix converges to the exact ordered unnormalized eigenvectors of the sample covariance matrix.

We have also demonstrated the claims on a synthetic data set of random samples drawn from a multivariate normal distribution and on MNIST data set.

There are several possible generalizations of this approach we are currently working on.

One is improving performance when the corresponding eigenvalues of two principal directions are very close and another is generalization of the loss for tensor decomposition.

Before we present the proof for the main theorems, the following two lemmas introduce some notations and basic relations that are required for the proofs.

Lemma 2.

The constant matrices T p ∈ R p×p and S p ∈ R p×p are defined as

Clearly, the diagonal matrix T p is positive definite.

Another matrix that will appear in the formula- The following properties of Hadamard product and matrices T p and S p are used throughout:

where, • is the Hadamard (element-wise) product.

Moreover, if Π 1 , Π 2 ∈ R p×p are permutation matrices then

3.

S p is invertible and its inverse is a symmetric tridiagonal matrix

4.

S p is positive definite.

is positive semidefinite.

If (not necessarily full rank) A has no zero column then S p • (A A) is positive definite.

7.

Let D D D, E E E ∈ R p×p be positive semidefinite matrices, where E E E has no zero diagonal element, and D D D is of rank r ≤ p. Also, let for any r ≤ p, 2.

This is a standard result (Horn & Johnson, 2012) , and no proof is needed.

3.

Directly compute S p S −1 p :

4. Firstly, note that S −1 p is symmetric and nonsingular so all the eigenvalues are real and nonzero.

It is also a diagonally dominant matrix (Horn & Johnson (2012) , Def 6.1.9) since ∀i ∈ {1, · · · , p} :

where the inequality is strict for the first and the last row and it is equal for the rows in the middle.

Moreover, by Gersgorin circle theorem (Horn & Johnson (2012)

Since ∀i : C i ≥ R i we have all the eigenvalues are non-negative.

They are also nonzero, hence, S −1 p is positive definite, which implies S p is also positive definite.

6.

Clearly, the matrix T p is achieved by setting the off-diagonal elements of S p to zero.

Hence,

For the diagonal matrices Hadamard product and matrix product are interchangeable so the latter may also be written as T p D D D. The same argument applies for the second identity.

7.

This property can easily be proved by induction on p and careful bookkeeping of indices.

Lemma 3 (Simultaneous diagonalization by congruence).

Let M 1 , M 2 ∈ R p×p , where M 1 is positive definite and M 2 is positive semidefinite.

Also, let D D D, E E E ∈ R r×r be positive definite diagonal matrices with r ≤ p.

Further, assume there is a C ∈ R r×p of rank r ≤ p such that

Then there exists a nonsingularC ∈ R p×p that its first r rows are the matrix C and

in which E E E ∈ R p−r×p−r is a nonnegative diagonal matrix.

Clearly, the rank of M 2 is r plus the number of nonzero diagonal elements of E E E .

Proof.

The proof is rather straightforward since this lemma is the direct consequence of Theorem 7.6.4 in Horn & Johnson (2012) .

The theorem basically states that if M 1 , M 2 ∈ R p×p is symmetric and M 1 is positive definite then there exists an invertible S ∈ R p×p such that SM 1 S = I p and SM 2 S is a diagonal matrix with the same inertia as M 2 .

Here, we have M 2 that is positive semidefinite and C ∈ R r×p of rank r ≤ p such that

Therefore, since S is of full rank p and

2 C is of rank r ≤ p, there exists p − r rows in S that are linearly independent of rows of D D D −1 2 C. EstablishC ∈ R p×p by adding those p − r rows to C. ThenC has p linearly independent rows so it is nonsingular, and fulfills the lemma's proposition that isC

Lemma 4.

Let A and B define a critical point of L. Further, let V ∈ R n×p and W ∈ R p×n are such that

Further, for

Finally, in case the critical A is of full rank p and so, (A, B) = (U Ip ΠD,B(U Ip ΠD)), for the encoder direction V with V F = O(ε) and W =W we have,

Proof.

As described in appendix B.1, the second order Taylor expansion for the loss L(A, B) is then given by eq. (63), i.e.

Now, based on the first item in Corollary 1, BΣ xx B is a p×p diagonal matrix, so based on eq. (27):

The substitution then yields eq. (29).

Finally, in the above equation

xx .

We have

Replace the above in eq. (30) and simplify:

which finalizes the proof.

For this proof we use the first and second order derivatives for L(A, B) wrt B derived in Lemma 5.

From eq. (66), we have that for a given A the second derivative wrt to B of the cost L (A, B) at B, and in the direction W is the quadratic form

The matrix Σ xx is positive-definite and by Lemma 2,

is convex in coefficients of B for a fixed matrix A. Also the critical points of L(A, B) for a fixed A is a matrix B that satisfies ∀W ∈ R p×n : d B L(A, B)W = 0 and hence, from eq. (64) we have

For a fixed A, the cost L(A, B) is convex in B, so any matrix B that satisfies the above equation corresponds to a minimum of L (A, B) .

Further, if A has no zero column then by Lemma 2,

Therefore, the cost L(A, B) becomes strictly convex and the unique global minimum is achieved at B =B(A) as defined in eq. (6).

For this proof we use the first and second order derivatives for L(A, B) wrt A derived in Lemma 6.

For a fixed B, based on eq. (69) the second derivative wrt to A of L(A, B) at A, and in the direction V is the quadratic form

The matrix Σ xx is positive-definite and by Lemma 2,

is convex in coefficients of A for a fixed matrix B. Based on eq. (67) the critical point of L(A, B) for a fixed B is a matrix A that satisfies for all

which is eq. (7).

Before we start, a reminder on notation and some useful identities that are used throughout the proof.

The matrix Σ := Σ yx Σ −1 xx Σ xy has an eigenvalue decomposition Σ = U ΛU , where the i th column of U , denoted as u i , is an eigenvector of Σ corresponding to the i th largest eigenvalue of Σ, denoted as λ i .

Also, Λ = diag(λ 1 , · · · , λ n ) is the diagonal vector of ordered eigenvalues of Σ, with λ 1 > λ 2 > · · · > λ n > 0.

We use the following notation to organize a subset of eigenvectors of Σ into a rectangular matrix.

Let for any r ≤ p,

That is the columns of U Ir are the ordered orthonormal eigenvectors of Σ associated with eigenvalues λ i1 < · · · < λ ir .

The following identities are then easy to verify:

The sufficient condition:

Let A ∈ R n×p of rank r ≤ p and no zero column be given by eq. (8), B ∈ R p×n given by eq. (9), and the accompanying conditions are met.

Notice that U Ir U Ir = I r implies that DC CD = DC U Ir U Ir CD = A A, so

xx =B(A), which is eq. (6).

Therefore, based on Proposition 1, for the given A, the matrix B defines a critical point of L (A, B) .

For the gradient wrt to A, first note that with B given by eq. (9) we have

The matrix Π C is a rectangular permutation matrix so Π C Λ Ir Π C is diagonal so as

.

Therefore, BΣ xx B is diagonal and by eq. (27) in Lemma 2-6 we have

which is eq. (7).

Therefore, based on Proposition Proposition 2, for the given B, the matrix A define a critical point of L(A, B).

Hence, A and B together define a critical point of L(A, B).

Based on Proposition 1 and Proposition 2, for A (with no zero column) and B, to define a critical point of L (A, B) , B has to beB(A) given by eq. (6), and A has to satisfy eq. (7).

That is

where, ∆ := U AT p (S p • (A A)) −1 T p A U is symmetric and positive semidefinite.

The LHS of the above equation is symmetric so the RHS is symmetric too, so Λ∆ = (Λ∆) = ∆ Λ = ∆Λ. Therefore, ∆ commutes with the diagonal matrix of eigenvalues Λ. Since, eigenvalues are assumed to be distinct, ∆ has to be diagonal as well.

By Lemma 2 T p (S p • (A A)) −1 T p is positive definite and U is an orthogonal matrix.

Therefore, r = rank(A) = rank(∆) = rank(U ∆U ), which implies that the diagonal matrix ∆, has r nonzero and positive diagonal entries.

There exists an r−index set I r corresponding to the nonzero diagonal elements of ∆. Forming a diagonal matrix ∆ Ir ∈ R r×r by filling its diagonal entries (in order) by the nonzero diagonal elements of ∆ we have

which indicates that the matrix A has the same column space as U Ir .

Therefore, there exists a full rank matrixC ∈ R r×p such that A = U IrC .

Since A has no zero column,C has no zero column.

Further, by normalizing the columns ofC we can write A = U Ir CD, where D ∈ R p×p is diagonal that contains the norms of columns ofC. Therefore, A is exactly in the form given by eq. (8).

The matrix C has to satisfy eq. (38) that is

Now that the structure of A has been identified, evaluateB(A) of eq. (6) by setting A = U Ir CD, that is

xx , which by defining Π C := (S p • (C C)) −1 T p C gives eq. (34) for B as claimed.

While C has to satisfy eq. (39), A and B in the given form have to satisfy eq. (37) that provides another condition for C as follows.

First, note that

Now, replace A and B in eq. (37) by their respective identities that we just derived.

Performing the same process for eq. (37) we have

Now we have to find C such that it satisfies eq. (39) and eq. (40).

To make the process easier to follow, lets have them in one place.

The matrix C ∈ R r×p have to satisfy

Since C is a rectangular matrix, solving above equations for C in this form seems intractable.

We use a trick to temporarily extend C into an invertible square matrix as follows.

•

is positive definite and M 2 is positive semidefinite, so they are simultaneously diagonalizable by congruence that is based on Lemma 3 and eq. (41) and eq. (42), there exists a nonsingularC ∈ R p×p such that C consists of the first r rows ofC andC

where,∆ Ir = ∆ Ir ⊕ I r−p is a p × p diagonal matrix andΛ Ir = Λ Ir ⊕ Λ is another p × p diagonal matrix, in which Λ ∈ R r−p×r−p is a nonnegative diagonal matrix.

• Substitute∆ Ir from eq. (43) in eq. (44), then left multiply byC −1 , and right multiply bȳ C I r;p :C

• Now we can revert back everything to C again.

Since C consists of the first r rows ofC we haveC I r;pC = C C, andC I r;pΛIrC = C Λ Ir C, which turns the above equation into

• In the above equation, replace

• By the second property of Lemma 2 we can collect diagonal matrices T −1 p 's around S p to arrive at

Both D D D r and E E E r in the above identity are positive semidefinite.

Moreover, since by assumption C has no zero columns, E E E r has no zero diagonal element.

Then the 7 th property of Lemma 2 implies the following two conclusions:

1.

The matrix D D D r is diagonal.

The rank of D D D r is r so it has exactly r positive diagonal elements and the rest is zero.

This argument is true for T

−1 T p C of rank r should have p − r zero rows.

Let J r be an r−index set corresponding to nonzero diagonal elements of Π C Λ Ir Π C .

Then the matrix Π C [J r , N r ] (r × r submatrix of Π C consist of its J r rows) is nonsingular.

2.

For every i, j ∈ J r and i = j, (E E E r ) i,j = 0.

Since E E E r := C C and so (E E E r ) i,j is the inner product of i th and j th columns of C, we conclude that the columns of C[N r , J r ] (r × r submatrix of C consist of its J r columns) are orthogonal or in other words C[N r , J r ] C[N r , J r ] is diagonal.

The columns of C are normalized.

Therefore, C[N r , J r ] C[N r , J r ] = I r and hence, C[N r , J r ] is an orthogonal matrix.

• We use the two conclusions to solve the original eq. (41) and eq. (42).

First use Π C := (S p • (C C)) −1 T p C to shrink them into :

Next, by the first conclusion, the matrix T

which is one of the two claimed conditions.

What is left is to show that Π C is a rectangular permutation matrix.

From the first conclusion we also have Π C has exactly r nonzero columns indexed by

By the second conclusion C[N r , J r ] is an orthogonal matrix therefore,

is an r × r positive definite diagonal matrix with Λ Ir having distinct diagonal elements, and

should be a square permutation matrix.

Putting back the zero columns, we conclude that C should be such that

permutation matrix and CΠ C = I r .

Note that it is possible to further analyze these conditions and determine the exact structure of C. However, this is not needed in general for the critical point analysis of the next theorem except for the case where r = p and C is a square invertible matrix.

In this case, square matrix Π C is of full rank p,

p T p Π = Π , which verifies eq. (10) and eq. (11) for A and B when A is of full rank p.

A.5 PROOF OF COROLLARY 1 1.

We already show in the proof Theorem 1 that for critical (A, B) the matrix BΣ xx B is given by eq. (36) that is

The matrix Π C is a p × r rectangular permutation matrix so

The diagonal matrix Λ Ir is of rank r therefore, BΣ xx B is of rank r. (A, B) is of the form given by eq. (8) and eq. (9) with the proceeding conditions on the invariance C. Therefore, the global map is

(49) Again by assumption (A, B) define a critical point of L(A, B) so by Theorem 1 they are of the form given by eq. (8) and eq. (9) with the proceeding conditions on the invariance C. Hence,

Hence, eq. (48) is satisfied.

For the second equation we use the first property of this corollary that is BΣ xx B is diagonal and satisfy eq. (7) of Proposition 2 that is

Hence, the second condition, eq. (49) is also satisfied.

Therefore, any critical point of L(A, B) is a critical point ofL(A, B).

A.6 PROOF OF LEMMA 1

Proof.

We have

which is eq. (17).

Proof.

The full rank matrices A * and B * given by eq. (18) and eq. (19) are clearly of the form given by Theorem 1 with I p = N p := {1, 2, · · · , p}, and Π p = I p .

Hence, they define a critical point of L (A, B) .

We want to show that these are the only local minima, that is any other critical (A, B) is a saddle points.

The proof is similar to the second partial derivative test.

However, in this case the Hessian is a forth order tensor.

Therefore, the second order Taylor approximation of the loss, derived in Lemma 4, is used directly.

To prove the necessary condition, we show that at any other critical point (A, B) , where the first order derivatives are zero, there exists infinitesimal direction along which the second derivative of loss is negative.

Next, for the sufficient condition we show that the any critical point of the form (A * , B * ) is a local and global minima.

Recall that U Ip is the matrix of eigenvectors indexed by the p−index set I p and Π is a p × p permutation matrix.

Since all the index sets I r , r ≤ p are assumed to be ordered, the only way to have U Np = U Ip Π is by having I p = N p and Π = I p .

Let A (with no zero column) and B define an arbitrary critical point of L(A, B).

Then Based on the previous theorem, either A = U Ir C with r < p or A = U Ip ΠD while in both cases B =B(A) given by eq. (6).

If (A, B) is not of the form of (A * , B * ) then there are three possibilities either 1) A = U Ir CD with r < p, or 2)

The first two cases corresponds to not having the "right" and/or "enough" eigenvectors, and the third corresponds to not having the "right" ordering.

We introduce the following notation and investigate each case separately.

Let ε > 0 and U i;j ∈ R n×p be a matrix of all zeros except the i th column, which contains u j ; the eigenvector of Σ corresponding to the j th largest eigenvalue.

Therefore,

where, E i ∈ R p×p is matrix of zeros except the i th diagonal element that contains 1.

In what follows, for each case we define a encoder direction V ∈ R n×p with V F = O(ε), and set the decoder

xx .

Then we use eq. (30) and eq. (31) of Lemma 4, to show that the given direction (V , W ) infinitesimally reduces the loss and hence, in every case the corresponding critical (A, B) is a saddle point.

1.

For the case A = U Ir CD, with r < p, note that based on the first item in Corollary 1, BΣ xx B is a p×p diagonal matrix of rank r so it has p−r zero diagonal elements.

Pick an i ∈ N p such that (BΣ xx B ) ii is zero and a j ∈ N p \ I r .

Set V = εU i;j D and W =W .

Clearly,

Notice, V F , W F = O(ε), so based on eq. (30) of Lemma 4, we have

is a positive definite matrix, as

.

Hence, any (A, B) = (U Ir CD,B(U Ir CD)) with r < p is a saddle point.

2.

Next, consider the case where A = U Ip ΠD with I p = N p .

Then there exists at least one j ∈ I p \ N p and i ∈ N p \

I p such that i < j (so λ i > λ j ).

Let σ be the permutation corresponding to the permutation matrix Π. Also, let ε > 0 and U σ(j);i ∈ R n×p be a matrix of all zeros except the σ(j) th column, which contains u i ; the eigenvector of Σ corresponding to the i th largest eigenvalue.

Set V = εU σ(j);i D and W =W .

Then, since i / ∈ I p we have

Since V F , W F = O(ε), based on eq. (31) of Lemma 4, we have

Note that in the above, the diagonal matrix Π Λ Ip Π has the same diagonal elements as Λ Ip but they are permuted by σ.

So E σ(j) Π Λ Ip Π selects σ(j) th diagonal element of Π Λ Ip Π that is the j th diagonal element of Λ Ip , which is nothing but λ j .

Now, since i < j

3.

Finally consider the case where A = U Np ΠD with Π = I p .

Since Π = I p , the permutation σ of the set N p , corresponding to the permutation matrix Π, has at least a cycle

Hence, Π can be decomposed as Π = Π (i1i2···i k )Π , whereΠ is the permutation matrix corresponding to other cycles of σ.

The cycle (i 1 i 2 · · · i k ) can be decomposed into transpositions as

Note that Π (i k i1) , the permutation matrix corresponding to transposition (i k i 1 ) is a symmetric involutory matrix, i.e. Π 2 (i k i1) = I p .

Set V = ε(U i1;i1 −U i k ;i k )ΠD and W =W .

Again we replace V and W in eq. (31) of Lemma 4.

There are some tedious steps to simplify the equation, which is given in appendix A.7.1.

The final result is as follows.

With the given V and W , the third and forth terms of the RHS of eq. (31) are canceled and the first two terms are simplified to

in which, m = max{k − 1, 2}. This means that If the selected cycle is just a transposition

By the above definition of i m , we have i m − i 1 > 0 and since

Hence, the first term in the above equation is negative and as ε → 0, we have L(A + V , B + W ) − L(A, B) < 0.

Therefore, any any (A, B) = (U Ip ΠD,B(U Ip ΠD)) with Π =

I p is a saddle point.

From Lemma 1 we know that the loss L(A, B) can be written in the form of eq. (17).

Use this

which is eq. (20), as claimed.

Notice that the above value is independent of the diagonal matrix D p .

From the necessary condition we know that any critical point not in the form of (A * , B * ) is a saddle point.

Hence, due to the convexity of the loss at least one (A * , B * ) is a global minimum but since the value of the loss at (A * , B * ) is independent of D p all these critical points yield the same value for the loss.

Therefore, any critical point in the form of (A * , B * ) is a local and global minima.

We investigate each term on the RHS separately.

but before note that

where,σ and its function inverseσ −1 are permutations corresponding toΠ andΠ respectively.

ΠT pΠ is a diagonal matrix where diagonal elements of T p are ordered based onσ −1 .

Moreover, recall that we decomposed the permutation matrix Π in A with a cycle

, where i 1 , i 2 , · · · i k are fixed points ofΠ. Therefore, withσ being the permutation corresponding toΠ we havẽ

where, m = max{k − 1, 2}. This means that If the selected cycle is just a transposition

For the first term we have

Under review as a conference paper at ICLR 2020

which is eq. (57) as claimed.

For the second term we have Finally, we have to show that the third and the forth terms of the eq. (31) are canceled.

First, observe that

Now, note that in both cases the matrices that are multiplied elementwise with ΠS pΠ are diagonal and hence, we only need to look at diagonal elements of ΠS pΠ .

Moreover,

where, i 1 · · · i k are fixed points of permutation corresponding toΠ soΠS pΠ has the same values at diagonal positions i 1 and i k as the original matrix S p .

The only permutation that is only on the left side is Π (i1i k ) which exchanges the i 1 and i k rows of S p .

Since S p is such that the elements at each row before the diagonal element are the same and i k > i 1 , we have the i 1 and i k diagonal elements of ΠS pΠ have the same value.

Let that value be denoted as s. Then the sum of the above two equations yields m(λ i1 + λ i k ) − m(λ i1 + λ i k ) = 0, as claimed.

In order to derive and analyze the critical points of the cost function which is a real-valued function of matrices we use the first and second order Fréchet derivatives as described in chapter 4 of Zeidler (1995) .

For a function f : R n×m → R the first order Fréchet derivative at the point A ∈ R n×m is a linear functional df (A) : R n×m → R such that

where, if V F , W F = O(ε) then R(V , W ) = O(ε 3 ).

Clearly, as at critical points where d A L(A, B)V +d B L(A, B)W = 0, as ε → 0 we have R V ,W (A, B) → 0 and the sign of the sum of the second order partial Fréchet derivatives determines the type of the critical point very much similar to second partial derivative test for two variable functions.

However, here for local minima we have to show the sign is positive in all directions and for saddle points have to show the sign is positive in some directions and negative at least in on direction.

Finally, note that the smoothness of the loss entails that Fréchet derivative and directional derivative (Gateaux) both exist and (foregoing some subtleties in definition) are the same.

@highlight

A new loss function for PCA with linear autoencoders that provably yields ordered exact eigenvectors 