Partial differential equations (PDEs) are widely used across the physical and computational sciences.

Decades of research and engineering went into designing fast iterative solution methods.

Existing solvers are general purpose, but may be sub-optimal for specific classes of problems.

In contrast to existing hand-crafted solutions, we propose an approach to learn a fast iterative solver tailored to a specific domain.

We achieve this goal by learning to modify the updates of an existing solver using a deep neural network.

Crucially, our approach is proven to preserve strong correctness and convergence guarantees.

After training on a single geometry, our model generalizes to a wide variety of geometries and boundary conditions, and achieves 2-3 times speedup compared to state-of-the-art solvers.

Partial differential equations (PDEs) are ubiquitous tools for modeling physical phenomena, such as heat, electrostatics, and quantum mechanics.

Traditionally, PDEs are solved with hand-crafted approaches that iteratively update and improve a candidate solution until convergence.

Decades of research and engineering went into designing update rules with fast convergence properties.

The performance of existing solvers varies greatly across application domains, with no method uniformly dominating the others.

Generic solvers are typically effective, but could be far from optimal for specific domains.

In addition, high performing update rules could be too complex to design by hand.

In recent years, we have seen that for many classical problems, complex updates learned from data or experience can out-perform hand-crafted ones.

For example, for Markov chain Monte Carlo, learned proposal distributions lead to orders of magnitude speedups compared to handdesigned ones BID20 BID12 .

Other domains that benefited significantly include learned optimizers BID1 and learned data structures BID9 .

Our goal is to bring similar benefits to PDE solvers.

Hand-designed solvers are relatively simple to analyze and are guaranteed to be correct in a large class of problems.

The main challenge is how to provide the same guarantees with a potentially much more complex learned solver.

To achieve this goal, we build our learned iterator on top of an existing standard iterative solver to inherit its desirable properties.

The iterative solver updates the solution at each step, and we learn a parameterized function to modify this update.

This function class is chosen so that for any choice of parameters, the fixed point of the original iterator is preserved.

This guarantees correctness, and training can be performed to enhance convergence speed.

Because of this design, we only train on a single problem instance; our model correctly generalizes to a variety of different geometries and boundary conditions with no observable loss of performance.

As a result, our approach provides: (i) theoretical guarantees of convergence to the correct stationary solution, (ii) faster convergence than existing solvers, and (iii) generalizes to geometries and boundary conditions very different from the ones seen at training time.

This is in stark contrast with existing deep learning approaches for PDE solving BID21 BID6 that are limited to specific geometries and boundary conditions, and offer no guarantee of correctness.

Our approach applies to any PDE with existing linear iterative solvers.

As an example application, we solve the 2D Poisson equations.

Our method achieves a 2-3?? speedup on number of multiplyadd operations when compared to standard iterative solvers, even on domains that are significantly different from our training set.

Moreover, compared with state-of-the-art solvers implemented in FEniCS BID13 , our method achieves faster performance in terms of wall clock CPU time.

Our method is also simple as opposed to deeply optimized solvers such as our baseline in FEniCS (minimal residual method + algebraic multigrid preconditioner).

Finally, since we utilize standard convolutional networks which can be easily parallelized on GPU, our approach leads to an additional 30?? speedup when run on GPU.

In this section, we give a brief introduction of linear PDEs and iterative solvers.

We refer readers to BID11 for a thorough review.

Linear PDE solvers find functions that satisfy a (possibly infinite) set of linear differential equations.

More formally, let F = {u : R k ??? R} be the space of candidate functions, and A : F ??? F be a linear operator; the goal is to find a function u ??? F that satisfies a linear equation Au = f, where f is another function R k ??? R given by our problem.

Many PDEs fall into this framework.

For example, heat diffusion satisfies DISPLAYFORM0 is the linear Laplace operator; u maps spatial coordinates (e.g. in R 3 ) into its temperature, and f maps spatial coordinates into the heat in/out flow.

Solving this equation lets us know the stationary temperature given specified heat in/out flow.

Usually the equation Au = f does not uniquely determine u. For example, u = constant for any constant is a solution to the equation ??? 2 u = 0.

To ensure a unique solution we provide additional equations, called "boundary conditions".

Several boundary conditions arise very naturally in physical problems.

A very common one is the Dirichlet boundary condition, where we pick some subset G ??? R k and fix the values of the function on G to some fixed value b, u(x) = b(x), for all x ??? G where the function b is usually clear from the underlying physical problem.

As in previous literature, we refer to G as the geometry of the problem, and b as the boundary value.

We refer to the pair (G, b)as the boundary condition.

In this paper, we only consider linear PDEs and boundary conditions that have unique solutions.

Most real-world PDEs do not admit an analytic solution and must be solved numerically.

The first step is to discretize the solution space DISPLAYFORM0 where D is a discrete subset of R. When the space is compact, it is discretized into an n ?? n ?? n ?? ?? ?? (k many) uniform Cartesian grid with mesh width h. Any function in F is approximated by its value on the n k grid points.

We denote the discretized function as a vector u in R n k .

In this paper, we focus on 2D problems (k = 2), but the strategy applies to any dimension.

We discretize all three terms in the equation Au = f and boundary condition (G, b).

The PDE solution u is discretized such that u i,j = u(x i , y j ) corresponds to the value of u at grid point (x i , y j ).

We can similarly discretize f and b. In linear PDEs, the linear operator A is a linear combination of partial derivative operators.

For example, for the Poisson equation DISPLAYFORM1 .

Therefore we can first discretize each partial derivative, then linearly combine the discretized partial derivatives to obtain a discretized A.Finite difference is a method that approximates partial derivatives in a discretized space, and as mesh width h ??? 0, the approximation approaches the true derivative.

For example, DISPLAYFORM2 , the Laplace operator in 2D can be correspondingly approximated as: DISPLAYFORM3 After discretization, we can rewrite Au = f as a linear matrix equation DISPLAYFORM4 where u, f ??? R n 2 , and A is a matrix in R n 2 ??n 2 (these are n 2 dimensional because we focus on 2D problems).

In many PDEs such as the Poisson and Helmholtz equation, A is sparse, banded, and symmetric.

We also need to include the boundary condition u(x) = b(x) for all x ??? G. If a discretized point (x i , y j ) belongs to G, we need to fix the value of u i,j to b i,j .

To achieve this, we first define e ??? {0, 1} n 2 to be a vector of 0's and 1's, in which 0 indicates that the corresponding point belongs to G. Then, we define a "reset" matrix DISPLAYFORM0 Intuitively G "masks" every point in G to 0.

Similarly, I ??? G can mask every point not in G to 0.

Note that the boundary values are fixed and do not need to satisfy Au = f .

Thus, the solution u to the PDE under geometry G should satisfy: DISPLAYFORM1 The first equation ensures that the interior points (points not in G) satisfy Au = f , and the second ensures that the boundary condition is satisfied.

To summarize, (A, G, f, b, n) is our PDE problem, and we first discretize the problem on an n ?? n grid to obtain (A, G, f, b, n).

Our objective is to obtain a solution u that satisfies Eq. (4), i.e. Au = f for the interior points and boundary condition u i,j = b i,j , ???(x i , y j ) ??? G.

A linear iterative solver is defined as a function that inputs the current proposed solution u ??? R n 2 and outputs an updated solution u .

Formally it is a function ?? : DISPLAYFORM0 where T is a constant update matrix and c is a constant vector.

For each iterator ?? there may be special vectors u * ??? R n 2 that satisfy u * = ??(u * ).

These vectors are called fixed points.

The iterative solver ?? should map any initial u 0 ??? R n 2 to a correct solution of the PDE problem.

This is formalized in the following theorem.

Definition 1 (Valid Iterator).

An iterator ?? is valid w.r.t.

a PDE problem (A, G, f, b, n) if it satisfies: a) Convergence: There is a unique fixed point u * such that ?? converges to u * from any initialization: DISPLAYFORM1 The fixed point u * is the solution to the linear system Au = f under boundary condition (G, b).

It is important to note that Condition (a) only depends on T and not the constant c.

Fixed Point: Condition (b) in Definition 1 contains two requirements: satisfy Au = f , and the boundary condition (G, b).

To satisfy Au = f a standard approach is to design ?? by matrix splitting: BID11 .

This naturally suggests the iterative update DISPLAYFORM2 DISPLAYFORM3 Because Eq. FORMULA11 is a rewrite of Au = f , stationary points u * of Eq. (6) satisfy Au * = f .

Clearly, the choices of M and N are arbitrary but crucial.

From Theorem 1, we must choose M such that the update converges.

In addition, M ???1 must easy to compute (e.g., diagonal).Finally we also need to satisfy the boundary condition (I ??? G)u = (I ??? G)b in Eq.4.

After each update in Eq. FORMULA11 , the boundary condition could be violated.

We use the "reset" operator defined in Eq. FORMULA6 to "reset" the values of u i,j to b i,j by Gu DISPLAYFORM4 The final update rule becomes DISPLAYFORM5 Despite the added complexity, it is still a linear update rule in the form of u = T u + c in Eq. FORMULA8 : DISPLAYFORM6 As long as M is a full rank diagonal matrix, fixed points of this equation satisfies Eq. (4).

In other words, such a fixed point is a solution of the PDE problem (A, G, f, b, n).

Proposition 1.

If M is a full rank diagonal matrix, and u * ??? R n 2 ??n 2 satisfies Eq. (7) , then u * satisfies Eq. (4).

A simple but effective way to choose M is the Jacobi method, which sets M = I (a full rank diagonal matrix, as required by Proposition 1).

For Poisson equations, this update rule has the following form,?? DISPLAYFORM0 For Poisson equations and any geometry G, the update matrix T = G(I ??? A) has spectral radius ??(T ) < 1 (see Appendix B).

In addition, by Proposition 1 any fixed point of the update rule Eq.(8,9) must satisfy Eq. (4).

Both convergence and fixed point conditions from Definition 1 are satisfied: Jacobi iterator Eq.(8,9) is valid for any Poisson PDE problem.

In addition, each step of the Jacobi update can be implemented as a neural network layer, i.e., Eq.

The Jacobi method has very slow convergence rate BID11 .

This is evident from the update rule, where the value at each grid point is only influenced by its immediate neighbors.

To propagate information from one grid point to another, we need as many iterations as their distance on the grid.

The key insight of the Multigrid method is to perform Jacobi updates on a downsampled (coarser) grid and then upsample the results.

A common structure is the V-cycle BID3 .

In each V-cycle, there are k downsampling layers followed by k upsampling layers, and multiple Jacobi updates are performed at each resolution.

The downsampling and upsampling operations are also called restriction and prolongation, and are often implemented using weighted restriction and linear interpolation respectively.

The advantage of the multigrid method is clear: on a downsampled grid (by a factor of 2) with mesh width 2h, information propagation is twice as fast, and each iteration requires only 1/4 operations compared to the original grid with mesh width h.

A PDE problem consists of five components (A, G, f, b, n).

One is often interested in solving the same PDE class A under varying f, discretization n, and boundary conditions (G, b).

For example, solving the Poisson equation under different boundary conditions (e.g., corresponding to different mechanical systems governed by the same physics).

In this paper, we fix A but vary G, f, b, n, and learn an iterator that solves a class of PDE problems governed by the same A. For a discretized PDE problem (A, G, f, b, n) and given a standard (hand designed) iterative solver ??, our goal is to improve upon ?? and learn a solver ?? that has (1) correct fixed point and (2) fast convergence (on average) on the class of problems of interest.

We will proceed to parameterize a family of ?? that satisfies (1) by design, and achieve (2) by optimization.

In practice, we can only train ?? on a small number of problems (A, f i , G i , b i , n i ).

To be useful, ?? must deliver good performance on every choice of G, f, b, and different grid sizes n.

We show, theoretically and empirically, that our iterator family has good generalization properties: even if we train on a single problem (A, G, f, b, n), the iterator performs well on very different choices of G, f, b, and grid size n.

For example, we train our iterator on a 64 ?? 64 square domain, and test on a 256 ?? 256 L-shaped domain (see FIG4 ).

For a fixed PDE problem class A, let ?? be a standard linear iterative solver known to be valid.

We will use more formal notation ??(u; G, f, b, n) as ?? is a function of u, but also depends on G, f, b, n. Our assumption is that for any choice of G, f, b, n (but fixed PDE class A), ??(u; G, f, b, n) is valid.

We previously showed that Jacobi iterator Eq.(8,9) have this property for the Poisson PDE class.

where H is a learned linear operator (it satisfies H0 = 0).

The term GHw can be interpreted as a correction term to ??(u; G, f, b, n).

When there is no confusion, we neglect the dependence on G, f, b, n and denote as ??(u) and ?? H (u).?? H should have similar computation complexity as ??. Therefore, we choose H to be a convolutional operator, which can be parameterized by a deep linear convolutional network.

We will discuss the parameterization of H in detail in Section 3.4; we first prove some parameterization independent properties.

The correct PDE solution is a fixed point of ?? H by the following lemma: Lemma 1.

For any PDE problem (A, G, f, b, n) and choice of H, if u * is a fixed point of ??, it is a fixed point of ?? H in Eq. (10).Proof.

Based on the iterative rule in Eq. (10), if u * satisfies ??(u DISPLAYFORM0 Moreover, the space of ?? H subsumes the standard solver ??. If H = 0, then ?? H = ??. Furthermore, denote ??(u) = T u + c, then if H = T , then since GT = T (see Eq. (7) ), DISPLAYFORM1 which is equal to two iterations of ??. Computing ?? requires one convolution T , while computing ?? H requires two convolutions: T and H. Therefore, if we choose H = T , then ?? H computes two iterations of ?? with two convolutions: it is at least as efficient as the standard solver ??.

We train our iterator ?? H (u; G, f, b, n) to converge quickly to the ground truth solution on a set DISPLAYFORM0 of problem instances.

For each instance, the ground truth solution u * is obtained from the existing solver ??. The learning objective is then DISPLAYFORM1 Intuitively, we look for a matrix H such that the corresponding iterator ?? H will get us as close as possible to the solution in k steps, starting from a random initialization u 0 sampled from a white Gaussian.

k in our experiments is uniformly chosen from [1, 20] , similar to the procedure in BID20 .

Smaller k is easier to learn with less steps to back-propagate through, while larger k better approximates our test-time setting: we care about the final approximation accuracy after a given number of iteration steps.

Combining smaller and larger k performs best in practice.

We show in the following theorem that there is a convex open set of H that the learning algorithm can explore.

To simplify the statement of the theorem, for any linear iterator ??(u) = T u + c we will refer to the spectral radius (norm) of ?? as the spectral radius (norm) of T .

Theorem 2.

For fixed G, f, b, n, the spectral norm of ?? H (u; G, f, b, n) is a convex function of H, and the set of H such that the spectral norm of ?? H (u; G, f, b, n) < 1 is a convex open set.

Proof.

See Appendix A.Therefore, to find an iterator with small spectral norm, the learning algorithm only has to explore a convex open set.

Note that Theorem 2 holds for spectral norm, whereas validity requires small spectral radius in Theorem 1.

Nonetheless, several important PDE problems (Poisson, Helmholtz, etc) are symmetric, so it is natural to use a symmetric iterator, which means that spectral norm is equal to spectral radius.

In our experiments, we do not explicitly enforce symmetry, but we observe that the optimization finds symmetric iterators automatically.

For training, we use a single grid size n, a single geometry G, f = 0, and a restricted set of boundary conditions b. The geometry we use is a square domain shown in FIG4 .

Although we train on a single domain, the model has surprising generalization properties, which we show in the following: Proposition 2.

For fixed A, G, n and fixed H, if for some f 0 , b 0 , ?? H (u; G, f 0 , b 0 , n) is valid for the PDE problem (A, G, f 0 , b 0 , n), then for all f and b, the iterator ?? H (u; G, f, b, n) is valid for the PDE problem (A, G, f, b, n).

The proposition states that we freely generalize to different f and b. There is no guarantee that we can generalize to different G and n. Generalization to different G and n has to be empirically verified: in our experiments, our learned iterator converges to the correct solution for a variety of grid sizes n and geometries G, even though it was only trained on one grid size and geometry.

Even when generalization fails, there is no risk of obtaining incorrect results.

The iterator will simply fail to converge.

This is because according to Lemma 1, fixed points of our new iterator is the same as the fixed point of hand designed iterator ??.

Therefore if our iterator is convergent, it is valid.

What is H trying to approximate?

In this section we show that we are training our linear function GH to approximate T (I ??? T ) ???1 : if it were able to approximate T (I ??? T ) ???1 perfectly, our iterator ?? H will converge to the correct solution in a single iteration.

Let the original update rule be ??(u) = T u + c, and the unknown ground truth solution be u * satisfying u * = T u * + c. Let r = u * ??? u be the current error, and e = u * ??? ??(u) be the new error after applying one step of ??. They are related by DISPLAYFORM0 In addition, let w = ??(u) ??? u be the update ?? makes.

This is related to the current error r by DISPLAYFORM1 From Eq. (10) we can observe that the linear operator GH takes as input ??'s update w, and tries to approximate the error e: GHw ???

e. If the approximation were perfect: GHw = e, the iterator ?? H would converge in a single iteration.

Therefore, we are trying to find some linear operator R, such that Rw = e. In fact, if we combine Eq. FORMULA4 and Eq. FORMULA4 , we can observe that T (I ??? T ) ???1 is (uniquely) the linear operator we are looking for DISPLAYFORM2 where (I ??? T ) ???1 exists because ??(T ) < 1, so all eigenvalues of I ??? T must be strictly positive.

Therefore, we would like our linear function GH to approximate T (I ??? T ) ???1 .Note that (I ??? T ) ???1 is a dense matrix in general, meaning that it is impossible to exactly achieve GH = T (I ???T ) ???1 with a convolutional operator H. However, the better GH is able to approximate T (I ??? T ) ???1 , the faster our iterator converges to the solution u * .

In our iterator design, H is a linear function parameterized by a linear deep network without nonlinearity or bias terms.

Even though our objective in Eq. FORMULA4 is a non-linear function of the parameters of the deep network, this is not an issue in practice.

In particular, BID2 observes that when modeling linear functions, deep networks can be faster to optimize with gradient descent compared to linear ones, despite non-convexity.

Conv model.

We model H as a network with 3 ?? 3 convolutional layers without non-linearity or bias.

We will refer to a model with k layers as "Convk", e.g. Conv3 has 3 convolutional layers.

U-Net model.

The Conv models suffer from the same problem as Jacobi: the receptive field grows only by 1 for each additional layer.

To resolve this problem, we design the deep network counterpart of the Multigrid method.

Instead of manually designing the sub-sampling / super-sampling functions, we use a U-Net architecture BID16 to learn them from data.

Because each layer reduces the grid size by half, and the i-th layer of the U-Net only operates on (2 ???i n)-sized grids, the total computation is only increased by a factor of 1 + 1/4 + 1/16 + ?? ?? ?? < 4/3 compared to a two-layer convolution.

The minimal overhead provides a very large improvement of convergence speed in our experiments.

We will refer to Multigrid and U-Net models with k sub-sampling layers as Multigridk and U-Netk, e.g. U-Net2 is a model with 2 sub-sampling layers.

We evaluate our method on the 2D Poisson equation with Dirichlet boundary conditions, ??? 2 u = f. There exist several iterative solvers for the Poisson equation, including Jacobi, Gauss-Seidel, conjugate-gradient, and multigrid methods.

We select the Jacobi method as our standard solver ??.To reemphasize, our goal is to train a model on simple domains where the ground truth solutions can be easily obtained, and then evaluate its performance on different geometries and boundary conditions.

Therefore, for training, we select the simplest Laplace equation, ??? 2 u = 0, on a square domain with boundary conditions such that each side is a random fixed value.

FIG4 shows an For testing, we use larger grid sizes than training.

For example, we test on 256??256 grid for a model trained on 64 ?? 64 grids.

Moreover, we designed challenging geometries to test the generalization of our models.

We test generalization on 4 different settings: (i) same geometry but larger grid, (ii) L-shape geometry, (iii) Cylinders geometry, and (iv) Poisson equation in same geometry, but f = 0.

The two geometries are designed because the models were trained on square domains and have never seen sharp or curved boundaries.

Examples of the 4 settings are shown in FIG4 .

As discussed in Section 2.4, the convergence rate of any linear iterator can be determined from the spectral radius ??(T ), which provides guarantees on convergence and convergence rate.

However, a fair comparison should also consider the computation cost of H. Thus, we evaluate the convergence rate by calculating the computation cost required for the error to drop below a certain threshold.

On GPU, the Jacobi iterator and our model can both be efficiently implemented as convolutional layers.

Thus, we measure the computation cost by the number of convolutional layers.

On CPU, each Jacobi iteration u i,j = 1 4 (u i???1,j + u i+1,j + u i,j???1 + u i,j+1 ) has 4 multiply-add operations, while a 3 ?? 3 convolutional kernel requires 9 operations, so we measure the computation cost by the number of multiply-add operations.

This metric is biased in favor of Jacobi because there is little practical reason to implement convolutions on CPU.

Nonetheless, we report both metrics in our experiments.

Table 1 shows results of the Conv model.

The model is trained on a 16 ?? 16 square domain, and tested on 64 ?? 64.

For all settings, our models converge to the correct solution, and require less computation than Jacobi.

The best model, Conv3, is ??? 5?? faster than Jacobi in terms of layers, and ??? 2.5?? faster in terms of multiply-add operations.

As discussed in Section 3.2, if our iterator converges for a geometry, then it is guaranteed to converge to the correct solution for any f and boundary values b. The experiment results show that our model not only converges but also converges faster than the standard solver, even though it is only trained on a smaller square domain.

For the U-Net models, we compare them against Multigrid models with the same number of subsampling and smoothing layers.

Therefore, our models have the same number of convolutional layers, and roughly 9/4 times the number of operations compared to Multigrid.

The model is trained on a 64 ?? 64 square domain, and tested on 256 ?? 256.The bottom part of Table 1 shows the results of the U-Net model.

Similar to the results of Conv models, our models outperforms Multigrid in all settings.

Note that U-Net2 has lower computation Table 1 : Comparisons between our models and the baseline solvers.

The Conv models are compared with Jacobi, and the U-Net models are compared with Multigrid.

The numbers are the ratio between the computation costs of our models and the baselines.

None of the values are greater than 1, which means that all of our models achieve a speed up on every problem and both performance metric (convolutional layers and multiply-add operations).

cost compared with Multigrid2 than U-Net3 compared to Multigrid 3.

This is because Multigrid2 is a relatively worse baseline.

U-Net3 still converges faster than U-Net2.

The FEniCS package BID13 provides a collection of tools with high-level Python and C++ interfaces to solve differential equations.

The open-source project is developed and maintained by a global community of scientists and software developers.

Its extensive optimization over the years, including the support for parallel computation, has led to its widespread adaption in industry and academia BID0 .We measure the wall clock time of the FEniCS model and our model, run on the same hardware.

The FEniCS model is set to be the minimal residual method with algebraic multigrid preconditioner, which we measure to be the fastest compared to other methods such as Jacobi or Incomplete LU factorization preconditioner.

We ignore the time it takes to set up geometry and boundary conditions, and only consider the time the solver takes to solve the problem.

We set the error threshold to be 1 percent of the initial error.

For the square domain, we use a quadrilateral mesh.

For the L-shape and cylinder domains, however, we let FEniCS generate the mesh automatically, while ensuring the number of mesh points to be similar.

FIG8 shows that our model is comparable or faster than FEniCS in wall clock time.

These experiments are all done on CPU.

Our model efficiently runs on GPU, while the fast but complex methods in FEniCS do not have efficient GPU implementations available.

On GPU, we measure an additional 30?? speedup (on Tesla K80 GPU, compared with a 64-core CPU).

Recently, there have been several works on applying deep learning to solve the Poisson equation.

However, to the best of our knowledge, previous works used deep networks to directly generate the solution; they have no correctness guarantees and are not generalizable to arbitrary grid sizes and boundary conditions.

Most related to our work are BID6 and BID17 , which learn deep networks to output the solution of the 2D Laplace equation (a special case where f = 0).

FIG4 ) trained a U-Net model that takes in the boundary condition as a 2D image and outputs the solution.

The model is trained by L1 loss to the ground truth solution and an adversarial discriminator loss.

BID17 ) also trained a U-net model but used a weakly-supervised loss.

There are other related works that solved the Poisson equation in concrete physical problems.

BID21 solved for electric potential in 2D/3D space; BID22 solved for pressure fields for fluid simulation; BID24 solved particle simulation of a PN Junction.

There are other works that solve other types of PDEs.

For example, many studies aimed to use deep learning to accelerate and approximate fluid dynamics, governed by the Euler equation or the Navier-Stokes equations BID8 BID23 BID4 BID10 .

BID5 use Bayesian optimization to design shapes with reduced drag coefficients in laminar fluid flow.

Other applications include solving the Schrodinger equation BID14 , turbulence modeling BID18 , and the American options and Black Scholes PDE BID19 .

A lot of these PDEs are nonlinear and may not have a standard linear iterative solver, which is a limitation to our current method since our model must be built on top of an existing linear solver to ensure correctness.

We consider the extension to different PDEs as future work.

We presented a method to learn an iterative solver for PDEs that improves on an existing standard solver.

The correct solution is theoretically guaranteed to be the fixed point of our iterator.

We show that our model, trained on simple domains, can generalize to different grid sizes, geometries and boundary conditions.

It converges correctly and achieves significant speedups compared to standard solvers, including highly optimized ones implemented in FEniCS.A PROOFS Theorem 1.

For a linear iterator ??(u) = T u + c, ?? converges to a unique stable fixed point from any initialization if and only if the spectral radius ??(T ) < 1.Proof.

Suppose ??(T ) < 1, then (I???T ) ???1 must exist because all eigenvalues of I???T must be strictly positive.

Let u * = (I ??? T ) ???1 c; this u * is a stationary point of the iterator ??, i.e. u * = T u * + c. For DISPLAYFORM0 Since ??(T ) < 1, we know BID11 , which means the error e k ??? 0.

Therefore, ?? converges to u * from any u 0 .

DISPLAYFORM1 Now suppose ??(T ) ??? 1.

Let ?? 1 be the largest absolute eigenvalue where ??(T ) = |?? 1 | ??? 1, and v 1 be its corresponding eigenvector.

We select initialization u 0 = u * + v 1 , then e 0 = v 1 .

Because |?? 1 | ??? 1, we have |?? k 1 | ??? 1, then T k e 0 = ?? k 1 v 1 ??? k?????? 0 However we know that under a different initialization?? 0 = u * , we have?? 0 = 0, so T k??0 = 0.

Therefore the iteration cannot converge to the same fixed point from different initializations u 0 and u 0 .Proposition 1 If M is a full rank diagonal matrix, and u * ??? R n 2 ??n 2 satisfies Eq. (7) , then u * satisfies Eq. (4).Proof of Proposition 1.

Let u * be a fixed point of Eq. (7) then The latter equation is equivalent to GM ???1 (Au * ??? f ) = 0.

If M is a full rank diagonal matrix, this implies G(Au * ??? f ) = 0, which is GAu * = Gf .

Therefore, u * satisfies Eq.(4).Theorem 2.

For fixed G, f, b, n, the spectral norm of ?? H (u; G, f, b, n) is a convex function of H, and the set of H such that the spectral norm of ?? H (u; G, f, b, n) < 1 is a convex open set.

Proof.

As before, denote ??(u) = T u + c. Observe that ?? H (u; G, f, b, n) = T u + c + GH(T u + c ??? u) = (T + GHT ??? GH)u + GHc + cThe spectral norm ?? 2 is convex with respect to its argument, and (T + GHT ??? GH) is linear in H. Thus, T + GHT ??? GH 2 is convex in H as well.

Thus, under the condition that T + GHT ??? GH 2 < 1, the set of H must be convex because it is a sub-level set of the convex function T + GHT ??? GH 2 .To prove that it is open, observe that ?? 2 is a continuous function, so T + GHT ??? GH 2 is a continuous map from H to the spectral radius of ?? H .

If we consider the set of H such that T + GHT ??? GH 2 < 1, this set is the preimage of (??? , 1) for any > 0.

As (??? , 1) is open, its preimage must be open.

Proposition 2.

For fixed A, G, n and fixed H, if for some f 0 , b 0 , ?? H (u; G, f 0 , b 0 , n) is valid for the PDE problem (A, G, f 0 , b 0 , n), then for all f and b, the iterator ?? H (u; G, f, b, n) is valid for the PDE problem (A, G, f, b, n).Proof.

From Theorem 1 and Lemma 1, our iterator is valid if and only if ??(T + GHT ??? GH) < 1.

The iterator T + GHT ??? GH only depends on A, G, and is independent of the constant c in Eq. (18).

Thus, the validity of the iterator is independent with f and b. Thus, if the iterator is valid for some f 0 and b 0 , then it is valid for any choice of f and b.

In Section 2.4.1, we show that for Poisson equation, the update matrix T = G(I ??? A).

We now formally prove that ??(G(I ??? A)) < 1 for any G.For any matrix T , the spectral radius is bounded by the spectral norm: ??(T ) ??? T 2 , and the equality holds if T is symmetric.

Since (I ??? A) is a symmetric matrix, ??(I ??? A) = I ??? A 2 .

It has been proven that ??(I ??? A) < 1 FIG4 .

Moreover, G 2 = 1.

Finally, matrix norms are sub-multiplicative, so DISPLAYFORM0 ??(T ) < 1 is true for any G. Thus, the standard Jacobi method is valid for the Poisson equation under any geometry.

<|TLDR|>

@highlight

We learn a fast neural solver for PDEs that has convergence guarantees.

@highlight

Develops a method to accelerate the finite difference method in solving PDEs and proposes a revised framework for fixed point iteration after discretization.

@highlight

The authors propose a linear method for speeding up PDE solvers.