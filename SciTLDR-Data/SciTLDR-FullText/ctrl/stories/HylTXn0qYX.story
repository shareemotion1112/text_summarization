We provide a theoretical algorithm for checking local optimality and escaping saddles at nondifferentiable points of empirical risks of two-layer ReLU networks.

Our algorithm receives any parameter value and returns: local minimum, second-order stationary point, or a strict descent direction.

The presence of M data points on the nondifferentiability of the ReLU divides the parameter space into at most 2^M regions, which makes analysis difficult.

By exploiting polyhedral geometry, we reduce the total computation down to one convex quadratic program (QP) for each hidden node, O(M) (in)equality tests, and one (or a few) nonconvex QP.

For the last QP, we show that our specific problem can be solved efficiently, in spite of nonconvexity.

In the benign case, we solve one equality constrained QP, and we prove that projected gradient descent solves it exponentially fast.

In the bad case, we have to solve a few more inequality constrained QPs, but we prove that the time complexity is exponential only in the number of inequality constraints.

Our experiments show that either benign case or bad case with very few inequality constraints occurs, implying that our algorithm is efficient in most cases.

Empirical success of deep neural networks has sparked great interest in the theory of deep models.

From an optimization viewpoint, the biggest mystery is that deep neural networks are successfully trained by gradient-based algorithms despite their nonconvexity.

On the other hand, it has been known that training neural networks to global optimality is NP-hard BID2 .

It is also known that even checking local optimality of nonconvex problems can be NP-hard (Murty & Kabadi, 1987) .

Bridging this gap between theory and practice is a very active area of research, and there have been many attempts to understand why optimization works well for neural networks, by studying the loss surface BID1 Yu & Chen, 1995; Kawaguchi, 2016; Soudry & Carmon, 2016; Nguyen & Hein, 2017; Safran & Shamir, 2018; Laurent & Brecht, 2018; Yun et al., 2019; Zhou & Liang, 2018; Wu et al., 2018; Shamir, 2018) and the role of (stochastic) gradientbased methods (Tian, 2017; BID4 Zhong et al., 2017; Soltanolkotabi, 2017; Li & Yuan, 2017; Zhang et al., 2018; BID5 Wang et al., 2018; Li & Liang, 2018; BID9 BID11 BID7 BID0 Zou et al., 2018; Zhou et al., 2019) .One of the most important beneficial features of convex optimization is the existence of an optimality test (e.g., norm of the gradient is smaller than a certain threshold) for termination, which gives us a certificate of (approximate) optimality.

In contrast, many practitioners in deep learning rely on running first-order methods for a fixed number of epochs, without good termination criteria for the optimization problem.

This means that the solutions that we obtain at the end of training are not necessarily global or even local minima.

Yun et al. (2018; 2019) showed efficient and simple global optimality tests for deep linear neural networks, but such optimality tests cannot be extended to general nonlinear neural networks, mainly due to nonlinearity in activation functions.

Besides nonlinearity, in case of ReLU networks significant additional challenges in the analysis arise due to nondifferentiability, and obtaining a precise understanding of the nondifferentiable points is still elusive.

ReLU activation function h(t) = max{t, 0} is nondifferentiable at t = 0.

This means that, for example, the function f (w, b) := (h(w T x + b) ??? 1) 2 is nondifferentiable for any (w, b) satisfying w T x+b = 0.

See FIG2 for an illustration of how the empirical risk of a ReLU network looks like.

Although the plotted function does not exactly match the definition of empirical risk we study in this paper, the figures help us understand that the empirical risk is continuous but piecewise differentiable, with affine hyperplanes on which the function is nondifferentiable.

Such nondifferentiable points lie in a set of measure zero, so one may be tempted to overlook them as "non-generic."

However, when studying critical points we cannot do so, as they are precisely such "non-generic" points.

For example, Laurent & Brecht (2018) study one-hidden-layer ReLU networks with hinge loss and note that except for piecewise constant regions, local minima always occur on nonsmooth boundaries.

Probably due to difficulty in analysis, there have not been other works that handle such nonsmooth points of losses and prove results that work for all points.

Some theorems (Soudry & Carmon, 2016; Nguyen & Hein, 2018) hold "almost surely"; some assume differentiability or make statements only for differentiable points (Nguyen & Hein, 2017; Yun et al., 2019) ; others analyze population risk, in which case the nondifferentiability disappears after taking expectation (Tian, 2017; BID4 BID10 Safran & Shamir, 2018; Wu et al., 2018 ).

In this paper, we take a step towards understanding nondifferentiable points of the empirical risk of one-hidden-layer ReLU(-like) networks.

Specifically, we provide a theoretical algorithm that tests second-order stationarity for any point of the loss surface.

It takes an input point and returns:(a) The point is a local minimum; or (b) The point is a second-order stationary point (SOSP); or (c) A descent direction in which the function value strictly decreases.

Therefore, we can test whether a given point is a SOSP.

If not, the test extracts a guaranteed direction of descent that helps continue minimization.

With a proper numerical implementation of our algorithm (although we leave it for future work), one can run a first-order method until it gets stuck near a point, and run our algorithm to test for optimality/second-order stationarity.

If the point is an SOSP, we can terminate without further computation over many epochs; if the point has a descent direction, our algorithm will return a descent direction and we can continue on optimizing.

Note that the descent direction may come from the second-order information; our algorithm even allows us to escape nonsmooth second-order saddle points.

This idea of mixing first and second-order methods has been explored in differentiable problems (see, for example, BID7 Reddi et al. (2017) and references therein), but not for nondifferentiable ReLU networks.

The key computational challenge in constructing our algorithm for nondifferentiable points is posed by data points that causes input 0 to the ReLU hidden node(s).

Such data point bisects the parameter space into two halfspaces with different "slopes" of the loss surface, so one runs into nondifferen-tiability.

We define these data points to be boundary data points.

For example, in FIG2 , if the input to our algorithm is (w, v) = (???2/3, 1/3), then there are two boundary data points: "blue" and "red."

If there are M such boundary data points, then in the worst case the parameter space divides into 2 M regions, or equivalently, there are 2 M "pieces" of the function that surround the input point.

Of course, naively testing each region will be very inefficient; in our algorithm, we overcome this issue by a clever use of polyhedral geometry.

Another challenge comes from the second-order test, which involves solving nonconvex QPs.

Although QP is NP-hard in general (Pardalos & Vavasis, 1991) , we prove that the QPs in our algorithm are still solved efficiently in most cases.

We further describe the challenges and key ideas in Section 2.1.

We consider a one-hidden-layer neural network with input dimension d x , hidden layer width d h , and output dimension d y .

We are given m pairs of data points and labels DISPLAYFORM0 , where x i ??? R dx and y i ??? R dy .

Given an input vector x, the output of the network is defined as DISPLAYFORM1 , and b 1 ??? R d h are the network parameters.

The activation function h is "ReLU-like," meaning h(t) := max{s + t, 0} + min{s ??? t, 0}, where s + > 0, s ??? ??? 0 and s + = s ??? .

Note that ReLU and Leaky-ReLU are members of this class.

In training neural networks, we are interested in minimizing the empirical risk DISPLAYFORM2 , where (w, y) : R dy ?? R dy ??? R is the loss function.

We make the following assumptions on the loss function and the training dataset: Assumption 1.

The loss function (w, y) is twice differentiable and convex in w. Assumption 2.

No d x + 1 data points lie on the same affine hyperplane.

Assumption 1 is satisfied by many standard loss functions such as squared error loss and crossentropy loss.

Assumption 2 means, if d x = 2 for example, no three data points are on the same line.

Since real-world datasets contain noise, this assumption is also quite mild.

In this section, we explain the difficulties at nondifferentiable points and ideas on overcoming them.

Our algorithm is built from first principles, rather than advanced tools from nonsmooth analysis.

Bisection by boundary data points.

Since the activation function h is nondifferentiable at 0, the behavior of data points at the "boundary" is decisive.

Consider a simple example d h = 1, so W 1 is a row vector.

If W 1 x i +b 1 = 0, then the sign of (W 1 +??? 1 )x i +(b 1 +?? 1 ) for any small perturbations ??? 1 and ?? 1 stays invariant.

In contrast, when there is a point x i on the "boundary," i.e., W 1 x i + b 1 = 0, then the slope depends on the direction of perturbation, leading to nondifferentiability.

As mentioned earlier, we refer to such data points as boundary data points.

DISPLAYFORM0 and similarly, the slope is s ??? for ??? 1 x i + ?? 1 ??? 0.

This means that the "gradient" (as well as higher order derivatives) of R depends on direction of (??? 1 , ?? 1 ).Thus, every boundary data point x i bisects the space of perturbations (??? j , ?? j ) 2 j=1 into two halfspaces by introducing a hyperplane through the origin.

The situation is even worse if we have M boundary data points: they lead to a worst case of 2 M regions.

Does it mean that we need to test all 2 M regions separately?

We show that there is a way to get around this issue, but before that, we first describe how to test local minimality or stationarity for each region.

Second-order local optimality conditions.

We can expand R((W j + ??? j , b j + ?? j ) 2 j=1 ) and obtain the following Taylor-like expansion for small enough perturbations (see Lemma 2 for details) DISPLAYFORM1 j=1 .

Notice now that in (1), at nondifferentiable points the usual Taylor expansion does not exist, but the corresponding "gradient" g(??) and "Hessian" H(??) now depend on the direction of perturbation ??.

Also, the space of ?? is divided into at most 2 M regions, and g(z, ??) and H(z, ??) are piecewise-constant functions of ?? whose "pieces" correspond to the regions.

One could view this problem as 2 M constrained optimization problems and try to solve for KKT conditions at z; however, we provide an approach that is developed from first principles and solves all 2 M problems efficiently.

Given this expansion (1) and the observation that derivatives stay invariant with respect to scaling of ??, one can note that (a) g(z, ??) T ?? ??? 0 for all ??, and (b) ?? T H(z, ??)?? ??? 0 for all ?? such that g(z, ??)T ?? = 0 are necessary conditions for local optimality of z, thus z is a "SOSP" (see Definition 2.2).

The conditions become sufficient if (b) is replaced with ?? T H(z, ??)?? > 0 for all ?? = 0 such that g(z, ??)T ?? = 0.

In fact, this is a generalized version of second-order necessary (or sufficient) conditions, i.e., ???f = 0 and ??? 2 f 0 (or ??? 2 f 0), for twice differentiable f .Efficiently testing SOSP for exponentially many regions.

Motivated from the second-order expansion (1) and necessary/sufficient conditions, our algorithm consists of three steps:(a) Testing first-order stationarity (in the Clarke sense, see Definition 2.1), DISPLAYFORM2 The tests are executed from Step (a) to (c).

Whenever a test fails, we get a strict descent direction ??, and the algorithm returns ?? and terminates.

Below, we briefly outline each step and discuss how we can efficiently perform the tests.

We first check first-order stationarity because it makes Step (b) easier.

Step (a) is done by solving one convex QP per each hidden node.

ForStep (b), we formulate linear programs (LPs) per each 2 M region, so that checking whether all LPs have minimum cost of zero is equivalent to checking g(z, ??)T ?? ??? 0 for all ??.

Here, the feasible sets of LPs are pointed polyhedral cones, whereby it suffices to check only the extreme rays of the cones.

It turns out that there are only 2M extreme rays, each shared by 2 M ???1 cones, so testing g(z, ??) T ?? ??? 0 can be done with only O(M ) inequality/equality tests instead of solving exponentially many LPs.

In Step (b), we also record the flat extreme rays, which are defined to be the extreme rays with g(z, ??)T ?? = 0, for later use in Step (c).

Step (c), we test if the second-order perturbation ?? T H(??)?? can be negative, for directions where g(z, ??)T ?? = 0.

Due to the constraint g(z, ??) T ?? = 0, the second-order test requires solving constrained nonconvex QPs.

In case where there is no flat extreme ray, we need to solve only one equality constrained QP (ECQP).

If there exist flat extreme rays, a few more inequality constrained QPs (ICQPs) are solved.

Despite NP-hardness of general QPs (Pardalos & Vavasis, 1991) , we prove that the specific form of QPs in our algorithm are still tractable in most cases.

More specifically, we prove that projected gradient descent on ECQPs converges/diverges exponentially fast, and each step takes O(p 2 ) time (p is the number of parameters).

In case of ICQPs, it takes O(p 3 + L 3 2 L ) time to solve the QP, where L is the number of boundary data points that have flat extreme rays (L ??? M ).

Here, we can see that if L is small enough, the ICQP can still be solved in polynomial time in p. At the end of the paper, we provide empirical evidences that the number of flat extreme rays is zero or very few, meaning that in most cases we can solve the QP efficiently.

In this section, we define a more precise notion of generalized stationary points and introduce some additional symbols that will be helpful in streamlining the description of our algorithm in Section 3.

Since we are dealing with nondifferentiable points of nonconvex R, usual notions of (sub)gradients do not work anymore.

Here, Clarke subdifferential is a useful generalization BID8 : Definition 2.1 (FOSP, Theorem 6.2.5 of BID3 ).

Suppose that a function f (z) : ??? ??? R is locally Lipschitz around the point z * ??? ???, and differentiable in ??? \ W where W has Lebesgue measure zero.

Then the Clarke differential of f at z * is DISPLAYFORM0 DISPLAYFORM1 Given an input data point x ??? R dx , we define O(x) := h(W 1 x + b 1 ) to be the output of hidden layer.

We note that the notation O(??) is overloaded with the big O notation, but their meaning will be clear from the context.

Consider perturbing parameters DISPLAYFORM2 , then the perturbed output??? (x) of the network and the amount of perturbation dY (x) can be expressed as DISPLAYFORM3 where J(x) can be thought informally as the "Jacobian" matrix of the hidden layer.

The matrix J(x) ??? R d h ??d h is diagonal, and its k-th diagonal entry is given by DISPLAYFORM4 where h is the derivative of h. We define h (0) := s + , which is okay because it is always multiplied with zero in our algorithm.

For boundary data points, DISPLAYFORM5 , as noted in Section 2.1.

We additionally define dY 1 (x) and dY 2 (x) to separate the terms in dY (x) that are linear in perturbations versus quadratic in perturbations.

DISPLAYFORM6 For simplicity of notation for the rest of the paper, we define for all i ??? [m] := {1, . . .

, m}, DISPLAYFORM7 In our algorithm and its analysis, we need to give a special treatment to the boundary data points.

To this end, for each node k ??? [d h ] in the hidden layer, define boundary index set B k as DISPLAYFORM8 The subspace spanned by vectorsx i for in i ??? B k plays an important role in our tests; so let us define a symbol for it, as well as the cardinality of B k and their sum: DISPLAYFORM9 , and u k ??? R dy be the k-th column of ??? 2 .

Next, we define the total number of parameters p, and vectorized perturbations ?? ??? R p : DISPLAYFORM10 , packed in the same order as ??.

Define a matrix DISPLAYFORM11 .

This quantity appears multiplie times and does not depend on the perturbation, so it is helpful to have a symbol for it.

We conclude this section by presenting one of the implications of Assumption 2 in the following lemma, which we will use later.

The proof is simple, and is presented in Appendix B.1.

Lemma 1.

If Assumption 2 holds, then M k ??? d x and the vectors {x i } i???B k are linearly independent.

In this section, we present SOSP-CHECK in Algorithm 1, which takes an arbitrary tuple DISPLAYFORM0 of parameters as input and checks whether it is a SOSP.

We first present a lemma that shows the explicit form of the perturbed empirical risk R(z +??) and identify first and second-order perturbations.

The proof is deferred to Appendix B.2.

end if 10: end for 11: For all ??'s s.t.

g(z, ??) DISPLAYFORM0 return SOSP.

14: else 15:return Local Minimum.

16: end if Lemma 2.

For small enough perturbation ??, DISPLAYFORM1 where g(z, ??) and H(z, ??) satisfy DISPLAYFORM2 .

Also, g(z, ??) and H(z, ??) are piecewise constant functions of ??, which are constant inside each polyhedral cone in space of ??.

Rough pseudocode of SOSP-CHECK is presented in Algorithm 1.

As described in Section 2.1, the algorithm consists of three steps: (a) testing first-order stationarity (b) testing g(z, ??)T ?? ??? 0 for all ??, and (c) testing DISPLAYFORM3 .

If the input point satisfies the secondorder sufficient conditions for local minimality, the algorithm decides it is a local minimum.

If the point only satisfies second-order necessary conditions, it returns SOSP.

If a strict descent direction ?? is found, the algorithm terminates immediately and returns ??.

A brief description will follow, but the full algorithm (Algorithm 2) and a full proof of correctness are deferred to Appendix A.

Line 1 of Algorithm 1 corresponds to testing if ??? W2 R and ??? b2 R are singletons with zero.

If not, the opposite direction is a descent direction.

More details are in Appendix A.1.1.Test for W 1 and b 1 is more difficult because g(z, ??) depends on ??? 1 and ?? 1 when there are boundary data points.

DISPLAYFORM0 ,?? R can be tested by solving a convex QP: DISPLAYFORM1 If the solution {s * i } i???B k does not achieve zero objective value, then we can directly return a descent direction.

For details please refer to FO-SUBDIFF-ZERO-TEST (Algorithm 3) and Appendix A.1.2.

DISPLAYFORM2 Linear program formulation.

Lines 5-6 are about testing if DISPLAYFORM3 Published as a conference paper at ICLR 2019 DISPLAYFORM4 Note that by Lemma 1,x i 's for i ??? B k are linearly independent.

So, given M k boundary data points, they divide the space DISPLAYFORM5 T is constant in each polyhedral cones, we can let ?? i ??? {???1, +1} for all i ??? B k , and define an LP for each {?? i } i???B k ??? {???1, +1} M k : DISPLAYFORM6 Solving these LPs and checking if the minimum value is 0 suffices to prove DISPLAYFORM7 It is equivalent to d x + 1 ??? M k linearly independent equality constraints.

So, the feasible set of LP (3) has d x + 1 linearly independent constraints, which implies that the feasible set is a pointed polyhedral cone with vertex at origin.

Since any point in a pointed polyhedral cone is a conical combination (linear combination with nonnegative coefficients) of extreme rays of the cone, checking nonnegativity of the objective function for all extreme rays suffices.

We emphasize that we do not solve the LPs (3) in our algorithm; we just check the extreme rays.

Computational efficiency.

Extreme rays of a pointed polyhedral cone in R dx+1 are computed from d x linearly independent active constraints.

For each i ??? B k , the extreme rayv DISPLAYFORM8 Note that there are 2M k extreme rays, and one extreme rayv i,k is shared by 2 DISPLAYFORM9 regardless of {?? j } j???B k \{i} .

Testing an extreme ray can be done with a single inequality test instead of 2 M k ???1 separate tests for all cones!

Thus, this extreme ray approach instead of solving individual LPs greatly reduces computation, from O(2 DISPLAYFORM10 Testing extreme rays.

For the details of testing all possible extreme rays, please refer to FO-INCREASING-TEST (Algorithm 4) and Appendix A.2.

FO-INCREASING-TEST computes all possible extreme rays??? k and tests if they satisfy DISPLAYFORM11 If the inequality is not satisfied by an extreme ray??? k , then this is a descent direction, so we return??? k .

If the inequality holds with equality, it means this is a flat extreme ray, and it needs to be checked in second-order test, so we save this extreme ray for future use.

How many flat extreme rays (g k (z,??? k ) T??? k = 0) are there?

Presence of flat extreme rays introduce inequality constraints in the QP that we solve in the second-order test.

It is ideal not to have them, because in this case there are only equality constraints, so the QP is easier to solve.

Lemma A.1 in Appendix A.2 shows the conditions for having flat extreme rays; in short, there is a flat extreme ray DISPLAYFORM12 The second-order test checks ?? T H(z, ??)?? ??? 0 for "flat" ??'s satisfying g(z, ??) T ?? = 0.

This is done with help of the function SO-TEST (Algorithm 5).

Given its input {?? i,k } k???[d h ],i???B k , it defines fixed "Jacobian" matrices J i for all data points and equality/inequality constraints for boundary data points, and solves the QP of the following form: DISPLAYFORM13 Constraints and number of QPs.

There are d h equality constraints of the form DISPLAYFORM14 These equality constraints are due to the nonnegative homogeneous property of activation h; i.e., scaling [W 1 ] k,?? and [b 1 ] k by ?? > 0 and scaling [W 2 ] ??,k by 1/?? yields exactly the same network.

So, these equality constraints force ?? to be orthogonal to the loss-invariant directions.

This observation is stated more formally in Lemma A.2, which as a corollary shows that any differentiable FOSP of R always has rank-deficient Hessian.

The other constraints make sure that the union of feasible sets of QPs is exactly {?? | g(z, ??)T ?? = 0} (please see Lemma A.3 in Appendix A.3 for details).

It is also easy to check that these constraints are all linearly independent.

If there is no flat extreme ray, the algorithm solves just one QP with d h + M equality constraints.

If there are flat extreme rays, the algorithm solves one QP with d h + M equality constraints, and 2 K more QPs with d h + M ??? L equality constraints and L inequality constraints, where DISPLAYFORM15 Recall from Section 3.2 that i ??? B k has a flat extreme ray if DISPLAYFORM16 Please refer to Appendix A.3 for more details.

Efficiency of solving the QPs (4).

Despite NP-hardness of general QPs, our specific form of QPs (4) can be solved quite efficiently, avoiding exponential complexity in p. After solving QP (4), there are three (disjoint) termination conditions: DISPLAYFORM17 where S is the feasible set of QP.

With the following two lemmas, we show that the termination conditions can be efficiently tested for ECQPs and ICQPs.

First, the ECQPs can be iteratively solved with projected gradient descent, as stated in the next lemma.

Lemma 3.

Consider the QP, where Q ??? R p??p is symmetric and A ??? R q??p has full row rank: DISPLAYFORM18 Then, projected gradient descent (PGD) updates DISPLAYFORM19 with learning rate ?? < 1/?? max (Q) converges to a solution or diverges to infinity exponentially fast.

Moreover, with random initialization, PGD correctly checks conditions (T1)-(T3) with probability 1.The proof is an extension of unconstrained case (Lee et al., 2016) , and is deferred to Appendix B.3.

Note that it takes O(p 2 q) time to compute (I ??? A T (AA T ) ???1 A)(I ??? ??Q) in the beginning, and each update takes O(p 2 ) time.

It is also surprising that the convergence rate does not depend on q.

In the presence of flat extreme rays, we have to solve QPs involving L inequality constraints.

We prove that our ICQP can be solved in O(p 3 + L 3 2 L ) time, which implies that as long as the number of flat extreme rays is small, the problem can still be solved in polynomial time in p. Lemma 4.

Consider the QP, where Q ??? R p??p is symmetric, A ??? R q??p and B ??? R r??

p have full row rank, and A T BT has rank q + r: DISPLAYFORM20 Then, there exists a method that checks whether (T1)-(T3) in O(p 3 + r 3 2 r ) time.

In short, we transform ?? to define an equivalent problem, and use classical results in copositive matrices (Martin & Jacobson, 1981; Seeger, 1999; Hiriart-Urruty & Seeger, 2010) ; the problem can be solved by computing the eigensystem of a (p???q ???r)??(p???q ???r) matrix, and testing copositivity of an r ?? r matrix.

The proof is presented in Appendix B.4.

During all calls to SO-TEST, whenever any QP terminated with (T3), then SOSP-CHECK immediately returns the direction and terminates.

After solving all QPs, if any of SO-TEST calls finished with (T2), then we conclude SOSP-CHECK with "SOSP."

If all QPs terminated with (T1), then we can return "Local Minimum."

For experiments, we used artificial datasets sampled iid from standard normal distribution, and trained 1-hidden-layer ReLU networks with squared error loss.

In practice, it is impossible to get to the exact nondifferentiable point, because they lie in a set of measure zero.

To get close to those points, we ran Adam (Kingma & Ba, 2014) using full-batch (exact) gradient for 200,000 iterations and decaying step size (start with 10 ???3 , 0.2?? decay every 20,000 iterations).

We observed that decaying step size had the effect of "descending deeper into the valley."

DISPLAYFORM0 DISPLAYFORM1 we counted the number of approximate boundary data points satisfying |[W 1 x???5 , which gives an estimate of M k .

Moreover, for these points, we solved the QP (2) using L-BFGS-B BID6 , to check if the terminated points are indeed (approximate) FOSPs.

We could see that the optimal values of (2) are close to zero (??? 10 ???6 typically, ??? 10 ???3 for largest problems).

After solving FORMULA25 , we counted the number of s * i 's that ended up with 0 or 1.

The number of such s * i 's is an estimate of L ??? K. We also counted the number of approximate boundary data points satisfying DISPLAYFORM2 , for an estimate of K.We ran the above-mentioned experiments for different settings of (d x , d h , m), 40 times each.

We fixed d y = 1 for simplicity.

For large d h , the optimizer converged to near-zero minima, making ??? i uniformly small, so it was difficult to obtain accurate estimates of K and L. Thus, we had to perform experiments in settings where the optimizer converged to minima that are far from zero.

TAB2 summarizes the results.

Through 280 runs, we observed that there are surprisingly many boundary data points (M ) in general, but usually there are zero or very few (maximum was 3) flat extreme rays (L).

This observation suggests two important messages: (1) many local minima are on nondifferentiable points, which is the reason why our analysis is meaningful; (2) luckily, L is usually very small, so we only need to solve ECQPs (L = 0) or ICQPs with very small number of inequality constraints, which are solved efficiently (Lemmas 3 and 4).

We can observe that M , L, and K indeed increase as model dimensions and training set get larger, but the rate of increase is not as fast as d x , d h , and m.

We provided a theoretical algorithm that tests second-order stationarity and escapes saddle points, for any points (including nondifferentiable ones) of empirical risk of shallow ReLU-like networks.

Despite difficulty raised by boundary data points dividing the parameter space into 2 M regions, we reduced the computation to d h convex QPs, O(M ) equality/inequality tests, and one (or a few more) nonconvex QP.

In benign cases, the last QP is equality constrained, which can be efficiently solved with projected gradient descent.

In worse cases, the QP has a few (say L) inequality constraints, but it can be solved efficiently when L is small.

We also provided empirical evidences that L is usually either zero or very small, suggesting that the test can be done efficiently in most cases.

A limitation of this work is that in practice, exact nondifferentiable points are impossible to reach, so the algorithm must be extended to apply the nonsmooth analysis for points that are "close" to nondifferentiable ones.

Also, current algorithm only tests for exact SOSP, while it is desirable to check approximate second-order stationarity.

These extensions must be done in order to implement a robust numerial version of the algorithm, but they require significant amount of additional work; thus, we leave practical/robust implementation as future work.

Also, extending the test to deeper neural networks is an interesting future direction.

Algorithm 2 SOSP-CHECK DISPLAYFORM0

if decr = True then 13: DISPLAYFORM0 DISPLAYFORM1 2: return {s * i } i???B k .

In this section, we present the detailed operation of SOSP-CHECK (Algorithm 2), and its helper functions FO-SUBDIFF-ZERO-TEST, FO-INCREASING-TEST, and SO-TEST (Algorithm 3-5).In the subsequent subsections, we provide a more detailed proof of the correctness of Algorithm 2.Recall that, by Lemmas 1 and 2, M k := |B k | ??? d x and vectors {x i } i???B k are linearly independent.

Also, we can expand R(z + ??) so that DISPLAYFORM0 return (True,??? k , {???} i???B k )8: .

DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 A.1 TESTING FIRST-ORDER STATIONARITY (LINES 1-3, 6-10 AND 15-17)A.1.1 TEST OF FIRST-ORDER STATIONARITY FOR W 2 AND b 2 (LINES 1-3)Lines 1-3 of Algorithm 2 correspond to testing if ??? W2 R = {0 dy??d h } and ??? b2 R = {0 dy }.

If they are not all zero, the opposite direction is a descent direction, as Line 2 returns.

To see why, suppose DISPLAYFORM5 If we apply perturbation (????? j , ???? j ) 2 j=1 where ?? > 0, we can immediately check that dY 1 ( DISPLAYFORM6 and also that DISPLAYFORM7 .

Then, by scaling ?? sufficiently small we can achieve R(z + ??) < R(z), which disproves that (W j , b j ) 2 j=1 is a local minimum.

A.1.2 TEST OF FIRST-ORDER STATIONARITY FOR W 1 AND b 1 (LINES 6-10 AND 15-17)Test for W 1 and b 1 is more difficult because g(z, ??) depends on ??? 1 and ?? 1 when there are boundary data points.

Recall that v DISPLAYFORM8 .

Thus we can separate k's and treat them individually.

, there is no boundary data point for k-th hidden node, so the Clarke subdifferential with respect to DISPLAYFORM0 Lines 15-17 handle this case; if the singleton element in the subdifferential is not zero, its opposite direction is a descent direction, so return that direction, as in Line 16.Test for zero in subdifferential.

For the case M k > 0, we saw that for boundary data points DISPLAYFORM1 Since the subdifferential is used many times, we give it a specific name DISPLAYFORM2 It solves a convex QP (2), and returns {s * DISPLAYFORM3 and apply perturbation (????? j , ???? j ) 2 j=1 where ?? > 0.

With this perturbation, we can check that DISPLAYFORM4 T ?? is strictly negative with magnitude O(??).

It is easy to see that ?? T H(z, ??)?? = O(?? 2 ), so by scaling ?? sufficiently small we can disprove local minimality of (W j , b j ) DISPLAYFORM5 Linear program formulation.

Lines 11-14 are essentially about testing if DISPLAYFORM6 T is constant in each polyhedral cone, we can let ?? i ??? {???1, +1} for all i ??? B k , and define an LP for each {?? i } i???B k ??? {???1, +1} M k : DISPLAYFORM7 Solving these LPs and checking if the minimum value is 0 suffices to prove DISPLAYFORM8 Note that any component of v k that is orthogonal to V k is also orthogonal to g k (z, v k ), so it does not affect the objective function of any LP (3).

Thus, the constraint v k ??? V k is added to the LP (3), which is equivalent to adding d x +1???M k linearly independent equality constraints.

The feasible set of LP (3) has d x + 1 linearly independent equality/inequality constraints, which implies that the feasible set is a pointed polyhedral cone with vertex at origin.

Since any point in a pointed polyhedral cone is a conical combination (linear combination with nonnegative coefficients) of extreme rays of the cone, checking nonnegativity of the objective function for all extreme rays suffices.

We emphasize that we do not solve the LPs (3) in our algorithm; we just check the extreme rays.

Computational efficiency.

Extreme rays of a pointed polyhedral cone in R dx+1 are computed from d x linearly independent active constraints.

Line 3 of Algorithm 4 is exactly computing such extreme rays: DISPLAYFORM9 ??? for each i ??? B k , tested in both directions.

Note that there are 2M k extreme rays, and one extreme rayv i,k is shared by 2 M k ???1 polyhedral cones.

Moreover,x T jv i,k = 0 for j ??? B k \ {i}, which indicates that For both direction of extreme rays??? k =v i,k and??? k = ???v i,k (Line 4), we check if g k (z,??? k ) T??? k ??? 0.

Whenever it does not hold (Lines 6-7),??? k is a descent direction, so FO-INCREASING-TEST returns it with True.

Line 13 of Algorithm 2 uses that??? k to return perturbations, so that scaling by small enough ?? > 0 will give us a point with R(z + ????) < R(z).

If equality holds (Lines 8-9), this means v k is a direction of perturbation satisfying g(z, ??) DISPLAYFORM10 T ?? = 0, so this direction needs to be checked if ?? T H(z, ??)?? ??? 0 too.

In this case, we add the sign of boundary data pointx i to S i,k for future use in the second-order test.

The operation with S i,k will be explained in detail in Appendix A.3.

After checking if g k (z,??? k ) Proof First note that we already assumed that all extreme rays??? k satisfy g k (z,??? k ) T??? k ??? 0, so SOSP-CHECK will reach Line 14 at the end.

Also note thatx i 's in i ??? B k are linearly independent (by Lemma 1), sox DISPLAYFORM11 DISPLAYFORM12 Thus,??? k is a flat extreme ray.

The case with s * .

Also, it follows from the definition of K and L (5) that DISPLAYFORM13 DISPLAYFORM14 Connection to KKT conditions.

As a side remark, we provide connections of our tests to the wellknown KKT conditions.

Note that the equality As mentioned in Section 2.1, given that g(z, ??) and H(z, ??) are constant functions of ?? in each polyhedral cone, one can define inequality constrained optimization problems and try to solve for KKT conditions for z directly.

However, this also requires solving 2 M problems.

The strength of our approach is that by solving the QPs (2), we can automatically compute the exact Lagrange multipliers for all 2 M subproblems, and dual feasibility is also tested in O(M ) time.

DISPLAYFORM15 DISPLAYFORM16 The second-order test checks ?? T H(z, ??)?? ??? 0 for "flat" ??'s satisfying g(z, ??) T ?? = 0.

This is done with help of the function SO-TEST in Algorithm 5.

Given its input {?? i,k } k???[d h ],i???B k , it defines fixed "Jacobian" matrices J i for all data points and equality/inequality constraints for boundary data points, and solves the QP (4).

Equality/inequality constraints.

In the QP (4), there are d h equality constraints of the form DISPLAYFORM17 DISPLAYFORM18 The proof of Lemma A.2 can be found in Appendix B.5.

A corollary of this lemma is that any differentiable FOSP of R always has rank-deficient Hessian, and the multiplicity of zero eigenvalue is at least d h .

Hence, these d h equality constraints on u k 's and v k 's force ?? to be orthogonal to the loss-invariant directions.

The equality constraints of the formx .

So there are L inequality constraints.

Now, the following lemma proves that feasible sets defined by these equality/inequality constraints added to (4) exactly correspond to the regions where DISPLAYFORM19 Recall from Lemma A.1 that DISPLAYFORM20 k , and DISPLAYFORM21 be the only element of i???B DISPLAYFORM22 The proof of Lemma A.3 is in Appendix B.6.In total, there are d h + M ??? L equality constraints and L inequality constraints in each nonconvex QP.

It is also easy to check that these constraints are all linearly independent.

How many QPs do we solve?

Note that in Line 19, we call SO-TEST with {?? i,k } k???[d h ],i???B k = 0, which results in a QP (4) with d h + M equality constraints.

This is done even when we have flat extreme rays, just to take a quick look if a descent direction can be obtained without having to deal with inequality constraints.

If there exist flat extreme rays (Line 22), the algorithm calls SO-TEST for each element of DISPLAYFORM23 In summary, if there is no flat extreme ray, the algorithm solves just one QP with d h + M equality constraints.

If there are flat extreme rays, the algorithm solves one QP with d h + M equality constraints, and 2 K QPs with d h + M ??? L equality constraints and L inequality constraints.

This is also an improvement from the naive approach of solving 2 M QPs.

Concluding the test.

After solving the QP, SO-TEST returns result to SOSP-CHECK.

The algorithm returns two booleans and one perturbation tuple.

The first is to indicate that there is no solution, i.e., there is a descent direction that leads to ??????. Whenever there was any descent direction then we immediately return the direction and terminate.

The second boolean is to indicate that there are nonzero ?? that satisfies ?? T H(z, ??)?? = 0.

After solving all QPs, if any of SO-TEST calls found out ?? = 0 such that g(z, ??)T ?? = 0 and ?? T H(z, ??)?? = 0, then we conclude SOSP-CHECK with "SOSP."

If all QPs terminated with unique minimum at zero, then we can conclude "Local Next, assume for the sake of contradiction that the M k := |B k | data pointsx i 's are linearly dependent, i.e., there exists a 1 , . . . , a M k ??? R, not all zero, such that DISPLAYFORM24 where a 2 , . . . , a M k are not all zero.

This implies that these M k points x i 's are on the same (M k ???2)-dimensional affine space.

To see why, consider for example the case M k = 3: a 2 (x 2 ??? x 1 ) = ???a 3 (x 3 ??? x 1 ), meaning that they have to be on the same line.

By adding any d x + 1 ??? M k additional x i 's, we can see that d x + 1 points are on the same (d x ??? 1)-dimensional affine space, i.e., a hyperplane in R dx .

This contradicts Assumption 2.B.2 PROOF OF LEMMA 2From Assumption 1, (w, y) is twice differentiable and convex in w. By Taylor expansion of (??) at (Y (x i ), y i ), DISPLAYFORM25 where the first-order term DISPLAYFORM26 Also, note that in each of the 2 M divided region (which is a polyhedral cone) of ??, J(x i ) stays constant for all i ??? [m]; thus, g(z, ??) and H(z, ??) are piece-wise constant functions of ??.

Specifically, since the parameter space is partitioned into polyhedral cones, we have g(z, ??) = g(z, ????) and H(z, ??) = H(z, ????) for any ?? > 0.

Suppose that w 1 , w 2 , . . . , w q are orthonormal basis of row(A).

Choose w q+1 , . . .

, w p so that w 1 , w 2 , . . . , w p form an orthonormal basis of R p .

Let W be an orthogonal matrix whose columns are w 1 , w 2 , . . .

, w p , and?? be an submatrix of W whose columns are w q+1 , . . .

, w p .

With this definition, note that I ??? A T (AA T ) ???1 A =???? T .Suppose that we are given ?? (t) satisfying A?? (t) = 0.

FORMULA4 ).

Let Q be a matrix of order r. Consider a nonempty index set J ??? [r].

Given J, Q J refers to the principal submatrix of Q with the rows and columns of Q indexed by J. Let 2[r]

\ ??? denote the set of all nonempty subsets of [r] .

Then ?? ??? ??(Q) if and only if there exists an index set J ??? 2[r]

\ ??? and a vector ?? ??? R |J| such that DISPLAYFORM0 In such a case, the vector ?? ??? R r by DISPLAYFORM1 is a Pareto-eigenvector of Q associated to the Pareto eigenvalue ??.

These lemmas tell us that the Pareto spectrum of Q can be calculated by computing eigensystems of all 2 r ??? 1 possible Q J , which takes O(r 3 2 r ) time in total, and from this we can determine whether a symmetric Q is copositive.

With the preliminary concepts presented, we now start proving our Lemma 4.

We will first transform ?? to eliminate the equality constraints and obtain an inequality constrained problem of the form minimize w:Bw???0 w T Rw.

From there, we can use the theorems from Martin & Jacobson (1981) , which tell us that by testing positive definiteness of a (p???q ???r)??(p???q ???r) matrix and copositivity of a r ?? r matrix we can determine which of the three categories the QP falls into.

Transforming ?? and testing positive definiteness take O(p 3 ) time and testing copositivity takes O(r 3 2 r ) time, so the test in total is done in O(p 3 + r 3 2 r ) time.

We now describe how to transform ?? and get an equivalent optimization problem of the form we want.

We assume without loss of generality that A = [A 1 A 2 ] where A 1 ??? R q??q is invertible.

If not, we can permute components of ??.

Then make a change of variables 1 A 2 has rank r, which means it has full row rank.

Before stating the results from Martin & Jacobson (1981), we will transform the problem a bit further.

Again, assume without loss of generality thatB = B 1B2 whereB 1 ???

R r??r is invertible.

Define another change of variables as the following: Given this transformation, we are ready to state the lemmas.

<|TLDR|>

@highlight

A theoretical algorithm for testing local optimality and extracting descent directions at nondifferentiable points of empirical risks of one-hidden-layer ReLU networks.

@highlight

Proposes an algorithm to check whether a given point is a generalized second-order stationary point.

@highlight

A theoretical algorithm, involving solving convex and non-convex quadratic programs, for checking local optimality and escaping saddles when training two-layer ReLU networks.

@highlight

Author proposes a method to check if a point is a stationary point or not and then classify stationary points as either local min or second-order stationary