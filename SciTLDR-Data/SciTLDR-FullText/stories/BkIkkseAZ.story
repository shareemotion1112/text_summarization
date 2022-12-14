In this paper, we study the problem of optimizing a two-layer artificial neural network that best fits a training dataset.

We look at this problem in the setting where the number of parameters is greater than the number of sampled points.

We show that for a wide class of differentiable activation functions (this class involves most nonlinear functions and excludes piecewise linear functions), we have that arbitrary first-order optimal solutions satisfy global optimality provided the hidden layer is non-singular.

We essentially show that these non-singular hidden layer matrix satisfy a ``"good" property for these big class of activation functions.

Techniques involved in proving this result inspire us to look at a new algorithmic, where in between two gradient step of hidden layer, we add a stochastic gradient descent (SGD) step of the output layer.

In this new algorithmic framework, we extend our earlier result and show that for all finite iterations the hidden layer satisfies the``good" property mentioned earlier therefore partially explaining success of noisy gradient methods and addressing the issue of data independency of our earlier result.

Both of these results are easily extended to hidden layers given by a flat matrix from that of a square matrix.

Results are applicable even if network has more than one hidden layer provided all inner hidden layers are arbitrary, satisfy non-singularity, all activations are from the given class of differentiable functions and optimization is only with respect to the outermost hidden layer.

Separately, we also study the smoothness properties of the objective function and show that it is actually Lipschitz smooth, i.e., its gradients do not change sharply.

We use smoothness properties to guarantee asymptotic convergence of $O(1/\text{number of iterations})$ to a first-order optimal solution.

Neural networks architecture has recently emerged as a powerful tool for a wide variety of applications.

In fact, they have led to breakthrough performance in many problems such as visual object classification BID13 , natural language processing BID5 and speech recognition BID17 .

Despite the wide variety of applications using neural networks with empirical success, mathematical understanding behind these methods remains a puzzle.

Even though there is good understanding of the representation power of neural networks BID1 , training these networks is hard.

In fact, training neural networks was shown to be NP-complete for single hidden layer, two node and sgn(??) activation function BID2 .

The main bottleneck in the optimization problem comes from non-convexity of the problem.

Hence it is not clear how to train them to global optimality with provable guarantees.

Neural networks have been around for decades now.

A sudden resurgence in the use of these methods is because of the following: Despite the worst case result by BID2 , first-order methods such as gradient descent and stochastic gradient descent have been surprisingly successful in training these networks to global optimality.

For example, Zhang et al. (2016) empirically showed that sufficiently over-parametrized networks can be trained to global optimality with stochastic gradient descent.

Neural networks with zero hidden layers are relatively well understood in theory.

In fact, several authors have shown that for such neural networks with monotone activations, gradient based methods will converge to the global optimum for different assumptions and settings BID16 BID10 BID11 BID12 ).Despite the hardness of training the single hidden layer (or two-layer) problem, enough literature is available which tries to reduce the hardness by making different assumptions.

E.g., BID4 made a few assumptions to show that every local minimum of the simplified objective is close to the global minimum.

They also require some independent activations assumption which may not be satisfied in practice.

For the same shallow networks with (leaky) ReLU activations, it was shown in Soudry & Carmon (2016) that all local minimum are global minimum of the modified loss function, instead of the original objective function.

Under the same setting, Xie et al. (2016) showed that critical points with large "diversity" are near global optimal.

But ensuring such conditions algorithmically is difficult.

All the theoretical studies have been largely focussed on ReLU activation but other activations have been mostly ignored.

In our understanding, this is the first time a theoretical result will be presented which shows that for almost all nonlinear activation functions including softplus, an arbitrary first-order optimal solution is also the global optimal provided certain "simple" properties of hidden layer.

Moreover, we show that a stochastic gradient descent type algorithm will give us those required properties for free for all finite number of iterations hence even if the hidden layer variables are data dependent, we still get required properties.

Our assumption on data distribution is very general and can be reasonable for practitioners.

This comes at two costs: First is that the hidden layer of our network can not be wider than the dimension of the input data, say d. Since we also look at this problem in over-parametrized setting (where there is hope to achieve global optimality), this constraint on width puts a direct upper-bound of d 2 on the number of data points that can be trained.

Even though this is a strong upper bound, recent results from margin bounds BID19 show that if optimal network is closer to origin then we can get an upper bound on number of samples independent of dimension of the problem which will ensure closeness of population objective and training objective.

Second drawback of this general setting is that we can prove good properties of the optimization variables (hidden layer weights) for only finite iterations of the SGD type algorithm.

But as it is commonly known, stochastic gradient descent converges to first order point asymptotically so ideally we would like to prove these properties for infinitely many iterations.

We compare our results to some of the prior work of Xie et al. (2016) and Soudry & Carmon (2016) .

Both of these papers use similar ideas to examine first order conditions but give quite different results from ours.

They give results for ReLU or Leaky ReLU activations.

We, on the other hand, give results for most other nonlinear activations, which can be more challenging.

We discuss this in section 3 in more detail.

We also formally show that even though the objective function for training neural networks is nonconvex, it is Lipschitz smooth meaning that gradient of the objective function does not change a lot with small changes in underlying variable.

To the best of our knowledge, there is no such result formally stated in the literature.

Soltanolkotabi et al. (2017) discuss similar results, but there constant itself depends locally on w max , a hidden layer matrix element, which is variable of the the optimization function.

Moreover, there result is probabilistic.

Our result is deterministic, global and computable.

This allows us to show convergence results for the gradient descent algorithm, enabling us to establish an upper bound on the number of iterations for finding an ??-approximate first-order optimal solution ( ???f () ??? ??).

Therefore our algorithm will generate an ??-approximate first-order optimal solution which satisfies aforementioned properties of the hidden layer.

Note that this does not mean that the algorithm will reach the global optimal point asymptotically.

As mentioned before, when number of iterations tend to infinity, we could not establish "good" properties.

We discuss technical difficulties to prove such a conjecture in more detail in section 5 which details our convergence results.

At this point we would also like to point that there is good amount of work happening on shallow neural networks.

In this literature, we see variety of modelling assumptions, different objective functions and local convergence results.

BID15 focuses on a class of neural networks which have special structure called "Identity mapping".

They show that if the input follows from Gaussian distribution then SGD will converge to global optimal for population objective of the "identity mapping" network.

BID3 show that for isotropic Gaussian inputs, with one hidden layer ReLU network and single non-overlapping convolutional filter, all local minimizers are global hence gradient descent will reach global optimal in polynomial time for the population objective.

For the same problem, after relaxing the constraint of isotropic Gaussian inputs, they show that the problem is NP-complete via reduction from a variant of set splitting problem.

In both of these studies, the objective function is a population objective which is significantly different from training objective in over parametrized domain.

In over-parametrized regime, Soltanolkotabi et al. (2017) shows that for the training objective with data coming from isotropic Gaussian distribution, provided that we start close to the true solution and know maximum singular value of optimal hidden layer then corresponding gradient descent will converge to the optimal solution.

This is one of its kind of result where local convergence properties of the neural network training objective function have studied in great detail.

Our result differ from available current literature in variety of ways.

First of all, we study the training problem in the over-parametrized regime.

In that regime, the training objective can be significantly different from population objective.

Moreover, we study the optimization problem for many general non-linear activation functions.

Our result can be extended to deeper networks when considering the optimization problem with respect to outermost hidden layer.

We also prove that stochastic noise helps in keeping the aforementioned properties of hidden layer.

This result, in essence, provides justification for using stochastic gradient descent.

Another line of study looks at the effect of over-parametrization in the training of neural networks BID9 Nguyen & Hein, 2017) .

These result are not for the same problem as they require huge amount of over-parametrization.

In essence, they require the width of the hidden layer to be greater than number of data points which is unreasonable in many settings.

These result work for fairly general activations as do our results but we require a moderate over-parametrization, width ?? dimension ??? number of data population, much more reasonable in practice as pointed before from margin bound results.

They also work for deeper neural network as do our results when optimization is with respect to outermost hidden layer (and aforementioned technical properties are satisfied for all hidden layers).

We define set [q] := {1, . . .

, q}. For any matrix A ??? R a??b , we write vect(A) ??? R ab??1 as vector DISPLAYFORM0 is the i-th element in vector z. B i (r) represents a l i -ball of radius r, centred at origin.

We define component-wise product of two vectors with operator .

We say that a collection of vectors, DISPLAYFORM1 Similarly, we say that collection of matrices, DISPLAYFORM2 A fully connected two-layer neural network has three parameters: hidden layer W , output layer ?? and activation function h. For a given activation function, h, we define neural network function as DISPLAYFORM3 In the above equation, W ??? R n??d is hidden layer matrix, ?? ??? R n is the output layer.

Finally h : R ??? R is an activation function.

The main problem of interest in this paper is the two-layer neural network problem given by DISPLAYFORM4 In this paper, we assume that DISPLAYFORM5 DISPLAYFORM6 (3.1) can also be written in a matrix vector product form: DISPLAYFORM7 where DISPLAYFORM8 Notice that if matrix D ??? R nd??N is of full column rank (which implies nd ??? N , i.e., number of samples is less than number of parameters) then it immediately gives us that s = 0 which means such a stationary point is global optimal.

This motivates us to investigate properties of h under which we can provably keep matrix D full column rank and develop algorithmic methods to help maintain such properties of matrix D. Note that similar approach was explored in the works of Soudry & Carmon (2016) and Xie et al. (2016) .

To get the full rank property for matrix D, Soudry & Carmon (2016) use leaky ReLu function.

Basically this leaky activation function adds noise to entries of matrix D which allows them to show matrix D is full rank and hence all local minimums are global minimums.

So this is essentially a change of model.

We, on the other hand, do not change model of the problem.

Moreover, we look at the algorithmic process of finding W differently.

We show that SGD will achieve full rank property of matrix D with probability 1 for all finite iterations.

So this is essentially a property of the algorithm and not of the model.

Even if that is the case, to show global optimality, we need to prove that matrix D is full column rank in asymptotic sense.

That question was partly answered in Xie et al. (2016) .

They show that matrix D is full column rank by achieving a lower bound on smallest singular value of matrix D. But to get this, they need two facts.

First, the activation function has to be ReLu so that they can find the spectrum of corresponding kernel matrix.

Second, they require a bound on discrepancy of weights W .

These conditions are strong in the sense that they restrict the analysis to a non-differentiable activation function and finding an algorithm satisfying discrepancy constraint on W can be a difficult task.

On other hand, our results are proved for a simple SGD type algorithm which is easy to implement.

But we do not get a lower bound on singular value of D in asymptotic sense.

There are obvious pluses and minuses for both types of results.

For the rest of the discussion, we will assume that n = d (our results can be extended to case n ??? d easily) and hence W is a square matrix.

In this setting, we develop the following algorithm whose output is a provable first-order approximate solution.

Here we present the algorithm and in next sections we will discuss conditions that are required to satisfy full rank property of matrix D as well as convergence properties of the algorithm.

In Algorithm 1, we use techniques inspired from alternating minimization to minimize with respect to ?? and W .

For minimization with respect to ??, we add gaussian noise to the gradient information.

This will be useful to prove convergence of this algorithm.

We use randomness in ?? to ensure some "nice" properties of W which help us in proving that matrix D generated along the trajectory of Algorithm 1 is full column rank.

More details will follow in next section.

The algorithm has two loops.

An outer loop implements a single gradient step with respect to hidden layer, W .

For each outer loop iteration, there is an inner loop which optimizes objective function with respect to ?? using a stochastic gradient descent algorithm.

In the stochastic gradient descent, we generate a noisy estimated of ??? ?? f (W, ??) as explained below.

Let ?? ??? R d be a vector whose elements are i.i.d.

Gaussian random variable with zero mean.

Then for a given value of W we define stochastic gradient w.r.t.

?? as follows: DISPLAYFORM9 (3.4)Then we know that DISPLAYFORM10 .

We can choose a constant ?? > 0 such that following holds DISPLAYFORM11 Moreover, in the algorithm we consider a case where ?? ??? R. Note that R can be kept equal to R d but that will make parameter selection complicated.

In our convergence analysis, we will use DISPLAYFORM12 for some constant R, to make parameter selection simpler.

We use prox-mapping P x : R d ??? R as follows: DISPLAYFORM13 In case R is a ball centred at origin, solution of (3.7) is just projection of x ??? y on that ball.

For case where R = R d then the solution is quantity x ??? y itself.

Initialize N o to predefined iteration count for outer ietaration Initialize N i to predefined iteration count for inner iteration Begin outer iteration: DISPLAYFORM0 Notice that the problem of minimization with respect to ?? is a convex minimization problem.

So we can implement many procedures developed in the Stochastic optimization literature to get the convergence to optimal value BID18 ).

In the analysis, we note that one does not even need to implement complete inner iteration as we can skip the stochastic gradient descent suboptimally given that we improve the objective value with respect to where we started, i.e., DISPLAYFORM1 In essence, if evaluation of f for every iteration is not costly then one might break out of inner iterations before running N i iterations.

If it is costly to evaluate function values then we can implement the whole SGD for convex problem with respect to ?? as specified in inner iteration of the algorithm above.

In each outer iteration, we take one gradient decent step with respect to variable W .

We have total of N o outer iterations.

So essentially we evaluate DISPLAYFORM2 Overall, this algorithm is new form of alternate minimization, where one iteration can be potentially left suboptimally and other one is only one gradient step.

We prove in this section that arbitrary first order optimal points are globally optimal.

One does not expect to have an arbitrary first order optimal point because it has to depend on data.

We still would like to put our analysis here because that inspires us to consider a new algorithmic framework in Section 3 providing similar results for all finite iterations of the algorithm.

We say that h : R ??? R satisfy the condition "C1" if DISPLAYFORM0 One can easily notice that most activation functions used in practice e.g., DISPLAYFORM1 satisfy the condition C1.

Note that h (x) also satisfy condition C1 for all of them.

In fact, except for very small class of functions (which includes linear functions), none of the continuously differentiable functions satisfy condition C1.

We first prove a lemma which establishes that columns of the matrix D (each column is a vector form of d??d matrix itself) are linearly independent when W = I d and h satisfies condition C1.

We later generalise it to any full rank W using a simple corollary.

The statement of following lemma is intuitive but its proof is technical.

DISPLAYFORM2 is full rank with measure 1.This means that if u i in the Problem (2.1) are coming from a Lebesgue measure then by corollary 4.2 DISPLAYFORM3 will be a full rank collection given that we have maintained full rank property of W .

Now note that in the first-order condition, given in (3.3), row of matrix D are scaled by constant factors ??[j]'s, j ??? [d].

Notice that we may assume ??[j] = 0 because otherwise there is no contribution of corresponding j-th row of W to the Problem (2.1) and we might as well drop it entirely from the optimization problem.

Hence we can rescale rows of matrix D by factor DISPLAYFORM4 without changing the rank.

In essence, corollary 4.2 implies that matrix D is full rank when W is full rank.

So by our discussion in earlier section, we show that satisfying first-order optimality is enough to show global optimality under condition C1 for data independent W. Remark 4.4 As a result of corollary above one can see that the collection of vectors h(W x i ) is full rank under the assumption that W is non-singular, x i ??? R d are independently sampled from Lebesgue measure and h satisfies condition C1.Remark 4.5 Since collection h(W u i ) is also full rank, we can say that z i := h(W 1 u i ) are independent and sampled from a Lebesgue measure for a non-singular matrix W 1 .

Applying the Lemma to z i , we have collection of matrices g(W 2 z i )z i T are full rank with measure 1 for non-singular W 2 and g satisfying condition C1.

So we see that for multiple hidden layers satisfying non-singularity, we can apply full rank property for collection of gradients with respect to outermost hidden layer.

has rank min{rank(W )d, N } with measure 1 by removing dependent rows and using remark 4.6.

Even though we have proved that collection h(Wis full rank in the previous section, we need such W to be independent of data.

In general, any algorithm will use data to find W and it appears that results in previous section are not meaningful in practice.

However, the analysis of Lemma 4.1 motivates the idea that stochastic noise of ?? might help in obtaining the required properties of W and D along the trajectory of Algorithm 1.

In this section we first prove that by using random noise in stochastic gradient on ?? gives a non-singular W k in every iteration.

Then using this fact, we prove that matrix D generated along the algorithm is also full rank.

The proof techniques are very similar to proof of Lemma 4.1.

Later on, we will also show that overall algorithm will converge to approximate first-order optimal solution to Problem (2.1) by using smoothness properties.

It should be noted however that this can not guarantee convergence to a global optimal solution.

To prove such a result, one needs to analyze the smallest singular value of random matrix D, defined in (3.3).

More specifically, we have to show that ?? min (D) decreases at the rate slower than the first-order convergence rate of the algorithm so that the overall algorithm converges to the global optimal solution.

Even if it is very difficult to prove such a result in theory, we think that such an assumption about ?? min (D) is reasonable in practice.

Now we analyze the algorithm.

For the sake of simplicity of notation, let us define 2) where N i is the inner iteration count in Algorithm 1.

Essentially ?? [k] contains the record of all random samples used until the k-th outer iteration in Algorithm 1 and ?? j [Ni] contains record of all random samples used in the inner iterations of j-th outer iteration.

DISPLAYFORM0 DISPLAYFORM1 where W k are matrices generated by Algorithm 1 and measure Pr{. DISPLAYFORM2 Now that we have proved that W k 's generated by the algorithm are full rank, we show that matrix D generated along the trajectory of the algorithm is full rank for any finite number of iterations.

We use techniques inspired from Lemma 4.1 but this time we use Lebesgue measure over ?? rather than data.

Over randomness of ??, we can show that our algorithm will not produce any W such that corresponding matrix D is rank deficient.

Since ?? is essentially designed to be independent of data so we will not produce rank deficient D throughout the process of randomized algorithm.

DISPLAYFORM3 and v is a random vector with Lebesgue measure in R d .

W , Z ??? R d??d and Z = 0.

Let h be a function which follows condition C1.

Also assume that W is full rank with measure 1 over randomness of v.

Then h(W u i )

u Proof.

We know that DISPLAYFORM4 Now apply Lemma 5.2 to obtain the required result.

Note that Lemma 5.3 is very similar to the result in section 4.

Some remarks are in order.

Remark 5.4 As a result of Lemma 5.3 above, one can see that collection of vectors h(W k u i ) is full rank for all finite iterations of Algorithm 1.Remark 5.5 If we have a neural network with multiple hidden layer, we can assume that inner layers are initialized to arbitrary full rank matrices and we are optimizing w.r.t.

outermost hidden layer.

Corollary 4.2 and Remark 4.4 give us that input to outermost hidden layer are independent vectors sampled from some lebesgue measure.

Then applying Algorithm 1 to optimize w.r.t.

outermost hidden layer will give us similar results as mentioned in Lemma 5.3.Hence we showed that algorithm will generate full rank matrix D for any finite iteration.

Now to prove convergence of the algorithm, we need to analyze the function f (defined in (2.1)) itself.

We show that f is a Lipschitz smooth function for any given instance of data {u DISPLAYFORM5 .

This will give us a handle to estimate convergence rates for the given algorithm.

Lemma 5.6 Assuming that h : R ??? R is such that its gradients, hessian as well as values are bounded and data DISPLAYFORM6 is given then there exists a constant L such that DISPLAYFORM7 Moreover, a possible upper bound on L can be as follows: DISPLAYFORM8 Remark 5.7 Before staing the proof, we should stress that assumptions on h is satisfied by most activation functions e.g., sigmoid, sigmoid symmetric, gaussian, gaussian symmetric, elliot, elliot symmetric, tanh, Erf.

Remark 5.8 Note that one can easily estimate value of L given data and ??.

Moreover, if we put constraints on ?? 2 then L is constant in every iteration of the algorithm 1.

As mentioned in section 3, this will provide an easier way to analyze the algorithm.

Lemma 5.9 Assuming that scalar function h is such that |h(??)| ??? u then there exists L ?? s.t.

DISPLAYFORM9 Notice that Lemma 5.9 gives us value of L ?? irrespective of value of W or data.

Also observe that f (W, ??) is convex function since hessian DISPLAYFORM10 which is the sum of positive semidefinite matrices.

By Lemma 5.9, we know that f (W, ??) is smooth as well.

So we can use following convergence result provided by BID14 for stochastic composite optimization.

A simplified proof can be found in appendix.

Theorem 5.10 Assume that stepsizes ?? i satisfy 0 < ?? i ??? 1/2L ?? , ??? i ??? 1.

Let {?? av i+1 } i???1 be the sequence computed according to Algorithm 1.

Then we have, DISPLAYFORM11 5) DISPLAYFORM12 i where ?? 1 is the starting point for inner iteration and ?? is defined in (3.5).

Now we look at a possible strategy of selecting stepsize ?? i .

Suppose we adopt a constant stepsize policy then we have DISPLAYFORM13

we get DISPLAYFORM0 By Lemma 5.6, the objective function for neural networks is Lipschitz-smooth with respect to the hidden layer, i.e., it satisfies eq (5.3).

Notice that it is equivalent to saying DISPLAYFORM1 (5.7) Since we have a handle on the smoothness of objective function, we can provide a convergence result for the overall algorithm.

DISPLAYFORM2 8) where R/2 is the radius of origin centred ball, R in algorithm, defined as R : DISPLAYFORM3 In view of Theorem 5.11, we can derive a possible way of choosing ?? k , ?? and N i to obtain a convergence result.

More specifically, if DISPLAYFORM4 L and ?? k ?? is chosen according to (5.6) then we have DISPLAYFORM5 Note that in the algorithm 1, we have proved that having a stochastic noise helps keeping matrix D full rank for all finite iterations.

Then in Theorem 5.11, we showed a methodical way of achieving approximate first order optimal.

So essentially at the end of the finitiely many steps of algorithm 1, we have a point W which satisfies full rank property of D and is approximately first order optimal.

We think this kind of result can be extended to variety of different first order methods developed for Lipschitz-smooth non-convex optimization problems.

More specifically, accelerated gradient method such as unified accelerated method proposed by BID8 or accelerated gradient method by BID7 can be applied in outer iteration.

We can also use stochastic gradient descent method for outer iteration.

For this, we need a stochastic algorithm that is designed for non-convex and Lipschitz smooth function optimization.

Randomized stochastic gradient method, proposed by BID6 , Stochastic variance reduction gradient method (SVRG) by Reddi et al. or Simplified SVRG by Allen-Zhu & Hazan can be employed in outer iteration.

Convergence of these new algorithms will follow immediately from the convergence results of respective studies.

Some work may be needed to prove that they hold matrix D full rank.

We leave that for the future work.

We also leave the problem of proving a bound on singular value for future.

This will close the gap between empirical results and theory.

Value of Lipschitz constant, L, puts a significant impact on the running time of the algorithm.

Notice that if L increases then correspondingly N o and N i increase linearly with L.

So we need methods by which we can reduce the value of the estimate of L. One possible idea would be to use l 1 -ball for feasible region of ??.

More specifically, if R = B 1 (R/2) then we can possibly enforce sparsity on ?? which will allow us to put better bound on L.

In this appendix, we provide proofs for auxiliary results.

The result is trivially true for d =1, we will show this using induction on d. DISPLAYFORM0 .

Note that it suffices to prove independence of vector DISPLAYFORM1 DISPLAYFORM2 Moreover, for any collection satisfying x i ??? Z i , corresponding collection of vector v i are linearly dependent, i.e., DISPLAYFORM3 Noticing the definition of Z 1 , we can choose > 0 s.t.

x 1 := x 1 + e 1 ??? Z 1 .

Since we ensure that x 1 ??? Z 1 then by (A.1) we have DISPLAYFORM4 So using (A.1) and (A.2) we get DISPLAYFORM5 2 components of v 1 ??? v 1 are zero.

Let us define: DISPLAYFORM6 . . .

DISPLAYFORM7 By definition we have y i ??? R d???1 are independently sampled from (d ??? 1)-dimensional Lebesgue measure.

So by inductive hypothesis, rank of collection of matrices h(y i )y DISPLAYFORM8 2 then ?? 2 = ?? ?? ?? = ?? N = 0 with measure 1, then by (A.3) we have w 1 = 0 with measure 1, which is contradiction to the fact that w 1 = 0 with measure 1.

This gives us DISPLAYFORM9 Notice that (A.4) in its matrix form can be written as linear system DISPLAYFORM10 By (A.6), we have that vector of ??'s lies in the null space of the matrix.

Finally by inductive hypothesis and (A.5) we conclude that the dimension of that space is DISPLAYFORM11 2 ??? R N ???1 be the basis of that null space i.e. DISPLAYFORM12 Define t i ??? R 2d???1 as: DISPLAYFORM13 then we can rewrite (A.3) as DISPLAYFORM14 where DISPLAYFORM15 2 and z 1 part of the equation is already satisfied due to selection of null space.

2 ) are constant.

Let us define the set S to be the index set of linearly independent DISPLAYFORM16 DISPLAYFORM17 2 ] and every other row is a linear combination of rows in S. Since (A.8) is consistent so the same combination must be valid for the rows of w 1 .

Now if N ??? d 2 ??? 1 then number of variables in (A.8) is ??? 2d ??? 3 but number of equations is 2d ??? 1, therefore at least two equations are linearly dependent on other equation.

This implies last (2d ??? 2) equations then function must be dependent on each other: DISPLAYFORM18 for some fixed combination ?? j , ?? j .

If we divide above equation by and take the limit as ??? 0 then we see that h satisfies following differential equation on interval (a DISPLAYFORM19 which is a contradiction to the condition C1!

Clearly this leaves only one case i.e. N = d 2 and (2d ??? 1) equations must satisfy dependency of the following form for all x1 ??? (a1 , b(1) DISPLAYFORM20 Again by similar arguments, the combination is fixed.

Let H(x) = xh(x) then dividing above equation by and taking the limit as ??? 0, we can see that h satisfies following differential equation: DISPLAYFORM21 DISPLAYFORM22 Here the second statement follows from the fact W is a non-singular matrix.

Now by Lemma 4.1 we have that collection h(x i )x i T is linearly independent with measure 1.

So DISPLAYFORM23 is linearly independent with measure 1.

Since any rotation is U is a full rank matrix so we have the result.

A.3 PROOF OF LEMMA 5.1 This is true for k = 0 trivially since we are randomly sampling matrix W 0 .

We now show this by induction on k.

Recall that gradient of f (W, ??) with respect to W can be written as DISPLAYFORM24 Notice that in effect, we are multiplying i-th row of the rank one matrix h (W u i )u i T by i-th element of vector ??.

So this can be rewritten as a matrix product DISPLAYFORM25 where ?? := diag{??[i], i = 1, . . .

, d}. So iterative update of the algorithm can be given as DISPLAYFORM26 Notice that given ?? [k] , vector ?? k+1 and corresponding diagonal matrix ?? k+1 are found by SGD in the inner loop so ?? k+1 is a random vector.

More specifically, since {?? DISPLAYFORM27 induces a Lebesgue measure on random variable DISPLAYFORM28 then W k is deterministic quantity.

For the sake of contradiction, take any vector v that is supposed to be in the null space of W k+1 with positive probability.

DISPLAYFORM29 Now the last equation is of the form DISPLAYFORM30 Suppose we can find such ?? with positive probability.

Then we can find hypercuboid Z := {x ??? R d |a < x < b} such that any ?? k+1 in given hypercuboid can solve equation (A.10).

By induction we have b = 0.

We may assume b[1] = 0.

Then to get contradiction on existence of Z, we observe that first equation in (A.10) is: DISPLAYFORM31 can not be 0.

Hence we arrive at a contradiction to the assumption that there existed a hypercuboid Z containing solutions of (A.10).

Since measure on ?? k+1 was induced by {?? DISPLAYFORM32 A.4 PROOF OF LEMMA 5.2We use induction on d. For d = 1 this is trivially true.

Now assume this is true for d ??? 1.

We will show this for d. DISPLAYFORM33 For simplicity of notation define t i := Zu i .

Due to simple linear algebraic fact provided by full rank property of W we have rank of collection (h(W u i )u DISPLAYFORM34 For the sake of contradiction, say the collection is rank deficient with positive probability then there exists d-dimensional volume V such that for all v ??? V, we have h(W u i )u i T is not full rank where DISPLAYFORM35 Without loss of generality, we may assume d-dimensional volume to be a hypercuboid V := {x ??? R d |a < x < b} (if not then we can inscribe a hypercuboid in that volume).

Let us take v ??? V and ?? small enough such that v := v + ??e 1 ??? V. Correspondingly we have z i and z i .

Note that DISPLAYFORM36 Here DISPLAYFORM37 Similarly we also have v i = c i g i .

Now by the act that v, v corresponding to z, z are in V, and our assumption of linear dependence for all v ??? V we get DISPLAYFORM38 .

Also by induction on d ??? 1, we have that DISPLAYFORM39 is an invertible matrix and rewrite one part of equation (A.12) as DISPLAYFORM40 .

So essentially we have satisfied one part of equations (A.12) and (A.13).

Notice that since we are moving only one coordinate of random DISPLAYFORM41 ) (by ?? incremental changes) keeping all other elements of v constant so we will have y i as constants which implies g i , G, G are constant.

So for the sake of simplicity of notation we define l : DISPLAYFORM42 Now, we look at the remaining part of two equation (A.12),(A.13): DISPLAYFORM43 which can be rewritten as DISPLAYFORM44 After (A.15) ??? (A.14), we have DISPLAYFORM45 Now note that (A.16), characterizes incremental changes in C, C, ?? due to ??.

So taking the limit as ?? ??? 0, we have DISPLAYFORM46 Here, last equation is due to product rule in calculus.

In (A.17), we see that we have 2d???1 equations and Assume that all the gradients in this proof are w.r.t.

W then we know that DISPLAYFORM47 DISPLAYFORM48 where the last inequality follows from Cauchy-Schwarz inequality.

DISPLAYFORM49 , ??? i then we are done.

Let ?? max := max DISPLAYFORM50 Suppose the Lipschitz constants for the first and second term are L i,L and L i,R respectively.

Then DISPLAYFORM51 ) and possible upper bound on value of L would become DISPLAYFORM52 Since the Hessian of scalar function h(??) is bounded so we have h (x) is Lipschitz continuous with constant L h .

Let r 1 , r 2 be two row vectors then we claim h (r 1 x) ??? h (r 2 x) 2 ??? L h x 2 .

r 1 ??? r 2 2 , ??? r 1 , r 2 because: h (r 1 x) ??? h (r 2 x) 2 ??? L h r 1 x ??? r 2 x ??? L h x 2 r 1 ??? r 2 2 From the relation above we have the following: Noting that DISPLAYFORM53 DISPLAYFORM54 we have DISPLAYFORM55 where u 1 and u 2 are upper bounds on scalar functions |h(??)| and |h (??)| respectively and d is rowdimension of W .A.7 PROOF OF THEOREM 5.11We know by Lemma 5.6 that f (??, ??) is a Lipschitz smooth function.

So using (5.7) we have DISPLAYFORM56 DISPLAYFORM57 (?? k ??? L/2?? 2 k ).

Now taking expectation with respect to ?? [No] (which is defined in (5.1)), we have DISPLAYFORM58 DISPLAYFORM59 .

@highlight

This paper talks about theoretical properties of first-order optimal point of two layer neural network in over-parametrized case