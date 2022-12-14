The weak contraction mapping is a self mapping that the range is always a subset of the domain, which admits a unique fixed-point.

The iteration of weak contraction mapping is a Cauchy sequence that yields the unique fixed-point.

A gradient-free optimization method as an application of weak contraction mapping is proposed to achieve global minimum convergence.

The optimization method is robust to local minima and initial point position.

Many gradient-based optimization methods, such as gradient descent method, Newton's method and so on, face great challenges in finding the global minimum point of a function.

As is known, searching for the global minimum of a function with many local minima is difficult.

In principle, the information from the derivative of a single point is not sufficient for us to know the global geometry property of the function.

For a successful minimum point convergence, the initial point is required to be sufficiently good and the derivative calculation need to be accurate enough.

In the gradientbased methods, the domain of searching area will be divided into several subsets with regards to local minima.

And eventually it will converge to one local minimum depends on where the initial point locates at.

Let (X,d) be a metric space and let T:X → X be a mapping.

For the inequality that, d(T (x), T (y)) ≤ qd(x, y), ∀x, y ∈ X.(1)if q ∈ [0, 1), T is called contractive; if q ∈ [0, 1], T is called nonexpansive; if q < ∞, T is called Lipschitz continuous(1; 2).

The gradient-based methods are usually nonexpansive mapping the solution exists but is not unique for general situation.

For instance, if the gradient descent method is written as a mapping T and the objective function has many local minima, then there are many fixed points accordingly.

From the perspective of spectra of bounded operator, for a nonexpansive mapping any minima of the objective function is an eigenvector of eigenvalue equation T (x) = λx ,in which λ = 1.

In the optimization problem, nonexpansive mapping sometimes works but their disadvantages are obvious.

Because both the existence and uniqueness of solution are important so that the contractive mapping is more favored than the nonexpansive mapping(3; 4).Banach fixed-point theorem is a very powerful method to solve linear or nonlinear system.

But for optimization problems, the condition of contraction mapping T : X → X that d(T (x), T (y)) ≤ qd(x, y) is usually too strict and luxury.

In the paper, we are trying to extend the Banach fixedpoint theorem to an applicable method for optimization problem, which is called weak contraction mapping.

In short, weak contraction mapping is a self mapping that always map to the subset of its domain.

It is proven that weak contraction mapping admits a fixed-point in the following section.

How to apply the weak contraction mapping to solve an optimization problem?

Geometrically, given a point, we calculate the height of this point and utilize a hyperplane at the same height to cut the objective function, where the intersection between the hyperplane and the objective function will form a contour or contours.

And then map to a point insider a contour, which the range of this mapping is always the subset of its domain.

The iteration of the weak contraction mapping yields a fixed-point, which coincides with the global minimum of the objective function.

In this section, the concept of weak contraction mapping and its fixed-point will be discussed in detail.

Definition 1.

Let (X, d and D) be a metric space.

Both the metric measurement d and D are defined in the space.

And the metric measurement D(X) refers to the maximum distance between two points in the vector space X: DISPLAYFORM0 Definition 2.

Let (X, d and D) be a complete metric space.

Then a mapping T : X → X is called weak contraction mapping on X if there exists DISPLAYFORM1 The weak contraction mapping is an extension of contraction map with a looser requirement that DISPLAYFORM2 and D) be a non-empty complete metric space with weak contraction mapping T : X → X. Then T admits a unique fixed-point x * in X when X 0 is decided.

Let x 0 ∈ X be arbitrary and define a sequence {x n } be setting: x n = T (x n−1 ).

The Theorem.1 is proven in the following lemmas.

By definition, there exists q DISPLAYFORM3 Cauchy sequence in (X, d and D) and hence converges to a limit x * in X 0 .Proof.

Let m, n ∈ N such that m > n. DISPLAYFORM4 Let > 0 be arbitrary, since q ∈ [0, 1), we can find a large N ∈ N such that DISPLAYFORM5 Hence, by choosing m, n large enough: DISPLAYFORM6 Thus, {x n } is Cauchy and converges to a point x * ∈ X 0 .

DISPLAYFORM7 Proof.

DISPLAYFORM8 ) with regards to a specific X 0 .Proof.

Suppose there exists another fixed-point y that T (y) = y, then choose the subspace X i that both the x * and y are the only elements in X i .

By definition, X i+1 = R(T (X i )) so that, both the x * and y are elements in X i+1 , namely, DISPLAYFORM9 Let a hyperplane L cut the objective function f(x), the intersection of L and f(x) forms a contour (or contours).

Observing that the contour (or contours) will divide X into two subspaces the higher subspace X > := {x | f (x) > h, ∀x ∈ X} and the lower subspace DISPLAYFORM10 Geometrically, the range of weak contraction mapping shrinks over iterates, such that, DISPLAYFORM11 .

Based on lemma.1.3, the D(X i ) measurement converges to zero as i goes to infinity, namely, lim DISPLAYFORM12 And the sequence of iteration DISPLAYFORM13 is Cauchy sequence that converge to the global minimum of objective function f(x) if the f(x) has a unique global minimum point.

Lemma 1.5.

Provided there is a unique global minimum point of an objective function, then x * is the global minimum point of the function.

Proof.

The global minimum point must be insider the lower space {X ≤ i , ∀i ∈ N 0 }.

Similar to the proof of uniqueness of fixed-point, suppose the global minimum point x min of objective function is different from x * .

By measuring the distance between fixed-point X * and the global minimum point DISPLAYFORM14 The inequality above indicates d(x * , x min ) = 0, thus x * = x min .Compared with contraction map, the weak contraction map is much easier to implement in the optimization problem as the requirement DISPLAYFORM15 t require x i in sequence {x n } must move closer to each other for every step but confine the range of x i to be smaller and smaller.

Therefore, the sequence {x n } can still be a Cauchy and has the asymptotic behavior to converge to the fixed-point.

Given the objective function f (x) has a unique global minimum point, the task is to find a weak contraction mapping T : X → X such that the unique fixed-point of mapping is the global minimum point of the function.

The weak contraction map for the optimization problem can be implemented in following way.

First, provide one arbitrary initial point x 0 to the function and calculate the height L = f (x 0 ) of the point and this height is the corresponding contours' level; Second, given the initial point map to another point inside the contour.

One practical way is to solve the equation f (x) = L and get n number of roots which locate on a contour(or contours) and then the average of these roots is the updated searching point.

And then repeat these process until the iteration of searching point converge.

This contour-based optimization algorithm utilizes the root-finding algorithm to solve the equation f (x) = L and get n number of roots.

The starting point for the root-finding algorithm is generated by a random number generator.

This stochastic process will help the roots to some extent widely distribute over the contour rather than concentrate on somewhere.

The inequality d(x m , x n ) ≤ q n D(X 0 ) indicates the rate of convergence, namely, the smaller q is the high rate of convergence will be achieved.

Geometrically, the equation f (x) = L is the intersection of the objective function and a hyperplane whose height is L. We hope the hyperplane move downward in a big step during each iterate and the centroid of the contour refers to the most likely minimum position.

Therefore, averaging the roots as an easy and effective way to map somewhere near the centroid.

And there is a trade-off between the number of roots on the contour and the rate of convergence.

The larger amount of roots on the contour, the more likely the average locates closer to the centroid of the contour, and then the less iterates are required for convergence.

In another words, the more time spend on finding roots on a contour, the less time spend on the iteration, vice verse.

The global minimum point x * is the fixed-point of the iteration x i+1 = T x i and solves the equation It is worth noting that the size of contour become smaller and smaller during the iterative process and eventually converge to a point, which is the minimum point of the function.

DISPLAYFORM0

As shown in the previous examples, averaging the roots on the contour is an effective approach to map a point inside the interior of the lower space X ≤ when it is convex.

However, in general situation, the space X ≤ is not guaranteed to be convex.

In that case, it is important to decompose the lower space X ≤ into several convex subsets.

In this study, the key intermediate step is to check whether two roots belong to the same convex subset and decompose all roots into several convex subsets accordingly.

One practical way to achieve that is to pair each two roots and scan function's value along the segment between the two roots and check whether there exists a point higher than contour's level.

Loosely speaking, if two roots belong to the same convex subset, the value of function along the segment is always lower than the contour's level.

Otherwise, the value of function at somewhere along the segment will be higher than the contour's level.

Traverse all the roots and apply this examination on them, then we can decompose the roots with regards to different convex subsets.

This method is important to map a point insider interior of a contour and make hyperplane move downwards.

To check whether two roots belong to the same convex subset, N number of random points along the segment between two roots are checked whether higher than the contour's level or not.

When we want to check the function's value along the segment between r m and r n .

The vector k = r m − r n is calculated so that the random point p i locate on the segment can be written as p i = r n + ( r m − r n ), ∈ (0, 1), where the is a uniform random number from 0 to 1.

Then check whether the inequality holds for all random point such that f (p i ) < f (r m ), ∀i ≤ N .

Obviously, the more random points on the segment are checked, the less likely the point higher than contour's level is missed(9; 10).After the set of roots are decomposed into several convex subsets, the averages of roots with regards to each subsets are calculated and the lowest one is returned as an update point from each iterate.

Thereafter, the remaining calculation is repeat the iterate over and over until convergence and return the converged point as the global minimum.

Nevertheless, the algorithm has been tested on Ackley function where the global minimum locates at (0,0) that f (0, 0) = 0.

And the first 6 iterates of roots and contours is shown FIG2 and the minimum point (-0.00000034,0.00000003) return by algorithm is shown in TABLE.

3.

The test result shows that the optimization algorithm is robust to local minima and able to achieve the global minimum convergence.

The quest to find to the global minimum pays off handsomely.

When the optimization method is tested on Ackley function, the average of roots and the level of contour for each iteration is shown above.

In summary, the main procedure of the stochastic contour-based optimization method is decomposed into following steps: 1.

Given the initial guess point x for the objective function and calculate the contour level L; 2.

Solve the equation f (x) = L and get n number of roots.

Decompose the set of roots into several convex subsets,return the lowest average of roots as an update point from each iterate; 3.

Repeat the above iterate until convergence.

The weak contraction mapping is a self mapping that always map to a subset of domain.

Intriguingly, as an extension of Banach fixed-point theorem, the iteration of weak contraction mapping is a Cauchy and yields a unique fixed-point, which fit perfectly with the task of optimization.

The global minimum convergence regardless of initial point position and local minima is very significant strength for optimization algorithm.

We hope that the advanced optimization with the development of the weak contraction mapping can contribute to empower the modern calculation.

@highlight

A gradient-free method is proposed for non-convex optimization problem 