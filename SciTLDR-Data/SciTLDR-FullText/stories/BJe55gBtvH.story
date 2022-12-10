Understanding the representational power of Deep Neural Networks (DNNs) and how their structural properties (e.g., depth, width, type of activation unit) affect the functions they can compute, has been an important yet challenging question in deep learning and approximation theory.

In a seminal paper, Telgarsky high- lighted the benefits of depth by presenting a family of functions (based on sim- ple triangular waves) for which DNNs achieve zero classification error, whereas shallow networks with fewer than exponentially many nodes incur constant error.

Even though Telgarsky’s work reveals the limitations of shallow neural networks, it doesn’t inform us on why these functions are difficult to represent and in fact he states it as a tantalizing open question to characterize those functions that cannot be well-approximated by smaller depths.

In this work, we point to a new connection between DNNs expressivity and Sharkovsky’s Theorem from dynamical systems, that enables us to characterize the depth-width trade-offs of ReLU networks for representing functions based on the presence of a generalized notion of fixed points, called periodic points (a fixed point is a point of period 1).

Motivated by our observation that the triangle waves used in Telgarsky’s work contain points of period 3 – a period that is special in that it implies chaotic behaviour based on the celebrated result by Li-Yorke – we proceed to give general lower bounds for the width needed to represent periodic functions as a function of the depth.

Technically, the crux of our approach is based on an eigenvalue analysis of the dynamical systems associated with such functions.

In approximation theory, one typically tries to understand how to best approximate a complicated family of functions using simpler functions as building blocks.

For instance, Weierstrass (1885) proved a general result stating that every continuous function can be uniformly approximated as closely as desired by a polynomial.

It wasn't until later that Vitushkin (1959) gave quantitative bounds between the approximation error and the polynomial's degree.

Drifting away from polynomials and given the recent breakthroughs of deep learning in a variety of difficult tasks like image classification, natural language processing, game playing and self-driving cars, researchers have tried to understand the approximation theory that governs neural networks.

This question of neural network expressivity, i.e. how architectural properties like the depth, width or the activation units affect the functions it can compute, has been a fundamental ongoing challenge with a rich history.

A classical result by (Cybenko (1989) , Hornik et al. (1989) , Fukushima (1980) ) demonstrates the expressive power of neural networks: it states that even two layered neural networks (using well known activation functions) can approximate any continuous function on a bounded domain.

The caveat is that the size of such networks may be exponential in the dimension of the input, which makes them highly susceptible to overfitting as well as impractical, since one can always add extra layers in their model aiming at increasing the representational power of the neural network.

More recently, in a seminal paper by Telgarsky (2016) , it was shown that there exist functions that can be represented by DNNs, i.e, by some particular choice of weights on their edges (and for a wide variety of standard activation units in their layers), yet cannot be approximated by shallow networks unless they are exponentially large.

More concretely, he showed that for any positive integer k, there exist neural networks with Θ(k 3 ) layers, Θ(1) nodes per layer, and Θ(1) distinct parameters which cannot be approximated by networks with O(k) layers, unless they have Ω(2 k ) nodes.

At a high level, he uses the number of oscillations present in certain functions as a notion of "complexity" that distinguishes between deep and shallow networks' representation capabilities via the following facts: a) functions with few oscillations poorly approximate functions with many oscillations, b) functions computed by networks with few layers must have few oscillations and c) functions computed by networks with many layers can have many oscillations.

Our main contribution is a novel connection between the theory of dynamical systems and the representational power of DNNs via the well-studied notion of periodic points, a notion that captures the important notion of fixed points of a continuous function.

Definition 1 (Period).

f n (x 0 ) = x 0 and (point of period n)

In particular, all numbers in C = {x 0 , f (x 0 ), f (f (x 0 )), . . .

, f n−1 (x 0 )} are distinct, each of which is a point of period n and the set C is called a cycle (or orbit) of period n. Observe that since f : [0, 1] → [0, 1] is continuous, it certainly has at least one point of period 1, which is called a fixed point.

For the rest of this paper, we focus on (continuous) Lipschitz functions f : [0, 1] → [0, 1], unless otherwise stated.

Note that the choice of interval [0, 1] is for simplicity of our presentation and that our results will hold for any closed interval [a, b].

As we observe, points of period 3 are contained in both Telgarsky (2016) and Schmitt (2000) constructions and this could as well have been a coincidence, however we show that the existence of periodic points of certain periods are actually one of the reasons explaining why depth is needed to represent functions that contain them (otherwise exponential width is required).

Towards this direction, we will make use of a deep result in the literature of iterated dynamical systems called Sharkovsky's Theorem Sharkovsky (1964; 1965) .

Consider the set of positive natural numbers N * = {1, 2, . . . } and define the following (decreasing) ordering called Sharkovsky's ordering as follows: 3 5 7 · · · (odd numbers bigger than one) 2 · 3 2 · 5 2 · 7 · · · (odd multiples of two but not two) 2 2 · 3 2 2 · 5 2 2 · 7 · · · (odd multiples of four but not four) . . .

This is a total ordering; we write l r or r l whenever l is to the left of r. Sharkovsky showed that this ordering describes which numbers can be periods for a continuous map on an interval; allowed periods need to be a suffix of the Sharkovsky ordering: Theorem 1.1 (Sharkovsky "Forcing" Theorem Sharkovsky (1964; 1965) ).

Let I be a closed interval and f : I → I be a continuous map.

If n is a period for f and n n , then n is also a period for f .

Remark 1.1.

Note that the number 3 is the maximum period according to Sharkovsky's ordering, so an important corollary is that a function having a point of period 3, must also have points of any period.

This special corollary is a weaker version of Sharkovsky's theorem and was proved some years later 2 in a celebrated result by Li & Yorke (1975) , who coined the term "chaos", as used in Mathematics.

1 As usual, f n (x0) denotes the composition of f with itself n times, evaluated at point x0.

2 Due to historical reasons during the late 20th century, the theory of dynamical systems saw a parallel development in the USA and the USSR, hence Sharkovsky's theorem (1964) remained unknown in the USA, until in 1975 a weaker version was rediscovered by James Yorke and his graduate student Tien-Yien Li, in their paper called "Period Three Implies Chaos".

We conclude the subsection with the definition of a prime period of a function f .

A function f has prime period n as long as it has a cycle of period n, but has no cycles with period greater than n according to the Sharkovsky ordering.

For example, in the interval [0, 1] , the function f (x) = 1 − x has prime period 2, since f (f (x)) = 1 − (1 − x) = x so all points are periodic with period 2, except the fixed point at 1/2.

Before formally stating our main theorems, we present an illustrative example inspired from Telgarsky's triangle wave construction and we connect it to DNNs' sensitivity to weight perturbations and their representational power.

An important ingredient in Telgarsky's proof, was the "triangular wave" function (sometimes referred to as the tent map or sawtooth) depicted in Figure 1b and given by:

He shows that the composition of t(x; 2) with itself k times (denoted by t k (x; 2)), will create exponentially (in k) many oscillations and as a result he is able to show a separation for the classification error when using a shallow vs a deep neural network as a predictor.

Our starting point is the observation that the triangular wave function t(x; 2) contains points of period 3, e.g. ( ).

It follows in particular, that t(x; 2) exhibits Li-Yorke Chaos (Li & Yorke (1975) ) in the sense that it contains all periods.

The compositions of such functions will look highly complex (see Figure 2 ) and in fact Telgarsky heavily relied on the highly oscillatory behavior of t(x; 2) to prove his depth separation result.

However, his result doesn't inform us on what would happen if one used a slightly modified version of the triangle wave t(x; 2)?

Observe that since a simple neural network with one hidden layer can represent the function t(x; 2), the question is basically equivalent to asking how modifying the weights on the edges of the neural network can affect its representational power (see Figure 1) , hence the title of the current subsection.

The main question is can we have a general theory that informs us on when will the function composition be hard to represent and when not?

Our paper's main point is to provide an answer by checking if the function at hand has a simple property, relating to the presence of chaotic behavior.

To illustrate our point, consider the generalized triangle wave function t(x; µ) parameterized by µ:

This function, parameterized by µ ranges from [0, µ/2], is closely related to the logistic map (f (x) := rx(1 − x)) used in Schmitt (2000) and exhibits a variety of limiting behaviors: for instance, it converges to a stable fixed point when µ ≤ 1, it exhibits chaos when µ = 2 etc.

Instead of µ = 2, if we set µ = 1, we get the network depicted in Figure 1a , 1d.

Note that compositions of t(x; 1) (created by the same neural network architecture but with slightly different weights), behave completely differently since in the µ = 1 case, we will not get a highly oscillatory behavior.

This can be seen in Figure 3 .

One difference between the two cases is the relative position of the map with the line y = x and this seems to be pointing that fixed points and their generalizations i.e. periodic orbits play an important role when dealing with function compositions.

Indeed, despite the wide range of possibilities one can expect by composing such functions, as we show, their behavior can be characterized using tools from dynamical systems; the exponential growth in complexity (or lack thereof) of these compositions can be explained by invoking a fundamental property of these continuous functions on bounded intervals which is the existence (or not) of periodic points of certain periods.

Similarly, we can argue about changing the parameters of the logistic map which is given by f (x; r) := rx(1 − x) used in Schmitt (2000) for sigmoidal networks (where f (x; 4) was used).

The properties of the logistic map are well known and was first studied by Robert May and Mitchell Feigenbaum (May (1976) and Feigenbaum (1976) ).

Thus we understand that smoothly changing r, we can obtain a plethora of behaviors for small changes in the weights.

Please refer to Appendix C for some figures that illuminate these differences in the logistic map.

(b) t 6 (x; µ) for the tent map with µ = 2 (blue) and µ = 1 (red).

We demonstrate that a simple property of f governs the depth-width trade-offs in order to represent it and we give quantitative bounds for them.

This simple property has to do with the periods that the function f contains.

Informally, our first main theorem states that if a function f contains periodic points with certain periods, then composing f with itself many times, will result in exponentially many oscillations, giving rise to complicated behaviors and chaos:

Assume that there exists a cycle of period n where n = m·p, p is an odd number greater than one and m being a power of two (it might be m = 1).

It holds that there exist x, y ∈ [0, 1] so that f mt "oscillates" (also look Definition 4) at least c t times between x and y for all t ∈ N * , where c is the positive root greater than one of the polynomial equation

Our second main theorem then draws the connection between the number of oscillations a function has and the depth-width trade-offs needed: Theorem 1.3.

Let k be a positive integer and f be a function as above.

We set ρ to be the positive root greater than one of the polynomial equation

with n := Using these theorems, we draw connections with previous results Telgarsky (2016), Schmitt (2000) in a unified way, thus identifying chaotic behavior as the main underlying thread for depth-width trade-offs.

Technically, our approach is based on an eigenvalue analysis of certain matrices associated with such periodic functions.

Understanding the benefits of depths on the expressive power a specific computational model can have, is an important area of research spanning different computational models and results come in the flavor of depth separation arguments.

Roughly speaking, many of the results in this area rely on a suitably defined notion of "complexity" of a function we would like to represent, and then proceed by proving that under this notion, deep models have significantly more power than shallower models.

For example, if the computational model of interest is the family of boolean or threshold circuits, depth lower bounds are given in Hastad (1986); Rossman et al. (2015) ; Håstad (1987); Parberry et al. (1994) ; Kane & Williams (2016) .

Furthermore, people have analyzed sumproduct networks (summation and product nodes) and studied trade-offs for depth (Delalleau & Bengio (2011); Martens & Medabalimi (2014) ). (2019)) and more.

Our work is more closely related to Telgarsky (2015; , and Schmitt (2000) since it is easy to see that their maps are chaotic, but we conjecture that many of the notions of complexity introduced in this line of research to showcase benefits of depth actually arise due to chaotic behavior.

In this sense, we conjecture that chaotic behavior is the main culprit for the failure of neural networks to represent certain functions, unless they are sufficiently deep (or have exponential width).

The crux of the proof of Sharkovsky's theorem provided by Burns & Hasselblatt (2011) contains a covering lemma that will be our starting point to prove our main results.

Before we proceed with the statement of the Covering Lemma, we provide one more important definition.

Definition 3 (Covering relation).

Let f be a function and I 1 , I 2 be two closed intervals.

We say that

For example, the triangle wave t(x; 2) that has the period 3 point be a continuous function and assume f has a cycle C of period n, where n > 1 is an odd number.

Denote β 0 , ..., β n−1 ∈ C the elements of the cycle in increasing order and define the sequence of closed intervals I 0 , ..., I n−2 where I i = [

β i , β i+1 ] (they have pairwise disjoint interiors).

Then, there exists a sub-collection of the aforementioned intervals (not necessarily in the same ordering) J 0 , ...

J r with 1 ≤ r ≤ n − 2 such that the following covering relation holds:

For pictorial illustration of the Covering Lemma, see Figure 4 .

In particular, observe that for n = 3 we get r = 1 so the covering relation is as in Figure 4 .

We conclude this section with the formal definition of crossings (or oscillations) and we refer the reader to Figure 5 for some exapmles.

Definition 4 (Crossings).

We say that a continuous function f :

Observe that if I f,x,y is used to denote 3 the number of intervals the functionf x,y (z) :

is piecewise constant and partitions [0, 1], then C x,y (f ) ≤ I f,x,y .

In this section, we prove our main theorem, the statement of which is given below.

Technically, we make use of the Lemma 1 (Covering Lemma) to show the exponential growth of the number of crossings.

Assume that there exists a cycle of period n where n = m · p, p is an odd number greater than one and m being a power of two (it might be m = 1).

It holds that there exist x, y ∈ [0, 1] so that C x,y (f mt ) is c t for all t ∈ N * , where c is the positive root greater than one of the polynomial equation

Counting the number of oscillations.

For a given continuous function f :

. .

, J r , where 1 ≤ r ≤ n − 2, be the intervals as promised from Lemma 1.

We define a sequence of vectors δ t ∈ N r+1 such that δ t i is defined as the number of times the function f t crosses the interval J i for all 0 ≤ i ≤ r. In particular we define f 0 to be the identity function and hence δ 0 = (1, . . .

, 1) (all ones vector).

For what follows, we will try to express recursively δ t in terms of δ t−1 and in the end we will show that δ k 0 is Ω(c k ) where c is some constant that depends on r. To build some intuition, we first analyze the case of period three and then we prove the general case.

3.1.1 WARM UP: THE CASE OF PERIOD 3 AND THE FIBONACCI SEQUENCE Assume that f has a cycle of period 3, that is the numbers {x 0 , f (x 0 ), f 2 (x 0 )} are distinct and f 3 (x 0 ) = x 0 for some x 0 ∈ [0, 1].

Let β 0 < β 1 < β 2 be the numbers x 0 , f (x 0 ), f 2 (x 0 ) in increasing order.

We define I 0 = [β 0 , β 1 ] and I 1 = [β 1 , β 2 ].

From Lemma 1, when n = 3, we can see that r = 1 and thus we have the following possibilities for the covering relations:

We define J 0 to be the interval among I 0 , I 1 that involves the self-loop covering and J 1 to be the remaining interval.

Define δ t ∈ N 2 as above, and so we get that: that corresponds to the covering relations between J 0 , J 1 (which consists of a directed cycle with a self-loop at vertex J 0 ).

The reason we have an inequality instead of an equality is because the Covering Lemma only guarantees that the number of times J 0 "covers" J 0 and J 1 is at least one and not necessarily exactly one.

We set α 0 = δ 0 and we define α t+1 = Aα t .

It is clear that δ t ≥ α t (entry-wise) for all t ∈ N. Moreover, α t 0 is the well-known Fibonacci sequence F t+1 (with F 0 = F 1 = 1), therefore

.

We conclude that δ

.

See also Figure 5 for a pictorial illustration about the proof for t = 1, 2, 3, 4.

Assume that f has a cycle of period n > 3 with n odd, that is the numbers

From Lemma 1 it follows that there is a subcollection of the intervals I 0 , ..., I n−2 (with not necessarily the same ordering) J 0 , ..., J r (1 ≤ r ≤ n − 2) such that

The interval J 0 is the one that involves the self-loop covering.

As in the case for n = 3, we define δ t which is in N r+1 , with δ t i capturing the number of times f t crosses the interval J i .

We get that:

where δ 0 = (1, . . . , 1) (all ones vector) and A ∈ R (r+1)×(r+1) is defined to be:

In words, A is the adjacency matrix of a graph with r +1 nodes that is a directed cycle that involves a self-loop at vertex J 0 .

We define α t in a similar way as in the case for period three, i.e., α t+1 = Aα t and α 0 = δ 0 so that δ t ≥ α t (entry-wise) for all t ∈ N. We can easily observe that the following holds:

Our next plan is to compute a lower bound on the spectral radius of the matrix A (denoted by sp(A )) with the following claim (proof in Appendix A).

Claim 3.1.

The characteristic polynomial of A is:

Let us call ρ r the largest root in absolute value of the polynomial π(λ) in A.1.

Since A is a nonnegative matrix, the largest root in absolute value is actually a positive real number (by the PerronFrobenius theorem).

It is easy to see that the polynomial in A.1 has always a root greater than one and less than two (by Bolzano's theorem, see π(1) = −1 < 0 and π(2) = 2 r+1 − 2 r − 1 = 2 r − 1 > 0).

Hence we have sp(A) = ρ r > 1.

Furthermore, it is easy to see that since A is a non-negative matrix (and powers of A are also non-negative), it holds that

for all t ≥ 1, that is the row with the largest sum of its entries is the first row (row for i = 0).

Using the fact that A t ∞ ≥ sp(A t ) = ρ t r , that is the spectral radius of a matrix is always at most any matrix norm, we conclude that

The case of odd period greater than three follows by noting that We would like to make the following two remarks:

Remark 3.1.

The spectral radius ρ r is strictly decreasing in r: this is easy to see since ρ r > 1 and is satisfying the equation x r+1 − x r = 1 (note that x r+1 − x r is increasing in r for x > 1).

This implies that smaller odd periods can potentially have a number of crossings that grows at faster rates than larger odd periods, hence giving rise to more complex behaviors.

See also Remark 4.1.

Remark 3.2 (The case of even period but not power of two).

Our result above is applied for cycles of period n = m · n where m is a power of two and n is an odd number greater than one.

The trick is to observe that if a function has cycle of period n, then f m has a cycle of period n (which is an odd number greater than one).

Therefore, the number of oscillations C x,y (f mt ) with x, y being the endpoints of J 0 , is at least ρ t n −2 for t ∈ N.

Proof of Theorem 3.1.

The proof now follows from the case analysis carried out in Sections 3.1.1, 3.1.2 and Remark 3.2.

Lemma 2 (Period power of two -proof in Appendix A).

There exist continuous functions f with prime period n that is a power of two so that the number of crossings C x,y (f t ) scales at most polynomially with t for any x, y ∈ [0, 1].

Building on Telgarsky (2015; 2016), the representation power of different networks will be measured via the classification error.

For a given collection of n points (x i , y i ) n i=1 with y i ∈ {0, 1}, one can define the classification error of a function g to be:

In this section, we argue that functions with cycles of period not a power of two, will have compositions for which any shallow neural network will have classification error a positive constant.

Assume we are given a continuous function f : [0, 1] → [0, 1] so that f has a cycle of period m × p where p is an odd number greater than one and m is a power of two.

From Theorem 3.1, there exist

2 where ρ r is defined to be the root that is greater than one of the polynomial equation λ r+1 − λ r − 1 = 0.

We set ρ := ρ p−2 , h := f k·m and assume that g : [0, 1] → [0, 1] is a neural network with l layers and u nodes (ReLU activations) per layer.

In Lemma 2.1 of Telgarsky (2015) , it is proved that a neural network with u ReLU units per layer and with l layers is piecewise affine with at most (2m) l pieces.

(note that we changed the threshold to be x+y 2 instead of 1 2 that was used in Telgarsky (2015)).

Since C x,y (h) is at least ρ k , it holds that there exist points (x i , y i )

such that h(x j ) = x, y j = 0 for j odd and h(x j ) = y, y j = 1 for j even.

It is clear that for this collection of points the classification error of the function h is zero, whereas the classification error for function g is bounded from below by

The above inequality is an application of Lemma 2.2 of Telgarsky (2015) (with careful counting it has been slightly improved).

By choosing u to be at most

8 it holds that the classification error R(g) ≥ 1 4 for any neural network g with u ReLUs and l layers.

The above discussion implies the following theorem: Theorem 4.1 (Classification Error Theorem).

Let k be a positive integer and f be a function of period m×p with p an odd number greater than one and m being a power of two (it might hold m = 1).

We set ρ to be the positive root greater than one of the polynomial equation

We can construct a sequence of points (x i , y i ) 2n i=1 with n := ρ k 2 so that the classification error of function f mk is zero, whereas the classification error of any neural network of l layers and u nodes

4 .

Remark 4.1.

Observe that if the number of units u per layer is constant and the number of layers l is o(k), then the classification error is always a positive constant for any neural network (whereas for f mk is zero).

Moreover, observe that since ρ is decreasing in p (recall p is the odd factor of the period), it holds that the classification error decreases as p increases (with fixed number of layers and nodes per layer).

This indicates that the composition of functions with large odd period is simpler than of functions with small odd period (period greater than one) following the intuition we have from the Sharkovsky's ordering.

A APPENDIX Claim A.1.

The characteristic polynomial of A is:

Proof.

Let I denote the identity matrix of size (r + 1) × (r + 1).

We consider the matrix:

Observe that λ = 0, 1 are not eigenvalues of the matrix A ., hence we can multiply the first row by 1 λ−1 , the second row by 1 λ(λ−1) , the third row by 1 λ 2 (λ−1) ,. . .

, the i-th row by 1 λ i−1 (λ−1) (and so on) and add them to the last row.

Let B be the resulting matrix:

It is clear that det(B) = 0 as an equation has the same roots as det(A − λI) = 0.

Since B is an upper triangular matrix, it follows that

.

We conclude that the eigenvalues of A (and hence of A) must be roots of (λ r − λ r−1 )λ − 1 and the claim follows.

Lemma 3 (Period power of two).

There exist continuous functions f with prime period n that is a power of two so that the number of crossings C x,y (f t ) scales at most polynomially with t for any x, y ∈ [0, 1].

Proof.

The easiest example one can construct is the function f : [0, 1] → [0, 1] that is defined f (x) = 1 − x. Observe that for any a ∈ [0, 1] one has f (f (a)) = a and moreover if a = 1 2 then f (a) = a. Hence f is a function of prime period two.

It is also clear that f t (x) = x if t is even and f t (x) = 1 − x if t is odd, so the number of crossings is always one for all t ∈ N * .

One other less trivial example is the following function (see also Figure 6 ):

It is not hard to see that this function has prime period four (f (1) = 4, f (4) = 2, f (2) = 3, f (3) = 1).

By letting δ A t ij = t + 3 for all t ∈ N * .

We conclude that α t 0 + α t 1 + α t 2 = t + 3, therefore the number of crossings for J 0 , J 1 , J 2 of the function f t grows linearly with t (and not exponentially).

Since the function we defined is of prime period four and is piecewise monotone (and so is any composition with itself) in each interval J 0 , J 1 , J 2 , we conclude that the number of crossings of f t for any possible pairs of values is at most linear in t.

Here, we show an example function, that has a point of period 5, but not period 3, thereby respecting the Sharkovsky ordering.

Our proof approach for general odd periods is similar to the case of period 3, by using the induced covering graph and counting the crossings over each interval.

This is illustrated in Figure 7 .

In this section, we illustrate how the compositions of the logistic map f (x; r) := rx(1 − x) behaves as r varies slightly.

We give certain examples in the form of Figure 8 .

It is known that the map when r = 3.9, has a point of period 3.

In contrast when r is reduced to 3.5 the map has a point of period 4 and further bringing r down to 3.2 will ensure that the map has a point of period 2.

The figures below illustrate how the oscillations grow under these scenarios.

In this section, we provide some additional theoretical and experimental remarks on our characterization.

If we add a bias term in the ReLU activation unit, e.g., use max(v, ) instead of max(v, 0) for the activation gates, where is a small number (positive or negative), then our results do not change; in particular our trade-off in Theorem 4.1 still holds (since the Lemma 2.2 from Telgarsky (2015) is for general sawtooth functions).

But, if one adds the bias term to the function f itself, then things get more interesting indeed: Suppose f has some period p where p is not a power of two; due to bifurcation phenomena (i.e., phenomena arising because we are at critical regimes of parameters such as the parameter µ in our generalized triangle wave function), then the compositions of the function (f +bias term) with itself may give rise to different behaviors qualitatively compared to f .

In particular, the function (f +bias term) might not have period p anymore.

Intuitively, one can think that the small bias term is amplified after many compositions and is not negligible anymore.

One such example is the triangle function f (x) = φx for 0 ≤ x ≤ 0.5 and φ(1−x) for 1/2 ≤ x ≤ 1 where φ = (1 + √ 5)/2 is the golden ratio.

This function has period 3, see Figure 9a .

However, if we consider the function g(x) = (φ − )x for 0 ≤ x ≤ 0.5 and (φ − )(1 − x) for 0.5 ≤ x ≤ 1 with > 0 (arbitrarily small positive) then g does not have period 3, see Figure 9b .

In this sense, period as a property can be brittle to numerical changes if we are at the critical point.

In this section, we provide experimental evidence for our results by training a neural network of constant width, but with increasing depth on a classification task that closely resembles the n-alternating points problem that appeared in Telgarsky (2015) and is the foundation of our separation results as well.

As mentioned before, this is a specific instance of a function that has a point of period 3.

For simplicity, we do not consider this original problem exactly but rather a "smoothed" variant of it, in order to make it more amenable to the training procedure.

Our goal is to create a diagram showing how the classification error drops as a function of the depth of the network for a fixed value of the width.

So we create 8000 equally spaced points from [0,1] (in increasing order), where the first 1000 points are of label 0, the second 1000 are label 1 and this label alternates every 1000 points.

This is what we call a "smoothed" alternating point problem.

Although, the theory would have used the classical 8-alternating points to argue about the lower bounds, in practice, performing training of deep (4 and above layers) and narrow networks (hidden layers with less than 4 neurons) with very few data points is a major challenge, see for instance Lu et al. (2018) .

Apart from the separation results that we show in theory, we show empirically that deep networks generally do improve the accuracy in Figure 10 : We see that depth does reduce the classification error for this particular task and when depth is 5, the classification error is close to 0.

The saturation in between may be attributed to the general uncertainties in the training/optimization.

this task compared to the shallow network and in fact a deep network with 5 layers can reach an accuracy of 99.04%.

Any additional uncertainties in the error is generally attributed to the training procedure.

To perform the experiments, we vary the depth of the neural network (excluding the input and the output layer) as d = 1, 2, 3, 4, 5.

In addition, we fix the neurons for each layer to be 6.

All activations are ReLU's, while the last layer is the classifier that uses a sigmoid to output probabilities.

Each model adds one extra hidden layer and we make use of the same hyper-parameters to train all networks.

Moreover, we require the training error or the classification error to tend to 0 during the training procedure, i.e, we will try and overfit the data (as we try to demonstrate a representation result, rather than a statistical/generalization result).

Thus, for the actual training we use the same parameters to train all the different models using the "ADAM" optimizer Kingma & Ba (2014) and make the epochs to be 200 in order to enable overfitting.

To record the training error, we verify that the training saturates by seeing the performance over the epochs and report by default the error in the last epoch.

The results are shown in Figure 10 .

The code (ipython notebook) is submitted along with the supplementary material.

In a nutshell, our paper provides a "natural" property of a function (periodic points of certain periods) and then derive depth-width trade-offs based on it.

This addresses some questions raised not only in Telgarsky (2016; 2015) 's works, but also in the paper Poole et al. (2016) that seeks to provide a natural, general measure of functional complexity helping us understand the benefits of depth.

On the contrary, many of previous depth separation results take a worst case approach for the representation question (showing that there exist functions implemented by deep networks that are hard to approximate with a shallow net).

However, it is not clear whether such analysis applies to the typical instances arising in practice of neural-networks.

We believe that our work together with Telgarsky (2016; 2015) and the paper Eldan & Shamir (2016) show a depth separation argument for very natural functions, such as the triangle waves or the indicator function of the unit ball.

Given a specific prediction task in practice, how could one assess the period?

We believe that this would be extremely useful yet a very difficult question that seems to be outside the reach of current techniques in the literature.

Previous works and our work so far are able to present depth separation for representing certain functions.

We point out that, intuitively, our characterization result consists of a certificate informing us qualitatively and quantitatively about which functions have complicated compositions and which not.

Similar to computational problems in class NP, if one is given the certificate (the points (x 1 , . . . , x p ), then one can easily verify (if we have oracle access to evaluate the function f ), if the given function has a p-periodic cycle with points (x 1 , . . . , x p ).

Nevertheless, we believe that finding the certificate for arbitrary continuous functions is not a straightforward problem, except maybe for particular restricted classes of functions.

Having said that, we want to emphasize that in many prediction problems that are inspired by physics, one may a priori expect to have complicated dynamics behavior and hence require deeper networks for better performance.

Such examples include efforts to solve the notorious 3-body problem or turbulent flows showing empirical evidence that complex physical processes require deep networks (see for instance, Ling et al. (2016) and Breen et al. (2019) that uses a 10 layered neural network).

@highlight

In this work, we point to a new connection between DNNs expressivity and Sharkovsky’s Theorem from dynamical systems, that enables us to characterize the depth-width trade-offs of ReLU networks 

@highlight

Shows how the expressive power of NN depends on its depth and width, furthering the understanding of the benefit of deep nets for representing certain function classes.

@highlight

The authors derive depth-width tradeoff conditions for when relu networks are able to represent periodic functions using dynamical systems analysis.