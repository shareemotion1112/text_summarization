The universal approximation theorem, in one of its most general versions, says that if we consider only continuous activation functions σ, then a standard feedforward neural network with one hidden layer is able to approximate any continuous multivariate function f to any given approximation threshold ε, if and only if σ is non-polynomial.

In this paper, we give a direct algebraic proof of the theorem.

Furthermore we shall explicitly quantify the number of hidden units required for approximation.

Specifically, if X in R^n is compact, then a neural network with n input units, m output units, and a single hidden layer with {n+d choose d} hidden units (independent of m and ε), can uniformly approximate any polynomial function f:X -> R^m whose total degree is at most d for each of its m coordinate functions.

In the general case that f is any continuous function, we show there exists some N in O(ε^{-n}) (independent of m), such that N hidden units would suffice to approximate f. We also show that this uniform approximation property (UAP) still holds even under seemingly strong conditions imposed on the weights.

We highlight several consequences: (i) For any δ > 0, the UAP still holds if we restrict all non-bias weights w in the last layer to satisfy |w| < δ. (ii) There exists some λ>0 (depending only on f and σ), such that the UAP still holds if we restrict all non-bias weights w in the first layer to satisfy |w|>λ. (iii) If the non-bias weights in the first layer are *fixed* and randomly chosen from a suitable range, then the UAP holds with probability 1.

hidden units (independent of m and ε), can uniformly approximate any polynomial function f : X → R m whose total degree is at most d for each of its m coordinate functions.

In the general case that f is any continuous function, we show there exists some N ∈ O(ε −n ) (independent of m), such that N hidden units would suffice to approximate f .

We also show that this uniform approximation property (UAP) still holds even under seemingly strong conditions imposed on the weights.

We highlight several consequences: (i) For any δ > 0, the UAP still holds if we restrict all non-bias weights w in the last layer to satisfy |w| < δ. (ii) There exists some λ > 0 (depending only on f and σ), such that the UAP still holds if we restrict all non-bias weights w in the first layer to satisfy |w| > λ. (iii) If the non-bias weights in the first layer are fixed and randomly chosen from a suitable range, then the UAP holds with probability 1.

A standard (feedforward) neural network with n input units, m output units, and with one or more hidden layers, refers to a computational model N that can compute a certain class of functions ρ : R n → R m , where ρ = ρ W is parametrized by W (called the weights of N ).

Implicitly, the definition of ρ depends on a choice of some fixed function σ : R → R, called the activation function of N .

Typically, σ is assumed to be continuous, and historically, the earliest commonly used activation functions were sigmoidal.

A key fundamental result justifying the use of sigmoidal activation functions was due to Cybenko (1989) , Hornik et al. (1989) , and Funahashi (1989) , who independently proved the first version of what is now famously called the universal approximation theorem.

This first version says that if σ is sigmoidal, then a standard neural network with one hidden layer would be able to uniformly approximate any continuous function f : X → R m whose domain X ⊆ R n is compact.

Hornik (1991) extended the theorem to the case when σ is any continuous bounded non-constant activation function.

Subsequently, Leshno et al. (1993) proved that for the class of continuous activation functions, a standard neural network with one hidden layer is able to uniformly approximate any continuous function f : X → R m on any compact X ⊆ R n , if and only if σ is non-polynomial.

Although a single hidden layer is sufficient for the uniform approximation property (UAP) to hold, the number of hidden units required could be arbitrarily large.

Given a subclass F of real-valued continuous functions on a compact set X ⊆ R n , a fixed activation function σ, and some ε > 0, let N = N (F, σ, ε) be the minimum number of hidden units required for a single-hidden-layer neural network to be able to uniformly approximate every f ∈ F within an approximation error threshold of ε.

If σ is the rectified linear unit (ReLU) x → max(0, x), then N is at least Ω( 1 √ ε ) when F is the class of C 2 non-linear functions (Yarotsky, 2017) , or the class of strongly convex differentiable functions (Liang & Srikant, 2016) ; see also (Arora et al., 2018) .

If σ is any smooth non-polynomial function, then N is at most O(ε −n ) for the class of C 1 functions with bounded Sobolev norm (Mhaskar, 1996) ; cf. (Pinkus, 1999, Thm. 6.8) , (Maiorov & Pinkus, 1999) .

As a key highlight of this paper, we show that if σ is an arbitrary continuous non-polynomial function, then N is at most O(ε −n ) for the entire class of continuous functions.

In fact, we give an explicit upper bound for N in terms of ε and the modulus of continuity of f , so better bounds could be obtained for certain subclasses F, which we discuss further in Section 4.

Furthermore, even for the wider class F of all continuous functions f : X → R m , the bound is still O(ε −n ), independent of m.

To prove this bound, we shall give a direct algebraic proof of the universal approximation theorem, in its general version as stated by Leshno et al. (1993) (i.e. σ is continuous and non-polynomial).

An important advantage of our algebraic approach is that we are able to glean additional information on sufficient conditions that would imply the UAP.

Another key highlight we have is that if F is the subclass of polynomial functions f : X → R m with total degree at most d for each coordinate function, then n+d d

hidden units would suffice.

In particular, notice that our bound N ≤ n+d d

does not depend on the approximation error threshold ε or the output dimension m.

We shall also show that the UAP holds even under strong conditions on the weights.

Given any δ > 0, we can always choose the non-bias weights in the last layer to have small magnitudes no larger than δ.

Furthermore, we show that there exists some λ > 0 (depending only on σ and the function f to be approximated), such that the non-bias weights in the first layer can always be chosen to have magnitudes greater than λ.

Even with these seemingly strong restrictions on the weights, we show that the UAP still holds.

Thus, our main results can be collectively interpreted as a quantitative refinement of the universal approximation theorem, with extensions to restricted weight values.

Outline: Section 2 covers the preliminaries, including relevant details on arguments involving dense sets.

Section 3 gives precise statements of our results, while Section 4 discusses the consequences of our results.

Section 5 introduces our algebraic approach and includes most details of the proofs of our results; details omitted from Section 5 can be found in the appendix.

Finally, Section 6 concludes our paper with further remarks.

Let N be the set of non-negative integers, let 0 n be the zero vector in R n , and let Mat(k, ) be the vector space of all k-by-matrices with real entries.

For any function f :

denote the t-th coordinate function of f (for each 1 ≤ t ≤ m).

Given α = (α 1 , . . . , α n ) ∈ N n and any n-tuple x = (x 1 , . . . , x n ), we write x α to mean x α1 1 · · · x αn n .

If x ∈ R n , then x α is a real number, while if x is a sequence of variables, then x α is a monomial, i.e. an n-variate polynomial with a single term.

Define W Given any X ⊆ R n , let C(X) be the vector space of all continuous functions f : X → R. We use the convention that every f ∈ C(X) is a function f (x 1 , . . .

, x n ) in terms of the variables x 1 , . . . , x n , unless n = 1, in which case f is in terms of a single variable x (or y).

We say that f is nonzero if f is not identically the zero function on X. Let P(X) be the subspace of all polynomial functions in C(X).

For each d ∈ N, let P ≤d (X) (resp.

P d (X)) be the subspace consisting of all polynomial functions of total degree ≤ d (resp.

exactly d).

More generally, let C(X, R m ) be the vector space of all continuous functions f : X → R m , and define P(X, R m ), P ≤d (X, R m ),

Throughout, we assume that σ ∈ C(R).

be the j-th column vector of W (k) , and let w (k) i,j be the (i, j)-th entry of W (k) (for k = 1, 2).

The index i begins at i = 0, while the indices j, k begin at j = 1, k = 1 respectively.

For convenience, let w W is given by the map

where "·" denotes dot product, and (1, x) denotes a column vector in R n+1 formed by concatenating 1 before x. The class of functions that neural networks N with one hidden layer can compute is precisely {ρ i,j is called a bias weight (resp.

non-bias weight) if i = 0 (resp.

i = 0).

Notice that we do not apply the activation function σ to the output layer.

This is consistent with previous approximation results for neural networks.

The reason is simple:

W has essentially the same effect as allowing for bias weights w (2) 0,j .

Although some authors, e.g. (Leshno et al., 1993) , do not explicitly include bias weights in the output layer, the reader should check that if σ is not identically zero, say σ(y 0 ) = 0, then having a bias weight w N +1,j = c σ(y0) ; this means our results also apply to neural networks without bias weights in the output layer (but with one additional hidden unit).

A key theme in this paper is the use of dense subsets of metric spaces.

We shall consider several notions of "dense".

First, recall that a metric on a set S is any function d : S × S → R such that for all x, y, z ∈ S, the following conditions hold:

(i) d(x, y) ≥ 0, with equality holding if and only if x = y;

The set S, together with a metric on S, is called a metric space.

For example, the usual Euclidean norm for vectors in R n gives the Euclidean metric (u, v) → u − v 2 , hence R n is a metric space.

In particular, every pair in W N can be identified with a vector in R (m+n+1)N , so W N , together with the Euclidean metric, is a metric space.

Given a metric space X (with metric d), and some subset U ⊆ X, we say that U is dense in X (w.r.t.

d) if for all ε > 0 and all x ∈ X, there exists some u ∈ U such that d(x, u) < ε.

Arbitrary unions of dense subsets are dense.

If U ⊆ U ⊆ X and U is dense in X, then U must also be dense in X.

A basic result in algebraic geometry says that if p ∈ P(R n ) is non-zero, then {x ∈ R n : p(x) = 0} is a dense subset of R n (w.r.t.

the Euclidean metric).

This subset is in fact an open set in the Zariski topology, hence any finite intersection of such Zariski-dense open sets is dense; see (Eisenbud, 1995) .

More generally, the following is true: Let p 1 , . . .

, p k ∈ P(R n ), and suppose that X := {x ∈ R n : p i (x) = 0 for all 1 ≤ i ≤ k}. If p ∈ P(X) is non-zero, then {x ∈ X : p(x) = 0} is a dense subset of X (w.r.t.

the Euclidean metric).

In subsequent sections, we shall frequently use these facts.

Let X ⊆ R n be a compact set.

(Recall that X is compact if it is bounded and contains all of its limit points.)

For any real-valued function f whose domain contains X, the uniform norm of f on X is f ∞,X := sup{|f (x)| : x ∈ X}. More generally, if f : X → R m , then we define f ∞,X := max{ f ∞,X : 1 ≤ j ≤ m}. The uniform norm of functions on X gives the uniform metric (f, g) → f − g ∞,X , hence C(X) is a metric space.

Theorem 2.1 (Stone-Weirstrass theorem).

Let X ⊆ R n be compact.

For any f ∈ C(X), there exists a sequence {p k } k∈N of polynomial functions in P(X) such that lim k→∞ f − p k ∞,X = 0.

Let X ⊆ R be compact.

For all d ∈ N and f ∈ C(X), define

( 1) A central result in approximation theory, due to Chebyshev, says that for fixed d, f , the infimum in (1) is attained by some unique p * ∈ P ≤d (X); see (Rivlin, 1981, Chap.

1) .

This unique polynomial p * is called the best polynomial approximant to f of degree d.

Given a metric space X with metric d, and any uniformly continuous function f : X → R, the modulus of continuity of f is a function

By the Heine-Cantor theorem, any continuous f with a compact domain is uniformly continuous.

Theorem 2.2 (Jackson's theorem; see (Rivlin, 1981, Cor.

1.4 .1)).

Let d ≥ 1 be an integer, and let Y ⊆ R be a closed interval of length r ≥ 0.

Suppose f ∈ C(Y ), and let p * be the best polynomial approximant to f of degree

Throughout this section, let X ⊆ R n be a compact set.

Theorem 3.1.

Let d ≥ 1 be an integer, and let f ∈ P ≤d (X, R m ) (i.e. each coordinate function f

Furthermore, the following holds: (i) Given any λ > 0, we can choose this W to satisfy the condition that |w

(ii) There exists some λ > 0, depending only on f and σ, such that we could choose the weights of W in the first layer to satisfy the condition that w

(1) j 2 > λ for all j. Theorem 3.2.

Let f ∈ C(X, R m ), and suppose σ ∈ C(R)\P(R).

Then for every ε > 0, there exists an integer N ∈ O(ε −n ) (independent of m), and some W ∈ W N , such that f − ρ σ W ∞,X < ε.

In particular, if we let D := sup{ x − y 2 : x, y ∈ X} be the diameter of X, then we can set N = n+dε dε

, where

)

Furthermore, we could choose this W to satisfy either (i) or (ii), where (i), (ii) are conditions on W as described in Theorem 3.1.

, and suppose that σ ∈ C(R)\P(R).

Then there exists λ > 0 (which depends only on f and σ) such that for every ε > 0, there exists an integer N (independent of m) such that the following holds:

Let W ∈ W N such that each w

is chosen uniformly at random from the set {u ∈ R n : u 2 > λ}. Then, with probability 1, there exist choices for the bias weights w

0,j (for 1 ≤ j ≤ N ) in the first layer, and (both bias and non-bias) weights w

The universal approximation theorem (version of Leshno et al. (1993) ) is an immediate consequence of Theorem 3.2 and the observation that σ must be non-polynomial for the UAP to hold, which follows from the fact that the uniform closure of P ≤d (X) is P ≤d (X) itself, for every integer d ≥ 1.

Alternatively, we could infer the universal approximation theorem by applying the Stone-Weirstrass theorem (Theorem 2.1) to Theorem 3.1.

Given fixed n, m, d, a compact set X ⊆ R n , and σ ∈ C(R)\P ≤d−1 (R), Theorem 3.1 says that we could use a fixed number N of hidden units (independent of ε) and still be able to approximate any function f ∈ P ≤d (X, R m ) to any desired approximation error threshold ε.

Our ε-free bound, although possibly surprising to some readers, is not the first instance of an ε-free bound: Neural networks with two hidden layers of sizes 2n + 1 and 4n + 3 respectively are able to uniformly approximate any f ∈ C(X), provided that we use a (somewhat pathological) activation function (Maiorov & Pinkus, 1999) ; cf. (Pinkus, 1999) .

Lin et al. (2017) showed that for fixed n, d, and a fixed smooth non-linear σ, there is a fixed N (i.e. ε-free), such that a neural network with N hidden units is able to approximate any f ∈ P ≤d (X).

An explicit expression for N is not given, but we were able to infer from their constructive proof that N = 4 n+d+1 d − 4 hidden units are required, over d − 1 hidden layers (for d ≥ 2).

In comparison, we require less hidden units and a single hidden layer.

Our proof of Theorem 3.2 is an application of Jackson's theorem (Theorem 2.2) to Theorem 3.1, which gives an explicit bound in terms of the values of the modulus of continuity ω f of the function f to be approximated.

The moduli of continuity of several classes of continuous functions have explicit characterizations.

For example, given constants k > 0 and 0 < α ≤ 1, recall that a continuous function f :

for all x, y ∈ X, and it is called α-Hölder if there is some constant c such that |f (x)−f (y)| ≤ c x−y α for all x, y ∈ X. The modulus of continuity of a k-Lipschitz (resp.

α-Hölder) continuous function f is ω f (t) = kt (resp.

ω f (t) = ct α ), hence Theorem 3.2 implies the following corollary.

n → R is α-Hölder continuous, then there is a constant k such that for every ε > 0, there exists some

An interesting consequence of Theorem 3.3 is the following: The freezing of lower layers of a neural network, even in the extreme case that all frozen layers are randomly initialized and the last layer is the only "non-frozen" layer, does not necessarily reduce the representability of the resulting model.

Specifically, in the single-hidden-layer case, we have shown that if the non-bias weights in the first layer are fixed and randomly chosen from some suitable fixed range, then the UAP holds with probability 1, provided that there are sufficiently many hidden units.

Of course, this representability does not reveal anything about the learnability of such a model.

In practice, layers are already pre-trained before being frozen.

It would be interesting to understand quantitatively the difference between having pre-trained frozen layers and having randomly initialized frozen layers.

Theorem 3.3 can be viewed as a result on random features, which were formally studied in relation to kernel methods (Rahimi & Recht, 2007) .

In the case of ReLU activation functions, Sun et al. (2019) proved an analog of Theorem 3.3 for the approximation of functions in a reproducing kernel Hilbert space; cf. (Rahimi & Recht, 2008) .

For a good discussion on the role of random features in the representability of neural networks, see (Yehudai & Shamir, 2019) .

The UAP is also studied in other contexts, most notably in relation to the depth and width of neural networks.

Lu et al. (2017) proved the UAP for neural networks with hidden layers of bounded width, under the assumption that ReLU is used as the activation function.

Soon after, Hanin (2017) strengthened the bounded-width UAP result by considering the approximation of continuous convex functions.

Recently, the role of depth in the expressive power of neural networks has gathered much interest (Delalleau & Bengio, 2011; Eldan & Shamir, 2016; Mhaskar et al., 2017; Montúfar et al., 2014; Telgarsky, 2016) .

We do not address depth in this paper, but we believe it is possible that our results can be applied iteratively to deeper neural networks, perhaps in particular for the approximation of compositional functions; cf. (Mhaskar et al., 2017) .

We begin with a "warm-up" result.

Subsequent results, even if they seem complicated, are actually multivariate extensions of this "warm-up" result, using very similar ideas.

Theorem 5.1.

Let p(x) be a real polynomial of degree d with non-zero coefficients, and let a 1 , . . .

, a d+1 be real numbers.

For each

Then f 1 , . . . , f d+1 are linearly independent if and only if a 1 , . . .

, a d+1 are distinct.

) be the i-th derivative of f j (resp.

p), and let α

is defined to be the determinant of the matrix M (x) := [f

.

Since f 1 , . . . , f d+1 are polynomial functions, it follows that (f 1 , . . .

, f d+1 ) is a sequence of linearly independent functions if and only if its Wronskian is not the zero function (LeVeque, 1956, Thm.

4.7(a) ).

Clearly, if a i = a j for i = j, then det M (x) is identically zero.

Thus, it suffices to show that if a 1 , . . .

, a d+1 are distinct, then the evaluation det M (1) of this Wronskian at x = 1 gives a non-zero value.

Now, the (i, j)-th entry of M (1) equals a

Note that the k-th diagonal entry of M is α

n , and any function g : R → R, let F g,x0 (W ) denote the sequence of functions (f 1 , . . .

, f N ), such that each f j : R n → R is defined by the map

Note that the value of m is irrelevant for defining g W ind n,N ;x0 .

Remark 5.3.

Given a = (a 1 , . . . , a n ) ∈ R n , consider the ring automorphism ϕ :

It is easy to show that |Λ n ≤d | = n+d d , and that the set M n ≤d of monomial functions forms a basis for P ≤d (R n ).

Sort the n-tuples in Λ n ≤d in colexicographic order, i.e. (α 1 , . . . , α n ) < (α 1 , . . . , α n ) if and only if α i < α i for the largest index i such that α i = α i .

Let λ 1 < · · · < λ (

Definition 5.5.

Given any W ∈ W n,m ).

Theorem 5.6.

Let m be arbitrary, let p ∈ P d (R n ), and suppose that p has all-non-zero coefficients.

Also, suppose that p 1 , . . .

, p k ∈ P(W n,m n+d d

) are fixed polynomial functions on the non-bias weights of the first layer.

Define the following sets:

If there exists W ∈ U such that the non-bias Vandermonde matrix of W is non-singular, then p U ind is dense in U (w.r.t.

the Euclidean metric).

Proof.

We essentially extend the ideas in the proofs of Theorem 5.1 and Corollary 5.4, using the notion of generalized Wronskians; see Appendix A for details.

βi .

Such matrices are well-studied in algebraic combinatorics, and the determinant of Q[W ] is a Schur polynomial; see (Stanley, 1999) .

In particular, if we choose positive pairwise distinct values for w

is non-singular, since a Schur polynomial can be expressed as a (non-negative) sum of certain monomials; see (Stanley, 1999, Sec. 7 .10) for details.

i,j | < λ for some fixed constant λ > 0.

Lemma 5.9.

Let d ≥ 1 be an integer, and suppose σ ∈ C(R).

For every r > 0, let Y r ⊆ R be a closed interval of length r, such that Y r ⊆

Y r whenever r ≤ r , and let σ r be the best polynomial approximant to σ| Yr (i.e. σ restricted to domain Y r ) of degree d. Then

Recall that any modulus of continuity ω f is subadditive (i.e. f (x + y) ≤ f (x) + f (y) for all x, y); see (Rivlin, 1981, Chap.

1 Outline of strategy for proving Theorem 3.1.

The first crucial insight is that P ≤d (R n ), as a real vector space, has dimension hidden units.

Every hidden unit represents a continuous function g j : X → R determined by its weights W and the activation function σ .

If g 1 , . . .

, g N can be well-approximated (on X) by linearly independent polynomial functions in P ≤d (R n ), then we can choose suitable linear combinations of these N functions to approximate all coordinate functions f [t] (independent of how large m is).

To approximate each g j , we consider a suitable sequence {σ λ k } ∞ k=1 of degree d polynomial approximations to σ, so that g j is approximated by a sequence of degree d polynomial functions { g

.

We shall also vary W concurrently with k, so that w The second crucial insight is that every function in P ≤d (R n ) can be identified geometrically as a point in Euclidean This last observation, in particular, when combined with Lemma 5.9, is a key reason why the minimum number N of hidden units required for the UAP to hold is independent of the approximation error threshold ε.

Proof of Theorem 3.1.

Fix some ε > 0, and for brevity, let N = n+d d .

The case that f is a constant function is trivial, so assume f is non-constant.

Fix a point x 0 ∈ X, and define f 0 ∈ C(X, R m ) by f

For every r ≥ 0, let Y r ⊆ R be a closed interval of length r, such that σ| Yr is not identically zero; Y r is well-defined, since σ ∈ P 0 (R) by assumption.

Without loss of generality, assume that Y r ⊆

Y r whenever r ≤ r. Also, let σ r be the best polynomial approximant to σ| Yr of degree d. Note that for any W ∈ W n,m N , we could choose σ ∈ C(X) to be arbitrarily close to σ in the uniform metric, such that ρ σ W − ρ σ W ∞,X is arbitrarily small.

Since σ ∈ C(R)\P ≤d−1 (R) by assumption, we may perturb σ if necessary, and assume without loss of generality that σ r ∈ P d (Y r ), and that σ r has all-non-zero coefficients.

For every h ∈ P ≤d (R n ), let ν(h) ∈ R N be the vector of coefficients with respect to the basis

, and let ν(h) ∈ R N −1 be the truncation of ν(h) by removing the first coordinate.

Note that q 1 (x) is the constant monomial, so this first coordinate is the coefficient of the constant term.

Define

In particular, c f = 0, since f is non-constant.

Next, define C := 8N (N − 1)c f > 0.

For every r > 0, let W r := {W ∈ W n,m N : w

(1) j 2 < r for all 1 ≤ j ≤ N }.

Let {λ k } k∈N be a divergent strictly increasing sequence of positive real numbers, such that

This sequence exists, since lim r→∞ σr−σ ∞,Yr r = 0 by Lemma 5.9.

For each k ∈ N, let y k be any element of Y λ k such that σ(y k ) = σ λ k (y k ); clearly y k exists, since if instead σ λ k (y) = σ(y) for all y ∈ Y λ k , then σ λ k cannot possibly be the best polynomial approximant to σ| Y λ k .

By perturbing {λ k } k∈N if necessary, we shall assume without loss of generality that σ(λ k ) = 0 for all k ∈ N. Also, for every k ∈ N, define

Each λ k is well-defined, since the compactness of X implies that x − x 0 2 is bounded.

Let r X (x 0 ) := sup{ x − x 0 2 : x ∈ X}, and note that

, we do not impose any restrictions on the bias weights.

Thus, given any such W , we could choose the bias weights of W

(1) to be w

(1)

By Corollary 5.7 and Remark 5.3,

, so such a W exists (with its bias weights given as above); we can furthermore choose W so that

k } is linearly independent and hence spans P ≤d (X).

Thus, for every 1 ≤ t ≤ m, there exist a

N,k ∈ R, which are uniquely determined once k is fixed, such that f

N,k g W N,k .

Evaluating both sides of this equation at x = x 0 , we then get

For each ∈ R, define the hyperplane H := {(u 1 , . . . , u N ) ∈ R N : u 1 = }.

Recall that q 1 (x) is the constant monomial, so the first coordinate of each ν( g

Observe that λ k → ∞ (and hence λ k → ∞) as k → ∞. Thus, when k is sufficiently large, we can

desired large radius r; we used here the fact that W could be chosen so that

In particular, for sufficiently large k, we can choose W so that this convex hull contains both points ν(f [t] ) and 0 N −1 .

This implies that for each 1 ≤ t ≤ m, we have

N,k ) and (b 1,k , . . .

, b N,k ) are barycentric coordinate vectors of the points ν(f [t] ) and 0 N −1 respectively, w.r.t.

would approach the barycenter with barycentric coordinate vector ( 1 N , . . .

, 1 N ).

Since λ k → ∞ as k → ∞, we infer that for every δ > 0, there exists some sufficiently large k, such that b

Since a

N,k are unique (for fixed k), we infer that a

Now, δ is an upper bound of the normalized distance of each barycentric coordinate of the two points ν(f [t] ) and

for any r > 0 such that B r is contained in the convex hull of { ν( g (Klamkin & Tsintsifas, 1979) .

Since

, so it follows from (4) and (6) that for sufficiently large k,

Now, for this sufficiently large k, define g : (3) and (7), it follows that

∞,X = a

j,k for each 1 ≤ j ≤ N , and let w

Notice that for all δ > 0, we showed in (6) that there is a sufficiently large k such that a

j,k ≤ 2δ.

Thus, for all λ > 0, we can choose W so that all non-bias weights in W (2) are contained in the interval (−λ, λ); this proves assertion (i) of the theorem.

Note also that we do not actually require δ > 0 to be arbitrarily small.

Suppose instead that we choose k ∈ N sufficiently large, so that the convex hull of { ν( g W 1,k ), . . .

, ν( g W N,k )} contains both points ν(f [t] ) and 0 N −1 .

In this case, observe that our choice of k depends only on f (via ν(f

and σ (via the definition of {λ k } k∈N ).

The inequality δ ≤

Consequently, our argument to show that f − ρ σ W ∞,X < ε holds verbatim, which proves assertion (ii) of the theorem.

Proof of Theorem 3.2.

Fix some ε > 0, and consider an arbitrary t ∈ {1, . . .

, m}. For each integer

d be the best polynomial approximant to f [t] of degree d. By Theorem 2.2, we have

Theorem 5.6 is rather general, and could potentially be used to prove analogs of the universal approximation theorem for other classes of neural networks, such as convolutional neural networks and recurrent neural networks.

In particular, finding a single suitable set of weights (as a representative of the infinitely many possible sets of weights in the given class of neural networks), with the property that its corresponding "non-bias Vandermonde matrix" (see Definition 5.5) is non-singular, would serve as a straightforward criterion for showing that the UAP holds for the given class of neural networks (with certain weight constraints).

We formulated this criterion to be as general as we could, with the hope that it would applicable to future classes of "neural-like" networks.

We believe our algebraic approach could be emulated to eventually yield a unified understanding of how depth, width, constraints on weights, and other architectural choices, would influence the approximation capabilities of arbitrary neural networks.

Finally, we end our paper with an open-ended question.

The proofs of our results in Section 5 seem to suggest that non-bias weights and bias weights play very different roles.

We could impose very strong restrictions on the non-bias weights and still have the UAP.

What about the bias weights?

First, we recall the notion of generalized Wronskians as given in (LeVeque, 1956, Chap.

4.3) .

Let ∆ 0 , . . .

, ∆ N −1 be any N differential operators of the form and let x = (x 1 , . . .

, x n ).

Recall that λ 1 < · · · < λ N are all the n-tuples in Λ n ≤d in the colexicographic order.

For each

be the coefficient of the monomial q k (x) in ∆ λi p(x).

Consider an arbitrary W ∈ U, and for each 1 ≤ j ≤ N , define f j ∈ P ≤d (R n ) by the map x →

p(w

1,j x 1 , . . .

, w

n,j x n ).

Note that F p,0n (W ) = (f 1 , . . .

, f N ) by definition.

Next, define the matrix M W (x) := [∆ i f j (x)] 1≤i,j≤N , and note that det M W (x) is the generalized Wronskian of (f 1 , . . .

, f N ) associated to ∆ 1 , . . .

, ∆ N .

In particular, this generalized Wronskian is well-defined, since the definition of the colexicographic order implies that λ k,1 + · · · + λ k,n ≤ k for all possible k. Similar to the univariate case, (f 1 , . . .

, f N ) is linearly independent if (and only if) its generalized Wronskian is not the zero function (Wolsson, 1989) .

Thus, to show that W ∈ p U ind , it suffices to show that the evaluation det M W (1 n ) of this generalized Wronskian at x = 1 n gives a non-zero value, where 1 n denotes the all-ones vector in R n .

Observe that the (i, j)-th entry of M W (1 n ) equals ( w It follows from the definition of the colexicographic order that λ j − λ i necessarily contains at least one strictly negative entry whenever j < i, hence we infer that M is upper triangular.

The diagonal entries of M are α

0n , . . .

, α

0n , and note that α

λi for each 1 ≤ i ≤ N , where λ i,1 !

·

·

· λ i,n ! denotes the product of the factorials of the entries of the n-tuple λ i .

In particular, λ i,1 !

·

·

· λ i,n ! = 0, and α (1) λi , which is the coefficient of the monomial q i (x) in p(x), is non-zero.

Thus, det(M ) = 0.

We have come to the crucial step of our proof.

If we can show that det(M ) = det(Q[W ]) = 0, then det(M W (1 n )) = det(M ) det(M ) = 0, and hence we can infer that W ∈ p U ind .

This means that p U ind contains the subset U ⊆ U consisting of all W such that Q[W ] is non-singular.

Note that det(Q [W ] ) is a polynomial in terms of the non-bias weights in W

(1) as its variables, so we could write this polynomial as r = r(W ).

Consequently, if we can find a single W ∈ U such that Q[W ] is non-singular, then r(W ) is not identically zero on U, which then implies that U = {W ∈ U : r(W ) = 0} is dense in U (w.r.t.

the Euclidean metric).

It was conjectured by Mhaskar (1996) that there exists some smooth non-polynomial function σ, such that at least Ω(ε −n ) hidden units is required to uniformly approximate every function in the class S of C 1 functions with bounded Sobolev norm.

As evidence that this conjecture is true, a heuristic argument was provided in (Mhaskar, 1996) , which uses a result by DeVore et al. (1989) ; cf. (Pinkus, 1999, Thm. 6.5) .

To the best of our knowledge, this conjecture remains open.

If this conjecture is indeed true, then our upper bound O(ε −n ) in Theorem 3.2 is optimal for general continuous non-polynomial activation functions.

For specific activation functions, such as the logistic sigmoid function, or any polynomial spline function of fixed degree with finitely many knots (e.g. the ReLU function), it is known that the minimum number N of hidden units required to uniformly approximate every function in S must satisfy (N log N ) ∈ Ω(ε −n ) (Maiorov & Meir, 2000) ; cf. (Pinkus, 1999, Thm. 6.7) .

Hence there is still a gap between the lower and upper bounds for N in these specific cases.

It would be interesting to find optimal bounds for these cases.

@highlight

A quantitative refinement of the universal approximation theorem via an algebraic approach.

@highlight

The authors derive the universal approximation property proofs algebraically and assert that the results are general to other kinds of neural networks and similar learners.

@highlight

A new proof of Leshno's version of the universal approximation property for neural networks, and new insights into the universal approximation property.