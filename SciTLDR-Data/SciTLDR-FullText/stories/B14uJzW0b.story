Deep learning models can be efficiently optimized via stochastic gradient descent, but there is little theoretical evidence to support this.

A key question in optimization is to understand when the optimization landscape of a neural network is amenable to gradient-based optimization.

We focus on a simple neural network two-layer ReLU network with two hidden units, and show that all local minimizers are global.

This combined with recent work of Lee et al. (2017); Lee et al. (2016) show that  gradient descent converges to the global minimizer.

For the duration of this paper, we will assume that x is standard normal in R n and all expectations are with respect to the standard normal.

The population loss function is: DISPLAYFORM0 Define DISPLAYFORM1 so the loss can be rewritten as (ignoring additive constants, then multiplied by 4): DISPLAYFORM2 g(w i , w j ) − 2g(w i , w * j ) .From BID0 we get DISPLAYFORM3 and DISPLAYFORM4 In this paper, we study the landscape of f over the manifold R = { w 1 = w 2 = 1}. The manifold gradient descent algorithm is: DISPLAYFORM5 where P R is the orthogonal projector onto the manifold R, and ∇ R is the manifold gradient of f .

In order to analyze the global convergence of manifold gradient descent, we need a characterization of all critical points.

We show that f (W ) have no spurious local minimizer on the manifold R. Theorem 4.1.

Assume wThe next theorem shows that manifold gradient descent with random initialization converges to the global minimizer Theorem 4.2.

With probability one, manifold gradient descent will converge to the global minimizers.

Proof.

The objective function f is infinitely differentiable on manifold R. Using Corollary 6 of BID16 , manifold gradient descent will converge to a local minimizer with probability one.

Since the only local minima for function f are w 1 = wThe second observation is we only need to compute the gradient on the manifold and check whether it's zero.

Define m(w 1 ) = sin θ 1 ∂f ∂w11 − cos θ 1 ∂f ∂w12 and m(w 2 ) = sin θ 2 ∂f ∂w21 − cos θ 2 ∂f ∂w22 .

Then for w 1 and w 2 , the norm of the manifold gradients are |m(w 1 )| and |m(w 2 )|.

Thus, we only need to check whether the value of function m is 0 and get rid of the absolute value sign.

Then we apply the polar coordinates onto the manifold gradients, and obtain: m(w 2 ) = 1 π (π − θ w1,w2 ) sin(θ 2 − θ 1 ) + cos θ 2 − sin θ 2 (6) + 1 π θ w2,w * 1 sin θ 2 − θ w2,w * 2 cos θ 2 .The last observation we need for this theorem is that we must divide this problem into several cases because each angle in (300) is a piecewise linear function.

If we discuss each case independently, the resulting functions are linear in the angles.

The details are in Appendix B. After the calculation of all cases, we found the positions of all the critical points: WLOG assume θ 1 ≤ θ 2 , then there are four critical points in the 2D case: (θ 1 , θ 2 ) = (0, ).

After finding all the critical points, we compute the manifold Hessian matrix for those points and show that there is a direction of negative curvature.

The details can be found in Appendix C. The next step is to reduce it to a three dimensional problem.

As stated in the two-dimensional case, the gradient is in span{w 1 , w 2 , w * 1 , w * 2 }, which is four-dimensional.

However, using the following lemma, we can reduce it to three dimensions and simplify the whole problem.

Lemma 4.4.

If (w 1 , w 2 ) is a critical point, then there exists a set of standard orthogonal basis (e 1 , e 2 , e 3 ) such that e 1 = w Even if we simplify the problem into three dimensional case, it still seems to be impossible to identify all critical points explicitly.

Our method to analyze the landscape of the loss surface is to find the properties of critical points and then show all saddle points and local maximizers have a direction of negative curvature.

The following two lemmas captures the main geometrical properties of the critical points in three dimensional case.

More detailed properties are given is Section 5.2 Lemma 4.5.arccos(−w 11 ) arccos(−w 21 ) = arccos(−w 12 ) arccos(−w 22 ) = − w 23 w 13 .The ratio in Lemma 4.5 captures an important property of all critical points.

For simplicity, based on D.5, we define DISPLAYFORM0 .Then from the properties of θ 1 , θ 2 and upper bound the value of k 0 we get Lemma 4.6.

θ 1 = θ 2 .That lemma shows that w 1 and w 2 must be on a plane whose projection onto span{w * 1 , w * 2 } is the bisector of w * 1 and w * 2 .

Combining this with the computation of Hessian, we conclude that we have found negative curvature for all possible critical points, which leads to the following proposition.

Here we provide some detailed proofs which are important for the understanding of the main theorem.

In general case, the following lemma shows we only need three dimension.

Lemma 5.1.

If (w 1 , w 2 ) is a critical point, then there exists a set of standard orthogonal basis (e 1 , e 2 , e 3 ) such that e 1 = w * 1 , e 2 = w * 2 and w 1 , w 2 lies in span{e 1 , e 2 , e 3 }.Proof.

If (w 1 , w 2 ) is a critical point, then DISPLAYFORM0 where matrix (I − w 1 w T 1 ) projects a vector onto the tangent space of w 1 .

Since DISPLAYFORM1 we get DISPLAYFORM2 which means that DISPLAYFORM3 )w * 2 lies in the direction of w 1 .

If θ w1,w2 = π, i.e., w 1 = −w 2 , then of course the four vectors have rank at most 3, so we can find the proper basis.

If θ w1,w2 < π, then we know that there exists a real number r such that DISPLAYFORM4 Since θ w1,w2 < π, we know that the four vectors w 1 , w 2 , w * 1 and w * 2 are linear dependent.

Thus, they have rank at most 3 and we can find the proper basis.

Next we will focus on the properties of critical points.

Assume (w 1 , w 2 ) is one of the critical points, from lemma D.1 we can find a set of standard orthogonal basis (e 1 , e 2 , e 3 ) such that e 1 = w * 1 , e 2 = w * 2 and w 1 , w 2 lies in span{e 1 , e 2 , e 3 }.

Furthermore, assume w 1 = w 11 e 1 + w 12 e 2 + w 13 e 3 and w 2 = w 21 e 1 + w 22 e 2 + w 23 e 3 , i.e., w 1 = (w 11 , w 12 , w 13 ) and w 2 = (w 21 , w 22 , w 23 ).

Since we have already found out all the critical points when w 13 = w 23 = 0, in the following we assume w Proof.

Adapting from the proof of lemma D.4 and we know that DISPLAYFORM0 Similarly, we have DISPLAYFORM1 Taking the first component of FORMULA0 and FORMULA0 gives us DISPLAYFORM2 DISPLAYFORM3 Thus, DISPLAYFORM4 Similarly, we get DISPLAYFORM5 Since ∀i, j ∈ [2], π − θ wi,w * j = arccos(−θ wij ), we know that DISPLAYFORM6 Using this equation, we obtain several properties of critical points.

The following two lemmas show the basic properties of critical points in three dimensional case.

Completed proofs are given in Appendix B and C. Lemma 5.3.

θ w1,w2 < π.

Lemma 5.4.

w 13 * w 23 < 0.These two lemmas restrict the position of critical points in some specific domains.

Then we construct a new function F in order to get more precise analysis.

Define DISPLAYFORM7 From the properties of that particular function and upper bound the value of k 0 we get Lemma 5.5.

θ 1 = θ 2 .That lemma shows that w 1 and w 2 must be on a plane whose projection onto span{w * 1 , w * 2 } is the bisector of w * 1 and w * 2 .

Although we cannot identify the critical points explicitly, we will show these geometric properties already capture the direction of negative curvature.

In this section, we partially characterize the structure of the critical points when w * 1 , w * 2 are nonorthogonal, but form an acute angle.

In other words, the angle between w * 1 and w * 2 is α ∈ (0, π 2 ).

Let us first consider the 2D cases, i.e., both w 1 and w 2 are in the span of w * 1 and w * 2 .

Similar to the original problem, after the technique of changing variables(i.e., using polar coordinates and assume θ 1 and θ 2 are the angles of w 1 and w 2 in polar coordinates), we divide the whole plane into 4 parts, which are the angle in [0, α], [α, π] , [π, π + α] and [π + α, 2π).

We have the following lemma: Lemma 6.1.

Assume w * 1 = w * 2 = 1, w * T 1 w * 2 > 0 and w 1 , w 2 ∈ span{w * 1 , w * 2 }.

When w 1 and w 2 are in the same part(one of four parts), the only critical points except the global minima are those when both w 1 and w 2 are on the bisector of w * 1 and w * 2 .Proof.

The complete proof is given in appendix E, the techniques are nearly the same as things in the original problem and a bit harder, so to be brief, we omit the proof details here.

For the three-dimensional cases cases of this new problem, it's interesting that the first few lemmatas are still true.

Specifically, Lemma D.1(restated as Lemma 4.4) to Lemma D.5(restated as Lemma 4.5) are still correct.

The proof is very similar to the proofs of those lemmas, except we need modification to the coefficients of terms in the expressions of the manifold gradients.

We did experiments to verify the theoretical results.

Since our results are restricted to the case of K = 2 hidden units, it is also natural to investigate whether general two-layer ReLU networks also have the property that all local minima are global minima.

Unfortunately as we show via numerical simulation, this is not the case.

We consider the cases of K from 2 to 11 hidden units and we set the dimension d = K. For each K, the true parameters are orthogonal to each other.

For each K, we run projected gradient descent with 300 different random initializations, and count the number of local minimum (critical points where the manifold Hessian is positive definite) with non-zero training error.

If we reach a sub-optimal local minimum, we can conclude the loss surface exhibits spurious local minima.

The bar plot showing the number of times gradient descent converged to spurious local minima is in FIG2 .

From the plot, we see there is no spurious local minima from K = 2 to K = 6.

However for K ≥ 7, we observe a clear trend that there are more spurious local minima when there are more hidden units.

In this paper, we provided recovery guarantee of stochastic gradient descent with random initialization for learning a two-layer neural network with two hidden nodes, unit-norm weights, ReLU activation functions and Gaussian inputs.

Experiments are also done to verify our results.

For future work, here we list some possible directions.

This paper focused on a ReLU network with only two hidden units, .

And the teaching weights must be orthogonal.

Those are many conditions, in which we think there are some conditions that are not quite essential, e.g., the orthogonal assumption.

In experiments we have already seen that even if they are not orthogonal, it still has some good properties such as the positions of critical points.

Therefore, in the future we can further relax or abandon some of the assumptions of this paper and preserve or improve the result we have.

The neural network we discussed in this paper is in some sense very simple and far from practice, although it is already the most complex model when we want to analyze the whole loss surface.

By experiments we have found that when it comes to seven hidden nodes with orthogonal true parameters, there will be some bad local minima, i.e., there are some local minima that are not global.

We believe that research in this paper can capture the characteristics of the whole loss surface and can help analyze the loss surface when there are three or even more hidden units, which may give some bounds on the performance of bad local minima and help us understand the specific non-convexity of loss surfaces.

Consider a neural network with 2 hidden nodes and ReLU as the activation function: DISPLAYFORM0 where σ(x) = max(0, x) is the ReLU function.

First we study the 2-D case, i.e., the input and all parameters are two dimensional.

Assume that the input follows standard normal distribution.

The loss function is population loss: DISPLAYFORM1 Define DISPLAYFORM2 then from BID0 we get DISPLAYFORM3 Thus, DISPLAYFORM4 Moreover, from FORMULA1 we get DISPLAYFORM5 Assume w * 1 = w * 2 and w * T 1 w * 2 = 0.

WLOG, let e 1 = w * 1 and e 2 = w * 2 .

Then we know that ∀i, j ∈ [2], g(w * i , w * j ) is a constant number.

Thus, define the objective function(which equals to 4l(W ) up to an additive constant) DISPLAYFORM6 Thus, DISPLAYFORM7 Similarly, for w 2 , the gradient is ∂f DISPLAYFORM8 Assume that w 1 = (w 11 , w 12 ) and w 2 = (w 21 , w 22 ), then the gradient can be expressed in this form: DISPLAYFORM9 and DISPLAYFORM10 Because of symmetry, for w 2 , the gradient is DISPLAYFORM11 and DISPLAYFORM12 B CRITICAL POINTS IN 2D CASES

In 2D cases, we can translate W to polar coordinates and fix w 1 = w 2 = 1, so there are two variables left: θ 1 and θ 2 , i.e., w 1 = (cos θ 1 , sin θ 1 ) and w 2 = (cos θ 2 , sin θ 2 ).For manifold gradient, we only need to consider its norm and check whether it's zero.

Under review as a conference paper at ICLR 2018To make life easier, it's better to simplify the m functions a bit using w 1 = (cos θ 1 , sin θ 1 ) and w 2 = (cos θ 2 , sin θ 2 ): DISPLAYFORM0 Similarly, DISPLAYFORM1 Then we can divide them into several cases and analyze them one by one to specify the positions and properties of the critical points.

DISPLAYFORM2 The norm of the manifold gradient w.r.t.

w 1 is DISPLAYFORM3 Similarly, the norm of m(w 2 ) is DISPLAYFORM4 Define DISPLAYFORM5 If m(w 1 ) = m(w 2 ) = 0, then DISPLAYFORM6 and DISPLAYFORM7 Thus, DISPLAYFORM8 Also note that FORMULA3 and we get DISPLAYFORM9 DISPLAYFORM10 and the inequality becomes equality only then DISPLAYFORM11 ≥ 0.(88) Note that (84) is because cos(2θ) − cos θ is always non-positive when 0 ≤ θ ≤ DISPLAYFORM12 In a word, there are two critical points in this case: DISPLAYFORM13 The norm of the manifold gradient w.r.t.

w 1 is DISPLAYFORM14 Similarly, DISPLAYFORM15 Define DISPLAYFORM16 DISPLAYFORM17 and the inequality becomes equality only then θ = π 2 or θ = π.

DISPLAYFORM18 Note that the inequality becomes equality only when θ cos θ = 0 and DISPLAYFORM19 and DISPLAYFORM20 Thus, DISPLAYFORM21 However, we know that h 2 (θ 1 ) < 0 and h 2 (θ 1 ) < 0, which makes a contradiction.

In a word, there is no critical point in this case.

DISPLAYFORM22 The norm of the manifold gradient w.r.t.

w 1 is DISPLAYFORM23 Similarly, the norm of m(w 2 ) is DISPLAYFORM24 Define DISPLAYFORM25 Let θ = θ + π, then DISPLAYFORM26 DISPLAYFORM27 Moreover, ∀θ ∈ [π, DISPLAYFORM28 DISPLAYFORM29 DISPLAYFORM30 = 0.Also, when θ ∈ [π, DISPLAYFORM31 so h 3 is an increasing function when θ ∈ [π, From Lemma B.1, DISPLAYFORM32 DISPLAYFORM33 DISPLAYFORM34 DISPLAYFORM35 Note that (132) becomes equality only when θ 1 = π or θ 1 = DISPLAYFORM36 Actually, this is symmetric to the B.3, so in this part I would like to specify this kind of symmetry.

We have already assumed that θ 1 ≤ θ 2 without loss of generality, and under this assumption, we can find another symmetry: From w 1 and w 2 , using line y = x as symmetry axis, we can get two new vectors w 1 and w 2 .

w 1 is not necessarily the image of w 1 because we need to preserve the assumption that θ 1 ≤ θ 2 , but there exists one and only one mapping such that θ 1 ≤ θ 2 .

In this kind of symmetry, the angles, including θ w1,w2 and θ wi,w * j where i, j ∈ [2], are the same, so the two symmetric cases share the same gradients, thus the symmetric critical points.

We use (i, j) ,where i, j ∈ [4], to represent the case that θ 1 is in the ith quadrant and θ 2 is in the jth one.

Using this kind of symmetry, we conclude that (1, 2) is equivalent to (1, 4) and (2, 3) is equivalent to (3, 4), so there are 4 cases left which are (1, 2), (1, 3), (2, 3) and (2, 4).

DISPLAYFORM37 Similar to previous cases, DISPLAYFORM38 and m(w 2 ) = 1 π (π − θ 2 + θ 1 ) sin(θ 2 − θ 1 ) + cos θ 2 − sin θ 2 (137) DISPLAYFORM39 Using previous definitions, we conclude that DISPLAYFORM40 DISPLAYFORM41 If m(w 1 ) = m(w 2 ) = 0, then m ( w 1 ) + m(w 2 ) = 0, i.e., DISPLAYFORM42 From (99) we know that DISPLAYFORM43 Thus, using lemma B.2, DISPLAYFORM44 That means the only case that h 1 (θ 1 ) + h 2 (θ 2 ) = 0 is when the inequality (143) becomes equality, which means that cos θ 1 = 1 and h 2 (θ 1 + π 2 ) = h 2 (θ 2 ) = − 1 2 .

Thus, we must have θ 1 = 0, and θ 2 = π 2 or θ 2 = π.

Plugging them back in FORMULA0 and FORMULA0 , we can verify that the first one is a critical point while the other is not.

Since (θ 1 , θ 2 ) = (0, π 2 ) has been counted in case 1, there are no new critical points in this case.

DISPLAYFORM45 Similar to previous cases, DISPLAYFORM46 and m(w 2 ) = 1 π (π − θ w1,w2 ) sin(θ 2 − θ 1 ) + cos θ 2 − sin θ 2 (146) DISPLAYFORM47 Thus, using previous definitions DISPLAYFORM48 and DISPLAYFORM49 If m(w 1 ) = m(w 2 ) = 0, then m(w 1 ) + m(w 2 ) = 0, i.e., DISPLAYFORM50 DISPLAYFORM51 Then we have the following lemma: Lemma B.3.

When 0 ≤ θ ≤ Proof.

From (123), h 3 (θ + π) = h 1 (θ) − cos θ + sin θ.

Thus, H(θ) = 2h 1 (θ) − cos θ + sin θ (152) DISPLAYFORM52 When 0 ≤ θ ≤ π 4 , since sin θ is a concave function for θ, we know that DISPLAYFORM53 Thus, DISPLAYFORM54 To make H(θ) = 0, we must have DISPLAYFORM55 Thus, DISPLAYFORM56 And to make H(θ) = 0, the only possibility is θ = π 2 , which ends the proof.

Remember that if m(w 1 ) = m(w 2 ) = 0, then we have h 3 (θ 2 ) = −h 1 (θ 1 ).If h 1 (θ 1 ) > 0, i.e., 0 ≤ θ 1 < π 4 , then from lemma B.3, H(θ 1 ) ≤ 0, which means that DISPLAYFORM57 Since h 3 is a strictly increasing function, we know that if h 3 (θ 2 ) = −h 1 (θ 1 ), then θ 2 ≥ θ 1 + π, so sin(θ 1 − θ 2 ) ≥ 0, and that means DISPLAYFORM58 Similarly, if h 1 (θ 1 ) < 0, i.e., DISPLAYFORM59 Thus, if h 3 (θ 2 ) = −h 1 (θ 1 ), then θ 2 ≤ θ 1 + π, so sin(θ 1 − θ 2 ) ≤ 0, and that means DISPLAYFORM60 The last possibility is h 1 (θ 1 ) = 0, i.e., θ 1 = π 4 .

Plugging it into (150) and we know that h 3 (θ 2 ) = 0, so θ 2 = 5π 4 .

And that is indeed a critical point.

In a word, the only critical point in this case is (θ 1 , θ 2 ) = ( DISPLAYFORM61 Like previous cases, DISPLAYFORM62 If m(w 1 ) = m(w 2 ) = 0, then m ( w 1 ) + m(w 2 ) = 0, i.e., DISPLAYFORM63 Let θ = θ 2 − π, then from (99) and FORMULA0 , we know that DISPLAYFORM64 Thus, from lemma B.2, DISPLAYFORM65 Therefore, in order to achieve h 2 (θ 1 ) + h 3 (θ 2 ) = 0, the only way is let (176) becomes equality, which means that θ 2 = 3π 2 and θ 1 = π 2 or π.

Plugging them into FORMULA0 and FORMULA0 we conclude that both of them are not critical points.

In a word, there is no critical point in this case.

DISPLAYFORM66 Similar to previous cases, DISPLAYFORM67 and m(w 2 ) = 1 π (π − θ w1,w2 ) sin(θ 2 − θ 1 ) + cos θ 2 − sin θ 2 (179) DISPLAYFORM68 2 , we must have θ w1,w2 = π 2 , so it must be true that DISPLAYFORM69 Therefore, using lemma B.2, DISPLAYFORM70 In a word, there is no critical point in this case.

In conclusion, based on the assumption that θ 1 ≤ θ 2 there are four critical points in the 2D case: Assume the manifold is R = {(w 1 , w 2 ) : w 1 2 = w 2 2 = 1}, then the Hessian on the manifold is DISPLAYFORM0 DISPLAYFORM1 where z = (z 1 , z 2 ) satisfies w DISPLAYFORM2 and DISPLAYFORM3 Then we can get when w 1 = w 2 and w 1 = −w 2 , DISPLAYFORM4 So this point is a saddle point.

In conclusion, we have four critical points: one is global maximal, the other three are saddle points.

DISPLAYFORM0 ) is a critical point, then there exists a set of standard orthogonal basis (e 1 , e 2 , e 3 ) such that e 1 = w * 1 , e 2 = w * 2 and w 1 , w 2 lies in span{e 1 , e 2 , e 3 }.Proof.

If (w 1 , w 2 ) is a critical point, then DISPLAYFORM1 where matrix (I − w 1 w T 1 ) projects a vector onto the tangent space of w 1 .

Since DISPLAYFORM2 we get DISPLAYFORM3 DISPLAYFORM4 )w * 2 lies in the direction of w 1 .

If θ w1,w2 = π, i.e., w 1 = −w 2 , then of course the four vectors have rank at most 3, so we can find the proper basis.

If θ w1,w2 < π, then we know that there exists a real number r such that DISPLAYFORM5 Since θ w1,w2 < π, we know that the four vectors w 1 , w 2 , w * 1 and w * 2 are linear dependent.

Thus, they have rank at most 3 and we can find the proper basis.

Next we will focus on the properties of critical points.

Assume (w 1 , w 2 ) is one of the critical points, from lemma D.1 we can find a set of standard orthogonal basis (e 1 , e 2 , e 3 ) such that e 1 = w * 1 , e 2 = w * 2 and w 1 , w 2 lies in span{e 1 , e 2 , e 3 }.

Furthermore, assume w 1 = w 11 e 1 + w 12 e 2 + w 13 e 3 and w 2 = w 21 e 1 + w 22 e 2 + w 23 e 3 , i.e., w 1 = (w 11 , w 12 , w 13 ) and w 2 = (w 21 , w 22 , w 23 ).

Since we have already found out all the critical points when w 13 = w 23 = 0, in the following we assume w Proof.

If θ w1,w2 = π, then w 1 = −w 2 , so w 2 is in the direction of w 1 .

We have already known from (196) DISPLAYFORM0 )w * 2 lies in span{e 1 , e 2 }, so w 1 ∈ span{e 1 , e 2 } and w 2 ∈ span{e 1 , e 2 }.

Thus, w 13 = w 23 = 0 and that contradicts with the assumption.

In a word, θ w1,w2 < π.

Lemma D.3.

w 13 * w 23 = 0.Proof.

We have already known from (196) DISPLAYFORM1 lies in the direction of w 1 .

Writing it in each dimension and we know that there exists a real number r 0 such that DISPLAYFORM2 From lemma D.2 we know that θ w1,w2 < π, so we can define DISPLAYFORM3 .Then the equations become DISPLAYFORM4 Similarly, we have DISPLAYFORM5 Since w 2 13 + w 2 23 = 0, at least one of those two variables cannot be 0.

WLOG, we assume that w 13 = 0.

If w 23 = 0, then from (207) we know that w 13 = 0, which contradicts the assumption.

Thus, w 23 = 0, which means that w 13 * w 23 = 0.Lemma D.4.

w 13 * w 23 < 0.Proof.

Adapting from the proof of lemma D.3, we know that DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 and DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 Furthermore, kk = From lemma D.2 we know that θ w1,w2 < π, and from lemma D.3 we know that both w 1 and w 2 are DISPLAYFORM12 > 0.

Therefore, we have DISPLAYFORM13 That means k < 0, so Proof.

Adapting from the proof of lemma D.4 and we know that DISPLAYFORM14 Similarly, we have DISPLAYFORM15 Taking the first component of FORMULA0 and FORMULA0 gives us DISPLAYFORM16 DISPLAYFORM17 Thus, DISPLAYFORM18 Similarly, we get DISPLAYFORM19 Since ∀i, j ∈ [2], π − θ wi,w * j = arccos(−θ wij ), we know that DISPLAYFORM20 For simplicity, based on D.5, we define k 0 = −k, θ 1 = π − θ w2,w * 1 and θ 2 = π − θ w2,w *

.

Then DISPLAYFORM0 WLOG, assume k 0 ≥ 1, otherwise we can switch w 1 and w 2 .Thus, DISPLAYFORM1 DISPLAYFORM2 Similarly, if we apply the change of variables onto the second component of (217), we will get DISPLAYFORM3 Thus, DISPLAYFORM4 Proof.

Note that when θ ∈ [0, DISPLAYFORM5 is a strict decreasing function w.r.t.

θ.

Note that G(0) = k 0 + 1 > 0 and DISPLAYFORM6 Then the only thing we need to prove is DISPLAYFORM7 Since the inequality (244) holds only when cos 2 , which means k 0 = 3 and k 0 = 1, which makes a contradiction.

Thus, DISPLAYFORM8 Therefore, DISPLAYFORM9 , which completes the proof.

Lemma D.10.

F (θ) is either strictly decreasing or first decrease and then increase when θ ∈ (θ 0 , DISPLAYFORM10 Proof.

DISPLAYFORM11 Define DISPLAYFORM12 , then H(θ)·F (θ) < 0(i.e., when H(θ) is positive, F (θ) is decreasing, otherwise F (θ) is increasing), and we know that DISPLAYFORM13 Note that (251) holds because θ > θ 0 ≥ π 2k0 .

Thus, H(θ) is a strictly decreasing function when θ ∈ (θ 0 , π k0 ].

We can see that DISPLAYFORM14 Thus, if H( DISPLAYFORM15 Otherwise, F (θ) first decrease and then increase when θ ∈ (θ 0 , DISPLAYFORM16 Proof.

From lemma D.10 we have already known that F (θ) is either strictly decreasing or first decrease and then increase when θ ∈ (θ 0 , π k0 ], so the maximum of the function value on an interval can only be at the endpoints of that interval, which means that we only need to prove F ( DISPLAYFORM17 Thus, h(x) is decreasing in [ , π].

However, we know that h( DISPLAYFORM18 which means that F ( DISPLAYFORM19 Lemma D.12.

θ 1 = θ 2 .Proof.

From the proof of lemma D.8 we get DISPLAYFORM20 Thus, DISPLAYFORM21 Using lemma D.9, If z = (tz 1 , z 2 ), ||z 1 || = ||z 2 || = 1 and w E 2D CASES WITH ASSUMPTION RELAXATION Since this section is pretty similar to B, I will try my best to make it brief and point out the most important things in the proof.

DISPLAYFORM22

After the changing of variables(i.e., polar coordinates), we know that w 1 = (cos θ 1 , sin θ 1 ) and w 2 = (cos θ 2 , sin θ 2 ). (301) Then when θ is in the first part to the fourth part, the function h will change to four different functions: DISPLAYFORM0 (305) WLOG, we assume θ 1 ≤ θ 2 .

E.2 0 ≤ θ 1 ≤ θ 2 ≤ α First, it's easy to verify that ∀θ ∈ [0, θ], h 1 (θ) + h 1 (α − θ) = 0.Besides, h 1 (θ) = sin θ + sin(α − θ) − (π − θ) cos θ − (π − α + θ) cos(α − θ)= 2 sin α 2 cos(θ − α 2 ) − (π − θ) cos θ − (π − α + θ) cos(α − θ)≤ 2 sin α 2 − π 2 (cos θ + cos(α − θ)) (308) DISPLAYFORM1 ≤ 2 sin α 2 − π cos α 2 < 0.When m(w 1 ) = m(w 2 ) = 0, we know that h 1 (θ 1 )+h 1 (θ 2 ) = 0, and because of those two properties above, we know that θ 1 + θ 2 = α.

Thus, θ 1 ∈ [0, α 2 ].

And we have the following lemma Lemma E.1.

m(w 1 ) ≤ 0.

m(w 1 ) = sin(α − 2θ 1 )(π − α + 2θ 1 ) − (π − α + θ 1 ) sin(α − θ 1 ) + (π − θ 1 ) sin θ 1 (311) ≥ sin(α − 2θ 1 )(π − α + θ 1 ) − (π − α + θ 1 ) sin(α − θ 1 ) + (π − θ 1 ) sin θ 1 (312)≥ sin(α − 2θ 1 )(π − α + θ 1 ) − (π − α + θ 1 ) sin(α − θ 1 ) + (π − α 2 ) sin θ 1= (π − α + θ 1 )(sin(α − 2θ 1 ) − sin(α − θ 1 )) + (π − α 2 ) sin θ 1≥ (π − α 2 )(sin(α − 2θ 1 ) − sin(α − θ 1 ) + sin θ 1 )= (π − α 2 )(sin(α − 2θ 1 ) − sin θ 1 − sin θ 1 cos(α − 2θ 1 ) − cos θ 1 sin(α − 2θ 1 )) (316) DISPLAYFORM0 Thus, the only possible critical points are m(w 1 ) = 0, which are 0 and α 2 .

After verification, we conclude that there are only two critical points in this case: (θ 1 , θ 2 ) = (0, α) or (θ 1 , θ 2 ) = ( α 2 , α 2 ).E.3 α ≤ θ 1 ≤ θ 2 ≤ π When m(w 1 ) = m(w 2 ) = 0, we know that h 1 (θ 1 ) + h 1 (θ 2 ) = 0.

However, when θ ∈ [α, π], we know that h 2 (θ) = (π − θ + α) sin(α − θ) − (π − θ) sin θ ≤ 0.The inequality cannot become equal because the possible values of θs such that each term equals zero has no intersection.

Thus, h 2 (θ) is always negative, which means that in this case there are no critical points.

E.4 π ≤ θ 1 ≤ θ 2 ≤ π + α It's easy to verify that ∀θ ∈ [π, π + α], h 3 (θ) + h 3 (2π + α − θ) = 0.

Furthermore, DISPLAYFORM1 > 0.Thus, from m(w 1 ) = m(w 2 ) = 0, we know that h 1 (θ 1 ) + h 1 (θ 2 ) = 0 we get θ 1 + θ 2 = 2π + α, which means that θ 1 ∈ [π, π + α 2 ], so we can prove the following lemma: Lemma E.2.

m(w 1 ) ≤ 0.Proof.

Let θ = θ 1 − π, then m(w 1 ) = (π − θ 2 + θ 1 ) sin(θ 1 − θ 2 ) + h 3 (θ 1 ) (322) = (π + θ − α + θ ) sin(2θ − α) + h 1 (θ ) + π sin θ − π sin(α − θ ) (323) ≤ (π + 2θ − α) sin(2θ − α) + sin(α − 2θ )(π + 2θ − α) + π(sin θ − sin(α − θ )) (324) ≤ π(sin θ − cos θ ) (325) ≤ 0.The first inequality is from lemma E.1.Thus, the only possible critical points are m(w 1 ) = 0, which are π and π + α 2 .

After verification, we conclude that there are only two critical points in this case: (θ 1 , θ 2 ) = (π, π + α) or (θ 1 , θ 2 ) = (π + α 2 , π + α 2 ).

@highlight

Recovery guarantee of stochastic gradient descent with random initialization for learning a two-layer neural network with two hidden nodes, unit-norm weights, ReLU activation functions and Gaussian inputs.