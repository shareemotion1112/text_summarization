Stochastic gradient descent (SGD) with stochastic momentum is popular in nonconvex stochastic optimization and particularly for the training of deep neural networks.

In standard SGD, parameters are updated by improving along the path of the gradient at the current iterate on a batch of examples, where the addition of a ``momentum'' term biases the update in the direction of the previous change in parameters.

In non-stochastic convex optimization one can show that a momentum adjustment provably reduces convergence time in many settings, yet such results have been elusive in the stochastic and non-convex settings.

At the same time, a widely-observed empirical phenomenon is that in training deep networks stochastic momentum appears to significantly improve convergence time, variants of it have flourished in the development of other popular update methods, e.g. ADAM, AMSGrad, etc.

Yet theoretical justification for the use of stochastic momentum has remained a significant open question.

In this paper we propose an answer: stochastic momentum improves deep network training because it modifies SGD to escape saddle points faster and, consequently, to more quickly find a second order stationary point.

Our theoretical results also shed light on the related question of how to choose the ideal momentum parameter--our analysis suggests that $\beta \in [0,1)$ should be large (close to 1), which comports with empirical findings.

We also provide experimental findings that further validate these conclusions.

SGD with stochastic momentum has been a de facto algorithm in nonconvex optimization and deep learning.

It has been widely adopted for training machine learning models in various applications.

Modern techniques in computer vision (e.g. Krizhevsky et al. (2012) ; He et al. (2016) ; Cubuk et al. (2018) ; Gastaldi (2017)), speech recognition (e.g. Amodei et al. (2016) ), natural language processing (e.g. Vaswani et al. (2017) ), and reinforcement learning (e.g. Silver et al. (2017) ) use SGD with stochastic momentum to train models.

The advantage of SGD with stochastic momentum has been widely observed (Hoffer et al. (2017) ; Loshchilov & Hutter (2019) ; Wilson et al. (2017) ).

Sutskever et al. (2013) demonstrate that training deep neural nets by SGD with stochastic momentum helps achieving in faster convergence compared with the standard SGD (i.e. without momentum).

The success of momentum makes it a necessary tool for designing new optimization algorithms in optimization and deep learning.

For example, all the popular variants of adaptive stochastic gradient methods like Adam (Kingma & Ba (2015) ) or AMSGrad (Reddi et al. (2018b) ) include the use of momentum.

Despite the wide use of stochastic momentum (Algorithm 1) in practice, justification for the clear empirical improvements has remained elusive, as has any mathematical guidelines for actually setting the momentum parameter-it has been observed that large values (e.g. ?? = 0.9) work well in practice.

It should be noted that Algorithm 1 is the default momentum-method in popular software packages such as PyTorch and Tensorflow.

1 In this paper we provide a theoretical analysis for SGD with 1: Required:

Step size parameter ?? and momentum parameter ??.

2: Init: w0 ??? R d and m???1 = 0 ??? R d .

3: for t = 0 to T do 4:

Given current iterate wt, obtain stochastic gradient gt := ???f (wt; ??t).

Update stochastic momentum mt := ??mt???1 + gt.

Update iterate wt+1 := wt ??? ??mt.

7: end for momentum.

We identify some mild conditions that guarantees SGD with stochastic momentum will provably escape saddle points faster than the standard SGD, which provides clear evidence for the benefit of using stochastic momentum.

For stochastic heavy ball momentum, a weighted average of stochastic gradients at the visited points is maintained.

The new update is computed as the current update minus a step in the direction of the momentum.

Our analysis shows that these updates can amplify a component in an escape direction of the saddle points.

In this paper, we focus on finding a second-order stationary point for smooth non-convex optimization by SGD with stochastic heavy ball momentum.

Specifically, we consider the stochastic nonconvex optimization problem, min w???R d f (w) := E ?????D [f (w; ??)], where we overload the notation so that f (w; ??) represents a stochastic function induced by the randomness ?? while f (w) is the expectation of the stochastic functions.

An ( , )-second-order stationary point w satisfies ???f (w) ??? and ??? 2 f (w) ??? I.

Obtaining a second order guarantee has emerged as a desired goal in the nonconvex optimization community.

Since finding a global minimum or even a local minimum in general nonconvex optimization can be NP hard (Anandkumar & Ge (2016) ; Nie (2015) ; Murty & Kabadi (1987) ; Nesterov (2000)), most of the papers in nonconvex optimization target at reaching an approximate second-order stationary point with additional assumptions like Lipschitzness in the gradients and the Hessian (e.g. Allen-Zhu & Li (2018) ; et al. (2018) ).

2 We follow these related works for the goal and aim at showing the benefit of the use of the momentum in reaching an ( , )-second-order stationary point.

We introduce a required condition, akin to a model assumption made in (Daneshmand et al. (2018) ), that ensures the dynamic procedure in Algorithm 2 produces updates with suitable correlation with the negative curvature directions of the function f .

Definition 1.

Assume, at some time t, that the Hessian H t = ??? 2 f (w t ) has some eigenvalue smaller than ??? and ???f (w t ) ??? .

Let v t be the eigenvector corresponding to the smallest eigenvalue of ??? 2 f (w t ).

The stochastic momentum m t satisfies Correlated Negative Curvature (CNC) at t with parameter ?? > 0 if

As we will show, the recursive dynamics of SGD with heavy ball momentum helps in amplifying the escape signal ??, which allows it to escape saddle points faster.

We show that, under CNC assumption and some minor constraints that upper-bound parameter ??, if SGD with momentum has properties called Almost Positively Aligned with Gradient (APAG), Almost Positively Correlated with Gradient (APCG), and Gradient Alignment or Curvature Exploitation (GrACE), defined in the later section, then it takes T = O((1 ??? ??) log(1/(1 ??? ??) ) ???10 ) iterations to return an ( , ) second order stationary point.

Alternatively, one can obtain an ( , ??? ) second order stationary point in T = O((1 ??? ??) log(1/(1 ??? ??) ) ???5 ) iterations.

Our theoretical result demonstrates that a larger momentum parameter ?? can help in escaping saddle points faster.

As saddle points are pervasive in the loss landscape of optimization and deep learning (Dauphin et al. (2014) ; Choromanska et al. (2015) ), the result sheds light on explaining why SGD with momentum enables training faster in optimization and deep learning.

Notation: In this paper we use E t [??] to represent conditional expectation E[??|w 1 , w 2 , . . .

, w t ], which is about fixing the randomness upto but not including t and notice that w t was determined at t ??? 1.

2.1 A THOUGHT EXPERIMENT.

Let us provide some high-level intuition about the benefit of stochastic momentum with respect to avoiding saddle points.

In an iterative update scheme, at some time t 0 the parameters w t0 can enter a saddle point region, that is a place where Hessian ??? 2 f (w t0 ) has a non-trivial negative eigenvalue, say ?? min (??? 2 f (w t0 )) ??? ??? , and the gradient ???f (w t0 ) is small in norm, say ???f (w t0 ) ??? .

The challenge here is that gradient updates may drift only very slowly away from the saddle point, and may not escape this region; see (Du et al. (2017) ; Lee et al. (2019) ) for additional details.

On the other hand, if the iterates were to move in one particular direction, namely along v t0 the direction of the smallest eigenvector of ??? 2 f (w t0 ), then a fast escape is guaranteed under certain constraints on the step size ??; see e.g. ).

While the negative eigenvector could be computed directly, this 2nd-order method is prohibitively expensive and hence we typically aim to rely on gradient methods.

With this in mind, Daneshmand et al. (2018) , who study non-momentum SGD, make an assumption akin to our CNC property described above that each stochastic gradient g t0 is strongly non-orthogonal to v t0 the direction of large negative curvature.

This suffices to drive the updates out of the saddle point region.

In the present paper we study stochastic momentum, and our CNC property requires that the update direction m t0 is strongly non-orthogonal to v t0 ; more precisely, E t0 [ m t0 , v t0 2 ] ??? ?? > 0.

We are able to take advantage of the analysis of (Daneshmand et al. (2018) ) to establish that updates begin to escape a saddle point region for similar reasons.

Further, this effect is amplified in successive iterations through the momentum update when ?? is close to 1.

Assume that at some w t0 we have m t0 which possesses significant correlation with the negative curvature direction v t0 , then on successive rounds m t0+1 is quite close to ??m t0 , m t0+2 is quite close to ?? 2 m t0 , and so forth; see Figure 1 for an example.

This provides an intuitive perspective on how momentum might help accelerate the escape process.

Yet one might ask does this procedure provably contribute to the escape process and, if so, what is the aggregate performance improvement of the momentum?

We answer the first question in the affirmative, and we answer the second question essentially by showing that momentum can help speed up saddle-point escape by a multiplicative factor of 1 ??? ??.

On the negative side, we also show that ?? is constrained and may not be chosen arbitrarily close to 1.

Let us now establish, empirically, the clear benefit of stochastic momentum on the problem of saddle-point escape.

We construct two stochastic optimization tasks, and each exhibits at least one significant saddle point.

The two objectives are as follows.

Problem (3) of these was considered by (Staib et al. (2019); Reddi et al. (2018a) ) and represents a very straightforward non-convex optimization challenge, with an embedded saddle given by the matrix H := diag([1, ???0.1]), and stochastic gaussian perturbations given by b i ??? N (0, diag([0.1, 0.001])); the small variance in the second component provides lower noise in the escape direction.

Here we have set n = 10.

Observe that the origin is in the neighborhood of saddle points and has objective We plot convergence in function value f (??) given in (3).

Initialization is always set as w0 = 0.

All the algorithms use the same step size ?? = 5 ?? 10 ???5 .

Fig. 4b : We plot convergence in relative distance to the true model w * , defined as min( wt ??? w * , wt + w * )/ w * , which more appropriately captures progress as the global sign of the objective (4) is unrecoverable.

All the algorithms are initialized at the same point w0 ??? N (0, I d /(10000d)) and use the same step size ?? = 5 ?? 10 ???4 .

value zero.

SGD and SGD with momentum are initialized at the origin in the experiment so that they have to escape saddle points before the convergence.

The second objective (4) appears in the phase retrieval problem, that has real applications in physical sciences (Cand??s et al. (2013); Shechtman et al. (2015) ).

In phase retrieval 3 , one wants to find an unknown w * ??? R d with access to but a few samples y i = (a i w * ) 2 ; the design vector a i is known a priori.

Here we have sampled The empirical findings, displayed in Figure 2 , are quite stark: for both objectives, convergence is significantly accelerated by larger choices of ??.

In the first objective (Figure 4a ), we see each optimization trajectory entering a saddle point region, apparent from the "flat" progress, yet we observe that large-momentum trajectories escape the saddle much more quickly than those with smaller momentum.

A similar affect appears in Figure 4b .

To the best of our knowledge, this is the first reported empirical finding that establishes the dramatic speed up of stochastic momentum for finding an optimal solution in phase retrieval.

Heavy ball method: The heavy ball method was originally proposed by Polyak (1964) .

It has been observed that this algorithm, even in the deterministic setting, provides no convergence speedup over standard gradient descent, except in some highly structure cases such as convex quadratic objectives where an "accelerated" rate is possible (Lessard et al. (2016) (2019)).

We provide a comprehensive survey of the related works about heavy ball method in Appendix A.

Reaching a second order stationary point: As we mentioned earlier, there are many works aim at reaching a second order stationary point.

We classify them into two categories: specialized algorithms and simple GD/SGD variants.

Specialized algorithms are those designed to exploit the negative curvature explicitly and escape saddle points faster than the ones without the explicit exploitation (2019)).

Our work belongs to this category.

In this category, perhaps the pioneer works are (Ge et al. (2015) ) and (Jin et al. (2017) ).

Jin et al. (2017) show that explicitly adding isotropic noise in each iteration guarantees that GD escapes saddle points and finds a second order stationary 3 It is known that phase retrieval is nonconvex and has the so-called strict saddle property: (1) every local minimizer {w * , ???w * } is global up to phase, (2) each saddle exhibits negative curvature (see e.g. (Sun et al. (2015; ; Chen et al. (2018) )) point with high probability.

Following (Jin et al. (2017) ), Daneshmand et al. (2018) assume that stochastic gradient inherently has a component to escape.

Specifically, they make assumption of the Correlated Negative Curvature (CNC) for stochastic gradient g t so that E t [ g t , v t 2 ]

??? ?? > 0.

The assumption allows the algorithm to avoid the procedure of perturbing the updates by adding isotropic noise.

Our work is motivated by (Daneshmand et al. (2018) ) but assumes CNC for the stochastic momentum m t instead.

In Appendix A, we compare the results of our work with the related works.

We assume that the gradient ???f is L-Lipschitz; that is, f is L-smooth.

Further, we assume that the Hessian ??? 2 f is ??-Lipschitz.

These two properties ensure that ???f

w???w 3 , ???w, w .

Furthermore, we assume that the stochastic gradient has bounded noise ???f (w) ??? ???f (w; ??) 2 ??? ?? 2 and that the norm of stochastic momentum is bounded so that m t ??? c m .

We denote ?? i M i as the matrix product of matrices {M i } and we use

to denote the spectral norm of the matrix M .

Our analysis of stochastic momentum relies on three properties of the stochastic momentum dynamic.

These properties are somewhat unusual, but we argue they should hold in natural settings, and later we aim to demonstrate that they hold empirically in a couple of standard problems of interest.

Definition 2.

We say that SGD with stochastic momentum satisfies Almost Positively Aligned with Gradient (APAG) 4 if we have

We say that SGD with stochastic momentum satisfies Almost Positively Correlated with Gradient (APCG) with parameter ?? if ???c > 0 such that,

where the PSD matrix M t is defined as

for any integer 1 ??? k ??? ?? ??? 1, and ?? is any step size chosen that guarantees each G s,t is PSD.

Definition 3.

We say that the SGD with momentum exhibits Gradient Alignment or Curvature Exploitation (GrACE) if ???c h ??? 0 such that

APAG requires that the momentum term m t must, in expectation, not be significantly misaligned with the gradient ???f (w t ).

This is a very natural condition when one sees that the momentum term is acting as a biased estimate of the gradient of the deterministic f .

APAG demands that the bias can not be too large relative to the size of ???f (w t ).

Indeed this property is only needed in our analysis when the gradient is large (i.e. ???f (w t ) ??? ) as it guarantees that the algorithm makes progress; our analysis does not require APAG holds when gradient is small.

APCG is a related property, but requires that the current momentum term m t is almost positively correlated with the the gradient ???f (w t ), but measured in the Mahalanobis norm induced by M t .

It may appear to be an unusual object, but one can view the PSD matrix M t as measuring something about the local curvature of the function with respect to the trajectory of the SGD with momentum dynamic.

We will show that this property holds empirically on two natural problems for a reasonable constant c .

APCG is only needed in our analysis when the update is in a saddle region with significant 2 versus iterations.

For (b), we only report them when the gradient is large ( ???f (wt) ??? 0.02).

It shows that the value is large than ???0.5 except the transition.

For (e), we observe that the value is almost always nonnegative.

Sub-figures (c) and (f): We plot the value of

Gs,t) and we only report the values when the update is in the region of saddle points.

For (f), we let Mt = (?? 500 s=1 Gs,t)(?? 500 s=1 Gs,t) and we observe that the value is almost always nonnegative.

The figures implies that SGD with momentum has APAG and APCG properties in the experiments.

Furthermore, an interesting observation is that, for the phase retrieval problem, the expected values might actually be nonnegative.

negative curvature, ???f (w) ??? and ?? min (??? 2 f (w)) ??? ??? .

Our analysis does not require APCG holds when the gradient is large or the update is at an ( , )-second order stationary point.

For GrACE, the first term on l.h.s of (7) measures the alignment between stochastic momentum m t and the gradient ???f (w t ), while the second term on l.h.s measures the curvature exploitation.

The first term is small (or even negative) when the stochastic momentum m t is aligned with the gradient ???f (w t ), while the second term is small (or even negative) when the stochastic momentum m t can exploit a negative curvature (i.e. the subspace of eigenvectors that corresponds to the negative eigenvalues of the Hessian ??? 2 f (w t ) if exists).

Overall, a small sum of the two terms (and, consequently, a small c h ) allows one to bound the function value of the next iterate (see Lemma 8).

On Figure 3 , we report some quantities related to APAG and APCG as well as the gradient norm when solving the previously discussed problems (3) and (4) using SGD with momentum.

We also report a quantity regarding GrACE on Figure 4 in the appendix.

The high level idea of our analysis follows as a similar template to (Jin et al. (2017); Daneshmand et al. (2018); Staib et al. (2019) ).

Our proof is structured into three cases: either (a) ???f (w) ??? , or (b) ???f (w) ??? and ?? min (??? 2 f (w)) ??? ??? , or otherwise (c) ???f (w) ??? and ?? min (??? 2 f (w)) ??? ??? , meaning we have arrived in a second-order stationary region.

The precise algorithm we analyze is Algorithm 2, which identical to Algorithm 1 except that we boost the step size to a larger value r on 4 Note that our analysis still go through if one replaces 1 2 on r.h.s.

of (5) with any larger number c < 1; the resulted iteration complexity would be only a constant multiple worse.

Step size parameters r and ??, momentum parameter ??, and period parameter T thred .

2: Init:

Get stochastic gradient gt at wt, and set stochastic momentum mt := ??mt???1 + gt.

Set learning rate:?? := ?? unless (t mod T thred ) = 0 in which case?? := r 6: wt+1 = wt ?????mt.

7: end for occasion.

We will show that the algorithm makes progress in cases (a) and (b).

In case (c), when the goal has already been met, further execution of the algorithm only weakly hurts progress.

Ultimately, we prove that a second order stationary point is arrived at with high probability.

While our proof borrows tools from (Daneshmand et al. (2018) ; Staib et al. (2019)), much of the momentum analysis is entirely novel to our knowledge.

Theorem 1.

Assume that the stochastic momentum satisfies CNC.

Set

, and

).

If SGD with momentum (Algorithm 2) has APAG property when gradient is large ( ???f (w) ??? ), APCG T thred property when it enters a region of saddle points that exhibits a negative curvature ( ???f (w) ??? and ?? min (??? 2 f (w)) ??? ??? ), and GrACE property throughout the iterations, then it reaches an ( , ) second order stationary point in

) iterations with high probability 1 ??? ??.

The theorem implies the advantage of using stochastic momentum for SGD.

Higher ?? leads to reaching a second order stationary point faster.

As we will show in the following, this is due to that higher ?? enables escaping the saddle points faster.

In Subsection 3.2.1, we provide some key details of the proof of Theorem 1.

The interested reader can read a high-level sketch of the proof, as well as the detailed version, in Appendix G.

Remark 1: (constraints on ??) We also need some minor constraints on ?? so that ?? cannot be too close to 1.

They are 1) On the other hand, for CNC-SGD, based on Table 3 in their paper, is T thred =?? 1 ?? .

One can clearly see that T thred of our result has a dependency 1 ??? ??, which makes it smaller than that of Daneshmand et al. (2018) for any same ?? and consequently demonstrates escaping saddle point faster with momentum.

Remark 3: (finding a second order stationary point) Denote a number such that ???t, g t ??? .

In Appendix G.3, we show that in the high momentum regime where

, Algorithm 2 is strictly better than CNC-SGD of Daneshmand et al. (2018) , which means that a higher momentum can help find a second order stationary point faster.

Empirically, we find out that c ??? 0 (Figure 3 ) and c h ??? 0 (Figure 4 ) in the phase retrieval problem, so the condition is easily satisfied for a wide range of ??.

In this subsection, we analyze the process of escaping saddle points by SGD with momentum.

Denote t 0 any time such that (t 0 mod T thred ) = 0.

Suppose that it enters the region exhibiting a small gradient but a large negative eigenvalue of the Hessian (i.e. ???f (w t0 ) ??? and ?? min (??? 2 f (w t0 )) ??? ??? ).

We want to show that it takes at most T thred iterations to escape the region and whenever it escapes, the function value decreases at least by F thred = O( 4 ) on expectation, where the precise expression of F thred will be determined later in Appendix E. The technique that we use is proving by contradiction.

Assume that the function value on expectation does not decrease at least F thred in T thred iterations.

Then, we get an upper bound of the expected distance E t0 [ w t0+T thred ??? w t0 2 ] ??? C upper .

Yet, by leveraging the negative curvature, we also show a lower bound of the form E t0 [ w t0+T thred ??? w t0 2 ] ??? C lower .

The analysis will show that the lower bound is larger than the upper bound (namely, C lower > C upper ), which leads to the contradiction and concludes that the function value must decrease at least F thred in T thred iterations on expectation.

Since

6 ), the dependency on ?? suggests that larger ?? can leads to smaller T thred , which implies that larger momentum helps in escaping saddle points faster.

Lemma 1 below provides an upper bound of the expected distance.

The proof is in Appendix C. Lemma 1.

Denote t 0 any time such that

We see that C upper,t in Lemma 1 is monotone increasing with t, so we can define C upper := C upper,T thred .

Now let us switch to obtaining the lower bound of E t0 [ w t0+T thred ??? w t0 2 ].

The key to get the lower bound comes from the recursive dynamics of SGD with momentum.

Lemma 2.

Denote t 0 any time such that (t 0 mod T thred ) = 0.

Let us define a quadratic ap-

Then we can write w t0+t ??? w t0 exactly using the following decomposition.

The proof of Lemma 2 is in Appendix D. Furthermore, we will use the quantities q v,t???1 , q m,t???1 , q q,t???1 , q w,t???1 , q ??,t???1 as defined above throughout the analysis.

Lemma 3.

Following the notations of Lemma 2, we have that

We are going to show that the dominant term in the lower bound of

, which is the critical component for ensuring that the lower bound is larger than the upper bound of the expected distance.

Following the conditions and notations in Lemma 1 and Lemma 2, we have that

Proof.

We know that ?? min (H) ??? ??? < 0.

Let v be the eigenvector of the Hessian H with unit norm that corresponds to ?? min (H) so that Hv = ?? min (H)v.

We have

where (a) is because v is with unit norm, (b) is by Cauchy-Schwarz inequality, (c), (d) are by the definitions, and (e) is by the CNC assumption so that

Observe that the lower bound in (8) is monotone increasing with t and the momentum parameter ??.

Moreover, it actually grows exponentially in t. To get the contradiction, we have to show that the lower bound is larger than the upper bound.

By Lemma 1 and Lemma 3, it suffices to prove the following lemma.

We provide its proof in Appendix E. Lemma 5.

Let F thred = O( 4 ) and ?? 2 T thred ??? r 2 .

By following the conditions and notations in Theorem 1, Lemma 1 and Lemma 2, we conclude that if SGD with momentum (Algorithm 2) has the APCG property, then we have that

In this paper, we identify three properties that guarantee SGD with momentum in reaching a secondorder stationary point faster by a higher momentum, which justifies the practice of using a large value of momentum parameter ??.

We show that a greater momentum leads to escaping strict saddle points faster due to that SGD with momentum recursively enlarges the projection to an escape direction.

However, how to make sure that SGD with momentum has the three properties is not very clear.

It would be interesting to identify conditions that guarantee SGD with momentum to have the properties.

Perhaps a good starting point is understanding why the properties hold in phase retrieval.

We believe that our results shed light on understanding the recent success of SGD with momentum in non-convex optimization and deep learning.

Heavy ball method: The heavy ball method was originally proposed by Polyak (1964) .

It has been observed that this algorithm, even in the deterministic setting, provides no convergence speedup over standard gradient descent, except in some highly structure cases such as convex quadratic objectives where an "accelerated" rate is possible (Lessard et al. (2016) ; Goh (2017)).

In recent years, some works make some efforts in analyzing heavy ball method for other classes of optimization problems besides the quadratic functions.

provide a unified analysis of stochastic heavy ball momentum and Nesterov's momentum for smooth non-convex objective functions.

They show that the expected gradient norm converges at rate O(1/ ??? t).

Yet, the rate is not better than that of the standard SGD.

We are also aware of the works (Ghadimi & Lan (2016; 2013) ), which propose some variants of stochastic accelerated algorithms with first order stationary point guarantees.

Yet, the framework in (Ghadimi & Lan (2016; 2013) ) does not capture the stochastic heavy ball momentum used in practice.

There is also a negative result about the heavy ball momentum.

Kidambi et al. (2018) show that for a specific strongly convex and strongly smooth problem, SGD with heavy ball momentum fails to achieving the best convergence rate while some algorithms can.

Reaching a second order stationary point: As we mentioned earlier, there are many works aim at reaching a second order stationary point.

We classify them into two categories: specialized algorithms and simple GD/SGD variants.

Specialized algorithms are those designed to exploit the negative curvature explicitly and escape saddle points faster than the ones without the explicit exploitation (2012)), in which the gradient g t is multiplied by a preconditioning matrix G t and the update is w t+1 = w t ??? G ???1/2 t g t .

The work shows that the algorithm can help in escaping saddle points faster compared to the standard SGD under certain conditions.

Fang et al. (2019) propose average-SGD, in which a suffix averaging scheme is conducted for the updates.

They also assume an inherent property of stochastic gradients that allows SGD to escape saddle points.

We summarize the iteration complexity results of the related works for simple SGD variants on Table 1 .

6 The readers can see that the iteration complexity of (Fang et al. (2019) (2019)) and our result.

So, we want to explain the results and clarify the differences.

First, we focus on explaining why the popular algorithm, SGD with heavy ball momentum, works well in practice, which is without the suffix averaging scheme used in (Fang et al. (2019) ) and is without the explicit perturbation used in (Jin et al. (2019) ).

Specifically, we focus on studying the effect of stochastic heavy ball momentum and showing the advantage of using it.

Furthermore, our analysis framework is built on the work of (Daneshmand et al. (2018) ).

We believe that, based on the insight in our work, one can also show the advantage of stochastic momentum by modifying the assumptions and algorithms in (Fang et al. (2019) ) or (Jin et al. (2019) ) and consequently get a better dependency on .

B LEMMA 6, 7, AND 8

In the following, Lemma 7 says that under the APAG property, when the gradient norm is large, on expectation SGD with momentum decreases the function value by a constant and consequently makes progress.

On the other hand, Lemma 8 upper-bounds the increase of function value of the next iterate (if happens) by leveraging the GrACE property.

Lemma 6.

If SGD with momentum has the APAG property, then, considering the update step w t+1 = w t ??? ??m t , we have that

Proof.

By the L-smoothness assumption,

Taking the expectation on both sides.

We have

where we use the APAG property in the last inequality.

Lemma 7.

Assume that the step size ?? satisfies ?? ??? 2 8Lc 2 m .

If SGD with momentum has the APAG property, then, considering the update step w t+1 = w t ??? ??m t , we have that

2 , where the last inequality is due to the constraint of ??.

Lemma 8.

If SGD with momentum has the GrACE property, then, considering the update step w t+1 = w t ??? ??m t , we have that

Proof.

Consider the update rule w t+1 = w t ??? ??m t , where m t represents the stochastic momentum and ?? is the step size.

By ??-Lipschitzness of Hessian, we have

Taking the conditional expectation, one has

C PROOF OF LEMMA 1

Lemma 1 Denote t 0 any time such that (t 0 mod T thred ) = 0.

Suppose that

Proof.

Recall that the update is w t0+1 = w t0 ??? rm t0 , and w t0+t = w t0+t???1 ??? ??m t0+t???1 , for t > 1.

We have that

where the first inequality is by the triangle inequality and the second one is due to the assumption that m t ??? c m for any t. Now let us denote

s=1 ?? s and let us rewrite g t = ???f (w t ) + ?? t , where ?? t is the zero-mean noise.

We have that

To proceed, we need to upper bound E t0 [4??

We have that

where (a) is because E t0 [?? t0+i ?? t0+j ] = 0 for i = j, (b) is by that ?? t 2 ??? ?? 2 and max t ?? t ??? 1 1????? .

Combining (14), (15), (16), (17),

Now we need to bound E t0 [

.

By using ??-Lipschitzness of Hessian, we have that

By adding ?? ???f (w t0+s???1 ), g t0+s???1 on both sides, we have ?? ???f (w t0+s???1 ), g t0+s???1 ??? f (w t0+s???1 ) ??? f (w t0+s ) + ?? ???f (w t0+s???1 ), g t0+s???1 ??? m t0+s???1

(20) Taking conditional expectation on both sides leads to

D PROOF OF LEMMA 2 AND LEMMA 3

Lemma 2 Denote t 0 any time such that (t 0 mod T thred ) = 0.

Let us define a quadratic approximation at w t0 , Q(w) := f (w t0 ) + w ??? w t0 , ???f (w t0 ) + 1 2 (w ??? w t0 ) H(w ??? w t0 ), where

??? qv,t???1 := ?? t???1 j=1 Gj ??? rmt 0 .

??? qm,t???1 := ??? t???1 s=1 ?? t???1 j=s+1 Gj ?? s mt 0 .

??? qq,t???1 :

??? qw,t???1 :

??? q ??,t???1 :

Then, w t0+t ??? w t0 = q v,t???1 + ??q m,t???1 + ??q q,t???1 + ??q w,t???1 + ??q ??,t???1 .

Denote t 0 any time such that (t 0 mod T thred ) = 0.

Let us define a quadratic approximation at w t0 ,

where H := ??? 2 f (w t0 ).

Also, we denote

Proof.

First, we rewrite m t0+j for any j ??? 1 as follows.

We have that

where (a) is by using (29) with j = t ??? 1, (b) is by subtracting and adding back the same term, and (c) is by ???Q(w t0+t???1 ) = ???f (w t0 ) + H(w t0+t???1 ??? w t0 ).

To continue, by using the nations in (28), we can rewrite (30) as

Recursively expanding (31) leads to

where (a) we use the notation that ?? t???1

and the notation that ?? t???1 j=t G j = 1 and (b) is by the update rule.

By using the definitions of {q ,t???1 } in the lemma statement, we complete the proof.

Lemma 3 Following the notations of Lemma 2, we have that

Proof.

Following the proof of Lemma 2, we have

Therefore, by using a + b 2 ??? a 2 + 2 a, b ,

E PROOF OF LEMMA 5

Lemma 5 Let F thred = O( 4 ) and ?? 2 T thred ??? r 2 .

By following the conditions and notations in Theorem 1, Lemma 1 and Lemma 2, we conclude that if SGD with momentum (Algorithm 2) has the APCG property, then we have that (65), (66) cr ??? (25), (39), (87), (89)

from (88) "

) from (82) W.l.o.g, we assume that c m , L, ?? 2 , c , c h , and ?? are not less than one and that ??? 1.

E.1 SOME CONSTRAINTS ON ??.

We require that parameter ?? is not too close to 1 so that the following holds,

??? 2) ?? 2 (1 ??? ??) 3 > 1.

??? 3) c (1 ??? ??) 2 > 1.

???

The constraints upper-bound the value of ??.

That is, ?? cannot be too close to 1.

We note that the ?? dependence on L, ??, and c are only artificial.

We use these constraints in our proofs but they are mostly artefacts of the analysis.

For example, if a function is L-smooth, and L < 1, then it is also 1-smooth, so we can assume without loss of generality that L > 1.

Similarly, the dependence on ?? is not highly relevant, since we can always increase the variance of the stochastic gradient, for example by adding an O(1) gaussian perturbation.

To prove Lemma 5, we need a series of lemmas with the choices of parameters on Table 3 .

Upper bounding E t0 [ q q,t???1 ]:

Lemma 9.

Following the conditions in Lemma 1 and Lemma 2, we have

Proof.

(37) where (a), (c), (d) is by triangle inequality, (b) is by the fact that Ax 2 ??? A 2 x 2 for any matrix A and vector x. Now that we have an upper bound of ???f (w t0+k ) ??? ???f (w t0+s ) ,

where (a) is by the assumption of L-Lipschitz gradient and (b) is by applying the triangle inequality (s ??? k) times and that w t ??? w t???1 ??? ?? m t???1 ??? ??c m , for any t. We can also derive an upper bound of E t0 [ ???f (w t0+s ) ??? ???Q(w t0+s ) ],

Above, (a) is by the fact that if a function f (??) has ?? Lipschitz Hessian, then

(c.f.

Lemma 1.2.4 in (Nesterov (2013))) and using the definition that

Combing (37), (38), (39), we have that

where on the last line we use the notation that

To continue, let us analyze ?? t???1 j=s+1 G j 2 first.

Above, we use the notation that ?? j := j k=1 ?? j???k .

For (a), it is due to that ?? := ????? min (H), ?? max (H) ??? L, and the choice of ?? so that 1 ??? ??L 1????? , or equivalently,

For (b), it is due to that ?? j ??? 1 for any j and ?? ??? .

Therefore, we can upper-bound the first term on r.h.s of (42) as

where (a) is by that fact that

is by using (44), and (c) is by using that

?? .

Now let us switch to bound

(47) where (a) is by the fact that (44), (c) is by using that

(1???z) 2 for any |z| ??? 1 and substituting z = 1 1+?? , which leads to

in which the last inequality is by chosen the step size ?? so that ?? ??? 1.

By combining (42), (46), and (47), we have that

Proof.

where the last inequality is because ?? is chosen so that 1 ??? ??L 1????? and the fact that ?? max (H) ??? L.

Lower bounding E t0 [2?? q v,t???1 , q q,t???1 ]:

Lemma 11.

Following the conditions in Lemma 1 and Lemma 2, we have

Proof.

By the results of Lemma 9 and Lemma 10

Under review as a conference paper at ICLR 2020

Lower bounding E t0 [2?? q v,t???1 , q ??,t???1 ]:

Lemma 12.

Following the conditions in Lemma 1 and Lemma 2, we have

Proof.

where (a) holds for some coefficients ?? k , (b) is by the tower rule, (c) is because q v,t???1 is measureable with t 0 , and (d) is by the zero mean assumption of ??'s.

Lower bounding E t0 [2?? q v,t???1 , q m,t???1 ]:

Lemma 13.

Following the conditions in Lemma 1 and Lemma 2, we have

Proof.

where (a) is by defining the matrix B : For (b) , notice that the matrix B is symmetric positive semidefinite.

To see that the matrix B is symmetric positive semidefinite, observe that each G j := (I ??? ?? j k=1 ?? j???k H) can be written in the form of G j = U D j U for some orthonormal matrix U and a diagonal matrix D j .

Therefore, the matrix product

U is symmetric positive semidefinite as long as each G j is.

So, (b) is by the property of a matrix being symmetric positive semidefinite.

Lower bounding 2??E t0 [ q v,t???1 , q w,t???1 ]: Lemma 14.

Following the conditions in Lemma 1 and Lemma 2, if SGD with momentum has the APCG property, then

Proof.

Define

where (a) is by the APCG property.

We also have that

where (a) and (b) is by (44).

Substituting the result back to (58), we get

(60) Using the fact that ???f (w t0 ) ??? completes the proof.

Recall that the strategy is proving by contradiction.

Assume that the function value does not decrease at least F thred in T thred iterations on expectation.

Then, we can get an upper bound of the expected distance E t0 [ w t0+T thred ??? w t0 2 ] ??? C upper but, by leveraging the negative curvature, we can also show a lower bound of the form E t0 [ w t0+T thred ??? w t0 2 ] ??? C lower .

The strategy is showing that the lower bound is larger than the upper bound, which leads to the contradiction and concludes that the function value must decrease at least F thred in T thred iterations on expectation.

To get the contradiction, according to Lemma 1 and Lemma 3, we need to show that

Yet, by Lemma 13 and Lemma 12, we have that ??E t0 [ q v,T thred ???1 , q m,T thred ???1 ]

??? 0 and ??E t0 [ q v,T thred ???1 , q ??,T thred ???1 ] = 0.

So, it suffices to prove that

and it suffices to show that Under review as a conference paper at ICLR 2020 as guaranteed by the constraint of ??.

So,

where (a) is by using the inequality log(1 + x) ??? x 2 with x = ???? j ??? 1 and (b) is by making

(1?????) 2 , which is equivalent to the condition that

Now let us substitute the result of (79) back to (77).

We have that

which is what we need to show.

By choosing T thred large enough,

for some constant c > 0, we can guarantee that the above inequality (81) holds.

Lemma 15 (Daneshmand et al. (2018)) Let us define the event

Set T = 2T thred f (w 0 ) ??? min w f (w) /(?????).

We return w uniformly randomly from w 0 , w T thred , w 2T thred , . . . , w kT thred , . . . , w KT thred , where K := T /T thred .

Then, with probability at least 1 ??? ??, we will have chosen a w k where ?? k did not occur.

Proof.

Let P k be the probability that ?? k occurs.

Summing over all K, we have ?????) ) ???6 ).

If SGD with momentum (Algorithm 2) has APAG property when gradient is large ( ???f (w) ??? ), APCG T thred property when it enters a region of saddle points that exhibits a negative curvature ( ???f (w) ??? and ?? min (??? 2 f (w)) ??? ??? ) , and GrACE property throughout the iterations, then it reaches an ( , ) second order stationary point in T = O((1 ??? ??) log( Lcm?? 2 ??c c h

(1?????)???? ) ???10 ) iterations with high probability 1 ??? ??.

In this subsection, we provide a sketch of the proof of Theorem 1.

The complete proof is available in Appendix G. Our proof uses a lemma in (Daneshmand et al. (2018) ), which is Lemma 15 below.

The lemma guarantees that uniformly sampling a w from {w kT thred }, k = 0, 1, 2, . . . , T /T thred gives an ( , )-second order stationary point with high probability.

We replicate the proof of Lemma 15 in Appendix F. We return w uniformly randomly from w 0 , w T thred , w 2T thred , . . . , w kT thred , . . . , w KT thred , where K := T /T thred .

Then, with probability at least 1 ??? ??, we will have chosen a w k where ?? k did not occur.

To use the result of Lemma 15, we need to let the conditions in (86) be satisfied.

We can bound E[f (w (k+1)T thred ) ??? f (w kT thred )|?? k ]

??? ???F thred , based on the analysis of the large gradient norm regime (Lemma 7) and the analysis for the scenario when the update is with small gradient norm but a large negative curvature is available (Subsection 3.2.1).

For the other condition, E[f (w (k+1)T thred ) ??? f (w kT thred )|?? , it requires that the expected amortized increase of function value due to taking the large step size r is limited (i.e. bounded by ?? ) when w kT thred is a second order stationary point.

By having the conditions satisfied, we can apply Lemma 15 and finish the proof of the theorem.

Proof.

Our proof is based on Lemma 15.

So, let us consider the events in Lemma 15, ?? k := { ???f (w kT thred ) ??? or ?? min (??? 2 f (w kT thred )) ??? ??? }.

We first show that E[f (w (k+1)T thred ) ??? f (w kT thred )|?? k ] ??? F thred .

When ???f (w kT thred ) ??? :

Now we are ready to use Lemma 15, since both the conditions are satisfied.

According to the lemma and the choices of parameters value on Table 3 , we can set T = 2T thred f (w 0 ) ??? min w f (w) /(??F thred ) = O((1 ??? ??) log( Lcm?? 2 ??c c h

(1?????)???? ) ???10 ), which will return a w that is an ( , ) second order stationary point.

Thus, we have completed the proof.

(1?????)???? ) ???10 ) for Algorithm 2.

Before making a comparison, we note that their result does not have a dependency on the variance of stochastic gradient (i.e. ?? 2 ), which is because they assume that the variance is also bounded by the constant (can be seen from (86) , Algorithm 2 is strictly better than that of Daneshmand et al. (2018) , which means that a higher momentum can help to find a second order stationary point faster.

<|TLDR|>

@highlight

Higher momentum parameter $\beta$ helps for escaping saddle points faster