We investigate the loss surface of neural networks.

We prove that even for one-hidden-layer networks with "slightest" nonlinearity, the empirical risks have spurious local minima in most cases.

Our results thus indicate that in general "no spurious local minim" is a property limited to deep linear networks, and insights obtained from linear networks may not be robust.

Specifically, for ReLU(-like) networks we constructively prove that for almost all practical datasets there exist infinitely many local minima.

We also present a counterexample for more general activations (sigmoid, tanh, arctan, ReLU, etc.), for which there exists a bad local minimum.

Our results make the least restrictive assumptions relative to existing results on spurious local optima in neural networks.

We complete our discussion by presenting a comprehensive characterization of global optimality for deep linear networks, which unifies other results on this topic.

Neural network training reduces to solving nonconvex empirical risk minimization problems, a task that is in general intractable.

But success stories of deep learning suggest that local minima of the empirical risk could be close to global minima.

BID5 use spherical spin-glass models from statistical physics to justify how the size of neural networks may result in local minima that are close to global.

However, due to the complexities introduced by nonlinearity, a rigorous understanding of optimality in deep neural networks remains elusive.

Initial steps towards understanding optimality have focused on deep linear networks.

This area has seen substantial recent progress.

In deep linear networks there is no nonlinear activation; the output is simply a multilinear function of the input.

BID1 prove that some shallow networks have no spurious local minima, and Kawaguchi (2016) extends this result to squared error deep linear networks, showing that they only have global minima and saddle points.

Several other works on linear nets have also appeared (Lu & Kawaguchi, 2017; Freeman & Bruna, 2017; Yun et al., 2018; Zhou & Liang, 2018; Laurent & Brecht, 2018a; b) .The theory of nonlinear neural networks (which is the actual setting of interest), however, is still in its infancy.

There have been attempts to extend the "local minima are global" property from linear to nonlinear networks, but recent results suggest that this property does not usually hold (Zhou & Liang, 2018) .

Although not unexpected, rigorously proving such results turns out to be non-trivial, forcing several authors (e.g., Safran & Shamir (2018) ; BID8 ; Wu et al. (2018) ) to make somewhat unrealistic assumptions (realizability and Gaussianity) on data.

In contrast, we prove existence of spurious local minima under the least restrictive (to our knowledge) assumptions.

Since seemingly subtle changes to assumptions can greatly influence the analysis as well as the applicability of known results, let us first summarize what is known; this will also help provide a better intuitive perspective on our results (as the technical details are somewhat involved).

There is a large and rapidly expanding literature of optimization of neural networks.

Some works focus on the loss surface BID1 Yu & Chen, 1995; Kawaguchi, 2016; Swirszcz et al., 2016; Soudry & Carmon, 2016; Xie et al., 2016; Nguyen & Hein, 2017; Safran & Shamir, 1.2 CONTRIBUTIONS AND SUMMARY OF RESULTS We summarize our key contributions more precisely below.

Our work encompasses results for both nonlinear and linear neural networks.

First, we study whether the "local minima are global" property holds for nonlinear networks.

Unfortunately, our results here are negative.

Specifically, we prove For piecewise linear and nonnegative homogeneous activation functions (e.g., ReLU), we prove in Theorem 1 that if linear models cannot perfectly fit the data, one can construct infinitely many local minima that are not global.

In practice, most datasets are not linearly fittable, hence this result gives a constructive proof of spurious local minima for generic datasets.

In contrast, several existing results either provide only one counterexample (Swirszcz et al., 2016; Zhou & Liang, 2018) , or make restrictive assumptions of realizability (Safran & Shamir, 2018; BID8 or linear separability (Laurent & Brecht, 2018a) .

This result is presented in Section 2.In Theorem 2 we tackle more general nonlinear activation functions, and provide a simple architecture (with squared loss) and dataset, for which there exists a local minimum inferior to the global minimum for a realizable dataset.

Our analysis applies to a wide range of activations, including sigmoid, tanh, arctan, ELU BID6 , SELU (Klambauer et al., 2017) , and ReLU.

Considering that realizability of data simplifies the analysis and ensures zero loss at global optima, our counterexample that is realizable and yet has a spurious local minimum is surprising, suggesting that the situation is likely worse for non-realizable data.

See Section 3 for details.

We complement our negative results by presenting the following positive result on linear networks: Assume that the hidden layers are as wide as either the input or the output, and that the empirical risk ((W j ) H+1 j=1 ) equals 0 (W H+1 W H · · · W 1 ), where 0 is a differentiable loss function and W i is the weight matrix for layer i. Theorem 4 shows if (Ŵ j ) H+1 j=1 is a critical point of , then its type of stationarity (local min/max, or saddle) is closely related to the behavior of 0 evaluated at the productŴ H+1 · · ·Ŵ 1 .

If we additionally assume that any critical point of 0 is a global minimum, Corollary 5 shows that the empirical risk only has global minima and saddles, and provides a simple condition to distinguish between them.

To the best of our knowledge, this is the most general result on deep linear networks and it subsumes several previous results, e.g., (Kawaguchi, 2016; Yun et al., 2018; Zhou & Liang, 2018; Laurent & Brecht, 2018b ).

This result is in Section 4.Notation.

For an integer a ≥ 1, [a] denotes the set of integers from 1 to a (inclusive).

For a vector v, we use [v] i to denote its i-th component, while [v]

[i] denotes a vector comprised of the first i components of v. Let 1 (·) (0 (·) ) be the all ones (zeros) column vector or matrix with size (·).

We study below whether nonlinear neural networks provably have spurious local minima.

We show in §2 and §3 that even for extremely simple nonlinear networks, one encounters spurious local minima.

We first consider ReLU and ReLU-like networks.

Here, we prove that as long as linear models cannot perfectly fit the data, there exists a local minimum strictly inferior to the global one.

Using nonnegative homogeneity, we can scale the parameters to get infinitely many local minima.

Consider a training dataset that consists of m data points.

The inputs and the outputs are of dimension d x and d y , respectively.

We aggregate these items, and write X ∈ R dx×m as the data matrix and Y ∈ R dy×m as the label matrix.

Consider the 1-hidden-layer neural network DISPLAYFORM0 .

We analyze the empirical risk with squared loss DISPLAYFORM1 F .

Next, define a class of piecewise linear nonnegative homogeneous functions h s+,s− (x) = max{s + x, 0} + min{s − x, 0},where s + > 0, s − ≥ 0 and s + = s − .

Note that ReLU and Leaky-ReLU are members of this class.

We use the shorthandX := X (C1.

DISPLAYFORM0 3) The activation function h ish s+,s− .(C1.4) The hidden layer has at least width 2: DISPLAYFORM1 Then, there is a spurious local minimum whose risk is the same as linear least squares model.

Moreover, due to nonnegative homogeneity ofh s+,s− , there are infinitely many such local minima.

Noticing that most real world datasets cannot be perfectly fit with linear models, Theorem 1 shows that when we use the activationh s+,s− , the empirical risk has bad local minima for almost all datasets that one may encounter in practice.

Although it is not very surprising that neural networks have spurious local minima, proving this rigorously is non-trivial.

We provide a constructive and deterministic proof for this problem that holds for general datasets, which is in contrast to experimental results of Safran & Shamir (2018) .

We emphasize that Theorem 1 also holds even for "slightest" nonlinearities, e.g., when s + = 1 + and s − = 1 where > 0 is small.

This suggests that the "local min is global" property is limited to the simplified setting of linear neural networks.

Existing results on squared error loss either provide one counterexample (Swirszcz et al., 2016; Zhou & Liang, 2018) , or assume realizability and Gaussian input (Safran & Shamir, 2018; BID8 .

Realizability is an assumption that the output is generated by a network with unknown parameters.

In real datasets, neither input is Gaussian nor output is generated by neural networks; in contrast, our result holds for most realistic situations, and hence delivers useful insight.

There are several results proving sufficient conditions for global optimality of nonlinear neural networks (Soudry & Carmon, 2016; Xie et al., 2016; Nguyen & Hein, 2017) .

But they rely on assumptions that the network width scales with the number of data points.

For instance, applying Theorem 3.4 of Nguyen & Hein (2017) to our network proves that ifX has linearly independent columns and other assumptions hold, then any critical point with W 2 = 0 is a global minimum.

However, linearly independent columns already imply row(X) = R m , so even linear models RX can fit any Y ; i.e., there is less merit in using a complex model to fit Y .

Theorem 1 does not make any structural assumption other than d 1 ≥ 2, and addresses the case where it is impossible to fit Y with linear models, which is much more realistic.

It is worth comparing our result with Laurent & Brecht (2018a) , who use hinge loss based classification and assume linear separability to prove "no spurious local minima" for Leaky-ReLU networks.

Their result does not contradict our theorem because the losses are different and we do not assume linear separability.

One might wonder if our theorem holds even with d 1 ≥ m. Venturi et al. (2018) showed that onehidden-layer neural networks with d 1 ≥ m doesn't have spurious valleys, hence there is no strict spurious local minima; however, due to nonnegative homogeneity ofh s+,s− we only have non-strict local minima.

Based on BID2 , one might claim that with wide enough hidden layer and random W 1 and b 1 , one can fit any Y ; however, this is not the case, by our assumption that linear models RX cannot fit Y .

Note that for any d 1 , there is a non-trivial region (measure > 0) in the parameter space where entry-wise) .

In this region, the output of neural networkŶ is still a linear combination of rows ofX, soŶ cannot fit Y ; in fact, it can only do as well as linear models.

We will see in the Step 1 of Section 2.2 that the bad local minimum that we construct "kills" d 1 − 1 neurons; however, killing many neurons is not a necessity, and it is just to simply the exposition.

In fact, any local minimum in the region W 1 X + b 1 1 T m > 0 is a spurious local minimum.

DISPLAYFORM2

The proof of the theorem is split into two steps.

First, we prove that there exist local minima (Ŵ j ,b j ) 2 j=1 whose risk value is the same as the linear least squares solution, and that there are infinitely many such minima.

Second, we will construct a tuple of parameters (W j ,b j ) 2 j=1 that has strictly smaller empirical risk than (Ŵ j ,b j ) 2 j=1 .Step 1: A local minimum as good as the linear solution.

The main idea here is to exploit the weights from the linear least squares solution, and to tune the parameters so that all inputs to hidden nodes become positive.

Doing so makes the hidden nodes "locally linear," so that the constructed (Ŵ j ,b j ) 2 j=1 that produce linear least squares estimates at the output become locally optimal.

Recall thatX = X T 1 m T ∈ R (dx+1)×m , and define a linear least squares loss 0 (R) := DISPLAYFORM0 T be the output of the linear least squares model, and similarlyȲ =WX.Let η := min {−1, 2 min iȳi }, a negative constant makingȳ i − η > 0 for all i. Define parameterŝ DISPLAYFORM1 where α > 0 is any arbitrary fixed positive constant, [W ] [dx] gives the first d x components ofW , and DISPLAYFORM2 (component-wise), given our choice of η.

Thus, all hidden node inputs are positive.

Moreover, DISPLAYFORM3 So far, we checked that (Ŵ j ,b j ) 2 j=1 has the same empirical risk as a linear least squares solution.

It now remains to show that this point is indeed a local minimum of .

To that end, we consider the perturbed parameters (Ŵ j + ∆ j ,b j + δ j ) 2 j=1 , and check their risk is always larger.

A useful point is that sinceW is a minimum of 0 (R) = DISPLAYFORM4 DISPLAYFORM5 they are aggregated perturbation terms.

We used (2) to obtain the last equality of (3).

Thus, DISPLAYFORM6 is indeed a local minimum of .

Since this is true for arbitrary α > 0, there are infinitely many such local minima.

We can also construct similar local minima by permuting hidden nodes, etc.

Step 2: A point strictly better than the local minimum.

The proof of this step is more involved.

In the previous step, we "pushed" all the input to the hidden nodes to positive side, and took advantage of "local linearity" of the hidden nodes near DISPLAYFORM7 j=1 is a spurious local minimum), we make the sign of inputs to the hidden nodes different depending on data.

To this end, we sort the indices of data points in increasing order ofȳ i ; i.e.,ȳ 1 ≤ȳ 2 ≤ · · · ≤ȳ m .

Define the set J : DISPLAYFORM8 The remaining construction is divided into two cases: J = ∅ and J = ∅, whose main ideas are essentially the same.

We present the proof for J = ∅, and defer the other case to Appendix A2 as it is rarer, and its proof, while instructive for its perturbation argument, is technically too involved.

Case 1: J = ∅. Pick any j 0 ∈ J .

We can observe that i≤j0 DISPLAYFORM9 , so thatȳ i − β < 0 for all i ≤ j 0 andȳ i − β > 0 for all i > j 0 .

Then, let γ be a constant satisfying 0 < |γ| ≤ȳ j 0 +1−ȳj 0

, whose value will be specified later.

Since |γ| is small enough, sign(ȳ i − β) = sign(ȳ i − β + γ) = sign(ȳ i − β − γ).

Now select parameters DISPLAYFORM0 Similarly, for i > j 0 ,ȳ i − β + γ > 0 and −ȳ i + β + γ < 0 results inŷ i =ȳ i + s+−s− s++s− γ.

Here, we push the outputsŷ i of the network by s+−s− s++s− γ fromȳ i , and the direction of the "push" varies depending on whether i ≤ j 0 or i > j 0 .The empirical risk for this choice of parameters is DISPLAYFORM1 , and choose small |γ| so that DISPLAYFORM2 is a spurious local minimum.

The proof of Theorem 1 crucially exploits the piecewise linearity of the activation functions.

Thus, one may wonder whether the spurious local minima seen there are an artifact of the specific nonlinearity.

We show below that this is not the case.

We provide a counterexample nonlinear network and a dataset for which a wide range of nonlinear activations result in a local minimum that is strictly inferior to the global minimum with exactly zero empirical risk.

Examples of such activation functions include popular activation functions such as sigmoid, tanh, arctan, ELU, SELU, and ReLU.We consider again the squared error empirical risk of a one-hidden-layer nonlinear neural network: DISPLAYFORM0 F , where we fix d x = d 1 = 2 and d y = 1.

Also, let h (k) (x) be the k-th derivative of h : R → R, whenever it exists at x. For short, let h and h denote the first and second derivatives.

For this network and dataset the following results hold: DISPLAYFORM1 at which equals 0.

2.

If there exist real numbers v 1 , v 2 , u 1 , u 2 ∈ R such that the following conditions hold: DISPLAYFORM2 such that the output of the network is the same as the linear least squares model, the risk DISPLAYFORM3 j=1 is a local minimum of .Theorem 2 shows that for this architecture and dataset, activations that satisfy (C2.1)-(C2.7) introduce at least one spurious local minimum.

Notice that the empirical risk is zero at the global minimum.

This means that the data X and Y can actually be "generated" by the network, which satisfies the realizability assumption that others use (Safran & Shamir, 2018; BID8 Wu et al., 2018) .

Notice that our counterexample is "easy to fit," and yet, there exists a local minimum that is not global.

This leads us to conjecture that with harder datasets, the problems with spurious local minima could be worse.

The proof of Theorem 2 can be found in Appendix A3.Discussion.

Note that the conditions (C2.1)-(C2.7) only require existence of certain real numbers rather than some global properties of activation h, hence are not as restrictive as they look.

Conditions (C2.1)-(C2.2) come from a choice of tuple (W j ,b j ) 2 j=1 that perfectly fits the data.

Condition (C2.3) is necessary for constructing (Ŵ j ,b j ) 2 j=1 with the same output as the linear least squares model, and Conditions (C2.4)-(C2.7) are needed for showing local minimality of (Ŵ j ,b j ) 2 j=1 via Taylor expansions.

The class of functions that satisfy conditions (C2.1)-(C2.7) is quite large, and includes the nonlinear activation functions used in practice.

The next corollary highlights this observation (for a proof with explicit choices of the involved real numbers, please see Appendix A5).

Corollary 3.

For the counterexample in Theorem 2, the set of activation functions satisfying conditions (C2.1)-(C2.7) include sigmoid, tanh, arctan, quadratic, ELU, and SELU.Admittedly, Theorem 2 and Corollary 3 give one counterexample instead of stating a claim about generic datasets.

Nevertheless, this example shows that for many practical nonlinear activations, the desirable "local minimum is global" property cannot hold even for realizable datasets, suggesting that the situation could be worse for non-realizable ones.

Remark: "ReLU-like" activation functions.

Recall the piecewise linear nonnegative homogeneous activation functionh s+,s− .

They do not satisfy condition (C2.7), so Theorem 2 cannot be directly applied.

Also, if s − = 0 (i.e., ReLU), conditions (C2.1)-(C2.2) are also violated.

However, the statements of Theorem 2 hold even forh s+,s− , which is shown in Appendix A6.

Recalling again s + = 1 + and s − = 1, this means that even with the "slightest" nonlinearity in activation function, the network has a global minimum with risk zero while there exists a bad local minimum that performs just as linear least squares models.

In other words, "local minima are global" property is rather brittle and can only hold for linear neural networks.

Another thing to note is that in Appendix A6, the bias parameters are all zero, for both (W j ,b j )

In this section we present our results on deep linear neural networks.

Assuming that the hidden layers are at least as wide as either the input or output, we show that critical points of the loss with a multilinear parameterization inherit the type of critical points of the loss with a linear parameterization.

As a corollary, we show that for differentiable losses whose critical points are globally optimal, deep linear networks have only global minima or saddle points.

Furthermore, we provide an efficiently checkable condition for global minimality.

Suppose the network has H hidden layers having widths d 1 , . . . , d H .

To ease notation, we set DISPLAYFORM0 The weights between adjacent layers are kept in matrices W j ∈ R dj ×dj−1 DISPLAYFORM1 , and the outputŶ of the network is given by the product of weight matrices with the data matrix: DISPLAYFORM2 j=1 be the tuple of all weight matrices, and W i:j denote the product W i W i−1 · · · W j+1 W j for i ≥ j, and the identity for i = j − 1.

We consider the empirical risk ((W j ) H+1 j=1 ), which, for linear networks assumes the form DISPLAYFORM3 where 0 is a suitable differentiable loss.

For example, when 0 (R) = DISPLAYFORM4 Remark: bias terms.

We omit the bias terms b 1 , . . .

, b H+1 here.

This choice is for simplicity; models with bias can be handled by the usual trick of augmenting data and weight matrices.

We are now ready to state our first main theorem, whose proof is deferred to Appendix A7.

Theorem 4.

Suppose that for all j, d j ≥ min{d x , d y }, and that the loss is given by (4), where 0 is differentiable on R dy×dx .

For any critical point (Ŵ j ) H+1 j=1 of the loss , the following claims hold: DISPLAYFORM0 j=1 is a saddle of .

DISPLAYFORM1 j=1 is a local min (max) of ifŴ H+1:1 is a local min (max) of 0 ; moreover, DISPLAYFORM2 j=1 is a global min (max) of if and only ifŴ H+1:1 is a global min (max) of 0 .3.

If there exists j * ∈ [H + 1] such thatŴ H+1:j * +1 has full row rank andŴ j * −1:1 has full column rank, then ∇ 0 (Ŵ H+1:1 ) = 0, so 2(a) and 2(b) hold.

Also, DISPLAYFORM3 j=1 is a local min (max) of .Let us paraphrase Theorem 4 in words.

In particular, it states that if the hidden layers are "wide enough" so that the product W H+1:1 can attain full rank and if the loss assumes the form (4) for a differentiable loss 0 , then the type (optimal or saddle point) of a critical point (Ŵ j ) H+1 j=1 of is governed by the behavior of 0 at the productŴ H+1:1 .Note that for any critical point (Ŵ j ) H+1 j=1 of the loss , either ∇ 0 (Ŵ H+1:1 ) = 0 or ∇ 0 (Ŵ H+1:1 ) = 0.

Parts 1 and 2 handle these two cases.

Also observe that the condition in Part 3 implies ∇ 0 = 0, so Part 3 is a refinement of Part 2.

A notable fact is that a sufficient condition for Part 3 isŴ H+1:1 having full rank.

For example, if d x ≥ d y , full-rankŴ H+1:1 implies rank(Ŵ H+1:2 ) = d y , whereby the condition in Part 3 holds with j * = 1.IfŴ H+1:1 is not critical for 0 , then (Ŵ j ) H+1 j=1 must be a saddle point of .

IfŴ H+1:1 is a local min/max of 0 , (Ŵ j ) H+1 j=1 is also a local min/max of .

Notice, however, that Part 2(a) does not address the case of saddle points; whenŴ H+1:1 is a saddle point of 0 , the tuple (Ŵ j ) H+1 j=1 can behave arbitrarily.

However, with the condition in Part 3, statements 2(a) and 3(a) hold at the same time, so thatŴ H+1:1 is a local min/max of 0 if and only if (Ŵ j ) H+1 j=1 is a local min/max of .

Observe that the same "if and only if" statement holds for saddle points due to their definition; in summary, the types (min/max/saddle) of the critical points (Ŵ j ) H+1 j=1 andŴ H+1:1 match exactly.

Although Theorem 4 itself is of interest, the following corollary highlights its key implication for deep linear networks.

Corollary 5.

In addition to the assumptions in Theorem 4, assume that any critical point of 0 is a global min (max).

For any critical point (Ŵ j ) Corollary 5 shows that for any differentiable loss function 0 whose critical points are global minima, the loss has only global minima and saddle points, therefore satisfying the "local minima are global" property.

In other words, for such an 0 , the multilinear re-parametrization introduced by deep linear networks does not introduce any spurious local minima/maxima; it only introduces saddle points.

Importantly, Corollary 5 also provides a checkable condition that distinguishes global minima from saddle points.

Since is nonconvex, it is remarkable that such a simple necessary and sufficient condition for global optimality is available.

DISPLAYFORM4 Our result generalizes previous works on linear networks such as Kawaguchi FORMULA2 ; Yun et al. (2018) ; Zhou & Liang (2018) , because it provides conditions for global optimality for a broader range of loss functions without assumptions on datasets.

Laurent & Brecht (2018b) proved that if DISPLAYFORM5 j=1 is a local min of , thenŴ H+1:1 is a critical point of 0 .

First, observe that this result is implied by Theorem 4.1.

So our result, which was proved in parallel and independently, is strictly more general.

With additional assumption that critical points of 0 are global minima, Laurent & Brecht (2018b) showed that "local min is global" property holds for linear neural networks; our Corollay 5 gives a simple and efficient test condition as well as proving there are only global minima and saddles, which is clearly stronger.

We investigated the loss surface of deep linear and nonlinear neural networks.

We proved two theorems showing existence of spurious local minima on nonlinear networks, which apply to almost all datasets (Theorem 1) and a wide class of activations (Theorem 2).

We concluded by Theorem 4, showing a general result studying the behavior of critical points in multilinearly parametrized functions, which unifies other existing results on linear neural networks.

Given that spurious local minima are common in neural networks, a valuable future research direction will be investigating how far local minima are from global minima in general, and how the size of the network affects this gap.

Another thing to note is that even though we showed the existence of spurious local minima in the whole parameter space, things can be different in restricted sets of parameter space (e.g., by adding regularizers).

Understanding the loss surface in such sets would be valuable.

Additionally, one can try to show algorithmic/trajectory results of (stochastic) gradient descent.

We hope that our paper will be a stepping stone to such future research.

A2 PROOF OF THEOREM 1, STEP 2, CASE 2

Case 2.

J = ∅.

We start with a lemma discussing what J = ∅ implies.

Lemma A.1.

If J = ∅, the following statements hold:1.

There are someȳ j 's that are duplicate; i.e. for some i = j,ȳ i =ȳ j .2.

Ifȳ j is non-duplicate, meaning thatȳ j−1 <ȳ j <ȳ j+1 ,ȳ j = y j holds.3.

Ifȳ j is duplicate, i:ȳi=ȳj (ȳ i − y i ) = 0 holds.4.

There exists at least one duplicateȳ j such that, for thatȳ j , there exist at least two different i's that satisfyȳ i =ȳ j andȳ i = y i .Proof We prove this by showing if any of these statements are not true, then we have J = ∅ or a contradiction.1.

If all theȳ j 's are distinct and J = ∅, by definition of J ,ȳ j = y j for all j.

This violates our assumption that linear models cannot perfectly fit Y .2.

If we haveȳ j = y j for a non-duplicateȳ j , at least one of the following statements must hold: i≤j−1 (ȳ i − y i ) = 0 or i≤j (ȳ i − y i ) = 0, meaning that j − 1 ∈ J or j ∈ J .3.

Supposeȳ j is duplicate and i:ȳi=ȳj (ȳ i − y i ) = 0.

Let k = min{i |ȳ i =ȳ j } and l = max{i |ȳ i =ȳ j }.

Then at least one of the following statements must hold: DISPLAYFORM0 4.

Since i:ȳi=ȳj (ȳ i − y i ) = 0 holds for any duplicateȳ j , ifȳ i = y i holds for one i then there must be at least two of them that satisfiesȳ i = y i .

If this doesn't hold for all duplicatē y i , with Part 2 this means thatȳ j = y j holds for all j.

This violates our assumption that linear models cannot perfectly fit Y .From Lemma A.1.4, we saw that there is a duplicate value ofȳ j such that some of the data points i satisfyȳ i =ȳ j andȳ i = y i .

The proof strategy in this case is essentially the same, but the difference is that we choose one of such duplicateȳ j , and then choose a vector v ∈ R dx to "perturb" the linear least squares solution [W ] [dx] in order to break the tie between i's that satisfiesȳ i =ȳ j andȳ i = y i .We start by defining the minimum among such duplicate valuesȳ * ofȳ j 's, and a set of indices j that satisfiesȳ j =ȳ * .ȳ * = min{ȳ j | ∃i = j such thatȳ i =ȳ j andȳ i = y i }, DISPLAYFORM1 Then, we define a subset of J * : DISPLAYFORM2 By Lemma A.1.4, cardinality of J * = is at least two.

Then, we define a special index in J * = : DISPLAYFORM3 Index j 1 is the index of the "longest" x j among elements in J * = .

Using the definition of j 1 , we can partition J * into two sets: DISPLAYFORM4 2 2 }.

For the indices in J * , we can always switch the indices without loss of generality.

So we can assume that j ≤ j 1 = max J * ≥ for all j ∈ J * ≥ and j > j 1 for all j ∈ J * < .We now define a vector that will be used as the "perturbation" to [W ] [dx] .

Define a vector v ∈ R dx , which is a scaled version of x j1 : DISPLAYFORM5 where the constants g and M are defined to be DISPLAYFORM6 The constant M is the largest x i 2 among all the indices, and g is one fourth times the minimum gap between all distinct values ofȳ i .

[dx] by a vector −αv T .

where α ∈ (0, 1] will be specified later.

Observe that DISPLAYFORM0 Recall that j ≤ j 1 = max J * ≥ for all j ∈ J * ≥ and j > j 1 for all j ∈ J * < .

We are now ready to present the following lemma: Lemma A.2.

Define DISPLAYFORM1 Proof First observe that, for any x i , |αv DISPLAYFORM2 By definition of g, we have 2g <ȳ j −ȳ i for anyȳ i <ȳ j .

Using this, we can see that DISPLAYFORM3 In words, ifȳ i andȳ j are distinct and there is an orderȳ i <ȳ j , perturbation of [W ] [dx] by −αv T does not change the order.

Also, since v is only a scaled version of x j1 , from the definitions of J * ≥ and J * DISPLAYFORM4 It is left to prove the statement of the lemma using case analysis, using the inequalities (A.1), (A.2), and (A.3).

For all i's such thatȳ i <ȳ * =ȳ j1 , DISPLAYFORM5 Similarly, for all i such thatȳ i >ȳ * =ȳ j2 , DISPLAYFORM6 For j ∈ J * ≥ (j ≤ j 1 ), we knowȳ j =ȳ * , sō DISPLAYFORM7 Also, for j ∈ J * < (j > j 1 ), DISPLAYFORM8 This finishes the case analysis and proves the first statements of the lemma.

One last thing to prove is that i>j1 DISPLAYFORM9 Recall from Lemma A.1.2 that for non-duplicateȳ j , we haveȳ j = y j .

Also by Lemma A.1.3 ifȳ j is duplicate, DISPLAYFORM10 Recall the definition of J * = = {j ∈ J * |ȳ j = y j }.

For j ∈ J * \J * = ,ȳ j = y j .

So, DISPLAYFORM11 Recall the definition of j 1 = argmax j∈J * = x j 2 .

For any other j ∈ J * = \{j 1 }, DISPLAYFORM12 , where the first ≥ sign is due to definition of j 1 , and the second is from Cauchy-Schwarz inequality.

Since x j1 and x j are distinct by assumption, they must differ in either length or direction, or both.

So, we can check that at least one of "≥" must be strict inequality, so x j1 2 2 > x j , x j1 for all j ∈ J * = \{j 1 }.

Thus, DISPLAYFORM13 Also, by Lemma A.1.3, DISPLAYFORM14 Wrapping up all the equalities, we can conclude that DISPLAYFORM15 finishing the proof of the last statement.

It is time to present the parameters (W j ,b j ) 2 j=1 , whose empirical risk is strictly smaller than the local minimum (Ŵ j ,b j ) 2 j=1 with a sufficiently small choice of α ∈ (0, 1].

Now, let γ be a constant such that DISPLAYFORM16 Its absolute value is proportional to α ∈ (0, 1], which is a undetermined number that will be specified at the end of the proof.

Since |γ| is small enough, we can check that sign(ȳ i − αv DISPLAYFORM17 Then, assign parameter values DISPLAYFORM18 With these parameter values,W DISPLAYFORM19 As we saw in Lemma A.2, for i ≤ j 1 ,ȳ i − αv DISPLAYFORM20 Similarly, for i > j 1 ,ȳ i − αv DISPLAYFORM21 Now, the squared error loss of this point is DISPLAYFORM22 Recall that DISPLAYFORM23 As seen in the definition of γ (A.4), the magnitude of γ is proportional to α.

Substituting (A.4), we can express the loss as DISPLAYFORM24 Recall that v T (x j1 − x j2 ) > 0 from (A.3).

Then, for sufficiently small α ∈ (0, 1], DISPLAYFORM25 DISPLAYFORM26 With these values, we can check thatŶ = [0 0 1], hence perfectly fitting Y , thus the loss DISPLAYFORM27

Given conditions (C2.3)-(C2.7) on v 1 , v 2 , u 1 , u 2 ∈ R, we prove below that there exists a local minimum (Ŵ j ,b j ) 2 j=1 for which the output of the network is the same as linear least squares model, and its empirical risk is ((Ŵ j ,b j ) 2 j=1 ) = .

Now assign parameter valuesŴ DISPLAYFORM0 With these values we can check thatŶ = .

It remains to show that this is indeed a local minimum of .

To show this, we apply perturbations to the parameters to see if the risk after perturbation is greater than or equal to ((Ŵ j ,b j ) 2 j=1 ).

Let the perturbed parameters bě DISPLAYFORM1 where δ 11 , δ 12 , δ 21 , δ 22 , β 1 , β 2 , 1 , 2 , and γ are small real numbers.

The next lemma rearranges the terms in ((W j ,b j ) 2 j=1 ) into a form that helps us prove local minimality of (Ŵ j ,b j ) 2 j=1 .

Appendix A4 gives the proof of Lemma A.3, which includes as a byproduct some equalities on polynomials that may be of wider interest.

Lemma A.3.

Assume there exist real numbers v 1 , v 2 , u 1 , u 2 such that conditions (C2.3)-(C2.5) hold.

Then, for perturbed parameters DISPLAYFORM2 where DISPLAYFORM3 + o(1), for i = 1, 2, and DISPLAYFORM4 + o(1), and o(1) contains terms that diminish to zero as perturbations vanish.

To make the the sum of the last three terms of (A.6) nonnegative, we need to satisfy α 1 ≥ 0 and α 2 3 − 4α 1 α 2 ≤ 0; these inequalities are satisfied for small enough perturbations because of conditions (C2.6)-(C2.7).

Thus, we conclude that DISPLAYFORM5 j=1 is a local minimum.

The goal of this lemma is to prove that DISPLAYFORM0 where o(1) contains terms that diminish to zero as perturbations decrease.

Using the perturbed parameters, DISPLAYFORM1 so the empirical risk can be expressed as DISPLAYFORM2 (A.8) So, the empirical risk (A.8) consists of three terms, one for each training example.

By expanding the activation function h using Taylor series expansion and doing algebraic manipulations, we will derive the equation (A.7) from (A.8).Using the Taylor series expansion, we can express h(v 1 + δ 11 + β 1 ) as DISPLAYFORM3 Using a similar expansion for h(v 2 + δ 21 + β 2 ), the first term of (A.8) can be written as DISPLAYFORM4 where we used u 1 h(v 1 )+u 2 h(v 2 ) = 1 3 .

To simplify notation, let us introduce the following function: DISPLAYFORM5 With this new notation t(δ 1 , δ 2 ), after doing similar expansions to the other terms of (A.8), we get DISPLAYFORM6 Before we show the lower bounds, we first present the following lemmas that will prove useful shortly.

These are simple yet interesting lemmas that might be of independent interest.

Lemma A.4.

For n ≥ 2, DISPLAYFORM7 where p n is a polynomial in a and b. All terms in p n have degree exactly n − 2.

When n = 2, DISPLAYFORM8 Proof The exact formula for p n (a, b) is as the following: DISPLAYFORM9 Using this, we can check the lemma is correct just by expanding both sides of the equation.

The rest of the proof is straightforward but involves some complicated algebra.

So, we omit the details for simplicity.

Lemma A.5.

For n 1 , n 2 ≥ 1, DISPLAYFORM10 where q n1,n2 and r n1,n2 are polynomials in a, b, c and d. All terms in q n1,n2 and r n1,n2 have degree exactly DISPLAYFORM11 Proof The exact formulas for q n1,n2 (a, b, d), q n2,n1 (c, d, b) , and r n1,n2 (a, b, c, d) are as the following: DISPLAYFORM12 Similarly, we can check the lemma is correct just by expanding both sides of the equation.

The remaining part of the proof is straightforward, so we will omit the details.

Using Lemmas A.4 and A.5, we will expand and simplify the "cross terms" part and "squared terms" part of (A.9).

For the "cross terms" in (A.9), let us split t(δ 1 , δ 2 ) into two functions t 1 and t 2 :

DISPLAYFORM13 It is easy to check that DISPLAYFORM14 Also, using Lemma A.4, we can see that DISPLAYFORM15 Consider the summation DISPLAYFORM16 We assumed that there exists a constant c > 0 such that |h (n) (v 1 )| ≤ c n n!. From this, for small enough perturbations δ 11 , δ 12 , and β 1 , we can see that the summation converges, and the summands converge to zero as n increases.

Because all the terms in p n (n ≥ 3) are of degree at least one, we can thus write DISPLAYFORM17 So, for small enough δ 11 , δ 12 , and β 1 , the term DISPLAYFORM18 dominates the summation.

Similarly, as long as δ 21 , δ 22 , and β 2 are small enough, the summation DISPLAYFORM19 .

In conclusion, for small enough perturbations, DISPLAYFORM20 Now, it is time to take care of the "squared terms."

We will express the terms as This time, we split t(δ 1 , δ 2 ) in another way, this time into three parts: DISPLAYFORM21 so that t(δ 1 , δ 2 ) = t 3 + t 4 (δ 1 ) + t 5 (δ 2 ).

DISPLAYFORM22 as seen in (A.10).

Next, we have (A.14) when perturbations are small enough.

We again used Lemma A.4 in the second equality sign, and the facts that p n1+n2 (·) = o(1) whenever n 1 + n 2 > 2 and that p 2 (·) = 1 2 .

In a similar way, DISPLAYFORM23 DISPLAYFORM24 Lastly, A.16) where the second equality sign used Lemma A.5 and the third equality sign used the facts that q n1,n2 (·) = o(1) and r n1,n2 (·) = o(1) whenever n 1 + n 2 > 2, and that q 1,1 (·) = 0 and r 1,1 (·) = 1 2 .

If we substitute (A.13), (A.14), (A.15), and (A.16) into (A.12), DISPLAYFORM25 DISPLAYFORM26 We are almost done.

If we substitute (A.10), (A.11), and (A.17) into (A.9), we can get DISPLAYFORM27 which is the equation (A.7) that we were originally aiming to show.

For the proof of this corollary, we present the values of real numbers that satisfy assumptions (C2.1)-(C2.7), for each activation function listed in the corollary: sigmoid, tanh, arctan, exponential linear units (ELU, Clevert et al. (2015) ), scaled exponential linear units (SELU, Klambauer et al. (2017) ).To remind the readers what the assumptions were, we list the assumptions again.

For (C2.1)-(C2.2), there exist real numbers v 1 , v 2 , v 3 , v 4 ∈ R such that DISPLAYFORM0 For (C2.3)-(C2.7), there exist real numbers v 1 , v 2 , u 1 , u 2 ∈ R such that the following assumptions hold: DISPLAYFORM1 ).For each function, we now present the appropriate real numbers that satisfy the assumptions.

When h is sigmoid, DISPLAYFORM0 Assumptions (C2.1)-(C2.2) are satisfied by DISPLAYFORM1 and assumptions (C2.3)-(C2.7) are satisfied by DISPLAYFORM2 Among them, (C2.4)-(C2.5) follow because sigmoid function is an real analytic function Krantz & Parks (2002

When h is quadratic, assumptions (C2.1)-(C2.2) are satisfied by DISPLAYFORM0 and assumptions (C2.3)-(C2.7) are satisfied by DISPLAYFORM1 In the case of s − > 0, assumptions (C2.1)-(C2.2) are satisfied by DISPLAYFORM2 The rest of the proof can be done in exactly the same way as the proof of Theorem 2.1, provided in Appendix A3.For s − = 0, which corresponds to the case of ReLU, define parameters DISPLAYFORM3 We can check thath DISPLAYFORM4

Assumptions (C2.3)-(C2.6) are satisfied by DISPLAYFORM0 Assign parameter valueŝ DISPLAYFORM1 It is easy to compute that the output of the neural network isŶ = .

Now, it remains to show that this is indeed a local minimum of .

To show this, we apply perturbations to the parameters to see if the risk after perturbation is greater than or equal to ((Ŵ j ,b j ) 2 j=1 ).

Let the perturbed parameters bě DISPLAYFORM2 where δ 11 , δ 12 , δ 21 , δ 22 , β 1 , β 2 , 1 , 2 , and γ are small enough real numbers.

Using the perturbed parameters, DISPLAYFORM3 so the empirical risk can be expressed as DISPLAYFORM4 Published as a conference paper at ICLR 2019To simplify notation, let us introduce the following function: DISPLAYFORM5 It is easy to check that DISPLAYFORM6

Before we start, note the following partial derivatives, which can be computed using straightforward matrix calculus: DISPLAYFORM0 For Part 1, we must show that if ∇ 0 (Ŵ H+1:1 ) = 0 then (Ŵ j )

j=1 is a saddle point of .

Thus, we show that (Ŵ j ) H+1 j=1 is neither a local minimum nor a local maximum.

More precisely, for each j, let B (W j ) be an -Frobenius-norm-ball centered at W j , and H+1 j=1 B (W j ) their Cartesian product.

We wish to show that for every > 0, there exist tuples (P j ) DISPLAYFORM0 To prove (A.18), we exploit ((Ŵ j ) H+1 j=1 ) = 0 (Ŵ H+1:1 ), and the assumption ∇ 0 (Ŵ H+1:1 ) = 0.

The key idea is to perturb the tuple (Ŵ j ) H+1 j=1 so that the directional derivative of 0 along P H+1:1 − W H+1:1 is positive.

Since 0 is differentiable, if P H+1:1 −Ŵ H+1:1 is small, then DISPLAYFORM1 Similarly, we can show ((Q j ) H+1 j=1 ) < ((Ŵ j ) H+1 j=1 ).

The key challenge lies in constructing these perturbations; we outline our approach below; this construction may be of independent interest too.

For this section, we assume that d x ≥ d y for simplicity; the case d y ≥ d x is treated in Appendix A7.2.Since ∇ 0 (Ŵ H+1:1 ) = 0, col(∇ 0 (Ŵ H+1:1 )) ⊥ must be a strict subspace of R dy .

Consider ∂ /∂W 1 at a critical point to see that (Ŵ H+1:2 ) T ∇ 0 (Ŵ H+1:1 ) = 0, so col(Ŵ H+1:2 ) ⊆ col(∇ 0 (Ŵ H+1:1 )) ⊥ R dy .

This strict inclusion implies rank(Ŵ H+1:2 ) < d y ≤ d 1 , so that null(Ŵ H+1:2 ) is not a trivial subspace.

Moreover, null(Ŵ H+1:2 ) ⊇ null(Ŵ H:2 ) ⊇ · · · ⊇ null(Ŵ 2 ).

We can split the proof into two cases: null(Ŵ H+1:2 ) = null(Ŵ H:2 ) and null(Ŵ H+1:2 ) = null(Ŵ H:2 ).

Recall that W 1 ∈ R d1×dx .

Given R ∈ R dy×dx , we can fill the first d y rows of W 1 with R and let any other entries be zero.

For all the other matrices W 2 , . . .

, W H+1 , we put ones to the diagonal entries while putting zeros to all the other entries.

We can check that, by this construction, R = W H+1:1 for this given R. DISPLAYFORM2 dy×dx , we can fill the first d x columns of W H+1 with R and let any other entries be zero.

For all the other matrices W 1 , . . .

, W H , we put ones to the diagonal entries while putting zeros to all the other entries.

By this construction, R = W H+1:1 for given R.Once this fact is given, by ((W j ) ).

To show thatŴ H+1:1 is a local min of 0 , we have to show there exists a neighborhood ofŴ H+1:1 such that, any point R in that neighborhood satisfies 0 (R) ≥ 0 (Ŵ H+1:1 ).

To prove this, we state the following lemma: Lemma A.6.

Suppose A :=Ŵ H+1:j * +1 has full row rank and B :=Ŵ j * −1:1 has full column rank.

Then, any R satisfying R −Ŵ H+1:1 F ≤ σ min (A)σ min (B) can be decomposed into R = V H+1:1 , where DISPLAYFORM3 and V j =Ŵ j for j = j * .

Also, V j −Ŵ j F ≤ for all j.

Proof Since A :=Ŵ H+1:j * +1 has full row rank and B :=Ŵ j * −1:1 has full column rank, σ min (A) > 0, σ min (B) > 0, and AA T and B T B are invertible.

Consider any R satisfying R −Ŵ H+1:1 F ≤ σ min (A)σ min (B) .

Given the definitions of V j 's in the statement of the lemma, we can check the identity that R = V H+1:1 by Therefore, DISPLAYFORM4 · σ min (A)σ min (B) = .The lemma shows that for any R = V H+1:1 satisfying R −Ŵ H+1:1 F ≤ σ min (A)σ min (B) , we have 0 (R) = 0 (V H+1:1 ) = ((V j ) H+1 j=1 ) ≥ ((Ŵ j ) H+1 j=1 ) = 0 (Ŵ H+1:1 ).

We can prove the local maximum part by a similar argument.

<|TLDR|>

@highlight

We constructively prove that even the slightest nonlinear activation functions introduce spurious local minima, for general datasets and activation functions.