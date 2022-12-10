The weight initialization and the activation function of deep neural networks have a crucial impact on the performance of the training procedure.

An inappropriate selection can lead to the loss of information of the input during forward propagation and the exponential vanishing/exploding of gradients during back-propagation.

Understanding the theoretical properties of untrained random networks is key to identifying which deep networks may be trained successfully as recently demonstrated by Schoenholz et al. (2017) who showed that for deep feedforward neural networks only a specific choice of hyperparameters known as the `edge of chaos' can lead to good performance.

We complete this analysis by providing quantitative results showing that, for a class of ReLU-like activation functions, the information propagates indeed deeper for an initialization at the edge of chaos.

By further extending this analysis, we identify a class of activation functions that improve the information propagation over ReLU-like functions.

This class includes the Swish activation, $\phi_{swish}(x) = x \cdot \text{sigmoid}(x)$, used in Hendrycks & Gimpel (2016), Elfwing et al. (2017) and Ramachandran et al. (2017).

This provides a theoretical grounding for the excellent empirical performance of $\phi_{swish}$ observed in these contributions.

We complement those previous results by illustrating the benefit of using a random initialization on the edge of chaos in this context.

Deep neural networks have become extremely popular as they achieve state-of-the-art performance on a variety of important applications including language processing and computer vision; see, e.g., BID8 .

The success of these models has motivated the use of increasingly deep networks and stimulated a large body of work to understand their theoretical properties.

It is impossible to provide here a comprehensive summary of the large number of contributions within this field.

To cite a few results relevant to our contributions, BID11 have shown that neural networks have exponential expressive power with respect to the depth while BID14 obtained similar results using a topological measure of expressiveness.

We follow here the approach of BID14 and BID16 by investigating the behaviour of random networks in the infinite-width and finite-variance i.i.d.

weights context where they can be approximated by a Gaussian process as established by BID10 and BID9 .In this paper, our contribution is two-fold.

Firstly, we provide an analysis complementing the results of BID14 and BID16 and show that initializing a network with a specific choice of hyperparameters known as the 'edge of chaos' is linked to a deeper propagation of the information through the network.

In particular, we establish that for a class of ReLU-like activation functions, the exponential depth scale introduced in BID16 is replaced by a polynomial depth scale.

This implies that the information can propagate deeper when the network is initialized on the edge of chaos.

Secondly, we outline the limitations of ReLU-like activation functions by showing that, even on the edge of chaos, the limiting Gaussian Process admits a degenerate kernel as the number of layers goes to infinity.

Our main result (4) gives sufficient conditions for activation functions to allow a good 'information flow' through the network (Proposition 4) (in addition to being non-polynomial and not suffering from the exploding/vanishing gradient problem).

These conditions are satisfied by the Swish activation φ swish (x) = x · sigmoid(x) used in BID4 , BID2 and BID15 .

In recent work, BID15 used automated search techniques to identify new activation functions and found experimentally that functions of the form φ(x) = x · sigmoid(βx) appear to perform indeed better than many alternative functions, including ReLU.

Our paper provides a theoretical grounding for these results.

We also complement previous empirical results by illustrating the benefits of an initialization on the edge of chaos in this context.

All proofs are given in the Supplementary Material.

We use similar notations to those of BID14 and BID9 .

Consider a fully connected random neural network of depth L, widths (N l ) 1≤l≤L , weights W 2 ) denotes the normal distribution of mean µ and variance σ 2 .

For some input a ∈ R d , the propagation of this input through the network is given for an activation function φ : R → R by (1)Throughout the paper we assume that for all l the processes y l i (.) are independent (across i) centred Gaussian processes with covariance kernels κ l and write accordingly y DISPLAYFORM0 .

This is an idealized version of the true processes corresponding to choosing N l−1 = +∞ (which implies, using Central Limit Theorem, that y l i (a) is a Gaussian variable for any input a).

The approximation of y l i (.) by a Gaussian process was first proposed by BID12 in the single layer case and has been recently extended to the multiple layer case by BID9 and BID10 .

We recall here the expressions of the limiting Gaussian process kernels.

For any input DISPLAYFORM1 where F φ is a function that depends only on φ.

This gives a recursion to calculate the kernel κ l ; see, e.g., BID9 for more details.

We can also express the kernel κ l in terms of the correlation c l ab in the l th layer used in the rest of this paper DISPLAYFORM2 where q l−1 a DISPLAYFORM3 b , is the variance, resp.

correlation, in the (l − 1) th layer and Z 1 , Z 2 are independent standard Gaussian random variables.

when it propagates through the network.

q l a is updated through the layers by the recursive formula q l a = F (q l−1 a ), where F is the 'variance function' given by DISPLAYFORM4 Throughout the paper, Z, Z 1 , Z 2 will always denote independent standard Gaussian variables.

We analyze here the limiting behaviour of q L a and c L a,b as the network depth L goes to infinity under the assumption that φ has a second derivative at least in the distribution sense 1 .

From now onwards, we will also assume without loss of generality that c 1 ab ≥ 0 (similar results can be obtained straightforwardly when c 1 ab ≤ 0).

We first need to define the Domains of Convergence associated with an activation function φ.

Remark : Typically, q in Definition 1 is a fixed point of the variance function defined in equation 2.

Therefore, it is easy to see that for any (σ b , σ w ) such that F is increasing and admits at least one fixed point, we have K φ,corr (σ b , σ w ) ≥ q where q is the minimal fixed point; i.e. q := min{x : F (x) = x}. Thus, if we re-scale the input data to have q 1 a ≤ q, the variance q l a converges to q. We can also re-scale the variance σ w of the first layer (only) to assume that q 1 a ≤ q for all inputs a. The next result gives sufficient conditions on (σ b , σ w ) to be in the domains of convergence of φ.

DISPLAYFORM0 DISPLAYFORM1 The proof of Proposition 1 is straightforward.

We prove that sup F (x) = σ 2 w M φ and then apply the Banach fixed point theorem; similar ideas are used for C φ,δ .Example : For ReLU activation function, we have M ReLU = 2 and C ReLU,δ ≤ 1 for any δ > 0.

almost surely and the outputs of the network are constant functions.

FIG3 illustrates this behaviour for d = 2 for ReLU and Tanh using a network of depth L = 10 with N l = 100 neurons per layer.

The draws of outputs of these networks are indeed almost constant.

To refine this convergence analysis, BID16 established the existence of q and c such that |q l a −q| ∼ e −l/ q and |c l ab −1| ∼ e −l/ c when fixed points exist.

The quantities q and c are called 'depth scales' since they represent the depth to which the variance and correlation can propagate without being exponentially close to their limits.

More precisely, if we write DISPLAYFORM0 ] then the depth scales are given by r = − log(α) −1 and c = − log(χ 1 ) −1 .

The equation χ 1 = 1 corresponds to an infinite depth scale of the correlation.

It is called the edge of chaos as it separates two phases: an ordered phase where the correlation converges to 1 if χ 1 < 1 and a chaotic phase where χ 1 > 1 and the correlations do not converge to 1.

In this chaotic regime, it has been observed in BID16 that the correlations converge to some random value c < 1 when φ(x) = Tanh(x) and that c is independent of the correlation between the inputs.

This means that very close inputs (in terms of correlation) lead to very different outputs.

Therefore, in the chaotic phase, the output function of the neural network is non-continuous everywhere.

Definition 2.

For (σ b , σ w ) ∈ D φ,var , let q be the limiting variance 2 .

The Edge of Chaos, hereafter EOC, is the set of values of (σ b , σ w ) satisfying DISPLAYFORM1 To further study the EOC regime, the next lemma introduces a function f called the 'correlation function' simplifying the analysis of the correlations.

It states that the correlations have the same asymptotic behaviour as the time-homogeneous dynamical system c DISPLAYFORM2 The condition on φ in Lemma 1 is violated only by activation functions with exponential growth (which are not used in practice), so from now onwards, we use this approximation in our analysis.

Note that being on the EOC is equivalent to (σ b , σ w ) satisfying f (1) = 1.

In the next section, we analyze this phase transition carefully for a large class of activation functions. (as we will see later EOC = {(0, √ 2)} for ReLU).

Unlike the output in FIG3 , this output displays much more variability.

However, we will prove here that the correlations still converges to 1 even in the EOC regime, albeit at a slower rate.

We consider activation functions φ of the form: φ(x) = λx if x > 0 and φ(x) = βx if x ≤ 0.

ReLU corresponds to λ = 1 and β = 0.

For this class of activation functions, we see (Proposition 2) that the variance is unchanged (q l a = q 1 a ) on the EOC, so that q does not formally exist in the sense that the limit of q l a depends on a. However, this does not impact the analysis of the correlations.

Proposition 2.

Let φ be a ReLU-like function with λ and β defined above.

Then for any σ w < 2 λ 2 +β 2 and DISPLAYFORM0 )} and, on the EOC, F (x) = x for any x ≥ 0.This class of activation functions has the interesting property of preserving the variance across layers when the network is initialized on the EOC.

However, we show in Proposition 3 below that, even in the EOC regime, the correlations converge to 1 but at a slower rate.

We only present the result for ReLU but the generalization to the whole class is straightforward.

Example : ReLU: The EOC is reduced to the singleton (σ FORMULA4 also performed a similar analysis by using the "Scaled Exponential Linear Unit" activation (SELU) that makes it possible to center the mean and normalize the variance of the post-activation φ(y).

The propagation of the correlation was not discussed therein either.

In the next result, we present the correlation function corresponding to ReLU networks.

This was first obtained in BID0 .We present an alternative derivation of this result and further show that the correlations converge to 1 at a polynomial rate of 1/l 2 instead of an exponential rate.

Proposition 3 (ReLU kernel).

Consider a ReLU network with parameters (σ DISPLAYFORM1

We now introduce a set of sufficient conditions for activation functions which ensures that it is then possible to tune (σ b , σ w ) to slow the convergence of the correlations to 1.

This is achieved by making the correlation function f sufficiently close to the identity function.

Proposition 4 (Main Result).

Let φ be an activation function.

Assume that (i) φ(0) = 0, and φ has right and left derivatives in zero and φ (0 + ) = 0 or φ (0 − ) = 0, and there DISPLAYFORM0 , the function F with parameters (σ b , σ w ) ∈ EOC is non-decreasing and lim σ b →0 q = 0 where q is the minimal fixed point of F , q := inf{x : DISPLAYFORM1 Note that ReLU does not satisfy the condition (ii) since the EOC in this case is the singleton (σ 2 b , σ 2 w ) = (0, 2).

The result of Proposition 4 states that we can make f (x) close to x by considering σ b → 0.

However, this is under condition (iii) which states that lim σ b →0 q = 0.

Therefore, practically, we cannot take σ b too small.

One might wonder whether condition (iii) is necessary for this result to hold.

The next lemma shows that removing this condition results in a useless class of activation functions.

The next proposition gives sufficient conditions for bounded activation functions to satisfy all the conditions of Proposition 4.

DISPLAYFORM2 , xφ(x) > 0 and xφ (x) < 0 for x = 0, and φ satisfies (ii) in Proposition 4.

Then, φ satisfies all the conditions of Proposition 4.The conditions in Proposition 5 are easy to verify and are, for example, satisfied by Tanh and Arctan.

We can also replace the assumption "φ satisfies (ii) in Proposition 4" by a sufficient condition (see Proposition 7 in the Supplementary Material).

Tanh-like activation functions provide better information flow in deep networks compared to ReLU-like functions.

However, these functions suffer from the vanishing gradient problem during back-propagation; see, e.g., BID13 and BID7 .

Thus, an activation function that satisfies the conditions of Proposition 4 (in order to have a good 'information flow') and does not suffer from the vanishing gradient issue is expected to perform better than ReLU.

Swish is a good candidate.

DISPLAYFORM3 1+e −x satisfies all the conditions of Proposition 4.It is clear that Swish does not suffer from the vanishing gradient problem as it has a gradient close to 1 for large inputs like ReLU.

FIG15 (a) displays f for Swish for different values of σ b .

We see that f is indeed approaching the identity function when σ b is small, preventing the correlations from converging to 1.

FIG15 (b) displays a draw of the output of a neural network of depth 30 and width 100 with Swish activation, and σ b = 0.2.

The outputs displays much more variability than the ones of the ReLU network with the same architecture.

We present in TAB0 some values of (σ b , σ w ) on the EOC as well as the corresponding limiting variance for Swish.

As condition (iii) of Proposition 4 is satisfied, the limiting variance q decreases with σ b .

Other activation functions that have been shown to outperform empirically ReLU such as ELU BID1 ), SELU BID6 ) and Softplus also satisfy the conditions of Proposition 4 (see Supplementary Material for ELU).

The comparison of activation functions satisfying the conditions of Proposition 4 remains an open question.

We demonstrate empirically our results on the MNIST dataset.

In all the figures below, we compare the learning speed (test accuracy with respect to the number of epochs/iterations) for different activation functions and initialization parameters.

We use the Adam optimizer with learning rate lr = 0.001.

The Python code to reproduce all the experiments will be made available on-line. .

In Figure 5 , we compare the learning speed of a Swish network for different choices of random initialization.

Any initialization other than on the edge of chaos results in the optimization algorithm being stuck eventually at a very poor test accuracy of ∼ 0.1 as the depth L increases (equivalent to selecting the output uniformly at random).

To understand what is happening in this case, let us recall how the optimization algorithm works.

Let

is the output of the network, and is the categorical cross-entropy loss.

In the ordered phase, we know that the output converges exponentially to a fixed value (same value for all X i ), thus a small change in w and b will not change significantly the value of the loss function, therefore the gradient is approximately zero and the gradient descent algorithm will be stuck around the initial value.

from the vanishing gradient problem.

Consequently, we expect Tanh to perform better than ReLU for shallow networks as opposed to deep networks, where the problem of the vanishing gradient is not encountered.

Numerical results confirm this fact.

FIG18 shows curves of validation accuracy with confidence interval 90% (30 simulations).

For depth 5, the learning algorithm converges faster for Tanh compared to ReLu.

However, for deeper networks (L ≥ 40), Tanh is stuck at a very low test accuracy, this is due to the fact that a lot of parameters remain essentially unchanged because the gradient is very small.

We have complemented here the analysis of BID16 which shows that initializing networks on the EOC provides a better propagation of information across layers.

In the ReLU case, such an initialization corresponds to the popular approach proposed in BID3 .

However, even on the EOC, the correlations still converge to 1 at a polynomial rate for ReLU networks.

We have obtained a set of sufficient conditions for activation functions which further improve information propagation when the parameters (σ b , σ w ) are on the EOC.

The Tanh activation satisfied those conditions but, more interestingly, other functions which do not suffer from the vanishing/exploding gradient problems also verify them.

This includes the Swish function used in BID4 , BID2 and promoted in BID15 but also ELU Clevert et al. (2016) .Our results have also interesting implications for Bayesian neural networks which have received renewed attention lately; see, e.g., Hernandez-Lobato & Adams FORMULA4 and BID9 .

They show that if one assigns i.i.d.

Gaussian prior distributions to the weights and biases, the resulting prior distribution will be concentrated on close to constant functions even on the EOC for ReLU-like activation functions.

To obtain much richer priors, our results indicate that we need to select not only parameters (σ b , σ w ) on the EOC but also an activation function satisfying Proposition 4.

We provide in the supplementary material the proofs of the propositions presented in the main document, and we give additive theoretical and experimental results.

For the sake of clarity we recall the propositions before giving their proofs.

A.1 CONVERGENCE TO THE FIXED POINT: PROPOSITION 1 DISPLAYFORM0 Proof.

To abbreviate the notation, we use q l := q l a for some fixed input a. Convergence of the variances: We first consider the asymptotic behaviour of q l = q l a .

Recall that q l = F (q l−1 ) where, DISPLAYFORM1 The first derivative of this function is given by: DISPLAYFORM2 where

Using the condition on φ, we see that for σ DISPLAYFORM0 , the function F is a contraction mapping, and the Banach fixed-point theorem guarantees the existence of a unique fixed point q of F , with lim l→+∞ q l = q. Note that this fixed point depends only on F , therefore, this is true for any input a, and K φ,var (σ b , σ w ) = ∞.Convergence of the covariances: Since M φ < ∞, then for all a, b ∈ R d there exists l 0 such that, for all l > l 0 , | q l a − q l b | < δ.

Let l > l 0 , using Gaussian integration by parts, we have DISPLAYFORM1 We cannot use the Banach fixed point theorem directly because the integrated function here depends on l through q l .

For ease of notation, we write c l := c l ab , we have DISPLAYFORM2 l is a Cauchy sequence and it converges to a limit c ∈ [0, 1] .At the limit DISPLAYFORM3 The derivative of this function is given by DISPLAYFORM4 By assumption on φ and the choice of σ w , we have sup x |f (x)| < 1, so that f is a contraction, and has a unique fixed point.

Since f (1) = 1, c = 1.

The above result is true for any a, b, therefore, DISPLAYFORM5 As an illustration we plot in FIG3 the variance for three different inputs with (σ b , σ w ) = (1, 1), as a function of the layer l.

In this example, the convergence for Tanh is faster than that of ReLU.

DISPLAYFORM6 where u 2 (x) := xZ 1 + √ 1 − x 2 Z 2 .

The first term goes to zero uniformly in x using the condition on φ and Cauchy-Schwartz inequality.

As for the second term, it can be written as DISPLAYFORM7 again, using Cauchy-Schwartz and the condition on φ, both terms can be controlled uniformly in x by an integrable upper bound.

We conclude using the Dominated convergence.

Proposition 2.

Let φ be a ReLU-like function with λ and β defined above.

Then for any σ w < 2 λ 2 +β 2 and DISPLAYFORM0 )} and, on the EOC, F (x) = x for any x ≥ 0.Proof.

We write q l = q l a throughout the proof.

Note first that the variance satisfies the recursion: DISPLAYFORM1 For all σ w < 2 λ 2 +β 2 , q = σ Proof.

In this case the correlation function f is given by DISPLAYFORM2 • Let x ∈ [0, 1], note that f is differentiable and satisfies, DISPLAYFORM3 which is also differentiable.

Simple algebra leads to DISPLAYFORM4 Since arcsin (x) = 1 √ 1−x 2 and f (0) = 1/2, DISPLAYFORM5 We conclude using the fact that arcsin = x arcsin + √ 1 − x 2 and f (1) = 1.• We first derive a Taylor expansion of f near 1.

Consider the change of variable x = 1 − t 2 with t close to 0, then DISPLAYFORM6 we obtain that DISPLAYFORM7 DISPLAYFORM8 l < c l+1 then by taking the image by f (which is increasing because f ≥ 0) we have that c l+1 < c l+2 , and we know that c 1 = f (c 0 ) ≥ c 0 , so by induction the sequence c l is increasing, and therefore it converges (because it is bounded) to the fixed point of f which is 1.

Now let γ l := 1 − c l ab for a, b fixed.

We note s = 2 √ 2 3π , from the series expansion we have that γ l+1 = γ l − sγ DISPLAYFORM9 DISPLAYFORM10 Assume that (i) φ(0) = 0, and φ has right and left derivatives in zero and at least one of them is different from zero (φ (0 + ) = 0 or φ (0 − ) = 0), and there exists K > 0 such that DISPLAYFORM11 , the function F with parameters (σ b , σ w,EOC ) is non-decreasing and lim σ b →0 q = 0 where q is the minimal fixed point of F , q := inf{x : DISPLAYFORM12 Proof.

We first prove that K φ,var (σ b , σ w ) ≥ q. We assume that σ b > 0, the case σ b = 0 is trivial since in this case q = 0 (the output of the network is zero in this case).Since F is continuous and DISPLAYFORM13 Using the fact that F is non-decreasing for any input a such that q 1 a ≤ q, we have q l is increasing and converges to the fixed point q. Therefore K φ,var (σ b , σ w ) ≥ q. Now we prove that on the edge of chaos, we have DISPLAYFORM14 The EOC equation is given by σ 2 w E[φ ( √ qZ) 2 ] = 1.

By taking the limit σ b → 0 on the edge of chaos, and using the fact that lim σ b →0 q = 0, we have σ DISPLAYFORM15 so that by taking the limit σ b → 0, and using the dominated convergence theorem, we have that DISPLAYFORM16 q + 1 and equation 6 holds.

Finally since f is strictly convex, for all DISPLAYFORM17 q , we conclude using the fact that lim σ b →0 DISPLAYFORM18 Note however that for all σ b > 0, if (σ b , σ w ) ∈ EOC, for any inputs a, b, we have lim l→∞ c l a,b = 1.

Indeed, since f is usually strictly convex (otherwise, f would be equal to identity on at least a segment of [0, 1]) and f (1) = 1, we have that f is a contraction (because f ≥ 0), therefore the correlation converges to the unique fixed point of f which is 1.

Therefore, in most of the cases, the result of Proposition 4 should be seen as a way of slowing down the convergence of the correlation to 1.

Proof.

Using the convexity of f and the result of Proposition 4, we have in the limit σ b → 0, DISPLAYFORM19 2 which implies that var(φ ( √ qZ)) = 0.

Therefore there exists a constant a 1 such that φ ( √ qZ) = a 1 almost surely.

This implies φ = a 1 almost everywhere.

Proposition 5.

Let φ be a bounded function such that φ(0) = 0, φ (0) > 0, φ (x) ≥ 0, φ(−x) = −φ(x), xφ(x) > 0 and xφ (x) < 0 for x = 0, and φ satisfies (ii) in Proposition 4.

Then, φ satisfies all the conditions of Proposition 4.Proof.

Let φ be an activation function that satisfies the conditions of Proposition 5.(i) we have φ(0) = 0 and φ (0) > 0.

Since φ is bounded and 0 < φ (0) < ∞, then there exists K such that DISPLAYFORM20 (ii) The condition (ii) is satisfied by assumption.(iii) Let σ b > 0 and σ w > 0.

Using equation 3 together with φ > 0, we have F (x) ≥ 0 so that F is non-decreasing.

Moreover, we have DISPLAYFORM21 DISPLAYFORM22 .

Now let's prove that the function DISPLAYFORM23 is increasing near 0 which means it is an injection near 0, this is sufficient to conclude (because we take q to be the minimal fixed point).

After some calculus, we have DISPLAYFORM24 Using Taylor expansion near 0, after a detailed but unenlightening calculation the numerator is equal to −2φ (0) DISPLAYFORM25 , therefore the function e is increasing near 0.(iv) Finally, using the notations U 1 := √ qZ 1 and U 2 (x) = √ q(xZ 1 + √ 1 − x 2 Z 2 ), the first and second derivatives of the correlation function are given by DISPLAYFORM26 where we used Gaussian integration by parts.

Let x > 0, we have that DISPLAYFORM27 where we used the fact that (Z 1 , Z 2 ) = (−Z 1 , −Z 2 ) (in distribution) and φ (−y) = −φ (y) for any y.

Using xφ (x) ≤ 0, we have 1 {u1≥0} φ (u 1 ) ≤ 0.

We also have for all y > 0, E[φ (U 2 (x))|U 1 = y] < 0, this is a consequence of the fact that φ is an odd function and that for x > 0 and y > 0, the mapping z 2 → xy + √ 1 − x 2 z 2 moves the center of the Gaussian distribution to a strictly positive number, we conclude that f (x) > 0 almost everywhere and assumption (iii) of Proposition 4 is verified.

Proof.

To abbreviate notation, we note φ := φ Swish = xe x /(1 + e x ) and h := e x /(1 + e x ) is the Sigmoid function.

This proof should be seen as a sketch of the ideas and not a rigourous proof.• we have φ(0) = 0 and φ (0) = • As illustrated in TAB0 in the main text, it is easy to see numerically that (ii) is satisfied.

Moreover, we observe that lim σ b →0 q = 0, which proves the second part of the (iii).• Now we prove that F > 0, we note g(x) := xφ (x)φ(x).

We have DISPLAYFORM28 Define G by DISPLAYFORM29 which holds true for any positive number x. We thus have g(x) > G(x) for all real numbers x.

Therefore E[g( √ xZ)] > 0 almost everywhere and F > 0.

The second part of (iii) was already proven above.• Let σ b > 0 and σ w > 0 such that q exists.

Recall that DISPLAYFORM30 In FIG3 , we show the graph of E[φ (U 1 )φ (U 2 (x))] for different values of q (from 0.1 to 10, the darkest line being for q = 10).

A rigorous proof can be done but is omitted here.

We observe that f has very small values when q is large, this is a result of the fact that φ is concentrated around 0.Remark :

On the edge of chaos, we have σ DISPLAYFORM31 this yields DISPLAYFORM32 The term E[φ ( √ qZ)φ( √ qZ)] is very small compared to 1 (∼ 0.01), therefore F (q) ≈ 1.Notice also that the theoretical results corresponds to the equivalent Gaussian process, which is just an approximation of the neural network.

Thus, using a value of (σ b , σ w ) close to the EOC should not essentially change the quality of the result.

We can replace the conditions "φ satisfies (ii)" in Proposition 5 by a sufficient condition.

However, this condition is not satisfied by Tanh.

Proposition 7.

Let φ be a bounded function such that DISPLAYFORM0 , xφ(x) > 0 and xφ (x) < 0 for x = 0, and |Eφ (xZ) 2 | |x| −2β for large x and some β ∈ (0, 1).

Then, φ satisfies all the conditions of Proposition 4.Proof.

Let φ be an activation function that satisfies the conditions of Proposition 7.

The proof is similar to the one of 5, we only need to show that having |Eφ (xZ) 2 | |x| −2β for large x and some β ∈ (0, 1) implies that (ii) of 4 is verified.

We have that σ −β , so that we can make the term σ 2 w |Eφ ( √ qZ) 2 | take any value between 0 and ∞. Therefore, there exists σ w such that (σ b , σ w ) ∈ EOC, and assumption (ii) of Proposition 4 holds.

In the proof of Proposition 5, we used the condition on φ (odd function) to prove that f > 0, however, in some cases when we can explicitly calculate f , we do not need φ to be defined.

This is the case for Hard-Tanh, which is a piecewise-linear version of Tanh.

We give an explicit calculation of f for the Hard-Tanh activation function which we note HT in what follows.

We compare the performance of HT and Tanh based on a metric which we will define later.

HT is given by DISPLAYFORM0 Recall the propagation of the variance q DISPLAYFORM1 where HT is the Hard-Tanh function.

We have DISPLAYFORM2 This yields DISPLAYFORM3 where DISPLAYFORM4 -EDGE OF CHAOS :To study the correlation behaviour, we will assume that the variance converges to q. We have E(HT ( √ qZ) 2 ) = E(1 − FIG3 shows the EOC curve (condition (ii) is satisfied).

FIG3 shows that is non-decreasing and FIG3 illustrates the fact that lim σ b →0 q = 0.

Finally, FIG3 shows that function f is convex.

Although the figures of F and f are shown just for one value of (σ b , σ w ), the results are true for any value of (σ b , σ w ) on the EOC.

TAB2 presents a comparative analysis of the validation accuracy of ReLU and Swish when the depth is larger than the width, in which case the approximation by a Gaussian process is not accurate (notice that in the approximation of a neural network by a Gaussian process, we first let N l → ∞, then we consider the limit of large L).

ReLU tends to outperforms Swish when the width is smaller than the depth and both are small, however, we still observe a clear advantage of Swish for deeper architectures.

<|TLDR|>

@highlight

How to effectively choose Initialization and Activation function for deep neural networks