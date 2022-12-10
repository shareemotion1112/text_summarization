We analyze the joint probability distribution on the lengths of the vectors of hidden variables in different layers of a fully connected deep network, when the weights and biases are chosen randomly according to Gaussian distributions, and the input is binary-valued.

We show that, if the activation function satisfies a minimal set of assumptions, satisfied by all activation functions that we know that are used in practice, then, as the width of the network gets large, the ``length process'' converges in probability to a length map that is determined as a simple function of the variances of the random weights and biases, and the activation function.



We also show that this convergence may fail for activation functions  that violate our assumptions.

second layer may not be independent, even for some permissible φ like the ReLU.

The results of this section contradict claims made in BID13 BID9 .

Section 5 describes some simulation experiments verifying some of the findings of the paper, and illustrating the dependence among the values of the hidden nodes.

Our analysis of the convergence of the length map borrows ideas from Daniely, et al. BID1 , who studied the properties of the mapping from inputs to hidden representations resulting from random Gaussian initialization.

Their theory applies in the case of activation functions with certain smoothness properties, and to a wide variety of architectures.

Our analysis treats a wider variety of values of σ w and σ b , and uses weaker assumptions on φ.

For n ∈ N, we use [n] to denote the set {1, 2, . . . , n}. If T is a n × m × p tensor, then, for i ∈ [n], let T i,:,: = T i,j,k jk , and define T i,j,: , etc., analogously.

Consider a deep fully connected width-N network with D layers.

Let W ∈ R D×N ×N .

An activation function φ maps R to R; we will also use φ to denote the function from R N to R N obtained by applying φ componentwise.

Computation of the neural activity vectors x 0,: , ..., x D,: ∈ R N and preactivations h 1,: , ..., h D,: ∈ R N proceeds in the standard way as follows:h ,: = W ,:,: x −1,: + b ,: x ,: = φ(h ,: ), for = 1, . . .

, D.We will study the process arising from fixing an arbitrary input x 0,: ∈ {−1, 1} N and choosing the parameters independently at random: the entries of W are sampled from Gauss 0, DISPLAYFORM0 Note that for all ≥ 1, all the components of h ,: and x ,: are identically distributed.

For the purpose of defining a limit, assume that, for a fixed, arbitrary function χ : N → {−1, 1}, for finite N , we have x 0,: = (χ(1), ..., χ(N )).

For > 0, if the limit exists (in the sense of "convergence in distribution"), let x be a random variable whose distribution is the limit of the distribution of x ,1 as N goes to infinity.

Define h and q similarly.

If P and Q are probability distributions, then d T V (P, Q) = sup E P (E) − Q(E), and if p and q are their densities, DISPLAYFORM0

In this section we characterize the length map of the hidden nodes of a deep network, for all activation functions satisfying the following assumptions.

Definition 1 An activation function φ is permissible if, (a) the restriction of φ to any finite interval is bounded; (b) |φ(x)| = exp(o(x 2 )) as |x| gets large.

1 ; and (c) φ is measurable.

Conditions (b) and (c) ensure that a key integral can be computed.

DISPLAYFORM0 If φ is permissible, then, since φ(cz) 2 exp(−z 2 /2) is integrable for all c, we have that q 0 , ...,q D ,r 0 , ...,r D are well-defined finite real numbers.

The following theorem shows that the length map q 0 , ..., q D converges in probability toq 0 , ...,q D .Theorem 2 For any permissible φ, σ w , σ b ≥ 0, any depth D, and any , δ > 0, there is an N 0 such that, for all N ≥ N 0 , with probability 1 − δ, for all ∈ {0, ..., D}, we have |q −q | ≤ .The rest of this section is devoted to proving Theorem 2.

Our proof will use the weak law of large numbers.

For any random variable X with a finite expectation, and any , δ > 0, there is an N 0 such that, for all N ≥ N 0 , if X 1 , ..., X N are i.i.d.

with the same distribution as X, then DISPLAYFORM0 In order to divide our analysis into cases, we need the following lemma, whose proof is in Appendix B.Lemma 4 If φ is permissible and not zero a.e.

, for all σ w > 0, for all ∈ {0, ..., D},q > 0 and r > 0.We will also need a lemma that shows that small changes in σ lead to small changes in Gauss(0, σ 2 ).Lemma 5 (see BID7 ) There is an absolute constant C such that, for all σ 1 , σ 2 > 0, DISPLAYFORM1 The following technical lemma is proved in Appendix C.

If φ is permissible, for all 0 < r ≤ s, for all β > 0, there is an a ≥ 0 such that, for all q ∈ [r, s], DISPLAYFORM0 Armed with these lemmas, we are ready to prove Theorem 2.First, if φ is zero a.e.

, or if σ w = 0, Theorem 2 follows directly from Lemma 3, together with a union bound over the layers.

Assume for the rest of the proof that φ(x) is not zero a.e.

, and that σ w > 0, so thatq > 0 andr > 0 for all .

DISPLAYFORM1 ,i .

Our proof of Theorem 2 is by induction.

The inductive hypothesis is that, for any , δ > 0 there is an N 0 such that, if N ≥ N 0 , then, with probability 1 − δ, for all ≤ , |q −q | ≤ and |r −r | ≤ .The base case holds because q 0 =q 0 = r 0 =r 0 = 1, no matter what the value of N is.

Now for the induction step; choose > 0, 0 < < min{q /4,r } and 0 < δ ≤ 1/2.

(Note that these choices are without loss of generality.) Let ∈ (0, ) take a value that will be described later, using quantities from the analysis.

By the inductive hypothesis, whatever the value of , there is an N 0 such that, if N ≥ N 0 , then, with probability 1 − δ/2, for all ≤ − 1, we have |q −q | ≤ and |r −r | ≤ .

Thus, to establish the inductive step, it suffices to show that, after conditioning on the random choices before the th layer, if |q −1 −q −1 | ≤ , and |r −1 −r −1 | ≤ , there is an N such that, if N ≥ N , then with probability at least 1 − δ/2 with respect only to the random choices of W ,:,: and b ,: , that |q −q | ≤ and |r −r | ≤ .

Given such an N , the inductive step can be satisfied by letting N 0 be the maximum of N 0 and N .Let us do that.

For the rest of the proof of the inductive step, let us condition on outcomes of the layers before layer , and reason about the randomness only in the th layer.

Let us further assume that |q −1 −q −1 | ≤ and |r −1 −r −1 | ≤ .

DISPLAYFORM2 Since we have conditioned on the values of h −1,1 , ..., h −1,N , each component of h ,i is obtained by taking the dot-product of x −1,: = φ(h −1,: ) with W ,i,: and adding an independent b ,i .

Thus, conditioned on h −1,1 , ..., h −1,N , we have that h ,1 , ..., h ,N are independent.

Also, since x −1,: is fixed by conditioning, each h ,i has an identical Gaussian distribution.

Since each component of W and b has zero mean, each h ,i has zero mean.

Choose an arbitrary i ∈ [N ].

Since x −1,: is fixed by conditioning and W ,i,1 , ..., W ,i,N and b ,i are independent, DISPLAYFORM3 We wish to emphasize the q is determined as a function of random outcomes before the th layer, and thus a fixed, nonrandom quantity, regarding the randomization of the th layer.

By the inductive hypothesis, we have DISPLAYFORM4 The key consequence of this might be paraphrased by saying that, to establish the portion of the inductive step regarding q , it suffices for q to be close to its mean.

Now, we want to prove something similar for r .

We have DISPLAYFORM5 which gives DISPLAYFORM6 Since |q −q | ≤ σ 2 w and we may choose to ensure ≤q 2σ 2 w , we haveq /2 ≤ q ≤ 2q .For β > 0 and κ ∈ (0, 1/2) to be named later, by Lemma 6, we can choose a such that, for all q ∈ [q /2, 2q ], DISPLAYFORM7 We claim that DISPLAYFORM8 So now we are trying to bound DISPLAYFORM9 Using changes of variables, we have DISPLAYFORM10 But since, for κ < 1/2, conditioning on an event of probability at least 1 − κ only changes a distribution by total variation distance at most 2κ, and therefore, applying Lemma 5 along with the fact that |q −q | ≤ σ 2 w , for the constant C from Lemma 5, we get DISPLAYFORM11 Tracing back, we have DISPLAYFORM12 If κ = min{ 24M , Recall that q is an average of N identically distributed random variables with a mean between 0 and 2q (which is therefore finite) and r is an average of N identically distributed random variables, each with mean between 0 andr + /2 ≤ 2r .

Applying the weak law of large numbers (Lemma 3), there is an N such that, if N ≥ N , with probability at least 1 − δ/2, both |q − E[q ]| ≤ /2 and |r − E[r ]| ≤ /2 hold, which in turn implies |q −q | ≤ and |r −r | ≤ , completing the proof of the inductive step, and therefore the proof of Theorem 2.

In this section, we show that, for some activation functions, the probability distribution of hidden nodes can have some surprising properties.

In this subsection, we will show that the hidden variables are sometimes not Gaussian.

Our proof will refer to the Cauchy distribution.

Definition 2 A distribution over the reals that, for x 0 ∈ R and γ > 0, has a density f given by FIG1 DISPLAYFORM0 DISPLAYFORM1 So, for all N , h 2,1 is Cauchy(0, √ N ).

Suppose that h 2,1 converged in distribution to some distribution P .

Since the cdf of P can have at most countably many discontinuities, we can cover the real line by a countable set of finite-length FIG1 , ... whose endpoints are points of continuity for P .

Since Cauchy(0, DISPLAYFORM2 Thus, the probability assigned by P to the entire real line is 0, a contradiction.

The following contradicts a claim made on line 8 of Section A.1 of BID13 .Theorem 10 If φ is either the ReLU or the Heaviside function, then, for every σ w > 0, σ b ≥ 0, and N ≥ 2, FIG1 are not independent.

Proof: We will show that E[h DISPLAYFORM0 , which will imply that h 2,1 and h 2,2 are not independent.

As mentioned earlier, because each component of h 1,: is the dot product of x 0,: with an independent row of W 1,:,: plus an independent component of b 1,: , the components of h 1,: are independent, and since x 1,: = φ(h 1,: ), this implies that the components of x 1,: are independent.

Since each row of W 1,:,: and each component of the bias vector has the same distribution, x 1,: is i.i.d.

We have DISPLAYFORM1 The components of W 2,:,: and x 1,: , along with b 2,1 , are mutually independent, so terms in the double sum with i = j have zero expectation, and E[h DISPLAYFORM2 .

For a random variable x with the same distribution as the components of x 1,: , this implies DISPLAYFORM3 Similarly, DISPLAYFORM4 Putting this together with (3), we have ) and Gauss(0, σ 2 ) for σ estimated from the data (shown in red).

Now, we calculate the difference using (4) for the Heaviside and ReLU functions.

DISPLAYFORM5 Heaviside.

Suppose φ is Heaviside function, i.e. φ(z) is the indicator function for z > 0.

In this case, since the components of h 1,: are symmetric about 0, the distribution of x 1,: is uniform over DISPLAYFORM6 DISPLAYFORM7 dz.

By symmetry this is DISPLAYFORM8 Similarly, E[ DISPLAYFORM9 2 .

Plugging these into (4) we get that, in the case the φ is the ReLU, that DISPLAYFORM10 completing the proof.

Here, we show, informally, that for φ at the boundary of the second condition in the definition of permissibility, the recursive formula defining the length mapq breaks down.

Roughly, this condition cannot be relaxed.

For any α > 0, if φ is defined by φ(x) = exp(αx 2 ), there exists a σ w , σ b s.t.q ,r is undefined for all ≥ 2.

For each N ∈ {10, 100, 1000}, we (a) initialized the weights 100 times, (b) plotted the histograms of all of the values of h [2, :] , along with the Cauchy(0, √ N ) distribution from the proof of Proposition 9, and Gauss(0, σ 2 ) for σ estimated from the data.

Consistent with the theory, the Cauchy(0, √ N ) distribution fits the data well.

To illustrate the fact that the values in the second hidden layer are not independent, for N = 1000 and the parameters otherwise as in the other experiment, we plotted histograms of the values seen in the second layer for nine random initializations of the weights in FIG5 .

When some of the values in the first hidden layer have unusually small magnitude, then the values in the second hidden layer coordinately tend to be large.

This is in contrast with the claim made at the end of Section 2.2 of BID9 .

Note that this is consistent with Theorem 2 establishing convergence in probability for permissible φ, since the φ used in this experiment is not permissible.

<|TLDR|>

@highlight

We prove that, for activation functions satisfying some conditions, as a deep network gets wide, the lengths of the vectors of hidden variables converge to a length map.