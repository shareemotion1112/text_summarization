We prove bounds on the generalization error of convolutional networks.

The bounds are in terms of the training loss, the number of parameters, the Lipschitz constant of the loss and the distance from the weights to the initial weights.

They are independent of the number of pixels in the input, and the height and width of hidden feature maps.

We present experiments with CIFAR-10, along with varying hyperparameters of a deep convolutional network, comparing our bounds with practical generalization gaps.

Recently, substantial progress has been made regarding theoretical analysis of the generalization of deep learning models (see Zhang et al., 2016; Dziugaite & Roy, 2017; Bartlett et al., 2017; Arora et al., 2018; Neyshabur et al., 2019; Wei & Ma, 2019) .

One interesting point that has been explored, with roots in (Bartlett, 1998) , is that even if there are many parameters, the set of models computable using weights with small magnitude is limited enough to provide leverage for induction (Bartlett et al., 2017; .

Intuitively, if the weights start small, since the most popular training algorithms make small, incremental updates that get smaller as the training accuracy improves, there is a tendency for these algorithms to produce small weights.

(For some deeper theoretical exploration of implicit bias in deep learning and related settings, see (Gunasekar et al., 2017; 2018a; b; Ma et al., 2018) .)

Even more recently, authors have proved generalization bounds in terms of the distance from the initial setting of the weights instead of the size of the weights (Bartlett et al., 2017; Neyshabur et al., 2019) .

This is important because small initial weights may promote vanishing gradients; it is advisable instead to choose initial weights that maintain a strong but non-exploding signal as computation flows through the network (see LeCun et al., 2012; Glorot & Bengio, 2010; Saxe et al., 2013; He et al., 2015) .

A number of recent theoretical analyses have shown that, for a large network initialized in this way, a large variety of well-behaved functions can be found through training by traveling a short distance in parameter space (see Du et al., 2019b; a; Allen-Zhu et al., 2019; Zou et al., 2018) .

Thus, the distance from initialization may be expected to be significantly smaller than the magnitude of the weights.

Furthermore, there is theoretical reason to expect that, as the number of parameters increases, the distance from initialization decreases.

Convolutional layers are used in all competitive deep neural network architectures applied to image processing tasks.

The most influential generalization analyses in terms of distance from initialization have thus far concentrated on networks with fully connected layers.

Since a convolutional layer has an alternative representation as a fully connected layer, these analyses apply in the case of convolutional networks, but, intuitively, the weight-tying employed in the convolutional layer constrains the set of functions computed by the layer.

This additional restriction should be expected to aid generalization.

In this paper, we prove new generalization bounds for convolutional networks that take account of this effect.

As in earlier analyses for the fully connected case, our bounds are in terms of the distance from the initial weights, and the number of parameters.

Additionally, our bounds are "size-free", in the sense that they are independent of the number of pixels in the input, or the height and width of the hidden feature maps.

Our most general bounds apply to networks including both convolutional and fully connected layers, and, as such, they also apply for purely fully connected networks.

In contrast with earlier bounds for settings like the one considered here, our bounds are in terms of a sum over layers of the distance from initialization of the layer.

Earlier bounds were in terms of product of these distances which led to an exponential dependency on depth.

Our bounds have linear dependency on depth which is more aligned with practical observations.

As is often the case for generalization analyses, the central technical lemmas are bounds on covering numbers.

Borrowing a technique due to Barron et al. (1999) , these are proved by bounding the Lipschitz constant of the mapping from the parameters to the loss of the functions computed by the networks. (Our proof also borrows ideas from the analysis of the fully connected case, especially (Bartlett et al., 2017; .)

Covering bounds may be applied to obtain a huge variety of generalization bounds.

We present two examples for each covering bound.

One is a standard bound on the difference between training and test error.

Perhaps the more relevant bound has the flavor of "relative error"; it is especially strong when the training loss is small, as is often the case in modern practice.

Our covering bounds are polynomial in the inverse of the granularity of the cover.

Such bounds seem to be especially useful for bounding the relative error.

In particular, our covering bounds are of the form (B/ ) W , where is the granularity of the cover, B is proportional to the Lipschitz constant of a mapping from parameters to functions, and W is the number of parameters in the model.

We apply a bound from the empirical process literature in terms of covering bounds of this form due to Giné & Guillou (2001) , who paid particular attention to the dependence of estimation error on B. This bound may be helpful for other analyses of the generalization of deep learning in terms of different notions of distance from initialization. (Applying bounds in terms of Dudley's entropy integral in the standard way leads to an exponentially worse dependence on B.) Related work.

Du et al. (2018) proved size-free bounds for CNNs in terms of the number of parameters, for two-layer networks.

Arora et al. (2018) analyzed the generalization of networks output by a compression scheme applied to CNNs.

Zhou & Feng (2018) provided a generalization guarantee for CNNs satisfying a constraint on the rank of matrices formed from their kernels.

Li et al. (2018) analyzed the generalization of CNNs under other constraints on the parameters.

Lee & Raginsky (2018) provided a size-free bound for CNNs in a general unsupervised learning framework that includes PCA and codebook learning.

is the kernel of convolutional layer number i, then op(K (i) ) refers to its operator matrix 1 and vec(K (i) ) denotes the vectorization of the kernel tensor K (i) .

For matrix M , M 2 denotes the operator norm of M .

For vectors, || · || represents the Euclidian norm, and || · || 1 is the L 1 norm.

For a multiset S of elements of some set Z, and a function g from Z to R, let

We will denote the function parameterized by Θ by f Θ .

In this section, we provide a bound for a clean and simple setting.

In the basic setting, the input and all hidden layers have the same number c of channels.

Each input x ∈ R d×d×c satisfies vec(x) ≤ 1.

We consider a deep convolutional network, whose convolutional layers use zero-padding (see Goodfellow et al., 2016) .

Each layer but the last consists of a convolution followed by an activation function that is applied componentwise.

The activations are 1-Lipschitz and nonexpansive (examples include ReLU and tanh).

The kernels of the convolutional layers are

-tensor obtained by concatening the kernels for the various layers.

Vector w represents the last layer; the weights in the last layer are fixed with ||w|| = 1.

Let W = Lk 2 c 2 be the total number of trainable parameters in the network.

take arbitrary fixed values (interpreted as the initial values of the kernels) subject to the constraint that, for all layers i, || op(K (i) 0 )|| 2 = 1.

(This is often the goal of initialization schemes.)

Let K 0 be the corresponding L × k × k × c × c tensor.

We provide a generalization bound in terms of distance from initialization, along with other natural parameters of the problem.

The distance is measured with

0 )|| 2 .

For β > 0, define K β to be the set of kernel tensors within || · || σ distance β of K 0 , and define F β to be set of functions computed by CNNs with kernels in K β .

That is,

Let : R × R → [0, 1] be a loss function such that (·, y) is λ-Lipschitz for all y. An example is the 1/λ-margin loss.

We will use S to denote a set {(x 1 , y 1 ), . . .

, (x m , y m )} = {z 1 , . . .

z m } of random training examples where each z t = (x t , y t ).

Theorem 2.1 (Basic bounds).

For any η > 0, there is a C > 0 such that for any β, δ > 0, λ ≥ 1, for any joint probability distribution P over R d×d×c × R, if a training set S of n examples is drawn independently at random from P , then, with probability at least 1 − δ, for all f ∈ F β ,

If Theorem 2.1 is applied with the margin loss, then E z∼P [ f (z)] is in turn an upper bound on the probability of misclassification on test data.

Using the algorithm from (Sedghi et al., 2018) , || · || σ may be efficiently computed.

Sedghi et al., 2018) , Theorem 2.1 yields the same bounds as a corollary if the definition of F β is replaced with the analogous definition using || vec(K) − vec(K 0 )|| 1 .

Definition 2.2.

For d ∈ N , a set G of functions with a common domain Z, we say that G is (B, d)-Lipschitz parameterized if there is a norm || · || on R d and a mapping φ from the unit ball w.r.t.

|| · || in R d to G such that, for all θ and θ such that ||θ|| ≤ 1 and ||θ || ≤ 1, and all z ∈ Z,

The following lemma is essentially known.

Its proof, which uses standard techniques (see Pollard, 1984; Talagrand, 1994; 1996; Barron et al., 1999; Van de Geer, 2000; Giné & Guillou, 2001; Mohri et al., 2018) , is in Appendix A. Lemma 2.3.

Suppose a set G of functions from a common domain

Then, for any η > 0, there is a C such that, for all large enough n ∈ N, for any δ > 0, for any probability distribution P over Z, if S is obtained by sampling n times independently from P , then, with probability at least 1 − δ, for all g ∈ G,

n and

We will prove Theorem 2.1 by showing that F β is βλe β , W -Lipschitz parameterized.

This will be achieved through a series of lemmas.

Lemma 2.5.

Choose K ∈ K β and a layer j.

Let g up be the function from the inputs to the whole network with parameters K to the inputs to the convolution in layer j, and let g down be the function from the output of this convolution to the output of the whole network, so that

Choose an input x to the network, and let u = g up (x).

Recalling that ||x|| ≤ 1, and the non-linearities are nonexpansive, we have u ≤

.

Since the non-linearities are 1-Lipschitz, and, recalling that

where the last inequality uses the fact that || op(

(1 + β i ), and the latter is maximized over the nonnegative β i 's subject

L ≤ e β , this completes the proof.

Now we prove a bound when all the layers can change between K andK.

Lemma 2.6.

For any K,K ∈ K β , for any input x to the network,

Proof.

Consider transforming K toK by replacing one layer of K at a time with the corresponding layer inK. Applying Lemma 2.5 to bound the distance traversed with each replacement and combining this with the triangle inequality gives

Now we are ready to prove our basic bound.

Proof (of Theorem 2.1).

Consider the mapping φ from the ball w.r.t.

Lemma 2.6 implies that this mapping is βλe β -Lipschitz.

Applying Lemma 2.3 completes the proof.

Since a convolutional network has an alternative parameterization as a fully connected network, the bounds of (Bartlett et al., 2017) have consequences for convolutional networks.

To compare our bound with this, first, note that Theorem 2.1, together with standard model selection techniques, yields a

(For more details, please see Appendix B.) Translating the bound of (Bartlett et al., 2017) to our setting and notation directly yields a bound on

whose main terms are proportional to

where, for a p × q matrix A, ||A|| 2,1 = ||(||A :,1 || 2 , ..., ||A :,q || 2 )|| 1 .

One can get an idea of how this bound relates to (1) by comparing the bounds in a simple concrete case.

Suppose that each of the convolutional layers of the network parameterized by K 0 computes the identity function, and that K is obtained from K 0 by adding to each entry.

In this case, disregarding edge effects, for all i, || op(K (i) )|| 2 = 1 + k 2 c and ||K − K 0 || σ = k 2 cL (as proved in Appendix C).

Also,

We get additional simplification if we set = 1 k 2 .

In this case, (2) gives a constant times

where (1) gives a constant times c 3/2 kL + ck log(λ) + log(1/δ)

√ n .

In this scenario, the new bound is independent of d, and grows more slowly with λ, c and L. Note that k ≤ d (and, typically, it is much less).

This specific case illustrates a more general effect that holds when the initialization is close to the identity, and changes to the parameters are on a similar scale.

In this section, we generalize Theorem 2.1.

The more general setting concerns a neural network where the input is a d × d × c tensor whose flattening has Euclidian norm at most χ, and network's output is a m-dimensional vector, which may be logits for predicting a one-hot encoding of an m-class classification problem.

The network is comprised of L c convolutional layers followed by L f fully connected layers.

The ith convolutional layer includes a convolution, with kernel K (i) ∈ R ki×ki×ci−1×ci , followed by a componentwise non-linearity and an optional pooling operation.

We assume that the non-linearity and any pooling operations are 1-Lipschitz and nonexpansive.

Let V (i) be the matrix of weights for the ith fully connected layer.

Let Θ = (

We assume that, for all y, (·, y) is λ-Lipschitz for all y and that (ŷ, y) ∈ [0, M ] for allŷ and y.

An example (x, y) includes a d × d × c-tensor x and y ∈ R m .

We let K

take arbitrary fixed values subject to the constraint that, for all convolutional layers i, || op(K (i) 0 )|| 2 ≤ 1+ν, and for all fully connected layers i, ||V

For β, ν ≥ 0, define F β,ν to be set of functions computed by CNNs as described in this subsection with parameters within || · || N -distance β of Θ 0 .

Let O β,ν be the set of their parameterizations.

Theorem 3.1 (General Bound).

For any η > 0, there is a constant C such that the following holds.

For any β, ν, χ > 0, for any δ > 0, for any joint probability distribution P over R d×d×c × R m such that, with probability 1, (x, y) ∼ P satisfies || vec(x)|| 2 ≤ χ, under the assumptions of this section, if a training set S of n examples is drawn independently at random from P , then, with probability at least 1 − δ, for all f ∈ F β,ν ,

CM (W (β + νL + log(χλβn)) + log(1/δ)) n and, if χλβe β ≥ 5,

W (β + νL + log(χλβ)) + log(1/δ) n and a bound of

holds for all χ, λ, β > 0.

We will prove Theorem 3.1 by using ||·|| N to witness the fact that F β,ν is χλβe νL+β , W -Lipschitz parameterized.

The first two lemmas concern the effect of changing a single layer.

Their proofs are very similar to the proof of Lemma 2.5, and are in the Appendices D and E. Lemma 3.2.

for all convolutional layers i = j and V (i) =Ṽ (i) for all fully connected layers i.

Then, for all examples (x, y),

for all convolutional layers i and V (i) =Ṽ (i) for all fully connected layers i = j. Then, for all examples (x, y),

Proof.

Consider transforming Θ toΘ by replacing one layer at a time of Θ with the corresponding layer inΘ. Applying Lemma 3.2 to bound the distance traversed with each replacement of a convolutional layer, and Lemma 3.3 to bound the distance traversed with each replacement of a fully connected layer, and combining this with the triangle inequality gives the lemma.

Now we are ready to prove our more general bound.

Proof (of Theorem 3.1).

Consider the mapping φ from the ball of || · || N -radius 1 centered at Θ 0 to F β,ν defined by φ(Θ) = f Θ 0 +βΘ .

Lemma 2.6 implies that this mapping is χλβe νL+β , WLipschitz.

Applying Lemma 2.3 completes the proof.

Theorem 3.1 applies in the case that there are no convolutional layers, i.e. for a fully connected network.

In this subsection, we compare its bound in this case with the bound of (Bartlett et al.,

where H is a Hadamard matrix (using the Sylvester construction), and χ = M = 1.

Then, dropping the superscripts, each layer V has

Further, in the notation of Theorem 3.1, W = D 2 L, and β = L, and ν = 0.

Plugging into to Theorem 3.1 yields a bound on the generalization gap proportional to DL + D L log(λ) + log(1/δ)

√ n where, in this case, the bound of (Bartlett et al., 2017 ) is proportional to

We trained a 10-layer all-convolutional model on the CIFAR-10 dataset.

The architecture was similar to VGG (Simonyan & Zisserman, 2014) .

The network was trained with dropout regularization and an exponential learning rate schedule.

We define the generalization gap as the difference between train error and test error.

In order to analyze the effect of the number of network parameters on generalization gap, we scaled up the number of channels in each layer, while keeping other elements of the architecture, including the depth, fixed.

Each network was trained repeatedly, sweeping over different values of the initial learning rate and batch sizes 32, 64, 128.

For each setting the results were averaged over five different random initializations.

Figure 1 shows the generalization gap for different values of W ||K − K 0 || σ .

As in the bound of Theorem 3.1, the generalization gap increases with W ||K − K 0 || σ .

Figure 2 shows that as the network becomes more over-parametrized, the generalization gap remains almost flat with increasing W .

This is expected due to role of overparametrization on generalization (Neyshabur et al., 2019 ).

An explanation of this phenomenon that is consistent with the bound presented here is that increasing W leads to a decrease in value of ||K − K 0 || σ ; see Figure 3a .

The fluctuations in Figure 3a are partly due to the fact that training neural networks is not an stable process.

We provide the medians ||K − K 0 || σ for different values of W in Figure 3b .

A PROOF OF LEMMA 2.3

Definition A.1.

If (X, ρ) is a metric space and H ⊆ X, we say that G is an -cover of H with respect to ρ if every h ∈ H has a g ∈ G such that ρ(g, h) ≤ .

Then N ρ (H, ) denotes the size of the smallest -cover of H w.r.t.

ρ.

Definition A.2.

For a domain Z, define a metric ρ max on pairs of functions from Z to R by

We need two lemmas in terms of these covering numbers.

The first is by now a standard bound from Vapnik-Chervonenkis theory (Vapnik & Chervonenkis, 1971; Vapnik, 1982; Pollard, 1984) .

For example, it is a direct consequence of (Haussler, 1992 , Theorem 3).

Lemma A.3.

For any η > 0, there is a constant C depending only on η such that the following holds.

Let G be an arbitrary set of functions from a common domain Z to [0, M ].

If there are constants B and d such that, N ρmax (G, ) ≤ B d for all > 0, then there is an absolute constant C such that, for all large enough n ∈ N, for any δ > 0, for any probability distribution P over Z, if S is obtained by sampling n times independently from P , then, with probability at least 1 − δ, for all g ∈ G,

n .

We will also use the following, which is the combination of (2.5) and (2.7) of (Giné & Guillou, 2001) .

Lemma A.4.

Let G be an arbitrary set of functions from a common domain Z to [0, M ].

If there are constants B ≥ 5 and d such that N ρmax (G, ) ≤ B d for all > 0, then there is an absolute constant C such that, for all large enough n ∈ N, for any δ > 0, for any probability distribution P over Z, if S is obtained by sampling n times independently from P , then, with probability at least 1 − δ, for all g ∈ G,

The following, which can be obtained by combining Talagrand's Lemma with the standard bound on Rademacher complexity in terms of the Dudley entropy integral (see (Van de Geer, 2000; Bartlett, 2013) ), yields a bound for small B. Lemma A.5.

Let G be an arbitrary set of functions from a common domain Z to [0, M ].

If there are constants B > 0 and d such that N ρmax (G, ) ≤ B d for all > 0, then there is an absolute constant C such that, for all large enough n ∈ N, for any δ > 0, for any probability distribution P over Z, if S is obtained by sampling n times independently from P , then, with probability at least 1 − δ, for all g ∈ G,

So now we want a bound on N ρmax (G, ) for Lipschitz-parameterized classes.

For this, we need the notion of a packing which we now define.

Definition A.6.

For any metric space (X, ρ) and any H ⊆ S, let M ρ (H, ) be the size of the largest subset of H whose members are pairwise at a distance greater than w.r.t.

ρ.

Lemma A.7 ((Kolmogorov & Tikhomirov, 1959) ).

For any metric space (X, ρ), any H ⊆ X, and any > 0, we have

We will also need a lemma about covering a ball by smaller balls.

This is probably also already known, and uses a standard proof (see Pollard, 1990 , Lemma 4.1), but we haven't found a reference for it.

Lemma A.8.

Let

• d be an integer,

• || · || be a norm

• ρ be the metric induced by || · ||, and

• κ, > 0.

A ball in R d of radius κ w.r.t.

ρ can be covered by 3κ d balls of radius .

Proof.

We may assume without loss of generality that κ > .

Let q > 0 be the volume of the unit ball w.r.t.

ρ in R d .

Then the volume of any α-ball with respect to ρ is α d q. Let B be the ball of radius r in R d .

The /2-balls centered at the members of any -packing of B are disjoint.

Since these centers are contained in B, the balls are contained in a ball of radius κ + /2.

Thus

Solving for M ρ (B, ) and applying Lemma A.7 completes the proof.

We now prove Lemma 2.3.

Let || · || be the norm witnessing the fact that G is (B, d)-Lipschitz parameterized, and let B be the unit ball in R d w.r.t.

|| · || and let ρ be the metric induced by || · ||.

Then, for any , we have

Applying Lemma A.8, this implies

Then applying Lemma A.3, Lemma A.4 and Lemma A.5 completes the proof.

B PROOF OF (1) For δ > 0, and for each j ∈ N, let β j = 5 × 2 j let δ j = 1 2j 2 .

Taking a union bound over an application of Theorem 2.1 for each value of j, with probability at least 1 − j δ j ≥ 1 − δ, for all j, and all f ∈ F βj

For any K, if we apply these bounds in the case of the least j such that ||K − K 0 || σ ≤ β j , we get

and simplifying completes the proof.

For the rest of this section, we number indices from 0, let [d] = {0, ..., d − 1}, and define ω = exp(2πi/d).

To facilitate the application of matrix notation, pad the k × k × c × c tensor J out with zeros to make a d × d × c × c tensorJ.

The following lemma is an immediate consequence of Theorem 6 of Sedghi et al. (2018) .

Lemma C.1 (Sedghi et al. (2018) ).

Let F be the complex

First, note that, by symmetry, for each u and v, all components of P (u,v) are the same.

Thus,

For any u, v,

Combining this with (3) and Lemma C.1, || op(J)|| 2 = ck 2 , which implies

0 )|| 2 , and, for each fully connected layer i,

0 || 2 .

Since is λ-Lipschitz w.r.t.

its first argument, we have that | (f Θ (x), y) − (fΘ(x), y)| ≤ λ|f Θ (x) − fΘ(x)|.

Let g up be the function from the inputs to the whole network with parameters Θ to the inputs to the convolution in layer j, and let g down be the function from the output of this convolution to the output of the whole network, so that f Θ = g down • f op(K (j) ) • g up .

Choose an input x to the network, and let u = g up (x).

Recalling that ||x|| ≤ χ, and that the non-linearities and pooling operations are non-expansive, we have u ≤ χ i<j op(K (i) )

.

Using the fact that the non-linearities 1-Lipschitz, we have

where the last inequality uses the fact that || op(K (i) ) − op(K

0 || 2 .

Since is λ-Lipschitz w.r.t.

its first argument, we have that | (f Θ (x), y) − (fΘ(x), y)| ≤ λ|f Θ (x) − fΘ(x)|.

Let g up be the function from the inputs to the whole network with parameters Θ to the inputs to fully connected layer layer j, and let g down be the function from the output of this layer to the output of the whole network, so that f Θ = g down • f V (j) • g up .

Choose an input x to the network, and let u = g up (x).

Recalling that ||x|| ≤ χ, and that the non-linearities and pooling operations are non-expansive, we have u ≤ χ

Since ( i (1 + ν + β i )) i =j (1 + ν + γ i ) ≤ ( i (1 + ν + β i )) ( i (1 + ν + γ i )), and the latter is maximized subject to ( i β i ) + i γ i ≤ β when each summand is β/L, this completes the proof.

<|TLDR|>

@highlight

We prove generalization bounds for convolutional neural networks that take account of weight-tying

@highlight

Studies the generalization power of CNNs and improves the upper bounds of generalization errors, showing correlation between the generalization error of learned CNNs and the upper bound's dominant term.

@highlight

This paper presents a generalization bound for convolutional neural networks based on the number of parameters, the Lipschitz constant, and the distance of the final weights from initialization.