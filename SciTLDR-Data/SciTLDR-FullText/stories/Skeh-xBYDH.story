This work provides an additional step in the theoretical understanding of neural networks.

We consider neural networks with one hidden layer and show that when learning symmetric functions, one can choose initial conditions so that standard SGD training efficiently produces generalization guarantees.

We empirically verify this and show that this does not hold when the initial conditions are chosen at random.

The proof of convergence investigates the interaction between the two layers of the network.

Our results highlight the importance of using symmetry in the design of neural networks.

Building a theory that can help to understand neural networks and guide their construction is one of the current challenges of machine learning.

Here we wish to shed some light on the role symmetry plays in the construction of neural networks.

It is well-known that symmetry can be used to enhance the performance of neural networks.

For example, convolutional neural networks (CNNs) (see Lecun et al. (1998) ) use the translational symmetry of images to classify images better than fully connected neural networks.

Our focus is on the role of symmetry in the initialization stage.

We show that symmetry-based initialization can be the difference between failure and success.

On a high-level, the study of neural networks can be partitioned to three different aspects.

Expressiveness Given an architecture, what are the functions it can approximate well?

Training Given a network with a "proper" architecture, can the network fit the training data and in a reasonable time?

Generalization Given that the training seemed successful, will the true error be small as well?

We study these aspects for the first "non trivial" case of neural networks, networks with one hidden layer.

We are mostly interested in the initialization phase.

If we take a network with the appropriate architecture, we can always initialize it to the desired function.

A standard method (that induces a non trivial learning problem) is using random weights to initialize the network.

A different reasonable choice is to require the initialization to be useful for an entire class of functions.

We follow the latter option.

Our focus is on the role of symmetry.

We consider the following class of symmetric functions S = S n = n ∑ i=0 a i · 1 |x|=i : a 1 , . . .

, a n ∈ {±1} , where x ∈ {0, 1} n and |x| = ∑ i x i .

The functions in this class are invariant under arbitrary permutations of the input's coordinates.

The parity function π(x) = (−1) |x| and the majority function are well-known examples of symmetric functions.

Expressiveness for this class was explored by Minsky and Papert (1988) .

They showed that the parity function cannot be represented using a network with limited "connectivity".

Contrastingly, if we use a fully connected network with one hidden layer and a common activation function (like sign, sigmoid, or ReLU) only O(n) neurons are needed.

We provide such explicit representations for all functions in S; see Lemmas 1 and 2.

We also provide useful information on both the training phase and generalization capabilities of the neural network.

We show that, with proper initialization, the training process (using standard SGD) efficiently converges to zero empirical error, and that consequently the network has small true error as well.

Theorem 1.

There exists a constant c > 1 so that the following holds.

There exists a network with one hidden layer, cn neurons with sigmoid or ReLU activations, and an initialization such that for all distributions D over X = {0, 1} n and all functions f ∈ S with sample size m ≥ c(n+log(1/δ ))/ε, after performing poly(n) SGD updates with a fixed step size h = 1/poly(n) it holds that

is the network after training over S.

The number of parameters in the network described in Theorem 1 is Ω(n 2 ).

So in general one could expect overfitting when the sample size is as small as O(n).

Nevertheless, the theorem provides generalization guarantees, even for such a small sample size.

The initialization phase plays an important role in proving Theorem 1.

To emphasize this, we report an empirical phenomenon (this is "folklore").

We show that a network cannot learn parity from a random initialization (see Section 5.3).

On one hand, if the network size is big, we can bring the empirical error to zero (as suggested in Soudry and Carmon (2016) ), but the true error is close to 1/2.

On the other hand, if its size is too small, the network is not even able to achieve small empirical error (see Figure 5 ).

We observe a similar phenomenon also for a random symmetric function.

An open question remains: why is it true that a sample of size polynomial in n does not suffice to learn parity (with random initialization)?

A similar phenomenon was theoretically explained by Shamir (2016) and Song et al. (2017) .

The parity function belongs to the class of all parities

where · is the standard inner product.

This class is efficiently PAC-learnable with O(n) samples using Gaussian elimination.

A continuous version of P was studied by Shamir (2016) and Song et al. (2017) .

To study the training phase, they used a generalized notion of statistical queries (SQ); see Kearns (1998) .

In this framework, they show that most functions in the class P cannot be efficiently learned (roughly stated, learning the class requires an exponential amount of resources).

This framework, however, does not seem to capture actual training of neural networks using SGD.

For example, it is not clear if one SGD update corresponds to a single query in this model.

In addition, typically one receives a dataset and performs the training by going over it many times, whereas the query model estimates the gradient using a fresh batch of samples in each iteration.

The query model also assumes the noise to be adversarial, an assumption that does not necessarily hold in reality.

Finally, the SQ-based lower bound holds for every initialization (in particular, for the initialization we use here), so it does not capture the efficient training process Theorem 1 describes.

Theorem 1 shows, however, that with symmetry-based initialization, parity can be efficiently learned.

So, in a nutshell, parity can not be learned as part of P, but it can be learned as part of S. One could wonder why the hardness proof for P cannot be applied for S as both classes consist of many input sensitive functions.

The answer lies in the fact that P has a far bigger statistical dimension than S (all functions in P are orthogonal to each other, unlike S).

The proof of the theorem utilizes the different behavior of the two layers in the network.

SGD is performed using a step size h that is polynomially small in n. The analysis shows that in a polynomial number of steps that is independent of the choice of h the following two properties hold: (i) the output neuron reaches a "good" state and (ii) the hidden layer does not change in a "meaningful" way.

These two properties hold when h is small enough.

In Section 5.2, we experiment with large values of h. We see that, although the training error is zero, the true error becomes large.

Here is a high level description of the proof.

The neurons in the hidden layer define an "embedding" of the inputs space X = {0, 1} n into R (a.k.a.

the feature map).

This embedding changes in time according to the training examples and process.

The proof shows that if at any point in time this embedding has good enough margin, then training with standard SGD quickly converges.

This is explained in more detail in Section 3.

It remains an interesting open problem to understand this phenomenon in greater generality, using a cleaner and more abstract language.

To better understand the context of our research, we survey previous related works.

The expressiveness and limitations of neural networks were studied in several works such as Rahimi and Recht (2008) ; Telgarsky (2016) ; Eldan and Shamir (2016) and Arora et al. (2016) .

Constructions of small ReLU networks for the parity function appeared in several previous works, such as Wilamowski et al. (2003) , Arslanov et al. (2016) , Arslanov et al. (2002) and Masato Iyoda et al. (2003) .

Constant depth circuits for the parity function were also studied in the context of computational complexity theory, see for example Furst et al. (1981) , Ajtai (1983) and Håstad (1987) .

The training phase of neural networks was also studied in many works.

Here we list several works that seem most related to ours.

Daniely (2017) analyzed SGD for general neural network architecture and showed that the training error can be nullified, e.g., for the class of bounded degree polynomials (see also Andoni et al. (2014) ).

Jacot et al. (2018) studied neural tangent kernels (NTK), an infinite width analogue of neural networks.

Du et al. (2018) showed that randomly initialized shallow ReLU networks nullify the training error, as long as the number of samples is smaller than the number of neurons in the hidden layer.

Their analysis only deals with optimization over the first layer (so that the weights of the output neuron are fixed).

Chizat and Bach (2018) provided another analysis of the latter two works.

Allen-Zhu et al. (2018b) showed that over-parametrized neural networks can achieve zero training error, as as long as the data points are not too close to one another and the weights of the output neuron are fixed.

Zou et al. (2018) provided guarantees for zero training error, assuming the two classes are separated by a positive margin.

Convergence and generalization guarantees for neural networks were studied in the following works. (2019) gave data-dependent generalization bounds for GD.

All these works optimized only over the hidden layer (the output layer is fixed after initialization).

Margins play an important role in learning, and we also use it in our proof.

Sokolic et al. (2016) , Sokolic et al. (2017) , Bartlett et al. (2017) and Sun et al. (2015) gave generalization bounds for neural networks that are based on their margin when the training ends.

From a practical perspective, Elsayed et al. (2018), Romero and Alquezar (2002) and Liu et al. (2016) suggested different training algorithms that optimize the margin.

As discussed above, it seems difficult for neural networks to learn parities.

Song et al. (2017) and Shamir (2016) demonstrated this using the language statistical queries (SQ).

This is a valuable language, but it misses some central aspects of training neural networks.

SQ seems to be closely related to GD, but does not seem to capture SGD.

SQ also shows that many of the parities functions ⊗ i∈S x i are difficult to learn, but it does not imply that the parity function ⊗ i∈[n] x i is difficult to learn.

Abbe and Sandon (2018) demonstrated a similar phenomenon in a setting that is closer to the "real life" mechanics of neural networks.

We suggest that taking the symmetries of the learning problem into account can make the difference between failure and success.

Several works suggested different neural architectures that take symmetries into account; see Zaheer et al. (2017), Gens and Domingos (2014) , and Cohen and Welling (2016).

Here we describe efficient representations for symmetric functions by networks with one hidden layer.

These representations are also useful later on, when we study the training process.

We study two different activation functions, sigmoid and ReLU (similar statement can be proved for other activations, like arctan).

Each activation function requires its own representation, as in the two lemmas below.

We start with the activation σ (ξ ) =

, since it helps to understand the construction for the ReLU activation.

The building blocks of the symmetric functions are indicators of |x| = i for i ∈ {0, 1, . . .

, n}. An indicator function is essentially the difference between two sigmoid functions:

where

A network with one hidden layer of n + 2 neurons with sigmoid activations and one bias neuron is sufficient to represent any function in S. The coefficients of the sigmoid gates are 0, ±1 in this representation.

The proofs of this lemma and the subsequent lemmas appear in the appendix.

A sigmoid function can be represented using ReLU(ξ ) = max{0, ξ } as the difference between two ReLUs

Hence, an indicator function can be represented using sign(1 |x|=i − 0.5) = sign(Γ i − 0.5) where

The lemma shows that a network with one hidden layer of n + 3 ReLU neurons and one bias neuron is sufficient to represent any function in S. The coefficients of the ReLU gates are 0, ±1, ±2 in this representation.

The goal of this section is to describe a small network with one hidden layer that (when initialized properly) efficiently learns symmetric functions using a small number of examples (the training is done via SGD).

Here we specify the architecture, initialization and loss function that is implicit in our main result (Theorem 1).

To guarantee convergence of SGD, we need to start with "good" initial conditions.

The initialization we pick depends on the activation function it uses, and is chosen with resemblance to Lemma 2 for ReLU.

On a high level, this indicates that understanding the class of functions we wish to study in term of "representation" can be helpful when choosing the architecture of a neural network in a learning context.

The network we consider has one hidden layer.

We denote by w i j the weight between coordinate j of the input and neuron i in the hidden layer.

We denote W this matrix of weights.

We denote by b i the bias of neuron i of the hidden layer.

We denote B this vector of weights.

We denote by m i is the weight from neuron i in the hidden layer to the output neuron.

We denote M this vector of weights.

We denote by b the bias of the output neuron.

Initialize the network as follows: The dimensions of W are (n + 3) × n. For all 1 ≤ i ≤ (n + 3) and 1 ≤ j ≤ n, we set w i j = 1 and b i = −i + 2.

We set M = 0 and b = 0.

To run SGD, we need to choose a loss function.

We use the hinge loss,

where v x = ReLU(W x + B)

is the output of the hidden layer on input x and β > 0 is a parameter of confidence.

A key property in the analysis is the 'margin' of the hidden layer with respect to the function being learned.

We are interested in the following set V in R d .

Recall that W is the weight matrix between the input layer and the hidden layer, and that B is the relevant bias vector.

Given W, B, we are interested in the set V = {v x : x ∈ X}, where v x = ReLU(W x + B).

In words, we think of the neurons in the hidden layer as defining an "embedding" of X in Euclidean space.

A similar construction works for other activation functions.

We say that Y : V → {±1} agrees with f ∈ S if for all x ∈ X it holds that

The following lemma bounds from below the margin of the initial V .

Lemma 3.

If Y is a partition that agrees with some function in S for the initialization described above then marg(Y ) ≥ Ω(1/ √ n).

Proof.

By Lemmas 1 and 2, we see that any function in S can be represented with a vector of weights M, b ∈ [−1, 1] Θ(n) of the output neuron together with a bias .

These M, b induce a partition

we have our desired result.

Before analyzing the full behavior of SGD, we make an observation: if the weights of the hidden layer are fixed with the initialization described above, then Theorem 1 holds for SGD with batch size 1.

This observation, unfortunately, does not suffice to prove Theorem 1.

In the setting we consider, the training of the neural network uses SGD without fixing any weights.

This more general case is handled in the next section.

The rest of this subsection is devoted for explaining this observation.

Novikoff (1962) showed that that the perceptron algorithm Rosenblatt (1958) makes a small number of mistakes for linearly separable data with large margin.

For a comprehensive survey of the perceptron algorithm and its variants, see Moran et al. (2018) .

Running SGD with the hinge loss induces the same update rule as in a modified perceptron algorithm, Algorithm 1.

Initialize:

Novikoff's proof can be generalized to any β > 0 and batches of any size to yield the following theorem; see Collobert and Bengio (2004) ; Krauth and Mezard (1987) and appendix A. Theorem 2.

For Y : V → {±1} with margin γ > 0 and step size h > 0, the modified perceptron algorithm performs at most

updates and achieves a margin of at least γβ h 2β h+(Rh) 2 , where R = max v∈V v .

So, when the weights of the hidden layer are fixed, Lemma 3 implies that the number of SGD steps is at most polynomial in n.

When we run SGD on the entire network, the layers interact.

For a ReLU network at time t, the update rule for W is as follows.

If the network classifies the input x correctly with confidence more than β , no change is made.

Otherwise, we change the weights in M by ∆M = yv x h, where y is the true label and h is the step size.

If also neuron i of the hidden fired on x, we update its incoming weights by ∆W i,: = ym i xh.

These update rules define the following dynamical system: (a)

where H is the Heaviside step function and • is the Hadamard pointwise product.

A key observation in the proof is that the weights of the last layer ((4) and (5)) are updated exactly as the modified perceptron algorithm.

Another key statement in the proof is that if the network has reached a good representation of the input (i.e., the hidden layer has a large margin), then the interaction between the layers during the continued training does not impair this representation.

This is summarized in the following lemma (we are not aware of a similar statement in the literature).

Lemma 4.

Let M = 0, b = 0, and V = {ReLU(W x + B) : x ∈ X} be a linearly separable embedding of X and with margin γ > 0 by the hidden layer of a neural network of depth two with ReLU activation and weights given by W, B. Let R X = max x∈X x , let R = max v∈V v , and 0 < h ≤ γ 5/2 100R 2 R X be the integration step.

Assuming R X > 1 and γ ≤ 1, and using β = R 2 h in the loss function, after t SGD iterations the following hold:

-Each v ∈ V moves a distance of at most O(R 2 X h 2 Rt 3/2 ).

-The norm M (t) is at most O(Rh √ t).

-The training ends in at most O(R 2 /γ 2 ) SGD updates.

Intuitively, this type of lemma can be useful in many other contexts.

The high level idea is to identify a "good geometric structure" that the network reaches and enables efficient learning.

Proof of Theorem 1.

There is an unknown distribution D over the space X. We pick i.i.d.

examples S = ((x 1 , y 1 ) , ..., (x m , y m )) where m ≥ c n+log(1/δ ) ε according to D, where y i = f (x i ) for some f ∈ S. Run SGD for O(n 4 ) steps, where the step size is h = O(1/n 6 ) and the parameter of the loss function is β = R 2 h with R = n 3/2 .

We claim that it suffices to show that at the end of the training (i) the network correctly classifies all the sample points x 1 , . . .

, x m , and (ii) for every x ∈ X such that there exists 1 ≤ i ≤ m with |x| = |x i |, the network outputs y i on x as well.

Here is why.

The initialization of the network embeds the space X into n + 4 dimensional space (including the bias neuron of the hidden layer).

Let V (0) be the initial embedding V (0) = {ReLU(W (0) x + B (0) ) : x ∈ X}. Although |X| = 2 n , the size of V (0) is n + 1.

The VC dimension of all the boolean functions over V (0) is n + 1.

Now, m samples suffice to yield ε true error for an ERM when the VC dimension is n + 1; see e.g. Theorem 6.7 in Shalev-Shwartz and Ben-David (2014) .

It remains to prove (i) and (ii) above.

By Lemma 3, at the beginning of the training, the partition of V (0) defined by the target f ∈ S has a margin of γ = Ω(1/ √ n).

We are interested in the eventual V * = {ReLU(W * x + B * ) : x ∈ X} embedding of X as well.

The modified perceptron algorithm together with Lemma 4 guarantees that after K ≤ 20R 2 /γ 2 = O(n 4 ) updates, (M * , b * ) separates the embedded sample V * S = {ReLU(W * x i + B * ) : 1 ≤ i ≤ m} with a margin of at least 0.9γ/3.

It remains to prove (ii).

Lemma 4 states that as long as less than K = O(n 4 ) updates were made, the elements in V moved at most O(1/n 2 ).

At the end of the training, the embedded sample V S is separated with a margin of at least 0.9γ/3 with respect to the hyperplane defined by M * and B * .

Each v * x for x ∈ X moved at most O(1/n 2 ) < γ/4.

This means that if |x| = |x i | then the network has the same output on x and x i .

Since the network has zero empirical error, the output on this x is y i as well.

A similar proof is available with sigmoid activation (with better convergence rate and larger allowed step size).

Remark.

The generalization part of the above proof can be viewed as a consequence of sample compression (Littlestone and Warmuth (1986) ).

Although the eventual network depends on all examples, the proof shows that its functionality depends on at most n + 1 examples.

Indeed, after the training, all examples with equal hamming weight have the same label.

Remark.

The parameter β = R 2 h we chose in the proof may seem odd and negligible.

It is a construct in the proof that allows us to bound efficiently the distance that the elements in V have moved during the training.

For all practical purposes β = 0 works as well (see Figure 4) .

We accompany the theoretical results with some experiments.

We used a network with one hidden layer of 4n + 3 neurons, ReLU activation, and the hinge loss with β = n 3 h. In all the experiments, we used SGD with mini-batch of size one and before each epoch we randomized the sample.

We observed similar behavior for larger mini-batches, other activation functions, and other loss functions.

The graphs that appear in the appendix A present the training error and the true error 2 versus the epoch of the training process.

In all the comparisons below, we chose a random symmetric function and a random sample from X.

Figure 2 demonstrates our theoretical results and also validates the performance of our initialization.

In one setting, we trained only the second layer (freezed the weights of the hidden layer) which essentially corresponds to the perceptron algorithm.

In the second setting, we trained both layers with a step size h = n −6 (as the theory suggests).

As expected, performance in both cases is similar.

We remark that SGD continues to run even after minimizing the empirical error.

This happens because of the parameter β > 0.

Here we experiment with two parameters in the proof, the step size h and the confidence parameter β .

In Figure 3 , we used three different step sizes, two of which much larger than the theory suggests.

We see that the training error converges much faster to zero, when the step size is larger.

This fast convergence comes at the expense of the true error.

For a large step size, generalization cease to hold.

Setting β = n 3 h is a construct in the proof.

Figure 4 shows that setting β = 0 does not impair the performance.

The difference between theory (requires β > 0) and practice (allows β = 0) can be explained as follows.

The proof bounds the worst-case movement of the hidden layer, whereas in practice an average-case argument suffices.

Figure 5 shows that even for n = 20, learning parity is hard from a random initialization.

When the sample size is small the training error can be nullified but the true error is large.

As the sample grows, it becomes much harder for the network to nullify even the training error.

With our initialization, both the training error and true error are minimized quickly.

Figure 6 demonstrates the same phenomenon for a random symmetric function.

Our initialization also delivers satisfying results when the input data it corrupted.

In figure 7, we randomly perturb (with probability p = 1 10 ) the labels and use the same SGD to train the model.

In figure 8 , we randomly shift every entry of the vectors in the space X by ε that is uniformly distributed in [−0.1, 0.1] n .

This work demonstrates that symmetries can play a critical role when designing a neural network.

We proved that any symmetric function can be learned by a shallow neural network, with proper initialization.

We demonstrated by simulations that this neural network is stable under corruption of data, and that the small step size is the proof is necessary.

We also demonstrated that the parity function or a random symmetric function cannot be learned with random initialization.

How to explain this empirical phenomenon is still an open question.

The works Shamir (2016) and Song et al. (2017) treated parities using the language of SQ.

This language obscures the inner mechanism of the network training, so a more concrete explanation is currently missing.

We proved in a special case that the standard SGD training of a network efficiently produces low true error.

The general problem that remains is proving similar results for general neural networks.

A suggestion for future works is to try to identify favorable geometric states of the network that guarantee fast convergence and generalization.

Proof.

For all k ∈ A and x ∈ X of weight k,

the first inequality holds since ∆ i (x) ≥ 0 for all i and x. For all k ∈ A and x ∈ X of weight k,

= 2 exp(−2.5)/(1 − exp(−5)) < 0.17; the first equality follows from the definition, the second equality follows from σ (5(x + 0.5)) − σ (5(x − 0.5)) = σ (5(x + 0.5)) + σ (5(−x + 0.5)) − 1 for all x, the first inequality neglects the negative sums, and the second inequality follows because exp(ξ ) > σ (ξ ) for all ξ .

Proof.

The proof follows from two observations:

For all i ∈ A and x of weight i it holds Γ i (x) = 1.

Proof.

We are interested in the maximal distance the embedding of an element x ∈ X has moved from its initial embedding:

To simplify equations (2)-(5) discussed above, we assume that during the optimization process the norm of the weights W and B grow at a maximal rate:

here the norm of a matrix is the 2 -norm.

To bound these quantities, we follow the modified perceptron proof and add another quantity to bound.

That is, the maximal norm R (t) of the embedded space X at time t satisfies (by assumption R X > 1)

we used that the spectral norm of a matrix is at most its 2 -norm.

We assume a worst-case where R (t) grows monotonically at a maximal rate.

By the modified perceptron algorithm and choice β = R 2 h, Sample of size 10n whose input dimension is n = 30.

@highlight

When initialized properly, neural networks can learn the simple class of symmetric functions; when initialized randomly, they fail.  