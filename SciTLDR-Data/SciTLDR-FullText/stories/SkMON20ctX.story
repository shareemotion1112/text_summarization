Studying the evolution of information theoretic quantities during Stochastic Gradient Descent (SGD) learning of Artificial Neural Networks (ANNs) has gained popularity in recent years.

Nevertheless, these type of experiments require estimating mutual information and entropy which becomes intractable for moderately large problems.

In this work we propose a framework for understanding SGD learning in the information plane which consists of observing entropy and conditional entropy of the output labels of ANN.

Through experimental results and theoretical justifications it is shown that, under some assumptions, the SGD learning trajectories appear to be similar for different ANN architectures.

First, the SGD learning is modeled as a Hidden Markov Process (HMP) whose entropy tends to increase to the maximum.

Then, it is shown that the SGD learning trajectory appears to move close to the shortest path between the initial and final joint distributions in the space of probability measures equipped with the total variation metric.

Furthermore, it is shown that the trajectory of learning in the information plane can provide an alternative for observing the learning process, with potentially richer information about the learning than the trajectories in training and test error.

How do information theoretic quantities behave during the training of ANNs?

This question was addressed by Shwartz-Ziv & Tishby (2017) in an attempt to explain the learning through the lens of the information bottleneck method (Tishby et al., 1999) .

In that work, the layers of an ANNs are considered random variables forming a Markov chain.

The authors constructed a 2D information plane by estimating the mutual information values between hidden layers, inputs, and outputs of ANNs.

Using this approach it was observed that the information bottleneck method provides an approximate explanation for SGD learning.

In addition, their experiments showed the role of compression in learning.

That initial paper motivated further work on this line of research BID17 BID9 .

The main practical limitation of that type of experiments is that it requires estimating mutual information between high dimensional continuous random variables.

This becomes prohibitive as soon we move to moderately large problems, such as the CIFAR-100 dataset, where the large ANNs are employed.

Other works dealing with information theoretic quantities tend to have these experimental limitations.

For instance, BID16 ; Xu & Raginsky (2017) ; BID3 used generic chaining techniques to show that generalization error can be upper bounded by the mutual information between the training dataset and output of the learning algorithm.

Nevertheless, estimating that mutual information to verify those results experimentally becomes intractable.

Furthermore, in our previous work BID1 we defined a novel 2D information plane that only requires to estimate information theoretic quantities between the correct and estimated labels.

Since these random variables are discrete and one-dimensional, this framework can be used to study learning in large recognition problems as well.

Moreover, that work provides a preliminary empirical study on the behavior of those information theoretic quantities during learning along with some connections between error and conditional entropy.

In this work, we extend the experiments from BID1 to more general scenarios and aim to characterize the observed behavior of SGD.

Our main contributions are as follows:• We define a 2D-information plane, inspired by the works of Shwartz-Ziv & Tishby (2017) , and use it to study the behavior of ANNs during SGD learning.

The main quantities are entropy of the output labels and its conditional entropy given true labels.• It is shown that if the learning is done perfectly and under some other mild assumptions, the entropy tends to increase to its maximum.• It is additionally shown that SGD learning trajectory follows approximately the shortest path in the space of probability measures equipped with the total variation metric.

The shortest path is characterized well by a Markov chain defined on probabilities of estimate labels conditioned on true labels.

To that end we provide theoretical and experimental justifications for constructing a simple Markovian model for learning, and compare it with SGD through experiments.

These experiments are conducted using various datasets such as MNIST BID13 , CIFAR-10/ CIFAR-100, spirals BID1 , as well as different ANN architectures like Fully Connected Neural Networks (FCNNs), LeNet-5 (LeCun et al., 1999) , and DenseNet BID11 .•

The trajectory, however, is not universal.

Through a set of experiments, it is shown that SGD learning trajectory differs significantly for different learning strategies, noisy labels, overfitting, and underfitting.

We show examples where this type of trajectories provide a richer view of the learning process than conventional training and test error, which allows us to spot undesired effects such as overfitting and underfitting.

The paper is organized as follows: Section 2 introduces the notation as well as elementary notions from information theory.

Section 3 formulates learning as a trajectory on the space of probability measures, defines the notion of shortest learning path, and provides a connection to Markov chains.

Section 4 constructs a simple Markov chain model for gradient based learning that moves along the shortest learning path.

Finally, Section 5 performs an empirical evaluation of the proposed model.

Let x ∈ X be a random vector belonging to some set X of possible inputs.

We assume that there exists a function, known as "oracle", that maps x to one of K ∈ N classes.

Formally, there exists a deterministic mapping c : X → Y where Y = {0, . . .

, K − 1} is the set of possible classes.

Then, letỹ ∈ Y denote the random variableỹ = c(x).

One common assumption, that is present in popular datasets such as MNIST, CIFAR-10, CIFAR-100, and Imagenet, is thatỹ is uniformly distributed.

We assume this to be true throughout this paper.

Note that the designer of the dataset has control over the marginal distribution ofỹ.

We model the effect of having error-prone labels, denoted by the random variable y, in the data by introducing discrete independent random noise z ∈ Y toỹ in the form of modulo addition 1 , that is y =ỹ ⊕ z ∈ Y. Let θ ∈ Θ be the vector, possibly random, containing of all tunable parameters in the hypothesis space Θ. Then a classifier is a deterministic function g : Θ × X → Y that aims to approximate c. Further,ŷ = g(θ, x) is defined to be the random variable of the label predicted by the classifier g(θ, ·).

Using this notation we define the three types of error: the dataset error p = P (y =ỹ), the test error ε = P (ŷ = y), the true errorε = P (ŷ =ỹ).

A summary of this system model is provided in FIG0 .

We shortly review some elementary concepts from information theory such as entropy and mutual information.

The entropy of a discrete random variable y ∈ Y is defined as 2 H(y) = − y∈Y P (y = y) log P (y = y) = −E log P (y) .The entropy is bounded by 0 ≤ H(y) ≤ log |Y| and it measures the amount uncertainty present in y. Similarly, the conditional entropy between two random variables y andŷ is DISPLAYFORM0 and it quantifies the uncertainty about y given thatŷ is known.

Finally, the mutual information I(y;ŷ) between y andŷ measures how much information does one random variable carry about the other.

It may be defined in terms of entropies as DISPLAYFORM1 Moreover, the following proposition is a well-known result from information theory, known as Fano's inequality, that relates test error ε and conditional entropy.

Proposition 1.

Fano's Inequality BID6 , Lemma 3.8) The value of H(y|ŷ) and H(ŷ|y) is upper bounded by a function of the expected error as DISPLAYFORM2 where the function Ψ : [0, 1] → R is defined as DISPLAYFORM3 This results provides an upper bound on conditional entropy in terms of ε, that is known to be sharp.

In the works of BID8 it has been shown that I(y;ŷ) gives an upper and lower bound on the minimal error 3 between y andŷ.

In addition, the minimal error is minimized when I(y;ŷ) reaches its maximum.

Therefore, learning can be modeled as finding θ such that I(y;ŷ) is maximized.

This can be written in terms of entropies as DISPLAYFORM4 As in our previous work BID1 we are interested on characterizing the trajectory in the 2D information plane, composed by H(ŷ) and H(ŷ|y), during the learning process of artificial neural networks (ANNs).

In FIG1 we observe that learning trajectory for the DenseNet architecture of 100 layers as it learns to classify data from the CIFAR-100 dataset, for p = 0.

Intuitively, when solving equation 1, maximizing H(ŷ) is more related with the unsupervised component of learning since it does not depend y. On the other hand, keeping H(ŷ|y) low while H(ŷ) increases can be seen as the supervised component of equation 1.

From this point of view it would be interesting to characterize the inflection point from which H(ŷ|y) starts decreasing, since it allows us to observe at which point SGD starts paying more attention to assigning labels correctly than to learn about the distribution of the input.

One also may wonder if this increasing-decreasing trajectory is an accidental result for that occurs only on this particular experimental setup, or if it is a fundamental property of SGD.

In BID1 we showed that this behavior seems to appear regardless of the activation function employed (see Appendix C) on the spirals and MNIST dataset.

Moreover, in further sections we provide a justification for this type of trajectory and show that it remains in other datasets.

As z is independent, minimizing ε amounts to g(θ, ·) learning c(·), regardless of the value of p BID0 .

For more information about error and entropy relations in the presence of noisy labels see Appendix A.

In gradient based training of ANNs, the tunable parameters of the networks θ are changed in time by the gradient updates of the loss function, in order to minimize the learning error for a particular problem at hand.

Let θ n ∈ Θ denote the tunable parameters of an ANN after n ≥ 1 training steps of SGD.

The parameters are initialized as θ 0 , which can be random or deterministic.

The set Θ can be seen as a high dimensional Euclidean space with the network parameter θ as its vector.

At the training step n, the outcome of the learning algorithm is captured by the random variablê y n which is modulated by the network function g(θ n , ·) applied to the random input data x, i.e., y n = g(θ n , x).

Therefore the SGD learning gives rise the following sequence of random variables DISPLAYFORM0 As n grows large with a successful training, the sequence of random variables converges approximately to the true labelsỹ which itself follows a joint distribution with x. Note that the above random variables are coupled through the common random variable x and the sequence of parameter updates θ n .If the probability distribution ofŷ n is denoted by p n , a first question is to see how SGD methods modify p n on the space of probability measures defined on Y. As a consequence, one can determine the trajectory of H(ŷ n ), which will be plotted later.

However it is additionally important in learning that the random variableŷ n approximates the true labels.

Therefore, a second question would be how SGD methods change the joint distribution of (ŷ n ,ỹ n ).

The answer could determine instead the trajectory of the conditional entropy H(ŷ n |ỹ n ).

We first study the trajectory of H(ŷ n ).The random variablesŷ n are defined as g(θ n , x).

Consider the sequence of random variables {θ n }.

Let T denote the set of training samples (x, y) that are obtained prior to training and independently.

In addition, let T n be a subset of T that is used at the step n for SGD update.

T n is assumed to be independent from (θ 0 , . . .

, θ n−1 ) and it is either deterministic and known all n or randomly chosen at each step.

These variations correspond to the variants of SGD.In pure gradient based methods without momentum based techniques, the network parameters obey the following recursive relation DISPLAYFORM1 where f denotes the update rule of SGD.

The model assumes that the SGD updates only depend on the parameters in the last step and the training set used in the current iteration.

We can assume that T n are i.i.d.

random variables if we neglect the effect of reusing training data in different batches.

The first conclusion is that the sequence of random variables {θ n } is a Markov chain.

The transition probability of this Markov chain can be obtained only from f and T 1 .

In that sense, the random process {θ n } is a homogeneous Markov chain.

Throughout this work, it is assumed that the Markov chain {θ n } has a stationary distribution which corresponds to the learned ANN.This proposition shows that SGD updates induce Markov property for weights of an ANN.

The sequence of random variablesŷ n = g(θ n , x) however is in general not a Markov chain, particularly because they are coupled through a common random variable x. Since we are interested in H(ŷ n ) and the distribution p n , these random variables can be decoupled by considering the random variables g(θ n , x n ) where x n are i.i.d.

random variables with the same distribution as x. Note that the value of the entropy function remains unchanged after decoupling, namely DISPLAYFORM2 The new sequence is a function of a Markov chain and i.i.d.

random variables.

The question whether the resulting sequence is a Markov chain has been addressed in Spreij FORMULA4 ; BID10 showing thatŷ n is not a Markov chain in general unless certain conditions are met by the function g(·).

Unfortunately the function g(·) is not injective and a non-injective function of a Markov chain is not Markov chain in general.

However the random variable g(θ n , x n ) can be seen as the observation of the Markov process {θ n } through a noisy memoryless channel g(·, x n ).

Therefore the random process {g(θ n , x n )} is a HMP.

See BID7 for an information theoretic survey.

If the learning is done perfectly, the HMP {g(θ n , x n )} converges to the uniform distribution of correct labels.

Since the random variables are discrete, entropy is a continuous function of the distribution p n .

Therefore as the correct labels are uniformly distributed, the entropy H(ŷ n ) approaches its maximum log K. The instantaneous entropy H(ŷ n ) would converge monotonically to the entropy of the stationary distribution log K if the sequence were to be a Markov chain.

This could explain the monotonicity of H(ŷ n ) in the experiments.

Although the sequence not a Markov chain but it is indeed a HMP, the following proposition shows that the entropy is lower-bounded by an increasing function.

Proposition 3.

Suppose that the Markov process {θ n } with the probability distribution q n has a stationary distribution q.

We have DISPLAYFORM3 The proof follows from data processing inequality for Kullback-Leibler divergence and is found in Appendix B. Note that since {θ n } is a Markov process, D(q n q) is non-increasing.

In the previous section, the non-decreasing property of H(ŷ n ) was investigated by modeling the network output as an HMP.

In the same spirit let us defineỹ n = c(x n ) to be i.i.d.

realizations of y. In the ideal situation where the network manages to learn successfully the true labels, we can saŷ y n converges toỹ n almost surely 4 .

The joint distribution of (ŷ n ,ỹ n ) specifies a point on the space of probability measures on Y × Y. The task of learning consists tuning the parameters θ n in a way that the joint distribution approaches the distribution of (ỹ n ,ỹ n ).

Therefore the gradient descent steps corresponds to a sequence of points, that is joint distributions of (ŷ n ,ỹ n ), on the space of probability measures on Y × Y with the end point ideally being the joint distribution of (ỹ n ,ỹ n ).

In this section, we investigate the gradient descent algorithm by exploring the path it takes on the space of probability measures on Y × Y 5 .

The trajectory of conditional entropy H(ŷ n |ỹ n ) is determined for the trajectory of probability distributions on the space of joint measures.

However it is in general difficult to precisely characterize this path.

Instead, one might ask how the gradient descent trajectory compares with a certain natural path on the space of distributions.

relevant question is to ask what the shortest path between these probability measures is on this space and how similar is the trajectory of SGD compared to this shortest path.

To be able to formally address this issue we require to define curves and lengths on the metric space of probability measures.

The space of probability distributions defined on the discrete space Y × Y, denoted by P(Y × Y), with total variation metric d T V (·, ·) is a simplex in a finite dimensional Euclidean space.

The total variation metric is equivalent to the L 1 -distance between the points in the corresponding Euclidean space.

A curve in this space is defined by a continuous function σ : [0, 1] → P(Y × Y).

The curve is called a shortest path if it has minimal length among all curves with endpoints σ(0) and σ(1).

Note that the length is measured in this space using L 1 -norm.

The following theorem guarantees that there is a shortest path on this space between probability measures and it can be traveled using a Markov chain.

Theorem 1.

The shortest path between two probability measures µ and ν on the space of discrete probability measures P(Y × Y) is given by tµ + (1 − t)ν for t ∈ [0, 1].

Furthermore if the probability measures are represented by row vectors there is a transition matrix Π with the stationary distribution ν such that µ n = µΠ n is on the line segment between µ and ν and lim n→∞ µ n = ν.

Proof.

Not that the space of probability distributions here is a bounded compact metric space with each two points connected by a rectifiable curve.

The existence of shortest path follows from BID4 , Corollary 2.5.20,Theorem 2.5.23).

The transition matrix in Theorem 1 is given simply by DISPLAYFORM0 where 1 = (1, 1, . . .

, 1).To see the implication of previous theorem more precisely, consider conditional distributions P (ŷ n ∈ ·|ỹ n = l) and letp l (n) ∈ R K be the following vector for l ∈ Y, that is DISPLAYFORM1 We use the compact notationP (n) ∈ R K×K for the matrix withp 0 (n), . . .

,p K−1 (n) as rows.

Note that, with the assumption thatỹ n is uniformly distributed, the joint distribution of (ŷ n ,ỹ n ) is fully determined byP (n) since P (ŷ,ỹ n ) = 1 K P (ŷ|ỹ n ).

Suppose that the initialization θ 0 is such that g(θ 0 , ·) initially maps all inputs to the same class (the first class is assumed for simplicity).

Therefore the initial distribution matrixP (0) assumes no knowledge about the input and is given bỹ DISPLAYFORM2 Ideally, in a learning algorithm, the matrixP (0) converges to an optimal distributionP * as n → ∞, that isP * = lim n→∞P (n).

If P (ŷ n =ỹ n ) = 1, we havẽ DISPLAYFORM3 Now that we set the initial distribution and stationary distributions, the following transition matrix provides a way to pass from the initial distribution to the stationary one.

Definition 1.

α-Simple Markov Learning Chain (α-SMLC) Given 0 < α < 1, the sequence of random pairs {(ŷ n ,ỹ n )} is an α-SMLC ifp l (n) =p l (n − 1)Π l , for every n ≥ 0, l ∈ Y with DISPLAYFORM4 The above construction provides a different transition matrix for eachp l (n) depending on l. The following theorem describes how an α-SMLC movesP (n) in the space of stochastic matrices.

It actually shows this construction leads to points on the shortest path between the measures.

Theorem 2.

If (ŷ n ,ỹ n ) is the n-th random pair generated of an α-SMLC, theñ DISPLAYFORM5 Proof.

c.f.

Appendix B This shows thatP (n) belongs to the continuous curve from equation 3, regardless of the choice of α.

Moreover, letŷ(t) be a version ofŷ n parametrized by t ∈ [0, 1] such that its conditional distribution withỹ n corresponds toP (n) = (1 − t)P (0) + tI. Proposition 4.

If {(ŷ n ,ỹ n )} is an α-SMLC and z follows the distribution DISPLAYFORM6 Proof.

c.f.

Appendix B Corollary 1.

In the setting of Proposition 4, if p → 0 then H(ŷ(t)|y) has one maximum at t = This result allows us to characterize the shape of the 2D curves (H(ŷ(t)), H(ŷ(t)|y)) for the above construction.

We now consider the implication of Markov assumption for error.

Defineŷ l n ∈ Y for all l ∈ Y to be random variables distributed according to the conditional probabilities P (ŷ l n = k) = P (g(θ n , x) = k|ỹ n = l) for all k ∈ Y. Note that, sinceỹ n is uniformly distributed, one can compute the joint distribution of (ŷ n ,ỹ n ) from the marginal distributions ofŷ Step ( Step (%) Proposition 5.

If {ŷ l n } is a stationary Markov Chain converging to the distributionp * = e l then DISPLAYFORM7 Corollary 2.

If {ŷ l n } is a stationary Markov Chain converging to the distributionp DISPLAYFORM8 This last results shows thatε is non-increasing with n (i.e., no over-fitting), which is a desirable property for any learning algorithm.

We will show through numerical simulations how a comparable behavior is observed for gradient descent methods.

In this section we compare gradient based learning to the α-SMLC model through empirical simulations.

First, we use the datasets where SGD is extremely successful at the classification task (ε ≈ 0), that is the MNIST dataset and the spirals dataset BID1 .

The spirals dataset constitutes a 2D-spiral classification task constructed as DISPLAYFORM0 where a ∈ [0, 1], b ∈ [0, 0.1] andỹ ∈ {0, 1, . . .

, K − 1} are independent uniformly distributed random variables and K = 3.

This dataset is divided into a training set of 50 000 samples and a test set of 2 000.

Furthermore, we train the FCNN of BID1 for the spirals dataset and use LetNet-5 (LeCun et al., 1999) for MNIST, achieving an average accuracy above 99% in both cases.

For the sake of completeness we perform more experiments on the CIFAR-10 and CIFAR-100 datasets using the DenseNet architecture from BID11 with 40 and 100 layers respectively.

FCNN and LetNet-5 are trained using Adam's optimizer, while DenseNet is trained with SGD.

More detailed explanation about the experimental setup is provided in Appendix C. We estimate use a naive estimator of entropy which consists on computing the empirical distribution of (ŷ, y) and directly calculating entropy afterwards.

This method is known to have an approximation error of BID15 , which is good enough for our experiments since N is much larger than K 2 in our datasets.

For cases with larger K one could use more sophisticated methods, such as Schürmann FORMULA6 ; BID2 .

For our experiments we introduce i.i.d.

noise to the dataset labels before training according to P (z = 0) = 1 − p and P (z = 0) = p/(K − 1).

We keep fixed α = 0.85 for all simulations.

DISPLAYFORM1 In FIG3 we show the similarity between the α-SMLC model and ANNs on the spirals and MNIST dataset.

This figure is obtained by averaging the entropy values over 100 realizations of training.

We observe ANNs move along the information plane in a similar way as the α-SMLC model, and converge to the optimal distribution.

In addition, we display the inflection point of H(y|ŷ) of the α-SMLC model for different values of p. Interestingly, as labels get noisier the inflection point occurs at a larger H(ŷ) value.

This seems to be the case for SGD learning as well.

This phenomena suggests that, as labels get noisier, a learning algorithm needs to know more about the input distribution before it can start assigning labels efficiently.

Similar conclusions can be drawn form FIG7 , which is also obtained by averaging over 100 realizations of training.

An interesting result is that, regardless of the value of p, a good learning algorithm should converge to a point that lies on the upper bound provided by Fano's inequality.

These Figures artificially mitigate the randomness induced by SGD on the trajectory by averaging over several realizations of training 6 .

How much SGD oscillates seems to depend on the experimental setup, such as the learning rate, dataset, and the structure of the classifier.

See Appendix C for examples of highly oscillating trajectories.

Our last experiment consists on investigating how underfitting and overfitting affects the trajectory of SGD.

To that end in FIG8 (a) we include an 1 norm regularization term into the loss function (details in Appendix C) controlled by a parameter λ.

As expected for sufficiently small λ the minimum error is attained.

Interestingly, as we induce underfitting by increasing this regularization coefficient the obtained models move away from Fano's bound.

This naturally leads to increased error values.

On the other hand, in FIG8 we increase the number of parameters of FCNN and reduce the dataset size in order to induce overfitting (details in Appendix C).

While overfitting leads to larger error as well as underfitting, it can be distinguished by its trajectory in the entropy-error plane.

Underfitting seems to push models away from Fano's bound while overfitting happens when an ANN is at the bound.

This experiment shows that the information plane provides a richer view beyond train and test error that allow us to observe effects that were previously hidden.

Further understanding about desired trajectories is interesting since it may allow practitioners to monitor models during training, spot undesired behaviors, and possibly tune hyperparameters accordingly.

Proof.

For independent noise z, we have DISPLAYFORM2 In BID1 it is shown that this bound is sharp when z is distributed such that Φ(H(z)) = p, thus p ≤ ε.

We generalize this result in the following theorem for an arbitrary distribution of z, under some mild conditions, and show that p ≤ ε is in fact a sharp lower bound for arbitrary z. Lemma 1.

Let z ∈ Y be a random variable with P (z = 0) = 1 − p, then DISPLAYFORM3 Proof.

Let us define β k P (z = k) and the auxiliary random variablez ∈ {1, . . .

, K − 1} with DISPLAYFORM4 and equality is attained if and only if P (ŷ =ỹ) = 1.Proof.

of Theorem 3 For the sake of notation let us define DISPLAYFORM5 From Proposition 1 we know that ε ≥ Φ(H(z)).

Since Ψ is an increasing function in the interval [0, 1 − 1 K ] and Φ is its inverse in that interval, we have that Φ is an increasing function as well.

From Lemma 1 (c.f.

Appendix B) we know that H(z) ≤ Ψ(p), this leads to DISPLAYFORM6 Then, if the bound from Proposition 1 were to be sharp, ε could reach values strictly lower than p. We will show that this is not possible.

DISPLAYFORM7 If ε < p we obtain (1 − p) < 1 − ε ≤ (1 − p) which is a contradiction, hence it must hold that ε ≥ p.

Finally, if ε = p then equation 10 yields DISPLAYFORM8 Since β max < (1 − p), this inequality holds if and only if δ 0 = 1, that is P (ŷ =ỹ) = 1.This theorem shows that the minimum expected error ε can only be attained if g(θ, ·) manages to denoise the labels, henceε = 0.

We extend this result by deriving bounds forε, given ε and p, in the following theorem.

Theorem 4.

BID0 Given DISPLAYFORM9 Proof.

of Theorem 4 DISPLAYFORM10 DISPLAYFORM11 which completes the proof.

Corollary 3.

If p < 1 2 , ε < 1 − 1 K , and DISPLAYFORM12 Proof.

Since Ψ is an increasing function in the interval [0, 1 − 1 K ], the proof follows from applying Theorem 4 on Proposition 1.In information theory there is a result of this kind, known as Mrs. Gerber's Lemma (MGL), that does not require knowledge about p and ε.

MGL provides an upper bound on H(ỹ|ŷ) given H(y|ŷ) for the case of K = 2.

This result also states that the minimum H(y|ŷ) is attained when H(ỹ|ŷ) = 0.

Since we assumedỹ to be uniformly distributed, this corresponds toε = 0, up to permutation ambiguities.

Generalizing MGL for arbitrary K is still an open question in information theory.

Nevertheless, BID12 successfully proved MGL for the cases where K is a power of 2.

We summarize that result in the following proposition.

Proposition 7.

Generalized Mrs. Gerber's Lemma for K = 2 n BID12 f 2 n (ỹ, z) = min DISPLAYFORM13 and k is an arbitrary positive integer and a b a(1 − b) + b(1 − a).We have derived inequalities that relate entropies and error values.

Then we showed that in the presence of corrupted labels, the best g(θ, ·) can do for minimizing ε is to learn c, regardless of the value of p.

Proof of Proposition 3: The proof follows from the following general theorem.

Theorem 5.

Let {y n } be an HMP defined as the observation of a Markov process {x n } through an arbitrary stationary memoryless channel with values in the state space Y. Suppose that the probability distributions on the respective state spaces of {x n } and {y n } are given by {q n } and {p n } with the stationary distribution q and p.

Then DISPLAYFORM0 Proof.

Based on the assumption above, x n and y n are related according to the conditional distribution characterized by the conditional probabilities {r(·|x) : x ∈ X}. Denote the joint distribution of x n and y n by µ n defined on X × Y and given by q n (x) × r(·|x).

The stationary joint distribution µ is defined by q × r. Using the chain rule BID5 , Theorem 2.5.3), we have: DISPLAYFORM1 .

On the other hand, the chain rule and the non-negativity of Kullback-Leibler divergence shows that: Using the fact that the stationary distribution ofŷ n is equal to the uniform distribution, we have: DISPLAYFORM2

pThen for uniformly distributed y, and z distributed according to equation 4, we can express this relation in matrix form as DISPLAYFORM0 Under review as a conference paper at ICLR 2019 80.2%

A fully connected ANN with four hidden layers of five neurons each, as FCNN, is trained on the spirals dataset.

For the MNIST dataset, the popular convolutional network called LeNet-5 LeCun et al. (1999) is used.

To train these networks we let the learning rate γ ∈ R start at a given γ max ∈ R and then decay by 40% per epoch until reaching some given minimum learning rate γ min <

γ max , that is γ = max{γ max 0.6 epoch , γ min }.

For the CIFAR-10 dataset we train a 100 layer DenseNet architecture as done in BID11 , but we stop the training after 10 epochs instead of the original 300 used by the authors.

The different configurations used for these experiments are summarized in TAB1 .

For instance, FIG1 shows 1 realization of SGD training for DenseNet on CIFAR-100.

In that figure we observe a rather stable trajectory, with not much oscillation.

However, in Figure 8 we average over 2 realizations of SGD learning for DenseNet on CIFAR-10 and obtain highly oscillating trajectories.

As expected, both trajectories follow a similar behavior as the α-SMLC model.

In FIG8 (a) the same model as in FIG3 is used but an additional regularization term is added to the loss function, that is λ w 1 where w is a vector containing all the weights in the network.

In FIG8 (b) a FCNN with a single hidden layer of size 100, and tanh activations, is used and the dataset sizes are reduced according to 3.

Other parameters are γ max = 10 −1 , γ max = 10 −2 , and the number of epochs is 10 000.

@highlight

We look at SGD as a trajectory in the space of probability measures, show its connection to Markov processes, propose a simple Markov model of SGD learning, and experimentally compare it with SGD using information theoretic quantities. 

@highlight

Constructs a Markov chain that follows a shorted path in TV metric on P and shows that trajectories of SGD and \alpha-SMLC have similar conditional entropy

@highlight

Studies the trajectory of H(\hat{y}) versus H(\hat{y}|y) on the information plane for stochastic gradient descent methods for training neural networks

@highlight

Describes SGD from the point of view of the distribution p(y',y) where y is (a possibly corrupted) true class-label and y' a model prediction.