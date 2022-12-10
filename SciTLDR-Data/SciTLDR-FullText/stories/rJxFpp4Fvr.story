The performance of deep neural networks is often attributed to their automated, task-related feature construction.

It remains an open question, though, why this leads to solutions with good generalization, even in cases where the number of parameters is larger than the number of samples.

Back in the 90s, Hochreiter and Schmidhuber observed that flatness of the loss surface around a local minimum correlates with low generalization error.

For several flatness measures, this correlation has been empirically validated.

However, it has recently been shown that existing measures of flatness cannot theoretically be related to generalization: if a network uses ReLU activations, the network function can be reparameterized without changing its output in such a way that flatness is changed almost arbitrarily.

This paper proposes a natural modification of existing flatness measures that results in invariance to reparameterization.

The proposed measures imply a robustness of the network to changes in the input and the hidden layers.

Connecting this feature robustness to generalization leads to a generalized definition of the representativeness of data.

With this, the generalization error of a model trained on representative data can be bounded by its feature robustness which depends on our novel flatness measure.

Neural networks (NNs) have become the state of the art machine learning approach in many applications.

An explanation for their superior performance is attributed to their ability to automatically learn suitable features from data.

In supervised learning, these features are learned implicitly through minimizing the empirical error E emp (f, S) = 1 /|S| (x,y)∈S (f (x), y) for a training set S ⊂ X × Y drawn iid according to a target distribution D : X × Y → [0, 1], and a loss function : Y × Y → R + .

Here, f : X → Y denotes the function represented by a neural network.

It is an open question why minimizing the empirical error during deep neural network training leads to good generalization, even though in many cases the number of network parameters is higher than the number of training examples.

That is, why deep neural networks have a low generalization error

which is the difference between expected error on the target distribution D and the empirical error on a finite dataset S ⊂ X × Y.

It has been proposed that good generalization correlates with flat minima of the non-convex loss surface (Hochreiter & Schmidhuber, 1997; 1995) and this correlation has been empirically validated (Keskar et al., 2016; Novak et al., 2018; Wang et al., 2018) .

Thus, for deep neural networks trained with stochastic gradient descent (SGD), this could present a (partial) explanation for their generalization performance (Zhang et al., 2016) , since minibatch SGD tends to converge to flat local minima (Zhang et al., 2018; Jastrzębski et al., 2017) .

This idea was elaborated on by Chaudhari et al. (2016) who suggest a new training method that favors flat over sharp minima even at the cost of a slightly higher empirical error -indeed solutions found by this algorithm exhibit better generalization performance.

Similarly, Dziugaite & Roy (2017) augment the loss to improve generalization and find that this promotes flat minima.

However, as Dinh et al. (2017) remarked, current flatness measures-which are based only on the Hessian of the loss function-cannot theoretically be related to generalization: For deep neural networks with ReLU activation functions, there are layer-wise reparameterizations that leave the network function unchanged (hence, also the generalization performance), but change any measure derived only from the loss Hessian.

Another, more intuitive explanation for generalization is that the function generalizes well if the extracted features encode a semantic similarity of the input that is robust to small changes-both in the input and the features.

This allows to generalize from the training set to novel, sufficiently similar data.

Starting from such a concept of robustness with respect to changes of features, we derive a measure of flatness that is invariant under the mentioned reparameterizations and that reduces to the well-known ridge regression penalty in the special case of a linear regression.

This brings three seemingly related properties into our focus: flatness, robustness, and generalization.

The exact relationship, however, between flatness of the loss surface around local minima (measuring changes of the empirical error for perturbations in parameter space), robustness (measuring changes of the error for perturbations in either input or feature space), and generalization (performance on unseen data from the target distribution) is not well-understood.

This paper provides new insights into this relationship.

The notion of feature robustness proposed in this paper measures the robustness of a function f = ψ • φ (e.g., a neural network) toward local changes in a feature space.

That is, f can be split into a composition of functions f (x) = (ψ • φ)(x) for x ∈ X , φ : X → R m and ψ : R m → Y. The function φ is considered as a feature extraction, mapping the input X into a feature space R m , while the function ψ corresponds to the model (e.g., a classifier) with R m as its domain (see Figure 1 for illustration).

It is the feature space defined by φ where we measure robustness toward small perturbations.

For neural networks, the activation values of any but the output layer can be viewed as a feature space.

A function f is called -feature robust on a dataset S ⊂ X × Y if small changes in the feature space defined by φ do not change the empirical error by more than .

This differs from the notion of robustness defined by Xu & Mannor (2012) using a cover of the sample space, which has been theoretically connected to generalization.

Flatness of the loss surface, however, is a local property and we require a more local version of robustness to derive a connection between flatness and robustness.

Then, indeed, feature-robustness is upper bounded by the proposed flatness measure.

To finally connect the two local properties of robustness and flatness to generalization, we necessarily need a notion describing how representative the given samples are for the true distribution.

We define a suitable notion, leading to an upper bound for the generalization error given by feature robustness together with representativeness.

In summary, our contributions are as follows: (i) For models of the form f (x) = (ψ • φ)(x) (e.g. most (deep) neural networks) that split up into a feature extractor φ and a model ψ on the feature space defined by φ, we define a property of feature robustness that measures the change of the loss function under small perturbations of the features.

This property is strongly related to flatness of the loss surface at local minima. (ii) We propose a novel flatness measure.

For neural networks with ReLU activation functions, it is invariant under layer-wise reparameterization, addressing a shortcoming of previous measures of flatness. (iii) We define a suitable notion of representativeness of a dataset connecting feature robustness to the generalization error in form of an upper bound. (iv) The proposed flatness measure is empirically shown to strongly correlate with good generalization performance.

Thereby, we recover Hessian based quantities as measures of flatness.

We will define a notion of robustness in feature space R m for the model f = (ψ • φ) : X → Y, which depends on a small number δ > 0, a training set S, and a feature selection defined by a matrix A ∈ R m×m of operator norm ||A|| ≤ 1.

In the case of neural networks split into a composition according to Figure 1 , traditionally, the activation values φ j (x) of neurons are considered as feature values.

The feature value defined by the j-th neuron in the feature space φ(x) ∈ R m can be written as φ j (x) = φ(x), e j , where e j denotes the j-th unit vector and ·, · the scalar product in R m .

However, it was shown by Szegedy et al. (2013) that, for any other direction v ∈ R m , ||v|| = 1, the values φ(x), v = proj v φ(x) obtained from the projection φ(x) onto v, can be likewise semantically interpreted as a feature.

We can single out the feature defined by v from φ(x) by multiplication with the projection matrix E v = vv T .

Similarly, multiplication of φ(x) with a matrix A corresponds to a weighted selection of rank(A)-many features in parallel (e.g., projection matrices on d-dimensional subspaces correspond to the selection of d many features).

This justifies our terminology considering a matrix A as a feature selection.

The same way that, for a sample input x, non-activated neurons φ j (x) = 0 are considered as non-expressed features, we call a selection of features defined by matrix A as non-expressed whenever Aφ(x) = 0.

We define our notion of feature robustness.

In words, feature robustness measures the mean change in loss over a dataset under small changes of features in the feature space.

Hereby, a matrix A determines which features shall be perturbed.

For each sample, the perturbation is linear in the expression of the feature.

Thereby, we only perturb features that are relevant for the output for a given sample and leave feature values unchanged that are not expressed (in the sense explained above).

With

the precise definition is given as follows:

m×m denotes a probability space over matrices such that ||A|| ≤ 1 for all A ∈ A, then we call the model

We will bound feature robustness at local minima for a dataset S uniformly over all feature selections A and dependent on δ.

With our interpretation, this corresponds to an upper bound of the change in loss when perturbing features in feature space R m .

In Appendix C.1 we note how feature robustness is related to noise injection in the layer of consideration, which is known to be related to generalization (An, 1996; Bishop, 1995) .

Consider a function f (x, w) = ψ(w, φ(x)) = g(wφ(x)), where ψ is the composition of a twice differentiable function g : R d → Y and a matrix product with a matrix w ∈ R d×m .

As before, φ : X → R m can be considered as a feature extractor.

We fix a loss function : Y × Y → R + for supervised learning and let w * denote a choice of parameters for which the empirical error E emp (w, S) = 1 /|S| (x,y)∈S (f (x, w), y), considered as a function on w, is at a local minimum on the training set S = {(x i , y i ) | i = 1, . . .

, N }.

In the following, we write z = φ(x).

For any matrix A ∈ R m×m we have that

Therefore,

The latter is the empirical error E emp (w + δwA, S) of the model f on the dataset S at parameters w + δwA. If δ is sufficiently small, then by Taylor expansion of E emp (w, S) with respect to parameters w around the critical point w * , we have that

with HE emp (w * , S) denoting the Hessian of the empirical error with respect to w, ·, · the scalar product with vectorized versions of the parameters and ||w|| F the Frobenius norm of w.

Subtracting E emp (w * , S) from (5), maximizing over matrices ||A|| ≤ 1 and using (4), we get that, for any feature selection A, the function (2) defining feature robustness is bounded by

where λ H max (w * ) denotes the largest eigenvalue of the Hessian HE emp (w * , S) of the empirical error at w * .

Here we used the identity that max ||x||=1 x T M x = λ M max for any symmetric matrix M , and that for matrices of norm ||A|| ≤ 1, we have ||w * A|| F ≤ ||w * || F .

We show details of the proof of (6) in the appendix.

We summarize the connection between feature robustness and flatness in terms of the Hessian in the following theorem.

Theorem 2.

Let : Y × Y → R + denote a loss function, δ a strictly positive (small) real number, A ∈ R m×m a matrix with ||A|| ≤ 1, and let f (x, w) = g(wφ(x)) be a model with g an arbitrary twice differentiable function on a matrix product of parameters w and the image of x under a (feature) function φ.

Let w * denote a local minimum of the empirical error on a dataset S.

Motivated by the relation of feature robustness with the Hessian H, we define a novel measure of flatness.

Note that the Hessian is computed with respect to those parameters w that are applied linearly on the feature space φ(X ) ⊆ R m .

Definition 3.

Let : Y × Y → R + denote a loss function and f (x, w) = g(wφ(x)) be a model with g : R m → Y an arbitrary twice differentiable function on a matrix product of parameters w and the image of x under a (feature) function φ : X → R m .

Then κ φ (w) shall denote a flatness measure of the loss surface defined by

Note that small values of κ φ (w) indicate flatness and high values indicate sharpness.

Linear regression with squared loss In the case of linear regression, f (x, w) = wx ∈ R (X = R d , g = id and φ = id), for any loss function , we compute second derivatives with respect to the parameters

If is the squared loss function (ŷ, y) = (ŷ − y) 2 , then ∂ 2 /∂ŷ 2 = 2 and the Hessian is independent of the parameters w.

In this case, κ id = c · ||w|| 2 with a constant c = 2λ max ( x∈S xx t ) and the measure κ id reduces to (a constant multiple of) the well-known Tikhonov (ridge) regression penalty.

We consider neural network functions

of a neural network of L layers with nonlinear activation function σ.

We hide a possible non-linearity at the output by integrating it in a loss function chosen for neural network training.

By letting

) denote the output of the composition of the first l−1 layers and g l (z) = w L σ(. . .

σ(z+b l ) . .

.)+b L the composition of the activation function of the l-th layer together with the rest of layers, we can write for each layer l, f (x, w l ) = g l (w l φ l (x)).

Using (7) we obtain for each layer of the neural network a measure of flatness at parameter values w:

with λ H,l max (w l ) the largest eigenvalue of the Hessian of the empirical error with respect to the parameters of the l-th layer.

By Theorem 2, κ l is related to small changes of feature values in layer l.

Corollary 4.

Let f denote a neural network function of an L-layer fully connected neural network.

For each layer l, 1 ≤ l ≤ L of size n l , let A ∈ R n l ×n l with ||A|| ≤ 1 correspond to feature selections of features in the l-th layer of the neural network.

Let w l * denote weights of the l-th layer at a local minimum of the empirical error.

Then the neural network is ((δ, S, A), )-feature robust in layer l at w * for =

For an everywhere well-defined Hessian of the loss function, we assumed our network function to be twice differentiable.

With the usual adjustments (equations only hold almost everywhere in parameter space), we can also consider neural networks with ReLU activation functions.

In this case, Dinh et al. (2017) noted that a linear reparameterization of one layer, w l → λw l for λ > 0, can lead to the same network function by simultaneously multiplying another layer by the inverse of λ,

Representing the same function, the generalization performance remains unchanged.

However, this linear reparameterization changes all common measures of the Hessian of the loss.

This constitutes an issue in relating flatness of the loss curve to generalization.

We counteract this behavior by the multiplication with ||w l || 2 .

Theorem 5.

Let f = f (w 1 , w 2 , . . . , w L ) denote a neural network function parameterized by weights w l of the l-th layer.

Suppose there are positive numbers

We provide a proof in Appendix A.2.

An Averaging Alternative Experimental work (Ghorbani et al., 2019) suggests that the spectrum of the Hessian has a lot of small values and only a few large outliers.

In this case, our flatness measure serving as an upper bound for feature robustness is governed by the outlier.

However, feature robustness for different feature selections is governed by different eigenvalues of the Hessian, according to (5).

We therefore consider the trace as an average of the spectrum.

We will show that this tracial averaging corresponds to feature robustness on average over all orthogonal feature selection matrices.

The following theorem specifies this connection between feature robustness and the unnormalized trace T r(HE emp (w * )) of the empirical error at a local minimum w * .

The details and the proof can be found in Appendix A.3.

Theorem 6.

Let : Y × Y → R + denote a loss function, δ a strictly positive (small) real number, and let f (x, w) = g(wφ(x)) be a model with g an arbitrary twice differentiable function on a matrix product of parameters w ∈ R d×m and the image of x under a (feature) function φ.

Let w * denote a local minimum of the empirical error on a dataset S and O m ⊂ R m×m denote the set of orthogonal matrices.

Then, (i) for each feature selection matrix

We therefore consider the unnormalized trace as a suitable and efficiently computable measure of flatness and define for each layer l of a neural network

The same arguments as those used to prove Theorem 5 also show the measure κ l T r to be independent with respect to the same layer-wise reparameterizations.

The analogue of Corollary 4 is as follows.

Corollary 7.

Let f denote a neural network function of an L-layer fully connected neural network.

n l ×n l denote the set of orthogonal feature selections in the l-th layer of the neural network.

Let w l * ∈ R n l+1 ×n l denote weights of the l-th layer at a local minimum of the empirical error.

Then the neural network is ((δ, S, O n ), )-feature robust in layer l on average over O n at w * for = T denotes the Gauss-Newton approximation of the loss Hessian.

Therefore, the Fisher-Rao norm considers the second partial derivative only into the direction defined by the given weight values w. Our measure considers all directions of moving away from a local minimum.

In particular, in contrast to these measures, we take the full spectrum of the Hessian into account, which results in a natural measure of flatness around a local minimum.

We also came across preprints by Tsuzuku et al. (2019) and Rangamani et al. (2019) , which propose a similar measure of flatness.

While the first one derives the flatness measure from a PAC Baysian approach, the latter considers the Riemannian metric on the quotient manifold obtained from the equivalence relation given by the refactorization of layers as above.

In this section we aim to study the relation between flatness, feature robustness and the generalization error (defined in (1)).

The connection of flat local minima with generalization in PAC Baysian bounds has been considered in several works (Neyshabur et al., 2017; Tsuzuku et al., 2019; Dziugaite & Roy, 2017) .

Arora et al. (2018) relates noise injection to generalization under the same setting.

The work of McAllester (1998; 1999) and Langford & Caruana (2001) initiated the PAC-Baysian approach to generalization, which measures the generalization error (usually for the 0-1 loss) of stochastic classifiers.

This leads to bounds on the expected true error over a distribution of models Q in terms of the expectation of the empirical error over Q and the Kullback-Leibler divergence between Q and some prior distribution P .

For example, Neyshabur et al. (2017) use work by McAllester (2003) to derive an inequality relating the generalization error for the 0-1 loss and expected sharpness

Here, P denotes a "prior" distribution which is fixed before seeing any data and KL denotes the KullbackLeibler divergence.

If we aim to use distributions with local support (as considered in feature robustness and the Taylor expansion relating feature robustness to flatness) with a data-independent prior P , the KL term goes to infinity as the distribution ν becomes increasingly localized.

Since feature robustness as a local property is related to generalization (Morcos et al., 2018), we aim to connect the local properties of flatness and feature robustness to generalization of a specific model by following a different approach.

Our approach will be independent of the loss function and work in the sample space instead of averages over models in parameter space.

Since feature robustness is a local property in neighborhoods around the points (x, y) ∈ S, to connect feature robustness to generalization we necessarily need an assumption of representativeness of the given data samples S. A simple computation shows that

The first term is exactly feature robustness on average over a probability distribution A of feature matrices.

For the second term, we accordingly define a notion on datasets S that describes how well the loss on the true distribution can be approximated by certain probability distributions.

The distributions we consider are composed of a dataset and (local) probability distributions around its points suitably restricted to local distributions λ i and ν i centered around the origin 0.

With Ω a collection of families Λ as above and H a hypothesis space, we say that

Interestingly, we naturally derived a definition of representativeness which is a generalization of classical -representativeness (see e.g. Definition 4.1 in (Shalev-Shwartz & Ben-David, 2014)), justifying the terminology.

Indeed, let Λ 0 denote the family of probability distributions where each λ i = δ 0 and ν i = δ 0 have full weight on the origin.

Then S is ( , {Λ 0 })-representative exactly when S is -representative in the classical sense.

Further, if S is -representative and S is ( , Ω)-representative for some Ω containing Λ 0 , then ≤ .

In our setting of a model f (x) = (ψ •φ)(x), which is split up into a feature extractor φ and a model ψ, we consider (φ(S), Λ)-representativeness for model ψ and specific choices for Λ = Λ δ,A .

Here, Λ δ,A is a family of probability distributions induced by a distribution A on feature matrices A such that ||A|| ≤ δ as follows: We assume that a Borel measure µ A is defined by a probability distribution A on matrices R m×m .

We then define Borel measures

Then λ i is the probability distribution defined by µ i .

We fix the distributions ν i = δ 0 and denote the set containing all families of distributions (λ i , ν i ) that can be generated this way by A δ .

The following result is a direct consequence of Equation 13 and our Definition 9.

Hence, for generalization we need a model that is feature-robust and training data that is sampled densely enough.

In the trivial case with A = δ 0 the distribution with full weight on the 0-matrix, we can choose δ = 0 to obtain = 0 and E gen ≤ .

The more feature robust a model is, the larger δ we can consider to use the flexibility of choosing a nontrivial A to lower the bound on representativeness and therefore the generalization error.

We hope that in future work it will be possible to find suitable distributions A that lead to computable generalization bounds.

In this section we empirically validate the practical usefulness of the proposed flatness measure.

A correlation between generalization and Hessian-based flatness measures at local minima has been observed previously, but the results of Dinh et al. (2017) questioned the usefulness of these measures.

We show that our measure does not only overcome the theoretical issues, but also preserves the strong correlation with the generalization error.

Previous works mostly use accuracy of the trained model on the testing dataset (Rangamani et al., 2019; Keskar et al., 2016) for evaluating the generalization properties of the achieved minimum.

Nevertheless this does not directly correspond to the theoretical definition of the generalization error (1).

For measuring the generalization error, we employ a Monte Carlo approximation of the target distribution defined by the testing dataset and measure the difference between loss value on this approximation and empirical error.

In order to track the correlation of the flatness measure to the generalization error, sufficiently different minima should be achieved by training.

The most popular technique is to train the model with small and large batch size (Rangamani et al., 2019; Keskar et al., 2016; Novak et al., 2018; Wang et al., 2018) , which we also employed.

A neural network (LeNet5 (LeCun et al.) ) is trained on CIFAR10 multiple times until convergence with various training setups.

This way, we obtain network configurations in multiple local minima.

In particular four different initialization schemes were considered (Xavier normal, Kaiming uniform, uniform in (−0.1, 0.1), normal with µ = 0 and σ 2 = 0.1), with four different mini-batch sizes (4, 32, 64, 512) and corresponding learning rates to keep the ration between them equal (0.001, 0.008, 0.02, 0.1) for the standard SGD optimizer.

Each of the setups was run for 9 times with different random initializations.

Here the generalization error is the difference between summed error values on test samples multiplied by 5 (since the size of the training set is 5 times larger) and summed error values on the training examples.

Figure 2 shows the approximated generalization error with respect to the flatness measure (for both κ l and κ l T r with l = 5 corresponding to the last hidden layer) for all network configurations.

The correlation is significant for both measures, and it is stronger (with ρ = 0.91) for κ 5 T r .

This indicates that taking into account the full spectrum of the Hessian is beneficial.

To investigate the invariance of the proposed measure to reparameterization, we apply the reparameterization discussed in Sec. 4 to all networks using random factors in the interval [5, 25] .

The impact of the reparameterization on the proposed flatness measure based on the trace in comparison to the traditional one is shown in Figure 3 .

While the proposed flatness measure is not affected, the one purely based on the Hessian has very weak correlation with the generalization error after the modifications.

To verify the relation described by Equation 6, we also compared feature robustness with δ = 0.001 and feature matrices A that have only one non-zero value 1 on the diagonal.

Figure 4 shows that up to outliers the robustness is bound by the flatness measure.

Additional experiments conducted on MNIST dataset are described in Appendix E, where we obtain correlation factors between the generalization error and tracial flatness κ

We established a theoretical connection between flatness, feature robustness and, under the assumption of representative data, the generalization error.

The relation between feature robustness and Hessianbased flatness measures has been established for κ l , which takes into account the maximum eigenvalue of the Hessian, and κ l T r , which uses the trace instead.

Empirically, the measure κ l T r based on the trace of the Hessian shows a stronger correlation with the generalization error.

This is not surprising, since it takes into account the whole spectrum of the Hessian and every eigenvalue corresponds to a feature selection matrix of feature robustness.

The tracial measure can be related to feature robustness by either bounding the maximum eigenvalue of the loss Hessian by its unnormalized trace or by averaging feature robustness over all orthogonal matrices A ∈ O m .

It is interesting to note that strong feature robustness does not exclude the possibility of adversarial examples, first observed by Szegedy et al. (2013) , since large changes of loss for individual samples (i.e. adversarial examples) may be hidden in the mean in the definition of feature robustness.

In Appendix C.2 we briefly discuss the freedom of perturbing individual points by suitable feature selection matrices A.

In contrast to existing measures of flatness, our proposed measure is invariant to layer-wise reparameterizations of ReLU networks.

However, we note that other reparameterizations are possible, e.g., we can use the positive homogeneity and multiply all incoming weights into a single neuron by a positive number λ > 0 and multiply all outgoing weights of the same neuron by 1 /λ.

While the Fisher-Rao norm suggested by Liang et al. (2019) is invariant to such reparameterizations, our proposed measures of flatness κ l and κ l T r are in general not.

In principle, variations of our flatness measures can be found that are invariant to such reparameterizations as well (see Appendix B) but their analysis, except for some empirical evaluations in Appendix E, is left for future work.

The second term in the generalization bound of Theorem 10 is given by our notion of representativeness.

In order to find specific bounds for the -representativeness of (S, A δ ), a distribution over matrices is required that induces a distribution which is similar to a localized kernel density estimation (KDE).

While our notion of representativeness is a generalization of classical representativeness, it remains open whether it is efficiently computable.

The more feature robust a model is, the more freedom there is to finding specific distributions over matrices that lead to bounds on the generalization error.

In Appendix D we give a computation of representativeness for a KDE with Gaussian kernels.

Taking things together, we proposed a novel and practically useful flatness measure that strongly correlates with the generalization error.

We theoretically investigated this connection by relating this measure to feature robustness.

This notion of robustness, together with a novel notion of representativeness provides a link to the generalization error.

To the best of our knowledge, this yields the first theoretical connection between a notion of robustness, flatness of the loss surface, and generalization error and can help to better understand the performance of deep neural networks.

First note that for ||A|| ≤ 1,

From (4) and (5) we get

≤ max

where we used the identity that max ||x||=1 x T M x = λ M max for any symmetric matrix M .

In this section, we discuss the proof to Theorem 5.

Before starting with the formal proof, we discuss the idea in a simplified setting to separate the essential insight from the complicated notation in the setting of neural networks.

Let F,F : R d → R denote twice differentiable functions such that F (w) =F (λw) for all w and all λ > 0.

Later, w will correspond to weights of a specific layer of the neural network and the functions F andF will correspond respectively to the neural network functions before and after reparameterizations of possibly all layers of the network.

We show that

Indeed, the second derivative ofF at λw with respect to coordinates w i , w j is given by the differential quotient as

Since this holds for all combinations of coordinates, we see that HF (λw) = 1 /λ 2 HF (w) for the Hessians of F andF , and hence ||λw|| 2 HF (λw) = λ 2 ||w|| 2 1 λ 2 HF (w) = ||w|| 2 HF (w).

We are given a neural network function f (x; w 1 , w 2 , . . .

, w L ) parameterized by weights w i of the i-th layer and positive numbers λ 1 , . . .

, λ L such that f (x; w 1 , w 2 , . . .

, w L ) = f (x; λ 1 w 1 , λ 2 w 2 , . . .

, λ L w L ) for all w i and all x. With w defined by

where

is the product of the squared norm of vectorized weight matrix w l with the maximal eigenvalue of the Hessian of the empirical error at w with respect to parameters w l .

Let F (u) := (x,y)∈S (f (x; w 1 , w 2 , . . .

, u, . . .

, w L ), y) denote the loss as a function on the parameters of the l-th layer before reparameterization.

Further, we letF (v) :=

, y) denote the loss as a function on the parameters of the l-th layer after reparameterization.

We define a linear function η by η(u) = λ l u. By assumption, we have thatF (η(w l )) = F (w l ) for all w l .

By the chain rule, we compute for any variable

Similarily, for second derivatives, we get for all i, j, s, t,

Consequently, the Hessian H of the empirical error before reparameterization and the HessianH after reparameterization satisfy H(w l , S) = λ 2 l ·H(λ l w l , S) and also λ

Proof.

(i) This is just a corollary of Theorem 2 using the trivial bound that the maximal eigenvalue is bounded by the unnormalized trace (sum of eigenvalues) for positive semidefinte matrices (where all eigenvalues are positive).

(ii) We consider the set of orthogonal matrices A ∈ O m as equipped with the (unique) normalized Haar measure.

(For the definition of the Haar measure, see e.g. Krantz & Parks (2008) .)

We need to

with F(δ, S, A) defined as in (2).

Using (4) and (5) we get, similarly to (6),

with ·, · the scalar product with vectorized versions of

We consider the vectorization of w * A ∈ R dm given by (w 1 * , . . .

, w d * ) T .

By Lemma 11 below, we get

Here, the notation HE emp (w j * , S) refers to the empirical error at w * but the derivatives are only taken over the parameters in the row w j * .

If w i * = 0, then by Proposition 3.2.1 of Krantz & Parks (2008) and the change of variables formula for measures, we get

for all 1 ≤ i, j ≤ d, where the latter expectation is taken over the normalized (uniform) Hausdorff measure over the sphere S m−1 ⊂ R m .

Now, using the unnormalized trace T r([h i,j ]) = i h i,i we compute with the help of the so-called Hutchinson's trick:

Note that zz

is a constant multiple of the identity matrix.

Putting things together we have

Lemma 11.

(i) Let H = [H i,j ] i,j be a positive semidefinite matrix in R 2m×2m that consists of

Proof.

(i) By definition, H is positive semidefinite if (H is symmetric and) z T Hz ≥ 0 for all z.

is positive definite together with (i), we

We are given a function f (x) = (ψ • φ)(x).

By assumption, f is ((δ, S, A), )-feature robust for all matrices ||A|| ≤ 1, which implies that

Further, we are given that φ(S) is ( , A δ )-representative for a hypothesis space H such that ψ ∈ H. By Definition 9 (ii) this means that there is some

That is, by Definition 9 (i),

Since Λ δ,A = (λ i , δ 0 ) i ∈ A δ , there exists a probability distribution A of matrices ||A|| ≤ 1 (so that ||δA|| ≤ δ) such that 1 |S|

(21) Putting things together, we get for the generalization error E gen (f, S) of model f ,

≤ + .

We present additional measures of flatness we have considered during our study.

The original motivation to study additional measures was given by the observation that there are other possible reparameterizations of a fully connected ReLU network than suitable multiplication of layers by positive scalars: We can use the positive homogeneity and multiply all incoming weights into a single neuron by a positive number λ > 0 and multiply all outgoing weights of the same neuron by 1 /λ.

Our previous measures of flatness κ l and κ l T r are in general not independent of the latter reparameterizations.

We therefore consider, for a layer l of size n l , feature robustness only for projection matrices E j ∈ R n l ×n l having zeros everywhere except a one at position (j, j).

At a local minimum w * of the empirical error, this leads to

where w l * (j) denotes the j-th column vector of weight matrix w l of layer l, and we only consider the Hessian with respect to these weight parameters.

We define for each layer l and neuron j in that layer a flatness measure by

For each l and j, this measure is invariant under all linear reparameterizations that do not change the network function.

The proof of the following theorem is given in Section B.1 Theorem 12.

Let f = f (w 1 , w 2 , . . . , w L ) denote a neural network function parameterized by weights w i of the i-th layer.

Suppose there are positive numbers λ

such that the products w λ l obtained from multiplying weight w

We define a measure of flatness for a full layer by combinations of the measures of flatness for each individual neuron.

Since each of the individual expressions is invariant under all linear reparameterizations, so are the maximum and sum.

Analogous to Theorem 2, we get an upper bound for feature robustness for projection matrices E j .

Theorem 13.

Let f denote a neural network function of a L-layer fully connected neural network.

For each layer l, 1 ≤ l ≤ L of size n l let E j ∈ R n l ×n l denote the projection matrix containing only zeros except a 1 at position (j, j).

Let w l * denote weights of the l-th layer at a local minimum of the empirical error.

One Value for all layers Our measure of flatness are strongly related to feature robustness, which evaluates the sensitivity toward small changes of features.

In a good predictor, generalization behavior should correlate with the amount of change of the loss under changes of discriminating features.

For neural networks, we can consider the output of each layer as a feature representation.

Each flatness measure κ l is then related by Corollary 13 to changes of the features of the l-th layer.

It is however clear that a low value of κ l for a specific layer l alone cannot explain good performance.

We therefore specify a common bound for all layers.

Denoting by w * the set of weights from all layers combined, we have ||w l * || F ≤ ||w * || F for all l.

Therefore, no matter which layer with activation values φ l (x i ) for each x i ∈ S we are perturbing with a matrix

and κ(w * ) = ||w * || 2 F · λ H max (w * ) can be considered as a common measure for all layers.

However, κ(w * ) is not invariant under the reparameterizations considered in Theorem 5.

We therefore consider more simple common bounds by combinations of the individual terms κ l , e.g. by taking the maximum of κ l over all layers, κ max (w * ) := max l κ l (w * ), or the sum κ Σ (w * ) := Table 1 summarizes all our measures of flatness, specifying whether each measure is defined per network, layer or neuron, and whether it is invariant layer-wise multiplication by a positive scalar (as considered in Theorem 5) or invariant under all linear reparameterization (as considered in Theorem 12).

As in Subsection A.2, we first present the idea in a simplified setting.

For the proof of Theorem 12 we need to consider the case when we multiply coordinates by different scalars.

Let F : R 2 → R denote twice differentiable functions such that F (v, w) =F (λv, µw) for all v ∈ R, w ∈ R and all λ, µ > 0.

In the formal proof, v, w will correspond to two outgoing weights for a specific neuron, while again F andF correspond to network functions before and after reparameterizations of all possibly all weights of the neural network.

Then

for all v, w and all λ, µ > 0.

Indeed, the second derivative ofF at (λv, µw) with respect to coordinates v, w is given by the differential quotient as

∂v∂w .

From the calculation above, we also see that

, and

It follows that

We are given a neural network function f (x; w 1 , w 2 , . . .

, w L ) parameterized by weights w i of the i-th layer and positive numbers λ

such that the products w λ l obtained from multiplying weight w

for all w i and all x. We aim to show that

for each j and l where ρ l (j)(w) = w l (j) T HE emp (w l (j), S)w l (j), w l (j) denotes the j-th column of the weight matrix at the l-th layer and HE emp (w l (j), S) denotes the Hessian of the empirical error with respect to the weight parameters in w l (j).

Similar to the above, we denote by w l (j) λ the product obtained from multiplying weight w l (j) i = w

The proof is very similar to the proof of Theorem 5, only this time we have to take the different parameters λ

into account.

For fixed layer l, we denote the j-th column of w l and w l (j).

denote the loss as a function on the parameters of the j-th column in the l-th layer before reparameterization and

Feature robustness is related to noise injection in the layer of consideration.

By defining a probability measure P A on matrices A ∈ R m×m of norm ||A|| ≤ 1, we can take expectations over matrices.

An expectation over such matrices induces for each sample x ∈ X an expectation over a probability distribution of vectors ξ ∈ R m with ||ξ|| ≤ ||φ(x)||.

We find the induced probability distribution P x from the measure P x defined by P x (T ) = P A ({A | Aφ(x) ∈ T }) for a measurable subset T ⊆ R m .

Then,

The latter is robustness to noise injection according to noise distribution P x for sample x in the feature space defined by φ.

Large changes of loss (adversarial examples) can be hidden in the mean in the definition of feature robustness.

We have seen that flatness of the loss curve with respect to some weights is related to the mean change in loss value when perturbing all data points x i into directions Ax i for some matrix A. For a common bound over different directions governed by the matrix A, we restrict ourselves to matrices ||A|| ≤ 1.

One may therefore wonder, what freedom of perturbing individual points do we have?

At first, note that for each fixed sample x i0 and direction z i0 there is a matrix A such that Ax i0 = z i0 , so each direction for each datapoint can be considered within a bound as above.

We get little insight over the change of loss for this perturbation however, since a larger change of the loss may go missing in the mean change of loss over all data points considered in the same bound.

The bound involving κ(w * ) from above does not directly allow to check the change of the loss when perturbing the samples x i independently into arbitrary directions .

For example, suppose we have two samples close to each other and we are interested in the change of loss when perturbing them into directions orthogonal to each other.

Specifically, suppose our dataset contains the points (1, 0, 0, . . .

, 0) and (1, , 0, . . . , 0) for some small , and we aim to check how the loss changes when perturbing (1, 0, 0, . . . , 0) into direction (1, 0, 0, . . . , 0) and (1, , 0, . . . , 0) orthogonally into direction (0, 1, 0, . . . , 0).

To allow for this simultaneous change, our matrix A has to be of the form

Hence, our desired alterations of the input necessarily lead to a large matrix norm ||A|| and our attainable bound with ||A|| 2 κ(w * ) becomes almost vacuous.

Feature robustness is not restricted to fully connected neural networks.

In this section, we briefly consider convolutional layers w * x.

Using linearity, we get w * (x + δx) = (w + δw) *

x. What about changes (w + δwA) for some matrix A?

Since convolution is a linear function, there is a matrix W such that − −− → w * x = W x and there is a matrix W A such that −−−−→ wA * x = W A x. We assume that the convolutional layer is dimensionality-reducing, W ∈ R n×m , m < n and that the matrix W has full rank, so that there is a matrix V with W V = I m .

1 Then

As a consequence, similar considerations of flatness and feature robustness can be considered for convolutional layers.

Before we proof this proposition it is important to note that this result-in its current form-cannot be used to obtain a generalization bound using Theorem 10: In Proposition 14, Λ = (λ i × ν i ) is chosen such that P (λi×νi) (z) = K h (z), where K h denotes the Gaussian kernel.

Theorem 10 requires the distribution to be induced by a probability distribution A on feature matrices A with A ≤ δ.

However, since Gaussians have support everywhere, the assumption that A ≤ δ for any finite δ > 0 does not hold.

A possible solution would be to use truncated Gaussian kernels, for which A ≤ δ can be ensured.

However, it remains an open question whether there exists a probability distribution A over feature matrices A that induces suitable truncated Gaussian distributions which would allow to compute practical bounds on the generalization error.

We now provide the proof to Proposition 14.

Proof.

Given a sample S ∼ D with |S| = N , its representativeness is defined as (z i + ξ)) P (λi×νi) (ξ)dξ

By assumption, a Kernel Density Estimator on sample S, i.e.,

with kernel K h , approximates P D (z) with approximation error .

Thus, we get that By substituting ζ = z − z i and choosing the (λ i × ν i ) such that P (λi×νi) (z) = K h (z) (which is possible since we assumed the bandwidth matrix to be diagonal), we can further rewrite this as

In addition to the evaluation on the CIFAR10 dataset with LeNet5 network, we also conducted experiments on the MNIST dataset.

For learning with this data, we employed a custom fully connected network with ReLU activations containing 4 hidden layers with 50, 50, 50, and 30 neurons correspondingly.

The output layer has 10 neurons with softmax activation.

The networks were trained till convergence on the training dataset of MNIST, moreover, the configurations that achieved larger than 0.07 training error were filtered out.

All the networks were initialized according to Xavier normal scheme with random seed.

For obtaining different convergence minima the batch size was varied between 1000, 2000, 4000, 8000 with learning rate changed from 0.02 to 1.6 correspondingly to keep the ratio constant.

All the configurations were trained with SGD.

Figure 5 shows the correlation between the layer-wise flatness measure based on the trace of the Hessian for the corresponding layer.

The values for all four hidden layers are calculated (the trace is not normalized) and aligned with values of generalization error (difference between normalized test error and train error).

The observed correlation is strong (with ρ ≥ 0.7) and varies slightly for different layers, nevertheless it is hard to identify the most influential layer for identifying generalization properties.

We also calculated neuron-wise flatness measures described in Sec. B for this network configurations.

In Figure 6 we depicted correlation between ρ l σ and generalization loss for each of the layers, and in Figure 7 -between ρ l and generalization loss.

The observed correlation is again significant, but compared to the previous measure we can see that it might differ considerably depending on the layer.

The network-wise flatness measures can based both on layer-wise and neuron-wise measures as defined in Sec. B. We computed κ max τ , κ Σ τ , ρ max , and ρ Σ and depicted them in Figure 8 .

Interesting to note, that each of the network-wise measures has a larger correlation with generalization loss than the original neuron-wise and layer-wise measures.

@highlight

We introduce a novel measure of flatness at local minima of the loss surface of deep neural networks which is invariant with respect to layer-wise reparameterizations and we connect flatness to feature robustness and generalization.

@highlight

The authors propose a notion of feature robustness which is invariant with respect to rescaling the weight and discuss the notion's relationship to generalization.

@highlight

This paper defines a notion of feature-robustness and combines it with epsilon representativeness of a function to describe a connection between flatness of minima and generalization in deep neural networks.