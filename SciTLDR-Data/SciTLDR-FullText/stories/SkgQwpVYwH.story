It is important to collect credible training samples $(x,y)$ for building data-intensive learning systems (e.g., a deep learning system).

In the literature, there is a line of studies on eliciting distributional information from self-interested agents who hold a relevant information.

Asking people to report complex distribution $p(x)$, though theoretically viable, is challenging in practice.

This is primarily due to the heavy cognitive loads required for human agents to reason and report this high dimensional information.

Consider the example where we are interested in building an image classifier via first collecting a certain category of  high-dimensional image data.

While classical elicitation results apply to eliciting a complex and generative (and continuous) distribution $p(x)$ for this image data, we are interested in eliciting samples $x_i \sim p(x)$ from agents.

This paper introduces a deep learning aided method to incentivize credible sample contributions from selfish and rational agents.

The challenge to do so is to design an incentive-compatible score function to score each reported sample to induce truthful reports, instead of an arbitrary or even adversarial one.

We show that with accurate estimation of a certain $f$-divergence function we are able to achieve approximate incentive compatibility in eliciting truthful samples.

We then present an efficient estimator with theoretical guarantee via studying the variational forms of $f$-divergence function.

Our work complements the literature of information elicitation via introducing the problem of \emph{sample elicitation}.  We also show a connection between this sample elicitation problem and $f$-GAN, and how this connection can help reconstruct an estimator of the distribution based on collected samples.

The availability of a large quantity of credible samples is crucial for building high-fidelity machine learning models.

This is particularly true for deep learning systems that are data-hungry.

Arguably, the most scalable way to collect a large amount of training samples is to crowdsource from a decentralized population of agents who hold relevant sample information.

The most popular example is the build of ImageNet (Deng et al., 2009 ).

The main challenge in eliciting private information is to properly score reported information such that the self-interested agent who holds a private information will be incentivized to report truthfully.

At a first look, this problem of eliciting quality data is readily solvable with the seminal solution for eliciting distributional information, called the strictly proper scoring rule (Brier, 1950; Winkler, 1969; Savage, 1971; Matheson & Winkler, 1976; Jose et al., 2006; Gneiting & Raftery, 2007) : suppose we are interested in eliciting information about a random vector X = (X 1 , ..., X d−1 , Y ) ∈ Ω ⊆ R d , whose probability density function is denoted by p with distribution P. As the mechanism designer, if we have a sample x drawn from the true distribution P, we can apply strictly proper scoring rules to elicit p: the agent who holds p will be scored using S (p, x) .

S is called strictly proper if it holds for any p and q that E x∼P [S(p, x) ] > E x∼P [S(q, x) ].

The above elicitation approach has two main caveats that limited its application:

• When the outcome space |Ω| is large and is even possibly infinite, it is practically impossible for any human agents to report such a distribution with reasonable efforts.

This partially inspired a line of follow-up works on eliciting property of the distributions, which we will discuss later.

In this work we aim to collect credible samples from self-interested agents via studying the problem of sample elicitation.

Instead of asking each agent to report the entire distribution p, we hope to elicit samples drawn from the distribution P truthfully.

We consider the samples x p ∼ P and x q ∼ Q. In analogy to strictly proper scoring rules 1 , we aim to design a score function S s.t.

for any q ̸ = p, where x ′ is a reference answer that can be defined using elicited reports.

Often, this scoring procedure requires reports from multiple peer agents, and x ′ is chosen as a function of the reported samples from all other agents (e.g., the average across all the reported xs, or a randomly selected x).

This setting will relax the requirements of high reporting complexity, and has wide applications in collecting training samples for machine learning tasks.

Indeed our goal resembles similarity to property elicitation (Lambert et al., 2008; Steinwart et al., 2014; Frongillo & Kash, 2015b ), but we emphasize that our aims are different -property elicitation aims to elicit statistical properties of a distribution, while ours focus on eliciting samples drawn from the distributions.

In certain scenarios, when agents do not have the complete knowledge or power to compute these properties, our setting enables elicitation of individual sample points.

Our challenge lies in accurately evaluating reported samples.

We first observe that the f -divergence function between two properly defined distributions of the samples can serve the purpose of incentivizing truthful report of samples.

We proceed with using deep learning techniques to solve the score function design problem via a data-driven approach.

We then propose a variational approach that enables us to estimate the divergence function efficiently using reported samples, via a variational form of the f -divergence function, through a deep neutral network.

These estimation results help us establish an approximate incentive compatibility in eliciting truthful samples.

It is worth to note that our framework also generalizes to the setting where there is no access to ground truth samples, where we can only rely on reported samples.

There we show that our estimation results admit an approximate Bayesian Nash Equilibrium for agents to report truthfully.

Furthermore, in our estimation framework, we use a generative adversarial approach to reconstruct the distribution from the elicited samples.

We want to emphasize that the deep learning based estimators considered above are able to handle complex data.

And with our deep learning solution, we are further able to provide estimates for the divergence functions used for our scoring mechanisms with provable finite sample complexity.

In this paper, we focus on developing theoretical guarantees -other parametric families either can not handle complex data, e.g., it is hard to handle images using kernel methods, or do not have provable guarantees on the sample complexity.

Our contributions are three-folds.

(1) We tackle the problem of eliciting complex distribution via proposing a sample elicitation framework.

Our deep learning aided solution concept makes it practical to solicit complex sample information from human agents.

(2) Our framework covers the case when the mechanism designer has no access to ground truth information, which adds contribution to the peer prediction literature.

(3) On the technical side, we develop estimators via deep learning techniques with strong theoretical guarantees.

This not only helps us establish approximate incentive-compatibility, but also enables the designer to recover the targeted distribution from elicited samples.

Our contribution can therefore be summarized as "eliciting credible training samples by deep learning, for deep learning".

The most relevant literature to our paper is strictly proper scoring rules and property elicitation.

Scoring rules were developed for eliciting truthful prediction (probability) (Brier, 1950; Winkler, 1969; Savage, 1971; Matheson & Winkler, 1976; Jose et al., 2006; Gneiting & Raftery, 2007) .

Characterization results for strictly proper scoring rules are given in McCarthy (1956); Savage (1971); Gneiting & Raftery (2007) .

Property elicitation notices the challenge of eliciting complex distributions (Lambert et al., 2008; Steinwart et al., 2014; Frongillo & Kash, 2015b) .

For instance, Abernethy & Frongillo (2012) characterize the score functions for eliciting linear properties, and Frongillo & Kash (2015a) study the complexity of eliciting properties.

Another line of relevant research is peer prediction, where solutions can help elicit private information when the ground truth verification might be missing (De Alfaro et al., 2016; Gao et al., 2016; Kong et al., 2016; Kong & Schoenebeck, 2018; 2019) .

Our work complements the information elicitation literature via proposing and studying the question of sample elicitation via a variational approach to estimate f -divergence functions.

Our work also extends the line of work on divergence estimation.

The simplest way to estimate divergence starts with the estimation of density function (Wang et al., 2005; Lee & Park, 2006; Wang et al., 2009; Zhang & Grabchak, 2014; Han et al., 2016) .

Another method based on the variational form (Donsker & Varadhan, 1975) of the divergence function comes into play (Broniatowski & Keziou, 2004; 2009; Nguyen et al., 2010; Kanamori et al., 2011; Ruderman et al., 2012; Sugiyama et al., 2012) , where the estimation of divergence is modeled as the estimation of density ratio between two distributions.

The variational form of the divergence function also motivates the well-know Generative Adversarial Network (GAN) (Goodfellow et al., 2014) , which learns the distribution by minimizing the Kullback-Leibler divergence.

Follow-up works include Nowozin et al. Bu et al. (2018) for this line of work.

For the distribution P, we denote by P n the empirical distribution given a set of samples

following P, i.e., P n = 1/n · n i=1 δ xi , where δ xi is the Dirac measure at x i .

We denote by

, where µ is the Lebesgue measure.

Also, we denote by ∥f ∥ ∞ = sup x∈X |f (x)| the L ∞ norm of f (·).

For any real-valued functions g(·) and h(·) defined on some unbounded subset of the real positive numbers, such that h(α) is strictly positive for all large enough values of α, we write

for some positive absolute constant c and any α > α 0 , where α 0 is a real number.

We denote by [n] the set {1, 2, . . .

, n}.

We formulate the question of sample elicitation.

We consider two scenarios.

We start with an easier case where we, as the mechanism designer, have access to a certain number of group truth samples.

This is a setting that resembles similarity to the proper scoring rule setting.

Then we move to the harder case where the inputs to our mechanism can only be elicited samples from agents.

Multi-sample elicitation with ground truth samples.

Suppose that the agent holds n samples, with each of them independently drawn from P, i.e., x i ∼ P 2 for i ∈ [n].

The agent can report each sample arbitrarily, which is denoted as r i (x i ) : Ω → Ω. There are n data {x * i } i∈ [n] independently drawn from the ground truth distribution Q 3 .

We are interested in designing a score function S(·) that takes inputs of each r i (·) and [n] ) such that if the agent believes that x * is drawn from the same distribution x * ∼ P, then for any {r j (·)} j∈ [n] , it holds with probability at least 1 − δ that

We name the above as (δ, ϵ)-properness (per sample) for sample elicitation.

When δ = ϵ = 0, it is reduced to the one that is similar to the properness definition in scoring rule literature (Gneiting & Raftery, 2007) .

We also shorthand r i = r i (x i ) when there is no confusion.

Agent believes that her samples are generated from the same distribution as of the ground truth samples, i.e., P and Q are same distributions.

Sample elicitation with peer samples.

Suppose there are n agents each holding a sample x i ∼ P i , where the distributions {P i } i∈ [n] are not necessarily the same -this models the fact that agents can have subjective biases or local observation biases.

This is a more standard peer prediction setting.

We denote by their joint distribution as P = P 1 × P 2 × ....

× P n .

Similar to the previous setting, each agent can report her sample arbitrarily, which is denoted as r i (x i ) : Ω → Ω for any i ∈ [n].

We are interested in designing and characterizing a score function S(·) that takes inputs of each r i (·) and {r j (x j )} j̸ =i : S(r i (x i ), {r j (x j )} j̸ =i ) such that for any {r j (·)} j∈ [n] , it holds with probability at least 1 − δ that

We name the above as (δ, ϵ)-Bayesian Nash Equilibrium (BNE) in truthful elicitation.

We only require that agents are all aware of above information structure as common knowledge, but they do not need to form beliefs about details of other agents' sample distributions.

Each agent's sample is private to herself.

It is well known that maximizing the expected proper scores is equivalent to minimizing a corresponding Bregman divergence (Gneiting & Raftery, 2007) .

More generically, we take the perspective that divergence functions have great potentials to serve as score functions for eliciting samples.

We define the f -divergence between two distributions P and Q with probability density function p and q, respectively, as

Here f (·) is a function satisfying certain regularity conditions, which will be specified later.

Solving our elicitation problem involves evaluating the D f (q∥p) successively based on the distributions P and Q, without knowing the probability density functions p and q. Therefore, we have to resolve to a form of D f (q∥p) which does not involve the analytic forms of p and q, but instead sample forms.

Following from Fenchel's convex duality, it holds that 3.1 ERROR BOUND AND ASSUMPTIONS Suppose we have the following error bound for estimating D f (q∥p): for any probability density functions p and q, it holds with probability at least

2) where δ(n) and ϵ(n) will be specified later in Section 4.

To obtain such an error bound, we need the following assumptions.

Assumption 3.1 (Bounded Density Ratio).

The density ratio θ * (x; p, q) = q(x)/p(x) is bounded such that 0 < θ 0 ≤ θ * ≤ θ 1 holds for positive absolute constants θ 0 and θ 1 .

The above assumption is standard in related literature (Nguyen et al., 2010; Suzuki et al., 2008) , which requires that the probability density functions p and q lie on a same support.

For simplicity of presentation, we assume that this support is Ω ⊂ R d .

We define the β-Hölder function class on Ω as follows.

where

We assume that the function t * (·; p, q) is β-Hölder, which guarantees the smoothness of t * (·; p, q).

In addition, we assume that the following regularity conditions hold for the function f (·) in the definition of f -divergence in (2.1).

We highlight that we only require that the conditions in Assumption 3.4 hold on the interval [θ 0 , θ 1 ], where the absolute constants θ 0 and θ 1 are specified in Assumption 3.1.

Thus, Assumption 3.4 is mild and it holds for many commonly used functions in the definition of f -divergence.

For example, in Kullback-Leibler (KL) divergence, we take f (u) = − log u, which satisfies Assumption 3.4; in Jenson-Shannon divergence, we take f (u) = u log u − (u + 1) log(u + 1), which also satisfies Assumption 3.4.

We will show that under Assumptions 3.1, 3.3, and 3.4, the bound (3.2) holds.

See Theorem 4.2 in Section 4 for details.

In this section, we focus on multi-sample elicitation with ground truth samples.

Under this setting, as a reminder, the agent will report multiple samples.

After the agent reported her samples, the mechanism designer obtains a set of ground truth samples {x * i } i∈ [n] ∼ Q to serve the purpose of evaluation.

This falls into the standard strictly proper scoring rule setting.

Our mechanism is presented in Algorithm 1.

Algorithm 1 consists of two steps: step 1 is to compute the function t(·; p, q), which enables us, in step 2, to pay agent using a linear-transformed estimated divergence between the reported samples and the true samples.

We have the following result.

Theorem 3.5.

The f -scoring mechanism in Algorithm 1 achieves (2δ(n), 2bϵ(n))-properness.

Algorithm 1 f -scoring mechanism for multiple-sample elicitation with ground truth 1.

Compute

2.

For i ∈ [n], pay reported sample r i using

for some constants a, b > 0.

The proof is mainly based on the error bound in estimating f -divergence and its non-negativity.

Not surprisingly, if the agent believes her samples are generated from the same distribution as the ground truth sample, and that our estimator can well characterize the difference between the two set of samples, she will be incentivized to report truthfully to minimize the difference.

We defer the proof to Section B.1.

The above mechanism in Algorithm 1, while intuitive, has the following two caveats:

• The agent needs to report multiple samples (multi-task/sample elicitation);

• Multiple samples from the ground truth distribution are needed.

To deal with such caveats, we consider the single point elicitation in an elicitation without verification setting.

Suppose there are 2n agents each holding a sample x i ∼ P i 4 .

We randomly partition the agents into two groups, and denote the joint distributions for each group's samples as P and Q with probability density functions p and q for each of the two groups.

Correspondingly, there are a set of n agents for each group, respectively, who are required to report their single data point according to two distributions P and Q, i.e., each of them holds {x

As an interesting note, this is also similar to the setup of a Generative Adversarial Network (GAN), where one distribution corresponds to a generative distribution x | y = 1, and another x | y = 0.

This is a connection that we will further explore in Section 5 to recover distributions from elicited samples.

We denote by the joint distribution of p and q as p ⊕ q (distribution as P ⊕ Q), and the product of the marginal distribution as p × q (distribution as P × Q).

We consider the divergence between the two distributions:

Motivated by the connection between mutual information and KL divergence, we define generalized f -mutual information in the follows, which characterizes the generic connection between a generalized f -mutual information and f -divergence.

Definition 3.6 (Kong & Schoenebeck (2019)).

The generalized f -mutual information between p and q is defined as

Further it is shown in Kong & Schoenebeck (2018; 2019) that the data processing inequality for mutual information holds for I f (p; q) when f is strictly convex.

We define the following estimators,

where P n and Q n are empirical distributions of the reported samples.

We denote x ∼ P n ⊕ Q n | r i as the conditional distribution when the first variable is fixed with realization r i .

Our mechanism is presented in Algorithm 2.

.

2.

Pay each reported sample r i using:

for some constants a, b > 0.

Similar to Algorithm 1, the main step in Algorithm 2 is to estimate the f -divergence between P n × Q n and P n ⊕ Q n using reported samples.

Then we pay agents using a linear-transformed form of it.

We have the following result.

Theorem 3.7.

The f -scoring mechanism in Algorithm 2 achieves (2δ(n), 2bϵ(n))-BNE.

The theorem is proved by error bound in estimating f -divergence, a max argument, and the data processing inequality for f -mutual information.

We defer the proof in Section B.2.

The job left for us is to establish the error bound in estimating the f -divergence to obtain ϵ(n) and δ(n).

Roughly speaking, if we solve the optimization problem (3.3) via deep neural networks with proper structure, it holds that

where c is a positive absolute constant.

We state and prove this result formally in Section 4.

Remark 3.8.

(1) When the number of samples grows, it holds that δ(n) and ϵ(n) decrease to 0 at least polynomially fast, and our guaranteed approximate incentive-compatibility approaches a strict one.

(2) Our method or framework handles arbitrary complex information, where the data can be sampled from high dimensional continuous space.

(3) The score function requires no prior knowledge.

Instead, we design estimation methods purely based on reported sample data.

(4) Our framework also covers the case where the mechanism designer has no access to the ground truth, which adds contribution to the peer prediction literature.

So far peer prediction results focused on eliciting simple categorical information.

Besides handling complex information structure, our approach can also be viewed as a data-driven mechanism for peer prediction problems.

In this section, we introduce an estimator of f -divergence and establish the statistical rate of convergence, which characterizes ϵ(n) and δ(n).

For the simplicity of presentation, in the sequel, we estimate the f -divergence D f (q∥p) between distributions P and Q with probability density functions p and q, respectively.

The rate of convergence of estimating f -divergence can be easily extended to that of mutual information.

By Section 3, estimating f -divergence between P and Q is equivalent to solving the following optimization problem,

In what follows, we propose an estimator of D f (q∥p).

By Assumption 3.3, it suffices to solve (4.1) on the function class C

To this end, we approximate solution to (4.1) by the family of deep neural networks.

We now define the family of deep neural networks as follows.

, where k 0 = d and k L+1 = 1, the family of deep neural networks is defined as

Here we write σ v (x) as σ(x − v) for notational convenience, where σ(·) is the ReLU activation function.

To avoid overfitting, the sparsity of the deep neural networks is a typical assumption in deep learning literature.

In practice, such a sparsity property is achieved through certain techniques, e.g., dropout (Srivastava et al., 2014) , or certain network architecture, e.g., convolutional neural network (Krizhevsky et al., 2012) .

We now define the family of sparse networks as follows,

where s is the sparsity.

In contrast, another approach to avoid overfitting is to control the norm of parameters.

See Section A.2 for details.

We now propose the following estimators

3)

The following theorem characterizes the statistical rate of convergence of the estimators defined in (4.3). (2β+d) .

Under Assumptions 3.1, 3.3, and 3.4, it holds with probability at

We defer the proof of the theorem in Section B.3.

By Theorem 4.2, the estimators in (4.3) achieve the optimal nonparametric rate of convergence (Stone, 1982) up to a logarithmic term.

By (3.2) and Theorem 4.2, we have

where c is a positive absolute constant.

After sample elicitation, a natural question to ask is how to learn a representative probability density function from the samples.

Denote the probability density function from elicited samples as p. Then, learning the probability density function p is to solve for

where Q is the probability density function space.

To see the connection between (5.1) and the formulation of f -GAN (Nowozin et al., 2016) , by combining (2.2) and (5.1), we have

which is the formulation of f -GAN.

Here the probability density function q(·) is the generator, while the function t(·) is the discriminator.

By the non-negativity of f -divergence, q * = p solves (5.1).

We now propose the following estimator

where D f (q∥p) is given in (4.3).

We define covering number as follows.

Definition 5.1 (Covering Number).

Let (V, ∥ · ∥ L2 ) be a normed space, and Q ⊂ V .

We say that

We impose the following assumption on the covering number of the probability density function space Q.

Recall that q * = p is the unique minimizer of the problem (5.1).

Therefore, the f -divergence D f ( q∥p) characterizes the deviation of q from p * .

The following theorem characterizes the error bound of estimating q * by q.

Theorem 5.3.

Under the same assumptions in Theorem 4.2 and Assumption 5.2, for sufficiently large sample size n, it holds with probability at least 1 − 1/n that

We defer the proof of the theorem in Section B.4.

In Theorem 5.3, the first term on the RHS of (5.3) characterizes the generalization error of the estimator in (5.2), while the second term characterizes the approximation error.

If the approximation error in (5.3) vanishes, then the estimator q converges to the true density function q * = p at the optimal nonparametric rate of convergence (Stone, 1982) up to a logarithmic term.

In this work, we introduce the problem of sample elicitation as an alternative to eliciting complicated distribution.

Our elicitation mechanism leverages the variational form of f -divergence functions to achieve accurate estimation of the divergences using samples.

We provide theoretical guarantee for both our estimators and the achieved incentive compatibility.

It reminds an interesting problem to find out more "organic" mechanisms for sample elicitation that requires (i) less elicited samples; and (ii) induced strict truthfulness instead of approximated ones.

with probability at least 1 − ε · exp(−γ 2 ).

We defer the proof of to Section B.5.

As a by-product, note that t

, based on the error bound established in Theorem A.1, we obtain the following result.

Corollary A.2.

Given 0 < ε < 1, for the sample size n ≳ [γ + γ −1 log(1/ε)] 2 , under Assumptions 3.1, 3.3, and 3.4, it holds with probability at least 1 − ε · exp(−γ 2 ) that

.

′ and f † has Lipschitz continuous gradient with parameter 1/µ 0 from Assumption 3.4 and Lemma D.6, we obtain the result from Theorem A.1.

In this section, we consider using norm of the parameters (specifically speaking, the norm of W j and v j in (4.1)) to control the error bound, which is an alternative of the network model shown in (4.2).

We consider the family of L-layer neural networks with bounded spectral norm for weight matrices

, where k 0 = d and k L+1 = 1, and

, which is denoted as

.

We write the following optimization problem,

Based on this formulation, we derive the error bound on the estimated f -divergence in the following theorem.

We only consider the generalization error bound in this setting.

Therefore, we assume that the ground truth t * (x; p, q) = f ′ (q(x)/p(x)) locates within Φ norm .

Before we state the theorem, we first define two parameters for the family of neural networks Φ norm (L, k, A, B) as follows

We proceed to state the theorem.

Theorem A.3.

We assume that t * (x; p, q) ∈ Φ norm .

Then for any 0 < ε < 1, with probability at least 1 − ε, it holds that

log(1/ε).

Here γ 1 and γ 2 are defined in (A.3).

We defer the proof to Section B.6.

The next theorem uses the results in Theorem A.3.

Recall that in Section §A.2, we assume that the minimizer t * to the population version problem (4.1) lies within the norm-controlled family of neural networks Φ norm (L, k, A, B) .

Theorem A.4.

Recall that we defined the parameter γ 1 and γ 2 of the family of neural networks Φ norm (L, k, A, B) in (A.3) , the estimated distribution q in (5.2), and the ground truth q * = p.

We denote the the covering number of the probability distribution function class Q as N 2 (δ, Q), then for any 0 < ε < 1, with probability at least 1 − ε, we have

where

We defer the proof to Section B.7.

B.1 PROOF OF THEOREM 3.5

If the player truthfully reports, she will receive the following expected payment per sample i: with probability at least 1 − δ(n),

Similarly, any misreporting according to a distribution p with distribution P will lead to the following derivation with probability at least 1 − δ

Combining above, and using union bound, leads to (2δ(n), 2bϵ(n))-properness.

Consider an arbitrary agent i. Suppose every other agent truthfully reports.

Reporting a r i ∼ P ̸ = P (denoting its distribution as p) leads to the following score

with probability at least 1 − δ(n) (the other δ(n) probability with maximum scoreS).

Now we prove that truthful reporting leads at least

of the divergence term:

with probability at least 1 − δ(n) (the other δ(n) probability with score at least 0).

Therefore the expected divergence terms differ at most by 2ϵ(n) with probability at least 1 − 2δ(n) (via union bound).

The above combines to establish a (2δ(n), 2bϵ(n))-BNE.

Step 1.

We proceed to bound ∥t * − t∥ L2(P) .

We first proceed to find some t ∈ Φ M (L, k, s) .

Note that the ground truth t * lies on a finite support

, and m ′ = log n, we then utilize Theorem D.5 to construct some

To this end, we know that t ∈ Φ M (L, k, s), with parameters L, k, and s given in the statement of Theorem 4.2.

We fix this t and invoke Theorem A.1, then with probability at least 1 − ε · exp(−γ 2 ), we have

and L, s given in the statement of Theorem 4.2, it holds that γ = O(N 1/2 log 5/2 n).

Moreover, by the choice N = n d/(2β+d) , combining (B.1) and taking ε = 1/n, we know that

2) with probability at least 1 − exp{−n d/(2β+d) log 5 n}.

Step 2.

We denote by

.

Then from Assumption 3.4 and Lemma D.6, we know that L(·) is strongly convex with a constant coefficient.

Note that by triangular inequality, we have

We proceed to bound A 1 and A 2 .

Bound on A 1 : Recall that L(·) is strongly convex.

Consequently, we have

with probability at least 1 − exp{−n d/(2β+d) log 5 n}, where the last inequality comes from (B.2).

Bound on A 2 : Note that both the functions t * (·) and f † (t * (·)) are bounded, then by Hoeffding's inequality, we obtain that

Therefore, by combining the above two bounds, we obtain that

log 7/2 n with probability at least 1−exp{−n (d−2β)/(2β+d) log 14 n}.

This concludes the proof of the theorem.

We first need to bound the max deviation of the estimated

The following lemma provides such a bound.

Lemma B.1.

Under the assumptions stated in Theorem 5.3, for any fixed density p, if the sample size n is sufficiently large, it holds that

2β+d · log 7 n with probability at least 1 − 1/n.

We defer the proof to Section C.1.

Now we turn to the proof of the theorem.

We denote by q ′ = argmin q∈Q D f ( q∥p), then with probability at least 1 − 1/n, we have

Here in the second line we use the optimality of q among all q ∈ Q to the problem (5.2), while the last inequality uses Lemma B.1 and Theorem 4.2.

Moreover, note that combining (B.3) , it holds that with probability at least 1 − 1/n,

This concludes the proof of the theorem.

B.5 PROOF OF THEOREM A.1

For any real-valued function ϱ, we write

, and E Qn (ϱ) = E x∼Qn [ϱ(x)] for notational convenience.

For any t ∈ Φ M (L, k, s), we establish the following lemma.

(B.4) Furthermore, to bound the RHS of the above inequality, we establish the following lemma.

Lemma B.3.

We assume that the function ψ : R → R is Lipschitz continuous and bounded such that |ψ(x)| ≤ M 0 for any |x| ≤ M .

Then under the assumptions stated in Theorem A.1, for any fixed t(x) ∈ Φ M , n ≳ [γ + γ −1 log(1/ε)]

2 and 0 < ε < 1, we have the follows

where

, and for any real numbers c 1 and c 2 , we denote by c 1 ∨ c 2 = max{c 1 , c 2 }.

Here γ takes the form γ = s 1/2 log(V 2 L),

We defer the proof to Section C.3.

Note that the results in Lemma B.3 also apply to the distribution Q, and by using the fact that the true density ratio θ * (x; p, q) = q(x)/p(x) is bounded below and above, we know that L 2 (Q) is indeed equivalent to L 2 (P).

We thus focus on L 2 (P) here.

By (B.4), Lemma B.3, and the Lipschitz property of f † according to Lemma D.6, with probability at least 1 − ε · exp(−γ 2 ), we have the following bound

where we recall that the notation γ = s 1/2 log(V 2 L) is a parameter related with the family of neural networks Φ M .

We proceed to analyze the dominant part on the RHS of (B.5).

* ∥ L2(P) dominates, then with probability at least 1−ε·exp(−γ 2 )

dominates, then with probability at least 1−ε·exp(−γ 2 )

Therefore, by combining the above three cases, we have

Further the triangular inequality gives us

with probability at least 1 − ε · exp(−γ 2 ).

Note that the above error bound holds for any t ∈ Φ M (L, k, s), especially for the choice t such that it minimizes ∥ t − t * ∥ L2 (P) .

Therefore, we have

with probability at least 1 − ε · exp(−γ 2 ).

This concludes the proof of the theorem.

We follow the proof in Li et al. (2018) .

We denote by the loss function in

, where x I follows the distribution P and x II follows Q. To prove the theorem, we first link the generalization error in our theorem to the empirical Rademacher complexity (ERC).

Given the data {x i } n i=1 , the ERC related with the class L(Φ norm ) is defined as

where ε i 's are i.i.d.

Rademacher random variables, i.e., P(ε i = 1) = P(ε i = −1) = 1/2.

Here the expectation E ε (·) is taken over the Rademacher random variables {ε i } i∈ [n] .

We introduce the following Lemma B.4 (Mohri et al., 2018) , which links the ERC to the generalization error bound.

Lemma B.4.

Assume that sup φ∈Φnorm |L(φ)| ≤ M 1 , then for any ε > 0, with probability at least 1 − ε, we have

where the expectation E x {·} is taken over x I ∼ P and x II ∼ Q.

Equipped with the above lemma, we only need to bound the ERC defined in (B.6).

Lemma B.5.

Let L be a Lipschitz continuous loss function and Φ norm be the family of networks defined in (A.1).

We assume that the input x ∈ R d is bounded such that ∥x∥ 2 ≤ B. Then it holds that

where γ 1 and γ 2 are given in (A.3).

We defer the proof to Section C.4.

Now we proceed to prove the theorem.

Recall that we assume that t * ∈ Φ norm .

For notational convenience, we denote by

where the second inequality follows from the fact that t is the minimizer of H(·).

On the other hand, if

where the second inequality follows that fact that t * is the minimizer of H(·).

Therefore, by (B.7), (B.8), and the fact that L(φ) ≲ L+1 j=1 B j for any φ ∈ Φ norm , we deduce that

log(1/ε) (B.9) with probability at least 1 − ε.

Here the second inequality follows from Lemma B.4.

By plugging the result from Lemma B.5 into (B.9), we deduce that with probability at least 1 − ε, it holds that

This concludes the proof of the theorem.

We first need to bound the max deviation of the estimated f -divergence D f (q∥p) among all q ∈ Q. We utilize the following lemma to provide such a bound.

Lemma B.6.

Assume that the distribution q is in the set Q, and we denote its L 2 covering number as N 2 (δ, Q).

Then for any target distribution p, we have

with probability at least 1 − ε.

Here b 2 (n, γ 1 , γ 2 ) = γ 1 n −1/2 log(γ 2 n) and c is a positive absolute constant.

We defer the proof to Section C.5.

Now we turn to the proof of the theorem.

We denote by q ′ = argmin q∈Q D f ( q∥p).

Then with probability at least 1 − ε, we have

where we use the optimality of q among all q ∈ Q to the problem (5.2) in the second inequality, and we uses Lemma B.6 and Theorem 4.2 in the last line.

Moreover, note that

This concludes the proof of the theorem.

C LEMMAS AND PROOFS C.1 PROOF OF LEMMA B.1

Recall that the covering number of Q is N 2 (δ, Q), we thus assume that there exists q 1 , . . .

, q N2(δ,Q) ∈ Q such that for any q ∈ Q, there exists some q k , where (2β+d) and union bound, we have

where the last line comes from Theorem 4.2.

Combining Assumption 5.2, when n is sufficiently large, it holds that

which concludes the proof of the lemma.

On the other hand, we denote that S = min{s > 1 :

.

For notational convenience, we denote the set

Then by the peeling device, we have the following

where c is a positive absolute constant, and for notational convenience we denote by

Here in the second line, we use the fact that for any

; in the forth line, we use the argument that since A s ⊆ Ψ M (2 −s+2 M 0 ), the probability of supremum taken over

is larger than the one over A s ; in the last line we invoke Theorem D.3.

Consequently, this gives us

(C.8)

Combining (C.6) and (C.8), we finish the proof of the lemma.

C.4 PROOF OF LEMMA B.5

The proof of the theorem utilizes following two lemmas.

The first lemma characterizes the Lipschitz property of φ(x; W, v) in the input x. Lemma C.1.

Given W and v, then for any φ(·; W, v) ∈ Φ norm and x 1 , x 2 ∈ R d , we have

We defer the proof to Section C.6.

The following lemma characterizes the Lipschitz property of φ(x; W, v) in the network parameter pair (W, v).

Lemma C.2.

Given any bounded x ∈ R d such that ∥x∥ 2 ≤ B, then for any weights

, and functions

We defer the proof to Section C.7.

We now turn to the proof of Lemma B.5.

Note that by Lemma C.2, we know that φ(

and the Lipschitz constant L w satisfies

In addition, we know that the covering number of

By the above facts, we deduce that the covering number of

for some positive absolute constant c 1 .

Then by Dudley entropy integral bound on the ERC, we know that

for some positive absolute constant c 2 .

Therefore, by calculations, we derive from (C.12) that

then we conclude the proof of the lemma by plugging in (C.9), (C.10), (C.11), and (C.13), and using the definition of γ 1 and γ 2 in (A.3).

C.5 PROOF OF LEMMA B.6

Remember that the covering number of Q is N 2 (δ, Q), we assume that there exists q 1 , . . .

, q N2(δ,Q) ∈ Q such that for any q ∈ Q, there exists some q k , where

where the second line comes from union bound, and the last line comes from Theorem A.3.

By this, we conclude the proof of the lemma.

C.6 PROOF OF LEMMA C.1

The proof follows by applying the Lipschitz property and bounded spectral norm of W j recursively:

Here in the third line we uses the fact that ∥W j ∥ 2 ≤ B j and the 1-Lipschitz property of σ vj (·), and in the last line we recursively apply the same argument as in the above lines.

This concludes the proof of the lemma.

C.7 PROOF OF LEMMA C.2

Recall that φ(x; W, v) takes the form

For notational convenience, we denote by φ

(C.14)

Moreover, note that for any ℓ ∈ [L], we have the following bound on ∥φ where the first inequality comes from the triangle inequality, and the second inequality comes from the bounded spectral norm of W i j , while the last inequality simply applies the previous arguments recursively.

Therefore, combining (C.14), we have

Similarly, by triangular inequality, we have

where the second inequality uses the bounded spectral norm of W L and 1-Lipschitz property of σ v L (·).

For notational convenience, we further denote y = φ

where the inequality comes from the 1-Lipschitz property of σ(·).

Moreover, combining (C.15), it holds that

By (C.17) and (C.18), we have

Here in the second inequality we recursively apply the previous arguments.

Further combining (C.16), we obtain that

where we use Cauchy-Schwarz inequality in the last line.

This concludes the proof of the lemma.

Lemma D.1.

The following statements for entropy hold.

1. Suppose that sup g∈G ∥g∥ ∞ ≤ M , then

for any δ > 0.

2.

For 1 ≤ q < ∞, and Q a distribution, we have

for any δ > 0.

Here H ∞ is the entropy induced by infinity norm.

3.

Based on the above two statements, suppose that sup g∈G ∥g∥ ∞ ≤ M , we have

by taking p = 2.

@highlight

This paper proposes a deep learning aided method to elicit credible samples from self-interested agents. 

@highlight

The authors propose a sample elicitation framework for the problem of eliciting credible samples from agents for complex distributions, suggest that deep neural frameworks can be applied in this framework, and connect sample elicitation and f-GAN.

@highlight

This paper studies the sample elicitation problem, proposing a deep learning approach that relies on the dual expression of the f-divergence which writes as a maximum over a set of functions t.