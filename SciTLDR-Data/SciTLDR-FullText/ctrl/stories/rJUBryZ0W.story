In representational lifelong learning an agent aims to continually learn to solve novel tasks while updating its representation in light of previous tasks.

Under the assumption that future tasks are related to previous tasks, representations should be learned in such a way that they capture the common structure across learned tasks, while allowing the learner sufficient flexibility to adapt to novel aspects of a new task.

We develop a framework for lifelong learning in deep neural networks that is based on generalization bounds, developed within the PAC-Bayes framework.

Learning takes place through the construction of a distribution over networks based on the tasks seen so far, and its utilization for learning a new task.

Thus, prior knowledge is incorporated through setting a history-dependent prior for novel tasks.

We develop a gradient-based algorithm implementing these ideas, based on minimizing an objective function motivated by generalization bounds, and demonstrate its effectiveness through numerical examples.

Learning from examples is the process of inferring a general rule from a finite set of examples.

It is well known in statistics (e.g., BID7 ) that learning cannot take place without prior assumptions.

This idea has led in Machine Learning to the notion of inductive bias BID23 .

Recent work in deep neural networks has achieved significant success in using prior knowledge in the implementation of structural constraints, e.g. the use of convolutions and weight sharing as building blocks, capturing the translational invariance of image classification.

However, in general the relevant prior information for a given task is not always clear, and there is a need for building prior knowledge through learning from previous interactions with the world.

Learning from previous experience can take several forms: Continual learning -a single model is trained to solve a task which changes over time (and hopefully not 'forget' the knowledge from previous times, (e.g., ).

Multi-task learning -the goal is to learn how to solve several observed tasks, while exploiting their shared structure.

Domain adaptation -the goal is to solve a 'target' learning task using a single 'source' learning task (both are observed, but usually the target has mainly unlabeled data).

Lifelong Learning / Meta-Learning / Learning-to-Learnthe goal is to extract knowledge from several observed tasks to be used for future learning on new (not yet observed) learning tasks.

In contrast to multi-task learning, the performance is evaluated on the new tasks.

We work within the framework of lifelong learning, where an agent learns through interacting with the world, transferring the knowledge acquired along its path to any new task it encounters.

This notion has been formulated by BID3 in a clear and simple context of 'task-environment'.

In analogy to the standard single-task learning in which data is sampled from an unknown distribution, Baxter suggested to model a lifelong learning setting as if tasks are sampled from an unknown task distribution (environment), so that knowledge acquired from previous tasks can be used in order to improve performance on a novel task.

Baxter's work not only provided an interesting and mathematically precise perspective for lifelong learning, but also provided generalization bounds demonstrating the potential improvement in performance due to prior knowledge.

Baxter's seminal work, has led to a large number of extensions and developments.

In this contribution we work within the framework formulated by BID3 , and, following the setup in BID25 , provide generalization error bounds within the PAC-Bayes framework.

These bounds are then used to develop a practical learning algorithm that is applied to neural networks, demonstrating the utility of the approach.

The main contributions of this work are the following.

(i) An improved and tighter bound in the theoretical framework of BID25 which can utilize different single-task PAC-Bayesian bounds.(ii) Developing a learning algorithm within this general framework and its implementation using probabilistic feedforward neural networks.

This yields transfer of knowledge between tasks through constraining the prior distribution on a learning network. (iii) Empirical demonstration of the performance enhancement compared to naive approaches and recent methods in this field.

As noted above, BID3 provided a basic mathematical formulation and initial results for lifelong learning.

While there have been many developments in this field since then (e.g., BID1 ; BID9 BID10 ; BID27 ), most of them were not based on generalization error bounds which is the focus of the present work.

An elegant extension of generalization error bounds to lifelong learning was provided by BID25 , mentioned above (more recently extended in BID26 ).

Their work, however, did not provide a practical algorithm applicable to deep neural networks.

More recently, Dziugaite & Roy (2017) developed a single-task algorithm based on PAC-Bayes bounds that was demonstrated to yield good performance in simple classification tasks.

Other recent theoretical approaches to lifelong or multitask learning (e.g. BID0 ; BID20 ) provide increasingly general bounds but have not led directly to practical learning algorithms.

In the standard setting for supervised learning a set of (usually) independent pairs of input/output DISPLAYFORM0 are given, each sample drawn from an unknown probability distribution D, namely (x i , y i ) ??? D. We will use the notation S ??? D m to denote the distribution over the full sample.

The usual learning goal is, based on S to find a function h ??? H, where H is the so-called hypothesis space, that minimizes the expected loss function E (h, z), where z = (x, y) 1 and (h, z) is a loss function bounded in [0, 1] .

As the distribution D is unknown, learning consists of selecting an appropriate h based on the sample S. In classification H is a space of classifiers mapping the input space to a finite set of classes.

As noted in the Introduction, an inductive bias is required for effective learning.

While in the standard approach to learning, described in the previous paragraph, one usually selects a single classifier (e.g., the one minimizing the empirical error), the PAC-Bayes framework, first formulated by BID22 , considers the construction of a complete probability distribution over H, and the selection of a single hypothesis h ??? H based on this distribution.

Since this distribution depends on the data it is referred to as a posterior distribution and will be denoted by Q. We note that while the term 'posterior' has a Bayesian connotation, the framework is not necessarily Bayesian, and the posterior does not need to be related to the prior through the likelihood function as in standard Bayesian analysis.

The PAC-Bayes framework has been widely studied in recent years, and has given rise to significant flexibility in learning, and, more importantly, to some of the best generalization bounds available BID2 BID21 ; Lever et al. (2013) .

The framework has been recently extended to the lifelong learning setting by BID25 , and will be extended and applied to neural networks in the present contribution.

Following the notation introduced above we define the generalization error and the empirical error used in the standard learning setting, DISPLAYFORM0 Since the distribution D is unknown, er (h, D) cannot be directly computed.1 Note that the framework is not limited to supervised learning and can also handle unsupervised learning.

In the PAC-Bayesian setting the learner outputs a distribution over the entire hypothesis space H, i.e, the goal is to provide a posterior distribution Q ??? M, where M denotes the set of distributions over H. The expected (over H) generalization error and empirical error are then given in this setting by averaging (1) over the posterior distribution, DISPLAYFORM0 This average describes a Gibbs prediction procedure -first drawing a hypothesis h from Q then applying it on the sample z. Similarly to (1), the generalization error er (Q, D) cannot be directly computed since D is unknown.

In this section we introduce a PAC-Bayesian bound for the single-task setting.

The bound will also serve us for the lifelong-learning setting in the next sections.

PAC-Bayesian bounds are based on specifying some reference distribution P ??? M. P is called the 'prior' since it must not depend on the observed data S. The distribution over hypotheses Q which is provided as an output from the learning process is called the posterior (since it is allowed to depended on S).

The classical PAC-Bayes theorem for single-task learning was formulated by BID22 .

Theorem 1 (McAllester's single-task bound).

Let P ??? M be some prior distribution over H. Then for any ?? ??? (0, 1], DISPLAYFORM0 where DISPLAYFORM1 The bound (3) can be interpreted as stating that with high probability the expected error er (Q, D) is upper bounded by the empirical error plus a complexity term.

Since, with high probability, the bound holds for all Q ??? M (uniformly), we can choose Q after observing the data S. By choosing Q that minimizes the bound we will get a learning algorithm with generalization guarantees.

Note that PAC-Bayesian bounds express a trade-off between fitting the data (empirical error) and a complexity/regularization term (distance from prior) which encourages selecting a 'simple' hypothesis, namely one similar to the prior.

The contribution of the prior-dependent regularization term to the objective is more significant for a smaller data set.

For asymptotically large sample size m, the complexity term converges to zero.

The specific choice of P affects the bound's tightness and so should express prior knowledge about the problem.

Generally, we want the prior to be close to posteriors which can achieve low training error.

For example, we may want to use a prior that prefers simpler hypotheses (Occam's razor).In general, the bound might not be tight, and can even be vacuous, i.e., greater than the maximal value of the loss.

However, BID8 recently showed that a PAC-Bayesian bound can achieve non-vacuous values with deep-networks and real data sets.

In this work our focus is on deriving an algorithm for lifelong-learning, rather than on actual calculation of the bound.

We expect that even if the numerical value of the bound is vacuous, it still captures the behavior of the generalization error and so minimizing the bound is a good learning strategy.

In our experiments we will plug in positive and unbounded loss functions in the bound, in contrast to the assumption on a bounded loss.

Theoretically, we can still claim that we are bounding a variation of the loss clipped to [0, 1] .

Furthermore, empirically the loss function are almost always smaller than one.

In this section we introduce the lifelong-learning setting.

In this setting a lifelong-learning agent observes several 'training' tasks from the same task environment.

The lifelong-learner must extract some common knowledge from these tasks, which will be used for learning new tasks from the same environment.

In the literature this setting is often called learning-to-learn, meta-learning or lifelonglearning.

We will formulate the problem and provide a generalization bound which will later lead to a practical algorithm.

Our work extends BID25 and establishes a more general bound.

Furthermore, we will demonstrate how to apply this result practically in non-linear deep models using stochastic learning.

The lifelong learning problem formulation follows BID25 .

We assume all tasks share the sample space Z, hypothesis space H and loss function : DISPLAYFORM0 The learning tasks differ in the unknown sample distribution D t associated with each task t. The lifelong-learning agent observes the training sets S 1 , ..., S n corresponding to n different tasks.

The number of samples in task i is denoted by m i .

Each observed dataset S i is assumed to be generated from an unknown sample distribution S i ??? D mi i .

As in Baxter FORMULA2 , we assume that the sample distributions D i are generated i.i.d.

from an unknown tasks distribution ?? .The goal of the lifelong-learner is to extract some knowledge from the observed tasks that will be used as prior knowledge for learning new (yet unobserved) tasks from ?? .

The prior knowledge comes in the form of a distribution over hypotheses, P ??? M. When learning a new task, the learner uses the observed task's data S and the prior P to output a posterior distribution Q(S, P ) over H. We assume that all tasks are learned via the same learning process.

Namely, for a given S and P there is a specific output Q(S, P ).

Hence Q() is a function: DISPLAYFORM1 The quality of a prior P is measured by the expected loss when using it to learn new tasks, as defined by, DISPLAYFORM2 Since we want to prove a PAC-Bayes style bound for lifelong-learning, we assume that the lifelonglearner does not select a single prior P , but instead infers a distribution Q over all prior distributions in M. 4 .

Since Q is inferred after observing the tasks, it is called the hyper-posterior distribution, and serves as a prior for a new task, i.e., when learning a new task, the learner draws a prior from Q and then uses it for learning.

Ideally, the performance of the hyper-posterior Q is measured by the expected generalization loss of learning new tasks using priors generated from Q. This quantity is denoted as the transfer error DISPLAYFORM3 While er (Q, ?? ) is not computable, we can however evaluate the empirical multi-task error DISPLAYFORM4 Although er (Q, ?? ) cannot be evaluated, we will prove a PAC-Bayes style upper bound on it, that can be minimized over Q.

It is important to emphasize that the hyper-posterior is evaluated on new, independent, tasks from the environment (and not on the observed tasks which ae used for meta-training).In the single-task PAC-Bayes setting one selects a prior P ??? M before seeing the data, and updates it to a posterior Q ??? M after observing the training data.

In the present lifelong setup, following the framework in BID25 , one selects an initial hyper-prior distribution P, essentially a distribution over prior distributions P , and, following the observation of the data from all tasks, updates it to a hyper-posterior distribution Q. As a simple example, assume the initial prior P is a Gaussian distribution over neural network weights, characterized by a mean and covariance.

A hyper distribution would correspond in this case to a distribution over the mean and covariance of P .

In this section we present a novel bound on the transfer error in the lifelong learning setup.

The theorem is proved in the appendix 8.1.Theorem 2 (Lifelong-learning PAC-Bayes bound).

Let Q : Z m ?? M ??? M be a mapping (singletask learning procedure), and let P be some predefined hyper-prior distribution.

Then for any ?? ??? (0, 1] the following inequality holds uniformly for all hyper-posteriors distributions Q with probability of at least 1 ??? ??, DISPLAYFORM0 Notice that the transfer error FORMULA8 is bounded by the empirical multi-task error (7) plus two complexity terms.

The first is the average of the task-complexity terms of the observed tasks.

This term converges to zero in the limit of a large number of samples in each task (m i ??? ???).

The second is an environment-complexity term.

This term converges to zero if infinite number of tasks is observed from the task environment (n ??? ???).

As in BID25 , our proof is based on two main steps.

The second step, similarly to BID25 , bounds the generalization error at the task-environment level (i.e, the error caused by observing only a finite number of tasks), er (Q, ?? ), by the average generalization error in the observed tasks plus the environment-complexity term.

The first step differs from BID25 .

Instead of using a single joint bound on the average generalization error, we use a single-task PAC-Bayes theorem to bound the generalization error in each task separately (when learned using priors from the hyper-posterior), and then use a union bound argument.

By doing so our bound takes into account the specific number of samples in each observed task (instead of their harmonic mean).

Therefore our bound is better adjusted the observed data set.

Another distinction is in the case in which an infinite number of tasks is observed, but each has only a few samples.

In contrast to BID25 , in Theorem 2 the hyper-prior still has an effect on the bound.

Intuitively, the prior knowledge we had before observing tasks (hyperprior) should still have an effect on the bound unless the observed tasks contain enough information (samples).Our proof technique can utilize different single-task bounds in each of the two steps.

In section 8.1 we use McAllester's bound (Theorem 1), which is tighter than the lemma used in BID25 .

Therefore, the complexity terms are in the form of BID25 .

This means the bound is tighter (e.g. see BID31 Theorems 5 and 6).

In section 8.2 we demonstrate how our technique can use other, possibly tighter, single-task bounds.

Finally, in the experiments section we empirically evaluate the transfer risk obtained when using the bounds as learning objectives and show that our bound leads to far better results.

DISPLAYFORM1

As in the single-task case, the bound of Theorem 2 can be evaluated from the training data and so can serve as a minimization objective for a principled lifelong-learning algorithm.

Since the bound holds uniformly for all Q, it is ensured to hold also for the inferred optimal Q * .

In this section we will derive a practical learning procedure that can applied to a large family of differentiable models, including deep neural networks.

In this section we will choose a specific form for the Hyper-posterior distribution Q, which enables practical implementation.

Given a parametric family of priors P ?? : ?? ??? R N P , the space of hyperposteriors consists of all distributions over R N P .

We will limit our search to a certain family of hyper-posteriors by choosing a Gaussian distribution in the space of prior parameters, DISPLAYFORM0 where ?? Q > 0 is a predefined constant.

Notice that Q appears in the bound (8) in two forms (i) divergence from the hyper-prior D KL (Q||P) and (ii) expectations over P ??? Q.First, by setting the hyper-prior as Gaussian, P = N 0, ?? 2 P I N P ??N P , where ?? P > 0 is another constant, we get a simple form for the KLD term, DISPLAYFORM1 Note that the hyper-prior serves as a regularization term for learning the prior.

Second, the expectations can be easily approximated using by averaging several Monte-Carlo samples of P .

Notice that sampling from Q ?? P means adding Gaussian noise to the prior parameters ?? P during training, ?? P = ?? P + ?? P , ?? P ??? N 0, ?? 2 Q I N P ??N P .

This means the learned parameters must be robust to perturbations, which encourages selecting solutions which are less prone to over-fitting and are expected to generalize better BID5 BID13 .

The term appearing on the RHS of the lifelong learning bound in (8) can be compactly written as DISPLAYFORM0 where we defined, DISPLAYFORM1 and DISPLAYFORM2 Theorem 2 allows us to choose any single-task learning procedure Q(S i , P ) : Z mi ?? M ??? M to infer a posterior.

We will use a procedure which minimizes J i (?? P ) due to the following advantages: (i) It minimizes a bound on the generalization error of the observed task (see section 8.1). (ii) It uses the prior knowledge gained from the prior P to get a tighter bound and a better learning objective. (iii) As will be shown next, formulating the single task learning as an optimization problem enables joint learning of the shared prior and the task posteriors.

To formulate the single-task learning as an optimization problem, we choose a parametric form for the posterior of each task Q ??i , ?? i ??? R N Q (see section 4.3 for an explicit example).

The single-task learning algorithm can be formulated as ?? * i = argmin ??i J i (?? P , ?? i ), where we abuse notation by denoting the term J i (?? P ) evaluated with posterior parameters ?? i as J i (?? P , ?? i ).The lifelong-learning problem of minimizing J(?? P ) over ?? P can now be written more explicitly, DISPLAYFORM3

In this section we make the lifelong-learning optimization problem (14) more explicit by defining a model for the posterior and prior distributions.

First, we define the hypothesis class H as a family of functions parameterized by a weight vector h w : w ??? R d .

Given this parameterization, the posterior and prior are distributions over R d .We will present an algorithm for any differentiable model 6 , but our aim is to use neural network (NN) architectures.

In fact, we will use Stochastic NNs BID12 BID4 since in our setting the weights are random and we are optimizing their posterior distribution.

The techniques presented next will be mostly based on BID4 .Next we define the posteriors Q ??i and the prior P ?? P as factorized Gaussian distributions 7 , DISPLAYFORM0 where for each task, the posterior parameters vector ?? i = (?? i , ?? i ) ??? R 2d is composed of the means and log-variances of each weight , ?? i,k and ?? i,k = log ?? 2 P,k , k = 1, ..., d.8 The shared prior vector ?? P = (?? P , ?? P ) ??? R 2d has a similar structure.

Since we aim to use deep models where d could be in the order of millions, distributions with more parameters might be impractical.

Since Q ??i and P ?? P are factorized Gaussian distributions the KLD takes a simple analytic form, DISPLAYFORM1

As an underlying optimization method, we will use stochastic gradient descent (SGD) 9 .

In each iteration, the algorithm takes a parameter step in a direction of an estimated negative gradient.

As is well known, lower variance facilitates convergence and its speed.

Recall that each single-task bound is composed of an empirical error term and a complexity term (12).

The complexity term is a simple function of D KL (Q ??i ||P ?? P ) (16), which can easily be differentiated analytically.

However, evaluating the gradient of the empirical error term is more challenging.

Recall the definition of the empirical error, er (Q ??i , S i ) = E w???Q ?? i (1/m i ) mi j=1 (h w , z i,j ).

This term poses two major challenges.

(i) The data set S i could be very large making it expensive to cycle over all the m i samples. (ii) The term (h w , z j ) might be highly non-linear in w, rendering the expectation intractable.

Still, we can get an unbiased and low variance estimate of the gradient.

First, instead of using all of the data for each gradient estimation we will use a randomly sampled mini-batch S i ??? S i .

Next, we require an estimate of a gradient of the form ??? ?? E w???Q ?? f (w) which 6 The only assumption on hw : w ??? R d is that the loss function (hw, z) is differentiable w.r.t w. 7 This choice makes optimization easier, but in principle we can use other distributions as long as the PDF is differentiable w.r.t the parameters.8 Note that we use ?? = log ?? 2 as a parameter in order to keep the parameters unconstrained (while ?? 2 = exp(??) is guaranteed to be strictly positive).9 Or some other variant of SGD.is a common problem in machine learning.

We will use the 're-parametrization trick BID29 BID15 which is an efficient and low variance method 10 .

The reparametrization trick is easily applicable in our setup since we are using Gaussian distributions.

The trick is based on describing the Gaussian distribution w ??? Q ??i (15) as first drawing ?? ??? N (0, I d??d ) and then applying the deterministic function w(?? i , ??) = ?? i + ?? i ?? (where is an element-wise multiplication).Therefore, we can switch the order of gradient and expectation to get DISPLAYFORM0 The expectation can be approximated by averaging a small number of Monte-Carlo samples with reasonable accuracy.

For a fixed sampled ??, the gradient ??? ?? f (w(?? i , ??)) is easily computable with backpropagation.

In summary, the Lifelong learning by Adjusting Priors (LAP) algorithm is composed of two phases In the first phase (Algorithm 1, termed "meta-training") several observed "training tasks" are used to learn a prior.

In the second phase (Algorithm 2, termed "meta-testing") the previously learned prior is used for the learning of a new task (which was unobserved in the first phase).

Note that the first phase can be used independently as a multi-task learning method.

Both algorithms are described in pseudo-code in the appendix (section 8.4).

To illustrate the setup visually, we will consider a simple toy example of a 2D estimation problem.

In each task, the goal is to estimate the mean of the data generating distribution.

In this setup, the samples z are vectors in R 2 .

The hypothesis class is a the set of 2D vectors, h ??? R 2 .

As a loss function we will use the Euclidean distance, (h, z) h ??? z 2 2 .

We artificially create the data of each task by generating 50 samples from the appropriate distribution: N (2, 1) , 0.1 2 I 2??2 in task 1, and N (4, 1) , 0.1 2 I 2??2 in task 2.

The prior and posteriors are 2D factorized Gaussian DISPLAYFORM0 We run Algorithm 1 (meta-training) with complexity terms according to Theorem 1.

As seen in FIG2 , the learned prior (namely, the prior learned from the two tasks) and single-task posteriors can be understood intuitively.

First, the posteriors are located close to the ground truth means of each task, with relatively small uncertainty covariance.

Second, the learned prior is located in the middle between the two posteriors, and its covariance is larger in the first dimension.

This is intuitively reasonable since the prior learned that tasks are likely to have values of around 1 in dimension 2 and values around 3 in the dimension 1, but with larger variance.

Thus, new similar tasks can be learned using this prior with fewer samples.

In this section we demonstrate the performance of our transfer method with image classification tasks solved by deep neural networks.11 .In image classification, the data samples, z (x, y), are pairs of a an image, x, and a label, y. The hypothesis class h w : w ??? R d is a the set of neural networks with a given architecture (which will be specified later).

As a loss function (h w , z) we will use the cross-entropy loss.

We conduct an experiment with a task environment in which each task is created by a random permutation of the labels of the MNIST dataset BID18 .

The meta-training set is composed of 5 tasks from the environment, each with 60, 000 training examples.

Following the meta-training phase, the learned prior is used to learn a new meta-test task with fewer training samples (2, 000).

The network architecture is a small CNN with 2 convolutional-layers, a linear hidden layer and a linear output layer.

See section 8.3 for more implementation details.

Figure 1: Toy example: the orange are red dots are the samples of task 1 and 2, respectively, and the green and purple dots are the means of the posteriors of task 1 and 2, respectively.

The mean of the prior is a blue dot.

The ellipse around each distribution's mean represents the covariance matrix.

We compare the average generalization performance (test error) in learning a new (meta-test) when using the following methods.

As a baseline, we measure the performance of learning without transfer from the training-tasks:??? Scratch-standard: standard learning from scratch (non-stochastic network).??? Scratch-stochastic: stochastic learning from scratch (stochastic network with no prior/complexity term).Other methods transfer knowledge from only one of the train tasks:??? Warm-start-transfer: Standard learning with initial weights taken from the standard learning of a single task from the meta-train set (with 60, 000 examples).??? Oracle-transfer: Same as the previous method, but all layers besides the output layer are frozen (unchanged from their initial value), which is a common practice for transfer learning in computer vision BID28 .

Note that in this method we are manually inserting prior knowledge based on our familiarity with the task environment.

Therefore this method can be considered an "oracle".Finally, we compare methods which transfer knowledge from all of the training tasks:??? LAP-M: The objective is based on Theorem 2 -the lifelong-learning bound obtained using Theorem 3 (McAllester's single-task bound).??? LAP-S: The objective is based on the lifelong-learning bound of eq. FORMULA2 This lifelong-learning bound is obtained using Theorem 4 (Seeger's single-task bound).??? LAP-PL: In this method we use the main theorem of BID25 as an objective for the algorithm, instead of Theorem 2.??? LAP-KLD:

Here we use a task-complexity term which is simply the KLD between the sampled prior and the task posterior and an environment-complexity term which is the KLD between the hyper-prior and hyper-posterior.

This minimization problem is equivalent to maximization of the Evidence-Lower-Bound (ELBO) when using a variational methods to approximate the maximum-likelihood parameters of a hierarchical generative model 12 .

Note that the ELBO can also be interpreted as an upper bound on the generalization error.

However the bound is looser than the one obtained using PAC-Bayesian methods 13 .???

Averaged-prior: Each of the training tasks is learned in a standard way to obtain a weights vector, w i .

The learned prior is set as an isotropic Gaussian with unit variances and a mean vector which is the average of w i , i = 1, .., n.

This prior is used for meta-testing as in LAP-S.??? MAML: The Model-Agnostic-Meta-Learning (MAML) algorithm by BID10 finds an optimal initial weight for learning tasks from a given environment.

We report the best results obtained with all combinations of the following representative hyperparameters: 1-3 gradient steps in meta-training, 1-20 gradient steps in meta-testing and ?? ??? {0.01, 0.1, 0.4}. As can be seen in TAB0 , the best results are obtained with the "oracle" method.

Recall that the oracle method has the "unfair" advantage of a "hand-engineered" transfer technique which is based on knowledge about the problem.

In contrast, the other methods must automatically learn the task environment by observing several tasks.

The LAP-M and LAP-S variants of the LAP algorithm improves considerably over learning from scratch and over the naive warm-start transfer and are very close the the "oracle" method.

As expected the the LAP-S variant preforms better since it uses a tighter bound (see section 8.2).The other variants of the LAP algorithm with other objectives performed much worse.

First, the results for LAP-PL demonstrate the importance of the tight generalization bound developed in our work.

Second, the results for the LAP-KLD show that deriving objectives from variational-inference techniques that maximize a lower bound on the model evidence, might be less a successful approach than deriving objectives which minimize upper bounds on the generalization error.

The results for the "averaged-prior" method are about the same as learning from scratch.

Due to the high non-linearity of the problem, averaging weights was not expected to perform well .The results of MAML are comparable to the results of the LAP algorithm.

Note that MAML is specifically suited for learning from many few-shot tasks, in which taking a small number of gradient steps in each task is effective for learning.

However, in our experiment, there are a few tasks but more than a few samples.

Still, the method performed quite well with several different sets of hyperparameters 14 .

We have presented a framework for representational lifelong learning, motivated by PAC-Bayes generalization bounds, and implemented through the adjustment of a learned prior, based on tasks encountered so far.

The framework bears conceptual similarity to the empirical Bayes method while not being Bayesian, and is implemented at the level of tasks rather than samples.

Combining the general approach with the rich representational structure of deep neural networks, and learning through gradient based methods leads to an efficient procedure for lifelong learning, as motivated theoretically and demonstrated empirically.

While our experimental results are preliminary, we believe that our work attests to the utility of using rigorous performance bounds to derive learning algorithms, and demonstrates that tighter bounds indeed lead to improved performance.

There are several open issues to consider.

First, the current version learns to solve all available tasks in parallel, while a more useful procedure should be sequential in nature.

This can be easily incorporated into our framework by updating the prior following each novel task.

Second, our method requires training stochastic models which is challenging due to the the high-variance gradients.

We we would like to develop new methods within our framework which have more stable convergence and are easier to apply in larger scale problems.

Third, there is much current effort in reinforcement learning to augment model free learning with model based components, where some aspects of the latter are often formulated as supervised learning tasks.

Incorporating our approach in such a context would be a worthwhile challenge.

In fact, a similar framework to ours was recently proposed within an RL setting BID33 , although it was not motivated from performance guarantees as was our approach, but rather from intuitive heuristic arguments.

In this section we prove Theorem 2.

The proof is based on two steps, both use McAllaster's classical PAC-Bayes bound.

In the first step we use it to bound the error which is caused due to observing only a finite number of samples in each of the observed tasks.

In the second step we use it again to bound the generalization error due to observing a limited number of tasks from the environment.

We start by restating the classical PAC-Bayes bound BID22 Shalev-Shwartz & BenDavid, 2014 ) using general notations.

Theorem 3 (Classical PAC-Bayes bound, general formulation).

Let X be some 'sample' space and X some distribution over X .

Let F be some 'hypothesis' space.

Define a 'loss function' g(f, X) : DISPLAYFORM0 .., X K } be a sequence of K independent random variables distributed according to X. Let ?? be some prior distribution over F (which must not depend on the samples X 1 , ..., X K ).

For any ?? ??? (0, 1], the following bound holds uniformly for all 'posterior' distributions ?? over F (even sample dependent), DISPLAYFORM1 First step We use Theorem 3 to bound the generalization error in each of the observed tasks when learning is done by an algorithm Q : Z mi ?? M ??? M which uses a prior and the samples to output a distribution over hypothesis.

Let i ??? 1, ..., n be the index of some observed task.

We use Theorem 3 with the following substitutions.

The samples are X k z i,j , K m i , and their distribution is X D i .

We define a 'tuple hypothesis'f = (P, h) where P ??? M and h ??? H. The 'loss function' is the regular loss which uses only the h element in the tuple, g(f, X) (h, z).

We define the 'prior over hypothesis', ?? (P, P ), as some distribution over M ?? H in which we first sample P from P and then sample h from P .

According to Theorem 3, the 'posterior over hypothesis' can be any distribution (even sample dependent), in particular, the bound will hold for the following family of distributions over M ?? H, ?? (Q, Q(S i , P )), in which we first sample P from Q and then sample h from Q = Q(S i , P ).

The KLD term is, DISPLAYFORM0 Plugging in to (17) we obtain that for any ?? i > 0 15 Recall that Q(Si, P ) is the posterior distribution which is the output of the learning algorithm Q() which uses the data Si and the prior P .where we define, ??(K, ??, ??, ??) = 1 K D KL (??||??) + log 2 ??? K ?? , and, er ??, X DISPLAYFORM1 Using the above theorem we get an alternative intra-task bound to (19), DISPLAYFORM2 2?? i + 2?? i er (Q(S i , P ), S i ), ???Q ??? 1 ??? ?? i , where, DISPLAYFORM3 While the classical bound of Theorem 1 converges at a rate of about 1/ ??? m (as in basic VC-like bounds), the bound of Theorem 4 converges even faster (at a rate if

The network architecture used for the permuted-labels experiment is a small CNN with 2 convolutional-layers of 10 and 20 filters, each with 5 ?? 5 kernels, a hidden linear layer with 50 units and a linear output layer.

Each convolutional layer is followed by max pooling operation with kernel of size 2.

Dropout with p = 0.5 is performed before the output layer.

In both networks we use ELU BID6 (with ?? = 1) as an activation function.

Both phases of the LAP algorithm (algorithms 1 and 2) ran for 200 epochs, with batches of 128 samples in each task.

We take only one Monte-Carlo sample of the stochastic network output in each step.

As optimizer we used ADAM BID14 with learning rate of 10 ???3 .

The means of the weights (?? parameters) are initialized randomly by N 0, 0.1 2 , while the log-var of the weights (?? parameters) are initialized by N ???10, 0.1 2 .

The hyper-prior and hyper-posterior parameters are ?? P = 2000 and ?? Q = 0.001 respectively and the confidence parameter was chosen to be ?? = 0.1 .To evaluate the trained network we used the maximum of the posterior for inference (i.e. we use only the means the weights).18 17 More recent works presented possibly tighter PAC-Bayesian bounds by taking into account the empirical variance BID34 or by specializing the bound deep for neural networks BID24 .

However, we leave the incorporation of these bounds for future work.18 Classifying using the the majority vote of several runs gave similar results in this experiment.

<|TLDR|>

@highlight

We develop a lifelong learning approach to transfer learning based on PAC-Bayes theory, whereby priors are adjusted as new tasks are encountered thereby facilitating the learning of novel tasks.

@highlight

A novel PAC-Bayesian risk bound that serves as an objective function for multi-task machine learning, and an algorithm for minimizing a simplified version of that objective function.

@highlight

Extends existing PAC-Bayes bounds to multi-task learning, to allow the prior to be adapted across different tasks.