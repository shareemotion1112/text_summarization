Approaches to continual learning aim to successfully learn a set of related tasks that arrive in an online manner.

Recently, several frameworks have been developed which enable deep learning to be deployed in this learning scenario.

A key modelling decision is to what extent the architecture should be shared across tasks.

On the one hand, separately modelling each task avoids catastrophic forgetting but it does not support transfer learning and leads to large models.

On the other hand, rigidly specifying a shared component and a task-specific part enables task transfer and limits the model size, but it is vulnerable to catastrophic forgetting and restricts the form of task-transfer that can occur.

Ideally, the network should adaptively identify which parts of the network to share in a data driven way.

Here we introduce such an approach called Continual Learning with Adaptive Weights (CLAW), which is based on probabilistic modelling and variational inference.

Experiments show that CLAW achieves state-of-the-art performance on six benchmarks in terms of overall continual learning performance, as measured by classification accuracy, and in terms of addressing catastrophic forgetting.

Continual learning (CL), sometimes called lifelong or incremental learning, refers to an online framework where the knowledge acquired from learning tasks in the past is kept and accumulated so that it can be reused in the present and future.

Data belonging to different tasks could potentially be non i.i.d. (Schlimmer & Fisher, 1986; Sutton & Whitehead, 1993; Ring, 1997; Schmidhuber, 2013; Nguyen et al., 2018; Schmidhuber, 2018) .

A continual learner must be able to learn a new task, crucially, without forgetting previous tasks (Ring, 1995; Srivastava et al., 2013; Serra et al., 2018; Hu et al., 2019) .

In addition, CL frameworks should continually adapt to any domain shift occurring across tasks.

The learning updates must be incremental -i.e, the model is updated at each task only using the new data and the old model, without access to all previous data (from earlier tasks) -due to speed, security and privacy constraints.

A compromise must be found between adapting to new tasks and enforcing stability to preserve knowledge from previous tasks.

Excessive adaptation could lead to inadvertent forgetting of how to perform earlier tasks.

Indeed, catastrophic forgetting is one of the main pathologies in continual learning (McCloskey & Cohen, 1989; Ratcliff, 1990; Robins, 1993; French, 1999; Pape et al., 2011; Goodfellow et al., 2014a; Achille et al., 2018; Diaz-Rodriguez et al., 2018; Zeno et al., 2018; Ahn et al., 2019; Parisi et al., 2019; Pfulb & Gepperth, 2019; Rajasegaran et al., 2019) .

Many approaches to continual learning employ an architecture which is divided a priori into (i) a slowly evolving, global part; and (ii) a quickly evolving, task-specific, local part.

This is one way to enable multi-task transfer whilst mitigating catastrophic forgetting, which has proven to be effective (Rusu et al., 2016b; Fernando et al., 2017; Yoon et al., 2018) , albeit with limitations.

Specifying a priori the shared global, and task-specific local parts in the architecture restricts flexibility.

As more complex and heterogeneous tasks are considered, one would like a more flexible, data-driven approach to determine the appropriate amount of sharing across tasks.

Here, we aim at automating the architecture adaptation process so that each neuron of the network can either be kept intact, i.e. acting as global, or adapted to the new task locally.

Our proposed variational inference framework is flexible enough to learn the range within which the adaptation parameters can vary.

We introduce for each neuron one binary parameter controlling whether or not to adapt, and two parameters to control the magnitude of adaptation.

All parameters are learnt via variational inference.

We introduce our framework as an expansion of the variational continual learning algorithm (Nguyen et al., 2018) , whose variational and sequential Bayesian nature makes it convenient for our modelling and architecture adaptation procedure.

Our modelling ideas can also be applied to other continual learning frameworks, see the Appendix for a brief discussion.

We highlight the following contributions: (1) A modelling framework which flexibly automates the adaptation of local and global parts of the (multi-task) continual architecture.

This optimizes the tradeoff between mitigating catastrophic forgetting and improving task transfer.

(2) A probabilistic variational inference algorithm which supports incremental updates with adaptively learned parameters.

( 3) The ability to combine our modelling and inference approaches without any significant augmentation of the architecture (no new neurons are needed).

(4) State-of-the-art results in six experiments on five datasets, which demonstrate the effectiveness of our framework in terms of overall accuracy and reducing catastrophic forgetting.

We briefly discuss three related approaches to continual learning: (a) regularisation-based, (b) architecture-based and (c) memory-based.

We provide more details of related work in Section A in the Appendix.

(a) A complementary approach to CLAW is the regularisation-based approach to balance adaptability with catastrophic forgetting: a level of stability is kept via protecting parameters that greatly influence the prediction against radical changes, while allowing the rest of the parameters to change without restriction (Li & Hoiem, 2016; Zenke et al., 2017; Chaudhry et al., 2018; Nguyen et al., 2018; Srivastava et al., 2013; Vuorio et al., 2018; Aljundi et al., 2019c) .

The elastic weight consolidation (EWC) algorithm by Kirkpatrick et al. (2017) is a seminal example, where a quadratic penalty is imposed on the difference between parameter values of the old and new tasks.

One limitation is the high level of hand tuning required.

(b) The architecture-based approach aims to deal with stability and adaptation issues by a fixed division of the architecture into global and local parts (Rusu et al., 2016b; Fernando et al., 2017; Shin et al., 2017; Kaplanis et al., 2018; Xu & Zhu, 2018; Yoon et al., 2018; Li et al., 2019b) .

(c) The memory-based approach relies on episodic memory to store data (or pseudo-data) from previous tasks (Ratcliff, 1990; Robins, 1993; Thrun, 1996; Schmidhuber, 2013; Hattori, 2014; Mocanu et al., 2016; Rebuffi et al., 2017; Kamra et al., 2017; Shin et al., 2017; Rolnick et al., 2018; van de Ven & Tolias, 2018; Wu et al., 2018; Titsias et al., 2019) .

Limitations include overheads for tasks such as data storage, replay, and optimisation to select (or generate) the points.

CLAW can as well be seen as a combination of a regularisation-based approach (the variational inference mechanism) and a modelling approach which automates the architecture building process in a data-driven manner, avoiding the overhead resulting from either storing or generating data points from previous tasks.

CLAW is also orthogonal to (and simple to combine with, if needed) memory-based methods.

In this paper, we use Variational Continual Learning (VCL, Nguyen et al., 2018) as the underlying continual learning framework.

However, our methods apply to other frameworks, see Appendix (Section A.1).

VCL is a variational Bayesian framework where the posterior of the model parameters θ is learnt and updated continually from a sequence of T datasets, {x

, where t = 1, 2, . . . , T and N t is the size of the dataset associated with the t-th task.

More specifically, denote by p(y|θ, x) the probability distribution returned by a discriminative classifier with input x, output y and parameters θ.

, we approximate the intractable posterior p(θ|D 1:t ) after observing the first t datasets via a tractable variational distribution q t as:

where q 0 is the prior p, p(

t ), and Z t is the normalizing constant which does not depend on θ but only on the data D. This framework allows the approximate posterior q t (θ) to be updated incrementally from the previous approximate posterior q t−1 (θ) in an online fashion.

In VCL, the approximation in (1) is performed by minimizing the following KL-divergence over a family Q of tractable distributions:

This framework can be enhanced to further mitigate catastrophic forgetting by using a coreset (Nguyen et al., 2018) , i.e. a representative set of data from previously observed tasks that can serve as memory and can be revisited before making a decision.

As discussed in the Related Work, this leads to overhead costs of memory and optimisation (selecting most representative data points).

Previous work on VCL considered simple models without automatic architecture building or adaptation.

In earlier CL approaches, the parts of the network architecture that are shared among the learnt tasks are designated a priori.

To alleviate this rigidity and to effectively balance adaptation and stability, we propose a multi-task, continual model in which the adaptation of the architecture is data-driven by learning which neurons need to be adapted as well as the maximum adaptation capacity for each.

All the model parameters (including those used for adaptation) are estimated via an efficient variational inference algorithm which incrementally learns from data of the successive tasks, without a need to store (nor generate) data from previous tasks and with no expansion in the network size.

With model parameters θ, the overall variational objective we aim at maximising at task with index t is equivalent to the following online marginal likelihood:

We propose a framework where the architecture, whose parameters are θ, is flexibly adapted based on the available tasks, via a learning procedure that will be described below.

With each task, we automate the adaptation of the neuron contributions.

Both the adaptation decisions (i.e. whether or not to adapt) and the maximum allowed degree of adaptation for every neuron are learnt.

We refer to the binary adaptation variable as α.

There is another variable s that is learnt in a multi-task fashion to control the maximum degree of adaptation, such that the expression b = s 1+e −a − 1 limits how far the task-specific weights can differ from the global weights, in case the respective neuron is to be adapted.

The parameter a depicts unconstrained adaptation, as described later.

2 We illustrate the proposed model to perform this adaptation by learning the probabilistic contributions of the different neurons within the network architecture on a task-by-task basis.

We follow this with the inference details.

Steps of the proposed modeling are listed as follows:

• For a task T , the classifier that we are modeling outputs:

• The task-specific weights w T can be expressed in terms of their global counterparts as follows:

The symbol • denotes an element-wise (Hadamard) multiplication.

• For each task T and each neuron j at layer i, α T i,j is a binary variable which indicates whether the corresponding weight is adapted (α T i,j = 1) or unadapted (α T i,j = 0).

Initially assume that the adaptation probability α T i,j follows a Bernoulli distribution with probability p i,j 3 , α T i,j ∼ Bernoulli(p i,j ).

Since this Bernoulli is not straightforward to optimise, and to adopt a scalable inference procedure based on continuous latent variables, we replace this Bernoulli with a Gaussian that has an equivalent mean and variance from which we draw α T i,j .

For the sake of attaining higher fidelity than what is granted by a standard Gaussian, we base our inference on a variational Gaussian estimation.

Though in a context different from continual learning and with different estimators, the idea of replacing Bernoulli with an equivalent Gaussian has proven to be effective with dropout (Srivastava et al., 2014; .

The approximation of the Bernoulli distribution by the corresponding Gaussian distribution is achieved by matching the mean and variance.

The mean and variance of the Bernoulli distribution are p i,j , p i,j (1 − p i,j ), respectively.

A Gaussian distribution with the same mean and variance is used to fit α

• The variable b T controls the strength of the adaptation and it limits the range of adaptation via:

So that the maximum adaptation is s.

The variable a T is an unconstrained adaptation value, similar to that in (Swietojanski & Renals, 2014) .

The addition of 1 is to facilitate the usage of a probability distribution while still keeping an adaptation range allowing for the attenuation or amplification of each neuron's contribution.

• Before facing the first dataset and learning task t = 1, the prior on the weights q 0 (w) = p(w) is chosen to be a log-scale prior, which can be expressed as: p(log |w|) ∝ c, where c is a constant.

The log-scale prior can alternatively be described as:

At a high level, adapting neuron contributions can be seen as a generalisation of attention mechanisms in the context of continual learning.

Applying this adaptation procedure to the input leads to an attention mechanism.

However, our approach is more general since we do not apply it only to the very bottom (i.e. input) layer, but throughout the whole network.

We next show how our variational inference mechanism enables us to learn the adaptation parameters.

We describe the details related to the proposed variational inference mechanism.

The adaptation parameters are included within the variational parameters.

The (unadapted version of the) model parameters θ consist of the weight vectors w. To automate adaptation, we perform inference on p i,j , which would have otherwise been a hyperparameter of the prior (Louizos et al., 2017; Molchanov et al., 2017; Ghosh et al., 2018) .

Multiplying w by (1 + bα) where α is distributed according to (5), then from (4) with random noise variable ∼ N (0, 1):

From (7) and (8), the corresponding KL-divergence between the variational posterior of w, q(w|γ) and the prior p(w) is as follows.

The subscripts are removed when q in turn is used as a subscript for improved readability.

The variational parameters are γ i,j and p i,j .

where the switch from (9) to (10) is due to the entropy computation (Bernardo & Smith, 2000) of the Gaussian q(w i,j |γ i,j ) defined in (8).

The switch from (10) to (11) is due to using a log-scale prior, similar to Appendix C in and to Section 4.2 in (Molchanov et al., 2017) .

E q(w|γ) log | | is computed via an accurate approximation similar to equation (14) in (Molchanov et al., 2017) , with slightly different values of k 1 , k 2 and k 3 .

This is a very close approximation via numerically pre-computing E q(w|γ) log | | using a third degree polynomial Molchanov et al., 2017) .

This is the form of the KL-divergence between the approximate posterior after the first task and the prior.

Afterwards, it is straightforward to see how this KL-divergence applies for the subsequent tasks in a manner similar to (2), but while taking into account the new posterior form and original prior.

The KL-divergence expression derived in (11) is to be minimised.

By minimising (11) with respect to p i,j and then using samples from the respective distributions to assign values to α i,j , adapted contributions of each neuron j at each layer i of the network are learnt per task.

Values of p i,j are constrained between 0 and 1 during training via projected gradient descent.

Using (6) to express the value of b i,j , and neglecting the constant term therein since it does not affect the optimisation, the KL-divergence in (11) is equivalent to:

Values of a i,j are straightforwardly learnt by minimising (12) with respect to a i,j .

This subsection explains how to learn the maximum adaptation variable s i,j .

Values of the maximum s i,j of the logistic function defined in (6) are learnt from multiple tasks.

For each neuron j at layer i, there is a general value s i,j and another value that is specific for each task t, referred to as s i,j,t .

This is similar to the meta-learning procedure proposed in (Finn et al., 2017) .

The following procedure to learn s is performed for each task t such that: (i) the optimisation performed to learn a task-specific value s i,j,t benefits from the warm initialisation with the general value s i,j rather than a random initial condition; and then (ii) the new information obtained from the current task t is ultimately reflected back to update the general value s i,j .

• First divide the sample N t into two halves.

For the first half, depart from the general value of s i,j as an initial condition, and use the assigned data examples from task t to learn the taskspecific values s i,j,t for the current task t. For neuron j at layer i, refer to the second term in (3),

The set of parameters θ contains s as well as other parameters, but we focus here on s in the f notation since the following procedure is developed to optimise s. Also, refer to the loss of the (classification) function f as Err(f ) = CE(f (x, θ) y), where CE stands for the cross-entropy:

• Now use the second half of the data from task t to update the general learnt value s i,j :

Where ω 1 and ω 2 are step-size parameters.

When testing on samples from task t after having faced future tasks t + 1, t + 2, . . ., the value of s i,j used is the learnt s i,j,t .

There is only one value per neuron, so the overhead resulting from storing such values is negligible.

The key steps of the algorithm are listed in Algorithm 1.

Input: A sequence of T datasets, {x

, where t = 1, 2, . . . , T and N t is the size of the dataset associated with the t-th task.

Output: q t (θ), where θ are the model parameters.

Initialise all p(|w i,j |) with a log-scale prior, as in (7).

for the current task t. for i = 1 . . .

# layers do for j = 1 . . .

# neurons at layer i do Compute p i,j using stochastic gradient descent on (11).

Compute s i,j,t using (13).

Update the corresponding general value s i,j using (14).

end for end for end for At task t, the algorithmic complexity of a single joint update of the parameters θ based on the additive terms in (12)

2 ), where L is the number of layers in the network, D is the (largest) number of neurons within a single layer, E is the number of samples taken from the random noise variable , and M is the minibatch size.

Each α is obtained by taking one sample from the corresponding p, so that does not result in an overhead in terms of the complexity.

Our experiments mainly aim at evaluating the following: (i) the overall performance of the introduced CLAW, depicted by the average classification accuracy over all the tasks; (ii) the extent to which catastrophic forgetting can be mitigated when deploying CLAW; and (iii) the achieved degree of positive forward transfer.

The experiments demonstrate the effectiveness of CLAW in achieving state-of-the-art continual learning results measured by classification accuracy and by the achieved reduction in catastrophic forgetting.

We also perform ablations in Section D in the Appendix which exhibit the relevance of each of the proposed adaptation parameters.

We perform six experiments on five datasets.

The datasets in use are: MNIST (LeCun et al., 1998) , notMNIST (Butalov, 2011) , Fashion-MNIST (Xiao et al., 2017) , Omniglot (Lake et al., 2011) and CIFAR-100 (Krizhevsky & Hinton, 2009) .

We compare the results obtained by CLAW to six different state-of-the-art continual learning algorithms: the VCL algorithm (Nguyen et al., 2018) (original form and one with a coreset), the elastic weight consolidation (EWC) algorithm (Kirkpatrick et al., 2017) , the progress and compress (P&C) algorithm , the reinforced continual learning (RCL) algorithm (Xu & Zhu, 2018) , the one referred to as functional regularisation for continual learning (FRCL) using Gaussian processes (Titsias et al., 2019) and the learn-to-grow (LTG) algorithm (Li et al., 2019b) .

Our main metric is the all-important classification accuracy.

We consider six continual learning experiments, based on the MNIST, notMNIST, Fashion-MNIST, Omniglot and CIFAR-100 datasets.

The introduced CLAW is compared to two VCL versions: VCL with no coreset and VCL with a 200-point coreset assembled by the K-center method (Nguyen et al., 2018) , EWC, P&C, RCL, FRCL (its TR version) and LTG 4 .

All the reported classification accuracy values reflect the average classification accuracy over all tasks the learner has trained on so far.

More specifically, assume that the continual learner has just finished training on a task t, then the reported classification accuracy at time t is the average accuracy value obtained from testing on equally sized sets each belonging to one of the tasks 1, 2, . . .

, t. For all the classification experiments, statistics reported are averages of ten repetitions.

Statistical significance and standard error of the average classification accuracy obtained after completing the last two tasks of each experiment are displayed in Section E in the Appendix.

As can be seen in Figure 1 , CLAW achieves state-of-the-art classification accuracy in all the six experiments.

The minibatch size is 128 for Split MNIST and 256 for all the other experiments.

More detailed descriptions of the results of every experiment are given next:

Permuted MNIST Using MNIST, Permuted MNIST is a standard continual learning benchmark (Goodfellow et al., 2014a; Kirkpatrick et al., 2017; Zenke et al., 2017) .

For each task t, the corresponding dataset is formed by performing a fixed random permutation process on labeled MNIST images.

This random permutation is unique per task, i.e. it differs for each task.

For the hyperparameter λ of EWC, which controls the overall contribution from previous data, we experimented with two values, λ = 1 and λ = 100.

We report the latter since it has always outperformed EWC with λ = 1 in this experiment.

EWC with λ = 100 has also previously produced the best EWC classification results (Nguyen et al., 2018) .

In this experiment, fully connected single-head networks with two hidden layers are used.

There are 100 hidden units in each layer, with ReLU activations.

Adam (Kingma & Ba, 2015) is the optimiser used in the 6 experiments with η = 0.001, β 1 = 0.9 and β 2 = 0.999.

Further experimental details are given in Section C in the Appendix.

Results of the accumulated classification accuracy, averaged over tasks, on a test set are displayed in Figure 1a .

After 10 tasks, CLAW achieves significantly (check the Appendix) higher classification results than all the competitors.

Split MNIST In this MNIST based experiment, five binary classification tasks are processed in the following sequence: 0/1, 2/3, 4/5, 6/7, and 8/9 (Zenke et al., 2017) .

The architecture used consists of fully connected multi-head networks with two hidden layers, each consisting of 256 hidden units with ReLU activations.

As can be seen in Figure 1b , CLAW achieves the highest classification accuracy.

Split Fashion-MNIST Fashion-MNIST is a dataset whose size is the same as MNIST but it is based on different (and more challenging) 10 classes.

The five binary classification tasks here are: T-shirt/Trouser, Pullover/Dress, Coat/Sandals, Shirt/Sneaker, and Bag/Ankle boots.

The architecture used is the same as in Split notMNIST.

In most of the continual learning tasks (including the more significant, later ones) CLAW achieves a clear classification improvement (Figure 1d ).

Omniglot This is a sequential learning task of handwritten characters of 50 alphabets (a total of over 1,600 characters with 20 examples each) belonging to the Omniglot dataset (Lake et al., 2011) .

We follow the same way via which this task has been used in continual learning before Titsias et al., 2019) ; handwritten characters from each alphabet constitute a separate task.

We thus have 50 tasks, which also allows to evaluate the scalability of the frameworks in comparison.

The model used is a CNN.

To deal with the convolutions in CLAW, we used the idea proposed and referred to as the local reparameterisation trick by Kingma et al. (2014; , where a single global parameter is employed per neuron activation in the variational distribution, rather than employing parameters for every constituent weight element 5 .

Further details about the CNN used are given in Section C.

The automatically adaptable CLAW achieves better classification accuracy (Figure 1e ).

This dataset consists of 60,000 colour images of size 32 × 32.

It contains 100 classes, with 600 images per class.

We use a split version CIFAR-100.

Similar to Lopez-Paz & Ranzato (2017), we perform a 20-task experiment with a disjoint subset of five classes per task.

CLAW achieves significantly higher classification accuracy (Figure 1f ) -also higher than the previous state of the art on CIFAR-100 by .

Details of the used CNN are in Section C.

A conclusion that can be taken from Figure 1 (a-f) is that CLAW consistently achieves state-of-the-art results (in all the 6 experiments).

It can also be seen that CLAW scales well.

For instance, the difference between CLAW and the best competitor is more significant with Split notMNIST than it is with the first two experiments, which are based on the smaller and less challenging MNIST.

Also, CLAW achieves good results with Omniglot and CIFAR-100.

To assess catastrophic forgetting, we show how the accuracy on the initial task varies over the course of the training procedure on the remaining tasks .

Since Omniglot (and CIFAR-100) contain a larger number of tasks: 50 (20) tasks, i.e. 49 (19) remaining tasks after the initial task, this setting is more relevant for Omniglot and CIFAR-100.

We nonetheless display the results for Split MNIST, Split notMNIST, Split Fashion-MNIST, Omniglot and CIFAR-100.

As can be seen in Figure 2 , CLAW (at times jointly) achieves state-of-the-art performance retention degrees.

Among the competitors, P&C and LTG also achieve high performance retention degrees.

An empirical conclusion that can be made out of this and the previous experiment, is that CLAW achieves better overall continual learning results, partially thanks to the way it addresses catastrophic forgetting.

The idea of adapting the architecture by adapting the contributions of neurons of each layer also seems to be working well with datasets like Omniglot and CIFAR-100, giving directions for imminent future work where CLAW can be extended for other application areas based on CNNs.

The purpose of this experiment is to assess the impact of learning previous tasks on the current task.

In other words, we want to evaluate whether an algorithm avoids negative transfer, by evaluating the relative performance achieved on a unique task after learning a varying number of previous tasks .

From Figure 3 , we can see that CLAW achieves state-of-the-art results in 4 out of the 5 experiments (at par in the fifth) in terms of avoiding negative transfer.

We introduced a continual learning framework which learns how to adapt its architecture from the tasks and data at hand, based on variational inference.

Rather than rigidly dividing the architecture into shared and task-specific parts, our approach adapts the contributions of each neuron.

We achieve The impact of learning previous tasks on a specific task (the last task) is inspected and used as a proxy for evaluating forward transfer.

This is performed by evaluating the relative performance achieved on a unique task after learning a varying number of previous tasks.

This means that the value at x-axis = 1 refers to the learning accuracy of the last task after having learnt solely one task (only itself), the value at 2 refers to the learning accuracy of the last task after having learnt two tasks (an additional previous task), etc.

Overall, CLAW achieves state-of-the-art results in 4 out of the 5 experiments (at par in the fifth) in terms of avoiding negative transfer.

Best viewed in colour.

that without having to expand the architecture with new layers or new neurons.

Results of six different experiments on five datasets demonstrate the strong empirical performance of the introduced framework, in terms of the average overall continual learning accuracy and forward transfer, and also in terms of effectively alleviating catastrophic forgetting.

We begin by briefly summarising the contents of the Appendix below:

• Related works are described in Section A, followed by a brief discussion on the potential applicability of CLAW to another continual learning (CL) framework in Section A.1.

• In Section E, we provide the statistical significance and standard error of the average classification accuracy results obtained after completing the last two tasks from each experiment.

• Further experimental details are given in Section C.

• In Section D and Figures 4-8 , we display the results of performed ablations which manifest the relevance of each adaptation parameter.

A complementary approach to CLAW, which could be combined with it, is the regularisation-based approach to balance adaptability with catastrophic forgetting: a level of stability is kept via protecting parameters that greatly influence the prediction against radical changes, while allowing the rest of the parameters to change without restriction (Li & Hoiem, 2016; Vuorio et al., 2018) .

In (Zenke et al., 2017) , the regulariser is based on synapses where an importance measure is locally computed at each synapse during training, based on their respective contributions to the change in the global loss.

During a task change, the less important synapses are given the freedom to change whereas catastrophic forgetting is avoided by preventing the important synapses from changing (Zenke et al., 2017) .

The elastic weight consolidation (EWC) algorithm, introduced by Kirkpatrick et al. (2017) , is a seminal example of this approach where a quadratic penalty is imposed on the difference between parameter values of the old and new tasks.

One limitation of EWC, which is rather alleviated by using minibatch or stochastic estimates, appears when the output space is not low-dimensional, since the diagonal of the Fisher information matrix over parameters of the old task must be computed, which requires a summation over all possible output labels (Kirkpatrick et al., 2017; Zenke et al., 2017; .

In addition, the regularisation term involves a sum over all previous tasks with a term from each and a hand-tuned hyperparameter that alters the weight given to it.

The accumulation of this leads to a lot of hand-tuning.

The work in (Chaudhry et al., 2018 ) is based on penalising confident fitting to the uncertain knowledge by a maximum entropy regulariser.

Another seminal algorithm based on regularisation, which can be applied to any model, is variational continual learning (VCL) (Nguyen et al., 2018) which formulates CL as a sequential approximate (variational) inference problem.

However, VCL has only been applied to simple architectures, not involving any automatic model building or adaptation.

The framework in incrementally matches the moments of the posterior of a Bayesian neural network that has been trained on the first and then the second task, and so on.

Other algorithms pursue regularisation approaches based on sparsity (Srivastava et al., 2013; .

For example, the work in (Aljundi et al., 2019c) encourages sparsity on the neuron activations to alleviate catastrophic forgetting.

The l 2 distance between the top hidden activations of the old and new tasks is used for regularisation in (Jung et al., 2016) .

This approach has achieved good results, but is computationally expensive due to the necessity of computing at least a forward pass for every new data point through the network representing the old task (Zenke et al., 2017) .

Other regularisation-based continual learning algorithms include (Ebrahimi et al., 2019; Park et al., 2019) .

Another approach is the architecture-based one where the principal aim is to administer both the stability and adaptation issues via dividing the architecture into reusable parts that are less prone to changes, and other parts especially devoted to individual tasks (Rusu et al., 2016b; Fernando et al., 2017; Yoon et al., 2018; Du et al., 2019; He et al., 2019; Li et al., 2019a; Xu et al., 2019) .

To learn a new task in the work by Rusu et al. (2016a) , the whole network from the previous task is first copied then augmented with a new part of the architecture.

Although this is effective in eradicating catastrophic forgetting, there is a clear scalability issue since the architecture growth can be prohibitively high, especially with an increasing number of tasks.

The work introduced in (Li et al., 2019b) bases its continual learning on neural architecture search, whereas the representation in (Javed & White, 2019 ) is optimised such that online updates minimize the error on all samples while limiting forgetting.

The framework proposed by Xu & Zhu (2018) interestingly aims at solving this neural architecture structure learning problem, while balancing the tradeoff between adaptation and stability, via designed reinforcement learning (RL) strategies.

When facing a new task, the optimal number of neurons and filters to add to each layer is cast as a combinatorial optimisation problem solved by an RL strategy whose reward signal is a function of validation accuracy and network complexity.

Another RL based framework is the one presented by Kaplanis et al. (2018) where catastrophic forgetting is mitigated at multiple time scales via RL agents with a synaptic model inspired by neuroscience.

Bottom layers (those near the input) are generally shared among the different tasks, while layers near the output are task-specific.

Since the model structure is usually divided a priori and no automatic architecture learning nor adaptation takes place, alteration on the shared layers can still cause performance loss on earlier tasks due to forgetting (Shin et al., 2017) .

A clipped version of maxout networks (Goodfellow et al., 2013 ) is developed in (Lin et al., 2018) where parameters are partially shared among examples.

The method in (Ostapenko et al., 2019 ) is based a dynamic network expansion accomplished by a generative adversarial network.

The memory-based approach, which is the third influential approach to address the adaptationcatastrophic forgetting tradeoff, relies on episodic memory to store data (or pseudodata) from previous tasks (Ratcliff, 1990; Robins, 1993; Hattori, 2014; Rolnick et al., 2018; Teng & Dasgupta, 2019) .

A major limitation of the memory-based approach is that data from previous tasks may not be available in all real-world problems (Shin et al., 2017; .

Another limitation is the overhead resulting from the memory requirements, e.g. storage, replay, etc.

In addition, the optimisation required to select the best observation to replay for future tasks is a source of further overhead (Titsias et al., 2019) .

In addition to the explicit replay form, some works have been based on generative replay (Thrun, 1996; Schmidhuber, 2013; Mocanu et al., 2016; Rebuffi et al., 2017; Kamra et al., 2017; Shin et al., 2017; van de Ven & Tolias, 2018; Wu et al., 2018) .

Notably, Shin et al. (2017) train a deep generative model based on generative adversarial networks (GANs, Goodfellow et al., 2014b; Goodfellow, 2016) to mimic past data.

This mitigates the aforementioned problem, albeit at the added cost of the training of the generative model and sharing its parameters.

Alleviating catastrophic forgetting via replay mechanisms has also been adopted in reinforcement learning, e.g. (Isele & Cosgun, 2018; Rolnick et al., 2018) .

A similar approach was introduced by Lopez-Paz & Ranzato (2017) where gradients of the previous task (rather than data examples) are stored so that a trust region consisting of gradients of all previous tasks can be formed to reduce forgetting.

Other algorithms based on replay mechanisms include (Aljundi et al., 2019a; .

Equivalent tradeoffs to the one between adaptation and stability can be found in the literature since the work in (Carpenter & Grossberg, 1987) , in which a balance was needed to resolve the stabilityplasticity dilemma, where the latter refers to the ability to rapidly adapt to new tasks.

The works introduced in (Chaudhry et al., 2018; shed light on the tradeoff between adaptation and stability, where they explore measures of intransigence and forgetting.

The former refers to the inability to adapt to new tasks and data, whereas an increase in the latter clearly signifies an instability problem.

Other recent works tackling the same tradeoff include (Riemer et al., 2019) where the transfer-interference (interference is catastrophic forgetting) tradeoff is optimised for the sake of maximising transfer and minimising interference by an algorithm based on experience replay and meta-learning.

Other recent algorithms include the ORACLE algorithm by Yoon et al. (2019) , which addresses the sensitivity of a continual learner to the order of tasks it encounters by establishing an order robust learner that represents the parameters of each task as a sum of task-shared and task-specific parameters.

The algorithm in (Titsias et al., 2019) achieves functional regularisation by performing approximate inference over the function (instead of parameter) space.

They use a Gaussian process obtained by assuming the weights of the last neural network layer to be Gaussian distributed.

Our model is also related to the multi-task learning approach (Caruana, 1997; Heskes, 2000; Bakker & Heskes, 2003; Stickland & Murray, 2019) .

As mentioned in the main document, ideas of the proposed CLAW can be applied to continual learning frameworks other than VCL.

The latter is more relevant for the inference part of CLAW since both are based on variational inference.

As per the modeling ideas, e.g. the binary adaptation parameter depicting whether or not to adapt, and the maximum allowed adaptation, these can be integrated within other continual learning frameworks.

For example, the algorithm in Xu & Zhu (2018) utilises reinforcement learning to adaptively expand the network.

The optimal number of nodes and filters to be added is cast as a combinatorial optimisation problem.

In CLAW, we do not expand the network.

As such, an extension of the work in (Xu & Zhu, 2018) can be inspired by CLAW where not only the number of nodes and filters to be added is decided for each task, but also a soft and more general version where an adaptation based on the same network size is performed such that the network expansion needed in (Xu & Zhu, 2018) can be further moderated.

In this section, we provide information about the statistical significance and standard error of CLAW and the competing continual learning frameworks.

In Table 1 , we list the average accuracy values (Figure 1 in the main document) obtained after completing the last two tasks from each of the six experiments.

A bold entry in Table 1 denotes that the classification accuracy of an algorithm is significantly higher than its competitors.

Significance results are identified using a paired t-test with p = 0.05.

Each average accuracy value is followed by the corresponding standard error.

Average classification accuracy resulting from CLAW is significantly higher than its competitors on the 6 experiments.

99.2 ± 0.2 % 93.

98.7 ± 0.3 % 95.8 ± 0.4 % 96.9 ± 0.5 % 92.9 ± 0.4 % 97.8 ± 0.4 % 97.7 ± 0.2 % 96.1 ± 0.6 % 97.8 ± 0.3 % Split notMNIST (task 5) 98.4 ± 0.2 % 92.1 ± 0.3 % 96.0 ± 0.3 % 92.3 ± 0.4 % 96.9 ± 0.5 % 97.3 ± 0.5 % 95.2 ± 0.7 % 97.4 ± 0.3 % Split Fashion-MNIST (task 4) 93.2 ± 0.2 % 90.0 ± 0.3 % 90.7 ± 0.2 % 89.4 ± 0.4 % 91.4 ± 0.3 % 91.1 ± 0.3 % 90.4 ± 0.2 % 92.5 ± 0.4 % Split Fashion-MNIST (task 5) 92.5 ± 0.2 % 88.0 ± 0.2 % 88.5 ± 0.4 % 87.6 ± 0.3 % 90.8 ± 0.2 % 89.7 ± 0.4 % 87.7 ± 0.4 % 91.1 ± 0.3 % Omniglot (task 49) 84.5 ± 0.2 % 81.1 ± 0.3 % 81.8 ± 0.3 % 78.2 ± 0.3 % 82.8 ± 0.2 % 80.1 ± 0.4 % 79.9 ± 0.3 % 83.6 ± 0.3 % Omniglot (task 50)

84.6 ± 0.3 % 80.7 ± 0.3 % 81.1 ± 0.4 % 77.3 ± 0.3 % 82.7 ± 0.3 % 80.2 ± 0.4 % 79.8 ± 0.5 % 83.5 ± 0.3 % CIFAR-100 (task 19) 95.6 ± 0.3 % 78.7 ± 0.4 % 80.8 ± 0.3 % 63.1 ± 0.5 % 68.3 ± 0.6 % 63.7 ± 0.6 % 77.4 ± 0.7 % 86.2 ± 0.4 % CIFAR-100 (task 20)

95.6 ± 0.3 % 77.2 ± 0.4 % 79.9 ± 0.4 % 62.4 ± 0.4 % 65.5 ± 0.6 % 60.4 ± 0.6 % 76.8 ± 0.6 % 85.6 ± 0.5 %

Here are some additional details about the datasets in use:

The MNIST dataset is used in both the Permuted MNIST and Split MNIST experiments.

The MNIST (Mixed National Institute of Standards and Technology) dataset (LeCun et al., 1998 ) is a handwritten digit dataset.

Each MNIST image consists of 28 × 28 pixels, which is also the pixel size of the notMNIST and Fashion-MNIST datasets.

The MNIST dataset contains a training set of 60,000 instances and a test set of 10,000 instances.

As mentioned in the main document, each experiment is repeated ten times.

Data is randomly split into three partitions, training, validation and test.

A portion of 60% of the data is reserved for training, 20% for validation and 20% for testing.

Statistics reported are the averages of these ten repetitions.

Number of epochs required per task to reach a saturation level for CLAW (and the bulk of the methods in comparison) was 10 epochs for all experiments except for Omniglot and CIFAR-100 (15 epochs).

Used values of ω 1 and ω 2 are 0.05 and 0.02, respectively.

For Omniglot, we used a network similar to the one used in , which consists of 4 blocks of 3 × 3 convolutions with 64 filters, followed by a ReLU and a 2 × 2 max-pooling.

The same CNN is used for CIFAR-100.

CLAW achieves clearly higher classification accuracy on both Omniglot and CIFAR-100 (Figures 1e and 1f ).

The plots displayed in this section empirically demonstrate how important the main adaptation parameters are in achieving the classification performance levels reached by CLAW.

In each of the Figures 4-9 , the classification performance of CLAW is compared to the following three cases: 1) when the parameter controlling the maximum degree of adaptation is not learnt in a multi-task fashion, i.e. when the respective general value s i,j is used instead of s i,j,t .

2) when adaptation always happens, i.e. the binary variable denoting the adaptation decision is always activated.

3) when adaptation never takes place.

The differences in classification accuracy between CLAW and each of the other three plots in Figures 4-9 empirically demonstrate the relevance of each adaptation parameter.

@highlight

A continual learning framework which learns to automatically adapt its architecture based on a proposed variational inference algorithm. 