Meta learning has been making impressive progress for fast model adaptation.

However, limited work has been done on learning fast uncertainty adaption for Bayesian modeling.

In this paper, we propose to achieve the goal by placing meta learning on the space of probability measures, inducing the concept of meta sampling for fast uncertainty adaption.

Specifically, we propose a Bayesian meta sampling framework consisting of two main components: a meta sampler and a sample adapter.

The meta sampler is constructed by adopting a neural-inverse-autoregressive-flow (NIAF) structure, a variant of the recently proposed neural autoregressive flows, to efficiently generate meta samples to be adapted.

The sample adapter moves meta samples to task-specific samples, based on a newly proposed and general Bayesian sampling technique, called optimal-transport Bayesian sampling.

The combination of the two components allows a simple learning procedure for the meta sampler to be developed, which can be efficiently optimized via standard back-propagation.

Extensive experimental results demonstrate the efficiency and effectiveness of the proposed framework, obtaining better sample quality and faster uncertainty adaption compared to related methods.

Meta learning (Schmidhuber, 1987; Andrychowicz et al., 2016) is an important topic in modern machine learning.

The goal is to learn some abstract concepts from different but related tasks, which can then be adapted and generalized to new tasks and environments that have never been encountered during training.

There has been lots of research on this topic.

A recent review classifies the methods as metric-based, model-based and optimization-based methods (Weng, 2018) .

Among these methods, learning-to-learn seeks to learn a meta optimizer that can be applied to different models, with some task-specific information such as current gradients as input (Andrychowicz et al., 2016) .

Model agnostic meta learning (MAML) aims to learn a meta parameter/model from a set of training tasks such that it can quickly adapt to models for new tasks (Finn et al., 2017) .

Many follow-up works have been proposed recently, including but not limited to the meta network (Munkhdalai & Yu, 2017) , the meta learner (Ravi & Larochelle, 2017) , the Reptile model (Nichol et al., 2018) , and the lately extensions to an online setting (Finn et al., 2019) , to model hierarchical relation (Yao et al., 2019) and sequential strategies (Ortega et al., 2019) , and to its stable version Antoniou et al. (2019) and to some theoretical analysis (Khodak et al., 2019) .

It is worth noting that all the aforementioned models are designed from an optimization perspective.

Bayesian modeling, in parallel with optimization, has also been gaining increasing attention and found various applications in deep learning.

Recent research has extended the above meta-learning methods to a Bayesian setting.

For example, Bayesian MAML (BMAML) replaces the stochasticgradient-descent (SGD) step with Stein variational gradient descent (SVGD) for posterior sampling (Yoon et al., 2018) .

Probabilistic MAML (PMAML) extends standard MAML by incorporating a parameter distribution of the adapted model trained via a variational lower bound (Finn et al., 2018) .

Amortized Bayesian Meta Learning extends the idea of MAML to amortized variational inference (Ravi & Beatson, 2019; Choi et al., 2019) .

VERSA (Gordon et al., 2019) uses an amortization network to approximate the posterior predictive distributions.

Meta particle flow realizes Bayes's rule based on ODE neural operator that can be trained in a meta-learning framework.

Though methodologically elegant with many interesting applications, the above methods lack the ability to uncertainty propagation/adaption, in the sense that uncertainty is either not considered (e.g., in MAML) or only considered in the specific task level (e.g., BMAML).

This could slow down model adaption or even inaccurate uncertainty modeling when considering from a Bayesian modeling perspective.

For example, suppose one is given samples from a set of Gaussians with different mean and covariance matrices, how can she/he efficiently leverage uncertainty in these samples to generate samples from a complex yet related distribution such as a Gaussian mixture?

To tackle this problem, we propose to perform meta learning on the space of probability measures, i.e., instead of adapting parameters to a new task, one adapts a meta distribution to new tasks.

When implementing distribution adaption in algorithms where distributions are approximated by samples, our distribution-adaptation framework becomes sample-to-sample adaption.

In other words, the meta parameter in standard MAML becomes meta samples in our method, where uncertainty can be well encoded.

For this reason, we call our framework Bayesian meta sampling.

Specifically, we propose a mathematically elegant framework for Bayesian meta sampling based on the theory of Wasserstein gradient flows (WGF) (Ambrosio et al., 2005) .

Our goal is to learn a meta sampler whose samples can be fast adapted to new tasks.

Our framework contains two main components: a meta sampler and a sample adapter.

For the meta sampler, we adopt the state-ofthe-art flow-based method to learn to transport noise samples to meta samples.

Our meta sampler is parameterized by a neural inverse-autoregressive flow (NIAF), an extension of the recently developed neural autoregressive flows (NAFs) (Huang et al., 2018) .

The NIAF consists of a meta-sample generator and an autoregressive conditioner model, which outputs the parameters of the meta-sample generator.

The NIAF takes some task-specific information (such as gradients of target distributions) and random noise as input and outputs meta samples from its generator.

These meta samples are then quickly adapted to task-specific samples of target distributions by feeding them to the sample adapter.

To ensure efficient and accurate adaptations to new task distributions, a novel optimal-transport Bayesian sampling (OP-sampling) scheme, based on Wasserstein gradient flows, is proposed as the adaptation mechanism of the sample adapter.

The OP-sampling is general and can ensure samples to be adapted in a way that makes the sample density evolve to a target distribution optimally, thus endowing the property of fast uncertainty adaption.

Finally, when one aims to perform specific tasks such as Bayesian classification with a task network, these samples are used to encode uncertainty into modeling.

To this end, we further develop an efficient learning algorithm to optimize the task network based on variational inference.

Extensive experiments are conducted to test the advantages of the proposed meta-sampling framework, ranging from synthetic-distribution to posterior-distribution adaption and to k-shot learning in Bayesian neural networks and reinforcement learning.

Our results demonstrate a better performance of the proposed model compared to related methods.

Our model combines ideas from Bayesian sampling, Wasserstein gradient flows and inverseautoregressive flows.

A detailed review of these techniques is provided in Section A of the Appendix.

Overall idea of the proposed meta-sampling framework.

The task is denoted with ?? .

The two components and specific inputs will be described in details.

In meta sampling, one is given a set of related distributions, e.g., posterior distributions of the weights of a set of Bayesian neural networks (BNNs), each of which is used for classification on a different but related dataset.

With our notation, each of the network and the related dataset is called a task, which is denoted as ?? .

Meta sampling aims to learn a meta sampler based on a set of training tasks so that samples from the meta sampler can be fast adapted to samples for an unseen new task.

Our overall idea of the proposed Bayesian meta sampling is to mimic a hierarchical-sampling procedure but in a much more efficient way.

Specifically, we propose to decompose meta sampling into two components: a meta sampler and a sample adapter.

The meta sampler is responsible for generating meta samples that characterize common statistics of different tasks; The sample adapter is designed for fast adaptation of meta samples to task-specific target distributions.

The meta sampler is parameterized as a conditional generator, and the sample adapter aggregates all local losses of different tasks to form a final loss for optimization based on optimal-transport theory.

Our method allows gradients to be directly backpropagated for meta-sampler updates.

The overall idea is illustrated in Figure 1 .

Comparisons with related works We distinguish our model with two mostly related works: the meta NNSGHMC (Gong et al., 2019) and the probabilistic MAML (PMAML) (Finn et al., 2018) .

The main differences lie in two aspects: meta representation and model architecture.

In terms of meta-model representation, our model adopts data/parameter samples, instead of determinstic parameters, as meta representation, and thus can be considered as a sample-to-sample adaption.

Meta NNSGHMC uses samples on different tasks, there is no concept of meta samples.

Finally, PMAML fully relies on variational inference whose representation power could be restricted.

In terms of model architecture, our model adopts the state-of-the-art autoregressive architectures, which can generate high-quality meta samples.

Furthermore, our model adopts a simpler way to define the objective function, which allows gradients to directly flow back for meta-sampler optimization.

It is worth noting that our framework reduces to MAML when only one meta sample is considered for sample adaption.

Since our methods aim for Bayesian sampling, it is more similar to meta NNSGHMC (Gong et al., 2019) ; whereas MAML aims for point estimation of a model.

Finally, we note that the recently proposed neural process (Garnelo et al., 2018) might also be used for meta-learning "few-shot function regression" as stated in the original paper.

However, to our knowledge, no specific work has been done for this purpose.

Task network 5(???; 9) This section aims to design a meta sampler for efficiently generating meta samples.

One idea is to use a nonparametric model to generate meta samples such as standard stochastic gradient MCMC (SG-MCMC) (Welling & Teh, 2011; Ma et al., 2015) and the Stein variational gradient descent (SVGD) (Liu & Wang, 2016) , where no parameters are consider in the model.

However, methods under this setting are typically slow, and the generated samples are usually highly correlated.

Most importantly, it would be hard to design nonparametric samplers that can share information between different tasks.

As a result, we propose to learn the meta sampler with a parametric model, which is also denoted as a generator.

There are two popular options to parameterize the meta sampler: with an explicit generator or with an implicit generator.

An explicit generator parameterizes the output (i.e., meta samples) as samples from an explicit distribution with a known density form such as Gaussian, thus limiting the representation power.

In the following, we propose to adopt an implicit generator for the meta sampler based on neural inverse-autoregressive flows (NIAF), an inverse extension of the recently proposed NAF (Huang et al., 2018 ) used for density estimation.

As will be seen, NIAF can incorporate task-specific information into an implicit generator and generates samples in an autoregressive manner efficiently.

Finally, meta samples are used in a task network to encode uncertainty for specific tasks such as Bayesian classification.

The architecture of the meta sampler is illustrated in Figure 2 .

Note the idea of using NIAF to generate network parameter is similar to the hypernetwork (Ha et al., 2016; Krueger et al., 2017) .

As an extention, Pradier et al. (2018) perform inference on a lower dimensional latent space.

Using hypernetworks to model posterior distributions have also been studied in (Pawlowski et al., 2017; Sheikh et al., 2017) .

Neural inverse-autoregressive flows Directly adopting the NAF for sample generation is inappropriate as it was originally designed for density evaluation.

To this end, we propose the NIAF for effective meta-sample generation.

Specifically, let z k denote the k-th element of a sample z to be generated;z denotes a sample from last round; ?? denotes the task-specific information.

In our case, we set ?? (z, ???z log p(z)) with p(??) denoting the target distribution (with possible hidden parameters).

NIAF generates the sample z = (z 1 , ?? ?? ?? , z k , ?? ?? ?? ) via an autoregressive manner, as:

where { k } are noise samples; G(??, ??; ?? ?? ??) is an invertible function (generator) parameterized by ?? ?? ?? and implemented as a DNN; and T is an autoregressive conditioner model to generate the parameters of the generator G at each step k, which is itself parameterized by ?? ?? ?? and implemented as a deep sigmoidal flow or a deep dense sigmoidal flow as in (Huang et al., 2018) .

According to Huang et al. (2018) , using strictly positive weights and strictly monotonic activation functions for G is sufficient for the entire network to be strictly monotonic, thus invertible.

The original NAFs are not designed for drawing samples, as one needs the inverse function G ???1 , which is not analytically solvable when G is implemented as a neural network.

Although it is stated in (Huang et al., 2018) that G ???1 can be approximated numerically, one needs repeated approximations, making it computationally prohibited.

Our proposed NIAF is designed specifically for sample generation by directly transforming the noise with a flow-based network G.

The task network In addition to the NIAF, a task network might be necessary for processing specific learning tasks.

In particular, if one is only interested in generating samples from some target distribution, a task network is not necessary as the meta samples will be used to adapt to task-specific samples.

However, if one wants to do classification with uncertainty, the task network should be defined as a classification network such as an MLP or CNN.

In this case, denoting the weights of the task network as W, we consider the task network as a Bayesian neural network, and propose two ways of parameterization to encode uncertainty of meta samples into the task network:

??? Sample parameterization: A sample of the weights of the task network is directly represented by a meta sample from our meta sampler, i.e., W = (z 1 , z 2 , ?? ?? ?? , z p ) with p the parameter dimensionality.

??? Multiplicative parameterization:

Adopting the idea of multiplicative normalizing flows (Louizos & Welling, 2017) , we define an inference network for the weights as the multiplication of z and a Gaussian variational distribution for W, i.e., the variational distribution is defined as the following semi-implicit distribution to approximate the true posterior distribution for W:

Here and in the following, we consider the task network parameterized as a one-layer MLP for notation simplicity, although our method applies to all other network structures; and we have used NIAF({ k }, ??) to denote the output of meta samples from the NIAF.

Comparing the two parameterizations, sample parameterization directly generates weights of the task network from the meta sampler, thus is more flexible in uncertainty modeling.

However, when the task network grows larger to deal with more complex data, this way of parameterization quickly becomes unstable or even intractable due to the high dimentionality of meta samples.

Multiplicative parameterization overcomes this issue by associating each element of a meta sample with one node of the task network, reducing the meta-sample dimensionality from O(N in ??N out ) to O(N in +N out ) with N in and N out the input and output sizes of the task network.

As a result, we adopt the multiplicative parameterization when dealing with large-scale problems in our experiments.

Efficient inference for these two cases will be described in Section 2.4.

Note a recent work on NAF inference proposes to first sample from a mean-field approximating distribution, which are then transformed by an NAF to a more expressive distribution (Webb et al., 2019) .

However, the approach is hard to scale to very high dimensional problems, e.g., posterior distributions of network parameters.

The output of the meta sampler contains shared (meta) information of all the tasks.

Task-specific samples are expected to be adapted fast from these meta samples.

This procedure is called sample adaption.

Since there are potentially a large number of tasks, learning task-wise parametric models for sample adaption is impractical.

Instead of using standard nonparametric samplers such as SG-MCMC or SVGD, we propose a general Bayesian sampling framework based on optimal-transport theory (Villani, 2008) for new task-sample adaption, where back-propagation can be directly applied.

A general Bayesian sampling framework based on optimal transport Let a task-specific target distribution be p ?? (z), indexed by ?? .

A standard way is to adapt the samples based on a Markov chain whose stationary distribution equals p ?? , e.g., via SG-MCMC.

However, Markov-chain-based methods might not be efficient enough in practice due to the potentially highly-correlated samples (Chen et al., 2018b) .

Furthermore, it is not obvious how to apply backpropagation (BP) in most of sampling algorithms.

To deal with these problems, we follow Chen et al. (2018b) and view Bayesian sampling from the Wasserstein-gradient-flow perspective (discussed in Section A.2), i.e., instead of evolving samples, we explicitly evolve the underlying sample density functions.

We will see that such a solution allows us to train the proposed meta sampler efficiently via standard BP.

Considering our meta-learning setting.

Since we aim to adapt meta samples to new tasks, it is reasonable to define the adaptation via task-wise WGFs, i.e., for each task, there is a WGF with a specific functional energy and the corresponding first variation, denoted respectively as E ?? and F ?? ??E?? ???? with the task index ?? .

Here ?? denotes the underlying density of the samples.

Consequently, ?? will evolve with a variant of the PDE by replacing E with E ?? in equation 8 for each task.

To solve the corresponding PDE, we prove Theorem 1 based on a discrete approximation of ?? with the evolved meta samples, which is termed optimal-transport Bayesian sampling (OT-Bayesian sampling).

Theorem 1 (Optimal-Transport Bayesian Sampling) Let ?? t at time t be approximated by parti-

.

This is a useful result to derive a learning algorithm for the meta sampler described in Section 2.4.

Energy functional design Choosing an appropriate energy function E ?? is important for efficient sample adaptation.

To achieve this, the following conditions should be satisfied: i) E ?? (??) should be convex w.r.t.

??; ii) The first variation F ?? could be calculated conveniently.

A general and convenient functional family is the f -divergence, which is defined, with our notation and a convex function f : R ??? R such that f (1) = 0, as:

The f -divergence is a general family of divergence metric.

With different functions f , it corresponds to different divergences including the popular KL divergence, inverse-KL divergences, and the Jensen-Shannon divergence.

For more details, please refer to (Nowozin et al., 2016) .

A nice property of f -divergence is that its first variation endows a convenient form as stated in Proposition 2.

* We use the bold letter z

to denote the i-th meta sample evolved with equation 3 at time t (or equation 4 at iteration k).

This should be distinguished from the normal unbold letter z k defined in Section 2.2.1, which denotes the k-th element of z.

?? (z) p?? (z) .

The first variation of the f -divergence endows the following form:

In our experiments, we focus on the KL-divergence, which corresponds to f (r) = r log r.

In this case,

Since the density ??(z) required in evaluating r is not readily available due to its implicit distribution, we follow Chen et al. (2018b) and use the meta samples {z (i) k } at the k-th step for approximation, resulting in

where ??(??, ??) is a kernel function.

The number of adaptation steps k should be set based on problems.

For tasks that vary significantly, a larger k should be chosen to ensure the quality of adapted samples.

To further improve the accuracy, inspired by Chen et al. (2018b) , we combine equation 5 with the first variation of SVGD, resulting in the following form at iteration k:

where ?? ??? 0 is a hyperparameter to balance the two terms.

We first describe how to train the proposed model under the two kinds of parameterization of the task network defined in Section 2.2.1.

Training in the sample-parameterization setting In this case, one only needs to optimize the conditional model T (??; ?? ?? ??) as all parameters of other networks are directly generated.

Specifically, because the energy functionals for each task ?? are designed so that the minima correspond to the target distributions, the objective thus can be defined over the whole task distribution p(?? ) as:

For notation simplicity, we will not distinguish among the adapted samples (i.e., z (i) k ) for different tasks.

Since the only parameter is ?? ?? ?? in the autoregressive conditioner model T (see Figure 2 , and note the parameters ?? ?? ?? and W for the meta generator and task network do not need to be learned as they are the outputs of T and G, respectively), its gradient can be directly calculated using chain rule:

where "

= " follows by the results from Section 2.3 and ??? z Training in the multiplicative-parameterization setting In this case, two sets of parameters are to be learned, ?? ?? ?? and W.

Since ?? ?? ?? and W are decoupled, one can still optimize ?? ?? ?? by adopting the same update equation as in the sample-parameterization setting.

For W, we follow Louizos & Welling (2017) and adopt variational inference.

Specifically, we first augment the task network with an auxiliary network with a conditional likelihood Ranganath et al. (2016) ; Louizos & Welling (2017) , with the inference network defined in (2), and writing the implicit distribution of z asq ?? ?? ?? (z) and the prior distribution of W as p(W), we arrive at the following ELBO: Different from Louizos & Welling (2017) , we only update (??, ?? ?? ??) by optimizing the above ELBO; while leave the update of ?? ?? ?? with the gradient calculated in equation 7, which reflects gradients of samples from the NIAF.

Note also that in the meta learning setting, the task network needs to be adapted for new tasks.

This can be done by standard MAML with the above ELBO as the new-task objective.

The whole algorithm is illustrated in Algorithm 1 in the Appendix.

A similar ideas of multiplicative parametrization was proposed recently in (Kristiadi & Fischer, 2019) , which used a compound density network to quantify the predictive uncertainty.

New-task sample generation After training, samples for a new task can be directly generated by feeding the task information ?? and some noise to the meta sampler depicted in Figure 1 .

Typically, one needs to run a few numbers of sample-adaption steps to generate good samples for new tasks.

Notably, the number of sample-adaption steps required to obtain good accuracy will be shown much less than simply starting a sampler from scratch in the experiments.

We conduct a series of experiments to evaluate the efficiency and effectiveness of our model, and compare it with related Bayesian sampling algorithms such as SGLD, SVGD and SGHMC.

The main compared algorithms for meta learning include the PMAML (Finn et al., 2018) , Amortized Bayesian Meta-Learning (ABML) (Ravi & Beatson, 2019) , and NNSGHMC (Gong et al., 2019) , a recently proposed meta SG-MCMC algorithm.

Inspired by Finn et al. (2017) , we denote our algorithm distribution agnostic meta sampling (DAMS).

We first demonstrate our proposed NIAF-based sampler is able to generate more effective samples compared to the popular Bayesian algorithms such as SVGD, SGLD and SGHMC, in a non-meta-sampling setting.

To this end, we apply standard Bayesian Logistic Regression (BLR) on several real datasets from the UCI repository: Australian (15 features, 690 samples), German (25 features, 1000 samples), Heart (14 features, 270 samples).

We perform posterior sampling for BLR using our proposed sampler, as well as SVGD, SGLD, SGHMC.

For a more detailed investigation of different components in our model, we also test the generator with different architectures, including generators with MLP (DAMS with MLP), IAF (DAMS with IAF), and NIAF (DAMS with NIAF).

We follow Liu & Wang (2016) and apply Gaussian priors for the parameters p 0 (w|??) = N (w; 0, ?? ???1 I) with p 0 (??) = Gamma(??, 1, 0.01).

A random selection of 80% data are used for training and the remaining for testing.

The testing accuracies are shown in Table 1 .

It is observed that DAMS with NIAF achieves the best performance in terms of accuracy.

The results also indicate the effectiveness and expressiveness of the proposed NIAF architecture in the OT-Bayesian sampling framework.

In this set of experiments, we aim to demonstrate the excellent meta-sample adaptability of our meta-sampling framework in different tasks.

An additional synthetic experiment on meta posterior adaption is presented in Section D.3 of the Appendix.

Gaussian mixture model We first conduct experiments to meta-sample several challenging Gaussian mixture distributions.

We consider mixtures of 4, 6 and 20 Gaussians.

Detailed distributional forms are given in the Appendix.

To setup a meta sampling scenario, we use 2, 3 and 14 Gaussian components with different means and covariance, respectively, for meta-training of the meta sampler.

After training, meta samples are adapted to samples from a target Gaussian mixture by following the new-task-sample-generation procedure described in Section 2.4.

We plot the convergence of 1000 meta samples to a target distribution versus a number of particle (sample) updates (iterations), measured with the maximum mean discrepancy (MMD) evaluated by samples.

For a fair comparison, we use the same number of samples (particles) to evaluation the MMD.

The results are shown in Figure 3 .

It is clear that our proposed meta sampler DAMS converges much faster and better than other sampling algorithms, especially on the most complicated mixture of 20-Gaussians.

The reason for the fast convergence (adaption) is partially due to the learned meta sampler, which provides good initialization for sample adaption.

This is further verified by inspecting how samples in the adaption process evolve, which are plotted in Figure 10 , 11 and 12 in the Appendix.

Finally, we test the proposed DAMS for meta sampling of BNNs on MNIST and CIFAR-10 (Krizhevsky, 2009).

We follow the experimental setting in (Gong et al., 2019) , and split the MNIST and CIFAR10 dataset into two parts for meta training and testing (sample adaption), respectively.

As we focus on fast adaptation, we show the accuracy within 200 iterations.

To deal with the high-dimensionality issue, we adopt the method of multiplicative parameterization proposed in Section 2.2.1.

We randomly pick 5 classes for training, and the remaining classes for testing.

A BNN is trained only on the training data for meta sample (weights of the BNN) generation, with each sample corresponding to a meta BNN.

In testing, the meta BNNs are adapted based on the testing data.

For the MNIST dataset, we parameterize a BNN as a CNN with two convolutional layers followed by a fully connected layer with 100 hidden units.

The kernel sizes of the two conv layers are 3 and 16, respectively.

A similar architecture is applied for the CIFAR10 dataset, but with 16 and 50 filters whose kernel sizes are 7 and 5 for the two convolutional layers, respectively.

The hidden units of the fully connected layer is 300.

a) Adaptation efficiency: For this purpose, we compare our model with NNSGHMC (Gong et al., 2019) , as well as with a non-meta-learning method to train from scratch.

To demonstrate the effectiveness of our NIAF structure for adaptive posterior sampling, we also compare it with the simple conditional version of MNF (Louizos & Welling, 2017) .

For NNSGHMC and our DAMS, 20 meta samples are used in training.

Figure 4 plots the learning curves of testing accuracy versus the number of iterations.

It is clearly seen that our DAMS adapts the fastest to new tasks, and is able to achieve the highest classification accuracy on all cases due to the effectiveness of uncertainty adaption.

To further demonstrate the superiority of DAMS over NNSGHMC, we list the test accuracy at different adaptation steps in Table 2 .

The results clearly show faster adaption and higher accuracy of the proposed DAMS compared to NNSGHMC.

It is also interesting to see that MNF, the non-meta learning method, performs better than NNSGHMC; while our method outperforms both, demonstrating the effectiveness of the proposed NIAF architecture.

b) Sample efficiency: To demonstrate sample efficiency of our framework, we compare it with both NNSGHMC and the standard Bayesian learning of DNNs with SGHMC.

To this end, we randomly select 5%, 20%, 30% of training data on CIFAR10 in a test task as training data for adaptation, and test on the same testing data.

Figure 5 shows the corresponding test accuracies for different settings.

It is observed that ours, the adaptation-based methods, obtain higher accuracies than the non-adaptive method of SGHMC.

Furthermore, our method achieves the best sample efficiency among other methods.

c) Uncertainty evaluation: Finally, we ablate study the uncertainty estimation of our, the standard SGHMC and NNSGHMC models in terms of test accuracy and negative loglikelihood.

The results on the CIFAR10 dataset are shown in Figure 6 and Table 3.

We follow Louizos & Welling (2017) to evaluate uncertainty via entropy of out-of-sample predictive distributions (Figure 6 (right) ).

We observe that uncertainty estimates with DAMS are better than others, since the probability of low entropy prediction is much lower than others.

Details are given in Section D.4 of the Appendix.

Following literature (Finn et al., 2017; 2018) , we further apply our framework for meta sampling of few-shot classification and reinforcement learning.

Model agnostic meta sampling for few-shot image classification We apply our method on two popular few-shot image-classification tasks on the Mini-Imagenet dataset, consisting of 64, 16, and 20 classes for training, validation and testing, respectively.

We compare our method with MAML and its variants with uncertainty modeling, including the Amortized Bayesian Meta-Learning (ABML) (Ravi & Beatson, 2019) and Probabilistic MAML (PMAML) (Finn et al., 2018) .

To get a better understanding of each component of our framework, we also conduct an ablation study with three variants of our model: MAML-SGLD, MAML-SGHMC and DAMS-SGLD.

MAML-SGLD and MAML-SGHMC correspond to the variants where SGLD and SGHMC are used to sample the parameters of the classifier, respectively; and DAMS-SGLD replaces the WGF component of DAMS-NIAF with SGLD.

Follow the setting in (Finn et al., 2017; 2018) , the network architecture includes a stacked 4-layer convolutional feature extractor, followed by a meta classifier with one single fully-connected layer using the multiplicative parameterization.

Testing results are presented in Table 4 .

With our method, we observe significant improvement of the classification accuracy at an early stage compared with MAML.

The learning curves are plotted in Figure 7 , further demonstrating the superiority of our method, which can provide an elegant initialization for the classification network.

Finally, from the ablation study, the results suggest both the NIAF and the WGF components contribute to the performance gain obtained by our method.

Meta sampling for reinforcement learning We next adapt our method for meta reinforcement learning.

We test and compare the models on the same MuJoCo continuous control tasks (Todorov et al., 2012) as used in (Finn et al., 2017) , including the goal velocity task and goal direction task for cheetah robots.

For a fair comparison, we leverage the TRPO-RL (Schulman et al., 2015) framework for meta updating following MAML method.

Specifically, we implement the policy network with two hidden layers with ReLu activation followed by a linear layer to produce the mean value of the Gaussian policy.

The first hidden layer is a fully connected layer, and we adopt the multiplicative parameterization for the second hidden layer.

As shown in Figure 8 , our method obtains higher rewards compared with MAML on both tasks, indicating the importance of effective uncertainty adaptation in RL.

We present a Bayesian meta-sampling framework, called DAMS, consisting of a meta sampler and a sample adapter for effective uncertainty adaption.

Our model is based on the recently proposed neural autoregressive flows and related theory from optimal transport, enabling a simple yet effective training procedure.

To make the proposed model scalable, an efficient uncertainty parameterization is proposed for the task network, which is trained by variational inference.

DAMS is general and can be applied to different scenarios with an ability for fast uncertainty adaptation.

Experiments on a series of tasks demonstrate the advantages of the proposed framework over other methods including the recently proposed meta SG-MCMC, in terms of both sample efficiency and fast uncertainty adaption.

This section provides a review of background on Bayesian sampling, Wasserstein gradient flows and autoregressive flows.

Bayesian sampling has been a long-standing tool in Bayesian modeling, with a wide range of applications such as uncertainty modeling (Li et al., 2015) , data generation (Feng et al., 2017; Chen et al., 2018a) and reinforcement learning (Zhang et al., 2018) .

Traditional algorithms include but are not limited to Metropolis-Hastings algorithm, importance sampling and Gibbs sampling (Gelman et al., 2004) .

Modern machine learning and deep learning have been pushing forward the development of large-scale Bayesian sampling.

Popular algorithms in this line of research include the family of stochastic gradient MCMC (SG-MCMC) (Welling & Teh, 2011; Ma et al., 2015) and the Stein variational gradient descent (SVGD) (Liu & Wang, 2016) .

Recently, a particle-optimization sampling framework that unifies SG-MCMC and SVGD has also been proposed (Chen et al., 2018b) , followed by some recent developments (Liu et al., 2019b; a) .

Generally speaking, all these methods target at sampling from some particular distributions such as the posterior distribution of the weights of a Bayesian neural network (BNN).

On the other hand, meta learning is a recently developed concept that tries to learn some abstract information from a set of different but related tasks.

A natural question by considering these two is: can we design meta sampling algorithms that learns to generate meta samples, which can be adapted to samples of a new task-specific distribution quickly?

This paper bridges this gap by proposing a mathematically sound framework for Bayesian meta sampling.

In optimal transport, a density function, ?? t , evolves along time t to a target distribution optimally, i.e., along the shortest path on the space of probability measures P(???), with ??? being a subset of R d .

The optimality is measured in the sense that ?? t moves along the geodesic of a Riemannian manifold induced by a functional energy, E : P(???) ??? R, under the 2-Wasserstein distance metric.

Formally, the trajectory of ?? t is described by the following partial differential equation (PDE):

where

??? zi f is the divergence operator; and F ??E ????t (?? t ) is called the first variation of E at ?? t (functional derivative on a manifold in P(???)).

To ensure ?? t to converge to a target distribution p such as the posterior distribution of the model parameters, one must design an appropriate E such that p = arg ?? min E(??).

A common choice is the popular KL-divergence, KL(??, p).

We will consider a more general setting in our framework presented later.

Note the WGF framework equation 8 allows to view Bayesian sampling from a density-optimization perspective.

For example, recent works (Chen et al., 2018b; Liu et al., 2019b; a) consider approximating ?? t with samples and evolve the samples according to equation 8.

Parts of our model will follow this sampling setting.

Our model relies on the concept of autoregressive flows for meta-sampler design.

We review some key concepts here.

More detailed comparisons are provided in the Appendix.

A normalizing flow defines an invertible transformation from one random variable to another z. A flexible way to implement this is to define it via implicit distributions, meaning sample generation is implemented as: i ??? q 0 ( i ), z i = G( i ; ?? ?? ??), where i indexes elements of and z; G represents a deep neural network (generator) parameterized by ?? ?? ??.

The autoregressive flow (AF) parameterizes a Gaussian conditional distribution for each z i , e.g., p(z i | z 1:i???1 ) = N (z i |?? i , exp(?? i )), where ?? i = g ??i (z 1:i???1 ) and ?? i = g ??i (z 1:i???1 ) are outputs of two neural networks g ??i and g ??i .

The sample generation process is: z i = i exp(?? i ) + ?? i , with ?? i = g ??i (z 1:i???1 ), ?? i = g ??i (z 1:i???1 ) and i ??? N (0, 1).

Instances of autoregressive flows include the Autoregressive Flow (AF) (Chen et al., 2017) and Masked Autoregressive Flow (MAF) (Papamakarios et al., 2017) .

The inverse autoregressive flow (IAF) (Kingma et al., 2016) is an instance of normalizing flow that uses MADE (Germain et al., 2015) , whose samples are generated as:

The neural autoregressive flow (NAF) (Huang et al., 2018) replaces the affine transformation used in the above flows with a deep neural network (DNN), i.e., t = f (z t , ?? ?? ?? = T (z 1:t???1 )), where f is a DNN transforming a complex sample distribution, p(z), to a simple latent representation q 0 ( ).

In NAF, q 0 is considered as a simple prior, and f is an invertible function represented by a DNN, whose weights are generated by T , an autoregressive conditional model.

Let the induced distribution of by f be p f ( ).

f is learned by minimizing the KL-divergence between p f ( ) and q 0 ( ).

Note the ?? i and ?? i are computed differently for AF and IAF, i.e., previous variables z 1:i???1 are used for AF and previous random noise 1:i???1 are used for IAF.

AF can be used for calculating the density p(z) of any sample z in one pass of the network.

However, drawing samples requires performing D sequential passes (D is the dimensionality of z).

Thus if D is large, drawing samples will be computationally prohibited.

IAF, by contrast, can draw samples and estimate densities of the generated samples with only one pass of the network.

However, calculating the sample density p(z) requires D passes to find the corresponding noise .

The advantage of NAF is that the mapping function f is much more expressive, and density evaluation is efficient.

However, drawing samples is much more computationally expensive.

To adopt the NAF framework for sampling, we propose the neural inverse-autoregressive flow (NIAF) in Section 2.2.1.

Our algorithm in the multiplicative-parameterization setting includes updating the flow parameter and learning the task network with variational inference, which is described in Algorithm 1.

Require: p(T ): distribution over tasks; Require: ??, ?? step size hyperparameter; randomly initialize ??, ?? ?? ??, ?? ?? ??.

while not done do Sample batch of tasks

Compute adapted parameters with gradient descent:

Proof For each task ?? , we first write out the corresponding WGF as

Note the WGF is defined in the sense of distributions Ambrosio et al. (2005) , meaning that for any smooth real functions u(z) with compact support, equation 9 indicates

Taking

(z), and for each particle letting u(z) = z, equation 10 then reduces to the following differential equation for each particle:

which is the particle evolution equation we need to solve.

Proof [Proof of Proposition 2]

First, we introduce the following result from Ambrosio et al. (2005) [page 120]:

In the f-divergence case, this corresponds tof (??) = p ?? f ( ?? p?? ).

Applying Lemma 3 and using the chain rule, we have

which completes the proof.

This section provides extra experimental results to demonstrate the effectiveness of our proposed method.

Analytic forms of synthetic distributions are provided below.

Mog4:

Mog6: We apply our DAMS for fast adaptive sampling on regression tasks.

We follow Huang et al. (2018) and apply DAMS to sample the posterior distribution of the frequency parameter of a sine-wave function, given only three data points.

The sinewave function is defined as y(t) = sin(2??f t), with a uniform prior U ([0, 2]) and a Gaussian likelihood N (y i ; y f (t i ), 0.125).

We design a meta-sampling setting to adapt the posterior distributions, p(f |D) on the training data, to that on new test data.

Specifically, meta training data (t, y) are {(0, 0), (2/5, 0), (4/5, 0)}.

For the first setting, meta testing consists of data {(0, 0), (3/5, 0), (6/5, 0)}.

For the second setting, meta testing consists of data {(0, 0), (4/5, 0), (8/5, 0)}. Meta training data corresponds to a posterior with two modes of f ??? {0.0, 5/4}. For the first setting, the test data corresponds to a posterior with three modes f ??? {0.0, 5/6, 5/3}. For the second setting, the test data corresponds to four modes f ??? {0.0, 5/8, 5/4, 15/8}. We compare our DAMS with the result of re-training from scratch with the test data.

Empirical distribution with samples and kernel density estimation are plotted in Figure 13 .

The first setting takes about 3.4K iterations to find the three modes in the posterior with re-training, while it takes about 0.8K iterations with meta adaptation.

For the second setting, it takes Under review as a conference paper at ICLR 2020 Figure 11 : Comparison among different samplers on adapting to Mixture of 20-Gaussian.

Top to Bottom row: DAMS, SVGD, SGLD and NNSGHMC more than 3.6K iterations to find the four modes with training from scratch, while it is about 0.9K iterations with meta adaptation.

For both test tasks, the sampler with re-training miss at least one mode compared with the meta sampler adaptation with the same number of iterations.

We can see that DAMS can adapt the training posterior to the test posterior much faster than re-training from scratch due to effective uncertainty adaption, obtaining more than 3X speedups.

We show the predictive uncertainty of DAMS compared to SGHMC and NNSGHMC by exploring the posterior of neural parameters, we estimate the uncertainty for out-of-distribution data samples Lakshminarayanan et al. (2017) .

We train different algorithms on the MNIST dataset, and estimate the entropy of the predictive distribution on the notMNIST dataset Bulatov (2011).

We follow Louizos & Welling (2017) and use the empirical CDF of entropy to evaluate the uncertainty.

Since the probability of observing a high confidence prediction is low, curves that are nearer to the bottom right of the figure estimates uncertainty better.

The predictive distribution of the trained model is expected to be uniform over the notMNIST digits as the samples from the dataset are from unseen classes.

The BNN is a CNN with 16 and 50 filters whose kernel sizes are 5 and 5 for the two convolutional layers,

@highlight

We proposed a Bayesian meta sampling method for adapting the model uncertainty in meta learning