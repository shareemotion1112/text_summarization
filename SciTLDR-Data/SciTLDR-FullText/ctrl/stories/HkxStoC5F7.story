This paper introduces a new framework for data efficient and versatile learning.

Specifically: 1) We develop ML-PIP, a general framework for Meta-Learning approximate Probabilistic Inference for Prediction.

ML-PIP extends existing probabilistic interpretations of meta-learning to cover a broad class of methods.

2) We introduce \Versa{}, an instance of the framework employing a flexible and versatile amortization network that takes few-shot learning datasets as inputs, with arbitrary numbers of shots, and outputs a distribution over task-specific parameters in a single forward pass.

\Versa{} substitutes optimization at test time with forward passes through inference networks, amortizing the cost of inference and relieving the need for second derivatives during training.

3) We evaluate \Versa{} on benchmark datasets where the method sets new state-of-the-art results, and can handle arbitrary number of shots, and for classification, arbitrary numbers of classes at train and test time.

The power of the approach is then demonstrated through a challenging few-shot ShapeNet view reconstruction task.

Many applications require predictions to be made on myriad small, but related datasets.

In such cases, it is natural to desire learners that can rapidly adapt to new datasets at test time.

These applications have given rise to vast interest in few-shot learning BID9 BID32 , which emphasizes data efficiency via information sharing across related tasks.

Despite recent advances, notably in meta-learning based approaches BID44 BID54 BID8 BID10 , there remains a lack of general purpose methods for flexible, data-efficient learning.

Due to the ubiquity of recent work, a unifying view is needed to understand and improve these methods.

Existing frameworks BID16 are limited to specific families of approaches.

In this paper we develop a framework for meta-learning approximate probabilistic inference for prediction (ML-PIP), providing this view in terms of amortizing posterior predictive distributions.

In Section 4, we show that ML-PIP re-frames and extends existing point-estimate probabilistic interpretations of meta-learning BID16 to cover a broader class of methods, including gradient based meta-learning BID10 BID44 , metric based meta-learning BID48 , amortized MAP inference BID43 and conditional probability modelling BID13 .The framework incorporates three key elements.

First, we leverage shared statistical structure between tasks via hierarchical probabilistic models developed for multi-task and transfer learning BID18 BID0 .

Second, we share information between tasks about how to learn and perform inference using meta-learning BID37 BID50 BID47 .

Since uncertainty is rife in small datasets, we provide a procedure for metalearning probabilistic inference.

Third, we enable fast learning that can flexibly handle a wide range of tasks and learning settings via amortization .Building on the framework, we propose a new method -VERSA -which substitutes optimization procedures at test time with forward passes through inference networks.

This amortizes the cost of inference, resulting in faster test-time performance, and relieves the need for second derivatives during training.

VERSA employs a flexible amortization network that takes few-shot learning datasets, and outputs a distribution over task-specific parameters in a single forward pass.

The network can handle arbitrary numbers of shots, and for classification, arbitrary numbers of classes at train and test time (see Section 3).

In Section 5, we evaluate VERSA on (i) standard benchmarks where the method sets new state-of-the-art results, (ii) settings where test conditions (shot and way) differ from training, and (iii) a challenging one-shot view reconstruction task.

We now present the framework that consists of (i) a multi-task probabilistic model, and (ii) a method for meta-learning probabilistic inference.

Two principles guide the choice of model.

First, the use of discriminative models to maximize predictive performance on supervised learning tasks BID39 .

Second, the need to leverage shared statistical structure between tasks (i.e. multi-task learning).

These criteria are met by the standard multi-task directed graphical model shown in FIG1 that employs shared parameters θ, which are common to all tasks, and task specific parameters {ψ Let X (t) and Y (t) denote all the inputs and outputs (both test and train) for task t. The joint probability of the outputs and task specific parameters for T tasks, given the inputs and global parameters is: m , ψ (t) , θ .

DISPLAYFORM0 In the next section, the goal is to meta-learn fast and accurate approximations to the posterior predictive distribution p( DISPLAYFORM1 for unseen tasks t.

This section provides a framework for meta-learning approximate inference that is a simple reframing and extension of existing approaches BID10 BID16 .

We will employ point estimates for the shared parameters θ since data across all tasks will pin down their value.

Distributional estimates will be used for the task-specific parameters since only a few shots constrain them.

Once the shared parameters are learned, the probabilistic solution to few-shot learning in the model above comprises two steps.

First, form the posterior distribution over the task-specific parameters p(ψ (t) |x (t) , D (t) , θ).

Second, compute the posterior predictive p(ỹ (t) |x (t) , θ).

These steps will require approximation and the emphasis here is on performing this quickly at test time.

We will describe the form of the approximation, the optimization problem used to learn it, and how to implement this efficiently below.

In what follows we initially suppress dependencies on the inputsx and shared parameters θ to reduce notational clutter, but will reintroduce these at the end of the section.

Specification of the approximate posterior predictive distribution.

Our framework approximates the posterior predictive distribution by an amortized distribution q φ (ỹ|D).

That is, we learn a feed-forward inference network with parameters φ that takes any training dataset D (t) and test inputx as inputs and returns the predictive distribution over the test outputỹ (t) .

We construct this by amortizing the approximate posterior q φ (ψ|D) and then form the approximate posterior predictive distribution using: DISPLAYFORM0 This step may require additional approximation e.g. Monte Carlo sampling.

The amortization will enable fast predictions at test time.

The form of these distributions is identical to those used in amortized variational inference BID8 .

In this work, we use a factorized Gaussian distribution for q φ (ψ|D (t) ) with means and variances set by the amortization network.

However, the training method described next is different.

Meta-learning the approximate posterior predictive distribution.

The quality of the approximate posterior predictive for a single task will be measured by the KL-divergence between the true and approximate posterior predictive distribution KL [p(ỹ|D) q φ (ỹ|D)].

The goal of learning will be to minimize the expected value of this KL averaged over tasks, DISPLAYFORM1 Training will therefore return parameters φ that best approximate the posterior predictive distribution in an average KL sense.

So, if the approximate posterior q φ (ψ|D) is rich enough, global optimization will recover the true posterior p(ψ|D) (assuming p(ψ|D) obeys identifiability conditions BID4 ).1 Thus, the amortized procedure meta-learns approximate inference that supports accurate prediction.

Appendix A provides a generalized derivation of the framework, grounded in Bayesian decision theory BID21 .The right hand side of Eq. (2) indicates how training could proceed: (i) select a task t at random, (ii) sample some training data D (t) , (iii) form the posterior predictive q φ (·|D (t) ) and, (iv) compute the log-density log q φ (ỹ DISPLAYFORM2 .

Repeating this process many times and averaging the results would provide an unbiased estimate of the objective which can then be optimized.

This perspective also makes it clear that the procedure is scoring the approximate inference procedure by simulating approximate Bayesian held-out log-likelihood evaluation.

Importantly, while an inference network is used to approximate posterior distributions, the training procedure differs significantly from standard variational inference.

In particular, rather than minimizing KL(q φ (ψ|D) p(ψ|D)), our objective function directly focuses on the posterior predictive distribution and minimizes KL(p(ỹ|D) q φ (ỹ|D)).End-to-end stochastic training.

Armed by the insights above we now layout the full training procedure.

We reintroduce inputs and shared parameters θ and the objective becomes: (D,ỹ,x) [log q φ (ỹ|x, θ)] = − E p (D,ỹ,x) log p(ỹ|x, ψ, θ)q φ (ψ|D, θ)dψ .

DISPLAYFORM3 We optimize the objective over the shared parameters θ as this will maximize predictive performance (i.e., Bayesian held out likelihood).

An end-to-end stochastic training objective for θ and φ is: DISPLAYFORM4 DISPLAYFORM5 individual feature extraction instance pooling regression onto weights and {ỹ DISPLAYFORM6 , where p represents the data distribution (e.g., sampling tasks and splitting them into disjoint training data D and test data {(x DISPLAYFORM7 ).

This type of training therefore uses episodic train / test splits at meta-train time.

We have also approximated the integral over ψ using L Monte Carlo samples.

The local reparametrization trick enables optimization.

Interestingly, the learning objective does not require an explicit specification of the prior distribution over parameters, p(ψ (t) |θ), learning it implicitly through q φ (ψ|D, θ) instead.

In summary, we have developed an approach for Meta-Learning Probabilistic Inference for Prediction (ML-PIP).

A simple investigation of the inference method with synthetic data is provided in Section 5.1.

In Section 4 we will show that this formulation unifies a number of existing approaches, but first we discuss a particular instance of the ML-PIP framework that supports versatile learning.

A versatile system is one that makes inferences both rapidly and flexibly.

By rapidly we mean that test-time inference involves only simple computation such as a feed-forward pass through a neural network.

By flexibly we mean that the system supports a variety of tasks -including variable numbers of shots or numbers of classes in classification problems -without retraining.

Rapid inference comes automatically with the use of a deep neural network to amortize the approximate posterior distribution q. However, it typically comes at the cost of flexibility: amortized inference is usually limited to a single specific task.

Below, we discuss design choices that enable us to retain flexibility.

Inference with sets as inputs.

The amortization network takes data sets of variable size as inputs whose ordering we should be invariant to.

We use permutation-invariant instance-pooling operations to process these sets similarly to BID42 and as formalized in BID58 .

The instance-pooling operation ensures that the network can process any number of training observations.

VERSA for Few-Shot Image Classification.

For few-shot image classification, our parameterization of the probabilistic model is inspired by early work from BID18 ; BID0 and recent extensions to deep learning BID1 BID43 .

A feature extractor neural network h θ (x) ∈ R d θ , shared across all tasks, feeds into a set of task-specific linear classifiers with softmax outputs and weights and biases ψ (t) = {W (t) , b (t) } (see FIG4 ).A naive amortization requires the approximate posterior q φ (ψ|D, θ) to model the distribution over full weight matrices in R d θ ×C (and biases).

This requires the specification of the number of few-shot classes C ahead of time and limits inference to this chosen number.

Moreover, it is difficult to metalearn systems that directly output large matrices as the output dimensionality is high.

We therefore propose specifying q φ (ψ|D, θ) in a context independent manner such that each weight vector ψ c depends only on examples from class c, by amortizing individual weight vectors associated with a DISPLAYFORM0 . . .

DISPLAYFORM1 individual feature extraction instance pooling regression onto stochastic inputs DISPLAYFORM2 is then concatenated with a test anglex and mapped onto a new image through the generator θ.

Right: Amortization network that maps k image/angle examples of a particular object-instance to the corresponding stochastic input.single softmax output instead of the entire weight matrix directly.

To reduce the number of learned parameters, the amortization network operates directly on the extracted features h θ (x): DISPLAYFORM3 Note that in our implementation, end-to-end training is employed, i.e., we backpropagate to θ through the inference network.

Here k c is the number of observed examples in class c and ψ c = {w c , b c } denotes the weight vector and bias of the linear classifier associated with that class.

Thus, we construct the classification matrix ψ DISPLAYFORM4 by performing C feed-forward passes through the inference network q φ (ψ|D, θ) (see FIG4 ).The assumption of context independent inference is an approximation.

In Appendix B, we provide theoretical and empirical justification for its validity.

Our theoretical arguments use insights from Density Ratio Estimation BID36 BID49 , and we empirically demonstrate that full approximate posterior distributions are close to their context independent counterparts.

Critically, the context independent approximation addresses all the limitations of a naive amortization mentioned above: (i) the inference network needs to amortize far fewer parameters whose number does not scale with number of classes C (a single weight vector instead of the entire matrix); (ii) the amortization network can be meta-trained with different numbers of classes per task, and (iii) the number of classes C can vary at test-time.

VERSA for Few-Shot Image Reconstruction (Regression).

We consider a challenging few-shot learning task with a complex (high dimensional and continuous) output space.

We define view reconstruction as the ability to infer how an object looks from any desired angle based on a small set of observed views.

We frame this as a multi-output regression task from a set of training images with known orientations to output images with specified orientations.

Our generative model is similar to the generator of a GAN or the decoder of a VAE: A latent vector ψ (t) ∈ R d ψ , which acts as an object-instance level input to the generator, is concatenated with an angle representation and mapped through the generator to produce an image at the specified orientation.

In this setting, we treat all parameters θ of the generator network as global parameters (see Appendix E.1 for full details of the architecture), whereas the latent inputs ψ (t) are the task-specific parameters.

We use a Gaussian likelihood in pixel space for the outputs of the generator.

To ensure that the output means are between zero and one, we use a sigmoid activation after the final layer.

φ parameterizes an amortization network that first processes the image representations of an object, concatenates them with their associated view orientations, and processes them further before instance-pooling.

From the pooled representations, q φ (ψ|D, θ) produces a distribution over vectors ψ (t) .

This process is illustrated in FIG5 .

In this section, we continue in the spirit of BID16 , and recast a broader class of metalearning approaches as approximate inference in hierarchical models.

We show that ML-PIP unifies a number of important approaches to meta-learning, including both gradient and metric based variants, as well as amortized MAP inference and conditional modelling approaches BID13 .

We lay out these connections, most of which rely on point estimates for the task-specific parameters corresponding to q( DISPLAYFORM0 .

In addition, we compare previous approaches to VERSA.Gradient-Based Meta-Learning.

Let the task-specific parameters ψ (t) be all the parameters in a neural network.

Consider a point estimate formed by taking a step of gradient ascent of the training loss, initialized at ψ 0 and with learning rate η.

DISPLAYFORM1 This is an example of semi-amortized inference BID24 , as the only shared inference parameters are the initialization and learning rate, and optimization is required for each task (albeit only for one step).

Importantly, Eq. (6) recovers Model-agnostic meta-learning BID10 , providing a perspective as semi-amortized ML-PIP.

This perspective is complementary to that of BID16 who justify the one-step gradient parameter update employed by MAML through MAP inference and the form of the prior p(ψ|θ).

Note that the episodic meta-train / meta-test splits do not fall out of this perspective.

Instead we view the update choice as one of amortization which is trained using the predictive KL and naturally recovers the test-train splits.

More generally, multiple gradient steps could be fed into an RNN to compute ψ * which recovers BID44 .

In comparison to these methods, besides being distributional over ψ, VERSA relieves the need to back-propagate through gradient based updates during training and compute gradients at test time, as well as enables the treatment of both local and global parameters which simplifies inference.

Metric-Based Few-Shot Learning.

Let the task-specific parameters be the top layer softmax weights and biases of a neural network ψ (t) = {w DISPLAYFORM2 .

The shared parameters are the lower layer weights.

Consider amortized point estimates for these parameters constructed by averaging the top-layer activations for each class, DISPLAYFORM3 where µ DISPLAYFORM4 These choices lead to the following predictive distribution: DISPLAYFORM5 which recovers prototypical networks BID48 ) using a Euclidean distance function d with the final hidden layer being the embedding space.

In comparison, VERSA is distributional and it uses a more flexible amortization function that goes beyond averaging of activations.

Amortized MAP inference.

BID43 proposed a method for predicting weights of classes from activations of a pre-trained network to support i) online learning on a single task to which new few-shot classes are incrementally added, ii) transfer from a high-shot classification task to a separate low-shot classification task.

This is an example usage of hyper-networks BID17 to amortize learning about weights, and can be recovered by the ML-PIP framework by pre-training θ and performing MAP inference for ψ.

VERSA goes beyond point estimates and although its amortization network is similar in spirit, it is more general, employing end-to-end training and supporting full multi-task learning by sharing information between many tasks.

Conditional models trained via maximum likelihood.

In cases where a point estimate of the task-specific parameters are used the predictive becomes DISPLAYFORM6 In such cases the amortization network that computes ψ * (D, θ) can be equivalently viewed as part of the model specification rather than the inference scheme.

From this perspective, the ML-PIP training procedure for φ and θ is equivalent to training a conditional model p(ỹ|ψ * φ (D, θ), θ) via maximum likelihood estimation, establishing a strong connection to neural processes BID13 .Comparison to Variational Inference (VI).

Standard application of amortized VI BID3 for ψ in the multi-task discriminative model optimizes the Monte Carlo approximated free-energy w.r.t.

φ and θ: DISPLAYFORM7 where ψ DISPLAYFORM8 .

In addition to the conceptual difference from ML-PIP (discussed in Section 2.1), this differs from the ML-PIP objective by i) not employing meta train / test splits, and ii) including the KL for regularization instead.

In Section 5, we show that VERSA significantly improves over standard VI in the few-shot classification case and compare to recent VI/meta-learning hybrids.

We evaluate VERSA on several few-shot learning tasks.

We begin with toy experiments to investigate the properties of the amortized posterior inference achieved by VERSA.

We then report few-shot classification results using the Omniglot and miniImageNet datasets in Section 5.2, and demonstrate VERSA's ability to retain high accuracy as the shot and way are varied at test time.

In Section 5.3, we examine VERSA's performance on a one-shot view reconstruction task with ShapeNet objects.

To investigate the approximate inference performed by our training procedure, we run the following experiment.

We first generate data from a Gaussian distribution with a mean that varies across tasks: DISPLAYFORM0 (11) We generate T = 250 tasks in two separate experiments, having N ∈ {5, 10} train observations and M = 15 test observations.

We introduce the inference network q φ (ψ|D DISPLAYFORM1 q ), amortizing inference as: DISPLAYFORM2 The learnable parameters φ = {w µ , b µ , w σ , b σ } are trained with the objective function in Eq. (4).

The model is trained to convergence with Adam BID25 using mini-batches of tasks from the generated dataset.

Then, a separate set of tasks is generated from the same generative process, and the posterior q φ (ψ|D) is inferred with the learned amortization parameters.

The true posterior over ψ is Gaussian with a mean that depends on the task, and may be computed analytically.

Fig. 4 shows the approximate posterior distributions inferred for unseen test sets by the trained amortization networks.

The evaluation shows that the inference procedure is able to recover accurate posterior distributions over ψ, despite minimizing a predictive KL divergence in data space.

We evaluate VERSA on standard few-shot classification tasks in comparison to previous work.

Specifically, we consider the Omniglot BID32 and miniImageNet BID44 datasets which are C-way classification tasks with k c examples per class.

VERSA follows the implementation in Sections 2 and 3, and the approximate inference scheme in Eq. (5).

We follow the experimental protocol established by BID54 for Omniglot and Ravi and Larochelle DISPLAYFORM0 Figure 4: True posteriors p(ψ|D) ( ) and approximate posteriors q φ (ψ|D) ( ) for unseen test sets ( ) in the experiment.

In both cases (five and ten shot), the approximate posterior closely resembles the true posterior given the observed data.(2017) for miniImagenet, using equivalent architectures for h θ .

Training is carried out in an episodic manner: for each task, k c examples are used as training inputs to infer q φ (ψ (c) |D, θ) for each class, and an additional set of examples is used to evaluate the objective function.

Full details of data preparation and network architectures are provided in Appendix D. TAB1 details few-shot classification performance for VERSA as well as competitive approaches.

The tables include results for only those approaches with comparable training procedures and convolutional feature extraction architectures.

Approaches that employ pre-training and/or residual networks BID1 BID43 BID46 BID15 BID12 have been excluded so that the quality of the learning algorithm can be assessed separately from the power of the underlying discriminative model.

For Omniglot, the training, validation, and test splits have not been specified for previous methods, affecting the comparison.

VERSA achieves a new state-of-the-art results (67.37% -up 1.38% over the previous best) on 5-way -5-shot classification on the miniImageNet benchmark and (97.66% -up 0.02%) on the 20-way -1 shot Omniglot benchmark for systems using a convolution-based network architecture and an end-to-end training procedure.

VERSA is within error bars of state-of-the-art on three other benchmarks including 5-way -1-shot miniImageNet, 5-way -5-shot Omniglot, and 5-way -1-shot Omniglot.

Results on the Omniglot 20 way -5-shot benchmark are very competitive with, but lower than other approaches.

While most of the methods evaluated in TAB1 adapt all of the learned parameters for new tasks, VERSA is able to achieve state-of-the-art performance despite adapting only the weights of the top-level classifier.

Comparison to standard and amortized VI.

To investigate the performance of our inference procedure, we compare it in terms of log-likelihood TAB0 ) and accuracy TAB1 to training the same model using both amortized and non-amortized VI (i.e., Eq. FORMULA2 ).

Derivations and further experimental details are provided in Appendix C. VERSA improves substantially over amortized VI even though the same amortization network is used for both.

This is due to VI's tendency to under-fit, especially for small numbers of data points BID52 BID53 which is compounded when using inference networks BID6 .

Using non-amortized VI improves performance substantially, but does not reach the level of VERSA and forming the posterior is significantly slower as it requires many forward / backward passes through the network.

This is similar in spirit to MAML BID10 , though MAML dramatically reduces the number of required iterations by finding good global initializations e.g., five gradient steps for miniImageNet.

This is in contrast to the single forward pass required by VERSA.Versatility.

VERSA allows us to vary the number of classes C and shots k c between training and testing (Eq. FORMULA14 ).

FIG7 shows that a model trained for a particular C-way retains very high accuracy as C is varied.

For example, when VERSA is trained for the 20-Way, 5-Shot condition, at test-time it can handle C = 100 way conditions and retain an accuracy of approximately 94%.

FIG7 shows similar robustness as the number of shots k c is varied.

VERSA therefore demonstrates considerable flexibility and robustness to the test-time conditions, but at the same time it is efficient as it only requires forward passes through the network.

The time taken to evaluate 1000 test tasks with a 5-way, 5-shot miniImageNet trained model using MAML (https://github.com/cbfinn/maml) is 302.9 seconds whereas VERSA took 53.5 seconds on a NVIDIA Tesla P100-PCIE-16GB GPU.

This is more than 5× speed advantage in favor of VERSA while bettering MAML in accuracy by 4.26%.

ShapeNetCore v2 BID5 ) is a database of 3D objects covering 55 common object categories with ∼51,300 unique objects.

For our experiments, we use 12 of the largest object categories.

We concatenate all instances from all 12 of the object categories together to obtain a dataset of 37,108 objects.

This dataset is then randomly shuffled and we use 70% of the objects for training, 10% for validation, and 20% for testing.

For each object, we generate 36 views of size 32 × 32 pixels spaced evenly every 10 degrees in azimuth around the object.

We evaluate VERSA by comparing it to a conditional variational autoencoder (C-VAE) with view angles as labels BID38 and identical architectures.

We train VERSA in an episodic manner and the C-VAE in batch-mode on all 12 object classes at once.

We train on a single view selected at random and use the remaining views to evaluate the objective function.

For full experimentation details see Appendix E. FIG10 shows views of unseen objects from the test set generated from a single shot with VERSA as well as a C-VAE and compares both to ground truth views.

Both VERSA and the C-VAE capture the correct orientation of the object in the generated images.

However, VERSA produces images that contain much more detail and are visually sharper than the C-VAE images.

Although important information is missing due to occlusion in the single shot, VERSA is often able to accurately impute this information presumably due to learning the statistics of these objects.

Table 2 provides quantitative comparison results between VERSA with varying shot and the C-VAE.

The quantitative metrics all show the superiority of VERSA over a C-VAE.

As the number of shots increase to 5, the measurements show a corresponding improvement.

Table 2 : View reconstruction test results.

Mean squared error (MSE -lower is better) and the structural similarity index (SSIM -higher is better) BID56 are measured between the generated and ground truth images.

Error bars not shown as they are insignificant.

We have introduced ML-PIP, a probabilistic framework for meta-learning.

ML-PIP unifies a broad class of recently proposed meta-learning methods, and suggests alternative approaches.

Building on ML-PIP, we developed VERSA, a few-shot learning algorithm that avoids the use of gradient based optimization at test time by amortizing posterior inference of task-specific parameters.

We evaluated VERSA on several few-shot learning tasks and demonstrated state-of-the-art performance and compelling visual results on a challenging 1-shot view reconstruction task.

a We report the performance of Prototypical Networks when training and testing with the same "shot" and "way", which is consistent with the experimental protocol of the other methods listed.

We note that Prototypical Networks perform better when trained on higher "way" than that of testing.

In particular, when trained on 20-way classification and tested on 5-way, the model achieves 68.20 ± 0.66%.

A generalization of the new inference framework presented in Section 2 is based upon Bayesian decision theory (BDT).

BDT provides a recipe for making predictionsŷ for an unknown test variablẽ y by combining information from observed training data D (t) (here from a single task t) and a loss function L(ỹ,ŷ) that encodes the cost of predictingŷ when the true value isỹ BID2 BID21 .

In BDT an optimal prediction minimizes the expected loss (suppressing dependencies on the inputs and θ to reduce notational clutter): DISPLAYFORM0 given the training data from task t.

BDT separates test and training data and so is a natural lens through which to view recent episodic approaches to training that utilize many internal training/test splits BID54 .

Based on this insight, what follows is a fairly dense derivation of an ultimately simple stochastic variational objective for meta-learning probabilistic inference that is rigorously grounded in Bayesian inference and decision theory.

Distributional BDT.

We generalize BDT to cases where the goal is to return a full predictive distribution q(·) over the unknown test variableỹ rather than a point prediction.

The quality of q is quantified through a distributional loss function L(ỹ, q(·)).

Typically, ifỹ (the true value of the underlying variable) falls in a low probability region of q(·) the loss will be high, and vice versa.

The optimal predictive q * is found by optimizing the expected distributional loss with q constrained to a distributional family Q: DISPLAYFORM1 Amortized variational training.

Here, we amortize q to form quick predictions at test time and learn parameters by minimizing average expected loss over tasks.

Let φ be a set of shared variational parameters such that q(ỹ) = q φ (ỹ|D) (or q φ for shorthand).

Now the approximate predictive distribution can take any training dataset D (t) as an argument and directly perform prediction ofỹ (t) .

The optimal variational parameters are found by minimizing the expected distributional loss across tasks DISPLAYFORM2 (A.3) Here the variables D,x andỹ are placeholders for integration over all possible datasets, test inputs and outputs.

Note that Eq. (A.3) can be stochastically approximated by sampling a task t and randomly partitioning into training data D and test data {x m ,ỹ m } M m=1 , which naturally recovers episodic minibatch training over tasks and data BID54 BID44 .

Critically, this does not require computation of the true predictive distribution.

It also emphasizes the meta-learning aspect of the procedure, as the model is learning how to infer predictive distributions from training tasks.

Loss functions.

We employ the log-loss: the negative log density of q φ atỹ.

In this case, DISPLAYFORM3 where KL[p(y) q(y)] is the KL-divergence, and H [p(y)] is the entropy of p. Eq. (A.4) has the elegant property that the optimal q φ is the closest member of Q (in a KL sense) to the true predictive p(ỹ|D), which is unsurprising as the log-loss is a proper scoring rule BID20 .

This is reminiscent of the sleep phase in the wake-sleep algorithm BID19 .

Exploration of alternative proper scoring rules BID7 and more task-specific losses BID31 ) is left for future work.

Specification of the approximate predictive distribution.

Next, we consider the form of q φ .

Motivated by the optimal predictive distribution, we replace the true posterior by an approximation: DISPLAYFORM4 In this section we lay out both theoretical and empirical justifications for the context-independent approximation detailed in Section 3.

A principled justification for the approximation is best understood through the lens of density ratio estimation BID36 BID49 .

We denote the conditional density of each class as p(x|y = c) and assume equal a priori class probability p(y = c) = 1/C. Density ratio theory then uses Bayes' theorem to show that the optimal softmax classifier can be expressed in terms of the conditional densities BID36 BID49 : DISPLAYFORM0 This implies that the optimal classifier will construct estimators for the conditional density for each class, that is exp(h(x) w c ) ∝ p(x|y = c).

Importantly for our approximation, notice that these estimates are constructed independently for each class, similarly to training a naive Bayes classifier.

VERSA mirrors this optimal form using: DISPLAYFORM1 where w c ∼ q φ (w|{x n |y n = c}) for each class in a given task.

Under ideal conditions (i.e., if one could perfectly estimate p(x|y = c)), the context-independent assumption holds, further motivating our design.

Here we detail a simple experiment to evaluate the validity of the context-independent inference assumption.

The goal of the experiment is to examine if weights may be context-independent without imposing the assumption on the amortization network.

To see this, we randomly generate fifty tasks from a dataset, where classes may appear a number of times in different tasks.

We then perform free-form (non-amortized) variational inference on the weights for each of the tasks, with a Gaussian variational distribution: DISPLAYFORM0 If the assumption is reasonable, we may expect the distribution of the weights of a specific class to be similar regardless of the additional classes in the task.

We examine 5-way classification in the MNIST dataset.

We randomly sample and fix fifty such tasks.

We train the model twice using the same feature extraction network used in the few-shot classification experiments, and fix the d θ to be 16 and 2.

We then train the model in an episodic manner by minibatching tasks at each iteration.

The model is trained to convergence, and achieves 99% accuracy on held out test examples for the tasks.

After training is complete we examine the optimized µ (t) φ for each class in each task.

FIG1 shows a t-SNE BID34 plot for the 16-dimensional weights.

We see that when reduced to 2-dimensions, the weights cluster according to class.

the classes cluster in 2-dimensional space as well.

However, there is some overlap (e.g., classes '1' and '2'), and that for some tasks a class-weight may appear away from the cluster.

FIG4 shows the same plot, but only for tasks that contain both class '1' and '2'.

Here we can see that for these tasks, class '2' weights are all located away from their cluster.

This implies that each class-weights are typically well-approximated as being independent of the task.

However, if the model lacks capacity to properly assign each set of class weights to different regions of space, for tasks where classes from similar regions of space appear, the inference procedure will 'move' one of the class weights to an 'empty' region of the space.

We derive a VI-based objective for our probabilistic model.

By "amortized" VI we mean that q φ (ψ|D (t) , θ) is parameterized by a neural network with a fixed-sized φ.

Conversely, "non-amortized" VI refers to local parameters φ (t) that are optimized independently (at test time) for each new task t, such that q(ψ|D (t) , θ) = N (ψ|µ φ (t) , Σ φ (t) ).

However, the derivation of the objective function does not change between these options.

For a single task t, an evidence lower bound (ELBO; BID55 ) may be expressed as: DISPLAYFORM0 We can then derive a stochastic estimator to optimize Eq. (C.1) by sampling D (t) ∼ p(D) (approximated with a training set of tasks) and simple Monte Carlo integration over ψ such that DISPLAYFORM1 Eq. (C.2) differs from our objective function in Eq. (4) in two important ways: (i) Eq. (4) does not contain a KL term for q φ (ψ|D (t) , θ) (nor any other form of prior distribution over ψ, and (ii) Eq. (C.1) does not distinguish between training and test data within a task, and therefore does not explicitly encourage the model to generalize in any way.

In this section we provide comprehensive details on the few-shot classification experiments.

Omniglot BID32 ) is a few-shot learning dataset consisting of 1623 handwritten characters (each with 20 instances) derived from 50 alphabets.

We follow a pre-processing and training procedure akin to that defined in BID54 .

First the images are resized to 28 × 28 pixels and then character classes are augmented with rotations of 90 degrees.

The training, validation and test sets consist of a random split of 1100, 100, and 423 characters, respectively.

When augmented this results in 4400 training, 400 validation, and 1292 test classes, each having 20 character instances.

For C-way, k c -shot classification, training proceeds in an episodic manner.

Each training iteration consists of a batch of one or more tasks.

For each task C classes are selected at random from the training set.

During training, k c character instances are used as training inputs and 15 character instances are used as test inputs.

The validation set is used to monitor the progress of learning and to select the best model to test, but does not affect the training process.

Final evaluation of the trained model is done on 600 randomly selected tasks from the test set.

During evaluation, k c character instances are used as training inputs and k c character instances are used as test inputs.

We use the Adam BID25 optimizer with a constant learning rate of 0.0001 with 16 tasks per batch to train all models.

The 5-way -5-shot and 5-way -1-shot models are trained for 80,000 iterations while the 20-way -5-shot model is trained for 60,000 iterations, and the 20-way -1-shot model is trained for 100,000 iterations.

In addition, we use a Gaussian form for q and set the number of ψ samples to L = 10.

miniImageNet BID54 ) is a dataset of 60,000 color images that is sub-divided into 100 classes, each with 600 instances.

The images have dimensions of 84 × 84 pixels.

For our experiments, we use the 64 training, 16 validation, and 20 test class splits defined by BID44 .

Training proceeds in the same episodic manner as with Omniglot.

We use the Adam BID25 optimizer and a Gaussian form for q and set the number of ψ samples to L = 10.

For the 5-way -5-shot model, we train using 4 tasks per batch for 100,000 iterations and use a constant learning rate of 0.0001.

For the 5-way -1-shot model, we train with 8 tasks per batch for 50,000 iterations and use a constant learning rate of 0.00025.

TAB0 to D.4 detail the neural network architectures for the feature extractor θ, amortization network φ, and linear classifier ψ, respectively.

The feature extraction network is very similar to that used in BID54 .

The output of the amortization network yields mean-field Gaussian parameters for the weight distributions of the linear classifier ψ.

When sampling from the weight distributions, we employ the local-reparameterization trick , that is we sample from the implied distribution over the logits rather than directly from the variational distribution.

To reduce the number of learned parameters, we share the feature extraction network θ with the pre-processing phase of the amortizaion network ψ.

Omniglot Shared Feature Extraction Network (θ):x → h θ (x) Output size Layers 28 × 28 × 1 Input image 14 × 14 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, SAME) 7 × 7 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, SAME) 4 × 4 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, SAME) 2 × 2 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, SAME) 256 flatten Table D .2: Feature extraction network used for miniImageNet few-shot learning.

Batch Normalization and dropout with a keep probability of 0.5 used throughout.

miniImageNet Shared Feature Extraction Network (θ):x → h θ (x) Output size Layers 84 × 84 × 1 Input image 42 × 42 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, VALID) 21 × 21 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, VALID) 10 × 10 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, VALID) 5 × 5 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, VALID) 2 × 2 × 64 conv2d (3 × 3, stride 1, SAME, RELU), dropout, pool (2 × 2, stride 2, VALID) 256 flatten E SHAPENET EXPERIMENTATION DETAILS E.1 VIEW RECONSTRUCTION TRAINING PROCEDURE AND NETWORK ARCHITECTURES ShapeNetCore v2 BID5 is an annotated database of 3D objects covering 55 common object categories with ∼51,300 unique objects.

For our experiments, we use 12 of the largest object categories.

Refer to Linear Classifier (ψ): h θ (x) → p(ỹ|x, θ, ψ t ) Output size Layers

Input features C fully connected, softmax object categories together to obtain a dataset of 37,108 objects.

This concatenated dataset is then randomly shuffled and we use 70% of the objects (25,975 in total) for training, 10% for validation (3,710 in total) , and 20% (7423 in total) for testing.

For each object, we generate V = 36, 128 × 128 pixel image views spaced evenly every 10 degrees in azimuth around the object.

We then convert the rendered images to gray-scale and reduce their size to be 32 × 32 pixels.

Again, we train our model in an episodic manner.

Each training iteration consists a batch of one or more tasks.

For each task an object is selected at random from the training set.

We train on a single view selected at random from the V = 36 views associated with each object and use the remaining 35 views to evaluate the objective function.

We then generate 36 views of the object with a modified version of our amortization network which is shown diagrammatically in FIG5 .

To evaluate the system, we generate views and compute quantitative metrics over the entire test set.

Tables E.2 to E.4 describe the network architectures for the encoder, amortization, and generator networks, respectively.

To train, we use the Adam BID25 optimizer with a constant learning rate of 0.0001 with 24 tasks per batch for 500,000 training iterations.

In addition, we set d φ = 256, d ψ = 256 and number of ψ samples to 1.

ShapeNet Encoder Network (φ): y → h Output size Layers 32 × 32 × 1 Input image 16 × 16 × 64 conv2d (3 × 3, stride 1, SAME, RELU), pool (2 × 2, stride 2, VALID) 8 × 8 × 64 conv2d (3 × 3, stride 1, SAME, RELU), pool (2 × 2, stride 2, VALID) 4 × 4 × 64 conv2d (3 × 3, stride 1, SAME, RELU), pool (2 × 2, stride 2, VALID) 2 × 2 × 64 conv2d (3 × 3, stride 1, SAME, RELU), pool (2 × 2, stride 2, VALID) d φ fully connected, RELU

<|TLDR|>

@highlight

Novel framework for meta-learning that unifies and extends a broad class of existing few-shot learning methods. Achieves strong performance on few-shot learning benchmarks without requiring iterative test-time inference.   

@highlight

This work tackles few-shot learning from a probabilistic inference viewpoint, achieving state-of-the-art despite simpler setup than many competitors