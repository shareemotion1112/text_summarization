Using variational Bayes neural networks, we develop an algorithm capable of accumulating knowledge into a prior from multiple different tasks.

This results in a rich prior capable of few-shot learning on new tasks.

The posterior can go beyond the mean field approximation and yields good uncertainty on the performed experiments.

Analysis on toy tasks show that it can learn from significantly different tasks while finding similarities among them.

Experiments on Mini-Imagenet reach state of the art with 74.5% accuracy on 5 shot learning.

Finally, we provide two new benchmarks, each showing a failure mode of existing meta learning algorithms such as MAML and prototypical Networks.

Recently, significant progress has been made to scale Bayesian neural networks to large tasks and to provide better approximations of the posterior distribution BID4 .

Recent works extend fully factorized posterior distributions to more general families BID22 BID21 .

It is also possible to sample from the posterior distribution trough mini-batch updates BID23 BID36 .However, for neural networks, the prior is often chosen for convenience.

This may become a problem when the number of observations is insufficient to overcome the choice of the prior.

In this regime, the prior must express our current knowledge on the task and, most importantly, our lack of knowledge on it.

In addition to that, a good approximation of the posterior under the small sample size regime is required, including the ability to model multiple modes.

This is indeed the case for Bayesian optimization BID30 , Bayesian active learning BID12 , continual learning BID20 , safe reinforcement learning BID3 , exploration-exploitation trade-off in reinforcement learning BID16 .

Gaussian processes BID27 have historically been used for these applications, but an RBF kernel constitute a prior that is unsuited for many tasks.

More recent tools such as deep Gaussian processes BID6 show great potential and yet their scalability whilst learning from multiple tasks needs to be improved.

Our contributions are as follow:1.

We provide a simple and scalable procedure to learn an expressive prior and posterior over models from multiple tasks.2.

We reach state of the art performances on mini-imagenet.3.

We propose two new benchmarks, each exposing a failure mode of popular meta learning algorithms.

In contrast, our method perform well on these benchmarks.• MAML BID11 does not perform well on a collection of sinus tasks when the frequency varies.• Prototypical Network BID29 )'s performance decrease considerably when the diversity of tasks increases.

Outline: We first describe the proposed approach in Section 2.

In Section 3, we extend to three level of hierarchies and obtain a model more suited for classification.

Section 4 review related methods and outline the key differences.

Finally, In Section 5, we conduct experiments on three different benchmarks to gain insight in the behavior of our algorithm.

By leveraging the variational Bayesian approach, we show how we can learn a prior over models with neural networks.

We start our analysis with the goal of learning a prior p(w|α) over the weights w of neural networks across multiple tasks.

We then provide a reduction of the Evidence Lower BOund (ELBO) showing that it is not necessary to explicitly model a distribution in the very high dimension of the weight space of neural networks.

Instead the algorithm learns a subspace suitable for expressing model uncertainty within the distributions of tasks considered in the multi-task environment.

This simplification results in a scalable algorithm which we refer to as deep prior.

To learn a probability distribution p(w|α) over the weights w of a network parameterized by α, we use a hierarchical Bayes approach across N tasks, with hyper-prior p(α).

Each task has its own parameters w j , with DISPLAYFORM0 , we have the following posterior: DISPLAYFORM1 The term p(y ij |x ij , w j ) corresponds to the likelihood of sample i of task j given a model parameterized by w j e.g. the probability of class y ij from the softmax of a neural network parameterized by w j with input x ij .

For the posterior p(α|D), we assume that the large amount of data available across multiple tasks will be enough to overcome a generic prior p(α), such as an isotropic Normal distribution.

Hence, we consider a point estimate of the posterior p(α|D) using maximum a posteriori 2 .We can now focus on the remaining term: p(w j |α).

Since w j is potentially high dimensional with intricate correlations among the different dimensions, we cannot use a simple Gaussian distribution.

Following inspiration from generative models such as GANs BID13 and VAE BID18 , we use an auxiliary variable z ∼ N (0, I dz ) and a deterministic function projecting the noise z to the space of w i.e. w = h α (z).

Marginalizing z, we have: p(w|α) = z p(z)p(w|z, α)dz = z p(z)δ hα(z)−w dz, where δ is the Dirac delta function.

Unfortunately, directly marginalizing z is intractable for general h α .

To overcome this issue, we add z to the joint inference and marginalize it at inference time.

Considering the point estimation of α, the full posterior is factorized as follows: DISPLAYFORM2 where p(y ij |x ij , w j ) is the conventional likelihood function of a neural network with weight matrices generated from the function h α i.e.: w j = h α (z j ).

Similar architecture has been used in BID21 and BID22 , but we will soon show that it can be reduced to a simpler architecture in the context of multi-task learning.

The other terms are defined as follows: DISPLAYFORM3 The task will consist of jointly learning a function h α common to all tasks and a posterior distribution p(z j |α, S j ) for each task.

At inference time, predictions are performed by marginalizing z i.e.: DISPLAYFORM4

In the previous section, we described the different components for expressing the posterior distribution of Equation 4.

While all these components are tractable, the normalization factor is still intractable.

To address this issue, we follow the Variational Bayes approach BID4 .Conditioning on α, we saw in Equation 1 that the posterior factorizes independently for all tasks.

This reduces the joint Evidence Lower BOund (ELBO) to a sum of individual ELBO for each task.

Given a family of distributions q θj (z j |S j , α), parameterized by {θ j } N j=1 and α, the Evidence Lower Bound for task j is: DISPLAYFORM0 where, DISPLAYFORM1 Notice that after simplification 3 , KL j is no longer over the space of w j but only over the space z j .

Namely, the posterior distribution is factored into two components, one that is task specific and one that is task agnostic and can be shared with the prior.

This amounts to finding a low dimensional manifold in the parameter space where the different tasks can be distinguished.

Then, the posterior p(z j |S j , α) only has to model which of the possible tasks are likely, given observations S j instead of modeling the high dimensional p(w j |S j , α).But, most importantly, any explicit reference to w has now vanished from both Equation 5 and Equation 6.

This simplification has an important positive impact on the scalability of the proposed approach.

Since we no longer need to explicitly calculate the KL on the space of w, we can simplify the likelihood function to p(y ij |x ij , z j , α), which can be a deep network parameterized by α, taking both x ij and z j as inputs.

This contrasts with the previous formulation, where h α (z j ) produces all the weights of a network, yielding an extremely high dimensional representation and slow training.

For modeling q θj (z j |S j , α), we can use N (µ j , σ j ), where µ j and σ j can be learned individually for each task.

This, however limits the posterior family to express a single mode.

For more flexibility, we also explore the usage of more expressive posterior, such as Inverse Autoregressive Flow (IAF) BID19 or Neural Autoregressive Flow BID17 .

This gives a flexible tool for learning a rich variety of multivariate distributions.

In principle, we can use a different IAF for each task, but for memory and computational reasons, we use a single IAF for all tasks and we condition 4 on an additional task specific context c j .Note that with IAF, we cannot evaluate q θj (z j |S j , α) for any values of z efficiently, only for these which we just sampled, but this is sufficient for estimating the KL term with a Monte-Carlo approxi-mation i.e.: DISPLAYFORM0 It is common to approximate KL j with a single sample and let the mini-batch average the noise incurred on the gradient.

We experimented with n mc = 10, but this did not significantly improve the rate of convergence.

In order to compute the loss proposed in Equation 5, we would need to evaluate every sample of every task.

To accelerate the training, we use a Monte-Carlo approximation as is commonly done through the mini-batch procedure.

First we replace summations with expectations: DISPLAYFORM0 Now it suffices to approximate the gradient with n mb samples across all tasks.

Thus, we simply concatenate all datasets into a meta-dataset and added j as an extra field.

Then, we sample uniformly 5 n mb times with replacement from the meta-dataset.

Notice the term n j appearing in front of the likelihood in Equation 7, this indicates that individually for each task it finds the appropriate trade-off between the prior and the observations.

Refer to Algorithm 1 for more details on the procedure.1: for i in 1 .. n mb :2: sample x, y and j uniformly from the meta dataset 3: DISPLAYFORM1 5: DISPLAYFORM2 Calculating the loss for a mini-batch 3 EXTENDING TO 3 LEVEL OF HIERARCHIESDeep prior gives rise to a very flexible way to transfer knowledge from multiple tasks.

However, there is still an important assumption at the heart of deep prior (and other VAE-based approach such as BID10 ): the task information must be encoded in a low dimensional variable z. In Section 5, we show that it is appropriate for regression, but for image classification, it is not the most natural assumption.

Hence, we propose to extend to a third level of hierarchy by introducing a latent classifier on the obtained representation.

This provides a simple way to enhance existing algorithm such as Prototypical Networks (Proto Net) BID29 .

, for a given 6 task j, we decomposed the likelihood p(S|z) into n i=1 p(y i |x i , z) by assuming that the neural network is directly predicting p(y i |x i , z).

Here, we introduce a latent variable v to make the prediction p(y i |x i , v).

This can be, for example, a Gaussian linear regression on the representation φ α (x, z) produced by the neural network.

The general form now factorizes as follow: DISPLAYFORM0 , which is commonly called the marginal likelihood.

To compute ELBO j in 5 and update the parameters α, the only requirement is to be able to compute the marginal likelihood p(S|z).

There are closed form solutions for, e.g., linear regression with Gaussian prior, but our aim is to compare with algorithms such as Prototypical Networks on a 5 We also explored a sampling scheme that always make sure to have at least k samples from the same task.

The aim was to reduce gradient variance on task specific parameters but, we did not observed any benefits.

6 We removed j from equations to alleviate the notation.classification benchmark.

Alternatively, we can factor the marginal likelihood as follow p(S|z) = n i=1 p(y i |x i , S 0..i−1 , z).

If a well calibrated task uncertainty is not required, one can also use a leave-one-out procedure n i=1 p(y i |x i , S \ {x i , y i }, z).

Both of these factorizations correspond to training n times the latent classifier on a subset of the training set and evaluating on a sample left out.

We refer the reader to Rasmussen (2004, Chapter 5) for a discussion on the difference between leave-one-out cross-validation and marginal likelihood.

For a practical algorithm, we propose a closed form solution for leave-one-out in prototypical networks.

In its standard form, the prototypical network produces a prototype c k by averaging all representations γ i = φ α (x i , z) of class k i.e. c k = 1 |K| i∈K γ i , where K = {

i : y i = k}. Then, predictions are made using p(y = k|x, α, z) ∝ exp (− c k − γ i 2 ).

k ∀k be the prototypes computed without example x i , y i in the training set.

Then, DISPLAYFORM0 We defer the proof to supplementary materials.

Hence, we only need to compute prototypes once and rescale the Euclidean distance when comparing with a sample that was used for computing the current prototype.

This gives an efficient algorithm with the same complexity as the original one and a good proxy for the marginal likelihood.

Hierarchical Bayes algorithms for multitask learning has a long history BID7 BID35 BID1 .

However most of the literature focuses on simple statistical models and does not consider transferring on new tasks.

More recently, BID10 and BID5 explore hierarchical Bayesian inference with neural networks and evaluate on new tasks.

Both papers use a two-level Hierarchical VAE for modeling the observations.

While similar, our approach differs in a few different ways.

We use a discriminative approach and focus on model uncertainty.

We show that we can obtain a posterior on z without having to explicitly encode S j .

We also explore the usage of more complex posterior family such as IAF.

these differences make our algorithm simpler to implement, and easier to scale to larger datasets.

Other works consider neural networks with latent variables BID32 BID9 BID33 but does not explore the ability to learn across multiple tasks.

Some recent works on meta-learning are also targeting transfer learning from multiple tasks.

ModelAgnostic Meta-Learning (MAML) BID11 ) finds a shared parameter θ such that for a given task, one gradient step on θ using the training set will yield a model with good predictions on the test set.

Then, a meta-gradient update is performed from the test error through the one gradient step in the training set, to update θ.

This yields a simple and scalable procedure which learns to generalize.

Recently BID14 considers a Bayesian version of MAML.

Additionally, BID28 ) also consider a meta-learning approach where an encoding network reads the training set and generates the parameters of a model, which is trained to perform well on the test set.

Finally, some recent interest in few-shot learning give rise to various algorithms capable of transferring from multiple tasks.

Many of these approaches BID34 BID29 find a representation where a simple algorithm can produce a classifier from a small training set.

BID2 use a neural network pre-trained on a standard multi-class dataset to obtain a good representation and use classes statistics to transfer prior knowledge to new classes.

Through experiments, we want to answer i) Can deep prior learn a meaningful prior on tasks?

ii) Can it compete against state of the art on a strong benchmark? iii) In which situations does deep prior and other approaches fail?

To gain a good insight into the behavior of the prior and posterior, we choose a collection of one dimensional regression tasks.

We also want to test the ability of the method to learn the task and not just match the observed points.

For this, we will use periodic functions and test the ability of the regressor to extrapolate outside of its domain.

Specifically, each dataset consists of (x, y) pairs (noisily) sampled from a sum of two sine waves with different phase and amplitude and a frequency ratio of 2: f (x) = a 1 sin(ω·x+b 1 )+a 2 sin(2·ω·x+b 2 ), where y ∼ N (f (x), σ 2 y ).

We construct a meta-training set of 5000 tasks, sampling ω ∼ U(5, 7), (b 1 , b 2 ) ∼ U(0, 2π) 2 and (a 1 , a 2 ) ∼ N (0, 1) 2 independently for each task.

To evaluate the ability to extrapolate outside of the task's domain, we make sure that each task has a different domain.

Specifically, x values are sampled according to N (µ x , 1), where µ x is sample from the meta-domain U (−4, 4) .

The number of training samples ranges from 4 to 50 for each task and, evaluation is performed on 100 samples from tasks never seen during training.

Model Once z is sampled from IAF, we simply concatenate it with x and use 12 densely connected layers of 128 neurons with residual connections between every other layer.

The final layer linearly projects to 2 outputs µ y and s, where s is used to produce a heteroskedastic noise, σ y = sigmoid(s) · 0.1 + 0.001.

Finally, we use p(y|x, z) = N (µ y (x, z), σ y (x, z)2 ) to express the likelihood of the training set.

To help gradient flow, we use ReLU activation functions and Layer Normalization 7 BID0 .Results Figure 1a depicts examples of tasks with 1, 2, 8, and 64 samples.

The true underlying function is in blue while 10 samples from the posterior distributions are faded in the background.

The thickness of the line represent 2 standard deviations.

The first plot has only one single data point and mostly represents samples from the prior, passing near this observed point.

Interestingly, all samples are close to some parametrization of Equation 5.1.

Next with only 2 points, the posterior is starting to predict curves highly correlated with the true function.

However, note that the uncertainty is over optimistic and that the posterior failed to fully represent all possible harmonics fitting these two points.

We discuss this issue more in depth in supplementary materials.

Next, with 8 points, it managed to mostly capture the task, with reasonable uncertainty.

Finally, with 64 points the model is certain of the task.

To add a strong baseline, we experimented with MAML BID11 .

After exploring a variety of values for hyper-parameter and architecture design we couldn't make it work for our two harmonics meta-task.

We thus reduced the meta-task to a single harmonic and reduced the base frequency range by a factor of two.

With these simplifications, we managed to make it converge, but the results are far behind that of deep prior even in this simplified setup.

Figure 1b shows some form of adaptation with 16 samples per task but the result is jittery and the extrapolation capacity is very limited.

these results were obtained with a densely connected network of 8 hidden layers of 64 units 8 , with residual connections every other layer.

The training is performed with two gradient steps and the evaluation with 5 steps.

To make sure our implementation is valid, we first replicated their regression result with a fixed frequency as reported in BID11 .Finally, to provide a stronger baseline, we remove the KL regularizer of deep prior and reduced the posterior q θj (z j |S j , α) to a deterministic distribution centered on µ j .

The mean square error is reported in Figure 2 for an increasing dataset size.

This highlights how the uncertainty provided by deep prior yields a systematic improvement.

BID34 proposed to use a subset of Imagenet to generate a benchmark for few-shot learning.

Each task is generated by sampling 5 classes uniformly and 5 training samples per class, the remaining images from the 5 classes are used as query images to compute accuracy.

The number of unique classes sums to 100, each having 600 examples of 84 × 84 images.

To perform meta-validation and meta-test on unseen tasks (and classes), we isolate 16 and 20 classes respectively from the original The baseline corresponds to the same model without the KL regularizer.

Each value is averaged over 100 tasks and 10 different restart.

right: 4 sample tasks from the Synbols dataset.

Each row is a class and each column is a sample from the classes.

In the 2 left tasks, the symbol have to be predicted while in the two right tasks, the font has to be predicted.

set of 100, leaving 64 classes for the training tasks.

This follows the procedure suggested in BID28 .

The training procedure proposed in Section 2 requires training on a fixed set of tasks.

We found that 1000 tasks yields enough diversity and that over 9000 tasks, the embeddings are not being visited often enough over the course of the training.

To increase diversity during training, the 5 × 5 training and test sets are re-sampled every time from a fixed train-test split of the given task 9 .We first experimented with the vanilla version of deep prior (2).

In this formulation, we use a ResNet BID15 network, where we inserted FILM layers BID26 between each residual block to condition on the task.

Then, after flattening the output of the final convolution layer and reducing to 64 hidden units, we apply a 64 × 5 matrix generated from a transformation of z. Finally, predictions are made through a softmax layer.

We found this architecture to be slow to train as the generated last layer is noisy for a long time and prevent the rest of the Matching Networks BID34 60.0 % Meta-Learner BID28 60.6 % MAML BID11 63.2% Prototypical Networks BID29 68.2 % SNAIL BID24 68.9 % Discriminative k-shot BID2 73.9 % adaResNet BID25 71.9 % Deep Prior (Ours) 62.7 % Deep Prior + Proto Net (Ours) 74.5 % 68.6 ± 0.5% 69.6 ± 0.8% + ResNet (12) 72.4 ± 1.0% 76.8 ± 0.4% + Conditioning 72.3 ± 0.6% 80.1 ± 0.9% + Leave-One-Out 73.9 ± 0.4% 82.7 ± 0.2% + KL 74.5 ± 0.5% 83.5 ± 0.4% Table 2 : Ablation Study of our model.

Accuracy is shown with 90% confidence interval over bootstrap of the validation set.network to learn.

Nevertheless, we obtained 62.6% accuracy on Mini-Imagenet, on par with many strong baselines.

To enhance the model, we combine task conditioning with prototypical networks as proposed in Section 3.

This approach alleviates the need to generate the final layer of the network, thus accelerating training and increasing generalization performances.

While we no longer have a well calibrated task uncertainty, the KL term still acts as an effective regularizer and prevents overfitting on small datasets 10 .

With this improvement, we are now the new state of the art with 74.5% TAB0 .

In Table 2 , we perform an ablation study to highlight the contributions of the different components of the model.

In sum, a deeper network with residual connections yields major improvements.

Also, task conditioning does not yield improvement if the leave-one-out procedure is not used.

Finally, the KL regularizer is the final touch to obtain state of the art.

In Section 5.2, we saw that conditioning helps, but only yields a minor improvement.

This is due to the fact that Mini-Imagenet is a very homogeneous collection of tasks where a single representation is sufficient to obtain good results.

To support this claim, we provide a new benchmark 11 of synthetic symbols which we refer to as Synbols.

Images are generated using various font family on different alphabets (Latin, Greek, Cyrillic, Chinese) and background noise (Figure 2, right) .

For each task we have to predict either a subset of 4 font families or 4 symbols with only 4 examples.

Predicting either fonts or symbols with two separate Prototypical Networks, yields 84.2% and 92.3% accuracy respectively, with an average of 88.3%.

However, blending the two collections of tasks in a single benchmark, brings prototypical network down to 76.8%.

Now, conditioning on the task with deep prior brings back the accuracy to 83.5%.

While there is still room for improvement, this supports the claim that a single representation will only work on homogeneous collection of tasks and that task conditioning helps learning a family of representations suitable for heterogeneous benchmarks.

Using a variational Bayes framework, we developed a scalable algorithm for hierarchical Bayesian learning of neural networks, called deep prior.

This algorithm is capable of transferring information from tasks that are potentially remarkably different.

Results on the Harmonics dataset shows that the learned manifold across tasks exhibits the properties of a meaningful prior.

Finally, we found that MAML, while very general, will have a hard time adapting when tasks are too different.

Also, we found that algorithms based on a single image representation only works well when all tasks can succeed with a very similar set of features.

Together, these findings allowed us to reach the state of the art on Mini-Imagenet.

7.1 PROOF OF LEAVE-ONE-OUT Theorem 1.

Let c −i k ∀k be the prototypes computed without example x i , y i in the training set.

Then, DISPLAYFORM0 Proof.

Let γ i = φ α (x i ), n = |K| and assume y i = k then, DISPLAYFORM1 When y i = k, the result is trivially γ i − c When experimenting with the Harmonics toy dataset in Section 5.1, we observed issues with repeatability, most likely due to local minima.

We decided to investigate further on the multimodality of posterior distributions with small sample size and the capacity of IAF to model them.

For this purpose we simplified the problem to a single sine function and removed the burden of learning the prior.

The likelihood of the observations is defined as follows: DISPLAYFORM2 f (x) = sin(5(ω · x + b)); y ∼ N (f (x), σ 2 y ), where σ y = 0.1 is given and p(ω) = p(b) = N (0, 1).

Only the frequency ω and the bias b are unknown 12 , yielding a bi-dimensional problem that is easy to visualize and quick to train.

We use a dataset of 2 points at x = 1.5 and x = 3 and the corresponding posterior distribution is depicted in FIG1 -middle, with an orange point at the location of the true underlying function.

Some samples from the posterior distribution can be observed in FIG1 -top.

We observe a high amount of multi-modality on the posterior distribution FIG1 .

Some of the modes are just the mirror of another mode and correspond to the same functions e.g. b + 2π or −f ; b + π.

But most of the time they correspond to different functions and modeling them is crucial for some application.

The number of modes varies a lot with the choice of observed dataset, ranging from a few to several dozens.

Now, the question is: "How many of those modes can IAF model?".

Unfortunately, FIG1 -bottom reveals poor capability for this particular case.

After carefully adjusting the hyperparameters 13 of IAF, exploring different initialization schemes and running multiple restarts, we rarely capture more than two modes (sometimes 4).

Moreover, it will not be able to fully separate the two modes.

There is systematically a thin path of density connecting each modes as a chain.

With longer training, the path becomes thinner but never vanishes and the magnitude stays significant.

@highlight

A scalable method for learning an expressive prior over neural networks across multiple tasks.

@highlight

The paper presents a method for training a probabilistic model for Multitasks Transfer Learning by introducing a latent variable per task to capture the commonality in the task instances.

@highlight

The work proposes a variational approach to meta-learning that employs latent variables corresponding to task-specific datasets.

@highlight

Aims to learn a prior over neural networks for multiple tasks. 