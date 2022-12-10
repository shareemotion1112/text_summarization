Meta-learning, or learning-to-learn, has proven to be a successful strategy in attacking problems in supervised learning and reinforcement learning that involve small amounts of data.

State-of-the-art solutions involve learning an initialization and/or learning algorithm using a set of training episodes so that the meta learner can generalize to an evaluation episode quickly.

These methods perform well but often lack good quantification of uncertainty, which can be vital to real-world applications when data is lacking.

We propose a meta-learning method which efficiently amortizes hierarchical variational inference across tasks, learning a prior distribution over neural network weights so that a few steps of Bayes by Backprop will produce a good task-specific approximate posterior.

We show that our method produces good uncertainty estimates on contextual bandit and few-shot learning benchmarks.

Deep learning has achieved success in domains that involve a large amount of labeled data BID26 or training samples BID23 BID30 .

However, a key aspect of human intelligence is our ability to learn new concepts from only a few experiences.

It has been hypothesized that this skill arises from accumulating prior knowledge and using it appropriately in new settings BID19 .Meta learning attempts to endow machine learning models with the same ability by training a metalearner to perform well on a distribution of training tasks.

The meta-learner is then applied to an unseen task, usually assumed to be drawn from a task distribution similar to the one used for training, with the hope that it can learn to solve the new task efficiently.

Popular meta-learning methods have advanced the state-of-the-art in many tasks, including the few-shot learning problem, where the model has to learn a new task given a small training set containing as few as one example per class.

Though performance on few-shot learning benchmarks has greatly increased in the past few years, it is unclear how well the associated methods would perform in real-world settings, where the relationship between training and evaluation tasks could be tenuous.

For success in the wild, in addition to good predictive accuracy, it is also important for meta-learning models to have good predictive uncertainty -to express high confidence when a prediction is likely to be correct but display low confidence when a prediction could be unreliable.

This type of guarantee in predictive ability would allow appropriate human intervention when a prediction is known to have high uncertainty.

Bayesian methods offer a principled framework to reason about uncertainty, and approximate Bayesian methods have been used to provide deep learning models with accurate predictive uncertainty BID7 BID20 .

By inferring a posterior distribution over neural network weights, we can produce a posterior predictive distribution that properly indicates the level of confidence on new unseen examples.

Accordingly, we consider meta-learning under a Bayesian view in order to transfer the aforementioned benefits to our setting.

Specifically, we extend the work of BID0 , who considered hierarchical variational inference for meta-learning.

The work primarily dealt with PAC-Bayes bounds in meta-learning and the experiments consisted of data with tens of training episodes and small networks.

In this paper, we show how the meta-learning framework of BID5 can be used to efficiently amortize variational inference for the Bayesian model of BID0 in order to combine the former's flexibility and scalability with the latter's uncertainty quantification.

DISPLAYFORM0

We first start by reviewing the hierarchical variational bayes formulation used in BID0 for meta-learning.

Assume we observe data from M episodes, where the i th episode consists of data DISPLAYFORM0 .

We assume a hierarchical model with global latent variable θ and episode-specific variables φ i , i = 1, . . .

M (see FIG0 ).Hierarchical variational inference can then be used to lower bound the likelihood of the data: DISPLAYFORM1 Amit & Meir (2018) solve this optimization problem via mini-batch gradient descent on the objective starting from random initialization for all variational parameters.

They maintain distinct variational parameters λ i for each episode i, each of which indexes a distribution over episode-specific weights q(φ i ; λ i ).

While they only consider problems with at most 10 or so training episodes and where each φ i is small (the weights of a 2-layer convolutional network), this approach is not scalable to problems with large numbers of episodes -such as few-shot learning, where we can generate millions of episodes by randomizing over classes and examples -and requiring deep networks.3 AMORTIZED BAYESIAN META-LEARNING 3.1 SCALING META-LEARNING WITH AMORTIZED VARIATIONAL INFERENCE Learning local variational parameters λ i for a large number of episodes M becomes difficult as M grows due to the costs of storing and computing each λ i .

These problems are compounded when each φ i is the weight of a deep neural network and each λ i are variational parameters of the weight distribution (such as a mean and standard deviation of each weight).

Instead of maintaining M different variational parameters λ i indexing distributions over neural network weights φ i , we compute λ i on the fly with amortized variational inference (AVI), where a global learned model is used to predict λ i from D i .

A popular use of AVI is training a variational autoencoder BID16 , where a trained encoder network produces the variational parameters for each data point.

Rather than training an encoder to predict λ i given the episode, we show that inference can be amortized by finding a good initialization, a la MAML BID5 .

We represent the variational parameters for each episode as the output of several steps of gradient descent from a global initialization.

DISPLAYFORM2 ) be the part of the objective corresponding to data D i .

Let the procedure SGD K (D, λ (init) , θ) represent the variational parameters produced after K steps of gradient descent on the objective L D (λ, θ) with respect to λ starting at the initialization λ (0) = λ (init) and where θ is held constant i.e.: DISPLAYFORM3 We represent the variational distribution for each dataset q θ (φ i |D i ) in terms of the local variational parameters λ i produced after K steps of gradient descent on the loss for dataset D i , starting from the global initialization θ: DISPLAYFORM4 .

Note that θ here serves as both the global initialization of local variational parameters and the parameters of the prior p(φ | θ).

We could pick a separate prior and global initialization, but we found tying the prior and initialization did not seem to have a negative affect on performance, while significantly reducing the number of total parameters necessary.

With this form of the variational distribution, this turns the optimization problem of (2) into DISPLAYFORM5 Because each q θ (φ i |D i ) depends on ψ via θ (the initialization for the variational parameters before performing K steps of gradient descent), we can also backpropagate through the computation of q via the gradient descent process to compute updates for ψ.

Though this backpropagation step requires computing the Hessian, it can be done efficiently with fast Hessian-vector products, which have been used in past work involving backpropagation through gradient updates BID21 BID15 .

This corresponds to learning a global initialization of the variational parameters such that a few steps of gradient descent will produce a good local variational distribution for any given dataset.

We assume a setting where M >> N , i.e. we have many more episodes than data points within each episode.

Accordingly, we are most interested in quantifying uncertainty within a given episode and desire accurate predictive uncertainty in q θ (φ i |D i ).

We assume that uncertainty in the global latent variables θ should be low due to the large number of episodes, and therefore use a point estimate for the global latent variables, letting q(θ; ψ) be a dirac delta function q(θ) = 1{θ = θ * }.

This removes the need for global variational parameters ψ and simplifies our optimization problem to arg min DISPLAYFORM6 where θ * is the solution to the above optimization problem.

Note that KL(q(θ) p(θ)) term can be computed even when q(θ) = 1{θ = θ * }, as KL(q(θ)||p(θ)) = − log p(θ * ).

In the few-shot learning problem, we must consider train and test splits for each dataset in each episode.

Using notation from previous work on few-shot learning Snell et al. (2017) , we will call the training examples in each dataset the support set and the test examples in each dataset the query DISPLAYFORM0 , and the assumption is that during evaluation, we are only given D DISPLAYFORM1 to determine our variational distribution q(φ i ) and measure the performance of the model by evaluating the variational distribution on corresponding D (Q)i .

In order to match what is done during training and evaluation, we consider a modified version of the objective of (4) that incorporates this support and query set split.

This means that for each episode i, we only have access to data D (S) i to compute the variational distribution, giving us the following objective: DISPLAYFORM2 where DISPLAYFORM3 i , θ, θ .

Note that the objective in this optimization problem still serves as a lower bound to the likelihood of all the episodic data because all that has changed is that we condition the variational distribution q on less information (using only the support set vs using the entire dataset).

Conditioning on less information potentially gives us a weaker lower bound for all the training datasets, but we found empirically that the performance during evaluation was better using this type of conditioning since there is no mismatch between how the variational distribution is computed during training vs evaluation.

With the objective (5) in mind, we now give details on how we implement the specific model.

We begin with the distributional forms of the priors and posteriors.

The formulation given above is flexible but we consider fully factorized Gaussian distributions for ease of implementation and experimentation.

We let θ = {µ θ , σ 2 θ }, where µ θ ∈ R D and σ 2 θ ∈ R D represent the mean and variance for each neural network weight, respectively.

Then, p(φ i |θ) is DISPLAYFORM0 is the following: DISPLAYFORM1 We let the prior p(θ) be DISPLAYFORM2 is the precision and a 0 and b 0 are the alpha and beta parameters for the gamma distribution.

Note that with the defined distributions, the SGD process here corresponds to performing Bayes by Backprop BID2 with the learned prior p(φ i |θ).Optimization of FORMULA10 is done via mini-batch gradient descent, where we average gradients over multiple episodes at a time.

The pseudo-code for training and evaluation is given in Algorithms 1 and 2 in the appendix.

The KL-divergence terms are calculated analytically whereas the expectations are approximated by averaging over a number of samples from the approximate posterior, as has been done in previous work BID16 BID2 .

The gradient computed for this approximation naively can have high variance, which can significantly harm the convergence of gradient descent BID17 .

Variance reduction is particularly important to the performance of our model as we perform stochastic optimization to obtain the posterior q θ φ|D (S) at evaluationtime also.

Previous work has explored reducing the variance of gradients involving stochastic neural networks, and we found this crucial to training the networks we use.

Firstly, we use the Local Reparametrization Trick (LRT) BID17 for fully-connected layers and Flipout BID35 for convolutional layers to generate fully-independent (or close to fully-independent in the case of Flipout) weight samples for each example.

Secondly, we can easily generate multiple weight samples in the few-shot learning setting simply by replicating the data in each episode since we only have a few examples per class making up each episode.

Both LRT and Flipout increase the operations required in the forward pass by 2 because they require two weight multiplications (or convolutions) rather than one for a normal fully-connected or convolutional layer.

Replicating the data does not increase the run time too much because the total replicated data still fits on a forward pass on the GPU.

Meta-learning literature commonly considers the meta-learning problem as either empirical risk minimization (ERM) or bayesian inference in a hierarchical graphical model.

The ERM perspective involves directly optimizing a meta learner to minimize a loss across training datasets BID29 .

Recently, this has been successfully applied in a variety of models for few-shot learning BID5 BID31 BID22 .

The other perspective casts meta-learning as bayesian inference in a hierarchical graphical model BID32 BID4 BID18 .

This approach provides a principled framework to reason about uncertainty.

However, hierarchical bayesian methods once lacked the ability to scale to complex models and large, high-dimensional datasets due to the computational costs of inference.

Recent developments in variational inference BID16 BID2 allow efficient approximate inference with complex models and large datasets.

These have been used to scale bayesian meta-learning using a variety of approaches.

BID3 infer episode-specific latent variables which can be used as auxillary inputs for tasks such as classification.

As mentioned before, BID0 learn a prior on the weights of a neural network and separate variational posteriors for each task.

Our method is very closely related to BID5 and recent work proposing Bayesian variants of MAML.

BID11 provided the first Bayesian variant of MAML using the Laplace approximation.

In concurrent work to this paper, BID14 and propose Bayesian variants of MAML with different approximate posteriors.

approximate MAP inference of the task-specific weights φ i , and maintain uncertainty only in the global model θ.

Our paper, however, considers tasks in which it is important to quantify uncertainty in task-specific weights -such as contextual bandits and few-shot learning.

BID14 focus on uncertainty in task-specific weights, as we do.

They use a point estimate for all layers except the final layer of a deep neural network, and use Stein Variational Gradient Descent to approximate the posterior over the weights in the final layer with an ensemble.

This avoids placing Gaussian restrictions on the approximate posterior; however, the posterior's expressiveness is dependant on the number of particles in the ensemble, and memory and computation requirements scale linearly and quadratically in the size of the ensemble, respectively.

The linear scaling requires one to share parameters across particles in order to scale to larger datasets.

Moreover, there has been other recent work on Bayesian methods for few-shot learning.

Neural Processes achieve Gaussian Process-like uncertainty quantification with neural networks, while being easy to train via gradient descent BID8 BID19 .

However, it has not been demonstrated whether these methods can be scaled to bigger benchmarks like miniImageNet.

BID10 adapt Bayesian decision theory to formulate the use of an amortization network to output the variational distribution over weights for each few-shot dataset.

Both BID14 and BID10 require one to specify the global parameters (those that are shared across all episodes and are point estimates) vs task-specific parameters (those that are specific to each episode and have a variational distribution over them).

Our method, however, does not require this distinction a priori and can discover it based on the data itself.

For example, in FIG4 , which shows the standard deviations of the learned prior, we see that many of the 1 st layer convolutional kernels have standard deviations very close to 0, indicating that these weights are essentially shared because there will be a large penalty from the prior for deviating from them in any episode.

Not needing to make this distinction makes it more straightforward to apply our model to new problems, like the contextual bandit task we consider.

We evaluate our proposed model on experiments involving contextual bandits and involving measuring uncertainty in few-shot learning benchmarks.

We compare our method primarily against MAML.

Unlike our model, MAML is trained by maximum likelihood estimation of the query set given a fixed number of updates on the support set, causing it to often display overconfidence in the settings we consider.

For few-shot learning, we additionally compare against Probabilistic MAML , a Bayesian version of MAML that maintains uncertainty only in the global parameters.

The first problem we consider is a contextual bandit task, specifically in the form of the wheel bandit problem introduced in BID28 .

The contextual bandit task involves observing a context X t from time t = 0, . . .

, n and requires the model to select, based on its internal state and X t , one of the k available actions.

Based on the context and the action selected at each time step, a reward is generated.

The goal of the model is to minimize the cumulative regret, the difference between the sum of rewards of the optimal policy and the model's policy.

The wheel bandit problem is a synthetic contextual bandit problem with a scalar hyperparameter that allows us to control the amount of exploration required to be successful at the problem.

The setup is the following: we consider a unit circle in R 2 split up into 5 areas determined by the hyperparameter δ.

At each time step, the agent is given a point X = (x 1 , x 2 ) inside the circle and has to determine which arm to select among k = 5 arms.

For X ≤ δ (the low-reward region), the optimal arm is k = 1, which gives reward r ∼ N (1.2, 0.01 2 ).

All other arms in this area give reward r ∼ N (1, 0.01 2 ).

For X > δ, the optimal arm depends on which of the 4 high-reward regions X is in.

Each of the 4 regions has an assigned optimal arm that gives reward r ∼ N (50, 0.01 2 ), whereas the other 3 arms will give r ∼ N (1.0, 0.01 2 ) and arm k = 1 will always give r ∼ N (1.2, 0.01 2 ).

The difficulty of the problem increases with δ, as it requires increasing amount of exploration to determine where the high-reward regions are located.

We refer the reader to BID28 for visual examples of the problem.

Thompson Sampling BID33 ) is a classic approach to tackling the exploration-exploitation trade-off involved in bandit problems which requires a posterior distribution over reward functions.

At each time step an action is chosen by sampling a model from the posterior and acting optimally with respect to the sampled reward function.

The posterior distribution over reward functions is then updated based on the observed reward for the action.

When the posterior initially has high variance because of lack of data, Thompson Sampling explores more and turns to exploitation only when the posterior distribution becomes more certain about the rewards.

The work of BID28 compares using Thompson Sampling for different models that approximate the posterior over reward functions on a variety of contextual bandit problems, including the wheel bandit.

We use the setup described in BID9 to apply meta-learning methods to the wheel bandit problem.

Specifically, for meta-learning methods there is a pre-training phase in which training episodes consist of randomly generated data across δ values from wheel bandit task.

Then, these methods are evaluated using Thompson sampling on problems defined by specific values of (2018) ) and results shown are mean and standard error for cumulative regret calculated across 50 trials δ.

We can create a random training episode for pre-training by first sampling M different wheel DISPLAYFORM0 , δ i ∼ U(0, 1), followed by sampling tuples of the form {(X, a, r)} N j=1 for context X, action a, and observed reward r. As in BID9 , we use M = 64 and N = 562 (where the support set has 512 items and the query set has 50 items).

We then evaluate the trained meta-learning models on specific instances of the wheel bandit problem (determined by setting the δ hyperparameter).

Whereas the models in BID28 have no prior knowledge to start off with when being evaluated on each problem, meta-learning methods, like our model and MAML, have a chance to develop some sort of prior that they can utilize to get a head start.

MAML learns a initialization of the neural network that it can then fine-tune to the given problem data, whereas our method develops a prior over the model parameters that can be utilized to develop an approximate posterior given the new data.

Thus, we can straightforwardly apply Thompson sampling in our model using the approximate posterior at each time step whereas for MAML we just take a greedy action at each time step given the current model parameters.

The results of evaluating the meta-learning methods using code made available by authors of BID28 after the pre-training phase are shown in Table 1 .

We also show results from NeuralLinear, one of the best performing models from BID28 , to display the benefit of the pretraining phase for the meta-learning methods.

We vary the number of contexts and consider n = 80, 000 (which was used in BID28 ) and n = 2, 000 (to see how the models perform under fewer time steps).

We can see that as δ increases and more exploration is required to be successful at the problem, our model has a increasingly better cumulative regret when compared to MAML.

Additionally, we notice that this improvement is even larger when considering smaller amount of time steps, indicating that our model converges to the optimal actions faster than MAML.

Lastly, in order to highlight the difference between our method and MAML, we visualize the learned prior p(φ | θ) in FIG1 by showing the expectation and standard-deviation of predicted rewards for specific arms with respect to the prior.

We can see that the standard deviation of the central low-reward arm is small everywhere, as there is reward little variability in this arm across δ values.

For the high-reward arm in the upper-right corner, we see that the standard deviation is high at the edges of the area in which this arm can give high reward (depending on the sampled δ value).

This variation is useful during exploration as this is the region in which we would like to target our exploration to figure out what δ value we are faced with in a new problem.

MAML is only able to learn the information associated with expected reward values and so is not well-suited for appropriate exploration but can only be used in a greedy manner.

We consider two few-shot learning benchmarks: CIFAR-100 and miniImageNet, where both datasets consist of 100 classes and 600 images per class and where CIFAR-100 has images of size 32 × 32 and miniImageNet has images of size 84 × 84.

We split the 100 classes into separate sets of 64 classes for training, 16 classes for validation, and 20 classes for testing for both of the datasets (using the split from BID27 for miniImageNet, while using our own for CIFAR-100 as a commonly used split does not exist).

For both benchmarks, we use the convolutional architecture Table 2 : Few-shot classification accuracies with 95% confidence intervals on CIFAR-100 and miniImageNet.used in BID5 , which consists of 4 convolutional layers, each with 32 filters, and a fully-connected layer mapping to the number of classes on top.

For the few-shot learning experiments, we found it necessary to downweight the inner KL term for better performance in our model.

While we focus on predictive uncertainty, we start by comparing classification accuracy of our model compared to MAML.

We consider 1-shot, 5-class and 1-shot, 10-class classification on CIFAR-100 and 1-shot, 5-class classification on miniImageNet, with results given in Table 2 .

For both datasets, we compare our model with our own re-implementation of MAML and Probabilistic MAML.

Note that the accuracy and associated confidence interval for our implementations for miniImageNet are smaller than the reference implementations because we use a bigger query set for test episodes (15 vs 1 example(s) per class) and we average across more test episodes (1000 vs 600), respectively, compared to BID5 .

Because we evaluate in a transductive setting BID25 , the evaluation performance is affected by the query set size, and we use 15 examples to be consistent with previous work BID27 .

Our model achieves comparable to a little worse on classification accuracy than MAML and Probabilistic MAML on the benchmarks.

To measure the predictive uncertainty of the models, we first compute reliability diagrams BID12 across many different test episodes for both models.

Reliability diagrams visually measure how well calibrated the predictions of a model are by plotting the expected accuracy as a function of the confidence of the model.

A well-calibrated model will have its bars align more closely with the diagonal line, as it indicates that the probability associated with a predicted class label corresponds closely with how likely the prediction is to be correct.

We also show the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) of all models, which are two quantitative ways to measure model calibration BID24 BID12 .

ECE is a weighted average of each bin's accuracy-to-confidence difference whereas MCE is the worst-case bin's accuracy-to-confidence difference.

Reliability diagrams and associated error scores are shown in FIG2 .

We see that across different tasks and datasets, the reliability diagrams and error scores reflect the fact that our model is always better calibrated on evaluation episodes compared to MAML and Probabilitic MAML.Another way we can measure the quality of the predictive uncertainty of a model is by measuring its confidence on out-of-distribution examples from unseen classes.

This tests the model's ability to be uncertain on examples it clearly does not know how to classify.

One method to visually measure this is by plotting the empirical CDF of a model's entropies on these out-of-distribution examples BID20 .

A model represented by a CDF curve that is towards the bottom-right is preferred, as it indicates that the probability of observing a high confidence prediction from the model is low on an out-of-distribution example.

We can plot the same type of curve in our setting by considering the model's confidence on out-of-episode examples for each test episode.

Empirical CDF curves for both MAML-based models and our model are shown in FIG3 .

We see that in general our model computes better uncertainty estimates than the comparison models, as the probability of a low entropy prediction is always smaller.

Lastly, we visualize the prior distribution p(φ|θ) that has been learned in tasks involving deep convolutional networks.

We show the standard deviations of randomly selected filters from the first convolutional layer to the last convolutional layer from our CIFAR-100 network trained on 1-shot, 5-class task in FIG4 .

Interestingly, the standard deviation of the prior for the filters increases as we go higher up in the network.

This pattern reflects the fact that across the training episodes the prior can be very confident about the lower-level filters, as they capture general, useful lower-level features and so do not need to be modified as much on a new episode.

The standard deviation for the higher-level filters is higher, reflecting that fact that these filters need to be fine-tuned to the labels present in the new episode.

This variation in the standard deviation represents different learning speeds across the network on a new episode, indicating which type of weights are general and which type of weights need to be quickly modified to capture new data.

We described a method to efficiently use hierarchical variational inference to learn a meta-learning model that is scalable across many training episodes and large networks.

The method corresponds to learning a prior distribution over the network weights so that a few steps of Bayes by Backprop will produce a good approximate posterior.

Through various experiments we show that using a Bayesian interpretation allows us to reason effectively about uncertainty in contextual bandit and CIFAR-100: 1-shot, 5-class CIFAR-100: 1-shot, 10-class miniImageNet: 1-shot, 5-class few-shot learning tasks.

The proposed method is flexible and future work could involve considering more expressive prior (and corresponding posterior) distributions to further improve the uncertainty estimates.

In algorithms 1 and 2 we give the pseudocode for meta-training and meta-evaluation, respectively.

Note that in practice, we do not directly parameterize variance parameters but instead parameterize the standard deviation as the output of the softplus function as was done in BID2 so that it is always non-negative.

Table 4 : Hyperparameters for our model for few-shot learning experiments.

<|TLDR|>

@highlight

We propose a meta-learning method which efficiently amortizes hierarchical variational inference across training episodes.

@highlight

An adaptation to MAML-type models that accounts for posterior uncertainty in task specific latent variables by employing variational inference for task-specific parameters in a hierarchical Bayesian view of MAML.

@highlight

The authors consider meta-learning to learn a prior over neural network weights, done via amortized variational inference.