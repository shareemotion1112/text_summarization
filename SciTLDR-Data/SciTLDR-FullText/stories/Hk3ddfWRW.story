Recent advances in learning from demonstrations (LfD) with deep neural networks have enabled learning complex robot skills that involve high dimensional perception such as raw image inputs.

LfD algorithms generally assume learning from single task demonstrations.

In practice, however, it is more efficient for a teacher to demonstrate a multitude of tasks without careful task set up, labeling, and engineering.

Unfortunately in such cases, traditional imitation learning techniques fail to represent the multi-modal nature of the data, and often result in sub-optimal behavior.

In this paper we present an LfD approach for learning multiple modes of behavior from visual data.

Our approach is based on a stochastic deep neural network (SNN), which represents the underlying intention in the demonstration as a stochastic activation in the network.

We present an efficient algorithm for training SNNs, and for learning with vision inputs, we also propose an architecture that associates the intention with a stochastic attention module.

We demonstrate our method on real robot visual object reaching tasks, and show that it can reliably learn the multiple behavior modes in the demonstration data.

Video results are available at https://vimeo.com/240212286/fd401241b9.

A key problem in robotic control is to simplify the problem of programming a complex behavior.

Traditional control engineering approaches, which rely on accurate manual modeling of the system environment, are very challenging to apply in modern robotic applications where most sensory inputs come from images and other high-dimensional signals such as tactile feedback.

In contrast, imitation learning, or learning from demonstration (LfD) approaches BID31 aim to directly learn a control policy from mentor or expert demonstrations.

The key advantages of LfD are simplicity and data-efficiency, and indeed, LfD has been successfully used for learning complex robot skills such as locomotion BID32 , driving BID27 BID30 , flying BID0 , and manipulation BID21 BID4 BID25 .

Recently, advances in deep representation learning BID12 have facilitated LfD methods with high dimensional perception, such as mapping raw images directly to controls BID9 .

These advances are capable of learning generalizable skills BID18 , and offer a promising approach for modern industrial challenges such as pick and place tasks BID5 .One challenge in LfD, however, is learning different modes of the same task.

For example, consider learning to pick up an object from a pile.

The demonstrator can choose to pick up a different object each time, yet we expect LfD to understand that these are similar demonstrations of the same pick-up skill, only with a different intention in mind.

Moreover, we want the learned robot behavior to display a similar multi-modal 1 nature.

Standard approaches for LfD with image inputs, such as learning with deep neural networks (NNs) BID27 BID9 BID18 , are not suitable for learning multimodal behaviors.

In their essence, NNs learn a deterministic mapping from observation to control, which cannot represent the inherently multi-modal latent intention in the demonstrations.

In practice, this manifests as an 'averaging' of the different modes in the data BID3 , leading to an undesirable policy.

A straightforward approach for tackling the multi-modal problem in LfD is to add a label for each mode in the data.

Thus, in the pick-up task above, the demonstrator would also explicitly specify the object she intends to pick-up beforehand.

Such an approach has several practical shortcomings: it requires the demonstrator to record more data, and requires the possible intentions to be specified in advance, making it difficult to use the same recorded data for different tasks.

More importantly, such a solution is conceptually flawed -it solves an algorithmic challenge by placing additional burden on the client.

In this work, we propose an approach for LfD with multi-modal demonstrations that does not require any additional data labels.

Our method is based on a stochastic neural network model, which represents the latent intention as a random activation in the network.

We propose a novel and efficient learning algorithm for training stochastic networks, and present a network architecture suitable for LfD with raw image inputs, where the intention takes the form of a stochastic attention over features in the image.

We show that our method can reliably reproduce behavior with multiple intentions in real-robot object reaching tasks.

Moreover, in scenarios where multiple intentions exist in the demonstration data, the stochastic neural networks perform better than their deterministic counterparts.

In this work we focus on a direct imitation learning approach, known as behavioral cloning BID27 ).

This approach does not require any model of the task dynamics, and does not require additional queries of the mentor or robot execution rollouts beyond the collected demonstrations (though such can be used to improve performance BID30 ).

An alternative approach is inverse reinforcement learning (IRL) BID24 BID0 Ziebart et al., 2008) , where a reward model that explains the demonstrated behavior is sought.

Recently, modelfree IRL approaches that can learn complex behavior policies from high-dimensional data were proposed BID7 BID14 .

These approaches, however, rely on taking additional policy rollouts as a fundamental step of the method, which, in realistic robot applications, requires substantial resources.

Multi-task IRL learns from unlabeled demonstrations generated by varying intentions or objectives BID2 BID6 .

BID6 propose a Bayesian approach for inferring the intention of an agent performing a series of tasks in a dynamic environment.

BID2 propose an EM for clustering the unlabeled demonstrations and then application of IRL for inferring the intention of a given cluster.

Both approaches have been shown promising results on relatively simple low dimensional problems.

Several recent works on multi-task IRL BID13 Wang et al., 2017; BID19 extended the generative adversarial imitation learning (GAIL) algorithm BID14 to high dimensional multi-modal demonstrations.

Our approach, in comparison, does not require taking additional robot rollouts.

Recently, BID28 proposed a method for learning multi-modal policies from raw visual inputs, using mixture density networks BID3 for generating outputs from a mixture of Gaussians distribution.

Their training method requires labeling each task with a specific signal.

While the modes, or intentions in our work can be seen as different tasks, we do not require any labeling of the intention in the demonstrations.

To our knowledge, this is the first LfD approach that can handle multiple modes in the demonstrations and: (1) does not require additional robot rollouts, (2) does not require a label for the mode, and (3) can work with raw image inputs.

The stochastic neural network model we use here is related to recently proposed generative models such as (conditional) variational autoencoders (VAEs) BID17 Sohn et al., 2015) , and generative adversarial nets (GANs) BID11 BID20 .

We opted for stochastic neural networks since GANs are known to have problems learning multi-modal distributions BID1 , and conditional VAEs BID26 require training an additional encoding network, which proved to be difficult in our experimental domain.

Very recently, in the context of multi-modal video prediction, BID8 proposed the K-best loss for training stochastic neural networks, which is similar to our proposed training algorithm.

In that work, stochastic neural networks with K-best loss were also shown to outperform conditional VAEs on some domains.

Our contribution, compared to the work of BID8 , is providing a formal mathematical treatment of this method, proposing optimistic sampling which significantly improves its performance, and showing its importance in a real world robotic imitation learning domain.

We first introduce some preliminary concepts for presenting our methods, and then present our problem formulation of imitation learning with multi-modal behaviors in expert demonstrations.

Learning from Demonstration To explicitly formulate the problem of imitation learning, let X and U denote the observation and action spaces of the robot, and let x t ∈ X and u t ∈ U denote an observation and a control command for the robot at time t. Given a data-set D of N trajectories T i with length T (for simplifying notations we drop the subscript i in T i ), where a demonstrated task is recorded in the form of sequential pairs of observations and actions DISPLAYFORM0 , LfD aims to learn a policy P : X → P(U) that is parametrized by feature weight vector θ ∈ Θ, such that it reliably performs the task.

Here P(U) represents the space of probability distributions defined on the action space U. Since each observation is associated with an action label, the imitation learning policy can be found by solving the maximum-likelihood (ML) objective: DISPLAYFORM1 , where we abbreviated the sequence of actions as u 1:T .

= u 1 , . . . , u T , and the sequence of observations as x 1:T .

= x 1 , . . . , x T .

This objective function is the empirical average of the conditional log likelihood, which is a consistent estimator of the expected conditional log likelihood: E [log P (u 1:T |x 1:T , θ)].

If, for example, the policy is Gaussian with parametrized mean vector f (x; θ) and an identity co-variance matrix, then the above supervised learning problem is equivalent to an 2 regression problem with objective function DISPLAYFORM2 Stochastic Neural Networks Multilayer perceptrons (MLPs) are general purpose function approximators for nonlinear regression in feedforward neural networks (NNs) BID12 .

Parametrized by the NN weights θ, the output of the MLP, f (x; θ), is often interpreted as the sufficient statistics of the conditional probability P (u|x; θ), if the conditional probability belongs to the exponential family (conditioned on the input x).

For example, if P (u|x; θ) is parametrized as an isotropic Gaussian distribution, it can be represented by N (u|f (x; θ), I).

The parameters θ are typically learned by maximizing the expected log likelihood function.

However, since the MLP activation functions are all deterministic, by nature the model P (u|x; θ) is a unimodal distribution.

For many structured prediction problems, we are interested in a conditional distribution that is multimodal.

To satisfy the multi-modality requirement, a common approach is to make the hidden variables in the NN stochastic.

Sigmoid belief nets (SBNs) BID22 are an early example of this idea, using binary stochastic hidden variables.

However, inference in SBNs is generally intractable, and costly to compute in practice.

Recently, Tang & Salakhutdinov (2013) introduced the stochastic feedforward neural network (SNN) for modeling the multi-modal conditional distribution P (u|x; θ).

Unlike SBNs, SNNs add to the deterministic NN latent features a stochastic latent variable z, and decompose the conditional distribution as: P (u|x; θ) = z P (u|x, z; θ)P (z).

It is also assumed that P (u|x, z, θ) and P (z) can be easily computed, for example, as in (Tang & Salakhutdinov, 2013) , where z is represented by Bernoulli random variable nodes in the network.

For learning the parameters θ, Tang & Salakhutdinov (2013) proposed a generalized EM algorithm, where importance sampling is used in the E-step, and error back-propagation is used in the M-step.

Problem Formulation In this work, we consider an imitation learning setting, where the mentor demonstrations of a particular task consist of multiple behaviors.

In particular, we assume that the demonstrator can perform the task in several different strategies, which we term as intentions.

As a concrete example, consider the task of picking up an object from a pile of different objects in the scene (see FIG0 (a) for an example).

In this case, the data-set of mentor demonstrations consists of a list of trajectories, where the target object in each trajectory is inherently decided by the demonstrator, and not explicitly labeled.

Our goal is to learn a stochastic policy that accurately mimics the mentor's policy, and accurately displays the multiple intentions demonstrated in the data.

In this section, we first present our LfD formulation based on the SNN model for learning multiple intentions, and then propose a sampling based algorithm for learning the SNN parameters.

We then present a particular SNN architecture that is suitable for vision-based inputs, where the stochastic intention takes the form of an attention over image features.

We model the intention of the demonstrator using a random vector z ∈ R M with probability distribution P(z).

For example, P(z) could be a unit normal distribution N (0, I), or a vector of independent multinomial probabilities.

Here we assume that throughout a single trajectory, the intention does not change, and the intention is independent of the observations 2 .

Therefore, the conditional data likelihood is obtained by marginalizing out the random variable of intention: P (u 1:T |x 1:T ; θ) = z P (u 1:T |x 1:T , z; θ)P (z).

SNNs can be viewed as directed graphical models (see Figure 4 in the appendix for a diagram) where at each time t ∈ {1, . . .

, T }, the generative process starts from an observation x t , combines with a latent intention z, which is the same throughout the trajectory, and then generates an action u t .

We also make the standard assumption that, given the intention z at each trajectory, the demonstrator policy is memory-less (a.k.a.

Markov), which implies the following equality: DISPLAYFORM0 Given an intention z, we model the action probability as log P (u|z, x; θ) ∝ −d(f (x, z; θ), u), where f is a deterministic NN that takes as input both the observation x and the intention z, and d is some distance metric.

One immediate example is when d(a, b) = a − b 2 , one obtains P (u|z, x; θ) as a normal distribution N (u|f (x, z; θ), σ 2 ) for some MLP mean predictor f (x, z; θ) and constant variance term σ 2 .

In our experiments, we found the 1 distance function to work well.

Note that when the intention variable z is fixed, the output action follows a unimodal distribution.

However, since z is a random vector that is input to a nonlinear NN computation, the distribution of f (x, z; θ), and thereby the output distribution P (u 1:T |x 1:T ; θ), can take a multi-modal form.

We first describe a basic Monte Carlo (MC) sampling algorithm for learning the parameters θ ∈ Θ of the SNN model in Section 4.1.

Let z 1 , . . .

, z N denote N samples of z, where z i ∼ P (z).

For each given parameter θ, sequence of observations x 1:T , and sequence of actions u 1:T , let r(z; x 1:T , u 1:T , θ) = P (u 1:T |x 1:T , z; θ) be the reward function, which associates an intention with the data likelihood given it.

The reason we use this terminology is to later connect the likelihood maximization problem with risk-sensitive optimization concepts that will be key in our approach.

A Monte Carlo approximation of the likelihood is given by, DISPLAYFORM0 A direct approach for computing θ would be to directly maximize DISPLAYFORM1 log r(z i ; x 1:T , u 1:T , θ), which is a MC estimate of E z∼P [log r(z; x 1:T , u 1:T , θ)], with gradient-based optimization.

By Jensen's inequality, the above term is a lower bound of the data log likelihood log P (u 1:T |x 1:T ; θ), corresponding to a maximum-likelihood approach.

While the estimator of the gradient is unbiased and consistent to E z∼P [∇ θ log r(z; x 1:T , u 1:T , θ)], in practice, such an approach suffers from extremely high variance (with respect to intention z).

To justify this observation, consider the sum of probabilities in (1).

Since each sampled intention z i is given an equal weight in explaining the observed sequence of actions, even sampled intentions that are very different from the ones that generated the data (i.e., have high cost) are expected to produce a high likelihood.

To reduce variance in training SNNs, in this section we introduce a sampling strategy whose gradient updates focuses only on the most correct underlying intentions -the intentions that have the highest reward.

We analyze the bias and variance trade-offs of this new sampling gradient estimate, and show that the variance of our proposed approximation is lower than that of a naive MC approach.

For any given threshold α ∈ [0, 1], let q α (θ) denote the (upper) α−quantile of the reward function q α (θ) .

= max w z:r(z;x 1:T ,u 1:T ,θ)≥w P (z) ≤ α .

This associates with the α−quantile of the underlying intention with highest likelihood probability.

We define an intention-driven probability distribution as follows: DISPLAYFORM0 This quantity can be interpreted as a weighted distribution that only samples from the α% of the most correct underlying intentions (i.e., the intentions that best explain the mentor demonstrations).

The expected reward induced by the intention-driven distribution is given by: DISPLAYFORM1 which is equal to the conditional likelihood function of the α% most correct intentions.

In the financial risk literature, this metric is known as the expected shortfall BID29 , and is typically used to evaluate the risky tail distribution of financial assets.

While our setting is completely different, we will use tools developed for expected shortfall estimation in our approach.

By definition of Q α , one has the following inequality.

DISPLAYFORM2 We propose to maximize E z∼Qα [r(z; x 1:T , u 1:T , θ)] using Monte Carlo sampling techniques.

Intuitively, since the support of Q α is limited to the most likely z values, estimating it using sampling has lower variance than estimating the original likelihood.

We will further elaborate on this point technically later in the section.

However, this comes at the cost of adding a bias z:Qα(z)=0 P (u 1:T |x 1:T , z; θ)P (z).

Empirically, we have found this procedure to work well.

To sample from Q α , we use empirical quantile estimation BID10 .

Let z ord 1 , . . .

, z ord N denote the MC samples z 1 , . . .

, z N sorted in descending order, according to the reward function r(z; x 1:T , u 1:T , θ).

Let N α = αN be the number of samples corresponding to the α−quantile.

Then we have the following empirical estimate: DISPLAYFORM3 .

It has been shown in Theorem 1 of BID10 that under standard assumptions, the above expression is a consistent estimator of E z∼Qα [r(z; x 1:T , u 1:T , θ)] with order O(N −1/2 ), which we have shown above to be a lower bound to the likelihood function.

For the special case of N α = 1 (when α = 1/N ), we can replace the sorting operation with a simple min operation, yielding a simple and intuitive algorithm -we choose the sampled z with the lowest error for updating the parameters θ.

In practice, maximizing the log-likelihood of the data is known to work well BID12 .

In our case, we correspondingly maximize E z∼Qα [log r(z; x 1:T , u 1:T , θ)], which, by the Jensen inequality 3 , is a lower bound on log E z∼Qα [r(z; x 1:T , u 1:T , θ)].

We therefore obtain the following gradient estimate G N,α (θ) := 1 Nα Nα i=1 ∇ θ log r(z ord i ; x 1:T , u 1:T , θ).

We term this sampling technique as intention-driven sampling (IDS).

Pseudocode is given in Algorithm 1.Optimistic Sampling: To predict actions at test time, we first sample z in the beginning of the episode and fix it, and then use the NN to predict u t = f (x t , z) at every time step.

While we could sample z from P (z), this might result in a z value that has a low likelihood to reproduce the demonstrations, i.e., a z that has low reward.

We observed this to be problematic in practice, and therefore devise an alternative approach, which we term optimistic sampling.

We propose to store a set of the K most recent z values that obtained the highest reward during training, and at inference time sample a z uniformly from this set.

This corresponds to sampling z from Q α (z), averaged over the training data.

Optimistic sampling dramatically improves the prediction performance in practice.

Analysis: By Theorem 4.2 of BID15 , G N,α (θ) is a consistent gradient estimator of the lower bound with asymptotic bias of O(N −1/2 ).

In Appendix 8, we deduce the following expression for the variance of G N,α (θ): DISPLAYFORM4 When N α = N , i.e., α = 1, we obtain the variance of standard MC sampling , i.e., V MC = 1 N Var(∇ θ log r(z; x 1:T , u 1:T , θ)).

On the other hand, when α → 1/N the variance is bounded in O(1/N 2 ).

This result is due to the fact that (i) |∇ θ (log r(z 1 ; x 1:T , u 1: DISPLAYFORM5 .

Therefore, one can treat α as a nob to trade-off bias (see FORMULA8 ) and variance (see (4)) in IDS.Algorithm 1: IDS Input: A minibatch of K samples {u t , x t , . . . , u t+K , x t+K } from the same demonstration trajectory T i Output: An update direction for θ, and a sample from Q α 1 Sample z 1 , . . .

, z N ∼ P (z) 2 Set z * = arg max zi P (u t:t+K |x t:t+K , z i ; θ)3 return ∇ θ log P (u t:t+K |x t:t+K , z * ; θ), and z * Comparison to SNNs: Broadly speaking, generalized EM algorithms (Tang & Salakhutdinov, 2013) can be seen as designing an importance sampling weight to reshape the sampling distribution, in order to lower the variance of the gradient estimate, by using entropy maximization or posterior distribution matching (more details can be found in Appendix 7).

For the specific SNN algorithm of Tang & Salakhutdinov (2013) applied to our NN architecture, the importance weights correspond to a soft-max over the reward defined above.

In IDS the importance weight is w(z) = 1 α 1{r(z; x 1:T , u 1:T , θ) ≥ q α (θ)}, which for N α = 1 amounts to replacing the soft-max with a hard max.

Interestingly, these two similar algorithms were developed from very different first principles.

In practice, however, the IDS algorithm integrates naturally with optimistic sampling, which leads to significantly better performance, as we show in our experiments.

In this section we present Intention-SNN (henceforth I-SNN), an architecture that implements the stochastic intention as an attention over particular features in the image.

This architecture is suitable for LfD domains where the visual observation contains information about multiple possible intentions, as in the object pick up task in Section 5.Our I-SNN architecture is presented in FIG0 , and is comprised of three modules.

The first module is a standard multi-layer (fully) convolutional neural network (CNN) feature extractor BID12 , followed by a spatial softmax layer.

The CNN maps the input image onto C feature maps.

The spatial softmax, introduced by BID18 , calculates for each feature map in its input the corresponding (x, y) position in the image where this feature is most active.

Let φ c,i,j denote the activation of feature map c at coordinate (i, j).

The spatial softmax output for that feature is (f c,x , f c,y ), where f c,x = i,j exp(φ c,i,j ) · i/ i ,j exp(φ c,i ,j ), and f c,y = i,j exp(φ c,i,j ) · j/ i ,j exp(φ c,i ,j ).

Thus, the output of the spatial softmax is of dimensions C × 2.

The second module applies a stochastic soft attention over the spatial softmax features.

We use a MLP to map the M -dimensional random intention vector z onto a C-dimensional vector w.

We then apply a softmax to obtain the attention weight vector a ∈ R C , a c = exp(w c )/ c exp(w c ) (Xu et al., 2015) .

The attention weight is multiplied with the spatial softmax output to obtain the attention-modulated feature activations: f c,x = f c,x · a c , and f c,y = f c,y · a c , which are input to the control network along with the robot's d-dimensional vector of current pose.

The control network is a standard MLP mapping R C×2+d to P (u).The intuition behind this architecture is that the stochastic intention can 'select' which features are relevant for making a control decision, by giving them higher weights in the attention.

When multiple objects are in the scene, each object would naturally be represented by different features, therefore the intention in this architecture can correspond to attending to a particular object 6 .

We demonstrate the effectiveness of our approach on learning a reaching task with the IRB-120 robot FIG0 , where mentor trajectories are collected in the form of sequential image-action pairs.

The main questions we seek to answer are: (1) Can our IDS learn to effectively reproduce multiple intentions in the demonstrations?

(2) How does IDS approach compare to a standard deterministic NN approach for LfD?

(3) How does training an I-SNN using IDS compare with training it using the SNN algorithm 7 of Tang & Salakhutdinov (2013)?To maintain a fair comparison, we evaluated deterministic NN policies with identical structure as I-SNN except for the stochastic intention module (cf.

Section 4.4, and FIG0 ).

In all our experiments we used the following parameters for the SNN training, which we found to work well consistently across different domains:(1) Loss function: we represent the output distribution as log P (u|h, x; θ) ∝ − f (x, h; θ) − u 1 , where f (x, h; θ) is the output of the control network, as described in Section 4.4.

This corresponds to an L 1 regression loss, which we found to perform better than the popular L 2 loss in our experiments.

(2) Monte Carlo samples: we chose N = 5, 6 We note that this architecture is not directly applicable for cases where two objects have exactly the same appearance, and therefore the same feature activation function.

Such cases can be handled by adding spatial information to the features or the attention module, which will be investigated in future work.

7 We also investigated using a conditional VAE (CVAE), however, despite extensive experimentation, we could not get the CVAE to work.

We attribute this to the fact that the recognition network in a CVAE needs to map the image to a latent variable distribution that 'explains' the observed action.

This mapping is complex, as it needs to understand from the image what goal the demonstration is aiming towards.

Our approach, on the other hand, does not require a recognition module for reducing variance during training.which we found to work well.

Higher values resulted in degraded performance at test time, due to the higher bias (see Section 4).

(3) Intention variable dimension: we chose z ∈ R 5 for all experiments.

We did not observe this parameter to be sensitive, and dimensions from 2 to 10 performed similarly.

(4) P(z): a 5-dimensional vector of independent uniform multinomials in {0 : 4}.

In this task, depicted in FIG0 , the objective is to navigate the end-effector of robot to reach a point above one of 3 objects in the scene -a soap box, a blue electronic box and a measuring cup.

We used a 6-DOF IRB-120 robot where the control is a 3-dimensional Cartesian vector applied to the the end effector u t = (d x t , d y t , d z t ) .

The observations consists of (1) a 480 × 640 RGB image of the scene (further cropped and resized to 64 × 64 resolution) using a point-grey camera mounted at a fixed angle; (2) the 3 dimensional end-effector Cartesian pose (see FIG0 for more details).Data Collection: We denote a specific scene arrangement of object placements and initial endeffector pose as a task configuration.

Task configurations were randomly generated by arbitrarily placing the three objects on the work bench and randomly initializing the end-effector pose.

Once a task configuration is generated, the position of the objects remain fixed for the entire episode.

For each task configuration we collect 3 demonstrations, each generated by a human navigating the end-effector using a 3DConnexion space-mouse to reach one of the objects.

At each time step t, the observation o t together with the Cartesian control signal u t (see above) are recorded.

We collected demonstrations from 468 different task configurations, for a total of 1404 demonstration trajectories.

Training: We compare using IDS and the SNN algorithm of Tang & Salakhutdinov (2013) (henceforth SNN) for training the I-SNN architecture.

To reduce the training time, we pre-trained the weights of the convolutional layers in the Feature Extraction module reusing the weights learned in the deterministic NN model.

For optimization, we also used Adam BID16 , with the default parameters using 90% of the data set for training and 10% for validation. .

We evaluate each model on 10 randomly generated task configurations, with 20 trials on each task configuration, for a total of 200 trials.

We run the model until it either succeeds to reach an object or fails.

For the IDS algorithm, we used optimistic sampling.

For SNN, we experimented with both optimistic and uniform sampling, however the latter resulted in a better overall performance.

TAB0 shows the overall success rate for every model across all 200 trials.

The deterministic NN model succeeded in reaching one of the objects only in 3 task configurations (and kept on reaching that same object for the 20 evaluations in each), and failed on the other 7 task configurations due to the averaging problem.

The stochastic algorithms performed significantly better by learning multiple modes of the problem.

IDS significantly out-performed the SNN algorithm, as we explain below.

We evaluate the mode learning ability by counting, for each task configuration, how many different objects were reached, as depicted in TAB0 (last four columns).

As expected, the deterministic NN could not reach more than one object.

I-SNN trained by SNN algorithm, on the other hand, could reach two different (but fixed) objects for 6 task configurations, and all three objects for the rest.

The best performance is achieved by I-SNN trained by IDS, reaching all three objects in all the tasks, thereby demonstrating a strong ability to learn all the modes in the data.

In the appendix (Figure 3 ), we show a histogram of reaching the different objects that demonstrates that IDS learned a near-uniform distribution over the modes.

Additionally, in the appendix ( Figure 5 ) we visualize the features that I-SNN attends to, showing that in each episode the model consistently attends to the same object throughout the execution.

In Figure 2 we explain the superior performance of IDS over SNN.

Since IDS focuses only on the best samples (through the max) compared to SNN which gives weight also to non best samples (through the soft-max), IDS better 'tunes' the network for the best samples.

Combined with optimistic sampling, which draws these best samples during execution, this leads to better results.

We presented an approach for learning from demonstrations that contain multiple modes of performing the same task.

Our method is based on stochastic neural networks, and represents the mode Figure 2 : Comparison of IDS and SNN algorithms.

We plot three different errors during training (on the training data), for the same model trained using IDS and SNN algorithm.

Left: the respective training loss for each method.

Since the max in IDS upper bounds the softmax in SNN, the loss plot for IDS lower bounds SNN.

Middle: the IDS loss on the training data, for both models.

Since the SNN is trained on a different loss function (softmax), its performance is worse.

This shows an important point: if, at test time, we use optimistic sampling to sample z from best samples during training, we should expect IDS to perform better than SNN.

Right: the average log-likelihood loss during training.

The SNN wins here, since the softmax encourages to increase the likelihood of 'incorrect' z values.

This provides additional motivation for using optimistic sampling.of performing the task by a stochastic vector -the intention, which is given as input to a feedforward neural network.

We presented a simple and efficient algorithm for training our models, and a particular implementation suitable for vision-based inputs.

As we demonstrated in real-robot experiments, our method can reliably learn to reproduce the different modes in the demonstration data, and outperforms standard approaches in cases where such different modes exist.

In future work we intend to investigate the extension of this approach to more complex manipulation tasks such as grasping and assembly, and domains with a very large number of objects in the scene.

An interesting point in our model is tying the features to the intention by an attention mechanism, and we intend to further investigate recurrent attention mechanisms (Xu et al., 2015) that could offer better generalization at inference time.

F (Q, θ) =E Q log P (u 1:T , z|x 1:T ; θ) Q(z|x 1:T , u 1:T ) ≤E P (·|x 1:T ,u 1:T ;θ) [log P (y 1:T |x 1:T ,θ)] DISPLAYFORM0 where F is the Kullback Liebler divergence between P (u 1:T |z, x 1:T ; θ) and Q(z|u 1:T , x 1:T ) given as follows:F (Q, θ) = −D KL (Q||P (·|x 1:T , u 1:T ; θ)) + log P (y 1:T |x 1:T ,θ).

Most importantly, it has also been shown in Theorem 2 of BID23 that if Q and θ form a pair of local maximizer to F , then θ is also a local maximum of the original likelihood maximization problem.

To maximize F w.r.t Q, one has the closed form solution based on Bayes theorem: Q * (z|u 1:T , x 1:T ; θ old ) =P (z|y 1:T , x 1:T , θ) DISPLAYFORM1 Here, {z 1 , . . .

, z N } is a sequence of latent random variables sampled i.i.d.

from the distribution P (z).Given parameter θ, denoted by θ old , immediately the posterior distribution Q that maximizes F is given by: Q * (z|x 1:T , u 1:T ) = P (z|x 1:T , u 1:T ; θ old ).

In this case, the above loss function is equivalent to the complete data log-likelihood * (θ, θ old ) := E P (·|u 1:T ,x 1:T ;θold) log P (x 1:T , z|u 1:T ; θ) P (z|x 1:T , u 1:T ; θ old ) , which is a lower bound of the log likelihood.

Furthermore, if θ = θ old , then clearly * (θ old , θ old ) is equal to the log-likelihood log P (y 1:T |x 1:T ,θ old ).Tang & Salakhutdinov (2013) present a generalized EM algorithm to train a SNN.

In the E-step, the following approximate posterior distribution is used:Q(z|u 1:T , x 1:T ; θ old ) :=r(z; x 1:T ,y 1:T , θ old )P (z), wherer (z; x 1:T ,y 1:T , θ old ) = r(z; x 1:T , u 1: DISPLAYFORM2 r(z i ; x 1:T , u 1:T , θ old ) is the the importance sampling weight.

Recall that for our distribution model, r(z; x 1:T , u 1:T , θ old ) ∝ exp(−d(f (x, z; θ), u)), therefore we obtain that the importance weights correspond to a soft-max over the prediction error.

In the M-step, the θ parameters are updated with the gradient vector with respect to the following optimization: θ ∈ arg max θ∈Θˆ (θ, θ old ), wherê DISPLAYFORM3 (z i ; x 1:T ,y 1:T , θ old ) log P (y 1:T , z i |x 1:T , θ)is the empirical expected log likelihood, andQ is the posterior distribution from the E-step.

Here we drop the last term in F because in our case Q that does not depend on θ.

Correspondingly, the gradient estimate is given by: DISPLAYFORM4 (z i )∇ θ log r(z i ; x 1:T , u 1:T , θ), the equality is due to the facts that log P (y 1:T , z|x 1:T , θ) = log r(z; x 1:T ,y 1:T , θ) + log P (z) and distribution P (z) is independent of θ.

To better understand this estimator, we will analyze the bias and variance of the gradient estimator.

Based on the construction of importance sampling weight, immediately the gradient estimator is consistent.

Furthermore, under certain regular assumptions, the bias is O(N −1/2 ).

(This means the gradient estimator is asymptotically unbiased.)

Furthermore, the variance of this estimator is given by DISPLAYFORM5 where the integrand is given by v(z; θ) =r(z; x 1:T ,y 1:T , θ old ) · (∇ θ log r(z; x 1:T , u 1:T , θ)) 2 ≥ 0.

In this section, we study the variance of the gradient estimate of CVaR(R(z; θ)).

This proof follows analogously from the analysis in Section 4.3 of BID15 for the case of estimating asymptotic variance in gradient.

Here we use R(z; θ) as the shorthand notation of the log reward function log r(z; x 1:T , y 1:T , θ).

For any given cut-off level α and sample size n, consider the following update formula of the CVaR gradient estimate DISPLAYFORM0 where S(z i ) = 1{R(z i ; θ) ≥ q α (θ)}, ∀i, and {z 1 , . . .

, z N } is sampled in an i.i.d.

fashion from P (h).

Notice that DISPLAYFORM1 The correlation comes from the fact that quantile is defined based on the order statistics of the reward, see Section 4 of BID15 for more details.

Now notice the following property: DISPLAYFORM2 where L i:N is the i−th order statistic from the N observations of reward {R( DISPLAYFORM3 .

Under the event that R(z 1 ; θ) > L N α −1:N −2 and R(z 2 ; θ) > L N α −1:N −2 , by the definition of the order statistic and by the i.i.d.

assumption of {z 1 , . . .

, z N }, one can deduce that L N α −1:N −2 is independent of R(z 1 ; θ) and R(z 2 ; θ).

Equipped with this condition, one has the following expression: On the other hand, following the same lines of analysis, one can also show that DISPLAYFORM4 DISPLAYFORM5 Therefore, combining all the above analysis together, one has the following expression: DISPLAYFORM6 From the results in Proposition 4.3 to 4.4 of BID15 , with q α (θ) = ∇ θ q α (θ) denote the gradient of the quantile, we have that DISPLAYFORM7 Therefore, the variance of G N can be expressed as: DISPLAYFORM8 This provides the variance of intention-driven sampling gradient estimate.9 ADDITIONAL MATERIAL states captured by a recurrent neural network) .

In such models, task level intention is not guaranteed to be consistently inferred at every step throughout the task execution; (b) in contrast I-SNN samples and uniformly commits to the same mode at the task level throughout the task execution.

Figure 5 : Visualization of the stochastic Intention network: every row shows 5 snapshots of a trajectory generated by running the I-SNN trained by the IDS algorithm.

Each run was generating by randomly sampling an intention at the beginning and using it throughout the run.

Smaller green circles show the 32 coordinates outputted by the spatial softmax layer.

The larger red circle shows the top spatial softmax feature that received the highest weight from the soft attention generated by the Stochastic Intention Network.

Note that for each run, the model consistently attends to the same mode that it randomly selected at the beginning of the run.

In this section we present simulation results that compare our IDS approach with a state-of-the-art CVAE based approach (Sohn et al., 2015) .

We consider a simplified task which we term 'Predict the Goal', depicted in Figure 6 .

Given an image with N randomly positioned targets with different colors, the task is to predict the location (i.e., x-y position) of one of them.

For training, we randomly selected one of the targets and provide its location as the supervisory signal.

This task captures the essence of the robotic task in the paper -image input and a low dimensional multi-modal output (with N modes).

It simplifies the image processing, and the fact that there is no trajectory -this is a single step decision making problem.

For our IDS algorithm, we used the I-SNN architecture as described in FIG0 , with a single conv layer (and without the additional robot pose input).

The output is the 2-dimensional target position.

Figure 6 : Predict the Goal Domain: in an image with N randomly positioned, different colored targets, the task is to predict the center of one of the targets.

The figure shows 9 random instances of a domain with N = 5 targets.

We also plot the training target positions (dark green dots, selected uniformly among the targets), and the predictions of the trained I-SNN (yellow dots).

Note that the predictions do not have to match the training targets, but have to be centered on some target in the image.

For the CVAE, the generation network P (u|x, z) is the same I-SNN.

For the recognition network q(z|x, u) we used an MLP mapping the spatial-softmax output and the target position to the mean and std of z. For the conditional prior network p(z|x) we used an MLP mapping the spatial-softmax output to the mean and std of z. Following the work of Sohn et al. FORMULA4 , we added to the training loss a term KL(q(z|x, u) p(z|x)).To make the comparison fair, we chose the latent variable z in IDS to be a standard Gaussian, the same as for the CVAE.

All network sizes and training parameters were the same for both methods, and we did not apply any pretraining of the conv layer.

For evaluating performance, we measure the shortest distance from the prediction to one of the target positions, on a held-out test set.

This error should go to zero if the model predicts one of the targets accurately.

Our results are reported in FIG3 .

We have tried various CVAE parameter settings (such as the minibatch sizes and dimension of z reported here, which the CVAE was sensitive to, among other parameters such MLP architectures and learning rates), and also tried annealing the KL term in the cost.

The CVAE works well for N = 2 targets, and with careful tuning also for N = 3, but we could not get it to work for N = 5 targets.

The IDS approach, on the other hand, worked well and robustly for all values of N we tried.

The convergence of IDS in all cases was also an order of magnitude faster than the CVAE.

@highlight

multi-modal imitation learning from unstructured demonstrations using stochastic neural network modeling intention. 

@highlight

A new sampling-based approach for inference in latent variable models that applies to multi-modal imitation learning and works better than deterministic neural networks and stochastic neural networks for a real visual robotics task.

@highlight

This paper shows how to learn several modalities using imitation learning from visual data using stochastic Neural Networks, and a method for learning from demonstrations where several modalities of the same task are given.