Currently the only techniques for sharing governance of a deep learning model are homomorphic encryption and secure multiparty computation.

Unfortunately, neither of these techniques is applicable to the training of large neural networks due to their large computational and communication overheads.

As a scalable technique for shared model governance, we propose splitting deep learning model between multiple parties.

This paper empirically investigates the security guarantee of this technique, which is introduced as the problem of model completion:  Given the entire training data set or an environment simulator, and a subset of the parameters of a trained deep learning model, how much training is required to recover the model’s original performance?

We define a metric for evaluating the hardness of the model completion problem and study it empirically in both supervised learning on ImageNet and reinforcement learning on Atari and DeepMind Lab.

Our experiments show that (1) the model completion problem is harder in reinforcement learning than in supervised learning because of the unavailability of the trained agent’s trajectories, and (2) its hardness depends not primarily on the number of parameters of the missing part, but more so on their type and location.

Our results suggest that model splitting might be a feasible technique for shared model governance in some settings where training is very expensive.

With an increasing number of deep learning models being deployed in production, questions regarding data privacy and misuse are being raised BID4 .

The trend of training larger models on more data BID28 , training models becomes increasingly expensive.

Especially in a continual learning setting where models get trained over many months or years, they accrue a lot of value and are thus increasingly susceptible to theft.

This prompts for technical solutions to monitor and enforce control over these models BID41 .

We are interested in the special case of shared model governance: Can two or more parties jointly train a model such that each party has to consent to every forward (inference) and backward pass (training) through the model?Two popular methods for sharing model governance are homomorphic encryption (HE; BID36 and secure multi-party computation (MPC; BID48 .

The major downside of both techniques is the large overhead incurred by every multiplication, both computationally, >1000x for HE BID29 BID14 , >24x for MPC BID25 BID7 , in addition to space (>1000x in case of HE) and communication (>16 bytes per 16 bit floating point multiplication in case of MPC).

Unfortunately, this makes HE and MPC inapplicable to the training of large neural networks.

As scalable alternative for sharing model governance with minimal overhead, we propose the method of model splitting: distributing a deep learning model between multiple parties such that each party holds a disjoint subset of the model's parameters.

Concretely, imagine the following scenario for sharing model governance between two parties, called Alice and Bob.

Alice holds the model's first layer and Bob holds the model's remaining layers.

In each training step (1) Alice does a forward pass through the first layer, (2) sends the resulting activations to Bob, (3) Bob completes the forward pass, computes the loss from the labels, and does a backward pass to the first layer, (4) sends the resulting gradients to Alice, and (5) Alice finishes the backward pass.

How much security would Alice and Bob enjoy in this setting?

To answer this question, we have to consider the strongest realistic attack vector.

In this work we assume that the adversary has access to everything but the missing parameters held by the other party.

How easy would it be for this adversary to recover the missing part of the model?

We introduce this as the problem of model completion:Given the entire training data set or an environment simulator, and a subset of the parameters of a trained model, how much training is required to recover the model's original performance?In this paper, we define the problem of model completion formally (Section 3.1), propose a metric to measure the hardness of model completion (Section 3.2), and provide empirical results (Section 4 and Section 5) in both the supervised learning (SL) and in reinforcement learning (RL).

For our SL experiments we use the AlexNet convolutional network BID26 and the ResNet50 residual network BID17 on ImageNet BID9 ); for RL we use A3C and Rainbow BID19 in the Atari domain BID1 and IMPALA BID11 on DeepMind Lab BID0 .

After training the model, we reinitialize one of the model's layers and measure how much training is required to complete it (see FIG1 ).Our key findings are: (1) Residual networks are easier to complete than nonresidual networks (

The closest well-studied phenomenon to model completion is unsupervised pretraining, first introduced by BID22 .

In unsupervised pretraining a subset of the model, typically the lower layers, is trained in a first pass with an unsupervised reconstruction loss BID10 .

The aim is to learn useful high-level representations that make a second pass with a supervised loss more computationally and sample efficient.

This second pass could be thought as model completion.

In this paper we study vertical model completion where all parameters in one layer have to be completed.

Instead we could have studied horizontal model completion where some parameters have to be completed in every layer.

Horizontal model completion should be easy as suggested by the effectiveness of dropout as a regularizer BID40 , which trains a model to be resilient to horizontal parameter loss.

Pruning neural networks BID27 ) is in a sense the reverse operation to model completion.

BID6 prune individual connections and BID33 prune entire feature maps using different techniques; their findings, lower layers are more important, are compatible with ours.

BID13 present empirical evidence for the lottery ticket hypothesis: only a small subnetwork matters (the 'lottery ticket') and the rest can be pruned away without loss of performance.

The model completion problem for this lottery ticket (which is spread over all layers) would be trivial by definition.

All of these works only consider removing parts of the model horizontally.

The model completion problem can also be viewed as transfer learning from one task to the same task, while only sharing a subset of the parameters BID34 BID44 .

BID49 investigate which layers in a deep convolutional model contain general versus task-specific representations; some of their experiments follow the same setup as we do here and their results are in line with ours, but they do not measure the hardness of model completion task.

Finally, our work has some connections to distillation of deep models BID5 BID3 .

Distillation can be understood as a 'reverse' of model completion, where we want to find a smaller model with the same performance instead of completing a smaller, partial model.

The literature revolves around two techniques for sharing model governance: homomorphic encryption (HE; BID36 and secure multi-party computation (MPC; BID48 BID8 .

Both HE and MPC have been successfully applied to machine learning on small datasets like MNIST BID14 BID32 BID7 BID46 and the Wisconsin Breast Cancer Data set BID16 .HE is an encryption scheme that allows computation on encrypted numbers without decrypting them.

It thus enables a model to be trained by an untrusted third party in encrypted form.

The encryption key to these parameters can be cryptographically shared between several other parties who effectively retain control over how the model is used.

Using MPC numbers can be shared across several parties such that each share individually contains no information about these numbers.

Nevertheless computational operations can be performed on the shared numbers if every party performs operations on their share.

The result of the computation can be reconstructed by pooling the shares of the result.

While both HE and MPC fulfill a similar purpose, they face different tradeoffs for the additional security benefits: HE incurs a large computational overhead BID29 while MPC incurs a much smaller computational overhead in exchange for a greater communication overhead BID25 .

Moreover, HE provides cryptographic security (reducing attacks to break the cipher on well-studied hard problems such as the discrete logarithm) while MPC provides perfect information-theoretic guarantees as long as the parties involved (3 or more) do not collude.

There are many applications where we would be happy to pay for the additional overhead because we cannot train the model any other way, for example in the health sector where privacy and security are critical.

However, if we want to scale shared model governance to the training of large neural networks, both HE and MPC are ruled out because of their prohibitive overhead.

In contrast to HE and MPC, sharing governance via model splitting incurs minimal computational and manageable communication overhead.

However, instead of strong security guarantees provided by HE and MPC, the security guarantee is bounded from above by the hardness of the model completion problem we study in this paper.

Let f θ be a model parameterized by the vector θ.

We consider two settings: supervised learning and reinforcement learning.

In our supervised learning experiments we evaluate the model f θ by its performance on the test loss L(θ).In reinforcement learning an agent interacts with an environment over a number of discrete time steps BID43 : In time step t, the agent takes an action a t and receives an observation o t+1 and a reward r t+1 ∈ R from the environment.

We consider the episodic setting in which there is a random final time step τ ≤ K for some constant K ∈ N, after which we restart with timestep t = 1.

The agent's goal is to maximize the episodic return G := τ t=1 r t .

Its policy is a mapping from sequences of observations to a distribution over actions parameterized by the model f θ .

To unify notation for SL and RL, we equate L(θ) = E at∼f θ (o1,...,ot−1) [−G] such that the loss function for RL is the negative expected episodic return.

To quantify training costs we measure the computational cost during (re)training.

To simplify, we assume that training proceeds over a number of discrete steps.

A step can be computation of gradients and parameter update for one minibatch in the case of supervised learning or one environment step in the case of reinforcement learning.

We assume that computational cost are constant for each step, which is approximately true in our experiments.

This allows us to measure training cost through the number of training steps executed.

Let T denote the training procedure for the model f θ and let θ 0 , θ 1 , . . .

be the sequence of parameter vectors during training where θ i denotes the parameters in training step i. Furthermore, let * := min{L(θ i ) | i ≤ N } denote the best model performance during the training procedure T (not necessarily the performance of the final weights).

We define the training cost as the random variable C T ( ) := arg min i∈N {L(θ i ) ≤ }, the number of training steps until the loss falls below the given threshold ∈ R. After we have trained the model f θ for N steps and thus end up with a set of trained parameters θ N with loss L(θ N ), we split the parameters θ N = [θ

How hard is the model completion problem?

To answer this question, we use the parameters DISPLAYFORM0 N are the previously trained parameters and θ 0 2 are freshly initialized parameters.

We then execute a (second) retraining procedure T ∈ T from a fixed set of available retraining procedures T .1 The aim of this retraining procedure is to complete the model, and it may be different from the initial training procedure T .

We assume that T ∈ T since retraining the entire model from scratch (reinitializing all parameters) is a valid way to complete the model.

Let θ 0 , θ 1 , . . .

be the sequence of parameter vectors obtained from running the retraining procedure T ∈ T .

Analogously to before, we define C T ( ) := arg min i∈N {L(θ i ) ≤ } as the retraining cost to get a model whose test loss is below the given threshold ∈ R. Note that by definition, for T = T we have that C T ( ) is equal to C T ( ) in expectation.

In addition to recovering a model with the best original performance * , we also consider partial model completion by using some higher thresholds * DISPLAYFORM1 .

These higher thresholds * α correspond to the relative progress α from the test loss of the untrained model parameters L(θ 0 ) to the best test loss DISPLAYFORM2 We define the hardness of model completion as the expected cost to complete the model as a fraction of the original training cost for the fastest retraining procedure T ∈ T available: DISPLAYFORM3 where the expectation is taken over all random events in the training procedures T and T .It is important to emphasize that the hardness of model completion is a relative measure, depending on the original training cost C T ( * α ).

This ensures that we can compare the hardness of model completion across different tasks and different domains.

In particular, for different values of α we compare like with like: MC-hardness T (α) is measured relative to how long it took to get the loss below the threshold * α during training.

Importantly, it is not relative to how long it took to train the model to its best performance * .

This means that naively counter-intuitive results such as MC-hardness T (0.8) being less than MC-hardness T (0.5) are possible.

Since C T ( ) and C T ( ) are nonnegative, MC-hardness T (α) is nonnegative.

Moreover, since T ∈ T by assumption, we could retrain all model parameters from scratch (formally setting T to T ).

Thus we have MC-hardness T (α) ≤ 1, and therefore MC-hardness is bounded between 0 and 1.

Equation 1 denotes an infimum over available retraining procedures T .

However, in practice there is a vast number of possible retraining procedures we could use and we cannot enumerate and run all of them.

Instead, we take an empirical approach for estimating the hardness of model completion: we investigate the following set of retraining strategies T to complete the model.

All the retraining strategies, if not noted otherwise, are built on top of the original training procedure T .

Our best result are only an upper bound on the hardness of model completion.

It is likely that much faster retraining procedures exist.

T 1 Optimizing θ 0 1 and θ 0 2 jointly.

We repeat the original training procedure T on the preserved parameters θ 0 1 and reinitialized parameters θ 0 2 .

The objective function is optimized with respect to all the trainable variables in the model.

We might vary in hyperparameters such as learning rates or loss weighting schemes compared to T , but keep hyperparameters that change the structure of the model (e.g. size and number of layers) fixed.

T 2 Optimizing θ 0 2 , but not θ 0 1 .

Similarly to T 1 , in this retraining procedure we keep the previous model structure.

However, we freeze the trained weights θ 0 1 , and only train the reinitialized parameters θ 0 2 .T 3 Overparametrizing the missing layers.

This builds on retraining procedure T 1 .

Overparametrization is a common trick in computer vision, where a model is given a lot more parameters than required, allowing for faster learning.

This idea is supported by the 'lottery ticket hypothesis' BID13 : a larger number of parameters increases the odds of a subpart of the network having random initialization that is more conducive to optimization.

T 4 Reinitializing parameters θ 0 2 using a different initialization scheme.

Previous research shows that parameter initialization schemes can have a big impact on convergence properties of deep neural networks BID15 BID42 .

In T 1 our parameters are initialized using a glorot uniform scheme.

This retraining procedure is identical to T 1 except that we reinitialize θ 0 2 using one of the following weight initialization schemes: glorot normal BID15 , msra BID18 or caffe BID24 .

Our main experimental results establish upper bounds on the hardness of model completion in the context of several state of the art models for both supervised learning and reinforcement learning.

In all the experiments, we train a model to a desired performance level (this does not have to be stateof-the-art performance), and then reinitialize a specific part of the network and start the retraining procedure.

Each experiment is run with 3 seeds, except IMPALA (5 seeds) and A3C (10 seeds).Supervised learning.

We train AlexNet BID26 and ResNet50 BID17 on the ImageNet dataset BID9 to minimize cross-entropy loss.

The test loss is the top-1 error rate on the test set.

AlexNet is an eight layer convolutional network consisting of five convolutional layers with max-pooling, followed by two fully connected layers and a softmax output layer.

ResNet50 is a 50 layer convolutional residual network: The first convolutional layer with max-pooling is followed by four sections, each with a number of ResNet blocks (consisting of two convolutional layers with skip connections and batch normalization), followed by average pooling, a fully connected layer and a softmax output layer.

We apply retraining procedures T 1 and T 2 and use a different learning rate schedule than in the original training procedure because it performs better during retraining.

All other hyperparameters are kept the same.

Reinforcement learning.

We consider three different state of the art agents: A3C BID31 , Rainbow BID19 and the IMPALA reinforcement learning agent BID11 .

A3C comes from a family of actor-critic methods which combine value learning and policy gradient approaches in order to reduce the variance of the gradients.

Rainbow is an extension of the standard DQN agent, which combines double Q-learning (van Hasselt, 2010), dueling networks BID47 ), distributional RL (Bellemare et al., 2017 and noisy nets BID12 .

Moreover, it is equipped with a replay buffer that stores the previous million transitions of the form (o t , a t , r t+1 , o t+1 ), which is then sampled using a prioritized weighting scheme based on temporal difference errors BID38 .

Finally, IMPALA is an extension of A3C, which uses the standard actor-critic architecture with off-policy corrections in order to scale effectively to a large scale distributed setup.

We train IMPALA with population based training BID23 .For A3C and Rainbow we use the Atari 2600 domain BID1 and for IMPALA DeepMind Lab BID0 .

In both cases, we treat the list of games/levels as a single learning problem by averaging across games in Atari and training the agent on all level in parallel in case of DeepMind Lab.

In order to reduce the noise in the MC-hardness metric, caused by agents being unable to learn the task and behaving randomly, we filter out the levels in which the original trained agent performs poorly.

We apply the retraining procedures T 1 , T 2 on all the models, and on A3C we apply additionally T 3 and T 4 .

All the hyperparameters are kept the same during the training and retraining procedures.

Further details of the training and retraining procedures for all models can be found in Appendix A, and the parameter counts of the layers are listed in Appendix B.

Our experimental results on the hardness of the model completion problem are reported in FIG4 .

These figures show on the x-axis different experiments with different layers being reinitialized (lower to higher layers from left to right).

We plot MC-hardness T (α) as a bar plot with error bars showing the standard deviation over multiple experiment runs with different seeds; the colors indicate different values of α.

The numbers are provided in Appendix C. In the following we discuss the results.1.

In the majority of cases, T 1 is the best of our retraining procedures.

From the retraining procedures listed in Section 3.3 we use T 1 and T 2 in all experiments and find that T 1 performs substantially better in all settings except two: First, for A3C, starting from the third convolutional layer, T 2 has lower MC-hardness for all the threshold levels ( FIG7 .

Second, T 2 performs well on all the layers when retraining ResNet-50, for all α ≤ 0.9 ( FIG2 ; the difference is especially visible at α = 0.9.For A3C we use all four retraining procedures.

The difference between T 1 and T 2 are shown in FIG7 .

For T 3 we tried replacing the first convolutional layer with two convolutional layers using a different kernel size, as well as replacing a fully connected layer with two fully connected layers of varying sizes.

The results were worse than using the same architecture and we were often unable to retrieve 100% of the original performance.

With T 4 we do not see any statistically significant difference in retraining time between the initialization schemes glorot normal, msra, and caffe.

FIG4 and FIG2 for T 1 , the model hardness for threshold α = 0.5 and α = 0.8 is much lower for ResNet50 than for AlexNet.

However, to get the original model performance (α = 1), both models need about 40% of the original training cost.

As mentioned above, T 2 works better than T 1 on ResNet50 for α ≤ 0.9.

An intact skip connection helps retraining for α ≤ 0.9 and T 1 , but not T 2 , as illustrated in the experiment S4 B1 -W FIG2 .

A noticeable outlier is S4 B1 at α = 0.9; it is unclear what causes this effect, but it reproduced every time we ran this experiment.

Residual neural networks use skip connections across two or more layers BID17 .

This causes the features in those layers to be additive with respect to the incoming features, rather than replacing them as in non-residual networks.

Thus lower-level and higher-level representations tend to be more spread out across the network, rather than being confined to lower and higher layers, respectively.

This would explain why model completion in residual networks is more independent of the location of the layer.3.

For A3C lower layers are often harder to complete than upper layers.

FIG7 shows that for A3C the lower layers are harder to complete than the higher layers since for each value of α the MC-hardness decreases from left to right.

However, this effect is much smaller for Rainbow ( Figure 5 ) and AlexNet FIG4 ).In nonresidual networks lower convolutional layers typically learn much simpler and more general features that are more task independent BID49 .

Moreover, noise perturbations of lower layers have a significantly higher impact on the performance of deep learning models since noise grows exponentially through the network layers BID35 .

Higher level activations are functions of the lower level ones; if a lower layer is reset, all subsequent activations will be invalidated.

This could imply that the gradients on the higher layers are incorrect and thus slow down training.4.

The absolute number of parameters has a minimal effect on the hardness of model completion.

If information content is spread uniformly across the model, then model completion should be a linear function in the number of parameters that we remove.

However, the number of parameters in deep models usually vary greatly between layers; the lower-level convolutional layers have 2-3 orders of magnitude fewer parameters than the higher level fully connected layers and LSTMs (see Appendix B).In order to test this explicitly, we performed an experiment on AlexNet both increasing and decreasing the total number of feature maps and fully connected units in every layer by 50%, resulting in approximately an order of magnitude difference in terms of parameters between the two models.

We found that there is no significant difference in MC-hardness across all threshold levels.5.

RL models are harder to complete than SL models.

Across all of our experiments, the model completion of individual layers for threshold α = 1 in SL FIG4 and FIG2 is easier than the model completion in RL FIG7 , Figure 5 , and Figure 6 ).

In many cases the same holds from lower thresholds as well.

By resetting one layer of the model we lose access to the agent's ability to generate useful experience from interaction with the environment.

As we retrain the model, the agent has to re-explore the environment to gather the right experience again, which takes extra training time.

While this effect is also present during the training procedure T , it is possible that resetting one layer makes the exploration problem harder than acting from a randomly initialized network.6.

When completing RL models access to the right experience matters.

To understand this effect better, we allow the retraining procedure access to Rainbow's replay buffer.

At the start of retraining this replay buffer is filled with experience from the fully trained policy.

Figure 5 shows that the model completion hardness becomes much easier with access to this replay buffer: the three left bar plots are lower than the three right.

This result is supported by the benefits of kickstarting BID39 , where a newly trained agent gets access to an expert agent's policy.

Moreover, this is consistent with findings by BID20 , who show performance benefits by adding expert trajectories to the replay buffer.

Our results shed some initial glimpse on the model completion problem and its hardness.

Our findings include: residual networks are easier to complete than non-residual networks, lower layers are often harder to complete than higher layers, and RL models are harder to complete than SL models.

Nevertheless several question remain unanswered: Why is the difference in MC-hardness less pronounced between lower and higher layers in Rainbow and AlexNet than in A3C?

Why is the absolute number of parameters insubstantial?

Are there retraining procedures that are faster than T 1 ?

Furthermore, our definition of hardness of the model completion problem creates an opportunity to modulate the hardness of model completion.

For example, we could devise model architectures with the explicit objective that model completion be easy (to encourage robustness) or hard (to increase security when sharing governance through model splitting).

Importantly, since Equation 1 can be evaluated automatically, we can readily combine this with architecture search BID50 .Our experiments show that when we want to recover 100% of the original performance, model completion may be quite costly: ∼ 40% of the original training costs in many settings; lower performance levels often retrain significantly faster.

In scenarios where a model gets trained over many months or years, 40% of the cost may be prohibitively expensive.

However, this number also has to be taken with a grain of salt because there are many possible retraining procedures that we did not try.

The security properties of model splitting as a method for shared governance require further investigation: in addition to more effective retraining procedures, an attacker may also have access to previous activations or be able to inject their own training data.

Yet our experiments suggest that model splitting could be a promising method for shared governance.

In contrast to MPC and HE it has a substantial advantage because it is cost-competitiveness with normal training and inference.

Learning rate Training batches Retraining batches 5e − 2 0 0 5e − 3 60e3 30e3 5e − 4 90e3 45e3 5e − 5 105e3 72.5e3 Table 1 :

AlexNet: Learning schedule for training and retraining procedures.

1e − 1 0 / 1e − 2 30e3 0 1e − 3 45e3 20e3

AlexNet We train this model for 120e3 batches, with batch size of 256.

We apply batch normalization on the convolutional layers and 2 -regularization of 1e-4.

Optimization is done using Momentum SGD with momentum of 0.9 and the learning rate schedule which is shown in Table 1 .

Note that the learning schedule during retraining is 50% faster than during training (for T 1 and T 2 ).For both retraining procedures T 1 and T 2 , we perform reset for each of the first 5 convolutional layers and the following 2 fully connected layers.

TAB3 shows the number of trainable parameters for each of the layers.

We perform all training and retraining procedures for 60e3 batches, with batch size of 64 and 2 -regularization of 1e-4.

Optimization is done using Momentum SGD with momentum of 0.9 and the learning rate schedule shown in TAB1 .For our experiments, we reinitialize the very first convolutional layer, as well as the first ResNet block for each of the four subsequent network sections.

In the 'S4 B1 -W' experiment, we leave out resetting the learned skip connection.

Finally, we also reset the last fully connected layer containing logits.

A3C Each agent is trained on a single Atari level for 5e7 environment steps, over 10 seeds.

We use the standard Atari architecture consisting of 3 convolutional layers, 1 fully connected layer and 2 fully connected 'heads' for policy and value function.

The number of parameters for each of those layers is shown in TAB5 .

For optimization, we use RMSProp optimizer with = 0.1, decay of 0.99 and α = 6e-4 that is linearly annealed to 0.

For all the other hyperparameters we refer to BID31 .

Finally, while calculating reported statistics we removed the following Atari levels, due to poor behaviour of the trained agent: Montezuma's Revenge, Venture, Solaris, Enduro, Battle Zone, Gravitar, Kangaroo, Skiing, Krull, Video pinball, Freeway, Centipede, and Robotank.

Rainbow Each agent is trained on a single Atari level for 20e6 environment frames, over 3 seeds.

Due to agent behaving randomly, we remove the following Atari games from our MC-hardness calculations: Montezuma's Revenge, Venture, and Solaris.

For our experiments, we use the same network architecture and hyperparameters as reported in BID19 and target the first 3 convolutional layers.

TAB6 has the total number of parameters for each of the 3 layers.

IMPALA We train a single agent over a suite of 28 DeepMind Lab levels for a total of 1 billion steps over all the environments, over 5 seeds.

During training we apply population based training (PBT; BID23 with population of size 12, in order to evolve the entropy cost, learning rate and for RMSProp.

For language modelling a separated LSTM channel is used.

In the results we report, we removed two DeepMind Lab levels due to poor behavior of the trained agent: 'language_execute_random_task' and 'psychlab_visual_search'.

All the other hyperparameters are retained from BID11 .

For our experiments, we reinitialize the first convolutional layer TAB7 .

1.00 1.00 0.00 1.00 1.00

@highlight

We study empirically how hard it is to recover missing parts of trained models