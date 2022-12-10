In partially observable (PO) environments, deep reinforcement learning (RL) agents often suffer from unsatisfactory performance, since two problems need to be tackled together: how to extract information from the raw observations to solve the task, and how to improve the policy.

In this study, we propose an RL algorithm for solving PO tasks.

Our method comprises two parts: a variational recurrent model (VRM) for modeling the environment, and an RL controller that has access to both the environment and the VRM.

The proposed algorithm was tested in two types of PO robotic control tasks, those in which either coordinates or velocities were not observable and those that require long-term memorization.

Our experiments show that the proposed algorithm achieved better data efficiency and/or learned more optimal policy than other alternative approaches in tasks in which unobserved states cannot be inferred from raw observations in a simple manner.

Model-free deep reinforcement learning (RL) algorithms have been developed to solve difficult control and decision-making tasks by self-exploration (Sutton & Barto, 1998; Mnih et al., 2015; Silver et al., 2016) .

While various kinds of fully observable environments have been well investigated, recently, partially observable (PO) environments (Hafner et al., 2018; Igl et al., 2018; Lee et al., 2019; Jaderberg et al., 2019) have commanded greater attention, since real-world applications often need to tackle incomplete information and a non-trivial solution is highly desirable.

There are many types of PO tasks; however, those that can be solved by taking the history of observations into account are more common.

These tasks are often encountered in real life, such as videos games that require memorization of previous events (Kapturowski et al., 2018; Jaderberg et al., 2019) and robotic control using real-time images as input (Hafner et al., 2018; Lee et al., 2019) .

While humans are good at solving these tasks by extracting crucial information from the past observations, deep RL agents often have difficulty acquiring satisfactory policy and achieving good data efficiency, compared to those in fully observable tasks (Hafner et al., 2018; Lee et al., 2019) .

For solving such PO tasks, several categories of methods have been proposed.

One simple, straightforward solution is to include a history of raw observations in the current "observation" (McCallum, 1993; Lee et al., 2019) .

Unfortunately, this method can be impractical when decision-making requires a long-term memory because dimension of observation become unacceptably large if a long history is included.

Another category is based on model-free RL methods with recurrent neural networks (RNN) as function approximators (Schmidhuber, 1990; 1991; Igl et al., 2018; Kapturowski et al., 2018; Jaderberg et al., 2019) , which is usually more tractable to implement.

In this case, RNNs need to tackle two problems simultaneously (Lee et al., 2019) : learning representation (encoded by hidden states of the RNN) of the underlying states of the environment from the state-transition data, and learning to maximize returns using the learned representation.

As most RL algorithms use a bootstrapping strategy to learn the expected return and to improve the policy (Sutton & Barto, 1998) , it is challenging to train the RNN stably and efficiently, since RNNs are relatively more difficult to train (Pascanu et al., 2013) than feedforward neural networks.

The third category considers learning a model of the environment and estimating a belief state, extracted from a sequence of state-transitions (Kaelbling et al., 1998; Ha & Schmidhuber, 2018; Lee et al., 2019) .

The belief state is an agent-estimated variable encoding underlying states of the environment that determines state-transitions and rewards.

Perfectly-estimated belief states can thus be taken as "observations" of an RL agent that contains complete information for solving the task.

Therefore, solving a PO task is segregated into a representation learning problem and a fully observable RL problem.

Since fully observable RL problems have been well explored by the RL community, the critical challenge here is how to estimate the belief state.

In this study, we developed a variational recurrent model (VRM) that models sequential observations and rewards using a latent stochastic variable.

The VRM is an extension of the variational recurrent neural network (VRNN) model (Chung et al., 2015) that takes actions into account.

Our approach falls into the third category by taking the internal states of the VRM together with raw observations as the belief state.

We then propose an algorithm to solve PO tasks by training the VRM and a feed-forward RL controller network, respectively.

The algorithm can be applied in an end-to-end manner, without fine tuning of a hyperparameters.

We then experimentally evaluated the proposed algorithm in various PO versions of robotic control tasks.

The agents showed substantial policy improvement in all tasks, and in some tasks the algorithm performed essentially as in fully observable cases.

In particular, our algorithm demonstrates greater performance compared to alternative approaches in environments where only velocity information is observable or in which long-term memorization is needed.

Typical model-based RL approaches utilize learned models for dreaming, i.e. generating statetransition data for training the agent (Deisenroth & Rasmussen, 2011; Ha & Schmidhuber, 2018; Kaiser et al., 2019) or for planning of future state-transitions (Watter et al., 2015; Hafner et al., 2018; Ke et al., 2019) .

This usually requires a well-designed and finely tuned model so that its predictions are accurate and robust.

In our case, we do not use VRMs for dreaming and planning, but for auto-encoding state-transitions.

Actually, PO tasks can be solved without requiring VRMs to predict accurately (see Appendix E).

This distinguishes our algorithm from typical model-based RL methods.

The work our method most closely resembles is known as stochastic latent actor-critic (SLAC, Lee et al. (2019) ), in which a latent variable model is trained and uses the latent state as the belief state for the critic.

SLAC showed promising results using pixels-based robotic control tasks, in which velocity information needs to be inferred from third-person images of the robot.

Here we consider more general PO environments in which the reward may depend on a long history of inputs, e.g., in a snooker game one has to remember which ball was potted previously.

The actor network of SLAC did not take advantage of the latent variable, but instead used some steps of raw observations as input, which creates problems in achieving long-term memorization of reward-related state-transitions.

Furthermore, SLAC did not include raw observations in the input of the critic, which may complicate training the critic before the model converges.

The scope of problems we study can be formulated into a framework known as partially observable Markov decision processes (POMDP) (Kaelbling et al., 1998) .

POMDPs are used to describe decision or control problems in which a part of underlying states of the environment, which determine state-transitions and rewards, cannot be directly observed by an agent.

A POMDP is usually defined as a 7-tuple (S, A, T, R, X, O, γ), in which S is a set of states, A is a set of actions, and T : S × A → p(S) is the state-transition probability function that determines the distribution of the next state given current state and action.

The reward function R : S × A → R decides the reward during a state-transition, which can also be probabilistic.

Moreover, X is a set of observations, and observations are determined by the observation probability function O : S × A → p(X).

By defining a POMDP, the goal is to maximize expected discounted future rewards t γ t r t by learning a good strategy to select actions (policy function).

Our algorithm was designed for general POMDP problems by learning the representation of underlying states s t ∈ S via modeling observation-transitions and reward functions.

However, it is expected to work in PO tasks in which s t or p(s t ) can be (at least partially) estimated from the history of observations x 1:t .

To model general state-transitions that can be stochastic and complicated, we employ a modified version of the VRNN (Chung et al., 2015) .

The VRNN was developed as a recurrent version of the variational auto-encoder (VAE, Kingma & Welling (2013) ), composed of a variational generation model and a variational inference model.

It is a recurrent latent variable model that can learn to encode and predict complicated sequential observations x t with a stochastic latent variable z t .

The generation model predicts future observations given the its internal states,

where f s are parameterized mappings, such as feed-forward neural networks, and d t is the state variable of the RNN, which is recurrently updated by

The inference model approximates the latent variable z t given x t and d t .

For sequential data that contain T time steps, learning is conducted by maximizing the evidence lower bound ELBO, like that in a VEA (Kingma & Welling, 2013) , where

where p and q are parameterized PDFs of z t by the generative model and the inference model, respectively.

In a POMDP, a VRNN can be used to model the environment and to represent underlying states in its state variable d t .

Thus an RL agent can benefit from a well-learned VRNN model since d t provides additional information about the environment beyond the current raw observation x t .

Soft actor-critic (SAC) is a state-of-the-art model-free RL that uses experience replay for dynamic programming, which been tested on various robotic control tasks and that shows promising performance (Haarnoja et al., 2018a; b) .

A SAC agent learns to maximize reinforcement returns as well as entropy of its policy, so as to obtain more rewards while keeping actions sufficiently stochastic.

A typical SAC implementation can be described as follows.

The state value function V (s), the state-action value function Q(s, a) and the policy function π(a|s) are parameterized by neural networks, indicated by ψ, λ, η, respectively.

Also, an entropy coefficient factor (also known as the temperature parameter), denoted by α, is learned to control the degree of stochasticity of the policy.

The parameters are learned by simultaneously minimizing the following loss functions.

where B is the replay buffer from which s t is sampled, and H tar is the target entropy.

To compute the gradient of J π (η) (Equation.

7), the reparameterization trick (Kingma & Welling, 2013 ) is used on action, indicated by a η (s t ).

Reparameterization of action is not required in minimizing J(α) (Equation.

8) since log π η (a|s t ) does not depends on α.

SAC was originally developed for fully observable environments; thus, the raw observation at the current step x t was used as network input.

In this work, we apply SAC in PO tasks by including the state variable d t of the VRNN in the input of function approximators of both the actor and the critic.

An overall diagram of the proposed algorithm is summarized in Fig. 1(a) , while a more detailed computational graph is plotted in Fig. 2 .

We extend the original VRNN model (Chung et al., 2015) to the proposed VRM model by adding action feedback, i.e., actions taken by the agent are used in the inference model and the generative model.

Also, since we are modeling state-transition and reward functions, we include the reward r t−1 in the current raw observation x t for convenience.

Thus, we have the inference model ( Fig. 1(c) ), denoted by φ, as

The generative model ( Fig. 1(b) ), denoted by θ here, is

For building recurrent connections, the choice of RNN types is not limited.

In our study, the longshort term memory (LSTM) (Hochreiter & Schmidhuber, 1997 ) is used since it works well in general cases.

So we have As in training a VRNN, the VRM is trained by maximizing an evidence lower bound ( Fig. 1(c) )

(11) In practice, the first term E q φ [log p θ (x t |z 1:t , x 1:t−1 )] can be obtained by unrolling the RNN using the inference model ( Fig. 1(c) ) with sampled sequences of x t .

Since q φ and p θ are parameterized Gaussian distributions, the KL-divergence term can be analytically expressed as

For computation efficiency in experience replay, we train a VRM by sampling minibatchs of truncated sequences of fixed length, instead of whole episodes.

Details are found in Appendix A.1.

Since training of a VRM is segregated from training of the RL controllers, there are several strategies for conducting them in parallel.

For the RL controller, we adopted a smooth update strategy as in Haarnoja et al. (2018a) , i.e., performing one time of experience replay every n steps.

To train the VRM, one can also conduct smooth update.

However, in that case, RL suffers from instability of the representation of underlying states in the VRM before it converges.

Also, stochasticity of RNN state variables d can be meaninglessly high at early stage of training, which may create problems in RL.

Another strategy is to pre-train the VRM for abundant epochs only before RL starts, which unfortunately, can fail if novel observations from the environment appear after some degree of policy improvement.

Moreover, if pre-training and smooth update are both applied to the VRM, RL may suffer from a large representation shift of the belief state.

To resolve this conflict, we propose using two VRMs, which we call the first-impression model and the keep-learning model, respectively.

As the names suggest, we pre-train the first-impression model and stop updating it when RL controllers and the keep-learning model start smooth updates.

Then we take state variables from both VRMs, together with raw observations, as input for the RL controller.

We found that this method yields better overall performance than using a single VRM (Appendix C).

Initialize the first-impression VRM M f and the keep-learning VRM M k , the RL controller C, and the replay buffer D, global step t ← 0.

repeat Initialize an episode, assign M with zero initial states.

while episode not terminated do Sample an action a t from π(a t |d t , x t ) and execute a t , t ← t + 1.

Compute 1-step forward of both VRMs using inference models.

if t == step start RL then For N epochs, sample a minibatch of samples from B to update M f (Eq. 11). end if if t > step start RL and mod(t, train interval KLV RM ) == 0 then Sample a minibatch of samples from B to update M k (Eq. 5, 6, 7, 8) . end if if t > step start RL and mod(t, train interval RL) == 0 then Sample a minibatch of samples from B to update R (Eq. 11) .

end if end while until training stopped

As shown in Fig. 1(a) , we use multi-layer perceptrons (MLP) as function approximators for V , Q, respectively.

Inputs for the Q t network are (x t , d t , a t ), and V t is mapped from (x t , d t ).

Following Haarnoja et al. (2018a), we use two Q networks λ 1 and λ 2 and compute Q = min(Q λ1 , Q λ2 ) in Eq. 5 and 7 for better performance and stability.

Furthermore, we also used a target value network for computing V in Eq. 6 as in Haarnoja et al. (2018a) .

The policy function π η follows a parameterized

where µ η and σ η are also MLPs.

In the execution phase ( Fig. 1(b) ), observation and reward x t = (X t , r t−1 ) are received as VRM inputs to compute internal states d t using inference models.

Then, the agent selects an action, sampled from π η (a t |d t , x t ), to interact with the environment.

To train RL networks, we first sample sequences of steps from the replay buffer as minibatches; thus, d t can be computed by the inference models using recorded observationsx t and actionsā t (See Appendix A.1.2).

Then RL networks are updated by minimizing the loss functions with gradient descent.

Gradients stop at d t so that training of RL networks does not involve updating VRMs.

To empirically evaluate our algorithm, we performed experiments in a range of (partially observable) continuous control tasks and compared it to the following alternative algorithms.

The overall procedure is summarized in Algorithm 1.

For the RL controllers, we adopted hyperparameters from the original SAC implementation (Haarnoja et al., 2018b) .

Both the keep-learning and first-impression VRMs were trained using learning rate 0.0008.

We pre-trained the first-impression VRM for 5,000 epochs, and updated the keep-learning VRM every 5 steps.

Batches of size 4, each containing a sequence of 64 steps, were used for training both the VRMs and the RL controllers.

All tasks used the same hyperparameters (Appendix A.1).

• SAC-MLP: The vanilla soft actor-critic implementation (Haarnoja et al., 2018a; b) , in which each function is approximated by a 2-layer MLP taking raw observations as input.

• SAC-LSTM: Soft actor-critic with recurrent networks as function approximators, where raw observations are processed through an LSTM layer followed by 2 layers of MLPs.

This allows the agent to make decisions based on the whole history of raw observations.

In this case, the network has to conduct representation learning and dynamic programming collectively.

Our algorithm is compared with SAC-LSTM to demonstrate the effect of separating representation learning from dynamic programming.

Note that in our algorithm, we apply pre-training of the first-impression model.

For a fair comparison, we also perform pre-training for the alternative algorithm with the same epochs.

For SAC-MLP and SAC-LSTM, pre-training is conducted on RL networks; while for SLAC, its model is pre-trained.

The Pendlum and CartPole (Barto et al., 1983) tasks are the classic control tasks for evaluating RL algorithms (Fig. 3, Left) .

The CartPole task requires learning of a policy that prevents the pole from falling down and keeps the cart from running away by applying a (1-dimensional) force to the cart, in which observable information is the coordinate of the cart, the angle of the pole, and their derivatives w.r.t time (i.e., velocities).

For the Pendulum task, the agent needs to learn a policy to swing an inverse-pendulum up and to maintain it at the highest position in order to obtain more rewards.

We are interested in classic control tasks because they are relatively easy to solve when fully observable, and thus the PO cases can highlight the representation learning problem.

Experiments were performed in these two tasks, as well as their PO versions, in which either velocities cannot be observed or only velocities can be observed.

The latter case is meaningful in real-life applications because an agent may not be able to perceive its own position, but can estimate its speed.

As expected, SAC-MLP failed to solve the PO tasks (Fig. 3) .

While our algorithm succeeded in learning to solve all these tasks, SAC-LSTM showed poorer performance in some of them.

In particular, in the pendulum task with only angular velocity observable, SAC-LSTM may suffer from the periodicity of the angle.

SLAC performed well in the CartPole tasks, but showed less satisfactory sample efficiency in the Pendulum tasks.

To examine performance of the proposed algorithm in more challenging control tasks with higher degrees of freedom (DOF), we also evaluated performance of the proposed algorithm in the OpenAI Roboschool environments (Brockman et al., 2016) .

The Roboschool environments include a number of continuous robotic control tasks, such as teaching a multiple-joint robot to walk as fast as possible without falling down (Fig. 4, Left) .

The original Roboschool environments are nearly fully observable since observations include the robot's coordinates and (trigonometric functions of) joint angles, as well as (angular and coordinate) velocities.

As in the PO classic control tasks, we also performed experiments in the PO versions of the Roboschool environments.

Using our algorithm, experimental results (Fig. 4) demonstrated substantial policy improvement in all PO tasks (visualization of the trained agents is in Appendix D).

In some PO cases, the agents achieved comparable performance to that in fully observable cases.

For tasks with unobserved velocities, our algorithm performed similarly to SAC-LSTM.

This is because velocities can be simply estimated by one-step differences in robot coordinates and joint angles, which eases representation learning.

However, in environments where only velocities can be observed, our algorithm significantly outperformed SAC-LSTM, presumably because SAC-LSTM is less efficient at encoding underlying states from velocity observations.

Also, we found that learning of a SLAC agent was unstable, i.e., it sometimes could acquire a near-optimal policy, but often its policy converged to a poor one.

Thus, average performance of SLAC was less promising than ours in most of the PO robotic control tasks.

Another common type of PO task requires long-term memorization of past events.

To solve these tasks, an agent needs to learn to extract and to remember critical information from the whole history of raw observations.

Therefore, we also examined our algorithm and other alternatives in a long-term memorization task known as the sequential target reaching task (Han et al., 2019) , in which a robot agent needs to reach 3 different targets in a certain sequence (Fig. 5, Left) .

The robot can control its two wheels to move or turn, and will get one-step small, medium, and large rewards when it reaches the first, second, and third targets, respectively, in the correct sequence.

The robot senses distances and angles from the 3 targets, but does not receive any signal indicating which target to reach.

In each episode, the robot's initial position and those of the three targets are randomly initialized.

In order to obtain rewards, the agent needs to infer the current correct target using historical observations.

We found that agents using our algorithm achieved almost 100% success rate (reaching 3 targets in the correct sequence within maximum steps).

SAC-LSTM also achieved similar success rate after convergence, but spent more training steps learning to encode underlying goal-related information from sequential observations.

Also, SLAC struggled hard to solve this task since its actor only received a limited steps of observations, making it difficult to infer the correct target.

One of the most concerned problems of our algorithm is that input of the RL controllers can experience representation change, because the keep-learning model is not guaranteed to converge if novel observation appears due to improved policy (e.g. for a hopper robot, "in-the-air" state can only happen after it learns to hop).

To empirically investigate how convergence of the keep-learning VRM affect policy improvement, we plot the loss functions (negative ELBOs) of the the keep-learning VRM for 3 example tasks (Fig. 6 ).

For a simpler task (CartPole), the policy was already near optimal before the VRM fully converged.

We also saw that the policy was gradually improved after the VRM mostly converged (RoboschoolAnt -no velocities), and that the policy and the VRM were being improved in parallel (RoboschoolAnt -velocities only).

The results suggested that policy could be improved with sufficient sample efficiency even the keep-learning VRM did not converge.

This can be explained by that the RL controller also extract information from the first-impression model and the raw observations, which did not experience representation change during RL.

Indeed, our ablation study showed performance degradation in many tasks without the first-impression VRM (Appendix C).

In this paper, we proposed a variational recurrent model for learning to represent underlying states of PO environments and the corresponding algorithm for solving POMDPs.

Our experimental results demonstrate effectiveness of the proposed algorithm in tasks in which underlying states cannot be simply inferred using a short sequence of observations.

Our work can be considered an attempt to understand how RL benefits from stochastic Bayesian inference of state-transitions, which actually happens in the brain (Funamizu et al., 2016) , but has been considered less often in RL studies.

We used stochastic models in this work which we actually found perform better than deterministic ones, even through the environments we used are deterministic (Appendix C).

The VRNN can be replaced with other alternatives (Bayer & Osendorfer, 2014; Goyal et al., 2017) to potentially improve performance, although developing model architecture is beyond the scope of the current study.

Moreover, a recent study (Ahmadi & Tani, 2019) showed a novel way of inference using back-propagation of prediction errors, which may also benefit our future studies.

Many researchers think that there are two distinct systems for model-based and model-free RL in the brain (Gläscher et al., 2010; Lee et al., 2014) and a number of studies investigated how and when the brain switches between them (Smittenaar et al., 2013; Lee et al., 2014) .

However, Stachenfeld et al. (2017) suggested that the hippocampus can learn a successor representation of the environment that benefits both model-free and model-based RL, contrary to the aforementioned conventional view.

We further propose another possibility, that a model is learned, but not used for planning or dreaming.

This blurs the distinction between model-based and model-free RL.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.

Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.

In International Conference on Machine Learning, pp.

1856-1865, 2018a.

In this section we describe the details of implementing our algorithm as well as the alternative ones.

Summaries of hyperparameters can be found in Table 1 and 2.

The first-impression model and the keep-learning model adopted the same architecture.

Size of d and z is 256 and 64, respectively.

We used one-hidden-layer fully-connected networks with 128 hidden neurons for the inference models µ φ,t , σ 2 φ,t = φ(x t , d t−1 , a t−1 ), as well as for µ θ,t , σ 2 θ,t = θ prior (d t−1 , a t−1 ) in the generative models.

For the decoder µ x,t , σ 2 x,t = θ decoder (z t , d t−1 ) in the generative models, we used 2-layers MLPs with 128 neurons in each layer.

The input processing layer f x is also an one-layer MLP with size-128.

For all the Gaussian variables, output functions for mean are linear and output functions for variance are softplus.

Other activation functions of the VRMs are tanh.

The RL controllers are the same as those in SAC-MLP (Section A.2.1) except that network inputs are raw observations together with the RNN states from the first-impression model and the keep-learning model.

To train the VRMs, one can use a number of entire episodes as a mini-batch, using zero initial states, as in Heess et al. (2015) .

However, when tackling with long episodes (e.g. there can be 1,000 steps in each episode in the robotic control tasks we used) or even infinite-horizon problems, the computation consumption will be huge in back-propagation through time (BPTT).

For better computation efficiency, we used 4 length-64 sequences for training the RNNs, and applied the burn-in method for providing the initial states (Kapturowski et al., 2018) , or more specifically, unrolling the RNNs using a portion of the replay sequence (burn-in period, up to 64 steps in our case) from zero initial states.

We assume that proper initial states can be obtained in this way.

This is crucial for the tasks that require long-term memorization, and is helpful to reduce bias introduces by incorrect initial states in general cases.

A.2.1 SAC-MLP We followed the original implementation of SAC in (Haarnoja et al., 2018a) including hyperparameters.

However, we also applied automatic learning of the entropy coefficient α (inverse of the the reward scale in Haarnoja et al. (2018a) ) as introduced by the authors in Haarnoja et al. (2018b) to avoid tuning the reward scale for each task.

To apply recurrency to SAC's function approximators, we added an LSTM network with size-256 receiving raw observations as input.

The function approximators of actor and critic were the same as those in SAC except receiving the LSTM's output as input.

The gradients can pass through the LSTM so that the training of the LSTM and MLPs were synchronized.

The training the network also followed Section A.1.2.

We mostly followed the implementation of SLAC explained in the authors' paper (Lee et al., 2019) .

One modification is that since their work was using pixels as observations, convolutional neural networks (CNN) and transposed CNNs were chosen for input feature extracting and output decoding layers; in our case, we replaced the CNN and transposed CNNs by 2-layers MLPs with 256 units in each layer.

In addition, the authors set the output variance σ 2 y,t for each image pixel as 0.1.

However, σ 2 y,t = 0.1 can be too large for joint states/velocities as observations.

We found that it will lead to better performance by setting σ y,t as trainable parameters (as that in our algorithm).

We also used a 2-layer MLP with 256 units for approximating σ y (x t , d t−1 ).

To avoid network weights being divergent, all the activation functions of the model were tanh except those for outputs.

For the robotic control tasks and the Pendulum task, we used environments (and modified them for PO versions) from OpenAI Gym (Brockman et al., 2016) .

The CartPole environment with a continuous action space was from Danforth (2018) , and the codes for the sequential target reaching tasks were provided by the authors (Han et al., 2019) .

In the no-velocities cases, velocity information was removed from raw observations; while in the velocities-only cases, only velocity information was retained in raw observations.

We summarize key information of each environment in Table 3 .

The performance curves were obtained in evaluation phases in which agents used same policy but did not update networks or record state-transition data.

Each experiment was repeated using 5 different random seeds.

This section demonstrated a ablation study in which we compared the performance of the proposed algorithm to the same but with some modification:

• With a single VRM.

In this case, we used only one VRM and applied both pre-training and smooth update to it.

• Only first-impression model.

In this case, only the first-impression model was used and pre-trained.

• Only keep-learning model.

In this case, only the keep-learning model was used and smooth-update was applied.

• Deterministic model.

In this case, the first-imporession model and the keep-learning model were deterministic RNNs which learned to model the state-transitions by minimizing mean-square error between prediction and observations instead of ELBO.

The network architecture was mostly the same as the VRM expect that the inference model and the generative model were merged into a deterministic one.

The learning curves are shown in Fig. 7 .

It can be seen that the proposed algorithm consistently performed similar as or better than the modified ones.

Here we show actual movements of the trained robots in the PO robotic control tasks (Fig. 8) .

It can be seen that the robots succeeded in learning to hop or walk, although their policy may be sub-optimal.

As we discussed in Section 2, our algorithm relies mostly on encoding capacity of models, but does not require models to make accurate prediction of future observations.

Fig. 9 shows open-loop (using the inference model to compute the latent variable z) and close-loop (purely using the generative model) prediction of raw observation by the keep-learning models of randomly selected trained agents.

Here we showcase "RoboschoolHopper -velocities only" and "Pendulum -no velocities" because in these tasks our algorithm achieved similar performance to those in fully-observable versions (Fig. 4) , although the prediction accuracy of the models was imperfect.

To empirically show how choice of hyperparameters of the VRMs affect RL performance, we conducted experiments using hyperparameters different from those used in the main study.

More specifically, the learning rate for both VRMs was randomly selected from {0.0004, 0.0006, 0.0008, 0.001} and the sequence length was randomly selected from {16, 32, 64} (the batch size was 256/(sequence length) to ensure that the total number of samples in a batch was 256 which matched with the alternative approaches).

The other hyperparameters were unchanged.

The results can be checked in Fig 10 for all the environments we used.

The overall performance did not significantly change using different, random hyperparameters of the VRMs, although we could observe significant performance improvement (e.g. RoboshoolWalker2d) or degradation (e.g. RoboshoolHopper -velocities only) in a few tasks using different haperparameters.

Therefore, the representation learning part (VRMs) of our algorithm does not suffer from high sensitivity to hyperparameters.

This can be explained by the fact that we do not use a bootstrapping (e.g. the estimation of targets of value functions depends on the estimation of value functions) (Sutton & Barto, 1998) update rule to train the VRMs.

G SCALABILITY Table 4 showed scalability of our algorithm and the alternative ones.

Table 4 : Wall-clock time and number of parameters of our algorithm and the alternative ones.

The working environment was a desktop computer using Intel i7-6850K CPU and the task is "Velocitiesonly RoboschoolHopper".

The wall-clock time include training the first-impression VRM or pretrainings.

<|TLDR|>

@highlight

A deep RL algorithm for solving POMDPs by auto-encoding the underlying states using a variational recurrent model