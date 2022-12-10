Recent research efforts enable study for natural language grounded navigation in photo-realistic environments, e.g., following natural language instructions or dialog.

However, existing methods tend to overfit training data in seen environments and fail to generalize well in previously unseen environments.

In order to close the gap between seen and unseen environments, we aim at learning a generalizable navigation model from two novel perspectives: (1) we introduce a multitask navigation model that can be seamlessly trained on both Vision-Language Navigation (VLN) and Navigation from Dialog History (NDH) tasks, which benefits from richer natural language guidance and effectively transfers knowledge across tasks; (2) we propose to learn environment-agnostic representations for navigation policy that are invariant among environments, thus generalizing better on unseen environments.

Extensive experiments show that our environment-agnostic multitask navigation model significantly reduces the performance gap between seen and unseen environments and outperforms the baselines on unseen environments by 16% (relative measure on success rate) on VLN and 120% (goal progress) on NDH, establishing the new state of the art for NDH task.

Navigation in visual environments by following natural language guidance (Hemachandra et al., 2015) is a fundamental capability of intelligent robots that simulate human behaviors, because humans can easily reason about the language guidance and navigate efficiently by interacting with the visual environments.

Recent efforts (Anderson et al., 2018b; Das et al., 2018; Thomason et al., 2019) empower large-scale learning of natural language grounded navigation that is situated in photorealistic simulation environments.

Nevertheless, the generalization problem commonly exists for these tasks, especially indoor navigation: the agent usually performs poorly on unknown environments that have never been seen during training.

One of the main causes for such behavior is data scarcity as it is expensive and time-consuming to extend either visual environments or natural language guidance.

The number of scanned houses for indoor navigation is limited due to high expense and privacy concerns.

Besides, unlike vision-only navigation tasks (Mirowski et al., 2018; Xia et al., 2018; Manolis Savva* et al., 2019; Kolve et al., 2017) where episodes can be exhaustively sampled in simulation, natural language grounded navigation is supported by human demonstrated interaction and communication in natural language.

It is impractical to fully collect and cover all the samples for individual tasks.

Therefore, it is essential though challenging to efficiently learn a more generalized policy for natural language grounded navigation tasks from existing data (Wu et al., 2018a; b) .

In this paper, we study how to resolve the generalization and data scarcity issues from two different angles.

First, previous methods are trained for one task at the time, so each new task requires training a brand new agent instance that can only solve the one task it was trained on.

In this work, we propose a generalized multitask model for natural language grounded navigation tasks such as Vision-Language Navigation (VLN) and Navigation from Dialog History (NDH), aiming at efficiently transferring knowledge across tasks and effectively solving both tasks with one agent simultaneously.

Moreover, although there are thousands of trajectories paired with language guidance, the underlying house scans are restricted.

For instance, the popular Matterport3D dataset (Chang et al., 2017) contains only 61 unique house scans in the training set.

The current models perform much better in seen environments by taking advantage of the knowledge of specific houses they have acquired over multiple task completions during training, but fail to generalize to houses not seen during training.

Hence we propose an environment-agnostic learning method to learn a visual representation that is invariant to specific environments but still able to support navigation.

Endowed with the learned environment-agnostic representations, the agent is further prevented from the overfitting issue and generalizes better on unseen environments.

To the best of our knowledge, we are the first to introduce natural language grounded multitask and environment-agnostic training regimes and validate their effectiveness on VLN and NDH tasks.

Extensive experiments demonstrate that our environment-agnostic multitask navigation model can not only efficiently execute different language guidance in indoor environments but also outperform the single-task baseline models by a large margin on both tasks.

Besides, the performance gap between seen and unseen environments is significantly reduced.

We also set a new state of the art on NDH with over 120% improvement in terms of goal progress.

Vision-and-Language Navigation.

Vision-and-Language Navigation (Anderson et al., 2018b; Chen et al., 2019) task requires an embodied agent to navigate in photo-realistic environments to carry out natural language instructions.

The agent is spawned at an initial pose p 0 = (v 0 , φ 0 , θ 0 ), which includes the spatial location, heading and elevation angles.

Given a natural language instruction X = {x 1 , x 2 , ..., x n }, the agent is expected to perform a sequence of actions {a 1 , a 2 , ..., a T } and arrive at the target position v tar specified by the language instruction X, which describes stepby-step instructions from the starting position to the target position.

In this work, we consider VLN task defined for Room-to-Room (R2R) (Anderson et al., 2018b ) dataset which contains instructiontrajectory pairs across 90 different indoor environments (houses).

Previous VLN methods have studied various aspects to improve the navigation performance, such as planning , data augmentation (Fried et al., 2018; Tan et al., 2019) , cross-modal alignment (Wang et al., 2019; Huang et al., 2019b) , progress estimation (Ma et al., 2019a) , error correction (Ma et al., 2019b; Ke et al., 2019) , interactive language assistance Nguyen & Daumé III, 2019) etc.

This work tackles VLN via multitask learning and environmentagnostic learning, which is orthogonal to all these prior arts.

Navigation from Dialog History.

Different from Visual Dialog (Das et al., 2017) which involves dialog grounded in a single image, the recently introduced Cooperative Vision-and-Dialog Navigation (CVDN) dataset (Thomason et al., 2019) includes interactive language assistance for indoor navigation, which consists of over 2,000 embodied, human-human dialogs situated in photo-realistic home environments.

The task of Navigation from Dialog History (NDH) is defined as: given a target object t 0 and a dialog history between humans cooperating to perform the task, the embodied agent must infer navigation actions towards the goal room that contains the target object.

The dialog history is denoted as < t 0 , Q 1 , A 1 , Q 2 , A 2 , ..., Q i , A i >, including the target object t 0 , the questions Q and answers A till the turn i (0 ≤ i ≤ k, where k is the total number of Q-A turns from the beginning to the goal room).

The agent, located in p 0 , is trying to move closer to the goal room by inferring from the dialog history that happened before.

Multitask Learning.

The basis of Multitask (MT) learning is the notion that tasks can serve as mutual sources of inductive bias for each other (Caruana, 1993) .

When multiple tasks are trained jointly, MT learning causes the learner to prefer the hypothesis that explains all the tasks simultaneously, hence leading to more generalized solutions.

MT learning has been successful in natural language processing (Collobert & Weston, 2008) , speech recognition (Deng et al., 2013) , computer vision (Girshick, 2015) , drug discovery (Ramsundar et al., 2015) , and Atari games (Teh et al., 2017) .

The deep reinforcement learning methods that have become very popular for training models on natural language grounded navigation tasks (Wang et al., 2019; Huang et al., 2019a; b; Tan et al., 2019) are known to be data inefficient.

In this work, we introduce multitask reinforcement learning for such tasks to improve data efficiency by positive transfer across related tasks.

Environment-agnostic Learning.

A few studies on agnostic learning have been proposed recently.

For example, Model-Agnostic Meta-Learning (MAML) (Finn et al., 2017) aims to train a model on a variety of learning tasks and solve a new task using only a few training examples.

Liu et al. (2018) proposes a unified feature disentangler that learns domain-invariant representation across multiple domains for image translation.

Other domain-agnostic techniques are also proposed for supervised (Li et al., 2018) and unsupervised domain adaption (Romijnders et al., 2019; Peng et al., 2019) .

In this work, we pair the environment classifier with a gradient reversal layer (Ganin & Lempitsky, 2015) to learn an environment-agnostic representation that can be better generalized on unseen environments in a zero-shot fashion where no adaptation is involved.

Distributed Actor-Learner Navigation Learning Framework.

To train models for the various language grounded navigation tasks like VLN and NDH, we develop a distributed actor-learner learning infrastructure 1 .

The framework design is inspired by IMPALA (Espeholt et al., 2018 ) and uses its off-policy correction method called V-trace to efficiently scale reinforcement learning methods to thousands of machines.

The framework additionally supports a variety of supervision strategies important for navigation tasks such as teacher-forcing (Anderson et al., 2018b) , studentforcing (Anderson et al., 2018b) and mixed supervision (Thomason et al., 2019) .

The framework is built using TensorFlow (Abadi et al., 2016) and supports ML accelerators (GPU, TPU).

3.1 OVERVIEW Our environment-agnostic multitask navigation model is illustrated in Figure 1 .

First, we adapt the reinforced cross-modal matching (RCM) model (Wang et al., 2019) and make it seamlessly transfer across tasks by sharing all the learnable parameters for both NDH and VLN, including joint word embedding layer, language encoder, trajectory encoder, cross-modal attention module (CM-ATT), and action predictor.

Furthermore, to learn the environment-agnostic representation z t , we equip the navigation model with an environment classifier whose objective is to predict which house the agent is.

But note that between trajectory encoder and environment classifier, a gradient reversal layer (Ganin & Lempitsky, 2015) is introduced to reverse the gradients backpropagated to the trajectory encoder, making it learn representations that are environment-agnostic and thus more generalizable in unseen environments.

During training, the environment classifier is minimizing the environment classification loss L env , while the trajectory encoder is maximizing L env and minimizing the navigation loss L nav .

The other modules are optimized with the navigation loss L nav simultaneously.

Below we introduce multitask reinforcement learning and environmentagnostic representation learning.

A more detailed model architecture is presented in Section 4.

Interleaved Multitask Data Sampling.

To avoid overfitting to individual tasks, we adopt an interleaved multitask data sampling strategy to train the model.

Particularly, each data sample within a mini-batch can be from either task, so that the VLN instruction-trajectory pairs and NDH dialogtrajectory pairs are interleaved in a mini-batch though they may have different learning objectives.

Reward Shaping.

Following prior art , we first implement a discounted cumulative reward function R for the VLN and NDH tasks:

where γ is the discounted factor, d(s t , v tar ) is the distance between state s t and the target location v tar , and d th is the maximum distance from v tar that the agent is allowed to terminate for success.

Different from VLN, NDH is essentially room navigation instead of point navigation because the agent is expected to reach a room that contains the target object.

Suppose the goal room is occupied by a set of nodes {v i } N 1 , we replace the distance function d(s t , v tar ) in Equation 1 with the minimum distance to the goal room

Navigation Loss.

Since human demonstrations are available for both VLN and NDH tasks, we use behavior cloning to constrain the learning algorithm to model state-action spaces that are most relevant to each task.

Following previous works (Wang et al., 2019) , we also use reinforcement learning to aid the agent's ability to recover from erroneous actions in unseen environments.

During multitask navigation model training, we adopt a mixed training strategy of reinforcement learning and behavior cloning, so the navigation loss function is:

where we use REINFORCE policy gradients (Williams, 1992) and supervised learning gradients to update the policy π.

b is the estimated baseline to reduce the variance and a * t is the human demonstrated action.

To further improve the generalizability of the navigation policy, we propose to learn a latent environment-agnostic representation that is invariant among seen environments.

We would like to get rid of the environment-specific features that are irrelevant to general navigation (e.g. unique house appearances), preventing the model from overfitting to specific seen environments.

We can reformulate the navigation policy as

where z t is a latent representation.

As shown in Figure 1 , p(a t |z t , s t ) is modeled by the policy module (including CM-ATT and action predictor) and p(z t |s t ) is modeled by the trajectory encoder.

In order to learn the environmentagnostic representation, we employ an environment classifier and a gradient reversal layer (Ganin & Lempitsky, 2015) .

The environment classifier is parameterized to predict the identity of the house where the agent is, so its loss function L env is defined as

where y * is the ground-truth house label.

The gradient reversal layer has no parameters.

It acts as an identity transform during forward-propagation, but multiplies the gradient by −λ and passes it to the trajectory encoder during back-propagation.

Therefore, in addition to minimizing the navigation loss L nav , the trajectory encoder is also maximizing the environment classification loss L env , trying to increase the entropy of the classifier in an adversarial learning manner where the classifier is minimizing the classification loss conditioned on the latent representation z t .

Language Encoder.

The natural language guidance (instruction or dialog) is tokenized and embedded into n-dimensional space X = {x 1 , x 2 , ..., x 3 } where the word vectors x i are initialized randomly.

The vocabulary is restricted to tokens that occur at least five times in the training instructions (The vocabulary used when jointly training VLN and NDH tasks is the union of the two tasks' vocabularies.).

All out-of-vocabulary tokens are mapped to a single out-of-vocabulary identifier.

The token sequence is encoded using a bi-directional LSTM (Schuster & Paliwal, 1997) to create H X following:

where − → h X t and ← − h X t are the hidden states of the forward and backward LSTM layers at time step t respectively, and the σ function is used to combine − → h X t and

Similar to benchmark models (Fried et al., 2018; Wang et al., 2019; Huang et al., 2019b) , at each time step t, the agent perceives a 360-degree panoramic view at its current location.

The view is discretized into k view angles (k = 36 in our implementation, 3 elevations by 12 headings at 30-degree intervals).

The image at view angle i, heading angle φ and elevation angle θ is represented by a concatenation of the pre-trained CNN image features with the 4-dimensional orientation feature [sin φ; cos φ; sin θ; cos θ] to form v t,i .

The visual input sequence V = {v 1 , v 2 , ..., v m } is encoded using a LSTM to create H V following:

is the attention-pooled representation of all view angles using previous agent state h t−1 as the query.

We use the dot-product attention (Vaswani et al., 2017) hereafter.

Policy Module.

The policy module comprises of cross-modal attention (CM-ATT) unit as well as an action predictor.

The agent learns a policy π θ over parameters θ that maps the natural language instruction X and the initial visual scene v 1 to a sequence of actions [a 1 , a 2 , ..., a n ].

The action space which is common to VLN and NDH tasks consists of navigable directions from the current location.

The available actions at time t are denoted as u t,1..l , where u t,j is the representation of the navigable direction j from the current location obtained similarly to v t,i .

The number of available actions, l, varies per location, since graph node connectivity varies.

As in Wang et al. (2019) , the model predicts the probability p d of each navigable direction d using a bilinear dot product:

where c Environment Classifier.

The environment classifier is a two-layer perceptron with a SoftMax layer as the last layer.

Given the latent representation z t (which is h V t in our setting), the classifier generates a probability distribution over the house labels.

Implementation Details.

In the experiments, we use a 2-layer bi-directional LSTM for the instruction encoder where the size of LSTM cells is 256 units in each direction.

The inputs to the encoder are 300-dimensional embeddings initialized randomly.

For the visual encoder, we use a 2-layer LSTM with a cell size of 512 units.

The encoder inputs are image features derived as mentioned in Section 4.

The cross-modal attention layer size is 128 units.

The environment classifier has one hidden layer of size 128 units followed by an output layer of size equal to the number of classes.

During training, some episodes in the batch are identical to available human demonstrations in the training dataset where the objective is to increase the agent's likelihood of choosing human actions (behavioral cloning (Bain & Sammut, 1999) ).

The rest of the episodes are constructed by sampling Figure 2: Selected tokens from the vocabulary for VLN (left) and NDH (right) tasks which gained more than 40 additional occurrences in the training dataset due to joint-training.

from agent's own policy.

In the experiments, unless otherwise stated, we use entire dialog history from NDH task for model training.

All the reported results in subsequent studies are averages of at least 3 independent runs.

Evaluation Metrics.

The agents are evaluated on two datasets, namely Validation Seen that contains new paths from the training environments and Validation Unseen that contains paths from previously unseen environments.

The evaluation metrics for VLN task are as follows: Path Length (PL) measures the total length of the predicted path; Navigation Error (NE) measures the distance between the last nodes in the predicted and the reference paths; Success Rate (SR) measures how often the last node in the predicted path is within some threshold distance of the last node in the reference path; Success weighted by Path Length (SPL) (Anderson et al., 2018a) measures Success Rate weighted by the normalized Path Length; and Coverage weighted by Length Score (CLS) measures predicted path's conformity to the reference path weighted by length score.

For NDH task, the agent's progress is defined as reduction (in meters) from the distance to the goal region at agent's first position versus at its last position (Thomason et al., 2019) .

Table 1 shows the results of training the navigation model using environment-agnostic learning (EnvAg) as well as multitask learning (MT-RCM).

First, both learning methods independently help the agent learn more generalized navigation policy as is evidenced by significant reduction in agent's performance gap between seen and unseen environments.

For instance, performance gap for agent's goal progress on NDH task drops from 3.85m to 0.92m using multitask learning and agent's success rate on VLN task between seen and unseen datasets drops from 9.26% to 8.39% using environmentagnostic learning.

Second, the two techniques are complementary-the agent's performance when trained with both the techniques simultaneously improves on unseen environments compared to when trained separately.

Finally, we note here that MT-RCM + EnvAg outperforms the state-of-theart goal progress of 2.10m (Thomason et al., 2019) on NDH validation unseen dataset by more than 120%.

At the same time, it outperforms the equivalent RCM baseline (Wang et al., 2019 ) of 40.6% success rate by more than 16% (relative measure) on VLN validation unseen dataset.

Next, we conduct studies to examine cross-task transfer using multitask learning alone.

One of the main advantages of multitask learning is that under-represented tokens in each of the individual tasks get a significant boost in the number of training samples.

Figure 2 illustrates that tokens with less than 40 occurrences end up with sometimes more than 300 occurrences during joint-training.

To examine the impact of dialog history in NDH task, we conduct studies with access to different parts of the dialog-the target object t o , the last oracle answer A i , the prefacing navigator question Q i and the full dialog history.

Table 2 shows the results of jointly training MT-RCM model on VLN and NDH tasks.

MT-RCM model learns a generalized policy that consistently outperforms the competing model with access to similar parts of the dialog on previously unseen environments.

As noted before, multitask learning significantly reduces the gap between the agent's performance on previously seen and unseen environments for both tasks.

Furthermore, we see a consistent and gradual increase in the success rate of MT-RCM on VLN task as it is trained on paths with richer dialog history from the NDH task.

This shows that the agent benefits from more complete information about the path implying the importance given by the agent to the language instructions in the task.

We also investigate the impact of parameter sharing of the language encoder for both tasks.

As shown in Table 3 , the model with shared language encoder for NDH and VLN tasks outperforms the model that has separate language encoders for the two tasks, hence demonstrating the importance of parameter sharing during multitask learning.

A more detailed analysis can be found in the Appendix.

From Table 1 , it can be seen that both VLN and NDH tasks benefit from environment-agnostic learning independently.

To further examine the generalization property due to environment-agnostic objective, we train a model with the opposite objective-learn to correctly predict the navigation environments by removing the gradient reversal layer (environment-aware learning).

Interesting Figure 3 : t-SNE visualization of trajectory encoder's output (1000 random paths across 11 different color-coded environments) for models trained with environment-aware objective (left) versus environment-agnostic objective (right).

results are observed in Table 4 that environment-aware learning leads to overfitting on the training dataset (performance on environments seen during training consistently increases for both tasks), while environment-agnostic learning leads to more generalizable policy which performs better on previously unseen environments.

Figure 3 further shows that due to environment-aware objective, the model learns to represent visual inputs from the same environment closer to each other while the representations of different environments are farther from each other resulting in a clustering learning effect.

On the other hand, the environment-agnostic objective leads to more general representation across different environments which results in better performance on unseen environments.

As discussed in Section 3.2, we conducted studies to shape the reward for NDH task.

The results in Table 5 indicate that incentivizing the agent to get closer to the goal room is better than to the exact goal location, because it is aligned with the objective of NDH task, which is to reach the room containing the goal object.

Detailed ablation is presented in Appendix showing that the same holds true consistently as the agent is provided access to different parts of the dialog history.

In this work, we show that the model trained using environment-agnostic multitask learning approach learns a generalized policy for the two natural language grounded navigation tasks.

It closes down the gap between seen and unseen environments, learns more generalized environment representations and effectively transfers knowledge across tasks outperforming baselines on both the tasks simultaneously by a significant margin.

At the same time, the two approaches independently benefit the agent learning and are complementary to each other.

There are possible future extensions to our work-the MT-RCM can further be adapted to other language-grounded navigation datasets, such as those using Street View (e.g., Touchdown (Chen et al., 2019) Table 6 presents a more detailed ablation of Table 5 using different parts of dialog history.

The results prove that agents rewarded for getting closer to the goal room consistently outperform agents rewarded for getting closer to the exact goal location.

Table 7 presents a more detailed analysis from Table 3 with access to different parts of dialog history.

The models with shared language encoder consistently outperform those with separate encoders.

Figure 4: Visualizing performance gap between seen and unseen environments for VLN and NDH tasks.

For VLN, the plotted metric is agent's success rate while for NDH, the metric is agent's progress.

As mentioned in Section 5.2, both multitask learning as well as environment-agnostic learning methods reduce the agent's performance gap between seen and unseen environments which is demonstrated in Figure 4 .

@highlight

We propose to learn a more generalized policy for natural language grounded navigation tasks via environment-agnostic multitask learning.