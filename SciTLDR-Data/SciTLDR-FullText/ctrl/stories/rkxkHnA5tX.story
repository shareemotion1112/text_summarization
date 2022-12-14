A noisy and diverse demonstration set may hinder the performances of an agent aiming to acquire certain skills via imitation learning.

However, state-of-the-art imitation learning algorithms often assume the optimality of the given demonstration set.

In this paper, we address such optimal assumption by learning only from the most suitable demonstrations in a given set.

Suitability of a demonstration is estimated by whether imitating it produce desirable outcomes for achieving the goals of the tasks.

For more efficient demonstration suitability assessments, the learning agent should be capable of imitating a demonstration as quick as possible, which shares similar spirit with fast adaptation in the meta-learning regime.

Our framework, thus built on top of Model-Agnostic Meta-Learning, evaluates how desirable the imitated outcomes are, after adaptation to each demonstration in the set.

The resulting assessments hence enable us to select suitable demonstration subsets for acquiring better imitated skills.

The videos related to our experiments are available at: https://sites.google.com/view/deepdj

Imagine that you intend to learn how to make a free throw in basketball, which requires you to throw the ball into the basket from a fixed position.

Without the proper knowledge, one may observe professional players perform a free throw on YouTube by obtaining numerous exemplary videos.

However, learning from every demonstration videos might lead to worse performance, as they may contain unsuitable or even irrelevant content.

The challenge of learning from noisy demonstration sets is as well crucial in the robot imitation learning regime, as demonstrations which are not aligned with achieving the intended goal deteriorate the learning process.

The assumptions of optimality (or at least sub-optimality) of the demonstrations are often made in state-of-the-art imitation learning algorithms (Ross & Bagnell, 2010; Ross et al., 2011; BID10 Sermanet et al., 2018) .

As a result, they are vulnerable to demonstrations that are potentially detrimental to the learning outcomes in the given set.

To address assumptions of optimality, in this paper, we aim for a generic framework capable of learning from a noisy demonstration set, via evaluating the suitability of imitated skills judged by task specific heuristics.

Prior works have handled deteriorated imitated outcomes due to noisy demonstration sets by utilizing expected Q-values provided in the demonstrations to avoid learning from bad demonstrated actions BID14 BID17 BID5 .

However, these works require demonstrations to have a rich representation such as incorporating the aforementioned Q-values, which may neither be available in other cases nor directly applicable to the agent training environment.

Our framework, on the other hand, does not require demonstrations to contain specific information, and hence is able to cope with any forms of expert demonstrations.

To be specific, we propose to first assess which demonstrations in the given set might be more suitable by learning from them, and then train the agent imitating only a selected subset.

In order to achieve selectively learning from suitable demonstrations, we examine if the learning outcomes are favorable after imitating each demonstration in a set.

Typically in robot learning regime, we are able to receive designed feedbacks in the target training environment to evaluate how well the agent performs.

Thus, in each task, we predefine specific heuristics for assessing if the agent exhibits imitation learning outcomes desirable for reaching the goals of the tasks.

The key challenges for the feasibility of assessing learned outcomes from each demonstration are the efficiency and generalization ability.

A framework should be capable of producing these assessable outcomes as quick as possible, and generalizing to unseen demonstrations.

To this end, we propose a framework with the demonstration suitability assessor leveraging meta-learning, where we train adaptive parameters via meta-imitation-learning.

The meta-imitation-learned parameters can thus: (1) produce assessable imitated outcomes at testing time quicker than both imitation learning from scratch and fine-tune a pretrained initialization BID3 , and (2) adapt to imitating newly sampled unseen demonstrations.

Overall, the imitated outcomes after adaptation will be judged by task heuristics to indicate the suitability of certain demonstrations.

We then train agents using imitation learning from the selected suitable demonstration subsets for obtaining better policies.

We demonstrate two empirical approaches to utilize the resulting suitability assessments.

One composes new subsets of demonstrations from the top ranked, the other iteratively fine-tunes the meta-learned parameters by strengthening or weakening certain demonstrations in a set according to current suitability judgments, producing a selected suitable subset at convergence.

In some cases, the distribution of demonstration sets can be imbalanced or multi-modal.

To prevent over-fitting to certain subsets of such demonstration distributions, we augment the meta-imitationtraining with a regularization objective-maximization of mutual information between the demonstration and the induced behavioral differences from imitating it.

This additional regularization term aims to make the meta-trained parameters more responsive to the demonstrations being imitated, and as a result, help differentiate better the adapted behaviors from a noisy and diverse set.

We test our framework on four different simulation sports environments in MuJuCo.

Our results, both qualitative and quantitative, show that the proposed method outperforms various baselines, including vanilla MAML and fine-tuning a pretrained initialization, on learning better policies from noisy demonstration sets.

Imitation Learning and Learning Diverse Behaviors In imitation learning, an agent tries to acquire skills by learning from demonstrated expert behaviors (Schaal, 1997; BID0 Ziebart et al., 2008; BID15 BID12 Ross et al., 2011) .

BID10 discover that imitation learning can be formulated as an adversarial training BID6 between demonstrations and imitated behaviors of the agent (GAIL), which is adopted in this work as the main imitation learning framework.

Extension works of GAIL (Wang et al., 2017; BID8 BID16 augment the original algorithm with a latent code or embedding for imitating a diverse set of demonstrations, increasing diversity of the imitated outcomes.

BID2 learn an encoder-decoder framework for digesting multiple demonstrations and achieving one-shot imitation learning during testing time.

BID9 train a deep q-network by initializing the replay buffer with expert demonstrations.

The demonstration assessing process of our work can also be viewed as leveraging the results of few-shot imitation learning from diverse demonstrations.

However, the emphasis of our goal is centered on utilizing the assessments to acquire better skills learning from selected suitable demonstrations.

Learning from Imperfect Demonstrations Despite remarkable performances in imitative learning of complex control behaviors from prior works, most of the algorithms assume the optimality (or at least sub-optimality) of the demonstrations.

The given demonstration set is usually presumed to contain only helpful demonstrations: it is not noisy.

BID13 presents several types of experts violating the optimality assumption and basic ways to cope with them.

Tomasello (2016) provides psychological evidences that infants imitate rationally, and that they are more prone to certain demonstrations that do not only fundamentally make sense but effectively help the development of their skills.

Grollman & Billard (2011) utilize failure demonstrations as negative constraint on the exploration to prevent learning agent from approaching behaviors likewise.

BID14 casts learning from limited demonstrations as a relaxed large-margin constrained optimization problem.

Several recent works BID5 BID17 ) also assume the expert representation contains expected Q-values and utilize them to filter bad demonstrated actions.

However, these additional information may not be available in other cases such as learning from raw video demonstrations (Sermanet et al., 2016; .

Moreover, specifically for Q-values utilized in aforementioned works, the assumption of these information being applicable to the agent training

Figure 1: Illustration of the framework: We first meta-imitation train a set of adaptive parameters on the entire demonstration set.

And then, during meta-testing phase, the imitation learning outcomes exhibited after adapting to a particular demonstration are assessed with predefined task heuristics.

The assessment result leads to a selected suitable subset for the agent to learn from.environment does not necessarily hold.

We thus aim for a more general framework capable of coping with any forms of demonstrations, while this work mainly based off state-action pairs.

Our work is also closely related to a series of recent advancements in meta learning, that we as well seek fast adaptive parameters which should learn how to learn from certain demonstrations.

BID3 propose a gradient-based model-agnostic framework (MAML) for training a policy prior that can quickly adapt to newly sampled tasks during testing (termed as meta-testing) time by small number of gradient update steps.

In another work BID4 MAML is adopted to train a set of adaptive parameters achieving gradient-based one-shot imitation learning during meta-testing time.

As an extension of MAML with regularization in an inner imitative learning objective, we are, to our best knowledge, the first to utilize the meta learning framework to evaluate a demonstration set for learning a better policy from selected demonstrations.

A finite-horizon Markov Decision Process (MDP) can be defined as M = (S, A, P, R, ??, p 0 , T ), where S and A denotes state and action spaces, P : S ?? A ?? S ??? R + denotes the state transition probability, R : S ?? A ??? R is a reward function, p 0 an initial state distribution, ?? is the reward discount factor, and T is the horizon of the MDP.

Let ?? = (s 0 , a 0 , ..., s T , a T ) be a trajectory of state and action pairs, the associated reward can be written as R(?? ) = T t=0 ?? t R(s t , a t ).

Reinforcement learning seeks a policy ?? ?? (a|s) parameterized by ?? that maximizes the expected total return: DISPLAYFORM0 In imitation learning regime, demonstrated expert behaviors are given while the reward function is missing, and the agent acquire desired skills learning from these demonstrations.

We now formulate our problem setup: given a noisy demonstration set V = {v 1 , ..., v n }, where each v i ??? V can be represented as state-action pairs: {(s i,t , a i,t )}, and several environment variants E = {e 1 , ..., e k }, where each e i denotes different goal in a multi-goal learning task.

We require a suitability assessment score S vi | ej ??? R of a demonstration v i under e j for judging and selecting suitable demonstrations.

And hence, the assessment scores {S vi | ej } can be the reference for selecting the most suitable demonstration as arg max i S vi | ej , v i ??? V , or a subset {v opt } for imitation learning.

Our goal is to learn a good policy from a noisy demonstration set by imitating the selected suitable demonstrations.

In order to produce imitated outcomes efficiently for suitability evaluation, we utilize the adaptation process in meta-learning framework, Model-Agnostic Meta-Learning (MAML) BID3 , to meta-imitation-learn a set of fast adaptive parameters from the given demonstration set.

At meta-testing time, the parameters should adapt to each sampled demonstration to produce assessable learning outcomes, leading to selected suitable subsets for the agent to learn from.

To alleviate insensitivity imitating a particular demonstration to an unbalanced set distribution, motivated by BID1 , we regularize the inner imitation objective of MAML with maximization of mutual information between the demonstration and the induced behavioral differences from imitating it.

Fig.1 depicts the overview of our framework, and the pseudo codes are provided in the appendix.

Following the conventions as described in MAML BID3 b) , meta-imitation-learning has two objectives, where the inner objective tries to optimize for adapting to a sampled individual demonstration, while the outer objective (meta objective) tries to optimize for fast adaptation across multiple newly sampled demonstrations.

Denote f ?? as the policy model mapping observations to actions parameterized by ??, the desired adaptive meta-parameters.

The inner objective in the MAMLbased meta-imitation-learning is an imitation objective, where ?? is updated to ?? i after learning from demonstration v i by gradient descent with learning rate ??, which can be written as: DISPLAYFORM0 Here L vi denotes the imitation loss imitating the sampled demonstration v i .

And hence the meta objective and the corresponding meta-imitation-update can be derived as (with meta learning rate ??): DISPLAYFORM1

The inner imitative update in Eq.1 is not limited to certain methods, we hence adopt two major imitation learning algorithms: generative adversarial imitation learning and behavioral cloning.

Generative Adversarial Imitation Learning (GAIL): Our primary adopted imitation learning method BID10 .

For each demonstration v i ??? V , we maintain a discriminator D i dedicated to treating demonstration v i as real data in the GAIL setting contributing to L vi in Eq. 1.

Specifically, each inner update from ?? to ?? i is computed with reward ??? log(D i (s, a)) via TRPO (Schulman et al., 2015) update rule.

In this paper, we denote this reward as R IL and the sparse task reward as R task .

In a standard imitation learning scheme, both R IL , R task should be taken into account.

However, R task would bias the training of the adaptive parameters ??, that ?? should be optimized for quick adaptation to any given demonstration rather than succeeding the task.

Therefore, we drop the R task term and denote our reward definition as R(s t , a t ) = R IL , 0 ??? t ??? T .In order to adapt to unseen demonstrations, we train a meta-discriminator D meta along with training the adaptive policy parameters ??.

Similar to Eq.2, D meta is meta-updated as: DISPLAYFORM0 where ?? i denotes the trajectories sampled from f ?? i (after learning from demonstration v i ), v i is the trajectory of demonstration as one episode of state-action pairs (expert trajectory in GAIL setting).

Each D meta,i is the updated discriminator parameters during the inner update for demonstration v i , which follows the GAIL update rule for discriminator.

Behavioral Cloning:

In behavioral cloning, the inner update in Eq.1 is simply a supervised training, where L vi = 1 T t f ?? (s i,t ) ????? i,t 2 using a mean squared error with?? i,t denoting the expert action from demonstration v i at time-step t.

In order to overcome the potential challenge of learning from an unbalanced demonstration distribution, we augment the inner update in meta-training phase with a regularization loss: maximization of the mutual information I(v i , ????? vi ) between the induced behavioral differences and the demonstration being imitated.

????? vi here denotes the differences of behaviors induced after an inner update learns from a particular demonstration v i , which we estimate using a sample-based method.

Maximizing I(v i , ????? vi ) can be approximated by optimizing the following variational lower bound: DISPLAYFORM0 where Q denotes the posterior network jointly trained with the adaptive policy parameters.

Since v i is drawn from a fixed given demonstration distribution (a set), H(v i ) can be treated as a constant, we thus only need to consider optimizing Q. Q can be trained by estimating ????? vi through sampling trajectories before and after the inner imitation update, denoted as ?? old and ?? new .

We implement Q using a siamese LSTM network taking as inputs both the old and new trajectories.

The output of the posterior network is an encoded feature of the demonstration v i .

We sample two sets of trajectories for both ?? old and ?? new as one used for training Q, and the other (test set) for estimating E[logQ(v i |????? vi )] to be used as the regularization reward.

We then augment the original reward function in GAIL with R M I = ???E[logQ(v i |????? vi )] as follows: DISPLAYFORM1 However, this regularization term should be applied to the parameters before the imitation update, since it is interpreted as the probability of such behavioral differences induced by a particular demonstration v i .

Denote the parameters for inner imitation update from demonstration v i as ?? i , our inner objective thus becomes: DISPLAYFORM2 and hence the update where we revert ?? i back to ?? i with the additional regularization term can be written as: DISPLAYFORM3 Which implies we can compute the gradients with respect to the old trajectories but update on the new parameters ?? i .

In practice, we implement an additional value network dedicated for this additional regularization reward.

As for behavioral cloning, we update the policy parameters ?? utilizing policy gradients computed from this R M I , while other parts remained as standard supervised training.

During meta-testing, we evaluate each demonstration v i ??? V and select the most suitable one(s) to imitate judged by some task heuristics K under environment variant e j .

Each adaptation at metatesting time will be run for a few iterations.

K is essentially a scoring function which takes as input trajectories generated from a policy ?? and outputs a real-valued number, ie.

score ?? = K(?? ?? ) ??? R. Denote ?? ?? as the meta-trained policy, and ?? ??i is the policy adapted to demonstration v i , the demonstration suitability assessment score is then computed as: DISPLAYFORM0 The most suitable demonstration v iopt|e j is then selected by: DISPLAYFORM1

To select a subset of suitable demonstrations to learn from, we design two empirical approaches that utilize assessment scores obtained in the meta-testing phase.

One approach, which is further described in Sec. 5.2 as one of our evaluation metrics, takes the top-K ranked demonstrations as the selected subset.

The other approach is an iterative algorithm that searches for a suitable subset using a slightly adjusted version of our meta-imitation-learning framework.

The intuition for this approach is described as follows: we introduce a weight c i , initialized as 1.0, for each v i in the demonstration set.

During meta-update, the meta-prior ?? is updated by a weighted sum of gradients computed when adapting to each demonstration DISPLAYFORM0 After each meta-testing, c i is adjusted based on the difference between current assessment score S vi | ej and the baseline score obtained from the meta-parameters S ?? | ej .

At convergence, we select a subset {v i |c i > }, where is a predefined threshold.

We include the pseudo code for obtaining such subset in the appendix B.

The goals of our experiments are to: (1) Compare different methods on assessing and ranking a given demonstration set according to the suitability of each demonstration.

(2) Examine if our meta-learning-based frameworks can generalize such suitability assessment to unseen demonstrations.

(3) Provide empirical approaches to learn a better policy from noisy demonstration set.

For brevity, we will term the demonstration suitability assessments as DSA.

We evaluate the effectiveness of the following methods on learning from a noisy demonstration set in four experimental environments:??? Avg Fine-Tune: A framework that fine-tunes towards a particular demonstration with a set of parameters pre-trained using the entire demonstration set.??? MAML: The standard MAML framework as described in BID3 ).???

MAML + MI: A MAML framework regularized with mutual information maximization of the demonstration and the induced behavioral differences, our core method as described in Sec. We built four different simulation sports environments as illustrated in FIG1 .

These environments aim to resemble real sports activities and their task heuristics are intuitively designed to fit human preferences.

We sample generated state-action pairs {(s t , a t )} from RL pre-trained agents during different stages of their RL training to compose the demonstration set.

Details in the appendix A.??? Free Throw: As shown in FIG1 , the thrower needs to make a basketball free-throw from a fixed position, where it scores if the ball is successfully thrown into the basket.

Two datasets are tested containing 8 and 20 demonstrations respectively.

The latter one with 20 demonstrations includes demonstrations from a taller thrower agent for verifying our concept that successful demonstrations are not necessarily the suitable ones, it has to depend on the learning agent.

??? Penalty Save: As shown in FIG1 , an ant agent is required to jump at the right timing with the proper force to block an automated incoming penalty shot.

This is a multi-goal task since the incoming shots can have different directions and speed.

There are 12 demonstrations in this dataset.??? Handstand: As shown in FIG1 , a humanoid is required to withstand an upside-down handstand position.

The appropriate body-pose is defined as: the hands must touch the ground while feet are raised over a threshold height.

There are 15 demonstrations in this dataset.

??? Martial Arts: As shown in FIG1 , a humanoid is required to make a jump kick to kick a target placed higher than the height of the humanoid.

There are 10 demonstrations in this dataset.

Here we explain the metrics used for evaluating and comparing our method and the baselines:??? Top-K combinations: After obtaining an empirical rank of the given set sorted by the adapted heuristics scores, we can combine the top-K ranked demonstrations to compose some potentially better datasets.

Eg.

top-3 means such dataset consists of 3 demonstrations ranked as the top 3 in the set according to the heuristics scores generated during adaptation using one of the methods.

The empirical rank generated by our method and the baselines are thus compared by the performances of agents learning from different top-K demonstration sets using imitation learning from scratch.

1 T i s i,t ????? i,t + a i,t ????? i,t , where (?? i ,?? i ) denotes the trajectories from demonstration v i , and (s i , a i ) denotes the generated trajectories from parameters adapted to v i .

As shown in FIG2 , each row consists of learning reward curves generated from imitating several top-K demonstration subsets composed by each method.

For empirical studies, suppose there are N demonstrations in the given noisy set, we select subsets from top-1, top-2, top-N/2, and top-N , which is simply learning from the entire original noisy set.??? Free Throw (FT): as shown in FIG2 top row, MAML + MI outperforms other baselines by resulting in better composition of top-K demonstration sets and successfully selects the most suitable demonstration from the set.

The learning curves for the 20 demonstration set consisting behaviors from a taller expert agent can be found in the appendix C.2.??? Penalty Save (PS): as shown in FIG2 second row, MAML + MI also outperforms other baselines.

Notice that since this environment is a multi-goal learning task, the DSA should depend on the environment variants as Eq.9.

The curves we show here are trained under a fixed environment variant, which are better illustrated in the videos.??? Handstand (HS): as shown in FIG2 third row, while learning from the top-K subsets selected by MAML + MI achieves the best results, MAML also selects a considerably suitable top-1 demonstration.

However, top-2 of MAML performs even worse than the baseline, implying that the second highest ranked demonstration negatively affect the agent outweighing the positiveness of the top-1 ranked demonstration.

All of the top-K combinations composed by Avg Fine-Tune perform worse than the baseline supposedly due to wrongly selecting detrimental demonstrations.??? Martial Arts (MA): as shown in FIG2 fourth row, it can be noticed that MAML + MI successfully achieves the task goal using the top-1 demonstration within 2,000 iterations, but neither MAML nor Avg Fine-Tune succeeded even being trained further.

For this environment, we further conducted experiments on generalization to unseen demonstrations and our approximate optimal subset selection algorithm, denoted as OSS.

The golden thick curves are the learning reward curves imitating the selected subset by the OSS for each method.

It can be seen that the performances of learning from our selected subset is significantly better in MAML + MI case, while other baselines fail to utilize this algorithm to select better suitable demonstration subset.

Learning curves generated adapting to unseen demonstrations can be found in appendix C.3.

The videos showing how the meta-trained parameters can adapt to certain demonstrations and their comparisons with different methods are shown on the project page.

Behavioral Cloning: The results on handstand and martial arts can be found in appendix C.4.ABD: Table 1 shows the computed behavioral distances between the adapted behaviors and the expert behaviors, across the entire demonstration set for a mean score.

It can be shown that MAML + MI outperforms other baselines by adapting better to the sampled demonstrations.

MAML works the best in the free throw environment, potentially due to posterior network Q was not well-trained using the small size of dataset.

In order to verify the efficiency as compared to demonstration judging via training from scratch, We set a limited quota be the total number of iterations required for meta-training the adaptive parameters ?? plus the number of gradient updates during adaptation to each demonstration, which sums up to around 6,000 and 7,500 for our martial arts and handstand environments respectively.

We then evenly distribute this quota to train different agents imitating each of the demonstration from the noisy set, and output their top choices.

FIG3 , the heuristics curves generated up to 500 iterations produce the required assessment scores, we verified that the top choices from both environments are of non-suitable demonstrations, the associated reward curves are presented in the appendix C.6.

We propose a framework to tackle the challenging problem -learning a good policy through imitation learning from a noisy demonstration set.

Our framework, built on top of MAML with a mutual information maximized regularization, learns a set of adaptive parameters from the given noisy set.

The agent should exhibit significant learning outcomes after fast adaptation to certain demonstrations where these outcomes can be evaluated via predefined task heuristics.

By being a learning framework, the system learns to discover the most suitable demonstrations for the agent from the expert rather than selecting based on hand crafted judging rules.

For future research direction, we hope this work can serve as the first trial to lure more advanced research on tackling imitation learning from noisy demonstration sets.

For evaluating the suitability of certain demonstrations, we require a task dependent knowledge to judge from the imitation learning outcomes after the adaptation to a particular demonstration.

We hereby describe the task heuristics for each environment in the following:??? Free Throw: The minimum distance between the basketball and the fixed basket.??? Penalty Save: The minimum distance between the agent and the incoming shot in an episode.??? Handstand: The height of two feet of a humanoid if both two hands are touching the ground without letting the head to hit the ground.

We accumulate this heuristics score when the aforementioned body pose condition is satisfied.??? Martial Arts: The minimum distance between right foot and the target if the head of the humanoid is higher than a pelvis.

Setups of our simulation environments are listed in

We build the policy networks for each of the simulation environments using a 2-layer MLP.

The posterior network Q used in MAML + MI is implemented as a siamese 1-layer Long-Short Term Memory (LSTM) BID11 ) network followed by a two consecutive fullyconnected networks to produce the 1-dimensional embedding as the prediction of the encoded demonstration feature.

In this paper, we simply use l2-norm of observations throughout the entire episode of a demonstration as its encoded feature.

For each environment, we used a meta batch size of 10 (except for one of the Free Throw dataset consisting of 8 demonstrations, we used 8) training for up to 2,000 meta updates.

We followed the criteria described in the original MAML work BID3 for choosing the MAML models with best average return as our adaptive parameters.

The learning rate in the inner objective Eq.1 is set to be ?? = 1.0 with KL-divergence constraints introduced by TRPO (Schulman et al., 2015) set to 0.15.

?? is halved after the first iteration during adaptation for free throw and penalty save environments.

For handstand and martial arts environments, we enlarge ?? by twice for the first iteration to 2.0, and then use 0.5 like the two aforementioned environments.

All our resulting curves are plotted with mean and 1 standard deviation of several trials.

Table 3 summarizes the network architecture for the policy networks and the posterior networks Q for each environment.

Algorithm1 summarizes the proposed method for a training the fast adaptive set of parameters.

During meta-testing, we adapt the meta-learned parameters to each demonstration in the set one at a time.

Algorithm2 summarizes the proposed algorithm for meta-testing as the suitability assessing phase of our framework.

We will use the same terminologies as in Sec. 3 and Sec. 4 Table 3 : Network architectures used in this paper, the MLP(h1,h2) are two-layer multi-layer perceptrons with hidden sizes h1 and h2.

Siamese-LSTM(h1, L) indicates how L layers of LSTM cells with hidden szie h1.

MLP(h1,h2,...,hn) for discriminators are multi-layer perceptrons with hidden sizes as indicated.example, we will denote the horizon of a trajectory as T to differentiate it from the entropy term H. Algorithm3 describes the proposed subset selection algorithm.

Algorithm 1 Info Meta Imitation Learning (Blued lines are GAIL ver.)Require: V = {v 1 , v 2 , ..., v n }: A set of demonstrations Require: E = {e 1 , e 2 , ..., e k }: An environment with k goals (k = 1 for 3 of our environments) Require: ??, ??: step size hyperparameters 1: randomly initialize ??, Q, D meta 2: Sample a batch of Demonstrations v i ??? V 3: while not done do

for all v i do

Sample a batch of environments e j ??? E 6:for all e j do 7:Sample K trajectories ?? = {(s 1 , a 1 , ..., s T )} using f ?? for updating D meta to D meta,i 8:Evaluate ???L ILi (f ?? ) using ?? and v i , and update DISPLAYFORM0 Update D meta to D meta,i with the gradient: DISPLAYFORM1 Sample K trajectories ?? ij = {(s 1 , a 1 , ..., s T )} using f ?? i after imitating v i under e j

Update posterior Q with ?? , ?? ij , and encoded v i

Sample new trajectories using f ?? and f ?? i as ?? new and ?? ij,new respectively 13:Compute R M Ii with Q, using newly sampled ?? new and ?? ij,new as FORMULA5 14: DISPLAYFORM0 ) with TRPO as FORMULA7 15:Sample K trajectories ?? ij = {(s 1 , a 1 , ..., s T )} using f ?? i Meta Update the ?? using all the ?? ij with TRPO update rules as FORMULA2 19:Meta Update the D meta using all the gradients collected from D meta,i as (3) 20: end while C MORE RESULTS

In Fig.5 , we show the adaptation curves from each of the method in every environments we tested on.

As shown in Fig.6 , both top-1 and top-2 obtained by MAML + MI are suitable demonstration subsets, with demonstration 1 and 5 as top-2 as depicted in Fig.5 top row.

Demonstration 2 in this set is actually a demonstration recorded from a much taller agent, which we show in this top-K curves it is not selected as a suitable one despite being a successful demonstration achieving the free throw task.

This phenomenon is better illustrated in the videos.

Require: V = {v 1 , v 2 , ..., v n }: A set of demonstrations Require: E = {e 1 , e 2 , ..., e k }: An environment with k goals (k = 1 for 3 of our environments) Require: ??, ??: step size hyperparameters Require: max_iter: maximum iterations for the meta-testing Require: ??: the meta-trained parameters Require: Pre-defined Task Heuristics K Require: Meta-Trained Discriminator D meta 1: for all e j ??? E do 2:for all v i ??? V do 3:Initialize ?? ij as meta parameters ?? 4: DISPLAYFORM0 for iter = 0 to max_iter do 6:Sample K trajectories ?? = {(s 1 , a 1 , ..., s T )} using f ??ij

Update D i with the gradient: DISPLAYFORM0 Evaluate ???L IL (f ??ij ) using ?? and v i , and update ?? ij ??? ?? ij ??? ?????L IL (f ??ij ) with TRPO 10:end for

Evaluate ?? ij for M random seeds 12:Compute DSA score DISPLAYFORM0 end for 14:Record the index i opt for the most suitable demonstration under e j with i opt = arg max i S vi | ej , ???v i ??? V 15: end for

Require: V = {v 1 , v 2 , ..., v n }: a set of demonstrations Require: N : predefined iterations Require: ??: constant for multiplying and dividing weights of demonstrations Require: : threshold for selecting final subset of demonstrations Require: W = {w 1 , w 2 , ..., w n }: weights for meta-learning each demonstration.

1: Initialize all w i to 1 2: while not done do DISPLAYFORM0 Update ?? with Imitation Learning using current set V for N iterations, applying weights W for the demonstrations during meta-update.

Get heuristic score for meta-trained parameters ??, store in S 5: DISPLAYFORM0 if ??? K,ij > 0 then 8: DISPLAYFORM1 else 10: In FIG6 we show that our framework can also work on selecting suitable demonstrations given an unseen demonstration set.

We meta-imitation-learned on a large set of demonstrations consisting of 100 martial arts demonstrations and tested it on a disjoint random sampled demonstration set.

From the reward curves, although requiring more number of iterations to train, MAML + MI selects a subset that eventually succeeds achieving the goal of the task, while other two baselines struggle finding a suitable top-K subset.

In FIG7 , we show learning reward curves using behavioral cloning as the inner imitative update in Eq.1 on two of our harder environments, handstand and martial arts.

For handstand environment, the top-2 and top-7 produced by MAML + MI are significantly consistently better than those from other two baselines, implying the top 7 ranked demonstration selected from MAMl + MI are of the more suitable subset.

Interestingly, MAML produce reasonably good top-2 as well, which is almost on par with the MAML + MI.

We hypothesize that behavioral cloning does not introduce too much randomness requiring further regularizing vanilla MAML.

However, behavioral cloning can not produce perfect handstand results due to the complexity of such task, while our primarily adopted imitation learning algorithm GAIL successfully produced perfect imitative outcomes.

For martial arts, both MAML + MI and MAML produce reasonable results, while MAML + MI produce better top-1 achieving better overall performances.

The Avg Fine-Tune method surprisingly works in this environment with behavioral cloning, hypothetically due to the steadiness of supervised training.

DISPLAYFORM2

The trends of each subset weight c i and how they change throughout the subset selection training is as shown in Fig.9 .

We again sub-sample only the best 3 evaluated weights and the worst 3.

It is obvious that MAML + MI diverges in a more consistent fashion and produce more reasonable final results.

In Fig.10 , we show the learning reward curves imitating the top-K demonstration combinations generated by training from scratch and snapshot at 500 iterations as described in Sec.5.4.

It is obvious that these learning outcomes are significantly worse than those presented in by MAML + MI and MAML.

This verifies that in most cases, training from scratch is not only inefficient but also prone to more randomness if reinforcement learning is involved (in our case, GAIL is involved).

<|TLDR|>

@highlight

We propose a framework to learn a good policy through imitation learning from a noisy demonstration set via meta-training a demonstration suitability assessor.

@highlight

Contributes a MAML based algorithm to imitation learning which automatically determines if provided demonstrations are "suitable".

@highlight

A method for doing imitation learning from a set of demonstrations that includes useless behavior, which selects the useful demonstrations by their provided performance gains at the meta-training time.