Deep reinforcement learning (RL) policies are known to be vulnerable to adversarial perturbations to their observations, similar to adversarial examples for classifiers.

However, an attacker is not usually able to directly modify another agent's observations.

This might lead one to wonder: is it possible to attack an RL agent simply by choosing an adversarial policy acting in a multi-agent environment so as to create natural observations that are adversarial?

We demonstrate the existence of adversarial policies in zero-sum games between simulated humanoid robots with proprioceptive observations, against state-of-the-art victims trained via self-play to be robust to opponents.

The adversarial policies reliably win against the victims but generate seemingly random and uncoordinated behavior.

We find that these policies are more successful in high-dimensional environments, and induce substantially different activations in the victim policy network than when the victim plays against a normal opponent.

Videos are available at https://attackingrl.github.io.

The discovery of adversarial examples for image classifiers prompted a new field of research into adversarial attacks and defenses (Szegedy et al., 2014) .

Recent work has shown that deep RL policies are also vulnerable to adversarial perturbations of image observations Kos and Song, 2017) .

However, real-world RL agents inhabit natural environments populated by other agents, including humans, who can only modify observations through their actions.

We explore whether it's possible to attack a victim policy by building an adversarial policy that takes actions in a shared environment, inducing natural observations which have adversarial effects on the victim.

RL has been applied in settings as varied as autonomous driving (Dosovitskiy et al., 2017) , negotiation (Lewis et al., 2017) and automated trading (Noonan, 2017) .

In domains such as these, an attacker cannot usually directly modify the victim policy's input.

For example, in autonomous driving pedestrians and other drivers can take actions in the world that affect the camera image, but only in a physically realistic fashion.

They cannot add noise to arbitrary pixels, or make a building disappear.

Similarly, in financial trading an attacker can send orders to an exchange which will appear in the victim's market data feed, but the attacker cannot modify observations of a third party's orders.

Figure 1: Illustrative snapshots of a victim (in blue) against normal and adversarial opponents (in red).

The victim wins if it crosses the finish line; otherwise, the opponent wins.

Despite never standing up, the adversarial opponent wins 86% of episodes, far above the normal opponent's 47% win rate.

environments are substantially more vulnerable to adversarial policies than in lower-dimensional Ant environments.

To gain insight into why adversarial policies succeed, we analyze the activations of the victim's policy network using a Gaussian Mixture Model and t-SNE (Maaten and Hinton, 2008) .

We find adversarial policies induce significantly different activations than normal opponents, and that the adversarial activations are typically more widely dispersed across time steps than normal activations.

A natural defence is to fine-tune the victim against the adversary.

We find this protects against that particular adversary, but that repeating the attack method finds a new adversary the fine-tuned victim is vulnerable to.

However, the new adversary is qualitatively different, physically interfering with the victim.

This suggests repeated fine-tuning might provide protection against a range of adversaries.

Our paper makes three contributions.

First, we propose a novel, physically realistic threat model for adversarial examples in RL.

Second, we demonstrate the existence of adversarial policies in this threat model, in several simulated robotics games.

Our adversarial policies reliably beat the victim, despite training with less than 3% as many timesteps and generating seemingly random behavior.

Third, we conduct a detailed analysis of why the adversarial policies work.

We show they create natural observations that are adversarial to the victim and push the activations of the victim's policy network off-distribution.

Additionally, we find policies are easier to attack in high-dimensional environments.

As deep RL is increasingly deployed in environments with potential adversaries, we believe it is important that practitioners are aware of this previously unrecognized threat model.

Moreover, even in benign settings, we believe adversarial policies can be a useful tool for uncovering unexpected policy failure modes.

Finally, we are excited by the potential of adversarial training using adversarial policies, which could improve robustness relative to conventional self-play by training against adversaries that exploit weaknesses undiscovered by the distribution of similar opponents present during self-play.

Most study of adversarial examples has focused on small p norm perturbations to images, which Szegedy et al. (2014) discovered cause a variety of models to confidently mispredict the class, even though the changes are visually imperceptible to a human.

Gilmer et al. (2018a) argued that attackers are not limited to small perturbations, and can instead construct new images or search for naturally misclassified images.

Similarly, Uesato et al. (2018) argue that the near-ubiquitous p model is merely a convenient local approximation for the true worst-case risk.

We follow in viewing adversarial examples more broadly, as "inputs to machine learning models that an attacker has intentionally designed to cause the model to make a mistake."

The little prior work studying adversarial examples in RL has assumed an p -norm threat model.

and Kos and Song (2017) showed that deep RL policies are vulnerable to small perturbations in image observations.

Recent work by Lin et al. (2017) generates a sequence of perturbations guiding the victim to a target state.

Our work differs from these previous approaches by using a physically realistic threat model that disallows direct modification of the victim's observations.

showed agents may become tightly coupled to the agents they were trained with.

Like adversarial policies, this results in seemingly strong polices failing against new opponents.

However, the victims we attack win against a range of opponents, and so are not coupled in this way.

Adversarial training is a common defense to adversarial examples, achieving state-of-the-art robustness in image classification (Xie et al., 2019) .

Prior work has also applied adversarial training to improve the robustness of deep RL policies, where the adversary exerts a force vector on the victim or varies dynamics parameters such as friction (Pinto et al., 2017; Mandlekar et al., 2017; Pattanaik et al., 2018) .

Our defence of fine-tuning the victim against the adversary is inspired by this work.

This work follows a rich tradition of worst-case analysis in RL.

In robust MDPs, the transition function is chosen adversarially from an uncertainty set (Bagnell et al., 2001; Tamar et al., 2014) .

Doyle et al. (1996) solve the converse problem: finding the set of transition functions for which a policy is optimal.

Methods also exist to verify controllers or find a counterexample to a specification.

Bastani et al. (2018) verify decision trees distilled from RL policies, while Ghosh et al. (2018) test black-box closedloop simulations.

Ravanbakhsh et al (2016) can even synthesise controllers robust to adversarial disturbances.

Unfortunately, these techniques are only practical in simple environments with lowdimensional adversarial disturbances.

By contrast, while our method lacks formal guarantees, it can test policies in complex multi-agent tasks and naturally scales with improvements in RL algorithms.

We model the victim as playing against an opponent in a two-player Markov game (Shapley, 1953) .

Our threat model assumes the attacker can control the opponent, in which case we call the opponent an adversary.

We denote the adversary and victim by subscript α and ν respectively.

The game M = (S, (A α , A ν ), T, (R α , R ν )) consists of state set S, action sets A α and A ν , and a joint state transition function T : S × A α × A ν → ∆ (S) where ∆ (S) is a probability distribution on S. The reward function R i : S × A α × A ν × S → R for player i ∈ {α, ν} depends on the current state, next state and both player's actions.

Each player wishes to maximize their (discounted) sum of rewards.

The adversary is allowed unlimited black-box access to actions sampled from π v , but is not given any white-box information such as weights or activations.

We further assume the victim agent follows a fixed stochastic policy π v , corresponding to the common case of a pre-trained model deployed with static weights.

Note that in safety critical systems, where attacks like these would be most concerning, it is standard practice to validate a model and then freeze it, so as to ensure that the deployed model does not develop any new issues due to retraining.

Therefore, a fixed victim is a realistic reflection of what we might see with RL-trained policies in real-world settings, such as with autonomous vehicles.

Since the victim policy π ν is held fixed, the two-player Markov game M reduces to a single-player MDP M α = (S, A α , T α , R α ) that the attacker must solve.

The state and action space of the adversary are the same as in M, while the transition and reward function have the victim policy π ν embedded:

where the victim's action is sampled from the stochastic policy a ν ∼ π ν (· | s).

The goal of the attacker is to find an adversarial policy π α maximizing the sum of discounted rewards:

Note the MDP's dynamics T α will be unknown even if the Markov game's dynamics T are known since the victim policy π ν is a black-box.

Consequently, the attacker must solve an RL problem.

We demonstrate the existence of adversarial policies in zero-sum simulated robotics games.

First, we describe how the victim policies were trained and the environments they operate in.

Subsequently, we provide details of our attack method in these environments, and describe several baselines.

Finally, we present a quantitative and qualitative evaluation of the adversarial policies and baseline opponents.

We attack victim policies for the zero-sum simulated robotics games created by Bansal et al. (2018a) , illustrated in Figure 2 .

The victims were trained in pairs via self-play against random old versions of their opponent, for between 680 and 1360 million time steps.

We use the pre-trained policy weights released in the "agent zoo" of Bansal et al. (2018b) .

In symmetric environments, the zoo agents are labeled ZooN where N is a random seed.

In asymmetric environments, they are labeled ZooVN and ZooON representing the Victim and Opponent agents.

All environments are two-player games in the MuJoCo robotics simulator.

Both agents observe the position, velocity and contact forces of joints in their body, and the position of their opponent's joints.

The episodes end when a win condition is triggered, or after a time limit, in which case the agents draw.

We evaluate in all environments from Bansal et al. (2018a) except for Run to Goal, which we omit as the setup is identical to You Shall Not Pass except for the win condition.

We describe the environments below, and specify the number of zoo agents and their type (MLP or LSTM): Sumo Ants (4, LSTM).

The same task as Sumo Humans, but with 'Ant' quadrupedal robot bodies.

We use this task in Section 5.2 to investigate the importance of dimensionality to this attack method.

Following the RL formulation in Section 3, we train an adversarial policy to maximize Equation 1 using Proximal Policy Optimization (PPO) (Schulman et al., 2017) .

We give a sparse reward at the end of the episode, positive when the adversary wins the game and negative when it loses or ties.

Bansal et al. (2018a) trained the victim policies using a similar reward, with an additional dense component at the start of training.

We train for 20 million time steps using Stable Baselines's PPO implementation (Hill et al., 2019) .

The hyperparameters were selected through a combination of manual tuning and a random search of 100 samples; see Section A in the appendix for details.

We compare our methods to three baselines: a policy Rand taking random actions; a lifeless policy Zero that exerts zero control; and all pre-trained policies Zoo * from Bansal et al. (2018a) .

We find the adversarial policies reliably win against most victim policies, and outperform the pre-trained Zoo baseline for a majority of environments and victims.

We report Key: The solid line shows the median win rate for Adv across 5 random seeds, with the shaded region representing the minimum and maximum.

The win rate is smoothed with a rolling average over 100, 000 timesteps.

Baselines are shown as horizontal dashed lines.

Agents Rand and Zero take random and zero actions respectively.

The Zoo baseline is whichever ZooM (Sumo) or ZooOM (other environments) agent achieves the highest win rate.

The victim is ZooN (Sumo) or ZooVN (other environments), where N is given in the title above each figure.

the win rate over time against the median victim in each environment in Figure 3 , with full results in Figure 6 in the supplementary material.

Win rates against all victims are summarized in Figure 4 .

Qualitative Evaluation The adversarial policies beat the victim not by performing the intended task (e.g. blocking a goal), but rather by exploiting weaknesses in the victim's policy.

This effect is best seen by watching the videos at https://attackingrl.github.io/. In Kick and Defend and You Shall Not Pass, the adversarial policy never stands up.

The adversary instead wins by taking actions that induce adversarial observations causing the victim's policy to take poor actions.

A robust victim could easily win, a result we demonstrate in Section 5.1.

This flavor of attacks is impossible in Sumo Humans, since the adversarial policy immediately loses if it falls over.

Faced with this control constraint, the adversarial policy learns a more high-level strategy: it kneels in the center in a stable position.

Surprisingly, this is very effective against victim 1, which in 88% of cases falls over attempting to tackle the adversary.

However, it proves less effective against victims 2 and 3, achieving only a 62% and 45% win rate, below Zoo baselines.

We further explore the importance of the number of dimensions the adversary can safely manipulate in Section 5.2.

Distribution Shift One might wonder if the adversarial policies are winning simply because they are outside the training distribution of the victim.

To test this, we evaluate victims against two simple off-distribution baselines: a random policy Rand (green) and a lifeless policy Zero (red).

These baselines win as often as 30% to 50% in Kick and Defend, but less than 1% of the time in Sumo and You Shall Not Pass.

This is well below the performance of our adversarial policies.

We conclude that most victim policies are robust to typical off-distribution observations.

Although our adversarial policies do produce off-distribution observations, this is insufficient to explain their performance.

In the previous section we demonstrated adversarial policies exist for victims in a range of competitive simulated robotics environments.

In this section, we focus on understanding why these policies exist.

Specifically, we establish that adversarial policies manipulate the victim through their body position; that victims are more vulnerable to adversarial policies in high-dimensional environments; and that activations of the victim's policy network differ substantially when playing an adversarial opponent.

Adv is the best adversary trained against the victim, and Rand is a policy taking random actions.

Zoo * N corresponds to ZooN (Sumo) or ZooON (otherwise).

Zoo * 1T and Zoo * 1V are the train and validation datasets, drawn from Zoo1 (Sumo) or ZooO1 (otherwise).

We have previously shown that adversarial policies are able to reliably win against victims.

In this section, we demonstrate that they win by taking actions to induce natural observations that are adversarial to the victim, and not by physically interfering with the victim.

To test this, we introduce a 'masked' victim (labeled ZooMN or ZooMVN) that is the same as the normal victim ZooN or ZooVN, except the observation of the adversary's position is set to a static value corresponding to a typical initial position.

We use the same adversarial policy against the normal and masked victim.

One would expect it to be beneficial to be able to see your opponent.

Indeed, the masked victims do worse than a normal victim when playing normal opponents.

For example, Figure 4b shows that in You Shall Not Pass the normal opponent ZooO1 wins 78% of the time against the masked victim ZooMV1 but only 47% of the time against the normal victim ZooV1.

However, the relationship is reversed when playing an adversary.

The normal victim ZooV1 loses 86% of the time to adversary Adv1 whereas the masked victim ZooMV1 wins 99% of the time.

This pattern is particularly clear in You Shall Not Pass, but the trend is similar in other environments.

This result is surprising as it implies highly non-transitive relationships may exist between policies even in games that seem to be transitive.

A game is said to be transitive if policies can be ranked such that higher-ranked policies beat lower-ranked policies.

Prima facie, the games in this paper seem transitive: professional human soccer players and sumo wrestlers can reliably beat amateurs.

Despite this, there is a non-transitive relationship between adversarial policies, victims and masked victims.

Consequently, we urge caution when using methods such as self-play that assume transitivity, and would recommend more general methods where practical (Balduzzi et al., 2019; Brown et al., 2019) .

Our findings also suggest a trade-off in the size of the observation space.

In benign environments, allowing more observation of the environment increases performance.

However, this also makes the agent more vulnerable to adversaries.

This is in contrast to an idealized Bayesian agent, where the value of information is always non-negative (Good, 1967) .

In the following section, we investigate further the connection between vulnerability to attack and the size of the observation space.

It is known that classifiers are more vulnerable to adversarial examples on high-dimensional inputs (Gilmer et al., 2018b; Khoury and Hadfield-Menell, 2018; Shafahi et al., 2019) .

We hypothesize a similar result for RL policies: the greater the dimensionality of the component P of the observation space under control of the adversary, the more vulnerable the victim is to attack.

We test this hypothesis in the Sumo environment, varying whether the agents are Ants or Humanoids.

The results in Figures 4c and 4d support the hypothesis.

The adversary has a much lower win-rate in the low-dimensional Sumo Ants (dim P = 15) environment than in the higher dimensional Sumo Humans (dim P = 24) environment, where P is the position of the adversary's joints.

In Section 5.1 we showed that adversarial policies win by creating natural observations that are adversarial to the victim.

In this section, we seek to better understand why these observations are adversarial.

We record activations from each victim's policy network playing a range of opponents, and analyse these using a Gaussian Mixture Model (GMM) and a t-SNE representation.

See Section B in the supplementary material for details of training and hyperparameters.

We fit a GMM on activations Zoo * 1T collected playing against a normal opponent, Zoo1 or ZooV1, holding out Zoo * 1V for validation.

Figure 5a shows that the adversarial policy Adv induces activations with the lowest log-likelihood, with random baseline Rand only slightly more probable.

Normal opponents Zoo * 2 and Zoo * 3 induce activations with almost as high likelihood as the validation set Zoo * 1V, except in Sumo Humans where they are as unlikely as Rand.

We plot a t-SNE visualization of the activations of Kick and Defend victim ZooV2 in Figure 5b .

As expected from the density model results, there is a clear separation between between Adv, Rand and the normal opponent ZooO2.

Intriguingly, Adv induces activations more widely dispersed than the random policy Rand, which in turn are more widely dispersed than ZooO2.

We report on the full set of victim policies in Figures 8 and 9 in the supplementary material.

The ease with which policies can be attacked highlights the need for effective defences.

A natural defence is to fine-tune the victim zoo policy against an adversary, which we term single training.

We also investigate dual training, randomly picking either an adversary or a zoo policy at the start of each episode.

The training procedure is otherwise the same as for adversaries, described in Section 4.2.

We report on the win rates in You Shall Not Pass in Figure 4b .

We find both the single ZooSV1 and dual ZooDV1 fine-tuned victims are robust to adversary Adv1, with the adversary win rate dropping from 87% to around 10%.

However, ZooSV1 catastrophically forgots how to play against the normal opponent ZooO1.

The dual fine-tuned victim ZooDV1 fares better, but still only wins 57% of the time against ZooO1, compared to 48% before fine-tuning.

This suggests ZooV1 may use features that are helpful against a normal opponent but which are easily manipulable (Ilyas et al., 2019) .

Although the fine-tuned victims are robust to the original adversarial policy Adv1, they are still vulnerable to our attack method.

New adversaries AdvS1 and AdvD1 trained against ZooSV1 and ZooDV1 win at equal or greater rates than before, and transfer successfully to the original victim.

However, the new adversaries AdvS1 and AdvD1 are qualitatively different, tripping the victim up by lying prone on the ground, whereas Adv1 causes ZooV1 to fall without ever touching it.

Contributions.

Our paper makes three key contributions.

First, we have proposed a novel threat model of natural adversarial observations produced by an adversarial policy taking actions in a shared environment.

Second, we demonstrate that adversarial policies exist in a range of zero-sum simulated robotics games against state-of-the-art victims trained via self-play to be robust to adversaries.

Third, we verify the adversarial policies win by confusing the victim, not by learning a generally strong policy.

Specifically, we find the adversary induces highly off-distribution activations in the victim, and that victim performance increases when it is blind to the adversary's position.

While it may at first appear unsurprising that a policy trained as an adversary against another RL policy would be able to exploit it, we believe that this observation is highly significant.

The policies we have attacked were explicitly trained via self-play to be robust.

Although it is known that self-play with deep RL may not converge, or converge only to a local rather than global Nash, self-play has been used with great success in a number of works focused on playing adversarial games directly against humans OpenAI, 2018) .

Our work shows that even apparently strong self-play policies can harbor serious but hard to find failure modes, demonstrating these theoretical limitations are practically relevant and highlighting the need for careful testing.

Our attack provides some amount of testing by constructively lower-bounding the exploitability of a victim policy -its performance against its worst-case opponent -by training an adversary.

Since the victim's win rate declines against our adversarial policy, we can confirm that the victim and its self-play opponent were not in a global Nash.

Notably we expect our attack to succeed even for policies in a local Nash, as the adversary is trained starting from a random point that is likely outside the victim's attractive basin.

Defence.

We implemented a simple defence: fine-tuning the victim against the adversary.

We find our attack can be successfully reapplied to beat this defence, suggesting adversarial policies are difficult to eliminate.

However, the defence does appear to protect against attacks that rely on confusing the victim: the new adversarial policy is forced to instead trip the victim up.

We therefore believe that scaling up this defence is a promising direction for future work.

In particular, we envisage a variant of population-based training where new agents are continually added to the pool to promote diversity, and agents train against a fixed opponent for a prolonged period of time to avoid local equilibria.

Table 1 gives the hyperparameters used for training.

The number of environments was chosen for performance reasons after observing diminishing returns from using more than 8 parallel environments.

The batch size, mini-batches, epochs per update, entropy coefficient and learning rate were tuned via a random search with 100 samples on two environments, Kick and Defend and Sumo Humans.

The total time steps was chosen by inspection after observing diminishing returns to additional training.

All other hyperparameters are the defaults in the PPO2 implementation in Stable Baselines (Hill et al., 2019) .

We repeated the hyperparameter sweep for fine-tuning victim policies for the defence experiments, but obtained similar results.

For simplicity, we therefore chose to use the same hyperparameters throughout.

We used a mixture of in-house and cloud infrastructure to perform these experiments.

It takes around 8 hours to train an adversary for a single victim using 4 cores of an Intel Xeon Platinum 8000 (Skylake) processor.

We collect activations from all feed forward layers of the victim's policy network.

This gives two 64-length vectors, which we concatenate into a single 128-dimension vector for analysis with a Gaussian Mixture Model and a t-SNE representation.

We fit models with perplexity 5, 10, 20, 50, 75, 100, 250 and 1000.

We chose 250 since qualitatively it produced the clearest visualization of data with a moderate number of distinct clusters.

We fit models with 5, 10, 20, 40 and 80 components with a full (unrestricted) and diagonal covariance matrix.

We used the Bayesian Information Criterion (BIC) and average log-likelihood on a heldout validation set as criteria for selecting hyperparameters.

We found 20 components with a full covariance matrix achieved the lowest BIC and highest validation log-likelihood in the majority of environment-victim pairs, and was the runner-up in the remainder.

Supplementary figures are provided on the subsequent pages.

Figure 9: t-SNE activations of victim Zoo1 (Sumo) or ZooV1 (other environments).

The results are the same as in Figure 8 but decomposed into individual opponents for clarity.

Model fitted with a perplexity of 250 to activations from 5000 timesteps against each opponent.

Opponent Adv is the best adversary trained against the victim.

Opponent Zoo is Zoo1 (Sumo) or ZooO1 (other environments).

See Figure 8 for results for other victims (one plot per victim).

<|TLDR|>

@highlight

Deep RL policies can be attacked by other agents taking actions so as to create natural observations that are adversarial.