We propose Cooperative Training (CoT) for training generative models that measure a tractable density for discrete data.

CoT coordinately trains a generator G and an auxiliary predictive mediator M. The training target of M is to estimate a mixture density of the learned distribution G and the target distribution P, and that of G is to minimize the Jensen-Shannon divergence estimated through M. CoT achieves independent success without the necessity of pre-training via Maximum Likelihood Estimation or involving high-variance algorithms like REINFORCE.

This low-variance algorithm is theoretically proved to be superior for both sample generation and likelihood prediction.

We also theoretically and empirically show the superiority of CoT over most previous algorithms in terms of generative quality and diversity, predictive generalization ability and computational cost.

Generative modeling is essential in many scenarios, including continuous data modeling (e.g. image generation BID6 , stylization BID17 , semisupervised classification BID13 ) and sequential discrete data modeling (e.g. neural text generation BID2 ).For discrete data with tractable density like natural language, generative models are predominantly optimized through Maximum Likelihood Estimation (MLE), inevitably introducing exposure bias BID14 , which results in that given a finite set of observations, the optimal parameters of the model trained via MLE do not correspond to the ones maximizing the generative quality.

Specifically, the model is trained on the data distribution of inputs and tested on a different distribution of inputs, namely, the learned distribution.

This discrepancy implies that in the training stage, the model is never exposed to its own errors and thus in the test stage, the errors made along the way will quickly accumulate.

On the other hand, for general generative modeling tasks, an effective framework, named Generative Adversarial Network (GAN) BID6 , was proposed to train an implicit density model for continuous data.

GAN introduces a discriminator D φ parametrized by φ to distinguish the generated samples from the real ones.

As is proved in BID6 , GAN essentially optimizes an approximately estimated Jensen-Shannon divergence (JSD) between the currently learned distribution and the target distribution.

GAN shows promising results in many unsupervised and semi-supervised learning tasks.

The success of GAN results in the naissance of a new paradigm of deep generative models, i.e. adversarial networks.

However, since the gradient computation requires backpropagation through the generator's output, GAN can only model the distribution of continuous variables, making it non-applicable for generating discrete sequences like natural language.

Researchers then proposed Sequence Generative Adversarial Network (SeqGAN) , which uses model-free policy gradient algorithm to optimize the original GAN objective.

With SeqGAN, the expected JSD between current and target discrete data distribution is minimized if the training is perfect.

SeqGAN shows observable improvements in many tasks.

Since then, many variants of SeqGAN have been proposed to improve its performance.

Nonetheless, SeqGAN is not an ideal algorithm for this problem, and current algorithms based on it cannot show stable, reliable and observable improvements that covers all scenarios, according to a previous survey .

The detailed reason will be discussed in detail in Section 2.In this paper, we propose Cooperative Training (CoT), a novel, low-variance, bias-free algorithm for training likelihood-based generative models on discrete data by directly optimizing a wellestimated Jensen-Shannon divergence.

CoT coordinately trains a generative module G, and an auxiliary predictive module M , called mediator, for guiding G in a cooperative fashion.

For theoretical soundness, we derive the proposed algorithm directly from the definition of JSD.

We further empirically and theoretically demonstrate the superiority of our algorithm over many strong baselines in terms of generative performance, generalization ability and computational performance in both synthetic and real-world scenarios.

Notations.

P denotes the target data distribution.

θ denotes the parameters of the generative module G. φ denotes the parameters of the auxiliary predictive mediator module M .

Any symbol with subscript g and m stands for that of the generator and mediator, respectively.

s stands for a complete sample from the training dataset or a generated complete sequence, depending on the specific context.

s t means the t-length prefix of the original sequence, i.e. an incomplete sequence of length t. x denotes a token, and x t stands for a token that appears in the t-th place of a sequence.

Thus s t = [x 0 , x 1 , x 2 , . . . , x t−1 ] while the initial case s 0 is ∅.

Maximum likelihood estimation is equivalent to minimizing the KL divergence using the samples from the real distribution:min DISPLAYFORM0 where G θ (s) is the estimated probability of s by G θ and p data is the underlying real distribution.

Limitations of MLE.

MLE is essentially equivalent to optimizing a directed Kullback-Leibler (KL) divergence between the target distribution P and the currently learned distribution G, denoted as KL(P G).

However, since KL divergence is asymmetric, given finite observations this target is actually not ideal.

As stated in BID0 , MLE tries to minimize DISPLAYFORM1 • When P (s) > 0 and G(s) → 0, the KL divergence grows to infinity, which means MLE assigns an extremely high cost to the "mode dropping" scenarios, where the generator fails to cover some parts of the data.• When G(s) > 0 and P (s) → 0, the KL divergence shrinks to 0, which means MLE assigns an extremely low cost to the scenarios, where the model generates some samples that do not locate on the data distribution.

Likewise, optimizing KL(G P ) will lead to exactly the reversed problems of the two situations.

An ideal solution is to optimize a symmetrized and smoothed version of KL divergence, i.e. the Jensen-Shannon divergence (JSD), which is defined as DISPLAYFORM2 where M = 1 2 (P + G).

However, directly optimizing JSD is conventionally considered as an intractable problem.

JSD cannot be directly evaluated and optimized since the equally interpolated distribution M is usually considered to be unconstructable, as we only have access to the learned model G instead of P .

SeqGAN incorporates two modules, i.e. the generator and discriminator, parametrized by θ and φ respectively, as in the settings of GAN.

By alternatively training these two modules, SeqGAN optimizes such an adversarial target: min Collect two equal-sized mini-batch of samples {sg} and {sp} from G θ and P , respectively 5: DISPLAYFORM0 Mix {sg} and {sp} as {s} 6:Update mediator M φ with {s} via Eq. (9) 7: end for 8:Generate a mini-batch of sequences {s} ∼ G θ 9:Update generator G θ with {s} via Eq. (13) 10: until CoT convergesThe objectives of generator G θ and discriminator D φ in SeqGAN can be formulated as DISPLAYFORM1 Discriminator: DISPLAYFORM2 where s ∼ G θ = [x 1 , ..., x n ] denotes a complete sequence sampled from the generator and the action value BID16 , the fact that SeqGAN is essentially based on model-free reinforcement learning makes it a non-trivial problem for SeqGAN to converge well.

As a result, SeqGAN usually gets stuck in some fake local optimals.

Specifically, although the discriminator can distinguish the samples from the generator easily, it is not able to effectively guide the generator because of the vanishing gradient, as is discussed in a recent survey .

Although this problem can be alleviated by reshaping the reward signals based on the relative rankings of the outputs in a mini-batch BID11 BID8 , they are more technical workarounds than essential solutions.

DISPLAYFORM3 Second, SeqGAN trained via REINFORCE (Williams, 1992) suffers from the "mode collapse" problem, which is similar to the original GAN.

That is to say, the learned distribution "collapses" to the other side of KL divergence, i.e. KL(G P ), which leads to the loss of diversity of generated samples.

In other words, SeqGAN trains the model for better generative quality at the cost of diversity.3 COOPERATIVE TRAINING

To be consistent with the goal that the target distribution should be well-estimated in both quality and diversity senses, an ideal algorithm for such models should be able to optimize a symmetric divergence or distance.

For sequential discrete data modeling, since the data distribution is decomposed into a sequential product of finite-dimension multinomial distributions (always based on the softmax form), the failures of effectively optimizing JSD when the generated and real data distributions are distant, as discussed in , will not appear.

As such, to optimize JSD is feasible.

However, to our knowledge, no previous algorithms provide a direct, low-variance optimization of JSD.

In this paper, we propose Cooperative Training (CoT), as shown in Algorithm 1, to directly optimize a well-estimated unbiased JSD for training such models.

Each iteration of Cooperative Training mainly consists of two parts.

The first part is to train a mediator M φ , which is a density function that estimates a mixture distribution of the learned generative distribution G θ and target latent distribution P = p data as DISPLAYFORM0 Since the mediator is only used as a density prediction module during training, the directed KL divergence is now free from so-called exposure bias for optimization of M φ .

Denote DISPLAYFORM1 Lemma 1 (Mixture Density Decomposition) DISPLAYFORM2 By Lemma 1, for each step, we can simply mix balanced samples from training data and the generator, then train the mediator via Maximum Likelihood Estimation with the mixed samples.

The objective J m (φ) for the mediator M parametrized by φ therefore becomes DISPLAYFORM3 Since the objective of MLE is bias-free for predictive purposes, the estimated M φ is also bias-free when adopted for estimating JSD.

The training techniques and details will be discussed in Section 4.After each iteration, the mediator is exploited to optimize an estimated Jensen-Shannon divergence for G θ : DISPLAYFORM4 Note that the gradient Eq. (10) should be performed for only one step because once G θ is updated the current mediator's estimation M φ becomes inaccurate.

For any sequence or prefix of length t, we have: DISPLAYFORM5 DISPLAYFORM6 The detailed derivations can be found in the supplementary material.

Note that Lemma 2 can be applied recursively.

That is to say, given any sequence s t of arbitrary length t, optimizing s t 's contribution to the expected JSD can be decomposed into optimizing the first term of Eq. FORMULA0 and solving an isomorphic problem for s t−1 , which is the longest proper prefix of s t .

When t = 1, since in Markov decision process the probability for initial state s 0 is always 1.0, it is trivial to prove that the final second term becomes 0.Therefore, Eq. (10) can be reduced through recursively applying Lemma 2.

After removing the constant multipliers and denoting the predicted probability distribution over the action space, i.e. G θ (·|s t ) and M φ (·|s t ), as π g (s t ) and π m (s t ) respectively, the gradient ∇ θ J g (θ) for training generator via Cooperative Training can be formulated as DISPLAYFORM7 For tractable density models with finite discrete action space in each step, the practical effectiveness of this gradient is well guaranteed for the following reasons.

First, with a random initialization of the model, the supports of distributions G θ and P are hardly disjoint.

Second, the first term of Eq. FORMULA0 is to minimize the cross entropy between G and M * , which tries to enlarge the overlap of two distributions.

Third, since the second term of Eq. FORMULA0 is equivalent to maximizing the entropy of G, it encourages the support of G to cover the whole action space, which avoids the case of disjoint supports between G and P .The overall objective of CoT can be formulated as finding the maximal entropy solution of max DISPLAYFORM8 Note the strong connections and differences between the optimization objective of CoT FORMULA0 and that of GAN (4).

FIG0 illustrates the whole Cooperative Training process.

CoT has theoretical guarantee on its convergence.

Theorem 3 (Jensen-Shannon Consistency) If in each step, the mediator M φ of CoT is trained to be optimal, i.e. M φ = M * = 1 2 (G θ + P ), then optimization via Eq. (14) leads to minimization of JSD(G P ).Proof.

Let p denote the intermediate states.

It would be used in the detailed proof.

All we need to show is DISPLAYFORM0 By inversely applying Lemma 2, the left part in Eq. (15) can be recovered as DISPLAYFORM1 which is equivalent to DISPLAYFORM2 Since now mediator is trained to be optimal, i.e. M φ = M * , we have DISPLAYFORM3 This means training through CoT leads to minimization ofĴSD(P G θ ).

When the mediator is trained to be optimal,ĴSD(P G θ ) = JSD(P G θ ).

This verifies the theorem.

CoT has several practical advantages over previous methods, including MLE, Scheduled Sampling (SS) BID3 and adversarial methods like SeqGAN .First, although CoT and GAN both aim to optimize an estimated JSD, CoT is exceedingly more stable than GAN.

This is because the two modules, namely generator and mediator, have similar tasks, i.e. to approach the same data distribution generatively and predictively.

The superiority of CoT over inconsistent methods like Scheduled Sampling is obvious, since CoT theoretically guarantees the training effectiveness.

Compared with methods that require pre-training in order to reduce variance like SeqGAN , CoT is computationally cheaper.

More specifically, under recommended settings, CoT has the same order of computational complexity as MLE.Besides, CoT works independently.

In practice, it does not require model pre-training via conventional methods like MLE.

This is the first time that unbiased unsupervised learning is achieved on sequential discrete data without using supervised approximation for variance reduction or sophisticated smoothing as in Wasserstein GAN with gradient penalty (WGAN-GP) BID7 .

An interesting problem is to ask why we need to train a mediator by mixing the samples from both sources G and P , instead of directly training a predictive modelP on the training set via MLE.

There are basically two points to interpret this.

To apply the efficient training objective 13, one needs to obtain not only the mixture density model M = 1 2 (P + G) but also its decomposed form in each timestep i.e. M φ (s) = n t=1 M φ (s t |s t−1 ), without which the term π m (s t ) in Eq 13 cannot be computed efficiently.

This indicates that if we directly estimate P and compute M = 1 2 (G + P ), the obtained M will be actually useless since its decomposed form is not available.

Besides, as a derivative problem of "exposure bias", there is no guarantee for the modelP to work well on the generated samples i.e. s ∼ G θ to guide the generator towards the target distribution.

Given finite observations, the learned distributionP is trained to provide correct predictions for samples from the target distribution P .

There is no guarantee thatP can stably provide correct predictions for guiding the generator.

Ablation study is provided in the appendix.

Following the synthetic data experiment setting in , we design a synthetic Turing test, in which the negative log-likelihood NLL oracle from an oracle LSTM is calculated for evaluating the quality of samples from the generator.

Particularly, to support our claim that our method causes little mode collapse, we calculated NLL test , which is to sample an extra batch of samples from the oracle, and to calculate the negative log-likelihood measured by the generator.

We show that under this more reasonable setting, our proposed algorithm reaches the state-of-the-art performance with exactly the same network architecture.

Note that models like LeakGAN BID8 contain architecture-level modification, which is orthogonal to our approach, thus will not be included in this part.

The results are shown in TAB2 .

Computational Efficiency Although in terms of time cost per epoch, CoT does not achieve the state-of-the-art, we do observe that CoT is remarkably faster than previous RL-GAN approaches.

Besides, consider the fact that CoT is a sample-based optimization algorithm, which involves time BID3 8.89 8.71/-(MLE) (The same as MLE) 32.54 ± 1.14s Professor Forcing BID10 9 To show the hyperparameter robustness of CoT, we compared it with the similar results as were evaluated in SeqGAN .

DISPLAYFORM0 cost in sampling from the generator, this result is acceptable.

The result also verifies our claim that CoT has the same order (i.e. the time cost only differs in a constant multiplier or extra lower order term) of computational complexity as MLE.Hyper-parameter Robustness.

We perform a hyper-parameter robustness experiment on synthetic data experiment.

When compared with the results of similar experiments as in SeqGAN , our approach shows less sensitivity to hyper-parameter choices, as shown in FIG1 .

Note that since in all our attempts, the evaluated JSD of SeqGAN fails to converge, we evaluated NLL oracle for it as a replacement.

Self-estimated Training Progress Indicator.

Like the critic loss, i.e. estimated Earth Mover Distance, in WGANs, we find that the training loss of the mediator (9), namely balanced NLL, can be a real-time training progress indicator as shown in FIG2 .

Specifically, in a wide range, balanced NLL is a good estimation of real JSD(G P ) with a steady translation, namely, balanced N LL = JSD(G P ) + H(G) + H(P ).

2.900 (σ = 0.025) 3.118 (σ = 0.018) 3.122 RankGAN BID11

As an important sequential data modeling task, zero-prior text generation, especially long and diversified text generation, is a good testbed for evaluating the performance of a generative model.

Following the experiment proposed in LeakGAN BID8 , we choose EMNLP 2017 WMT News Section as our dataset, with maximal sentence length limited to 51.

We pay major attention to both quality and diversity.

To keep the comparison fair, we present two implementations of CoT, namely CoT-basic and CoT-strong.

As for CoT-basic, the generator follows the settings of that in MLE, SeqGAN, RankGAN and MaliGAN.

As for CoT-strong, the generator is implemented with the similar architecture in LeakGAN.For quality evaluation, we evaluated BLEU on a small batch of test data separated from the original dataset.

For diversity evaluation, we evaluated the estimated Word Mover Distance BID9 , which is calculated through training a discriminative model between generated samples and real samples with 1-Lipschitz constriant via gradient penalty as in WGAN-GP BID7 .

To keep it fair, for all evaluated models, the architecture and other training settings of the discriminative models are kept the same.

The results are shown in TAB4 and TAB5 .

In terms of generative quality, CoT-basic achieves state-of-the-art performance over all the baselines with the same architecture-level capacity, especially the long-term robustness at n-gram level.

CoT-strong using a conservative generation strategy, i.e. setting the inverse temperature parameter α higher than 1, as in BID8 achieves the best performance over all compared models.

In terms of generative diversity, the results show that our model achieves the state-of-the-art performance on all metrics including NLL test , which is the optimization target of MLE.

We proposed Cooperative Training, a novel training algorithm for generative modeling of discrete data.

CoT optimizes Jensen-Shannon Divergence, which does not have the exposure bias problem as the forward KLD.

Models trained via CoT shows promising results in sequential discrete data modeling tasks, including sample quality and the generalization ability in likelihood prediction tasks.

B SAMPLE COMPARISON AND DISCUSSION TAB6 shows samples from some of the most powerful baseline models and our model.

• CoT produces remarkably more diverse and meaningful samples when compared to Leak-GAN.• The consistency of CoT is significantly improved when compared to MLE.

The Optimal Balance for Cooperative Training We find that the same learning rate and iteration numbers for the generator and mediator seems to be the most competitive choice.

As for the architecture choice, we find that the mediator needs to be slightly stronger than the generator.

For the best result in the synthetic experiment, we adopt exactly the same generator as other compared models and a mediator whose hidden state size is twice larger (with 64 hidden units) than the generator.

Theoretically speaking, we can and we should sample more batches from G θ and P respectively for training the mediator in each iteration.

However, if no regularizations are used when training the mediator, it can easily over-fit, leading the generator's quick convergence in terms of KL(G θ P ) or NLL oracle , but divergence in terms of JSD(G θ P ).

Empirically, this could be alleviated by applying dropout techniques BID15 with 50% keeping ratio before the output layer of RNN.

After applying dropout, the empirical results show good consistency with our theory that, more training batches for the mediator in each iteration is always helpful.

However, applying regularizations is not an ultimate solution and we look forward to further theoretical investigation on better solutions for this problem in the future.

(5)

" I think it was alone because I can do that, when you're a lot of reasons, " he said.(6) It's the only thing we do, we spent 26 and $35(see how you do is we lose it," said both sides in the summer.

CoT(1) We focus the plans to put aside either now, and which doesn't mean it is to earn the impact to the government rejected.(2) The argument would be very doing work on the 2014 campaign to pursue the firm and immigration officials, the new review that's taken up for parking.(3) This method is true to available we make up drink with that all they were willing to pay down smoking.(4) The number of people who are on the streaming boat would study if the children had a bottle -but meant to be much easier, having serious ties to the outside of the nation.(5) However, they have to wait to get the plant in federal fees and the housing market's most valuable in tourism.

MLE (1) after the possible cost of military regulatory scientists, chancellor angela merkel's business share together a conflict of major operators and interest as they said it is unknown for those probably 100 percent as a missile for britain.(2) but which have yet to involve the right climb that took in melbourne somewhere else with the rams even a second running mate and kansas.

(3) " la la la la 30 who appeared that themselves is in the room when they were shot her until the end " that jose mourinho could risen from the individual .

(4) when aaron you has died, it is thought if you took your room at the prison fines of radical controls by everybody, if it's a digital plan at an future of the next time.

Possible Derivatives of CoT The form of equation 13 can be modified to optimize other objectives.

One example is the backward KLD (a.k.a.

Reverse KLD) i.e. KL(G P ).

In this case, the objective of the so-called "Mediator" and "Generator" thus becomes:"Mediator", now it becomes a direct estimatorP φ of the target distribution P : DISPLAYFORM0 Generator: DISPLAYFORM1 Such a model suffers from so-called mode-collapse problem, as is analyzed in Ian's GAN Tutorial BID5 .

Besides, as the distribution estimatorP φ inevitably introduces unpredictable behaviors when given unseen samples i.e. samples from the generator, the algorithm sometimes fails (numerical error) or diverges.

In our successful attempts, the algorithm produces similar (not significantly better than) results as CoT. The quantitive results are shown as follows: Although under evaluation of weak metrics like BLEU, if successfully trained, the model trained via Reverse KL seems to be better than that trained via CoT, the disadvantage of Reverse KL under evaluation of more strict metric like eWMD indicates that Reverse KL does fail in learning some aspects of the data patterns e.g. completely covering the data mode.

<|TLDR|>

@highlight

We proposed Cooperative Training, a novel training algorithm for generative modeling of discrete data.