Sequence generation models such as recurrent networks can be trained with a diverse set of learning algorithms.

For example, maximum likelihood learning is simple and efficient, yet suffers from the exposure bias problem.

Reinforcement learning like policy gradient addresses the problem but can have prohibitively poor exploration efficiency.

A variety of other algorithms such as RAML, SPG, and data noising, have also been developed in different perspectives.

This paper establishes a formal connection between these algorithms.

We present a generalized entropy regularized policy optimization formulation, and show that the apparently divergent algorithms can all be reformulated as special instances of the framework, with the only difference being the configurations of reward function and a couple of hyperparameters.

The unified interpretation offers a systematic view of the varying properties of exploration and learning efficiency.

Besides, based on the framework, we present a new algorithm that dynamically interpolates among the existing algorithms for improved learning.

Experiments on machine translation and text summarization demonstrate the superiority of the proposed algorithm.

Sequence generation is a ubiquitous problem in many applications, such as machine translation BID28 , text summarization BID13 BID25 , image captioning BID15 , and so forth.

Great advances in these tasks have been made by the development of sequence models such as recurrent neural networks (RNNs) with different cells BID12 BID6 and attention mechanisms BID1 BID19 .

These models can be trained with a variety of learning algorithms.

The standard training algorithm is based on maximum-likelihood estimation (MLE) which seeks to maximize the log-likelihood of ground-truth sequences.

Despite the computational simplicity and efficiency, MLE training suffers from the exposure bias BID24 .

That is, the model is trained to predict the next token given the previous ground-truth tokens; while at test time, since the resulting model does not have access to the ground truth, tokens generated by the model itself are instead used to make the next prediction.

This discrepancy between training and test leads to the issue that mistakes in prediction can quickly accumulate.

Recent efforts have been made to alleviate the issue, many of which resort to the reinforcement learning (RL) techniques BID24 BID2 BID8 .

For example, BID24 adopt policy gradient BID29 that avoids the training/test discrepancy by using the same decoding strategy.

However, RL-based approaches for sequence generation can face challenges of prohibitively poor sample efficiency and high variance.

For more practical training, a diverse set of methods has been developed that are in a middle ground between the two paradigms of MLE and RL.

For example, RAML adds reward-aware perturbation to the MLE data examples; SPG BID8 leverages reward distribution for effective sampling of policy gradient.

Other approaches such as data noising BID34 ) also show improved results.

In this paper, we establish a unified perspective of the broad set of learning algorithms.

Specifically, we present a generalized entropy regularized policy optimization framework, and show that the apparently diverse algorithms, such as MLE, RAML, SPG, and data noising, can all be re-formulated as special instances of the framework, with the only difference being the choice of reward and the values of a couple of hyperparameters ( FIG0 ).

In particular, we show MLE is equivalent to using a delta-function reward that assigns 1 to samples that exactly match data examples while −∞ to any other samples.

Such extremely restricted reward has literally disabled any exploration of the model beyond training data, yielding the exposure bias.

Other algorithms essentially use rewards that are more smooth, and also leverage model distribution for exploration, which generally results in a larger effective exploration space, more difficult training, and better test-time performance.

Besides the new understandings of the existing algorithms, the unified perspective also facilitates to develop new algorithms for improved learning.

We present an example new algorithm that, as training proceeds, gradually expands the exploration space by annealing the reward and hyperparameter values.

The annealing in effect dynamically interpolates among the existing algorithms.

Experiments on machine translation and text summarization show the interpolation algorithm achieves significant improvement over the various existing methods.

Sequence generation models are usually trained to maximize the log-likelihood of data by feeding the ground-truth tokens during decoding.

Reinforcement learning (RL) addresses the discrepancy between training and test by also using models' own predictions at training time.

Various RL approaches have been applied for sequence generation, such as policy gradient BID24 and actor-critic BID2 .

Softmax policy gradient (SPG) BID8 additionally incorporates the reward distribution to generate high-quality sequence samples.

The algorithm is derived by applying a log-softmax trick to adapt the standard policy gradient objective.

Reward augmented maximum likelihood (RAML) is an algorithm in between MLE and policy gradient.

It is originally developed to go beyond the maximum likelihood criteria and incorporate task metric (such as BLEU for machine translation) to guide the model learning.

Mathematically, RAML shows that MLE and maximum-entropy policy gradient are respectively minimizing KL divergences in opposite directions.

We reformulate both SPG and RAML in a new perspective, and show they are precisely instances of a general entropy regularized policy optimization framework.

The new framework provides a more principled formulation for both algorithms.

Besides the algorithms discussed in the paper, there are other learning methods for sequence models.

For example, Hal BID11 BID16 ; BID32 use a learningto-search paradigm for sequence generation or structured prediction.

Scheduled Sampling adapts MLE by randomly replacing ground-truth tokens with model predictions as the input for decoding the next-step token.

Our empirical comparison shows improved performance of the proposed algorithm.

Policy optimization for reinforcement learning is studied extensively in robotics and game environment.

For example, BID23 introduce a relative entropy regularization to reduce information loss during learning.

BID26 develop a trust-region approach for monotonic improvement.

BID7 ; BID17 ; BID0 study the policy optimization algorithms in a probabilistic inference perspective.

The entropy-regularized policy optimization formulation presented here can be seen as a generalization of many of the previous policy optimization methods, as shown in the next section.

Besides, we formulate the framework in the sequence generation context.

We first present a generalized formulation of an entropy regularized policy optimization framework, to which a broad set of learning algorithms for sequence generation are connected.

In particular, we show the conventional maximum likelihood learning is a special case of the policy optimization formulation.

This provides new understandings of the exposure bias problem as well as the exploration efficiency of the algorithms.

We further show that the framework subsumes as special cases other well-known learning methods that were originally developed in diverse perspectives.

We thus establish a unified, principled view of the broad class of works.

Let us first set up the notations for the sequence generation setting.

Let x be the input and y = (y 1 , . . .

, y T ) the sequence of T tokens in the target space.

For example, in machine translation, x is the sentence in source language and y is in target language.

Let (x, y * ) be a training example drawn from the empirical data distribution, where y * is the ground truth sequence.

We aim to learn a sequence generation model p θ (y|x) = t p θ (y t |y 1:t−1 , x) parameterized with θ.

The model can, for example, be a recurrent network.

It is worth noting that though we present in the sequence generation context, the formulations can straightforwardly be extended to other settings such as robotics and game environment.

Policy optimization is a family of reinforcement learning (RL) algorithms that seeks to learn the parameter θ of the model p θ (a.k.a policy).

Given a reward function R(y|y * ) ∈ R (e.g., BLEU score in machine translation) that evaluates the quality of generation y against the true y * , the general goal of policy optimization is to maximize the expected reward.

A rich research line of entropy regularized policy optimization (ERPO) stabilizes the learning by augmenting the objective with information theoretic regularizers.

Here we present a generalized formulation of ERPO.

Assuming a general distribution q(y|x) (more details below), the objective we adopt is written as DISPLAYFORM0 where KL(· ·) is the Kullback-Leibler divergence forcing q to stay close to p θ ; H(·) is the Shannon entropy imposing maximum entropy assumption on q; and α and β are balancing weights of the respective terms.

In the RL literature, the distribution q has taken various forms, leading to different policy optimization algorithms.

For example, setting q to a non-parametric policy and β = 0 results in the prominent relative entropy policy search BID23 algorithm.

Assuming q as a parametric distribution and α = 0 leads to the commonly-used maximum entropy policy gradient BID36 BID10 .

Letting q be a variational distribution and β = 0 corresponds to the probabilistic inference formulation of policy gradient BID0 BID17 .

Related objectives have also been used in other popular RL algorithms BID26 BID30 .We assume a non-parametric q.

The above objective can be maximized with an EM-style procedure that iterates two coordinate ascent steps optimizing q and θ, respectively.

At iteration n: DISPLAYFORM1 The E-step is obtained with simple Lagrange multipliers.

Note that q has a closed-form solution in the E-step.

We can have an intuitive interpretation of its form.

First, it is clear to see that if α → ∞, we have q n+1 = p n θ .

This is also reflected in the objective Eq.(1) where the weight α encourages q to be close to p θ .

Second, the weight β serves as the temperature of the q softmax distribution.

In particular, a large temperature β → ∞ makes q a uniform distribution, which is consistent to the outcome of an infinitely large maximum entropy regularization in Eq.(1).

In terms of the M-step, the update rule can be interpreted as maximizing the log-likelihood of samples from the distribution q.

In the context of sequence generation, it is sometimes more convenient to express the equations at token level, as shown shortly.

To this end, we decompose R(y|y * ) along the time steps: DISPLAYFORM2 where ∆R(y t |y * , y 1:t−1 ) measures the reward contributed by token y t .

The solution of q in Eq.(2) can then be re-written as: DISPLAYFORM3 The above ERPO framework has three key hyperparameters, namely (R, α, β).

In the following, we show that different values of the three hyperparameters correspond to different learning algorithms ( FIG0 ).

We first connect MLE to the above general formulation, and compare and discuss the properties of MLE and regular ERPO from the new perspective.

Maximum likelihood estimation is the most widely-used approach to learn a sequence generation model due to its simplicity and efficiency.

It aims to find the optimal parameter value that maximizes the data log-likelihood: DISPLAYFORM0 As discussed in section 1, MLE suffers from the exposure bias problem as the model is only exposed to the training data, rather than its own predictions, by using the ground-truth subsequence y * 1:t−1 to evaluate the probability of y * t .

We show that the MLE objective can be recovered from Eq.(2) with specific reward and weight configurations.

Consider a δ-reward defined as 1 : DISPLAYFORM1 Let (R = R δ , α → 0, β = 1).

From the E-step of Eq.(2), we have q(y|x) = 1 if y = y * and 0 otherwise.

The M-step is therefore equivalent to arg max θ log p θ (y * |x), which recovers precisely the MLE objective in Eq.(5).That is, MLE can be seen as an instance of the policy optimization algorithm with the δ-reward and the above weight values.

Any sample y that fails to match precisely the data y * will receive a negative infinite reward and never contribute to model learning.

The ERPO reformulation of MLE provides a new statistical explanation of the exposure bias problem.

Specifically, a very small α value makes the model distribution ignored during sampling from q, while the δ-reward permits only samples that match training examples.

The two factors in effect make void any exploration beyond the small set of training data FIG1 ), leading to a brittle model that performs poorly at test time due to the extremely restricted exploration.

On the other hand, however, a key advantage of the δ-reward specification is that its regular reward shape allows extreme pruning of the huge sample space, resulting in a space that includes exactly the training examples.

This makes the MLE implementation very simple and the computation very efficient in practice.

On the contrary, common rewards (e.g., BLEU) used in policy optimization are more smooth than the δ-reward, and permit exploration in a broader space.

However, such rewards usually do not have a regular shape as the δ-reward, and thus are not amenable to sample space pruning.

Generally, a larger exploration space would lead to a harder training problem.

Also, when it comes to the huge sample space, the rewards are still very sparse (e.g., most sequences have BLEU=0 against a reference sequence).

Such reward sparsity can make exploration inefficient and even impractical.1 For token-level, define R δ (y1:t|y * ) = t/T * if y1:t = y * 1:t and −∞ otherwise, where T * is the length of y * .

Note that the R δ value of y = y * can also be set to any constant larger than −∞. Given the opposite algorithm behaviors in terms of exploration and computation efficiency, it is a natural idea to seek a middle ground between the two extremes to combine the advantages of both.

A broad set of such approaches have been recently developed.

We re-visit some of the popular ones, and show that these apparently divergent approaches can all be reformulated within our ERPO framework (Eqs.1-4) with varying reward and weight specifications.

RAML was originally proposed to incorporate task metric reward into the MLE training, and has shown superior performance to the vanilla MLE.

Specifically, it introduces an exponentiated reward distribution e(y|y * ) ∝ exp{R(y|y * )} where R, as in vanilla policy optimization, is a task metric such as BLEU.

RAML maximizes the following objective: DISPLAYFORM0 That is, unlike MLE that directly maximizes the data log-likelihood, RAML first perturbs the data proportionally to the reward distribution e, and maximizes the log-likelihood of the resulting samples.

The RAML objective reduces to the vanilla MLE objective if we replace the task reward R in e(y|y * ) with the MLE δ-reward (Eq.6).

The relation between MLE and RAML still holds within our new formulation (Eqs.1-2).

In particular, similar to how we recovered MLE from Eq.(2), let (α → 0, β = 1) 2 , but set R to the task metric reward, then the M-step of Eq.(2) is precisely equivalent to maximizing the above RAML objective.

Formulating within the same framework allows us to have an immediate comparison between RAML and others.

In particular, compared to MLE, the use of smooth task metric reward R instead of R δ permits a larger effective exploration space surrounding the training data FIG1 ), which helps to alleviate the exposure bias problem.

On the other hand, α → 0 as in MLE still limits the exploration as it ignores the model distribution.

Thus, RAML takes a step from MLE toward regular RL, and has effective exploration space size and exploration efficiency in between.

SPG BID8 was developed in the perspective of adapting the vanilla policy gradient BID29 to use reward for sampling.

SPG has the following objective: DISPLAYFORM0 where R is a common reward as above.

As a variant of the standard policy gradient algorithm, SPG aims to address the exposure bias problem and shows promising results BID8 .We show SPG can readily fit into our ERPO framework.

Specifically, taking gradient of Eq.(8) w.r.t θ, we immediately get the same update rule as in Eq. FORMULA1 with (α = 1, β = 0, R = common reward).Note that the only difference between the SPG and RAML configuration is that now α = 1.

SPG thus moves a step further than RAML by leveraging both the reward and the model distribution for full exploration FIG1 ).

Sufficient exploration at training time would in theory boost the test-time performance.

However, with the increased learning difficulty, additional sophisticated optimization and approximation techniques have to be used BID8 to make the training practical.

Adding noise to training data is a widely adopted technique for regularizing models.

Previous work BID34 has proposed several data noising strategies in the sequence generation context.

For example, a unigram noising, with probability γ, replaces each token in data y * with a sample from the unigram frequency distribution.

The resulting noisy data is then used in MLE training.

Though previous literature has commonly seen such techniques as a data pre-processing step that differs from the above learning algorithms, we show the ERPO framework can also subsume data noising as a special instance.

Specifically, starting from the ERPO reformulation of MLE which takes (R = R δ , α → 0, β = 1) (section 3.2), data noising can be formulated as using a locally relaxed variant of R δ .

For example, assume y has the same length with y * and let ∆ y,y * be the set of tokens in y that differ from the corresponding tokens in y * , then a simple data noising strategy that randomly replaces a single token y * t with another uniformly picked token is equivalent to using a reward R δ (y|y * ) that takes 1 when |∆ y,y * | = 1 and −∞ otherwise.

Likewise, the above unigram noising BID34 ) is equivalent to using a reward DISPLAYFORM0 where u(·) is the unigram frequency distribution.

With a relaxed (i.e., smoothed) reward, data noising expands the exploration space of vanilla MLE locally FIG1 ).

The effect is essentially the same as the RAML algorithm (section 3.3), except that RAML expands the exploration space based on the task metric reward.

Other Algorithms Ranzato et al. (2016) made an early attempt to address the exposure bias problem by exploiting the classic policy gradient algorithm BID29 and mixing it with MLE training.

We show in the supplementary materials that the algorithm is closely related to the ERPO framework, and can be recovered with moderate approximations.

Section 2 discusses more relevant algorithms for sequence generation learning.

We have presented the generalized ERPO framework, and connected a series of well-used learning algorithms by showing that they are all instances of the framework with certain specifications of the three hyperparameters (R, α, β).

Each of the algorithms can be seen as a point in the hyperparameter space FIG0 ).

Generally, a point with a more restricted reward function R and a very small α tends to have a smaller effective exploration space and allow efficient learning (e.g., MLE), while in contrast, a point with smooth R and a larger α would lead to a more difficult learning problem, but permit more sufficient exploration and better test-time performance (e.g., (softmax) policy gradient).The unified perspective provides new understandings of the existing algorithms, and also facilitates to develop new algorithms for further improvement.

Here we present an example algorithm that interpolates the existing ones.

The interpolation algorithm exploits the natural idea of starting learning from the most restricted yet easiest problem configuration, and gradually expands the exploration space to reduce the discrepancy from the test time.

The easy-to-hard learning paradigm resembles the curriculum learning BID4 ).

As we have mapped the algorithms to points in the hyperparameter space, interpolation becomes very straightforward, which requires only annealing of the hyperparameter values.

Specifically, in the general update rules Eq.(2), we would like to anneal from using R δ to using smooth common reward, and anneal from exploring by only R to exploring by both R and p θ .

Let R comm denote a common reward (e.g., BLEU).

The interpolated reward can be written in the form R = λR comm + (1 − λ)R δ , for λ ∈ [0, 1].

Plugging R into q in Eq. FORMULA1 and re-organizing the scalar weights, we obtain the numerator of q in the form: c · (λ 1 log p θ + λ 2 R comm + λ 3 R δ ), whereModel BLEU MLE 26.44 ± 0.18 RAML 27.22 ± 0.14 SPG BID8 26.62 ± 0.05 MIXER BID24 26.53 ± 0.11 Scheduled Sampling 26.76 ± 0.17Ours 27.82 ± 0.11 Table 1 : Results of machine translation.(λ 1 , λ 2 , λ 3 ) is defined as a distribution (i.e., λ 1 +λ 2 +λ 3 = 1), and, along with c ∈ R, are determined by (α, β, λ).

For example, λ 1 = α/(α + 1).

We gradually increase λ 1 and λ 2 and decrease λ 3 as the training proceeds.

Further, noting that R δ is a Delta function (Eq.6) which would make the above direct function interpolation problematic, we borrow the idea from the Bayesian spike-and-slab factor selection method BID14 .

That is, we introduce a categorical random variable z ∈ {1, 2, 3} that follows the distribution (λ 1 , λ 2 , λ 3 ), and augment q as q(y|x, z) ∝ exp{c · (1(z = 1) log p θ + 1(z = 2)R comm + 1(z = 3)R δ )}.

The M-step is then to maximize the objective with z marginalized out: DISPLAYFORM0 The spike-and-slab adaption essentially transforms the product of experts in q to a mixture, which resembles the bang-bang rewarded SPG method BID8 where the name bang-bang refers to a system that switches abruptly between extreme states (i.e., the z values).

Finally, similar to BID8 , we adopt the token-level formulation (Eq.4) and associate each token with a separate variable z.

We provide the pseudo-code of the interpolation algorithm in the supplements.

It is notable that Ranzato et al. FORMULA0 also develop an annealing strategy that mixes MLE and policy gradient training.

As discussed in section 3 and the supplements, the algorithm can be seen as a special instance of the ERPO framework (with moderate approximation) we have presented.

Next section shows improved performance of the proposed, more general algorithm compared to BID24 .

We evaluate the above interpolation algorithm in the tasks of machine translation and text summarization.

The proposed algorithm consistently improves over a variety of previous methods.

Code will be released upon acceptance.

Setup In both tasks, we follow previous work BID24 and use an attentional sequence-to-sequence model BID19 where both the encoder and decoder are single-layer LSTM recurrent networks.

The dimensions of word embedding, RNN hidden state, and attention are all set to 256.

We apply dropout of rate 0.2 on the recurrent hidden state.

We use Adam optimization for training, with an initial learning rate of 0.001 and batch size of 64.

At test time, we use beam search decoding with a beam width of 5.

Please see the supplementary materials for more configuration details.

Dataset Our dataset is based on the common IWSLT 2014 BID5 German-English machine translation data, as also used in previous evaluation BID24 .

After proper pre-processing as described in the supplementary materials, we obtain the final dataset with train/dev/test size of around 146K/7K/7K, respectively.

The vocabulary sizes of German and English are around 32K and 23K, respectively.

Results The BLEU metric BID22 ) is used as the reward and for evaluation.

Table 1 shows the test-set BLEU scores of various methods.

Besides the approaches described above, we also compare with the Scheduled Sampling method which combats the exposure bias by feeding model predictions at randomly-picked decoding steps during training.

From the table, we can see the various approaches such as RAML provide improved performance over the vanilla MLE, as more sufficient exploration is made at training time.

Our proposed new algorithm performs best, as it interpolates among the existing algorithms to gradually increase the exploration space and solve the generation problem better.

FIG2 shows the test-set BLEU scores against the training steps.

We can see that, with annealing, our algorithm improves BLEU smoothly, and surpasses other algorithms to converge at a better point.

Table 2 : Results of text summarization.

Dataset We use the popular English Gigaword corpus BID9 for text summarization, and pre-processed the data following BID25 .

The resulting dataset consists of 200K/8K/2K source-target pairs in train/dev/test sets, respectively.

More details are included in the supplements.

Results The ROUGE metrics (including -1, -2, and -L) BID18 are the most commonly used metrics for text summarization.

Following previous work BID8 , we use the summation of the three ROUGE metrics as the reward in the learning algorithms.

Table 2 show the results on the test set.

The proposed interpolation algorithm achieves the best performance on all the three metrics.

For easier comparison, FIG3 shows the improvement of each algorithm compared to MLE in terms of ROUGE-L. The RAML algorithm, which performed well in machine translation, falls behind other algorithms in text summarization.

In contrast, our method consistently provides the best results.

We have presented a unified perspective of a variety of well-used learning algorithms for sequence generation.

The framework is based on a generalized entropy regularized policy optimization formulation, and we show these algorithms are mathematically equivalent to specifying certain hyperparameter configurations in the framework.

The new principled treatment provides systematic understanding and comparison among the algorithms, and inspires further enhancement.

The proposed interpolation algorithm shows consistent improvement in machine translation and text summarization.

We would be excited to extend the framework to other settings such as robotics and game environments.

A POLICY GRADIENT & MIXER BID24 made an early attempt to address the exposure bias problem by exploiting the policy gradient algorithm BID29 .

Policy gradient aims to maximizes the expected reward: DISPLAYFORM0 where R P G is usually a common reward function (e.g., BLEU).

Taking gradient w.r.t θ gives: DISPLAYFORM1 We now reveal the relation between the ERPO framework we present and the policy gradient algorithm.

Starting from the M-step of Eq.(2) and setting (α = 1, β = 0) as in SPG (section 3.4), we use p θ n as the proposal distribution and obtain the importance sampling estimate of the gradient (we omit the superscript n for notation simplicity): DISPLAYFORM2 where Z θ = y exp{log p θ + R} is the normalization constant of q, which can be considered as adjusting the step size of gradient descent.

We can see that Eq.(12) recovers Eq.(11) if we further set R = log R P G , and omit the scaling factor Z θ .

In other words, policy gradient can be seen as a special instance of the general ERPO framework with (R = log R P G , α = 1, β = 0) and with Z θ omitted.

The MIXER algorithm BID24 incorporates an annealing strategy that mixes between MLE and policy gradient training.

Specifically, given a ground-truth example y * , the first m tokens y * 1:m are used for evaluating MLE loss, and starting from step m + 1, policy gradient objective is used.

The m value decreases as training proceeds.

With the relation between policy gradient and ERPO as established above, MIXER can be seen as a specific instance of the proposed interpolation algorithm (section 4) that follows a restricted annealing strategy for token-level hyperparameters (λ 1 , λ 2 , λ 3 ).

That is, for t < m in Eq.4 (i.e.,the first m steps), (λ 1 , λ 2 , λ 3 ) is set to (0, 0, 1) and c = 1, namely the MLE training; while for t > m, (λ 1 , λ 2 , λ 3 ) is set to (0.5, 0.5, 0) and c = 2.

Algorithm 1 summarizes the interpolation algorithm described in section 4.

Get training example (x, y * )

for t = 0, 1, . . .

, T do 5: DISPLAYFORM0 if z = 1 then Sample token y t ∼ exp{c · log p θ (y t |y 1:t−1 , x)} 8: DISPLAYFORM1 Sample token y t ∼ exp{c · ∆R comm (y t |y 1:t−1 , y * )} 10: DISPLAYFORM2 Sample token y t ∼ exp{c · ∆R δ }, i.e., set y t = y * t

end if

end for

Update θ by maximizing the log-likelihood log p θ (y|x)

Anneal λ by increasing λ 1 and λ 2 and decreasing λ 3 16: until convergence C EXPERIMENTAL SETTINGS C.1 DATA PRE-PROCESSING For the machine translation dataset, we follow BID20 for data pre-processing.

In text summarization, we sampled 200K out of the 3.8M pre-processed training examples provided by BID25 for the sake of training efficiency.

We used the refined validation and test sets provided by BID35 .

For RAML , we use the sampling approach (n-gram replacement) by BID20 to sample from the exponentiated reward distribution.

For each training example we draw 10 samples.

The softmax temperature is set to τ = 0.4.For Scheduled Sampling , the decay function we used is inverse-sigmoid decay.

The probability of sampling from model i = k/(k + exp (i/k)), where k is a hyperparameter controlling the speed of convergence, which is set to 500 and 600 in the machine translation and text summarization tasks, respectively.

For MIXER BID24 , the advantage function we used for policy gradient is R(y 1:T |y * )− R(y 1:m |y * ).For the proposed interpolation algorithm, we initialize the weights as (λ 1 , λ 2 , λ 3 ) = (0.04, 0, 0.96), and increase λ 1 and λ 2 while decreasing λ 3 every time when the validation-set reward decreases.

Specifically, we increase λ 1 by 0.06 once and increase λ 2 by 0.06 for four times, periodically.

For example, at the first time the validation-set reward decreases, we increase λ 1 , and at the second to fifth time, we increase λ 2 , and so forth.

The weight λ 3 is decreased by 0.06 every time we increase either λ 1 or λ 2 .

Notice that we would not update θ when the validation-set reward decreases.

Here we present additional results of machine translation using a dropout rate of 0.3 (Table 3 ).

The improvement of the proposed interpolation algorithm over the baselines is comparable to that of using dropout 0.2 (Table 1 in the paper).

For example, our algorithm improves over MLE by 1.5 BLEU points, and improves over the second best performing method RAML by 0.49 BLEU points.

(With dropout 0.2 in Table 1 , the improvements are 1.42 BLEU and 0.64, respectively.)

We tested with dropout 0.5 and obtained similar results.

The proposed interpolation algorithm outperforms existing approaches with a clear margin.

FIG6 shows the convergence curves of the comparison algorithms.

Model BLEU MLE 26.63 ± 0.11 RAML 27.64 ± 0.09 SPG BID8 26.89 ± 0.06 MIXER BID24 27.00 ± 0.13 Scheduled Sampling 27.03 ± 0.15 Ours 28.13 ± 0.12 Table 3 : Results of machine translation when dropout is 0.3.

Table 3 ) picked according to the validation set performance.

<|TLDR|>

@highlight

A unified perspective of various learning algorithms for sequence generation, such as MLE, RL, RAML, data noising, etc.