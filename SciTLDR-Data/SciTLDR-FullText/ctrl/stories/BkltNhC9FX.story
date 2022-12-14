Modern neural architectures critically rely on attention for mapping structured inputs to sequences.

In this paper we show that prevalent attention architectures do not adequately model the dependence among the attention and output tokens across a predicted sequence.

We present an alternative architecture called  Posterior Attention Models that after a principled factorization of the full joint distribution of the attention and output variables, proposes two major changes.

First, the position where attention is marginalized is changed from the input to the output.

Second, the attention propagated to the next decoding stage is a posterior attention distribution conditioned on the output.

Empirically on five translation and two morphological inflection tasks the proposed posterior attention models yield better BLEU score and alignment accuracy than existing attention models.

Attention is a critical module of modern neural models for sequence to sequence learning as applied to tasks like translation, grammar error correction, morphological inflection, and speech to text conversion.

Attention specifies what part of the input is relevant for each output.

Many variants of attention have been proposed including soft BID1 BID14 , sparse BID15 , local BID14 , hard (Xu et al., 2015; Zaremba & Sutskever, 2015) , monotonic hard (Yu et al., 2016; BID0 , hard non-monotonic (Wu et al., 2018; , and variational BID6 , The most prevalent of these is soft attention that computes attention for each output as a multinomial distribution over the input states.

The multinomial probabilities serve as weights, and an attention weighted sum of input states serves as relevant context for the output and subsequent attention.

Soft attention is end to end differentiable, easy to implement, and hence widely popular.

Hard attention and sparse attentions are difficult to implement and not popularly used.

In this paper we revisit the statistical soundness of the way soft attention and other variants capture the dependence between attention and output variables, and among multiple attention variables along the length of the sequence.

Our investigation leads to a more principled model that we call the Posterior Attention Model (PAM).

We start with an explicit joint distribution of all output and attention variables in a predicted sequence.

We then propose a tractable approximation that retains the advantages of forward dependence and token-level decomposition and thus leads to efficient training and inference.

The computations performed at each decode step has two important differences with existing models.

First, at each decoding step the probability of the output token is a mixture of output probability for each attention.

In contrast, existing models take a mixture of the input, and compute a single output distribution from this diffused mixed input.

We show that our direct coupling of output and attention gives the benefit of hard attention without its computational challenges.

Second, we introduce the notion of a posterior attention distribution, that is, the attention distribution conditioned on the current output.

We show that it is both statistically sounder and more accurate to condition subsequent attention on the output corrected posterior attention, rather than the output independent prior attention as in existing models.

We evaluate the posterior attention model on five translation tasks and two morphological inflection tasks.

We show that posterior attention provides improved BLEU score, higher alignment accuracy, and better input coverage.

We also empirically analyze the reasons behind the improved performance of the posterior attention model.

We discover that the entropy of posterior attention is much lower than entropy of soft attention.

This is a significant finding that challenges the current practice of computing attention distribution without considering the output token.

The running time overhead of posterior attention is only 40% over existing soft-attention.

Our goal is to model the conditional distribution Pr(y|x) of an output sequence y = y 1 , . . .

, y n given an input sequence x = x 1 , . . . , x m .

Each output y t is a discrete token from a typically large vocabulary V .

Each x j can be any abstract input.

Typically a RNN encodes the input sequence into a sequence of state vectors x 1 , . . .

, x m , which we jointly denote as x 1:m .

Each y t depends not only on other tokens in the sequence, but on some specific focused part of the input sequence.

A hidden variable a t , called the attention variable, denotes which part of x 1:m the output y t depends on.

We denote the set of all attention as a = a 1 , . . .

, a n .

During training the input x and output y are observed but the attention a is hidden.

Hence, we write Pr(y|x) as DISPLAYFORM0 Pr(y 1 , . . .

, y n , a 1 , . . .

, a n |x 1:m )The number of variables involved in this summation is daunting, and we need to approximate.

We first review how existing soft attention-based encoder decoder models handle this challenge.

Existing Encoder-Decoder (ED) networks factorize Pr(y|x 1:m ) by applying chain rule on y variables as n t=1 Pr(y t |x 1:m , y 1 , . . .

, y t???1 ).

A decoder RNN summarizes the variable length history y 1 , . . .

, y t???1 as a decoder state s t , so that Pr(y|x 1:m ) = n t=1 Pr(y t |x 1:m , s t ).

The distribution of each attention variable a t is computed as a function of the decoder state and encoder state as: xa,st) .

Here A ?? (., .) is an end-to-end trained function of input state x a and decoder state s t .

We will use the short form P t (a) for Pr(a t |x 1:m , s t ).

Thereafter, an attention weighted sum of the input states a P t (a)x a called input context c t is computed.

The distribution of y t is computed from c t (capturing attention) and s t capturing previous y as: DISPLAYFORM0 DISPLAYFORM1 Next, c t is fed to the decoder RNN along with y t for computing the next state: s t+1 = RNN(s t , c t , y t ).

as an approximation of the full joint distribution in Equation 1, we find that the treatment of the attention variables has been rather ad hoc.

Attention was introduced as an after-thought of factorizing on the y t variables, the interaction among various a t s is not expressed, and the influence of a t on y t by diffusing the inputs is unprincipled.

There have been a number of recent works (Wu et al., 2018; BID6 which model attention as latent-alignment variable in the joint distribution Equation 1.

The model becomes more tractable by assuming that the output y t at each step is dependent only on a t and previous outputs y <t i.e. P (y t |y <t , a <t , a t ) = P (y t |y <t , a t ).

Both Wu et al. (2018) further assume that attention at each time step is independent of attention at other timesteps, and marginalize over all attentions at each time-step as in Equation 3.

BID6 also rely on the same assumption but instead of direct marginalization use variational methods.

All these models can be considered as a neuralization of IBM Model 1.

DISPLAYFORM0 Such mean-field assumption while making the model significantly simpler ignore relationships between attentions which is undesirable.

Moreover as we will see in the experiments, they also ignore consistency between attention and output variables.

We next present a principled model of the interaction of the various attention and output variables, which is as efficient as the mean field factorization approach while allowing more realistic latent behavior.

We call our proposed approach: Posterior Attention Models or PAM.

Our goal is to express the joint distribution as a product of tractable terms computed at each time step much like in existing ED model, but via a less ad hoc treatment of the attention variables a 1 , . . .

, a n .

We use y <t , a <t to denote all output and attention variables before t that is, y 1 , , . . .

y t???1 , a 1 , . . .

a t???1 .Here and in the rest of the paper we will drop x 1:m to use the shorter form P (y) for Pr(y|x 1:m ).

We first factorize Eq 1 via chain rule, like in ED but jointly on both a and y. DISPLAYFORM0 P (y n |y <n , a <n , a n )P (a n |y <n , a <n )P (a <n |y <n )P (y <n )We then make two mild assumptions: First, the same local attention assumption that y t is dependent only on a t and previous outputs y <t as detailed earlier.

Second, a Markovian assumption on the attention variables i.e., P (a n |y <n , a <n ) = P (a n |y <n , a n???1 ).These together allows us to simplify the above joint as: DISPLAYFORM1 an P (y n |y <n , a n )an???1 P (a n |y <n , a n???1 )P (a n???1 |y <n ) = n t=1 at P (y t |y <t , a t ) DISPLAYFORM2 The last equality is after applying the same rewrite recursively on P (y <n ).

Thus, we have expressed the joint distribution as a product of factors that apply at each decoding step t while conditioning only on previous outputs and attention.

The term at???1 P (a t |a t???1 , y <t )P (a t???1 |y <t ) = P (a t |y <t ) is the attention at step 't' conditioned on all previous outputs.

For reasons that will soon become clear we call this the prior attention at t and denote as Prior t (a).

We call P (a t???1 |y <t ) = P (a t???1 |y <(t???1) , y t???1 ) as the posterior attention Postr(a t???1 ) since this is the attention distribution after observing the output label at the corresponding step, unlike in prior attention.

We expect this attention to be more accurate than the prior that is computed without knowledge of the output token at that step.

We compute posterior attention at any t using prior attention at t ??? 1 by applying Bayes rule as follows:Postr t (a t ) = P (a t |y <t , y t ) = P (y t |y <t , a t )P (a t |y <t ) P (y t |y <t ) = P (y t |y <t , a t )Prior t (a t ) P (y t |y <t ) (4) DISPLAYFORM3 The above equations give us the important result that the attention at step t should be computed from the posterior attention of the previous step.

Intuitively, also it makes sense because attention reflects an alignment of the input and output, and its distribution will improve if the output is known.

We get into details of computing such coupled attention in Section 2.3.1.

We use the RNN to summarize y <t as a fixed length vector s t as in current ED models.

We then discuss three different methods we explored for computing at???1 P (a t |s t , a t???1 , y t???1 )Postr t???1 (a t???1 ).

Our methods are designed to be light-weight in terms of the number of extra parameters they consume beyond the default soft-attention methods to have the fairest comparison.

Postr-Joint The simplest of these uses the same decoder RNN to absorb the posterior attention of the previous step.

We linearize the function using the first order Taylor expansion to efficiently approximate computation of Prior t (a) similar to the deterministic technique of Xu et al. FORMULA1 Prior t (a t ) = a P (a t |s t???1 , y t???1 , a )Postr t???1 (a ) ??? P (a t |s t???1 , y t???1 , DISPLAYFORM0 The above equation suggests that the decoder RNN state should be updated as s t = RNN(s t???1 , a Postr t???1 (a )x a , y t???1 ).

The computation here is thus similar to existing ED model's but the crucial difference is that the context used to update the RNN is computed from posterior attention, and not the prior attention.

We will see that this leads to large improvement in accuracy.

Proximity biased Next we experiment with models that explicitly couple adjacent attention.

These models utilize an index based coupling between attention positions of the form DISPLAYFORM1 where A ?? (x at , s t ) is the attention logit computed from the previous RNN step, k(a t , a t???1 ) is the attention coupling energy and Z is the normalization constant.

In the proximity based attention the coupling energy k(a t , a t???1 ) is given by I(|a t ??? a t???1 | < 3)?? at???at???1 .

This model provides a greater focus on attending states within a window of size five centered around the recently attended input state.

We label this model as Prox-Postr-Joint in our experiments.

Monotonicity biased This method differs from the above proximity-based attention only in how it defines the coupling energy k(a t , a t???1 ).

As the name implies, in this method k(a t , a t???1 ) is a monotonic energy given by I(a t > a t???1 )?? at???at???1???1 .

This model provides a positive exponentially decaying bias towards encoder states which are to the right of the current attended state, thus influecning attention to be more monotonic.

As we shall see tasks with natural monotonic attention benefit from this form of bias.

This model is denoted as Mono-Postr-Joint in our experiments.

In FIG0 we put together the final set of equations that are used to compute the output distribution and contrast with existing attention model.

We call this overall architecture as Posterior Attention Model (PAM).

First note that in PAM, we explicitly compute a joint distribution of output and attention at each step and marginalize out the attention.

Thus, the output is a mixture of multiple output distributions each of which is a function of one focused input (like in hard attention), and not a diffused sum of the input (like in soft attention).

This difference in the way attention is marginalized is not only statistically sound, but also leads to higher accuracy.

The only downside of the joint model is that we need to compute m softmaxes for each output y t , and this may be impractical when the vocabulary size is large.

A simple and effective fix to this is to select the top-K attentions based on Prior t and compute the final output distribution as.

DISPLAYFORM0 Small values of K (order 6), suffice to provide good performance The second difference is that the attention distribution that is propagated to the next step is posterior to observing the current output.

We derived this from a principled rewrite of the joint distribution, and were pleasantly surprised to see significant accuracy gains by this subtle difference in the way the decoder state is updated.

Computing the posterior attention does not incur any additional overheads because the joint attention-output distribution was already materialized in the first equation.

However, due to the sparsity induced by the top-K operation on attention probabilities, the posterior probabilities are unrealistically sparse.

As such we augment the posterior attention using input from standard attention, by using an equally weighted combination of the two distributions.

Third, the prior attention distribution is explicitly conditioned on the previous attention.

This allowed us to incorporate various application-specific natural biases like proximity and monotonicity of adjacent attentions.

Our rewrite although somewhat unconventional was derived to satisfy two important goals: First, explain the need to propagate posterior attention to subsequent steps.

Second, to factorize the joint as the product of the local distribution at each time t which allows efficient gradient updates and minimal changes to existing beam-search inference.

A more conventional rewrite for handling Markovian dependencies is the standard forward algorithm which works as follows.

First we write: DISPLAYFORM0 p(y t |y <t , a t )p(a t |y <t , a t???1 ).

DISPLAYFORM1 Pr(y t |s t , DISPLAYFORM2 Pr(y|x 1:m ) = n t=1 m a=1 P (y t |s t , x a )Prior t (a) (11) DISPLAYFORM3 Postr t (a) = P (y t |s t , x a )Prior t (a) a P (y t |s t , x a )Prior t (a )Prior t (a t ) = a P (a t |s t???1 , a )Postr t???1 (a ) (14)P (a t |s t???1 , a ) = See Section 2.3.1 (15) Then use the forward algorithm to compute: DISPLAYFORM4 which then gives the joint distribution as P (y) = a ?? n (a).

This expression is neither in the outer product form, nor does it motivate the need for posterior attention.

The de facto standard for sequence to sequence learning via neural networks is the encoder decoder model.

Ever since their first introduction in BID1 , many different attention models have been proposed.

We discuss them here.

Soft Attention is the de-facto mechanism for seq2seq learning et al (2018).

It was proposed for translation in BID1 and refined further in BID14 .

The output derives from an attention averaged context.

The advantage is end to end differentiability.

Hard Attention was proposed in Xu et al. (2015) and attends to exactly one input state for an output.

The merit of hard attention is that the output is determined from a single input rather than an average of all inputs.

Accordingly it has proven useful in when explicit focus is beneficial such as model adaptation and catastrophic forgetting BID19 .

However due to non-differentiability, training Hard-Attention requires the REINFORCE Williams (1992) algorithm and is subject to high variance, requiring careful tricks to train reliably.

Yu et al. (2016) keep the encoder and decoder independent to allow for easier marginalization.

BID0 use a monotonic hard attention and avoid the problem, by supervising hard attention with external alignment information.

Our model in equation 11 uses hard attention on the encoder states.

However unlike standard hard attention we do not use a one-hot attention and instead are computing the exact marginalization.

Sparse/Local Attention Many attempts have been made to bridge the gap between soft and hard attention.

BID14 proposes local attention that averages a window of input.

This has been refined later to include syntax BID3 BID18 and has been explored for image captioning in BID10 .

A related idea to hard attention is to make it sparse using sparsity inducing operators BID15 BID17 .

However, all sparse/local attention methods continue to compute P (y) from an attention weighted sum of inputs like in soft attention.

Yang et al. (2016) have previously modeled relationship between the attentions at different time steps by using a recurrent history mechanism.

The attention history of an input word and its surrounding words are captured in a summary vector by an RNN, which is provided as further input to the attention mechanism for incorporating dependence on history.

While both works model dependence between attention at different steps, our principled rewrite of the joint distribution shows that posterior attention should be the link to the next attention.

Latent Attention Models Our model can be considered as a generalization of the work of to the case where attention is also provided to the RNN.

The model in also factorizes the joint distribution and are identical to our Prior-Joint model.

However unlike these models we explicitly model the posterior attention distribution and attention coupling.

BID6 proposes to learn the posterior attention via variational methods.

A key difference is while their model tries to supervise attention using a posterior inference network via a KL term, we directly use the actual posterior for computing attention in the subsequent steps.

In our method, posterior attention is used in identical roles across training and inference, unlike in BID6 's that rely on variational training.

Structured Attention Networks Similar to this work, BID11 interpret attention as latent structural variable.

The authors then take advantage of easy inference in certain graphical models to implement forms of segmental and syntactic attention.

These works only focus on attention at each step independently whereas our focus is modeling the dependency among adjacent attention.

Moreover our posterior attention framework is independent of how the prior attention at each position is modeled.

In this paper we assumed a multinomial distribution but the structured distribution of BID11 can also benefit from our posterior coupling.

BID16 .

We use a 2 layer bi-directional encoder and 2 layer decoder with 512 LSTM units and 0.2 dropout with vanilla SGD optimizer.

We use word level encoding for all translation tasks.

Our results are in Table 1 where we show perplexity (PPL) and BLEU with beam size 4 and 10.

All Postr-Joint variants and Prior-Joint outperform soft attention and sparse-attention by large margins.

Moreover models with posterior attention show improvement over those which use prior attention.

We also observe small improvements over the more sophisticated and compute-intensive variational attention model likely due to the use of the exact posterior during inference instead.1 This clearly shows the performance advantage of joint modeling and posterior attention.

We shall analyze the reasons for these improvements later.

Next we explore the impact of different coupling models discussed in 2.3.1.

For that focus on methods Postr-Joint, Prox-Postr-Joint, and Mono-Postr-Joint in Table 1 .

We obtain some gains over Postr-Joint by explicitly modeling attention coupling.

For language-pairs with a natural monotonic alignment like German-English, Mono-Postr-Joint slightly outperforms other models by (0.1-0.2 BLEU points).

English-Vietnamese is a more non-monotonic pair and as expected we do not find gains by incorporating a monotonic bias.

To demonstrate the use of our model beyond translation, we next consider the task of generating morphological inflections.

We use inflection forms for German Nouns (de-N) and German Verbs (de-V) from BID7 's. A model is trained separately for each type of inflection to predict the inflected character sequence.

We train a one layer encoder and decoder with 128 hidden LSTM units each with a dropout rate of 0.2 using Adam and measure 0/1 accuracy.

We also ran the 100 units wide two layer LSTM with hard-monotonic attention model BID0 Table 2 : Test accuracy for morphological inflection Using joint modeling we get significant gains (0.3 points) even against task-specific hard-monotonic attention, showing that our approach is more general than translation.

Moreover when we use Mono-Postr-Joint which has a structural bias towards task-specific monotonic attention, we obtain immense improvements (upto 1 accuracy point) over joint models.

We attempt to get more insights on why posterior attention models score over soft attention in end to end accuracy.

We show that the main reason is better alignment of input and output because of a more precise attention model.

We demonstrate that by first showing some anecdotes of better alignment, then showing that posterior attention is more focused (has lower entropy), provides better alignment accuracy, and better input coverage.

For these runs we perform experiments in the teacher forcing setup so as to compare two models' distributions under identical inputs.

Anecdotal Examples Fig2 presents the heatmap of difference between Postr-Joint and SoftAttention on two different sentences.

In each figure the red regions represents where Postr-Joint has greater attention and blue where soft-attention has greater focus.

One can observe that Soft-Attention is far more diffused.

More importantly, we can see that Postr-Joint is able to correct mistakes and provides the appropriate context for the next step.

For example in Fig2a Soft-Attention (blue) has maximum focus on the source word 'generationen' when the target word is innovation which corresponds to 'innovationen'; on the other hand Postr-Joint is able to correct this.

Similarly while producing the phrase 'but the same' Postr-Joint focuses the attention on the source word 'dasselbe' Fig2b.

This provides insight into as to how by providing better contexts via incorporating the target, posterior attention can outperform prior attention.

Attention Entropy Vs Accuracy We expect Soft-Attn to be worse hit by high attention uncertainty than other models.

This, if true, could illustrate that P (y t |x t ) distribution can be learned more easily .

Note the smoother accuracy decay in Postr-Joint and the entropy distibution for Sot-Attention if the input is 'pure', rather than diffused via pre-aggregation.

To this end we plot the accuracy of Postr-Joint, Prior-Joint and Soft-Attn under increasing attention entropy in Figure 3 on the EnglishGerman pair.

As one can expect the accuracy drops off quickly as attention uncertainty rises.

The plot also presents the histogram of the fraction of cases with different attention uncertainties.

Soft attention models (blue) have significantly higher number of cases of high attention uncertainty, leading to low performance.

One of the primary means by which joint models outperformed soft-attention is by reducing the number of such cases.

These figures also provide insight into another mechanism by which posterior attention boosts performance.

One can see that the accuracy drops off much more smoothly wrt attention uncertainty in posterior attention models (green).

In fact in cases of high attention certainty (low attention entropy) Postr-Joint slightly underperforms Prior-Joint, however due to relatively stabler behavior gives better performance overall.

Alignment accuracy Failure of attention to produce latent structures which correspond to linguistic structures has been noted by BID13 BID9 .Based on few examples, we hypothesize that Posterior Attention should be able to produce better alignments.

To test this we used the RWTH German-English dataset which provides alignment information manually tagged by experts, and compare the alignment accuracy for Soft, Prior-Joint and Postr-Joint attentions.

Following the procedure in BID9 the most attended source word for each target word is taken as the aligned word.

We used the AER metric Koehn (2010) to compare these against the expert alignments.

Fraction of covered tokens A natural expectation for translation is that by the time the entire output sentence has been produced, attention would have covered the entire input sequence.

A loss based on this precise heuristic was used in BID5 to improve the performance of a attention based seq2seq model for speech transcription.

In this experiment we try to indirectly assess reliability of different attention models via measuring whether cumulatively attention has focused on the entire input sequence.

We plot the frequency distribution of the coverage in Fig4.

Note that in soft attention model, there are many sentences which do not receive enough attention during the entire decoding process.

Prior-Joint and Postr-Joint have similar behavior with few instances of one outperforming the other, however both outperform soft attention by huge margins.

We show in this paper that none of the existing attention models adequately model the dependence of the output and attention along the length of the output for general sequence prediction tasks.

We propose a factorization of the joint distribution, and develop practical approximations that allows the joint distribution to decompose over output tokens, much like in existing attention.

Our more principled probabilistic joint modeling of the dependency structure leads to three important differences.

First, the output token distribution is obtained by aggregating predictions across all attention.

Second, the concept of conditioning attention on the current output i.e. a posterior attention for inferring the next output becomes important.

Our experiments show that it is sounder, more meaningful and more accurate to condition subsequent attention distribution on the posterior attention.

Thirdly, via directly exposing attention coupling, we have a principled way to directly incorporate task-specific structural biases and prior knowledge into attention.

We experimented with some simple biases and found boosts in related tasks.

Our work opens avenues for future work in scaling these techniques to large-scale models and multi-headed attention.

Another promising line is to incorporate more complex biases like phrasal structure or image segments into joint attention models.

<|TLDR|>

@highlight

Computing attention based on posterior distribution leads to more meaningful attention and better performance

@highlight

This paper proposes a sequence to sequence model where attention is treated as a latent variable, and derives novel inference procedures for this model, obtaining improvements in machine translation and morphological inflection generation tasks.

@highlight

This paper presents a novel posterior attention model for seq2seq problems