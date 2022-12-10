We introduce the masked translation model (MTM) which combines encoding and decoding of sequences within the same model component.

The MTM is based on the idea of masked language modeling and supports both autoregressive and non-autoregressive decoding strategies by simply changing the order of masking.

In experiments on the WMT 2016 Romanian-English task, the MTM shows strong constant-time translation performance, beating all related approaches with comparable complexity.

We also extensively compare various decoding strategies supported by the MTM, as well as several length modeling techniques and training settings.

Neural machine translation (NMT) has been developed under the encoder-decoder framework (Sutskever et al., 2014) with an intermediary attention mechanism (Bahdanau et al., 2015) .

The encoder learns contextualized representations of source tokens, which are used by the decoder to predict target tokens.

These two components have individual roles in the translation process, and they are connected via an encoder-decoder attention layer (Bahdanau et al., 2015) .

Many advances in NMT modeling are based on changes in the internal layer structure (Gehring et al., 2017; Wang et al., 2017; Vaswani et al., 2017; Dehghani et al., 2019; , tweaking the connection between the layers (Zhou et al., 2016; Shen et al., 2018; Bahar et al., 2018; Li et al., 2019a) , or appending extra components or latent variables (Gu et al., 2016; Zhang et al., 2016; Shah & Barber, 2018; ) -all increasing the overall architectural complexity of the model while keeping the encoder and decoder separated.

Our goal is to simplify the general architecture of machine translation models.

For this purpose, we propose the masked translation model (MTM) -a unified model which fulfills the role of both the encoder and decoder within a single component.

The MTM gets rid of the conventional decoder as well as the encoder-decoder attention mechanism.

Its architecture is only a sequence encoder with self-attention layers, trained with an objective function similar to masked language modeling (Devlin et al., 2019) .

In order to model the translation problem, the MTM is given the concatenation of the source and target side from a parallel sentence pair.

This approach is similar to the translation language model presented by Lample & Conneau (2019) , but focuses on the target side, i.e. the masking is applied to some selected positions in the target sentence.

The MTM is trained to predict the masked target words relying on self-attention layers which consider both the source sentence and a masked version of the target sentence.

Trained in this way, the model is perfectly suitable for non-autoregressive decoding since the model learned to predict every position in parallel, removing the dependency on decisions at preceding target positions.

Within its extremely simple architecture, one can realize various decoding strategies, e.g., using left-to-right, non-autoregressive, or iterative decoding by merely adjusting the masking schemes in search.

We present a unified formulation of the MTM for different decoding concepts by factorizing the model probability over a set of masked positions.

The MTM has several advantages over the conventional encoder-decoder framework:

• A simpler architecture

• A variety of decoding strategies including constant-time approaches (Section 3.3.1)

On the WMT 2016 Romanian→English translation task, our MTM achieves a better performance than comparable non-autoregressive/constant-time methods while keeping its simple architecture.

Using our general formulation of the MTM, we compare the translation performance of various decoding strategies.

Moreover, we show that this model allows for decoding speed-up by merely adjusting the number of iterations at the small cost of translation performance.

There have been some attempts to combine the encoder and decoder into a single component for simplified translation modeling.

share the encoder and decoder parameters of a Transformer translation model (Vaswani et al., 2017) and allow the encoder-decoder attention to access the inner layers of the encoder as well.

Fonollosa et al. (2019) extend this idea by adding locality constraints in all attention layers.

Radford et al. (2019) train a Transformer decoder on a large monolingual corpus as a language model and use it as an unsupervised translation model on pairs of source and target sentences.

Similarly to our work, all these approaches couple the encoding and decoding on the self-attention level.

However, their decoding considers only left-side target context, enabling only left-to-right autoregressive translation.

Furthermore, their encoding of a source sentence is limited to the source side itself, while our MTM can refine the source representations according to a partial target hypothesis represented in the decoder states.

Both aspects hinder their methods from making the best use of the bidirectional representation power of the combined model.

Non-autoregressive NMT, which predicts all target words in parallel, potentially exploits full bidirectional context in decoding.

To make the parallel decoding produce a reasonable hypothesis, Gu et al. (2018) reuse the source words as inputs to the decoder and insert an additional attention module on the positional embeddings.

Lee et al. (2018) use a separate decoder to revise the target hypotheses iteratively, where Ghazvininejad et al. (2019) train a single decoder with MLM objectives for both the first prediction and its refinements.

To improve the integrity of the hypotheses, one could also employ an autoregressive teacher to guide the states of the non-autoregressive decoder Li et al., 2019b) , apply sentence-level rewards in training (Shao et al., 2019) , or integrate generative flow latent variables (Ma et al., 2019) .

The self-attention layers of their decoders attend to all target positions, including past and future contexts.

However, all these methods still rely on the encoder-decoder framework.

In this work, we collapse the boundary of the encoding and decoding of sequences and realize non-autoregressive NMT with a unified model.

Regardless of the encoding or decoding, the self-attention layers of our MTM attend to all available source and target words for flexible information flow and a model of simplicity.

A common problem in non-autoregressive sequence generation is that the length of an output should be predefined beforehand.

The problem has been addressed by averaging length difference (Li et al., 2019b) , estimating fertility (Gu et al., 2018) , dynamically inserting blank outputs via connectionist temporal classification (CTC) (Libovickỳ & Helcl, 2018) , or directly predicting the total length from the encoder representations (Lee et al., 2018; Ghazvininejad et al., 2019; Ma et al., 2019) .

In this work, we train a separate, compact length model on given bilingual data.

The training of an MTM is based on the MLM objective (Devlin et al., 2019) , developed for pretraining representations for natural language understanding tasks .

Lample & Conneau (2019) concatenate source and target sentences and use them together as the input to an MLM, where both source and target tokens are randomly masked for the training.

This improves cross-lingual natural language inference in combination with the original MLM objective, but has not been applied to translation tasks.

We use the concatenated input sentences but selectively mask out only target tokens to implement source→target translation.

As for inference with an MLM, Wang & Cho (2019) and Ghazvininejad et al. (2019) propose to build up an output sequence iteratively by adjusting the input masking for each iteration.

In this work, the generation procedures of both works are tested and compared within our MTM, along with other autoregressive/non-autoregressive decoding strategies (Section 4.1).

We introduce the masked translation model (MTM) focusing on three aspects: model architecture, training, and decoding.

In the corresponding sections, we show how the MTM 1) relaxes the con- ...

...

...

ventional architectural constraint of the encoder-decoder framework, 2) learns to perform both nonautoregressive translation and its refinements, and 3) does translation with various decoding strategies in a variable number of iterations.

Given a source sentence f

.., f J (source input) the goal of an MTM p θ is to generate the correct translation e I 1 = e 1 , ..., e i , ..., e I (target output) by modeling:

1 , e I i+1 ) (target input) is a corrupted version -a subset -of the surrounding context in the target sentence e I 1 .

Therefore, the MTM models the true target sentence e I 1 independently for each position, given both the true source and a noisy version of the target context.

Figure 1 illustrates the MTM network architecture which is the same as the MLM presented by Devlin et al. (2019) or the encoder of a Transformer network (Vaswani et al., 2017) with an output softmax layer on top of all corrupted positions.

In particular, an MLM consists of N transformer encoder layers each containing two blocks: a self-attention layer with full bidirectional context as well as two linear layers with a RELU activation in between.

Layer normalization is applied before every block, and a residual connection is used to combine the output of a block with its input (Vaswani et al., 2017) .

The source f , whose space is shared over the source and target languages by a joint subword vocabulary.

A positional encoding vector (Vaswani et al., 2017 ) is added where the position index is reset at the first target token.

Similarly to and Lample & Conneau (2019) , we add language embeddings to distinguish source and target vocabularies efficiently.

The embedded representations pass through N transformer encoder layers, where the attention has no direction constraints; this allows the model to use the full bidirectional context beyond the language boundary.

Note that the hidden representations of source words can also attend to those of (partially hypothesized) target tokens, which is impossible in the encoder-decoder architecture.

During training the target side inputẽ I 1 may be 1) fully masked out, resembling the initial stage of translation before hypothesizing any target words, 2) partially corrupted, simulating intermediate hypotheses of the translation process that need to be corrected, 3) or even the original target sentence e I 1 , representing the final stage of translation which should not be refined further.

The different levels of corruptions are used to model all plausible cases which we encounter in the decoding process -from primitive hypotheses to high-quality translations (Section 3.3).

Given bilingual training data D = {(f Note that a model for length prediction is trained separately and the details of which can be found Appendix A.

We cannot expect a single step of parallel decoding to output an optimal translation from only the source context.

Therefore, we consider the second scenario of refining the hypothesis, where we simulate a partial hypothesis in training by artificially corrupting the given target sentence in the input.

For training an MTM, we formulate this corruption by probabilistic models to 1) select the positions to be corrupted (p s ) and 2) decide how such a position is corrupted (p c ).

The goal of the MTM is now to reconstruct the original target sentence e I 1 .

This leads to the training loss:

where C is a set of positions to be corrupted.

The corrupted target inputẽ I 1 is generated in two steps of random decisions (Devlin et al., 2019 ):

1.

Target positions i ∈ C for the corruption are sampled from a uniform distribution until ρ s · I samples are drawn, with hyperparameter ρ s ∈ [0, 1].

We denote this selection process by C ∼ p s (C|e I 1 ) = p s (C|I) as a simplified notation.

2.

The specific type of corruption for each selected position i ∈ C is chosen independently by samplingẽ i from p c (ẽ i |e i ).

Note that we train the model to reconstruct the original token e i only for the corrupted positions i ∈ C. For the remaining positions i / ∈ C, the corrupted sentence is filled with the original word, i.e.ẽ i = e i .

These uncorrupted positions provide the context to the network for the denoising, and no loss is applied for these positions to prevent a bias towards copying.

We optimize this criterion in a stochastic way, where C and theẽ i are sampled anew for each epoch and each training instance.

In principle, the MTM is a denoising autoencoder of the target sentence conditioned on a source sentence.

The MTM training can be customized by selecting p s and p c appropriately.

Following Devlin et al. (2019) , the probability p s is defined as a uniform distribution over all target positions, without considering the content of the sentence.

For the corruption model p c , we define a set of operations and assign a probability mass ρ o ∈ [0, 1] to each operation o.

We use the three operations presented by Devlin et al. (2019) :

• Replace with a mask token:

• Replace with a random word e * uniformly sampled from the target vocabulary V e :

Random:

• Keep unchanged:

Original: e Figure 2 shows an example of corrupting a target input sentence in the MTM training.

Here all positions except 1 are corrupted, i.e. C = {2, 3, 4}. At Position 2 the original word is kept by the Keep operation of p c but in contrast to Position 1 (No Operation)

there is a training loss added for Position 2.

3.3 DECODING As described above, the MTM is designed and trained to deal with intermediate hypotheses of varying quality as target input.

Accordingly, decoding with an MTM consists of multiple iterations τ = 1, ..., T : A non-autoregressive generation of the initial hypothesis (for τ = 1) and several steps of iterative refinements (inspired by Lee et al. (2018) ) of the hypothesis (for τ > 1).

In the context of this work, an iteration of the MTM during decoding refers to one forward pass of the model based on a given source and target input, followed by the selection of a target output based on the predictions of the MTM.

To simplify the notation, we denote the given source sentence by F = f

and the generated target translation byÊ =ê I 1 .

Similar to traditional translation models, the goal of decoding with an MTM is to find a hypothesisÊ that maximizes the translation probability:

{p(E, I|F )} = arg max

We approximate this by a two stage maximization process where we first determine the most likely target lengthÎ := arg max I {p(I|F )} followed by:

Instead of a left-to-right factorization of the target sentence which is common in autoregressive decoding, we perform a step-wise optimization on the whole sequence.

For this we define the sequenceÊ (τ ) starting fromÊ (0) :=</M>,...

,</M> by selecting the best hypothesis given the predecessorÊ (τ −1) , i.e.:Ê (τ ) := arg max

Introducing an intermediate representationẼ allows for a more fine-grained control of the decoding process:

where the probability p θ is modeled by a neural network with parameters θ and the masked sequencẽ

is modelled by p m , which defines a specific search strategy (Section 3.3.1).

In most scenarios, the search strategy is defined to be deterministic, which has the effect that all probability mass of p m is concentrated on one masked sequenceẼ (τ −1) and we can reformulate Equation (8) as:

and thus the score is defined solely by the MTM network.

The iterative procedure presented in Equation (10) describes a greedy optimization, as it selects the currently best hypothesis in each iteration.

This does not provide any guarantee for the quality of

Sample a masked sequence: 1) ) Compute the model output greedily for every position: e

the final output.

However, if we assume that the model score improves in each iteration, i.e. if we can show that:

then we know that the maximum score is obtained in the last iteration T .

To the best of our knowledge, it is not possible to provide a theoretical proof for this property, yet we will show empirical evidence that it holds in practice (see Section 4.1).

Thus, in order to find the best hypothesis, it is sufficient to follow the recursive definition presented in Equation (10), which can be computed straight forward resulting in an iterative decoding scheme.

Algorithm 1 describes this process of MTM decoding.

In short, 1) generate a hypothesis (Ê), 2) select positions to be masked, and 3) feed the masked hypothesis (Ẽ) back to the next iteration.

Note that the output for each position e (τ ) i is computed in the same forward pass without a dependency on other words e (τ ) i (i = i) from the same decoding step.

This means that the first iteration (τ = 1) is non-autoregressive decoding (Gu et al., 2018; Libovickỳ & Helcl, 2018) .

Nonautoregressive models tend to suffer from the multimodality problem (Gu et al., 2018) , where conditionally independent models are inadequate to capture the highly multimodal distribution of target translations.

Our MTM decoding prevents this problem making iterative predictionsÊ (τ ) each conditioned on the previous sequence.

EachÊ (τ ) results from one forward pass, yielding a complexity of O(T ) decoding steps instead of O(I) as in traditional autoregressive NMT.

Thus the MTM decoding can potentially be much faster than conditional decoding of a standard encoder-decoder NMT model, as the number of iterations is not dictated by the target output length I. Furthermore, compared to the pure non-autoregressive decoding with only one iteration, our decoding algorithm may collapse the multimodal distribution of the target translation by conditioning on the previous output (Lee et al., 2018) .

The masking probability p m introduced in Equation (9) resembles the two-step corruption of the training (Equation (2)):

where C is a set of positions to be masked.

Similarly to the training, the corruption is performed only for i ∈ C (τ ) and the remaining positions i / ∈ C (τ ) are kept untouched.

For the corruption model p c in decoding, only the Mask operation is activated, i.e. ρ mask = 1 and ρ o = 0 for o = mask.

This leads to the following simple decisions:

The resulting masked sequenceẼ (τ ) is supposed to shift the model's focus towards a selected number of words, chosen by the decoding strategy p s .

Given this definition of p c above, a masked (intermediate) hypothesis in decoding is determined solely by the position selection p s , which differs by decoding strategy.

Each decoding strategy starts from a fully masked target input, i.e. C (0) = {1, ..., I}, and uncovers positions incrementally in each iteration.

The simplest solution is to simply feed back the completely unmasked sequence in each iteration (Lee et al., 2018) :

This method works with the richest context from the output of the previous iteration.

This may, however, hurt the model's performance as the first output is often quite poor, and the focus of the model is spread across the whole sentence.

Instead of unmasking all positions from the beginning, one can unmask the sequence randomly one position at a time (Wang & Cho, 2019) , inspired by Gibbs sampling (Geman & Geman, 1984) :

Note that this method is nondeterministic and it takes at least I iterations before the output is conditioned on the completely unmasked sequence.

A deterministic alternative to the random strategy is to unveil the sequence step-wise, starting from the left-most position in the target sequence.

In every decoding iteration τ = 1, ..., T , the index i = τ − 1 is removed from the set of masked positions:

This corresponds to the traditional autoregressive NMT, but the parallel nature of our MTM decoding inherently enables to update the prediction for any position at any time, e.g. the prediction for the first position can change in the last iteration.

Furthermore, it allows for different step-wise strategies -revealing the sequence right-to-left (R2L) or starting from the middle in both directions (middleout) -without re-training the model.

L2R decoding ignores a huge advantage of the MTM, which is the property that the fully bidirectional model can predict sequence elements in any given order.

This characteristic is leveraged by masking a decreasing number of K(τ ) positions in each iteration:

At each iteration, K(τ ) positions with the lowest model score p θ (confidence) remain masked:

where the number of masked positions K(τ ) is chosen to be linearly decreasing over the number of iterations T (Ghazvininejad et al., 2019) :

One can also unmask positions one by one, i.e. K(τ ) = I − τ , sacrificing potential improvements in decoding speed.

Fertility-based (Gu et al., 2018) 29.1 -CTC (Libovickỳ & Helcl, 2018) 24.7 -Imitation learning 28.9 -Reinforcement learning (Shao et al., 2019) 27.9 -Generative flow (Ma et al., 2019) 30.

We implemented the MTM in the RETURNN framework (Doetsch et al., 2017) and evaluate the performance on the WMT 2016 Romanian→English translation task 1 .

All data used in the experiments are preprocessed using the MOSES (Koehn et al., 2007) tokenizer and frequent-casing (Vilar et al., 2010) .

We learn a joint source/target byte pair encoding (BPE) (Sennrich et al., 2016) with 20k merge operations on the bilingual training data.

Unless mentioned otherwise, we report results on the newstest2016 test set, computed with case sensitivity and tokenization using the software SacreBLEU 2 (Post, 2018) .

The MTMs in our experiments follow the base configuration of Vaswani et al. (2017) , however, with a depth of 12 layers and 16 attention heads.

They are trained using Adam (Kingma & Ba, 2015) with an initial learning rate of 0.0001 and a batch size of 7,200 tokens.

Dropout is applied with a probability of 0.1 to all hidden layers and word embeddings.

We set a checkpoint after every 400k sentence pairs of training data and reduce the learning rate by a factor of 0.7 whenever perplexity on the development set (newsdev2016) does not improve for nine consecutive checkpoints.

The final models are selected after 200 checkpoints based on the development set perplexity.

During training, we select a certain percentage ρ s of random target tokens to be corrupted.

This parameter is selected randomly from a uniform distribution ρ s ∼ U[0, 1] in every training step.

We further deviate from Devlin et al. (2019) , by selecting the hyperparameters for corruption to be ρ mask = 0.6, ρ rand = 0.3, and ρ keep = 0.1, which performed best for MTMs in our preliminary experiments.

The main results of the MTM are given in Table 1 along with comparative baselines.

In total, our MTM, despite its extremely simple architecture, outperforms comparable constant-time NMT methods which do not depend on sequence lengths.

Compared to the conventional autoregressive baseline, the MTM falls behind by only -2.4% BLEU with a constant number of decoding steps and the lower model complexity.

Furthermore, a control experiment using the gold length instead of a predicted length improves the results from the same MTM model by 1.5% BLEU.

This result minimizes the gap between our MTM and a comparable encoder-decoder model down to 0.9% BLEU, while our model has the ability to improve the decoding speed without retraining by simply reducing the number of iterations, thus trading in performance against speed.

Note that all other methods shown in Table 1 are based on the encoder-decoder architecture, which is more sophisticated.

Moreover, the performance of Gu et al. (2018) , Lee et al. (2018) , and Shao et al. (2019) relies heavily on knowledge distillation from a well-trained autoregressive model, which involves building an additional NMT model and translating the entire training data with that model.

This causes a lot more effort and computation time in training, while the MTM requires no such complicated steps, and its training is entirely end-to-end.

et al. (2019) demonstrate in their work that even better results could be possible by computing multiple translations in parallel for a set of most likely length hypotheses.

This approach or even a beam-search variant of our iterative unmasking, will be a focus in our future work.

As described in Section 3.3.1, the flexibility of the MTM allows us to easily implement different decoding strategies within a single model.

Pure non-autoregressive decoding, i.e., a single forward pass to predict all target positions simultaneously, yields poor translation performance of 13.8% BLEU on the validation set (newsdev2016), which implies that several decoding iterations are needed to produce a good hypothesis.

We know this to be true if the inequality in Equation (11) holds, i.e. if we see our model score improving in every iteration.

Figure 3 shows that we can actually observe this property by monitoring the average model score throughout all iterations.

Outputs for individual sentences might still worsen between two iterations.

The overall score, however, shows a steady improvement in each iteration.

In Figure 4 , we compare various decoding strategies and plot their performance for different number of decoding iterations T . "Fully unmasking", i.e. re-predicting all positions based on the previous hypothesis, improves the hypothesis fast in the early iterations but stagnates at 22% BLEU.

L2R, R2L, and confidence-based one-by-one all unmask one position at a time and show a very similar tendency with confidence-based one-by-one decoding reaching the strongest final performance of 31.9% BLEU.

Confidence-based fixed-T unmasks several target positions per time step and achieves similar performance.

In contrast to position-wise unmasking, the decoding with a fixed number of T (and linear unmasking) only needs ten decoding iterations to reach close to optimal performance.

We test "middle-out" a variation of the L2R strategy to see whether the generation order is negligible as long as the sentence is generated contiguously.

While this improves the hypothesis faster -most likely due to its doubled rate of unmasking -the final performance is worse than those of L2R or R2L decoding.

Random selection of the decoding positions shows comparably fast improvements up to the 10th iterations, keeping up with middle-out, even though it reveals the hypothesis for a single position per iteration, however its performance saturates below most other decoding strategies.

Overall the best result for a low iteration count is obtained with confidence-based decoding with a fixed number of iterations.

This shows that it is possible and sometimes even beneficial to hypothesize several positions simultaneously.

We conclude that the choice of the decoding strategy has substantial impacts on the performance and hypothesize that a good decoding strategy relies on the model score to choose which target positions should be unmasked.

In this work we simplify the existing Transformer architecture by combining the traditional encoder and decoder elements into a single component.

The resulting masked translation model is trained by concatenating source and target and applying BERT-style masking to the target sentence.

The novel training strategy introduced with the MTM requires a rethinking of the search process and allows for various new decoding strategies to be applied in the theoretical framework we developed in this work.

A detailed comparison shows that unmasking the sequence one-by-one gives the overall best performance, be it left-to-right, right-to-left, or confidence-based.

Unveiling a constant number of tokens based on confidence in each decoding step, however, can achieve reasonable performance with a fixed, much smaller number of iterations.

We show that there is a potential of at least 1.5 % BLEU improvement that can be achieved by more elaborate length models, which yields itself as a good start for further research.

Furthermore, we plan to extend the decoding strategies to work with beam search and verify our observations on further language pairs.

In this section we present several ablation studies as well as a deeper look into the decoding to further investigate the MTM.

Autoregressive models determine the output length by stopping the generation process once a special token that marks the end of the sentence (e.g. '</S>') is predicted.

This approach is not compatible with the MTM decoding as target tokens for all positions are predicted in parallel.

Therefore we assume a given output length I and a train length model p(I|f J 1 ) on the bilingual training data.

In training, the true length is used, and in decoding, we choose arg max I p(I|f • Count-based Table.

For the unseen source lengths in training, we assume I = J.

• Poisson Distribution: a more smoothly parametrized model.

For J's not appearing in the training data, we back off to a global distribution with the parameter λ that is learned via maximum likelihood overall source/target length pairs in the training data, i.e.

• Recurrent Neural Network (RNN):

We take the last hidden state as the input to a target length classifier, i.e. a linear projection with a softmax layer over the possible target lengths I ∈ {1, 2, ..., 200}.

• Bidirectional RNN: a variation of the above which employs a BiLSTM and uses the last hidden state of the forward and the backward LSTM for length prediction.

Table 2 verifies that the translation performance depends highly on the length model.

We also report the averaged absolute difference of the predicted length and the reference length.

The simplest count-based model shows the worst performance, where the target length is merely determined by a frequency table.

The Poisson distribution models the length more systematically, giving +1.1% BLEU against the count-based model.

RNN models consider not only the source length but also the whole sentence semantics, and slightly outperform the Poisson distribution.

The bidirectional RNN model predicts the target length even better, but the translation performance only improves marginally.

Note that using the Reference length improves even further by +1.6% BLEU over our strongest system, which shows that a good length model is a crucial component of the MTM system.

As discussed earlier, the MTM architecture is equivalent to a traditional transformer encoder.

Nevertheless, the way it is applied to the task at hand differs very much, even compared to an MLM.

Thus it was crucial to do a thorough hyperparameter search to obtain an optimal model performance.

The baseline MTM configuration we present here is already the product of many preliminary experiments, setting the number of attention heads h = 16, dropout P drop = 0.1, and the learning rate reduction scheme.

The ablation study presented in table 3 highlights the importance of both the number of attention heads and especially dropout.

It also shows that it was crucial to increase the model depth N compared to the standard transformer encoder, by matching the total number of layers N = 12 as they are used in an encoder-decoder model.

In this section, we show a detailed derivation of the decoding process which justifies our iterative optimization procedure and modularizes the unmasking procedure to apply the application of various decoding strategies.

Assuming we have a hypothesized target lengthÎ, the goal is to find an optimal target sequenceÊ given lengthÎ and source sentence F :

The MTM decoding to find such a sequence is performed in T iterations, whose intermediate hypotheses are introduced as latent variables E (1) , . . .

, E (T −1) :

= arg max

where the last iteration should provide the final prediction, i.e. E (T ) := E. Applying the chain rule and a first-order Markov assumption on E (τ ) yields:

= arg max

with E 0 := </M>, . . . , </M> a sequence of lengthÎ. In a next step, we approximate the sum by a maximization and subsequently apply a logarithm to get:

To simplify further derivations, we introduce the score function Q here and do another approximation by considering only the score Q from a single maximum timestep instead of the full sum over τ = 1, . . .

, T .

≈ arg max

Even with this approximation, we are still trying to find a value for each E (τ ) that optimizes the score of another iterationτ via the connection of dependencies in Q. As this is impractical to compute, we alleviate the problem by focusing on a step-wise maximization.

For this we define the sequenceÊ (τ ) (withÊ (0) := E (0) ) as the best hypothesis of iteration τ givenÊ (τ −1) , i.e.: −1) , F,Î) = arg max

If we useÊ (τ −1) instead of E (τ −1) as the dependency in Q, we restrict the optimization to maximizing each step independently, given its predecessors optimum:

Ideally, this iterative procedure should improve the score Q in each iteration, i.e.:

Q(E (τ ) |Ê (τ −1) , F,Î) ∀τ = 1, . . . , T − 1

While Equation (27) is not true for the general case we observe empirically that this the statement is true on average (see Figure 3 ).

This means that the maximum score to be obtained in the last iteration T :Ê (F,Î) ≈ arg max

which can be simplified to:

= arg max

Q(E (T ) |Ê (T −1) , F,Î) =Ê

For completeness we report the strongest result for each decoding strategy from Figure 4 in Table 4 .

<|TLDR|>

@highlight

We use a transformer encoder to do translation by training it in the style of a masked translation model.