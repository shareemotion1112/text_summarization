A distinct commonality between HMMs and RNNs is that they both learn hidden representations for sequential data.

In addition, it has been noted that the backward computation  of  the  Baum-Welch  algorithm  for  HMMs  is  a  special  case  of  the back propagation algorithm used for neural networks (Eisner (2016)).

Do these observations  suggest  that,  despite  their  many apparent  differences,  HMMs  are a special case of RNNs?

In this paper,  we investigate a series of architectural transformations between HMMs and RNNs, both through theoretical derivations and empirical hybridization, to answer this question.

In particular, we investigate three key design factors—independence assumptions between the hidden states and the observation, the placement of softmax, and the use of non-linearity—in order to pin down their empirical effects.

We present a comprehensive empirical study to provide insights on the interplay between expressivity and interpretability with respect to language modeling and parts-of-speech induction.

Sequence is a common structure among many forms of naturally occurring data, including speech, text, video, and DNA.

As such, sequence modeling has long been a core research problem across several fields of machine learning and AI.

By far the most widely used approach for decades is the Hidden Markov Models of BID1 ; BID10 , which assumes a sequence of discrete latent variables to generate a sequence of observed variables.

When the latent variables are unobserved, unsupervised training of HMMs can be performed via the Baum-Welch algorithm (which, in turn, is based on the forward-backward algorithm), as a special case of ExpectationMaximization (EM) BID4 ).

Importantly, the discrete nature of the latent variables has the benefit of interpretability, as they recover contextual clustering of the output variables.

In contrast, Recurrent Neural Networks (RNNs), introduced later in the form of BID11 and BID6 networks, assume continuous latent representations.

Notably, unlike the hidden states of HMMs, there is no probabilistic interpretation of the hidden states of RNNs, regardless of their many different architectural variants (e.g. LSTMs of BID9 , GRUs of BID3 and RANs of BID13 ).Despite their many apparent differences, both HMMs and RNNs model hidden representations for sequential data.

At the heart of both models are: a state at time t, a transition function f : h t−1 → h t in latent space, and an emission function g : h t → x t .

In addition, it has been noted that the backward computation in the Baum-Welch algorithm is a special case of back-propagation for neural networks BID5 ).

Therefore, a natural question arises as to the fundamental relationship between HMMs and RNNs.

Might HMMs be a special case of RNNs?In this paper, we investigate a series of architectural transformations between HMMs and RNNsboth through theoretical derivations and empirical hybridization.

In particular, we demonstrate that the forward marginal inference for an HMM-accumulating forward probabilities to compute the marginal emission and hidden state distributions at each time step-can be reformulated as equations for computing an RNN cell.

In addition, we investigate three key design factors-independence assumptions between the hidden states and the observation, the placement of soft-max, and the use of non-linearity-in order to pin down their empirical effects.

Above each of the models we indicate the type of transition and emission cells used.

H for HMM, R for RNN/Elman and F is a novel Fusion defined in §3.3.

It is particularly important to understanding this work to track when a vector is a distribution (resides in a simplex) versus in the unit cube (e.g. after a sigmoid non-linearity).

These cases are indicated by c i and c i , respectively.

Our work is supported by several earlier works such as BID23 and BID25 that have also noted the connection between RNNs and HMMs (see §7 for more detailed discussion).

Our contribution is to provide the first thorough theoretical investigation into the model variants, carefully controlling for every design choices, along with comprehensive empirical analysis over the spectrum of possible hybridization between HMMs and RNNs.

We find that the key elements to better performance of the HMMs are the use of a sigmoid instead of softmax linearity in the recurrent cell, and the use of an unnormalized output distribution matrix in the emission computation.

On the other hand, multiplicative integration of the previous hidden state and input embedding, and intermediate normalizations in the cell computation are less consequential.

We also find that HMM outperforms other RNNs variants for unsupervised prediction of the next POS tag, demonstrating the advantages of discrete bottlenecks for increased interpretability.

The rest of the paper is structured as follows.

First, we present in §2 the derivation of HMM marginal inference as a special case of RNN computation.

Next in §3, we explore a gradual transformation of HMMs into RNNs.

In §4, we present the reverse transformation of Elman RNNs back to HMMs.

Finally, building on these continua, we provide empirical analysis in §5 and §6 to pin point the empirical effects of varying design choices over the possible hybridization between HMMs and RNNs.

We discuss related work in §7 and conclude in §8.

We start by defining HMMs as sequence models, together with the forward-backward algorithm which is used for inference.

Then we show that, by rewriting the forward algorithm, the computation can be viewed as updating a hidden state at each time step by feeding the previous word prediction, and then computing the next word distribution, similar to the way RNNs are structured.

The resulting architecture corresponds to the first cell in FIG0 .

(1:n) = {x (1) , . . . , x (n) } be a sequence of random variables, where each x is drawn from a vocabulary V of size v, and an instance x is represented as an integer w or a one-hot vector e (w) , where w corresponds to an index in V.1 We also define a corresponding sequence of hidden variables h(1:n) = {h (1) , . . .

, h (n) }, where h ∈ {1, 2, . . .

m}. The distribution P (x) is defined by marginalizing over h, and factorizes as follows: DISPLAYFORM0 We define the hidden state distribution, referred to as the transition distribution, as DISPLAYFORM1 and the emission (output) distribution as DISPLAYFORM2

Inference for HMMs (marginalizing over the hidden states to compute the observed sequence probabilities) is performed with the forward-backward algorithm.

The backward algorithm is equivalent to automatically differentiating the forward algorithm BID5 .

Therefore, while traditional HMM implementations had to implement both the forward and backward algorithm, and train the model with the EM algorithm, we only implement the forward algorithm in standard deep learning software, and perform end-to-end minibatched SGD training, efficiently parallelized on the GPU.Let w = {w (1) , . . .

, w (n) } be the observed sequence, and w (i) the one-hot representation of w (i) .

The forward probabilities a are defined recurrently (i.e., sequentially recursively) as DISPLAYFORM0 DISPLAYFORM1 This can be rewritten by defining DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 and substituting a, so that equation 6 is rewritten as (left below) or if expressed directly in terms of the parameters used to define the distributions with vectorized computations (right below): DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 Here w (i) used as a one-hot vector, and the bias vectors b and d are omitted for clarity.

Note that the computation of s (i) can be delayed until time step i + 1.

The computation step can therefore be rewritten to let c be the recurrent vector (equivalent logspace formulations presented on the right): DISPLAYFORM9 , DISPLAYFORM10 (15)This can be viewed as a step of a recurrent neural network with tied input and output embeddings: Equation 14 embeds the previous prediction, equations 15 and 16, the transition step, updates the hidden state c, corresponding to the cell of a RNN, and equations 17 and 18, the emission step, computes the output next word probability.

We can now compare this formulation against the definition of a Elman RNN with tied embeddings and a sigmoid non-linearity.

These equations correspond to the first and last cells in FIG0 .

The Elman RNN has the same parameters, except for an additional input matrix U ∈ R m×m .

FIG0 .

We will evaluate these model variants empirically, and also investigate their interpretability.

DISPLAYFORM11

By relaxing the independence assumption of the HMM transition probability distribution we can increase the expressiveness of the HMM "cell" by modelling more complex interactions between the fed word and the hidden state.

Following Tran et al. FORMULA0 we define the transition distribution as DISPLAYFORM0 where W ∈ R m×m×m , B ∈ R m×m .

As the tensor-based methods increases the number of parameters considerably, we also propose an additive version: DISPLAYFORM0 DISPLAYFORM1 Gating-based feeding: Finally we propose a more expressive model where interaction is controlled via a gating mechanism and the feeding step uses unnormalized embeddings (this does not violate the HMM factorization): DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 2 where normalize(y) = DISPLAYFORM5

Another way to make HMMs more expressive is to relax their independence assumptions through delaying when vectors are normalized to probability distributions by applying the softmax function.

The computation of the recurrent vector c (i) = P (h (i) |x (1:i−1) ) is replaced with DISPLAYFORM0 Both c and s are still valid probability distributions, but the independence assumption in the distribution over h (i) no longer holds.

A further transformation is to delay the emission softmax until after multiplication with the hidden vector.

This effectively replaces the HMM's emission computation with that of the RNN: DISPLAYFORM0 This formulation breaks the independence assumption that the output distribution is only conditioned on the hidden state assignment.

Instead it can be viewed as taking the expectation over the (unnormalized) embeddings with respect to the state distribution c, then softmaxed (H R in FIG0 .

We can go further towards RNNs and replace the softmax in the transition by a sigmoid non-linearity.

The sigmoid is placed in the same position as the delayed softmax.

The recurrent state c is no longer a distribution so the output has to be renormalized so the emission still computes a distribution: DISPLAYFORM0 DISPLAYFORM1 This model could also be combined with a delayed emission softmax -which we'll see makes it closer to an Elman RNN.

This model is indicated as F for fusion in FIG0 4 TRANSFORMING AN RNN TOWARDS AN HMM Analogously to making the HMM more similar to Elman RNNs, we can make Elman networks more similar to HMMs.

Examples of these transformations can be seen in the last 2 cells in FIG0 .

First, we use the Elman cell with an HMM emission.

This requires the hidden state be a distribution, thus we consider two options.

One is to replace the sigmoid non-linearity with the softmax function: DISPLAYFORM0 DISPLAYFORM1 This model is depicted as R H in FIG0 .

The second formulation is to keep the sigmoid nonlinearity, but normalize the hidden state output in the emission computation: DISPLAYFORM2 DISPLAYFORM3

In the HMM cell, the integration of the previous recurrent state and the input embedding is modelled through an element-wise product instead of adding affine transformations of the two vectors.

We can modify the Elman cell to do a similar multiplicative integration: DISPLAYFORM0 Or, using a single transformation matrix: DISPLAYFORM1

Finally, and most extreme, we experiment with replacing the sigmoid non-linearity with a softmax: DISPLAYFORM0 And a more flexible variant, where the softmax is applied only to compute the emission distribution, while the sigmoid non-linearity is still applied to recurrent state: DISPLAYFORM1 DISPLAYFORM2

Our formulations investigate a series of small architectural changes to HMMs and Elman cells.

In particular, these changes raise questions about the expressivity and importance of (1) normalization within the recurrence and (2) independence assumptions during emission.

In this section, we analyze the effects of these changes quantitatively via a standard language modeling benchmark.

We follow the standard PTB language modeling setup BID2 BID16 .

We work with one-layer models to enable a direct comparison between RNNs and HMMs and a budget of 10 million parameters (typically corresponding to hidden state sizes of around 900).

Models are trained with batched backpropagation through time (35 steps).

Input and output embeddings are tied in all models.

Models are optimized with a grid search over optimizer parameters for two strategies: SGD 4 and AMSProp.

AMSProp is based on the optimization setup proposed by BID14 .

We see from the results in TAB4 (also depicted in Figure 2 ) that the HMM models perform significantly worse than the Elman network, as expected.

Interestingly, many of the HMM variants that

Figure 2: This plot shows how perplexities change under our transformations, and which lead the models to converge and pass each other.in principle have more expressivity or weaker independence assumptions do not perform better than the vanilla HMM.

This includes delaying the transition or emission softmax, and most of the feeding models.

The exception is the gated feeding model, which does substantially better, showing that gating is an effective way of incorporating more context into the transition matrix.

Using a sigmoid non-linearity before the output of the HMM cell (instead of a softmax) does improve performance (by 44 ppl), and combining that with delaying the emission softmax gives a substantial improvement (almost another 100 ppl), making it much closer to some of the RNN variants.

We also evaluate variants of Elman RNNs: Just replacing the sigmoid non-linearity with the softmax function leads to a substantial drop in performance (120 ppl), although it still performs better than the HMM variants where the recurrent state is a distribution.

Another way to investigate the effect of the softmax is to normalize the hidden state output just before applying the emission function, while keeping the sigmoid non-linearity: This performs somewhat worse than the softmax non-linearity, which indicates that it is significant whether the input to the emission function is normalized or softmaxed before multiplying with the (emission) embedding matrix.

As a comparison for how much the softmax non-linearity acts as a bottleneck, a neural bigram model outperforms these approaches, obtaining 177 validation perplexity on this same setup.

Replacing the RNN emission function with that of an HMM leads to even worse performance than the HMM: Using a softmax non-linearity or a sigmoid followed by normalization does not make a significant difference.

Using multiplicative integration leads to only a small drop in performance compared to a vanilla Elman RNN, and doing so with a single transformation matrix (making it comparable to what an RNN is doing) leads to only a small further drop.

In contrast, preliminary experiments showed that the second transformation matrix is crucial in the performance of the vanilla Elman network.

In our experimental setup an LSTM performs only slightly better than the Elman network (80 vs 87 perplexity).

While more extensive hyperparameter tuning BID14 or more sophisticated optimization and regularization techniques BID15 would likely improve performance, that is not the goal of this evaluation.

A strength of HMM bottlenecks is forcing the model to produce an interpretable hidden representation.

A classic example of this property is part-of-speech tag induction.

It is therefore natural to ask whether changes in the architecture of our models correlate with their ability to discover syntactic properties.

We evaluate this by analyzing the models implicitly predicted tag distribution at each time step.

Specifically, while no model is likely to predict the correct next word, we assume the HMMs errors will preserve basic tag-tag patterns of the language, and that this may not be true for RNNs.

We test this by computing the accuracy of predicting the tag of the word in the sequence out of the next word distribution.

None of the models were trained to perform this task.

First, we compute a tag distribution p(t|w) for every word in the training portion of the Penn Treebank.

Next, we multiply this value by the model's p(w) = x i , and sum across the vocabulary.

This provides us the model's distribution over tags at the given time p(t) i .

We compare the most likely marginal tag against the ground truth to compute a tagging accuracy.

This evaluation rewards models which place their emission probability mass predominantly on words of the correct part-of-speech.

We compute this metric across both the full PTB tagset and the universal tags of BID18 .The HMM allows for Viterbi decoding which allows us to compute p(t|max dim (c i )).

The more distributed the models' representations are, the more the tag distribution given the max dimension will differ from the complete marginal.

For HMMs with distributional hidden states the maximum dimension provided the best performance.

In contrast, Elman models perform best when conditioned on the full hidden state.

Results are shown in TAB2 and plotted against perplexity in FIG3 .

DISPLAYFORM0 Recently, a number of recent papers have identified variants of gated RNNs which are simpler than LSTMs but perform competitively or satisfy properties that LSTMs lack.

Foerster et al. FORMULA0 proposed RNNs without recurrent non-linearities to improve interpretability.

BID0 proposed gated RNN variants with type constraints.

BID17 identified a class of RNNs called rational recurrences, in which the hidden states can be computed by WFSAs.

Another strand of recent work proposed neural models that learn discrete, interpretable structure: BID26 introduced a mixture of softmax model where the output distribution is conditioned on discrete latent variable.

BID20 proposed a language model that jointly learns unsupervised syntactic (tree) structure, while BID21 used neural hidden Markov models for Part-of-Speech induction.

BID24 and BID22 proposed models for segmental structure over sequences, while neural transduction models with discrete latent alignments have also been proposed BID27 .

In this work, we presented a theoretical and empirical investigation into the model variants over the spectrum of possible hybridization between HMMs and RNNs.

By carefully controlling for every design choices, we provide new insights into several factors including independence assumptions, the placement of softmax, and the use of nonliniarity and how these choices influence the interplay between expressiveness and interpretability.

Comprehensive empirical results demonstrate that the key elements to better performance of the HMM are the use of a sigmoid instead of softmax linearity in the recurrent cell, and the use of an unnormalized output distribution matrix in the emission computation.

Multiplicative integration of the previous hidden state and input embedding, and intermediate normalizations in the cell computation are less consequential.

We also find that HMM outperforms other RNNs variants in a next POS tag prediction task, which demonstrates the advantages of models with discrete bottlenecks in increased interpretability.

<|TLDR|>

@highlight

Are HMMs a special case of RNNs? We investigate a series of architectural transformations between HMMs and RNNs, both through theoretical derivations and empirical hybridization and provide new insights.