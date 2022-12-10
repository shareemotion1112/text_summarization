A distinct commonality between HMMs and RNNs is that they both learn hidden representations for sequential data.

In addition, it has been noted that the backward computation of the Baum-Welch algorithm for HMMs is a special case of the back-propagation algorithm used for neural networks (Eisner (2016)).

Do these observations suggest that, despite their many apparent differences, HMMs are a special case of RNNs?

In this paper, we show that that is indeed the case, and investigate a series of architectural transformations between HMMs and RNNs, both through theoretical derivations and empirical hybridization.

In particular, we investigate three key design factors—independence assumptions between the hidden states and the observation, the placement of softmaxes, and the use of non-linearities—in order to pin down their empirical effects.

We present a comprehensive empirical study to provide insights into the interplay between expressivity and interpretability in this model family with respect to language modeling and parts-of-speech induction.

The sequence is a common structure among many forms of naturally occurring data, including speech, text, video, and DNA.

As such, sequence modeling has long been a core research problem across several fields of machine learning and AI.

By far the most widely used approach for decades is Hidden Markov Models BID1 BID10 , which assumes a sequence of discrete latent variables to generate a sequence of observed variables.

When the latent variables are unobserved, unsupervised training of HMMs can be performed via the Baum-Welch algorithm (which, in turn, is based on the forward-backward algorithm), as a special case of Expectation-Maximization (EM) BID4 .

Importantly, the discrete nature of the latent variables has the benefit of interpretability, as they recover contextual clustering of the output variables.

In contrast, Recurrent Neural Networks (RNNs) BID11 BID6 introduced later assume continuous latent representations.

Their hidden states have no probabilistic interpretation, regardless of many different architectural variants, such as LSTMs BID9 , GRUs BID3 and RANs BID13 .Despite their many apparent differences, both HMMs and RNNs model hidden representations for sequential data.

At the heart of both models are: a state at time t, a transition function f : h t−1 → h t in latent space, and an emission function g : h t → x t .

In addition, it has been noted that the backward computation in the Baum-Welch algorithm is a special case of back-propagation for neural networks BID5 .

Therefore, a natural question arises as to the fundamental relationship between HMMs and RNNs.

Might HMMs be a special case of RNNs?In this paper, we investigate a series of architectural transformations between HMMs and RNNsboth through theoretical derivations and empirical hybridization.

In particular, we demonstrate that forward marginal inference for an HMM-accumulating forward probabilities to compute the marginal emission and hidden state distributions at each time step-can be reformulated as equations for computing an RNN cell.

In addition, we investigate three key design factors-independence Figure 1 : Above each of the models we indicate the type of transition and emission cells used.

H for HMM, R for RNN/Elman and F is a novel Fusion defined in §3.3.

It is particularly important to track when a vector is a distribution (resides in a simplex) versus in the unit cube (e.g. after a sigmoid non-linearity).

These are indicated by c i and c i , respectively.

SM stands for softmax rows.assumptions between the hidden states and observations, the placement of softmaxes, and the use of non-linearities-in order to pin down their empirical effects.

While we focus on HMMs with discrete outputs, our analysis framework could be extended to HMMs over continuous observations.

Our work builds on earlier work that have also noted the connection between RNNs and HMMs BID23 BID25 (see §7).

Our contribution is to provide the first thorough theoretical investigation into the model variants, carefully controlling for every design choices, along with comprehensive empirical analysis over the spectrum of possible hybridization between HMMs and RNNs.

We find that the key elements to better performance of the HMMs are the use of a sigmoid instead of softmax linearity in the recurrent cell, and the use of an unnormalized output distribution matrix in the emission computation.

On the other hand, multiplicative integration of the previous hidden state and input embedding, and intermediate normalizations in the cell computation are less consequential.

We also find that HMMs outperform other RNNs variants for unsupervised prediction of the next POS tag, demonstrating the advantages of discrete bottlenecks for increased interpretability.

The paper is structured as follows.

First, we present the derivation of HMM marginal inference as a special case of RNN computation ( §2).

Next we explore a gradual transformation of HMMs into RNNs ( §3), followed by the reverse transformation of Elman RNNs back to HMMs ( §4).

Finally we provide empirical analysis in §5 and §6 to pin point the effects of varying design choices over possible hybridizations between HMMs and RNNs.

We start by defining HMMs as discrete sequence models, together with the forward-backward algorithm which is used for inference.

Then we show that, by rewriting the forward algorithm, the computation can be viewed as updating a hidden state at each time step by feeding the previous word prediction, and then computing the next word distribution, similar to the way RNNs are structured.

The resulting architecture corresponds to the first cell in Figure 1 .

(1:n) = {x BID0 , . . . , x (n) } be a sequence of random variables, where each x is drawn from a vocabulary V of size v, and an instance x is represented as an integer w or a one-hot vector e (w) , where w corresponds to an index in V. BID0 We also define a corresponding sequence of hidden variables h(1:n) = {h BID0 , . . .

, h (n) }, where h ∈ {1, 2, . . .

m}. The distribution P (x) is defined by marginalizing over h, and factorizes as follows: DISPLAYFORM0 We define the hidden state distribution, referred to as the transition distribution, and the the emission (output) distribution as DISPLAYFORM1 DISPLAYFORM2

Inference for HMMs (marginalizing over the hidden states to compute the observed sequence probabilities) is performed with the forward-backward algorithm.

The backward algorithm is equivalent to automatically differentiating the forward algorithm BID5 .

Therefore, while traditional HMM implementations had to implement both the forward and backward algorithm, and train the model with the EM algorithm, we only implement the forward algorithm in standard deep learning software, and perform end-to-end minibatched SGD training, efficiently parallelized on a GPU.Let w = {w BID0 , . . .

, w (n) } be the observed sequence, and w (i) the one-hot representation of w (i) .

The forward probabilities a are defined recurrently (i.e., sequentially recursively) as DISPLAYFORM0 DISPLAYFORM1 This can be rewritten by defining DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 and substituting a, so that equation 6 is rewritten as left below, or expressed directly in terms of the parameters used to define the distributions with vectorized computations (right below): DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 Here w (i) is used as a one-hot vector, and the bias vectors b and d are omitted for clarity.

Note that the computation of s (i) can be delayed until time step i + 1.

The computation step can therefore be rewritten to let c be the recurrent vector (equivalent logspace formulations presented on the right): DISPLAYFORM9 , DISPLAYFORM10 DISPLAYFORM11 (15) DISPLAYFORM12 This can be viewed as a step of a recurrent neural network with tied input and output embeddings: Equation 14 embeds the previous prediction, equations 15 and 16, the transition step, updates the hidden state c, corresponding to the cell of a RNN, and equations 17 and 18, the emission step, computes the output next word probability.

We can now compare this formulation against the definition of an Elman RNN with tied embeddings and a sigmoid non-linearity.

These equations correspond to the first and last cells in Figure 1 .

The Elman RNN has the same parameters, except for an additional input matrix U ∈ R m×m .

DISPLAYFORM13 ), DISPLAYFORM14

Having established the relation between HMMs and RNNs, we propose a number of models that we hypothesize have intermediate expressiveness between HMMs and RNNs.

The architecture transformations can be seen in the first 3 cells in Figure 1 .

We will evaluate these model variants empirically ( §5), and investigate their interpretability ( §6).

By relaxing the independence assumption of the HMM transition probability distribution we can increase the expressiveness of the HMM "cell" by modelling more complex interactions between the fed word and the hidden state.

These model variants are non-homogeneous HMMs.

Tensor-based feeding:Following BID21 we define the transition distribution as DISPLAYFORM0 where W ∈ R m×m×m , B ∈ R m×m .

As tensor-based feeding increases the number of parameters considerably, we also propose an additive version: DISPLAYFORM0 DISPLAYFORM1

Finally we propose a more expressive model where interaction is controlled via a gating mechanism and the feeding step uses unnormalized embeddings (this does not violate the HMM factorization): DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 where DISPLAYFORM3 2 where normalize(y) =

Another way to make HMMs more expressive is to relax their independence assumptions through delaying when vectors are normalized to probability distributions by applying the softmax function.

The computation of the recurrent vector c DISPLAYFORM0 ) is replaced with DISPLAYFORM1 Both c and s are still valid probability distributions, but the independence assumption in the distribution over h (i) no longer holds.

A further transformation is to delay the emission softmax until after multiplication with the hidden vector.

This effectively replaces the HMM's emission computation with that of the RNN: DISPLAYFORM0 This formulation breaks the independence assumption that the output distribution is only conditioned on the hidden state assignment.

Instead it can be viewed as taking the expectation over the (unnormalized) embeddings with respect to the state distribution c, then softmaxed (H R in Fig 1) .

We can go further towards RNNs and replace the softmax in the transition by a sigmoid non-linearity.

The sigmoid is placed in the same position as the delayed softmax.

The recurrent state c is no longer a distribution so the output has to be renormalized so the emission still computes a distribution: DISPLAYFORM0 DISPLAYFORM1 This model could also be combined with a delayed emission softmax -which we'll see makes it closer to an Elman RNN.

This model is indicated as F (fusion) in Figure 1 .

Analogously to making the HMM more similar to Elman RNNs, we can make Elman networks more similar to HMMs.

Examples of these transformations can be seen in the last 2 cells in Figure 1 .

First, we use the Elman cell with an HMM emission function.

This requires the hidden state be a distribution.

We consider two options: One is to replace the sigmoid non-linearity with the softmax function (R H in Figure 1 ): DISPLAYFORM0 DISPLAYFORM1 The second formulation is to keep the sigmoid non-linearity, but normalize the hidden state output inside the emission computation: DISPLAYFORM2 DISPLAYFORM3

Second, we experiment with replacing the sigmoid non-linearity with a softmax: DISPLAYFORM0 As a more flexible variant, the softmax is applied only to compute the emission distribution, while the sigmoid non-linearity is still applied to recurrent state: DISPLAYFORM1 DISPLAYFORM2

In the HMM cell, the integration of the previous recurrent state and the input embedding is modelled through an element-wise product instead of adding affine transformations of the two vectors.

We can modify the Elman cell to do a similar multiplicative integration: DISPLAYFORM0 Or, using a single transformation matrix: DISPLAYFORM1

Our formulations investigate a series of small architectural changes to HMMs and Elman cells.

In particular, these changes raise questions about the expressivity and importance of (1) normalization within the recurrence and (2) independence assumptions during emission.

In this section, we analyze the effects of these changes quantitatively via a standard language modeling benchmark.

We follow the standard PTB language modeling setup BID2 BID16 .

We work with one-layer models to enable a direct comparison between RNNs and HMMs and a budget of 10 million parameters (typically corresponding to hidden state sizes of around 900).

Models are trained with batched backpropagation through time (35 steps).

Input and output embeddings are tied in all models.

Models are optimized with a grid search over optimizer parameters for two strategies: SGD BID3 and AMSProp.

AMSProp is based on the optimization setup proposed in BID14 .

BID4

We see from the results in TAB3 (also depicted in FIG2 ) that the HMM models perform significantly worse than the Elman network, as expected.

Interestingly, many of the HMM variants that in principle have more expressivity or weaker independence assumptions do not perform better than the vanilla HMM.

This includes delaying the transition or emission softmax, and most of the feeding models.

The exception is the gated feeding model, which does substantially better, showing that gating is an effective way of incorporating more context into the transition matrix.

Using a sigmoid non-linearity before the output of the HMM cell (instead of a softmax) does improve performance (by 44 ppl), and combining that with delaying the emission softmax gives a substantial improvement (almost another 100 ppl), making it much closer to some of the RNN variants.

We also evaluate variants of Elman RNNs: Just replacing the sigmoid non-linearity with the softmax function leads to a substantial drop in performance (120 ppl), although it still performs better than the HMM variants where the recurrent state is a distribution.

Another way to investigate the effect of the softmax is to normalize the hidden state output just before applying the emission function, while keeping the sigmoid non-linearity: This performs somewhat worse than the softmax non-linearity, which indicates that it is significant whether the input to the emission function is normalized or softmaxed before multiplying with the (emission) embedding matrix.

As a comparison for how much the softmax non-linearity acts as a bottleneck, a neural bigram model outperforms these approaches, obtaining 177 validation perplexity on this same setup.

Replacing the RNN emission function with that of an HMM leads to even worse performance than the HMM; a softmax non-linearity or a sigmoid followed by normalization does not make a significant difference.

Multiplicative integration leads to only a small drop in performance compared to a vanilla Elman RNN, and doing so with a single transformation matrix (more comparable to the HMM) leads to only a small further drop.

In contrast, preliminary experiments showed that the second transformation matrix is crucial in the performance of the vanilla Elman network.

To put our results in context, we also compare against an LSTM baseline with the same number of parameters, using the same regularization and hyperparameter search strategy as for our other models.

While more extensive hyperparameter tuning BID14 or more sophisticated optimization and regularization techniques BID15 would improve performance, the goal here is just to do a fair comparison within the computational resources we had available to optimize all models, not to compete with state-of-the-art performance.

TAB1 : Tagging accuracies for representative models.

Accuracy is calculated by converting p(w) to p(t) according to WSJ tag distributions.

A strength of HMM bottlenecks is forcing the model to produce an interpretable hidden representation.

A classic example of this property in language modeling is part-of-speech tag induction.

It is therefore natural to ask whether changes in the architecture of our models correlate with their ability to discover syntactic properties.

We evaluate this by analyzing the models' implicitly predicted tag distributions at each time step.

Specifically we hypothesize that the HMMs will preserve basic tag-tag patterns of the language, and that this may not be true for RNNs.

We test this by computing the accuracy of predicting the tag of the next word in the sequence out of the next word distribution.

None of the models were trained to perform this task.

We estimate the model's distribution over POS tags at each time step, p(t), by marginalizing over the model's output word distribution p(w) and the context-independent tag distribution p(t|w) for every word in the training portion of the PTB.

We compare the most likely marginal tag against the ground truth to compute a tagging accuracy.

This evaluation rewards models which place their emission probability mass predominantly on words of the correct part-of-speech.

We compute this metric across both the full PTB tagset and universal tags (UPOS) BID18 .Viterbi decoding in HMMs enable us to compute the tag distribution conditioned on the highest scoring (Viterbi) state at each time step.

This leads to better performance than marginalizing over hidden state values, showing that the states encode meaningful word clusterings.

In contrast, Elman models perform best when conditioned on the full hidden state rather than the maximum dimension only.

Results are shown in TAB1 and plotted against perplexity in FIG3 .

A number of recent papers have identified variants of gated RNNs which are simpler than LSTMs but perform competitively or satisfy properties that LSTMs lack.

These variants include RNNs without recurrent non-linearities to improve interpretability BID7 , gated RNN variants with type constraints BID0 , and a class of RNNs called rational recurrences, in which the hidden states can be computed by WFSAs BID17 .

Our goal was instead to compare RNNs against HMMs, which while clearly less expressive can provide complementary insights into the strengths of RNNs.

Another strand of recent work proposed neural models that learn discrete, interpretable structure: BID27 introduced a mixture of softmaxes model where the output distribution is conditioned on discrete latent variable.

Other work includes language modeling that jointly learns unsupervised syntactic (tree) structure BID20 and neural hidden Markov models for Part-of-Speech induction BID21 .

Models of segmental structure over sequences BID24 BID22 and neural transduction models with discrete latent alignments BID28 have also been proposed.

In this work, we presented a theoretical and empirical investigation into model variants over the spectrum of possible hybridization between HMMs and RNNs.

By carefully controlling all design choices, we provide new insights into several factors including independence assumptions, the placement of softmax, and the use of nonlinearities.

Comprehensive empirical results demonstrate that the key elements to better performance of the RNN are the use of a sigmoid instead of softmax linearity in the recurrent cell, and the use of an unnormalized output distribution matrix in the emission computation.

Multiplicative integration of the previous hidden state and input embedding, and intermediate normalizations in the cell computation are less consequential.

HMMs outperforms other RNNs variants in a next POS tag prediction task, which demonstrates the advantages of models with discrete bottlenecks in increased interpretability.

<|TLDR|>

@highlight

Are HMMs a special case of RNNs? We investigate a series of architectural transformations between HMMs and RNNs, both through theoretical derivations and empirical hybridization and provide new insights.

@highlight

This paper explores if HMMs are a special case of RNNs using language modeling and POS tagging