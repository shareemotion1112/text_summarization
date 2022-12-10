Highly regularized LSTMs achieve impressive results on several benchmark datasets in language modeling.

We propose a new regularization method based on decoding the last token in the context using the predicted distribution of the next token.

This biases the model towards retaining more contextual information, in turn improving its ability to predict the next token.

With negligible overhead in the number of parameters and training time, our Past Decode Regularization (PDR) method achieves a word level perplexity of 55.6 on the Penn Treebank and 63.5 on the WikiText-2 datasets using a single softmax.

We also show gains by using PDR in combination with a mixture-of-softmaxes, achieving a word level perplexity of 53.8 and 60.5 on these datasets.

In addition, our method achieves 1.169 bits-per-character on the Penn Treebank Character dataset for character level language modeling.

These results constitute a new state-of-the-art in their respective settings.

Language modeling is a fundamental task in natural language processing.

Given a sequence of tokens, its joint probability distribution can be modeled using the auto-regressive conditional factorization.

This leads to a convenient formulation where a language model has to predict the next token given a sequence of tokens as context.

Recurrent neural networks are an effective way to compute distributed representations of the context by sequentially operating on the embeddings of the tokens.

These representations can then be used to predict the next token as a probability distribution over a fixed vocabulary using a linear decoder followed by Softmax.

Starting from the work of BID16 , there has been a long list of works that seek to improve language modeling performance using more sophisticated recurrent neural networks (RNNs) BID26 ; BID27 ; BID28 ; BID17 ).

However, in more recent work vanilla LSTMs BID7 ) with relatively large number of parameters have been shown to achieve state-of-the-art performance on several standard benchmark datasets both in word-level and character-level perplexity BID14 b) ; BID12 ; BID25 ).

A key component in these models is the use of several forms of regularization e.g. variational dropout on the token embeddings BID3 ), dropout on the hidden-to-hidden weights in the LSTM BID24 ), norm regularization on the outputs of the LSTM and classical dropout BID23 ).

By carefully tuning the hyperparameters associated with these regularizers combined with optimization algorithms like NT-ASGD (a variant of the Averaged SGD), it is possible to achieve very good performance.

Each of these regularizations address different parts of the LSTM model and are general techniques that could be applied to any other sequence modeling problem.

In this paper, we propose a regularization technique that is specific to language modeling.

One unique aspect of language modeling using LSTMs (or any RNN) is that at each time step t, the model takes as input a particular token x t from a vocabulary W and using the hidden state of the LSTM (which encodes the context till x t ) predicts a probability distribution w t+1 on the next token x t+1 over the same vocabulary as output.

Since x t can be mapped to a trivial probability distribution over W , this operation can be interpreted as transforming distributions over W BID9 ).

Clearly, the output distribution is dependent on and is a function of x t and the context further in the past and encodes information about it.

We ask the following question -How much information is it possible to decode about the input distribution (and hence x t ) from the output distribution w t+1 ?

In general, it is impossible to decode x t unambiguously.

Even if the language model is perfect and correctly predicts x t+1 with probability 1, there could be many tokens preceding it.

However, in this case the number of possibilities for x t will be limited, as dictated by the bigram statistics of the corpus and the language in general.

We argue that biasing the language model such that it is possible to decode more information about the past tokens from the predicted next token distribution is beneficial.

We incorporate this intuition into a regularization term in the loss function of the language model.

The symmetry in the inputs and outputs of the language model at each step lends itself to a simple decoding operation.

It can be cast as a (pseudo) language modeling problem in "reverse", where the future prediction w t+1 acts as the input and the last token x t acts as the target of prediction.

The token embedding matrix and weights of the linear decoder of the main language model can be reused in the past decoding operation.

We only need a few extra parameters to model the nonlinear transformation performed by the LSTM, which we do by using a simple stateless layer.

We compute the cross-entropy loss between the decoded distribution for the past token and x t and add it to the main loss function after suitable weighting.

The extra parameters used in the past decoding are discarded during inference time.

We call our method Past Decode Regularization or PDR for short.

We conduct extensive experiments on four benchmark datasets for word level and character level language modeling by combining PDR with existing LSTM based language models and achieve new state-of-the-art performance on three of them.

Let X = (x 1 , x 2 , · · · , x t , · · · , x T ) be a sequence of tokens.

In this paper, we will experiment with both word level and character level language modeling.

Therefore, tokens can be either words or characters.

The joint probability P (X) factorizes into DISPLAYFORM0 Let c t = (x 1 , x 2 , · · · , x t ) denote the context available to the language model for x t+1 .

Let W denote the vocabulary of tokens, each of which is embedded into a vector of dimension d. Let E denote the token embedding matrix of dimension |W | × d and e w denote the embedding of w ∈ W .

An LSTM computes a distributed representation of c t in the form of its hidden state h t , which we assume has dimension d as well.

The probability that the next token is w can then be calculated using a linear decoder followed by a Softmax layer as DISPLAYFORM1 where b w is the entry corresponding to w in a bias vector b of dimension |W | and | w represents projection onto w.

Here we assume that the weights of the decoder are tied with the token embedding matrix E (Inan et al. FORMULA0 ; BID18 ).

To optimize the parameters of the language model θ, the loss function to be minimized during training is set as the cross-entropy between the predicted distribution P θ (w|c t ) and the actual token x t+1 .

DISPLAYFORM2 Note that Eq.(2), when applied to all w ∈ W produces a 1 × |W | vector w t+1 , encapsulating the prediction the language model has about the next token x t+1 .

Since this is dependent on and conditioned on c t , w t+1 clearly encodes information about it; in particular about the last token x t in c t .

In turn, it should be possible to infer or decode some limited information about x t from w t+1 .

We argue that by biasing the model to be more accurate in recalling information about past tokens, we can help it in predicting the next token better.

To this end, we define the following decoding operation to compute a probability distribution over w c ∈ W as the last token in the context.

Here f θr is a non-linear function that maps vectors in R d to vectors in R d and b θr is a bias vector of dimension |W |, together with parameters θ r .

In effect, we are decoding the past -the last token in the context x t .

This produces a vector w r t of dimension 1 × |W |.

The cross-entropy loss with respect to the actual last token x t can then be computed as DISPLAYFORM3 DISPLAYFORM4 Here P DR stands for Past Decode Regularization.

L P DR captures the extent to which the decoded distribution of tokens differs from the actual tokens x t in the context.

Note the symmetry between Eqs. FORMULA1 and FORMULA4 .

The "input" in the latter case is w t+1 and the "context" is provided by a nonlinear transformation of w t+1 E. Different from the former, the context in Eq. FORMULA4 does not preserve any state information across time steps as we want to decode only using w t+1 .

The term w t+1 E can be interpreted as a "soft" token embedding lookup, where the token vector w t+1 is a probability distribution instead of a unit vector.

We add λ P DR L P DR to the loss function in Eq. FORMULA2 as a regularization term, where λ P DR is a positive weighting coefficient, to construct the following new loss function for the language model.

DISPLAYFORM5 Thus equivalently PDR can also be viewed as a method of defining an augmented loss function for language modeling.

The choice of λ P DR dictates the degree to which we want the language model to incorporate our inductive bias i.e. decodability of the last token in the context.

If it is too large, the model will fail to predict the next token, which is its primary task.

If it is zero or too small, the model will retain less information about the last token which hampers its predictive performance.

In practice, we choose λ P DR by a search based on validation set performance.

Note that the trainable parameters θ r associated with PDR are used only during training to bias the language model and are not used at inference time.

This also means that it is important to control the complexity of the nonlinear function f θr so as not to overly bias the training.

As a simple choice, we use a single fully connected layer of size d followed by a Tanh nonlinearity as f θr .

This introduces few extra parameters and a small increase in training time as compared to a model not using PDR.

We present extensive experimental results to show the efficacy of using PDR for language modeling on four standard benchmark datasets -two each for word level and character level language modeling.

For the former, we evaluate our method on the Penn Treebank (PTB) BID16 ) and the WikiText-2 (WT2) BID13 ) datasets.

For the latter, we use the Penn Treebank Character (PTBC) BID16 ) and the Hutter Prize Wikipedia Prize BID8 ) (also known as Enwik8) datasets.

Key statistics for these datasets is presented in Table 1 .As mentioned in the introduction, some of the best existing results on these datasets are obtained by using extensive regularization techniques on relatively large LSTMs BID14 b) ; BID25 ).

We apply our regularization technique to these models, the so called AWD-LSTM.

We consider two versions of the model -one with a single softmax (AWD-LSTM) and one with a mixture-of-softmaxes (AWD-LSTM-MoS).

The PDR regularization term is computed according to Eq.(4) and Eq.(5).

We call our model AWD-LSTM+PDR when using a single softmax and AWD-LSTM-MoS+PDR when using a mixture-of-softmaxes.

We largely follow the experimental procedure of the original models and incorporate their dropouts and regularizations in our experiments.

The relative contribution of these existing regularizations and PDR will be analyzed in Section 6.There are 7 hyperparameters associated with the regularizations used in AWD-LSTM (and one extra with MoS).

PDR also has an associated weighting coefficient λ P DR .

For our experiments, we set λ P DR = 0.001 which was determined by a coarse search on the PTB and WT2 validation sets.

For the remaining ones, we perform light hyperparameter search in the vicinity of those reported for AWD-LSTM in BID14 b) and for AWD-LSTM-MoS in BID25 .

For the single softmax model (AWD-LSTM+PDR), for both PTB and WT2, we use a 3-layered LSTM with 1150, 1150 and 400 hidden dimensions.

The word embedding dimension is set to d = 400.

For the mixture-of-softmax model, we use a 3-layer LSTM with dimensions 960, 960 and 620, embedding dimension of 280 and 15 experts for PTB and a 3-layer LSTM with dimensions 1150, 1150 and 650, embedding dimension of d = 300 and 15 experts for WT2.

Weight tying is used in all the models.

For training the models, we follow the same procedure as AWD-LSTM i.e. a combination of SGD and NT-ASGD, followed by finetuning.

We adopt the learning rate schedules and batch sizes of BID14 and BID25 in our experiments.

For PTBC, we use a 3-layer LSTM with 1000, 1000 and 200 hidden dimensions and a character embedding dimension of d = 200.

For Enwik8, we use a LSTM with 1850, 1850 and 400 hidden dimensions and the characters are embedded in d = 400 dimensions.

For training, we largely follow the procedure laid out in BID15 .

For each of the datasets, AWD-LSTM+PDR has less than 1% more parameters than the corresponding AWD-LSTM model (during training only).

The maximum observed time overhead due to the additional computation is less than 3%.

The results for PTB are shown in TAB2 .

With a single softmax, our method (AWD-LSTM+PDR) achieves a perplexity of 55.6 on the PTB test set, which improves on the current state-of-the-art with a single softmax by an absolute 1.7 points.

The advantages of better information retention due to PDR are maintained when combined with a continuous cache pointer BID5 ), where our method yields an absolute improvement of 1.2 over AWD-LSTM.

Notably, when coupled with dynamic evaluation BID10 ), the perplexity is decreased further to 49.3.

To the best of our knowledge, ours is the first method to achieve a sub 50 perplexity on the PTB test set with a single softmax.

Note that, for both cache pointer and dynamic evaluation, we coarsely tune the associated hyperparameters on the validation set.

Using a mixture-of-softmaxes, our method (AWD-LSTM-MoS+PDR) achieves a test perplexity of 53.8, an improvement of 0.6 points over the current state-of-the-art.

The use of dynamic evaluation pushes the perplexity further down to 47.3.

PTB is a restrictive dataset with a vocabulary of 10K words.

Achieving good perplexity requires considerable regularization.

The fact that PDR can improve upon existing heavily regularized models is empirical evidence of its distinctive nature and its effectiveness in improving language models.

TAB3 shows the perplexities achieved by our model on WT2.

This dataset is considerably more complex than PTB with a vocabulary of more than 33K words.

AWD-LSTM+PDR improves over the current state-of-the-art with a single softmax by a significant 2.3 points, achieving a perplexity of 63.5.

The gains are maintained with the use of cache pointer (2.4 points) and with the use of dynamic evaluation (1.7 points).

Using a mixture-of-softmaxes, AWD-LSTM-MoS+PDR achieves perplexities of 60.5 and 40.3 (with dynamic evaluation) on the WT2 test set, improving upon the current state-of-the-art by 1.0 and 0.4 points respectively.

We consider the Gigaword dataset BID0 with a truncated vocabulary of about 100K tokens with the highest frequency and apply PDR to a baseline 2-layer LSTM language model with embedding and hidden dimensions set to 1024.

We use all the shards from the training set for training and a few shards from the heldout set for validation (heldout-0,10) and test (heldout-20,30,40

Sate-of-the-art Methods (Single Softmax) tuned the PDR coefficient coarsely in the vicinity of 0.001.

While the baseline model achieved a validation (test) perplexity of 44.3 (43.1), on applying PDR, the model achieved a perplexity of 44.0 (42.5).

Thus, PDR is relatively less effective on larger datasets, a fact also observed for other regularization techniques on such datasets BID25 ).

The results on PTBC are shown in Table 4 .

Our method achieves a bits-per-character (BPC) performance of 1.169 on the PTBC test set, improving on the current state-of-the-art by 0.006 or 0.5%.

It is notable that even with this highly processed dataset and a small vocabulary of only 51 tokens, our method improves on already highly regularized models.

Finally, we present results on Enwik8 in Table 4 : Bits-per-character on the PTBC test set.

Ha et al. FORMULA0 TAB4 .

AWD-LSTM+PDR achieves 1.245 BPC.

This is 0.012 or about 1% less than the 1.257 BPC achieved by AWD-LSTM in our experiments (with hyperparameters from BID15 ).

In this section, we analyze PDR by probing its performance in several ways and comparing it with current state-of-the-art models that do not use PDR.

PTB Valid WT2 Valid AWD-LSTM (NoReg) 108.6 142.7 AWD-LSTM (NoReg) + PDR 106.2 137.6 Table 6 : Validation perplexities for AWD-LSTM without any regularization and with only PDR.To verify that indeed PDR can act as a form of regularization, we perform the following experiment.

We take the models for PTB and WT2 and turn off all dropouts and regularization and compare its performance with only PDR turned on.

The results, as shown in Table 6 , validate the premise of PDR.

The model with only PDR turned on achieves 2.4 and 5.1 better validation perplexity on PTB and WT2 as compared to the model without any regularization.

Thus, biasing the LSTM by decoding the distribution of past tokens from the predicted next-token distribution can indeed act as a regularizer leading to better generalization performance.

Next, we plot histograms of the negative log-likelihoods of the correct context tokens x t in the past decoded vector w r t computed using our best models on the PTB and WT2 validation sets in 1(a).

The NLL values are significantly peaked near 0, which means that the past decoding operation is able to decode significant amount of information about the last token in the context.

To investigate the effect of hyperparameters on PDR, we pick 60 sets of random hyperparameters in the vicinity of those reported by BID14 and compute the validation set perplexity after training (without finetuning) on PTB, for both AWD-LSTM+PDR and AWD-LSTM.

Their histograms are plotted in FIG4 .

The perplexities for models with PDR are distributed slightly to the left of those without PDR.

There appears to be more instances of perplexities in the higher range for models without PDR.

Note that there are certainly hyperparameter settings where adding PDR leads to lower validation complexity, as is generally the case for any regularization method.

To show the qualitative difference between AWD-LSTM+PDR and AWD-LSTM, in FIG5 , we plot a histogram of the entropy of the predicted next token distribution w t+1 for all the tokens in the validation set of PTB achieved by their respective best models.

The distributions for the two models is slightly different, with some identifiable patterns.

The use of PDR has the effect of reducing the entropy of the predicted distribution when it is in the higher range of 8 and above, pushing it into the range of 5-8.

This shows that one way PDR biases the language model is by reducing the entropy of the predicted next token distribution.

Indeed, one way to reduce the cross-entropy between We also compare the training curves for the two models in FIG5 on PTB.

Although the two models use slightly different hyperparameters, the regularization effect of PDR is apparent with a lower validation perplexity but higher training perplexity.

The corresponding trends shown in FIG5 for WT2 have similar characteristics.

We perform a set of ablation experiments on the best AWD-LSTM+PDR models for PTB and WT2 to understand the relative contribution of PDR and the other regularizations used in the model.

The results are shown in TAB6 .

In both cases, PDR has a significant effect in decreasing the validation set performance, albeit lesser than the other forms of regularization.

This is not surprising as PDR does not influence the LSTM directly.

Our method builds on the work of using sophisticated regularization techniques to train LSTMs for language modeling.

In particular, the AWD-LSTM model achieves state-of-the-art performance with a single softmax on the four datasets considered in this paper BID14 b) ).

BID12 also achieve similar results with highly regularized LSTMs.

By addressing the so-called softmax bottleneck in single softmax models, BID25 use a mixture-of-softmaxes to achieve significantly lower perplexities.

PDR utilizes the symmetry between the inputs and outputs of a language model, a fact that is also exploited in weight tying BID9 ; BID18 ).

Our method can be used with untied weights as well.

Although motivated by language modeling, PDR can also be applied to seq2seq models with shared input-output vocabularies, such as those used for text summarization and neural machine translation (with byte pair encoding of words) BID18 ).

Regularizing the training of an LSTM by combining the main objective function with auxiliary tasks has been successfully applied to several tasks in NLP BID19 BID21 ).

In fact, a popular choice for the auxiliary task is language modeling itself.

This in turn is related to multi-task learning BID2 ).Specialized architectures like Recurrent Highway Networks BID27 ) and NAS BID28 ) have been successfully used to achieve competitive performance in language modeling.

The former one makes the hidden-to-hidden transition function more complex allowing for more refined information flow.

Such architectures are especially important for character level language modeling where strong results have been shown using Fast-Slow RNNs BID17 ), a two level architecture where the slowly changing recurrent network tries to capture more long range dependencies.

The use of historical information can greatly help language models deal with long range dependencies as shown by BID13 ; BID10 ; BID20 .

Finally, in a recent paper, BID4 achieve improved performance for language modeling by using frequency agnostic word embeddings, a technique orthogonal to and combinable with PDR.

<|TLDR|>

@highlight

Decoding the last token in the context using the predicted next token distribution acts as a regularizer and improves language modeling.

@highlight

The authors introduce the idea of past decoding for the purpose of regularization for improved perplexity on Penn Treebank

@highlight

Proposes an additional loss term to use when training an LSTM LM and shows that by adding this loss term they can achieve SOTA perplexity on a number of LM benchmarks.

@highlight

Suggests a new regularization technique which can be added on top of those used in AWD-LSTM of Merity et al. (2017) with little overhead.