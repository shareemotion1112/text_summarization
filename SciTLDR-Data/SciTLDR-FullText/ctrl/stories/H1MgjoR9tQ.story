Continuous Bag of Words (CBOW) is a powerful text embedding method.

Due to its strong capabilities to encode word content, CBOW embeddings perform well on a wide range of downstream tasks while being efficient to compute.

However, CBOW is not capable of capturing the word order.

The reason is that the computation of CBOW's word embeddings is commutative, i.e., embeddings of XYZ and ZYX are the same.

In order to address this shortcoming, we propose a learning algorithm for the Continuous Matrix Space Model, which we call Continual Multiplication of Words (CMOW).

Our algorithm is an adaptation of word2vec, so that it can be trained on large quantities of unlabeled text.

We empirically show that CMOW better captures linguistic properties, but it is inferior to CBOW in memorizing word content.

Motivated by these findings, we propose a hybrid model that combines the strengths of CBOW and CMOW.

Our results show that the hybrid CBOW-CMOW-model retains CBOW's strong ability to memorize word content while at the same time substantially improving its ability to encode other linguistic information by 8%.

As a result, the hybrid also performs better on 8 out of 11 supervised downstream tasks with an average improvement of 1.2%.

Word embeddings are perceived as one of the most impactful contributions from unsupervised representation learning to natural language processing from the past few years BID10 .

Word embeddings are learned once on a large-scale stream of words.

A key benefit is that these pre-computed vectors can be re-used almost universally in many different downstream applications.

Recently, there has been increasing interest in learning universal sentence embeddings.

BID24 have shown that the best encoding architectures are based on recurrent neural networks (RNNs) BID5 BID25 or the Transformer architecture BID2 .

These techniques are, however, substantially more expensive to train and apply than word embeddings BID14 BID2 .

Their usefulness is therefore limited when fast processing of large volumes of data is critical.

More efficient encoding techniques are typically based on aggregated word embeddings such as Continuous Bag of Words (CBOW), which is a mere summation of the word vectors BID19 .

Despite CBOW's simplicity, it attains strong results on many downstream tasks.

Using sophisticated weighting schemes, the performance of aggregated word embeddings can be further increased BID0 , coming even close to strong LSTM baselines BID26 BID13 such as InferSent BID5 .

This raises the question how much benefit recurrent encoders actually provide over simple word embedding based methods BID32 .

In their analysis, BID13 suggest that the main difference may be the ability to encode word order.

In this paper, we propose an intuitive method to enhance aggregated word embeddings by word order awareness.

The major drawback of these CBOW-like approaches is that they are solely based on addition.

However, addition is not all you need.

Since it is a commutative operation, the aforementioned methods are not able to capture any notion of word order.

However, word order information is crucial for some tasks, e.g., sentiment analysis BID13 .

For instance, the following two sentences yield the exact same embedding in an addition-based word embedding aggregation technique: "The movie was not awful, it was rather great." and "The movie was not great, it was rather awful."

A classifier based on the CBOW embedding of these sentences would inevitably fail to distinguish the two different meanings (Goldberg, 2017, p. 151) .To alleviate this drawback, BID28 propose to model each word as a matrix rather than a vector, and compose multiple word embeddings via matrix multiplication rather than addition.

This so-called Compositional Matrix Space Model (CMSM) of language has powerful theoretical properties that subsume properties from vector-based models and symbolic approaches.

The most obvious advantage is the non-commutativity of matrix multiplication as opposed to addition, which results in order-aware encodings.

In contrast to vector-based word embeddings, there is so far no solution to effectively train the parameters of word matrices on large-scale unlabeled data.

Training schemes from previous work were specifically designed for sentiment analysis BID34 BID1 .

Those require complex, multi-stage initialization, which indicates the difficulty of training CMSMs.

We show that CMSMs can be trained in a similar way as the well-known CBOW model of word2vec BID19 .

We make two simple yet critical changes to the initialization strategy and training objective of CBOW.

Hence, we present the first unsupervised training scheme for CMSMs, which we call Continual Multiplication Of Words (CMOW).

We evaluate our model's capability to capture linguistic properties in the encoded text.

We find that CMOW and CBOW have properties that are complementary.

On the one hand, CBOW yields much stronger results at the word content memorization task.

CMOW, on the other hand, offers an advantage in all other linguistic probing tasks, often by a wide margin.

Thus, we propose a hybrid model to jointly learn the word vectors of CBOW and the word matrices for CMOW.Our experimental results confirm the effectiveness of our hybrid CBOW-CMOW approach.

At comparable embedding size, CBOW-CMOW retains CBOW's ability to memorize word content while at the same time improves the performance on the linguistic probing tasks by 8%.

CBOW-CMOW outperforms CBOW at 8 out of 11 supervised downstream tasks scoring only 0.6% lower on the tasks where CBOW is slightly better.

On average, the hybrid model improves the performance over CBOW by 1.2% on supervised downstream tasks, and by 0.5% on the unsupervised tasks.

In summary, our contributions are: (1) For the first time, we present an unsupervised, efficient training scheme for the Compositional Matrix Space Model.

Key elements of our scheme are an initialization strategy and training objective that are specifically designed for training CMSMs.

(2) We quantitatively demonstrate that the strengths of the resulting embedding model are complementary to classical CBOW embeddings.

(3) We successfully combine both approaches into a hybrid model that is superior to its individual parts.

After giving a brief overview of the related work, we formally introduce CBOW, CMOW, and the hybrid model in Section 3.

We describe our experimental setup and present the results in Section 4.

The results are discussed in Section 5, before we conclude.

We present an algorithm for learning the weights of the Compositional Matrix Space Model BID28 .

To the best of our knowledge, only BID34 and BID1 have addressed this.

They present complex, multi-level initialization strategies to achieve reasonable results.

Both papers train and evaluate their model on sentiment analysis datasets only, but they do not evaluate their CMSM as a general-purpose sentence encoder.

Other works have represented words as matrices as well, but unlike our work not within the framework of the CMSM.

BID11 represent only relational words as matrices.

BID29 and BID3 argue that while CMSMs are arguably more expressive than embeddings located in a vector space, the associativeness of matrix multiplication does not reflect the hierarchical structure of language.

Instead, they represent the word sequence as a tree structure.

BID29 directly represent each word as a matrix (and a vector) in a recursive neural network.

BID3 present a two-layer architecture.

In the first layer, pre-trained word embeddings are mapped to their matrix representation.

In the second layer, a non-linear function composes the constituents.

Sentence embeddings have recently become an active field of research.

A desirable property of the embeddings is that the encoded knowledge is useful in a variety of high-level downstream tasks.

To this end, BID4 and introduced an evaluation framework for sentence encoders that tests both their performance on downstream tasks as well as their ability to capture linguistic properties.

Most works focus on either i) the ability of encoders to capture appropriate semantics or on ii) training objectives that give the encoders incentive to capture those semantics.

Regarding the former, large RNNs are by far the most popular BID5 BID16 BID31 BID22 BID14 BID18 BID25 BID17 , followed by convolutional neural networks BID7 .

A third group are efficient methods that aggregate word embeddings BID33 BID0 BID23 BID26 .

Most of the methods in the latter group are word order agnostic.

Sent2Vec BID23 is an exception in the sense that they also incorporate bigrams.

Despite also employing an objective similar to CBOW, their work is very different to ours in that they still use addition as composition function.

Regarding the training objectives, there is an ongoing debate whether language modeling BID25 BID27 , machine translation BID18 , natural language inference BID5 , paraphrase identification BID33 , or a mix of many tasks BID30 ) is most appropriate for incentivizing the models to learn important aspects of language.

In our study, we focus on adapting the well-known objective from word2vec BID19 for the CMSM.

We formally present CBOW and CMOW encoders in a unified framework.

Subsequently, we discuss the training objective, the initialization strategy, and the hybrid model.

We start with a lookup table for the word matrices, i.e., an embedding, E ∈ R m×d×d , where m is the vocabulary size and d is the dimensionality of the (square) matrices.

We denote a specific word matrix of the embedding by E[·].

By ∆ ∈ { , } we denote the function that aggregates word embeddings into a sentence embedding.

Formally, given a sequence s of arbitrary length n, the sequence is encoded as DISPLAYFORM0 For ∆ = , the model becomes CBOW.

By setting ∆ = (matrix multiplication), we obtain CMOW.

Because the result of the aggregation for any prefix of the sequence is again a square matrix of shape d × d irrespective of the aggregation function, the model is well defined for any non-zero sequence length.

Thus, it can serve as a general-purpose text encoder.

Throughout the remainder of this paper, we denote the encoding step by enc DISPLAYFORM1 , where flatten concatenates the columns of the matrices to obtain a vector that can be passed to the next layer.

Motivated by its success, we employ a similar training objective as word2vec BID20 .

The objective consists of maximizing the conditional probability of a word w O in a certain context s: p(w O | s).

For a word w t at position t within a sentence, we consider the window of tokens (w t−c , . . .

, w t+c ) around that word.

From that window, a target word w O := {w t+i } , i ∈ {−c, . . . , +c} is selected.

The remaining 2c words in the window are used as the context s.

The training itself is conducted via negative sampling NEG-k, which is an efficient approximation of the softmax BID20 .

For each positive example, k negative examples (noise words) are drawn from some noise distribution P n (w).

The goal is to distinguish the target word w O from the randomly sampled noise words.

Given the encoded input words enc ∆ (s), a logistic regression with weights v ∈ R m×d 2 is conducted to predict 1 for context words and 0 for noise words.

The negative sampling training objective becomes: DISPLAYFORM0 In the original word2vec BID19 , the center word w O := w t is used as the target word.

In our experiments, however, this objective did not yield to satisfactory results.

We hypothesize that this objective is too easy to solve for a word order-aware text encoder, which diminishes incentive for the encoder to capture semantic information at the sentence level.

Instead, we propose to select a random output word w O ∼ U({w t−c , . . .

, w t+c }) from the window.

The rationale is the following: By removing the information at which position the word was removed from the window, the model is forced to build a semantically rich representation of the whole sentence.

For CMOW, modifying the objective leads to a large improvement on downstream tasks by 20.8% on average, while it does not make a difference for CBOW.

We present details in the appendix (Section B.1).

So far, only BID34 and BID1 have proposed algorithms for learning the parameters for the matrices in CMSMs.

Both works devote particular attention to the initialization, noting that a standard initialization randomly sampled from N (0, 0.1) does not work well due to the optimization problem being non-convex.

To alleviate this, the authors of both papers propose rather complicated initialization strategies based on a bag-of-words solution BID34 or incremental training, starting with two word phrases BID1 .

We instead propose an effective yet simple strategy, in which the embedding matrices are initialized close to the identity matrix.

We argue that modern optimizers based on stochastic gradient descent have proven to find good solutions to optimization problems even when those are non-convex as in optimizing the weights of deep neural networks.

CMOW is essentially a deep linear neural network with flexible layers, where each layer corresponds to a word in the sentence.

The output of the final layer is then used as an embedding for the sentence.

A subsequent classifier may expect that all embeddings come from the same distribution.

We argue that initializing the weights randomly from N (0, 0.1) or any other distribution that has most of its mass around zero is problematic in such a setting.

This includes the Glorot initialization BID8 , which was designed to alleviate the problem of vanishing gradients.

FIG0 illustrates the problem: With each multiplication, the values in the embedding become smaller (by about one order of magnitude).

This leads to the undesirable effect that short sentences have a drastically different representation than larger ones, and that the embedding values vanish for long sequences.

To prevent this problem of vanishing values, we propose an initialization strategy, where each word embedding matrix E[w] ∈ R d×d is initialized as a random deviation from the identity matrix: DISPLAYFORM0 It is intuitive and also easy to prove that the expected value of the multiplication of any number of such word embedding matrices is again the identity matrix (see Appendix A).

FIG0 shows how our initialization strategy is able to prevent vanishing values.

For training CMSMs, we observe a substantial improvement over Glorot initialization of 2.8% on average.

We present details in Section B.2 of the appendix.

The simplest combination is to train CBOW and CMOW separately and concatenate the resulting sentence embeddings at test time.

However, we did not find this approach to work well in preliminary experiments.

We conjecture that there is still a considerable overlap in the features learned by each model, which hinders better performance on downstream tasks.

To prevent redundancy in the learned features, we expose CBOW and CMOW to a shared learning signal by training them jointly.

To this end, we modify Equation 1 as follows: DISPLAYFORM1 Intuitively, the model uses logistic regression to predict the missing word from the concatenation of CBOW and CMOW embeddings.

Again, E i ∈ R m×di×di are separate word lookup tables for CBOW and CMOW, respectively, and v ∈ R m×(d

We conducted experiments to evaluate the effect of using our proposed models for training CMSMs.

In this section, we describe the experimental setup and present the results on linguistic probing as well as downstream tasks.

In order to limit the total batch size and to avoid expensive tokenization steps as much as possible, we created each batch in the following way: 1,024 sentences from the corpus are selected at random.

After tokenizing each sentence, we randomly select (without replacement) at maximum 30 words from the sentence to function as center words for a context window of size c = 5, i.e., we generate up to 30 training samples per sentence.

By padding with copies of the neutral element, we also include words as center words for which there are not enough words in the left or the right context.

For CBOW, the neutral element is the zero matrix.

For CMOW, the neutral element is the identity matrix.

We trained our models on the unlabeled UMBC news corpus BID12 , which consists of about 134 million sentences and 3 billion tokens.

Each sentence has 24.8 words on average with a standard deviation of 14.6.

Since we only draw 30 samples per sentence to limit the batch size, not all possible training examples are used in an epoch, which may result in slightly worse generalization if the model is trained for a fixed number of epochs.

We therefore use 0.1% of the 134 million sentences for validation.

After 1,000 updates (i.e., approximately every millionth training sample) the validation loss is calculated, and training terminates after 10 consecutive validations of no improvement.

Following BID20 , we limit the vocabulary to the 30,000 mostfrequent words for comparing our different methods and their variants.

Out-of-vocabulary words are discarded.

The optimization is carried out by Adam BID15 with an initial learning rate of 0.0003 and k = 20 negative samples as suggested by BID20 for rather small datasets.

For the noise distribution P n (w) we again follow BID20 and use U(w) 3/4 /Z, where Z is the partition function to normalize the distribution.

We have trained five different models: CBOW and CMOW with d = 20 and d = 28, which lead to 400-dimensional and 784-dimensional word embeddings, respectively.

We also trained the Hybrid CBOW-CMOW model with d = 20 for each component, so that the total model has 800 parameters per word in the lookup tables.

We report the results of two more models: H-CBOW is the 400-dimensional CBOW component trained in Hybrid and H-CMOW is the respective CMOW component.

Below, we compare the 800-dimensional Hybrid method to the 784-dimensional CBOW and CMOW models.

After training, only the encoder of the model enc E ∆ is retained.

We assess the capability to encode linguistic properties by evaluating on 10 linguistic probing tasks .

In particular, the Word Content (WC) task tests the ability to memorize exact words in the sentence.

Bigram Shift (BShift) analyzes the encoder's sensitivity to word order.

The downstream performance is evaluated on 10 supervised and 6 unsupervised tasks from the SentEval framework BID4 .

We use the standard evaluation configuration, where a logistic regression classifier is trained on top of the embeddings.

Considering the linguistic probing tasks (see TAB0 ), CBOW and CMOW show complementary results.

While CBOW yields the highest performance at word content memorization, CMOW outperforms CBOW at all other tasks.

Most improvements vary between 1-3 percentage points.

The difference is approximately 8 points for CoordInv and Length, and even 21 points for BShift.

The hybrid model yields scores close to or even above the better model of the two on all tasks.

In terms of relative numbers, the hybrid model improves upon CBOW in all probing tasks but WC and SOMO.

The relative improvement averaged over all tasks is 8%.

Compared to CMOW, the hybrid model shows rather small differences.

The largest loss is by 4% on the CoordInv task.

However, due to the large gain in WC (20.9%), the overall average gain is still 1.6%.We now compare the jointly trained H-CMOW and H-CBOW with their separately trained 400-dimensional counterparts.

We observe that CMOW loses most of its ability to memorize word content, while CBOW shows a slight gain.

On the other side, H-CMOW shows, among others, improvements at BShift.

TAB1 shows the scores from the supervised downstream tasks.

Comparing the 784-dimensional models, again, CBOW and CMOW seem to complement each other.

This time, however, CBOW has the upperhand, matching or outperforming CMOW on all supervised downstream tasks except TREC by up to 4 points.

On the TREC task, on the other hand, CMOW outperforms CBOW by 2.5 points.

Our jointly trained model is not more than 0.8 points below the better one of CBOW and CMOW on any of the considered supervised downstream tasks.

On 7 out of 11 supervised tasks, the joint model even improves upon the better model, and on SST2, SST5, and MRPC the difference is more than 1 point.

The average relative improvement over all tasks is 1.2%.Regarding the unsupervised downstream tasks TAB2 , CBOW is clearly superior to CMOW on all datasets by wide margins.

For example, on STS13, CBOW's score is 50% higher.

The hybrid model is able to repair this deficit, reducing the difference to 8%.

It even outperforms CBOW on two of the tasks, and yields a slight improvement of 0.5% on average over all unsupervised downstream tasks.

However, the variance in relative performance is notably larger than on the supervised downstream tasks.

Our CMOW model produces sentence embeddings that are approximately at the level of fastSent BID14 .

Thus, CMOW is a reasonable choice as a sentence encoder.

Essential to the success of our training schema for the CMOW model are two changes to the original word2vec training.

First, our initialization strategy improved the downstream performance by 2.8% compared to Glorot initialization.

Secondly, by choosing the target word of the objective at random, the performance of CMOW on downstream tasks improved by 20.8% on average.

Hence, our novel training scheme is the first that provides an effective way to obtain parameters for the Compositional Matrix Space Model of language from unlabeled, large-scale datasets.

Regarding the probing tasks, we observe that CMOW embeddings better encode the linguistic properties of sentences than CBOW.

CMOW gets reasonably close to CBOW on some downstream tasks.

However, CMOW does not in general supersede CBOW embeddings.

This can be explained by the fact that CBOW is stronger at word content memorization, which is known to highly correlate with the performance on most downstream tasks ).

Yet, CMOW has an increased performance on the TREC question type classification task (88.0 compared to 85.6).

The rationale is that this particular TREC task belongs to a class of downstream tasks that require capturing other linguistic properties apart from Word Content .Due to joint training, our hybrid model learns to pick up the best features from CBOW and CMOW simultaneously.

It enables both models to focus on their respective strengths.

This can best be seen by observing that H-CMOW almost completely loses its ability to memorize word content.

In return, H-CMOW has more capacity to learn other properties, as seen in the increase in performance at BShift and others.

A complementary behavior can be observed for H-CBOW, whose scores on Word Content are increased.

Consequently, with an 8% improvement on average, the hybrid model is substantially more linguistically informed than CBOW.

This transfers to an overall performance improvement by 1.2% on average over 11 supervised downstream tasks, with large improvements on sentiment analysis tasks (SST2, SST5), question classification (TREC), and the sentence representation benchmark (STS-B).

The improvements on these tasks is expected because they arguably depend on word order information.

On the other tasks, the differences are small.

Again, this can be explained by the fact that most tasks in the SentEval framework mainly depend on word content memorization , where the hybrid model does not improve upon CBOW.Please note, the models in our study do not represent the state-of-the-art for sentence embeddings.

BID24 show that better scores are achieved by LSTMs and Transformer models, but also by averaging word embedding from fastText BID21 .

These embeddings were trained on the CBOW objective, and are thus very similar to our models.

However, they are trained on large corpora (600B tokens vs 3B in our study), use large vocabularies (2M vs 30k in our study), and incorporate numerous tricks to further enhance the quality of their models: word subsampling, subword-information, phrase representation, n-gram representations, position-dependent weighting, and corpus de-duplication.

In the present study, we focus on comparing CBOW, CMOW, and the hybrid model in a scenario where we have full control over the independent variables.

To single out the effect of the independent variables better, we keep our models relatively simple.

Our analysis yields interesting insights on what our models learn when trained separately or jointly, which we consider more valuable in the long term for the research field of text representation learning.

We offer an efficient order-aware extension to embedding algorithms from the bag-of-words family.

Our 784-dimensional CMOW embeddings can be computed at the same rate as CBOW embeddings.

We empirically measured in our experiments 71k for CMOW vs. 61k for CBOW in terms of encoding sentences per second.

This is because of the fast implementation of matrix multiplication in GPUs.

It allows us to encode sentences approximately 5 times faster than using a simple Elman RNN of the same size (12k per second).

Our matrix embedding approach also offers valuable theoretical advantages over RNNs and other autoregressive models.

Matrix multiplication is associative such that only log 2 n sequential steps are necessary to encode a sequence of size n. Besides parallelization, also dynamic programming techniques can be employed to further reduce the number of matrix multiplication steps, e. g., by pre-computing frequent bigrams.

We therefore expect our matrix embedding approach to be specifically well-suited for large-scale, time-sensitive text encoding applications.

Our hybrid model serves as a blueprint for using CMOW in conjunction with other existing embedding techniques such as fastText BID21 .

We have presented the first efficient, unsupervised learning scheme for the word order aware Compositional Matrix Space Model.

We showed that the resulting sentence embeddings capture linguistic features that are complementary to CBOW embeddings.

We thereupon presented a hybrid model with CBOW that is able to combine the complementary strengths of both models to yield an improved downstream task performance, in particular on tasks that depend on word order information.

Thus, our model narrows the gap in terms of representational power between simple word embedding based sentence encoders and highly non-linear recurrent sentence encoders.

We made the code for this paper available at https://github.com/florianmai/ word2mat.

The statement that we formally proof is the following.

For any sequence s = s 1 . . .

s n : DISPLAYFORM0 The basis (n = 1) follows trivially due to the expected value of each entry being the mean of the normal distribution.

For the induction step, let E[ DISPLAYFORM1

In Section 3.2, we describe a more general training objective than the classical CBOW objective from BID19 .

The original objective always sets the center word from the window of tokens (w t−c , . . . , w t+c ) as target word, w O = w t .

In preliminary experiments, this did not yield satisfactory results.

We believe that this objective is too simple for learning sentence embeddings that capture semantic information.

Therefore, we experimented a variant where the target word is sampled randomly from a uniform distribution, w O := U({w t−c , . . .

, w t+c }).To test the effectiveness of this modified objective, we evaluate it with the same experimental setup as described in Section 4.

TAB3 lists the results on the linguistic probing tasks.

CMOW-C and CBOW-C refer to the models where the center word is used as the target.

CMOW-R and CBOW-R refer to the models where the target word is sampled randomly.

While CMOW-R and CMOW-C perform comparably on most probing tasks, CMOW-C yields 5 points lower scores on WordContent and BigramShift.

Consequently, CMOW-R also outperforms CMOW-C on 10 out of 11 supervised downstream tasks and on all unsupervised downstream tasks, as shown in TAB4 , respectively.

On average over all downstream tasks, the relative improvement is 20.8%.

For CBOW, the scores on downstream tasks increase on some tasks and decrease on others.

The differences are miniscule.

On average over all 16 downstream tasks, CBOW-R scores 0.1% lower than CBOW-C. In Section 3.3, we present a novel random initialization strategy.

We argue why it is more adequate for training CMSMs than classic strategies that initialize all parameters with random values close to zero, and use it in our experiments to train CMOW.To verify the effectiveness of our initialization strategy empirically, we evaluate it with the same experimental setup as described in Section 4.

The only difference is the initialization strategy, where we include Glorot initialization BID8 and the standard initialization from N (0, 0.1).

TAB6 shows the results on the probing tasks.

While Glorot achieves slightly better results on BShift and TopConst, CMOW's ability to memorize word content is improved by a wide margin by our initialization strategy.

This again affects the downstream performance as shown in TAB7 and 9, respectively: 7 out of 11 supervised downstream tasks and 4 out of 5 unsupervised downstream tasks improve.

On average, the relative improvement of our strategy compared to Glorot initialization is 2.8%.

<|TLDR|>

@highlight

We present a novel training scheme for efficiently obtaining order-aware sentence representations.