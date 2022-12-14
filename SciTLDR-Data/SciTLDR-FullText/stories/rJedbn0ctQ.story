We propose a simple and robust training-free approach for building sentence representations.

Inspired by the Gram-Schmidt Process in geometric theory, we build an orthogonal basis of the subspace spanned by a word and its surrounding context in a sentence.

We model the semantic meaning of a word in a sentence based on two aspects.

One is its relatedness to the word vector subspace already spanned by its contextual words.

The other is its novel semantic meaning which shall be introduced as a new basis vector perpendicular to this existing subspace.

Following this motivation, we develop an innovative method based on orthogonal basis to combine pre-trained word embeddings into sentence representation.

This approach requires zero training and zero parameters, along with efficient inference performance.

We evaluate our approach on 11 downstream NLP tasks.

Experimental results show that our model outperforms all existing zero-training alternatives in all the tasks and it is competitive to other approaches relying on either large amounts of labelled data or prolonged training time.

The concept of word embeddings has been prevalent in NLP community in recent years, as they can characterize semantic similarity between any pair of words, achieving promising results in a large number of NLP tasks BID14 BID18 BID20 .

However, due to the hierarchical nature of human language, it is not sufficient to comprehend text solely based on isolated understanding of each word.

This has prompted a recent rise in search for semantically robust embeddings for longer pieces of text, such as sentences and paragraphs.

Based on learning paradigms, the existing approaches to sentence embeddings can be categorized into two categories: i) parameterized methods and ii) non-parameterized methods.

Parameterized sentence embeddings.

These models are parameterized and require training to optimize their parameters.

SkipThought BID11 is an encoder-decoder model that predicts adjacent sentences.

BID15 proposes an unsupervised model, Sent2Vec, to learn an n-gram feature in a sentence to predict the center word from the surrounding context.

Quick thoughts (QT) BID12 replaces the encoder with a classifier to predict context sentences from candidate sequences.

BID10 proposes?? la carte to learn a linear mapping to reconstruct the center word from its context.

BID5 generates the sentence encoder InferSent using Natural Language Inference (NLI) dataset.

Universal Sentence Encoder utilizes the transformer BID24 for sentence embeddings.

The model is first trained on large scale of unsupervised data from Wikipedia and forums, and then trained on the Stanford Natural Language Inference (SNLI) dataset.

BID27 propose the gated recurrent averaging network (GRAN), which is trained on Paraphrase Database (PPDB) and English Wikipedia.

BID23 leverages a multi-task learning framework to generate sentence embeddings.

BID28 learns the paraphrastic sentence representations as the simple average of updated word embeddings.

Non-parameterized sentence embedding.

Recent work BID0 shows that, surprisingly, a weighted sum or transformation of word representations can outperform many sophisticated neural network structures in sentence embedding tasks.

These methods are parameter-free and require no further training upon pre-trained word vectors.

BID0 constructs a sentence embedding called SIF as a sum of pre-trained word embeddings, weighted by reverse document frequency.

BID19 concatenates different power mean word embeddings as a sentence vector in p-mean.

As these methods do not have a parameterized model, they can be easily adapted to novel text domains with both fast inference speed and high-quality sentence embeddings.

In view of this trend, our work aims to further advance the frontier of this group and make its new state-of-the-art.

In this paper, we propose a novel sentence embedding algorithm, Geometric Embedding (GEM), based entirely on the geometric structure of word embedding space.

Given a d-dim word embedding matrix A ??? R d??n for a sentence with n words, any linear combination of the sentence's word embeddings lies in the subspace spanned by the n word vectors.

We analyze the geometric structure of this subspace in R d .

When we consider the words in a sentence one-by-one in order, each word may bring in a novel orthogonal basis to the existing subspace.

This new basis can be considered as the new semantic meaning brought in by this word, while the length of projection in this direction can indicate the intensity of this new meaning.

It follows that a word with a strong intensity should have a larger influence in the sentence's meaning.

Thus, these intensities can be converted into weights to linearly combine all word embeddings to obtain the sentence embedding.

In this paper, we theoretically frame the above approach in a QR factorization of the word embedding matrix A. Furthermore, since the meaning and importance of a word largely depends on its close neighborhood, we propose the sliding-window QR factorization method to capture the context of a word and characterize its significance within the context.

In the last step, we adapt a similar approach as BID0 to remove top principal vectors before generating the final sentence embedding.

This step is to ensure commonly shared background components, e.g. stop words, do not bias sentence similarity comparison.

As we build a new orthogonal basis for each sentence, we propose to have disparate background components for each sentence.

This motivates us to put forward a sentence-specific principal vector removal method, leading to better empirical results.

We evaluate our algorithm on 11 NLP tasks.

In all of these tasks, our algorithm outperforms all non-parameterized methods and many parameterized approaches.

For example, compared to SIF BID0 , the performance is boosted by 5.5% on STS benchmark dataset, and by 2.5% on SST dataset.

Plus, the running time of our model compares favorably with existing models.

The rest of this paper is organized as following.

In Section 2, we describe our sentence embedding algorithm GEM.

We evaluate our model on various tasks in Section 3 and Section 4.

Finally, we summarize our work in Section 5.

Let us consider the idea of word embeddings BID14 , where a word w i is projected as a vector v wi ??? R d .

Any sequence of words can be viewed as a subspace in R d spanned by its word vectors.

Before the appearance of the ith word, S is a subspace in R d spanned by {v w1 , v w2 , ..., v wi???1 }.

Its orthonormal basis is {q 1 , q 2 , ..., q i???1 }.

The embedding v wi of the ith word w i can be decomposed into DISPLAYFORM0 (1) where i???1 j=1 r j q j is the part in v wi that resides in subspace S, and q i is orthogonal to S and is to be added to S. The above algorithm is also known as Gram-Schmidt Process.

In the case of rank deficiency, i.e., v wi is already a linear combination of {q 1 , q 2 , ...q i???1 }, q i is a zero vector and r i = 0.

In matrix form, this process is also known as QR factorization, defined as follows.

QR factorization.

Define an embedding matrix of n words as A = [A :,1 , A :,2 , ..., A :,n ] ??? R d??n , where A :,i is the embedding of the ith word w i in a word sequence (w 1 , . . .

, w i , . . . , w n ).

A ??? R d??n can be factorized into A = QR, where the non-zero columns in Q ??? R d??n are the orthonormal basis, and R ??? R n??n is an upper triangular matrix.

The process above computes the novel semantic meaning of a word w.r.t all preceding words.

As the meaning of a word influences and is influenced by its close neighbors, we now calculate the novel orthogonal basis vector q i of each word w i in its neighborhood, rather than only w.r.t the preceding words.

Definition 1 (Contextual Window Matrix) Given a word w i , and its m-neighborhood window inside the sentence (w i???m , . . . , w i???1 , w i , w i+1 , . . .

, w i+m ) , define the contextual window matrix of word w i as: DISPLAYFORM1 Here we shuffle v wi to the end of S i to compute its novel semantic information compared with its context.

Now the QR factorization of S i is DISPLAYFORM2 Note that q i is the last column of Q i , which is also the new orthogonal basis vector to this contextual window matrix.

Next, in order to generate the embedding for a sentence, we will assign a weight to each of its words.

This weight should characterize how much new and important information a word brings to the sentence.

The previous process yields the orthogonal basis vector q i .

We propose that q i represents the novel semantic meaning brought by word w i .

We will now discuss how to quantify i) the novelty of q i to other meanings in w i , ii) the significance of q i to its context, and iii) the corpus-wise uniqueness of q i w.r.t the whole corpus.

We propose that a word w i is more important to a sentence if its novel orthogonal basis vector q i is a large component in v wi .

This can be quantified as a novelty score: DISPLAYFORM0 where r is the last column of R i , and r ???1 is the last element of r.

Connection to least square.

From QR factorization theory, the novel orthogonal basis q i is also the normalized residual in the least square problem min Cx ??? v wi 2 , i.e. q It follows that ?? n is the exponential of the normalized distance between v wi and the subspace spanned by its context.

The significance of a word is related to how semantically aligned it is to the meaning of its context.

To identify principal directions, i.e. meanings, in the contextual window matrix S i , we employ Singular Value Decomposition.

The columns of U , {U :,j } n j=1 , are an orthonormal basis of A's columns subspace and we propose that they represent a set of semantic meanings from the context.

Their corresponding singular values {?? j } n j=1 , denoted by ??(A), represent the importance associated with {U :,j } n j=1 .

The SVD of w i 's contextual window matrix is DISPLAYFORM0 Intuitively, a word is more important if its novel semantic meaning has a better alignment with more principal meanings in its contextual window.

This can be quantified as ??(S i ) (q DISPLAYFORM1 , where denotes element-wise product.

Therefore, we define the significance of w i in its context to be: DISPLAYFORM2 It turns out ?? s can be rewritten as DISPLAYFORM3 We use the fact that V i is an orthogonal matrix and q i is orthogonal to all but the last column of S i , v wi .

Therefore, ?? s is essentially the distance between w i and the context hyper-plane, normalized by the context size.

Similar to the idea of inverse document frequency (IDF) BID22 , a word that is commonly present in the corpus is likely to be a stop word, thus its corpus-wise uniqueness is small.

In our solution, we compute the principal directions of the corpus and then measure their alignment with the novel orthogonal basis vector q i .

If there is a high alignment, w i will be assigned a relatively low corpus-wise uniqueness score, and vice versa.

As proposed in BID0 , given a corpus containing a set of sentences, each sentence embedding is first computed as a linear combination of its word embeddings, thus generating a sentence embedding matrix X = [c 1 , c 2 , . . .

, c N ] ??? R d??N for a corpus S with N sentences.

Then principal vectors of X are computed.

In comparison, we do not form the sentence embedding matrix X after we finalize the sentence embedding.

Instead, we obtain an intermediate coarse-grained sentence embedding matrix X c = [g 1 , . . .

, g N ] as follows.

Suppose the SVD of the sentence matrix of the ith sentence is S = [v w1 , . . .

, v wn ] = U ??V T .

Then the coarse-grained embedding for the ith sentence is defined as: DISPLAYFORM0 where f (?? j ) is a monotonically increasing function.

We then compute the top K principal vectors DISPLAYFORM1

In contrast to BID0 , we select different principal vectors of X c for each sentence, as different sentences may have disparate alignments with the corpus.

For each sentence, {d 1 , ..., d K } are re-ranked in descending order of their correlation with sentence matrix S. The correlation is defined as DISPLAYFORM0 Next, the top h principal vectors after re-ranking based on DISPLAYFORM1 Finally, a word w i with new semantic meaning vector q i in this sentence will be assigned a corpuswise uniqueness score: DISPLAYFORM2 This ensures that common stop words will have their effect diminished since their embeddings are closely aligned with the corpus' principal directions.

A sentence vector c s is computed as a weighted sum of its word embeddings, where the weights come from three scores: a novelty score (?? n ), a significance score (?? s ) and a corpus-wise uniqueness score (?? u ).

DISPLAYFORM0 We provide a theoretical explanation of Equation FORMULA14 in Appendix.

Sentence-Dependent Removal of Principal Components.

BID0 shows that given a set of sentence vectors, removing projections onto the principal components of the spanned subspace can significantly enhance the performance on semantic similarity task.

However, as each sentence may have a different semantic meaning, it could be sub-optimal to remove the same set of principal components from all sentences.

Therefore, we propose the sentence-dependent principal component removal (SDR), where we rerank top principal vectors based on correlation with each sentence.

Using the method from Section 2.4.2, we obtain D = {d t1 , ..., d tr } for a sentence s. The final embedding of this sentence is then computed as: DISPLAYFORM1 Ablation experiments show that sentence-dependent principal component removal can achieve better result.

The complete algorithm is summarized in Algorithm 1 with an illustration in FIG1 .

We evaluate our model on the STS Benchmark BID2 , a sentence-level semantic similarity dataset from SemEval and SEM STS.

The goal for a model is to predict a similarity score of two sentences given a sentence pair.

The evaluation is by the Pearson's coefficient r between humanlabeled similarity (0 -5 points) and predictions.

Experimental settings.

We report two versions of our model, one only using GloVe word vectors (GEM + GloVe), and the other using word vectors concatenated from LexVec, fastText and PSL BID29 ) (GEM + L.F.P).

The final similarity score is computed as an inner product of Form matrix S ??? R d??n , S :,j = v wj and w j is the jth word in s 5:The SVD is S = U ??V DISPLAYFORM0 The ith column of the coarse-grained sentence embedding matrix X c :,i is U (??(S)) Re-rank {d 1 , ..., for word w i in s do 13: DISPLAYFORM1 DISPLAYFORM2 is the contextual window matrix of w i .14: DISPLAYFORM3 DISPLAYFORM4 16: Parameterized models Reddit + SNLI 81.4 78.2 GRAN BID27 81.8 76.4 InferSent BID5 80.1 75.8 Sent2Vec BID15 78.7 75.5 Paragram-Phrase (Wieting et al., 2015a) 73.9 73.2 Table 2 : MAP on CQA subtask B normalized sentence vectors.

Since our model is non-parameterized, it does not utilize any information from the dev set when evaluating on the test set and vice versa.

Hyper-parameters are chosen at m = 7, h = 17, K = 45, and t = 3 by conducing hyper-parameters search on dev set.

Results on the dev and test set are reported in TAB5 .

As shown, on the test set, our model has a 5.5% higher score compared with another non-parameterized model SIF, and 25.5% higher than the baseline of averaging L.F.P word vectors.

It also outperforms most parameterized models including GRAN, InferSent, and Sent2Vec.

Of all evaluated models, our model only ranks second to Reddit + SNLI, which is trained on the Reddit conversations dataset (600 million sentence pairs) and SNLI (570k sentence pairs).

In comparison, our proposed method requires no external data and no training.

Table 3 : Results on supervised tasks.

Sentence embeddings are fixed for downstream supervised tasks.

Best results for each task are underlined, best results from models in the same category are in bold.

SIF results are extracted from Arora et al. FORMULA1 and BID19 , and some training time is collected from BID12 .

DISPLAYFORM5

We evaluate our model on subtask B of the SemEval Community Question Answering (CQA) task, another semantic similarity dataset.

Given an original question Q o and a set of the first ten related questions (Q 1 , ..., Q 10 ) retrieved by a search engine, the model is expected to re-rank the related questions according to their similarity with respect to the original question.

Each retrieved question Q i is labelled "PerfectMatch", "Relevant" or "Irrelevant", with respect to Q o .

Mean average precision (MAP) is used as the evaluation measure.

We encode each question text into a unit vector u. Retrieved questions {Q i } 10 i=1 are ranked according to their cosine similarity with Q o .

Results are shown in Table 2 .

For comparison, we include results from the best models in 2017 competition: SimBow BID4 , KeLP BID7 , and Reddit + SNLI tuned.

Note that all three benchmark models require training, and SimBow and KeLP leverage optional features including usage of comments and user profiles.

In comparison, our model only uses the question text without any training.

Our model clearly outperforms both Reddit + SNLI tuned and SimBow-primary, and on par with KeLP model.

We further test our model on nine supervised tasks, including seven classification tasks: movie review (MR) BID17 , Stanford Sentiment Treebank (SST) BID21 , questiontype classification (TREC) BID25 , opinion polarity (MPQA) BID26 , product reviews (CR) BID9 , subjectivity/objectivity classification (SUBJ) BID16 and paraphrase identification (MRPC) BID6 .

We also evaluate on SICK similarity (SICK-R), the SICK entailment (SICK-E) BID13 .

The sentence embeddings generated are fixed and only the downstream task-specific neural structure is learned.

The four hyper-parameters are chosen the same as in STS benchmark experiment.

Results are in Table 3.

GEM outperforms all non-parameterized sentence embedding models, including SIF, p-mean BID19 , and BOW on GloVe.

It also compares favorably with most of parameterized models, including?? la carte BID10 , FastSent BID8 , InferSent, QT, Sent2Vec, SkipThought-LN (with layer normalization) BID11 , SDAE BID8 , STN BID23 and USE .

Note that sentence representations generated by GEM have much smaller dimension compared to most of benchmark models, and the subsequent neural structure has fewer learnable parameters.

The fact that GEM does well on several classification tasks (e.g. TREC and SUBJ) indicates that the proposed weight scheme is able to recognize important words in the sentence.

Also, GEM's competitive performance on sentiment tasks shows that exploiting the geometric structures of two sentence subspaces is beneficial.

Figure 2: Sensitivity tests on four hyper-parameters, the window size m in contextual window matrix, the number of candidate principal components K, the number of principal components to remove h, and the exponential power of singular value in coarse sentence embedding.

Ablation Study.

As shown in in Table 4 , every GEM weight (?? n , ?? s , ?? u ) and proposed principal components removal methods contribute to the performance.

As listed on the left, adding GEM weights improves the score by 8.6% on STS dataset compared with averaging three concatenated word vectors.

The sentence-dependent principal component removal (SDR) proposed in GEM improves 0.3% compared to directly removing the top h corpus principal components (SIR).

Using GEM weights and SDR together yields an overall improvement of 19.7%.

As shown on the right in Table 4 , every weight contributes to the performance of our model.

For example, three weights altogether improve the score in SUBJ task by 0.38% compared with only using ?? n .

Sensitivity Study.

We evaluate the effect of all four hyper-parameters in our model: the window size m in the contextual window matrix, the number of candidate principal components K, the number of principal components to remove h, and the power of the singular value in coarse sentence embedding, i.e. the power t in f (?? j ) = ?? t j in Equation FORMULA9 .

We sweep the hyper-parameters and test on STSB dev set, SUBJ, and MPQA.

Unspecified parameters are fixed at m = 7, K = 45, h = 17 and t = 3.

As shown in Figure 2 , our model is quite robust with respect to hyper-parameters.

Inference speed.

We also compare the inference speed of our algorithm on the STSB test set with the benchmark models SkipThought and InferSent.

SkipThought and InferSent are run on a NVIDIA Tesla P100, and our model is run on a CPU (Intel R Xeon R CPU E5-2690 v4 @2.60GHz).

For fair comparison, batch size in InferSent and SkipThought is set to be 1.

The results are shown in TAB9 .

It shows that without acceleration from GPU, our model is still faster than InferSent and is 54% faster than SkipThought.

We proposed a simple non-parameterized method 1 to generate sentence embeddings, based entirely on the geometric structure of the subspace spanned by word embeddings.

Our sentence embedding evolves from the new orthogonal basis vector brought in by each word, which represents novel semantic meaning.

The evaluation shows that our method not only sets up the new state-of-the-art of non-parameterized models but also performs competitively when compared with models requiring either large amount of training data or prolonged training time.

In future work, we plan to consider multi-characters, i.e. subwords, into the model and explore other geometric structures in sentences.

The novelty score (?? n ), significance score (?? s ) and corpus-wise uniqueness score (?? u ) are larger when a word w has relatively rare appearance in the corpus and can bring in new and important semantic meaning to the sentence.

Following the section 3 in BID0 , we can use the probability of a word w emitted from sentence s in a dynamic process to explain eq. (9) and put this as following Theorem with its proof provided below.

Theorem 1.

Suppose the probability that word w i is emitted from sentence s is 2 : DISPLAYFORM0 where c s is the sentence embedding, Z = wi???V exp( c s , v wi ) and V denotes the vocabulary.

Then when Z is sufficiently large, the MLE for c s is: The joint probability of sentence s is then To simplify the notation, let ?? = ?? n + ?? s + ?? u .

It follows that the log likelihood f (w i ) of word w i emitted from sentence s is given by f wi (c s ) = log( exp( c s , v wi ) Z + e ????? ) ??? log(N )

@highlight

A simple and training-free approach for sentence embeddings with competitive performance compared with sophisticated models requiring either large amount of training data or prolonged training time.

@highlight

Presented a new training-free way of generating sentence embedding with systematic analysis

@highlight

Proposes a new geometry-based method for sentence embedding from word embedding vectors by quantifying the novelty, significance, and corpus-uniqueness of each word

@highlight

This paper explores sentence embedding based on orthogonal decomposition of the spanned space by word embeddings