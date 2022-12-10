Most deep learning for NLP represents each word with a single point or single-mode region in semantic space, while the existing multi-mode word embeddings cannot represent longer word sequences like phrases or sentences.

We introduce a phrase representation (also applicable to sentences) where each phrase has a distinct set of multi-mode codebook embeddings to capture different semantic facets of the phrase's meaning.

The codebook embeddings can be viewed as the cluster centers which summarize the distribution of possibly co-occurring words in a pre-trained word embedding space.

We propose an end-to-end trainable neural model that directly predicts the set of cluster centers from the input text sequence (e.g., a phrase or a sentence) during test time.

We find that the per-phrase/sentence codebook embeddings not only provide a more interpretable semantic representation but also outperform strong baselines (by a large margin in some tasks) on benchmark datasets for unsupervised phrase similarity, sentence similarity, hypernym detection, and extractive summarization.

Many widely-applicable NLP models learn a representation from only co-occurrence statistics in the raw text without any supervision.

Examples include word embedding like word2vec (Mikolov et al., 2013) or GloVe (Pennington et al., 2014) , sentence embeddings like skip-thoughts (Kiros et al., 2015) , and contextualized word embedding like ELMo (Peters et al., 2018) and BERT (Devlin et al., 2019) .

Most of these models use a single embedding to represent one sentence or one phrase and can only provide symmetric similarity measurement when no annotation is available.

However, a word or phrase might have multiple senses, and a sentence can involve multiple topics, which are hard to analyze based on a single embedding without supervision.

To address the issue, word sense induction methods (Lau et al., 2012) and recent multi-mode word embeddings (Neelakantan et al., 2014; Athiwaratkun & Wilson, 2017; Singh et al., 2018) represent each target word as multiple points or regions in a distributional semantic space by (explicitly or implicitly) clustering all the words appearing beside the target word.

In Figure 1 , the multi-mode representation of real property is illustrated as an example.

Real property can be observed in legal documents where it usually means a real estate, while a real property can also mean a true characteristic in philosophic discussions.

The previous approaches discover those senses by clustering observed neighboring words (e.g., company and tax).

In contrast with topic modeling like LDA (Blei et al., 2003) , the approaches need to solve a distinct clustering problem for every target word while topic modeling finds a single set of clusters by clustering all the words in the corpus.

Extending these multi-mode representations to arbitrary sequences like phrases or sentences is difficult due to two efficiency challenges.

First, there are usually many more unique phrases and sentences in a corpus than there are words, while the number of parameters for clustering-based approaches is O(|V | × |K| × |E|), where |V | is number of unique sequences, |K| is number of modes/clusters, and |E| is the number of embedding dimensions.

Estimating and storing such a large number of parameters take time and space.

More important, many unique sequences imply much fewer co-occurring words to be clustered for each sequence, especially for long sequences Figure 1 : The target phrase real property is represented by four clustering centers.

The previous work discovers the four modes by finding clustering centers which well compress the embedding of observed co-occurring words.

Instead, our compositional model learns to predict the embeddings of cluster centers from the sequence of words in the target phrase so as to reconstruct the (unseen) co-occurring distribution well.

like sentences, so an effective model needs to overcome this sample efficient challenge (i.e., sparseness in the co-occurring statistics).

However, clustering approaches often have too many parameters to learn the compositional meaning of each sequence without overfitting.

Nevertheless, the sentences (or phrases) sharing multiple words tend to have similar cluster centers, so we should be able to compress many redundant parameters in these local clustering problems to circumvent the challenges.

In this work, we adopt a neural encoder and decoder to achieve the goal.

As shown in Figure 1 , instead of clustering co-occurring words beside a target sequence at test time as in previous approaches, we learn a mapping between the target sequence (i.e., phrases or sentences) and the corresponding cluster centers during training so that we can directly predict those cluster centers using a single forward pass of the neural network for an arbitrary unseen input sequences during testing.

To allow the neural network to generate the cluster centers in an arbitrary order, we use a nonnegative and sparse coefficient matrix to dynamically match the sequence of predicted cluster centers and the observed set of co-occurring word embeddings during training.

After the coefficient matrix is estimated for each input sequence, the gradients are back-propagated to cluster centers (i.e., codebook embeddings) and weights of decoder and encoder, which allows us to train the whole model jointly and end-to-end.

In experiments, we show that the proposed model captures the compositional meanings of words in unsupervised phrase similarity tasks much better than averaging their (contextualized) word embeddings, strong baselines that are widely used in practice.

In addition to similarity, our model can also measure asymmetric relations like hypernymy without any supervision.

Furthermore, the multimode representation is shown to outperform the single-mode alternatives in sentence representation, especially as demonstrated in our extractive summarization experiment.

In this section, we first formalize our training setup in Section 2.1.

Next, our objective function and the architecture of our prediction mode will be described in Section 2.2 and Section 2.3, respectively.

The approach is summarized in Figure 2 using an example sentence.

Figure 2: Our model for sentence representation.

We represent each sentence as multiple codebook embeddings (i.e., clustering centers) predicted by our sequence to embeddings model.

Our loss encourages the model to generate codebook embeddings whose linear combination can well reconstruct the embeddings of co-occurring words (e.g., Music), while not able to reconstruct the negatively sampled words (i.e., the co-occurring words from other sentences) to avoid predicting common topics which co-occur with every sentence (e.g., is in this example).

We express tth sequence of words in the corpus as I t = w xt ...w yt <eos>, where x t and y t are the start and end position of the target sequence, respectively, and <eos> is the end of sequence symbol.

We assume neighboring words beside each target phrase or sentence are related to some aspects of the sequence, so given I t as input, our training signal is to reconstruct a set of neighboring words,

is a fixed window size.

For sentence representation, N t is the set of all words in the previous and the next sentence.

Notice that the training signal for sentences and phrases are different, which means we need to train one model for phrase and one model for sentence if both representations are desired.

Since there are not many co-occurring words for a long sequence (none are observed for unseen testing sequences), our goal is to cluster the set of words which could "possibly" occur beside the sequence instead of the actual occurring words in the training corpus (e.g., the hidden co-occurring distribution instead of green and underlined words in Figure 2 ).

Although most of the possibly co-occurring words are not observed in the corpus, we can still learn to predict them by observing co-occurring words from similar sequences.

To focus on the semantics rather than syntax, we view the co-occurring words as a set rather than a sequence as in skip-thoughts (Kiros et al., 2015) .

Notice that our model considers the word order information in the input sequence I t , but ignores the order of the co-occurring words N t .

In this work, we model the distribution of co-occurring words in a pre-trained word embedding space.

The embeddings of co-occurring words N t are arranged into a matrix W (N t ) = [w t j ] j=1...|Nt| with size |E| × |N t |, where |E| is the dimension of pre-trained word embedding, and each of its column w t j is a normalized word embedding whose 2-norm is 1.

The normalization makes the cosine distance between two words become half of their Euclidean distance.

Similarly, we denote the predicted cluster centers c t k of the input sequence I t as a |E| × K matrix

where F is our neural network model and K is the number of clusters.

We fix the number of clusters K in this work to simplify the design of our prediction model and how it is applied to downstream tasks.

The effect of different cluster numbers will be discussed in the experimental section.

The reconstruction loss of k-means clustering in the word embedding space (Singh et al., 2018) can be written as

, where M k,j = 1 if the jth word belongs to the k cluster and 0 otherwise.

That is, M is a permutation matrix which matches the cluster centers and co-occurring words and allow the neural network to generate the centers in an arbitrary order.

Non-negative sparse coding (NNSC) (Hoyer, 2002) relaxes the constraints by allowing the coefficient M k,j to be a positive value but encouraging it to be 0.

In this work, we adopt the relaxation because for all neural architectures (including transformers and LSTMs) we tried, the models using a NNSC loss learn to generates diverse K cluster centers while the predicted cluster centers using the kmeans loss collapse to much fewer modes which cannot well capture the conditional co-occurrence distribution.

We hypothesize that it is because a NNSC loss is smoother and easier to be optimized for a neural network, while finding the nearest cluster center in the kmeans loss cannot stably encourage predicted clusters to play different roles for reconstructing the embeddings of observed co-occurring words.

Using NNSC, we define our reconstruction error as

where λ is a hyper-parameter controlling the sparsity of M .

We force the coefficient value M k,j ≤ 1 to avoid the neural network learning to predict centers with small magnitudes which makes the optimal values of M k,j large and unstable.

Having multiple outputs and estimating the permutation between the prediction and ground truth words is often computationally expensive (Stern et al., 2018; Qin et al., 2019) .

However, the proposed loss is efficient because we minimize the L2 distance in a pre-trained embedding space as in Kumar & Tsvetkov (2019) rather than using softmax, and M Ot can be efficiently estimated on the fly using convex optimization (we use RMSprop (Tieleman & Hinton, 2012) in our implementation).

After M Ot is estimated, we can treat it as a constant and back-propagate the gradients to our neural network to achieve end-to-end training.

To prevent the neural network from predicting the same global topics regardless of the input, our loss function for tth sequence is defined as

where N rt is a set of co-occurring words of a randomly sampled sequence I rt .

In our experiment, we use SGD to solve F = arg min F t L t (F ).

Our method could be viewed as a generalization of Word2Vec (Mikolov et al., 2013 ) that can encode the compositional meaning of the words and decode multiple embeddings.

Our neural network architecture is similar to transformation-based sequence to sequence (seq2seq) model (Vaswani et al., 2017) .

We use the same encoder T E(I t ), which transforms the input sequence into a contextualized embeddings

where the goal of the encoder is to map the sentences which are likely to have similar co-occuring word distribution closer together.

Different from the typical seq2seq model (Sutskever et al., 2014; Vaswani et al., 2017) , our decoder does not need to make discrete decisions because our outputs are a sequence of embeddings instead of words.

This allows us to predict all the codebook embeddings in a single forward pass while still well capturing the dependency between output without the need of auto-regressive decoding.

Similar to BERT, we treat the embedding of <eos> as the sentence representation.

To make different codebook embeddings capture different aspects, we pass the embeddings of <eos> to different linear layers L k before becoming the input of the decoder T D. Specifically, the codebook embeddings

We find that removing the attention on the e xt ...e yt , contextualized word embeddings from the encoder, significantly increases our validation loss for sentence representation because there are often too many facets to be compressed into a single embedding.

On the other hand, the attention does not change the performance of phrase representation too much, and we remove the attention connection between encoder and decoder (i.e., encoder and decoder have the same architecture) when evaluating our phrase representation.

Notice that the framework is flexible.

We can replace the encoder and decoder with other architectures.

Besides transformers, we also try (bi-)LSTMs in our experiments.

This flexibility also allows us to incorporate other input features (e.g., the author who writes the sentence).

We first visualize the cluster centers predicted by our model in Table 1 (like we visualize the meaning of the red cluster center in Figure 2 using the word song or Music).

The centers summarize the target sequence well and more codebook embeddings capture more semantic facets of a phrase or a sentence.

Due to the difficulty of evaluating the topics conditioned on the input sequence using the classic metrics for global topic modeling, we show that the codebook embeddings can be used to improve the performances of various unsupervised semantic tasks, which indirectly measures the quality of the generated topics.

We use the cased version (840B) of pre-trained GloVe embedding (Pennington et al., 2014) for sentence representation and use the uncased version (42B) for phrase representation.

2 Our model is trained on Wikipedia 2016 while the stop words are removed from the set of co-occurring words.

In the phrase experiments, we only consider noun phrases, and their boundaries are extracted by applying simple regular expression rules to POS tags before training.

The sentence boundaries and POS tags are detected using spaCy.

3 Our models do not need resources such as PPDB (Pavlick et al., 2015) or other multi-lingual resources, so our models are compared with the baselines that only use the raw text and sentence/phrase boundaries.

This setting is particularly practical for the domains with low resources such as scientific literature.

To control the effect of embedding size, we set the number of dimensions in our transformers as the GloVe embedding size (300).

Limited by computational resources, we train all the models using one modern GPU within a week.

Because of the relatively small model size, we find that our models underfit the data after a week (i.e., the training loss is very close to the validation loss).

It is hard to make a fair comparison with BERT (Devlin et al., 2019) .

BERT is trained on a masked language modeling loss, which preserves more syntax information and can produce an effective pretrained embedding for many supervised downstream tasks.

We report the performance of the BERT base model, which is still trained using more parameters, more output dimensions, larger corpus, and more computational resources during training compared with our models.

Furthermore, BERT uses a word piece model to alleviate the out-of-vocabulary problem.

Nevertheless, we still provide its unsupervised performances based on cosine similarity as a reference.

Semeval 2013 task 5(a) English (Korkontzelos et al., 2013) and Turney 2012 (Turney, 2012) are two standard benchmarks for evaluating phrase similarity (Yu & Dredze, 2015; Huang et al., 2017) .

BiRD (Asaadi et al., 2019) and WikiSRS (Newman-Griffis et al., 2018), which are recently collected, contain ground truth phrase similarities derived from human annotations.

The task of Semeval 2013 is to distinguish similar phrase pairs from dissimilar phrase pairs.

In Turney (5), given each query bigram, the goal is to identify the unigram that is most similar to the query bigram among 5 unigram candidates, 5 and Turney (10) adds 5 more negative phrase pairs by pairing the reverse of bigram with the 5 unigrams.

BiRD and WikiSRS-Rel measure the relatedness of phrases and WikiSRS-Sim measures the similarity of phrases.

For our model, we evaluate two scoring functions that measure phrase similarity.

The first way averages the contextualized word embeddings from our transformer encoder as the phrase embedding and computes the cosine similarity between two phrase embeddings.

We label the method as Ours Emb.

The similar phrases should have similar multi-facet embeddings, so we compute the reconstruction error from the set of normalized codebook embeddings of one phrase S 1 q to the embeddings of the other phrase S 2 q , vice versa, and add them together to become a symmetric distance SC:

where

When ranking for retrieving similar phrases, we use the negative distance to represent similarity.

We compare our performance with 5 baselines.

GloVe Avg and Word2Vec Avg compute the cosine similarity between two averaged word embeddings, which has been shown to be a strong baseline (Asaadi et al., 2019) .

BERT CLS and BERT Avg are the cosine similarities between CLS embeddings and between the averages of contextualized word embeddings from BERT (Devlin et al., 2019) , respectively.

FCT LM Emb (Yu & Dredze, 2015) learns the weights of linearly combining word embeddings based on several linguistic features.

The performances are presented in Table 2 .

SemEval 2013 and Turney have training and testing split and the performances in test sets are reported.

Our models significantly outperform all baselines in the 4 datasets.

Furthermore, our strong performances in Turney (10) verify that our encoder incorporates the word order information when producing phrase embeddings.

The results indicate the effectiveness of non-linearly composing word embeddings (unlike GloVe, Word2Vec, and FCT baselines) in order to predict the set of co-occurring word embeddings (unlike the BERT baselines).

The performance of Ours (K=1) is usually slightly better than Ours (K=10).

This result supports the finding of Dubossarsky et al. (2018) that multi-mode embeddings may not improve the performance in word similarity benchmarks even if they capture more senses or aspects of polysemies.

Even though being slightly worse, the performances of Ours (K=10) remain strong compared with baselines.

This indicates that the similarity performance is not sensitive to the number of clusters, which alleviates the problem of selecting K in practice.

STS benchmark (Cer et al., 2017 ) is a widely used sentence similarity task.

Each model predicts a semantic similarity score between each sentence pair, and the scores are compared with average similarity annotations using the Pearson correlation coefficient.

Intuitively, when two sentences are less similar to each other, humans tend to judge the similarity based on how they are similar in each aspect.

Thus, we also compare the performances on lower half the datasets where their ground truth similarities are less than the median similarity score, and we call this benchmark STSB Low.

In addition to BERT CLS, BERT Avg, and GloVe Avg, we compare our method with word mover's distance (WMD) (Kusner et al., 2015) and cosine similarity between skip-thought embeddings (ST Cos) (Kiros et al., 2015) .

Recently, Arora et al. (2017) propose to weight the word w in each sentence according to α α+p(w) , where α is a constant and p(w) is the probability of seeing the word w in the corpus.

Following its recommendation, we set α to be 10 −4 in STS benchmark.

After the weighting, we adopt the post-processing method from Arora et al. (2017) to remove the first principal component that is estimated using the training distribution and denote the method as GloVe SIF.

The post-processing is not desired in some applications (Singh et al., 2018), so we also report the performance before removing principal components, which is called GloVe Prob_avg.

The strong performance of (weighted) average embedding (Milajevs et al., 2014; Arora et al., 2017) suggests that we should consider the embeddings of words in the sentence in addition to the sentence embedding(s) when measuring the sentence similarity.

This is hard to be achieved in other sentence representation methods because their sentence embeddings and word embeddings are in different semantic spaces.

Since our multi-facet embeddings are in the same space of word embeddings, we can use the multifacet embeddings to estimate the word importance (in terms of predicting possible co-occurring words beside the sentence).

To compute the importance of a word in the sentence, we first compute the cosine similarity between the word and all predicted codebook embeddings, truncate the negative similarity to 0, and sum all similarity.

Specifically, our simple importance/attention weighting for all the words in the query sentence S q is defined by

where 1 is an all-one vector.

The importance weighting is multiplied with the original weighting vectors on GloVe Avg (uniform weight), GloVe Prob_avg, and GloVe SIF to generate the results of Our Avg, Our Prob_avg, and Our SIF, respectively.

We compare all the results in the development set and test set in Table 3 .

Ours SC, which matches between two sets of topics outperforms WMD, which matches between two sets of words in the sentence, and also outperforms BERT Avg, especially in STSB Low.

All the scores in Ours (K=10) are significantly better than Ours (K=1) ,which demonstrates the benefits of multi-mode representation.

Multiplying the proposed attention weighting boosts the performance from (weighted) averagingbased methods especially in STSB Low and when we do not rely on the generalization assumption of our training distribution.

We also test a variant of our method which uses a bi-LSTM as the encoder and a LSTM as the decoder.

Its average validation loss (-0.1289) in equation 2 is significantly higher than that of the transformer alternative (-0.1439), and it performances on Table 3 are also worse.

Notice the architecture of this variant is similar to skip-thoughts except that skip-thoughts decodes a sequence instead of a set.

The variant significantly outperforms ST Cos, which further justifies our approach of ignoring the order of co-occurring words in our NNSC loss.

We apply our model to HypeNet (Shwartz et al., 2016) , an unsupervised hypernymy detection dataset, based on an assumption that the co-occurring words of a phrase are often less related to some of its hyponyms.

For instance, fly is a co-occurring word of animal which is less related to brown dog.

Thus, the predicted codebook embeddings of a hyponym S hypo q (e.g., brown dog), which cluster the embeddings of co-occurring words (e.g., eats), often reconstruct the embeddings of its hypernym S hyper q (e.g., animal) better than the other way around (i.e., Er(

Based on the assumption, our asymmetric scoring function is defined as

The AUC of detecting hypernym among other relations and accuracy of detecting the hypernym direction are compared in Table 4 .

Our methods outperform baselines, which only provide symmetric similarity measurement, and Ours (K=1) performs similarly compared with Ours (K=10).

A good summary should cover multiple aspects that well represent all topics/concepts in the whole document.

The objective can be quantified as discovering a summary A with a set of normalized embeddings C(A) which best reconstructs the distribution of normalized word embedding w in the document D (Kobayashi et al., 2015) .

That is,

where γ w is the importance of the word w, which is set as α α+p(w) as we did in Section 3.2.

We further assume that the summary A consists of T sentences A 1 ...

A T and the embedding set of the summary is the union of the embedding sets of the sentences C(A) = ∪ T t=1 C(A t ), and we greedily select sentences to optimize equation 8 as did in Kobayashi et al. (2015) .

This extractive summarization method provides us a way to evaluate the embedding(s) of sentence.

Our model can generate multiple codebook embeddings, which capture its different aspects as we see in Table 1 , to represent each sentence in the document, so we let C(A t ) = { F u (A t )}, a set of column vectors in the matrix F u (A t ).

We compare our approach with other alternative ways of modeling the aspects of sentences.

For example, we can compute average word embeddings as a single-aspect sentence embedding.

This baseline is labeled as Sent Emb.

We can also use the embedding of all the words in the sentences as different aspects of the sentences.

Since longer sentences have more words, we normalize the gain of the reconstruction loss by the sentence length.

The method is denoted as W Emb.

In contrast, the fixed number of codebook embeddings in our method avoids the problem.

We also test the baselines of selecting random sentences (Rnd) and first n sentences (Lead) in the document.

The results on the testing set of CNN/Daily Mail (Hermann et al., 2015; See et al., 2017) are compared using F1 of ROUGE (Lin & Hovy, 2003) in Table 5 .

R-1, R-2, and Len mean ROUGE-1, ROUGE-2, and average summary length, respectively.

All methods choose 3 sentences by following the setting in Zheng & Lapata (2019) .

Unsup, No Order means the methods do not use the sentence position information in the documents.

In CNN/Daily Mail or other English news corpora, the sentence order information is a very strong signal.

For example, the unsupervised methods such as Lead-3 are very strong baselines (Bohn & Ling, 2018) with performances similar to supervised methods such as RL (Celikyilmaz et al., 2018) , a state-of-the-art approach in this evaluation.

To evaluate the quality of unsupervised sentence embeddings, we focus on comparing the unsupervised methods which do not assume the first few sentences form a good summary.

In Table 5 , predicting more aspects (i.e., using higher cluster numbers K) yields better results, and setting K = 100 gives us the best performance after selecting 3 sentences.

This demonstrates that larger cluster numbers K is desired in this application.

Our method allows us to set K to be a relatively large number because of greatly alleviating the computational and sample efficiency challenges.

Topic modeling (Blei et al., 2003) has been extensively studied and widely applied due to its interpretability and flexibility of incorporating different forms of input features (Mimno & McCallum, 2008) .

Cao et al. (2015) ; Srivastava & Sutton (2017) demonstrate that neural networks could be applied to discover semantically coherent topics.

However, instead of optimizing a global topic model, our goal is to jointly and efficiently discovering different sets of topics/clusters on the small subsets of words that co-occur with target phrases or sentences.

Sparse coding on word embedding space is used to model the multiple aspects of a word (Faruqui et al., 2015; Arora et al., 2018) , and parameterizing word embeddings using neural networks is used to test hypothesis (Han et al., 2018) and save storage space (Shu & Nakayama, 2018) .

Besides, to capture asymmetric relations such as entailment, words are represented as single or multiple regions in Gaussian embeddings (Vilnis & McCallum, 2015; Athiwaratkun & Wilson, 2017) rather than a single point.

However, the challenges of extending these methods to longer sequences are not addressed in these studies.

One of our main challenges is to design a neural decoder for a set rather a sequence while modeling the dependency between the elements.

This requires a matching step between two sets and compute the distance loss after the matching (Eiter & Mannila, 1997) .

One popular loss is called Chamfer distance, which is widely adopted in the auto-encoder models for point clouds (Yang et al., 2018; Liu et al., 2019) , while more sophisticated matching loss options are also proposed (Stewart et al., 2016; Balles & Fischbacher, 2019) .

The goal of the studies focus on measuring symmetric distances between the ground truth set and predicted set (usually with the equal size), while our set decoder tries to reconstruct a set using a set of much fewer bases.

Other ways to achieving the permutation invariants loss for neural networks includes removing the elements in the ground truth set which have been predicted (Welleck et al., 2018) , beam search (Qin et al., 2019) , or predicting the permutation using a CNN (Rezatofighi et al., 2018) , a transformer (Stern et al., 2019; Gu et al., 2019) or reinforcement learning (Welleck et al., 2019) .

In contrast, our goal is to efficiently predict a set of clustering centers that can well reconstruct the set of observed instances instead of predicting the set or sequence of the observed instances.

In this work, we overcome the computational and sampling efficiency challenges of learning the multi-mode representation for long sequences like phrases or sentences.

We use a neural encoder to model the compositional meaning of the target sequence and use a neural decoder to predict a set of codebook embeddings as the representation of the sentences or phrases.

During training, we use a non-negative sparse coefficient matrix to dynamically match the predicted codebook embeddings to a set of observed co-occurring words and allow the neural decoder to predict the clustering centers with an arbitrary permutation.

We demonstrate that the proposed models can learn to predict interpretable clustering centers conditioned on an (unseen) sequence, and the representation outperforms widely-used baselines such as BERT, skip-thoughts and various approaches based on GloVe in several unsupervised benchmarks.

The experimental results also suggest that multi-facet embeddings perform the best when the input sequence (e.g., a sentence) involves many aspects, while multi-facet and single-facet embeddings perform similarly good when the input sequence (e.g., a phrase) usually involves only one aspect.

In the future, we would like to train a single model which could generate multi-facet embeddings for both phrases and sentences, and evaluate the method as a pre-trained embedding approach for supervised or semi-supervised settings.

Furthermore, we plan to apply this method to other unsupervised learning tasks that heavily rely on co-occurrence statistics such as graph embedding or recommendation.

Given the computational resource constraints, we keep our model simple enough to have a nearly converged training loss after 1 or 2 epoch(s).

Since training takes a long time, we do not fine-tune the hyper-parameters in our models.

We use a much smaller model compared with BERT but the architecture details in our transformer and most of its hyper-parameters are the same as the ones used in BERT.

The sparsity penalty weights on coefficient matrix λ is set to be 0.4.

The maximal sentence size is set to be 50 and we ignore the sentences longer than that.

The maximal number of co-occurring words is set to be 30 (after removing the stop words), and we sub-sample the words if there are more words in the previous and next sentence.

The number of dimensions in transformers is set to be 300.

For sentence representation, the number of transformer layers on the decoder side is 5 and dropout on attention is 0.1 for K = 10 and the number of transformer layer on the decoder side is set to be 1 for K = 1 because we do not need to model the dependency of output basis.

For phrase representation, the number of transformer layers on the decoder side is 2 and the dropout on attention is 0.5.

The window size d

t is set to be 5.

We will release the code to reveal more hyper-parameter setting details.

All the architecture and hyperparameters (except the number of codebook embeddings) in our models are determined by the validation loss of the self-supervised co-occurring word reconstruction task.

The number of codebook embeddings K is chosen by the performance of training data in each task, but we observe that the performances are usually not sensitive to the numbers as long as K is large enough.

We also suspect that the slight performance drops of models with too large K might just be caused by the fact that larger K needs longer training time and 1 week of training is insufficient to make the model converges.

For skip-thoughts, the hidden embedding of is set to be 600.

To make the comparison fair, we retrain the skip-thoughts in Wikipedia 2016 for 2 weeks.

As mentioned in Section 3, our model has fewer parameters than the BERT base model and uses much less computational resources for training, so we only present the BERT base performance in the experiment sections.

Nevertheless, we still wonder how well BERT large can perform in these unsupervised semantic tasks, so we compare our method with BERT Large in Table 6, Table 7, and  Table 8 .

As we can see, BERT large is usually better than BERT base in the similarity tasks, but perform worse in the hypernym detection task.

The performance gains of BERT in similarity tasks might imply that training a larger version of our model might be a promising future direction.

Although increasing the model size boosts the performance of BERT, our method is still much better in most of the cases, especially in the phrase similarity tasks.

One of the main reasons we hypothesize is that BERT is trained by predicting the masked words in the input sequence and the objective function might not be good if sequences are short (like phrases).

C SUMMARIZATION COMPARISON GIVEN THE SAME SUMMARY LENGTH In Section 3.4, we compare our methods with other baselines when all the methods choose the same number of sentences.

We suspect that the bad performances for W Emb (*) methods (i.e., representing each sentence using the embedding of words in the sentence) might come from the tendency of selecting shorter sentences.

6 To verify the hypothesis, we plot the R-1 performance of different unsupervised summarization methods that do not use sentence order information versus the sentence length in Figure 3 .

In the figure, we first observe that Ours (K=100) significantly outperforms W Emb (GloVe) and Sent Emb (GloVe) when summaries have similar length.

In addition, we find that W Emb (*) actually usually outperform Sent Emb (*) when comparing the summaries with a similar length.

Notice that this comparison might not be fair because W Emb (*) are allowed to select more sentences given the same length of summary and it might be easier to cover more topics in the document using more sentences.

In practice, preventing choosing many short sentences might be preferable in an extractive summarization if fluency is an important factor.

Nevertheless, if our goal is simply to maximize the ROUGE F1 score given a fixed length of summary without accessing the ground truth summary and sentence order information, the figure indicates that Ours (K=100) is the best choice when the summary length is less than around 50 words and W Emb (BERT) becomes the best method for a longer summary.

The BERT in this figure is the BERT base model.

The mixed results suggest that combining our method with BERT in a way might be a promising direction to get the best performance in this task (e.g., use contextualized word embedding from BERT as our pre-trained word embedding space).

We visualize predicted embeddings from 10 randomly selected sentences in our validation set.

The format of the file is similar to Table 1.

The first line of an example is always the prepossessed input sentence, where <unk> means an out-of-vocabulary placeholder.

The embedding in each row is visualized by the nearest five neighbors in a GloVe embedding space and their cosine similarities to the vector.

@highlight

We propose an unsupervised way to learn multiple embeddings for sentences and phrases 