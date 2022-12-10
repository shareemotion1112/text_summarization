We address two challenges of probabilistic topic modelling in order to better estimate the probability of a word in a given context, i.e., P(wordjcontext) : (1) No Language Structure in Context: Probabilistic topic models ignore word order by summarizing a given context as a “bag-of-word” and consequently the semantics of words in the context is lost.

In this work, we incorporate language structure by combining a neural autoregressive topic model (TM) with a LSTM based language model (LSTM-LM) in a single probabilistic framework.

The LSTM-LM learns a vector-space representation of each word by accounting for word order in local collocation patterns, while the TM simultaneously learns a latent representation from the entire document.

In addition, the LSTM-LM models complex characteristics of language (e.g., syntax and semantics), while the TM discovers the underlying thematic structure in a collection of documents.

We unite two complementary paradigms of learning the meaning of word occurrences by combining a topic model and a language model in a unified probabilistic framework, named as ctx-DocNADE.

(2) Limited Context and/or Smaller training corpus of documents: In settings with a small number of word occurrences (i.e., lack of context) in short text or data sparsity in a corpus of few documents, the application of TMs is challenging.

We address this challenge by incorporating external knowledge into neural autoregressive topic models via a language modelling approach: we use word embeddings as input of a LSTM-LM with the aim to improve the wordtopic mapping on a smaller and/or short-text corpus.

The proposed DocNADE extension is named as ctx-DocNADEe.



We present novel neural autoregressive topic model variants coupled with neural language models and embeddings priors that consistently outperform state-of-theart generative topic models in terms of generalization (perplexity), interpretability (topic coherence) and applicability (retrieval and classification) over 6 long-text and 8 short-text datasets from diverse domains.

Probabilistic topic models, such as LDA BID1 , Replicated Softmax (RSM) (Salakhutdinov & Hinton, 2009 ) and Document Neural Autoregressive Distribution Estimator (DocNADE) variants BID12 BID34 BID15 BID8 are often used to extract topics from text collections, and predict the probabilities of each word in a given document belonging to each topic.

Subsequently, they learn latent document representations that can be used to perform natural language processing (NLP) tasks such as information retrieval (IR), document classification or summarization.

However, such probabilistic topic models ignore word order and represent a given context as a bag of its words, thereby disregarding semantic information.

To motivate our first task of extending probabilistic topic models to incorporate word order and language structure, assume that we conduct topic analysis on the following two sentences: When estimating the probability of a word in a given context (here: P ("bear"|context)), traditional topic models do not account for language structure since they ignore word order within the context and are based on "bag-of-words" (BoWs) only.

In this particular setting, the two sentences have the same unigram statistics, but are about different topics.

On deciding which topic generated the word "bear" in the second sentence, the preceding words "market falls" make it more likely that it was generated by a topic that assigns a high probability to words related to stock market trading, where "bear territory" is a colloquial expression in the domain.

In addition, the language structure (e.g., syntax and semantics) is also ignored.

For instance, the word "bear" in the first sentence is a proper noun and subject while it is an object in the second.

In practice, topic models also ignore functional words such as "into", which may not be appropriate in some scenarios.

Recently, BID23 have shown that a deep contextualized LSTM-based language model (LSTM-LM) is able to capture different language concepts in a layer-wise fashion, e.g., the lowest layer captures language syntax and topmost layer captures semantics.

However, in LSTM-LMs the probability of a word is a function of its sentence only and word occurrences are modeled in a fine granularity.

Consequently, LSTM-LMs do not capture semantics at a document level.

To this end, recent studies such as TDLM BID14 , Topic-RNN (Dieng et al., 2016) and TCNLM BID32 have integrated the merits of latent topic and neural language models (LMs); however, they have focused on improving LMs with global (semantics) dependencies using latent topics.

Similarly, while bi-gram LDA based topic models BID31 BID33 and n-gram based topic learning BID15 can capture word order in short contexts, they are unable to capture long term dependencies and language concepts.

In contrast, DocNADE variants BID12 BID8 ) learns word occurrences across documents i.e., coarse granularity (in the sense that the topic assigned to a given word occurrence equally depends on all the other words appearing in the same document); however since it is based on the BoW assumption all language structure is ignored.

In language modeling, BID17 have shown that recurrent neural networks result in a significant reduction of perplexity over standard n-gram models.

Contribution 1: We introduce language structure into neural autoregressive topic models via a LSTM-LM, thereby accounting for word ordering (or semantic regularities), language concepts and long-range dependencies.

This allows for the accurate prediction of words, where the probability of each word is a function of global and local (semantics) contexts, modeled via DocNADE and LSTM-LM, respectively.

The proposed neural topic model is named as contextualized-Document Neural Autoregressive Distribution Estimator (ctx-DocNADE) and offers learning complementary semantics by combining joint word and latent topic learning in a unified neural autoregressive framework.

For instance, FIG0 (left and middle) shows the complementary topic and word semantics, based on TM and LM representations of the term "fall".

Observe that the topic captures the usage of "fall" in the context of stock market trading, attributed to the global (semantic) view.

While this is a powerful approach for incorporating language structure and word order in particular for long texts and corpora with many documents, learning from contextual information remains challenging in settings with short texts and few documents, since (1) limited word co-occurrences or little context (2) significant word non-overlap in such short texts and (3) small training corpus of documents lead to little evidence for learning word co-occurrences.

However, distributional word representations (i.e. word embeddings) BID22 have shown to capture both the semantic and syntactic relatedness in words and demonstrated impressive performance in NLP tasks.

For example, assume that we conduct topic analysis over the two short text fragments: Deal with stock index falls and Brace for market share drops.

Traditional topic models with "BoW" assumption will not be able to infer relatedness between word pairs such as (falls, drops) due to the lack of word-overlap and small context in the two phrases.

However, in the distributed embedding space, the word pairs are semantically related as shown in FIG0 (left).

DISPLAYFORM0 Related work such as BID26 employed web search results to improve the information in short texts and BID24 introduced word similarity via thesauri and dictionaries into LDA.

BID5 and BID20 integrated word embeddings into LDA and Dirichlet Multinomial Mixture (DMM) BID21 models.

Recently, BID8 extends DocNADE by introducing pre-trained word embeddings in topic learning.

However, they ignore the underlying language structure, e.g., word ordering, syntax, etc.

In addition, DocNADE and its extensions outperform LDA and RSM topic models in terms of perplexity and IR.Contribution 2: We incorporate distributed compositional priors in DocNADE: we use pre-trained word embeddings via LSTM-LM to supplement the multinomial topic model (i.e., DocNADE) in learning latent topic and textual representations on a smaller corpus and/or short texts.

Knowing similarities in a distributed space and integrating this complementary information via a LSTM-LM, a topic representation is much more likely and coherent.

Taken together, we combine the advantages of complementary learning and external knowledge, and couple topic-and language models with pre-trained word embeddings to model short and long text documents in a unified neural autoregressive framework, named as ctx-DocNADEe.

Our approach learns better textual representations, which we quantify via generalizability (e.g., perplexity), interpretability (e.g., topic extraction and coherence) and applicability (e.g., IR and classification).To illustrate our two contributions, we apply our modeling approaches to 7 long-text and 8 short-text datasets from diverse domains and demonstrate that our approach consistently outperforms state-ofthe-art generative topic models.

Our learned representations, result in a gain of: (1) 4.6% (.790 vs .755) in topic coherence, (2) 6.5% (.615 vs .577) in precision at retrieval fraction 0.02, and (3) 4.4% (.662 vs .634) in F 1 for text classification, averaged over 6 long-text and 8 short-text datasets.

When applied to short-text and long-text documents, our proposed modeling approaches generate contextualized topic vectors, which we name textTOvec.

The code is available at https: //github.com/pgcool/textTOvec.

Generative models are based on estimating the probability distribution of multidimensional data, implicitly requiring modeling complex dependencies.

Restricted Boltzmann Machine (RBM) BID9 and its variants BID11 are probabilistic undirected models of binary data.

RSM (Salakhutdinov & Hinton, 2009 ) and its variants BID7 are generalization of the RBM, that are used to model word counts.

However, estimating the complex probability distribution of the underlying high-dimensional observations is intractable.

To address this challenge, NADE BID13 decomposes the joint distribution of binary observations into autoregressive conditional distributions, each modeled using a feed-forward network.

Unlike for RBM/RSM, this leads to tractable gradients of the data negative log-likelihood.

An extension of NADE and RSM, DocNADE BID12 ) models collections of documents as orderless bags of words (BoW approach), thereby disregarding any language structure.

In other words, it is trained to learn word representations reflecting the underlying topics of the documents only, ignoring syntactical and semantic features as those encoded in word embeddings BID0 BID18 BID22 BID23 .

DocNADE BID15 DISPLAYFORM0 , where each autoregressive conditional p(v i |v <i ) for the word observation v i is computed using the preceding observations v <i ∈ {v 1 , ..., v i−1 } in a feed-forward neural network for i ∈ {1, ...D}, DISPLAYFORM1 where g(·) is an activation function, U ∈ R K×H is a weight matrix connecting hidden to output, e ∈ R H and b ∈ R K are bias vectors, W ∈ R H×K is a word representation matrix in which a column W :,vi is a vector representation of the word v i in the vocabulary, and H is the number of hidden units (topics).

The log-likelihood of any document v of any arbitrary length is given by: DISPLAYFORM2 Note that the past word observations v <i are orderless due to BoWs, and may not correspond to the words preceding the ith word in the document itself.

Input: A training document v Input: Word embedding matrix E Output: log p(v) 1: a ← e 2: q(v) = 1 3: for i from 1 to D do 4:compute hi and p(vi|v<i) 5: Table 1 : Computation of hi and p(vi|v<i) in DocNADE, ctx-DocNADE and ctx-DocNADEe models, correspondingly used in estimating log p(v) (Algorithm 1).

DISPLAYFORM0

We propose two extensions of the DocNADE model: (1) ctx-DocNADE: introducing language structure via LSTM-LM and (2) ctx-DocNADEe: incorporating external knowledge via pre-trained word embeddings E, to model short and long texts.

The unified network(s) account for the ordering of words, syntactical and semantic structures in a language, long and short term dependencies, as well as external knowledge, thereby circumventing the major drawbacks of BoW-based representations.

Similar to DocNADE, ctx-DocNADE models each document v as a sequence of multinomial observations.

Let [x 1 , x 2 , ..., x N ] be a sequence of N words in a given document, where x i is represented by an embedding vector of dimension, dim.

Further, for each element v i ∈ v, let c i = [x 1 , x 2 , ..., x i−1 ] be the context (preceding words) of ith word in the document.

Unlike in DocNADE, the conditional probability of the word v i in ctx-DocNADE (or ctx-DocNADEe) is a function of two hidden vectors: LSTM-based components of ctx-DocNADE, respectively: DISPLAYFORM0 DISPLAYFORM1 where h In the weight matrix W of DocNADE BID12 , each row vector W j,: encodes topic information for the jth hidden topic feature and each column vector W :,vi is a vector for the word v i .

To obtain complementary semantics, we exploit this property and expose W to both global and local influences by sharing W in the DocNADE and LSTM-LM componenents.

Thus, the embedding layer of LSTM-LM component represents the column vectors.

DISPLAYFORM2 ctx-DocNADE, in this realization of the unified network the embedding layer in the LSTM component is randomly initialized.

This extends DocNADE by accounting for the ordering of words and language concepts via context-dependent representations for each word in the document.ctx-DocNADEe, the second version extends ctx-DocNADE with distributional priors, where the embedding layer in the LSTM component is initialized by the sum of a pre-trained embedding matrix E and the weight matrix W. Note that W is a model parameter; however E is a static prior.

Algorithm 1 and Table 1 show the log p(v) for a document v in three different settings: Doc-NADE, ctx-DocNADE and ctx-DocNADEe.

In the DocNADE component, since the weights in the matrix W are tied, the linear activation a can be re-used in every hidden layer and computational complexity reduces to O(HD), where H is the size of each hidden layer.

In every epoch, we run an LSTM over the sequence of words in the document and extract hidden vectors h LM i , corresponding to c i for every target word v i .

Therefore, the computational complexity in ctx-DocNADE or ctx-DocNADEe is O(HD + N), where N is the total number of edges in the LSTM network BID10 BID27 .

The trained models can be used to extract a textTOvec representation, i.e., h(v DISPLAYFORM3 ctx-DeepDNEe: DocNADE and LSTM can be extended to a deep, multiple hidden layer architecture by adding new hidden layers as in a regular deep feed-forward neural network, allowing for improved performance.

In the deep version, the first hidden layer is computed in an analogous fashion to DocNADE variants (equation 1 or 2).

Subsequent hidden layers are computed as: DISPLAYFORM4 where n is the total number of hidden layers (i.e., depth) in the deep feed-forward and LSTM networks.

For d=1, the hidden vectors h DN i,1 and h LM i,1 correspond to equations 1 and 2.

The conditional p(v i = w|v <i ) is computed using the last layer n, i.e., h i,n = h TAB10 : State-of-the-art comparison: IR (i.e, IR-precision at 0.02 fraction) and classification F 1 for short texts, where Avg: average over the row values, the bold and underline: the maximum for IR and F1, respectively.

DISPLAYFORM5

We apply our modeling approaches (in improving topic models, i.e, DocNADE using language concepts from LSTM-LM) to 8 short-text and 7 long-text datasets of varying size with single/multiclass labeled documents from public as well as industrial corpora.

We present four quantitative measures in evaluating topic models: generalization (perplexity), topic coherence, text retrieval and categorization.

See the appendices for the data description and example texts.

TAB1 shows the data statistics, where 20NS and R21578 signify 20NewsGroups and Reuters21578, respectively.

Baselines: While, we evaluate our multi-fold contributions on four tasks: generalization (perplexity), topic coherence, text retrieval and categorization, we compare performance of our proposed models ctx-DocNADE and ctx-DocNADEe with related baselines based on: (1) word representation: glove BID22 , where a document is represented by summing the embedding vectors of it's words, (2) document representation: doc2vec (Le & Mikolov, 2014), (3) LDA based BoW TMs: ProdLDA (Srivastava & Sutton, 2017) and SCHOLAR 1 BID3 ) (4) neural BoW TMs: DocNADE and NTM BID2 and , (5) TMs, including pre-trained word embeddings: Gauss-LDA (GaussianLDA) BID5 , and glove-DMM, glove-LDA BID20 .

(6) jointly 2 trained topic and language models: TDLM (Lau et al., 2017), Topic-RNN (Dieng et al., 2016) and TCNLM BID32 .

Experimental Setup: DocNADE is often trained on a reduced vocabulary (RV) after pre-processing (e.g., ignoring functional words, etc.); however, we also investigate training it on full text/vocabulary (FV) ( TAB1 ) and compute document representations to perform different evaluation tasks.

The FV setting preserves the language structure, required by LSTM-LM, and allows a fair comparison of DocNADE+FV and ctx-DocNADE variants.

We use the glove embedding of 200 dimensions.

All the baselines and proposed models (ctx-DocNADE, ctx-DocNADEe and ctx-DeepDNEe) were run in the FV setting over 200 topics to quantify the quality of the learned representations.

To better initialize the complementary learning in ctx-DocNADEs, we perform a pre-training for 10 epochs with λ set to 0.

See the appendices for the experimental setup and hyperparameters for the following tasks, including the ablation over λ on validation set.

BID14 for all the short-text datasets to evaluate the quality of representations learned in the spare data setting.

For a fair comparison, we set 200 topics and hidden size, and initialize with the same pre-trained word embeddings (i.e., glove) as used in the ctx-DocNADEe.

DISPLAYFORM0

To evaluate the generative performance of the topic models, we estimate the log-probabilities for the test documents and compute the average held-out perplexity (P P L) per word as, P P L = exp − 1 z z t=1 DISPLAYFORM0 , where z and |v t | are the total number of documents and words in a document v t .

For DocNADE, the log-probability log p(v t ) is computed using L DN (v); however, we ignore the mixture coefficient, i.e., λ=0 (equation 2) to compute the exact log-likelihood in ctx-DocNADE versions.

The optimal λ is determined based on the validation set.

TAB5 quantitatively shows the PPL scores, where the complementary learning with λ = 0.01 (optimal) in ctx-DocNADE achieves lower perplexity than the baseline DocNADE for both short and long texts, e.g., (822 vs 846) and (1358 vs 1375) on AGnewstitle and 20NS 4 datasets, respectively in the FV setting.

We compute topic coherence BID4 BID19 BID7 to assess the meaningfulness of the underlying topics captured.

We choose the coherence measure proposed by BID25 , which identifies context features for each topic word using a sliding window over the reference corpus.

Higher scores imply more coherent topics.

We use the gensim module (radimrehurek.com/gensim/models/coherencemodel.html, coherence type = c v) to estimate coherence for each of the 200 topics (top 10 and 20 words).

TAB7 shows average coherence over 200 topics, where the higher scores in ctx-DocNADE compared to DocNADE (.772 vs .755) suggest that the contextual information and language structure help in generating more coherent topics.

The introduction of embeddings in ctx-DocNADEe boosts the topic coherence, leading to a gain of 4.6% (.790 vs .755) on average over 11 datasets.

Note that the proposed models also outperform the baselines methods glove-DMM and glove-LDA.

Qualitatively, Table 8 illustrates an example topic from the 20NSshort text dataset for DocNADE, ctx-DocNADE and ctx-DocNADEe, where the inclusion of embeddings results in a more coherent topic.

Additional Baslines: We further compare our proposed models to other approaches that combining topic and language models, such as TDLM (Lau et al., 2017), Topic-RNN (Dieng et al., 2016) and TCNLM BID32 .

However, the related studies focus on improving language models using topic models: in contrast, the focus of our work is on improving topic models for textual representations (short-text or long-text documents) by incorporating language concepts (e.g., word ordering, syntax, semantics, etc.) and external knowledge (e.g., word embeddings) via neural language models, as discussed in section 1.To this end, we follow the experimental setup of the most recent work, TCNLM and quantitatively compare the performance of our models (i.e., ctx-DocNADE and ctx-DocNADEe) in terms of topic coherence (NPMI) on BNC dataset.

The sliding window is one of the hyper-parameters for computing topic coherence BID25 BID32 .

A sliding window of 20 is used in TCNLM; in addition we also present results for a window of size 110.

λ is the mixture weight of the LM component in the topic modeling process, and (s) and (l) indicate small and large model, respectively.

The symbol '-' indicates no result, since word embeddings of 150 dimensions are not available from glove vectors. (Right): The top 5 words of seven learnt topics from our models and TCNLM.

The asterisk (*) indicates our proposed models and (#) taken from TCNLM BID32 .ues of λ illustrates the relevance of the LM component for topic coherence (DocNADE corresponds to λ=0).

Similarly, the inclusion of word embeddings (i.e., ctx-DocNADEe) results in more coherent topics than the baseline DocNADE.

Importantly, while ctx-DocNADEe is motivated by sparse data settings, the BNC dataset is neither a collection of short-text nor a corpus of few documents.

Consequently, ctx-DocNADEe does not show improvements in topic coherence over ctx-DocNADE.In TAB8 (right), we further qualitatively show the top 5 words of seven topics (topic name summarized by BID32 ) from TCNML and our models.

Observe that ctx-DocNADE captures a topic expression that is a collection of only verbs in the past participle.

Since the BNC dataset is unlabeled, we are here restricted to comparing model performance in terms of topic coherence only.

Text Retrieval: We perform a document retrieval task using the short-text and long-text documents with label information.

We follow the experimental setup similar to BID15 , where all test documents are treated as queries to retrieve a fraction of the closest documents in the original training set using cosine similarity measure between their textTOvec representations (section 2.2).

To compute retrieval precision for each fraction (e.g., 0.0001, 0.005, 0.01, 0.02, 0.05, etc.), we average the number of retrieved training documents with the same label as the query.

For multi-label datasets, we average the precision scores over multiple labels for each query.

Since, BID28 Hinton (2009) and BID15 have shown that RSM and DocNADE strictly outperform LDA on this task, we solely compare DocNADE with our proposed extensions.

TAB10 and 4 show the retrieval precision scores for the short-text and long-text datasets, respectively at retrieval fraction 0.02.

Observe that the introduction of both pre-trained embeddings and language/contextual information leads to improved performance on the IR task noticeably for short texts.

We also investigate topic modeling without pre-processing and filtering certain words, i.e. the FV setting and find that the DocNADE(FV) or glove(FV) improves IR precision over the baseline RV setting.

Therefore, we opt for the FV in the proposed extensions.

On an average over the 8 shorttext and 6 long-text datasets, ctx-DocNADEe reports a gain of 7.1% (.630 vs .588) ( in IR-precision.

In addition, the deep variant (d=3) with embeddings, i.e., ctx-DeepDNEe shows competitive performance on TREC6 and Subjectivity datasets.

FIG4 , 3d, 3e and 3f) illustrate the average precision for the retrieval task on 6 datasets.

Observe that the ctx-DocNADEe outperforms DocNADE(RV) at all the fractions and demonstrates a gain of 6.5% (.615 vs .577) in precision at fraction 0.02, averaged over 14 datasets.

Additionally, our proposed models outperform TDLM and ProdLDA 6 (for 20NS) by noticeable margins.

We perform text categorization to measure the quality of our textTovec representations.

We consider the same experimental setup as in the retrieval task and extract textTOvec of 200 dimension for each document, learned during the training of ctx-DocNADE variants.

To perform text categorization, we employ a logistic regression classifier with L2 regularization.

While, ctx-DocNADEe and ctx-DeepDNEe make use of glove embeddings, they are evaluated against the topic model baselines with embeddings.

For the short texts TAB10 , the glove leads DocNADE in classification performance, suggesting a need for distributional priors in the topic model.

Therefore, the ctx-DocNADEe reports a gain of 4.8% (.705 vs .673) and 3.6%(.618 vs .596) in F 1, compared to DocNADE(RV) on an average over the short TAB10 and long TAB4 texts, respectively.

In result, a gain of 4.4% (.662 vs .634) overall.

In terms of classification accuracy on 20NS dataset, the scores are: DocNADE (0.734), ctxDocNADE (0.744), ctx-DocNADEe (0.751), NTM (0.72) and SCHOLAR (0.71).

While, our proposed models, i.e., ctx-DocNADE and ctx-DocNADEe outperform both NTM (results taken from BID2 , FIG1 ) and SCHOLAR (results taken from BID3 , TAB1 ), the DocNADE establishes itself as a strong neural topic model baseline.

To further interpret the topic models, we analyze the meaningful semantics captured via topic extraction.

Table 8 shows a topic extracted using 20NS dataset that could be interpreted as computers, which are (sub)categories in the data, confirming that meaningful topics are captured.

Observe that the ctx-DocNADEe extracts a more coherent topic due to embedding priors.

To qualitatively inspect the contribution of word embeddings and textTOvec representations in topic models, we analyse the text retrieved for each query using the representations learned from DocNADE and ctxDoocNADEe models.

Table 9 illustrates the retrieval of the top 3 texts for an input query, selected from TMNtitle dataset, where #match is YES if the query and retrievals have the same class label.

Observe that ctx-DocNADEe retrieves the top 3 texts, each with no unigram overlap with the query.

vga, screen, computer, color, svga, graphics computer, sell, screen, offer, bar, macintosh, color, powerbook, vga, card, san, windows, sold, cars, terminal, forsale, utility, monitor, svga, offer gov, vesa computer, processor .554 .624 .667 Table 8 : A topic of 20NS dataset with coherence -DocNADEe Query :: "emerging economies move ahead nuclear plans" #match ctx-#IR1 :: imf sign lifting japan yen YES #IR2 :: japan recovery takes hold debt downgrade looms YES #IR3 :: japan ministers confident treasuries move YES DocNADE #IR1 :: nuclear regulator back power plans NO #IR2 :: defiant iran plans big rise nuclear NO #IR3 :: japan banks billion nuclear operator sources YES Table 9 : Illustration of the top-3 retrievals for an input query Additionally, we show the quality of representations learned at different fractions (20%, 40%, 60%, 80%, 100%) of training set from TMNtitle data and use the same experimental setup for the IR and classification tasks, as in section 3.3.

In FIG5 , we quantify the quality of representations learned and demonstrate improvements due to the proposed models, i.e., ctx-DocNADE and ctx-DocNADEe over DocNADE at different fractions of the training data.

Observe that the gains in both the tasks are large for smaller fractions of the datasets.

For instance, one of the proposed models, i.e., ctxDocNADEe (vs DocNADE) reports: (1) a precision (at 0.02 fraction) of 0.580 vs 0.444 at 20% and 0.595 vs 0.525 at 100% of the training set, and (2) an F1 of 0.711 vs 0.615 at 20% and 0.726 vs 0.688 at 100% of the training set.

Therefore, the findings conform to our second contribution of improving topic models with word embeddings, especially in the sparse data setting.

DISPLAYFORM0

In this work, we have shown that accounting for language concepts such as word ordering, syntactic and semantic information in neural autoregressive topic models helps to better estimate the probability of a word in a given context.

To this end, we have combined a neural autoregressive topic-(i.e., DocNADE) and a neural language (e.g., LSTM-LM) model in a single probabilistic framework with an aim to introduce language concepts in each of the autoregressive steps of the topic model.

This facilitates learning a latent representation from the entire document whilst accounting for the local dynamics of the collocation patterns, encoded in the internal states of LSTM-LM.

We further augment this complementary learning with external knowledge by introducing word embeddings.

Our experimental results show that our proposed modeling approaches consistently outperform stateof-the-art generative topic models, quantified by generalization (perplexity), topic interpretability (coherence), and applicability (text retrieval and categorization) on 15 datasets.

Label: training Instructors shall have tertiary education and experience in the operation and maintenance of the equipment or sub-system of Plant.

They shall be proficient in the use of the English language both written and oral.

They shall be able to deliver instructions clearly and systematically.

The curriculum vitae of the instructors shall be submitted for acceptance by the Engineer at least 8 weeks before the commencement of any training.

Label: maintenance The Contractor shall provide experienced staff for 24 hours per Day, 7 Days per week, throughout the Year, for call out to carry out On-call Maintenance for the Signalling System.

Label: cables Unless otherwise specified, this standard is applicable to all cables which include single and multi-core cables and wires, Local Area Network (LAN) cables and Fibre Optic (FO) cables.

Label: installation The Contractor shall provide and permanently install the asset labels onto all equipment supplied under this Contract.

The Contractor shall liaise and co-ordinate with the Engineer for the format and the content of the labels.

The Contractor shall submit the final format and size of the labels as well as the installation layout of the labels on the respective equipment, to the Engineer for acceptance.

Label: operations, interlocking It shall be possible to switch any station Interlocking capable of reversing the service into "Auto-Turnaround Operation".

This facility once selected shall automatically route Trains into and out of these stations, independently of the ATS system.

At stations where multiple platforms can be used to reverse the service it shall be possible to select one or both platforms for the service reversal.

TAB10 : Perplexity scores for different λ in Generalization task: Ablation over validation set labels are not used during training.

The class labels are only used to check if the retrieved documents have the same class label as the query document.

To perform document retrieval, we use the same train/development/test split of documents discussed in data statistics (experimental section) for all the datasets during learning.

See TAB1 for the hyperparameters in the document retrieval task.

We used gensim (https://github.com/RaRe-Technologies/gensim) to train Doc2Vec models for 12 datasets.

Models were trained with distributed bag of words, for 1000 iterations using a window size of 5 and a vector size of 500.

We used the same split in training/development/test as for training the Doc2Vec models (also same split as in IR task) and trained a regularized logistic regression classifier on the inferred document vectors to predict class labels.

In the case of multilabel datasets (R21578,R21578title, RCV1V2), we used a one-vs-all approach.

Models were trained with a liblinear solver using L2 regularization and accuracy and macro-averaged F1 score were computed on the test set to quantify predictive power.

We used LFTM (https://github.com/datquocnguyen/LFTM) to train glove-DMM and glove-LDA models.

Models were trained for 200 iterations with 2000 initial iterations using 200 topics.

For short texts we set the hyperparameter beta to 0.1, for long texts to 0.01; the mixture parameter lambda was set to 0.6 for all datasets.

The setup for the classification task was the same as for doc2vec; classification was performed using relative topic proportions as input (i.e. we inferred the topic distribution of the training and test documents and used the relative distribution as input Topic coherence (NPMI) using 20 topics: DocNADE (.18) and SCHOLAR (.35), i.e., SCHOLAR BID3 generates more coherence topics than DocNADE, though worse in PPL and text classification (see section 3.3) than DocNADE, ctx-DocNADE and ctx-DocNADEe.

IR tasks: Since, SCHOLAR BID3 without meta-data equates to ProdLDA and we have shown in section 3.3 that ProdLDA is worse on IR tasks than our proposed models, therefore one can infer the performance of SCHOLAR on IR task.

The experimental results above suggest that the DocNADE is better than SCHOLAR in generating good representations for downstream tasks such as information retrieval or classification, however falls behind SCHOLAR in interpretability.

The investigation opens up an interesting direction for future research.

@highlight

Unified neural model of topic and language modeling to introduce language structure  in topic models for contextualized topic vectors 