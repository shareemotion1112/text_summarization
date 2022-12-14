Visual grounding of language is an active research field aiming at enriching text-based representations with visual information.

In this paper, we propose a new way to leverage visual knowledge for sentence representations.

Our approach transfers the structure of a visual representation space to the textual space by using two complementary sources of information: (1) the cluster information: the implicit knowledge that two sentences associated with the same visual content describe the same underlying reality and (2) the perceptual information contained within the structure of the visual space.

We use a joint approach to encourage beneficial interactions during training between textual, perceptual, and cluster information.

We demonstrate the quality of the learned representations on semantic relatedness, classification, and cross-modal retrieval tasks.

Building linguistic vectors that represent semantics is a long-standing issue in Artificial Intelligence.

Distributional Semantic Models BID36 BID41 are well-known recent efforts in this direction, making use of the distributional hypothesis BID16 on text corpora to learn word embeddings.

At another granularity level, having high-quality general-purpose sentence representations is crucial for all models that encode sentences into semantic vectors, such as the ones used in machine translation BID0 or question answering BID42 .

Moreover, encoding semantics of sentences is paramount because sentences describe relationships between objects and thus convey complex and high-level knowledge better than individual words, which mostly refer to a single concept BID38 .Relying only on text can lead to biased representations and unrealistic predictions (e.g., text-based models could predict that "the sky is green" BID1 ).

Besides, it has been shown that human understanding of language is grounded in physical reality and perceptual experience (FincherKiefer, 2001 ).

To overcome this limitation, one emerging approach is the visual grounding of language, which consists of leveraging visual information, usually from images, to enhance word representations.

Two methods showing substantial improvements have emerged: (1) the sequential technique combines textual and visual representations that were separately learned BID3 BID44 , and (2) the joint method learns a common multimodal representation from multiple sources simultaneously BID29 .

In the case of words, the latter has proven to produce representations that perform better on intrinsic and downstream tasks.

While there exist numerous approaches to learning sentence representations from text corpora only, and to learning multimodal word embeddings, the problem of the visual grounding of sentences is quite new to the research community.

To the best of our knowledge, the only work in the field is BID26 .

The authors propose a sequential model: linguistic vectors, learned from a purely textual corpus, are concatenated with grounded vectors, which were independently learned from a captioning dataset.

However, the two sources are considered separately, which might prevent beneficial interactions between textual and visual modalities during training.

We propose a joint model to learn multimodal sentence representations, based on the assumption that the meaning of a sentence is simultaneously grounded in its textual and visual contexts.

In our case, the textual context of a sentence consists of adjacent sentences in a text corpus.

Within a distinct dataset, the visual context is learned from a paired video and its associated captions.

Indeed, we propose to use videos instead of images because of their temporal aspect, since sentences often describe actions grounded in time.

The key challenge is to capture visual information.

Usually, to transfer information from the visual space to the textual one, one space is projected onto the other BID26 BID29 .

However, as pointed out by BID9 , projections are not sufficient to transfer neighborhood structure between modalities.

In our work, we rather propose to exploit the visual space by preserving the overall structure, i.e. conserving the similarities between related elements across spaces.

More precisely, we take visual context into account by distinguishing two types of complementary information sources.

First, the cluster information, which consists in the implicit knowledge that sentences associated with the same video refer to the same underlying reality.

Second, the perceptual information, which is the high-level information extracted from a video using a pre-trained CNN.Regarding these considerations, we formulate three Research Questions (RQ):??? RQ1:

Is perceptual information useful to improve sentence representations???? RQ2: Are cluster and perceptual information complementary, and does their combination compete with previous models based on projections between visual and textual spaces???? RQ3: Is a joint approach better suited than a sequential one regarding the multimodal acquisition of textual and visual knowledge?Our contribution is threefold: (1) We propose a joint multimodal framework for learning grounded sentence representations; (2) We show that cluster and perceptual information are complementary sources of information; (3) To the best of our knowledge, obtained results achieve state-of-the-art performances on multimodal sentence representations.

Our framework learns multimodal representations for sentences by jointly leveraging the textual and visual contexts of a sentence.

The textual resource is a large text corpus C T of ordered sentences.

The visual resource is a distinct video corpus C V , whose videos are associated with one or more descriptive captions.

A sentence S is represented by s = F ?? (S) and its corresponding video V S by v s = G ?? (V S ), where F (resp.

G) is a sentence (resp.

video) encoder parameterized by ?? (resp.

?? ).

We propose to use a joint approach where the sentence encoder F ?? is learned by jointly optimizing a textual objective L T (??) on C T and a visual objective L V (??, ?? ) on C V .

So far, this method has only been applied to words, with good results BID29 BID53 .

Note that C T and C V are not parallel corpora but that ?? is shared between both objectives; in other terms, sentence representations are influenced by their distinct textual and visual contexts.

Any sentence encoder F ?? and textual objective L T can be used such as SkipTought BID28 , FastSent BID19 or QuickThought BID34 .

In this paper, we focus on SkipThought, and present evidences that our approach also improves over FastSent (section 4.3).

In the following, we introduce hypotheses and their derived objectives to tackle the modeling of L V .

Most visual grounding works use projections between the textual space and the visual space BID26 BID29 to integrate visual information.

However, when a cross-modal mapping is learned, the projection of the source modality does not resemble the target modality, in the sense of neighborhood topology BID9 .

This suggests that projections between spaces is not an appropriate approach to incorporate visual semantics.

Instead, we propose a new way to structure the textual space with the help of the visual modality.

Without even considering the content of videos, the fact that sentences describe or not a same underlying reality is an implicit source of information that we name the cluster information.

For convenience, two sentences are said to be visually equivalent (resp.

visually different) if they are associated with the same video (resp.

different videos), i.e. if they describe the same (resp.

different) underlying reality.

We call cluster a set of visually equivalent sentences.

Leveraging the cluster information may be useful to improve the structure of the textual space: intuitively, representations of visually equivalent sentences should be close, and representations of visually different sentences should be separated.

We thus formulate the following hypothesis (see red elements in FIG0 ): Red arrows represent the gradient of the loss derived from the cluster hypothesis (C), which gathers visually equivalent sentences.

For clarity's sake, the term in equation 1 that separates negative pairs is not represented.

The green arrow and angles illustrate the loss derived from the perceptual hypothesis (S), which requires cosine similarities to correlate across the two spaces.

The point at the center of each space is the origin.

DISPLAYFORM0 Cluster Hypothesis (C): A sentence should be closer to a visually equivalent sentence than to a visually different sentence.

We translate this hypothesis into the constraint cos(s, s DISPLAYFORM1 is a visually equivalent (resp.

different) sentence to s. Following BID24 ; BID5 , we use a max-margin ranking loss to ensure the gap between both terms is higher than a fixed margin m: DISPLAYFORM2 where (s, s + ) cover visually equivalent pairs; visually different sentences s ??? are randomly sampled.

The cluster hypothesis ignores the structure of the visual space and only uses the visual modality as a proxy to assess if two sentences are visually equivalent or different.

Moreover, a ranking loss simply drives visually different sentences apart in the representation space, even if their corresponding videos are closely related.

To cope with this limitation, we suggest to take into account the structure of the visual space and use the content of videos, and then propose a novel approach which does not require cross-modal projections.

The intuition is that the structure of the textual space should be modeled on the structure of the visual one to extract visual semantics.

We choose to preserve similarities between related elements across spaces.

We thus formulate the following hypothesis, illustrated with green elements in FIG0 : DISPLAYFORM3 The similarity between two sentences in the textual space should be correlated with the similarity between their corresponding videos in the visual space.

We translate this hypothesis into the loss L P = ????? vis , where ?? vis = ??(cos(s, s ), cos(v s , v s )) and ?? is the Pearson correlation.

The final multimodal loss is a linear combination of the aforementioned objectives, weighted by hyperparameters ?? T , ?? P and ?? C : DISPLAYFORM4 To evaluate the impact of visual semantics on sentence grounding, we examine several types of visual context.

As done in Yao et al. (2016) ; BID15 , visual features are extracted using the penultimate layer of a pretrained CNN.

A video is represented as a set of n images (I k ) k??? [1,n] .

Let (i k ) k??? [1,n] be the representations of these images obtained with the pre-trained CNN.

We present below three simple ways to represent a video V .

Note that our model can be generalized to more complex video representations BID22 BID46 .One Frame (F ): this simple setting amounts at keeping the first frame and ignoring the rest of the sequence (any other frame might be used).

The visual context vector is v = i 1 .Average (A): the temporal aspect is ignored, and the scene is represented by the average of the individual frame features: v = 1 n n k=1 i k BID54 .

Temporal Grounding (T ): the intuition is that, in a video, not all frames are relevant to sentence understanding.

An attention mechanism allows us to focus on important frames.

We set: v = n k=1 ?? k i k , where ?? k = softmax(< w u w , N.i k >).

The sum ranges over the words w of the sentence s, u w is the fixed pretrained word embedding of w, and N is a learned projection.3 EVALUATION PROTOCOL 3.1 DATASETS Textual dataset.

Following BID28 ; BID19 , we use the Toronto BookCorpus dataset as the textual corpus C T .

This corpus consists of 11K books: this makes a total of 74M ordered sentences, with an average of 13 words per sentence.

Visual dataset.

We use the MSVD dataset BID7 as the visual corpus C V .

This video captioning dataset consists of 1970 videos and 80K English descriptions.

On average, a video lasts 10 seconds and has about 41 associated sentences.

Model Scenarios.

We test different variants of our multimodal model presented in section 2.

We note these variants M I V (?? T , ?? P , ?? C ), which depend on:??? the initialization I ??? {p, ???}: the sentence encoder F ?? is either pretrained using the textual objective L T (I = p), or initialized randomly (I = ???).??? the visual representation V ??? {F, A, T, R}: where F , A or T are the video modelings described in Section 2.3.

We introduce a baseline R, where visual vectors are randomly sampled from a normal distribution to measure the information brought by the video content.

Baselines.

We propose two extensions of multimodal word embedding models to sentences:??? Projection (P): Inspired by BID29 , this baseline is projecting videos in the textual space, while our model keeps both spaces separated.

The visual loss is a ranking objective: DISPLAYFORM0 where W is a trainable projection matrix and m a fixed margin.

We note P I V (?? T ) the variants of this baseline using the global loss DISPLAYFORM1 ??? Sequential (SEQ): Inspired by Collell Talleda et al. FORMULA2 , we learn a linear regression model (W, b) to predict the visual representation from the SkipThought representations.

The multimodal sentence embedding is the concatenation of the original SkipThought vector and its predicted representation: ST ??? W ST + b, projected into a lower-dimensional space using PCA.

This baseline can also be seen as a simpler variant of the model in BID26 .

In line with previous works on sentence embeddings BID28 BID19 , we consider several benchmarks to evaluate the quality of our learned multimodal representations:Semantic relatedness: We use two well-known semantic similarity benchmarks: STS BID6 and SICK BID35 , which consist of pairs of sentences that are associated with human-labeled similarity scores.

STS is subdivided in three textual sources: Captions contains sentences with a strong visual content, describing everyday-life actions, whereas the others contain more abstract sentences: news headlines in News and posts from users forum in Forum.

Correlations (Spearman/Pearson) are measured between the cosine similarity of our learned sentence embeddings and human-labeled scores.

Hyperparameters are tuned on SICK/trial (results on SICK/train+test are reported in tables).Classification benchmarks: We use six sentence classification benchmarks: paraphrase identification (MSRP) BID11 , opinion polarity (MPQA) BID50 , movie review sentiment (MR) BID39 , subjectivity/objectivity classification (SUBJ) (Scott et al., 2004) , question-type classification (TREC) BID48 ) and customer product reviews (CR) BID21 .

For each dataset, a logistic regression classifier is learned from the extracted sentence embeddings; we report the classification accuracy.

Cross-modal retrieval on COCO: We consider the image search/annotation tasks on the MS COCO dataset BID30 .

A pairwise triplet-loss is optimized in order to bring corresponding sentences and images closer in a multimodal latent space.

Evaluation is performed using Recall@K.

To analyze the quality of the textual space, we report some measures (computed in %) defined on the MSVD test set:??? ?? vis measures if the similarities between sentences correlate with the similarities between videos.??? E intra = E vs=v s [cos(s, s )] measures the homogeneity of each cluster, by measuring the average similarity of sentences within a cluster.??? E inter = E vs =v s [cos(s, s )] measures how well clusters are separated from each other (i.e. average similarity between sentences of two different clusters).

Videos are sampled at a 3 frames per second rate; afterwards, frames are processed using a pretrained VGG network BID45 .

The multimodal loss L is optimized with Adam optimizer (Kingma & Ba, 2014) and a learning rate ?? = 8.10 ???4 .

Hyperparameters are tuned using the Pearson correlation measure on SICK trial: m = m = 0.5, ?? = 2.5.10 ???4 , and mini-batch size of 32 for L V .

We perform extensive experiments with L T based on the SkipThought model, using an embedding size of 2400 and the same network hyperparameters as in BID28 .

The perceptual hypothesis holds that the information within videos is useful to ground sentence representations.

In our model, this hypothesis translates into the perceptual loss L = L P (i.e. model M p .

FIG0 ).

Since the perceptual loss is the only component exploiting video content, we compare, in TAB0 , the different video encoders on intrinsic evaluation benchmarks, namely semantic relatedness.

The first observation is that our model M outperforms the purely textual baseline ST for all video encoders, which shows that perceptual information from videos is useful to improve representations.

We also observe that using random visual anchors (R) improves over ST.

This validates our cluster hypothesis, since grouping visually equivalent sentences improves representation -even when anchors bear no perceptual semantics.

We further observe that F, A, T > R, which shows that the perceptual information from videos brings a more semantically meaningful structure to the representation space.

Finally, regarding the different ways to encode a video, we observe that leveraging more than one frame can be slightly beneficial to learn grounded sentence representations, e.g. A obtains +3.3% average relative improvement over F on ?? SICK P earson .

Selecting relevant frames (T ) in the video rather than considering all frames with equal importance (A) improves the quality of the embeddings.

It is worth noting that discrepancies between the modeling choices F , A, T are relatively low.

This could be explained by the fact that videos from the MSVD dataset are short (10 seconds on average) and contain very few shot transitions.

Thus, nearly all frames can provide a relevant visual context for associated sentences.

We believe that higher differences would be exhibited for a dataset containing longer videos.

In the remaining experiments, we therefore select A as the video model, since it offers a good balance between effectiveness (T ) and efficiency (F ).

We study here the influence of perceptual and cluster information on the embedding space structure.

To do so, we report, in TAB1 , the structural measures on three versions of our model -M c (cluster information), M p (perceptual information) and M b (combination of both), as well as on baselines ST and P. For M and P, we discard the textual loss to isolate the effect of the different hypotheses.

As expected, solely using cluster information leads to the highest E intra and lowest E inter , which suggests that M c is the most efficient model at separating visually different sentences.

Using only perceptual information in M p logically leads to highly correlated textual and visual spaces (highest ?? A vis ), but the local neighborhood structure is not well preserved (lowest E intra and highest E inter ).

M b and P are optimized for both forming well-separated clusters and capturing the perceptual information within the representation space.

This translates into a high E intra and low E inter .

However, the main difference lies in the fact that M b is better at preserving the geometry of the visual space (higher ?? Table 4 .3 reports the effectiveness of the sentence embeddings obtained from our scenarios and baselines on semantic relatedness and classification tasks.

We first observe that multimodal models generally outperform the text-only baseline ST on both semantic relatedness and classification benchmarks.

Interestingly, we notice that the STS/Captions benchmark gives the highest discrepancies compared to the text-only baseline, probably because these sentences have a highly visual content.

Second, we notice that a high ?? T leads to high classification scores, whereas a low ?? T leads to high semantic relatedness scores.

There is a trade-off between semantic relatedness and classification scores, that we can set properly by tuning ?? T .

Indeed, properly weighting the textual contribution in the global loss L is task-dependent, for every grounding model.

This echoes the problem reported in BID12 in the context of word embeddings: there is no strong correlation between the semantic relatedness scores and extrinsic evaluation (e.g. classification) scores.

As a qualitative analysis, we illustrate in Table 3 that, due to our multimodal model, concrete knowledge acquired via visual grounding can be transferred to abstract sentences.

To do so, we manually build abstract sentence queries using words with low concreteness (between 2.5 and 3.5) from the USF dataset BID37 .

Then, nearest neighbors are retrieved from all sentences of the MS COCO training set.

We see that our multimodal model is more accurate than the purely textual model to capture visual meaning, even for sentences that are not inherently visual.

For example, on the first line of Table 3 , ST's sentence contradicts the query by depicting the man as "smiling", whereas M's sentence gives a concrete vision of horror: "grabs his head while screaming".

The observation that perceptual information propagates from concrete sentences to abstract ones is analogous to findings made in previous research on word embeddings .

Table 3 : Qualitative analysis: finding the nearest neighbor of a given query in the textual space.

Query DISPLAYFORM0 A man is horrified An older man in a suit is smiling The man is holding his face and screaming This is a tragedy I think this is a huge food court View from the survivor of a motorcycle accident Two people are in love Two people are out in the ocean kitesurfing A couple of people that are next to each other Table 4 : RQ2,3: Semantic relatedness and classification performances.

M(?? T ) stands for M(?? T , 0.1, 1).

Note that, in all models, sentence vectors have the same dimension (2400).

To further answer RQ2, we compare our model M with the projection baseline P. Our model obtains higher results than P on semantic relatedness tasks and comparable ones on classification tasks.

For example, M p has 5%/3% average relative improvement over P on semantic relatedness tasks.

This suggests that preserving the structure of the visual space is more effective than learning cross-modal projections, as outlined in section 4.2.

Indeed, this statement is strengthened by the fact that our model also improves over a sequential state-of-the-art model BID26 .

Since their textual baseline is weaker than ours (due to differences in the encoder and the dimensionality), we do not report their results in Table 4 .3.

However, we compare, between both approaches, the discrepancy ??? between the best multimodal model and the respective text-only baseline, while keeping dimensionality constant.

On the benchmarks MPQA, MR, SUBJ and MSRP, our ??? is higher than theirs.

To answer RQ3, we compare joint and sequential approaches.

We notice that joint models M and P globally perform better than the sequential baseline SEQ on classification and semantic relatedness tasks.

For instance, M ??? A (500) has 5%/9% average relative improvement (resp.

1%) over SEQ on semantic relatedness (resp.

classification benchmarks).

Therefore, the joint approach shows superior performances to the sequential one, confirming results reported for grounded word embeddings BID53 .

Finally, our models trained from scratch perform slightly better than pretrained ones.

This might be due to the fact that visual and textual information are integrated in a joint manner from the beginning of training, which leads to better interactions between visual and textual modalities.

To further evaluate the quality of the embeddings, we perform cross-modal retrieval experiments on the COCO dataset BID30 .

In TAB5 , we report the results of our best performing models, which corroborates our previous statements on semantic relatedness and classification.

Finally, we probe that our model is independent from the choice of the textual encoder and objective L T , we use the FastSent model BID19 instead of the SkipThought model.

We observe similar improvements in performances (e.g. ??? STS = 4/4 and ??? SICK = 7/7 for the best performing model M p A (0)), confirming that our visual grounding strategy applies to any textual model.

Sentence representations: Several approaches have been proposed over the last years to build semantic representations for sentences.

On the one hand, supervised techniques produce task-specific sentence embeddings.

For example, in a classification context, they are built using recurrent networks with LSTM BID20 , recursive networks BID47 , convolutional networks BID23 , or self-attentive networks BID32 .

On the other hand, unsupervised methods aim at producing more general and task-independent sentence representations.

Closer to our contribution, SkipThought BID28 and FastSent BID19 are based on the distributional hypothesis (Harris, 1954) applied to sentences, i.e. sentences that appear in similar contexts should have similar meanings.

In the SkipThought model, a sentence is encoded with a GRU network, and two GRU decoders are trained to reconstruct the adjacent sentences.

In FastSent, the embedding of a sentence is the sum of its word embeddings; the learning objective is to predict all words in the adjacent sentences using a negative sampling loss.

The present paper extends these works by integrating visual information.

Language grounding: To understand the way language conveys meaning, the traditional approach consists of considering language as a purely symbolic system based on words and syntactic rules BID8 BID4 .

However, Fincher-Kiefer FORMULA2 ; W. BID49 insist on the intuition that language has to be grounded in the real world and perceptual experience.

The importance of real-world grounding is stressed in BID14 , where an important bias is reported: the frequency at which objects, relations, or events occur in natural language are significantly different from their real-world frequency.

Thus, leveraging visual resources, in addition to textual resources, is a promising way to acquire common-sense knowledge BID31 BID52 and cope with the bias between text and reality.

Following this intuition, Multimodal Distributional Semantic Models have been developed to cope with the lack of perceptual grounding in Distributional Semantic Models BID36 BID40 .

Two lines of work can be distinguished.

First, the sequential approach separately builds textual and visual representations and combines them, via concatenation BID25 BID10 , linear weighted combination BID2 , and Canonical Correlation Analysis BID33 .

Second, the joint approach is intuitively closer to the way humans learn language semantics by hearing words and sentences in perceptual contexts.

The advantage is that the visual information of concrete words is transferred to more abstract words that do not necessarily have associated visual data .

Closer to our contribution, BID29 presents the Multimodal Skip-Gram model, where the Word2vec objective BID36 ) is optimized jointly with a max-margin ranking objective aiming at bringing concrete word vectors closer to their corresponding visual features.

Similarly, BID53 show that not only the visual appearance of objects is important to word understanding, but also their context in the image, i.e. surroundings and neighboring objects.

However, these models learn word representations while our model is intended to learn sentence representations.

Very recently, BID26 have set ground for multimodal sentence representations.

The authors propose a sequential method: language-only representations obtained from a text corpus (Toronto BookCorpus) are concatenated to grounded sentence vectors obtained from a caption dataset (MS COCO).

A LSTM sentence encoder is trained to predict the representation of the corresponding image using a ranking loss and/or to predict other captions depicting the same image.

Our work is different in several ways from theirs: we use a joint approach instead of a sequential one, and we distinguish and exploit cluster and perceptual information; moreover, we use videos instead of sentences and our framework is applicable to any textual sentence representation model.

In this paper, we proposed a joint multimodal model to learn sentence representations and our learned grounded sentence embeddings show state-of-the-art performances.

Besides, our main findings are the following: (1) Both perceptual and cluster information are useful to learn sentence representations, in a complementary way.

(2) Preserving the structure of the visual space, by modeling textual similarities on visual ones, outperforms a strategy based on projecting one space into the other.

(3) A joint approach is more appropriate than a sequential method to learn multimodal representation for sentences.

As future work, we would investigate the contribution of the temporal knowledge contained in videos for sentence grounding.

@highlight

We propose a joint model to incorporate visual knowledge in sentence representations

@highlight

The paper proposes a method to use videos paired with captions to improve sentence embeddings

@highlight

This submission proposes a model for sentence learning sentence representations that are grounded, based on associated video data.

@highlight

Proposes a method for improving text-based sentence embeddings through a joint multimodal framework.