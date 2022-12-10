Search engine has become a fundamental component in various web and mobile applications.

Retrieving relevant documents from the massive datasets is challenging for a search engine system, especially when faced with verbose or tail queries.

In this paper, we explore a vector space search framework for document retrieval.

Specifically, we trained a deep semantic matching model so that each query and document can be encoded as a low dimensional embedding.

Our model was trained based on BERT architecture.

We deployed a fast k-nearest-neighbor index service for online serving.

Both offline and online metrics demonstrate that our method improved retrieval performance and search quality considerably, particularly for tail queries

Search engine has been widely applied in plenty of areas on the internet, which receives a query provided by users and returns a list of relevant documents within sub-seconds, helping users obtain their desired information instantaneously.

Numerous technologies have been developed and utilized in real-world search engine systems .

However, the existing semantic gap between search queries and documents, makes it challenging to retrieve the most relevant documents from tens of millions of documents.

Therefore, there is still a large proportion of search requests that can not be satisfied perfectly, especially for long tail queries.

A search engine system is usually composed of three main modules, -query understanding module -retrieval module -ranking module The query understanding module first parses the original query string into a structured query object BID32 .

More specifically, the query understanding module includes several subtasks, such as word segmentation, query correction, term importance analyze, query expansion, and query rewrite, etc.

After the query string was parsed, an index module accepts the parsed query, and then retrieve the candidate documents.

We call this stage the retrieval stage or the first round stage.

Most web-scale search engine systems use the term inverted index for document retrieval, where term is the most basic unit in the whole retrieval procedure.

In the first round stage, the retrieved documents are ranked by a simple relevance model, eg TF-IDF, BM25, and the top-N documents with the highest score are submitted to the next stage for ranking.

Finally, the documents scored largest by a ranking function are returned to users eventually.

For a search system described above, the final retrieval performance is highly enslaved by these query understanding module.

Take word segmentation as an example: this task segments raw continuous query string into a list of segmented terms.

Since the word segmentation algorithm has the risk of wrong segmentation.

If the error segmented term does not appear in the document space, then no document could be retrieved in the first round stage, and it will return a result page without any document which damages the user's experience seriously.

There is a lot of work focused on better understanding queries to retrieve more relevant documents.

However, since the final performance is influenced by all parts of the query understanding module.

Attempts to optimize only one part is usually hard to contribute to a significant enhancement.

To avoid the problems mentioned above, we propose a novel complementary retrieval sys-tem that retrieves documents without the traditional term-based retrieval framework.

That is, instead of parse raw query into a structured query, we directly map both queries and documents into a low dimension of embedding.

Then in the online serving, the k-nearest-neighbor documents of the given query in the latent embedding space are searched for retrieval.

Recently, we have witnessed tremendous successful applications of deep learning techniques in information retrieval circle, like query document relevance matching BID14 BID34 BID33 , query rewriting BID13 , and search result ranking BID12 BID10 .

However, it is still hard to directly retrieve relevant documents using an end2end fashion based on knearest-neighbor search in latent space, especially for long tail queries.

The latest far-reaching advancement in natural language processing with deep learning, BERT BID8 , provides a turning point to make end2end retrieval realizable.

In this paper, we present a document retrieval framework as a supplement to the traditional inverted index based retrieval system.

We design a new architecture to retrieve documents without a traditional term-based query understanding pipeline, which avoids performance decay by each subtask of query understanding.

We use BERT architecture as the general encoder of query and document strings, then we fine-tuned the pre-trained BERT model with human annotated data and negative sampling technique.

Finally, we conduct both offline and online experiments to verify our proposed method.

To sum up, our main contributions are described below:1.

We design a novel end2end document retrieval framework ，which is a supplement to traditional term-based methods.2.

Our model is trained on transformer architecture, and a series of training techniques are developed for performance enhancement.3.

The proposed techniques can not only be used in document retrieval but also have a significant improvement for search ranking.

The rest of the paper is organized as follows.

We concisely review the related work in Section 2.

Sections 3 mainly describes our proposed methods.

Offline and online experiments are detailed given in Section 4 and Section 5 respectively.

Finally, we conclude and discuss future work in Section 6.

There is a variety of work on search query understanding BID29 , including query correction BID5 , query term weighting BID39 , query expansion(Azad and Deepak, 2017) and query reformulation BID3 .

In general, these kinds of methods coherently rewrite the raw query into a new query, by replacing, adding, or removing terms or phrases in the raw query.

The rewritten query gets better expression and therefore can retrieve more relevant documents than the original one.

Besides the inverted index, vector search engines BID9 have also been widely applied in many information seeking tasks, like image search BID16 and recommendation system BID6 .To retrieve documents using a vector search, we need to map a piece of text into a low-dimensional numerical vector.

Various embedding techniques have been developed and proven to have the powerful capability of capturing the semantic meaning of natural language text BID23 , BID27 BID22 .

However, these kinds of models are still not capable of complicating text encoding, especially for long tail text queries.

More recently, researchers have been describing the various architecture of neural models BID24 .

In text relevance matching area, we can divide most models into two typical categories, namely representation BID14 based models and interaction based models BID26 BID35 BID7 .

The representation models , like DSSM, are trained to obtain high-level representations of query and document respectively, then use vector distance between the query and document embedding for text relevance score.

While the interaction based models first compute the term correlation matrix between query and doc-uments and calculate semantic matching similarity based on the correlation matrix.

Both representation models and interaction based models could be trained from massive click feedback data BID18 BID0 or industrial annotation.

These two kinds of model architecture are broadly deployed in realworld search engine systems, especially in ranking phase.

For the representational models, once we obtained the high-level representation of raw texts, we can retrieve documents through the knearest-neighbor space search.

However, the performance of representation based models are usually poorer than interaction based models, which makes k-nearest-neighbor retrieval hard to deploy in the real-world systems, since too many irrelevant documents retrieved may even damage overall performance.

BID38 developed an architecture to transform the text into a sparse representation, while they still retrieve documents using a term-based index like lucene 1 because the nonzero value in the sparse representations is treated as virtual terms.

BID2 BID11 ) developed a uniform query and document embedding framework by generating ngram embedding using user session and click data, and then generalize it to arbitrary text by mean average pooling of ngram embedding.

Since ngram is a common and effective skill in a variety of NLP tasks, training a good ngram representation requires a massive of datasets, which may be a bottleneck for many researchers and companies.

Meanwhile, the model capacity of DSSM and its' variations makes it not capable to capture complex semantic meanings of natural language.

Recently, ELMo(Peters et al., 2018), GPT-2 BID31 and BERT BID8 show the great power of unsupervised pretraining in NLP tasks.

The BERT model is built on a 12 layer transformer architecture, pre-trained with large scale text data.

The pre-trained models can be fine-tuned easily and outperform many state-of-art models in various NLP tasks.

We used the pre-trained BERT-Base(Chinese) 2 model released by Google and fine-tuned the model for semantic representation.

Our fine-tuned model outperformed many state-of-art models in deep relevance matching, and obtain a great success in se-1 https://lucene.apache.org/ 2 https://github.com/google-research/bert

In this section, we first illustrate our proposed semantic retrieval framework, which is composed of both offline and online parts respectively.

Then, we introduce the model structure used for encoding queries and titles, and the techniques we used to boost the performance.

Figure 1 shows our proposed system architecture.

The offline module includes model training, document embedding inference, and semantic index builder.

While in online serving, both query's semantic embedding and traditional term base query parser are computed, and then those two results are sent to semantic index service and inverted index service respectively for document retrieval.

Finally, documents retrieved from both index services are merged and sent to ranking service for document scoring.

The pre-trained BERT model can be leveraged for semantic ranking and matching BID30 in various ways.

We developed two models here: BERT(rep) and BERT(rel).

The BERT(rep) model uses the pre-trained BERT model to obtain embedding of query and doc respectively, while the BERT(rel) model concatenates query and document first and get the one representation for a query document pair.

The final score of a query, document pair is computed as below: DISPLAYFORM0 In the equation 1, we use the mean average of last layer as encoder output for each query and document, and compute the dot product of two embedding as matching score, where L represents the max sequence length.

We also tried directly using the last layer of [CLS] term's embedding, but performed worse than the average pooling described in equation 1.

DISPLAYFORM1 The equation 2 use embedding of last layer's [CLS] token and weighted sum it to a scalar by vector w, where w is a full connection layer with only one output.

The model capacity of this method is more powerful than BERT(rel) because it calculates the term interaction between the query and document in the self-attention layers.

However, since the BERT(rel) model is an interaction base model, this model can not be applied to semantic retrieval.

Both two models are trained through a supervised learning fashion, with a pairwise max margin hinge loss to distinguish relatively positive and negative samples.

The loss function for one query is: DISPLAYFORM2 where p i and p j represent to model score computed for each < query, document > pair, and y i and y j is the label for each document respectively.

τ is the hyper parameter called margin to determine how far the model need to push a pair away from each other.

The margin parameter is tuned for the best performance here.

We use the additive data sampling technique to further enhance model performance.

Therefore, the data we used to train our model is comprised of two parts, human annotated data, and negative sampled data.

Negative sampling has been successfully applied in many tasks, such as neural language modelling BID23 , e-commerce list embedding BID10 , graph embedding BID37 and so on.

Sampling negative training instance is also useful for model training in this scenario, since different from traditional term-based retrieval method, the vector space search is much more likely to retrieve irrelevant documents.

Thus we propose to augment more irrelevant documents.

When the negative samples were added to training, the model learned to push relevant and irrelevant documents away from each other, then the model is more robust to noisy documents.

A straightforward way of negative sample mining is to select negative samples corresponding to a uniform distribution over the whole corpus, in particular, irrelevant documents here.

However, this simple strategy fails to generate hard negative samples, which provide more important information for the model.

Therefore, we propose another negative sampling method.

At first, we train a baseline model with only human annotated data.

Then we use this model to encode documents and queries.

After that, we use an unsupervised cluster algorithm to assign each document and query a cluster id.

Finally, we uniformly random selected negative documents from the cluster that query was distributed.

For convenient, we call this kind of negative sampling name of N EG cluster , and globally sampled data name of N EG global .

We append N EG cluster and N EG global to the raw dataset for per query and fine-tuned the model again to obtain our final model.reWe show the whole training procedure in the Algorithm 1 Training Framework of our proposed model

human annotated data D, BERT pre-trained model M 1: M1 ← {D, M }, fine-tune the model M by D 2: compute embedding E for query and doc using M1 3: compute cluster centroids C by E 4: for all d ∈ Docs do 5:compute closest centroid C d for d 6: end for 7: for all q ∈ Query do 8:compute closest centroid Cq for q 9: uniform sample N EG global from whole doc set 10: uniform sample N EG cluster among docs where DISPLAYFORM0 D1(q) = {D(q) ∪ N EG global ∪ N EG cluster } 12: end for 13: M2 ← {D1, M }, fine-tune the model M by D1 Ensure:BERT model M2Algorithm 1, and sample's meaning is much closer to query than that of N EG global , which makes the model more robust for hard samples.

Once the model was trained, we need to serve it on the fly.

We first computed the embedding of all documents and build a vector index using faiss 3 BID19 , which was open sourced by facebook and support k-nearestneighbor search for vector data in milliseconds.

We developed a c++ based semantic index server to provide efficient concurrent online service.

Our model was inferenced on a GPU server, and inference speed was accelerated 2 times faster than tfserving through a c++ based library developed by us.

During the online serving, when a query was received, the GPU server first inferences the query embedding, and downstream sends the query embedding to semantic index service for document retrieval.

For the balance of efficiency and effect, we retrieve k most similar documents in the semantic service for next stage ranking, where k is set to 20 here.

In this section, we carry out offline experiments to illustrate the performance of our proposed semantic retrieval methods.

In the experiment, we train the model with 1 epoch, use Adam(Kingma and Ba, 2014) with a learning rate of 10 −5 , β1 = 0.9, β2 = 0.999.

The data annotated by human editors is a list of triplets like <query, doc, relevance>. The rele-3 https://github.com/facebookresearch/faiss vance score has three grade 0, 1, 2, which represents bad, f air and excellent respectively.

The dataset contains 36159 queries and 1181229 query doc pairs.

Beside the dataset for training, we additionally annotated a small dataset for test, the test dataset contains 2703 queries and 84244 querydoc pairs.

The summarize of dataset is shown at Table 2.

We evaluate our proposed model from ranking and retrieval aspects.

We compared the ranking performance using Normalized Discounted Cumulative Gain(NDCG), and retrieval performance with Recall.

The way how these metrics are calculated will be introduced in Section 4.4 and Section 4.5 respectively.

• ClickSim A relevance matching model BID17 which use web-scale click data to generate term representations for query and document, and use cosine similarity to represent query document relevance.• K-NRM An interaction based matching model using kernel pooling BID35 .•

Match Pyramid An interaction based matching model using convolutions on term matching matrix BID25 ).•

DSSM A representation based model proposed by Microsoft Research BID14 .

The model proposed here using word vectors pretrained on document title corpus.

And three full connection layer with size of 300, 300, and 128 dimensions are used for text encoding.

We use metric Recall to evaluate the model's retrieval performance here.

This metric measures how many relevant documents are retrieved by a given model.

For a given query q, the Recall rate is calculated as, DISPLAYFORM0 where Ret q represents the retrieved documents for q, Rel q stands for all the relevant documents for query q, where relevant documents are defined as document relevance annotated larger than 0 here.

To evaluate the recall performance offline, we first built semantic index both for our model and baseline model.

We computed representation for document title of each model, then we used the representation embedding to build semantic index.

Once queries' embedding of each model were computed, we retrieved the top k documents by knearest-neighbor search.

Besides comparing the recall measure of different models only using semantic index, we compared the recall enhancement when the semantic index was added to the lexical inverted index.

We used a commercial term-based inverted index engine developed by us and build a lexical index with it.

Both lexical inverted index and semantic index were built to retrieve documents, with top 300 and top 20 respectively.

Then we calculated the recall of the union set.

In the experiment, since document size of testset is small, we need a larger document corpus to make the recall measured more accurately.

Therefore, both semantic index and lexical index were built with all human annotated data, including trainset and testset.

And recall metric were calculated using only queries in the testset.

TAB3 shows the result of different models, BERT(rep) outperforms baseline model DSSM significantly in the recall measure.

And after adding our model as a supplement to the lexical index, the recall rate is improved from 54.9% to 69.4%.

While the baseline model, DSSM performs poorly on this task.

-NDCG score Since our proposed model could not only be applied in document retrieval but also applied in the ranking stage.

We measured the model's ranking quality through Normalized Discounted Cumulative Gain(NDCG).

For a ranked document list, the NDCG for a query is calculated as, DISPLAYFORM0 where IDCG n represents the DCG score when the list was perfectly ranked by relevance.

We compute following variation of Discounted Cumulative Gain(DCG) BID15 , DISPLAYFORM1 According to the equation 6, higher relevance label contribute to higher weight in the computation.

We calculate N DCG with different rank list size of {1, 3, 5} respectively.

Table 4 shows that our model is superior to the state of art deep relevance matching models, and BERT(rep) model is slightly worse than the BERT(rel) model since BERT(rel) model uses self-attention between the query and title tokens before aggregates final score.

However, both the BERT(rep) model and BERT(rel) model outperform other baselines significantly.

We feed the doc product of query doc embedding into a gbdt ranking model BID4 as a relevance feature, and observe the feature importance after the tree model was trained.

The feature importance was computed by the statistics collected during the tree ensemble training procedure.

TAB5 shows that without adding BERT(rel) feature, the BERT(rep) feature ranks first in the ranking function, and accounts for 34% of importance among all features in our ranking function.

In Section 3.2.1, we described two negative sampling generator method: the N EG global samples and N EG cluster for training data enhancement.

We tuned the negative samples size, and obtained the best performance with 10 N EG global and 10 N EG cluster respectively.

After adding negative samples, the average negative sample size for a given query increased from 19.9 to 39.9.

TAB6 shows the model performance with different kinds of negative samples.

Only adding N EG global can improve NDCG@3 at about 0.5%, when adding N EG cluster , the NDCG@3 is further improved by 0.8%.

Therefore, the overall measurements are enhanced by 1.4% after additive sampling.

pooling layer used in BERT model

In this paper, we use the reduce-mean of the last layer as BERT(rep) model's pooled output.

Different layers of BERT may own different aspects of knowledge about the input sequence.

To verify the effectiveness of different layers, we trained different models, with pooled output from different layers respectively.

From FIG1 , the red solid line shows that the layer closest to last obtains higher NDCG measure.

This is reasonable since higher layers make the model contains more parameters.

Besides comparing the results of different layers, we also developed a method aggregate the embedding of all layers.

In this method, an attention layer calculates the weight across different layers, therefore a weighted sum of each layer's embedding on each position is the final representation of each term.

After that, we used reduce-mean of all terms' embedding as the final pooled output.

The result of aggregation is shown as the green dot line, which does not outperform simple average pooling on the last layer.

Meanwhile, we also tried using [CLS] term's embedding of the last layer as pooled output, but it behaved even worse.

In conclusion, using mean average pooling of the last layer as final pooled output performs best in this scenario, even though some work claims aggregating layers is useful BID21 .5 Online Evaluation and Case Study

After offline evaluations, we conduct an online a/b test to further verify our proposed system.

In the online experiment procedure, 40 percent of online traffic were randomly distributed to four groups, 2 control groups, and 2 experimental groups.

The metric we used to evaluate is the Clicked Search Rate(CSR), which is computed as: DISPLAYFORM0 After a week's observation, as shown in TAB7 , the overall CSR of two experimental groups both surpass two control groups by 0.65%, which is relatively a huge improvement to our experience.

We also examined the online performance for queries with different frequency.

We split queries into Top, Torso, and Tail by query search times in a day.

Since our proposed method mainly focuses on boosting the performance of long tail queries, we can see the CSR metric is not significant in the Top and Torso query part.

But the metric increased by nearly 1.05 % in the Tail part, which contributed to the largest algorithm iteration in the first half of 2019.

This section highlights some good cases after our system was deployed online.

We show the final result ranked at top 6 for query "送外卖不认识路" (do not know the way to deliver food) at TAB8 , where SEMANTIC represents the document retrieved from the proposed semantic index, and LEXICAL for traditional termbased inverted index.

In this case, three documents are retrieved from semantic index, and the relevance is also much better than the document from traditional inverted index.

Notice that there are many ways to express "不 认 识 路"(do not know the way) in Chinese, while the semantic index retrieved documents indeed capture the several alternatives of expressing it: "不知道路线", "不认路", "不懂路".

And the term retrieved document only contains the same term "不认识路" as query expressed.

In this paper, we present an architecture for semantic document retrieval.

In this architecture, we first train a deep representation model for query and document embedding, then we build our semantic index using a fast k-nearest-neighbor vector search engine.

Both offline and online experiments have shown that retrieval performance is greatly enhanced by our method.

For the future work, we would like to explore a more general framework that could use more signals involved for semantic retrievals, like document quality features, recency features, and other text encoding models.

@highlight

A deep semantic framework for textual search engine document retrieval