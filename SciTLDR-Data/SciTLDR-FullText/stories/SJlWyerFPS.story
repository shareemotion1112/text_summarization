The objective in deep extreme multi-label learning is to jointly learn feature representations and classifiers to automatically tag data points with the most relevant subset of labels from an extremely large label set.

Unfortunately, state-of-the-art deep extreme classifiers are either not scalable or inaccurate for short text documents.

This paper develops the DeepXML algorithm which addresses both limitations by introducing a novel architecture that splits training of head and tail labels.

DeepXML increases accuracy by (a) learning word embeddings on head labels and transferring them through a novel residual connection to data impoverished tail labels; (b) increasing the amount of negative training data available by extending state-of-the-art negative sub-sampling techniques; and (c) re-ranking the set of predicted labels to eliminate the hardest negatives for the original classifier.

All of these contributions are implemented efficiently by extending the highly scalable Slice algorithm for pretrained embeddings to learn the proposed DeepXML architecture.

As a result, DeepXML could efficiently scale to problems involving millions of labels that were beyond the pale of state-of-the-art deep extreme classifiers as it could be more than 10x faster at training than XML-CNN and AttentionXML.

At the same time, DeepXML was also empirically determined to be up to 19% more accurate than leading techniques for matching search engine queries to advertiser bid phrases.

Objective: This paper develops the DeepXML algorithm for deep extreme multi-label learning applied to short text documents such as web search engine queries.

DeepXML is demonstrated to be significantly more accurate and an order of magnitude faster to train than state-of-the-art deep extreme classifiers XML-CNN (Liu et al., 2017) and AttentionXML (You et al., 2018) .

As a result, DeepXML could efficiently train on problems involving millions of labels on a single GPU that were beyond the scaling capabilities of leading deep extreme classifiers.

This allowed DeepXML to be applied to the problem of matching millions of advertiser bid phrases to a user's query on a popular web search engine where it was found to increase prediction accuracy by more than 19 percentage points as compared to the leading techniques currently in production.

Deep extreme multi-label learning: The objective in deep extreme multi-label learning is to learn feature representations and classifiers to automatically tag data points with the most relevant subset of labels from an extremely large label set.

Note that multi-label learning is a generalization of multi-class classification which aims to predict a single mutually exclusive label.

Notation:

Throughout the paper: N refers to number of training points, d refers to representation dimension, and L refers to number of labels.

Additionaly, Y refers to the label matrix where y ij = 1 if j th label is relevant to i th instance, and 0 otherwise.

Please note that differences in accuracies are reported in absolute percentage points unless stated otherwise.

Matching queries to bid phrases: Web search engines allow ads to be served for not just queries bidded on directly by advertisers, referred to as bid phrases, but also for related queries with matching intent.

Thus matching a query that was just entered by the user to the relevant subset of millions of advertiser bid phrases in milliseconds is an important research application which forms the focus of this paper.

DeepXML reformulates this problem as an extreme multi-label learning task by treating each of the top 3 Million monetizable advertiser bid phrases as a separate label and learning a deep classifier to predict the relevant subset of bid phrases given an input query.

For example, given the user query "what is diabetes type 2" as input, DeepXML predicts that ads corresponding to the bid phrases "what is type 2 diabetes mellitus", "diabetes type 2 definition", "do i have type 2 diabetes", etc.

could be relevant to the user.

Note that other high-impact applications have also been reformulated as the extreme classification of short text documents such as queries, webpage titles, etc.

For instance, (Jain et al., 2019) applied extreme multi-label learning to recommend the subset of relevant Bing queries that could be asked by a user instead of the original query.

Similarly, extreme multi-label learning could be used to predict which subset of search engine queries might lead to a click on a webpage from its title alone for scenarios where the webpage content might not be available due to privacy concerns, latency issues in fetching the webpage, etc.

State-of-the-art extreme classifiers: Unfortunately, state-of-the-art extreme classifiers are either not scalable or inaccurate for queries and other short text documents.

In particular, leading extreme classifiers based on bag-of-words (BoW) features (Prabhu et al., 2018b) and pretrained embeddings (Jain et al., 2019) are highly scalable but inaccurate for documents having only 3 or 4 words.

While feature engineering (Arora, 2017; Joulin et al., 2017; Wieting & Kiela, 2019) , including taking sub-word tokens, bigram tokens, etc can ameliorate the problem somewhat, their accuracy still lags that of deep learning methods which learn features specific to the task at hand.

However, such methods, as exemplified by the state-of-the-art XML-CNN (Liu et al., 2017) and AttentionXML (You et al., 2018) , can have prohibitive training costs and have not been shown to scale beyond a million labels on a single GPU.

At the same time, there is a lot of scope for improving accuracy as XML-CNN and AttentionXML's architectures have not been specialized for short text documents.

Tail labels: It is worth noting that all the computational and statistical complexity in extreme classification arises due to the presence of millions of tail labels each having just a few, often a single, training point.

Such labels can be very hard to learn due to data paucity.

However, in most applications, predicting such rare tail labels accurately is much more rewarding than predicting common and obvious head labels.

This motivates DeepXML to have specialized architectures for head and tail labels which lead to accuracy gains not only in standard metrics which assign equal weights to all labels but also in propensity scored metrics designed specifically for long-tail extreme classification.

DeepXML: DeepXML improved both accuracy and scalability over existing deep extreme classifiers by partitioning all L labels into a small set of head labels, with cardinality less than 0.1L, containing the most frequently occuring labels and a large set of tail labels containing everything else.

DeepXML first represented a document by the tf-idf weighted linear combination of its word-vector embeddings as this architecture was empirically found to be more suitable for short text documents than the CNN and attention based architectures of XML-CNN and AttentionXML respectively.

The word-vector embeddings of the training documents were learnt on the head labels where there was enough data available to learn a good quality representation of the vocabulary.

Accuracy was then further boosted by the introduction of a novel residual connection to fine-tune the document representation for head labels.

This head architecture could be efficiently learnt on a single GPU with a fully connected final output layer due to the small number of labels involved.

The word-vector embeddings were then transferred to the tail network where there wasn't enough data available to train them from scratch.

Accuracy gains could potentially be obtained by fine tuning the embeddings but this led to a dramatic increase in the training and prediction costs.

As an efficient alternative, DeepXML achieved state-of-the-art accuracies by fine tuning only the residual connection based document representation for tail labels.

A number of modifications were made to the highly scalable Slice classifier (Jain et al., 2019) for pre-trained embeddings to allow it to also train the tail residual connection without sacrificing scalability.

Finally, instead of learning an expensive ensemble of base classifiers to increase accuracy (Prabhu et al., 2018b; You et al., 2018) , DeepXML improved performance by re-ranking the set of predicted labels to eliminate the hardest negatives for the base classifier with only a 10% increase in training time.

Results: Experiments on medium scale datasets of short text documents with less than a million labels revealed that DeepXML's accuracy gains over XML-CNN and AttentionXML could be up to 3.92 and 4.32 percentage points respectively in terms of precision@k and up to 5.32 and 4.2 percentage points respectively in terms of propensity-scored precision@k.

At the same time, DeepXML could be up to 15× and 41× faster to train than XML-CNN and AttentionXML respectively on these datasets using a single GPU.

Furthermore, XML-CNN and AttentionXML were unable to scale to a proprietary dataset for matching queries to bid phrases containing 3 million labels and 21 million training points on which DeepXML trained in 14 hours on a single GPU.

On this dataset, DeepXML was found to be at least 19 percentage points more accurate than Slice, Parabel (Prabhu et al., 2018b) , and other leading query bid phrase-matching techniques currently running in production.

Contributions: This paper makes the following contributions: (a) It proposes the DeepXML architecture for short text documents that is more accurate than state-of-the-art extreme classifiers; (b) it proposes an efficient training algorithm that allows DeepXML to be an order of magnitude more scalable than leading deep extreme classifiers; and (c) it demonstrates that DeepXML could be significantly better at matching user queries to advertiser bid phrases as compared to leading techniques in production on a popular web search engine.

Source code for DeepXML and the short text document datasets used in this paper can be downloaded from (Anonymous, 2019) .

Much work has been done in extreme multi-label classification which can be broadly categorized in two categories: a) learning the classifier with pre-computed features (Agrawal et al., 2013; Cissé et al., 2013; Prabhu & Varma, 2014; Yu et al., 2014; Weston et al., 2013; Mineiro & Nikos, 2014; 2017; 2019; Yen et al., 2016; Xu et al., 2016; Babbar & Schölkopf, 2017; 2019; Niculescu-Mizil & Abbasnejad, 2017; Papanikolaou & Tsoumakas, 2017; Prabhu et al., 2018a; Barezi et al., 2019; Jasinska et al., 2016) , b) jointly learning feature representation along with the classifier (Chen & Lin, 2012; Balasubramanian & Lebanon, 2012; Bi & Kwok, 2013; Liu et al., 2017; Jernite et al., 2017; Wydmuch et al., 2018; Bhatia et al., 2015; Tagami, 2017; You et al., 2018; Krichene et al., 2019; Barezi et al., 2019) .

More extensive survey of extreme classification and deep learning approaches can be found in section A.5 in the supplementary material.

Traditionally, extreme classification approaches used sparse BoW features due to fast & efficient feature computation as well as state-of-the-art performance for a large number of labels.

However, for short text documents such as queries, which form the focus of the paper, deep learning representations are more effective than BoW (Jain et al., 2019) .

Unfortunately, existing deep learning approaches for extreme classification yielding state-of-the-art accuracy are not scalable, while scalable approaches have not been shown to yield state-of-the-art accuracy.

In particular, XML-CNN, acheives state-ofthe-art accuracy on short text documents but have not been shown to scale beyond a million labels, whereas AttentionXML was found to be slightly more scalable but less accurate as shown in table 1.

Scalability: Deep learning techniques have been very successful and comprehensively beaten BoW features in small output space (Kim, 2014; Yang et al., 2016) .

Unfortunately, the scalablity of such techniques degrades in extreme multi-label (XML) setting as the final fully connected layer leads to cost linear in number of labels (Liu et al., 2017) .

Traditional approach to solve this problem is negative sampling (Mikolov et al., 2013) .

Unfortunately, at the extreme scale, negative sampling has to be applied more aggressively as demonstrated in Fig 5 in the appendix.

To eliminate this problem, many approaches have been proposed such as tree based (Prabhu et al., 2018b; You et al., 2018; Mikolov et al., 2013; Jernite et al., 2017) and hashing based (Shrivastava & Li, 2014; Vijayanarasimhan et al., 2014) , approximate nearest neighbor sub-sampling techniques (Jain et al., 2019; Reddi et al., 2018) .

Adding to that, prediction time complexity remains O(dL), which is not suitable for real-time predictions.

Approaches such as Maximum Inner Product Search (Yen et al., 2018) and Local Sensitive Hashing (Niculescu-Mizil & Abbasnejad, 2017) can speed-up one-vs.-all inference but it leads to further loss in accuracy when applied post-training.

Accuracy: Tail labels are harder to predict as compared to head labels due to training data scarcity but might also be more informative and rewarding in XML setting.

For instance, predicting tag "Artificial intelligence researchers" could be more informative than tag "1954 deaths" on Wikipedia page of "Alan Turing".

Hence, propensity based precision and nDCG are the focus of the paper. (Wei & Li, 2018) demonstrates that trimming tail labels leads to marginal decay in performance on vanilla metrics.

However, a) trimming tail labels lead to loss in propensity based metrics, b) this approach has been demonstrated to scale to ≈ 30K labels only.

Researchers have also tried to improve performance on tail labels by directly optimizing propensity scored metrics , posing learning in the presence of adversarial perturbations (Babbar & Schölkopf, 2019) and treating tail labels as outliers (Xu et al., 2016) .

Although, these approaches boost performance on tail labels, however they are not well suited for short text documents due to, a) support only for fixed features and, b) large training and prediction time.

Matching user queries to bid phrases: Approaches in this domain can be categorized as embeddings based (Jain et al., 2019) , sequence-to-sequence models (Gao et al., 2012; Jones et al., 2006; Riezler & Liu, 2010; Lian et al., 2019) and query graph based models (Ioannis et al., 2008) .

Unfortunately, the trigger coverage, suggestion density and quality of recommendations could be poor for many of these techniques.

For instance, query graph based methods can only recommend suggestions for previously seen triggers thereby limiting their trigger coverage.

Additionally, Sequence-to-sequence models suffers from expensive training and prediction cost.

Although, efficient structures such as trie (Lian et al., 2019) have been deployed to reduce output complexity, however, at the cost of limited bid phrase coverage.

DeepXML-h DeepXML-t Residual Block

State-of-the-art deep extreme classifiers are neither scalable nor accurate for short text documents because: (a) representations learnt through CNNs (XML-CNN), LSTM+Attention (AttentionXML) or context (Bert (Devlin et al., 2018) , Elmo (Peters et al., 2018) , etc.) might not be accurate given limited tail data with only a few words per document and might also be expensive to compute; and (b) they face scalability issues as the final fully connected output layer has millions of outputs making both the forward pass as well as gradient backpropagation prohibitive for even a single training point.

DeepXML addresses these limitations by using a feature representation inspired by FastText (Joulin et al., 2017) and an output layer inspired by Slice (Jain et al., 2019) as these have been demonstrated to be both accurate and scalable for short text documents in their individual capacities.

Unfortunately, combining FastText and Slice in a straight forward fashion also turns out to be neither accurate nor scalable.

The rest of this section details the design choices and modifications that needed to be made in order to get the combination to work.

DeepXML, FastText & Slice: FastText represents a document as an efficiently computable linear combination of its (sub) word-vector embeddings making it highly scalable and well suited for extreme classification scenarios.

Unfortunately, the FastText architecture is linear and low-capacity thereby leading to a loss in accuracy when there isn't a vast amount of unsupervised data available for training.

Furthermore, previous attempts (Wydmuch et al., 2018) at learning FastText representations in a supervised manner by replacing the fully connected output layer by a fixed Probabilistic Label Trees (PLT) extreme classifier (Jasinska et al., 2016) have led to even worse accuracies than XML-CNN and AttentionXML.

Replacing the fixed tree PLT by learnt Slice improves accuracy somewhat but does not lead to state-of-the-art results and also greatly increases training time (please see ablation experiments in Section 5).

DeepXML addresses these limitations by adding a non linearity and residual block to make up for the lack of FastText's expressive power and training the word-vector embeddings using a fully connected output layer on a small number of head labels rather than through Slice.

Please refer to Figure 1 for full architecture of DeepXML.

Once the word-vector embeddings have been trained on the head, they are frozen and transferred to the tail where only the residual block is fine tuned.

This increases both accuracy as there isn't enough training data available to learn good quality word embeddings from scratch on the tail as well as scalability as fine-tuning the head embeddings on the tail would prove too expensive on large problems.

DeepXML-h: Model parameters i.e. words embeddings, residual block and fully connected classifier are learnt with Binary Cross Entropy loss and Adam optimizer.

DeepXML-h can be efficiently trained with a fully connected final layer on a single GPU, as L h (Label set containing head labels)contains only a small subset of labels.

In practice, the size of L h does not grow beyond 0.2M even for datasets with millions of labels.

Additionally, an approximate nearest neighbour search (ANNS) structure was trained over label centroids L

Here, |s| is the size of label shortlist queried from ANNS structure during prediction which is kept as 300 in practice.

Unfortunately, ANNS trained over label centroids may lead to poor recall values when a single centroid is unable to capture diversity in training instances for labels with highly different contexts (say L h ).

For instance, 280K articles, ranging from scientists to athletes, are tagged with the 'living people' tag in WikiTitle-500K (Bhatia et al., 2016) dataset.

Slice increases the shortlist to improve recall; however, at the cost of 6× increase in prediction time.

DeepXML-h tackles this issue by allowing multiple representations for labels in L h .

Documents are clustered using the KMeans algorithm into c clusters for each label in this set.

Therefore, c representatives for each label in L h are computed.

This leads to 5% increase in recall@300 and 6% precision@300 with a shortlist of size 300.

Clustering will lead to |L h − L h | + c|L h | label representations and hence could potentially lead to increased time complexity.

However, clustering just the top 3 labels into 300 clusters seems to work well in our experiments without significant change in training time, and it doesn't impact prediction time at all.

DeepXML-t: DeepXML still relies on Slice to fine-tune the residual block in the tail network and learn the weights in the fully connected output layer for millions of tail labels.

Slice cuts down the time for both the forward pass and the gradient backpropagation from linear to logarithmic in the number of labels.

Slice achieves this by first representing each label by the unit normalized mean of the feature vectors of the training points tagged with the label.

It then learns an ANNS data structure (Malkov & Yashunin, 2016) over the label representations to determine the most likely labels for a given data point (please see the supplementary material for a more detailed description).

This technique was shown to efficiently scale to millions of labels and training points when the feature representation was fixed (Jain et al., 2019) .

Unfortunately, when the feature representation is being learnt, the ANNS data structure needs to be constantly updated as the label representations change with each training batch.

This can lead to a marked slowdown as maintaining the ANNS data structure can incur significant computational cost.

DeepXML speeds up training by redefining the label representation to be the unit normalized mean of the document representation before the residual block computed using the learnt word embeddings alone.

This allows the ANNS data structure to be learnt just once after the word embeddings have been learnt on the head labels and before training starts on the tail.

Unfortunately, this also leads to a loss in training accuracy as the label representation is now an approximation to the true representation that should have been defined as the unit normalized mean of the document representations computed after the residual block.

This loss in accuracy can be compensated by requiring Slice to generate 3x more nearest labels to a given data point.

This allowed the hard negative labels to now be present in the longer list but this significantly increased training time.

It was empirically determined that a more efficient strategy was to extend the shortlist by adding randomly sampled negative labels as this led to no loss in accuracy with only a minimal increase in training time.

Initially, classifier scores and ANNS scores from both DeepXML-h and DeepXML-t are merged in a single vectorŷ.

As demonstrated in Fig. 1 ,ŷ clf −h is the classifier score (logit) from DeepXML-h andŷ anns−h is the cosine similarity from ANNS-h.

Similarly,ŷ clf −t is the classifier score from DeepXML-t andŷ ann−t is the cosine similarity from ANNS-t.

The final DeepXML score vector (ŷ) is computed as follows :

Note thatŷ clf −t ,ŷ clf −h ,ŷ anns−h , andŷ anns−t are sparse vectors, σ is a sparse-sigmoid function computed only at non-zero entries and β ∈ [0, 1].

The average cost of prediction can be broken down into the following four components: a) computing dense feature representation: O(dγ), b) generating a shortlist: O(d log |L t |), c) computing classifier scores: O(d|s|).

Here, γ is the average number of features per document and |s| is the shortlist size in ANNS.

Extreme classifiers such as Parabel (Prabhu et al., 2018b) , AttentionXML (You et al., 2018) learn multiple models to get better prediction accuracy.

However, this leads to increased training and prediction time (200% for both Parabel and AttentionXML) linearly with the number of models.

Whereas, DeepXML-RE learns a re-ranker with training cost logarithmic in the number of labels by training over a shortlist of negative labels.

Specifically, false positive labels predicted by DeepXML (a.k.a hardest negatives) are selected as negatives for DeepXML-RE.

This leads to only 10 − 20% increase in training time.

The architecture of DeepXML-RE is same as DeepXML-t, i.e., a word embedding layer, residual block, and classifier.

Model parameters are learnt with binary cross entropy loss and SparseAdam optimizer.

During prediction, DeepXML-RE evaluates on labels shortlisted by DeepXML only, thereby incurring a prediction cost of O(d|s|).

Experiments were carried out on the Query to Bid phrases (Q2B-3M) dataset, with 3 million labels, by mining the logs of a popular search engine.

Each user query was treated as an instance and relevant advertiser bid phrases became its labels.

Unfortunately, only Slice (Jain et al., 2019) and Parabel (Prabhu et al., 2018b ) could scale to this dataset.

DeepXML is also compared to state-of-the-art methods for query keyword prediction such as Simrank++ (Ioannis et al., 2008 ) and a sequence-to-sequence model based on BERT (Devlin et al., 2018; Lian et al., 2019) .

Experiments were also carried out on four moderate size datasets (Anonymous (2019) ).

The applications considered were tagging Wikipedia pages (WikiTitles-500K), suggesting relevant articles (WikiSeeAlsoTitles-250K), and item-to-item recommendation of Amazon products (AmazonTitles-670K and AmazonTitles-3M).

Please refer to supplementary Table 4 for dataset statistics.

For these moderate size datasets DeepXML was compared to leading deep learning and BoW feature-based methods including XML-CNN, Attention-XML, Slice, AnnexML (Tagami, 2017) , PfastreXML , Parabel (Prabhu et al., 2018b) , XT (Wydmuch et al., 2018) , and DiSMEC (Babbar & Schölkopf, 2017) .

The implementation of all algorithms was provided by the respective authors.

DeepXML has 7 hyperparameters: (a) learning rate and epochs for DeepXML-h; (b) learning rate and epochs for DeepXML-t; (c) embedding dimensions (d); (d) shortlist size (|s|) and (e) threshold to split label set into head and tail labels.

Results are reported for d = 300 and |s| = 300.

The label threshold is chosen via cross-validation (please refer to section A.2 in the supplementary material).

Binary cross-entropy loss and the SparseAdam optimizer were used to update the model parameters.

Please refer to Table 5 in the supplementary section for full parameter settings.

Table 1 shows that DeepXML and DeepXML-RE can outperform state-of-the-art BoW-based approaches as well.

Note that on AmazonTitles-3M, PfastreXML is 5.6 percentage points better than DeepXML as well as DeepXML-RE for propensity-scored metrics but incurs a loss of 10 percentage points for vanilla precision, which is clearly unacceptable for real world tasks where both metrics are important.

Please refer to Section A.3 in the appendix for more discussion.

Table 6 includes example documents which demonstrate that DeepXML can make accurate and diverse predictions.

This section discuss impact of feature representations (DeepXML-SW, DeepXML-f) and classifiers (DeepXML-fr, DeepXML-ANNS, DeepXML-P, DeepXML-NS).

The detailed architectures are included in section A.6 in the supplementary section.

Sub-word features: DeepXML can also exploit sub-word features (Joulin et al., 2017) for an additional 1% gain in precision as demonstrated in Table 3 .

Label split and classifier: DeepXML-fr refers to a joint DeepXML architecture, i.e. without splitting labels, trained with a fully connected layer.

As demonstrated in Table 1 , the proposed algorithm is more scalable and accurate on tail labels w.r.t DeepXML-fr.

DeepXML could be up to 1-3% more accurate then DeepXML-NS & DeepXML-ANNS where the classifier is trained using negative sampling.

Additionally, DeepXML was found to be 1.3% more accurate than DeepXML-P which uses a tree based classifier.

Pre-trained features: DeepXML could be upto 10% more accurate than pre-trained representations with Slice classifier (refer to Tables 1 & 3) such as FastText, BERT (Devlin et al., 2018) and SIF (Arora, 2017).

This paper developed DeepXML, an algorithm to jointly learn representations for extreme multilabel learning on text data.

The proposed algorithm addresses the key issues of scalability and low accuracy (especially on tail labels and very short documents) with existing approaches such as Slice, AttentionXML, and XML-CNN, and hence improves on them substantively.

Experiments revealed that DeepXML-RE can lead to a 1.0-4.3 percentage point gain in performance while being 33-42× faster at training than AttentionXML.

Furthermore, DeepXML was upto 15 percentage points more accurate than leading techniques for matching search engine queries to advertiser bid phrases.

We note that DeepXML's gains are predominantly seen to be on predicting tail labels (for which very few direct word associations are available at train time) and on short documents (for which very few words are available at test time).

This indicates that the method is doing especially well, compared to earlier approaches, at learning word representations which allow for richer and denser associations between words -which allow for the words to be well-clustered in a meaningful semantic space, and hence useful and generalisable information about document labels extracted even when the number of direct word co-occurrences observed is very limited.

In the future we would like to better understand the nature of these representations, and explore their utility for other linguistic tasks.

Table 5 lists the parameter settings for different data sets.

Experiments were performed with a random-seed of 22 on a P40 GPU card with CUDA 10, CuDNN 7.4, and Pytorch 1.2 (Paszke et al., 2017) .

Figure 4: Precision@5 in k(%) most frequent labels Table 5 : Parameter setting for DeepXML on different datasets.

Dropout with probability 0.5 was used for all datasets.

Learning rate is decayed by Decay factor after interval of Decay steps.

For HNSW, values of construction parameter M = 100, ef C = 300 and query parameter, ef S = 300.

Denoted by '|', DeepXML-h and DeepXML-t might take different values for some parameters.

Note that DeepXML-t uses a shortlist of size 500 during training.

However, a shortlist of size 300 queried from ANNS is used at prediction time for both DeepXML-h and DeepXML-t.

A

Label set L is divided into two disjoint sets, i.e. L h and L t based on the frequency of the labels.

Labels with a frequency more than splitting threshold γ are kept in set L h and others in L t .

The splitting threshold γ is chosen while ensuring that most of the features (or words) are covered in documents that one at least one instances of label in the set L h and |L h | < 0.2M .

Two components for DeepXML, DeepXML-h and DeepXML-t, are trained on L h and L t .

Please note that other strategies like clustering of labels, connected components of labels in a graph were also tried, but the above-mentioned strategy provides good results without any additional overhead.

More sophisticated algorithms for splitting such as label clustering, may yield better results, however at the cost of increased training time.

DeepXML, DeepXML-RE yields 3 − 4% better accuracy on propensity scored metrics and can be upto 2% more accurate on vanilla metrics.

Note that PfastreXML outperform DeepXML and DeepXML-RE on AmazonTitles-3M in propensity scored metrics, however suffers a substantial loss of 10% on vanilla precision and nDCG which is unacceptable for real world applications.

Performance has been evaluated using propensity scored precision@k and nDCG@k, which are unbiased and more suitable metric in the extreme multi-labels setting Babbar & Schölkopf, 2019; Prabhu et al., 2018a; .

The propensity model and values available on The Extreme Classification Repository (Bhatia et al., 2016) were used.

Performance has also been evaluated using vanilla precision@k and nDCG@k (with k = 1, 3 and 5) for extreme classification.

For a predicted score vectorŷ ∈ R L and ground truth vector y ∈ {0, 1} L :

y l log(l + 1)

Here, p l is propensity score of the label l proposed in .

Representations: Deep architectures such as CNN (Liu et al., 2017) , MLP , and LSTM with Attention (You et al., 2018) have been applied to learn semantically and syntactically rich features.

Barring AttentionXML (You et al., 2018) , these methods suffer from low accuracy, which is more prominent on tail labels indicating inept document representation for tail labels.

However, performance of AttentionXML also degrades for short text documents as discussed in section 4.3.

Parabel: Parabel learns a hierarchy over labels to select hardest negatives for each labels and hence bring down the training cost to O (N d log L) .

However, Parabel is designed specifically for sparse BoW features and its performance degrades on low dimensional features.

Slice: Negative sampling is a popular approach to reduce training complexity in extreme classification setting.

Several strategies have been proposed in literature to select negatives labels for each instance Mikolov et al. (2013) ; Reddi et al. (2018) or negative examples for each label (Yen et al., 2017; Jain et al., 2019; Prabhu et al., 2018b) .

Slice approach has been shown to scale to 100 million labels and more accurate than alternate approaches (Yen et al., 2017; Prabhu et al., 2018b) Experiments were carried out with several variations of DeepXML where labels were not splitted in head and tail labels.

Here are the different configurations: DeepXML-f: This variation refers to a word embedding layer, ReLU non-linearity, Dropout and a fully connected output layer.

DeepXML-fr: This variation refers to a word embedding layer, ReLU non-linearity, Dropout, a residual block and a fully connected output layer.

DeepXML-NS: This variation refers to a word embedding layer, ReLU non-linearity, Dropout, a residual block and a classifier trained via negative sampling (Mikolov et al., 2013) .

DeepXML-SW: DeepXML can exploit sub-words features as proposed in (Joulin et al., 2017) .

This variations refers to DeepXML with sub-word information.

Here, character tri-grams were added to the vocabulary in addition to unigrams.

DeepXML-ANNS: This variation refers to a word embedding layer, ReLU non-linearity, Dropout, a residual block and a classifier trained via Slice.

This version trains only with hardest negatives labels selected via ANNS (Jain et al., 2019) .

However, the hardest negatives keeps changing as the document representations are being updated.

Hence, it requires ANNS graph to be trained multiple time.

Here, ANNS graph was updated after an interval of 5 epochs.

DeepXML-P: This variation refers to a word embedding layer, ReLU non-linearity, Dropout, a residual block and a shallow tree based classifier proposed in AttentionXML.

@highlight

Scalable and accurate deep multi label learning with millions of labels.