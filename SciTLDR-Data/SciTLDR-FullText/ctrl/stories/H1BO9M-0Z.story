Learning high-quality word embeddings is of significant importance in achieving better performance in many down-stream learning tasks.

On one hand, traditional word embeddings are trained on a large scale corpus for general-purpose tasks, which are often sub-optimal for many domain-specific tasks.

On the other hand, many domain-specific tasks do not have a large enough domain corpus to obtain high-quality embeddings.

We observe that domains are not isolated and a small domain corpus can leverage the learned knowledge from many past domains to augment that corpus in order to generate high-quality embeddings.

In this paper, we formulate the learning of word embeddings as a lifelong learning process.

Given knowledge learned from many previous domains and a small new domain corpus, the proposed method can effectively generate new domain embeddings by leveraging a simple but effective algorithm and a meta-learner, where the meta-learner is able to provide word context similarity information at the domain-level.

Experimental results demonstrate that the proposed method can effectively learn new domain embeddings from a small corpus and past domain knowledges\footnote{We will release the code after final revisions.}. We also demonstrate that general-purpose embeddings trained from a large scale corpus are sub-optimal in domain-specific tasks.

Learning word embeddings BID18 ; BID29 ; BID14 BID17 c) ; BID22 ) has received a significant amount of attention due to its high performance on many down-stream learning tasks.

Word embeddings have been shown effective in NLP tasks such as named entity recognition BID26 ), sentiment analysis BID13 ) and syntactic parsing BID6 ).

Such embeddings are shown to effectively capture syntactic and semantic level information associated with a given word BID14 ).The "secret sauce" of training word embedding is to turn a large scale in-domain corpus into billions of training examples.

There are two common assumptions for training word embeddings: 1) the training corpus is largely available and bigger than the training data of the potential downstream learning tasks; and 2) the topic of the training corpus is closely related to the topic of the down-stream learning tasks.

However, real-world learning tasks often do not meet one of these assumptions.

For example, a domain-specific corpus that is closely related to a down-stream learning task may often be of limited size.

If we lump different domain corpora together and train general-purpose embeddings over a large scale corpus (e.g., GloVe embeddings BID22 ) are trained from the corpus Common Crawl, which covers almost any topic on the web), the performance of such embeddings on many domain-specific tasks is sub-optimal (we show this in Section 6).

A possible explanation is that although many domain words share similar meanings with the same out-of-domain words, with no in-domain awareness, dumping many out-of-domain co-occurrences as training examples may bias in-domain embeddings. (e.g., if the domain is about food, then an out-of-domain "python" as a programming language can bias "java", while the indomain word "chocolate" is more likely to help).To solve the problem of the limited domain corpus, one possible solution is to use transfer learning BID20 ) for training domain-specific embeddings BID2 ; BID31 ).

However, these methods just manage to leverage out-of-domain embeddings trained from a large scale corpus to help limited in-domain corpus.

The very in-domain corpus is never expanded.

Also, one common assumption of these works is that a pair of similar source domain and target domain is manually identified in advance.

In reality, given many domains, manually catching useful information in so many domains are very hard.

In contrast, we humans learn the meaning of a word more smartly.

We accumulate different domain contexts for the same word.

When a new learning task comes, we may quickly identify the new domain contexts and borrow the word meanings from existing domain contexts.

This is where lifelong learning comes to the rescue.

Lifelong machine learning (LML) is a continual learning paradigm that retains the knowledge learned in past tasks 1, . . .

, n, and uses it to help learning the new task n + 1 BID28 ; BID27 ; BID4 ).

In the setting of word embedding: we assume that the learning system has seen n domain corpora: (D 1 , . . .

, D n ), when a new domain corpus D n+1 comes by demands from that domain's potential down-stream learning tasks, the learning system can automatically generate word embeddings for the n + 1-th domain by effectively leveraging useful past domain knowledge.

The main challenges of this task are 2 fold.

1) How to identify useful past domain knowledge to train the embeddings for the new domain.

2) How to automatically identify such kind of information, without help from human beings.

To tackle these challenges, the system has to learn how to identify similar words in other domains for a given word in a new domain.

This, in general, belongs to metalearning BID30 ; BID21 ).

Here we do not focus on specific embedding learning but focus on learning how to characterize corpora of different domains for embedding purpose.

The main contributions of this paper can be summarized as follows: 1) we propose the problem of lifelong word embedding, which may benefit many down-stream learning tasks.

We are not aware of any existing work on word embedding using lifelong learning 2) we propose a lifelong embedding learning method, which leverages meta-learning to aggregate useful knowledge from past domain corpora to generate embeddings for the new domain.

Learning word embeddings has been studied for a long time BID18 ).

Many earlier methods employ complex neural network architectures BID5 ; BID16 ).

Recently, a simple and effective unsupervised model called skip-gram BID15 c) ) was proposed to turn plain text corpus into large-scale training examples without any human annotation.

It uses the current word to predict the surrounding words in a context window by maximizing the likelihood of the predictions.

The learned parameters for each word are then the embeddings of that word.

Although such embeddings can be trained in large scale and easily obtained online BID22 ; BID1 ), they are sub-optimal for many domain-specific tasks BID2 ; BID31 ).

Domain corpus also suffers from limited size to train high-quality embeddings.

Our work is most related to Lifelong Machine Learning (LML)) (or lifelong learning).

Much of the work on LML focused on supervised learning BID28 ; BID27 ; BID24 ; BID4 ) Recent years, several works have also been done in the unsupervised setting, mainly on topic modeling BID3 ), information extraction BID17 ) and graph labeling BID25 ).

However, we are not aware of any existing research that has been done on using lifelong learning for word embedding.

LML is related to transfer learning and multi-task learning BID20 ), which have been leveraged in word embeddings BID2 ; BID31 ).

However, LML is different from transfer learning (see the survey book from BID4 ).

Given many domains with uncertain relevance for the new domain, the lack of guidance on which kind of information is worth learning from the past domains is a problem.

And there's no good measure of similarity of two words in different domains.

The proposed method leverages meta-learning BID30 ), which is to perform machine learning on learning tasks.

Recently, meta-learning (or learning to learn) has been used to learn parameters of an optimizer BID0 ), to learn neural architectures BID8 ).

We leverage a meta-learner to accumulate knowledge during lifelong learning.

The overall lifelong learning process is depicted in FIG0 .

Given a series of domain corpora DISPLAYFORM0 . .

, d n , the lifelong learning system first learns a meta-model (learner) on domainlevel word context similarity from the first m domain corpus.

As more domains arrive, the system accumulates the knowledge of domain corpora.

When a new domain D n+1 comes, the system uses the meta-learner to catch past domain knowledge that is useful and related to the new domain D n+1 as augmented knowledge.

With the augmented knowledge, the word embedding learning process is performed and the resulting embeddings are used for further down-stream learning tasks.

The metalearner here plays a central role in automatically identifying useful knowledge from past domains to help the new domain.

By using a pairwise network, the meta-learner finds words in the past domains that are similar to the new domain.

Then the co-occurrence knowledge of those similar words from the past domain is used together with the new domain corpus to train the new domain embeddings.

In this subsection, we describe how a meta-learner can help to identify similar words from many past domain corpora.

When it comes to borrowing knowledge from past domains, the first problem is what to borrow.

Although binary cross-domain embeddings are studied in BID2 ; BID31 , they mostly assume that a relevant domain is already identified and shared words between two domains have similar meanings.

In reality, given a wide spectrum of domains, borrowing knowledge from a non-relevant domain may not be helpful or even harmful to word embeddings (we show this Section 6).

The meaning of one word in one domain may be quite different from the same word in another.

For example, the word "java" in the programming context is different from the restaurant context.

Borrowing the knowledge from a restaurant corpus can be harmful to the representations of "java" in a programming context.

On top of learning embeddings for specific domains, we build a meta-learner to learn a general word context similarity from the first m domains, where m n. In practice, if n is small, m domains can simply be sampled from n domains.

Here since our experiments are conducted on hundreds of domains, we hold-out m domains to train the meta-learner.

The expected input to the meta-learner is a pair of the same word from similar ("java" from two corpora of the restaurant domain) or different domains (e.g. "java" from the restaurant domain or the programming domain).

The output of the meta-learner is whether two words are from the same domain or not.

We first characterize words in domain corpora.

Given a specific word in a domain, we choose its co-occurrence counts with f frequent words within a context window (like word2vec) as the discrete features (a sparse vector of length f ) of a word in that domain.

This is inspired by the fact that a good dictionary (e.g. Longman dictionary) uses only a few thousand words to explain all other words.

We denote the selected the top f frequent words over m domains as V wf .

Then given a domain corpus D i , we sample l subcorpora D i,j ∼ P (D i ) by selecting a fixed amount of chunks in D i .

A chunk can be a sentence or a document in D i .

We randomly select a fixed amount of chunks because the word features built from the sub-corpus are expected to be on the same scale.

Then we randomly select a subset of words from top f words as training example words V meta .

These training example words are the same in all domains D 1:m .

We use these words in V meta as co-occurrence features and build features u w i,j,k for the word w k ∈ V meta on the j-th sub-corpus of the i-th domain.

We build word features for all m domain sub-corpora D 1:m,1:l .Finally, a pairwise meta-learner is trained on pairs of word features drawn from different domain sub-corpora for the same word.

Given a word w k ∈ V meta , a pair of word features (u w i,j,k , u w i,j ,k ), where j = j , forms a postive example; whereas (u w i,j,k , u w i ,j ,k ) with i = i (j and j can be equal or not) forms a negative example.

The m domains are split into disjoint m t training domains, validation domains, and testing domains.

So both the validation and testing examples are unseen examples during training.

We enforce such isolation and wish the trained meta-learner can be more generally applied to the rest n − m domains.

We train a simple but efficient neural network to learn pairwise domain-level word context similarity.

The idea of making such a network small but high-throughput is crucial in lifelong settings.

This is because the meta-learner is heavily used in the later lifelong learning process.

Given so many domains with so many words asking for detecting similarity, a small pairwise network with fewer parameters is desirable to save more memory being used for high-throughput inference.

The proposed pairwise network contains only one shared fully-connected layer (normalized by the co-occurrence feature) to learn continuous features from co-occurrence (discrete) features, a cosine function to learn similarity and a sigmoid layer to generate predictions like linear regression.

The network is parameterized as follows: DISPLAYFORM0 where | · | 1 is the l1-norm, W s and b are weights and σ(·) is the sigmoid function.

Cosine similarity is defined as Cosine(x, y) = x·y ||x||2·||y||2 .

Most trainable weights of this simple network reside in W 1 , which learn continuous features on the f words.

These weights can also be interpreted as an embedding matrix for the f words.

These f word embeddings serve as general word embeddings to explain domain-specific words.

We train the meta-learner over a hold-out domain set as the base meta-learner M .

Then we fine-tune the meta-learner based on new domain corpus, as described in the next section.

The previous section ends up with a base meta-learner M .

In this section, we describe how the lifelong learning system works based on M , n − m domains, and the new domain corpus D n+1 .

Assume the lifelong learning system has seen n domain corpora.

The system stores knowledge into a knowledge base K. The knowledge base K contains a base meta-learner M trained over the first m domains, fine-tuned meta-learners M m+1:n and knowledge over past n − m domains.

The knowledge includes the vocabulary of word features V wf , n − m domain corpora D m:n , vocabularies on n − m domains V m:n , and word features on those vocabularies E m:n .

The word features E m:n are computed from one sample from each domain corpus.

Given a new domain corpus D n+1 , the lifelong learning system first fine-tunes the base meta-learner M .

This ends with a fine-tuned meta-learner M n+1 for this new domain.

The tuning process makes the meta-learner more suitable for the new domain to retrieve past knowledge.

Tuning examples are sampled similarly as the training examples of base meta-learner, except that negative examples are sampled between D n+1 and D m+1:n .

Then the lifelong learning system retrieves similar in-domain word context information as augmented knowledge, which is used in embedding training in the next subsection.

The retrieval process is described in Algorithm 1.

Firstly, line 1-2 build word features for the new domain corpus.

These two operations are already done when preparing fine-tuning data for the meta-learner.

Here we just mention them for storing knowledge purpose in line 13.

Line 3 retrieves past domain knowledge, which is the reversed process similar to line 13 for the new domain.

Line 4 defines the variable that stores useful past knowledge.

Line 5-12 retrieves relevant words from past domains and store them in A. More importantly, the fine-tuned meta-learner at line 9 finds similar words from past domains.

Then we only keep similar words with a probability higher than a threshold delta at line 10.

This threshold controls the quality of the accumulated words O. Algorithm 1: Lifelong domain-level word context retrieval Input : a knowledge base K containing knowledge over past (n − m) domains, a new domain corpus D n+1 , and a fine-tuned meta-learner M n+1 .

Output: a word co-occurrence set A, where each element is a 2-tuple (w t , w c ), representing useful knowledge from past domains.

DISPLAYFORM0

In this subsection, we first describe the skip-gram model introduced by BID15 in the context of a new domain in the lifelong setting.

Given a new domain corpus D n+1 with a vocabulary V n+1 , the goal of the skip-gram model is to learn a vector representation for each word w ∈ V n+1 in that domain.

Assume the domain corpus is represented as a sequence of words D n+1 = (w 1 , . . . , w T ), the objective of the skip-gram model is to maximize the following log-likelihood: DISPLAYFORM0 where C t is the set of indices of words surrounding word w t in a fixed context window; N t is a set of indices of words (negative samples) drawn from the vocabulary V n+1 for the t-th word; u and v represent word vectors (or embeddings) we are trying to learn.

The goal of skip-gram is to independently predict the presence (or absence) of context words w c given the word w t .

When the size T of the corpus is extremely large, the skip-gram model, in fact, can be fed with billions of training examples.

So the vector of a word can be trained to have a good representation of the similarity with the word's context words.

However, depending on the specific down-stream tasks, many domain corpora may not have a large scale corpus.

And a random sequence of words drawn from other domains may not truly reflect the distribution P (w c |w t ) in domain D n+1 .

This is where the previously computed augmented word co-occurrence A come to rescue.

Assume our lifelong learning system has seen m domains to build the meta-learner M and n − m domains to build the knowledge into K. Given a new domain corpus D n+1 , we first perform Algorithm 1 to obtain the augmented word co-occurrence from past domains A. Then this co-occurence information A is integrated into the objective function of skip-gram as following: DISPLAYFORM1 where w c is a random word drawn from the vocabulary.

We use the default hyperparameters of skip-gram model BID15 ).

Note that in the skip-gram model as we scan through the corpus w t can also be w c 's context word.

But in the augmented information here, we do not allow such bi-directional co-occurrence happen since w t may not be a useful context word for the word w c in the (n + 1)-th domain.

We present extensive evaluations to assess the effectiveness of our approach.

Following the suggestions of BID19 ; BID7 , we leverage the learned word embeddings as continuous features in several domain-specific down-stream tasks, including document classification, aspect extraction, and sentiment classification.

We do not evaluate the learned embeddings directly as in traditional word embedding papers BID15 ; BID22 ) because domain-specific dictionaries of similar / non-similar words are in general not available.

We use the Amazon Review datasets BID9 as a huge collection of multiple-domain corpus.

We consider each second-level category (the first level is department) as a domain and aggregate all reviews under each category as one domain corpus.

This ends up with a rather diverse domain collection.

Due to limited computing resources, we limit each domain corpus up to 60 MB.

We randomly select 3 domains ("Computer Components", "Cats Supply" and "Kitchen Storage and Organization") as new domains for down-stream tasks on product type classification and sentiment classification.

Then we deliberately pick the "Laptops" domain as the new domain for aspect extraction task since the annotation is on Laptop reviews.

Each new domain corpus is cut to 10 MB and 30 MB in order to test the practical performance of a small new domain.

We randomly select 56 (m) domains to train and evaluate the meta-learner.

Lastly, three random collections of 50, 100 and 200 (n − m) domains corpora are used as past domains.

We split the 56 domains as 39 (m t ) domains for training, 5 domains for validation and 12 domains for testing.

So the validation and testing domain corpora have no overlapping with the training domain corpora.

This leads to a more general base meta-learner for many unseen new domains.

We sample 2 (l) sub-corpora over the set of reviews from each domain and limit the size of the sub-corpora to 10 MB.

We select top 5000 words as word features (f ).

We randomly select 500 words (|V meta | = 500) from each domain and ignore words with zero counts on co-occurrence to obtain pairwise examples.

This ends up with 80484 training examples, 6234 validation examples, and 20740 testing examples.

The f1-score of meta-learner is 81%.We further fine-tune the meta-learner for each new domain.

We sample 3000 words from each new domain, which ends with slightly fewer than 6000 samples after ignoring zero co-occurrences.

We select 3500 examples for training, 500 examples for validation and 2000 examples for testing.

The testing f1-score is shown in TAB0 .

Finally, we empirically set delta = 0.7 as the threshold.

We use 3 down-stream tasks to evaluate the effectiveness of our approach.

For each task, we leverage an embedding layer to store the pre-trained embeddings.

We choose our embedding dimensions as 300, which is the same size as many pre-trained embeddings (GloVec.800B BID22 ) or fastText Wiki English (Bojanowski et al. FORMULA1 ).

We freeze the embedding layers during training, so the result is less affected by the rest of the model and the training data.

To make the performance of all tasks relatively consistent, we leverage the same Bi-LSTM model BID10 ) on top of the embedding layer to learn task-specific features from different embeddings.

The input size of Bi-LSTM is the same as the embedding layer and the output size is 128.

All tasks leverage many-to-one Bi-LSTMs for classification purpose except aspect extraction, which uses many-to-many Bi-LSTM for sequence labeling.

In the end, a fully-connected layer and a softmax activation are applied after Bi-LSTM, with the output size specific to each task.

No Embedding (NE): We randomly initialize the word vectors and train the word embedding layer during the training process of each down-stream task.

Note that only in this baseline do we allow embeddings trainable.fastText: This is the lower-cased embeddings pre-trained from English Wikipedia using fastText BID1 ).

We lower the cases of all corpora of down-stream tasks to match the words in this embedding.

Note that although the corpus of Wikipedia contains a wide spectrum of domains covering almost everything of human knowledge, the amount of corpus for a specific domain (e.g, a product) may not be large enough.

The total amount of Wikipedia is just several billions of tokens, which is on the same scale as Amazon Review datasets (8 billion tokens).GoogleNews: This is the pre-trained embeddings using word2vec 2 based on part of the Google News datasets, which contains 100 billion words.

GloVe.

Twitter.27B: This embedding is pre-trained using GloVe BID22 ) based on Tweets, which have 27 billion words.

Note this embedding is lower-cased and has 200 dimensions.

GloVe.6B: This is the lower-cased embeddings pre-trained from Wikipedia and Gigaword 5, which has 6 billions of tokens.

GloVe.840B: This is the cased embeddings pre-trained from Common Crawl, which has 840 billions of tokens.

This embedding corpus is the largest one among all embeddings.

It contains almost all web pages available before 2015.

We show that although GloVe.840B is general enough on almost any task, its performance is sub-optimal on many domain tasks.

This task is to classify a review into a product type (leaf-level category in Amazon product category system).

There are many product types under each domain (2nd-level category).

We use the randomly selected 3 domains as the new domains to form 3 multi-class classification sub-tasks.

These domains are: Computer Components, Kitchen Storage and Organization and Cats Supplies.

For each sub-task, we randomly draw 1200 reviews for each product type.

We drop classes with less than 1200 reviews.

This ends up with 13, 17 and 11 classes for Computer Components, Kitchen Storage and Organization and Cats Supplies, respectively.

For each sub-task, we keep 10000 reviews as the testing data (to make the result more accurate) and split the rest as 7:1 for training and validation data, respectively.

All sub-tasks are evaluated on accuracy.

We train and evaluate each sub-task on each baseline 10 times (with different initialization) and average the results.

From BID11 BID12 ).

We use the dataset from SemEval-2014 Task 4: Aspect-based sentiment analysis BID23 ) as a downstream new domain task.

This dataset contains human annotated Laptop aspects and their polarities.

It has 3045 training examples and 800 testing examples.

We use the Laptop domain corpus from the Amazon Review Dataset as the new domain corpus to train the lifelong embedding.

We leverage the original evaluation script to report precision, recall, and F1-score.

Again, we average 10 runs of the results.

From TAB3 , we can see that aspect extraction is quite different from product type classification.

Again, our lifelong embedding performs best.

Surprisingly, the performance of 200D + ND 30M is very good.

This indicates aspect extraction requires both good general embedding and domainspecific embeddings.

For example, good representations of general words can help to identify nearby aspects and good aspects words can also help.

We select 6000 4-rating reviews as positive reviews and 6000 2-rating reviews as negative reviews from 3 domains used in product type classification to form 3 sentiment classification sub-tasks.

Again, to ensure enough number of valid digits in the results, we use 10000 out of 12000 reviews for testing.

The results are averaged over 10 runs.

From TAB4 , we can see the performance of most domain-specific baselines is very close (We omit the minor differences between different sizes of past domains).

Sentiment classification, in general, requires polarity words to determine the sentiment of a document.

Pre-trained general embeddings may introduce non-polarity information into the embeddings.

When domain corpus is leveraged, the difference is small.

This is close to our previous experience.

A possible explanation is that sentiment classification relies on sentiment words like "good" or "bad".

However, those words have similar context words, e.g., "This phone is good." and "This phone is bad.".

So the co-occurrence-based training corpus of embedding is not good for learning the embeddings of sentiment words.

Although most cross-domain embedding papers focus on leveraging different existing pre-trained embeddings, we focus on expanding the domain-specific corpus.

We believe if we can expand the domain-specific training corpus on a much larger scale (like breaking the training corpus of GloVe.860B into many domains), the performance of the proposed method is much better.

However, our focus does not forbid our method from leveraging existing cross-domain transfer learning method (?

BID2 ; BID31 ).

A simple way of leveraging existing embeddings in these papers is to concatenate existing pre-trained embeddings with domain-specific embeddings.

To demonstrate our method further improves the domain-specific parts of the downstream tasks, we further evaluate two methods: (1) GloVe.840B&ND 30M, which concatenates new domain alone embeddings with GloVe.860B; (2) GloVe.840B&LL 200D + ND 30M, which concatenates our lifelong embeddings with GloVe.860B.As shown in TAB5 , concatenating embeddings improve the performance a lot.

Our method further improves the domain-specific parts of the embeddings.

While existing cross-domain embedding methods can only use the 30 MB corpus of the new domain, our method allows those methods to further leverage the expanded corpus.

In this paper, we formulate a lifelong word embedding learning process.

Given many previous domains and a small new domain corpus, the proposed method can effectively generate new domain embeddings by leveraging a simple but effective algorithm and a meta-learner.

The meta-learner is able to provide word context similarity information on domain-level.

Such information can help to accumulate new domain-specific training corpus in order to get better embedding.

Experimental results show that the proposed method is effective in learning new domain embeddings from a small corpus and past domain knowledge.

<|TLDR|>

@highlight

learning better domain embeddings via lifelong learning and meta-learning

@highlight

Presents a lifelong learning method for learning word embeddings.

@highlight

This paper proposes an approach to learn embeddings in new domains and significantly beats the baseline on an aspect extraction task. 