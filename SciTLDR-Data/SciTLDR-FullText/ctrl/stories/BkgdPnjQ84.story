We address the problem of open-set authorship verification, a classification task that consists of attributing texts of unknown authorship to a given author when the unknown documents in the test set are excluded from the training set.

We present an end-to-end model-building process that is universally applicable to a wide variety of corpora with little to no modification or fine-tuning.

It relies on transfer learning of a deep language model and uses a generative adversarial network and a number of text augmentation techniques to improve the model's generalization ability.

The language model encodes documents of known and unknown authorship into a domain-invariant space, aligning document pairs as input to the classifier, while keeping them separate.

The resulting embeddings are used to train to an ensemble of recurrent and quasi-recurrent neural networks.

The entire pipeline is bidirectional; forward and backward pass results are averaged.

We perform experiments on four traditional authorship verification datasets, a collection of machine learning papers mined from the web, and a large Amazon-Reviews dataset.

Experimental results surpass baseline and current state-of-the-art techniques, validating the proposed approach.

We investigate the applicability of transfer learning techniques to Authorship Verification (AV) problems, and propose a a method that uses some of the most recent advances in deep learning to achieve state of the art results on a variety of datasets.

AV seeks to determine whether two or more text documents have been written by the same author.

Some applications of AV include plagiarism analysis, sock-puppet detection, blackmailing, and email spoofing prevention BID7 .

Traditionally, studies on AV consider a closed and limited set of authors, and a closed set of documents written by such authors.

During the training step, some of these documents (sometimes as long as a novel) are used.

The goal can be formulated as to successfully identify whether the authors of a pair of documents are identical BID14 BID19 BID11 .

This type of AV tasks assumes access to the writing samples of all possible authors during the training step, which is not realistic.

Recently, the AV problem has changed to reflect realistic -and more challenging-scenarios.

The goal is no longer to individually learn the writing style of the authors (like in traditional AV methods), but to learn what differentiates two different authors within a corpus.

This task involves predicting authorship of documents that may not have been previously encountered within the training set; in fact, the presence of the authors in the training data is not guaranteed either.

That is, the test set may contain out of training sample data; given a set of authors of unknown papers contained within the training data, A unknown train , and a set of authors of unknown papers in the test data, A unknown test , it is neither unreasonable nor unexpected to find that A unknown train ???A unknown test = ???. Some other challenges arise in modern AV tasks, making authorship verification of a given pair of documents hard to infer.

One is the lack of training data, which can manifest itself in any one or more of the following: the training set may be small, samples of available writings may be limited, or the length of the given documents may be insufficient.

Another is the test and train documents belonging to different genre and/or topics, both within their respective sets as well as between the train and the test set -implying they were likely drawn from different distributions.

The challenge is to ensure robustness in a multitude of possible scenarios.

Regardless of the AV problem specifics, generally we assume a training dataset made of sets of triples: DISPLAYFORM0 with x i X known , x j X unknown a realization from random variables X known and X unknown , and the label y i,j Y is drawn from a random variable Y , producing a total of P sets of realizations, each potentially by a different author, thus forming up to P source domains, because it can be argued that a collection of literary works by one author forms a latent domain of it's own.

The goal is to learn a prediction function f : X ??? Y that can generalize well and make accurate predictions regarding documents written by authors both inside and outside of the training set, even if those documents were not seen in training.

Less formally, in AV the task is composed of multiple sub-problems: for each given sub-set of texts, we are provided one or more documents that need to be verified and one or more that are known to be of identical authorship.

We approach the AV problem by designing a straightforward deep document classification model that relies on transfer learning a deep language model, ensembles, an adversary, differential learning rates, and data augmentation.

In order to ensure the design's versatility and robustness, we perform authorship verification on a collection of datasets that have little in common in terms of size, distribution, origins, and manner they were designed.

For evaluation, we consider standard AV corpora with minimal amount of training data, PAN-2013 BID12 , PAN-2014E and PAN-2014N BID27 , PAN-2015 BID28 , a collection of scientific papers mined from the web BID2 , and Amazon Reviews dataset BID8 .

The proposed approach performs well in all scenarios with no specific modifications and minimal fine-tuning, defeating all baselines, PAN competition winners, as well as the recent Transformation Encoder and PRNN models that were recently shown to perform well on AV tasks.

BID8 .

Our method consists of three major components: augmentation, transfer learning, and the training/testing process itself.

At a high level, we augment the data, fine-tune a deep LSTM-based language model (LM) known as ULMFit BID9 on the augmented training set, train an ensemble of RNN and QRNN classifiers with the encoding produced by the LM forward and backward, and evaluate the test data while performing test-time data augmentation.

We utilize various data augmentation techniques in order to improve model generalization TAB0 .

They broadly fall into two categories, document manipulation and adversarial noise injection with the LM.

In addition, most of these techniques can be applied to the test set documents during evaluation; however, some do more harm than good when used in such manner.

Noise injection is performed by a 5-layer LSTM model that was pre-trained on Wikipedia and fine-tuned on our data.

In our setup, it acts as a generator with a 3-layer RNN classifier working as a critic.

Adversarial loss function is a weighted average of the two losses DISPLAYFORM0 where g is the LM, f is an RNN and h is the linear classifier trained on RNN's average then max pooled and flattened 2 top layers.

We use a weighted average because the nature of loss functions is very different.

To improve quality of augmentation, we devised the following approach (Algorithm 1).

Given a training set consisting of a number of problems, with each problem containing one or more documents known to be written by the same author and a single document of unknown authorship, we cycle through each problem in the training set.

If for a given problem the ground truth answer is positive, we train on all documents and try injecting noise.

If the critic can tell the fake, it means our new document is most likely too different from actual ones by this author to be of any use; we then try training some more, and inject shorter sentences and less of them.

The process continues until critic is fooled or generator diverges -an unlikely event because critic is not hard to fool.

We hypothesize that documents form latent domains of their own based on various linguistic characteristics, making it beneficial to transform the pairs of documents into a domain-invariant space.

Documents forming latent domains means that authorship verification is a separate but similar task for each domain.

We cannot exploit the similarity between tasks directly because the data distributions are different, and not accounting for that while building a model would violate basic principles of machine learning BID21 .

Domain Adaptation (DA), a subset of Transfer learning, addresses such Algorithm 1: Noise injection algorithm using language model with an adversary problems by establishing knowledge transfer from a labeled source domain to an unlabeled (or partially labeled) target domain, by exploring domain-invariant features or invariants which transfer across domains BID22 BID21 BID5 BID29 , or by embedding the data into domain-invariant subspace.

Another issue that we must address comes from the nature of the data.

As the documents come in pairs, they are not readily suitable for standard classifiers.

A naive approach of concatenation produces poor results, and various distance function schema suitable for most linear models are not very suitable for RNNs.

To address these problems we utilize a deep language model that produces an encoder capable of producing an embedding representing a pair of documents.

It also alleviates the need for data by being pre-trained on a large set of Wikipedia articles BID9 .

The domain discrepancy issue is in part mitigated too, because the resulting embedding subspace features are more invariant.

In a gist, our model ( FIG0 ) is a bi-directional pipeline of recurrent neural networks.

It is built on top of a pre-trained 5-layer LSTM model and takes it's last 3 (2 intermediate hidden ones and the final embedding output) layers as inputs by pooling them together.

We use an ensemble of sequence classifiers, one based on an RNN and the other using a QRNN BID1 , a recent addition to the RNN family that combines some properties of recurrent and convolutional networks.

Both are 3-layer models with the last 2 layers average then max pooled and passed through a ReLU non-linearity and then to logit units.

We output probabilities rather than labels.

The predictions made by RNN and QRNN are averaged.

In taking advantage of improved generalization through making the model bi-directional, we faced two challenges.

First, the pre-trained LSTM model we used is uni-directional.

Second, QRNN design used in this paper does not support bi-directional training, either.

We circumvented the issue by tokenizing and numericalizing the text data and first training in regular fashion on a normal pre-trained Wikipedia model, then loading the numericalized tokens backwards and using a model that was trained on Wikipedia backwards, as well.

At test time, we reversed each document and gave the normal ones to the forward model and backward ones the backward version, then averaged the results of two runs, effectively reaping the benefits of using a bi-directional RNN without actually doing so.

We call our design 2WD-UAV in reference to ensembling of two versions of RNN for authorship verification and because of it's ULMFit heritage.

The architecture is implemented in PyTorch with elements of fast.ai library BID10 .

PAN We use all available authorship identification datasets released by PAN 1 ( TAB2 .

Each PAN dataset consists of a training and test corpus and each corpus has a various number of distinct problems.

Each problem is composed of one to five writings by a single person (implicitly disjoint For PAN2014 and PAN2015 and explicitly disjoint for PAN2013), and one piece of writing of unknown authorship.

In the other words, we are given up to five pairs of documents where one document's authorship is known and the other one's is not.

Two documents of a pair might be from significantly different genres Table 3 .

Similarity functions.

x, y: document feature vectors, n: # of features in x and y

Chi2 kernel exp(????? i [ DISPLAYFORM0 ]) Cosine similarity xy T /(||x||||y||) DISPLAYFORM1 and topics.

The length of a document changes from a few hundred to a few thousand words.

PAN2014 includes two datasets: Essays and Novels.

The paired documents in PAN datasets are used for our experiments.

For a problem P = (S, T ), S (source) is the first document and T (target) is the second document of a PAN problem BID8 .

Amazon Reviews We use a dataset made by selecting 300 authors with at least 40 reviews to make the positive and negative candidate sets.

Then, for each author, the positive candidate set is all possible and unique combinations of the author's reviews.

A positive class consists of 4500 review pairs from this positive candidate set at random.

The negative candidate set is made of all unique and possible combinations of review pairs having different authors.

For this dataset, the negative class of equal size with the positive class was created by random selection from the negative candidate set.

In prior work, 5-fold cross validation was used for this data.

We do the same in order for our results to be comparable.

BID8 .

MLPA* This schema was created using MPLA-400 dataset that contains 20 articles by each of the top-20 authors by citation in Machine Learning BID2 .

In MLPA*, only publications from MPLA-400 that are written by a single author and have no co-authors are used BID8 .

To keep the distribution of authors and classes balanced, MPLA* contains an equal number of single-authorship articles from all existing 20 authors.

The positive class consists of the pairs which are made up of all possible combinations of same-authorship articles (20 ?? 9 2 = 720).

The negative class includes the pairs that are randomly selected from the set of all unique combinations of articles of different authorship and is of the same size as the positive class.

Like Amazon Reviews, MLPA* dataset authors recommend using 5-fold cross validation BID8 .

We compare our method with the top methods of PAN AV competition between 2013 and 2015 ( TAB2 ).

The results of each method for one year of the competition are available and we report them here.

Our comparisons are not impacted by different parameter settings and implementation details of these methods as long as we keep the test and training sets the same as theirs.

We choose several classifiers widely used in the area with the seven similarity measures to set strong baselines (Table 3) .

Since each example in our underlying dataset structure comprises two documents, we need to adapt it to the structure of an ordinary classifier input by converting them to one single entity.

A simple direct way is to concatenate their feature vectors.

However, our experiments show it provides weak results mostly equal to the random label assignment.

So, we define the summary vector as a single unit representative of each example/problem P = (D S , D T ) by utilizing several similarity measures.

The summary vector comprises a class of several metrics each measuring one aspect of the closeness of the two documents (D S and D T ) of the pair for all underlying feature sets.

For any two feature vector documents x, y their summary vector is sum(x, y) = [sim j i (x, y)] where sim j i (x, y) 1???i???M,1???j???F computes the ith similarity metric of M metrics in Table 3 under jth of F = 7 feature sets (Section 3.2) between x, y. Then, we use a suite of classifiers including SVM, Gaussian Naive Bayes (GNB), K-Nearest Neighbor (KNN), Logistic Regression (LR), Decision Tree (DT) and MultiLayer Perception (MLP) to predict the class label.

All baselines are implemented by the scikit-learn library BID23 .

2WD-UAV For our model, a number of important parameters are set.

Most importantly, to achieve our results, we make use of recent work on alternating learning rates, as well as one-cycle learning policy BID25 BID26 .

The basic approach to training is as follows:-Contract learning rate lr for one cycle -Freeze it and save -Give the learning rate on next layer a very large value -Freeze it and save unfreeze the previous one -Assign a very small value to the next layer -Continue cycling until gradients explode -Return the last saved checkpoint -this is the global minimumWe also use a range of momentum across layers, as well as different learning rates for each.

For the optimizer we choose AdamW BID18 , an improved version of Adam BID13 with better weight decay regularization.

We begin with weight decay of 0.03 and regularize by adjusting it as training progresses.

2 .

Gaussian distribution is chosen for Naive Bayes.

For K-Nearest Neighbor we set K=3.

The L-2 regularization is used for Logistic Regression.

For document expansion, we set the size of the sliding window to l = 10.

On average it expands one document into 30 smaller documents for PAN datasets.

All other parameters are selected based on pilot experiments.

We report accuracy, the Area Under Receiver Operating Characteristic (ROC) curve BID3 (AUC).

The higher AUC and Score indicate more effective classification.

We compare our proposed model 2WD-UAV with several relevant baselines.

Table 4 evaluates our model with PAN datasets for different years and also the best performing model in the relevant competition years for PAN.

Results show that 2WD-UAV consistently outperforms all baselines and all best-reported models in PAN competitions for all years in the Score metric.

The Score metric is essentially Accuracy ?? ROC thereby measuring joint performance gains as both ROC and accuracy are important.

2WD-UAV outperforms in Accuracy for all competitors in PAN14Essay and PAN13 dataset.

It is the second best in PAN15 just offset by one decimal point.

While it is not performing the best in accuracy for PAN14Novels, it yields competitive performances of accuracy and outperforms all others in ROC metric.

2WD-UAV also outperforms all other models in the ROC metric for PAN15.

For PAN14E and PAN 13, it outperforms several baselines and offers stellar performance in ROC metric, just to be second to MLP and CNG respectively.

While it is true that the proposed approach is not always the best performing on PAN data in every metric except Score, we believe one reason is due to the inherently smaller data sizes (both total words of data per author to train upon and also the total number of authors to scale up training) that make the approach a little weak.

Hence, we further explored larger datasets of Amazon Reviews BID8 and MLPA* BID8 BID2 in Table 5 which shows significant performance gains in accuracy, defeating a variety of baselines.

All in all, we do find stable and consistent performance gains with Table 5 .

Accuracy using 5-fold cross-validation on MLPA* and Amazon Reviews.

Domain Adaptation Documents forming latent domains means that authorship verification is a separate but similar task for each domain.

We cannot exploit the similarity between tasks directly because the data distributions are different, and not accounting for that while building a model would violate basic principles of machine learning BID21 .

Domain Adaptation (DA) addresses such problems by establishing knowledge transfer from a labeled source domain to an unlabeled (or partially labeled) target domain, and by exploring domain-invariant features or invariants which transfer across domains BID22 BID21 BID5 BID29 .

Authorship Verification In vast majority of the AV approaches, the writing style of a questioned author is known to us as we are given some scripts of the author and the task is to determine whether a piece of work is written by the same person.

The depth of difference between two sets of documents is measured using the unmasking technique while ignoring the negative examples BID14 .

This one-class technique achieves high accuracy for 21 considerably large books (ebook above 500K).

A simple feed forward three-layer neural network, an auto-encoder, is used for AV considering it a one-class classification problem BID19 .

They observe the behavior of the neural network for documents by different authors and build a classifier for each author.

Their idea originates from one of the first applications of auto-encoder in classification as a novelty detector BID11 .

AV is also studied for detecting sock-puppets who deliberately change their writing styles to pass the filters and provide opinion Spam.

A spy induction method is proposed to leverage the test data in training step under "out-of-training" setting BID7 where a questioned author is from a closed set of candidates while appearing unknown to the verifier.

However, in a more realistic case we have no specified writing samples of a questioned author and there is no closed candidate set of authors.

Since 2013, a surge of interest arose for this type of AV problem.

BID24 investigate whether a document is one of the outliers in a corpus by generalizing the Many-Candidate method by BID15 .

The best method of PAN 2014 for Essays dataset optimizes a decision tree.

Its method is enriched by adopting variety of features and similarity measures BID4 .

However, for the Novels dataset, the other dataset of that year, the best results are achieved by an author verifier using fuzzy C-Means clustering BID20 .

In an alternative approach, BID16 generate a set of impostor documents and apply iterative feature randomization to compute the similarity distance between pairs of documents.

One of the more interesting and powerful approaches investigates the language model of all authors using a shared recurrent layer and builds a classifier for each author BID0 .

Parallel recurrent neural network and transformation auto-encoder approaches were recently shown to produce excellent results far a variety of AV problems BID8 .

The AV problem is also studied by a non Machine Learning model comprising of a compression algorithm, a dissimilarity method and a threshold.

When evaluated on PAN datasets, this approach stands at first ranking position for the two out of four PAN datasets BID6 .

Recently, Linguistic traits of sock-puppets are deeply studied to verify the authorship of a pair of accounts in online discussion communities BID17 .

Authorship verification has always been a challenging problem.

It can be even more difficult when no writing samples of questioned author/authors is given.

In this paper, we explore the possibility of a more general approach to the problem, one that does not rely on having most of the authors within the training set.

To this end, we use transfer and adversarial learning learning, data augmentation, ensemble methods, and cutting edge developments in training deep models to produce an architecture that is to the best of our knowledge novel at least to problem setting.

Our design exhibits a high degree of robustness and stability when dealing with out-of-sample (previously unseen) authors and lack of training data and delivers state-of-the-art performance.

<|TLDR|>

@highlight

We propose and end-to-end model-building process that is universally applicable to a wide variety of authorship verification corpora and outperforms state-of-the-art with little to no modification or fine-tuning.