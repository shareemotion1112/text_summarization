Aspect extraction in online product reviews is a key task in sentiment analysis and opinion mining.

Training supervised neural networks for aspect extraction is not possible when ground truth aspect labels are not available, while the unsupervised neural topic models fail to capture the particular aspects of interest.

In this work, we propose a weakly supervised approach for training neural networks for aspect extraction in cases where only a small set of seed words, i.e., keywords that describe an aspect, are available.

Our main contributions are as follows.

First, we show that current weakly supervised networks fail to leverage the predictive power of the available seed words by comparing them to a simple bag-of-words classifier.

Second, we propose a distillation approach for aspect extraction where the seed words are considered by the bag-of-words classifier (teacher) and distilled to the parameters of a neural network (student).

Third, we show that regularization encourages the student to consider non-seed words for classification and, as a result, the student outperforms the teacher, which only considers the seed words.

Finally, we empirically show that our proposed distillation approach outperforms (by up to 34.4% in F1 score) previous weakly supervised approaches for aspect extraction in six domains of Amazon product reviews.

Aspect extraction is a key task in sentiment analysis, opinion mining, and summarization BID11 BID8 Pontiki et al., 2016; BID0 .

Here, we focus on aspect extraction in online product reviews, where the goal is to identify which features (e.g., price, quality, look) of a product of interest are discussed in individual segments (e.g., sentences) of the product's reviews.

Recently, rule-based or traditional supervised learning approaches for aspect extraction have been outperformed by deep neural networks BID16 BID22 , while unsupervised probabilistic topic models such as Latent Dirichlet Allocation (LDA) BID2 have been shown to produce less coherent topics than neural topic models BID9 BID5 BID17 : when a large amount of training data is available, deep neural networks learn better representations of text than previous approaches.

In this work, we consider the problem of classifying individual segments of online product reviews to predefined aspect classes when ground truth aspect labels are not available.

Indeed, both sellers and customers are interested in particular aspects (e.g., price) of a product while online product reviews do not usually come with aspect labels.

Also, big retail stores like Amazon sell millions of different products and thus it is infeasible to obtain manual aspect annotations for each product domain.

Unfortunately, fully supervised neural approaches cannot be applied under this setting, where no labels are available during training.

Moreover, the unsupervised neural topic models do not explicitly model the aspects of interest, so substantial human effort is required for mapping the learned topics to the aspects of interest.

Here, we investigate whether neural networks can be effectively trained under this challenging setting using only weak supervision in the form of a small set of seed words, i.e., descriptive keywords for each aspect.

For example, words like "price," "expensive," "cheap," and "money" are represen-tative of the "Price" aspect.

While a traditional aspect label is only associated with a single review, a small number of seed words can implicitly provide (noisy) aspect supervision for many reviews.

Training neural networks using seed words only is a challenging task.

Indeed, we show that current weakly supervised networks fail to leverage the predictive power of the seed words.

To address the shortcomings of previous approaches, we propose a more effective approach to "distill" the seed words in the neural network parameters.

First, we present necessary background for our work.2 BACKGROUND: NEURAL NETWORKS FOR ASPECT EXTRACTION Consider a segment s = (x 1 , x 2 , . . .

, x N ) composed of N words.

Our goal is to classify s to K aspects of interest {α 1 , . . .

, α K }, including the "General" aspect α GEN .

In particular, we focus on learning a fixed-size vector representation h = EMB(s) ∈ R l and using h to predict a probability distribution p = p 1 , . . .

, p K over the K aspect classes of interest: p = CLF(h).The state-of-the-art approaches for segment embedding use word embeddings: each word x j of s indexes a row of a word embedding matrix W b ∈ R V ×d to get a vector representation w xj ∈ R d , where V is the size of a predefined vocabulary and d is the dimensionality of the word embeddings.

The set of word embeddings {w x1 , ..., w x N } is then transformed to a vector h using a vector composition function such as the unweighted/weighted average of word embeddings BID20 BID1 , Recurrent Neural Networks (RNNs) BID19 BID21 , and Convolutional Neural Networks (CNNs) BID10 BID7 .

During classification (CLF), h is fed to a neural network followed by the softmax function to get p 1 , . . .

, p K .Supervised approaches use ground-truth aspect labels at the segment level to jointly learn the EMB and CLF function parameters.

However, aspect labels are not available in our case.

Unsupervised neural topic models avoid the requirement of aspect labels via autoencoding BID9 BID5 .

In their Aspect Based Autoencoder (ABAE), BID5 reconstruct an embedding h for s as a convex combination of K aspect embeddings: DISPLAYFORM0 is the k-th row of the aspect embedding matrix A ∈ R K×d .

The aspect embeddings A (as well as the EMB and CLF function parameters) are learned by minimizing the segment reconstruction error.

1 Unfortunately, unsupervised approaches like ABAE do not utilize information about the K aspects of interest and thus the probabilities p 1 , . . . , p K cannot be used directly 2 for our downstream application.

To address this issue, BID0 proposed a weakly supervised extension of ABAE.

Their model, named Multi-seed Aspect Extractor, or MATE, learns more informative aspect representations by also considering a distinct set of seed words G k = {g k1 , . . .

, g kL } for each aspect.

In particular, MATE initializes the k-th row of the aspect embedding matrix A to the weighted 3 average of the corresponding seed word embeddings: DISPLAYFORM1 As initializing the aspect embeddings to particular values does not guarantee that the aspect embeddings after training will still correspond to the aspects of interest, BID0 fix (but do not fine tune) the aspect embeddings A and the word embeddings W b throughout training.

However, as we will show next, MATE fails to effectively leverage the predictive power of seed words.

In this work, we propose a weakly supervised approach for segment-level aspect extraction that leverages seed words as a stronger signal for supervision than MATE.

Indeed, in contrast to BID0 , who use average seed word vectors only for initialization, we use the individual seed words for supervision throughout the whole training process.

Our approach adopts the paradigm of knowledge distillation BID6 , according to which a simpler network (student) is trained to imitate the predictions of a complex network (teacher).

After training, the parameters of the teacher will have been "distilled" to the parameters of the student and hopefully the student will perform comparably to the teacher for the task at hand.

Our motivation is different: we can easily represent domain knowledge in simple and interpretable models but not in more complex neural networks.

Thus, in our work, the teacher is a simple bag-of-words classifier that encodes the seed words, while the student is a more complex neural network, which is trained to distill the domain knowledge encoded by the teacher, as we describe below.

Teacher: A bag-of-words classifier using seed words.

In BID6 , the teacher is trained on a labeled dataset.

Here, we do not have training labels but rather seed words G that are predictive of the K aspects.

Incorporating G into (generalized) linear bag-of-words classifiers is straightforward: here, we initialize the weight matrix W ∈ R V ×K and bias vector b ∈ R K of a logistic regression classifier using the seed words: DISPLAYFORM0 Under this weight configuration we consider seed words in an intuitive way: if at least one seed word appears in a segment, then the teacher assigns a higher score to the corresponding aspect than to α GEN , otherwise α GEN gets the highest score among all aspects.

4 However, assigning hard binary weights to the teacher leads to ignoring the non-seed words, i.e., if a segment does not contain any seed words belonging to G k , then the probability of the k-th aspect is zero.

Of course, non-seed words can also be predictive for an aspect (especially given that we only consider a small, incomplete set of seed words).

Next, we describe the architecture of the student network and how it can be trained to also consider non-seed words for aspect extraction.

Student: An embedding-based neural network.

The student network is an embedding-based neural network: a segment is first embedded (h = EMB(s)) and then classified to the K aspects (p = CLF(h)).

For the EMB function we experiment with two choices: the unweighted average of word2vec embeddings and contextualized embeddings using BERT BID4 .

5 For the CLF function we use the softmax classifier: p S = softmax(W s h + b s ), where W s ∈ R d×K and b s ∈ R k are the softmax classifier's weight and bias parameters, respectively.

We train the student network to imitate the teacher's predictions by minimizing the cross entropy between the student's and the teacher's predictions.

Even if the teacher's predictions is the only supervision signal, the student can learn to generalize by associating aspects with non-seed words in addition to seed words.

To encourage this behavior we use L2 regularization on the student's weights, and dropout on the word embeddings.

To better understand the extent to which non-seed words can predict the aspects of interest, we experiment with completely dropping the seed words from the student's input.

In particular, while the teacher receives the original segment during training, the student receives an edited version of the segment, where seed words belonging in G have been replaced by an "UNK" id (like out-of-vocabulary words) and thus do not provide useful information for aspect classification.

To imitate the predictions of the teacher, the student has to associate aspects with non-seed words during training.

For training and evaluation, we use the OPOSUM dataset BID0 , a subset of the Amazon Product Dataset BID12 .

OPOSUM contains Amazon reviews from six domains: Laptop Bags, Keyboards, Boots, Bluetooth Headsets, Televisions, and Vacuums.

Aspect labels (for 9 aspects) are available at the segment-level 6 but only for the validation and test sets.

For dataset details, see BID0 .

For a fair comparison, we use exactly the same 30 seed words (per aspect and domain) used in MATE.In our experiments, we use exactly the same pre-processing (tokenization, stemming, and embedding) procedure as in BID0 Table 1 : Micro-averaged F1 reported for 9-class aspect extraction in Amazon product reviews.training set without using any aspect labels, and only use the seed words G via the teacher.

We follow the same evaluation procedure as in BID0 : we tune the hyperparameters on the validation set and report the micro-averaged F1 (9-class classification) in the test set averaged over 5 runs.

We compare the following models and baselines:• ABAE: The unsupervised autoencoder of BID5 , where the learned topics were mapped to the 9 aspects as a post-hoc step.• MATE-*: The weakly supervised autoencoder of BID0 .

Various configurations include the initialization of the aspect matrix A using the unweighted/weighted average of word embeddings and an extra multi-task training objective (MT).• Teacher-BOW: A bag-of-words classifier with the weight configuration of Equation 1.• Student-*: The student trained to imitate Teacher-BOW.

Our experiments include bag-ofwords (BOW) classifiers, the (unweighted) average of word2vec embeddings (W2V), and pre-trained BERT embeddings.

Table 1 reports the evaluation results for aspect extraction.

The rightmost column reports the average performance across the 6 domains.

MATE-* models outperform ABAE: using the seed words as a weak source of supervision leads to more accurate aspect predictions.

Teacher-BOW uses the seed words but performs poorly.

On the other hand, Teacher-BOW outperforms the MATE-* models: Teacher-BOW leverages the seed words effectively and in an intuitive way according to our knowledge about aspect extraction.

Student-BOW outperforms Teacher-BOW: the two models share the same architecture but regularizing the student's weights allows for non-seed words to also be considered for aspect extraction.

The benefits of our distillation approach are highlighted using embedding-based networks.

Student-W2V outperforms Teacher-BOW and Student-BOW, showing that obtaining segment representations as the average of word embeddings is more effective than bag-of-words representations for this task.

Even when the seed words are not shown to the student during training (Student-W2V-DSW), it effectively learns to use non-seed words to determine the aspect, thus leading to more accurate predictions compared to the teacher.

Student-W2V outperforms the previously best performing model (i.e., MATE-weighted-MT) by 17.5%: although both models use exactly the same seed words, our distillation approach leverages the seed words more effectively for supervision than just for initialization.

To demonstrate the simplicity and effectiveness of our approach, we do not use weights for the seed words nor a multitask objective for Student-W2V.

Therefore, a fair comparison considers MATE-unweighted, which is outperformed by Student-W2V by 34.4%.

Considering more sophisticated methods for segment embedding are promising to yield further performance improvements: Student-BERT achieves the best performance over all models.

In the future we plan to experiment with better methods for handling noisy seed words, and interactive learning approaches for learning better seed words.

@highlight

We effectively leverage a few keywords as weak supervision for training neural networks for aspect extraction.

@highlight

Discusses a variant of knowledge distillation which uses a "teacher" based on a bag-of-words classifier with seed words and a "student" which is an embedding-based neural network.