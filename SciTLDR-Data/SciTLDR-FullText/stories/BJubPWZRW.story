We present Cross-View Training (CVT), a simple but effective method for deep semi-supervised learning.

On labeled examples, the model is trained with standard cross-entropy loss.

On an unlabeled example, the model first performs inference (acting as a "teacher") to produce soft targets.

The model then learns from these soft targets (acting as a ``"student").

We deviate from prior work by adding multiple auxiliary student prediction layers to the model.

The input to each student layer is a sub-network of the full model that has a restricted view of the input  (e.g., only seeing one region of an image).

The students can learn from the teacher (the full model) because the teacher sees more of each example.

Concurrently, the students improve the quality of the representations used by the teacher as they learn to make predictions with limited data.

When combined with Virtual Adversarial Training, CVT improves upon the current state-of-the-art on semi-supervised CIFAR-10 and semi-supervised SVHN.

We also apply CVT to train models on five natural language processing tasks using hundreds of millions of sentences of unlabeled data.

On all tasks CVT substantially outperforms supervised learning alone, resulting in models that improve upon or are competitive with the current state-of-the-art.

Deep learning classifiers work best when trained on large amounts of labeled data.

However, acquiring labels can be costly, motivating the need for effective semi-supervised learning techniques that leverage unlabeled examples during training.

Many semi-supervised learning algorithms rely on some form of self-labeling.

In these approaches, the model acts as both a "teacher" that makes predictions about unlabeled examples and a "student" that is trained on the predictions.

As the teacher and the student have the same parameters, these methods require an additional mechanism for the student to benefit from the teacher's outputs.

One approach that has enjoyed recent success is adding noise to the student's input BID0 BID50 .

The loss between the teacher and the student becomes a consistency cost that penalizes the difference between the model's predictions with and without noise added to the example.

This trains the model to give consistent predictions to nearby data points, encouraging smoothness in the model's output distribution with respect to the input.

In order for the student to learn effectively from the teacher, there needs to be a sufficient difference between the two.

However, simply increasing the amount of noise can result in unrealistic data points sent to the student.

Furthermore, adding continuous noise to the input makes less sense when the input consists of discrete tokens, such in natural language processing.

We address these issues with a new method we call Cross-View Training (CVT).

Instead of only training the full model as a student, CVT adds auxiliary softmax layers to the model and also trains them as students.

The input to each student layer is a sub-network of the full model that sees a restricted view of the input example (e.g., only seeing part of an image), an idea reminiscent of cotraining BID1 .

The full model is still used as the teacher.

Unlike when using a large amount of input noise, CVT does not unrealistically alter examples during training.

However, the student layers can still learn from the teacher because the teacher has a better, unrestricted view of the input.

Meanwhile, the student layers improve the model's representations (and therefore the teacher) as they learn to make accurate predictions with a limited view of the input.

Our method can be easily combined with adding noise to the students, but works well even when no noise is added.

We propose variants of our method for Convolutional Neural Network (CNN) image classifiers, Bidirectional Long Short-Term Memory (BiLSTM) sequence taggers, and graph-based dependency parsers.

For CNNs, each auxiliary softmax layer sees a region of the input image.

For sequence taggers and dependency parsers, the auxiliary layers see the input sequence with some context removed.

For example, one auxiliary layer is trained to make predictions without seeing any tokens to the right of the current one.

We first evaluate Cross-View Training on semi-supervised CIFAR-10 and semi-supervised SVHN.

When combined with Virtual Adversarial Training BID39 , CVT improves upon the current state-of-the-art on both datasets.

We also train semi-supervised models on five tasks from natural language processing: English dependency parsing, combinatory categorical grammar supertagging, named entity recognition, text chunking, and part-of-speech tagging.

We use the 1 billion word language modeling benchmark BID3 as a source of unlabeled data.

CVT works substantially better than purely supervised training, resulting in models that improve upon or are competitive with the current state-of-the-art on every task.

We consider these results particularly important because many recently proposed semi-supervised learning methods work best on continuous inputs and have only been evaluated on vision tasks BID0 BID50 BID26 BID59 .

In contrast, CVT can handle discrete inputs such as language very effectively.

Semi-supervised learning in general has been widely studied BID2 .

Early approaches to deep semi-supervised learning pre-train neural models on unlabeled data, which has been successful for applications in computer vision BID21 BID28 and natural language processing BID7 BID46 .

More recent work incorporates generative models based on autoencoders BID22 BID47 or Generative Adversarial Networks BID55 BID51 into the training.

Self-Training.

One of the earliest approaches to semi-supervised learning is self-training BID52 BID11 .

Initially, a classifier is trained on labeled data only.

In each subsequent round of training, the classifier, acting as a "teacher," labels some of the unlabeled data and adds it to the training set.

Then, acting as a "student," it is retrained on the new training set.

The new examples added each round act as noisy "pseudo labels" BID29 that the model can learn from.

Many recent approaches train the student with soft targets from the teacher's output distribution rather than a hard label, making the procedure more akin to knowledge distillation BID16 .Consistency Training and Distributional Smoothing.

Recent works add noise to the student's input BID0 BID50 .

This trains the model to give consistent predictions to nearby data points, encouraging distributional smoothness in the model.

Inspired by the success of adversarial training BID12 , BID37 extend this idea by adversarially selecting the perturbation to the input.

Other approaches focus on improving the targets provided by the teacher by tracking an exponential moving average of its predictions BID26 or its weights BID59 .

Our method is complimentary to these previous approaches, and can be combined with them effectively.

Co-Training.

Co-Training BID1 BID41 trains two models with disjoint views of the input.

On unlabeled data, each one acts as a "teacher" for the other model.

In contrast, our approach trains a single unified model where auxiliary prediction layers see different, but not necessarily independent views of the input.

Auxiliary Prediction Layers.

Another way of leveraging unlabeled data is through the addition of auxiliary "self-supervised" losses.

These approaches train auxiliary prediction layers on tasks where performance can be measured without human-provided labels.

Previous work has jointly trained image classifiers with tasks like relative position and colorization BID9 , sequence taggers with language modeling BID48 , and reinforcement learning agents with predicting changes in the environment BID20 .

Unlike these approaches, our auxiliary losses are based on self-labeling, not labels deterministically constructed from the input.

Data Augmentation.

Data augmentation, such as random translations or crops of input images, bears some similarity to our method in that it also exposes the model to different views of input examples.

Data augmentation has become a common practice for both supervised and semi-supervised training of image classifiers BID54 .

We first provide a general description of Cross-View Training.

We then present specific constructions for auxiliary prediction layers that work well for image classification, sequence tagging, and dependency parsing.

We use D l = {(x 1 , y 1 ), (x 2 , y 2 ), ..., (x N , y N )} to represent a labeled dataset and D ul = {x 1 , x 2 , ..., x M } to represent an unlabeled dataset.

We use p ?? (y|x i ) to denote the output distribution over classes produced by a model with parameters ?? on input x i .

Our approach uses a standard cross-entropy loss over the labeled data: DISPLAYFORM0 On unlabeled data, a popular approach is to add a consistency cost encouraging distributional smoothness in the model.

First, the model produces soft targets for the current example:?? i = p ?? (y|x i ).

The model is then trained to minimize the consistency cost DISPLAYFORM1 where D is a distance function (we use KL divergence) and ?? is a perturbation to the input that can be chosen randomly or adversarially.

As is common in prior work, we hold the teacher's prediction y i fixed during training (i.e., we don't back-propagate through it) so the student learns to imitate the teacher, but not vice versa.

Our dependency parsing models use auxiliary layers analogous to the "forward" and "backward" sequence tagging ones.

Cross-View Training adds k additional prediction layers p )produced by the model.

It outputs a distribution over labels, usually with a softmax layer (an affine transformation followed by a softmax activation function) applied to this representation: DISPLAYFORM2 At test time, only the main prediction layer p ?? is used.

Each h j is chosen such that it only uses a part of each input x i ; the particular choice can depend on the task and model architecture.

We propose variants for CNN image classifiers, BiLSTM sequence taggers, and graph-based dependency parsers in sections 3.2, 3.3, and 3.4.

We add the distances between the output distributions of the teacher and auxiliary students to the consistency loss, resulting in a cross-view consistency (CVC) loss: DISPLAYFORM3 We combine the supervised and CVC losses into the total loss, L = L sup + ?? 2 L CVC , and minimize it with stochastic gradient descent.

At each step, L sup is computed over a minibatch of labeled examples and L CVC is computed over a minibatch of unlabeled examples.

?? 1 and ?? 2 are hyperparameters controlling the strength of the auxiliary prediction layers and the strength of the unsupervised loss.

For all experiments we set ?? 1 = k and ?? 2 = 1 unless indicated otherwise.

See FIG0 for an illustration of the training procedure.

Although adding noise or an adversarial perturbation to the input generally improves results, L CVC can be trained without this enhancement (i.e., setting ?? = 0).

In this case, the first term inside the expectation disappears (the student will exactly match the teacher, so the distance is zero).

In contrast, L consistency requires a nonzero ?? to make the student and teacher output different distributions.

In most neural networks, a few additional softmax layers is computationally cheap compared to the portion of the model building up representations (such as a CNN or RNN).

Therefore our method contributes little overhead to training time over consistency training.

CVT does not change inference time because the auxiliary layers are only used during training.

Our image recognition models are based on Convolutional Neural Networks, which produce a set of features H(x i ) ??? R n??n??d from an image x i .

The first two dimensions of H index into the spatial coordinates of feature vectors and d is the size of the feature vectors.

For shallower CNNs, a particular feature vector corresponds to a region of the input image.

For example, H 0,0 would be a d-dimensional vector of features extracted from the upper left corner.

For deeper CNNs, a particular feature vector would be extracted from the whole image, but still only use a "region" of the representations from an earlier layer.

The CNNs in our experiment are all in the first category.

The primary prediction layer for our CNNs take as input the mean of H over the first two dimensions, which results in a d-dimensional vector that is fed into a softmax layer: p ?? (y|x i ) = SML(global average pool(H)).We add n 2 auxiliary softmax layers to the top of the CNN.

The jth layer takes a single feature vector as input, as shown in the left of FIG1 : p j ?? (y|x i ) = SML(H j/n ,j mod n ).

We also experimented with adding auxiliary softmaxes to the outputs of earlier layers in the CNN, but found this did not improve performance.

In sequence tagging, each example ( DISPLAYFORM0 We assume an L-layer bidirectional RNN sequence tagging model, which has become standard for many sequence tagging tasks BID13 BID14 .

Each layer runs an RNN such as an LSTM BID18 in the forward direction (taking x t i as input at each step t) and the backward direction (taking x T ???t+1 i as input at each step) and concatenates the results.

A softmax layer on top of the outputs of the last BiRNN layer, DISPLAYFORM1 The auxiliary softmax layers take DISPLAYFORM2 , the outputs of the forward and backward RNNs in the first BiRNN layer, as inputs.

We add the following four softmax layers to the model (see the right of FIG1 ): DISPLAYFORM3 The "forward" and "backward" prediction layers use the RNN's current output to predict the current token.

The "future" and "past" layers use the RNN's previous output (or, equivalently, they predict the label for the next token).

The forward layer makes each prediction without seeing the right context of the current token.

The future layer makes each prediction without the right context or the current token itself.

Therefore it works like a neural language model that, instead of predicting which token comes next in the sequence, predicts which class of token comes next in the sequence.

In a dependency parse, words in a sentence are treated as nodes in a graph.

Typed directed edges connect the words, forming a tree structure describing the syntactic structure of the sentence.

In particular, each word x To give a specific example, in the sentence "The small dog barked", the correct label for "small" would be the edge ("dog", "small", adjectival-modifier).We use a neural graph-based dependency parser similar to the one from BID10 .

It first runs a BiRNN encoder over the sentence as described in section 3.3, producing a sequence of DISPLAYFORM0 is passed through two separate multilayer perceptrons, one producing a representation for x t i as a head word and one producing a representation for it as a dependent.

A bilinear classifier applied to these representations produces a score for each candidate edge.

Lastly, these scores are passed through a softmax layer to produce probabilities.

Mathematically, the probability of an edge is given as p ?? ((u, t, r) DISPLAYFORM1 Where s is the scoring function s(z 1 , z 2 , r) = MLP head (z 1 )(W r + W )MLP dep (z 2 ).

The bilinear classifier uses a weight matrix W r specific to the candidate relation as well as a weight matrix W shared across all relations.

We add four auxiliary prediction layers to our model for cross-view training: DISPLAYFORM2 Each auxiliary layer has some missing context (not seeing either the preceding or following words) for the candidate head and candidate dependent.

All the parameters for the scoring function of each auxiliary prediction layer are layer-specific.

To validate our approach, we evaluate Cross-View Training on two semi-supervised learning benchmarks.

These discard most of the labels from standard image image recognition datasets to artificially make them semi-supervised.

As a sterner test of our approach, we also apply CVT to five tasks from Natural Language Processing (NLP) using hundreds of millions of unlabeled sentences for semi-supervised learning.

Data.

We experiment on two semi-supervised image recognition benchmarks.

These are constructed from the CIFAR-10 ( BID23 ) and Street View House Numbers (SVHN) BID40 datasets.

Following previous work, we make the datasets semi-supervised by only using the provided labels for a subset of the examples in the training set; the rest are treated as unlabeled examples.

Model.

We use the convolutional neural network from Miyato et al. We add 36 auxiliary softmax layers to the 6 ?? 6 collection of feature vectors produced by the CNN.

Each auxiliary layer sees a patch of the image ranging in size from 21 ?? 21 pixels (the corner) to 29 ?? 29 pixels (the center) of the 32 ?? 32 pixel images.

We optimize L with ?? 1 = 1 and each minibatch consisting of 32 labeled and 128 unlabeled examples.

Miyato et al. use Virtual Adversarial Training (VAT) , minimizing L consistency with the input perturbation ?? chosen adversarially.

We train our cross-view models (which instead use L CVC ) both with and without this adversarial noise.

We report results with and without using data augmentation (random translations for SVHN and random translations and horizontal flipping for CIFAR-10) in Table 1 .Results.

CVT works well as semi-supervised learning method without any noise being added to the student.

When random noise is added, it performs close to VAT (the standard-deviation-based confidence intervals intersect) while training much faster (requiring only one backwards pass for each training minibatch, while VAT requires an additional one to compute the adversarial perturbation).

Our method can easily be combined with VAT, resulting in further improvements and state-of-theart results.

The benefit of CVT is less when data augmentation is applied, perhaps because random translations of the input expose the model to different "views" in a similar manner as with CVT.

We believe the gains on SVHN are smaller than CIFAR-10 because the digits in SVHN occur in the center of the image, so the auxiliary softmaxes seeing the sides and corner do not learn as effectively.

We also note that incorporating auxiliary softmax layers into the supervised loss L sup does not improve results (see Appendix C).

This indicates that the benefit of CVT comes from the improved self-training mechanism, not the additional losses regularizing the model.

Model Analysis.

To understand why CVT produces better results, we compare the behavior of the VAT and CVT (with adversarial noise) models trained on CIFAR-10.

First, we record the average value of each feature vector produced by the CNNs when they run over the test set.

As shown in the left of FIG5 , the CVT model has higher activation strengths for the feature vectors corresponding to the edges of the image.

We hypothesize that the VAT model fits to the data while primarily using BID8 f Miyato et al. (2017b) *We found Miyato et al.'s implementation produces slightly different results than the ones they report in their paper.

Table 1 : Error rates on semi-supervised learning benchmarks.

We report means and standard deviations from 5 runs.

+ after a dataset means data augmentation was applied.

In contrast, the model with CVT must learn meaningful representations for the edge regions in order to train the corresponding auxiliary softmax layers.

As these feature vectors are more useful, their magnitude become larger so they contribute more to the final representation produced by the global average pool.

To compare to discriminatory power of the feature vectors, we freeze the weights of the CNNs and add auxiliary softmax layers that are trained from scratch.

We then measure the accuracies of the added layers (see the center and right of FIG5 .

Unsurprisingly, the VAT model, which only learns representations that will be useful after the average pool, has much lower accuracies from individual feature vectors.

The difference is particularly striking in the sides and corners, where CVT accuracies are around 50% higher (they are about 25% higher in the center).

This finding further indicates that CVT is improving the model's representations, particularly for the outside parts of images.

Data.

Although the widely-used benchmarks in the previous section provide validation of our approach, they are small datasets that are artificially made to be semi-supervised.

In this section, we show CVT is successful on well-studied tasks where semi-supervised learning is rarely applied.

In particular, we train semi-supervised models on the following NLP tasks:??? Combinatory Category Grammar (CCG) Supertagging: Labeling words with CCG supertags: lexical categories that encode information about the predicate-argument structure of the sentence.

CCG is widely used in syntactic and semantic parsing.

We use data from CCGBank BID19 and report word-level accuracy.??? Text Chunking: Dividing a sentence into syntactically correlated parts (e.g., a noun phrase followed by a verb phrase).

We use the CoNLLL-2000 shared task data (Tjong Kim BID60 and report the F1 score over predicted chunks.??? Named Entity Recognition (NER): Identifying and classifying named entities (organizations, places, etc.) in a sentence.

We use the CoNLL-2003 dataset (Tjong Kim BID61 and report entity-level F1 score.??? Part-of-Speech (POS) Tagging: Labeling words with their syntactic categories (e.g., determiner, adjective, etc.) .

We use the Wall Street Journal (WSJ) portion of the Penn Treebank BID36 and report word-level accuracy.??? Dependency Parsing:

Inferring a tree-structure describing the syntactic structure of a sentence.

We use the Penn Treebank converted to Stanford Dependencies (version 3.3.0) and report unlabeled and labeled attachment score (UAS and LAS).We use the 1 Billion Word Language Model Benchmark BID3 as a pool of unlabeled sentences for semi-supervised learning.

Models.

We use a CNN-BiLSTM sequence tagging model BID5 .

The model first represents each word as the sum of a word embedding and the output of a character-level CNN.

This sequence of word representations is then fed through two BiLSTM layers and a softmax layer to produce predictions.

See Appendix A for details about the model.

Our dependency parser uses the same CNN-BiLSTM encoder as our sequence tagger.

As described in Section 3.4, a MLP-Bilinear classifier on top of the encoder makes the predictions.

Although it is common for dependency parsers to take words and part-of-speech tags as inputs, our model only takes words as inputs.

See Appendix B for details about the model.

BID38 were able to apply Virtual Adversarial Training to document classification, but we found VAT ineffective for our word-level tasks.

Although we experimented with constraining the word embeddings to unit length and adding random or adversarial perturbations to them during training, it did not improve performance.

This is perhaps because, unlike with RGB values in an image, words are discrete, so adding noise to their representations is less meaningful.

Instead, we add dropout to the student but not the teacher.

Recent work BID48 has shown that jointly training a neural language model with sequence taggers improves results.

We report accuracies with and without this enhancement (training the language model on the unlabeled data).

See TAB1 for sequence tagging results and TAB3 for dependency parsing results.

Results.

CVT significantly improves over the supervised baseline on all tasks, both with and without the auxiliary language modeling objective.

We report a new state-of-the-art for CCG-supertagging and pure dependency parsing (i.e., without using constituency parse annotations) and results competitive with the current state-of-the-art on the other tasks.

Our dependency parsing result is particularly important because our model does not include part-of-speech tags as input, which other works have shown to improve performance notably BID10 BID4 .

Of the prior results listed in the BID31 b c BID30 d BID62 e BID15 f Peters et al. (2017) g * The full TagLM model has many times more parameters than ours.

TagLM-2048 is of more comparable size to our models, although still larger.

TAB1 : Results for sequence tagging tasks.

We report the means and standard deviation of 5 runs.

"Baseline" trains with L sup , "Consistency" trains with" L sup + L consistency , and "CVT" trains with L sup + L CVC .

+LM means language modeling is added as an auxiliary task on the unlabeled data.

Depparse UAS Depparse LAS BID15 94.67 92.90 BID35 94.9 93.0 BID53 95.33 - BID10 95 BID25 , and BID32 because these train constituency parsers and convert the system outputs to dependency parses.

They produce higher scores, but have access to more information during training and do not apply to datasets without constituency annotations.

Although the large TagLM model is competitive with ours for Chunking and NER, reducing the size of TagLM to having 2048 hidden units already causes it to perform worse than our model.

Although there has been a large body of work successfully applying consistency-cost-based learning to vision tasks, we find it does not provide the same gains for NLP.

Training a model with the consistency loss L consistency did not improve over the baseline for sequence tagging and only slightly improved over the baseline for dependency parsing.

This result is perhaps due to the lack of benefit from adding noise when the input consists of discrete tokens as discussed earlier.

CVT, on the other hand, works well as a semi-supervised learning method for NLP.

mance, but the "future" and "past" layers are more beneficial than the "forward" and "backward" ones, perhaps because theses provide a more distinct and challenging view of the input.

Training Larger NLP Models.

Most sequence taggers and dependency parsers in prior use work small LSTMs (hidden state sizes of at most 500 units) because larger models yield little to no gains in performance BID49 .

We found our own supervised approaches and, to a lesser extent, our models when only using language modeling as the auxiliary task to also not benefit from increasing the model size.

In contrast, when using CVT accuracy scales much better with model size (see FIG6 .

This result suggests the appropriate semi-supervised learning methods may enable the development of larger, more sophisticated models for natural language processing tasks with limited amounts of labeled data.found this to slightly improve accuracy and significantly reduce the variance in accuracy between models trained with different random initializations.

For Chunking and Named Entity Recognition, we use a BIOES tagging scheme.

The model is trained using SGD with momentum BID45 BID57 .

Word embeddings are initialized with GloVe vectors BID42

We use the same 2-layer CNN-BiLSTM encoder and the same hyperparameters as listed in Appendix A. The MLPs used to produce representations for candidate head and dependent words have one hidden layer of size 512 with a ReLU activation and an output layer of size 256.

We apply dropout to the output of the hidden layer.

We omit punctuation from evaluation, which is standard practice for the PTB-SD 3.3.0 dataset.

In initial experiments, we explored whether cross-view losses could benefit purely supervised classifiers.

To do this, we trained models with the following objective: DISPLAYFORM0 See Section 3.1 for an explanation of the notation.

We hoped that adding auxiliary softmax layers with different views of the input would act as a regularizer on the model.

However, we found little to no benefit from this approach.

For sequence tagging, results improved slightly on CCG and POS but degraded on NER and Chunking.

For image recognition, we augmented WideResNet BID63 with auxiliary softmax layers and evaluated it on CIFAR-10 and CIFAR-100.

On both datasets, the augmented model performed slightly worse (by ???0.2% on CIFAR-10 and ???0.9% on CIFAR-100).We also experimented with using L sup-cv instead of of L sup on semi-supervised CIFAR-10 and CIFAR-10+.

Surprisingly, it (slightly) decreased performance for all of the methods we experimented with: supervised training, VAT, CVT, and CVT with adversarial noise.

We note we only tried these experiments with ?? 1 = 1, but this value of ?? 1 did work well for the semi-supervised setting.

These negative results suggest that the gains are from CVT are from the improved self-training mechanism, not the additional prediction layers regularizing the model.

@highlight

Self-training with different views of the input gives excellent results for semi-supervised image recognition, sequence tagging, and dependency parsing.