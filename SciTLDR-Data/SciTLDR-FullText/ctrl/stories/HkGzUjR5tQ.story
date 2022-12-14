We propose a new architecture termed Dual Adversarial Transfer Network (DATNet) for addressing low-resource Named Entity Recognition (NER).

Specifically, two variants of DATNet, i.e., DATNet-F and DATNet-P, are proposed to explore effective feature fusion between high and low resource.

To address the noisy and imbalanced training data, we propose a novel Generalized Resource-Adversarial Discriminator (GRAD).

Additionally, adversarial training is adopted to boost model generalization.

We examine the effects of different components in DATNet across domains and languages and show that significant improvement can be obtained especially for low-resource data.

Without augmenting any additional hand-crafted features, we achieve new state-of-the-art performances on CoNLL and Twitter NER---88.16% F1 for Spanish, 53.43% F1 for WNUT-2016, and 42.83% F1 for WNUT-2017.

Named entity recognition (NER) is an important step in most natural language processing (NLP) applications.

It detects not only the type of named entity, but also the entity boundaries, which requires deep understanding of the contextual semantics to disambiguate the different entity types of same tokens.

To tackle this challenging problem, most early studies were based on hand-crafted rules, which suffered from limited performance in practice.

Current methods are devoted to developing learning based algorithms, especially neural network based methods, and have been advancing the state-of-the-art consecutively BID7 BID23 BID6 BID33 .

These end-to-end models generalize well on new entities based on features automatically learned from the data.

However, when the annotated corpora is small, especially in the low resource scenario BID56 , the performance of these methods degrades significantly since the hidden feature representations cannot be learned adequately.

Recently, more and more approaches have been proposed to address low-resource NER.

Early works BID5 BID24 primarily assumed a large parallel corpus and focused on exploiting them to project information from high-to low-resource.

Unfortunately, such a large parallel corpus may not be available for many low-resource languages.

More recently, cross-resource word embedding BID9 BID0 BID52 was proposed to bridge the low and high resources and enable knowledge transfer.

Although the aforementioned transferbased methods show promising performance in low-resource NER, there are two issues deserved to be further investigated on: 1) Representation Difference -they did not consider the representation difference across resources and enforced the feature representation to be shared across languages/domains; 2) Resource Data Imbalance -the training size of high-resource is usually much larger than that of low-resource.

The existing methods neglect such difference in their models, resulting in poor generalization.

In this work, we present an approach termed Dual Adversarial Transfer Network (DATNet) to address the above issues in a unified framework for low-resource NER.

Specifically, to handle the representation difference, we first investigate on two architectures of hidden layers (we use bidirectional long-short term memory (BiLSTM) model as hidden layer) for transfer.

The first one is that all the units in hidden layers are common units shared across languages/domains.

The second one is composed of both private and common units, where the private part preserves the independent language/domain information.

Extensive experiments are conducted to show their advantages over each other in different situations.

On top of common units, the adversarial discriminator (AD) loss is introduced to encourage the resource-agnostic representation so that the knowledge from high resource can be more compatible with low resource.

To handle the resource data imbalance issue, we further propose a variant of the AD loss, termed Generalized Resource-Adversarial Discriminator (GRAD), to impose the resource weight during training so that low-resource and hard samples can be paid more attention to.

In addition, we create adversarial samples to conduct the Adversarial Training (AT), further improving the generalization and alleviating over-fitting problem.

We unify two kinds of adversarial learning, i.e., GRAD and AT, into one transfer learning model, termed Dual Adversarial Transfer Network (DATNet), to achieve end-to-end training and obtain the state-of-the-art performance on a series of NER tasks-88.16% F1 for CoNLL-2002 Spanish, 53.43% and 42.83% F1 for WNUT-2016 .

Different from prior works, we do not use additional hand-crafted features and do not use cross-lingual word embeddings while addressing the cross-language tasks.

NER is typically framed as a sequence labeling task which aims at automatic detection of named entities (e.g., person, organization, location and etc.) from free text BID35 .

The early works applied CRF, SVM, and perception models with handcrafted features BID44 BID42 BID31 .

With the advent of deep learning, research focus has been shifting towards deep neural networks (DNN), which requires little feature engineering and domain knowledge BID23 BID57 .

BID7 proposed a feed-forward neural network with a fixed sized window for each word, which failed in considering useful relations between long-distance words.

To overcome this limitation, BID6 presented a bidirectional LSTM-CNNs architecture that automatically detects word-and character-level features.

BID33 further extended it into bidirectional LSTM-CNNs-CRF architecture, where the CRF module was added to optimize the output label sequence.

proposed task-aware neural language model termed LM-LSTM-CRF, where character-aware neural language models were incorporated to extract characterlevel embedding under a multi-task framework.

Transfer Learning for NER Transfer learning can be a powerful tool to low resource NER tasks.

To bridge high and low resource, transfer learning methods for NER can be divided into two types: the parallel corpora based transfer and the shared representation based transfer.

Early works mainly focused on exploiting parallel corpora to project information between the high-and low-resource language BID53 BID5 BID24 BID10 .

For example, BID5 and BID10 proposed to jointly identify and align bilingual named entities.

On the other hand, the shared representation methods do not require the parallel correspondence BID46 .

For instance, BID9 proposed cross-lingual word embeddings to transfer knowledge across resources.

BID52 presented a transfer learning approach based on a deep hierarchical recurrent neural network (RNN), where full/partial hidden features between source and target tasks are shared.

BID38 BID39 utilized the Wikipedia entity type mappings to improve low-resource NER.

BID2 built massive multilingual annotators with minimal human expertise by using language agnostic techniques.

BID36 created a cross-language NER system, which works well for very minimal resources by translate annotated data of high-resource into low-resource.

BID8 proposed character-level neural CRFs to jointly train and predict low-and high-resource languages.

BID40 proposes a large-scale cross-lingual named entity dataset which contains 282 languages for evaluation.

In addition, multi-task learning BID51 BID32 BID45 BID1 BID15 BID28 shows that jointly training on multiple tasks/languages helps improve performance.

Different from transfer learning methods, multi-task learning aims at improving the performance of all the resources instead of low resource only.

Adversarial Learning Adversarial learning originates from Generative Adversarial Nets (GAN) BID12 , which shows impressing results in computer vision.

Recently, many papers have tried to apply adversarial learning to NLP tasks.

BID30 presented an adversarial multi-task learning framework for text classification.

BID14 applied the adversarial discriminator to POS tagging for Twitter.

BID18 proposed a language discriminator to enable language-adversarial training for cross-language POS tagging.

Apart from adversarial discriminator, adversarial training is another concept originally introduced by BID49 BID13 to improve the robustness of image classification model by injecting malicious perturbations into input images.

Recently, BID37 proposed a semi-supervised text classification method by applying adversarial training, where for the first time adversarial perturbations were added onto word embeddings.

BID54 applied adversarial training to POS tagging.

Different from all these adversarial learning methods, our method integrates both the adversarial discriminator and adversarial training in an unified framework to enable end-to-end training.

In this section, we introduce DATNet in more details.

We first describe a base model for NER, and then discuss two proposed transfer architectures for DATNet.

We follow state-of-the-art models for NER task BID23 BID6 BID33 , i.e., LSTM-CNNs-CRF based structure, to build the base model.

It consists of the following pieces: character-level embedding, word-level embedding, BiLSTM for feature representation, and CRF as the decoder.

The character-level embedding takes a sequence of characters in the word as atomic units input to derive the word representation that encodes the morphological information, such as root, prefix, and suffix.

These character features are usually encoded by character-level CNN or BiLSTM, then concatenated with word-level embedding to form the final word vectors.

On top of them, the network further incorporates the contextual information using BiLSTM to output new feature representations, which is subsequently fed into CRF layer to predict label sequence.

Although both of the word-level layer and the character-level layer can be implemented using CNNs or RNNs, we use CNNs for extracting character-level and RNNs for extracting word-level representation.

FIG0 shows the the architecture of the base model.

Previous works have shown that character features can boost sequence labeling performance by capturing morphological and semantic information BID28 .

For low-resource dataset to obtain high-quality word features, character features learned from other language/domain may provide crucial information for labeling, especially for rare and out-of-vocabulary words.

Character-level encoder usually contains BiLSTM BID23 and CNN BID6 BID33 approaches.

In practice, BID47 observed that the difference between the two approaches is statistically insignificant in sequence labeling tasks, but character-level CNN is more efficient and has less parameters.

Thus, we use character-level CNN and share character features between high-and low-resource tasks to enhance the representations of low-resource.

To learn a better word-level representation, we concatenate character-level features of each word with a latent word embedding as DISPLAYFORM0 ], where the latent word embedding w emb i is initialized with pre-trained embeddings and fixed during training.

One unique characteristic of NER is that the historical and future input for a given time step could be useful for label inference.

To exploit such a characteristic, we use a bidirectional LSTM architecture BID16 ) to extract contextualized word-level features.

In this way, we can gather the information from the past and future for a particular time frame t as follows, DISPLAYFORM1 After the LSTM layer, the representation of a word is obtained by concatenating its left and right context representation as follows, DISPLAYFORM2 To consider the resource representation difference on word-level features, we introduce two kinds of transferable word-level encoder in our model, namely DATNet-Full Share (DATNet-F) and DATNetPart Share (DATNet-P).

In DATNet-F, all the BiLSTM units are shared by both resources while word embeddings for different resources are disparate.

The illustrative figure is depicted in the FIG0

Different from DATNet-F, the DATNet-P decomposes the BiLSTM units into the shared component and the resource-related one, which is shown in the FIG0 .

In order to make the feature representation extracted from the source domain more compatible with those from the target domain, we encourage the outputs of the shared BiLSTM part to be resourceagnostic by constructing a resource-adversarial discriminator, which is inspired by the LanguageAdversarial Discriminator proposed by BID18 .

Unfortunately, previous works did not consider the imbalance of training size for two resources.

Specifically, the target domain consists of very limited labeled training data, e.g., 10 sentences.

In contrast, labeled training data in the source domain are much richer, e.g., 10k sentences.

If such imbalance was not considered during training, the stochastic gradient descent (SGD) optimization would make the model more biased to high resource BID27 .

To address this imbalance problem, we impose a weight ?? on two resources to balance their influences.

However, in the experiment we also observe that the easily classified samples from high resource comprise the majority of the loss and dominate the gradient.

To overcome this issue, we further propose Generalized Resource-Adversarial Discriminator (GRAD) to enable adaptive weights for each sample (note that the sample here means each sentence of resource), which focuses the model training on hard samples.

To compute the loss of GRAD, the output sequence of the shared BiLSTM is firstly encoded into a single vector via a self-attention module BID3 , and then projected into a scalar r via a linear transformation.

The loss function of the resource classifier is formulated as: DISPLAYFORM0 where I i???D S , I i???D T are the identity functions to denote whether a sentence is from high resource (source) and low resource (target), respectively; ?? is a weighting factor to balance the loss contribution from high and low resource; the parameter (1 ??? r i ) ?? (or r ?? i ) controls the loss contribution from individual samples by measuring the discrepancy between prediction and true label (easy samples have smaller contribution); and ?? scales the contrast of loss contribution from hard and easy samples.

In practice, the value of ?? does not need to be tuned much and usually set as 2 in our experiment.

Intuitively, the weighting factors ?? and (1 ??? r i ) ?? reduce the loss contribution from high resource and easy samples, respectively.

Note that though the resource classifier is optimized to minimize the resource classification error, when the gradients originated from the resource classification loss are back-propagated to the other model parts than the resource classifier, they are negated for parameter updates so that these bottom layers are trained to be resource-agnostic.

The label decoder induces a probability distribution over sequences of labels, conditioned on the word-level encoder features.

In this paper, we use a linear chain model based on the first-order Markov chain structure, termed the chain conditional random field (CRF) BID22 , as the decoder.

In this decoder, there are two kinds of cliques: local cliques and transition cliques.

Specifically, local cliques correspond to the individual elements in the sequence.

And transition cliques, on the other hand, reflect the evolution of states between two neighboring elements at time t ??? 1 and t and we define the transition distribution as ??.

Formally, a linear-chain CRF can be written as p(y|h 1: DISPLAYFORM0 W yt h t , where Z(h 1:T ) is a normalization term and y is the sequence of predicted labels as follows: y = y 1:T .

Model parameters are optimized to maximize this conditional log likelihood, which acts as the objective function of the model.

We define the loss function for source and target resources as follows, S = ??? i log p(y|h 1:T ), T = ??? i log p(y|h 1:T ).

So far our model can be trained end-to-end with standard back-propagation by minimizing the following loss: DISPLAYFORM0 Recent works have demonstrated that deep learning models are fragile to adversarial examples BID13 .

In computer vision, those adversarial examples can be constructed by changing a very small number of pixels, which are virtually indistinguishable to human perception BID43 .

Recently, adversarial samples are widely incorporated into training to improve the generalization and robustness of the model, which is so-called adversarial training (AT) BID37 .

It emerges as a powerful regularization tool to stabilize training and prevent the model from being stuck in local minimum.

In this paper, we explore AT in context of NER.

To be specific, we prepare an adversarial sample by adding the original sample with a perturbation bounded by a small norm to maximize the loss function as follows: DISPLAYFORM1 where ?? is the current model parameters set.

However, we cannot calculate the value of ?? exactly in general, because the exact optimization with respect to ?? is intractable in neural networks.

Following the strategy in BID13 , this value can be approximated by linearizing it as follows, DISPLAYFORM2 where can be determined on the validation set.

In this way, adversarial examples are generated by adding small perturbations to the inputs in the direction that most significantly increases the loss function of the model.

We find such ?? against the current model parameterized by ??, at each training step, and construct an adversarial example by x adv = x + ?? x .

Noted that we generate this adversarial example on the word and character embedding layer, respectively, as shown in the FIG0 (b) and 1(c).

Then, the classifier is trained on the mixture of original and adversarial examples to improve the generalization.

To this end, we augment the loss in Eqn.

2 and define the loss function for adversarial training as: DISPLAYFORM3 where (??; x), (??; x adv ) represents the loss from an original example and its adversarial counterpart, respectively.

Note that we present the AT in a general form for the convenience of presentation.

For different samples, the loss and parameters should correspond to their counterparts.

For example, for the source data with word embedding w S , the loss for AT can be defined as follows, AT = (??; w S ) + (??; w S,adv ) with w S,adv = w S + ?? w S and = GRAD + S .

Similarly, we can compute the perturbations ?? c for char-embedding and ?? w T for target word embedding.

In order to evaluate the performance of DATNet, we conduct the experiments on following widely used NER datasets: CoNLL-2003 English NER BID20 , CoNLL-2002 Spanish & Dutch NER BID19 , WNUT-2016 English Twitter NER (Zeman, 2017 .

The statistics of these datasets are described in TAB1 .

We use the official split of training/validation/test sets.

Since our goal is to study the effects of transferring knowledge from high-resource dataset to low-resource dataset, unlike previous works BID7 BID6 BID52 to append one-hot gazetteer features to the input of the CRF layer, and the works BID41 BID25 BID1 to introduce orthographic feature as additional input for learning social media NER in tweets, we do not experiment with hand-crafted features and only consider words and characters embeddings as the inputs of our model.

To be noted, we used only train set for model training for all datasets except the WNUT-2016 NER dataset.

Since in this dataset, all the previous studies merged the training and validation sets together for training, we followed the same way for fair comparison.

In addition to the CoNLL and WNUT datasets, we also experiment on the cross-language named entity dataset described in BID40 , which contains datasets for 282 languages, to evaluate our methods and investigate the transferability of different linguistic families and branches in both low-and high-resource scenarios.

We choose 9 languages in our experiment, where Galician (gl), West Frisian (fy), Ukrainian (uk) and Marathi (mr) are target languages, the corresponding source languages are Spanish (es), Dutch (nl), Russian (ru) and Hindi (hi), and Arabic (ar) is also a source language, which is from different linguistic family.

Following the setting in BID8 , we also simulate the low-and high-resource scenarios by creating 100 and 10,000 sentences split for training target language datasets, respectively.

Then we create 1,000 sentences split for validation and test, respectively.

For source languages, we create 10,000 sentence split for training only.

For high-resource scenario, we only conduct experiments on Galician (gl-high) and Ukrainian (uk-high).The list of selected datasets are described in TAB2 .

BID28 .

For the named entity datasets selected from BID40 , we use 300-dimensional pre-trained word embeddings trained by fastText package 3 on Wikipedia BID4 , and the 30-dimensional randomly initialized character embeddings are used for all the datasets.

We set the filter number as 20 for char-level CNN and the dimension of hidden states of the word-level LSTM as 200 for both base model and DATNet-F. For DATNet-P, we set 100 for source, share, and target LSTMs dimension, respectively.

Parameters optimization is performed by Adam optimizer BID21 with gradient clipping of 5.0 and learning rate decay strategy.

We set the initial learning rate of ?? 0 = 0.001 for all experiments.

At each epoch t, learning rate ?? t is updated using ?? t = ?? 0 /(1 + ?? ?? t), where ?? is decay rate with 0.05.

To reduce over-fitting, we also apply Dropout BID48 to the embedding layer and the output of the LSTM layer, respectively.

In this section, we compare our approach with state-of-the-art (SOTA) methods on CoNLL and WNUT benchmark datasets.

In the experiment, we exploit all the source data (i.e., CoNLL-2003 English NER) and target data to improve performance of target tasks.

The averaged results with standard deviation over 10 repetitive runs are summarized in TAB3 , and we also report the best results on each task for fair comparison with other SOTA methods.

From results, we observe that incorporating the additional resource is helpful to improve performance.

TAB4 summarizes the results of our methods under different cross-language transfer settings as well as the comparison with BID8 .

In this experiment, we study the transferability between languages not only from same linguistic family and branch, but also from different linguistic families or branches.

According to the results, DATNets outperform the transfer method of BID8 for both low-and high-resource scenarios within the same linguistic family and branch (i.e., in-family in-branch) transfer case.

We also observe that: 1) For the low-resource scenario, transfer learning is significantly helpful for improving the performance of target datasets within both same and different linguistic family or branch (i.e., in/cross-family in/cross-branch) transfer cases, while the improvements are more prominent under the in-family in-branch case.

2) For the high-resource scenario, say, when the target language data is sufficient, the improvements of transfer learning are not very distinct compared with that for low-resource scenario under in-family in-branch case.

We also find that there is no effect by transferring knowledge from Arabic to Galician and Ukrainian.

We suspect that it is caused by the great linguistic differences between source and target languages, since, for example, Arabic and Galician are from totally different linguistic families.

In this section, we investigate on improvements with transfer learning under multiple low-resource settings with partial target data.

To simulate a low-resource setting, we randomly select subsets of target data with varying data ratio at 0.

05, 0.1, 0.2, 0.4, 0.6, and 1.0.

For example, 20, 748 training tokens are sampled from the training set under a data ratio of r = 0.1 for the dataset CoNLL-2002 Spanish NER (Cf.

TAB1 ).

The results for cross-language and cross-domain transfer are shown in FIG2 (a) and 2(b), respectively, where we compare the results with each part of DATNet under various data ratios.

From those figures, we have the following observations: 1) both adversarial training and adversarial discriminator in DATNet consistently contribute to the performance improvement; 2) the transfer learning component in the DATNet consistently improve over the base model results and the improvement margin is more substantial when the target data ratio is lower.

For example, when the data ratio is 0.05, DATNet-P model outperforms the base model by more than 4% absolutely in F1-score on Spanish NER and DATNet-F model improves around 13% absolutely in F1-score compared to base model on WNUT-2016 NER.In the second experiment, we further investigate DATNet on the extremely low resource cases, e.g., the number of training target sentences is 10, 50, 100, 200, 500 and 1,000.

The setting is quite challenging and fewer previous works have studied before.

The results are summarized in TAB5 .

We have two interesting observations 5 : 1) DATNet-F outperforms DATNet-P on cross-language transfer when the target resource is extremely low, however, this situation is reversed when the target dataset size is large enough (here for this specific dataset, the threshold is 100 sentences); 2) DATNet-F is always superior to DATNet-P on cross-domain transfer.

For the first observation, it is because DATNet-F with more shared hidden units is more efficient to transfer knowledge than DATNet-P when data size is extremely small.

For the second observation, because cross-domain transfer are in the same language, more knowledge is common between the source and target domains, requiring more shared hidden features to carry with these knowledge compared to cross-language transfer.

Therefore, for cross-language transfer with an extremely low resource and cross-domain transfer, we suggest using DATNet-F model to achieve better performance.

As for cross-language transfer with relatively more training data, DATNet-P model is preferred.

In the proposed DATNet, both GRAD and AT play important roles in low resource NER.

In this experiment, we further investigate how GRAD and AT help transfer knowledge across language/domain.

In the first experiment 6 , we used t-SNE BID34 to visualize the feature distribution of BiLSTM outputs without AD, with normal AD (GRAD without considering data imbalance), and with the proposed GRAD in FIG3 .

From this figure, we can see that the GRAD in DATNet makes the distribution of extracted features from the source and target datasets 5 For other tasks/languages we have the similar observation, we only report CoNLL-2002 Spanish and WNUT-2016 Twitter results due to the page limit.

6 We used data ratio ?? = 0.5 for training model and randomly selected 10k testing data for visualization.

much more similar by considering the data imbalance, which indicates that the outputs of BiLSTM are resource-invariant.

To better understand the working mechanism, TAB6 further reports the quantitative performance comparison between models with different components.

We observe that GRAD shows the stable superiority over the normal AD regardless of other components.

There are no always winner between DATNet-P and DATNet-F on different settings.

DATNet-P architecture is more suitable to cross-language transfer while DATNet-F is more suitable to cross-domain transfer.

From the previous results, we know that AT helps enhance the overall performance by adding perturbations to inputs with the limit of = 5, i.e., ?? 2 ??? 5.

In this experiment, we further investigate how target perturbation w T with fixed source perturbation w S = 5 in AT affects knowledge transfer and the results on Spanish NER are summarized in TAB7 .

The results generally indicate that less training data require a larger to prevent over-fitting, which further validates the necessity of AT in the case of low resource data.

Finally, we analyze the discriminator weight ?? in GRAD and results are summarized in TAB8 .

From the results, it is interesting to find that ?? is directly proportional to the data ratio ??, basically, which means that more target training data requires larger ?? (i.e., smaller 1 ??? ?? to reduce training emphasis on the target domain) to achieve better performance.

In this paper we develop a transfer learning model DATNet for low-resource NER, which aims at addressing two problems remained in existing work, namely representation difference and resource data imbalance.

We introduce two variants of DATNet, DATNet-F and DATNet-P, which can be chosen for use according to the cross-language/domain user case and the target dataset size.

To improve model generalization, we propose dual adversarial learning strategies, i.e., AT and GRAD.

Extensive experiments show the superiority of DATNet over existing models and it achieves new state-of-the-art performance on CoNLL NER and WNUT NER benchmark datasets.

<|TLDR|>

@highlight

We propose a new  architecture termed Dual Adversarial Transfer Network (DATNet) for addressing low-resource Named Entity Recognition (NER) and achieve new state-of-the-art performances on CoNLL and Twitter NER.