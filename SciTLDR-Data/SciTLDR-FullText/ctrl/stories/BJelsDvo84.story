We present EDA: easy data augmentation techniques for boosting performance on text classification tasks.

EDA consists of four simple but powerful operations: synonym replacement, random insertion, random swap, and random deletion.

On five text classification tasks, we show that EDA improves performance for both convolutional and recurrent neural networks.

EDA demonstrates particularly strong results for smaller datasets; on average, across five datasets, training with EDA while using only 50% of the available training set achieved the same accuracy as normal training with all available data.

We also performed extensive ablation studies and suggest parameters for practical use.

Text classification is a fundamental task in natural language processing (NLP).

Machine learning and deep learning have achieved high accuracy on tasks ranging from sentiment analysis (Tang et al., 2015) to topic classification BID24 , but high performance is often dependent on the size and quality of training data, which is often tedious to collect.

Automatic data augmentation is commonly used in vision BID20 BID22 BID10 and speech (Cui et al., 2015; BID7 and can help train more robust models, particularly when using smaller datasets.

However, because it is difficult to come up with generalized rules for language transformation, universal data augmentation techniques in NLP have not been explored.

Previous work has proposed techniques for data augmentation in NLP.

One popular study generated new data by translating sentences into French and back into English BID28 .

Other works have used predictive language models for synonym replacement BID8 and data noising as smoothing BID27 .

Although these techniques are valid, they are not often used in practice because they have a high cost of implementation relative to performance gain.

In this paper, we present a simple set of universal data augmentation techniques for NLP called EDA (easy data augmentation).

To the best of our knowledge, we are the first to comprehensively explore text editing techniques for data augmentation.

We systematically evaluate EDA on five benchmark classification tasks, and results show that EDA provides substantial improvements on all five tasks and is particularly helpful for smaller datasets.

Code will be made publicly available.

Operation Sentence None A sad, superior human comedy played out on the back roads of life.

SR A lamentable, superior human comedy played out on the backward road of life.

RI A sad, superior human comedy played out on funniness the back roads of life.

RS A sad, superior human comedy played out on roads back the of life.

RD A sad, superior human out on the roads of life.

Frustrated by the measly performance of text classifiers trained on small datasets, we tested a number of augmentation operations loosely inspired by those used in vision and found that they helped train more robust models.

Here, we present the full details of EDA.

For a given sentence in the training set, we perform the following operations: 1.

Synonym Replacement (SR): Randomly choose n words from the sentence that are not stop words.

Replace each of these words with one of its synonyms chosen at random.

2.

Random Insertion (RI): Find a random synonym of a random word in the sentence that is not a stop word.

Insert that synonym into a random position in the sentence.

Do this n times.

3.

Random Swap (RS): Randomly choose two words in the sentence and swap their positions.

Do this n times.

4.

Random Deletion (RD): Randomly remove each word in the sentence with probability p. Since long sentences have more words than short ones, they can absorb more noise while maintaining their original class label.

To compensate, we vary the number of words changed, n, for SR, RI, and RS based on the sentence length l with the formula n=?? l, where ?? is a parameter that indicates the percent of the words in a sentence are changed (we use p=?? for RD).

Furthermore, for each original sentence, we generate n aug augmented sentences.

Examples of augmented sentences are shown in TAB0 .

We note that synonym replacement has been used previously BID9 BID26 , but to our knowledge, random insertions, swaps, and deletions have not been studied.

We conduct experiments on five benchmark text classification tasks: (1) SST-2:

Stanford Sentiment Treebank BID21 , (2) CR: customer reviews BID3 BID13 , (3) SUBJ: subjectivity/objectivity dataset BID15 ), (4) TREC: question type dataset BID11 , and (5) PC: Pro-Con dataset BID2 .

Summary statistics are shown in TAB3 in the Appendix.

Furthermore, we hypothesize that EDA is more helpful for smaller datasets, so we delegate the following sized datasets by selecting a random subset of the full training set with N train ={500, 2,000, 5,000, all available data}.We run experiments for two state-of-the-art models in text classification.

(1) Recurrent neural networks (RNNs) are suitable for sequential data.

We use a LSTM-RNN BID12 .

(2) Convolutional neural networks (CNNs) have also achieved high performance for text classification.

We implement them as described in BID6 .

Details are in Section 6.1 in the Appendix.

We run both CNN and RNN models with and without EDA across all five datasets for varying training set sizes.

Average performances (%) are shown in Table 2 .

Of note, average improvement was 0.8% for full datasets and 3.0% for N train =500.

Table 2 : Average performances (%) across five text classification tasks for models without and without EDA on different training set sizes.

Overfitting tends to be more severe when training on smaller datasets.

By conducting experiments using a restricted fraction of the available training data, we show that EDA has more significant improvements for smaller training sets.

We run both normal training and EDA training for the following training set fractions (%): {1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100} .

FIG0 shows average performance across all datasets.

The best average accuracy without augmentation, 88.3%, was achieved using 100% of the training data.

Models trained using EDA surpassed this number by achieving an average accuracy of 88.6% while only using 50% of the available training data.

Results for individual datasets are displayed in FIG3 (Appendix).

In data augmentation, input data is altered while class labels are maintained.

However, if sentences are significantly changed, then original class labels may no longer be valid.

We take a visualization approach to examine whether EDA operations significantly change the meanings of augmented sentences.

First, we train an RNN on the pro-con classification task (PC) without augmentation.

Then, we apply EDA to the test set by generating nine augmented sentences per original sentence.

These are fed into the RNN along with the original sentences, and we extract the outputs from the last dense layer.

We apply t-SNE (Van Der Maaten, 2014) to these vectors and plot their 2-D representations FIG1 ).

We found that the resulting latent space representations for augmented sentences closely surrounded those of the original sentences.

3.4 ABLATION STUDIES So far, we have shown encouraging empirical results.

In this section, we perform ablation studies to explore the effects of each component in EDA.

Synonym replacement has been previously used BID9 BID26 , but the other three EDA operations have not yet been explored.

One could hypothesize that the bulk of EDA's performance gain is from synonym replacement, so we isolate each of the EDA operations to determine their individual ability to boost performance.

For all four operations, we ran models using a single operation while varying the augmentation parameter ??={0.05, 0.1, 0.2, 0.3, 0.4, 0.5} FIG2 ).It turns out that all four EDA operations contribute to performance gain.

For SR, improvement was good for small ??, but high ?? hurt performance, likely because replacing too many words in a sentence changed the identity of the sentence.

For RI, performance gains were more stable for different ?? values, possibly because the original words in the sentence and their relative order were maintained in this operation.

RS yielded high performance gains at ?????0.2, but declined at ?????0.3 since performing too many swaps is equivalent to shuffling the entire order of the sentence.

RD had the highest gains for low ?? but severely hurt performance at high ??, as sentences are likely unintelligible if up to half the words are removed.

Improvements were more substantial on smaller datasets for all operations, and ??=0.1 appeared to be a "sweet spot" across the board.

The natural next step is to determine how the number of generated augmented sentences per original sentence, n aug , affects performance.

We calculate average performances over all datasets for n aug ={1, 2, 4, 8, 16, 32}, as shown in FIG2 (middle).On smaller training sets, overfitting was more likely, so generating many augmented sentences yielded large performance boosts.

For larger training sets, adding more than four augmented sentences per original sentence was unhelpful since models tend to generalize properly when large quantities of real data are available.

Based on these results, we recommend parameters for practical use in FIG2 (right).

Related work is creative but often complex.

Back-translation BID18 , translational data augmentation BID1 , and noising BID27 have shown improvements in BLEU measure for machine translation.

For other tasks, previous approaches include task-specific heuristics BID5 and back-translation BID19 BID28 .

Regarding synonym replacement (SR), one study showed a 1.4% F1-score boost for tweet classification by finding synonyms with k-nearest neighbors using word embeddings BID26 ).

Another study found no improvement in temporal analysis when replacing headwords with synonyms BID9 , and mixed results were reported for using SR in character-level text classification ; however, neither work conducted extensive ablation studies.

Most studies explore data augmentation as a complementary result for translation or in a taskspecific context, so it is hard to directly compare EDA to previous literature.

But there are two studies similar to ours that evaluate augmentation techniques on multiple datasets.

BID4 proposed a generative model that combines a variational auto-encoder (VAE) and attribute discriminator to generate fake data, demonstrating a 3% gain in accuracy on two datasets.

BID8 showed that replacing words with other words that were predicted from the sentence context using a bi-directional language model yielded a 0.5% gain on five datasets.

However, training a variational auto-encoder or bidirectional LSTM language model is a lot of work.

EDA yields similar results but is much easier to use because it does not require training a language model and does not use external datasets.

In TAB5 (Appendix), we show EDA's ease of use compared to other techniques.

We have shown that simple data augmentation operations can boost performance on text classification tasks.

Although improvement is at times marginal, EDA substantially boosts performance and reduces overfitting when training on smaller datasets.

Continued work on this topic could include exploring the theoretical underpinning of the EDA operations.

We hope that EDA's simplicity makes a compelling case for its widespread use in NLP.

This section contains implementation details, dataset statistics, and detailed results not included in the main text.

All code for EDA and the experiments in this paper will be made available.

The following implementation details were omitted from the main text:Synonym thesaurus.

All synonyms for synonym replacements and random insertions were generated using WordNet BID14 .

We suspect that EDA will work with any thesaurus.

Word embeddings.

We use 300-dimensional Common-Crawl word embeddings trained using GloVe BID16 .

We suspect that EDA will work with any pre-trained word embeddings.

CNN.

We use the following architecture: input layer, 1-D convolutional layer of 128 filters of size 5, global 1D max pool layer, dense layer of 20 hidden units with ReLU activation function, softmax output layer.

We initialize this network with random normal weights and train against the categorical cross-entropy loss function with the adam optimizer.

We use early stopping with a patience of 3 epochs.

RNN.

The architecture used in this paper is as follows: input layer, bi-directional hidden layer with 64 LSTM cells, dropout layer with p=0.5, bi-directional layer of 32 LSTM cells, dropout layer with p=0.5, dense layer of 20 hidden units with ReLU activation, softmax output layer.

We initialize this network with random normal weights and train against the categorical cross-entropy loss function with the adam optimizer.

We use early stopping with a patience of 3 epochs.

Summary statistics for the five datasets used are shown in TAB3 .

DISPLAYFORM0

In FIG3 , we show performance on individual text classification tasks for both normal training and training with EDA, with respect to percent of dataset used for training.

In FIG3 , we compare the EDA's ease of use to that of related work.1 BID4

How does using EDA improve text classification performance?

While it is hard to identify exactly how EDA improves the performance of classifiers, we believe there are two main reasons.

The first is that generating augmented data similar to original data introduces some degree of noise that helps prevent overfitting.

The second is that using EDA can introduce new vocabulary through the synonym replacement and random insertion operations, allowing models to generalize to words in the test set that were not in the training set.

Both these effects are more pronounced for smaller datasets.

Why should I use EDA instead of other techniques such as contextual augmentation, noising, GAN, or back-translation?

All of the above are valid techniques for data augmentation, and we encourage you to try them, as they may actually work better than EDA, depending on the dataset.

But because these techniques require the use of a deep learning model in itself to generate augmented sentences, there is often a high cost of implementing these techniques relative to the expected performance gain.

With EDA, we aim to provide a set of simple techniques that are generalizable to a range of NLP tasks.

Is there a chance that using EDA will actually hurt my performance?

Considering our results across five classification tasks, it's unlikely but there's always a chance.

It's possible that one of the EDA operations can change the class of some augmented sentences and create mislabeled data.

But even so, "deep learning is robust to massive label noise" BID17 .For random insertions, why do you only insert words that are synonyms, as opposed to inserting any random words?

Data augmentation operations should not change the true label of a sentence, as that would introduce unnecessary noise into the data.

Inserting a synonym of a word in a sentence, opposed to a random word, is more likely to be relevant to the context and retain the original label of the sentence.

<|TLDR|>

@highlight

Simple text augmentation techniques can significantly boost performance on text classification tasks, especially for small datasets.