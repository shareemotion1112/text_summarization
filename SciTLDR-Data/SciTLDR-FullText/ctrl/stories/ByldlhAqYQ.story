Transfer learning aims to solve the data sparsity for a specific domain by applying information of another domain.

Given a sequence (e.g. a natural language sentence), the transfer learning, usually enabled by recurrent neural network (RNN), represent the sequential information transfer.

RNN uses a chain of repeating cells to model the sequence data.

However, previous studies of neural network based transfer learning simply transfer the information across the whole layers, which are unfeasible for seq2seq and sequence labeling.

Meanwhile, such layer-wise transfer learning mechanisms also lose the fine-grained cell-level information from the source domain.



In this paper, we proposed the aligned recurrent transfer, ART, to achieve cell-level information transfer.

ART is in a recurrent manner that different cells share the same parameters.

Besides transferring the corresponding information at the same position, ART transfers information from all collocated words in the source domain.

This strategy enables ART to capture the word collocation across domains in a more flexible way.

We conducted extensive experiments on both sequence labeling tasks (POS tagging, NER) and sentence classification (sentiment analysis).

ART outperforms the state-of-the-arts over all experiments.

Most previous NLP studies focus on open domain tasks.

But due to the variety and ambiguity of natural language BID7 BID18 , models for one domain usually incur more errors when adapting to another domain.

This is even worse for neural networks since embeddingbased neural network models usually suffer from overfitting BID12 .

While existing NLP models are usually trained by the open domain, they suffer from severe performance degeneration when adapting to specific domains.

This motivates us to train specific models for specific domains.

The key issue of training a specific domain is the insufficiency of labeled data.

Transfer learning is one promising way to solve the insufficiency BID8 .

Existing studies BID3 BID8 have shown that (1) NLP models in different domains still share many common features (e.g. common vocabularies, similar word semantics, similar sentence syntaxes), and (2) the corpus of the open domain is usually much richer than that of a specific domain.

Our transfer learning model is under the pre-training framework.

We first pre-train the model for the source domain.

Then we fine-tune the model for the target domain.

Recently, some pre-trained models (e.g. BERT BID4 , ELMo (Peters et al., 2018) , GPT-2 (Radford et al., 2019) ) successfully learns general knowledge for text.

The difference is that these models use a large scale and domain-independent corpus for pre-training.

In this paper, we use a small scale but domaindependent corpus as the source domain for pre-training.

We argue that, for the pre-training corpus, the domain relevance will overcome the disadvantage of limited scale.

Most previous transfer learning approaches BID11 BID6 only transfer information across the whole layers.

This causes the information loss from cells in the source domain. ''Layer-wise transfer learning" indicates that the approach represents the whole sentence by a single vector.

So the transfer mechanism is only applied to the vector.

We highlight the effectiveness of precisely capturing and transferring information of each cell from the source domain in two cases.

First, in seq2seq (e.g. machine translation) or sequence labeling (e.g. POS tagging) tasks, all cells directly affect the results.

So layer-wise information transfer is unfeasible for these tasks.

Second, even for the sentence classification, cells in the source domain provide more fine-grained information to understand the target domain.

For example, in figure 1, parameters for "hate" are insufficiently trained.

The model transfers the state of "hate" from the source domain to understand it better.

Target: Sometimes I really hate RIBs.

Source:

Sometimes I really hate RIBs.

Figure 1 : Motivation of ART.

The orange words "sometimes" and "hate" are with insufficiently trained parameters.

The red line indicates the information transfer from the corresponding position.

The blue line indicates the information transfer from a collocated word.

Besides transferring the corresponding position's information, the transfer learning algorithm captures the cross-domain long-term dependency.

Two words that have a strong dependency on each other can have a long gap between them.

Being in the insufficiently trained target domain, a word needs to represent its precise meaning by incorporating the information from its collocated words.

Here "collocate" indicates that a word's semantics can have long-term dependency on other words.

To understand a word in the target domain, we need to precisely represent its collocated words from the source domain.

We learn the collocated words via the attention mechanism BID1 .

For example, in figure 1, "hate" is modified by the adverb "sometimes", which implies the act of hating is not serious.

But the "sometimes" in the target domain is trained insufficiently.

We need to transfer the semantics of "sometimes" in the source domain to understand the implication.

Therefore, we need to carefully align word collocations between the source domain and the target domain to represent the long-term dependency.

In this paper, we proposed ART (aligned recurrent transfer), a novel transfer learning mechanism, to transfer cell-level information by learning to collocate cross-domain words.

ART allows the celllevel information transfer by directly extending each RNN cell.

ART incorporates the hidden state representation corresponding to the same position and a function of the hidden states for all words weighted by their attention scores.

Cell-Level Recurrent Transfer ART extends each recurrent cell by taking the states from the source domain as an extra input.

While traditional layer-wise transfer learning approaches discard states of the intermediate cells, ART uses cell-level information transfer, which means each cell is affected by the transferred information.

For example, in figure 1, the state of "hate" in the target domain is affected by "sometimes" and "hate" in the source domain.

Thus ART transfers more fine-grained information.

Learn to Collocate and Transfer For each word in the target domain, ART learns to incorporate two types of information from the source domain: (a) the hidden state corresponding to the same word, and (b) the hidden states for all words in the sequence.

Information (b) enables ART to capture the cross-domain long-term dependency.

ART learns to incorporate information (b) based on the attention scores BID1 of all words from the source domain.

Before learning to transfer, we first pre-train the neural network of the source domain.

Therefore ART is able to leverage the pre-trained information from the source domain.

In this section, we elaborate the general architecture of ART.

We will show that, ART precisely learns to collocate words from the source domain and to transfer their cell-level information for the target domain.

Architecture The source domain and the target domain share an RNN layer, from which the common information is transferred.

We pre-train the neural network of the source domain.

Therefore the shared RNN layer represents the semantics of the source domain.

The target domain has an additional RNN layer.

Each cell in it accepts transferred information through the shared RNN layer.

Such information consists of (1) the information of the same word in the source domain (the red edge in figure 2); and (2) the information of all its collocated words (the blue edges in FIG0 ).

ART uses attention BID1 to decide the weights of all candidate collocations.

The RNN cell controls the weights between (1) and (2) by an update gate.

FIG0 shows the architecture of ART.

The yellow box contains the neural network for the source domain, which is a classical RNN.

The green box contains the neural network for the target domain.

S i and T i are cells for the source domain and target domain, respectively.

T i takes x i as the input, which is usually a word embedding.

The two neural networks overlap each other.

The source domain's neural network transfers information through the overlapping modules.

We will describe the details of the architecture below.

Note that although ART is only deployed over RNNs in this paper, its attentive transfer mechanism is easy to be deploy over other structures (e.g. Transformer BID20 )) DISPLAYFORM0 where θ S is the parameter (recurrent weight matrix).Information Transfer for the Target Domain Each RNN cell in the target domain leverages the transferred information from the source domain.

Different from the source domain, the i-th hidden state in the target domain h T i is computed by: DISPLAYFORM1 where h T i−1 contains the information passed from the previous time step in the target domain (h T i−1 ), and the transferred information from the source domain (ψ i ).

We compute it by: DISPLAYFORM2 where θ f is the parameter for f .Note that both domains use the same RNN function with different parameters (θ S and θ T ).

Intuitively, we always want to transfer the common information across domains.

And we think it's easier to represent common and shareable information with an identical network structure.

Learn to Collocate and Transfer We compute ψ i by aligning its collocations in the source domain.

We consider two kinds of alignments: FORMULA0 The alignment from the corresponding position.

This makes sense since the corresponding position has the corresponding information of the source domain.

FORMULA1 The alignments from all collocated words of the source domain.

This alignment is used to represent the long-term dependency across domains.

We use a "concentrate gate" u i to control the ratio between the corresponding position and collocated positions.

We compute ψ i by: DISPLAYFORM3 where DISPLAYFORM4 π i denotes the transferred information from collocated words.• denotes the element-wise multiplication.

W u and C u are parameter matrices.

In order to compute π i , we use attention BID1 to incorporate information of all candidate positions in the sequence from the source domain.

We denote the strength of the collocation intensity to position j in the source domain as α ij .

We merge all information of the source domain by a weighted sum according to the collocation intensity.

More specifically, we define π i as: DISPLAYFORM5 where DISPLAYFORM6 a(h T i , h S j ) denotes the collocation intensity between the i-th cell in the target domain and the j-th cell in the source domain.

The model needs to be evaluated O(n 2 ) times for each sentence, due to the enumeration of n indexes for the source domain and n indexes for the target domain.

Here n denotes the sentence length.

By following BID1 , we use a single-layer perception: DISPLAYFORM7 where W a and U a are the parameter matrices.

Since U a h S j does not depend on i, we can pre-compute it to reduce the computation cost.

Update New State To compute f , we use an update gate z i to determine how much of the source domain's information ψ i should be transferred.

ψ i is computed by merging the original input x i , the previous cell's hidden state h T i−1 and the transferred information ψ i .

We use a reset gate r i to determine how much of h T i−1 should be reset to zero for ψ i .

More specifically, we define f as: DISPLAYFORM8 where DISPLAYFORM9 Here these W, U, C are parameter matrices.

Model Training: We first pre-train the parameters of the source domain by its training samples.

Then we fine-tune the pre-trained model with additional layers of the target domain.

The fine-tuning uses the training samples of the target domain.

All parameters are jointly fine-tuned.

In this section, we illustrate how we deploy ART over LSTM.

LSTM specifically addresses the issue of learning long-term dependency in RNN.

Instead of using one hidden state for the sequential memory, each LSTM cell has two hidden states for the long-term and short-term memory.

So the ART adaptation needs to separately represent information for the long-term memory and short-term memory.

The source domain of ART over LSTM uses a standard LSTM layer.

The computation of the t-th cell is precisely specified as follows: DISPLAYFORM0 Here h S t and c S t denote the short-term memory and long-term memory, respectively.

In the target domain, we separately incorporate the short-term and long-term memory from the source domain.

More formally, we compute the t-the cell in the target domain by: DISPLAYFORM1 DISPLAYFORM2 where f (h FORMULA0 ), respectively.

Bidirectional Network We use the bidirectional architecture to reach all words' information for each cell.

The backward neural network accepts the x i (i = 1 . . .

n) in reverse order.

We compute the final output of the ART over LSTM by concatenating the states from the forward neural network and the backward neural network for each cell.

We evaluate our proposed approach over sentence classification (sentiment analysis) and sequence labeling task (POS tagging and NER).

All the experiments run over a computer with Intel Core i7 4.0GHz CPU, 32GB RAM, and a GeForce GTX 1080 Ti GPU.Network Structure: We use a very simple network structure.

The neural network consists of an embedding layer, an ART layer as described in section 3, and a task-specific output layer for the prediction.

We will elaborate the output layer and the loss function in each of the tasks below.

We use 100d GloVe vectors BID13 as the initialization for ART and all its ablations.

Competitor Models We compare ART with the following ablations.• LSTM (no transfer learning): It uses a standard LSTM without transfer learning.

It is only trained by the data of the target domain.• LSTM-u It uses a standard LSTM and is trained by the union data of the source domain and the target domain.

• LSTM-s It uses a standard LSTM and is trained only by the data of the source domain.

Then parameters are used to predicting outputs of samples in the target domain.• Layer-Wise Transfer (LWT) (no cell-level information transfer): It consists of a layerwise transfer learning neural network.

More specifically, only the last cell of the RNN layer transfers information.

This cell works as in ART.

LWT only works for sentence classification.

We use LWT to verify the effectiveness of the cell-level information transfer.• Corresponding Cell Transfer (CCT) (no collocation information transfer): It only transfers information from the corresponding position of each cell.

We use CCT to verify the effectiveness of collocating and transferring from the source domain.

We also compare ART with state-of-the-art transfer learning algorithms.

For sequence labeling, we compare with hierarchical recurrent networks (HRN) and FLORS BID17 BID23 .

For sentence classification, we compare with DANN BID6 , DAmSDA BID6 , AMN BID10 , and HATN BID11 .

Note that FLORS, DANN, DAmSDA, AMN and HATN use labeled samples of the source domain and unlabeled samples of both the source domain and the target domain for training.

Instead, ART and HRN use labeled samples of both domains.

Datasets: We use the Amazon review dataset BID2 , which has been widely used for cross-domain sentence classification.

It contains reviews for four domains: books, dvd, electronics, kitchen.

Each review is either positive or negative.

We list the detailed statistics of the dataset in TAB0 .

We use the training data and development data from both domains for training and validating.

And we use the testing data of the target domain for testing.

Model Details: To adapt ART to sentence classification, we use a max pooling layer to merge the states of different words.

Then we use a perception and a sigmoid function to score the probability of the given sentence being positive.

We use binary cross entropy as the loss function.

The dimension of each LSTM is set to 100.

We use the Adam BID9 optimizer.

We use a dropout probability of 0.5 on the max pooling layer.

Results: We report the classification accuracy of different models in TAB1 .

The no-transfer LSTM only performs accuracy of 76.3% on average.

ART outperforms it by 9.5%.

ART also outperforms its ablations and other competitors.

This overall verifies the effectiveness of ART.Effectiveness of Cell-Level Transfer LWT only transfers layer-wise information and performs accuracy of 77.4% on average.

But ART and CCT transfer more fine-grained cell-level information.

CCT outperforms LWT by 2.9%.

ART outperforms LWT by 8.4%.

This verifies the effectiveness of the cell-level transfer.

Effectiveness of Collocation and Transfer CCT only transfers the corresponding position's information from the source domain.

It achieves accuracy of 80.3% on average.

ART outperforms CCT by 5.5% on average.

ART provides a more flexible way to transfer a set of positions in the source domain and represent the long-term dependency.

This verifies the effectiveness of ART in representing long-term dependency by learning to collocate and transfer.

Minimally Supervised Domain Adaptation We also evaluate ART when the number of training samples for the target domain is much fewer than that of the source domain.

For each target domain in the Amazon review dataset, we combine the training/development data of rest three domains as the source domain.

We show the results in TAB2 .

ART outperforms all the competitors by a large margin.

This verifies its effectiveness in the setting of minimally supervised domain adaptation.

We evaluate the effectiveness of ART w.r.t.

sequence labeling.

We use two typical tasks: POS tagging and NER (named entity recognition).

The goal of POS tagging is to assign part-of-speech tags to each word of the given sentence.

The goal of NER is to extract and classify the named entity in the given sentence.

Model Settings: POS tagging and NER are multi-class labeling tasks.

To adapt ART to them, we follow HRN and use a CRF layer to compute the tag distribution of each word.

We predict the tag with maximized probability for each word.

We use categorical cross entropy as the loss function.

The dimension of each LSTM cell is set to 300.

By following HRN, we use the concatenation of 50d word embeddings and 50d character embeddings as the input of the ART layer.

We use 50 1d filters for CNN char embedding, each with a width of 3.

We use the Adagrad BID5 optimizer.

We use a dropout probability of 0.5 on the max pooling layer.

We use the dataset settings as in .

For POS Tagging, we use Penn Treebank (PTB) POS tagging, and a Twitter corpus BID16 as different domains.

For NER, we use CoNLL 2003 BID19 and Twitter BID16 as different domains.

Their statistics are shown in TAB3 .

By following BID16 , we use 10% training samples of Twitter (Twitter/0.1), 1% training samples of Twitter (Twitter/0.01), and 1% training samples of CoNLL (CoNLL/0.01) as the training data for the target domains to simulate a low-resource setting.

Note that the label space of these tasks are different.

So some baselines (e.g. LSTM-u, LSTM-s) cannot be applied.

ART aligns and transfers information from different positions in the source domain.

Intuitively, we use the alignment and attention matrices to represent cross-domain word dependencies.

So positions with stronger dependencies will be highlighted during the transfer.

We visualize the attention matrix for sentiment analysis to verify this.

We show the attention matrices for the short-term memory h and for the long-term memory c in FIG2 .

Effectiveness of the Cell-Level Transfer From figure 3, words with stronger emotions have more attentions.

For example, the word "pleased" for the long-term memory and "easy to move" for the short-term memory have strong attention for almost all words, which sits well with the intuition of cell-level transfer.

The target domain surely wants to accept more information from the meaningful words in the source domain, not from the whole sentence.

Notice that c and h have different attentions.

Thus the two attention matrices represent discriminative features.

Effectiveness in Representing the Long-Term Dependency We found that the attention reflects the long-term dependency.

For example, in FIG2 , although all words in the target domain are affected by the word "easy", the word "very" has highest attention.

This makes sense because "very" is actually the adverb for "easy", although they are not adjacent.

ART highlights cross-domain word dependencies and therefore gives a more precise understanding of each word.

Neural network-based transfer learning The layer-wise transfer learning approaches BID7 BID0 BID25 represent the input sequence by a non-sequential vector.

These approaches cannot be applied to seq2seq or sequence labeling tasks.

To tackle this problem, algorithms must transfer cell-level information in the neural network .

Some approaches use RNN to represent cell-level information.

trains the RNN layer by domain-independent auxiliary labels.

BID26 trains the RNN layer with pivots.

However, the semantics of a word can depend on its collocated words.

These approaches cannot represent the collocated words.

In contrast, ART successfully represents the collocations by attention.

Pre-trained models ART uses a pre-trained model from the source domain, and fine-tunes the model with additional layers for the target domain.

Recently, pre-trained models with additional layers are shown to be effectiveness for many downstream models (e.g. BERT BID4 , ELMo (Peters et al., 2018) ).

As a pre-trained model, ELMo uses bidirectional LSTMs to generate contextual features.

Instead, ART uses attention mechanism in RNN that each cell in the target domain directly access information of all cells in the source domain.

ART and these pre-trained models have different goals.

ART aims at transfer learning for one task in different domains, while BERT and ELMo focus on learning general word representations or sentence representations.

In this paper, we study the problem of transfer learning for sequences.

We proposed the ART model to collocate and transfer cell-level information.

ART has three advantages: (1) it transfers more fine-grained cell-level information, and thus can be adapted to seq2seq or sequence labeling tasks; (2) it aligns and transfers a set of collocated words in the source sentence to represent the cross domain long-term dependency; (3) it is general and can be applied to different tasks.

Besides, ART verified the effectiveness of pre-training models with the limited but relevant training corpus.

<|TLDR|>

@highlight

Transfer learning for sequence via learning to align cell-level information across domains.

@highlight

The paper proposed to use RNN/LSTM with collocation alignment as a representation learning method for transfer learning/domain adaptation in NLP.