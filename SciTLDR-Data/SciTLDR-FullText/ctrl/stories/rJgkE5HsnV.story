This paper addresses the problem of incremental domain adaptation (IDA).

We assume each domain comes sequentially, and that we could only access data in the current domain.

The goal of IDA is  to build a unified model performing well on all the encountered domains.

We propose to augment a recurrent neural network (RNN) with a directly parameterized memory bank, which is retrieved by an attention mechanism at each step of RNN transition.

The memory bank provides a natural way of IDA: when adapting our model to a new domain, we progressively add new slots to the memory bank, which increases the model capacity.

We learn the new memory slots and fine-tune existing parameters by back-propagation.

Experiments show that our approach significantly outperforms naive fine-tuning and previous work on IDA, including elastic weight consolidation and the progressive neural network.

Compared with expanding hidden states, our approach is more robust for old domains, shown by both empirical and theoretical results.

Domain adaptation aims to transfer knowledge from a source domain to a target domain in a machine learning system.

This is important for neural networks, which are data-hungry and prone to overfitting.

In this paper, we focus on incremental domain adaptation (IDA), where we assume different domains come one after another.

We only have access to the data in the current domain, but hope to build a unified model that performs well on all the domains that we have encountered (Xu et al., 2014; Rusu et al., 2016; Kirkpatrick et al., 2017) .Incremental domain adaptation is useful in various scenarios.

Suppose a company is doing business with different partners over a long period of time.

The company can only access the data of the partner with a current contract.

However, the machine learning model is the company's property (if complying with the contract).

Therefore, it is desired to preserve as much knowledge as possible in the model and not to rely on the availability of the data.

Another important application of IDA is a quick adaptation to new domains.

If the environment of a deployed machine learning system changes frequently, traditional methods like jointly training all domains require the learning machine to be re-trained from scratch every time.

Fine-tuning a neural network by a few steps of gradient updates does transfer quickly, but it suffers from the catastrophic forgetting problem (Kirkpatrick et al., 2017) .

Suppose we do not know the domain of a data point when predicting; the (single) finetuned model cannot predict well for samples in previous domains, as it tends to "forget" quickly during fine-tuning.

A recent trend of domain adaptation in the deep learning regime is the progressive neural network (Rusu et al., 2016) , which progressively grows the network capacity if a new domain comes.

Typically, this is done by enlarging the model with new hidden states and a new predictor ( FIG0 ).

To avoid interfering with existing knowledge, the newly added hidden states are not fed back to the previously trained states.

During training, they fix all existing parameters, and only train the newly added ones.

For inference, they use the new predictor for all domains.

This is sometimes undesired as the new predictor is trained with only the last domain.

In this paper, we propose a progressive memory bank for incremental domain adaptation.

Our model augments a recurrent neural network (RNN) with a memory bank, which is a set of distributed, real-valued vectors capturing domain knowledge.

The memory is retrieved by an attention mechanism.

When our model is adapted to new domains, we progressively increase the slots in the memory bank.

But different from (Rusu et al., 2016) , we fine-tune all the parameters, including RNN and the existing memory slots.

Empirically, when the model capacity increases, the RNN does not forget much even if the entire network is fine-tuned.

Compared with expanding RNN hidden states, the newly added memory slots do not contaminate existing knowledge in RNN states, as will be shown by a theorem.

We evaluate our approach 1 on Natural Language Inference and Dialogue Response Generation.

Experiments support our hypothesis that the proposed approach adapts well to target domains without catastrophic forgetting of the source.

Our model outperforms the naïve fine-tuning method, the original progressive neural network, as well as other IDA techniques including elastic weight consolidation (EWC) (Kirkpatrick et al., 2017) .Detailed related work is provided in Appendix A.

Our model is based on an RNN.

At each time step, the RNN takes the embedding of the current word as input, and changes its states accordingly.

This can be represented by DISPLAYFORM0 where h i and h i−1 are the hidden states at time steps i and i − 1, respectively.

x i is the input at the ith step.

Typically, long short term memory (LSTM) (Hochreiter & Schmidhuber, 1997) or Gated Recurrent Units (GRU) (Cho et al., 2014) are used as RNN transitions.

In the rest of this section, we will describe a memory augmented RNN, and how it is used for incremental domain adaptation (IDA).

We enhance the RNN with an external memory bank, as shown in FIG0 .

The memory bank augments the overall model capacity by storing additional parameters in memory slots.

At each time step, our model computes an attention probability to retrieve memory content, which is then fed to the computation of RNN transition.

Particularly, we adopt a key-value memory bank, inspired by Miller et al. (2016) .

Each memory slot contains a key vector and a value vector.

The former is used to compute the attention weight for memory retrieval, whereas the latter is the value of memory content.

For the ith step, the memory mechanism computes an attention probability α i by DISPLAYFORM0 where m (key) j is the key vector of the jth slot of the memory (among N slots in total).

Then the model retrieves memory content by a weighted sum of all memory values, where the weight is the attention probability, given by DISPLAYFORM1 is the value vector of the jth memory slot.

We call c i the memory content.

Then, c i is concatenated with the current word x i , and fed to the RNN at step i to compute RNN state transition.

The memory bank in our model captures distributed knowledge; this is different from other work where memory slots correspond to specific entities (Eric et al., 2017) .

The attention mechanism enables us to train both memory content and its retrieval end-to-end, along with other parameters.

The memory bank in Subsection 2.1 can be progressively expanded to adapt a model in a source domain to new domains.

This is done by adding new memory slots to the bank which are learned exclusively from the target data.

Suppose the memory bank is expanded with another M slots in a new domain, in addition to previous N slots.

We then have N + M slots in total.

The model computes attention probability over the expanded memory and obtains the attention vector in the same way as Equations (2) - (4) , except that the summation is computed from 1 to N + M .

To initialize the expanded model, we load all previous parameters, including RNN weights and the learned N slots, but randomly initialize the progressively expanded M slots.

During training, we update all parameters by gradient descent.

The process is applied whenever a new domain comes, as shown in Algorithm 1 in Appendix A.We would like to discuss the following issues.

DISPLAYFORM0 where h

We evaluate our approach on natural language inference.

This is a classification task to determine the relationship between two sentences, the target labels being entailment, contradiction, and neutral.

we train a bi-directional LSTM (BiLSTM), following the original MultiNLI paper (Williams et al., 2018) .

Our BiL-STM achieves an accuracy of 68.37 on the official MultiNLI test set, which is better than 67.51 reported in the original MultiNLI paper (Williams et al., 2018) using BiLSTM.

This shows that our implementation and tuning are fair for the basic BiLSTM, and that our model is ready for the study of IDA.

The details of network architecture, training and hyper-parameter tuning are given in Appendix C.

We want to compare our approach with a large number of baselines and variants, and thus choose two domains as a testbed: Fic as the source and Gov as the target.

We show results in Table 1 .First, we analyze the performance of RNN and the memoryaugmented RNN (Lines 1-2 vs. Lines 3-4).

They have generally similar performance, showing that, in the non-transfer setting, the memory bank does not help the RNN much, and thus is not a typical RNN architecture in previous literature.

However, This later confirms that the performance improvement is indeed due to our IDA technique, instead of simply a better neural architecture.

We then apply two straightforward methods of domain adaptation: multi-task learning (Line 5) and fine-tuning (Line 6).

Multi-task learning jointly optimizes source and target objectives, denoted by "S+T." On the other hand, the fine-tuning approach trains the model on the source first, and then finetunes on the target.

In our experiments, these two methods perform similarly on the target domain, which is consistent with (Mou et al., 2016 performs significantly worse than multi-task learning, as it suffers from the catastrophic forgetting problem.

We notice that, in terms of source performance, the fine-tuning approach (Line 6) is slightly better than trained on the source domain only (Line 3).

This is probably because our domains are highly correlated as opposed to (Kirkpatrick et al., 2017) , and thus training with more data on target improves the performance on source.

However, fine-tuning does achieve the worst performance on source compared with other domain adaptation approaches (among Lines 5-8).

Thus, we nevertheless use the terminology "catastrophic forgetting", and our research goal is still to improve IDA performance.

The main results of our approach are Lines 7 and 8.

We see that on both source and target domains, our approach outperforms the fine-tuning method alone where the memory size is not increased (comparing Lines 7 and 6).

This verifies our conjecture that, if the model capacity is increased sufficiently, the new domain does not override the learned knowledge much in the neural network.

Our proposed approach is also "orthogonal" to the expansion of the vocabulary size, where target-specific words are randomly initialized and learned on the target domain.

As seen, this combines well with our memory expansion and yields the best performance on both source and target (Line 8).We now compare an alternative way of increasing model capacity, i.e., expanding hidden states (Lines 9 and 10).

For fair comparison, we ensure that the total number of model parameters after memory expansion is equal to the number of model parameters after hidden state expansion.

We see that the performance of hidden state expansion is poor especially on the source domain, even if we fine-tune all parameters.

This experiment provides empirical evidence to our theorem that expanding memory is more robust than expanding hidden states.

We also compare the results with previous work on IDA.

EWC (Kirkpatrick et al., 2017) does not achieve satisfactory results.

We investigate other published papers using the same method and find inconsistent results: EWC works well in some applications (Zenke et al., 2017; Lee et al., 2017) but performs poorly on others (Yoon et al., 2018; Wu et al., 2018) ; (Wen & Itti, 2018) even report near random performance with EWC.

We also re-implement the progressive neural network (Rusu et al., 2016) .

We use the target predictor to do inference for both source and target domains.

Progressive neural network (Rusu et al., 2016) also yields low performance, particularly on source, probably because the predictor is trained with only the target domain.

We measure the statistical significance of the results with one-tailed Wilcoxon's signed-rank test (Wilcoxon, 1945) .

Each method is compared with Line 8: ↑ and ⇑ denote "significantly better" with p < 0.05 and p < 0.01 respectively.

↓ and ⇓ similarly denote "significantly worse".

The absence of an arrow indicates that the performance difference compared with Line 8 is statistically insignificant with p < 0.05.

The test shows that our approach is significantly better than others, both on source and target.

Having analyzed our approach, baselines, and variants on two domains in detail, we test the performance of IDA with multiple domains, namely, Fic, Gov, Slate, Tel, and Travel.

We assume these domains come one after another, and our goal is to achieve high performance on both new and previous domains.

TAB3 shows that our approach of progressively growing memory bank achieves the same performance as fine-tuning on the last domain (both with vocabulary expansion).

But for all previous 4 domains, we achieve significantly better performance.

Our model is comparable to multi-task learning on all domains.

It also outperforms EWC and the progressive neural network in all domains; the results are consistent with Table 1 .

This provides evidence of the effectiveness for IDA with more than two domains.

It should also be mentioned that multi-task learning requires data from all domains to be available at the same time.

It is not an incremental approach for domain adaptation, and thus cannot be applied to the scenarios introduced in Section 1.

We include this setting mainly because we are curious about the performance of non-incremental domain adaptation.

In this paper, we propose a progressive memory network for incremental domain adaptation (IDA).

We augment an RNN with an attention-based memory bank.

During IDA, we add new slots to the memory bank and tune all parameters by back-propagation.

Empirically, the progressive memory network does not suffer from the catastrophic forgetting problem as in naïve fine-tuning.

Our intuition is that the new memory slots increase the neural network's model capacity, and thus, the new knowledge less overrides the existing network.

Compared with expanding hidden states, our progressive memory bank provides a more robust way of increasing model capacity, shown by both a theorem and experiments.

We also outperform previous work for IDA, including elastic weight consolidation (EWC) and the original progressive neural network.

Bayer, J., Osendorfer, C., Korhammer, D., Chen, N., Urban, S., and van der Smagt, P. On fast dropout and its applicability to recurrent networks.

Rusu et al. FORMULA0 propose a progressive neural network that progressively increases the number of hidden states ( FIG0 ).

To avoid overriding existing information, they propose to fix the weights of the learned network, and do not feed new states to old ones.

This results in multiple predictors, requiring that a data sample is labeled with its domain during the test time.

Should different domains be highly correlated to each other, the predictor of a previous domain cannot make use of new data to improve performance.

If we otherwise use the last predictor to predict samples from all domains, its performance may be low for previous domains, as the predictor is only trained with the last domain.

Yoon et al. (2018) propose an extension of the progressive network.

They identify which existing hidden units are relevant for the new task (with their sparse penalty), and finetune only the corresponding subnetwork.

However, sparsity is not common for RNNs in NLP applications, as sparse recurrent connections are harmful.

A similar phenomenon is that dropout of recurrent connections is harmful (Bayer et al., 2013) .

Our work is related to memory-based neural networks.

Sukhbaatar et al. (2015) propose an end-to-end memory network that assigns a slot for an entity, and aggregates information by multiple attention-based layers.

In their work, they design the architecture for bAbI question answering, and assign a memory slot for each sentence.

Such idea can be extended to various scenarios, for example, assigning slots to external knowledge for question answering (Das et al., 2017) and assigning slots to dialog history for a conversation system (Madotto et al., 2018) .Another type of memory in the neural network regime is the neural Turing machine (NTM) (Graves et al., 2016) .

Their memory is not directly parameterized, but is read or written by a neural controller.

Therefore, such memory serves as temporary scratch paper, but does not store knowledge itself.

In NTM, the memory information and operation are fully distributed/neuralized, as they do not correspond to the program on a true (non-neural) Turing machine.

Zhang et al. (2018b) combine the above two styles of memory for task-oriented dialog systems, where they have both slot-value memory and read-and-write memory.

Different from the above work, our memory bank stores knowledge in a distributed fashion, where each slot does not correspond to a concrete entity.

Our memory is directly parameterized, interacting in a different way from RNN weights.

Thus, it provides a natural way of incremental domain adaptation.

Our proposed IDA process is shown in Algorithm 1.

It is noted that the following theorem does not explicitly prove results for IDA, but shows that expanding memory is more stable than expanding hidden states.

This is particularly important at the beginning steps of IDA, as the progressively growing parameters are randomly initialized and are basically noise.

Although our theoretical analysis uses a restricted setting (i.e., vanilla RNN transition and linear activation), it provides the key insight that our approach is appropriate for IDA.

tions.

That is, DISPLAYFORM0 where h Proof: We first make a few assumptions.

Let h i−1 be the hidden state of the last step.

We focus on one step of transition and assume that h i−1 is the same when the model capacity is increased.

We consider a simplified case where the RNN has vanilla transition with the linear activation function.

We measure the effect of model expansion quantitatively by the expected norm of the difference on h i before and after model expansion.

Suppose the original hidden state h i is D-dimensional.

We assume each memory slot is d-dimensional, and that the additional RNN units when expanding the hidden state are also d-dimensional.

We further assume every variable in the expanded memory and expanded weights ( W in Figure 2 ) are iid with zero mean and variance σ 2 .

This assumption is reasonable as it enables a fair comparison of expanding memory and expanding hidden states.

Finally, we assume every variable in the learned memory slots, i.e., m jk , follows the same distribution (zero mean, variance σ 2 ).

This assumption may not be true after the network is trained, but is useful for proving theorems.

We compute how the original dimensions in the hidden state are changed if we expand RNN.

We denote the expanded hidden states by h i−1 and h i for the two time steps.

We denote the weights connecting from h i−1 to h i by W ∈ R D×d .

We focus on the original D-dimensional space, denoted as h (s) i .

The connection is shown in Figure 2a .

We have DISPLAYFORM1 where FORMULA0 is due to the independence and zero-mean assumptions of every element in W and h i−1 .

FORMULA0 is due to the independence assumption between W and h i−1 .Next, we compute the effect of expanding memory slots.

DISPLAYFORM2 i is the RNN hidden state after memory expansion.

∆c def = c − c, where c and c are the attention content vectors before and after memory expansion, respectively, at the current time step.

4 W (c) is the weight matrix connecting attention content to RNN states.

The connection is shown in Figure 2b .

Reusing the result of (13), we immediately obtain DISPLAYFORM3 where ∆c k is an element of the vector ∆c.

To prove Equation (2) , it remains to show that Var(∆c k ) ≤ σ 2 .

We now analyze how attention is computed.

Let α 1 , · · · , α N +M be the unnormalized attention weights over the N + M memory slots.

We notice that α 1 , · · · , α N remain the same after memory expansion.

Then, the original attention probability is given by α j = α j /( α 1 + · · · + α N ) for j = 1, · · · , N .

After memory expansion, the attention probability becomes α j = α j /( α 1 + · · · + α N +M ), illustrated in FIG7 .

We have 4 We omit the time step in the notation for simplicity.

Unnormalized measure DISPLAYFORM0 where DISPLAYFORM1 By our assumption of total attention DISPLAYFORM2 Then, we have DISPLAYFORM3 Here, (30) is due to the assumption that m jk is independent and zero-mean, and (31) is due to the independence assumption between β j and m jk .

To obtain (35), we notice that DISPLAYFORM4 2 ≤ 1, concluding our proof.

Note:

In the theorem (and in experiments), memory expansion and hidden state expansion are done such that the total number of model parameters remain the same.

The condition DISPLAYFORM5 α i,j in our theorem requires that the total attention to existing memory slots is larger than to the progressively added slots.

This is a reasonable assumption because: (1) During training, attention is trained in an ad hoc fashion to align information, and thus some of α i,j for 1 ≤ j ≤ N might be learned so that it is larger than a random memory slot; and (2) For a new domain, we do not add a huge number of slots, and thus N +M j=N +1 α i,j will not dominate.

We follow the original MultiNLI paper (Williams et al., 2018) to choose the base model and most of the settings: 300D RNN hidden states, 300D pretrained GloVe embeddings (Pennington et al., 2014) for initialization, batch size of 32, and the Adam optimizer for training.

The initial learning rate for Adam is tuned over the set {0.3, 0.03, 0.003, 0.0003, 0.00003}. It is set to 0.0003 based on validation performance.

For the memory part, we set each slot to be 300D, which is the same as the RNN and embedding size.

We tune the number of progressive memory slots in FIG9 , which shows the validation performance on the source (Fic) and target (Gov) domains.

We see that the performance is close to fine-tuning alone if only one memory slot is added.

It improves quickly between 1 and 200 slots, and tapers off around 500.

We thus choose to add 500 slots for each domain.

Table 3 shows the dynamics of IDA with our progressive memory network.

Comparing the upper-triangular values (in gray, showing out-of-domain performance) with diagonal values, we see that our approach can be quickly adapted to the new domain in an incremental fashion.

Comparing lower-triangular values with the diagonal, we see that our approach does not suffer from the catastrophic forgetting problem as the performance of previous domains is gradually increasing if trained with more domains.

After all data are observed, our model achieves the best performance in most domains (last row in Table 3 ), despite the incremental nature of our approach.

We evaluate our approach on the task of dialogue response generation.

Given an input text sequence, the task is to generate an appropriate output text sequence as a response in human-computer dialogue.

For the target domain, we manually construct a very small dataset to mimic the scenario where quick adaptation has to be done to a new domain with little training data.

In particular, we choose a random subset of 15k messageresponse pairs from the Ubuntu Dialog Corpus (Lowe et al., 2015) , a dataset of conversations about Ubuntu.

We use a 9k-3k-3k data split.

The base model is a sequence-to-sequence (Seq2Seq) neural network (Sutskever et al., 2014) Following previous work, we use BLEU-2 (Eric et al., 2017; Madotto et al., 2018) and average Word2Vec embedding similarity (W2V-Sim) (Serban et al., 2017; Zhang et al., 2018a) as the evaluation metrics.

BLEU-2 is the geometric mean of unigram and bigram word precision penalized by length, and correlates with human satisfaction to some extent (Liu et al., 2016) .

W2V-Sim is defined as the cosine similarity between the averaged Word2Vec embeddings of the model outputs and the ground truths.

Intuitively, BLEU measures hard word-level overlap between two sequences, whereas W2V-Sim measures soft similarity in a distributed semantic space.

The results for dialogue response generation are shown in Table 4.

We see that BLEU-2 and W2V similarity are not necessarily consistent.

For example, the memory-augmented RNN trained solely on source achieves the best source BLEU-2, whereas the proposed progressive memory has the highest W2V cosine similarity on S. However, our model variants (either expanding the vocabulary or not) achieve the best performance on most metrics (Lines 7 and 8).

Moreover, it consistently outperforms all other IDA approaches.

Following Experiment I, we conduct statistical test compared with Line 8.

The test shows that our method is significantly better than the other IDA methods.

you should be able to install the grub cd at the drive TAB9 .

Sample outputs of our IDA model S→T (F+M+V) from TAB9 .

In general, the evaluation of dialogue systems is noisy due to the lack of appropriate metrics (Liu et al., 2016) .

Nevertheless, our experiment provides additional evidence of the effectiveness of our approach.

It also highlights our model's viability for both classification and generation tasks.

<|TLDR|>

@highlight

We present a neural memory-based architecture for incremental domain adaptation, and provide theoretical and empirical results.