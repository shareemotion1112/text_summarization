We investigate the extent to which individual attention heads in pretrained transformer language models, such as BERT and RoBERTa, implicitly capture syntactic dependency relations.

We employ two methods—taking the maximum attention weight and computing the maximum spanning tree—to extract implicit dependency relations from the attention weights of each layer/head, and compare them to the ground-truth Universal Dependency (UD) trees.

We show that, for some UD relation types, there exist heads that can recover the dependency type significantly better than baselines on parsed English text, suggesting that some self-attention heads act as a proxy for syntactic structure.

We also analyze BERT fine-tuned on two datasets—the syntax-oriented CoLA and the semantics-oriented MNLI—to investigate whether fine-tuning affects the patterns of their self-attention, but we do not observe substantial differences in the overall dependency relations extracted using our methods.

Our results suggest that these models have some specialist attention heads that track individual dependency types, but no generalist head that performs holistic parsing significantly better than a trivial baseline, and that analyzing attention weights directly may not reveal much of the syntactic knowledge that BERT-style models are known to learn.

Pretrained Transformer models like OpenAI GPT BID9 and BERT BID1 have shown stellar performance on language understanding tasks.

BERT and BERTbased models significantly improve the state-ofthe-art on many tasks such as constituency parsing BID5 , question answering BID11 , and have attained top positions on the GLUE leaderboard .

As BERT becomes a staple component of many NLP models, many researchers have attempted to analyze the linguistic knowledge that BERT has learned by analyzing the BERT model BID3 or training probing classifiers on the contextualized embeddings of BERT BID12 .BERT, as a Transformer-based language model, computes the hidden representation at each layer for each token by attending to all the tokens in an input sentence.

The attention heads of Transformer have been claimed to capture the syntactic structure of the sentences BID13 .

Intuitively, for a given token, some specific tokens in the sentence would be more linguistically related to it than the others, and therefore the selfattention mechanism should be expected to allocate more weight to the linguistically related tokens in computing the hidden state of the given token.

In this work, we aim to investigate the hypothesis that syntax is implicitly encoded by BERT's self-attention heads.

We use two relation extraction methods to extract dependency relations from all the self-attention heads of BERT.

We analyze the resulting dependency relations to investigate whether the attention heads of BERT implicitly track syntactic dependencies significantly better than chance, and what type of dependency relations BERT learn.

We extract the dependency relations from the self-attention heads instead of the contextualized embeddings of BERT.

In contrast to probing models, our dependency extraction methods require no further training.

Our experiments suggest that the attention heads of BERT encode most dependency relation types with substantially higher accuracy than our baselines-a randomly initialized Transformer and relative positional baselines.

Finetuning BERT on the syntax-oriented CoLA does not appear to impact the accuracy of extracted dependency relations.

However, when fine-tuned on the semantics-oriented MNLI dataset, there is a slight improvement in accuracy for longer-term clausal relations and a slight loss in accuracy for shorter-term relations.

Overall, while BERT models obtain non-trivial accuracy for some dependency types such as nsubj, obj, nmod, aux, and conj, they do not substantially outperform the trivial right-branching trees in terms of undirected unlabeled attachment scores (UUAS).

Therefore, although the attention heads of BERT reflect a small number of dependency relation types, it does not reflect the full extent of the significant amount of syntactic knowledge BERT is shown to learn by the previous probing work.

There has been substantial work so far on extracting syntactic trees from the attention heads of Transformer-based neural machine translation (NMT) models.

BID6 aggregate the attention weights across the self-attention layers and heads to form a single attention weight matrix.

Using this matrix, they propose a method to extract constituency and (undirected) dependency trees by recursively splitting and constructing the maximum spanning tree respectively.

In contrast, BID10 train Transformer-based machine translation model on different language-pairs and extract the dependency trees using the maximum spanning tree algorithm on the attention weights of the encoder for each layer and head individually.

In work concurrent with ours, BID14 focus on finding confident attention heads of the Transformer encoder based on a heuristic of the concentration of attention weights on single tokens.

They identify that these heads appear to serve three specific functions: attending to relative positions, syntactic relations, and rare words.

Prior work on the analysis of the contextualized embeddings of BERT has shown that BERT learns significant knowledge of syntax BID3 .

BID12 introduce a probingstyle method for evaluating syntactic knowledge in BERT and show that BERT encodes syntax more than semantics.

BID4 train a structural probing model that maps the hidden representations of each token to an innerproduct space that corresponds to syntax tree distance.

They show that the learned spaces of strong models such as BERT and ELMo BID7 are better able to reconstruct dependency trees compared to baselines that can encode features for training a parser but aren't capable of parsing themselves.

BERT BID1 ) is a Transformer-based masked language model pretrained on BooksCorpus BID21 and English Wikipedia that has attained stellar performance on a variety of downstream NLP tasks.

We run our experiments on the pretrained cased and uncased versions of the BERT-large model, which is a Transformer model consisting of 24 self-attention layers with 16 heads each.

For a given dataset, we feed each input sentence through BERT and capture the attention weights for each individual head and layer.

BID8 report that they achieve performance gains on the GLUE benchmark by supplementing pre-trained BERT with data-rich supervised tasks such as the Multi-Genre Natural Language Inference dataset (MNLI; BID18 .

Therefore, we also run experiments on the uncased BERT-large model fine-tuned on the Corpus of Linguistic Acceptability (CoLA; BID16 and MNLI, to investigate the impact of fine-tuning on a syntax-related task (CoLA) and a semantic-related task (MNLI) on the structure of attention weights and resultant extracted dependency relations.

We refer to these fine-tuned models as CoLA-BERT and MNLI-BERT.

As a baseline, we apply the same relation extraction methods to the BERT-large model with randomly initialized weights (which we refer to as random BERT) as the previous work has shown that randomly initialized sentence encoders perform surprisingly well on a suite of NLP tasks BID20 BID17 .

We aim to test the hypothesize that the attention heads of BERT learn syntactic relations implicitly, and that self-attention between two words encodes information about their dependency relation.

We use two methods for extracting relations from the attention weights in BERT.

Both methods operate on the weight matrix W ∈ (0, 1) T ×T for a given head at a given layer, where T is the number of tokens in the sequence, and the rows and columns correspond to the attending and attended tokens respectively (such that each row sums to 1).

We exclude [CLS] and [SEP] tokens from the attention matrices, which allows us to focus on inter-word attention.

Where the tokenization of our parsed corpus does not match the BERT tokenization, we merge the non-matching tokens until they are mutually compatible, and sum the attention weights for the corresponding columns and rows.

We then apply either of the two extraction methods to the attention matrix.

To handle the subtokens within the merged tokens, we set all subtokens except for the first to depend on the first subtoken.

This approach is largely similar to that in BID4 .

We use the English Parallel Universal Dependencies (PUD) treebank from the CoNLL 2017 shared task BID19 as gold standard for our evaluation.

We assign a relation (w i , w j ) between word w i and w j if j = argmax W [i] for each row i in attention matrix W .

Based on this simple method, we extract relations for all sentences in our evaluation datasets.

The relations extracted using this method need not form a valid tree, or even be fully connected.

The resulting edge directions may or may not match the canonical directions in a tree, so we evaluate the resulting arcs as undirected.

Maximum Spanning Tree To extract valid dependency trees from the attention weights for a given layer and head, we follow the approach of BID10 and treat the matrix of attention weight tokens as a complete weighted directed graph, with the edges pointing from the output token to each attended token.

As in Raganato and Tiedemann, we take the root of the gold dependency tree as the starting node and apply the Chu-Liu-Edmonds algorithm BID0 BID2 to compute the maximum spanning tree.

The resulting tree is a valid undirected dependency tree.

Relative position baselines Many dependency relations tend to occur in specific positions relative to the parent word.

For example, nsubj mostly occurs between a verb and the adjacent word before verb.

As an example, FIG0 shows the distribution of relative positions for four major UD rela- (2019) , we compute the most common positional offset between a parent and child word for a given dependency relation, and formulate a baseline based on that most common relative positional offset.

FIG1 and Table 1 describes the accuracy for nsubj, obj, advmod, and amod and the 10 most frequent relation types in the dataset using relations extracted based on the maximum attention weight method.

We also include advcl and csubj in Table 1 as it shows the behavior of MNLI-BERT that tends to track longer-term clasual dependencies better than BERT and CoLA-BERT.

Additionally, FIG2 shows the accuracy for nsubj, obj, advmod, and amod relations extracted based on the maximum spanning tree algorithm.

The pre-trained and fine-tuned BERT models outperform random BERT substantially for all dependency types.

They also outperform the relative position baselines for more than 75% of relation types.

They outperform all baselines by a large margin for nsubj and obj, but only slightly better for advmod and amod.

These results suggest that the self-attention weights in trained BERT models implicitly encode certain dependency relations.

Moreover, we do not observe very substantial changes in accuracy by fine-tuning on CoLA and MNLI.

However, both BERT and CoLA-BERT have similar or slightly better performance than MNLI-BERT, except for clausal dependencies such as advcl (adverbial clause modifier) and csubj (clausal subject) where MNLI-BERT outperforms BERT and CoLA-BERT by more than 5 absolute points in accuracy.

This suggests that semantic-oriented fine-tuning task encourages effective long-distance dependencies.

FIG3 describes the maximum undirected unlabeled attachment scores (UUAS) across each Table 1 : Highest accuracy for the most frequent dependency types (excluding nsubj, obj, advmod, and amod).

We include advcl and csubj although they are not among the ten most frequent relation types as MNLI-BERT outperform other models for these dependency types.

Bold marks the highest accuracy for each dependency type.

Italics marks accuracies that outperform our trivial baselines.

layer.

BERT, CoLA-BERT, and MNLI-BERT achieve significantly higher UUAS than the random BERT.

Although BERT models perform better than the right-branching baseline in most cases, the performance gap is not very large.

However, as the performance gap between BERT models and the trivial right-branching baseline is not substantial, we cannot confidently conclude that the attention heads of BERT track syntactic dependencies.

Additionally, the performance of BERT and the fine-tuned BERTs are similar, as fine-tuning on CoLA and MNLI does not have a large impact on UUAS.

In this work, we investigate whether the attention heads of BERT exhibit the implicit syntax depen- dency by extracting and analyzing the dependency relations from the attention heads of BERT at all layers.

We use two simple dependency relation extraction methods that require no additional training, and observe that there are attention heads of BERT that track more than 75% of the dependency types with higher accuracy than our baselines.

However, the hypothesis that the attention heads of BERT track the dependency syntax is not well-supported as the linguistically uninformed baselines outperform BERT on nearly 25% of the dependency types.

Additionally, BERT's performance in terms of UUAS is only slightly higher than that of the trivial right-branching trees, suggesting that the dependency syntax learned by the attention heads is trivial.

Additionally, we observe that fine-tuning on the CoLA and MNLI does not affect the pattern of self-attention, although the fine-tuned models shows different performance from BERT on the GLUE benchmark.

@highlight

Attention weights don't fully expose what BERT knows about syntax.