In natural language inference, the semantics of some words do not affect the inference.

Such information is considered superficial and brings overfitting.

How can we represent and discard such superficial information?

In this paper, we use first order logic (FOL) - a classic technique from meaning representation language – to explain what information is superficial for a given sentence pair.

Such explanation also suggests two inductive biases according to its properties.

We proposed a neural network-based approach that utilizes the two inductive biases.

We obtain substantial improvements over extensive experiments.

In natural language inference (Bowman et al., 2015) , the semantics of some words do not affect the inference.

In figure 1a , if we discard the semantics of some words (e.g. Avatar, fun, adults, children) from s 1 and s 2 , we obtain s 1 and s 2 , respectively.

Without figuring out the specific meaning of these words, one can still infer that they are contradictory.

In this case, the semantics of Avatar, fun, adults, and children are superficial for the inference.

Such superficial information brings overfitting to models.

Recent studies already noticed that superficial information will hurt the generalization of the model (Jia and Liang, 2017) , especially in unseen domains .

Without distinguishing the superficial semantics, an NLI model can learn to predict contradiction for sentence pairs with "children" or "adults" by example 1 in Figure 1a .

On the other hand, if we discard the superficial information during inference, we can prevent such overfitting.

s 1 :

Avatar is fun for children, not adults.

s 2 :

Avatar is fun for adults, not children.

Label: contradiction After discarding Avatar, fun, adults, children : s 1 : A is B for C, not D.

s 2 : A is B for D, not C. Label: contradiction After discarding Avatar, fun, adults, children and their correspondence information: s 1 : − is − for −, not −.

s 2 : − is − for −, not −. Label: unknown (a) s 3 : Avatar is fun for all people.

s 4 : Avatar is fun for adults only.

Common sense: People include adults and children.

Label: contradiction (b) Figure 1: Examples.

Some approaches have been proposed to reduce such overfitting.

HEX identifies the superficial information by projecting the textural information out.

HEX defines the textural information w.r.t.

the background of images for image classification, which cannot be generalized to other tasks (e.g. NLP).

For NLP, the attention mechanism (Bahdanau et al., 2015) is able to discard some words by assigning them low attention scores.

But such mechanism is more about the semantic similarity or relatedness of the words, not the superficial semantics.

In example 1 of figure 1, the two Avatar in the two sentences will have a high attention score, since their similarity is 1 (Vaswani et al., 2017) .

But we have shown that these words are superficial for inference.

So previous approaches cannot be applied to modeling the superficial information in natural language inference.

On top of that, a more critical issue is the lack of mathematical definition of such superficial information in previous studies.

Why do people think the semantics of adults and children are superficial?

In this paper, we tackle this question via the toolkit of first-order logic (FOL).

FOL is a classic technique of meaning representation language, which provides a sound computational basis for the inference.

We explain such superficial information from the perspective of FOL.

Furthermore, such explanation suggests two inductive biases, which are used to design our NLI model.

By representing natural language sentences by FOL, the sentence pair and its FOLs are logically equivalent.

The conversion of figure 1a is shown in figure 2a .

The entailment (resp.

contradiction) between s 1 and s 2 is equivalent to F OL(s 1 ) |= F OL(s 2 ) (resp.

F OL(s 1 ) |= ¬F OL(s 2 )).

Thus we successfully convert the problem of identifying superficial information in NLI to identifying the superficial information in FOL inference.

The superficial information exists in the non-logical symbols in FOL.

From the specification of the FOL representation (Russell and Norvig, 1995) , the symbols of FOL include the logical symbols and non-logical symbols.

In figure 1a , the contradiction remains if we discard the semantics of Avatar, fun, adults, children, which are non-logical symbols.

We can surely change these non-logical symbols to new symbols without changing the results of F OL(s 1 ) |= F OL(s 2 ) or F OL(s 1 ) |= ¬F OL(s 2 ).

However, there is a big gap between the FOL representation and the natural language: people use common sense when understanding the natural language.

For example, people are able to infer the contradiction between s 3 and s 4 in figure 1b, because they have the common sense that people include adults and children.

The FOLs of s 3 , s 4 and the common sense are shown in figure 2b.

With the common sense, the contradiction between s 3 and s 4 is equivalent to CS ∧ F OL(s 3 ) |= ¬F OL(s 4 ), where CS denotes the FOL of the common sense.

With the common sense, some non-logical symbols in the two sentences are not superficial, because we need these non-logical symbols for joint inference with the common sense.

For example, in figure 2b , the non-logical symbols Adult and P eople are not superficial.

This brings the major challenge of using FOL to identify the superficial information, because the common sense can hardly be obtained.

Since the common sense is unknown, we restrict the definition of superficial symbols.

We regard a non-logical symbol as superficial, if it is superficial for all possible common sense.

We show the necessary condition of the superficial symbols to avoid the effect of the common sense, which is unknown.

We show that the necessary condition is related to the semantical formula-variable (FV) independence (Lang et al., 2003) , which is NP-complete.

Nevertheless, the properties of the FOL suggest two inductive biases for superficial information identification: word information discard and correspondence information representation.

We propose a neural network-based approach to incorporate such two inductive biases.

We point out that we need to retain the correspondence information of the discarded words.

From the perspective of FOL, although the semantics of some non-logical symbols are independent for inference, the correspondence information still affects the inference.

More specifically, we need to represent the occurrence of one word in different positions in the sentence pair.

This is also intuitive from the perspective of natural language inference.

For example, in figure 1a, although adults and children are superficial, we need to be aware that for is followed by adults in s 1 , while for is followed by adults in s 2 .

Otherwise, as illustrated in s 1 and s 2 , we cannot infer their relation.

We summarize our contributions in this paper below:

• We proposed the problem of identifying and discarding superficial information for robust natural language inference.

We use FOL to precisely define what information is superficial.

• We analyze the superficial information from the perspective of FOL.

We show that the superficial non-logical symbols are related to the semantical formula-variable (FV) independence in reasoning.

We give two properties of the superficial information, and design neural networks to reflect the two inductive biases accordingly.

• We implement a neural network-based algorithm based on the two inductive biases.

The experimental results over extensive settings verify the effectiveness of our proposed method.

Learning Robust Natural Language Representation.

Noticing that traditional neural networks for the natural language easily fail in adversarial examples (Jia and Liang, 2017; Rajpurkar et al., 2018) , learning robust representations is important for NLP tasks.

A critical metric of the robustness is whether the model can be applied to a different data distribution .

Adversarial training (Goodfellow et al., 2014) is one way to increase the robustness for NLP models (Goodfellow et al., 2014) .

It has been applied to NLP tasks such as relation extraction (Wu et al., 2017) , sentence classification (Liu et al., 2017) .

The idea is to use adversarial training to learn a unified data distribution for different domains.

But the domain-specific information of the target domain must be known.

In contrast, we want to learn a robust model that can be applied without knowing the target domain.

And we learn robust representations by projecting superficial information out.

HEX ) is a recent approach to project textural information out of images.

It relies on two models to represent the whole semantics and superficial semantics, respectively.

Few studies reveal how to do this for NLP.

Omit Superficial Information by Attention.

The attention mechanism (Bahdanau et al., 2015) gives different weights to different words according to their attention scores.

Attention and its variations are successful in many NLP tasks (Vaswani et al., 2017; Devlin et al., 2018; Cui et al., 2019) .

Literally, attention also projects some words out by assigning them low attention scores.

However, the attention scores cannot be used to project superficial information of the overlapping words out.

Attention gives two words high attention scores if they are similar or equal, even if they are superficial.

So we cannot use attention to discard superficial information of overlapping words.

As illustrated in section 1, much superficial information for cross-sentence inference lies in these overlapping words.

Natural Language Inference uses neural networks to improve its accuracy (Bowman et al., 2016) .

Recent studies (Shen et al., 2018b ;a) apply attention mechanism (Bahdanau et al., 2015) to model the word correlations.

State-of-the-art approaches (Devlin et al., 2018; Liu et al., 2019 ) are fine-tuned over the large-scale pre-training models.

omit the syntaxes of more complicated elements of FOL (e.g. formula) since they are irrelevant to this paper.

Examples of FOLs are shown in figure 2.

Firstly, we revealed the relation between natural language inference and FOL inference.

The general purpose of NLI is to determine the contradiction, entailment, and neutral relations of two sentences.

If we convert the two sentences into two FOLs, the relation of the FOLs directly reflects the inference label of the two sentences, as shown in Table 1 .

NLI label FOL FOL with common sense entailment Table 1 : NLI labels and FOL relations.

People understand natural language with external common sense.

We show the mapping between natural language inference and FOL inference with common sense in table 1.

Obviously, the conversion from a natural language sentence to a FOL sentence is not trivial.

We highlight that our paper do not require an algorithm to implement such conversion.

We only use FOL to explain the superficial information in NLI, and to suggest inductive biases for our algorithm.

We analyze the superficial information in the entailment relation.

The other two relations (i.e. contradiction and neural) can be analyzed similarly.

Note that the entailment relation depends on the common sense, which is unknown for NLI.

So we restrict the definition of the superficial information in FOLs w.r.t.

all possible common sense.

Definition 1.

Given F OL(s 1 ), F OL(s 2 ), with non-logical symbol space V , we define a non-logical symbol ns ∈ V is superficial, if replacing ns to with ns (s.t.

ns ∈ V ) in F OL(s 1 ), F OL(s 2 ) satisfies that ∀CS,

is equivalent to

, where F OL (s 1 ), F OL (s 2 ) are the FOLs after the replacement.

Since CS can have arbitrary sentences, analyzing the superficial symbols with CS is challenging.

We first derive a necessary condition in theorem 1 to avoid the effect of CS.

Theorem 1.

Given F OL(s 1 ), F OL(s 2 ), a non-logical symbol ns is superficial, only if

is equivalent to

Theorem 1 provides a necessary condition for identifying superficial non-logical symbols that only considers F OL(s 1 ) and F OL(s 2 ).

Thus it is feasible to address whether the necessary condition is true by only using F OL(s 1 ) and F OL(s 2 ).

The condition in theorem1 is similar to the semantic FV independence problem (Lang et al., 2003) in reasoning, which is NP-complete (Lang et al., 2003) .

However, we can still utilize its properties to help identify the superficial information.

We show this in theorem 2.

Theorem 2.

Given two FOLs F OL A (s 1 ) and F OL A (s 2 ), with their non-logical symbol set A = {a 1 , · · · , a n }.

∀B = {b 1 , · · · , b n }, where each b i is a non-logical symbol, if we replace each a i with b i in F OL A (s 1 ) and F OL A (s 2 ) to get F OL B (s 1 ) and F OL B (s 2 ) respectively, we have

is equivalent to

.

Note that both A and B contain n distinct non-logical symbols.

Theorem 2 points out that, from the perspective of FOL, the semantics about non-logical symbols do not affect the implication of two FOLs.

Note that we need to guarantee that the n non-logical symbols in B are distinct.

We need to reserve the correspondence of these symbols to reserve their relation.

The theorem is easy to prove because uniformly modifying the non-logical symbols in two FOL does not change their implication.

The properties of superficial information in FOLs suggests what information should be discarded in natural language inference.

In this subsection, we elaborate two types of inductive biases, and how we use neural network to represent these inductive biases.

More details of the neural network are shown in section 4.4.

Word Information Discard From theorem 1, the necessary condition of a word being superficial is that it corresponds to a non-logical symbol, and F OL(s 1 ) |= F OL(s 2 ) is equivalent to F OL (s 1 ) |= F OL (s 2 ).

As we use the word embedding to represent the word information, we use a scalar α for each word to indicate how likely the word is superficial.

We multiply the word embedding by α for each word.

Note that one word in different positions should have a unique α, since we assume they correspond to the same symbol and thereby whether they are superficial are identical.

Correspondence Information Representation In theorem 2, although we can replace each symbol to a new symbol, the symbols should be replaced accordingly.

So for the superficial non-logical symbols, their correspondence information affects the inference.

This can be easily illustrated from the perspective of NLI in figure 1a .

If we discard the superficial symbols but reserve their correspondence information, we will get s 1 and s 2 , from which their contradiction can be still inferred.

But if we discard both the superficial symbols and their correspondence information to get s 1 and s 2 , their relation is infeasible to infer.

In order to represent the correspondence information, we use a graph neural network which connects the same words in different positions of the word pairs.

Thus the correspondence information is able to propagate through these positions.

Architecture Our proposed neural network consists of three major modules, which is shown in figure 3 .

The first module is the superficial information projection module, which is motivated by the word information discard in section 4.3.

For each word w i , we compute its superficial factor α i , which is a scalar indicating how superficial the word is.

α i = 1 means the word corresponds to non-logical symbols that we want to keep the information during inference, or the word corresponds to a logical symbol.

α w = 0 means the word is totally useless.

The embedding of each word is multiplied by the α i .

The second module is a standard NLI model.

We can use arbitrary NLI models (e.g. ESIM Chen et al. (2017) , MwAN Tan et al. (2018) ) as this module.

The output of this model is a sequence of embeddings, indicating the states of the words.

The third module represents the correspondence information in section 4.3.

We need to keep the correspondence of the superficial symbols via a graph neural network.

Superficial information projection To discard the words with superficial information, we multiply the embedding of each word by its superficial factor α.

More specifically, the embedding of a word w i is computed by:

, where w i is in the one-hot representation, E is the embedding matrix.

Note that α i is the same for one word in different positions of the sentence pair.

To achieve this, we simply use a single perceptron layer over the embeddings to compute such α.

, where M is the parameter matrix for α, [; ] denotes the concatenation operation, t i denotes whether w i is overlapped in the sentence pair (t i = 1) or not (t i = 0).

Correspondence representation To represent the cross-sentence correspondence information, we use a graph neural network.

For the same word which occurs in different positions in the sentence pair, we use an edge between all position pairs to represent the correspondence information.

Intuitively, for words that are superficial, we only need to retain their correspondence information, and vice versa.

As α i denotes whether the information should be retained, we set the weight of the edge to 1 − α i for word w i .

More formally, we denote the states at time as S T ∈ R n×d , where n is the total length of the sentence pair and d is the dimension of the hidden states.

By following the graph neural network in Kipf and Welling (2016), we update S T by:

, where W T is the parameter matrix, S 0 is the output of the standard NLI module, and A ∈ R n×n is the adjacency matrix to represent such correspondence:

, where θ is used to make the sum of each row in A equals to 1.

Figure 3 show how we connect the words "fun", "for", and "children" in different positions in the sentence pair.

By using the edges, even if the model discards the semantics of "children", it is able to represent that the word is behind "and" in the first sentence, and behind "only" in the second sentence.

Therefore we retain the correspondence information by the graph neural network.

Datasets We use the datasets including MNLI (Williams et al., 2018) , SNLI (Bowman et al., 2015) , QNLI , DNLI (Welleck et al., 2018) , RTE (Dagan et al., 2005) , MRPC (Dolan and Brockett, 2005) , and SciTail (Khot et al., 2018) .

More details are shown in appendix D.

Competitors Since our proposed framework can use different NLI models as the second module, we use standard NLI models for both comparison and for NLI module.

These models include BiLSTM, ESIM (Chen et al., 2017) , MwAN (Tan et al., 2018) , and CAFE (Tay et al., 2018) .

We compare with HEX , which projects superficial statistics out.

We also compare with the pretraining model Elmo (Peters et al., 2018) , Roberta (Liu et al., 2019) , which achieves state-of-the-art results in NLI.

More details of the experimental setup are shown in appendix D and appendix E.

Effectiveness We evaluate the effectiveness of our proposed approaches in the single domain setting.

The training and test data are from the same domain.

Ablations We evaluate the effectiveness of the two inductive biases in section 4.3, i.e., word information discard and correspondence information representation.

We use an ablation study in Table 3 to evaluate them.

Here −word means no word discard (i.e. α only works in the correspondence representation module).

−correspond means no correspondence representation module.

From the results, both inductive biases improve the effectiveness.

The word information discard is more crucial.

Table 3 : Ablation over single domains.

State-of-the-art NLI results are from the fine-tuning of pre-training models.

We use Elmo (Peters et al., 2018) and Roberta Liu et al. (2019) , a recent pre-training model, as the word embeddings module in our architecture Liu et al. (2019) .

We use the pooling layer in ESIM for final classification.

The results are shown in Table 4 .

While our proposed method outperforms the original ESIM+ELMO by a large margin, the accuracies are slightly improved for Roberta.

This makes sense because Roberta already reached a very high accuracy.

We evaluate the robustness of our approaches in unseen domains.

We choose one dataset as the source domain for training, and another dataset as the target unseen domain for testing.

The model is only trained by the training data in the source domain.

Table 5 shows the performance of different models.

From the results, we see that by using our proposed method, the accuracy improves significantly.

We visualize the α to deeply analyze its performance in Figure 4 .

Each grid of a word represent its α.

Our approach successfully projects superficial words out.

For example, in figure 4a, the words "women" and "bar" are mostly discarded, while both words do not affect the inference.

The same intuitive discarding happens in the words "man" and "shirt" in figure 4b .

We also visualize and analyze the attention mechanism in appendix G.

In this paper, we study the problem of projecting superficial information out for NLI.

The projection prevents models from overfitting and makes them more robust.

Specially, we explain the superficial information from the perspective of FOL, and project them out in a neural network-based architecture.

We conduct extensive experiments to verify the effectiveness of our proposed approach.

The results verify that our proposed approaches increase the baselines by a large margin.

Single domain In the single domain setting, the training data and the test data have the same distribution P X , y. More formally, given training data T rain = {x n , y n ∼ P X,y } N n=1 , the goal is to predict the labels of the test data T est = {x m , y m ∼ P X,y } X , y and P (t) X , y, respectively.

We evaluate the model that is trained on the source domain T rain

and is tested on the target domain T est (t) = {x

.

Note that in the unseen domain NLI, P

X , y and T est (t) are unknown during training.

This setting is more challenging than traditional domainadaptation (Ajakan et al., 2014; Cui et al., 2019) and domain generalization (Muandet et al., 2013) from the perspective that the test domain is unknown during training.

Syntax of the atoms in FOL Symbol type Table 6 : The syntax of FOL, specified in Backus-Naur form (Russell and Norvig, 1995) .

Proof.

For a non-logical symbol ns, since ∀CS,

is equivalent to

For CS = T rue, CS ∧ F OL(s 1 ) = F OL(s 1 ), CS ∧ F OL (s 1 ) = F OL (s 1 ).

Thus for a nonlogical symbol ns, we have

is equivalent to

D EXPERIMENTAL SETTINGS AND DATASETS All the experiments run over a computer with Intel Core i7 4.0GHz CPU, 32GB RAM, and a GeForce GTX 1080 Ti GPU.

For SNLI, we remove the "the other" category to make its labels comparable with MNLI.

We evaluate the accuracy and f1-score for MRPC, since its labels are imbalanced.

We list the statistics of the datasets in Table 7 .

We use a pooling layer over our proposed architecture for sentence pair classification.

For ESIM, MwAN and CAFE, we use the pooling layer in their original papers.

For BiLSTM, we follow .

We use a max pooling layer to produce the vectors u, v of each sentence, and pass [u; v; |u − v|; u * v] to an MLP classifier which has a hidden layer with tanh activation.

We apply sof tmax over the output layer.

Hex relies on a textural model to generate superficial information, and a raw model to generate all information.

We use a two-layer BiLSTM over the overlapping words to generate the superficial information.

And we use another BiLSTM over the raw sentence to generate all information.

Hyper-parameters For BiLSTM, the dimension of the hidden states is set to 300.

For other models, we use the dimension of the hidden states as their original papers.

For ESIM, CAFE, and MwAN, their dimensions are set to 300, 300, 75, respectively.

We use the AMSGrad (Reddi et al., 2018) optimizer except Roberta, in which we use AdamW (Loshchilov and Hutter, 2019) .

We use 300d GloVe vectors (Pennington et al., 2014) as the initialization for the word embedding except Roberta.

For the T -layer graph neural network, to achieve the best performance, we set T = 3 for BiLSTM, ESIM and Roberta, T = 2 for MwAN, and T = 1 for CAFE.

We show the ablations over unseen domains in Figure 5 show the attention matrix w/o discarding superficial information in ESIM.

Clearly, the attention matrix after discarding superficial information is more intuitive.

It concentrates on "usual" and "slightly lower", which imply the contradiction relation.

In contrast, the matrix of the standard ESIM focus on the repeated words (e.g. "Toronto", "stock"), which are not critical for the inference.

@highlight

We use neural networks to project superficial information out for natural language inference by defining and identifying the superficial information from the perspective of first-order logic.

@highlight

This paper tries to reduce superficial information in natural language inference to prevent overfitting, and introduces a graph neural network to model relation between premise and hypothesis. 

@highlight

An approach to treat natural language inference using first-order logic and to infuse NLI models with logical information to be more robust at inference.