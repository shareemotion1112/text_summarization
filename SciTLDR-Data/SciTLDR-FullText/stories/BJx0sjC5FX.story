Recurrent neural networks (RNNs) can learn continuous vector representations of symbolic structures such as sequences and sentences; these representations often exhibit linear regularities (analogies).

Such regularities motivate our hypothesis that RNNs that show such regularities implicitly compile symbolic structures into tensor product representations (TPRs; Smolensky, 1990), which additively combine tensor products of vectors representing roles (e.g.,  sequence positions) and vectors representing fillers (e.g., particular words).

To test this hypothesis, we introduce Tensor Product Decomposition Networks (TPDNs), which use TPRs to approximate existing vector representations.

We demonstrate using synthetic data that TPDNs can successfully approximate linear and tree-based RNN autoencoder representations, suggesting that these representations exhibit interpretable compositional structure; we explore the settings that lead RNNs to induce such structure-sensitive representations.

By contrast, further TPDN experiments show that the representations of four models trained to encode naturally-occurring sentences can be largely approximated with a bag of words, with only marginal improvements from more sophisticated structures.

We conclude that TPDNs provide a powerful method for interpreting vector representations, and that standard RNNs can induce compositional sequence representations that are remarkably well approximated byTPRs; at the same time, existing training tasks for sentence representation learning may not be sufficient for inducing robust structural representations

Compositional symbolic representations are widely held to be necessary for intelligence BID8 Fodor & Pylyshyn, 1988) , particularly in the domain of language BID7 .

However, neural networks have shown great success in natural language processing despite using continuous vector representations rather than explicit symbolic structures.

How can these continuous representations yield such success in a domain traditionally believed to require symbol manipulation?One possible answer is that neural network representations implicitly encode compositional structure.

This hypothesis is supported by the spatial relationships between such vector representations, which have been argued to display geometric regularities that parallel plausible symbolic structures of the elements being represented (Mikolov et al. 2013 ; see Figure 1 ).Analogical relationships such as those in Figure 1 are special cases of linearity properties shared by several methods developed in the 1990s for designing compositional vector embeddings of symbolic structures.

The most general of these is tensor product representations (TPRs; BID22 .

Symbolic structures are first decomposed into filler-role bindings; for example, to represent the sequence [5, 2, 4] , the filler 5 may be bound to the role of first element, the filler 2 may be bound to the role of second element, and so on.

Each filler f i and -crucially -each role r i has a vector embedding; these two vectors are combined using their tensor product f i ⊗ r i , and these tensor products are summed to produce the representation of the sequence: f i ⊗ r i .

This linear combination can predict the linear relations between sequence representations illustrated in Figure 1 .

(a) (b) (c) Figure 1 : Plots of the first two principal components of (a) word embeddings BID14 , (b) digit-sequence embeddings learned by an autoencoder (Section 2), and (c) sentences (InferSent: Conneau et al. 2017) .

All demonstrate systematicity in the learned vector spaces.

In this article, we test the hypothesis that vector representations of sequences can be approximated as a sum of filler-role bindings, as in TPRs.

We introduce the Tensor Product Decomposition Network (TPDN) which takes a set of continuous vector representations to be analyzed and learns filler and role embeddings that best predict those vectors, given a particular hypothesis for the relevant set of roles (e.g., sequence indexes or structural positions in a parse tree).To derive structure-sensitive representations, in Section 2 we look at a task driven by structure, not content: autoencoding of sequences of meaningless symbols, denoted by digits.

The focus here is on sequential structure, although we also devise a version of the task that uses tree structure.

For the representations learned by these autoencoders, TPDNs find excellent approximations that are TPRs.

In Section 3, we turn to sentence-embedding models from the contemporary literature.

It is an open question how structure-sensitive these representations are; to the degree that they are structuresensitive, our hypothesis is that they can be approximated by TPRs.

Here, TPDNs find less accurate approximations, but they also show that a TPR equivalent to a bag-of-words already provides a reasonable approximation; these results suggest that these sentence representations are not robustly structure-sensitive.

We therefore return to synthetic data in Section 4, exploring which architectures and training tasks are likely to lead RNNs to induce structure-sensitive representations.

To summarize the contributions of this work, TPDNs provide a powerful method for interpreting vector representations, shedding light on hard-to-understand neural architectures.

We show that standard RNNs can induce compositional representations that are remarkably well approximated by TPRs and that the nature of these representations depends, in intrepretable ways, on the architecture and training task.

Combined with our finding that standard sentence encoders do not seem to learn robust representations of structure, these findings suggest that more structured architectures or more structure-dependent training tasks could improve the compositional capabilities of existing models.

The Tensor Product Decomposition Network (TPDN), depicted in FIG0 , learns a TPR that best approximates an existing set of vector encodings.

While TPDNs can be applied to any structured space, including embeddings of images or words, this work focuses on applying TPDNs to sequences.

The model is given a hypothesized role scheme and the dimensionalities of the filler and role embeddings.

The elements of each sequence are assumed to be the fillers in that sequence's representation; for example, if the hypothesized roles are indexes counting from the end of the sequence, then the hypothesized filler-role pairs for [5, 2, 4] would be (4:last, 2:second-to-last, 5:third-to-last).The model then learns embeddings for these fillers and roles that minimize the distance between the TPRs generated from these embeddings and the existing encodings of the sequences.

Before the comparison is performed, the tensor product (which is a matrix) is flattened into a vector, and a linear transformation M is applied (see Appendix B for an ablation study showing that this transformation, which was not a part of the original TPR proposal, is necessary).

The overall function computed by the architecture is thus M (flatten( i r i ⊗ f i )).

To establish the effectiveness of the TPDN at uncovering the structural representations used by RNNs, we first apply the TPDN to sequence-to-sequence networks trained on an autoencoding objective: they are expected to encode a sequence of digits and then decode that encoding to reproduce the same sequence ( FIG0 ).

In addition to testing the TPDN, this experiment also addresses a scientific question: do different architectures (specifically, unidirectional, bidirectional, and tree-based sequence-to-sequence models) induce different representations?

Digit sequences: The sequences consisted of the digits from 0 to 9.

We randomly generated 50,000 unique sequences with lengths ranging from 1 to 6 inclusive and averaging 5.2; these sequences were divided into 40,000 training sequences, 5,000 development sequences, and 5,000 test sequences.

Architectures:

For all sequence-to-sequence networks, we used gated recurrent units (GRUs, Cho et al. (2014) ) as the recurrent units.

We considered three encoder-decoder architectures: unidirectional, bidirectional, and tree-based.

3 The unidirectional encoders and decoders follow the setup of BID26 : the encoder is fed the input elements one at a time, left to right, updating its hidden state after each element.

The decoder then produces the output sequence using the final hidden state of the encoder as its input.

The bidirectional encoder combines left-to-right and right-toleft unidirectional encoders BID19 ; for symmetry, we also create a bidirectional decoder, which has both a left-to-right and a right-to-left unidirectional decoder whose hidden states are concatenated to form bidirectional hidden states from which output predictions are made.

Our final topology is tree-based RNNs BID17 BID24 , specifically the Tree-GRU encoder of BID6 and the tree decoder of Chen et al. (2018) .

These architectures require a tree structure as part of their input; we generated a tree for each sequence using a deterministic algorithm that groups digits based on their values (see Appendix C).

To control for initialization effects, we trained five instances of each architecture with different random initializations.

Role schemes: We consider 6 possible methods that networks might use to represent the roles of specific digits within a sequence; see FIG1 for examples of these role schemes.1.

Left-to-right: Each digit's role is its index in the sequence, counting from left to right.

2.

Right-to-left: Each digit's role is its index in the sequence, counting from right to left.

3.

Bidirectional: Each digit's role is an ordered pair containing its left-to-right index and its right-to-left index (compare human representations of spelling, Fischer-Baum et al. 2010 ).

4.

Wickelroles: Each digit's role is the digit before it and the digit after it BID35 .

5.

Tree positions: Each digit's role is its position in a tree, such as RRL (left child of right child of right child of root).

The tree structures are given by the algorithm in Appendix C.6.

Bag-of-words: All digits have the same role.

We call this a bag-of-words because it represents which digits ("words") are present and in what quantities, but ignores their positions.

We hypothesize that RNN autoencoders will learn to use role representations that parallel their architectures: left-to-right roles for a unidirectional network, bidirectional roles for a bidirectional network, and tree-position roles for a tree-based network.

Evaluation: We evaluate how well a given sequence-to-sequence network can be approximated by a TPR with a particular role scheme as follows.

First, we train a TPDN with the role scheme in question (Section 1.1).

Then, we take the original encoder/decoder network and substitute the fitted TPDN for its encoder FIG0 ).

We do not conduct any additional training upon this hybrid network; the decoder retains exactly the weights it learned in association with the original encoder, while the TPDN retains exactly the weights it learned for approximating the original encoder (including the weights on the final linear layer).

We then compute the accuracy of the resulting hybrid network; we call this metric the substitution accuracy.

High substitution accuracy indicates that the TPDN has approximated the encoder well enough for the decoder to handle the resulting vectors.

Performance of seq2seq networks: The unidirectional and tree-based architectures both performed the training task nearly perfectly, with accuracies of 0.999 and 0.989 (averaged across five runs).

Accuracy was lower (0.834) for the bidirectional architecture; this might mean that the hidden size of 60 becomes too small when divided into two 30-dimensional halves, one half for each direction.

Quality of TPDN approximation: For each of the six role schemes, we fitted a TPDN to the vectors generated by the trained encoder, and evaluated it using substitution accuracy (Section 2.1).

The results, in FIG1 , show that different architectures do use different representations to solve the task.

The tree-based autoencoder can be well-approximated using tree-position roles but not using any of the other role schemes.

By contrast, the unidirectional architecture is approximated very closely (with a substitution accuracy of over 0.99 averaged across five runs) by bidirectional roles.

Left-to-right roles are also fairly successful (accuracy = 0.87), and right-to-left roles are decidedly unsuccessful (accuracy = 0.11).

This asymmetry suggests that the unidirectional network uses mildly bidirectional roles: while it is best approximated by bidirectional roles, it strongly favors one direction over the other.

Though the model uses bidirectional roles, then, roles with the same left-to-right position (e.g. (2,3), (2,4), and (2,5)) can be collapsed without much loss of accuracy.

Finally, the bidirectional architecture is not approximated well by any of the role schemes we investigated.

It may be implementing a role scheme we did not consider, or a structure-encoding scheme other than TPR.

Alternately, it might simply not have adopted any robust method for representing sequence structure; this could explain why its accuracy on the training task was relatively low (0.83).

Will the TPDN's success with digit-sequence autoencoders extend to models trained on naturally occurring data?

We explore this question using sentence representations from four models: InferSent (Conneau et al., 2017) , a BiLSTM trained on the Stanford Natural Language Inference (SNLI) corpus BID3 ; Skip-thought (Kiros et al., 2015) , an LSTM trained to predict the sentence before or after a given sentence; the Stanford sentiment model (SST) BID25 , a tree-based recursive neural tensor network trained to predict movie review sentiment; and SPINN BID4 , a tree-based RNN trained on SNLI.

More model details are in Appendix E.

We now fit TPDNs to these four sentence encoding models.

We experiment with all of the role schemes used in Section 2 except for Wickelroles; for sentence representations, the vocabulary size |V | is so large that the Wickelrole scheme, which requires |V | 2 distinct roles, becomes intractable.

Preliminary experiments showed that the TPDN performed poorly when learning the filler embeddings from scratch, so we used pretrained word embeddings; for each model, we use the word embeddings used by that model.

We fine-tuned the embeddings with a linear transformation on top of the word embedding layer (though the embeddings themselves remain fixed).

Thus, what the model has to learn are: the role embeddings, the linear transformation to apply to the fixed filler embeddings, and the final linear transformation applied to the sum of the filler/role bindings.

We train TPDNs on the sentence embeddings that each model generates for all SNLI premise sentences BID3 .

For other training details see Appendix E. Table 1a shows the mean squared errors (MSEs) for various role schemes.

In general, the MSEs show only small differences between role schemes, except that tree-position roles do noticeably outperform other role schemes for SST.

Notably, bag-of-words roles perform nearly as well as the other role schemes, in stark contrast to the poor performance of bag-of-words roles in Section 2.

MSE is useful for comparing models but is less useful for assessing absolute performance since the exact value of this error is not very interpretable.

In the next section, we use downstream tasks for a more interpretable evaluation.

Tasks: We assess how the tensor product approximations compare to the models they approximate at four tasks that are widely accepted for evaluating sentence embeddings: (1) Stanford Sentiment Treebank (SST), rating the sentiment of movie reviews BID25 ; (2) Microsoft Research Evaluation: We use SentEval (Conneau & Kiela, 2018) to train a classifier for each task on the original encodings produced by the sentence encoding model.

We freeze this classifier and use it to classify the vectors generated by the TPDN.

We then measure what proportion of the classifier's predictions for the approximation match its predictions for the original sentence encodings.4Results:

For all tasks besides SNLI, we found no marked difference between bag-of-words roles and other role schemes (Table 2a) .

For SNLI, we did see instances where other role schemes outperformed bag-of-words (Table 2b ).

Within the SNLI results, both tree-based models (SST and SPINN) are best approximated with tree-based roles.

InferSent is better approximated with structural roles than with bag-of-words roles, but all structural role schemes perform similarly.

Finally, Skip-thought cannot be approximated well with any role scheme we considered.

It is unclear why Skip-thought has lower results than the other models.

Overall, even for SNLI, bag-of-words roles provide a fairly good approximation, with structured roles yielding rather modest improvements.

Based on these results, we hypothesize that these models' representations can be characterized as a bag-of-words representation plus some incomplete structural information that is not always encoded.

This explanation is consistent with the fact that bag-of-words roles yield a strong but imperfect approximation for the sentence embedding models.

However, this is simply a conjecture; it is possible that these models do use a robust, systematic structural representation that either involves a role scheme we did not test or that cannot be characterized as a tensor product representation at all.

We now complement the TPDN tests with sentence analogies.

By comparing pairs of minimally different sentences, analogies might illuminate representational details that are difficult to discern in individual sentences.

We construct sentence-based analogies that should hold only under certain role schemes, such as the following analogy (expressed as an equation as in Mikolov et al. 2013) : DISPLAYFORM0 A left-to-right role scheme makes (1) equivalent to (2) (f :r denotes the binding of filler f to role r): In (2), both sides reduce to now:2, so (1) holds for representations using left-to-right roles.

However, if (2) instead used right-to-left roles, it would not reduce in any clean way, so (1) would not hold.

We construct a dataset of such role-diagnostic analogies, where each analogy should only hold for certain role schemes.

For example, (1) works for left-to-right roles or bag-of-words roles, but not the other role schemes.

The analogies use a vocabulary based on Ettinger et al. (2018) to ensure plausibility of the constructed sentences.

For each analogy, we create 4 equations, one isolating each of the four terms (e.g. I see =

I see now -you know now +

you know).

We then compute the Euclidean distance between the two sides of each equation using each model's encodings.

The results are in Table 1b .

InferSent, Skip-thought, and SPINN all show results most consistent with bidirectional roles, while SST shows results most consistent with tree-based or bidirectional roles.

The bag-of-words column shows poor performance by all models, indicating that in controlled enough settings these models can be shown to have some more structured behavior even though evaluation on examples from applied tasks does not clearly bring out that structure.

These analogies thus provide independent evidence for our conclusions from the TPDN analysis: these models have a weak notion of structure, but that structure is largely drowned out by the non-structure-sensitive, bag-of-words aspects of their representations.

However, the other possible explanations mentioned above−namely, the possibilities that the models use alternate role schemes that we did not test or that they use some structural encoding other than tensor product representation−still remain.

The previous section suggested that all sentence models surveyed did not robustly encode structure and could even be approximated fairly well with a bag of words.

Motivated by this finding, we now investigate how aspects of training can encourage or discourage compositionality in learned representations.

To increase interpretability, we return to the setting (from Section 2) of operating over digit sequences.

We investigate two aspects of training: the architecture and the training task.

Teasing apart the contribution of the encoder and decoder: In Section 2, we investigated autoencoders whose encoder and decoder had the same topology (unidirectional, bidirectional, or treebased).

To test how each of the two components contributes to the learned representation, we now expand the investigation to include networks where the encoder and decoder differ.

We crossed all three encoder types with all three decoder types (nine architectures in total).

The results are in TAB6 in Appendix D. The decoder largely dictates what roles are learned: models with unidirectional decoders prefer mildly bidirectional roles, models with bidirectional decoders fail to be well-approximated by any role scheme, and models with tree-based decoders are best approximated by tree-based roles.

However, the encoder still has some effect: in the tree/uni and tree/bi models, the tree-position roles perform better than they do for the other models with the same decoders.

Though work on novel architectures often focuses on the encoder, this finding suggests that focusing on the decoder may be more fruitful for getting neural networks to learn specific types of representations.

The contribution of the training task: We next explore how the training task affects the representations that are learned.

We test four tasks, illustrated in Table 3a : autoencoding (returning the input sequence unchanged), reversal (reversing the input), sorting (returning the input digits in ascending order), and interleaving (alternating digits from the left and right edges of the input).

Table 3b gives the substitution accuracy for a TPDN trained to approximate a unidirectional encoder that was trained with a unidirectional decoder on each task.

Training task noticeably influences the learned representations.

First, though the model has learned mildly bidirectional roles favoring the left-to-right direction for autoencoding, for reversal the right-to-left direction is far preferred over left-to-right.

For interleaving, the model is approximated best with strongly bidirectional roles: that is, bidirectional roles work nearly perfectly, while neither unidirectional scheme works well.

Finally, for sorting, bag-of-words roles work nearly as well as all other schemes, suggesting that the model Table 3 : (a) Tasks used to test for the effect of task on learned roles (Section 4).

(b) Accuracy of the TPDN applied to models trained on these tasks with a unidirectional encoder and decoder.

All numbers are averages across five random initializations.

has learned to discard most structural information since sorting does not depend on structure.

These experiments suggest that RNNs only learn compositional representations when the task requires them.

This result might explain why the sentence embedding models do not seem to robustly encode structure: perhaps the training tasks for these models do not heavily rely on sentence structure (e.g. BID12 achieved high accuracy on SNLI using a model that ignores word order), such that the models learn to ignore structural information, as was the case with models trained on sorting.

There are several approaches for interpreting neural network representations.

One approach is to infer the information encoded in the representations from the system's behavior on examples targeting specific representational components, such as semantics BID13 Dasgupta et al., 2018; BID16 or syntax (Linzen et al., 2016) .

Another approach is based on probing tasks, which assess what information can be easily decoded from a vector representation BID20 BID2 Kádár et al. 2017; Ettinger et al. 2018 ; compare work in cognitive neuroscience, e.g. BID9 ).

Our method is wider-reaching than the probing task approach, or the Mikolov et al. FORMULA0 analogy approach: instead of decoding a single feature, we attempt to exhaustively decompose the vector space into a linear combination of filler-role bindings.

The TPDN's successful decomposition of sequence representations in our experiments shows that RNNs can sometimes be approximated with no nonlinearities or recurrence.

This finding is related to the conclusions of Levy et al. (2018) , who argued that LSTMs dynamically compute weighted sums of their inputs; TPRs replace the weights of the sum with the role vectors.

Levy et al. (2018) also showed that recurrence is largely unnecessary for practical applications.

BID33 report very good performance for a sequence model without recurrence; importantly, they find it necessary to incorporate sequence position embeddings, which are similar to the left-to-right roles discussed in Section 2.

Methods for interpreting neural networks using more interpretable architectures have been proposed before based on rules and automata BID10 BID34 .Our decomposition of vector representations into independent fillers and roles is related to work on separating latent variables using singular value decomposition and other factorizations BID29 BID0 .

For example, in face recognition, eigenfaces BID21 BID30 and TensorFaces (Vasilescu & Terzopoulos, 2002; BID32 use such techniques to disentangle facial features, camera angle, and lighting.

Finally, there is a large body of work on incorporating explicit symbolic representations into neural networks (for a recent review, see BID1 ; indeed, tree-shaped RNNs are an example of this approach.

While our work is orthogonal to this line of work, we note that TPRs and other filler-role representations can profitably be used as an explicit component of neural models (Koniusz et al., 2017; BID11 Huang et al., 2018; BID28 BID18 .

What kind of internal representations could allow simple sequence-to-sequence models to perform the remarkable feats they do, including tasks previously thought to require compositional, symbolic representations (e.g., translation)?

Our experiments show that, in heavily structure-sensitive tasks, sequence-to-sequence models learn representations that are extremely well approximated by tensorproduct representations (TPRs), distributed embeddings of symbol structures that enable powerful symbolic computation to be performed with neural operations BID23 .

We demonstrated this by approximating learned representations via TPRs using the proposed tensor-product decomposition network (TPDN).

Variations in architecture and task were shown to induce different types and degrees of structure-sensitivity in representations, with the decoder playing a greater role than the encoder in determining the structure of the learned representation.

TPDNs applied to mainstream sentence-embedding models reveal that unstructured bag-of-words models provide a respectable approximation; nonetheless, this experiment also provides evidence for a moderate degree of structuresensitivity.

The presence of structure-sensitivity is corroborated by targeted analogy tests motivated by the linearity of TPRs.

A limitation of the current TPDN architecture is that it requires a hypothesis about the representations to be selected in advance.

A fruitful future research direction would be to automatically explore hypotheses about the nature of the TPR encoded by a network.

Here we analyze how several aspects of the TPDN architecture contribute to our results.

For all of the experiements described in this section, we used TPDNs to approximate a sequence-to-sequence network with a unidirectional encoder and unidirectional decoder that was trained to perform the reversal task (Section 4); we chose this network because it was strongly approximated by right-toleft roles, which are relatively simple (but still non-trivial).

One area where our model diverges from traditional tensor product representations is in the presence of the final linear layer (step 5 in FIG0 ).

This layer is necessary if one wishes to have freedom to choose the dimensionality of the filler and role embeddings; without it, the dimensionality of the representations that are being approximated must factor exactly into the product of the dimensionality of the filler embeddings and the dimensionality of the role embedding (see FIG0 .

It is natural to wonder whether the only contribution of this layer is in adjusting the dimensionality or whether it serves a broader function.

TAB3 shows the results of approximating the reversal sequence-tosequence network with and without this layer; it indicates that this layer is highly necessary for the successful decomposition of learned representations.

(Tables follow all appendix text.)

Two of the parameters that must be provided to the TPDN are the dimensionality of the filler embeddings and the dimensionality of the role embeddings.

We explore the effects of these parameters in FIG7 .

For the role embeddings, substitution accuracy increases noticeably with each increase in dimensionality until the dimensionality hits 6, where accuracy plateaus.

This behavior is likely due to the fact that the reversal seq2seq network is most likely to employ right-to-left roles, which involves 6 possible roles in this setting.

A dimensionality of 6 is therefore the minimum embedding size needed to make the role vectors linearly independent; linear independence is an important property for the fidelity of a tensor product representation BID22 .

The accuracy also generally increases as filler dimensionality increases, but there is a less clear point where it plateaus for the fillers than for the roles.

The body of the paper focused on using the tensor product (f i ⊗ r i , see FIG0 ) as the operation for binding fillers to roles.

There are other conceivable binding operations.

Here we test two alternatives, both of which can be viewed as special cases of the tensor product or as related to it: circular convolution, which is used in holographic reduced representations BID15 , and elementwise product (f i r i ).

Both of these are restricted such that roles and fillers must have the same embedding dimension (N f = N r ).

We first try setting this dimension to 20, which is what was used as both the role and filler dimension in all tensor product experiments with digit sequences.

Red indicates accuracy under 1%; dark blue indicates accuracy over 80%.

The models whose substitution accuracies are displayed are all TPDNs trained to approximate a sequence-to-sequence model that was trained on the task of reversal.

We found that while these dimensions were effective for the tensor product binding operation, they were not effective for elementwise product and circular convolution TAB4 ).

When the dimension was increased to 60, however, the elementwise product performed roughly as well as as the tensor product; circular convolution now learned one of the two viable role schemes (right-to-left roles) but failed to learn the equally viable bidirectional role scheme.

Thus, our preliminary experiments suggest that these other two binding operations do show promise, but seem to require larger embedding dimensions than tensor products do.

At the same time, they still have fewer parameters overall compared to the tensor product because their final linear layers (of dimensionality N ) are much smaller than those used with a tensor product (of dimensionality N 2 ).

When inputting digit sequences to our tree-based model, the model requires a predefined tree structure for the digit sequence.

We use the following algorithm to generate this tree structure: at each timestep, combine the smallest element of the sequence (other than the last element) with its neighbor immediately to the right, and replace the pair with that neighbor.

If there is a tie for the smallest digit, choose the leftmost tied digit.

For example, the following shows step-by-step how the tree for the sequence 523719 would be generated:• 5 2 3 7 1 9• 5 2 3 7 [1 9] DISPLAYFORM0

Section 4 summarized the results of our experiments which factorially varied the training task, the encoder and the decoder.

Here we report the full results of these experiments in two tables: TAB5 shows the accuracies achieved by the sequence-to-sequence models at the various training tasks, and TAB6 shows the substitution accuracies of TPDNs applied to the trained sequence-to-sequence models for all architectures and tasks.

As much as possible, we standardized parameters across all sequence-to-sequence models that we trained on the digit-sequence tasks.

For all decoders, when computing a new hidden state, the only input to the recurrent unit is the previous hidden state (or parent hidden state, for a tree-based decoder), without using any previous outputs as inputs to the hidden state update.

This property is necessary for using a bidirectional decoder, since it would not be possible to generate the output both before and after each bidirectional decoder hidden state.

We also inform the decoder of when to stop decoding; that is, for sequential models, the decoder stops once its output is the length of the sequence, while for tree-based models we tell the model which positions in the tree are leaves.

Stopping could alternately be determined by some action of the decoder (e.g., generating an end-of-sequence symbol); for simplicity we chose the strategy outlined above instead.

For all architectures, we used a digit embedding dimensionality of 10 (chosen arbitrarily) and a hidden layer size of 60 (this hidden layer size was chosen because 60 has many integer factors, making it amenable to the dimensionality analyses in Appendix B.2).

For the bidirectional architectures, the forward and backward recurrent layers each had a hidden layer size of 30, so that their concatenated hidden layer size was 60.

For bidirectional decoders, a linear layer condensed the 60-dimensional encoding into 30 dimensions before it was passed to the forward and backward decoders.

The networks were trained using the Adam optimizer (Kingma & Ba, 2015) with the standard initial learning rate of 0.001.

We used negative log likelihood, computed over the softmax probability distributions for each output sequence element, as the loss function.

Training proceeded with a batch size of 32, with loss on the held out development set computed after every 1,000 training examples.

Training was halted when the loss on the heldout development set had not improved for any of the development loss checkpoints for a full epoch of training (i.e. 40,000 training examples).

Once training completed, the parameters from the best-performing checkpoint were reloaded and used for evaluation of the network.

When applying TPDNs to the digit-based sequence-to-sequence models, we always used 20 as both the filler embedding dimension and the role embedding dimension.

This decision was based on the experiments in Appendix B.2; we selected filler and role embedding dimensions that were safely above the cutoff needed to lead to successful decomposition.

The TPDNs were trained with the same training regimen as the sequence-to-sequence models, except that, instead of using negative log likelihood as the loss function, for the TPDNs we used mean squared error between the predicted vector representation and the actual vector representation from the original sequence-to-sequence network.

The TPDNs were given the sequences of fillers (i.e. the digits), the roles hypothesized to go with those fillers, the sequence embeddings produced by the RNN, and the dimensionalities of the filler embeddings, role embeddings, and final linear transformation.

The parameters that were updated by training were the specific values for the filler embeddings, the role embeddings, and the final linear transformation.

For all four sentence encoding models, we used publicly available and freely downloadable pretrained versions found at the following links:• we use the SPINN-PI-NT version, which is equivalent to a tree-LSTM BID27 with 300-dimensional hidden states.

For training a TPDN to approximate the sentence encoding models, the filler embedding dimensions were dictated by the size of the pretrained word embeddings; these dimensions were 300 for InferSent and SPINN, 620 for Skip-thought, and 25 for SST.

The linear transformation applied to the word embeddings did not change their size.

For role embedding dimensionality we tested all role dimensions in {1, 5, 10, 20, 40, 60}. The best-performing dimension was chosen based on preliminary experiments and used for all subsequent experiments; we thereby chose role dimensionalities of 10 for InferSent and Skip-thought, 20 for SST, and 5 for SPINN.

In general, role embedding dimensionalities of 5, 10, and 20 all performed noticeably better than 1, 40, and 60, but there was not much difference between 5, 10, and 20.The training regimen for the TPDNs on sentence models was the same as for the TPDNs trained on digit sequences.

The TPDNs were given the sequences of fillers (i.e. the words), the roles hypothesized to go with those fillers, the sequence embeddings produced by the RNN, the initial pretrained word embeddings, the dimensionalities of the linearly-transformed filler embeddings, the role embeddings, and the final linear transformation.

The parameters that were updated by training were the specific values for the role embeddings, the linear transformation that was applied to the pretrained word embeddings, and the final linear transformation.

The sentences whose encodings we trained the TPDNs to approximate were the premise sentences from the SNLI corpus BID3 .

We also tried instead using the sentences in the WikiText-2 corpus (Merity et al., 2016) but found better performance with the SNLI sentences.

This is plausibly because the shorter, simpler sentences in the SNLI corpus made it easier for the model to learn the role embeddings without distraction from the fillers.

For each TPDN trained to approximate a sentence encoder, we evaluate it on four downstream tasks: (i) Stanford Sentiment Treebank (SST), which is labeling the sentiment of movie reviews BID25 ; this task is further subdivided into SST2 (labeling the reviews as positive or negative) and SST5 (labeling the reviews on a 5-point scale, where 1 means very negative and 5 means very positive).

The metric we report for both tasks is accuracy.(ii) Microsoft Research Paraphrase Corpus (MRPC), which is labeling whether two sentences are paraphrases of each other (Dolan et al., 2004) .

For this task, we report both accuracy and F1. (iii) Semantic Textual Similarity Benchmark (STS-B), which is giving a pair of sentences a score on a scale from 0 to 5 indicating how similar the two sentences are BID5 .

For this task, we report Pearson and Spearman correlation coefficients. (iv) Stanford Natural Language Inference (SNLI), which involves labeling a pair of sentences to indicate whether the first entails the second, contradicts the second, or neither BID3 .

For this task, we report accuracy as the evaluation metric.

The first results we report for the TPDN approximations of sentence encoders is similar to the substitution accuracy used for digit encoders.

Here, we use SentEval (Conneau & Kiela, 2018) Table 9 : The proportion of times that a classifier trained on a sentence encoding model gave the same downstream-task predictions based on the original sentence encoding model and based on a TPDN approximating that model, where the TPDN uses the role schemes indicated by the column header.

For all tasks but STS-B, these numbers show the proportion of predictions that matched; chance performance is 0.5 for SST2 and MRPC, 0.2 for SST5, and 0.33 for SNLI.

For STS-B, the metric shown is the Pearson correlation between the TPDN's similarity ratings and the original model's similarity ratings; chance performance here is 0.0.

Table 10 : Downstream task performance for classifiers trained and tested on the TPDNs that were trained to approximate each of the four applied models.

The rightmost column indicates the performance of the original model (without the TPDN approximation).

@highlight

RNNs implicitly implement tensor-product representations, a principled and interpretable method for representing symbolic structures in continuous space.