This paper proposes Metagross (Meta Gated Recursive Controller), a new neural sequence modeling unit.

Our proposed unit is characterized by recursive parameterization of its gating functions, i.e., gating mechanisms of Metagross are controlled by instances of itself, which are repeatedly called in a recursive fashion.

This can be interpreted as a form of meta-gating and recursively parameterizing a recurrent model.

We postulate that our proposed inductive bias provides modeling benefits pertaining to learning with inherently hierarchically-structured sequence data (e.g., language, logical or music tasks).

To this end, we conduct extensive experiments on recursive logic tasks (sorting, tree traversal, logical inference), sequential pixel-by-pixel classification, semantic parsing, code generation, machine translation and polyphonic music modeling, demonstrating the widespread utility of the proposed approach, i.e., achieving state-of-the-art (or close) performance on all tasks.

Sequences are fundamentally native to the world we live in, i.e., language, logic, music and time are all well expressed in sequential form.

To this end, the design of effective and powerful sequential inductive biases has far-reaching benefits across many applications.

Across many of these domains, e.g., natural language processing or speech, the sequence encoder lives at the heart of many powerful state-of-the-art model architectures.

Models based on the notion of recurrence have enjoyed pervasive impact across many applications.

In particular, the best recurrent models operate with gating functions that not only ameliorate vanishing gradient issues but also enjoy fine-grain control over temporal compositionality (Hochreiter & Schmidhuber, 1997; .

Specifically, these gating functions are typically static and trained via an alternate transformation over the original input.

In this paper, we propose a new sequence model that recursively parameterizes the recurrent unit.

More concretely, the gating functions of our model are now parameterized repeatedly by instances of itself which imbues our model with the ability to reason deeply 1 and recursively about certain inputs.

To achieve the latter, we propose a soft dynamic recursion mechanism, which softly learns the depth of recursive parameterization at a per-token basis.

Our formulation can be interpreted as a form of meta-gating since temporal compositionality is now being meta-controlled at various levels of abstractions.

Our proposed method, Meta Gated Recursive Controller Units (METAGROSS), marries the benefits of recursive reasoning with recurrent models.

Notably, we postulate that this formulation brings about benefits pertaining to modeling data that is instrinsically hierarchical (recursive) in nature, e.g., natural language, music and logic, an increasingly prosperous and emerging area of research (Shen et al., 2018; Wang et al., 2019; Choi et al., 2018) .

While the notion of recursive neural networks is not new, our work is neither concerned with syntax-guided composition (Tai et al., 2015; Socher et al., 2013; nor unsupervised grammar induction (Shen et al., 2017; Choi et al., 2018; Havrylov et al., 2019; Yogatama et al., 2016) .

Instead, our work is a propulsion on a different frontier, i.e., learning recursively parameterized models which bears a totally different meaning.

Overall, the key contributions of this work are as follows:

??? We propose a new sequence model.

Our model is distinctly characterized by recursive parameterization of recurrent gates, i.e., compositional flow is controlled by instances of itself,?? la repeatedly and recursively.

We propose a soft dynamic recursion mechanism that dynamically and softly learns the recursive depth of the model at a token-level.

??? We propose a non-autoregressive parallel variation of METAGROSS,that when equipped with the standard Transformer model (Vaswani et al., 2017) , leads to gains in performance.

??? We evaluate our proposed method on a potpourri of sequence modeling tasks, i.e., logical recursive tasks (sorting, tree traversal, logical inference), pixel-wise sequential image classification, semantic parsing, neural machine translation and polyphonic music modeling.

METAGROSS achieves state-of-the-art performance (or close) on all tasks.

This section introduces our proposed model.

METAGROSS is fundamentally a recurrent model.

Our proposed model accepts a sequence of vectors X ??? R ??d as input.

The main unit of the Metagross unit h t = Metagross n (x t , h t???1 ) is defined as follows:

where ?? r is a nonlinear activation such as tanh.

?? s is the sigmoid activation function.

In a nutshell, the Metagross unit recursively calls itself until a max depth L is hit.

When n = L, f t and o t are parameterized by:

is the forget and output gate of METAGROSS at time step t while at the maximum depth L. We also include an optional residual connection h n t = h t + x t to facilitate gradient flow down the recursive parameterization of METAGROSS.

We propose learning the depth of recursion in a data-driven fashion.

To learn ?? t , ?? t , we use the following:

where F * (xt) = W x t + b is a simple linear transformation layer applied to sequence X across the temporal dimension.

Intuitively, ??, ?? control the extent of recursion, enabling a soft depth pertaining to the hierarchical parameterization.

Alternatively, we may also consider a static variation where:

where the same value of ??, ?? is computed based on global information from the entire sequence.

Note that this strictly cannot be used for autoregressive decoding.

Finally, we note that it is also possible to assign ?? ??? R, ?? ??? R to be trainable scalar parameters.

Intuitively, F * n ??? * ??? F, O, Z are level-wise parameters of METAGROSS.

We parameterize F n with either level-wise RNN units or simple linear transformations.

We postulate that METAGROSS can also be useful as a non-autoregressive parallel model.

This can be interpreted as a form of recursive feed-forward layer that is used in place of recurrent META-GROSS for speed benefits.

In early experiments, we find this a useful enhancement to state-of-theart Transformer (Vaswani et al., 2017 ) models.

The non-autoregressive variant of METAGROSS is written as follows:

More concretely, we dispense with the reliance on the previous hidden state.

This can be used in place of any position-wise feed-forward layer.

In this case, note that F * n (x t ) are typically positionwise functions as well.

We conduct experiments on a suite of diagnostic synthetic tasks and real world tasks.

We evaluate our model on three diagnostic logical tasks as follows:

??? Task 1 (SORT SEQUENCES) -The input to the model is a sequence of integers.

The correct output is the sorted sequence of integers.

Since mapping sorted inputs to outputs can be implemented in a recursive fashion, we evaluate our model's ability to better model recursively structured sequence data.

Example input output pair would be 9, 1, 10, 5, 3 ??? 1, 3, 5, 9, 10.

??? Task 2 (TREE TRAVERSAL) -We construct a binary tree of maximum depth N .

The goal is to generate the postorder tree traversal given the inorder and preorder traversal of the tree.

Note that this is known to arrive at only one unique solution.

The constructed trees have random sparsity where we assign a probability p of growing the tree up to maximum depth N .

Hence, the trees can be of varying depths 2 .

This requires inferring hierarchical structure and long-term reasoning across sequences.

We concatenate the postorder and inorder sequences, delimiteted by a special token.

We evaluate on n ??? {3, 4, 5, 8, 10}. For n = {5, 8}, we ensure that each tree traversal has at least 10 tokens.

For n = 10, we ensure that each path has at least 15 tokens.

Example input output pair would be 13, 15, 4, 7, 5, X, 13, 4, 15, 5, 7 ??? 7, 15, 13, 4, 5.

??? Task 3 (LOGICAL INFERENCE) -We use the standard logical inference dataset 3 proposed in (Bowman et al., 2014) .

This is a classification task in which the goal is to determine the semantic equivalence of two statements expressed with logic operators such as not, and, and or.

The language vocabulary is of six words and three logic operators.

As per prior work (Shen et al., 2018) , the model is trained on sequences with 6 or less operations and evaluated on sequences of 6 to 12 operations.

For Task 1 and Task 2, we frame these tasks as a Seq2Seq (Sutskever et al., 2014) task and evaluate models on exact match accuracy and perplexity (P) metrics.

We use a standard encoder-decoder architecture with attention .

We vary the encoder module with BiLSTMs, Stacked BiLSTMs (3 layers) and Ordered Neuron LSTMs (Shen et al., 2018) .

For Task 3 (logical inference), we use the common setting in other published works.

Results on Sorting and Tree Traversal Table 1 reports our results on the Sorting and Tree Traversal task.

All models solve the task with n = 3.

However, the task gets increasingly harder with a greater maximum possible length and largely still remains a challenge for neural models today.

The relative performance of METAGROSS is on a whole better than any of the baselines, especially pertaining to perplexity.

We also found that S-BiLSTMs are always better than LSTMs on this task and Ordered LSTMs are slightly worst than vanilla BiLSTMs.

However, on sorting, ON-LSTMs are much better than standard BiLSTMs and S-BiLSTMs.

TREE TRAVERSAL SORT n = 3 n = 4 n = 5 n = 8 n = 10 n = 5 n = 10 (Shen et al., 2018) .

METAGROSS achieves state-of-the-art performance.

Table  2 reports our results on logical inference task.

We compare with mainly other published work.

METAGROSS is a strong and competitive model on this task, outperforming ON-LSTM by a wide margin (+12% on the longest nunber of operations).

Performance of our model also exceeds Tree-LSTM, which has access to ground truth syntax.

Our model achieves state-of-the-art performance on this dataset even when considering models with access to syntactic information.

We evaluate our model on its ability to model and capture long-range dependencies.

More specifically, the sequential pixel-wise image classification problem treats pixels in images as sequences.

We use the well-established pixel-wise MNIST and CIFAR-10 datasets.

We use 3 layered META-GROSS of 128 hidden units each.

Results on Pixel-wise Image Classification Table 3 reports the results of METAGROSS against other published works.

Our method achieves state-of-the-art performance on the CIFAR-10 dataset, outperforming the recent Trellis Network (Bai et al., 2018b) .

On the other hand, results on MNIST are reasonable, outperforming a wide range of other published works.

On top of that, our method has 8 times less parameters than Trellis network (Bai et al., 2018b) while achieving similar or better performance.

This ascertains that METAGROSS is a reasonably competitive long-range sequence encoder.

We run our experiments on the publicly released code 4 of (Yin & Neubig, 2018) , replacing the recurrent decoder with our METAGROSS decoder.

Hyperparameter details followed the codebase of (Yin & Neubig, 2018) (Rabinovich et al., 2017) 85.7 85.3 --ASN+Att (Rabinovich et al., 2017) 87.1 85.9 --TranX (Yin & Neubig, 2018) 88 Table 4 reports our experimental results on Semantic Parsing (GEO, ATIS, JOBS) and Code Generation (DJANGO).

We observe that TranX + METAGROSS outperforms all competitor approaches, achieving stateof-the-art performance.

More importantly, the performance gain over the base TranX method allows us to observe the ablative benefits of METAGROSS.

We conduct experiments on two IWSLT datasets which are collections derived from TED talks.

Specifically, we compare on the IWSLT 2014 German-English and IWSLT 2015 EnglishVietnamese datasets.

We compare against a suite of published results and strong baselines.

For our method, we replaced the multi-head aggregation layer in the Transformer networks (Vaswani et al., 2017 ) with a parallel non-autoregressive adaptation of METAGROSS.

The base models are all linear layers.

For our experiments, we use the standard implementation and hyperparameters in Tensor2Tensor 5 (Vaswani et al., 2018) , using the small (S) and base (B) setting for Transformers.

Model averaging is used and beam size of 8/4 and length penalty of 0.6 is adopted for De-En and En-Vi respectively.

For our model, max depth is tuned amongst {1, 2, 3}. We also ensure to compare, in an ablative fashion, our own reported runs of the base Transformer models.

Model BLEU MIXER (Ranzato et al., 2015) 21.83 AC+LL (Bahdanau et al., 2016) 28.53 NPMT 28.96 Dual Transfer 32.35 Transformer S (Vaswani et al., 2017) 32.86 Layer-wise (He et al., 2018) 35 Model BLEU (Luong & Manning, 2015) 23.30 Att-Seq2Seq 26.10 NPMT 27.69 NPMT + LM 28.07 Transformer B (Vaswani et al., 2017) 28.43 Transformer B + METAGROSS 30.81 Table 6 : Experimental results on Neural Machine Translation on IWSLT 2015 En-Vi.

We evaluate METAGROSS on the polyphonic music modeling.

We use three well-established datasets, namely Nottingham, JSB Chorales and Piano Midi (Boulanger-Lewandowski et al., 2012) .

The input to the model are 88-bit sequences, each corresponding to the 88 keys of the piano.

The task is evaluated on the Negative Log-likelihood (NLL).

We compare with a wide range of published works (Chung et al., 2014; Bai et al., 2018a; Song et al., 2019) Model Nott JSB Piano GRU (Chung et al.) 3.13 8.54 8.82 LSTM (Song et al.) 3.25 8.61 7.99 G2-LSTM (Li et al.) 3.21 8.67 8.18 B-LSTM (Song et al.) 3.16 8.30 7.55 TCN (Bai et al.) 3.07 8.10 -TCN (our run) 2.95 8.13 7.53 METAGROSS 2.88 8.12 7.49 Table 7 : Experimental Results (NLL) on Polyphonic Music Modeling.

Table 7 reports our scores on this task.

METAGROSS achieves stateof-the-art performance on the Nottingham and Piano midi datasets, outperforming a wide range of competitive models such as Gumbel Gate LSTMs (Li et al., 2018b ).

This section reports some analysis and discussion regarding the proposed model.

Table 9 : Optimal Maximum Depth N and base unit for different tasks.

Table 8 reports some ablation studies on the semantic parsing and code generation tasks.

We observe that the base unit and optimal maximum depth used is task dependent.

For ATIS dataset, using the linear transform as the base unit performs the best.

Conversely, the linear base unit performs worse than the recurrent base unit (LSTM) on the DJANGO dataset.

On a whole, we also observed this across other tasks, i.e., the base unit and maximum depth of METAGROSS is a critical choice for most tasks.

Table 9 reports the optimal max depth N and best base unit for each task.

3.6.2 ANALYSIS OF SOFT DYNAMIC RECURSION Figure 6 illustrates the depth gate values on CIFAR and MNIST datasets.

These values reflect the ?? and ?? values in METAGROSS, signifying how the parameter tree is being constructed during training.

This is reflected as L and R in the figures representing left and right gates.

Firstly, we observe that our model indeed builds data-specific parameterization of the network.

This is denoted by how METAGROSS builds different 6 trees for CIFAR and MNIST.

Secondly, we analyze the dynamic recursion depth with respect to time steps.

The key observation that all datasets have very diverse construction of recursive parameters.

The recursive gates fluctuate aggressively on CI-FAR while remaining more stable on Music modeling.

Moreover, we found that the recursive gates remain totally constant on MNIST.

This demonstrates that our model has the ability to adjust the dynamic construction adaptively and can revert to static recursion over time if necessary.

We find that compelling.

The adaptive recursive depth is made more intriguing by observing how the recursive parameterization alters on CIFAR and Music datasets.

From Figure 8 we observe that the structure of the network changes in a rhythmic fashion, in line with our intuition of musical data.

When dealing with pixel information, the tree structure changes adaptively according to the more complex information processed by the network.

The study of effective inductive biases for sequential representation learning has been a prosperous research direction.

This has spurred on research across multiple fronts, starting from gated recurrent models (Hochreiter & Schmidhuber, 1997; , convolution (Bai et al., 2018a) to the recently popular self-attention based models (Vaswani et al., 2017) .

The intrinsic hierarchical structure native to many forms of sequences have long fascinated and inspired many researchers (Socher et al., 2013; Bowman et al., 2014; .

The study of recursive networks, popularized by (Socher et al., 2013) has provided a foundation for learning syntax-guided composition in language processing research.

Along the same vein, (Tai et al., 2015) proposed Tree-LSTMs which guide LSTM composition with grammar.

Recent attempts have been made to learn this process without guidance nor syntax-based supervision (Choi et al., 2018; Shen et al., 2017; Havrylov et al., 2019; Yogatama et al., 2016) .

Ordered Neuron LSTMs (Shen et al., 2018) proposed structured gating mechanisms, imbuing the recurrent unit with a tree-structured inductive bias. (Tran et al., 2018) shows that recurrence is important for modeling hierarchical structure.

Notably, learning hierachical representations across multiple time-scales (El Hihi & Bengio, 1996; Schmidhuber, 1992; Koutnik et al., 2014; Chung et al., 2016; Hafner et al., 2017) have also demonstrated reasonable success.

Learning an abstraction and controller over a base recurrent unit is also another compelling direction.

First proposed by Fast Weights (Schmidhuber, 1992) , several recent works explore this notion.

HyperNetworks (Ha et al., 2016) learns to generate weights for another recurrent unit, i.e., a form of relaxed weight sharing.

On the other hand, RCRN (Tay et al., 2018) explicitly parameterizes the gates of a RNN unit with other RNN units.

Recent attempts to speed up the recurrent unit are also reminiscent of this particular notion (Bradbury et al., 2016; Lei et al., 2018) .

The marriage of recursive and recurrent architectures is also notable.

This direction is probably the closest relevance to our proposed method, although with vast differences. (Liu et al., 2014) proposed Recursive Recurrent Networks for machine translation which are concerned with the more traditional syntactic supervision concept of vanilla recursive nets. (Jacob et al., 2018) proposed RRNet, which learns hierarchical structures on the fly.

RR-Net proposes to learn to split or merge nodes at each time step, which makes it reminiscent of (Choi et al., 2018; Shen et al., 2018) . (AlvarezMelis & Jaakkola, 2016) proposed doubly recurrent decoders for tree-structured decoding.

The core of their method is a depth and breath-wise recurrence which is similar to our model.

However, METAGROSS is concerned with learning gating controllers which is different from the objective of decoding trees.

Our work combines the idea of external meta-controllers (Schmidhuber, 1992; Ha et al., 2016; Tay et al., 2018) with recursive architectures.

In particular, our recursive parameterization is also a form of dynamic memory which gives our model improved expressiveness in similar spirit to memoryaugmented recurrent models (Santoro et al., 2018; Graves et al., 2014; Tran et al., 2016) .

We proposed Meta Gated Recursive Controller Units (METAGROSS) a sequence model characterized by recursive parameterization of gating functions.

Our proposed method achieves very promising and competitive results on a spectrum of benchmarks across multiple modalities (e.g., language, logic, music).

We propose a non-autoregressive variation of METAGROSS, which allows simple drop-in enhancement to state-of-the-art Transformers.

We study and visualise our network as it learns a dynamic recursive parameterization, shedding light on the expressiveness and flexibility to learn dynamic parameter structures depending on the data.

<|TLDR|>

@highlight

Recursive Parameterization of Recurrent Models improve performance 