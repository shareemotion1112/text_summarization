Robustness verification that aims to formally certify the prediction behavior of  neural networks has become an important tool for understanding the behavior of a given model and for obtaining safety guarantees.

However, previous methods are usually limited to relatively simple neural networks.

In this paper, we consider the robustness verification problem for Transformers.

Transformers have complex self-attention layers that pose many challenges for verification, including cross-nonlinearity and cross-position dependency, which have not been discussed in previous work.

We resolve these challenges and develop the first verification algorithm for Transformers.

The certified robustness bounds computed by our method are significantly tighter than those by naive Interval Bound Propagation.

These bounds also shed light on interpreting Transformers as they consistently reflect the importance of words in sentiment analysis.

Deep neural networks have been successfully applied to many domains.

However, a major criticism is that these black box models are difficult to analyze and their behavior is not guaranteed.

Moreover, it has been shown that the predictions of deep networks become unreliable and unstable when tested in unseen situations, e.g., in the presence of small and adversarial perturbation to the input (Szegedy et al., 2013; Goodfellow et al., 2014; Lin et al., 2019) .

Therefore, neural network verification has become an important tool for analyzing and understanding the behavior of neural networks, with applications in safety-critical applications (Katz et al., 2017; Julian et al., 2019; Lin et al., 2019) , model explanation (Shih et al., 2018) and robustness analysis (Tjeng et al., 2019; Wang et al., 2018c; Gehr et al., 2018; Wong & Kolter, 2018; Singh et al., 2018; Weng et al., 2018; Zhang et al., 2018) .

Formally, a neural network verification algorithm aims to provably characterize the prediction of a network within some input space.

For example, given a K-way classification model f : R d → R K , we can verify some linear specification (defined by a vector c) as below:

where S is a predefined input space.

For example, in the robustness verification problem that we are going to focus on in this paper, S = {x | x−x 0 p ≤ } is defined as some small p -ball around the original example x 0 , and setting up c = 1 y0 − 1 y can verify whether the logit output of class y 0 is always greater than another class y within S.

This is a nonconvex optimization problem which makes computing the exact solution challenging, and thus algorithms are recently proposed to find lower bounds of Eq. (1) in order to efficiently obtain a safety guarantee (Gehr et al., 2018; Weng et al., 2018; Zhang et al., 2018; Singh et al., 2019) .

Moreover, extension of these algorithms can be used for verifying some properties beyond robustness, such as rotation or shift invariant (Singh et al., 2019) , conservation of energy (Qin et al., 2019) and model correctness (Yang & Rinard, 2019) .

However, most of existing verification methods focus on relatively simple neural network architectures, such as feed-forward and recurrent neural networks, and cannot handle complex structures.

In this paper, we develop the first robustness verification algorithm for Transformers (Vaswani et al., 2017) with self-attention layers.

Transformers have been widely used in natural language processing (Devlin et al., 2018; Yang et al., 2019; Liu et al., 2019) and many other domains (Parmar et al., 2018; Kang & McAuley, 2018; Li et al., 2019b; Su et al., 2019; Li et al., 2019a) .

For frames under perturbation in the input sequence, we aim to compute a lower bound such that when these frames are perturbed within p -balls centered at the original frames respectively and with a radius of , the model prediction is certified to be unchanged.

To compute such a bound efficiently, we adopt the linear-relaxation framework (Weng et al., 2018; Zhang et al., 2018 ) -we recursively propagate and compute linear lower bound and upper bound for each neuron with respect to the input within perturbation set S.

We resolve several particular challenges in verifying Transformers.

First, Transformers with selfattention layers have a complicated architecture.

Unlike simpler networks, they cannot be written as multiple layers of linear transformations or element-wise operations.

Therefore, we need to propagate linear bounds differently for self-attention layers.

Second, dot products, softmax, and weighted summation in self-attention layers involve multiplication or division of two variables under perturbation, namely cross-nonlinearity, which is not present in feed-forward networks.

Ko et al. (2019) proposed a gradient descent based approach to find linear bounds, however it is inefficient and poses a computational challenge for transformer verification as self-attention is the core of transformers.

In contrast, we derive closed-form linear bounds that can be computed in O(1) complexity.

Third, neurons in each position after a self-attention layer depend on all neurons in different positions before the self-attention (namely cross-position dependency), unlike the case in recurrent neural networks where outputs depend on only the hidden features from the previous position and the current input.

Previous works (Zhang et al., 2018; Weng et al., 2018; Ko et al., 2019) have to track all such dependency and thus is costly in time and memory.

To tackle this, we introduce an efficient bound propagating process in a forward manner specially for self-attention layers, enabling the tighter backward bounding process for other layers to utilize bounds computed by the forward process.

In this way, we avoid cross-position dependency in the backward process which is relatively slower but produces tighter bounds.

Combined with the forward process, the complexity of the backward process is reduced by O(n) for input length n, while the computed bounds remain comparably tight.

Our contributions are summarized below:

• We propose an effective and efficient algorithm for verifying the robustness of Transformers with self-attention layers.

To our best knowledge, this is the first method for verifying Transformers.

• We resolve key challenges in verifying Transformers, including cross-nonlinearity and crossposition dependency.

Our bounds are significantly tighter than those by adapting Interval Bound Propagation (IBP) (Mirman et al., 2018; .

• We quantitatively and qualitatively show that the certified lower bounds consistently reflect the importance of input words in sentiment analysis, which justifies that the computed bounds are meaningful in practice.

Robustness Verification for Neural Networks.

Given an input x 0 and a small region B p (x 0 , ) := {x | x − x 0 p ≤ }, the goal of robustness verification is to verify whether the prediction of the neural network is unchanged within this region.

This problem can be mathematically formulated as Eq. (1).

If Eq. (1) can be solved optimally, then we can derive the minimum adversarial perturbation of x by conducting binary search on .

Equivalently, we obtain the maximum such that any perturbation within B p (x 0 , ) cannot change the predicted label.

Several works focus on solving Eq. (1) exactly and optimally, using mixed integer linear programming (MILP) (Tjeng et al., 2019; Dutta et al., 2018) , branch and bound (BaB) , and satisfiability modulo theory (SMT) (Ehlers, 2017; Katz et al., 2017) .

Unfortunately, due to the nonconvexity of model f , solving Eq. (1) is NP-hard even for a simple ReLU network (Katz et al., 2017) .

Therefore, we can only hope to compute a lower bound of Eq. (1) efficiently by using relaxations.

Many algorithms can be seen as using convex relaxations for non-linear activation functions (Salman et al., 2019) , including using duality (Wong & Kolter, 2018; , abstract domains (Gehr et al., 2018; Singh et al., 2018; Mirman et al., 2018; Singh et al., 2019) , layer-by-layer reachability analysis (Wang et al., 2018b; Weng et al., 2018; Zhang et al., 2018; and semi-definite relaxations (Raghunathan et al., 2018; .

Additionally, robustness verification can rely on analysis on local Lipschitz constants (Hein & Andriushchenko, 2017; Zhang et al., 2019) .

However, existing methods are mostly limited to verifying networks with relatively simple architectures, such as feed-forward networks and RNNs (Wang et al., 2018a; Akintunde et al., 2019; Ko et al., 2019) , while none of them are able to handle Transformers.

Transformers and Self-Attentive Models.

Transformers (Vaswani et al., 2017) based on the selfattention mechanism, further with pre-training on large-scale corpora, such as BERT (Devlin et al., 2018) , XLNet (Yang et al., 2019) , RoBERTa (Liu et al., 2019) , achieved state-of-the-art performance on many NLP tasks.

Self-attentive models are also useful beyond NLP, including VisualBERT for extracting features from both text and images (Li et al., 2019b; Su et al., 2019) , image transformer for image generation (Parmar et al., 2018) , acoustic models for speech recognition, sequential recommendation (Kang & McAuley, 2018) and graph embedding (Li et al., 2019a) .

The robustness of NLP models has been studied, especially many methods have been proposed to generate adversarial examples (Papernot et al., 2016; Jia & Liang, 2017; Zhao et al., 2017; Alzantot et al., 2018; Ebrahimi et al., 2018) .

In particular, Hsieh et al. (2019) showed that Transformers are more robust than LSTMs.

However, there is not much work on robustness verification for NLP models.

Ko et al. (2019) verified RNN/LSTM.

Jia et al. (2019); Huang et al. (2019) used Interval Bound Propagation (IBP) for certified robustness training of CNN and LSTM.

In this paper, we propose the first verification method for Transformers.

We aim to verify the robustness of a Transformer whose input is a sequence of frames

We take binary text classification as an running example, where x (i) is a word embedding and the model outputs a score y c (X) for each class c (c ∈ {0, 1}).

Nevertheless, our method for verifying Transformers is general and can also be applied in other applications.

0 ] correctly classified by the model, let P = {r 1 , r 2 , · · · , r t }(1 ≤ r k ≤ n) be the set of perturbed positions, where t is the number of perturbed positions.

Thus the perturbed input will belong to S :

Assuming that c is the gold class, the goal of robustness verification is to compute {min X∈S y c (X) − y 1−c (X)} := δ (X).

If δ (X) > 0, the output score of the correct class will always be larger than the incorrect one within S .

As mentioned previously, computing the exact values of δ (X) is NP-hard, and thus our goal is to efficiently compute a lower bound δ L (X) ≤ δ (X).

We obtain δ L (X) by computing the bounds of each neuron when X is perturbed within S (δ L can be regarded as a final neuron).

A Transformer layer can be decomposed into a number of sub-layers, where each sub-layer contains neurons after some operation.

These operations can be categorized into three categories: 1) linear transformations, 2) unary nonlinear functions, and 3) operations in self-attention.

Each sub-layer contains n positions in the sequence and each position contains a group of neurons.

We assume that the Transformer we verify has m sub-layers in total, and the value of the j-th neuron at the i-th position in the l-th sub-layer is Φ

vector for the specified sub-layer and position.

Specially, Φ (0,i) = x (i) taking l = 0.

We aim to compute a global lower bound f

We compute bounds from the first sub-layer to the last sub-layer.

For neurons in the l-th layer, we aim to represent their bounds as linear functions of neurons in a previous layer, the l -th layer:

where

,U are parameters of linear lower and upper bounds respectively.

Using linear bounds enables us to efficiently compute bounds with a reasonable tightness.

We initially have

Generally, we use a backward process to propagate the bounds to previous sub-layers, by substituting Φ (l ,i) with linear functions of previous neurons.

It can be recursively conducted until the input layer l = 0.

Since

is constant, we can regard the bounds as linear functions of the perturbed embeddings

, and take the global bounds for

where 1/p + 1/q = 1 with p, q ≥ 1.

These steps resemble to CROWN (Zhang et al., 2018) which is proposed to verify feed-forward networks.

We further support verifying self-attentive Transformers which are more complex than feed-forward networks.

Moreover, unlike CROWN that conducts a full backward process, we combine the backward process with a forward process (see Sec. 3.3) to reduce the computational complexity of verifying Transformers.

Linear transformations and unary nonlinear functions are basic operations in neural networks.

We show how bounds Eq. (2) at the l -th sub-layer are propagated to the (l − 1)-th layer.

Linear Transformations If the l -th sub-layer is connected with the (l − 1)-th sub-layer with a linear transformation

are parameters of the linear transformation, we propagate the bounds to the (l −1)-th layer by substituting Φ (l ,k) (X):

where "L/U " means that the equations hold for both lower bounds and upper bounds respectively.

If the l -th layer is obtained from the (l − 1)-th layer with an unary nonlinear function Φ

, to propagate linear bounds over the nonlinear function, we first bound

are parameters such that the inequation holds for all Φ (l −1,k) j (X) within its bounds computed previously.

Such linear relaxations can be done for different functions, respectively.

We provide detailed bounds for functions involved in Transformers in Appendix B.

We then back propagate the bounds:

mean to retain positive and negative elements in vector

respectively and set other elements to 0.

Self-attention layers are the most challenging parts for verifying Transformers.

We assume that Φ (l−1,i) (X) is the input to a self-attention layer.

We describe our method for computing bounds for one attention head, and bounds for different heads of the multi-head attention in Transformers can be easily concatenated.

, and values v (l,i) (X) with different linear projections, and their bounds can be obtained as described in Sec. 3.2.

We also keep their linear bounds that are linear functions of the perturbed embeddings.

, where ⊕ indicates vector concatenation, and thereby we represent the linear bounds as linear functions of x (r) :

where q/k/v and q/k/v mean that the inequation holds for queries, keys and values respectively.

We then bound the output of the self-attention layer starting from

We bound multiplications and divisions in the selfattention mechanism with linear functions.

We aim to bound bivariate function z = xy or z =

We provide a proof in Appendix C. However, directly bounding z = x y is tricky; fortunately, we can bound it indirectly by first bounding a unary function y = 1 y and then bounding the multiplication z = xy.

For the self-attention mechanism, instead of using the backward process like CROWN (Zhang et al., 2018) , we compute bounds with a forward process which we will show later that it can reduce the computational complexity.

Attention scores are computed from q (l,i) (X) and

, it is bounded by:

We then obtain the bounds of S

),

In this way, linear bounds of q (l,i) (X) and k (l,i) (X) are forward propagated to S (l) i,j .

Attention scores are normalized into attention probabilities with a softmax, i.e.

S (l)

is an unary nonlinear function and can be bounded by α

, where:

By summing up bounds of each exp(S i,k ) ready, we forward propagate the bounds toS (l) i,j with a division similarly to bounding q

, which can be regarded as a dot product ofS

with a transposing.

Therefore, bounds ofS

similarly to bounding S (l) i,j .

In this way, we obtain the output bounds of the self-attention:

Recall that x (r) is a concatenation of

into t vectors with equal dimensions, Ω

, Ω

, such that Eq. (5) becomes

Backward Process to Self-Attention Layers When computing bounds for a later sub-layer, the lth sub-layer, using the backward process, we directly propagate the bounds at the the closest previous self-attention layer assumed to be the l -th layer, to the input layer, and we skip other previous sublayers.

The bounds propagated to the l -th layer are as Eq. (2).

We substitute Φ (l ,k) (X) with linear bounds in Eq. (6):

We take global bounds as Eq. (3) and Eq. (4) to obtain the bounds of the l-th layer.

Introducing a forward process can significantly reduce the complexity of verifying Transformers.

With the backward process only, we need to compute

, where the major cost is on Λ (l,i,l ,k) and there are O(m 2 n 2 ) such matrices to compute.

The O(n 2 ) factor is from the dependency between all pairs of positions in the input and output respectively, which makes the algorithm inefficient especially when the input sequence is long.

In contrast, the forward process represents the bounds as linear functions of the perturbed positions only instead of all positions by computing Ω (l,i) and Θ (l,i) .

Imperceptible adversarial examples may not have many perturbed positions (Gao et al., 2018; Ko et al., 2019) , and thus we may assume that the number of perturbed positions, t, is small.

The major cost is on Ω (l,i) while there are only O(mn) such matrices and the sizes of Λ (l,i,l ,k) and Ω (l,i) are relatively comparable for a small t. We combine the backward process and the forward process.

The number of matrices Ω is O(mn) in the forward process, and for the backward process, since we do not propagate bounds over self-attention layers and there is no cross-position dependency in other sub-layers, we only compute Λ (l,i,l ,k) such that i = k, and thus the number of matrices Λ is reduced to O(m 2 n).

So the total number of matrices Λ and Ω we compute is O(m 2 n) and is O(n) times smaller than O(m 2 n 2 ) when only the backward process is used.

Moreover, the backward process makes bounds tighter compared to solely the forward one, as we show in Appendix D.

We conduct experiments on two sentiment analysis datasets: Yelp (Zhang et al., 2015) and SST-2 (Socher et al., 2013) .

Yelp consists of 560,000/38,000 examples in the training/test set and SST-2 consists of 67,349/872/1,821 examples in the training/development/test set.

Each example is a sentence or a sentence segment (for the training data of SST-2 only) labeled with a sentiment polarity.

We verify the robustness of Transformers trained from scratch.

For the main experiments, we consider N -layer models (N ≤ 3), with 4 attention heads, hidden sizes of 256 and 512 for self-attention and feed-forward layers respectively, and we use ReLU activations for feed-forward layers.

We remove the variance related terms in layer normalization, making Transformers verification bounds tighter while the clean accuracies remain comparable (see Appendix E for discussions).

Although our method can be in principal applied to Transformers with any number of layers, we do not use large-scale pre-trained models such as BERT because they are too challenging to be tightly verified for now.

Current state-of-the-art verification methods for feed-forward networks either produce loose bounds for large networks (Zhang et al., 2018; Singh et al., 2019) , or do not scale due to computational limits (Raghunathan et al., 2018; ; Transformers contain more nonlinear operations than feed-forward networks so they are even more challenging for verification.

Dataset N Acc.

Table 1 : Clean accuracies and computed bounds for 1-position perturbation.

Bounds include upper bounds (obtained by an enumeration based method), certified lower bounds by IBP and our method respectively.

We also report the gap between upper bounds and our lower bounds (represented as the percentage of lower bounds relative to upper bounds).

We compute bounds for each possible option of perturbed positions and report the minimum ("Min") and average ("Avg") among them.

Table 2 : Bounds by IBP and our method for 2-position perturbation constrained by 2 -norm.

We compute certified lower bounds for different models on different datasets.

We include 1-position perturbation constrained by 1 / 2 / ∞ -norms and 2-position perturbation constrained by 2 -norm.

We compare our lower bounds with those computed by the Interval Bound Propagation (IBP) baseline.

For 1-position perturbation, we also compare with upper bounds computed by enumerating all the words in the vocabulary and finding the word closest to the original one such that the word substitution alters the predicted label.

This method has an exponential complexity with respect to the vocabulary size and can hardly be extended to perturbations on 2 or more positions; thus we do not include upper bounds for 2-position perturbation.

For each example, we enumerate possible options of perturbed positions (there are n t options), and we integrate results from different options by taking the minimum or average respectively.

We report the average results on 10 correctly classified random test examples with sentence lengths no more than 32 for 1-position perturbation and 16 for 2-position perturbation.

Table 1 and Table 2 present the results for 1-position and 2-position perturbation respectively.

Our certified lower bounds are significantly larger and thus tighter than those by IBP.

For 1-position perturbation, the lower bounds are consistently smaller than the upper bounds, and the gap between the upper bounds and our lower bounds are reasonable compared with that in previous work on verification of feed-forward networks, e.g. in (Weng et al., 2018; Zhang et al., 2018 ) the upper bounds are in the order of 10 times larger than lower bounds.

This demonstrates that our proposed method can compute robustness bounds for Transformers in a similar quality to the bounds of simpler neural networks.

Table 3 : Comparison of certified lower bounds and computation time (sec) by different methods.

In the following, we show the effectiveness of combining the backward process with a forward process.

We compare our proposed method (Backward & Forward) to two variations: FullyForward propagates bounds in a forward manner for all sub-layers besides self-attention layers; Fully-Backward computes bounds for all sub-layers including self-attention layers using the backward bound propagation and without the forward process.

We compare the tightness of bounds and computation time of the three methods.

We use smaller models with the hidden sizes reduced by half, and we use 1-position perturbation only, to accommodate Fully-Backward with large computational cost.

Experiments are conducted on an NVIDIA TITAN X GPU.

Table 4: Average importance scores of the most/least important words identified from 100 examples respectively on SST by different methods.

For the most important words identified, larger important scores are better, and vice versa.

Additionally, we show most/least important words identified from 10 examples on the Yelp dataset.

Boldfaced words are considered to have strong sentiment polarities, and they should appear as most important words rather than least important ones.

The certified lower bounds can reflect how sensitive a model is to the perturbation of each input token.

Intuitively, if a word is more important to the prediction, the model is more sensitive to its perturbation.

Therefore, the certified lower bounds can be used to identify important words.

In the following, we conduct an experiment to verify whether important words can be identified by our certified lower bounds.

We use a 1-layer Transformer classifier under 1-position perturbation constrained by 2 -norm.

We compare our method with two baselines that also estimate local vulnerability: Upper uses upper bounds; Gradient identifies the word whose embedding has the largest 2 -norm of gradients as the most important and vice versa.

Quantitative Analysis on SST SST contains sentiment labels for all phrases on parse trees, where the labels range from very negative (0) to very positive (4), and 2 for neutral.

For each word, assuming its label is x, we take |x − 2|, i.e. the distance to the neutral label, as the importance score, since less neutral words tend to be more important for the sentiment polarity of the sentence.

We evaluate on 100 random test input sentences and compute the average importance scores of the most or least important words identified from the examples.

In Table 4 , compared to the baselines ("Upper" and "Grad"), the average importance score of the most important words identified by our lower bounds are the largest, while the least important words identified by our method have the smallest average score.

This demonstrates that our method identifies the most and least important words more accurately compared to baseline methods.

We further analyze the results on a larger dataset, Yelp.

Since Yelp does not provide per-word sentiment labels, importance scores cannot be computed as on SST.

Thus, we demonstrate a qualitative analysis.

We use 10 random test examples and collect the words identified as the most and least important word in each example.

In Table 4 , most words identified as the most important by certified lower bounds are exactly the words reflecting sentiment polarities (boldfaced words), while those identified as the least important words are mostly stopwords.

Baseline methods mistakenly identify more words containing no sentiment polarity as the most important.

This again demonstrates that our certified lower bounds identify word importance better than baselines and our bounds provide meaningful interpretations in practice.

While gradients evaluate the sensitivity of each input word, this evaluation only holds true within a very small neighborhood (where the classifier can be approximated by a first-order Taylor expansion) around the input sentence.

Our certified method gives valid lower bounds that hold true within a large neighborhood specified by a perturbation set S, and thus it provides more accurate results.

We propose the first robustness verification method for Transformers, and tackle key challenges in verifying Transformers, including cross-nonlinearity and cross-position dependency, for efficient and effective verification.

Our method computes certified lower bounds that are significantly tighter than those by IBP.

Quantitative and qualitative analyses further show that our bounds are meaningful and can reflect the importance of different words in sentiment analysis.

A ILLUSTRATION OF DIFFERENT BOUNDING PROCESSES Figure 1 : Illustration of three different bounding processes: Fully-Forward (a), Fully-Backward (b), and Backward&Forward (c).

We show an example of a 2-layer Transformer, where operations can be divided into two kinds of blocks, "Feed-forward" and "Self-attention".

"Self-attention" contains operations in the self-attention mechanism starting from queries, keys, and values, and "Feed-forward" contains all the other operations including linear transformations and unary nonlinear functions.

Arrows with solid lines indicate the propagation of linear bounds in a forward manner.

Each backward arrow A k → B k with a dashed line for blocks A k , B k indicates that there is a backward bound propagation to block B k when computing bounds for block A k .

Blocks with blue rectangles have forward processes inside the blocks, while those with green rounded rectangles have backward processes inside.

Backward & Forward algorithm, we use backward processes for the feed-forward parts and forward processes for self-attention layers, and for layers after self-attention layers, they no longer need backward bound propagation to layers prior to self-attention layers.

In this way, we resolve the cross-position dependency in verifying Transformers while still keeping bounds comparably tight as those by using fully backward processes.

Empirical comparison of the three frameworks are presented in Sec. 4.3.

We show in Sec. 3.2 that linear bounds can be propagated over unary nonlinear functions as long as the unary nonlinear functions can be bounded with linear functions.

Such bounds are determined for each neuron respectively, according to the bounds of the input for the function.

Specifically, for a unary nonlinear function σ(x), with the bounds of x obtained previously as x ∈ [l, u], we aim to derive a linear lower bound α L x + β L and a linear upper bound

where parameters α L , β L , α U , β U are dependent on l, u and designed for different functions σ(x) respectively.

We introduce how the parameters are determined for different unary nonlinear functions involved in Transformers such that the linear bounds are valid and as tight as possible.

Bounds of ReLU and tanh has been discussed by Zhang et al. (2018) , and we further derive bounds of e x , 1

x , x 2 , √ x. x 2 and √ x is only used when the layer normalization is not modified for experiments to study the impact of our modification.

For the following description, we define the endpoints of the function to be bounded within the range as (l, σ(l)) and (u, σ(u)).

We describe how the lines corresponding to the linear bounds of different functions can be determined, and thereby parameters α L , β L , α U , β U can be determined accordingly.

ReLU For ReLU activation, σ(x) = max(x, 0).

ReLU σ(x) is inherently linear on segments (−∞, 0] and [0, ∞) respectively, so we make the linear bounds exactly σ(x) for u ≤ 0 or l ≥ 0; and for l < 0 < u, we take the line passing the two endpoints as the upper bound; and take σ L (x) = 0 when u < |l| and σ L (x) = 1 when u ≥ |l| as the lower bound, to minimize the gap between the lower bound and the original function.

1+e −2x .

tanh is concave for l ≥ 0, and thus we take the line passing the two endpoints as the lower bound and take a tangent line passing ((l+u)/2, σ((l+u)/2) as the upper bound.

For u ≤ 0, tanh is convex, and thus we take the line passing the two endpoints as the upper bound and take a tangent line passing ((l + u)/2, σ((l + u)/2) as the lower bound.

For l < 0 < u, we take a tangent line passing the right endpoint and

as the lower bound, and take a tangent line passing the left endpoint and

L and d U can be found with a binary search.

Exp σ(x) = exp(x) = e x is convex, and thus we take the line passing the two endpoints as the upper bound and take a tangent line passing (d, σ(d)) as the lower bound.

Preferably, we take d = (l + u)/2.

However, e x is always positive and used in the softmax for computing normalized attention probabilities in self-attention layers, i.e. exp(S (l) i,j ) and

i,k ) appears in the denominator of the softmax, and to make reciprocal function 1 x finitely bounded the range of x should not pass 0.

Therefore, we impose a constraint to force the lower bound function to be always positive, i.e.

Reciprocal For the reciprocal function, σ(x) = 1 x .

It is used in the softmax and layer normalization and its input is limited to have l > 0 by the lower bounds of exp(x), and √ x. With l > 0, σ(x) is convex.

Therefore, we take the line passing the two endpoints as the upper bound.

And we take the tangent line passing ((l + u)/2, σ((l + u)/2)) as the lower bound.

Square For the square function, σ(x) = x 2 .

It is convex and we take the line passing the two endpoints as the upper bound.

And we tan a tangent line passing (d, σ(d)) (d ∈ [l, u] ) as the lower bound.

We still prefer to take d = (l+u)/2.

x 2 appears in computing the variance in layer normalization and is later passed to a square root function to compute a standard derivation.

To make the input to the square root function valid, i.e. non-negative, we impose a constraint σ

2 is the tangent line passing (d, σ(d) ).

For u ≤ 0, x 2 is monotonously decreasing, the constraint we impose is equivalent to σ L (u) = 2du − d 2 ≥ 0, and with d ≤ 0, we have d ≥ 2u.

So we take d = max((l + u)/2, 2u).

For l ≥ 0, x 2 is monotonously increasing, the constraint we impose is equivalent to σ L (l) = 2dl − d 2 ≥ 0, and with

2 is negative for d = 0 and is zero for d = 0.

And for l < 0 < u, the constraint we impose is equivalent to σ L (0) = −d 2 ≥ 0, and thus we take d = 0.

Square root For the square root function, σ(x) = √

x. It is used the to compute a standard derivation in the layer normalization and its input is limited to be positive by the lower bounds of x 2 and a smoothing constant, so l > 0.

σ(x) is concave, and thus we take the line passing the two endpoints as the lower bound and take the tangent line passing ((l + u)/2, σ((l + u)/2)) as the upper bound.

We provide a mathematical proof of optimal parameters for linear bounds of multiplications used in Sec. 3.3.

We also show that linear bounds of division can be indirectly obtained from bounds of multiplications and the reciprocal function.

For each multiplication, we aim to bound z = xy with two linear bounding planes z

, where x and y are both variables and x ∈ [l x , u x ], y ∈ [l y , u y ] are concrete bounds of x, y obtained from previous layers, such that:

Our goal is to determine optimal parameters of bounding planes

U , such that the bounds are as tight as possible.

We define a difference function F L (x, y) which is the difference between the original function z = xy and the lower bound z

To make the bound as tight as possible, we aim to minimize the integral of the difference function

, which is equivalent to maximizing

For an optimal bounding plane, there must exist a point

To ensure that F L (x, y) ≥ 0 within the concerned area, we need to ensure that the minimum value of F L (x, y) is be non-negative.

We show that we only need to check cases when (x, y) is any of (l x , l y ), (l x , u y ), (u x , l y ), (u x , u y ), i.e. points at the corner of the considered area.

The partial derivatives of F L are: y 1 ) and (x 1 , y 1 ) cannot be the point with the minimum value of F L (x, y).

On the other hand, if there is (x 1 , y 1 )(x 1 = l x , y 1 ∈ (l y , u y )),

i.e. on one border of the concerned area but not on any corner,

This property holds for the other three borders of the concerned area.

Therefore, other points within the concerned area cannot have smaller function value F L (x, y), so we only need to check the corners, and the constraints on F L (x, y) become

We substitute γ L in Eq. (7) with Eq. (8), yielding

where

We have shown that the minimum function value F L (x, y) within the concerned area cannot appear in (l x , u x ) × (l y , u y ), i.e. it can only appear at the border, while (x 0 , y 0 ) is a point with a minimum function value F L (x 0 , y 0 ) = 0, (x 0 , y 0 ) can also only be chosen from the border of the concerned area.

At least one of x 0 = l x and x 0 = u x holds.

If we take x 0 = l x :

And from Eq. (8) we obtain

For the other case if we take x 0 = u x :

We take α L = u y similarly as in the case when x 0 = l x , and then

, so we can simply adopt the first one.

We also notice that V L 1 , V L 2 are independent of y 0 , so we may take any y 0 within [l y , u y ] such as y 0 = l y .

Thereby, we obtain the a group of optimal parameters of the lower bounding plane:

We derive the upper bound similarly.

We aim to minimize

If we take x 0 = l x :

To minimize V U 1 , we take α U = u y , and then

For the other case if we take x 0 = u x :

(l x − u x )α U ≥ −u x y 0 + max(l x l y − β U (l y − y 0 ), l x u y − β U (u y − y 0 )) = max(l x l y − u x l y , l x u y − u x u y ) = (l x − u x ) min(l y , u y ) = (l x − u x )l y So α U ≤ l y

To minimize V U 2 , we take α U = l y , and then

Since V

We have shown that closed-form linear bounds of multiplications can be derived.

However, we find that directly bounding z = x y is tricky.

If we try to derive a lower bound z L = α L xβ L y + γ for z = x y as Appendix C.1, the difference function is

The partial derivatives of F L are:

If there is (x 1 , y 1 ) ∈ (l x , u x ) × (l y , u y ) such that

<|TLDR|>

@highlight

We propose the first algorithm for verifying the robustness of Transformers.