Human annotation for syntactic parsing is expensive, and large resources are available only for a  fraction of languages.

A question we ask is whether one can leverage abundant unlabeled texts to improve syntactic parsers, beyond just using the texts to obtain more generalisable lexical features (i.e. beyond word embeddings).

To this end, we propose a novel latent-variable generative model for semi-supervised syntactic dependency parsing.

As exact inference is intractable, we introduce a differentiable relaxation to obtain approximate samples and compute gradients with respect to the parser parameters.

Our method (Differentiable Perturb-and-Parse) relies on differentiable dynamic programming over stochastically perturbed edge scores.

We demonstrate effectiveness of our approach with experiments on English, French and Swedish.

A dependency tree is a lightweight syntactic structure exposing (possibly labeled) bi-lexical relations between words BID77 BID24 , see Figure 1 .

This representation has been widely studied by the NLP community leading to very efficient state-of-the-art parsers BID30 BID12 BID43 , motivated by the fact that dependency trees are useful in downstream tasks such as semantic parsing BID66 , machine translation BID11 BID4 , information extraction BID9 BID42 , question answering BID8 and even as a filtering method for constituency parsing BID34 , among others.

Unfortunately, syntactic annotation is a tedious and expensive task, requiring highly-skilled human annotators.

Consequently, even though syntactic annotation is now available for many languages, the datasets are often small.

For example, 31 languages in the Universal Dependency Treebank, 1 the largest dependency annotation resource, have fewer than 5,000 sentences, including such major languages as Vietnamese and Telugu.

This makes the idea of using unlabeled texts as an additional source of supervision especially attractive.

In previous work, before the rise of deep learning, the semi-supervised parsing setting has been mainly tackled with two-step algorithms.

On the one hand, feature extraction methods first learn an intermediate representation using an unlabeled dataset which is then used as input to train a supervised parser BID35 BID83 BID7 BID73 .

On the other hand, the self-training and co-training methods start by learning a supervised parser that is then used to label extra data.

Then, the parser is retrained with this additional annotation BID68 BID25 BID50 .

Nowadays, unsupervised feature extraction is achieved in neural parsers by the means of word embeddings BID55 BID65 .

The natural question to ask is whether one can exploit unlabeled data in neural parsers beyond only inducing generalizable word representations.

Figure 1: Dependency tree example: each arc represents a labeled relation between the head word (the source of the arc) and the modifier word (the destination of the arc).

The first token is a fake root word.

Our method can be regarded as semi-supervised Variational Auto-Encoder (VAE, Kingma et al., 2014) .

Specifically, we introduce a probabilistic model (Section 3) parametrized with a neural network (Section 4).

The model assumes that a sentence is generated conditioned on a latent dependency tree.

Dependency parsing corresponds to approximating the posterior distribution over the latent trees within this model, achieved by the encoder component of VAE, see Figure 2a .

The parameters of the generative model and the parser (i.e. the encoder) are estimated by maximizing the likelihood of unlabeled sentences.

In order to ensure that the latent representation is consistent with treebank annotation, we combine the above objective with maximizing the likelihood of gold parse trees in the labeled data.

Training a VAE via backpropagation requires marginalization over the latent variables, which is intractable for dependency trees.

In this case, previous work proposed approximate training methods, mainly differentiable Monte-Carlo estimation BID27 BID67 and score function estimation, e.g. REINFORCE BID80 .

However, REINFORCE is known to suffer from high variance BID56 .

Therefore, we propose an approximate differentiable Monte-Carlo approach that we call Differentiable Perturb-and-Parse (Section 5).

The key idea is that we can obtain a differentiable relaxation of an approximate sample by (1) perturbing weights of candidate dependencies and (2) performing structured argmax inference with differentiable dynamic programming, relying on the perturbed scores.

In this way we bring together ideas of perturb-and-map inference BID62 BID45 and continuous relaxation for dynamic programming BID53 .

Our model differs from previous works on latent structured models which compute marginal probabilities of individual edges BID26 ; BID41 .

Instead, we sample a single tree from the distribution that is represented with a soft selection of arcs.

Therefore, we preserve higher-order statistics, which can then inform the decoder.

Computing marginals would correspond to making strong independence assumptions.

We evaluate our semi-supervised parser on English, French and Swedish and show improvement over a comparable supervised baseline (Section 6).Our main contributions can be summarized as follows: (1) we introduce a variational autoencoder for semi-supervised dependency parsing; (2) we propose the Differentiable Perturb-and-Parse method for its estimation; (3) we demonstrate the effectiveness of the approach on three different languages.

In short, we introduce a novel generative model for learning latent syntactic structures.

A dependency is a bi-lexical relation between a head word (the source) and a modifier word (the target), see Figure 1 .

The set of dependencies of a sentence defines a tree-shaped structure.

2 In the parsing problem, we aim to compute the dependency tree of a given sentence.

Formally, we define a sentence as a sequence of tokens (words) from vocabulary W. We assume a one-to-one mapping between W and integers 1 . . .

|W|.

Therefore, we write a sentence of length n as a vector of integers s of size n + 1 with 1 ≤ s i ≤ |W| and where s 0 is a special root symbol.

A dependency tree of sentence s is a matrix of booleans T ∈ {0, 1} (n+1)×(n+1) with T h,m = 1 meaning that word s h is the head of word s m in the dependency tree.

DISPLAYFORM0 Figure 2: (a) Illustration of our probabilistic model with random variables s, T and z for sentences, dependency trees and sentence embeddings, respectively.

The gray area delimits the latent space.

Solid arcs denote the generative process, dashed arcs denotes posterior distributions over the latent variables.

(b) Stochastic computation graph.

(c) Illustration of the decoder when computing the probability distribution of s 4 , the word at position 4.

Dashed arcs at the bottom represent syntactic dependencies between word at position 4 and previous positions.

At each step, the LSTM takes as input an embedding of the previous word (s 0 is a special start-of-sentence symbol).

Then, the GCN combines different outputs of the LSTM by transforming them with respect to their syntactic relation with the current position.

Finally, the probability of s 4 is computed via the softmax function.

More specifically, a dependency tree T is the adjacency matrix of a directed graph with n + 1 vertices v 0 . . .

v n .

A matrix T is a valid dependency tree if and only if this graph is a v 0 -rooted spanning arborescence, 3 i.e. the graph is connected, each vertex has at most one incoming arc and the only vertex without incoming arc is v 0 .

A dependency tree is projective if and only if, for each arc v h → v m , if h < m (resp.

m < h) then there exists a path with arcs T from v h to each vertex v k such that h < k < m (resp.

m < k < h).

From a linguistic point of view, projective dependency trees combine contiguous phrases (sequence of words) only.

Intuitively, this means that we can draw the dependency tree above the sentence without crossing arcs.

Given a sentence s, an arc-factored dependency parser computes the dependency tree T which maximizes a weighting function f (T ; W ) = h,m T h,m W h,m , where W is a matrix of dependency (arc) weights.

This problem can be solved with a O(n 2 ) time complexity BID75 BID51 .

If we restrict T to be a projective dependency tree, then the optimal solution can be computed with a O(n 3 ) time complexity using dynamic programming BID16 .

Restricting the search space to projective trees is appealing for treebanks exhibiting this property (either exactly or approximately): they enforce a structural constraint that can be beneficial for accuracy, especially in a low-resource scenario.

Moreover, using a more restricted search space of potential trees may be especially beneficial in a semi-supervised scenario: with a more restricted space a model is less likely to diverge from a treebank grammar and capture non-syntactic phenomena.

Finally, Eisner's algorithm BID16 can be described as a deduction system BID64 , a framework that unifies many parsing algorithms.

As such, our methodology could be applied to other grammar formalisms.

For all these reasons, in this paper, we focus on projective dependency trees only.

We now turn to the learning problem, i.e. estimation of the matrix W .

We assume that we have access to a set of i.i.d.

labeled sentences L = { s, T , . . . } and a set of i.i.d.

unlabeled sentences U = {s, . . . }.

In order to incorporate unlabeled data in the learning process, we introduce a generative model where the dependency tree is latent (Subsection 3.1).

As such, we can maximize the likelihood of observed sentences even if the ground-truth dependency tree is unknown.

We learn the parameters of this model using a variational Bayes approximation (Subsection 3.2) augmented with a discriminative objective on labeled data (Subsection 3.3).

Under our probabilistic model, a sentence s is generated from a continuous sentence embedding z and with respect to a syntactic structure T .

We formally define the generative process of a sentence of length n as: DISPLAYFORM0 This Bayesian network is shown in Figure 2a .

In order to simplify notation, we omit conditioning on n in the following.

T and z are latent variables and p(s|T , z) is the conditional likelihood of observations.

We assume that the priors p(T ) and p(z) are the uniform distribution over projective trees and the multivariate standard normal distribution, respectively.

The true distribution underlying the observed data is unknown, so we have to learn a model p θ (s|T , z) parametrized by θ that best fits the given samples: DISPLAYFORM1 Then, the posterior distribution of latent variables p θ (T , z|s) models the probability of underlying representations (including dependency trees) with respect to a sentence.

This conditional distribution can be written as: DISPLAYFORM2 In the next subsection, we explain how these two quantities can be estimated from data.

Computations in Equation 1 and Equation 2 require marginalization over the latent variables: DISPLAYFORM0 which is intractable in general.

We rely on the Variational Auto-Encoder (VAE) framework to tackle this challenge BID27 BID67 .

We introduce a variational distribution q φ (T , z|s) which is intended to be similar to p θ (T , z|s).

More formally, we want KL [q φ (T , z|s) p θ (T , z|s)] to be as small as possible, where KL is the Kulback-Leibler (KL) divergence.

Then, the following equality holds: DISPLAYFORM1 where log p θ (s) is called the evidence.

The KL divergence is always positive, therefore by removing the last term we have: DISPLAYFORM2 where the right-hand side is called the Evidence Lower Bound (ELBO).

By maximizing the ELBO term, the divergence KL [q φ (T , z|s) p θ (T , z|s)] is implicitly minimized.

Therefore, we define a surrogate objective, replacing the objective in Equation 1: DISPLAYFORM3 The ELBO in Equation 4 has two components.

First, the KL divergence with the prior, which usually has a closed form solution.

For the distribution over dependency trees, it can be computed with the semiring algorithm of BID38 .

Second, the non-trivial term DISPLAYFORM4 During training, Monte-Carlo method provides a tractable and unbiased estimation of the expectation.

Note that a single sample from q φ (T , z|s) can be understood as encoding the observation into the latent space, whereas regenerating a sentence from the latent space can be understood as decoding.

However, training a VAE requires the sampling process to be differentiable.

In the case of the sentence embedding, we follow the usual setting and define q φ (z|s) as a diagonal Gaussian: backpropagation through the the sampling process z ∼ q φ (z|s) can be achieved thanks to the reparametrization trick BID27 BID67 .

Unfortunately, this approach cannot be applied to dependency tree sampling T ∼ q φ (T |s).

We tackle this issue in Section 5.

VAEs are a convenient approach for semi-supervised learning (Kingma et al., 2014) and have been successfully applied in NLP BID33 BID81 BID85 BID82 .

In this scenario, we are given the dependency structure of a subset of the observations, i.e. T is an observed variable.

Then, the supervised ELBO term is defined as: DISPLAYFORM0 Note that our end goal is to estimate the posterior ditribution over dependency trees q φ (T |s), i.e. the dependency parser, which does not appear in the supervised ELBO.

We want to explicitly use the labeled data in order to learn the parameters of this parser.

This can be achieved by adding a discriminative training term to the overall loss.

The loss function for training a semi-supervised VAE is: DISPLAYFORM0 where the first term is the standard loss for supervised learning of log-linear models BID23 BID36 ).

In this section, we describe the neural parametrization of the encoder distribution q φ (Subsection 4.1) and the decoder distribution p θ (Subsection 4.2).

A visual representation is given in Figure 2b .

We factorize the encoder as q φ (T , z|s) = q φ (T |s)q φ (z|s).

The categorical distribution over dependency trees is parametrized by a log-linear model BID36 where the weight of an arc is given by the neural network of BID30 .The sentence embedding model is specified as a diagonal Gaussian parametrized by a LSTM, similarly to the seq2seq framework BID72 BID5 .

That is: DISPLAYFORM0 where m and v are mean and variance vectors, respectively.

We use an autoregressive decoder that combines an LSTM and a Graph Convolutional Network (GCN, Kipf & Welling, 2016; .

The LSTM keeps the history of generated words, while the GCN incorporate information about syntactic dependencies.

The hidden state of the LSTM is initialized with latent variable z (the sentence embedding).

Then, at each step 1 ≤ i ≤ n, an embedding associated with word at position i − 1 is fed as input.

A special start-of-sentence symbol embedding is used at the first position.

Let o i be the hidden state of the LSTM at position i. The standard seq2seq architecture uses this vector to predict the word at position i.

Instead, we transform it in order to take into account the syntactic structure described by the latent variable T .

Due to the autoregressive nature of the decoder, we can only take into account dependencies T h,m such that h < i and m < i.

Before being fed to the GCN, the output of the LSTM is fed to distinct multi-layer perceptrons 6 that characterize syntactic relations: if s h is the head of s i , o h is transformed with MLP , if s m is a modifier of s i , o m is transformed with MLP , and lastly o i is transformed with MLP .

Formally, the GCN is defined as follows: DISPLAYFORM0 The output vector g i is then used to estimate the probability of word s i .

The neural architecture of the decoder is illustrated on Figure 2c .

Encoder-decoder architectures are usually straightforward to optimize with the back-propagation algorithm BID40 BID37 ) using any autodiff library.

Unfortunately, our VAE contains stochastic nodes that can not be differentiated efficiently as marginalization is too expensive or intractable (see Figure 2b for the list of stochastic nodes in our computation graph).

BID27 and BID67 proposed to rely on a Monte-Carlo estimation of the gradient.

This approximation is differentiable because the sampling process is moved out of the backpropagation path.

In this section, we introduce our Differentiable Perturb-and-Parse operator to cope with the distribution over dependency trees.

Firstly, in Subsection 5.1, we propose an approximate sampling process by computing the best parse tree with respect to independently perturbed arc weights.

Secondly, we propose a differentiable surrogate of the parsing algorithm in Subsection 5.2.

Sampling from a categorical distributions can be achieved through the Gumbel-Max trick BID20 BID44 .

8 Unfortunately, this reparametrization is difficult to apply when the discrete variable can take an exponential number of values as in Markov Random Fields (MRF).

BID62 proposed an approximate sampling process: each component is perturbed independently.

Then, standard MAP inference algorithm computes the sample.

This technique is called perturb-and-map.

Arc-factored dependency parsing can be expressed as a MRF where variable nodes represent arcs, singleton factors weight arcs and a fully connected factor forces the variable assignation to describe a valid dependency tree BID71 .

Therefore, we can apply the perturb-and-map method to dependency tree sampling: DISPLAYFORM0 where G(0, 1) is the Gumbel distribution, that is sampling matrix P is equivalent to setting P i,j = − log(− log U i,j )) where U i,j ∼ Uniform(0, 1).Algorithm 1 This function search the best split point for constructing an element given its span.

b is a one-hot vector such that b i−k = 1 iff k is the best split position.

DISPLAYFORM1 s ← null-initialized vec.

of size j −

i 3:for i ≤ k < j do 4: DISPLAYFORM2 b ← ONE-HOT-ARGMAX(s) 6: DISPLAYFORM3 has contributed the optimal objective, this function sets T i,j to 1.

Then, it propagates the contribution information to its antecedents.

1: function BACKTRACK-URIGHT(i, j, T ) 2: DISPLAYFORM4 for i ≤ k < j do 5: DISPLAYFORM5 6: DISPLAYFORM6 The (approximate) Monte-Carlo estimation of the expectation in Equation 3 is then defined as: DISPLAYFORM7 where denotes a Monte-Carlo estimation of the gradient, P ∼ G(0, 1) is sampled in the last line and EISNER is an algorithm that compute the projective dependency tree with maximum (perturbed) weight BID16 .

Therefore, the sampling process is outside of the backpropagation path.

Unfortunately, the EISNER algorithm is built using ONE-HOT-ARGMAX operations that have illdefined partial derivatives.

We propose a differentiable surrogate in the next section.

We now propose a continuous relaxation of the projective dependency parsing algorithm.

We start with a brief outline of the algorithm using the parsing-as-deduction formalism, restricting this presentation to the minimum needed to describe our continuous relaxation.

We refer the reader to BID16 for an in-depth presentation.

The parsing-as-deduction formalism provides an unified presentation of many parsing algorithms BID64 BID70 .

In this framework, a parsing algorithm is defined as a deductive system, i.e. as a set of axioms, a goal item and a set of deduction rules.

Each deduced item represents a sub-analysis of the input.

Regarding implementation, the common way is to rely on dynamic programming: items are deduced in a bottom-up fashion, from smaller sub-analyses to large ones.

To this end, intermediate results are stored in a global chart.

For projective dependency parsing, the algorithm builds a chart whose items are of the form DISPLAYFORM0 represents a sub-analysis where every word s k , i ≤ k ≤ j is a descendant of s i and where s j cannot have any other modifier (resp.

can have).

The two other types are defined similarly for descendants of word s j .

In the first stage of the algorithm, the maximum weight of items are computed (deduced) in a bottom-up fashion.

For example, the weight WEIGHT[i j] is defined as the maximum of WEIGHT DISPLAYFORM1 assumes a dependency with head s i and modifier s j .

In the second stage, the algorithm retrieves arcs whose scores have contributed to the optimal objective.

Part of the pseudo-code for the first and second stages are given in Algorithm 1 and Algorithm 2, respectively.

Note that, usually, the second stage is implemented with a linear time complexity but we cannot rely on this optimization for our continuous relaxation.

This algorithm can be thought of as the construction of a computational graph where WEIGHT, BACKPTR and CONTRIB are sets of nodes (variables).

This graph includes ONE-HOT-ARGMAX operations that are not differentiable (see line 5 in Algorithm 1).

This operation takes as input a vector of weights v of size k and returns a one-hot vector o of the same size with o i = 1 if and only if v i is the element of maximum value: DISPLAYFORM2 We follow a recent trend BID22 BID45 BID18 in differentiable approximation of the ONE-HOT-ARGMAX function and replace it with the PEAKED-SOFTMAX operator: DISPLAYFORM3 1≤j≤k exp( 1 /τ v j ) where τ > 0 is a temperature hyperparameter controlling the smoothness of the relaxation: when τ → ∞ the relaxation becomes equivalent to ONE-HOT-ARGMAX.

With this update, the parsing algorithm is fully differentiable.12 Note, however, that outputs are not valid dependency trees anymore.

Indeed, then an output matrix T contains continuous values that represent soft selection of arcs.

BID53 introduced a alternative but similar approach for tagging with the Viterbi algorithm.

We report pseudo-codes for the forward and backward passes of our continuous relaxation of EISNER's algorithm in Appendix F.

The fact that T is a soft selection of arcs, and not a combinatorial structure, does not impact the decoder.

Indeed, a GCN can be run over weighted graphs, the message passed between nodes is simply multiplied by the continuous weights.

This is one of motivations for using GCNs rather than a Recursive LSTMs BID74 in the decoder.

On the one hand, running a GCN with a matrix that represents a soft selection of arcs (i.e. with real values) has the same computational cost than using a standard adjacency matrix (i.e. with binary elements) if we use matrix multiplication on GPU.

13 On the other hand, a recursive network over a soft selection of arcs requires to build a O(n 2 ) set of RNN-cells that follow the dynamic programming chart where the possible inputs of a cell are multiplied by their corresponding weight in T, which is expensive and not GPU-friendly.

We ran a series of experiments on 3 different languages to test our method for semi-supervised dependency parsing: English, French and Swedish.

Details about corpora can be found in Appendix C. The size of each dataset is reported in TAB0 .

Note that the setting is especially challenging for Swedish: the amount of unlabeled data we use here barely exceeds that of labeled data.

The hyperparameters of our network are described in Appendix D. In order to ensure that we do not bias our model for the benefit of the semi-supervised scenario, we use the same parameters as BID30 for the parser.

Also, we did not perform any language-specific parameter selections.

This makes us hope that our method can be applied to other languages with little extra effort.

We stress that no part-of-speech tags are used as input in any part of our network.

For English, the supervised parser took 1.5 hours to train on a NVIDIA Titan X GPU while the semi-supervised parser without sentence embedding, which sees 2 times more instances per epoch, took 3.5 hours to train.

Table 2 : (a) Parsing results: unlabeled attachment score / labeled attachment score.

We also report results with the parser of BID30 which uses a different discriminative loss for supervised training.

(b) Recall / Precision evaluation with respect to dependency lengths for the supervised parser and the best semi-supervised parser on the English test set.

Bold numbers highlight the main differences.

(c) Recall / Precision evaluation with respect to dependency labels for multi-word expressions (mwe), adverbial modifiers (advmod) and appositional modifiers (appos).

For each dataset, we train under the supervised and the semi-supervised scenario.

Moreover, in the semi-supervised setting, we experiment with and without latent sentence embedding z. We compare only to the model of BID30 .

Recently, even more accurate models have been proposed (e.g., BID12 .

In principle, the ideas introduced in recent work are mostly orthogonal to our proposal as we can modify our VAE model accordingly.

For example, we experimented with using bi-affine attention of BID12 , though it has not turned out beneficial in our low-resource setting.

Comparing to multiple previous parsers would have also required tuning each of them on our dataset, which is infeasible.

Therefore, we only report results with a comparable baseline, i.e. trained with a structured hinge loss BID30 BID76 .

We did not perform further tuning in order to ensure that our analysis is not skewed toward one setting.

Parsing results are summarized in Table 2a .We observe a score increase in all three languages.

Moreover, we observe that VAE performs slightly better without latent sentence embedding.

We assume this is due to the fact that dependencies are more useful when no information leaks in the decoder through z. Interestingly, we observe an improvement, albeit smaller, even on Swedish, where we used a very limited amount of unlabeled data.

We note that training with structured hinge loss gives stronger results than our supervised baseline.

In order to maintain the probabilistic interpretation of our model, we did not include a similar term in our model.

We conducted qualitative analyses for English.

14 We report scores with respect to dependency lengths in Table 2b .

We observe that the semi-supervised parser tends to correct two kind of errors.

Firstly, it makes fewer mistakes on root attachments, i.e. the recall is similar between the two parsers but the precision of the semi-supervised one is higher.

We hypothesis that root attachment errors come at a high price in the decoder because there is only a small fraction of the vocabulary that is observed with this syntactic function.

Secondly, the semi-supervised parser recovers more long distance relations, i.e. the recall for dependencies with a distance superior or equal to 7 is higher.

Intuitively, we assume these dependencies are more useful in the decoder: for short distance dependencies, the LSTM efficiently captures the context of the word to predict, whereas this infor-mation could be vanishing for long distances, meaning the GCN has more impact on the prediction.

We also checked how the scores differ across dependency labels.

We report main differences in Tables 2c.

The largest improvements are obtained for multi-word expressions: this is particularly interesting because they are known to be challenging in NLP.

Dependency parsing in the low-ressource scenario has been of interest in the NLP community due to the expensive nature of annotation.

On the one hand, transfer approaches learn a delexicalized parser for a resource-rich language which is then used to parse a low-resource one BID2 BID52 .

On the other hand, the grammar induction approach learns a dependency parser in an unsupervised manner.

BID32 introduced the first generative model that outperforms the right-branching heuristic in English.

Close to our work, BID6 use an auto-encoder setting where the decoder tries to rebuild the source sentence.

However, their decoder is unstructured (e.g. it is not auto-regressive).Variational Auto-Encoders BID27 BID67 have been investigated in the semi-supervised settings (Kingma et al., 2014) for NLP.

BID33 learn a semantic parser where the latent variable is a discrete sequence of symbols.

BID85 successfully applied the variational method to semi-supervised morphological re-inflection where discrete latent variables represent linguistic features (e.g. tense, part-of-speech tag).

BID82 proposed a semi-supervised semantic parser.

Similarly to our model, they rely on a structured latent variable.

However, all of these systems use either categorical random variables or the REINFORCE score estimator.

To the best of our knowledge, no previous work used continuous relaxation of a dynamic programming latent variable in the VAE setting.

The main challenge is backpropagation through discrete random variables.

BID45 and BID22 first introduced the Gumbel-Softmax operator for the categorical distribution.

There are two issues regarding more complex discrete distributions.

Firstly, one have to build a reparametrization of the the sampling process.

BID62 showed that low-order perturbations provide samples of good qualities for graphical models.

Secondly, one have to build a good differentiable surrogate to the structured arg max operator.

Early work replaced the structured arg max with structured attention BID26 .

However, computing the marginals over the parse forest is sensitive to numerical stability outside specific cases like non-projective dependency parsing BID41 BID78 .

BID53 proposed a stable algorithm based on dynamic program smoothing.

Our approach is highly related but we describe a continuous relaxation using the parsing-as-deduction formalism.

BID63 propose to replace the true gradient with a proxy that tries to satisfy constraints on a arg max operator via a projection.

However, their approach is computationally expensive, so they remove the tree constraint on dependencies during backpropagation.

A parallel line of work focuses on sparse structures that are differentiable BID49 BID59 .

We presented a novel generative learning approach for semi-supervised dependency parsing.

We model the dependency structure of a sentence as a latent variable and build a VAE.

We hope to motivate investigation of latent syntactic structures via differentiable dynamic programming in neural networks.

Future work includes research for an informative prior for the dependency tree distribution, for example by introducing linguistic knowledge BID57 BID61 or with an adversarial training criterion BID46 .

This work could also be extended to the unsupervised scenario.where z is the sample.

As such, e ∼ N (0, 1) is an input of the neural network for which we do not need to compute partial derivatives.

This technique is called the reparametrization trick BID27 BID67 .

Sampling from a categorical distributions can be achieved through the Gumbel-Max trick BID20 BID44 .

Randomly generated Gumbel noise is added to the log-probability of every element of the sample space.

Then, the sample is simply the element with maximum perturbed log-probability.

Let d ∈ k be a random variable taking values in the corner of the unit-simplex of dimension k with probability: DISPLAYFORM0 where w is a vector of weights.

Sampling d ∼ p(d) can be re-expressed as follows: DISPLAYFORM1 where G(0, 1) is the Gumbel distribution.

Sampling g ∼ G(0, 1) is equivalent to setting g i = − log(− log u i )) where u i ∼ Uniform(0, 1).

If w is computed by a neural network, the sampling process is outside the backpropagation path.

English We use the Stanford Dependency conversion BID10 of the Penn Treebank BID48 with the usual section split: 02-21 for training, 22 for development and 23 for testing.

In order to simulate our framework under a low-resource setting, the annotation is kept for 10% of the training set only: a labeled sentence is the sentence which has an index (in the training set) modulo 10 equal to zero.

French We use a similar setting with the French Treebank version distributed for the SPMRL 2013 shared task and the provided train/dev/test split (Abeillé et al., 2000; BID69 .Swedish We use the Talbanken dataset which contains two written text parts: the professional prose part (P) and the high school students' essays part (G).

We drop the annotation of (G) in order to use this section as unlabeled data.

We split the (P) section in labeled train/dev/test using a pseudo-randomized scheme.

We follow the splitting scheme of but fix section 9 as development instead of k-fold cross-validation.

Sentence i is allocated to section i mod 10.

Then, section 1-8 are used for training, section 9 for dev and section 0 for test.

Encoder: word embeddings We concatenate trainable word embeddings of size 100 with external word embeddings.

15 We use the word-dropout settings of BID30 .

For English, external embeddings are pre-trained with the structured skip n-gram objective BID39 .

16 For French and Swedish, we use the Polyglot embeddings BID3 .

17 We stress out that no part-of-speech tag is used as input in any part of our network.

Encoder: dependency parser The dependency parser is built upon a two-stack BiLSTM with a hidden layer size of 125 (i.e. the output at each position is of size 250).

Each dependency is then weighted using a single-layer perceptron with a tanh activation function.

Arc label prediction rely on a similar setting, we refer to the reader to BID30 for more information about the parser's architecture.

Encoder: sentence embedding The sentence is encoded into a fixed size vector with a simple leftto-right LSTM with an hidden size of 100.

The hidden layer at the last position of the sentence is then fed to two distinct single-layer perceptrons, with an output size of 100 followed by a piecewise tanh activation function, that computes means and standard deviations of the diagonal Gaussian distribution.

Decoder The decoder use fixed pre-trained embeddings only.

The recurrent layer of the decoder is a LSTM with an hidden layer size of 100.

MLP , MLP and MLP are all single-layer perceptrons with an output size of 100 and without activation function.

Training We encourage the VAE to rely on latent structures close to the targeted ones by bootstrapping the training procedure with labeled data only.

In the first two epochs, we train the network with the discriminative loss only.

Then, for the next two epochs, we add the supervised ELBO term (Equation 5).

Finally, after the 6th epoch, we also add the unsupervised ELBO term (Equation 3).

We train our network using stochastic gradient descent for 30 epochs using Adadelta (Zeiler, 2012) with default parameters as provided by the Dynet library .

In the semisupervised scenario, we alternate between labeled and unlabeled instances.

The temperature of the PEAKED-SOFTMAX operator is fixed to τ = 1.

Dynamic programs for parsing have been studied as abstract algorithms that can be instantiated with different semirings BID17 .

For example, computing the weight of the best parse relies on the R, max, + semiring.

This semiring can be augmented with set-valued operations to retrieve the best derivation.

However, a straightforward implementation would have a O(n 5 ) space complexity: for each item in the chart, we also need to store the set of arcs.

Under this formalism, the backpointer trick is a method to implicitly constructs these sets and maintain the optimal O(n 3 ) complexity.

Our continuous relaxation replaces the max operator with a smooth surrogate and the set values with a soft-selection of sets.

Unfortunately, R, PEAKED-SOFTMAX is not a commutative monoid, therefore the semiring analogy is not transposable.

We describe how we can embed a continuous relaxation of projective dependency parsing as a node in a neural network.

During the forward pass, we are given arc weights W and we compute the relaxed projective dependency tree T that maximize the arc-factored weight h,m T h,m ×W h,m .

Each output variable T h,m ∈ [0, 1] is a soft selection of dependency with head-word s h and modifier s m .

During back-propagation, we are given partial derivatives of the loss with respect to each arc and we compute the ones with respect to arc weights: DISPLAYFORM0 Note that the Jacobian matrix has O(n 4 ) values but we do need to explicitly compute it.

The space and time complexity of the forward and backward passes are both cubic, similar to Eisner's algorithm.

The forward pass is a two step algorithm:1.

First, we compute the cumulative weight of each item and store soft backpointers to keep track of contribution of antecedents.

This step is commonly called to inside algorithm.

2.

Then, we compute the contribution of each arc thanks to the backpointers.

This step is somewhat similar to the arg max reconstruction algorithm.

The outline of the algorithm is given in Algorithm 3.The inside algorithm computes the following variables:• a[i j][k] is the weight of item [i j] if we split its antecedent at k.• b[i j][k] is the soft backpointer to antecedents of item [i j] with split at k.• c[i j] is the cumulative weight of item [i j].and similarly for the other chart values.

The algorithm is given in Algorithm 5.The backpointer reconstruction algorithm compute the contribution of each arc.

We follow backpointers in reverse order in order to compute the contribution of each itemc[i j].

The algorithm is given in Algorithm 6.

During the backward pass, we compute the partial derivatives of variables using the chain rule, i.e. in the reverse order of their creation: we first run backpropagation through the backpointer reconstruction algorithm and then through the inside algorithm (see Algorithm 4).

Given the partial derivatives in FIG3 , backpropagation through the backpointer reconstruction algorithm is straighforward to compute, see Algorithm 7.

Partial derivatives of the inside algorithm's variables are given in Figure 4 .

∀i < k ≤ j : DISPLAYFORM0

@highlight

Differentiable dynamic programming over perturbed input weights with application to semi-supervised VAE