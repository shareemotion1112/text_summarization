We describe two end-to-end autoencoding models for semi-supervised graph-based dependency parsing.

The first model is a Local Autoencoding Parser (LAP) encoding the input using continuous latent variables in a sequential manner; The second model is a Global Autoencoding Parser (GAP) encoding the input into dependency trees as latent variables, with exact inference.

Both models consist of two parts: an encoder enhanced by deep neural networks (DNN) that can utilize the contextual information to encode the input into latent variables, and a decoder which is a generative model able to reconstruct the input.

Both LAP and GAP admit a unified structure with different loss functions for labeled and unlabeled data with shared parameters.

We conducted experiments on WSJ and UD dependency parsing data sets, showing that our models can exploit the unlabeled data to boost the performance given a limited amount of labeled data.

Dependency parsing captures bi-lexical relationships by constructing directional arcs between words, defining a head-modifier syntactic structure for sentences, as shown in Figure 1 .

Dependency trees are fundamental for many downstream tasks such as semantic parsing (Reddy et al., 2016; , machine translation (Bastings et al., 2017; Ding & Palmer, 2007) , information extraction (Culotta & Sorensen, 2004; Liu et al., 2015) and question answering (Cui et al., 2005) .

As a result, efficient parsers (Kiperwasser & Goldberg, 2016; Ma et al., 2018) have been developed using various neural architectures.

While supervised approaches have been very successful, they require large amounts of labeled data, particularly when neural architectures are used.

Syntactic annotation is notoriously difficult and requires specialized linguistic expertise, posing a serious challenge for low-resource languages.

Semisupervised parsing aims to alleviate this problem by combining a small amount of labeled data and a large amount of unlabeled data, to improve parsing performance over labeled data alone.

Traditional semi-supervised parsers use unlabeled data to generate additional features, assisting the learning process (Koo et al., 2008) , together with different variants of self-training (S??gaard & Rish??j, 2010) .

However, these approaches are usually pipe-lined and error-propagation may occur.

In this paper, we propose two end-to-end semi-supervised parsers based on probabilistic autoencoder models illustrated in Figure 3 , Locally Autoencoding Parser (LAP) and Globally Autoencoding Parser (GAP).

In LAP, continuous latent variables are used to support tree inference by providing a better representation, while in GAP, the latent information forms a probability distribution over dependency trees corresponding to the input sentence.

A similar idea has been proposed by Corro & Titov (2018) , but our GAP model differs fundamentally from their parser, as GAP does not sample from the posterior of the latent tree structure to approximate the Evidence Lower Bound (ELBO).

Instead it relies on a tractable algorithm to directly compute the posterior to calculate the ELBO.

We summarize our contributions as follows:

1.

We proposed two autoencoding parsers for semi-supervised dependency parsing, with complementary strengths, trading off speed vs. accuracy; 2.

We propose a tractable inference algorithm to compute the expectation and marginalization of the latent dependency tree posterior analytically for GAP, avoiding sampling from the posterior to approximate the expectation (Corro & Titov, 2018) ; 3.

We show improved performance of both LAP and GAP with unlabeled data on WSJ and UD data sets empirically, and improved results of GAP comparing to a recently proposed semi-supervised parser (Corro & Titov, 2018) .

Most dependency parsing studies fall into two major groups: graph-based and transition-based (Kubler et al., 2009) .

Graph-based parsers (McDonald, 2006) regard parsing as a structured prediction problem to find the most probable tree, while transition-based parsers (Nivre, 2004; 2008) treat parsing as a sequence of actions at different stages leading to a dependency tree.

While earlier works relied on manual feature engineering, in recent years the hand-crafted features were replaced by embeddings and deep neural architectures, leading to improved performance in both graph-based parsing (Nivre, 2014; Pei et al., 2015) and transition-based parsing (Chen & Manning, 2014; Dyer et al., 2015; Weiss et al., 2015) .

More recent works rely on neural architectures for learning a representation for scoring structural decisions Andor et al. (2016) ; Kiperwasser & Goldberg (2016) ; Wiseman & Rush (2016) .

The annotation difficulty for this task, has also motivated work on unsupervised (grammar induction) and semi-supervised approaches to parsing (Tu & Honavar, 2012; Jiang et al., 2016; Koo et al., 2008; Li et al., 2014; Kiperwasser & Goldberg, 2015; Cai et al., 2017; Corro & Titov, 2018) .

Similar to other structured prediction tasks, directly optimizing the objective is difficult when the underlying probabilistic model requires marginalizing over the dependency trees.

Variational approaches are a natural way for alleviating this problem, as they try to improve the lower bound of the original objective, and were applied in several recent NLP works (Stratos, 2019; Kim et al., 2019b; Chen et al., 2018; Kim et al., 2019b; a) .

Variational Autoencoder (VAE) (Kingma & Welling, 2014) is particularly useful for latent representation learning, and is studied in semi-supervised context as the Conditional VAE (CVAE) (Sohn et al., 2015) .

The work mostly related to ours is (Corro & Titov, 2018) as they consider the dependency tree as the latent variable, but their work takes a second approximation to the variational lower bound by an extra step to sample from the latent dependency tree, without identifying a tractable inference.

We show that with the given structure, exact inference on the lower bound is achievable without approximation by sampling, which tightens the lower bound.

A dependency graph of a sentence can be regarded as a directed tree spanning all the words of the sentence, including a special "word"-the ROOT-to originate out.

Assuming a sentence length of l, a dependency tree can be denoted as T = (< h 1 , m 1 >, . . .

, < h l???1 , m l???1 >), where h t is the index in the sequence of the head word of the dependency connecting the tth word m t as a modifier.

Our graph-based parser is constructed by following the standard structured prediction paradigm (McDonald et al., 2005; Taskar et al., 2005) .

In inference, based on the parameterized scoring function S ?? with parameter ??, the parsing problem is formulated as finding the most probable directed spanning tree for a given sentence x: where T * is the highest scoring parse tree and T is the set of all valid trees for the sentence x.

It is common to factorize the score of the entire graph into the summation of its substructures: the individual arc scores (McDonald et al., 2005) :

whereT represents the candidate parse tree, and s ?? is a function scoring each individual arc.

s ?? (h, m) describes the likelihood of forming an arc from the head h to its modifier m in the tree.

Through out this paper, the scoring is based on individual arcs, as we focus on first order parsing.

We used the same neural architecture as that in Kiperwasser & Goldberg (2016)

.

In this formulation, we first use two parameters to extract two different representations that carry two different types of information: a head seeking for its modifier (h-arc); as well as a modifier seeking for its head (m-arc).

Then a nonlinear function maps them to an arc score.

For a single sentence, we can form a scoring matrix as shown in Figure 4 , by filling each entry in the matrix using the score we obtained.

Therefore, the scoring matrix is used to represent the head-modifier arc score of all the possible arcs connecting words in a sentence (Zheng, 2017) .

Using the scoring arc matrix, we build graph-based parsers.

Since exploring neural architectures for scoring is not our focus, we did not explore other architectures, however performance shall be further improved using advanced neural architectures .

Variational Autoencoder (VAE).

The typical VAE is a directed graphical model with Gaussian latent variables, denoted by z. A generative process first generates a set of z from the prior distribution ??(z) and the data x is generated as P ?? (x|z) parameterized by ?? given input x, In our scenario, x is an input sequence and z is a sequence of latent variables corresponding to it.

The VAE framework seeks to maximize the complete log-likelihood log P (x) by marginalizing out the latent variable z. Since direct parameter estimation of log P (x) is usually intractable, a common solution is to maximize its Evidence Lower Bound (ELBO) by introducing an auxiliary posterior Q(x|z) distribution that encodes the input into the latent space.

Tree Conditional Random Field.

Linear chain CRF models an input sequence x = (x 1 . . .

x l ) of length l with labels y = (y 1 . . .

y l ) with globally normalized probability

where Y is the set of all the possible label sequences, and S(x, y) the scoring function, usually decomposed as emission (

for first order models.

Tree CRF models generalize linear chain CRF to trees.

For dependency trees, if POS tags are given, the tree CRF model tries to resolve which node pairs should be connected with direction, such that the arcs form a tree.

The potentials in the dependency tree take an exponential form, thus the conditional probability of a parse tree T , given the sequence, can be denoted as:

where

is the partition function that sums over all possible valid dependency trees in the set T(x) of the given sentence x.

We extend the original VAE model for sequence labeling (Chen et al., 2018) to dependency parsing by building a latent representation position-wise to form a sequential latent representation.

It has been shown that under the VAE framework the latent representation can reflect the desired properties of the raw input (Kingma & Welling, 2014) .

This inspired us to use the continuous latent variable as neural representations for the dependency parsing task.

Typically, each token in the sentence is represented by its latent variable z t , which is a high-dimensional Gaussian variable.

This configuration on the one hand ensures the continuous latent variable retains the contextual information from lower-level neural models to assist finding its head or its modifier; on the other hand, it forces tokens of similar properties closer in the euclidean space.

We adjust the original VAE setup in our semi-supervised task by considering examples with labels, similar to recent conditional variational formulations (Sohn et al., 2015; Miao & Blunsom, 2016; Zhou & Neubig, 2017) .

We propose a full probabilistic model for any certain sentence x, with the unified objective to maximize for supervised and unsupervised parsing as follows:

This objective can be interpreted as follows: if the training example has a golden tree T with it, then the objective is the log joint probability P ??,?? (T , x); if the golden tree is missing, then the objective is the log marginal probability P ?? (x).

The probability of a certain tree is modeled by a tree-CRF in Eq. 1 with parameters ?? as P ?? (T |x).

Given the assumed generative process P (x|z), directly optimizing this objective is intractable, we instead optimize its ELBO (We show the details in the appendix, proving J lap is the ELBO of J in Lemma A.1):

[log P ?? (T |z)] .

Instead of autoencoding the input locally at the sequence level, we could alternatively directly regard the dependency tree as the structured latent variable to reconstruct the input sentence, by building a model containing both a discriminative component and a generative component.

The discriminative component builds a neural CRF model for dependency tree construction, and the generative model reconstructs the sentence from the factor graph as a Bayesian network, by assuming a generative process in which each head generates its modifier.

Concretely, the latent variable in this model is the dependency tree structure.

We model the discriminative component in our model as P ?? (T |x) parameterized by ??, taking the same form as in Eq. 1.

Typically in our model, ?? are the parameters of the underlying neural networks, whose architecture is described in Sec. 3.1.

We use a set of conditional categorical distributions to construct our Bayesian network decoder.

More specifically, using the head h and modifier m notation, each head reconstructs its modifier with the probability P (m t |h t ) for the tth word in the sentence (0th word is always the special "ROOT" word), which is parameterized by the set of parameters ??. Given ?? as a matrix of |V| by |V|, where |V| is the vocabulary size, ?? mh is the item on row m column h denoting the probability that the head word h would generate m. In addition, we have a simplex constraint m???V ?? mh = 1.

The probability of reconstructing the input x as modifiers m in the generative process is

where l is the sentence length and P (m t |h t ) represents the probability a head generating its modifier.

With the design of the discriminative component and the generative component of the proposed model, we have a unified learning framework for sentences with or without golden parse tree.

The complete data likelihood of a given sentence, if the golden tree is given, is

where s ??,?? (h, m) = s ?? (h, m) + log ?? mh , with m, x and T all observable.

For unlabeled sentences, the complete data likelihood can be obtained by marginalizing over all the possible parse trees in the set T(x):

where

We adapted a variant of Eisner (1996) 's algorithm to marginalize over all possible trees to compute both Z and U , as U has the same structure as Z, assuming a projective tree.

We use log-likelihood as our objective function.

The objective for a sentence with golden tree is: for sentence x l i with golden parse tree T l i in the labeled data set {x, T } l do

Stochastically update the parameter ?? in the encoder using Adam while fixing the decoder.

Compute the posterior Q(T ) in an arc factored manner for x u i tractably.

10:

Compute the expectation of all possible (h(head) ??? m(modif ier)) occurrence in the sentence x based on Q(T ).

Update buffer B using the expectation to the power for Obtain ?? globally and analytically based on the buffer B and renew the decoder.

14: end for If the input sentence does not have an annotated golden tree, then the objective is:

(2) Thus, during training, the objective function with shared parameters is chosen based on whether the sentence in the corpus has golden parse tree or not.

Directly optimizing the loss in Eq.2 is difficult for the unlabeled data, and may lead to undesirable shallow local optima without any constraints.

Instead, we derive the evidence lower bound (ELBO) of log P ??,?? (m|x) as follows, by denoting Q(T ) = P ??,?? (T |m, x) as the posterior:

Instead of maximizing the log-likelihood directly, we alternatively maximize the ELBO, so our new objective function for unlabeled data becomes max

In addition, to account for the unambiguity in the posterior, we incorporate entropy regularization (Tu & Honavar, 2012 ) when applying our algorithm, by adding an entropy term ??? T Q(T ) log Q(T ) with a non-negative factor ?? when the input sentence does not have a golden tree.

Adding this regularization term is equivalent as raising the expectation of Q(T ) to the power of 1 1????? .

We annealed ?? from 1 to 0.3 from the beginning of training to the end, as in the beginning, the generative model is well initialized by sentences with golden trees that resolve disambiguity.

In practice, we found the model benefits more by fixing the parameter ?? when the data is unlabeled and optimizing the ELBO w.r.t.

the parameter ??. We attribute this to the strict convexity of the ELBO w.r.t.

??, by sketching the proof in the appendix.

The details of training are shown in Alg.

1.

The common approach to approximate the expectation of the latent variables from the posterior distribution Q(T ) is via sampling in VAE-type models (Kingma & Welling, 2014) .

In a significant contrast to that, we argue in this model the expectation of the latent variable (which is the dependency tree structure) is analytically tractable by designing a variant of the inside-outside algorithm (Eisner, 1996; Paskin, 2001) in an arc decomposed manner.

We leave the detailed derivation in the appendix.

A high-level explanation is that assuming the dependency tree is projective, specialized belief propagation algorithm exists to compute not only the marginalization but also the expectation analytically, making inference tractable.

Data sets First we compared our models' performance with strong baselines on the WSJ data set, which is the Stanford Dependency conversion (De Marneffe & Manning, 2008) of the Penn Treebank (Marcus et al., 1993) using the standard section split: 2-21 for training, 22 for development and 23 for testing.

Second we evaluated our models on multiple languages, using data sets from UD (Universal Dependency) 2.3 (Mcdonald et al., 2013) .

Since semi-supervised learning is particularly useful for low-resource languages, we believe those languages in UD can benefit from our approach.

The statistics of the data used in our experiments are described in Table 3 in appendix.

To simulate the low-resource language environment, we used 10% of the whole training set as the annotated, and the rest 90% as the unlabeled.

Input Representation and Architecture Since we use the same neural architecture in all of our models, we specify the details of the architecture once, as follows: The internal word embeddings have dimension 100 and the POS embeddings have dimension 25.

The hidden layer of the bi-LSTM layer is of dimension 125.

The nonlinear layers used to form the head and the modifier representation both have 100 dimension.

For LAP, we use separate bi-LSTMs for words and POSs.

In GAP, using "POS to POS" decoder only yield the satisfactory performance.

This echos the finding that complicated decoders may cause "posterior collapse" (van den Oord et al., 2017; Kim et al., 2018) .

Training In the training phase, we use Adam (Kingma & Ba, 2014) to update all the parameters in both LAP and GAP, except the parameters in the decoder in GAP, which are updated by using their global optima in each epoch.

We did not take efforts to tune models' hyper-parameters and they remained the same across all the experiments.

We first evaluate our models on the WSJ data set and compared the model performance with other semi-supervised parsing models, including CRFAE (Cai et al., 2017) , which is originally designed for dependency grammar induction but can be modified for semi-supervised parsing, and "differentiable Perturb-and-Parse" parser (DPPP) (Corro & Titov, 2018) .

To contextualize the results, we also experiment with the supervised neural margin-based parser (NMP) (Kiperwasser & Goldberg, 2016) , neural tree-CRF parser (NTP) and the supervised version of LAP and GAP, with only the labeled data.

To ensure a fair comparison, our experimental set up on the WSJ is identical as that in DPPP and we use the same 100 dimension skip-gram word embeddings employed in an earlier transition-based system (Dyer et al., 2015) .

We show our experimental results in Table 1 .

As shown in this table, both of our LAP and GAP model are able to utilize the unlabeled data to increase the overall performance comparing with only using labeled data.

Our LAP model performs slightly worse than the NMP model, which we attribute to the increased model complexity by incorporating extra encoder and decoders to deal with the latent variable.

However, our LAP model achieved comparable results on semi-supervised parsing as the DPPP model, while our LAP model is simple and straightforward without additional inference procedure.

Instead, the DPPP model has to sample from the posterior of the structure by using a "GUMBEL-MAX trick" to approximate the categorical distribution at each step, which is intensively computationally expensive.

Further, our GAP model achieved the best results among all these methods, by successfully leveraging the the unlabeled data in an appropriate manner.

We owe this success to such a fact: GAP is able to calculate the exact expectation of the arc-decomposed latent variable, the dependency tree structure, in the ELBO for the complete data likelihood when the data is unlabeled, rather than using sampling Model UAS DPPP (Corro & Titov, 2018)(L) 88.79 DPPP (Corro & Titov, 2018) (L+U) 89.50 CRFAE (Cai et al., 2017)(L+U) 82.34 NMP (Kiperwasser & Goldberg, 2016) Table 2 : In this table we compare different models on multiple languages from UD.

Models were trained in a fully supervised fashion with labeled data only (noted as "L") or semi-supervised (notes as "L+U").

"ST" stands for self-training.

to approximate the true expectation.

Self-training using NMP with both labeled and unlabeled data is also included as a base-line, where the performance is deteriorated without appropriately using the unlabeled data.

We also evaluated our models on multiple languages from the UD data and compared the model performance with the semi-supervised version of CRFAE and the fully supervised NMP and NTP.

To fully simulate the low-resource scenario, no external word embeddings were used.

We summarize the results in Table 2 .

First, when using labeled data only, LAP and GAP have similar performance as NMP and NTP.

Second, we note that our LAP and GAP models do benefit from the unlabeled data, compared to using labeled data only.

Both our LAP and GAP model are able to exploit the hidden information in the unlabeled data to improve the performance.

Comparing between LAP and GAP, we notice GAP in general has better performance than LAP, and can better leverage the information in the unlabeled data to boost the performance.

These results validate that GAP is especially useful for low-resource languages with few annotations.

We also experimented using self-training on the labeled and unlabeled data with the NMP model.

As results show, selftraining deteriorate the performance especially when the size of the training data is small.

In this paper, we present two semi-supervised parsers, which are locally autoencoding parser (LAP) and globally autoencoding parser (GAP).

Both of them are end-to-end learning systems enhanced with neural architecture, capable of utilizing the latent information within the unlabeled data together with labeled data to improve the parsing performance, without using external resources.

More importantly, our GAP model outperforms the previous published (Corro & Titov, 2018) semisupervised parsing system on the WSJ data set.

We attribute this success to two reasons: First, our GAP model consists both a discriminative component and a generative component.

These two components are constraining and supplementing each other such that final parsing choices are made in a checked-and-balanced manner to avoid over-fitting.

Second, instead of sampling from posterior of the latent variable (the dependency tree) (Corro & Titov, 2018) , our model analytically computes the expectation and marginalization of the latent variable, such that the global optima can be found for the decoder, which leads to an improved performance.

A APPENDIX

Lemma A.1.

J lap is the ELBO (evidence lower bound) of the original objective J , with an input sequence x.

Denote the encoder Q is a distribution used to approximate the true posterior distribution P ?? (z|x), parameterized by ?? such that Q encoding the input into the latent space z.

Proof.

Combining U and L leads to the fact:

In practice, similar as VAE-style models, E z???Q ?? (z|x)

[log P ?? (x|z)] is approximated by

, where z j is the jthe sample of N samples sampled from Q ?? (z|x).

At prediction stage, we simply use ?? z rather than sampling z.

Here we used a mean field approximation (Tanaka, 1999) together with the conditional independence assumption by assuming P ?? (z|x) ??? l t=1 Q ?? (z t |x t ).

The generative model P ?? (x|z) acting as decoder parameterized by ?? tries to regenerate the specific input x t at time step t from the latent space z t , as we assume conditional independence in the generative process among P ?? (x t |z t ).

The encoder and the decoder are trained jointly in the classical variational autoencoder framework, by minimizing the KL divergence between the approximated posterior and the true posterior.

We describe the encoder and decoder formulation.

We parameterize the encoder Q ?? (z t |x t ) in such a way: First a bi-LSTM is used to obtain a non-linear transformation h t of the original x t ; then two separate MLPs are used to compute the mean ?? zt and the variance ?? 2 zt .

The generative story P ?? (x t |z t ) follows such parameterization: we used a MLP of two hidden layers in-between to take z t as the input, and then predict the word (or POS tag) over the vocabulary, such that the reconstruction probability can be measured.

Following traditional VAE training paradigms, we also apply the "re-parameterization" trick (Kingma & Welling, 2014) to circumvent the non-differentiable sampling procedure to sample z t from the Q ?? (z t |x t ).

Instead of directly sample from N (?? zt , ?? 2 zt ), we form z t = ?? zt + ?? 2 zt by sampling ??? N (0, I).

In addition, to avoid hindering learning during the initial training phases, following previous works (Chen et al., 2018; Bowman et al., 2016) , we anneal the temperature on the KL divergence term from a small value to 1.

From an empirical Bayesian perspective, rather than fixing the prior using some certain distributions, it is beneficial to estimate the prior distribution directly from the data by treating prior's parameters part of the model parameters.

Similar to the approach used in the previous study (Chen et al., 2018) , LAP also learns the priors from the data by updating them iteratively.

We initialize the priors from a standard Gaussian distribution N (0, I), where I is an identity matrix.

During the training, the current priors are updated using the last optimized posterior, following the rule:

where P (x) represents the empirical data distribution, and k the iteration step.

Empirical Bayesian is also named as "maximum marginal likelihood", such that our approach here is to marginalize over the missing observation as a random variable.

In previous studies (Chen & Manning, 2014; Kiperwasser & Goldberg, 2016) exploring parsing using neural architectures, POS tags and external embeddings have been shown to contain important information characterizing the dependency relationship between a head and a child.

Therefore, in addition to the variational autoencoding framework taking as input the randomly initialized word embeddings, optionally we can build the same structure for POS to reconstruct tags and for external embeddings to reconstruct words as well, whose variational objectives are U p and U e respectively.

Hence, the final variational objective can be a combination of three: U = U w (The original U in Lemma A.1) + U p + U e (or just U = U w + U p if external embeddings are not used).

Assuming the sentence is of length l, and we have obtained a arc decomposed scoring matrix S of size l ??l, and an entry S[i, j] i =j,j =0 stands for the arc score where ith word is the head and jth word the modifier.

We first describe the inside algorithm to compute the marginalization of all possible projective trees in Algo.2.

We then describe the outside algorithm to compute the outside tables in Algo.

3.

In this algorithm, stands for the logaddexp operation.

Finally, with the inside table ??, outside table ?? and the marginalization Z of all possible latent trees, we can compute the expectation of latent tree in an arc-decomposed manner.

Algo.

4 describes the procedure.

It results the matrix P containing the expectation of all individual arcs by marginalize over all other arcs except itself.

Light modification is needed in our study to calculate the expectation w.r.t.

the posterior distribution Q(T ) = P ??,?? (T |m, x), as we have In this section we derive the strict convexity of ELBO w.r.t.

??. Since we only care about the term containing ??, the KL divergence term degenerates to a constant.

For sentence i, Q(T i ) has been derived in the previous section as matrix P and 1 is the indication function.

Q(1(h ??? m)) log ?? mh Q(1(h ??? m)) is a Bernoulli distribution, indicating whether the arc (h ??? m) exists.

s.t.

DATA SET STATISTICS

We show the details of the statistics of the WSJ data set, which is the Stanford Dependency conversion (De Marneffe & Manning, 2008) of the Penn Treebank (Marcus et al., 1993) and the statistics of the languaes we used in UD (Universal Dependency) 2.3 (Mcdonald et al., 2013) here.

@highlight

We describe two end-to-end autoencoding parsers for semi-supervised graph-based dependency parsing.