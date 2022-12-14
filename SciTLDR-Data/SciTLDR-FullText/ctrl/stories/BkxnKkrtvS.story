We investigate methods for semi-supervised learning (SSL) of a neural linear-chain conditional random field (CRF) for Named Entity Recognition (NER) by treating the tagger as the amortized variational posterior in a generative model of text given tags.

We first illustrate how to incorporate a CRF in a VAE, enabling end-to-end training on semi-supervised data.

We then investigate a series of increasingly complex deep generative models of tokens given tags enabled by end-to-end optimization, comparing the proposed models against supervised and strong CRF SSL baselines on the Ontonotes5 NER dataset.

We find that our best proposed model consistently improves performance by $\approx 1\%$ F1 in low- and moderate-resource regimes and easily addresses degenerate model behavior in a more difficult, partially supervised setting.

Named entity recognition (NER) is a critical subtask of many domain-specific natural language understanding tasks in NLP, such as information extraction, entity linking, semantic parsing, and question answering.

State-of-the-art models treat NER as a tagging problem (Lample et al., 2016; Ma & Hovy, 2016; Strubell et al., 2017; Akbik et al., 2018) , and while they have become quite accurate on benchmark datasets in recent years (Lample et al., 2016; Ma & Hovy, 2016; Strubell et al., 2017; Akbik et al., 2018; Devlin et al., 2018) , utilizing them for new tasks is still expensive, requiring a large corpus of exhaustively annotated sentences (Snow et al., 2008) .

This problem has been largely addressed by extensive pretraining of high-capacity sentence encoders on massive-scale language modeling tasks Devlin et al., 2018; Howard & Ruder, 2018; Radford et al., 2019; Liu et al., 2019b) , but it is natural to ask if we can squeeze more signal from our unlabeled data.

Latent-variable generative models of sentences are a natural approach to this problem: by treating the tags for unlabeled data as latent variables, we can appeal to the principle of maximum marginal likelihood (Berger, 1985; Bishop, 2006) and learn a generative model on both labeled and unlabeled data.

For models of practical interest, however, this presents multiple challenges: learning and prediction both require an intractable marginalization over the latent variables and the specification of the generative model can imply a posterior family that may not be as performant as the current state-of-the-art discriminative models.

We address these challenges using a semi-supervised Variational Autoencoder (VAE) (Kingma et al., 2014) , treating a neural tagging CRF as the approximate posterior.

We address the issue of optimization through discrete latent tag sequences by utilizing a differentiable relaxation of the Perturb-and-MAP algorithm (Papandreou & Yuille, 2011; Mensch & Blondel, 2018; Corro & Titov, 2018) , allowing for end-to-end optimization via backpropagation (Rumelhart et al., 1988) and SGD (Robbins & Monro, 1951) .

Armed with this learning approach, we no longer need to restrict the generative model family (as in Ammar et al. (2014) ; Zhang et al. (2017) ), and explore the use of rich deep generative models of text given tag sequences for improving NER performance.

We also demonstrate how to use the VAE framework to learn in a realistic annotation scenario where we only observe a biased subset of the named entity tags.

Our contributions can be summarized as follows:

1.

We address the problem of semi-supervised learning (SSL) for NER by treating a neural CRF as the amortized approximate posterior in a discrete structured VAE.

To the best of our knowledge, we are the first to utilize VAEs for NER.

2.

We explore several variants of increasingly complex deep generative models of text given tags with the goal of improving tagging performance.

We find that a joint tag-encoding Transformer (Vaswani et al., 2017) architecture leads to an ??? 1% improvement in F1 score over supervised and strong CRF SSL baselines.

3.

We demonstrate that the proposed approach elegantly corrects for degenerate model performance in a more difficult partially supervised regime where sentences are not exhaustively annotated and again find improved performance.

4. Finally, we show the utility of our method in realistic low-and high-resource scenarios, varying the amount of unlabeled data.

The resulting high-resource model is competitive with state-of-the-art results and, to the best of our knowledge, achieves the highest reported F1 score (88.4%) for models that do not use additional labeled data or gazetteers.

We first introduce the tagging problem and tagging model.

We then detail our proposed modeling framework and architectures.

NER is the task of assigning coarsely-typed categories to contiguous spans of text.

State-of-the-art approaches (Lample et al., 2016; Ma & Hovy, 2016; Strubell et al., 2017; Akbik et al., 2018; Liu et al., 2019a) do so by treating span extraction as a tagging problem, which we now formally define.

We are given a tokenized text sequence x 1:N ??? X N and would like to predict the corresponding tag sequence y 1:N ??? Y N which correctly encodes the observed token spans.

1 In this work, we use the BILOU (Ratinov & Roth, 2009 ) tag-span encoding, which assigns four tags for each of the C span categories (e.g., B-PER, I-PER, L-PER, U-PER for the PERSON category.)

The tag types B, I, L, U respectively encode beginning, inside, last, and unary tag positions in the original span.

Additionally we have one O tag for tokens that are not in any named entity span.

Thus our tag space has size |Y| = 4C + 1.

We call the NER task of predicting tags for tokens inference, and model it with a discriminative distribution q ?? (y 1:N |x 1:N ) having parameters ??.

Following state-of-the-art NER approaches (Lample et al., 2016; Ma & Hovy, 2016; Strubell et al., 2017; Akbik et al., 2018) , we use a neural encoding of the input followed by a linear-chain CRF (Lafferty et al., 2001 ) decoding layer on top.

We use the same architecture for q ?? throughout this work, as follows:

1.

Encode the token sequence, represented as byte-pairs, with a fixed pretrained language model.

2 That is, we first calculate:

In our first experiments exploring the use of pretrained autoregressive information for generation ( ??3.1), we use the GPT2-SM model (Radford et al., 2019; Hugging Face, 2019) .

In the experiments after ( ??3.2) we use the RoBERTa-LG model (Liu et al., 2019b; Hugging Face, 2019) .

2. Down-project the states: h

.

Combine local and transition potentials: ?? yi,yi+1 = s yi + T yi,yi+1 , T yi,yi+1 ??? R 5.

Using special start and end states y 0 = * , y N +1 = with binary potentials ?? * ,y = T * ,y , ?? y, = T y, and the forward algorithm (Lafferty et al., 2001) to compute the the partition function Z, we can compute the joint distribution:

Our tagging CRF has trainable parameters ?? = {W 1 , b 1 , V, b 2 , T } 3 and we learn them on a dataset of fully annotated sentences D S = {(x i 1:N i , y i 1:N i )} using stochastic gradient descent (SGD) and maximum likelihood estimation.

2.3 SEMI-SUPERVISED CRF-VAE

We now present the CRF-VAE, which treats the tagging CRF as the amortized approximate posterior in a Variational Autoencoder.

We first describe our loss formulations for semi-supervised and partially supervised data.

We then address optimizing these objectives end-to-end using backpropagation and the Relaxed Perturb-and-MAP algorithm.

Finally, we propose a series of increasingly complex generative models to explore the potential of our modeling framework for improving tagging performance.

The purpose of this work is to consider methods for estimation of q ?? in semi-supervised data regimes, as in Kingma et al. This marginalization is intractable for models that are not factored among y i , so we resort to optimizing the familiar evidence lower bound (ELBO) (Jordan et al., 1999; Blei et al., 2017) with an approximate variational posterior distribution, which we set to our tagging model q ?? .

We maximize the ELBO on unlabeled data in addition to maximum likelihood losses for both the inference and generative models on labeled data, yielding the following objectives:

where ?? is scalar hyper-parameter used to balance the supervised loss L S and the unsupervised loss L U (Kingma et al., 2014) .

?? is a scalar hyper-parameter used to balance the reconstruction and KL terms for the unsupervised loss (Bowman et al., 2015; Higgins et al., 2017) .

We note that, unlike a traditional VAE, this model contains no continuous latent variables.

Assuming that supervised sentences are completely labeled is a restrictive setup for semi-supervised learning of a named entity tagger.

It would be useful to be able to learn the tagger on sentences which are only partially labeled, where we only observe some named entity spans, but are not guaranteed all entity spans in the sentence are annotated and no O tags are manually annotated.

4 This presents a challenge in that we are no longer able to assume the usual implicit presence of O tags, since unannotated tokens are ambiguous.

While it is possible to optimize the marginal likelihood of the CRF on only the observed tags y O , O ??? {1, . . .

, N } in the sentence (Tsuboi et al., 2008) , doing so naively will result in a degenerate model that never predicts O, by far the most common tag (Jie et al., 2019) .

Interestingly, this scenario is easily addressed by the variational framework via the KL term.

We do this by reformulating the objective in Equation 5 to account for partially observed tag sequences:

} be the partially observed dataset where, for some sentence i, O ??? {1, . . .

, N i } is the set of observed positions and U = {1, . . .

, N i } \ O is the set of unobserved positions.

Our partially supervised objective is then

which can be optimized as before using the constrained forward-backward and KL algorithms detailed in Appendix B.

We also explore using this approach simply for regularization of the CRF posterior by omitting the token model p ?? (x|y).

Since we do not have trainable parameters for the generative model in this case, the reconstruction likelihood drops out of the objective and we have, for a single datum

the following loss:

Optimizing Equations 5 and 6 with respect to ?? and ?? using backpropagation and SGD is straightforward for every term except for the expectation terms E q ?? (y|x) [log p ?? (x|y)].

To optimize these expectations, we first make an Monte Carlo approximation using a single sample drawn from q ?? .

This discrete sample, however, is not differentiable with respect to ?? and blocks gradient computation.

While we may appeal to score function estimation (Miller, 1967; Williams, 1992; Paisley et al., 2012; Ranganath et al., 2014; Mohamed et al., 2019) to work around this, its high-variance gradients make successful optimization difficult. (2019), we can compute approximate samples from q ?? that are differentiable with respect to ?? using the Relaxed Perturb-and-MAP algorithm (Corro & Titov, 2018; Kim et al., 2019) .

Due to space limitations, we leave the derivation of Relaxed Perturb-and-MAP for linear-chain CRFs to Appendix A and detail the resulting CRF algorithms in Appendix B.

We model the prior distribution of tag sequences y 1:N as the per-tag product of a fixed categorical distribution p(y 1:N ) = i p(y i ).

The KL between q ?? and this distribution can be computed in polynomial time using a modification of the forward recursion derived in Mann & McCallum (2007) , detailed in Appendix B.

We experiment with several variations of architectures for p ?? (x 1:N |y 1:N ), presented in order of increasing complexity.

The CRF Autoencoder (Ammar et al., 2014; Zhang et al., 2017) is the previous state-of-the-art semi-supervised linear-chain CRF, which we consider a strong baseline.

This model uses a tractable, fully factored generative model of tokens given tags and does not require approximate inference.

Due to space limitations, we have detailed our implementation in Appendix C.

MF: This is our simplest proposed generative model.

We first embed the relaxed tag samples, represented as simplex vectors y i ??? ??? |Y| , into R dy p as the weighted combination of the input vector representations for each possible tag:

We then compute factored token probabilities with an inner product

where ?? X is the softmax function normalized over X .

This model is generalization of the CRF Autoencoder architecture in Appendix C where the tag-token parameters ?? x,y are computed with a low-rank factorization W U .

The restrictive factorization of MF is undesirable, since we expect that information about nearby tags may be discriminative of individual tokens.

To test this, we extend MF to use the full tag context by encoding the embedded tag sequence jointly using a two-layer transformer (Vaswani et al., 2017) with four attention heads per layer before predicting the tokens independently.

That is,

MF-GPT2:

Next, we see if we can leverage information from a pretrained language model to provide additional training signal to p ?? .

We extend MF by adding the fixed pretrained language modeling parameters from GPT2 to the token scores:

where z xi and h 0 i are the input token embeddings and hidden states from GPT2, respectively.

We additionally normalize the scales of the factors by the square root of the vector dimensionalities to prevent the GPT2 scores from washing out the tag-encoding scores (d yp = 300 and d GP T 2 = 768).

We add the same autoregressive extention to MT, using the tag encodings v instead of embeddings u.

MT-GPT2-PoE: We also consider an autoregressive extension of MT, similar to MT-GPT2, that uses a product of experts (PoE) (Hinton, 2002) factorization instead

MT-GPT2-Residual: Our last variation directly couples GPT2 with p ?? by predicting a residual via a two-layer MLP based on the tag encoding and GPT2 state:

For the MF-GPT2, MT-GPT2, and MT-GPT2-PoE models, we choose these factorizations specifically to prevent the trainable parameters from conditioning on previous word information, removing the possibility of the model learning to ignore the noisy latent tags in favor of the strong signal provided by pretrained encodings of the sentence histories (Bowman et al., 2015; Kim et al., 2018) .

We further freeze the GPT2 parameters for all models, forcing the only path for improving the generative likelihood to be through the improved estimation and encoding of the tags y 1:N .

We experiment first with the proposed models generative models for SSL and PSL in a moderately resourced regime (keeping 10% labeled data) to explore their relative merits.

We then evaluate our best generative model from these experiments, (MT), with an improved bidirectional encoder language model in a low-and high-resource settings, varying the amount of unlabeled data.

For data, we use the OntoNotes 5 (Hovy et al., 2006) NER corpus, which consists of 18 entity types annotated in 82,120 train, 12,678 validation, and 8,968 test sentences.

We begin by comparing the proposed generative models, M* along with the following baselines:

1.

Supervised (S): The supervised tagger trained only on the 10% labeled data.

2.

Supervised 100% (S*): The supervised tagger trained on the 100% labeled data, used for quantifying the performance loss from using less data.

3. AE-Exact: The CRF Autoencoder using exact inference (detailed in Appendix C.) 4.

AE-Approx: The same tag-token pair parameterization used by the CRF Autoencoder, but trained with the approximate ELBO objective as in Equation 11 instead of the exact objective in Equation 12.

The purpose here is to see if we lose anything by resorting to the approximate ELBO objective.

To simulate moderate-resource SSL, we keep annotations for only 10% of the sentences, yielding 8, 212 labeled sentences with 13, 025 annotated spans and 73, 908 unlabeled sentences.

Results are shown in Table 1 .

All models except S* use this 10% labeled data.

We first evaluate the proposed models and baselines without the use of a prior, since the use of a locally normalized factored prior can encourage overly uncertain joint distributions and degrade performance (Jiao et al., 2006; Mann & McCallum, 2007; Corro & Titov, 2018) .

We then explore the inclusion of the priors for the supervised and MT models with ?? = 0.01.

We explore two varieties of prior tag distributions: (1) the "gold" empirical tag distribution (Emp) from the full training dataset and (2) a simple, but informative, hand-crafted prior (Sim) that places 50% mass on the O tag and distributes the rest of its mass evenly among the remaining tags.

We view (2) as a practical approach, since it does not require knowledge of the gold tag distribution, and use (1) to quantify any relative disadvantage from not using the gold prior.

We find that including the prior with a small weight, ?? = 0.01, marginally improved performance and interestingly, the simple prior outperforms the empirical prior, most likely because it is slightly smoother and does not emphasize the O tag as heavily.

Curiously, we found that the approximate training of the CRF Autoencoder AE-Approx outperformed the exact approach AE-Exact by nearly 2% F1.

We also note that our attempts to leverage signal from the pretrained autoregressive GPT2 states had negligible or negative effects on performance, thus we conclude that it is the addition of the joint encoding transformer architecture MT that provides the most gains (+0.8% F1).

We also evaluate the supervised and transformer-based generative models, S and MT, on the more difficult PSL setup, where naively training the supervised model on the marginal likelihood of observed tags produces a degenerate model, due to the observation bias of never having O tags.

In this setting we drop 90% of the annotations from sentences randomly, resulting in 82,120 incompletely annotated sentences with 12,883 annotations total.

We compare the gold and simple priors for each model.

From the bottom of Table 1 , we see that again our proposed transformer model MT outperforms the supervised-only model, this time by +1.3% F1.

We also find that in this case, the MT models need to be trained with higher prior weights ?? = 0.1, otherwise they diverge towards using the O tag more uniformly with the other tags to achieve better generative likelihoods.

5 Code and experiments are available online at github.com/<anonymizedforsubmission> 6 In preliminary SSL experiments we found ?? > 0.01 to have a negative impact on performance, likely due to global/local normalization mismatch of the CRF and the prior.

7 The empirical prior puts 85% mass on the O tag Table 1 : Semi-supervised and partially-supervised models on 10% supervised training data: best in bold, second best underlined.

The proposed MT* improves performance in SSL and PSL by +1.1% F1 and +1.3% F1, respectively.

Next we explore our best proposed architecture MT and the supervised baseline in low-and highresource settings (1% and 100% training data, respectively) and study the effects of training with an additional 100K unlabeled sentences sampled from Wikipedia (detailed in Appendix E).

Since we found no advantage from using pretrained GPT2 information in the previous experiment, we evaluate the use of the bidirectional pretrained language model, RoBERTa (Liu et al., 2019b ), since we expect bidirectional information to highly benefit performance (Strubell et al. (2017) ; Akbik et al. (2018) , among others).

We also experiment with a higher-capacity tagging model, S-LG, by adding more trainable Transformers (L = 4, A = 8, H = 1024) between the RoBERTa encodings and down-projection layers.

From Table 2 we see that, like in the 10% labeled data setting, the CRF-VAE improves upon the supervised model by 0.9% F1 in this 1% setting, but we find that including additional data from Wikipedia has a negative impact.

A likely reason for this is the domain mismatch between Ontonotes5 and Wikipedia (news and encyclopedia, respectively).

In the high-resource setting, we find that using RoBERTa significantly improves upon GPT2 (+5.7% F1) and the additional capacity of S-LG further improves performance by +2.2% F1.

Although we do not see a significant improvement from semi-supervised training with Wikipedia sentences, our model is competitive with previous state-of-the-art NER approaches and outperforms all previous approaches that do not use additional labeled data or gazetteers.

Utilizing unlabeled data for semi-supervised learning in NER has been studied considerably in the literature.

A common approach is a two-stage process where useful features are learned from unsupervised data, then incorporated into models which are then trained only on the supervised data (Fernandes & Brefeld, 2011; Kim et al., 2015) .

With the rise of neural approaches, large-scale word vector (Mikolov et al., 2013; Pennington et al., 2014) and language model pretraining methods Akbik et al., 2018; Devlin et al., 2018) can be regarded in the same vein.

Table 2 : Low-and high-resource results with RoBERTa, varying available unlabeled data.

Best scores not using additional labeled data in bold.

??? Uses additional labeled data or gazetteers.

Another approach is to automatically create silver-labeled data using outside resources, whose low recall induces a partially supervised learning problem.

Bellare & McCallum (2007) approach the problem by distantly supervising (Mintz et al., 2009 ) spans using a database.

Carlson et al. (2009) similarly use a gazetteer and adapt the structured perceptron (Collins, 2002) to handle partially labeled sequences, while Yang et al. (2018) optimize the marginal likelihood (Tsuboi et al., 2008) of the distantly annotated tags.

Yang et al. (2018) 's method, however, still requires some fully labeled data to handle proper prediction of the O tag.

The problem setup from Jie et al. (2019) is the same as our PSL regime, but they use a cross-validated self-training approach.

Greenberg et al. (2018) use a marginal likelihood objective to pool overlapping NER tasks and datasets, but must exploit datasetspecific constraints, limiting the allowable latent tags to debias the model from never predicting O tags.

Generative latent-variable approaches also provide an attractive approach to learning on unsupervised data.

Ammar et al. (2014) present an approach that uses the CRF for autoencoding and Zhang et al. (2017) extend it to neural CRFs, but both require the use of a restricted factored generative model to make learning tractable.

Deep generative models of text have shown promise in recent years, with demonstrated applications to document representation learning , sentence generation (Bowman et al., 2015; Kim et al., 2018) , compression , translation (Deng et al., 2018) , and parsing (Corro & Titov, 2018) .

However, to the best of our knowledge, this framework has yet to be utilized for NER and tagging CRFs.

A key challenge for learning VAEs with discrete latent variables is optimization with respect to the inference model parameters ??.

While we may appeal to score function estimation (Williams, 1992; Paisley et al., 2012; Ranganath et al., 2014; , its empirical high-variance gradients make successful optimization difficult.

Alternatively, obtaining gradients with respect to ?? can be achieved using the relaxed Gumbel-max trick (Jang et al., 2016; Maddison et al., 2016) and has been recently extended to latent tree-CRFs by (Corro & Titov, 2018) , which we make use of here for sequence CRFs.

We proposed a novel generative model for semi-supervised learning in NER.

By treating a neural CRF as the amortized variational posterior in the generative model and taking relaxed differentiable samples, we were able to utilize a transformer architecture in the generative model to condition on more context and provide appreciable performance gains over supervised and strong baselines on both semi-supervised and partially-supervised datasets.

We also found that inclusion of powerful pretrained autoregressive language modeling states had neglible or negative effects while using a pretrained bidirectional encoder offers significant performance gains.

Future work includes the use of larger in-domain unlabeled corpora and the inclusion of latent-variable CRFs in more interesting joint semi-supervised models of annotations, such as relation extraction and entity linking.

Gumbel, 1954) and ?? ??? 0 be the temperature:

We know from Papandreou & Yuille (2011) that the MAP sequence from this perturbed distribution is a sample from the unperturbed distribution.

Coupled with the property that the zero temperature limit of the Gibbs distribution is the MAP state (Wainwright et al., 2008) , it immediately follows that the zero temperature limit of the perturbedq is a sample from q:

??? lim ?? ???0q

where q ?? (y|x; ?? ) is the tempered but unperturbed q ?? and "one-hot" is a function that converts elements of Y N to a one-hot vector representation.

Thus we can use the temperature ?? to anneal the perturbed joint distributionq ?? (y|x; ?? ) to a sample from the unperturbed distribution,??? ??? q ?? .

When ?? > 0,q ?? (y|x; ?? ) is differentiable and can be used for end-to-end optimization by allowing us to approximate the expectation with a relaxed single-sample Monte Carlo estimate:

where we have modified log p ?? (x|y) to accept the simplex representations of y 1:N fromq ?? instead of discrete elements, which has the effect of log p ?? (x|y) computing a weighted combination of its input vector representations for y ??? Y similarly to an attention mechanism or the annotation function in Kim et al. (2017) (see Equation 7.)

This can be thought of as a generalization of the Gumbel-softmax trick from Jang et al. (2016); Maddison et al. (2016) to structured joint distributions.

The statements in (8-10) also imply something of practical interest: we can compute (1) the argmax (Viterbi decoding) and its differentiable relaxation; (2) a sample and its differentiable relaxation; (3) the partition function; and (4) the marginal tag distributions, all using the same sum-product algorithm implementation, controlled by the temperature and the presence of noise.

We have detailed the algorithm in Appendix B.

In Algorithm 1 we have detailed the stable, log-space implementation of the generalized forwardbackward algorithm for computing (1) the argmax (Viterbi decoding) and its differentiable relaxation; (2) a sample and its differentiable relaxation; (3) the partition function; and (4) the marginal tag distributions below.

While this algorithm does provide practical convenience, we note that real implementations should have separate routines for computing the partition function (running only the forward algorithm), and the discrete ?? = 0 Viterbi algorithm, since it is more numerically stable and efficient.

We also have included the dynamic program for computing the constrained KL divergence between q ?? and a factored p(y) in Algorithm 2.

The idea of using a CRF to reconstruct tokens given tags for SSL has been explored before by Ammar et al. (2014) ; Zhang et al. (2017) , which we consider to be a strong baseline and restate ?? yi,yi+1 + log p ?? (x i |y i )} ??? log Z(??) = log Z(?? + log p ?? ) ??? log Z(??)

where log Z(?? + log p ?? ) is a slight abuse of notation intended to illustrate that the first term in Equation 12 is the same computation as the partition function, but with the generative log-likelihoods added to the CRF potentials.

We note that instead of using the Mixed-EM procedure from Zhang et al. (2017) , we model p ?? (x i |y i ) using free logit parameters ?? x,y for each token-tag pair and normalize using a softmax, which allows for end-to-end optimization via backpropagation and SGD.

We train each model to convergence using early-stopping on the F1 score of the validation data, with a patience of 10 epochs.

For all models that do not have trainable transformers, we train using the Adam optimizer (Kingma & Ba, 2014 ) with a learning rate of 0.001, and a batch size of 128.

For those with transformers (MT*), we train using Adam, a batch size of 32, and the Noam learning rate schedule from Vaswani et al. (2017) with a model size of d yp = 300 and 16, 000 warm-up steps (Popel & Bojar, 2018) .

Additionally, we use gradient clipping of 5 for all models and a temperature of ?? = .66 for all relaxed sampling models.

We implemented our models in PyTorch (Paszke et al., 2017) using the AllenNLP framework and the Hugging Face (2019) implementation of the pretrained GPT2 and RoBERTa.

We have made all code, data, and experiments available online at github.com/ <anonymizedforsubmission> for reproducibility and reuse.

All experimental settings can be reproduced using the configuration files in the repo.

For the experiments in ??3.2, we gather an additional training corpus of out-of-domain encyclopedic sentences from Wikipedia.

To try to get a sample that better aligns with the Ontonotes5 data, these sentences were gathered with an informed process, which was performed as follows:

1. Using the repository <anonymized for submission>, we extract English Wikipedia and align it with Wikidata.

2.

We then look up the entity classes from the Ontonotes5 specification (Hovy et al., 2006) in Wikidata and, for each NER class, find all Wikidata classes that are below this class in ontology (all subclasses).

3.

We then find all items which are instances of these classes and also have Wikipedia pages.

These are the Wikipedia entities which are likely to be instances of the NER classes.

<|TLDR|>

@highlight

We embed a CRF in a VAE of tokens and NER tags for semi-supervised learning and show improvements in low-resource settings.