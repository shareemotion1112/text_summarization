Effectively inferring discriminative and coherent latent topics of short texts is a critical task for many real world applications.

Nevertheless, the task has been proven to be a great challenge for traditional topic models due to the data sparsity problem induced by the characteristics of short texts.

Moreover, the complex inference algorithm also become a bottleneck for these traditional models to rapidly explore variations.

In this paper, we propose a novel model called Neural Variational Sparse Topic Model (NVSTM) based on a sparsity-enhanced topic model named Sparse Topical Coding (STC).

In the model, the auxiliary word embeddings are utilized to improve the generation of representations.

The Variational Autoencoder (VAE) approach is applied to inference the model efficiently, which makes the model easy to explore extensions for its black-box inference process.

Experimental results onWeb Snippets, 20Newsgroups, BBC and Biomedical datasets show the effectiveness and efficiency of the model.

With the great popularity of social networks and Q&A networks, short texts have been the prevalent information format on the Internet.

Uncovering latent topics from huge volume of short texts is fundamental to many real world applications such as emergencies detection BID18 , user interest modeling BID19 , and automatic query-reply BID16 .

However, short texts are characteristic of short document length, a very large vocabulary, a broad range of topics, and snarled noise, leading to much sparse word co-occurrence information.

Thus, the task has been proven to be a great challenge to traditional topic models.

Moreover, the complex inference algorithm also become a bottleneck for these traditional models to rapidly explore variations.

To address the aforementioned issue, there are many previous works introducing new techniques such as word embeddings and neural variational inference to topic models.

Word embeddings are the low-dimensional real-valued vectors for words.

It have proven to be effective at capturing syntactic and semantic information of words.

Recently, many works have tried to incorporate word embeddings into topic models to enrich topic modeling BID5 BID7 BID22 .

Yet these models general rely on computationally expensive inference procedures like Markov Chain Monte Carlo, which makes them hard to rapidly explore extensions.

Even minor changes to model assumptions requires a re-deduction of the inference algorithms, which is mathematic challenging and time consuming.

With the advent of deep neural networks, the neural variational inference has emerged as a powerful approach to unsupervised learning of complicated distributions BID8 BID17 BID14 .

It approximates the posterior of a generative model with a variational distribution parameterized by a neural network, which allows back-propagation based function approximations in generative models.

The variational autoencoder (VAE) BID8 , one of the most popular deep generative models, has shown great promise in modeling complicated data.

Motivated by the promising potential of VAE in building generative models with black-box inference process, there are many works devoting to inference topic models with VAE BID20 BID13 BID4 .

However, these methods yield the same poor performance in short texts as LDA.Based on the analysis above, we propose a Neural Variational Sparse Topic Model (NVSTM) based on a sparsity-enhanced topic model STC for short texts.

The model is parameterized with neural networks and trained with VAE.

It still follows the probabilistic characteristics of STC.

Thus, the model inherit s the advantages of both sparse topic models and deep neural networks.

Additionally, we exploit the auxiliary word embeddings to improve the generation of short text representations.1.

We propose a novel Neural Variational Sparse Topic Model (NVSTM) to learn sparse representations of short texts.

The VAE is utilized to inference the model effectively.

2.

The general word semantic information is introduced to improve the sparse representations of short texts via word embeddings.

3.

We conduct experiments on four datasets.

Experimental results demonstrate our model's superiority in topic coherence and text classification accuracy.

The rest of this paper is organized as follows.

First, we reviews related work.

Then, we present the details of the proposed NVSTM, followed by the experimental results.

Finally, we draw our conclusions.

Topic models.

Traditional topic models and their extensions BID0 BID2 BID12 have been widely applied to many tasks such as information retrieval, document classification and so on.

These models work well on long texts which have abundant word co-occurrence information for learning, but get stuck in short texts.

There have been many efforts to address the data sparsity problem of short texts.

To achieve sparse representations in the documenttopic and topic-term distributions, BID21 introduced a Spike and Slab prior to model the sparsity in finite and infinite latent topic structures of text.

Similarly, BID10 proposed a dual-sparse topic model that addresses the sparsity in both the topic mixtures and the word usage.

These models are inspired by the effect of the variation of the Dirichlet prior on the probabilistic topic models.

There are also some non-probabilistic sparse topic models aiming at extracting focused topics and words by imposing various sparsity constraints.

BID6 formalized topic modeling as a problem of minimizing loss function regularized by lasso.

Subsequently, Zhu & Xing presented sparse topical coding (STC) by utilizing the Laplacian prior to directly control the sparsity of inferred representations.

However, over complicated inference procedure of these sparse topic models has limited their applications and extensions.

Topic Models with Word Embeddings.

Since word embeddings can capture the semantic meanings of words via low-dimensional real-valued vectors, there have been a large number of works on topic models that incorporate word embeddings to improve topic modeling.

BID5 proposed a new technique for topic modeling by treating the document as a collection of word embeddings and topics itself as multivariate Gaussian distributions in the embedding space.

However, the assumption that topics are unimodal in the embedding space is not appropriate, since topically related words can occur distantly from each other in the embedding space.

Therefore, BID7 proposed latent concept topic model (LCTM), which modeled a topic as a distribution of concepts, where each concept defined another distribution of word vectors.

BID15 proposed Latent Feature Topic Modeling (LFTM), which extended LDA to incorporate word embeddings as latent features.

Lately, BID22 proposed a novel correlated topic model using word embeddings, which is enable to exploit the additional word-level correlation information in word embeddings and directly model topic correlation in the continuous word embedding space.

However, these models also have trouble to rapidly explore extensions.

Neural Variational Inference for topic models.

Neural variational inference is capable of approximating the posterior of a generative model with a variational distribution parameterized by a neural network BID8 BID17 BID14 .

The variational autoencoder (VAE), as one of the most popular neural variational inference approach, has shown great promise in building generative models with black-box inference process BID8 .

To break the bottleneck of over complicated inference procedure in topic models, there are many efforts devoting to inference topic models with VAE.

BID20 presents auto-encoding variational Bayes (AEVB) based inference method for latent Dirichlet allocation (LDA), tackling the problems caused by the Dirichlet prior and component collapsing in AEVB.

BID13 presents alternative neural approaches in topic modeling by providing parameterized distributions over topics.

It allows training the topic model via back-propagation under the framework of neural variational inference.

BID4 combines certain motivating ideas behind variations on topic models with modern techniques for variational inference to produce a flexible framework for topic modeling that allows for rapid exploration of different models.

Nevertheless, aforementioned works are based on traditional LDA, thus bypass the sparsity problem of short texts.

Drawing inspiration from the above analysis, we propose a novel neural variational sparse topic model NVSTM based on VAE for short texts, which combines the merits of neural networks and sparsity-enhanced topic models.

In this section, we start from describing Sparse Topical Coding (STC).

Based on it, we further propose Neural Variational Sparse Topic Model (NVSTM).

Later, we focus on the discussion of the inference process for NVSTM.

Firstly, we define that D = {1, ..., M } is a document set with size M , T = {1, ..., K} is a topic collection with K topics, V = {1, .., N } is a vocabulary with N words, and w d = {w d,1 , .., w d,|I| } is a vector of terms representing a document d, where I is the index of words in document d, and w d,n (n ∈ I) is the frequency of word n in document d. Moreover, we denote β ∈ R N ×K as a topic dictionary for the whole document set with k bases, DISPLAYFORM0 K is the word code of word n in document d. To yield interpretable patterns, (θ, s, β) are constrained to be to be non-negative.

In standard STC, each document and each word is represented as a low-dimensional code in topic space.

Based on the topic dictionary β with K topic bases sampled from a uniform distribution, the generative process is described as follows: DISPLAYFORM0

STC reconstructs each observed word count from a linear combination of a set of topic bases, where the word code is utilized as the coefficient vector.

To achieve sparse word codes, STC defines DISPLAYFORM0 The composite distribution is super-Gaussian: DISPLAYFORM1 With the Laplace term, the composite distribution tends to yield sparse word codes.

For the same purpose, the prior distribution p(θ d ) of sparse document codes is a Laplace prior Laplace(0, λ −1 ).

Additionally, According to the above generative process, we have the joint distribution: DISPLAYFORM2 To simplify the calculation, the document code can be collapsed and later obtained via an aggregation of the individual word codes of all its terms.

Although STC has closed form coordinate descent equations for parameters (θ, s, β), it is inflexible for its complex inference process.

To address the aforementioned issue, we introduce black box inference methods into STC.

We present NVSTM based on VAE and introduces word embeddings.

As in BID1 , we remove the document code and generate it via a simple aggregation of all sampled word codes among all topics: DISPLAYFORM0 .

Analogous to the generative process in STC, our model follows the generative story below for each document d: DISPLAYFORM1 The graphical representation of NVSTM is depicted in Figure 1 .

Different from STC, we replace the super-Gaussian with a uniform distribution.

In the inference process, we adopt the variational posterior Laplace(s d,nk ; 0, σ d,nk (w d,n )) to approximate the intractable posterior p(s d,nk |w d,n ) for the sparse word codes, where σ d,nk is the scale parameter of Laplace distribution.

Therefore, in the above generative process, each word code vector is generated from the uniform prior distribution.

The observed word count is sampled from Poisson distribution.

Different from traditional STC, we replace the uniform distribution of the topic dictionary with a topic dictionary neural network.

In the topic dictionary neural network, we introduce the word semantic information via word embeddings to enrich the feature space for short texts.

The topic dictionary neural network is comprised of following:Word embedding layer (E ∈ R N ×300 ): Supposing the word number of the vocabulary is N , this layer devotes to transform each word to a distributed embedding representation.

Here, we adopt the pre-trained embeddings by GloVe based on a large Wikipedia dataset 1 .

Given a word embedding matrix E, we map each word to a 300-dimensional embedding vector, which can capture subtle semantic relationships between words.

Topic dictionary layers (β ∈ R N ×K ): This layers aim at converting E to a topic dictionary similar to the one in STC.

DISPLAYFORM2 where f is a multilayer perceptron.

To conform to the framework of STC, we make a simplex projection among the output of topic dictionary neural network.

We normalize each column of the dictionary via the simplex projection as follow: DISPLAYFORM3 The simplex projection is the same as the sparsemax activation function in BID11 , which declares how the Jacobian of the projection can be efficiently computed, providing the theoretical base of its employment in a neural network trained with backpropagation.

After the simplex projection, each column of the topic dictionary is promised to be sparse, non-negative and united.

Based the above generative process, the traditional variational inference for the model is to minimize the follow optimization problem, which is a lower bound to the marginal log likelihood: DISPLAYFORM4 where q(s|γ) is approximate variational posterior, and γ is the variational parameter.

In this paper, we employ the VAE to carry out neural variational inference for our model.

Variational Autoencoder (VAE) is one of the most popular deep generative network.

It is a black-box variational method which bridges the conceptual and language gap of neural networks and probability generative models.

From neural network perspective, a variational autoencoder consists of an encoder network, a decoder network, and a loss function.

In our model, the encoder network is to parametrize the approximate posterior q θ (s|w), which takes input as the observed word count to output the latent variable s with the variational parameters θ: DISPLAYFORM0 where f e (w d,n ) is a multilayer perceptron acting on the word counts w d,n in document d, and logσ d,nk is the scale parameter of the approximate posterior, from which the word codes s d,nk are sampled.

The decoder network outputs the observed data w with given s and the generative parameters φ, which is denoted as p φ (w|s, β).

According to STC, we define the decoder network DISPLAYFORM1 DISPLAYFORM2 where f d is a multilayer perceptron.

Based on VAE, we rewrite the ELBO as: DISPLAYFORM3 The first term is a regularizer that constraints the Kullback-Leibler divergence between the encoder's distribution distribution and the prior of the latent variables.

The second term is the reconstruction loss, which encourages the decoder to reconstruct the data in minimum cost.

We devote to differentiate and optimize the lower bound above with stochastic gradient decent (SGD).

However, the gradient of the lower bound is tricky since the error is unable to back propagate through a random drawn variable s, which is a non-continuous and has no gradient.

Similar to the standard VAE, we make a differentiable transformation, called reparameterization trick.

We approximate s with an auxiliary noise variable ε ∼ U (−0.5, 0.5): DISPLAYFORM0 Through reparametrization, we can take s as a function with the parameter b deriving from the encoder network.

It allows the reconstruction error to flow through the whole network.

FIG0 presents the complete VAE inference process for NVSTM.

Moreover, in order to achieve interpretable word codes as in STC, we constrain s to be non-negative, activation function on the output s of encoder.

After apply the reparameterization trick to the variational lower bound, we can yield DISPLAYFORM1 where Θ represents the set of all the model.

As explained above, the decoding term logp(w d,n |s d,nk , β nk ) is the Poisson distribution, and β is generated by a topic dictionary neural network.

After the differentiable transformation, the variation objective function can be computed in closed form and efficiently solved with SGD.

The detailed algorithm is shown in Algorithm 1.

To evaluate the performance of our model, we present a series of experiments below.

The objectives of the experiments include: (1) the qualitative evaluations: classification accuracy of documents and sparse ratio of latent representations; (2) the qualitative inspection: the quality of extracted topics and document representations.

Our evaluation is based on the four datasets:• 20Newsgroups: The classic 20 newsgroups dataset, which is comprised of 18775 newsgroup articles with 20 categories, and contains 60698 unique words 2 .• Web Snippet: The web snippet dataset, which includes 12340 Web search snippets in 8 categories.

We remove the words with fewer than 3 characters or whose document frequency less than 3 in the dataset.

After the preprocessing, it contains 5581 unique words.

3 .• BBC:

It consists of 2225 BBC news articles from 2004-2005 with 5 classes.

We only use the title and headline of each article.

We remove the stop words and the words whose document frequency less than 3 in the dataset 4 .•

Biomedical: It consists of 20000 paper titles from 20 different MeSH in BioASQ's official website.

We convert letters into lower case and remove the words whose document frequency less than 3 in the dataset.

After preprocessing, there are 19989 documents with 20 classes 5 .Statistics on the four datasets after preprocessing is reported in TAB0 .We compare our model with five topic models: et al., 2003) .

A classical probabilistic topic model.

We use the open source LDA implemented by collapsed Gibbs sampling 6 .

We use the default settings with iteration number n = 2000, the Dirichlet parameter for distribution over topics α = 0.1 and the Dirichlet parameter for distribution over words η = 0.01.

DISPLAYFORM0 • STC (Zhu & Xing) .

A sparsity-enhanced topic model which has been proven to perform better than many existing models.

We adopt the implementation of STC released by its authors 7 .

We set the regularization constants as λ = 0.2, ρ = 0.001 and the maximum number of iterations of hierarchical sparse coding, dictionary learning as 100.• NTM BID3 .

A recently proposed neural network based topic model, which has been reported to outperform the Replicated Softmax model 8 .

In NTM, the learning rate is 0.01 and the regularization factor is 0.001.

During the pre-training procedure for all weight matrices, they are initialized with a uniform distribution in interval [-4*sqrt(6./(n visible+n hidden)),4*sqrt(6./(n visible+n hidden))], where n visible=784 and n hidden=500.• DocNADE BID9 ).

An unsupervised neural network topic model of documents and have shown that it is a competitive model both as a generative model and as a document representation learning algorithm 9 .

In DocNADE, we choose the sigmoid activate function, the hidden size is 50, the learning rate is 0.01 , the bath size is 64 and the max training number is 1000.• GaussianLDA BID5 .

A new technique for topic modeling by treating the document as a collection of word embeddings and topics itself as multivariate Gaussian distributions in the embedding space 10 .

We use default values for the parameters.

Our model is implemented in Python via TensorFlow.

For four datasets, we utilize the pre-trained 300-dimensional word embeddings from Wikipedia by GloVe, which is fixed during training.

For each out-of-vocabulary word, we sample a random vector from a normal distribution in interval [0, 1].

We adopted ADAM optimizer for weight updating with an initial learning rate of 4e − 4 for four dataset.

All weight matrices are initialized with a uniform distribution in interval [0, 1e − 5].

In practice, we found that our model is stable with the size of hidden layer, and set it to 500.

To evaluate the effectiveness of the representation of documents learned by NVSTM, we perform text classification tasks on web snippet, 20NG, BBC and Biomedical using the document codes learned by topic models as the feature representation in a multi-class SVM.

For each method, after obtaining the document representations of the training and test sets, we trained an classifier on the training set using the scikit-learn library.

We then evaluated its predictive performance on the test set.

On web snippet, we utilize 80% documents for training and 20% for testing.

On the 20NG dataset, we keep 60% documents for training and 40% for testing, which is the same configuration as in BID10 .

For BBC and Biomedical dataset, we also keep 60% documents for training and 40% for testing.

TAB2 report the classification accuracy under different methods with different settings on the number of topics among the four datasets.

It clearly denotes that 1) In the four datasets, the NVSTM yields the highest accuracy.

2) In general, the neural network based NVSTM, NTM ,DocNADE and GLDA generate better document representations than STC and LDA, demonstrating the representative advantage of neural networks in distributed word representations.

3) Sparse models NVSTM are superior to non-sparse models (DocNADE, NTM, GLDA and LDA) separately.

It indicates that sparse topic models are more capable to extract topics from short documents.

In this part, we quantitatively investigate the word codes and documents codes learned by our model.

We compute the average word code as s n = 1 Dn d∈Dn s d,n over all documents that word n appears in.

Table 4 shows the average word codes of some representative words learned by NVSTM and LDA in 8 categories of web snippet.

For each category, we also present the topics learned by NVSTM in TAB3 .

We list top-9 words according to their probabilities under each topic.

In Table 4 , the results illustrate that the codes discovered by NVSTM are apparently much sparser than those discovered by LDA.

It tends to focus on narrow spectrum of topics and obtains discriminative and sparse representations of word.

In contrast, LDA generates word codes with many non-zeros due to the data sparsity, leading to a confused topic distribution.

Besides, in NVSTM, it is clear that each non-zero element in the word codes represents the topical meaning of words in corresponding position.

The weights of these elements express their relationship with the topics.

Noticed that there are words (e.g. candidates) have only a small range of topical meanings, indicating a narrow usage of those terms.

While other words (e.g. hockey and marketing) tend to have a broad spectrum of topical meanings, denoting a general usage of those terms.

Here, each document code is calculated as BID1 .

To demonstrate the quality of the learned representations by our model, we produce a t-SNE projection with for the document codes of the four datasets learned by our model in FIG3 .

For Web Snippet, we sample 10% of the whole document codes.

For 20newsgroups and Biomedical, we sample 30% of the whole document codes.

As for BBC, we present the whole document codes.

It is obvious to see that all documents are clustered into distinct categories, which is equal to the ground truth number of categories in the four datasets.

It proves the semantic effectiveness of the documents codes learned by our model.

DISPLAYFORM0

We propose a neural sparsity-enhanced topic model NVSTM, which is the first effort in introducing effective VAE inference algorithm to STC as far as we know.

We take advantage of VAE to simplify the inference process, which require no model-specific algorithm derivations.

With the employing of word embeddings and neural network framework, NVSTM is able to generate clearer and semanticenriched representations for short texts.

The evaluation results demonstrate the effectiveness and efficiency of our model.

Future work can include extending our model with other deep generative models, such as generative adversarial network (GAN).

<|TLDR|>

@highlight

a neural sparsity-enhanced topic model based on VAE