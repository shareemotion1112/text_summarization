Previous work (Bowman et al., 2015; Yang et al., 2017) has found difficulty developing generative models based on variational autoencoders (VAEs) for text.

To address the problem of the decoder ignoring information from the encoder (posterior collapse), these previous models weaken the capacity of the decoder to force the model to use information from latent variables.

However, this strategy is not ideal as it degrades the quality of generated text and increases hyper-parameters.

In this paper, we propose a new VAE for text utilizing a multimodal prior distribution, a modified encoder, and multi-task learning.

We show our model can generate well-conditioned sentences without weakening the capacity of the decoder.

Also, the multimodal prior distribution improves the interpretability of acquired representations.

Research into generative models for text is an important field in natural language processing (NLP) and various models have been historically proposed.

Although supervised learning with recurrent neural networks is the predominant way to construct generative language models BID22 BID28 BID26 , auto-regressive word-by-word sequence generation is not good at capturing interpretable representations of text or controlling text generation with global features BID1 .

In order to generate sentences conditioned on probabilistic latent variables, BID1 proposed Variational Autoencoders (VAEs) BID11 for sentences.

However, some serious problems that prevent training of the model have been reported.

The problem that has been mainly discussed in previous papers is called "posterior collapse" BID25 .

Because decoders for textual VAEs are trained with "teacher forcing" BID27 , they can be trained to some extent without relying on latent variables.

As a result, the KL term of the optimization function (Equation 1) converges to zero and encoder input is ignored BID1 .

Successful textual VAEs have solved this problem by handicapping the decoder so the model is forced to utilize latent variables BID1 BID30 .

However, we believe that weakening the capacity of the decoder may lower the quality of generated texts and requires careful hyper-parameter turning to find the proper capacity.

Therefore, we take a different approach.

We focus on two overlooked problems.

First, previous research fails to address the problem inherent to the structure of VAEs.

The fundamental cause of posterior collapse (apart from teacher forcing) is the existence of a suboptimal local minimum for the KL term.

Second, although existing models use a LSTM as the encoder, it is known that this simple model is not sufficient for text generation tasks (Bahdanau et al., 2014; BID14 BID26 .

In this work, we propose a new architecture for textual VAEs with two modifications to solve these problems.

First, we use a multimodal prior distribution and an unimodal posterior distribution to eliminate the explicit minima of ignoring the encoder (Chapter 3.2).

Multimodal prior distributions for VAEs have been proposed recently for image and video tasks BID7 BID3 .

Specifically, our model uses a Gaussian Mixture distribution as prior distribution which is trained with the method proposed by BID23 .(a) The overall architecture of existing models.(b) The overall architecture of our model.

In the encoder, hidden states of the self-attention Encoder and BoW are concatenated.

The decoder estimates BoW of the input text from the latent variables as a sub-task in addition to generating text.

In our model, the prior distribution of the latent variables is a Gaussian mixture model.

Second, we modify the encoder (Chapter 3.3).

We empirically compare a number of existing encoders and adopt a combination of two.

The first is the recently proposed method of embedding text into fixed-size variables using the attention mechanism BID12 .

Although this method was originally proposed for classification tasks, we show this encoder is also effective at text generation tasks.

The second is a a Bag-of-Words encoding of input text to help the encoder.

It has been reported that a simple Bag-of-Words encoding is effective at embedding the semantic content of a sentence BID18 .

Our experiments show that the modified encoder produces improved results only when other parts of the model are modifed as well to stabilize training.

Additionally, our results imply that the self-attention encoder captures grammatical structure and Bag-of-Words captures semantic content.

Finally, to help the model acquire meaningful latent variables without weakening the decoder, we add multi-task learning (Chapter 3.4).

We find that a simple sub-task of predicting words included in the text significantly improves the quality of output text.

It should be noted that this task does not cause posterior collapse as it does not require teacher forcing.

With these modifications, our model outperforms baselines on BLEU score, showing that generated texts are well conditioned on information from the encoder (Chapter 4.3).

Additionally, we show that each component of the multimodal prior distribution captures grammatical or contextual features and improves interpretability of the global features (Chapter 4.5).

BID1 is the first work to apply VAEs to language modeling.

They identify the problem of posterior collapse for textual VAEs and propose the usage of word dropout and KL annealing.

BID16 models text as Bag-of-Words with VAEs.

This is part of the motivation behind the usage of Bag-of-Words for textual VAEs.

BID30 hypothesize that posterior collapse can be prevented by controlling the capacity of the decoder and propose a model with a dilated CNN decoder which allows changing the effective filter size.

BID21 use a deconvolutional layer without teacher forcing to force the model into using information from the encoder.

Our use of a multimodal prior distribution is inspired by previous works which try to modify prior distributions of VAEs.

BID7 and BID3 apply a VAE with Gaussian Mixture prior distribution to video and clustering, respectively.

BID23 propose the construction of a prior distribution from a mixture of posterior distributions of some trainable pseudo-inputs.

Another recent proposal to restrict the latent variables is to use discrete latent variables BID20 BID25 .

Some discrete autoencoder models for text modeling has been proposed .

While some results show promise, discretization such as Gumbel-Softmax BID6 and Vector Quantization BID25 ) is required to train discrete autoencoders with gradient descent as the gradient of discrete hidden state cannot be calculated directly.

A multimodal prior distribution can be regarded as a smoothed autoencoder model with discrete latent variables BID3 ) without a requirement for discretization.

3.1 VARIATIONAL AUTOENCODER FOR TEXT GENERATION 3.1.1 VARIATIONAL AUTOENCODER A RNN language model is trained to learn a probability distribution of the next word x t conditioned on all previous words x 1 , x 2 , . . .

, x t???1 BID17 .

A language model conditioned on a deterministic latent vector z (such as input text representation) has been proposed as well BID22 : DISPLAYFORM0 Although these models can be regarded as a generative model with auto-regressive sampling, they cannot capture interpretable probabilistic structures of global features.

BID1 propose a new language model which explicitly captures probabilistic latent variables of global features with Variational Autoencoders BID11 .Variational Autoencoders (VAEs) are one way to construct a generative model based on neural networks, which learns Variational Bayes through gradient decent.

A VAE has an encoder q ?? (z|x) and a decoder p ?? (x|z) each parameterized by a neural network.

In many cases, a standard Gaussian distribution is used for the prior distribution of the latent vector p(z) and a Gaussian distribution is used for q ?? (z|x).

Instead of directly maximizing the intractable marginal probability p(x) = p(z)p ?? (x|z)dz, we maximize the evidence lower bound: DISPLAYFORM1 (1) = L ELBO As the model samples from q ?? (z|x), the reparameterization trick BID11 can be used to train the model with gradient descent.

Previous work on textual VAEs BID1 BID30 simply applied this model to sequence-to-sequence text generation models FIG0 ).

Recent works BID1 BID30 have identified several obstacles for training VAEs for text generation.

One of the largest problems, referred to as "posterior collapse" BID25 , is that training textual VAEs often drives the second term of Equation 1 (KL term) close to zero BID1 .

When the KL term becomes zero, no information from the input text is reflected on latent variables since q ?? (z|x) and p(z) are identical.

This is an undesirable outcome since latent variables are expected to capture a meaningful representation of input to generate conditional output.

However, to aid stabilization, the previous ground truth word is given to the decoder each time during training (teacher forcing BID27 ).

As this technique is applied to textual VAEs as well, a simple language model based on LSTM can be trained without information from the decoder and cause posterior collapse.

In order to solve this problem, previous methods try to weaken the decoder to force the model to use information from the encoder.

However, weakening the capacity of the decoder is not an ideal strategy since it can lower the quality of generated text and requires additional hyper-parameters specifying decoder capacity.

In this paper, we propose three modifications to the model and successfully improve upon textual VAEs without restricting the capacity of the decoder.

These modifications are explained in the following chapters: 3.2, 3.3, and 3.4.

In typical VAEs, a standard normal distribution N (0, 1) is used as the prior distribution p(z) and a normal distribution N (??, ?? 2 ) is used as the posterior distribution q ?? (z|x).

Although this model is also used for previous textual VAE models BID1 BID30 , there is a trivial local minimum p(z) = q ?? (z|x) which makes KL(q ?? (z|x)|p(z)) in Equation 1 zero, manifesting in what is referred to as posterior collapse.

Roughly speaking, we can avoid this if q ?? (z|x) cannot be identical to p(z).

One simple way to achieve this is to use a multimodal distribution as the prior distribution p(z) and an unimodal distribution as the posterior distribution q ?? (z|x).

This idea is motivated by recently proposed VAE models with a multimodal prior distribution for image and video generation BID7 BID3 .

We provide further explanation in Appendix A and discuss that modification for the decoder is not necessary if the problems in prior distribution is fixed.

The problem with using a multimodal distribution as a prior for VAEs is deciding on what kind of distribution to use.

Models which learn a multimodal prior distribution along with other parts of the VAE have been recently proposed BID7 BID3 BID23 .

One successful model uses a multimodal prior distribution of a variational mixture of posteriors prior (VampPrior) BID23 .

VampPrior VAEs have multiple trainable pseudo-inputs u k and regard the mixture of the posterior distributions of the pseudo-inputs DISPLAYFORM0 as the prior distribution (K is a pre-defined number of pseudo-inputs).

Pseudoinputs are trained at the same time as the other components of the VAE.

Although pseudo-inputs have the same size as the input image for the VAE in the original work BID23 , we use pseudo-inputs which are projected onto ?? and ?? directly FIG1 ).In our experiments, we find a multimodal prior distribution performs unsupervised clustering and each component of multimodal prior distribution captures specific features of a sentence.

Moreover, the components themselves also form clusters, creating a hierarchical structure within the representation space (Chapter 4.5).

Existing models of textual VAEs use a simple LSTM as an encoder BID1 BID30 .

However, recent research into text generation has found that simple LSTMs do not have enough capacity to encode information from the whole text.

Motivated by the results of our experiments (Chapter 3.3), we propose concatenating the representation from the self-attention encoder and Bag-of-Words information.

Ideally, self-attention encodes grammatical structure and Bag-of-Words encodes overall meaning.

Our experiments imply our model is successful in this kind of division of roles (Chapter 4.4).

The attention mechanism (Bahdanau et al., 2014; BID14 ) is a popular model to encode text with LSTMs.

Since VAEs are models with fixed size probabilistic latent variables, this mechanism with variable size representation cannot be applied directly.

Therefore, we use a recently proposed method called self-attention BID12 (Figure 3 ), an effective model to embed text into a fixed Figure 3 : A self-attention encoder.

This model encodes variable length input into a fixed length representation using an attention mechanism.

The fixed length representation is acquired by summing up the hidden states of the bi-directional LSTM based on attention weights.

Attention weights a s1 , . . .

, a sn are calculated by (a s1 , . . .

, a sn ) = softmax(w s2 tanh(W 1 H T )).size vector representation for classification tasks using an attention mechanism.

Our experiments show that embedded representations from self-attention are useful for text generation.

The self-attention model uses hidden states of bi-directional LSTM h 1 , . . .

, h n with variable length.

To acquire a fixed sized representation m s , hidden states are summarized with attention weights m s = n i=1 a si h i .

Attention weights are calculated by using a weight matrix W 1 with shape d-by-2u (u is the size of a hidden state of bi-directional LSTM and d is a hyper-parameter) and a vector w s2 with size d: DISPLAYFORM0 Here H is a n-by-2u matrix of the hidden states H = (h 1 , . . .

, h n ).

To get richer information, r different weights (r is a hyper-parameter) are calculated with a r-by-d weight matrix W 2 = (w 12 , . . . , w r2 ) in the model: DISPLAYFORM1 Here the softmax is performed along the second dimension.

Finally, a fixed sized representation is acquired by M = AH.

We simply flatten the matrix M into a representation vector.

All parameters are trained with gradient descent.

Previous research shows the effectiveness of Bag-of-Words in NLP tasks such as text classification BID5 .

Because the difficulty of encoding the content of the input sentence with LSTM is known, we propose using a simple Bag-of-Words input to encode the content of the sentence for text generation tasks.

Also, since VAEs are trained in a stochastic manner, it is difficult to train the encoder.

Since Bag-of-Words input is much easier to train compared to LSTMs and self-attention encoders, it will help stabilize training.

We simply summarize word representation of all words in the input text and project this vector with a linear layer.

In NLP deep learning tasks, some methods to improve the performance of the main task with multitask learning has been reported.

For example, multi-lingual training even improves the result of each language in translation task BID4 and sub-task of phone recognition improves the result of speech recognition BID24 .

One of the effects of multi-task learning is said that it enables to acquire better intermediate representations BID13 .

Also, a recently proposed model to encode chemical structure with VAEs show that multi-task learning improves the quality of embedded representation BID19 .To address the largest problem of VAEs for text, the difficulty in learning meaningful latent variables, we propose using multi-task learning in our model.

However, using additional information such as grammatical properties or labels is not desirable for language modeling with textual VAEs.

We find that the simple task of predicting words in output text can help the model improve the quality of output text.

Additionally, this sub-task will alleviate the problem of posterior collapse since it does not contain auto-regressive structure which in turn requires training with teacher forcing.

We compare our model with two models proposed by Bowman et al. FORMULA5 and BID30 .

Basically, we use the same configurations for these models.

For the model of BID30 , we use a SCMM-VAE model in the original paper and pretrain the encoder.

For the multimodal prior distribution model, we report the score of a prior distribution with 500 components and analyze the acquired representation space with one with 100 components for ease of analysis.

We use 100,000 sentences from a scale document dataset "Yahoo!

Answers Comprehensive Questions and Answers version 1.0" for training to acquire the results.

For details of the dataset and model parameters, see Appendix B.

We compare a self-attention encoder BID12 , a LSTM encoder, and a Bag-of-Words encoder with tasks to embed a text into 128 sized vector and show the results in Table 1 .

First, we compare the models on a sequence-to-sequence autoencoder model.

We show that the self-attention encoder works best in terms of BLEU score.

However, we find that the self-attention encoder has a higher false negative rate compared to even a simple LSTM at the task of predicting the words in an input text.

From this result, we hypothesize that the self-attention encoder is good at acquiring the structure of a sentence or focusing on specific information but is not good at embedding all the information in a sentence.

From these results, we decided to use self-attention and Bag-of-Words for our encoder.

The results for language modeling are shown in TAB2 .

We report the reconstruction loss (negative log likelihood) of text, KL divergence and BLEU of textual VAEs.

The results show that multi-task learning and a multimodal prior distribution in isolation both improve the model.

On the other hand, changing the encoder in isolation has no influence on results.

Note that this is not the case for non-VAE models.

However, when multi-task learning is also used, incorporating Bag-of-Words input (the first modification of the encoder) improves the score.

Moreover, when we use a multimodal prior distribution, the self-attention encoder, the second modification of the encoder, outperforms the LSTM encoder.

This result implies that it is difficult to train the encoder (especially the self-attention encoder) of VAEs unlesss the overall model is improved as well.

Therefore, when other parts of the model are improved in tandem and training becomes more stable, the improved ability of the encoder is utilized.

Finally, our model with all modifications (the last line) outperforms baselines by a significant margin.

Our model uses self-attention and Bag-of-Words as the encoder.

We show the results which imply that self-attention acquires grammatical structure and Bag-of-Words provides semantic content.

is it possible to death penalty in the world to death penalty ?

Table 3 : Sampling from the posterior distribution of our model when different input is given to the self-attention and Bag-of-Words encoders.

"SA" is a sentence given to self-attention encoder and "BoW" is a sentence given to the Bag-of-Words encoder.

For details, see Chapter 4.4.

For more samples, see Table 7 in Appendix D.First, to see the relationship between these two encoders, we analyze generated sentences when different sentences are provided to self-attention and Bag-of-Words encoder.

We show examples of the results in Table 3 .

Generated sentences in Table 3 have similar grammatical structure to the input of the self-attention encoder and nouns in the sentences are strongly affected by the Bag-of-Words encoder.

Moreover, by looking into the attention weights of the self-attention encoder, we can see which parts of a sentence the encoder focuses on as shown by BID12 .

We show the maximum attention weight for each word in FIG2 .

We can see that the self-attention encoder assigns a larger weight to words which determine the structure of a sentence such as interrogatives and prepositions rather than nouns.

In addition, attention weights are similar between sentences which share grammatical structure even when nouns or word lengths differ.

We show our model properly acquires a representation of sentences and a multimodal prior distribution helps us interpret acquired representation with unsupervised clustering.

By sampling from each component, we can see that our model successfully performs clustering.

We find that sentences allocated to to components respectively have one of at least two things in common: grammatical structure or topic.

For sentences sampled from components, please see Table 8 in Appendix D. Table 4 .We show a new method to interpret the global structure of the acquired representation space.

We analyze the representation space by visualizing the means of 100 components in the multimodal prior distribution of our model with t-SNE BID15 and show the result in FIG3 .

In addition to the fact that each component clusters together, we now see the clusters themselves form into larger clusters, creating a hierarchical relationship.

We take a further look into two clear clusters indicated in FIG3 .

First, we sample from component 38, 56, and 94 in cluster 1 and show the result in Table 4 .

From the sampled sentences, we can see that components in cluster 1 share grammatical structure "[interrogative] can I [verb]" and each component has its own topics (computer, politics, culture).

On the other hand, components in cluster 2 share the topics (politics or human relationship) and each component has its own grammatical structure.

Also, from FIG3 , components 52, 31, and 37 seem to be on the circle in this order and we can see the continuous changes of grammatical structure in this order.

Thus, we can observe that our model acquires a hierarchical structure of sentences and the structure can be easily interpreted through analysis of components in the multimodal prior distribution.

As models with multimodal distributions are relatively new, we hope methods to control multimodal prior distribution are investigated further in future works.

However, we emphasize that our result is already impressive since without a multiomodal prior, extensive search with sampling or additional labels is required to interpret the structure of acquired text representation.

A multimodal prior distribution makes it much easier to understand the structure of the representation space though analysis of components of the distribution.

how do u get a group of african american to join ?

how do i increase a mortgage loan in indiana ?

how do you get married in canada ?

Table 4 : Samples from components of the prior distribution from cluster 1 (above) and 2 (below) in FIG3 .

Components in cluster 1 share grammatical structure and components in cluster 2 share topics.

Please see Chapter 4.5 for more details.

and increases hyper-parameters.

We show (i) multimodal prior distribution, (ii) improvement of the encoder and (iii) multi-task learning can improve the model with a simple LSTM decoder.

We show theoretical justification for a multimodal prior distribution as a solution for posterior collapse.

We use the equivalent objective for ELBO (Equation 1) by BID31 : DISPLAYFORM0 where DISPLAYFORM1 , which does not depend on any parameters.

This objective can be minimized to zero without utilizing latent variables under the assumption that (i) the decoder is sufficiently flexible and (ii) the posterior distribution can be trained so p(z) = q ?? (z|x) BID31 .

This nature of ELBO causes posterior collapse in VAEs.

There are two simple ways to break these assumptions.

First, if the capacity of the decoder is restricted, assumption (i) cannot be satisfied.

This is the theoretical underpinning for previous approaches used in textual VAEs BID1 BID30 which restrict the capacity of the decoder.

However, as previously discussed, weakening the decoder is undesirable.

Additionally, hyper-parameter search is required to strike a balance between the two terms if the KL term is not modified as well.

Therefore, we propose to break the assumption (ii) with a multimodal prior distribution.

When the prior distribution p(z) is a multimodal distribution and the posterior distribution q ?? (z|x) is an unimodal distribution, there is no way to satisfy p(z) = q ?? (z|x).

Moreover, there will be multiple minima for KL(q ?? (z|x)|p(z)).

When Kullback-Leibler divergence KL(q|p) between the Gaussian mixture distribution p and the normal distribution q is minimized (here we assume that q is trainable), q will be allocated to one component of p since KL(q|p) = q(z) log q(z) p(z) dz becomes larger when p is assigned a low probability in an area where q is assigned a high probability FIG4 ).

In such a formulation, there is no clear global minima for the KL term and the posterior distribution is not forced to ignore information from the encoder.

We propose a hypothesis that the modification of the decoder is not necessary if multimodal prior distribution is used.

In practice, it is natural to assume that training the decoder so p D (x) = p ?? (x|z) for all z is much harder than to make KL(q ?? (z|x)|p(z)) = 0.

Under the assumption, the model will be trained so KL(q ?? (z|x)|p(z)) = 0 as the first step and this condition force the decoder to be trained so p D (x) = p ?? (x|z) for all z when there is no modification of the model.

Although this is the opposite way from the explanation by BID31 , this process is more natural in practice since it is easy to train prior distribution.

Therefore, if we modify the model to avoid KL(q ?? (z|x)|p(z)) = 0, the decoder will not try to satisfy p D (x) = p ?? (x|z) for all z but learn the conditioned distribution for each z. This analysis motivates us to modify textual VAE without weakening the capacity of the decoder.

The results of our experiments are consistent with this hypothesis.

B .

As test and validation dataset, we use 10,000 sentences each.

We set the maximum length of a sentence to 60 words (ignore the rest of the sentence, the average length of the original sentences is 38.12 words) and use the most common 40,000 words for this experiment.

Our model uses self-attention and Bag-of-Words in the encoder and a LSTM for the decoder.

The size of the hidden state of LSTM is 256 for both for LSTM and self-attention.

The size of the word embedding is 256 and the size of the latent variables is 128.

For the self-attention encoder, we use d = 350 and r = 30.

In accordance with BID1 BID30 , we feed the latent variables on every step of the decoder LSTM by concatenating it with the word embedding.

We applied 0.4 word dropout for input text to the decoder for our model and the model from BID1 .

In this paper, we modify the model without restricting the capacity of the decoder.

However, the method used by BID1 called word dropout, which was originally proposed to weaken the decoder, is now seen as a method of smoothing BID29 .

As this method is also effective and harmless for non-VAE text generation task, we use word dropout for our model.

In addition, we pretrain the encoder and the decoder with sequence-to-sequence text generation for our multi-prior distribution model.

Note that it was impossible to pretain decoders for previous models since it can result in posterior collapse.

For multi-prior distribution, we compare 4 numbers of components [50, 100, 500, 2000] and found that performance is not sensitive to this hyperparameter, although a larger number of components results in a slightly better score TAB5 .

As using a prior distribution with many components leads to overfitting, over-regularization, and high computational complexity BID23 , we report the score of a prior distribution with 500 components and analyze the acquired representation space with 100 components for ease of analysis.

We compare our model with two models proposed by BID1 and BID30 .

Basically, we use the same configurations for these models.

For the model of BID30 , we use the SCMM-VAE model in the original paper and pretrain the encoder.

We use Adam BID10 for the optimizer.

According to our experiments, setting the learning rate to 5 ?? 10 ???4 and ?? 1 to 0.5 performs the best.

For KL weight annealing, we set the initial weight for the KL term to be 0 and increase it linearly to 1 until epoch 30.

After KL weight annealing, we train for 80 epochs with learning rate decay (0.95 for every epoch).

Table 6 : Semi-supervised learning.

LM-LSTM and SA-LSTM come from BID2 , they denotes the LSTM initialized with an autoencoder and a language model.

The methods of semisupervised learning with VAEs use the same scheme as BID30 .

LSTM is a simple supervised model.

The structure of semi-supervised models using VAEs is taken from BID30 .

We use the topic of a sentence from the dataset as a label and feed the encoded representation from the encoder to the discriminator.

We report the results of semi-supervised learning in Table 6 .

Our models do not differ from semi-supervised learning baselines.

This result can be understood because this semi-supervised learning assumes that label information is helpful or necessary to generate proper sentences.

Our experiments show that our model both is conditioned by the encoder and also generates proper sentences without labels.

This is consistent with the reasoning from BID30 that the best models for language modeling and semi-supervised learning are different.

We show additional samples for Table 3 in Table 7 .

Please see what happens to the death penalty for a day ?

SA is it true that australia likes war to update their new improve weapons ?

BoW definition of traditional education the death penalty and violence in a community is it true that a good alternative to get a degree of education ?is it possible to death penalty in the world to death penalty ? is it true that the age of people to change lots of traditional ?is there a place to get a peaceful death penalty in the world ? is it a good idea of having a 100 % of education ?are there any place in the city to make a death penalty ?

Table 7 : Sampling from posterior distribution of our model when different texts are input to selfattention and Bag-of-Words of the encoder.

"SA" is a sentence given to self-attention encoder and "BoW" is a sentence to Bag-of-Words encoder.

For detail, see Chapter 4.4.

1 is it true that the only way to provide the holy rabbit through the same answer ? is it possible to go to a police officer ?

is it possible to have to pay for her home do n't you think the president is the worst president of the us is the liberal , the jewish religion will be cut the food to do ?

who is the president of the united states ?

Table 8 : Samples from components of prior distribution.

Component 1, 22, and 76 generate sentences with common structure.

On the other hand, component 60, 68, and 83 generate structurally diverse sentences on the same topics (computer, sports).

<UNK>is a word not in the dictionary.

For detail, see Chapter 4.5.

We report text from 6 components of a multimodal prior distribution from our model in Table 8 .

We found two types of features allocated for components.

The first one is grammatical structure.

Components 1, 22, and 77 in Table 8 each generate similarly structured sentences: sentences from component 1 begin with "it is true that" or "it is possible to", sentences from component 22 begin with "does anyone (anybody)", and sentences from component 77 begin with "what is the best way to".

This result is straightforward to interpret as properly acquiring grammatical structure will lower reconstruction loss.

More interestingly, sentences generated the next type of components, namely components 60, 68, and 83 are each on the same topic.

Sentences generated from component 60 are about sports, those from component 68 are about computer (music), and those from component 83 are about politics.

However, these sentences do not share grammatical structure and generate sentences with diverse structures.

<|TLDR|>

@highlight

We propose a model of variational autoencoders for text modeling without weakening the decoder, which improves the quality of text generation and interpretability of acquired representations.