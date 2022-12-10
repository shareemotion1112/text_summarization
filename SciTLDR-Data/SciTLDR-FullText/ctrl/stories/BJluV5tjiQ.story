We propose a novel yet simple neural network architecture for topic modelling.

The method is based on training an autoencoder structure where the bottleneck represents the space of the topics distribution and the decoder outputs represent the space of the words distributions over the topics.

We exploit an auxiliary decoder to prevent mode collapsing in our model.

A key feature for an effective topic modelling method is having sparse topics and words distributions, where there is a trade-off between the sparsity level of topics and words.

This feature is implemented in our model by L-2 regularization and the model hyperparameters take care of the trade-off.

We show in our experiments that our model achieves competitive results compared to the state-of-the-art deep models for topic modelling, despite its simple architecture and training procedure.

The “New York Times” and “20 Newsgroups” datasets are used in the experiments.

Topic models are among the key models in Natural Language Processing (NLP) that aim to represent 14 a large body of text using only a few concepts or topics, on a completely unsupervised basis.

Topic 15 modeling has found its application in many different areas including bioinformatics [13] Sample a word in position n, w n ∼ Multinomial(β z )

Two important objectives that LDA implicitly tries to achieve and they make this model suitable for

There is a trade-off between these two objectives.

If a document is represented using only a few 61 topics, then number of words with high probability in those topics should be large, and if topics 62 are represented using only a few words then we need a large number of topics to cover the words 63 in the document.

The sparsity of the distributions is a property of the Dirichlet distribution that is 64 controlled by its concentration parameters.

Also, based on LDA, the distribution of the words in a 65 document is a mixture of multinomials.

In our model we follow the main principals of the LDA algorithm, i.e. sparse distributions for the 67 topics and words and the final distribution of the words in a document is a mixture of multinomial.

On the other hand, we try to avoid the difficulties of training the LDA model.

Since our downstream 69 task is finding topics in the documents, and not generating new documents, we do not need to 70 learn the true posterior probability, or find ways to approximate it.

Therefore we leave the latent 71 representation unconstrained with regard to its distribution.

We first encode the documents to the topic space Z using f topic (x; φ), which is implemented by 73 a neural network with parameter set φ.

To make sure Z is a probability space we use a softmax z k β k , will be a reconstruction of the input vector x. We 77 intentionally do not use a matrix multiplication notation so that we can explain the constraints on 78 β k 's in a simpler and more explicit way.

To make both topic and words distributions sparse, we impose an L-2 norm constraint on them.

Maximizing the L-2 norm over a positive, sum-to-one vector, concentrates the probability mass over of the algorithm will be as follows: DISPLAYFORM0 where distance D is the cross entropy, and γ and η are hyperparameters of the model.

The trade-off 85 of the sparsity in topics and words distributions can be controlled by tuning γ and η.

We observed that training the model using Eq. (1) causes mode collapsing, in the sense that only a 88 very few topics will have meaningful words in them and the rest of the topics have high probability 89 over some random words.

Also, all the probability mass of the topics distribution for all of the 90 documents are concentrated on those specific topics.

In other words, all the documents are encoded 91 to the same set of topics and the model cannot capture the variations in the documents.

We believe 92 this is due to the fact that f β (z) is not a powerful function for backpropagating the error signal from 93 the output to the previous layers of the network.

To resolve this issue and produce a richer Z space,

we attach an auxiliary decoder to the latent representation, which we call it f AUX (z; ϕ) and it is a 95 neural network with parameter set ϕ. The output of this decoder, denoted byx, also reconstructs the 96 input document.

Our observations show that by adding this decoder we can separate the documents' 97 representations in the latent space.

In both topic and word level, instead of sampling, we consider z and β k 's (for all k ∈ {1, 2, .., K})

as a normalized typical set of the distribution z and β k '

s. This is to avoid sampling from the multi-100 nomial distribution for which there is no easy way, e.g. reparameteraztion trick in [7] for Gaussian 101 family, to backpropagate the error for training the neural networks.

Therefore the overall objective 102 of our model is: DISPLAYFORM0 where λ is another hyperparameter of the model that controls the role of the auxiliary decoder in 104 training.

FIG2 shows the structure of the networks.

In this section we compare the performance of the proposed algorithm with LDA with collapsed coherence (higher is better) and perplexity score (lower is better) of the results.

This dataset doesn't need a preprocessing phase, as the common words and stop words has already 115 been removed from it.

We try performing topic modeling using 25 and 50 topics for this dataset (CG 116 in the tables mean Collapsed Gibbs and the best results are indicated by bold symbols).

The 20 Newsgroups has D = 11, 000 training documents.

We follow the same preprocessing in

[14], tokenization, removing some of the non UTF-8 characters and English stop word removal.

These are all done using scikit-learn package.

After this preprocessing the vocabulary size is 123 N = 2, 000.

For this dataset we try training the models with 50 and 200 topics.

<|TLDR|>

@highlight

A deep model for topic modelling