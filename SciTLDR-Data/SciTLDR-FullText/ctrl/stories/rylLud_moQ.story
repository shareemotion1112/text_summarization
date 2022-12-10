In the past few years, various advancements have been made in generative models owing to the formulation of Generative Adversarial Networks (GANs).

GANs have been shown to perform exceedingly well on a wide variety of tasks pertaining to image generation and style transfer.

In the field of Natural Language Processing, word embeddings such as word2vec and GLoVe are state-of-the-art methods for applying neural network models on textual data.

Attempts have been made for utilizing GANs with word embeddings for text generation.

This work presents an approach to text generation using Skip-Thought sentence embeddings in conjunction with GANs based on gradient penalty functions and f-measures.

The results of using sentence embeddings with GANs for generating text conditioned on input information are comparable to the approaches where word embeddings are used.

Numerous efforts have been made in the field of natural language text generation for tasks such as sentiment analysis BID35 and machine translation BID7 BID24 .

Early techniques for generating text conditioned on some input information were template or rule-based engines, or probabilistic models such as n-gram.

In recent times, state-of-the-art results on these tasks have been achieved by recurrent BID23 BID20 and convolutional neural network models trained for likelihood maximization.

This work proposes an Code available at: https://github.com/enigmaeth/skip-thought-gan approach for text generation using Generative Adversarial Networks with Skip-Thought vectors.

GANs BID9 are a class of neural networks that explicitly train a generator to produce high-quality samples by pitting against an adversarial discriminative model.

GANs output differentiable values and hence the task of discrete text generation has to use vectors as differentiable inputs.

This is achieved by training the GAN with sentence embedding vectors produced by Skip-Thought , a neural network model for learning fixed length representations of sentences.

Deep neural network architectures have demonstrated strong results on natural language generation tasks BID32 .

Recurrent neural networks using combinations of shared parameter matrices across time-steps BID30 BID20 BID3 with different gating mechanisms for easing optimization BID13 BID3 have found some success in modeling natural language.

Another approach is to use convolutional neural networks that reuse kernels across time-steps with attention mechanism to perform language generation tasks BID15 .Supervised learning with deep neural networks in the framework of encoder-decoder models has become the state-of-the-art methods for approaching NLP problems (Young et al.) .

Stacked denoising autoencoder have been used for domain adaptation in classifying sentiments BID8 and combinatory categorical autoencoders demonstrate learning the compositionality of sentences BID12 .

Recent text generation models use a wide variety of GANs such as gradient policy based sequence generation framework BID34 BID6 for performing natural language generation tasks.

Other architectures such as those proposed in with RNN and variational auto-encoder generator with CNN discriminator and in BID11 with leaky discriminator to guide generator through high-level extracted features have also shown great results.

This section introduces Skip-Thought Generative Adversarial Network with a background on models that it is based on.

The Skip-Thought model induces embedding vectors for sentences present in training corpus.

These vectors constitute the real distribution for the discriminator network.

The generator network produces sentence vectors similar to those from the encoded real distribution.

The generated vectors are sampled over training and decoded to produce sentences using a Skip-Thought decoder conditioned on the same text corpus.

Skip-Thought is an encoder-decoder framework with an unsupervised approach to train a generic, distributed sentence encoder.

The encoder maps sentences sharing semantic and syntactic properties to similar vector representations and the decoder reconstructs the surrounding sentences of an encoded passage.

The sentence encoding approach draws inspiration from the skip-gram model in producing vector representations using previous and next sentences.

The Skip-Thought model uses an RNN encoder with GRU activations BID4 and an RNN decoder with conditional GRU, the combination being identical to the RNN encoder-decoder of used in neural machine translation.

For a given sentence tuple (s i−1 , s i , s i+1 ), let w t i denote the t-th word for sentence s i , and let x t i denote its word embedding.

The model has three components: Encoder.

Encoded vectors for a sentence s i with N words w i , w i+1 ,...,w n are computed by iterating over the following sequence of equations: DISPLAYFORM0 Decoder.

A neural language model conditioned on the encoder output h i serves as the decoder.

Bias matrices C z , C r , C are introduced for the update gate, reset gate and hidden state computation by the encoder.

Two decoders are used in parallel, one each for sentences s i + 1 and s i − 1.

The following equations are iterated over for decoding: DISPLAYFORM1 Objective.

For the same tuple of sentences, objective function is the sum of log-probabilities for the forward and backward sentences conditioned on the encoder representation: DISPLAYFORM2

Generative Adversarial Networks BID9 are deep neural net architectures comprised of two networks, contesting with each other in a zero-sum game framework.

For a given data, GANs can mimic learning the underlying distribution and generate artificial data samples similar to those from the real distribution.

Generative Adversarial Networks consists of two players -a Generator and a Discriminator.

The generator G tries to produce data close to the real distribution P (x) from some stochastic distribution P (z) termed as noise.

The discriminator D's objective is to differentiate between real and generated data G(z).The two networks -generator and discriminator compete against each other in a zero-sum game.

The minimax strategy dictates that each network plays optimally with the assumption that the other network is optimal.

This leads to Nash equilibrium which is the point of convergence for GAN model.

Objective.

BID9 have formulated the minimax game for a generator G, discriminator D adversarial network with value function V (G, D) as: DISPLAYFORM0

The STGAN model uses a deep convolutional generative adversarial network, similar to the one used in (Radford et al.) .

The generator network is updated twice for each discriminator network update to prevent fast convergence of the discriminator network.

The Skip-Thought encoder for the model encodes sentences with length less than 30 words using 2400 GRU units BID4 with word vector dimensionality of 620 to produce 4800-dimensional combineskip vectors. .

The combine-skip vectors, with the first 2400 dimensions being uni-skip model and the last 2400 bi-skip model, are used as they have been found to be the best performing in the experiments 1 .

The decoder uses greedy decoding taking argmax over softmax output distribution for given time-step which acts as input for next time-step.

It reconstructs sentences conditioned on a sentence vector by randomly sampling from the predicted distributions with or without a preset beam width.

Unknown tokens are not included in the vocabulary.

A 620 dimensional RNN word embeddings is used with 1600 hidden GRU decoding units.

Gradient clipping with Adam optimizer BID16 ) is used, with a batch size of 16 and maximum sentence length of 100 words for decoder.

The training process of a GAN is notably difficult BID28 and several improvement techniques such as batch normalization, feature matching, historical averaging BID28 and unrolling GAN (Metz et al.) have been suggested for making the training more stable.

Training the Skip-Thought GAN often results in mode dropping (Arjovsky & Bottou; Srivastava et al.) with a parameter setting where it outputs a very narrow distribution of points.

To overcome this, it uses minibatch discrimination by looking at an entire batch of samples and modeling the distance between a given sample and all the other samples present in that batch.

The minimax formulation for an optimal discriminator in a vanilla GAN is Jensen-Shannon Distance between the generated distribution and the real distribution.

used Wasserstein distance or earth mover's distance to demonstrate how replacing distance measures can improve training loss for GAN.

BID10 have incorporated a gradient penalty regularizer in WGAN objective for discriminator's loss function.

The experiments in this work use the above f-measures to improve performance of Skip-Thought GAN on text generation.

GANs can be conditioned on data attributes to generate samples BID21 BID25 .

In this experiment, both the generator and discriminator are conditioned on Skip-Thought encoded vectors .

The encoder converts 70000 sentences from the BookCorpus dataset with a training/test/validation split of 5/1/1 into vectors used as real samples for discriminator.

The decoded sentences are used to evaluate model performance under corpus level BLEU-2, BLEU-3 and BLEU-4 metrics (Papineni et al.) , once using only test set as reference and then entire corpus as reference.

i can n't see some shopping happened .

i had a police take watch out of my wallet .get him my camera found a person 's my watch .

here i collect my telephone card and telephone number delta airlines flight six zero two from six p.m. to miami, please?

Table 2 .

Sample sentences generated from training on CMU-SE Dataset; mode collapse is overcome by using minibatch discrimination.

Formation of sentences further improved by changing f-measure to Wasserstein distance along with gradient penalty regularizer.

Language generation is done on a dataset comprising simple English sentences referred to as CMU-SE 2 in BID26 .

The CMU-SE dataset consists of 44,016 sentences with a vocabulary of 3,122 words.

For encoding, the vectors are extracted in batches of sentences having the same length.

The samples represent how mode collapse is manifested when using least-squares distance BID18 f-measure without minibatch discrimination.

Table 2(a) contains sentences generated from STGAN using least-squares distance BID18 in which there was no mode collapse observed, while 2(b) contains examples wherein it is observed.

Table 2(c) shows generated sentences using gradient penalty regularizer(GAN-GP).

Table 2 (d) has samples generated from STGAN when using Wasserstein distance f-measure as WGAN ) and 2(e) contains samples when using a gradient penalty regularizer term as WGAN-GP BID10 .

Another performance metric that can be computed for this setup has been described in BID26 which is a parallel work to this.

Simple CFG 3 and more complex ones like Penn Treebank CFG generate samples BID5 which are used as input to GAN and the model is evaluated by computing the diversity and accuracy of generated samples conforming to the given CFG.Skip-Thought sentence embeddings can be used to generate images with GANs conditioned on text vectors for text-to-image conversion tasks like those achieved in BID27 BID2 .

These embeddings have also been used to Models like neuralstoryteller 4 which use these sentence embeddings can be experimented with generative adversarial networks to generate unique samples.

<|TLDR|>

@highlight

Generating text using sentence embeddings from Skip-Thought Vectors with the help of Generative Adversarial Networks.