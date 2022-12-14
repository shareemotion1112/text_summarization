While autoencoders are a key technique in representation learning for continuous structures, such as images or wave forms, developing general-purpose autoencoders for discrete structures, such as text sequence or discretized images, has proven to be more challenging.

In particular, discrete inputs make it more difficult to learn a smooth encoder that preserves the complex local relationships in the input space.

In this work, we propose an adversarially regularized autoencoder (ARAE) with the goal of learning more robust discrete-space representations.

ARAE jointly trains both a rich discrete-space encoder, such as an RNN, and a simpler continuous space generator function, while using generative adversarial network (GAN) training to constrain the distributions to be similar.

This method yields a smoother contracted code space that maps similar inputs to nearby codes, and also an implicit latent variable GAN model for generation.

Experiments on text and discretized images demonstrate that the GAN model produces clean interpolations and captures the multimodality of the original space, and that the autoencoder produces improvements in semi-supervised learning as well as state-of-the-art results in unaligned text style transfer task using only a shared continuous-space representation.

Recent work on regularized autoencoders, such as variational BID15 BID29 and denoising BID37 variants, has shown significant progress in learning smooth representations of complex, high-dimensional continuous data such as images.

These codespace representations facilitate the ability to apply smoother transformations in latent space in order to produce complex modifications of generated outputs, while still remaining on the data manifold.

Unfortunately, learning similar latent representations of discrete structures, such as text sequences or discretized images, remains a challenging problem.

Initial work on VAEs for text has shown that optimization is difficult, as the decoder can easily degenerate into a unconditional language model BID2 .

Recent work on generative adversarial networks (GANs) for text has mostly focused on getting around the use of discrete structures either through policy gradient methods BID40 or with the Gumbel-Softmax distribution BID17 .

However, neither approach can yet produce robust representations directly.

A major difficulty of discrete autoencoders is mapping a discrete structure to a continuous code vector while also smoothly capturing the complex local relationships of the input space.

Inspired by recent work combining pretrained autoencoders with deep latent variable models, we propose to target this issue with an adversarially regularized autoencoder (ARAE).

Specifically we jointly train a discrete structure encoder and continuous space generator, while constraining the two models with a discriminator to agree in distribution.

This approach allows us to utilize a complex encoder model, such as an RNN, and still constrain it with a very flexible, but more limited generator distribution.

The full model can be then used as a smoother discrete structure autoencoder or as a latent variable GAN model where a sample can be decoded, with the same decoder, to a discrete output.

Since the system produces a single continuous coded representation-in contrast to methods that act on each RNN state-it can easily be further regularized with problem-specific invariants, for instance to learn to ignore style, sentiment or other attributes for transfer tasks.

Experiments apply ARAE to discretized images and sentences, and demonstrate that the key properties of the model.

Using the latent variable model (ARAE-GAN), the model is able to generate varied samples that can be quantitatively shown to cover the input spaces and to generate consistent image and sentence manipulations by moving around in the latent space via interpolation and offset vector arithmetic.

Using the discrete encoder, the model can be used in a semi-supervised setting to give improvement in a sentence inference task.

When the ARAE model is trained with task-specific adversarial regularization, the model improves the current best results on sentiment transfer reported in BID33 and produces compelling outputs on a topic transfer task using only a single shared code space.

All outputs are listed in the Appendix 9 and code is available at (removed for review).

In practice unregularized autoencoders often learn a degenerate identity mapping where the latent code space is free of any structure, so it is necessary to apply some method of regularization.

A popular approach is to regularize through an explicit prior on the code space and use a variational approximation to the posterior, leading to a family of models called variational autoencoders (VAE) BID15 BID29 .

Unfortunately VAEs for discrete text sequences can be challenging to train-for example, if the training procedure is not carefully tuned with techniques like word dropout and KL annealing BID2 , the decoder simply becomes a language model and ignores the latent code (although there has been some recent successes with convolutional models BID32 BID39 ).

One possible reason for the difficulty in training VAEs is due to the strictness of the prior (usually a spherical Gaussian) and/or the parameterization of the posterior.

There has been some work on making the prior/posterior more flexible through explicit parameterization BID28 BID16 BID4 .

A notable technique is adversarial autoencoders (AAE) BID23 which attempt to imbue the model with a more flexible prior implicitly through adversarial training.

In AAE framework, the discriminator is trained to distinguish between samples from a fixed prior distribution and the input encoding, thereby pushing the code distribution to match the prior.

While this adds more flexibility, it has similar issues for modeling text sequences and suffers from mode-collapse in our experiments.

Our approach has similar motivation, but notably we do not sample from a fixed prior distribution-our 'prior' is instead parameterized through a flexible generator.

Nonetheless, this view (which has been observed by various researchers BID35 BID24 BID22 ) provides an interesting connection between VAEs and GANs.

The success of GANs on images have led many researchers to consider applying GANs to discrete data such as text.

Policy gradient methods are a natural way to deal with the resulting non-differentiable generator objective when training directly in discrete space BID7 BID38 .

When trained on text data however, such methods often require pre-training/co-training with a maximum likelihood (i.e. language modeling) objective BID40 .

This precludes there being a latent encoding of the sentence, and is also a potential disadvantage of existing language models (which can otherwise generate locally-coherent samples).

Another direction of work has been through reparameterizing the categorical distribution with the Gumbel-Softmax trick BID13 BID21 )-while initial experiments were encouraging on a synthetic task BID17 , scaling them to work on natural language is a challenging open problem.

There has also been a flurry of recent, related approaches that work directly with the soft outputs from a generator BID9 Sai Rajeswar, 2017; BID33 BID26 .

For example, Shen et al. BID33 exploits adversarial loss for unaligned style transfer between text by having the discriminator act on the RNN hidden states and using the soft outputs at each step as input to an RNN generator, utilizing the Professor-forcing framework BID18 .

Our approach instead works entirely in code space and does not require utilizing RNN hidden states directly.

Discrete Structure Autoencoders Define X = V n to be a set of discrete structures where V is a vocabulary of symbols and P x to be a distribution over this space.

For instance, for binarized images V = {0, 1} and n is the number of pixels, while for sentences V is the vocabulary and n is the sentence length.

A discrete autoencoder consists of two parameterized functions: a deterministic encoder function enc ?? : X ??? C with parameters ?? that maps from input to code space and a conditional decoder distribution p ?? (x | c) over structures X with parameters ??.

The parameters are trained on a cross-entropy reconstruction loss: DISPLAYFORM0 The choice of the encoder and decoder parameterization is specific to the structure of interest, for example we use RNNs for sequences.

We use the notation,x = arg max x p ?? (x | enc ?? (x)) for the (approximate) decoder mode.

When x =x the autoencoder is said to perfectly reconstruct x.

Generative Adversarial Networks GANs are a class of parameterized implicit generative models BID8 .

The method approximates drawing samples from a true distribution c ??? P r by instead employing a latent variable z and a parameterized deterministic generator functio?? c = g ?? (z) to produce samplesc ??? P g .

Initial work on GANs minimizes the Jensen-Shannon divergence between the distributions.

Recent work on Wasserstein GAN (WGAN) , replaces this with the Earth-Mover (Wasserstein-1) distance.

GAN training utilizes two separate models: a generator g ?? (z) maps a latent vector from some easy-to-sample source distribution to a sample and a critic/discriminator f w (c) aims to distinguish real data and generated samples from g ?? .

Informally, the generator is trained to fool the critic, and the critic to tell real from generated.

WGAN training uses the following min-max optimization over generator parameters ?? and critic parameters w, DISPLAYFORM1 where f w : C ??? R denotes the critic function,c is obtained from the generator,c = g ?? (z), and P r and P g are real and generated distributions.

If the critic parameters w are restricted to an 1-Lipschitz function set W, this term correspond to minimizing Wasserstein-1 distance W (P r , P g ).

We use a naive approximation to enforce this property by weight-clipping, i.e. .

DISPLAYFORM2

Ideally, a discrete autoencoder should be able to reconstruct x from c, but also smoothly assign similar codes c and c to similar x and x .

For continuous autoencoders, this property can be enforced directly through explicit regularization.

For instance, contractive autoencoders BID30 regularize their loss by the functional smoothness of enc ?? .

However, this criteria does not apply when inputs are discrete and we lack even a metric on the input space.

How can we enforce that similar discrete structures map to nearby codes?Adversarially regularized autoencoders target this issue by learning a parallel continuous-space generator with a restricted functional form to act as a smoother reference encoding.

The joint objective regularizes the autoencoder to constrain the discrete encoder to agree in distribution with its continuous counterpart: DISPLAYFORM0 Above W is the Wasserstein-1 distance between P r the distribution of codes from the discrete encoder model (enc ?? (x) where x ??? P(x)) and P g is the distribution of codes from the continuous generator model (g ?? (z) for some z, e.g. z ??? N (0, I)).

To approximate Wasserstein-1 term, the W function includes an embedded critic function which is optimized adversarially to the encoder and generator as described in the background.

The full model is shown in Figure 1 .To train the model, we use a block coordinate descent to alternate between optimizing different parts of the model: (1) the encoder and decoder to minimize reconstruction loss, (2) the WGAN critic function to approximate the W term, (3) the encoder and generator to adversarially fool the critic to minimize W : DISPLAYFORM1 The full training algorithm is shown in Algorithm 1.

discrete struct.

encoder code (P r ) decoder reconstruction loss DISPLAYFORM2 Figure 1: ARAE architecture.

The model can be used as an autoencoder, where a structure x is encoded and decoded to producex, and as a GAN (ARAE-GAN), where a sample z is passed though a generator g ?? to produce a code vector, which is similarly decoded tox.

The critic function fw is only used at training to help approximate W .

for number of training iterations do (1) Train the autoencoder for reconstruction DISPLAYFORM0 Backpropagate reconstruction loss, DISPLAYFORM1 , and update.

Sample DISPLAYFORM2 DISPLAYFORM3 Backpropagate adversarial loss DISPLAYFORM4 ) and update.

Extension: Code Space Transfer One benefit of the ARAE framework is that it compresses the input to a single code vector.

This framework makes it ideal for manipulating discrete objects while in continuous code space.

For example, consider the problem of unaligned transfer, where we want to change an attribute of a discrete input without supervised examples, e.g. to change the topic or sentiment of a sentence.

First, we extend the decoder to condition on a transfer variable denoting this attribute y which is known during training, to learn p ?? (x | c, y).

Next, we train the code space to be invariant to this attribute, to force it to be learned fully by the decoder.

Specifically, we further regularize the code space to map similar x with different attribute labels y near enough to fool a code space attribute classifier, i.e.: DISPLAYFORM5 where L class (??, u) is the loss of a classifier p u (y | c) from code space to labels (in our experiments we always set ?? (2) = 1).

To incorporate this additional regularization, we simply add two more gradient update steps: (2b) training a classifier to discriminate codes, and (3b) adversarially training the encoder to fool this classifier.

The algorithm is shown in Algorithm 2.

Note that similar technique has been introduced in other domains, notably in images BID19 and video modeling BID6 .

We experiment with three different ARAE models: (1) an autoencoder for discretized images trained on the binarized version of MNIST, (2) an autoencoder for text sequences trained using the Stanford Natural Language Inference (SNLI) corpus BID1 , and (3) an autoencoder trained DISPLAYFORM0 , and compute code-vectors c DISPLAYFORM1 Backpropagate adversarial classifier loss DISPLAYFORM2 for text transfer (Section 6.2) based on the Yelp and Yahoo datasets for unaligned sentiment and topic transfer.

All three models utilize the same generator architecture, g ?? .

The generator architecture uses a low dimensional z with a Gaussian prior p(z) = N (0, I), and maps it to c. Both the critic f w and the generator g ?? are parameterized as feed-forward MLPs.

The image model uses fully-connected NN to autoencode binarized images.

Here X = {0, 1} n where n is the image size.

The encoder used is a feed-forward MLP network mapping from {0, DISPLAYFORM3 The text model uses a recurrent neural network (RNN) for both the encoder and decoder.

Here X = V n where n is the sentence length and V is the vocabulary of the underlying language.

Define an RNN as a parameterized recurrent function h j = RNN(x j , h j???1 ; ??) for j = 1 . . .

n (with h 0 = 0) that maps a discrete input structure x to hidden vectors h 1 . . .

h n .

For the encoder, we define enc ?? (x) = h n = c. For decoding we feed c as an additional input to the decoder RNN at each time step, i.e.h j = RNN(x j ,h j???1 , c; ??), and further calculate the distribution over V at each time step via softmax, p ?? (x | c) = n j=1 softmax(Wh j + b) xj where W and b are parameters (part of ??).

Finding the most likely sequencex under this distribution is intractable, but it is possible to approximate it using greedy search or beam search.

In our experiments we use an LSTM architecture BID12 for both the encoder/decoder and decode using greedy search.

The text transfer model uses the same architecture as the text model but extends it with a code space classifier p(y|c) which is modeled using an MLP and trained to minimize cross-entropy.

Our baselines utilize a standard autoencoder (AE) and the cross-aligned autoencoder BID33 for transfer.

Note that in both our ARAE and standard AE experiments, the encoded code from the encoder is normalized to lie on the unit sphere, and the generated code is bounded to lie in (???1, 1) n by the tanh function at output layer.

We additionally experimented with the sequence VAE introduced by BID2 and the adversarial autoencoder (AAE) model BID23 on the SNLI dataset.

However despite extensive parameter tuning we found that neither model was able to learn meaningful latent representations-the VAE simply ignored the latent code and the AAE experienced mode-collapse and repeatedly generated the same samples.

The Appendix 12 includes detailed descriptions of the hyperparameters, model architecture, and training regimes.

Our experiments consider three aspects of the model.

First we measure the empirical impact of regularization on the autoencoder.

Next we apply the discrete autoencoder to two applications, unaligned style transfer and semi-supervised learning.

Finally we employ the learned generator network as an implicit latent variable model (ARAE-GAN) over discrete sequences.

Our main goal for ARAE is to regularize the model produce a smoother encoder by requiring the distribution from the encoder to match the distribution from the continuous generator over a simple latent variable.

To examine this claim we consider two basic statistical properties of the code space during training of the text model on SNLI, shown in FIG1 .

On the left, we see that the 2 norm of c and codec converge quickly in ARAE training.

The encoder code is always restricted to be on the unit sphere, and the generated codec quickly learns to match it.

The middle plot shows the convergence of the trace of the covariance matrix between the generator and the encoder as training progresses.

We find that variance of the encoder and the generator match after several epochs.

To check the smoothness of the model, for both ARAE/AE, we take a sentence and calculate the average cosine similarity of 100 randomly-selected sentences that had an edit-distance of at most 5 to the original sentence.

We do this for 250 sentences and calculate the mean of the average cosine similarity.

FIG1 (right) shows that the cosine similarity of nearby sentences is quite high for the ARAE than in the case for the AE.

Edit-distance is not an ideal proxy for similarity in sentences, but it is often a sufficient condition.

Finally an ideal representation should be robust to small changes of the input around the training examples in code space BID30 .

We can test this property by feeding a noised input to the encoder and (i) calculating the score given to the original input, and (ii) checking the reconstructions.

TAB2 (right) shows an experiment for text where we add noise by permuting k words in each sentence.

We observe that the ARAE is able to map a noised sentence to a natural sentence, (though not necessarily the denoised sentence).

TAB2 (left) shows empirical results for these experiments.

We obtain the reconstruction error (i.e. negative log likelihood) of the original (non-noised) sentence under the decoder, utilizing the noised code.

We find that when k = 0 (i.e. no swaps), the regular AE better reconstructs the input as expected.

However, as we increase the number of swaps and push the input further away from the data manifold, the ARAE is more likely to produce the original sentence.

We note that unlike denoising autoencoders which require a domain-specific noising function BID10 BID37 , the ARAE is not explicitly trained to denoise an input, but learns to do so as a byproduct of adversarial regularization.

Unaligned Text Transfer A smooth autoencoder combined with low reconstruction error should make it possible to more robustly manipulate discrete objects through code space without dropping off the data manifold.

To test this hypothesis, we experimented with two unaligned text transfer tasks.

For these tasks, we attempt to change one attribute of a sentence without aligned examples of this change.

To perform this transfer, we learn a code space that can represent an input that is agnostic to this attribute, and a decoder that can incorporate the attribute (as described in Section 4).

We experiment with unaligned transfer of sentiment on the Yelp corpus and topic on the Yahoo corpus BID41 .

we came on the recommendation of a bell boy and the food was amazing .

the people who ordered off the menu did n't seem to do much better .

ARAE we came on the recommendation and the food was a joke .

ARAE the people who work there are super friendly and the menu is good .

Cross-AE we went on the car of the time and the chicken was awful .Cross-AE the place , one of the office is always worth you do a business .

For sentiment we follow the same setup as BID33 and split the Yelp corpus into two sets of unaligned positive and negative reviews.

We train an ARAE as an autoencoder with two separate decoders, one for positive and one for negative sentiment, and incorporate adversarial training of the encoder to remove sentiment information from the code space.

We test by encoding in sentences of one class and decoding, greedily, with the opposite decoder.

Our evaluation is based on four automatic metrics, shown in Table 2 : (i) Transfer: measuring how successful the model is at transferring sentiment based on an automatic classifier (we use the fastText library BID14 ).(ii) BLEU: measuring the consistency between the transferred text and the original.

We expect the model to maintain as much information as possible and transfer only the style; (iii) Perplexity: measuring the fluency of the generated text; (iv) Reverse Perplexity: measuring the extent to which the generations are representative of the underlying data distribution.

1 Both perplexity numbers are obtained by training an RNN language model.

We additionally perform human evaluations on the cross-aligned AE and our best ARAE model.

We randomly select 1000 sentences (500/500 positive/negative), obtain the corresponding transfers from both models, and ask Amazon Mechanical Turkers to evaluate the sentiment (Positive/Neutral/Negative) and naturalness (1-5, 5 being most natural) of the transferred sentences.

We create a separate task in which we show the Turkers the original and the transferred sentences, and ask them to evaluate the similarity based on sentence structure (1-5, 5 being most similar).

We explicitly ask the Turkers to disregard sentiment in their similarity assessment.

In addition to comparing against the cross-aligned AE of BID33 , we also compare against a vanilla AE trained without adversarial regularization.

For ARAE, we experimented with different ?? (1) weighting on the adversarial loss (see section 4) with ??(1) a = 1, ??(1) b = 10.

We generally set ?? (2) = 1.

Experimentally the adversarial regularization enhances transfer and perplexity, but tends to make the transferred text less similar to the original, compared to the AE.

Some randomly selected sentences are shown in figure 6 and more samples are shown available in Appendix 9.The same method can be applied to other style transfer tasks, for instance the more challenging Yahoo QA data BID41 Semi-Supervised Training We further utilize ARAE in a standard AE setup for semi-supervised training.

We experiment on a natural language inference task, shown in Table 5 (right).

We use 22.2%, 10.8% and 5.25% of the original labeled training data, and use the rest of the training set for unlabeled training.

The labeled set is randomly picked.

The full SNLI training set contains 543k sentence pairs, and we use supervised sets of 120k, 59k and 28k sentence pairs respectively for the three settings.

As a baseline we use an AE trained on the additional data, similar to the setting explored in BID5 .

For ARAE we use the subset of unsupervised data of length < 15, which roughly includes 655k single sentences (due to the length restriction, this is a subset of 715k sentences that were used for AE training).

As observed by BID5 , training on unlabeled data with an AE objective improves upon a model just trained on labeled data.

Training with adversarial regularization provides further gains.

After training, an ARAE can also be used as an implicit latent variable model controlled by z and the generator g ?? , which we refer to as ARAE-GAN.

While models of this form have been widely used for generation in other modalities, they have been less effective for discrete structures.

In this section, we attempt to measure the effectiveness of this induced discrete GAN.A common test for a GANs ability mimic the true distribution P r is to train a simple model on generated samples from P g .

While there are pitfalls of this evaluation BID34 , it provides a starting point for text modeling.

Here we generate 100k samples from (i) ARAE-GAN, (ii) an AE 2 , (iii) a RNN LM trained on the same data, and (iv) the real training set (samples from the models are 2 To "sample" from an AE we fit a multivariate Gaussian to the code space after training and generate code vectors from this Gaussian to decode back into sentence space.

Medium Table 5 : Left.

Semi-Supervised accuracy on the natural language inference (SNLI) test set, respectively using 22.2% (medium), 10.8% (small), 5.25% (tiny) of the supervised labels of the full SNLI training set (rest used for unlabeled AE training).

Right.

Perplexity (lower is better) of language models trained on the synthetic samples from a GAN/AE/LM, and evaluated on real data (Reverse PPL).A man is on the corner in a sport area .

A man is on corner in a road all .

A lady is on outside a racetrack .

A lady is outside on a racetrack .

A lot of people is outdoors in an urban setting .A lot of people is outdoors in an urban setting .A lot of people is outdoors in an urban setting .A man is on a ship path with the woman .

A man is on a ship path with the woman .

A man is passing on a bridge with the girl .

A man is passing on a bridge with the girl .

A man is passing on a bridge with the girl .

A man is passing on a bridge with the dogs .

A man is passing on a bridge with the dogs .A man in a cave is used an escalator .A man in a cave is used an escalator A man in a cave is used chairs .

A man in a number is used many equipment A man in a number is posing so on a big rock .People are posing in a rural area .

People are posing in a rural area.

Figure 3: Sample interpolations from the ARAE-GAN.

Constructed by linearly interpolating in the latent space and decoding to the output space.

Word changes are highlighted in black.

Results of the ARAE.

The top block shows output generation of the decoder taking fake hidden codes generated by the GAN; the bottom block shows sample interpolation results.

shown in Appendix 10).

All models are of the same size to allow for fair comparison.

We train an RNN language model on generated samples and evaluate on held-out data to calculate the reverse perplexity.

As can be seen from Table 5 , training on real data (understandably) outperforms training on generated data by a large margin.

Surprisingly however, we find that a language model trained on ARAE-GAN data performs slightly better than one trained on LM-generated/AE-generated data.

We further found that the reverse PPL of an AAE BID23 was quite high (980) due to mode-collapse.

Another property of GANs (and VAEs) is that the Gaussian form of z induces the ability to smoothly interpolate between outputs by exploiting the structure of the latent space.

While language models may provide a better estimate of the underlying probability space, constructing this style of interpolation would require combinatorial search, which makes this a useful feature of text GANs.

We experiment with this property by sampling two points z 0 and z 1 from p(z) and constructing intermediary points z ?? = ??z 1 + (1 ??? ??)z 0 .

For each we generate the argmax outputx ?? .

The samples are shown in FIG0 (left) for text and in FIG0 (right) for a discretized MNIST ARAE-GAN.A final intriguing property of image GANs is the ability to move in the latent space via offset vectors (similar to the case with word vectors BID25 ).

For example, Radford et al. BID27 observe that when the mean latent vector for "men with glasses" is subtracted from the mean latent vector for "men without glasses" and applied to an image of a "woman without glasses", the resulting image is that of a "woman with glasses".

To experiment with this property we generate 1 million sentences from the ARAE-GAN and compute vector transforms in this space to attempt to change main verbs, subjects and modifier (details in Appendix 11).

Some examples of successful transformations are shown in FIG2 (right).

Quantitative evaluation of the success of the vector transformations is given in FIG2 (left).

We present adversarially regularized autoencoders, as a simple approach for training a discrete structure autoencoder jointly with a code-space generative adversarial network.

The model learns a improved autoencoder as demonstrated by semi-supervised experiments and improvements on text transfer experiments.

It also learns a useful generative model for text that exhibits a robust latent space, as demonstrated by natural interpolations and vector arithmetic.

We do note that (as has been frequently observed when training GANs) our model seemed to be quite sensitive to hyperparameters.

Finally, while many useful models for text generation already exist, text GANs provide a qualitatively different approach influenced by the underlying latent variable structure.

We envision that such a framework could be extended to a conditional setting, combined with other existing decoding schemes, or used to provide a more interpretable model of language.

One can interpret the ARAE framework as a dual pathway network mapping two distinct distributions into a similar one; enc ?? and g ?? both output code vectors that are kept similar in terms of Wasserstein distance as measured by the critic.

We provide the following proposition showing that under our parameterization of the encoder and the generator, as the Wasserstein distance converges, the encoder distribution (c ??? P r ) converges to the generator distribution (c ??? P g ), and further, their moments converge.

This is ideal since under our setting the generated distribution is simpler than the encoded distribution, because the input to the generator is from a simple distribution (e.g. spherical Gaussian) and the generator possesses less capacity than the encoder.

However, it is not so simple that it is overly restrictive (e.g. as in VAEs).

Empirically we observe that the first and second moments do indeed converge as training progresses (Section 6.1).

Proposition 1.

Let P be a distribution on a compact set ?? , and (P n ) n???N be a sequence of distributions on ?? .

Further suppose that W (P n , P) ??? 0.

Then the following statements hold:(i) P n P (i.e. convergence in distribution).(ii) All moments converge, i.e. for all k > 1, k ??? N, DISPLAYFORM0 Proof.

(i) has been proved in BID36 Theorem 6.9.For (ii), using The Portmanteau Theorem, (i) is equivalent to: DISPLAYFORM1 for all bounded and continuous function f : R d ??? R, where d is the dimension of the random variable.

The k-th moment of a distribution is given by DISPLAYFORM2 Our encoded code is bounded as we normalize the encoder output to lie on the unit sphere, and our generated code is also bounded to lie in (???1, 1) n by the tanh function.

Hence Original definitely a great choice for sushi in las vegas !

Original i was so very disappointed today at lunch .

ARAE definitely a _num_ star rating for _num_ sushi in las vegas .

ARAE i highly recommend this place today .

Cross-AE not a great choice for breakfast in las vegas vegas !

Cross-AE i was so very pleased to this .

DISPLAYFORM3 Original the best piece of meat i have ever had !

Original i have n't received any response to anything .

ARAE the worst piece of meat i have ever been to !

ARAE i have n't received any problems to please .

Cross-AE the worst part of that i have ever had had !

Cross-AE i have always the desert vet .Original really good food , super casual and really friendly .

Original all the fixes were minor and the bill ?

ARAE really bad food , really generally really low and decent food .

ARAE all the barbers were entertaining and the bill did n't disappoint .

Cross-AE really good food , super horrible and not the price .

Cross-AE all the flavors were especially and one !

Original it has a great atmosphere , with wonderful service .

Original small , smokey , dark and rude management .

ARAE it has no taste , with a complete jerk .

ARAE small , intimate , and cozy friendly staff .

Cross-AE it has a great horrible food and run out service .

Cross-AE great , , , chips and wine .Original their menu is extensive , even have italian food .

Original the restaurant did n't meet our standard though .

ARAE their menu is limited , even if i have an option .

ARAE the restaurant did n't disappoint our expectations though .

Cross-AE their menu is decent , i have gotten italian food .

Cross-AE the restaurant is always happy and knowledge .Original everyone who works there is incredibly friendly as well .

Original you could not see the stage at all !

ARAE everyone who works there is incredibly rude as well .

ARAE you could see the difference at the counter !

Cross-AE everyone who works there is extremely clean and as well .

Cross-AE you could definitely get the fuss !

Original there are a couple decent places to drink and eat in here as well .

Original room is void of all personality , no pictures or any sort of decorations .

ARAE there are a couple slices of options and _num_ wings in the place .

ARAE room is eclectic , lots of flavor and all of the best .

Cross-AE there are a few night places to eat the car here are a crowd .

Cross-AE it 's a nice that amazing , that one 's some of flavor .Original if you 're in the mood to be adventurous , this is your place !

Original waited in line to see how long a wait would be for three people .

ARAE if you 're in the mood to be disappointed , this is not the place .

ARAE waited in line for a long wait and totally worth it .

Cross-AE if you 're in the drive to the work , this is my place !

Cross-AE another great job to see and a lot going to be from dinner .Original we came on the recommendation of a bell boy and the food was amazing .

Original the people who ordered off the menu did n't seem to do much better .

Cross-AE we came on the recommendation and the food was a joke .

ARAE the people who work there are super friendly and the menu is good .

Cross-AE we went on the car of the time and the chicken was awful .

Cross-AE the place , one of the office is always worth you do a business .Original service is good but not quick , just enjoy the wine and your company .

Original they told us in the beginning to make sure they do n't eat anything .

ARAE service is good but not quick , but the service is horrible .

ARAE they told us in the mood to make sure they do great food .

Cross-AE service is good , and horrible , is the same and worst time ever .

Cross-AE they 're us in the next for us as you do n't eat .Original the steak was really juicy with my side of salsa to balance the flavor .

Original the person who was teaching me how to control my horse was pretty rude .

ARAE the steak was really bland with the sauce and mashed potatoes .

ARAE the person who was able to give me a pretty good price .

Cross-AE the fish was so much , the most of sauce had got the flavor .

Cross-AE the owner 's was gorgeous when i had a table and was friendly .Original other than that one hell hole of a star bucks they 're all great !

Original he was cleaning the table next to us with gloves on and a rag .

ARAE other than that one star rating the toilet they 're not allowed .

ARAE he was prompt and patient with us and the staff is awesome .

Cross-AE a wonder our one came in a _num_ months , you 're so better !

Cross-AE he was like the only thing to get some with with my hair .

A woman is seeing a man in the river .

There passes a woman near birds in the air .

Some ten people is sitting through their office .

The man got stolen with young dinner bag .

Monks are running in court .

The Two boys in glasses are all girl .

The man is small sitting in two men that tell a children .

The two children are eating the balloon animal .

A woman is trying on a microscope .

The dogs are sleeping in bed .

Two Three woman in a cart tearing over of a tree .

A man is hugging and art .

The fancy skier is starting under the drag cup in .

A dog are <unk> a A man is not standing .

The Boys in their swimming .

A surfer and a couple waiting for a show .

A couple is a kids at a barbecue .

The motorcycles is in the ocean loading I 's bike is on empty The actor was walking in a a small dog area .

no dog is young their mother LM Samples a man walking outside on a dirt road , sitting on the dock .

A large group of people is taking a photo for Christmas and at night .

Someone is avoiding a soccer game .

The man and woman are dressed for a movie .

Person in an empty stadium pointing at a mountain .

Two children and a little boy are <unk> a man in a blue shirt .

A boy rides a bicycle .

A girl is running another in the forest .

the man is an indian women .Figure 5: Text samples generated from ARAE-GAN, a simple AE, and from a baseline LM trained on the same data.

To generate from an AE we fit a multivariate Gaussian to the learned code space and generate code vectors from this Gaussian.

We generate 1 million sentences from the ARAE-GAN and parse the sentences to obtain the main verb, subject, and modifier.

Then for a given sentence, to change the main verb we subtract the mean latent vector (t) for all other sentences with the same main verb (in the first example in FIG2 this would correspond to all sentences that had "sleeping" as the main verb) and add the mean latent vector for all sentences that have the desired transformation (with the running example this would be all sentences whose main verb was "walking").

We do the same to transform the subject and the modifier.

We decode back into sentence space with the transformed latent vector via sampling from p ?? (g(z + t)).

Some examples of successful transformations are shown in FIG2 (right).

Quantitative evaluation of the success of the vector transformations is given in FIG2 (left).

For each original vector z we sample 100 sentences from p ?? (g(z + t)) over the transformed new latent vector and consider it a match if any of the sentences demonstrate the desired transformation.

Match % is proportion of original vectors that yield a match post transformation.

As we ideally want the generated samples to only differ in the specified transformation, we also calculate the average word precision against the original sentence (Prec) for any match.

<|TLDR|>

@highlight

Adversarially Regularized Autoencoders learn smooth representations of discrete structures allowing for interesting results in text generation, such as unaligned style transfer, semi-supervised learning, and latent space interpolation and arithmetic.