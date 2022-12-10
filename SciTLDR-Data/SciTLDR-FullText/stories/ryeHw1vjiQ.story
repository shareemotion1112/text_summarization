In this work, we exploited different strategies to provide prior knowledge to commonly used generative modeling approaches aiming to obtain speaker-dependent low dimensional representations from short-duration segments of speech data, making use of available information of speaker identities.

Namely, convolutional variational autoencoders are employed, and statistics of its learned posterior distribution are used as low dimensional representations of fixed length short-duration utterances.

In order to enforce speaker dependency in the latent layer, we introduced a variation of the commonly used prior within the variational autoencoders framework, i.e. the model is simultaneously trained for reconstruction of inputs along with a discriminative task performed on top of latent layers outputs.

The effectiveness of both triplet loss minimization and speaker recognition are evaluated as implicit priors on the challenging cross-language NIST SRE 2016 setting and compared against fully supervised and unsupervised baselines.

Lower Bound (ELBO) given by: 48 ELBO(λ) = log(p(X)) − KL(q(Z|X, λ)||p(Z|X)),whose terms can be rearranged, and ELBO can be simplified to:49 ELBO(λ) = E q [log p(X|Z)] − KL(log q(Z|X, λ)||p(Z)).Two main components present in above equation are the inference model q(Z|X, λ) and the generative First term in above equation is equivalent to maximum likelihood estimation, thus it is in general 57 substituted by a reconstruction loss, while the second term can be seen as a regularizer, which tries to 58 ensure that the approximation follows the prior distribution as much as possible.

The posterior q θ (Z|X) is in general assumed to be an uncorrelated Gaussian.

In order to train the VAE 60 using stochastic gradient descent, the reparametrization trick (7; 8) is employed allowing gradients 61 computation through the sampling process between encoder and decoder.

Hence, the outputs of the 62 encoder network are the statistics of q θ (Z|X) and Z -input for the decoder -is ultimately obtained 63 by Z = µ(X) + σ(X) · , where µ(X) and σ(X) are the encoder's outputs given X, while is 64 sampled from N (0, I).

Speaker verification consists of accepting or rejecting a claimed identity by comparing two spoken 66 utterances, the first of these utterances being used for enrollment (produced by the speaker with the 67 target identity) and the second utterance is obtained from the verified speaker (9).

Under the text-independent setting, speaker verification is performed on top of unconstrained spoken depending on given class labels.

Our training loss is thus defined by: DISPLAYFORM0 where the first term, the mean squared error between the input X and its reconstructed pair X , is into an output layer and cross-entropy loss is measured using available labels.

We evaluate the described setting on the speaker verification task.

RMSProp is employed for

Evaluation is performed on top of the cross-language NIST SRE 2016 setting (11).

Test data in

Tagalog and Cantonese are available, while train data is in English.

Embeddings obtained with a 99 standard VAE, along with our two proposed strategies using two distinct D(µ(X), y) previously 100 described choices are compared with x-vectors, a fully-supervised approach shown to outperform 101 i-vectors (12) in the full-recording setting (13).

Train data is composed of: Switchboard-2, phases As expected, including speaker identities relevantly increases the discriminability of learned repre-

sentations when compared to a fully-unsupervised VAE, in both Tagalog and Cantonese evaluations.

We further notice that performing speaker recognition on top of statistics of the posterior is more 125 effective than the metric learning approach of triplet loss minimization alone.

adaptation yields a huge improvement in such cases.

We further evaluate the discriminability of the representations corresponding to the statistics of

@highlight

We evaluate the effectiveness of having auxiliary discriminative tasks performed on top of statistics of the posterior distribution learned by variational autoencoders to enforce speaker dependency.

@highlight

Propose an autoencoder model to learn a representation for speaker verification using short-duration analysis windows.

@highlight

A modified version of the variational autoencoder model that tackles the speaker recognition problem in the context of short-duration segments