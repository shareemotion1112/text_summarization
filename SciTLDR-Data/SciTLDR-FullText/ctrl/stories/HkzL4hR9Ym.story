Good representations facilitate transfer learning and few-shot learning.

Motivated by theories of language and communication that explain why communities with large number of speakers have, on average, simpler languages with more regularity, we cast the representation learning problem in terms of learning to communicate.

Our starting  point sees traditional autoencoders as  a single encoder with a fixed decoder partner that must learn to communicate.

Generalizing from there, we introduce community-based autoencoders in which multiple encoders and decoders collectively learn representations by being randomly paired up on successive training iterations.

Our experiments show that increasing community sizes reduce idiosyncrasies in the learned codes, resulting in more invariant representations with increased reusability and structure.

The importance of representation learning lies in two dimensions.

First and foremost, representation learning is a crucial building block of a neural model being trained to perform well on a particular task, i.e., representation learning that induces the "right" manifold structure can lead to models that generalize better, and even extrapolate.

Another property of representation learning, and arguably the most important one, is that it can facilitate transfer of knowledge across different tasks , essential for transfer learning and few-shot learning among others BID0 .

With this second point in mind, we can define good representations as the ones that are reusable, induce the abstractions that capture the "right" type of invariances and can allow for generalizing very quickly to a new task.

Significant efforts have been made to learn representations with these properties; one frequently explored direction involves trying to learn disentangled representations BID12 BID6 BID5 BID17 ), while others focus on general regularization methods BID15 BID18 .

In this work, we take a different approach to representation learning, inspired by successful abstraction mechanisms found in nature, to wit human language and communication.

Human languages and their properties are greatly affected by the size of their linguistic community BID11 BID19 BID16 BID9 .

Small linguistic communities of speakers tend to develop more structurally complex languages, while larger communities give rise to simpler languages (Dryer & Haspelmath, 2013) .

Moreover, we even observe structural simplification as the effective number of speakers grows, as in the example of English language BID10 .

A similar relation between number of speakers and linguistic complexity can also be observed during linguistic communication.

Speakers, aiming at maximizing communication effectiveness, adapt and shape their conceptualizations to account for the needs of their specific partners, a phenomenon often termed in dialogue research as partner specificity BID2 ).

As such, speakers form conceptual pacts with their listeners BID1 , and in some extreme cases, these pacts are so ad-hoc and idiosyncratic that overhearers cannot follow the discussion BID13 )!

But how are all these linguistic situations related to representation learning?

We start by drawing an analogy between language and representations induced by the traditional and extensively used framework of autonencoders (AE).

In the traditional AE set-up, there is a fixed pair of a single encoder and a single decoder that are trained to maximize a reconstruction loss.

However, encoders and decoders co-adapt to one another, yielding idiosyncratic representations.

The encoders spend repre-sentational capacity modeling any kind of information about the data that could allow the decoder to successfully reconstruct the input; as long as the encoder and the decoder agree on a representation protocol, this information need not be abstract or systematic.

This has a negative impact on the reusability of the representations, something that afterall is a key objective of representation learning.

Evidence of this co-adaption is found in the above-mentioned efforts targeting generalization.

The human language analogy of the traditional AE setup would be an extreme version of the conceptual pact experiments from BID13 , where two people never communicate with anybody else: the resulting language would be very hard to understand for any outsider.

In this work we test whether removing this co-adaptation between encoders and decoders can yield better generalization, much as dropout removes co-adaptation between activations and thereby yields better generalization in general neural networks.

We hypothesize that machines that communicate not with a specific partner but with a multitude of partners, will shape the representations they communicate to be simpler in nature.

We introduce a simple framework that we term communitybased autoencoders (CbAEs), in which there exist multiple encoders and decoders, and at every training iteration one of each is randomly sampled to perform a traditional autoencoder (AE) training step.

Given that the identity of the decoder is not revealed to the encoder during the encoding of the input, the induced representation should be such that all decoders can use it to successfully reconstruct the input.

A similar argument holds for the decoder, which at reconstruction time does not have access to the identity of the encoder.

We conjecture that this process will reduce the level of idiosyncrasy, resulting in representations that are invariant to the diverse encoders and decoders.

We apply CbAEs to two standard computer vision datasets and probe their representations along two axes; their reusability and their structural properties.

We find that in contrast to representations induced within a traditional AE framework 1) the CbAE-induced representations encode abstract information that is more easily extracted and re-used for a different task 2) CbAE representations provide an interface that is easier to learn for new users 3) and the underlying topology of the CbAE representations is more aligned to human perceptual data that are disentangled and structured.

Background One of the simplest and most widely used ways to do representation learning is to train an autoencoder, i.e., encode the input x, usually in a lower-dimensional representation, z = e(x, θ) using some parameters θ, then use the z representation to decode back the input x = d(z, φ) through another set of parameters φ.

θ and φ are trained by minimizing a reconstruction loss, e.g.,: DISPLAYFORM0 The resulting latent vector z is then treated as the induced representation of the input data, and is often re-used for other problems, such as supervised learning or reinforcement learning.

Because this approach is very general and can be applied to any data set, it holds the promise of being able to leverage existing unlabelled data, in order to then quickly solve other problems, using much less data and/or computation.

However, the loss in Eq. 1 has an important flaw: it does not directly incentivize the formation of latents that have all the properties of good representations, such as appropriate abstraction and reusability.

As a result, significant amounts of research effort have been dedicated to finding a better loss BID18 , BID7 , inter alia).Our method The CbAE framework (see FIG0 is inspired by the hypothesis that the size of a linguistic community has a causal effect on the structural properties of its language.

Unlike the traditional autoencoder framework, which uses a single encoder paired with a single decoder, the CbAE set-up involves a community of K enc encoders and K dec decoders.

1 As such, we are not dealing with a single autoencoder, but rather a collection of K enc × K dec autoencoders.

No single encoder and decoder are associated with one another, but rather the community of encoders are associated with the community of decoders and all combinations may be used together.

Importantly, while the network architectures can be (and in fact in this work are) identical across a community (i.e., all encoders and decoders have the same number and organization of units and weights) there is no weight-sharing among members of the community.

Training procedure At each training step, given a data point x, we form an autoencoder by randomly sampling an encoder and a decoder from the respective communities.

Then, we perform a traditional autoencoding step where we minimize the mean-squared (L 2 ) loss between the input x and its decoding (see Eq. 1 and Algorithm 1).

Trivially, the traditional autoencoder training protocol can be recovered by setting DISPLAYFORM1 1 optimize e i and d i with respect to L i end for There are two main reasons why we think this will have a positive effect on the quality of the representations.

First, given that the chosen encoder e i for iteration i does not have a priori information about the identity of the chosen decoder d i , and given that there are a number of decoders all with different weights, the encoder should produce a latent z i that is potentially decodable by all different decoders.

Similarly, given that each decoder d i receives over its training lifetime latents from a number of different encoders, the decoder should learn to decode representations produced by all encoders.

We hypothesize that this training regime will produce latents that are less prone to have idiosyncrasies rooted in the co-adaptation between a particular pair of encoder and decoder.

Relation to dropout The CbAE setup is reminiscent of dropout BID15 : The entire community can be viewed as one much larger and highly parallel model, from which at each iteration a selection of weights (corresponding to one specific community member) is chosen.

However, a crucial difference is that here, the choice of weights happens in a very correlated way; it is not a random set of weights, but one of K enc or K dec non-overlapping subsets that is selected at each training step.

As a consequence, the weights in one community member (e.g. an encoder) will be much slower to adjust (if at all) to those in the rest of the community, and a higher degree of diversity is maintained.

There is some mutual adjustment, of course, but it is a second-order effect: encoder e i and encoder e j will only get information about each other's encoding through the gradients of decoders that have learned to decode them.

The curse of co-adaptation The goal of our method is to avoid co-adaptation between the encoder and decoder.

However, due to their flexibility, neural networks are in principle capable of co-adapting to several partner modules at once.

As a consequence, the encoders can avoid convergence and still learn to produce latents from which the decoders can successfully reconstruct the input by capitalizing on encoder-specific information.

Intuitively, we can think of this as the encoder essentially "signing" the latents with their unique ID.

We test whether this indeed manifests in the setup by training a linear classifier whose task is to identify the encoder from the latent representation: p e (z) = exp(w Table 1 : Encoder identification error rate on MNIST.As Table 1 shows, the encoder classifiers perform significantly better than chance in spite of having to keep up with shifting representations, indicating that pairwise co-adaptation does indeed happen to some extent for all community sizes.

The nonmonotonous behaviour seen on the row labelled without entropy loss is due to the two competing effects.

As the community size grows, the encoder identification task becomes harder (hence the lower chance), and the error rate naturally increases.

However, larger communities also lead to slower rates of representation shift for every individual encoder, making it easier for the encoder classifier to keep up with their changing representation.

A similar phenomenon of co-adaptation is often encountered in domain-adaption neural frameworks.

To alleviate this, adversarial losses or gradient reversal layers BID4 are introduced to penalize representations from retaining domain-specific information.

Here, in order to counteract the all-to-all pairwise co-adaptation effect, we add a simple adversarial loss forcing the encoders to be indistinguishable for the encoder classifier while keeping the encoder classifier itself fixed.

In particular, the extra loss term is the negative entropy of the classifier, L entropy (z) = e p e (z) log p e (z).

We use MNIST and CIFAR-100, with community sizes of 1, 2, 4, 8 and 16.

The batch size is fixed at 128 throughout all experiments.

We use the Adam optimizer with a learning rate of 10 −4 .

The encoders are straightforward convolutional neural networks of VGG-flavour, with depths of 6 (MNIST) and 10 (CIFAR-100) layers respectively.

For the details we refer the reader to the Appendix.

The decoders implement the corresponding transpose convolutions.

Having to respond to more communication partners makes the job of the individual encoders and decoders harder.

This effect is seen in the reconstruction loss (see FIG2 , where an increase in community size leads to a penalty in the reconstruction error even when correcting for the amount of training data seen by each community member.

Note that this is not necessarily limiting when considering the desired properties of the representation, since the pixel-loss is merely a self-supervision signal: some pixel-level information is lost, but ultimately pixel-level information is not the true goal of the representation learning exercise, as discussed in Section 1.

The interesting question is however: given the capacity of the latents, are we trading-off reconstruction performance for other more relevant properties such us reusability or structure?

The experiments presented in section 3 aim at answering precisely this question.

In the previous section we introduced the CbAE set-up and found that the reconstruction loss increases as the community size grows.

However, the reconstruction loss in this setup is just a learning signal for representation learning, rather than the end goal.

Ultimately, we are interested in good representations that could allow for generalization, knowledge transfer and reusability.

In this section, having trained the CbAEs, we devise a number of parametric (Section 3.1) and non-parametric (Section 3.2) evaluation methods that probe the representations for exactly these properties.

Training new encoders and decoders Human languages have the property that the more regular and systematic they are, the easier they are for learners to acquire.

We examine whether the latent interface found by the CbAE setup has the same property, i.e., is easier to learn for new users.

To do so, we train newly initialized encoders and decoders.

The hypothesis is that if the CbAE-trained encoders and decoders have learned to encode information in the representation in a systematic way, rather than in an ad-hoc and idiosyncratic way, this would result in the new, untrained, encoders and decoders being able to learn the representation with less effort, which we define operationally as better sample complexity (see below).

This evaluation task is illustrated in the two leftmost panels of FIG3 .Transferring representations to a new task Next, we investigate the transfer capabilities of the representations to a different task; we freeze the CbAE encoders, perform supervised learning on image classification by training linear classifiers and evaluate their sample complexity.

The hypothesis is that the CbAE framework induces abstract representations of the input data that would allow a simple linear classifier to achieve a better sample complexity.

Experimental setup In these probe tasks, only the newly initialized evaluation modules (new encoders, new decoders, and linear classifiers) are trained, and the encoders and decoders trained in the CbAE setup are frozen.

The tasks share the same basic set-up: all frozen members of the community of encoders or decoders are coupled individually to an evaluation module, which is trained to perform best-response to their pre-trained partner.

For the new encoders and decoders, we use the same architecture as for the CbAE-trained ones.

We use the Adam optimizer with a learning rate of 10 −4 .

For the linear classifiers, we fit a linear layer, followed by a softmax, on the latents of each CbAE-trained encoder.

We use the Adam optimizer with a learning rate of 10 −3 and optimize the cross-entropy between the predicted labelŷ and the actual label y. We use a minibatch size of 128 throughout all experiments.

Sample complexity gain For every parametric probe task, we first record the average performance achieved by all modules trained with a given community after a given number of training iterations.

We then obtain the number of training iterations needed for the traditional AE (i.e., community size 1) to reach the same performance, and compute the ratio of these two training durations as the sample complexity gain.

The above formulation takes the following form in more mathematical notation: Given a learning curve L(i) which maps an iteration i to an obtained result L(i), we define the inverse learning curve: DISPLAYFORM0 The inverse learning curve returns the first iteration at which the result dropped below the argument L .

Equipped with this function, the sample complexity gain of curve L at iteration i relative to curve L baseline is straightforward to compute: DISPLAYFORM1

Transferring representations to a new task After training the image classifier, we evaluate its performance on the test set.

FIG4 shows the sample complexity gain relative to the traditional AE case Overall, we find that classifiers trained on the latents learned by larger communities learn faster.

For example, the leftmost bar on the MNIST plot shows that a classifier trained on the latents from community of size 2 needs 8 iterations to reach a performance that takes almost twice (1.6 times) as many iterations for a classifier trained on the latents from a classical AE.

Moreover, the MNIST plot clearly shows that larger communities lead to faster classifier learning.

As the classifier training progresses, the gains relative to the classical AE become smaller; this suggests that there is might still be some co-adaption, which however is significantly delayed by the introduction of the community.

This positive effect of the community is also clear since the largest community is still speeding up relative to the classical AE after 32 iterations.

In the case of CIFAR-100, we observe similar sample complexity gains, but the community size effect appears to be reversed.

We attribute this to the larger model used (and needed) for this data set, which presents a significantly more complex task both for autoencoders and for classifiers.

This larger model in turn requires more CbAE iterations for the communities to learn to represent the data at all, with each community member only seeing 1/K of the CbAE iterations (K being the community size).

Training new encoders and decoders FIG5 shows the results for training new decoders on MNIST and CIFAR-100.

The new decoders learn faster with encoders trained in larger communities.

Moreover, we observe that although there are large sample complexity gains over the baseline, in absolute numbers these gains are smaller than the ones obtained in the previous transfer task.

While the image classification task requires a CbAE encoder to have produced a representation capturing a certain level of abstraction, training a new decoder requires the CbAE encoder to represent precise information about the data.

The fact that CbAEs are better in the former than the latter suggests that their representations are more abstract in nature.

Evidently, we are ready to accept this trade-off; as discussed in the introduction, abstraction is the holy grail of representation learning.

FIG6 shows the results for training new encoders on MNIST and CIFAR-100.

The new encoders learn faster with decoders trained in larger communities, and display roughly the same sample complexity gain pattern as the new decoders, albeit with slightly better absolute numbers.

To understand this, we note that there is an asymmetry between encoders and decoders, in that the decoders can learn to decode a large hypervolume in latent space into roughly the same image, encompassing the encodings of all individual encoders.

A new encoder then only has to learn to encode an image somewhere into that hypervolume to get a reasonable reconstruction error.

A new decoder, however, has to learn best-response to a specific encoder, which essentially involves more adaptation to residual idiosyncrasies and can therefore be a more difficult task.

Comparison to other regularization mechanisms We have performed the same probe analyses presented in sections 3.1.1 and 3.1.2 on the representations learned by a traditional AE setup enhanced with (neuron-level) dropout, and found no gains relative to the dropout-free setting.

Exploring variations on dropout that interpolate between large expected overlaps between subsets (the traditional implementation) and fully mutually exclusive subsets (our method) is an interesting direction of future work.

Moreover, all our models are trained with batch normalization, indicating that the gains we find are orthogonal to the specific regularization advantages it provides.

Finally, we ask the question of to what degree the speaker-invariance bias imposed by the CbAE framework induces abstract representations that share the same underlying structure with human perceptual data.2 As a proxy of human perceptual data, we use the Visual Attributes for Concepts Dataset (VisA) of BID14 , which contains human-generated per-concept attribute annotations for concrete concepts (e.g., cat, chair, cat) spanning across different categories (e.g., mammals, furniture, vehicles), annotated with general visual attributes (e.g., has whiskers, has seat).

TAB0 presents some examples of the conceptual representations found in VisA. As we can see, concepts representations are structured and disentangled.

Therefore, achieving high similarity with these would indicate that the CbAE-induced representations encode similar conceptual abstract information.

Most importantly, this is an independent task and requires no additional training of parameters.

Representation Similarity Analysis For measuring the similarities between the human perceptual data and the CbAE-induced representations, we perform Representational Similarity Analysis (RSA) in the two topologies, a method popular in neuroscience BID8 .

For each community configuration, we sample 5,000 images and encode them with all encoders.

Following that, for each encoder-specific set of latents, we apply concept-based late fusion, meaning that we average in a single latent all latents belonging to the same concept, to arrive to 68 concept-based representations.

We then compute two sets of pairwise similarities of the 68 concepts, i.e., one set using their concept-based CbAE-induced latent representations and one set the concept-based VisA attribute representations.

With these two lists of cosine similarities in hand, the RSA between the two topologies is taken as the Spearman correlation of these two lists of similarities.

Given that RSA is a second-order similarity, we are not asking the question of how similar (say in terms of cosine) the two spaces are, but rather how similar their topology is, i.e., whether points that are nearby in the latent space are also nearby in the VisA space.

Results TAB1 summarizes our results.

For each community configuration we report the mean RSA performance (obtained by averaging the RSA scores produced by the different encoders) and the maximum performance.

To account for the potential confounder caused by the different initializations in the CbAE, we compare the results with the best result found from same number of independent AE.

We observe that the mean similarity increases with the size of the population, confirming the hypothesis that CbAE produce on average abstract representations that to some degree reflect the topology of the highly structured and disentangled human data.

Moreover, we observe even higher gains when looking at the best RSA value within a CbAE; training an encoder within a community of diverse partners can lead to more abstract and structured representations than training a diverse set of independent encoders each with a fixed decoder partner.

This result rejects an alternative hypothesis; the gains cannot by explained just by increasing the diversity of the initializations, it is the community training of these diverse models that leads to increases performance.

Finally, while the largest CbAE (i.e., community size 16) has higher RSA similarity than the baseline, it shows the smallest gains, a pattern consistent with the the rest of the parametric probe results of CIFAR-100.

We attribute this to the fact that this community had the smallest number of iterations per member, and had therefore not had the opportunity to learn to represent the data well yet.

We have presented Community-based AutoEncoders, a framework in which multiple encoders and decoders collectively learn representations by being randomly paired up on successive training iterations, encouraging a similar lack of co-adaptation that dropout does at the activation level, at model level.

Analogous to the structural simplicity found in languages with many speakers, we find that the latent representations induced in this scheme are easier to use and more structured.

This result is philosophically interesting in that it suggests that the community size effects found in human languages are general properties of any representation learning system, opening avenues to potential synergies between representation learning linguistics.

The price for obtaining these representations is the increase in computational requirements, which is linear in the community size.

Due to the reusability of the resulting representations, this cost may be amortized over a number of applications trained on top of the encoders.

Furthermore, the community-based training procedure is highly parallelizable, since only the latents and corresponding backpropagated errors need to be sent between the encoders and decoders.

<|TLDR|>

@highlight

Motivated by theories of language and communication, we introduce community-based autoencoders, in which multiple encoders and decoders collectively learn structured and reusable representations.

@highlight

The authors tackle the problem of representation learning, aim to build reusable and structured represenation, argue co-adaptation between encoder and decoder in traditional AE yields poor representation, and introduce community based auto-encoders.

@highlight

The paper presents a community based autoencoder framework to address co-adaptation of encoders and decoders and aims at constructing better representations.