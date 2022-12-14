Learning disentangling representations of the independent factors of variations that explain the data in an unsupervised setting is still a major challenge.

In the following paper we address the task of disentanglement and introduce a new state-of-the-art approach called Non-synergistic variational Autoencoder (Non-Syn VAE).

Our model draws inspiration from population coding, where the notion of synergy arises when we describe the encoded information by neurons in the form of responses from the stimuli.

If those responses convey more information together than separate as independent sources of encoding information, they are acting synergetically.

By penalizing the synergistic mutual information within the latents we encourage information independence and by doing that disentangle the latent factors.

Notably, our approach could be added to the VAE framework easily, where the new ELBO function is still a lower bound on the log likelihood.

In addition, we qualitatively compare our model with Factor VAE and show that this one implicitly minimises the synergy of the latents.

Our world is hierarchical and compositional, humans can generalise better since we use primitive concepts that allow us to create complex representations BID11 ).

Towards the creation of truly intelligent systems, they should learn in a similar way resulting in an increase of their performance since they would capture the underlying factors of variation of the data BID2 ; ; ).

In addition, good representations improve the performance for tasks involving transfer learning and multi-task learning; since it will capture the explanatory factors.

According to BID18 , a compositional representation should create new elements from the combination of primitive concepts resulting in a infinite number of new representations.

For example if our model is trained with images of white wall and then is presented a boy with a white shirt, it should identify the color white as a primitive element.

Intuitively, our model will be able to construct different and multiple representations from the primitives.

Furthermore, a disentangled representation has been interpreted in different ways, for instance BID2 define it as one where single latent variables are sensitive to changes in generative factors, while being invariant to changes in other factors.

In addition, we agree with BID12 , which mentions that a disentangle representation should be factorised and interpretable.

Intuitevely, the model could learn generative factors such as position, scale or colour; if it is disentangle it should be able to traverse along the position variable without changing the scale or the colour.

It's worth noting that disentangled representations have been useful for a variety of downstream tasks such as domain adaptation by training a Reinforcement Learning agent that uses a disentangled representation of its environment BID13 ; or for learning disentangled primitives grounded in the visual domain discovered in an unsupervised manner BID14 .

The original Variational autoencoder (VAE) framework BID17 ; BID21 ) has been used extensively for the task of disentanglement by modifying the original ELBO formulation; for instance ??-VAE is presented in BID12 which increases the latent capacity by penalising the KL divergence term with a ?? hyperparameter.

In addition, BID16 achieved a more robust disentangled representation by using the model called Factor VAE which penalises the total correlation of the latent variables encouraging the independence of the latents; a similar approach is shown in BID5 , where they present a clever ELBO decomposition based on BID15 .

Other approaches rely on information bottleneck presented in BID23 to model frameworks for this task such as BID1 ; BID0 .

Furthermore, BID6 describe a model based on Generative Adversarial Networks BID8 ) by encouraging the mutual information between the latents and the output of the generator.

Notably, in BID11 , they describe comprehensively the ??-VAE model using a neuroscience and information theory approaches; they suggest that by encouraging redundancy reduction the model achieves statistical independence within the latents.

This model inspired us to look into different fields for new ways to enforce disentanglement of the latents.

To understand our model, we need first to describe Synergy BID7 ; BID22 ) being a popular notion of it as how much the whole is greater than the sum of its parts.

It's common to describe it with the XOR gate, since we need two independent variables to fully specified the value of the output.

Following, we describe the synergy from two related fields.

Computing the multivariate synergistic information is an ongoing topic of research BID22 ; Williams & Beer (2010); BID3 ; BID9 .

Most of the current research in this topic uses the Partial information diagram described by BID24 .

In order to understand the importance of the Synergy information in our framework it's essential to describe the relations with the Unique and Redundant Information.

Introducing the notation from Williams & Beer (2010), let's consider the random variable S and a random vector R = {R 1 , R 2 , .., R n }, being our goal to decompose the information that the variable R provides about S; the contribution of these partial information could come from one element from R 1 or from subsets of R (ie.

R 1 , R 2 ).

Considering the case with two variables, {R 1 , R 2 }, we could separate the partial mutual information in unique (U nq(S; R 1 ) and U nq(S; R 2 ) ), the information that only R 1 or R 2 provides about S is redundant (Rdn(S; R 1 , R 2 )), which it could be provided by R 1 or R 2 ; and synergistic (Syn(S; R 1 , R 2 )), which is only provided by the combination of R 1 and R 2 .

The figure 1 depicts the decomposition; this diagram is also called PI-diagram (Partial information) .

Additional notation is used for a better visualisation in the case of more variables: Unique {1},{2}; Redundant {1}{2} and Synergistic {12}. For the case of 2 variables (X 1 , X 2 ), we expect four contributions to the mutual information as described in BID3 ; BID20 : DISPLAYFORM0 It's easy to see that the number of terms increases exponentially as the number of sources increases.

The best measure for synergy is an ongoing topic of research.

In the subsection we are going to talk about the synergy metrics.

For neural codes there are three types of independence when it comes to the relation between stimuli and responses; which are the activity independence, the conditional independence and the information independence.

One of the first measures of synergy for sets of sources of information came from this notion of independence.

In BID24 it is stated that if the responses come from different features of the stimulus, the information encoded in those responses should be added to estimate the mutual information they provide about the stimulus.

Formally: DISPLAYFORM0 However, we just saw in the previous sections that the I(S; R 1 ) and I(S; R 2 ) could be decomposed in their unique and redundant and synergistic terms.

Intuitively, this formulation only holds if there is no redundant or synergistic information present; which means in the context of population coding that the responses encoded different parts of the stimulus.

If the responses R 1 and R 2 convey more information together than separate, we can say we have synergistic information; if the information is less, we have redundant information.

That's the reason why in BID7 , the synergy is considered as measure of information independence: The intuition behind this metric is that synergy should be defined as the "whole beyond the maximum of its parts".

The whole is described as the mutual information between the joint X and the outcome Y; whereas the maximum of all the possible subsets is interpreted as the maximum information that any of the sources A i provided about each outcome.

Formally, this is stated as: DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 This metric derives from Williams & Beer (2010) and BID9 , however one of the differences with this metric with the one presented in BID9 is that in this one we are considering the specific mutual information I max in a group of latents A i , whereas in the paper mentioned it considers only an individual latent.

Notably the I max can be expressed in terms of the KL divergence.

DISPLAYFORM4 Putting together the equation 7 and 5, we have the following: DISPLAYFORM5 In the following section we are going to use the intuition provided by the above equation in the VAE framework for the task of disentanglement.

The motivation of our contribution is inspired in this concept and driven by the belief that synergy is not desirable for the task of disentanglement, since we want the latents to be independently informative as possible about the data, instead of needing many latents to specify the data.

Therefore, we argue that by penalising the synergistic information within the latents and the data, we would encourage the disentanglement of the underlying factors of variation.

This hypothesis is also inspired in the information presented in BID9 ; BID7 , where it is stated that the synergy is a measure of independence information in the responses of the stimuli.

First, we need to change the notation to match the VAE framework notation (Z are the latents and X is the observations).

Also, A i is a subset of the latents, such that A i ??? {Z 1 , Z 2 , ..., Z n } and Z is the joint of the latents.

Formally: Z = d i Z i , where d is the number of dimensions of the latent variables.

Besides, from the VAE standard framework, we know that the distribution p(A i |x) is intractable which is why we need to use an approximate distribution q ?? (A i |x) parametrised by ?? parameters.

It's important to notice that this KL divergence could be computed in the same way as in the VAE framework; the only difference is the number of dimensions used for the random variable z. In the original VAE framework, we compute the KL divergence considering the joint Z = d i Z i ; whereas for the Synergy metric we don't use the joint but a subset of the latents.

For instance, if A i = Z 2 Z 5 Z 8 , we have the following expression: DISPLAYFORM0 Taking in account these considerations, we express the equation 8 as follows: DISPLAYFORM1 We start with the original ELBO formulation BID21 ; BID17 ) and add the penalised term corresponding to the synergy, where ?? is a hyperparameter: DISPLAYFORM2 DISPLAYFORM3 Expanding the S max term, we have the llowing: DISPLAYFORM4 From Hoffman & Johnson (2016), we know that the KL term in the ELBO loss is decomposed in D KL q ?? (z n ) p(z n ) + I(x n ; z) when we use the aggregate posterior and define the loss over the empirical distribution of the data p data (x).

Taking in account that, we can express the equation 12 as follows: DISPLAYFORM5 If we penalise the synergy (see Eq. 12), we will be penalising the mutual information term which is not desirable for this task BID16 ; we can see this effect explicitly in Eq. 14.

Therefore, we use only the second term to perform the optimisation which means maximising the subset of latents with the most amount of MI per outcome.

DISPLAYFORM6 It's easy to see in Eq. 15 that it's not a guaranteed lower bound on the log likelihood p x anymore, which is why we decided to penalise the subset of latents with the minimum specific mutual information (ie.

A w ).

In practice we found that computing the maximum subset A i for each outcome of x is too computational intensive, which is why we decided to use a mini-batch approximation, which is the version we show in the pseudo-code using a two step optimisation in the next section.

The final version we are going to use is the one below, where Imax is the KL term of the synergy term.

DISPLAYFORM7 , batch size m, latent dimension d, weight of synergy loss ??, discount factor ??, optimiser optim, reparametrisation function g ?? .??, ?? ??? Initialise VAE parameters repeat 3: DISPLAYFORM8 worst index ??? get index greedy(mu, logvar, ??) L syn ??? ?? * Imax(mu, logvar, worst index) See Eq.16 for Imax function DISPLAYFORM9 : Gradients of Syn loss minibatch 12: until convergence of objectiveIn the algorithm shown above we see in practice that we get better results when we sample the values of mu and logvar from the encoder for the step 2 of the optimisation.

We use a greedy approximation of the best latents by following a greedy policy (See Appendix).

For disentanglement, the dataset most commonly used is the dsprites dataset , which consists on 2D shapes generated from independent latent factors.

We used the same architecture and optimizer as Factor VAE BID16 for training our model.

In order to test qualitatively the Non-Syn VAE model, we decided to tranverse the latents and plot the mean activations of the latents.

In FIG2 (left), we see clearly that our model disentangles the factors of variation.

Likewise, on the right we see the mean activation of each active latent averaged across shapes, rotations and scales.

After looking at the figure above we can state that our model achieves state-of-the-art results using a qualitatively benchmark.

Interestingly, both models perform quite similar in this test.

Also, we decided to compute the same synergy term from the Non-Syn VAE in Factor VAE (just compute it, we didn't use it for training).

The hypothesis was that if Factor VAE achieves disentanglement, it should minimise the synergy as well.

We train Factor VAE using the same parameters and architecture described in BID16 .

We show the first 4000 steps for the Synergy term (i.e. ?? * Imax), since most of the interaction happens in the first steps.

In this paper we presented the intuition and derivation of the lower bound of a model that uses a novel approach inspired by the information theory and Neuroscience fields to achieve the disentanglement of the underlying factor of variations in the data.

After looking at the results,we can state that our model achieved state-of-the-art results, with a performance close to FactorVAE BID16 .

This is not the first time that a model draws ideas from information theory.

Many models BID23 ; BID1 ; BID0 used the information bottleneck presented in BID23 using the VAE framework.

Therefore, we truly believe that we should keep looking at the neuroscience and information theory fields for inspiration.

In general, we don't need to replicate or simulate biological models; however we should analyse the intuition about the known main mechanisms of our brain and adapt those to our models.

<|TLDR|>

@highlight

Minimising the synergistic mutual information within the latents and the data for the task of disentanglement using the VAE framework.

@highlight

Proposes a new objective function for learning dientangled representations in a variational framework by minimizing the synergy of the information provided.

@highlight

The authors aim at training a VAE that has disentangled latent representations in a "synergistically" maximal way. 

@highlight

This paper proposes a new approach to enforcing disentanglement in VAEs using a term that penalizes the synergistic mutual information between the latent variables.