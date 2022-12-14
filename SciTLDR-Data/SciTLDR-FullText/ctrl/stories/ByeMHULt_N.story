While much recent work has targeted learning deep discrete latent variable models with variational inference, this setting remains challenging, and it is often necessary to make use of potentially high-variance gradient estimators in optimizing the ELBO.

As an alternative, we propose to optimize a non-ELBO objective derived from the Bethe free energy approximation to an MRF's partition function.

This objective gives rise to a saddle-point learning problem, which we train inference networks to approximately optimize.

The derived objective requires no sampling, and can be efficiently computed for many MRFs of interest.

We evaluate the proposed approach in learning high-order neural HMMs on text, and find that it often outperforms other approximate inference schemes in terms of true held-out log likelihood.

At the same time, we find that all the approximate inference-based approaches to learning high-order neural HMMs we consider underperform learning with exact inference by a significant margin.

There has been much recent interest in learning deep generative models with discrete latent variables BID29 BID28 BID12 BID25 BID15 Lee et al., 2018, inter alia) , especially in the case where these latent variables have structure -that is, where the interdependence between the discrete latents is modeled.

Most recent work has focused on learning these models with variational inference BID14 , and in particular with variational autoencoders (VAEs) BID17 BID32 .Variational inference has a number of convenient properties, including that it involves the maximization of the evidence lower-bound (ELBO), a lower bound on the log marginal likelihood of the data.

At the same time, when learning models with discrete latent variables variational inference may require the use of potentially high-variance gradient estimators, which are obtained during learning by sampling from the variational posterior; see Appendix A for an empirical investigation into the variance of various popular estimators when learning neural text HMMs with VAEs.

In this paper we investigate learning discrete latent variable models with an alternative objective to the ELBO.

In particular, we propose to approximate the intractable log marginal likelihood with an objective deriving from the Bethe free energy BID1 , a quantity which is intimately related to loopy belief propagation (LBP) BID30 BID45 BID9 BID9 , and which is the basis for "outer approximations" to the marginal polytope BID39 .

The Bethe free energy is attractive because if all the factors in the factor graph associated with the model have low degree, it can often be evaluated efficiently, without any need for approximation by sampling (see Section 2).

Of course, requiring all factors in the factor graph to be of low degree severely limits the expressiveness of directed graphical models.

It does not, however, limit the expressiveness of markov random fields (MRFs) (i.e., undirected graphical models) as severely, since we can simply have an extremely loopy MRF, with arbitrary pairwise factors; see FIG1 (c) and Section 2.2.We accordingly propose to learn deep, undirected graphical models with latent variables, using a saddlepoint objective that makes use of the Bethe free energy approximation to the model's partition functions.

We further amortize inference by using "inference networks" BID36 BID17 BID13 BID38 in optimizing the saddle-point objective.

Unlike the ELBO, our objective will not form a lower bound on the log marginal likelihood, but an approximation to it.

At the same time (and unlike other recent work on MRFs with a variational flavor BID21 BID23 ), this objective can be optimized efficiently, without sampling, and in our experiments in learning neural HMMs on text it outperforms other approximate inference methods in terms of held out log likelihood.

We emphasize, however, that despite the improvement observed when training with the proposed objective, in our experiments all approximate inference methods were found to significantly underperform learning with exact inference; see Section 4.3.

Let G = (V ??? F, E) be a factor graph BID5 BID20 , with V the set of variable nodes, F the set of factor nodes, and E the set of undirected edges between elements of V and elements of F; see FIG1 (b) and (c) for two examples.

We will refer collectively to variables in V that are always observed as x, and to variables which are never observed as z. In an MRF, the joint distribution then factorizes as DISPLAYFORM0 where ?? indexes elements of F, the potentials functions ?? ?? are assumed to always be positive and are parameterized by ??, and Z(??) is the partition function, given by: DISPLAYFORM1 We use the notation x ?? and z ?? to denote the subvectors of x and z, respectively, that participate in the factor indexed by ??. (We will similarly denote subvectors of the realizations of x and z as x ?? and z ?? ).

For example, if we had DISPLAYFORM2 Marginalizing out the unobserved variables yields: DISPLAYFORM3 where DISPLAYFORM4 is the partition function with x "clamped" to x (that is, it is the partition function of P (z | x; ??)).

When learning a latent variable MRF, it is therefore natural to minimize DISPLAYFORM5 with respect to ??.1 To minimize notation, we will also allow subvectors to be empty.

The Bethe free energy BID1 ) is a function of a parameterized factor graph and its marginals, which can be interpreted as an approximation of its clamped or unclamped partition function.

To define it, first let ?? ?? (x ?? , z ?? ) ??? [0, 1] be the marginal probability of random subvectors x ?? and z ?? taking on the values x ?? and z ?? , respectively, which can be obtained by marginalizing out all variables that do not participate in the factor ??.

We will refer to the vector consisting of the concatenation of the marginal probabilities for each instantiation of each factor as ?? ??? [0, 1] n .

As a concrete example, consider the 10 factors in FIG1 Following BID46 , the Bethe free energy is then defined as DISPLAYFORM0 where we use the notation q ?? (x, z) to refer to a distribution with marginals ?? , and where ne(x v ) refers to the number of factor-neighbors node v has in the graphical model.

The second line in equation 2 can be derived from the first by rewriting q ?? (x, z) as a product of its marginals divided by a (different) product of its marginals, which is always possible for distributions represented by tree-structured graphical models BID39 , and then simplifying the resulting expressions; see BID6 for an explicit derivation.

Crucially, since log Z(??) does not depend on ?? , in the case of a tree-structured graphical model we have min ?? F (?? ) = ??? log Z(??), since the KL divergence vanishes when ?? matches the marginals of P (x, z; ??).In the case where the graphical model is not tree-structured, we may still define F (?? ) as in equation 2, but in general we will only have min ?? F (?? ) ??? ??? log Z(??) (see BID44 , , and for more precise characterizations), since the marginals ?? may not correspond to any distribution, and equation 2 no longer corresponds to a true KL divergence.

Nonetheless, the Bethe approximation may often work well in practice BID46 BID27 ).

Indeed, Yedidia et al. (2001 show that loopy belief propagation corresponds to finding fixed points of the constrained optimization problem min ?? ???C F (?? ), where C contains "pseudo-marginal" vectors ?? that meet the following constraints: (1) each marginal distribution contained in ?? consists of positive elements that sum to one, and (2) the marginals contained in ?? are locally consistent in the sense that for any random variable v in the graph, and for any two factors ??, ?? involving v, we have DISPLAYFORM1 for all values k that v might take on.

In other words, the marginal probability of any random variable v taking on a particular value k is consistent between all factors in which v participates.

To use the variable z 1 in FIG1 (c) as an example, local consistency requires that DISPLAYFORM2 where factors have been ordered top-down, left-to-right in the factor graph in FIG1 .

Note that there are pseudo-marginals ?? ??? C that do not correspond to the marginals of any distribution; see BID39 for an example.

The attractive feature of the Bethe approximation in equation 2 for our purposes is that while it is exponential in the degree of each factor (because it must consider every marginal), it is only linear in the number of factors.

Thus, evaluating the Bethe free energy of a factor graph with a large number of small-degree (e.g., pairwise) factors remains tractable.

As noted above, while this restriction severely limits the expressiveness of directed graphical models, MRFs are free to have arbitrary pairwise dependence.

This can be seen, for example, in FIG1 , which shows two different factor graphs for a 3rd order HMM BID31 .

While

We now consider learning by making use of the Bethe approximation to minimize an approximation to the log marginal likelihood of equation 1.

In particular, from the previous section we have that DISPLAYFORM0 where F x (?? x ) is the Bethe free energy of equation 2 with x clamped to particular values.

That is, F x does not consider marginals corresponding to settings of x that do not agree with x, and thus ?? x will in general be smaller than ?? .

In particular, to continue the example of FIG1 (c) from the previous section, where all variables are assumed to be binary, ?? x will be in [0, 1] 32 instead of [0, 1] 40 , since for each variable in x 1 , . . .

, x 4 we ignore the two marginals corresponding to the unobserved value.

We may then define the following loss, as an approximation (but neither an upper nor lower bound) of equation 1: DISPLAYFORM1 and thus we arrive at the following saddle-point learning problem: min DISPLAYFORM2 3.1 CONSTRAINED OPTIMIZATION While LBP can be used to find pseudo-marginals representing fixed points of F and F x , it is somewhat unappealing in the context of deep generative modeling, primarily because it is an iterative message-passing algorithm often requiring multiple rounds for convergence, and where the order in which messages are passed appears to significantly impact results BID48 BID19 .Instead, we propose to predict approximate minimizers of the Bethe free energy using trainable inference networks f (G; ??) and f x (G, x; ?? x ), which will compute approximate minimizers of F and F x , respectively.

2 These inference networks, which are similar to graph neural networks BID33 BID24 BID18 BID47 ) but somewhat simpler, will attempt to find minimizers living in the constraint set C, which, as noted in Section 2.2, consists of vectors ?? containing pseudo-marginals which are positive and sum to one, and which respect local consistency.

We address the parameterization of the inference networks, and how they handle these constraints on the pseudo-marginals below.

Equality Constraints It is clear that the constraints that the pseudo-marginals sum to one and that they exhibit local consistency are linear constraints on ?? , and so they can be expressed as A?? = b, where A ??? {0, 1} m??n consists of m linearly independent constraint rows and and n is the length of ?? .

It is standard in linear-equality constrained optimization to optimize over the subspace defined by these linear constraints instead of the original optimization variable ?? BID2 .

In particular, given a feasible point?? and a basis V ??? R n??n???m for the null space of A, any solution can be written as V u +?? , and so we may optimize over u instead of ?? .When using inference networks to compute minimizers, however, it is more natural to have these networks output vectors of length n (i.e., the size of ?? , which is linear in the factors of the MRF), rather than of size n ??? m, which depends on the number of constraints.

Accordingly, we may equivalently write minimizers as V V + ?? +?? , where ?? ??? R n , and V + is the Moore-Penrose pseudoinverse of V .

In particular, V V + is an orthogonal projection on the range of V , a basis for the null space of A. Thus, given an inference network f that computes vectors in R n , the vector (V V + f (G; ??) +?? ) ??? R n will satisfy the equality constraints imposed by C, assuming?? does.

Positivity Constraints In order to keep elements of the predicted pseudo-marginals positive, we simply impose a penalty during training on predicting a pseudo-marginal with non-positive elements. (If, during learning, a predicted pseudo-marginal is non-positive, we set it to a small positive constant).

We found a linear penalty on non-positive values to work well: given a vector ?? ??? R n , we define the penalty function C(??) = ??? 1 n n i=1 min{?? i , 0}. We thus arrive at the following training objective: DISPLAYFORM3 where P and P x are the orthogonal projections V V + defined by the constraints on ?? and ?? x , respectively.

In order to investigate how the approximate inference approach outlined above compares with other popular approaches to approximate inference, we will use approximate inference to learn in a setting in which we can tractably compute log marginal likelihoods.

We consider in particular learning 2nd and 3rd order neural HMMs, as in FIG1 , on text, using various flavors of amortized variational inference as well as the Bethebased objective F introduced above. (Note that Bethe based objectives are inexact for loopy MRFs, such as the HMM-style MRFs with long-distance pairwise factors in FIG1 ).Because the Bethe objective is only interesting in the case of undirected models (e.g., FIG1 ), we will strictly speaking be comparing full HMMs (learned with VAEs) with a less expressive product of expertstyle variant (learned with the Bethe-based objective).

However, our experiments confirm that both model classes can obtain similar held-out perplexities when learned with exact inference.

We accordingly begin by briefly outlining how the neural HMMs and their associated inference networks are parameterized.

To simplify the notation somewhat, in what follows we will view HMMs as parameterizing a joint distribution over a sequence of T discrete observations x 1:T and a sequence of T discrete latent variables z 1:T , where each observation z t takes one of V values, and each latent z t takes one of K values.

Neural Directed HMM We parameterize the HMM's emission distribution P (x t | z t = k), as softmax(W LayerNorm(e k + MLP(e k ))), where e k ??? R d is an embedding corresponding to the k'th discrete value z t can take on, W ??? R V ??d is a word embedding matrix with a row for each word in the vocabulary, and layer normalization BID0 ) is used to stabilize training.

We parameterize the transition distribution DISPLAYFORM0 , where U ??? R K??M K and the e k are shared with the emission parameterization.

Mean Field Style Inference Network When performing amortized variational inference with a mean field-like posterior, we obtain approximate posteriors q(z t | x 1:T ) for each timestep t as softmax(Qh t ), where h t ??? R d2 is the output of a bidirectional LSTM BID11 BID8 run over the observations x 1:T , and Q ??? R K??d2 ; note that because the bidirectional LSTM consumes all the observations this posterior is less restrictive than traditional mean field.

Structured Inference Network Instead of assuming the approximate posterior q(z 1:T | x 1:T ) factorizes independently over timestep posteriors as in mean field, we can assume it is given by the posterior of a first-order (and thus more tractable) HMM.

We parameterize this inference HMM identically to the neural directed HMM above, except that it conditions on the observed sequence x 1:T by concatenating the averaged hidden states of a bidirectional LSTM run over the sequence onto the e k .Undirected Neural HMM We parameterize the emission factors ?? ?? (x t , z t ) as locally normalized distributions, in exactly the same way as the neural directed HMM above.

In order to fairly compare with the directed HMM, the transition factors ?? ?? (z s = k 1 , z t = k 2 ) are homogeneous (i.e., independent of the timestep), and are given by r T k2 LayerNorm([a |t???s| ; e k1 ] + MLP([a |t???s| ; e k1 ])), where a |t???s| is the embedding vector corresponding to factors relating two nodes that are |t ??? s| steps apart, and where e k1 and r k2 are again discrete state embedding vectors.

Bethe Inference Networks In the case of sequential models like HMMs, it is fairly simple to parameterize f (G; ??) and f x (G, x; ?? x ).

For f (G; ??) we form an embedding for each factor ?? ?? (z s , z t ) by concatenating embedding vectors corresponding to the s'th and t'th timesteps with the K 2 log potential values log ?? ?? (z s = k 1 , z t = k 2 ) (as given above), and then run a bidirectional LSTM over these factor-embeddings, ordered in ascending order of z s .

We then predict pseudo-marginals for each factor with a linear layer applied to the LSTM output.

The parameterization of f x (G, x; ?? x ) is similar, except we also concatenate the log observation potentials log ?? ?? (x s , z s ) and log ?? ?? (x t , z t ) onto the embedding of factor ?? ?? (z s , z t ) before feeding it to the LSTM.

Table 1 : Results of learning high order neural text HMMs.

"Full" subtables give the performance of learning directed HMMs with a VAE objective and mean field-style ("MF") posterior approximation plus baseline ("BL"), with an IWAE objective (and L = 5 or L = 10 samples) and mean field-style posterior approximation, with a VAE objective and first order HMM ("FO") posterior approximation, and with the exact log marginal.

VAE and IWAE objectives use REINFORCE-like gradient estimators.

"Pairwse MRF" subtables give the performance of learning pairwise MRF HMMs with the F objective but exact marginals, the F objective with approximate marginals given by inference networks, and by maximizing the exact log marginal.

For non-PPL objectives, we show the exponentiated, token-averaged (and negated, in the case of ELBO or IWAE) objective in order to be comparable with PPL.

Train We trained the HMM models described above with K = 20 latent states under both VAE-style and Bethe objectives on 16,737 sentences of length at most 20 from the Penn Treebank corpus BID26 , and evaluated them on a held out sample of 1,585 sentences.

Models and objectives were evaluated in terms of their true perplexity on the held out data, which can be computed reasonably efficiently with dynamic programs.

We found all models and objectives to be fairly sensitive to hyperparameters and random seeds, and so we report the best results obtained (in terms of held out true perplexity) by each model and objective after a random search over hyperparameter and seed settings.

The saddle-point objective F was optimized by alternating a maximization step wrt ??, a minimization step wrt ?? x , and a minimization step wrt ??.

Projection matrices P for each graph structure were pre-calculated and stored, and feasible pseudo-marginals?? can be obtained by assigning all marginals to be uniform.

Code for duplicating experiments is available at https://github.com/swiseman/bethe-min, and details on hyperparameters are given in Appendix B.

We begin with the results obtained by maximizing the true log marginal likelihood of the training data under both the directed ("Full" in Table 1 ) and undirected models ("Pairwise MRF" in Table 1 ), by backpropagating gradients through the relevant dynamic programs.

These results establish how well our models perform under exact inference, and are shown in the last row of each subtable in Table 1 .

We see that perplexities are roughly comparable between the directed and undirected models when trained with exact inference.

We now consider the remaining directed HMM results of Table 1 , where the models are trained with approximate inference.

In the first row of each "Full" subtable there, we show the result of maximizing the ELBO using a mean field-style posterior approximation and the REINFORCE BID43 gradient estimator, with an input-dependent baseline to reduce variance BID29 .

The results are quite poor, with this approximate inference scheme leading to a gain of almost 200 points in perplexity over exact inference.

Using the tighter IWAE BID3 objectives improves performance slightly in all cases, though the most dramatic performance improvement comes from using a first-order HMM posterior in maximizing the ELBO, which can be sampled from exactly using quantities calculated with the forward algorithm BID31 BID4 BID34 BID49 .

While these results are encouraging, note that in general we may not have an exact dynamic program for sampling from a lower-order structured model, and that moreover we still appear to incur a perplexity penalty of more than 100 points over exact inference; see Appendix A for an empirical comparison of the variance of these estimators.

Moving to the MRF results, the second row of each "Pairwise MRF" subtable in Table 1 contains the results of optimizing F as a saddle point problem.

While this approach too underperforms exact inference by approximately 100 points in perplexity, somewhat remarkably it manages to consistently outperform the best approximate inference results for the directed models by a fair margin.

The first row of each "Pairwise MRF" subtable in Table 1 attempts to determine whether the jump in perplexity when moving to the F objective is due to the approximate inference or to the approximate objective, by minimizing the F objective using the exact marginals, as calculated by a dynamic program. (Note that this is not equivalent to the negative log marginal likelihood, since the factor graphs are loopy).

Interestingly, we see that this performs almost as well as the exact objective, suggesting that, at least for HMM models, the F objective is reasonable, and approximate inference remains the problem.

Despite these encouraging results, we note that there are several drawbacks to the proposed approach.

In particular, we find that in practice F indeed can over-or under-estimate perplexity.

Moreover, while ELBO values are not perfectly correlated with their corresponding true perplexities, values of F seem even less correlated, which necessitates finding correlated proxies of perplexity that may be monitored during training.

Finally, we note that explicitly calculating the projection onto the nullspace of A may be prohibitive for some models (e.g., large RBMs BID35 ), and so other approaches to tackling the constrained optimization problem are likely necessary.

We have presented an objective for learning latent-variable MRFs based on the Bethe approximation to the partition function, which can often be efficiently evaluated and requires no sampling.

This objective leads to slightly better held-out perplexities than other approximate inference methods when learning neural HMMs.

Future work will examine scaling the proposed method to larger, non-sequential MRFs, and whether F -like objectives can be made to better correlate with the true perplexity.

Here we empirically investigate the variance of the various VAE-style gradient estimators discussed above in learning a directed 3rd-order neural HMM.

In FIG4 we plot the standard deviation of the components of the gradient with respect to the e k -the embedding vectors corresponding to each discrete latent stateaveraged over all the components, as training progresses.

These component-wise standard deviations are estimated from 5000 samples from the variational posterior, every 5 minibatches, for the first 350 minibatches.

FIG4 provides evidence that in practice there is indeed substantial variance associated with these estimators, and, comparing with the results in Table 1 , that performance at least appears to be inversely correlated with the variance during training.

Model hyperparameters are given in TAB1 .

All models were trained with minibatches of size 16.

MRF models were trained with Adam (Kingma & Ba, 2014) , while the directed models performed better with SGD.

<|TLDR|>

@highlight

Learning deep latent variable MRFs with a saddle-point objective derived from the Bethe partition function approximation.

@highlight

A method for learning deep latent-variable MRF with an optimization objective that utilizes Bethe free energy, that also solves the underlying constraints of Bethe free energy optimizations.

@highlight

An objective for learning latent variable MRFs based on Bethe free energy and amortized inference, different from optimizing the standard ELBO.