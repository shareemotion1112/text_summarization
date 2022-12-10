We address the problem of marginal inference for an exponential family defined over the set of permutation matrices.

This problem is known to quickly become intractable as the size of the permutation increases, since its involves the computation of the permanent of a matrix, a #P-hard problem.

We introduce Sinkhorn variational marginal inference as a scalable alternative, a method whose validity is ultimately justified by the so-called Sinkhorn approximation of the permanent.

We demonstrate the efectiveness of our method in the problem of probabilistic identification of neurons in the worm C.elegans

Let P ∈ R n×n be a binary matrix representing a permutation of n elements (i.e. each row and column of P contains a unique 1).

We consider the distribution over P defined as

where A, B F is the Frobenius matrix inner product, log L is a parameter matrix and Z L is the normalizing constant.

Here we address the problem of marginal inference, i.e. computing the matrix of expectations ρ := E(P).

This problem is known to be intractable since it requires access to Z L , also known as the permanent of L, and whose computation is known to be a #P-hard problem Valiant (1979) To overcome this difficulty we introduce Sinkhorn variational marginal inference, which can be computed efficiently and is straightforward to implement.

Specifically, we approximate ρ as S(L), the Sinkhorn operator applied to L (Sinkhorn, 1964) .

S(L) is defined as the (infinite) successive row and column normalization of L (Adams and Zemel, 2011; , a limit that is known to result in a doubly stochastic matrix (Altschuler et al., 2017) .

In section 2 we argue the Sinkhorn approximation is sensible, and in section 3 we describe the problem of probabilistic inference of neural identity in C.elegans and demonstrate the Sinkhorn approximation produces the best results.

Our argument bases on the well-known relation between marginal inference and the normalizing constant (Wainwright and Jordan, 2008), valid for exponential families.

Specifically, (1.1) defines an exponential family with sufficient statistic P and parameter log L.

By virtue of Theorem 3.4 in Wainwright and Jordan (2008) :

where M is the marginal polytope (here, the Birkhoff polytope, the set of doubly stochastic matrices) and A * (µ) is the dual function log Z L , i.e.

Moreover, for a given L, µ(L) achieving the supremum in (2.1) is exactly the matrix of marginals, µ(L) = ρ L and the dual function A * (µ(L)) coincides with the negative entropy of (1.1).

Then, marginal inference of ρ L and computation of the permanent Z L = perm(L) are linked by the optimization problem in (2.2).

As in any generic variational inference scheme Wainwright and Jordan (2008), we obtain an approximate ρ by replacing the variational representation of Z L in (2.1), by a different, more tractable optimization problem.

Typically, the quality of the approximated ρ depends on how tight is the approximation to Z L .

Our approximation is based on replacing the intractable dual function A * (µ) by a component-wise entropy, and whose solution is exactly S(L).

In detail, the following variational representation holds Helmbold and Warmuth, 2009 ):

By using the component-wise entropy in (2.1) we obtain an approximation of the normalizing constant, that we call as the Sinkhorn permanent (Linial et al., 2000) , perm S (L).

In the following proposition we provide bounds for this approximation.

The following bounds hold

We note the Sinkhorn approximation has recently been proposed independently (Powell and Smith, 2019) .

However, there the approximation is proposed rather heuristically, without any appeal to a theoretical framework.

Additionally, the so-called Bethe variational inference method (Wainwright and Jordan, 2008 ) is a rather general rationale for obtaining variational approximations in graphical models, where the dual function A * (µ) is approximated by the value it would take if the underlying Markov random field had a tree structure (Yedidia et al., 2001) .

This approximation has successfully been applied to permutations (Huang and Jebara, 2007; Chertkov et al., 2010; Vontobel, 2014; Tang et al., 2015) , where the corresponding approximate marginal B(L) is computed through belief propagation (Huang and Jebara, 2007; Vontobel, 2013) , enjoying also better theoretical guarantees than the Sinkhorn approximation.

Indeed, for the Bethe approximation of the permanent, perm B (·) the following bounds are known (Gurvits and Samorodnitsky, 2014; Anari and Rezaei, 2018

However, there are also important computational differences.

A single iteration of the Sinkhorn algorithms corresponds to a row and column normalization, but the message computations in the belief propagation-like routine for the Bethe approximation are more complex.

Explicit formulae of such Sinkhorn and Bethe iterations are available in Appendix C.

Fig 1 (b) shows that in practice the Bethe approximation also produces better permanent approximations, confirming theoretical predictions.

We considered the simple case where n = 8 and the permanent and marginal can be computed by enumeration, so comparisons with ground truth are possible.

However, and quite interestingly, in many cases (see Figs 1(a) and A.1(a) in the Appendix) the Sinkhorn approximation produced qualitatively better marginals, putting more mass on more non-zero entries than the Bethe approximation, regardless of possibly worse permanents.

Additionally, we observed that for moderate n the Sinkhorn approximation scaled better.

For example, if n = 710, each Bethe iteration took on average 0.035 seconds, while each Sinkhorn iteration took only 0.0027 seconds (see Fig A. 2 in the Appendix for details).

Comparison of Bethe and Sinkhorn approximations.

1,000 submatrices of size n = 8 were randomly sampled from the C.elegans dataset described in section 3.

(a) Examples of a (log) true marginal matrix ρ along with Sinkhorn and Bette approximation.

The rightmost plot is a histogram of the log permanent across the samples.

(b) Differences between approximate and true log permanent (left) and mean absolute errors of log marginals (right) for our two approximations.

We considered additional 1,000 'random' submatrices made by uniformly sampling entries between the minimum and maximum values of each C.elegans submatrix Finally, we note that sampling-based methods may be also used for marginal inference.

Indeed, quite sophisticated samplers have been proposed to show polynomial approximability of the permanent (Jerrum and Sinclair, 1989) ; however, their practical appeal is limited.

In section 3 we show that an elementary MCMC sampler failed to produce sensible marginal inferences at reasonable time.

The worm C.elegans is a unique species since their nervous system is stereotypical; i.e., the number of neurons (roughly, 300) and the connections between those neurons remain unchanged from animal to animal.

Recent advances in neurotechnology have enabled whole brain imaging so that the long-standing fundamental question about how the activity in the worm brain relates to its behavior in the world can be now studied and settled.

However, before that, a technical problem has to be solved: given volumetric images of the worm neurons have to be identified; that is, canonical labels (names) must be assigned to each.

We applied our methodology for such probabilistic neural identification in the context of NeuroPAL (Yemini et al., 2019) , a multicolor C.elegans transgene where neuron colors were designed to facilitate neural identification (see Fig 1 for an example) .

Specifically, given n observed neurons represented as vectors in R 6 (position and color), we aim to estimate the matrix of marginal ρ such that ρ k,i is the probability that observed neuron k is identified with the canonical identity i. These probabilities are relevant as they provide uncertainty estimates for model predictions, giving a much more complete picture than point estimates (e.g. a permutation found via maximum likelihood).

We consider a gaussian model for each canonical neuron, whose parameters (µ k , Σ k ) are inferred beforehand from previously annotated worms (see Yemini et al. (2019) for details).

Let π denote the permutation so that π(k) is the canonical index of the k − th observed neuron.

Then, the likelihood of observing data Y = (y k ) writes as:

(3.1)

Suppose a flat prior is assumed over P. Then, it is plain to verify that equation (3.1) induces a posterior over P that has the form of (1.1), with L defined as

Figure 2: A worm's head displaying the deterministic coloring scheme identical across all NeuroPAL worms, with neuron names (determined by a human) over each neuron.

In the context of NeuroPAL we consider a downstream task involving the computation of the approximate probabilistic neural identifies ρ.

Specifically, in this task a human is asked to manually label the neurons for which the model estimates are the most uncertain; i.e., the rows of ρ that are closest to the uniform distribution.

As the human progressively annotates neurons this uncertainty resolves and the corresponding model update lead to an increases in identification accuracy for the remaining neurons.

Ideally the human will only require a few annotations to reach a high accuracy, and therefore, as a proxy for approximation quality we measure how much faster accuracy increases in comparison to simple baselines; e.g., where at each time a neuron is randomly chosen.

Results are shown in Fig 3 , and further details are described in the Appendix.

We considered several alternatives: i) Sinkhorn approximation, ii) Bethe approximation, iii) MCMC, iv) the random baseline described above, v) a naive baseline where uncertainty estimates are made by scaling only the rows of the likelihood matrix, i.e., without imposing any one-to-one assignment structure, and vi) a 'ground truth', the protocol where the labels that are chosen are the ones where the model makes a wrong prediction (this oracle cannot be realized in practice).

Results of Sinkhorn and Bethe approximations are similar but the former slightly better, presumably a consequence of more accurate estimates of low probability marginals (see Figs 1(a) and A.1(a)).

They both are substantially better than any baseline other than the oracle.

Contrarily, we see MCMC does not provide better results than the naive baseline, suggesting lack of convergence for chain lengths leading to computational times comparable to the ones of approximated methods.

We have introduced the Sinkhorn approximation for marginal inference, and our it is a sensible alternative to sampling, and it may provide faster, simpler and more accurate approximate marginals than the Bethe approximation, despite typically leading to worse permanent approximations.

We leave for future work a thorough analysis of the relation between quality of permanent approximation and corresponding marginals.

Also, it can be verified that S(L) = diag(x)Ldiag(y), where diag(x), diag(y) are some positive vectors x, y turned into diagonal matrices (Peyré et al., 2019) .

Then,

Additionally, we obtain the (log) Sinkhorn approximation of the permanent of L, perm S (L), by evaluating S(L) in the problem it solves, (2.3).

By simple algebra and using the fact that S(L) is a doubly stochastic matrix we see that

By combining the last three displays we obtain

from which the result follows.

We used the dataset described in Yemini et al. (2019) .

This consists on ten NeuroPAL worm heads with available human labels, and with number of neurons n ranging from 180 to 195.

Each of these worms is summarized through a n × n log-likelihood matrix L computed with the methods described in (Yemini et al., 2019, Supplemental Information) .

For both the Sinkhorn and Bethe approximation we used 200 iterations.

These values led to the computation times described in Fig 1, and preliminary results showed were sufficient to ensure convergence (that is, none of the results would change dramatically for a larger number of iterations).

For the MCMC sampler we used the method described in Diaconis (2009) .

We used 100 chains of length 1000, and for each of them considered we took as samples of the multiples of 10 starting from iteration 500 on.

All results were obtained on a desktop computer with an Intel Xeon W-2125 processor.

r e t u r n np .

exp ( logP )

Bethe approximation The following is an efficient log-space implementation of the message passing algorithm described in (Vontobel, 2013, Lemma 29) , which was subsequently simplified by Pontobel (2019) .

The parameter eps is introduced for numerical estability. . .

, 710, a number of 1, 000 submatrices of size n were randomly drawn from the ten available log likelihood C.elegans matrices (see text on Appendix B, indexes were drawn with replacement).

Error bars are omitted because they were too small to be noticed.

<|TLDR|>

@highlight

New methodology for variational marginal inference of permutations based on Sinkhorn algorithm, applied to probabilistic identification of neurons