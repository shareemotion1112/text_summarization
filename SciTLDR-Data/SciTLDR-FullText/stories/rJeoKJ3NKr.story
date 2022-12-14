Typical amortized inference in variational autoencoders is specialized for a single probabilistic query.

Here we propose an inference network architecture that generalizes to unseen probabilistic queries.

Instead of an encoder-decoder pair, we can train a single inference network directly from data, using a cost function that is stochastic not only over samples, but also over queries.

We can use this network to perform the same inference tasks as we would in an undirected graphical model with hidden variables, without having to deal with the intractable partition function.

The results can be mapped to the learning of an actual undirected model, which is a notoriously hard problem.

Our network also marginalizes nuisance variables as required.

We show that our approach generalizes to unseen probabilistic queries on also unseen test data, providing fast and flexible inference.

Experiments show that this approach outperforms or matches PCD and AdVIL on 9 benchmark datasets.

Learning the parameters of an undirected probabilistic graphical model (PGM) with hidden variables using maximum likelihood (ML) is a notably difficult problem (Welling and Sutton, 2005; Kuleshov and Ermon, 2017; Li et al., 2019) .

When all variables are observed, the range of applicable techniques is broadened (Sutton and McCallum, 2005; Sutton and Minka, 2006; Sutton and McCallum, 2007; Bradley, 2013) , but the problem remains intractable in general.

When hidden variables are present, the intractability is twofold: (a) integrating out the hidden variables (also a challenge in directed models) and (b) computing the partition function.

The second problem is generally deemed to be harder (Welling and Sutton, 2005) .

After learning, the probabilistic queries are in most cases not tractable either, so one has to resort to approximations such as belief propagation or variational inference.

These approximations operate in the same way regardless of whether the model is directed, and do not need to compute the partition function.

In general, ML learning is harder than inference both in directed and undirected models, but even more so in the latter case.

Approximate inference via belief propagation (BP) or variational inference (VI) can be cast as an optimization problem.

As such, it rarely has a closed-form solution and is instead solved iteratively, which is computationally intensive.

To address this problem, one can use amortized inference.

A prime example of this are variational autoencoders (Kingma and Welling, 2013) : a learned function (typically a neural network) is combined with the reparameterization trick (Rezende et al., 2014; Titsias and L??zaro-Gredilla, 2014) to compute the posterior over the hidden variables given the visible ones.

Although a variational autoencoder (VAE) performs inference much faster than actual VI optimization, this is not without limitations: they are specialized to answer a single predefined query.

In contrast, BP and VI answer arbitrary queries, albeit usually need more computation time.

The end goal of learning the parameters of a PGM is to obtain a model that can answer arbitrary probabilistic queries.

A probabilistic query requests the distribution of a subset of the variables of the model given some (possibly soft) evidence about another subset of variables.

This allows, for instance, to train a model on full images and then perform inpainting in a test image in an arbitrary region that was not known at training time.

Since the end goal is to be able to perform arbitrary inference, in this work we suggest to learn a system that is able to answer arbitrary probabilistic queries and avoid ML learning altogether, which completely sidesteps the difficulties associated to the partition function.

This puts directed and undirected models on equal footing in terms of usability.

To this end, we first unroll inference (we will use BP, but other options are possible) over iterations into a neural network (NN) that outputs the result of an arbitrary query, and then we train said NN to increase its prediction accuracy.

At training time we randomize the queries, looking for a consistent parameterization of the NN that generalizes to new queries.

The hope for existence of such a parameterization comes from BP actually working for arbitrary queries in a graphical model with a single parameterization.

We call this approach query training (QT).

The starting point is an unnormalized PGM parameterized by ??.

Its probability density can be expressed as p(x; ??) = p(v, h; ??) ??? exp(??(v, h; ??)) , where v are the visible variables available in our data and h are the hidden variables.

A query is a binary vector q of the same dimension as v that partitions the visible variables in two subsets: One for which (soft) evidence is available (inputs) and another whose conditional probability we want to estimate (outputs).

Learning undirected models via query training this is not without limitations: they are specialized to answer a single predefined query.

In contrast, BP and VI answer arbitrary queries, albeit usually need more computation time.

The end goal of learning the parameters of a PGM is to obtain a model that can answer arbitrary probabilistic queries.

A probabilistic query requests the distribution of a subset of the variables of the model given some (possibly soft) evidence about another subset of variables.

This allows, for instance, to train a model on full images and then perform inpainting in a test image in an arbitrary region that was not known at training time.

Since the end goal is to be able to perform arbitrary inference, in this work we suggest to learn a system that is able to answer arbitrary probabilistic queries and avoid ML learning altogether, which completely sidesteps the di culties associated to the partition function.

This puts directed and undirected models on equal footing in terms of usability.

To this end, we first unroll inference (we will use BP, but other options are possible) over iterations into a neural network (NN) that outputs the result of an arbitrary query, and then we train said NN to increase its prediction accuracy.

At training time we randomize the queries, looking for a consistent parameterization of the NN that generalizes to new queries.

The hope for existence of such a parameterization comes from BP actually working for arbitrary queries in a graphical model with a single parameterization.

We call this approach query training (QT).

The starting point is an unnormalized PGM parameterized by ???.

Its probability density can be expressed as p(x; ???) = p(v, h; ???) / exp( (v, h; ???)) , where v are the visible variables available in our data and h are the hidden variables.

A query is a binary vector q of the same dimension as v that partitions the visible variables in two subsets: One for which (soft) evidence is available (inputs) and another whose conditional probability we want to estimate (outputs).

Figure 1: One step of query training.

A random sample from the training data is split according to a random query mask in input and output dimensions.

The input is processed inside the QT-NN by N identical stages, producing an estimation of the sample.

The cross-entropy between the true and estimated outputs is computed.

A random sample from the training data is split according to a random query mask in input and output dimensions.

The input is processed inside the QT-NN by N identical stages, producing an estimation of the sample.

The cross-entropy between the true and estimated outputs is computed.

The query-trained neural network (QT-NN) follows from specifying a graphical model ??(v, h; ??), a temperature T and a number of inference timesteps N over which to run parallel BP.

The general equations of the QT-NN are given next in Section 2.2, and the equations for the simple case in which the PGM is an RBM is provided in Appendix A. As depicted in Fig. 1 , a QT-NN takes as input a sample v from the dataset and a query mask q. The query q blocks the network from accessing the "output" variables, and instead only offers access to the "input" variables.

Which variables are inputs and which ones are outputs is precisely the information that q contains.

Then the QT-NN produces as output an estimationv of the whole input sample.

Obviously, we only care about how well the network estimates the variables that it did not see at the input.

So we measure how well v matches the correct v in terms of cross-entropy (CE), but only for the variables that q regards as "output".

Taking expectation wrt v and q, we get the loss function that we use to train the QT-NN

We minimize this loss wrt ??, T via stochastic gradient descent, sampling from the training data and some query distribution.

The number of QT-NN layers N is fixed a priori.

One can think of the QT-NN as a more flexible version of the encoder in a VAE: instead of hardcoding inference for a single query (normally, hidden variables given visible variables), the QT-NN also takes as input a mask q specifying which variables are observed, and provides inference results for unobserved ones.

Note that h is never observed.

For a given set of graphical model parameters ?? and temperature T we can write a feedforward function that approximately resolves arbitrary inference queries by unrolling the parallel BP equations for N iterations.

First, we combine the available evidence v and the query q into a set of unary factors.

Unary factors specify a probability density function over a variable.

Therefore, for each dimension inside v that q labels as "input", we provide a (Dirac or Kronecker) delta centered at the value of that dimension.

For the "output" dimensions and hidden variables h we set the unary factor to an uninformative, uniform density.

Finally, soft evidence, if present, can be incorporated through the appropriate density function.

The result of this process is a unary vector of factors u that contains an informative density exclusively about the inputs and whose dimensionality is the sum of the dimensionalities of v and h. Each dimension of u will be a real number for binary variables, and a full distribution in the general case.

Once v and the query q are encoded in u, we can write down the equations of parallel BP over iterations as an NN with N layers, i.e., the QT-NN.

To simplify notation, let us consider a factor graph that contains only pairwise factors.

Then the probabilistic predictions of the QT-NN and the messages from each layer to the next can be written as:

Here m (n) collects all the messages 1 that exit layer n ??? 1 and enter layer n. Messages have direction, so m (n) ij is different from m (n) ji .

Observe how the input term u is re-fed at every layer.

The output of the network is a beliefv i for each variable i, which is obtained by a softmax in the last layer.

All these equations follow simply from unrolling BP over iterations, with its messages encoded in log-space.

The portion of the parameters ?? relevant to the factor between variables i and j is represented by ?? ij = ?? ji , and the portion that only affects variable i is contained in ?? i .

Observe that all layers share the same parameters.

The functions f ?? ij (??) are directly derived from ??(x; ??) using the BP equations, and therefore inherit its parameters.

Finally, parameter T is the "temperature" of the message passing, and can be set to T = 1 to retrieve the standard sum-product belief propagation or to 0 to recover max-product belief revision.

Values in-between interpolate between sum-product and max-product and increase the flexibility of the NN.

See Appendix A for the precise equations obtained when the PGM is an RBM.

If the distribution over queries only contains queries with a single variable assigned as output (and the rest as input), and there are no hidden variables, the above cost function reduces to pseudo-likelihood training (Besag, 1975) .

Query training is superior to pseudo-likelihood (PL) in two ways: Firstly, it provides an explicit mechanism for handling hidden variables, and secondly and more importantly, it preserves learning in the face of high correlations in the input data, which results in catastrophic failure when using PL.

If two variables a and b are highly correlated, PL will fail to learn the weaker correlation between a and z, since b will always be available during training to predict a, rendering any correlation with z useless at training time.

If at test time we want to predict a from z because b is not available, the prediction will fail.

In contrast, query training removes multiple variables from the input, driving the model to better leverage all available sources of information.

Early works in learning undirected PGMs relied on contrastive energies (Hinton, 2002; Welling and Sutton, 2005) .

More recent approaches are NVIL (Kuleshov and Ermon, 2017) and AdVIL (Li et al., 2019) , with the latter being regarded as superior.

We will use an RBM in our experiments and compare QT with PCD which is very competitive in this setting (Tieleman, 2008; Marlin et al., 2010) .

We also show results for AdVIL, although it is not necessarily expected to be superior to PCD for this model.

We use exactly the same datasets and preprocessing used in the AdVIL paper, with the same RBM sizes, check (Li et al., 2019) for further details.

The random queries are generated by assigning each variable to input or output with 0.5 chance.

We report the normalized cross-entropy (NCE), which is the aggregated cross-entropy over the test data, divided by the cross-entropy of a uniform model under the same query (i.e., values below 1.0 mean a better-than-trivial model).

1.

For a fully connected graph, the number of messages is quadratic in the number of variables, showing the advantage of a sparse connectivity pattern, which can be encoded in the PGM choice.

Computing the NCE for QT is as simple as running the trained QT-NN.

PCD and AdVIL, however, cannot solve arbitrary inference queries directly and one has to resort to slow Gibbs sampling in the learned model.

Alternatively, one can turn the RBM weights learned by this methods into a QT-NN with T = 1 (essentially, running BP for a fixed number of iterations).

We also provide those results as PCD-BP and AdVIL-BP.

For PCD we train for 1000 epochs and cross-validate the learning parameter.

For AdVIL we use the code provided by the authors.

For QT we unfold BP in N = 10 layers and use ADAM to learn the weights.

The validation set is used to choose the learning rate and for early stopping.

We use minibatches of size 500.

The T parameter is learned during training.

The results are shown in Table 3 .

QT-NN produces significantly better results for most datasets (marked in boldface), showing that it has learned to generalize to new probabilistic queries on unseen data.

Query training is a general approach to learn to infer when the inference target is unknown at training time.

It offers the following advantages: 1) no need to estimate the partition function or its gradient (the "sleep" phase of other common algorithms); 2) produces an inference network, which can be faster and more accurate than iterative VI or BP because its weights are trained to compensate for the imperfections of approximate inference run for a small number of iterations; 3) arbitrary queries can be solved.

In contrast, a VAE is only trained to infer the posterior over the hidden variables, or some other constant query.

Why would QT-NNs generalize to new queries or scale well?

The worry is that only a small fraction of the exponential number of potential queries is seen during training.

The existence of a single inference network that works reasonably well for many different queries follows from the existence of a single PGM in which BP can approximate inference.

The discoverability of such a network from limited training data is not guaranteed.

However, there is hope for it, since the amount of training data required to adjust the model parameters should scale with the number of these, and not with the number of potential queries.

Just like training data should come from the same distribution as test data, the training queries must come from the same distribution the test queries to avoid "query overfitting".

In future work we will show how QT can be used in more complex undirected models, such as grid MRFs.

Other interesting research avenues are modifications to allow sample generation and unroll other inference mechanisms, such as VI.

??? We use 0 HV to represent a matrix of zeros of size H ?? V .

??? Similarly 1 V represents a matrix of ones of size V ?? 1.

??? When any of the above defined scalar functions is used with matrix arguments, the function is applied elementwise.

Some observations:

??? The Hadamard product with q effectively removes the information from the elements of v not present in the query mask, replacing them with 0, which corresponds to a uniform binary distribution in logit space.

??? The output of the network isv and??, the inferred probability of 1 for both the visible and hidden units.

The output?? is inferred but actually not used during training.

??? The computation of f w (x) as specified above is designed to be numerically robust.

It starts by computing f MP w (x), which would be the value of f w (x) for a temperature T = 0, i.e., max-product message passing, and then performs a correction on top for positive temperatures.

@highlight

Instead of learning the parameters of a graphical model from data, learn an inference network that can answer the same probabilistic queries.