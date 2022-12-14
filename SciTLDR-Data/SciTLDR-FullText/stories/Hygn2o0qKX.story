The ability of overparameterized deep networks to generalize well has been linked to the fact that stochastic gradient descent (SGD) finds solutions that lie in flat, wide minima in the training loss -- minima where the output of the network is resilient to small random noise added to its parameters.

So far this observation has been used to provide generalization guarantees only for neural networks whose parameters are either \textit{stochastic} or \textit{compressed}.

In this work, we present a general PAC-Bayesian framework that leverages this observation to provide a bound on the original network learned -- a network that is deterministic and uncompressed.

What enables us to do this is a key novelty in our approach: our framework allows us to show that if on training data, the interactions between the weight matrices satisfy certain conditions that imply a wide training loss minimum, these conditions themselves {\em generalize} to the interactions between the matrices on test data, thereby implying a wide test loss minimum.

We then apply our general framework in a setup where we assume that the pre-activation values of the network are not too small (although we assume this only on the training data).

In this setup, we provide a generalization guarantee for the original (deterministic, uncompressed) network, that does not scale with product of the spectral norms of the weight matrices -- a guarantee that would not have been possible with prior approaches.

Modern deep neural networks contain millions of parameters and are trained on relatively few samples.

Conventional wisdom in machine learning suggests that such models should massively overfit on the training data, as these models have the capacity to memorize even a randomly labeled dataset of similar size (Zhang et al., 2017; Neyshabur et al., 2015) .

Yet these models have achieved state-ofthe-art generalization error on many real-world tasks.

This observation has spurred an active line of research (Soudry et al., 2018; BID2 BID11 ) that has tried to understand what properties are possessed by stochastic gradient descent (SGD) training of deep networks that allows these networks to generalize well.

One particularly promising line of work in this area (Neyshabur et al., 2017; BID0 has been bounds that utilize the noise-resilience of deep networks on training data i.e., how much the training loss of the network changes with noise injected into the parameters, or roughly, how wide is the training loss minimum.

While these have yielded generalization bounds that do not have a severe exponential dependence on depth (unlike other bounds that grow with the product of spectral norms of the weight matrices), these bounds are quite limited: they either apply to a stochastic version of the classifier (where the parameters are drawn from a distribution) or a compressed version of the classifier (where the parameters are modified and represented using fewer bits).In this paper, we revisit the PAC-Bayesian analysis of deep networks in Neyshabur et al. (2017; and provide a general framework that allows one to use noise-resilience of the deep network on training data to provide a bound on the original deterministic and uncompressed network.

We achieve this by arguing that if on the training data, the interaction between the 'activated weight matrices' (weight matrices where the weights incoming from/outgoing to inactive units are zeroed out) satisfy certain conditions which results in a wide training loss minimum, these conditions themselves generalize to the weight matrix interactions on the test data.

After presenting this general PAC-Bayesian framework, we specialize it to the case of deep ReLU networks, showing that we can provide a generalization bound that accomplishes two goals simultaneously: i) it applies to the original network and ii) it does not scale exponentially with depth in terms of the products of the spectral norms of the weight matrices; instead our bound scales with more meaningful terms that capture the interactions between the weight matrices and do not have such a severe dependence on depth in practice.

We note that all but one of these terms are indeed quite small on networks in practice.

However, one particularly (empirically) large term that we use is the reciprocal of the magnitude of the network pre-activations on the training data (and so our bound would be small only in the scenario where the pre-activations are not too small).

We emphasize that this drawback is more of a limitation in how we characterize noise-resilience through the specific conditions we chose for the ReLU network, rather than a drawback in our PAC-Bayesian framework itself.

Our hope is that, since our technique is quite general and flexible, by carefully identifying the right set of conditions, in the future, one might be able to derive a similar generalization guarantee that is smaller in practice.

To the best of our knowledge, our approach of generalizing noise-resilience of deep networks from training data to test data in order to derive a bound on the original network that does not scale with products of spectral norms, has neither been considered nor accomplished so far, even in limited situations.

One of the most important aspects of the generalization puzzle that has been studied is that of the flatness/width of the training loss at the minimum found by SGD.

The general understanding is that flatter minima are correlated with better generalization behavior, and this should somehow help explain the generalization behavior BID7 BID6 BID8 .

Flatness of the training loss minimum is also correlated with the observation that on training data, adding noise to the parameters of the network results only in little change in the output of the network -or in other words, the network is noise-resilient.

Deep networks are known to be similarly resilient to noise injected into the inputs (Novak et al., 2018) ; but note that our theoretical analysis relies on resilience to parameter perturbations.

While some progress has been made in understanding the convergence and generalization behavior of SGD training of simple models like two-layered hidden neural networks under simple data distributions (Neyshabur et al., 2015; Soudry et al., 2018; BID2 BID11 , all known generalization guarantees for SGD on deeper networks -through analyses that do not use noise-resilience properties of the networks -have strong exponential dependence on depth.

In particular, these bounds scale either with the product of the spectral norms of the weight matrices BID0 BID1 or their Frobenius norms BID4 .

In practice, the weight matrices have a spectral norm that is as large as 2 or 3, and an even larger Frobenius norm that scales with ??? H where H is the width of the network i.e., maximum number of hidden units per layer.1 Thus, the generalization bound scales as say, 2 D or H D 2 , where D is the depth of the network.

At a high level, the reason these bounds suffer from such an exponential dependence on depth is that they effectively perform a worst case approximation of how the weight matrices interact with each other.

For example, the product of the spectral norms arises from a naive approximation of the Lipschitz constant of the neural network, which would hold only when the singular values of the 1 To understand why these values are of this order in magnitude, consider the initial matrix that is randomly initialized with independent entries with variance 1 ??? H. It can be shown that the spectral norm of this matrix, with high probability, lies near its expected value, near 2 and the Frobenius norm near its expected value which is ??? H. Since SGD is observed not to move too far away from the initialization regardless of H (Nagarajan & Kolter, 2017) , these values are more or less preserved for the final weight matrices.weight matrices all align with each other.

However, in practice, for most inputs to the network, the interactions between the activated weight matrices are not as adverse.

By using noise-resilience of the networks, prior approaches BID0 Neyshabur et al., 2017) have been able to derive bounds that replace the above worst-case approximation with smaller terms that realistically capture these interactions.

However, these works are limited in critical ways.

BID0 use noise-resilience of the network to modify and "compress" the parameter representation of the network, and derive a generalization bound on the compressed network.

While this bound enjoys a better dependence on depth because its applies to a compressed network, the main drawback of this bound is that it does not apply on the original network.

On the other hand, Neyshabur et al. (2017) take advantage of noise-resilience on training data by incorporating it within a PAC-Bayesian generalization bound BID14 .

However, their final guarantee is only a bound on the expected test loss of a stochastic network.

In this work, we revisit the idea in Neyshabur et al. (2017) , by pursuing the PAC-Bayesian framework BID14 to answer this question.

The standard PAC-Bayesian framework provides generalization bounds for the expected loss of a stochastic classifier, where the stochasticity typically corresponds to Gaussian noise injected into the parameters output by the learning algorithm.

However, if the classifier is noise-resilient on both training and test data, one could extend the PAC-Bayesian bound to a standard generalization guarantee on the deterministic classifier.

Other works have used PAC-Bayesian bounds in different ways in the context of neural networks.

BID9 ; BID3 optimize the stochasticity and/or the weights of the network in order to numerically compute good (i.e., non-vacuous) generalization bounds on the stochastic network.

BID0 derive generalization bounds on the original, deterministic network by working from the PAC-Bayesian bound on the stochastic network.

However, as stated earlier, their work does not make use of noise resilience in the networks learned by SGD.OUR CONTRIBUTIONS The key contribution in our work is a general PAC-Bayesian framework for deriving generalization bounds while leveraging the noise resilience of a deep network.

While our approach is applied to deep networks, we note that it is general enough to be applied to other classifiers.

In our framework, we consider a set of conditions that when satisfied by the network, makes the output of the network noise-resilient at a particular input datapoint.

For example, these conditions could characterize the interactions between the activated weight matrices at a particular input.

To provide a generalization guarantee, we assume that the learning algorithm has found weights such that these conditions hold for the weight interactions in the network on training data (which effectively implies a wide training loss minimum).

Then, as a key step, we generalize these conditions over to the weight interactions on test data (which effectively implies a wide test loss minimum) 2 .

Thus, with the guarantee that the classifier is noise-resilient both on training and test data, we derive a generalization bound on the test loss of the original network.

Finally, we apply our framework to a specific set up of ReLU based feedforward networks.

In particular, we first instantiate the above abstract framework with a set of specific conditions, and then use the above framework to derive a bound on the original network.

While very similar conditions have already been identified in prior work BID0 Neyshabur et al., 2017 ) (see Appendix G for an extensive discussion of this), our contribution here is in showing how these conditions generalize from training to test data.

Crucially, like these works, our bound does not have severe exponential dependence on depth in terms of products of spectral norms.

We note that in reality, all but one of our conditions on the network do hold on training data as necessitated by the framework.

The strong, non-realistic condition we make is that the pre-activation values of the network are sufficiently large, although only on training data; however, in practice a small proportion of the pre-activation values can be arbitrarily small.

Our generalization bound scales inversely with the smallest absolute value of the pre-activations on the training data, and hence in practice, our bound would be large.

Intuitively, we make this assumption to ensure that under sufficiently small parameter perturbations, the activation states of the units are guaranteed not to flip.

It is worth noting that BID0 Neyshabur et al. (2017) too require similar, but more realistic assumptions about pre-activation values that effectively assume only a small proportion of units flip under noise.

However, even under our stronger condition that no such units exist, it is not apparent how these approaches would yield a similar bound on the deterministic, uncompressed network without generalizing their conditions to test data.

We hope that in the future our work could be developed further to accommodate the more realistic conditions from BID0 Neyshabur et al. (2017) .

In this section, we present our general PAC-Bayesian framework that uses noise-resilience of the network to convert a PAC-Bayesian generalization bound on the stochastic classifier to a generalization bound on the deterministic classifier.

NOTATION.

Let KL(??? ???) denote the KL-divergence.

Let ??? , ??? ??? denote the 2 norm and the ??? norms of a vector, respectively.

Let ??? 2 , ??? F , ??? 2,??? denote the spectral norm, Frobenius norm and maximum row 2 norm of a matrix, respectively.

Consider a K-class learning task where the labeled datapoints (x, y) are drawn from an underlying distribution D over X ?? {1, 2, ???, K} where X ??? R N .

We consider a classifier parametrized by weights W. For a given input x and class k, we denote the output of the classifier by f (x; W) [k] .

In our PAC-Bayesian analysis, we will use U ??? N (0, ?? 2 ) to denote parameters whose entries are sampled independently from a Gaussian, and W + U to denote the entrywise addition of the two sets of parameters.

We use DISPLAYFORM0 Given a training set S of m samples, we let (x, y) ??? S to denote uniform sampling from the set.

Finally, for any ?? > 0, let L ?? (f (x; W) , y) denote a margin-based loss such that the loss is 0 only when f (x; W) [y] ??? max j???y f (x; W) [j] + ??, and 1 otherwise.

Note that L 0 corresponds to 0-1 error.

See Appendix A for more notations.

TRADITIONAL PAC-BAYESIAN BOUNDS.

The PAC-Bayesian framework BID14 b) allows us to derive generalization bounds for a stochastic classifier.

Specifically, letW be a random variable in the parameter space whose distribution is learned based on training data S. Let P be a prior distribution in the parameter space chosen independent of the training data.

The PAC-Bayesian framework yields the following generalization bound on the 0-1 error of the stochastic classifier that holds with probability 1 ??? ?? over the draw of the training set S of m samples 3 : DISPLAYFORM1 Typically, and in the rest of this discussion,W is a Gaussian with covariance ?? 2 I for some ?? > 0 centered at the weights W learned based on the training data.

Furthermore, we will set P to be a Gaussian with covariance ?? 2 I centered at the random initialization of the network like in BID3 , instead of at the origin, like in BID0 .

This is because the resulting KL-divergence -which depends on the distance between the means of the prior and the posterior -is known to be smaller, and to save a ??? H factor in the bound (Nagarajan & Kolter, 2017).

To extend the above PAC-Bayesian bound to a standard generalization bound on a deterministic classifier W, we need to replace the training and the test loss of the stochastic classifier with that of the original, deterministic classifier.

However, in doing so, we will have to introduce extra terms in the upper bound to account for the perturbation suffered by the train and test loss under the Gaussian perturbation of the parameters.

To tightly bound these two terms, we need that the network is noise-resilient on training and test data respectively.

Our hope is that if the learning algorithm has found weights such that the network is noise-resilient on the training data, we can then generalize this noise-resilience over to test data as well, allowing us to better bound the excess terms.

DISPLAYFORM0 For convenience, we also define an additional R + 1th set to be the singleton set containing the margin of the classifier on the input: f (x; W) [y] ??? max j???y f (x; W) [j] .

Note that if this term is positive (negative) then the classification is (in)correct.

We will also denote the constant ??? ??? R+1,1 as ?? class .ORDERING OF THE SETS OF PROPERTIES We now impose a crucial constraint on how these sets of properties depend on each other.

Roughly speaking, we want that for a given input, if the first r ??? 1 sets of properties approximately satisfy the condition in Equation 1, then the properties in the rth set are noise-resilient i.e., under random parameter perturbations, these properties do not suffer much perturbation.

This kind of constraint would naturally hold for deep networks if we have chosen the properties carefully e.g., we will show that, for any given input, the perturbation in the pre-activation values of the dth layer is small as long as the absolute pre-activation values in the layers below d ??? 1 are large, and a few other norm-bounds on the lower layer weights are satisfied.

We formalize the above requirement by defining expressions ??? r,l (??) that bound the perturbation in the properties ?? r,l , in terms of the variance ?? 2 of the parameter perturbations.

For any r ??? R + 1 and for any (x, y), our framework requires the following to hold: DISPLAYFORM1 Let us unpack the above constraint.

First, although the above constraint must hold for all inputs (x, y), it effectively applies only to those inputs that satisfy the pre-condition of the if-then statement: namely, it applies only to inputs (x, y) that approximately satisfy the first r ??? 1 conditions in DISPLAYFORM2 .

Next, we discuss the second part of the above if-then statement which specifies a probability term that is required to be small for all such inputs.

In words, the first event within the probability term above is the event that for a given random perturbation U, the properties involved in the rth condition suffer a large perturbation.

The second is the event that the properties involved in the first r ??? 1 conditions do not suffer much perturbation; but, given that these r ??? 1 conditions already hold approximately, this second event implies that these conditions are still preserved approximately under perturbation.

In summary, our constraint requires the following: for any input on which the first r ??? 1 conditions hold, there should be very few parameter perturbations that significantly perturb the rth set of properties while preserving the first r ??? 1 conditions.

When we instantiate the framework, we have to derive closed form expressions for the perturbation bounds ??? r,l (??) (in terms of only ?? and the constants ??? ??? r,l ).

As we will see, for ReLU networks, we will choose the properties in a way that this constraint naturally falls into place in a way that the perturbation bounds ??? r,l (??) do not grow with the product of spectral norms (Lemma E.1).THEOREM STATEMENT In this setup, we have the following 'margin-based' generalization guarantee on the original network.

That is, we bound the 0-1 test error of the network by a margin-based error on the training data.

Our generalization guarantee, which scales linearly with the number of conditions R, holds under the setting that the training algorithm always finds weights such that on the training data, the conditions in Equation 1 is satisfied for all r = 1, ???, R. Theorem 3.1.

Let ?? * be the maximum standard deviation of the Gaussian parameter perturbation such that the constraint in Equation 2 holds with ??? r,l (?? ??? ) ??? ??? ??? r,l ???r ??? R + 1 and ???l.

Then, for any ?? > 0, with probability 1 ??? ?? over the draw of samples S from D m , for any W we have that, if W satisfies the conditions in Equation 1 for all r ??? R and for all training examples (x, y) ??? S, then DISPLAYFORM3 The crux of our proof (in Appendix D) lies in generalizing the conditions of Equation 1 satisfied on the training data to test data one after the other, by proving that they are noise-resilient on both training and test data.

Crucially, after we generalize the first r ??? 1 conditions from training data to test data (i.e., on most test and training data, the r ??? 1 conditions are satisfied), we will have from Equation 2 that the rth set of properties are noise-resilient on both training and test data.

Using the noise-resilience of the rth set of properties on test/train data, we can generalize even the rth condition to test data.

We emphasize a key, fundamental tool that we present in Theorem C.1 to convert a generic PACBayesian bound on a stochastic classifier, to a generalization bound on the deterministic classifier.

Our technique is at a high level similar to approaches in BID12 BID13 .

In Section C.1, we argue how this technique is more powerful than other approaches in Neyshabur et al. FORMULA2 ; BID10 ; BID5 in leveraging the noiseresilience of a classifier.

The high level argument is that, to convert the PAC-Bayesian bound, these latter works relied on a looser output perturbation bound, one that holds on all possible inputs, with high probability over all perturbations i.e., a bound on max x f (x; W) ??? f (x; W + U) ??? w.h.p over draws of U. In contrast, our technique relies on a subtly different but significantly tighter bound: a bound on the output perturbation that holds with high probability given an input i.e., a bound on f (x; W) ??? f (x; W + U) ??? w.h.p over draws of U for each x. When we do instantiate our framework as in the next section, this subtle difference is critical in being able to bound the output perturbation without suffering from a factor proportional to the product of the spectral norms of the weight matrices (which is the case in Neyshabur et al. FORMULA2 ).

NOTATION.

In this section, we apply our framework to feedforward fully connected ReLU networks of depth D (we care about D > 2) and width H (which we will assume is larger than the input dimensionality N , to simplify our proofs) and derive a generalization bound on the original network that does not scale with the product of spectral norms of the weight matrices.

Let ?? (???) denote the ReLU activation.

We consider a network parameterized by DISPLAYFORM0 We denote the value of the hth hidden unit on the dth layer before and after the activation by DISPLAYFORM1 to be the Jacobian of the pre-activations of layer d with respect to the pre-activations of layer d ??? for d ??? ??? d (each row in this Jacobian corresponds to a unit in layer d).

In short, we will call this, Jacobian d d ??? .

Let Z denote the random initialization of the network.

Informally, we consider a setting where the learning algorithm satisfies the following conditions on the training data that make it noise-resilient on training data: a) the 2 norm of the hidden layers are all small, b) the pre-activation values are all sufficiently large in magnitude, c) the Jacobian of any layer with respect to a lower layer, has rows with a small 2 norm, and has a small spectral norm.

We cast these conditions in the form of Equation 1 by appropriately defining the properties ??'s and the margins ??? ??? 's in the general framework.

We note that these properties are quite similar to those already explored in BID0 ; Neyshabur et al. FORMULA2 ; we provide more intuition about these properties, and how we cast them in our framework in Appendix E.1.Having defined these properties, we first prove in Lemma E.1 in Appendix E a guarantee equivalent to the abstract inequality in Equation 2.

Essentially, we show that under random perturbations of the parameters, the perturbation in the output of the network and the perturbation in the input-dependent properties involved in (a), (b), (c) themselves can all be bounded in terms of each other.

Crucially, these perturbation bounds do not grow with the spectral norms of the network.

Having instantiated the framework as above, we then instantiate the bound provided by the framework.

Our generalization bound scales with the bounds on the properties in (a) and (c) above as satisfied on the training data, and with the reciprocal of the property in (b) i.e., the smallest absolute value of the pre-activations on the training data.

Additionally, our bound has an explicit dependence on the depth of the network, which arises from the fact that we generalize R = O(D) conditions.

Most importantly, our bound does not have a dependence on the product of the spectral norms of the weight matrices.

Theorem 4.1. (shorter version; see Appendix F for the complete statement) For any margin ?? class > 0, and any ?? > 0, with probability 1 ??? ?? over the draw of samples from D m , for any W, we have that: DISPLAYFORM2 (an upper bound on the spectral norm of the Jacobian for each layer).In FIG0 , we show how the terms in the bound vary for networks of varying depth with a small width of H = 40 on the MNIST dataset.

We observe that B layer-2 , B output , B jac-row-2 , B jac-spec typically lie in the range of [10 0 , 10 2 ] and scale with depth as ??? 1.57 D .

In contrast, the equivalent term from Neyshabur et al. FORMULA2 consisting of the product of spectral norms can be as large as 10 3 or 10 5 and scale with D more severely as 2.15 D .The bottleneck in our bound is B preact , which scales inversely with the magnitude of the smallest absolute pre-activation value of the network.

In practice, this term can be arbitrarily large, even though it does not depend on the product of spectral norms/depth.

This is because some hidden units can have arbitrarily small absolute pre-activation values -although this is true only for a small proportion of these units.

To give an idea of the typical, non-pathological magnitude of the pre-activation values, we plot two other variations of B preact : a) 5%-B preact which is calculated by ignoring 5% of the training datapoints with the smallest absolute pre-activation values and b) median-B preact which is calculated by ignoring half the hidden units in each layer with the smallest absolute pre-activation values for each input.

We observe that median-B preact is quite small (of the order of 10 2 ), while 5%-B preact , while large (of the order of 10 4 ), is still orders of magnitude smaller than B preact .

In Figure 2 we show how our overall bound and existing product-of-spectral-norm-based bounds BID1 BID0 vary with depth.

While our bound is orders of magnitude larger than prior bounds, the key point here is that our bound grows with depth as 1.57 D while prior bounds grow with depth as 2.15D indicating that our bound should perform asymptotically better with respect to depth.

Indeed, we verify that our bound obtains better values than the other existing bounds when D = 28 (see Figure 2 b ).

We also plot hypothetical variations of our bound replacing B preact with 5%-B preact (see "Ours-5%") and median-B preact (see "Ours-Median") both of which perform orders of magnitude better than our actual bound (note that these two hypothetical bounds do not actually hold good).

In fact for larger depth, the bound with 5%-B preact performs better than all other bounds (including existing bounds).

This indicates that the only bottleneck in our bound comes from the dependence on the smallest pre-activation magnitudes, and if this particular dependence is addressed, our bound has the potential to achieve tighter guarantees for even smaller D such as D = 8.

In the left, we vary the depth of the network (fixing H = 40) and plot the logarithm of various generalization bounds ignoring the dependence on the training dataset size and a log(DH) factor in all of the considered bounds.

Specifically, we consider our bound, the hypothetical versions of our bound involving 5%-B preact and median-B preact respectively, and the bounds from Neyshabur et al. DISPLAYFORM3 and Bartlett et al. FORMULA2 maxx DISPLAYFORM4 both of which have been modified to include distance from initialization instead of distance from origin for a fair comparison.

Observe the last two bounds have a plot with a larger slope than the other bounds indicating that they might potentially do worse for a sufficiently large D. Indeed, this can be observed from the plots on the right where we report the distribution of the logarithm of these bounds for D = 28 across 12 runs (although under training settings different from the experiments on the left; see Appendix F.3 for the exact details).We refer the reader to Appendix F.3 for added discussion where we demonstrate how all the quantities in our bound vary with depth for H = 1280 ( Finally, as noted before, we emphasize that the dependence of our bound on the pre-activation values is a limitation in how we characterize noise-resilience through our conditions rather than a drawback in our general PAC-Bayesian framework itself.

Specifically, using the assumed lower bound on the pre-activation magnitudes we can ensure that, under noise, the activation states of the units do not flip; then the noise propagates through the network in a tractable, "linear" manner.

Improving this analysis is an important direction for future work.

For example, one could modify our analysis to allow perturbations large enough to flip a small proportion of the activation states; one could potentially formulate such realistic conditions by drawing inspiration from the conditions in Neyshabur et al. FORMULA2 ; BID0 .However, we note that even though these prior approaches made more realistic assumptions about the magnitudes of the pre-activation values, the key limitation in these approaches is that even under our non-realistic assumption, their approaches would yield bounds only on stochastic/compressed networks.

Generalizing noise-resilience from training data to test data is crucial to extending these bounds to the original network, which we accomplish.

In this work, we introduced a novel PAC-Bayesian framework for leveraging the noise-resilience of deep neural networks on training data, to derive a generalization bound on the original uncompressed, deterministic network.

The main philosophy of our approach is to first generalize the noise-resilience from training data to test data using which we convert a PAC-Bayesian bound on a stochastic network to a standard margin-based generalization bound.

We apply our approach to ReLU based networks and derive a bound that scales with terms that capture the interactions between the weight matrices better than the product of spectral norms.

For future work, the most important direction is that of removing the dependence on our strong assumption that the magnitude of the pre-activation values of the network are not too small on training data.

More generally, a better understanding of the source of noise-resilience in deep ReLU networks would help in applying our framework more carefully in these settings, leading to tighter guarantees on the original network.

We will use upper-case symbols to denote matrices, and lower-case bold-face symbols to denote vectors.

In order to make the mathematical statements/derivations easier to read, if we want to emphasize a term, say x, we write, x.

Recall that we consider a neural netork of depth D (i.e., D ??? 1 hidden layers and one output layer) mapping from R N ??? R K , where K is the number of class labels in the learning task.

The layers are fully connected with H units in each hidden layer, and with ReLU activations ?? (???) on all the hidden units and linear activations on the output units.

We denote the parameters of the network using the symbol W, which in turn denotes a set of weight matrices DISPLAYFORM0 to be the Jacobian corresponding to the pre-activation values of layer d with respect to the pre-activation values of layer d ??? on an input x. That is, DISPLAYFORM1 In other words, this corresponds to the product of the 'activated' portion of the matrices DISPLAYFORM2 , where the weights corresponding to inactive inputs are zeroed out.

In short, we will call this '

Jacobian d d ??? '.

Note that each row in this Jacobian corresponds to a unit on the dth layer, and each column corresponds to a unit on the d ??? th layer.

We will denote the parameters of a random initialization of the network by Z = (Z 1 , Z 2 , ???, Z d ).

Let D be an underlying distribution over R N ?? {1, 2, ???, K} from which the data is drawn.

In our PAC-Bayesian analysis, we will use U to denote a set of D weight matrices U 1 , U 2 , ???, U D whose entries are sampled independently from a Gaussian.

Furthermore, we will use U d to denote only the first d of the randomly sampled weight matrices, and W + U d to denote a network where the d random matrices are added to the first d weight matrices in W. Note that W + U 0 = W. Thus, f (x; W + U d ) is the output of a network where the first d weight matrices have been perturbed.

In our analysis, we will also need to study a perturbed network where the hidden units are frozen to be at the activation state they were at before the perturbation; we will use the notation W[+U d ] to denote the weights of such a network.

For our statements regarding probability of events, we will use ???, ???, and ?? to denote the intersection, union and complement of events (to disambiguate from the set operators).

In this section, we present some standard results.

The first two results below will be useful for our noise resilience analysis.

HOEFFDING BOUND Lemma B.1.

For i = 1, 2, ???, n, let X i be independent random variables sampled from a Gaussian with mean ?? i and variance ?? 2 i .

Then for all t ??? 0, we have: DISPLAYFORM0 Or alternatively, for ?? ??? (0, 1] DISPLAYFORM1 Note that an identical inequality holds good symmetrically for the event ??? n i=1 X i ??? ?? i ??? ???t, and so the probability that the event ??? n i=1 X i ??? ?? i > t holds, is at most twice the failure probability in the above inequalities.

Lemma B.2.

Let U be a H 1 ?? H 2 matrix where each entry is sampled from N (0, ?? 2 ).

Let x be an arbitrary vector in DISPLAYFORM0 Proof.

U x is a random vector sampled from a multivariate Gaussian with mean E[U x] = 0 and co-variance E[U xx T U T ].

The (i, j)th entry in this covariance matrix is E[(u DISPLAYFORM1 2 .

When i ??? j, since u i and u j are independent random variables, we will have E[(u DISPLAYFORM2 SPECTRAL NORM OF ENTRY-WISE GAUSSIAN MATRIX The following result (Tropp, 2012) bounds the spectral norm of a matrix with Gaussian entries, with high probability: DISPLAYFORM3 2 ) or alternatively, for any ?? > 0, DISPLAYFORM4 2 ) KL DIVERGENCE OF GAUSSIANS.

We will use the following KL divergence equality to bound the generalization error in our PAC-Bayesian analyses.

Lemma B.4.

Let P be the spherical Gaussian N (?? 1 , ?? 2 I) and Q be the spherical Gaussian N (?? 2 , ?? 2 I).

Then, the KL-divergence between Q and P is: DISPLAYFORM5 In this section, we will present our main PAC-Bayesian theorem that will guide our analysis of generalization in our framework.

Concretely, our result extends the generalization bound provided by conventional PAC-Bayesian analysis BID13 -which is a generalization bound on the expected loss of a distribution of classifiers i.e., a stochastic classifier -to a generalization bound on a deterministic classifier.

The way we reduce the PAC-Bayesian bound to a standard generalization bound, is different from the one pursued in previous works BID0 BID10 .The generalization bound that we state below is a bit more general than standard generalization bounds on deterministic networks.

Typically, generalization bounds are on the classification error; however, as discussed in the main paper we will be dealing with generalizing multiple different conditions on the interactions between the weights of the network from the training data to test data.

So to state a bound that is general enough, we consider a set of generic functions ?? r (W, x, y) for r = 1, 2, ???R ??? (we use R ??? to distinguish it from R, the number of conditions in the abstract classifier of Section 3.1).

Each of these functions compute a scalar value that corresponds to some input-dependent property of the network with parameters W for the datapoint (x, y).

As an example, this property could simply be the margin of the function on the yth class i.e., f (x; W) [y] ??? max j???y f (x; W) [j] .

Theorem C.1.

Let P be a prior distribution over the parameter space that is chosen independent of the training dataset.

Let U be a random variable sampled entrywise from N (0, ?? 2 ).

Let ?? r (???, ???, ???) and ??? r > 0 for r = 1, 2, ???R ??? , be a set of input-dependent properties and their corresponding margins.

We define the network W to be noise-resilient with respect to all these functions, at a given data point (x, y) if: DISPLAYFORM6 , W) denote the probability over the random draw of a point (x, y) drawn from D, that the network with weights W is not noise-resilient at (x, y) according to Equation 3.

DISPLAYFORM7 , W) denote the fraction of data points (x, y) in a dataset S for which the network is not noise-resilient according to Equation 3.

Then for any ??, with probability 1 ??? ?? over the draws of a sample set S = {(x i , y i ) ??? D i = 1, 2, ???, m}, for any W we have: DISPLAYFORM8 The reader maybe curious about how one would bound the term ?? D in the above bound, as this term corresponds to noise-resilience with respect to test data.

This is precisely what we bound later when we generalize the noise-resilience-related conditions satisfied on train data over to test data.

The above approach differs from previous approaches used by BID0 ; BID10 in how strong a noise-resilience we require of the classifier to provide the generalization guarantee.

The stronger the noise-resilience requirement, the more price we have to pay when we jump from the PAC-Bayesian guarantee on the stochastic classifier to a guarantee on the deterministic classifier.

We argue that our noise-resilience requirement is a much milder condition and therefore promises tighter guarantees.

Our requirement is in fact philosophically similar to BID12 BID13 , although technically different.

More concretely, to arrive at a reasonable generalization guarantee in our setup, we would need that ?? D and?? S are both only as large as O(1 ??? m).

In other words, we would want the following for (x, y) ??? D and for (x, y) ??? S: DISPLAYFORM0 Previous works require a noise resilience condition of the form that with high probability a particular perturbation does not perturb the classifier output on any input.

For example, the noise-resilience condition used in BID0 written in terms of our notations, would be: DISPLAYFORM1 The main difference between the above two formulations is in what makes a particular perturbation (un)favorable for the classifier.

In our case, we deem a perturbation unfavorable only after fixing the datapoint.

However, in the earlier works, a perturbation is deemed unfavorable if it perturbs the classifier output sufficiently on some datapoint from the domain of the distribution.

While this difference is subtle, the earlier approach would lead to a much more pessimistic analysis of these perturbations.

In our analysis, this weakened noise resilience condition will be critical in analyzing the Gaussian perturbations more carefully than in Neyshabur et al. FORMULA2 i.e., we can bound the perturbation in the classifier output more tightly by analyzing the Gaussian perturbation for a fixed input point.

Note that one way our noise resilience condition would seem stronger in that on a given datapoint we want less than 1 ??? m mass of the perturbations to be unfavorable for us, while in previous bounds, there can be as much as 1 2 probability mass of perturbations that are unfavorable.

In our analysis, this will only weaken our generalization bound by a ln ??? m factor in comparison to previous bounds (while we save other significant factors).

Proof.

The starting point of our proof is a standard PAC-Bayesian theorem BID13 which bounds the generalization error of a stochastic classifier.

Let P be a data-independent prior over the parameter space.

Let L(W, x, y) be any loss function that takes as input the network parameter, and a datapoint x and its true label y and outputs a value in [0, 1].

Then, we have that, with probability 1 ??? ?? over the draw of S ??? D m , for every distribution Q over the parameter space, the following holds: DISPLAYFORM0 In other words, the statement tells us that except for a ?? proportion of bad draws of m samples, the test loss of the stochastic classifierW ??? Q would be close to its train loss.

This holds for every possible distribution Q, which allows us to cleverly choose Q based on S.

As is the convention, we choose Q to be the distribution of the stochastic classifier picked from N (W, ?? 2 I) i.e., a Gaussian perturbation of the deterministic classifier W.RELATING TEST LOSS OF STOCHASTIC CLASSIFIER TO DETERMINISTIC CLASSIFIER.

Now our task is to bound the loss for the deterministic classifier W, Pr (x,y)???D [???r ?? r (W, x, y) < 0].

To this end, let us define the following margin-based variation of this loss for some c ??? 0: DISPLAYFORM1 and so we have Pr (x,y)???D [???r ?? r (W, x, y) DISPLAYFORM2 First, we will bound the expected L 0 of a deterministic classifier by the expected L 1 2 of the stochastic classifier; then we will bound the test L 1 2 of the stochastic classifier using the PAC-Bayesian bound.

We will split the expected loss of the deterministic classifier into an expectation over datapoints for which it is noise-resilient with respect to Gaussian noise and an expectation over the rest.

To write this out, we define, for a datapoint (x, y), N(W, x, y) to be the event that W is noise-resilient at (x, y) as defined in Equation 3 in the theorem statement: DISPLAYFORM3 To further continue the upper bound on the left hand side, we turn our attention to the stochastic classifier's loss on the noise-resilient part of the distribution D (we will lower bound this term in terms of the first term on the right hand side above).

For simplicity of notations, we will write D ??? to denote the distribution D conditioned on N(W, x, y).

Also, let U(W, x, y) be the favorable event that for a given data point (x, y) and a draw of the stochastic classifier,W, it is the case that for every r, ?? r (W, x, y) ??? ?? r (W, x, y) ??? ??? r 2.

Then, the stochastic classifier's loss L 1 2 on D ??? is: DISPLAYFORM4 splitting the inner expectation over the favorable and unfavorable perturbations, and using linearity of expectations, DISPLAYFORM5 to lower bound this, we simply ignore the second term (which is positive) DISPLAYFORM6 Next, we use the following fact: if L 1 2 (W, x, y) = 0, then for all r, ?? r (W, x, y) ??? ??? r 2 and ifW is a favorable perturbation of W, then for all r, ?? r (W, DISPLAYFORM7 Hence ifW is a favorable perturbation then, DISPLAYFORM8 .

Therefore, we can lower bound the above expression by replacing the stochastic classifier with the deterministic classifier (and thus ridding ourselves of the expectation over Q): DISPLAYFORM9 Since the favorable perturbations for a fixed datapoint drawn from D ??? have sufficiently high probability (that is, PrW ???Q U(W, x, y) ??? 1 ??? 1 ??? m), we have: DISPLAYFORM10 Thus, we have a lower bound on the stochastic classifier's loss that is in terms of the deterministic classifier's loss on the noise-resilient datapoints.

Rearranging it, we get an upper bound on the latter: DISPLAYFORM11 Thus, we have an upper bound on the expected loss of the deterministic classifier W on the noiseresilient part of the distribution.

Plugging this back in the first term of the upper bound on the deterministic classifier's loss on the whole distribution D in Equation 5 we get : DISPLAYFORM12 , W) rearranging, we get: DISPLAYFORM13 rewriting the expectation over D ??? explicitly as an expectation over D conditioned on N(W, x, y), we get: DISPLAYFORM14 the first term above is essentially an expectation of a loss over the distribution D with the loss set to be zero over the non-noise-resilient datapoints and set to be L 1 2 over the noise-resilient datapoints; thus we can upper bound it with the expectation of the L 1 2 loss over the whole distribution D: DISPLAYFORM15 Now observe that we can upper bound the first term here using the PAC-Bayesian bound by plugging in L 1 2 for the generic L in FIG5 ; however, the bound would still be in terms of the stochastic classifier's train error.

To get the generalization bound we seek, which involves the deterministic classifier's train error, we need to take one final step mirroring these tricks on the train loss.

LOSS.

Our analysis here is almost identical to the previous analysis.

Instead of working with the distribution D and D ??? we will work with the training data set S and a subset of it S ??? for which noise resilience property is satisfied by W. Below, to make the presentation neater, we use (x, y) ??? S to denote uniform sampling from S.

First, we upper bound the stochastic classifier's train loss (L 1 2 ) as follows: DISPLAYFORM0 splitting over the noise-resilient points S ??? ((x, y) ??? S for which N(W, x, y) holds) like in Equation 5, we can upper bound as: DISPLAYFORM1 We can upper bound the first term by first splitting it over the favorable and unfavorable perturbations like we did before: DISPLAYFORM2 To upper bound this, we apply a similar argument.

First, if L 1 2 (W, x, y) = 1, then ???r such that ?? r (W, x, y) < ??? r 2 and ifW is a favorable perturbation then for that value of r, ?? r (W, x, y) < ?? r (W, x, y) + ??? r 2 < ??? r .

Thus ifW is a favorable perturbation then, L 1 (W, x, y) = 1 whenever L 1 2 (W, x, y) = 1 i.e., L 1 2 (W, x, y) ??? L 1 (W, x, y).

Next, we use the fact that the unfavorable perturbations for a fixed datapoint drawn from S ??? have sufficiently low probability i.e., PrW ???Q ??U(W, x, y) ??? 1 ??? m. Then, we get the following upper bound on the above equations, by replacing the stochastic classifier with the deterministic classifier (and thus ignoring the expectation over Q): DISPLAYFORM3 Plugging this back in the first term of Equation 7, we get: DISPLAYFORM4 , W) since the first term is effectively the expectation of a loss over the whole distribution with the loss set to be zero on the non-noise-resilient points and set to L 1 over the rest, we can upper bound it by setting the loss to be L 1 over the whole distribution:

DISPLAYFORM5 Applying the above upper bound and the bound in Equation 6 into the PAC-Bayesian result of Equation 4 yields our result (Note that combining these equations would produce the term DISPLAYFORM6 which is at most DISPLAYFORM7 , which we reflect in the final bound. ).

In this section, we present the proof for the abstract generalization guarantee presented in Section 3.

Our proof is based on the following recursive inequality that we demonstrate for all r ??? R (we will prove a similar, but slightly different inequality for r = R + 1): DISPLAYFORM0 Recall that the rth condition in Equation 1 is that ???l, ?? r,l > ??? ??? r,l .

Above, we bound the probability mass of test points such that any one of the first r conditions in Equation 1 is not even approximately satisfied, in terms of the probability mass of points where one of the first r ??? 1 conditions is not even approximately satisfied, and a term that corresponds to how much error there can be in generalizing the rth condition from the training data.

Our proof crucially relies on Theorem C.1.

This theorem provides an upper bound on the proportion of test data that fail to satisfy a set of conditions, in terms of four quantities.

The first quantity is the proportion of training data that do not satisfy the conditions; the second and third quantities, which we will in short refer to as?? S and?? D , correspond to the proportion of training and test data on which the properties involved in the conditions are not noise-resilient.

The fourth quantity is the generalization error.

First, we consider the base case when r = 1, and apply the PAC-Bayes-based guarantee from Theorem C.1 on the first set of properties {?? 1,1 , ?? 1,2 , ???} and their corresponding constants DISPLAYFORM1 Effectively this establishes that the noise-resilience requirement of Equation 3 in Theorem C.1 holds on all possible inputs, thus proving our claim that the terms?? S and?? D would be zero.

Thus, we will get that DISPLAYFORM2 which proves the recursion statement for the base case.

To prove the recursion for some arbitrary r ??? R, we again apply the PAC-Bayes-based guarantee from Theorem C.1, but on the union of the first r sets of properties.

Again, we will have that the first term in the guarantee would be zero, since the corresponding conditions are satisfied on the training data.

Now, to bound the proportion of bad points?? S and?? D , we make the following claim:the network is noise-resilient as per Equation 3 in Theorem C.1 for any input that satisfies the r???1 conditions approximately i.e., ???q ??? r???1 and ???l, ?? q,l (W, x, y) > 0.The above claim can be used to prove Equation 8 as follows.

Since all the conditions are assumed to be satisfied by a margin on the training data, this claim immediately implies that ?? S is zero.

Similarly, this claim implies that for the test data, we can bound ?? D in terms of Pr (x,y)???D [

???q<r ???l ?? q,l (W, x, y) < 0], thus giving rise to the recursion in Equation 8.

Now, to prove our claim, consider an input (x, y) such that ?? q,l (W, x, y) > 0 for q = 1, 2, ???, r ??? 1 and for all possible l. First from the assumption in our theorem statement that ??? q,l (?? ??? ) ??? ??? ??? q,l , we have the following upper bound on the proportion of parameter perturbations under which any of the properties in the first r sets suffer a large perturbation: DISPLAYFORM3 Now, this can be expanded as a summation over q = 1, 2, ???, r as: DISPLAYFORM4 and because (x, y) satisfies ?? q,l (W, x, y) > 0 for q = 1, 2, ???, r ??? 1 and for all possible l, by the constraint assumed in Equation 2, we have: DISPLAYFORM5 Thus, we have that (x, y) satisfies the noise-resilience condition from Equation 3 in Theorem C.1 if it also satisfies ?? q,l (W, x, y) > 0 for q = 1, 2, ???, r ??? 1 and for all possible l.

This proves our claim, and hence in turn proves the recursion in Equation 8.Finally, we can apply a similar argument for the R + 1th set of input-dependent properties (which is a singleton set consisting of the margin of the network) with a small change since the first term in the guarantee from Theorem C.1 is not explicitly assumed to be zero; we will get an inequality in terms of the number of training points that are not classified correctly by a margin, giving rise to the margin-based bound: DISPLAYFORM6 Note that in the first term on the right hand side, ?? R+1,1 (W, x, y) corresponds to the margin of the classifier on (x, y).

Now, by using the fact that the test error is upper bounded by the left hand side in the above equation, applying the recursion on the right hand side R + 1 times, we get our final result.

In this section, we will quantify the noise resilience of a network in different aspects.

Each of our bounds has the following structure: we fix an input point (x, y), and then say that with high probability over a Gaussian perturbation of the network's parameters, a particular input-dependent property of the network (say the output of the network, or the pre-activation value of a particular unit h at a particular layer d, or say the Frobenius norm sof its active weight matrices), changes only by a small magnitude proportional to the variance ?? 2 of the Gaussian perturbation of the parameters.

A key feature of our bounds is that they do not involve the product of the spectral norm of the weight matrices and hence save us an exponential factor in the final generalization bound.

Instead, the bound in the perturbation of a particular property will be in terms of i) the magnitude of the some 'preceding' properties (typically, these are properties of the lower layers) of the network, and ii) how those preceding properties themselves respond to perturbations.

For example, an upper bound in the perturbation of the dth layer's output would involve the 2 norm of the lower layers d ??? < d, and how much they would blow up under these perturbations.

E.2 SOME NOTATIONS.To formulate our lemma statement succinctly, we design a notation wherein we define a set of 'tolerance parameters' which we will use to denote the extent of perturbation suffered by a particular property of the network.

Let?? denote a 'set' (more on what we mean by a set below) of positive tolerance values, consisting of the following elements: ??? We call?? a 'set' to denote a group of related constants into a single symbol.

Each element in this set has a particular semantic associated with it, unlike the standard notation of a set, and so when we refer to, say?? d d ??? ????? , we are indexing into the set to pick a particular element.??? We will use the subscript?? d to index into a subset of only those tolerance values corresponding to layers from 1 until d.

Next we define two events.

The first event formulates the scenario that for a given input, a particular perturbation of the weights until layer d brought about very little change in the properties of these layers (within some tolerance levels).

The second event formulates the scenario that the perturbation did not flip the activation states of the network.

Definition E.1.

Given an input x, and an arbitrary set of constants?? ??? , for any perturbation U of W, we denote by PERT-BOUND(W+U ,?? ??? , x) the event that:??? for each?? DISPLAYFORM0 ??? for each?? ??? d ????? ??? , the maximum perturbation in the preactivation of hidden units on layer d DISPLAYFORM1 ??? for each?? DISPLAYFORM2 ??? for each?? DISPLAYFORM3 NOTE: If we supply only a subset of?? (say?? d instead of the whole of?? ) to the above event, PERT-BOUND(W + U, ???, x), then it would denote the event that the perturbations suffered by only that subset of properties is within the respective tolerance values.

Next, we define the event that the perturbations do not affect the activation states of the network.

Definition E.2.

For any perturbation U of the matrices W, let UNCHANGED-ACTS d (W + U, x) denote the event that none of the activation states of the first d layers change on perturbation.

E.3 MAIN LEMMA.Our results here are styled similar to the equations required by Equation 2 presented in the main paper.

For a given input point and for a particular property of the network, roughly, we bound the the probability that a perturbation affects the value of the property while none of the preceding preceding properties themselves are perturbed beyond a certain tolerance level.

Lemma E.1.

Let?? be a set of constants (that denote the amount of perturbation in the properties preceding a considered property).

For any?? > 0, define?? ??? (which is a bound on the perturbation of a considered property) in terms of?? and the perturbation parameter ??, for all d = 1, 2, ???, D and for all d ??? = d ??? 1, ???, 1 as follows: DISPLAYFORM4 2 ) for any d. Then, the following statements hold good:1.

Bound on perturbation of of 2 norm of the output of layer d. DISPLAYFORM5 3.

Bound on perturbation of 2 norm on the rows of the Jacobians d d ??? .P r U ??PERT-BOUND(W + U, {?? P r U ??PERT-BOUND(W + U, {?? DISPLAYFORM6 DISPLAYFORM7 Proof.

For the most part of this discussion, we will consider a perturbed network where all the hidden units are frozen to be at the same activation state as they were at, before the perturbation.

We will denote the weights of such a network by W [+U] and its output at the dth layer by f d (x; W[+U]).

By having the activations states frozen, the Gaussian perturbations propagate linearly through the activations, effectively remaining as Gaussian perturbations; then, we can enjoy the well-established properties of the Gaussian even after they propagate.

DISPLAYFORM8 Now, the spectral norm of Y d ?????? is at most the products of the spectral norms of each of these three matrices.

Using Lemma B.3, the spectral norm of the middle term U d ?????? can be bounded by ?? 2H ln 2D?? ?? with high probability 1 ????? D over the draws of U d ?????? .

We will also decompose the spectral norm of the first term so that our final bound does not involve any Jacobian of the dth layer.

When d ?????? = d, this term has spectral norm 1 because the Jacobian d d is essentially the identity matrix.

When DISPLAYFORM0 .

Furthermore, since, for a ReLU network, DISPLAYFORM1 effectively W d with some columns zerod out, the spectral norm of the Jacobian is upper bounded by the spectral norm W d .Putting all these together, we have that with probability 1 ????? D over the draws of U d ?????? , the following holds good:

DISPLAYFORM2 By a union bound, we then get that with probability 1 ????? over the draws of U d , we can upper bound Equation 14 as: DISPLAYFORM3

?? Note that the above bound simultaneously holds over all d ??? (without the application of a union bound).

Finally we get the result of the lemma by a similar argument as in the case of the perturbation bound on the output of each layer.

10 Again, note that the below succinct formula works even for corner cases like d Below, we present our main result for this section, a generalization bound on a class of networks that is based on certain norm bounds on the training data.

We provide a more intuitive presentation of these bounds after the proof in Appendix F.3.

Theorem.

4.1 For any ?? > 0, with probability 1 ??? ?? over the draw of samples S ??? D m , for any W, we have that: DISPLAYFORM0 DISPLAYFORM1 GROUPING AND ORDERING THE PROPERTIES.

Now to apply the abstract generalization bound in Theorem 3.1, recall that we need to come up with an ordered grouping of the functions above such that we can realize the constraint given in Equation 2.

Specifically, this constraint effectively required that, for a given input, the perturbation in the properties grouped in a particular set be small, given that all the properties in the preceding sets satisfy the corresponding conditions on them.

To this end, we make use of Lemma E.1 where we have proven perturbation bounds relevant to the properties we have defined above.

Our lemma also naturally induces dependencies between these properties in a way that they can be ordered as required by our framework.

The order in which we traverse the properties is as follows, as dictated by Lemma E.1.

We will go from layer 0 uptil D. For a particular layer d, we will first group the properties corresponding to the spectral norms of the Jacobians of that layer whose corresponding margins are {?? DISPLAYFORM2 Next, we will group the row 2 norms of the Jacobians of layer d, whose corresponding margins are {?? DISPLAYFORM3 Followed by this, we will have a singleton set of the layer output's 2 norm whose corresponding margin is ?? ??? d .

We then will group the pre-activations of layer d, each of which has the corresponding margin ?? ??? d .

For the output layer, instead of the pre-activations or the output 2 norm, we will consider the margin-based property we have defined above.12 13 Observe that the number of sets R that we have created in this manner, is at most 4D since there are at most 4 sets of properties in each layer.

that is required by our framework.

For any r, the rth set of properties need to satisfy the following statement: DISPLAYFORM0 Furthermore, we want the perturbation bounds ??? r,l (??) to satisfy ??? r,l (?? ??? ) ??? ??? ??? r,l , where ?? ??? is the standard deviation of the parameter perturbation chosen in the PAC-Bayesian analysis.

The next step in our proof is to show that our choice of ?? ??? , and the input-dependent properties, all satisfy the above requirements.

To do this, we instantiate Lemma E.1 with ?? = ?? ??? as in Theorem 4.1 DISPLAYFORM1 Then, it can be verified that the values of the perturbation bounds in?? ??? in Lemma E.1 can be upper bounded by the corresponding value in C ??? 2.

In other words, we have that for our chosen value of ??, the perturbations in all the properties and the output of the network can be bounded by the constants specified in C ??? 2.

Succinctly, let us say: DISPLAYFORM2 Given that these perturbation bounds hold for our chosen value of ??, we will focus on showing that a constraint of the form Equation 2 holds for the row 2 norms of the Jacobians d d ??? for all d ???

< d. A similar approach would apply for the other properties.

First, we note that the sets of properties preceding the ones corresponding to the row 2 norms of Jacobian d d ??? , consists of all the properties upto layer d ??? 1.

Therefore, the precondition for Equation 2 which is of the form ??(W, x, y) > 0 for all the previous properties ??, translates to norm bound on these properties involving the constants C ??? d???1 as discussed in Fact F.1.

Succinctly, these norm bounds can be expressed as NORM-BOUND(W + U, DISPLAYFORM3 Given that these norm bounds hold for a particular x, our goal is to argue that the rest of the constraint in Equation 2 holds.

To do this, we first argue that given these norm bounds, if PERT-BOUND(W + U, C ??? d???1 2, x) holds, then so does UNCHANGED-ACTS d???1 (W + U, x).

This is because, the event PERT-BOUND(W + U, C ??? d???1 2, x) implies that the pre-activation values of layer d ??? 1 suffer a perturbation of at most ?? DISPLAYFORM4 holds, we have that the preactivation values of this layer have a magnitude of at least ?? ??? DISPLAYFORM5 From these two equations, we have that the hidden units even at layer d???1 of the network do not change their activation state (i.e., the sign of the pre-activation does not change) under this perturbation.

We can similarly argue for the layers below d ??? 1, thus proving that UNCHANGED- DISPLAYFORM6 Then, from the above discussion on the activation states, and from Equation 15, we have that Lemma E.1 boils down to the following inequality, when we plug ?? = ?? ??? :12 For layer 0, the only property that we have defined is the 2 norm of the input.

13 Note that the Jacobian for d d is nothing but an identity matrix regardless of the input datapoint; thus we do not need any generalization analysis to bound its value on a test datapoint.

Hence, we ignore it in our analysis, as can be seen from the list of properties that we have defined.

Note, that the resulting bound would have a log term that does not affect our bound in an asymptotic sense.

In this section, we provide more detailed demonstration of the dependence of the terms in our bound on the depth/width of the network.

In all the experiments, including the ones in the main paper (except the one in Figure 2 (b)) we use SGD with learning rate 0.1 and mini-batch size 64.

We train the network on a subset of 4096 random training examples from the MNIST dataset to minimize cross entropy loss.

We stop training when we classify at least 0.99 of the data perfectly, with a margin of ?? class = 10.

In Figure 2 (b) where we train networks of depth D = 28, the above training algorithm is quite unstable.

Instead, we use Adam with a learning rate of 10 ???5 until the network achieves an accuracy of 0.95 on the training dataset.

Finally, we note that all logarithmic transformations in our plots are to the base 10.In FIG2 we show how the norm-bounds on the input-dependent properties of the network do not scale as large as the product of spectral norms.

For the remaining experiments in this section, we will present a slightly looser bound than the one presented in our main result, motivated by the fact that computing our actual bound is expensive as it involves computing spectral norms of ??(D 2 ) Jacobians on m training datapoints.

We note that even this looser bound does not have a dependence on the product of spectral norms, and has similar overall dependence on the depth.

Specifically, we will consider a bound that is based on a slightly modified noise-resilience analysis.

Recall that in Lemma E.1, when we considered the perturbation in the row 2 norm Jacobian d d ??? , we bounded Equation 13 in terms of the spectral norms of the Jacobians.

Instead of taking this route, if we retained the bound in Equation 13, we will get a slightly different upper bound on the perturbation of the Jacobian row 2 norm as: DISPLAYFORM0 By using this bound in our analysis, we can ignore the spectral norm terms ?? Thus, the row 2 norms of these Jacobians must be split into separate sets of properties, and the bound on them generalized one after the other (instead of grouped into one set and generalized all at one go as before).

This would

Recall from the discussion in the introduction in the main paper that, prior works (Neyshabur et al., 2017; BID0 have also characterized noise resilience in terms of conditions on the interactions between the activated weight matrices.

Below, we discuss the conditions assumed by these works, which parallel the conditions we have studied in our paper (such as the bounded 2 norm in each layer).

There are two main high level similarities between the conditions studied across these works.

First, these conditions -all of which characterize the interactions between the activated weights matrices in the network -are assumed only for the training inputs; such an assumption implies noise-resilience of the network on training inputs.

Second, there are two kinds of conditions assumed.

The first kind allows one to bound the propagation of noise through the network under the assumption that the activation states do not flip; the second kind allows one to bound the extent to which the activation states do flip.

@highlight

We provide a PAC-Bayes based generalization guarantee for uncompressed, deterministic deep networks by generalizing noise-resilience of the network on the training data to the test data.