The gap between the empirical success of deep learning and the lack of strong theoretical guarantees calls for studying simpler models.

By observing that a ReLU neuron is a product of a linear function with a gate (the latter determines whether the neuron is active or not), where both share a jointly trained weight vector, we propose to decouple the two.

We introduce GaLU networks — networks in which each neuron is a product of a Linear Unit, defined by a weight vector which is being trained, with a Gate, defined by a different weight vector which is not being trained.

Generally speaking, given a base model and a simpler version of it, the two parameters that determine the quality of the simpler version are whether its practical performance is close enough to the base model and whether it is easier to analyze it theoretically.

We show that GaLU networks perform similarly to ReLU networks on standard datasets and we initiate a study of their theoretical properties, demonstrating that they are indeed easier to analyze.

We believe that further research of GaLU networks may be fruitful for the development of a theory of deep learning.

An artificial neuron with the ReLU activation function is the function f w (x) : R d → R such that f w (x) = max{x w, 0} = 1 x w≥0 · x w .The latter formulation demonstrates that the parameter vector w has a dual role; it acts both as a filter or a gate that decides if the neuron is active or not, and as linear weights that control the value of the neuron if it is active.

We introduce an alternative neuron, called Gated Linear Unit or GaLU for short, which decouples between those roles.

A 0 − 1 GaLU neuron is a function g w,u (x) : R d → R such that g w,u (x) = 1 x u≥0 · x w .(1) GaLU neurons, and therefore GaLU networks, are at least as expressive as their ReLU counterparts, since f w = g w,w .

On the other hand, GaLU networks appear problematic from an optimization perspective, because the parameter u cannot be trained using gradient based optimization (since ∇ u g w,u (x) is always zero).

In other words, training GaLU networks with gradient based algorithms is equivalent to initializing the vector u and keeping it constant thereafter.

A more general definition of a GaLU network is given in section 2.The main claim of the paper is that GaLU networks are on one hand as effective as ReLU networks on real world datasets (section 3) while on the other hand they are easier to analyze and understand (section 4).

Many recent works attempt to understand deep learning by considering simpler models, that would allow theoretical analysis while preserving some of the properties of networks of practical utility.

Our model is most closely related to two such proposals: linear networks and non-linear networks in which only the readout layer is being trained.

Deep linear networks is a popular model for analysis that lead to impressive theoretical results (e.g. BID14 ; BID4 ; BID7 ).

Linear networks are useful in order to understand how well gradient-based optimization algorithms work on non-convex problems.

The weakness of linear network is that their expressive power is very limited: linear networks can only express linear functions.

It means that their usefulness to understand the practical success of standard networks is somewhat limited.

Training only the readout layer is an alternative attempt to understand deep learning through simpler models, that also gave theoretical interesting results (e.g. BID13 BID8 ; BID2 ).

The idea is that all the layers but the last one implement a non-linear constant transformation, and the last layer is learning a linear function on top of this transformation.

The weakness of this model is that there is a big practical difference between training all the layers of a network and training only the last one.

Our model is similar in certain aspects to both of those models, but it enjoys a much better practical utility than either one.

See section 3 for an empirical comparison.

Recall the definition of a basic GaLU neuron given in equation 1.

We consider a more general GaLU neuron of the form g w,u,σ (x) = σ(x u) · x w for some non-linear scalar function σ : R → R. If σ is differentiable, we could train the vectors u with gradient based algorithms, but the focus of this paper is on untrained gates.

That is, we assume that the vectors {u} are kept to their initial values throughout the optimization procedure and only the linear part of the GaLU neurons is being optimized.

GaLU networks with a single hidden layer have the following property: for any given example, the values of the gates in the network remain constant.

In networks with more than one hidden layer this not true.

Consider a standard fully connected feed-forward network, let x (0) be the input to the network and let x(1) , x (2) , . . .

be the inputs to intermediate layers of the network.

The output of a GaLU neuron at layer i will be σ(x (i−1) u) · x (i−1) w .

So while the filter parameter vector, u,is not optimized upon, the value of the gate, σ(x (i−1) u), can change as x (i−1) changes.

This adds an additional complication to the dynamics of the optimization that we wish to avoid.

An alternative way to define a GaLU neuron at layer i is σ(x (0) u) · x (i−1) w .

In that case, the value of the gate is determined by the original input, and only the linear part depends on the output of the previous layer of the network.

We call such a neuron a GaLU0 neuron, and a GaLU0 network is a network where all the neurons are GaLU0 neurons.

In GaLU0 networks the gate values remain constant along the training, producing simpler dynamics.

In order to check the hypothesis that effectiveness of ReLU networks stems mostly from the ability to train the linear part of the neurons, and not the gate part, we tested both GaLU0 1 and ReLU networks on the standard MNIST (LeCun & Cortes, 2010) and Fashion-MNIST BID17 datasets.

For both, we used PCA to reduce the input dimension to 64, and then trained a two hidden layers fully-conneted networks on them, with k hidden neurons at each hidden layer.

FIG0 summarizes the results, showing that GaLU0 and ReLU achieve similar results, both outperforming linear networks of the same size.

Training only the readout layer of a ReLU network gave much poorer results (which were omitted from the graphs for clarity).

All models were trained using the same architectures: two fully connected hidden layers with k neurons.

The input dimension was reduced to 64 with PCA.Consider a GaLU network with a single hidden layer of k neurons: DISPLAYFORM0 A convenient property of a GaLU neuron is that it is linear in the weights w j , hence, α j g wj ,uj (x) = g αj wj ,uj (x).

It means that the network can be rewritten as DISPLAYFORM1 withw j = α j w j .

Because we want to optimize over the weights w 1 , . . .

, w k , α 1 , . . . , α k , we might as well optimize over the reparameterizationw 1 , . . .

,w k without losing expressive power.

It means that in a GaLU network of this form, it is sufficient to train the first layer of the network, as the readout layer adds nothing to the expressiveness of the network.

The previous term can be further simplified: DISPLAYFORM2 So it turns out that a GaLU network is nothing more than a random non-linear transformation Φ u : R d → R kd and then a linear function.

There are different notions for the expressivity of a model, and one of the simplest ones is the finitesample expressivity over a random sample.

This notion fits well to our model, because we are not interested in the absolute expressivity of a GaLU network, but of the expressivity of a GaLU network with random filters.

So the question is how well does a randomly-initialized network can fit a random sample.

Note that given the constant filters, solving for the best weights is a convex problem.

Hence, there is no "expressivity -optimization gap" in GaLU networks -every expressivity results is immediately also an optimization result.

DISPLAYFORM0 and y 1 , . . . , y m ∼ N (0, 1), all of which are independent.

Clearly, it is impossible to generalize from the sample to unseen examples; the best possible test loss is 1, and is achieved by the constant prediction 0.

However, it is an interesting problem because it allows us to measure the expressivity of GaLU networks, by showing how much overfit we can expect from the network for a non-adversarial sample.

Equivalently, it tells us how well the network can perform memorization tasks, where the only solution is to memorize the entire sample.

We train the network for the standard mean-squareerror regression loss.

Because the network is simply linear function over a constant non-linear transformation, and because we use the MSE loss, there is a closed form solution to the optimization problem DISPLAYFORM1 withX + being a pseudo-inverse ofX.

This gives us DISPLAYFORM2 be arbitrary vectors.

DefineX as above.

Let y 1 , . . .

, y m ∼ N (0, 1) be independent random normal variables.

Define the expected squared loss on the training set, for weights w, as L S (w).

Then, DISPLAYFORM3 Proof Every vector y = (y 1 , . . . , y m ) ∈ R m can be decomposed to a sum y = a + b where a is in the span of the columns ofX and b is in the null space ofX. It follows that min w L S (w) = b 2 /m.

The claim follows because if y ∼ N (0, I m ) then the expected value of b 2 is m − rank X .It is always true that rank(X) ≤ min{m, kd}. Empirical experimentation shows that if x 1 , . . .

, x m , u 1 , . . . , u k ∼ N (0, I d ) then with high probability rank(X) = min{m, kd}.

The fact that the GaLU network turned out to be only a linear function on top of a non-linear transformation seems to be a peculiar mathematical accident, with little relevance to standard networks.

So we empirically tested the behavior of both ReLU and GaLU networks on the above model.

It turns out that ReLU outperforms GaLU by a small margin -it is never better than GaLU with double the number of neurons, and is often worse than that.

ReLU can outperform GaLU, even though it is less expressive, because we don't train the value of the the filters u 1 , . . . , u k at all for the GaLU networks.

It turns out that SGD over a ReLU network converges to better filters than a simple random initialization.

One way to measure how much better those filters are is by trying to improve the initial filters of the GaLU network by randomly replacing them.

Consider for example the simple algorithm given in algorithm 1.Running this algorithm improves the results of the GaLU networks, making them more competitive with the ReLU ones.

FIG1 summarizes our results.

An important fact about artificial neural networks is that they have small generalization error in many real-life problems.

Otherwise they wouldn't be very useful as a learning algorithm.

ZhangAlgorithm 1 Improve GaLU filters Input: A sample S, number of neurons k, number of iterations n. Initialize u 1 , u 2 , . . . , u k randomly.

Find an optimal solution w 1 , . . . , w k .

for i = 1 to n do Pick j ∼ Uniform {1, 2, ..., k}. Pickũ j randomly.

Find an optimal solution for a GaLU network with filters u 1 , . . . , u j−1 ,ũ j , u j+1 , . . . , u k .

If the new solution is better than the current one, update u j =ũ j . end for One of the main experiments they run was to train the network over a sample with randomized labels, and to observe that the network still achieved small training loss (but large test loss, naturally).

So any generalization bound that can be applied to the randomized sample is necessarily too weak to explain the generalization of the natural sample.

As our goal is to show that GaLU networks exhibit similar phenomena as ReLU networks, but may be easier to analyze, we first construct a similar experiment to that of BID18 and compare the performance of GaLU and ReLU networks.

Consider the following natural model.

Let c 1 , . . .

, c n ∼ N (0, I d ) be n clusters centers, each one with a random labels b 1 , . . .

, b n .

A data point (x, y) is generated by picking a random index i ∼ Uniform {1, 2, . . .

, n}, and setting x = c i + ξ for ξ ∼ N (0, σ We fixed the number of samples m = 1000, the input dimension d = 30, the number of clusters n = 30, the number of hidden neurons k = 30 and σ x = 0.1.

We calculated the train and test errors for different values of σ y and p and for a GaLU and ReLU networks.

The results are summarized in FIG3 .

We can clearly see that GaLU and ReLU have similar statistical behavior, and that while the train error is always small, as the labels become noisier the generalization error increases.

This matches the spirit of experiments reported in BID18 .

Next, we turn to an analysis of this phenomenon.

Since one hidden layer GaLU networks can be cast as linear predictors, we can rely on classic norm-based generalization bounds for linear predictors.

In particular, for p ∈ {1, 2}, consider the class of linear predictors H p = {x → x w : w p ≤ B p }.

BID15 .

This also induces an upper bound on the gap between the test and train loss (see again BID15 for Lipschitz loss functions and see BID16 for the relation between Rademacher complexity and the generalization of smooth losses such as the squared loss).

The question is whether the 1 / 2 norm of w is correlated with the amount of noise in the data.

To study this, we depict the gap between train and test error as a function of the norm of w for GaLU networks.

As can be seen in FIG5 , for both the 1 and 2 norm, there is a clear linear relation between w 2 p and the generalization gap.

While the constants are far from what the bounds state, the linear correlation is very clear.

Note that figure 4 deals with GaLU networks that were trained as linear functions (by using the closed form solution for the MSE loss), and indeed shows that such network with such training behave as the theory states for linear predictors.

We do not get the same behavior when we (unnecessarily) train both layers of the network using SGD.

This matches the discussion in Section 5 of BID18 , where the correlation between the 2 norm of the weights in a ReLU network and the test loss is discussed, and it is argued that there are more factors that affect the generalization properties.

Indeed, many followup works show different capacity measures that may be more adequate for studying the generalization of deep learning (See for example Bartlett et al. (2017) ).

We next show a rather different analysis for a particular instance of linear regression.

Consider a simple linear regression using the MSE, and denote the train and test loss by DISPLAYFORM0 Given a training set S, the MSE estimator is defined as w(S) := arg min w L S (w).We start with the following lemma.

Lemma 1 (Follows from Corollary 2 of BID12 ) For a scalar σ ≥ 0 and a vector β ∈ R d , let D σ,β be the distribution over R d × R which is defined by the following generative DISPLAYFORM1 This lemma provides a complete analysis for the following experiment, which is similar to the experiments reported by BID18 .

We compare two distributions, the first is D σ,β for some vector β ∈ R d and for σ being close to 0, and the second is D 1,0 .

Note that the first distribution corresponds to a case in which we would like to be able to generalize, while the second distribution corresponds to a case in which we are fitting random noise and do not expect to generalize.

We set the training set size to be m = d + 2 and we analyze the MSE estimator, w(S).

As the lemma shows, the expected training losses on the first and second distributions are DISPLAYFORM2 respectively.

Hence, the training loss should be small on both of the distributions.

In contrast, the expected test loss on the first distribution is DISPLAYFORM3 while the expected test loss on the second distribution is DISPLAYFORM4 We see that while the train loss can be small on both distributions, in the test loss we see a big gap between the first distribution (assuming σ 1/ √ d) and the second distribution of purely random labels.

This is exactly the type of phenomenon reported in BID18 -a sample with a small amount of noise achieves both small train and test losses, but a sample with random labels achieves a small train loss but a large test loss.

Note that this is a natural property of the least squares solution, without any explicit regularization, picking a minimal-norm solution or using a specific algorithm for solving the problem.

Lemma 1 gives us a very sharp analysis of linear regression.

Unfortunately, the assumptions of Lemma 1 (which are based on the assumptions of Corollary 2 in BID12 ) are too strong -we need that m > d + 1 and that the instances will be generated based on a Gaussian distribution.

While BID12 also includes asymptotic results that are applicable for a larger set of distributions, we leave the application of them to GaLU networks for future work.

In the analysis of the R d → R case we used the fact that a GaLU neuron g w,u is linear in the parameter w, and it allowed us to rephrase the problem as a convex problem.

In the R d → R d case the situation is not as simple.

In this case, every hidden neuron has d outgoing edges, and so we cannot use the same reparametrization trick as before.

Even so, the output of a GaLU neuron is still linear in the parameter w.

It means that for convex loss functions, finding the optimal weights for the first layer, keeping the weights of the second one constant, is a convex problem.

The same doesn't hold for ReLU networks.

Finding the optimal weights for the second layer, keeping the weights of the first one constant, is also a convex problem.

Even more specifically, the optimization problem over the two layers is biconvex (see BID3 for a survey).

So instead of applying SGD, we can apply biconvex optimization algorithms, such as Alternate Convex Search (ACS).

In the case of the MSE loss, there is a closed form solution for each step of ACS, and using it outperforms SGD for small enough samples 2 .

Even though it is of limited practical use, this algorithm might be interesting for the derivation of theoretical bounds for such networks.

In addition, it turns out that as we increase the output dimension d , GaLU and ReLU networks becomes more similar.

In section 4.1.1 we measured the difference between ReLU and GaLU for the problem where all the variables are i.i.d.

N (0, 1), and it turned out that ReLU outperforms GaLU to a small extent.

We repeated this experiment with larger d , and saw that the difference between the two vanished quickly (see FIG6 ).

ber of neurons k such that a one hidden layer network achieves MSE< 0.3 on the random regression problem.

As the output dimension d grows, more neurons are needed.

As demonstrated in figure 2, GaLU networks needs more neurons than ReLU networks for output dimension d = 1.

For larger d GaLU is slightly better, but it is clear that the two networks exhibit very similar behavior.

We used fixed sample size (m = 1024) and input dimension (d = 32) in the generation of this graph.

The standard paradigm in deep learning is to use neurons of the form σ x w for some differentiable non linear function σ : R → R. In this article we proposed a different kind of neurons, σ i,j · x w, where σ i,j is some function of the example and the neuron index that remains constant along the training.

Those networks achieve similar results to those of their standard counterparts, and they are easier to analyze and understand.

To the extent that our arguments are convincing, it gives new directions for further research.

Better understanding of the one hidden layer case (from section 5) seems feasible.

And as GaLU and ReLU networks behave identically for this problem, it gives us reasons to hope that understanding the behavior of GaLU networks would also explain ReLU networks and maybe other non-linearities as well.

As for deeper network, it is also not beyond hope that GaLU0 networks would allow some better theoretical analysis than what we have so far.

@highlight

We propose Gated Linear Unit networks — a model that performs similarly to ReLU networks on real data while being much easier to analyze theoretically.