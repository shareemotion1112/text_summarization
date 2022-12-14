Empirical evidence suggests that neural networks with ReLU activations generalize better with over-parameterization.

However, there is currently no theoretical analysis that explains this observation.

In this work, we study a simplified learning task with over-parameterized convolutional networks that empirically exhibits the same qualitative phenomenon.

For this setting, we provide a theoretical analysis of the optimization and generalization performance of gradient descent.

Specifically, we prove data-dependent sample complexity bounds which show that over-parameterization improves the generalization performance of gradient descent.

Most successful deep learning models use a number of parameters that is larger than the number of parameters that are needed to get zero-training error.

This is typically referred to as overparameterization.

Indeed, it can be argued that over-parameterization is one of the key techniques that has led to the remarkable success of neural networks.

However, there is still no theoretical account for its effectiveness.

One very intriguing observation in this context is that over-parameterized networks with ReLU activations, which are trained with gradient based methods, often exhibit better generalization error than smaller networks BID11 Novak et al., 2018) .

This somewhat counterintuitive observation suggests that first-order methods which are trained on over-parameterized networks have an inductive bias towards solutions with better generalization performance.

Understanding this inductive bias is a necessary step towards a full understanding of neural networks in practice.

Providing theoretical guarantees for this phenomenon is extremely challenging due to two main reasons.

First, to show a generalization gap, one needs to prove that large networks have better sample complexity than smaller ones.

However, current generalization bounds that are based on complexity measures do not offer such guarantees.

Second, analyzing the dynamics of first-order methods on networks with ReLU activations is a major challenge.

Indeed, there do not exist optimization guarantees even for simple learning tasks such as the classic XOR problem in two dimensions.

1 To advance this issue, we focus on a particular learning setting that captures key properties of the over-parameterization phenomenon.

We consider a high-dimensional extension of the XOR problem, which we refer to as the "XOR Detection problem (XORD)".

The XORD is a pattern recognition task where the goal is to learn a function which classifies binary vectors according to whether they contain a two-dimensional binary XOR pattern (i.e., (1, 1) or (???1, ???1)).

This problem contains the classic XOR problem as a special case when the vectors are two dimensional.

We consider learning this function with gradient descent trained on an over-parameterized convolutional neural network (i.e., with multiple channels) with ReLU activations and three layers: convolutional, max pooling and fully connected.

As can be seen in FIG0 , over-parameterization improves generalization in this problem as well.

Therefore it serves as a good test-bed for understanding the role of over-parameterization.

1 We are referring to the problem of learning the XOR function given four two-dimensional points with binary entries, using a moderate size one-hidden layer neural network (e.g., with 50 hidden neurons).

Note that there are no optimization guarantees for this setting.

Variants of XOR have been studied in BID10 ; Sprinkhuizen-Kuyper & Boers (1998) but these works only analyzed the optimization landscape and did not provide guarantees for optimization methods.

We provide guarantees for this problem in Sec. 9.

3).

The figure shows the test error obtained for different number of channels k. The blue curve shows test error when restricting to cases where training error was zero.

It can be seen that increasing the number of channels improves the generalization performance.

Experimental details are provided in Section 8.2.1..

In this work we provide an analysis of optimization and generalization of gradient descent for XORD.

We show that for various input distributions, ranges of accuracy and confidence parameters, sufficiently over-parameterized networks have better sample complexity than a small network which can realize the ground truth classifier.

To the best of our knowledge, this is the first example which shows that over-paramaterization can provably improve generalization for a neural network with ReLU activations.

Our analysis provides a clear distinction between the inductive bias of gradient descent for overparameterized and small networks.

It reveals that over-parameterized networks are biased towards global minima that detect more patterns in the data than global minima found by small networks.

2 Thus, even though both networks succeed in optimization, the larger one has better generalization performance.

We provide experiments which show that the same phenomenon occurs in a more general setting with more patterns in the data and non-binary input.

We further show that our analysis can predict the behavior of over-parameterized networks trained on MNIST and guide a compression scheme for over-parameterized networks with a mild loss in accuracy (Sec. 6).

In recent years there have been many works on theoretical aspects of deep learning.

We will refer to those that are most relevant to this work.

First, we note that we are not aware of any work that shows that generalization performance provably improves with over-parameterization.

This distinguishes our work from all previous works.

Several works study convolutional networks with ReLU activations and their properties BID4 b; BID0 .

All of these works consider convolutional networks with a single channel.

BID2 and BID7 provide guarantees for SGD in general settings.

However, their analysis holds for over-parameterized networks with an extremely large number of neurons that are not used in practice (e.g., the number of neurons is a very large polynomial of certain problem parameters).

Furthermore, we consider a 3-layer convolutional network with max-pooling which is not studied in these works.

Soltanolkotabi et al. (2018) , BID3 and study the role of overparameterization in the case of quadratic activation functions.

BID1 provide generalization guarantees for over-parameterized networks with Leaky ReLU activations on linearly separable data.

Neyshabur et al. (2018) prove generalization bounds for neural networks.

However, these bounds are empirically vacuous for over-parameterized networks and they do not prove that networks found by optimization algorithms give low generalization bounds.

We begin with some notations and definitions.

Let d ??? 4 be an integer.

We consider a classification problem in the space {??1} 2d .

Namely, the space of vectors of 2d coordinates where each coordinate can be +1 or ???1.

Given a vector x ??? {??1} 2d , we consider its partition into d sets of two coordinates as follows x = (x 1 , ..., x d ) where x i ??? {??1}2 .

We refer to each such x i as a pattern in x.

Neural Architecture: We consider learning with the following three-layer neural net model.

The first layer is a convolutional layer with non-overlapping filters and multiple channels, the second layer is max pooling and the third layer is a fully connected layer with 2k hidden neurons and weights fixed to values ??1.

Formally, for an input x = (x 1 , ..., x d ) ??? R 2d where x i ??? R 2 , the output of the network is given by: DISPLAYFORM0 where W ??? R 2k??2 is the weight matrix whose rows are the w (i) vectors followed by the u DISPLAYFORM1 vectors, and ??(x) = max{0, x} is the ReLU activation applied element-wise.

See FIG6 for an illustration of this architecture.

Remark 3.1.

Because there are only 4 different patterns, the network is limited in terms of the number of different rules it can implement.

Specifically, it is easy to show that its VC dimension is at most 15 (see Sec. 10).

Despite this limited expressive power, there is a generalization gap between small and large networks in this setting, as can be seen in FIG0 , and in our analysis below.

Data Generating Distribution:

Next we define the classification rule we will focus on.

Let P XOR correspond to the following two patterns: P XOR = {(1, 1), (???1, ???1)}. Define the classification rule: DISPLAYFORM2 Namely, f * detects whether a pattern in P XOR appears in the input.

In what follows, we refer to P XOR as the set of positive patterns and {??1} 2 \ P XOR as the set of negative patterns.

Let D be a distribution over X ?? {??1} such that for all (x, y) ??? D we have y = f * (x).

We say that a point (x, y) is positive if y = 1 and negative otherwise.

Let D + be the marginal distribution over {??1} 2d of positive points and D ??? be the marginal distribution of negative points.

In the following definition we introduce the notion of diverse points, which will play a key role in our analysis.

Definition 3.2 (Diverse Points).

We say that a positive point (x, 1) is diverse if for all z ??? {??1} 2 there exists 1 ??? i ??? d such that x i = z. We say that a negative point DISPLAYFORM3 For ?? ??? {???, +} define p ?? to be the probability that x is diverse with respect to D ?? .

For example, if both D + and D ??? are uniform, then by the inclusion-exclusion principle it follows that DISPLAYFORM4 For each set of binary patterns A ??? {??1} 2 define p A to be the probability to sample a point which contains all patterns in A and no patterns in A c (the complement of A).

Let A 1 = {2}, A 2 = {4}, A 3 = {2, 4, 1} and A 4 = {2, 4, 3}.The following quantity will be useful in our analysis: DISPLAYFORM5 Learning Setup: Our analysis will focus on the problem of learning f * from training data with a three layer neural net model.

The learning algorithm will be gradient descent, randomly initialized.

As in any learning task in practice, f * is unknown to the training algorithm.

Our goal is to analyze the performance of gradient descent when given data that is labeled with f * .

We assume that we are given a training set S = S + ??? S ??? ??? {??1} 2d ?? {??1} 2 where S + consists of m IID points drawn from D + and S ??? consists of m IID points drawn from D ??? .

Importantly, we note that the function f * can be realized by the above network with k = 2.

Indeed, the network N defined by the filters w(1) = (3, 3), DISPLAYFORM6 2d .

It can be seen that for k = 1, f * cannot be realized.

Therefore, any k > 2 is an over-parameterized setting.

Training Algorithm: We will use gradient descent to optimize the following hinge-loss function.

DISPLAYFORM7 for ?? ??? 1.

4 We assume that gradient descent runs with a constant learning rate ?? and the weights are randomly initiliazed with IID Gaussian weights with mean 0 and standard deviation ?? g .

Furthermore, only the weights of the first layer, the convolutional filters, are trained.

We will need the following notation.

Let W t be the weight matrix in iteration t of gradient descent.

DISPLAYFORM0 2 the ith convolutional filter at iteration t. Similarly, for DISPLAYFORM1 t ??? R 2 to be the k + i convolutional filter at iteration t. We assume that each DISPLAYFORM2 0 is initialized as a Gaussian random variable where the entries are IID and distributed as N (0, ?? 2 g ).

In each iteration, gradient descent performs the update DISPLAYFORM3

In this section we state our main result that demonstrates the generalization gap between overparameterized networks and networks with k = 2.

Define the generalization error to be the difference between the 0-1 test error and the 0-1 training error.

For any , ?? and training algorithm let m( , ??) be the sample complexity of a training algorithm, namely, the number of minimal samples the algorithm needs to get at most generalization error with probability at least 1 ??? ??.

We consider running gradient descent in two cases, when k ??? 120 and k = 2.

In the next section we exactly define under which set of parameters gradient descent runs, e.g., which constant learning rates.

Fix parameters p + and p ??? of a distribution D and denote by c < 10 ???10 a negligible constant.

Assume that gradient descent is given a sample of points drawn from D + and D ??? .

We denote the sample complexity of gradient descent in the cases k ??? 120 and k = 2, by m 1 and m 2 , respectively.

The following result shows a data dependent generalization gap (recall the definition of p * in Eq. 3).Theorem 4.1.

Let D be a distribution with paramaters p + , p ??? and p DISPLAYFORM0 The proof follows from Theorem 5.2 and Theorem 5.3 which we state in the next section.

The proof is given in Sec. 8.8.

One surprising fact of this theorem is that m 1 (0, ??) ??? 2.

Indeed, our analysis shows that for an over-parameterized network and for sufficiently large p + and p ??? , one diverse positive point and one diverse negative suffice for gradient descent to learn f * with high probability.

We note that even in this case, the dynamics of gradient descent is highly complex.

This is due to the randomness of the initialization and to the fact that there are multiple weight filters in the network, each with different dynamics.

See Sec. 5 for further details.

We will illustrate the guarantee of Theorem 4.1 with several numerical examples.

In all of the examples we assume that for the distribution D, the probability to sample a positive point is 1 2 and 4 In practice it is common to set ?? to 1.

In our analyis we will need ?? ??? 8 to guarantee generalization.

In Section 8.3 we show empirically, that for this task, setting ?? to be larger than 1 results in better test performance than setting ?? = 1.5 Note that BID6 show that fixing the last layer weights to ??1 does not degrade performance in various tasks.

This assumption also appeared in other works BID1 BID8 .

6 We note that this generalization gap holds for global minima (0 train error).

Therefore, the theorem can be read as follows.

For k ??? 120, given 2 samples, with probability at least 1 ??? ??, gradient descent converges to a global minimum with at most test error.

On the other hand, for k = 2 and given number of samples less than 2 log DISPLAYFORM1 , with probability greater than ??, gradient descent converges to a global minimum with error greater than .

See Section 5 for further details.

DISPLAYFORM2 (it is easy to constuct such distributions).

In the first example, we assume that p + = p ??? = 0.98 and ?? = 1 ??? 0.98 DISPLAYFORM3 05.

In this case we get that for any 0 ??? < 0.005, m 1 ( , ??) ??? 2 whereas m 2 ( , ??) ??? 129.

For the second example consider the case where p + = p ??? = 0.92.

It follows that for ?? = 0.16 and any 0 ??? < 0.02 it holds that m 1 ( , ??) ??? 2 and m 2 ( , ??) ??? 17.

For = 0 and any ?? > 0, by setting p + and p ??? to be sufficiently close to 1, we can get an arbitrarily large gap between m 1 ( , ??) and m 2 ( , ??).

In contrast, for sufficiently small p + and p ??? , e.g., in which p + , p ??? ??? 0.7, our bound does not guarantee a generalization gap.

In this section we sketch the proof of Theorem 4.1.

The theorem follows from two theorems: Theorem 5.2 for over-parameterized networks and Theorem 5.3 for networks with k = 2.

We formally show this in Sec. 8.8.

In Sec. 5.1 we state Theorem 5.2 and outline its proof.

In Sec. 5.2 we state Theorem 5.3 and shortly outline its proof.

Finally, for completeness, in Sec. 9 we also provide a convergence guarantee for the XOR problem with inputs in {??1}, which in our setting is the case of d = 1.

In what follows, we will need the following formal definition for a detection of a pattern by a network.

DISPLAYFORM0 We say that a pattern v (positive or negative) is detected by the network N W with confidence DISPLAYFORM1 The above definition captures a desired property of a network, namely, that its filters which are connected with a positive coefficient in the last layer, have high correlation with the positive patterns and analogously for the remaining filters and negative patterns.

We note however, that the condition in which a network detects all patterns is not equivalent to realizing the ground truth f * .

The former can hold without the latter and vice versa.

Theorem 5.2 and Theorem 5.3 together imply a clear characterization of the different inductive biases of gradient descent in the case of small (k = 2) and over-parameterized networks.

The characterization is that over-parameterized networks are biased towards global minima that detect all patterns in the data, whereas small networks with k = 2 are biased towards global minima that do not detect all patterns (see Definition 5.1).

In Sec. 8.5 we show this empirically in the XORD problem and in a generalization of the XORD problem.

In the following sections we will need several notations.

Define x 1 = (1, 1), x 2 = (1, ???1), x 3 = (???1, ???1), x 4 = (???1, 1) to be the four possible patterns in the data and the following sets: DISPLAYFORM2 We denote by x + a positive diverse point and x ??? a negative diverse point.

Define the following sum: DISPLAYFORM3 Finally, in all of the results in this section we will denote by c < 10 ???10 a negligible constant.

The main result in this section is given by the following theorem.

, k ??? 120 and ?? ??? 8.

Then, with probability DISPLAYFORM0 iterations, it converges to a global minimum which satisfies: DISPLAYFORM1 , all patterns are detected with confidence c d .This result shows that given a small training set size, and sufficiently large p + and p ??? , overparameterized networks converge to a global minimum which realizes the classifier f * with high probability and in a constant number of iterations.

Furthermore, this global minimum detects all patterns in the data with confidence that increases with over-parameterization.

The full proof of Theorem 5.2 is given in Sec. 8.6.We will now sketch its proof.

With probability at least (p + p ??? ) m all training points are diverse and we will condition on this event.

From Sec. 10 we can assume WLOG that the training set consists of one positive diverse point x + and one negative diverse point x ??? (since the network will have the same output on all same-label diverse points).

We note that empirically over-parameterization improves generalization even when the training set contains non-diverse points (see FIG0 and Sec. 8.2).

Now, to understand the dynamics of gradient descent it is crucial to understand the dynamics of the sets in Eq. 5.

This follows since the gradient updates are expressed via these sets.

Concretely, let DISPLAYFORM2 then the gradient update is given as follows: DISPLAYFORM3 the gradient update is given by: DISPLAYFORM4 Furthermore, the values of N W (x + ) and N W (x ??? ) depend on these sets and their corresponding weight vectors, via sums of the form S + t , defined above.

The proof consists of a careful analysis of the dynamics of the sets in Eq. 5 and their corresponding weight vectors.

For example, one result of this analysis is that for all t ??? 1 and i ??? {1, 3} we have W There are two key technical observations that we apply in this analysis.

First, with a small initialization and with high probability, for all 1 ??? j ??? k and 1 ??? i ??? 4 it holds that w DISPLAYFORM5 .

This allows us to keep track of the dynamics of the sets in Eq. 5 more easily.

For example, by this observation it follows that if for some j * ??? W + t (2) it holds that j * ??? W + t+1 (4), then for all j such that j ??? W + t (2) it holds that j ??? W + t+1 (4).

Hence, we can reason about the dynamics of several filters all at once, instead of each one separately.

Second, by concentration of measure we can estimate the sizes of the sets in Eq. 5 at iteration t = 0.

Combining this with results of the kind W + t (i) = W + 0 (i) for all t, we can understand the dynamics of these sets throughout the optimization process.

The theorem consists of optimization and generalization guarantees.

For the optimization guarantee we show that gradient descent converges to a global minimum.

To show this, the idea is to characterize the dynamics of S + t using the characterization of the sets in Eq. 5 and their corresponding weight vectors.

We show that as long as gradient descent did not converge to a global minimum, S + t cannot decrease in any iteration and it is upper bounded by a constant.

Furthermore, we show that there cannot be too many consecutive iterations in which S + t does not increase.

Therefore, after sufficiently many iterations gradient descent will converge to a global minimum.

We will now outline the proof of the generalization guarantee.

Denote the network learned by gradient descent by N W T .

First, we show that the network classifies all positive points correctly.

Define the following sums for all 1 ??? i ??? 4: DISPLAYFORM6 First we notice that for all positive z we have N W T (z) min{X .

Hence, we can show that each positive point is classified correctly.

The proof that all negative points are classified correctly and patterns x 2 and x 4 are detected is similar but slightly more technical.

We refer the reader to Sec. 8.6 for further details.

DISPLAYFORM7

The following theorem provides generalization lower bounds of global minima in the case that k = 2 and in a slightly more general setting than the one given in Theorem 5.2.

Theorem 5.3.

Let S = S + ??? S ??? be a training set as in Sec. 3.

Assume that gradient descent runs with parameters ?? = c?? k where c ?? ??? DISPLAYFORM0 , k = 2 and ?? ??? 1.

Then the following holds: DISPLAYFORM1 48 , gradient descent converges to a global minimum that has non-zero test error.

Furthermore, for c d ??? 2c ?? , there exists at least one pattern which is not detected by the global minimum with confidence c d .2.

The non-zero test error above is at least p * .The theorem shows that for a training set that is not too large and given sufficiently large p + and p ??? , with constant probability, gradient descent will converge to a global minimum that is not the classifier f * .

Furthermore, this global minimum does not detect at least one pattern.

The proof of the theorem is given in Sec. 8.7.We will now provide a short outline of the proof.

Let w DISPLAYFORM2 T be the filters of the network at the iteration T in which gradient descent converges to a global minimum.

The proof shows that gradient descent will not learn f * if one of the following conditions is met: a) DISPLAYFORM3 T ?? x 4 > 0.

Then by using a symmetry argument which is based on the symmetry of the initialization and the training data it can be shown that one of the above conditions is met with high constant probability.

Finally, it can be shown that if one of these conditions hold, then at least one pattern is not detected.

We perform several experiments that corroborate our theoretical findings.

In Sec. 8.5 we empirically demonstrate our insights on the inductive bias of gradient descent.

In Sec. 6.2 we evaluate a model compression scheme implied by our results, and demonstrate its success on the MNIST dataset.

In this section we perform experiments to examine the insights from our analysis on the inductive bias of gradient descent.

Namely, that over-parameterized networks are biased towards global minima that detect more patterns in the data than global minima found by smaller networks.

We check this both on the XORD problem which contains 4 possible patterns in the data and on an instance of an extension of the XORD problem, that we refer to as the Orthonormal Basis Detection (OBD) problem, which contains 60 patterns in the data.

In Sec. 8.5 we provide details on the experimental setups.

standard training (red), the small network that uses clusters from the large network (blue), and the large network (120 channels) with standard training (green).

It can be seen that the large network is effectively compressed without losing much accuracy.

Due to space considerations, we will not formally define the OBD problem in this section.

We refer the reader to Sec. 8.5 for a formal definition.

Informally, The OBD problem is a natural extension of the XORD problem that contains more possible patterns in the data and allows the dimension of the filters of the convolutional network to be larger.

The patterns correspond to a set of orthonormal vectors and their negations.

The ground truth classifier in this problem can be realized by a convolutional network with 4 channels.

In FIG1 we show experiments which confirm that in the OBD problem as well, overparameterization improves generalization.

We further show the number of patterns detected in %0 training error solutions for different number of channels, in both the XORD and OBD problems.

It can be clearly seen that for both problems, over-parameterized networks are biased towards %0 training error solutions that detect more patterns, as predicted by the theoretical results.

By inspecting the proof of Theorem 5.2, one can see that the dynamics of the filters of an overparameterized network are such that they either have low norm, or they have large norm and they point to the direction of one of the patterns (see, e.g., Lemma 8.4 and Lemma 8.6).

This suggests that by clustering the filters of a trained over-parameterized network to a small number of clusters, one can create a significantly smaller network which contains all of the detectors that are needed for good generalization performance.

Then, by training the last layer of the network, it can converge to a good solution.

Following this insight, we tested this procedure on the MNIST data set and a 3 layer convolutional network with convolutional layer with multiple channels and 3 ?? 3 kernels, max pooling layer and fully connected layer.

We trained an over-parameterized network with 120 channels, clustered its filters with k-means into 4 clusters and used the cluster centers as initialization for a small network with 4 channels.

Then we trained only the fully connected layer of the small network.

In FIG5 we show that for various training set sizes, the performance of the small network improves significantly with the new initialization and nearly matches the performance of the overparameterized network.

In this paper we consider a simplified learning task on binary vectors and show that overparameterization can provably improve generalization performance of a 3-layer convolutional network trained with gradient descent.

Our analysis reveals that in the XORD problem overparameterized networks are biased towards global minima which detect more relevant patterns in the data.

While we prove this only for the XORD problem and under the assumption that the training set contains diverse points, our experiments clearly show that a similar phenomenon occurs in other settings as well.

We show that this is the case for XORD with non-diverse points FIG0 ) and in the more general OBD problem which contains 60 patterns in the data and is not restricted to binary inputs FIG1 .

Furthermore, our experiments on MNIST hint that this is the case in MNIST as well FIG5 .By clustering the detected patterns of the large network we could achieve better accuracy with a small network.

This suggests that the larger network detects more patterns with gradient descent even though its effective size is close to that of a small network.

We believe that these insights and our detailed analysis can guide future work for showing similar results in more complex tasks and provide better understanding of this phenomenon.

It would also be interesting to further study the implications of such results on model compression and on improving training algorithms.

Behnam Neyshabur, Zhiyuan Li, Srinadh Bhojanapalli, Yann LeCun, and Nathan Srebro.

We tested the generalization performance in the setup of Section3.

We considered networks with number of channels 4,6,8,20,50,100 and 200 .

The distribution in this setting has p + = 0.5 and p ??? = 0.9 and the training sets are of size 12 (6 positive, 6 negative).

Note that in this case the training set contains non-diverse points with high probability.

The ground truth network can be realized by a network with 4 channels.

For each number of channels we trained a convolutional network 100 times and averaged the results.

In each run we sampled a new training set and new initialization of the weights according to a gaussian distribution with mean 0 and standard deviation 0.00001.

For each number of channels c, we ran gradient descent with learning rate 0.04 c and stopped it if it did not improve the cost for 20 consecutive iterations or if it reached 30000 iterations.

The last iteration was taken for the calculations.

We plot both average test error over all 100 runs and average test error only over the runs that ended at 0% train error.

In this case, for each number of channels 4, 6, 8, 20, 50, 100 ,200 the number of runs in which gradient descent converged to a 0% train error solution is 62, 79, 94, 100, 100, 100, 100, respectively.

Figure 5 shows that setting ?? = 5 gives better performance than setting ?? = 1 in the XORD problem.

The setting is similar to the setting of Section 8.2.1.

Each point is an average test error of 100 runs. .

Because the result is a lower bound, it is desirable to understand the behaviour of gradient descent for values outside these ranges.

In Figure 6 we empirically show that for values outside these ranges, there is a generalization gap between gradient descent for k = 2 and gradient descent for larger k.

We will first formally define the OBD problem.

Fix an even dimension parameter d 1 ??? 2.

In this problem, we assume there is an orthonormal basis B = {v 1 , ..., v d1 } of R d1 .

Divide B into two equally sized sets B 1 and B 2 , each of size d1 2 .

Now define the set of positive patterns to be P = {v | v ??? B 1 } ??? {???v | v ??? B 1 } and negative patterns to be DISPLAYFORM0 we assume the input domain is X ??? R d1d2 and each x ??? X is a vector such that x = (x 1 , ..., x d2 ) where each x i ??? P OBD .

We define the ground truth classifier f OBD : X ??? {??1} such that f OBD (x) = 1 if and only there exists at least one x i such that x i ??? P .

Notice that for d 1 = 2 and by normalizing the four vectors in {??1} 2 to have unit norm, we get the XORD problem.

We note that the positive patterns in the XORD problem are defined to be P XOR and the negative patterns are {??1} 2 \ P XOR .Let D be a distribution over X 2d ??

{??1} such that for all (x, y) ??? D, y = f OBD (x).

As in the XORD problem we define the distributions D + and D ??? .

We consider the following learning task which is the same as the task for the XORD problem.

We assume that we are given a training set S = S + ??? S ??? ??? {??1} d1d2 ?? {??1} where S + consists of m IID points drawn from D + and S ??? consists of m IID points drawn from D ??? .

The goal is to train a neural network with randomly initialized gradient descent on S and obtain a network N : DISPLAYFORM1 We consider the same network as in the XORD problem (Eq. 1), but now the filters of the convolution layer are d 1 -dimensional.

Formally, for an input x = (x 1 , ..., x d ) ??? X the output of the network is given by DISPLAYFORM2 where W ??? R 2k??d1 is the weight matrix which contains in the first k rows the vectors w (i) ??? R d1 , in the next k rows the vectors u (i) ??? R d1 and ??(x) = max{0, x} is the ReLU activation applied element-wise.

We performed experiments in the case that d 1 = 30, i.e., in which there are 60 possible patterns.

In FIG1 , for each number of channels we trained a convolutional network given in Eq. 9 with gradient descent for 100 runs and averaged the results.

The we sampled 25 positive points and 25 negative points in the following manner.

For each positive point we sampled with probability 0.25 one of the numbers [4, 6, 8, 10 ] twice with replacement.

Denote these numbers by m 1 and m 2 .

Then we sampled m 1 different positive patterns and m 2 different negative patterns.

Then we filled a 60d 1 -dimensional vectors with all of these patterns.

A similar procedure was used to sample a negative point.

We considered networks with number of channels 4,6,8,20,100 and 200 and 500.

Note that the ground truth network can be realized by a network with 4 channels.

For each number of channels we trained a convolutional network 100 times and averaged the results.

For each number of channels c, we ran gradient descent with learning rate 0.2 c and stopped it if it did not improve the cost for 20 consecutive iterations or if it had 0% training error for 200 consecutive iterations or if it reached 30000 iterations.

The last iteration was taken for the calculations.

We plot both average test error over all 100 runs and average test error only over the runs that ended at 0% train error.

For each number of channels 4, 6, 8, 20, 100, 200 ,500 the number of runs in which gradient descent converged to a 0% train error solution is 96, 99, 100, 100, 100, 100, 100, respectively.

For each 0% train error solution we recorded the number of patterns detected with c d = 0.0001 according to the Definition 5.1 (generalized to the OBD problem).

In the XORD problem we recorded similarly the number of patterns detected in experiments which are identical to the experiments in Section 8.2.1, except that in this case p + = p ??? = 0.98.

We will first need a few notations.

Define x 1 = (1, 1), x 2 = (1, ???1), x 3 = (???1, ???1), x 4 = (???1, 1) and the following sets: DISPLAYFORM0 We can use these definitions to express more easily the gradient updates.

Concretely, let j ??? W + t (i 1 ) ??? W ??? t (i 2 ) then the gradient update is given as follows: DISPLAYFORM1 the gradient update is given by: DISPLAYFORM2 We denote by x + a positive diverse point and x ??? a negative diverse point.

Define the following sums for ?? ??? {+, ???}: DISPLAYFORM3 By the conditions of the theorem, with probability at least (p + p ??? ) m all the points in the training set are diverse.

From now on we will condition on this event.

Furthermore, without loss of generality, we can assume that the training set consists of one diverse point x + and one negative points x ??? .

This follows since the network and its gradient have the same value for two different positive diverse points and two different negative points.

Therefore, this holds for the loss function defined in Eq. 4 as well.

We will now proceed to prove the theorem.

In Section 8.6.1 we prove results on the filters at initialization.

In Section 8.6.2 we prove several auxiliary lemmas.

In Section 8.6.3 we prove upper bounds on S ??? t , P + t and P ??? t for all iterations t. In Section 8.6.4 we characterize the dynamics of S + t and in Section 8.6.5 we prove an upper bound on it together with upper bounds on N Wt (x + ) and ???N Wt (x ??? ) for all iterations t.

We provide an optimization guarantee for gradient descent in Section 8.6.6.

We prove generalization guarantees for the points in the positive class and negative class in Section 8.6.7 and Section 8.6.8, respectively.

We complete the proof of the theorem in Section 8.6.9.

Lemma 8.1.

With probability at least 1 ??? 4e ???8 , it holds that DISPLAYFORM0 Proof.

Without loss of generality consider DISPLAYFORM1 2 , we get by Hoeffding's inequality DISPLAYFORM2 The result now follows by the union bound.

Lemma 8.2.

With probability DISPLAYFORM3 Proof.

Let Z be a random variable distributed as N (0, ?? 2 ).

Then by Proposition 2.1.2 in Vershynin (2017), we have DISPLAYFORM4 Therefore, for all 1 ??? j ??? k and 1 ??? i ??? 4, DISPLAYFORM5 and DISPLAYFORM6 The result follows by applying a union bound over all 2k weight vectors and the four points DISPLAYFORM7 From now on we assume that the highly probable event in Lemma 8.2 holds.

DISPLAYFORM8 Proof.

By Lemma 8.2 we have DISPLAYFORM9 and similarly ???N W0 (x ??? ) < 1.

Therefore, by Eq. 10 and Eq. 11 we get: DISPLAYFORM10 2.

For i ??? {2, 4} and j ??? W + 0 (i), it holds that w DISPLAYFORM11 4.

For i ??? {2, 4} and j ??? U + 0 (i), it holds that u DISPLAYFORM12 2.

For i ??? {2, 4} and j ??? W + 0 (i), it holds that w DISPLAYFORM13 4.

For i ??? {2, 4} and j ??? U + 0 (i), it holds that u DISPLAYFORM14 As before, by Lemma 8.2 we have N W2 (x + ) < ?? and ???N W2 (x ??? ) < 1.

Lemma 8.4.

For all t ??? 1 we have W DISPLAYFORM0 Proof.

We will first prove that DISPLAYFORM1 To prove this, we will show by induction on t ??? 1, that for all j ??? W + 0 (i) ??? W + 0 (l), where l ??? {2, 4} the following holds: DISPLAYFORM2 The claim holds for t = 1 by the proof of Lemma 8.3.

Assume it holds for t = T .

By the induction hypothesis there exists an l ??? {2, 4} such that j ??? W + T (i) ??? W ??? T (l ).

By Eq. 10 we have, DISPLAYFORM3 where a ??? {0, 1} and b ??? {???1, 0}.

T ??x l = w (j) 0 ??x l then l = l and either w (j) DISPLAYFORM0 T ?? x l < 0 and l = l. It follows that either w DISPLAYFORM1 In both cases, we have w DISPLAYFORM2

In order to prove the lemma, it suffices to show that DISPLAYFORM0 .., k}. We will show by induction on t ??? 1, that for all j ??? W + 0 (2) ??? W + 0 (4), the following holds: DISPLAYFORM1 The claim holds for t = 1 by the proof of Lemma 8.3.

Assume it holds for t = T .

By the induction hypothesis j ??? W + T (2) ??? W + T (4).

Assume without loss of generality that j ??? W + T (2).

This implies that j ??? W ??? T (2) as well.

Therefore, by Eq. 10 we have DISPLAYFORM2 where a ??? {0, 1} and b ??? {0, ???1}. By the induction hypothesis, w DISPLAYFORM3 where the first inequality follows since j ??? W + T (2) and the second by Eq. 13.

This implies that DISPLAYFORM4 Otherwise, assume that a = 0 and b = ???1.

By Lemma 8.2 we have w T ?? x 2 < 0 and j / ??? W + T (2), which is a contradiction.

DISPLAYFORM5 which concludes the proof.

Lemma 8.5.

For all t ??? 0 we have DISPLAYFORM6 0 + ?? t ??x 2 for ?? t ??? Z. This follows since the inequalities u DISPLAYFORM7 .

Assume by contradiction that there exist an iteration t for which u DISPLAYFORM8 0 + ?? t???1 ??x 2 where ?? t???1 ??? Z. 9 Since the coefficient of x i changed in iteration t, we have j ??? U + t???1 (1) ??? U + t???1 (3).

However, this contradicts the claim above which shows that if u DISPLAYFORM9 Lemma 8.6.

Let i ??? {1, 3} and l ??? {2, 4}. DISPLAYFORM10 Proof.

First note that by Eq. 11 we generally have u DISPLAYFORM11 , by the gradient update in Eq. 11 it holds that a t ??? {0, ???1}. Indeed, a 0 = 0 and by the gradient update if a t???1 = 0 or a t???1 = ???1 then a t ??? {???1, 0}.Assume by contradiction that there exists an iteration t > 0 such that b t = ???1 and b t???1 = 0.

Note that by Eq. 11 this can only occur if j ??? U + t???1 (l).

We have u DISPLAYFORM12 Lemma 8.7.

Let DISPLAYFORM13 and Y DISPLAYFORM14 Then for all t, DISPLAYFORM15 9 Note that in each iteration ??t changes by at most ??.

Proof.

We will prove the claim by induction on t. For t = 0 this clearly holds.

Assume it holds for t = T .

Let j 1 ??? W + T (1) and j 2 ??? W + T (3).

By Eq. 10, the gradient updates of the corresponding weight vector are given as follows: DISPLAYFORM16 where a ??? {0, 1} and b 1 , b 2 ??? {???1, 0, 1}. By Lemma 8.4, j 1 ??? W + T +1 (1) and j 2 ??? W + T +1 (3).

Therefore, DISPLAYFORM17 DISPLAYFORM18 Proof.

In Lemma 8.4 we showed that for all t ??? 0 and j ??? W DISPLAYFORM19 t ?? x 2 ??? ?? .

This proves the first claim.

The second claim follows similarly.

Without loss of generality, let j ??? U + t (1).

By Lemma 8.5 it holds that U DISPLAYFORM20 Therefore, by Lemma 8.6 we have u (j) t x 1 < ??, from which the claim follows.

For the third claim, without loss of generality, assume by contradiction that for j ??? U DISPLAYFORM21 , from which the claim follows.

Lemma 8.9.

The following holds: DISPLAYFORM0 Under review as a conference paper at ICLR 2019 DISPLAYFORM1 Proof.1.

The equality follows since for each i ??? {1, 3}, l ??? {2, 4} and j ??? W DISPLAYFORM2 2.

In this case for each i ??? {1, 3}, l ??? {2, 4} and j ??? W DISPLAYFORM3 3.

This equality follows since for each i ??? {1, 3}, l ??? {2, 4} and j ??? W DISPLAYFORM4 (since x l will remain the maximal direction).

Therefore, DISPLAYFORM5 In the second case, where we have DISPLAYFORM6 t ?? x i < ?? for i ??? {1, 3}. Note that by Lemma 8.6, any DISPLAYFORM7 .

By all these observations, we have DISPLAYFORM8 By Eq. 14 and Eq. 15, it follows that, Z DISPLAYFORM9 .

Applying these observations b times, we see that Y DISPLAYFORM10 4) where the equality follows by Lemma 8.4.

By Lemma 8.9, we have S DISPLAYFORM11 Hence, we can conclude that DISPLAYFORM12 Proof.

Define DISPLAYFORM13 First note that by Lemma 8.4 we have W DISPLAYFORM14 where the second equality follows by Lemma 8.4.

DISPLAYFORM15 (16) To see this, note that by Lemma 8.6 and Lemma 8.5 it holds that u DISPLAYFORM16 0 ?? x 2 and thus Eq. 16 holds.

Now assume that j ??? U + T (l) for l ??? {2, 4}. Then DISPLAYFORM17 Proof.

The claim holds for t = 0.

Consider an iteration T .

If DISPLAYFORM18 ?? , where the last inequality follows from the previous observation.

Hence, DISPLAYFORM19 The proof of the second claim follows similarly.

DISPLAYFORM20 The third claim holds by the following identities and bounds DISPLAYFORM21 ?? by the previous claims.

We are now ready to prove a global optimality guarantee for gradient descent.

Proposition 8.13.

Let k > 16 and ?? ??? 1.

With probabaility at least 1 ??? DISPLAYFORM0 iterations, gradient descent converges to a global minimum.

Proof.

First note that with probability at least 1 ??? DISPLAYFORM1 ???8 the claims of Lemma 8.1 and Lemma 8.2 hold.

Now, if gradient descent has not reached a global minimum at iteration t then either DISPLAYFORM2 where the last inequality follows by Lemma 8.1.

DISPLAYFORM3 by Lemma 8.9.

However, by Lemma 8.10, it follows that after 5 consecutive iterations t < t < t + 6 in which DISPLAYFORM4 To see this, first note that for all t, N Wt (x + ) ??? ??+3c ?? by Lemma 8.12.

Then, by Lemma 8.10 we have DISPLAYFORM5 where the second inequality follows by Lemma 8.1 and the last inequality by the assumption on k.

Assume by contradiction that GD has not converged to a global minimum after T = 7(??+1+8c??) DISPLAYFORM6 iterations.

Then, by the above observations, and the fact that S + 0 > 0 with probability 1, we have DISPLAYFORM7 However, this contradicts Lemma 8.12.

We will first need the following three lemmas.

Lemma 8.14.

With probability at least 1 ??? 4e ???8 , it holds that DISPLAYFORM0 Proof.

The proof is similar to the proof of Lemma 8.1.Lemma 8.15.

Assume that gradient descent converged to a global minimum at iteration T .

Then there exists an iteration T 2 < T for which S DISPLAYFORM1 Proof.

Assume that for all 0 ??? t ??? T 1 it holds that N Wt (x + ) < ?? and ???N Wt (x ??? ) < 1.

By continuing the calculation of Lemma 8.3 we have the following: DISPLAYFORM2 2.

For i ??? {2, 4} and j ??? W + 0 (i), it holds that w DISPLAYFORM3 Therefore, there exists an iteration DISPLAYFORM4 It suffices to show that for all T 1 ??? t < T 2 the following holds: DISPLAYFORM5 The first claim follows since at any iteration N Wt (x + ) can decrease by at most 2??k = 2c ?? .

For the second claim, let t < t be the latest iteration such that N W t (x + ) ??? ??.

Then at iteration t it holds that ???N W t (x ??? ) < 1 and N W t (x + ) ??? ??.

Therefore, for all i ??? {1, 3}, l ??? {2, 4} and DISPLAYFORM6 t + ??x l .

Hence, by Lemma 8.5 and Lemma 8.6 it holds that U + t +1 (1) ??? U + t +1 (3) = ???. Therefore, by the gradient update in Eq. 11, for all 1 ??? j ??? k, and all t < t ??? t we have u DISPLAYFORM7 The above argument shows that DISPLAYFORM8 Assume that k ??? 64 and gradient descent converged to a global minimum at iteration T .

Then, DISPLAYFORM9 Proof.

Notice that by the gradient update in Eq. 10 and Lemma 8.2, X + t can be strictly larger than max DISPLAYFORM10 .

We know by Lemma 8.15 that there exists T 2 < T such that S + T2 ??? ??+1???3c ?? and that N Wt (x + ) < ?? and ???N Wt (x ??? ) ??? 1 only for t > T 2 .

Since S + t ??? ?? + 1 + 8c ?? for all t by Lemma 8.12, there can only be at most DISPLAYFORM11 where the second inequality follows by Lemma 8.1 and the third inequality by the assumption on k.

We are now ready to prove the main result of this section.

Assume without loss of generality that z i = (???1, ???1) = x 3 .

Define

Notice that DISPLAYFORM0 Furthermore, by Lemma 8.7 we have DISPLAYFORM1 and by Lemma 8.14, DISPLAYFORM2 .

Combining this fact with Eq. 19 and Eq. 20 we get DISPLAYFORM3 which implies together with Eq. 18 that X DISPLAYFORM4 where the first inequality is true because DISPLAYFORM5 The second inequality in Eq. 21 follows since P + T ??? c ?? and by appyling Lemma 8.16.

Finally, the last inequality in Eq. 21 follows by the assumption on k.10 Hence, z is classified correctly.

We will need the following lemmas.

Lemma 8.18.

With probability at least 1 ??? 8e ???8 , it holds that DISPLAYFORM0 Under review as a conference paper at ICLR 2019Proof.

The proof is similar to the proof of Lemma 8.1 and follows from the fact that DISPLAYFORM1 Lemma 8.19.

Let DISPLAYFORM2 Then for all t, there exists X, Y ??? 0 such that |X| ??? ?? U + 0 (2) , |Y | ??? ?? U + 0 (4) and DISPLAYFORM3 Proof.

First, we will prove that for all t there exists a t ??? Z such that for DISPLAYFORM4 11 We will prove this by induction on t.

For t = 0 this clearly holds.

Assume it holds for an iteration t. Let j 1 ??? U ??? 0 (2) and j 2 ??? U ??? 0 (4).

By the induction hypothesis, there exists a T ??? Z such that u DISPLAYFORM5 .

In either case, by Eq. 11, we have the following update at iteration t + 1: DISPLAYFORM6 where a ??? {???1, 0, 1}. Hence, u DISPLAYFORM7 ??? (a t + a)??x 2 .

This concludes the proof by induction.

Now, consider an iteration t, j 1 ??? U + 0 (2), j 2 ??? U + 0 (4) and the integer a t defined above.

If a t ??? 0 then DISPLAYFORM8 ) which proves the claim in the case that a t ??? 0.If a t < 0 it holds that 11 Recall that by Lemma 8.5 we know that DISPLAYFORM9 Under review as a conference paper at ICLR 2019 DISPLAYFORM10 Since for all 1 ??? j ??? k it holds that u DISPLAYFORM11 4) which concludes the proof.

Lemma 8.20.

Let DISPLAYFORM12 Then for all t, DISPLAYFORM13 Proof.

We will first prove that for all t there exists an integer a t ??? 0 such that for DISPLAYFORM14 ?? x 4 + ??a t .

We will prove this by induction on t.

For t = 0 this clearly holds.

Assume it holds for an iteration t. DISPLAYFORM15 .

By the induction hypothesis, there exists an integer a t ??? 0 such that u DISPLAYFORM16 , it follows that if a t ??? 1 we have the following update at iteration T + 1: DISPLAYFORM17 where a ??? {???1, 0, 1}. Hence, u DISPLAYFORM18 0 ??x 2 +??(a t +a) and u DISPLAYFORM19 .

This concludes the proof by induction.

Now, consider an iteration t, DISPLAYFORM20 and the integer a t defined above.

We have, DISPLAYFORM21 It follows that DISPLAYFORM22 which concludes the proof.

We are now ready to prove the main result of this section.

Proof.

With probability at least 1 ??? ??? 2k ??? ??e 8k ??? 16e ???8 Proposition 8.13 and Lemma 8.18 hold.

It suffices to show generalization on negative points.

Assume that gradient descent converged to a global minimum at iteration T .

Let (z, ???1) be a negative point.

Assume without loss of generality that z i = x 2 for all 1 ??? i ??? d. Define the following sums for l ??? {2, 4}, DISPLAYFORM23 First, we notice that DISPLAYFORM24 We note that by the analysis in Lemma 8.18, it holds that for any t, j 1 ??? U + 0 (2) and j 2 ??? U + 0 (4), either j 1 ??? U + t (2) and j 2 ??? U + t (4), or j 1 ??? U + t (4) and j 2 ??? U + t (2).

We assume without loss of generality that j 1 ??? U + T (2) and j 2 ??? U + T (4).

It follows that in this case DISPLAYFORM25 12 Otherwise we would replace Y ??? T (2) with Y ??? T (4) and vice versa and continue with the same proof.

DISPLAYFORM26 .

By Lemma 8.20 and Lemma 8.18 DISPLAYFORM27 and by Lemma 8.19 and Lemma 8.18 there exists Y ??? c ?? such that: DISPLAYFORM28 Plugging these inequalities in Eq. 24 we get: DISPLAYFORM29 By Lemma 8.16 we have X ??? T ??? 34c ?? .

Hence, by using the inequality S ??? T ??? c ?? we conclude that DISPLAYFORM30 where the last inequality holds for k > 64 We will now prove pattern detection results.

In the case of over-paramterized networks, in Proposition 8.17 we proved that DISPLAYFORM31 1+??(k) .

Since for i ??? {1, 3} it holds that D xi ??? X + (i), it follows that patterns x 1 and x 3 are detected.

Similarly, in Proposition 8.21our analysis implies that, without loss of generality, DISPLAYFORM32 (under the assumption that we assumed without loss of generality), it follows that patterns x 2 and x 4 are detected.

The confidence of the detection is at 2 by a symmetry argument.

This will finish the proof of the theorem.

For the proof, it will be more convenient to denote the matrix of weights at iteration t as a tuple of 4 vectors, i.e., W t = w To see this, we will illustrate this through one case, the other cases are similar.

Assume, for example, that arg max 1???l???4 u( FORMULA0 t ?? x l = 3 and arg max l???{2,4} u(1) t ?? x l = 2 and assume without loss of generality that N W T ?? x 1 ??? 2c ?? and therefore, x 1 cannot be detected with confidence greater than 2c ?? .

2. Let Z 1 be the set of positive points which contain only the patterns x 1 , x 2 , x 4 , Z 2 be the set of positive points which contain only the patterns x 3 , x 2 , x 4 .

Let Z 3 be the set which contains the negative point with all patterns equal to x 2 and Z 4 be the set which contains the negative point with all patterns equal to x 4 .

By the proof of the previous section, if the event E holds, then there exists 1 ??? i ??? 4, such that gradient descent converges to a solution at iteration T which errs on all of the points in Z i .

Therefore, its test error will be at least p * (recall Eq. 3).

In this section we assume that we are given a training set S ??? {??1} 2 ?? {??1} 2 consisting of points (x 1 , 1), (x 2 , ???1), (x 3 , 1), (x 4 , ???1), where x 1 = (1, 1), x 2 = (???1, 1), x 3 = (???1, ???1) and x 4 = (1, ???1).

Our goal is to learn the XOR function with gradient descent.

Note that in the case of two dimensions, the convolutional network introduced in Section 3 reduces to the following two-layer fully connected network.

DISPLAYFORM0 We consider running gradient descent with a constant learning rate ?? ??? c?? k , c ?? ??? 1 and IID gaussian initialization with mean 0 and standard deviation ?? g = c?? 16k 3/2 .

We assume that gradient descent minimizes the hinge loss (W ) = where optimization is only over the first layer.

We will show that gradient descent converges to the global minimum in a constant number of iterations.

For each point x i ??? S define the following sets of neurons: DISPLAYFORM1 t ?? x i < 0

<|TLDR|>

@highlight

We show in a simplified learning task that over-parameterization improves generalization of a convnet that is trained with gradient descent.