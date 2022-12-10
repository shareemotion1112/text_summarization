In this paper, we propose a generalization of the BN algorithm, diminishing batch normalization (DBN), where we update the BN parameters in a diminishing moving average way.

Batch normalization (BN) is very effective in accelerating the convergence of a neural network training phase that it has become a common practice.

Our proposed DBN algorithm remains the overall structure of the original BN algorithm while introduces a weighted averaging update to some trainable parameters.

We provide an analysis of the convergence of the DBN algorithm that converges to a stationary point with respect to trainable parameters.

Our analysis can be easily generalized for original BN algorithm by setting some parameters to constant.

To the best knowledge of authors, this analysis is the first of its kind for convergence with Batch Normalization introduced.

We analyze a two-layer model with arbitrary activation function.

The primary challenge of the analysis is the fact that some parameters are updated by gradient while others are not.

The convergence analysis applies to any activation function that satisfies our common assumptions.

For the analysis, we also show the sufficient and necessary conditions for the stepsizes and diminishing weights to ensure the convergence.

In the numerical experiments, we use more complex models with more layers and ReLU activation.

We observe that DBN outperforms the original BN algorithm on Imagenet, MNIST, NI and CIFAR-10 datasets with reasonable complex FNN and CNN models.

Deep neural networks (DNN) have shown unprecedented success in various applications such as object detection.

However, it still takes a long time to train a DNN until it converges.

Ioffe & Szegedy identified a critical problem involved in training deep networks, internal covariate shift, and then proposed batch normalization (BN) to decrease this phenomenon.

BN addresses this problem by normalizing the distribution of every hidden layer's input.

In order to do so, it calculates the preactivation mean and standard deviation using mini-batch statistics at each iteration of training and uses these estimates to normalize the input to the next layer.

The output of a layer is normalized by using the batch statistics, and two new trainable parameters per neuron are introduced that capture the inverse operation.

It is now a standard practice Bottou et al. (2016) ; He et al. (2016) .

While this approach leads to a significant performance jump, to the best of our knowledge, there is no known theoretical guarantee for the convergence of an algorithm with BN.

The difficulty of analyzing the convergence of the BN algorithm comes from the fact that not all of the BN parameters are updated by gradients.

Thus, it invalidates most of the classical studies of convergence for gradient methods.

In this paper, we propose a generalization of the BN algorithm, diminishing batch normalization (DBN), where we update the BN parameters in a diminishing moving average way.

It essentially means that the BN layer adjusts its output according to all past mini-batches instead of only the current one.

It helps to reduce the problem of the original BN that the output of a BN layer on a particular training pattern depends on the other patterns in the current mini-batch, which is pointed out by Bottou et al..

By setting the layer parameter we introduce into DBN to a specific value, we recover the original BN algorithm.

We give a convergence analysis of the algorithm with a two-layer batch-normalized neural network and diminishing stepsizes.

We assume two layers (the generalization to multiple layers can be made by using the same approach but substantially complicating the notation) and an arbitrary loss function.

The convergence analysis applies to any activation function that follows our common assumption.

The main result shows that under diminishing stepsizes on gradient updates and updates on mini-batch statistics, and standard Lipschitz conditions on loss functions DBN converges to a stationary point.

As already pointed out the primary challenge is the fact that some trainable parameters are updated by gradient while others are updated by a minor recalculation.

Contributions.

The main contribution of this paper is in providing a general convergence guarantee for DBN.

Specifically, we make the following contributions.• In section 4, we show the sufficient and necessary conditions for the stepsizes and diminishing weights to ensure the convergence of BN parameters.• We show that the algorithm converges to a stationary point under a general nonconvex objective function.

This paper is organized as follows.

In Section 2, we review the related works and the development of the BN algorithm.

We formally state our model and algorithm in Section 3.

We present our main results in Sections 4.

In Section 5, we numerically show that the DBN algorithm outperforms the original BN algorithm.

Proofs for main steps are collected in the Appendix.

Before the introduction of BN, it has long been known in the deep learning community that input whitening and decorrelation help to speed up the training process.

In fact, Orr & Müller show that preprocessing the data by subtracting the mean, normalizing the variance, and decorrelating the input has various beneficial effects for back-propagation.

Krizhevsky et al. propose a method called local response normalization which is inspired by computational neuroscience and acts as a form of lateral inhibition, i.e., the capacity of an excited neuron to reduce the activity of its neighbors.

Gülçehre & Bengio propose a standardization layer that bears significant resemblance to batch normalization, except that the two methods are motivated by very different goals and perform different tasks.

Inspired by BN, several new works are taking BN as a basis for further improvements.

Layer normalization BID2 is much like the BN except that it uses all of the summed inputs to compute the mean and variance instead of the mini-batch statistics.

Besides, unlike BN, layer normalization performs precisely the same computation at training and test times.

Normalization propagation that Arpit et al. uses data-independent estimations for the mean and standard deviation in every layer to reduce the internal covariate shift and make the estimation more accurate for the validation phase.

Weight normalization also removes the dependencies between the examples in a minibatch so that it can be applied to recurrent models, reinforcement learning or generative models Salimans & Kingma (2016) .

Cooijmans et al. propose a new way to apply batch normalization to RNN and LSTM models.

Given all these flavors, the original BN method is the most popular technique and for this reason our choice of the analysis.

To the best of our knowledge, we are not aware of any prior analysis of BN.BN has the gradient and non-gradient updates.

Thus, nonconvex convergence results do not immediately transfer.

Our analysis explicitly considers the workings of BN.

However, nonconvex convergence proofs are relevant since some small portions of our analysis rely on known proofs and approaches.

The optimization problem for a network is an objective function consisting of a large number of component functions, that reads: DISPLAYFORM0 where DISPLAYFORM1 .., N , are real-valued functions for any data record X i .

Index i associates with data record X i and target response y i (hidden behind the dependency of f on i) in the training set.

Parameters θ include the common parameters updated by gradients directly associated with the loss function, i.e., behind the part that we have a parametric model, while BN parameters λ are introduced by the BN algorithm and not updated by gradient methods but by the mini-batch statistics.

We define that the derivative of f i is always taken with respect to θ: DISPLAYFORM2 The deep network we analyze has 2 fully-connected layers with D 1 neurons each.

The techniques presented can be extended to more layers with additional notation.

Each hidden layer computes y = a(W u) with activation function a(·) and u is the input vector of the layer.

We do not need to include an intercept term since the BN algorithm automatically adjusts for it.

BN is applied to the output of the first hidden layer.

We next describe the computation in each layer to show how we obtain the output of the network.

The notations introduced here is used in the analysis.

FIG0 shows the full structure of the network.

The input data is vector X, which is one of DISPLAYFORM3 is the set of all BN parameters and vector θ = W 1 , W 2 , (β DISPLAYFORM4 is the set of all trainable parameters which are updated by gradients.

Matrices W 1 , W 2 are the actual model parameters and β, γ are introduced by BN.

The value of j th neuron of the first hidden layer is DISPLAYFORM5 where W 1,j,· denotes the weights of the linear transformations for the j th neuron.

The j th entry of batch-normalized output of the first layer is DISPLAYFORM6 DISPLAYFORM7 The objective function for the i th sample is DISPLAYFORM8 where l i (·) is the loss function associated with the target response y i .

For sample i, we have the following complete expression for the objective function: DISPLAYFORM9 Function f i (X i : θ, λ) is nonconvex with respect to θ and λ.

Algorithm 1 shows the algorithm studied herein.

There are two deviations from the standard BN algorithm, one of them actually being a generalization.

We use the full gradient instead of the more popular stochastic gradient (SG) method.

It essentially means that each batch contains the entire training set instead of a randomly chosen subset of the training set.

An analysis of SG is potential future research.

Although the primary motivation for full gradient update is to reduce the burdensome in showing the convergence, the full gradient method is similar to SG in the sense that both of them go through the entire training set, while full gradient goes through it deterministically and the SG goes through it in expectation.

Therefore, it is reasonable to speculate that the SG method has similar convergence property as the full algorithm studied herein.

Algorithm 1 DBN: Diminishing Batch-Normalized Network Update Algorithm 1: Initialize θ ∈ R n1 and λ ∈ R n2 2: for iteration m=1,2,... do 3: DISPLAYFORM0 The second difference is that we update the BN parameters (θ, λ) by their moving averages with respect to diminishing α (m) .

The original BN algorithm can be recovered by setting α (m) = 1 for every m. After introducing diminishing α (m) , λ (m) and hence the output of the BN layer is determined by the history of all past data records, instead of those solely in the last batch.

Thus, the output of the BN layer becomes more general that better reflects the distribution of the entire dataset.

We use two strategies to decide the values of α (m) .

One is to use a constant smaller than 1 for all m, and the other one is to decay the α (m) gradually, such as α (m) = 1/m.

In our numerical experiment, we show that Algorithm 1 outperforms the original BN algorithm, where both are based on SG and non-linear activation functions with many layers FNN and CNN models.

The main purpose of our work is to show that Algorithm 1 converges.

In the general case, we focus on the nonconvex objective function.

Here are the assumptions we used for the convergence analysis.

Assumption 1 (Lipschitz continuity on θ and λ).

For every i we have DISPLAYFORM0 Noted that the Lipschitz constants associated with each of the above inequalities are not necessarily the same.

HereL is an upper bound for these Lipschitz constants for simplicity.

Assumption 2 (bounded parameters).

Sets P and Q are compact set, where θ ∈ P and λ ∈ Q. Thus, there exists a constant M that weights W and parameters λ are bounded element-wise by this constant M .

DISPLAYFORM1 This also implies that the updated θ, λ in Algorithm 1 remain in P and Q, respectively.

Assumption 3 (diminishing update on θ).

The stepsizes of θ update satisfy DISPLAYFORM2 This is a common assumption for diminishing stepsizes in optimization problems.

Assumption 4 (Lipschitz continuity of l i (·)).

Assume the loss functions l i (·) for every i is continuously differentiable.

It implies that there existsM such that DISPLAYFORM3 Assumption 5 (existence of a stationary point).

There exists a stationary point (θ DISPLAYFORM4 We note that all these are standard assumptions in convergence proofs.

We also stress that Assumption 4 does not directly imply 1.

Since we assume that P and Q are compact, then Assumptions 1, 4 and 5 hold for many standard loss function such as softmax and MSE.Assumption 6 (Lipschitz at activation function).

The activation function a(·) is Lipschitz with constant k: DISPLAYFORM5 Since for all activation function there is a(0) = 0, the condition is equivalent to |a(x) − a(0)| ≤ k x − 0 .

We note that this assumption works for many popular choices of activation functions, such as ReLU and LeakyReLu.

We first have the following lemma specifying sufficient conditions for λ to converge.

Proofs for main steps are given in the Appendix.

Theorem 7 Under Assumptions 1, 2, 3 and 6, if {α (m) } satisfies DISPLAYFORM0 We give a discussion of the above conditions for α (m) and η (m) at the end of this section.

With the help of Theorem 7, we can show the following convergence result.

Lemma 8 Under Assumptions 4, 5 and the assumptions of Theorem 7, when DISPLAYFORM1 we have DISPLAYFORM2 This result is similar to the classical convergence rate analysis for the non-convex objective function with diminishing stepsizes, which can be found in Bottou et al. (2016) .Lemma 9 Under the assumptions of Lemma 8, we have DISPLAYFORM3 This theorem states that for the full gradient method with diminishing stepsizes the gradient norms cannot stay bounded away from zero.

The following result characterizes more precisely the convergence property of Algorithm 1.Lemma 10 Under the assumptions stated in Lemma 8, we have DISPLAYFORM4 Our main result is listed next.

Theorem 11 Under the assumptions stated in Lemma 8, we have DISPLAYFORM5 We cannot show that {θ (m) }'s converges (standard convergence proofs are also unable to show such a stronger statement).

For this reason, Theorem 11 does not immediately follow from Lemma 10 together with Theorem 7.

The statement of Theorem 11 would easily follow from Lemma 10 if the convergence of {θ (m) } is established and the gradient being continuous.

We show in the Appendix that the set of sufficient and necessary conditions to satisfy the assumptions of Theorem 7 are h > 1 and k ≥ 1.

The set of sufficient and necessary conditions to satisfy the assumptions of Lemma 8 are h > 2 and k ≥ 1.

For example, we can pick DISPLAYFORM0 ) to achieve the above convergence result in Theorem 11.

We conduct the computational experiments with Theano and Lasagne on a Linux server with a Nvidia Titan-X GPU.

We use MNIST LeCun et al. (1998) , CIFAR-10 Krizhevsky & Hinton (2009) and Network Intrusion (NI) kdd (1999) datasets to compare the performance between DBN and the original BN algorithm.

For the MNIST dataset, we use a four-layer fully connected FNN (784 × 300 × 300 × 10) with the ReLU activation function and for the NI dataset, we use a four-layer fully connected FNN (784 × 50 × 50 × 10) with the ReLU activation function.

For the CIFAR-10 dataset, we use a reasonably complex CNN network that has a structure of (Conv-Conv-MaxPool-DropoutConv-Conv-MaxPool-Dropout-FC-Dropout-FC), where all four convolution layers and the first fully connected layers are batch normalized.

We use the softmax loss function and l 2 regularization with for all three models.

All the trainable parameters are randomly initialized before training.

For all 3 datasets, we use the standard epoch/minibatch setting with the minibatch size of 100, i.e., we do not compute the full gradient and the statistics are over the minibatch.

We use AdaGrad Duchi, John and Hazan, Elad and Singer (2011) to update the learning rates η (m) for trainable parameters, starting from 0.01.

We test all the choices of α (m) with the performances presented in Figure 2 .

Figure 2 shows that all the non-zero choices of α (m) converge properly.

The algorithms converge without much difference even when α (m) in DBN is very small, e.g., 1/m 2 .

However, if we select α (m) = 0, the algorithm is erratic.

Besides, we observe that all the non-zero choices of α (m) converge at a similar rate.

The fact that DBN keeps the batch normalization layer stable with a very small α (m) suggests that the BN parameters do not have to be depended on the latest minibatch, i.e., the original BN.We compare a selected set of the most efficient choices of α TAB2 shows the best result obtained from each choice of α (m) .

Most importantly, it suggests that the choices of α (m) = 1/m and 1/m 2 perform better than the original BN algorithm.

Besides, all the constant less-than-one choices of α (m) perform better than the original BN, showing the importance of considering the mini-batch history for the update of the BN parameters.

The BN algorithm in each figure converges to similar error rates on test datasets with different choices of α (m) except for the α (m) = 0 case.

Among all the models we tested, α (m) = 0.25 is the only one that performs top 3 for all three datasets, thus the most robust choice.

To summarize, our numerical experiments show that the DBN algorithm outperforms the original BN algorithm on the MNIST, NI and CIFAT-10 datasets with typical deep FNN and CNN models.

On the analytical side, we believe an extension to more than 2 layers is doable with significant augmentations of the notation.

The following proofs are shortened to corporate with AAAI submission page limit.

Proposition 12 There exists a constant M such that, for any θ and fixed λ, we have DISPLAYFORM0 Proof.

By Assumption 5, we know there exists (θ * , λ * ) such that ∇f (θ * , λ * ) 2 = 0.

Then we have DISPLAYFORM1 where the last inequality is by Assumption 1.

We then have ∇f (θ, λ) DISPLAYFORM2 because sets P and Q are compact by Assumption 2.

Proof.

This is a known result of the Lipschitz-continuous condition that can be found in Bottou et al. (2016) .

We have this result together with Assumption 1.

Lemma 14 When DISPLAYFORM0 is a Cauchy series.

Proof.

By Algorithm 1, we have DISPLAYFORM1 We defineα DISPLAYFORM2 Then we have DISPLAYFORM3 It remains to show that DISPLAYFORM4 DISPLAYFORM5 implies the convergence of {μ (m) }.

By (28), we have Π DISPLAYFORM6 It is also easy to show that there exists C and Mc such that for all m ≥ Mc, we have DISPLAYFORM7 Therefore, lim DISPLAYFORM8 Thus the following holds: DISPLAYFORM9 and DISPLAYFORM10 From equation 29 and equation 32 it follows that the sequence {μ DISPLAYFORM11 } is a Cauchy series.

Lemma 15 Since {μ DISPLAYFORM12 } is a Cauchy series, {µ DISPLAYFORM13 } is a Cauchy series.

Proof.

We know that µ

} is a Cauchy series.

Proof.

We define σ DISPLAYFORM0 Since {µ (m) j } is convergent, there exists c1, c2 and N1 such that for any m > N1, −∞ < c1 < µ DISPLAYFORM1 Inequality equation 35 is by the following fact: DISPLAYFORM2 where b and ai for every i are arbitrary real scalars.

Besides, equation 39 is due to −2aic ≤ max{−2|ai|c, 2|ai|c}.Inequality equation 36 follow from the square function being increasing for nonnegative numbers.

Besides these facts, equation 36 is also by the same techniques we used in equation 23-equation 25 where we bound the derivatives with the Lipschitz continuity in the following inequality: DISPLAYFORM3 Inequality equation 37 is by collecting the bounded terms into a single boundML ,M .

Therefore, DISPLAYFORM4 Using the similar methods in deriving equation 28 and equation 29, it can be seen that a set of sufficient conditions ensuring the convergence for {σ DISPLAYFORM5 Therefore, the convergence conditions for {σ (m) j } are the same as for {µ DISPLAYFORM6 It is clear that these lemmas establish the proof of Theorem 7.

Proposition 17 Under the assumptions of Theorem 7, we have |λ (m) −λ|∞ ≤ am, where DISPLAYFORM0 M1 and M2 are constants.

Proof.

For the upper bound of σ DISPLAYFORM1 , by equation 38, we have DISPLAYFORM2 We defineσj :=σ DISPLAYFORM3 The first inequality comes by substituting p by m and by taking lim as q → ∞ in equation 41.

The second inequality comes from equation 30.

We then obtain, DISPLAYFORM4 The second inequality is by (1 − α (1) )...(1 − α (m) ) < 1, the third inequality is by equation 30 and the last inequality can be easily seen by induction.

By equation 44, we obtain DISPLAYFORM5 Therefore, we have DISPLAYFORM6 The first inequality is by equation 45, the second inequality is by equation 41, the third inequality is by equation 31 and the fourth inequality is by adding the nonnegative termσ j C α (m) to the right-hand side.

For the upper bound of µ DISPLAYFORM7 Let us define Am := μ (m) −μ (∞) and Bm := μj DISPLAYFORM8 } is a Cauchy series, by equation 27, |μ DISPLAYFORM9 .

Therefore, the first term in equation 47 is bounded by DISPLAYFORM10 For the second term in equation 47, recall that C : DISPLAYFORM11 where the inequality can be easily seen by induction.

Therefore, the second term in equation 47 is bounded by DISPLAYFORM12 From these we obtain DISPLAYFORM13 DISPLAYFORM14 where M1 and M2 are constants defined as DISPLAYFORM15 Proposition 18 Under the assumptions of Theorem 7, DISPLAYFORM16 , where am is defined in Proposition 17.Proof.

For simplicity of the proof, let us define DISPLAYFORM17 where √ n2 is the dimension of λ.

The second inequality is by Assumption 1 and the fourth inequality is by Proposition 17.

Inequality equation 51 implies that for all m and i, we have |x DISPLAYFORM18

This is established by the following four cases.

DISPLAYFORM0 by Proposition 12.

DISPLAYFORM1 The last inequality is by Proposition 12.All these four cases yield equation 52.Proposition 19 Under the assumptions of Theorem 7, we havē DISPLAYFORM2 where M is a constant and am is defined in Proposition 17.Proof.

By Proposition 13, DISPLAYFORM3 2 .

Therefore, we can sum it over the entire training set from i = 1 to N to obtain DISPLAYFORM4 In Algorithm 1, we define the update of θ in the following full gradient way: DISPLAYFORM5 which implies DISPLAYFORM6 By equation 56 we haveθ DISPLAYFORM7 .

We now substituteθ := θ (m+1) , θ := θ (m) and λ :=λ into equation 54 to obtain DISPLAYFORM8 The first inequality is by plugging equation 56 into equation 54, the second inequality comes from Proposition 12 and the third inequality comes from Proposition 18.

Here we show Theorem 11 as the consequence of Theorem 7 and Lemmas 8, 9 and 10.6.4.1 PROOF OF LEMMA 8Here we show Lemma 8 as the consequence of Lemmas 20, 21 and 22.Lemma 20 DISPLAYFORM0 Proof.

By plugging equation 45 and equation 43 into equation 58, we have the following for all j: DISPLAYFORM1 It is easy to see that the the following conditions are sufficient for right-hand side of equation 59 to be finite: DISPLAYFORM2 Therefore, we obtain DISPLAYFORM3 Lemma 21 Under Assumption 4, DISPLAYFORM4 is a set of sufficient conditions to ensure DISPLAYFORM5 Proof.

By Assumption 4, we have DISPLAYFORM6 By the definition of fi(·), we then have DISPLAYFORM7 The first inequality is by the Cauchy-Schwarz inequality, and the second one is by equation 60.

To show the finiteness of equation 64, we only need to show the following two statements: DISPLAYFORM8 and DISPLAYFORM9 Proof of equation 65: For all j we have DISPLAYFORM10 The inequality comes from |W Finally, we invoke Lemma 14 to assert that DISPLAYFORM11 Proof of equation 66: For all j we have DISPLAYFORM12 The first term in equation 68 is finite since {µ Noted that function f (σ) = 1 σ + B is Lipschitz continuous since its gradient is bounded by 1 Next we show that each of the four terms in the right-hand side of equation 75 is finite, respectively.

For the first term, DISPLAYFORM13 is by the fact that the parameters {θ, λ} are in compact sets, which implies that the image of fi(·) is in a bounded set.

For the second term, we showed its finiteness in Lemma 21.

The right-hand side of equation 77 is finite because DISPLAYFORM14 and DISPLAYFORM15 The second inequalities in equation 78 and equation 79 come from the stated assumptions of this lemma.

For the fourth term, DISPLAYFORM16 holds, because we have In Lemmas 20, 21 and 22, we show that {σ (m) } and {µ (m) } are Cauchy series, hence Lemma 8 holds.

This proof is similar to the the proof by Bertsekas & Tsitsiklis (2000) .Proof.

By Theorem 8, we have DISPLAYFORM0 If there exists a > 0 and an integerm such that ∇f (θ (m) ,λ) 2 ≥ Since k ≥ 1 due to Assumption 3, we conclude that k + h > 2.

Therefore, the conditions for η (m) and α (m) to satisfy the assumptions of Theorem 7 are h > 1 and k ≥ 1.

For the assumptions of Theorem 7, the first condition DISPLAYFORM0 requires h > 2.Besides, the second condition is DISPLAYFORM1 The inequality holds because for any p > 1, we have DISPLAYFORM2 Therefore, the conditions for η (m) and α (m) to satisfy the assumptions of Lemma 8 are h > 2 and k ≥ 1.

@highlight

We propose a extension of the batch normalization, show a first-of-its-kind convergence analysis for this extension and show in numerical experiments that it has better performance than the original batch normalizatin.