We formulate an information-based optimization problem for supervised classification.

For invertible neural networks, the control of these information terms is passed down to the latent features and parameter matrix in the last fully connected layer, given that mutual information is invariant under invertible map.

We propose an objective function and prove that it solves the optimization problem.

Our framework allows us to learn latent features in an more interpretable form while improving the classification performance.

We perform extensive quantitative and qualitative experiments in comparison with the existing state-of-the-art classification models.

Quantities of information are nonlinear measures capable of describing complex relationship between unstructured data and they form the basis of the probabilistic algorithms in the literature of machine learning.

Information theoretic methods are also reported to be effective on improving deep generative models BID6 ; Kim & Mnih (2018) ) and deep learning models for classification (Grandvalet & Bengio (2004) ; Pereyra et al. (2017) ).

Information Bottleneck (IB) problem BID12 ) is formulated as: DISPLAYFORM0 where the solution random variable T is interpreted as a minimal sufficient representation of signal X for label Y and the mutual information is defined as DISPLAYFORM1 p(x, y) log p(x, y) p(x)p(y) dydx .The term I(X; T ) has its origins in Lossy Compression and Rate-Distortion Theory (Cover & BID7 , conveying an simple idea of "keep only what is relevant".However, Saxe et al. (2018) argued that the mutual information I(X; T ) between signal X and feature T in intermediate layer is infinite, as the transformation from X to continuous random variable T is deterministic.

In addition they showed experimentally that layers equipped with ReLU actually do not compress too much information, which is supported by many recent work on the invertibility of the neural network BID8 ; Jacobsen et al. (2018) ).

This motivates us to consider a different problem with similar principle idea: we would like to establish a theoretically valid objective that allows the neural network to extract only the relevant information for classification from the data.

We focus on the discrete prediction random variable Y inferred by the probabilistic model P( Y |X) and introduce the following information optimization problem for supervised classification:maximize I(Y ; Y ) subject to I(X; Y ) − I(Y ; Y ) < τ , for some τ > 0 .The intuition behind this objective lies in twofold:Information perspective: A good classification model should be robust against irrelevant features of X, and prevent over-fitting in the learning process.

In optimization problem (3) we maximize the relevant information I(Y ; Y ), while constraining the irrelevant information I(X; Y ) − I(Y ; Y )that X has about Y .

Although I(X; Y ) − I(Y ; Y ) converges to zero as I(Y ; Y ) approaching its maximum (see FIG0 ), in practice it's never attained due to the limited capacity of the models or over-fitting.

Our proposed constrain addresses the problem of over-fitting: if two models achieve the similar classification accuracy, this constraint prefers the one that does not overfit to spurious factors of variation in X (e.g., pixel-level artifact/noise in the image that accidentally correlates to the labels in the training data).Prediction confidence perspective: A good classification model should not be certain about its decision which is in fact wrong.

However, modern neural networks are too confident in their predictions (Guo et al., 2017; Szegedy et al., 2015; Pereyra et al., 2017) .

To be more precise, high capacity neural networks mostly assign labels of data with prediction confidence near 0 or 1.

In particular, they assign 0 probability to some correct labels and therefore do not have enough flexibility to correct themselves from making the wrong prediction.

We propose to compress the irrelevant information I(X; Y ) − I(Y ; Y ), where minimizing I(X; Y ) decreases the confidence on all predictions but maximizing I(Y ; Y ) increases the confidence on the correct predictions.

Therefore the overall effect reduces the certainty on the false prediction of Y (see FIG0 ).To solve this optimization problem, we first present some insights on the dynamics of deep neural network, which can be decomposed into two stages: (i) Transformation stage: samples {X k } k=1:n of the high dimensional unstructured signal X are transformed under the deep invertible (information preserving) feature map F to become (almost) linearly separable; (ii) Classification stage: the weight matrix w in the last fully connected layer together with the Softmax function, takes structured features {F (X k )} k=1:n as input and gives predictions.

Invertibility of F allows neural networks to pass the control of I(X; Y ) = I(F (X); Y ) towards F (X) and w in the last layer, where F (X) can be interpreted as transformed signal that preserves information about the original signal X and the inference model becomes conceptually linear with classifier w (see FIG0 ).

In Section 2 we derive objective function (7) and prove that it solves the optimization problem (3).

We show the classification performance is improved in Section 4.1 and the features F (X) are sculpted into a form with more interpretability entry-wise in Section 4.2. .

The optimal solution is obtained when the smaller disks coincide, which is typically not achieved in practice.

In particular, the trained model may be extremely confident in its prediction (when H( Y ) lies inside of H(X)), but predicts the wrong label (having large grey area).

Our optimization problem explicitly prohibits the growth of grey area throughout the training.

(R) Logic chart of our formulation: our proposed optimization problem only involves F (X) and w, allowing us to have control over the latent feature F (X) directly.

The invertibility property has been empirically demonstrated for complex non-linear deep neural networks that are widely used in practice.

We will discuss the literature in Section 3.

In addition, we prove in Proposition C.1 that a lower bound of classification error is minimized if neural network is invertible.

Our contribution: Our contribution lies in the following: (i) we formulated a novel information optimization problem for supervised classification; (ii) we propose a simple objective function that improves supervised deep learning with better performance and interpretability; (iii) we formally justify the use of 1 , 2 regularization from an information perspective.

Different from the naive regularization on w, our regularization on w T F (X) is novel and effective.

Consider a classification problem where the training data D = {(x k , y k )} k=1:n are sampled from random variable pair (X, Y ) with unknown joint distribution.

Each x k is fed into a deep probabilistic model, which outputs probability densities and predicts y k , a realization of the prediction random variable Y .

Let C denote the label class and X denote the signal space, then the mutual information between random variables, e.g., continuous X and discrete Y , is defined as DISPLAYFORM0 p(x)p( y) dx, and the entropy of Y is defined as H(Y ) = − y∈C p(y) log p(y).We first call out our assumptions used throughout our analysis.

(I): we assume the marginal densities of Y, Y are uniform over C; (II): there exists a unique true label for every sample of X.Mutual information is bounded and its gradient with respect to logits is approximately zero over a large domain.

In particular if the logits are initially small for true labels, gradient updates cannot effectively correct them.

Therefore it's not practical to train mutual information terms directly.

In this section we introduce alternative terms and prove that they are feasible for our purpose.

We show in Proposition 2.2 that I(Y ; Y ) is maximized if the classical cross entropy objective is minimized.

On the other hand, we show in Proposition 2.1 that for invertible DISPLAYFORM1 We derive our objective function (7) in Section 2.2.

Our experimental result in Section 4.1 verifies that the proposed objective function does compress the irrelevant information I(X; Y ) − I(Y ; Y ).

Without loss of generality, we consider the binary classification problem, i.e. the label class C = {±1}. To tract the population quantities I(F (X); Y ) and I(Y ; Y ), we decompose each of them into an empirical part and a probabilistic bound, which is negligible if sample size n is large.

In Proposition 2.1, we show that in order to compress I(F (X); Y ), we need to compress the norm of classifier w and feature F (X).

In particular, smaller |w T F (X)| represents lower confidence of the model on its predictions Y , indicating a smaller amount of mutual information I(F (X); Y ).

The proof is provided in Appendix A.Proposition 2.1.

I(X; Y ) = I(F (X); Y ) is well estimated by its empirical version n k=1 y∈C p( y|x k ) log(2p( y|x k ))/n with high probability, which shares the same unique (global) minimum with DISPLAYFORM0 Denote the sigmoid function with σ(a) = 1/(1 + e −a ).

Proposition 2.2 establishes the relationship between maximization over mutual information I(Y ; Y ) and minimization over cross entropy − n k=1 log σ(y k w T F (x k )); higher confidence of the model on its correct predictions over the samples indicates a larger value of I(Y ; Y ).

The proof is provided in Appendix B.Proposition 2.2.

I(Y ; Y ) is well estimated by its empirical version y y π y y log π y y πy+ π + y with high probability, which shares the same unique (global) maximum with DISPLAYFORM1 unbiased estimate of p(y, y) and π y+ = y∈C π y y , π + y = y∈C π y y .

In Lagrangian form of optimization problem (3), the constant τ can be dropped and the objective becomes DISPLAYFORM0 Consider a single signal x k and its true label y k , we propose the following objective function for binary supervised classification problem: DISPLAYFORM1 where R is some regularizer.

According to results in Section 2.1, minimizing (5) allows us to maximize I(Y ; Y ) while constraining I(X; Y ).

We typically choose the hyper-parameter α > 0 to be a reasonably small number.

The intuition comes from the observation that λ/(1 + λ) is upper bounded by one.

If we compress I(X; Y ) harshly, then neural networks may choose to minimize I(F (X); Y ) at a cost of minimizing I(Y ; Y ).Recall from the information theoretic perspective of our proposed optimization problem, our regularizer should prefer a model that does not overfit among all the ones with high training accuracy.

In this case neural networks assign only large logits w T y k F (x k ) to true label y k for each signal x k , and generalization of (5) to multi-class case for I(F (X); Y ) can be simplified to constraining w DISPLAYFORM2 , where w j is the jth column of w, assigning feature F (x k ) a probability to label j. We propose to simply constrain w T y k F (x k ) and does not encourage increasing w DISPLAYFORM3 In our experiment, we take the Elastic Net approach by BID15 using a combination of 2 and 1 regularizers: we use Holder's inequality to bound |w DISPLAYFORM4 In practice we assume sup F (X) to be a constant and is absorbed into the hyper-parameter.

Our proposed objective function is of the form: DISPLAYFORM5 .

(7) Notice that classical training methods maximize cross entropy and therefore I(Y ; Y ), but do not compress I(X; Y ) explicitly.

In Equation (7) , we explicitly encourage the compression of irrelevant information I(X; Y ) − I(Y ; Y ).

As we demonstrate in Section 4.3, the proposed objective function does encourage a smaller amount of I(X; Y ) − I(Y ; Y ) throughout the training process.

Recent experimental work reported that neural networks with invertible structure have better performance.

BID8 showed that images can be resconstructed from the latent features in AlexNet through an inverting process; this reconstruction is further improved by BID14 , where they built an encoder-decoder structure to encourage invertibility and showed reconstructive objective is beneficial to the performance of the neural network (e.g., VGGNet).

Shang et al. FORMULA0 proposed an invertible activation scheme named CReLU to preserve information; BID10 analyzed theoretically the invertibility of CNN; Jacobsen et al. FORMULA0 built a theoretic invertible structure whose performance is comparable to ResNet He et al. (2015) .

Invertibility seems to be an intriguing property or design principle that often emerges in the recent state-of-the-art deep architectures.

Information theoretic methods are reported to be effective attacking machine learning problems.

In deep learning, IB was first introduced in BID11 and the follow-up experimental work Shwartz-Ziv & Tishby (2017) .

They argued that DNN structure forms a markov chain and information is compressed layer by layer.

The theoretical breakthrough by BID0 Information Maximization(RIM) approach for classification problems; BID6 proposed to add information ingredients to the objective of GANs, encourage to learn disentangled representations.

Our framework decomposes deep neural network into a composition of nonlinear transformation map and a linear probablistic model; this idea was originated in BID5 where they considered the blind separation problem and decompose the prediction Y to be the sum of an invertible deterministic part and a stochastic part.

BID2 and Kolchinsky et al. (2017) also studied the IB problem in a stochastic setting.

Our idea of explicit regularization on w and F (X) is related to the margin based and stability based interpretations of generalization in deep learning respectively, studied by BID3 , BID4 , Neyshabur et al. (2017) , Sun et al. (2015) .

In our experiments we build the feature map F with ResNet or InvNet (introduced in Section 4.2).

In Appendix D we prove that ResNet by He et al. (2015) is invertible under mild assumptions.

We prefix the name of the model trained under our objective with "Reg", i.e. RegResNet/RegInvNet.

We report the accuracy of ResNet on test data of CIFAR100 in TAB0 We compare the performance between our proposed regularization on w T y k F (x k ) and the naive regularization on w. In both form of regularization we take α 1 = 0 and test over different choices of α 2 .

We pick smaller α 2 for naive regularization because it's applied to the full matrix.

We observe the performance of ResNet-32 under naive regularization drops monotonically as α 2 increases.

Under a suitable choice of hyperparameters, RegResNet outperforms ResNet by a noticeable margin.

It implies that our proposed constrain on the irrelevant information I(X; Y ) − I(Y ; Y ) is beneficial to the classification performance.

However, if the hyperparameters are too large, the performance drops, i.e. α 2 = 0.05 for ResNet-32 and α 2 = 0.15 for ResNet-Wide; this matches the discussion in Section 2.2 that the model may try to reduce the relevant information I(Y ; Y ).

Our approach addresses the problem of over-fitting; ResNet-Wide is improved by a larger margin compared to ResNet-32 because ResNet-Wide has higher capacity and is therefore over-fits more to the training data.

We introduce another invertible structured neural network on MNIST dataset and analyze the feature F (X) learned in the last layer qualitatively.

The feature map F is built to be LeNet-300-100 and the decoder D has the opposite structure.

At each step during the training process, we update the autoencoder F + D and the InvNet F + w alternatively; our regularization is applied to w as usual.

In this section we report our result with α 1 = α 2 = 0.002.

We feed 1k testing samples of digit 9 into the neural network to get 1k samples of features F (X).

Recall that the features F (X) are vectors of dimension 100 and w is a matrix of size 100 × 10.

We calculate the mean and standard deviation of each entry of the features from these 1k samples.

From FIG3 we see that under our regularization, the entry-wise products w 10 T i F (X) i becomes sparse as only a few entries have large values for both feature and weight.

This implies that the information needed to compute the logits for classification is encoded into only a few entries of the feature, which we regard as relevant entries.

Our regularization forces small w 10 on irrelevant entries, so the logits it outputs are not sensitive to variations in these entries; on the other hand, we do encourage large w 10 on the relevant entries.

This matches our motivation that a neural network should be robust against irrelevant information I (

It had been argued by Szegedy et al. (2013) that it is the space but not individual units in high level features that encodes interpretable information.

Under our regularization, a meaningful basis of the informative space is found; in particular, the features of 9 are encoded into 10 entries (see FIG3 .

On the other hand, features that have high values in these entries are expected to be the features of digit 9.To validate this conjecture, for each digit, we find the entry of its feature mean with the highest value.

We use the micro and macro average ROC metric on these feature entries and compare the results from InvNet and RegInvNet in FIG4 .

The curve with larger area underneath indicates higher representative power of individual entries learned in the features.

We conclude that under our regularization, relevant information for classification is encoded into only a few key entries of the features, and these entries are highly indicative and interpretable.

We trained ResNet-32 on CIFAR10, the feature F (X) is a vector of size 64, the classifier w is a matrix of size 64 × 10.

Note that the product w T F (X) is a vector of size 10 representing the probability assigned by the model for each class.

Under our framework, deep learning can be conceptually simplified to regularized linear regression if we regard F (X) as input data.

However, F (X) depends on the model parameters in the previous layers so it's not fixed like real data.

Moreover we observe in our experiments that naive regularization on w alone will upscale the norm of F (X), which neutralizes the effect of regularization.

In Figure 4 we show that as under our regularization, the 2 -norm of w is suppressed while the 2 -norms of feature F (X) remain similar.

In addition, several rows of w are trained to be zero, which implies that many entries of the feature F (X) are regarded by the network as irrelevant information for classification, since any variance in the entries of F (X) where the corresponding rows of w are zero has no influence on the probability assigned to each label class by the model.

Figure 4 : Compression of the RegResNet-32 (Blue) and the original ResNet-32 (Orange) on CIFAR-10 over the training process: the first plot records the average 2 norm of the last layer features F (X) in a batch; the second plot records the average 2 norm of the columns of w; the third plot records the ratio of zero entries among all entries of w; the plots for trained w with/without regularization after 84000 steps are listed on the right.

Best test accuracy are 92.86% and 92.64% for regularized and original ResNet-32 respectively.

Under our regularization, the norm of the feature learned remains similar and the norm of classifier w is smaller.

Therefore w is less sensitive to "support" and "outlier" features.

Invertibility allows us to treat F (X) as transformed data that preserves all the information from X, and therefore work on the information regularization problem under a linear scheme.

In Appendix D we prove that ResNet is fairly invertible due to the intrinsic invertibility of the operator I + L given L < 1.

In this section we build a PlainNet by using only L as the operator for each building block, so the theoretical guarantee for invertibility is not present for PlainNet.

In TAB2 , we see that PlainNet-32 can still benefit from our regularization, however, it's performance is less stable compared to ResNet-32 if the hyper-parameters are too large.

The reason is for PlainNet, the feature in the last layer F (X) does not preserve information about X very well, so it has a higher demand on the capacity of the classifer w and is therefore more sensitive to our regularization.

We give an interpretation of the deep learning dynamics by decomposing it into an signal transformation stage and feature classification stage, where we emphasis importance of the classifier w in the last fully connected layer given that the feature map F is invertible.

Then we take the advantage of the fact that mutual information quantities are invariance under invertible mapping to attack our proposed information optimization problem for supervised classification in deep learning.

Our theory justifies the use of direct regularization terms on w, F (X) for neural networks with invertibility property.

Our regularization improves the performance of neural networks by a noticeable margin and is capable of encouraging the interpretability of the entries of features learned in the last layer.

A PROOF OF PROPOSITION 2.1 Proposition 2.1 establishes a connection between I(X, Y ) and the absolute value of the logits |w T F (X)| for the binary case.

Intuitively, decreasing the confidence of the model on its predictions will decrease the mutual information I(X; Y ).Proposition 2.1.

I(X; Y ) = I(F (X); Y ) is well estimated by its empirical version (Montecarlo approximation) with high probability, which shares the same unique (global) minimum with DISPLAYFORM0 The mutual information I(X; Y ) is given as DISPLAYFORM1 Apply the assumption (II), the marginal distribution of Y is uniformly distributed: DISPLAYFORM2 Substituting FORMULA16 into FORMULA15 yields DISPLAYFORM3 According to the Hoeffding's inequality for bounded random variables [Proposition 2.2.6, Vershynin FORMULA0 ], let M, m denote upper and lower bounds of the integrand of (10) correspondingly, we have DISPLAYFORM4 Equivalently, with probability at least 1 − δ, DISPLAYFORM5 Here n k=1 y∈C p( y|x k ) log(2p( y|x k ))/n is a Monte carlo estimation of RHS of I(X; Y ).

Recall that, for the binary case p( y|x) = p( y|F (x)) can expressed as DISPLAYFORM6 Then we have DISPLAYFORM7 which is bounded by [0, log(2)].Take M = log(2), m = 0, we have DISPLAYFORM8 hold with probability at least 1 − δ.

The conclusion follows from the fact that n k=1 y∈C p( y|x k ) log(2p( y|x k ))/n has a unique global minimum at w T F (x k ) = 0 for each x k .

Consider the training samples {(x k , y k )} k=1:n , each x k is fed into a deep probabilistic model which outputs probability densities and predicts y k , a realization of the prediction Y .

Let C = {±1} be the binary class and n y be the counts of observed occurrences of k satisfying y k = y ∈ C, then n = y∈C n y = y n y , where we omit the range over C for convenience.

We denote the true joint probability with π y y = p(y, y), the marginal probabilities with π y+ = y π y y and π + y = y π y y .

The mutual information I(Y ; Y ) can be expressed as DISPLAYFORM0 Our empirical mutual information I( π) is defined as DISPLAYFORM1 where π y y = 1 2ny DISPLAYFORM2 Proposition 2.2 establishes a connection between I(Y ; Y ) and the cross-entropy objective for the binary case.

Intuitively, increasing the confidence of the model on its correct predictions will establish a more deterministic relationship between Y and Y and thus increase the mutual information I(Y ; Y ).

is well estimated by I( π) with high probability, which shares the same unique (global) maximum with DISPLAYFORM0 , for all k ∈ {1, ..., n}. Proposition 2.2 follows from Proposition B.1 and Proposition B.2, where Proposition B.1 shows that I(Y ; Y ) is well approximated by I( π) with high probability and Proposition B.2 shows the remaining claims in Proposition 2.2.As shown in Lemma B.1, π y y = 1 2ny DISPLAYFORM1 is an unbiased estimate of π y y .

Here σ to represent the sigmoid function defined by σ(x) = e x / (e x + 1).

By leveraging the concentration property of bounded variables, i.e., σ( yw T F (x i )), the estimation error can be bounded with high probability (Lemma B.2).

Lemma B.1.

The empirical joint probability, defined as DISPLAYFORM2 is an unbiased estimate of the true joint distribution π y y .

Lemma B.2.

With probability at least 1 − δ, we have DISPLAYFORM3 Proposition B.1.

With probability at least 1 − δ, DISPLAYFORM4 Proof.

To estimate the empirical mutual information given fixed samples, we use the approach by Hutter & Zaffalon (2005) .

In particular, taylor expansion gives DISPLAYFORM5 where ∆ y y = π y y −π y y .

Hence, Eq (21) together with Lemma B.2 yield, with probability exceeding 1 − |C| 2 δ, DISPLAYFORM6 Notice that, in the binary case the cardinality |C| = 2.Next we prove the intermediate results, Lemmas B.1 and B.2.Proof of Lemma B.1.

Direct derivation on the true joint distribution π y y gives DISPLAYFORM7 Given assumption (I) which states that the marginal density of Y is uniform over C, for every true label y ∈ C we have p(y) = 1 2 .

We can therefore rewrite (23) as DISPLAYFORM8 According to assumption (II), p(x|y) is a probability density over space of signal x with true label y.

The Monte Carlo estimation of (24) gives the empirical joint probability which is unbiased: DISPLAYFORM9 Proof of Lemma B.2.

Again, by leveraging the Hoeffding's inequality for bounded random variables [Proposition 2.2.6 of Vershynin FORMULA0 ], we have DISPLAYFORM10 where X y is the data random variable whose true label is y.

Equivalently, with probability at least 1 − δ, DISPLAYFORM11 where M, m are upper and lower bounds of random variable σ( yw T F (X y )), respectively.

Substitute FORMULA2 and FORMULA2 into FORMULA2 , with probability at least 1 − δ, DISPLAYFORM12 To estimate the upper and lower bounds M, m for σ( yw T F (X)), we use the Taylor's theorem: DISPLAYFORM13 and DISPLAYFORM14 given that the derivative of sigmoid function is bounded by 1 4 .

It follows that DISPLAYFORM15 Also notice that M, m are the bounds for sigmoid function, so their difference is at most 1.From derivations above, we can rewrite FORMULA2 as DISPLAYFORM16 and the lemma follows.

Proposition B.2.

The empirical mutual information I( π) shares the same unique (global) maximum with DISPLAYFORM17 Proof.

The empirical information I( π) is defined by DISPLAYFORM18 where the empirical joint probability is given by DISPLAYFORM19 It then follows that for any y ∈ C, DISPLAYFORM20 In binary case it means that DISPLAYFORM21 and the empirical mutual information can decomposed as DISPLAYFORM22 We differentiate (37) with respect to π 11 and π (−1)(−1) , and calculate the critical points over the domain [0,1 2 ] for both variables, which gives DISPLAYFORM23 Observe that (38) is a global minimum over [0, DISPLAYFORM24 Since this is the unique critical point where the derivative vanishes, the global maximums can only be obtained on the boundaries.

In particular, if we restrict ( π 11 , π (−1)(−1) ) on [ FORMULA3 is a strictly increasing function over π 11 , π (−1)(−1) and the unique global maximum is obtained at DISPLAYFORM25 DISPLAYFORM26 The proposition follows from the definition (34) of π y y that FORMULA3 is only approached when DISPLAYFORM27

We show in Proposition C.1 that the lower bound for the classification error is itself lower bounded by a constant, which is attained if F is invertible.

Although the performance of the model also depends on the classifier w, our bound claims that an invertible feature map F could provide a better environment for the classifier w to perform well.

Intuitively, invertibility preserves the information of the signal X as it flows through the neural network and reaches the classifier w; on the other hand, w potentially performs better on the input that preserves all information of the data compared to the one that doesn't.

Proposition C.1. (Fano's Inequality) The classification error is lower bounded as follows: DISPLAYFORM0 The lower bound satisfies DISPLAYFORM1 for all F , and the equality is attained if F is invertible.

Let Z = F (X) and the machinery of deep learning can be decribed by the following Markov Chain: DISPLAYFORM2 Lemma C.1 is a technical result that helps to prove Proposition C.1.

The information Z = F (X) has about the true labels Y is maximized when F is invertible, which is beneficial in the sense that the key information influential for classification can be well preserved.

Lemma C.1 (Chain Rule).

Given the Markov Chain assumption equation 42, we have DISPLAYFORM3 and the second equality is attained if F is invertible.

Proof.

We will only prove the second inequality and the first inequality follows by a similar argument.

Consider the decomposition DISPLAYFORM4 Similarly we obtain DISPLAYFORM5 equation FORMULA8 According to the Markov Chain setting, Y and Z are conditionally independent given X, hence I(Y ; Z|X) = 0; in addition, the mutual information I(X; Y |Z) is nonnegative.

It follows from (46) that DISPLAYFORM6 Next we present a lower bound for the classification error.

This lower bound is negatively related to the mutual information I(Y ; F (X)), and it attains its minimum if F is invertible.

Although the performance also depends on the classifier w, Proposition C.1 implies that an invertible feature map F allows more chances for the classifier w to perform well.

Proof of Proposition C.1.

Consider the random variable E defined as: DISPLAYFORM7 By the Chain Rule following from similar arguments presented in Lemma C.1, we have DISPLAYFORM8 Note that H(E|Y, Y ) = 0, since the value of E is determined given the knowledge of Y, Y .

It then follows that DISPLAYFORM9 On the other hand, Lemma C.1 shows that DISPLAYFORM10 which gives DISPLAYFORM11 Substitute it into (50) yields the result DISPLAYFORM12 As for the second statement, Lemma C.1 shows that DISPLAYFORM13 It then follows that, DISPLAYFORM14 where the equality is attained if F is invertible.

ResNet is designed to allow the model to "learn" the identity map easily.

Specifically, the input vector x and output vector y of a building block are related by DISPLAYFORM0 where the operator L could be a composition of activation functions, convolutions, dropout(Srivastava et al. FORMULA0 ) and batch normalization(Ioffe & Szegedy FORMULA0 ).

It's shown in Lemma D.1 below that if the operator norm |L| < 1, then L + I is theoretically guaranteed to have an inverse, which enables information preservation among intermediate layers.

In Figure 5 we experimentally verify that |L| < 1 for all building blocks during the training process.

In general, operations such as ReLU, pooling, drop-out are not invertible BID8 ; it's challenging to build a strictly invertible network (Jacobsen et al., 2018) .

From this point of view, the beauty of ResNet lies in the fact that it's guaranteed to be invertible regardless how L evolves during the training process, as long as |L| < 1.Although the usual design of ResNet does involve non-invertible components such as pooling, we argue that ResNet still has descent invertible property compared to the majority of other neural network designs.

We also experimentally verify that our regularization does not improve a very deep ResNet on its performance by a clear margin; we speculate that information will lose more as it goes deeper.

Proof.

It's well known that C 0,1 (U ) is a Banach space (Lax).

DISPLAYFORM1 Since |L| < 1, (57) is a Cauchy sequence and coverges in Banach space.

Convergence sequences can be multiplied termwise, it follows that DISPLAYFORM2 So B(I + L) = I. The other equality (I + L)B = I can be shown to hold in the same fashion.

Figure 5 : We measure the operator norm of L in each building block of ResNet-32 over 80k training steps.

It can be observed that, the operator norms are all bounded by 1 throughout the training process, which verifies the hypothesis made in Appendix D. We conlude that ResNet is invertible.

We implement all models using Tensorflow.

We modify the ResNet based on the code provided at https://github.com/tensorflow/models/tree/master/research/resnet.

We follow the same learning rate scheme proposed in the original code.

For the InvNet on MNIST, we train the network with initial learning rate 0.1, and decay it by 0.7 every 10 epochs.

For both InvNet and ResNet, we apply the 1 norm regularization every 30 iterations.

We observe that each digits have their specific entries with high value assigned to both weights and feature means.

An additional observation is that typically the feature entry with low mean also has low standard deviation, such entry rarely contributes to the logits for classification if the corresponding value of weight is also small.

We reproduce the results in Section 4.2 on i-RevNet (Jacobsen et al. FORMULA0 ).

The statistics and ROC curves of the features produced by i-RevNet and Reg-i-RevNet on CIFAR10 is similar to those presented in Section 4.2.

The histogram for values of feature entries of Digit 9 has a decaying shape but with a heavier tail compared to that of Gaussian with small variance.

The spasity of w depends on our choice of hyperparameters.

For example in Figure 4 we measure the sparsity of the learned w of RegResNet for CIFAR10: about 60% entries of w are precisely zero.

Our objective is DISPLAYFORM0 Note that (60) is composed of functions in the form a log a where a is the output of a softmax function on logits.

For simplicity we consider the binary case where a = 1 + e (61) Assume |w T F (x)| is large, if w T F (x) > 0 then the numerator decays exponentially with respect to |w T F (x)| and the denominator converges to 1; if wF (x) < 0 then the denonimator grows exponentially with respect to |w T F (x)| and dominates the numerator.

To conclude, for large |w T F (x)| the gradient is decaying to zero exponentially.

So the effect of punishing large logits |w T F (x)| by this objective is not clear as the gradient vanishes for large |w T F (x)|.

We also analyzed the general softmax functions for multi-labels and found they exhibit similar properties.

We propose to use a surrogate function w T F (X) to minimize this objective to make the regularization effect more clear.

We prove that our regularizer achieves the same goal compared to the mutual information objective but provides a better gradient for training.

@highlight

we propose a regularizer that improves the classification performance of neural networks

@highlight

the authors propose to train a model from a point of maximizing mutual information between the predictions and the true outputs, with a regularization term that minimizes irrelevant information while learning.

@highlight

Proposes to decompose the parameters into an invertible feature map F and a linear transformation w in the last layer to maximize mutual information I(Y, \hat{T}) while constraining irrelevant information