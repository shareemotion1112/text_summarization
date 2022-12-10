Recent research about margin theory has proved that maximizing the minimum margin like support vector machines does not necessarily lead to better performance, and instead, it is crucial to optimize the margin distribution.

In the meantime, margin theory has been used to explain the empirical success of deep network in recent studies.

In this paper, we present ODN (the Optimal margin Distribution Network), a network which embeds a loss function in regard to the optimal margin distribution.

We give a theoretical analysis for our method using the PAC-Bayesian framework, which confirms the significance of the margin distribution for classification within the framework of deep networks.

In addition, empirical results show that the ODN model always outperforms the baseline cross-entropy loss model consistently across different regularization situations.

And our ODN model also outperforms the cross-entropy loss (Xent), hinge loss and soft hinge loss model in generalization task through limited training data.

In the history of machine learning research, the large margin principle has played an important role in theoretical analysis of generalization ability, meanwhile, it also achieves remarkable practical results for classification BID3 and regression problems BID4 .

More than that, this powerful principle has been used to explain the empirical success of deep neural network.

BID1 and present a margin-based multi-class generalization bound for neural networks that scales with their margin-normalized spectral complexity using two different proving tools.

Moreover, BID0 proposes a stronger generalization bounds for deep networks via a compression approach, which are orders of magnitude better in practice.

As for margin theory, BID17 first introduces it to explain the phenomenon that AdaBoost seems resistant to overfitting problem.

Two years later, BID2 indicates that the minimum margin is important to achieve a good performance.

However, BID16 conjectures that the margin distribution, rather than the minimum margin, plays a key role in being empirically resistant to overfitting problem; this has been finally proved by BID6 .

In order to restrict the complexity of hypothesis space suitably, a possible way is to design a classifier to obtain optimal margin distribution.

BID6 proves that, to attain the optimal margin distribution, it is crucial to consider not only the margin mean but also the margin variance.

Inspired by this idea, Zhang & Zhou (2016) proposes the optimal margin distribution machine (ODM) for binary classification, which optimizes the margin distribution through the first-and second-order statistics, i.e., maximizing the margin mean and minimizing the margin variance simultaneously.

To expand this method to the multi-class classification problem, Zhang & Zhou (2017) presents a multi-class version of ODM.Based on these recent works, we consider the expansion of the optimal margin distribution principle on deep neural networks.

In this paper, we propose an optimal margin distribution loss for convolution neural networks, which is not only maximizing the margin mean but also minimizing the margin variance as ODM does.

Moreover, we use the PAC-Bayesian framework to derive a novel generalization bound based on margin distribution.

Comparing to the spectrally-normalized margin bounds of BID1 and , our generalization bound shows that we can restrict the capacity of the model by setting an appropriate ratio between the first-order statistic and the second-order statistic rather than trying to control the whole product of the spectral norms of each layer.

And we empirically evaluate our loss function on deep network across different datasets and model structures.

Specifically, we consider the performance of these models in generalization task through limited training data.

Recently, many researchers try to explain the experimental success of deep neural network.

One of the research direction is to explain why the deep learning does not have serious overfitting problem.

Although several common techniques, such as dropout (Srivastava et al., 2014) , batch normalization BID7 , and weight decay BID10 , do improve the generalization performance of the over-parameterized deep models, these techniques do not have a solid theoretical foundation to explain the corresponding effects.

As for our optimal margin distribution loss, it has a generalization bound to prove that we can restrict the complexity of hypothesis space reasonably through searching appropriate statistics dependent on data distribution.

In experimental section, we compare our optimal margin distribution loss with the baseline cross-entropy loss under different regularization methods.

Consider the classification problem with input domain X = {x| x ∈ R n } and output domain Y = {1, . . .

, k}, we denote a labeled sample as z ∈ (X , Y).

Suppose we use a network generating a prediction score for the input vector x ∈ X to class i, through a function f i : X → R, for i = 1, . . .

, k. The predicted label is chosen by the class with maximal score, i.e. h(x) = arg max i f i (x).Define the decision boundary of each class pair {i, j} as: DISPLAYFORM0 Constructed on this definition, the margin distance of a sample point x to the decision boundary D i,j is defined by the smallest translation of the sample point to establish the equation as: DISPLAYFORM1 In order to approximate the margin distance in the nonlinear situation, BID5 has offered a linearizing definition: DISPLAYFORM2 Naturally, this pairwise margin distance leads us to the following definition of the margin for a labeled sample z = (x, y): DISPLAYFORM3 Therefore, the defined classifier h misclassifies (x, y) if and only if the margin is negative.

Given a hypothesis space H S of functions mapping X to Y, which can be learned by the fixed deep neural network through the training set S, our purpose is to find a way to learn a decision function h ∈ H S such that the generalization error DISPLAYFORM4 In this work, we introduce a type of margin loss, and connect it to deep neural networks.

The origin loss function has been specially adapted for the difference between deep learning models and linear models by us as following definition: DISPLAYFORM5 where r is the margin mean, θ is the margin variance and µ is a parameter to trade off two different kinds of deviation (keeping the balance on both sides of the margin mean).

In the Appendix A, we explain the reason why we use these three hyper-parameters to construct such a optimal margin distribution loss function.

FIG0 shows, equation 1 will produce a linear loss decreasing progressively when the margins of sample points satisfy γ h ≤ r − θ and a square loss increasing progressively when the margins satisfy γ h ≥ r + θ.

Therefore, our margin loss function will enforce the tie which has zero loss to contain the sample points as many as possible.

So the parameters of the classifier will be determined not only by the samples that are close to the decision boundary but also by the samples that are away from the decision boundary.

In other words, our loss function is aimed at finding a decision boundary which is determined by the whole sample margin distribution, instead of the minority samples that have minimum margins.

To verify superiority of the optimal margin distribution network, our paper verifies it both theoretically and empirically.

To present a new margin bound for our optimal margin distribution loss, some notations are needed.

Consider that the convolution neural networks can be regarded as a special structure of the fully connected neural networks, we simplify the definition of the deep networks.

Let f w (x) : X → R k be the function learned by a L-layer feed-forward network for the classification task with parameters DISPLAYFORM0 , here φ is the ReLU activation function.

Let f i w denote the output of layer i before activation and ρ be an upper bound on the number of output units in each layer.

Recursively, we can redefine the deep network: DISPLAYFORM1 w (x)).

Let · F , · 1 and · 2 denote the Frobenius norm, the element-wise 1 norm and the spectral norm respectively.

In order to facilitate the theoretical derivation of our formula, we simplify the definition of the loss function: DISPLAYFORM2 Specially, define the L 0 as r = θ and θ → ∞, actually equal to the 0-1 loss.

And let L r,θ (f w ) be the empirical estimate of the optimal margin distribution loss.

So we will denote the expected risk and the empirical risk as L 0 (f w ) and L 0 (f w ), which are bounded between 0 and 1.In the PAC-Bayesian framework, one expresses the prior knowledge by defining a prior distribution over the hypothesis class.

Following the Bayesian reasoning approach, the output of the learning algorithm is not necessarily a single hypothesis.

Instead, the learning process defines a posterior probability over H, which we denote by Q. In the context of a supervised learning problem, where H contains functions from X to Y, one can think of Q as defining a randomized prediction rule.

We consider the distribution Q which is learned from the training data of form f w+u , where u is a random variable whose distribution may also depend on the training data.

Let P be a prior distribution over H that is independent of the training data, the PAC-Bayesian theorem states that with possibility at least 1 − δ over the choice of an i.i.d.

training set S = {z 1 , ..., z m } sampled according to (X , Y), for all distributions Q over H (even such that depend on S), we have BID14 : DISPLAYFORM3 Note that the left side of the inequality is based on f w+u .

To derive an expected risk bound L 0 (f w ) for a single predictor f w , we have to relate this PAC-Bayesian bound to the expected perturbed loss just like derive the Lemma 1 in their paper.

Based on the inequality 2, we introduce a perturbed restriction which is related to the margin distribution (the margin mean r and margin variance θ): Lemma 1.

Let f w (x) : X → R k be any predictor with parameters w, and P be any distribution on the parameters that is independent of the training data.

Then, for any r > θ > 0, δ > 0, with probability at least 1 − δ over the training set of size m, for any w, and any random perturbation u s.t.

DISPLAYFORM4 , we have: DISPLAYFORM5 The margin variance information does not change the conclusion of the perturbed restriction, the proof of this lemma is similar to Lemma 1 in .In order to bound the change caused by perturbation, we have to bring in three definitions that are used to formalize error-resilience in BID0 as follows: Definition 1. (Layer Cushion).

The layer cushion of layer i is defined to be largest number µ i such that for any x ∈ S: DISPLAYFORM6 Intuitively, cushion considers how much smaller the output DISPLAYFORM7 w (x) 2 .

However, for nonlinear operators the definition of error resilience is less clean.

Let's denote DISPLAYFORM8 the operator corresponding to the portion of the deep network from layer i to layer j, and by J i,j its Jacobian.

If infinitesimal noise is injected before level i then M i,j passes it like J i,j , a linear operator.

When the noise is small but not infinitesimal then one hopes that we can still capture the local linear approximation of the nonlinear operator M by define Interlayer Cushion: Definition 2. (Interlayer Cushion).

For any two layers i < j, we define the interlayer cushion µ i,j , as the largest number such that for any x ∈ S: DISPLAYFORM9 Furthermore, for any layer i we define the minimal interlayer cushion as µ i→ = min i≤j≤L µ i,j = min{ DISPLAYFORM10 The next condition qualifies a common appearance: if the input to the activations is well-distributed and the activations do not correlate with the magnitude of the input, then one would expect that on average, the effect of applying activations at any layer is to decrease the norm of the pre-activation vector by at most some small constant factor.

Definition 3. (Activation Contraction).

The activation contraction c is defined as the smallest number such that for any layer i and any x ∈ X : DISPLAYFORM11 To guarantee that the perturbation of the random variable u will not cause a large change on the output with high possibility, we need a perturbation bound to relate the change of output to the structure of the network and the prior distribution P over H. Fortunately, Neyshabur et al. FORMULA5 proved a restriction on the change of the output by norms of the parameter weights.

In the following lemma, we preset our hyper-parameters r and θ, s.t.

the parameter weights w ∈ H satisfying f w (x) 2 ≤ r + θ, when fixing W L 2 = 1.

Thus, we can bound this change in terms of the spectral norm of the layer and the presetting hyper-parameters:Lemma 2.

For any L > 0, let f w : X → R k be a L-layer network.

Then for any w ∈ H satisfying f w (x) 2 ≤ r + θ, and x ∈ X , and any perturbation DISPLAYFORM12 the change of the output of the network can be bounded as follows: DISPLAYFORM13 The proof of this lemma is given in Appendix B. Eventually, we utilize all above bounding lemmas and the error-resilience definitions to derive the following new margin based generalization bound for our Optimal margin Distribution Network.

Theorem 1. (Generalization Bound).

For any L, ρ > 0, let f w : X → R k be a L-layer feed-forward network with ReLU activations.

Then, for any δ > 0, r > θ > 0, with probability ≥ 1 − δ over a training set of size m, for any w, we have: DISPLAYFORM14 The proof of this theorem is given in Appendix B.Remark.

Comparing with the spectral complexity in Bartlett et al. FORMULA5 : DISPLAYFORM15 which is dominated by the product of spectral norms across all, our margin bound is relevant to r, θ dependent on the margin distribution and µ i and µ i→ dependent on the network structure.

The product value in equation 3 is extremely large and is hard to control it, but the parameter in our generalization bound is easy to restrict.

Explicitly, the factor consisted of hyper-parameters DISPLAYFORM16 is a monotonicity increasing function with regard to the ratio θ r ∈ [0, 1).

Under the assumption of separability, we can come to the conclusion that maller θ and larger r make the complexity smaller.

Searching a suitable value of r and θ for the specific data distribution will lead us to a better generalization performance.

In this section, we empirically evaluate the effectiveness of our optimal margin distribution loss on generalization tasks, comparing it with three other loss functions: cross-entropy loss (Xent), hinge loss, and soft hinge loss.

We first compare them under limited training data situation, using only part of the MNIST dataset BID11 to train and evaluate the models deploying the four different losses, with the used data ratio ranging from 0.125% to 100%.

Similar experiments are also performed on the legend CIFAR-10 dataset BID8 ).

Then we compare them under different regularization situations, investigating the combination of optimal margin distribution loss with dropout and batch normalization.

Finally, we visualize and compare the features learned by the deep learning model with the four lose functions as well as the margin distribution from those models.

Here we introduce three commonly used loss functions in deep learning for comparison in the experimental section:Cross-entropy Loss (Xent): DISPLAYFORM0 Hinge Loss: DISPLAYFORM1 where γ 0 is a hyper-parameter to control the minimum margin as support vector machine does.

Soft Hinge Loss: DISPLAYFORM2 where k is the number of classes.

Regarding the deep models, we use the following combination of datasets and models: a simple deep convolutional network for MNIST, original Alexnet (Krizhevsky et al., 2012) for CIFAR-10.In terms of the implementation of optimal margin distribution loss, as shown in Section 2, there is a gradient term in the loss itself, which can make the computation expensive.

To reduce computational cost as Elsayed et al. FORMULA5 do, in the backpropagation step we considered the gradient term ∇ x f y (x) − ∇ x max i =y f i (x) 2 as a constant, so that we recomputed the value of ∇ x f y (x) − ∇ x max i =y f i (x) 2 at every forward propagation step.

Furthermore, since the denominator item could be too small, which would cause numerical problem, we added an with small value to the denominator so that clip the loss at some threshold.

For special hyperparameters, including the margin mean parameter and margin variance parameter for the ODN model, and margin parameter for hinge loss model, we performed hyperparameter searching.

We held out 5000 samples of the training set as a validation set, and used the remaining samples to train models with different special hyperparameters values, on both the MNIST dataset and the CIFAR-10 dataset.

As for the common hyperparameters, such as, learning rate and momentum, we set them as the default commonly used values in Pytorch for all the models.

We chose batch stochastic gradient descent as the optimizer.

Evaluated on the testing dataset, the baseline cross-entropy model achieves a test accuracy of 99.09%; the hinge loss model achieves 98.95% on MNIST dataset; the soft-hinge loss model achieves 99.14% and the ODN model achieves 99.16%.

On the CIFAR-10 dataset, the baseline cross-entropy model trained on the remaining training samples achieves a test accuracy of 83.51%; the hinge loss model achieves 82.15%; the soft-hinge loss model achieves 81.96% and the ODN model achieves 84.61%.

It is well-known that deep learning method is very data-hungry, which means that if the training data size decreases, the model's performance can decrease significantly.

In reality, this disadvantage of deep learning method can restrict its application seriously since sufficient amount of data is not always available.

On the other hand, one of the desirable property of optimal margin distribution loss based models is that it can generalize well even when the training data is insufficient because the optimal margin distribution loss can restrict the complexity of the hypothesis space suitably.

To evaluate the performance of optimal margin distribution loss based models under insufficient training data setting, we randomly chose some fraction of the training set, in particular, from 100% of the training samples to 0.125% on the MNIST dataset, and from 100% of the training samples to 0.5% on the CIFAR-10 dataset, and train the models accordingly.

In FIG1 , we show the test accuracies of cross-entropy, hinge, soft hinge, and optimal margin distribution loss based models trained on different fractions of the MNIST and CIFAR-10 dataset.

As shown in the figure, the test accuracies of all these four models increase as the fraction of training samples increases.

Obviously, the ODN models proposed by our paper outperform all the other models constantly across different datasets and different fractions.

Furthermore, the less training data there are, the larger performance gain the ODN model can have.

On the MNIST dataset, the optimal margin distribution loss based model outperforms cross-entropy loss model by around 4.95%, hinge loss model by around 6.84% and soft-hinge loss model by around 3.03% on the smallest training set which contains only 0.125% of the whole training samples.

Similarly, The ODN model outperforms cross-entropy loss model by around 9.9%, hinge loss model by around 10.1%, and soft hinge loss model by 13.4% on the smallest CIFAR-10 dataset which contains only 0.5% of the whole training samples.

We also compared our optimal margin distribution loss with the baseline cross-entropy loss under different regularization methods and different amounts of training data, whose results are shown in TAB0 .

As suggested by TAB0 , our loss can outperform the baseline loss consistently across different situations, no matter whether dropout, batch normalization or all the CIFAR-10 dataset are used or not.

Specifically, when the size scale of training samples is small (5% fraction of the CIFAR-10 training set), the advantage of our optimal margin distribution loss is more significant.

Moreover, our optimal margin distribution loss can cooperate with batch normalization and dropout, achieving the best performance in Table.

1, which is shown in bold red text.

Unlike dropout and batch normalization which are lack of solid theory ground, our optimal margin distribution loss has the margin bound , which guides us to find the suitable ratio θ r to restrict the capacity of models and alleviate the overfitting problem efficiently.

Since the performance of the ODN models is excellent, we hope to see that the distributions of data in the learned feature space (the last hidden layer) are consistent with the generalization results.

In this experiment, we use t-SNE method to visualize the data distribution on the last hidden layer for training samples and test samples.

Fig. 3 and Fig. 4 plots the 2-dimension embedding image on limited MNIST and CIFAR-10 dataset, which is only 1% of the whole training samples.

t-SNE BID12 ) is a tool to visualize high-dimensional data.

It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

Consistently, we can find that the result of our ODN model is better than all the others, the distribution of the samples which has the same label is more compact.

To quantify the degree of compactness of the distribution, we perform a variance decomposition on the data in the embedding space.

By comparing the ratio of the intra-class variance to the inter-class variance in TAB1 , we can know that our optimal margin distribution loss alway attain the most compact distribution in these four loss functions.

Moreover, the visualization result is consistent with the margin distribution of these four models in FIG3 , which means getting an optimal margin distribution is helpful to deriving a good learned features space.

And that representation features space can further alleviate the overfitting problem of deep learning.

Hence, the optimal margin distribution loss function can significantly outperforms the other loss functions in generalization task through limited training data.

FIG3 plots the kernel density estimates of margin distribution producted by cross-entropy loss, hinge loss, soft hinge loss and ODN models on dataset MNIST.

As can be seen, our ODN model derives a large margin mean with a smallest margin variance in all these four models.

By calculating the value of ratio between the margin mean and the margin standard deviation, we know that the ratio in our ODN model is 3.20 which is significantly larger than 2.38 in the cross-entropy loss, 2.35 in the hinge loss and 2.63 in the soft hinge loss.

The distribution of our model becomes more "sharper", which prevents the instance with small margin, so our method can still perform well as the training data is limited, which is also consistent with the result in FIG1 .

Recent studies disclose that maximizing the minimum margin for decision boundary does not necessarily lead to better generalization performance, and instead, it is crucial to optimize the margin distribution.

However, the influence of margin distribution for deep networks still remains undiscussed.

We propose ODN model trying to design a loss function which aims to control the ratio between the margin mean and the margin variance.

Moreover, we present a theoretical analysis for our method, which confirms the significance of margin distribution in generalization performance.

As for experiments, the results validate the superiority of our method in limited data problem.

And our optimal margin distribution loss function can cooperate with batch normalization and dropout, achieving a better generalization performance.

We are grateful to Yu Li and Kangle Zhao for discussions and helpful feedback on the manuscript.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.

Dropout: a simple way to prevent neural networks from overfitting.

Inspired by the optimal margin distribution principle, Zhang & Zhou (2017) propose the multi-class optimal margin distribution machine, which characterizes the margin distribution according to the first-and second-order statistics.

Specially, letγ denote the margin mean, and the optimal margin distribution machine can be formulated as: DISPLAYFORM0 where Ω(w) is the regularization term to penalize the norm of the weights, η and λ are trading-off parameters, ξ i and are the deviation of the margin γ h (x i , y i ) to the margin mean.

It is evident that DISPLAYFORM1 )/m is exactly the margin mean.

In the linear situation, scaling w does not affect the final classification results such as SVM, the margin mean can be normalized as 1, then the deviation of the margin of (x i , y i ) to the margin mean is |γ h (x i , y i ) − 1|, and the formula can be reconstruct as: DISPLAYFORM2 where µ ∈ (0, 1] is parameter to trade off two different kinds of deviation (keeping the balance on both sides of the margin mean).

θ ∈ [0, 1) is a parameter of the zero loss band, which can control the number of support vectors.

In other words, θ is a parameter to control the margin variance, while the data which is out of this zero loss band will be used to update the weights to minimize the loss.

For this reason, we simply regard it as the margin variance.

However, under the non-linear setting in our paper, we can not directly linearly normalize the margin mean to the value 1.

So we assume that the normalized margin mean is r, then the optimization target can be reformulated as: DISPLAYFORM3 In our paper, we use the linear approximation BID5 to normalize the magnitude of the norm of weights, so we can just transform this optimization target to a loss function as: DISPLAYFORM4 There is always some noise in the actual data, when deep network try to fit these data, the performance of model get worse with a larger margin variance.

So we hope the larger side of the margin mean has a larger loss, which can effectively control the noise-fitting ability of models.

That is why we adapt the smaller side of the margin mean to hinge form as: DISPLAYFORM5 Proof.

of Theorem 1.The proof involves chiefly two steps.

In the first step we bound the maximum value of perturbation of parameters to satisfied the condition that the change of output restricted by hyper-parameter of margin r, using Lemma 2.

In the second step we proof the final margin generalization bound through Lemma 1 with the value of KL term calculated based on the bound in the first step.

DISPLAYFORM6 and consider a network structured by normalized weights W i = β Wi 2 W i .

Due to the homogeneity of the ReLU, we have that for feedforward networks with ReLU activations f w = f w , so the empirical and expected loss is the same for w and w. Furthermore, we can also get that DISPLAYFORM7 .

Hence, we can just assume that the spectral norm is equal across the layers, i.e. for any layer i, W i 2 = β.

When we choose the distribution of the prior P to be N (0, σI), i.e. u ∼ N (0, σI), the problem is that we will set the parameter σ according to β, which can not depend on the learned predictor w or its norm.

proposed a method that can avoid this block: they set σ based on an approximation β on a pre-determined grid.

By formalizing this method, we can establish the generalization bound for all w for which |c 0 β − β| ≤ 1 L β, while given a constant c, and ensuring that each relevant value of cβ is covered by some β on the grid, i.e. Since u ∼ N (0, σ 2 I), we get the following bound for the spectral norm of U i according to the matrix extension of Hoeffding's inequalities (Tropp, 2012; BID13 : DISPLAYFORM8 Taking the union bound over layers, with probability ≥ 1 2 , the spectral norm of each layer perturbation U i is bounded by σ 2ρ ln(4Lρ).

Plugging this into Lemma 2 we have that with probability ≥ from the above inequality.

Naturally, we can calculate the KL-diversity in Lemma 1 with the chosen distributions for P ∼ N (0, σ 2 I).

DISPLAYFORM9 Hence, for anyβ, with probability ≥ 1 − δ and for all w such that, |β − β| ≤ 1 L β, we have: This proof method based on PAC-Bayesian framework has been raised by , we use this convenient tool for proofing generalization bound with our loss function which can obtain the optimal margin distribution.

DISPLAYFORM10

<|TLDR|>

@highlight

This paper presents a deep neural network embedding a loss function in regard to the optimal margin distribution, which alleviates the overfitting problem theoretically and empirically.

@highlight

Presents a PAC-Bayesian bound for a margin loss