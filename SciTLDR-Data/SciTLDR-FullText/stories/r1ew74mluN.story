An adversarial feature learning (AFL) is a powerful framework to learn representations invariant to a nuisance attribute, which uses an adversarial game between a feature extractor and a categorical attribute classifier.

It theoretically sounds in term of it maximize conditional entropy between attribute and representation.

However, as shown in this paper, the AFL often causes unstable behavior that slows down the convergence.

We propose an {\em attribute perception matching} as an alternative approach, based on the reformulation of conditional entropy maximization as {\em pair-wise distribution matching}. Although the naive approach for realizing the pair-wise distribution matching requires the significantly large number of parameters, the proposed method requires the same number of parameters with AFL but has a better convergence property.

Experiments on both toy and real-world dataset prove that our proposed method converges to better invariant representation significantly faster than AFL.

How to learn representations invariant to nuisance attribute a is technical challenges raised in domain generalizaton BID1 BID9 BID4 BID8 , fair classification, privacy-protection BID2 BID5 , and many other area.

Assume that we are given a training dataset made of pairs S = (x i , y i , a i ) DISPLAYFORM0 , where x is an observation, y is a target of x, and a is a corresponding intrinsic attribute of K-way categorical variable A. The goal of invariant representation learning is to obtain an encoder E that reduces information about attribute a while maintaining information about y.

An adversarial game between a feature extractor and an attribute classifier, called adversarial feature learning BID11 , is a powerful framework for this purpose.

The key of AFL is to measure the invariance by leveraging the discriminative power of neural network beyond the pre-defined metric such as l 2 distance or maximum mean discrepancy.

That is, if the external network (also referred to as a discriminator) can predict a from z = E(x), AFL regards z to have considerable information about a. Formally, the AFL solves the following optimization problem: DISPLAYFORM1 where q M and q D is the conditional probability that M and D gives a correct estimation of y and a respectively.

As BID11 explained, this alternating procedure can be regarded as a way to maximize the conditional entropy H(A|Z) = a∈A,z∈Z −p(a, z) log p(a|z), where A and Z is a support set of a and z. BID11 also showed that the min-max game has an equilibrium, in which E maximize the conditional entropy H(A|Z).

It has been show superior performance in fairclassification, privacy-protection, and domain generalization tasks BID3 BID2 BID11 BID5 , compared to the predifined metric approaches BID12 BID7 BID8 .Despite the theoretical justifications, the above min-max formulation is suspicious for several practical issues.

Namely, the gradient from the discriminator vanishes if the discriminator sufficiently trained since E[log q D (a|z=E(x))] is small then.

Besides, in mathematical level, it only keeps away representations from the non-desired point where we can predict a label correctly, but not ensure that it approaches the desired invariant point.

Please also refer FIG1 for visualization of the instability.

Note that, Generative Adversarial Networks community, which utilize similar formulation to generate realistic images, evade similar issues by the incorporating alternative objectives, such as the Wasserstein distance BID0 .

However, the Wasserstein distance is defined over two distributions and applying to our setting (consisting of multiple distributions) is not trivial.

This paper holds the following contributions to the invariant feature learning problem.

First, we empirically show that AFL is suffered from practical issues that significantly slow down the convergence.

We then reformulate the optimization problem of AFL as pair-wise distribution matching and derive parameter practical realization of pairwise distribution matching while inheriting the merit of AFL that leveraging the discriminative power to measure the invariance.

It is worth mentioning that the reformulation enable us to use Wasserstein metric in theory, however, it is still computationally infeasible in practice because a naive way to calculate the Wasserstein distance between all the pair of the distributions requires O(K 2 ) discriminators, where K = |A|, which raise computational issues both in terms of parameter size and forward/backward time.

Finally, we empirically validate the superior performance of our proposed method on both artificial dataset and real-world datasets.2 CONVERGENCE ISSUES OF AFL FIG1 -(a-e) visualize a behavior of AFL optimization on synthesized data.

Each figure corresponds to the different timestep of the alternating optimization.

The dataset consists of samples from three Gaussian distributions with different means ([sin( DISPLAYFORM2 , for i ∈ 1, 2, 3, respectively) and the same variance, assuming that each distribution corresponds to different attributes.

In each figure, dots represent the data point, color represents the attribute (domain id), and the contour plot represents the discriminator's decision boundary.

A float value on the top of figures is the negative log-likelihood (NLL) of the dataset measured by the discriminator D (the multi-layer perceptron with 100 hidden units followed by a ReLU activation).

Similarly, a float value in parenthesis on the top of figures is an NLL of a post-hoc classifier D eval that have the same architecture as D. To be more specific, we first train the discriminator 100 times with 128 batch size and train D and E iteratively with stochastic gradient descent with learning rate=0.1.

FIG1 -(f,g) shows the gradient vector fields of different time steps for a = blue, where the arrow represents the direction of the gradient, and the norm represents its magnitude.

For simplicity, we only show the vector fields of a = blue, but the observations are quite similar for the other a.

The figure reveals two practical issues in AFL optimization.

(1) The distribution alignment is initially quite slow (compare with the behavior of the proposed method shown in Figure 2 ).

This is because the gradient is small when the discriminator correctly distinguishes a a. (2) AFL behavior is unstable.

The distributions somewhat align after 40 steps (given 0.683 NLL with the post-hoc classifier), but it is catastrophically failed five steps later because the discriminator did not capture the true conditional entropy (implied by the mostly similar counterplot of D) and therefore gave a false gradient as shown in (f) and (g).

The intuitive reason for this phenomenon is that AFLs loss essentially tries to pull a distribution apart from the non-desired point, i.e., the point where we can correctly predict the label.

The problem of AFL is that it only keeps away a distribution from the non-desired point, but not ensure it approaches the desired invariant point.

After several steps, D starts to follow the change of the distribution (as shown in FIG1 .

The instability of the AFL also appears in the significant gap between the NLL of the D and D eval .

Note that the second issue may be alleviated if D has a sufficiently large capacity and is trained many times at each iteration.

However, this is not a realistic assumption since it is fair to say that real datasets are more complicated than this toy situations, making it more challenging to find the supremum.

The next question we must answer is how to maximize conditional entropy while avoiding the issues mentioned above.

Assume that A is a categorical random variable drawn from a uniform discrete distribution, and denote that support set of a and z as A and Z; then, the following theorem holds: Theorem 1.

The maximum conditional entropy H(A|Z) is − log 1 K , and H(A|Z) is maximized if and only if p(z|a=i) = p(z|a=j) for all i = j ∈ 1, · · · , K and z ∈ Z.The proof is followed in the appendix.

One implication of the theorem is, the empirical measurements of conditional entropy should be boudned.

It suggests another problematic point of AFL objective since it is not lower bounded.

Another implication is that this theorem permits us to rethink the problem of the conditional entropy maximization as a problem of aligning all pairs of distributions p(z|a = i) and p(z|a = j).

This reformulation is significant since we can now use any conventional measurement of two distributions.

However, the naive approach requires to match all pairs of the distributions (= K C 2 ), which would be cumbersome to compute.

We, therefore, propose a computationally efficient way to realize pairwise distribution matching, moving attribute perception matching.

As with AFL, APM is based on the alternating training of attribute classifier D and feature extractor E, but APM deceives D differently.

Formally, we propose the following APM matching objective: DISPLAYFORM0 where k D is some distance function defined over either a hidden representation of the discriminator D or output probability q D (a|E(x)) itself and C aj is the moving average of the centroid for attribute a j in the attribute perception.

Although there are many valid choices for k D (., .), including the simple l 2 norm, we primarily use the well-known Kullback-Leibler (KL) divergence.

We initialized C 0 aj via computing the centroids using all training data points.

The key of the proposal is that it only requires the same number of parameters as AFL, but it ensures that representations of an attribute approach to the representations of the other attributes.

Also, it inherits the merit of AFL that leveraging the discriminative power to measure the invariance.

Figure 2-(a-c) shows the behavior of the APM under the exact same experimental settings shown in FIG1 .

The proposed method maximizes the conditional entropy significantly faster than AFL: 1.061 after only five iterations, but AFL gives 0.683 after 40 iterations.

Note that the value match the theoretical maximum value of the conditional entropy (log Another merit of the proposed method is that we can enforce semantic alignment with a simple modification.

Individually, semantic alignment can be carried out by merely computing the centroids for each (attribute, label) tuple and aligning the perceptions of {x, y, a} between only centroids of the same label y = y but different attributes a = a. Although this modification does not help to maximize the conditional entropy, it prevents performance degradation of predicting y. Since most applications of invariant feature learning require to keep a information about y we use this modification for all the later-described experiments.

We use three domain generalization tasks and two user-anonymization tasks.

MNISTR BID4 and PACS BID6 ) is a well known image-based datasets designed for the purpose of domain generalization.

We also test the methods of noise-robust speech recognition scenarios using Google Speech Command Dataset (Speech).

Regarding user-anonymization, we use two user-anonymization tasks on the data of wearables, OppG and USC BID5 .

The neural networks require to learn representations that help activity classification and at the same time, prevent to access the information about users (userID).

As baselines, we use (1) A CNN trained on the aggregation of data from all source domains.

(2) AFL BID11 , which was explained in Section 3.1.

FORMULA4 RevGrad ( BID3 ) is a slightly modified version of AFL, which uses the gradient reversal layer to train all the networks.

(4) CrossGrad BID10 ) is regarded as a state-of-the-art method in domain generalization tasks.

(5) Activation Matching (AM), which trains the encoder with the regularization of the l 2 distance on a feature space.

(6) APM is our proposal.

For all datasets and methods, we used RMSprop for each optimization.

For all datasets except PACS, we set the learning rate to 0.001 and the batch size to 128.

For PACS, we set the learning rate to 5e − 5 and the batch size to 64.

For a fair comparison, hyperparameters were tuned on a validation set for each baseline.

For the adversarial-training-based method, we optimized weighting parameter λ from {0.001, 0.01, 0.1, 1.0}, except for MNISTR, for which it was optimized from {0.01, 0.1, 1.0, 2.0}. The value of α for CrossGrad was selected from {0.1, 0.25, 0.5, 0.75, 0.9}. We set the decay rate γ to 0.7 for all experiments.

In all the experiments, we selected the data of one or several domains for the test set and used the data of a disjoint domain as the training/validation data.

Specifically, we split the data of the disjoint domain into groupings of 80% and 20%.

We denote the test domain by a suffix (e.g., MNISTR-M0).

We measured the label classification accuracy and the level of invariance.

We measured the level of invariance by training a post-hoc classifier f eva following previous studies BID11 ; BID5 .

FIG2 compares the performance the invariance of representations.

For each method, we used the largest weighting parameter λ on the condition that label classification accuracy did not significantly decrease.

The results show that the proposed method stably achieves better invariant representations except for speech dataset.

For example, APM achieved 20% of the A-Acc in MNISTR, which is nearly perfect invariance as there are five domains in the validation data, while RevGrad and AFL achieved 30% at best.

On speech dataset, the performance of AFL and APM is mostly similar.

These results confirm that the proposed method stably achieves more invariant representation compared with AFL.

TAB0 summarizes the methods' classification performance on three different datasets: MNISTR, Speech, and PACS.

The leftmost column of each table represents the test domain.

We report the mean accuracy.

We can make the following observations.

(1) APM demonstrates the best or comparable performance on all three datasets, including CrossGrad, which is regarded as the state-of-the-art method in domain generalization tasks.

(2) RevGrad and AFL often fail to improve performance even when compared with a standard CNN.

These results suggest that the previous adversarial-training-based method suffered from the lack of semantic alignment when applied to domain generalization.

FORMULA4 The Wilcoxon rank sum test shows that APM is statistically better than CNN, RevGrad, and AFL with p < 0.01, and than CrossGrad with p < 0.05.

This paper proposes a new approach to incorporating desired invariance to representations learning, based on the observations that the current state-of-the-art AFL has practical issues.

Empirical results on both toy and real-world datasets support the stable performance of the proposed method to learn invariant features and superior performance on domain generalization tasks.

A PROOF OF THE THEOREM 1Proof.

Using the Lagrange multiplier method, the derivative of DISPLAYFORM0 is equal to zero for the maximum entropy H(A|Z).

Solving the simultaneous equations, we can say p(a=1|z) = p(a=2|z) = · · · = p(a=K|z) = 1 K for all z ∈ Z when the conditional entropy is maximized, and based on the definition, the conditional entropy become − log DISPLAYFORM1 holds ∀i = j ∈ A and z ∈ Z.

@highlight

This paper proposes a new approach to incorporating desired invariance to representations learning, based on the observations that the current state-of-the-art AFL has practical issues.