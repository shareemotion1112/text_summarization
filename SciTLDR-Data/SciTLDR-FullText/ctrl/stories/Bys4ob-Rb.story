While neural networks have achieved high accuracy on standard image classification benchmarks, their accuracy drops to nearly zero in the presence of small adversarial perturbations to test inputs.

Defenses based on regularization and adversarial training have been proposed, but often followed by new, stronger attacks that defeat these defenses.

Can we somehow end this arms race?

In this work, we study this problem for neural networks with one hidden layer.

We first propose a method based on a semidefinite relaxation that outputs a certificate that for a given network and test input, no attack can force the error to exceed a certain value.

Second, as this certificate is differentiable, we jointly optimize it with the network parameters, providing an adaptive regularizer that encourages robustness against all attacks.

On MNIST, our approach produces a network and a certificate that no that perturbs each pixel by at most $\epsilon = 0.1$ can cause more than $35\%$ test error.

Despite the impressive (and sometimes even superhuman) accuracies of machine learning on diverse tasks such as object recognition BID18 , speech recognition BID55 , and playing Go BID46 , classifiers still fail catastrophically in the presence of small imperceptible but adversarial perturbations BID50 BID17 BID25 .

In addition to being an intriguing phenonemon, the existence of such "adversarial examples" exposes a serious vulnerability in current ML systems BID13 BID45 .

While formally defining an "imperceptible" perturbation is difficult, a commonly-used proxy is perturbations that are bounded in ∞ -norm BID17 BID31 BID53 ; we focus on this attack model in this paper, as even for this proxy it is not known how to construct high-performing image classifiers that are robust to perturbations.

While a proposed defense (classifier) is often empirically shown to be successful against the set of attacks known at the time, new stronger attacks are subsequently discovered that render the defense useless.

For example, defensive distillation BID42 and adversarial training against the Fast Gradient Sign Method BID17 were two defenses that were later shown to be ineffective against stronger attacks BID53 .

In order to break this arms race between attackers and defenders, we need to come up with defenses that are successful against all attacks within a certain class.

However, even computing the worst-case error for a given network against all adversarial perturbations in an ∞ -ball is computationally intractable.

One common approximation is to replace the worst-case loss with the loss from a given heuristic attack strategy, such as the Fast Gradient Sign Method BID17 or more powerful iterative methods BID7 BID31 .

Adversarial training minimizes the loss with respect to these heuristics.

However, this essentially minimizes a lower bound on the worst-case loss, which is problematic since points where the bound is loose have disproportionately lower objective values, which could lure and mislead an optimizer.

Indeed, while adversarial training often provides robustness against a specific attack, it often fails to generalize to new attacks, as described above.

Another approach is to compute the worst-case perturbation exactly using discrete optimization BID21 (b) Figure 1 : Illustration of the margin function f (x) for a simple two-layer network.

(a) Contours of f (x) in an ∞ ball around x. Sharp curvature near x renders a linear approximation highly inaccurate, and f (A fgsm (x)) obtained by maximising this approximation is much smaller than f (A opt (x)).(b) Vector field for ∇f (x) with length of arrows proportional to ∇f (x) 1 .

In our approach, we bound f (A opt (x)) by bounding the maximum of ∇f (x) 1 over the neighborhood (green arrow).In general, this could be very different from ∇f (x) 1 at just the point x (red arrow).et al., 2017) .

Currently, these approaches can take up to several hours or longer to compute the loss for a single example even for small networks with a few hundred hidden units.

Training a network would require performing this computation in the inner loop, which is infeasible.

In this paper, we introduce an approach that avoids both the inaccuracy of lower bounds and the intractability of exact computation, by computing an upper bound on the worst-case loss for neural networks with one hidden layer, based on a semidefinite relaxation that can be computed efficiently.

This upper bound serves as a certificate of robustness against all attacks for a given network and input.

Minimizing an upper bound is safer than minimizing a lower bound, because points where the bound is loose have disproportionately higher objective values, which the optimizer will tend to avoid.

Furthermore, our certificate of robustness, by virtue of being differentiable, is trainable-it can be optimized at training time jointly with the network, acting as a regularizer that encourages robustness against all ∞ attacks.

In summary, we are the first (along with the concurrent work of BID24 ) to demonstrate a certifiable, trainable, and scalable method for defending against adversarial examples on two-layer networks.

We train a network on MNIST whose test error on clean data is 4.2%, and which comes with a certificate that no attack can misclassify more than 35% of the test examples using ∞ perturbations of size = 0.1.Notation.

For a vector z ∈ R n , we use z i to denote the i th coordinate of z. For a matrix Z ∈ R m×n , Z i denotes the i th row.

For any activation function σ : R → R (e.g., sigmoid, ReLU) and a vector z ∈ R n , σ(z) is a vector in R n with σ(z) i = σ(z i ) (non-linearity is applied element-wise).

We use B (z) to denote the ∞ ball of radius around z ∈ R d : B (z) = {z | |z i − z i | ≤ for i = 1, 2, . . .

d}. Finally, we denote the vector of all zeros by 0 and the vector of all ones by 1.

Score-based classifiers.

Our goal is to learn a mapping C : X → Y, where X = R d is the input space (e.g., images) and Y = {1, . . .

, k} is the set of k class labels (e.g., object categories).

Assume C is driven by a scoring function f i : X → R for all classes i ∈ Y, where the classifier chooses the class with the highest score: C(x) = arg max i∈Y f i (x).

Also, define the pairwise margin DISPLAYFORM0 for every pair of classes (i, j).

Note that the classifier outputs C(x) = i iff f ij (x) > 0 for all alternative classes j = i. Normally, a classifier is evaluated on the 0-1 loss DISPLAYFORM1 This paper focuses on linear classifiers and neural networks with one hidden layer.

For linear classifiers, For neural networks with one hidden layer consisting of m hidden units, the scoring function is DISPLAYFORM2 DISPLAYFORM3 , where W ∈ R m×d and V ∈ R k×m are the parameters of the first and second layer, respectively, and σ is a non-linear activation function applied elementwise (e.g., for ReLUs, σ(z) = max(z, 0)).

We will assume below that the gradients of σ are bounded: σ (z) ∈ [0, 1] for all z ∈ R; this is true for ReLUs, as well as for sigmoids (with the stronger bound σ (z) ∈ [0, 1 4 ]).

Attack model.

We are interested in classification in the presence of an attacker A : X → X that takes a (test) input x and returns a perturbationx.

We consider attackers A that can perturb each feature x i by at most ≥ 0; formally, A(x) is required to lie in the ∞ ball B (x) def = {x | x − x ∞ ≤ }, which is the standard constraint first proposed in BID50 .

Define the adversarial loss with respect to A as A (x, y) = I[C(A(x)) = y].We assume the white-box setting, where the attacker A has full knowledge of C. The optimal (untargeted) attack chooses the input that maximizes the pairwise margin of an incorrect class i over the correct class y: A opt (x) = arg maxx ∈B (x) max i f iy (x).

For a neural network, computing A opt is a non-convex optimization problem; heuristics are typically employed, such as the Fast Gradient Sign Method (FGSM) BID17 , which perturbs x based on the gradient, or the Carlini-Wagner attack, which performs iterative optimization BID8 .

For ease of exposition, we first consider binary classification with classes Y = {1, 2}; the multiclass extension is discussed at the end of Section 3.3.

Without loss of generality, assume the correct label for x is y = 2.

Simplifying notation, let f (x) = f 1 (x) − f 2 (x) be the margin of the incorrect class over the correct class.

Then A opt (x) = arg maxx ∈B (x) f (x) is the optimal attack, which is successful if f (A opt (x)) > 0.

Since f (A opt (x)) is intractable to compute, we will try to upper bound it via a tractable relaxation.

In the rest of this section, we first review a classic result in the simple case of linear networks where a tight upper bound is based on the 1 -norm of the weights (Section 3.1).

We then extend this to general classifiers, in which f (A opt (x)) can be upper bounded using the maximum 1 -norm of the gradient at any pointx ∈ B (x) (Section 3.2).

For two-layer networks, this quantity is upper bounded by the optimal value f QP (x) of a non-convex quadratic program (QP) (Section 3.3), which in turn is upper bounded by the optimal value f SDP (x) of a semidefinite program (SDP).

The SDP is convex and can be computed exactly (which is important for obtainining actual certificates).

To summarize, we have the following chain of inequalities: DISPLAYFORM0 which implies that the adversarial loss A (x) = I[f (A(x)) > 0] with respect to any attacker A is upper bounded by I[f SDP (x) > 0].

Note that for certain non-linearities such as ReLUs, ∇f (x) does not exist everywhere, but our analysis below holds as long as f is differentiable almost-everywhere.

For (binary) linear classifiers, we have f (x) = (W 1 − W 2 ) x, where W 1 , W 2 ∈ R d are the weight vectors for the two classes.

For any inputx ∈ B (x), Hölder's inequality with x −x ∞ ≤ gives: DISPLAYFORM0 Note that this bound is tight, obtained by taking DISPLAYFORM1

For more general classifiers, we cannot compute f (A opt (x)) exactly, but motivated by the above, we can use the gradient to obtain a linear approximation g: DISPLAYFORM0 Published as a conference paper at ICLR 2018Using this linear approximation to generate A(x) corresponds exactly to the Fast Gradient Sign Method (FGSM) BID17 .

However, f is only close to g whenx is very close to x, and people have observed the gradient masking phenomenon BID53 BID41 in several proposed defenses that train against approximations like g, such as saturating networks BID34 , distillation BID42 , and adversarial training BID17 .

Specifically, defenses that try to minimize ∇f (x) 1 locally at the training points result in loss surfaces that exhibit sharp curvature near those points, essentially rendering the linear approximation g(x) meaningless.

Some attacks BID53 evade these defenses and witness a large f (A opt (x)).

Figure 1a provides a simple illustration.

We propose an alternative approach: use integration to obtain an exact expression for f (x) in terms of the gradients along the line between x andx: DISPLAYFORM1 where the inequality follows from the fact that tx DISPLAYFORM2 The key difference between FORMULA8 and FORMULA7 is that we consider the gradients over the entire ball B (x) rather than only at x ( Figure 1b ).

However, computing the RHS of FORMULA8 is intractable in general.

For two-layer neural networks, this optimization has additional structure which we will exploit in the next section.

We now unpack the upper bound (4) for two-layer neural networks.

Recall from Section 2 that DISPLAYFORM0 is the difference in second-layer weights for the two classes.

Let us try to bound the norm of the gradient ∇f (x) 1 forx ∈ B (x).

If we apply the chain rule, we see that the only dependence onx is σ (Wx), the activation derivatives.

We now leverage our assumption that σ (z) ∈ [0, 1] m for all vectors z ∈ R m , so that we can optimize over possible activation derivatives s ∈ [0, 1] m directly independent of x (note that there is potential looseness because not all such s need be obtainable via somex ∈ B (x)).

Therefore: DISPLAYFORM1 where ( FORMULA11 into FORMULA8 , we obtain an upper bound on the adversarial loss that we call f QP : DISPLAYFORM2 Unfortunately, (6) still involves a non-convex optimization problem (since W diag(v) is not necessarily negative semidefinite).

In fact, it is similar to the NP-hard MAXCUT problem, which requires maximizing x Lx over x ∈ [−1, 1] d for a graph with Laplacian matrix L.While MAXCUT is NP-hard, it can be efficiently approximated, as shown by the celebrated semidefinite programming relaxation for MAXCUT in BID15 .

We follow a similar approach here to obtain an upper bound on f QP (x).First, to make our variables lie in DISPLAYFORM3 we reparametrize s to produce: DISPLAYFORM4 Next pack the variables into a vector y ∈ R m+d+1 and the parameters into a matrix M : DISPLAYFORM5 In terms of these new objects, our objective takes the form: DISPLAYFORM6 Note that every valid vector y ∈ [−1, +1] m+d+1 satisfies the constraints yy 0 and (yy ) jj = 1.

Defining P = yy , we obtain the following convex semidefinite relaxation of our problem: DISPLAYFORM7 Note that the optimization of the semidefinite program depends only on the weights v and W and does not depend on the inputs x, so it only needs to be computed once for a model (v, W ).Semidefinite programs can be solved with off-the-shelf optimizers, although these optimizers are somewhat slow on large instances.

In Section 4 we propose a fast stochastic method for training, which only requires computing the top eigenvalue of a matrix.

Generalization to multiple classes.

The preceding arguments all generalize to the pairwise margins f ij , to give: DISPLAYFORM8 DISPLAYFORM9 4 TRAINING THE CERTIFICATEIn the previous section, we proposed an upper bound (12) on the loss A (x, y) of any attack A, based on the bound (11).

Normal training with some classification loss cls (V, W ; x n , y n ) like hinge loss or cross-entropy will encourage the pairwise margin f ij (x) to be large in magnitude, but won't necessarily cause the second term in (11) involving M ij to be small.

A natural strategy is thus to use the following regularized objective given training examples (x n , y n ), which pushes down on both terms: DISPLAYFORM10 where λ ij > 0 are the regularization hyperparameters.

However, computing the gradients of the above objective involves finding the optimal solution of a semidefinite program, which is slow.

Duality to the rescue.

Our computational burden is lifted by the beautiful theory of duality, which provides the following equivalence between the primal maximization problem over P , and a dual minimization problem over new variables c (see Section A for details): ij ∈ R D that are optimized at the same time as the parameters V and W , resulting in an objective that can be trained efficiently using stochastic gradient methods.

DISPLAYFORM11 The final objective.

Using (14), we end up optimizing the following training objective: DISPLAYFORM12 The objective in FORMULA4 DISPLAYFORM13 for any attack A. As we train the network, we obtain a quick upper bound on the worst-case adversarial loss directly from the regularization loss, without having to optimize an SDP each time.

In Section 3, we described a function f ij SDP that yields an efficient upper bound on the adversarial loss, which we obtained using convex relaxations.

One could consider other simple ways to upper bound the loss; we describe here two common ones based on the spectral and Frobenius norms.

DISPLAYFORM0 where W 2 is the spectral norm (maximum singular value) of W .

This yields the following upper bound that we denote by f spectral : DISPLAYFORM1 This measure of vulnerability to adversarial examples based on the spectral norms of the weights of each layer is considered in BID50 and BID12 .Frobenius bound: For ease in training, often the Frobenius norm is regularized (weight decay) instead of the spectral norm.

Since W F ≥ W 2 , we get a corresponding upper bound f frobenius : DISPLAYFORM2 In Section 6, we empirically compare our proposed bound using f ij SDP to these two upper bounds.

We evaluated our method on the MNIST dataset of handwritten digits, where the task is to classify images into one of ten classes.

Our results can be summarized as follows: First, in Section 6.1, we show that our certificates of robustness are tighter than those based on simpler methods such as Frobenius and spectral bounds (Section 5), but our bounds are still too high to be meaningful for general networks.

Then in Section 6.2, we show that by training on the certificates, we obtain networks with much better bounds and hence meaningful robustness.

This reflects an important point: while accurately analyzing the robustness of an arbitrary network is hard, training the certificate jointly leads to a network that is robust and certifiably so.

In Section 6.3, we present implementation details, design choices, and empirical observations that we made while implementing our method.

Networks.

In this work, we focus on two layer networks.

In all our experiments, we used neural networks with m = 500 hidden units, and TensorFlow's implementation of Adam BID23 as the optimizer; we considered networks with more hidden units, but these did not substantially improve accuracy.

We experimented with both the multiclass hinge loss and cross-entropy.

All hyperparameters (including the choice of loss function) were tuned based on the error of the Projected Gradient Descent (PGD) attack BID31 ) at = 0.1; we report the hyperparameter settings below.

We considered the following training objectives providing 5 different networks:

Cross-entropy loss and no explicit regularization.

Hinge loss and a regularizer λ( W F + v 2 ) with λ = 0.08.

Hinge loss and a regularizer λ( W 2 + v 2 ) with λ = 0.09.

Cross-entropy with the adversarial loss against PGD as a regularizer, with the regularization parameter set to 0.5.

We found that this regularized loss works better than optimizing only the adversarial loss, which is the defense proposed in BID31 .

We set the step size of the PGD adversary to 0.1, number of iterations to 40, and perturbation size to 0.3.

of Section 4.

Implementation details and hyperparameter values are detailed in Section 6.3.Evaluating upper bounds.

Below we will consider various upper bounds on the adversarial loss Aopt (based on our method, as well as the Frobenius and spectral bounds described in Section 5).

Ideally we would compare these to the ground-truth adversarial loss Aopt , but computing this exactly is difficult.

Therefore, we compare upper bounds on the adversarial loss with a lower bound on Aopt instead.

The loss of any attack provides a valid lower bound and we consider the strong Projected Gradient Descent (PGD) attack run against the cross-entropy loss, starting from a random point in B (x), with 5 random restarts.

We observed that PGD against hinge loss did not work well, so we used cross-entropy even for attacking networks trained with the hinge loss.

For each of the five networks described above, we computed upper bounds on the 0-1 loss based on our certificate (which we refer to as the "SDP bound" in this section), as well as the Frobenius and spectral bounds described in Section 5.

While Section 4 provides a procedure for efficiently obtaining an SDP bound as a result of training, for networks not trained with our method we need to solve an SDP at the end of training to obtain certificates.

Fortunately, this only needs to be done once for every pair of classes.

In our experiments, we use the modeling toolbox YALMIP BID27 with Sedumi BID49 as a backend to solve the SDPs, using the dual form (14); this took roughly 10 minutes per SDP (around 8 hours in total for a given model).In FIG2 , we display average values of the different upper bounds over the 10, 000 test examples, as well as the corresponding lower bound from PGD.

We find that our bound is tighter than the Frobenius and spectral bounds for all the networks considered, but its tightness relative to the PGD lower bound varies across the networks.

For instance, our bound is relatively tight on Fro-NN, but unfortunately Fro-NN is not very robust against adversarial examples (the PGD attack exhibits large error).

In contrast, the adversarially trained network AT-NN does appear to be robust to attacks, but our certificate, despite being much tighter than the Frobenius and spectral bounds, is far away from the PGD lower bound.

The only network that is both robust and has relatively tight upper bounds is SDP-NN, which was explicitly trained to be both robust and certifiable as described in Section 4; we examine this network and the effects of training in more detail in the next subsection.

In the previous section, we saw that the SDP bound, while being tighter than simpler upper bounds, could still be quite loose on arbitrary networks.

However, optimizing against the SDP certificate seemed to make the certificate tighter.

In this section, we explore the effect of different optimization objectives in more detail.

First, we plot on a single axis the best upper bound (i.e., the SDP bound) and the lower bound (from PGD) on the adversarial loss obtained with each of the five training objectives discussed above.

This is given in FIG3 .Neither spectral nor Frobenius norm regularization seems to be helpful for encouraging adversarial robustness-the actual performance of those networks against the PGD attack is worse than the upper bound for SDP-NN against all attacks.

This shows that the SDP certificate actually provides a useful training objective for encouraging robustness compared to other regularizers.

Separately, we can ask whether SDP-NN is robust to actual attacks.

We explore the robustness of our network in FIG3 , where we plot the performance of SDP-NN against 3 attacks-the PGD attack from before, the Carlini-Wagner attack BID8 ) (another strong attack), and the weaker Fast Gradient Sign Method (FGSM) baseline.

We see substantial robustness against all 3 attacks, even though our method was not explicitly trained with any of them in mind.

Next, we compare to other bounds reported in the literature.

A rough ceiling is given by the network of BID31 , which is a relatively large four-layer convolutional network adversarially trained against PGD.

While this network has no accompanying certificate of robustness, it was evaluated against a number of attack strategies and had worst-case error 11% at = 0.3.

Another set of numbers comes from BID10 , who use formal verification methods to compute A opt exactly on 10 input examples for a small (72-node) variant of the Madry et al. network.

The authors reported to us that this network misclassifies 6 out of 10 examples at = 0.05 (we note that 4 out of 10 of these were misclassified to start with, but 3 of the 4 can also be flipped to a different wrong class with some < 0.07).At the value = 0.1 for which it was tuned, SDP-NN has error 16% against the PGD attack, and an upper bound of 35% error against any attack.

This is substantially better than the small 72-node network, but also much worse than the full Madry et al. network.

How much of the latter looseness comes from conservatism in our method, versus the fact that our network has only two layers?

We can get some idea by considering the AT-NN network, which was trained similarly to Madry et al., but uses the same architecture as SDP-NN.

From FIG3 , we see that the error of SDP-NN against PGD (16%) is not much worse than that of AT-NN (11%), even though AT-NN was explicitly trained against the PGD attack.

This suggests that most of the gap comes from the smaller network depth, Network PGD error SDP bound LP bound SDP-NN 15% 35% 99% LP-NN 22% 93% 26% BID24 .

Numbers are reported for = 0.1.

LP-NN has a certificate (provided by the LP bound) that no attack can misclassify more than 26% of the examples.rather than from conservatism in the SDP bound.

We are currently in the process of extending our approach to deeper networks, and optimistic about obtaining improved bounds with such networks.

Finally, we compare with the approach proposed in BID24 whose work appeared shortly after an initial version of our paper.

They provide an upper bound on the adversarial loss using linear programs (LP) followed by a method to efficiently train networks to minimize this upper bound.

In order to compare with SDP-NN, the authors provided us with a network with the same architecture as SDP-NN, but trained using their LP based objective.

We call this network LP-NN.

TAB2 shows that LP-NN and SDP-NN are comparable in terms of their robustness against PGD, and the robustness guarantees that they come with.

Interestingly, the SDP and LP approaches provide vacuous bounds for networks not trained to minimize the respective upper bounds (though these networks are indeed robust).

This suggests that these two approaches are comparable, but complementary.

Finally, we note that in contrast to this work, the approach of BID24 extends to deeper networks, which allows them to train a four-layer CNN with a provable upper bound on adversarial error of 8.4% error.

We implemented our training objective in TensorFlow, and implemented λ + max as a custom operator using SciPy's implementation of the Lanczos algorithm for fast top eigenvector computation; occasionally Lanczos fails to converge due to a small eigen-gap, in which case we back off to a full SVD.

We used hinge loss as the classification loss, and decayed the learning rate in steps from 10 −3 to 10 −5 , decreasing by a factor of 10 every 30 epochs.

Each gradient step involves computing top eigenvectors for 45 different matrices, one for each pair of classes (i, j).

In order to speed up computation, for each update, we randomly pick i t and only compute gradients for pairs (i t , j), j = i t , requiring only 9 top eigenvector computations in each step.

For the regularization parameters λ ij , the simplest idea is to set them all equal to the same value; this leads to the unweighted regularization scheme where λ ij = λ for all pairs (i, j).

We tuned λ to 0.05, which led to reasonably good bounds.

However, we observed that certain pairs of classes tended to have larger margins f ij (x) than other classes, which meant that certain label pairs appeared in the maximum of (12) much more often.

That led us to consider a weighted regularization scheme with λ ij = w ij λ, where w ij is the fraction of training points for which the the label i (or j) appears as the maximizing term in (12).

We updated the values of these weights every 20 epochs.

FIG4 compares the PGD lower bound and SDP upper bound for the unweighted and weighted networks.

The weighted network is better than the unweighted network for both the lower and upper bounds.

Finally, we saw in Equation 16 of Section 4 that the dual variables c ij provide a quick-to-compute certificate of robustness.

FIG4 shows that the certificates provided by these dual variables are very close to what we would obtain by fully optimizing the semidefinite programs.

These dual certificates made it easy to track robustness across epochs of training and to tune hyperparameters.

In this work, we proposed a method for producing certificates of robustness for neural networks, and for training against these certificates to obtain networks that are provably robust against adversaries.

Related work.

In parallel and independent work, BID24 also provide provably robust networks against ∞ perturbations by using convex relaxations.

While our approach uses a single semidefinite program to compute an upper bound on the adversarial loss, Kolter & Wong (2017) use separate linear programs for every data point, and apply their method to networks of depth up to four.

In theory, neither bound is strictly tighter than the other, and our experiments TAB2 suggest that the two bounds are complementary.

Combining the approaches seems to be a promising future direction.

BID21 and the follow-up BID10 also provide certificates of robustness for neural networks against ∞ perturbations.

That work uses SMT solvers, which are a tool from the formal verification literature.

The SMT solver can answer the binary question "Is there an adversarial example within distance of the input x?", and is correct whenever it terminates.

The main drawback of SMT and similar formal verification methods is that they are slow-they have worst-case exponential-time scaling in the size of the network; moreover, to use them during training would require a separate search for each gradient step.

BID20 use SMT solvers and are able to analyze state-of-the-art networks on MNIST, but they make various approximations such that their numbers are not true upper bounds.

BID2 provide tractable certificates but require to be small enough to ensure that the entire ∞ ball around an input lies within the same linear region.

For the networks and values of that we consider in our paper, we found that this condition did not hold.

Recently, Hein & Andriushchenko (2017) proposed a bound for guaranteeing robustness to p -norm perturbations, based on the maximum p p−1 -norm of the gradient in the -ball around the inputs.

BID19 show how to efficiently compute this bound for p = 2, as opposed to our work which focuses on ∞ and requires different techniques to achieve scalability.

BID31 perform adversarial training against PGD on the MNIST and CIFAR-10 datasets, obtaining networks that they suggest are "secure against first-order adversaries".

However, this is based on an empirical observation that PGD is nearly-optimal among gradient-based attacks, and does not correspond to any formal robustness guarantee.

Finally, the notion of a certificate appears in the theory of convex optimization, but means something different in that context; specifically, it corresponds to a proof that a point is near the optimum of a convex function, whereas here our certificates provide upper bounds on non-convex functions.

Additionally, while robust optimization BID3 provides a tool for optimizing objectives with robustness constraints, applying it directly would involve the same intractable optimization for A opt that we deal with here.

Other approaches to verification.

While they have not been explored in the context of neural networks, there are approaches in the control theory literature for verifying robustness of dynamical systems, based on Lyapunov functions (Lyapunov, 1892; BID29 .

We can think of the activations in a neural network as the evolution of a time-varying dynamical system, and attempt to prove stability around a trajectory of this system BID51 BID52 .

Such methods typically use sum-of-squares verification BID38 BID43 and are restricted to relatively low-dimensional dynamical systems, but could plausibly scale to larger settings.

Another approach is to construct families of networks that are provably robust a priori, which would remove the need to verify robustness of the learned model; to our knowledge this has not been done for any expressive model families.

Adversarial examples and secure ML.

There has been a great deal of recent work on the security of ML systems; we provide only a sampling here, and refer the reader to BID0 , BID4 , BID41 , and BID14 for some recent surveys.

Adversarial examples for neural networks were first discovered by BID50 , and since then a number of attacks and defenses have been proposed.

We have already discussed gradientbased methods as well as defenses based on adversarial training.

There are also other attacks based on, e.g., saliency maps BID40 , KL divergence BID33 , and elastic net optimization BID11 ; many of these attacks are collated in the cleverhans repository .

For defense, rather than making networks robust to adversaries, some work has focused on simply detecting adversarial examples.

However, BID7 recently showed that essentially all known detection methods can be subverted by strong attacks.

As explained in BID0 , there are a number of different attack models beyond the testtime attacks considered here, based on different attacker goals and capabilities.

For instance, one can consider data poisoning attacks, where an attacker modifies the training set in an effort to affect test-time performance.

BID35 , BID26 , and BID5 have demonstrated poisoning attacks against real-world systems.

Other types of certificates.

Certificates of performance for machine learning systems are desirable in a number of settings.

This includes verifying safety properties of air traffic control systems BID21 and self-driving cars (O' Kelly et al., 2016; , as well as security applications such as robustness to training time attacks BID48 .

More broadly, certificates of performance are likely necessary for deploying machine learning systems in critical infrastructure such as internet packet routing BID54 BID47 .

In robotics, certificates of stability are routinely used both for safety verification BID30 BID32 and controller synthesis BID1 BID51 .In traditional verification work, Rice's theorem BID44 ) is a strong impossibility result essentially stating that most properties of most programs are undecidable.

Similarly, we should expect that verifying robustness for arbitrary neural networks is hard.

However, the results in this work suggest that it is possible to learn neural networks that are amenable to verification, in the same way that it is possible to write programs that can be formally verified.

Optimistically, given expressive enough certification methods and model families, as well as strong enough specifications of robustness, one could even hope to train vector representations of natural images with strong robustness properties, thus finally closing the chapter on adversarial vulnerabilities in the visual domain.

All code, data and experiments for this paper are available on the Codalab platform at https://worksheets.codalab.org/worksheets/ 0xa21e794020bb474d8804ec7bc0543f52/.

<|TLDR|>

@highlight

We demonstrate a certifiable, trainable, and scalable method for defending against adversarial examples.

@highlight

Proposes a new defense against security attacks on neural networks with the atack model that outputs a security certificate on the algorithm.

@highlight

Derives an upper bound on adversarial perturbation for neural networks with one hidden layer