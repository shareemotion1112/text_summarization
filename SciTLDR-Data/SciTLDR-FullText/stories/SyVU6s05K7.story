Learning a deep neural network requires solving a challenging optimization problem: it is a high-dimensional, non-convex and non-smooth minimization problem with a large number of terms.

The current practice in neural network optimization is to rely on the stochastic gradient descent (SGD) algorithm or its adaptive variants.

However, SGD requires a hand-designed schedule for the learning rate.

In addition, its adaptive variants tend to produce solutions that generalize less well on unseen data than SGD with a hand-designed schedule.

We present an optimization method that offers empirically the best of both worlds: our algorithm yields good generalization performance while requiring only one hyper-parameter.

Our approach is based on a composite proximal framework, which exploits the compositional nature of deep neural networks and can leverage powerful convex optimization algorithms by design.

Specifically, we employ the Frank-Wolfe (FW) algorithm for SVM, which computes an optimal step-size in closed-form at each time-step.

We further show that the descent direction is given by a simple backward pass in the network, yielding the same computational cost per iteration as SGD.

We present experiments on the CIFAR and SNLI data sets, where we demonstrate the significant superiority of our method over Adam, Adagrad, as well as the recently proposed BPGrad and AMSGrad.

Furthermore, we compare our algorithm to SGD with a hand-designed learning rate schedule, and show that it provides similar generalization while often converging faster.

The code is publicly available at https://github.com/oval-group/dfw.

Since the introduction of back-propagation BID23 , stochastic gradient descent (SGD) has been the most commonly used optimization algorithm for deep neural networks.

While yielding remarkable performance on a variety of learning tasks, a downside of the SGD algorithm is that it requires a schedule for the decay of its learning rate.

In the convex setting, curvature properties of the objective function can be used to design schedules that are hyper-parameter free and guaranteed to converge to the optimal solution (Bubeck, 2015) .

However, there is no analogous result of practical interest for the non-convex optimization problem of a deep neural network.

An illustration of this issue is the diversity of learning rate schedules used to train deep convolutional networks with SGD: BID25 and He et al. (2016) adapt the learning rate according to the validation performance, while BID27 , BID3 and BID8 use pre-determined schedules, which are respectively piecewise constant, geometrically decaying, and cyclic with a cosine annealing.

While these protocols result in competitive or state-of-the-art results on their learning task, there does not seem to be a consistent methodology.

As a result, finding such a schedule for a new setting is a time-consuming and computationally expensive effort.

To alleviate this issue, adaptive gradient methods have been developed BID36 BID4 BID21 , and borrowed from online convex optimization (Duchi et al., 2011) .

Typically, these methods only require the tuning of the initial learning rate, the other hyper-parameters being considered robust across applications.

However, it has been shown that such adaptive gradient methods obtain worse generalization than SGD BID32 .

This observation is corroborated by our experimental results.

In order to bridge this performance gap between existing adaptive methods and SGD, we introduce a new optimization algorithm, called Deep Frank-Wolfe (DFW).

The DFW algorithm exploits the composite structure of deep neural networks to design an optimization algorithm that leverages efficient convex solvers.

In more detail, we consider a composite (nested) optimization problem, with the loss as the outer function and the function encoded by the neural network as the inner one.

At each iteration, we define a proximal problem with a first-order approximation of the neural network (linearized inner function), while keeping the loss function in its exact form (exact outer function).

When the loss is the hinge loss, each proximal problem created by our formulation is exactly a linear SVM.

This allows us to employ the powerful Frank-Wolfe (FW) algorithm as the workhorse of our procedure.

There are two by-design advantages to our method compared to the SGD algorithm.

First, each iteration exploits more information about the learning objective, while preserving the same computational cost as SGD.

Second, an optimal step-size is computed in closed-form by using the FW algorithm in the dual (Frank & Wolfe, 1956 BID5 .

Consequently, we do not need a hand-designed schedule for the learning rate.

As a result, our algorithm is the first to provide competitive generalization error compared to SGD, all the while requiring a single hyper-parameter and often converging significantly faster.

We present two additional improvements to customize the use of the DFW algorithm to deep neural networks.

First, we show how to smooth the loss function to avoid optimization difficulties arising from learning deep models with SVMs (Berrada et al., 2018) .

Second, we incorporate Nesterov momentum (Nesterov, 1983) to accelerate our algorithm.

We demonstrate the efficacy of our method on image classification with the CIFAR data sets (Krizhevsky, 2009) using two architectures: wide residual networks BID35 and densely connected convolutional neural networks BID3 ; we also provide experiments on natural language inference with a Bi-LSTM on the SNLI corpus (Bowman et al., 2015) .

We show that the DFW algorithm often strongly outperforms previous methods based on adaptive learning rates.

Furthermore, it provides comparable or better accuracy to SGD with hand-designed learning rate schedules.

In conclusion, our contributions can be summed up as follows:• We propose a proximal framework which preserves information from the loss function.• For the first time for deep neural networks, we demonstrate how our formulation gives at each iteration (i) an optimal step-size in closed form and (ii) an update at the same computational cost as SGD.• We design a novel smoothing scheme for the dual optimization of SVMs.• To the best of our knowledge, the resulting DFW algorithm is the first to offer comparable or better generalization to SGD with a hand-designed schedule on the CIFAR data sets, all the while converging several times faster and requiring only a single hyperparameter.

Non Gradient-Based Methods.

The success of a simple first-order method such as SGD has led to research in other more sophisticated techniques based on relaxations (Heinemann et al., 2016 BID37 , learning theory (Goel et al., 2017) , Bregman iterations BID29 , and even second-order methods BID22 BID10 BID18 , Desjardins et al., 2015 BID9 , Grosse & Martens, 2016 , Ba et al., 2017 , Botev et al., 2017 BID11 .

While such methods hold a lot of promise, their relatively large per-iteration cost limits their scalability in practice.

As a result, gradient-based methods continue to be the most popular optimization algorithms for learning deep neural networks.

Adaptive Gradient Methods.

As mentioned earlier, one of the main challenges of using SGD is the design of a learning rate schedule.

Several works proposed alternative first-order methods that do not require such a schedule, by either modifying the descent direction or adaptively rescaling the step-size (Duchi et al., 2011 BID36 BID24 BID4 BID38 BID21 .

However, as noted above, the adaptive variants of SGD sometimes provide subpar generalization BID32 .Learning to Learn and Meta-Learning.

Learning to learn approaches have also been proposed to optimize deep neural networks.

Baydin et al. (2018) and BID33 learn the learning rate to avoid a hand-designed schedule and to improve practical performance.

Such methods can be combined with our proposed algorithm to learn its proximal coefficient, instead of considering it as a fixed hyper-parameter to be tuned.

Meta-learning approaches have also been suggested to learn the optimization algorithm BID0 BID20 BID31 BID7 .

This line of work, which is orthogonal to ours, could benefit from the use of DFW to optimize the meta-learner.

Optimization and Generalization.

Several works study the relationship between optimization and generalization in deep learning.

In order to promote generalization within the optimization algorithm itself, BID15 proposed the Path-SGD algorithm, which implicitly controls the capacity of the model.

However, their method required the model to employ ReLU non-linearity only, which is an important restriction for practical purposes.

Hardt et al. (2016) , Arpit et al. (2017) , BID17 , BID2 and Chaudhari & Soatto (2018) analyzed how existing optimization algorithms implicitly regularize deep neural networks.

However this phenomenon is not yet fully understood, and the resulting empirical recommendations are sometimes opposing (Hardt et al., 2016 BID2 .Proximal Methods.

The back-propagation algorithm has been analyzed in a proximal framework in (Frerix et al., 2018 ).

Yet, the resulting approach still requires the same hyper-parameters as SGD and incurs a higher computational cost per iteration.

Linear SVM Sub-Problems.

A main component of our formulation is to formulate sub-problems as linear SVMs.

In an earlier work (Berrada et al., 2017) , we showed that neural networks with piecewise linear activations could be trained with the CCCP algorithm BID34 , which yielded approximate SVM problems to be solved with the BCFW algorithm BID5 .

However this algorithm only updates the parameters of one layer at a time, which slows down convergence significantly in practice.

Closest to our approach are the works of BID1 and BID26 .

BID1 suggested to create a local SVM based on a first-order Taylor expansion and a proximal term, in order to lower the error of every data sample while minimizing the changes in the weights.

However their method operated in a non-stochastic setting, making the approach infeasible for large-scale data sets.

BID26 , a parallel work to ours, also created an SVM problem using a first-order Taylor expansion, this time in a mini-batch setting.

Their work provided interesting insights from a statistical learning theory perspective.

While their method is well-grounded, its significantly higher cost per iteration impairs its practical speed and scalability.

As such, it can be seen as complementary to our empirical work, which exploits a powerful solver and provides state-of-the-art scalability and performance.

Before describing our formulation, we introduce some necessary notation.

We use · to denote the Euclidean norm.

Given a function φ, ∂φ(u) û is the derivative of φ with respect to u evaluated atû.

According to the situation, this derivative can be a gradient, a Jacobian or even a directional derivative.

Its exact nature will be clear from context throughout the paper.

We also introduce the first-order Taylor expansion of φ around the pointû: Tûφ(u) = φ(û) + (∂φ(u) û ) (u −û).

For a positive integer p, we denote the set {1, 2, ..., p} as [p] .

For simplicity, we assume that stochastic algorithms process only one sample at each iteration, although the methods can be trivially extended to mini-batches of size larger than one.

We suppose we are given a data set (x i , y i ) i∈ [N ] , where each DISPLAYFORM0 is a sample annotated with a label y i from the output space Y. The data set is used to estimate a parameterized model represented by the function f .

Given its (flattened) parameters w ∈ R p , and an input DISPLAYFORM1 , a vector with one score per element of the output space Y. For instance, f can be a linear map or a deep neural network.

Given a vector of scores per label s ∈ R |Y| , we denote by L(s, y i ) the loss function that computes the risk of the prediction scores s given the ground truth label y i .

For example, the loss L can be cross-entropy or the multi-class hinge loss: DISPLAYFORM2 DISPLAYFORM3 The cross-entropy loss (1) tries to match the empirical distribution by driving incorrect scores as far as possible from the ground truth one.

The hinge loss (2) attempts to create a minimal margin of one between correct and incorrect scores.

The hinge loss has been shown to be more robust to over-fitting than cross-entropy, when combined with smoothing techniques that are common in the optimization literature (Berrada et al., 2018) .

To simplify notation, we introduce DISPLAYFORM4 .

Finally, we denote by ρ(w) the regularization (typically the squared Euclidean norm).

We now write the learning problem under its empirical risk minimization form: DISPLAYFORM5 3.2 A PROXIMAL APPROACH Our main contribution is a formulation which exploits the composite nature of deep neural networks in order to obtain a better approximation of the objective at each iteration.

Thanks to the careful approximation design, this approach yields sub-problems that are amenable to efficient optimization by powerful convex solvers.

In order to understand the intuition of our approach, we first present a proximal gradient perspective on SGD.The SGD Algorithm.

At iteration t, the SGD algorithm selects a sample j at random and observes the objective estimate ρ(w t ) + L j (f j (w t )).

Then, given the learning rate η t , it performs the following update on the parameters: DISPLAYFORM6 Equation FORMULA6 is the closed-form solution of a proximal problem where the objective has been linearized by the first-order Taylor expansion T wt (Bubeck, 2015) : DISPLAYFORM7 To see the relationship between (4) and FORMULA7 , one can set the gradient with respect to w to 0 in equation FORMULA7 , and observe that the resulting equation is exactly (4).

In other words, SGD minimizes a first-order approximation of the objective, while encouraging proximity to the current estimate w t .However, one can also choose to linearize only a part of the composite objective BID6 .

Choosing which part to approximate is a crucial decision, because it yields optimization problems with widely different properties.

In this work, we suggest an approach that lends itself to fast optimization with robust convex solvers and preserves information about the learning task by keeping an exact loss function.

Loss-Preserving Linearization.

In detail, at iteration t, with selected sample j, we introduce the proximal problem that linearizes the regularization ρ and the model f j , but not the loss function L: Figure 1: We illustrate the different approximations on a synthetic composite objective function Φ(w) = L(f (w)) (Φ is plotted in black).

In this example, L is a maximum of linear functions (similarly to a hinge loss) and f is a non-linear smooth map.

We denote the current iterate by w t , and the point minimizing Φ by w * .

On the left-hand side, one can observe how the SGD approximation is a single line (tangent at Φ(w t ), in blue), while the LPL approximation is piecewise linear (in orange), and thus matches the objective curve (in black) more closely.

On the right-hand side, an identical proximal term is added to both approximations to visualize equations FORMULA7 and FORMULA8 .

Thanks to the better accuracy of the LPL approximation, the iterate w LPL t+1 gets closer to the solution w * than w SGD t+1 .

This effect is particularly true when the proximal coefficient 1 2ηt is small, or equivalently, when the learning rate η t is large.

Indeed, the accuracy of the local approximation becomes more important when the proximal term is contributing less (e.g. when η t is large).

DISPLAYFORM8 In figure 1, we provide a visual comparison of equations FORMULA7 and FORMULA8 in the case of a piecewise linear loss.

As will be seen, by preserving the loss function, we will be able to achieve good performance across a number of tasks with a fixed η t = η.

Consequently, we will provide the first algorithm to accurately learn deep neural networks with only a single hyper-parameter while offering similar performance compared to SGD with a hand-designed schedule.

We focus on the optimization of equation FORMULA8 when L is a multi-class hinge loss (2).

The results of this section were originally derived for linear models BID5 .

Our contribution is to show for the first time how they can be exploited for deep neural networks thanks to our formulation (6).

We will refer to the resulting algorithm for neural networks as Deep Frank-Wolfe (DFW).

We begin by stating the key advantage of our method.

Proposition 1 (Optimal step-size, BID5 ).

Problem (6) with a hinge loss is amenable to optimization with Frank-Wolfe in the dual, which yields an optimal step-size γ t ∈ [0, 1] in closed-form at each iteration t.

This optimal step-size can be obtained in closed-form because the hinge loss is convex and piecewise linear.

In fact, the approach presented here can be applied to any loss function L that is convex and piecewise linear (another example would be the l 1 distance for regression for instance).Since the step-size can be computed in closed-form, the main computational challenge is to obtain the update direction, that is, the conditional gradient of the dual.

In the following result, we show that by taking a single step per proximal problem, this dual conditional gradient can be computed at the same cost as a standard stochastic gradient.

The proof is available in appendix A.5.

If a single step is performed on the dual of (6), its conditional gradient is given by −∂ (ρ(w) + L y (f x (w))) wt .

Given the step-size γ t , the resulting update can be written as: DISPLAYFORM0 In other words, the cost per iteration of the DFW algorithm is the same as SGD, since the update only requires standard stochastic gradients.

In addition, we point out that in a mini-batch setting, the conditional gradient is given by the average of the gradients over the mini-batch.

As a consequence, we can use batch Frank-Wolfe in the dual rather than coordinate-wise updates, with the same parallelism as SGD over the samples of a mini-batch.

One can observe how the update (7) exploits the optimal step-size γ t ∈ [0, 1] given by Proposition 1.

There is a geometric interpretation to the role of this step-size γ t .

When γ t is set to its minimal value 0, the resulting iterate does not move along the direction ∂L j (f j (w)) wt .

Since the step-size is optimal, this can only happen if the current iterate is detected to be at a minimum of the piecewise linear approximation.

Conversely, when γ t reaches its maximal value 1, the algorithm tries to move as far as possible along the direction ∂L j (f j (w)) wt .

In that case, the update is the same as the one obtained by SGD (as given by equation FORMULA6 ).

In other words, γ t can automatically decay the effective learning rate, hereby preventing the need to design a learning rate schedule by hand.

As mentioned previously, the DFW algorithm performs only one step per proximal problem.

Since problem FORMULA8 is only an approximation of the original problem (3), it may be unnecessarily expensive to solve it very accurately.

Therefore taking a single step per proximal problem may help the DFW algorithm to converge faster.

This is confirmed by our experimental results, which show that DFW is often able to minimize the learning objective (3) at greater speed than SGD.

We present two improvements to customize the application of our algorithm to deep neural networks.

Smoothing.

The SVM loss is non-smooth and has sparse derivatives, which can cause difficulties when training a deep neural network (Berrada et al., 2018) .

In Appendix A.6, we derive a novel result that shows how we can exploit the smooth primal cross-entropy direction and inexpensively detect when to switch back to using the standard conditional gradient.

Nesterov Momentum.

To take advantage of acceleration similarly to the SGD baseline, we adapt the Nesterov momentum to the DFW algorithm.

We defer the details to the appendix in A.7 for space reasons.

We further note that the momentum coefficient µ is typically set to a high value, say 0.9, and does not contribute significantly to the computational cost of cross-validation.

The main steps of DFW are shown in Algorithm 1.

As the key feature of our approach, note that the step-size is computed in closed-form in step 10 of the algorithm (colored in blue).Note that only the hyper-parameter η will be tuned in our experiments: we will use the same batch-size, momentum and number of epochs as the baselines in our experiments (unless specified otherwise).

In addition, we point out again that when γ t = 1, we recover the SGD step with Nesterov momentum.

In sections A.5 and A.6 of the appendix, we detail the derivation of the optimal step-size (step 10) and the computation of the search direction (step 7).

The computation of the dual search direction is omitted here for space reasons.

However, its implementation is straightforward in practice, and its computational cost is linear in the size of the output space.

Finally, we emphasize that the DFW algorithm is motivated by an empirical perspective.

While our method is not guaranteed to converge, our experiments show an effective minimization of the learning objective for the problems encountered in practice.

We compare the Deep Frank Wolfe (DFW) algorithm to the state-of-the-art optimizers.

We show that, across diverse data sets and architectures, the DFW algorithm outperforms adaptive gradient methods (with the exception of one setting, DN-10, where it obtains similar performance to AMSGrad and BPGrad).

In addition, the DFW algorithm offers competitive and sometimes superior performance to Receive data of mini-batch (x i , y i ) i∈B 6: DISPLAYFORM0 ∀i ∈ B, s DISPLAYFORM1 Dual direction (details in Appendix A.6) 8: DISPLAYFORM2 Derivative of (smoothed) loss function 9:r t = ∂ρ(w) wt Derivative of regularization 10: DISPLAYFORM3 Step-size 11: DISPLAYFORM4 12: DISPLAYFORM5 Parameters update 13: DISPLAYFORM6 end for 15: end for SGD at a lower computational cost, even though SGD has the advantage of a hand-designed schedule that has been chosen separately for each of these tasks.

Our experiments are implemented in pytorch BID19 , and the code is available at https://github.com/oval-group/dfw.

All models are trained on a single Nvidia Titan Xp card.

Data Set & Architectures.

The CIFAR-10/100 data sets contain 60,000 RGB natural images of size 32 × 32 with 10/100 classes (Krizhevsky, 2009).

We split the training set into 45,000 training samples and 5,000 validation samples, and use 10,000 samples for testing.

The images are centered and normalized per channel.

Unless specified otherwise, no data augmentation is employed.

We perform our experiments on two modern architectures of deep convolutional neural networks: wide residual networks BID35 , and densely connected convolutional networks BID3 .

Specifically, we employ a wide residual network of depth 40 and width factor 4, which has 8.9M parameters, and a "bottleneck" densely connected convolutional neural network of depth 40 and growth factor 40, which has 1.9M parameters.

We refer to these architectures as WRN and DN respectively.

All the following experimental details follow the protocol of BID35 and BID3 .

The only difference is that, instead of using 50,000 samples for training, we use 45,000 samples for training, and 5,000 samples for the validation set, which we found to be essential for all adaptive methods.

While Deep Frank Wolfe (DFW) uses an SVM loss, the baselines are trained with the Cross-Entropy (CE) loss since this resulted in better performance.

Method.

We compare DFW to the most common adaptive learning rates currently used: Adagrad (Duchi et al., 2011 ), Adam (Kingma & Ba, 2015 , the corrected version of Adam called AMSGrad BID21 , and BPGrad BID38 .

For these methods and for DFW, we cross-validate the initial learning rate as a power of 10.

We also evaluate the performance of SGD with momentum (simply referred to as SGD), for which we follow the protocol of BID35 and BID3 .

For all methods, we set a budget of 200 epochs for WRN and 300 epochs for DN.

Furthermore, the batch-size is respectively set to 128 and 64 for WRN and DN as in BID35 and BID3 .

For DN, the l 2 regularization is set to 10 −4 as in BID3 .

For WRN, the l 2 is cross-validated between 5.10 DISPLAYFORM0 , as in BID35 , and 10 −4, a more usual value that we have found to perform better for some of the methods (in particular DFW, since the corresponding loss function is an SVM instead of CE, for which the value of 5.10 DISPLAYFORM1 was designed).

The value of the Nesterov momentum is set to 0.9 for BPGrad, SGD and DFW.

DFW has only one hyper-parameter to tune, namely η, which is analogous to an initial learning rate.

For SGD, the initial learning rate is set to 0.1 on both WRN and DN.

Following BID35 and BID3 , it is then divided by 5 at epochs 60, 120 and 180 for WRN, and by 10 at epochs 150 and 225 for DN.Results.

We present the results in Table 1 Observe that DFW significantly outperforms the adaptive gradient methods, particularly on the more challenging CIFAR-100 data set.

On the WRN-CIFAR-100 task in particular, DFW obtains a testing accuracy which is about 7% higher than all other adaptive methods and outperforms SGD with a hand-designed schedule by 1%.

The inferior generalization of adaptive gradient methods is consistent with the findings of BID32 .

On all tasks, the accuracy of DFW is comparable to SGD.

Note that DFW converges significantly faster than SGD: the network reaches its final performance several times faster than SGD in all cases.

We illustrate this with an example in figure 2, which plots the training and validation errors on DN-CIFAR-100.

In figure 3 , one can see how the step-size is automatically decayed by DFW on this same experiment: we compare the effective step-size γ t η for DFW to the manually tuned η t for SGD.

Step-Size SGD DFW Figure 3 : The (automatic) evolution of γ t η for the DFW algorithm compared to the "staircase" hand-designed schedule of η t for SGD.Data Augmentation.

Since data augmentation provides a significant boost to the final accuracy, we provide additional results that make use of it.

Specifically, we randomly flip the images horizontally and randomly crop them with four pixels padding.

For methods that do not use a hand-designed schedule, such data augmentation introduces additional variance which makes the adaptation of the step-size more difficult.

Therefore we allow the batch size of adaptive methods (e.g. all methods but SGD) to be chosen as 1x, 2x or 4x, where x is the original value of batch-size (64 for DN, 128 for WRN).

Due to the heavy computational cost of the cross-validation (we tune the batch-size, regularization and initial learning rate), we provide results for SGD, DFW and the best performing adaptive gradient method, which is AMSGrad.

For SGD the hyper-parameters are kept the same as in BID35 and BID3 .

We present the results in TAB4 TAB9 of BID3 .

The small difference between the results of SGD and SGD * can be explained by the fact that we use 5,000 fewer training samples in our experiments (these are kept for validation).

The results of this table show that DFW systematically outperforms AMSGrad on this task (by up to 7% on WRN-100).These results confirm that DFW consistently outperforms AMSGrad, which is the best adaptive baseline on these tasks.

In particular, DFW obtains a test accuracy which is 7% better than AMSGrad on WRN-100.

Data Set.

The Stanford Natural Language Inference (SNLI) data set is a large corpus of 570k pairs of sentences (Bowman et al., 2015) .

Each sentence is labeled by one of the three possible labels: entailment, neutral and contradiction.

This allows the model to learn the semantics of the text data from a three-way classification problem.

Thanks to its scale and its supervised labels, this data set allows large neural networks to learn high-quality text embeddings.

As Conneau et al. (2017) demonstrate, the SNLI corpus can thus be used as a basis for transfer learning in natural language processing, in the same way that the ImageNet data set is used for pre-training in computer vision.

Method.

We follow the protocol of (Conneau et al., 2017) to learn their best model, namely a bi-directional LSTM of about 47M parameters.

In particular, the reported results use SGD with an initial learning rate of 0.1 and a hand-designed schedule that adapts to the variations of the validation set: if the validation accuracy does not improve, the learning rate is divided by a factor of 5.

We also report results on Adagrad, Adam, AMSGrad and BPGrad.

Following the official SGD baseline, Nesterov momentum is deactivated.

Using their open-source implementation, we replace the optimization by the DFW algorithm, the CE loss by an SVM, and leave all other components unchanged.

In this experiment, we use the conditional gradient direction rather than the CE gradient, since three-way classification does not cause sparsity in the derivative of the hinge loss (which is the issue that originally motivated our use of a different direction).

We cross-validate our initial proximal term as a power of ten, and do not manually tune any schedule.

In order to disentangle the importance of the loss function from the optimization algorithm, we run the baselines with both an SVM loss and a CE loss.

The initial learning rate of the baselines is also cross-validated as a power of ten.

Results.

The results are presented in Note that these results outperform the reported testing accuracy of 84.5% in (Conneau et al., 2017) that is obtained with CE.

This experiment, which is performed on a completely different architecture and data set than the previous one, confirms that DFW outperforms adaptive gradient methods and matches the performance of SGD with a hand-designed learning rate schedule.6 THE IMPORTANCE OF THE STEP-SIZE

It is worth discussing the subtle relationship between optimization and generalization.

In order to emphasize the impact of implicit regularization, all results presented in this section do not use data augmentation.

As a first illustrative example, we consider the following experiment: we take the protocol to train the DN network on CIFAR-100 with SGD, and simply change the initial learning rate to be ten times smaller, and the budget of epochs to be ten times larger.

As a result, the final training objective significantly decreases from 0.33 to 0.069.

Yet at the same time, the best validation accuracy decreases from 70.94% to 68.7%.

A similar effect occurs when decreasing the value of the momentum, and we have observed this across various convolutional architectures.

In other words, accurate optimization is less important for generalization than the implicit regularization of a high learning rate.

We have observed DFW to accurately optimize the learning objective in our experiments.

However, given the above observation, we believe that its good generalization properties are rather due to its capability to usually maintain a high learning rate at an early stage.

Similarly, the good generalization performance of SGD may be due to its schedule with a large number of steps at a high learning rate.

The previous section has qualitatively hinted at the importance of the step-size for generalization.

Here we quantitatively analyze the impact of the initial learning rate η on both the training accuracy (quality of optimization) and the validation accuracy (quality of generalization).

We compare results of the DFW and SGD algorithms on the CIFAR data sets when varying the value of η as a power of 10.

The results on the validation set are summarized in FIG2 , and the performance on the training set is reported in Appendix B.On the training set, both methods obtain nearly perfect accuracy across at least three orders of magnitude of η (details in Appendix B.4).

In contrast, the results of figure 4 confirm that the validation performance is sensitive to the choice of η for both methods.

In some cases where η is high, SGD obtains a better performance than DFW.

This is because the handdesigned schedule of SGD enforces a decay of η, while the DFW algorithm relies on an automatic decay of the step-size γ t for effective convergence.

This automatic decay may not happen if a small proximal term (large η) is combined with a local approximation that is not sufficiently accurate (for instance with a small batch-size).However, if we allow the DFW algorithm to use a larger batch size, then the local approximation becomes more accurate and it can handle large values of η as well.

Interestingly, choosing a larger batch-size and a larger value of η can result in better generalization.

For instance, by using a batchsize of 256 (instead of 64) and η = 1, DFW obtains a test accuracy of 72.64% on CIFAR-100 with the DN architecture (SGD obtains 70.33% with the settings of BID3 ).

Our empirical evidence indicates that the initial learning rate can be a crucial hyper-parameter for good generalization.

We have observed in our experiments that such a choice of high learning rate provides a consistent improvement for convolutional neural networks: accurate minimization of the training objective with large initial steps usually leads to good generalization.

Furthermore, as mentioned in the previous section, it is sometimes beneficial to even increase the batch-size in order to be able to train the model using large initial steps.

In the case of recurrent neural networks, however, this effect is not as distinct.

Additional experiments on different recurrent architectures have showed variations in the impact of the learning rate and in the best-performing optimizer.

Further analysis would be required to understand the effects at play.

We have introduced DFW, an efficient algorithm to train deep neural networks.

DFW predominantly outperforms adaptive gradient methods, and obtains similar performance to SGD without requiring a hand-designed learning rate schedule.

We emphasize the generality of our framework in Section 3, which enables the training of deep neural networks to benefit from any advance on optimization algorithms for linear SVMs.

This framework could also be applied to other loss functions that yield efficiently solvable proximal problems.

In particular, our algorithm already supports the use of structured prediction loss functions BID28 BID30 , which can be used, for instance, for image segmentation.

We have mentioned the intricate relationship between optimization and generalization in deep learning.

This illustrates a major difficulty in the design of effective optimization algorithms for deep neural networks: the learning objective does not include all the regularization needed for good generalization.

We believe that in order to further advance optimization for deep neural networks, it is essential to alleviate this problem and expose a clear objective function to optimize.

This work was supported by the EPSRC grants AIMS CDT EP/L015987/1, Seebibyte EP/M013774/1, EP/P020658/1 and TU/B/000048, and by Yougov.

We also thank the Nvidia Corporation for the GPU donation.

For completeness, we prove results for our specific instance of Structural SVM problem.

We point out that the proofs of sections A.1, A.2 and A.3 are adaptations from BID5 .

Propositions are numbered according to their appearance in the paper.

In this section, we assume the loss L to be a hinge loss: DISPLAYFORM0 We suppose that we have received a sample (x, y).

We simplify the notation f (w, x) = f x (w) and L(u, y) = L y (u).

For simplicity of the notation, and without loss of generality, we consider the proximal problem obtained at time t = 0: DISPLAYFORM1 Let us define the classification task loss: DISPLAYFORM2 Using this notation, the multi-class hinge loss can be written as: DISPLAYFORM3 Indeed, we can successively write: DISPLAYFORM4 We are now going to re-write problem (9) as the sum of a quadratic term and a pointwise maximum of linear functions.

Forȳ ∈ Y, let us define: DISPLAYFORM5 Then we have that: DISPLAYFORM6 Therefore, problem (9) can be written as: DISPLAYFORM7 We notice that the term ρ(w 0 ) in b is a constant that does not depend on w norȳ, therefore we can simplify the expression of b to: DISPLAYFORM8 We introduce the following notation: DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 We will also use the indicator vector: 1 y ∈ R

, which is equal to 1 at index y and 0 elsewhere.

Lemma 1 (Dual Objective).

The Lagrangian dual of FORMULA20 is given by: DISPLAYFORM0 Given the dual variables α, the primal can be computed asŵ = −Aα.

Proof.

We derive the Lagrangian of the primal problem.

For that, we write the problem in the following equivalent ways: DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 (by strong duality).We can now write the KKT conditions of the inner minimization problem: DISPLAYFORM5 This gives α ∈ P andŵ = −Aα, since A = (ηaȳ)ȳ ∈Y by definition.

By injecting these constraints in (24), we obtain: DISPLAYFORM6 which finally gives the desired result.

Lemma 2 (Optimal Step-Size).

Suppose that we make a step in the direction of s ∈ P in the dual.

We define the corresponding primal variables w s = −As and λ s = b s, as well as λ = b α.

Then the optimal step-size is given by: DISPLAYFORM0 Proof.

Given the direction s, we take the step α + γ(s − α).

The new objective is given by: DISPLAYFORM1 In order to compute the optimal step-size, we compute the derivative of the above expression with respect to gamma, and set it to 0: DISPLAYFORM2 We can isolate the unique term containing γ: DISPLAYFORM3 This yields: DISPLAYFORM4 We can then inject the primal variables and simplify: DISPLAYFORM5

We present here the primal-dual algorithm that solves (9) using the previous results: DISPLAYFORM0 Initialization w 0 − Aα with α = 1 y 2: λ 1 = 0 Initialization b α with α = 1 y 3: t = 1 4: while not converged do Choose direction s t ∈ P (e.g. conditional gradient or smoothed loss) 6: DISPLAYFORM1 λ s = b s t 8: DISPLAYFORM2 Optimal-step-size 9: DISPLAYFORM3 t = t + 1 12: end while Note that when f x is linear, and when the search direction s is given by the conditional gradient, we recover the standard Frank-Wolfe algorithm for SVM BID5 .

We now provide some simplification to the steps 6, 8 and 9 of Algorithm 2 when a single step is taken, as is the case in the DFW algorithm.

This corresponds to the iteration t = 1.Proposition 2 (Cost per iteration).

If a single step is performed on the dual of (6), its conditional gradient is given by −∂ (ρ(w) + L y (f x (w))) wt .

The resulting update can be written as: DISPLAYFORM0 Proof.

It is known that for linear SVMs, the direction of the dual conditional gradient is given by the negative sub-gradient of the primal BID5 , Bach, 2015 .

We apply this result to the Taylor expansion of the network, which is the local model used for the proximal problem.

Then we have that at iteration t = 1, the conditional gradient is given by: DISPLAYFORM1 It now suffices to notice that a first-order Taylor expansion does not modify the derivative at its point of linearization: for a function φ, ∂T w0 φ(w) w0 = ∂φ(w) w0 .

By applying this property and the chain rule to FORMULA5 , we obtain that the conditional gradient is given by: DISPLAYFORM2 This completes the proof that the conditional gradient direction is given by a stochastic gradient.

We now prove equation FORMULA5 in the next lemma.

Lemma 3.

Suppose that we apply the Proximal Frank-Wolfe algorithm with a single step.

Let δ t = ∂ s t (f x,ȳ (w 0 ) − f x,y (w 0 ))ȳ ∈Y and r t = ∂ w ρ(w 0 ).

Then we can rewrite step 6 as: DISPLAYFORM3 In addition, we can simplify steps 8 and 9 of Algorithm 2 to: DISPLAYFORM4 DISPLAYFORM5 Proof.

Again, since we perform a single step of FW, we assume t = 1.

To prove equation FORMULA5 , we note that: DISPLAYFORM6 We point out the two following results: DISPLAYFORM7 and: w t − w 0 − w s = −ηr t + ηr t + ηδ t = ηδ t .(41) Since λ 1 = 0 by definition, equation FORMULA5 is obtained with a simple application of equations 40 and 41.

Finally, we prove equation 38 by writing: DISPLAYFORM8 A.6 SMOOTHING THE LOSS As pointed out in the paper, the SVM loss is non-smooth and has sparse derivatives, which can prevent the effective training of deep neural networks (Berrada et al., 2018) .

Partial linearization can solve this problem by locally smoothing the dual BID12 .

However, this would introduce a temperature hyper-parameter which is undesirable.

Therefore, we note that DFW can be applied with any direction that is feasible in the dual, since it computes an optimal step-size.

In particular, the following result states that we can use the well-conditioned and non-sparse gradient of cross-entropy.

Proposition 3.

The gradient of cross-entropy in the primal gives a feasible direction in the dual.

Furthermore, we can inexpensively detect when this feasible direction cannot provide any improvement in the dual, and automatically switch to the conditional gradient when that is the case.

For simplicity, we divide Proposition 3 into two distinct parts: first we show how the CE gradient gives a feasible direction in the dual, and then how it can be detected to be an ascent direction.

Lemma 4.

The gradient of cross-entropy in the primal gives a feasible direction in the dual.

In other words, the gradient of cross-entropy g in the primal is such that there exists a dual search direction s ∈ P verifying g = −As.

Proof.

We consider the vector of scores (f x,ȳ (w))ȳ ∈Y ∈ R

.

We compute its softmax: DISPLAYFORM0 j∈Y exp(fx,j (w)) ȳ∈Y .

Clearly, s ce ∈ P by property of the softmax.

Furthermore, by going back to the definition of A, one can easily verify that −As ce is exactly the primal gradient given by a backward pass through the cross-entropy loss instead of the hinge loss.

This concludes the proof.

The previous lemma has shown that we can use the gradient of cross-entropy as a feasible direction s ce in the dual.

The next step is to make it a dual ascent direction, that is a direction which always permits improvement on the dual objective (unless at the optimal point).

In what follows, we show that we can inexpensively (approximately) compute a sufficient condition for s ce to be an ascent direction.

If the condition is not satisfied, then we can automatically switch to use the subgradient of the hinge loss (which is known as an ascent direction in the dual).Lemma 5.

Let s ∈ P be a feasible direction in the dual, and v = (T w0 f x (w t )ȳ + ∆(ȳ, y) − T w0 f x (w t ) y )ȳ ∈Y ∈ R |Y| be the vector of augmented scores output by the linearized model.

Let us assume that we apply the single-step Proximal Frank-Wolfe algorithm (that is, we have t = 1), and that ρ is a non-negative function.

Then s v > 0 is a sufficient condition for s to be an ascent direction in the dual.

Proof.

Let s ∈ P, v = (T w0 f x (w t )ȳ + ∆(ȳ, y) − T w0 f x (w t ) y )ȳ ∈Y .

By definition, we have that: DISPLAYFORM1 Therefore: DISPLAYFORM2 ⇐⇒ (As) (w t − w 0 ) + ηs b − ηT w0 ρ(w) > 0, (since s ∈ P and η > 0) DISPLAYFORM3 We have just shown that if s v > 0, then γ t > 0.

Since γ t is an optimal step-size, this indicates that s is an ascent direction (we would obtain γ t = 0 for a direction s that cannot provide improvement).Approximate Condition.

In practice, we consider that T w0 f x (w t ) f x (w 0 ).

Indeed, for t = 1, we have that T w0 f x (w) − f x (w 0 ) = O( w t − w 0 ), and w t − w 0 = η∂ w ρ(w 0 )) , which is typically very small (we use a weight decay coefficient in the order of 1e −4in our experimental settings).

Therefore, we replace T w0 f x (w) by f x (w 0 ) in the above criterion, which becomes inexpensive since f x (w 0 ) is already computed by the forward pass.

As can be seen in the previous primal-dual algorithms, taking a step in the dual can be decomposed into two stages: the initialization and the movement along the search direction.

The initialization step is not informative about the optimization problem.

Therefore, we discard it from the momentum velocity, and only accumulate the step along the conditional gradient (scaled by γ t η).

This results in the following velocity update: DISPLAYFORM0

In this section we provide the convergence plots of the different algorithms on the CIFAR data sets without data augmentation.

In some cases the training performance can show some oscillations.

We emphasize that this is the result of cross-validating the initial learning rate based on the validation set performance: sometimes a better-behaved convergence would be obtained on the training set with a lower learning rate.

However this lower learning rate is not selected because it does not provide the best validation performance.

@highlight

We train neural networks by locally linearizing them and using a linear SVM solver (Frank-Wolfe) at each iteration.