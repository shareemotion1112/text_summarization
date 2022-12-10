We study adversarial robustness of neural networks from a margin maximization perspective, where margins are defined as the distances from inputs to a classifier's decision boundary.

Our study shows that maximizing margins can be achieved by minimizing the adversarial loss on the decision boundary at the "shortest successful perturbation", demonstrating a close connection between adversarial losses and the margins.

We propose Max-Margin Adversarial (MMA) training to directly maximize the margins to achieve adversarial robustness.

Instead of adversarial training with a fixed $\epsilon$, MMA offers an improvement by enabling adaptive selection of the "correct" $\epsilon$ as the margin individually for each datapoint.

In addition, we rigorously analyze adversarial training with the perspective of margin maximization, and provide an alternative interpretation for adversarial training, maximizing either a lower or an upper bound of the margins.

Our experiments empirically confirm our theory and demonstrate MMA training's efficacy on the MNIST and CIFAR10 datasets w.r.t.

$\ell_\infty$ and $\ell_2$ robustness.

Figure 1: Illustration of decision boundary, margin, and shortest successful perturbation on application of an adversarial perturbation.

Despite their impressive performance on various learning tasks, neural networks have been shown to be vulnerable to adversarial perturbations (Szegedy et al., 2013; Biggio et al., 2013 ).

An artificially constructed imperceptible perturbation can cause a significant drop in the prediction accuracy of an otherwise accurate network.

The level of distortion is measured by the magnitude of the perturbations (e.g. in ∞ or 2 norms), i.e. the distance from the original input to the perturbed input.

Figure 1 shows an example, where the classifier changes its prediction from panda to bucket when the input is perturbed from the blue sample point to the red one.

Figure 1 also shows the natural connection between adversarial robustness and the margins of the data points, where the margin is defined as the distance from a data point to the classifier's decision boundary.

Intuitively, the margin of a data point is the minimum distance that x has to be perturbed to change the classifier's prediction.

Thus, the larger the margin is, the farther the distance from the input to the decision boundary of the classifier is, the more robust the classifier is w.r.t.

this input.

Although naturally connected to adversarial robustness, "directly" maximizing margins has not yet been thoroughly studied in the adversarial robustness literature.

Instead, the method of minimax adversarial training (Madry et al., 2017; Huang et al., 2015) is arguably the most common defense to adversarial perturbations due to its effectiveness and simplicity.

Adversarial training attempts to minimize the maximum loss within a fixed sized neighborhood about the training data using projected gradient descent (PGD).

Despite advancements made in recent years (Hendrycks et al., 2019; Zhang et al., 2019a; Shafahi et al., 2019; Zhang et al., 2019b; Stanforth et al., 2019; Carmon et al., 2019) , adversarial training still suffers from a fundamental problem, the perturbation length has to be set and is fixed throughout the training process.

In general, the setting of is arbitrary, based on assumptions on whether perturbations within the defined ball are "imperceptible" or not.

Recent work (Guo et al., 2018; Sharma et al., 2019) has demonstrated that these assumptions do not consistently hold true, commonly used settings assumed to only allow imperceptible perturbations in fact do not.

If is set too small, the resulting models lack robustness, if too large, the resulting models lack in accuracy.

Moreover, individual data points may have different intrinsic robustness, the variation in ambiguity in collected data is highly diverse, and fixing one for all data points across the whole training procedure is likely suboptimal.

Instead of further improving adversarial training with a fixed perturbation magnitude, we revisit adversarial robustness from the margin perspective, and propose Max-Margin Adversarial (MMA) training, a practical algorithm for direct input margin maximization.

By directly maximizing margins calculated for each data point, MMA training allows for optimizing the "current robustness" of the data, the "correct" at this point in training for each sample individually, instead of robustness w.r.t.

a predefined magnitude.

While it is intuitive that one can achieve the greatest possible robustness by maximizing the margin of a classifier, this maximization has technical difficulties.

In Section 2, we overcome these difficulties and show that margin maximization can be achieved by minimizing a classification loss w.r.t.

model parameters, at the "shortest successful perturbation".

This makes gradient descent viable for margin maximization, despite the fact that model parameters are entangled in the constraints.

We further analyze adversarial training (Madry et al., 2017; Huang et al., 2015) from the perspective of margin maximization in Section 3.

We show that, for each training example, adversarial training with fixed perturbation length is maximizing a lower (or upper) bound of the margin, if is smaller (or larger) than the margin of that training point.

As such, MMA training improves adversarial training, in the sense that it selects the "correct" , the margin value for each example.

Finally in Section 4, we test and compare MMA training with adversarial training on MNIST and CIFAR10 w.r.t.

∞ and 2 robustness.

Our method achieves higher robustness accuracies on average under a variety of perturbation magnitudes, which echoes its goal of maximizing the average margin.

Moreover, MMA training automatically balances accuracy vs robustness while being insensitive to its hyperparameter setting, which contrasts sharply with the sensitivity of standard adversarial training to its fixed perturbation magnitude.

MMA trained models not only match the performance of the best adversarially trained models with carefully chosen training under different scenarios, it also matches the performance of ensembles of adversarially trained models.

In this paper, we focus our theoretical efforts on the formulation for directly maximizing the input space margin, and understanding the standard adversarial training method from a margin maximization perspective.

We focus our empirical efforts on thoroughly examining our MMA training algorithm, comparing with adversarial training with a fixed perturbation magnitude.

Although not often explicitly stated, many defense methods are related to increasing the margin.

One class uses regularization to constrain the model's Lipschitz constant (Cisse et al., 2017; Ross & Doshi-Velez, 2017; Hein & Andriushchenko, 2017; Tsuzuku et al., 2018) , thus samples with small loss would have large margin since the loss cannot increase too fast.

If the Lipschitz constant is merely regularized at the data points, it is often too local and not accurate in a neighborhood.

When globally enforced, the Lipschitz constraint on the model is often too strong that it harms accuracy.

So far, such methods have not achieved strong robustness.

There are also efforts using first-order approximation to estimate and maximize input space margin (Elsayed et al., 2018; Sokolic et al., 2017; Matyasko & Chau, 2017) .

Similar to local Lipschitz regularization, the reliance on local information often does not provide accurate margin estimation and efficient maximization.

Such approaches have also not achieved strong robustness at this point in time.

We defer some detailed discussions on related works to Appendix B, including a comparison between MMA training and SVM.

We focus on K-class classification problems.

Denote S = {x i , y i } as the training set of inputlabel data pairs sampled from data distribution D. We consider the classifier as a score function

, parametrized by θ, which assigns score f i θ (x) to the i-th class.

The predicted label of x is then decided byŷ = arg max i f

θ (x, y) = I(ŷ = y) be the 0-1 loss indicating classification error, where I(·) is the indicator function.

For an input (x, y), we define its margin w.r.t.

the classifier f θ (·) as:

Figure 2: A 1-D example on how margin is affected by decreasing the loss at different locations.

Proposition 2.1.

Let (δ) = δ .

Given a fixed θ, assume that δ * is unique, (δ) and L(δ, θ) are C 2 functions in a neighborhood of (θ, δ * ), and the matrix

is a scalar.

Remark 2.1.

By Proposition 2.1, the margin's gradient w.r.t.

to the model parameter θ is proportional to the loss' gradient w.r.t.

θ at δ * , the shortest successful perturbation.

Therefore to perform gradient ascent on margin, we just need to find δ * and perform gradient descent on the loss.

Margin maximization for non-smooth loss and norm: Proposition 2.1 requires the loss function and the norm to be C 2 at δ * .

This might not be the case for many functions used in practice, e.g. ReLU networks and the ∞ norm.

Our next result shows that under a weaker condition of directional differentiability (instead of C 2 ), learning θ to maximize the margin can still be done by decreasing L(θ, δ * ) w.r.t.

θ, at θ = θ 0 .

Due to space limitations, we only present an informal statement here.

Rigorous statements can be found in the Appendix A.2.

Proposition 2.2.

Let δ * be unique and L(δ, θ) be the loss of a deep ReLU network.

There exists some direction v in the parameter space, such that the loss L(δ, θ)| δ=δ * can be reduced in the direction of v. Furthermore, by reducing L(δ, θ)| δ=δ * , the margin is also guaranteed to be increased.

Figure 2 illustrates the relationship between the margin and the adversarial loss with an imaginary example.

Consider a 1-D example in Figure 2 (a) , where the input example x is a scalar.

We perturb x in the positive direction with perturbation δ.

As we fix (x, y), we overload L(δ, θ) = L LM θ (x + δ, y), which is monotonically increasing on δ, namely larger perturbation results in higher loss.

Let L(·, θ 0 ) (the dashed curve) denote the original function before an update step, and δ * 0 = arg min L(δ,θ0)≥0 δ denote the corresponding margin (same as shortest successful perturbation in 1D).

As shown in Figure 2 (b), as the parameter is updated to θ 1 such that L(δ * 0 , θ 1 ) is reduced, the new margin δ * 1 = arg min L(δ,θ1)≥0 δ is enlarged.

Intuitively, a reduced value of the loss at the shortest successful perturbation leads to an increase in margin.

In practice, we find the gradients of the "logit margin loss" L LM θ to be unstable.

The piecewise nature of the L LM θ loss can lead to discontinuity of its gradient, causing large fluctuations on the boundary between the pieces.

It also does not fully utilize information provided by all the logits.

In our MMA algorithm, we instead use the "soft logit margin loss" (SLM)

which serves as a surrogate loss to the "logit margin loss" L LM θ (x, y) by replacing the the max function by the LogSumExp (sometimes also called softmax) function.

One immediate property is that the SLM loss is smooth and convex (w.r.t.

logits).

The next proposition shows that SLM loss is a good approximation to the LM loss.

.

(6) Therefore, to simplify the learning algorithm, we perform gradient descent on model parameters using L CE θ (x + δ * , y).

As such, we use L CE θ on both clean and adversarial examples, which in practice stabilizes training:

where δ * = arg min L SLM θ (x+δ,y)≥0 δ is found with the SLM loss, and H θ = {i : d θ (x i , y i ) < d max } is the set of examples that have margins smaller than the hinge threshold.

To implement MMA, we still need to find the δ * , which is intractable in general settings.

We propose an adaptation of the projected gradient descent (PGD) (Madry et al., 2017) attack to give an approximate solution of δ * , the Adaptive Norm Projective Gradient Descent Attack (AN-PGD).

In AN-PGD, we apply PGD on an initial perturbation magnitude init to find a norm-constrained perturbation δ 1 , then we search along the direction of δ 1 to find a scaled perturbation that gives L = 0, we then use this scaled perturbation to approximate * .

Note that AN-PGD here only serves as an algorithm to give an approximate solution of δ * , and it can be decoupled from the remaining parts of MMA training.

Other attacks that can serve a similar purpose can also fit into our MMA training framework, e.g. the Decoupled Direction and Norm (DDN) attack (Rony et al., 2018) .

Algorithm 1 describes the Adaptive Norm PGD Attack (AN-PGD) algorithm.

Remark 2.3.

Finding the δ * in Proposition 2.2 and Proposition 2.1 requires solving a non-convex optimization problem, where the optimality cannot be guaranteed in practice.

Previous adversarial training methods, e.g. Madry et al. (2017) , suffer the same problem.

Nevertheless, as we show later in Figure 4 , our proposed MMA training algorithm does achieve the desired behavior of maximizing the margin of each individual example in practice.

Binary search to find , the zero-crossing of L(x + ηδ u , y) w.r.t.

η, η ∈ [0, δ 1 ) 7: end if

Figure 3: Visualization of loss landscape in the input space for MMA and PGD trained models.

In practice, we observe that when the model is only trained with the objective function in Eq. (7), the input space loss landscape is very flat, which makes PGD less efficient in finding δ * for training, as shown in Figure 3 .

Here we choose 50 examples from both the training and test sets respectively, then perform the PGD attack with = 8/255 and keep those failed perturbations.

For each, we linearly interpolate 9 more points between the original example and the perturbed, and plot their logit margin losses.

In each sub- figure To alleviate this issue, we add an additional clean loss term to the MMA objective in Eq. (7) to lower the loss on clean examples, such that the input space loss landscape is steeper.

Specifically, we use the following combined loss

The model trained with this combined loss and d max = 32 is the MMA-32 shown in Figure 3c .

Adding the clean loss is indeed effective.

Most of the loss curves are more tilted, and the losses of perturbed examples are lower.

We use L CB θ for MMA training in the rest of the paper due to its higher performance.

A more detailed comparison between L CB θ and L MMA θ is delayed to Appendix E.

2.5 THE MMA TRAINING ALGORITHM Algorithm 2 summarizes our practical MMA training algorithm.

During training for each minibatch, we 1) separate it into 2 batches based on if the current prediction matches the label; 2) find δ * for each example in the "correct batch"; 3) calculate the gradient of θ based on Eqs. (7) and (8).

Algorithm 2 Max-Margin Adversarial Training.

Inputs: The training set {(x i , y i )}.

Outputs: the trained model f θ (·).

Parameters: contains perturbation lengths of training data.

min is the minimum perturbation length.

max is the maximum perturbation length.

A(x, y, init ) represents the approximate shortest successful perturbation returned by an algorithm A (e.g. AN-PGD) on the data example (x, y) and at the initial norm init .

1: Randomly initialize the parameter θ of model f , and initialize every element of as min 2: repeat 3:

Make predictions on B and into two: wrongly predicted B 0 and correctly predicted B 1

Initialize an empty batch B adv 1 6:

Retrieve perturbation length i from 8:

Update the i in as δ *

, the combined loss on B 0 , B 1 , and B adv 1 , w.r.t.

θ, according to Eqs. (7) and (8) 12:

Perform one step gradient step update on θ 13: until meet training stopping criterion 3 UNDERSTANDING ADVERSARIAL TRAINING THROUGH MARGIN MAXIMIZATION Through our development of MMA training in the last section, we have shown that margin maximization is closely related to adversarial training with the optimal perturbation length δ * .

In this section, we further investigate the behavior of adversarial training in the perspective of margin maximization.

Adversarial training (Huang et al., 2015; Madry et al., 2017) minimizes the "worst-case" loss under a fixed perturbation magnitude , as follows.

Looking again at Figure 2 , we can see that an adversarial training update step does not necessarily increase the margin.

In particular, as we perform an update to reduce the value of loss at the fixed perturbation , the parameter is updated from θ 0 to θ 1 .

After this update, we can imagine two different scenarios for the updated loss functions L θ1 (·) (the solid curve) in Figure 2 (c) and (d).

In both (c) and (d), L θ1 ( ) is decreased by the same amount.

However, the margin is increased in (c) with

Formalizing the intuitive analysis, we present two theorems connecting adversarial training and margin maximization.

For brevity, fixing

Remark 3.1.

In other words, adversarial training, with the logit margin loss and a fixed perturbation length 1) exactly maximizes the margin, if is equal to the margin; 2) maximizes a lower bound of the margin, if is less than the margin; 3) maximizes an upper bound of the margin, if is greater than the margin.

Next we look at adversarial training with the cross-entropy loss (Madry et al., 2017) through the connection between cross-entropy and the soft logit margin loss from Proposition 2.4.

We first look at adversarial training on the SLM loss.

Fixing {(x, y)}, let d

Corollary 3.1.

Assuming an update from adversarial training changes θ 0 to θ 1 , such that

Remark 3.2.

In other words, if is less than or equal to the SLM-margin, adversarial training, with the SLM loss and a fixed perturbation length , maximizes a lower bound of the SLM-margin, thus a lower bound of the margin.

Recall Proposition 2.4 shows that L CE θ and L SLM θ have the same gradient direction w.r.t.

both the model parameter and the input.

In adversarial training (Madry et al., 2017) , the PGD attack only uses the gradient direction w.r.t.

the input, but not the gradient magnitude.

Therefore, in the inner maximization loop, using the SLM and CE loss will result in the same approximate δ On the other hand, when is larger then the margin, such a relation no longer exists.

We can anticipate that when is too large, adversarial training is likely maximizing an upper bound of the margin, which might not necessarily increase the margin.

This suggests that for adversarial training with a large , starting with a smaller then gradually increasing it could help, since the lower bound of the margin is maximized at the start of training.

Results in Sections 4.1 and 4.2 corroborate exactly with this theoretical prediction.

We empirically examine several hypotheses and compare MMA training with different adversarial training algorithms on the MNIST and CIFAR10 datasets under ∞ / 2 -norm constrained perturbations.

Due to space limitations, we mainly present results on CIFAR10-∞ for representative models in Table 1 .

Full results are in Table 2 to 13 in Appendix F. Implementation details are also left to the appendix, including neural network architecture, training and attack hyperparameters.

Our results confirm our theory and show that MMA training is stable to its hyperparameter d max , and balances better among various attack lengths compared to adversarial training with fixed perturbation magnitude.

This suggests that MMA training is a better choice for defense when the perturbation length is unknown, which is often the case in practice.

Measuring Adversarial Robustness: We use the robust accuracy under multiple projected gradient descent (PGD) attacks (Madry et al., 2017) as the robustness measure.

Specifically, given an example, each model is attacked by both repeated randomly initialized whitebox PGD attacks and numerous transfer attacks, generated from whitebox PGD attacking other models.

If any one of these attacks succeed, then the model is considered "not robust under attack" on this example.

For each dataset-norm setting and for each example, under a particular magnitude , we first perform N randomly initialized whitebox PGD attacks on each individual model, then use N · (m − 1) PGD attacks from all the other models to perform transfer attacks, where m is the total number of models considered under each setting.

In our experiments, we use N = 10 for models trained on CIFAR10, thus the total number of the "combined" (whitebox and transfer) set of attacks is 320 for CIFAR10-∞ (m = 32).

3 We use ClnAcc for clean accuracy, AvgAcc for the average over both clean accuracy and robust accuracies at different 's, AvgRobAcc for the average over only robust accuracies under attack.

As discussed in Section 3, MMA training enlarges margins of all training points,while PGD training, by minimizing the adversarial loss with a fixed , might fail to enlarge margins for points with initial margins smaller than .

This is because when d θ (x, y) < , PGD training is maximizing an upper bound of d θ (x, y), which may not necessarily increase the margin.

To verify this, we track how the margin distribution changes during training processes in two models under the CIFAR10-2 4 case, Specifically, we randomly select 500 training points, and measure their margins after each training epoch.

We use the norm of the perturbation, generated by the 1000-step DDN attack (Rony et al., 2018) , to approximate the margin.

The results are shown in Figure 4 , where each subplot is a histogram (rotated by 90 • ) of margin values.

For the convenience of comparing across epochs, we use the vertical axis to indicate margin value, and the horizontal axis for counts in the histogram.

The number below each subplot is the corresponding training epoch.

Margins mostly concentrate near 0 for both models at the beginning.

As training progresses, both enlarge margins on average.

However, in PGD training, a portion of the margins stay close to 0 throughout the training process, at the same time, also pushing some margins to be even higher than 2.5, likely because PGD training continues to maximize lower bounds of this subset of the total margins, as we discussed in Section 3, the value that the PGD-2.5 model is trained for.

MMA training, on the other hand, does not "give up" on those data points with small margins.

At the end of training, 37.8% of the data points for PGD-2.5 have margins smaller than 0.05, while only 20.4% for MMA.

As such, PGD training enlarges the margins of "easy data" which are already robust enough, but "gives up" on "hard data" with small margins.

Instead, MMA training pushes the margin of every data point, by finding the proper .

In general, when the attack magnitude is unknown, MMA training is more capable in achieving a better balance between small and large margins, and thus a better balance among adversarial attacks with various as a whole.

Our previous analysis in Section 3 suggests that when the fixed perturbation magnitude is small, PGD training increases the lower bound of the margin.

On the other hand, when is larger than the margin, PGD training does not necessarily increase the margin.

This is indeed confirmed by our experiments.

PGD training fails at larger , in particular = 24/255 for the CIFAR10-∞ as shown in Table 1 .

We can see that PGD-24's accuracies at all test 's are around 10%.

Aiming to improve PGD training, we propose a variant of PGD training, named PGD with Linear Scaling (PGDLS), in which we grow the perturbation magnitude from 0 to the fixed magnitude linearly in 50 epochs.

According to our theory, gradually increasing the perturbation magnitude could avoid picking a that is larger than the margin, thus managing to maximizing the lower bound of the margin rather than its upper bound, which is more sensible.

It can also be seen as a "global magnitude scheduling" shared by all data points, which is to be contrasted to MMA training that gives magnitude scheduling for each individual example.

We use PGDLS-to represent these models and show their performances also in Table 1 .

We can see that PGDLS-24 is trained successfully, whereas PGD-24 fails.

At = 8 or 16, PGDLS also performs similar or better than PGD training, confirming the benefit of training with small perturbation at the beginning.

4.3 COMPARING MMA TRAINING WITH PGD TRAINING From the first three columns in Table 1 , we can see that MMA training is very stable with respect to its hyperparameter, the hinge threshold d max .

When d max is set to smaller values (e.g. 12 and 20), MMA models attain good robustness across different attacking magnitudes, with the best clean accuracies in the comparison set.

When d max is large, MMA training can still learn a reasonable model that is both accurate and robust.

For MMA-32, although d max is set to a "impossible-tobe-robust" level at 32/255, it still achieves 84.36% clean accuracy and 47.18% robust accuracy at 8/255, thus automatically "ignoring" the demand to be robust at larger 's, including 20, 24, 28 and 32, recognizing its infeasibility due to the intrinsic difficulty of the problem.

In contrast, PGD trained models are more sensitive to their specified fixed perturbation magnitude.

In terms of the overall performance, we notice that MMA training with a large d max , e.g. 20 or 32, achieves high AvgAcc values, e.g. 28.86% or 29.39%.

However, for PGD training to achieve a similar performance, needs to be carefully picked (PGD-16 and PGDLS-16), and their clean accuracies suffer a significant drop.

We also compare MMA models with ensemble of PGD trained models.

PGD-ens/PGDLS-ens represents the ensemble of PGD/PGDLS trained models with different (s).

The ensemble produces a prediction by performing a majority vote on label predictions, and using the softmax scores as the tie breaker.

MMA training achieves similar performance compared to the ensembled PGD models.

PGD-ens maintains a good clean accuracy, but it is still marginally outperformed by MMA-32 w.r.t.

robustness at various 's. Further note that 1) the ensembling requires significantly higher computation costs both at training and test times; 2) Unlike attacking individual models, attacking ensembles is still relatively unexplored in the literature, thus our whitebox PGD attacks on the ensembles may not be sufficiently effective; 5 and 3) as shown in Appendix F, for MNIST-∞ / 2 , MMA trained models significantly outperform the PGD ensemble models.

Testing on gradient free attacks: As a sanity check for gradient obfuscation (Athalye et al., 2018), we also performed the gradient-free SPSA attack (Uesato et al., 2018) , to all our ∞ -MMA trained models on the first 100 test examples.

We find that, in all cases, SPSA does not compute adversarial examples successfully when gradient-based PGD did not.

In this paper, we proposed to directly maximize the margins to improve adversarial robustness.

We developed the MMA training algorithm that optimizes the margins via adversarial training with perturbation magnitude adapted both throughout training and individually for the distinct datapoints in the training dataset.

Furthermore, we rigorously analyzed the relation between adversarial training and margin maximization.

Our experiments on CIFAR10 and MNIST empirically confirmed our theory and demonstrate that MMA training outperforms adversarial training in terms of sensitivity to hyperparameter setting and robustness to variable attack lengths, suggesting MMA is a better choice for defense when the adversary is unknown, which is often the case in practice.

Proof.

Recall (δ) = δ .

Here we compute the gradient for d θ (x, y) in its general form.

Consider the following optimization problem:

where ∆(θ) = {δ : L θ (x+δ, y) = 0}, and L(δ, θ) are both C 2 functions 6 .

Denotes its Lagrangian by L(δ, λ), where L(δ, λ) = (δ) + λL θ (x + δ, y) For a fixed θ, the optimizer δ * and λ * must satisfy the first-order conditions (FOC)

Put the FOC equations in vector form,

Note that G is C 1 continuously differentiable since and L(δ, θ) are C 2 functions.

Furthermore, the Jacobian matrix of G w.r.t (δ, λ) is

which by assumption is full rank.

Therefore, by the implicit function theorem, δ * and λ * can be expressed as a function of θ, denoted by δ * (θ) and λ * (θ).

where the second equality is by Eq. (10).

The implicit function theorem also provides a way of computing

which is complicated involving taking inverse of the matrix

Here we present a relatively simple way to compute this gradient.

Note that by the definition of

and δ * (θ) is a differentiable implicit function of θ restricted to this level set.

Differentiate with w.r.t.

θ on both sides:

Combining Eq. (11) and Eq. (12),

Lastly, note that

6 Note that a simple application of Danskin's theorem would not be valid as the constraint set ∆(θ) depends on the parameter θ.

Therefore, one way to calculate λ * (θ) is by

We provide more detailed and formal statements of Proposition 2.2.

For brevity, consider a K-layers fully-connected ReLU network, f (θ; x) = f θ (x) as a function of θ.

where the D k are diagonal matrices dependent on ReLU's activation pattern over the layers, and W k 's and V are the weights (i.e. θ).

Note that f (θ; x) is a piecewise polynomial functions of θ with finitely many pieces.

We further define the directional derivative of a function g, along the direction of v, to be:

t .

Note that for every direction v, there exists α > 0 such that f (θ; x) is a polynomial restricted to a line segment [θ, θ + α v].

Thus the above limit exists and the directional derivative is well defined.

We first show the existence of v and t for l(

Proposition A.1.

For > 0, t ∈ [0, 1], and θ 0 ∈ Θ, there exists a direction v ∈ Θ, such that the derivative of l θ0, v, (t) exists and is negative.

Moreover, it is given by

is negative.

The Danskin theorem provides a way to compute the directional gradient along this direction v. We basically apply a version of Danskin theorem for directional absolutely continuous maps and semicontinuous maps (Yu, 2012).

1. the constraint set {δ : δ ≤ } is compact; 2.

L(θ 0 + t v; x + δ, y) is piecewise Lipschitz and hence absolutely continuous (an induction argument on the integral representation over the finite pieces).

3. L(θ 0 + t v; x + δ, y) is continuous on both δ and along the direction v and hence upper semi continuous.

Hence we can apply Theorem 1 in Yu (2012).

Therefore, for any > 0, if θ 0 is not a local minimum, then there exits a direction d, such that for

Our next proposition provides an alternative way to increase the margin of f θ .

Proposition A.2.

Assume f θ0 has a margin 0 , and θ 1 such that l θ0, v, 0 (t) ≤ l θ1, v, 0 (0) , then f θ1 has a larger margin than 0 .

Proof.

Since f θ0 has a margin 0 , thus max

To see the equality (constraint not binding), we use the following argument.

The envolope function's continuity is passed from the continuity of L(θ 0 ; x + δ, y).

The inverse image of a closed set under continuous function is closed.

If δ * lies in the interior of max δ ≤ 0 L v, (θ 0 ; x + δ, y) ≥ 0, we would have a contradiction.

Therefore the constraint is not binding, due to the continuity of the envolope function.

By Eq. (15), max δ ≤ 0 L(θ 1 ; x + δ, y) < 0.

So for the parameter θ 1 , f θ1 has a margin 1 > 0 .

Therefore, the update θ 0 → θ 1 = θ 0 + t v increases the margin of f θ .

≤ log(exp(

The following lemma helps relate the objective of adversarial training with that of our MMA training.

Here, we denote L θ (x + δ, y) as L(δ, θ) for brevity.

Lemma A.1.

Given (x, y) and θ , assume that L(δ, θ) is continuous in δ, then for ≥ 0, and

Proof.

Eq. (23).

We prove this by contradiction.

Suppose max δ ≤ L(δ, θ) > ρ.

When = 0, this violates our asssumption ρ ≥ L(0, θ) in the theorem.

So assume > 0.

Since L(δ, θ) is a continuous function defined on a compact set, the maximum is attained byδ such that δ ≤ and L(δ, θ) > ρ.

Note that L(δ, θ)) is continuous and ρ ≥ L(0, θ), then there existsδ ∈ 0,δ i.e. the line segment connecting 0 andδ, such that δ < and L(δ, θ) = ρ.

This follows from the intermediate value theorem by restricting L(δ, θ) onto 0,δ .

This contradicts min L(δ,θ)≥ρ δ = .

If max δ ≤ L(δ, θ) < ρ, then {δ :

δ ≤ } ⊂ {δ : L(δ, θ) < ρ}. Every point p ∈ {δ : δ ≤ } is in the open set {δ : L(δ, θ) < ρ}, there exists an open ball with some radius r p centered at p such that B rp ⊂ {δ : L(δ, θ) < ρ}. This forms an open cover for {δ : δ ≤ }.

Since {δ : δ ≤ } is compact, there is an open finite subcover U such that:

Since U is finite, there exists h > 0 such that {δ :

Eq. (24).

Assume that min L(δ,θ)≥ρ δ > , then {δ : L(δ, θ) ≥ ρ} ⊂ {δ : δ > }.

Taking complementary set of both sides, {δ : δ ≤ } ⊂ {δ : L(δ, θ) < ρ}.

Therefore, by the compactness of {δ : δ ≤ }, max δ ≤ L(δ, θ) < ρ, contradiction.

A.5 PROOF OF THEOREM 3.1

We first prove that ∀ , *

For 1), = d θ0 .

By definition of margin in Eq.

(1), we have ρ

We next discuss a few related works in details.

First-order Large Margin: Previous works (Elsayed et al., 2018; Sokolic et al., 2017; Matyasko & Chau, 2017) have attempted to use first-order approximation to estimate the input space margin.

For first-order methods, the margin will be accurately estimated when the classification function is linear.

MMA's margin estimation is exact when the shortest successful perturbation δ * can be solved, which is not only satisfied by linear models, but also by a broader range of models, e.g. models that are convex w.r.t.

input x. This relaxed condition could potentially enable more accurate margin estimation which improves MMA training's performance.

(Cross-)Lipschitz Regularization: Tsuzuku et al. (2018) enlarges their margin by controlling the global Lipschitz constant, which in return places a strong constraint on the model and harms its learning capabilities.

Instead, our method, alike adversarial training, uses adversarial attacks to estimate the margin to the decision boundary.

With a strong method, our estimate is much more precise in the neighborhood around the data point, while being much more flexible due to not relying on a global Lipschitz constraint.

Hard-Margin SVM (Vapnik, 2013) in the separable case:

Assuming that all the training examples are correctly classified and using our notations on general classifiers, the hard-margin SVM objective can be written as: max

On the other hand, under the same "separable and correct" assumptions, MMA formulation in Eq. (3) can be written as

which is maximizing the average margin rather than the minimum margin in SVM.

Note that the theorem on gradient calculation of the margin in Section 2.1 also applies to the SVM formulation of differentiable functions.

Because of this, we can also use SGD to solve the following "SVM-style" formulation:

As our focus is using MMA to improve adversarial robustness which involves maximizing the average margin, we delay the maximization of minimum margin to future work.

For 2 robustness, we also compare to models adversarially trained on the "Decoupled Direction and Norm" (DDN) attack (Rony et al., 2018) , which is concurrent to our work.

The DDN attack aims to achieve successful perturbation with minimal 2 norm, thus, DDN could be used as a drop-in replacement for the AN-PGD attack for MMA training.

We performed evaluations on the downloaded 7 DDN trained models.

The DDN MNIST model is a larger ConvNet with similar structure to our LeNet5, and the CIFAR10 model is wideresnet-28-10, which is similar but larger than the wideresnet-28-4 that we use.

DDN training, "training on adversarial examples generated by the DDN attack", differs from MMA in the following ways.

When the DDN attack does not find a successful adversarial example, it returns the clean image, and the model will use it for training.

In MMA, when a successful adversarial example cannot be found, it is treated as a perturbation with very large magnitude, which will be ignored by the hinge loss when we calculate the gradient for this example.

Also, in DDN training, there exists a maximum norm of the perturbation.

This maximum norm constraint does not exist for MMA training.

When a perturbation is larger than the hinge threshold, it will be ignored by the hinge loss.

There also are differences in training hyperparameters, which we refer the reader to Rony et al. (2018) for details.

Despite these differences, in our experiments MMA training achieves similar performances under the 2 cases.

While DDN attack and training only focus on 2 cases, we also show that the MMA training framework provides significant improvements over PGD training in the ∞ case.

We train LeNet5 models for the MNIST experiments and use wide residual networks (Zagoruyko & Komodakis, 2016) with depth 28 and widen factor 4 for all the CIFAR10 experiments.

For all the experiments, we monitor the average margin from AN-PGD on the validation set and choose the model with largest average margin from the sequence of checkpoints during training.

The validation set contains first 5000 images of training set.

It is only used to monitor training progress and not used in training.

Here all the models are trained and tested under the same type of norm constraints, namely if trained on ∞ , then tested on ∞ ; if trained on 2 , then tested on 2 .

The LeNet5 is composed of 32-channel conv filter + ReLU + size 2 max pooling + 64-channel conv filter + ReLU + size 2 max pooling + fc layer with 1024 units + ReLU + fc layer with 10 output classes.

We do not preprocess MNIST images before feeding into the model.

For training LeNet5 on all MNIST experiments, for both PGD and MMA training, we use the Adam optimizer with an initial learning rate of 0.0001 and train for 100000 steps with batch size 50.

In our initial experiments, we tested different initial learning rate at 0.0001, 0.001, 0.01, and 0.1 and do not find noticeable differences.

We use the WideResNet-28-4 as described in Zagoruyko & Komodakis (2016) for our experiments, where 28 is the depth and 4 is the widen factor.

We use "per image standardization" 8 to preprocess CIFAR10 images, following Madry et al. (2017) .

For training WideResNet on CIFAR10 variants, we use stochastic gradient descent with momentum 0.9 and weight decay 0.0002.

We train 50000 steps in total with batch size 128.

The learning rate is set to 0.3 at step 0, 0.09 at step 20000, 0.03 at step 30000, and 0.009 at step 40000.

This setting is the same for PGD and MMA training.

In our initial experiments, we tested different learning rate at 0.03, 0.1, 0.3, and 0.6, and kept using 0.3 for all our later experiments.

We also tested a longer training schedule, following Madry et al. (2017) , where we train 80000 steps with different learning rate schedules.

We did not observe improvement with this longer training, therefore kept using the 50000 steps training.

For models trained on MNIST, we use 40-step PGD attack with the soft logit margin (SLM) loss defined in Section 3, for CIFAR10 we use 10 step-PGD, also with the SLM loss.

For both MNIST and CIFAR10, the step size of PGD attack at training time is 2.5 number of steps · In AN-PGD, we always perform 10 step binary search after PGD, with the SLM loss.

For AN-PGD, the maximum perturbation length is always 1.05 times the hinge threshold: max = 1.05d max .

The initial perturbation length at the first epoch, init , have different values under different settings.

init = 0.5 for MNIST 2 , init = 0.1 for MNIST ∞ , init = 0.5 for CIFAR10 2 , init = 0.05 for CIFAR10 2 .

In epochs after the first, init will be set to the margin of the same example from last epoch.

Trained models: Various PGD/PGDLS models are trained with different perturbation magnitude , denoted by PGD-or PGDLS-.

PGD-ens/PGDLS-ens represents the ensemble of PGD/PGDLS trained models with different 's. The ensemble makes prediction by majority voting on label predictions, and uses softmax scores as the tie breaker.

We perform MMA training with different hinge thresholds d max , also with/without the additional clean loss (see next section for details).

We use OMMA to represent training with only L MMA θ in Eq. (7), and MMA to represent training with the combined loss in Eq. (8).

When train For each d max value, we train two models with different random seeds, which serves two purposes: 1) confirming the performance of MMA trained models are not significantly affected by random initialization; 2) to provide transfer attacks from an "identical" model.

As such, MMA trained models are named as OMMA/MMA-d max -seed.

Models shown in the main body correspond to those with seed "sd0".

With regard to ensemble models, for MNIST-2 PGD/PGDLS-ens, CIFAR10-2 PGD/PGDLS-ens, MNIST-∞ PGDLS-ens, and CIFAR10-∞ PGDLS-ens, they all use the PGD (or PGDLS) models trained at all testing (attacking) '

s. For CIFAR10-∞ PGD-ens, PGD-24,28,32 are excluded for the same reason.

For both ∞ and 2 PGD attacks, we use the implementation from the AdverTorch toolbox (Ding et al., 2019b) .

Regarding the loss function of PGD, we use both the cross-entropy (CE) loss and the Carlini & Wagner (CW) loss.

As previously stated, each model will have N whitebox PGD attacks on them, N/2 of them are CE-PGD attacks, and the other N/2 are CW-PGD attacks.

Recall that N = 50 for MNIST and N = 10 for CIFAR10.

At test time, all the PGD attack run 100 iterations.

We manually tune the step size parameter on a few MMA and PGD models and then fix them thereafter.

The step size for MNIST-∞ when = 0.3 is 0.0075, the step size for CIFAR10-∞ when = 8/255 is 2/255, the step size for MNIST-2 when = 1.0 is 0.25, the step size for CIFAR10-2 when = 1.0 is 0.25.

For other values, the step size is linearly scaled accordingly.

The ensemble model we considered uses the majority vote for prediction, and uses softmax score as the tie breaker.

So it is not obvious how to perform CW-PGD and CE-PGD directly on them.

Here we take 2 strategies.

The first one is a naive strategy, where we minimize the sum of losses of all the models used in the ensemble.

Here, similar to attacking single models, we CW and CE loss here and perform the same number attacks.

The second strategy is still a PGD attack with a customized loss towards attacking ensemble models.

For the group of classifiers in the ensemble, at each PGD step, if less than half of the classifiers give wrong classification, we sum up the CW losses from correct classifiers as the loss for the PGD attack.

If more than half of the classifiers give wrong classification, then we find the wrong prediction that appeared most frequently among classifiers, and denote it as label0, with its corresponding logit, logit0.

For each classifier, we then find the largest logit that is not logit0, denoted as logit1.

The loss we maximize, in the PGD attack, is the sum of "logit1 -logit0" from each classifier.

Using this strategy, we perform additional (compared to attacking single models) whitebox PGD attacks on ensemble models.

For MNIST, we perform 50 repeated attacks, for CIFAR10 we perform 10.

These are also 100-step PGD attacks.

We expect more carefully designed attacks could work better on ensembles, but we delay it to future work.

We further examine the effectiveness of adding a clean loss term to the MMA loss.

We represent MMA trained models with the MMA loss in Eq. (7) as MMA-d max .

In Section 2.4, we introduced MMAC-d max models to resolve MMA-d max model's problem of having flat input space loss landscape and showed its effectiveness qualitatively.

Here we demonstrate the quantitative benefit of adding the clean loss.

We observe that models trained with the MMA loss in Eq. (7) have certain degrees of TransferGaps.

The term TransferGaps represents the difference between robust accuracy under "combined (whitebox+transfer) attacks" and under "only whitebox PGD attacks".

In other words, it is the additional attack success rate that transfer attacks bring.

For example, OMMA-32 achieves 53.70% under whitebox PGD attacks, but achieves a lower robust accuracy at 46.31% under combined (whitebox+transfer) attacks, therefore it has a TransferGap of 7.39% (See Appendix F for full results.).

After adding the clean loss, MMA-32 reduces its TransferGap at = 8/255 to 3.02%.

This corresponds to our observation in Section 2.4 that adding clean loss makes the loss landscape more tilted, such that whitebox PGD attacks can succeed more easily.

Recall that MMA trained models are robust to gradient free attacks, as described in Section 4.3.

Therefore, robustness of MMA trained models and the TransferGaps are likely not due to gradient masking.

We also note that TransferGaps for both MNIST-∞ and 2 cases are almost zero for the MMA trained models, indicating that TransferGaps, observed on CIFAR10 cases, are not solely due to the MMA algorithm, data distributions (MNIST vs CIFAR10) also play an important role.

Another interesting observation is that, for MMA trained models trained on CIFAR10, adding additional clean loss results in a decrease in clean accuracy and an increase in the average robust accuracy, e.g. OMMA-32 has ClnAcc 86.11%, and AvgRobAcc 28.36%, whereas MMA-32 has ClnAcc 84.36%, and AvgRobAcc 29.39%.

The fact that "adding additional clean loss results in a model with lower accuracy and more robustness" seems counter-intuitive.

However, it actually confirms our motivation and reasoning of the additional clean loss: it makes the input space loss landscape steeper, which leads to stronger adversaries at training time, which in turn poses more emphasis on "robustness training", instead of clean accuracy training.

We present all the empirical results in Table 2 to 13.

Specifically, we show model performances under combined (whitebox+transfer) attacks in Tables 2 to 5 .

This is our proxy for true robustness measure.

We show model performances under only whitebox PGD attacks in Tables 6 to 9 .

We show TransferGaps in Tables 10 to 13 .

In these tables, PGD-Madry et al. models are the "secret" models downloaded from https:// github.com/MadryLab/mnist_challenge and https://github.com/MadryLab/ cifar10_challenge/. DDN-Rony et al. models are downloaded from https://github.

com/jeromerony/fast_adversarial/.

For MNIST PGD-Madry et al. models, our whitebox attacks brings the robust accuracy at = 0.3 down to 89.79%, which is at the same level with the reported 89.62% on the website, also with 50 repeated random initialized PGD attacks.

For CIFAR10 PGD-Madry et al. models, our whitebox attacks brings the robust accuracy at = 8/255 down to 44.70%, which is stronger than the reported 45.21% on the website, with 10 repeated random initialized 20-step PGD attacks.

As our PGD attacks are 100-step, this is not surprising.

As we mentioned previously, DDN training can be seen as a specific instantiation of the general MMA training idea, and the DDN-Rony et al. models indeed performs very similar to MMA trained models when d max is set relatively low.

Therefore, we do not discuss the performance of DDN-Rony et al. separately.

In Section 4, we have mainly discussed different phenomena under the case of CIFAR10-∞ .

For CIFAR10-2 , we see very similar patterns in Tables 5, 9 and 13.

These include

• MMA training is fairly stable to d max , and achieves good robustness-accuracy trade-offs.

On the other hand, to achieve good AvgRobAcc, PGD/PGDLS trained models need to have large sacrifices on clean accuracies.

• Adding additional clean loss increases the robustness of the model, reduce TransferGap, at a cost of slightly reducing clean accuracy.

As a simpler datasets, different adversarial training algorithms, including MMA training, have very different behaviors on MNIST as compared to CIFAR10.

We first look at MNIST-∞ .

Similar to CIFAR10 cases, PGD training is incompetent on large 's, e.g. PGD-0.4 has significant drop on clean accuracy (to 96.64%) and PGD-0.45 fails to train.

PGDLS training, on the other hand, is able to handle large 's training very well on MNIST-∞ , and MMA training does not bring extra benefit on top of PGDLS.

We suspect that this is due to the "easiness" of this specific task on MNIST, where finding proper for each individual example is not necessary, and a global scheduling of is enough.

We note that this phenomenon confirms our understanding of adversarial training from the margin maximization perspective in Section 3.

Under the case of MNIST-2 , we notice that MMA training almost does not need to sacrifice clean accuracy in order to get higher robustness.

All the models with d max ≥ 4.0 behaves similarly w.r.t.

both clean and robust accuracies.

Achieving 40% robust accuracy at = 3.0 seems to be the robustness limit of MMA trained models.

On the other hand, PGD/PGDLS models are able to get higher robustness at = 3.0 with robust accuracy of 44.5%, although with some sacrifices to clean accuracy.

This is similar to what we have observed in the case of CIFAR10.

We notice that on both MNIST-∞ and MNIST-2 , unlike CIFAR10 cases, PGD(LS)-ens model performs poorly in terms of robustness.

This is likely due to that PGD trained models on MNIST usually have a very sharp robustness drop when the used for attacking is larger than the used for training.

Another significant differences between MNIST cases and CIFAR10 cases is that TransferGaps are very small for OMMA/MMA trained models on MNIST cases.

This again is likely due to that MNIST is an "easier" dataset.

It also indicates that the TransferGap is not purely due to the MMA training algorithm, it is also largely affected by the property of datasets.

Although previous literature (Ding et al., 2019a; Zhang et al., 2019c ) also discusses related topics on the difference between MNIST and CIFAR10 w.r.t.

adversarial robustness, they do not directly explain the observed phenomena here.

We delay a thorough understanding of this topic to future work.

<|TLDR|>

@highlight

We propose MMA training to directly maximize input space margin in order to improve adversarial robustness primarily by removing the requirement of specifying a fixed distortion bound.

@highlight

An adaptive margin-based adversarial training approach to train robust DNNs, by maximizing the shortest margin of inputs to the decision boundary, that makes adversarial training with large perturbation possible.

@highlight

A method for robust learning against adversarial attacks where the input space margin is directly maximized and a softmax variant of the max-margin is introduced.