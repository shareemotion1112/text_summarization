Adversarial examples are perturbed inputs designed to fool machine learning models.

Adversarial training injects such examples into training data to increase robustness.

To scale this technique to large datasets, perturbations are crafted using fast single-step methods that maximize a linear approximation of the model's loss.

We show that this form of adversarial training converges to a degenerate global minimum, wherein small curvature artifacts near the data points obfuscate a linear approximation of the loss.

The model thus learns to generate weak perturbations, rather than defend against strong ones.

As a result, we find that adversarial training remains vulnerable to black-box attacks, where we transfer perturbations computed on undefended models, as well as to a powerful novel single-step attack that escapes the non-smooth vicinity of the input data via a small random step.

We further introduce Ensemble Adversarial Training, a technique that augments training data with perturbations transferred from other models.

On ImageNet, Ensemble Adversarial Training yields models with strong robustness to black-box attacks.

In particular, our most robust model won the first round of the NIPS 2017 competition on Defenses against Adversarial Attacks.

Machine learning (ML) models are often vulnerable to adversarial examples, maliciously perturbed inputs designed to mislead a model at test time BID4 BID36 BID15 BID29 .

Furthermore, BID36 showed that these inputs transfer across models: the same adversarial example is often misclassified by different models, thus enabling simple black-box attacks on deployed models BID23 .Adversarial training BID36 increases robustness by augmenting training data with adversarial examples. showed that adversarially trained models can be made robust to white-box attacks (i.e., with knowledge of the model parameters) if the perturbations computed during training closely maximize the model's loss.

However, prior attempts at scaling this approach to ImageNet-scale tasks BID12 ) have proven unsuccessful BID20 .It is thus natural to ask whether it is possible, at scale, to achieve robustness against the class of black-box adversaries Towards this goal, BID20 adversarially trained an Inception v3 model BID38 on ImageNet using a "single-step" attack based on a linearization of the model's loss BID15 .

Their trained model is robust to single-step perturbations but remains vulnerable to more costly "multi-step" attacks.

Yet, BID20 found that these attacks fail to reliably transfer between models, and thus concluded that the robustness of their model should extend to black-box adversaries.

Surprisingly, we show that this is not the case.

We demonstrate, formally and empirically, that adversarial training with single-step methods admits a degenerate global minimum, wherein the model's loss can not be reliably approximated by a linear function.

Specifically, we find that the model's decision surface exhibits sharp curvature near the data points, thus degrading attacks based on a single gradient computation.

In addition to the model of BID20 , we reveal similar overfitting in an adversarially trained Inception ResNet v2 model BID37 , and a variety of models trained on MNIST BID22 .We harness this result in two ways.

First, we show that adversarially trained models using single-step methods remain vulnerable to simple attacks.

For black-box adversaries, we find that perturbations crafted on an undefended model often transfer to an adversarially trained one.

We also introduce a simple yet powerful single-step attack that applies a small random perturbation-to escape the nonsmooth vicinity of the data point-before linearizing the model's loss.

While seemingly weaker than the Fast Gradient Sign Method of BID15 , our attack significantly outperforms it for a same perturbation norm, for models trained with or without adversarial training.

Second, we propose Ensemble Adversarial Training, a training methodology that incorporates perturbed inputs transferred from other pre-trained models.

Our approach decouples adversarial example generation from the parameters of the trained model, and increases the diversity of perturbations seen during training.

We train Inception v3 and Inception ResNet v2 models on ImageNet that exhibit increased robustness to adversarial examples transferred from other holdout models, using various single-step and multi-step attacks BID15 BID7 BID19 .

We also show that our methods globally reduce the dimensionality of the space of adversarial examples BID40 .

Our Inception ResNet v2 model won the first round of the NIPS 2017 competition on Defenses Against Adversarial Attacks BID21 , where it was evaluated on other competitors' attacks in a black-box setting.

BID16 BID24 BID31 BID28 BID10 and many remain vulnerable to adaptive attackers BID7 b; BID3 .

Adversarial training BID36 BID15 BID20 appears to hold the greatest promise for learning robust models.

show that adversarial training on MNIST yields models that are robust to whitebox attacks, if the adversarial examples used in training closely maximize the model's loss.

Moreover, recent works by BID34 , BID33 and BID18 even succeed in providing certifiable robustness for small perturbations on MNIST.

As we argue in Appendix C, the MNIST dataset is peculiar in that there exists a simple "closed-form" denoising procedure (namely feature binarization) which leads to similarly robust models without adversarial training.

This may explain why robustness to white-box attacks is hard to scale to tasks such as ImageNet BID20 .

We believe that the existence of a simple robust baseline for MNIST can be useful for understanding some limitations of adversarial training techniques.

BID36 found that adversarial examples transfer between models, thus enabling blackbox attacks on deployed models. showed that black-box attacks could succeed with no access to training data, by exploiting the target model's predictions to extract BID39 a surrogate model.

Some prior works have hinted that adversarially trained models may remain vulnerable to black-box attacks: BID15 found that an adversarial maxout network on MNIST has slightly higher error on transferred examples than on white-box examples.

further showed that a model trained on small perturbations can be evaded by transferring perturbations of larger magnitude.

Our finding that adversarial training degrades the accuracy of linear approximations of the model's loss is as an instance of a gradient-masking phenomenon BID30 , which affects other defensive techniques BID31 BID7 BID28 BID5 BID2 .

We consider a classification task with data x ∈ [0, 1] d and labels y true ∈ Z k sampled from a distribution D. We identify a model with an hypothesis h from a space H. On input x, the model outputs class scores h(x) ∈ R k .

The loss function used to train the model, e.g., cross-entropy, is L(h(x), y).

For some target model h ∈ H and inputs (x, y true ) the adversary's goal is to find an adversarial example x adv such that x adv and x are "close" yet the model misclassifies x adv .

We consider the wellstudied class of ∞ bounded adversaries BID15 that, given some budget , output examples x adv where x adv − x ∞ ≤ .

As we comment in Appendix C.1, ∞ robustness is of course not an end-goal for secure ML.

We use this standard model to showcase limitations of prior adversarial training methods, and evaluate our proposed improvements.

We distinguish between white-box adversaries that have access to the target model's parameters (i.e., h), and black-box adversaries with only partial information about the model's inner workings.

Formal definitions for these adversaries are in Appendix A. Although security against white-box attacks is the stronger notion (and the one we ideally want ML models to achieve), black-box security is a reasonable and more tractable goal for deployed ML models.

Following , we consider an adversarial variant of standard Empirical Risk Minimization (ERM), where our aim is to minimize the risk over adversarial examples: argue that adversarial training has a natural interpretation in this context, where a given attack (see below) is used to approximate solutions to the inner maximization problem, and the outer minimization problem corresponds to training over these examples.

Note that the original formulation of adversarial training BID36 BID15 ), which we use in our experiments, trains on both the "clean" examples x and adversarial examples x adv .

DISPLAYFORM0 We consider three algorithms to generate adversarial examples with bounded ∞ norm.

The first two are single-step (i.e., they require a single gradient computation); the third is iterative-it computes multiple gradient updates.

We enforce x adv ∈ [0, 1] d by clipping all components of x adv .Fast Gradient Sign Method (FGSM).

This method BID15 linearizes the inner maximization problem in (1): DISPLAYFORM1 Single-Step Least-Likely Class Method (Step-LL).

This variant of FGSM introduced by BID19 b) targets the least-likely class, y LL = arg min{h(x)}: DISPLAYFORM2 Although this attack only indirectly tackles the inner maximization in (1), BID20 find it to be the most effective for adversarial training on ImageNet.

Iterative Attack (I-FGSM or Iter-LL).

This method iteratively applies the FGSM or Step-LL k times with step-size α ≥ /k and projects each step onto the ∞ ball of norm around x. It uses projected gradient descent to solve the maximization in (1).

For fixed , iterative attacks induce higher error rates than single-step attacks, but transfer at lower rates BID19 b) .

When performing adversarial training with a single-step attack (e.g., the FGSM or Step-LL methods above), we approximate Equation (1) by replacing the solution to the inner maximization problem adv FGSM in (2)).

That is, we solve DISPLAYFORM0 For model families H with high expressive power, this alternative optimization problem admits at least two substantially different global minima h * :• For an input x from D, there is no x adv close to x (in ∞ norm) that induces a high loss.

That is, DISPLAYFORM1 In other words, h * is robust to all ∞ bounded perturbations.• The minimizer h * is a model for which the approximation method underlying the attack (i.e., linearization in our case) poorly fits the model's loss function.

That is, DISPLAYFORM2 Thus the attack when applied to h * produces samples x adv that are far from optimal.

Note that this second "degenerate" minimum can be more subtle than a simple case of overfitting to samples produced from single-step attacks.

Indeed, we show in Section 4.1 that single-step attacks applied to adversarially trained models create "adversarial" examples that are easy to classify even for undefended models.

Thus, adversarial training does not simply learn to resist the particular attack used during training, but actually to make that attack perform worse overall.

This phenomenon relates to the notion of Reward Hacking BID1 wherein an agent maximizes its formal objective function via unintended behavior that fails to captures the designer's true intent.

The degenerate minimum described in Section 3.3 is attainable because the learned model's parameters influence the quality of both the minimization and maximization in (1).

One solution is to use a stronger adversarial example generation process, at a high performance cost .

Alternatively, BID3 suggest training an adversarial generator model as in the GAN framework BID14 .

The power of this generator is likely to require careful tuning, to avoid similar degenerate minima (where the generator or classifier overpowers the other).We propose a conceptually simpler approach to decouple the generation of adversarial examples from the model being trained, while simultaneously drawing an explicit connection with robustness to black-box adversaries.

Our method, which we call Ensemble Adversarial Training, augments a model's training data with adversarial examples crafted on other static pre-trained models.

Intuitively, as adversarial examples transfer between models, perturbations crafted on an external model are good approximations for the maximization problem in (1).

Moreover, the learned model can not influence the "strength" of these adversarial examples.

As a result, minimizing the training loss implies increased robustness to black-box attacks from some set of models.

Domain Adaptation with multiple sources.

We can draw a connection between Ensemble Adversarial Training and multiple-source Domain Adaptation BID26 BID43 .

In Domain Adaptation, a model trained on data sampled from one or more source distributions S 1 , . . .

, S k is evaluated on samples x from a different target distribution T .Let A i be an adversarial distribution obtained by sampling (x, y true ) from D, computing an adversarial example x adv for some model such that x adv − x ∞ ≤ , and outputting (x adv , y true ).

In Ensemble Adversarial Training, the source distributions are D (the clean data) and A 1 , . . .

, A k (the attacks overs the currently trained model and the static pre-trained models).

The target distribution takes the form of an unseen black-box adversary A * .

Standard generalization bounds for Domain Adaptation BID26 BID43 Figure 1: Gradient masking in single-step adversarial training.

We plot the loss of model v3 adv on points DISPLAYFORM0 where g is the signed gradient and g ⊥ is an orthogonal adversarial direction.

Plot (b) is a zoom of (a) near x. The gradient poorly approximates the global loss.

We give a formal statement of this result and of the assumptions on A * in Appendix B. Of course, ideally we would like guarantees against arbitrary future adversaries.

For very low-dimensional tasks (e.g., MNIST), stronger guarantees are within reach for specific classes of adversaries (e.g., ∞ bounded perturbations BID34 BID33 BID18 ), yet they also fail to extend to other adversaries not considered at training time (see Appendix C.1 for a discussion).

For ImageNet-scale tasks, stronger formal guarantees appear out of reach, and we thus resort to an experimental assessment of the robustness of Ensemble Adversarially Trained models to various non-interactive black-box adversaries in Section 4.2.

We show the existence of a degenerate minimum, as described in Section 3.3, for the adversarially trained Inception v3 model of BID20 .

Their model (denoted v3 adv ) was trained on a Step-LL attack with ≤ 16/256.

We also adversarially train an Inception ResNet v2 model BID37 using the same setup.

We denote this model by IRv2 adv .

We refer the reader to BID20 for details on the adversarial training procedure.

We first measure the approximation-ratio of the Step-LL attack for the inner maximization in (1).

As we do not know the true maximum, we lower-bound it using an iterative attack.

For 1,000 random test points, we find that for a standard Inception v3 model, step-LL gets within 19% of the optimum loss on average.

This attack is thus a good candidate for adversarial training.

Yet, for the v3 adv model, the approximation ratio drops to 7%, confirming that the learned model is less amenable to linearization.

We obtain similar results for Inception ResNet v2 models.

The ratio is 17% for a standard model, and 8% for IRv2 adv .

Similarly, we look at the cosine similarity between the perturbations given by a single-step and multi-step attack.

The more linear the model, the more similar we expect both perturbations to be.

The average similarity drops from 0.13 for Inception v3 to 0.02 for v3 adv .

This effect is not due to the decision surface of v3 adv being "too flat" near the data points: the average gradient norm is larger for v3 adv (0.17) than for the standard v3 model (0.10).We visualize this "gradient-masking" effect BID30 by plotting the loss of v3 adv on examples DISPLAYFORM0 where g is the signed gradient of model v3 adv and g ⊥ is a signed vector orthogonal to g. Looking forward to Section 4.1, we actually chose g ⊥ to be the signed gradient of another Inception model, from which adversarial examples transfer to v3 adv .

Figure 1 shows that the loss is highly curved in the vicinity of the data point x, and that the gradient poorly reflects the global loss landscape.

Similar plots for additional data points are in Figure 4 .We show similar results for adversarially trained MNIST models in Appendix C.2.

On this task, input dropout BID35 mitigates adversarial training's overfitting problem, in some cases.

Presumably, the random input mask diversifies the perturbations seen during training (dropout at intermediate layers does not mitigate the overfitting effect).

BID27 find that input dropout significantly degrades accuracy on ImageNet, so we did not include it in our experiments.

Top FORMULA4 4.1 ATTACKS AGAINST ADVERSARIALLY TRAINED NETWORKS BID20 found their adversarially trained model to be robust to various single-step attacks.

They conclude that this robustness should translate to attacks transferred from other models.

As we have shown, the robustness to single-step attacks is actually misleading, as the model has learned to degrade the information contained in the model's gradient.

As a consequence, we find that the v3 adv model is substantially more vulnerable to single-step attacks than BID20 predicted, both in a white-box and black-box setting.

The same holds for the IRv2 adv model.

In addition to the v3 adv and IRv2 adv models, we consider standard Inception v3, Inception v4 and Inception ResNet v2 models.

These models are available in the TensorFlow-Slim library BID0 .

We describe similar results for a variety of models trained on MNIST in Appendix C.2.Black-box attacks.

TAB1 shows error rates for single-step attacks transferred between models.

We compute perturbations on one model (the source) and transfer them to all others (the targets).

When the source and target are the same, the attack is white-box.

Adversarial training greatly increases robustness to white-box single-step attacks, but incurs a higher error rate in a black-box setting.

Thus, the robustness gain observed when evaluating defended models in isolation is misleading.

Given the ubiquity of this pitfall among proposed defenses against adversarial examples BID7 BID5 BID30 , we advise researchers to always consider both white-box and black-box adversaries when evaluating defensive strategies.

Notably, a similar discrepancy between white-box and black-box attacks was recently observed in BID6 .Attacks crafted on adversarial models are found to be weaker even against undefended models (i.e., when using v3 adv or IRv2 adv as source, the attack transfers with lower probability).

This confirms our intuition from Section 3.3: adversarial training does not just overfit to perturbations that affect standard models, but actively degrades the linear approximation underlying the single-step attack.

A new randomized single-step attack.

The loss function visualization in Figure 1 shows that sharp curvature artifacts localized near the data points can mask the true direction of steepest ascent.

We thus suggest to prepend single-step attacks by a small random step, in order to "escape" the non-smooth vicinity of the data point before linearizing the model's loss.

Our new attack, called R+FGSM (alternatively, R+Step-LL), is defined as follows, for parameters and α (where α < ): DISPLAYFORM1 Note that the attack requires a single gradient computation.

The R+FGSM is a computationally efficient alternative to iterative methods that have high success rates in a white-box setting.

Our attack can be seen as a single-step variant of the general PGD method from .

TAB2 compares error rates for the Step-LL and R+Step-LL methods (with = 16/256 and α = /2).

The extra random step yields a stronger attack for all models, even those without adversarial training.

This suggests that a model's loss function is generally less smooth near the data points.

We further compared the R+Step-LL attack to a two-step Iter-LL attack, which computes two gradient steps.

Surprisingly, we find that for the adversarially trained Inception v3 model, the R+Step-LL attack is stronger than the two-step Iter-LL attack.

That is, the local gradients learned by the adversarially trained model are worse than random directions for finding adversarial examples!

TAB8 .Step We find that the addition of this random step hinders transferability (see TAB11 ).

We also tried adversarial training using R+FGSM on MNIST, using a similar approach as .

We adversarially train a CNN (model A in TAB6 ) for 100 epochs, and attain > 90.0% accuracy on R+FGSM samples.

However, training on R+FGSM provides only little robustness to iterative attacks.

For the PGD attack of with 20 steps, the model attains 18.0% accuracy.

We now evaluate our Ensemble Adversarial Training strategy described in Section 3.4.

We recall our intuition: by augmenting training data with adversarial examples crafted from static pre-trained models, we decouple the generation of adversarial examples from the model being trained, so as to avoid the degenerate minimum described in Section 3.3.

Moreover, our hope is that robustness to attacks transferred from some fixed set of models will generalize to other black-box adversaries.

We train Inception v3 and Inception ResNet v2 models BID37 on ImageNet, using the pre-trained models shown in TAB4 .

In each training batch, we rotate the source of adversarial examples between the currently trained model and one of the pre-trained models.

We select the source model at random in each batch, to diversify examples across epochs.

The pre-trained models' gradients can be precomputed for the full training set.

The per-batch cost of Ensemble Adversarial Training is thus lower than that of standard adversarial training: using our method with n − 1 pre-trained models, only every n th batch requires a forward-backward pass to compute adversarial gradients.

We use synchronous distributed training on 50 machines, with minibatches of size 16 (we did not pre-compute gradients, and thus lower the batch size to fit all models in memory).

Half of the examples in a minibatch are replaced by Step-LL examples.

As in BID20 , we use RMSProp with a learning rate of 0.045, decayed by a factor of 0.94 every two epochs.

To evaluate how robustness to black-box attacks generalizes across models, we transfer various attacks crafted on three different holdout models (see TAB4 ), as well as on an ensemble of these models (as in BID23 ).

We use the Step-LL, R+Step-LL, FGSM, I-FGSM and the PGD attack from using the hinge-loss function from BID7 .

Our results are in Table 4 .

For each model, we report the worst-case error rate over all black-box attacks transfered from each of the holdout models (20 attacks in total).

Results for MNIST are in TAB10 .Convergence speed.

Convergence of Ensemble Adversarial Training is slower than for standard adversarial training, a result of training on "hard" adversarial examples and lowering the batch size.

BID20 report that after 187 epochs (150k iterations with minibatches of size 32), the v3 adv model achieves 78% accuracy.

Ensemble Adversarial Training for models v3 adv-ens3 and v3 adv-ens4 converges after 280 epochs (450k iterations with minibatches of size 16).

The Inception ResNet v2 model is trained for 175 epochs, where a baseline model converges at around 160 epochs.

Table 4 : Error rates (in %) for Ensemble Adversarial Training on ImageNet.

Error rates on clean data are computed over the full test set.

For 10,000 random test set inputs, and = 16 /256, we report error rates on white-box Step-LL and the worst-case error over a series of black-box attacks (Step-LL, R+Step-LL, FGSM, I-FGSM, PGD) transferred from the holdout models in TAB4 .

For both architectures, we mark methods tied for best in bold (based on 95% confidence).

White-box attacks.

For both architectures, the models trained with Ensemble Adversarial Training are slightly less accurate on clean data, compared to standard adversarial training.

Our models are also more vulnerable to white-box single-step attacks, as they were only partially trained on such perturbations.

Note that for v3 adv-ens4 , the proportion of white-boxStep-LL samples seen during training is 1 /4 (instead of 1 /3 for model v3 adv-ens3 ).

The negative impact on the robustness to white-box attacks is large, for only a minor gain in robustness to transferred samples.

Thus it appears that while increasing the diversity of adversarial examples seen during training can provide some marginal improvement, the main benefit of Ensemble Adversarial Training is in decoupling the attacks from the model being trained, which was the goal we stated in Section 3.4.Ensemble Adversarial Training is not robust to white-box Iter-LL and R+Step-LL samples: the error rates are similar to those for the v3 adv model, and omitted for brevity (see BID20 for Iter-LL attacks and TAB2 for R+Step-LL attacks).

BID20 conjecture that larger models are needed to attain robustness to such attacks.

Yet, against black-box adversaries, these attacks are only a concern insofar as they reliably transfer between models.

Black-box attacks.

Ensemble Adversarial Training significantly boosts robustness to all attacks transferred from the holdout models.

For the IRv2 adv-ens model, the accuracy loss (compared to IRv2's accuracy on clean data) is 7.4% (top 1) and 3.1% (top 5).

We find that the strongest attacks in our test suite (i.e., with highest transfer rates) are the FGSM attacks.

Black-box R+Step-LL or iterative attacks are less effective, as they do not transfer with high probability (see BID20 and TAB11 ).

Attacking an ensemble of all three holdout models, as in BID23 , did not lead to stronger black-box attacks than when attacking the holdout models individually.

Our results have little variance with respect to the attack parameters (e.g., smaller ) or to the use of other holdout models for black-box attacks (e.g., we obtain similar results by attacking the v3 adv-ens3 and v3 adv-ens4 models with the IRv2 model).

We also find that v3 adv-ens3 is not vulnerable to perturbations transferred from v3 adv-ens4 .

We obtain similar results on MNIST (see Appendix C.2), thus demonstrating the applicability of our approach to different datasets and model architectures.

The NIPS 2017 competition on adversarial examples.

Our Inception ResNet v2 model was included as a baseline defense in the NIPS 2017 competition on Adversarial Examples BID21 .

Participants of the attack track submitted non-interactive black-box attacks that produce adversarial examples with bounded ∞ norm.

Models submitted to the defense track were evaluated on all attacks over a subset of the ImageNet test set.

The score of a defense was defined as the average accuracy of the model over all adversarial examples produced by all attacks.

Our IRv2 adv-ens model finished 1 st among 70 submissions in the first development round, with a score of 95.3% (the second placed defense scored 89.9%).

The test data was intentionally chosen as an "easy" subset of ImageNet.

Our model achieved 97.9% accuracy on the clean test data.

After the first round, we released our model publicly, which enabled other users to launch white-box attacks against it.

Nevertheless, a majority of the final submissions built upon our released model.

The winning submission (team "liaofz" with a score of 95.3%) made use of a novel adversarial The dimensionality of the adversarial cone.

For 500 correctly classified points x, and for ∈ {4, 10, 16}, we plot the probability that we find at least k orthogonal vectors r i such that r i ∞ = and x + r i is misclassified.

For ≥ 10, model v3 adv shows a bimodal phenomenon: most points x either have 0 adversarial directions or more than 90.denoising technique.

The second placed defense (team "cihangxie" with a score of 92.4%) prepends our IRv2 adv-ens model with random padding and resizing of the input image BID42 .It is noteworthy that the defenses that incorporated Ensemble Adversarial Training faired better against the worst-case black-box adversary.

Indeed, although very robust on average, the winning defense achieved as low as 11.8% accuracy on some attacks.

The best defense under this metric (team "rafaelmm" which randomly perturbed images before feeding them to our IRv2 adv-ens model) achieved at least 53.6% accuracy against all submitted attacks, including the attacks that explicitly targeted our released model in a white-box setting.

Decreasing gradient masking.

Ensemble Adversarial Training decreases the magnitude of the gradient masking effect described previously.

For the v3 adv-ens3 and v3 adv-ens4 models, we find that the loss incurred on a Step-LL attack gets within respectively 13% and 18% of the optimum loss (we recall that for models v3 and v3 adv , the approximation ratio was respectively 19% and 7%).

Similarly, for the IRv2 adv-ens model, the ratio improves from 8% (for IRv2 adv ) to 14%.

As expected, not solely training on a white-box single-step attack reduces gradient masking.

We also verify that after Ensemble Adversarial Training, a two-step iterative attack outperforms the R+Step-LL attack from Section 4.1, thus providing further evidence that these models have meaningful gradients.

Finally, we revisit the "Gradient-Aligned Adversarial Subspace" (GAAS) method of BID40 .

Their method estimates the size of the space of adversarial examples in the vicinity of a point, by finding a set of orthogonal perturbations of norm that are all adversarial.

We note that adversarial perturbations do not technically form a "subspace" (e.g., the 0 vector is not adversarial).

Rather, they may form a "cone", the dimension of which varies as we increase .

By linearizing the loss function, estimating the dimensionality of this cone reduces to finding vectors r i that are strongly aligned with the model's gradient g = ∇ x L(h(x), y true ).

BID40 give a method that finds k orthogonal vectors r i that satisfy g r i ≥ · g 2 · 1 √ k (this bound is tight).

We extend this result to the ∞ norm, an open question in BID40 .

In Section E, we give a randomized combinatorial construction BID11 , that finds k orthogonal vectors r i satisfying r i ∞ = and E g r i ≥

· g 1 · 1 √ k. We show that this result is tight as well.

For models v3, v3 adv and v3 adv-ens3 , we select 500 correctly classified test points.

For each x, we search for a maximal number of orthogonal adversarial perturbations r i with r i ∞ = .

We limit our search to k ≤ 100 directions per point.

The results are in FIG2 .

For ∈ {4, 10, 16}, we plot the proportion of points that have at least k orthogonal adversarial perturbations.

For a fixed , the value of k can be interpreted as the dimension of a "slice" of the cone of adversarial examples near a data point.

For the standard Inception v3 model, we find over 50 orthogonal adversarial directions for 30% of the points.

The v3 adv model shows a curious bimodal phenomenon for ≥ 10: for most points (≈ 80%), we find no adversarial direction aligned with the gradient, which is consistent with the gradient masking effect.

Yet, for most of the remaining points, the adversarial space is very high-dimensional (k ≥ 90).

Ensemble Adversarial Training yields a more robust model, with only a small fraction of points near a large adversarial space.

Previous work on adversarial training at scale has produced encouraging results, showing strong robustness to (single-step) adversarial examples BID15 BID20 ).

Yet, these results are misleading, as the adversarially trained models remain vulnerable to simple black-box and white-box attacks.

Our results, generic with respect to the application domain, suggest that adversarial training can be improved by decoupling the generation of adversarial examples from the model being trained.

Our experiments with Ensemble Adversarial Training show that the robustness attained to attacks from some models transfers to attacks from other models.

We did not consider black-box adversaries that attack a model via other means than by transferring examples from a local model.

For instance, generative techniques BID3 might provide an avenue for stronger attacks.

Yet, a recent work by BID41 found Ensemble Adversarial Training to be resilient to such attacks on MNIST and CIFAR10, and often attaining higher robustness than models that were adversarially trained on iterative attacks.

Moreover, interactive adversaries (see Appendix A) could try to exploit queries to the target model's prediction function in their attack, as demonstrated in .

If queries to the target model yield prediction confidences, an adversary can estimate the target's gradient at a given point (e.g., using finite-differences as in ) and fool the target with our R+FGSM attack.

Note that if queries only return the predicted label, the attack does not apply.

Exploring the impact of these classes of black-box attacks and evaluating their scalability to complex tasks is an interesting avenue for future work.

We provide formal definitions for the threat model introduced in Section 3.1.

In the following, we explicitly identify the hypothesis space H that a model belongs to as describing the model's architecture.

We consider a target model h ∈ H trained over inputs (x, y true ) sampled from a data distribution D. More precisely, we write h ← train(H, X train , Y train , r) , where train is a randomized training procedure that takes in a description of the model architecture H, a training set X train , Y train sampled from D, and randomness r.

Given a set of test inputs X, Y = { (x 1 , y 1 ) , . . . , (x m , y m )} from D and a budget > 0, an adversary A produces adversarial examples DISPLAYFORM0 We evaluate success of the attack as the error rate of the target model over X adv : DISPLAYFORM1 We assume A can sample inputs according to the data distribution D. We define three adversaries.

Definition 2 (White-Box Adversary).

For a target model h ∈ H, a white-box adversary is given access to all elements of the training procedure, that is train (the training algorithm), H (the model architecture), the training data X train , Y train , the randomness r and the parameters h. The adversary can use any attack (e.g., those in Section 3.2) to find adversarial inputs.

White-box access to the internal model weights corresponds to a very strong adversarial model.

We thus also consider the following relaxed and arguably more realistic notion of a black-box adversary.

Definition 3 (Non-Interactive Black-Box Adversary).

For a target model h ∈ H, a non-interactive black-box adversary only gets access to train (the target model's training procedure) and H (the model architecture).

The adversary can sample from the data distribution D, and uses a local algorithm to craft adversarial examples X adv .Attacks based on transferability BID36 fall in this category, wherein the adversary selects a procedure train and model architecture H , trains a local model h over D, and computes adversarial examples on its local model h using white-box attack strategies.

Most importantly, a black-box adversary does not learn the randomness r used to train the target, nor the target's parameters h. The black-box adversaries in our paper are actually slightly stronger than the ones defined above, in that they use the same training data X train , Y train as the target model.

We provide A with the target's training procedure train to capture knowledge of defensive strategies applied at training time, e.g., adversarial training BID36 BID15 or ensemble adversarial training (see Section 4.2).

For ensemble adversarial training, A also knows the architectures of all pre-trained models.

In this work, we always mount black-box attacks that train a local model with a different architecture than the target model.

We actually find that black-box attacks on adversarially trained models are stronger in this case (see TAB1 ).The main focus of our paper is on non-interactive black-box adversaries as defined above.

For completeness, we also formalize a stronger notion of interactive black-box adversaries that additionally issue prediction queries to the target model .

We note that in cases where ML models are deployed as part of a larger system (e.g., a self driving car), an adversary may not have direct access to the model's query interface.

Definition 4 (Interactive Black-Box Adversary).

For a target model h ∈ H, an interactive blackbox adversary only gets access to train (the target model's training procedure) and H (the model architecture).

The adversary issues (adaptive) oracle queries to the target model.

That is, for arbitrary inputs x ∈ [0, 1] d , the adversary obtains y = arg max h(x) and uses a local algorithm to craft adversarial examples (given knowledge of H, train, and tuples (x, y)).

show that such attacks are possible even if the adversary only gets access to a small number of samples from D. Note that if the target model's prediction interface additionally returns class scores h(x), interactive black-box adversaries could use queries to the target model to estimate the model's gradient (e.g., using finite differences) , and then apply the attacks in Section 3.2.

We further discuss interactive black-box attack strategies in Section 5.

We provide a formal statement of Theorem 1 in Section 3.4, regarding the generalization guarantees of Ensemble Adversarial Training.

For simplicity, we assume that the model is trained solely on adversarial examples computed on the pre-trained models (i.e., we ignore the clean training data and the adversarial examples computed on the model being trained).

Our results are easily extended to also consider these data points.

Let D be the data distribution and A 1 , . . .

, A k , A * be adversarial distributions where a sample (x, y)is obtained by sampling (x, y true ) from D, computing an x adv such that x adv − x ∞ ≤ and returning (x adv , y true ).

We assume the model is trained on N data points Z train , where N k data points are sampled from each distribution A i , for 1 ≤ i ≤ k. We denote A train = {A 1 , . . .

, A k }.

At test time, the model is evaluated on adversarial examples from A * .For a model h ∈ H we define the empirical risk DISPLAYFORM0 and the risk over the target distribution (or future adversary) DISPLAYFORM1 We further define the average discrepancy distance BID26 ) between distributions A i and A * with respect to a hypothesis space H as DISPLAYFORM2 This quantity characterizes how "different" the future adversary is from the train-time adversaries.

Intuitively, the distance disc(A train , A * ) is small if the difference in robustness between two models to the target attack A * is somewhat similar to the difference in robustness between these two models to the attacks used for training (e.g., if the static black-box attacks A i induce much higher error on some model h 1 than on another model h 2 , then the same should hold for the target attack A * ).

In other words, the ranking of the robustness of models h ∈ H should be similar for the attacks in A train as for A * .Finally, let R N (H) be the average Rademacher complexity of the distributions A 1 , . . .

, A k BID43 .

Note that R N (H) → 0 as N → ∞. The following theorem is a corollary of Zhang et al. (2012, Theorem 5 .2): Theorem 5.

Assume that H is a function class consisting of bounded functions.

Then, with probability at least 1 − , DISPLAYFORM3 Compared to the standard generalization bound for supervised learning, the generalization bound for Domain Adaptation incorporates the extra term disc H (A train , A * ) to capture the divergence between the target and source distributions.

In our context, this means that the model h * learned by Ensemble Adversarial Training has guaranteed generalization bounds with respect to future adversaries that are not "too different" from the ones used during training.

Note that A * need not restrict itself to perturbation with bounded ∞ norm for this result to hold.

We re-iterate our ImageNet experiments on MNIST.

For this simpler task, show that training on iterative attacks conveys robustness to white-box attacks with bounded ∞ norm.

Our goal is not to attain similarly strong white-box robustness on MNIST, but to show that our observations on limitations of single-step adversarial training, extend to other datasets than ImageNet.

The MNIST dataset is a simple baseline for assessing the potential of a defense, but the obtained results do not always generalize to harder tasks.

We suggest that this is because achieving robustness to ∞ perturbations admits a simple "closed-form" solution, given the near-binary nature of the data.

Indeed, for an average MNIST image, over 80% of the pixels are in {0, 1} and only 6% are in the range [0.2, 0.8].

Thus, for a perturbation with ≤ 0.3, binarized versions of x and x adv can differ in at most 6% of the input dimensions.

By binarizing the inputs of a standard CNN trained without adversarial training, we obtain a model that enjoys robustness similar to the model trained by .

Concretely, for a white-box I-FGSM attack, we get at most 11.4% error.

The existence of such a simple robust representation begs the question of why learning a robust model with adversarial training takes so much effort.

Finding techniques to improve the performance of adversarial training, even on simple tasks, could provide useful insights for more complex tasks such as ImageNet, where we do not know of a similarly simple "denoising" procedure.

These positive results on MNIST for the ∞ norm also leave open the question of defining a general norm for adversarial examples.

Let us motivate the need for such a definition: we find that if we first rotate an MNIST digit by 20°, and then use the I-FGSM, our rounding model and the model from achieve only 65% accuracy (on "clean" rotated inputs, the error is < 5%).

If we further randomly "flip" 5 pixels per image, the accuracy of both models drops to under 50%.

Thus, we successfully evade the model by slightly extending the threat model (see FIG3 ).Of course, we could augment the training set with such perturbations (see BID13 ).

An open question is whether we can enumerate all types of "adversarial" perturbations.

In this work, we focus on the ∞ norm to illustrate our findings on the limitations of single-step adversarial training on ImageNet and MNIST, and to showcase the benefits of our Ensemble Adversarial Training variant.

Our approach can easily be extended to consider multiple perturbation metrics.

We leave such an evaluation to future work.

We repeat experiments from Section 4 on MNIST.

We use the architectures in TAB6 .

We train a standard model for 6 epochs, and an adversarial model with the FGSM ( = 0.3) for 12 epochs.

During adversarial training, we avoid the label leaking effect described by BID20 by using the model's predicted class arg max h(x) instead of the true label y true in the FGSM, We first analyze the "degenerate" minimum of adversarial training, described in Section 3.3.

For each trained model, we compute the approximation-ratio of the FGSM for the inner maximization problem in equation FORMULA0 .

That is, we compare the loss produced by the FGSM with the loss of a strong iterative attack.

The results appear in TAB7 .

As we can see, for all model architectures, adversarial training degraded the quality of a linear approximation to the model's loss.

We find that input dropout BID35 ) (i.e., randomly dropping a fraction of input features during training) as used in architecture B limits this unwarranted effect of adversarial training.

If we omit the input dropout (we call this architecture B * ) the single-step attack degrades significantly.

We discuss this effect in more detail below.

For the fully connected architecture D, we find that the learned model is very close to linear and thus also less prone to the degenerate solution to the min-max problem, as we postulated in Section 3.3.Attacks.

TAB8 compares error rates of undefended and adversarially trained models on whitebox and black-box attacks, as in Section 4.1.

Again, model B presents an anomaly.

For all other models, we corroborate our findings on ImageNet for adversarial training: (1) black-box attacks trump white-box single-step attacks; (2) white-box single-step attacks are significantly stronger if prepended by a random step.

For model B adv , the opposite holds true.

We believe this is because input dropout increases diversity of attack samples similarly to Ensemble Adversarial Training.

While training with input dropout helps avoid the degradation of the single-step attack, it also significantly delays convergence of the model.

Indeed, model B adv retains relatively high error on white-box FGSM examples.

Adversarial training with input dropout can be seen as comparable to training with a randomized single-step attack, as discussed in Section 4.1.The positive effect of input dropout is architecture and dataset specific: Adding an input dropout layer to models A, C and D confers only marginal benefit, and is outperformed by Ensemble Adversarial Training, discussed below.

Moreover, BID27 find that input dropout significantly degrades accuracy on ImageNet.

We thus did not incorporate it into our models on ImageNet. , uses 3 pre-trained models ({A, C, D} or {B, C, D}).

We train all models for 12 epochs.

We evaluate our models on black-box attacks crafted on models A,B,C,D (for a fair comparison, we do not use the same pre-trained models for evaluation, but retrain them with different random seeds).

The attacks we consider are the FGSM, I-FGSM and the PGD attack from with the loss function from BID7 ), all with = 0.3.

The results appear in TAB10 .For each model, we report the worst-case and average-case error rate over all black-box attacks.

Ensemble Adversarial Training significantly increases robustness to black-box attacks, except for architecture B, which we previously found to not suffer from the same overfitting phenomenon that affects the other adversarially trained networks.

Nevertheless, model B adv-ens achieves slightly better robustness to white-box and black-box attacks than B adv .

In the majority of cases, we find that using a single pre-trained model produces good results, but that the extra diversity of including three pre-trained models can sometimes increase robustness even further.

Our experiments confirm our conjecture that robustness to black-box attacks generalizes across models.

Indeed, we find that when training with three external models, we attain very good robustness against attacks initiated from models with the same architecture (as evidenced by the average error on our attack suite), but also increased robustness to attacks initiated from the fourth holdout model D TRANSFERABILITY OF RANDOMIZED SINGLE-STEP PERTURBATIONS.In Section 4.1, we introduced the R+Step-LL attack, an extension of the Step-LL method that prepends the attack with a small random perturbation.

In TAB11 , we evaluate the transferability of R+Step-LL adversarial examples on ImageNet.

We find that the randomized variant produces perturbations that transfer at a much lower rate (see TAB1 for the deterministic variant).

BID40 consider the following task for a given model h: for a (correctly classified) point x, find k orthogonal vectors {r 1 , . . .

, r k } such that r i 2 ≤ and all the x + r i are adversarial (i.e., arg max h(x + r i ) = y true ).

By linearizing the model's loss function, this reduces to finding k orthogonal vectors r i that are maximally aligned with the model's gradient g = ∇ x L(h(x), y true ).

BID40 left a construction for the ∞ norm as an open problem.

We provide an optimal construction for the ∞ norm, based on Regular Hadamard Matrices BID11 .

Given the ∞ constraint, we find orthogonal vectors r i that are maximally aligned with the signed gradient, sign(g).

We first prove an analog of BID40 , Lemma 1).Lemma FORMULA5 .

Then, we have DISPLAYFORM0 from which we obtain α ≤ k DISPLAYFORM1 This result bounds the number of orthogonal perturbations we can expect to find, for a given alignment with the signed gradient.

As a warm-up consider the following trivial construction of k orthogonal vectors in {−1, 1} d that are "somewhat" aligned with sign(g).

We split sign(g) into k"chunks" of size

@highlight

Adversarial training with single-step methods overfits, and remains vulnerable to simple black-box and white-box attacks. We show that including adversarial examples from multiple sources helps defend against black-box attacks.