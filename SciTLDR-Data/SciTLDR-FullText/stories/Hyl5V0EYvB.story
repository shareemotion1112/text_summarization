Most existing defenses against adversarial attacks only consider robustness to L_p-bounded distortions.

In reality, the specific attack is rarely known in advance and adversaries are free to modify images in ways which lie outside any fixed distortion model; for example, adversarial rotations lie outside the set of L_p-bounded distortions.

In this work, we advocate measuring robustness against a much broader range of unforeseen attacks, attacks whose precise form is unknown during defense design.



We propose several new attacks and a methodology for evaluating a defense against a diverse range of unforeseen distortions.

First, we construct novel adversarial JPEG, Fog, Gabor, and Snow distortions to simulate more diverse adversaries.

We then introduce UAR, a summary metric that measures the robustness of a defense against a given distortion.

Using UAR to assess robustness against existing and novel attacks, we perform an extensive study of adversarial robustness.

We find that evaluation against existing L_p attacks yields redundant information which does not generalize to other attacks; we instead recommend evaluating against our significantly more diverse set of attacks.

We further find that adversarial training against either one or multiple distortions fails to confer robustness to attacks with other distortion types.

These results underscore the need to evaluate and study robustness against unforeseen distortions.

Neural networks perform well on many benchmark tasks (He et al., 2016) yet can be fooled by adversarial examples (Goodfellow et al., 2014) or inputs designed to subvert a given model.

Adversaries are usually assumed to be constrained by an L ∞ budget (Goodfellow et al., 2014; Madry et al., 2017; Xie et al., 2018) , while other modifications such as adversarial geometric transformations, patches, and even 3D-printed objects have also been considered (Engstrom et al., 2017; Brown et al., 2017; Athalye et al., 2017) .

However, most work on adversarial robustness assumes that the adversary is fixed and known in advance.

Defenses against adversarial attacks are often constructed in view of this specific assumption (Madry et al., 2017) .

In practice, adversaries can modify and adapt their attacks so that they are unforeseen.

In this work, we propose novel attacks which enable the diverse assessment of robustness to unforeseen attacks.

Our attacks are varied ( §2) and qualitatively distinct from current attacks.

We propose adversarial JPEG, Fog, Gabor, and Snow attacks (sample images in Figure 1 ).

We propose an unforeseen attack evaluation methodology ( §3) that involves evaluating a defense against a diverse set of held-out distortions decoupled from the defense design.

For a fixed, held-out distortion, we then evaluate the defense against the distortion for a calibrated range of distortion sizes whose strength is roughly comparable across distortions.

For each fixed distortion, we summarize the robustness of a defense against that distortion relative to a model adversarially trained on that distortion, a measure we call UAR.

We provide code and calibrations to easily evaluate a defense against our suite of attacks at https://github.com/iclr-2020-submission/ advex-uar.

By applying our method to 87 adversarially trained models and 8 different distortion types ( §4), we find that existing defenses and evaluation practices have marked weaknesses.

Our results show

New Attacks JPEG Fog Gabor Snow

Figure 1: Attacked images (label "espresso maker") against adversarially trained models with large ε.

Each of the adversarial images above are optimized to maximize the classification loss.

that existing defenses based on adversarial training do not generalize to unforeseen adversaries, even when restricted to the 8 distortions in Figure 1 .

This adds to the mounting evidence that achieving robustness against a single distortion type is insufficient to impart robustness to unforeseen attacks (Jacobsen et al., 2019; Jordan et al., 2019; Tramèr & Boneh, 2019) .

Turning to evaluation, our results demonstrate that accuracy against different L p distortions is highly correlated relative to the other distortions we consider.

This suggest that the common practice of evaluating only against L p distortions to test a model's adversarial robustness can give a misleading account.

Our analysis demonstrates that our full suite of attacks adds substantive attack diversity and gives a more complete picture of a model's robustness to unforeseen attacks.

A natural approach is to defend against multiple distortion types simultaneously in the hope that seeing a larger space of distortions provides greater transfer to unforeseen distortions.

Unfortunately, we find that defending against even two different distortion types via joint adversarial training is difficult ( §5).

Specifically, joint adversarial training leads to overfitting at moderate distortion sizes.

In summary, we propose a metric UAR to assess robustness of defenses against unforeseen adversaries.

We introduce a total of 4 novel attacks.

We apply UAR to assess how robustness transfers to existing attacks and our novel attacks.

Our results demonstrate that existing defense and evaluation methods do not generalize well to unforeseen attacks.

We consider distortions (attacks) applied to an image x ∈ R 3×224×224 , represented as a vector of RGB values.

Let f : R 3×224×224 → R 100 be a model mapping images to logits 1 , and let (f (x), y) denote the cross-entropy loss.

For an input x with true label y and a target class y = y, our adversarial attacks attempt to find x such that 1.

the attacked image x is obtained by applying a constrained distortion to x, and 2.

the loss (f (x ), y ) is minimized (targeted attack).

Adversarial training (Goodfellow et al., 2014 ) is a strong defense baseline against a fixed attack (Madry et al., 2017; Xie et al., 2018) which updates using an attacked image x instead of the clean image x at each training iteration.

We consider 8 attacks: L ∞ (Goodfellow et al., 2014) , L 2 (Szegedy et al., 2013; Carlini & Wagner, 2017) , L 1 (Chen et al., 2018) , Elastic (Xiao et al., 2018) , JPEG, Fog, Gabor, and Snow.

We show sample attacked images in Figure 1 and the corresponding distortions in Figure 2 .

The JPEG, Fog, L∞ (4.1m, 11.1k, 32) L2 (1.3m, 4.8k, 99) L1 (224k, 2.6k, 218) Elastic (3.1m, 15.2k, 253) New Attacks JPEG (5.4m, 18.7k, 255) Fog (4.1m, 13.2k, 89) Gabor (3.7m, 13.3k, 50) Snow (11.3m, 32.0k, 255) Figure 2: Scaled pixel-level differences between original and attacked images for each attack (label "espresso maker").

The L 1 , L 2 , and L ∞ norms of the difference are shown after the attack name.

Our novel attacks display behavior which is qualitatively different from that of the L p attacks.

Attacked images are shown in Figure 1 , and unscaled differences are shown in Figure 9 , Appendix B.1.

Gabor, and Snow attacks are new to this paper, and the L 1 attack uses the Frank-Wolfe algorithm to improve on previous L 1 attacks.

We now describe the attacks, whose distortion sizes are controlled by a parameter ε.

We clamp output pixel values to [0, 255] .

Existing attacks.

The L p attacks with p ∈ {1, 2, ∞} modify an image x to an attacked image x = x+δ.

We optimize δ under the constraint δ p ≤ ε, where · p is the L p -norm on R 3×224×224 .

The Elastic attack warps the image by allowing distortions x = Flow(x, V ), where V : {1, . . .

, 224} 2 → R 2 is a vector field on pixel space, and Flow sets the value of pixel (i, j) to the bilinearly interpolated original value at (i, j) + V (i, j).

We construct V by smoothing a vector field W by a Gaussian kernel (size 25 × 25, std.

dev.

3 for a 224 × 224 image) and optimize W under W (i, j) ∞ ≤ ε for all i, j.

This differs in details from Xiao et al. (2018) but is similar in spirit.

Novel attacks.

As discussed in Shin & Song (2017) for defense, JPEG compression applies a lossy linear transformation JPEG based on the discrete cosine transform to image space, followed by quantization.

The JPEG attack imposes the L ∞ -constraint JPEG(x) − JPEG(x ) ∞ ≤ ε on the attacked image x .

We optimize z = JPEG(x ) and apply a right inverse of JPEG to obtain x .

Initialization Optimized

Figure 3: Snow before and after optimization.

Our novel Fog, Gabor, and Snow attacks are adversarial versions of non-adversarial distortions proposed in the literature.

Fog and Snow introduce adversarially chosen partial occlusions of the image resembling the effect of mist and snowflakes, respectively; stochastic versions of Fog and Snow appeared in Hendrycks & Dietterich (2019) .

Gabor superimposes adversarially chosen additive Gabor noise (Lagae et al., 2009) onto the image; a stochastic version appeared in Co et al. (2019) .

These attacks work by optimizing a set of parameters controlling the distortion over an L ∞ -bounded set.

Specifically, values for the diamond-square algorithm, sparse noise, and snowflake brightness ( Figure 3 ) are chosen adversarially for Fog, Gabor, and Snow, respectively.

Optimization.

To handle L ∞ and L 2 constraints, we use randomly-initialized projected gradient descent (PGD), which optimizes the distortion δ by gradient descent and projection to the L ∞ and L 2 balls (Madry et al., 2017) .

For L 1 constraints, this projection is more difficult, and previous L 1 attacks resort to heuristics (Chen et al., 2018; Tramèr & Boneh, 2019) .

We use the randomly- Figure 4: Accuracies of L 2 and Elastic attacks at different distortion sizes against a ResNet-50 model adversarially trained against L 2 at ε = 9600 on ImageNet-100.

At small distortion sizes, the model appears to defend well against Elastic, but large distortion sizes reveal a lack of transfer.

initialized Frank-Wolfe algorithm (Frank & Wolfe, 1956 ), which replaces projection by a simpler optimization of a linear function at each step (pseudocode in Appendix B.2).

We now propose a method to assess robustness against unforeseen distortions, which relies on evaluating a defense against a diverse set of attacks that were not used when designing the defense.

Our method must address the following issues:

• The range of distortion sizes must be wide enough to avoid the misleading behavior in which robustness appears to transfer at low distortion sizes but not at high distortion sizes ( Figure 4 ); • The set of attacks considered must be sufficiently diverse.

We first provide a method to calibrate distortion sizes and then use it to define a summary metric that assesses the robustness of a defense against a specific unforeseen attack.

Using this metric, we are able to assess diversity and recommend a set of attacks to evaluate against.

Calibrate distortion size using adversarial training.

As shown in Figure 4 , the correlation between adversarial robustness against different distortion types may look different for different ranges of distortion sizes.

It is therefore critical to evaluate on a wide enough range of distortion size ε.

We choose the minimum and maximum distortion sizes ε using the following principles; sample images at ε min and ε max are shown in Figure 5b.

1.

The minimum distortion size ε min is the largest ε for which the adversarial validation accuracy against an adversarially trained model is comparable to that of a model trained and evaluated on unattacked data (for ImageNet-100, within 3 of 87).

2.

The maximum distortion size ε max is the smallest ε which either (a) yields images which confuse humans when applied against adversarially trained models or (b) reduces accuracy of adversarially trained models (ATA below) to below 25.

In practice, we select ε min and ε max according to these criteria from a sequence of ε which is geometrically increasing with ratio 2.

We choose to evaluate against adversarially trained models because attacking against strong defenses is necessary to produce strong visual distortions ( Figure 5a ).

We introduce the constraint that humans recognize attacked images at ε max because we find cases for L 1 , Fog, and Snow where adversarially trained models maintain non-zero accuracy for distortion sizes producing images incomprehensible to humans.

An example for Snow is shown in Figure 5b .

UAR: an adversarial robustness metric.

We measure a model's robustness against a specific distortion type by comparing it to adversarially trained models, which represent an approximate ceiling on performance with prior knowledge of the distortion type.

For distortion type A and size ε, let the Adversarial Training Accuracy ATA(A, ε) be the best adversarial accuracy on the test set that can be achieved by adversarially training a specific architecture (ResNet-50 for ImageNet-100, ResNet-56 for CIFAR-10) against A. other than ResNet-50 or ResNet-56, we recommend using the ATA values computed with these architectures to allow for uniform comparisons.

Given a set of distortion sizes {ε 1 , . . .

, ε n }, we propose the summary metric UAR (Unforeseen Attack Robustness) normalizing the accuracy of a model M against adversarial training accuracy:

Here Acc(A, ε, M ) is the accuracy of M against distortions of type A and magnitude ε.

We expect most UAR scores to be lower than 100 against held-out distortion types, as an UAR score greater than 100 means that a defense is outperforming an adversarially trained model on that distortion.

The normalizing factor in (1) is required to keep UAR scores roughly comparable between distortions, as different distortions can have different strengths as measured by ATA at the chosen distortion sizes.

Having too many or too few ε k values in a certain range may cause an attack to appear artificially strong or weak because the functional relation between distortion size and attack strength (measured by ATA) varies between attacks.

To make UAR roughly comparable between distortions, we evaluate at ε increasing geometrically from ε min to ε max by factors of 2 and take the subset of ε whose ATA values have minimum 1 -distance to the ATA values of the L ∞ attack at geometrically increasing ε.

For example, when calibrating Elastic in Table 1 , we start with ε min = 0.25 and ε max = 16 based on our earlier criteria.

We then compute the ATAs at the 7 geometrically increasing ε values ε ∈ {0.25, 0.5, 1, 2, 4, 8, 16}. We consider size-6 subsets of those ATA values, view them as vectors of length 6 in decreasing order, and compute the 1 -distance between these vectors and the vector for L ∞ shown in the first row of Table 1 .

Finally, we select the ε values for Elastic in Table 1 as those corresponding to the size-6 subset with minimum 1 -distance to the vector for L ∞ .

For our 8 distortion types, we provide reference values of ATA(A, ε) on this calibrated range of 6 distortion sizes on ImageNet-100 (Table 1 , §4) and CIFAR-10 (Table 3 , Appendix C.3.2).

This allows UAR computation for a new defense using 6 adversarial evaluations and no adversarial training, reducing computational cost from 192+ to 6 NVIDIA V100 GPU-hours on ImageNet-100.

Evaluate against diverse distortion types.

Since robustness against different distortion types may have low or no correlation (Figure 6b ), measuring performance on different distortions is important to avoid overfitting to a specific type, especially when a defense is constructed with it in mind (as with adversarial training).

Our results in §4 demonstrate that choosing appropriate distortion types to evaluate against requires some care, as distortions such as L 1 , L 2 , and L ∞ that may seem different can actually have highly correlated scores against defenses (see Figure 6 ).

We instead recommend evaluation against our more diverse attacks, taking the L ∞ , L 1 , Elastic, Fog, and Snow attacks as a starting point.

We apply our methodology to the 8 attacks in §2 using models adversarially trained against these attacks.

Our results reveal that evaluating against the commonly used L p -attacks gives highly correlated information which does not generalize to other unforeseen attacks.

Instead, they suggest that evaluating on diverse attacks is necessary and identify a set of 5 attacks with low pairwise robustness transfer which we suggest as a starting point when assessing robustness to unforeseen adversaries.

Dataset and model.

We use two datasets: CIFAR-10 and ImageNet-100, the 100-class subset of ImageNet-1K (Deng et al., 2009) containing every 10 th class by WordNet ID order.

We use ResNet-56 for CIFAR-10 and ResNet-50 as implemented in torchvision for ImageNet-100 (He et al., 2016) .

We give training hyperparameters in Appendix A.

Adversarial training and evaluation procedure.

We construct hardened models using adversarial training (Madry et al., 2017) .

To train against attack A, for each mini-batch of training images, we select a uniform random (incorrect) target class for each image.

For maximum distortion size ε, we apply the targeted attack A to the current model with distortion size ε ∼ Uniform(0, ε) and update the model with a step of stochastic gradient descent using only the resulting adversarial images (no clean images).

The random size scaling improves performance especially against smaller distortions.

We use 10 optimization steps for all attacks during training except for Elastic, where we use 30 steps due to its more difficult optimization problem.

When PGD is used, we use step size ε/ √ steps, the optimal scaling for non-smooth convex functions (Nemirovski & Yudin, 1978; 1983) .

We adversarially train 87 models against the 8 attacks from §2 at the distortion sizes described in §3 and evaluate them on the ImageNet-100 and CIFAR-10 validation sets against 200-step targeted attacks with uniform random (incorrect) target class.

This uses more steps for evaluation than train-

Fog ε = 8192

Gabor ε = 3200 (a) UAR scores for adv.

trained defenses (rows) against attacks (columns) on ImageNet-100.

See Figure 12 for more ε values and Appendix C.3.2 for CIFAR-10 results.

ing per best practices (Carlini et al., 2019) .

We use UAR to analyze the results in the remainder of this section, directing the reader to Figures 10 and 11 (Appendix C.2) for exhaustive results and to Appendix D for checks for robustness to random seed and number of attack steps.

Existing defense and evaluation methods do not generalize to unforeseen attacks.

The many low off-diagonal UAR scores in Figure 6a make clear that while adversarial training is a strong baseline against a fixed distortion, it only rarely confers robustness to unforeseen distortions.

Notably, we were not able to achieve a high UAR against Fog except by directly adversarially training against it.

Despite the general lack of transfer in Figure 6a , the fairly strong transfer between the L p -attacks is consistent with recent progress in simultaneous robustness to them (Croce & Hein, 2019) .

Figure 6b shows correlations between UAR scores of pairs of attacks A and A against defenses adversarially trained without knowledge 3 of A or A .

The results demonstrate that defenses trained without knowledge of L p -attacks have highly correlated UAR scores against the different L p attacks, but this correlation does not extend to their evaluations against other attacks.

This suggests that L pevaluations offer limited diversity and may not generalize to other unforeseen attacks.

The L ∞ , L 1 , Elastic, Fog, and Snow attacks offer greater diversity.

Our results on L p -evaluation suggest that more diverse attack evaluation is necessary for generalization to unforeseen attacks.

As the unexpected correlation between UAR scores against the pairs (Fog, Gabor) and (JPEG, L 1 ) in Figure 6b demonstrates, even attacks with very different distortions may have correlated behaviors.

Considering all attacks in Figure 6 together results in signficantly more diversity, which we suggest for evaluation against unforeseen attacks.

We suggest the 5 attacks (L ∞ , L 1 , Elastic, Fog, and Snow) with low UAR against each other and low correlation between UAR scores as a good starting point.

A natural idea to improve robustness against unforeseen adversaries is to adversarially train the same model against two different types of distortions simultaneously, with the idea that this will cover a larger portion of the space of distortions.

We refer to this as joint adversarial training (Jordan et al., 2019; Tramèr & Boneh, 2019) .

For two attacks A and A , at each training step, we compute the attacked image under both A and A and backpropagate with respect to gradients induced by the image with greater loss.

This corresponds to the "max" loss described in Tramèr & Boneh (2019) .

We jointly train models for (L ∞ , L 2 ), (L ∞ , L 1 ), and (L ∞ , Elastic) using the same setup as before

Normal Training

L∞ ε = 16, L1 ε = 612000

Normal Training Transfer for jointly trained models.

Figure 7 reports UAR scores for jointly trained models using ResNet-50 on ImageNet-100; full evaluation accuracies are in Figure 19 (Appendix E).

Comparing to Figure 6a and Figure 12 (Appendix E), we see that, relative to training against only L 2 , joint training against (L ∞ , L 2 ) slightly improves robustness against L 1 without harming robustness against other attacks.

In contrast, training against (L ∞ , L 1 ) is worse than either training against L 1 or L ∞ separately (except at small ε for L 1 ).

Training against (L ∞ , Elastic) also performs poorly.

Joint training and overfitting.

Jointly trained models achieve high training accuracy but poor validation accuracy (Figure 8 ) that fluctuates substantially for different random seeds (Table 4 , Appendix E.2).

Figure 8 shows the overfitting behavior for (L ∞ , Elastic): L ∞ validation accuracy decreases significantly during training while training accuracy increases.

This contrasts with standard adversarial training (Figure 8 ), where validation accuracy levels off as training accuracy increases.

Overfitting primarily occurs when training against large distortions.

We successfully trained against the (L ∞ , L 1 ) and (L ∞ , Elastic) pairs for small distortion sizes with accuracies comparable to but slightly lower than observed in Figure 11 for training against each attack individually ( Figure 18 , Appendix E).

This agrees with behavior reported by Tramèr & Boneh (2019) on CIFAR-10.

Our intuition is that harder training tasks (more diverse distortion types, larger ε) make overfitting more likely.

We briefly investigate the relation between overfitting and model capacity in Appendix E.3; validation accuracy appears slightly increased for ResNet-101, but overfitting remains.

We have seen that robustness to one attack provides limited information about robustness to other attacks, and moreover that adversarial training provides limited robustness to unforeseen attacks.

These results suggest a need to modify or move beyond adversarial training.

While joint adversarial training is one possible alternative, our results show it often leads to overfitting.

Even ignoring this, it is not clear that joint training would confer robustness to attacks outside of those trained against.

Evaluating robustness has proven difficult, necessitating detailed study of best practices even for a single fixed attack (Papernot et al., 2017; Athalye et al., 2018) .

We build on these best practices by showing how to choose and calibrate a diverse set of unforeseen attacks.

Our work is a supplement to existing practices, not a replacement-we strongly recommend following the guidelines in Papernot et al. (2017) and Athalye et al. (2018) in addition to our recommendations.

Some caution is necessary when interpreting specific numeric results in our paper.

Many previous implementations of adversarial training fell prone to gradient masking (Papernot et al., 2017; Engstrom et al., 2018) , with apparently successful training occurring only recently (Madry et al., 2017; Xie et al., 2018) .

While evaluating with moderately many PGD steps (200) helps guard against this, (Qian & Wegman, 2019) shows that an L ∞ -trained model that appeared robust against L 2 actually had substantially less robustness when evaluating with 10 6 PGD steps.

If this effect is pervasive, then there may be even less transfer between attacks than our current results suggest.

For evaluating against a fixed attack, DeepFool Moosavi-Dezfooli et al. (2015) and CLEVER Weng et al. (2018) can be seen as existing alternatives to UAR.

They work by estimating "empirical robustness", which is the expected minimum ε needed to successfully attack an image.

However, these apply only to attacks which optimize over an L p -ball of radius ε, and CLEVER can be susceptible to gradient masking Goodfellow (2018).

In addition, empirical robustness is equivalent to linearly averaging accuracy over ε, which has smaller dynamic range than the geometric average in UAR.

Our results add to a growing line of evidence that evaluating against a single known attack type provides a misleading picture of the robustness of a model (Sharma & Chen, 2017; Engstrom et al., 2017; Jordan et al., 2019; Tramèr & Boneh, 2019; Jacobsen et al., 2019) .

Going one step further, we believe that robustness itself provides only a narrow window into model behavior; in addition to robustness, we should seek to build a diverse toolbox for understanding machine learning models, including visualization (Olah et al., 2018; Zhang & Zhu, 2019) , disentanglement of relevant features (Geirhos et al., 2018) , and measurement of extrapolation to different datasets (Torralba & Efros, 2011) or the long tail of natural but unusual inputs (Hendrycks et al., 2019) .

Together, these windows into model behavior can give us a clearer picture of how to make models reliable in the real world.

For ImageNet-100, we trained on machines with 8 NVIDIA V100 GPUs using standard data augmentation He et al. (2016) .

Following best practices for multi-GPU training Goyal et al. (2017), we ran synchronized SGD for 90 epochs with batch size 32×8 and a learning rate schedule with 5 "warm-up" epochs and a decay at epochs 30, 60, and 80 by a factor of 10.

Initial learning rate after warm-up was 0.1, momentum was 0.9, and weight decay was 10 −4 .

For CIFAR-10, we trained on a single NVIDIA V100 GPU for 200 epochs with batch size 32, initial learning rate 0.1, momentum 0.9, and weight decay 10 −4 .

We decayed the learning rate at epochs 100 and 150.

We show the images corresponding to the ones in Figure 2 , with the exception that they are not scaled.

The non-scaled images are shown in Figure 9 .

We chose to use the Frank-Wolfe algorithm for optimizing the L 1 attack, as Projected Gradient Descent would require projecting onto a truncated L 1 ball, which is a complicated operation.

In contrast, Frank-Wolfe only requires optimizing linear functions g x over a truncated L 1 ball; this can be done by sorting coordinates by the magnitude of g and moving the top k coordinates to the boundary of their range (with k chosen by binary search).

This is detailed in Algorithm 1.

We will present results with two additional versions of the JPEG attack which impose L 1 or L 2 constraints on the attack in JPEG-space instead of the L ∞ constraint discussed in Section 2.

To avoid confusion, in this appendix, we denote the original JPEG attack by L ∞ -JPEG and these variants by New Attacks JPEG (5.4m, 18.7k, 255) Fog (4.1m, 13.2k, 89) Gabor (3.7m, 13.3k, 50) Snow (11.4m, 32.0k, 255) Figure 9: Differences of the attacked images and original image for different attacks (label "espresso maker").

The L 1 , L 2 , and L ∞ norms of the difference are shown in parentheses.

As shown, our novel attacks display qualitatively different behavior and do not fall under the L p threat model.

These differences are not scaled and are normalized so that no difference corresponds to white.

Algorithm 1 Pseudocode for the Frank-Wolfe algorithm for the L 1 attack.

s k ← index of the coordinate of g by with k th largest norm 9:

end for 10:

S k ← {s 1 , . . .

, s k }.

12:

else 16:

end if

end for 19:

20:

21:

Averagex with previous iterates 31: end for 32:x ← x (T ) we find that they have extremely similar results, so we omit L 1 -JPEG in the full analysis for brevity and visibility.

Calibration values for these attacks are shown in Table 2 .

We show the full results of all adversarial attacks against all adversarial defenses for ImageNet-100 in Figure 11 .

As described, the L p attacks and defenses give highly correlated information on heldout defenses and attacks respectively.

Thus, we recommend evaluating on a wide range of distortion types.

Full UAR scores are also provided for ImageNet-100 in Figure 12 .

We further show selected results in Figure 13 .

As shown, a wide range of ε is required to see the full behavior.

Attack (adversarial training) Normal training

Gabor ε = 6.25

Gabor ε = 12.5

Gabor ε = 25

Gabor ε = 400

Gabor ε = 800

Gabor ε = 1600

Figure 13: Adversarial accuracies of attacks on adversarially trained models for different distortion sizes on ImageNet-100.

For a given attack ε, the best ε to train against satisfies ε > ε because the random scaling of ε during adversarial training ensures that a typical distortion during adversarial training has size smaller than ε .

We show the results of adversarial attacks and defenses for CIFAR-10 in Figure 14 .

We experienced difficulty training the L 2 and L 1 attacks at distortion sizes greater than those shown and have omitted those runs, which we believe may be related to the small size of CIFAR-10 images.

The ε calibration procedure for CIFAR-10 was similar to that used for ImageNet-100.

We started with the perceptually small ε min values in Table 3 and increased ε geometrically with ratio 2 until adversarial accuracy of an adversarially trained model dropped below 40.

Note that this threshold Table 3 and Figure 15 .

We omitted calibration for the L 2 -JPEG attack because we chose too small a range of ε for our initial training experiments, and we plan to address this issue in the future.

We replicated our results for the first three rows of Figure 11 with different random seeds to see the variation in our results.

As shown in Figure 16 , deviations in results are minor.

We replicated the results in Figure 11 with 50 instead of 200 steps to see how the results changed based on the number of steps in the attack.

As shown in Figure 17 , the deviations are minor.

We show the evaluation accuracies of jointly trained models in Figure 18 .

We show all the attacks against the jointly adversarially trained defenses in Figure 19 .

In Table 4 , we study the dependence of joint adversarial training to random seed.

We find that at large distortion sizes, joint training for certain pairs of distortions does not produce consistent results over different random initializations.

Table 4 : Train and val accuracies for joint adversarial training at large distortion are dependent on seed.

For train and val, ε is chosen uniformly at random between 0 and ε, and we used 10 steps for L ∞ and L 1 and 30 steps for elastic.

Single adversarial training baselines are also shown.

As a first test to understand the relationship between model capacity and overfitting, we trained ResNet-101 models using the same procedure as in Section 5.

Briefly, overfitting still occurs, but ResNet-101 achieves a few percentage points higher than ResNet-50.

We show the training curves in Figure 20 and the training and validation numbers in Table 5 .

"Common" visual corruptions such as (non-adversarial) fog, blur, or pixelation have emerged as another avenue for measuring the robustness of computer vision models (Vasiljevic et al., 2016; Hendrycks & Dietterich, 2019; Geirhos et al., 2018) .

Recent work suggests that robustness to such common corruptions is linked to adversarial robustness and proposes corruption robustness as an easily computed indicator of adversarial robustness (Ford et al., 2019) .

We consider this alternative to our methodology by testing corruption robustness of our models on the ImageNet-C benchmark.

Experimental setup.

We evaluate on the 100-class subset of the corruption robustness benchmark ImageNet-C introduced in (Hendrycks & Dietterich, 2019) with the same classes as ImageNet-100, which we call ImageNet-C-100.

It is the ImageNet-100 validation set with 19 common corruptions at 5 severities.

We use the JPEG files available at https://github.com/hendrycks/ robustness.

We show average accuracies by distortion type in Figure 21 .

Adversarial training against small distortions increases corruption robustness.

The first column of each block in Figure 21 shows that training against small adversarial distortions generally increases average accuracy compared to an undefended model.

However, training against larger distortions often decreases average accuracy, largely due to the resulting decrease in clean accuracy.

Adversarial distortions and common corruptions can affect defenses differently.

Our L p -JPEG and elastic attacks are adversarial versions of the corresponding common corruptions.

While training against adversarial JPEG at larger ε improves robustness against adversarial JPEG attacks (Figure 12 in Appendix C.2), Figure 21 shows that robustness against common JPEG corruptions decreases as we adversarially train against JPEG at larger ε, though it remains better than for normally trained models.

Similarly, adversarial Elastic training at large ε begins to hurt robustness to its common counterpart.

This is likely because common corruptions are easier than adversarial distortions, hence the increased robustness does not make up for the decreased clean accuracy.

We show sample images of our attacks against undefended models trained in the normal way in Figure 22 .

Normal training

Elastic ε = 0.125 Figure 15: UAR scores on CIFAR-10.

Displayed UAR scores are multiplied by 100 for clarity.

Attack (evaluation)

L2 ε = 150 L2 ε = 300 L2 ε = 600 L2 ε = 1200 L2 ε = 2400 L2 ε = 4800 Normal training B ri g h tn e ss C o n tr a st E la st ic P ix e la te J P E G S p e c k le N o is e G a u ss ia n B lu r S p a tt e r S a tu ra te

Normal training L∞ ε = 1 L∞ ε = 2 L∞ ε = 4 L∞ ε = 8 L∞ ε = 16 L∞ ε = 32 L2 ε = 150 L2 ε = 300 L2 ε = 600 L2 ε = 1200 L2 ε = 2400 L2 ε = 4800 L1 ε = 9562.44 L1 ε = 19125 L1 ε = 76500 L1 ε = 153000 L1 ε = 306000 L1 ε = 612000

@highlight

We propose several new attacks and a methodology to measure robustness against unforeseen adversarial attacks.