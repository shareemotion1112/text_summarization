The worst-case training principle that minimizes the maximal adversarial loss, also known as adversarial training (AT), has shown to be a state-of-the-art approach for enhancing adversarial robustness against norm-ball bounded input perturbations.

Nonetheless, min-max optimization beyond the purpose of AT has not been rigorously explored in the research of adversarial attack and defense.

In particular, given a set of risk sources (domains), minimizing the maximal loss induced from the domain set can be reformulated as a general min-max problem that is different from AT.

Examples of this general formulation include attacking model ensembles, devising universal perturbation under multiple inputs or data transformations, and generalized AT over different types of attack models.

We show that these problems can be solved under a unified and theoretically principled min-max optimization framework.

We also show that the self-adjusted domain weights learned from our method provides a means to explain the difficulty level of attack and defense over multiple domains.

Extensive experiments show that our approach leads to substantial performance improvement over the conventional averaging strategy.

Training a machine learning model that is capable of assuring its worst-case performance against all possible adversaries given a specified threat model is a fundamental yet challenging problem, especially for deep neural networks (DNNs) (Szegedy et al., 2013; Goodfellow et al., 2015; Carlini & Wagner, 2017) .

A common practice to train an adversarially robust model is based on a specific form of min-max training, known as adversarial training (AT) (Goodfellow et al., 2015; Madry et al., 2017) , where the minimization step learns model weights under the adversarial loss constructed at the maximization step in an alternative training fashion.

On datasets such as MNIST and CIFAR-10, AT has achieved the state-of-the-art defense performance against p -norm-ball input perturbations (Athalye et al., 2018b) .

Motivated by the success of AT, one follow-up question that naturally arises is: Beyond AT, can other types of min-max formulation and optimization techniques advance the research in adversarial robustness?

In this paper, we give an affirmative answer corroborated by the substantial performance gain and the ability of self-learned risk interpretation using our proposed min-max framework on several tasks for adversarial attack and defense.

We demonstrate the utility of a general formulation for min-max optimization minimizing the maximal loss induced from a set of risk sources (domains).

Our considered min-max formulation is fundamentally different from AT, as our maximization step is taken over the probability simplex of the set of domains.

Moreover, we show that many problem setups in adversarial attacks and defenses can in fact be reformulated under this general min-max framework, including attacking model ensembles (Tramèr et al., 2018; Liu et al., 2018) , devising universal perturbation to input samples (Moosavi-Dezfooli et al., 2017) or data transformations (Athalye & Sutskever, 2018; Brown et al., 2017) , and generalized AT over multiple types of threat models (Tramèr & Boneh, 2019; Araujo et al., 2019 ).

However, current methods for solving these tasks often rely on simple heuristics (e.g.,

Work Recent studies have identified that DNNs are highly vulnerable to adversarial manipulations in various applications (Szegedy et al., 2013; Carlini et al., 2016; Jia & Liang, 2017; Lin et al., 2017; Huang et al., 2017; Carlini & Wagner, 2018; Zhao et al., 2018; Eykholt et al., 2018; Chen et al., 2018a; Lei et al., 2019) , thus leading to an arms race between adversarial attacks (Carlini & Wagner, 2017; Athalye et al., 2018b; Goodfellow et al., 2014; Papernot et al., 2016a; Moosavi Dezfooli et al., 2016; Chen et al., 2018b; Xu et al., 2019) and defenses (Madry et al., 2017; Papernot et al., 2016b; Meng & Chen, 2017; Xie et al., 2017; Xu et al., 2018) .

One intriguing property of adversarial examples is the transferability across multiple domains (Liu et al., 2017; Tramèr et al., 2017; Papernot et al., 2017; Su et al., 2018) , which indicates a more challenging yet promising research directiondevising universal adversarial perturbations over model ensembles (Tramèr et al., 2018; Liu et al., 2018) , input samples (Moosavi-Dezfooli et al., 2017; Metzen et al., 2017; Shafahi et al., 2018) and data transformations (Athalye et al., 2018b; Athalye & Sutskever, 2018; Brown et al., 2017 ).

However, current approaches suffer from a significant performance loss for resting on the uniform averaging strategy.

We will compare these works with our min-max method in Sec. 4.

As a natural extension following min-max attack, we study the generalized AT under multiple perturbations (Tramèr & Boneh, 2019; Araujo et al., 2019; Kang et al., 2019; Croce & Hein, 2019) .

Finally, our min-max framework is adapted and inspired by previous literature on robust learning over multiple domains (Qian et al., 2018; Rafique et al., 2018; Lu et al., 2018; 2019a) .

We begin by introducing the principle of robust learning over multiple domains and its connection to a specialized form of min-max optimization.

We then show that the resulting min-max formulation fits into various attack settings for adversarial exploration: a) ensemble adversarial attack, b) universal adversarial perturbation and c) robust perturbation over data transformations.

Finally, we propose a generalized adversarial training (AT) framework under mixed types of adversarial attacks to improve model robustness.

Consider K loss functions {F i (v)} (each of which is defined on a learning domain), the problem of robust learning over K domains can be formulated as (Qian et al., 2018; Rafique et al., 2018; Lu et al., 2018)

where v and w are optimization variables, V is a constraint set, and P denotes the probability simplex P = {w | 1 T w = 1, w i ∈ [0, 1], ∀i}. Since the inner maximization problem in (1) is a linear function of w over the probabilistic simplex, problem (1) is thus equivalent to

where [K] denotes the integer set {1, 2, . . .

, K}.

Benefit and computation challenge of (1) Compared to multi-task learning in a finite-sum formulation which minimizes K losses on average, problem (1) provides consistently robust worst-case performance across all domains.

This can be explained from the epigraph form of (2), minimize v∈V,t t, subject to

where t is an epigraph variable (Boyd & Vandenberghe, 2004 ) that provides the t-level robustness at each domain.

Although the min-max problem (1) offers a great robustness interpretation as in (3), solving it becomes more challenging than solving the finite-sum problem.

It is clear from (2) that the inner maximization problem of (1) always returns the one-hot value of w, namely, w = e i , where e i is the ith standard basis vector, and i = arg max i {F i (v)}.

The one-hot coding reduces the generalizability to other domains and induces instability of the learning procedure in practice.

Such an issue is often mitigated by introducing a strongly concave regularizer in the inner maximization step (Lu et al., 2018; Qian et al., 2018) .

Regularized problem formulation Spurred by (Qian et al., 2018) , we penalize the distance between the worst-case loss and the average loss over K domains.

This yields

where γ > 0 is a regularization parameter.

As γ → 0, problem (4) is equivalent to (1).

By contrast, it becomes the finite-sum problem when γ → ∞ since w → 1/K. In this sense, the trainable w provides an essential indicator on the importance level of each domain.

The larger the weight is, the more important the domain is.

We call w domain weights in this paper.

We next show how the principle of robust learning over multiple domains can fit into various settings of adversarial attack and defense problems.

The general goal of adversarial attack is to craft an adversarial example x = x 0 + δ ∈ R d to mislead the prediction of machine learning (ML) or deep learning (DL) systems, where x 0 denotes the natural example with the true label t 0 , and δ is known as adversarial perturbation, commonly subject to

given small number .

Here the p norm enforces the similarity between x and x 0 , and the input space of ML/DL systems is normalized to [0, 1] d .

Ensemble attack over multiple models Consider K ML/DL models

, the goal is to find robust adversarial examples that can fool all K models simultaneously.

In this case, the notion of 'domain' in (4) is specified as 'model', and the objective function F i in (4) signifies the attack loss f (δ; x 0 , y 0 , M i ) given the natural input (x 0 , y 0 ) and the model M i .

Thus, problem (4) becomes

where w encodes the difficulty level of attacking each model.

and a single model M, our goal is to find the universal perturbation δ so that all the corrupted K examples can fool M. In this case, the notion of 'domain' in (4) is specified as 'example', and problem (4) becomes

where different from (5), w encodes the difficulty level of attacking each example.

Adversarial attack over data transformations Consider K categories of data transformation {p i }, e.g., rotation, lightening, and translation (Athalye et al., 2018a) , our goal is to find the adversarial attack that is robust to data transformations.

In this case, the notion of 'domain' in (4) is specified as 'data transformer', and problem (4) becomes

where E t∼pi [f (t(x 0 +δ); y 0 , M)] denotes the attack loss under the distribution of data transformation p i , and w encodes the difficulty level of attacking each type of transformed example x 0 .

Conventional AT is restricted to a single type of norm-ball constrained adversarial attack (Madry et al., 2017) .

For example, AT under ∞ attack yields minimize

where θ ∈ R n denotes model parameters, δ denotes -tolerant ∞ attack, and f tr (θ, δ; x, y) is the training loss under perturbed examples {(x + δ, y)}. However, there possibly exist blind attacking spots across multiple types of adversarial attacks so that AT under one attack would not be strong enough against another attack (Araujo et al., 2019) .

Thus, an interesting question is how to generalize AT under multiple types of adversarial attacks.

One possible way is to use the finite-sum formulation

where δ i ∈ X i is the ith type of adversarial perturbation defined on X i , e.g., different p attacks.

Moreover, one can map 'attack type' to 'domain' considered in (1).

We then perform AT against the strongest adversarial attack across K attack types in order to avoid blind attacking spots.

That is, upon defining F i (θ) := maximize δi∈Xi f tr (θ, δ i ; x, y), we solve the problem of the form (2), minimize

In fact, problem (10) is in the min-max-max form, however, Lemma 1 shows that problem (10) can be further simplified to the min-max form.

Lemma 1.

Problem (10) is equivalent to

where w ∈ R K represent domain weights, and P has been defined in (1).

Proof: see Appendix A.

Similar to (4), a strongly concave regularizer −γ/2 w − 1/K 2 2 can be added into the inner maximization problem of (11), which can boost the stability of the learning procedure and strike a balance between the max and the average attack performance.

However, solving problem (11) and its regularized version is more complicated than (8) since the inner maximization involves both domain weights w and adversarial perturbations {δ i }.

We finally remark that there was an independent work (Tramèr & Boneh, 2019) which also proposed the formulation (10) for AT under multiple perturbations.

However, what we propose here is the regularized formulation of (11).

As will be evident later, the domain weights w in our formulation have strong interpretability, which learns the importance level of different attacks.

Most significantly, our work has different motivation from (Tramèr & Boneh, 2019) , and our idea applies to not only AT but also attack generation in Sec. 2.2.

In this section, we delve into technical details on how to efficiently solve problems of robust adversarial attacks given by the generic form (4) and problem (11) for generalized AT under mixed types of adversarial attacks.

We propose the alternating one-step projected gradient descent (APGD) method (Algorithm 1) to solve problem (4).

For clarity, we repeat problem (4) under the adversarial perturbation δ and its constraint set X defined in Sec. 2.2,

We show that at each iteration, APGD takes only one-step PGD for outer minimization and one-step projected gradient ascent for inner maximization (namely, PGD for its negative objective function).

We also show that each alternating step has a closed-form expression, and the main computational complexity stems from computing the gradient of the attack loss w.r.t.

the input.

Therefore, APGD is computationally efficient like PGD, which is commonly used for design of conventional single p -norm based adversarial attacks (Madry et al., 2017) .

Outer minimization Considering w = w (t−1) and (4), we perform one-step PGD to update δ at iteration t,

where proj(·) denotes the Euclidean projection operator, i.e., proj X (a) = arg min x∈X x − a 2 2 at the point a, α > 0 is a given learning rate, and ∇ δ denotes the first order gradient w.r.t.

δ.

In (13), the projection operation becomes the key to obtain the closed-form of the updating rule (13).

Recall from Sec. 2.2 that X = {δ| δ p ≤ ,č ≤ δ ≤ĉ}, where p ∈ {0, 1, 2, ∞}, andč = −x 0 andĉ = 1 − x 0 (implyingč ≤ 0 ≤ĉ).

If p = ∞, then the projection function becomes the clip function.

However, when p ∈ {0, 1, 2}, the closed-form of projection operation becomes non-trivial.

In Proposition 1, we derive the solution of proj X (a) under different p norms.

Proposition 1.

Given a point a ∈ R d and a constraint set X = {δ| δ p ≤ ,č ≤ δ ≤ĉ}, the Euclidean projection δ * = proj X (a) has a closed-form solution when p ∈ {0, 1, 2}.

Proof: See Appendix B.

Inner maximization By fixing δ = δ (t) and letting ψ(w) :

in problem (4), we then perform one-step PGD (w.r.t.

−ψ) to update w,

where β > 0 is a given learning rate,

In (14), the second equality holds due to the closed-form of projection operation onto the probabilistic simplex P (Parikh et al., 2014) , where (·) + denotes the elementwise nonnegative operator, i.e., (x) + = max{0, x}, and µ is the root of the equation 1

, the root µ exists within the interval [min i {b i } − 1/K, max i {b i } − 1/K] and can be found via the bisection method (Boyd & Vandenberghe, 2004) .

Convergence analysis We remark that APGD follows the gradient primal-dual optimization framework (Lu et al., 2019a) , and thus enjoys the same optimization guarantees.

In Theorem 1, we demonstrate the convergence rate of Algorithm 1 for solving problem (4).

Theorem 1. (inherited from primal-dual min-max optimization) Suppose that in problem (4) F i (δ) has L-Lipschitz continuous gradients, and X is a convex compact set.

Given learning rates α ≤ We next propose the alternating multi-step projected gradient descent (AMPGD) method to solve the regularized version of problem (11), which is repeated as follows

Algorithm 2 AMPGD to solve problem (15)

given w (t−1) and δ (t−1) , perform SGD to update θ (t) (Madry et al., 2017) 4:

given θ (t) , perform R-step PGD to update w (t) and δ (t)

Problem (15) is in a more general non-convex non-concave min-max setting, where the inner maximization involves both domain weights w and adversarial perturbations {δ i }.

It was shown in (Nouiehed et al., 2019) that the multi-step PGD is required for inner maximization in order to approximate the near-optimal solution.

This is also in the similar spirit of AT (Madry et al., 2017) , which executed multi-step PGD attack during inner maximization.

We summarize AMPGD in Algorithm 2.

At step 4 of Algorithm 2, each PGD step to update w and δ can be decomposed as

where let w

1 := w (t−1) and δ

.

Here the subscript t represents the iteration index of AMPGD, and the subscript r denotes the iteration index of R-step PGD.

Clearly, the above projection operations can be derived for closed-form expressions through (14) and Lemma 1.

To the best of our knowledge, it is still an open question to build theoretical convergence guarantees for solving the general non-convex non-concave min-max problem like (15), except the work (Nouiehed et al., 2019) which proposed O(1/T ) convergence rate if the objective function satisfies Polyak-Łojasiewicz conditions (Karimi et al., 2016) .

Improved robustness via diversified p attacks.

It was recently shown in (Kariyappa & Qureshi, 2019; Pang et al., 2019) that the diversity of individual neural networks improves adversarial robustness of an ensemble model.

Spurred by that, one may wonder if the promotion of diversity among p attacks is beneficial to adversarial robustness?

We measure the diversity between adversarial attacks through the similarity between perturbation directions, namely, input gradients {∇ δi f tr (θ, δ i ; x, y)} i in (15).

We find that there exists a strong correlation between input gradients for different p attacks.

Thus, we propose to enhance their diversity through the orthogonality-promoting regularizer used for encouraging diversified prediction of ensemble models in (Pang et al., 2019) ,

where G ∈ R d×K is a d × K matrix, each column of which corresponds to a normalized input gradient ∇ δi f tr (θ, δ i ; x, y) for i ∈ [K], and h(θ, {δ i }; x, y) reaches the maximum value 0 as input gradients become orthogonal.

With the aid of (16), we modify problem (15) to minimize

The rationale behind (17) is that the adversary aims to enhance the effectiveness of attacks from diversified perturbation directions (inner maximization), while the defender robustifies the model θ, which makes diversified attacks less effective (outer minimization).

In this section, we first evaluate the proposed min-max optimization strategy on three attack tasks.

We show that our approach leads to substantial improvement compared with state-of-the-art attack methods such as ensemble PGD (Liu et al., 2018) and expectation over transformation (EOT) (Athalye et al., 2018b; Brown et al., 2017; Athalye et al., 2018a ).

We next demonstrate the effectiveness of the generalized AT for multiple types of adversarial perturbations.

We show that the use of trainable domain weights in problem (15) can automatically adjust the risk level of different attacks during the training process even if the defender lacks prior knowledge on the strength of these attacks.

We also show that the promotion of diversity of p attacks help improve adversarial robustness further.

We thoroughly evaluate our APGD/AMPGD algorithm on MNIST and CIFAR-10.

A set of diverse image classifiers (denoted from Model A to Model H) are trained, including multi-layer perceptrons (

Most current works play a min-max game from a defender's perspective, i.e., adversarial training.

However, we show the great strength of min-max optimization also lies at the side of attack generation.

Note that problem formulations (5)-(7) are applicable to both untargeted and targeted attack.

Here we focus on the former setting and use C&W loss function (Carlini & Wagner, 2017; Madry et al., 2017) .

The details of crafting adversarial examples are available in Appendix C.2.

Ensemble attack over multiple models We craft adversarial examples against an ensemble of known classifiers.

The work (Liu et al., 2018 , 5th place at CAAD-18) proposed an ensemble PGD attack, which assumed equal importance among different models, namely, w i = 1/K in problem (1).

Throughout this task, we measure the attack performance via ASR all -the attack success rate (ASR) of fooling model ensembles simultaneously.

Compared to the ensemble PGD attack (Liu et al., 2018) , our approach results in 40.79% and 17.48% ASR all improvement averaged over different p -norm constraints on MNIST and CIFAR-10, respectively.

In what follows, we provide more detailed experiment results and analysis.

In Table 1 , we show that our min-max APGD significantly outperforms ensemble PGD in ASR all .

Taking ∞ -attack on MNIST as an example, our min-max attack leads to a 90.16% ASR all , which largely outperforms 48.17% (ensemble PGD).

The reason is that Model C, D are more difficult to attack, which can be observed from their higher test accuracy on adversarial examples.

As a result, although the adversarial examples crafted by assigning equal weights over multiple models are able to attack {A, B} well, they achieve a much lower ASR (i.e., 1 -Acc) in {C, D}. By contrast, APGD automatically handles the worst case {C, D} by slightly sacrificing the performance on {A, B}: 31.47% averaged ASR improvement on {C, D} versus 0.86% degradation on {A, B}. More results on CIFAR-10 and more complicated DNNs (e.g., GoogLeNet) are provided in Appendix D. Lastly, we highlight that tracking domain weights w provides us novel insights for model robustness and understanding attack procedure.

From our theory, a model with higher robustness always corresponds to a larger w because its loss is hard to attack and becomes the "worst" term.

This hypothesis can be verified empirically.

According to Figure 1c , we have w c > w d > w a > w bindicating a decrease in model robustness for C, D, A and B, which is exactly verified by Acc C > Acc D > Acc A > Acc B in Table 1 ( ∞ -norm).

Universal perturbation over multiple examples We evaluate APGD in universal perturbation on MNIST and CIFAR-10, where 10,000 test images are randomly divided into equal-size groups (containing K images per group) for universal perturbation.

We measure two types of ASR (%), ASR avg and ASR gp .

Here the former represents the ASR averaged over all images in all groups, and the latter signifies the ASR averaged over all groups but a successful attack is counted under a more restricted condition: images within each group must be successfully attacked simultaneously by universal perturbation.

When K = 5, our approach achieves 42.63% and 35.21% improvement over the averaging strategy under MNIST and CIFAR-10, respectively.

In Table 2 , we compare the proposed min-max strategy (APGD) with the averaging strategy on the attack performance of generated universal perturbations.

As we can see, our method always achieves higher ASR gp for different values of K. The universal perturbation generated from APGD can successfully attack 'hard' images (on which the average-based PGD attack fails) by self-adjusting domain weights, and thus leads to a higher ASR gp .

Besides, the min-max universal perturbation also offers interpretability of "image robustness" by associating domain weights with image visualization; see Figure A9 and A10 (Appendix F) for an example in which the large domain weight corresponds to the MNIST letter with clear appearance (e.g., bold letter).

Robust adversarial attack over data transformations EOT (Athalye et al., 2018a ) achieves stateof-the-art performance in producing adversarial examples robust to data transformations.

From (7), we could derive EOT as a special case when the weights satisfy w i = 1/K (average case).

For each input sample (ori), we transform the image under a series of functions, e.g., flipping horizontally and γ = 4. (flh) or vertically (flv), adjusting brightness (bri), performing gamma correction (gam) and cropping (crop), and group each image with its transformed variants.

Similar to universal perturbation, ASR avg and ASR gp are reported to measure the ASR over all transformed images and groups of transformed images (each group is successfully attacked signifies successfully attacking an example under all transformers).

In Table 3 , compared to EOT, our approach leads to 9.39% averaged lift in ASR gp over given models on CIFAR-10 by optimizing the weights for various transformations.

Due to limited space, we leave the details of transformers in Append C.3 and the results under randomness (e.g., flipping images randomly w.p.

0.8; randomly clipping the images at specific range) in Appendix D.

Compared to vanilla AT, we show the generalized AT scheme produces models robust to multiple types of perturbation, thus leads to stronger "overall robustness".

We measure the training performance using two types of Acc (%): Acc max adv and Acc avg adv , where Acc max adv denotes the test accuracy over examples with the strongest perturbation ( ∞ or 2 ), and Acc avg adv denotes the averaged test accuracy over examples with all types of perturbations ( ∞ and 2 ).

Moreover, we measure the overall worst-case robustness S in terms of the area under the curve 'Acc max adv vs. ' (see Figure 3b) .

In Table 4 , we present the test accuracy of MLP in different training schemes: a) natural training, b) single-norm: vanilla AT ( ∞ or 2 ), c) multi-norm: generalized AT (avg and min max), and d) generalized AT with diversity-promoting attack regularization (DPAR, λ = 0.1 in problem (16)).

If the adversary only performs single-type attack, training and testing on the same attack type leads to the best performance (diagonal of ∞ -2 block).

However, when facing ∞ and 2 attacks simultaneously, multi-norm generalized AT achieves better Acc max adv and Acc avg adv than single-norm AT.

In particular, the min-max strategy slightly outperforms the averaging strategy under multiple perturbation norms.

Figure A2 (Appendix E).

DPAR further boosts the adversarial test accuracy, which implies that the promotion of diversified p attacks is a beneficial supplement to adversarial training.

In Figure 3 , we offer deeper insights on the performance of generalized AT.

During the training procedure we fix ∞ ( for ∞ attack during training) as 0.2, and change 2 from 0.2 to 5.6 ( ∞ × √ d) so that the ∞ and 2 balls are not completely overlapped (Araujo et al., 2019) .

In Figure 3a , as 2 increases, 2 -attack becomes stronger so the corresponding w also increases, which is consistent with min-max spirit -defending the strongest attack.

We remark that min max or avg training does not always lead to the best performance on Acc max adv and Acc avg adv , especially when the strengths of two attacks diverge greatly (see Table A8 ).

This can be explained by the large overlapping between ∞ and 2 balls (see Figure A3 ).

However, Figure 3b and 3c show that AMPGD is able to achieve a rather robust model no matter how changes (red lines), which empirically verifies the effectiveness of our proposed training scheme.

In terms of the area-under-the-curve measure S , AMPGD achieves the highest worst-case robustness: 6.27% and 17.64% improvement compared to the vanilla AT with ∞ and 2 attacks.

Furthermore, we show in Figure A4a that our min-max scheme leads to faster convergence than the averaging scheme due to the benefit of self-adjusted domain weights.

In this paper, we propose a general min-max framework applicable to both adversarial attack and defense settings.

We show that many problem setups can be re-formulated under this general framework.

Extensive experiments show that proposed algorithms lead to significant improvement on multiple attack and defense tasks compared with previous state-of-the-art approaches.

In particular, we obtain 17.48%, 35.21% and 9.39% improvement on attacking model ensembles, devising universal perturbation to input samples, and data transformations under CIFAR-10, respectively.

Our minmax scheme also generalizes adversarial training (AT) for multiple types of adversarial attacks, attaining faster convergence and better robustness compared to the vanilla AT and the average strategy.

Moreover, our approach provides a holistic tool for self-risk assessment by learning domain weights.

where w ∈ R K represent domain weights, and P has been defined in (1).

Similar to (1), problem (10) is equivalent to

Recall that F i (θ) := maximize δi∈Xi f tr (θ, δ i ; x, y), problem can then be written as

According to proof by contradiction, it is clear that problem (19) is equivalent to

B PROOF OF PROPOSITION 1 Proposition 1.

Given a point a ∈ R d and a constraint set X = {δ| δ p ≤ ,č ≤ δ ≤ĉ}, the Euclidean projection δ * = proj X (a) has the closed-form solution when p ∈ {0, 1, 2}.

1) If p = 1, then δ * is given by

where x i denotes the ith element of a vector x; P [či,ĉi] (·) denotes the clip function over the interval

where λ 2 ∈ (0, a 2 / − 1] is the root of

3) If p = 0 and ∈ N + , then δ * is given by

where [η] denotes the -th largest element of η, and δ i = P [či,ĉi] (a i ).

1 norm When we find the Euclidean projection of a onto the set X , we solve

where I [č,ĉ] (·) is the indicator function of the set [č,ĉ].

The Langragian of this problem is

The minimizer δ * minimizes the Lagrangian, it is obtained by elementwise soft-thresholding

where x i is the ith element of a vector x, P [či,ĉi] (·) is the clip function over the interval

The primal, dual feasibility and complementary slackness are

, where λ 1 is given by the root of the equation

Bisection method can be used to solve the above equation for λ 1 , starting with the initial interval (0, max

2 norm When we find the Euclidean projection of a onto the set X , we solve

where I [č,ĉ] (·) is the indicator function of the set [č,ĉ].

The Langragian of this problem is

The minimizer δ * minimizes the Lagrangian, it is

The primal, dual feasibility and complementary slackness are

λ2+1 a i , where λ 2 is given by the root of the equation

Bisection method can be used to solve the above equation for λ 2 , starting with the initial interval (0,

2 > 2 in this case, and

0 norm For 0 norm in X , it is independent to the box constraint.

So we can clip a to the box constraint first, which is δ i = P [či,ĉi] (a i ), and then project it onto 0 norm.

We find the additional Euclidean distance of every element in a and zero after they are clipped to the box constraint, which is

It can be equivalently written as

To derive the Euclidean projection onto 0 norm, we find the -th largest element in η and call it [η] .

We keep the elements whose corresponding η i is above or equals to -th, and set rest to zeros.

The closed-form solution is given by .1) that the problem is convex and the solution can be derived using KKT conditions.

However, Proposition 1 in our paper is different from (Hein & Andriushchenko, 2017, Proposition 4.1).

First, we place p norm as a hard constraint rather than minimizing it in the objective function.

This difference will make our Lagrangian function more involved with a newly introduced nonnegative Lagrangian multiplier.

Second, the problem of our interest is projection onto the intersection of box and p constraints.

Such a projection step can then be combined with an attack loss (no need of linearization) for generating adversarial examples.

Third, we cover the case of 0 norm.

C EXPERIMENT SETUP (2015) .

For the last four models, we use the exact same architecture as original papers and evaluate them only on CIFAR-10 dataset.

The details for model architectures are provided in Table A1 .

For compatibility with our framework, we implement and train these models based on the strategies adopted in pytorch-cifar classifiers for 50 epochs with Adam and a constant learning rate of 0.001.

For CIFAR-10 classifers, the models are trained for 250 epochs with SGD (using 0.8 nesterov momentum, weight decay 5e −4 ).

The learning rate is reduced at epoch 100 and 175 with a decay rate of 0.1.

The initial learning rate is set as 0.01 for models {A, B, C, D, H} and 0.1 for {E, F, G}. Note that no data augmentation is employed in the training. (2017) with a confidence parameter κ = 50.

Cross-entropy loss is also supported in our implementation.

The adversarial examples are generated by 20-step PGD/APGD unless otherwise stated (e.g., 50 steps for ensemble attacks).

Note that proposed algorithms are robust and will not be affected largely by the choices of hyperparameters (α, β, γ).

In consequence, we do not finely tune the parameters on the validation set.

Specifically, The learning rates α, β and regularization factor γ for Table 1 Moreover, both deterministic and stochastic transformations are considered in our experiments.

In particular, Table 3 and Table A6 are deterministic settings -rot: rotating images 30 degree clockwise; crop: cropping images in the center (0.8 × 0.8) and resizing them to 32 × 32; bri: adjusting the brightness of images with a scale of 0.1; gam: performing gamma correction with a value of 1.3.

Differently, in Table A5 , we introduce randomness for drawing samples from the distribution -rot: rotating images randomly from -10 to 10 degree; crop: cropping images in the center randomly (from 0.6 to 1.0); other transformations are done with a probability of 0.8.

In experiments, we adopt tf.image API 7 for processing the images.

Table A3 shows the performance of average (ensemble PGD Liu et al. (2018)) and min-max (APGD) strategies for attacking model ensembles.

Our min-max approach results in 15.69% averaged improvement on ASR all over models {A, E, F, H} on CIFAR-10.

, γ = 6.

The attack iteration for APGD is set as 50.

Opt.

To further demonstrate the effectiveness of self-adjusted weighting factors in proposed min-max framework, we compare with heuristic weighting schemes in Table A4 Table A4 shows that our min-max approach outperforms all static heuristic weighting schemes by a large margin.

Specifically, our min-max APGD also achieves significant improvement compared to w static setting, where the converged optimal weights are statically (i.e., invariant w.r.t different images and attack procedure) adopted.

It again verifies the benefits of proposed min-max approach by automatically learning the weights for different examples during the process of ensemble attack generation (see Figure 1c ).

Table A5 and A6 compare the performance of average (EOT Athalye et al. (2018a) ) and min-max (APGD) strategies.

Our approach results in 4.31% and 8.22% averaged lift over four models {A, B, C, D} on CIFAR-10 under given stochastic and deterministic transformation sets.

and γ = 10.

To further explore the utility of quadratic regularizer on the probability simplex in proposed min-max framework, we conducted sensitivity analysis on γ and show how the proposed regularization affects the eventual performance ( Figure A1 ) taking ensemble attack as an example.

The experimental setting is the same as Table 1 except for altering the value of γ from 0 to 10.

Figure A1 shows that too small or too large γ leads to relative weak performance due to the unstable convergence and penalizing too much for average case.

When γ is around 4, APGD will achieve the best performance so we adopted this value in the experiments (Table 1) .

Moreover, when γ → ∞, the regularizer term dominates the optimization objective and it becomes the average case.

Figure A2 presents "overall robustness" comparison of our min-max generalized AT scheme and vanilla AT with single type of attacks ( ∞ and 2 ) on MNIST (LeNet).

Similarly, our min-max training scheme leads to a higher "overall robustness" measured by S .

In practice, due to the lacking knowledge of the strengths/types of the attacks used by adversaries, it is meaningful to enhance "overall robustness" of models under the worst perturbation (Acc max adv ).

Specifically, our min-max generalized AT leads to 6.27% and 17.63% improvement on S compared to single-type AT with ∞ and 2 attacks.

Furthermore, weighting factor w of the probability simplex helps understand the behavior of AT under mixed types of attacks.

Our AMPGD algorithm will adjust w automatically according to the min-max principle -defending the strongest attack.

In Figure A2a , as 2 increases, 2 -attack becomes stronger so its corresponding w increases as well.

When 2 ≥ 2.5, 2 -attack dominates the adversarial training process.

That is to say, our AMPGD algorithm will put more weights on stronger attacks even if the strengths of attacks are unknown, which is a meritorious feature in practice.

Shafahi et al. (2018) also propose a variant of adversarial training to defend universal perturbations over multiple images.

To produce universal perturbations, they propose uSGD to conduct gradient descent on the averaged loss of one-batch images.

In consequence, their approach can be regarded as a variant of our generalized AT in average case.

The difference is that they do AT across multiple adversarial images under universal perturbation rather than mixed p -norm perturbations.

We added UAT [1] as one of our defense baselines in Table A7 .

The universal perturbation is generated by uSGD ( ∞ norm, = 0.3) with a batch size of 128 following Shafahi et al. (2018) .

We find that a) our proposed approach outperforms UAT under per-image p attacks.

Taking A7a as an example, our avg and min max generalized AT (with DPAR) result in average 17.85% and 17.97% improvement in adversarial test accuracy (ATA), b) our approach has just 3.72% degradation in ATA when encountering universal attacks, and c) both methods yield very similar normal test accuracy.

It is not surprising that our average and min-max training schemes can achieve better overall robustness while maintaining competitive performance on defending universal perturbation.

This is because the defensed model is trained under more general ( p norm) and diversity promoted perturbations.

As a result, proposed generalized AT is expected to obtain better overall robustness and higher transferability as shown in Table 4 and A7.

As reported in Sec. 4.2, our min-max generalized AT does not always result in the best performance on the success rate of defending the worst/strongest perturbation (Acc max adv ) for given ( ∞ , 2 ) pair, especially when the strengths of two attacks diverge greatly (e.g., for ∞ and 2 attacks are 0.2 and 0.5).

In what follows, we provide explanation and analysis about this finding inspired by recent work Araujo et al. (2019) .

and inside 2 ball (right, red area).

In particular, the red (blue) area in (a) (or (b)) represents the percentage of adversarial examples crafted by ∞( 2) attack that also belong to 2 ( ∞) ball.

We generate adversarial examples on 10,000 test images for each attack.

(c): Average p norm of adversarial examples as a function of perturbation magnitude 2 .

The top (bottom) side represents the 2-norm ( ∞) of the adversarial examples generated by ∞ ( 2) attack as 2 for generalized AT increases.

Note that the same as the AT procedure is used while attacking trained robust models.

Figure A3 shows the real overlap of ∞ and 2 norm balls in adversarial attacks for MLP model on MNIST.

Ideally, if 2 satisfies ∞ < 2 < ∞ × √ d, ∞ and 2 balls will not cover each other completely Araujo et al. (2019) .

In other words, AT with ∞ and 2 attacks cannot interchange with each other.

However, the real range of 2 for keeping 2 and ∞ balls intersected is not

, because crafted adversarial examples are not uniformly distributed in p -norm balls.

In Figure A3b , 99.98% adversarial examples devising using 2 attack are also inside ∞ ball, even if 0.2 < 2 = 0.5 < 5.6.

In consequence, AT with ∞ attack is enough to handle 2 -attack in overwhelming majority cases, which results in better performance than min-max optimization (Table A8a ).

Figure A3c presents the average p distance of adversarial examples with 2 increasing.

The average 2 -norm (green line) of adversarial examples generated by ∞ attack remains around 2.0 with a slight rising trend.

This is consistent to our setting -fixing 2 as 0.2.

It also indicates model robustness may effect the behavior of attacks -as 2 increases, robustly trained MLP model becomes more robust against 2 examples, so the ∞ attacker implicitly increases 2 norm to attack the model more effectively.

On the other hand, the average ∞ -norm increases substantially as 2 increases from 0.5 to 2.5.

When 2 arriving at 0.85, the average ∞ norm gets close to 0.2, so around half adversarial examples generated by 2 -attack are also inside ∞ balls, which is consistent with Table A3b .

Figure A4 shows the learning curves of model A under different AT schemes, where two setting are plotted: (a) ( ∞ , 2 ) = (0.2, 0.5); (b) ( ∞ , 2 ) = (0.2, 2.0).

Apart from better worst-case robustness shown in Table A8 , our min-max generalized AT leads to a faster convergence compared to average-based AT, especially when the strengths of two attacks diverge greatly.

For instance, when 2 = 0.5 (Figure A4a Tracking domain weight w of the probability simplex from our algorithms is an exclusive feature of solving problem 1.

In Sec. 4, we show the strength of w in understanding the procedure of optimization and interpreting the adversarial robustness.

Here we would like to show the usage of w in measuring "image robustness" on devising universal perturbation to multiple input samples.

Table A9 and A10 show the image groups on MNIST with weight w in APGD and two metrics (distortion of 2-C&W, minimum for ∞ -PGD) of measuring the difficulty of attacking single images.

The binary search is utilized to searching for the minimum perturbation.

Although adversaries need to consider a trade-off between multiple images while devising universal perturbation, we find that weighting factor w in APGD is highly correlated under different p norms.

Furthermore, w is also highly related to minimum distortion required for attacking a single image Table A8 : Adversarial training of MNIST models with single attacks ( ∞ and 2) and multiple attacks (avg. and min max).

During the training process, the perturbation magnitude ∞ is fixed as 0.2, and 2 are changed from 0.5 to 3.0 with a step size of 0.5.

For min-max scheme, the adversarial examples are crafted using 20-step ∞-APGD with α = 1 6

, β = 1 50

and γ = 4.

The ratio of adversarial and benign examples in adversarial training is set as 1.0.

For diversity-promoting attack regularizer (DPAR) in generalized AT, the hyperparameter λ = 0.1.

(a) ( ∞ , 2 ) = (0.2, 0.5) successfully.

It means the inherent "image robustness" exists and effects the behavior of generating universal perturbation.

Larger weight w usually indicates an image with higher robustness (e.g., fifth 'zero' in the first row of Table A9 ), which usually corresponds to the MNIST letter with clear appearance (e.g., bold letter).

@highlight

A unified min-max optimization framework for adversarial attack and defense