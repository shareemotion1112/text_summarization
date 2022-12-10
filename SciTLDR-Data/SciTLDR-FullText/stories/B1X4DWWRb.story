Predictive models that generalize well under distributional shift are often desirable and sometimes crucial to machine learning applications.

One example is the estimation of treatment effects from observational data, where a subtask is to predict the effect of a treatment on subjects that are systematically different from those who received the treatment in the data.

A related kind of distributional shift appears in unsupervised domain adaptation, where we are tasked with generalizing to a distribution of inputs that is different from the one in which we observe labels.

We pose both of these problems as prediction under a shift in design.

Popular methods for overcoming distributional shift are often heuristic or rely on assumptions that are rarely true in practice, such as having a well-specified model or knowing the policy that gave rise to the observed data.

Other methods are hindered by their need for a pre-specified metric for comparing observations, or by poor asymptotic properties.

In this work, we devise a bound on the generalization error under design shift, based on integral probability metrics and sample re-weighting.

We combine this idea with representation learning, generalizing and tightening existing results in this space.

Finally, we propose an algorithmic framework inspired by our bound and verify is effectiveness in causal effect estimation.

A long-term goal in artificial intelligence is for agents to learn how to act.

This endeavor relies on accurately predicting and optimizing for the outcomes of actions, and fundamentally involves estimating counterfactuals-what would have happened if the agent acted differently?

In many applications, such as the treatment of patients in hospitals, experimentation is infeasible or impractical, and we are forced to learn from biased, observational data.

Doing so requires adjusting for the distributional shift between groups of patients that received different treatments.

A related kind of distributional shift arises in unsupervised domain adaptation, the goal of which is to learn predictive models for a target domain, observing ground truth only in a source domain.

In this work, we pose both domain adaptation and treatment effect estimation as special cases of prediction across shifting designs, referring to changes in both action policy and feature domain.

We separate policy from domain as we wish to make causal statements about the policy, but not about the domain.

Learning from observational data to predict the counterfactual outcome under treatment B for a patient who received treatment A, one must adjust for the fact that treatment A was systematically given to patients of different characteristics from those who received treatment B. We call this predicting under a shift in policy.

Furthermore, if all of our observational data comes from hospital P , but we wish to predict counterfactuals for patients in hospital Q, with a population that differs from P , an additional source of distributional shift is at play.

We call this a shift in domain.

Together, we refer to the combination of domain and policy as the design.

The design for which we observe ground truth is called the source, and the design of interest the target.

The two most common approaches for addressing distributional shift are to learn shift-invariant representations of the data BID0 or to perform sample re-weighting or matching (Shimodaira, 2000; BID13 .

Representation learning approaches attempt to extract only information from the input that is invariant to a change in design and predictive of the variable of interest.

Such representations are typically learned by fitting deep neural networks in which activations of deeper layers are regularized to be distributionally similar across designs BID0 BID15 .

Although representation learning can be shown to reduce the error associated to distributional shift BID15 in some cases, standard approaches are biased, even in the limit of infinite data, as they penalize the use also of predictive information.

In contrast, re-weighting methods correct for distributional shift by assigning higher weight to samples from the source design that are representative of the target design, often using importance sampling.

This idea has been well studied in, for example, the causal inference BID20 , domain adaptation (Shimodaira, 2000) and reinforcement learning BID19 literature.

For example, in causal effect estimation, importance sampling is equivalent to re-weighting units by the inverse probability of observed treatments (treatment propensity).

Re-weighting with knowledge of importance sampling weights often leads to asymptotically unbiased estimators of the target outcome, but may suffer from high variance in finite samples (Swaminathan & Joachims, 2015) .A significant hurdle in applying re-weighting methods is that optimal weights are rarely known in practice.

There are a variety of methods to learn these weights.

Weights can be estimated as the inverse of estimated feature or treatment densities BID20 BID7 but this plug-in approach can lead to highly unstable estimates.

More stable methods learn weights by minimizing distributional distance metrics BID8 BID13 BID4 Zubizarreta, 2015) .

Closely related, matching (Stuart, 2010) produces weights by finding units in the source design that are close in some metric to units in the target design.

Specifying a distributional or unit-wise metric is challenging, especially if the input space is high-dimensional where no metric incorporating all features can ever be made small.

This has inspired heuristics such as first performing variable selection and then finding a matching in the selected covariates.

Our key algorithmic contribution is to show how to combine the intuition behind shift-invariant representation learning and re-weighting methods by jointly learning a representation Φ of the input space and a weighting function w(Φ) to minimize a) the re-weighted empirical risk and b) a re-weighted measure of distributional shift between designs.

This is useful also for the identity representation Φ(x) = x, as it allows for principled control of the variance of estimators through regularization of the re-weighting function w(x), mitigating the issues of exact importance sampling methods.

Further, this allows us to evaluate w on hold-out samples to select hyperparameters or do early stopping.

Finally, letting w depend on Φ alleviates the problem of choosing a metric by which to optimize sample weights, as Φ is trained to extract information predictive of the outcome.

We capture these ideas in an upper bound on the generalization error under a shift in design and specialize it to the case of treatment effect estimation.

We bring together two techniques used to overcome distributional shift between designs-re-weighting and representation learning, with complementary robustness properties, generalizing existing methods based on either technique.

We give finite-sample generalization bounds for prediction under design shift, without assuming access to importance sampling weights or to a well-specified model, and develop an algorithmic framework to minimize these bounds.

We propose a neural network architecture that jointly learns a representation of the input and a weighting function to improve balance across changing settings.

Finally, we apply our proposed algorithm to the task of predicting causal effects from observational data, achieving state-of-the art results on a widely used benchmark.

The goal of this work is to accurately predict outcomes of interventions T ∈ T in contexts X ∈ X drawn from a target design p π (X, T ).

The outcome of intervening with t ∈ T is the potential outcome Y (t) ∈ Y (Imbens & Rubin, 2015, Ch.

1-2), which has a stationary distribution p t (Y | X) given context X. Assuming a stationary outcome is akin to the covariate shift assumption (Shimodaira, 2000) , often used in domain adaptation.1 For example, in the classical binary setting, Y (1) represents the outcome under treatment and Y (0) the outcome under control.

The target design consists of two components: the target policy p π (T | X), which describes how one intends to map observations of contexts (such as patient prognostics) to interventions (such as pharmacological treatments) and the target domain p π (X), which describes the population of contexts to which the policy will be applied.

The target design is known to us only through m unlabeled sam-ples (x 1 , t 1 ), . . .

, (x m , t m ) from p π (X, T ).

Outcomes are only available to us in labeled samples from a source domain: (x 1 , t 1 , y 1 ), . . .

, (x n , t n , y n ), where (x i , t i ) are draws from a source design p µ (X, T ) and y i = y i (t i ) is a draw from p T (Y | X), corresponding only to the factual outcome Y (T ) of the treatment administered.

Like the target design, the source design consists of a domain of contexts for which we have data and a policy, which describes the (unknown) historical administration of treatment in the data.

Only the factual outcomes of the treatments administered are observed, while the counterfactual outcomes y i (t) for t = t i are, naturally, unobserved.

Our focus is the observational or off-policy setting, in which interventions in the source design are performed non-randomly as a function of X, p µ (T | X) = p µ (T ).

This encapsulates both the covariate shift often observed between treated and control populations in observational studies and the covariate shift between the domain of the study and the domain of an eventual wider intervention.

Examples of this problem are plentiful: in addition to the example given in the introduction, consider predicting the return of an advertising policy based on the historical results of a different policy, applied to a different population of customers.

We stress that we are interested in the causal effect of an intervention T on Y , conditioned on X. As such, we cannot think of X and T as a single variable.

Without additional assumptions, it is impossible to deduce the effect of an intervention based on observational data alone BID18 , as it amounts disentangling correlation and causation.

Crucially, for any unit i, we can observe the potential outcome y i (t) of at most one intervention t. In our analysis, we make the following standard assumptions.

Assumption 1 (Consistency, ignorability and overlap).

For any unit i, assigned to intervention t i , we observe Y i = Y (t i ).

Further, {Y (t)} t∈T and the data-generating process p µ (X, T, Y ) satisfy strong ignorability: {Y (t)} t∈T ⊥ ⊥ T | X and overlap: DISPLAYFORM0 Assumption 1 is a sufficient condition for causal identifiability BID20 .

Ignorability is also known as the no hidden confounders assumption, indicating that all variables that cause both T and Y are assumed to be measured.

Under ignorability therefore, any domain shift in p(X) cannot be due to variables that causally influence T and Y , other than through X. Under Assumption 1, potential outcomes equal conditional expectations: DISPLAYFORM1 , and we may predict Y (t) by regression.

We further assume common domain support, ∀x ∈ X : p π (X = x) > 0 ⇒ p µ (X = x) > 0.

Finally, we adopt the notation p(x) := p(X = x).

We attempt to learn predictors f : DISPLAYFORM0 Recall that under Assumption 1, this conditional expectation is equal to the (possibly counterfactual) potential outcome Y (t), conditioned on X. Our goal is to ensure that hypotheses f are accurate under a design p π that deviates from the data-generating process, p µ .

This is unlike standard supervised learning for which p π = p µ .

We measure the (in)ability of f to predict outcomes under π, using the expected risk, DISPLAYFORM1 is an appropriate loss function, such as the squared loss, L(y, y ) := (y − y ) 2 or the log-loss, depending on application.

As outcomes under the target design p π are not observed, even through a finite sample, we cannot directly estimate (1) using the empirical risk under p π .

A common way to resolve this is to use importance sampling (Shimodaira, 2000)-the observation that if p µ and p π have common support, with w DISPLAYFORM2 Hence, with access to w * , an unbiased estimator of R π (f ) may be obtained by re-weighting the (factual) empirical risk under µ, DISPLAYFORM3 Unfortunately, importance sampling weights can be very large when p π is large and p µ small, resulting in large variance inR & Joachims, 2015) .

More importantly, p µ (x, t) is rarely known in practice, and neither is w * .

In principle, however, any re-weighting function w with the following property yields a valid risk under the re-weighted distribution p w µ .

DISPLAYFORM4 DISPLAYFORM5 We denote the re-weighted density p w µ (x, t) := w(x, t)p µ (x, t).A natural candidate in place of w * is an estimateŵ * based on estimating densities p π (x, t) and p µ (x, t).

In this work, we adopt a different strategy, learning parameteric re-weighting functions w from observational data, that minimize an upper bound on the risk under p π .

An important special case of our setting is when treatments are binary, T ∈ {0, 1}, often interpreted as treating (T = 1) or not treating (T = 0) a unit, and the domain is fixed across designs, p µ (X) = p π (X).

This is the classical setting for estimating treatment effects-the effect of choosing one intervention over another BID17 .

2 The effect of an intervention T = 1 in context X, is measured by the conditional average treatment effect (CATE), DISPLAYFORM0 Predicting τ for unobserved units typically involves prediction of both potential outcomes 3 .

In a clinical setting, knowledge of τ is necessary to assess which medication should be administered to a certain individual.

Historically, the (population) average treatment effect, ATE = E x∼p [τ (x)], has received comparatively much more attention BID20 , but is inadequate for personalized decision making.

Using predictors f (x, t) of potential outcomes Y (t) in contexts X = x, we can estimate the CATE byτ (x) = f (x, 1) − f (x, 0) and measure the quality using the mean squared error (MSE), DISPLAYFORM1 (4) In Section 4, we argue that estimating CATE from observational data requires overcoming distributional shift with respect to the treat-all and treat-none policies, in predicting each respective potential outcome, and show how this can be used to derive generalization bounds for CATE.

A large body of work has shown that under assumptions of ignorability and having a wellspecified model, various regression methods for counterfactual estimation are asymptotically consistent BID4 BID1 BID3 .

However, consistency results like these provide little insight into the case of model misspecification.

Under model misspecification, regression methods may suffer from additional bias when generalizing across designs due to distributional shift.

A common way to alleviate this is importance sampling, see Section 2.

This idea is used in propensity-score methods BID2 , that use treatment assignment probabilities (propensities) to re-weight samples for causal effect estimation, and more generally in re-weighted regression, see e.g. (Swaminathan & Joachims, 2015) .

A major drawback of these methods is the assumption that the design density is known.

To address this, others BID8 BID13 , have proposed learning sample weights w to minimize the distributional distance between samples under p π and p w µ , but rely on specifying the data representation a priori, without regard for which aspects of the data actually matter for outcome prediction and policy estimation.

On the other hand, BID12 Shalit et al. (2017) proposed learning representations for counterfactual inference, inspired by work in unsupervised domain adaptation BID16 .

The drawback of this line of work is that the generalization bounds of Shalit et al. FORMULA4 and BID15 are loose-even if infinite samples are available, they are not guaranteed to converge to the lowest possible error.

Moreover, these approaches do not make use of important information that can be estimated from data: the treatment/domain assignment probabilities.

We give a bound on the risk in predicting outcomes under a target design p π (T, X) based on unlabeled samples from p π and labeled samples from a source design p µ (T, X).

Our result combines representation learning, distribution matching and re-weighting, resulting in a tighter bound than the closest related work, Shalit et al. (2017) .

The predictors we consider are compositions f (x, t) = h(Φ(x), t) where Φ is a representation of x and h an hypothesis.

We first give an upper bound on the risk in the general design shift setting, then show how this result can be used to bound the error in prediction of treatment effects.

In Section 5 we give a result about the asymptotic properties of the minimizers of this upper bound.

Risk under distributional shift Our bounds on the risk under a target design capture the intuition that if either a) the target design π and source design µ are close, or b) the true outcome is a simple function of x and t, the gap between the target risk and the re-weighted source risk is small.

These notions can be formalized using integral probability metrics (IPM) (Sriperumbudur et al., 2009 ) that measure distance between distributions w.r.t.

a normed vector space of functions H. Definition 2.

The integral probability metric (IPM) distance, associated with a normed vector space of functions H, between distributions p and q is, DISPLAYFORM0 Important examples of IPMs include the Wasserstein distance, for which H is the family of functions with Lipschitz constant at most 1, and the Maximum Mean Discrepancy for which H are functions in the norm-1 ball in a reproducing kernel Hilbert space.

Using definitions 1-2, and the definition of re-weighted risk, see FORMULA4 , we can state the following result (see Appendix A.2 for a proof).

Lemma 1.

For hypotheses f with loss f such that f / f H ∈ H, and p µ , p π with common support, there exists a valid re-weighting w of p µ , see Definition 1, such that, DISPLAYFORM1 The first inequality is tight for importance sampling weights, w( DISPLAYFORM2 The bound of Lemma 1 is tighter if p µ and p π are close (the IPM is smaller), and if the loss lives in a small family of functions H (the supremum is taken over a smaller set).

Lemma 1 also implies that there exist weighting functions w(x, t) that achieve a tighter bound than the uniform weighting w(x, t) = 1, implicitly used by Shalit et al. (2017) .

While importance sampling weights result in a tight bound in expectation, neither the design densities nor their ratio are known in general.

Moreover, exact importance weights often result in large variance in finite samples BID6 .

Here, we will search for a weighting function w, that minimizes a finite-sample version of (5), trading off bias and variance.

We examine the empirical value of this idea alone in Section 6.1.

We now introduce the notion of representation learning to combat distributional shift.

Representation learning The idea of learning representations that reduce distributional shift in the induced space, and thus the source-target error gap, has been applied in domain adaptation BID0 , algorithmic fairness (Zemel et al., 2013) and counterfactual prediction (Shalit et al., 2017) .

The hope of these approaches is to learn predictors that predominantly exploit information that is common to both source and target distributions.

For example, a face detector should be able to recognize the structure of human features even under highly variable environment conditions, by ignoring background, lighting etc.

We argue that re-weighting (e.g. importance sampling) should also only be done with respect to features that are predictive of the outcome.

Hence, in Section 5, we propose using re-weightings that are functions of learned representations.

We follow the setup of Shalit et al. (2017) , and consider learning twice-differentiable, invertible representations Φ : X → Z, where Z is the representation space, and Ψ : Z → X is the inverse representation, such that Ψ(Φ(x)) = x for all x. Let E denote space of such representation functions.

For a design π, we let p π,Φ (z, t) be the distribution induced by Φ over Z × T , with p w π,Φ (z, t) := p π,Φ (z, t)w(Ψ(z), t) its re-weighted form andp w π,Φ its re-weighted empirical form, following our previous notation.

Finally, we let G ⊆ {h : Z × T → Y} denote a set of hypotheses h(Φ, t) operating on the representation Φ and let F the space of all compositions, F = {f = h(Φ(x), t) : h ∈ G, Φ ∈ E}. We can now relate the expected target risk R π (f ) to the re-weighted empirical source riskR w µ (f ).

Theorem 1.

Given is a labeled sample (x 1 , t 1 , y 1 ), ..., (x n , t n , y n ) from p µ , and an unlabeled sample (x 1 , t 1 ), ..., (x m , t m ) from p π , with corresponding empirical measuresp µ andp π .

Suppose that Φ is a twice-differentiable, invertible representation, that h(Φ, t) is an hypothesis, and DISPLAYFORM3 where L is the squared loss, L(y, y ) = (y − y ) 2 , and assume that there exists a constant B Φ > 0 such that h,Φ /B Φ ∈ H ⊆ {h : Z × T → Y}, where H is a reproducing kernel Hilbert space of a kernel, k such that k((z, t), (z, t)) < ∞. Finally, let w be a valid re-weighting of p µ,Φ .

Then with probability at least 1 − 2δ, DISPLAYFORM4 where C DISPLAYFORM5 .

A similar bound exists where H is the family of functions Lipschitz constant at most 1, and IPM H the Wasserstein distance, but with worse sample complexity.

See Appendix A.2 for a proof of Theorem 1 that involves applying finite-sample generalization bounds to Lemma 1, as well as moving to the space induced by the representation Φ.Theorem 1 has several implications: non-identity feature representations, non-uniform sample weights, and variance control of these weights can all contribute to a lower bound.

Using uniform weights w(x, t) = 1 in (6), results in a bound similar to that of Shalit et al. FORMULA4 and BID15 .

When π = µ, minimizing uniform-weight bounds results in biased hypotheses, even in the asymptotical limit, as the IPM term does not vanish when the sample size increases.

This is an undesirable property, as even k-nearest-neighbor classifiers are consistent in the limit of infinite samples.

We consider minimizing (6) with respect to w, improving the tightness of the bound.

Theorem 1 indicates that even though importance sampling weights w * yield estimators with small bias, they can suffer from high variance, as captured by the factor V µ (w, f ).

The factor B Φ is not known in general as it depends on the true outcome, and is determined by f H as well as the determinant of the Jacobian of Ψ, see Appendix A.2.

Qualitatively, B Φ measures the joint complexity of Φ and f and is sensitive to the scale of Φ-as the scale of Φ vanishes, B Φ blows up.

To prevent this in practice, we normalize Φ. As B Φ is unknown, Shalit et al. (2017) substituted a hyperparameter α for B Φ , but discussed the difficulties of selecting its value without access to counterfactual labels.

In our experiments, we explore a heuristic for adaptively choosing α, based on measures of complexity of the observed held-out loss as a function of the input.

Finally, the term C Theorem 1 is immediately applicable to the case of unsupervised domain adaptation in which there is only a single potential outcome of interest, T = {0}. In this case, DISPLAYFORM6 Conditional average treatment effects A simple argument shows that the error in predicting the conditional average treatment effect, MSE(τ ) can be bounded by the sum of risks under the constant treat-all and treat-none policies.

As in Section 2.2, we consider the case of a fixed domain p π (X) = p µ (X) and binary treatment T = {0, 1}. Let R πt (f ) denote the risk under the constant policy π t such that ∀x ∈ X : p πt (T = t | X = x) = 1.

Proposition 1.

We have with MSE(τ ) as in (4) and R πt (f ) the risk under the constant policy π t , DISPLAYFORM7 The proof involves the relaxed triangle inequality and the law of total probability.

By Proposition 1, we can apply Theorem 1 to R π1 and R π0 separately, to obtain a bound on MSE(τ ).

For brevity, we refrain from stating the full result, but emphasize that it follows from Theorem 1.

In Section 6.2, we evaluate our framework in treatment effect estimation, minimizing this bound.

Motivated by the theoretical insights of Section 4, we propose to jointly learn a representation Φ(x), a re-weighting w(x, t) and an hypothesis h(Φ, t) by minimizing a bound on the risk under the target design, see (6).

This approach improves on previous work in two ways: it alleviates the bias of Shalit et al. (2017) when sample sizes are large, see Section 4, and it increases the flexibility of the balancing method of BID8 by learning the representation to balance.

For notational brevity, we let w i = w(x i , t i ).

Recall thatp w π,Φ is the re-weighted empirical distribution of representations Φ under p π .

The training objective of our algorithm is the RHS of (6), with hyperparameters β = (α, λ h , λ w ) substituted for model (and representation) complexity terms, DISPLAYFORM0 where R(h) is a regularizer of h, such as 2 -regularization.

We can show the following result.

Theorem 2.

Suppose H is a reproducing kernel Hilbert space given by a bounded kernel.

Suppose weak overlap holds in that DISPLAYFORM1 Consequently, under the assumptions of Thm.

1, for sufficiently large α and λ w , DISPLAYFORM2 In words, the minimizers of (8) converge to the representation and hypothesis that minimize the counterfactual risk, in the limit of infinite samples.

Implementation Minimization of L π (h, Φ, w; β) over h, Φ and w is, while motivated by Theorem 2, a difficult optimization problem to solve in practice.

For example, adjusting w to minimize the empirical risk term may result in overemphasizing "easy" training examples, resulting in a poor local minimum.

Perhaps more importantly, ensuring invertibility of Φ is challenging for many representation learning frameworks, such as deep neural networks.

In our implementation, we deviate from theory on these points, by fitting the re-weighting w based only on imbalance and variance terms, and don't explicitly enforce invertibility.

As a heuristic, we split the objective, see (8), in two and use only the IPM term and regularizer to learn w. In short, we adopt the following alternating procedure.

DISPLAYFORM3 The re-weighting function w(x, t) could be represented by one free parameter per training point, as it is only used to learn the model, not for prediction.

However, we propose to let w be a parametric function of Φ(x).

Doing so ensures that information predictive of the outcome is used for balancing, and lets us compute weights on a hold-out set, to perform early stopping or select hyperparameters.

This is not possible with existing re-weighting methods such as BID8 BID13 .

An example architecture for the treatment effect estimation setting is presented in FIG2 .

By Proposition 1, estimating treatment effects involves predicting under the two constant policiestreat-everyone and treat-no-one.

In Section 6, we evaluate our method in this task.

As noted by Shalit et al. (2017) , choosing hyperparameters for counterfactual prediction is fundamentally difficult, as we cannot observe ground truth for counterfactuals.

In this work, we explore setting the balance parameter α adaptively.

α is used in (8) in place of B Φ , a factor measuring the complexity of the loss and representation function as functions of the input, a quantity that changes during training.

As a heuristic, we use an approximation of the Lipschitz constant of f , with f = h(Φ(x), t), based on observed examples: DISPLAYFORM4 We use a moving average over batches to improve stability.

We create a synthetic domain adaptation experiment to highlight the benefit of using a learned reweighting function to minimize weighted risk over using importance sampling weights w * (x) = DISPLAYFORM0 for small sample sizes.

We observe n labeled source samples, distributed according to p µ (x) = N (x; m µ , I d ) and predict for n unlabeled target samples drawn according to DISPLAYFORM1 and c ∼ N (0, 1) and let y = σ(β x + c) where σ(z) = 1/(1 + e −z ).

Importance sampling weights, w * (x) = p π (x)/p µ (x), are known.

In experiments, we vary n from 10 to 600.

We fit (misspecified) linear models 4 f (x) = β x + γ to the logistic outcome, and compare minimizing a weighted source risk by a) parameterizing sample weights as a small feed-forward neural network to minimize (8) (ours) b) using importance sampling weights (baseline), both using gradient descent.

For our method, we add a small variance penalty, λ w = 10 −3 , to the learned weights, use MMD with an RBF-kernel of σ = 1.0 as IPM, and let α = 10.

We compare to exact importance sampling weights (IS) as well as clipped IS weights (ISC), w M (x) = min(w(x), M ) for M ∈ {5, 10}, a common way of reducing variance of re-weighting methods (Swaminathan & Joachims, 2015) .In FIG4 , we see that our proposed method behaves well at small sample sizes compared to importance sampling methods.

The poor performance of exact IS weights is expected at smaller samples, as single samples are given very large weight, resulting in hypotheses that are highly sensitive to the training set.

While clipped weights alleviates this issue, they do not preserve relevance ordering of high-weight samples, as many are given the truncation value M , in contrast to the reweighting learned by our method.

True domain densities are known only to IS methods.

We evaluate our framework in the CATE estimation setting, see Section 2.2.

Our task is to predict the expected difference between potential outcomes conditioned on pre-treatment variables, for a held-out sample of the population.

We compare our results to ordinary least squares (OLS) (with one regressor per outcome), OLS-IPW (re-weighted OLS according to a logistic regression estimate of propensities), Random Forests, Causal Forests (Wager & Athey, 2017) , BID5 , and CFRW (Shalit et al., 2017) (with Wasserstein penalty).

Finally, we use as baseline (IPM-WNN): first weights are found by IPM minimization in the input space BID8 BID13 , then used in a re-weighted neural net regression, with the same architecture as our method.

Our implementation, dubbed RCFR for Re-weighted CounterFactual Regression, parameterizes representations Φ(x), weighting functions w(Φ, t) and hypotheses h(Φ, t) using neural networks, trained by minimizing (8).

We use the RBF-kernel maximum mean discrepancy as the IPM BID9 .

For a description of the architecture, training procedure and hyperparameters, see Appendix B. We compare results using uniform w = 1 and learned weights, setting the balance parameter α either fixed, by an oracle (test-set error), or adaptively using the heuristic described in Section 5.

To pick other hyperparameters, we split training sets into one part used for function fitting and one used for early stopping and hyperparameter selection.

Hyperparameters for regularization are chosen based on the empirical loss on a held-out source (factual) sample.

CATE Error, RMSE(τ ) DISPLAYFORM0 (b) For small imbalance penalties α, re-weighting (low λw) has no effect.

For moderate α, less uniform re-weighting (smaller λw) improves the error, c) for large α, weighting helps, but overall error increases.

Best viewed in color.

FORMULA4 , and a synthesized continuous outcome that can be used to compute the ground-truth CATE error.

Average results over 100 different realizations/settings of the outcome are presented in TAB0 .

We see that our proposed method achieves state-of-the-art results, and that adaptively choosing α does not hurt performance much.

Furthermore, we see a substantial improvement from using nonuniform sample weights.

In FIG4 we take a closer look at the behavior of our model as we vary its hyperparameters on the IHDP dataset.

Between the two plots we can draw the following conclusions: a) For moderate to large α ∈ [10, 100], we observe a marginal gain from using the IPM penalty.

This is consistent with the observations of Shalit et al. (2017) .

b) For large α ∈ [10, 1000], we see a large gain from using a non-uniform re-weighting (small λ w ).

c) While large α makes the factual error more representative of the counterfactual error, using it without re-weighting results in higher absolute error.

We believe that the moderate sample size of this dataset is one of the reasons for the usefulness of our method.

See Appendix C.2 for a complementary view of these results.

We have proposed a theory and an algorithmic framework for learning to predict outcomes of interventions under shifts in design-changes in both intervention policy and feature domain.

The framework combines representation learning and sample re-weighting to balance source and target designs, emphasizing information from the source sample relevant for the target.

Existing reweighting methods either use pre-defined weights or learn weights based on a measure of distributional distance in the input space.

These approaches are highly sensitive to the choice of metric used to measure balance, as the input may be high-dimensional and contain information that is not predictive of the outcome.

In contrast, by learning weights to achieve balance in representation space, we base our re-weighting only on information that is predictive of the outcome.

In this work, we apply this framework to causal effect estimation, but emphasize that joint representation learning and re-weighting is a general idea that could be applied in many applications with design shift.

Our work suggests that distributional shift should be measured and adjusted for in a representation space relevant to the task at hand.

Joint learning of this space and the associated re-weighting is attractive, but several challenges remain, including optimization of the full objective and relaxing the invertibility constraint on representations.

For example, variable selection methods are not covered by our current theory, as they induce a non-ivertible representation, but a similar intuition holds there-only predictive attributes should be used when measuring imbalance.

We believe that addressing these limitations is a fruitful path forward for future work.

We denote the re-weighted density p w µ (x, t) := w(x, t)p µ (x, t).Expected & empirical risk We let the (expected) risk of f measured by h under p µ be denoted DISPLAYFORM0 where l h is an appropriate loss function, and the empirical risk over a sample DISPLAYFORM1 We use the superscript w to denote the re-weighted risks DISPLAYFORM2 Definition A1 (Importance sampling).

For two distributions p, q on Z, of common support, ∀z ∈ Z : p(z) > 0 ⇐⇒ q(z) > 0, we call DISPLAYFORM3 the importance sampling weights of p and q. Definition 2 (Restated).

The integral probability metric (IPM) distance, associated with the function family H, between distributions p and q is defined by DISPLAYFORM4 We begin by bounding the expected risk under a distribution p π in terms of the expected risk under p µ and a measure of the discrepancy between p π and p µ .

Using definition 2 we can show the following result.

Lemma 1 (Restated).

For hypotheses f with loss f such that f / f H ∈ H, and p µ , p π with common support, there exists a valid re-weighting w of p µ , see Definition 1, such that, DISPLAYFORM5 The first inequality is tight for importance sampling weights, w(x, t) = p π (x, t)/p µ (x, t).

The second inequality is not tight for general f , even if f ∈ H, unless p π = p µ .Proof.

The results follows immediately from the definition of IPM.

DISPLAYFORM6 Further, for importance sampling weights w IS (x, t) = π(t;x) µ(t;x) , for any h ∈ H, DISPLAYFORM7 and the LHS is tight.

We could apply Lemma 1 to bound the loss under a distribution q based on the weighted loss under p. Unfortunately, bounding the expected risk in terms of another expectation is not enough to reason about generalization from an empirical sample.

To do that we use Corollary 2 of BID6 , restated as a Theorem below.

Theorem A1 (Generalization error of re-weighted loss BID6 ).

For a loss function h of any hypothesis h ∈ H ⊆ {h : X → R}, such that d = Pdim({ h : h ∈ H}) where Pdim is the pseudo-dimension, and a weighting function w(x) such that E p [w] = 1, with probability 1 − δ over a sample (x 1 , ..., x n ), with empirical distributionp, DISPLAYFORM8 we get the simpler form DISPLAYFORM9 We will also need the following result about estimating IPMs from finite samples from Sriperumbudur et al. (2009)

.Theorem A2 (Estimation of IPMs from empirical samples (Sriperumbudur et al., 2009) ).

Let M be a measurable space.

Suppose k is measurable kernel such that sup x∈M k(x, x) ≤ C ≤ ∞ and H the reproducing kernel Hilbert space induced by k, with ν := sup x∈M,f ∈H f (x) < ∞. Then, witĥ p,q the empirical distributions of p, q from m and n samples respectively, and with probability at least 1 − δ, DISPLAYFORM10 We consider learning twice-differentiable, invertible representations Φ : X → Z, where Z is the representation space, and Ψ :

Z → X is the inverse representation, such that Ψ(Φ(x)) = x for all x. Let E denote space of such representation functions.

For a design π, we let p π,Φ (z, t) be the distribution induced by Φ over Z × T , with p w π,Φ (z, t) := p π,Φ (z, t)w(Ψ(z), t) its re-weighted form andp w π,Φ its re-weighted empirical form, following our previous notation.

Note that we do not include t in the representation itself, although this could be done in principle.

Let G ⊆ {h : Z × T → Y} denote a set of hypotheses h(Φ, t) operating on the representation Φ and let F denote the space of all compositions, F = {f = h(Φ(x), t) : h ∈ G, Φ ∈ E}. We now restate and prove Theorem 1.

Given is a labeled sample D µ = {(x 1 , t 1 , y 1 ), ..., (x n , t n , y n )} from p µ , and an unlabeled sample D π = {(x 1 , t 1 ), ..., (x m , t m )} from p π , with corresponding empirical measuresp µ andp π .

Suppose that Φ is a twice-differentiable, invertible representation, that h(Φ, t) is an hypothesis, and DISPLAYFORM0 where L is the squared loss, L(y, y ) = (y − y ) 2 , and assume that there exists a constant B Φ > 0 such that h,Φ /B Φ ∈ H ⊆ {h : Z × T → Y}, where H is a reproducing kernel Hilbert space of a kernel, k such that k((z, t), (z, t)) < ∞. Finally, let w be a valid re-weighting of p µ,Φ .

Then with probability at least 1 − 2δ, DISPLAYFORM1 where C F n,δ measures the capacity of F and has only logarithmic dependence on n, D H m,n,δ measures the capacity of H, σ 2 Y is the expected variance in potential outcomes, and DISPLAYFORM2 A similar bound exists where H is the family of functions Lipschitz constant at most 1, but with worse sample complexity.

Proof.

We have by definition DISPLAYFORM3 f (x, t, y)p(y | t, x)(p π (x, t) − p Proof.

Let f * = Φ * • h * ∈ arg min f ∈F R π (f ) and let w * (x, t) = p π,Φ (Φ * (x), t)/p µ,Φ (Φ * (x), t).

Since min h,Φ,w L π (h, Φ, w; β) ≤ L π (h * , Φ * , w * ; β), it suffices to show that L π (h * , Φ * , w * ; β) = R π (f * ) + O(1/ √ n + 1/ √ m).

We will work term by term: DISPLAYFORM4 For term D , letting w * i = w * (x i , t i ), we have that by weak overlap DISPLAYFORM5 For term A , under ignorability, each term in the sum in the first term has expectation equal to R π (f * ) and so, so by weak overlap and bounded second moments of loss, we have A = R π (f * ) + O p (1/ √ n).

For term B , since h * is fixed we have deterministically that DISPLAYFORM6 Finally, we address term C , which when expanded can be written as DISPLAYFORM7 w * i h(Φ * (x i ), t i )).10 −4 10 −2 10 −0 10 2 ∞ Re-weighting regularization λw (uniformity) Figure 3 : Error in CATE estimation on IHDP as a function of re-weighting regularization strength λ w (left) and source prediction error (right).

We see in the left-hand plot that a) for small imbalance penalties α, re-weighting (low λ w ) has no effect, b) for moderate α, less uniform re-weighting (smaller λ w ) improves the error, c) for large α, weighting helps, but overall error increases.

In the right-hand plot, we compare the ratio of CATE error to source error.

Color represents α (see left) and size λ w .

We see that for large α, the source error is more representative of CATE error, but does not improve in absolute value without weighting.

Here, α was fixed.

Best viewed in color.

In Figure 3 , we see two different views of the IHDP results.

@highlight

A theory and algorithmic framework for prediction under distributional shift, including causal effect estimation and domain adaptation