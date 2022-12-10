Current practice in machine learning is to employ deep nets in an overparametrized limit, with the nominal number of parameters typically exceeding the number of measurements.

This resembles the situation in compressed sensing, or in sparse regression with $l_1$ penalty terms, and provides a theoretical avenue for understanding phenomena that arise in the context of deep nets.

One such phenonemon is the success of deep nets in providing good generalization in an interpolating regime with zero training error.

Traditional statistical practice calls for regularization or smoothing to prevent "overfitting" (poor generalization performance).

However, recent work shows that there exist data interpolation procedures which are statistically consistent and provide good generalization performance\cite{belkin2018overfitting} ("perfect fitting").

In this context, it has been suggested that "classical" and "modern" regimes for machine learning are separated by a peak in the generalization error ("risk") curve, a phenomenon dubbed "double descent"\cite{belkin2019reconciling}. While such overfitting peaks do exist and arise from ill-conditioned design matrices, here we challenge the interpretation of the overfitting peak as demarcating the regime where good generalization occurs under overparametrization.



We propose a model of Misparamatrized Sparse Regression (MiSpaR) and analytically compute the GE curves for $l_2$ and $l_1$ penalties.

We show that the overfitting peak arising in the interpolation limit is dissociated from the regime of good generalization.

The analytical expressions are obtained in the so called "thermodynamic" limit.

We find an additional interesting phenomenon: increasing overparametrization in the fitting model increases sparsity, which should intuitively improve performance of $l_1$ penalized regression.

However, at the same time, the relative number of measurements decrease compared to the number of fitting parameters, and eventually overparametrization does lead to poor generalization.

Nevertheless, $l_1$ penalized regression can show good generalization performance under conditions of data interpolation even with a large amount of overparametrization.

These results provide a theoretical avenue into studying inverse problems in the interpolating regime using overparametrized fitting functions such as deep nets.

Modern machine learning has two salient characteristics: large numbers of measurements m, and non-linear parametric models with very many fitting parameters p, with both m and p in the range of 10 6

9 for many applications.

Fitting data with such large numbers of parameters stands in contrast to the inductive scientific process where models with small numbers of parameters are normative.

Nevertheless, these large-parameter models are successful in dealing with real life complexity, raising interesting theoretical questions about the generalization ability of models with large numbers of parameters, particularly in the overparametrized regime µ = p/m > 1.

Classical statistical procedures trade training (TE) and generalization error (GE) by controlling the model complexity.

Sending TE to zero (for noisy data) is expected to increase GE [10] .

However deep nets seem to over-parametrize and drive TE to zero (data interpolation) while maintaining good GE [18, 5] .

Over-parametrization has the benefit that global minima of the empirical loss function Note that for µ↵ > 1, the GE values for the l 1 case are close to zero, whereas the values for the l 2 penalized case can be much larger.

Note also that the overfitting peak is much larger for ↵ < 1 than for ↵ > 1, and that the region of good generalization starts at µ = 1/↵, which can be to the left or right of the overfitting peak depending on the value of the undersampling parameter ↵.

For the simulations with ! 0, in the l 2 case a pseudoinverse was used.

For the l 1 case a numerically small value = 10 5 was used, and it was checked that the results do not change on decreasing .

proliferate and become easier to find [12, 15] .

These observations have led to recent theoretical activity [4, 5, 11] .

Regression and classification algorithms have been shown that interpolate data but also generalize optimally [4] .

An interesting related phenomenon has been noted: the existence of a peak in GE with increasing fitting model complexity [2, 1, 8, 9] .

In [2] it was suggested that this peak separates a classical regime from a modern (interpolating) regime where over-parametrization improves performance.

While the presence of a peak in the GE curve is in stark contrast with the classical statistical folk wisdom where the GE curve is thought to be U-shaped, understanding the significance of such peaks is an open question, and motivates the current paper.

Parenthetically, similar over-fitting peaks were reported almost twenty years ago (cf.

statistical physics approach to learning) and attributed to increased fitting model entropy near the peak (see in particular Figs 4.3 and 5.2 in [7] ).

1.

We introduce a model, Misparametrized (or Misspecified) Sparse Regression (MiSpaR), which separates the number of measurements m, the number of model parameters n (which can be controlled for sparsity by a parameter ⇢), and the number of fitting degrees of freedom p. 2.

We obtain analytical expressions for the GE and TE curves for l 2 penalized regression in the "high-dimensional" or "thermodynamic" asymptotic regime m, p, n !

1 keeping the ratios µ = p/m and ↵ = m/n fixed.

We are also able to analytically compute GE for l 1 penalized regression, and exhibit explicit expressions for µ < 1 and µ >> 1 as ! 0.

3.

We show that for ! 0 and for > 0, the overfitting peak appears at the data interpolation point µ = 1 (p = m) for both l 2 and l 1 penalized interpolation (GE ⇠ |1 µ| 1 near µ = 1), but does not demarcate the point at which "good generalization" first occurs, which for small corresponds to the point p = n (µ↵ = 1) (Figure 1 ).

The region of good generalization can start before or after the overfitting peak.

The overfitting peak is suppressed for finite .

4.

For infinitely large overparametrization, generalization does not occur: GE(µ !

1) = 1 for both l 2 and l 1 penalized interpolation.

However, for small values of the sparsity parameter ⇢ and measurement noise variance 2 , there is a large range of values of µ where l 1 regularized interpolation generalizes well, but l 2 penalized interpolation does not (Fig. 1 ).

This range is given by 1 << log(µ) <<

2 , ⇢/↵ << 1.

In this regime the sparsity penalty is effective, and suppresses noise-driven mis-estimation of parameters for the l 1 penalty.

This shows how generalization properties of penalized interpolation depend strongly on the inductive bias, and are not properties of data interpolation per se.

This has important implications for the usage of deep nets for solving inverse problems.

2 for small µ µ c ) and GE 1 (µ ! 1) = 1.

6.

For = 0 and ↵ > ↵ c (⇢), GE 1 goes to zero linearly at µ↵ = 1 (GE 1 / (

.

In this case GE 1 goes to zero with a nontrivial

on the left, but rises quadratically on the right

Usually in linear regression the same (known) design matrix x ij is used both for data generation and for parameter inference.

In MiSpaR the generative model has a fixed number n of parameters j , which generate m measurements y i , but the number of parameters p in the inference model is allowed to vary freely, with p < n corresponding the under-parametrized and p > n the over-parametrized case.

For the under-parametrized case, a truncated version of the design matrix is used for inference, whereas for the over-parametrized case, the design matrix is augmented with extra rows.

In addition, we assume that the parameters in the generative model are sparse, and consider the effect of sparsity-inducing regularization in the interpolation limit.

Combining misparametrization with sparsity is important to our study for two reasons • Dissociating data interpolation (which happens when µ = 1, ! 0) from the regime where good generalization can occur (this is controlled by the undersampling ↵ as well as by the model sparsity ⇢).

• We are able to study the effect of different regularization procedures on data interpolation in an analytically tractable manner and obtain analytical expressions for the generalization error.

Generative Model ("Teacher") We assume that the (known/measured) design variables are i.i.d.

Gaussian distributed 2 from one realization of the generative model to another with variance 1/n.

This choice of variance is important to fix normalization.

Other choices have also been employed in the literature (notably x ij ⇠ N (0, 1/m)) -this is important to keep in mind when comparing with literature formulae where factors of ↵ may need to be inserted appropriately to obtain a match.

Undersampling: ↵ = m/n Sparsity:

Here ⇡( ) is the distribution of the non-zero model parameters.

We assume this distribution to be Gaussian as this permits closed form evaluation of integrals appearing in the l 1 case.

Note that we term µ = p/m as overpametrization (referring to the case where µ > 1) and we term ↵ = m/n as undersampling (referring to the case where ↵ < 1).

Inference Model ("Student") The design matrix used for inference is mis-parametrized or mis-specified: under-specified (or partially observed) when µ↵ < 1 ⌘ p < n; over-specified, with extra, effect-free rows in the design matrix when µ↵ > 1 ⌘ p > n

Parameter inference is carried out by minimizing a penalized mean squared error

Note that for p > n, the model parameters are augmented by p n zero entries.

We consider l 2 and l 1 penalties (correspondingly V ( ) =

Note that the expectation E is taken simultaneously over the parameter values, the design matrix and measurement noise.

We obtain exact analytical expressions for the risk (generalization error) in the (thermodynamic) limit where n, p, m all tend to infinity, but the ratios ↵ = m/n, µ = p/m are held finite.

Similar "thermodynamic" or "high-dimensional" limiting procedures are used in statistical physics, eg in the study of random matrices and spin-glass models in large spatial dimensions [14, 13] .

Such limits are also well-studied in modern statistics [17] (for example to understand phase-transition phenomena in the LASSO algorithm [6] ).

While there is a large literature on the LASSO phase transition, we were unable to find any computations of the GE curves that span across the underparametrized and overparametrized regimes in a systematic model as presented here.

We derive analytical formulae for TE and GE with l 2 or ridge regularization.

For l 1 regularization, explicit formulae are given in some parameter regimes.

More generally for the l 1 case we obtain a pair of simultaneous nonlinear equations in two variables which implicitly define the MSE.

These can be solved numerically to obtain the GE.

The nonlinear equations are given in closed form without hidden parameters and do not require integration.

Analytical Formulae: T E 2 , GE 2 are the training and generalization errors for the l 2 penalized case, and GE 1 the generalization error for the l 1 penalized case.

Due to lack of space we do not present the analytical formulae for > 0 as these expressions are complex, but the corresponding analytical expressions were used to generated the theory curves in Fig.1 for the case > 0.

The derivations employ the cavity mean field theory approach [16] .

Here 2 ef f = 2 + ⇢(1 µ↵).

Note that the formulae for GE agree for the l 2 and l !

cases in the underparametrized regime µ < 1, but diverge in the overparametrized regime: infinitesimal l 1 regularization provides no better generalization than the pseudoinverse based procedure unless there is overparametrization.

Further note that "good generalization" (GE small) begins when µ↵ > 1 not at the overfitting peak (µ = 1).

@highlight

Proposes an analytically tractable model and inference procedure (misparametrized sparse regression, inferred using L_1 penalty and studied in the data-interpolation limit) to study deep-net related phenomena in the context of inverse problems. 