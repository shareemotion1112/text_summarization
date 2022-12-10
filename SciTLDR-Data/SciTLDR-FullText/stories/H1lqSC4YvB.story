The reparameterization trick has become one of the most useful tools in the field of variational inference.

However, the reparameterization trick is based on the standardization transformation which restricts the scope of application of this method to distributions that have tractable inverse cumulative distribution functions or are expressible as deterministic transformations of such distributions.

In this paper, we generalized the reparameterization trick by allowing a general transformation.

Unlike other similar works, we develop the generalized transformation-based gradient model formally and rigorously.

We discover that the proposed model is a special case of control variate indicating that the proposed model can combine the advantages of CV and generalized reparameterization.

Based on the proposed gradient model, we propose a new polynomial-based gradient estimator which has better theoretical performance than the reparameterization trick under certain condition and can be applied to a larger class of variational distributions.

In studies of synthetic and real data, we show that our proposed gradient estimator has a significantly lower gradient variance than other state-of-the-art methods thus enabling a faster inference procedure.

Most machine learning objective function can be rewritten in the form of an expectation:

where θ is a parameter vector.

However, due to the intractability of the expectation, it's often impossible or too expensive to calculate the exact gradient w.r.t θ, therefore it's inevitable to estimate the gradient ∇ θ L in practical applications.

Stochastic optmization methods such as reparameterization trick and score function methods have been widely applied to address the stochastic gradient estimation problem.

Many recent advances in large-scale machine learning tasks have been brought by these stochastic optimization tricks.

Like in other stochastic optimzation related works, our paper mainly focus on variational inference tasks.

The primary goal of variational inference (VI) task is to approximate the posterior distribution in probabilistic models (Jordan et al., 1999; Wainwright & Jordan, 2008) .

To approximate the intractable posterior p(z|x) with the joint probability distribution p(x, z) over observed data x and latent random variables z given, VI introduces a parameteric family of distribution q θ (z) and find the best parameter θ by optimizing the Kullback-Leibler (KL) divergence D KL (q(z; θ) p(z|x)).

The performance of VI methods depends on the capacity of the parameteric family of distributions (often measured by Rademacher complexity) and the ability of the optimizer.

In this paper, our method tries to introduce a better optimizer for a larger class of parameteric family of distributions.

The main idea of our work is to replace the parameter-independent transformation in reparameterization trick with generalized transformation and construct the generalized transformation-based (G-TRANS) gradient with the velocity field which is related to the characteristic curve of the sublinear partial differential equation associated with the generalized transformation.

Our gradient model further generalizes the G-REP (Ruiz et al., 2016) and provides a more elegant and flexible way to construct gradient estimators.

We mainly make the following contributions:

1.

We develop a generalized transformation-based gradient model based on the velocity field related to the generalized transformation and explicitly propose the unbiasedness constraint on the G-TRANS gradient.

The proposed gradient model provides a more poweful and flexible way to construct gradient estimators.

2.

We show that our model is a generalization of the score function method and the reparameterization trick.

Our gradient model can reduce to the reparameterization trick by enforcing a transport equation constraint on the velocity field.

We also show our model's connection to control variate method.

3.

We propose a polynomial-based gradient estimator that cannot be induced by any other existing generalized reparameterization gradient framework, and show its superiority over similar works on several experiments.

The rest of this paper is organized as follows.

In Sec.2 we review the stochastic gradient variational inference (SGVI) and stochastic gradient estimators.

In Sec.3 we propose the generalized transformation-based gradient.

In Sec.4 we propose the polynomial-based G-TRANS gradient estimator.

In Sec.5 we study the performance of our gradient estimator on synthetic and real data.

In Sec.6 we review the related works.

In Sec.7 we conclude this paper and discuss future work.

To obtain the best variational parameter θ, rather than minimize the KL divergence D KL (q(z; θ) p(z|x)), we usually choose to maximize the evidence lower bound (ELBO) (Jordan et al., 1999) ,

where

The entropy term H[q(z; θ)] is often assumed to be available analytically and usually omitted in the procedure of stochastic optimization.

This stochastic optimization problem is the basic setting for our method and experiments.

Without extra description, we only consider the simplified version of the ELBO:

Generally, this expectation is intractable to compute, let alone its gradient.

Therefore, a common stochastic optimization method for VI task is to construct a Monte Carlo estimator for the exact gradient of the ELBO w.r.t θ.

Among those gradient estimators, the score function method and the reparamterization trick are most popular and widely applied.

Score function method.

The score function estimator, also called log-derivative trick or reinforce Glynn (1990); Williams (1992) is a general way to obtain unbiased stochastic gradients of the ELBO (Paisley et al., 2012; Ranganath et al., 2014; Mnih & Gregor, 2014) .

The simplest variant of the score function gradient estimator is defined as:

and then we can build the Monte Carlo estimator by drawing samples from the variational distribution q θ (z) independently.

Although the score function method is very general, the resulting gradient estimator suffers from high variance.

Therefore, it's necessary to apply variance reduction (VR) methods such as RaoBlackwellization (Casella & Robert, 1996) and control variates (Robert & Casella, 2013) in practice.

Reparameterization trick.

In reparameterization trick, we assume that there is an invertible and continuously differentiable standardization function φ(z, θ) that can transform the variational distribution q(z; θ) into a distribution s(ρ) that don't depend on the variational parameter θ as follows,

θ (ρ) Then the reparameterization trick can turn the computation of the gradient of the expectation into the expectation of the gradient:

Although this reparameterization can be done for many commonly used distributions, such as the Gaussian distribution, it's hard to find appropriate standardization functions for a number of standard distributions, such as Gamma, Beta or Dirichlet because the standardization functions will inevitably involve special functions.

On the other hand, though the reparameterization trick is not as generally applicable as the score function method, it does result in a gradient estimator with lower variance.

Define a random variable ρ by an invertible differentiable transformation ρ = φ(z, θ), where φ is commonly called generalized standardization transformation (Ruiz et al., 2016 ) since it's dependent on the variational parameter θ.

Theorem 3.1.

Let θ be any component of the variational parameter θ, the probability density function of ρ be w(ρ, θ) and

where

The proof details of the Theorem.3.1 are included in the Appendix.

A.1.

We refer to the gradient ∂L ∂θ with v θ satisfying the unbiasedness constraint as generalized transformation-based (G-TRANS) gradient.

We can construct the G-TRANS gradient estimator by choosing v θ of specific form.

In the following, we demonstrate that the score function gradient and reparameterization gradient are special cases of our G-TRANS gradient model associating with special velocity fields.

Remark.

The score function method is a special case of the G-TRANS model when

The standardization function φ doesn't depend on the parameter θ when v θ = 0 according to the velocity field equation (Equ.5).

Conversely, for any φ that doesn't depend on θ, we have

= 0, thus the resulting gradient estimator has a same variance as the score function estimator.

Remark.

The reparameterization trick is a special case when

The detailed computation to obtain the transport equation (Equ.7) is included in the Appendix.

A.1.

The transport equation is firstly introduced by (Jankowiak & Obermeyer, 2018) , however, their work derive this equation by an analog to the optimal transport theory.

In 1-dimensional case, for any standardization distributions w(ρ) that doesn't depend on the parameter θ, the variance of the resulting gradient estimator is some constant (for fixed θ) determined by the unique 1-dimensional solution of the transport equation.

For the existence of the velocity field v θ and the generalized standardization transformation φ(z, θ), g(z, θ) must satisfy some strong differential constraints (Evans, 2010) .

We can see that the G-TRANS model is a special case of the control variate method with a complex differential structure.

This connection to CV means our gradient model can combine the advantages of CV and generalized reparameterization.

Theorem.3.1 transforms the generalized unbiased reparameterization procedure into finding the appropriate velocity field that satisfy the unbiasedness constraint.

It's possible to apply variational optimization theory to find the velocity field with the least estimate variance, however, the solution to the Euler-Lagrange equation contains f (z) in the integrand which makes it impractical to use in real-world model (See Appendix.

A.2 for details).

By introducing the notion of velocity field, we provide a more elegant and flexible way to construct gradient estimator without the need to compute the Jacobian matrix for a specific transformation.

In the next section, we introduce a polynomial-based G-TRANS gradient estimator that cannot be incorporated into any other existing generalized reparameterized gradient framework and is better than the reparameterization gradient estimator theoretically.

In this section, we always assume that the base distribution q(z, θ) can be factorized as

where N is the dimension of the random variable z, θ i is a slice of θ and θ i share no component with θ j if i = j. We consider an ad-hoc velocity field family:

We always assume v θ ah to be continuous which guarantees the existence of the solution to the velocity field equation.

We verify in the Appendix.

A.3 that v θ ah (z, θ) satisfy the unbiasedness constraint if h(z, θ) is bounded.

It's easy to see that the gradient estimator that results from v θ ah is more general than the score function method or reparameterization trick since they are two special cases when h(z, θ) = 0 or h(z, θ) = f (z) respectively.

In this paper, we mainly consider a more special family of the v θ ah (z, θ):

where

zi ∂q(z ,θ) ∂θ dz i ), but their properties are similar (we present some theoretical results of v θ dp in the Appendix.

A.4).

Therefore we only consider v θ poly (z, θ) here.

We refer to v θ poly as polynomial velocity field.

Proposition 4.1.

For distributions with analytical high order moments such as Gamma, Beta or Dirichlet distribution, the expectation

are polynomials of random variable z. Therefore, for distribution with analytical high order moments,

With Proposition.4.1, we can write the G-TRANS gradient for the polynomial velocity field as:

Thus we can construct a G-TRANS gradient estimator based upon the polynomial velocity field with a samplez drawn from q(z, θ):

The polynomial-based G-TRANS gradient estimator has a form close to control variate, thus cannot be induced by any other existing generalized reparameterized gradient framework.

In the following, we show that the polynomial-based G-TRANS gradient estimator performs better than the reparameterization gradient estimator under some condition.

dz N ), then the gradient estimator resulted from polynomial velocity field has a smaller variance than the reparameterization gradient estimator.

Proof.

Since E q [P k (z, θ) ∂ ∂θ log q] can be resolved analytically, we have

then by reorganizing the expression Var(− ∂f ∂zi

), we can prove this proposition.

As an example about how to choose a good polynomial, for

), we can obtain a polynomial-based G-TRANS gradient estimator that is better than the reparameterization gradient estimator according to the Proposition.4.2.

And we can adjust the value of C i (θ) to obtain better performance.

According to the approximation theory, we can always find a polynomial P k (z, θ) that is close enough to f (z), and in this case, we can dramatically reduce the variance of the resulting gradient estimator.

For example, within the convergence radius, we can choose P k (z, θ) to be the k-th degree Taylor polynomial of f (z) with the remainder |f (z) − P k (z, θ)| being small.

In the practical situation, however, it's often difficult to estimate the coefficients of the polynomial P k (z, θ).

And when k is large, we need to estimate O(N k ) coefficients which is almost impossible in real-world applications.

Therefore in the following experiments, we only consider k < 2.

In this section, we use a Dirichlet distribution to approximate the posterior distribution for a probilistic model which has a multinomial likelihood with a Dirichlet prior.

We use Gamma distributions to simulate Dirichlet distributions.

If

Then the problem we study here can be written as:

with f (z) being the multinomial log-likelihood.

We use shape parameter α = (α 1 , . . .

, α K ) to parameterize the variational Dirichlet distribution.

To construct polynomial-based G-TRANS gradient estimator for the factorized distribution K k=1 Gamma(z k ; α k , 1), we need an accurate and fast way to approximate the derivative of the lower incomplete gamma function (part of the gamma CDF) w.r.t the shape parameter.

The lower incomplete gamma function γ(α, z) is a special function and does not admit analytical expression for derivative w.r.t.

the shape parameter.

However, for small α and z, we have

In practice, we take the first 200 terms from this power series.

And the approximation error is smaller than 10 −9 when α < 5 and z < 20 with double precision floating point number.

For large α, we use central finite difference to approximate the derivative.

This approximation scheme for lower incomplete gamma function can also be used to construct polynomial-based G-TRANS gradient estimator for distributions that can be simulated by the Gamma distribution such as Beta distribution and Dirichlet distribution.

We follow the experiment setting in Naesseth et al. (2017) .

Fig.1 shows the resulting variance of the first component of the gradient based on samples simulated from a Dirichlet distribution with K = 100 components, and gradients are computed with N = 100 trials.

We use P 1 (z) = c · z to construct the G-TRANS gradient estimator, and we assign 0.2,0 and −0.1 to c successively as α 1 increases.

Results.

From Fig.1 , we can see that the IRG (Figurnov et al., 2018 ) method and our G-TRANS gradient estimator has obviously lower gradient variance than the RSVI (even with the shape augmentation trick (Naesseth et al., 2017) ) or G-REP (Ruiz et al., 2016) method.

Further, our G-TRANS gradient estimator outperforms the IRG method when α 1 is large though there is no obvious difference between these two methods when α 1 is small.

In this section, we study the performance of our G-TRANS gradient estimator on the Sparse Gamma deep exponential family (DEF) model (Ranganath et al., 2015) with the Olivetti faces dataset that consists of 64 × 64 gray-scale images of human faces in 8 bits.

We follow the Sparse Gamma DEF setting in Naesseth et al. (2017) where the DEF model is specified by: (Naesseth et al., 2017) .

C is the polynomial coefficient, B denotes shape augmentation (Figurnov et al., 2018) and optimal concentration is α = 2.

Here n is the number of observations, is the layer number, k denotes the k-th component in a specific layer and d is the dimension of the output layer (layer 0).

z n,k is local random variable, w k,k is global weight that connects different layers like deep neural networks, and x n,d denotes the set of observations.

We use the experiment setting in Naesseth et al. (2017) .

α z is set to 0.1, all priors on the weights are set to Gamma(0.1, 0.3), and the top-layer local variables priors are set to Gamma(0.1, 0.1).

The model consists of 3 layers, with 100, 40, and 15 components in each.

All variational Gamma distributions are parameterized by the shape and mean.

For non-negative variational parameters θ, the transfomration θ = log(1 + exp(ϑ)) is applied to avoid constrained optimization.

In this experiment, we use the step-size sequence ρ n proposed by Kucukelbir et al. (2017) :

(16) δ = 10 −16 , t = 0.1, η = 0.75 is used in this experiment.

The best result of RSVI is reproduced with B = 4 (Naesseth et al., 2017) .

We still use P 1 (z) = c · z to construct the G-TRANS gradient estimator and we use c = −10.0 for all time.

Results.

From Fig.2, We can see that G-TRANS achieves significant improvements in the first 1000 runs and exceeds RSVI though with a slower initial improvement.

G-TRANS achieves obviously better accuracy than ADVI, BBVI, G-REP and RSVI, and keeps improving the ELBO even after 75000 runs.

G-TRANS is faster than the IRG in early training stage which means G-TRANS has a lower gradient variance.

However, this speed advantage of G-TRANS gradually decreases as the step size goes down in the later training stage.

There are already some lines of research focusing on extending the reparameterization trick to a larger class of distributions.

The G-REP (Ruiz et al., 2016) generalizes the reparameterization gradient by using a standardization transformation that allows the standardization distribution to depend weakly on variational parameters.

Our gradient model gives a more elegant expression of the generalized reparameterized gradient than that of G-REP which decomposes the gradient as g rep + g cor .

Different from G-REP, our model hides the transformation behind the velocity field thus the expensive computation of the Jacobian matrix of the transformation is evaded.

And it's more flexible to construct gradient estimator with the velocity field than the very detailed transformation.

The RSVI (Naesseth et al., 2017 ) develops a similar generalized reparameterized gradient model with the tools from rejection sampling literatures.

RSVI introduces a score function gradient term to compensate the gap that is caused by employing the proposal distribution of a rejection sampler as a surrogate distribution for reparameterization gradient, although the score function gradient term can often be ignored in practice to reduce the gradient variance at the cost of small bias.

Unlike RSVI, our gradient estimator can be constructed with deterministic procedure which avoids the additional stochasticity introduced by the accept-reject steps thus lower gradient variance.

The path-wise derivative (Jankowiak & Obermeyer, 2018 ) is closely related to our model.

They obtain the transport equation by an analog to the displacement of particles, while we derive the transport euqation for reparameterization gradient by rigorous mathematical deduction.

The path-wise gradient model can be seen as a special case of our G-TRANS gradient model.

Their work only focus on standard reparameterization gradient while our model can admit generalized transformation-based gradient.

The velocity field used in their work must conform to the transport equation while we only require the velocity field to satisfy the unbiasedness constraint.

The implicit reparameterization gradient (IRG) (Figurnov et al., 2018) differentiates from the path-wise derivative only by adopting a different method for multivariate distributions.

There are also some other works trying to address the limitations of standard reparameterization.

Graves (2016) applies implicit reparameterization for mixture distributions and Knowles (2015) uses approximations to the inverse CDF to derive gradient estimators.

Both work involve expensive computation that cannot be extended to large-scale variational inference.

Schulman et al. (2015) expressed the gradient in a similar way to G-REP and automatically estimate the gradient in the context of stochastic computation graphs, but their work is short of necessary details therefore cannot be applied to general variational inference task directly.

ADVI (Kucukelbir et al., 2017) transforms the random variables such that their support are on the reals and then approximates transformed random variables with Gaussian variational posteriors.

However, ADVI struggles to approximate probability densities with singularities as noted by Ruiz et al. (2016) .

We proposed a generalized transformation-based (G-TRANS) gradient model which extends the reparameterization trick to a larger class of variational distributions.

Our gradient model hides the details of transformation by introducing the velocity field and provides a flexible way to construct gradient estimators.

Based on the proposed gradient model, we introduced a polynomial-based G-TRANS gradient estimator that cannot be induced by any other existing generalized reparameterization gradient framework.

In practice, our gradient estimator provides a lower gradient variance than other state-of-the-art methods, leading to a fast converging process.

For future work, We can consider how to construct G-TRANS gradient estimators for distributions that don't have analytical high-order moments.

We can also utilize the results from the approximation theory to find certain kinds of high-order polynomial functions that can approximate the test function effectively with cheap computations for the coefficients.

Constructing velocity fields with the optimal transport theory is also a promising direction.

A.1 PROOF OF THEOREM.3.1

We assume that transformed random variable ρ = φ(z, θ) is of the same dimension as z.

And we assume that there exists ψ(ρ, θ) that satisfy the constraint z = ψ(φ(z, θ), θ).

Firstly, by the change-of-variable technique, we have

Take derivative w.r.t θ (any component of θ) at both sizes, we have

With the rule of determinant derivation, we have

Substitute the Equ.19 into Equ.18, we have

, we obtain the first conclusion of the Theorem.3.1.

As for the second part, we have

Thus we obtain the second part of the Theorem.3.1.

Proof ends.

As a by-product, if we make ∂ ∂θ w(φ(z, θ), θ) = 0 , we can obtain the transport equation for the reparameterization trick:

And ∂ ∂θ w(φ(z, θ), θ) = 0 also means that the standardization distribution is independent with θ which is the core of the reparameterization trick.

For the simplicity of the proof, we only consider the 1-dimensional here.

And denote where with the unbiased constraint, we have E q(z,θ) [r(z, θ)] = E q(z,θ) [f (z) ∂q(z,θ) ∂θ q(z,θ) ] = const, so we need to consider the term E q(z,θ) [(r θ (z, θ)) 2 ] only.

According to the Euler-Lagrange equation, we have

Simplify it, we have (f ∂q ∂θ

Then we have

Thus we have

which is usually intractable in real world practice.

Here we verify that v

If h(z, θ) is bounded, we have

Therefore, E q θ [

If we take the dual polynomial velocity field v θ dp in the G-TRANS framework, we can reach a dual result to the Proposition.4.2: Proposition A.1.

If Cov(P k ∂ log q(z,θ) ∂θ , (2f − P k ) ∂ log q(z,θ) ∂θ ) > 0, then the gradient estimator resulted from dual polynomial velocity field has a smaller gradient variance than the score function gradient estimator.

The proof is similar to that of Proposition.4.2.

@highlight

We propose a novel generalized transformation-based gradient model and propose a polynomial-based gradient estimator based upon the model.