GloVe and Skip-gram word embedding methods learn word vectors by decomposing a denoised matrix of word co-occurrences into a product of low-rank matrices.

In this work, we propose an iterative algorithm for computing word vectors based on modeling word co-occurrence matrices with Generalized Low Rank Models.

Our algorithm generalizes both Skip-gram and GloVe as well as giving rise to other embedding methods based on the specified co-occurrence matrix, distribution of co-occurences, and the number of iterations in the iterative algorithm.

For example, using a Tweedie distribution with one iteration results in GloVe and using a Multinomial distribution with full-convergence mode results in Skip-gram.

Experimental results demonstrate that multiple iterations of our algorithm improves results over the GloVe method on the Google word analogy similarity task.

Word embeddings are low dimensional vector representations of words or phrases.

They are applied to word analogy tasks and used as feature vectors in numerous tasks within natural language processing, computational linguistics, and machine learning.

They are constructed by various methods which rely on the distributional hypothesis popularized by Firth: "words are characterized by the company they keep" BID9 .

Two seminal methodological approaches to finding word embeddings are Skip-gram [Mikolov et al., 2013a] and GloVe [Pennington et al., 2014] .

Both methods input a corpus D, process it into a word co-occurence matrix X, then output word vectors with some dimension d.

Skip-gram processes a corpus with w words into a count co-occurence matrix X ∈ R w×w , where x ij is the number of times word w i appears in the same context as the word w j .

Here, two words being in the same context means that they're within l c tokens of each other.

Define this co-occurence matrix to be the count co-occurence matrix.

Next, Skip-gram [Pennington et al., 2014 where u u u T i is the i th row of U , then defines the word vectors to be the rows ofÛ .GloVe processes a corpus with w words into a harmonic co-occurence matrix X ∈ R w×w where x ij is the harmonic sum of the number of tokens between words w i and w j over each co-occurrence.

That is, x ij = p1<p2,|p1−p2|≤lc,D(p1)=wi,D(p2)=wj h(x ij ) u u u DISPLAYFORM0 where a i and b j are bias terms, h(x ij ) = (min{x ij , x max }) .75 is the weight, and x max is some prespecified cutoff.

GloVe then defines the estimated word vectors to be the rows of 1 2Û + 1 2V .

In both Skip-gram and GloVe, a matrix of co-occurences X is introduced by processing the corpus, and an objective function is introduced to find a low rank factorization related to the co-occurences X. In this paper, we derive the objective functions from a model-based perspective.

We introduce an iterative algorithm, and show that problem (1) results from running the iterative algorithm on full-convergence mode for a Multinomial model and problem (2) is one step of the iterative algorithm for a Tweedie model.

This algorithm additionally allows us to introduce methods to "fill in the gaps" between Skip-gram and GloVe and to introduce altogether new methods for finding word vectors.

We saw that Skip-gram and GloVe compute a co-occurence matrix X which results from processing the corpus D and an objective function J to relate the matrix X to a product of low rank matrices U and V .

Many existing approaches for explaining word embedding methods do so by identifying or deriving the co-occurence matrix X or the objective function J. In this section, we review relevant work in this area, which helps frame our approach discussed in Section 4.1.Much of the related work involves using the co-occurence matrix from Skip-gram.

For the remainder of this section, let X be the count co-occurence matrix.

Early approaches to finding low-dimensional embeddings of words relied on the singular value decomposition [Landauer et al., 1998, Turney and Pantel, 2010] .

These methods would truncate the singular value decomposition by zeroing out the small singular values.

BID7 show that this is equivalent to using an objective function J which is invariant to orthogonal transformation.

For simplicity, we specialize to the Frobenius norm and say these early approaches find arg min DISPLAYFORM0 F is the objective function and X is the co-occurence matrix.

The co-occurence matrix and the loss function for Skip-gram can be read off from problem (1): the co-occurence matrix is X and the objective function is written in problem (1) with u u u T i v v v j replaced by m ij .

BID4 find a probabilistic interpretation of this loss function related to a Multinomial distribution, but do not take advantage of it and only replace the inner product with a (higher dimensional) variant, somewhat similar to the approach in Tifrea et al. [2018] .

Mikolov et al. [2013a] introduce Skip-gram with negative sampling (SGNS), a variant of Skip-gram.

If we view Skip-gram as maximizing the true positive rate of predicting a word will appear within a context window of another word, we can view SNGS as maximizing the true positive rate plus k times an approximation of the true negative rate.

When k = 0, Skip-gram and SGNS coincide.

BID19 use a heuristic argument to interpret SGNS as using a co-occurence matrix that is a shifted PMI matrix.2 However, they did not determine the objective function.

Later, Li et al. [2015] and Landgraf and Bellay [2017] explicitly identified both the co-occurence matrix and the objective function.

They find a different co-occurence matrix than BID19 , one that does not depend on k, while their loss function does depend on k. Surprisingly, they establish that SGNS is finding a low-rank matrix related to X, the same matrix that Skip-gram uses.

The loss function is w,w i,j=1 DISPLAYFORM1 2 Define the total number of times word wi appears to be xi· = w j=1 xij, the total number of times context wj appears to be x·j = w i=1 xij, and the total number of words to be x·· = w,w i,j=1 xij.

The shifted PMI matrix has entries log DISPLAYFORM2 Landgraf and Bellay [2017] explain that this loss function has a probabilistic interpretation, and they use that interpretation to recover the shifted PMI matrix as a prediction from within their model.

The approach in this paper will be to view the entries of the co-occurence matrix as random variables and introduce an objective function via the likelihood of that random variable.

Our approach is most similar to Landgraf and Bellay [2017] and, to a lesser extent, BID4 .

In order proceed, some background in probabilistic modeling and estimation needs to be developed.

In this section, we review iteratively reweighted least squares (IRLS) for generalized linear models and review generalized low rank models [Udell et al., 2016] .

Further background (and notation) in exponential dispersion families and generalized linear models is developed in Section A.

Generalized linear models (GLMs) are a flexible generalization of linear regression where the mean is a not necessarily linear function of a coefficient β β β and the response has an error distribution which is an exponential dispersion family.

The coefficient β β β is unknown and a target of estimation.

The standard approach to estimate β β β is maximum likelihood estimation [Fisher, 1922, Section 7] to produce the maximum likelihood estimator, or MLE,β β β.

A computational approach to find the MLE is through Fisher scoring, a variant of Newton's method on the log likelihood which uses the expectation of the Hessian in place of the Hessian [Agresti, 2015, Section 4.5] .

Define (β β β) to be the log likelihood.

Specifically, Fisher scoring produces a sequence of estimates {β β β DISPLAYFORM0 ), where ∇ is the gradient and D 2 is the Hessian.

Upon plugging in the gradient and expected Hessian for an exponential dispersion family, a surprising identity emerges: each iteration of Fisher scoring is equivalent to minimizing a weighted least squares objective: DISPLAYFORM1 where the weight H (t) and pseudo-response z (t) at iteration t have DISPLAYFORM2 DISPLAYFORM3 , and µ DISPLAYFORM4

Principal components analysis BID13 is one well-known method for finding a low rank matrix related to X ∈ R w×c .

In principal components analysis, we model x ij ind.

DISPLAYFORM0 A maximum likelihood estimator for u u u i is taken to be a low-dimensional embedding of the i th row of X. The low-dimensional embedding enables interpretability and reduces noise.

However, data cannot always be viewed as being drawn from a normal distribution, so it's necessary to extend the method of principal components to non-normal data.

The extension can be made in a manner similar to the extension from linear models to generalized linear models: the new model, called a generalized low rank model [Udell et al., 2016] allows us to estimate model-based low-dimensional embeddings of non-normal data.

Definition 1 For some exponential dispersion family ED(µ, ϕ) with mean parameter µ and dispersion parameter ϕ, the model for X ∈ R w×c is a generalized low rank model with link function g when DISPLAYFORM1 DISPLAYFORM2 where u u u i , v v v j ∈ R d are the rows of matrices U ∈ R w×d and V ∈ R c×d , respectively, and a a a ∈ R w and b b b ∈ R c are bias (or offset) terms.

The difference between the generalized low rank model and the generalized linear model is in the systematic component in equation FORMULA12 .

Here, the data is modeled as having its link-transformed mean be a matrix with rank at most d. This formalizes the way in which we relate the co-occurence matrix X to a low rank factorization.

When the link function g is taken to be canonical, the generalized low rank model is identical to ePCA BID3 .

The generalization is worthwhile since the canonical link can be inappropriate, as we will see, for instance, in Section 5.1.

We now present a method to find word vectors.

A key innovation in the method is an iterative algorithm inspired by IRLS to find a maximum likelihood estimator in a generalized low rank model.

Our method has three steps:Step 1 Choose a co-occurence matrix X ∈ R w×c to summarize the document. (Note, in many cases c = w so that the "contexts" are just the words.)Step 2 Choose a plausible exponential dispersion family to model the entries of the co-occurence matrix.

Choose a corresponding link function.

Step 3 Choose a number of iterations r to run IWLRLS (Algorithm (1)) with the input specified above to output word vectors.

DISPLAYFORM0 ; Evaluate the least squares problem arg min DISPLAYFORM1 Algorithm 1: Iteratively weighted low rank least squares (IWLRLS) algorithm for GLRMsThe first step of our method processes the corpus in order to extract the linguistic information.

Some co-occurence statistics use more information than others: for instance, the harmonic co-occurence matrix makes use of the number of tokens between words while the count co-occurence matrix does not.

A typical tuning parameters here is the length l c of the context window.

We view this step as involving a "linguistic" choice.

The second step specifies a distribution for the co-occurence matrix.

A distribution can be considered as plausibly corresponding to reality if it can be derived by a connection to the corpus.

In our Co-occurence: Harmonic Table 1 : The rows refers to the number of steps of IWLRLS.

A "·" represents no existing work.

All filled-in positions in the lowest row were established in previous work.

framework, the model is explicit: this is helpful since knowing a model provides interpretation for its output [Gilpin et al., 2018, Section II.A.] .

The choice of distribution will often determine, through convention, the link function, so the link function often does not need to be separately chosen.

We view this step as involving a "statistical" choice.

DISPLAYFORM2 The third step runs IWLRLS, a generalized version of IRLS.

Recall that IRLS is derived by iteratively maximizing a second order Taylor expansion of the likelihood as a function β.

The Taylor expansion is centered at the previous iterate.

IWLRLS can be derived by iteratively maximizing a second order Taylor expansion of the likelhood as a function of η subject to the constraint 6.

We view this as a "computational" choice that we fix in advance.

In the following subsections, we run through many examples of our method as it would be used in practice.

There are two distinct choices of co-occurence matrices that are made.

Various choices of distributions recover common methods for finding word vectors.

An altogether new estimator is proposed via an improvement of the assumed distribution in Skip-gram.

Casting these estimators in this general framework provides an interpretation and understanding of them: we make explicit their assumptions and therefore know the driver of their behavior.

We will apply our proposed method under the choice of the harmonic co-occurence matrix and the Tweedie distribution: one iteration of IWLRLS will recover GloVe.

Step 1 The first step of our method is to pick a co-occurence matrix that summarizes the corpus.

We choose the harmonic co-occurence matrix X ∈ R w×w .Step 2 Now we must determine a plausible distribution for the co-occurence matrix that is an exponential dispersion family.

Recall that the Tweedie distribution has the property mentioned in equation FORMULA0 that it is a sum of Poisson many independent Gamma distributions.

An informal way to write this is that DISPLAYFORM0 We argue that the Tweedie distribution is reasonable by connecting the Poisson and Inverse Gamma distributions displayed above to attributes of the corpus.

Intuitively, it is reasonable that the number of times word w i and word w j co-occur within the corpus can be modeled as having a Poisson distribution.

Another choice of distribution is that of an Inverse Gamma distribution for the number of tokens between word w i and word w j at some co-occurence, although it is an approximation as the number of tokens is an integer while the Inverse Gamma is supported on non-integers.

Instead of using the canonical link function, we will take g(µ) = log µ, which is standard [Smyth, 1996] .

A problem with the canonical link function preventing its use is that its range is nonpositive.

Step 3 Next, we find the form of the weight H and the pseudo-response Z that the Tweedie distribution provides.

This amounts to plugging in the cumulant generating function ψ that is given in Section A.1.

This results in DISPLAYFORM1 When the algorithm is initialized withμ (0) = X, the pseudo-response simplifies to z ij = log x ij .Taking the power p = 1.25, the weight simplifies to x 3/4 ij .

In summary, we've shown that: Result 1 Inputting the harmonic co-occurence matrix, the Tweedie distribution with power p = 1.25, the log link, and the number of iterations k = 1 into IWLRLS results in GloVe (without the algorithmic regularization induced by truncating the weights.)Given this connection, we can extend GloVe for several iterations rather than one or even use the full likelihood.

We experiment with this using real data examples in Section 6.

This result shows that even though the first iteration does not depend on word pairs where x ij = 0, later iterations do.

We now consider an alternative first step: we choose another co-occurence matrix to summarize the corpus.

Then, we make multiple possible choices for step 2 to illustrate connections to previous work that step 3 recovers.

Various choices for step 2 will recover the SVD BID17 ], Skip-gram [Mikolov et al., 2013a] , a new estimator which is a distributional improvement over those, and Skip-gram with negative sampling [Mikolov et al., 2013b] .Step 1 We choose the count co-occurence matrix.

Step 2 A proposed distribution for the entries of X is the Gaussian distribution.

This may not be the best choice, since the entries of X are non-negative integers.

As is usual, we take the link function to be g(µ) = µ. We restrict the systematic component to not include the bias terms, so that η ij = u u u DISPLAYFORM0 Step 3 We showed in Section A.1 that the cumulant generation function from the normal distribution is ψ(θ) = 1 2 θ 2 .

This makes it so that DISPLAYFORM1 In other words, the IWLRLS algorithm will always converge in one iteration, so our method recovers the method of computing a truncated SVD of X by BID7 .Another choice that could have been made in step 2 is to have the link function g(µ) = log µ. This still may not be the best choice since the normal distribution still has the same problems as before.

Step 2 Another proposed distribution for the entries of X is a Multinomial distribution.

Specifically, we could propose that the the row of X corresponding to word w i has the distribution x x x i ∼ Multinomial w j=1 x ij , π π π , where π π π ∈ R w is vector of probabilities of word w i appearing within a context window with the other words and w j=1 x ij is the total number of times word w i appears in the corpus.

We take the link function to be the multi-logit.

3 Cotterell et al. [2017] show that the objective function of Skip-gram coincides with the likelihood of this model when the bias terms are removed, so that the systematic component η ij = u u u Step 3 The Poisson trick BID2 can be used to reduce estimation in a Multinomial model to estimation in a particular Poisson model.

LetÛ ,V be the maximum likelihood estimators in the Multinomial generalized low rank model described in step 2.

Using this trick, it holds thatâ a a, (the same)Û , and (the same)V are maximum likelihood estimators in a Poisson generalized low rank model with independent responses x ij and systematic component DISPLAYFORM0 Notice that there is only one bias term.

The weight and pseudo-response are DISPLAYFORM1

In the previous subsubsection, we saw that the choice of Multinomial model implicitly gives rise to a Poisson model with a systematic component given by equation FORMULA20 .

Since it could be most appropriate to have bias terms for both rows and columns due to the symmetry of the co-occurence matrix, we directly introduce a Poisson estimator with a non-restricted systematic component.

Step 2 Another proposed distribution is a Poisson.

Due to the "law of rare events" [Durrett, 2010, Section 3.6 .1], this is a plausible model.

We use the canonical link function g(µ) = log µ.Step 3 The cumulant generating function is ψ(θ) = exp(θ) BID0 , so that the weight and pseudo-response are given by equations (10).

BID1 propose an estimator which is a close variant of one iteration of IWLRLS.

At one point in their derivation, they (using our notation) take η ij = u u u i − v v v j 2 2 + c, where c is an arbitrary constant which does not depend on the word.

This is inspired by their theorem 2.2.

On the other hand, taking η ij = u u u T i v v v j + a i + b j (as in equation 6) in their derivation recovers one iteration of IWLRLS.

The Negative-Binomial distribution is commonly used as an alternative for the Poisson in the presence of over-dispersion, which is the case when the variance is higher than the mean.

It produces the same weight and pseudo-response as the Poisson.

Step 2 We model x ij ind.

DISPLAYFORM0 is an inflated count, η ij = u u u T i v v v j , and k ≥ 0.

Landgraf and Bellay [2017] showed that a maximum likelihood estimator from this model with canonical link g(π) = log π 1−π is identical to a SGNS estimator.

Step 3 The cumulant generating function for the binomial distribution is ψ(θ) = log(1 + exp θ), so the weight and pseudo-response are: DISPLAYFORM1

In Section 4.1 we introduced the IWLRLS algorithm to compute word vectors such as those produced by GloVe or SGNS.

We now conduct quantitative evaluation experiments on an English word analogy task, a variety of word similarity tasks [Mikolov et al., 2013a ] to demonstrate the performance of the algorithm.

First, in Section 6.1 we introduce the analogy similarity task for evaluating word vectors.

In Section 6.2 we present results of the algorithm with different distributions according to those presented in Section 5.1 and 5.2.

In Section B.1 we provide parameter configurations and training procedures, and in Sections B.2-B.5 we present results of IWLRLS in numerous scenarios showcasing improvement through multiple iterations and robustness to other model parameters.

We introduce the word analogy task following the presentation of [Pennington et al., 2014 ].

The word analogy task is a dataset of 19, 544 statements of the basic form "a is to b as c is to __", which are divided into a semantic and syntactic subsets.

The semantic statements are typically analogies relating to people, places, or nouns such as "Athens is to Greece as Berlin is to __", while the syntactic questions relate to verb or adjective forms such as "dance is to dancing as fly is to __".

The basic analogy statement is answered by finding the closest vector u u u d to u u u b − u u u a + u u u c 4 in the embedding space via cosine similarity 5 .

The task has been shown under specific assumptions to be provably solvable by methods such as GloVe and Skip-gram BID8 BID12 ] and as such is closely related to solving the objectives introduced in Sections 1 and 4.1.

In this section, results of the IWLRLS algorithm are performed for the Tweedie, Multinomial, and Poisson models.

Based on the additional experiments in Sections B.2-B.6 we train the Tweedie model with p = 1.25 (Section B.3) and for all models include weight truncation to penalize large co-occurrences (Section B.4), regularization terms (outlined in Section B.5), and include only a single bias term within the systematic component of the Tweedie model (Section B.6).Step To demonstrate the effectiveness of performing multiple iterations of the IWLRLS algorithm, we present results for the one-step estimator, an early-stopped estimator, and the full-likelihood estimator.

Of particular interest in our results are the Tweedie one-step estimator (a variant of the GloVe method), and the full-likelihood estimator for the Multinomial (a variant of the Skip-gram method).

For the results in TAB1 , the full-likelihood result is taken to be the iteration which achieves the maximum total accuracy on the analogy task, and the early-stop algorithm is taken to be an iteration between the one-step and full-likelihood iterations which performs best in total accuracy on the analogy task.

For both the Tweedie and Multinomials, the full-likelihood result is the result after 3 iterations and the early-stopped result is the result after 2 iterations.

For the Poisson model, the full-likelihood result is the result after 9 iterations, and the early-stopped result is the result after 3 iterations.

We find a small difference in total accuracy on the analogy task with the one-step estimator (GloVe) and the full-likelihood differing by roughly 1%.

We find a similar relationship in the Poisson estimator and further note that the early-stopped estimator for the Poisson has very similar accuracy to the full-likelihood algorithm.

Finally, the Multinomial model yields a difference of 2% between the full-likelihood algorithm (Skip-gram) and the one-step algorithm.

The early-stopped algorithm for the Multinomial also performs 1% higher than the one-step algorithm indicating a fair tradeoff between running an additional iteration and stopping after only one iteration.

We present a general model-based methodology for finding word vectors from a corpus.

This methodology involves choosing the distribution of a chosen co-occurrence matrix to be an exponential dispersion family and choosing the number of iterations to run our algorithm.

In Table 1 , we see that our methodology unifies the dominant word embedding methods available in the literature and provides new and improved methods.

We introduce an extension of Skip-gram that is stopped before full-convergence analagously to GloVe and an extension to GloVe beyond one iteration.

Experimental results on a small corpus demonstrate our method improves upon GloVe and Skip-gram on the Google word analogy similarity task.

It is our hope that this methodology can lead to the development of better, more statistically sound, word embeddings and consequently improve results on many other downstream tasks.

Further background in exponential dispersion families and generalized linear models is developed here.

We begin by discussing exponential dispersion families, the distribution of the response in generalized linear models.

Definition 2 Let y ∈ R be a random variable.

If the density function f (y; θ, ϕ) of y satisfies DISPLAYFORM0 over its support, then the distribution of y is in the exponential dispersion family.

The parameter θ is the natural parameter, ϕ is the dispersion parameter, and the function ψ is the cumulant generating function.

In many cases, the function δ(ϕ) is very simple, meaning that, for instance, δ(ϕ) = 1 or δ(ϕ) = ϕ.The function c(y; ϕ) can be viewed as the normalizing constant ensuring that the density integrates to one.

When y follows a distribution in the exponential dispersion family with natural parameter θ, its mean µ = ψ (θ), so we can equivalently specify the mean µ or the natural parameter θ.

Many classical distributions such as the Poisson, Normal, Binomial, and Gamma distribution are exponential dispersion families.

For example, when y ∼ Normal(µ, σ 2 ) is a normal distribution with mean µ and variance σ 2 , its log density satisfies DISPLAYFORM1 showing that here the natural parameter θ = µ, the dispersion parameter ϕ = σ 2 , the functions ψ(θ) = 1 2 θ 2 , δ(ϕ) = ϕ, and c(y; ϕ) = 1 2 log(2πσ 2 ) + y 2 2σ 2 .

The Tweedie distribution BID15 , of particular importantance to us, also lies within the exponential dispersion family.

Instead of defining the Tweedie distribution through the form of its density, we will define it through the relationship between its mean and variance.

This relies on a result from [Jørgensen, 1987, Theorem 1] that distributions within the exponential dispersion family are defined by the relationship between their mean and variance.

Definition 3 A random variable y has a Tweedie distribution with power parameter p ∈ {0} ∪ [1, ∞) when DISPLAYFORM2 and the distribution of y is an exponential dispersion family.

In this case, we write y ∼ Tweedie p (µ, ϕ), where µ = E(y) is the mean.

The Normal distribution discussed above has a variance that does not depend on the mean.

In our new notation, this means that the Normal distribution is a Tweedie distribution with power parameter p = 0.

The Poisson distribution has variance equal to the mean and is in the exponential dispersion family, so is a Tweedie distribution with power parameter p = 1 and dispersion parameter ϕ = 1.

A Gamma distribution with shape parameter α and rate parameter β is a Tweedie distribution with power p = 2, mean µ = α β , and dispersion parameter ϕ = α −1 .We will only consider Tweedie distributions with power parameter p ∈ (1, 2).

These distributions are also known as compound Poisson-Gamma distributions due to the representation DISPLAYFORM3 where n ∼ Poisson(λ) and g i DISPLAYFORM4 ∼ Gamma(α, β), and λ = BID15 .

It is important to note that the Tweedie distribution has positive mass at zero, an important characteristic for capturing the zero-inflation prominent in some co-occurence matrices due to some words never appearing within the same context.

Specifically, DISPLAYFORM5 Using other arguments related to representations of the mean and variance in terms of the cumulant generating function ψ, BID15 show that the Tweedie distribution has ψ(θ) DISPLAYFORM6

Exponential dispersion families are defined over real numbers.

Now, we generalize their definition to a multivariate setting.

Definition 4 Let y y y ∈ R m be a random vector.

If the density function f (y y y; θ θ θ, ϕ) of y y y satisfies log f (y y y; θ θ θ, ϕ) = y y y DISPLAYFORM0 over its support, then the distribution of y y y is in the multivariate exponential dispersion family.

The parameter θ θ θ ∈ R m is the natural parameter, ϕ ∈ R is the dispersion parameter, and the function ψ : R m → R is the cumulant generating function.

A collection of independent draws from the same exponential dispersion family is a multivariate exponential dispersion family.

To see this, let y i (i = 1, . . .

, m) be i.i.d.

from an exponential dispersion family.

Then, the density of y y y satisfies log f (y y y; θ θ θ, ϕ) = DISPLAYFORM1 + c(y j ; ϕ), which has cumulant generation function ψ(θ θ θ) = m j=1 ψ(θ j ).

Another useful example of a multivariate exponential dispersion family is the Multinomial.

Let x x x ∈ R c have be distributed as x x x ∼ multinomial(s, π π π), where s ∈ N is the total number of draws and π π π ∈ R c is the probability vector.

Introduce a change of parameters where DISPLAYFORM2 showing that the multinomial distribution is in the multivariate exponential dispersion family with ψ(θ θ θ) = s log ( c k=1 exp θ k ) .

We start by reviewing the linear model.

Given a response y y y ∈ R n comprising n observations, the model for y y y is a linear model with covariates x x x i ∈ R p when y i ind.∼ Normal(x x x T i β β β, σ 2 ) for all i ∈ {1, . . . , n}. In vector notation, this reads that y y y ∼ Normal(Xβ β β, σ 2 I), where X ∈ R n×p is a matrix with i th row x x x T i .

This is one of the more primitive models in the statistical modeling toolbox and isn't always appropriate for the data.

Generalized linear models remove the the assumptions of normality and that the mean is a linear function of the coefficients β β β.

Definition 5 For some exponential dispersion family ED(µ, ϕ) with mean parameter µ and dispersion parameter ϕ, the model for y y y ∈ R n is a generalized linear model with link function g when DISPLAYFORM0 DISPLAYFORM1 In the first line of the displayed relationships, the distribution of the response y y y is described.

In the third line, the systematic component η i expresses the effect of the covariates x x x i .

The second line connects the distribution to the covariates through the link function.

That is, the covariates effect a link-transformed mean.

The canonical link (b ) −1 is often chosen as the link function, due to its computational and mathematical properties BID0 .

Other times, the canonical link is inappropriate and there are alternative default choices.

Generalized linear models are used as the default modeling framework in many fields of applied science for non-normal distributions [McCullagh and Nelder, 1989] .

When g(µ) = µ is the identity map and ED is the Normal distribution, the generalized linear model is simply the linear model.

When g(µ) = logit(µ) = log 1−µ µ and ED is the Binomial distribution, the generalized linear model is logistic regression.

Further, a generalized linear model can be viewed as a no-hidden-layer neural network with activation function g.

We train our models on the English text8 corpus 6 with approximately 17 million tokens.

We filter out word types that occur fewer than 50 times to obtain a vocabulary size of approximately 11, 000; a ratio consistent with other embedding literature experiments 7 .The adjustable model configurations in IWLRLS are the choice of power parameter p, penalty tuning parameter λ, and co-occurrence processing step.

We experiment with different choices of p ∈ {1.1, 1.25, 1.5, 1.75, 1.9}, different choices of processing including no processing, clamping the weights (as in GloVe) and truncating the outliers in the co-occurrence matrix (elaborated on in Section B.4, and set the penalty tuning parameter λ = 0.002.

The estimated word vectors are the rows of DISPLAYFORM0 .

For all of our experiments, we set the dimension of the word vectors to d = 150, and the objective function at each iteration is optimized using Adagrad BID5 with a fixed learning rate of 0.1 8 .

Models are trained for up to 50 epochs (50 passes through the co-occurrence matrix) with batches of size 512.

We evaluate the impact of multiple iterations of the IWLRLS algorithm on all models, but examine different additions to the model only when p = 1.25.

We believe the impact of these changes will be present however for any value of p.

We present results of multiple iterations of our IWLRLS algorithm with different distributions.

In particular, we perform multiple iterations of the IWLRLS algorithm with Tweedie distribution and weight truncation to match the GloVe objective function and processing by setting the weight function in our model from h(x) = x 2−p to h(x) = (min{x, x max }) .75 with x max = 10 and p = 1.25.

We also presents results for an early-stopped version of skip-gram, and the new Poisson estimator.

The results are summarized in FIG4 .

We remark on a few observations based on these results.

First, as the number of steps increases, the accuracy on the analogy task increases for the first few iterations.

Second, relatively few steps are needed with the accuracy of Tweedie model performing best at the first and second steps of the algorithm, and the Multinomial and model performing best in steps 3-5 but with very similar performance at earlier steps.

The Poisson model performs best after 9 iteration, however performs nearly identically to the result of an early stopped algorithm at 3 iterations.

In conclusion, we find that early-stopped and one-step versions of the algorithm can perform comparably to full-likelihood methods.

In this section, we examine the effect of the choice of the power p in the tuning parameter when you run a Tweedie generalized low rank model.

Table 3 : Results for multiple choices of p for one and two iterations.

The Results in Table 3 show that values of p which are high perform poorly, while values of p below 1.5 perform similarly.

We find that p = 1.25 performs the best, and view this value of p as a good choice as it accounts for zero-inflation present in the co-occurence X. This also agrees with the results of [Pennington et al., 2014] and BID16 ].An even more interesting and perhaps more appropriate way to estimate the power p of the Tweedie distribution is in a data-driven and model-based way.

This approach is taken in BID16 .

In future work, we plan to use an improved estimating equation relative to BID16 ] to estimate p as part of the algorithm.

This would be modeling the marginal distribution of the co-occurences as being Tweedie with the same power.

Under a similar assumption, modified likelihood calculations are tractable and so are another possibility.

We plan to explore this in future work.

We set p = 1.25 in our algorithm with Tweedie distribution, and explore the effect of different strategies in handling large entries in the co-occurrence matrix X. One strategy is to simply input X into step 3 of our method.

A second strategy is to clamp the weight h(·) that results from step 3 of our method by taking h(x) = (min{x, x max }).75 as in GloVe.

A third strategy is to input min{x, x max } for each entry of the matrix X, where x max = 10, into step 3 of our method 9 .We find that no adjustment to the weights and GloVe's method of weight truncation both perform similarly with weight truncation slightly outperforming no adjustment.

We suspect a more significant improvement will show with larger corpora such as a full Wikipedia corpus.

Alternative approaches to alleviating the problem of large co-occurences are to use a more robust distribution or link function.

Indeed, the weight truncation in GloVe can be directly mimicked by Table 4 : Results for multiple choices of regularizing the large values of the co-occurrence matrix.

Our strategies are (1) harmonic matrix, (2) truncation of the weight only, (3) truncation of the co-occurrence matrix to x max = 10.either altering the distribution or the link function.

The desired form can be found via the weight and pseudo-response equations in algorithm 1.

We leave this to future work.

Table 5 : Results for including the penalty term in Equation FORMULA0 and not including the diagonal terms.

We consider regularizing the word vectors by including the penalty DISPLAYFORM0 with λ = .002 for two reasons.

One is to reduce noise in the estimation of the word vectors.

Udell et al. [2016, Lemma 7.3] show that penalizing by (16) is equivalent to penalizing by λ U V T * , the nuclear norm of U V T .

Since penalizing the nuclear norm U V T shrinks the dimension of the embedding and larger dimensional embeddings tend to be better [Melamud et al., 2016] , we choose a small tuning parameter to reduce noise while still preserving the dimension.

Another reason is to symmetrically distribute the singular values ofÛV T to both matricesÛ andV .Write the singular value decompositionÛV T = U ΣV T , for U and V orthogonal and Σ diagonal.

Mu et al. [2018, Theorem 1] shows that using penalty (16) results in havingÛ = U Σ 1/2 Q and V = V Σ 1/2 Q for some orthogonal matrix Q. This is desirable since it was argued empirically by Levy et al. [2015] that a symmetric distribution of singular values works optimally on semantic tasks.

Finally, we experiment with whether the penalty introduced in Equation FORMULA0 improves results and accurately reduces noise in the estimate.

We also consider not including the diagonal elements of X as a form of regularization and experiment here as well, as these terms are often large (can be considered as outliers) and do not contain a great deal of linguistic information.

Table 5 demonstrates the included regularization within the IWLRLS algorithm with Tweedie distribution and p = 1.25 improves results.

In Experiment 1, we found that the Multinomial model outperforms the Poisson model, although the Poisson model has an additional bias term to model context word frequencies.

This result was fairly counterintuitive, so we additionally experiment with having only a single bias term a i in the Tweedie model as in the Multinomial model.

We find overall that the Tweedie model with a systematic component without the bias term b j performs slightly better than the Tweedie model with systematic component containing both bias terms a i and b j .

We hope to further study the impact of bias terms and other systematic components in future work.

Table 6 : Results for including the bias term on the context word b j in addition to a i .

<|TLDR|>

@highlight

We present a novel iterative algorithm based on generalized low rank models for computing and interpreting word embedding models.