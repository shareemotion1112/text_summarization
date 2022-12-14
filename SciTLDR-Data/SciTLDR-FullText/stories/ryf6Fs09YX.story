Within many machine learning algorithms, a fundamental problem concerns efficient calculation of an unbiased gradient wrt parameters $\boldsymbol{\gamma}$ for expectation-based objectives $\mathbb{E}_{q_{\boldsymbol{\gamma}} (\boldsymbol{y})} [f (\boldsymbol{y}) ]$.

Most existing methods either ($i$) suffer from high variance, seeking help from (often) complicated variance-reduction techniques; or ($ii$) they only apply to reparameterizable continuous random variables and employ a reparameterization trick.

To address these limitations, we propose a General and One-sample (GO) gradient that ($i$) applies to many distributions associated with non-reparameterizable continuous {\em or} discrete random variables, and ($ii$) has the same low-variance as the reparameterization trick.

We find that the GO gradient often works well in practice based on only one Monte Carlo sample (although one can of course use more samples if desired).

Alongside the GO gradient, we develop a means of propagating the chain rule through distributions, yielding statistical back-propagation, coupling neural networks to common random variables.

Neural networks, typically trained using back-propagation for parameter optimization, have recently demonstrated significant success across a wide range of applications.

There has been interest in coupling neural networks with random variables, so as to embrace greater descriptive capacity.

Recent examples of this include black-box variational inference (BBVI) BID17 BID33 BID29 BID11 BID32 BID31 Zhang et al., 2018) and generative adversarial networks (GANs) BID8 BID28 Zhao et al., 2016; BID1 BID20 .

Unfortunately, efficiently backpropagating gradients through general distributions (random variables) remains a bottleneck.

Most current methodology focuses on distributions with continuous random variables, for which the reparameterization trick may be readily applied BID17 BID9 .As an example, the aforementioned bottleneck greatly constrains the applicability of BBVI, by limiting variational approximations to reparameterizable distributions.

This limitation excludes discrete random variables and many types of continuous ones.

From the perspective of GAN, the need to employ reparameterization has constrained most applications to continuous observations.

There are many forms of data that are more-naturally discrete.

The fundamental problem associated with the aforementioned challenges is the need to efficiently calculate an unbiased low-variance gradient wrt parameters ?? for an expectation objective of the form E q?? (y) [f (y)]1 .

We are interested in general distributions q ?? (y), for which the components of y may be either continuous or discrete.

Typically the components of y have a hierarchical structure, and a subset of the components of y play a role in evaluating f (y).Unfortunately, classical methods for estimating gradients of E q?? (y) [f (y)] wrt ?? have limitations.

The REINFORCE gradient (Williams, 1992) , although generally applicable (e.g., for continuous and discrete random variables), exhibits high variance with Monte Carlo (MC) estimation of the expectation, forcing one to apply additional variance-reduction techniques.

The reparameterization trick (Rep) BID38 BID17 BID33 works well, with as few as only one MC sample, but it is limited to continuous reparameterizable y. Many efforts have been devoted to improving these two formulations, as detailed in Section 6.

However, none of these methods is characterized by generalization (applicable to general distributions) and efficiency (working well with as few as one MC sample).The key contributions of this work are based on the recognition that REINFORCE and Rep are seeking to solve the same objective, but in practice Rep yields lower-variance estimations, albeit for a narrower class of distributions.

Recent work BID32 has made a connection between REINFORCE and Rep, recognizing that the former estimates a term the latter evaluates analytically.

The high variance by which REINFORCE approximates this term manifests high variance in the gradient estimation.

Extending these ideas, we make the following main contributions.

(i) We propose a new General and One-sample (GO) gradient in Section 3, that principally generalizes Rep to many non-reparameterizable distributions and justifies two recent methods BID5 BID15 ; the "One sample" motivating the name GO is meant to highlight the low variance of the proposed method, although of course one may use more than one sample if desired. (ii) We find that the core of the GO gradient is something we term a variable-nabla, which can be interpreted as the gradient of a random variable wrt a parameter. (iii) Utilizing variablenablas to propagate the chain rule through distributions, we broaden the applicability of the GO gradient in Sections 4-5 and present statistical back-propagation, a statistical generalization of classic back-propagation BID36 .

Through this generalization, we may couple neural networks to general random variables, and compute needed gradients with low variance.

To motivate this paper, we begin by briefly elucidating common machine learning problems for which there is a need to efficiently estimate gradients of ?? for functions of the form E q?? (y) [f (y)].

Assume access to data samples {x i } i=1,N , drawn i.i.d.

from the true (and unknown) underlying distribution q(x).

We seek to learn a model p ?? (x) to approximate q(x).

A classic approach to such learning is to maximize the expected log likelihood?? = argmax ?? E q(x) [log p ?? (x)], perhaps with an added regularization term on ??.

Expectation E q(x) (??) is approximated via the available data samples, a?? ?? = argmax ?? 1 N N i=1 log p ?? (x i ).

It is often convenient to employ a model with latent variables z, i.e., p ?? (x) = p ?? (x, z)dz = p ?? (x|z)p(z)dz, with prior p(z) on z. The integral wrt z is typically intractable, motivating introduction of the approximate posterior q ?? (z|x), with parameters ??.

The well-known evidence lower bound (ELBO) BID16 BID2 ) is defined as ELBO(??, ??; x) = E q ?? (z|x) [log p ?? (x, z) ??? log q ?? (z|x)](1) DISPLAYFORM0 where p ?? (z|x) is the true posterior, and KL(?? ??) represents the Kullback-Leibler divergence.

Variational learning seeks (??,??) = argmax ??,?? N i=1 ELBO(??, ??; x i ).

While computation of the ELBO has been considered for many years, a problem introduced recently concerns adversarial learning of p ?? (x), or, more precisely, learning a model that allows one to efficiently and accurately draw samples x ??? p ?? (x) that are similar to x ??? q(x).

With generative adversarial networks (GANs) BID8 , one seeks to solve DISPLAYFORM1 where D ?? (x) is a discriminator with parameters ??, quantifying the probability x was drawn from q(x), with 1 ??? D ?? (x) representing the probability that it was drawn from p ?? (x).

There have been many recent extensions of GAN BID28 Zhao et al., 2016; BID1 BID20 , but the basic setup in (3) holds for most.

To optimize (1) and (3), the most challenging gradients that must be computed are of the form ??? ?? E q?? (y) [f (y)]; for (1) y = z and ?? = ??, while for (3) y = x and ?? = ??.

The need to evaluate expressions like ??? ?? E q?? (y) [f (y)] arises in many other machine learning problems, and consequently it has generated much prior attention.

The reparameterization trick (Rep) is limited to reparameterizable random variables y with continuous components.

There are situations for which Rep is not readily applicable, e.g., where the components of y may be discrete or nonnegative Gamma distributed.

We seek to gain insights from the relationship between REINFORCE and Rep, and generalize the types of random variables y for which the latter approach may be effected.

We term our proposed approach a General and One-sample (GO) gradient.

In practice, we find that this approach works well with as few as one sample for evaluating the expectation, and it is applicable to more general settings than Rep.

Recall that Rep was first applied within the context of variational learning BID17 , as in (1).

Specifically, it was assumed q ?? (y) = v q ?? (y v ), omitting explicit dependence on data x, for notational convenience; y v is component v of y. In BID17 DISPLAYFORM0 , with mean ?? v (??) and variance ?? 2 v (??).

In the following we generalize q ?? (y v ) such that it need not be Gaussian.

Applying integration by parts BID32 ) DISPLAYFORM1 where y ???v denotes y with y v excluded, and Q ?? (y v ) is the cumulative distribution function (CDF) of q ?? (y v ).

The "0" term is readily proven to be zero for any Q ?? (y v ), with the assumption that f (y) doesn't tend to infinity faster than ??? ?? Q ?? (y v ) tending to zero when y v ??? ?????.The "Key" term exactly recovers the one-dimensional Rep when reparameterization BID32 .

Further, applying ??? ?? q ?? (y v ) = q ?? (y v )??? ?? log q ?? (y v ) in (6) yields REINFORCE.

Consequently, it appears that Rep yields low variance by analytically setting to zero the unnecessary but high-variance-injecting "0" term, while in contrast REINFORCE implicitly seeks to numerically implement both terms in (7).

DISPLAYFORM2 We generalize q ?? (y) for discrete y v , here assuming y v ??? {0, 1, . . .

, ???}. It is shown in Appendix A.2 that this framework is also applicable to discrete y v with a finite alphabet.

It may be shown (see Appendix A.2) that DISPLAYFORM3 "Key" (8) where Q ?? (y v ) = yv n=0 q ?? (n), and Q ?? (???) = 1 for all ??.

Theorem 1 (GO Gradient).

For expectation objectives E q?? (y) [f (y)], where q ?? (y) satisfies (i) q ?? (y) = v q ?? (y v ); (ii) the corresponding CDF Q ?? (y v ) is differentiable wrt parameters ??; and (iii) one can calculate ??? ?? Q ?? (y v ), the General and One-sample (GO) gradient is defined as DISPLAYFORM4 where DISPLAYFORM5 T , and DISPLAYFORM6 All proofs are provided in Appendix A, where we also list g q?? (yv) ?? for a wide selection of possible q ?? (y), for both continuous and discrete y. Note for the special case with continuous y, GO reduces to Implicit Rep gradients BID5 and pathwise derivatives BID15 ; in other words, GO provides a principled explanation for their low variance, namely their foundation (implicit differentiation) originates from integration by parts.

For high-dimensional discrete y, calculating D y [f (y)] is computationally expensive.

Fortunately, for f (y) often used in practice special properties hold that can be exploited for efficient parallel computing.

Also for discrete y v with finite support, it is possible that one could analytically evaluate a part of expectations in (9) for lower variance, mimicking the local idea in BID42 ; BID41 .

Appendix I shows an example illustrating how to handle these two issues in practice.

The GO gradient in Theorem 1 can only handle single-layer mean-field q ?? (y), characterized by an independence assumption on the components of y. One may enlarge the descriptive capability of q ?? (y) by modeling it as a marginal distribution of a deep model BID32 BID2 .

Hereafter, we focus on this situation, and begin with a 2-layer model for simple demonstration.

Specifically, consider DISPLAYFORM0 where ?? = {?? y , ?? ?? }, y is the leaf variable, and the internal variable ?? is assumed to be continuous.

Components of y are assumed to be conditionally independent given ??, but upon marginalizing out ?? this independence is removed.

, and via Theorem 1 DISPLAYFORM0 Lemma 1.

Equation (10) exactly recovers the Rep gradient in (5), if ?? = ?? y and q ?? (y) has reparameterization y = ?? ?? (??), ?? ??? q(??) for differentiable ?? ?? (??) and easily sampled q(??).Lemma 1 shows that Rep is a special case of our deep GO gradient in the following Theorem 2.

Note neither Implicit Rep gradients nor pathwise derivatives can recover Rep in general, because a neural-network-parameterized y = ?? ?? (??) may lead to non-trivial CDF Q ?? (y).For the gradient wrt ?? ?? , we first apply Theorem 1, yielding DISPLAYFORM1 For continuous internal variable ?? one can apply Theorem 1 again, from which DISPLAYFORM2 Now extending the same procedure to deeper models with L layers, we generalize the GO gradient in Theorem 2.

Random variable y (L) is assumed to be the leaf variable of interest, and may be continuous or discrete; latent/internal random variables {y (1) , . . .

, y (L???1) } are assumed continuous (these generalize ?? from above).

DISPLAYFORM3 v |y (l???1) ), and one has access to variable-nablas g DISPLAYFORM4 and g DISPLAYFORM5 , as defined in Theorem 1, the General and One-sample (GO) gradient is defined as DISPLAYFORM6 where DISPLAYFORM7 Corollary 1.

The deep GO gradient in Theorem 2 exactly recovers back-propagation BID36 when each element distribution q ?? (l) (y DISPLAYFORM8 v |y (l???1) ) is specified as the Dirac delta function located at the activated value after activation function.

BID36 with the deep GO gradient in Theorem 2.

(i) In deterministic deep neural networks, one forward-propagates information using activation functions, like ReLU, to sequentially activate {y (l) } l=1,?????? ,L (black solid arrows), and then back-propagates gradients from Loss f (??) to each parameter ?? (l) via gradient-flow through {y (k) } k=L,?????? ,l (red dashed arrows). (ii) Similarly for the deep GO gradient with 1 MC sample, one forward-propagates information to calculate the expected loss function f (y (L) ) using distributions as statistical activation functions, and then uses variable-nablas to sequentially back-propagate gradients through random variables {y (k) } k=L,?????? ,l to each ?? (l) , as in (12).

Recall the motivating discussion in Section 2, in which we considered generative model p ?? (x, z) and inference model q ?? (z|x), the former used to model synthesis of the observed data x and the latter used for inference of z given observed x. In recent deep architectures, a hierarchical representation for cumulative latent variables z = (z (1) , . . .

, z (L) ) has been considered BID33 BID30 Zhou et al., 2015; BID32 BID3 Zhang et al., 2018) .

As an example, there are models with DISPLAYFORM0 ).

When performing inference for such models, it is intuitive to consider first-order Markov chain structure for DISPLAYFORM1 ).

The discussion in this section is most relevant for variational inference, for computation of DISPLAYFORM2 , and consequently we specialize to that notation in the subsequent discussion (we consider representations in terms of z, rather than the more general y notation employed in Section 4).Before proceeding, we seek to make clear the distinction between this section and Section 4.

In the latter, only the leaf variable DISPLAYFORM3 Published as a conference paper at ICLR 2019 is because in Section 4 the underlying model is a marginal distribution of z (L) , i.e., q ?? (z (L) ), which is relevant to the generators of GANs; see (3), with DISPLAYFORM4 were added there to enhance the modeling flexibility of q ?? (z (L) ).

In this section, the deep set of random variables z = (z (1) , . . .

, z (L) ) are inherent components of the underlying generative model for x, i.e., p ?? ( DISPLAYFORM5 Hence, all components of z manifested via inference model q ?? (z|x) = q ?? (z (1) , . . .

, z (L) |x) play a role in f (z).

Besides, no specific structure is imposed on p ?? (x, z) and q ?? (z|x) in this section, moving beyond the aforementioned first-order Markov structure.

For a practical application, one may employ domain knowledge to design suitable graphical models for p ?? (x, z) and q ?? (z|x), and then use the following Theorem 3 for training.

Theorem 3 (Statistical Back-Propagation).

For expectation objectives DISPLAYFORM6 where DISPLAYFORM7 represents L ??? I continuous or discrete leaf variables with no children except f (??), and q ?? (??) is constructed as a hierarchical probabilistic graphical model DISPLAYFORM8 with each element distribution q ?? (z v |pa(z v )) having accessible variable-nablas as defined in Theorem 1, pa(z v ) denotes the parent variables of z v , the General and One-sample (GO) gradient for ?? k ??? ?? is defined as DISPLAYFORM9 where ch(?? k ) denotes the children variables of ?? k , and with DISPLAYFORM10 and DISPLAYFORM11 Statistical back-propagation in Theorem 3 is relevant to hierarchical variational inference (HVI) BID32 BID13 BID23 ) (see Appendix G), greatly generalizing GO gradients to the inference of directed acyclic probabilistic graphical models.

In HVI variational distributions are specified as hierarchical graphical models constructed by neural networks.

Using statistical back-propagation, one may rely on GO gradients to perform HVI with low variance, while greatly broadening modeling flexibility.

There are many methods directed toward low-variance gradients for expectation-based objectives.

Attracted by the generalization of REINFORCE, many works try to improve its performance via efficient variance-reduction techniques, like control variants BID23 BID42 BID10 BID24 Tucker et al., 2017; BID9 or via data augmentation and permutation techniques (Yin & Zhou, 2018) .

Most of this research focuses on discrete random variables, likely because Rep (if it exists) works well for continuous random variables but it may not exist for discrete random variables.

Other efforts are devoted to continuously relaxing discrete variables, to combine both REINFORCE and Rep for variance reduction BID14 BID22 Tucker et al., 2017; BID9 .Inspired by the low variance of Rep, there are methods that try to generalize its scope.

The Generalized Rep (GRep) gradient employs an approximate reparameterization whose transformed distribution weakly depends on the parameters of interest.

Rejection sampling variational inference (RSVI) BID25 exploits highly-tuned transformations in mature rejection sampling simulation to better approximate Rep for non-reparameterizable distributions.

Compared to the aforementioned methods, the proposed GO gradient, containing Rep as a special case for continuous random variables, applies to both continuous and discrete random variables with the same low-variance as the Rep gradient.

Implicit Rep gradients BID5 and pathwise derivatives BID15 are recent low-variance methods that exploit the gradient of the expected function; they are special cases of GO in the single-layer continuous settings.

The idea of gradient backpropagation through random variables has been exploited before.

RE-LAX BID9 , employing neural-network-parametrized control variants to assist REINFORCE for that goal, has a variance potentially as low as the Rep gradient.

SCG BID40 utilizes the generalizability of REINFORCE to construct widely-applicable stochastic computation graphs.

However, REINFORCE is known to have high variance, especially for highdimensional problems, where the proposed methods are preferable when applicable BID40 .

Stochastic back-propagation BID33 BID4 , focusing mainly on reparameterizable Gaussian random variables and deep latent Gaussian models, exploits the product rule for an integral to derive gradient backpropagation through several continuous random variables.

By comparison, the proposed statistical back-propagation based on the GO gradient is applicable to most distributions for continuous random variables.

Further, it also flexibly generalizes to hierarchical probabilistic graphical models with continuous internal variables and continuous/discrete leaf ones.

We examine the proposed GO gradients and statistical back-propagation with four experiments: (i) simple one-dimensional (gamma and negative binomial) examples are presented to verify the GO gradient in Theorem 1, corresponding to nonnegative and discrete random variables; (ii) the discrete variational autoencoder experiment from Tucker et al. FORMULA0 and BID9 is reproduced to compare GO with the state-of-the-art variance-reduction methods; (iii) a multinomial GAN, generating discrete observations, is constructed to demonstrate the deep GO gradient in Theorem 2; (iv) hierarchical variational inference (HVI) for two deep non-conjugate Bayesian models is developed to verify statistical back-propagation in Theorem 3.

Note the experiments of BID5 and BID15 additionally support our GO in the single-layer continuous settings.

Many mature machine learning frameworks, like TensorFlow (Abadi et al.) and PyTorch BID27 , are optimized for implementation of methods like back-propagation.

Fortunately, all gradient calculations in the proposed theorems obey the chain rule in expectation, enabling convenient incorporation of the proposed approaches into existing frameworks.

Experiments presented below were implemented in TensorFlow or PyTorch with a Titan Xp GPU.

Code for all experiments can be found at github.com/YulaiCong/GOgradient.Notation Gam(??, ??) denotes the gamma distribution with shape ?? and rate ??, NB(r, P ) the negative binomial distribution with number of failures r and success probability P , Bern(P ) the Bernoulli distribution with probability P , Mult(n, P ) the multinomial distribution with number of trials n and event probabilities P , Pois(??) the Poisson distribution with rate ??, and Dir(??) the Dirichlet distribution with concentration parameters ??.

We first consider illustrative one-dimensional "toy" problems, to examine the GO gradient for both continuous and discrete random variables.

The optimization objective is expressed as DISPLAYFORM0 where for continuous z we assume p(z|x) = Gam(z; ?? 0 , ?? 0 ) for given set (?? 0 , ?? 0 ), with q ?? (z) = Gam(z; ??, ??) and ?? = {??, ??}; for discrete z we assume p(z|x) = NB(z; r 0 , p 0 ) for given set (r 0 , p 0 ), with q ?? (z) = NB(z; r, p) and ?? = {r, p}. Stochastic gradient ascent with onesample-estimated gradients is used to optimize the objective, which is equivalent to minimizing KL(q ?? (z) p(z|x)).

RSVI BID25 , and their modified version using the "sticking" idea , denoted as GRep-Stick and RSVI-Stick, respectively.

For RSVI and RSVI-Stick, the shape augmentation parameter is set as 5 by default.

The only difference between GRep and GRep-Stick (also RSVI and RSVI-Stick) is the latter does not analytically express the entropy .

GO empirically provides more stable learning curves, as shown in FIG2 (c).

For the discrete case corresponding to the NB distribution, GO is compared to REINFORCE (Williams, 1992) .

To address the concern about comparison with the same number of evaluations of the expected function 3 , another curve of REINFORCE using 2 samples is also added, termed REINFORCE2.

It is apparent from FIG2 (d) that, thanks to analytically removing the "0" terms in (8), the GO gradient has much lower variance, even in this simple one-dimensional case.

DISPLAYFORM1

To demonstrate the low variance of the proposed GO gradient, we consider the discrete variational autoencoder (VAE) experiment from REBAR (Tucker et al., 2017) and RELAX BID9 , to make a direct comparison with state-of-the-art variance-reduction methods.

Since the statistical back-propagation in Theorem 3 cannot handle discrete internal variables, we focus on the single-latent-layer settings (1 layer of 200 Bernoulli random variables), i.e., DISPLAYFORM0 where P z is the parameters of the prior p ?? (z), NN P x|z (z) means using a neural network to project the latent binary code z to the parameters P x|z of the likelihood p ?? (x|z), and NN P z|x (x) is similarly defined for q ?? (z|x).

The objective is given in (1).

See Appendix I for more details.

BID37 .

All methods are run with the same learning rate for 1, 000, 000 iterations.

The black line represents the best training ELBO of REBAR and RELAX.

ELBOs are calculated using all training/validation data.

Figure 3 (also Figure 9 of Appendix I) shows the training curves versus iteration and running time for the compared methods.

Even without any variance-reduction techniques, GO provides better performance, faster convergence rate, and better running efficiency (about ten times faster in achieving the best training ELBO of RERAR/RELAX in this experiment).

We believe GO's better performance originates from: (i) its inherent low-variance nature; (ii) GO has less parameters compared to REBAR and RELAX (no control variant is adopted for GO); (iii) efficient batch processing methods (see Appendix I) are adopted to benefit from parallel computing.

TAB0 presents the best training/validation ELBOs under various experimental settings for the compared methods.

GO provides the best performance in all situations.

Additional experimental results are given in Appendix I.Many variance-reduction techniques can be used to further reduce the variance of GO, especially when complicated models are of interest.

Compared to RELAX, GO cannot be directly applied when f (y) is not computable or where the interested model has discrete internal variables (like multilayer Sigmoid belief networks BID26 ).

For the latter issue, we present in Appendix B.4 a procedure to assist GO (or statistical back-propagation in Theorem 3) in handling discrete internal variables.

To demonstrate the deep GO gradient in Theorem 2, we adopt multinomial leaf variables x and construct a new multinomial GAN (denoted as MNGAN-GO) for generating discrete observations with a finite alphabet.

The corresponding generator p ?? (x) is expressed as DISPLAYFORM0 For brevity, we integrate the generator's parameters ?? into the NN notation, and do not explicitly express them.

Details for this example are provided in Appendix J.We compare MNGAN-GO with the recently proposed boundary-seeking GAN (BGAN) BID12 on 1-bit (1-state, Bernoulli leaf variables x), 1-bit (2-state), 2-bit (4-state), 3-bit (8-state) and 4-bit (16-state) discrete image generation tasks, using quantized MNIST datasets BID18 .

TAB1 presents inception scores BID39 of both methods.

MNGAN-GO performs better in general.

Further, with GO's assistance, MNGAN-GO shows more potential to benefit from richer information coming from more quantized states.

For demonstration, FIG4 shows the generated samples from the 4-bit experiment, where better image quality and higher diversity are observed for the samples from MNGAN-GO.

To demonstrate statistical back-propagation in Theorem 3, we design variational inference nets for two nonconjugate hierarchical Bayesian models, i.e., deep exponential families (DEF) BID30 and deep latent Dirichlet allocation (DLDA) (Zhou et al., 2015; BID3 .

DISPLAYFORM0 For demonstration, we design the inference nets q ?? (z|x) following the first-order Markov chain construction in Section 5, namely DISPLAYFORM1 DISPLAYFORM2 Further details are provided in Appendix K. One might also wish to design inference nets that have structure beyond the above first-order Markov chain construction, as in Zhang et al. FORMULA0 ; we do not consider that here, but Theorem 3 is applicable to that case.

DISPLAYFORM3 HVI for a 2-layer DEF is first performed, with the ELBO curves shown in Figure 5 .

GO enables faster and more stable convergence.

Figure 6 presents the HVI results for a 3-layer DLDA, for which stable ELBOs are again observed.

More importantly, with the GO gradient, one can utilize pure gradient-based methods to efficiently train such complicated nonconjugate models for meaningful dictionaries (see Appendix K for more implementary details).

For expectation-based objectives, we propose a General and One-sample (GO) gradient that applies to continuous and discrete random variables.

We further generalize the GO gradient to cases for which the underlying model is deep and has a marginal distribution corresponding to the latent variables of interest, and to cases for which the latent variables are hierarchical.

The GO-gradient setup is demonstrated to yield the same low-variance estimation as the reparameterization trick, which is only applicable to reparameterizable continuous random variables.

Alongside the GO gradient, we constitute a means of propagating the chain rule through distributions.

Accordingly, we present statistical back-propagation, to flexibly integrate deep neural networks with general classes of random variables.

A PROOF OF THEOREM 1We first prove (7) in the main manuscript, followed by its discrete counterpart, i.e., (8) in the main manuscript.

Then, it is easy to verify Theorem 1.A.1 PROOF OF EQUATION FORMULA3 IN THE MAIN MANUSCRIPT Similar proof in one-dimension is also given in the supplemental materials of BID32 .We want to calculate DISPLAYFORM0 where y ???v denotes y with y v excluded.

Without loss of generality, we assume y v ??? (??????, ???).

DISPLAYFORM1 , and we have DISPLAYFORM2 where DISPLAYFORM3 , we then apply integration by parts (or partial integration) to get DISPLAYFORM4 With Q ?? (???) = 1 and Q ?? (??????) = 0, it's straightforward to verify that the first term is always zero for any Q ?? (y v ), thus named the "0" term.

For discrete variables y, we have DISPLAYFORM0 where y v ??? {0, 1, ?? ?? ?? , N } and N is the size of the alphabet.

To handle the summation of products of two sequences and develop discrete counterpart of FORMULA3 , we first introduce Abel transformation.

Abel transformation.

Given two sequences {a n } and {b n }, with n ??? {0, ?? ?? ?? , N }, we define B 0 = b 0 and B n = n k=0 b n for n ??? 1.

Accordingly, we have DISPLAYFORM1 (a n+1 ??? a n )B n .Substituting n = y v , a n = f (y), b n = ??? ?? q ?? (y v ), and B n = ??? ?? Q ?? (y v ) into the above equation, we have DISPLAYFORM2 .

Note the first term equals zero for both finite alphabet, i.e., N < ???, and infinite alphabet, i.e., N = ???. When N = ???, we get (8) in the main manuscript.

With the above proofs for Eqs. FORMULA3 and (8) , one could straightforwardly verify Theorem 1.

DISPLAYFORM0 (1 ??? p) r p y (y + r)B(1 ??? p; r, y + 1) + y + r p y r 2 ??3F2(r, r, ???y; r + 1, r + 1; 1 ??? p) DISPLAYFORM1 DISPLAYFORM2 is the digamma function.

??(x, y) is the upper incomplete gamma function.

B(x; ??, ??) is the incomplete beta function.

pFq (a1, ?? ?? ?? , ap; b1, ?? ?? ?? , bp; x) is the generalized hypergeometric function.

T (m, s, x) is a special case of Meijer G-function BID7 .

For simpler demonstration, we first use a 2-layer model to show why discrete internal variables are challenging.

Then, we present an importance-sampling proposal that might be useful under specific situations.

Finally, we present a strategy to learn discrete internal variables with the statistical back-propagation in Theorem 3 of the main manuscript.

Assume q ?? (y) being the marginal distribution of the following 2-layer model DISPLAYFORM0 where ?? = {?? y , ?? ?? }, q ?? (y, ??) = q ?? y (y|??)q ?? ?? (??) = v q ?? y (y v |??) ?? k q ?? ?? (?? k ), and both the leaf variable y and the internal variable ?? could be either continuous or discrete.

Accordingly, the objective becomes DISPLAYFORM1 For gradient wrt ?? y , using Theorem 1, it is straight to show DISPLAYFORM2 For gradient wrt ?? ?? , we first have DISPLAYFORM3 Withf (??) = E q?? y (y|??) [f (y)], we then apply Theorem 1 and get DISPLAYFORM4 where DISPLAYFORM5 DISPLAYFORM6 Next, we separately discuss the situations where ?? k is continuous or discrete.

One can directly apply Theorem 1 again, namely DISPLAYFORM0 Substituting FORMULA3 into FORMULA37 , we have DISPLAYFORM1

In this case, we need to calculate DISPLAYFORM0 The keys are again partial integration and Abel transformation.

For simplicity, we first assume one-dimensional y, and separately discuss y being continuous and discrete.

For continuous y, we apply partial integration tof (??) and get DISPLAYFORM1 Accordingly, we have DISPLAYFORM2 Removing the "0" term, we have DISPLAYFORM3 For discrete y, by similarly exploiting Abel transformation, we hav?? DISPLAYFORM4 Accordingly, we get DISPLAYFORM5 Unifying FORMULA6 for continuous y and FORMULA0 for discrete y, we have DISPLAYFORM6 where we define??? DISPLAYFORM7 Multi-dimensional y. Based on the above one-dimensional foundation, we next move on to multidimensional situations.

With definitions y :i {y 1 , ?? ?? ?? , y i } and y i: {y i , ?? ?? ?? , y V }, where V is the dimensionality of y, we have DISPLAYFORM8 Apply FORMULA0 and we have DISPLAYFORM9 Similarly, we add extra terms to the above equation to enable applying (21) again as DISPLAYFORM10 Accordingly, we apply (21) to the first two terms and have DISPLAYFORM11 So forth, we summarize the pattern into the following equation as DISPLAYFORM12 where A ?? k (y, V ) is iteratively calculated as DISPLAYFORM13 Despite elegant structures within (23), to calculate it, one must iterate over all dimensions of y, which is computational expensive in practice.

More importantly, it is straightforward to show that deeper models will have similar but much more complicated expressions.

Next, we present an intuitive proposal that might be useful under specific situations.

The key idea is to use different extra items to enable "easy-to-use" expression for DISPLAYFORM0 Apply (21) to the adjacent two terms for V times, we have DISPLAYFORM1 where we can apply the idea of importance sampling and modify the above equation to DISPLAYFORM2 Note that importance sampling may not always work well in practice BID2 .We further define the generalized variable-nabla as DISPLAYFORM3 With the generalized variable-nabla, we unify (24) for discrete ?? k and FORMULA3 for continuous ?? k and get DISPLAYFORM4 which apparently obeys the chain rule.

Accordingly, we have the gradient for ?? ?? in FORMULA37 as DISPLAYFORM5 One can straightforwardly verify that, with the generalized variable-nabla defined in (25), the chain rule applies to DISPLAYFORM6 , where one can freely specify both leaf and internal variables to be either continuous or discrete.

The only problem is that, for discrete internal variables, the importance sampling trick used in (24) may not always work as expected.

Practically, if one has to deal with a q ?? (y (1) , ?? ?? ?? , y (L) ) with discrete internal variables y (l) , l < L, we suggest the strategy in Figure 7 , with which one should expect a close performance but enjoy much easier implementation with statistical back-propagation in Theorem 3 of the main manuscript.

In fact, one can always add additional continuous internal variables to the graphical models to remedy the performance loss or even boost the performance.(a) (b) Figure 7 : A strategy for discrete internal variables.

Blue and red circles denote continuous and discrete variables, respectively.

The centered dots represent the corresponding distribution parameters.

(a) Practically, one uses a neural network (black arrow) to connect the left variable to the parameters of the center discrete one, and then uses another neural network to propagate the sampled value to the next.

(b) Instead, we suggest "extracting" the discrete variable as a leaf one and propagate its parameters to the next.

C PROOF OF LEMMA 1First, a marginal distribution q ?? (y) with reparameterization y = ?? ?? ( ), ??? q( ) can be expressed as a joint distribution, namely DISPLAYFORM0 where q ?? (y| ) = ??(y ??? ?? ?? ( )), ??(??) is the Dirac delta function, and ?? ?? ( ) could be flexibly specified as a injective, surjective, or bijective function.

Next, we align notations and rewrite (10) as DISPLAYFORM1 where DISPLAYFORM2 Accordingly, we have DISPLAYFORM3 Substituting the above equations into (26), we get DISPLAYFORM4 which is the multi-dimensional Rep gradient in (5) of the main manuscript.

Firstly, with the internal variable ?? being continuous, FORMULA11 and FORMULA13 in the main manuscript are proved by FORMULA37 and FORMULA37 in Section B, respectively.

Then, by iteratively generalizing the similar derivations to deep models and utilizing the fact that the GO gradients with variable-nablas in expectation obey the chain rule for models with continuous internal variables, Theorem 2 could be readily verified.

When all q ?? (i) y (i) |y (i???1) s are specified as Dirac delta functions, namely DISPLAYFORM0 where ??(?? (i) , y (i???1) ) denotes the activated values after activation functions, the objective becomes DISPLAYFORM1 where DISPLAYFORM2 Back-Propagation.

For the objective in (28), the Back-Propagation is expressed as DISPLAYFORM3 where DISPLAYFORM4 Deep GO Gradient.

We consider the continuous special case, where DISPLAYFORM5 With q ?? (i+1) y (i+1) |y (i) s being Dirac delta functions, namely, DISPLAYFORM6 By substituting the above equation into (12) in Theorem 2, and then comparing it with(29), one can easily verify Corollary 1.

Based on the proofs for Theorem 1 and Theorem 2, it is clear that, if one constrains all internal variables to be continuous, the GO gradients in expectation obey the chain rule.

Therefore, one can straightforwardly utilizing the chain rule to verify Theorem 3.

Actually, Theorem 3 may be seen as the chain rule generalized with random variables, among which the internal ones are only allowed to be continuous.

In Hierarchical Variational Inference, the objective is to maximize the evidence lower Bound (ELBO) DISPLAYFORM0 For the common case with z = {z (1) , ?? ?? ?? , z (L) }, it is obvious that Theorem 3 of the main manuscript can be applied when optimizing ??.

Practically, there are situations where one might further put a latent variable ?? in reference q ?? (z|x), namely q ?? (z|x) = q ?? z (z|??)q ?? ?? (??)d?? with ?? = {?? z , ?? ?? }.

Following BID32 , we briefly discuss this situation here.

We first show that there is another unnecessary variance-injecting "0" term.

DISPLAYFORM1 where the second "0" term is straightly verified as DISPLAYFORM2 Eliminating the "0" term from (31), one still has another problem, that is, log q ?? (z|x) is usually non-trivial when q ?? (z|x) is marginal.

For this problem, we follow BID32 to use another lower bound ELBO2 of the ELBO in (30).

DISPLAYFORM3 where r ?? (??|z, x), evaluable, is an additional variational distribution to approximate the variational posterior q ?? (??|z, x).

Accordingly, we get the ELBO2 for Hierarchical Variational Inference as DISPLAYFORM4 Note similar to (31), the unnecessary "0" term related to log q ?? (z, ??|x) should also be removed.

Accordingly, we have DISPLAYFORM5 Obviously, Theorem 3 is readily applicable to provide GO gradients.

We first consider illustrative one-dimensional "toy" problems, to examine the GO gradient in Theorem 1 for both continuous and discrete random variables.

The optimization objective is expressed as max DISPLAYFORM0 where for continuous z we assume p(z|x) = Gam(z; ?? 0 , ?? 0 ) for set (?? 0 , ?? 0 ), with q ?? (z) = Gam(z; ??, ??) and ?? = {??, ??}; for discrete z we assume p(z|x) = NB(z; r 0 , p 0 ) for set (r 0 , p 0 ), with q ?? (z) = NB(z; r, p) and ?? = {r, p}. Stochastic gradient ascent with one-sample-estimated gradients is used to optimize the objective, which is equivalent to minimizing KL(q ?? (z) p(z|x)).

FIG8 shows the experimental results.

For the nonnegative continuous z associated with the gamma distribution, we compare our GO gradient with GRep , RSVI BID25 , and their modified version using the "sticking" idea , denoted as GRep-Stick and RSVI-Stick respectively.

For RSVI and RSVI-Stick, the shape augmentation parameter is set as 5 by default.

The only difference between GRep and GRep-Stick (also RSVI and RSVI-Stick) is the latter does NOT analytically express the entropy E q ?? (z) For the discrete case corresponding to the NB distribution, GO is compared to REINFORCE (Williams, 1992) .

To estimate gradient, REINFORCE uses 1 sample of z and 1 evaluation of the expected function; whereas GO uses 1 sample and 2 evaluations.

To address the concern about comparison with the same number of evaluations of the expected function, another curve of REINFORCE using 2 samples (thus 2 evaluations of the expected function) is also added, termed REINFORCE2.

It is apparent from Figures 8(g)-8(l) that, thanks to analytically removing the "0" terms, the GO gradient has much lower variance and thus faster convergence, even in this simple one-dimensional case.[

Complementing the discrete VAE experiment of the main manuscript, we present below its experimental settings, implementary details, and additional results.

Since the presented statistical back-propagation in Theorem 3 of the main manuscript cannot handle discrete internal variables, we focus on the single-latent-layer settings (1 layer of 200 Bernoulli random variables) for fairness, i.e., DISPLAYFORM0 Referring to the experimental settings in BID9 , we consider , that for the second parameter (gamma ?? or NB p), and the ELBO, respectively.

The first two rows correspond to the gamma toys with posterior parameters ??0 = 1, ??0 = 0.5 and ??0 = 0.01, ??0 = 0.5, respectively.

The last two rows show NB toy results with r0 = 10, p0 = 0.2 and r0 = 0.5, p0 = 0.2, respectively.

In each iteration, gradient variances are estimated with 20 Monte Carlo samples (each sample corresponds to one gradient estimate), among which the last one is used to update parameters.

100 Monte Carlo samples are used to calculate the ELBO in the NB toys.??? 1-layer linear model: DISPLAYFORM1 where ??(??) is the sigmoid function.??? Nonlinear model: DISPLAYFORM2 where tanh(??) is the hyperbolic-tangent function.

The used datasets and other experimental settings, including the hyperparameter search strategy, are the same as those in BID9 .For such single-latent-layer settings, it is obvious that Theorem 3 (also Theorem 1) can be straightforwardly applied.

However, since Bernoulli distribution has finite support, as mentioned in the main manuscript, we should analytically express some expectations for lower variance, as detailed below.

Notations of (8) and FORMULA6 of the main manuscript are used for clarity and also for generalization.

In fact, we should take a step back and start from (8) of the main manuscript, which is equivalent to analytically express an expectation in (9), namely DISPLAYFORM3 where DISPLAYFORM4 with P v (??) being the Bernoulli probability of Bernoulli random variable y v .

Accordingly, we have DISPLAYFORM5 For better understanding only, with abused notations DISPLAYFORM6 T , one should observe a chain rule within the above equation.

To assist better understanding of how to practically cooperate the presented GO gradients with deep learning frameworks like TensorFlow or PyTorch, we take (34) as an example, and present for it the following simple algorithm.

Algorithm 1 An algorithm for (34) as an example to demonstrate how to practically cooperate GO gradients with deep learning frameworks like TensorFlow or PyTorch.

One sample is assumed for clarity.

Practically, an easy-to-use trick for changing gradients of any function h(x) is to defin?? DISPLAYFORM7 Rely on the mature auto-differential software for back-propagating gradients For efficient implementation of D y f (y), one should exploit the prior knowledge of function f (y).

For example, f (y)s are often neural-network-parameterized.

Under that settings, one should be able to exploit tensor operation to design efficient implementation of D y f (y).

Again we take (34) as an example, and assume f (y) has the special structure DISPLAYFORM8 where ??(??) is an element-wise nonlinear activation function, and r(??) is a function that takes in a vector and outputs a scalar.

One can easily modify the above f (y) for the considered discrete VAE experiment.

Since y v s are now Bernoulli random variables with support {0, 1}, we have that DISPLAYFORM9 Published as a conference paper at ICLR 2019 (h) Omniglot Nonlinear Running-Time Figure 9 : Training/Validation ELBOs for the discrete VAE experiments.

Rows correspond to the experimental results on the MNIST/Omniglot dataset with the 1-layer-linear/nonlinear model, respectively.

Shown in the first/second column is the ELBO curves as a function of iteration/running-time.

All methods are run with the same learning rate for 1, 000, 000 iterations.

The black line represents the best training ELBO of REBAR and RELAX.

ELBOs are calculated using all training/validation data.

Note GO does not suffer more from over-fitting, as clarified in the text.

Then to efficiently calculate the f (y ???v , y v + a v )s, we use the following batch processing procedure to benefit from parallel computing.

DISPLAYFORM10 ???Step 1: Define ?? y h as the matrix whose element [?? y h] vj represents the "new" h * j when input {

y ???v , y v + a v } in (35).

Then, we have DISPLAYFORM11 where W :j is the j-th column of the matrix W. Note the vth row of ?? y h, i.e., [?? y h] v: , happens to be the "new" h * when input {y ???v , y v + a v }.???Step 2: Similarly, we define ?? y f as the vector whose element DISPLAYFORM12 Utilizing ?? y h obtained in Step 1, we have DISPLAYFORM13 where r(??) is applied to each row of the matrix DISPLAYFORM14 Note the above batch processing procedure can be easily extended to deeper neural networks.

Accordingly, we have DISPLAYFORM15 , where represents the matrix element-wise product.

Now we can rely on Algorithm 1 to solve the problem whose objective has its gradient expressed as (34), for example the inference of the single-latent-layer discrete VAE in (32).All training curves versus iteration/running-time are given in Figure 9 , where it is apparent that GO provides better performance, a faster convergence rate, and a better running efficiency in all situations.

The average running time per 100 iterations for the compared methods are given in TAB5 , where GO is 2 ??? 4 times faster in finishing the same number of training iterations.

We also quantify the running efficiency of GO by considering its running time to achieve the best training ELBO (within 1, 000, 000 training iterations) of RERAR/RELAX, referring to the black lines shown in the second-column subfigures of Figure 9 .

It is clear that GO is approximately 5 ??? 10 times more efficient than REBAR/RELAX in the considered experiments.

As shown in the second and fourth rows of Figures 9, for the experiments with nonlinear models all methods suffer from over-fitting, which originates from the redundant complexity of the adopted neural networks and appeals for model regularizations.

We detailedly clarify these experimental results as follows.??? All the compared methods are given the same and only objective, namely to maximize the training ELBO on the same training dataset with the same model; GO clearly shows its power in achieving a better objective.??? The "level" of over-fitting is ultimately determined by the used dataset, model, and objective; it is independent of the adopted optimization method.

Different optimization methods just reveal different optimizing trajectories, which show different sequences of training objectives and over-fitting levels (validation objectives).??? Since all methods are given the same dataset, model, and objective, they have the same over-fitting level.

Because GO has a lower variance, and thus more powerful optimization capacity, it gets to the similar situations much faster than REBAR/RELAX.

Note this does not mean GO suffers more from over-fitting.

In fact, GO provides better validation ELBOs in all situations, as shown in Figure 9 and also TAB0 of the main manuscript.

In practice, GO can benefit from the early-stopping trick to get a better generalization ability.

Complementing the multinomial GAN experiment in the main manuscript, we present more details as follows.

For a quantitative assessment of the computational complexity, our PyTorch code takes about 30 minutes to get the most challenging 4-bit task in Figs. 4 and 11, with a Titan Xp GPU.Firstly, recall that the generator p ?? (x) of the developed multinomial GAN (denoted as MNGAN-GO) is expressed as ??? N (0, I), x ??? Mult(1, NN P ( )), where NN P ( ) denotes use of a neural network to project to distribution parameters P. For brevity, we integration the generator's parameters ?? into the NN notation and do not explicitly express them.

Multinomial leaf variables x is used to describe discrete observations with a finite alphabet.

To train MNGAN-GO, the vanilla GAN loss BID8 ) is used.

A deconvolutional neural network as in BID28 is used to map to P in the generator.

The discriminator is constructed as a multilayer perceptron.

Detailed model architectures are given in Table 5 .

Figure 10 illustrates the pipeline of MNGAN-GO.

Note MNGAN-GO has a smaller number of parameters, compared to BGAN BID12 .For clarity, we briefly discuss the employed data preprocessing.

Taking MNIST for an example, the original data are 8-bit grayscale images, with pixel intensities ranging from 0 to 255.

For the n-bit experiment, we obtain the real data, such as the 2-bit one in Figure 10 , by rescaling and quantizing the pixel intensities to the range [0, 2 n ??? 1], having 2 n different states (values).For the 1-bit special case, the multinomial distribution reduces to the Bernoulli distribution.

Of course, one could intuitively employ the redundant multinomial distribution, which is denoted as 1-bit (2-state) in TAB1 .

An alternative and popular approach is to adopt the Bernoulli distribution to remove the redundancy by only modeling its probability parameters; we denote this case as 1-bit (1-state, Bernoulli) in TAB1 .

FIG0 shows the generated samples from the compared models on different quantized MNIST.

It is obvious that MNGAN-GO provides images with better quality and wider diversity in general.

Note for the challenging DLDA task in (37), we find it tricky to naively apply pure-gradient-based learning methods.

The main reason is: the latent code z (l) s and their gamma shape parameters ?? (l+1) z (l+1) s are usually extremely sparse, meaning most elements are almost zero; a gamma distribution z ??? Gam(??, ??) with almost-zero ?? has an increasingly steep slope when z approaches zero, namely the gradient wrt z shall have an enormous magnitude that unstablize the learning procedure.

Even though it might not be sufficient to just use the first-order gradient information, empirically the following tricks help us get the presented reasonable results.??? Let z (l) ??? T z , where T z = 1e ???5 is used in the experiments;??? Let c (l) ??? T c , where T c = 1e ???5 ;??? Let ?? (l+1) z (l+1) ??? T ?? with T ?? = 0.2;??? Use a factor to compromise the likelihood and prior for each z (l) .For more details, please refer to our released code.

We are working on exploiting higher-order information (such as Hessian) to help remedy this issue.

@highlight

a Rep-like gradient for non-reparameterizable continuous/discrete distributions; further generalized to deep probabilistic models, yielding statistical back-propagation

@highlight

Presents a gradient estimator for expectation-based objectives that is unbiased, has low variance, and applies to either continuous and discrete random variables.

@highlight

An improved method for computing derivates of the expectation, and a new gradient estimator of low variance that allows training of generative models in which observations or latent variables are discrete.

@highlight

Designs a low variance gradient for distributions associated with continuous or discrete random variables.