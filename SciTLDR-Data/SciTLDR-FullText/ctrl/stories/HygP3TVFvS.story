Gaussian processes are ubiquitous in nature and engineering.

A case in point is a class of neural networks in the infinite-width limit, whose priors correspond to Gaussian processes.

Here we perturbatively extend this correspondence to finite-width neural networks, yielding non-Gaussian processes as priors.

The methodology developed herein allows us to track the flow of preactivation distributions by progressively integrating out random variables from lower to higher layers, reminiscent of renormalization-group flow.

We further develop a perturbative prescription to perform Bayesian inference with weakly non-Gaussian priors.

Gaussian processes model many phenomena in the physical world.

A prime example is Brownian motion (Brown, 1828) , modeled as the integral of Gaussian-distributed bumps exerted on a pointlike solute (Einstein, 1905) .

The theory of elementary particles (Weinberg, 1995) also becomes a Gaussian process in the free limit where interactions between particles are turned off, and manybody systems as complex as glasses come to be Gaussian in the infinite-dimensional, mean-field, limit (Parisi & Zamponi, 2010) .

In the context of machine learning, Neal (1996) pointed out that a class of neural networks give rise to Gaussian processes in the infinite-width limit, which can perform exact Bayesian inference from training to test data (Williams, 1997) .

They occupy a corner of theoretical playground wherein the karakuri of neural networks is scrutinized (Lee et al., 2018; Matthews et al., 2018; Jacot et al., 2018; Chizat et al., 2018; Geiger et al., 2019) .

In reality, Gaussian processes are but mere idealizations.

Brownian particles have finite-size structure, elementary particles interact, and many-body systems respond nonlinearly.

In order to understand rich phenomena exhibited by these real systems, Gaussian processes rather serve as starting points to be perturbed around.

Indeed many edifices in theoretical physics are build upon the successful treatment of non-Gaussianity, with a notable example being renormalization-group flow (Kadanoff, 1966; Wilson, 1971; Weinberg, 1996; Goldenfeld, 2018) .

In the quest to elucidate behaviors of real neural networks away from the infinite-width limit, it is thus natural to wonder if the similar treatment of non-Gaussianity yields equally elegant and powerful machinery.

Here we set out on this program, perturbatively treating finite-width corrections to neural networks.

Prior distributions of outputs are obtained through progressively integrating out preactivation of neurons layer by layer, yielding non-Gaussian priors.

Intriguingly, intermediate recursion relations and their derivation resemble renormalization-group flow (Goldenfeld, 2018; Mehta & Schwab, 2014) .

Such a recursive approach further enables us to treat finite-width corrections on Bayesian inference and their regularization effects, with arbitrary activation functions.

The rest of the paper is structured as follows.

In Section 2 we review and set up basic concepts.

Our master recursive formulae (R1,R2,R3) are derived in Section 3, which control the flow of preactivation distributions from lower to higher layers.

After an interlude with concrete examples in Section 4, we extend the Gaussian-process Bayesian inference method to non-Gaussian priors in Section 5 and use the resulting scheme to study inference of neural networks at finite widths.

We conclude in Section 6 with dreams.

In this paper we study real finite-width neural networks in the regime where the number of neurons in hidden layers is asymptotically large whereas input and output dimensions are kept constant.

Let us focus on a class of neural networks termed multilayer perceptrons, with model parameters, θ = b ( ) i , W ( ) i,j , and an activation function, σ.

For each input, x ∈ R n0 , a neural network outputs a vector, z(x; θ) = z (L) ∈ R n L , recursively defined as sequences of preactivations through

i,j x j for i = 1, . . .

, n 1 ,

Following Neal (1996), we assume priors for biases and weights given by independent and identically distributed Gaussian distributions with zero means, E b

i,j = 0, and variances

Higher moments are then obtained by Wick's contractions (Wick, 1950; Zee, 2010) .

For instance,

For those unfamiliar with Wick's contractions and connected correlation functions (a.k.a.

cumulants), a pedagogical review is provided in Appendix A as our formalism heavily relies on them.

In the infinite-width limit where n 1 , n 2 , . . .

, n L−1 → ∞ (but finite n 0 and n L ), it has been argued -with varying degrees of rigor (Neal, 1996; Lee et al., 2018; Matthews et al., 2018) -that the prior distribution of outputs is governed by the Gaussian process with a kernel

and all the higher moments given by Wick's contractions.

In particular, there exists a recursive formula that lets us evaluate this kernel for any pair of inputs (Lee et al., 2018) [c.f.

Equation (R1)].

Importantly, once the values of the kernel are evaluated for all the pairs of

..,ND , consisting of N R training inputs with target outputs and N E test inputs with unknown targets, we can perform exact Bayesian inference to yield mean outputs as predictions for N E test data (Williams, 1997; Williams & Rasmussen, 2006) [c.f.

Equation (GPM)].

This should be contrasted with stochastic gradient descent (SGD) optimization (Robbins & Monro, 1951) , through which typically a single estimate for the optimal model parameters of the posterior, θ , is obtained and used to predict outputs for test inputs; Bayesian inference instead marginalizes over all model parameters, performing an ensemble average over the posterior distribution (MacKay, 1995) .

We shall now study real finite-width neural networks in the regime n 1 , . . .

, n L−1 ∼ n 1.

1 At finite widths, there are corrections to Gaussian-process priors.

In other words, a whole tower of 1 Note that input and output dimensions, n0 and nL, are arbitrary.

To be precise, defining n1, . . .

, nL−1 ≡ µ1n, . . .

, µL−1n, we send n 1 while keeping C ( )

, µ1, . . .

, µL−1, n0, and nL constants, and compute the leading 1/n corrections.

In particular it is crucial to keep the number of outputs nL constant in order to consistently perform Bayesian inference within our approach.

nontrivial preactivation correlation functions beyond the kernel,

collectively dictate the distribution of preactivations.

Our aim is to trace the flow of these distributions progressively and cumulatively all the way up to the last layer whereat Bayesian inference is executed.

More specifically, we shall inductively and self-consistently show that two-point preactivation correlation functions take the form

and connected four-point preactivation correlation functions is symmetric under α 1 ↔ α 2 , α 3 ↔ α 4 , and (α 1 , α 2 ) ↔ (α 3 , α 4 ).

At the first layer the preactivation distribution is exactly Gaussian for any finite widths and hence Equations (KS) and (V) are trivially satisfied, with

= 0 , and

Obtained in Section 3 are the recursive formulae that link these core kernel, self-energy, and fourpoint vertex at the -th layer to those at the ( + 1)-th layer while in Section 5 these tensors at the last layer = L are used to yield the leading 1/n correction for Bayesian inference at finite widths.

Our Schwinger operator approach is orthogonal to the replica approach by Cohen et al. (2019) and, unlike the planar diagrammatic approach by Dyer & Gur-Ari (2019) , applies to general activation functions, made possible by accumulating corrections layer by layer rather than dealing with them all at once.

More substantially, in contrast to these previous approaches, we here study finite-width effects on Bayesian inference and find that the renormalization-group picture naturally emerges.

As auxiliary objects in recursive steps, let us introduce activation correlation functions

Our basic strategy is to establish relations

zigzagging between sets of preactivation correlation functions and sets of activation correlation functions, keeping track of leading finite-width corrections.

Below, relations G ( ) → H ( ) are obtained by integrating out preactivations while relations H ( ) → G ( +1) are obtained by integrating out biases and weights.

At first glance the algebra in this paper may look horrifying but repeated applications of Wick's contractions are all there is to it.

The results are summarized in Section 3.2.

2 In the main text we place tildes on objects that depend only on sample indices α's in order to distinguish them from those that depend both on sample indices α's and neuron indices i's.

3 Given that the means of biases and weights are zero, G

The remaining task is to relate preactivation correlations G ( ) to activation correlations H ( ) within the same layer, which will complete the zigzag relation (ZIGZAG) for these correlation functions.

4 Once the sorcery of Wick's contractions and connected correlation functions is mastered, it is simple to derive the following combinatorial hack (Appendix A.4): viewing prior preactivations

as a random (n N D )-dimensional vector and defining the Gaussian integral with the kernel z

, the prior average

capture 1/n corrections due to selfenergy and four-point vertex, respectively, and are defined as

where the sample indices are lowered by using the inverse core kernel as a metric, meaning

Using the above hack, we can evaluate the activation correlations by straightforward algebra with -you guessed it -Wick's contractions.

As the Gaussian integral is diagonal in the neuron index i, we just need to disentangle cases with repeated and unrepeated neuron indices.

The solution for this exercise is in Appendix B: this is the most cumbersome algebra in the paper and the ability to perform it certifies the graduation from the magical school of Wick's crafts and wizardly cumulants.

4 The nontrivial parts of the inductive proof for Equations (KS) and (V) are to show (i) that the right-hand side of Equation (10) is finite as n → ∞, (ii) that the leading contribution of Equation (9) is the Gaussianprocess kernel, and (iii) that higher-point connected preactivation correlation functions are all suppressed by O 1 n 2 , all of which are verified in obtaining the recursive equations.

See Appendix B for a full proof.

Denoting the Gaussian integral with the core kernel z

..,ND , and plugging in results of Appendix B into Equations (9) and (10), we arrive at our master recursion relations

, and

For = 1, a special note about the ratio n n −1 is in order: even though n 0 stays constant while n 1 1, the terms proportional to that ratio are identically zero due to the complete Gaussianity (R0).

The preactivation distribution in the first layer (R0) sets the initial condition for the flow from lower to higher layers dictated by these recursive equations.

Once recursed up to the last layer = L, the resulting distribution of outputs z = z (L) can be succinctly encoded by the probability distribution

with the potential

, and (D1)

By now, the reader should be deriving this through Wick's contractions without solicitation.

It is important to note that n L is constant and thus H 1 [z] can consistently be treated perturbatively.

5 If nL were of order n 1, the potential H would become a large-n vector model, for which we would have to sum the infinite series of bubble diagrams (Moshe & Zinn-Justin, 2003) .

The recursive relations obtained above can be evaluated numerically (Lee et al., 2018) [or sometimes analytically for ReLU (Cho & Saul, 2009 )], which is a perfectly adequate approach: at the leading order it involves four-dimensional Gaussian integrals at most.

Here, continuing the theme of wearing out Wick's contractions, we develop an alternative analytic method that works for any polynomial activations (Liao & Poggio, 2017) , providing another perfectly cromulent approach.

For a general polynomial activation of degree p, σ(z) = p k=0 a k z k , the nontrivial term in Equation (R1) can be expanded as

Each term can then be evaluated by Wick's contractions and the same goes for all the terms in Equations (R2) and (R3).

Below and in Appendix C, we illustrate this procedure with simple examples.

When the activation function is linear, σ(z) = z, multilayer perceptrons go under the aweinspiring name of deep linear networks (Saxe et al., 2013) .

Setting C

= 0 and C ( ) W = 1 for simplicity, our recursion relations reduce to K α1,α2

, and S α1,α2

Solving them yields the layer-independent core kernel and zero self-energy

and the linearly layer-dependent four-point vertex

.

It succinctly reproduces the result that can be obtained through planar diagrams in this special setup (Dyer & Gur-Ari, 2019) .

Quadratic activation (Li et al., 2018) is worked out in Appendix C.1.

The recursion relations simplify drastically for the case of a single input, N D = 1, as worked out in detail in Appendix C.2.

For instance, for rectified linear unit (ReLU) activation with C ( ) b = 0 and C ( ) W = 2, we obtain the layer-independent core kernel, zero self-energy, and the four-point vertex

2 .

Interestingly, as for deep linear networks, the factor (1/n ) appears again.

This factor has also been found by Hanin & Rolnick (2018) , which provides guidance for network architectural design through its minimization.

We generalize this factor for monomial activations in Appendix C.2.1

Here we put our theory to the test.

For concreteness, take a single black-white image of handwritten digits with 28-by-28 pixels (i.e. n 0 = 784) from the MNIST dataset (LeCun et al., 1998) without preprocessing, set depth L = 3, bias variance C ( ) b = 0, weight variance C ( ) W = C W , and widths (n 0 , n 1 , n 2 , n 3 ) = (784, n, 2n, 1), and use activations σ(z) = z (linear) with C W = 1 and max(0, z) (ReLU) with C W = 2.

In Figure 1 , for each width-parameter n of the hidden layers we record the prior distribution of outputs over 10 6 instances of Gaussian weights and compare it with the theoretical prediction -obtained by cranking the knob from the initial condition (R0) through the recursion relations (R1-R3) to the distribution (D0-D2).

The prior distribution becomes increasingly non-Gaussian as networks narrow and the deviation from the Gaussian-process prior is correctly captured by our theory.

Higher-order perturbative calculations are expected to systematically improve the quality -and extend the range -of the agreement.

Additional experiments are performed in Appendix C.3, which further corroborates our theory.

Figure 1: Comparison between theory and experiments for prior distributions of outputs for a single input.

The agreement between our theoretical predictions (smooth thick lines) and experimental data (rugged thin lines) is superb, correctly capturing the initial deviations from Gaussian processes at n = ∞ (black), all the way down to n ∼ 10 for linear activation and to n ∼ 30 for ReLU activation.

Let us take off from the terminal point of Section 3: we have obtained the recursive equations (R0-R3) for the Gaussian-process kernel and the leading finite-width corrections and codified them in the weakly non-Gaussian prior distributions

with ≡

..,n L .

We shall develop a formalism to infer outputs for test inputs a lá Bayes, perturbatively extending the textbook by Williams & Rasmussen (2006) .

For field theorists, our calculation is just a tree-level background field calculation (Weinberg, 1996) in disguise.

Taking the liberty of notations, we let the number of input-data arguments dictate the summation over sample indices α inside the potential H, and denote the joint probabilities

Given the training targets y R , the posterior distribution of test outputs are given by Bayes' rule:

The leading Gaussian-process contributions can be segregated out through the textbook manipulation (Williams & Rasmussen, 2006) [c.f.

Appendix D.1]: denoting the full Gaussian-process kernel in the last layer as

and the Gaussian-process posterior mean prediction as

and defining a fluctuation

For any function F, its expectation over the Bayesian posterior (Bayes) then turns into

where the deviation kernel (δz E )

and the normalization factor

In particular the mean posterior output is given by

], recalling Equation (D2) for H 1 , and using Wick's contractions for one last time, the mean prediction becomes

With additional manipulations in Appendix D, this expression is simplified into the actionable form that is amenable to use in practice.

For illustration, there, we also show a simple preliminary experiment, which indicates the 1/n regularization effect for sufficiently large width n and small amount of training data N R .

This is in line with expectations that finite widths ameliorate overfitting and that non-Gaussian priors increase the expressivity of neural functions, but additional large-scale extensive experiments would be desirable in the future.

In this paper, we have developed the perturbative formalism that captures the flow of preactivation distributions from lower to higher layers.

The spiritual resemblance between our recursive equations and renormalization-group flow equations in high-energy and statistical physics is highly appealing.

It would be exciting to investigate the structure of fixed points away from the Gaussian asymptopia (Schoenholz et al., 2016) and fully realize the dream articulated by Mehta & Schwab (2014) beyond their limited example of a mapping between two antiquated techniques -the audacious hypothesis that neural networks wash away microscopic irrelevancies and extract relevant features.

In addition we have developed the perturbative Bayesian inference scheme universally applicable whenever prior distributions are weakly non-Gaussian, and have applied it to the specific cases of neural networks at finite widths.

In light of finite-width regularization effects, it would be prudent to revisit the empirical comparison between SGD optimization and Bayesian inference at finite widths (Lee et al., 2018; Novak et al., 2019) , especially for convolutional neural networks.

Finally, given surging interests in SGD dynamics within the large-width regime (Jacot et al., 2018; Chizat et al., 2018; Cohen et al., 2019; Dyer & Gur-Ari, 2019) , it would be natural to adapt our formalism for investigating corrections to neural tangent kernels, and even nonperturbatively aspire to capture a phase transition out of lazy-learning into feature-learning regimes.

Welcome to the magical school of Wick's crafts and wizardly cumulants.

Here is all you need to know in order to follow the calculations in the paper.

In the main text, Wick's contractions are used trivially for integrating out biases and weights as straightforward applications of Appendix A.1 while they are used more nontrivially for integrating out preactivations, with concepts of cumulants reviewed in Appendix A.2 and A.3, culminating in a hack derived in Appendix A.4.

For most parts, we shall forget about the neuron index i for pedagogy and put them back in at the very end.

For Gaussian-distributed variables z = {z α } α=1,...,N with a kernel K α,α , moments

(S1) For any odd m such moments identically vanish.

For even m, Isserlis-Wick's theorem states that

where the sum is over all the possible pairings of m variables,

For a proof, see for example Zee (2010) .

In order to understand and use the theorem, it is instructive to look at a few examples:

and

Given general (not necessarily Gaussian) random variables, connected correlation functions are defined inductively through

where the sum is over all the possible subdivisions of m variables into s > 1 clusters of sizes (ν 1 , . . .

, ν s ) as (k

νs ).

In order to understand the definition, it is again instructive to look at a few examples.

Assuming that all the odd moments vanish,

Rearranging them in particular yields

If these examples do not suffice, here is yet another example to chew on:

and hence

We emphasize that these are just renderings of the definition (S6).

The power of this definition will be illustrated in the next two subsections.

We often encounter situations with the hierarchy

where 1 is a small perturbative parameter and here again odd moments are assumed to vanish.

Often comes with the hierarchical structure is the asymptotic limit → 0 where

with the Gaussian kernel K α1,α2 at zero and the leading self-energy correction S α1,α2 .

Let us also denote the leading four-point vertex

For instance this hierarchy holds for weakly-coupled field theories -from which we are importing names such as self-energy, vertex, and metric -and, in this paper, such hierarchical structure is inductively shown to hold for prior preactivations z ( ) with = 1 n −1 in the regime n 1 , . . .

, n L−1 ∼ n 1.

Note that, by definition, K α1,α2 and S α1,α2 are symmetric under α 1 ↔ α 2 and V α1,α2,α3,α4

is symmetric under permutations of (α 1 , α 2 , α 3 , α 4 ).

With the review of connected correlation functions passed us, we can now readily see that

where in the last equality Wick's theorem was used backward.

So far we have reviewed the standard technology of Wick's contractions, connected correlation functions, and all that.

Here is one sorcery, which lets us magically evaluate E [z α1 · · · z αm ] by mindless repetitions of Wick's contractions.

Throughout, we shall assume the hierarchical structure (S12), 7 which is the inductive assumption in the main text, and start from its consequence (CLUSTER).

Below, let us use the inverse kernel K −1 α1,α2

as a metric to lower indices:

First note that

7 More precisely, we shall only use the weaker assumption that

for m ≥ 6 along with Equations (S13) and (S14).

where the symmetry α 1 ↔ α 2 of S α1,α2 was used.

Hence, defining

we obtain, for one term in Equation (CLUSTER),

The similar algebraic exercise renders the other term in Equation (CLUSTER) to be

In summary, for any function

In order to get the expressions used in the main text at the -th layer, we need only to put back neuron indices i by replacing α → (α, i), identify = 1 n −1 , and use the inductive assumptions (KS)

The operators in Equations (OS') and (OV') then become

i.e., the operators in Equations (OS) and (OV) in the main text.

In this Appendix, we provide a full inductive proof for one of the main claims in the paper, streamlined in the main text.

Namely, we assume at the -th layer that Equations (KS) and (V) hold and that all the higher-point connected preactivation correlation functions are of order O 1 n 2 -which are trivially true at = 1 -and prove the same for the ( + 1)-th layer.

We assume the full mastery of Appendix A or, conversely, this section can be used to test the mastery of wicked tricks.

First, trivial Wick's contractions yield

Studiously disentangling cases with different numbers of repetitions in neuron indices (j 1 , . . .

, j k ), we notice that at order O 1 n , terms without repetition or with only one repetition contribute, finding

where we used the inductive hierarchical assumption at the -th layer, i.e., its consequence (HACK) and denoted a single-neuron random vectorz ( ) = {z α } α=1,...,ND and the Gaussian integral with the core kernel z

As special cases, we obtain expressions advertised in the main text to be contained in this Appendix:

Assembling everything,

In particular,

completing our inductive proof.

Note that B (α1,α2),(α3,α4) = O 1 n .

Nowhere in our derivation had we assumed anything about the form of activation functions.

The only potential exceptions to our formalism are exponentially growing activation functions -which we have never seen in practice -that would make the Gaussian integrals unintegrable.

Let us take multilayer perceptrons with quadratic activation, σ(z) = z 2 , and study the distributions of preactivations in the second layer as another illustration of our technology.

From the master recursion relations (R1-R3) with the initial condition (R0), Wickology yields

, and

=0 .

where

.

These expressions are used in Appendix D.2 for the experimental study of finite-width corrections on Bayesian inference.

The recursive relations simplify drastically for the case of a single input, N D = 1.

Setting C ( ) b = 0 for simplicity and dropping α index, our recursive equations reduce to

, and (S34)

Under review as a conference paper at ICLR 2020

For monomial activations, σ(z) = z p , such as in deep linear networks (Saxe et al., 2013 ) and quadratic activations (Li et al., 2018) ,

, and (S37)

In particular the four-point vertex solution is given by

The factor 1 n p 2 generalizes the factor 1 n for linear and ReLU activations.

Following Hanin & Rolnick (2018) , this factor guides us to narrow hidden layers as we pass through nonlinear activations.

ReLU activation, σ(z) = max(0, z), can also be worked out for a single input through Wick's contractions, noting that the Gaussian integral is halved, yielding

, and (S41)

Setting C ( ) W = 2 for simplicity, these equations can be solved, leading to

(1) , and (S44)

Here is an extended version of experiments in Section 4.3.

As in the main text, take a single blackwhite image of hand-written digits from the MNIST dataset as an n 0 = 784-dimensional input, without preprocessing.

Set bias variance C and max(0, z) (ReLU) with C W = 2.

For all three cases, we consider both depth L = 2 with widths (n 0 , n 1 , n 2 ) = (784, n, 1) and depth L = 3 with widths (n 0 , n 1 , n 2 , n 3 ) = (784, n, 2n, 1).

As in Figure 1 , in Figure S1 , for each width-parameter n of the hidden layers we record the prior distribution of outputs over 10 6 instances of Gaussian weights and compare it with the theoretical prediction.

Results again corroborate our theory.

Figure S1 : Comparison between theory and experiments for prior distributions of outputs for a single input.

Our theoretical predictions (smooth thick lines) and experimental data (rugged thin lines) agree, correctly capturing the initial deviations from the Gaussian processes (black, n = ∞), at least down to n = n with n ∼ 10 for linear cases, n ∼ 30 for ReLU cases and depth L = 2 quadratic case, and n ∼ 100 for depth L = 3 quadratic case.

This also illustrates that nonlinear activations quickly amplify non-Gaussianity.

This expression simplifies drastically through the identity

which can be checked explicitly, recalling

RR K RE .

Incidentally, this identity can also be used to prove Equation (GP∆).

Now equipped with this identity, recalling φ

Finally, denoting the matrix inside the parenthesis to be are actionable, i.e., easy to program.

It turns out that for deep linear networks the leading finite-width correction given above vanishes, and the first correction is likely to show up at higher order in 1/n asymptotic expansion, which is not carried out in this paper.

Here we instead use the L = 2 multilayer perceptron with the quadratic activation for illustration, plugging Equations (S30,S31,S32) into Equations (NGPM') and (NGPM").

Set C Figure S2 indicate the regularization effects of finite widths, at least when the number of training samples, N R , is small, resulting in peak performance at finite widths.

Figure S2 : Test accuracy for N E = 10000 MNIST test data as a function of the inverse width = 1/n L−1 of the hidden layer with quadratic activation.

For each number N R of subsampled training data, the result is averaged over 10 distinct choices of such subsamplings.

For small numbers of training data, finite widths result in regularization effects, improving the test accuracy.

<|TLDR|>

@highlight

We develop an analytical method to study Bayesian inference of finite-width neural networks and find that the renormalization-group flow picture naturally emerges.