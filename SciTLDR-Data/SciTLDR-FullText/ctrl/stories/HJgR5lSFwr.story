This paper aims to address the limitations of mutual information estimators based on variational optimization.

By redefining the cost using generalized functions from nonextensive statistical mechanics we raise the upper bound of previous estimators and enable the control of the bias variance trade off.

Variational based estimators outperform previous methods especially in high dependence high dimensional scenarios found in machine learning setups.

Despite their performance, these estimators either exhibit a high variance or are upper bounded by log(batch size).

Our approach inspired by nonextensive statistical mechanics uses different generalizations for the logarithm and the exponential in the partition function.

This enables the estimator to capture changes in mutual information over a wider range of dimensions and correlations of the input variables whereas previous estimators saturate them.

Understanding the relationship between two variables is a fundamental problem in machine learning, finance, signal processing and other fields.

To quantify such a relationship, we use mutual information, which measures the mutual dependence between two variables.

The mutual information I(X, Y ) represents the ratio of two probabilities that can account for the nonlinear dependence and is defined as follows:

It is a major challenge to estimate the mutual information in practical scenarios that involve limited samples without knowing the distributions or higher-order statistics McAllester & Statos (2018) ; Paninski (2003) .

For instance, existing methods such as the k-NN based Kraskov et al. (2004) and its variations Gao et al. (2015) ; Wang et al. (2009); Lord et al. (2018) , or KDE based Khan et al. (2007) ; Suzuki et al. (2008) calculate the mutual information by estimating the probability density from the available samples.

Although these approaches perform well in the low dimensional and low dependence case, they do not scale well when either the dimension of the variables or the dependence between variables increases.

Such scenarios are often encountered in machine learning setups.

Estimators based on variational bounds, Belghazi et al. (2018) ; Poole et al. (2018) ; Nguyen et al. (2010) ; ; Zhang (2007); Foster & Grassberger (2011) , perform much better in this scenarios.

These estimators are inspired by the Donsker & Varadhan (1983) representation which states that there exists a function from the sample space to real number that satisfies the following equality: I(X, Y ) = sup f :Ω→R E p(x,y) [f (x, y)] − log E p(y) e f (x,y)

Estimators based on variational bounds replace the function f in the above equation with a neural network trained to maximize a lower bound of the mutual information.

The training process terminates when the lower bounds exhibit convergence, and these bounds are then interpreted as the estimated mutual information values.

This NN-based approach requires good representations of lower bounds and guaranteed convergence for a wide range of dependence between the input variables which leads to numerous challenges.

Current state-of-the-art estimators, when applied to high dimensional high dependence scenarios, either exhibit a high variance or are bounded by log(K), where K is the batch size.

In this work:

1.

We propose new variational lower bounds on the mutual information inspired by methods from nonextensive statistical mechanics.

2.

We review generalized versions of the logarithm and exponential function, define a generalized version of the partition function, and use them to control the trade off between variance and bias of the estimator.

3.

We outperform previous estimators in capturing the trend when varying the correlation and the dimension of the input variables by using different generalizations for the logarithm and partition function.

In what follows, we present most of the recent developments on variational lower bounds for estimation of mutual information.

For a thorough review of variational estimation of mutual information we refer the reader to the work of Poole et al. (2018) .

Most commonly, when estimating the mutual information, both p(x|y) and p(x) distributions in equation 1 are unknown.

The only thing we have are samples from these distributions.

In this case, to calculate the mutual information using the variational method we introduce a variational distribution p(x|y) to learn from samples.

Multiplying and dividing equation 1 by this distribution and rewriting the expression we obtain the Barber & Agakov (2003) lower bound on the mutual information:

The last inequality holds since the Kullback-Liebler divergence is always non-negative.

The bound is tight whenp(x|y) = p(x|y), i.e., our variational distribution converges to the true distribution.

To further simplify this variational optimization problem we makep(x|y) have the form in Poole et al. (2018) :p

where f is the critic, Z is the partition function, and K the number of samples.

Equation 4 can also be written asp(x|y) = p(x) · Kσ(x; y) where σ(x; y) is the softmax function.

In this form we see that softmax function inp(x|y) maps the output of the critic to a probability.

However, since the number of states (outputs) is unknown, its average is used instead of the partition function.

Replacing equation 4 into equation 3, nesting the expectations Rainforth et al. (2018) , and bounding log Z(y) ≤ Z(y) a(y) + log a(y) − 1 ∀x, a(y) > 0, which is a tight bound for a(y) = Z(y), results in the Nguyen et al. (2010) bound, I NWJ (X, Y ):

where a(y) was set to be the natural constant and the second expectation computed with shuffled y values.

This bound reaches the true mutual information when the critic approaches the optimal critic f * (x, y) = 1 + log p(x|y) p(x) .

I NWJ estimator performs well in high dimensional high dependence scenarios but exhibits high variance due to the exponential function in the partition function which amplifies the critic errors.

One way to reduce the variance is to make the critic depend on multiple samples and use a Monte Carlo estimator for the partition function.

To accomplish this we set the optimal critic to depend on K additional samples f * (x 1:K , y) = 1 + log xi,y) .

Replacing this new multisample critic in equation 5 and averaging over the batch after iteratively fixing each sample as x 1 , we obtain the noise contrastive estimator I NCE (X, Y ) introduced by Oord et al. (2018):

Although, as a result of averaging, the I NCE estimator exhibits a much lower variance compared to I NWJ , this estimator is upper bounded by I NCE ≤ log K where K is the batch size used to train the critic.

This bound can be verified by extracting the K from the denominator outside the sum,

where the average is always non-positive since softmax returns a probability which is less or equal to 1.

One possible solution to solve the high variance of I NWJ and the upper bound of I NCE , proposed by Poole et al. (2018) , is to use a nonlinear interpolation between the two.

To accomplish that, the optimal critic for I NWJ , equation 5, is set to f

is the secondary critic, and α ∈ [0, 1] interpolates between I NWJ (α = 0) and I NCE (α = 1).

Plugging this critic in equation 5 we obtain Poole et al. (2018) bound I α :

Here, similar to I NWJ case, the second expectation is calculated using y values independently sampled from x. The nonlinear interpolation in this case increases the upper bound of the estimator from log K to log(K/α).

Although these estimators perform much better in high dimensional high dependence scenarios than the ones based on k-NN or KDE, they still have their own limitations.

In particular, I NWJ has a high variance due to the exponential in the partition function that limits its use.

I NCE solves the variance problem by averaging the partition function which results in the estimator being bounded by log(K) where K is the batch size used to train the critic.

For the most common values of the batch size, the upper bound of this estimator is between 6 and 8, much lower than the values of mutual information encountered in the machine learning setup.

I α estimator, which is a combination of the two, can reach higher values than I NCE , however, the variance is not far from I NWJ variance and it requires training two critics.

In what follows we first present a generalization of the exponential and logarithm function from the nonextensive statistical mechanics.

Then we use these functions to derive a generalized version of the previous bounds and show how, by using the generalization parameter q, we can overcome their limitations.

The most known universality class in statistical mechanics is the Boltzmann-Gibbs class.

In this class the entropy is defined in terms of a probability set, and in particular, for a thermodynamic system at equilibrium it is calculated as:

where E i is the energy of the system in state i, β = 1/k B T is the thermodynamic beta, and Z BG is the partition function that encodes the distribution of the probabilities among states.

This expression found much attention in many fields outside statistical mechanics and most notably in information theory.

In the information theoretic context, the Bolzmann constant k B = 1 and the pair −βE i is an unnormalized function for which the entropy is calculated.

This is exactly the case of expression 4 where we normalize the critic using the partition function.

The Boltzmann-Gibbs entropy was generalized in the work of Tsallis (1988) which introduces a parameter q, named generalization, to control weight of the individual probabilities.

Tsallis generalized entropy is defined as:

where log q is the q-logarithm which converges to the classical logarithm when q → 1.

The generalization q can be viewed as a "bias".

When q < 1 smaller probabilities are amplified, whereas when q > 1 the larger ones are amplified.

Although a majority of systems are well described by the classical entropy and partition function, more complex system such as brain activity Tong et al. (2002) , financial markets Michael & Johnson (2003) , and black holes Majhi (2017) often do not follow the classical laws and are better captured by the generalized version.

Over the years, Tsallis entropy evolved into a separate branch of statistical mechanics known as nonextensive statistical mechanics Tsallis (2009) .

This new branch extensively makes use of generalized logarithm and exponential shown in Figure 1a and 1b defined as:

where [z] + = max(z, 0).

The following identities, which we will use in the next section to derive new variational bounds, are true: exp q (x + y) = exp q (x) ⊗ q exp q (y) and log q (xy) = log q (x) ⊕ q log q (y) where ⊕ q and ⊗ q are the q-addition and q-multiplication.

For the definition of the q-operators and a brief introduction to q-algebra, consult Appendix B.

Similar to generalized entropy, the generalized mutual information is defined as:

Moreover, we can define a generalized version of the partition function using the generalized exponential:

The advantage of the generalized partition function is the freedom to choose how much weight large values will receive when mapping to probability.

Figure 1c shows how the triplet (1, 2, 3) is mapped into probabilities for different values of q. We will use this in the next section to improve the perfomance of the estimators.

In this section, we derive a generalized version of previously introduced bounds using Tsallis statistics in order to overcome the limitations of the previous variational bounds.

Although, the generalized mutual information differs in value with the classical mutual information due to the generalization.

When the generalization parameter q →

1 we recover the classical logarithm log(x), for q = 0 a linear function that passes through (1, 0), and for other values of q a nonlinear interpolation between the two.

(b) qgeneralization of the exponential function.

The behaviour of exp q (x) is similar to log q (x) with respect to the generalization parameter q. For the same value of q the two functions are the inverse of each other, i.e., exp q (log q (x)) = x and log q (exp q (x)) = x. (c) Normalizing the triplet (1, 2, 3) using the q-generalization of the partition function for different values of q. Higher the value of q, higher the probability mass is attributed to the largest value in the set.

The two have the same properties and capture the dependence between input variables.

To obtain a variational bound on the generalized mutual information we start from the definition, expression 11, which we multiply and divide by the variational distributionp(x|y):

Again the inequality holds since the D KLq (p(x|y)||p(x|y)) is a positive number for all values of q and is tight whenp(x|y) = p(x|y) as 0 is a neutral element for the generalized q-addition.

Again, this expression can be further simplified by selectingp(x|y) be of the form:

where Z q is the generalized partition function.

For q = 0 we recover the counting measure, for q = 1 we obtain the softmax function, and for other value we obtain a nonlinear interpolation between the two.

In what follows we set the q of the partition function equal to the Tsallis divergence, however, there is no reason to have them equal as the partition function has the role of a normalizer.

Replacing equality 14 into 13 we obtain:

Next, to obtain the Nguyen, Wainwright and Jordan bound, I NWJ , we bound the partition function using the following inequality log q Z q (y) ≤ Zq(y) a(y) + log q a(y) − 1 ∀x, q, a(y) > 0 which is tight when a(y) = Z q (y).

Setting a(y) = exp q (y) yields the bound:

which reaches the true mutual information when the critic approaches the optimal critic f * (x, y) = 1 ⊕ q log q p(x|y)

p(x) that can be verified by replacing this expression for critic in equation 16.

To obtain the q-equivalent for the multisample bound I NCE we replace the optimal critic with the multisample optimal critic f * (x 1:k , y) = 1 ⊕ q log q y) ) in inequality 16 which results in:

By iteratively reindexing each sample in x 1:K as x 1 and averaging the result, the last term becomes:

Replacing expression 18 into 17, and then as in previous step iteratively reindexing each sample in x 1:K as x 1 and averaging the result we obtain:

which is our proposed lower bound on mutual information based on nonextensive statistical mechanics which we name accordingly I NES .

Similar to the original I NCE , equation 6, our estimator is upper bounded by log q (K).

However, we can overcome this limitation by choosing a generalization q < 1 for the logarithm.

In addition, since the generalization of the logarithm is independent of the generalization of the partition function we can set them independently.

Thus, we control how the critic output is mapped to a probability using the generalization q of the partition function and the estimator bound through the generalization q of the logarithm.

To show this, we rewrite expression 19 as:

where q 1 is the logarithm generalization and q 2 is the partition function generalization.

We use Gaussian distributed random vectors to test our estimator and compare its performance with previous works.

For the particular case when the vector elements with the same index have correlation ρ and 0 when indices do not match, the mutual information can be calculated using the following formula: For each experiment we use 3000 samples to estimate the mutual information and train the critic in batches of 512 samples to maximize the variational bounds introduced in the previous section.

The result of these experiments are shown in Figure 2 .

We observe that for low dimensions dim(X) < 10, a majority of the estimators can capture the trend, Figure 2a .

However, for dimensions dim(X) > 10, I NWJ and I NCE saturate.

Although both I NES and I α do follow the trend of the true mutual information for all input dimensions, I NES outperforms I α which uses two critics and requires both of them to converge.

Note, the returned value of I NES estimator is not a classical mutual information as it uses a generalized logarithm, however, it has the same properties as the classical mutual information.

In contrast to the input dimension experiment, estimators capture the trend much better over a wider range of correlations between inputs, Figure 2b .

The two estimators that saturate first in this case are I NCE and I α for correlations ρ > 0.9.

I NES captures the trend and remains consistent across the whole range up to ρ > 0.99.

From Figures 2c and 2d we can see that the performance of the I NES estimators comes with a cost, it exhibits higher variance than I NCE but lower than the one of I NWJ .

The main goal of the current work was to improve the performance of mutual information estimators in the high dependence high dimensional scenarios which are often encountered in machine learning setups.

We reviewed previous variational lower bounds and extended them using generalized logarithm and exponential functions from nonextensive statistical mechanics.

One of the most significant findings to emerge from this study was that we can control the trade off between the bias and the variance of the estimator by independently tuning the generalizations q of the logarithm and partition function.

As a result, we are able to better capture the trend when varying the correlation and dependence of the input variables.

This method greatly improves upon the I NCE estimator which has a low variance but a high bias and I α estimator which requires two critics that can sometimes be challenging to train.

The major limitation of the proposed estimator is that its results are not equal in value with the classical mutual information due to use of generalization.

Despite that, I NES still captures the dependence between the input variable and can be applicable to machine learning problems where a mutual information estimator is needed such as feature selection, variational autoencoders, and generative adversarial networks.

Addition and subtraction operations in q-algebra are defined as follow:

x ⊕ q y = x + y + (1 − q)xy x q y = x − y 1 + (1 − q)y

However, exponential and logarithmic identities have a slightly different form:

exp q (x + y) = exp q (x) ⊗ q exp q (y) exp q (x ⊕ q y) = exp q (x) · exp q (y) log q (xy) = log q (x) ⊕ q log q (y) log q (x ⊗ q y) = log q (x) + log q (y)

<|TLDR|>

@highlight

Mutual information estimator based nonextensive statistical mechanics

@highlight

This paper tries to establish novel variational lower bounds for mutual information by introducing parameter q and defining q-algebra, showing that the lower bounds have smaller variance and achieves high values.