We introduce our Distribution Regression Network (DRN) which performs regression from input probability distributions to output probability distributions.

Compared to existing methods, DRN learns with fewer model parameters and easily extends to multiple input and multiple output distributions.

On synthetic and real-world datasets, DRN performs similarly or better than the state-of-the-art.

Furthermore, DRN generalizes the conventional multilayer perceptron (MLP).

In the framework of MLP, each node encodes a real number, whereas in DRN, each node encodes a probability distribution.

The field of regression analysis is largely established with methods ranging from linear least squares to multilayer perceptrons.

However, the scope of the regression is mostly limited to real valued inputs and outputs BID4 BID14 .

In this paper, we perform distribution-todistribution regression where one regresses from input probability distributions to output probability distributions.

Distribution-to-distribution regression (see work by BID17 ) has not been as widely studied compared to the related task of functional regression BID3 .

Nevertheless, regression on distributions has many relevant applications.

In the study of human populations, probability distributions capture the collective characteristics of the people.

Potential applications include predicting voting outcomes of demographic groups BID5 and predicting economic growth from income distribution BID19 .

In particular, distribution-to-distribution regression is very useful in predicting future outcomes of phenomena driven by stochastic processes.

For instance, the Ornstein-Uhlenbeck process, which exhibits a mean-reverting random walk, has wide-ranging applications.

In the commodity market, prices exhibit mean-reverting patterns due to market forces BID23 .

It is also used in quantitative biology to model phenotypic traits evolution BID0 .Variants of the distribution regression task have been explored in literature BID18 .

For the distribution-to-distribution regression task, BID17 proposed an instance-based learning method where a linear smoother estimator (LSE) is applied across the inputoutput distributions.

However, the computation time of LSE scales badly with the size of the dataset.

To that end, BID16 developed the Triple-Basis Estimator (3BE) where the prediction time is independent of the number of data by using basis representations of distributions and Random Kitchen Sink basis functions.

BID9 proposed the Extrapolating the Distribution Dynamics (EDD) method which predicts the future state of a time-varying probability distribution given a sequence of samples from previous time steps.

However, it is unclear how it can be used for the general case of regressing distributions of different objects.

Our proposed Distribution Regression Network (DRN) is based on a completely different scheme of network learning, motivated by spin models in statistical physics and similar to artificial neural networks.

In many variants of the artificial neural network, the network encodes real values in the nodes BID21 BID10 BID1 .

DRN is novel in that it generalizes the conventional multilayer perceptron (MLP) by encoding a probability distribution in each node.

Each distribution in DRN is treated as a single object which is then processed by the connecting weights.

Hence, the propagation behavior in DRN is much richer, enabling DRN to represent distribution regression mappings with fewer parameters than MLP.

We experimentally demonstrate that compared to existing methods, DRN achieves comparable or better regression performance with fewer model parameters.

Figure 1 : (Left) An example DRN with multiple input probability distributions and multiple hidden layers mapping to an output probability distribution. (Right) A connection unit in the network, with 3 input nodes in layer l − 1 connecting to a node in layer l.

Each node encodes a probability distribution, as illustrated by the probability density function P (l) k .

The tunable parameters are the connecting weights and the bias parameters at the output node.

and Y i are univariate continuous distributions with compact support, the regression task is to learn the function f which maps the input distributions to the output distribution.

DISPLAYFORM0 No further assumptions are made on the form of the distribution.

It is trivial to generalize our method to regress to multiple output distributions but for simplicity of explanation we shall restrict to single output regressions in the following discussions.

Fig. 1 illustrates how the regression in Eq. (1) is realized.

DRN generalizes the traditional neural network structure by encoding each node with a probability distribution and connecting the nodes with real-valued weights.

The input data consists of one or more probability distributions which are fed into the first layer and propagated layerwise through the hidden layers.

We emphasize our network is not a Bayesian network even though each node encodes a probability.

Unlike bayes net where the conditional probability among variables are learnt by maximizing the likelihood over observed data, DRN regresses probability distributions using a feedforward network, similar to MLP.

At each node in the hidden layer, the probability distribution is computed from the probability distributions of the incoming nodes in the previous layer and the network parameters consisting of the weights and bias parameters (see right of Fig. 1 ).

Pk represents the probability density function (pdf) of the k th node in the l th layer and P DISPLAYFORM0 k ) is the density of the pdf when the node variable is s DISPLAYFORM1 Before obtaining the probability distribution P (l) k , we first compute its unnormalized formP DISPLAYFORM2 k is computed by marginalizing over the product of the unnormalized conditional probabilitỹ Q(s DISPLAYFORM3 ) and the incoming node probabilities.

DISPLAYFORM4 represent the variables of the lower layer nodes and E is the energy given a set of node variables, which we define later in Eq. (4).

The unnormalized conditional probability has the same form as the Boltzmann distribution in statistical mechanics, except that the partition function is omitted.

This omission reduces the computational complexity of our model through factorization, shown later in Eq. (5).Our energy function formulation is motivated by work on spin models in statistical physics where spin alignment to coupling fields and critical phenomena are studied BID12 BID8 BID27 .

Energy functions are also used in other network models where a scalar energy is associated to each configuration of the nodes BID24 BID11 .

In such energybased models, the parameters are learnt such that the observed configurations of the variables have lower energies than unobserved ones.

However, the energy function used in DRN is part of the forward propagation process and is not directly optimized.

For a given set of node variables, the energy function is DISPLAYFORM5 ki is the weight connecting the i th node in the lower layer to the upper layer node.

b respectively.

The support length of the distribution is given by ∆. All terms in Eq. (4) are normalized by the support length so that the energy function is invariant with respect to the support.

Eq. (2) can be factorized such that instead of having multidimensional integrals, there are n univariate integrals: DISPLAYFORM6 k ) captures the bias terms of the energy function in Eq. (4).

DISPLAYFORM7 Finally, the probability distribution from Eq. (2) is normalized.

DISPLAYFORM8 The propagation of probability distributions within a connection unit forms the basis for forward propagation.

Forward propagation is performed layerwise from the input layer using Eq. (2) to (7).

The forward propagation in DRN has some important properties.

Fig. 2 illustrates the propagation behavior for a connection unit with one input node where the bias values b DISPLAYFORM0 a,k and b DISPLAYFORM1 q,k are set as zero.

DISPLAYFORM2 Figure 2: Propagation behavior for a connection unit with one input node.

The biases are set as zero in these examples.

When weight is zero, the output distribution is flat.

Positive weights causes the output distribution to have the same peak position as the input distribution while negative weights causes the output pdf to 'repel' away from the input peak.

When the weight is a sufficiently large positive number, the propagation tends towards the identity mapping.

When the weight is zero, the output distribution is flat and the output distribution is independent of the input.

With a positive weight, the output distribution is 'attracted' to the peak of the input distribution whereas a negative weight causes the output distribution to be 'repelled' away from the input peak.

In addition, the weight magnitude represents the strength of the 'attraction' or 'repulsion'.

When the weight is a sufficiently large positive number, the propagation tends towards the identity mapping (top right example in Fig. 2 ).

The implication is that like in neural networks, a deeper network should have at least the same complexity as a shallow one, as the added layers can produce the identity function.

Conversely, a small positive weight causes the output peak to be at the same position as the input peak, but with more spread (second example on left column of Fig. 2 ).The remaining absolute and quadratic bias terms in Eq. (4) have a similar role as the bias in a traditional neural network.

Depending on the bias values b q,k respectively.

The weight and bias values play a similar role as the inverse temperature in the Boltzmann distribution in statistical physics BID12 BID27 .

The cost function of the network given a set network parameters is measured by the Jensen-Shannon (JS) divergence between the label (Y i ) and predicted DISPLAYFORM0 and D KL is the Kullback-Liebler divergence.

The Jensen-Shannon divergence is a suitable cost function as it is symmetric and bounded.

The network cost function C net is the average D JS over all M training data: DISPLAYFORM1

In our experiments, the integrals in Eq. (5) and (7) are performed numerically.

This is done through discretization from continuous probability density functions (pdf) to discrete probability mass functions (pmf).

Given a continuous pdf with finite support, the range of the continuous variable is partitioned into q equal widths and the probability distribution is binned into the q states.

The estimation error arising from the discretization step will decrease with larger q.

The network cost is a differentiable function over the network parameters.

We derive the cost gradients similar to backpropagation in neural networks BID22 .

We use chain rule to derive at each node a q-by-q matrix which denotes the derivative of the final layer node distribution with respect to the current node distribution.

DISPLAYFORM0 where DISPLAYFORM1 is the final layer output probability distribution.

From the derivative DISPLAYFORM2

We evaluate DRN on synthetic and real-world datasets and compare its performance to the state-ofthe-art 3BE method and a fully-connected multilayer perceptron (MLP).

For each of the datasets, DRN achieves similar or higher accuracy with fewer model parameters.

In MLP, each discretized probability mass function is represented by q nodes.

The MLP consists of fully connected hidden layers with ReLU units and a softmax final layer, and is optimized with mean squared error using Adam.

Unlike DRN and MLP where the distribution pdfs are directly used by the methods, 3BE assumes the input and output distributions are observed through i.i.d.

samples.

Hence, for the first two datasets we provide 3BE with sufficient samples from the underlying distribution such that errors from density estimation are minimal.

The first experiment involves a synthetic dataset similar to the one used by BID17 DISPLAYFORM0 The function h transforms the means and standard deviations using the non-linear function shown in Fig. 3a .

The transformation is such that the two gaussian means will remain in their respective ranges.

The sample input-output data pairs in Fig. 3b shows the complexity of the regression task with various behavior like peak splitting and peak spreading.

1000 training data and 1000 testing data were created to evaluate the regression methods.

For DRN and MLP, the pdfs are discretized into q = 100 states and for 3BE, 10,000 samples from each data distribution are generated.

While 3BE gives a continuous distribution as the output, DRN and MLP output the discrete pmf and require conversion to continuous pdf.

Following BID18 , the regression performance on the test set is measured by the L2 loss between the continuous predicted distribution,Ŷ (s) and the true distribution.

We study how the regression accuracy varies with respect to the number of model parameters.

For DRN and MLP, the number of parameters are varied using different depths and widths of the networks and for 3BE, we vary the number of Random Kitchen Sink features.

We present the detailed DRN architecture in Appendix B. Fig. 4a shows the L2 loss on the test set as we vary the number of model parameters.

Note that the x-axis is presented on the log scale.

DRN's test performance is comparable to the other methods and uses fewer model parameters to attain reasonable performance.

We note there is little overfitting for the three methods, as shown in the plots comparing train and test loss in Fig. 4b , though 3BE starts to exhibit overfitting when the number of model parameters approaches 10,000.

Because of the Boltzmann distribution term (ref.

Eq. 3), DRN models the diffusion process very well.

For this experiment, we evaluate our model on data generated from the stochastic OrnsteinUhlenbeck (OU) process BID25 which combines the notion of random walk with a drift towards a long-term mean.

The OU process has wide-ranging applications.

In the commodity market, prices exhibit mean-reverting patterns due to market forces and hence modelling the prices with the OU process helps form valuation strategies BID23 BID28 .The OU process is described by a time-varying gaussian pdf.

With the long-term mean set at zero, the pdf has a mean of µ(t) = y exp(−θt) and variance of σ 2 (t) = DISPLAYFORM0 .

t represents time, y is the initial point mass position, and D and θ are the diffusion and drift coefficients respectively.

The regression task is to map from an initial gaussian distribution at t init to the resulting distribution after some time step ∆t.

The gaussian distributions are truncated with support of [0, 1] .

With different sampled values for y ∈ [0.3, 0.9] and t init ∈ [0.01, 2], pairs of distributions are created for ∆t = 1, D = 0.003 and θ = 0.1.

For DRN and MLP, q = 100 was used for discretization of the pdfs while 10,000 samples were taken for each distribution to train 3BE.

We compare the number of model parameters required to achieve a small L2 test loss with 100 training data.

We also increased the training size to 1000 and attained similar results.

TAB0 and FIG5 show that a simple DRN of one input node connecting to one output node with 5 parameters performs similarly as MLP and 3BE.

MLP requires 1 fully-connected hidden layer with 3 nodes, with a total of 703 network parameters.

3BE requires 64 projection coefficients for both input and output distributions and 17 Random Kitchen Sink features, resulting in 272 model parameters.

The regression by DRN on two random test samples are shown in FIG5 and we see that DRN is able to demonstrate the OU process.

FIG5 shows the 5 DRN parameters after training.

The values of these parameters are interpreted as follows.

The weight parameter is positive, hence the output peak position is positively correlated to the input peak position.

Moreover, w = 75.3 is such that the network mimics the diffusion property of the OU process.

The bias position λ a is negative and its magnitude is 5 times the distribution support, causing the output peak to be displaced leftwards of the input peak.

These two observations reflect the random walk and mean-reverting properties of the OU process.

Figure 6 : Single-layer network used in DRN for the stock dataset with 7 model parameters (3 weights, 4 bias parameters).

We demonstrate that DRN can be useful for an important real-world problem and outperforms 3BE and MLP in terms of prediction accuracy.

With greater integration of the global stock markets, there is significant co-movement of stock indices BID6 BID2 .

In a study by BID26 , it was found that the previous day stock returns of the Nikkei and Dow Jones Industrial Average (Dow) are good predictors of the FTSE return.

Modelling the co-movement of global stock indices has its value as it facilitates investment decisions.

Stock indices are weighted average of the constituent companies' prices in a stock exchange, and existing research has primarily focused on the movement of returns of the indices.

However, for our experiment, we predict the future distribution of returns over the constituent companies in the index as it provides more information than just a weighted average.

Our regression task is as follows.

Given the current day's distribution of returns of constituent companies in FTSE, Dow and Nikkei, predict the distribution of returns for constituent companies in FTSE k days later.

The logarithmic return for the company's stock at day t is given by ln(V t /V t−1 ), where V t and V t−1 represent its closing price at day t and t − 1 respectively.

The stock data consists of 9 years of daily returns from January 2007 to December 2015.

To adapt to changing market conditions, we use a sliding-window training scheme where the data is split into windows of training, validation and test sets and moved foward in time BID7 .

A new window is created and the network is retrained after every 300 days (which is the size of test set).

For each test set, the previous 500 and 100 days were used for training and validation.

To reduce the noise in the data, we performed exponential window averaging on the price series for each stock with a window of 50 days following common practice BID15 .

The logarithmic returns of the constituent company stocks form the samples for the distributions of the returns.

For DRN and MLP, the pdf is estimated using kernel density estimation with a gaussian kernel function with bandwidth of 0.001 and q = 100 was used for discretization of the pdf.

The authors of 3BE have extended their method for multiple input functions (see joint motion prediction experiment in BID16 ).

We followed their method and concatenated the basis coefficients obtained from the three input distributions.

In addition, for 3BE we scale the return samples to [0, 1] before applying cosine basis projection.

The predicted distribution is then scaled back to the original range for quantification of the regression performance.

First, we performed evaluations for the task of predicting the next-day distributions.

As we do not have the underlying true pdf for this real-world dataset, the regression performance is measured by the log-likelihood of the test samples.

TAB1 shows the test log-likelihoods, where higher loglikelihood is favorable.

Interestingly, the single-layer network in DRN (see Fig. 6 ) was sufficient to perform well, using just 7 network parameters.

In comparison, MLP and 3BE require 4110 and 8100 parameters respectively.

To visualize the regression results on the test set, we compare for each day the first two moments (mean and variance) of the predicted distribution and the ground truth (see 1-day ahead panels of FIG6 ).

Each point represents one test data and we show the Pearson correlation coefficients between the predicted and labelled moments.

DRN has the best regression performance as the points lie closest to the diagonal line where the predicted and labelled moments are equal, and its correlation values are highest.

As an extension, we predict the FTSE returns distribution several days ahead.

The second and third rows of FIG6 and FIG6 show the moment plots for 5 and 10 days ahead respectively.

Expectedly, the performance deterioriates as the number of days increases.

Still, DRN outperforms the rest as shown by the moment plots and the correlation values.

FIG7 summarizes the results by showing the average absolute error of the mean and variance as the number of days-ahead increases.

For all experiments, DRN consistently has the lowest error.

Finally, we conducted experiments on a real-world cell dataset similar to the one used in BID17 .

The dataset is a time-series of images of NIH3T3 fibroblast cells.

There are 277 time frames taken at 5-minute intervals, containing 176 to 222 cells each.

In each frame, we measured the long and short-axis nuclear length of the cells and scaled the lengths to [0, 1] .

At each time-frame, given the distribution of long-axis length, we predict the distribution of the short-axis length.

The first 200 frames were used for training and last 77 for testing.

For DRN and MLP, the pdf is estimated using kernel density estimation with a gaussian kernel of bandwidth 0.02 and q = 100 was used for discretization.

We compare the log-likelihood on test data in TAB3 .

DRN had the best loglikelihood with a simple network of one input node connecting to one output node.

In contrast, MLP and 3BE used more model parameters but achieved lower log-likelihoods.

This validated DRN's advantage at learning distribution regressions on real-world data with fewer model parameters.

The distribution-to-distribution regression task has many useful applications ranging from population studies to stock market prediction.

In this paper, we propose our Distribution Regression Network which generalizes the MLP framework by encoding a probability distribution in each node.

Our DRN is able to learn the regression mappings with fewer model parameters compared to MLP and 3BE.

MLP has not been used for distribution-to-distribution regression in literature and we have adapted it for this task.

Though both DRN and MLP are network-based methods, they encode the distribution very differently.

By generalizing each node to encode a distribution, each distribution in DRN is treated as a single object which is then processed by the connecting weight.

Thus, the propagation behavior in DRN is much richer, enabling DRN to represent the regression mappings with fewer parameters.

In 3BE, the number of model parameters scales linearly with the number of projection coefficients of the distributions and number of Random Kitchen Sink features.

In our experiments, DRN is able to achieve similar or better regression performance using less parameters than 3BE.

Furthermore, the runtime for DRN is competitive with other methods (see comparison of mean prediction times in Appendix C).For future work, we look to extend DRN for variants of the distribution regression task such as distribution-to-real regression and distribution classification.

Extensions may also be made for regressing multivariate distributions.

In this section, show the DRN network architecture used for the synthetic dataset results presented in Fig. 4a .

There is one input node and one output node connected by a number of hidden layers of arbitrary width.

All layers are fully-connected.

We compare the mean prediction time per data for DRN and the baseline methods.

All runs were conducted on the CPU.

For the synthetic dataset, we have shown the test loss for varying parameter sizes.

For a fair comparison of runtime, for each method we chose a model size which gave a test L2 loss of about 0.37.

For all the datasets, MLP has the fastest prediction time, followed by DRN and then 3BE.

<|TLDR|>

@highlight

A learning network which generalizes the MLP framework to perform distribution-to-distribution regression