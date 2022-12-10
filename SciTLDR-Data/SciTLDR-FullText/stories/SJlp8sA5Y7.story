While deep neural networks have achieved groundbreaking prediction results in many tasks, there is a class of data where existing architectures are not optimal -- sequences of probability distributions.

Performing forward prediction on sequences of distributions has many important applications.

However, there are two main challenges in designing a network model for this task.

First, neural networks are unable to encode distributions compactly as each node encodes just a real value.

A recent work of Distribution Regression Network (DRN) solved this problem with a novel network that encodes an entire distribution in a single node, resulting in improved accuracies while using much fewer parameters than neural networks.

However, despite its compact distribution representation, DRN does not address the second challenge, which is the need to model time dependencies in a sequence of distributions.

In this paper, we propose our Recurrent Distribution Regression Network (RDRN) which adopts a recurrent architecture for DRN.

The combination of compact distribution representation and shared weights architecture across time steps makes RDRN suitable for modeling the time dependencies in a distribution sequence.

Compared to neural networks and DRN, RDRN achieves the best prediction performance while keeping the network compact.

Deep neural networks have achieved state-of-the-art results in many tasks by designing the network architecture according to the data type.

For instance, the convolutional neural network (CNN) uses local filters to capture the features in an image and max pooling to reduce the image representation size.

By using a series of convolution and max pooling layers, CNN extracts the semantic meaning of the image.

The recurrent architecture of recurrent neural networks (RNN) when unrolled, presents a shared weight structure which is designed to model time dependencies in a data sequence.

However, among the major network architectures, the multilayer perceptron, convolutional neural network and recurrent neural network, there is no architecture suitable for representing sequences of probability distributions.

Specifically, we address the task of forward prediction on distribution sequences.

There are two main challenges in designing a network for sequences of probability distributions.

First, conventional neural networks are unable to represent distributions compactly.

Since each node encodes only a real value, a distribution has to be decomposed to smaller parts that are represented by separate nodes.

When the distribution has been decomposed into separate nodes, the notion of distribution is no longer captured explicitly.

Similarly, for image data, the fully-connected multilayer perceptron (MLP), unlike convolutional neural networks, fails to capture the notion of an image.

A recently proposed network, Distribution Regression Network (DRN) BID8 , has solved this problem.

DRN uses a novel representation of encoding an entire distribution in a single node, allowing DRN to use more compact models while achieving superior performance for distribution regression.

It has been shown that DRN can achieve better accuracies with 500 times fewer parameters compared to MLP.

However, despite the strengths of DRN, it is a feedforward network and hence it does not address a second problem, which is the need to model time dependencies in a distribution sequence.

We address these two challenges and propose a recurrent extension of DRN, named the Recurrent Distribution Regression Network (RDRN).

In the hidden states of RDRN, each node represents a distribution, thus containing much richer information while using fewer weights compared to the real-valued hidden states in RNN.

This compact representation consequently results in better generalization performance.

Compared to DRN, the shared weights in RDRN captures time dependencies better and results in better prediction performance.

By having both compact distribution representations and modeling of time dependencies, RDRN is able to achieve superior prediction performance compared to the other methods.

Performing forward prediction on time-varying distributions has many important applications.

Many real-world systems are driven by stochastic processes.

For such systems, the Fokker-Planck equation BID21 has been used to model the time-varying distribution, with applications in astrophysics BID13 ), biological physics (Guérin et al., 2011 , animal population studies BID0 and weather forecasting BID18 .

In these applications, it is very useful to predict the future state of the distribution.

For example, the Ornstein-Uhlenbeck process, which is a specific case of the Fokker-Planck equation, has been used to model and predict commodity prices BID22 .

Extrapolating a time-varying distribution into the future has also been used for predictive domain adaptation, where a classifier is trained on data distribution which drifts over time BID9 .Various machine learning methods have been proposed for distribution data, ranging from distribution-to-real regression BID16 to distribution-to-distribution regression BID14 .

The Triple-Basis Estimator (3BE) has been proposed for the task of function-to-function regression.

It uses basis representations of functions and learns a mapping from Random Kitchen Sink basis features BID14 .

The authors have applied 3BE for distribution regression, showing improved accuracy and speed compared to an instance-based learning method BID15 .

More recently, BID8 proposed the Distribution Regression Network which extends the neural network representation such that an entire distribution is encoded in a single node.

With this compact representation, DRN showed better accuracies while using much fewer parameters than conventional neural networks and 3BE BID14 .The above methods are for general distribution regression.

For predicting the future state of a timevarying distribution, it is important to model the time dependencies in the distribution sequence.

BID9 proposed the Extrapolating the Distribution Dynamics (EDD) method which predicts the future state of a time-varying distribution given a sequence of samples from previous time steps.

EDD uses the reproducing kernel Hilbert space (RKHS) embedding of distributions and learns a linear mapping to model the dynamics of how the distribution evolves between adjacent time steps.

EDD is shown to work for a few variants of synthetic data, but the performance deteriorates for tasks where the dynamics is non-linear.

Since the regression is performed with just one input time step, it is unclear how EDD can be extended for more complex trajectories that require multiple time steps of history.

Another limitation is that the EDD can only learn a single trajectory of the distribution and not from multiple trajectories.

We address the task of forward prediction from a time-varying distribution: Given a series of distributions with T equally-spaced time steps, X(1) , X (2) , · · · , X (T ) , we want to predict X (T +k) , ie.

the distribution at k time steps later.

We assume the distributions to be univariate.

The input at each time step may consist of more than one distribution, for instance, when tracking multiple independent distributions over time.

In this case, the input distribution sequence is denoted as (X DISPLAYFORM0 n ), where there are n data distributions per time step.

Performing prediction on distribution sequences requires both compact distribution representations and modeling of time dependencies.

While the recurrent neural network works well for time series data, it has no efficient representation for distributions.

As for DRN, although it has a compact representation for distributions, the feedforward architecture does not capture the time dependencies in the distribution sequence.

Hence, we propose our Recurrent Distribution Regression Network (RDRN) which is a recurrent extension of DRN.

DISPLAYFORM1

Neural network models work well if the network architecture is designed according to the data type.

Convolutional neural networks are suited for image data as they employ convolution to capture local features from neighboring image pixels.

Such important data domain knowledge is not built in the fully-connected multilayer perceptron.

For analysis of distributions, there are no conventional neural network architectures like what CNN does for images.

To that end, BID8 proposed the Distribution Regression Network (DRN) for the task of distribution-to-distribution regression.

To cater to distribution data, DRN has two main innovations: 1) each network node encodes an entire distribution and 2) the forward propagation is specially designed for propagating distributions, with a form inspired by statistical physics.

We give a brief description of DRN following the notations of BID8 .

FIG0 illustrates the propagation in DRN.

Similar to MLP, DRN consists of multiple fully-connected layers connecting the data input to the output in a feedforward manner, where each connection has a real-valued weight.

The novelty of DRN is that each node in the network encodes an entire probability distribution.

The distribution at each node is computed using the distributions of the incoming nodes, the weights and the bias parameters.

Let P (l) k represent the probability density function (pdf) of the k th node in the l th layer where DISPLAYFORM0 k ) is the density of the pdf when the node variable is s DISPLAYFORM1 k is computed by marginalizing over the product of the unnormalized conditional probabilityQ(s DISPLAYFORM2 ) and the incoming probabilities.

DISPLAYFORM3 (1) DISPLAYFORM4 E is the energy for a given set of node variables, DISPLAYFORM5 ki is the weight connecting the i th node in layer l − 1 to the k th node in layer l. b DISPLAYFORM6 a,k are the quadratic and absolute bias terms acting on positions λ (l) q,k and λ (l) a,k respectively.

∆ is the support length of the distribution.

After obtaining the unnormalized probability, the distribution from Eq. (1) is normalized.

Forward propagation is performed layer-wise obtain the output prediction.

With such a propagation method, DRN exhibits useful propagation properties such as peak spreading, peak shifting, peak splitting and the identity mapping BID8 .

Due to space constraints, we refer the readers to BID8 for a more detailed description.

Since DRN is a feedforward network, it does not explicitly capture the time dependencies in distribution sequences.

In this work, we introduce our Recurrent Distribution Regression Network (RDRN) which is a recurrent extension of DRN.

The input data is a distribution sequence as described in Section 3.1.

FIG0 shows an example network for RDRN, where the network takes in T time steps of distributions to predict the distribution at T + k. The hidden state at each time step may consist of multiple distributions.

The arrows represent fully-connected weights.

The input-hidden weights U and the hidden-hidden weights W are shared across the time steps.

V represents the weights between the final hidden state and the output distribution.

The bias parameters for the hidden state nodes are also shared across the time steps.

The hidden state distributions at t = 0 represents the 'memory' of all past time steps before the first input and can be initialized with any prior information.

In our experiments, we initialize the t = 0 hidden states as uniform distributions as we assume no prior information is known.

We formalize the propagation for the general case where there can be multiple distributions for each time step in the data input layer and the hidden layer.

Let n and m be the number of distributions per time step in the data layer and hidden layers respectively.

Propagation starts from t=1 and performed through the time steps to obtain the hidden state distributions.

DISPLAYFORM0 i ) represents the input data distribution at node i and time step t, when the node variable is r DISPLAYFORM1 k ) represents the density of the pdf of the k th hidden node at time step t when the node variable is s DISPLAYFORM2 k ) represents the unnormalized form.

The hidden state distributions at each time step is computed from the hidden state distributions from the previous time step and the input data distribution from the current time step.

DISPLAYFORM3 The energy function is given by DISPLAYFORM4 where for each time step, u ki is the weight connecting the i th input distribution to the k th hidden node.

Similarly, for the hidden-hidden connections, w kj is the weight connecting the j th hidden node in the previous time step to the k th hidden node in the current time step.

As in DRN, the hidden node distributions are normalized before propagating to the next time step.

At the final time step, the output distribution is computed from the hidden state distributions, through the weight vector V and bias parameters at the output node.

Following BID8 , the cost function for the forward prediction task is measured by the Jensen-Shannon (JS) divergence BID11 between the label and output distributions.

Optimization is performed by backpropagation through time.

We adopt the same parameter initialization method as BID8 , where the network weights and bias are randomly initialized following a uniform distribution and the bias positions are uniformly sampled from the support length of the data distribution.

The integrals in Eq. (4) are performed numerically, by partitioning the distribution into q bins, resulting in a discrete probability mass function.

We conducted experiments on four datasets which involve prediction of time-varying distributions.

To evaluate the effectiveness of the recurrent structure in RDRN, we compare with DRN where the input distributions for all time steps are concatenated at the input layer.

We also compare with conventional neural network architectures and other distribution regression methods.

The benchmark methods are DRN, RNN, MLP and 3BE.

For the final dataset, we also compare with EDD as the data involves only a single trajectory of distribution.

Among these methods, RNN and EDD are designed to take in the inputs sequentially over time while for the rest the inputs from all T time steps are concatenated.

Since DRN is a feedforward network, the distributions for all input time steps are concatenated and fed in together.

The architecture consists of fullyconnected layers, where each node encodes an entire distribution.

DRN is optimized using JS divergence.

Recurrent Neural Network (RNN) At each time step, the distribution is discretized into bins and represented by separate input nodes.

The RNN architecture consists of a layer of hidden states, where the number of nodes is chosen by cross validation.

The input-hidden and hidden-hidden weights are shared across time steps.

The final hidden state is transformed by the hidden-output weights and processed by a softmax layer to obtain the output distribution.

The cost function is the mean squared error between the predicted and output distribution bins.

Multilayer Perceptron (MLP) The input layer consists of the input distributions for all time steps and each distribution is discretized into bins that are represented by separate nodes.

Hence, for T input time steps and discretization size of q, there will be T × q input nodes in MLP.

MLP consists of fully-connected layers and a final softmax layer, and is optimized with the mean squared error.

Triple-Basis Estimator (3BE) For 3BE, each distribution is represented by its sinusoidal basis coefficients.

The number of basis coefficients and number of Random Kitchen Sink basis functions are chosen by cross validation.

Extrapolating the Distribution Dynamics (EDD) Since EDD learns from a single trajectory of distribution, it is unsuitable for most of the datasets.

We performed EDD for the final dataset which has only a single distribution trajectory.

For the RKHS embedding of the distributions, we use the radial basis function kernel, following Lampert (2015).

For the first experiment, we chose a dataset where the output distribution has to be predicted from multiple time steps of past distributions.

Specifically, we track a Gaussian distribution whose mean varies in the range [0.2, 0.8] sinusoidally over time while the variance is kept constant at 0.01.

Given a few consecutive input distributions taken from time steps spaced ∆t = 0.2 apart, we predict the next time step distribution.

FIG1 illustrates how the mean changes over time.

It is apparent that we require more than one time step of past distributions to predict the future distribution.

For instance, at two different time points, the distribution means can be the same, but one has increasing mean while the other has a decreasing mean.

To create the dataset, for each data we randomly sample the first time step from [0, 2π].

The distributions are truncated with support of [0, 1] and discretized with q=100 bins.

We found that for all methods, a history length of 3 time steps is optimal.

Following BID16 the regression performance is measured by the L2 loss.

TAB0 shows the regression results, where lower L2 loss is favorable.

20 training data was sufficient for RDRN and DRN to give good predictions.

RDRN's regression accuracy is the best, followed by DRN.

FIG1 shows four test data, where the input distributions at t=1, 2, 3 are shown, along with the label output for t=4 and RDRN's prediction.

We observe good fit for the predictions.

Additionally, the top and bottom left data shows that two data can have the same mean at t=3, but are moving in opposite directions.

Hence, to predict the next distribution at t=4, multiple time steps in history are required as input and the model has to determine the direction of movement from the history of distributions.

Since RDRN is designed to model time dependencies in the distribution sequence, it is able to infer the direction of the mean movement well.

In contrast, the neural network counterparts of RNN and MLP showed considerable overfitting which is likely due to the fact that excessive number of nodes are used to represent the distributions, resulting in many model parameters.

In the field of climate modeling, variability of climate measurements due to noise is an important factor to consider BID7 .

The next experiment is based on the work of BID10 , where they model the heat flux at the sea surface as a time-varying one-dimensional distribution.

Specifically, the evolution of the heat flux over time obeys the stochastic Ornstein-Uhlenbeck (OU) process BID23 , and the diffusion and drift coefficients are determined from real data measurements obtained by BID17 .

The OU process is described by a time-varying Gaussian distribution.

With the long-term mean set at zero, the pdf has a mean of µ(t) = y exp(−θt) and variance of σ 2 (t) = DISPLAYFORM0 .

t represents time, y is the initial point mass position, and D and θ are the diffusion and drift coefficients respectively.

For the energy balance climate model, D = 0.0013, θ = 2.86, and each unit of the nondimensional time corresponds to 55 days BID10 .

At t =0, the distribution is a delta-function at position y. To create a distribution sequence, we first sample y ∈ [0.02, 0.09].

For each sampled y, we generate 6 Gaussian distributions at t 0 − 4δ, t 0 − 3δ, ..

The regression task is as follows:

Given the distributions at t 0 − 4δ, t 0 − 3δ, ..., t 0 , predict the distribution at t 0 + 0.02.

With different sampled values for y and t 0 , we created 100 training and 1000 test data.

The regression performance is measured by the L2 loss.

The regression results on the test set are shown in TAB0 .

RDRN's regression accuracy is the best, followed by DRN.

This is followed by the neural network architectures MLP and RNN.

It is noteworthy that RDRN and RNN, which explicitly capture time dependencies in the architecture, perform better than their feedforward counterparts.

In addition, the recurrent models perform best with more time steps (T =5) compared to the feedforward models (T =3), which may suggest that the recurrent architecture captures the time dependencies in the data sequence better than a feedforward one.

In terms of model compactness, RDRN and DRN use at 2-3 orders of magnitude fewer model parameters compared to the other methods, owing to the compact distribution representation.

RDRN can be used to track the distribution drift of image datasets.

For the next experiment, we use the CarEvolution dataset BID20 which was used by BID9 for the domain adaptation problem.

The dataset consists of 1086 images of cars manufactured from the years 1972 to 2013.

We split the data into intervals of 5 years (ie. 1970-1975, 1975-1980, · · · , 2010-2015) where each interval has an average of 120 images.

This gives 9 time intervals and for each interval, we create the data distribution from the DeCAF(fc6) features BID3 of the car images using kernel density estimation.

The DeCAF features have 4096 dimensions.

Performing accurate density estimation in very high dimensions is challenging due to the curse of dimensionality BID4 .

Here we make the approximation that the DeCAF features are independent, such that the joint probability is a product of the individual dimension probabilities.

The regression task is to predict the next time step distribution of features given the previous T time step distributions.

We found T =2 to work best for all methods.

The first 7 intervals were used for the train set while the last 3 intervals were used for the test set, giving 5 training and 1 test data.

The regression performance is measured by the negative log-likelihood of the test samples following BID15 , where lower negative log-likelihood is favorable.

The regression results are shown in Table 2a .

RDRN has the best prediction performance, followed by DRN.

RNN had difficulty in optimization possibly due to the high number of input dimensions, so the results are not presented.

EDD has the fewest number of parameters as it assumes the dynamics of the distribution follows a linear mapping between the RKHS features of consecutive time steps (ie.

T =1).

However, as the results show, the EDD model may be too restrictive for this dataset.

We show RDRN is useful for forward prediction of price movements in stock markets.

We adopt a similar experimental setup as BID8 .

There have been studies that show that movement of indices of stock markets in the world correlate with each other, providing a basis for predicting future stock returns BID6 BID2 .

Specifically, the previous day stock returns of the Nikkei and Dow Jones Industrial Average (Dow) are found to be good predictors of the FTSE return BID24 .

Furthermore, predicting the entire distribution of stock returns has been found to be more useful for portfolio selection compared to just a single index value BID1 .Following the setup in BID8 , our regression task is as follows: given the past T days' distribution of returns of constituent companies in FTSE, Dow and Nikkei, predict the distribution of returns for constituent companies in FTSE k days later.

We used 9 years of daily returns from 2007 to 2015 and performed exponential window averaging on the price series following common practice BID12 .

The regression performance is measured by the negative log-likelihood of the test samples.

The RDRN architecture used is shown in Figure 3 , where the data input consists past 3 days of distribution returns and one layer of hidden states with 3 nodes per time step is used.

We tested on forward prediction of 1 and 10 days ahead.

Table 2b shows the regression results.

As before, RDRN and DRN's performance surpasses the other methods by a considerable margin.

For 1 day ahead prediction, RDRN's performance is slightly below DRN, but for 10 days ahead, RDRN's performance is better.

This may suggest that the 1 day ahead task is simpler and does not involve long time dependencies.

On the other hand, predicting 10 days ahead is a more challenging task which may benefit from having a longer history of stock movements.

We further visualize the results by comparing the mean and variance of the predicted and the label distributions, as shown in FIG4 .

Each point represents one test data and we show the correlation coefficients between the predicted and labeled moments.

As expected, the regression for all methods deteriorates for the 10 days ahead prediction.

RDRN and DRN have the best regression performance as the points lie closest to the diagonal line.

For the 10 days ahead task, the predicted distributions for RDRN are much better predicted than the other methods, showing RDRN's strength in predicting with longer time steps ahead.

On the other hand, RNN shows some sign of regression to the mean, as the means of the output distributions are limited to a small range about zero.

Neural network models work well by designing the architecture according to the data type.

However, among the conventional neural network architectures, there is none that is designed for time-varying probability distributions.

There are two key challenges in learning from distribution sequences.

First, we require a suitable representation for probability distributions.

Conventional neural networks, however, do not have suitable representations for distributions.

As each node encodes only a real value, the distribution has to be split into smaller parts which are then represented by independent nodes.

Hence, the neural network is agnostic to the distribution nature of the input data.

A recently proposed Distribution Regression Network (DRN) addresses this issue.

DRN has a novel network representation where each node encodes a distribution, showing improved accuracies compared to neural networks.

However, a second challenge remains, which is to model the time dependencies in the distribution sequence.

Both the recurrent neural network (RNN) and the Distribution Regression Network address only either one of the challenges.

In this work, we propose our Recurrent Distribution Regression Network (RDRN) which extends DRN with a recurrent architecture.

By having an explicit distribution representation in each node and shared weights across time steps, RDRN performs forward prediction on distribution sequences most effectively, achieving better prediction accuracies than RNN, DRN and other regression methods.

@highlight

We propose an efficient recurrent network model for forward prediction on time-varying distributions.

@highlight

This paper proposes a method for creating neural nets that maps historical distributions onto distributions and applies the method to several distribution prediction tasks.

@highlight

Proposes a Reccurent Distribution Regression Network which uses a recurrent architecture upon a previous model Distribution Regression Network.

@highlight

This paper is on regressing over probability distributions by studying time varying distributions in a recurrent neural network setting