We propose Significance-Offset Convolutional Neural Network, a deep convolutional network architecture for regression of multivariate asynchronous time series.

The model is inspired by standard autoregressive (AR) models and gating mechanisms used in recurrent neural networks.

It involves an AR-like weighting system, where the final predictor is obtained as a weighted sum of adjusted regressors, while the weights are data-dependent functions learnt through a convolutional network.

The architecture was designed for applications on asynchronous time series and is evaluated on such datasets: a hedge fund proprietary dataset of over 2 million quotes for a credit derivative index, an artificially generated noisy autoregressive  series  and  household  electricity  consumption  dataset.

The  pro-posed architecture achieves promising results as compared to convolutional and recurrent neural networks.

The code for the numerical experiments and the architecture implementation will be shared online to make the research reproducible.

Time series forecasting is focused on modeling the predictors of future values of time series given their past.

As in many cases the relationship between past and future observations is not deterministic, this amounts to expressing the conditional probability distribution as a function of the past observations: p(X t+d |X t , X t−1 , . . .) = f (X t , X t−1 , . .

.).This forecasting problem has been approached almost independently by econometrics and machine learning communities.

In this paper we examine the capabilities of convolutional neural networks (CNNs), BID25 in modeling the conditional mean of the distribution of future observations; in other words, the problem of autoregression.

We focus on time series with multivariate and noisy signal.

In particular, we work with financial data which has received limited public attention from the deep learning community and for which nonparametric methods are not commonly applied.

Financial time series are particularly challenging to predict due to their low signal-to-noise ratio (cf.

applications of Random Matrix Theory in econophysics BID24 BID3 ) and heavy-tailed distributions BID8 .

Moreover, the predictability of financial market returns remains an open problem and is discussed in many publications (cf.

efficient market hypothesis of BID11 ).A common situation with financial data is that the same signal (e.g. value of an asset) is observed from different sources (e.g. financial news, analysts, portfolio managers in hedge funds, marketmakers in investment banks) in asynchronous moments of time.

Each of these sources may have a different bias and noise with respect to the original signal that needs to be recovered (cf.

time series in FIG0 ).

Moreover, these sources are usually strongly correlated and lead-lag relationships are possible (e.g. a market-maker with more clients can update its view more frequently and precisely than one with fewer clients).

Therefore, the significance of each of the available past observations might be dependent on some other factors that can change in time.

Hence, the traditional econometric models such as AR, VAR, VARMA (Hamilton, 1994) might not be sufficient.

Yet their relatively good performance motivates coupling such linear models with deep neural networks that are capable of learning highly nonlinear relationships.

Quotes from four different market participants (sources) for the same CDS 1 throughout one day.

Each trader displays from time to time the prices for which he offers to buy (bid) and sell (ask) the underlying CDS.

The filled area marks the difference between the best sell and buy offers (spread) at each time.

For these reasons, we propose SignificanceOffset Convolutional Neural Network, a Convolutional Network extension of standard autoregressive models BID34 BID35 equipped with a nonlinear weighting mechanism and provide empirical evidence on its competitiveness with standard multilayer CNN and recurrent Long-Short Term Memory network BID18 .

The mechanism is inspired by the gating systems that proved successful in recurrent neural networks BID18 BID6 and highway networks BID37 .2 RELATED WORK 2.1 TIME SERIES FORECASTING Literature in time series forecasting is rich and has a long history in the field of econometrics which makes extensive use of linear stochastic models such as AR, ARIMA and GARCH processes to mention a few.

Unlike in machine learning, research in econometrics is more focused on explaining variables rather than improving out-of-sample prediction power.

In practice, one can notice that these models 'over-fit' on financial time series: their parameters are unstable and out-of-sample performance is poor.

Reading through recent proceedings of the main machine learning venues (e.g. ICML, NIPS, AIS-TATS, UAI), one can notice that time series are often forecast using Gaussian processes BID31 BID38 BID19 , especially when time series are irregularly sampled BID9 BID26 .

Though still largely independent, researchers have started to "bring together the machine learning and econometrics communities" by building on top of their respective fundamental models yielding to, for example, the Gaussian Copula Process Volatility model BID42 .

Our paper is in line with this emerging trend by coupling AR models and neural networks.

Over the past 5 years, deep neural networks have surpassed results from most of the existing literature in many fields BID33 : computer vision BID23 , audio signal processing and speech recognition BID32 , natural language processing (NLP) BID1 BID7 BID14 BID21 .

Although sequence modeling in NLP, i.e. prediction of the next character or word, is related to our forecasting problem (1), the nature of the sequences is too dissimilar to allow using the same cost functions and architectures.

Same applies to the adversarial training proposed by BID28 for video frame prediciton, as such approach favors most plausible scenarios rather than outputs close to all possible outputs, while the latter is usually required in financial time series due to stochasticity of the considered processes.

Literature on deep learning for time series forecasting is still scarce (cf.

BID12 for a recent review).

Literature on deep learning for financial time series forecasting is even scarcer though interest in using neural networks for financial predictions is not new BID30 BID29 .

More recent papers include BID36 that used 4-layer perceptrons in modeling price change distributions in Limit Order Books, and BID2 who applied more recent WaveNet architecture of van den BID39 to several short univariate and bivariate time-series (including financial ones).

Despite claim of applying deep learning, BID17 use autoencoders with a single hidden layer to compress multivariate financial data.

Besides these and claims of secretive hedge funds (it can be marketing surfing on the deep learning hype), no promising results or innovative architectures were publicly published so far, to the best of our knowledge.

In this paper, we investigate the gold standard architectures' (simple Convolutional Neural Network (CNN), Residual Network, multi-layer LSTM) capabilities on AR-like artificial asynchronous and noisy time series, and on real financial data from the credit default swap market where some inefficiencies may exist, i.e. time series may not be totally random.

Gating mechanisms for neural networks were first proposed by BID18 and proved essential in training recurrent architectures BID21 due to their ability to overcome the problem of vanishing gradient.

In general, they can be expressed as DISPLAYFORM0 where f is the output function, c is a 'candidate output' (usually a nonlinear function of x), ⊗ is an element-wise matrix product and σ : R → [0, 1] is a sigmoid nonlinearity that controls the amount of the output passed to the next layer (or to further operations within a layer).

Appropriate compositions of functions of type 2 lead to the popular recurrent architectures such as LSTM BID18 and GRU BID6 .A similar idea was recently used in construction of highway networks BID37 which enabled successful training of deeper architectures.

van den BID40 and BID10 proposed gating mechanisms (respectively with hyperbolic tangent and linear 'candidate outputs') for training deep convolutional neural networks.

The gating system that we propose is aimed at weighting a number of different 'candidate predictors' and therefore is most closely related to the softmax gating used in MuFuRU (Multi-Function Recurrent Unit, BID41 ), i.e. DISPLAYFORM1 where ( The idea of weighting outputs of the intermediate layers within a neural networks is also used in attention networks (See e.g. BID4 ) that proved successful in such tasks as image captioning and machine translation.

Our approach is similar as the separate inputs (time series steps) are weighted in accordance with learned functions of these inputs, yet different since we model these functions as multi-layer CNNs (instead of projections on learned directions) and since we do not use recurrent layers.

The latter is important in the above mentioned tasks as it enables the network to remember the parts of the sentence/image already translated/described.

Time series observed in irregular moments of time make up significant challenges for learning algorithms.

Gaussian processes provide useful theoretical framework capable of handling asynchronous data; however, due to assumed Gaussianity they are inappropriate for financial datasets, which often follow fat-tailed distributions BID8 ).

On the other hand, prediction of even a simple autoregressive time series such us AR(2) given by X(t) = αX(t − 1) + βX(t − 2) + ε(t) 2 may involve highly nonlinear functions when sampled irregularly.

Precisely, it can be shown that the conditional expectation DISPLAYFORM0 where a k and b k are rational functions of α and β (See Appendix A for the proof).

This would not be a problem if k was fixed, as then one would be interested in estimating of a k and b k directly; this, however, is not the case with asynchronous sampling.

When X is an autoregressive series of higher order and more past observations are available, the analogous expectation E[X(t n )|{X(t n−m ), m = 1, . . .

, M }] would involve more complicated functions that in general may not possess closed forms.

In real-world applications we often deal with multivariate time series whose dimensions are observed separately and asynchronously.

This adds even more difficulty to assigning appropriate weights to the past values, even if the underlying data structure is linear.

Furthermore, appropriate representation of such series might be not obvious as aligning such series at fixed frequency may lead to loss of information (if too low frequency is chosen) or prohibitive enlargement of the dataset (especially when durations have varying magnitudes), see Figure 2A .

As an alternative, we might consider representing separate dimensions as a single one with dimension and duration indicators as additional features.

Figure 2B presents this approach, which is going to be at the core of the proposed architecture.

DISPLAYFORM1 Figure 2: (A) Fixed sampling frequency and it's drawbacks; keeping all available information leads to much more datapoints.

(B) Proposed data representation for the asynchronous series.

Consecutive observations are stored together as a single value series, regardless of which series they belong to; this information, however, is stored in indicator features, alongside durations between observations.

A natural model for prediction of such series could be an LSTM, which, given consecutive input values and respective durations (X(t n ), t n − t n−1 ) =: x n in each step would memorize the series values and weight them at the output according to the durations.

However, the drawback of such approach lies in imbalance between the needs for memory and for nonlinearity: the weights that such network needs to assign to the memorized observations potentially require several layers of nonlinearity to be computed properly, while past observations might just need to be memorized as they are.

For these reasons we shall consider a model that combines simple autoregressive approach with neural network in order to allow learning meaningful data-dependent weights DISPLAYFORM2 are modeled using neural network.

To allow more flexibility and cover situations when e.g. observed values of x are biased, we should consider the summation over terms α m (x n−m ) · f (x n−m ), where f is also a neural network.

We formalize this idea in Section 4.

Suppose that we are given a multivariate time series (x n ) n ⊂ R d and we aim to predict the conditional future values of a subset of elements of x n DISPLAYFORM0 where DISPLAYFORM1 .

We consider the following estimator of y n DISPLAYFORM2 where DISPLAYFORM3 • σ is a normalized activation function independent on each row, i.e. DISPLAYFORM4 for any a 1 , . . .

, a d I ∈ R M and σ such that σ(a) DISPLAYFORM5 • ⊗ is Hadamard (element-wise) matrix multiplication.

The summation in 7 goes over the columns of the matrix in bracket; hence the i-th element of the output vectorŷ n is a linear combination of the i-th row of the matrix F (x −M n ).

We are going to consider S to be a fully convolutional network (composed solely of convolutional layers) and F of the form DISPLAYFORM6 where W ∈ R d I ×M and off : R d → R d I is a multilayer perceptron.

In that case F can be seen as a sum of projection (x → x I ) and a convolutional network with all kernels of length 1.

Equation (7) can be rewritten asŷ DISPLAYFORM7 where W m , S m (·) are m-th columns of matrices W and S(·).Figure 3: A scheme of the proposed SOCNN architecture.

The network preserves the timedimension up to the top layer, while the number of features per timestep (filters) in the hidden layers is custom.

The last convolutional layer, however, has the number of filters equal to dimension of the output.

The Weighting frame shows how outputs from offset and significance networks are combined in accordance with Eq. 10.We will call the proposed network a Significance-Offset Convolutional Neural Network (SOCNN), while off and S respectively the offset and significance (sub)networks.

The network scheme is shown in Figure 3 .

Note that when off ≡ 0 and σ ≡ 1 the model simplifies to the collection of d I separate AR(M ) models for each dimension.

Note that the form of Equation FORMULA0 enforces the separation of temporal dependence (obtained in weights W m ), the local significance of observations S m (S as a convolutional network is determined by its filters which capture local dependencies and are independent on the relative position in time) and the predictors off(x n−m ) that are completely independent on position in time.

This provides some amount of interpretability of the fitted functions and weights.

For instance, each of the past observations provides an adjusted single regressor for the target variable through the offset network.

Note that due to asynchronous sampling procedure, consecutive values of x might be heterogenous, hence On the other hand, significance network provides data-dependent weights for all regressors and sums them up in autoregressive manner.

Figure 7 in Appendix E.2 shows sample significance and offset activations for the trained network.

Relation to asynchronous data As mentioned before, one of the common problems with time series are the varying durations between consecutive observations.

A simple approach at data-preprocessing level is aligning the observations at some chosen frequency by e.g. duplicating or interpolating observations.

This, however, might extend the size of an input and, therefore, model complexity.

The other idea is to treat the duration and/or time of the observation as another feature, as presented in Figure 2B .

This approach is at the core of the SOCNN architecture: the significance network is aimed at learning the high-level features that indicate the relative importance of past observations, which, as shown in Section 3, could be predominantly dependent on time and duration between observations.

Loss function L 2 error is a natural loss function for the estimators of expected value DISPLAYFORM0 As mentioned above, the output of the offset network can be seen as a collection of separate predictors of the changes between corresponding observations x I n−m and the target variable y n off(x n−m ) y n − x I n−m .For that reason, we consider the auxiliary loss function equal to mean squared error of such intermediate predictions DISPLAYFORM1 The total loss for the sample (x DISPLAYFORM2 where y n is given by Eq. 10 and α ≥ 0 is a constant.

In Section 5.2 we discuss the empirical findings on the impact of positive values of α on the model training and performance, as compared to α = 0 (lack of auxiliary loss).

We evaluate the proposed model on a financial dataset of bid/ask quotes sent by several market participants active in the credit derivatives market, artificially generated datasets and household electric power consumption dataset available from UCI repository BID27 , comparing its performance with simple CNN, single-and multi-layer LSTM BID18 and 25-layer ResNet BID16 .Apart from performance evaluation of SOCNNs, we discuss the impact of the network components, such as auxiliary loss and the depth of the offset sub-network.

The details of the training process and hyperparameters used in the proposed architecture as well as in benchmark models are presented in C.

We test our network architecture on the artificially generated datasets of multivariate time series.

We consider two types of series:1.

Synchronous series.

The series of K noisy copies ('sources') of the same univariate autoregressive series ('base series'), observed together at random times.

The noise of each copy is of different type.2.

Asynchronous series.

The series of observations of one of the sources in the above dataset.

At each time, the source is selected randomly and its value at this time is added to form a new univariate series.

The final series is composed of this series, the durations between random times and the indicators of the 'available source' at each time.

The details of the simulation process are presented in Appendix D. We consider synchronous and asynchronous series X K×N where K ∈ {16, 64} is the number of sources and N = 10, 000, which gives 4 artificial series in total 3 .

The household electricity dataset 4 contains measurements of 7 different quantities related to electricity consumption in a single household, recorded every minute for 47 months, yielding over 2 million observations.

Since we aim to focus on asynchronous time-series, we alter it so that a single observation contains only value of one of the seven features, while durations between consecutive observations range from 1 to 7 minutes 5 .The regression aim is to predict all of the features at the next time step.

The proposed model was designed primarily for forecasting incoming non-anonymous quotes received from the credit default swap market.

The dataset contains 2.1 million quotes from 28 different sources, i.e. market participants.

Each quote is characterized by 31 features: the offered price, 28 indicators of the quoting source, the direction indicator (the quote refers to either a buy or a sell offer) and duration from the previous quote.

For each source and direction we aim at predicting the next quoted price from this given source and direction considering the last 60 quotes.

We formed 15 separate prediction tasks; in each task the model was trained to predict the next quote by one of the fifteen most active market participants 6 .This dataset, which is proprietary, motivated the aforementioned construction of artificial asynchronous time series datasets based on its statistical features for reproducible research purpose.

TAB0 presents the detailed results from the artificial and electricity datasets.

The proposed networks outperform significantly the benchmark networks on the asynchronous, electricity and quotes datasets.

For the synchronous datasets, on the other hand, SOCNN almost matches the results of the benchmarks.

This similar performance could have been anticipated -the correct weights of the past values in synchronous artificial datasets are far less nonlinear than in case when separate dimensions are observed asynchronously.

For this reason, the significance network's potential is not fully utilized.

We can also observe that the depth of the offset network has negligible or negative This means that the significance network has crucial impact on the performance, which is in-line with the potential drawbacks of the LSTM network discussed in Section 3: obtaining proper weights for the past observations is much more challenging than getting good predictors from the single past values.

The small positive auxiliary weight helped achieve more stable test error throughout training in many cases.

The higher weights of auxiliary loss considerably improved the test error on asynchronous datasets (See TAB1 ); however for other datasets its impact was negligible.

In general, the proposed SOCNN had significantly lower variance of the test and validation errors, especially in the early stage of the training process and for quotes dataset.

Figure 4 presents the learning curves for two different artificial datasets.

To understand better why SOCNN obtained better results than the other networks, we check how these networks react to the presence of additional noise in the input terms 8 .

Figure 5 presents changes in the mean squared error and significance and offset network outputs with respect to the level of noise.

SOCNN is the most robust out of the compared networks and, together with singlelayer LSTM, least prone to overfitting.

Despite the use of dropout and cross-validation, the other models tend to overfit the training set and have non-symmetric error curves on test dataset.

Figure 5 : Experiment comparing robustness of the considered networks for Asynchronous 16 dataset.

The plots show how the error would change if an additional noise term was added to the input series.

The dotted curves show the total significance and average absolute offset (not to scale) outputs for the noisy observations.

Interestingly, significance of the noisy observations increases with the magnitude of noise; i.e. noisy observations are far from being discarded by SOCNN.

In this article, we proposed a weighting mechanism that, coupled with convolutional networks, forms a new neural network architecture for time series prediction.

The proposed architecture is designed for regression tasks on asynchronous signals in the presence of high amount of noise.

This approach has proved to be successful in forecasting financial and artificially generated asynchronous time series outperforming popular convolutional and recurrent networks.

The proposed model can be further extended by adding intermediate weighting layers of the same type in the network structure.

Another possible generalization that requires further empirical studies can be obtained by leaving the assumption of independent offset values for each past observation, i.e. considering not only 1x1 convolutional kernels in the offset sub-network.

Finally, we aim at testing the performance of the proposed architecture on other real-life datasets with relevant characteristics.

We observe that there exists a strong need for common 'econometric' datasets benchmark and, more generally, for time series (stochastic processes) regression.

APPENDIX A NONLINEARITY IN THE ASYNCHRONOUSLY SAMPLED AUTOREGRESSIVE TIME SERIES Lemma 1.

Let X(t) be an AR(2) time series given by DISPLAYFORM0 where (ε(t)) t=1,2,... are i.i.d.

error terms.

Then DISPLAYFORM1 for any t > k ≥ 2, where a k , b k are rational functions of a and b.

Proof.

The proof follows a simple induction.

It is sufficient to show that DISPLAYFORM2 where DISPLAYFORM3 and E k (t) is a linear combination of {ε(t − i), i = 0, 1, . . .

, k − 2}. Basis of the induction is trivially satisfied via 15.

In the induction step, we assume that 17 holds for k. For t > k + 1 we have DISPLAYFORM4 .

Multiplying sides of this equation by b and adding av k X(t − 1) we obtain DISPLAYFORM5 Since aX(t − 1) + bX(t − 2) = X(t) − ε(t) we get DISPLAYFORM6 As DISPLAYFORM7 is a linear combination of {ε(t − i), i = 0, 1, . . .

, k − 1}, the above equation proves 17 for k = k + 1.

To see how robust each of the networks is, we add noise terms to part of the input series and evaluate them on such datapoints, assuming unchanged output.

We consider varying magnitude of the noise terms, which are added only to the selected 20% of past steps at the value dimension 9 .

Formally the procedure is following:1.

Select randomly N obs = 6000 observations (X n , y n ) (half of which coming from training set and half from test set).2.

Add noise terms to the observations X n p := X n + Ξ n · γ p , for {γ p } 128 p=1 evenly distributed on [−6σ, 6σ] , where σ is a standard deviation of the differences of the series being predicted and DISPLAYFORM0 3.

For each p evaluate each of the trained models on dataset X n p , y n N obs

, separately for n's originally coming from training and test sets.

To evaluate the model and the significance of its components, we perform a grid search over some of the hyperparameters, more extensively on the artificial and electricity datasets.

These include the offset sub-network's depth (we consider depths of 1 and 10 for artificial and electricity datasets; 1 for Quotes data) and the auxiliary weight α (compared values: {0, 0.1, 0.01}).

For all networks we have chosen LeakyReLU activation function (23) DISPLAYFORM0 with leak rate a = .1 as an activation function.

We compare the performance of the proposed model with CNN, ResNet, multi-layer LSTM networks and linear (VAR) model.

The benchmark networks were designed so that they have a comparable number of parameters as the proposed model.

Consequently, LeakyReLU activation function (23) with leak rate .1 was used in all layers except the top ones where linear activation was applied.

For CNN we provided the same number of layers, same stride (1) and similar kernel size structure.

In each trained CNN, we applied max pooling with the pool size of 2 every two convolutional layers 10 .

TAB3 presents the configurations of the network hyperparameters used in comparison.

The training and validation sets were sampled randomly from the first 80% of timesteps in each series, with ratio 3 to 1.

The remaining 20% of data was used as a test set.

All models were trained using Adam optimizer BID22 which we found much faster than standard Stochastic Gradient Descent in early tests.

We used batch size of 128 for artificial data and 256 for quotes dataset.

We also applied batch normalization BID20 in between each convolution and the following activation.

At the beginning of each epoch, the training samples were shuffled.

To prevent overfitting we applied dropout and early stopping 12 .

Weights were initialized following the normalized uniform procedure proposed by BID13 .

Experiments were carried out using implementation relying on Tensorflow BID0 and Keras front end BID5 .

For artificial data we optimized the models using one K20s NVIDIA GPU while for quotes dataset we used 8-core Intel Core i7-6700 CPU machine only.

We simulate a multivariate time series composed of K noisy observations of the same autoregressive signal.

The simulated series are constructed as follows:Let p 1 , p 2 , . . .

, p K ∈ (0, 1) and define Figure 6: Simulated synchronous (left) and asynchronous (right) artificial series.

Note the different durations between the observations from different sources in the latter plot.

For clarity, we present only 6 out of 16 total dimensions.

DISPLAYFORM0 DISPLAYFORM1 We call {X t } N t=1 and {X t } N t=1 synchronous and asynchronous time series, respectively.

We simulate both of the processes for N = 10, 000 and each K ∈ {16, 64}.

The original dataset has 7 features: global active power, global reactive power, voltage, global intensity, sub-metering 1, sub-metering 2 and sub-metering 3, as well as information on date and time.

We created asynchronous version of this dataset in two steps:1.

Deterministic time step sampling.

The durations between the consecutive observations are periodic and follow a scheme [1min, 2min, 3min, 7min, 2min, 2min, 4min, 1min, 2min, 1min] ; the original observations in between are discarded.

In other words, if the original observations are indexed according to time (in minutes) elapsed since the first observation, we keep the observations at indices n such that n mod 25 ≡ k ∈ [0, 1, 3, 6, 13, 15, 17, 21, 22, 24] .2.

Random feature sampling.

At each remaining time step, we choose one out of seven features that will be available at this step.

The probabilities of the features were chosen to be proportional to [1, 1.5, 1.5 2 , 1.5 6 ] and randomly assigned to each feature before sampling (so that each feature has constant probability of its value being available at each time step.

At each time step the sub-sampled dataset is 10-dimensional vector that consists information about the time, date, 7 indicator features that imply which feature is available, and the value of this feature.

The length of the sub-sampled dataset is above 800 thousand, i.e. 40% of the original dataset's length.

The schedule of the sampled timesteps and available features is attached in the data folder in the supplementary material.

In Figure 7 we present significance and offset activations for three input series, from the network trained on electricity dataset.

Each row represents activations corresponding to past values of a single feature.

Figure 7 : Activations of the significance and offset sub-networks for the network trained on Electricity dataset.

We present 25 most recent out of 60 past values included in the input data, for 3 separate datapoints.

Note the log scale on the left graph.

<|TLDR|>

@highlight

Convolutional architecture for learning data-dependent weights for autoregressive forecasting of time series.