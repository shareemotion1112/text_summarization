Time series forecasting plays a crucial role in marketing, finance and many other quantitative fields.

A large amount of methodologies has been developed on this topic, including ARIMA, Holt–Winters, etc.

However, their performance is easily undermined by the existence of change points and anomaly points, two structures commonly observed in real data, but rarely considered in the aforementioned methods.

In this paper, we propose a novel state space time series model, with the capability to capture the structure of change points and anomaly points, as well as trend and seasonality.

To infer all the hidden variables, we develop a Bayesian framework, which is able to obtain distributions and forecasting intervals for time series forecasting, with provable theoretical properties.

For implementation, an iterative algorithm with Markov chain Monte Carlo (MCMC), Kalman filter and Kalman smoothing is proposed.

In both synthetic data and real data applications, our methodology yields a better performance in time series forecasting compared with existing methods, along with more accurate change point detection and anomaly detection.

Time series forecasting has a rich and luminous history, and is essentially important in most of business operations nowadays.

The main aim of time series forecasting is to carefully collect and rigorously study the past observations of a time series to develop an appropriate model which could describe the inherent structure of the series, in order to generate future values.

For instance, the internet companies are interested in the number of daily active users (DAU) , say, what is DAU after certain period of time, or when will reach their target DAU goal.

Time series forecasting is a fruitful research area with many existing methodologies.

The most popular and frequently used time series model might be the Autoregressive Integrated Moving Average (ARIMA) BID4 BID31 BID8 BID16 .

Taking seasonality into consideration, BID4 proposed the Seasonal ARIMA.

The Holt-Winters method BID30 ) is also very popular by using exponential smoothing.

State space model BID10 BID24 BID5 also attracts much attention, which is a linear function of an underlying Markov process plus additive noise.

Exponential Smoothing State Space Model (ETS) decomposes times series into error, trend, seasonal that change over time.

Recently, deep learning is applied for time-series trend learning using LSTM BID26 , bidirectional dynamic Boltzmann machine BID23 is applied for time-series long-term dependency learning, and coherent probabilistic forecast BID25 ) is proposed for a hierarchy or an aggregation-level comprising a set of time series.

Orthogonal to these works, this paper focuses on robust ways of time series forecasting in presence of change points and anomalies.

In Internet time series forecasting, Google develops the Bayesian structure time series (BSTS) model BID5 BID24 to capture the trend, seasonality, and similar components of the target series.

Recently, Facebook proposes the Prophet approach BID27 based on a decomposable model with interpretable parameters that can be intuitively adjusted by analyst.

However, as in the DAU example, some special events like Christmas Holiday or President Election, newly launched apps or features, may cause short period or long-term change of DAU, leading to weird forecasting of those traditional models.

The aforementioned special cases are well known as• Anomaly points.

The items, events or observations that don't conform to an expected pattern or other items in the dataset, leading to a sudden spike or decrease in the series.• Change points.

A market intervention, such as a new product launch or the onset of an advertising (or ad) campaign, may lead to the level change of the original series.

Time series forecasting without change/anomaly point detection and adjustment may also lead to bizarre forecasting since these models might learn the abrupt changes in the past.

There are literatures on detecting anomaly or change points individually, examples can be found in BID29 ; BID22 ; BID3 ; BID21 BID29 .

However, the aforementioned change point detection models could not support detection in the presence of seasonality, while the presence of trend/change point is not handled by the anomaly detection models.

Most importantly, there is a discrepancy between anomaly/change points detection and adjustment, and commonly used manually adjustment might be a bit arbitrary.

Unfortunately, the forecasting gap caused by abnormal and change points, to the best of our knowledge, has not been given full attention and no good solution has been found so far.

This paper is strongly motivated by bridging this gap.

In this paper, to overcome the limitations of the most (if not all) current models that the anomaly points and change points are not properly considered, we develop a state space time series forecasting model in the Bayesian framework that can simultaneously detect anomaly and change points and perform forecasting.

The learned structure information related to anomaly and change points is automatically incorporated into the forecasting process, which naturally enhances the model prediction based on the feedback of state-space model.

To solve the resultant optimization problem, an iterative algorithm based on Bayesian approximate inference with Markov chain Monte Carlo (MCMC), Kalman filter and Kalman smoothing is proposed.

The novel model could explicitly capture the structure of change points, anomaly points, trend and seasonality, as also provide the distributions and forecasting intervals due to Bayesian forecasting framework.

Both synthetic and real data sets show the better performance of proposed model, in comparison with existing baseline.

Moreover, our proposed model outperforms state-of-the-art models in identifying anomaly and change points.

To summarize, our work has the following contributions.• We proposed a robust 1 Bayesian state-space time series forecasting model that is able to explicitly capture the structures of change points and anomalies (which are generally ignored in most current models), and therefore automatically adapt for forecasting by incorporating the prior information of trend, seasonality, as well as change points and anomalies using state space modeling.

Due to the enhancement of model description capability, the results of model prediction and abnormal and change points detection are mutually improved.• To solve the resultant optimization problem, an effective algorithm based on approximate inference using Markov chain Monte Carlo (MCMC) is proposed with theoretical guaranteed forecasting paths.• Our proposed method outperforms the state-of-the-art methods in time series forecasting in presence of change points and anomalies, and detects change points and anomalies with high accuracy and low false discovery rate on both tasks, outperforming popular change point and anomaly detection methods.

Our method is flexible to capture the structure of time series under various scenarios with any component combinations of trend, seasonality, change points and anomalies.

Therefore our method can be applied in many settings in practice.

State space time series model BID13 has been one of the most popular models in time series analysis.

It is capable of fitting complicated time series structure including linear trend and seasonality.

However, times series observed in real life are almost all prevailed with outliers.

Change points, less in frequency but are still widely observed in real time series analysis.

Unfortunately, both structures are ignored in the classic state space time series model.

In the section, we aim to address this issue by introducing a novel state space time series model.

Let y = (y 1 , y 2 , . . .

, y n ) be a sequence of time series observations with length n. The ultimate goal is to forecast (y n+1 , y n+2 , . . .).

The accuracy in forecasting lies in a successful decomposition of y into existing components.

Apart from the residuals, we assume the time series is composed by trend, seasonality, change points and anomaly points.

In a nutshell, we have an additive model with time series = trend + seasonality + change point + anomaly point + residual.

As the classical state space model, we have observation equation and transition equations to model y and hidden variables.

We use µ = (µ 1 , µ 2 , . . .

, µ n ) to model trend, and use γ = (γ 1 , γ 2 , . . .

, γ n ) to model seasonality.

We use a binary vector z a = (z a 1 , z a 2 , . . .

, z a n ) to indicate anomaly points.

Then we have Observation equation: DISPLAYFORM0 The deviation between the observation y t and its "mean" µ t + γ t is modeled by t and o t , depending on the value of z a t .

If z a t = 1, then y t is an anomaly point; otherwise it is not.

Distinguished from the residues = ( 1 , 2 , . . .

, n ), the anomaly is captured by o = (o 1 , o 2 , . . . , o n ) which has relative large magnitude.

The hidden state variable µ and γ have intrinsic structures.

There are two transition equations, for trend and seasonality separately Transition Equations: Trend: DISPLAYFORM1 DISPLAYFORM2 In Equation (2), δ = (δ 1 , δ 2 , . . .

, δ n ) can be viewed as the "slope" of the trend, measuring how fast the trend changes over time.

The change point component is also incorporated in Equation (2) by a binary vector z c = (z is one of the five additive components along with trend, seasonality, anomaly points and residuals.

Here we model the change point directly into the trend component.

Though differing in formulation, they are equivalent to each other.

We choose to model in as in Equation FORMULA1 due to simplicity, and its similarity with the definition of anomaly points in Equation (1).The seasonality component is presented in Equation (3).

Here S is the length of one season and w = (w 1 , w 2 , . . . , w n ) is the noise for seasonality.

The seasonality component is assumed to have almost zero average in each season.

The observation equation and transition equations (i.e., Equation (1,2,3)) define how y is generated from all the hidden variables including change points and anomaly points.

We continue to explore this new model, under a Bayesian framework.

Bayesian methods are widely used in many data analysis fields.

It is easy to implement and interpret, and it also has the ability to produce posterior distribution.

The Bayesian method on state space time series model has been investigated in BID24 BID5 .

In this section, we also consider Bayesian framework for our novel state space time series model.

We assume all the noises are normally distributed DISPLAYFORM0 where σ , σ o , σ u , σ r , σ v , σ w are parameters for standard deviation.

As binary vectors, a natural choice is to model anomaly point indicator z a and change point indicator z c to the model them as Bernoulli random variables DISPLAYFORM1 where p a , p c are probabilities for each point to be an anomaly or change point.

For simplicity, we denote α t = (µ t , δ t , γ t , γ t−1 , . . . , γ t−(S−2) ) to include the main hidden variables (except z a t and z c t ) in the transition equations.

All the α t are well defined and can be generated from the previous status, except α 1 .

We denote a 1 to be the parameter for α 1 , which can be interpreted as the "mean" for α 1 .With Bayesian framework, we are able to represent our model graphically as in FIG2 .

As shown in FIG2 , the only observations are y and all the others are hidden.

In this paper, we assume there is no additional information on all the hidden states.

If we have some prior information, for example, some points are more likely to be change points, then our model can be easily modified to incorporate such information, by using proper prior.

In FIG2 , we use squares and circles to classify unknown variables.

Despite all being unknown, they actually behave differently according to their own functionality.

For those in squares, they behave like turning parameters.

Once they are initialized or given, those in circles behaves like latent variables.

We call the former "parameters" and the latter "latent variable", as listed in TAB0 .

The "mean" for the initial trend and seasonality p = (pa, pc)Probabilities for each point to be anomaly or change point σ = (σ , σo, σu, σr, σv, σw) Standard deviationThe discrepancy between these two categories is clearly captured by the joint likelihood function.

From FIG2 , the joint distribution (i.e., the likelihood function) can be written down explicitly as DISPLAYFORM2 2 ) is the density function for normal distribution with mean x 1 and standard deviation x 2 .

Here we slightly abuse the notation by using µ 0 , δ 0 , γ 0 , γ −1 , . . .

, γ 2−S , which are actually the corresponding coordinates of a 1 .As long with other probabilistic graphical models, our model can also be viewed as a generative model.

Given the parameters a 1 , p, σ, we are able to generate time series.

We present the generative procedure as follows.

Algorithm 1: Generative Procedure Input: Parameters a 1 , σ = (σ , σ o , σ u , σ r , σ v , σ w ) and p a , p c , length of time series to generate m Output: Time series y = (y 1 , y 2 , . . .

, y m ) 1 Generate the indexes where anomalies or change points occur DISPLAYFORM3 2 Generate all the noises , o, u, r, v, w as independent normal random variables with mean zero and standard deviation σ , σ o , σ u , σ r , σ v , σ w respectively; 3 Generate {α t } m t=1 sequentially by the transition functions in Equation FORMULA1

This section is about inferring unknown variables from y, given the Bayesian setting described in the previous section.

The main framework here is to sequentially update each hidden variable by fixing the remaining ones.

As stated in the previous section, there are two different categories of unknown variables.

Different update schemes need to be used due to the difference in their functionality.

For the latent variables, we implement Markov chain Monte Carlo (MCMC) for inference.

Particular, we use Gibbs sampler.

We will elaborate the details of updates in the following sections.

In this section, we focus on updating α assuming all the other hidden variables are given and fixed.

The essence of Gibbs sampler is to obtain posterior distribution p a1,p,σ (α|y, z).

This can be achieved by a combination of Kalman filter, Kalman smoothing and the so-called "fake-path" trick.

We provide some intuitive explanation here and refer the readers to BID10 for detailed implementation.

Kalman filter and Kalman smoothing are classic algorithms in signal processing and pattern recolonization for Bayesian inference.

It is well related to other algorithms especially message passing algorithm.

Kalman filter collects information forwards to obtain E(α t |y 1 , y 2 , . . .

, y t ); while Kalman smoothing distribute information backwards to achieve E(α t |y).However, the combination of Kalman filter and Kalman smoothing is not enough, as it only gives the the expectations of marginal distributions {E(α t |y)} n t=1 , instead of the joint distribution required for Gibbs sampler.

To address this issue, we can use the "fake-path" trick described in Brodersen et al. FORMULA0 ; BID10 .

The main idea underlying this trick lies on the fact that the covariance structure of p(α t |y) is not dependent on the means.

If we are able to obtain the covariance by some other way, then we can add it up with {E(α t |y)} n t=1 to obtain a sample from p(α|y).

This trick involves three steps.

Note that all the other hidden variables z, p, σ are given.1.

Pick some vectorã 1 , and generate a sequence of time seriesỹ from it by Algorithm 1.

In this way, we also observeα.

2.

Obtain {E(α t |ỹ)} n t=1 fromỹ by Kalman filter and Kalman smoothing.

3.

We use {α t − E(α t |ỹ) + E(α t |y)} n t=1 as our sampling from the conditional distribution.

In this section, we update z by Gibbs sampler, assuming α, a 1 , p, σ are all given and fixed.

We need to obtain the conditional distribution p a1,p,σ (z|y, α).

Note that in the graphical model described in Section 2, {z are still independent Bernoulli random variables, but possibly with different success probabilities.

Thus, we can take the calculation point by point.

For example, for the anomaly detection for the t-th point, we have DISPLAYFORM0 .

And the prior on z a t is P(z a t = 1) = p a and P(z a t = 0) = p 1 .

Let p a t = P(z a t = 1|y, α).

Directly calculation leads to DISPLAYFORM1 This equality holds for all t = 1, 2, . . .

, n. Similarly for change point detection, let p As mentioned above, all the coordinates in z are still independent Bernoulli random variables conditioned on y, α.

Thus, for Gibbs sampler, we can generate z by sampling independently with DISPLAYFORM2 For change point detection here, we have an additional segment control step.

After obtaining {z c t } n t=1 as mentioned above, we need to make sure that the change points detected satisfy some additional requirement on the length of segment among two consecutive change points.

This issue arises from the ambiguity between the definitions of change point and anomaly points.

For example, consider a time series with value (0, 0, 0, 0, 1, 1, 1, 0, 0, 0).

We can view it with two change points, one increases the trend by 1 and the other decreases it by 1.

Alternatively, we can also argue the three 1s in this time series are anomalies, though next to each other.

One way to address this ambiguity is by defining the minimum length of segment (denoted as ).

In this toy example, if we set the minimum length to be 4, then they are anomaly points; if we set it to be 3, then we regard them to be change points.

But a more complicated criterion is needed than using minimum length as the time series usually own much more complex structure than this toy example.

Consider time series FIG0 ) and the minimum time series parameter = 3.

It is reasonable to view it with one change point with increment 1, and the two -1s should be regarded as anomalies.

As a combination of all these factors, we propose the following segment control method.

A default value for the parameter is the length of seasonality, i.e., = S.Algorithm 2: Segment control on change points Input: change point binary vector z c ,trend µ, standard deviation for outliers σ r , change point minimum segment Output: change point binary vector z

The parameters σ, a 1 and p need both initialization and update.

We have different initializations and update schemes for each of them.

For all the standard deviations, once we obtain α and z, we update them by taking the empirical standard deviation correspondingly.

For σ δ and σ γ , the calculation is straightforward as they only involve δ and γ respectively.

For σ , σ o , σ u and σ r , it is a bit more involved due to z. Nevertheless, we can obtain the following update equations for all of them: DISPLAYFORM0 DISPLAYFORM1 Note that in some iterations, when there is no change point or anomaly detected in z, then the updates above for σ o , σ r are not well-defined.

In those cases, we simply let them remain the same.

To initialize σ, we let them all equal to the standard deviation of y.

For a 1 , we initialize it by letting its first coordinate to be equal to the average of y 1 , y 2 , . . .

, y S , and all the remaining coordinates to be equal to 0.

Since a 1 can be interpreted as the mean vector of α 1 , in this way the trend is initialized to be matched up with average of the first season, and the slope and seasonality are initialized to be equal to 0.

We update a 1 by using information of α.

We let the first two coordinates (trend and slope) of a 1 to be equal to those of α 1 , and we let the remaining coordinates (seasonality) of a 1 to be equal to those of α S+1 .

The reason why we do not let a 1 to be equal to α 1 entirely is due to the consideration on convergence and robustness.

Since we initialize the seasonality part in a 1 as 0, it will remain 0 if we let a 1 equals α 1 entirely (due to the mechanism how we update α 1 as described in Section 4.1.

We can avoid such trouble via using α S+1 .For p, we initialize them to be equal to 1/n.

If we have additional information on the number of change points or anomaly points, we can initiate them with different values, for example, 0.1/n, or 10/n.

We can update p after obtaining z, but we choose not to, also for the sake of robustness.

In the early iterations when the algorithm is far from convergence, it is highly possible that z a or z c may turn out to be all 0.

If we update p, say, by taking the proportion of change point or anomaly points in z. Then p a or p c might be 0, and it may get stuck in 0 in the remaining iterations.

Once we infer all the latent variables α, z and tune all the parameters p, a 1 , σ, we are able to forecast the future time series y future .

From the graphical model described in Section 3, the future forecasting only involves α n instead of the whole α.

Note that we assume that there exists no change point and anomaly point in the future.

This is reasonable as in most cases we have no additional information on the future time series.

Given α n and σ we can use our predictive procedures (i.e., Algorithm 1) to generate future time series y future .

We can further integrate out α n to have the posterior predictive distribution as p σ (y future |y).The forecasting on future time series is not deterministic.

There are two sources for the randomness in y future .

One comes from the inference of α n (and also σ) from y. Under the Bayesian framework in Section 3, we have a posterior distribution over α n rather than a single point estimation.

The second one comes from the forecasting function itself.

The forecasting involves intrinsic noise like t , u t , v t and w t .

Thus, the predictive density function p σ (y future |y, α n ) will lead to different path even with fixed σ and α n .

In this way we are able to obtain distribution and predictive interval for forecasting.

We also suggest to take the average of multiple forecasting paths, as the posterior mean for the forecasting.

The average of multiple forecasting paths (denoted asȳ future ), if the number of paths is large enough, always takes the form as a combination of linear trend and seasonality.

This can be observed in both our synthesis data (Section 7) and real data analysis (Section 8).

This seems to be surprising at the first glance, but makes some sense intuitively.

Under our assumption, we have no information on the future, and thus a reliable way to forecast the future is to use the information collected at the end of observed time series, i.e., trend µ n , slope δ n and seasonality structure.

Theorem 1 gives mathematical explanation of the linearity ofȳ future , in both mean and standard deviation.

Theorem 1.

Let N be the number of future time series paths we generate from Algorithm 1).

Let m be the number of points we are going to forecast.

Denote {y to be the future paths.

Defineȳ future = (ȳ n+1 ,ȳ n+2 , . . . ,ȳ n+m ) to be the average such that DISPLAYFORM0 Then for all j = 1, 2, . . .

, N , we haveȳ n+j as a normal distribution with mean and variance as DISPLAYFORM1 Consequently, for all j = 1, 2, . . .

, m, E[ȳ n+j ] is in a linear form with respect to j, and the standard deviation ofȳ n+j also takes a approximately linear form with respect to j.

Proof.

Recall that α n , σ are given and fixed, and we assume there is no change point or anomaly in the future time series.

The Equation (2) leads to δ n+j = δ n + j l=1 v n+l , which implies that DISPLAYFORM2 For the seasonality part, simple linear algebra together with Equation 3 leads to γ n+j = γ n−S+(j mod S) + j l=1 w n+l .

Thus, DISPLAYFORM3 Due to the independence and Gaussian distribution of all the noises,ȳ n+j is also normally distributed and its means and variance can be calculated accordingly.

Our proposed method can be divided into three parts: initialization, inference, and forecasting.

Section 4 and Section 5 provide detailed explanation and reasoning for each of them.

We present a whole picture of our proposed methodology in Algorithm 3.Algorithm 3: Proposed Algorithm Input: Observed time series y = (y 1 , y 2 , . . .

, y n ), seasonality length S, length of time series for forecasting m, number of predictive paths N , change point minimum segment l Output: Change point detection z c , anomaly points z a , forecasting result y future = (y n+1 , y n+1 , . . .

, y n+m ) and its distribution or predictive intervals Part I: Initialization; 1 Initialize σ , σ o , σ u , σ r , σ v , σ w all with the empirical standard deviation of y; 2 Initialize a 1 such that its first coordinate equals to the average of (y 1 , y 2 , . . . , y S ) and all the remaining S coordinates with 0; 3 Initialize p a and p c by 1/n.

Then generate z a and z c as independent Bernoulli random variables with success probability p a and p c respectively; DISPLAYFORM0 Infer α by Kalman filter, Kalman smoothing and "fake-path" trick described in Section 4.1;

Update z a and z c by sampling from DISPLAYFORM0 , where the success probability {p Update σ by Equation FORMULA11 to (8);

Update a 1 such that its first two coordinates equal to the those of α 1 and the remaining (S − 1) coordinates equals to those of α S+1 ;

Calculate the likelihood function L a1,p,σ (y, α, z) given in Equation (4); end Part III: Forecasting; 10 With a n and σ, use the generate procedure in Algorithm 1 to generate future time series y future with length m. Repeat the generative procedure to obtain multiple future paths y DISPLAYFORM0 future , . . .

, yfuture ; 11 Combine all the predictive paths give the distribution for the future time series forecasting.

If needed, calculate the point-wise quantile to obtain predictive intervals.

Use the point-wise average as our final forecasting result.

It is worth mentioning that our proposed methodology is downward compatible with many simpler state space time series models.

By letting p c = 0, we assume there is no change point in the time series.

By letting p a = 0, we assume there is no anomaly point in the time series.

If both p c and p a are set to be 0, then our model is reduced to the classic state space time series model.

Also, the seasonality and slope can be removed from our model, if we know there exists no such structure in the data.

In this section, we study the synthetic data generated from our model.

We let S = 7 and provide values for σ and a 1 .

The change points and anomaly points are randomly generated.

We use our generative procedure (Algorithm 1) to generate time series with total length 500 by fixed parameters.

The first 350 points will be used as training set and the remaining 150 points will be used to evaluate the performance of forecasting.

When generating, we let the time series have weekly seasonality with S = 7.

For σ we have σ = 0.1, σ u = 0.1, σ v = 0.0004, σ w = 0.01, σ r = 1, σ o = 4.

For α 1 we have value for µ as 20, value for δ as 0, and value for seasonality as (1, 2, 4, −1, −3, −2)/10.

For p we have p c = 4/350 and p a = 10/350.

Despite that, to make sure that at least one change point is in existence, we force z c 330 = 1 and r 330 = 2.

That is, for each time series we generate, its 330th point is a change point with the mean shifted up by 3.

Also to be consistence with our assumption, we force z c i = z a i = 0, ∀351 ≤ i ≤ 500 so there exists no change point or anomaly point in the testing part.

The top panel of FIG9 shows one example of synthesis data.

The blue line marks the separation between training and testing set.

The blue dashed line indicates the locations for the change point, while the yellow dots indicate the positions of anomaly points.

Also see FIG9 for illustration on the results returned by implementing our proposed algorithm on the same dataset.

The red line gives the fitting results in the first 350 points and forecasting results in the last 150 points.

The change points detected are marked with vertical red dotted line, and the anomaly detected are flagged with purple squares.

FIG9 shows that on this dataset, our proposed algorithm yields perfect detection on both change points and anomaly points.

In FIG9 , the gray part indicates the 90% predictive interval for forecasting.

We run our generative model 100 times to produce 100 different time series, and implement multiply methods on each of them, and aggregate the results together for comparison.

We include the following methodologies.

For time series forecasting, we compare our method against Bayesian Structural Time Series (BSTS) BID24 BID5 ), Seasonal Decomposition of Time Series by Loess (STL) BID7 ), Seasonal ARIMA BID4 , Holt-Winters (Holt, 2004) , Exponential Smoothing State Space Model (ETS) ), and the Prophet R package by BID27 .

We evaluate the performances by mean absolute percentage error (MAPE), mean square error (MSE) and mean absolute error (MAE) on forecasting set.

The mathematical definition of these three criterion is given as follows.

Let x 1 , x 2 , . . . , x n be the true value andx 1 ,x 2 , . . . ,x n be the estimation or predictive values.

Then we have DISPLAYFORM0 The comparison of our proposed algorithm and the aforementioned algorithms are included below in TAB1 .

As we mentioned in Section 6, our algorithm is downward compatible with the cases ignoring the existence of change point or anomaly, by setting p c = 0 or p a = 0.

We also run proposed algorithm on the synthetic data with p c = 0 (no change point), or p a = 0 (no anomaly point), or p c = p a = 0 (no change and anomaly point), for the purpose of numeric comparison.

From TAB1 it turns out that our proposed algorithm achieves the best performance compared to other existing methods.

Our proposed algorithm also performs better compared with the cases ignoring change point or anomaly point.

This is a convincing evidence on the importance of incorporating both change point structure and anomaly point structure when modeling, for time series forecasting.

We also compare our proposed method with other existing change point detection methods and anomaly detection algorithm with respect to the performance of detections.

We evaluate the performance by two criterions: True Positive Rate (TPR) and False Positive (FP).

TPR measures the percentage of change points or anomalies to be correctly detected.

FP count the number of points wrongly detected as change points or anomaly points.

The mathematical definitions of TPR and FP are as follows.

Let (z 1 , z 2 , . . .

, z n ) be the true binary vector for change points or anomalies, and (ẑ 1 ,ẑ 2 , . . .

,ẑ n ) are the estimated ones.

Then DISPLAYFORM1 From the definition, we can see high TPR and low FP means the algorithm has better performance in detection.

The comparison on change point detection is shown in TAB2 .

We compare our results against three popular change point detection methods: Bayesian Change Point (BCP) BID3 , Change-Point (CP) BID21 and Breakout (twitter, 2017) .

From TAB2 our proposed method outperforms the most of the others by both TPR and FP.

We have smaller TPR compared to CP, but we are better in FP.

In TAB3 , we also compare the performance of our algorithm on anomaly detection with three existing common anomaly detection methods: the AnomalyDetection package by BID29 , RAD by BID22 and Tsoutlier by BID6 .

The comparison is listed in TAB3 .

We can see our method also outperforms most of the others with respect to anomaly detection, by both TPR and FP.

RAD has slightly better TPR but its FP is much worse compared with ours.

In this section, we implement our proposed method on real-world datasets.

We also compare its performance against other existing time series forecasting methodologies.

We consider two datasets, one is a public data called Well-log dataset, and the other is an unpublished internet traffic dataset.

The bottom panels of FIG10 and FIG11 give the result of our proposed algorithms.

The blue line separates the training set and testing set.

We use red line to show our fitting and forecasting result, vertical red dashed line to indicate change points and purple dots to indicate anomaly points.

The gray part shows 90% predication interval.

This dataset BID11 BID20 was collected when drilling a well.

It measures the nuclear magnetic response, which provides geophysical information to analyze the structure of rock surrounding the well.

This dataset is public and available online 2 .

It has 4050 points in total.

We split it such that the first 3000 points are used as training set and last 1000 points are used to evaluate the forecasting performance.

From FIG10 , it is obvious that there exists no seasonality or slope structure in the dataset.

This motivates us not to include these two components in our model.

We implement our proposed algorithm without seasonality and slope, and compare the forecasting performance with other methods in TAB4 .

Our method outperforms BSTS, ARIMA, ETS and Prophet.

However in TAB4 the performance can be slightly improved if we ignore the existence of anomaly points by letting p a = 0.

This may be caused by model mis-specification as the data may not generated in a way not entirely captured by our model.

Nevertheless, the performances of our method considering anomaly points or not, are comparable to each other.

In this dataset there is no ground-truth of change point and anomaly point on their locations or even existence.

However, from bottom panel of FIG10 , there are some obvious changes in the sequence and they all successfully captured by our algorithm.

Our second real data is an Internet traffic data acquired from a major Tech company (see FIG11 ).

It is a daily traffic data, with seasonality S = 7.

We use the first 800 observations as training set and evaluate the performance of forecasting on the remaining 265 points.

The bottom panel of FIG11 show the result from implementing our algorithm.

We also do the comparison of forecasting performance of our proposed algorithm together with other existing methods, shown in TAB5 .

We can also see that our algorithm outperforms all the other algorithms with respect to MAPE, MSE and MAE.

From FIG11 our proposed algorithm identifies one change point (the 576th point, indicated by the vertical red dashed line), which can be confirmed that this is exactly the only one change point existing in this time series caused by the change of counting methods, by some external information.

Thus, we give the perfect change point detection in this Internet traffic data.

For this Internet traffic dataset, since we have ground-truth for change point, we can compare the performance of change point detection of different methodologies.

BCP returns posterior distribution, which peaks in the the 576th point with posterior probability value 0.5.

And it also returns with many other points with posterior probability value around 0.1.

CP returns 4 change points, where the 576th point (the only true one) is one of them.

Breakout returns 8 change points without including the 576th point.

To sum up, our proposed method achieves the best change point detection in this real dataset.

Compared to the aforementioned models, our work differs in Bayesian modeling which samples posterior to estimate hidden components given the independent Bernoulli priors of changing point and anomalies.

@highlight

We propose a novel state space time series model with the capability to capture the structure of change points and anomaly points, so that it has a better forecasting performance when there exist change points and anomalies in the time series.