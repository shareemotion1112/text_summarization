We propose to tackle a time series regression problem by computing temporal evolution of a probability density function to provide a probabilistic forecast.

A Recurrent Neural Network (RNN) based model is employed to learn a nonlinear operator for temporal evolution of a probability density function.

We use a softmax layer for a numerical discretization of a smooth probability density functions, which transforms a function approximation problem to a classification task.

Explicit and implicit regularization strategies are introduced to impose a smoothness condition on the estimated probability distribution.

A Monte Carlo procedure to compute the temporal evolution of the distribution for a multiple-step forecast is presented.

The evaluation of the proposed algorithm on three synthetic and two real data sets shows advantage over the compared baselines.

Application of the deep learning for manufacturing processes has attracted a great attention as one of the core technologies in Industry 4.0 BID15 .

In many manufacturing processes, e.g. blast furnace, smelter, and milling, the complexity of the overall system makes it almost impossible or impractical to develop a simulation model from the first principles.

Hence, system identification from sensor observations has been a long-standing research topic BID24 .

Still, when the observation is noisy and there is no prior knowledge on the underlying dynamics, there is only a very limited number of methods for the reconstruction of nonlinear dynamics.

In this work, we consider the following class of problems, where the system is driven by a complex underlying dynamical system, e.g., ∂y ∂t = F(y(t), y(t − τ ), u(t)).Here, y(t) is a continuous process, F is a nonlinear operator, τ is a delay-time parameter, and u(t) is an exogenous forcing, such as control parameters.

At time step t, we then observe a noisy measurement of y(t) which can be defined by the following noise model DISPLAYFORM0 where ν t is a multiplicative and t is an additive noise process.

In FORMULA0 and FORMULA1 , we place no assumption on function F, do not assume any distributional properties of noises ν t and t , but assume the knowledge of the control parameters u(t).Since the noise components, ν t and t , are stochastic processes, the observationŷ t is a random variable.

In this work, we are interested in computing temporal evolution of the probability density function (PDF) ofŷ, given the observations up to time step t, i.e., p(ŷ t+n | Y 0:t , U 0:t+n−1 ) for n ≥ 1, where Y 0:t = (ŷ 0 , · · · ,ŷ t ) is a trajectory of the past observations and U 0:t+n−1 = (u 0 , · · · , u t+n−1 ) consists of the history of the known control actions, U 0:t−1 , and a future control scenario, U t:t+n−1 .

We show, in Section 3, a class of problems, where simple regression problem of forecasting the value ofŷ t+n is not sufficient or not possible, e.g., chaotic systems.

Note that the computation of time evolution of a PDF has been a long-standing topic in statistical physics.

For a simple Markov process, there are well-established theories based on the Fokker-Planck equation.

However, it is very difficult to extend those theories to a more general problem, such as delay-time dynamical systems, or apply it to complex nonlinear systems.

Modeling of the system (1) has been extensively studied in the past, in particular, under the linearity assumptions on F and certain noise models, e.g., Gaussian t and ν t = 1 in (2).

The approaches based on auto-regressive processes BID18 and Kalman filter BID9 are good examples.

Although these methods do estimate the predictive probability distribution and enable the computation of the forecast uncertainty, the assumptions on the noise and linearity in many cases make it challenging to model real nonlinear dynamical systems.

Recently, a nonlinear state-space model based on the Gaussian process, called the Gaussian Process State Space Model (GPSSM), has been extended for the identification of nonlinear system BID5 BID4 .

GPSSM is capable of representing a nonlinear system and is particularly advantageous when the size of the data set is relatively small that it is difficult to train a deep learning model.

However, the joint Gaussian assumption of GPSSM may restrict the representation capability for a complex non-Gaussian noise.

A recent success of deep learning created a flurry of new approaches for time series modeling and prediction.

The ability of deep neural networks, such as RNN, to learn complex nonlinear spatiotemporal relationships in the data enabled these methods to outperform the classical time series approaches.

For example, in the recent works of BID20 BID11 ; BID3 , the authors proposed different variants of the RNN-based algorithms to perform time series predictions and showed their advantage over the traditional methods.

Although encouraging, these approaches lack the ability to estimate the probability distribution of the predictions since RNN is a deterministic model and unable to fully capture the stochastic nature of the data.

To enable RNN to model the stochastic properties of the data, BID2 augmented RNN with a latent random variable included in the hidden state and proposed to estimate the resulting model using variational inference.

In a similar vein, the works of BID0 ; BID14 extend the traditional Kalman filter to handle nonlinear dynamics when the inference becomes intractable.

Their approach is based on formulating the variational lower bound and optimizing it under the assumption of Gaussian posterior.

Another recent line of works enabled stochasticity in the RNN-based models by drawing a connection between Bayesian variation inference and a dropout technique.

In particular, BID6 showed that the model parameter uncertainty (which then leads to uncertainty in model predictions), that traditionally was estimated using variational inference, can be approximated using a dropout method (a random removal of some connections in the network structure).

The prediction uncertainty is then estimated by evaluating the model outputs at different realizations of the dropout weights.

Following the ideas of BID6 , BID27 proposed additional ways (besides modeling the parameter uncertainty) to quantify the forecast uncertainty in RNN, which included the model mis-specification error and the inherent noise of the data.

We propose an RNN-model to compute the temporal evolution of a PDF, p(ŷ t+n | Y 0:t , U 0:t+n−1 ).To avoid the difficulties in directly estimating the continuous function, we use a numerical discretization technique, which converts the function approximation problem to a classification task (see Section 2.2).

We note that the use of the traditional cross-entropy (CE) loss in our formulated classification problem can be problematic since it is oblivious to the class ordering.

To address this, we additionally propose two regularizations for CE to account for a geometric proximity between the classes (see Sections 2.2.1 and 2.2.2).

The probability distribution of one-step-ahead prediction, p(ŷ t+1 | Y 0:t , U 0:t ) can now be simply estimated from the output softmax layer of RNN (see Section 2.2), while to propagate the probability distribution further in time, for a multiple-step forecast, we propose a sequential Monte Carlo (SMC) method (see Section 2.4).

For clarity, we present most derivations for univariate time series but also show the extension to multivariate data in Section 2.3.

We empirically show that the proposed modeling approach enables us to represent a continuous PDF of any arbitrary shape, including the ability to handle the multiplicative data noises in (2).

Since the probability distribution is computed, the RNN-model can also be used for a regression task by computing the expectation (see Section 2.4).

Hereafter, we use DE-RNN for the proposed RNN model, considering the similarity with the density-estimation task.

In summary, the contributions of this work are as follows: (i) formulate the classical regression problem for time series prediction as a predictive density-estimation problem, which can be solved by a classification task (ii) propose an approach to compute the time evolution of probability distribution using SMC on the distributions from DE-RNN (iii) proposed two regularizations for CE loss to capture the ordering of the classes in the discretized PDF.

We evaluated the proposed algorithm on three synthetic and two real datasets, showing its advantage over the baselines.

Note that DE-RNN has a direct relevance to a wide range of problems in physics and engineering, in particular, for uncertainty quantification and propagation BID26 .

In this Section we present the details of the proposed algorithm using a specific form of RNN, called Long Short-Term Memory (LSTM) network.

Although in the following presentation and experiments we used LSTM, other networks, e.g., GRU BID1 , can be used instead.

The Long Short-Term Memory network (LSTM) BID10 BID8 consists of a set of nonlinear transformations of input variables z t ∈ R m ;Gating functions: DISPLAYFORM0 Internal state: DISPLAYFORM1 DISPLAYFORM2 Here, ϕ S and ϕ T , respectively, denote the sigmoid and hyperbolic tangent functions, L is a linear layer, which includes a bias, s t ∈ R Nc is the internal state, h t ∈ R Nc is the output of the LSTM network, N c is the number of the LSTM units, and a b denote a component-wise multiplication.

Interesting observation can be made about equation (4).

We can re-write equation FORMULA3 as DISPLAYFORM3 for some functions f and g. With a simple re-scaling, this equation can be interpreted as a first-order Euler scheme for a linear dynamical system, DISPLAYFORM4 Thus, LSTM can be understood as a series expansion, where a complex nonlinear dynamical system is approximated by a combination of many simpler dynamical systems.

Usually, LSTM network is supplemented by feed-forward neural networks, e.g., DISPLAYFORM5 in which x t is the input feature.

Using (5), we can denote by Ψ e and Ψ d a collection of the operators from input to internal state (encoder) and from internal state to the output P (decoder): DISPLAYFORM6

In this Section we first consider the problem of modeling the conditional PDF, p(ŷ t+1 | Y 0:t , U 0:t ).

Althoughŷ t+1 has a dependence on the past trajectories of bothŷ and u, using the "state space" LSTM model argument in Section 2.1, the conditional PDF can be modeled as a Markov process DISPLAYFORM0 Hence, to simplify the problem, we consider a task of estimating the PDF of a random variablê y, given an input x, i.e., p(ŷ|x).

The obtained results can then be directly applied to the original problem of estimating p(ŷ t+1 |s t ).

DISPLAYFORM1 Then, a discrete probability distribution can be defined DISPLAYFORM2 where it is clear that p(k|x) is a numerical discretization of the continuous PDF, p(ŷ|x).

Using the LSTM from Section 2.1, the discrete probability p(k|x) can be modeled by the softmax layer (P ) as an output of Ψ d in (9) such that DISPLAYFORM3 Thus, the original problem of estimating a smooth function, p(ŷ|x), is transformed into a classification problem of estimating p(k|x) in a discrete space.

Obviously, the size of the bin, |I j |, affects the fidelity of the approximation.

The effects of the bin size are presented in Section 3.1.

There is a similarity between the discretization and the idea of BID16 .

However, it should be noted that the same discretization technique, often called "finite volume method", has been widely used in the numerical simulations of partial differential equations for a long time.

The discretization naturally leads to the conventional cross-entropy (CE) minimization.

Suppose we have a data set, D R = {(ŷ i , x i );ŷ i ∈ R, x i ∈ R, and i = 1, . . .

, N }.

We can define a mapping DISPLAYFORM4 D C provides a training data set for the following CE minimization problem, DISPLAYFORM5 Note, however, that the CE minimization does not explicitly guarantee the smoothness of the estimated distribution.

Since CE loss function depends only on P i of a correct label, δ cnk , as a result, in the optimization problem every element P i , except for the one corresponding to the correct label, P cn , is penalized in the same way, which is natural in the conventional classification tasks where a geometric proximity between the classes is not relevant.

In the present study, however, the softmax layer, or class probability, is used as a discrete approximation to a smooth function.

Hence, it is expected that P cn and P cn±1 (i.e., the nearby classes) should be close to each other.

To address this issue, in the following Sections 2.2.1 and 2.2.2, we propose two types of regularization to impose the class proximity structure in the CE loss.

To explicitly impose the smoothness between the classes, we propose to use a regularized crossentropy (RCE) minimization, defined by the following loss function DISPLAYFORM0 where λ is a penalty parameter and the Laplacian matrix DISPLAYFORM1 RCE is analogous to the penalized maximum likelihood solution for density estimation BID23 .

Assuming a uniform bin size, |I 0 | = · · · = |I K | = δy, the Laplacian of a distribution can be approximated by a Taylor expansion p (ŷ|x) DISPLAYFORM2 In other words, RCE aims to smooth out the distribution by penalizing local minima or maxima.

Alternative to adding an explicit regularization to CE, the smoothness can be achieved by enforcing a spatial correlation in the network output.

Here, we use an one-dimensional convolution layer to enforce smoothness.

Let o ∈ R K denote the last layer of DE-RNN, which was the input to the softmax layer.

We can add a convolution layer, o ∈ R K , on top of o, such that DISPLAYFORM0 where the parameter h determines the smoothness of the estimated distribution.

Then, o is supplied to the softmax layer.

Using (17), DE-RNN can now be trained by the standard CE.

The implicit regularization, here we call convolution CE (CCE), is analogous to a kernel density estimation.

In the modeling of multivariate time series, it is usually assumed that the noise is independent, i.e., the covariance of the noise is a diagonal matrix.

In this case, it is straightforward to extend DE-RNN, by using multiple softmax layers as the output of DE-RNN.

However, such an independent noise assumption significantly limits the representative capability of an RNN.

Here, we propose to use a set of independently trained DE-RNNs to compute the joint PDF of a multivariate time series.

Letŷ t be a l-dimensional multivariate time series;ŷ t = (ŷ DISPLAYFORM0 The joint PDF can be represented by a product rule, DISPLAYFORM1 where the dependency on the past trajectory ( Y 0:t , U 0:t ) is omitted in the notation for simplicity.

Directly learning the joint PDF, p(ŷ t+1 | Y 0:t , U 0:t ), in a tensor product space is not scalable.

Instead, a set of DE-RNN is trained to represent the conditional PDFs shown on the right hand side of the above expression.

Then, the joint PDF can be computed by a product of the Softmax outputs of the DE-RNNs.

Note that, although it requires training l DE-RNNs to compute the full joint PDF, there is no dependency between the DE-RNNs in the training phase.

So, the set of DE-RNNs can be trained in parallel, which can greatly reduce the training time.

The details of the multivariate DE-RNN are explained in Appendix A.

The inputs to a DE-RNN are (ŷ t , u t ), and the output is the probability distribution, DISPLAYFORM0 Note that D C is used only in the training stage.

Then, the moments of the predictive distribution can be easily evaluated, e.g., DISPLAYFORM1 DISPLAYFORM2 , and α i−1/2 = 0.5(α i−1 + α i ).

Next, we consider a multiple-step forecast, which corresponds to computing a temporal evolution of the probability distribution, i.e., p(ŷ t+n | Y 0:t , U 0:t+n−1 ) for n > 1.

For simplicity, the multiple-step forecast is shown only for a univariate time series.

An extension to a multivariate time series is straightforward (Appendix A).Applying the results of Section 2.2, once the distribution ofŷ t+1 in (10) is computed, the distribution ofŷ t+2 can be similarly obtained as p(ŷ t+2 |s t+1 ).

Observe that s t+1 is computed from a deterministic function of s t , u t+1 , andŷ t+1 , i.e., DISPLAYFORM3 Here, u t+1 and s t are already known, whileŷ t+1 is a random variable, whose distribution p(ŷ t+1 |s t ) is computed from the deterministic function Ψ d (s t ).

Then, s t+1 is also a random variable.

The distribution, p(s t+1 |s t , u t+1 ), can be obtained by applying a change of variables on p(ŷ t+1 |s t ) with a nonlinear mapping Ψ e .

Repeating this process, the multiple-step-ahead predictive distribution can therefore be computed as DISPLAYFORM4 Since the high dimensional integration in (19) is intractable, we propose to approximate it by a sequential Monte Carlo method.

The Monte Carlo procedure is outlined in Algorithm 1.

Input: Y 0:t , U 0:t , number of Monte Carlo samples, N s , and forecast horizon n Output: p(ŷ t+n | Y 0:t , U 0:t+n−1 ) (density estimation fromŷ t+n ) Initialization: Set LSTM states to s 0 = h 0 = 0 Perform a sequential update of LSTM up to time t from the noisy observations ( Y 0:t ).

DISPLAYFORM0 Make N s replicas of the internal state, s DISPLAYFORM1 Compute the predictive distribution ofŷ i t+1 for each sample DISPLAYFORM2 Sample the target variable at t + 1,ŷ i t+1 , from the distribution 1.

Sample the class label from the discrete distribution: DISPLAYFORM3

In this section, DE-RNN is tested against three synthetic and two real data sets.

The LSTM architecture used in all of the numerical experiments is identical.

Two feed-forward networks are used before and after the LSTM; DISPLAYFORM0 in which ϕ SP and ϕ SM denote the softplus and softmax functions, respectively.

The size of the bins is kept uniform, i.e., |I 1 | = · · · = |I K | = δy.

The LSTM is trained by using ADAM BID13 with a minibath size of 20 and a learning rate of η = 10 −3 .

First, we consider a modified Cox-Ingersoll-Ross (CIR) process, which is represented by the following stochastic differential equation, DISPLAYFORM0 in which W is the Weiner process.

The original CIR process is used to model the valuation of interest rate derivatives in finance.

Equation FORMULA0 is solved by the forward Euler method with the time step size δt = 0.1.

The simulation is performed for T = (0, 160000]δt to generate the training data and T = (160000, 162000]δt is used for the testing.

Note that the noise component of CIR is multiplicative, which depends on y(t).

The experiments are performed for two different bin sizes, dy = 0.08 and 0.04.

The DE-RNN has 64 LSTM cells.

FIG0 shows the errors in the expectation and the standard deviation with respect to the analytical solution; DISPLAYFORM1 Here, p T denotes the true distribution of the CIR process.

The normalized root mean-square errors (NRMSE) are defined as DISPLAYFORM2 DISPLAYFORM3 in which · denotes an average over the testing data, p L is the distribution from the LSTM, and sd[y] denotes the standard deviation of the data.

The error in the expectation is normalized against a zeroth-order prediction, which assumes y t+1 = y t .In FIG0 , it is clearly shown that the prediction results are improved when a regularization is used to impose a smoothness condition.

FIG0 and (b), for RCE, e µ and e σ become smaller when a smaller δy is used.

As expected, e σ increases when λ is large.

But, for the smaller bin size, δy = 0.04, both e µ and e σ are not so sensitive to λ.

Similar to RCE, e µ and e σ for CCE decrease at first as the penalty parameter h increases.

However, in general, RCE provides a better prediction compared to CCE.NRMEs are listed in TAB0 .

For a comparison, the predictions by AR(1) and KF are shown.

The CIR process is essentially a first-order autoregressive process.

So, it is not surprising to see that AR(1) and KF, which are designed for the first-order AR process, outperforms DE-RNN for the prediction of the expectation.

However, e σ of AR(1) and KF are much larger than that of DE-RNN, because those models assume an additive noise.

The Gaussian process (GP) model has a relatively large e µ .

But, GP outperforms AR(1) and KF in the prediction of the noise (e σ ).

Still, e σ of RCE and CCE are less than 4%, while that of GP is about 10%, indicating that DE-RNN can model the complex noise process much better.

In FIG1 , a 200-step forecast by DE-RNN is compared with a Monte-Carlo solution of equation FORMULA0 .

DE-RNN is trained with δy = 0.04 and λ = 200.

For the DE-RNN forecast, the testing data is supplied to DE-RNN for the first 100 time steps, i.e., for t = −10 to t = 0, and the SMC multiple-step forecast is performed for the next 200 time steps with 20,000 samples.

It is shown that the multiple-step forecast by DE-RNN agrees very well with the MC solution of the CIR process.

Note that, in FIG1 (b), the noise process, as reflected in sd[y t ], is a function of y t , and hence the multi-step forecast of the noise increases rapidly first and then decreases before reaching a plateau.

The SMC forecast can accurately capture the behavior.

Such kind of behavior can not be represented if a simple additive noise is assumed.

For the next test, we applied DE-RNN for a time series generated from the Mackey-Galss equation BID19 ; DISPLAYFORM0 We use the parameters adopted from Gers (2001), α = 0.2, β = 10, γ = 0.1, and τ = 17.The Mackey-Glass equation is solved by using a third-order Adams-Bashforth method with a time step size of 0.02.

The time series is generated by down-sampling, such that the time interval between consecutive data is δt = 1.

A noisy observation is made by adding a white noise; DISPLAYFORM1 t is a zero-mean Gaussian random variable with the noise level sd[ t ] = 0.3sd [y] .

A time series of the length 1.6 × 10 5 δt is generated for the model trainig and another time series of length 2 × 10 3 δt is made for the validation.

DE-RNN is trained for δy = 0.04sd[y] and consists of 128 LSTM cells.

FIG2 shows the noisy observation and the expectation of the next-step prediction, E[ŷ t+1 |ŷ t ], in a phase space.

It is shown that DE-RNN can filter out the noise and reconstruct the original dynamics accurately.

Even though the noisy data are used as an input, E[ŷ t+1 |ŷ t ] accurately represents the original attractor of the chaotic system, indicating a strong de-noising capability of DE-RNN.The estimated probability distribution is shown in FIG2 .

Without a regularization, the standard CE results in a noisy distribution, while the distribution from CCE shows a smooth Gaussian shape.

FIG7 : (a) 500-step forecast by a regression LSTM (•) and the ground truth ( ).

(b) The color contours denote a 500-step forecast of the probability distribution, p(ŷ n+s |ŷ s ), and the dashed lines are 95%-CI.

The ground truth is shown as the solid line ( ).The prediction errors are shown in table 2.

NRMSEs are defined as, DISPLAYFORM2 NRMSEs are computed with respect to the ground truth.

Again, e µ compares the prediction error to the zeroth-order prediction.

In this example, the errors are not so sensitive to the regularization parameters.

The best result is achieved by CCE.

DE-RNN can make a very good estimation of the noise.

The error in the noise component, e σ , is only 2% ∼ 5%.

Unlike the CIR process, NRMSEs from KF and ARIMA are much larger than those of DE-RNN.

Because the underlying process is a delay-time nonlinear dynamical system, those linear models can not accurately approximate the complex dynamics.

Since GP is capable of representing a nonlinear behavior of data, GP outperforms KF and ARIMA both in e µ and e σ .

In particular, e σ of GP is similar to that of DE-RNN.However, e µ of GP is about 1.5 times larger than DE-RNN.A multiple-step forecast of the Mackey-Glass time series is shown in FIG7 .

In the validation time series, the observations in t ∈ [1, 100]δt are supplied to the DE-RNN to develop the internal state, and a 500-step forecast is made for t ∈ [101, 600]δt.

In FIG7 (a), it is shown that a multiple-step forecast by a standard regression LSTM approximates y(t) very well initially, e.g, for t < 80δt, but eventually diverges for larger t. Because of the Mackey-Glass time series is chaotic, theoretically it is impossible to make a long time forecast.

But, in the DE-RNN forecast, y(t) is bounded by the 95%-confidence interval (CI) even for the 500-step forecast.

Note that the uncertainty, denoted by 95%-CI grows only at a very mild rate in time.

In fact, it is observed that CI is not a monotonic function of time.

In DE-RNN, the 95%-CI may grow or decrease following the dynamics of the system, while for the conventional time series models, such as ARIMA and KF, the uncertainty is a non-decreasing function of time.3.3 MAUNA LOA CO 2 OBSERVATIONIn this experiments, DE-RNN is tested against the atmospheric CO 2 observation at Mauna Loa Observatory, Hawaii BID12 ).

The CO 2 data set consists of weekly-average atmospheric CO 2 concentration from Mar-29-1958 to Sep-23-2017 FIG3 .

The data from Mar-29-1958 to Apr-01-2000 is used to train DE-RNN and a 17-year forecast is made from Apr-01-2000 to Sep-23-2017 .

This CO 2 data has been used in previous studies BID6 BID22 .

In DE-RNN, 64 LSTM cells and δy = 0.1sd[dy t ], in which dy t = y t+1 − y t , are used.

The 17-year DE-RNN forecast, with 1,000 MC samples, is shown in FIG3 (b).

DE-RNN well represents the growing trend and the oscillatory patten of the CO 2 data.

The CO 2 data is nonstationary, where the rate of increase of CO 2 is an increasing function of time.

Since DE-RNN is trained against the history data, where the rate of CO 2 increase is smaller than the current, it is expected that the forecast will underestimate the future CO 2 .

E[ŷ n+s |ŷ s ] agrees very well with the observation for the first 200 weeks, but eventually underestimates CO 2 concentration.

It is interesting to observe that the upper bound of the 95%-CI grows more rapidly than the expectation and provides a good approximation of the observation.

For a comparison, the forecast by a regression LSTM is also shown.

Similar to the chaotic Mackey-Glass time series, the regression LSTM makes a good prediction for a short time, e.g., t < 100 weeks, but eventually diverges from the observation.

Note that the lower bound of 95%-CI encompasses the regression LSTM.

FIG3 (c) shows a forecast made by GP, following setup suggested by BID21 .

For a shortterm forecast (< 50 weeks), GP provides a sharper estimate of the uncertainty, i.e., a smaller 95%-CI interval.

However, even for a mid-term forecast, 100 ∼ 600 weeks, the ground truth is close or slightly above the upper bound of 95%-CI.

Because of the different behaviors, it is difficult to conclude which method, DE-RNN or GP, provides a better forecast.

But, it should be noted that the GP is based on a combination of handcrafted kernels specifically designed for this particular problem, while such careful tuning is not required for DE-RNN.

In the last experiment, IBM Power System S822LC and NAS Parallel Benchmark (NPB) are used to generate the temperature trace.

FIG4 shows the temperature of a CPU.

The temperature sensor generates a discrete data, which has a resolution of 1 • C. The CPU temperature is controlled by three major parameters; CPU frequency, CPU utilization, and cooling fan speed.

In this experiment, we have randomized the CPU frequencies and job arrival time to mimic the real workload behavior, while the fan speed is fixed to 3300RPM.

The time step size is δt = 2 seconds.

Accurate forecast of CPU temperature for a future workload scenario is essential in developing an energy-efficient control strategy for the thermal management of a cloud system.

FIG4 (c) and (d) show multiple-step forecasts of the CPU temperature by RCE and a regression LSTM, respectively.

The bin size is δy = 0.18• C, which is smaller than the sensor resolution.

In the forecast, the future control parameters are given to DE-RNN.

In other words, DE-RNN predicts the probability distribution of future temperature with respect to a control scenario, i.e., p(ŷ t+n | Y 0:t , U 0:t+n−1 ).

The forecast is made by using 5,000 Monte Carlo samples.

Here, 1800-step forecast is made, t = 0 ∼ 3, 600 sec. and only the results in t ∈ (50, 1800) sec. is shown.

While the regression LSTM makes a very noisy prediction near the local peak temperature at t 500, RCE provides a much more stable forecast.

TAB2 shows the l ∞ -errors, i.e., maximum absolute difference.

The maximum error is observed near the peak temperature at t 500.

ARIMA, KF, and GP are also tested for the multiple-step forecast, but the results are not shown because their performance is much worse than the LSTMs.

The changes in the temperature are associated with step changes of some control parameters.

Such abrupt transitions seem to cause the large oscillation in the regression LSTM prediction.

But, for RCE or CCE, the prediction is made from an ensemble of Monte Carlo samples, which makes it more robust to such abrupt changes.

Also, note that for t < 200 sec., RCE prediction ( 53.4• C) is in between the two discrete integer levels, 53• C and 54• C, which correctly reflects the uncertainty in the measurement, while the regression LSTM ( 53.1• C) more closely follows one of the two levels.

Finally, in this experiment we evaluate the performance of the multivariate DE-RNN, for which we used a noisy multivariate time series generated by the Lorenz equations BID17 dy DISPLAYFORM0 DISPLAYFORM1 We used the coefficients from BID17 , which are α 1 = 10, α 2 = 8/3, and α 3 = 28.

The system of equations FORMULA1 is solved by using a third-order Adams-Bashforth method with a time step size of 0.001 and a time series data set is generated by downsampling, such that δt = 0.02.

A multivariate joint normal distribution is added to the ground truth to generate a noisy time series, i.e.,   ŷ DISPLAYFORM2 Here, the noise level is set to σ 1 = 0.2sd[y DISPLAYFORM3 t+1 , Y 0:t ), and p(ŷ DISPLAYFORM4 t+1 , Y 0:t ), re- DISPLAYFORM5 (29) The moments of the joint PDF are computed by the Monte Carlo method with a sample size of 5 × 10 3 .

It is shown that DE-RNN makes a very good prediction of both expectations and covariances.

The error in the covariance is less than 4% except for Σ 12 .

It is shown that DE-RNN is able to make correct predictions of not only the magnitude, but also the signs of the covariance.

A vector autoregressive model (VAR) also predicts the signs of the covariance correctly.

But, the errors in the expectation and covariances are much larger than DE-RNN.

The GP used in the experiment assumes an independent noise, i.e. Σ = ρ 2 I. Hence, e Σij is not evaluated for GP.

Similar to the MackeyGlass time series, GP outperforms VAR, but the errors are larger than DE-RNN.

The error in e σ is about 10 times larger than DE-RNN, while e µ of GP is about 3 times larger than DE-RNN.

FIG6 shows 300-step forecasts of the Lorenz time series.

The expectations from DE-RNNs make a good prediction of the ground truth up to t − s < 1.5.

Then, the expectations start to diverge from the ground truth, which is accompanied by a sudden increase in the 95%-CI.

For a longer-time forecast, e.g., t − s > 4, the 95%-CI exhibits an oscillatory patten depending on the dynamics of the Lorenz system and well captures the oscillations in the ground truth.

We present DE-RNN to compute the time evolution of a probability distribution for complex time series data.

DE-RNN employs LSTM to learn multiscale, nonlinear dynamics from the noisy observations, which is supplemented by a softmax layer to approximate a probability density function.

To assign probability to the softmax output, we use a mapping from R to N + , which leads to a cross-entropy minimization problem.

To impose a geometric structure in the distribution, two regularization strategies are proposed.

The regularized cross-entropy method is analogous to the penalized maximum likelihood estimate for the density estimation, while the convolution cross-entropy method is motivated by the kernel density estimation.

The proposed algorithm is validated against three synthetic data set, for which we can compare with the analytical solutions, and two real data sets.

Recall the product rule from Section 2.3 DISPLAYFORM0 The extension of DE-RNN algorithm for univariate data to an l-dimensional multivariate time series is straightforward.

First, define the discretization grid points, DISPLAYFORM1 Ki ), for every variable, i.e., i = 1, · · · , l. Here, K i is the number of the discretization intervals for the i-th variable.

Then, we can define the discretization intervals, I DISPLAYFORM2 , and the mapping functions, DISPLAYFORM3 The first component of the product rule is the marginal PDF, p(ŷt+1 ).

Hereafter, the obvious dependency on Y 0:t , U 0:t in the notation is omitted for simplicity.

The marginal PDF can be computed by the same method as for the univariate time series.

To train DE-RNNs for the conditional PDFs for the i-th variable, p(ŷ DISPLAYFORM4 t+1 ), the original time series data of length N , D R = {ŷ t ;ŷ t ∈ R l , and t = 1, . . .

, N }, is discretized by using C (i) , which gives us DISPLAYFORM5 ∈ N + ,ŷ t ∈ R l , and t = 1, . . .

, N }, where c DISPLAYFORM6 , is computed by an LSTM as DISPLAYFORM7 in which s DISPLAYFORM8 t is the internal state of the i-th variable.

In other words, in the training of the DE-RNN for p(ŷ DISPLAYFORM9 t+1 ), the input vector is the variables at the current time step,ŷ t , combined with the conditioning variables in the next step, (ŷ DISPLAYFORM10 t+1 ), and the target is the class label, c DISPLAYFORM11 t+1 , in the next time step.

The DE-RNN can be trained by minimizing RCE or CCE as described in Section 2.2.

Observe that during the training phase, each DE-RNN is independent from each other.

Therefore, all the DE-RNNs can be trained in parallel, significantly improving the computational efficiency, enabling the method to scale only linearly with the number of dimensions l.

Once all the conditional PDFs are obtained, the joint PDF can be computed by a product of the DE-RNN outputs.

For a demonstration, the covariance of a bivariate time series can be computed as DISPLAYFORM12 where the time index, (t+1), is omitted in the notation of the softmax output, Pj , and the subscript j denotes the j-th element of the softmax layer.

Note that, although there is no dependency between DE-RNNs during training, in the prediction phase of computing the joint PDF, there is a hierarchical dependency between all the DE-RNNs.

This kind of direct numerical integration does not scale well for number of dimensions l 1.

For a high dimensional system, a sparse grid BID25 or a Monte Carlo method can be used to speed up the numerical integration.

We outline a Monte Carlo procedure in Algorithm 2.

Comparing with Algorithm 1, an extension of Algorithm 2 for a multiple-step forecast of a multivariate time series is straightforward.

In this Section we present a few extra details of the proposed DE-RNN algorithm.

FIG10 shows the architecture of DE-RNN as was used during the experiments in Section 3.

In Figure 9 we show the process of computing one-step-ahead predictions of univariate time series as was presented in Section 2.2.

Note that since DE-RNN estimates approximation of the predictive probability distribution, the predicted value, e.g., for time step t+1, is the discrete approximation of E[ŷ t+1 | Y 0:t , U 0:t ], i.e., the expectation ofŷ t+1 given all the observations and control inputs up to time t. Finally, in for n = 1, N s do Draw a sample,ŷ(1),n t+1 , from Pt+1 .

for j = 2, l doCompute the conditional distribution forŷ end for end for FIG0 we show the details of the multi-step forecast for univariate time series as was presented in Algorithm 1, in Section 2.4.

Using sequential Monte Carlo method, the discrete approximation of the predictive distribution p(ŷ t+n | Y 0:t , U 0:t+n−1 ) is estimated using N s samples.

Figure 9 : Details of the computation for the one-step-ahead predictions.

At a given step the model computes a discrete approximation of the expectation for the next step observation.

DISPLAYFORM0 DISPLAYFORM1

@highlight

Proposed RNN-based algorithm to estimate predictive distribution in one- and multi-step forecasts in time series prediction problems