We present Tensor-Train RNN (TT-RNN), a novel family of neural sequence architectures for multivariate forecasting in environments with nonlinear dynamics.

Long-term forecasting in such systems is highly challenging, since there exist long-term temporal dependencies, higher-order correlations and sensitivity to error propagation.

Our proposed tensor recurrent architecture addresses these issues by learning the nonlinear dynamics directly using higher order moments and high-order state transition functions.

Furthermore, we decompose the higher-order structure using the tensor-train (TT) decomposition to reduce the number of parameters while preserving the model performance.

We theoretically establish the approximation properties of Tensor-Train RNNs for general sequence inputs, and such guarantees are not available for usual RNNs.

We also demonstrate significant long-term prediction improvements over general RNN and LSTM architectures on a range of simulated environments with nonlinear dynamics, as well on real-world climate and traffic data.

One of the central questions in science is forecasting: given the past history, how well can we predict the future?

In many domains with complex multivariate correlation structures and nonlinear dynamics, forecasting is highly challenging since the system has long-term temporal dependencies and higher-order dynamics.

Examples of such systems abound in science and engineering, from biological neural network activity, fluid turbulence, to climate and traffic systems (see FIG0 ).

Since current forecasting systems are unable to faithfully represent the higher-order dynamics, they have limited ability for accurate long-term forecasting.

Therefore, a key challenge is accurately modeling nonlinear dynamics and obtaining stable long-term predictions, given a dataset of realizations of the dynamics.

Here, the forecasting problem can be stated as follows: how can we efficiently learn a model that, given only few initial states, can reliably predict a sequence of future states over a long horizon of T time-steps?

Common approaches to forecasting involve linear time series models such as auto-regressive moving average (ARMA), state space models such as hidden Markov model (HMM), and deep neural networks.

We refer readers to a survey on time series forecasting by BID2 and the references therein.

A recurrent neural network (RNN), as well as its memory-based extensions such as the LSTM, is a class of models that have achieved good performance on sequence prediction tasks from demand forecasting BID5 to speech recognition BID15 and video analysis BID9 .

Although these methods can be effective for short-term, smooth dynamics, neither analytic nor data-driven learning methods tend to generalize well to capturing long-term nonlinear dynamics and predicting them over longer time horizons.

To address this issue, we propose a novel family of tensor-train recurrent neural networks that can learn stable long-term forecasting.

These models have two key features: they 1) explicitly model the higher-order dynamics, by using a longer history of previous hidden states and high-order state interactions with multiplicative memory units; and 2) they are scalable by using tensor trains, a structured low-rank tensor decomposition that greatly reduces the number of model parameters, while mostly preserving the correlation structure of the full-rank model.

In this work, we analyze Tensor-Train RNNs theoretically, and also experimentally validate them over a wide range of forecasting domains.

Our contributions can be summarized as follows:• We describe how TT-RNNs encode higher-order non-Markovian dynamics and high-order state interactions.

To address the memory issue, we propose a tensor-train (TT) decomposition that makes learning tractable and fast.• We provide theoretical guarantees for the representation power of TT-RNNs for nonlinear dynamics, and obtain the connection between the target dynamics and TT-RNN approximation.

In contrast, no such theoretical results are known for standard recurrent networks.•

We validate TT-RNNs on simulated data and two real-world environments with nonlinear dynamics (climate and traffic).

Here, we show that TT-RNNs can forecast more accurately for significantly longer time horizons compared to standard RNNs and LSTMs.

Forecasting Nonlinear Dynamics Our goal is to learn an efficient model f for sequential multivariate forecasting in environments with nonlinear dynamics.

Such systems are governed by dynamics that describe how a system state x t ∈ R d evolves using a set of nonlinear differential equations: DISPLAYFORM0 where ξ i can be an arbitrary (smooth) function of the state x t and its derivatives.

Continous time dynamics are usually described by differential equations while difference equations are employed for discrete time.

In continuous time, a classic example is the first-order Lorenz attractor, whose realizations showcase the "butterfly-effect", a characteristic set of double-spiral orbits.

In discretetime, a non-trivial example is the 1-dimensional Genz dynamics, whose difference equation is: DISPLAYFORM1 where x t denotes the system state at time t and c, w are the parameters.

Due to the nonlinear nature of the dynamics, such systems exhibit higher-order correlations, long-term dependencies and sensitivity to error propagation, and thus form a challenging setting for learning.

Given a sequence of initial states x 0 . . .

x t , the forecasting problem aims to learn a model f DISPLAYFORM2 that outputs a sequence of future states x t+1 . . .

x T .

Hence, accurately approximating the dynamics ξ is critical to learning a good forecasting model f and accurately predicting for long time horizons.

First-order Markov Models In deep learning, common approaches for modeling dynamics usually employ first-order hidden-state models, such as recurrent neural networks (RNNs).

An RNN with a single RNN cell recursively computes the output y t from a hidden state h t using: DISPLAYFORM3 where f is the state transition function, g is the output function and θ are model parameters.

An RNN only considers the most recent hidden state in its state transition function.

A common parametrization scheme for (4) is a nonlinear activation function applied to a linear map of x t and h t−1 as: LSTMs BID8 and GRUs BID3 .

For instance, LSTM cells use a memory-state, which mitigate the "exploding gradient" problem and allow RNNs to propagate information over longer time horizons.

Although RNNs are very expressive, they compute h t only using the previous state h t−1 and input x t .

Such models do not explicitly model higher-order dynamics and only implicitly model long-term dependencies between all historical states h 0 . . .

h t , which limits their forecasting effectiveness in environments with nonlinear dynamics.

DISPLAYFORM4

To effectively learn nonlinear dynamics, we propose Tensor-Train RNNs, or TT-RNNs, a class of higher-order models that can be viewed as a higher-order generalization of RNNs.

We developed TT-RNNs with two goals in mind: explicitly modeling 1) L-order Markov processes with L steps of temporal memory and 2) polynomial interactions between the hidden states h · and x t .First, we consider longer "history": we keep length L historic states: DISPLAYFORM0 where f is an activation function.

In principle, early work BID7 has shown that with a large enough hidden state size, such recurrent structures are capable of approximating any dynamics.

Second, to learn the nonlinear dynamics ξ efficiently, we also use higher-order moments to approximate the state transition function.

We construct a higher-order transition tensor by modeling a degree P polynomial interaction between hidden states.

Hence, the TT-RNN with standard RNN cell is defined by: DISPLAYFORM1 where α index the hidden dimension, i · index historic hidden states and P is the polynomial degree.

Here, we defined the L-lag hidden state as: DISPLAYFORM2 We included the bias unit 1 to model all possible polynomial expansions up to order P in a compact form.

The TT-RNN with LSTM cell, or "TLSTM", is defined analogously as: DISPLAYFORM3 where • denotes the Hadamard product.

Note that the bias units are again included.

TT-RNN serves as a module for sequence-to-sequence (Seq2Seq) framework BID18 , which consists of an encoder-decoder pair (see FIG1 ).

We use tensor-train recurrent cells both the encoder and decoder.

The encoder receives the initial states and the decoder predicts x t+1 , . . .

, x T .

For each timestep t, the decoder uses its previous prediction y t as an input.

Unfortunately, due to the "curse of dimensionality", the number of parameters in W α with hidden size H grows exponentially as O(HL P ), which makes the high-order model prohibitively large to train.

To overcome this difficulty, we utilize tensor networks to approximate the weight tensor.

Such networks encode a structural decomposition of tensors into low-dimensional components and have been shown to provide the most general approximation to smooth tensors BID11 .

The most commonly used tensor networks are linear tensor networks (LTN), also known as tensor-trains in numerical analysis or matrix-product states in quantum physics BID12 .A tensor train model decomposes a P -dimensional tensor W into a network of sparsely connected low-dimensional tensors DISPLAYFORM0 DISPLAYFORM1 as depicted in Figure ( 3).

When r 0 = r P = 1 the {r d } are called the tensor-train rank.

With tensortrain, we can reduce the number of parameters of TT-RNN from (HL + 1) P to (HL + 1)R 2 P , with R = max d r d as the upper bound on the tensor-train rank.

Thus, a major benefit of tensor-train is that they do not suffer from the curse of dimensionality, which is in sharp contrast to many classical tensor decompositions, such as the Tucker decomposition.

A significant benefit of using tensor-trains is that we can theoretically characterize the representation power of tensor-train neural networks for approximating high-dimensional functions.

We do so by analyzing a class of functions that satisfies some regularity condition.

For such functions, tensor-train decompositions preserve weak differentiability and yield a compact representation.

We combine this property with neural network estimation theory to bound the approximation error for TT-RNN with one hidden layer in terms of: 1) the regularity of the target function f , 2) the dimension of the input space, 3) the tensor train rank and 4) the order of the tensor.

In the context of TT-RNN, the target function f (x), with x = s ⊗ . . .

⊗ s, describes the state transitions of the system dynamics, as in (6).

Let us assume that f (x) is a Sobolev function: f ∈ H k µ , defined on the input space I = I 1 × I 2 × · · ·

I d , where each I i is a set of vectors.

The space H k µ is defined as the functions that have bounded derivatives up to some order k and are L µ -integrable: DISPLAYFORM0 where D (i) f is the i-th weak derivative of f and µ ≥ 0.

1 Any Sobolev function admits a Schmidt decomposition: DISPLAYFORM1 , where {λ} are the eigenvalues and {γ}, {φ} are the associated eigenfunctions.

Hence, we can decompose the target function f ∈ H k µ as: DISPLAYFORM2 where DISPLAYFORM3 We can truncate (13) to a low dimensional subspace (r < ∞), and obtain the functional tensor-train (FTT) approximation of the target function f : DISPLAYFORM4 In practice, TT-RNN implements a polynomial expansion of the state s as in (6), using powers [s, s ⊗2 , · · · , s ⊗p ] to approximate f T T , where p is the degree of the polynomial.

We can then bound the approximation error using TT-RNN, viewed as a one-layer hidden neural network: 1 A weak derivative generalizes the derivative concept for (non)-differentiable functions and is implicitly defined as: DISPLAYFORM5 DISPLAYFORM6 is the size of the state space, r is the tensor-train rank and p is the degree of high-order polynomials i.e., the order of tensor.

For the full proof, see the Appendix.

From this theorem we see: 1) if the target f becomes smoother, it is easier to approximate and 2) polynomial interactions are more efficient than linear ones in the large rank region: if the polynomial order increases, we require fewer hidden units n.

This result applies to the full family of TT-RNNs, including those using vanilla RNN or LSTM as the recurrent cell, as long as we are given a state transitions (x t , s t ) → s t+1 (e.g. the state transition function learned by the encoder).

We validated the accuracy and efficiency of TT-RNN on one synthetic and two real-world datasets, as described below; Detailed preprocessing and data statistics are deferred to the Appendix.

Genz dynamics The Genz "product peak" (see FIG3 a) is one of the Genz functions BID6 , which are often used as a basis for high-dimensional function approximation.

In particular, BID1 used them to analyze tensor-train decompositions.

We generated 10, 000 samples of length 100 using (2) with w = 0.5, c = 1.0 and random initial points.

Traffic The traffic data (see FIG3 b) of Los Angeles County highway network is collected from California department of transportation http://pems.dot.ca.gov/. The prediction task is to predict the speed readings for 15 locations across LA, aggregated every 5 minutes.

After upsampling and processing the data for missing values, we obtained 8, 784 sequences of length 288.Climate The climate data (see FIG3 c) is collected from the U.S. Historical Climatology Network (USHCN) (http://cdiac.ornl.gov/ftp/ushcn_daily/).

The prediction task is to predict the daily maximum temperature for 15 stations.

The data spans approximately 124 years.

After preprocessing, we obtained 6, 954 sequences of length 366.

Experimental Setup To validate that TT-RNNs effectively perform long-term forecasting task in (3), we experiment with a seq2seq architecture with TT-RNN using LSTM as recurrent cells (TLSTM).

For all experiments, we used an initial sequence of length t 0 as input and varied the forecasting horizon T .

We trained all models using stochastic gradient descent on the length-T sequence regression loss L(y,ŷ) = T t=1 ||ŷ t − y t || 2 2 , where y t = x t+1 ,ŷ t are the ground truth and model prediction respectively.

For more details on training and hyperparameters, see the Appendix.

We compared TT-RNN against 2 set of natural baselines: 1st-order RNN (vanilla RNN, LSTM), and matrix RNNs (vanilla MRNN, MLSTM), which use matrix products of multiple hidden states without factorization BID14 ).

We observed that TT-RNN with RNN cells outperforms vanilla RNN and MRNN, but using LSTM cells performs best in all experiments.

We also evaluated the classic ARIMA time series model and observed that it performs ∼ 5% worse than LSTM.Long-term Accuracy For traffic, we forecast up to 18 hours ahead with 5 hours as initial inputs.

For climate, we forecast up to 300 days ahead given 60 days of initial observations.

For Genz dynamics, we forecast for 80 steps given 5 initial steps.

All results are averages over 3 runs.

We now present the long-term forecasting accuracy of TLSTM in nonlinear systems.

FIG4 shows the test prediction error (in RMSE) for varying forecasting horizons for different datasets.

We can see that TLSTM notably outperforms all baselines on all datasets in this setting.

In particular, TLSTM is more robust to long-term error propagation.

We observe two salient benefits of using TT-RNNs over the unfactorized models.

First, MRNN and MLSTM can suffer from overfitting as the number of weights increases.

Second, on traffic, unfactorized models also show considerable instability in their long-term predictions.

These results suggest that tensor-train neural networks learn more stable representations that generalize better for long-term horizons.

To get intuition for the learned models, we visualize the best performing TLSTM and baselines in FIG5 for the Genz function "corner-peak" and the statetransition function.

We can see that TLSTM can almost perfectly recover the original function, while LSTM and MLSTM only correctly predict the mean.

These baselines cannot capture the dynamics fully, often predicting an incorrect range and phase for the dynamics.

In FIG6 we show predictions for the real world traffic and climate dataset.

We can see that the TLSTM corresponds significantly better with ground truth in long-term forecasting.

As the ground truth time series is highly chaotic and noisy, LSTM often deviates from the general trend.

While both MLSTM and TLSTM can correctly learn the trend, TLSTM captures more detailed curvatures due to the inherent high-order structure.

Speed Performance Trade-off We now investigate potential trade-offs between accuracy and computation.

FIG7 displays the validation loss with respect to the number of steps, for the best performing models on long-term forecasting.

We see that TT-RNNs converge significantly faster than other models, and achieve lower validation-loss.

This suggests that TT-RNN has a more efficient representation of the nonlinear dynamics, and can learn much faster as a result.

Hyper-parameter Analysis The TLSTM model is equipped with a set of hyper-parameters, such as tensor-train rank and the number of lags.

We perform a random grid search over these hyperparameters and showcase the results in Table 1 .

In the top row, we report the prediction RMSE for the largest forecasting horizon w.r.t tensor ranks for all the datasets with lag 3.

When the rank is too low, the model does not have enough capacity to capture non-linear dynamics.

when the rank is too high, the model starts to overfit.

In the bottom row, we report the effect of changing lags (degree of orders in Markovian dynamics).

For each setting, the best r is determined by cross-validation.

For different forecasting horizon, the best lag value also varies.

We have also evaluated TT-RNN on long-term forecasting for chaotic dynamics, such as the Lorenz dynamics (see FIG8 ).

Such dynamics are highly sensitive to input perturbations: two close points can move exponentially far apart under the dynamics.

This makes long-term forecasting highly challenging, as small errors can lead to catastrophic longterm errors.

FIG8 shows that TT-RNN can predict up to T = 40 steps into the future, but diverges quickly beyond that.

We have found no state-of-the-art prediction model is stable in this setting.

Classic work in time series forecasting has studied auto-regressive models, such as the ARMA or ARIMA model BID2 , which model a process x(t) linearly, and so do not capture nonlinear dynamics.

Our method contrasts with this by explicitly modeling higher-order dependencies.

Using neural networks to model time series has a long history.

More recently, they have been applied to room temperature prediction, weather forecasting, traffic prediction and other domains.

We refer to BID13 for a detailed overview of the relevant literature.

From a modeling perspective, BID7 ) considers a high-order RNN to simulate a deterministic finite state machine and recognize regular grammars.

This work considers a second order mapping from inputs x(t) and hidden states h(t) to the next state.

However, this model only considers the most recent state and is limited to two-way interactions.

BID17 proposes multiplicative RNN that allow each hidden state to specify a different factorized hidden-to-hidden weight matrix.

A similar approach also appears in BID14 , but without the factorization.

Our method can be seen as an efficient generalization of these works.

Moreover, hierarchical RNNs have been used to model sequential data at multiple resolutions, e.g. to learn both short-term and long-term human behavior BID20 .Tensor methods have tight connections with neural networks.

For example, BID4 shows convolutional neural networks have equivalence to hierarchical tensor factorizations.

BID10 BID19 employs tensor-train to compress large neural networks and reduce the number of weights.

BID19 forms tensors from reshaping inputs and decomposes the input-output weights.

Our model forms tensors from high-order hidden states and decomposes the hidden-output weights.

BID16 propose to parameterizes the supervised learning models with matrix-product states for image classification.

This work however, to the best of our knowledge, is the first work to consider tensor networks in RNNS for sequential prediction tasks for learning in environments with nonlinear dynamics.

In this work, we considered forecasting under nonlinear dynamics.

We propose a novel class of RNNs -TT-RNN.

We provide approximation guarantees for TT-RNN and characterize its representation power.

We demonstrate the benefits of TT-RNN to forecast accurately for significantly longer time horizon in both synthetic and real-world multivariate time series data.

As we observed, chaotic dynamics still present a significant challenge to any sequential prediction model.

Hence, it would be interesting to study how to learn robust models for chaotic dynamics.

In other sequential prediction settings, such as natural language processing, there does not (or is not known to) exist a succinct analytical description of the data-generating process.

It would be interesting to further investigate the effectiveness of TT-RNNs in such domains as well.

We provide theoretical guarantees for the proposed TT-RNN model by analyzing a class of functions that satisfy some regularity condition.

For such functions, tensor-train decompositions preserve weak differentiability and yield a compact representation.

We combine this property with neural network estimation theory to bound the approximation error for TT-RNN with one hidden layer, in terms of: 1) the regularity of the target function f , 2) the dimension of the input space, and 3) the tensor train rank.

In the context of TT-RNN, the target function f (x) with x = s ⊗ . . .

⊗ s, is the system dynamics that describes state transitions, as in (6).

Let us assume that f (x) is a Sobolev function: f ∈ H k µ , defined on the input space I = I 1 × I 2 × · · ·

I d , where each I i is a set of vectors.

The space H k µ is defined as the set of functions that have bounded derivatives up to some order k and are L µ -integrable: DISPLAYFORM0 where D (i) f is the i-th weak derivative of f and µ ≥ 0.

Any Sobolev function admits a Schmidt decomposition: DISPLAYFORM0 , where {λ} are the eigenvalues and {γ}, {φ} are the associated eigenfunctions.

Hence, we can decompose the target function f ∈ H k µ as: DISPLAYFORM1 where DISPLAYFORM2 We can truncate Eqn 13 to a low dimensional subspace (r < ∞), and obtain the functional tensor-train (FTT) approximation of the target function f : DISPLAYFORM3 .FTT approximation in Eqn 13 projects the target function to a subspace with finite basis.

And the approximation error can be bounded using the following Lemma: Lemma 7.1 (FTT Approximation BID1 ).

Let f ∈ H k µ be a Hölder continuous function, defined on a bounded domain I = I 1 × · · · ×

I d ⊂ R d with exponent α > 1/2, the FTT approximation error can be upper bounded as DISPLAYFORM4 for r ≥ 1 and DISPLAYFORM5 Lemma 7.1 relates the approximation error to the dimension d, tensor-train rank r,and the regularity of the target function k. In practice, TT-RNN implements a polynomial expansion of the input states s, using powers [s, s ⊗2 , · · · , s ⊗p ] to approximate f T T , where p is the degree of the polynomial.

We can further use the classic spectral approximation theory to connect the TT-RNN structure with the degree of the polynomial, i.e., the order of the tensor.

Let DISPLAYFORM6 Given a function f and its polynomial expansion P T T , the approximation error is therefore bounded by: 2 A weak derivative generalizes the derivative concept for (non)-differentiable functions and is implicitly defined as: DISPLAYFORM7 Where p is the order of tensor and r is the tensor-train rank.

As the rank of the tensor-train and the polynomial order increase, the required size of the hidden units become smaller, up to a constant that depends on the regularity of the underlying dynamics f .

We trained all models using the RMS-prop optimizer and employed a learning rate decay of 0.8 schedule.

We performed an exhaustive search over the hyper-parameters for validation.

TAB3 reports the hyper-parameter search range used in this work.

For all datasets, we used a 80% − 10% − 10% train-validation-test split and train for a maximum of 1e 4 steps.

We compute the moving average of the validation loss and use it as an early stopping criteria.

We also did not employ scheduled sampling, as we found training became highly unstable under a range of annealing schedules.

Genz Genz functions are often used as basis for evaluating high-dimensional function approximation.

In particular, they have been used to analyze tensor-train decompositions BID1 .

There are in total 7 different Genz functions.

(1) g 1 (x) = cos(2πw + cx), (2) DISPLAYFORM0 2 π|x−w| (6) DISPLAYFORM1 For each function, we generated a dataset with 10, 000 samples using FORMULA1 with w = 0.5 and c = 1.0 and random initial points draw from a range of [−0.1, 0.1].Traffic We use the traffic data of Los Angeles County highway network collected from California department of transportation http://pems.dot.ca.gov/. The dataset consists of 4 month speed readings aggregated every 5 minutes .

Due to large number of missing values (∼ 30%) in the raw data, we impute the missing values using the average values of non-missing entries from other sensors at the same time.

In total, after processing, the dataset covers 35 136, time-series.

We treat each sequence as daily traffic of 288 time stamps.

We up-sample the dataset every 20 minutes, which results in a dataset of 8 784 sequences of daily measurements.

We select 15 sensors as a joint forecasting tasks.

Climate We use the daily maximum temperature data from the U.S. Historical Climatology Network (USHCN) daily (http://cdiac.ornl.gov/ftp/ushcn_daily/) contains daily measurements for 5 climate variables for approximately 124 years.

The records were collected across more than 1 200 locations and span over 45 384 days.

We analyze the area in California which contains 54 stations.

We removed the first 10 years of day, most of which has no observations.

We treat the temperature reading per year as one sequence and impute the missing observations using other non-missing entries from other stations across years.

We augment the datasets by rotating the sequence every 7 days, which results in a data set of 5 928 sequences.

We also perform a DickeyFuller test in order to test the null hypothesis of whether a unit root is present in an autoregressive model.

The test statistics of the traffic and climate data is shown in TAB5 , which demonstrate the non-stationarity of the time series.

Genz functions are basis functions for multi-dimensional FIG0 visualizes different Genz functions, realizations of dynamics and predictions from TLSTM and baselines.

We can see for "oscillatory", "product peak" and "Gaussian ", TLSTM can better capture the complex dynamics, leading to more accurate predictions.

Chaotic dynamics such as Lorenz attractor is notoriously different to lean in non-linear dynamics.

In such systems, the dynamics are highly sensitive to perturbations in the input state: two close points can move exponentially far apart under the dynamics.

We also evaluated tensor-train neural networks on long-term forecasting for Lorenz attractor and report the results as follows:

Lorenz The Lorenz attractor system describes a two-dimensional flow of fluids (see FIG8 ): dx dt = σ(y − x), dy dt = x(ρ − z) − y, dz dt = xy − βz, σ = 10, ρ = 28, β = 2.667.This system has chaotic solutions (for certain parameter values) that revolve around the so-called Lorenz attractor.

We simulated 10 000 trajectories with the discretized time interval length 0.01.

We sample from each trajectory every 10 units in Euclidean distance.

The dynamics is generated using σ = 10 ρ = 28, β = 2.667.

The initial condition of each trajectory is sampled uniformly random from the interval of [−0.1, 0.1].

FIG0 shows 45 steps ahead predictions for all models.

HORNN is the full tensor TT-RNN using vanilla RNN unit without the tensor-train decomposition.

We can see all the tensor models perform better than vanilla RNN or MRNN.

TT-RNN shows slight improvement at the beginning state.

TT-RNN shows more consistent, but imperfect, predictions, whereas the baselines are highly unstable and gives noisy predictions.

<|TLDR|>

@highlight

Accurate forecasting over very long time horizons using tensor-train RNNs