Granger causality is a widely-used criterion for analyzing interactions in large-scale networks.

As most physical interactions are inherently nonlinear, we consider the problem of inferring the existence of pairwise Granger causality between nonlinearly interacting stochastic processes from their time series measurements.

Our proposed approach relies on modeling the embedded nonlinearities in the measurements using a component-wise time series prediction model based on Statistical Recurrent Units (SRUs).

We make a case that the network topology of Granger causal relations is directly inferrable from a structured sparse estimate of the internal parameters of the SRU networks trained to predict the processes’ time series measurements.

We propose a variant of SRU, called economy-SRU, which, by design has considerably fewer trainable parameters, and therefore less prone to overfitting.

The economy-SRU computes a low-dimensional sketch of its high-dimensional hidden state in the form of random projections to generate the feedback for its recurrent processing.

Additionally, the internal weight parameters of the economy-SRU are strategically regularized in a group-wise manner to facilitate the proposed network in extracting meaningful predictive features that are highly time-localized to mimic real-world causal events.

Extensive experiments are carried out to demonstrate that the proposed economy-SRU based time series prediction model outperforms the MLP, LSTM and attention-gated CNN-based time series models considered previously for inferring Granger causality.

The physical mechanisms behind the functioning of any large-scale system can be understood in terms of the networked interactions between the underlying system processes.

Granger causality is one widely-accepted criterion used in building network models of interactions between large ensembles of stochastic processes.

While Granger causality may not necessarily imply true causality, it has proven effective in qualifying pairwise interactions between stochastic processes in a variety of system identification problems, e.g., gene regulatory network mapping (Fujita et al. (2007) ), and the mapping of human brain connectome (Seth et al. (2015) ).

This perspective has given rise to the canonical problem of inferring pairwise Granger causal relationships between a set of stochastic processes from their time series measurements.

At present, the vast majority of Granger causal inference methods adopt a model-based inference approach whereby the measured time series data is modeled using with a suitable parameterized data generative model whose inferred parameters ultimately reveal the true topology of pairwise Granger causal relationships.

Such methods typically rely on using linear regression models for inference.

However, as illustrated in the classical bivariate example by Baek & Brock (1992) , linear model-based Granger causality tests can fail catastrophically in the presence of even mild nonlinearities in the measurements, thus making a strong case for our work which tackles the nonlinearities in the measurements by exploring new generative models of the time series measurements based on recurrent neural networks.

Consider a multivariate dynamical system whose evolution from an initial state is fully characterized by n distinct stochastic processes which can potentially interact nonlinearly among themselves.

Our goal here is to unravel the unknown nonlinear system dynamics by mapping the entire network of pairwise interactions between the system-defining stochastic processes, using Granger causality as the qualifier of the individual pairwise interactions.

In order to detect the pairwise Granger causal relations between the stochastic processes, we assume access to their concurrent, uniformly-sampled measurements presented as an n-variate time series x = {x t : t ∈ N} ⊂ R n .

Let x t,i denote the i th component of the n-dimensional vector measurement x t , representing the measured value of process i at time t. Motivated by the framework proposed in Tank et al. (2017) , we assume that the measurement samples x t , t ∈ N are generated sequentially according to the following nonlinear, component-wise autoregressive model:

x t,i = f i (x t−p:t−1,1 , x t−p:t−1,2 , . . . , x t−p:t−1,n ) + e t,i , i = 1, 2, . . .

n,

where x t−p:t−1,j {x t−1,j , x t−2,j , . . .

, x t−p,j } represents the most recent p measurements of the j th component of x in the immediate past relative to current time t. The scalar-valued component generative function f i captures all of the linear and nonlinear interactions between the n stochastic processes up to time t − 1 that decide the measured value of the i th stochastic process at time t.

The residual e i,t encapsulates the combined effect of all instantaneous and exogenous factors influencing the measurement of process i at time t, as well as any imperfections in the presumed model.

Equation 1 may be viewed as a generalization of the linear vector autoregressive (VAR) model in the sense that the components of x can be nonlinearly dependent on one another across time.

The value p is loosely interpreted to be the order of the above nonlinear autoregressive model.

We now proceed to interpret Granger causality in the context of the above component-wise time series model.

Recalling the standard definition by Granger (1969) , a time series v is said to Granger cause another time series u if the past of v contains new information above and beyond the past of u that can improve the predictions of current or future values of u. For x with its n components generated according to equation 1, the concept of Granger causality can be extended as suggested by Tank et al. (2018) as follows.

We say that series j does not Granger cause series i if the componentwise generative function f i does not depend on the past measurements in series j, i.e., for all t ≥ 1 and all distinct pairs x t−p:t−1,j and x t−p:t−1,j , f i (x t−p:t−1,1 , . . .

, x t−p:t−1,j , . . .

, x t−p:t−1,n ) = f i x t−p:t−1,1 , . . . , x t−p:t−1,j , . . . , x t−p:t−1,n .

(2) From equation 1, it is immediately evident that under the constraint in equation 2, the past of series j does not assert any causal influence on series i, in alignment with the core principle behind Granger causality.

Based on the above implication of equation 2, the detection of Granger noncausality between the components of x translates to identifying those components of x whose past is irrelevant to the functional description of each individual f i featured in equation 1.

Note that any reliable inference of pairwise Granger causality between the components of x is feasible only if there are no unobserved confounding factors in the system which could potentially influence x. In this work, we assume that the system of interest is causally sufficient (Spirtes & Zhang (2016)), i.e., none of the n stochastic processes (whose measurements are available) have a common Granger-causing-ancestor that is unobserved.

We undertake a model-based inference approach wherein the time series measurements are used as observations to learn an autoregressive model which is anatomically similar to the componentwise generative model described in equation 1 except for the unknown functions f i replaced with their respective parameterized approximations denoted by g i .

Let Θ i , 1 ≤ i ≤ n denote the complete set of parameters encoding the functional description of the approximating functions {g i } n i=1 .

Then, the pairwise Granger causality between series i and the components of x is deduced from Θ i which is estimated by fitting g i 's output to the ordered measurements in series i. Specifically, if the estimated Θ i suggests that g i 's output is independent of the past measurements in series j, then we declare that series j is Granger noncausal for series i.

We aim to design the approximation function g i to be highly expressive and capable of well-approximating any intricate causal coupling between the components of x induced by the component-wise function f i , while simultaneously being easily identifiable from underdetermined measurements.

By virtue of their universal approximation property (Schäfer & Zimmermann (2006) ), recurrent neural networks or RNNs are a particularly ideal choice for g i towards inferring the pairwise Granger causal relationships in x. In this work, we investigate the use of a special type of RNN called the statistical recurrent unit (SRU) for inferring pairwise Granger causality between multiple nonlinearly interacting stochastic processes.

Introduced by Oliva et al. (2017) , an SRU is a highly expressive recurrent neural network designed specifically for modeling multivariate time series data with complex-nonlinear dependencies spanning multiple time lags.

Unlike the popular gated RNNs (e.g., long short-term memory (LSTM) (Hochreiter & Schmidhuber (1997) ) and gated recurrent unit (GRU)) (Chung et al. (2014) ), the SRU's design is completely devoid of the highly nonlinear sigmoid gating functions and thus less affected by the vanishing/exploding gradient issue during training.

Despite its simpler ungated architecture, an SRU can model both short and long-term temporal dependencies in a multivariate time series.

It does so by maintaining multi-time scale summary statistics of the time series data in the past, which are preferentially sensitive to different older portions of the time series x. By taking appropriate linear combinations of the summary statistics at different time scales, an SRU is able to construct predictive causal features which can be both highly component-specific and lag-specific at the same time.

From the causal inference perspective, this dual-specificity of the SRU's predictive features is its most desirable feature, as one would argue that causal effects in reality also tend to be highly localized in both space and time.

The main contributions of this paper can be summarized as follows:

1.

We propose the use of statistical recurrent units (SRUs) for detecting pairwise Granger causality between the nonlinearly interacting stochastic processes.

We show that the entire network of pairwise Granger causal relationships can be inferred directly from the regularized block-sparse estimate of the input-layer weight parameters of the SRUs trained to predict the time series measurements of the individual processes.

2.

We propose a modified SRU architecture called economy SRU or eSRU in short.

The first of the two proposed modifications is aimed at substantially reducing the number of trainable parameters in the standard SRU model without sacrificing its expressiveness.

The second modification entails regularizing the SRU's internal weight parameters to enhance the interpretability of its learned predictive features.

Compared to the standard SRU, the proposed eSRU model is considerably less likely to overfit the time series measurements.

3.

We conduct extensive numerical experiments to demonstrate that eSRU is a compelling model for inferring pairwise Granger causality.

The proposed model is found to outperform the multi-layer perceptron (MLP), LSTM and attention-gated convolutional neural network (AG-CNN) based models considered in the earlier works.

In the proposed scheme, each of the unknown generative functions f i , 1 ≤ i ≤ n in the presumed component-wise model of x in (1) is individually approximated by a distinct SRU network.

The i th SRU network sequentially processes the time series measurements x and outputs a next-step prediction sequencex

denotes the predicted value of component series i at time t + 1.

The predictionx i,t+1 is computed in a recurrent fashion by combining the current input sample x t at time t with the summary statistics of past samples of x up to and including time t − 1 as illustrated in Figure 1 .

The following update equations describe the sequential processing of the input time series x within the i th SRU network in order to generate a prediction of x i,t+1 .

Feedback:

Recurrent statistics:

Multi-scale summary statistics:

Single-scale summary statistics:

Output features:

Output prediction:

The function h in the above updates is the elementwise Rectified Linear Unit (ReLU) operator, h(·) := max(·, 0), which serves as the nonlinear activation in the three dedicated single layer neural networks that generate the recurrent statistics φ i,t , the feedback r i,t and the output features o i,t in the i th SRU network.

In order to generate the next-step prediction of series i at time t, the i th SRU network first prepares the feedback r i,t by nonlinearly transforming its last hidden state u i,t−1 .

As stated in equation 3a, a single layer ReLU network parameterized by weight matrix W Linear C o n c a t e n a t e Concatenate H α1 (z)

. . .

For values of scale α ≈ 1, the single-scale summary statistic u α i,t in equation 3d is more sensitive to the recent past measurements in x. On the other hand, α ≈ 0 yields a summary statistic that is more representative of the older portions of the input time series.

Oliva et al. (2017) elaborates on how the SRU is able to generate output features (o i,t , 1 ≤ i ≤ n) that are preferentially sensitive to the measurements from specific past segments of x by taking appropriate linear combinations of the summary statistics corresponding to different values of α in A. in regulates the influence of the individual components of the input time series x on the generation of the recurrent statistics φ i,t , and ultimately the next-step prediction of series i.

In real-world dynamical systems, the networked interactions are typically sparse which implies that very few dimensions of the input time series x actually play a role in the generation of its individual components.

Bearing this property of the networked interactions in mind, we are interested in learning the parameters Θ We propose to learn the parameters Θ

SRU of the i th SRU network by minimizing the penalized mean squared prediction error loss as shown below.

In the above, the network outputx i,t depends nonlinearly on W

in according to the composite relation described by the updates (3a)-(3f) and W in (:, j) being estimated as the all-zeros vector is that the past measurements in series j do not influence the predicted future value of series i.

In this case, we declare that series j does not Granger-cause series i. Moreover, the index set supporting the non-zero columns in the estimated weight matrixŴ

in enumerates the components of x which are likely to Granger-cause series i. Likewise, the entire network of pairwise Granger causal relationships in x can be deduced from the non-zero column support of the estimated weight matrices W (i) in , 1 ≤ i ≤ n in the n SRU networks trained to predict the components of x.

The component-wise SRU optimization problem in equation 4 is nonconvex and potentially has multiple local minima.

To solve forΘ

SRU , we use first-order gradient-based methods such as stochastic gradient descent which have been found to be consistently successful in finding good solutions of nonconvex deep neural network optimization problems (Allen-Zhu et al. (2019)).

Since our approach of detecting Granger noncausality hinges upon correctly identifying the all-zero columns of W (i) in , it is important that the first-order gradient based parameter updates used for minimizing the penalized SRU loss ensure that majority of the coefficients in W in , we follow the same approach as Tank et al. (2018) and resort to a first-order proximal gradient descent algorithm to find a regularized solution of the SRU optimization.

The gradients needed for executing the gradient descent updates of the SRU network parameters are computed efficiently using the backpropagation through time (BPTT) procedure (Jaeger (2002) ).

By computing the summary statistics of past measurements at sufficiently granular time scales, an SRU can learn predictive causal features which are highly localized in time.

While a higher granularity of α in A translates to a more general SRU model that fits better to the time series measurements, it also entails substantial increase in the number of trainable parameters.

Since measurement scarcity is typical in causal inference problems, the proposed component-wise SRU based time series prediction model is usually overparameterized and thus susceptible to overfitting.

The typical high dimensionality of the recurrent statistic φ t accentuates this issue.

To alleviate the overfitting concerns, we propose two modifications to the standard SRU (Oliva et al. (2017) ) aimed primarily at reducing its likelihood of overfitting the time series measurements.

The modifications are relevant regardless of the current Granger causal inference context, and henceforth we refer to the modified SRU as Economy-SRU (eSRU).

We propose to reduce the number of trainable parameters in the i th SRU network by substituting the feedback ReLU network parameterized by W

Figure 2: Proposed two-stage feedback in economy-SRU.

the associated time series measurements, their highdimensional summary statistics learned by the SRU network as u i,t tend to be highly structured, and thus u i,t has significantly fewer degrees of freedom relative to its ambient dimension.

Thus, by projecting the md φ -dimensional u i,t onto the d r ( md φ ) rows of D (i) r , we obtain its low-dimensional embedding v i,t which nonetheless retains most of the contextual information conveyed by the uncompressed u i,t 1 Johnson & Lindenstrauss (1984) ; Dirksen (2014) .

The second stage of the proposed feedback network is a single/multi-layer ReLU network which maps the sketched summary statistics v i,t to the feedback vector r i,t .

The second stage ReLU network is parameterized by weight matrix W ,(i) r ∈ R dr×d r and bias b

Compared to the standard SRU's feedback whose generation is controlled by md φ d r + d r trainable parameters, the proposed feedback network has only d r d r + d r trainable parameters, which is substantially fewer when d r md φ .

Consequently, the modified SRU is less susceptible to overfitting.

In the standard SRU proposed by Oliva et al. (2017) , there are no restrictions on the weight matrix W In this spirit, we propose the following penalized optimization problem to estimate the parameters Θ

r } of the eSRU model equipped with the two-stage feedback proposed in Section 4.1:

(5) Here λ 1 and λ 2 are positive constants that bias the group sparse penalizations against the eSRU's fit to the measurements in the i th component series.

The term

o obtained by extracting the weight coefficients indexed by set G j,k .

As shown via an example in Fig. 3 , the index set G j,k enumerates the m weight coefficients in the row vector W

o to the effect that each predictive feature in o i,t depends on only a few components of the recurrent statistic φ i,t via their linearly mixed multi-scale exponentially weighted averages.

We opine that the learned linear mixtures, represented by the intermediate products W

, are highly sensitive to certain past segments of the input time series x. Consequently, the output features in o i,t are both time-localized and component-specific, a common trait of real-world causal effects.

1 Gaussian random matrices of appropriate dimensions are approximately isometries with overwhelming probability (Johnson & Lindenstrauss (1984) ).

However, instead of using n independent instantiations of a Gaussian random matrix for initializing D (i) r , 1 ≤ i ≤ n, we recommend initializing them with the same random matrix, as the latter strategy reduces the probability that any one of them is spurious encoder by n-fold.

Figure 3: An illustration of the proposed group-wise mixing of the multi-timescale summary statistics u i,t in the i th SRU (with d φ = 5) towards generating the j th predictive feature in o i,t .

The weights corresponding to the same colored connections belong to the same group.

The above group-sparse regularization of the weight coefficients in W in , is pivotal to enforcing that the occurrence of any future pattern in a time series can be attributed to the past occurrences of a few highly time-localized patterns in the ancestral time series.

The results of our numerical experiments further confirm that by choosing λ 1 and λ 2 appropriately, the proposed group-wise sparsity inducing regularization of W

We evaluate the performance of the proposed SRU-and eSRU-based component-wise time series models in inferring pairwise Granger causal relationships in a multivariate time series.

The proposed models are compared to the existing MLP-and LSTM-based models in Tank et al. (2018) and the attention-gated CNN-based model (referred hereafter as Temporal Causal Discovery Framework (TCDF)) in Nauta et al. (2019) .

To ensure parity between the competing models, the maximum size of all the input/hidden/output layers in the different NN/RNN time series models is fixed to 10, unless specified otherwise.

The complete list of tuned hyperparameters of the considered models used for different datasets is provided in Appendix G. The performance of each method is qualified in terms of its AUROC (Area Under the Receiver Operating Characteristic curve).

Here, the ROC curve illustrates the trade off between the true-positive rate (TPR) and the false-positive rate (FPR) achieved by the methods towards the detection of n 2 pairwise Granger causal relationships between the n measured processes in the experiment.

The ROC curves of SRU and eSRU models are obtained by sweeping through different values of the regularization parameter λ 1 in equation 4 and equation 5, respectively.

Likewise, the ROCs of component-wise MLP and LSTM models are obtained by varying λ 1 's counterpart in Tank et al. (2018) .

For TCDF, the ROC curve is obtained by varying the threshold that is applied to attention scores of the trained AG-CNN model in Nauta et al. (2019).

In the first set of experiments, the time series measurements x intended for Granger causal inference are generated according to the Lorenz-96 model which has been extensively used in climate science for modeling and prediction purposes (Schneider et al. (2017) ).

In the Lorenz-96 model of an nvariable system, the individual state trajectories of the n variables are governed by the following set of odinary differential equations:

where the first and the second terms on the RHS represent the advection and the diffusion in the system, respectively, and the third term F is the magnitude of the external forcing.

The system dynamics becomes increasingly chaotic for higher values of F (Karimi & Paul (2010) ).

We evaluate and compare the accuracy of the proposed methods in inferring pairwise Granger causal relationships between n = 10 variables with Lorenz-96 dynamics.

We consider two settings: F = 10 and F = 40 in order to simulate two different strengths of nonlinearity in the causal interactions between the variables.

Here, the ground truth is straightforward i.e., for any 1 ≤ i ≤ n, the i th component of time series x is Granger caused by its components with time indices from i − 2 to i + 1.

In the case of weak nonlinear interactions (F = 10), from Table 1a , we observe that eSRU achieves the highest AUROC among all competing models.

The gap in performance is more pronounced when fewer time series measurements (T = 250) are available.

In case of stronger nonlinear interactions (F = 40), we observe that both SRU and eSRU are the only models that are able to perfectly recover the true Granger causal network (Table 1b) .

Surprisingly, the SRU and eSRU models perform poorer when F is small.

This could be attributed to the proposed models not sufficiently regularized when fitted to weakly-interacting time series measurements that are less nonlinear.

In the second set of simulations, we consider the time series measurements x to be generated according to a 3 rd order linear VAR model:

where the matrices A (i) , i = 1, 2, 3 contain the regression coefficients which model the linear interactions between its n = 10 components.

The noise term w t is Gaussian distributed with zero mean and covariance 0.01I.

We consider a sparse network of Granger causal interactions with only 30% of the regression coefficients in A i selected uniformly being non-zero and the regression matrices A i being collectively joint sparse (same setup as in Bolstad et al. (2011) ).

All non-zero regression coefficients are set equal to 0.0994 which guarantees the stability of the simulated VAR process.

From Table 2 , we observe that all time series models generally achieve a higher AUROC as the number of measurements available increases.

For T = 500, the component-wise MLP and the proposed eSRU are statistically tied when comparing their average AUROCs.

For T = 1000, eSRU significantly outperforms the rest of the time series models and is able to recover the true Granger causal network almost perfectly.

In the third set of experiments, we apply the different learning methods to estimate the connections in the human brain from simulated blood oxygenation level dependent (BOLD) imaging data.

Here, the individual components of x comprise T = 200 time-ordered samples of the BOLD signals simulated for n = 15 different brain regions of interest (ROIs) in a human subject.

To conduct the experiments, we use simulated BOLD time series measurements corresponding to the five different human subjects (labelled as 2 to 6) in the Sim-3.mat file shared at https://www.fmrib.ox.ac.uk/datasets/netsim/index.html.

The generation of the Sim3 dataset is described in Smith et al. (2011) .

The goal here is to detect the directed connectivity between different brain ROIs in the form of pairwise Granger causal relationships between the components of x.

From Table 3 , it is evident that eSRU is more robust to overfitting compared to the standard SRU and detects the true Granger causal relationships more reliably.

Interestingly, a single-layer cMLP model is found to outperform more complex cLSTM and attention gated-CNN (TCDF) models; however we expect the latter models to perform better when more time series measurements are available.

In the final set of experiments, we evaluate the performance of the different time series models in inferring gene regulation networks synthesized for the DREAM-3 In Silico Network Challenge (Prill et al. (2010); Marbach et al. (2009) ).

Here, the time series x represents the in silico measurements of the gene expression levels of n = 100 genes, available for estimating the gene regulatory networks of E.coli and yeast.

A total of five gene regulation networks are to be inferred (two for E.coli and three for yeast) from the networks' gene expression level trajectories recorded while they recover from 46 different perturbations (each trajectory has 21 time points).

All NN/RNN models are implemented with 10 neurons per layer, except for the componentwise MLP model which has 5 neurons per layer.

From Table 4 , we can observe that the proposed SRU and eSRU models are generally more accurate

In this work, we addressed the problem of inferring pairwise Granger causal relationships between stochastic processes that interact nonlinearly.

We showed that the such causality between the processes can be robustly inferred from the regularized internal parameters of the proposed eSRU-based recurrent models trained to predict the time series measurements of the individal processes.

Future work includes:

i Investigating the use of other loss functions besides the mean-square error loss which can capture the exogenous and instantaneous causal effects in a more realistic way.

ii Incorporating unobserved confounding variables/processes in recurrent models.

iii Inferring Granger causality from multi-rate time series measurements.

Initial efforts in testing for nonlinear Granger causality focused mostly on the nonparameteric approach.

Baek & Brock (1992) Diks & Panchenko (2006) to the multivariate setting.

The biggest common drawback of these nonparameteric tests is the large sample sizes required to robustly estimate the conditional probabilities that constitute the test statistic.

Furthermore, the prevalent strategy in these methods of testing each one of the variable-pairs individually to detect pairwise Granger causality is unappealing from a computational standpoint, especially when a very large number of variables are involved.

In the model driven approach, the Granger causal relationships are inferred directly from the parameters of a data generative model fitted to the time series measurements.

Compared to the nonparameteric approach, the model-based inference approach is considerably more sample efficient, however the scope of inferrable causal dependencies is dictated by the choice of data generative model.

Nonlinear kernel based regression models have been found to be reasonably effective in testing of nonlinear Granger causality.

Kernel methods rely on linearization of the causal interactions in a kernel-induced high dimensional feature space; the linearized interactions are subsequently modeled using a linear VAR model in the feature space.

Based on this idea, Marinazzo et al. (2008) proposes a kernel Granger causality index to detect pairwise nonlinear Granger causality in the multivariate case.

In Sindhwani et al. (2013); Lim et al. (2014) , the nonlinear dependencies in the time series measurements are modeled using nonlinear functions expressible as sums of vector valued functions in the induced reproducing kernel Hilbert space (RKHS) of a matrix-valued kernel.

In Lim et al. (2014) , additional smoothness and structured sparsity constraints are imposed on the kernel parameters to promote consistency of the time series fitted nonlinear model.

Shen et al. (2016) proposes a nonlinear kernel-based structural VAR model to capture instantaneous nonlinear interactions.

The existing kernel based regression models are restrictive as they consider only additive linear combinations of the RKHS functions to approximate the nonlinear dependencies in the time series.

Furthermore, deciding the optimal order of kernel based regression models is difficult as it requires prior knowledge of the mimimum time delay beyond which the causal influences are negligible.

By virtue of their universal approximation ability, RNNs offer a pragmatic way forward in modeling of complex nonlinear dependencies in the time series measurements for the purpose of inferring Granger causal relationships.

However, they all adopt the same naïve strategy whereby each pairwise causal relationship is tested individually by estimating its causal connection strength.

The strength of the causal connection from series j to series i is determined by the ratio of mean-squared prediction errors incurred by unrestricted and restricted RNN models towards predicting series i using the past measurement sequences of all n component including and excluding the j th component alone, respectively.

The pairwise testing strategy however does not scale well computationally as the number of component series becomes very large.

This strategy also fails to exploit the typical sparse connectivity of networked interactions between the processes which has unlocked significant performance gains in the existing linear methods (Bahadori & Liu (2013) ; Bolstad et al. (2011) ).

In a recent work by Tank et al. (2018) , the pairwise Granger causal relationships are inferred directly from the weight parameters of component-wise MLP or LSTM networks fitted to the time series measurements.

By enforcing column-sparsity of the input-layer weight matrices in the fitted MLP/LSTM models, their proposed approach returns a sparsely connected estimate of the underlying Granger causal network.

Due to its feedforward architecture, a traditional MLP network is not well-suited for modeling ordered data such as a time series.

Tank et al. (2018) demonstrated that the MLP network can learn short range temporal dependencies spanning a few time delays by letting the network's input stage process multi-lag time series data over sliding windows.

However, modeling long-range temporal dependencies using the same approach requires a larger sliding window size which entails an inconvenient increase in the number of trainable parameters.

The simulation results in Tank et al. (2018) indicate that MLP models are generally outperformed by LSTM models in extracting the true topology of pairwise Granger causality, especially when the processes interact in a highly nonlinear and intricate manner.

While purposefully designed for modeling short and long term temporal dependencies in a time series, the LSTM (Hochreiter & Schmidhuber (1997) ) is very general and often too much overparameterized and thus prone to overfitting.

While using overparameterized models for inference is preferable when there is abundant training data available to leverage upon, there are several applications where the data available for causal inference is extremely scarce.

It is our opinion that using a simpler RNN model combined with meaningful regularization of the model parameters is the best way forward in inferring Granger causal relationships from underdetermined time series measurements.

Building on the ideas put forth by Tank et al. (2018)

Here,

2 is the unregularized SRU loss function, η is the gradientdescent stepsize and S λ1η is the elementwise soft-thresholding operator defined below.

The columns of weight matrix W

in the i th eSRU model are also updated in exactly the same fashion as above.

Likewise, the j th row of the group-norm regularized weight matrix W

o in the eSRU optimization in equation 5 is updated as shown below.

The gradient of the unregularized loss function l i , 1 ≤ i ≤ n associated with the SRU and eSRU models used in the above updates is evaluated via the backpropagation through time (BPTT) procedure (Jaeger (2002) Table 5 , we compare the Granger causality detection performance of this particular eSRU variant and the proposed design wherein D We observe that the performance of these two models is statistically tied, which indicates that the randomly constructed D (i) r is able to distill the necessary information from the high-dimensional summary statistics u i,t−1 required for generating the feedback r i,t .

Based on these results, we recommend using the proposed eSRU design with its randomly constructed encoding map D (i) r , because of its simpler design and reduced training complexity.

In order to highlight the importance of learning time-localized predictive features in detecting Granger causality, we compare the following two time series models: Once again, we use the same experimental settings as mentioned in Section 5.

From Table 6 , we observe that barring the Lorenz-96(T =250/500,F =40) datasets, for which nearly perfect recovery of the Granger causal network is achieved, the average AUROC improves consistently for the other datasets by switching from unstructured ridge regularization to the proposed group-sparse regularization of the output weight matrix W

• Activation function for SRU and eSRU models While the standard SRU proposed by Oliva et al. (2017) uses ReLU neurons, we found in our numerical experiments that using the Exponential Linear Unit (ELU) activation resulted in better performance.

The ELU activation function is defined as

In our simulations, the constant α is set equal to one.

• Number of neural layers in SRU model To approximate the generative functions f i in equation 1, we consider the simplest architecture for the SRU networks, whereby the constituent ReLU networks generating the recurrent features, output features and feedback have a single layer feedforward design with equal number of neurons.

• Number of neural layers in Economy-SRU model The ReLU networks used for generating the recurrent and output features in the proposed eSRU model have a single-layer feedforward design.

However, the second stage of eSRU's modified feedback can be either single or multi-layered feedforward network.

Provided that d r md φ , a multi-layer implementation of the second stage of eSRU's feedback can still have fewer trainable parameters overall compared to the SRU's single layer feedback network.

The simulation results in Section 5 are obtained using a two-layer ReLU network in the second stage of eSRU's feedback for the DREAM-3 experiments, and while using a three-layer design for the Lorenz-96, VAR and NetSim experiments.

• Self-interactions in Dream-3 gene networks

The in-silico gene networks synthesized for the DREAM-3 challenge have no selfconnections.

Noting that none of the Granger causal inference methods evaluated in our experiments intentionally suppress the self-interactions, the reported AUROC values are computed by ignoring any self-connections in the inferred Granger causal networks.

cMLP & cLSTM models Pytorch implementation of the componentwise MLP and LSTM models are taken from https: //github.com/icc2115/Neural-GC.

Pytorch implementation of the attention-gated CNN based Temporal Causal Discovery Framework (TCDF) is taken from https://github.com/M-Nauta/TCDF.

Proposed SRU and Economy-SRU models Pytorch implementations of the proposed componentwise SRU and eSRU models are shared at https://github.com/sakhanna/SRU_for_GCI.

The receiver operating characteristics (ROC) of different Granger causal inference methods are compared in Figures 4-7 .

Here, an ROC curve represents the trade-off between the true-positive rate (TPR) and the false-positive rate (FPR) achieved by a given method while inferring the underlying pairwise Granger causal relationships.

Table 11 : Economy-SRU model configuration

<|TLDR|>

@highlight

A new recurrent neural network architecture for detecting pairwise Granger causality between nonlinearly interacting time series. 