We propose a method to automatically compute the importance of features at every observation in time series, by simulating counterfactual trajectories given previous observations.

We define the importance of each observation as the change in the model output caused by replacing the observation with a generated one.

Our method can be applied to arbitrarily complex time series models.

We compare the generated feature importance to existing methods like sensitivity analyses, feature occlusion, and other explanation baselines to show that our approach generates more precise explanations and is less sensitive to noise in the input signals.

Multi-variate time series data are ubiquitous in application domains such as healthcare, finance, and others.

In such high stakes applications, explaining the model outcome is crucial to build trust among end-users.

Finding the features that drive the output of time series models is a challenging task due to complex non-linear temporal dependencies and cross-correlations in the data.

The explainability problem is significantly exacerbated when more complex models are used.

Most of the current work in time series settings focus on evaluating globally relevant features Hmamouche et al., 2017) .

However, often global feature importance represents relevant features for the entire population, that may not characterize local explanations for individual samples.

Therefore we focus our work on individualized feature importance in time series settings.

In addition, besides identifying relevant features, we also identify the relevant time instances for those specific features, i.e., we identify the most relevant observations.

To the best of our knowledge this is the first sample-specific feature importance explanation benchmark at observation level for time series models.

In this work, we propose a counterfactual based method to learn the importance of every observation in a multivariate time series model.

We assign importance by evaluating the expected change in model prediction had an observation been different.

We generate plausible counterfactual observations based on signal history, to asses temporal changes in the underlying dynamics.

The choice of the counterfactual distribution affects the quality of the explanation.

By generating counterfactuals based on signal history, we ensure samples are realistic under individual dynamics, giving explanations that are more reliable compared to other ad-hoc counterfactual methods.

In this section we describe our method, Feed Forward Counterfactual (FFC), for generating explanation for time series models.

A feature is considered important if it affects the model output the most.

In time series, the dynamics of the features also change over time, which may impact model outcome.

As such it is critical to also identify the precise time points of such changes.

In order to find important observations for highdimensional time series models, we propose to use a counterfactual based method.

Specifically, (a)

(b) FFC procedure: CounterfactualXt is generated using signal history.

We look into the difference of the original output yt andŷ (i,t) where observation x (i,t) is replaced with a counterfactual.

we assign importance to an observation x i,t (feature i at time t) based on its effect on the model output at time T (> t).

We replace observation x i,t with a counterfactualx i,t , to evaluate this effect.

Figure 1 demonstrates how importance of a feature is characterized by replacing an observation with a counterfactual.

Multi-variate time series data is available in the form of X (n) ∈ R d×T (where d is the number of features with T observations over time) for n ∈ [N ] samples.

We are interested in black-box estimators that receive observations up to time t, x t ∈ R d , and generate output y t at every time point t ∈ [T ].

F denotes the target black-box model we wish to explain through the proposed approach called FFC.

For exposition, throughout the paper, the index n over samples has been dropped for notational clarity.

We index features with subscript i. x −i,t indicates features at time t excluding feature i.

The notation used for exposition work is briefly summarized in Table 1 .

Observed outcome of the black-box model F, at time t Generative Model and Estimator Gi :

Latent encoding of history up to t Table 1 : Notation used in the paper.

We assign importance score to each observation x i,t for t ∈ [T ] and i ∈ [d], following the definition: Definition 1.

Feature Importance: The importance of the observation i at time t, denoted by Imp(i, t) is defined as E p(xi,t|X0:t−1) [|F(X 0:t ) − F(X 0:t−1 , x −i,t ,x i,t )|], where | · | denotes the absolute value andx i,t is the counterfactual sample.

That is, the importance of an observation for feature i at time t is defined as the change in model output when the observation is replaced by a generated counterfactual.

The counterfactual observation can come from any distribution, however the quality of the counterfactual random variable directly affects the quality of the explanation.

We generate the counterfactual sample conditioned on signal history up to time t by sampling from the distribution p(x i,t |X 0:t−1 ).

Using a conditional generator guarantees that our counterfactuals are sampled not only within domain but also specifically likely under the individual sample X (n) , as opposed to having a generator that models data across population.

Conditioning on the history also allows us to learn the dynamics of the signal and therefore generate a plausible counterfactual given the past observations.

p(x t |X 0:t−1 ) represents the distribution at time t, if there were no change in the dynamics of the signals.

The counterfactualx i,t is sampled from the marginal distribution p(bx i,t |X 0:t−1 ), obtained from p(x t |X 0:t−1 ).

Let F(X 0:t−1 , x −i,t ,x i,t ) be the output of the model at time T , when x i,t is replaced by the generated counterfactualx i,t .

We estimate feature importance Imp(i, t) as E p(xi,t|X0:t−1) [|F(X 0:t ) − F(X 0:t−1 , x −i,t ,x i,t )|], summarized in figure 2(b).

Our proposed method has the following compelling properties in explaining the estimator F:

Time Importance (TI) For every time series, highlighting relevant time events for different features is important for actionability of the explanations.

For instance in a clinical setting, just knowing a feature like heart rate is relevant, is not sufficient to intervene -it is also important to know when a deterioration had happened.

With FFC, the most eventful time instances can be obtained as:

(1) We can thus rank time instances in order of importance.

That is, time t 1 t 2 , if

Feature Importance (FI) At any time instance t, our method assigns importance to every feature of x i,t .

The magnitude of our importance function reflects relative importance.

Comparing the importance values across features gives the flexibility to report a subset of important features at each time point t and also reflects the correlation between various features of the time series.

We approximate the conditional distribution of p(x t |X 0:t−1 ) using a recurrent latent variable generator model G, introduced in Chung et al. (2015) .

The architecture we use is provided in Figure 2(a) .

The conditional generator G models p(x t |z t−1 ) where z t−1 ∈ R k is the latent representation of history of the time series up to time t. The latent representation is a continuous random variable, modeling the distribution parameters.

We only use past information in the time series to reflect temporal dependencies.

Using the recurrent structure allows us to model a non-stationary generative model that can also handle varying length of observations.

Implementation details of the generator are in the Appendix.

Our counterfactuals are not derived by looking at future values which could be done for reliable imputation.

Counterfactuals should represent the past dynamics.

Note that our derived feature importance is limited by the quality of imputation that may have been utilized by the black-box risk predictor.

For experimental evaluation on the effect of generator specifications on counterfactuals and the quality of explanations, see Section 4.1.

The proposed procedure is summarized in Algorithm 1.

We assume that we are given a trained block box model F, and the data (without labels) it had been trained on.

Using the training data, we first train the generator that generates the conditional distribution, (denoted by G).

In our implementation we model x as a multivariate Gaussian with full covariance to model all existing correlation between features.

The counterfactualx i,t is then sampled from G and passed to the black-box model to evaluate the effect on the black-box outcome.

A common method of explaining model performance, in time-series deep learning, is via visualization of activations of latent layers (Strobelt et al., 2018; Siddiqui et al., 2019; Ming et al., 2017)

Return Importance_M atrix sensitivity analysis (Bach et al., 2015; Yang et al., 2018) .

Understanding latent representations, sensitivity and its relationship to overall model behavior is useful for model debugging.

However, these but are too refined to be useful to the end users like clinicians.

Attention models (Vaswani et al., 2017; Vinayavekhin et al., 2018; Xu et al., 2018) are the most commonly known explanation mechanisms for sequential data.

However, because of the complex mappings to latent space in recurrent models, attention weights cannot be directly attributed to individual observations of the time series (Guo et al., 2018) .

To resolve this issue to some extent, Choi et al. (2016) propose an attention model for mortality prediction of ICU patients based on clinical visits.

However attention weights may not be consistent as explanations (Jain and Wallace, 2019) .

In vision, prior works tackle explainability from the counterfactual perspective, finding regions of the image that affect model prediction the most.

Fong and Vedaldi (2017) assumes higher importance for inputs that when replaced by an uninformative reference value, maximally change the classifier output.

A criticism to such methods is that they may generate out-of-distribution counterfactuals, leading to unreliable explanations.

Chang et al. (2019) address this issue for images using conditional generative models for inpainting regions with realistic counterfactuals.

Evaluating sample based feature importance remains largely unstudied for time series models.

While more widely studied for image classification, (Bach et al., 2015; Fong and Vedaldi, 2017) these methods cannot be directly extended to time series models due to complex time-series dynamics.

Most efforts in this domain focus on population level feature importance (Tyralis and Papacharalampous, 2017) .

Suresh et al. (2017) is one of the few methods addressing sample based feature importance and use a method similar to Fong and Vedaldi (2017), called "feature occlusion".

They replace each time series observation by a sample from uniform noise to evaluate its effect on model outcome to attribute feature importance.

We argue that carefully choosing the counterfactual selection policy is necessary for derive reliable importances.

Specifically, replacing observations with noisy out-of-domain samples can lead to arbitrary changes to model output that are not reflective of systematic behavior in the domain.

Even if an observation is sampled from the domain distribution, it does not characterize temporal dynamics and dependencies well enough, potentially highlighting features that only reflect global model behavior, as opposed to sample specific feature importance.

We therefore model the data-distribution in order to generate reliable counterfactuals.

We demonstrate the implications of the choice of the generator (and hence the counterfactuals) on the quality of explanation.

We evaluate our explainability method for finding important features in time series on 2 simulated datasets and 2 real datasets.

Our goal is two-fold a) comparison to existing feature importance baselines in time series and b) evaluating the choice of generators on the quality of counterfactuals and explanations.

We compare to existing feature importance baselines described below:

1.

Feature Occlusion (FO) (Suresh et al., 2017) : Method introduced in Suresh et al. (2017) .

This method is an ad-hoc approach for generating counterfactuals.

When replacing x i,t with a random sample from the uniform distribution, the change in model output defines the importance for x i,t .

We augment the method introduced in Suresh et al. (2017) by sampling counterfactuals from the bootstrapped distribution over each feature.

This avoids generating out-of-distribution samples.

3.

Sensitivity Analysis (SA): This method evaluates the sensitivity of the output to every observation, by taking the derivative of y t with respect to x i,t , at every time point.

4. LIME (Ribeiro et al., 2016) : One of the most commonly used explainabilty methods that assigns local importance to features.

Although LIME does not assign temporal importance, for this baseline, we use LIME at every time point to generate feature importances.

Evaluating the quality of explanations is challenging due to the lack of a gold standard/ground truth for the explanations.

Additionally, explanations are reflective of model behavior, therefore such evaluations are tightly linked to the reliability of the model itself.

Therefore we created the simulated environment in order to test our method.

In this experiment, we simulate a time series data such that only one feature determines the outcome.

Specifically, the outcome (label) changes to 1 as soon as a spike is observed in the relevant feature.

We keep the task fairly simple for two main reasons: 1) to ensure that the black-box classifier can indeed learn the right relation between the important feature and the outcome, which allows us to focus on evaluating the quality of the explanations without worrying about the quality of the classifier.

2) to have a gold standard for the explanations since the exact important event predictive of the outcome are known.

We expect the explanations to assign importance only to the one relevant feature, at the exact time of spike, even in the presence of spikes in other non-relevant features.

To simulate these data, we generate d = 3 (independent) sequences as a standard non-linear auto-regressive moving average (NARMA) time series of the form:

, where the order is 2 and u ∼ Normal(0, 0.01).

We add linear trends to the features and introduce random spikes over time for every feature.

Note that since spikes are not correlated over time, no of the generators (used in FFC, AFO, FO) will learn to predict it.

The important feature in this setting is feature 1.

The complete procedure is described in Appendix A.2.1.

We train an RNN-based black-box model on this data, resulting in AUC= 0.99 on the test set.

Figure 7 demonstrates explanations of each of the compared approaches on simulated data for 2 test samples.

As shown in Figure 7 (a), Sensitivity analysis does not pick up on the importance of the spike.

Feature occlusion gives false positive importance to spikes that happen in non-important signals as well as the important one.

Augmented feature occlusion resolves this problem since it samples the counterfactuals from the data distribution, however, it generates noisier results as it samples from the bootstrap distribution.

The proposed method (FFC) only assigns importance to the first feature at the time of spike.

Hence, FFC generates fewer false relevance scores.

Note that all baseline methods provide an importance for evry sample at every time point.

The true explanation should highlight feature 1 at time points of spike.

Using this ground truth, we evaluate the AUROC and AUPRC of the generated explanations denoted by (exp).

II, we also show in the third column that the log-probabilities of our counterfactuals are higher under the true distribution.

The first simulation does not necessarily evaluate feature importance under complex state dynamics as is common in applications.

In this simulation, we create a dataset with complex dynamics with known ground truth explanations.

The dataset consists of multivariate time series signals with 3 features.

A Hidden Markov Model with 2 latent states, with linear transition matrix and multivariate normal emission probabilities is used to simulate observations.

The the outcome y is a random variable, which, in state 1, is only affected by feature 1 and in state 2, only affected by feature 2.

Also, we add non-stationarity to the time series by modeling the state transition probabilities as a function of time.

The ground truth explanation for output at time T is the observation x i,t where i is the feature that drives the output in the current state and t indicates when feature i became important.

In a time series setting, a full explanation for outcome at t = T should include the most important feature variable as well as the time point of importance (here state change).

Figure 4 demonstrates assigned importance for a time series sample.

The shaded regions indicate the top 5 important observations (x i,t ) for each method, the color indicating the corresponding feature i. AFO, FO and FFC are able to learn the state dynamics and are able to find the important feature of each state.

However, the top importance values in AFO and FO do not correspond to the important time points.

Only in FFC, the top important observations are indicative of state changes.

Table 2 shows the performance compared to ground-truth explanations for this data.

As mentioned earlier, the quality of explanations rely on the quality of the counterfactuals.

The counterfactuals should reflect the underlying latent dynamics for an individual sample.

Counterfactuals under the marginal (as used by AFO) need not be likely for a specific sample.

The conditional distribution we use, on the other hand, models the underlying characteristic of an individual sample, while the marginal is an average over the population.

Counterfactuals unlikely under an individual patient's dynamics can result in inaccurate importance assignments since they can potentially overestimate the change in model outcome significantly.

We demonstrate this by evaluating the log probability of the counterfactual under the true generator distribution p * (x t |X 0:t−1 ).

Results are summarized in Table 2 , Column 3.

Since we simulate data using an HMM, we can analytically derive the distribution p * (x i,t |X 0:t−1 ).

Details of the derivation are included in Appendix A.2.1.

Following the procedure in Algorithm 1, we train a conditional generators for non-static time series features.

We compare results across all four existing methods by visualizing importance scores over time.

Figure 5 shows an example trajectory of a patient and the predicted outcome.

We plot the importance score over time for top 3 signals, selected by each method.

Shaded regions in bottom four rows indicate the most important observations, color representing the feature.

As shown in Figure  5 , counterfactual based methods mostly agree and pick the same signals as important features.

We further evaluate this by looking into accordance scores among methods, indicating the percentage of agreement.

This analysis is provided in the Appendix A.3, and the heat map in Figure 10 demonstrates the average score across test samples.

However, the methods don't agree on the exact time importance.

As we observe in Figure 5 and other patient trajectories, FFC assigns importance to observations at the precise times of signal change.

This is exactly as expected from the method.

The FFC counterfactual is conditioned on patient history and thus the counterfactual represents an observation assuming a patient had not change state.

Since evaluation of explanations can be subjective, we also use intervention information present in patient records to evaluate clinical applicability across baselines.

Clinicians intervene with a medication or procedure when there is a notable, adverse change in patient trajectory.

Therefore, picking up the most relevant features before an intervention onset is indicative of clinical validity of the method.

While we cannot directly associate an intervention with a specific cause (observation), we look at the overall distribution of signals considered important by each of the methods, prior to intervention onset.

Figure 6 shows these histograms for a number of interventions.

We see consistent

This experiment evaluates the utility of our method for attributing importance to GHG tracers across different locations in California.

The GHG data consists of 15 time series signals from 2921 grid cells.

A target time series is a synthetic signal of GHG concentrations.

We use an RNN model to estimate GHG concentrations using all tracers.

Evaluating which tracers are most useful in reconstructing this synthetic signal can be posed as a feature importance problem for weather modeling over time.

In order to quantitatively evaluate the proposed method on real data, we evaluate how well the method performs at selecting globally relevant methods as a proxy.

We aggregate the importance of all features over time (and training samples) and retrain the black-box by i) removing top 10 relevant features as indicated by each method ii) using top 3 relevant features only .

The performance summary is provided in Table 3 suggesting that among methods that derive instance wise feature importance over time, FFC also generates reasonable global relevance of features.

Results for both MIMIC-III and GHG datasets are summarized in

We additionally evaluate the quality of the proposed FFC method using the randomization tests proposed as 'Sanity Checks' in Adebayo et al. (2018) .

Two randomization tests are designed to test for sensitivity of the explanations to i) the black-box model parameters using a model parameter randomization test, and ii) sensitivity to data labels using a using a data randomization test.

We conduct this evaluation for Simulation Data II.

This test evaluates how different explanations are when the black-box model is trained on permuted labels (breaking the correlation between features and output label).

If explanations truly rely on the output labels, as suggested in our definition, then the explanation quality should differ significantly when a model trains on permuted labels.

We evaluate the drop in the AUROC and AUPRC of the generated explanations compared to the ground truth.

This test evaluates how different explanation quality is when the parameters of the model are arbitrarily shuffled.

Significant differences in generated explanations suggests the proposed method is sensitive to black-box model parameters.

In Adebayo et al. (2018) , these tests are conducted for saliency map methods for images by evaluating the distance between saliency maps for different randomizations.

The results are included for Simulated Data II, measured with AUROC and AUPRC as ground-truth explanations are available.

The results of both tests are included in Table 4 .

They indicate the drops in explanation performance for both randomization tests.

The performance of the model used for model randomization test drops to 0.52 AUROC as opposed to 0.95 for the original trained model on this simulated task (Simulation Data II).

For data randomization, performance of the model drops to 0.62 from 0.95 in terms of AUROC.

AUROCs and AUPRCs for FFC drop the most, suggesting the FFC explanation method is sensitive to perturbations in output labels (as tested by the data randomization test) and to randomization in model parameters.

Significant deterioration compared to explanation performance in Table 2 (for Simulation Data II) indicates that the proposed method passes the sanity checks.

We propose a new definition for obtaining sample-based feature importance for high-dimensional time series data.

We evaluate the importance of each feature at every time point, to locate highly important observations.

We define important observations as those that cause the biggest change in model output had they been different from the actual observation.

This counterfactual observation is generated by modeling the conditional distribution of the underlying data dynamics.

We propose a generative model to sample such counterfactuals.

We evaluate and compare the proposed definition and algorithm to several existing approaches.

We show that our method is better at localizing important observations over time.

This is one of the first methods that provides individual feature importance over time.

Future extension to this work will include analysis on real datasets annotated with feature importance explanations.

The method will also be extended to evaluate change in risk based on most relevant subsets of observations.

A.1 SIMULATED DATA I

To simulate these data, we generate d = 3 (independent) sequences as a standard non-linear autoregressive moving average (NARMA) time series.

Note also that we add linear trends to features 1 and 2 of the form: x(t + 1) = 0.5x(t) + 0.5x(t)

l−1 i=0 x(t − l) + 1.5u(t − (l − 1))u(t) + 0.5 + α d t for t ∈ [80], α > 0 (0.065 for feature 2 and 0.003 for feature 1), and where the order l = 2, u ∼ Normal(0, 0.03).

We additionally add linear trends to features.

We add spikes to each sample (uniformly at random over time) and for every feature d following the procedure below:

where κ > 0 indicates the additive spike.

The label y t = 1 ∀t > t 1 , where t 1 = min g d , i.e. the label changes to 1 when a spike is encountered in the first feature and is 0 otherwise.

We sample our time series using the python TimeSynth 1 package.

Number of samples generated: 10000 (80%,20% split).

The output y t at every step is assigned using the logit in 3.

Depending on the hidden state at time t, only one of the features contribute to the output and is deemed influential to the output.

The true conditional distribution can be derived using the forward algorithm (Bishop, 2006) as follows:

where,

where p(s t−1 |X 0:t−1 ) is estimated using the forward algorithm.

Our generator G i is trained using an RNN (GRU).

We model the latent state z t with a multivariate Gaussian with diagonal covariance and observations with a multivariate Gaussian with full covariance.

The counterfactual for observation i at time t can now be sampled by marginalizing over other features at time t. i.e, x i,t ∼ x−i p(x|X 0:t−1 ).

Feature selection and data processing: For this experiment, we select adult ICU admission data from the MIMIC dataset.

We use static patients' static, vital measurements and lab result for the analysis.

The task is to predict 48 hour mortality based on 48 hours of clinical data, therefor we remove samples with less than 48 hours of data.

Parameter Settings for conditional Generator: The recurrent network with specifications show in 8 learns a hidden latent vector h t representing the history.

h t is then concatenated with x −i,t and fed into a non-linear 1-layer MLP to model the conditional distribution p(x i , t|X 0:t−1 ).

Additional importance plots are provided in Figure 9 .

Adam (learning rate = 0.0001, β 1 = 0.9, β 2 = 0.999, weight decay = 0) Accordance testing:

For this test we look into how much different baselines agree on important feature assignment.

As we observed from the experiments, counterfactual methods mostly agree on the most important features for individual samples.

We define accordance score between 2 methods as the percentage of top n signals both identified as important.

A score of 80 means on average over the test data, 80 of the assignments were similar.

This is depicted in Figure 10 .

In this section we compare the run-time across multiple baselines.

Table 9 shows inference runtime for all the baseline methods on a machine with Quadro 400 GPU and Intel(R) Xeon(R) CPU E5-1620 v4 @ 3.50GHz CPU.

The runtime for the counterfactual approaches (FFC, FO and AFO) depends only on the length of the time series.

It is also the case for FFC since the conditional generator models the joint distribution of all features.

This is an advantage, over approaches like LIME, the runtime depends both on the length of the signal as well as the number of features.

Overall, FFC performs reasonably compared to ad-hoc counterfactual approaches, since inference on the RNN conditional generator is efficient.

This is one of the reasons that the RNN generator model is used to approximate the conditional distribution.

Table 9 : Run-time results for simulated data and MIMIC experiment.

Parameter Settings for Generator: The settings are provided in Table 10 .

Figure 11 shows the training loss of black-box that was used to present feature important results in Section 4.4.

<|TLDR|>

@highlight

Explaining Multivariate Time Series Models by finding important observations in time using Counterfactuals