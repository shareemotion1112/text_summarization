Electronic Health Records (EHR) comprise of longitudinal clinical observations portrayed with sparsity, irregularity, and high-dimensionality which become the major obstacles in drawing reliable downstream outcome.

Despite greatly numbers of imputation methods are being proposed to tackle these issues, most of the existing methods ignore correlated features or temporal dynamics and entirely put aside the uncertainty.

In particular, since the missing values estimates have the risk of being imprecise, it motivates us to pay attention to reliable and less certain information differently.

In this work, we propose a novel variational-recurrent imputation network (V-RIN), which unified imputation and prediction network, by taking into account the correlated features, temporal dynamics, and further utilizing the uncertainty to alleviate the risk of biased missing values estimates.

Specifically, we leverage the deep generative model to estimate the missing values based on the distribution among variables and a recurrent imputation network to exploit the temporal relations in conjunction with utilization of the uncertainty.

We validated the effectiveness of our proposed model with publicly available real-world EHR dataset, PhysioNet Challenge 2012, and compared the results with other state-of-the-art competing methods in the literature.

Electronic Health Records (EHR) store longitudinal data comprising of patient's clinical observations in the intensive care unit (ICU).

Despite the surge of interest in clinical research on EHR, it still holds diverse challenging issues to be tackled with, such as high-dimensionality, temporality, sparsity, irregularity, and bias (Cheng et al., 2016; Lipton et al., 2016; Yadav et al., 2018; Shukla & Marlin, 2019) .

Specifically, sequences of the medical events are recorded irregularly in terms of variables and time, due to various reasons such as lack of collection or documentation, or even recording fault (Wells et al., 2013; Cheng et al., 2016) .

In fact, since it carries essential information regarding the patient's health status, improper handling of missing values might draw an unintentional bias (Wells et al., 2013; Beaulieu-Jones et al., 2017) yielding unreliable downstream analysis and verdict.

Complete-case analysis is one approach to draw the clinical outcome by disregarding the missing values and relying only on the observed values.

However, excluding the missing data shows poor performance at high missing rates and also requires modeling separately for different dataset.

In fact, the missing values and their patterns are correlated with the target labels (Che et al., 2018) .

Thus, we resort to the imputation approach to improve clinical outcomes prediction as the downstream task.

There exist numerous proposed strategies in imputing missing values in the literature.

Brick & Kalton (1996) classified the imputation methods of being deterministic or stochastic in terms of the utilization of the randomness.

While deterministic methods such as mean (Little & Rubin, 1987) and median filling (Acu??a & Rodriguez, 2004) produced only one possible value, it is desirable to generate samples by considering the data distribution, thus leading to stochastic-based imputation methods.

Moreover, since we are dealing with multivariate time series, an adequate imputation model should reflect several properties altogether, namely, 1) temporal relations, 2) correlations across variables, and additionally 3) offering a probabilistic interpretation for uncertainty estimation (Fortuin et al., 2019) .

Recently, the rise of the deep learning models offers potential solutions in accommodating aforementioned conditions.

Variational autoencoders (VAEs) (Kingma & Welling, 2014) and generative adversarial networks (GANs) (Goodfellow et al., 2014) exploited the latent distribution of highdimensional incomplete data and generated comparable data points as the approximation estimates for the missing or corrupted values (Nazabal et al., 2018; Luo et al., 2018; Jun et al., 2019) .

However, even though these models employed the stochastic approach in inferring and generating samples, they scarcely utilized the uncertainty.

In addition, such deep generative models are insufficient in estimating the missing values of multivariate time series, due to their nature of ignoring temporal relations between a span of time points.

Hence, it requires additional approaches to model the temporal dynamics, such as Gaussian process (Fortuin et al., 2019) or recurrent neural network (RNNs) (Luo et al., 2018; Jun et al., 2019) .

On the other hand, by the virtue of RNNs which have proved a remarkable performance in modeling the sequential data, we can estimate the complete data by taking into account the temporal characteristics.

GRU-D (Che et al., 2018) proposed a modified gated-recurrent unit (GRU) cell to model missing patterns in the form of masking vector and temporal delay.

Likewise, BRITS (Cao et al., 2018) modeled the temporal relations by bi-directional dynamics, and also considered features correlation by regression layers in estimating the missing values.

However, they didn't take into account the uncertainty in estimating the missing values.

That is, since the imputation estimates are not thoroughly accurate, we may introduce their fidelity score denoted by the uncertainty, which enhances the task performance by emphasizing the reliable or less uncertain information and vice versa (He, 2010; Gemmeke et al., 2010; Jun et al., 2019) .

In this work, we define our primary task as prediction of in-hospital mortality on EHR data.

However, since the data are characterized by sparse and irregularly-sampled, we devise an effective imputation model as the secondary problem but major concern in this work.

We propose a novel variational-recurrent imputation network (V-RIN), which unified imputation and prediction network for multivariate time series EHR data, governing both correlations among variables and temporal relations.

Specifically, given the sparse data, an inference network of VAE is employed to capture data distribution in the latent space.

From this, we employ a generative network to obtain the reconstructed data as the imputation estimates for the missing values as well as the uncertainty indicating the imputation fidelity score.

Then, we integrate the temporal and feature correlations into a combined vector and feed it into a novel uncertainty-aware GRU in the recurrent imputation network.

Finally, we obtain the mortality prediction as a clinical verdict from the complete imputed data.

In general, our main contributions in this paper are as follows:

??? We estimate the missing values by utilizing deep generative model combined with recurrent imputation network to capture both features correlations and the temporal dynamics jointly, yielding the uncertainty.

??? We effectively incorporate the uncertainty with the imputation estimates in our novel uncertainty-aware GRU cell for better prediction result.

??? We evaluated the effectiveness of the proposed models by training the imputation and prediction networks jointly using the end-to-end manner, achieving the superior performance among other state-of-the-art competing methods on real-world multivariate time series EHR data.

Imputation strategies were extensively devised to resolve the issue of sparse high-dimensional time series data by means of the statistics, machine learning, and deep learning methods.

For instance, previous works exploited statistical attributes of observed data, such as mean (Little & Rubin, 1987) and median filling (Acu??a & Rodriguez, 2004) , which clearly ignored the temporal relations as well as the correlations among variables.

From the machine learning approaches, expectationmaximization (EM) algorithm (Dempster et al., 1977) , k-nearest neighbor (KNN) (Troyanskaya et al., 2001) , principal component analysis (PCA) (Oba et al., 2003; Mohamed et al., 2009) were proposed by taking into account the relationships of the features either in the original or latent space.

Furthermore, multiple imputation by chained equations (MICE) (White et al., 2011; Azur et al., 2011) introduced variability by means of repeating imputation process for multiple times.

Yet, these methods ignore the temporal relations as the crucial attributes in time series modeling.

The deep learning-based imputation models are closely related to our proposed models.

Nazabal et al. (2018) leveraged VAEs to generate stochastic imputation estimates by exploiting the distribution and correlations of features in the latent space.

However, it ignores the temporal relations and the uncertainties as well.

Recently, GP-VAE (Fortuin et al., 2019) were proposed to obtain the latent representation by means of VAEs and model the temporal dynamics in the latent space using Gaussian process.

However, since the model is merely focused on the imputation task, they required a separate model for further downstream outcome.

To deal with the time series data, a series of RNNs-based imputation models were proposed.

GRU-D (Che et al., 2018) took into account the temporal dynamics by incorporating the missing patterns, together with the mean imputation and forward filling with past values.

Similarly, GRU-I (Luo et al., 2018) trained the RNNs using adversarial scheme of GANs as the stochastic approach.

In the meantime, BRITS (Cao et al., 2018) were proposed to combine the feature correlations and temporal dynamics networks using bi-directional dynamics, which enhanced the accuracy by estimating missing values in both forward and backward directions.

Likewise, M-RNN (Yoon et al., 2017) utilized bi-directional recurrent dynamics by operating interpolation (intra-stream) and imputation (inter-stream).

Despite temporal dynamics are considered in their proposed models, yet the uncertainty for imputation was scarcely incorporated.

Our proposed model differs from the above models in ways of integrating imputation and prediction networks jointly.

In particular, for the imputation network, we model both feature and temporal relations by means of deep generative model and recurrent imputation networks, respectively.

Furthermore, we introduce the imputation fidelity of estimates as the uncertainty, compensating the potential impairment of imputation estimates.

Specifically, it is noteworthy that our proposed model provides the reliable estimates, while giving the penalty to the unreliable ones determined by its uncertainties.

Thereby, we expect to get better estimates of the missing values leading to a better prediction performance as a downstream task.

Our architecture consists of two key networks: imputation and prediction network, as depicted in Fig. 1 .

The imputation network is devised on VAEs to capture the latent distribution of the sparse data by means of inference network (encoder E).

Then, the generative network (decoder D) estimates reconstructed data distribution.

We regard its mean as the imputation estimates, while exploiting the variance as the uncertainty to be further utilized in the recurrent imputation network for reliable prediction outcome.

Moreover, the succeeding recurrent imputation networks is built upon RNNs to model the temporal dynamics.

In addition, for each time step, we use the regression layer to explore the feature correlation in imputing the missing values.

Eventually, by unifying VAEs and RNNs systematically, we expect to acquire a more likely estimate by taking into account the features correlations, temporal relations over time as well as utilization of the uncertainty.

We describe each of the networks more specifically in the following section after introducing the notations.

Given the multivariate time series EHR data of N number of patients, a set of clinical observations and their corresponding label is denoted as

.

For each patient, we denote

T ] ??? R T ??D , where T and D represent time points and variables, respectively, x (n) t ??? R D denotes all observed variables at t-th time point, and x (n),d t is the d-th element of variables at time t. In addition, it has corresponding clinical label y (n) ??? {0, 1} representing the in-hospital mortality which is a binary classification problem in our scenario.

For the sake of clarity, hereafter we omit the superscript (n).

To address the missing values, we introduce the masking matrix M ??? {0, 1}

T ??D indicating whether the values are observed or missing, and additionally define a new data representatio??

T ??D , where we initialize the missing value with zero as follows:

Besides, the time gap between the two observed values carries a piece of essential information.

Thus, we further introduce time delay matrix ??? ??? R T ??D , which is derived from s ??? R T , denoting the timestamp of the measurement.

For the t = 1, we set ??? t = 1.

While for the rest (t > 1), we set the time delay by referring to the masking matrix as follows:

otherwise.

Given the observations at each time pointx t , we infer z ??? R k D as the latent representation by making use of the inference network, utilizing the true posterior distribution p ?? (z|x t ).

Intuitively, we assume thatx t is generated from some unobserved random variable z by some conditional distribution p ?? (x t |z), while z is generated from a prior distribution p ?? (z).

Therefore, we define the marginal likelihood as p ?? (x t ) = p ?? (x t |z)p ?? (z)dz.

However, since it is intractable due to involvement of the true posterior p ?? (z|x t ), we approximate it with q ?? (z|x t ) using a Gaussian distribution N (?? z , ?? 2 z ), where the mean and log-variance are obtained such that:

, where E {??,??} denotes the inference network with parameter ??.

Furthermore, we apply the reparameterization trick proposed by Kingma & Welling (2014) as z = ?? z + ?? z z , where z ??? N (0, I), and denotes the element-wise multiplication, thus, making it possible to be differentiated and trained using standard gradient methods.

Furthermore, given this latent vector z, we estimate p ?? (x t |z) by means of the generative network D with parameter ?? as :

where ?? x and ?? 2 x denote the mean and variance of reconstructed data distribution, respectively.

We apply another reparameterization trick in the data space to obtainx t = ?? x + ?? x x with x ??? N (0, I).

We regard this reconstructed data as the estimates to the missing values and maintain the observed values inx t as follows:

In the meantime, we regard the variance of reconstructed data as the uncertainty to be further utilized in the recurrent imputation process.

For this purpose, we introduce an uncertainty matrix

We quantify this uncertainty as the fidelity score of the missing values etimates.

In particular, we set the corresponding uncertainty as zero if the data is observed, indicating that we are confident with full trust to the observation, while set this as a value ?? d x,t if the corresponding value is missing as:

As a result of VAE-based imputation network, we obtain the set {X,??} denoting the imputed values and its corresponding uncertainty, respectively.

Figure 2: Graphical illustrations of (a) vanilla GRU cell and (b) our modified GRU cell incorporating the uncertainty (F u : update gate, F r : reset gate).

The recurrent imputation network is based on RNNs, where we further model the temporal relations in the imputed data and exploit the uncertainties.

While both GRU (Cho et al., 2014) or long-short term memory (LSTM) (Hochreiter & Schmidhuber, 1997) are feasible choices to be employed, inspired from the previous work of Che et al. (2018), we leverage the uncertainty-aware GRU cell to further consider uncertainty and the temporal decaying factor, which is depicted in Fig. 2b .

Specifically, at each time step t, we produce the uncertainty decay factor ?? t in the Eq. (1) using negative exponentional rectifier to guarantee ?? t ??? (0, 1], and further element-wise multiply this withx t to emphasizes the reliable estimates and give penalties to the uncertain ones expressed by the Eq. (2) resulting in

By employing the GRU, we obtain the hidden state h as the comprehensive information compiled from the preceding sequences.

Thus, given the previous hidden states h t???1 , we produce the current complete observation estimates x r t through the following regression equation:

Hence, we have a pair of imputed values {x ?? t , x r t } corresponding to missing values estimates based on features correlations and temporal relations, respectively.

We then merge these information jointly to get combined vector c t comprising both estimates:

where ??? denotes a concatenation operation.

Finally, we obtain the complete vector x c t by replacing the missing values with the combined estimates as follows:

As time delay ??? t is essential element to capture temporal relations from the data (Che et al., 2018) , we also introduce the temporal decay factor ?? t ??? (0, 1] in the Eq. (6) as

We utilize this factor to control the influence of past observations embedded into hidden states using the form of (h t???1 ?? t ).

In addition to this, we concatenate the complete vector with corresponding mask, and then feed it into the uncertainty-aware GRU cell as illustrated in the Fig. 2b expressed as:

Lastly, to predict the in-hospital mortality as the clinical outcome, we utilize the last hidden state h T to get the predicted label?? such that:

Note that W {??,r,c,??,h,y} , V h and b {??,r,c,??,h,y} } are our learnable parameters in recurrent imputation network.

We specify the composite loss function comprising of the imputation and prediction loss function to tune all model parameters jointly, which are ?? = {??, ??, W {??,r,c,??,h,y} , V h , b {??,r,c,??,h,y} }.

By means of VAEs, we define the loss function L vae to maximize the variational evidence lower bound (ELBO) that comprises of the reconstruction loss term and the Kullback-Leibler divergence.

We add 1 -regularization to introduce the sparsity into the network with ?? 1 as the hyperparameter.

Moreover, we measure the difference between the observed data and the combined imputation estimates by the mean absolute error (MAE) as the L reg .

Furthermore, we define the binary cross-entropy loss function L pred to evaluate the prediction of in-hospital mortality as follows:

Finally, we define the overall loss function L total as:

with ?? and ?? are the hyperparameters to represent the ratio between the L vae and L reg , respectively, and ?? 2 is the weight decay hyperparameter.

Lastly, we use stochastic gradient decent in an end-toend manner to optimize the model parameters during the training.

We evaluated our proposed model on publicly available real-world EHR dataset, PhysioNet 2012 Challenge (Goldberger et al., 2000; Silva et al., 2012) , which consists of 35 irregularly-sampled clinical variables (i.e. heart and respiration rate, blood pressure, etc.) from nearly 4,000 patients during first 48 hours of medical care in the ICU.

We excluded 3 patients with no observation at all.

We further sampled the observations hourly and take the last values in case of multiple measurements within this period, resulting data with mean missing rates of 80.51% and maximum of 99.84%.

We predicted in-hospital mortality which are imbalanced with 554 positive mortality label (13.86%).

For the inference network of VAEs, we employed three layers of feedforward networks with hidden units of {64,24,10}, where 10 denotes the dimension of latent representation.

The generative network has equal number of hidden units with those of inference network, but in the reverse order.

We utilized Rectified Linear Unit (ReLU) as the non-linear activation function for each hidden layer.

As for the recurrent imputation network, we used modified GRU with 64 hidden units.

We trained the model using Adam optimizer with 200 epochs, 64 mini-batches and a learning rate of 0.0001.

We set ?? 1 and ?? 2 equally with 0.0001.

Finally, we reported the test result on in-hospital mortality prediction task from the 5-fold cross validation in terms of the average of Area Under the ROC curve (AUC) and Area Under Precision-Recall Curve (AUPRC).

Additionally, to assess the effectiveness of our model in handling the imbalanced issue, we presented the balanced accuracy as well.

We compared the performance of our proposed models with the closely-related state-of-the-art competing models in the literature:

??? M-RNN (Yoon et al., 2017 ) exploited multi-directional RNNs which executing both interpretation and imputation to infer the missing data. (Chung et al., 2015) (Jun et al., 2019) (Ours) ??? GRU-I (Luo et al., 2018) made use of adversarial scheme based on RNNs to consider both feature correlation and temporal dynamics altogether with its temporal decay as well.

??? BRITS (Cao et al., 2018) utilized bi-directional dynamics in estimating the missing values based on features correlations and temporal relations.

There are several variants of this model: BRITS-I utilized bi-directional dynamics considering only the temporal relations; RITS utilized unidirectional dynamics but use both feature and temporal correlations; while RITS-I utilized the unidirectional dynamics relying solely on temporal relations.

??? V-RIN (Ours) is based on our proposed model except that we ignored the uncertainty.

Specifically, we excluded the Eq. (1-2), and replaced x ?? t withx t in Eq. (4).

??? V-RIN-full (Ours) executed all operations in the proposed model including feature-based correlations, temporal relations and the uncertainty to further mitigate the estimates bias.

As part of the ablation studies, first we investigated the effect of varying the pair {??, ??} as the hyperparameters of the imputation on both V-RIN and V-RIN-full model on the in-hospital mortality prediction task.

We reflected these parameters as the ratio to weigh the imputation between feature and temporal relations in estimating the missing values in order to achieve the optimal performance.

For each parameter, we defined a set of range values for these hyperparameters as [0.01, 1.0] and presented the corresponding performances in the Fig 3.

Both models were able to achieve high performance in terms of the average AUC score of 0.8281 for V-RIN and 0.8335 for V-RIN-full.

V-RIN achieved its peak with setting {?? = 0.25, ?? = 0.75}, while V-RIN-full with {?? = 0.01, ?? = 0.5}. We interpreted these findings as by emphasizing more on the temporal relations than the features correlations in imputing the missing values, the model are able to obtain its best performance.

However, once we tried to increase the ??, we observed that the performance were degraded to some degree.

Hence, it proved that both features and temporal are essential in estimating the missing values with some latent proportion. (Che et al., 2018) 0.8094 ?? 0.0142 0.4571 ?? 0.0248 57.9539 ?? 3.2072 GRU-I (Luo et al., 2018) 0.7831 ?? 0.0205 0.4029 ?? 0.0471 58.3328 ?? 0.6374 RITS-I (Cao et al., 2018) 0.8103 ?? 0.0137 0.4511 ?? 0.0319 61.5995 ?? 1.3904 RITS (Cao et al., 2018) 0.8110 ?? 0.0129 0.4558 ?? 0.0284 60.3869 ?? 1.2256 BRITS-I (Cao et al., 2018) 0.8184 ?? 0.0116 0.4510 ?? 0.0351 58.3711 ?? 3.4079 BRITS (Cao et al., 2018) 0.8238 ?? 0.0100 0.4782 ?? 0.0340 59.4221 ?? 2.3565 V-RIN (Ours) 0.8281 ?? 0.0075 0.4767 ?? 0.0257 62.0363 ?? 2.1644 V-RIN-full (Ours) 0.8335 ?? 0.0106 0.4907 ?? 0.0365 63.1127 ?? 3.0619

Furthermore, Table 1 presented the comparison of our model with closely related models such as VRNN (Chung et al., 2015) which integrates VAEs for each time steps of RNNs.

However, we observed that the performance is considered as low in terms of AUC and AUPRC as well.

In addition, we compared also with VAEs followed by RNNs (VAE+RNN) without incorporating neither the temporal decay factor nor the uncertainty (Jun et al., 2019) .

We noticed a performance improvement in VAE+RNN which executes the imputation process by firstly exploiting the features correlations followed by temporal dynamics in exact order.

Furthermore, by introducing the temporal decay in V-RIN, it helped a lot for the model to learn the temporal dynamics effectively resulting better AUC and AUPRC.

Finally, once we introduced the uncertainty which is incorporated in the imputation network of V-RIN-full, we observed a significant enhancement of both AUC and AUPRC.

This is undeniable evidence of the advantage of utilizing the uncertainty in further downstream task.

We presented the experimental result of the in-hospital mortality prediction in comparison with other competing methods in terms of average AUC, AUPRC and balanced accuracy in Table 2 .

In practice, we removed 10% of the observed data randomly to make a fair comparison with BRITS model variants.

Our V-RIN model is directly comparable to other competing models in ways that it estimates the missing values without incorporating the uncertainty.

It achieved better performance in terms of AUC and balanced accuracy, and slightly comparable to BRITS in terms of AUPRC.

However, we note that BRITS-I and BRITS utilized bi-directional dynamics to estimate the missing values achieving relatively higher performance.

Although M-RNN employed similar strategies using bidirectional dynamics, it struggles to perform the task properly.

Furthermore, our V-RIN is closely related to GRU-D, GRU-I, RITS-I, and RITS in ways of exploiting the temporal decay factor.

However, V-RIN outperformed all aforementioned models indicating the effectiveness of missing values estimation using deep generative models by means of VAEs.

Meanwhile, GRU-I which also makes use of deep generative model using adversarial strategies performed inferior compared to our model.

This might be due to the fact that they employed imputation and prediction model separately.

Ultimately, we obtained the highest overall performance results with the proposed V-RIN-full including the average balanced accuracy, indicating the effectiveness of the model in handling the imbalance issue which is non-trivial.

Thereby, these findings reassure that the utilization of the uncertainty is truly beneficial in estimating the missing values.

Hence, our model were able to achieve reliable downstream task and outperformed all comparative models.

In this paper, we proposed a novel unified framework comprising of imputation and prediction network for sparse high-dimensional multivariate time series.

It combined deep generative model with recurrent model to capture features correlations and temporal relations in estimating the missing values and yielding uncertainty.

We utilized the uncertainties as the fidelity of our estimation and incorporated them for clinical outcome prediction.

We evaluated the effectiveness of proposed model with PhysioNet 2012 Challenge dataset as the real-world EHR multivariate time series data, proving the superiority of our model in the in-mortality prediction task, compared to other state-of-the-art comparative models in the literature.

@highlight

Our variational-recurrent imputation network (V-RIN) takes into account the correlated features, temporal dynamics, and further utilizes the uncertainty to alleviate the risk of biased missing values estimates.

@highlight

A missing data imputation network to incorporate correlation, temporal relationships, and data uncertainty for the problem of data sparsity in EHRs, which yields higher AUC on mortality rate classification tasks.

@highlight

The paper presented a method that combines VAE and uncertainty aware GRU for sequential missing data imputation and outcome prediction.