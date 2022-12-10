In this paper, we present a new deep learning architecture for addressing the problem of supervised learning with sparse and irregularly sampled multivariate time series.

The architecture is based on the use of a semi-parametric interpolation network followed by the application of a prediction network.

The interpolation network allows for information to be shared across multiple dimensions of a multivariate time series during the interpolation stage, while any standard deep learning model can be used for the prediction network.

This work is motivated by the analysis of physiological time series data in electronic health records, which are sparse, irregularly sampled, and multivariate.

We investigate the performance of this architecture on both classification and regression tasks, showing that our approach outperforms a range of baseline and recently proposed models.

Over the last several years, there has been significant progress in developing specialized models and architectures that can accommodate sparse and irregularly sampled time series as input BID25 BID20 BID21 BID12 BID4 ).

An irregularly sampled time series is a sequence of samples with irregular intervals between their observation times.

Irregularly sampled data are considered to be sparse when the intervals between successive observations are often large.

Of particular interest in the supervised learning setting are methods that perform end-to-end learning directly using multivariate sparse and irregularly sampled time series as input without the need for a separate interpolation or imputation step.

In this work, we present a new model architecture for supervised learning with multivariate sparse and irregularly sampled data: Interpolation-Prediction Networks.

The architecture is based on the use of several semi-parametric interpolation layers organized into an interpolation network, followed by the application of a prediction network that can leverage any standard deep learning model.

In this work, we use GRU networks BID6 as the prediction network.

The interpolation network allows for information contained in each input time series to contribute to the interpolation of all other time series in the model.

The parameters of the interpolation and prediction networks are learned end-to-end via a composite objective function consisting of supervised and unsupervised components.

The interpolation network serves the same purpose as the multivariate Gaussian process used in the work of BID12 , but remove the restrictions associated with the need for a positive definite covariance matrix.

Our approach also allows us to compute an explicit multi-timescale representation of the input time series, which we use to isolate information about transients (short duration events) from broader trends.

Similar to the work of BID21 and BID4 , our architecture also explicitly leverages a separate information channel related to patterns of observation times.

However, our representation uses a semi-parametric intensity function representation of this information that is more closely related to the work of Lasko (2014) on modeling medical event point processes.

Our architecture thus produces three output time series for each input time series: a smooth interpolation modeling broad trends in the input, a short time-scale interpolation modeling transients, and an intensity function modeling local observation frequencies.

This work is motivated by problems in the analysis of electronic health records (EHRs) BID25 BID21 BID12 BID4 ).

It remains rare for hospital systems to capture dense physiological data streams.

Instead, it is common for the physiological time series data in electronic health records to be both sparse and irregularly sampled.

The additional issue of the lack of alignment in the observation times across physiological variables is also very common.

We evaluate the proposed architecture on two datasets for both classification and regression tasks.

Our approach outperforms a variety of simple baseline models as well as the basic and advanced GRU models introduced by BID4 across several metrics.

We also compare our model with to the Gaussian process adapter BID19 and multi-task Gaussian process RNN classifier BID12 .

Further, we perform full ablation testing of the information channels our architecture can produce to assess their impact on classification and regression performance.

The problem of interest in this work is learning supervised machine learning models from sparse and irregularly sampled multivariate time series.

As described in the introduction, a sparse and irregularly sampled time series is a sequence of samples with large and irregular intervals between their observation times.

Such data commonly occur in electronic health records, where they can represent a significant problem for both supervised and unsupervised learning methods (Yadav et al., 2018) .

Sparse and irregularly sampled time series data also occur in a range of other areas with similarly complex observation processes including climate science (Schulz & Stattegger, 1997) , ecology BID7 , biology BID27 , and astronomy (Scargle, 1982) .A closely related (but distinct) problem is performing supervised learning in the presence of missing data BID0 .

The primary difference is that the missing data problem is generally defined with respect to a fixed-dimensional feature space BID22 .

In the irregularly sampled time series problem, observations typically occur in continuous time and there may be no notion of a "normal" or "expected" sampling frequency for some domains.

Methods for dealing with missing data in supervised learning include the pre-application of imputation methods (Sterne et al., 2009) , and learning joint models of features and labels (Williams et al., 2005) .

Joint models can either be learned generatively to optimize the joint likelihood of features and labels, or discriminately to optimize the conditional likelihood of the labels.

The problem of irregular sampling can be converted to a missing data problem by discretizing the time axis into non-overlapping intervals.

Intervals with no observations are then said to contain missing values.

This is the approach taken to deal with irregular sampling by BID25 as well as BID21 .

This approach forces a choice of discretization interval length.

When the intervals are long, there will be less missing data, but there can also be multiple observations in the same interval, which must be accounted for using ad-hoc methods.

When the intervals are shorter, most intervals will contain at most one value, but many intervals may be empty.

Learning is generally harder as the amount of missing data increases, so choosing a discretization interval length must be dealt with as a hyper-parameter of such a method.

One important feature of missing data problems is the potential for the sequence of observation times to itself be informative BID22 .

Since the set of missing data indicators is always observed, this information is typically easy to condition on.

This technique has been used successfully to improve models in the domain of recommender systems (Salakhutdinov et al., 2007) .

It was also used by BID21 to improve performance of their GRU model.

The alternative to pre-discretizing irregularly sampled time series to convert the problem of irregular sampling into the problem of missing data is to construct models with the ability to directly use an irregularly sampled time series as input.

The machine learning and statistics literature include several models with this ability.

In the probabilistic setting, Gaussian process models have the ability to represent continuous time data via the use of mean and covariance functions BID26 .

These models have non-probabilistic analogues that are similarly defined in terms of kernels.

For example, BID24 present a kernel-based method that can be used to produce a similarity function between two irregularly sampled time series.

BID20 subsequently provided a generalization of this approach to the case of kernels between Gaussian process models.

BID19 showed how the re-parameterization trick BID17 could be used to extend these ideas to enable end-to-end training of a deep neural network model (feed-forward, convolutional, or recurrent) stacked on top of a Gaussian process layer.

While the basic model of BID19 was only applied to univariate time series, in follow-up work the model was extended to multivariate time series using a multi-output Gaussian process regression model BID12 .

However, modeling multivariate time series within this framework is quite challenging due to the constraints on the covariance function used in the GP layer.

BID12 deal with this problem using a sum of separable kernel functions BID2 , which limit the expressiveness of the model.

An important property of the above models is that they allow for incorporating all of the information from all available time points into a global interpolation model.

Variants differ in terms of whether they only leverage the posterior mean when the final supervised problem is solved, or whether the whole posterior is used.

A separate line of work has looked at the use of more local interpolation methods while still operating directly over continuous time inputs.

For example, BID4 presented several methods based on gated recurrent unit (GRU) networks BID6 combined with simple imputation methods including mean imputation and forward filling with past values.

BID4 additionally considered an approach that takes as input a sequence consisting of both the observed values and the timestamps at which those values were observed.

The previously observed input value is decayed over time toward the overall mean.

In another variant the hidden states are similarly decayed toward zero.

Yoon et al. (2017) presented another similar approach based on multi-directional RNN which operate across streams in addition to within streams.

However, these models are limited to using global information about the structure of the time series via its empirical mean value, and current or past information about observed values.

The global structure of the time series is not directly taken into account.

BID5 focus on a similar problem of modeling multi-rate multivariate time series data.

This is similar to the problem of interest in that the observations across time series can be unaligned.

The difference is that the observations in each time series are uniformly spaced, which is a simpler case.

In the case of missing data, they use forward or linear interpolation, which again does not capture the global structure of time series.

Similarly, BID1 presented an autoregressive framework for regression tasks with irregularly sampled time series data.

It is not clear how it can be extended for classification.

The model proposed in this work is similar to that of BID19 and BID12 in the sense that it consists of global interpolation layers.

The primary difference is that these prior approaches used Gaussian process representations within the interpolation layers.

The resulting computations can be expensive and, as noted, the design of covariance functions in the multivariate case can be challenging.

By contrast, our proposed model uses semi-parametric, deterministic, feed-forward interpolation layers.

These layers do not encode uncertainty, but they do allow for very flexible interpolation both within and across layers.

Also similar to BID19 and BID12 , the interpolation layers in our architecture produce regularly sampled interpolants that can serve as inputs for arbitrary, unmodified, deep classification and regression networks.

This is in contrast to the approach of BID4 , where a recurrent network architecture was directly modified, reducing the modularity of the approach.

Finally, similar to BID21 , our model includes information about the times at which observations occur.

However, instead of pre-discretizing the inputs and viewing this information in terms of a binary observation mask or set of missing data indicators, we directly model the sequence of observation events as a point process in continuous time using a semi-parametric intensity function BID18 .

In this section, we present the proposed modeling framework.

We begin by presenting notation, followed by the model architecture and learning criteria.

We let D = {(s n , y n )|n = 1, ..., N } represent a data set containing N data cases.

An individual data case consists of a single target value y n (discrete for classification and real-valued in the case of regression), as well as a D-dimensional, sparse and irregularly sampled multivariate time series s n .

Different dimensions d of the multivariate time series can have observations at different times, as well as different total numbers of observations L dn .

Thus, we represent time series d for data case n as a tuple DISPLAYFORM0 is the list of time points at which observations are defined and DISPLAYFORM1 is the corresponding list of observed values.

The overall model architecture consists of two main components: an interpolation network and a prediction network.

The interpolation network interpolates the multivariate, sparse, and irregularly sampled input time series against a set of reference time points r = [r 1 , ..., r T ].

We assume that all of the time series are defined within a common time interval (for example, the first 24 or 48 hours after admission for MIMIC-III dataset).

The T reference time points r t are chosen to be evenly spaced within that interval.

In this work, we propose a two-layer interpolation network with each layer performing a different type of interpolation.

The second component, the prediction network, takes the output of the interpolation network as its input and produces a predictionŷ n for the target variable.

The prediction network can consist of any standard supervised neural network architecture (fully-connected feedforward, convolutional, recurrent, etc).

Thus, the architecture is fully modular with respect to the use of different prediction networks.

In order to train the interpolation network, the architecture also includes an auto-encoding component to provide an unsupervised learning signal in addition to the supervised learning signal from the prediction network.

FIG0 shows the architecture of the proposed model.

We describe the components of the model in detail below.

We begin by describing the interpolation network.

The goal of the interpolation network is to provide a collection of interpolants of each of the D dimensions of an input multivariate time series defined at the T reference time points r = [r 1 , ..., r T ].

In this work, we use a total of C = 3 outputs for each of the D input time series.

The three outputs (discussed in detail below) capture smooth trends, transients, and observation intensity information.

We define f θ (r, s n ) to be the function computing the outputŝ n of the interpolation network.

The outputŝ n is a fixed-sized array with dimensions (DC) × T for all inputs s n .

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 The second .

DISPLAYFORM3 In the experiments presented in the next section, we use a total of three interpolation network outputs per dimension d as the input to the prediction network.

We use the smooth, cross-channel interpolants χ d to capture smooth trends, the transient components τ d to capture transients, and the intensity functions λ d to capture information about where observations occur in time.

Following the application of the interpolation network, all D dimensions of the input multivariate time series have been re-represented in terms of C outputs defined on the regularly spaced set of reference time points r 1 , ..., r T (in our experiments, we use C = 3 as described above).

Again, we refer to the complete set of interpolation network outputs asŝ n = f θ (r, s n ), which can be represented as a matrix of size (DC) × T .The prediction network must takeŝ n as input and output a predictionŷ n = g φ (ŝ n ) = g φ (f θ (r, s n )) of the target value y n for data case n.

There are many possible choices for this component of the model.

For example, the matrixŝ n can be converted into a single long vector and provided as input to a standard multi-layer feedforward network.

A temporal convolutional model or a recurrent model like a GRU or LSTM can instead be applied to time slices of the matrixŝ n .

In this work, we conduct experiments leveraging a GRU network as the prediction network.

To learn the model parameters, we use a composite objective function consisting of a supervised component and an unsupervised component.

This is due to the fact that the supervised component alone is insufficient to learn reasonable parameters for the interpolation network parameters given the amount of available training data.

The unsupervised component used corresponds to an autoencoder-like loss function.

However, the semi-parametric RBF interpolation layers have the ability to exactly fit the input points by setting the RBF kernel parameters to very large values.

To avoid this solution and force the interpolation layers to learn to properly interpolate the input data, it is necessary to hold out some observed data points x jdn during learning and then to compute the reconstruction loss only for these data points.

This is a well-known problem with high-capacity autoencoders, and past work has used similar strategies to avoid the problem of trivially memorizing the input data without learning useful structure.

To implement the autoencoder component of the loss, we introduce a set of masking variables m jdn for each data point (t jdn , x jdn ).

If m jdn = 1, then we remove the data point (t jdn , x jdn ) as an input to the interpolation network, and include the predicted value of this time point when assessing the autoencoder loss.

We use the shorthand notation m n s n to represent the subset of values of s n that are masked out, and (1 − m n ) s n to represent the subset of values of s n that are not masked out.

The valuex jdn that we predict for a masked input at time point t jdn is the value of the smooth cross-channel interpolant at that time point, calculated based on the un-masked input values: DISPLAYFORM0 We can now define the learning objective for the proposed framework.

We let P be the loss for the prediction network (we use cross-entropy loss for classification and squared error for regression).

We let I be the interpolation network autoencoder loss (we use standard squared error).

We also include 2 regularizers for both the interpolation and prediction networks parameters.

δ I , δ P , and δ R are hyper-parameters that control the trade-off between the components of the objective function.

DISPLAYFORM1

In this section, we present experiments based on both classification and regression tasks with sparse and irregularly sampled multivariate time series.

In both cases, the input to the prediction network is a sparse and irregularly sampled time series, and the output is a single scalar representing either the predicted class or the regression target variable.

We test the model framework on two publicly available real-world datasets: MIMIC-III 3 − a multivariate time series dataset consisting of sparse and irregularly sampled physiological signals collected at Beth Israel Deaconess Medical Center from 2001 to 2012 BID16 , and UWaveGesture 4 − a univariate time series data set consisting of simple gesture patterns divided into eight categories BID23 ).

Details of each dataset can be found in the Appendix A.1.

We use the MIMIC-III mortality and length of stay prediction tasks as example classification and regression tasks with multivariate time series.

We use the UWave gesture classification task for assessing training time and performance relative to univariate baseline models.

We compare our proposed model to a number of baseline approaches including off-the-shelf classification and regression models learned using basic features, as well as more recent approaches based on customized neural network models.

For non-neural network baselines, we evaluate Logistic Regression (Hosmer Jr et al., 2013), Support Vector Machines (SVM) BID8 , Random Forests (RF) BID3 ) and AdaBoost BID11 for the classification task.

For the length of stay prediction task, we apply Linear Regression BID14 , Support Vector Regression (SVR), AdaBoost Regression BID10 and Random Forest Regression.

Standard instances of all of these models require fixed-size feature representations.

We use temporal discretization with forward filling to create fixed-size representation in case of missing data and use this representation as feature set for non-neural network baselines.

We compare to several existing deep learning baselines built on GRUs using simple interpolation or imputation approaches.

In addition, we compare to current state-of-the-art models for mortality prediction including the work of BID4 .

Their work proposed to handle irregularly sampled and missing data using recurrent neural networks (RNNs) by introducing temporal decays in the input and/or hidden layers.

We also evaluate the scalable end-to-end Gaussian process adapter BID19 as well as multi-task Gaussian process RNN classifier BID12 for irregularly sampled univariate and multivariate time series classification respectively.

This work is discussed in detail in Section 2.

The complete set of models that we compare to is as follows:• GP-GRU: End-to-end Gaussian process with GRU as classifier.• GRU-M: Missing observations replaced with the global mean of the variable across the training examples.• GRU-F: Missing values set to last observed measurement within that time series (referred to as forward filling).• GRU-S: Missing values replaced with the global mean.

Input is concatenated with masking variable and time interval indicating how long the particular variable is missing.• GRU-D: In order to capture richer information, decay is introduced in the input as well as hidden layer of a GRU.

Instead of replacing missing values with the last measurement, missing values are decayed over time towards the empirical mean.• GRU-HD: A variation of GRU-D where decay in only introduced in the hidden layer.

In this section, we present the results of the classification and regression experiments, as well as the results of ablation testing of the internal structure of the interpolation network for the proposed model.

We use the UWaveGesture dataset to assess the training time and classification performance relative to the baseline models.

We use the standard train and test sets (details are given in appendix A.1).

We report the training time taken for convergence along with accuracy on test set.

For MIMIC-III, we create our own dataset (appendix A.1) and report the results of a 5-fold cross validation experiment in terms of the average area under the ROC curve (AUC score), average area under the precision-recall curve (AUPRC score), and average cross-entropy loss for the classification task.

For the regression task, we use average median absolute error and average fraction of explained variation (EV) as metrics.

We also report the standard deviation over cross validation folds for all metrics.

Training and implementation details can be found in appendix A.2.

FIG2 shows the classification performance on the UWaveGesture dataset.

The proposed model and the Gaussian process adapter BID19 significantly outperform the rest of the baselines.

However, the proposed model achieves similar performance to the Gaussian process adapter, but with a 50x speed up (note the log scale on the training time axis).

On the other hand, the training time of the proposed model is approximately the same order as other GRU-based models, but it achieves much better accuracy.

TAB1 compares the predictive performance of the mortality and length of stay prediction task on MIMIC-III.

We note that in highly skewed datasets as is the case of MIMIC-III, AUPRC BID9 can give better insights about the classification performance as compared to AUC score.

The proposed model consistently achieves the best average score over all the metrics.

We note that a paired t-test indicates that the proposed model results in statistically significant improvements over all baseline models (p < 0.01) with respect to all the metrics except median absolute error.

The version of the proposed model used in this experiment includes all three interpolation network outputs (smooth interpolation, transients, and intensity function).An ablation study shows that the results on the regression task can be further improved by using only two outputs (transients, and intensity function), achieving statistically significant improvements over 2 0 2 2 2 4 2 6 2 8 2 10 2 12 2 14 2 16 Training time in seconds (log scale) all the baselines.

Results for the ablation study are given in Appendix A.3.

Finally, we compare the proposed model with multiple baselines on a previous MIMIC-III benchmark dataset BID13 , which uses a reduced number of cohorts as compared to the one used in our experiments.

Appendix A.4 shows the results on this benchmark dataset, where our proposed approach again outperforms prior approaches.

In this paper, we have presented a new framework for dealing with the problem of supervised learning in the presence of sparse and irregularly sampled time series.

The proposed framework is fully modular.

It uses an interpolation network to accommodate the complexity that results from using sparse and irregularly sampled data as supervised learning inputs, followed by the application of a prediction network that operates over the regularly spaced and fully observed, multi-channel output provided by the interpolation network.

The proposed approach also addresses some difficulties with prior approaches including the complexity of the Gaussian process interpolation layers used in BID19 BID12 , and the lack of modularity in the approach of BID4 .

Our framework also introduces novel elements including the use of semi-parametric, feed-forward interpolation layers, and the decomposition of an irregularly sampled input time series into multi-ple distinct information channels.

Our results show statistically significant improvements for both classification and regression tasks over a range of baseline and state-of-the-art methods.

Ruslan This data set contains sparse and irregularly sampled physiological signals, medications, diagnostic codes, inhospital mortality, length of stay and more.

We focus on predicting in-hospital mortality and length of stay using the first 48 hours of data.

We extracted 12 standard physiological variables from each of the 53,211 records obtained after removing hospital admission records with length of stay less than 48 hours.

TAB2 shows the features, sampling rates (per hour) and their missingness information computed using the union of all time stamps that exist in any dimension of the input time series.

In our experiments, each admission record corresponds to one data case (s n , y n ).

Each data case n consists of a sparse and irregularly sampled time series s n with D = 12 dimensions.

Each dimension d of s n corresponds to one of the 12 vital sign time series mentioned above.

In the case of classification, y n is a binary indicator where y n = 1 indicates that the patient died at any point within the hospital stay following the first 48 hours and y n = 0 indicates that the patient was discharged at any point after the first 48 hours.

There are 4310 (8.1%) patients with a y n = 1 mortality label.

The complete data set is D = {(s n , y n )|n = 1, ..., N }, and there are N = 53, 211 data cases.

The goal in the classification task is to learn a classification function g of the form y n ← g(s n ) whereŷ n is a discrete value.

In the case of regression, y n is a real-valued regression target corresponding to the length of stay.

Since the data set includes some very long stay durations, we let y n represent the log of the length of stay in days for all models.

We convert back from the log number of days to the number of days when reporting results.

The complete data set is again D = {(s n , y n )|n = 1, ..., N } with N = 53, 211 data cases (we again require 48 hours worth of data).

The goal in the regression task is to learn a regression function g of the formŷ n ← g(s n ) whereŷ n is a continuous value.

UWave dataset is an univariate time series data consisting of simple gesture patterns divided into eight categories.

The dataset has been split into 3582 train and 896 test instances.

Out of the training data, 30% is used for validation.

Each time series contains 945 observations.

We follow the same data preparation method as in BID19 where we randomly sample 10% of the observations points from each time series to create a sparse and irregularly sampled data.

The model is learned using the Adam optimization method in TensorFlow with gradients provided via automatic differentiation.

However, the actual multivariate time series representation used during learning is based on the union of all time stamps that exist in any dimension of the input time series.

Undefined observations are represented as zeros and a separate missing data mask is used to keep track of which time series have observations at each time point.

Equations 1 to 5 are modified such that data that are not available are not taken into account at all.

This implementation is exactly equivalent to the computations described, but supports parallel computation across all dimensions of the time series for a given data case.

Finally, we note that the learning problem can be solved using a doubly stochastic gradient based on the use of mini batches combined with re-sampling the artificial missing data masks used in the interpolation loss.

In practice, we randomly select 20% of the observed data points to hold out from every input time series.

For the time series missing entirely, our interpolation network assigns the starting point (time t=0) value of the time series to the global mean before applying the two-layer interpolation network.

In such cases, the first interpolation layer just outputs the global mean for that channel, but the second interpolation layer performs a more meaningful interpolation using the learned correlations from other channels.

The Logistic Regression model is trained with cross entropy loss with regularization strength set to 1.

The support vector classifier is used with a RBF kernel and trained to minimize the soft margin loss.

We use the cross entropy loss on the validation set to select the optimal number of estimators in case of Adaboost and Random Forest.

Similar to the classification setting, the optimal number of estimators for regression task in Adaboost and Random Forest is chosen on the basis of squared error on validation set.

We evaluate all models using a five-fold cross-validation estimate of generalization performance.

In the classification setting, all the deep learning baselines are trained to minimize the cross entropy loss while the proposed model uses a composite loss consisting of cross-entropy loss and interpolation loss (with δ R = 1) as described in section 3.2.3.

In the case of the regression task, all baseline models are trained to minimize squared error and the proposed model is again trained with a composite loss consisting of squared error and interpolation loss.

We follow the multi-task Gaussian process implementation given by BID12 and treat the number of hidden units and hidden layers as hyper-parameters.

For all of the GRU-based models, we use the already specified parameters BID4 .

The models are learned using the Adam optimization.

Early stopping is used on a validation set sub-sampled from the training folds.

In the classification case, the final outputs of the GRU hidden units are used in a logistic layer that predicts the class.

In the regression case, the final outputs of the GRU hidden units are used as input for a dense hidden layer with 50 units, followed by a linear output layer.

We independently tune the hyper-parameters of each baseline method.

For GRU-based methods, hidden units are searched over the range {2 5 , 2 6 , · · · , 2 11 }.

Learning is done in same way as described above.

We evaluate all the baseline models on the test set and compare the training time and accuracy.

For the Gaussian process model, we use the squared exponential covariance function.

We use the same number of inducing points for both the Gaussian process and the proposed model.

The Gaussian process model is jointly trained with the GRU using stochastic gradient descent with Nesterov momentum.

We apply early stopping based on the validation set.

In this section, we address the question of the relative information content of the different outputs produced by the interpolation network used in the proposed model for MIMIC-III dataset.

Recall that for each of the D = 12 vital sign time series, the interpolation network produces three outputs: a smooth interpolation output (SI), a non-smooth or transient output (T), and an intensity function (I).

The above results use all three of these outputs.

To assess the impact of each of the interpolation network outputs, we conduct a set of ablation experiments where we consider using all sub-sets of outputs for both the classification task and for the regression task.

TAB3 shows the results from five-fold cross validation mortality and length of stay prediction experiments.

When using each output individually, smooth interpolation (SI) provides the best performance in terms of classification.

Interestingly, the intensity output is the best single information source for the regression task and provides at least slightly better mean performance than any of the baseline methods shown in TAB1 .

Also interesting is the fact that the transients output performs significantly worse when used alone than either the smooth interpolation or the intensity outputs in the classification task.

When considering combinations of interpolation network components, we can see that the best performance is obtained when all three outputs are used simultaneously in classification tasks.

For the regression task, the intensity output provides better performance in terms of median absolute error while a combination of intensity and transients output provide better explained variance score.

However, the use of the transients output contributes almost no improvement in the case of the AUC and cross entropy loss for classification relative to using only smooth interpolation and intensity.

Inter-estingly, in the classification case, there is a significant boost in performance by combining smooth interpolation and intensity relative to using either output on its own.

In the regression setting, smooth interpolation appears to carry little information.

In this section, we compare the performance of the proposed model on a previous MIMIC-III benchmark dataset BID13 .

This dataset only consists of patients with age > 18.

Again, we focus on predicting in-hospital mortality using the first 48 hours of data.

This yields training and test sets of size 17,903 and 3,236 records respectively.

We compare the proposed model to multiple baselines from BID13 .

In all the baselines, the sparse and irregularly sampled time-series data has been discretized into 1-hour intervals.

If there are multiple observations in an interval, the mean or last observation is assigned to that interval, depending on the baseline method.

Similarly, if an interval contains no observations, the mean or forward filling approach is used to assign a value depending on the baseline method.

We compare with a logistic regression model and a standard LSTM network.

In the multitask setting, multiple tasks are predicted jointly.

Unlike the standard LSTM network where the output/hiddenstate from the last time step is used for prediction, we provide supervision to the model at each time step.

In this experiment, we use an LSTM as the prediction network in the proposed model to match the baselines.

@highlight

This paper presents a new deep learning architecture for addressing the problem of supervised learning with sparse and irregularly sampled multivariate time series.

@highlight

Proposes a framework for making predictions on sparse, irregularly sampled time-series data using an interpolation module that models the missing values in using smooth interpolation, non-smooth interpolation, and intensity. 

@highlight

Solves the problem of supervised learning with sparse and irregularly sampled multivariate time series using a semi-parametric interpolation network followed by a prediction network.