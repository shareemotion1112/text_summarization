Many real-world time series, such as in activity recognition, finance, or climate science, have changepoints where the system's structure or parameters change.

Detecting changes is important as they may indicate critical events.

However, existing methods for changepoint detection face challenges when (1) the patterns of change cannot be modeled using simple and predefined metrics, and (2) changes can occur gradually, at multiple time-scales.

To address this, we show how changepoint detection can be treated as a supervised learning problem, and propose a new deep neural network architecture that can efficiently identify both abrupt and gradual changes at multiple scales.

Our proposed method, pyramid recurrent neural network (PRNN), is designed to be scale-invariant, by incorporating wavelets and pyramid analysis techniques from multi-scale signal processing.

Through experiments on synthetic and real-world datasets, we show that PRNN can detect abrupt and gradual changes with higher accuracy than the state of the art and can extrapolate to detect changepoints at novel timescales that have not been seen in training.

Changepoints, when the structure or parameters of a system change, are critical to detect in many domains.

In medicine, finance, climate science and other fields, these changes can indicate that important events have occurred (e.g. onset of illness or a financial crisis), or changed in important ways (e.g. increasing illness severity).

In both cases, these affect decision-making.

Changepoint detection (CPD) aims to find these critical times.

However, changes may result in complex patterns across multiple observed variables, and can be hard to recognize, especially in multivariate timeseries where interdependencies exist among variables.

Further, not all changepoints lead to a sudden transition, many occur over a duration of time (e.g. weightloss, transition between activities) and are harder to identify.

Various methods have been proposed for CPD including parametric methods BID0 BID49 BID34 , which make strong assumptions about data distributions, and nonparametric methods BID12 BID39 , which are based on engineered divergence metrics or kernel functions.

Most parametric methods are highly context specific, and face difficulty when changes result in complex temporal patterns that are hard to model manually.

For nonparametic methods, the main drawback is that these methods rely heavily on the choice of parameters or kernels.

To handle data from different domains, BID8 proposed a nonparametric CPD method.

However, like many other CPD methods, it can only detect abrupt changes.

Yet in real-world applications, the effect of a change may be gradual and may happen over different durations.

Some methods have been explicitly designed for detecting gradual changepoints BID2 BID20 , but cannot handle changes occurring at arbitrary timescales.

In some applications, like detecting changes in activity, how quickly someone transitions from sitting to standing should not affect accuracy at detecting the transition.

In contrast, Deep Neural Networks (DNN) have been used for time series forecasting BID43 and classification as they can learn functions automatically.

These can be more easily adapted to new tasks if there is sufficient training data.

However, DNNs typically need enough examples of all possible ways a pattern can appear, and thus all possible transition speeds, to reliably detect it in test data.

Since this data is costly and may be infeasible to collect in some cases, it is ideal to have a scale-invariant approach that can generalize beyond observed timescales.

We propose a novel DNN architecture for CPD using supervised learning.

Our approach makes two key contributions to neural network architecture: a trainable wavelet layer that transforms input into a pyramid of multiscale feature maps; and Pyramid recurrent neural networks (PRNN), which build a multi-scale Recurrent Neural Network (RNN) on top of a multi-channel Convolutional Neural Network (CNN) processing the wavelet layer.

Finally, we use a binary classifier on the PRNN output to detect changepoints.

On both simulated and real-world data, we show that the proposed model can encode short-term and long-term temporal patterns and detect from abrupt to extremely gradual changepoints.

The model is scale invariant, and can detect changes at any timescale, regardless of those seen in training.

We focus on the task of CPD, but this architecture may have more general applications in time series analysis.

Changepoint detection CPD is a core problem for times-series analysis.

One approach is to use a model and find times when observations deviate from what is predicted by the model.

Bayesian Online ChangePoint Detection (BOCPD) BID0 can find changepoints in an online manner, but makes the limiting assumption that the time series between changes has a stationary exponential-family distribution.

More generally, Bayesian techniques require full definition of the likelihood function BID33 BID34 , which may be difficult to specify.

Nonparametric models increase the flexibility, such as in BID39 which is an extension of BOCPD to Gaussian Processes.

This however, may significantly increase computational complexity.

BID44 introduced Gaussian Graphical Models (GGMs) for CPD, extending BID13 to handle mutlivariate time series.

GGM is offline and models the correlations between multivariate time series using multivariate Gaussian.

This method is closest to ours as a result, but makes strong assumptions about the data distribution.

Non-Bayesian techniques exist, such as BID46 , which uses an autoregressive model for each time series segment, but this model is limiting.

To eliminate the need to specify a model, model-free approaches have emerged, such as densityratio estimation methods BID26 BID32 BID28 BID29 , kernel methods BID19 , and other techniques that define custom divergence functions like difference of covariance matrix BID4 BID3 or carefully engineered statistics BID6 BID22 BID38 BID16 .

However, covariance matrix based methods cannot deal with the case when the change point does not cause significant variations in covariance matrix.

Statistics based methods such as MMD BID16 , Hotellig T-square BID9 , CUSUM (Page, 1954) , or generalized likelihood ratio (GLR) BID23 have their own limitations like relying heavily on the choice of kernels (MMD) or parameters (Hotellig T-square), being highly dependent on prior information (CUSUM), or having high complexity for large sample size (GLR).

Thus while such models might work in a specific application, they cannot be readily used in a different domain without re-engineering the divergence or kernel functions.

Few methods were explicitly designed to detect gradual changes, though BOCPD has been extended this way by reformulating changes as segments instead of points BID2 .

Alternatively, gradual changes can be formulated as concept drifts BID20 .

We do not reformulate the changepoint detection problem, and instead make the model scale-invariant, so it can handle short-and long-term temporal patterns.

This results in a model that can generalize to novel time-scales without extra effort.

A similar problem is anomaly detection BID24 BID17 .

For instance, BID15 learns a one-class Support Vector Machine (SVM) on normal data, and distinguishes normal from abnormal in new data.

However, a changepoint is not always a transition to an abnormal state and may be between two normal states, such as human activities.

Our proposed approach is not limited to binary classification, and can be re-purposed by training with a one-class loss that is used for anomaly detection.

Deep learning Core challenges for CPD are scaling with more variables and recognizing changes resulting in complex patterns involving many variables.

Deep neural networks provide a promising solution for CPD, as they can learn to recognize complex patterns without engineering of features and metrics.

CNNs for instance, learn to extract increasingly abstract features from raw data through a stack of non-linear convolutions.

This leads to recognition of complex patterns such as hundreds of object types in natural images BID42 .

RNNs on the other hand, learn complex temporal patterns in sequences of arbitrary length, which is used in applications such as human activity recognition with wearable sensors BID18 .

These are exactly the type of pattern changes that pose challenges for CPD.

On the other hand, a key feature of CNNs is shiftinvariance, meaning the prediction will not change even if a pattern shifts in time or space.

Gated variants of RNN such as Long Short-Term Memory (LSTM) networks BID21 and attention-augmented networks BID1 can also learn shift-invariance, due to their ability to control which part of data to attend or ignore.

Ideally, a CPD method should perform equally well on test data regardless of whether changes happen faster or slower than seen in training data.

However, the fixed resolution of CNN and RNN architectures makes them sensitive to scale.

CNNs have been extended to model multiple scales simultaneously BID40 , but this is not a scale invariant method, as features are simply concatenated.

For RNNs, BID11 propose a hierarchical architecture to process a sequence through successive RNN layers, at different resolutions.

However, layers of RNN there resemble layers of convolution in CNNs (modeling the signal at a different abstraction level) and are not invariant to scale changes at the same abstraction level.

Therefore, we propose a new architecture, PRNN, that exploits both CNN and RNN, while augmenting them with scale invariance.

Another limitation of CNNs, and to some extent RNNs for CPD, is the difficulty of modeling longterm dependencies.

However, this is necessary to recognize gradual changes.

Dilated convolutions have recently allowed long-term dependency modeling in CNNs BID48 BID36 .

RNNs are naturally built to model long-term dependencies, but suffer from vanishing gradients.

Extensions such as LSTMs and Gated Recurrent Units (GRU) BID10 solve the problem of vanishing gradients, but still have limited memory space.

Intuitively, information from an infinitely long sequence cannot be stored in a fixed-dimensional RNN cell.

To reduce the computation complexity for conventional RNN, Campos et al. (2017) proposed Skip RNN to skip state updates while preserving the performance of baseline RNN models.

Their skipping-state-updates operation has the advantage of avoiding redundant RNN updates.

However, this has the risk of skipping temporal dependencies, especially for long term dependencies, which can hurt the overall performance of RNN.

To address this, recent work has augmented RNNs with various types of memory or stack BID41 BID25 , but these methods are not scale-invariant.

Our PRNN, models infinitely long sequences with its multi-scale RNN, which forms a stack of memory cells in an arbitrary number of levels.

A higher-level RNN cell in a stack has lower resolution, and thus can store longer dependencies at no additional computational cost, while a lower-level RNN cell has a high resolution and prevents the loss of details in the short term.

Frameworks like Feature pyramid networks BID31 and wavelet CNN BID14 has been proposed to deal with images with different scales or resolutions.

However, both of them cannot be applied directly on multivariate time series for change point detection as they cannot model the temporal dependencies in multivariate time series.

We propose a new class of deep learning architectures called Pyramid Recurrent Neural Networks (PRNNs).

The model takes a multi-variate time series and transforms it into a pyramid of multi-scale feature maps using a trainable wavelet layer (NWL).

All pyramid levels are processed in parallel using multiple streams of CNN with shared weights, yielding a pyramid of more abstract feature maps.

Next, we build a multi-scale RNN on top of the pyramid feature map, to encode longer-term, dependencies.

The PRNN output is used to detect changes at each time step with a binary classifier.

CNNs can learn to recognize complex patterns in multivariate time series, partly due to parametersharing across time (via the convolution operation), which leads to shift-invariance.

However, CNNs are not scale-invariant, so a learned pattern cannot necessarily be recognized when it appears more gradually or more quickly.

To augment CNNs with scale invariance, we introduce Deep Wavelet Neural Networks (DWNN), which consist of a proposed Neural Wavelet Layer followed by parallel streams of CNN.The Neural Wavelet Layer (NWL) can be seen as a set of multi-scale convolutions with trainable kernels, which are applied in parallel on each variable of the input time series.

The input to the NWL is a multivariate time series, X ∈ R T ×c , where T is the number of timepoints and c is the number of variables.

The NWL takes X and produces multiple feature maps, which together form a pyramid of convolution responses.

That is: DISPLAYFORM0 An example is shown in FIG0 .

Specifically, the NWL uses the filter bank technique ?

for discrete wavelet transform.

Given a pair of separating convolutional kernels (typically a low-pass and a highpass kernel), it convolves the signal with both, outputs the high-pass response, and down-samples the low-pass response for the next iteration.

It repeats this process and in each iteration outputs an upper level of the output pyramid.

Although traditional wavelets such as Haar or Gabor ? can be used, we have experimentally found that initializing the filter banks with random numbers and training them using backpropagation with the rest of the network leads to higher accuracy.

More formally, the NWL is characterized by its trainable kernels K DISPLAYFORM1 ∈ R τ ×c for all variables v ∈ {1...c}, where τ is the kernel size.

Given each channel of X as input (e.g. X (v) ), the NWL iteratively computes lowpass and highpass responses, starting with L1 and H1 , that are: DISPLAYFORM2 where * is convolution and ω is a downsampling operation (e.g. implemented by linear interpolation).

At the i-th iteration of the wavelet transform, given L DISPLAYFORM3 This operation is repeated for a pre-specified number of times, k, or until the length of L (v) DISPLAYFORM4 becomes smaller than a threshold.

The hyperparameter, k, can be selected using cross-validation.

A larger k (or smaller threshold) results in a larger receptive field at the highest level of the pyramid, enabling the detection of more gradual patterns.

However, a large k also brings more computation and also requires a larger buffer in the case of online processing.

The output of each iteration i ∈ {1...k} for variables v ∈ {1...c} can be concatenated to form DISPLAYFORM5 where [.|.] indicates concatenation.

The output of the NWL is the stack of all H i .

These are called different levels of a pyramid throughout this paper.

In the original filter bank method the last lowpass response, L k , is also stacked with the output but we did not observe an improvement with L k .The key advantage of a NWL over a conventional convolution layer is that a single wavelet can encode the input with multiple granularities at once, whereas a single convolution only encodes a single granularity.

Although different layers of a CNN have different granularities, they encode the data at a different level of abstraction, and thus cannot simultaneously extract the same pattern at different scales.

On the other hand, a single wavelet layer can encode changes with the same patterns at different paces, simultaneously into the same feature map, at different levels of the pyramid.

We will use the proposed NWL as a part of a larger, deeper architecture, which is described in the rest of this section.

Hence, an important aspect of NWL is that it can be used as a layer of a deep network, in composition with other neural layer types such as convolutional and fully connected layers.

For example, the input to a wavelet layer can be the output of a convolutional layer.

Alternatively, to stack a convolutional layer on the output of a wavelet layer, one should apply the convolution on each level of the wavelet pyramid, resulting in a pyramid-shaped output.

Accordingly, a network composed of one wavelet layer and an arbitrary number of other layers, can take a multi-variate time series as input, and produce a pyramid-shaped response as output.

We refer to such a network architecture as a Deep Wavelet Neural Network (DWNN).

In this paper we use a specific form of DWNN, which starts with a NWL, directly applied on the input time series X, followed by parallel streams of CNN with shared parameters, each of which takes one level of the NWL pyramid.

More specifically, we use an -layer CNN with a down-sampling stride of p j at the j-th layer, which results in a total down-sampling factor of P = j=1 p j , and with f j feature maps at the j-th layer.

We apply that CNN in parallel on each level of the output pyramid of the NWL, which means for each i ∈ {1...k}, it gets H i ∈ R T /2 i−1 ×c and outputs DISPLAYFORM6

The output of the DWNN is a multi-scale pyramid of sequential feature maps that encode short-term temporal patterns at different times and scales.

It is common to process sequential features using an RNN, to encode longer-term temporal patterns.

However, conventional RNNs process a single sequence, not a multi-scale pyramid of sequences.

Similar to the need for a wavelet layer, RNNs are not scale-invariant, meaning if an RNN can recognize a pattern, it does not necessarily imply it can recognize a temporally shortened or stretched instance of the same pattern without having seen this scale in the training data.

Further, RNNs fail to learn very gradual patterns, due to limited memory.

While this can be addressed by memory-augmented networks, they remain sensitive to scale.

To address these issues, we introduce a novel hierarchically connected variant of RNNs.

Our proposed network, PRNN, scans the multi-scale output of a DWNN, and simultaneously encodes temporal patterns at different scales.

An RNN is applied in parallel on different levels of the input pyramid.

On each level at each step, it takes as input the corresponding entry from the input pyramid, along with the most recent output of the RNN operating at the upper level.

We concatenate those two vectors and feed as input to the RNN.

We refer to this technique as Pyramid Recurrent Layer (PRL).Denoting the value at level i of the input pyramid at time t as C i [t] , and assuming the downsampling ratio in wavelet transform is d, (i.e., each level of the pyramid has d-times the length of its upper level) we can write the recurrent state at level i and time t as: DISPLAYFORM0 where σ is a nonlinear activation function such as ReLU, and W 1 , W 2 , W 3 and b are trainable parameters of this layer.

These parameters define a linear transformation of the current state, past state, and higher-level state, as illustrated in FIG1 .

Note that the proposed hierarchical structure is agnostic of the function of each cell.

Although we used a simple RNN cell for illustration, we could use any variant of RNNs such as a Long Short-Term Memory (LSTM) BID21 or Skip RNN BID5 as our RNN cell.

The proposed architecture can be compared with an RNN operating on a single data sequence.

If the data granularity is high, the RNN likely fails to model long-term dependencies, due to the wellknown problem of vanishing gradients.

One can lower the data granularity, so long-term patterns can be summarized in fewer steps, but this results in the loss of details.

Accordingly, conventional RNNs were not designed to effectively detect both abrupt and gradual patterns at the same time.

On the other hand, in the proposed PRL, each RNN unit is provided with inputs from the same level of granularity as well as the level above.

The RNN that operates at the lowest level, in turn, receives information from all levels of granularity.

FIG1 illustrates the effect of forgetting using decreasing color saturation.

While it is impossible to keep track of the past through the lower level alone, the information path from upper levels connect the past to present in only three steps.

This lets the PRL model long-term patterns, while it can still model details through the lower levels.

We propose PRNN as a composition of a DWNN and a PRL.

An input time series of arbitrary length is transformed through a DWNN into a pyramid-shaped representation, which is then fed into a PRL.

For CPD and other classification problems, a logistic regression layer is built on the output of the RNN cells that operate at the lowest level of the pyramid.

This layer produces detection scores at each time step with the highest possible granularity.

Specifically, the detection score for time t is: DISPLAYFORM0 where σ is the sigmoid function and W o and b o are trainable parameters.

The classification loss at each time is the cross entropy loss written as: DISPLAYFORM1 where y * t is the ground truth at time t. We optimize this loss using stochastic gradient descent on parameters of the classifier (W o and b o ), PRL (W 1 , W 2 , W 3 and b), and NWL (K l and K h ).

We compare the proposed PRNN to conventional deep learning baselines.

Using both simulated and real-world datasets, we show that PRNNs can detect abrupt and gradual changes more accurately than baseline approaches and can be used for activity recognition by learning labels for different changes.

Synthetic dataset We create a synthetic dataset to evaluate accuracy at simultaneously detecting gradual and abrupt changes.

We construct 2000 time series each with 12 variables and 8192 time steps (a power of two chosen to avoid rounding errors in downsampling).

Each time series is a combination of a Brownian process and white noise and has 4 changepoints at randomly chosen times.

A change is a shift in the mean of 4 randomly chosen dimensions, with randomly chosen speed (duration of change) and amount of shift.

A speed of 0 gives an abrupt change, while longer ones provide more challenging cases to recognize.

An example of the simulated time series together with ground truth and detection results are shown in FIG2 .

We randomly split the data in half, 1000 for training and 1000 for testing.

To demonstrate robustness of the proposed method against variability in scale, we also do a split by scale, where all changes in one half are strictly more gradual than all in the other half.

Opportunity dataset For real-world evaluation, we first use the OPPORTUNITY activity recognition dataset BID7 , which consists of on-body sensor recordings from 4 participants performing activities of daily living, such as cleaning a table.

Each participant has 6 records (runs) of around 20min each.

Values of 72 sensors from 10 modalities were recorded at 30Hz, and manually labeled with 18 activity types.

Following BID18 , we ignore variables with missing values, which leads to 79 variables for each record.

We use run 2 of subject 1 for validation and runs 4 and 5 of subjects 2 and 3 for test, and the rest for training.

To repurpose this activity recognition dataset for CPD, we consider the transition between two activities as a change.

This transition can take place at various durations, which makes the task challenging.

As ground truth, we use the temporal annotation provided with the OPPORTUNITY dataset to determine moments that the activity type changes.

FIG5 shows a sample of this dataset with ground truth and detection results.

We also test our methods on the Bee Waggle Dance data BID35 .

Honey bees perform waggle dance to communicate with other bees about the orientation and distance to the food sources.

The Bee Waggle Dance data includes six videos of bee waggle dances with 30 frames per second.

The data include 3 variables encoding the honey bee's position and head angle at each frame.

Using the position and angle information, each frame is labeled with activity of "turn left", "turn right", or "waggle dance." Similar to the OPPORTUNITY dataset, we consider the transition between two activities of the honey bee as a change point.

We test our method and other baselines on "sequence 1" of the bee data.

We train on the first 256 frames (a power of 2 chosen to avoid rounding errors) and test on the other 768 frames.

We use small size of training data to see how the proposed method behaves and for consistency with other prior works BID39 .

We compare the proposed architecture to the following unsupervised CPD method and supervised deep-learning baselines: GGM Xuan & Murphy (2007) is related to BOCPD BID0 , a classic method for CPD, but was selected to provide fairer comparison against our approach as it is offline and incorporates multivariate time series.

CNN We use a CNN that takes a time series as input and predicts a sequence of detection scores for changes.

Due to the widely used max-pooling layers, the output has a lower temporal granularity compared to the input.

We denote the ratio of output length to the input length as γ.

RCN We apply an RNN to the output of the CNN.

The output has the same granularity as CNN, while each step of the output has a larger receptive field that encodes all the past data.

DWNN We use the proposed DWNN, which is formed by applying an NWL to the input time series and feeding the output pyramid levels to parallel branches of a CNN.

The output of CNN branches are upsampled to have the same size and fused by arithmetic mean.

PRNN We apply the complete proposed method which consists of a DWNN followed by a Pyramid Recurrent Layer to fuse levels of the pyramid.

PRNN-S As a final baseline, we replace the conventional RNN cell in our PRNN method with a Skip RNN BID5 which was found to have lower time complexity.

This enables us to test whether an efficient RNN can preserve the performance of PRNN.

All of the deep-learning baselines share a core CNN architecture on which the additional modules are built.

We fix the architecture of the core CNN to be feature maps, and z is the pooling stride.

Each convolution layer is followed by max-pooling and ReLU activation.

The output of all baselines are fed to a fully connected perceptron with sigmoid activation which results in binary detection scores at each time step.

The granularity ratio γ for this architecture is 1/16.

For DWNN, PRNN, and PRNN-S, we used a 7-level wavelet with kernel size 3 for both synthetic and OPPORTUNITY dataset.

For Bee Waggle Dance data, due to the small size of the data and the more abrupt activities (compared with synthetic and OPPORTUNITY dataset) of honey bee, we used a 5-level wavelet with kernel size 3.

For all datasets RCN and PRNN used an LSTM cell with 256 hidden units.

We train all models using Adam (Kingma & Ba, 2014) with early stopping to avoid overfitting.

At test time, the models take a time series and predict a sequence of detection scores.

To detect changepoints, we apply non-maximum suppression with a sliding window of length ω and filter the maximum values with a threshold.

We evaluate AUC by iterating over this threshold.

Hyperparameter ω controls how nearby two distinct changes can be detected and is tuned for each method separately using cross-validation.

The real world datasets (Bee data and OPPORTUNITY data) are more challenging than the synthetic data, as they include diverse changepoints formed by transitions between many activity types.

To address this, we use multitask learning, training the model to both detect changes and classify activity by changing the output dimension of the last fully connected later to have multiple units (19 for OPPORTUNITY data, and 4 for Bee data).

For OPPORTUNITY data, the first 18 units predict a log probability for each activity and the last 1 unit outputs the probability of a change point (for bee data, it's 3 units and 1 unit).

We define a softmax cross-entropy loss on those 18 units and add it as a regularization term to the objective function.

Multitask learning improved the results equally for all baselines, because the model has auxiliary information, namely the activity type and not just the existence of a change.

For GGM, we use the full covariance model instead of the independent features model to capture the correlations between features.

We use a uniform prior as in BID44 , and set the pruning threshold to 10 −20 .

Since there is no training for GGM, we evaluate the algorithm using the same test data as all other methods we compared on both synthetic and real world dataset.

We evaluate precision and recall, and report AUC.

As detected changepoints may not exactly match the true changepoints, we use a tolerance parameter η that sets how close a detected change must be to a true change to be considered a correct detection.

We match detected changepoints to the closest true changepoint within η time steps.

Precision is the number of matched detections divided by the number of all detections, and recall is the number of matches divided by the number of true changes.

For the synthetic dataset, three different train/test splits were used to demonstrate extrapolation from gradual to abrupt changes and vice versa.

FIG3 shows the results for a random split (mixing scales), while FIG3 and 4b use the scale-variant split introduced in section 4.1.

In the scale-variant split, the model needs to extrapolate patterns learned from training data to scales that have not been observed.

This is extremely challenging for a model that is not scale-invariant.

This is apparent in FIG3 where both CNN and RCN show worse results in parts compared to their own performance in FIG3 .

From experiment of FIG3 to FIG3 , AUC of CNN (RCN) decreased from 41% (39%) to 15% (11%) when the tolerance is 64 steps (2 6 ).

This is because the methods are not designed to be specifically robust against scale variability.

In the transition from 4c to 4a, DWNN, PRNN, and PRNN-S like other methods inevitably suffer from a performance drop, which is due to the increase in task difficulty.

However, the amount of this drop is substantially lower for DWNN, due to the wavelet layer and shared parameters across scales.

At a tolerance of 64 steps, for instance, the performance drop for DWNN is 20%, which is lower than PRNN-S (22%) and CNN (26%).

Again DWNN and CNN respectively work better than PRNN and RCN in this setting, which is consistent with the overall results (See Appendix A for AUC details).While recognizing abrupt changes from gradual training ones FIG3 is easier than recognizing a mix of scales, CNN and RCN perform worse than our approach due to their inability to generalize in scale.

In FIG3 , when tolerance is 64 steps, AUC for CNN and RCN are 66% and 30%, which are lower than both PRNN (72%) and DWNN (79%).

In contrast, DWNN, PRNN-S, and PRNN have higher AUC than their own performance in the mixed experiment FIG3 .

The performance of PRNN at different tolerances is 20-25% higher on average in the mixed experiments than the experiment of "train abrupt, test gradual" FIG3 .

In the train on gradual and test on abrupt experiment (4b), DWNN performs even better than PRNN and PRNN-S in all tolerances, and similarly, CNN outperforms RCN.

This shows recurrent architectures are generally less effective for this kind of extreme generalization.

The high performance of DWNN 4b also shows the effectiveness of the added wavelet layer in modeling both gradual and abrupt changes in time series.

However, in realworld cases we are more likely to have a mix of scales in both training and test, and it is in this case ( FIG3 ) that PRNN is most accurate.

As shown in the AUC plots, it is in general more difficult to recognize gradual changes.

It is possible to adapt our work to detect segments rather than specific points (e.g. as in BID2 ), if instead of applying a non-maximum suppression on the output score map of change, we perform binary segmentation to detect intervals with continuously high detection score.

FIG2 shows example results for our scale invariant PRNN and scale sensitive CNN.

Overall CNN has a higher false positive rate, while also missing one of the changes.

While detected changes and ground truth are not always precisely aligned, the small gaps are acceptable in the case of gradual changes, where it can be hard to define a single moment when the change occurs.

FIG5 shows results and AUC plots for the OPPORTUNITY dataset.

In the time series, we see that CNN has a missed detection and at least one false positive around time 300, while PRNN detects all changes close to their actual times.

In FIG4 we see that PRNN outperforms other methods at all tolerance levels.

In contrast to the synthetic data, PRNN-S has significantly lower AUC than both PRNN and DWNN for every tolerance.

It may be that Skip RNN is skipping important information encoded in our wavelet later.

Finally, the performance of GGM is lowest for all cases.

This is not surprising, as it is an unsupervised method, and does not learn from previously observed patterns.

When the tolerance is 64 (around 2 seconds, η = 2 6 = 64), a reasonable value for practical activity recognition use, PRNN achieves 81% AUC while DWNN, RCN, CNN, and PRNN-S respectively achieve 75%, 74%, 69%, 47%.

Full results can be seen in Appendix B.1.

The five deep learning methods, PRNN, PRNN-S, RCN, DWNN, and CNN, respectively took 110, 105, 80, 24, and 6 minutes to train and converge on the OPPORTUNITY dataset.

Recurrent methods generally take longer due to backpropagation through time.

However, this only happens during training, and does not affect test complexity.

One can compare PRNN to RCN, and DWNN to CNN, and observe an increase in time complexity.

This is due to repeating computations on multiple levels of a pyramid.

This however, only multiplies the time complexity by a constant factor, since the length of pyramid levels exponentially vanish.

Note that DWNN has a superior performance to RCN in most cases, while also being faster to train.

FIG4 shows AUC plots for all methods we tested on Bee Waggle Dance dataset.

Our PRNN method outperforms other methods when the value of η is no less than 5 (around 1 second) with AUC of 93%.

Similar to the result on OPPORTUNITY dataset, GGM has the lowest AUC for all tolerances.

When the tolerance is 64 (around 2 seconds, η = 2 6 = 64), PRNN achieves 93% AUC while PRNN-S, RCN, and CNN respectively achieve 64%, 84%, 78% (see Appendix B.2 for AUC details).

Similar to OPPORTUNITY dataset, the drop of AUC for PRNN-S is caused by the skipping of states updates.

However, compared with the OPPORTUNITY data where PRNN-S has maximum AUC of 51%, PRNN-S for Bee Waggle Dance data has higher maximum AUC of 64%.

This is because the changes in honey bee activities are more abrupt than human activities, so the skipped updates have lower impact on the detection performance.

From the AUC plots, a change in tolerance affects our PRNN much less compared to other methods.

For instance, when the tolerance is lowered from 32 (η = 2 5 = 32) to 16 (η = 2 4 = 16), the AUC of RCN drops significantly (from 84% to 18%), while AUC of PRNN drops much less (from 93% to 61%).

CNN has a dramatic drop in accuracy from η = 4 to η = 3, suggesting it is consistently detecting changes with a delay.

Thus, PRNN is less sensitive to this parameter and more reliable for real world cases.

We propose a new class of DNNs that are scale-invariant, and show they can detect from abrupt to gradual changepoints in multimodality time series.

The core is 1) augmenting CNNs with trainable Wavelet layers to recognize short-term multi-scale patterns; and 2) building a pyramid-shaped RNN on top of the multi-scale feature maps to simultaneously model long-term patterns and fuse multiscale information.

The final model can detect events involving short-and long-term patterns at various scales, which is a difficult task for conventional DNNs.

Although this reduces the amount of training data required to learn from changes, the proposed method still requires clean labels.

Experiments show our approach detects changes quickly, with lower sensitivity to the tolerance parameter than other approaches.

For real-world applications, this leads to much higher reliability.

In future work we will real-world challenges (e.g. noisy data, missing/noisy labels) by incorporating robustness, semi-supervised learning methods, and multi-view learning techniques.

TAB0 -3 show the AUC (Area Under the ROC Curve) results for synthetic data.

To detect changepoints, we apply non-maximum suppression with a sliding window of length ω and filter the maximum values with a threshold.

We evaluate AUC by iterating over this threshold.

Since changepoints may not exactly match the true changepoints, we use a tolerance parameter η that sets how close a detected change must be to a true change to be considered a correct detection.

We match detected changepoints to the closest true changepoint within η time step.

TAB0 shows the results for the experiment of "train abrupt and test gradual" for synthetic data.

TAB1 shows the results for the experiment of "train gradual and test abrupt" for synthetic data.

TAB2 shows the results for the experiment of "train all and test all" for synthetic data.

TAB3 shows the results for Opportunity data.

TAB4 shows the results for Bee Waggle Dance data.

<|TLDR|>

@highlight

We introduce a scale-invariant neural network architecture for changepoint detection in multivariate time series.

@highlight

The paper leverages the concept of wavelet transform within a deep architecture to solve change point detection.

@highlight

This paper proposes a pyramid based neural net and applies it to 1D signals with underlying processes occurring at different time scales where the task is change point detection