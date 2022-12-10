Adoption of deep learning in safety-critical systems raise the need for understanding what deep neural networks do not understand.

Several methodologies to estimate model uncertainty have been proposed, but these methodologies constrain either how the neural network is trained or constructed.

We present Outlier Detection In Neural networks (ODIN), an assumption-free method for detecting outlier observations during prediction, based on principles widely used in manufacturing process monitoring.

By using a linear approximation of the hidden layer manifold, we add prediction-time outlier detection to models after training without altering architecture or training.

We demonstrate that ODIN efficiently detect outliers during prediction on Fashion-MNIST, ImageNet-synsets and speech command recognition.

Thanks to the powerful transformations learned by deep neural networks, deep learning is applied in an increasing number of applications.

But deep neural networks, as all data-driven models, tend to fail when input data differs from training data.

Adopting deep learning in increasingly complex and possibly safety-critical systems makes it crucial to know not only whether the model's predictions are accurate, but also whether the model should predict at all.

If the model is able to detect outlier observations post-training directly, the system can fall back to safe behaviour minimizing negative consequences of faulty predictions.

By understanding the limits of models' learned representations and detecting when observations are not recognized, autonomous decision making based on deep learning can be improved.

In manufacturing process control, predictive models have long been used to predict process outcomes as well as detecting outlier input for decades BID8 BID17 BID9 .

A widely used model for this purpose is Partial Least Squares regression (PLS) BID33 , which project input data onto a set of linear latent variables prior to prediction.

In the latent variable space, the distance from new observations to the training data distribution is used to detect outlier observations.

The latent variables can also be used to approximate input observations, meaning that outliers can also be detected by measuring the distance to the latent variable subspace itself.

However, the layers of a neural network learn a non-linear mapping from input to output data spaces rather than a single linear subspace.

This makes it difficult to directly determine the limits of a neural network model's knowledge in the same manner as for a PLS model.

In this paper we present Outlier Detection In Neural networks (ODIN), a method for detecting outliers during prediction in deep neural networks.

Based on principles long used in manufacturing process control, we propose using a linear approximation of intermediate activations to provide a fixed representation of the training data.

By comparing new data to this fixed representation we are able to detect outliers during prediction without imposing constraints on architecture or training.

This allows us to use ODIN as a plug-in method allowing reliable and safe autonomous decision making based on deep learning.

A wide collection of methods allowing neural networks to describe uncertainty in predictions, which can be used to determine if new observations are outliers, have been proposed.

For decades, many methods have been formulated within a Bayesian framework BID7 BID21 allowing neural networks to predict probability distributions rather than point inferences.

The predictive uncertainty can then be estimated by the entropy or variance of the predicted distribution.

BID10 proposed MC-dropout, using prediction time dropout and Monte-Carlo sampling.

In summary, MC-dropout make multiple predictions per inference while the network is randomly perturbed by drop-out which results in a predicted distribution.

A number of alternatives to using dropout to perturb Monte-Carlo samples have been proposed in recent years including: sampling based on batch-normalization parameters BID29 , model ensembles BID18 , multiple prediction heads in a shared base network BID25 BID14 BID22 , variational inference of weight distribution instead of regular point weights BID0 and Laplace approximation of distributions from existing weights BID26 .

However, the mentioned methods constrain either how the network is constructed BID18 BID25 BID14 BID22 or how the network is trained BID10 BID29 limiting their use in systems already in production.

Several methods also rely on multiple inferences per prediction BID10 BID29 BID0 BID26 .

This limits their use in real-time systems or systems with limited computational resources.

An alternative approach for estimating uncertainty in classification problems is presented by where linear classifiers are trained to classify the target output given intermediate layers of a given base model.

The linear classifier outputs are then fed to a meta-model that is trained to estimate whether or not the base model is correct.

Another alternative approach is proposed by BID19 that leverage Generative Adversarial Networks (GANs) BID12 to augment the original dataset with border-line outliers.

A deep neural classifier is then trained to output high uncertainty for outlier observations and low uncertainty for the original observations.

This method does however involve training a GAN that can be difficult to train to convergence BID23 .Anomaly detection is closely related to prediction time outlier detection, and there are many methods for flagging deviating observations.

Non neural-methods include one-class support vector machines BID27 , local observation density BID2 , distances BID16 , isolation forests BID20 and many others.

A multitide of methods based on deep neural networks, typically autoencoders have been developed as well BID35 BID3 BID36 BID24 .

Of particular relevance to this work is BID24 , that use reconstruction residual as metric to flag outliers.

Important to note is that outlier detection systems are based on training a separate model to detect deviant observations.

Prediction time outlier detection, on the other hand, describes the limits of a predictive model's knowledge.

In this section we briefly describe the Partial Least Squares regression model, and how its latent variable approximation of the input data space is used to detect outliers after training.

We then describe how we can apply similar principles in neural networks by using a linear approximation of the hidden layer manifold, in a method we call Outlier Detection In Neural networks (ODIN).

Partial least squares regression (PLS) BID32 BID11 ) is a widely used regression model within manufacturing process control.

Similar to Principal Component Analysis (PCA), PLS assumes that high-dimensional data resides in a sub-space of the original data space spanned by so called latent variables and formulated in terms of matrix decomposition.

The PLS model is summarized as: DISPLAYFORM0 where the n × m input matrix X is decomposed into n × k latent variable matrix T = [t 1 ... t k ] and m × k loading matrix P = [p 1 ... p k ] with residual matrix E. The n × p response matrix Y is predicted using T multiplied with response weight matrix C = [c 1 ... c k ] with residuals F .

The latent variable-matrix T spans a orthogonal subspace of the data-matrix X and maximize the covariance between X and Y .

Note that the PLS model of X is similar to how PCA approximates the data through matrix decomposition but PCA finds latent variables t i that maximize the variance in X rather than the covariance between two matrices.

The columns in T are typically calculated sequentially, where the first column t 1 is found through basis-vectors w 1 ∈ R m and c 1 ∈ R p solving the optimization problem: DISPLAYFORM1 The corresponding loading vector p 1 is then chosen so thatX 1 = X − t 1 p T 1 is uncorrelated with t 1 .

This is achieved by selecting p 1 as: DISPLAYFORM2 The subsequent vectors t i , w i , p i where i ∈ [2, ..., k] are then calculated by repeating equations 2 and 3 usingX DISPLAYFORM3 The latent variable formulation means that PLS carries its own model of the training data and provides two ways of detecting outliers during prediction.

Since new observations are projected to the low-dimensional sub-space spanned by the latent variables, both distance to the sub-space itself and distance to training observations within the sub-space can be used to detect outliers.

Distance to sub-space is typically measured using the residual sum of squares (RSS).

RSS for a new observation row vector x new ∈ R p is given by: DISPLAYFORM4 where x new is approximated as DISPLAYFORM5 There are several ways to estimate the distance to training observations within the sub-space and a common choice is the Mahalanobis distance.

The Mahalanobis distance is a well-used statistical distance measuring how many standard deviations away an observations is from the origin in a multivariate probability normal distribution.

Given a fitted PLS model, the training data projections can be approximated as a multivariate normal distribution with covariance matrix C T = E(T T T ).

Then the Mahalanobis distance for x new is given by: DISPLAYFORM6 Alternatively, to compensate for using a linear model of a possibly non-linear manifold, a density based metric within the latent variable space may be used.

For instance, by using the Local Outlier Factor (LOF) BID2 , observations within low-density regions may be flagged as outliers instead of only using the Mahalanobis distance.

In contrast to PLS, data are not typically linearly mapped to a single sub-space in deep neural networks.

Instead, a neural network performs as a nested series of non-linear transformations.

That is, the activation vector a i of an observation vector x from a layer i is given by: DISPLAYFORM0 with weight-matrices W k and activation functions f k .According to the manifold hypothesis, data is assumed to reside near a region of low dimensionality that may be highly entangled.

One possible explanation for why deep neural networks work well is that they are able to disentangle complicated manifolds BID1 .

If deep neural networks disentangle manifolds, we may find a transformation from a complicated data manifold to a manifold that is approximately linear.

If this hypothesis holds true, we can apply a simple trick to add prediction time outlier detection to neural network models by building a model of the data representation within the model.

Given n × m activation matrix DISPLAYFORM1 T of training data X, where the row vectors a i,k are given by equation 6, we approximate the activation manifold using PCA as: DISPLAYFORM2 where the n × k latent variable matrix T Ai contain the projections of the A i onto the orthonormal sub-space spanned by columns of the m × k loading matrix P Ai .

Now we have a fixed orthogonal approximation of the activation manifold that we can use to detect outliers during prediction analogously to PLS (see 3.1).

Meaning that we can measure the distance from new observations to the activation manifold as the residual sum of squares similar to equation 4 using the observation activation a i,new with projection t ai,new = a i,new P Ai : DISPLAYFORM3 Similarily, the distance from training observations within the manifold can be measured using Mahalanobis distance or density based approaches as the Local Outlier Factor within the linear approximation.

For the Mahalanobis distance, the covariance matrix of the activation projections DISPLAYFORM4 We choose to call our method Outlier Detection In Neural networks, ODIN, since we use the intermediate data representations within the neural network itself.

In contrast to common Bayesian approaches, we do not perturb predictions to produce prediction distributions.

We simply measure the distance from new observations to the training data to determine whether they are outliers or not.

In the following sections we demonstrate how to detect outliers during prediction time using ODIN on different classification tasks.

We choose classification tasks for demonstration since it is straightforward to simulate outliers by excluding a subset of the classes.

We also explore how to choose what layer's activations to use and rank of PCA approximation.

For comparison, we also perform outlier detection using MC-Dropout BID10 since it is well-established and straightforward to implement even though it has received criticism BID25 .

To provide a simple classification problem with outliers encountered during prediction, we use the Fashion-MNIST BID34 dataset.

Fashion-MNIST consists of 70 000 greyscale 28x28 pixel images, out of which 10 000 are test set images, of ten categories of fashion products.

We excluded five classes to use as outliers, including all shoes (sandals, ankle boots and sneakers) and two clothing classes (pullover and shirts).

The intuition is that shoe-images are strong outliers since all shoe-related information is absent from training data, and excluded clothes are more subtle outliers since the training data contain other upper body garments.

We trained a small convolutional neural network (CNN, for architecture see Figure A .1) on five out of the ten classes.

We used rmsprop BID30 ) optimization, categorical cross entropy loss function, batch size 128 for 10 epochs and kept 10 % of the images as validation set and achieved a test set accuracy of 97 %.

To use for outlier detection, we extracted features from both max-pooling layers (without global average pooling) for all images.

We evaluate ODIN using different outlier metrics (RSS, Mahalanobis distance and LOF), five levels of explained variance (R2) of the PCA model (50-99 %) and different layers of extracted features using the area under the receiver operating characteristic curve (ROC-AUC) as performance metric, (see FIG0 , complete results in TAB1 .1).

We calculate the ROC-AUC comparing how well test set observations are separated from outlier observations.

For comparison, we also used MCdropout to calculate the image-wise entropy from 50 Monte Carlo samples per image and evaluated the results using ROC-AUC in the same way as ODIN.All metrics clearly separate strong outliers (shoes) from the test set images FIG0 left) with RSS being most sucessful (ROC-AUC 0.97 compared to Mahalanobis 0.94 and LOF 0.91).

There is a trend that they benefit from increased PCA R2.

Surprisingly MC-dropout failed to detect shoe outliers (ROC-AUC 0.495).

The subtle outliers (non-shoes) are significantly more difficult to detect FIG0 , and LOF is most successful doing so (best ROC-AUC 0.71 compared to RSS 0.60, Mahalanobis 0.61 and MC-Dropout 0.63).To conclude, the Fashion-MNIST experiment show that ODIN successfully detect outliers in a simple image classification problem.

Strong outliers seem to be best detected by measuring distance to manifold while subtle outliers are better detected in low-density regions of the linear approximation using LOF.

In order to provide a more complex example we demonstrate prediction time outlier detection using a pre-trained CNN on image synsets from ImageNet BID6 ).

We train a cat vs. dog classifier on the cat-and dog-synsets from ImageNet and used the car-and horse-synsets as outliers.

We used an Inception v3-network BID28 pre-trained on ImageNet, freezing all Inception module weights during training.

We replaced the penultimate layer with a hidden layer of 128 ReLu units and a single sigmoid output with 50 % dropout before and after the ReLu-layer and trained for 50 epochs using the Adam optimizer BID15 and achieved a test set accuracy of 93 %.We extracted features from each inception module in the Inception-v3 network and pooled them feature-map wise using global average pooling.

For each layer of features, we performed outlier detection with five levels of explained variance (R2) for the PCA model (50-99 %) and different outlier We are able to convincingly detect cars using our cats vs. dogs classifier (best ROC-AUC for RSS, 0.96, Mahalanobis distance 0.94 and LOF 0.93).

Horses are also detected as outliers even though they share visual features with cats and dogs (best ROC-AUC for RSS 0.76, Mahalanobis distance 0.75 and LOF 0.69).

Since we used dropout for the last fully connected layers, we also performed MC-dropout achieving similar results (ROC-AUC: 0.86 for cars, 0.61 for horses).

The degree of explained variance was not as influental in this experiment as in the Fashion-MNIST experiment, but both Mahalanobis distance and LOF fail to detect both cars and horses using 50 % R2.

Interestingly, the performance of all metrics peak at inception module 8 where an auxilliary output was used during training on ImageNet BID28 .To conclude, the experiment on cats and dogs show that ODIN reliably detect outliers using a pretrained CNN on real-world images.

ODIN performs slightly better than MC-dropout but does not rely on using dropout, or any type of constraint on the training procedure.

In line with the results from the Fashion-MNIST experiment, higher PCA R2 produce more reliable results.

To show that ODIN for prediction time outlier detection works for not only CNN-based image classification, we perform a speech command recognition experiment using a LSTM-based model.

We use the Speech Commands dataset BID31 ) that consists of 105 000 short utterances of 35 words recorded at 16 kHz sampling-rate.

The words includes digits zero to nine, command words yes, no, up, down, left, right, on, off, stop, go, backward, forward, follow, learn and visual.

The dataset also include a set of arbitrary words bed, bird, cat, dog, happy, house, marvin, sheila, tree and wow.

In our experiment, we train a classification model of both digits and command words and use the arbitrary words as outliers.

We transform the utterances into 64 Mel-Frecuency Cepstral Coefficients BID5 , using a frame-length of 2048 samples and frame-stride of 512 samples.

We train a three layer bi-directional LSTM-model with 30 % dropout after each LSTM-layer and softmax output (see architecture in Figure C .1) for 30 epochs, using the Adam-optimizer BID15 and batch-size 512 resulting in test-set accuracy of 78 % for the 25 classes.

The classification accuracy is lower than the 88 % accuracy of the baseline CNN:s BID31 ), but we believe it is sufficient for demonstrating prediction time outlier detection.

For outlier detection, we extracted training set features from the third LSTM-layer and fitted a PCAmodel explaining 99 % of the variance and chose RSS and Mahalanobis distance limits to be the 9th deciles of the training set distances.

We then extracted features from the test set and outlier classes and projected them onto the PCA-model and calculated RSS and Mahalanobis distances.

Using precision, recall and F1-score we evaluated outlier detection at the 9th deciles (see FIG2 , complete results in TAB1 .1).

We also combined RSS and Mahalanobis distance classifications using OR and AND combined classification.

For comparison, we also used MC-dropout with 10 Monte Carlo samples per utterance and calculated the sample-wise Shannon entropy.

We performed outlier detection using the 9th decile of training set entropies as threshold, and evaluated MC-dropout in the same manner as ODIN.Detecting outliers in the speech-commands dataset is difficult for both ODIN and MC-dropout with best word-wise F1-scores ranging from 0.25 for tree, which is phonetically similar to three, to 0.47 for house.

ODIN consistently outperform MC-dropout.

Additionally, since two metrics are used, we also have the opportunity to raise precision or recall by using AND-or OR-combination of classification according to the two metrics.

Depending on the application, either precision or recall may be more important than the other.

The Speech Command experiment shows that ODIN performs well for recurrent neural networks on a speech recognition task in addition to image classification.

We also demonstrate how it can be used in practice, by selecting classification threshold and evaluating our choice using precision and recall.

We also show how combinations of the different metrics available may be used to tune the precision/recall ratio.

Deep neural networks are powerful transformers that have shown great success in many applications.

But, in order to adopt deep learning in safety-critical applications it is crucial to understand when new observations do not match the data used during training.

To imitate linear latent variable models used in manufacturing process monitoring, we use a linear approximation of the hidden layer manifolds to measure distance to and within the manifold.

We compare our results to MC-dropout, a well established Bayesian approach, and consistently detect outliers post-training without imposing any constraints on either architecture or training procedure.

We demonstrate our method in two image classification experiments, with and without a pre-trained network, and a speech recognition example using a recurrent neural network.

By defining the limits of our neural networks' knowledge, ODIN contribute to safer use of deep learning.

APPENDIX A APPENDIX

Input FORMULA0 Softmax (

Input (

@highlight

An add-on method for deep learning to detect outliers during prediction-time