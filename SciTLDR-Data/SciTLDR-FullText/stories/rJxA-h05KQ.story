We present a new method for uncertainty estimation and out-of-distribution detection in neural networks with softmax output.

We extend softmax layer with an additional constant input.

The corresponding additional output is able to represent the uncertainty of the network.

The proposed method requires neither additional parameters nor multiple forward passes nor input preprocessing nor out-of-distribution datasets.

We show that our method performs comparably to more computationally expensive methods and outperforms baselines on our experiments from image recognition and sentiment analysis domains.

The applications of computational learning systems might cause intrusive effects if we assume that predictions are always as accurate as during the experimental phase.

Examples include misclassified traffic signs BID5 and an image tagger that classified two African Americans as gorillas BID3 .

This is often caused by overconfidence of models that has been observed in the case of deep neural networks BID8 .

Such malfunctions can be prevented if we estimate correctly the uncertainty of the machine learning system.

Beside AI safety, uncertainty is useful in the active learning setting in which data collection process is expensive or time consuming BID12 BID32 .While uncertainty estimation in neural networks is an active field of research, the current methods are rarely adopted.

It is desirable to develop a method that does not create an additional computational overhead.

Such a method could be used in environments that focus on quick training and/or inference.

If such a method is simple, the ease of implementation should encourage practitioners to develop danger-aware systems in their work.

We suggest a method that measures uncertainty of the neural networks with a softmax output layer.

We replace this layer with Inhibited Softmax layer BID33 , and we show that it can be used to express the uncertainty of the model.

In our experiments the method outperforms baselines and performs comparably with more computationally expensive methods on the out-of-distribution detection task.

We contribute with:??? The mathematical explanation why the additional Inhibited Softmax output can be interpreted as an uncertainty measure.??? The additions to the Inhibited Softmax that improve its uncertainty approximation properties.??? The benchmarks comparing Inhibited Softmax, baseline and contemporary methods for measuring uncertainty in neural networks.

The modern Bayesian Neural Networks BID0 BID11 BID25 BID27 BID37 BID9 Zhang et al., 2018; BID15 aim to confront this issue by inferring distribution over the models' weights.

This approach has been inspired by Bayesian approaches suggested as early as the nineties BID2 BID29 .

A very popular regularisation mean -dropout -also can be a source of approximate Bayesian inference BID7 .

Such technique, called Monte Carlo dropout BID6 , belongs to the Bayesian Neural Networks class and has been since used in the real-life scenarios (e.g. Leibig et al., 2017) .

In the Bayesian Neural Networks the uncertainty is modelled by computing the predictive entropy or mutual information over the probabilities coming from stochastic predictions BID34 .Other methods to measure uncertainty of neural networks include a non-Bayesian ensemble BID19 , a student network that approximates the Monte Carlo posterior predictive distribution BID16 , modelling Markov chain Monte Carlo samples with a GAN BID38 , Monte Carlo Batch Normalization BID35 and the nearest neighbour analysis of penultimate layer embedding BID28 .The concept of uncertainty is not always considered as a homogeneous whole.

Some of the authors distinguish two types of uncertainties that influence predictions of machine learning models BID14 : epistemic uncertainty and aleatoric uncertainty.

Epistemic uncertainty represents the lack of knowledge about the source probability distribution of the data.

This uncertainty can be reduced by increasing the size of the training data.

Aleatoric uncertainty arises from homoscedastic, heteroscedastic and label noises and cannot be reduced by the model.

We will follow another source BID27 ) that defines the third type: distributional uncertainty.

It appears when the test distribution differs from the training distribution, i.e. when new observations have different nature then the ones the model was trained on.

A popular benchmark for assessing the ability of the models to capture the distributional uncertainty is distinguishing the original test set from out-of-distribution dataset BID10 .

There are works that focus only on this type of uncertainty BID22 .

ODIN BID24 does not require changing already existing network and relies on gradient-based input preprocessing.

Another work BID4 ) is close to the functionality of our method, as it only adds a single densely connected layer and uses a single forward pass for a sample.

Bayesian neural networks are more computationally demanding as they usually require multiple stochastic passes and/or additional parameters to capture the priors specification.

To the best of our knowledge, our method is the first that improves upon the baseline, and meets all the following criteria:??? No additional learnable parameters required.??? Only single forward pass needed.??? No additional out-of-distribution or adversarial observations required.??? No input preprocessing.

The technique we use, Inhibited Softmax, has been successfully used for the prediction of background class in the task of extraction the objects out of aerial imagery BID33 .

The original work does not mention other possible applications of this softmax modification.

In this section we will define the Inhibited Softmax function.

We will provide mathematical rationale on why it can provide uncertainty estimation when used as the output function of a machine learning model.

Later we will present several adjustments which we have made to the model architecture when applying Inhibited Softmax to a multilayer neural network.

Inhibited Softmax function IS c is given by: Let x ??? R n and c ??? R, then IS c is a function which maps R n to R n .

The i-th output is equal to: DISPLAYFORM0 Following equation holds: DISPLAYFORM1 where: DISPLAYFORM2 and S(x) is the standard softmax function applied to vector x. We will later refer to P c (x) as "certainty factor".

Now let's assume that IS c is the output of a multiclass classification model trained with the crossentropy loss function l IS .

Assuming that the true class of a given example is equal to t the loss is equal to: DISPLAYFORM3 where l S is the cross-entropy loss function for a model with a standard softmax output.

As one may see -the optimisation process both minimises classification error (given by l S ) and maximises the certainty factor P c (x) for all training examples.

This is the intuition that explains why Inhibited Softmax serves as an uncertainty estimator -as P c (x) is maximised only for cases from training distribution.

If P c estimates the certainty of the model, in order to provide a valid uncertainty score we will introduce the following function: DISPLAYFORM4 which is minimised during the optimisation process.

It is worth to mention that it might be interpreted as an artificial softmax output from the additional channel.

Although P u is minimised during the optimisation process we would like to ensure that its low values are obtained solely because of the training process and neither because of the trivial solutions nor accidental network structure.

Because of that we applied the following network adjustments:??? removing bias terms from the inhibited softmax layer.

This was done in order to prevent the network from minimising P u (x) that can be achieved by increasing the values of bias terms which are independent of data.??? changing the activation function to a kernel function in the penultimate layer of the network.

The main aim of this adjustment was to make activations of the layer noticeably greater from 0 only for a narrow, learnable region in the input space of the penultimate network layer.

Therefore, the activations of that layer corresponding to out-of-domain examples are likely to be close to 0, which, combined with the lack of bias term, results in vanishing input to IS.In order to combat the overconfidence of the network we:??? add the activity regularisation.

Standard softmax classification is invariant to translation along the all-ones vector.

On the other hand, ??? log P c is a decreasing function of x i .

As l is is a sum of standard classification error and ??? log P c increasing all values of x i by a constant decreases the loss, which causes gradient optimisation methods to increase x i boundlessly.

In order to address this issue we introduced the following regularisation method: DISPLAYFORM0 where the gradient of the additional term is parallel to the all-ones vector and thus does not affect the standard softmax classification loss.??? apply l 2 regularisation to the weights of the output layer.

It indirectly limits the values of x i , as we use a bounded activation function.

These adjustments significantly increased the certainty estimation properties of Inhibited Softmax.

The dependency between performance and applying these changes to model architecture is presented in Appendix 1.

We have compared various ways of estimating uncertainty in neural networks (hereinafter referred to as "methods").

For the benchmarks we implement these methods on top of the same base neural network.

We use following experiments to check their quality:??? Out-of-distribution (OOD) examples detection -following BID10 we use ROC AUC and average precision (AP) metrics to check the classifier's ability to distinguish between the original test set and a dataset coming from another probability distribution.

This experiments show whether the method measures well the distributional uncertainty on a small sample of out-of-distribution datasets.??? Predictive performance experiment -given a dataset, we split it into train, test and validation sets.

We report accuracy and negative log loss on the test set.

Any method should not deteriorate predictive performance of the network.??? Wrong prediction detection -we expect that the more confident the model is, the more accurate its predictions on in-distribution dataset should be.

In this experiment the ground truth labels are used to construct two classes after the prediction on the test dataset is performed.

The classes represent the correctness of the classifier prediction.

Then, the uncertainty measure is used to compute TPRs and FPRs.

We report ROC AUC scores on this setting.

This experiment shows whether the method measures well the combination of epistemic and aleatoric uncertainty on a small sample of datasets.

In this experiment we do not report average precision score, as it would be distorted by different levels of misclassification in the predictions.

Uncertainty measure Abbreviation Inhibited Softmax probability of the artificial softmax output DISPLAYFORM0 entropy of the probabilities BASEE Monte Carlo Dropout BID7 predictive entropy of the probabilities from 50 stochastic forward passes MCD Bayes By Backprop with a Gaussian prior BID0 predictive entropy of the probabilities from 10 stochastic forward passes BBP Deep Ensembles without adversarial training BID19 predictive entropy of the probabilities from 5 base neural networks TAB0 shows the methods and respective uncertainty measures that will be benchmarked 1 .

We establish two baselines.

Both of them work on the unmodified base neural network, but uncertainty is measured in different ways, using either the maximum of probabilities over classes or entropy of probabilities.

The method we suggest to use is referred as IS.

We have chosen these methods as they have been already used for benchmarking (e.g. BID25 , and they are well-known in the Bayesian Neural Network community.

In the case of Inhibited Softmax we set l2 penalty to 0.01, activity regularisation to 10 ???6 , c to 1 and we use rescaled Cauchy distribution's PDF (f (x) = 1 1+x 2 ).

The datasets 2 and the respective base neural networks we have chosen for the experiments are reported in TAB1 .The base network for CIFAR-10 consists of 3 2D convolutional layers with 2D batch norm and 0.25 dropout.

The convolving filter size was 3.

Each convolutional layer was followed by 2D maximum pooling over 3x3 neurons with stride 2.

The number of filters in the consecutive layers are 80, 160 and 240.

Then there are 3 fully-connected layers.

After the first fully-connected layer we apply 0.25 dropout.

The number of neurons in the consecutive dense layers are 200, 100, 10.In the experiments we report averages over three training and prediction procedures on the same training-test splits.

In computer vision OOD tasks, Inhibited Softmax improves upon baselines with an exception of the task of discriminating MNIST from black and white CIFAR-10 TAB3 .

Our method still achieves very high detection performance (0.996 ROC AUC and 0.999 AP).

This dataset is the least similar to MNIST.

In contrast to other tested datasets against the digit recognition networks, various shades of gray dominate the images.

IS is better than BASE on NOTMNIST (0.977 ROC AUC vs 0.958) and Omniglot (0.97 ROC AUC vs 0.956).

IS' ROC AUC performance on MNIST/NOTMNIST and CIFAR-10/SVHN is similar to MCD (resp.

0.977 vs 0.974 and 0.923 vs 0.927).

IS achieves the best result on CIFAR-10/LFW-A task and all the methods vastly outperform the baselines.

Inhibited Softmax improves upon other methods on the sentiment analysis task.

Especially large improvement can be observed on the test against the Movie Reviews dataset.

For example, the ROC AUC of IS (0.875) is much greater than ROC AUC of MCD (0.836).

Methods other than IS are not much better than the baseline (BBP's 0.845 ROC AUC), sometimes being worse (DE's 0.835 ROC AUC).

IS is also the best on the test against Reuters-21578 and Customer reviews (resp.

0.822 and 0.731).

Two baselines achieve the same results on sentiment analysis experiment as there is no difference in ranking of the examples between the chosen uncertainty measures.

We do not corroborate the results from the baseline publication BID10 .

We discovered that in that paper the out-of-distribution samples for Movie Reviews were constructed by taking single lines from the dataset file, while the reviews span over few lines.

Our results show that the detection is a tougher task when full reviews are used (BASE achieves 0.837 ROC AUC vs 0.94 ROC AUC BID10 ).To understand where the improvements of IS in the sentiment OOD tasks come from, we trained the base network with the same regularisation consisting of l2 penalty on weights and the activity regularizer.

Such an improved baseline achieved 0.853 on IMDB/Movie Reviews and 0.838 on IMDB/Reuters-21578.

Both results are better than all the methods but IS, with the latter improving also upon IS.

On the other hand, this enhanced baseline did not improve on IMDB/Customer Reviews achieving 0.712 ROC AUC.In our experiments Inhibited Softmax does not deteriorate the predictive performance of the neural network TAB4 .

Its accuracy was similar to the baselines on every task, for example on IMDB dataset the accuracy is 0.04% lower and on CIFAR-10 0.19% higher.

Ensembling the networks gives the best predictive performance.

We observed that text models perform very well on Movie Reviews dataset.

Despite coming from a different probability distribution the latter dataset contains strong sentiment retrieved by the networks for the prediction of the correct label.

Wrong prediction detection results (Table 5) show that IS is the only method that is able to detect misclassified observations better than a random classifier (0.687 ROC AUC) on the sentiment task.

In practice, the overlap of the correctly detected out-of-distribution observations between Inhibited Softmax and Bayesian methods is surprisingly large.

To demonstrate it, we compare Monte Carlo dropout and our method on an experiment from BID34 .

We train a fully connected variational autoencoder (VAE) on the MNIST dataset.

Then, we create a grid in the latent space and for each point we generate a sample.

We plot the uncertainty estimation of the methods on generated samples from these points together with labelled latent encoding of the test samples FIG0 ).

Both methods are unable to detect out of distribution samples generated from the bottom left corner of the 2D latent space.

Another example for similarity is that both of the methods do not estimate high uncertainty in area where blue and purple classes intersect in the latent space.

This leads to a hypothesis that there exist samples that are tougher to detect by uncertainty measures for any recently proposed method.

Similarly to the ideas from adversarial attacks field, it might be worth to investigate how to construct such samples.

We believe it might be a way to improve uncertainty sampling performance.

We notice that working on following aspects can enhance the uncertainty estimation:??? Developing an analogous to IS method for regression.??? Limiting the number of required hyperparameters for Inhibited Softmax.??? Expanding the method to hidden layers.

This is especially promising as the Inhibited Softmax performs better than other methods on a shallow network in our sentiment analysis experiment.

On deeper networks IS has not have yet such advantage and it might be possible to outperform other methods.

Although we showed by experiments that the architecture adjustments applied to the network architecture are beneficial, we are still lacking the full and sound mathematical explanation of their influence on model behaviour.

Such analysis could result in both better procedure for setting Inhibited Softmax hyperparameters as well as new adjustments to the network structure.

We presented a new method for uncertainty estimation -Inhibited Softmax.

The method can be easily applied to various multilayer neural network architectures and does not require additional parameters, multiple stochastic forward passes or OOD examples.

The results show that the method outperforms baseline and performs comparably to the other methods.

The method does not deteriorate predictive performance of the classifier.

The predictive performance from IMDB/Movie Reviews experiment suggests that even if the observation comes from another probability distribution and the uncertainty measure is able to detect it, the network can still serve as a useful classifier.

The improvement of the baseline on the sentiment task after adding suggested regularisation indicates it might be worth to apply such measures to other uncertainty estimation methods.

We show the performance of our methods in the experiments on CIFAR-10 and MNIST datasets if the hyperparameters are changed FIG1 .

The results are averages over three runs of experiments.

Without l2 penalty or with too strong a penalty (e.g. 0.1) the performance in terms of accuracy on CIFAR-10, wrong prediction detection on CIFAR-10 and out-of-distribution detection on CIFAR-10/SVHN deteriorates.

Moreover, without l2 penalty, the network performs worse on OOD detection on MNIST/NOTMNIST and MNIST/CIFAR-10 tasks.

Similarly, the activity regularizer penalty is important.

The networks without it performed worse on all the checked tasks with an exception of OOD detection on CIFAR-10/SVHN.

With too much of the regularization, the networks are unable to fit the data well.

It results in a sharp drop in results of all experiments on CIFAR-10.

We show also that it is possible to replace the rescaled Cauchy PDF function with another kernel function.

Here, we show a comparison with rescaled Gaussian PDF (exp DISPLAYFORM0 2 ) and a custom nonlinear function: DISPLAYFORM1 Still, non-kernel activation functions like ReLU do not perform well.

We check the performance with changed l2 penalty (left), changed activation function (middle) and changed activity regularization penalty (right).

Omniglot consists of black letters on white background.

We negated the images so that they resemble more the images from MNIST.

Without the negation, all the methods performed very well (between 0.999 and 1 in ROC AUC) on the out-of-distribution detection task.

In the sentiment analysis task, before feeding the data to the networks we preprocessed it by removing stopwords and words that did not occur in the pretrained embedding.

We use a pretrained embedding in order to model vocabulary that exists in the OOD sets and was not present in the in-distribution dataset.

Regarding the baseline publication BID10 : we were able to corroborate the results on IMDB/Movie Reviews experiment when we split the observations from Movie Reviews into single lines and use the same randomly initialized embeddings.

The model was trained on full reviews from IMDB.

We argue that in such setting the use of average pooling after the embedding invalidates the experiment.

The input is padded with zeros to 400 words.

Now, if the sentence is very short, say 10 words, the true average of the embed words will be diminished by all the zeros after the sentence.

Thus, the uncertainty estimation method needs only to correctly work in a very narrow region centred at zero in order to achieve high scores in the experiment.

For the state-of-the art methods we compared with we made following choices:??? Deep Ensembles -we skipped adversarial training, as adversarial training is a way to improve performance of any of the methods used in the paper.

We use an ensemble of 5 base networks.??? Monte Carlo Dropout -for MNIST we use dropout probability 0.25 on all but last layers, 0.5 on the only trainable layer in the sentiment experiment, and on CIFAR-10 network 0.25 only on the last but one layer.

In larger networks setting dropout on many layers required greater number of epochs to achieve top performance.

We run 50 forward passes for variational prediction.??? Bayes By Backprop -we observed that there is a trade-off between accuracy and OOD detection performance that depends on the initialisation of the variance.

We chose initialisation that led to the best combination of accuracy and OOD detection performance in our view.

We run 10 forward passes for variational prediction.

We followed the original publications when possible.

For example, the number of networks in DE and number of inferences in BBP and MCD are taken from the original descriptions of the algorithms.

In the visualisation section of the paper the uncertainties were normalised so that the predictive entropy and IS' probabilities could be visually compared.

The normalisation for a method was performed by ranking the uncertainties and splitting them into 400 equal bins.

Then, the bins are plotted.

White colour represents the bin with the most uncertainty, the black -with the least.

For better understanding of the latent space we visualise the images decoded from the grid from the latent space FIG2 ).

@highlight

Uncertainty estimation in a single forward pass without additional learnable parameters.

@highlight

A new method for computing output uncertainty estimates in DNNs for classification problems that matches state-of-the-art methods for uncertainty estimation and outperforms them in out-of-distribution detection tasks.

@highlight

The authors present inhibited softmax, a modification of the softmax through adding a constant activation which provides a measure for uncertainty. 