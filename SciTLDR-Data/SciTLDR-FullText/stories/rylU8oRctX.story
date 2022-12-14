Deep learning has become a widely used tool in many computational and classification problems.

Nevertheless obtaining and labeling data, which is needed for strong results, is often expensive or even not possible.

In this paper three different algorithmic approaches to deal with limited access to data are evaluated and compared to each other.

We show the drawbacks and benefits of each method.

One successful approach, especially in one- or few-shot learning tasks, is the use of external data during the classification task.

Another  successful approach, which achieves state of the art results in semi-supervised learning (SSL) benchmarks, is consistency regularization.

Especially virtual adversarial training (VAT) has shown strong results and will be investigated in this paper.

The aim of consistency regularization is to force the network not to change the output, when the input or the network itself is perturbed.

Generative adversarial networks (GANs) have also shown strong empirical results.

In many approaches the GAN architecture is used in order to create additional data and therefor to increase the generalization capability of the classification network.

Furthermore we consider the use of unlabeled data for further performance improvement.

The use of unlabeled data is investigated both for GANs and VAT.

Deep neural networks have shown great performance in a variety of tasks, like speech or image recognition.

However often extremely large datasets are necessary for achieving this.

In real world applications collecting data is often very expensive in terms of cost or time.

Furthermore collected data is often unbalanced or even incorrect labeled.

Hence performance achieved in academic papers is hard to match.

Recently different approaches tackled these problems and tried to achieve good performance, when otherwise fully supervised baselines failed to do so.

One approach to learn from very few examples, the so called few-shot learning task, consists of giving a collection of inputs and their corresponding similarities instead of input-label pairs.

This approach was thoroughly investigated in BID9 , BID33 , BID28 and gave impressive results tested on the Omniglot dataset BID12 ).

In essence a task specific similarity measure is learned, that embeds the inputs before comparison.

Furthermore semi-supervised learning (SSL) achieved strong results in image classification tasks.

In SSL a labeled set of input-target pairs (x, y) ??? D L and additionally an unlabeled set of inputs x ??? D U L is given.

Generally spoken the use of D U L shall provide additional information about the structure of the data.

Generative models can be used to create additional labeled or unlabeled samples and leverage information from these samples BID26 , BID18 ).

Furthermore in BID2 it is argued, that GAN-based semi-supervised frameworks perform best, when the generated images are of poor quality.

Using these badly generated images a classifier with better generalization capability is obtained.

On the other side uses generative models in order to learn feature representations, instead of generating additional data.

Another approach in order to deal with limited data is consistency regularization.

The main point of consistency regularization is, that the output of the network shall not change, when the input or the network itself is perturbed.

These perturbations may also result in inputs, which are not realistic anymore.

This way a smooth manifold is found on which the data lies.

Different approaches to consistency regularization can be found in BID15 , BID23 , BID11 , and BID32 .The aim of this paper is to investigate how different approaches behave compared to each other.

Therefore a specific image and sound recognition task is created with varying amount of labeled data.

Beyond that it is further explored how different amounts of unlabeled data support the tasks, whilst also varying the size of labeled data.

The possible accuracy improvement by labeled and unlabeled examples is compared to each other.

Since there is a correlation between category mismatch of unlabeled data and labeled data BID20 ) reported, we investigate how this correlation behaves for different approaches and datasets.

When dealing with little data, transfer learning BID34 , BID0 ) offers for many use cases a good method.

Transfer learning relies on transferring knowledge from a base model, which was trained on a similar problem, to another problem.

The weights from the base model, which was trained on a seperate big dataset, are then used as initializing parameters for the target model.

The weights of the target model are afterwards fine-tuned.

Whilst often yielding good results, nevertheless a similar dataset for the training of the base model is necessary.

Many problems are too specific and similar datasets are not available.

In BID15 transfer learning achieves better results than any compared consistency regularization method, when transferring from ImageNet BID3 ) to CIFAR-10 (Krizhevsky (2009) ).

On contrast, no convincing results could be achieved when transferring from ImageNet to SVHN BID16 , although the task itself remains a computer vision problem.

Therefore the generalization of this approach is somehow limited.

In order to increase the generalization of this work transfer learning is not investigated.

Instead this paper focuses on generative models, consistency regularization, and the usage of external data during the classification of new samples.

Since there exist several algorithms for each of these approaches, only one representative algorithm for each of the three approaches is picked and compared against each other.

The usage of external data after training during the classification task is a common technique used in few shot learning problems.

Instead of input-label pairs, the network is trained with a collection of inputs and their similarities.

Due to its simplicity and good performance the approach by BID9 , which is inspired by Bromley et al. (2014) , is used in this paper.

BID9 uses a convolutional siamese neural network, which basically learns an embedding of the inputs.

The same convolutional part of the network is used for two inputs x 1 and x 2 .

After the convolution each input is flattened into a vector.

Afterwards the L 1 distance between the two embeddings is computed and fed into a fully-connected layer, which outputs a similarity between [0, 1].In order to classify a test image x into one of K categories, a support set {x k } K k=1 with examples for each category is used.

The input x is compared to each element in the support set and the category corresponding to the maximum similarity is returned.

When there are more examples per class the query can be repeated several times, such that the network returns the class with the highest average similarity.

Using this approach is advantageous, when the number of categories is high or not known at all.

On the downside the prediction of the category depends on a support set and furthermore the computational effort of predicting a category increases with O(K), since a comparison has to be made for each category.

Consistency regularization relies on increasing the robustness of a network against tiny perturbations of the input or the network.

For perturbations of the input d (f (x; ??), f (x; ??)) shall be minimized, whereas d is a distance measurement like euclidean distance or Kullback-Leibler divergence andx is the perturbed input.

It is possible to sample x from both D L and D U L .An empirical investigation BID20 has shown, that many consistency regularization methods, like mean teacher BID32 ), ??-model BID23 , BID11 ), and virtual adversarial training (VAT) BID15 are quite hard to compare, since the results may rely on many parameters (network, task, etc.).

Nevertheless VAT is chosen in this work, since it achieves convincing results on many tasks.

VAT is a training method, which is greatly inspired by adversarial training BID5 ).

The perturbation r adv of the input x can be computed as DISPLAYFORM0 ,where ?? and are hyperparameters, which have to be tuned for each task.

After the perturbation was added to x consistency regularization is applied.

The distance between the clean (not perturbed) prediction and perturbed prediction d(f (x, ??), f (x + r adv , ??)) shall be minimized.

In order to reduce the distance the gradients are just backpropagated through f (x+r adv ).

Combining VAT with entropy minimization BID6 it is possible to further increase the performance BID15 .

For entropy minimization an additional loss term is computed as: DISPLAYFORM1 and added to the overall loss.

This way the network is forced to make more confident predictions regardless of the input.

Generative models are commonly used for increasing the accuracy or robustness of models in a semior unsupervised manner , BID35 BID29 , BID18 , ).A popular approach is the use of generative adversarial neural networks (GANs), introduced by BID4 .

The goal of a GAN is to train a generator network G, wich produces realistic samples by transforming a noise vector z as x f ake = G(z, ??), and a discriminator network D, which has to distinguish between real samples x real ??? p Data and fake samples x f ake ??? G.In this paper the training method defined in BID26 is used.

Using this approach the output of D consists of K + 1 categories, whereas K is the number of categories the classifier shall be actually trained on.

One additional extra category is added for samples generated by D. Since the output of D is over-parameterized the logit output l K+1 , which represents the fake category, is permanently fixed to 0 after training.

The loss function consists of two parts L supervised and L unsupervised , which can be computed as: DISPLAYFORM0 DISPLAYFORM1 L supervised represents the standard classification loss, i.e. negative log probability.

L unsupervised itself again consists of two parts, the first part forces the network to output a low probability of fake category for inputs x ??? p data and corresponding a high probability for inputs x ??? G. Since the the category y is not used in L unsupervised , the input x can be sampled from both D L and D U L .

In order to further improve the performance feature matching is used, as described in BID26 .

Three different experiments are conducted in this paper using the MNIST BID13 ) and UrbanSound8k BID24 dataset.

The UrbanSound8k dataset consists of 8732 sound clips with a maximum duration of 4 s. Each sound clip represents a different urban noise class like drilling, engine, jackhammer, etc.

Before using the sound files for training a neural network, they are prepared in a similar manner to BID25 , in essence each sound clip is transferred to a log-scaled mel-spectrogram with 128 components covering the frequency range between 0-22050 Hz.

The window size is chosen to be 23 ms and hop size of the same duration.

Sound snippets with shorter duration as 4 s are repeated and concatenated until a duration of 4 s is reached.

The preprocessing is done using librosa BID14 ).

For training and evaluation purposes a random snippet with a length of 3 s is selected, resulting in an input size of 128 ?? 128.In the first experiment no external unlabeled data is used.

Instead, the amount of labeled data in each category is varied and the three methods are compared to each other.

In the second experiment the amount of labeled and unlabeled data is varied, in order to explore how unlabeled data can compensate labeled data.

The last experiment considers class distribution mismatch while the amount of labeled and unlabeled data is fixed.

In the second and third experiment only two methods are compared, since only generative models and consistency regularization allow the use of external unlabeled data.

All methods are compared to a standard model.

When using the MNIST dataset the standard model consists of three convolutional layers, followed by two fully-connected layers.

For the UrbanSound8k dataset the standard model consists of four convolutional layers, followed by three fully-connected layers.

ReLU nonlinearities were used in all hidden layers.

The training was done by using the Adam optimizer BID7 ).

Furthermore batch normalization BID19 ), dropout BID30 ), and max-pooling was used between convolutional layers.

For further increasing the generalization capability L 2 regularization (Ng FORMULA0 ) is used.

The models, representing the three different approaches, have the same computational power as the standard model, in essence three/ four convolutional layers and two/ three fully connected layers.

The number of hidden dimensions and other per layer hyperparameters (e.g. stride, padding) is kept equal to the corresponding standard models.

The hyperparameters were tuned manually on the training dataset by performing gridsearch and picking the most promising results.

Whereas the L 2 and batchnorm coefficients, as well as dropout rate are shared across all models for each dataset.

The test accuracy was calculated in all experiments with a separate test dataset, which contains 500 samples per category for the MNIST dataset and, respectively, 200 samples per category for the UrbanSound8k dataset.

Train and test set have no overlap.

All experiments were conducted using the PyTorch framework BID21 ).

In this experiment the amount of labeled data is varied.

Furthermore there is not used any unlabeled external data.

For each amount of labeled data and training approach (i.e. baseline, VAT Miyato et al. (2018) , GAN Salimans et al. (2016) , and siamese neural network BID9 ) the training procedure was repeated eight times.

Afterwards the mean accuracies and standard deviations have been calculated.

Figure 1 shows the results obtained in this experiment for the MNIST dataset.

The amount of labeled data per category was varied on a logarithmic scale in the range between [0, 200] with 31 steps.

Using 200 labeled samples per category the baseline network is able to reach about 95 % accuracy.

With just one labeled sample per class the baseline networks reaches already around 35 %, which is a already good compared to 10 %, when random guessing.

Generally all three methods are consistent with the literature, such that they are superior over baseline in the low data regime (1-10 samples per category).

Using a siamese neural network the accuracy can be significantly improved in the low data regime.

With just one labeled sample the siamese architecture already reaches around 45 %.

When using a dataset with more categories, like Omniglot, the advantage of using siamese networks should be even higher in the low data regime.

The performance of this approach becomes worse compared to the baseline model, when using more than 10 labeled examples per class.

VAT has a higher benefit compared to GAN for up to 20 labeled samples per category.

For higher numbers of labeled samples both methods show only little (0-2 %) improvement over the baseline results.

Similar results are obtained on the UrbanSound8k dataset (figure 2).

As for the experiment on the MNIST dataset the amount of labeled data was varied on a logarithmic scale in the range between [0, 200], but with 6 steps instead of 31, since the computational effort was much higher.

The siamese network yields a large improvement when there is only one labeled sample, but fast returns worse results than the baseline network.

On contrast the usage of VAT or GAN comes with a benefit in terms of accuracy for higher amounts of labeled data.

Nevertheless these both methods are either not able to further improve the accuracy for high amounts of labeled data (more than 100).

Furthermore the accuracy even declines compared to baseline for 200 labeled samples.

The observation, that adversarial training can decrease accuracy, is inline with literature BID27 , BID31 ), where it was shown that in high data regimes there may be a trade-off between accuracy and robustness.

Whereas in some cases adversarial training can improve accuracy in the low data regime.

Both methods show a significant increase in terms of accuracy when the amount of labeled data is low and corresponding the amount of unlabeled data is high.

When the amount of labeled data increases the amount of necessary unlabeled data also increases in order to achieve the same accuracy improvements.

VAT achieves better results with less unlabeled data compared to GAN, when there is little labeled data (??? 2-10 examples per category).

On contrast GANs achieve better results when there is a moderate amount of labeled examples (??? 10-50 examples per category) and also many unlabeled examples.

When the amount labeled examples is high both methods behave approximately equal.

The results for the UrbanSound8k dataset can be seen in FIG2 .

Overall similar results as for the MNIST dataset are achieved, in terms of having high benefits, when the amount of labeled data is low and concurrently the amounts of unlabeled data is high.

Nevertheless the total improvement is lower and for high amounts of labeled data, more unlabeled data is necessary in order to get an improvement at all.

For the VAT the amount of unlabeled data need to have similar magnitudes as the amount of labeled data in order to get an improvement at all.

Further the same observation as before can be made, that VAT achieves better results with less unlabeled data, when there is little labeled data

In this experiment the possibility of adding additional unlabeled examples, which do not correspond to the target labels (mismatched samples), is investigated.

This experiment was done for VAT in BID20 .

In this work the investigation is extended in such a way that the results for VAT are compared to GAN.

Furthermore not only the extend of mismatch, but also the influence of the amount of additional unlabeled examples is investigated.

Both datasets (MNIST and UrbanSound8k) consist of 10 categories with label values [0, 9] and the aim is to train a neural network, which is able to classify inputs corresponding to categories [0, 6] , hence the network has six outputs.

Mismatched examples belong to categories [7, 9] .

The number of labeled examples per category is fixed to be five.

Having five labeled samples it can be seen in FIG1 , that the accuracy improvement shows a strong dependency on the amount of unlabeled samples.

The total number of unlabeled examples is varied between {30, 120, 600}. Furthermore the mismatch for each number of unlabeled examples is varied between 0-100 % using a 10 % increment, e.g. when the amount of unlabeled examples is set to be 120 and the mismatch is 70 % the unlabeled examples consist of 84 examples belonging to categories [0, 6] and 36 examples belonging to categories [7, 9] .

The distribution across the categories in the six matched and four remaining mismatched classes is kept approximately equal, with a maximum difference of ??1.

For each amount of mismatch and method eight neural networks have been trained.

Afterwards their average accuracies and standard deviations have been calculated.

For baseline results also eight neural networks have been trained and their average accuracy and standard deviation computed.

Since the number of classes is reduced to 6 the accuracy, when compared to the previous experiments, is higher with the same amount of labeled data.

FIG4 shows the results of this experiment for the MNIST dataset.

Overall the accuracy decreases for both methods when the class mismatch increases, which is in line with literature BID20 ).

As in the experiments before, the GAN method shows little to no accuracy improvement, when the additional amount of unlabeled data is low (30 unlabeled samples).

For 120 and respectively 600 additional unlabeled elements both methods show an approximate equal maximal accuracy improvement, when there is no class mismatch.

When the class mismatch is very high (80-100 %) using VAT results in worse performance than baseline results.

Using GANs the performance is in worst case at the same level as baseline performance.

GAN shows a linear correlation between accuracy and class mismatch.

On contrast VAT shows a parabolic trend.

Overall increasing the amount of unlabeled data seems to increase the robustness towards class mismatch.

All in all both methods show an accuracy improvement even for high amounts (> 50 %) of class mismatch.

Whereas VAT performs better, when the amount of mismatch is low.

FIG5 shows the results obtained with the UrbanSound8k dataset.

Overall there seems to be no, or only little correlation between class mismatch and accuracy.

Only for the GAN, when using 30 or 120 unlabeled samples, a small correlation can be observed.

This is a surprising observation, since in the previous experiment and in BID20 a decrease in terms of accuracy is reported for increasing class mismatch.

In essence it can be stated, that adding samples, which do not necessarily belong to the target classes, can improve the overall accuracy.

This is especially interesting for training classifiers on hard to obtain or rare samples (rare disease, etc.).

Nevertheless it has to be checked whether adding this samples hurts the performance or not.

Furthermore the correlation between more unlabeled data and accuracy can be observed, as in the previous experiments.

In this paper three methods for dealing with little data have been compared to each other.

When the amount of labeled data is very little and no unlabeled data is available, siamese neural networks offer the best alternative in order to achieve good results in terms of accuracy.

Furthermore when there is additional unlabeled data available using GANs or VAT offer a good option.

VAT outperforms GAN when the amount of data is low.

On contrast GANs should be preferred for moderate or high amounts of data.

Nevertheless both methods must be tested for any individual use case, since the behavior of these methods may change for different datasets.

Surprising results have been obtained on the class mismatch experiment.

It was observed that adding samples, which do not belong to the target classes, not necessarily reduce the accuracy.

Whether adding such samples improves or reduce the accuracy, may heavily depend on how closely these samples/ classes are related to the target samples/ classes.

An interesting questions remains whether datasets which perform good in transfer learning tasks (e.g. transferring from ImageNet to CIFAR-10) also may be suitable for such semi-supervised learning tasks.

Furthermore any combinations of three examined methods can bear interesting results, e.g.

VAT could be applied to the discriminator in the GAN framework.

Also a combination of GAN and siamese neural networks could be useful, in this case the siamese neural network would have two outputs, one for the source and one for the similarity.

@highlight

Comparison of siamese neural networks, GANs, and VAT for few shot learning. 