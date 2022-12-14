In this paper, we turn our attention to the interworking between the activation functions and the batch normalization, which is a virtually mandatory technique to train deep networks currently.

We propose the activation function Displaced Rectifier Linear Unit (DReLU) by conjecturing that extending the identity function of ReLU to the third quadrant enhances compatibility with batch normalization.

Moreover, we used statistical tests to compare the impact of using distinct activation functions (ReLU, LReLU, PReLU, ELU, and DReLU) on the learning speed and test accuracy performance of standardized VGG and Residual Networks state-of-the-art models.

These convolutional neural networks were trained on CIFAR-100 and CIFAR-10, the most commonly used deep learning computer vision datasets.

The results showed DReLU speeded up learning in all models and datasets.

Besides, statistical significant performance assessments (p<0.05) showed DReLU enhanced the test accuracy presented by ReLU in all scenarios.

Furthermore, DReLU showed better test accuracy than any other tested activation function in all experiments with one exception, in which case it presented the second best performance.

Therefore, this work demonstrates that it is possible to increase performance replacing ReLU by an enhanced activation function.

The recent advances in deep learning research have produced more accurate image, speech, and language recognition systems and generated new state-of-the-art machine learning applications in a broad range of areas such as mathematics, physics, healthcare, genomics, financing, business, agriculture, etc.

Although advances have been made, accuracy performance enhancements have usually demanded considerably deeper or more complex models, which tend to increase the required computational resources (processing time and memory usage).Instead of increasing deep models depth or complexity, a less computational expensive alternative approach to enhance deep learning performance across-the-board is to design more efficient activation functions.

Even if computational resources are no issue, to employ enhanced activation functions nevertheless contributes to speeding up learning and achieving higher accuracy.

Indeed, by allowing the training of deep neural networks, the discovery of Rectified Linear Units (ReLU) BID19 BID4 BID13 was one of the main factors that contributed to deep learning advent.

ReLU allowed achieving higher accuracy in less time by avoiding the vanishing gradient problem BID9 .

Before ReLU, activation functions such as Sigmoid and Hyperbolic Tangent were unable to train deep neural networks because of the absence of the identity function for positive input.

However, ReLU presents drawbacks.

For example, some researchers argument that zero slope avoids learning for negative values BID18 BID6 .

Therefore, other activation functions like Leaky Rectifier Linear Unit (LReLU) BID18 , Parametric Rectifier Linear Unit (PReLU) BID6 and Exponential Linear Unit (ELU) were proposed (Appendix A).

Unfortunately, there is no consensus about how these proposed nonlinearities compare to ReLU, which therefore remains the most used activation function in deep learning.

Similar to activation functions, batch normalization BID11 currently plays a fundamental role in training deep architectures (Appendix B) .

This technique normalizes the inputs of each layer, which is equivalent to normalizing the outputs of the deep model previous layer.

However, before being used as inputs for the subsequent layer, the normalized data are typically fed into activation functions (nonlinearities), which necessarily skew the otherwise normalized distributions.

In fact, ReLU only produces non-negative activations, which is harmful to the previously normalized data.

The outputs mean values after ReLU are no longer zero, but rather necessarily positives.

Therefore, the ReLU skews the normalized distribution (Section 2).Aiming to mitigate the mentioned problem, we concentrate our attention on the interaction between activation functions and batch normalization.

We conjecture that nonlinearities that are more compatible with batch normalization present higher performance.

After that, considering that an identity transformation preserves any statistical distribution, we assume that to extend the identity function from the first quadrant to the third implies less damage to the normalization procedure.

Hence, we investigate and propose the activation function Displaced Rectifier Linear Unit (DReLU), which partially prolongs the identity function beyond origin.

Hence, DReLU is essentially a ReLU diagonally displaced into the third quadrant.

Different from all other previous mentioned activation functions, the inflection of DReLU does not happen at the origin, but in the third quadrant.

Considering the widespread adoption and practical importance, we used Convolutional Neural Networks (CNN) BID16 BID13 in our experiments.

Moreover, as particular examples of CNN architectures, we used the previous ImageNet Large Scale Visual Recognition Competition (ILSVRC) winners Visual Geometry Group (VGG) BID22 and Residual Networks (ResNets) BID5 c) .

These architectures have distinctive designs and depth to promote generality to the conclusions of this work.

In this regard, we evaluated how replacing the activation function impacts the performance of well established and widely used standard state-of-the-art models.

Finally, we decided to employ the two most broadly used computer vision datasets by deep learning research community: CIFAR-100 BID14 CIFAR-10 (Krizhevsky, 2009) .In this systematic comparative study, performance assessments were carried out using statistical tests with a significance level of 5% (Appendix C.5).

At least ten executions of each of experiment were executed.

However, when the mentioned significance level was not achieved, ten additional runs were performed.

Consider x the input of a layer composed of a generic transformation Wx+b followed by a nonlinearity, for instance, ReLU.

After the addition of the batch normalization layer, the overall joint transformation performed by the block (composed layer) is given by: DISPLAYFORM0 For a moment, consider the intermediate activation y produced inside the block: DISPLAYFORM1 Without loss of generality, assume ?? = 1 and ?? = 0 BID11 .

Therefore, since y is the output of a batch normalization layer, we can rewrite unbiased estimators for the expected value and variance of any given dimension k as follows (Appendix B): DISPLAYFORM2 Consequently, we investigate the expected value and variance of the activations distribution produced by the combined layer.

Therefore, if z is the output of the block using a ReLU nonlinearity, it immediately follows that: DISPLAYFORM3 The application of ReLU removes all negative values from a distribution.

Hence, it necessarily produces positive mean as output.

Therefore, it can be written: DISPLAYFORM4 Considering that a distribution with??[?? (k) ] = 0 has to present negative values, replacing all negative activations with zeros makes the variance of z (k) necessarily lower than the variance of the original distribution?? (k) .

Consequently, we can rewrite: DISPLAYFORM5 Therefore, despite batch normalization, the activations after the whole block are not perfectly normalized since these outputs present neither zero mean nor unit variance.

Consequently, regardless of the presence of a batch normalization layer, after the ReLU, the inputs passed to the next composed layer have neither mean of zero nor variance of one that was the objective in the first place.

In this sense, ReLU skews an otherwise previous normalized output.

In other words, ReLU reduces the correction of the internal covariance shift promoted by the batch normalization layer.

Consequently, we conclude the ReLU bias shift effect is directly related to the drawback ReLU generates to the batch normalization.

Consequently, we propose DReLU, which is essentially a diagonally displaced ReLU.

It generalizes both ReLU and SReLU by allowing its inflection to move diagonally from the origin to any point of the form (?????, ?????).

If ?? = 0, DReLU becomes ReLU.

If ?? = 1, DReLU becomes SReLU.

Therefore, the slope zero component of the activation function provides negative activations, instead of null ones.

Unlike ReLU, in DReLU learning can happen for negative inputs since gradient is not necessarily zero.

The following equation defines DReLU: DISPLAYFORM6 DReLU can be regarded as a generalization of the Shifted Rectifier Linear Unit (SReLU) .

In fact, instead of always prolong the identity to the point (???1, ???1), in DReLU we established a hyperparameter ?? that defines the most appropriate point (?????, ?????) where the inflection should happen.

In this sense, SReLU is a particular case of DReLU where ?? = 1.

Considering that the experiments we performed to determine the DReLU hyperparameter contemplates ?? = 1 as a possible value for ??, it was not necessary to include SReLU in the present comparative study.

Indeed, our experiments showed that the addition of the parameter ?? allowed DReLU to significantly outperform SReLU in all of our hyperparameter definition experiments (Appendix C.6).

Considering that DReLU replaces ReLU in Eq. 1, the activations of the composed layer become: DISPLAYFORM7 Since DReLU extends the identity function into the third quadrant, it is no longer possible to conclude Eq. 5 is valid.

Therefore, the consequence presented in the mentioned equation is probably at least minimized.

In this case, we can conclude that??[z (k) ] DReLU is much probably near to zero than?? [z (k) ] ReLU .

Hence, we can rewrite: DISPLAYFORM8 Furthermore, DReLU exhibits a noise-robust deactivation state for very negative inputs, a feature not granted by LReLU and PReLU.

A noise-robust deactivation state is achieved by setting the slope zero for highly negative values of input .

Some authors argument that activation functions with this propriety improve learning .

Finally, DReLU is less computationally complex than LReLU, PReLU, and ELU.

In fact, since DReLU has the same shape of ReLU, it essentially has the same computational complexity.

In this comparative study, we define an experiment as the training of a deep model using a distinct activation function on a given dataset.

If not otherwise mentioned, we conducted ten executions of each experiment.

We define a scenario as the set of experiments regarding all activation functions on a specific dataset using a particular model.

In this regard, this paper presents the consolidated results of six scenarios (two datasets versus three models) that correspond to 30 experiments, which in turn represents a total of 320 executions (training of deep neural networks).

In two cases, we executed 20 instead of 10 runs of a given experiment to achieve the desired statistical significance.

We trained the models during 100 epochs since it was enough to the test accuracy to saturate.

At epochs 40 and 70, we evaluated the test accuracy of the partially trained models.

Therefore, we were able to assess how fast each model was learning to generalize based on the activation function used by the model.

This is important for compare the expected performance of the activation functions in applications where the models need to provide high test accuracy training only a few tens of epochs.

Since the training time of an epoch shows no significant difference among the activation functions, we said that the nonlinearity that provided the best test accuracy in these terms to be learning faster.

The Appendix C provides a detailed explanation of the performed experiments.

All experiments were conducted without using dropout BID23 since recent studies have shown that, despite improving the training time, dropout provides unclear contributions to the overall deep model performance BID11 .

Moreover, dropout has recently become a technique restricted to fully-connected layers, which in turn are being less used and replaced by an average pooling layer in more recent architectures BID5 BID10 BID7 BID25 BID12 .

Therefore, since currently fully connected layers are rarely used in modern CNN, the usage of dropout is accordingly becoming unusual.

This can be demonstrated by observing that the most recent CNN models are not using dropout, but only batch normalization BID5 BID10 BID7 BID25 BID12 .

Particularly in the case of DenseNets BID10 , the results just using batch normalization are significantly better than using both techniques.

This recent tendency of design modern deep networks using only batch normalization but avoiding dropout can also be observed in the discriminative and generative convolutional models recently used in Generative Adversarial Networks (GANs) BID20 .

Hence, we emphasize that we designed the experiments of this comparative study to reflect the scenario we believe is currently the most likely and relevant from the perspective of training modern CNNs, which contemplates the use exclusively of batch normalization and no dropout.

The comparative study provided in the paper was designed to be self-contained to avoid misleading comparisons of experiments performed in entirely different situations.

In this sense, it should be noticed that the results presented by the papers that proposed the activation functions which are being used in this study (LReLU BID18 , PReLU BID6 and ELU ), must not be compared to the ones presented here because of the following reasons.

First, the studies previously mentioned were performed with use of dropout and without batch normalization, which is a technique that was not available when the mentioned studies were conducted.

The only exception is ELU, where a few tested scenarios used batch normalization.

However, even in those cases, dropout was always and intensively employed.

Second, the cited studies did not use standardized models such as VGG or ResNet where the only factor of change was the compared activation functions.

For example, in ELU paper, hand designed models were used to compare the performance of the activation function or in some cases completely different models were compared.

Third, the results presented by the mentioned works did not use statistical tests.

In this sense, considering the variation of the performance of the experiments based on different initialization or data shuffling, the conclusions may not be much trustable from a statistical point of view.

Hence, the conclusions regarding the performance of the cited activation functions based on their original paper may not be valid in the context where only batch normalization, but no dropout, is used to regularize the deep models.

In fact, one of the significant contributions of this paper is providing a systematic statistical supported comparative study using standardized models in the currently predominant scenario of using batch normalization without dropout.

In the following subsections, we analyze the tested scenarios.

In each case, we first discuss the activation functions learning speed based on test accuracy obtained for the partially trained models.

Subsequently, we comment about the test accuracy performances of the activation functions, which corresponds to the respective model test accuracy evaluated after 100 epochs.

Naturally, we consider that an activation function presents better test accuracy if it showed the higher test accuracy for the final trained models on a particular dataset.

In all scenarios, the null hypotheses were the test accuracy samples taken from different activation functions originated from the same distribution.

In other works, all the compared activation functions have the same test accuracy performance in the particular scenario.

The null hypotheses were rejected for all scenarios TAB0 , which means that with statistical significance (p < 0.05) at least one of the activation functions presents a test accuracy performance that is different from the others activation functions.

Therefore, we used the Conover-Iman post-hoc tests for pairwise multiple comparisons for all combination of datasets and models (Tables 3, 4, 5, 7, 8, 9) .

In these tables, the best results and p-values of the comparison of DReLU to other activation functions are in bold.

The TAB1 presents the mean of the nonlinearities layers mean activations performed in the CIFAR-100 training dataset.

It shows that DReLU is more capable of reducing the bias shift effect during training than ReLU.

Therefore, as expected and in agreement with 9, all of our experiments showed that the identity mapping extension produced less damage to the normalization performed by the previous layer and in fact mitigated the bias shift effect when compared to ReLU.

In the case of ResNet-56, DReLU overcame the test accuracy results of the other activation functions on either 40 and 70 epochs once again.

Therefore, we concluded DReLU generated the fastest learning FIG3 .

Regarding test accuracy, DReLU outperformed ReLU (p < 0.00228) and all other options, with exception to LReLU.

Although, no statistical significance was achieved in this pairwise comparison TAB4 .

In ResNet-100, DReLU also provided the fastest learning in all situations FIG4 .

Finally, DReLU test accuracy outperformed ReLU (p < 2.3 ?? 10 ???5 ) and all others activation functions again TAB6 .

Therefore, in CIFAR-100 as a whole, DReLU presented the fastest learning for all three models considered.

The results showed DReLU always outperformed ReLU test accuracy in all studied models.

Besides, DReLU was the most accurate in two evaluated models and the second in the other scenario.

4.2 CIFAR-10 DATASETThe TAB7 presents the averaged nonlinearities layers mean activations performed in the CIFAR-10 training dataset.

It shows again that DReLU is more efficient to reduce the bias shift effect during training than ReLU.

Hence, as sugested by 9, the experiments showed that the identity mapping extension produced less damage to the normalization performed by the previous layer and indeed mitigated the bias shift effect when compared to ReLU.

In VGG-19 model, DReLU provided faster learning than any other activation function for either 40 and 70 epochs FIG6 .

Moreover, DReLU presented the best test accuracy performance TAB9 .

We performed 20 executions of either DReLU and ReLU experiment.

In the case of ResNet-56, DReLU provided faster learning than any other nonlinearity on either 40 and 70 epochs in this scenario FIG7 .

Furthermore, DReLU was again the most accurate followed by ReLU TAB11 .

In relation to ResNet-110, DReLU provided the fastest leaning on either 40 and 70 epochs once more FIG9 .

DReLU was the most accurate solution also for this scenario TAB13 .

In this case, we also performed 20 runs of DReLU and ReLU.

Hence, in the CIFAR-10 as a whole, DReLU also presented the best learning speed for all considered models.

Moreover, the results showed DReLU again surpassed ReLU test accuracy in all analyzed scenarios.

Furthermore, DReLU was the most accurate activation function in all evaluated models.

4.3 DISCUSSION Primarily, we reemphasize the studies that proposed the activation functions compared in this paper used significantly different (specifically designed) models from the (standardized) ones used in this study.

In this sense, it is not possible to make a direct comparison between their results and the ones presented in this work.

In fact, in the mentioned papers, the usage regular of dropout may have produced a no ideal performance from ReLU since its fast training capacity was probably reduced.

In the mentioned papers, no experiments were executed using batch normalization without dropout.

As this study presents a significantly different scenario, we can expect different conclusions from this work.

Moreover, this paper performed statistical tests to prove that, in the conditions which experiments were executed, the proposed solution is consistently better than the other options.

To do that, we performed a significant number of executions of each experiment.

It is an important point to consider since the performance of same models presents a slight variation each time they are trained on a given dataset.

Taking into consideration the previous comments, in our experiments, regarding the CIFAR-100 dataset, DReLU presented the fastest learning for all three models considered.

Moreover, the results showed DReLU always outperformed ReLU test accuracy in all studied models.

Besides, DReLU was the most accurate in two evaluated models and the second in the other one.

In the CIFAR-10 dataset, DReLU also presented the best learning speed for all considered models.

Moreover, the results showed DReLU surpassed ReLU test accuracy in all analyzed scenarios.

Actually, DReLU was the most accurate activation function in all evaluated models.

It is important to mention that we commonly observed that ReLU usually produced the second best training speed and test accuracy performance.

This apparent surprise result may be explained by the use of batch normalization.

Indeed, the correction of the internal covariate shift problem enabled by the batch normalization technique acted relatively in benefit of ReLU and detriment of the other previously proposed units.

Hence, batch normalization significantly helped to avoid the so-called "dying ReLU" problem BID12 BID18 BID3 .In fact, even if a substantial gradient pushes the weights to the zero gradient region of ReLU, the normalization process tends to bring them back to inflection region of ReLU, which avoids the ReLU to die.

This fact can explain why ReLU typically outperforms LReLU, PReLU, and ELU in these situations but apparently did not when these activation functions were proposed a few years ago before the batch normalization advent.

The fact that batch normalization relatively helped ReLU in detriment of LReLU, PReLU and ELU make particularly impressive the ability of DReLU to overcome the performance of ReLU in exclusively batch normalized networks.

Particularly remarkable is the ability of DReLU to enhance the training speed during the first decades significantly.

In this paper, we have proposed a novel activation function for deep learning architectures, referred to as DReLU.

The results showed that DReLU presented better learning speed than the all alternative activation functions, including ReLU, in all models and datasets.

Moreover, the experiments showed DReLU was more accurate than ReLU in all situations.

Besides, DReLU also outperformed test accuracy results of all others investigated activation functions (LReLU, PReLU, and ELU) in all scenarios with one exception.

The experiments used batch normalization but avoided dropout.

Furthermore, they were designed to cover standard and commonly used datasets (CIFAR-100 and CIFAR-10) and models (VGG and Residual Networks) of several depths and architectures.

In addition to enhancing deep learning performance (learning speed and test accuracy), DReLU is less computationally expensive than LReLU, PReLU, and ELU.

Moreover, the mentioned gains were obtained by just replacing the activation function of the model, without any increment in depth or architecture complexity, which usually increases the computational resource requirements as processing time and memory usage.

This paper showed that the batch normalization procedure acted in the benefice of ReLU while other previews proposed activation functions appear not to perform as expected.

We believe this happened because batch normalization avoids the so-called "dying ReLU" problem, something that others activation functions were already not affected by in first place.

Furthermore, considering some evaluated models included skip connections, which are a tendency in the design of deep architectures like ResNets, we conjecture the results may generalize to other deep architectures such DenseNets BID10 ) that also use this structure.

Currently, all major activation functions adopt the identity transformation to positive inputs, some particular function for negative inputs, and an inflection on the origin.

In the following subsections, we describe the activation functions compared in this work.

A.1 RELU ReLU has become the standard activation function used in deep networks FIG10 .

Its simplicity and high performance are the main factors behind this fact.

The follow equation defines ReLU: DISPLAYFORM0 The Eq. (10) implies ReLU has slope zero for negative inputs and slope one for positive values.

It was first used to improve the performance of Restricted Boltzmann Machines (RBM) BID19 .

After that, ReLU was used in other neural networks architectures BID4 .

Finally, ReLU has provided superior performance in the supervised training of convolutional neural network models BID13 .The identity for positive input values produced high performance by avoiding the vanishing gradient problem BID9 .

A conceivable drawback is the fact ReLU necessarily generates positive mean outputs or activations, which generate the bias shift effect .

Consequently, while the identity for positive inputs is unanimously accepted as a reliable design option, there is no consensus on how to define promising approaches for negative values.

A.2 LRELU LReLU was introduced during the study of neural network acoustic models BID18 FIG10 ).

This activation function was proposed to avoid slope zero for negative inputs.

The following equation defines LReLU: DISPLAYFORM1 LReLU has no zero slope if ?? = 0.

In fact, it was designed to allow learning to happen even for negative inputs BID6 .

Moreover, since LReLU does not necessarily produce only positive activations, the bias shift effect may be reduced.

PReLU is also defined by the Eq. 11, but in this case ?? is a learnable rather than a fixed parameter BID6 FIG10 .

The idea behind the PReLU design is to learn the best slope for negative inputs.

However, this approach may implicate in overfitting since the learnable parameters may adjust to specific characteristics of the training data.

A.4 ELU ELU has inspiration in the natural gradient BID0 FIG10 .

Similarly to LReLU and PReLU, ELU avoids producing necessarily positive mean outputs by allowing negative activation for negative inputs.

The Eq. 12 defines ELU: DISPLAYFORM0 The main drawback of ELU is its higher computational complexity when compared to activation functions such as ReLU, LReLU, and DReLU.

In machine learning, normalizing the distribution of the input data decreases the training time and improves test accuracy BID24 .

Consequently, normalization also improves neural networks performance BID17 .

A standard approach to normalizing input data distributions is the mean-standard technique.

In this method, the input data is transformed to present zero mean and standard deviation of one.

However, if instead of working with shallow machine learning models, we are dealing with deep architectures; the problem becomes more sophisticated.

Indeed, in a deep structure, the output of a layer works as input data to the next.

Therefore, in this sense, each layer of a deep model has his own "input data" that is composed of the previous layer output or activations.

The only exception is the first layer, for which the input is the original data.

In fact, consider that the stochastic gradient decent (SGD) is being used to optimize the parameters ?? of a deep model.

Assume S a sample of m training examples in a mini-batch; then the SGD minimizes the loss given by the equation: DISPLAYFORM0 Furthermore, consider x the output of the layer i???1.

These activations are fed into layer i as inputs.

In turn, the outputs of layer i are fed into layer i+1 producing the overall loss given bellow: DISPLAYFORM1 In the above equation, F i+1 and F i are the transformation produced by the layers i + 1 and i, respectively.

The G function represents the mapping perpetrated by the above layers combined with the loss function adopted as the criterion.

Considering that the output of layer i is given by y = F i (x, ?? i ), we can rewrite the above equation as follows: DISPLAYFORM2 Applying equation (13) to equation (15) and considering a learning rate ??, we can write the equation to update the parameters of the layer i+1 as the follows: DISPLAYFORM3 In fact, the above equation is mathematically equivalent to training the layer i+1 on the input data given by y, which in turn is the output of the previous layer.

Therefore, indeed, we can understand the output of the previous layer as "input data" for the effect of training the current layer using SGD.Considering each layer has its own "input data" (the output of the previous layer), normalizing only the actual input data of a deep neural network produces a limited effect in enhancing learning speed and test accuracy.

Moreover, during the training process, the distribution of the input of each layer changes, which makes training even harder.

Indeed, the parameters of a layer are updated while its input (the activations of the previous layer) is modified.

This phenomenon is called internal covariant shift, which is a major factor that hardens the training of deep architectures BID11 .

In fact, while the data of shallow models is normalized and static during training, the input of a deep model layer, which is the output of the previous one, is neither a priori normalized nor static throughout training.

Batch normalization is an effective method to mitigate the internal covariant shift BID11 .

This approach, which significantly improves training speed and test accuracy, proposes normalizing the inputs of the layers when training deep architectures.

The layers inputs normalization is performed after the each mini-batch training to synchronizing with the deep network parameters update.

Therefore, when using batch normalization, for a input DISPLAYFORM4 , each individual dimension is transformed as follows: DISPLAYFORM5

The experiments were executed on a machine configured with an Intel(R) Core(TM) i7-4790K CPU, 16 GB RAM, 2 TB HD and a GeForce GTX 980Ti card.

The operational system was Ubuntu 14.04 LTS with CUDA 7.5, cuDNN 5.0, and Torch 7 deep learning library.

We trained the models on the CIFAR-100 and CIFAR-10 datasets.

The CIFAR-100 image dataset aggregates 100 classes, which in turn contain 600 example images each.

From these, 500 are used for training, and 100 are employed for testing.

Each example is a 32x32 RGB image.

The CIFAR-10 dataset possesses ten classes containing 6000 images, from which 5000 are used for training, and 1000 are left for testing.

Again, each training example is an RGB image of size 32x32.The experiments used regular mean-standard preprocessing.

Therefore, each feature was normalized to present zero mean and unit standard deviation throughout that training data.

The features were also redimensioned using the same parameters before performing the inference during test.

The data augmentation performed was randomized horizontal flips and random crops.

Therefore, before training, each image was flipped horizontally with a 0.5 probability.

Moreover, four pixels reflected from the picture opposite sides were added to expand it vertically and horizontally.

Finally, a 32x32 random crop was taken from the enlarged image.

The random crop was then used to train.

In this paper, we used the parameters originally proposed by the activation function designers BID6 since these have been kept in subsequent papers (Shah et al., 2016; BID8 .

Therefore, we kept ?? = 0.25 for both LReLU and PReLU, and for ELU we maintained ?? = 1.0.

We decided to keep those parameters because we consider the original authors and the follower papers, which respectively proposed and kept the original parameterizations, performed hyperparameter search and validation procedures to estimate parameter values that provide high performance for their proposed or used work.

As a particular instance of a VGG model, we used the VGG variant with nineteen layers .

To train models with a considerably different number of layers, we chose the pre-activation ResNet with fifty-six layers (ResNet-56) and also the pre-activation ResNet with one hundred ten layers (ResNet-110).

We employed pre-activation ResNets because they present better performance when compared to the original aproach BID5 .

The experiments used the Kaiming initialization BID6 .

The experiments used an initial learning rate of 0.1, and a learning rate decay of 0.2 with steps in the epochs 60, 80 and 90 for both CIFAR-10 and CIFAR-100 datasets.

The experiments employed mini-batches of size 128 and stochastic gradient descent with Nesterov acceleration technique as the optimization method.

The moment was set to 0.9, and the weight decay was equal to 0.0005.

To make assessments about the performance of activation functions, we chose the Kruskal-Wallis one-way analysis of variance statistical tests BID15 because the weight initialization was different for each experiment execution.

Consequently, we had independent and also possibly different size samples for the test accuracy distributions.

Moreover, the Kruskal-Wallis tests can also be used to confront samples obtained from more than two sources at once, which is appropriated in our study since, for a given scenario, we are comparing five activation functions simultaneously.

Besides, it does not assume a normal distribution of the residuals, which makes it more general than the parametric equivalent one-way analysis of variance (ANOVA).

We used the Conover-Iman post-hoc tests BID2 for pairwise multiple comparisons.

Clearly, ?? must not go to infinity because we would lose the nonlinearity in such a case.

Therefore, a compromise that causes less damage to the normalization while keeps the activation function nonlinearity has to be achieved.

To define the hyperparameter value of DReLU, we performed the experiments showed in TAB0 .

The results are the mean and standard deviation values of each experiment five executions.

Based on these experimental results, we decided to set ?? = 0.05.

The blue line presents the performance of SReLU, which can be observed is considerably lower than the DReLU using the chosen hyperparameter ?? = 0.05.

<|TLDR|>

@highlight

A new activation function called Displaced Rectifier Linear Unit is proposed. It is showed to enhance the training and inference performance of batch normalized convolutional neural networks.

@highlight

The paper compares and suggests against the usage of batch normalization after using rectifier linear units

@highlight

This paper proposes an activation function, called displaced ReLU, to improve the performance of CNNs that use batch normalization.