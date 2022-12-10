Deep neural networks are widely used in various domains, but the prohibitive computational complexity prevents their deployment on mobile devices.

Numerous model compression algorithms have been proposed, however, it is often difficult and time-consuming to choose proper hyper-parameters to obtain an efficient compressed model.

In this paper, we propose an automated framework for model compression and acceleration, namely PocketFlow.

This is an easy-to-use toolkit that integrates a series of model compression algorithms and embeds a hyper-parameter optimization module to automatically search for the optimal combination of hyper-parameters.

Furthermore, the compressed model can be converted into the TensorFlow Lite format and easily deployed on mobile devices to speed-up the inference.

PocketFlow is now open-source and publicly available at https://github.com/Tencent/PocketFlow.

Deep learning has been widely used in various areas, such as computer vision, speech recognition, and natural language translation.

However, deep learning models are often computational expensive, which limits further applications on mobile devices with limited computational resources.

To address this dilemma between accuracy and computational complexity, numerous algorithms have been proposed to compress and accelerate deep networks with minimal performance degradation.

Commonly-used approaches include low-rank decomposition BID15 BID14 , channel pruning (a.k.a.

structured pruning) BID6 BID17 , weight sparsification (a.k.a.

non-structured pruning) BID16 , and weight quantization BID1 BID2 .

However, these algorithms usually involve several hyper-parameters that may have a large impact on the compressed model's performance.

It can be quite difficult to efficiently choose proper hyper-parameter combinations for different models and learning tasks.

Recently, some researches adopted reinforcement learning methods to automatically determine hyperparameters for channel pruning BID4 and weight sparsification BID5 algorithms.

In this paper, we present an automated framework for compressing and accelerating deep neural networks, namely PocketFlow.

We aim at providing an easy-to-use toolkit for developers to improve the inference efficiency with little or no performance degradation.

PocketFlow has inte-grated a series of model compression algorithms, including structured/non-structured pruning and uniform/non-uniform quantization.

A hyper-parameter optimizer is incorporated to automatically determine hyper-parameters for model compression components.

After iteratively training candidate compressed models and adjusting hyper-parameters, a final compressed model is obtained to maximally satisfy user's requirements on compression and/or acceleration ratios.

The resulting model can be exported as a TensorFlow-Lite file for efficient deployment on mobile devices.

The proposed framework mainly consists of two categories of algorithm components, i.e. learners and hyper-parameter optimizers, as depicted in FIG0 .

Given an uncompressed original model, the learner module generates a candidate compressed model using some randomly chosen hyperparameter combination.

The candidate model's accuracy and computation efficiency is then evaluated and used by hyper-parameter optimizer module as the feedback signal to determine the next hyper-parameter combination to be explored by the learner module.

After a few iterations, the best one of all the candidate models is output as the final compressed model.

A learner refers to some model compression algorithm augmented with several training techniques as shown in FIG0 .

Below is a list of model compression algorithms supported in PocketFlow: NonUniformQuantLearner weight quantization with non-uniform reconstruction levels BID2 All the above model compression algorithms can trained with fast fine-tuning, which is to directly derive a compressed model from the original one by applying either pruning masks or quantization functions.

The resulting model can be fine-tuned with a few iterations to recover the accuracy to some extent.

Alternatively, the compressed model can be re-trained with the full training data, which leads to higher accuracy but usually takes longer to complete.

To further reduce the compressed model's performance degradation, we adopt network distillation to augment its training process with an extra loss term, using the original uncompressed model's outputs as soft labels.

Additionally, multi-GPU distributed training is enabled for all learners to speed-up the time-consuming training process.

For model compression algorithms, there are several hyper-parameters that may have a large impact on the final compressed model's performance.

It can be quite difficult to manually determine proper values for these hyper-parameters, especially for developers that are not very familiar with algorithm details.

Therefore, we introduce the hyper-parameter optimizer module to iteratively search for the optimal hyper-parameter setting.

In PocketFlow, we provide several implementations of hyper-parameter optimizer, based on models including Gaussian Processes (GP) BID11 , Tree-structured Parzen Estimator (TPE) BID0 , and Deterministic Deep Policy Gradients (DDPG) BID10 .

The hyper-parameter setting is optimized through an iterative process.

In each iteration, the hyper-parameter optimizer chooses a combination of hyperparameter values, and the learner generates a candidate model with fast fast-tuning.

The candidate model is evaluated to calculate the reward of the current hyper-parameter setting.

After that, the hyper-parameter optimizer updates its model to improve its estimation on the hyper-parameter space.

Finally, when the best candidate model (and corresponding hyper-parameter setting) is selected after some iterations, this model can be re-trained with full data to further reduce the performance loss.

For empirical evaluation, we adopt PocketFlow to compress and accelerate classification models on the CIFAR-10 BID9 and ILSVRC-12 BID12 data sets.

In FIG1 , we use ChannelPrunedLearner to speed-up ResNet-56 BID3 to reduce its computational complexity.

We observe that the accuracy loss under 2.5× acceleration is 0.4% and under 3.3× acceleration is 0.7%, and compressed models are more efficient and effective that the shallower ResNet-44 model.

In FIG1 , we use WeightSparseLearner to compress MobileNet BID7 to reduce its model size.

We discover that the compressed model achieves similar classification accuracy with much smaller model size than MobileNet, Inception-v1 BID13 , and ResNet-18 models.

The compressed models generated by PocketFlow can be exported as TensorFlow Lite models and directly deployed on mobile devices using the mobile-optimized interpreter.

In TAB2 , we compare the classification accuracy, model size, and inference latency 1 of original and compressed models.

With ChannelPrunedLearner, the compressed model achieves 1.53× speed-up with 2.0% loss in the top-5 classification accuracy.

With UniformQuantLearner, we achieve 2.46× speed-up after applying 8-bit quantization on the MobileNet model, while the top-5 accuracy loss is merely 0.6%.

In this paper, we present the PocketFlow framework to boost the deployment of deep learning models on mobile devices.

Various model compression algorithms are integrated and hyper-parameter optimizers are introduced into the training process to automatically generate highly-accurate compressed models with minimal human effort.

<|TLDR|>

@highlight

We propose PocketFlow, an automated framework for model compression and acceleration, to facilitate deep learning models' deployment on mobile devices.