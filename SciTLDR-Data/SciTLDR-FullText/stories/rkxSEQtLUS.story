In the visual system, neurons respond to a patch of the input known as their classical receptive field (RF), and can be modulated by stimuli in the surround.

These interactions are often mediated by lateral connections, giving rise to extra-classical RFs.

We use supervised learning via backpropagation to learn feedforward connections, combined with an unsupervised learning rule to learn lateral connections between units within a convolutional neural network.

These connections allow each unit to integrate information from its surround, generating extra-classical receptive fields for the units in our new proposed model (CNNEx).

We demonstrate that these connections make the network more robust and achieve better performance on noisy versions of the MNIST and CIFAR-10 datasets.

Although the image statistics of MNIST and CIFAR-10 differ greatly, the same unsupervised learning rule generalized to both datasets.

Our framework can potentially be applied to networks trained on other tasks, with the learned lateral connections aiding the computations implemented by feedforward connections when the input is unreliable.

While feedforward convolutional neural networks have resulted in many practical successes [1] , they are highly susceptible to adversarial attacks [2] .

In contrast, the brain makes use of extensive recurrent connections, including lateral and feedback connections, which may provide some level of immunity to these attacks (for results on human adversarial examples, see [3] ).

Additionally, the brain is able to build rich internal representations of information with little to no labeled data, which is a form of unsupervised learning, in contrast to the supervised learning required by most models.

We present a model incorporating lateral connections (learned using a modified Hebbian rule) into convolutional neural networks, with feedforward connections trained in a supervised manner.

When applying different noise perturbations to the MNIST [4] and CIFAR-10 [5] datasets, lateral connections in our model improve the overall performance and robustness of these networks.

Our results suggest that integration of lateral connections into convolutional neural networks is an important area of future research.

Orientation and distance dependence of lateral connections.

A) Left: Connection probability as a function of difference in preferred orientation between excitatory neurons observed experimentally (from [6] ).

Right: Normalized connection probability between excitatory neurons as a function of inter-somatic distance as reported experimentally in mouse auditory cortex [7] .

B, C): Model predictions for orientation and distance dependence (k 1 represents the target neuron) of positive (B) and negative (C) lateral connection weights for filters constructed using estimates of spatial receptive field (RF) sizes from in-vivo recordings in mouse V1 [8] .

Red (blue) bars/lines represent positive (negative) weights and dashed black lines represent Gaussian fits for distance dependence (standard deviations σ expt = 114µm, σ pos = 120 µm and σ neg = 143 µm for experiment, model positive and negative weights respectively).

Predicted connections qualitatively match with experimental data.

A number of normative and dynamical models relating contextual modulation of neuronal responses and lateral connectivity have been proposed in the literature.

Normative models based on sparse coding [9, 10, 11, 12, 13, 14, 15] predict anti-Hebbian lateral connections between excitatory neurons, in contrast with the experimentally observed like-to-like excitatory connectivity (but see [16] which extends the sparse coding model to learn like-to-like horizontal connections by including a pairwise coupling term in the prior).

Our model bears close resemblance to the MGSM (mixture of Gaussian scale mixtures) model of natural images proposed by Coen-Cagli, Dayan and Schwartz [17] , which infers contextual interactions between the RF and surround that would lead to optimal coding of images.

Recent work has demonstrated the ability to learn either flexible normalization [18] or divisive normalization [19] in deep neural networks.

In addition to these approaches, other normalization schemes from the machine learning field (e.g. batch normalization [20] , layer normalization [21] , local response normalization [22] ) have been used to accelerate training of neural networks.

In contrast with these approaches, we propose a computational role for each pyramidal neuron (or unit in a deep neural network) in terms of how it integrates lateral input information optimally from a Bayesian perspective.

Our proposal enables us to incorporate optimal lateral connections (learned in an unsupervised manner) straightforwardly into feedforward neural networks.

where the second term on the right side represents the contribution from the extra-classical RF, α represents a hyperparameter that tunes the strength of the lateral connections, and W mn,(l) jk are the synaptic weights from surrounding units n on to unit m within layer l. These weights are learned in an unsupervised manner using the rule: where .

x represents an average over the set of training images.

Note that this formula differs from a Hebbian learning rule, in that only the covariance between the feedforward responses of units leads to changes in the lateral connections.

A derivation of the above equations is shown in the Appendix.

Comparison to experimental data.

Using natural images from the Berkeley Segmentation Dataset [23] and a dictionary of classical RF features parameterized from mouse V1 electrophysiological responses [8] , we computed lateral connection weights using Eq. 2.

Figure 1 shows that predicted lateral connections qualitatively match the orientation and distance dependence of connectivity observed in mouse cortex.

Datasets.

We used the standard train/test splits, holding out 10% of the training data for validation.

We also added two types of noise to the original images: additive white Gaussian noise (AWGN) and salt-and-pepper noise (SPN).

The mean of the AWGN was set to zero and the standard deviation varied in increasing levels from 0.1-0.5.

For the SPN, the fraction of noisy pixels varied in increasing levels from 0.1-0.5.

Example stimuli (original and noisy images) are shown in Figure 2 .

Network architecture and training.

The model architectures used are described in Table 1 .

The CNNEx and CNN-PM models have the same number of parameters, with the additional convolutional layers in CNN-PM being standard feedforward layers trained in a supervised manner (in contrast to the lateral connections given by Eq. 1 for CNNEx, which are trained in an unsupervised manner).

MNIST models were trained for 10 epochs with a minibatch size of 64 using stochastic gradient descent with a learning rate of 0.01 and a momentum value of 0.5.

CIFAR models were trained for 50 epochs and a momentum value of 0.9, keeping all other hyperparameters the same.

We trained 10 different instantiations using different random seeds to ensure the robustness of our results.

All experiments were performed using Pytorch (v. 0.3.1) on a NVIDIA GTX 1080 Ti GPU.

Lateral connections.

We first trained feedforward weights in the network using supervised learning.

After freezing the feedforward weights, we introduced lateral connections given by Eqn.

2 between units in the first two convolutional layers.

Importantly, we note that the lateral connection weights are only learned once at the end of supervised training, by keeping the feedforward weights of the network fixed and using the activations of units over a set of training images.

We do not update the feedforward weights of the network by backpropagating through the computed lateral connections.

Future work will explore new methods for semi-supervised learning, which combine supervised learning of feedforward weights with unsupervised learning of the lateral connections.

Network regularization.

We chose a weight decay value of 0.005 and a dropout fraction of 0.5.

Weight decay acted on all non-bias parameters of the model, while dropout was applied after each convolutional layer in the model, as well as after the first fully connected layer.

We also tested the combination of these regularization techniques with the lateral connections.

Validation and testing.

Lateral connections had a spatial extent of 7x7 (3x3) pixels in the first (second) convolutional layers, with connections from the same spatial location set to zero.

Hyperparameters α for each of the two layers were chosen based on a grid search over the parameter range {0.1, 0.01, 0.001, 0.0001} using the validation dataset, followed by a finer search starting from this coarse value.

We did not use lateral connections for the two fully-connected layers.

Model performance.

Table 2 summarizes model test accuracy with and without regularization (weight decay + dropout) for the network architectures described in Table 1 .

For MNIST, the learned lateral connections provide improvement over the baseline and parameter-matched models only at higher levels of Gaussian noise (0.5) and salt-and-pepper noise (0.3 and above).

For CIFAR-10, we found that the learned lateral connections provide improved robustness to noise for almost all noise levels, with the cost of slightly decreased accuracy on the original images.

We did not try fine-tuning the models after incorporating the learned lateral connections, which may help recover some of this loss in accuracy.

Furthermore, we found that lateral connections combined with regularization often resulted in even better performance on both MNIST and CIFAR-10.

Effect of lateral connections.

Lateral connections had two complimentary roles: redundancy reduction leading to sparsification of feature activations on images without noise, and noise reduction on noisy images leading to feature activations closer to those on the original images.

The use of contextual information to modulate unit responses may underlie the ability of the CNNEx model to achieve higher accuracies under noisy conditions.

We show both effects on features in the first convolutional layer of our MNIST model (Figure 3 ).

In our model, lateral connections capture structure in the statistics of the world via unsupervised learning.

This structure allows for inference that can make use of the integration of information across space and features.

By combining these lateral connections with features learned in a supervised manner using backpropagation, the network does not learn any arbitrary structure present in the world, but only the structure of features which is needed to solve a particular task.

As a result, our method allows us to predict the structure of the world which is relevant to a given task.

The vast majority of deep neural networks are feedforward in nature, although recurrent connections have been added to convolutional neural networks [24, 25] .

Recurrent connections have also been used to implement different visual attention mechanisms [26, 27] .

However, these networks are still largely trained in a supervised manner.

An exception are ladder networks, which have been proposed as a means to combine supervised and unsupervised learning in deep neural networks [28] .

However, different from our approach, ladder networks use noise injection to introduce an unsupervised cost function based on reconstruction of the internal activity of the network.

Our model instead relies on a modified Hebbian learning rule which learns the optimal lateral connections between features within each layer based solely on the activations of units coding for these features.

Neurons are inherently noisy, and their responses can vary even to the same stimulus.

These neurons are embedded in cortical circuits that must perform computations in the absence of information, such as under visual occlusion.

Optimal lateral connections can provide additional robustness to these networks by allowing integration of information from multiple sources (i.e. different features and spatial locations).

This type of computation is also potentially useful for applications in which artificial neurons are not simulated with high fidelity, e.g. in neuromorphic computing.

We chose a relatively simple network architecture as a proof-of-concept for our model.

As such, we did not achieve state-of-the art performance on either image dataset.

This accuracy could be further improved by either fine-tuning models after learning the optimal lateral connections or using deeper network architectures with more parameters.

Future experiments will also have to test the scalability of learning optimal lateral connections on more complex network architectures and larger image datasets (e.g. ImageNet), and whether these connections provide any benefit against noise or other types of perturbations such as adversarial images.

@highlight

CNNs with biologically-inspired lateral connections learned in an unsupervised manner are more robust to noisy inputs. 