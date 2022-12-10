Adversarial examples have somewhat disrupted the enormous success of machine learning (ML) and are causing concern with regards to its trustworthiness: A small perturbation of an input results in an arbitrary failure of an otherwise seemingly well-trained ML system.

While studies are being conducted to discover the intrinsic properties of adversarial examples, such as their transferability and universality, there is insufficient theoretic analysis to help understand the phenomenon in a way that can influence the design process of ML experiments.

In this paper, we deduce an information-theoretic model which explains adversarial attacks universally as the abuse of feature redundancies in ML algorithms.

We prove that feature redundancy is a necessary condition for the existence of adversarial examples.

Our model helps to explain the major questions raised in many anecdotal studies on adversarial examples.

Our theory is backed up by empirical measurements of the information content of benign and adversarial examples on both image and text datasets.

Our measurements show that typical adversarial examples introduce just enough redundancy to overflow the decision making of a machine learner trained on corresponding benign examples.

We conclude with actionable recommendations to improve the robustness of machine learners against adversarial examples.

Deep neural networks (DNNs) have been widely applied to various applications and achieved great successes BID5 BID36 BID16 .

This is mostly due to their versatility: DNNs are able to be trained to fit a target function.

Therefore, it raises great concerns given the discovery that DNNs are vulnerable to adversarial examples.

These are carefully crafted inputs, which are often seemingly normal within the variance of the training data but can fool a well-trained model with high attack success rate BID14 .

Adversarial examples can be generated for various types of data, including images, text, audio, and software BID4 BID6 , and for different ML models, such as classifiers, segmentation models, object detectors, and reinforcement learning systems BID20 BID17 .

Moreover, adversarial examples are transferable BID38 BID23 )-if we generate adversarial perturbation against one model for a given input, the same perturbation will have high probability to be able to attack other models trained on similar data, regardless how different the models are.

Last but not the least, adversarial examples cannot only be synthesized in the digital world but also in the physical world BID7 BID21 , which has caused great real-world security concerns.

Given such subtle, yet universally powerful attacks against ML models, several defensive methods have been proposed.

For example, ; BID9 pre-process inputs to eliminate certain perturbations.

Other work BID1 suggest to push the adversarial instance into random directions so they hopefully escape a local minimum and fall back to the correct class.

The authors are aware of ongoing work to establish metrics to distinguish adversarial examples from benign ones so that one can filter out adversarial examples before they are used by ML models.

However, so far, all defense and detection methods have shown to be adaptively attackable.

Therefore, intelligent attacks against intelligent defenses become an arms race.

Defending against adversarial examples remains an open problem.

In this paper, we propose and validate a theoretical model that can be used to create an actionable understanding of adversarial perturbations.

Based upon the model, we give recommendations to modify the design process of ML experiments such that the effect of adversarial attacks is mitigated.

We illustrate adversarial examples using an example of a simple perceptron network that learns the Boolean equal operator and then generalize the example into a universal model of classification based on Shannon's theory of communication.

We further explain how adversarial examples fit the thermodynamics of computation.

We prove a necessary condition for the existence of adversarial examples.

In summary, the contributions of the paper are listed below:• a model for adversarial examples consistent with related work, physics and information theory;• a proof that using redundant features is a necessary condition for the vulnerability of ML models to adversarial examples;• extensive experiments that showcase the relationship between data redundancy and adversarial examples• actionable recommendations for the ML process to mitigate adversarial attacks.

Given a benign sample x, an adversarial example x adv is generated by adding a small perturbation to x (i.e. x adv = x + ), so that x adv is misclassified by the targeted classifier g. Related work has mostly focused on describing the properties of adversarial examples as well as on defense and detection algorithms.

Goodfellow et al. have hypothesized that the existence of adversarial examples is due to the linearity of DNNs BID14 .

Later, boundary-based analysis has been derived to show that adversarial examples try to cross the decision boundaries BID15 .

More studies regarding to data manifold have also been leveraged to better understand these perturbations BID25 BID13 Wang et al., 2016) .

While these works provide hints to obtain a more fundamental understanding, to the best of our knowledge, no study was able to create a model that results in actionable recommendations to improve the robustness of machine learners against adversarial attacks.

Prior work do not a measurement process or theoretically show the necessary or sufficient conditions for the existence of adversarial examples.

Several approaches have been proposed to generate adversarial examples.

For instance, the fast gradient sign method has been proposed to add perturbations along the gradient directions BID14 .

Other examples are optimization algorithms that search for the minimal perturbation BID3 BID23 .

Based on the adversarial goal, attacks can be classified into two categories: targeted and untargeted attacks.

In a targeted attack, the adversary's objective is to modify an input x such that the target model g classifies the perturbed input x adv as a targeted class chosen, which differs from its ground truth.

In a untargeted attack, the adversary's objective is to cause the perturbed input x adv to be misclassified in any class other than its ground truth.

Based on the adversarial capabilities, these attacks can be categorized as white-box and black-box attacks, where an adversary has full knowledge of the classifier and training data in the white-box setting BID37 BID14 BID2 BID28 BID32 BID0 BID19 BID21 , but zero knowledge about them in the black-box setting BID31 BID23 BID29 BID30 .Interestingly enough, adversarial examples are not restricted to ML.

Intuitively speaking, and consistent with the model that is presented in this paper, acoustic noise masking could be regarded as an adversarial attack on our hearing system.

Acoustic masking happens, for example, when a clear sinusoid tone cannot be perceived anymore because a small amount of white noise has been added to the signal BID33 ).

This effect is exploited in MP3 audio compression and privacy applications.

Similar examples exist, such as optical illusions in the visual domain BID34 and defense mechanisms against sensor-guided attacks (Warm et al., 1997) in the military domain.

Intuitively speaking, we want to explain the phenomenon shown in FIG1 , which depicts a plane filled with points that were originally perfectly separable with two 2D linear separations.

As the result of perturbing several points by a mere 10% of the original position, the separation of the two classes requires many more than two linear separators.

That is, a small amount of noise can overflow the separation capability of a network dramatically.

In the following section, we introduce an example model along which we will derive our mathematical understanding, consistent with our experiments in Section 4 and the related work mentioned in Section 2.

Consider a perceptron network which implements the Boolean equal function ("NXOR") between the two variables x 1 and x 2 .

The input x 3 is redundant in the sense that the result of x 1 == x 2 is not influenced by the value of x 3 .The first obvious observation is that adding x 3 doubles the input space.

Instead of 2 2 = 4 possible input pairs, we now have 2 3 = 8 possible input triples that the network needs to map to sustain the result x 1 == x 2 for all possible combinations of x 1 , x 2 , x 3 .

The network architecture shown in FIG0 , for example, theoretically has the capacity to be trained to learn all 8 input triples BID11 .

Translating this example into a practical ML scenario, however, this would mean that we have to exhaustively train the entire input space for all possible settings of the noise.

This is obviously unfeasible.

We will therefore continue our analysis in a more practical setting.

We assume a network like in FIG0 is correctly trained to model x 1 == x 2 in the absence of a third input.

One example configuration is shown.

Now, we train weights w 1 and w 2 to try to suppress the redundant input x 3 by going through all possible combinations for w i ∈ {−1, 0, 1}. This weight choice is without losing generality as the inputs x i are ∈ {0, 1} (see BID35 ).

An adversarial example is defined as a triple (x 1 , x 2 , x 3 ) such that the output of the network is not the result of x 1 == x 2 .

Simulating through all configurations exhaustively results in TAB1 .

The only case that allows for 100% accuracy, i.e., no adversarial examples, is the setting w 1 = w 2 = 0, in which case x 3 is suppressed completely.

In the other cases, we can roughly say that the more the network pays attention to x 3 , the worse the result (allowing edges).

That is, the result is better if one of the w i is set to 0 compared to none.

Furthermore, the higher the potential, defined as the difference between the maximum and the minimum possible activation value as scaled by the w i , the worse the result is.

The intuition behind this is that higher potential change leads to higher potential impacts to the overall network.

Using this simple model, one can see the importance of suppressing noise.

Thresholds of neurons taking redundant inputs should be high, or equivalently, weights should be close to 0 (and equal to 0 in the optimal scenario).

Now generalizing the example to a large network training images with 'real-valued' weights, it becomes clear that redundant bits of an image should be suppressed by low enough weights otherwise it is easy to generate an exponential explosion of patterns needed to be recognized.

The generalization of the example from the previous section is shown in FIG0 .

The model shows a machine learner performing the task of matching an unknown noisy pattern to a known pattern (label).

For example, a perceptron network implements a function of noisy input data.

It quantizes the input to match a known pattern and then outputs the results of the learned function from a known pattern to a known output.

Formally, the random variable Y encodes unknown patterns that are sent over a noisy channel.

The observation at the output of the channel is denoted by the random variable X. For example, X could represent image pixels.

The machine learner then erases all the noise bits in X to match against trained patterns which are then mapped to known outputsŶ , for example, the labels.

It is well known from the thermodynamics of computing BID10 that setting memory bits and copying them is theoretically energy agnostic.

However, resetting bits to zero is not.

In other words, we need to spend energy (computation) to reset the noisy bits added by the channel and captured in the observation to get to a distribution of patternsŶ that is isomorphic to the original (unknown) distribution of patterns Y .

Connecting back to the NXOR example from the previous section, Y would be the distribution over the input variables x 1 and x 2 .

The noise added is modeled by x 3 andŶ is the desired output isomorphic to x 1 and x 2 being equal.

Now assuming a fully trained model, this model allows us to explain several phenomena explored in the introduction and Section 2.First, as illustrated in the previous section, we can view the machine learner as a trained bit eraser.

That is, the machine learner has been trained to erase exactly those bits that are irrelevant to the pattern to be matched.

This elimination of irrelevance constitutes the generalization capability.

For a black box adversarial attack, we therefore just need to add enough irrelevant input to overflow this bit erasure function.

As a result, insufficient redundant bits can be absorbed and the remaining bits now create an exponential explosion for the pattern matching functionality.

In a whitebox attack, an attacker can guess and check against the bit erasing patterns of the trained machine learner and create a sequence of input bits that specifically overflows the decision making.

In both cases, our model predicts that adversarial patterns should be harder to learn as they consist of more bits to erase.

This is confirmed in our experiments in Section 4.

It is also clear that the theoretical minimum overflow is one bit, which means, small perturbations can have big effects.

This will be made rigorous in Section 3.3.

It is also well known that, for example, in the image domain one bit of difference is not perceivable by a human eye.

Training with noisy examples will most likely make the machine learner more robust as it will learn to reduce redundancies better.

However, a specific whitebox attack (with lower entropy than random noise), which constitutes a specific perceptron threshold overflow, will always be possible because training against the entire input space is unfeasible.

Second, with training data available, it is highly likely that a surrogate machine learner will learn to erase the same bits.

This means that similar bit overflows will work on both the surrogate and the original ML attack, thus explaining transferability-based attacks.

In the following we will present a proof based on the model presented in Section 3.2 and the currently accepted definition of adversarial examples (Wang et al., 2016) that shows that feature redundancy is indeed a necessary condition for adversarial examples.

Throughout, we assume that a learning model can be expressed as f (·) = g(T (·)), where T (·) represents the feature extraction function and g(·) is a simple decision making function, e.g., logistic regression, using the extracted features as the input.

Definition 1 (Adversarial example (Wang et al., 2016) ).

Given a ML model f (·) and a small perturbation δ, we call x an adversarial example if there exists x, an example drawn from the benign data distribution, such that f (x) = f (x ) and x − x ≤ δ.

We first observe that ∀x, x ∃δ such that x − x ≤ δ =⇒ f (x) = f (x ) is the generalization assumption of a machine learner.

The existence of an adversarial x is therefore equivalent to a contradiction of the generalization assumption.

This is, x could be called a counter example.

Practically speaking, a single counter example to the generalization assumption does not make the machine learner useless though.

In the following, and as explained in previous sections, we connect the existence of adversarial examples to the information content of features used for making predictions.

DISPLAYFORM0 Theorem 1.

Suppose that the feature extractor T (X) is a sufficient statistic for Y and that there exist adversarial examples for the ML model f (·) = g(T (·)), where g(·) is an arbitrary decision making function.

Then, T (X) is not a minimal sufficient statistic.

We leave the proof to the appendix.

The idea of the proof is to explicitly construct a feature extractor with lower entropy than T (X) using the properties of adversarial examples.

Theorem 1 shows that the existence of adversarial examples implies that the feature representation contains redundancy.

We would expect that more robust models will generate more succinct features for decision making.

We will corroborate this intuition in Section 4.2.

In this section, we provide empirical results to justify our theoretical model for adversarial examples.

Our experiments aim to answer the following questions.

First, are adversarial examples indeed more complex (e.g. they contain more redundant bits with respect to the target that need to be erased by the machine learner)?

If so, adversarial examples should require more parameters to memorize in a neural network.

Second, is feature redundancy a large enough cause of the vulnerability of DNNs that we can observe it in a real-world experiment?

Third, can we exploit the higher complexity of adversarial examples to possibly detect adversarial attacks?

Fourth, does quantization of the input indeed not harm classification accuracy?

Our model implies that adversarial examples generally have higher complexity than benign examples.

In order to evaluate this claim practically, we need to show that this complexity increase is in fact an increase of irrelevant bits with regards to the encoding performed in neural networks towards a target function.

This can be established by showing that adversarial examples are more difficult to memorize than benign examples.

In other words, a larger model capacity is required for training adversarial examples.

To quantitatively measure how much extra capacity is needed, we measure the capacity of multi-layer perceptrons (MLP) models with or without non-linear activation function (ReLU) on MNIST.

Here we define the model capacity as the minimal number of parameters needed to memorize all the training data.

To explore the capacity, we first build an MLP model with one hidden layer (units: 64).

This model is efficient enough to achieve high performance and memorize all training data (with ReLU).

After that, weights are reduced by randomly setting some of their values to zero and marking them untrainable.

The error is set to evaluate the training success (training accuracy is larger than 1 − ).

We explore the minimal number of parameters and utilize binary search to reduce computation complexity.

Finally, we change different and repeat the above steps.

As illustrated in Figure 3 , the benign examples always require fewer number of weights to memorize on different datasets with various attack methods.

It is shown that adversarial examples indeed require larger capacity.

From the training/testing process given in Figure 4 , we can draw the same conclusion.

The benign examples are always fitted and predicted more efficiently than adversarial examples given the same model.

That is to say, adversarial examples have more complexity and therefore require higher model capacity.

We now investigate if there are possible ways to exploit the higher complexity of adversarial examples to possibly detect adversarial attacks.

That is to say, we need a machine-learning independent measure of entropy to evaluate how much benign and adversarial examples differ.

For images, we utilized Maximum Likelihood (MLE), Minimax (JVHW) BID18 and compression estimators for BID2 ) on both MNIST and CIFAR-10 dataset.

These are four metrics for entropy measurement, all of which indicate higher unpredictability with the value increasing.

For compression estimation, prior work BID12 has found that an optimal quantification ratio exists for DNNs and appropriate perceptual compression is not harmful.

Therefore, we consider such information as redundancy and set the quality scale to 20 following their strategy.

We also reproduce the experiments in our settings and obtain the same results shown in FIG4 .

As shown in TAB0 , the benign images have smallest complexity in all of the four metrics, which suggests less entropy (lower complexity) and therefore higher predictability.

Similarly, we also design four metrics for text entropy estimation including mean bits per character, byte-wise entropy (BW), bit-wise entropy (bW) and compression size.

More specifically, BW and bW are calculated based on the histogram of the bytes or the bits per word.

It is worthwhile to note that all of the metrics are measured on adversarial-benign altered pairs, because adversarial algorithms only modify specific words of the texts.

In our evaluations, FGSM, FGVM BID27 and DeepFool attacks are implemented.

From TAB1 , we can draw the conclusion that adversarial texts introduce more redundant bits with regards to the target function which results in higher complexity and therefore higher entropy.

A reduction of adversarial attacks via entropy measurement is therefore potentially possible for both data types.

Inspired by Theorem 1, we investigate the relation between feature redundancy and the robustness of a ML model.

We expect that more robust models would employ more succinct features for prediction.

To validate this, we design models with different levels of robustness and measure the entropy of extracted features (i.e., the input to the final activation layer for decisions).

In experiments, we choose All-CNNs network for it has no fully-connected layer and the last convolutional layer is directly followed by the global average pooling and softmax activation layer, which is convenient for the estimation of entropy.

In other words, we can estimate the entropy of the feature maps extracted by the last convolutional layer using perceptual compression and MLE/JVHW estimators.

Specifically, we train different models on benign examples and control the ratios of adversarial examples in adversarial re-training period to obtain models with different robustness.

In general, the larger ratio of adversarial examples we re-train, the more robust models we will obtain.

The robustness in experiments is measured by the test accuracy on adversarial examples.

Then we obtain the feature maps on adversarial examples generated by these models and compress them to q = 20, following BID12 .

Finally, we measure the compressed entropy of using MLE and JVHW estimator like Section 4.2.

As illustrated in FIG5 , the estimated entropy (blue dots) decreases as the classification accuracy (red dots) increases for all the three adversarial attacks (FGSM, DeepFool, CW) and the two datasets (MNIST, CIFAR), which means that the redundancy of last-layer feature maps is lower when the models become more robust.

Surprisingly, adding adversarial examples into the training set serves as an implicit regularizer for feature redundancy.

Our theoretical and empirical results presented in this paper consistently show that adversarial examples are enabled by irrelevant input that the networks was not trained to suppress.

In fact, a single bit of redundancy can be exploited to cause the ML models to make arbitrary mistakes.

Moreover, redundancy exploited against one model can also affect the decision of another model trained on the same data as that other model learned to only cope with the same amount of redundancy (transferability-based attack).

Unfortunately, unlike the academic example in Section 3.1, we almost never know how many variables we actually need.

For image classification, for example, the current assumption is that each pixel serves as input and it is well known that this is feeding the network redundant information e.g., nobody would assume that the upper-most left-most pixel contributes to an object recognition result when the object is usually centered in the image.

Nevertheless, the highest priority actionable recommendation has to be to reduce redundancies.

Before deep learning, manually-crafted features reduced redundancies assumed by humans before the data entered the ML system.

This practice has been abandoned with the introduction of deep learning, explaining the temporal correlation with the discovery of adversarial examples.

Short of going back to manual feature extraction, automatic techniques can be used to reduce redundancy.

Obviously, adaptive techniques, like auto encoders, will be susceptible to their own adversarial attacks.

However, consistent with our experiments in Section 4.2, and dependent on the input domain, we recommend to use lossy compression.

Similar results using quantization have been reported for MP3 and audio compression BID12 as well as molecular dynamics BID22 .

In general, we recommend a training procedure where input data is increasingly quantized while training accuracy is measured.

The point where the highest quantization is achieved at limited loss in accuracy, is the point where most of the noise and least of the content is lost.

This should be the point with least redundancies and therefore the operation point least susceptible to adversarial attacks.

In terms of detecting adversarial examples, we showed in Section 4 that estimating the complexity of the input using surrogate methods, such as different compression techniques, can serve as a prefilter to detect adversarial attacks.

We will dedicate future work to this topic.

Ultimately, however, the only way to practically guarantee adversarial attacks cannot happen is to present every possible input to the machine learner and train to 100% accuracy, which contradicts the idea of generalization in ML itself.

There is no free lunch.

A PROOF OF THEOREM 1Proof.

Let X be the set of admissible data points and X denote the set of adversarial examples,We prove this theorem by constructing a sufficient statistic T (X) that has lower entropy than T (X).

Consider DISPLAYFORM0 where x is an arbitrary benign example in the data space.

Then, for all x ∈ X , g(T (x)) = g(T (x )).

It follows that T (x) = T (x ), ∀x ∈ X .

On the other hand, T (x) = T (x) by construction.

Let the probability density of T (X) be denoted by p(t), where t ∈ T (X ), and the probability density of T (X) be denoted by q(t) where t ∈ T (X \ X ).

Then, q(t) = p(t) + w(t) for t ∈ T (X \ X ), where w(t) corresponds to the part of benign example probability that is formed by enforcing an originally adversarial example' feature to be equal to the feature of an arbitrary benign example according to (2).

Furthermore, t∈T (X \X ) w(t) = t∈T (X ) p(t).

We now compare the entropy of T (X) and T (X): DISPLAYFORM1 It is evident that U 1 ≥ 0.

Note that for any p(t), there always exists a configuration of w(t) such that U 2 ≥ 0.

For instance, let t * = arg max t∈T (X \X ) p(t).

Then, we can let w(t * ) = t∈T (X ) p(t) and w(t) = 0 for t = t * .

With this configuration of w(t), U 2 = (p(t * ) + w(t * )) log((p(t * ) + w(t * )) − p(t * ) log p(t * ) (6) Due to the fact that x log x is a monotonically increasing function, U 2 ≥ 0.To sum up, both U 1 and U 2 are non-negative; as a result, H(T (X)) > H(T (X)) (7) Thus, we constructed a sufficient statistic T (·) that achieves lower entropy than T (·), which, in turn, indicates that T (X) is not a minimal sufficient statistic.

Apart from the adversarial examples, we also observed the same phenomenon for random noise that redundancy will lead to the failure of DNNs.

We tested datasets with different signal-to-noise ratios (SNR), generated by adding Gaussian noise to the real pixels.

The SNR is obtained by controlling the variance of the Gaussian distribution.

Finally, we derived the testing accuracy on hand-crafted noisy testing data.

As shown in FIG6 and 7b, a small amount of random Gaussian noise will add complexity to examples and cause the DNNs to fail.

For instance, noisy input sets with one tenth the signal strength of the benign examples result in only 34.3% test accuracy for DenseNet on CIFAR-10.

This indeed indicates, and is consistent with related work, that small amounts of noise can practically fool ML models in general.

@highlight

A new theoretical explanation for the existence of adversarial examples