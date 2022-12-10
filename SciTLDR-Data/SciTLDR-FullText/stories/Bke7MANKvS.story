Deep artificial neural networks can achieve an extremely small difference between training and test accuracies on identically distributed training and test sets, which is a standard measure of generalization.

However, the training and test sets may not be sufficiently representative of the empirical sample set, which consists of real-world input samples.

When samples are drawn from an underrepresented or unrepresented subset during inference, the gap between the training and inference accuracies can be significant.

To address this problem, we first reformulate a classification algorithm as a procedure for searching for a source code that maps input features to classes.

We then derive a necessary and sufficient condition for generalization using a universal cognitive similarity metric, namely information distance, based on Kolmogorov complexity.

Using this condition, we formulate an optimization problem to learn a more general classification function.

To achieve this end, we extend the input features by concatenating encodings of them, and then train the classifier on the extended features.

As an illustration of this idea, we focus on image classification, where we use channel codes on the input features as a systematic way to improve the degree to which the training and test sets are representative of the empirical sample set.

To showcase our theoretical findings, considering that corrupted or perturbed input features belong to the empirical sample set, but typically not to the training and test sets, we demonstrate through extensive systematic experiments that, as a result of learning a more general classification function, a model trained on encoded input features is significantly more robust to common corruptions, e.g., Gaussian and shot noise, as well as adversarial perturbations, e.g., those found via projected gradient descent, than the model trained on uncoded input features.

Generalization error in deep learning is typically defined as the difference between training and test errors measured on identically distributed training and test sets.

This traditional approach fails to take into account how representative these sets are of the empirical sample set from which input samples are drawn at inference time.

When the training and test sets are not sufficiently representative of the empirical sample set, the difference between training and inference errors can be significant, thus rendering the learned classification function ineffective.

The lack of the latter kind of generalization results in unreliable decisions, raising questions about how robust, fair, and safe a learned classification function is (Varshney & Alemzadeh, 2017) .

A natural question then arises: is there a necessary and sufficient condition ensuring that deep learning classifiers generalize in this broader sense?

If so, how can this condition be satisfied in a real-world setting?

To answer these questions, we draw on algorithmic information theory, which proposes a complexity measure, Kolmogorov complexity, as the absolute information content of any object, e.g., a computer program, function, or set.

After deriving a necessary and sufficient condition for generalization using the information distance (Bennett et al., 1998) , which is a universal cognitive similarity metric based on Kolmogorov complexity, and formulating an optimization problem for generalization, we turn our attention to coding theory in order to learn a more general classification function by extending the input features to a classifier with systematically generated encodings of the original features.

For a classification task, we assume that there exists a true classification function.

Given training and test sets, neither of which are sufficiently representative of the the empirical sample set from which input samples are drawn during inference, a learning algorithm is asked to find the true classification function.

In this work, we study how well the learned classification function generalizes with respect to the true classification function.

In other words, we study the problem of how to minimize the generalization error, which we define as the difference between the training error and inference error measured on the empirical sample set, as opposed to the difference between the training error and test error.

We use robustness to both common corruptions and adversarial robustness to measure how well a learned classification function generalizes on the empirical sample set, which contains corrupted or perturbed samples.

Universal cognitive similarity metric.

In order to find a necessary and sufficient condition for generalization in deep learning, we use the normalized information distance.

A key finding in algorithmic information theory is that the normalized information distance is a universal cognitive similarity metric: the normalized information distance between two objects minorizes any other admissible distance up to an additive logarithmic term (Bennett et al., 1998) .

In other words, although different learning algorithms will pick up on different dominating input features, depending on the classification task that they perform, every such dominating feature will be detected by the normalized information distance.

Classification function as a source code.

We formulate a learning algorithm as a procedure for searching for a source code based on training examples.

We show that the learned classification function is a lossy compressor: the classifier discards some information.

The input features thus cannot be recovered from the class label.

We use the normalized information distance between the true source code (true classification function) and the learned source code (learned classification function) to find a necessary and sufficient condition ensuring generalization, and then formulate the problem of learning a more general classification function as an optimization problem.

Compression-based similarity metric.

The normalized information distance provides the theoretical tools needed to learn more general source codes, but in practice the normalized information distance is not effectively computable.

We therefore use a compression-based similarity metric (Cilibrasi & Vitányi, 2005) based on a real-world compressor to approximate this theoretical construct.

Specifically, we use the normalized compression distance between the true source code and learned source code to derive an effectively computable condition on the compressed size of the learned source code to identify encodings of the input features that help to learn a more general source code.

Encoding input features.

In a typical communication system, a source code is followed by a channel code which is then followed by a physical channel.

In this paper, the learned source code (learned classification function) is preceded by one or more input codes that help ensure the learned classifier is more general by generating relations between input features that are not captured by the set of available input features.

In order to showcase our findings for a specific classification task, we use channel codes on the input features for CIFAR-10 image classification.

Precisely, we use a four-dimensional (4-D) five-level pulse-amplitude modulation (5-PAM) trellis-coded modulation (TCM) scheme (Ungerboeck, 1982; Hatamian et al., 1998; IEEE, 2015) to systematically generate multiple encodings of the set of available input features.

In doing so, we enable the deep neural network (DNN) to learn information from the empirical sample set which it could not learn from the uncoded input features alone.

The generalization error is thereby reduced.

The impact of generalization.

Through image classification experiments, we show that a model trained on arbitrarily encoded input features is significantly more robust to common corruptions, such as Gaussian noise and shot noise, and to adversarial perturbations, like those generated via projected gradient descent (PGD) (Madry et al., 2018) , than a model trained on uncoded input features.

The role of code design.

The code used on the input features can be designed in various ways for a classification task, and designing input codes is an important step to learning a more general classification function from the set of available input features.

We show that merely increasing the number of input channels of a DNN does not confer any robustness to Gaussian noise or to PGD.

How to design efficient input codes to build encoded DNNs is an intriguing research direction for achieving generalization in deep learning.

The literature on generalization, e.g. (Zhang et al., 2017; Neyshabur et al., 2017) , is largely concerned with minimizing the generalization error, defined as the difference between training and test errors measured on identically distributed training and test sets.

Minimizing this form of generalization error does not address the problem of generalizing to input samples drawn from an empirical sample set of which the training and test sets are not sufficiently representative, as we do herein.

In this subsection, we compare our work with domain-generalization, domain-adaptation, and dataaugmentation techniques to highlight their differences.

There is a substantial body of literature on domain generalization (Muandet et al., 2013; Li et al., 2017; Motiian et al., 2017; Shankar et al., 2018) , which aims to better generalize to unknown domains by training on samples drawn from different domains, not a single source, which is a limitation that our work does not have.

In this work, there is no need to draw training samples from a different domain.

We show that encoding the given training set enables a DNN to learn different relations between features that it could not learn from the uncoded training set alone.

There has also been much work on domain adaptation (Daumé III & Marcu, 2006; Saenko et al., 2010; Ganin & Lempitsky, 2015; Tzeng et al., 2017; Sun & Saenko, 2016; Morerio et al., 2018; Volpi et al., 2018a ) that addresses the problem of generalization to a priori fixed target domains, which is a different approach from ours because these algorithms need to access samples from the target distributions during an adaptation phase.

Importantly, our approach does not require accessing new samples during an adaptation phase in order to achieve generalization to the empirical sample set.

Similar to the domain adaptation work, there has been some work on adversarial training (Goodfellow et al., 2015; Lee & Raginsky, 2018; Sinha et al., 2018) , which aims to achieve robustness (Zhang et al., 2019) to adversarial perturbations by using training samples perturbed by a specific adversarial-perturbation method.

Adversarial training can be computationally costly because it requires generating adversarially perturbed training samples in each epoch of training, unlike in our work where input encodings need to be generated only once before training.

Furthermore, as there are numerous adversarial-perturbation methods (Goodfellow et al., 2015; Kurakin et al., 2017b; Madry et al., 2018) , an adversarially trained DNN does not necessarily generalize well to samples subjected to an adversarial perturbation method that was not used for training (Madry et al., 2018) .

There is also a substantial body of work on data-augmentation techniques (LeCun et al., 1995; Volpi et al., 2018b) , which perform simple label-preserving transformations of the training samples to provide a DNN with additional data points to learn from.

In this work, we do not generate new samples to increase the diversity of the training set; instead, we take a theoretically-grounded approach to extend the input features with their encodings in order to enable a DNN to learn a sufficiently complex classification function from the set of available input samples.

Our goal is to minimize the generalization error (defined in Appendix B) for a classification task, defined as the difference between training error and inference error, given a training set and a test set, both of which are not sufficiently representative of the empirical sample set from which input samples are drawn at inference time.

To accomplish this goal, we derive a necessary and sufficient condition under which a classifier will generalize well, and, based on that condition, cast the search for a classifier with good generalization (defined in Appendix B) as an optimization problem.

Our approach requires that we describe and compute the absolute information content of any object, e.g., a computer program, function, or set, in order to determine which of a pair of learned classification functions contains more information of the true classification function.

The appropriate tool here is a concept in algorithmic information theory: Kolmogorov complexity (defined in Appendix B).

Defining the amount of information in individual objects in terms of their Kolmogorov complexity has the advantage that it refers to these objects in isolation, not as outcomes of a known random source.

In contrast, quantifying the amount of information in individual objects based on their Shannon entropy requires that these objects be treated as members of a set of objects with an associated probability distribution.

This understanding is fundamental to our study because applying Shannon entropy to "an estimate of the quantity of information contained in a novel or in the translation of a novel into another language relative to the original" would not be clear (Kolmogorov, 1983) .

As a DNN may be employed to learn a classification function from a set of features contained in such an object as, for example, a document, image, video, or sound, we study the Kolmogorov complexity of the set of input features, model, and outputs of the DNN.

In our quest to find a condition ensuring our running definition of generalization, we require a distance function that measures how similar two objects are in any aspect so we can decide which of two learned classification functions is closer to the true classification function.

The closer a learned classification function is to the true classification function, the better its generalization error.

This distance function should satisfy the metric (in)equalities in order for it to have a meaning in the context of generalization.

For example, this distance function would have to be symmetric; i.e., the distance from object a to object b must be equal to that from object b to object a.

The normalized information distance (Bennett et al., 1998) between objects a and b, defined as

where K(a) denotes the Kolmogorov complexity of object a and K(a|b) denotes the Kolmogorov complexity of object a given b, satisfies the metric (in)equalities and is also a universal cognitive similarity metric because D I (a, b) minorizes all other normalized admissible distances up to a negligible additive error term.

This means that all effective similarities between a pair of objects are discovered by the normalized information distance; i.e., two objects that are close according to some effective similarity are also close according to the normalized information distance.

The main intuition behind normalizing the information distance max(K(a|b), K(b|a)) is that two larger objects that differ by a small amount are closer than two smaller objects that are different by the same amount: the absolute difference between two objects does not measure similarity as such, but the relative difference does (Cilibrasi & Vitányi, 2005) .

A successful DNN distills information useful for its classification task T from its input features x. In doing so, the DNN has to learn a classification function f (.) from the set X n of its input features to an m-ary alphabet A of classes u in such a way that some information in its input features is given less weight in determining its relevance to the class decisionû, and then entirely discarded by the arg max operation (Goldfeld et al., 2019) .

A deep learning classifier is thus acting as a source code C (defined in Appendix B).

Proofs of the following mathematical statements are given in Appendix A.

Lemma 1.

For a classification task T wherein each n-dimensional input sample x is mapped to a class u drawn from an m-ary signal alphabet A, the true output function f (·) of a learning algorithm is a source code C for a multivariate random variable X.

Lemma 1 reformulates a learning algorithm as a procedure for searching for a source code C for a multivariate random variable X, which compresses the values that this random variable takes, namely the input samples x. When a DNN generalizes well with respect to the true classification function f (·), it is able to decide which information in its input features is more relevant to making a particular class decision.

A DNN is a lossy compressor when the absolute information content of any of its input samples x is larger than that of the class u to which it is mapped.

Corollary 1.

The true source code C = f (·) of a learning algorithm used for the classification task T is a lossy compressor when the Kolmogorov complexity K( x) of one of its input samples is larger than the number of bits required to represent the corresponding class u.

Corollary 1 formalizes a deep learning classifier as a lossy compressor, so the source code C that corresponds to the true output function f (·) is not uniquely decodable; i.e., its input samples x cannot be recovered from the class u to which they are mapped.

A DNN can be trained to learn a source code that generalizes well with respect to the true source code, but first we will analyze the similarity between these two source codes by using the normalized information distance.

Source codes are designed for the most efficient representation of data (Cover & Thomas, 1991) .

Whether it is designed for a data-transmission or a data-storage system, a source code, whether lossless or lossy, should retain information about the data necessary to accomplish a given task.

The same consideration applies to a learning system.

The information in the input features of a learning system is represented by the classification function that it learns; thus, a neural network can be viewed as a source code that encodes inputs features for its classification task.

The reformulation of a learning algorithm as a procedure for searching for a source code allow us to exploit theoretical results from algorithmic information theory and coding theory for deep learning, thereby avoiding the necessity to reinvent theory that is already established in these fields.

Given that source codes are designed for the most efficient representation of data (Cover & Thomas, 1991) , we will exploit the duality of a source code and a channel code to learn a classification function that represents the input features more efficiently for the classification task T ; i.e., a more general classification function.

Showing that a deep learning classifier is a non-uniquely decodable source code is also fundamental to understanding that the normalized information distance between the input features and the output cannot be used to derive a condition for generalization in deep learning.

This results from the fact that deriving such a condition would require finding the conditional Kolmogorov complexity K( x|y) of the input features with respect to the output, which is impossible because the source code is not uniquely decodable; i.e., the program to go from the output to the input features cannot be found.

A necessary and sufficient condition for generalization based on the normalized information distance can hence be found only between a learned source code and the true source code.

The normalized information distance

between the true source code C and learned source codeC reveals how generalC is with respect to C. A necessary and sufficient condition ensuring that learned source codeC 0 is more general than learned source codeC 1 with respect to the true source code C is

Equation 3 is a direct result of using the normalized information distance as a universal cognitive similarity metric to determine whether learned source codeC 0 orC 1 is more general with respect to the true source code C. Because the normalized information distance (Bennett et al., 1998 ) is a metric that uncovers all effective similarities between the true source code and a learned source code, learning a source code that is closer to the true source code C under this metric ensures achieving generalization.

The normalized information distance D I (C,C) between the true source code C and the learned source codeC must thus be minimized in order to minimize the generalization error.

Theorem 1.

When a learning algorithm used for the classification task T finds a suboptimal source codeC instead of the true source code C, the optimization problem for the generalization ofC is minC(D I (C,C)) = minC max(K(C|C), K(C|C)).

Theorem 1 has formulated the optimization objective for generalization as the minimization of D I (C,C) and states that to achieve generalization we should make the learned function sufficiently complex for the classification task T .

Theorem 1 states that the Kolmogorov complexity K(C|C) of the program that computes how to go from the learned source codeC to the true source code C or the Kolmogorov complexity K(C|C) of the program that computes how to go from the true source code C to the learned source codeC, whichever is larger, must be minimized in order to minimize generalization error.

Thus, the goal is to increase the complexity of the learned source codeC, but not beyond the complexity of the true source code C. Therefore, Occam's first razor (Domingos, 1999) still holds: simpler classifiers generalize better than complex ones.

However, a classifier that input features encoded input features

Encoded Model does not perform well on its empirical sample set X n is too simple for its classification task.

Ideally, the learning algorithm would learn the true source code C, achieving the best possible performance metrics determined by its classification task T .

In practice, because the learning algorithm will see only a small subset X n S of the possible inputs at training time, the learned source codeC will be a partial function of the true source code C at perfect training accuracy (that is, when the classifier has sufficient capacity to memorize the training samples (Zhang et al., 2017) ).

Whether a model is over-fit or under-fit is conventionally determined on a cross-validation set and/or test set that are/is identically distributed with the training set, all of which are subsets of the empirical sample set.

Being more general on such a cross-validation set and/or test set does not as such guarantee generalization on the empirical sample set X n because the latter may contain corrupted or perturbed samples and/or there may be samples in the empirical sample set that are out of distribution of the cross-validation set and test set.

Therefore, whether a model is over-fit or under-fit does not have a consequence for Theorem 1.

Next, we target learning a source code that is more general on the empirical sample set X n , not only on a cross-validation set and/or test set.

In this work, we increase the complexity of the learned source codeC by generating I encodings E 0 , E 1 , . . .

E I−1 of the available input features x S that capture relations between the features which are not learned well from the original features, and then append these encodings to the original features.

Note that the available input features are denoted by x S , which are drawn from the set X n S of available features; i.e., x S ∈ X n S , which is a subset of the empirical sample set X n .

By providing a different view of the relations between the features, the encodings E i help the learning algorithm to learn a more complex source codeC E whose normalized information distance D I (C,C E ) to the true source code C is less than D I (C,C).

This results in learning a more general source code.

Theorem 2.

For classification task T , a more general suboptimal codeC E is learned from the concatenation { x S , E i ( x S )}, where

The effective capacity of several successful DNN architectures is sufficiently large to memorize the set X n S of available input samples (Zhang et al., 2017) .

Any encoding

is the set of available encoded samples such that Y n S ⊆ X n S , when concatenated with the uncoded input samples x S , thus increases the Kolmogorov complexity of the learned source code, which is now calledC E .

The task of the source code is to find the most efficient representation of its input data Cover & Thomas (1991) .

In a typical communication system, the source code compresses the input, then a channel code adds redundancy to guard against noise in the channel, then the encoded information is transmitted over the physical channel.

The design goal for the source and channel codes is to achieve the channel capacity (maximum mutual information between the channel input and output).

In contrast, Theorem 2 considers a learning system in which an input code is followed by a learned source code, the classification function, and the design goal is for the composition of the input and source codes to generalize as well as possible (see Figure 1) .

In other words, in a learning system the "physical channel" precedes the source code, and it can be seen as a process whereby the empirical sample set X n is reduced to the set X n S of available input samples and/or whereby common corruptions, such as Gaussian noise, and adversarial perturbations, such as those generated by PGD, are applied to the set of available input samples.

Because the "physical channel" comes first in a learning system, there is no access to the set of information bits.

Only a subset of these information bits can be accessed, which may have been subjected to common corruptions or adversarial perturbations.

It is therefore crucial for a learning algorithm to compress its features while retaining information useful for its classification task.

One way to accomplish this is to extend the input features with encodings that capture relations between features that are useful for classification and not captured well by the original set of input features.

The classification task T does not change when input features are extended by their encodings because it is defined by the mapping between the input features and the output (Goodfellow et al., 2016) , which remains the same because the only input to the encoded model is the uncoded input features (see Figure 1) .

The encoder is simply a new layer in the encoded model, which is designed from an encoder and an uncoded model.

The normalized information distance is based on the notion of Kolmogorov complexity, which is not a partial recursive function; i.e., it is not effectively computable.

While we can use normalized information distance to analyze whether a source codeC E learned from the concatenation { x S , E i ( x S )} of the encoded input samples E i ( x S ) with the uncoded input samples x S is more general with respect to the true source code C, in practice we may need to approximate the normalized information distance with the normalized compression distance, so we can determine which of any pair of source codes is more general with respect to the true source code C.

Based on a real-world compressor, the normalized compression distance (Cilibrasi & Vitányi, 2005 )

approximates the normalized information distance D I (C,C E ), where Z is a real-world compressor.

Thus, the generalization condition and minimization of D I (C,C E ) can be cast in effectively computable forms.

Note that neither Equation 2 nor Equation 4 is a training criterion, but they specify the normalized information distance and the normalized compression distance between the true source code C and a learned source code, respectively.

They are used to derive theoretical results, particularly the use of input codes to achieve generalization as illustrated by experiments in Section 3.

Proposition 1 states for classification task T that the compressed size Z(C E ) of the source codeC E learned from the concatenation { x S , E i ( x S )} of the encoded input samples E i ( x S ) and the uncoded input samples x S is larger than the compressed size Z(C) of the source codeC learned from the uncoded input samples alone x S .

Proposition 2.

When a learning algorithm used for classification task T finds a suboptimal source codeC E instead of the true source code C, the effectively computable optimization problem for the generalization ofC E is minC

Proposition 2 shows that the compressed size Z(C E ) of the source codeC E learned from the concatenation { x S , E i ( x S )} of the encoded input samples E i ( x S ) and the uncoded input samples x S must be maximized until it reaches the compressed size Z(C) of the true source code C to learn the most general source code with respect to the true source code C for the classification task T .

This statement is a consequence of the fact thatC E is a partial function of C at perfect training accuracy.

In other words, the source codeC E learned from the concatenation { x S , E i ( x S )} of the encoded input samples E i ( x S ) and the uncoded input samples x S can be made more general if the encoded input samples E i ( x S ) bear information of relations between input features that are not represented by its input samples.

A channel encoder generates encodings from its input features that enable a classifier to learn relations between these features not captured by the set of available input samples.

Concatenated together, these features are then input to a model to produce a class decision.

For example, we use a 4-D 5-PAM TCM scheme (Ungerboeck, 1982; Hatamian et al., 1998; IEEE, 2015) as a systematic way to generate multiple encodings of input features.

As shown in Figure 2 , the channel encoder flattens the input features by them into 2 × 2 patches of features, then, starting from the upper left feature and ending at the lower left feature, ordering them in a sequence going in the clockwise direction.

The features are traversed twice in order to avoid the initialization length of the channel code.

This particular scheme is used because it focuses on local relations between features.

Exploration of other flattening schemes is left for future research.

The features in the CIFAR-10 dataset are represented by eight bits.

The flattened features are fed to the convolutional encoder, which produces one extra bit out of the two least significant bits of the eight bits representing each feature.

The 4-D 5-PAM TCM symbol mapper then maps each nine bits into four equidistant 5-PAM symbols, which are then mapped to 12 bits by the bit mapper.

The bit mapper uses different symbol-to-bit mappings to generate different encodings of the input features, and the matrix used for generating these encodings is given in Appendix C.7.

Each encoding has the same size as the original input samples.

Figure 5 in Appendix C.1 shows three CIFAR-10 images and four of their encodings, which are arbitrarily chosen.

As seen in this figure, each encoding conveys a different view of the input features, which helps the source code (learned classification function) model relations between the features that are useful for the image classification task.

Note that using channel codes on the input features is not a data-augmentation technique: the encodings are appended to the input features, not treated as new input samples.

These encodings enable the classifier to learn from the set of available input samples a source code that is sufficiently complex for its classification task.

As in a data-transmission or data-storage system, the source code is designed for the most efficient representation of the data, which is the set of available input features for the classification task at hand, and the channel code is independently designed for the channel.

This combination is key to achieving generalization in deep learning, and how best to design a channel code for a given classification task is an intriguing future research direction.

Let the set of available input samples subjected to common corruptions and adversarial perturbations belong to the empirical sample space from which input samples are drawn during inference.

To show that using channel codes on the input features results in learning a more general source code with respect to the true source code, we conduct experiments on the CIFAR-10 and CIFAR-10-C (Hendrycks & Dietterich, 2019) datasets to show increased robustness to common corruptions and adversarial perturbations.

For CIFAR-10 and CIFAR-10-C, we train uncoded VGG-11 and VGG-16 models, encoded VGG-11 and VGG-16 models, and an uncoded ResNet-18 model.

The VGG networks are modified only by adding the encoder and increasing the number of input channels.

The encoded models use the same training criterion as the uncoded models, namely the cross-entropy loss.

The training setup and the achieved test accuracies are given in Appendix C.2.

In all experiments conducted on the encoded models, we use arbitrary encodings.

The input samples are corrupted or perturbed before they are input to the encoded models, as the uncorrupted or unperturbed input samples would not be accessible by a neural network in a real-world application.

Increasing the number of encodings may reduce the generalization error, but at the expense of increased run time.

However, encoding the training and test samples is a one-time process that can be done prior to training, unlike adversarial training, which requires generating perturbed input samples in each epoch.

In Appendix C.6, we show that increasing the number of input channels does not, as such, confer robustness to Gaussian noise or to PGD.

Designing efficient input codes for a given classification task considering the generalization error and the required number of encodings is a direction for future research.

To the best of the authors' knowledge, there is no other published method that can achieve robustness to both common corruptions and adversarial perturbations.

The set of available input samples may be subjected to common corruptions before reaching a realworld image classifier.

For example, Gaussian noise can appear in low-lighting conditions, and shot noise is caused by the discrete nature of light.

To show robustness to such corruptions, we conduct experiments on the CIFAR-10-C and CIFAR-10 datasets.

We use four common corruptions in our experiments, namely Gaussian noise, shot noise, impulse noise, and speckle noise.

The CIFAR-10-C dataset consists of the 10,000-sample CIFAR-10 test set subjected to five different noise levels, called severity, so it has 50,000 samples in all.

As shown in Figure 3 , increasing the number of arbitrary encodings concatenated to the original input features increases robustness to Gaussian noise, shot noise, and impulse noise.

The results for speckle noise is given in Appendix C.3.

For example, when test samples are subjected to impulse noise with a severity level of 4, we see a sharper increase in the number of test errors for the uncoded VGG-11 model than that for the VGG-11 model with 32 encodings.

Note that the vertical axis in these plots is cumulative: the number of test errors made at the previous severity level is added to that at the current severity level.

Table 1 in Appendix C.4 compares the encoded VGG-11 model with 32 encodings with previously published methods on the CIFAR-10-C dataset, which shows that the encoded VGG-11 model achieves the highest inference accuracy (defined in Appendix B) against shot noise with a severity level of 5 compared with all the other works listed in this table.

Additional experimental results on Gaussian noise are shown in Figure 6 in Appendix C.3.

To show robustness to adversarial perturbations without adversarial training, we conduct experiments on the CIFAR-10 dataset.

We use the white-box PGD attack (Madry et al., 2018) and transfer attacks from an uncoded VGG-16 and an uncoded ResNet-18 model to evaluate the adversarial robustness of the encoded VGG-16 models.

The results for the black-box boundary attack (Brendel et al., 2018) are given in Appendix C.3.

The white-box PGD attacks use the gradient of the loss function with respect to the uncoded input features in the encoded VGG-16 models because the channel encoder is part of the encoded VGG-16 models; i.e., the only input to the encoded model is the uncoded input features.

The encoder is simply a new layer of the neural network architecture, whose outputs are computed directly from the uncoded input features.

Changing the outputs of the encoder layer is tantamount to changing the outputs of any other layer of the model, which is a threat model that falls out of the scope of our work.

For the CIFAR-10 experiments, we use different numbers of encodings, and robustness to all adversarial perturbations in our experiments systemically increased with an increasing number of arbitrary encodings concatenated to the input features.

Figure 4 shows the results for the white-box PGD and transfer PGD attacks.

The plot on the left shows the increase in robustness to white-box PGD starting from a random perturbation around the natural example and using 20 iterations and a step size of 0.003. (Madry et al., 2018) .

The difficult problem of achieving adversarial robustness in a real-world application requires a holistic approach.

Our approach, which does not depend on adversarial training, can be readily used in combination with adversarial training and other known methods to achieve greater robustness to adversarial perturbations.

We presented a theoretical and experimental framework for defining and understanding generalization in deep learning, defined as the difference between training and inference errors.

The theoretical findings and experimental results show that a learned classification function must be sufficiently complex for a classification task in order to be closer to the true classification function.

Another insight from this study is that concatenating encodings of input features to the original input features helps to achieve generalization in deep learning by enabling the classifier to learn relations between features not captured by the original inputs.

Experiments demonstrate that a model trained on arbitrarily encoded input features is more robust to common corruptions and adversarial perturbations and that using more encodings may be beneficial to minimize the generalization error.

Designing input codes to help a DNN learn a more general classification function with a minimum number of encodings is an intriguing research direction to achieve reliability in machine learning.

Proof of Lemma 1.

For classification task T , a learning algorithm is asked to produce the true output function f (·) : X n → A. There exists a source code C for a random variable X, which is also a mapping from the sample space X n of X to the m-ary signal alphabet A from which a class u is drawn.

The true output function f (·) is equivalent to the source code C for the random variable X because their domain X n and codomain A are equal and the image of both functions is the same for each input sample x in the domain X n .

Proof of Corollary 1.

If the Kolmogorov complexity K( x) of an input sample x is larger than the number of bits required to describe the class u to which it is mapped, which is at most log 2 m bits, then some information about the input sample x is lost.

Satisfying this condition, the true source code C is a lossy compressor.

Proof of Theorem 1.

The normalized information distance

is a universal cognitive similarity metric that minorizes all other admissible distances up to a negligible additive error term.

This means that decreasing the normalized information distance D I (C,C) ensures that the true source code C and the learned source codeC are more similar; i.e., the learned source codeC is more general with respect to the true source code C. In a real-world setting, because the empirical sample space X n may be too large, the learning algorithm sees an input sample x S drawn from a subset X n S of X n ; i.e., X n S ⊂ X n .

Put differently, the set X n S of available input samples on which a neural network is trained and tested is a subset of the empirical sample set X n which the trained neural network sees during inference.

This means that true source code C bears information of all possible relations between input features that are useful for the classification task T , whereas the learned source codeC bears information of a subset of all possible relations between the input features.

The Kolmogorov complexity of the true source code is thus larger than that of a source code learned from the set of available input samples by a sufficiently high-capacity neural network, which can memorize its input samples (Zhang et al., 2017) ; i.e., K(C) > K(C).

Therefore,

is an optimization problem for the generalization of the learned source codeC with respect to the true source code C.

Proof of Theorem 2.

Any encoding E i : X n S → Y n S that bears information useful for the classification task T that is not entirely represented by the subset X n S of uncoded input samples; i.e., Y n S ⊆ X n S , when concatenated with the uncoded input samples x S , increases the Kolmogorov complexity of the learned source code, which is now calledC E , because a sufficiently high-capacity neural network can memorize its input samples (Zhang et al., 2017) .

Put differently, the Kolmogorov complexity K(C E ) of the source codeC E learned from a concatenation { x S , E i ( x S )} of uncoded and encoded input samples is larger than that of the source codeC learned from uncoded input samples alone if the encodings bear information of relations between input features that are not represented by the uncoded input samples.

As the the true source code C bears information of all possible relations between input features, the Kolmogorov complexity K(C E ) of the source codeC E learned from a concatenation { x S , E i ( x S )} of uncoded input samples and their encodings bearing information of a subset of all possible relations between input features is upper bounded by the Kolmogorov complexity K(C) of the true source code C; i.e., K(C) > K(C E ).

In other words, a sufficiently high-capacity neural network can memorize its input samples (Zhang et al., 2017) without being assisted by encodings E i .

However, the encodings E i bear information of relations between input features, which help to increase the Kolmogorov complexity of the learned source code if they are useful for the classification task T ; i.e., they are contained in the empirical sample set X n , of the neural network and if the information in the mappings contained in the input code, which is used to generate the encodings E i , is not represented in the set X n S of available input samples.

The conditional Kolmogorov complexities {K(C|C), K(C|C)} are thus both larger than {K(C|C E ), K(C E |C)}, respectively, because the program that computes how to go fromC E to C is shorter in length than the program that computes how to go fromC to C. The same holds in the reverse direction.

Therefore,

The source codeC E learned from the concatenation { x S , E i ( x S )} is thus more general than the source codeC learned from x S .

Proof of Proposition 1.

As the normalized information distance D I (C,C E ) is not effectively computable, it can be approximated for practical purposes by the normalized compression distance

where Z is a real-world compressor.

The learning algorithm sees an input sample x S drawn from a subset X n S of X n as the empirical sample space X n may be too large.

Because a sufficiently high-capacity neural network can memorize its input samples (Zhang et al., 2017) , the compressed size of the true source code is larger than that of the learned source code; i.e., Z(C) > Z(C E ).

At perfect training accuracy, the compressed size Z({C,C E }) of the concatenation {C,C E } is equal to Z(C), asC E is a partial function of C. For a sufficiently high training accuracy, we can consider |Z({C,C E }) − Z(C)| to be negligible for the purposes of generalization.

As the generalization condition D I (C,C E ) < D I (C,C) is not effectively computable, an equivalent effectively computable condition is useful for practical purposes.

As

for the purposes of generalization, the effectively computable condition

is equivalent to

Proof of Proposition 2.

By the Proof of Proposition 1, the effectively computable optimization problem for the generalization ofC E with respect to C is

B DEFINITIONS

Inference Accuracy.

The classification accuracy measured on a subset of the empirical sample set X n , which may be subjected to common corruptions or adversarial perturbations and which may be out of distribution of the training set, is defined as inference accuracy.

The definition of inference accuracy can be contrasted with that of test accuracy by considering that the former is measured on a subset of the empirical sample set X n which consists of corrupted or perturbed samples which may be out of distribution of the training set and that the latter is measured on the test set which consists of uncorrupted and unperturbed samples that are presumed to come from the same distribution as the training set.

Generalization Error.

The difference between the training error measured on the training set and inference error measured on a subset of the empirical sample set X n , which may be subjected to common corruptions or adversarial perturbations and which may be out of distribution of the training set, is defined as the generalization error.

This definition is different from that of prior works (Neyshabur et al., 2017; Kawaguchi et al., 2017) , which define generalization error as the difference between the training error measured on the training set and test error measured on the test set.

Generalization.

A learned classification function is said to be more general with a decreasing generalization error.

This definition is different from that of prior works (Neyshabur et al., 2017; Kawaguchi et al., 2017) , which define a learned classification function to be more general with a decreasing difference between the training error measured on the training set and test error measured on the test set.

Source Code.

A source code C for a random variable X is a mapping from the sample space X n of X to an m-ary signal alphabet A.

Source codes can be designed for the most efficient representation of the data (Cover & Thomas, 1991) .

Channel codes appropriate for a channel can be designed separately and independently.

This combination is as efficient as any other method that can be designed by considering both problems together.

We refer the reader to a textbook such as (Cover & Thomas, 1991) for a detailed understanding of source codes.

Kolmogorov Complexity.

The Kolmogorov complexity K U (x) of a string x with respect to a universal computer U is defined as

where p denotes a program, and l(p) denotes the length of the program p.

Thus, K U (x) is the shortest description length of x over all descriptions interpreted by computer U .

We fix such a universal computer U as reference and write K U (x) = K(x).

We refer the reader to a textbook such as (Cover & Thomas, 1991) and (Bennett et al., 1998; Cilibrasi & Vitányi, 2005) for a detailed understanding of Kolmogorov complexity.

C.1 ENCODED CIFAR-10 IMAGES Figure 5 shows three CIFAR-10 images and four of their encodings, which are arbitrarily chosen.

This figure shows that each encoding conveys a different view of the input features, which helps the learned source code model relations between the features that are useful for the image classification task.

All models are trained in PyTorch with 16 random initializations.

We train the networks over 450 epochs with a batch size of 128 and with a dynamic learning rate equal to 0.1 until epoch 150, 0.01 until epoch 250, and 0.001 until epoch 450 (Kuang, 2019

To show the robustness of the encoded VGG-11 models to Gaussian noise beyond the noise levels included in the CIFAR-10-C dataset, we apply Gaussian noise with zero mean and variance σ 2 w to the CIFAR-10 test set.

The average input-feature energy equals

where x i is a feature of the input sample x, k is the number of input samples in the test set, and n is the number of features in an input sample.

We define the signal-to-noise ratio to be

In Figure 6 , we show on the left plot that increasing the number of arbitrary encodings concatenated to the input features significantly increases robustness to Gaussian noise applied to the CIFAR-10 test set with signal-to-noise ratios from 25 to 0 dB. For example, at a signal-to-noise ratio of 12 dB, the inference accuracy of the VGG-11 model with 32 encodings is 61.15%, whereas that of the uncoded VGG-11 model is 21.49%.

On the right plot, the experimental results for the VGG-16 model with 32 encodings tested on the samples in the CIFAR-10-C dataset corrupted by Gaussian noise and shot noise are given.

The results indicate that using a larger encoded model does not necessarily confer more robustness to such common corruptions as Gaussian noise and shot noise than a smaller encoded model.

In Figure 7 , the results of the experiments conducted on the CIFAR-10-C dataset corrupted by speckle noise and black-box boundary attack experiments on the CIFAR-10 dataset are shown.

On the right plot, as with the other type of common-corruption experiments, we see that increasing the number of encodings increases robustness to speckle noise.

On the left plot, we see that the encoded model is significantly more robust to the boundary attack than the uncoded model.

For example, at a normalized 2 distance of 0.01, an inference accuracy of approximately 50% is achieved by the model with 32 encodings, whereas the inference accuracy of the uncoded model already drops to 0% at an 2 distance much closer to 0.

Table 1 compares the encoded VGG-11 model with 32 encodings with previously published methods on the CIFAR-10-C dataset.

At a severity level of 5, the encoded VGG-11 model achieves the highest inference accuracy against shot noise compared with all the other works listed in this table, which use a ResNet-18 or ResNet-26 model.

The highest inference accuracy (77.30%) against Gaussian noise is attained by the adversarial logit pairing (ALT) method published in Kannan et al. (2018) , but the test accuracy of this method is 83.50%, whereas the encoded VGG-11 model achieves the second highest inference accuracy (75.22%) against Gaussian noise with a test accuracy of 90.19%.

Our results seem to indicate that using a larger number of encodings improves robustness to common corruptions, so the inference accuracy achieved by the channel-coding method may be improved by merely increasing the number of encodings or designing higher performance codes.

C.5 PERFORMANCE COMPARISON: ADVERSARIAL PERTURBATIONS Table 2 compares the encoded VGG-16 model with previously published defenses on the CIFAR-10 dataset.

The experimental results do not imply that encoded input features are more robust to common corruptions and adversarial perturbations than uncoded features.

The encoded input features simply bear information of relations between input features that are not represented by the uncoded input features, so a source code learned from a concatenation of uncoded and encoded input features bears more information of the true source code than a source code learned from uncoded input features alone.

A source code learned from a concatenation of uncoded and encoded input features is thus more robust than a source code learned from uncoded input features alone.

To study the impact of increasing the number of input channels of the uncoded VGG-11 and VGG-16 models, we conducted experiments on the encoded VGG-11 and and VGG-16 models that use identical encodings; i.e., the input features are replicated across additional input channels (the "encoders" are just identity functions).

In Figure 8 , we see on the left that increasing the number of input channels of the uncoded VGG-11 model confers no robustness to Gaussian noise whatsoever.

The plot on the right shows that increasing the number of input channels of the uncoded VGG-16 model does not confer robustness to white-box PGD either.

Robustness to Gaussian noise (left) and the PGD attack (right) is tested by providing identical samples from the CIFAR-10 test set to the increased number of input channels.

The bit mapper in Figure 2 uses the matrix

to map four 5-PAM symbols into 12 bits.

In this symbol-to-bit mapping matrix, the i th row corresponds to the encoding E i , where 0 ≤ i ≤ 31.

Each symbol in the 5-PAM symbol alphabet is converted into three bits by using the corresponding three columns in this matrix.

For example, the first symbol in the 5-PAM symbol alphabet for the encoding E 3 is converted to [1 0 0] by drawing the bits from the third row and third, fourth, and fifth columns of the symbol-to-bit mapping matrix.

After all four of the 5-PAM symbols are converted into their respective three bits, these bits are concatenated to each other, determining the value of the corresponding feature in the encoded sample.

@highlight

We present a theoretical and experimental framework for defining, understanding, and achieving generalization, and as a result robustness, in deep learning by drawing on algorithmic information theory and coding theory.