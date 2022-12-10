Deep Neural Networks (DNNs) are increasingly deployed in cloud servers and autonomous agents due to their superior performance.

The deployed DNN is either leveraged in a white-box setting (model internals are publicly known) or a black-box setting (only model outputs are known) depending on the application.

A practical concern in the rush to adopt DNNs is protecting the models against Intellectual Property (IP) infringement.

We propose BlackMarks, the first end-to-end multi-bit watermarking framework that is applicable in the black-box scenario.

BlackMarks takes the pre-trained unmarked model and the owner’s binary signature as inputs.

The output is the corresponding marked model with specific keys that can be later used to trigger the embedded watermark.

To do so, BlackMarks first designs a model-dependent encoding scheme that maps all possible classes in the task to bit ‘0’ and bit ‘1’.

Given the owner’s watermark signature (a binary string), a set of key image and label pairs is designed using targeted adversarial attacks.

The watermark (WM) is then encoded in the distribution of output activations of the DNN by fine-tuning the model with a WM-specific regularized loss.

To extract the WM, BlackMarks queries the model with the WM key images and decodes the owner’s signature from the corresponding predictions using the designed encoding scheme.

We perform a comprehensive evaluation of BlackMarks’ performance on MNIST, CIFAR-10, ImageNet datasets and corroborate its effectiveness and robustness.

BlackMarks preserves the functionality of the original DNN and incurs negligible WM embedding overhead as low as 2.054%.

Deep neural networks and other Deep Learning (DL) variants have revolutionized various critical fields ranging from biomedical diagnosis and autonomous transportation to computer vision and natural language processing BID7 BID25 .

Training a highly accurate DNN is a costly process since it requires: (i) processing massive amounts of data acquired for the target application; (ii) allocating substantial computing resources to fine-tune the underlying topology (i.e., type and number of hidden layers), and hyper-parameters (i.e., learning rate, batch size), and DNN weights to obtain the most accurate model.

Therefore, developing a high-performance DNN is impractical for the majority of customers with constrained computational capabilities.

Given the costly process of designing/training, DNNs are typically considered to be the intellectual property of the model builder and needs to be protected to preserve the owner's competitive advantage.

Digital watermarking has been immensely leveraged over the past decade for ownership protection in the multimedia domain where the host of the watermark can be images, video contents, and functional artifacts such as digital integrated circuits BID9 BID12 BID24 .

However, the development of DNN watermarking techniques is still in its early stage.

Designing a coherent DNN watermarking scheme for model ownership proof is challenging since the embedded WM is required to yield high detection rates and withstand potential attacks while minimally affecting the original functionality and overhead of the target DNN.Existing DNN watermarking techniques can be categorized into two types depending on the application scenario. '

White-box' watermarking assumes the availability of model internals (e.g., weights) for WM extraction BID31 whereas 'black-box' watermarking assumes that the output predictions can be obtained for WM detection BID20 BID1 .

On the one hand, white-box WMs have a larger capacity (carrying multiple-bit information) but limited appli-cability due to the strong assumption.

On the other hand, black-box WMs enable IP protection for Machine Learning as a Service (MLaaS) BID25 where only zero-bit watermarking methods have been proposed.

It is desirable to develop a systematic watermarking approach that combines the advantages of both types of WMs.

While all present black-box watermarking papers embed the WM as a statistical bias in the decision boundaries of the DNN (high accuracy on the WM trigger set), our work is the first to prove that it is feasible to leverage the model's predictions to carry a multi-bit string instead of a one-bit boolean decision (existence or not of the WM).By introducing BlackMarks, this paper makes the following contributions:• Proposing BlackMarks, the first end-to-end black-box watermarking framework that enables multi-bit WM embedding.

BlackMarks possesses higher capacity compared to prior works and only requires the predictions of the queried model for WM extraction.• Characterizing the requirements for an effective watermarking methodology in the deep learning domain.

Such metrics provide new perspectives for model designers and enable coherent comparison of current and pending DNN IP protection techniques.• Performing extensive evaluation of BlackMarks' performance on various benchmarks.

Experimental results show that BlackMarks enables robust WM embedding with high detection rates and low false alarm rates.

As a side benefit, we find out that BlackMarks' WM embedding process improves the robustness of the model against adversarial attacks.

Digital watermarks are invisible identifiers embedded as an integral part of the host design and have been widely adopted in the multimedia domain for IP protection BID18 BID5 .

Conventional digital watermarking techniques consist of two phases: WM embedding and WM extraction.

FIG0 shows the workflow of a typical constraint-based watermarking system.

The original problem is used as the cover constraints to hide the owner's WM signature.

To embed the WM, the IP designer creates the stego-key and a set of additional constraints that do not conflict with cover constraints.

Combining these two constraints yields the stego-problem, which is solved to produce the stego-solution.

Note that the stego-solution simultaneously satisfies both the original constraints and the WM-specific constraints, thus enables the designer to extract the WM and claim the authorship.

An effective watermarking method is required to meet a set of criteria including imperceptibility, robustness, verifiability, capacity, and low overhead BID30 .

IP protection of valuable DNN models is a subject of increasing interests to researchers and practitioners.

BID31 take the first step towards DNN watermarking by embedding the WM in the weights of intermediate layers and training the model with an additional regularization loss.

The WM is later extracted from the weights of the marked layer assuming a white-box scenario.

BID27 present the first generic watermarking approach that is applicable in both white-box and black-box settings by embedding the WM in the activation maps of the intermediate layers and the output layer, respectively.

To alleviate the constraint on the availability of model internals during WM extraction, several papers propose zero-bit watermarking techniques that are applicable in the black-box scenario.

BID20 craft adversarial samples to carry the 'zero-bit' WM and embeds them in the decision boundary of the model by fine-tuning the DNN with the WM.

Null hypothesis testing is performed to detect the WM based on the remote model's response to the WM query images.

BID0 suggest to use the incorrectly classified images from the training data as the WM trigger images and generate random labels as the corresponding key labels.

A commitment scheme is applied to the trigger set to produce the WM marking key and the verification key.

The existence of the WM is determined by querying the model with the marking keys and performing statistical hypothesis testing.

BID34 propose three WM generation algorithms ('unrelated', 'content', 'noise') and embed the WM by training the model with the concatenation of the training set and the WM set.

To detect the WM, the remote model is queried by the WM set and the corresponding accuracy is thresholded to make the binary decision.

To the best of our knowledge, none of the prior works has addressed the problem of multi-bit black-box watermarking.3 BLACKMARKS OVERVIEW FIG1 shows the global flow of BlackMarks framework.

BlackMarks consists of two main phases: WM embedding and WM extraction.

The workflow of each phase is explained below.

BID31 BID20 , we believe verifiability and integrity are two other major factors that need to be considered when designing a practical DNN watermarking methodology.

Verifiabiity is important because the embedded signature should be accurately extracted using the pertinent WM keys; the model owner is thereby able to detect any misuse of her model with a high probability.

Integrity ensures that the IP infringement detection policy yields a minimal false alarms rate, meaning that there is a very low chance of falsely proving the ownership of an unmarked model used by a third party.

BlackMarks satisfies all the requirements listed in TAB1 as we empirically show in Section 5.

Accuracy of the target neural network shall not be degraded as a result of watermark embedding.

Watermark extraction shall yield minimal false negatives; WM shall be effectively detected using the pertinent keys.

Embedded watermark shall withstand model modifications such as pruning, fine-tuning, or WM overwriting.

The authorship of the unmarked models will not be falsely claimed by the watermarking method.

Watermarking methodology shall be capable of embedding a large amount of information in the target DNN.

Communication and computational overhead of watermark embedding and extraction shall be negligible.

The watermark shall be secure against brute-force attacks and leave no tangible footprints in the target DNN.

To validate the robustness of a potential DL watermarking approach, one should evaluate the performance of the proposed methodology against (at least) three types of contemporary attacks: (i) model fine-tuning.

This attack involves re-training of the original model to alter the model parameters and find a new local minimum while preserving the accuracy. (ii) parameter pruning.

Pruning is a common approach for efficient DNN execution, particularly on embedded devices.

We identify model pruning as another attack approach that might affect the watermark extraction.(iii) watermark overwriting.

A third-party user who is aware of the methodology used for DNN watermarking may try to embed a new WM in the distributed model and overwrite the original one.

BlackMarks takes advantage of the fact that there is not a unique solution for modern non-convex optimization problems used by DL models BID4 BID26 to embed the WM signature in the output activations of the target DNN.

In this section, we detail the workflow of WM embedding and extraction phase shown in FIG1 and discuss the watermarking overhead.

We use image classification as the cover problem in this paper, however, BlackMarks can be easily generalized to other data applications for IP protection of the deployed models.

Algorithm 1 summarizes the steps of BlackMarks' WM embedding.

BlackMarks encodes the ownerspecific WM information in the distribution of the output activations (before softmax) while preserving the correct functionality on the original task.

The rationale behind our approach is to explore the unused space within the high dimensional DNN (Rouhani et al., 2018b) for WM embedding.

BlackMarks formulates WM embedding as a one-time, 'post-processing' step that is performed on the pre-trained DNN locally by the owner before model distribution/deployment.

We explicitly discuss each of the steps outlined in Algorithm 1 in the following of this section.

Figure 3: BlackMarks' WM embedding algorithm.

1 Encoding Scheme Design.

Recall that our objective is to encode a binary string (owner's signature) into the predictions made by the DNN when queried by the WM keys.

BlackMarks designs a model-dependent encoding scheme that maps the class predictions to binary bits.

The encoding scheme (f ) is obtained by clustering the output activations corresponding to all categories (C) into two groups based on their similarity.

To do so, BlackMarks computes the averaged output activations triggered by images in each class and divides them into two groups using K-means clustering.

The resulting encoding scheme specifies the labels corresponding to bit '0' and bit '1', respectively.2 Key Generation 1.

The key generation module takes the encoding scheme, the owner's private signature (b ∈ {0, 1} K , where K is also used as the key length), and the original training data as inputs.

The output is a set of WM key image-label pairs for the target DNN.

More specifically, BlackMarks deploys targeted adversarial attacks to craft the WM key images and labels.

If the given bit in b is '0', the source class and the target class for the WM image ('adversarial sample') are determined by uniformly randomly selecting a class label that belongs to the cluster '0' and cluster '1' determined by the encoding scheme (f ), respectively.

The source class is used as the corresponding WM key label.

The WM keys for bit '1' in b are generated in a similar way.

We use targeted Momentum Iterative Method (MIM, BID8 and Jacobian-based Saliency Map Approach (JSMA, in our experiments (see Appendix A.1).

BlackMarks framework is generic and compatible with other targeted adversarial attack emthods for key generation.

BlackMarks aims to design specific WM key images as the queries for model authentication instead of crafting standard adversarial samples that are indistinguishable to human eyes to fool the DNN.

However, BlackMarks' WM key images can be considered as a generalization of 'adversarial samples' with relaxed constraints on the perturbation level and a different objective.

We assume that the WM signature and the key generation parameters (e.g., source and target classes, maximum distortion, step size, and the number of attack iterations) are secrets specified by the owner.

It's worth noting that the transferability of adversarial samples BID21 might lead to false positives of WM detection as shown in BID20 .

To address this problem, we set the initial key size to be larger than the owner's desired value K > K and generate the WM keys accordingly (K = 5 × K in our experiments).

The intuition here is that we want to filter out the highly transferable WM keys that are located near the decision boundaries.

3 Model Fine-tuning.

To enable seamless encoding of the WM information, BlackMarks incorporates an additive WM-specific embedding loss (L W M ) to the conventional cross-entropy loss (L 0 ) during DNN fine-tuning where a mixture of the WM keys and (a subset of) the original training data is fed to the model.

The formulation of the total regularized loss (L R ) is given in Eq. FORMULA0 where the embedding strength λ controls the contribution of the additive loss.

Here, we use Hamming Distance as the loss function L W M to measure the difference between the extracted signature (obtained by decode predictions) and the true signature b. DISPLAYFORM0 Note that without the additional regularization loss (L W M ), this retraining procedure resembles 'adversarial training' BID16 .

All of the existing zero-bit black-box watermarking papers BID20 BID34 BID0 leverages 'adversarial training' for WM embedding to ensure that the marked model has a high classification accuracy on the WM trigger set.

However, such an approach does not directly apply to multi-bit WM embedding where we care about the difference between the decoded signature and the original one instead of the difference between the received predictions and the WM key labels.

BlackMarks identifies this inherent requirement of multi-bit watermarking and formulates an additive embedding loss (L W M ) to encode the WM.

The rationale behind our design is that, when queried by WM key images, an additional penalty shall be applied if the prediction of the marked model does not belong to the same code-bit cluster as the corresponding WM key label.

4 Key Generation 2.

Once the model is fine-tuned with the regularized loss in step 3, we first find out the indices of initial WM keys that are correctly classified by the watermarked model (denoted by I M * ).

To identify and remove WM keys images with high transferability, we construct T variants of the original unmarked model (T = 3 in our experiments) to find out the indices of common misclassified initial keys (denoted by I M ).

Finally, the intersection of I M * and I M determines the indices of proper key candidates to carry the WM signature.

A random subset of candidate WM keys is then selected as the final WM keys according to the owner's key size (K).

In the global flow FIG1 , we merge the two key generation steps into one module for simplicity.

To extract the signature from the remote DNN (M ), the owner queries the model with the WM key images (X key ) generated in step 4 of WM embedding and obtains the corresponding predictions (Y key M ).

Each prediction is then decoded to a binary value using the encoding scheme (f ) designed in WM embedding.

The decoding is repeated for all predictions on the WM keys and yields the recovered signature (b ).

Finally, the BER between the true signature (b) and the extracted one (b ) is computed.

The owner can prove the authorship of the model if the BER is zero.

Here, we analyze the computation and communication overhead of WM extraction.

The runtime overhead of the one-time WM embedding is empirically studied in Section 5.6.

For the remote DNN service provider, the computation overhead of WM extraction is equal to the cost of one forward pass of WM key images through the underlying DNN.

For the model owner, the computation cost consists of two parts: (i) decoding the prediction response Y key M to a binary vector by finding out which cluster ('0' or '1' in the encoding scheme f ) contains each prediction; and (ii) performing an element-wise comparison between the recovered signature (b ) and the true one (b) to compute the BER.

In this case, the communication overhead is equal to the key length (K) multiplied by the sum of the input image dimension and one to submit the queries and read back the predicted labels.

We evaluate BlackMarks' performance on various datasets including MNIST (LeCun et al., 1998), CIFAR10 BID14 ) and ImageNet BID6 , with three different neural network architectures.

The experimental setup and the network architectures are given in Appendix A.1 and A.2, respectively.

We explicitly evaluate BlackMarks' performance with respect to each requirement listed in TAB1 as follows.

Empirical results prove that BlackMarks is effective and applicable across various datasets and DNN architectures.5.1 FIDELITY Fidelity requires that the accuracy of the target neural network shall not be significantly degraded after WM embedding.

TAB2 compares the baseline DNN accuracy (Column 2) and the accuracy of marked models (Column 3 and 4) after WM embedding.

As demonstrated, BlackMarks respects the fidelity requirement by simultaneously optimizing for the classification accuracy of the underlying model (the cross-entropy loss), as well as the additive WM-specific loss as discussed in Section 4.1.

In some cases (e.g. WideResNet benchmark), we even observe a slight accuracy improvement compared to the baseline.

This improvement is mainly due to the fact that the additive loss L W M in Eq. (1) act as a regularizer during DNN training.

Regularization, in turn, helps the model to mitigate over-fitting by inducing a small amount of noise to DNNs .

BlackMarks enables robust DNN watermarking and reliably extracts the embedded WM for ownership verification.

We evaluate the robustness of BlackMarks against three state-of-the-art removal attacks as discussed in Section 3.2.

These attacks include parameter pruning BID11 , model fine-tuning BID29 , and watermark overwriting BID31 .Model Fine-tuning.

Fine-tuning is a type of transformation attack that a third-party user might use to remove the WM information.

To perform such an attack, the adversary retrains the distributed marked model using the original training data with the conventional cross-entropy loss (excluding L W M ).

TAB6 in Appendix A.3 summarizes the impact of fine-tuning on the watermark detection rate across all benchmarks.

As can be seen from the table, the WM signature embedded by BlackMarks framework can be successfully extracted with zero BER even after the model is fine-tuned for various numbers of epochs.

Parameter Pruning.

We use the pruning approach proposed in BID11 to sparsify the weights in the target watermarked DNN.

To prune a specific layer, we first set α% of the parameters that possess the smallest weight values to zero.

The model is then sparsely fine-tuned using crossentropy loss to compensate for the accuracy drop caused by pruning.

FIG2 demonstrates the impact of pruning on WM extraction.

One can see that BlackMarks tolerates up to 95%, 80%, and 90% parameter pruning for MNIST, CIFAR-10, and ImageNet benchmark, respectively.

As illustrated in FIG2 , in cases where DNN pruning yields a substantial BER value, the sparse model suffers from a large accuracy drop.

Therefore, one cannot remove BlackMarks' embedded WM by excessive pruning while attaining a comparable accuracy with the baseline.

Watermark Overwriting.

Assuming the attacker is aware of the watermarking methodology, he may attempt to corrupt the original WM by embedding a new one.

In our experiments, we assume the adversary knows the targeted adversarial attack method employed by the model owner while the owner's signature and the key generation parameters remain secret.

In this case, the attacker generates another set of WM keys with his own signature and key generation parameters to fine-tune the marked model following the steps outlined in Algorithm 1.

TAB3 summarizes the accuracy of the overwritten DNN and the BER of the original WM signature (K = 20, 30, 50) for all three benchmarks.

In our experiments, we assume the attacker uses the same key length as the owner to generate the new WM keys.

BlackMarks can successfully extract the original WM in the overwritten DNN with zero BER, indicating its verifiability and robustness against WM overwriting attacks.

The malicious adversary may try to find/design the exact WM key used by the model owner (key collision) and disturbs the WM extraction.

The security of BlackMarks's WM key set is determined by the uncertainties involved in the key generation process (step 2 in Algorithm 1).

Since the key generation parameters and the WM signature are assumed to be secret information provided by the owner as discussed in Section 4.1, even if the attacker is aware of the adversarial attack method used to generate the WM key, he cannot reproduce the exact same key due to the large searching space in the targeted adversarial attack method.

Therefore, BlackMarks is secure against brute-force attacks.

Integrity requires that the watermarking technique shall not falsely claim the authorship of unmarked models.

For multi-bit watermarking, such requirement means that if an unmarked model is queried by the owner's WM key set, the BER between the decoded signature from the model's predictions and the owner's signature shall not be zero.

To evaluate the integrity of BlackMarks, we choose six unmarked models for each benchmark and summarize the results in TAB4 .

The first three models (M1-M3) have the same network topology but different weights as the watermarked model and the other three models (M4-M6) have different topologies as the marked model.

For each benchmark, the owner queries these six unmarked models with her WM keys and tries to extract the signature.

The computed BER is non-zero in all cases for three benchmarks, indicating that BlackMarks avoids claiming the ownership of unmarked DNNs and yields low false positive rates.

TAB1

One apparent advantage of BlackMarks over existing zero-bit black-box watermarking methods is its higher capacity as we discuss in Section 2.

To further improve the capacity of the WM, BlackMarks can be easily generalized to embed more complex signatures instead of binary vectors.

The amount of information carried by the owner's WM signature can be measured by entropy (Jaynes, 1957).

More generally, if the owner specifies her signature (a numeric vector) with base B and length K, the corresponding entropy can be computed as: DISPLAYFORM0 As can be seen from Eq. (2), a longer signature with a larger base value carries more information.

Since we use a binary vector (B = 2) as the WM signature in this paper, the entropy can be simplified as H = K. To extend BlackMarks framework for embedding a base-N signature, the owner needs to set the number of components in K M eans Clustering to N (Algorithm 1) and change the encoding as well as decoding scheme of predictions correspondingly.

BlackMarks is the first generic multi-bit watermarking framework that possesses high capacity in the black-box setting.

The WM extraction overhead is discussed in Section 4.3.

Here, we analyze the runtime overhead incurred by WM embedding.

Recall that the WM is inserted in the model by one-time fine-tuning of the target DNN with the regularized loss shown in Eq. (1).

As such, the computation overhead to embed a WM is determined by computing the additive loss term L W M during DNN training.

BlackMarks has no communication overhead for WM embedding since the embedding process is performed locally by the model owner.

To quantify the computation overhead for WM embedding, we measure the normalized runtime time ratio of fine-tuning the pre-trained model with the WMspecific loss and the time of training the original DNN from scratch.

To embed the WM, we use the entire training data for MNIST and CIFAR-10 benchmark and 10% of the training data for ImageNet benchmark in our experiments.

The selected training data is concatenated with the WM key set to fine-tune the model.

The results are visualized in FIG4 , showing that BlackMarks incurs a reasonable additional overhead for WM embedding (as low as 2.054%) even for large benchmarks.

Recall that WM embedding leverages a similar approach as 'adversarial training' while incorporating a WM-specific regularization loss (Section 4.1).

Here, we study the effect of WM embedding on the model's robustness against adversarial attacks.

TAB8 in Appendix A.3 compares the robustness of the pre-trained unmarked model and the corresponding watermarked model (K = 50) against different white-box adversarial attacks.

It can be seen that for each type of the attack, the marked model has higher accuracy on the adversarial samples compared to the unmarked baseline.

Such improvement is intuitive to understand since during WM embedding, the first term (cross-entropy loss) in the total regularized loss (see Eq.(1)) enforces the model to learn the correct predictions on training data as well as on the WM keys ('adversarial samples'), thus having a similar effect as 'adversarial training' BID16 .

Therefore, BlackMarks has a side benefit of improving the model's robustness against adversarial attacks.

In the future, we plan to extend BlackMarks framework to the multi-user setting for fingerprinting purpose.

BID3 present the first collusion-resilient DNN fingerprinting approach for unique user tracking in the white-box setting.

To the best of our knowledge, no black-box fingerprinting has been proposed due to the lack of black-box multi-bit watermarking schemes.

BlackMarks proves the feasibility of black-box fingerprinting methods and builds the technical foundation.

We propose BlackMarks, the first black-box multi-bit watermarking framework for IP protection of DNNs.

To the best of our knowledge, this work provides the first empirical evidence that embedding and extracting multi-bit information using the model's predictions are possible.

Our comprehensive evaluation of BlackMarks' performance on various benchmarks corroborates that BlackMarks coherently embeds robust watermarks in the output predictions of the target DNN with an additional overhead as low as 2.054%.

BlackMarks possesses superior capacity compared to all existing zerobit watermarking techniques and paves the way for future black-box fingerprinting techniques.

For ImageNet dataset (where the total number of categories is C = 1000), the sizes of the code-bit cluster '0' and cluster '1' are larger than ones in MNIST and CIFAR-10 dataset.

Therefore, the searching space for targeted adversarial samples is larger and the probability of WM key collision is smaller, ensuring the robustness of the generated WM keys against the WM over-writing attack.

JSMA is not applied to the ImageNet benchmark since the excessive memory requirement BID32 ) cannot be satisfied by our 11.74GiB test machine.

Model Fine-tuning for WM Embedding.

To embed the WM, we set the hyper-parameter λ to 0.5 for MNIST and CIFAR-10 benchmark, and to 0.01 for ImageNet benchmark in our experiments.

The pre-trained unmarked model is fine-tuned for 15 epochs with the regularized loss in Eq. FORMULA0 for all benchmarks.

We use the same batch size and the optimizer setting used for training the original neural network, except that the learning rate is reduced by a factor of 10.

Such retraining procedure coherently encodes the WM key in the distribution of output activations while preventing the accuracy drop on the legitimate data.

We summarize the DNN topologies used in each benchmark and the corresponding WM embedding results in TAB5 .

Here, K denotes the size of the WM key set, which is also equal to the length of the owner's WM signature.

In this section, we provide supplementary experimental results to support the evaluation of BlackMarks' performance in the paper.

Robustness against Model Fine-tuning Attack.

TAB6 shows the effect of model fine-tuning on BlackMarks WM extraction as discussed in Section 5.2.

The BER remains zero after the marked model is fine-tuned for various numbers of epochs, suggesting that BlackMarks is robust against fine-tuning attacks.

Note that the number of fine-tuning epochs for ImageNet benchmark is smaller than the other two since the number of epochs needed to train the ImageNet benchmark from scratch is 70 whereas the other benchmarks take around 200 epochs to be trained.

Evaluation of BlackMarks' Integrity.

TAB7 presents the results of the integrity evaluation of BlackMarks when different WM key sizes are used.

One can see that the BER between extracted signature from the unmarked model and the owner's true signature is non-zero in all cases, indicating that BlackMarks respects the integrity requirement with various key lengths.

Effect of BlackMarks' WM Embedding on Model's Robustness.

TAB8 compares the robustness of the pre-trained unmarked model and the corresponding marked model against various whitebox adversarial attacks as discussed in Section 6.

It can be observed that the watermarked model has higher classification accuracy on the adversarial samples compared to the unmarked model, indicating that BlackMarks WM embedding process has a positive impact on the model's robustness against adversarial attacks.

<|TLDR|>

@highlight

Proposing the first watermarking framework for multi-bit signature embedding and extraction using the outputs of the DNN. 

@highlight

Proposes a method for multi-bit watermarking of neural networks in a black-box setting and demonstrate that the predictions of existing models can carry a multi-bit string that can later be used to verify ownership.

@highlight

The paper proposes an approach for model watermarking where the watermark is a bit string embedded in the model as part of a fine-tuning procedure