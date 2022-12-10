In search for more accurate predictive models, we customize capsule networks for the learning to diagnose problem.

We also propose Spectral Capsule Networks, a novel variation of capsule networks, that converge faster than capsule network with EM routing.

Spectral capsule networks  consist of spatial coincidence filters that detect entities based on the alignment of extracted features on a one-dimensional linear subspace.

Experiments on a public benchmark learning to diagnose dataset not only shows the success of capsule networks on this task, but also confirm the faster convergence of the spectral capsule networks.

The potential for improvement of the quality of care via artificial intelligence has led to significant advances in predictive modeling for healthcare BID11 BID1 BID16 BID0 BID12 BID19 BID15 .

For accurate prediction, the models in healthcare need to not only identify risk factors, but also distill the complex and hierarchal temporal interactions among symptoms, conditions, and medications.

It has been argued that traditional deep neural networks might not be efficient in capturing the hierarchical structure of the entities in the images BID13 BID2 BID4 BID23 .

They argue that networks that preserve variations in the input perform superior to those that drop variations (equivariant vs. invariant architectures), as the upper layer can have access to the spatial relationship of the entities detected by the lower layers.

In particular, in capsule networks BID7 BID17 BID8 ) the capsules are designed to have both activation and pose components, where the latter is responsible for preserving the variations in the detected entity.

In this work, we first develop a version of capsule networks with EM routing (EM-Capsules) and show that it can accurately predict diagnoses.

We observe that EM-Capsules converge slowly in our dataset and are sensitive to the selection of hyperparameters such as learning rate.

To address these issues, we propose Spectral Capsule Networks (S-Capsules) that are also spatial coincidence filters, similar to EM-Capsules.

In contrast to EM-Capsules, S-Capsules measure the coincidence as the degree of alignment of the votes from below capsules in a one-dimensional linear subspace, rather than centralized clusters.

In S-Capsules, the variation (pose) component is the normal vector of a linear subspace that preserves most of the variance in the votes coming from below capsules and the activation component is computed based on the ratio of preserved variance.

Our experiments on a benchmark learning to diagnose task BID5 ) defined on the publicly available MIMIC-III dataset BID9 highlight the success of capsule networks.

Moreover, we confirm that the proposed S-Capsules converge faster than EM-Capsules.

Finally, we show that the elements of the S-Capsules' variation (pose) vector are significantly correlated with the commonly used hand-engineered features.

The learning to diagnose (phenotyping) task BID11 BID5 )

is a multivariate time series classification task, where we need to predict a patient's diseases based on his time series of vital signs and lab results.

It is a multi-label classification task, meaning that a patient can be diagnosed with multiple diseases or no disease.

Due to lack of space, we describe both our customized EM-Capsules for this task and the proposed S-Capsules by outlining and comparing the three steps in their forward pass and defer the details to Figure 2 in the appendix.

Only step 3 is different between the two architectures.

In both networks, capsules have two components: an activation α ∈ [0, 1] and a pose (variation) 1 vector u ∈ R d .

The choice of having a pose vector instead of a matrix in BID8 ) is due to the time series nature of our features.

Step 1: Extracting features.

First, we use one-dimensional convolutions to extract lower dimensional features for processing by capsules.

We use three residual blocks BID6 with increasing dilation inspired by BID21 to not only increase the receptive field of the convolutions but also reduce the dimension of the input with fewer layer and parameters.

Finally, we flatten the output of the residual network to have a 120-dimensional vector ready to be processed by capsule layers.

Step 2: Primary capsules.

In both architectures, we use two dense residual networks per capsule to create the activation and pose components of the primary capsules.

The choice of residual blocks as transformation operations instead of the linear map is because in healthcare we do not have a formal understanding of the deformations in the data, whereas in computer vision distortions such as rotation are well-studied BID23 .

Residual blocks allow simple non-linear transformations without over-parameterization of the network.

Step 3: Capsule to capsule computation.

The EM-Capsule network uses the EM-routing procedure as described in BID8 expect the fact that we choose the transformations to be residual blocks.

To describe the capsule computations in S-Capsule networks, consider two layers L and L + 1 with n L and n L+1 capsules, respectively.

The jth capsule in layer L + 1 computes the weighted votes from the layer L capsules as y j,i = α i R j,i (u i ) for each capsule i in layer L, where R j,i (·) is a dense residual block.

Then it concatenates all the weighted votes from layer L as a matrix Y j ∈ R n L ×d and computes the singular value decomposition of Y j = USV .

The pose vector for capsule j is simple the first (dominant) right singular vector u j = V[0, :], which is the normal vector of a linear subspace preserving most of the variance in the vote vectors of the capsules in layer L. The activation for the jth capsule is computed using the singular values s k as DISPLAYFORM0 where b is discriminatively trained and η is linearly annealed during the training.

Note that the ratio DISPLAYFORM1 is the fraction of variance of the votes from the below layer captured in the onedimensional subspace defined by u j and measures the agreement between the votes for the pose of the jth capsule.

We train the entire network in an end to end discriminative training with binary cross-entropy loss.

We also extended the spread loss in BID8 to the multi-label setting, but it performed measurably worse than the binary cross-entropy loss in our experiments.

We also found that adding a skip connection from the features extracted in Step 1 to the activations of the last capsule layer improves the performance.

Step 3 in S-Capsules only need the top-1 SVD which is more efficient than the full decomposition, reducing the computational cost of the network.2 The activations and poses computed inStep 3 are inherently normalized: we always have u j 2 = 1 and DISPLAYFORM0 Given a stable SVD implementation, the inherently normalized outputs stabilize the training of SCapsule networks.

Computation of the activations using the variance preserved in the top singular value has an additional self-annealing impact: random matrices usually have a sizable rank-1 component BID22 BID20 , which can prevent the death of capsules in the initial phases of training.

In practice, we found S-Capsules can operate with larger learning rates.

The learning to diagnose benchmark in BID5 extracts 78-dimensional multivariate time series for each patient from MIMIC-III dataset BID9 .

The data are divided into training/validation/test partitions of sizes 29,250/6,371/6,281 patients.

We follow the preprocessing and discretization process in the benchmark and also crop the time series to the last 50 time stamps.

We train all algorithms using Adam (Kingma & Ba, 2014) with batch size 64 and half the learning rate whenever the validation accuracy plateaus.

We tune the other hyperparameters using the provided validation set.

FIG0 shows the convergence behavior of EM-Capsules and S-Capsules as we train more batches.

FIG0 show the decrease of binary cross entropy and FIG0 shows the increase of micro-AUC.

The small rises and declines on figures b and c, respectively, are due to model selection with a separate validation set.

The intervals in figures b and c are one standard deviation intervals obtained by 1000 times bootstrapping on the test set.

For fairness, the learning rate is set to be equal for both algorithms, though S-Capsules could learn with higher rates.

The figures confirm that SCapsules learn faster and generalize better compared to EM-Capsules.

S-Capsules achieve the final AUC of 80.50% beating the accuracy of EM-Capsules and deep GRU networks -the common baseline-, which are 80.17% and 80.02%, respectively.

Interpreting pose vector of the output capsules.

It is interesting to analyze if the pose vectors of the output capsules preserve the variations in the input data.

While in computer vision this analysis can be done by visual inspection BID17 , understanding the patterns in medical data requires significantly more expertise.

Instead of visual inspection, we choose to construct the common hand-engineered medical features BID14 BID11 for the continuous variables in the input time series and measure the correlation between each dimension of the pose vector and them.

Given 13 continuous input variables and 7 features extracted from each, we construct 91 hand engineered features.

We test the Spearman correlation between each of the 15 dimensions of the pose vector and the 91 features.

After Bonferroni correction BID18 , at a p-value of 5%, we observe that 47.40% of times the pose vector elements are significantly correlated with the hand-engineered features.

This result indicates that the pose vectors do preserve a significant amount of variations in the input data.

Clearly, we do not like this percentage to be too large either, as we know that the hand-engineered features are not the perfect summary of the input data.

In this work, we customized capsule networks with EM routing BID8 ) for learning to diagnose task.

We also proposed spectral capsule networks to improve stability and convergence speed of the capsule networks.

Similar to EM-Capsules, S-Capsules are also spatial coincidence filters and look for agreement of the below capsules.

However, spectral capsules measure the agreement by the amount of alignment in a linear subspace, rather than a centralized cluster.

Setting aside the attention mechanism in EM-Capsules, the connection between S-Capsules and EM-Capsules is analogous to the connection between Gaussian Mixture Models and Principal Component Analysis.

This analogy suggests why S-Capsules are more robust during the training.

Our preliminary results confirm the superior convergence speed of the proposed S-Capsule network and preservation of variations in the data in its pose vectors.

Input time series Length = 50 Dim = 78

Step 1: Feature extraction Via 3 layers of Conv1d Residual blocksStep 2: Primary capsulesStep 3: Output capsules 40 capsules With dim=15

Kernel width = 3 Dilation increasing 1 to 6 Channels: [200, 150, 100, 80, 40, 15] Residual Blocks Spectral Mapping Figure 2 : Details of the S-Capsules architecture.

The mapping from flattened features and between two capsule layers are described in details in Section 2.

<|TLDR|>

@highlight

A new capsule network that converges faster on our healthcare benchmark experiments.

@highlight

Presents a variant of capsule networks that instead of using EM routing employs a linear subspace spanned by the dominant eigenvector on the weighted votes matrix from the previous capsule.

@highlight

The paper proposes an improved routing method, which employs tools of eigendecomposition to find capsule activation and pose.