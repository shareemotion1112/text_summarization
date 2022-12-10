Dreams and our ability to recall them are among the most puzzling questions in sleep research.

Specifically, putative differences in brain network dynamics between individuals with high versus low dream recall rates, are still poorly understood.

In this study, we addressed this question as a classification problem where we applied deep convolutional networks (CNN) to sleep EEG recordings to predict whether subjects belonged to the high or low dream recall group (HDR and LDR resp.).

Our model achieves significant accuracy levels across all the sleep stages, thereby indicating subtle signatures of dream recall in the sleep microstructure.

We also visualized the feature space to inspect the subject-specificity of the learned features, thus ensuring that the network captured population level differences.

Beyond being the first study to apply deep learning to sleep EEG in order to classify HDR and LDR, guided backpropagation allowed us to visualize the most discriminant features in each sleep stage.

The significance of these findings and future directions are discussed.

as well as EEG signal decoding [3] .

DL methods allow the identification of optimal discriminative 23 patterns in a given minimally pre-processed dataset, thus reducing the reliance on a priori selection 24 of features.

However, the interpretability of such deep models has been a major roadblock in the 25 context of neuroimaging applications.

In this work, we used a convolutional neural network (CNN) for classification of sleep EEG recordings 27 into two groups (HDR vs LDR).

Subsequently, we explored various techniques to visualize the 28 features learned by the network.

The sleep study consisted of 36 participants (18 male, mean age 23 ± 3 yrs).

Brief interview was 32 conducted to determine which dream recall group the subject belonged.

Those The preprocessed data (Section 2.2) is passed through the extractor (architecture described in 2).

The which enables the network to capture spatial patterns across the recording electrodes.

The output of To test the model's cross subject accuracy, we used a leave 2-subjects-out strategy, wherein the model 73 was trained on 34 subjects and tested on 2 held-out subjects (1 each from both dream recall groups).

18-Fold cross validation was done, and in each case 2 different subjects were a part of the testing set.

To assess the subject specificity of the features extracted by the network, we visualized a low-81 dimensional representation of the feature space using t-Distributed Stochastic Neighbor Embedding

(t-SNE) [4] .

To this end, we generated the tSNE plots for the output of the extractor part of the 83 network, as shown in Fig. 3 .

The left image corresponds to the feature space learned by our trained model.

There were no visible 85 clusters specific to a subject (each subject being represented by a color).

We compared this plot to the 86 feature space learned by a network trained to identify subjects from the EEG recordings (the right 87 image).

The formation of subject-specific clusters corresponds to the extracted features containing 88 subject-specific information.

This analysis confirmed that our proposed network did not learn features 89 based on subject-specific information.

Therefore, the learned feature space corresponds to population 90 level differences between the two groups.

This study illustrates how deep learning can be used to data-mine the neural activities of HDR and

LDR.

Specifically, we trained a deep CNN to classify between LDR and HDR and achieved significant 100 decoding accuracies.

Furthermore, we used tSNE to check for subject overfitting and GB to identify 101 the brain regions carrying group-specific differences, thus illustrating the use of visualisation tools 102 for deep models trained on neural data.

Future work will involve explorations of other dimensions,

including the frequency components of the data that contribute most to the classification.

@highlight

We investigate the neural basis of dream recall using convolutional neural network and feature visualization techniques, like tSNE and guided-backpropagation.