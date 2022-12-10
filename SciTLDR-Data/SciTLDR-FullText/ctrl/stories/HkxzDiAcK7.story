This paper presents noise type/position classification of various impact noises generated in a building which is a serious conflict issue in apartment complexes.

For this study, a collection of floor impact noise dataset is recorded with a single microphone.

Noise types/positions are selected based on a report by the Floor Management Center under Korea Environmental Corporation.

Using a convolutional neural networks based classifier, the impact noise signals converted to log-scaled Mel-spectrograms are classified into noise types or positions.

Also, our model is evaluated on a standard environmental sound dataset ESC-50 to show extensibility on environmental sound classification.

Some conflicts between residents originated from incorrect source localization by human hearing.

Also, correctly identifying noise types/locations is the first step for the noise reduction.

Therefore, noise type/position classification is a technique required to identify impact noise.

Various impact noises such as footstep and hammer hitting in a living space incur annoyance to residents BID17 .

Chronic noise in a living space is a significant threat to resident's health BID18 BID15 .

In some case, impact noise arises conflict between residents.

Since more than 60 % of the residential buildings in Korea are apartment housings BID24 , the conflict has become serious social issue BID11 BID17 .

In 2012, the Korea government established the Floor Noise Management Center under Korea Environment Corporation affiliated with the Ministry of Environment BID4 for impact noise identification and conflict mediation.

The center has handled 119,500 civil complaints of impact noise over 6 years BID5 ).There are several related works on noise reduction BID2 BID12 , annoyance measurement BID17 , and noise measurement BID7 BID18 .

However, impact noise classification is studied only in our previous work (Anonymous-authors, 2018) .

Our previous work studies classification of the impact noises using a convolutional neural networks (CNN) based model.

Our model classifies impact noise recordings into labeled categories.

It shows extensibility of CNN to impact noise classification.

But, our model is evaluated on the limited data generated on limited positions.

And, the labels of dataset are categorized into noise type-position combined form.

In order to improve our previous work, this study expands the previous work as follows.

First, 1, 000 impact noise data is newly gathered on 10 more positions in the building.

The new data is set as test set to validate robustness of our model.

Second, the classification problem is divided into following two problems: noise type classification problem and position classification problem.

This form is considered as more adequate problem definition.

Also, the number of samples per category is increased which is expected to improve performance of our model.

Third, our model is validated on a standard environment sound dataset.

This validation can show the extensibility of our model to other problems.

We expect that this work can contribute to other fields.

Expected fields are noise type/position classification in a very complex structure, and environmental sound classification.

Since a dataset for noise type/position classification of impact noise does not exist, we built an impact noise dataset in our past work (Anonymous-authors, 2018) .

It is composed of audio clips of impact noise recorded by a smartphone microphone (Samsung Galaxy S6).

In this work, we gathered impact noise data on 10 more postions (19 locations in total) in the building to expand the dataset.

We planned dataset collection based on a report by the Floor-Noise-Management-Center (2018).

In the report, from 2012 to 2018, the center received 119,550 complaints from victims suffering from impact noise.

The center visited 28.1% of the victims to identify impact noise.

79.4% of the complaints were caused by the upper floor residents and 16.3% of the complaints were by the lower floor residents.

Identified noise types are listed in the following order: footstep (71.0%), hammering (3.9%), furniture (3.3%), home appliances (vacuum cleaner, laundry machine, and television) (3.3%), door (2.0%) and so on.

Unidentified or unrecorded sources account for 10.1% of the total.

Based on these results in the report, we focused on generation of impact noise on the upper floor(3F) and the lower floor(1F).

In addition to them, impact noises on the 2F(the middle floor) are also recorded to check whether our model can distinguish the noise generated on this floor from the noises on the other floors.

Also, top four noise types which occupy 81.5% of the identified noise types are selected.

Furthermore, it could hurt a person who generates the noise.

Thus, usually, an impact ball (2.5 kg, 185 mm) or a bang machine (7.3 kg) is used to produce low frequency noise of footstep noise BID7 .

Instead of using them, a medicine ball (2.0 kg, 200 mm) is used to produce the low frequency noise.

Since a laundry machine and a television are hard to transport and install at the noise location, only vacuum cleaner is used to generate noise.

VC is generated only on 2F because vacuum cleaner noise on the upper floor and the lower floor are barely audible at the receiver position in the building.

Sampling frequency and sample duration are set as 44,100 Hz and approximately 5 s, respectively.

TAB0 is summary of the finalized impact noise dataset.

It contains 2,950 floor impact noises in total and they can be classified into 59 categories.

Each category contains 50 recordings of floor impact noise.

DISPLAYFORM0 In this section, we explain our noise type/position classifier for impact noise generated in a building.

In Section 3.1, applications of CNN(convolutional neural networks) in audio area are briefly reviewed.

Noise type and position classifications are presented in details in Section 3.2.1 and Section 3.2.2, respectively.

In Section 3.3, our method is applied to classification of other standard environment sound dataset (ESC-50) to examine that our method can be extended to environmental sound classification problems.

CNN is well known for its remarkable performance than those of conventional machine learning techniques in visual recognition tasks.

CNN is also widely used in audio domain, such as environmental sound classification BID19 BID23 and music classification BID9 BID10 .

Input features of their models are time-frequency patch or raw waveform instead of using RGB color space image.

But, their design pattern is fundamentally same with that used in visual recognition task which is composed of convolutional layers, pooling layers, and fully connected layers.

There are several works which employ a model for visual recognition task to audio domain.

BID6 showed state-of-the-art models for visual recognition perform well on audio event classification.

Amiriparian et al. (2017) Usually, a CNN based model contains a large number of learnable parameters and its performance is limited if dataset is small BID16 .

In such a situation, transfer learning, known as a technique to improve the model performance, can be introduced BID29 BID16 BID13 BID26 .

The technique trains parameters of networks on a training data in source task.

In target task, the parameters are transferred and finetuned on a target data.

Pre-training of parameters in source task offers efficient learning because the parameters are pre-initialized in the source task BID26 .

BID0 and BID21 pre-trained their models on ImageNet dataset and transferred the parameters to models in target tasks.

Although these studies are visual knowledge transfer, the models perform well in audio domain.

VGG16 by BID25 is selected for this study instead of designing a new network architecture.

There are several reasons why we select the model as a baseline model in this study.

First, the model performs well in audio domain.

In particular, the performance difference between the state-of-the-art model is not large(at most 0.024 area under curve) in (Hershey et al., Figure 2 : Transferring pre-trained parameters.

C and FC represent convolutional layer and fully connected layer, respectively BID16 .2017).

Second, its pre-trained parameters are accessible on Visual Geometry Group (VGG) website and managed by the group.

Figure 2 illustrates the model used for this study.

The impact noise dataset contains smaller samples per category than a very large scale dataset.

Therefore, this shortage of the dataset can limit performance of our model for classification of noise type/position.

In order to overcome the limitation, pre-trained parameters by BID25 on ImageNet are transferred to VGG16.

An adaption layer which reduces output dimension to the number of categories is added and all the parameters of the model are fine-tuned on noise types or positions of the impact noises.

We named the model as VGG16-PRE.All the impact noise signals are converted to log-scaled Mel-spectrograms using LibROSA(version 0.5.1) BID14 .

Size of the log-scaled Mel-spectrogram is fixed to 224 × 224 by VGG16 whose input dimension is 224 × 224 × 3.

Log-scaled Mel-spectrogram is obtained by the following steps.

s with time duration of 3 s is extracted from each recording in the dataset.

The time duration covers almost of floor impact noise duration.

Event start in the metadata is referred for finding an initial location of each recording.

S is squared magnitude of short time Fourier transform of s using 2, 048 point fast Fourier transform (FFT), window size of 591, and hop size of 591.

The window size offers high time resolution of the time-frequency patch avoiding overlapping for the given input size and the time duration.

F S gives a Mel-spectrogram M , where F is a Mel-filter bank.

Frequency range of the Mel-filterbank is set as 0 − 22, 050 Hz.

The Mel-spectrogram is converted to a logscaled Mel-spectrogram P = 10 log M /M m , where M m is the maximum element of M .

Since VGG16-PRE has 3 input channels, P is supplied to all the channels.

The impact noises are labeled into 5 noise types: MB, HD, HH, CD, and VC.

Dimension of the adaptation layer (FCa) is set as 5 and the pre-trained parameters are transferred to VGG16-PRE.

L 2 -regularization is applied to the last layer with penalty value of 0.01.

VGG16-PRE is fine-tuned on the impact noises whose number of recordings are not written italics in TAB0 .

We named this dataset as TV-set(training and validation set).

The others are not used for the fine-tuning but purely used for testing the fine-tuned model.

Since they are generated out of the positions used for finetuning, it can be used for testing the robustness of noise type classification.

We named this dataset as TS-set(test set).VGG16-PRE is evaluated using 5-fold cross validation.

Usually, it is used for evaluating a model when a dataset is small.

Also, every model fine-tuned on k-th fold of TV-set is tested against TSset.

The fine-tuning minimizes cross-entropy loss with logits using mini-batch gradient descent with learning rate of 0.001 and mini-batch size of 30.

The global mean value of the input channel is changed to the mean of the training data.

The parameters of VGG16-PRE are not frozen for all the layers.

A softmax classifier is employed.

A model with the highest validation accuracy is saved during 30 epochs of training on each fold.

Validation accuracy on each fold of TV-set and test accuracy on TS-set are measured, respectively.

The impact noises in TV-set are labeled into 9 positions depending on their impact positions: 1F00m, 1F06m, 1F12m, 2F00m, 2F06m, 2F12m, 3F00m, 3F06m, and 3F12m, where the first two characters represents floor and the followings are distance from the receiver position in X direction.

One unique point of position classification is that TS-set is composed of impact noises generated out of the 9 positions used for fine-tuning.

So, it is an interesting point that to observe classification of TS-set into the 9 positions by a model fine-tuned on TV-set.

For fine-tuning a model, dimension of the adaptation layer (FCa) is set as 9 and the pre-trained parameters are transferred to VGG16-PRE.

The later steps including optimization and evaluation methods are same with those in Section 3.2.1 except performance measurement on TS-set.

We suggest a performance test for position classification on TS-set as follows.

FIG3 illustrates noise(source) positions on 3F where the impact noises are generated.

Intuitively, two dashed lines can divide the positions into three groups.

These two dashed lines are assumed as virtual boundaries.

The impact noises generated on 3F3m and 3F9m are excluded in performance measurement because they are on the boundaries.

True label of an impact noise in TS-set is assumed as the closest position in TV-set.

For example, true label of an impact noise whose source position is 3F8m is assumed as 3F6m.

Finally, test accuracy is measured using the assumed labels.

The impact noise can be considered as environmental sound.

In this section, VGG16-PRE is evaluated on a standard environmental sound dataset ESC-50 (Piczak, 2015b) .

Actually, this evaluation is out of scope for impact noise identification.

However, through the evaluation, VGG16-PRE can be verified on a standard sound dataset.

Also, robustness and extensibility of VGG16-PRE to environmental sound classification can be shown.

ESC-50 is composed of 50 categories and each category contains 40 environmental sounds.

ESC-50 is pre-arranged into five folds for fair performance comparison.

Time duration and sampling frequency of each audio clip are 5 s and 44, 100 Hz, respectively.

They are converted to log-scaled Mel-spectrograms by the method in Section 3.2.

Window size and hop size are set as 985 in order to use all time range of audio clip avoiding overlapping.

VGG16-PRE is fine-tuned on each fold for 10 epochs.

Mini-batch size and learning rate are set as 30 and 0.001, respectively.

Also, validation accuracy is measured.

TAB1 shows accuracies of the noise type classifier on TV-set and TS-set.

The first column of the table represents dataset.

The first row of the table represents noise types of the impact noises.

Validation accuracy on TV-set is measured as 99.7 %.

Test accuracy on the TS-set is measured as 99.2 %.

Since the classifier is trained only on the TV-set, test accuracy can be lower than the validation accuracy.

One notable result is, for noise type classification, VGG16-PRE shows robustness on position change.

As shown in TAB0 , impact positions used for generating the TS-set are out of those used for TV-set generation, but the accuracy difference between the validation accuracy and test accuracy is 0.5 %.

TAB2 shows validation accuracy of the position classifier on TV-set.

The first row of the table represents the 9 positions used for generating the TV-set.

The second row shows the corresponding validation accuracies to the 9 positions.

Average of the accuracies is 94.1 %.

When the accuracies are divided into 3 groups by floor, then validation accuracy on 1F is lower than that on 3F.

TAB3 shows test accuracy of the position classifier on TS-set, where the first row represents the true labels assumed in Section 3.2.2.

The second row of the table shows the corresponding test accuracies to the assumed true labels.

Average of the accuracies is 69.6 %.

Since positions of the impact noises in TS-set are different from those in TV-set, the test accuracy can be lower than the validation accuracy.

If the position classification is changed to floor classification, then the test accuracy is raised to 98.8 %.

FIG4 shows confusion matrices drawn with the validation and the test results.

The confusion matrix at the left is drawn with the validation results.

In the confusion matrix, the followings are observed.

Most of the errors are the misclassifications to neighboring positions on the same floor.

Especially, impact noises at X = 6 m are classified to the nearby locations.

It is also observed in TAB2 .The confusion matrix at the right is drawn with the test results.

The true labels are separately represented into two noise types: HH and MB, in order to observe the position classification to noise types.

The predicted labels are the true labels assumed in Section 3.2.2.

The dotted lines indicate a subset of the 9 positions used for training the model.

When test accuracy is separately calculated depending on noise type, test accuracies are 74.1 % for HH and 65.0 % for MB.

On ESC-50 repository, evaluation results of other models designed for environmental sound classification are reported (Piczak, 2015a).

TAB4 shows validation accuracies of our model on ESC-50 and the top-ranked models on ESC-50 repository.

Our model shows 12.3 % higher validation accuracy than the best model BID22 .

This experimental result supports that visual knowledge transfer can be effective to environmental sound classification.

BID22 0.865 EnvNet-v2 0.849 CNN pre-trained on AudioSet BID8 0.835 FIG5 shows confusions of VGG16-PRE on ESC-50.

In the confusion matrix, confusions between ESC-50 categories can be observed.

Also, validation accuracy to each category can be observed.

The most confusing category is engine.

The categories of ESC-50 can be loosely rearranged into 5 major categories: Animals, Natural soundscapes & water sounds, Human/non-speech sounds, Interior/domestic sounds, and Exterior/urban noises.

The most confusing major category is Exterior/urban noises (validation accuracy is 97.6 %).

In this study, a convolutional neural networks based model is proposed for noise type/position classification of impact noise.

An impact noise dataset is built for evaluation of our model.

The dataset is built based on a report by the Floor Management Center.

The dataset is divided into a trainingvalidation set and a test set.

The models for noise type and position classifications are separately designed, but their architectures are fundamentally same except the dimension of the adaptation layers.

VGG16 with an adaptation layer is employed for the tasks instead of designing a new model.

Since the impact noise dataset is small, parameters of VGG16 pre-trained on ImageNet are transferred to a model.

is the best accuracy ever reported on ESC-50 repository.

The result shows potential of the method to environmental sound classification as well as impact noise classification.

Future works include impact noise generation at other buildings and apartment houses, and evaluation of the model on another standard environmental sound dataset.

<|TLDR|>

@highlight

This paper presents noise type/position classification of various impact noises generated in a building which is a serious conflict issue in apartment complexes

@highlight

This work describes the use of convolutional neural networks in a novel application area of building noise type and noise position classification. 