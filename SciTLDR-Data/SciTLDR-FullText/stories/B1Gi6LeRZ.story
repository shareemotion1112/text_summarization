Deep learning methods have achieved high performance in sound recognition tasks.

Deciding how to feed the training data is important for further performance improvement.

We propose a novel learning method for deep sound recognition: Between-Class learning (BC learning).

Our strategy is to learn a discriminative feature space by recognizing the between-class sounds as between-class sounds.

We generate between-class sounds by mixing two sounds belonging to different classes with a random ratio.

We then input the mixed sound to the model and train the model to output the mixing ratio.

The advantages of BC learning are not limited only to the increase in variation of the training data; BC learning leads to an enlargement of Fisher’s criterion in the feature space and a regularization of the positional relationship among the feature distributions of the classes.

The experimental results show that BC learning improves the performance on various sound recognition networks, datasets, and data augmentation schemes, in which BC learning proves to be always beneficial.

Furthermore, we construct a new deep sound recognition network (EnvNet-v2) and train it with BC learning.

As a result, we achieved a performance surpasses the human level.

Sound recognition has been conventionally conducted by applying classifiers such as SVM to local features such as MFCC or log-mel features BID13 BID27 BID14 .

Convolutional neural networks (CNNs) BID12 , which have achieved success in image recognition tasks BID11 BID22 BID8 , have recently proven to be effective in tasks related to series data, such as speech recognition BID0 BID18 BID4 and natural language processing BID10 .

Some researchers applied CNNs to sound recognition tasks and achieved high performance BID2 BID3 BID25 .The amount and quality of training data and how to feed it are important for machine learning, particularly for deep learning.

Various approaches have been proposed to improve the sound recognition performance.

The first approach is to efficiently use limited training data with data augmentation.

Researchers proposed increasing the training data variation by altering the shape or property of sounds or adding a background noise BID25 BID20 .

Researchers also proposed using additional training data created by mixing multiple training examples BID15 BID24 .

The second approach is to use external data or knowledge.

BID2 proposed learning rich sound representations using a large amount of unlabeled video datasets and pre-trained image recognition networks.

The sound dataset expansion was also conducted BID21 BID17 BID6 .In this paper, as a novel third approach we propose a learning method for deep sound recognition: Between-Class learning (BC learning).

Our strategy is to learn a discriminative feature space by recognizing the between-class sounds as between-class sounds.

We generate between-class sounds by mixing two sounds belonging to different classes with a random ratio.

We then input the mixed sound to the model and train the network to output the mixing ratio.

Our method focuses on the characteristic of the sound, from which we can generate a new sound simply by adding the waveform data of two sounds.

The advantages of BC learning are not limited only to the increase in variation of the training data; BC learning leads to an enlargement of Fisher's criterion BID5 (i.e., the ratio of the between-class distance to the within-class variance) in the feature space, and a regularization of the positional relationship among the feature distributions of the classes.

The experimental results show that BC learning improves the performance on various sound recognition networks, datasets, and data augmentation schemes, in which BC learning proves to be always beneficial.

Furthermore, we constructed a new deep sound recognition network (EnvNet-v2) and trained it with BC learning.

As a result, we achieved a 15.1% error rate on a benchmark dataset ESC-50 (Piczak, 2015b), which surpasses the human level.

We argue that our BC learning is different from the so-called data augmentation methods we introduced above.

Although BC learning can be regarded as a data augmentation method from the viewpoint of using augmented data, the novelty or key point of our method is not mixing multiple sounds, but rather learning method of training the model to output the mixing ratio.

This is a fundamentally different idea from previous data augmentation methods.

In general, data augmentation methods aim to improve the generalization ability by generating additional training data which is likely to appear in testing phase.

Thus, the problem to be solved is the same in both training and testing phase.

On the other hand, BC learning uses only mixed data and labels for training, while mixed data does not appear in testing phase.

BC learning is a method to improve the classification performance by solving a problem of predicting the mixing ratio between two different classes.

To the best of our knowledge, this is the first time a learning method that employs a mixing ratio between different classes has been proposed.

We intuitively describe why such a learning method is effective and demonstrate the effectiveness of BC learning through wide-ranging experiments.

We introduce recent deep learning methods for sound recognition.

Piczak (2015a) proposed to apply CNNs to the log-mel features extracted from raw waveforms.

The log-mel feature is calculated for each frame of sound and represents the magnitude of each frequency area, considering human auditory perception BID4 .

Piczak created a 2-D feature-map by arranging the log-mel features of each frame along the time axis and calculated the delta log-mel feature, which was the first temporal derivative of the static log-mel feature.

Piczak then classified these static and delta feature-maps with 2-D CNN, treating them as a two-channel input in a manner quite similar to the RGB inputs of the image.

The log-mel feature-map exhibits locality in both time and frequency domains BID0 .

Therefore, we can accurately classify this feature-map with CNN.

We refer to this method as Logmel-CNN.Some researchers also proposed methods to learn the sounds directly from 1-D raw waveforms, including feature extraction.

BID2 proposed a sound recognition network using 1-D convolutional and pooling layers named SoundNet and learned the sound feature using a large amount of unlabeled videos (we describe the details of it in the next section).

BID3 also proposed a network using 1-D convolutional and pooling layers, but they stacked more layers.

They reported that the network with 18 layers performed the best.

BID25 proposed a network using both 1-D and 2-D convolutional and pooling layers named EnvNet.

First, EnvNet extracts a frequency feature of each short duration of section with 1-D convolutional and pooling layers and obtain a 2-D feature-map.

Next, it classifies this feature-map with 2-D convolutional and pooling layers in a similar manner to Logmel-CNN.

Learning from the raw waveform is still a challenging problem because it is difficult to learn raw waveform features from limited training data.

However, the performance of these systems is close to that of Logmel-CNN.

We describe the approaches to achieve high sound recognition performance from two views: approaches involving efficient use of limited training data and those involving external data/knowledge.

First, we describe data augmentation as an approach of efficiently using limited training data.

One of the most standard and important data augmentation methods is cropping BID16 BID2 BID25 .

The training data variation increases, and we are able to more Figure 1: Pipeline of BC learning.

We create each training example by mixing two sounds belonging to different classes with a random ratio.

We input the mixed sound to the model and train the model to output the mixing ratio using the KL loss.efficiently train the network when the short section (approximately 1-2 s) of the training sound cropped from the original data, and not the whole section, is input to the network.

A similar method is generally used in the test phase.

Multiple sections of test data are input with a stride, and the average of the output predictions is used to classify the test sound.

BID20 proposed the usage of additional training data created by time stretching, pitch shifting, dynamic range compression, and adding background noise chosen from an external dataset.

Researchers also proposed using additional training data created by mixing multiple training examples.

BID15 applied this method to polyphonic sound event detection.

BID24 applied this method to single-label sound event classification, but only the sounds belonging to the same class were mixed.

Our method is different from both of them in that we employ a mixing ratio between different classes for training.

Next, we describe the approaches of utilizing external data/knowledge.

BID2 proposed to learn rich sound representations using pairs of image and sound included in a large amount of unlabeled video dataset.

They transferred the knowledge of pre-trained large-scale image recognition networks into sound recognition network by minimizing the KL-divergence between the output predictions of the image recognition networks and that of the sound network.

They used the output of the hidden layer of the sound recognition network as the feature when applying to the target sound classification problem.

They then classified it with linear SVM.

They could train a deep sound recognition network (SoundNet8) and achieve a 74.2% accuracy on a benchmark dataset, ESC-50 (Piczak, 2015b) , with this method.

In this section, we propose a novel learning method for deep sound recognition BC learning.

Fig. 1 shows the pipeline of BC learning.

In standard learning, we select a single training example from the dataset and input it to the model.

We then train the model to output 0 or 1.

By contrast, in BC learning, we select two training examples from different classes and mix these two examples using a random ratio.

We then input the mixed data to the model and train the model to output the mixing ratio.

BC learning uses only mixed data and labels, and thus never uses pure data and labels for training.

Note that we do not mix any examples in testing phase.

First, we provide the details of BC learning in Section 3.2.

We mainly explain the method of mixing two sounds, which should be carefully designed to achieve a good performance.

Then, in Section 3.3, we explain why BC learning leads to a discriminative feature space.

BC learning optimizes a model using mini-batch stochastic gradient descent the same way the standard learning does.

Each data and label of a mini-batch is generated by mixing two training examples belonging to different classes.

Here, we describe how to mix two training examples.

Let x 1 and x 2 be two sounds belonging to different classes randomly selected from the training dataset, and t 1 and t 2 be their one-hot labels.

Note that x 1 and x 2 may have already been preprocessed or applied data augmentation, and they have the same length as that of the input of the network.

We generate a random ratio r from U (0, 1) , and mix two sets of data and labels with this ratio.

We mix two labels simply by r t 1 + (1 − r) t 2 , because we aim to train the model to output the mixing ratio.

We then explain how to mix x 1 and x 2 .

The simplest method is r x 1 + (1 − r) x 2 .

However, the following mixing formula is slightly better, considering that sound energy is proportional to the square of the amplitude: DISPLAYFORM0 However, auditory perception of a sound mixed with Eqn.

FORMULA0 would not be x 1 : x 2 = r : (1 − r) if the difference of the sound pressure level of x 1 and x 2 is large.

For example, if the amplitude of x 1 is 10 times as large as that of x 2 and we mix them with 0.2 : 0.8, the sound of x 1 would still be dominant in the mixed sound.

In this case, training the model with a label of {0.2, 0.8} is inappropriate.

We then consider using a new coefficient p(r, G 1 , G 2 ) instead of r , and mix two sounds by DISPLAYFORM1 , where G 1 and G 2 is the sound pressure level of x 1 and DISPLAYFORM2 respectively.

We define p so that the auditory perception of the mixed sound becomes r : (1 − r).

We hypothesize that the ratio of auditory perception for the network is the same as that of amplitude because the main component functions of CNNs, such as conv/fc, relu, max pooling, and average pooling, satisfy homogeneity (i.e., f (α x) = αf (x)) if we ignore the bias.

We then set up an equation about the ratio of amplitude p · 10 DISPLAYFORM3 (1 − r) using unit conversion from decibels to amplitudes and solve it for p. Finally, we obtain the proposed mixing method: DISPLAYFORM4 We show this mixing method performs better than Eqn.

FORMULA0 in the experiments.

We calculate the sound pressure level G 1 and G 2 using A-weighting, considering that human auditory perception is not sensitive to low and high frequency areas.

We can also use simpler sound pressure metrics such as root mean square (RMS) energy instead of an A-weighting sound pressure level.

However, the performance worsens, as we show in the experiments.

We create short windows (∼ 0.1 s) on the sound and calculate a time series of A-weighted sound pressure levels {g 1 , g 2 , . . .

, g t } .Then, we define G as the maximum of those time series ( G = max{g 1 , g 2 , . . .

, g t } ).

We define the f and θ as the model function and the model parameters, respectively.

We input the generated mini-batch data {x DISPLAYFORM0 to the model and obtain the output {f θ ( DISPLAYFORM1 .

We expect that our mini-batch ratio labels {t DISPLAYFORM2 represent the expected class probability distribution.

Therefore, we use the KL-divergence between the labels and the model outputs as the loss function, instead of the usual cross-entropy loss.

We optimize KL-divergence with back-propagation and stochastic gradient descent because it is differentiable: DISPLAYFORM3 DISPLAYFORM4 where m is the number of classes, and η is the learning rate.

BC leaning leads to an enlargement of Fisher's criterion (i.e., the ratio of the between-class distance to the within-class variance).

We explain the reason in Fig. 2 DISPLAYFORM0 Figure 2: BC learning enlarges Fisher's criterion in the feature space, by training the model to output the mixing ratio between two classes.

We hypothesize that a mixed sound mixr(x1, x2) is projected into the point near the internally dividing point of f (x1) and f (x2) , considering the characteristic of sounds.

Middle: When Fisher's criterion is small, some mixed examples are projected into one of the classes, and BC learning gives a large penalty.

Right: When Fisher's criterion is large, most of the mixed examples are projected into betweenclass points, and BC learning gives a small penalty.

Therefore, BC learning leads to such a feature space.

Besides, we can generate a new sound simply by adding the waveform data of two sounds, and humans can recognize both of two sounds and perceive which of two sounds is louder or softer from the mixed sound.

Therefore, it is expected that an internally dividing point of the input space almost corresponds to that of the semantic feature space, at least for sounds.

Then, the feature distribution of the mixed sounds of class A and class B with a certain ratio would be located near the internally dividing point of the original feature distribution of class A and B, and the variance of the feature distribution of the mixed sounds is proportional to the original feature distribution of class A and B. To investigate whether this hypothesis is correct or not, we visualized the feature distributions of the standard-learned model using PCA.

We used the activations of fc6 of EnvNet BID25 against training data of ESC-10 (Piczak, 2015b).

The results are shown in FIG1 .

The magenta circles represent the feature distribution of the mixed sounds of dog bark and rain with a ratio of 0.8 : 0.2, and the black dotted line represents the trajectory of the feature when we input a mixture of two particular sounds to the model changing the mixing ratio from 0 to 1.

This figure shows that the mixture of two sounds is projected into the point near the internally dividing point of two features, and the features of the mixed sounds are distributed between two classes, as we expected.

If Fisher's criterion is small, the feature distribution of the mixed sounds becomes large, and would have a large overlap with one or both of the feature distribution of class A and B (Fig. 2(middle) ).

In this case, some mixed sounds are projected into one of the classes as shown in this figure, and the model cannot output the mixing ratio.

BC learning gives a penalty to this situation because BC learning trains a model to output the mixing ratio.

If Fisher's criterion is large, on the other hand, the overlap becomes small (Fig. 2(right) ).

The model becomes able to output the mixing ratio, and BC learning gives a small penalty.

Therefore, BC learning enlarges Fisher's criterion between any two classes in the feature space.

We expect that BC learning also has the effect of regularizing the positional relationship among the class feature distributions.

In standard learning, there is no constraint on the positional relationship among the classes, as long as the features of each two classes are linearly separable.

We found that a standard-learned model sometimes misclassifies a mixed sound of class A and class B as a class other than A or B. FIG2 (lower left) shows an example of transition of output probability of standard-learned model when we input a mixture of two particular training sounds (dog bark and rain) to the model changing the mixing ratio from 0 to 1.

The output probability of dog bark monotonically increases and that of rain monotonically decreases as we expected, but the model classifies the mixed sound as baby cry when the mixing ratio is within the range of 0.45 -0.8.

This is an undesirable state because there is little possibility that a mixed sound of two classes becomes a sound of other classes.

In this case, we assume that the features of each class are distributed as in FIG2 (upper left).

The decision boundary of class C appears between class A and class B, and the trajectory of the features of the mixed sounds crosses the decision boundary of class C.BC learning can avoid the situation in which the decision boundary of other class appears between two classes, because BC learning trains a model to output the mixing ratio instead of misclassifying the mixed sound as different classes.

We show the transition of the output probability in FIG2 (lower right), when using the same two examples as that used in FIG2 (lower left).

We assume that the features of each class are distributed as in FIG2 (upper right).

The feature distributions of the three classes make an acute-angled triangle, and the decision boundary of class C does not appear between class A and class B. Note that it is assumed that the dimension of the feature space is greater than or equal to the number of classes minus 1.

However, because the network is generally designed as such, it is not a problem.

In this way, BC learning enlarges Fisher's criterion, and at the same time, regularizes the positional relationship among the classes in the feature space.

Hence, BC learning improves the generalization ability.

In this section, we train various types of sound recognition networks with both standard and BC learning, and demonstrate the effectiveness of BC learning.

Datasets.

We used ESC-50, ESC-10 (Piczak, 2015b), and UrbanSound8K BID21 to train and evaluate the models.

ESC-50, ESC-10, and UrbanSound8K contain a total of 2,000, 400, and 8,732 examples consisting of 50, 10, and 10 classes, respectively.

We removed completely silent sections in which the value was equal to 0 at the beginning or end of examples in the ESC-50 and ESC-10 datasets.

We converted all sound files to monaural 16-bit WAV files.

We evaluated the performance of the methods using a K-fold cross-validation (K = 5 for ESC-50 and ESC-10, and K = 10 for UrbanSound8K), using the original fold settings.

We performed cross-validation 5 times for ESC-50 and ESC-10, and showed the standard error.

Preprocessing and data augmentation.

We used a simple preprocessing and data augmentation scheme.

Let T be the input length of a network [s].

In the training phase, we padded T /2 s of zeros on each side of a training sound and randomly cropped a T -s section from the padded sound.

We mixed two cropped sounds with a random ratio when using BC learning.

In the testing phase, we also padded T /2 s of zeros on each side of a test sound and cropped 10 T -s sections from the padded sound at regular intervals.

We then input these 10 crops to the network and averaged all softmax outputs.

Each input data was regularized into a range of from −1 to +1 by dividing it by 32,768, that is, the full range of 16-bit recordings.

Learning settings.

All models were trained with Nesterov's accelerated gradient using a momentum of 0.9, weight decay of 0.0005, and mini-batch size of 64.

The only difference in the learning settings between standard and BC learning is the number of training epochs.

BC learning tends to require more training epochs than does standard learning, while standard learning tends to overfit with many training epochs.

To validate the comparison, we first identified an appropriate standard learning setting for each network and dataset (details are provided in the appendix), and we doubled the number of training epochs when using BC learning.

Later in this section, we examine the relationship between the number of training epochs and the performance.

First, we trained various types of existing networks.

We selected EnvNet BID25 as a network using both 1-D and 2-D convolutions, SoundNet5 BID2 and M18 BID3 as networks using only 1-D convolution, and Logmel-CNN (Piczak, 2015a) + BN as a network using log-mel features.

Logmel-CNN + BN is an improved version of Logmel-CNN that we designed in which, to convolutional layers, we apply batch normalization BID9 to the output and remove the dropout BID23 .

Note that all networks and training codes are our implementation using Chainer v1.24 BID26 .The results are summarized in the upper half of TAB1 .

Our BC learning improved the performance of all networks on all datasets.

The performance on ESC-50, ESC-10, and UrbanSound8K was improved by 4.5-6.4%, 1.5-4.0%, and 1.8-4.8%, respectively.

We show the training curves of EnvNet on ESC-50 in FIG3 .

Note that the curves show the average of all trials.

To investigate the effectiveness of BC learning on deeper networks, we constructed a deep sound recognition network based on EnvNet, which we refer to as EnvNet-v2, and trained it with both standard and BC learning.

The main differences between EnvNet and EnvNet-v2 are as follows: 1) EnvNet uses a sampling rate of 16 kHz for the input waveforms, whereas EnvNet-v2 uses 44.1 kHz; and 2) EnvNet consists of 7 layers, whereas EnvNet-v2 consists of 13 layers.

A detailed configuration is provided in the appendix.

The results are also shown in the upper half of TAB1 , and the training curves on ESC-50 are given in FIG3 .

The performance was also improved with BC learning, and the degree of the improvement was greater than other networks (7.4%, 3.6%, and 7.5% on ESC-50, ESC-10, and UrbanSound8K, respectively).

The error rate of EnvNet-v2 trained with BC learning was the lowest on ESC-50 and UrbanSound8K among all the models including Logmel-CNN + BN, which uses powerful hand-crafted features.

Moreover, the error rate on ESC-50 (18.2%) is comparable to human performance reported by Piczak (2015b) (18.7%).

The point is not that our EnvNet-v2 is well designed, but that our BC learning successfully elicits the true value of a deep network.

We compared the performances of standard and BC learning when using a stronger data augmentation scheme.

In addition to zero padding and random cropping, we used scale augmentation with a factor randomly selected from [0.8, 1.25] and gain augmentation with a factor randomly selected from [−6 dB, +6 dB].

Scale augmentation was performed before zero padding (thus, before mixing when employing BC learning) using linear interpolation, and gain augmentation was performed just before inputting to the network (thus, after mixing when using BC learning).The results for EnvNet-v2 are shown in the lower half of TAB1 , and the training curves on ESC-50 are given in FIG3 .

With BC learning, the performance was significantly improved even when we used a strong data augmentation scheme.

Furthermore, the performance on ESC-50 (15.1%) surpasses the human performance (18.7%).

BC learning performs well on various networks, datasets, and data augmentation schemes, and using BC learning is always beneficial.

EnvNet standard EnvNet BC (ours) Figure 6 : Error rate vs. # of training epochs.

We investigated the relationship between the performance and the number of training epochs, because the previously described experiments were conducted using different numbers of training epochs (we used 2× training epochs for BC learning).

Fig. 6 shows the error rate of EnvNet on ESC-10 and ESC-50 with various numbers of training epochs.

This figure shows that for standard learning, approximately 600 training epochs are sufficient for both ESC-10 and ESC-50.

However, this number is insufficient for BC learning.

Although BC learning performed better than standard learning with 600 epochs, improved performance was achieved when using more training epochs (900 and 1,200 epochs for ESC-10 and ESC-50, respectively).

However, if the number of training epochs was small, the performance of BC learning was lower than that of standard learning.

We can say that BC learning always improves the performance as long as we use a sufficiently large number of training epochs.

Additionally, the number of training epochs needed would become large when there are many classes.

To understand the part that is important for BC learning, we conducted an ablation analysis.

We trained EnvNet on ESC-50 using various settings.

All results are shown in TAB2 .

We also performed 5-fold cross-validation five times and show the standard error.

Mixing method.

We compared the mixing formula (Eqn.

1 vs. Eqn.

2, which consider the sound pressure levels of two sounds) and the calculation method for sound pressure levels (RMS vs. Aweighting).

As shown in TAB2 , the proposed mixing method using Eqn.

2 and A-weighting performed the best.

Considering the difference in the sound pressure levels is important for BC learning, and the method used to define the sound pressure levels also has an effect on the performance.

Label.

We compared the different labels that we applied to the mixed sound.

As shown in TAB2 , the proposed ratio label of t = r t 1 + (1 − r) t 2 performed the best.

When we applied a single label of the dominant sound (i.e., t = t 1 if r > 0.5, otherwise t = t 2 ) and trained the model using softmax cross entropy loss, the performance was improved compared to that of standard learning.

When we applied a multi-label (i.e., t = t 1 + t 2 ) and trained the model using sigmoid cross entropy loss, the performance was better than when using a single label.

However, the performance was worse than when using our ratio label in both cases.

The model can learn the between-class examples more efficiently when using our ratio label.

Number of mixed classes.

We investigated the relationship between the performance and the number of sound classes that we mixed.

N = 1 in TAB2 means that we mixed two sounds belonging to the same class, which is similar to BID24 .

N = 1 or 2 means that we completely randomly selected two sounds to be mixed; sometimes these two sounds were the same class.

N = 2 or 3 means that we mixed two and three sounds belonging to different classes with probabilities of 0.5 and 0.5, respectively.

When we mixed three sounds, we generated a mixing ratio from Dir(1,1,1) and mixed three sounds using a method that is an extended version of Eqn.

2 to three classes.

As shown in TAB2 , the proposed N = 2 performed the best.

N = 2 or 3 also achieved a good performance.

It is interesting to note that the performance of N = 3 is worse than that of N = 2 despite the larger variation in training data.

We believe that the most important factor is not the training data variation but rather the enlargement of Fisher's criterion and the regularization of the positional relationship among the feature distributions.

Mixing more than two sounds leads to increased training data variation, but we expect that cannot efficiently achieve them.

Where to mix.

Finally, we investigated what occurs when we mix two examples within the network.

We input two sounds to be mixed into the model and performed the forward calculation to the mixing point.

We then mixed the activations of two sounds at the mixing point and performed the rest of the forward calculation.

We mixed two activations h 1 and h 2 simply by r h 1 + (1 − r) h 2 .

As shown in TAB2 , the performance tended to improve when we mixed two examples at the layer near the input layer.

The performance was the best when we mixed in the input space.

Mixing in the input space is the best choice, not only because it performs the best, but also because it does not require additional forward/backward computation and is easy to implement.

We proposed a novel learning method for deep sound recognition, called BC learning.

Our method improved the performance on various networks, datasets, and data augmentation schemes.

Moreover, we achieved a performance surpasses the human level by constructing a deeper network named EnvNet-v2 and training it with BC learning.

BC learning is a simple and powerful method that improves various sound recognition methods and elicits the true value of large-scale networks.

Furthermore, BC learning is innovative in that a discriminative feature space can be learned from betweenclass examples, without inputting pure examples.

We assume that the core idea of BC learning is generic and could contribute to the improvement of the performance of tasks of other modalities.

This work was supported by JST CREST Grant Number JPMJCR1403, Japan.

A LEARNING SETTINGS TAB3 shows the detailed learning settings of standard learning.

We trained the model by beginning with a learning rate of Initial LR, and then divided the learning rate by 10 at the epoch listed in LR schedule.

To improve convergence, we used a 0.1× smaller learning rate for the first Warmup epochs.

We then terminated training after # of epochs epochs.

We doubled # of epochs and LR schedule when using BC learning, as we mentioned in the paper.

Table 4 shows the configuration of our EnvNet-v2 used in the experiments.

EnvNet-v2 consists of 10 convolutional layers, 3 fully connected layers, and 5 max-pooling layers.

We use a sampling rate of 44.1 kHz, which is the standard recording setting, and a higher resolution than existing networks BID16 BID2 BID3 BID25 , in order to use rich high-frequency information.

The basic idea is motivated by EnvNet BID25 , but the advantages of other successful networks are incorporated.

First, we extract short-time frequency features with the first two temporal convolutional layers and a pooling layer (conv1-pool2).

Second, we swap the axes and convolve in time and frequency domains with the later layers (conv3-pool10).

In this part, we hierarchically extract the temporal features by stacking the convolutional and pooling layers with decreasing their kernel size in a similar manner to SoundNet BID2 .

Furthermore, we stack multiple convolutional layers with a small kernel size in a similar manner to M18 BID3 and VGG BID22 , to extract more rich features.

Finally, we produce output predictions with fc11-fc13 and the following softmax activation.

Single output prediction is calculated from 66,650 input samples (approximately 1.5 s at 44.1 kHz).

We do not use padding in convolutional layers.

We apply ReLU activation for all the hidden layers and batch normalization BID9 to the output of conv1-conv10.

We also apply 0.5 of dropout BID23 to the output of fc11 and fc12.

We use a weight initialization of BID7 for all convolutional layers.

We initialize the weights of each fully connected layer using Gaussian distribution with a standard deviation of √ 1/n, where n is the input dimension of the layer.

@highlight

We propose an novel learning method for deep sound recognition named BC learning.

@highlight

Authors defined a new learning task that requires a DNN to predict mixing ratio between sounds from two different classes to increase discriminitive power of the final learned network.

@highlight

Proposes a method to improve the performance of a generic learning method by generating "in between class" training samples and presents the basic intuition and necessity of the proposed technique.