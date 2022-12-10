There has been an increasing use of neural networks for music information retrieval tasks.

In this paper, we empirically investigate different ways of improving the performance of convolutional neural networks (CNNs) on spectral audio features.

More specifically, we explore three aspects of CNN design: depth of the network, the use of residual blocks along with the use of grouped convolution, and global aggregation over time.

The application context is singer classification and singing performance embedding and we believe the conclusions extend to other types of music analysis using convolutional neural networks.

The results show that global time aggregation helps to improve the performance of CNNs the most.

Another contribution of this paper is the release of a singing recording dataset that can be used for training and evaluation.

Deploying deep neural networks to solve music information retrieval problems has benefited from advancements in other areas such as computer vision and natural language processing.

In this paper, experiments are designed to investigate whether a few of the recent signature advancements in the deep learning community can improve the learning capability of deep neural networks when applied on time-frequency representations.

Because time-frequency representations are frequently treated as 2-D images similarly to image input for computer vision models, convolution layers are popular choices as the first processing layers for time-frequency representations in audio and music analysis applications.

One of the recent convolutional layer variants is the residual neural network with a bottleneck design , ResNet.

Another variant built upon ResNet is to use grouped convolution inside the bottleneck as a generalization of the Inception Net BID12 BID22 , ResNeXt.

These two variants have enabled more deepening of the convolutional layers of deep neural networks.

Most existing music information retrieval research using convolutional neural networks (CNNs), utilizes vanilla convolutional layers with no more than 5 layers.

In this paper, the two convolution layer variants mentioned and a deeper architecture with more than 5 convolution layers is proposed and shown to be effective on audio time-frequency representations.

Conceptually, convolution layers take care of learning local patterns (neighboring pixels in images or time frame/frequency bins in time-frequency representations) presented in the input matrices.

After learning feature maps from the convolution layers, one of the reoccurring issues, when the input is a time-frequency representation, is how to model or capture temporal relations.

Recurrent neural networks has been used to solve this problem BID6 BID1 BID3 BID4 .

Recent developments from natural language processing in attention mechanisms BID0 BID3 BID15 provide a different approach to model temporal dependencies and relations.

In this paper, the attention mechanism is viewed as a special case of a global aggregation operation along the timeaxis that has learnable parameters.

Typical aggregation operations such as average or max have no learnable parameters.

The effects of global aggregation along the time axis using either average, max or the attention mechanism is investigated experimentally.

Two specific applications are investigated in this paper: 1) singer classification of monophonic recordings, and 2) singing performance embedding.

The goal of singer classification is to predict the singer's identity given an audio recording as input.

A finite set of possible singers is considered so this is a classification task.

In singer performance embedding the goal is to create an embedding space in which singers with similar styles can be projected to be closer to each other compared to singers with different styles.

Ideally, it should be possible to identify "singing style" or "singing characteristics" by examining (and listening to) the clusters formed from the projections of audio recordings onto the embedding space.

Many tasks in music and audio analysis can be formulated in a similar way, in which similarity plays an essential role, therefore we believe that the conclusions of this paper generalize to other audio and music tasks.

The main challenge and interesting point about this application context is how to isolate the "singer effect" from the "song effect".

Classic hand-crafted audio features capture general aspects of similarity.

When the same song is performed by different singers audio-based similarity tends to be higher than when different songs are performed -i.e the "song effect" BID14 .

In order to effectively model singing we need to learn a representation that emphasizes singer similarity while at the same time reduces the effect of song similarity.

As an analogy, consider the computer vision problem of face identification.

When learning representations for this task we want the information about the identity of the face to be minimally affected by the effect of the environment and pose.

The interfering "song effect " is even more dominant in the singing voice case than that of the environment/pose effect in face recognition.

Extending this analogy with computer vision, singing performance embedding is analogous to the use of learning an embedded space for face verification BID5 BID17 .

In this approach, an embedded space of faces is learned with the goal of having pictures of the same person close to each other, and having pictures of different persons away from each other in the learned embedding space.

This is accomplished by utilizing a siamese neural network instead of a classifier BID5 BID8 BID16 .

The large amount of identities make the use of a classifier impractical.

By learning an embedding space for singing voice audio recordings that places recordings of the same identity closer to each other, and pushes the ones with different identities away from each other, ideally "singing style" or "singing characteristics" can be identified by examining (and listening to) the clusters formed from the embeddings of audio recordings in the learned embedding space.

For both the singer identity classification and the singing performance embedding, we employ an architecture that uses CNNs to extract features followed by a global aggregation layer after which fully connected dense layers are used.

The difference between the architectures used for these two tasks is that, for singer identity classification, the output layer is the standard softmax layer that outputs classification probabilities for each singer included in the dataset, but for the singing performance embedding, the output layer is a fully connected linear layer that will embed each input sample into a fixed length vector after which a copy of the network is used to construct a siamese architecture to learn the embedding space.

Practically, having a model that embeds singing recordings into short fixed length vectors enables the possibility of fastening the similarity comparison of two long spectrogram sequences (differ in lengths) by calculating the Euclidean distance between their fixed length embedding vectors BID16 .

This allows a large database of singing recordings to be queried by input query singing recordings more efficiently.

In order to evaluate the singing performance embedding model in an unbiased way (not biasing towards the collection of songs sang by a singer), a new set of "balanced" singing recordings are gathered and released.

The newly released dataset is an addition to the existing DAMP (Smith, 2013) data set of monophonic vocal music performances.

The paper is structured as follows.

In section 2, the details of the neural network constructing blocks used in the experiments are described.

The dataset used and the experiment details are disclosed in section 3.

Discussions and conclusions are in Sec.4.

The neural network architectures used in the experiments follow a general design pattern depicted in Figure 1 .

The general design pattern is to feed the input time-frequency features as 2-D images into convolutional layers/blocks, then feed the extracted feature maps to a global time-wise aggregation layer.

The output from the global time-wise aggregation layer is fed into dense layers, followed by the output layer.

The details of each construction block are described below.

The basic convolution layer being used is the vanilla convolution layer that has shared weights and tied biases across channels without any modification.

The other variant being used in our experiments is the residual network design with the bottleneck block introduced in ResNet .

This variant is extended by using the grouped convolutional block, introduced in ResNeXt BID22 , on top of the ResNet.

Depictions of the vanilla convolution building block, the ResNet, and ResNeXt are shown in FIG0 .

Let the outlets in FIG0 be y, inlets be x, and f , g, h be convolution operations.

The vanilla convolutional block (a) in FIG0 would be y = g(f (x)), while the ResNet bottleneck block (b) is y = x + f (g(h(x))) and the ResNeXt bottleneck block is y = x + Γ(x) with Γ(·) being the grouped convolution consisting of series of sliced f (g(h(·))) operations over the channel axis of the input.

Under the ResNeXt configuration, the ResNet configuration is a special case where the cardinality parameter equals 1 BID22 .

A max pooling layer with pool size (2, 2), and stride of (2, 2) is placed between convolutional blocks in the following way: The first convolutional layer is followed immediately by a max pooling layer, while for all the remaining layers the max pooling layers are inserted between every two consecutive convolutional layers/blocks.

A distinction between the terms convolution layer and block needs to be made here.

A convolutional layer refers to a single vanilla convolution layer, while a convolutional block refers to any of the three architecture patterns show in FIG0 .

In TAB0 , in the column for the number of CNN filters, each number represents the number of output channels for each convolutional layer or block, with normal text for layer and bold text for block.

Batch normalizations are applied after each non-linearity activation throughout the convolutional layer/blocks.

Global Time-Wise Aggregation Layer Dense Layers Output Layer Input Figure 1 : An overview of the neural network architecture used in this paper.

Before feeding the output of the convolutional layers to the global time-wise aggregation, the 3 − D feature map having the shape (# of channels, # of time frames, # of frequency bins) as their dimensions will be reshaped as 2-D matrices having the shape (# of time frames, # of channels× # of frequency bins)

Originally, the attention mechanism was introduced for sequence-to-sequence learning BID0 in an RNN architecture, that allows the prediction at each time-step to access information from every step in the input hidden sequence in a weighted way.

Since the experiments done in this paper do not need sequence-to-sequence prediction, the feed-forward version of attention proposed in BID15 is used instead of the original one.

The feed-forward attention is formulated as follows:

Given the input matrix X ∈ R N ×D representing N frames of D dimensional feature vectors, a weight vector σ ∈ R N over the time-steps is calculated by DISPLAYFORM0 where DISPLAYFORM1 and f is a non-linear function (tanh for the experiments done in this paper), and w ∈ R D and b ∈ R are the learnable parameters, which can be learned by back-propagation.

The outputX of the feed-forward attention layer is then calculated viâ and whereX can be considered a weighted average of X with weights σ, determined by the learnable parameters w and b. This attention operation can also be viewed as an aggregation operation over the time-axis similar to max or average.

The idea of aggregation over a specific axis could then be generalized by having the feed-forward attention, max and average all in the same family, except that the later two have no learnable parameters.

This family of operations is different from the standard max/average pooling in convolution layers, in that the aggregation is global to the scope of the the input sample i.e the aggregation will reduce the dimension of the aggregation axis to 1.

A specific realization of the network architecture including both convolutional and the global aggregation parts can be found in Appendix B. DISPLAYFORM2

The two tasks explored in this paper are singer identity classification and singing performance embedding.

In terms of experimentation with different hyper parameters and network architectures, the singer classification problem provides clear evaluation criteria in terms of model performance.

That way different hyper parameters and architectural choices can be compared to each other.

On the other hand, the embedding task allows a more exploratory way to understand the input in the sense that it is the spatial relationships between the embedded samples that are interesting to us.

For both tasks, numerical evaluation metrics, as well as plots of the embedded samples from the singing performance embedding are provided in order for readers to examine the results both quantitatively and qualitatively.

The dataset being used for the singer identity classification is the DAMP dataset 1 .

The DAMP dataset has a total of 34620 solo singing recordings by 3462 singers with each singer having 10 recordings.

The collections of songs sang by each singer are different, and some singers sing the same song multiples times.

Therefore the DAMP dataset is "unbalanced", and making it difficult for the learning algorithm not to be biased to the singer-specific collection of songs when learning to predict the singer identity.

Therefore an additional dataset with each singer singing the same collection of songs available for training and evaluation is collected and released.

The set of added collections of solo singing recordings is named the DAMP-balanced dataset.

DAMP-balanced has a total of 24874 singing recordings sang by 5429 singers.

The song collection of the DAMP-balanced has 14 songs.

The structure of the DAMP-balanced is that the last 4 songs are designed to be the test set, and the first 10 songs could be partitioned into any 6/4 train/validation split (permutation) that the singers in train and validation set sang the same 6/4 songs collections according to the 6/4 split (the number of total recordings for train and validation set are different from split to split, since there are different number of singers that all sang the same 6/4 split for different split).

The DAMPbalanced dataset is suitable for the singing performance embedding task while the original DAMP dataset can be used to train singer identity classification algorithms.

The song list and detailed descriptions of the DAMP-balanced is provided in Appendix A.

The input to the neural networks are time-frequency representations extracted from raw audio signals.

These time-frequency representations are obtained by applying the short-time Fourier transform that extracts frequency information for each short time window analyzed.

As a result, most time-frequency representations take the form of 2-D matrices with one axis representing time while the other axis represents frequencies.

The entries [j, i] of the matrix represent the intensity of a particular frequency i corresponding to a particular time frame j.

In this paper, the Mel-scaled magnitude spectrogram (Mel-spectrogram) ) is used as the input feature to the neural network.

Mel-spectrogram are used as input to neural network tasks in BID3 BID7 BID4 .

The other common choice of audio time-frequency input, the constant-Q transformed spectrogram (CQT), which is used extensively in music information retrieval tasks BID20 due to its capability of preserving the constant octave relationships between frequency bins (log-scaled frequency bins).

Since all neural network configurations using CQT perform worse than their Mel-spectrogram versions, only a few representative results of using CQT are shown in TAB0 .

The reason why CQT works worse is that although the CQT preserves a linear relationships of the intervals of different pitches, the linear relationships do not apply to the distances between different harmonics of one pitch.

Since the audio recording being analyzed here only has one single singing voice at each time frame, the constant octave relationship does not help the neural networks learning the time-frequency patterns for singing voices.

The audio recordings are all re-sampled to have 22050Hz sampling rate, then the Mel-scaled magnitude spectrograms are obtained using a Fast Fourier Transform (FFT) with a length of 2048 samples, hop size of 512 samples, a Hanning window and 96 Mel-scaled frequency bins.

The extracted Melspectrogram is squared to obtain the power spectrogram which is then transformed into decibels (dB).

The values below −60dB are clipped to be zero and an offset is added to the whole power spectrogram in order to have values between 0 and 60.For both tasks, each singing performance audio recording is transformed to a Mel-spectrogram as described above.

The Mel-spectrogram of each recording is then chopped into overlapping matrices each of which has a duration of 6 seconds (256 time steps) and 20% hop size.

For both tasks the gradient decent is optimized by ADAM BID11 ) with a learning rate 0.0001 and a batch size of 32.

A drop out of 10% is applied at the last fully connected dense layers.

L 2 weight regularizations with a weight 1e − 6 are applied on all the learnable weights in the neural network.

The above hyper parameters are chosen by the Bayesian optimization package SPEARMINT BID19 .

For both tasks, an early stopping test on the validation set is applied every 50 epochs.

For the singer identity classification, the patience is 300 epochs with at least 99.5% improvement, and the patience for the singing performance embedding task is 1000.

The non-linear activation function used in all convolution layers and fully connected layers is the rectified linear unit activation function.

The convolutional filter sizes are (10, 10) for the first convolutional layer and (5, 5) for all subsequent convolutional layers.

For the fully connected dense layer, 3 layers with each having 1024 hidden units are used before the last output layer.

A subset of 46 singers (23 males and 23 females) corresponding to 460 solo singing recordings from the DAMP dataset is selected for the singer classification problem.

A 10-fold cross validation is used to obtain the test accuracies for different models with each fold using 1 recording from each singer as the test set, while the training is performed on the rest 9 recordings with 1 of them selected randomly as the validation set for early stopping.

For the classification task we explore different combinations of neural network configurations in terms of using either the vanilla CNN or ResNeXt building blocks.

Also different number of layers and different types of aggregation such as max, average, feed-forward attention or no global aggregation are also investigated.

A baseline SVM classifier is also included by having the mean and standard deviation of chroma, MFCC, spectral centroid, spectral roll-off, and spectral flux BID2 extracted from each ∼ 6 second clip as the input.

The experimental results and associated measures of different models are displayed in TAB0 .

The number of convolution filters is chosen so that the total number of parameters are on the same scale between different configurations.

From TAB0 , it can be seen that the baseline method achieved 27% accuracy which is above the random prediction of 2.2% ( FORMULA0 ), while all the neural network models far exceeded the baseline by at least 35%.

For all the neural network models, the use of any global aggregation method improved the performance by 5% ∼ 10%.

Among the neural network models, global aggregation with average or feed-forward attention has slightly better performance than max except for the shallower CNN.

For the singing performance embedding experiment, a subset of 6/4/4 train/validation/test split from the DAMP-balanced is used 2 .

The total number of recordings and singers for this specific split are 276/88/224 and 46/22/56 respectively.

We would like an embedding space that places recordings by the same singer closer to each other and pushes recordings by different singers away from each other, and a siamese neural network architecture BID5 BID8 BID16 ) is used.

The inner twin neural network is constructed following the same principle described earlier in section 2.

The embedding dimension for the linear fully connected output layer is chosen to be 16 by SPEARMINT.

Since a siamese network learns the embedding by shortening or lengthening the distance between pairs of embedded vectors based on their label, pairs of samples from the dataset are arranged and labeled.

Denote a pair of samples by x 1 , x 2 ∈ R D , and y a binary label that equals 1 when x 1 , x 2 have the same identity and equals 0 when their identities are different.

The distance metric optimized over the siamese network in this experiment is the squared euclidean distance DISPLAYFORM0 then the contrastive loss BID5 BID8 BID16 ) is used as the optimization goal and is defined as DISPLAYFORM1 where G is the non-linear function that represents the neural network and m is the target margin between embedded vectors having different identities, and m = 1 throughout the experiments.

To train the siamese networks, pairs of chopped samples from the same singer or different ones are randomly sampled in a 1 : 1 ratio and fed into the siamese networks.

The contrastive losses on the test set for different network configurations are shown in TAB1 .

The cardinalities for the ResNeXt configurations in TAB1 are 4.The training and validation error over epochs are plotted in FIG1 .

The observation from the training/validation plots are that, 1) feed-forward attention and average aggregation tend to overfit the data more than max and no aggregation by looking at training errors, 2) feed-forward attention and average aggregations reach early stopping earlier than max and no aggregation by looking at the best validation epoch, 3) Shallow architectures work slightly better than deeper ones if their number of parameters are on the same scale.

Results showing qualitative characteristic of the embedding are shown in Figure 4 .

In Figure 4 , the embeddings of 40 performances sang by 10 singers with each singer sang the same 4 songs from the test split are plotted.

The embedding of a performance is obtained by taking the mean of the embeddings from all the chopped input samples of that performance.

A comparison is made between the embeddings from the shallow ResNeXt architecture with/without feed-forward attention and the handcrafted features used in the baseline case for singer classification.

Both the embeddings and the extracted handcrafted audio features are projected down to a 2-D space by t-SNE BID13 .

It is obvious that the baseline handcrafted audio feature captured the "song" effect while the learned embeddings from our singing performance embedding experiment were able to group together the performances by the same singers while invariant to the "song" effect.

The t-SNE projections of performed 6-second clips before summarized into songs are shown in FIG4 in Appendix B. To have another quantitative assessment of the embeddings, leave-one-out k-nearest neighbor classifications using the embedded 16-dimensional performance vectors are used as training points.

For each k and each network configuration, every sample is used as test sample once and the classification accuracies are obtained by averaging over the outcomes of every test sample for all k and network configurations.

For the k-nearest neighbor singer classification, all the 224 performances from 56 singers are used.

The classification results with multiple ks among the shallow ResNeXt configurations with/without feed-forward attention and the handcrafted features are shown in Figure 5 .

In addition, k-nearest neighbor classifications on performed songs are also conducted to demonstrate the "song effect".

From the k-nearest neighbor classification results on singers and songs, it is evidence that the "song" effect exists and singing performance embedding learning is able to dilute the "song" effect while extracting features that are more relevant in terms of characterizing singers.

Also the feed-forward global aggregation helped the enhancement of "singer style" while reducing "song effect" slightly by looking at the k-nearest neighbor classification accuracies.

It is worth mentioning that the k-nearest neighbor classification on performed songs is only possible due to the "balanced" nature of the dataset.

In this paper, empirical investigations into how recent developments in the deep learning community could help solving singer identification and embedding problems were conducted.

From the experiment results, the obvious take away is that global aggregation over time improves performance by a considerable margin in general.

The performances among the three aggregation strategies; max, average and feed-forward attention, are very close.

The advantage of using feedforward attention from observing the experiment results is that it accelerates the learning process compared to other non-learnable global aggregations.

One way to explain such observation is that the feed-forward attention layer learns a "frequency template" for each convolutional channel fed into it.

These "frequency templates" are encoded in w and enable each convolutional channel fed into it to focus on different parts along the frequency axis (Since w ∈ R D with D = num of channels × num of frequency bins).

In this paper we also have shown that training a deep neural networks having more than 15 convolutional layers on time-frequency input is definitely feasible with the help of global time aggregation.

To the authors' knowledge, there is no previous music information retrieval research utilizing neural networks having more than 10 convolutional layers.

A dataset consisting of over 20000 single singing voice recordings is also released and described in this paper.

The released dataset, DAMP-balanced, could be partitioned in a way that for singer classification, the performed songs for each singer are the same.

For future works, replacing max-pooling with striding in convolutional layers which recent works in CNN suggest will be experimented.

To improve global-aggregation, taking temporal order into consideration during the global-aggregation operation as suggested in BID21 will also be experimented.

Also the proposed neural network configurations will be experimented in other music information retrieval tasks such as music structure segmentation.

The DAMP-balanced dataset is a separate dataset from the original DAMP, but comes with the same format of metadata as the original DAMP dataset does.

The audio recordings and metadata of the DAMP-balanced dataset are collected by querying the internal database of the Sing!

Karaoke app hosted by Smule, Inc., which is the same as the original DAMP dataset.

The difference between the DAMP and the DAMP-balanced datasets lies at how the querying is done to collect the audio recordings and the metadata.

For the original DAMP, 10 singing performances from 3462 Sing!

Karaoke app users are randomly selected.

There are no specific constraints on the collections of songs performed by each user.

As a result, each user sang different collections of songs from each other and one song could be sang multiple times by one user.

On the contrary, the queries to retrieve audio recordings and metadata for the DAMP-balanced dataset specifically ask for a group of users that all sang one specific collections of songs at least once, with only one performance returned for each song and each user, per query.

14 popular songs (defined as the more times being sang the more popular) over the past year and are listed in TAB2 .

For the first 10 songs, 210 × 2 queries were created to retrieve audio recordings and metadata that cover all different combinations of splitting the 10 songs into 6/4 song collections.

Each query returns a set of users, along with their singing performances and metadata, such that all users in that returned set has only one performance of each of the songs in the specific 6 or 4 song collection.

For example, the train/validation sets used in this paper was the first 6 songs in TAB2 as training set and the following 4 songs as validation set.

This specific split has 276 performances for training and 88 performances for validation, and that lead to 46 and 22 singers respectively.

Different 6/4 split results in different number of singers in each set thus making the total number performances of different songs differ from each other.

For example, if instead the first 4 songs and the following 6 songs are taken as the 6/4 split of the first 10 songs, the first 4-song collection will have 459 users and 1836 performances while the following 6-song collection having 3 users and 18 performances.

The "balanced" structure of the DAMP-balanced allows possible train/validation rotation within the first 10 songs while leaving the last 4 songs as test set, or provides more possible "balanced" test sets for models training on other datasets.

<|TLDR|>

@highlight

Using deep learning techniques on singing voice related tasks.