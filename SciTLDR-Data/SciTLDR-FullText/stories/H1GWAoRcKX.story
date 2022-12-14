Over the past few years, various tasks involving videos such as classification, description, summarization and question answering have received a lot of attention.

Current models for these tasks compute an encoding of the video by treating it as a sequence of images and going over every image in the sequence, which becomes computationally expensive for longer videos.

In this paper, we focus on the task of video classification and aim to reduce the computational cost by using the idea of distillation.

Specifically, we propose a Teacher-Student network wherein the teacher looks at all the frames in the video but the student looks at only a small fraction of the frames in the video.

The idea is to then train the student to minimize  (i)  the difference between the final representation computed by the student and the teacher and/or (ii) the difference between the distributions predicted by the teacher and the student.

This smaller student network which involves fewer computations but still learns to mimic the teacher can then be employed at inference time for video classification.

We experiment with the YouTube-8M dataset and show  that the proposed student network can reduce the inference time by upto 30% with a negligent drop in the performance.

Today video content has become extremely prevalent on the internet influencing all aspects of our life such as education, entertainment, communication etc.

This has led to an increasing interest in automatic video processing with the aim of identifying activities BID16 BID24 , generating textual descriptions BID5 , generating summaries BID25 BID13 , answering questions BID8 and so on.

On one hand, with the availability of large scale datasets BID18 BID19 BID9 BID0 BID23 for various video processing tasks, it has now become possible to train increasingly complex models which have high memory and computational needs but on the other hand there is a demand for running these models on low power devices such as mobile phones and tablets with stringent constraints on latency and computational cost.

It is important to balance the two and design models which can learn from large amounts of data but still be computationally cheap at inference time.

With the above motivation, we focus on the task of video classification BID0 and aim to reduce the computational cost at inference time.

Current state of the art models for video classification (Yue-Hei BID24 BID10 BID17 treat the video as a sequence of images (or frames) and compute a representation of the video by using a Recurrent Neural Network (RNN).

The input to the RNN at every time step is an encoding of the corresponding image (frame) at that time step as obtained from a Convolutional Neural Network.

Computing such a representation for longer videos can be computationally very expensive as it requires running the RNN for many time steps.

Further, for every time step the corresponding frame from the video needs to pass through a convolutional neural network to get its representation.

Such computations are still feasible on a GPU but become infeasible on low end devices which have power, memory and computational constraints.

Typically, one can afford more computational resources at training time but a less expensive model is desired at test time.

We propose to achieve this by using the idea of distillation wherein we train a computationally expensive teacher network which computes a representation for the video by processing all frames in the video.

We then train a relatively inexpensive student network whose objective is to process only a few frames of the video and produce a representation which is very similar to the representation computed by the teacher.

This is achieved by minimizing (i) the squared error loss between the representations of the student network and the teacher network and/or (ii) by minimizing the difference between the output distributions (class probabilities) predicted by the two networks.

We refer to this as the matching loss.

FIG2 illustrates this idea where the teacher sees every frame of the video but the student sees fewer frames, i.e., every j-th frame of the video.

At inference time, we then use the student network for classification thereby reducing the time required for processing the video.

We experiment with two different methods of training the Teacher-Student network.

In the first method (which we call Serial Training), the teacher is trained independently and then the student is trained to match the teacher with or without an appropriate regularizer to account for the classification loss.

In the second method (which we call Parallel Training), the teacher and student are trained jointly using the classification loss as well as the matching loss.

We experiment with the YouTube-8M dataset and show that the smaller student network reduces the inference time by upto 30% while still achieving a classification performance which is very close to that of the expensive teacher network.

Since we focus on task of video classification in the context of the YouTube-8M dataset (Abu-ElHaija et al., 2016), we first review some recent works on video classification.

In the latter half of this section, we review some relevant work on model compression in the context of image processing.

On average the videos in the YouTube-8M dataset dataset have a length of 200 seconds.

Each video is represented using a sequence of frames where every frame corresponds to one second of the video.

These one-second frame representations are pre-computed and provided by the authors.

The authors also proposed a simple baseline model which treats the entire video as a sequence of these onesecond frames and uses an Long short-term memory networks (LSTM) to encode this sequence.

Apart from this, they also propose some simple baseline models like Deep Bag of Frames (DBoF) and Logistic Regression BID0 .

Various other classification models BID12 BID10 BID3 BID17 have been proposed and evaluated on this dataset which explore different methods of: (i) feature aggregation in videos (temporal as well as spatial) BID3 BID12 , (ii) capturing the interactions between labels and (iii) learning new non-linear units to model the interdependencies among the activations of the network BID12 .

We focus on one such state of the art model, viz., a hierarchical model whose performance is comparable to that of a single (non-ensemble) best performing model (Multi Scale CNN-LSTM reported in ) on this dataset.

We take this model as the teacher network and train a comparable student network as explained in the next section.

Recently, there has been a lot of work on model compression in the context of image classification.

We refer the reader to the survey paper by BID4 for a thorough review of the field.

For brevity, here we refer to only those papers which use the idea of distillation.

For example, BID1 ; BID6 ; BID11 ; BID2 use Knowledge Distillation to learn a more compact student network from a computationally expensive teacher network.

The key idea is to train a shallow student network to mimic the deeper teacher network, ensuring that the final output representations produced by the student network are very close to those produced by the teacher network.

This is achieved by training the student network using soft targets (or class probabilities) generated by the teacher instead of the hard targets present in the training data.

There are several other variants of this technique, for example, BID15 extend this idea to train a student model which not only learns from the outputs of the teacher but also uses the intermediate representations learned by the teacher as additional hints.

This idea of Knowledge Distillation has also been tried in the context of pruning networks in various domains BID2 BID14 .

For example, BID2 exploited the idea of intermediate representation matching along with Knowledge Distillation to compress state-of-art models for multiple object detection.

On similar lines, BID14 combine the ideas of Quantization and Knowledge Distillation for better model compression in different image classification models.

Stu- DISPLAYFORM0 STUDENT (every j th frame) DISPLAYFORM1 Figure 1: Architecture of TEACHER-STUDENT network for video classification dent teacher networks have also been proposed for speech recognition BID21 and reading comprehension BID7 .

While in these works, the teacher and student differ in the number of layers, in our case, the teacher and student network differ in the number of time steps or frames processed by the two networks.

To the best of our knowledge, this is the first work which uses a Teacher-Student network for video classification.

Given a fixed set of m classes y 1 , y 2 , y 3 , ..., y m ??? Y and a video V containing N frames (F 0 , F 1 , . . .

, F N ???1 ), the goal of video classification is to identify all the classes to which the video belongs.

In other words, for each of the m classes we are interested in predicting the probability P (y i |V).

This probability can be parameterized using a neural network f which looks at all the frames in the video to predict: DISPLAYFORM0 The focus of this work is to design a simpler network g which looks at only a fraction of the N frames at inference time while still allowing it to leverage the information from all the N frames at training time.

To achieve this, we propose a teacher student network as described below wherein the teacher has access to more frames than the student.

The teacher network can be any state of the art video classification model and in this work we consider the hierarchical RNN based model.

This model assumes that each video contains a sequence of b equal sized blocks.

Each of these blocks in turn is a sequence of l frames thereby making the entire video a sequence of sequences.

In the case of the YouTube-8M dataset, these frames are one-second shots of the video and each block b is a collection of l such one-second frames.

The model contains a lower level RNN to encode each block (sequence of frames) and a higher level RNN to encode the video (sequence of blocks).

As is the case with all state of the art models for video classification, this teacher network looks at all the N frames of video (F 0 , F 1 , . . .

, F N ???1 ) and computes an encoding E T of the video, which is then fed to a simple feedforward neural network with a multi-class output layer containing a sigmoid neuron for each of the Y classes.

This teacher network f can be summarized by the following set of equations: DISPLAYFORM0 The CN N used in the above equations is typically a pre-trained network and its parameters are not fine-tuned.

The remaining parameters of the teacher network which includes the parameters of the LST M as well as the output layer (with W parameters) are learnt using a standard multi-label classification loss L CE , which is a sum of the cross-entropy loss for each of the Y classes.

We refer to this loss as L CE where the subscript CE stands for cross entropy between the true labels y and predictions??.

DISPLAYFORM1 STUDENT: In addition to this teacher network, we introduce a student network which only processes every j th frame (F 0 , F j , F 2j , . . .

, F N j ???1 ) of the video (as shown in FIG2 ) and computes a representation E S of the video from these N j frames.

Similar to the teacher, the student also uses a hierarchical recurrent neural network to compute this representation which is again fed to a simple feedforward neural network with a multi-class output layer.

The parameters of this output layer are shared between the teacher and the student.

The student is trained to minimize the squared error loss between the representation computed by the student network and the representation computed by the teacher.

We refer to this loss as L rep where the subscript rep stands for representations.

DISPLAYFORM2 We also try a simple variant of the model, where in addition to ensuring that the final representations E S and E T are similar, we also ensure that the intermediate representations (I S and I T ) of the models are similar.

In particular, we ensure that the representation of the frames j, 2j and so on computed by the teacher and student network are very similar by minimizing the squared error distance between the corresponding intermediate representations.

We refer to this loss as L I rep where the superscript I stands for intermediate.

DISPLAYFORM3 Alternately, the student can also be trained to minimize the difference between the class probabilities predicted by the teacher and the student.

We refer to this loss as L pred where the subscript pred stands for predicted probabilities.

More specifically if P T = {p DISPLAYFORM4 where d is any suitable distance metric such as KL divergence or squared error loss.

TRAINING: Intuitively, it makes sense to train the teacher first and then use this trained teacher to guide the student.

We refer to this as the Serial mode of training as the student is trained after the teacher as opposed to jointly.

For the sake of analysis, we use different combinations of loss function to train the student as described below:(a) L rep : Here, we operate in two stages.

In the first stage, we train the student network to minimize the L rep as defined above, i.e., we train the RNN parameters of the student network to produce representations which are very similar to the teacher network.

The idea is to let the student learn by only mimicking the teacher and not worry about the final classification loss.

In the second stage, we then plug-in the classifier trained along with the teacher (see Equation 1 ) and finetune all the parameters of the student and the classifier using the cross entropy loss, L CE .

In practice, we found that the finetuning done in the second stage helps to improve the performance of the student.

DISPLAYFORM5 Here, we train the student to jointly minimize the representation loss as well as the classification less.

The motive behind this was to ensure that while mimicking the teacher, the student also keeps an eye on the final classification loss from the beginning (instead of being finetuned later as in the case above).

Table 1 : Performance Comparison of proposed Teacher-Student models using different StudentLoss variants, with their corresponding baselines using k frames.

Teacher-Skyline refers to the default model which process all the frames in a video.

DISPLAYFORM6 Finally, we also add in L pred to enable the student to learn from the soft targets obtained from the teacher in addition to hard target labels present in the training data.

FIG2 illustrates the process of training the student with different loss functions.

For the sake of completeness, we also tried an alternate mode in which train the teacher and student in parallel such that the objective of the teacher is to minimize L CE and the objective of the student is to minimize one of the 3 combinations of loss functions described above.

In this section, we describe the dataset used for our experiments, the hyperparameters that we considered, the baseline models that we compared with and the effect of different loss functions and training methods.

Since the authors did not release the test set, we used the original validation set as test set and report results on it.

In turn, we randomly sampled 48,163 examples from the training data and used these as validation data for tuning the hyperparameters of the model.

We trained our models using the remaining 5,738,718 training examples.

For all our experiments, we used Adam Optimizer with the initial learning rate set to 0.001 and then decreased it exponentially with 0.95 decay rate.

We used a batch size of 256.

For both the student and teacher networks we used a 2-layered MultiLSTM Cell with cell size of 1024 for both the layers of the hierarchical model.

For regularization, we used dropout (0.5) and L 2 regularization penalty of 2 for all the parameters.

We trained all the models for 5 epochs and then picked the best model based on validation performance.

We did not see any benefit of training beyond 5 epochs.

For the teacher network we chose the value of l (number of frames per block ) to be 20 and for the student network we set the value of l to 5 or 3 depending on the reduced number of frames considered by the student.

We used the following metrics as proposed in BID0 for evaluating the performance of different models : ??? GAP (Global Average Precision): is defined as DISPLAYFORM0 where p(i) is the precision at prediction i, ???r(i) is the change in recall at prediction i and P is the number of top predictions that we consider.

Following the original YouTube-8M Kaggle competition we use the value of P as 20.??? mAP (Mean Average Precision) : The mean average precision is computed as the unweighted mean of all the per-class average precisions.

As mentioned earlier the student network only processes k frames in the video.

We have considered different baselines to explore the possible selections of frames in a video sequence.

We report results with different values of k : 6, 10, 15, 20, 30 and compare the performance of our student networks with the following models: a) Teacher-Skyline: The original hierarchical model which processes all the frames of the video.

This, in some sense, acts as the upper bound on the performance.

b) Uniform-k : A hierarchical model trained from scratch which only processes k frames of the video.

These frames are separated by a constant interval and are thus equally spaced.

However, unlike the student model this model does not try to match the representations produced by the full teacher network.

c) Random-k: A hierarchical model which only processes k frames of the video.

These frames are sampled randomly from the video and may not be equally spaced.

d) First-k: A hierarchical model which processes the starting k frames of the video.

e) Middle-k: A hierarchical model which processes the middle k frames of the video.

f) Last-k: A hierarchical model which processes the last k frames of the video.

g) First-Middle-Last-k: A hierarchical model which processes k frames by selecting the starting

The results of our experiments are summarized mainly in Tables 1 (performance) and 4 (computation time).

We also report some additional results in Table 2 equally spaced k frames performs better than all the other baselines.

The performance gap between Uniform-k and the other baselines is even more significant when the value of k is small.

The main purpose of this experiment was to decide the right way of selecting frames for the student network.

Based on these results, we ensured that for all our experiments, we fed equally spaced k frames to the student.

Also, these experiments suggest that Uniform-k is a strong baseline to compare against.2.

Comparing Teacher-Student Network with Uniform-k Baseline: As mentioned above, the Uniform-k is a simple but effective way of reducing the number of frames to be processed.

We observe that all the teacher-student models outperform this strong baseline.

Further, in a separate experiment as reported in Table 3 we observe that when we reduce the number of training examples seen by the teacher and the student, then the performance of the Uniform-k baseline drops and is much lower than that of the corresponding teacher student network.

This suggests that the teacher student network can be even more useful when the amount of training data is limited.3.

Serial Versus Parallel Training of Teacher-Student: While the best results in Table 1 are obtained using Serial training, if we compare the corresponding rows of Serial and Parallel training we observe that there is not much difference between the two.

We found this to be surprising and investigated this further.

In particular, we compared the performance of the teacher after different epochs in the Parallel training setup with the performance of the a static teacher trained independently (Serial).

We plotted this performance in FIG3 and observed that after 3-4 epochs of training, the Parallel teacher is able to perform at par with Serial teacher (the constant blue line).

As a result, the Parallel student now learns from this trained teacher for a few more epochs and is almost able to match the performance of the Serial student.

This trend is same across the different combinations of loss functions that we used.

Apart from evaluating the final performance of the model in terms of MAP and GAP, we also wanted to check if the representations learned by the teacher and student are indeed similar.

To do this, we chose top-5 classes (class1:Vehicle, class2: Concert, class3: Association football, class4: Animal, class5: Food) in the Youtube-8M dataset and visualized the TSNE-embeddings of the representations computed by the student and the teacher for the same video (see FIG5 .

We use the darker shade of a color to represent teacher embeddings of a video and a lighter shade of the same color to represent the students embeddings of the same video.

We observe that the dark shades and the light shades of the same color almost completely overlap showing that the student and teacher representations are indeed very close to each other.

Intuitively, it seemed that the student should benefit more if we train it to match the intermediate representations of the teacher at different timesteps as opposed to only the final representation at the last time step.

However, as reported in Table 2 , we did not see any benefit of matching intermediate representations.6.

Computation time of different models: Lastly, the main aim of this work was to ensure that the computational cost and time is minimized at inference time.

The computational cost can be measured in terms of the number of FLOPs.

The main result from the table 4 is that when k=30, the inference time drops by 30% where the number of FLOPs reduces by 90%, but the performance of the model is not affected.

In particular, as seen in Table 4 : Comparison of FLOPs and evaluation time of models using k frames with Skyline model on original validation set using Tesla k80s GPU 0.5-0.9% and 0.9-2% respectively as compared to the teacher skyline.

This shows that the proposed teacher student model is an effective way of reducing the computational cost and time.

We proposed a method to reduce the computation time for video classification using the idea of distillation.

Specifically, we first train a teacher network which computes a representation of the video using all the frames in the video.

We then train a student network which only processes k frames of the video.

We use different combinations of loss functions which ensures that (i) the final representation produced by the student is the same as that produced by the teacher and (ii) the output probability distributions produced by the student are similar to those produced by the teacher.

We compare the proposed models with a strong baseline and skyline and show that the proposed model outperforms the baseline and gives a significant reduction in terms of computational time and cost when compared to the skyline.

In particular, We evaluate our model on the YouTube-8M dataset and show that the computationally less expensive student network can reduce the computation time by upto 30% while giving similar performance as the teacher network.

As future work, we would like to evaluate our model on other video processing tasks such as summarization, question answering and captioning.

We would also like to experiment with more complex and different teacher networks other than the hierarchical RNN considered in this work.

We would also like to independently train an agent which learns to select the most favorable k frames of the video as opposed to simply using equally spaced k frames.

@highlight

Teacher-Student framework for efficient video classification using fewer frames 

@highlight

The paper proposes an idea to distill from a full video classification model a small model that only receives smaller number of frames.

@highlight

The authors present a teacher-student network to solve video classification problem, proposing serial and parallel training algorithms aimed at reducing computational costs.