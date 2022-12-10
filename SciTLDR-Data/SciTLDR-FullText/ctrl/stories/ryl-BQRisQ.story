Collecting high-quality, large scale datasets typically requires significant resources.

The aim of the present work is to improve the label efficiency of large neural networks operating on audio data through multitask learning with self-supervised tasks on unlabeled data.

To this end, we trained an end-to-end audio feature extractor based on WaveNet that feeds into simple, yet versatile task-specific neural networks.

We describe three self-supervised learning tasks that can operate on any large, unlabeled audio corpus.

We demonstrate that, in a scenario with limited labeled training data, one can significantly improve the performance of a supervised classification task by simultaneously training it with these additional self-supervised tasks.

We show that one can improve performance on a diverse sound events classification task by nearly 6\% when jointly trained with up to three distinct self-supervised tasks.

This improvement scales with the number of additional auxiliary tasks as well as the amount of unsupervised data.

We also show that incorporating data augmentation into our multitask setting leads to even further gains in performance.

Although audio tag classification does not require the fine temporal resolution found in raw audio 91 waveforms, our chosen auxiliary tasks (or any arbitrary auditory task for which we may desire our 92 model to be sufficient) require higher temporal resolutions.

To satisfy this, we chose to build our 93 model following the WaveNet architecture [39] .

are processed using small, task-specific neural networks built atop a task-agnostic trunk.

The trunk architecture principally follows the structure of WaveNet, with several blocks of stacked, dilated, and causal convolutions between every convolution layer.

Outputs from the trunk are fed into task-specific heads (details in Section 3.1).As shown Figure 1 , our WaveNet trunk is composed of N blocks, where each block consists of S dilated causal convolution layers, with dilation factors increasing from 1 to 2 S − 1, residual connections and saturating nonlinearities.

We label the blocks using b = 1, · · · , N .

We use indices ∈ [1 + (b − 1)S, bS] to label layers in block b. Each layer, , of the WaveNet trunk consists of a "residual atom" which involves two computations, labeled as "Filter" and "Gate" in the figure.

Each residual atom computation produces a hidden state vector h ( ) and a layer output x ( ) defined via DISPLAYFORM0 where denotes element-wise products, represents the regular convolution operation, denotes dilated convolutions with a dilation factor of 2 mod bS if is a layer in block b + 1, σ denotes the sigmoid function and W ( ) gate and W ( ) f ilter are the weights for the gate and filter, respectively.

The first ( = 0) layer -represented as the initial stage marked "1 × 1 Conv" in Figure 1 -applies causal convolutions to the raw audio waveforms X = (X 1 , X 2 , · · · , X T ), sampled at 16 kHz, to produce an output DISPLAYFORM1 Given the structure of the trunk laid out above, any given block b has an effective receptive field of 1 + b(2 S − 1).

Thus the total effective receptive field of our trunk is τ = 1 + N (2 S − 1).

Following an extensive hyperpameter search over various configurations, we settled on [N = 3] blocks comprised of [S = 6] layers each for our experiments.

Thus our trunk has a total receptive field of τ = 190, which corresponds to about 12 milliseconds of audio sampled at 16kHz.

As indicated above, each task-specific head is a simple neural network whose input data is first 102 constrained to pass through a trunk that it shares with other tasks.

Each head is free to process this 103 input to its advantage, independent of the other heads.

Each task also specifies its own objective function, as well as a task-specific optimizer, with cus-105 tomized learning rates and annealing schedules, if necessary.

We arbitrarily designate supervised 106 tasks as the primary tasks and refer to any self-supervised tasks as auxiliary tasks.

In the experiments 107 reported below, we used "audio tagging" as the primary supervised classification task and "next-step 108 prediction", "noise reduction" and "upsampling" as auxiliary tasks training on various amounts of unlabeled data.

The parameters used for each of the task specific heads can be found in TAB1 accompanying supplement to this paper.

Figure 2: The head architectures were designed to be simple, using only as few layers as necessary to solve the task.

Simpler head architectures force the shared trunk to learn a representation suitable for multiple audio tasks.

The next-step prediction task can be succinctly formalized as follows: given a sequence 113 {x t−τ +1 , · · · , x t } of frames of an audio waveform, predict the next value x t+1 in the sequence.

This prescription allows one to cheaply obtain arbitrarily large training datasets from an essentially 115 unlimited pool of unlabeled audio data.

Our next-step prediction head is a 2-layer stack of 1×1 convolutional layers with ReLU nonlinearities 117 in all but the last layer.

The first layer contains 128 units, while the second contains a single output unit.

The head takes in τ frames of data from the trunk, where τ is the trunk's effective receptive field, and 119 produces an output which represents the model's prediction for the next frame of audio in the sequence.

The next-step head treats this as a regression problem, using the mean squared error of the difference 121 between predicted values and actual values as a loss function, i.e. given inputs {x t−τ +1 , · · · , x t },

the head produces an output y t from which we compute a loss L MSE (t) = (y t − x t+1 ) 2 and then 123 aggregate over the frames to get the total loss.

We would like to note that the original WaveNet implementation treated next-step prediction as a 125 classification problem, instead predicting the bin-index of the audio following a µ-law transform.

We 126 found that treating the task as a regression problem worked better in multitask situations but make no

claims on the universality of this choice.

In defining the noise reduction task, we adopt the common approach of treating noise as an additive 130 random process on top of the true signal: if {x t } denotes the clean raw audio waveform, we obtain 131 the noisy version viax t := x t + ξ t where ξ t an arbitrary noise process.

For the denoising task, the 132 model is trained to predict the clean sample, x t , given a window x t− well-adapted to solving either task.

Thus, our noise reduction head has a structure similar to the 136 next-step head.

It is trained to minimize a smoothed L1 loss between the clean and noisy versions of 137 the waveform inputs, i.e. for each frame t, the head produces an outputŷ t , and we compute the loss DISPLAYFORM0 and then aggregate over frames to obtain the total loss.

We used the smooth L1 loss because it 139 provided a more stable convergence for the denoising task than mean squared error.

In the same spirit as the denoising task, one can easily create an unsupervised upsampling task

Again, given the formal similarity of the upsampling task to the next-step prediction and noise-151 reduction tasks, we used an upsampling head with a structure virtually identical to those described 152 above.

As with the denoising task, we used a smooth L1 loss function (see eqn.

FORMULA2 having manually-verified labels and the remaining 5763 having non-verified labels, meaning they 177 were automatically categorized using user-provided metadata.

The test set is composed of 1600 audio 178 clips with manually-verified labels which are used for the final scoring.

The Librispeech dataset 1 (comprised of read English speech sampled at 16 kHz) was used as a proxy

for a large unlabeled dataset.

The models described below were trained using clips from either the 182 "train-clean-100" or "train-other-500 versions".

Models trained with 5, 50 and 100 hours of unlabeled 183 data were sourced from "train-clean-100", while the model trained with 500 hours was sourced

entirely from "train-other-500".

Due to memory constraints, we limited the duration of each utterance 185 to 2 seconds which we obtained by cropping from a random position in the original clip.

This dataset 186 was only used to train the auxiliary tasks.

We trained the model using raw audio waveform inputs taken from the FSDKaggle2018 and Lib-189 rispeech datasets.

All code for the experiments described here was written in the PyTorch framework

[29].

All audio samples were first cropped to two seconds in duration and downsampled to 16 kHz.

To normalize for the variation in onset times for different utterances, the 2 seconds were randomly both the main task and the auxiliary tasks, heuristically favoring performance on the main task.

We jointly trained the model on all tasks simultaneously by performing a forward pass for each task, important parameters of the model can be found in TAB4 of the accompanying supplement to this 217 paper.

As discussed above, we used audio tagging as the main task to investigate whether supervised First, we trained a purely supervised model on 2 seconds of non-silence audio extracted using random

In this experiment, we added each of the self-supervised tasks to the baseline model discussed above,

simultaneously training them using 100 hours of unlabeled data sampled from the Librispeech dataset 236 along with the main supervised task.

We notice that, addition of any self-supervised task showed an average improvement of 4.6% to the MAP@3 score compared to the main task's baseline performance.

Adding a pair of tasks gave an average improvement of 4.55% over baseline, showing no improvement over adding a single task.

Training with three additional tasks yielded the best results with an improvement of 5.33% over the main task.

Looking at MAP@3 scores throughout training showed that convergence in every multitask setting was stable, with gradual improvements for increasing number of tasks.

The best performance values on the test sets for a sequence of task additions can be found in TAB1 .The set of experiments described above demonstrate that, for a fixed amount of unlabeled data (100 hours), simultaneously training a supervised task with various self-supervised tasks yields a significant improvement in the main task's performance.

To further test how performance changes with increasing amounts of data, we re-trained our model while varying the amount of unlabeled data used to train the auxiliary tasks.

We noticed that even without any additional unlabeled data, the MAP@3 score with three additional tasks was significantly better than the score obtained on a single task.

This demonstrates that addition of self-supervised tasks improves the performance of main task.

Increasing the size of the unlabeled data for the auxiliary tasks increases the size of the multitask benefit (Figure 3 ).The MAP@3 Scores at different levels of unlabaled data showed progressive improvement to 0.656, 0.669, with 5 and 50 hours respectively.

We observed a peak MAP@3 score of 0.694 with 500 hours of unlabeled data, which is an improvement of 8.94% over the main task's baseline performance.

Next, we explore several approaches to data augmentation and compare them with multitask learning.

Previous work has demonstrated the effectiveness of data augmentation through simple techniques, such as noise injection, and pitch shifting [35, 42, 44] .

We compared our proposed method with traditional data augmentation strategies by retraining our model only for the main task after applying the aforementioned augmentations to the FSDKaggle2018 training data.

The MAP@3 values for the data augmentation experiments on the test sets can be found in TAB2 .

We observed a peak MAP@3 score of 0.703 with pitch shifting augmentation which is similar in scale to that of our best multitask performance gains.

In an attempt to observe how both the techniques work together, we combined data augmentation with multitask learning and obtain an MAP@3 score of 0.726 which was the best score among all the experiments we conducted.

Figure 3: Improved MAP@3 scores with increasing amounts of unlabeled data.

Shown are the MAP@3 scores on test set when the main task is trained with 3 auxiliary tasks with 0, 5, 50, 100, and 500 hours of unlabeled data respectively.

The amount of labelled data is held constant for the whole experiment.

We see a smooth increase in performance with increasing amounts of unlabeled data.

where one has a limited quantity of labeled data.

We have also shown that the performance of the 244 supervised task improves by increasing either the number of auxiliary self-supervised tasks or the 245 quantity of unlabeled data or both.

We attain a peak performance boost of 8.94% over the baseline 246 with the inclusion of 3 self-supervised tasks when trained with additional 500 hours of unlabeled data.

Finally, our multitask learning scheme further benefits when the training data for the data-constrained 248 task is augmented using standard techniques.

Since our results suggest that the performance gain with 249 our approach is additive when used with data augmentation, it may be interesting to use multitask 250 learning with other augmentation approaches to observe if they complement each other in different 251 settings.

We have strived to systematically present our results within a coherent multitask learning framework.

<|TLDR|>

@highlight

Improving label efficiency through multi-task learning on auditory data