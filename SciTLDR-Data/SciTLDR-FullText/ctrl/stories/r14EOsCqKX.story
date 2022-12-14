The convergence rate and final performance of common deep learning models have significantly benefited from recently proposed heuristics such as learning rate schedules, knowledge distillation, skip connections and normalization layers.

In the absence of theoretical underpinnings, controlled experiments aimed at explaining the efficacy of these strategies can aid our understanding of deep learning landscapes and the training dynamics.

Existing approaches for empirical analysis rely on tools of linear interpolation and visualizations with dimensionality reduction, each with their limitations.

Instead, we revisit the empirical analysis of heuristics through the lens of recently proposed methods for loss surface and representation analysis, viz.

mode connectivity and canonical correlation analysis (CCA), and hypothesize reasons why the heuristics succeed.

In particular, we explore knowledge distillation and learning rate heuristics of (cosine) restarts and warmup using mode connectivity and CCA.

Our empirical analysis suggests that: (a) the reasons often quoted for the success of cosine annealing are not evidenced in practice; (b) that the effect of learning rate warmup is to prevent the deeper layers from creating training instability; and (c) that the latent knowledge shared by the teacher is primarily disbursed in the deeper layers.

The introduction of heuristics such as normalization layers BID19 BID0 , residual connections BID11 , and learning rate strategies BID26 BID9 Smith, 2017) have greatly accelerated progress in Deep Learning.

Many of these ingredients are now commonplace in modern architectures, and some of them have also been buttressed with theoretical guarantees BID1 BID28 BID10 .

However, despite their simplicity and efficacy, why some of these heuristics work is still relatively unknown.

Existing attempts at explaining these strategies empirically have been limited to intuitive explanations and the use of tools such as spectrum analysis (Sagun et al., 2017) , linear interpolation between two models and low-dimensional visualizations of the loss surface.

In our work, we instead use recent tools built specifically for analyzing deep networks, viz.

, mode connectivity and singular value canonical correlation analysis (SVCCA) (Raghu et al., 2017) .

We investigate three strategies in detail: (a) cosine learning rate decay, (b) learning rate warmup, and (c) knowledge distillation, and list the summary of our contributions at the end of this section.

Cosine annealing BID26 , also known as stochastic gradient descent with restarts (SGDR), and more generally cyclical learning rate strategies (Smith, 2017) , have been recently proposed to accelerate training of deep networks BID3 .

The strategy involves reductions and restarts of learning rates over the course of training, and was motivated as means to escape spurious local minima.

Experimental results have shown that SGDR often improves convergence both from the standpoint of iterations needed for convergence and the final objective.

Learning rate warmup BID9 also constitutes an important ingredient in training deep networks, especially in the presence of large or dynamic batch sizes.

It involves increasing the learning rate to a large value over a certain number of training iterations followed by decreasing the learning rate, which can be performed using step-decay, exponential decay or other such schemes.

The strategy was proposed out of the need to induce stability in the initial phase of training with large learning rates (due to large batch sizes).

It has been employed in training of several architectures at scale including ResNets and Transformer networks (Vaswani et al., 2017) .Further, we investigate knowledge distillation (KD) BID13 .

This strategy involves first training a (teacher) model on a typical loss function on the available data.

Next, a different (student) model (typically much smaller than the teacher model) is trained, but instead of optimizing the loss function defined using hard data labels, this student model is trained to mimic the teacher model.

It has been empirically found that a student network trained in this fashion significantly outperforms an identical network trained with the hard data labels.

We defer a detailed discussion of the three heuristics, and existing explanations for their efficacy to sections 3, 4 and 5 respectively.

Finally, we briefly describe the tools we employ for analyzing the aforementioned heuristics.

Mode connectivity (MC) is a recent observation that shows that, under circumstances, it is possible to connect any two local minima of deep networks via a piecewise-linear curve BID5 .

This shows that local optima obtained through different means, and exhibiting different local and generalization properties, are connected.

The authors propose an algorithm that locates such a curve.

While not proposed as such, we employ this framework to better understand loss surfaces but begin our analysis in Section 2 by first establishing its robustness as a framework.

Deep network analyses focusing on the weights of a network are inherently limited since there are several invariances in this, such as permutation and scaling.

Recently, Raghu et al. (2017) propose using CCA along with some pre-processing steps to analyze the activations of networks, such that the resulting comparison is not dependent on permutations and scaling of neurons.

They also prove the computational gains of using CCA over alternatives ( BID25 ) for representational analysis and employ it to better understand many phenomenon in deep learning.

??? We use mode connectivity and CCA to improve understanding of cosine annealing, learning rate warmup and knowledge distillation.

For mode connectivity, we also establish the robustness of the approach across changes in training choices for obtaining the modes.??? We demonstrate that the reasons often quoted for the success of cosine annealing are not substantiated by our experiments, and that the iterates move over barriers after restarts but the explanation of escaping local minima might be an oversimplification.??? We show that learning rate warmup primarily limits weight changes in the deeper layers and that freezing them achieves similar outcomes as warmup.??? We show that the latent knowledge shared by the teacher in knowledge distillation is primarily disbursed in the deeper layers.2 EMPIRICAL TOOLS 2.1 MODE CONNECTIVITY introduce a framework, called mode connectivity, to obtain a low loss (or high accuracy, in the case of classification) curve of simple form, such as a piecewise linear curve, that connects optima (modes of the loss function) found independently.

This observation suggests that points at the same loss function depth are connected, somewhat contrary to several empirical results claiming that minima are isolated or have barriers between them 1 .Let w a ??? R D and w b ??? R D be two modes in the D-dimensional parameter space obtained by optimizing a given loss function L(w) (like the cross-entropy loss).

We represent a curve connecting Validation accuracy corresponding to models on the following 6 different curves -curve GA represents curve connecting mode G (one found with default hyperparameters) and mode A (using large batch size), similarly, curve GB connects mode G and mode B (using Adam), curve GC connects to mode C (using linearly decaying learning rate), curve GD to mode D (with lesser L2 regularization), curve GE to mode E (using a poor initialization), and curve GF to mode F (without using data augmentation).

t = 0 corresponds to mode G for all plots.w a and w b by ?? ?? (t) : [0, 1] ??? R D , such that ?? ?? (0) = w a and ?? ?? (1) = w b .

To find a low loss path, we find the set of parameters ?? ??? R D that minimizes the following loss: DISPLAYFORM0 where U (0, 1) is the uniform distribution in the interval [0, 1].

To optimize (??) for ??, we first need to chose a parametric form for ?? ?? (t).

One of the forms proposed by is a polygonal chain with a single bend at ?? as follows DISPLAYFORM1 To minimize (??), we sample t ??? U [0, 1] at each iteration and use ??? ?? L(?? ?? (t)) as an unbiased estimate for the true gradient ??? ?? (??) to perform updates on ??, where ?? is initialized with 1 2 (w a +w b ).

To demonstrate that the curve-finding approach works in practice, use two optima found using different initializations but a common training scheme which we detail below.

We explore the limits of this procedure by connecting optima obtained from different training strategies.

Our goal of this investigation is to first establish the robustness of the framework in order to seamlessly use it as a tool for analysis.

In particular, we experiment with different initializations, optimizers, data augmentation choices, and hyperparameter settings including regularization, training batch sizes, and learning rate schemes.

We note in passing that while the framework was proposed to connect two points in the parameter space that are at equal depth in the loss landscape, it is well-defined to also connect points at different depths; in this case, the path corresponds to one that minimizes the average loss along the curve.

Conventional wisdom suggests that the different training schemes mentioned above will converge to regions in the parameter space that are vastly different from each other.

Examples of this include size of minibatches used during training BID22 , choice of optimizer BID12 Wilson et al., 2017) , initialization BID8 and choice of regularizer.

Having a high accuracy connection between these pairs would seem counterintuitive.

For obtaining the reference model (named mode G), we train the VGG-16 model architecture (Simonyan & Zisserman, 2014) using CIFAR-10 training data BID23 for 200 epochs with SGD.

We then build 6 variants of the reference mode G as follows: we obtain mode A using a training batch size of 4000, mode B by using the Adam optimizer instead of SGD, mode C with a linearly decaying learning rate instead of the step decay used in mode G, mode D using a smaller weight decay of 5 ?? 10 ???6 , mode E by increasing the variance of the initialization distribution to 3 ?? 2/n and mode F using no data augmentation.

Note that for the set of modes {A, B, C, D, E, F }, all the other hyper-parameters and settings except the ones mentioned above are kept same as that for mode G. We use the mode connectivity algorithm on each of the 6 pairs of modes including G and another mode, resulting in curves GA, GB, GC, GD, GE, and GF .

FIG0 shows the validation accuracy for models on each of the 6 connecting curves during the 20th, 40th, 60th and 80th epochs of the mode connectivity training procedure and also for models on the line segment joining the two endpoints (corresponding to the initialization for ?? at epoch 0).

As described in Section 2.1, for a polychain curve GX (connecting modes G and X using the curve described by ??), model parameters ?? ?? (t) on the curve are given by p ?? ?? (t) = 2(tp ?? + (0.5 ??? t)p G ) if 0 ??? t ??? 0.5 and p ?? ?? (t) = 2((t ??? 0.5)p X + (1 ??? t)p ?? ) if 0.5 < t ??? 1 where p G , p ?? and p X are parameters of the models G, ??, and X respectively.

Thus ?? ?? (0) = G and ?? ?? (1) = X.In a few epochs of the curve training, for all 6 pairs, we can find a curve such that each point on it generalizes almost as well as models from the pair that is being connected.

Note that by virtue of existence of these 6 curves, there exists a high accuracy connecting curve (albeit with multiple bends) for each of the 7 2 pairs of modes.

We refer the reader to Appendix 7 for a t-SNE plot of the modes and their connections, and also for additional plots and details.

Having established the high likelihood of the existence of these curves, we use this procedure along with interpolation of the loss surface between parameters at different epochs as tools to analyze the dynamics of SGD and SGDR.

Canonical correlation analysis (CCA) is a classical tool from multivariate statistics BID16 that investigates the relationships between two sets of random variables.

Raghu et al. (2017) have proposed coupling CCA with pre-processing steps like Singular Value Decomposition (SVD) or Discrete Fourier Transform (DFT) to design a similarity metric for two neural net layers that we want to compare.

These layers do not have to be of the same size or belong to the same network.

Given a dataset with m examples X = {x 1 , . . .

x m }, we denote the scalar output of the neuron z l i (i-th neuron of layer l) for the input x i by f z L i (x i ).

These scalar outputs can be stacked (along n different neurons and m different datapoints) to create a matrix L ??? R m??n representing the output of a layer corresponding to the entire dataset.

This choice of comparing neural network layers using activations instead of weights and biases is crucial to the setup proposed.

Indeed, invariances due to re-parameterizations and permutations limit the interpretability of the model weights BID4 .

However, under CCA of the layers, two activation sets are comparable by design.

Given representations corresponding to two layers L a ??? R ma??n and L b ??? R m b ??n , SVCCA first performs dimensionality reduction using SVD to obtain L a ??? R m a ??n and L b ??? R m b ??n while preserving 99% of the variance.

The subsequent CCA step involves transforming L a and L b to a 1 L a and b 1 L b respectively where {a 1 , b 1 } is found by maximizing the correlation between the transformed subspaces, and the corresponding correlation is denoted by ?? 1 .

This process continues, using orthogonality constraints, till c = min{m a , m b } leading to the set of correlation values {?? 1 , ?? 2 . . .

?? c } corresponding to c pairs of canonical variables {{a 1 , b 1 }, {a 2 , b 2 }, . . .

{a c , b c }} respectively.

We refer the reader to Raghu et al. (2017) for details on solving these optimization problems.

The average of these c correlations 1 n i ?? i is then considered as a measure of the similarity between the two layers.

For convolutional layers, Raghu et al. (2017) suggest using a DFT pre-processing step before CCA, since they typically have a large number of neurons (m a or m b ), where performing raw SVD and CCA would be computationally too expensive.

This procedure can then be employed to compare different neural network representations and to determine how representations evolve over training iterations.

BID26 introduced SGDR as a modification to the common linear or step-wise decay of learning rates.

The strategy decays learning rates along a cosine curve and then, at the end of the decay, restarts them to its initial value.

The learning rate at the t-th epoch in SGDR is given by the following expression in (1) where ?? min and ?? max are the lower and upper bounds respectively for the learning rate.

T cur represents how many epochs have been performed since the last restart and a warm restart is simulated once T i epochs are performed.

Also T i = T mult ?? T i???1 , meaning the period T i for the learning rate variation is increased by a factor of T mult after each restart.

While the strategy has been claimed to outperform other learning rate schedulers, little is known why this has been the case.

One explanation that has been given in support of SGDR is that it can be useful to deal with multi-modal functions, where the iterates could get stuck in a local optimum and a restart will help them get out of it and explore another region; however, BID26 do not claim to observe any effect related to multi-modality.

BID18 propose an ensembling strategy using the set of iterates before restarts and claim that, when using the learning rate annealing cycles, the optimization path converges to and escapes from several local minima.

We empirically investigate if this is actually the case by interpolating the loss surface between parameters at different epochs and studying the training and validation loss for parameters on the hyperplane passing through 2 the two modes found by SGDR and their connectivity.

Further, by employing the CCA framework as described in Section 2.2, we investigate the progression of training, and the effect of restarts on the model activations.

We train a VGG-16 network (Simonyan & Zisserman, 2014) on the CIFAR-10 dataset using SGDR.

For our experiments, we choose T 0 = 10 epochs and T mult = 2 (warm restarts simulated every 10 epochs and the period T i doubled at every new warm restart), ?? max = 0.05 and ?? min = 10 ???6 .

We also perform VGG training using SGD (with momentum of 0.9) and a step decay learning rate scheme (initial learning rate of ?? 0 = 0.05, scaled by 5 at epochs 60 and 150).

In order to understand the loss landscape on the optimization path of SGDR, the pairs of iterates obtained just before the restarts {w 30 , w 70 }, {w 70 , w 150 } and {w 30 , w 150 } are given as inputs to the mode connectivity algorithm, where w n is the model corresponding to parameters at the n-th epoch of training.

FIG1 (b) shows the training loss for models along the line segment joining these pairs and those on the curve found through mode connectivity.

For the baseline case of SGD training, we connect the iterates around the epochs when we decrease our learning rate in the step decay learning rate scheme.

Thus, we chose {w 55 , w 65 }, {w 145 , w 165 } and {w 55 , w 165 } as input pairs to the mode connectivity algorithm.

FIG1 (c) shows the training loss for models along the line segments joining these pairs and the curves found through mode connectivity.

From FIG1 (b), it is clear that for the pairs {w 30 , w 150 } and {w 70 , w 150 } the training loss for points on segment is much higher than the endpoints suggesting that SGDR indeed finds paths that move over a barrier 3 in the training loss landscape.

In contrast, for SGD (without restarts) in FIG1 (c) none of the three pairs show evidence of having a training loss barrier on the line segment joining them.

Instead there seems to be an almost linear decrease of training loss along the direction of these line segments, suggesting that SGD's trajectory is quite different from SGDR's.

We present additional experiments, including results for other metrics, in Appendix 8.To further understand the SGDR trajectory, we evaluate the intermediate points on the hyperplane in the D-dimensional space defined by the three points: w 70 , w 150 and w 70???150 , where w 70???150 is the bend point that defines the high accuracy connection for the pair {w 70 , w 150 }. ) suggests that SGDR helps the iterates converge to a different region although neither of w 70 or w 150 are technically a local minimum, nor do they appear to be lying in different basins, hinting that BID18 's claims about SGDR converging to and escaping from local minima might be an oversimplification.

4 Another insight we can draw from FIG3 (a) is that the path found by mode connectivity corresponds to lower training loss than the loss at the iterates that SGDR converges to (L(w 150 ) > L(w 70???150 )).

However, FIG3 (b) shows that models on this curve seem to overfit and not generalize as well as the iterates w 70 and w 150 .

Thus, although gathering models from this connecting curve might seem as a novel and computationally cheap way of creating ensembles, this generalization gap alludes to one limitation in doing so; point to other shortcomings of curve ensembling in their original work.

In FIG3 , the region of the plane between the iterates w 70 and w 150 corresponds to higher training loss but lower validation loss than the two iterates.

This hints at a reason why averaging iterates to improve generalization using cyclic or constant learning rates has been found to work well.

Finally, in FIG0 in Appendix 9, we present the CCA similarity plots for two pairs of models: epochs 10 and 150 (model at the beginning and end of training), and epochs 150 and 155 (model just before and just after a restart).

For standard SGD training, Raghu et al. (2017) observe that the activations of the shallower layers bear closer resemblance than the deeper layers between a partially and fully trained network from a given training run.

For SGDR training, we witness similar results (discussed in Appendix 9), meaning that the representational similarities between the network layers at the beginning and end of training are alike for SGDR and SGD, even though restarts lead to a trajectory that tends to cross over barriers.

Learning rate warmup is a common heuristic used by many practitioners for training deep neural nets for computer vision BID9 and natural language processing BID2 Vaswani et al., 2017) tasks.

Theoretically, it can be shown that the learning dynamics of SGD rely on the ratio of the batch size and learning rate (Smith et al., 2017; BID21 BID14 .

And hence, an increase in batch size over a baseline requires an accompanying increase in learning rate for comparable training.

However, in cases when the batch size is increased significantly, the curvature of the loss function typically does not support a proportional increase in the learning rate.

Warmup is hence motivated as a means to use large learning rates without causing training instability.

We particularly focus on the importance of the learning rate schedule's warmup phase in the large batch (LB) training of deep convolutional neural networks as discussed in Goyal Using CCA as a tool to study the learning dynamics of neural networks through training iterations, we investigate the differences and similarities for the following 3 training configurations -(a) large batch training with warmup (LB + warmup), (b) large batch training without warmup (LB no warmup) and (c) small batch training without warmup (SB no warmup).

We train a VGG-11 architecture on the CIFAR-10 (Krizhevsky et al., 2014) dataset using SGD with momentum of 0.9.

Learning rate for the small batch case (batch-size of 100) is set to 0.05, and for the large batch cases (batch-size of 5000) is set to 2.5 as per the scaling rule.

For the warmup, we increase the learning rate from 0 to 2.5 over the first 200 iterations.

Subsequently, we decrease the learning rate as per the step decay schedule for all runs, scaling it down by a factor of 10 at epochs 60, 120 and 150.

We plot the learning rate and validation accuracy for these 3 cases in Figure 4 Figure 4 (c) plots the similarity for layer i of iter a with the same layer of iter b (this corresponds to diagonal elements of the matrices in FIG6 ) for these three setups.

An evident pattern in FIG6 , (b) and (c) is the increase in similarity for the last few layers (stack of fully-connected layers) for the LB + warmup and SB cases, which is absent in the LB without warmup case.

This suggests that when used with the large batch size and learning rate, warmup tends to avoid unstably large changes in the fully-connected (FC) stack for this network configuration.

To validate this proposition, we train using the LB without warmup setup, but freezing the fully-connected stack for the first 20 epochs 5 (LB no warmup + FC freeze).

Figure 4( M denotes the i-th layer of network M , T denotes the teacher network (VGG16), S distilled is the student network trained using distillation and S indep.

is the student network trained using hard training labels.suggesting the validity our proposition in this case.

We refer the reader to Appendix 10 for analogous results for ResNet-18 and ResNet-32 BID11 ; thus also demonstrating the generality of our claim.

Finally, note from Figure 4 (d) that no qualitative difference exists in the trajectory beyond the warmup when compared to the standard training approach (Raghu et al., 2017) .

We study knowledge distillation as proposed by BID13 using CCA to measure representational similarity between layers of the teacher and student model.

Distillation involves training a "student" model using the output probability distribution of a "teacher" model.

This has been widely known to help the student model perform better than it would, if it were trained using hard labels due to knowledge transfer from the teacher model.

The reason often quoted for the success of distillation is the transfer of dark knowledge from the teacher to the student BID13 , and more recently, as an interpretation of importance weighing BID6 .

We investigate if this knowledge transfer is limited to certain parts of the network, and if representational similarity between layers of the student and teacher model and a student can help answer this question.

To construct an example of distillation that can be used for our analysis, we use a VGG-16 model (Simonyan & Zisserman, 2014) as our teacher network and a shallow convolutional network ([conv, maxpool, relu] x2, fc, relu, fc, fc, softmax) as the student network.

We train the shallow network for CIFAR-10 using the teacher's predicted probability distribution (softened using a temperature of 5), (S distilled ), and for the baseline, train another instance of the same model in a standard way using hard labels, (S indep. ).

Over 5 runs for each of the two setups, we find the distillation training attains the best validation accuracy at 85.18% while standard training attains its best at 83.01%.

We compare their layer-wise representations with those of the teacher network (T ).

FIG8 shows the CCA plots and the absolute value of their difference.

The scores of these two pairs are quite similar for the shallow layers of the student network relative to the deeper layers, suggesting that the difference that knowledge distillation brings to the training of smaller networks is restricted to the deeper layers (fc stack).

Similar results are obtained through different configurations for the student and teacher when the student benefits from the teacher's knowledge.

We hypothesize that the dark knowledge transferred by the teacher is localized majorly in the deeper (discriminative) layers, and less so in the feature extraction layers.

We also note that this is not dissimilar to the hypothesis of BID6 , and also relates ot the results from the literature on fine-tuning or transfer learning BID8 Yosinski et al., 2014; BID17 which suggest training of only higher layers.

Heuristics have played an important role in accelerating progress of deep learning.

Founded in empirical experience, intuition and observations, many of these strategies are now commonplace in architectures.

In the absence of strong theoretical guarantees, controlled experiments aimed at explaining the the efficacy of these strategies can aid our understanding of deep learning and the training dynamics.

The primary goal of our work was the investigation of three such heuristics using sophisticated tools for landscape analysis.

Specifically, we investigate cosine annealing, learning rate warmup, and knowledge distillation.

For this purpose, we employ recently proposed tools of mode connectivity and CCA.

Our empirical analysis sheds light on these heuristics and suggests that: (a) the reasons often quoted for the success of cosine annealing are not evidenced in practice; (b) that the effect of learning rate warmup is to prevent the deeper layers from creating training instability; and (c) that the latent knowledge shared by the teacher is primarily disbursed in the deeper layers.

Inadvertently, our investigation also leads to the design of new heuristics for practically improving the training process.

Through our results on SGDR, we provide additional evidence for the success of averaging schemes in this context.

Given the empirical results suggesting the localization of the knowledge transfer between teacher and student in the process of distillation, a heuristic can be designed that only trains portions of the (pre-trained) student networks instead of the whole network.

For instance, recent results on self-distillation BID6 show improved performance via multiple generations of knowledge distillation for the same model.

Given our results, computational costs of subsequent generations can be reduced if only subsets of the model are trained, instead of training the entire model.

Finally, the freezing of weights instead of employing learning rate warmup allows for comparable training performance but with reduced computation during the warmup phase.

We note in passing that our result also ties in with results of Hoffer et al. FORMULA2 The learning rate is initialized to 0.05 and scaled down by a factor of 5 at epochs {60, 120, 160} (step decay).

We use a training batch size of 100, momentum of 0.9, and a weight decay of 0.0005.

Elements of the weight vector corresponding to a neuron are initialized randomly from the normal distribution N (0, 2/n) where n is the number of inputs to the neuron.

We also use data augmentation by random cropping of input images.

Figures 7, 8 and 9 show the Validation Loss, Training Accuracy and Training Loss respectively for the curves joining the 6 pairs discussed in Section 2.1.1.

These results too, confirm the overfitting or poor generalization tendency of models on the curve.

We use t-SNE BID27 to visualize these 7 modes and the ?? points that define the connectivity for the 6 pairs presented in Section 2.1.1, in a 2-dimensional plot in FIG0 .

Since t-SNE is known to map only local information correctly and not preserve global distances, we caution the reader about the limited interpretability of this visualization, it is presented simply to establish the notion of connected modes.

The W n in FIG3 is equivalent to meaning it is the point on the plane (linear combination of w 70 , w 150 and ??) with the least l-2 distance from the original point (iterate in this case).

DISPLAYFORM0 8.3 CONNECTING MODES w 30 AND w 70 FROM SGDRIn Section 3, we present some experiments and make observations on the trajectory of SGDR by using the plane defined by the points w 70 , w 150 and w 70???150 .

Here we plot the Training loss and Validation loss surface in FIG0 for another plane defined by SGDR's iterates w 30 , w 70 and their connection w 30???70 to ensure the reader that the observations made are general enough.

The VGG-16 architecture used in Section 3 does not include Batch Normalization, which has been known to alter properties of the loss surface (Santurkar et al. (2018) ).

Therefore we train VGG-16 with Batch Normalization using SGDR to verify if our observations hold for this case too.

As pointed out in Appendix A.2 of , at the test stage, we compute the Batch Normalization statistics for a network on the curve with an additional pass over the data, since these are not collected during training.

Except Batch Normalization, other training parameters are kept the same as discussed for Section 3.Figure 13(a) shows the training loss for models along the line segment and MC curve joining the pair of iterates from SGDR.

For the two pairs {w 30 , w 150 } and {w 70 , w 150 }, we again observe a higher training loss for models on the line segment, suggesting that for this setup too, SGDR finds paths that move over a barrier in the training loss landscape.

We further evaluate the intermediate points FIG0 , we present the CCA similarity plots comparing two pairs of models: epochs 10 and 150, and epochs 150 and 155.

The (i, j) th block of the matrix denotes the correlation between the i th layer of the first model and the j th layer of the other.

A high correlation implies that the layers learn similar representations and vice versa.

We present the former to compare against the typical stepwise or linear decay of SGD, and the latter to demonstrate the immediate effect of restarting on the model.

Raghu et al. (2017) showed in their work that for typical SGD training, a CCA similarity plot between a partially and completed trained network reveals that the activations of the shallower layers bears closer resemblance in the two models than the deeper layers.

We note that, despite the restart, a similar tendency is seen in SGDR training as well.

This again suggests that the restart does not greatly impact the model, both in weights and representations, and especially so in the shallower layers.

A comparison of epochs 150 and 155, i.e., before and after a restart also stands as evidence for this hypothesis.

In Figure 4 (d), we show that the stability induced by warmup when training with large batches and learning rates can also be obtained by holding the FC stack frozen.

This experiment was conducted on the VGG-11 network (Simonyan & Zisserman, 2014) .

To demonstrate the generality of our claim, we present additional experiments on two ResNet architectures: 18 and 32.

The setup for this experiment is identical to the VGG-11 one with one change: instead of the learning rate being set to 2.5, which is the learning rate for SB (0.05) times the batch size increase (50??), we set it to 5.0 since SB training is better with 0.1.

For the warmup case, we linearly increase the learning rate from 0 to 5 again for 20 epochs.

Experiments on other configurations yielded similar results.

Whether these results remain true also for training larger datasets, such as ImageNet, remains to be shown and is a topic of future research.

<|TLDR|>

@highlight

We use empirical tools of mode connectivity and SVCCA to investigate neural network training heuristics of learning rate restarts, warmup and knowledge distillation.