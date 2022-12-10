Deep neural models, such as convolutional and recurrent networks, achieve phenomenal results over spatial data such as images and text.

However, when considering tabular data, gradient boosting of decision trees (GBDT) remains the method of choice.

Aiming to bridge this gap, we propose \emph{deep neural forests} (DNF) --  a novel architecture that combines elements from decision trees as well as dense residual connections.

We present the results of extensive empirical study in which we examine the performance of GBDTs, DNFs and (deep) fully-connected networks.

These results indicate that DNFs achieve comparable results to GBDTs on tabular data, and open the door to end-to-end neural modeling of multi-modal data.

To this end, we present a successful application of DNFs as part of a hybrid architecture for a multi-modal driving scene understanding classification task.

While deep neural models have gained supremacy in many applications, it is often the case that the winning hypothesis class in learning problems involving tabular data is decision forests.

Indeed, in Kaggle competitions, gradient boosting decision trees (GBDTs) (Chen & Guestrin, 2016; Friedman, 2001) are often the superior model.

1 Decision forest techniques have several distinct advantages: they can handle heterogeneous feature types, they are insensitive to feature scaling, and perhaps, most importantly, they perform a rudimentary kind of "feature engineering" automatically by considering conjunctions of decision stumps.

These types of features may be a key reason for the relative success of GBDTs over tabular data.

In contrast, deep neural models (CNNs, RNNs) have become the preeminent favorites in cases where the data exhibit a spatial proximity structure (namely, video, images, audio, and text) .

In certain problems, such as image classification, by restricting the model to exploit prior knowledge of the spatial structure (e.g., translation and scale invariances), these models are capable of generating problem dependent representations that almost completely overcome the need for expert knowledge.

However, in the case of tabular data, it is often very hard to construct (deep) neural models that achieve performance on the level of GBDTs.

In particular, the "default" fully connected networks (FCNs), which do not reflect any specific inductive bias toward tabular data, are often inferior to GBDTs on these data.

There have been a few works aiming at the construction of neural models for tabular data (see Section 2).

However, for the most part, these attempts relied on conventional decision tree training in their loop and currently, there is still no widely accepted neural architecture that can effectively replace GBDTs.

This deficiency prevents or makes it harder to utilize neural models in many settings and constitutes a lacuna in our understanding of neural networks.

Our objective in this work is to create a neural architecture that can be trained end-to-end using gradient based optimization and achieve comparable or better performance to GBDTs on tabular data.

Such an architecture is desirable because it will allow the treatment of multi-modal data involving both tabular and spatial data in an integrated manner while enjoying the best of both GBDTs and deep models.

Moreover, while GBDTs can handle medium size datasets ("Kaggle scale"), they do not scale well to very large datasets ("Google scale"), where their biggest computational disadvantage is the need to store (almost) the entire dataset in-memory 2 (see Appendix C for details as well as a real-life example of this limitation).

A purely neural model for tabular data, which is trained with SGD, should be scalable beyond these limits.

A key-point in successfully applying deep models is the construction of architectures that contain inductive bias relevant to the application domain.

This quest for appropriate inductive bias in the case of tabular data is not yet well understood (not to mention that there can be many kinds of tabular data).

However, we do know that tree and forest methods tend to perform better than vanilla FCNs on these data.

Thus, our strategy is to borrow properties of decision trees and forests into the network structure.

We present a generic neural architecture whose performance can be empirically similar to GBDTs on tabular data.

The new architecture, called Deep Neural Forest (DNF), combines elements from both decision forests and residual/dense nets.

The main building block of the proposed architecture is a stack of neural branches (NBs), which are neural approximations of oblique decision branches that are connected via dense residual links (Huang et al., 2017) .

The final DNF we propose is an ensemble of such stacks (see details in Section 3).

We present an empirical study where we compare DNFs to the FCNs and GBDTs baselines, optimized over their critical parameters.

We begin with a synthetic checkerboard problem, which can be viewed as a hypothetical challenging tabular classification task.

We then consider several relatively large tasks, including two past Kaggle competitions.

Our results indicate that DNFs consistently outperform FCNs, and achieve comparable performance to GBDTs.

We also address applications of DNFs over multi-modal data and examine an integrated application of DNFs, CNNs and LSTMs over a multi-modal classification task for driving scene understanding involving both sensor recording and video (Ramanishka et al., 2018) .

We show that the replacement of the FCN component by DNF in the hybrid deep architecture of Ramanishka et al. (2018) , which was designed to handle these multi-modal data, leads to significant performance improvement.

There have been a few attempts to construct neural networks with improved performance on tabular data.

In all these works, decision trees or forests are considered as the competition.

A recurring idea in some of these works is the explicit use of conventional decision tree induction algorithms, such as ID3 (Quinlan, 1979) , or conventional forest methods, such as GBDT (Friedman, 2001) that are trained over the data at hand, and then parameters of the resulting decision trees are explicitly or implicitly "imported" into a neural network using teacher-student distillation (Ke et al., 2018) , explicit embedding of tree paths in a specialized network architecture (Seyedhosseini & Tasdizen, 2015) , and explicit utilization of forests as the main building block of layers (Feng et al., 2018) .

This reliance on conventional decision tree or forest methods as an integral part of the proposed solution prevents end-to-end neural optimization, as we propose here.

This deficiency is not only a theoretical nuisance but also makes it hard to use such models on very large datasets and in combination with other neural modules (see also discussion in Appendix Section C).

There are a few other recent techniques aiming to cope with tabular data using pure neural optimization.

Yang et al. (2018) considered a method to approximate a single node of a decision tree using a soft binning function that transforms continuous features into one-hot features.

The significant advantage of this tree based model is that it is intrinsically interpretable, as if it were a conventional decision tree.

Across a number of datasets, this method obtained results comparable to a single decision tree and an FCN (with two hidden layers).

This method, however, is limited to settings in which the number of features is small (e.g., 12).

Focusing on microbiome data, a recent study by Shavitt & Segal (2018) presented an elegant regularization technique, which produces extremely sparse networks that are suitable for microbiome tabular datasets with relatively large feature spaces that only have a small number of informative features.

The main building block in our construction is a Neural Branch (NB).

An NB represents a "soft conjunction" of a fixed number of (orthonormal) linear models.

The purpose of an NB is to emulate the inductive bias existing in a path from the root to leaf in a decision tree (i.e., a branch).

The second important element is the use of depth to allow for composite, hierarchical features.

Thus, depth is created by vertically stacking layers of NBs using dense residual links as in DenseNet (Huang et al., 2017) .

We now provide a detailed description of the proposed architecture.

When ignoring the hierarchy, a decision tree can be viewed as a disjunction of a set of conjunctions over decision stumps.

Each conjunction corresponds to one path from the root to a leaf.

Thus, any decision tree can be represented as a disjunctive normal form formula.

A Neural Tree (NT) is an approximation of a disjunctive normal form formula.

Each conjunction in the NT is called a neural branch (NB).

While the basic units in a decision tree are decision stumps, the NT uses affine models, as in oblique decision trees (Murthy et al., 1994) .

The NT is constructed using soft binary OR and AND gates.

For a given (binary) vector x = (x 1 , . . . , x d ) ∈ {−1, 1} d .

We implement soft, differentiable versions of such gates as follows.

Notice that by replacing tanh by a binary activation, we obtain an exact implementation of the corresponding logical gates, which are well known (Anthony, 2005) .

3 Importantly, both units do not have any trainable parameters.

For simplicity, given a vector x ∈ R d we define the AND(x) operator on r sub-groups, each of size k, as follows,

where

Formally, the NT is a three-layer network (two hidden layers), where only the first hidden layer, which represents the internal decision nodes (oblique separators), is trainable.

Denoting by x ∈ R d a column of input feature vector, the functional form of an NT(x) : R d → R with a layer of r NBs, each NB with depth k, is

where NB(x) : R d → R r is the output of r NBs, W ∈ R d×kr determines the (oblique) linear separators in each of the "nodes" such that each of its columns corresponds to one "node".

and b ∈ R kr is a bias vector term that corresponds to the threshold term in decision tree nodes.

In our design, each decision node belongs only to a single branch.

When considering the decision boundaries induced by a single branch of an axis-aligned decision tree, it is clear that the decision boundary of a specific node is usually orthogonal to all the other decision boundaries defined by the other nodes in the same branch.

We impose this constraint, which prevents unnecessary redundancy, by "encouraging" orthonormality through the loss function.

Thus, when optimizing NTs, we include the following orthonormality constraint in our loss function, which imposes both orthogonality and unit length regularization simultaneously.

Figure 1: A DNT with four layers of NBs, each layer with five NBs.

The input of each layer is a concatenation of the input of the previous layer with its output.

Moreover, the input x is multiplied element-wise with a binary maskm (see section 3.3) before it is fed into the DNT.

where λ is a hyper-parameter, andĨ ∈ R kr×kr is defined bỹ

if nodes i and j are in the same branch;

The "magic" in deep neural models is their ability to create a hierarchical structure by automatically creating composite features.

One of the most interesting convolutional architectures is DenseNet (Huang et al., 2017) whose connectivity pattern ensures maximum information flow between layers.

Each DenseNet layer is connected directly with each other layer.

Moreover, in contrast to ResNet, which combines features through summation before they are passed into a layer, DenseNet combines features by concatenating them.

A notable advantage of DenseNet is its improved flow of information and gradients throughout the network, which makes it easy to train.

Moreover, the DenseNet connectivity pattern elicits the generation of composite features involving a mix of high and low-level features.

Since we want to retain these desirable properties of deep neural models, we introduce depth into our construct through dense residual links.

Thus, A Deep Neural Tree (DNT) is a stack of layers of NBs that are interconnected using dense residual links, while the OR gate is applied only on the last NBs layer.

Clearly, an NT is a DNT with a single layer of NBs and, in the sequel, we refer to an NT as a DNT.

A diagram of a DNT with four layers of NBs appears in Figure 1 .

One of the key components of decision trees is their greedy feature selection at any split.

Such a component gives the decision trees, among other things, the ability to exclude irrelevant features.

Li et al. (2016) presented a neural component for feature selection.

Their solution is based on a heavily regularized (with elastic net regularization) mask that multiplies the input vector elementwise.

In their work, they mention a crucial drawback in the proposed component, which arises in cases where the mask weight of a specific feature was approximately zero.

In such cases, the corresponding weight in the first layer that multiplies this feature became very large.

They tackled this problem by applying heavy regularization on the network layers.

In our study, for each DNT, we add an independent mask that multiplies the input vector elementwise.

A heavy elastic net regularization is applied to the mask weights, and a binary threshold is used to circumvent the above pitfall.

Denoting this mask by m ∈ R d , the feature selection component is formally defined as follows,

DifferentiableSign(x) sign(x), forward pass; σ(x), backward pass;

m DifferentiableSign(|m| − ), Wherem is the mask that multiplies the input, defines an epsilon neighborhood around zero for which the value of the mask is set to zero, and σ is the sigmoid activation.

In words, if the value of the regularized mask is close to zero, set it to exact zero; otherwise, set it to one.

Since the sign function is not differentiable, we use a smooth approximation of the sign function for calculating the gradients in the backward pass.

The power of decision trees can be significantly amplified when using many of them via ensemble methods, such as bagging or boosting.

The final Deep Neural Forest (DNF) architecture is a weighted ensemble of DNTs (see a diagram on Appendix D).

A DNF is implemented by concatenating the DNTs outputs and applying one fully-connected layer.

The functional form of a DN F (x) is,

where w i ∈ R are trainable weights, which are optimized simultaneously with the DNTs.

Accordingly, we will refer to a weighted ensemble of NTs as Neural Forest (NF).

It is well-known that high-quality ensembles should be constructed from a diverse set of low-bias base learners.

To amplify ensemble diversity, we used both localization and random feature sampling.

These techniques are applied individually for each base learner (DNT).

Meir et al. (2000) showed the benefit of using an ensemble of localized base learners.

Motivated by their result, we assign for each DNT a Gaussian with a trainable parameter mean vector µ, and a constant (isotropic) covariance matrix Σ = σ 2 I (where σ is a fixed hyperparameter for the entire forest).

For each instance x, the output of the DNF is thus,

where D is the probability density function of the multivariate normal distribution, and W p is a learnable projection matrix shared among all DNTs, which is used to obtain a linear embedding of the input to a low dimension.

We note that this matrix is necessary to avoid learning of high-dimensional Gaussian, for which the probability density function is approximately zero (using isotropic covariance matrix Σ = σ 2 I with σ > 1).

This mechanism allows each DNT to specialize in a certain local sub-space and makes it oblivious to instances that are distant from its focal point µ.

Finally, another method we used is feature sampling, which is widely used in tree-based algorithms to increase diversity.

Therefore, for each DNT, we randomly sample a fixed subset of features, where the number of features to be drawn is a hyper-parameter.

To gain some intuition and perspective on the performance of DNFs, GBDTs, and FCNs, in this section, we consider simple synthetic classification tasks that can be viewed as an extreme case of tabular data.

FCNs (even those with one hidden layer) are universal approximators (Cybenko, 1989) and can represent a good approximation to any (nicely behaved) function; nevertheless, training them using gradient methods is sometimes challenging.

A well-known hard case is the problem of learning parity.

While it is fairly easy to manually construct a small network that computes parity (Wilamowski et al., 2003) , it is notoriously hard to train these networks to learn parity and similar problems using gradient methods (Shalev-Shwartz et al., 2017; Abbe & Sandon, 2018) .

Somewhat surprisingly, we show here that the training of FCNs is difficult even in much simpler checkerboard problems, which appear benign compared to parity.

In the checkerboard classification problem, the feature space, X = [−1, 1] 2 , is a two-dimensional square.

Each of the two features is uniformly distributed.

In a n × n checkerboard instance , the binary label, Y = {±1}, is defined by evenly dividing X into n 2 uniform squares, and the label is alternating along rows and columns as in the game of checkerboard.

A 7 × 7 checkerboard example is depicted in Figure 2a , where blue and orange dots represent ±1 labeled points sampled from the underlying distribution.

Checkerboard problems naturally extend the XOR problem where a 2 × 2 checkerboard is XOR.

It is not hard to construct a 2-hidden layers FCN that solves the checkerboard perfectly.

However, in the following experiment, we observe that such a perfect solution is not reachable via SGD training.

Consider the following experiment where we tested FCNs, GBDTs and DNFs over 19 n×n checkerboard instances where n = 2, 3, . . .

, 20.

For each of these checkerboard problems, we randomly generated 10,000 i.i.d.

labeled samples (partitioned into to 1K points for training, 1K for validation and 8K for testing) over which we evaluated the performance of the three models.

For GBDT we employed the powerful XGBoost implementation (Chen & Guestrin, 2016) .

The hyperparameters for FCNs and GBDTs were aggressively optimized using an exhaustive grid search.

For each checkerboard instance, a total of 1000 different configurations were tested for the FCNs, which included architectures with depth (number of hidden layers) in the range [1, 4] and width in the set {64, 128, 256, 512, 1024}. Moreover, the hyperparameter optimization included a search for a dropout rate and L 1 regularization coefficient.

The FCNs and the DNFs were trained using stochastic gradient descent (SGD) with the Adam optimizer and a learning rate scheduler.

We did not limit the number of epochs, but we used an early stopping trigger consisting of 50 epochs.

Accordingly, an exhaustive grid search was done for the decision tree algorithms where exact details of the hyperparameter ranges can be found on Appendix A in Table 3 .

The checkerboard experiment results are depicted in Figure 2b .

The x-axis is the checkerboard size (n), and the y-axis is accuracy over the test set, where each point is the mean over five independent trials and error bars represent standard error of the mean.

We see that the performance of all three methods is deteriorated when the checkerboard size is increased.

This tendency can be anticipated because the average number of training points in each checkerboard cell is decreasing (we keep the training set size 1000 for all boards).

It is surprising, however, that FCNs completely fail to generate prediction better than random guessing for n ≥ 14 board sizes.

Moreover, it is evident that XGBoost consistently and significantly outperform the FCNs over all problems with n > 2.

While DNFs are slightly behind at small ns, for all n > 9 they achieve the best results.

Interestingly, the best results of the FCNs were obtained using networks containing millions of parameters, while DNFs mostly outperformed them using only (approximately) 4K trainable parameters.

As a side note, we found that the batch size has a critical effect on the results; with mini-batches larger than 512, FCNs do not exhibit any advantage over random guessing for all board sizes n ≥ 14.

Before continuing with additional synthetic setups, we emphasize that Checkerboard-like phenomena are quite common in tabular datasets.

Consider, for example, the well-known Titanic dataset (Dua & Graff, 2017) .

Consider male age versus survival probability where a missing age is labeled with −1.

Observing the figures on Appendix B, we can see a major increase from −1 to kids at ages 0-12.

Beyond age 12 we see a major decrease for the adult male population.

At age 25 we see a small increase and then again at age 45 we see a small decrease.

This example also extends to the 2D case where both age and ticket fares are used.

To demonstrate the ability of DNFs to deal with irrelevant features, we generate the data from a simple XOR-problem (2 × 2 checkerboard) with additional irrelevant features, where each irrelevant feature was drawn from the standard normal distribution.

The results are depicted in Figure 2c .

Clearly, DNFs sustain excellent performance with increasing numbers of irrelevant features, while we see some deterioration of the other methods.

As might be expected, the top-performing FCNs were networks with one hidden layer and strong L 1 regularization.

Here again, the representation efficiency of DNFs was evident; while the FCNs utilized around 10K neurons, the DNFs required less than 200 neurons.

As the number of irrelevant features increases, it can be seen that XGBoost experiences some trouble, which we believe is mainly due to the high symmetry of the problem, where both relevant and irrelevant features have approximately zero information gain.

The purpose of this last synthetic experiment is to examine the effect of depth (through dense residual connectivity) in DNFs.

We, therefore, compare the performance of a DNF and a basic neural tree (NT), which does not include residual links (see Section 3.1).

The data was generated using a checkerboard together with an additional binary feature that was uniformly sampled from {0, 1}. The label of each instance is a XOR between the binary feature and the label defined by the checkerboard.

In order to solve this problem, an interaction of a low-level feature (the binary feature) with high-level features (the checkerboard pattern) must be learned.

As in the first experiment, we considered 19 board sizes with n = 2, 3, . . .

, 20.

The results can be seen in Figure 2d .

While the NTs excel on checkerboards with n ∈ {4, . . .

, 8}, it is clear that for n ≥ 11 the DNF is the leading model.

FCNs were not included in this study because their performance on the checkerboard alone was already significantly inferior.

In this section, we examine the performance of DNFs and the baselines (FCNs and XGBoost) on several tabular datasets.

The datasets used in this study are from Kaggle 4 and OpenML 5 (Vanschoren et al., 2014) .

A summary of these datasets appears on Appendix E.

For each dataset, all models were trained on the raw data without any feature engineering or selection.

Only feature standardization was applied.

Hyper-parameters for each model were optimized using a grid search (the range for each hyper-parameter in Appendix A in Table 4 ).

DNFs were trained using stochastic gradient descent (SGD) with the Adam optimizer and a learning rate scheduler.

Dropout was applied to the layer obtained from the concatenation of the DNTs, and L 1 regularization was applied on the last layer (which computes a weighted sum of the DNTs).

The FCNs were trained with SGD, Adam, and a learning rate scheduler as well.

We did not limit the number of epochs but used an early stopping after 30 epochs.

The results are summarized in Table 1 .

For each dataset, the best result appears in bold.

Notice that 'log loss' scores should be minimized and 'roc auc' -maximized.

It is evident that the DNF performance is on par with the XGBoost performance, while FCNs are way behind.

Table 1 : Tabular data experiments: mean score over 5-fold cross-validation.

For the Kaggle competitions, we also included Kaggle-computed results obtained via the "late submission" system.

So far, GBDTs have dominated the tabular data domain, while the visual and textual domains have been entirely dominated by deep models (CNNs, RNNs).

In cases of multi-modal tasks involving tabular data as well, e.g., images, it is tempting to try and combine GBDTs and CNNs.

However, as GBDTs are not differentiable (and not scaleable; see Appendix C), their integration with CNNs can be problematic.

Typically, FCNs are used instead of GBDTs, and as we know, their utilization can degrade performance.

In this section, we examine the utilization of DNFs, instead of FCNs in a hybrid model to handle the multi-modal data of the Honda Research Institute Driving Dataset (HRI-DD) (Ramanishka et al., 2018) .

These data span 104 hours of real human driving and combine synchronized video and sensors measurements.

The video consists of 1280 × 720 frames at 30 fps, and there are six different sensors: car speed, accelerator and braking pedal positions, yaw rate, steering wheel angle, and the rotation speed of the steering wheel.

Four classification tasks were defined over these data, all of which are related to understanding driver behavior.

We considered the first task: Goal-oriented action that involves the driver's manipulation of the vehicle in a navigation task.

This is an 11-class task, and among the classes are 'right turn', 'left turn', 'branch', 'line change' and 'merge'.

In their study, Ramanishka et al. (2018) presented baseline results for this task.

Their architecture consists of three components: for handling images, they used a CNN, whose main body is a pretrained InceptionResnet-V2 (Szegedy et al., 2017) , with an additional trainable convolutional layer.

For the sensor data, they used an FCN.

The embedding obtained from these two components was fused (concatenated) and then fed into an LSTM.

In our study, we utilized the exact same structure with exactly the same hyper-parameters and training parameters.

The only change we made was to replace the FCN component with a DNF.

We performed a comparative experiment on two tasks.

The first task is to predict the navigation labels using only the sensors (i.e., in this setting the CNN was omitted), which is a composite multi-modal task that combines tabular data in a sequential manner (hence, the LSTM component remains).

In the second task, we utilized both the video and the sensors.

This task is a (composite) multi-modal task that combines tabular data and images as a time-series.

The results over these two tasks are summarized in Table 2 .

The baseline results (those with the FCN in their model) that we present were obtained by Ramanishka et al. (2018 Table 2 : Each column is the average precision per class obtained on the test set.

The last column is the mean average precision of all classes.

The first two rows correspond to the sensor-only task and last two rows, to the sensors+video data.

We introduced deep neural forest (DNF) -a novel deep architecture designed to handle tabular data.

DNFs emulate some of the inductive bias existing in decision trees, and elicit the generation of composite features using depth and dense residual connectivity.

Our empirical study of DNFs suggests that they significantly outperform FCNs over tabular data tasks, and achieve comparable performance to GBDTs, which so far were the SOTA choice for such data.

Our initial study of a complex real-life multi-modal scenario of driving scene classification yielded substantial performance gains.

This work raises several interesting open challenges.

First, more work is needed to fully substantiate DNFs and distill the essential elements in this architecture.

Second, adopting the sequential optimization approach of GBDTs to DNFs can potentially lead to further large improvements, in the same way that GBDTs improve over random forests.

Finally, we believe that a better theoretical understanding of the characteristics and inductive bias of tabular data can play a key role in achieving further performance gains in tabular and multi-modal settings.

The exact details of the hyperparameter ranges that considered in section 4 can be found in In Table  .

Accordingly, the details of section 5 can be found in In Table .

XGBoost To demonstrate the checkerboard phenomena in tabular data, we plot the probability estimates for the Titanic dataset (Dua & Graff, 2017) .

The goal of this task is to predict individual passenger survival.

For the demonstration here, we considered two real-valued features.

The first is age, which might be missing and replaced with −1 (a common practice).

The second is the ticket fare.

The effect of gender is so great, so we chose to display plots on the male population.

The first plot is univariate, where the x-axis is age and the y-axis is the survival probability.

Clearly, there is a sharp transition change from −1 (missing data) to 0 (babies) and another sharp transition at 14 (kids).

There are two more softer transitions at 25 and 42.

These transitions indicate a checkerboard behavior.

The second plot if bivariate, where the x-axis is age and the y-axis is ticket fare, while the color indicates the survival probability.

The checkerboard-like pattern is apparent.

Figure 3 : Survival probability of male population taken from the Titanic dataset.

On the left plot, we see the survival vs age which exhibits multiple sharp transitions that resemble a 1D checkerboard behavior.

On the right plot, the example is extended to a 2D checkerboard, survival vs age and fare.

Gradient boosting (XGBoost, LightGBM, CatBoost) biggest computation disadvantage is the need to store (almost) the entire dataset in-memory.

Several optimizations are deployed to help with this issue.

LightGBM (max bin parameter) and CatBoost (feature border type parameter) perform pre-computation that quantizes features to small integers.

A random subset of the features can be selected for the entire tree (not just per node) 6 .

At small to medium scale, these optimizations enable training to be performed efficiently on a single computer.

But when considering large datasets, such as the Honda Research Institute Driving Dataset (HRI-DD) of Section 6, GBDT techniques are less effective.

For instance, the HRI-DD set consists of ∼1.2M samples, where each sample is represented by 6 floats from the sensors, and 8 × 8 × 1536 floats for the images.

In order to hold all these data in memory we thus need ∼440GB of RAM.

While such RAM sizes are available, to achieve reasonable performance on the HRI-DD task, one should model these data as a time-series, and each data point needs to be represented as a vector instances, resulting in memory requirements that can easily exceed the available RAM.

Figure 4 : A DNF is implemented by concatenating the DNTs outputs and applying one fullyconnected layer.

A description of the tabular datasets that were used in section 5,

@highlight

An architecture for tabular data, which emulates branches of decision trees and uses dense residual connectivity 

@highlight

This paper proposes deep neural forest, an algorithm which targets tabular data and integrates strong points of gradient boosting of decision trees.

@highlight

A novel neural network architecture mimicking how decision forests work to tackle the general problem of training deep models for tabular data and showcasing effectiveness on par with GBDT.