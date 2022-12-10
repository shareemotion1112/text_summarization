Gradient-based meta-learning techniques are both widely applicable and proficient at solving challenging few-shot learning and fast adaptation problems.

However, they have practical difficulties when operating on high-dimensional parameter spaces in extreme low-data regimes.

We show that it is possible to bypass these limitations by learning a data-dependent latent generative representation of model parameters, and performing gradient-based meta-learning in this low-dimensional latent space.

The resulting approach, latent embedding optimization (LEO), decouples the gradient-based adaptation procedure from the underlying high-dimensional space of model parameters.

Our evaluation shows that LEO can achieve state-of-the-art performance on the competitive miniImageNet and tieredImageNet few-shot classification tasks.

Further analysis indicates LEO is able to capture uncertainty in the data, and can perform adaptation more effectively by optimizing in latent space.

Humans have a remarkable ability to quickly grasp new concepts from a very small number of examples or a limited amount of experience, leveraging prior knowledge and context.

In contrast, traditional deep learning approaches BID24 BID39 treat each task independently and hence are often data inefficient -despite providing significant performance improvements across the board, such as for image classification BID41 BID14 , reinforcement learning BID29 BID40 , and machine translation BID3 BID44 .

Just as humans can efficiently learn new tasks, it is desirable for learning algorithms to quickly adapt to and incorporate new and unseen information.

Few-shot learning tasks challenge models to learn a new concept or behaviour with very few examples or limited experience BID6 BID23 .

One approach to address this class of problems is meta-learning, a broad family of techniques focused on learning how to learn or to quickly adapt to new information.

More specifically, optimization-based meta-learning approaches BID34 BID7 aim to find a single set of model parameters that can be adapted with a few steps of gradient descent to individual tasks.

However, using only a few samples (typically 1 or 5) to compute gradients in a high-dimensional parameter space could make generalization difficult, especially under the constraint of a shared starting point for task-specific adaptation.

In this work we propose a new approach, named Latent Embedding Optimization (LEO), which learns a low-dimensional latent embedding of model parameters and performs optimization-based meta-learning in this space.

Intuitively, the approach provides two advantages.

First, the initial parameters for a new task are conditioned on the training data, which enables a task-specific starting point for adaptation.

By incorporating a relation network into the encoder, this initialization can better consider the joint relationship between all of the input data.

Second, by optimizing in the lower-dimensional latent space, the approach can adapt the behaviour of the model more effectively.

Further, by allowing this process to be stochastic, the ambiguities present in the few-shot data regime can be expressed.

We demonstrate that LEO achieves state-of-the-art results on both the miniImageNet and tieredImageNet datasets, and run an ablation study and further analysis to show that both conditional parameter generation and optimization in latent space are critical for the success of the method.

Source code for our experiments is available at https://github.com/deepmind/leo.

We define the N -way K-shot problem using the episodic formulation of BID47 .

Each task instance T i is a classification problem sampled from a task distribution p(T ).

The tasks are divided into a training meta-set S tr , validation meta-set S val , and test meta-set S test , each with a disjoint set of target classes (i.e., a class seen during testing is not seen during training).

The validation meta-set is used for model selection, and the testing meta-set is used only for final evaluation.

Each task instance T i ∼ p (T ) is composed of a training set D tr and validation set D val , and only contains N classes randomly selected from the appropriate meta-set (e.g. for a task instance in the training meta-set, the classes are a subset of those available in S tr ).

In most setups, the training set D tr = (x k n , y k n ) | k = 1 . . .

K; n = 1 . . .

N contains K samples for each class.

The validation set D val can contain several other samples from the same classes, providing an estimate of generalization performance on the N classes for this problem instance.

We note that the validation set of a problem instance D val (used to optimize a meta-learning objective) should not be confused with the held-out validation meta-set S val (used for model selection).

Model-agnostic meta-learning (MAML) BID7 is an approach to optimization-based meta-learning that is related to our work.

For some parametric model f θ , MAML aims to find a single set of parameters θ which, using a few optimization steps, can be successfully adapted to any novel task sampled from the same distribution.

For a particular task instance T i = D tr , D val , the parameters are adapted to task-specific model parameters θ i by applying some differentiable function, typically an update rule of the form: DISPLAYFORM0 where G is typically implemented as a step of gradient descent on the few-shot training set D tr , DISPLAYFORM1 Generally, multiple sequential adaptation steps can be applied.

The learning rate α can also be meta-learned concurrently, in which case we refer to this algorithm as Meta-SGD .

During meta-training, the parameters θ are updated by back-propagating through the adaptation procedure, in order to reduce errors on the validation set D val : DISPLAYFORM2 The approach includes the main ingredients of optimization-based meta-learning with neural networks: initialization is done by maintaining an explicit set of model parameters θ; the adaptation procedure, or "inner loop", takes θ as input and returns θ i adapted specifically for task instance T i , by iteratively using gradient descent (Eq. 1); and termination, which is handled simply by choosing a fixed number of optimization steps in the "inner loop".

MAML updates θ by differentiating through the "inner loop" in order to minimize errors of instance-specific adapted models f θ i on the corresponding validation set (Eq. 2).

We refer to this process as the "outer loop" of meta-learning.

In the next section we use the same stages to describe Latent Embedding Optimization (LEO).

The primary contribution of this paper is to show that it is possible, and indeed beneficial, to decouple optimization-based meta-learning techniques from the high-dimensional space of model parameters.

We achieve this by learning a stochastic latent space with an information bottleneck, conditioned on the input data, from which the high-dimensional parameters are generated.

Require: Training meta-set S tr ∈ T Require: Learning rates α, η 1: Randomly initialize φe, φr, φ d 2: Let φ = {φe, φr, φ d , α} 3: while not converged do 4: for number of tasks in batch do 5:Sample task instance Ti ∼ S DISPLAYFORM0 Encode D tr to z using g φe and g φr 8:Decode z to initial params θi using g φ d 9:Initialize z = z, θ i = θi 10:for number of adaptation steps do 11:Compute training loss L DISPLAYFORM1 Perform gradient step w.r.t.

z : DISPLAYFORM2 Decode z to obtain θ i using g φ d 14:end for 15:Compute validation loss L DISPLAYFORM3 end for 17:Perform gradient step w.r.t φ: DISPLAYFORM4 f θ i 18: end while Figure 1 : High-level intuition for LEO.

While MAML operates directly in a high dimensional parameter space Θ, LEO performs meta-learning within a low-dimensional latent space Z, from which the parameters are generated.

Instead of explicitly instantiating and maintaining a unique set of model parameters θ, as in MAML, we learn a generative distribution of model parameters which serves the same purpose.

This is a natural extension: we relax the requirement of finding a single optimal θ * ∈ Θ to that of approximating a data-dependent conditional probability distribution over Θ, which can be more expressive.

The choice of architecture, composed of an encoding process, and decoding (or parameter generation) process, enables us to perform the MAML gradient-based adaptation steps (or "inner loop") in the learned, low-dimensional embedding space of the parameter generative model (Figure 1 ).

The high-level operation is then as follows (Algorithm 1).

First, given a task instance T i , the inputs {x k n } are passed through a stochastic encoder to produce a latent code z, which is then decoded to parameters θ i using a parameter generator 1 .

Given these instantiated model parameters, one or more adaptation steps are applied in the latent space, by differentiating the loss with respect to z, taking a gradient step to get z , decoding new model parameters, and obtaining the new loss.

Finally, optimized codes are decoded to produce the final adapted parameters θ i , which can be used to perform the task, or compute the task-specific meta-loss.

In this way, LEO incorporates aspects of model-based and optimization-based meta-learning, producing parameters that are first conditioned on the input data and then adapted by gradient descent.

FIG0 shows the architecture of the resulting network.

Intuitively, the decoder is akin to a generative model, mapping from a low-dimensional latent code to a distribution over model parameters.

The encoding process ensures that the initial latent code and parameters before gradient-based adaptation are already data-dependent.

This encoding process also exploits a relation network that allows the latent code to be context-dependent, considering the pairwise relationship between all classes in the problem instance.

In the following sections, we explain the LEO procedure more formally.

The first stage is to instantiate the model parameters that will be adapted to each task instance.

Whereas MAML explicitly maintains a single set of model parameters, LEO utilises a datadependent latent encoding which is then decoded to generate the actual initial parameters.

In what follows, we describe an encoding scheme which leverages a relation network to map the few-shot examples into a single latent vector.

This design choice allows the approach to consider context when producing a parameter initialization.

Intuitively, decision boundaries required for fine-grained distinctions between similar classes might need to be different from those for broader classification.

Encoding The encoding process involves a simple feed-forward mapping of each data point, followed by a relation network that considers the pair-wise relationship between the data in the problem instance.

The overall encoding process is defined in Eq. 3, and proceeds as follows.

First, each example from a problem instance DISPLAYFORM0 is processed by an encoder network g φe : R nx → R n h , which maps from input space to a code in an intermediate hidden-layer code space H. Then, codes in H corresponding to different training examples are concatenated pair-wise (resulting in (N K) 2 pairs in the case of K-shot classification) and processed by a relation network g φr , in a similar fashion to and BID43 .

The (N K) 2 outputs are grouped by class and averaged within each group to obtain the (2 × N ) parameters of a probability distribution in a low-dimensional space Z = R nz , where n z dim(θ), for each of the N classes.

Thus, given the K-shot training samples corresponding to a class n: DISPLAYFORM1 . .

K the encoder g φe and relation network g φr together parameterize a class-conditional multivariate Gaussian distribution with a diagonal covariance, which we can sample from in order to output a class-dependent latent code z n ∈ Z as follows: DISPLAYFORM2 Intuitively, the encoder and relation network define a stochastic mapping from one or more class examples to a single code in the latent embedding space Z corresponding to that class.

The final latent code can be obtained as the concatenation of class-dependent codes: DISPLAYFORM3 Decoding Without loss of generality, for few-shot classification, we can use the class-specific latent codes to instantiate just the top layer weights of the classifier.

This allows the meta-learning in latent space to modulate the important high-level parameters of the classifier, without requiring the generator to produce very high-dimensional parameters.

In this case, f θ i is a N -way linear softmax classifier, with model parameters θ i = w n | n = 1 . . .

N , and each x k n can be either the raw input or some learned representation 2 .

Then, given the latent codes z n ∈ Z, n = 1 . . .

N , the decoder function g φ d : Z → Θ is used to parameterize a Gaussian distribution with diagonal covariance in model parameter space Θ, from which we can sample class-dependent parameters w n : DISPLAYFORM4 In other words, codes z n are mapped independently to the top-layer parameters θ i of a softmax classifier using the decoder g φ d , which is essentially a stochastic generator of model parameters.

Given the decoded parameters, we can then define the "inner loop" classification loss using the cross-entropy function, as follows: DISPLAYFORM0 It is important to note that the decoder g φ d is a differentiable mapping between the latent space Z and the higher-dimensional model parameter space Θ. Primarily, this allows gradient-based optimization of the latent codes with respect to the training loss, with z n = z n − α∇ zn L tr Ti .

The decoder g φ d will convert adapted latent codes z n to effective model parameters θ i for each adaptation step, which can be repeated several times, as in Algorithm 1.

In addition, by backpropagating errors through the decoder, the encoder and relation net can learn to provide a data-conditioned latent encoding z that produces an appropriate initialization point θ i for the classifier model.

For each task instance T i , the initialization and adaptation procedure produce a new classifier f θ i tailored to the training set D tr of the instance, which we can then evaluate on the validation set of that instance D val .

During meta-training we use that evaluation to differentiate through the "inner loop" and update the encoder, relation, and decoder network parameters: φ e , φ r , and φ d .

Meta-training is performed by minimizing the following objective: DISPLAYFORM0 where p(z n ) = N (0, I).

Similar to the loss defined in BID15 we use a weighted KL-divergence term to regularize the latent space and encourage the generative model to learn a disentangled embedding, which should also simplify the LEO "inner loop" by removing correlations between latent space gradient dimensions.

The third term in Eq. (6) encourages the encoder and relation net to output a parameter initialization that is close to the adapted code, thereby reducing the load of the adaptation procedure if possible.

L 2 regularization was used with all weights of the model, as well as a soft, layer-wise orthogonality constraint on decoder network weights, which encourages the dimensions of the latent code as well as the decoder network to be maximally expressive.

In the case of linear encoder, relation, and decoder networks, and assuming that C d is the correlation matrix between rows of φ d , then the regularization term takes the following form: DISPLAYFORM1 2.3.5 BEYOND CLASSIFICATION AND LINEAR OUTPUT LAYERS Thus far we have used few-shot classification as a working example to highlight our proposed method, and in this domain we generate only a single linear output layer.

However, our approach can be applied to any model f θi which maps observations to outputs, e.g. a nonlinear MLP or LSTM, by using a single latent code z to generate the entire parameter vector θ i with an appropriate decoder.

In the general case, z is conditioned on D tr by passing both inputs and labels to the encoder.

Furthermore, the loss L Ti is not restricted to be a classification loss, and can be replaced by any differentiable loss function which can be computed on D tr and D val sets of a task instance T i .

The problem of few-shot adaptation has been approached in the context of fast weights BID16 BID1 , learning-to-learn BID38 BID46 BID17 BID0 , and through meta-learning.

Many recent approaches to meta-learning can be broadly categorized as metric-based methods, which focus on learning similarity metrics for members of the same class (e.g. BID20 BID47 BID42 ; memory-based methods, which exploit memory architectures to store key training examples or directly encode fast adaptation algorithms (e.g. BID37 BID34 ; and optimization-based methods, which search for parameters that are conducive to fast gradientbased adaptation to new tasks (e.g. BID7 .Related work has also explored the use of one neural network to produce (some fraction of) the parameters of another BID13 BID21 , with some approaches focusing on the goal of fast adaptation.

BID30 meta-learn an algorithm to change additive biases across deep networks conditioned on the few-shot training samples.

In contrast, BID11 use an attention kernel to output class conditional mixing of linear output weights for novel categories, starting from a pre-trained deep model.

BID33 learn to output top linear layer parameters from the activations provided by a pre-trained feature embedding, but they do not make use of gradient-based adaptation.

None of the aforementioned approaches to fast adaptation explicitly learn a probability distribution over model parameters, or make use of latent variable generative models to characterize it.

Approaches which use optimization-based meta-learning include MAML BID7 and REPTILE BID31 .

While MAML backpropagates the meta-loss through the "inner loop", REPTILE simplifies the computation by incorporating an L 2 loss which updates the meta-model parameters towards the instance-specific adapted models.

These approaches use the full, high-dimensional set of model parameters within the "inner loop", while BID25 learn a layer-wise subspace in which to use gradient-based adaptation.

However, it is not clear how these methods scale to large expressive models such as residual networks (especially given the uncertainty in the few-shot data regime), since MAML is prone to overfitting BID28 .

Recognizing this issue, BID52 train a deep input representation, or "concept space", and use it as input to an MLP meta-learner, but perform gradient-based adaptation directly in its parameter space, which is still comparatively high-dimensional.

As we will show, performing adaptation in latent space to generate a simple linear layer can lead to superior generalization.

Probabilistic meta-learning approaches such as those of BID2 and BID12 have shown the advantages of learning Gaussian posteriors over model parameters.

Concurrently with our work, and propose probabilistic extensions to MAML that are trained using a variational approximation, using simple posteriors.

However, it is not immediately clear how to extend them to more complex distributions with a more diverse set of tasks.

Other concurrent works have introduced deep parameter generators that can better capture a wider distribution of model parameters, but do not employ gradientbased adaptation.

In contrast, our approach employs both a generative model of parameters, and adaptation in a low-dimensional latent space, aided by a data-dependent initialization.

Finally, recently proposed Neural Processes BID9 bear similarity to our work: they also learn a mapping to and from a latent space that can be used for few-shot function estimation.

However, coming from a Gaussian processes perspective, their work does not perform "inner loop" adaptation and is trained by optimizing a variational objective.

We evaluate the proposed approach on few-shot regression and classification tasks.

This evaluation aims to answer the following key questions: (1) Is LEO capable of modeling a distribution over model parameters when faced with uncertainty?

(2) Can LEO learn from multimodal task distributions and is this reflected in ambiguous problem instances, where multiple distinct solutions are possible?

(3) Is LEO competitive on large-scale few-shot learning benchmarks?

To answer the first two questions we adopt the simple regression task of .

1D regression problems are generated in equal proportions using either a sine wave with random amplitude and phase, or a line with random slope and intercept.

Inputs are sampled randomly, creating a multimodal task distribution.

Crucially, random Gaussian noise with standard deviation 0.3 is added to regression targets.

Coupled with the small number of training samples (5-shot), the task is challenging for 2 main reasons: (1) learning a distribution over models becomes necessary, in order to account for the uncertainty introduced by noisy labels; (2) problem instances may be likely under both modes: in some cases a sine wave may fit the data as well as a line.

Faced with such ambiguity, learning a generative distribution of model parameters should allow several different likely models to be sampled, in a similar way to how generative models such as VAEs can capture different modes of a multimodal data distribution.

We used a 3-layer MLP as the underlying model architecture of f θ , and we produced the entire parameter tensor θ with the LEO generator, conditionally on D tr , the few-shot training inputs concatenated with noisy labels.

For further details, see Appendix A. In FIG1 we show samples from a single model trained on noisy sines and lines, with true regression targets in black and training samples marked with red circles and vertical dashed lines.

Plots (a) and (b) illustrate how LEO captures some of the uncertainty in ambiguous problem instances within each mode, especially in parts of the input space far from any training samples.

Conversely, in parts which contain data, models fit the regression target well.

Interestingly, when both sines and lines could explain the data, as shown in panels (c) and (d), we see that LEO can sample very different models, from both families, reflecting its ability to represent parametric uncertainty appropriately.

In order to answer the final question we scale up our approach to 1-shot and 5-shot classification problems defined using two commonly used ImageNet subsets.

The miniImageNet dataset BID47 is a subset of 100 classes selected randomly from the ILSVRC-12 dataset BID36 with 600 images sampled from each class.

Following the split proposed by BID34 , the dataset is divided into training, validation, and test meta-sets, with 64, 16, and 20 classes respectively.

The tieredImageNet dataset BID35 ) is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy BID4 .

This set of nodes is partitioned into 20, 6, and 8 disjoint sets of training, validation, and testing nodes, and the corresponding classes form the respective meta-sets.

As argued in BID35 , this split near the root of the ImageNet hierarchy results in a more challenging, yet realistic regime with test classes that are less similar to training classes.

Two potential difficulties of using LEO to instantiate parameters with a generator network are:(1) modeling distributions over very high-dimensional parameter spaces; and (2) requiring metalearning (and hence, gradient computation in the inner loop) to be performed with respect to a high-dimensional input space.

We address these issues by pre-training a visual representation of the data and then using the generator to instantiate the parameters for the final layer -a linear softmax classifier operating on this representation.

We train a 28-layer Wide Residual Network (WRN-28-10) BID50 with supervised classification using only data and classes from the training meta-set.

Recent state-of-the-art approaches use the penultimate layer representation BID52 BID33 BID2 BID11 ; however, we choose the intermediate feature representation in layer 21, given that higher layers tend to specialize to the training distribution BID49 .

For details regarding the training, evaluation, and network architectures, see Appendix B.

Following the LEO adaptation procedure (Algorithm 1) we also use fine-tuning 3 by performing a few steps of gradient-based adaptation directly in parameter space using the few-shot set D tr .

This is similar to the adaptation procedure of MAML, or Meta-SGD ) when the learning rates are learned, with the important difference that starting points of fine-tuning are custom generated by LEO for every task instance T i .

Empirically, we find that fine-tuning applies a very small change to the parameters with only a slight improvement in performance on supervised classification tasks.

miniImageNet test accuracy 1-shot 5-shotMatching networks BID47 43.56 ± 0.84% 55.31 ± 0.73% Meta-learner LSTM BID34 43.44 ± 0.77% 60.60 ± 0.71% MAML BID7 48.70 ± 1.84% 63.11 ± 0.92% LLAMA BID12 49.40 ± 1.83% -REPTILE BID31 49.97 ± 0.32% 65.99 ± 0.58% PLATIPUS 50.13 ± 1.86% -

54.24 ± 0.03% 70.86 ± 0.04% SNAIL BID28 55.71 ± 0.99% 68.88 ± 0.92% BID11 56.20 ± 0.86% 73.00 ± 0.64% BID2 56.30 ± 0.40% 73.90 ± 0.30% BID30 57.10 ± 0.70% 70.04 ± 0.63% DEML+Meta-SGD BID52 4 58.49 ± 0.91% 71.28 ± 0.69% TADAM 58.50 ± 0.30% 76.70 ± 0.30% BID33 59 BID35 53.31 ± 0.89% 72.69 ± 0.74% Relation Net (evaluated in BID27 54.48 ± 0.93% 71.32 ± 0.78% Transductive Prop.

Nets BID27 57.41 ± 0.94% 71.55 ± 0.74%

62.95 ± 0.03% 79.34 ± 0.06% LEO (ours)66.33 ± 0.05% 81.44 ± 0.09% Table 1 : Test accuracies on miniImageNet and tieredImageNet.

For each dataset, the first set of results use convolutional networks, while the second use much deeper residual networks, predominantly in conjuction with pre-training.

The classification accuracies for LEO and other baselines are shown in Table 1 .

LEO sets the new state-of-the-art performance on the 1-shot and 5-shot tasks for both miniImageNet and tieredImageNet datasets.

We also evaluated LEO on the "multi-view" feature representation used by BID33 with miniImageNet, which involves significant data augmentation compared to the approaches in Table 1 .

LEO is state-of-the-art using these features as well, with 63.97 ± 0.20% and 79.49 ± 0.70% test accuracies on the 1-shot and 5-shot tasks respectively.

To assess the effects of different components, we also performed an ablation study, with detailed results in Table 2 .

To ensure a fair comparison, all approaches begin with the same pre-trained Table 2 : Ablation study and comparison to Meta-SGD.

Unless otherwise specified, LEO stands for using the stochastic generator for latent embedding optimization followed by fine-tuning.features (Section 4.2.2).

The Meta-SGD case performs gradient-based adaption directly in the parameter space in the same way as MAML, but also meta-learns the inner loop learning rate (as we do for LEO).

The main approach, labeled as LEO in the table, uses a stochastic parameter generator for several steps of latent embedding optimization, followed by fine-tuning steps in parameter space (see subsection 4.2.3).

All versions of LEO are at or above the previous state-of-the-art on all tasks.

The largest difference in performance is between Meta-SGD and the other cases (all of which exploit a latent representation of model parameters), indicating that the low-dimensional bottleneck is critical for this application.

The "conditional generator only" case (without adaptation in latent space) yields a poorer result than LEO, and even adding fine-tuning in parameter space does not recover performance; this illustrates the efficacy of the latent adaptation procedure.

The importance of the data-dependent encoding is highlighted by the "random prior" case, in which the encoding process is replaced by the prior p(z n ), and performance decreases.

We also find that incorporating stochasticity can be important for miniImageNet, but not for tieredImageNet, which we hypothesize is because the latter is much larger.

Finally, the fine-tuning steps only yield a statistically significant improvement on the 5-shot tieredImageNet task.

Thus, both the data-conditional encoding and latent space adaptation are critical to the performance of LEO.(b) (a) (c) Figure 4 : t-SNE plot of latent space codes before and after adaptation: (a) Initial codes z n (blue) and adapted codes z n (orange); (b) Same as (a) but colored by class; (c) Same as (a) but highlighting codes z n for validation class "Jellyfish" (left) and corresponding adapted codes z n (right).

To qualitatively characterize the learnt embedding space, we plot codes produced by the relational encoder before and after the LEO procedure, using a 5-way 1-shot model and 1000 task instances from the validation meta-set of miniImageNet.

Figure 4 shows a t-SNE projection of class conditional encoder outputs z n as well as their respective final adapted versions z n .

If the effect of LEO were minimal, we would expect latent codes to have roughly the same structure before and after adaptation.

In contrast, Figure 4 (a) clearly shows that latent codes change substantially during LEO, since encoder output codes form a large cluster (blue) to which adapted codes (orange) do not belong.

Figure 4(b) shows the same t-SNE embedding as (a) colored by class label.

Note that encoder DISPLAYFORM0 Figure 5: Curvature and coverage metrics for a number of different models, computed over 1000 problem instances drawn uniformly from the test meta-set.

For all plots, the whiskers span from the 5 th to 95 th percentile of the observed quantities.outputs, on the left side of plot (b), have a lower degree of class conditional separation compared to z n clusters on the right, suggesting that qualitatively different structure is introduced by the LEO procedure.

We further illustrate this point by highlighting latent codes for the "Jellyfish" validation class in Figure 4 (c), which are substantially different before and after adaptation.

The additional structure of adapted codes z n may explain LEO's superior performance over approaches predicting parameters directly from inputs, since the decoder may not be able to produce sufficiently different weights for different classes given very similar latent codes, especially when the decoder is linear.

Conversely, LEO can reduce the uncertainty of the encoder mapping, which is inherent in the few-shot regime, by adapting latent codes with a generic, gradient-based procedure.

We hypothesize that by performing the inner-loop optimization in a lower-dimensional latent space, the adapted solutions do not need to be close together in parameter space, as each latent step can cover a larger region of parameter space and effect a greater change on the underlying function.

To support this intuition, we compute a number of curvature and coverage measures, shown in Figure 5 .The curvature provides a measure of the sensitivity of a function with respect to some space.

If adapting in latent space allows as much control over the function as in parameter space, one would expect similar curvatures.

However, as demonstrated in Figure 5 (a), the curvature for LEO in z space (the absolute eigenvalues of the Hessian of the loss) is 2 orders of magnitude higher than in θ, indicating that a fixed step in z will change the function more drastically than taking the same step directly in θ.

This is also observed in the "gen+ft" case, where the latent embedding is still used, but adaptation is performed directly in θ space.

This suggests that the latent bottleneck is responsible for this effect.

Figure 5 (b) shows that this is due to the expansion of space caused by the decoder.

In this case the decoder is linear, and the singular values describe how much a vector projected through this decoder grows along different directions, with a value of one preserving volume.

We observe that the decoder is expanding the space by at least one order of magnitude.

Finally, Figure 5 (c) demonstrates this effect along the specific gradient directions used in the inner loop adaptation: the small gradient steps in z taken by LEO induce much larger steps in θ space, larger than the gradient steps taken by Meta-SGD in θ space directly.

Thus, the results support the intuition that LEO is able to 'transport' models further during adaptation by performing meta-learning in the latent space.

We have introduced Latent Embedding Optimization (LEO), a meta-learning technique which uses a parameter generative model to capture the diverse range of parameters useful for a distribution over tasks, and demonstrated a new state-of-the-art result on the challenging 5-way 1-and 5-shot miniImageNet and tieredImageNet classification problems.

LEO achieves this by learning a lowdimensional data-dependent latent embedding, and performing gradient-based adaptation in this space, which means that it allows for a task-specific parameter initialization and can perform adaptation more effectively.

Future work could focus on replacing the pre-trained feature extractor with one learned jointly through meta-learning, or using LEO for tasks in reinforcement learning or with sequential data.

We used the experimental setup of for 1D 5-shot noisy regression tasks.

Inputs were sampled uniformly from [−5, 5] .

A multimodal task distribution was used.

Half of the problem instances were sinusoids with amplitude and phase sampled uniformly from [0.1, 5] and [0, π] respectively.

The other half were lines with slope and intercept sampled uniformly from the interval [−3, 3] .

Gaussian noise with standard deviation 0.3 was added to regression targets.

A.2 LEO NETWORK ARCHITECTURE As TAB3 shows, the underlying model f θ (for which parameters θ were generated) was a 3-layer MLP with 40 units in all hidden layers and rectifier nonlinearities.

A single code z was used to generate θ with the decoder, conditioned on concatenated inputs and regression targets from D tr which were passed as inputs to the encoder.

Sampling of latent codes and parameters was used both during training and evaluation.

The encoder was a 3-layer MLP with 32 units per layer and rectifier nonlinearities; the bottleneck embedding space size was: n z = 16.

The relation network and decoder were both 3-layer MLPs with 32 units per layer.

For simplicity we did not use biases in any layer of the encoder, decoder nor the relation network.

Note that the last dimension of the relation network and decoder outputs are two times larger than n z and dim(θ) respectively, as they are used to parameterize both the means and variances of the corresponding Gaussian distributions.

We used the standard 5-way 1-shot and 5-shot classification setups, where each task instance involves classifying images from 5 different categories sampled randomly from one of the meta-sets, and D tr contains 1 or 5 training examples respectively.

D val contains 15 samples during metatraining, as decribed in BID7 , and all the remaining examples during validation and testing, following BID33 .We did not employ any data augmentation or feature averaging during meta-learning, or any other data apart from the corresponding training and validation meta-sets.

The only exception is the special case of "multi-view" embedding results, where features were averaged over representations of 4 corner and central crops and their horizontal mirrored versions, which we provide for full comparison with BID33 .

Apart from the differences described here, the feature training pipeline closely followed that of BID33 .B.2 FEATURE PRE-TRAINING As described in Section 4.2.2, we trained dataset specific feature embeddings before meta-learning, in a similar fashion to BID33 and BID2 .

A Wide Residual Network WRN-28-10 (Zagoruyko & Komodakis, 2016b) with 3 steps of dimensionality reduction was used to clas-sify images of 80 × 80 pixels from only the meta-training set into the corresponding training classes (64 in case of miniImageNet and 351 for tieredImageNet).

We used dropout (p keep = 0.5) inside residual blocks, as described in BID51 , which is turned off during evaluation and for dataset export.

An L2 regularization term of 5e −4 was used, 0.9 Nesterov momentum, and SGD with a learning rate schedule.

The initial learning rate was 0.1 and it was multiplied with 0.2 at the steps given in Table 4 .

Mini-batches were of size of 1024.

Data augmentation for pre-training was similar to the inception pipeline (Szegedy et al.) , with color distortions and image deformations and scaling in training mode.

For 64-way evaluation accuracy and dataset export we used only the center crop (with a ratio of 80 92 : about 85.95% of the image) which was then resized to 80 × 80 and passed to the network.

Step FORMULA0 Step FORMULA2 Step FORMULA10 Step FORMULA12 Step FORMULA13 Table 4 : Learning rate annealing schedules used to train feature extractors for miniImageNet and tieredImageNet.

Activations in layer 21, with average pooling over spatial dimensions, were precomputed and saved as feature embeddings with n x = dim(x) = 640, which substantially simplified the meta-learning process.

We used the same network architecture of parameter generator for all datasets and tasks.

The encoder and decoder networks were linear with the bottleneck embedding space of size n z = 64.The relation network was a 3-layer fully connected network with 128 units per layer and rectifier nonlinearities.

For simplicity we did not use biases in any layer of the encoder, decoder nor the relation network.

Table 5 summarizes this information.

Note that the last dimension of the relation network and decoder outputs are two times larger than n z and dim(x) respectively, as they are used to parameterize both the means and variances of the corresponding Gaussian distributions.

The "Meta-SGD (our features)" baseline used the same one-layer softmax classifier as base model.

Table 5 : Architecture details for 5-way 1-shot miniImageNet and tieredImageNet.

The shapes correspond to the meta-training phase.

We used a meta-batch of 12 task instances in parallel.

We used a parallel implementation similar to that of BID7 , where the "inner loop" is performed in parallel on a batch 12 problem instances for every meta-update.

Using a relation network in the encoder has negligible computational cost given that k 2 is small in typical k-shot learning domains, and the relation network is only used once per problem instance, to get the initial model parameters before adaptation.

Within the LEO "inner loop" we perform 5 steps of adaptation in latent space, followed by 5 steps of fine-tuning in parameter space.

The learning rates for these spaces were meta-learned in a similar fashion to Meta-SGD , after being initialized to 1 and v Function g φr from Eq. (3) is applied 25 times (once for each pair of inputs in D tr ) and then averaged into 5 class-specific means and variances.

0.001 for the latent and parameter spaces respectively.

We applied dropout independently on the feature embedding in every step, with the probability of not being dropped out p keep chosen (together with other hyperparameters) using random search based on the validation meta-set accuracy.

Parameters of the encoder, relation, and decoder networks as well as per-parameter learning rates in latent and parameter spaces were optimized jointly using Adam BID19 to minimize the meta-learning objective (Eq. 6) over problem instances from the training meta-set, iterating for up to 100 000 steps, with early stopping using validation accuracy.

Meta-learning objectives can lead to difficult optimization processes in practice, specifically when coupled with stochastic sampling in latent and parameters spaces.

For ease of experimentation we clip the meta-gradient, as well as its norm, at an absolute value of 0.1.

Please note this was only done for the encoder, relation, decoder networks and learning rates, not the inner loop latent space adaptation gradients.

Table 6 : Values of hyperparameters chosen to maximize meta-validation accuracy during random search.

To find the best values of hyperparameters, we performed a random grid search and we choose the set which lead to highest validation meta-set accuracy.

The reported performance of our models is an average (± a standard deviation) over 5 independent runs (using different random seeds) with the best hyperparameters kept fixed.

The result of a single run is an average accuracy over 50000 task instances.

After choosing hyperparameters (given in Table 6 ) we used both meta-training and meta-validation sets for training, in line with recent state-of-the-art approaches, e.g. BID33 .The evaluation of each of the LEO baselines follow the same procedure; in particular, we perform a separate random search for each of them.

B.6 TRAINING TIME Training of LEO took 1-2 hours for miniImageNet and around 5 hours for tieredImageNet on a multi-core CPU (for each of the 5 independent runs).

Our approach allows for caching the feature embeddings before training LEO, which leads to a very efficient meta-learning process.

Training of the image extractor was more compute-intensive, taking 5 hours for miniImageNet and around a day for tieredImageNet using 32 GPUs.

In summary, there are three stages in our approach to meta-training:1.

In the first stage we use 64-way classification to pre-train the feature embedding only on the meta-training set, hence without the meta-validation classes.

2.

In the second stage we train LEO on the meta-training set with early stopping on metavalidation, and we choose the best hyperparameters using random grid search.3.

In the third stage we train LEO again from scratch 5 times using the embedding trained in stage 1 and the chosen set of hyperparameters from stage 2.

However, in this stage we metalearn on embeddings from both meta-train and meta-validation sets, with early-stopping on meta-validation.

While it may not be intuitive to use early stopping on meta-validation in stage 3, it is still a proxy for good generalization since it favors models with high performance on classes excluded during feature embedding pre-training.

The procedure for evaluation is similar to meta-training, except that we disable stochasticity and dropout.

Naturally, instead of computing the meta-training loss, the parameters (adapted based on L tr Ti ) are only used for inference on that particular task.

That is:1.

A problem instance is drawn from the evaluation meta-set.2.

The few-shot samples are encoded to latent space, then decoded; the means are used to initialize the parameters of the inference model.3.

A few steps of adaptation are performed in latent space, followed (optionally) by a few steps of adaptation in parameter space.4.

The resulting parameters are used as the final adapted model for that particular problem instance.

@highlight

Latent Embedding Optimization (LEO) is a novel gradient-based meta-learner with state-of-the-art performance on the challenging 5-way 1-shot and 5-shot miniImageNet and tieredImageNet classification tasks.

@highlight

A new meta-learning framework that learns data-dependent latent space, performs fast adaptation in the latent space, is effective for few-shot learning, has task-dependent initialization for adaptation, and works well for multimodal task distribution.

@highlight

This paper proposes a latent embedding optimization method for meta-learning, and claims the contribution is to decouple optimization-based meta-learning techniques from high-dimensional space of model parameters.