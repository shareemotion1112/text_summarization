Few-Shot Learning (learning with limited labeled data) aims to overcome the limitations of traditional machine learning approaches which require thousands of labeled examples to train an effective model.

Considered as a hallmark of human intelligence, the community has recently witnessed several contributions on this topic, in particular through meta-learning, where a model learns how to learn an effective model for few-shot learning.

The main idea is to acquire prior knowledge from a set of training tasks, which is then used to perform (few-shot) test tasks.

Most existing work assumes that both training and test tasks are drawn from the same distribution, and a large amount of labeled data is available in the training tasks.

This is a very strong assumption which restricts the usage of meta-learning strategies in the real world where ample training tasks following the same distribution as test tasks may not be available.

In this paper, we propose a novel meta-learning paradigm wherein a few-shot learning model is learnt, which simultaneously overcomes domain shift between the train and test tasks via adversarial domain adaptation.

We demonstrate the efficacy the proposed method through extensive experiments.

Few-Shot Learning aims to learn a prediction model from very limited amount of labelled data BID13 .

Specifically, given a K−shot, N −class data for a classification task, the aim is to learn a multi-class classification model for N − classes, with K−labeled training examples for each class.

Here K is usually a small number (e.g. 1, or 5).

Considered as one of the hallmarks of human intelligence BID12 , this topic has received considerable interest in recent years BID13 BID11 BID33 BID2 .

Modern techniques solve this problem through meta-learning, using an episodic learning paradigm.

The main idea is to use a labeled training dataset to effectively acquire prior knowledge, such that this knowledge can be transferred to novel tasks where few-shot learning is to be performed.

Different from traditional transfer learning BID21 BID35 , here few-shot tasks are simulated using the labeled training data through episodes, in order to acquire prior knowledge that is specifically tailored for performing few-shot tasks.

For example, given a set of labeled training data with a finite label space Y train , the epsiodic paradigm is used to acquire prior knowledge which is stored in a model.

Each episode is generated i.i.d from an unknown task distribution τ train .

This model is then used to do a novel few shot classification task which is drawn from an unknown task distribution τ test .

The test task comprises small amount of labeled data with a finite label space Y test , and the sets Y train and Y test are (possibly) mutually exclusive.

Using this labeled data, and acquired prior knowledge, the goal is to predict the labels of all unlabeled instances in the test task.

A very restrictive assumption of existing meta-learning approaches for few-shot learning is that train and test tasks are drawn from the same distribution, i.e., τ train = τ test .

In this scenario, the metalearner's objective is to minimize its expected loss over the tasks drawn from the task distribution τ train .

This assumption prohibits the use of meta-learning strategies for real-world applications, where training tasks with ample labeled data, and drawn from the same distribution as the test tasks are very unlikely to be available.

Consider the case of a researcher or practitioner who wishes to train a prediction model for their own dataset where labeled data is very limited.

It is unreasonable to assume that they would have a large corpus of labeled data for a set of related tasks in the same domain.

Without this, they are not able to train effective few-shot models for their task.

A more desirable option is to use the training tasks where ample training data is available, and adapt the model to be effective on test tasks in a different domain.

A possible way to tackle this problem could be through the use of domain adaptation techniques BID4 BID7 that address the domain shift between the training and test data.

However, all of these approaches address the single-task scenario, i.e., Y train = Y test , where the training data and test data are sampled from the same task but there is a domain shift at a data-level.

This is in contrast to the meta-learning setting where the training data contains multiple tasks and the goal is to learn new tasks from test data, i.e., domain shift exists at a task-level and Y train ∩ Y test = ∅. As a result, these domain adaptation approaches cannot be directly applied.

We show an overview of different problem settings in TAB0 .

DISPLAYFORM0 In order to solve the few-shot learning problem under a domain shift we propose a novel meta-learning paradigm: Meta-Learning with Domain Adaptation (MLDA).

Existing meta-learning approaches for few-shot learning use only the given training data to learn a model, and as a result they do not account for any domain shift between the training tasks and the few-shot test tasks.

In contrast, we assume that the model has access to the unlabeled instances in the domain of the few-shot test tasks prior to the training procedure, and utilize these instances for incorporating the domain-shift information.

We train the model under the episodic-learning paradigm, but in each episode we aim to train a model which achieves two goals: first the model should be good at few-shot learning, and second the model should be unaffected by a possible domain shift.

The first goal is achieved by updating the model based on the few-shot learning loss suffered by the model for a given episode.

The second goal is achieved by an adversarial domain adaptation approach, where a mapping is used which styles the training task to resemble the test task.

In this way, the trained model can perform few-shot predictions on the test tasks, and achieve what we term task-level domain adaptation.

The episodic update is done via Prototypical Networks BID28 (as a specific instantiation, though other approaches can be applied), where on a simulated few-shot task (a small support set behaves as training, and a query set behaves as test data), an embedding is produced for both support and query instances.

The mean of support embedding of each class is the prototype, and query instances are labeled based on their distance to these prototypes.

Based on the loss on these query instances, the embedding function is updated.

For achieving invariance to domain shift, we follow the principle of adversarial domain adaptation, but we differ from the traditional approaches in that we are performing task-level domain adaptation, whereas they performed data-level domain adaptation.

The early approaches to adversarial domain adaptation aimed at obtaining a feature embedding that was invariant to both the training domain and the test domain, as well as learning a prediction model in the training domain BID4 .

However, these approaches possibly learnt a highly unconstrained feature embedding (particularly when the embedding was very high dimensional), and were outperformed by GAN-based approaches (often used for image translation) BID29 BID38 BID7 .

As a result we use a mapping function to style the training tasks to resemble test tasks, and optimize it using a GAN loss.

The overall framework delivers a model that uses training tasks from one distribution to meta-learn a few-shot model for a task from another distribution.

We perform extensive experiments to show the efficacy of the proposed method.2 RELATED WORK 2.1 META-LEARNING FOR FEW-SHOT LEARNING Few-Shot Learning refers to learning a prediction model from small amount of labeled data BID1 BID12 .

Early approaches used a Bayesian model BID1 , or hand-designed priors BID13 .

More recently, meta-learning approaches have become extremely successful for addressing few-shot learning BID33 BID2 .

Instead of training a model directly on the few-shot data, meta-learning approaches use a corpus of labeled data, and simulate few-shot tasks on them to learn how to do few-shot learning.

Some approaches follow the non-parametric principle, and develop a differentiable K−nearest neighbour solution BID33 BID27 BID28 .

The main concept is to learn an embedding space that is tailored for performing effective K-nearest neighbour.

BID20 extend these approaches with metric scaling to condition the embedding based on the given task.

Another category of meta-learning aims to learn how to quickly adapt a model in few gradient steps for a few-shot learning task BID2 BID23 .

These optimization based approaches aim to learn an initialization from a set of training tasks, which can be quickly adapted (e.g. one-step gradient update) when presented with a novel few-shot task.

Some other approaches consider using a "memory"-based approach BID26 BID19 .

There have also been approaches that try to enhance meta-learning performance through use of additional information.

For example, BID24 use unlabeled data to develop semi supervised few-shot learning.

BID37 use external data to generate concepts, and performs meta-learning in the concept space.

However, all of these approaches assume that the training tasks and testing tasks are drawn from the same distribution (τ train = τ test ).

If there is a task-level domain shift, the above approaches will fail to perform few-shot learning on novel test tasks.

Our approach of meta-learning with domain adaptation overcomes this domain shift, to perform few-shot learning on tasks in a different domain.

Domain adaptation has been studied extensively in recent years, particularly for computer vision applications BID25 .

The idea is to exploit labeled data in one domain (called source domain) to perform a prediction task in another domain (called the target domain), which does not have any labels (unsupervised domain adaptation).

Most approaches employed two objectives: one to learn a prediction model in the source domain, and second to find an embedding space between the two domains that achieves domain invariance, thus making the model trained on the source domain applicable to the task in the target domain.

In the era of deep learning, some early approaches aimed to align feature distribution in some embedding space using statistical measures (e.g. Maximum Mean Discrepancy) BID30 BID17 .

This was followed by several successful efforts for domain adaptation using an adversarial loss BID6 .

BID3 ; BID4 aimed to learn a feature embedding such that a domain classifier would not be able to distinguish whether the instance was drawn from the source or target domain.

Consequent efforts tried to learn an embedding on the source data, from which an instance in the target domain could be reconstructed BID5 .

BID31 proposed to train a model in the source domain, and using a GAN loss try to embed the target domain to the same feature distribution (using a GAN loss) as the (now fixed) source domain.

Another line of work using GAN-loss is for image-toimage translation, where images in one domain are mapped to appear like images in another domain BID29 .

Most of these approaches have demonstrated application to domain adaptation tasks as well.

Another recently introduced concept is cycle consistency which first maps an instance from the source to target, and then maps this synthetic instance back to the source to get back to original instance BID38 BID10 BID34 , and this concept has been extended for domain adaptation as well BID7 .

All of these approaches aim to solve the same task in both domains (i.e., the label space is the same in both domains).

They perform domain adaptation at the data-level (and not the task level).

This means that they cannot solve a new task with a different label space.

In contrast our approach performs a task-level domain adaptation, and can solve new tasks.

There have been some efforts at the intersection of few-shot (and meta-learning) and domain adaptation.

BID18 consider supervised domain adaptation, which is similar to unsupervised domain adaptation setting, except that few labeled instances in the target domain are available.

Like the previous approaches, it can not be used for a novel task with a different label space.

BID9 's problem setting resembles traditional unsupervised domain adaptation, except that the model training is done using a meta-learning principle.

use meta-learning to address domain generalization where a single trained model for a given task, can be applied to any new domain with a different data distribution.

They too consider solving the same task in a new domain, and do not consider the few-shot setting.

A closely related work is Domain Adaptive Meta Learning BID36 , but their problem setting is different (which is more suitable for the problem they address: imitation learning) from what we address in this paper.

They consider the scenario where a task has training data drawn from one domain and test data drawn from another domain (independent of whether it has been drawn from τ train or τ test ).

Thus, they do not violate τ train = τ test .

In their simulated task, ample labeled training data is available for both the source and target domains.

In contrast, we consider the scenario where the training tasks and test tasks are drawn from different distributions, and we have very limited labeled data for test tasks (tasks in the target domain).

Formally, let X be an input space and Y be a discrete label space.

Let D be a distribution over X × Y. During meta-training, the meta-learner has access to a large labeled dataset S train that typically contains thousands of instances for a large number of classes C. At the i-th iteration of episodic learning paradigm, the meta-learner samples a K−shot N -class classification task T i from S train , which contains a small "training" (assumed to have labels for all instances for this task) set S support i(with K examples from each class) and a "test" (assumed to not have labels of any instances for this task) set S .

The meta-learner then backpropagates the gradient of the loss (x,y)∈S query L(p(y|x, S support ), y) for updating the model.

In the meta-testing phase, the resulting meta-learner is used to solve the novel K−shot N -class classification tasks, which are assumed to be generated i.i.d.

from an unknown task distribution τ test .

The labeled training set and unlabeled test examples are given to the classification algorithm and the algorithm outputs class probabilities.

Existing meta-learning approaches assume that both training and testing tasks are drawn from the same distribution, i.e., τ train = τ test .

However, this may not be the case in several real-world scenarios (i.e., τ train = τ test ).

Consider the case of a researcher who wants to do few-shot classification on a newly collected image recognition dataset (task drawn from τ test ).

This researcher must now find a large amount of labeled data from which tasks can be drawn from the same task distribution (τ test ), failing which the researcher does not have a clear approach to acquire the relevant prior knowledge.

The alternative is for the researcher to find tasks drawn from a different distribution, where ample labeled data is available, and perform task-level domain adaptation in order to learn a few-shot model suitable for their own task.

Thus, we make a distinction between the task drawn from τ train and τ test , as (D respectively), and may also have a mutually exclusive discrete label space (e.g. Y train ∩ Y test = ∅).

Our overall goal is to learn a meta-learner that can utilize tasks drawn from τ train to acquire a good prior for few-shot learning, and overcome the task-level domain-shift in order to learn unobserved few-shot tasks drawn from τ test .

The general setting can be seen in Figure 1 .

Next, we briefly describe our proposed few-shot learning approach under task-level domain shift.

Data from which Test Tasks are drawn from Figure 1: Problem Setting for Meta-Learning with Domain Adaptation.

Tasks are drawn from τtrain, on which meta learning is performed, such that the learner can do effective meta-testing for tasks drawn from a different distribution τtest.

The images are adapted from the Omniglot dataset BID12 , where the left block has some original instances of hand-written characters in the original domain, and in the right block, we have a set of different omniglot characters (or classes) and the data is also in a different domain.

Here, we give the overview of our proposed learning paradigm: Meta Learning with Domain Adaptation (MLDA).

We have two objectives that need to be optimized simultaneously.

First, we want to learn a feature extractor that can learn discriminative features which can be used for fewshot learning on novel tasks.

Second, we want these features to be invariant to the train task distribution and test task distribution, i.e., for a task DISPLAYFORM0 , we want to adapt it to resemble a task drawn from (D m test , D test ).

Specifically, in the meta-learning phase, we consider a feature extractor F : X train → R d which takes an input instance x ∈ X train and returns a d−dimensional embedding.

This feature extractor in turn is a composition function F(x) =F (G(x) ), where G : DISPLAYFORM1 The feature extractor F is trained to learn a representation suitable for few-shot learning (by optimizing objective L f s ).

G aims to achieve task-level domain invariance by translating instances from domain X train to instances in domain X test .

G is trained using an adversarial loss, inspired by recent success of GAN-based BID6 domain adapation methods BID31 BID38 BID7 .

G (along with the corresponding discriminator D) is trained to achieve domain adaptation (by optimizing objective L da ).

We also use a mapping G : X test → X train to obtain cyclic consistency, wherein we try to translate generated instance G(x) to produce the original instance x. The overall objective function is given by: DISPLAYFORM2 Note that L f s is optimized using only labeled training data of tasks drawn from τ train and L da is optimized using unlabeled data of tasks drawn from both τ train and τ test .

The overall framework can be seen in FIG3 .

Next, we will describe motivation and technical details of these components.

The proposed method for few-shot learning under task-level domain shift using adversarial domain adaptation.

A task sampled from τtrain in every episode.

This task is used to update the parameters with the aim of achieving 2 goals: 1) It follows a Prototypical Networks learning scheme to acquire few-shot learning ability, and 2) It styles the task to appear indistinguishable from a task drawn from τtest.

Task-level domain invariance is achieved through the usage of a GAN loss and a cycle-consistency loss (L da = LGAN + L cycle ).

There have been several approaches for few-shot learning via meta-learning in literature BID33 BID2 BID28 .

In principle, our proposed paradigm is agnostic to any of these approaches.

In our work, we follow a recent state of the art approach: Prototypical Networks BID28 to instantiate our framework for meta learning with domain adaptation.

For a given task DISPLAYFORM0 Networks use a feature extractor F : X train → R d to compute a d−dimensional embedding for each instance.

Using this feature extractor, the mean vector embedding is computed for each class c n for n = 1, . . .

, N , which are the prototypes of each class: DISPLAYFORM1 For a given query instance x, Prototypical Network will produce a probability distribution over the classes using: DISPLAYFORM2 where dist : DISPLAYFORM3 is a function measuring the distance between the embeddings of a query instance and a class prototype.

The few-shot loss L f s is the negative log-probability: DISPLAYFORM4 This loss is evaluated on the query set S query i, and backpropagated to update the parameters of feature extractor F. In this setup, F does not account for a domain shift between τ train and τ test .

Consequently, we use F(x) =F(G(x)), where G will help incorporate the domain shift information.

Here, we describe how to perform task-level domain adaptation and learn the mapping parameters G.

We use the GAN loss BID6 to learn the mapping G : X train → X test , and its corresponding discriminator D. The objective function is denoted as: DISPLAYFORM0 Here G tries to generate instances that appear to be similar to the instances in domain X test , and D tries to distinguish between translated instances G(x train )and the real samples x test .

This objective is minimized under the parameters of G and maximized under the parameters of D. This effectively leads to translating tasks drawn from τ train to be translated such that they are indistinguishable from tasks drawn from τ test .Despite the ability of adversarial networks to produce an output indistinguishable from the test domain X test , with a large capacity, it is not inconceivable for the network to map the same set of input images in the train domain X train to any random permutation of images in the test domain (a form of overfitting).

This is because the objective is highly unconstrained.

As a result, we use a cycle-consistency loss BID38 BID7 , which uses a new mapping G : X test → X train which will take as input the translated instance G(x) and try to invert this function to get back the original instance, i.e., x → G(x) → G (G(x)) ≈ x. Using an L1-loss the task-cycle-consistency loss is given as: DISPLAYFORM1 Combining the objectives from equation 5 and equation 6, we get the domain adaptation objective as L da = L GAN + Lcycle.

The objective in equation 1 is the basic objective of our proposed framework.

We also consider two advanced variants that help improve the performance of the domain adaptation.

First, we consider an identity loss where we encourage G to behave like an identity mapping when it receives an instance from X test , thereby behaving as an identity function for a test task.

We also introduce a reverse direction mapping to map instances from test tasks X test → X train , and a corresponding cycle loss to reconstruct back the instance in X test .

All these objectives get tied together to deliver an appropriate task-level domain adaptation for a few-shot learning task.

These improvements follow from state of the art image-to-image translation techniques BID29 BID38 .

The setting we follow is: we have meta-training data in the original domain, unlabeled data in the target domain which is used for domain adaptation, and the meta-test data, from which test tasks will be drawn.

There is no overlap between the data used for domain adaptation, and the meta-test data.

Being a new problem setting, there have not been any approaches in literature directly addressing this problem.

In order to be comparable, we adapt some of the techniques in Meta-Learning and Domain Adaptation, to make them suitable for our setting.

Specifically, we consider 3 state of the art domain adaptation baselines RevGrad BID4 , ADDA BID31 , and CyCADA BID7 .

These baselines are trained on the meta-train data to learn a multi-class classifier and the unlabeled target domain data is used for domain adaptation.

During meta-testing, these models are used as feature extractors, and K-NN is performed for prediction.

We consider two meta-learning baselines MAML BID2 and Prototypical Networks BID28 , which are trained on meta-train data; however these approaches do not consider the domain-shift issues.

We construct a baseline that combines meta-learning with task-level domain shift.

It is a combination of Prototypical Networks BID28 with Gradient Reversal BID4 , which we call Meta-RevGrad.

Meta-RevGrad jointly optimizes PN-loss and a domain-confusion loss at the feature level where the embedded features of training tasks are made to appear like embedded features of test tasks resulting in the objective: λL f s + (1 − λ)L RevGrad .

Readers are refered to BID4 for greater detail on L RevGrad .

Here λ ∈ (0, 1) is a trade-off parameter between few-shot performance and domain adaptation.

We try several values of λ = 0.9, 0.8, 0.7, 0.6, 0.5 and report the best result.

For our proposed method for Meta Learning with Domain Adaptation: MLDA, we consider three variants: the basic version MLDA based on equation 1; MLDA+idt, which considers the previous objective and an identity loss (see Section 3.3.2); and MLDA+idt+revMap which adds an additional component of (reverse) mapping testing tasks to train tasks (see Section 3.3.2).Most of our code was implemented in PyTorch BID22 ) (for both baselines and proposed method).

We follow the same model size and parameter setting for our models as the ones used in prior work for Prototypical Networks BID28 and CycleGAN BID38 .

Jointly optimizing the objective in equation 1 can be very noisy (oscillating) and slow.

To ease the implementation, we follow a two-step procedure for the optimization.

We first optimize the objective with respect to all parameters exceptF. Then, all of these parameters are frozen, andF is learned.

Another issue with the GAN-based training is that the task generator lacks randomness, and always maps the same input task to the same output task (which can limit the meta-learning efficacy if the test-task domain is very diverse).

To address this, during the GAN training we store intermediate models (e.g. a model saved after every epoch) and generate tasks using each of these models.

This is similar to Snapshot Ensembles BID8 , where multiple models under one training cycle to increase robustness.

We provide greater detail on the implementations in the appendix.

We use Omniglot dataset (popularly used for benchmarking few-shot classification).

The dataset comprises over 1,600 hand written characters, with 20 instances each.

The dataset was further expanded by applying rotations.

Inspired by a popular domain adaptation benchmark: MNIST to MNIST-M BID4 , we design a new benchmark, suited for the few-shot learning under domain-shift: Omniglot to Omniglot-M. Omniglot-M is constructed in the same manner as MNIST-M, i.e., by randomly blending different Omniglot characters with different color background from BSDS500 BID0 .

We follow the same meta-train and meta-test split as previous approaches, (but meta-train and meta-test are in different domains).

Unlabeled data from validation split (mutually exclusive with meta-test) is used for domain adaptation.

We consider Omniglot to Omniglot-M, and Omniglot-M to Omniglot.

We evaluate 1-shot, 5-class and 5-shot, 5-class tasks.

The results can be seen in TAB1 .

We can see that the basic domain adaptation and meta-learning approaches are not able to get a very good performance, as domain adaptation approaches are not suitable for few-shot learning, and meta-learning methods do not account for domain-shift.

Meta-RevGrad is able to occasionally offer some improvement over the basic techniques, but is outperformed by our proposed MLDA.

In general, MLDA can outperform all the baselines by a big margin.

This can be observed in the case of both Omniglot → Omniglot-M and Omniglot-M → Omniglot.

Similar performance trends are observed for both 1-shot and 5-shot tasks.

Within the variants of MLDA, we see that identity loss, and the reverse mappings are able to offer substantial boost to the overall performance, indicating better quality task-level domain adaptation.

Omniglot → Omniglot-M Omniglot-M → Omniglot Method 1-shot, 5-way 5-shot, 5-way 1-shot, 5-way 5-shot, 5-way RevGrad BID4 26.68% 29.15% 69.89% 85.29% ADDA BID31 27.18% 34.45% 69.10% 86.15% CyCADA BID7 28.97% 40.30% 83.08% 95.18% MAML BID2 26.22% 30.46% 74.14% 83.41% PN BID28 27

We conducted similar experiments using Office-Home Dataset BID32 , in particular data from two domains: Clipart and Product.

There are a total of 65 classes, and we randomly split them into 3 sets, labelled data for meta-train (25 classes), unlabeled data for domain adaptation (20 classes), and the meta-test data (20 classes).

All images were resized to 84x84x3, and all models were trained from scratch (pretrained models were not used).

We consider Clipart to Product, and Product to Clipart.

We evaluate 1-shot, 5-class and 5-shot, 5-class tasks.

The results can be seen in TAB3 .

The observations here are similar in trend to those observed for the character recognition experiments.

The basic meta-learning approaches are quite poor (even though better than random).

Meta-RevGrad can offer some improvement, and MLDA gives an even better performance.

The performance trend is fairly consistent for both 1-shot and 5-shot tasks.

Clipart → Product Product → Clipart Method 1-shot, 5-way 5-shot, 5-way 1-shot, 5-way 5-shot, 5-way RevGrad BID4 25.42% 43.11% 27.05% 36.69% ADDA BID31 31.99% 42.57% 27.63% 31.17% CyCADA BID7 30.48% 51.08% 29.20% 44.04% MAML BID2 35.75% 51.12% 32.15% 44.14% PN BID28 36

In this paper we investigated a novel problem setting: Meta-Learning for few-shot learning under task-level domain shift.

Existing meta learning paradigm for few-shot learning was designed under the assumption that both training tasks and test tasks were drawn from the same distribution.

This may not be the case for real world applications, where researchers may not find ample labeled data to simulate training tasks to be drawn from the same distribution as their test tasks.

To alleviate this, we propose a meta learning with domain adaptation paradigm, which performs meta-learning by incorporating few-shot learning and task-level domain adaptation unified into a single meta-learner.

We instantiate our few-shot model with Prototypical Networks and adopt an adversarial approach for task level domain adaptation.

We conduct several experiments to validate the proposed ideas.6 APPENDIX: DATASET CONSTRUCTION DISPLAYFORM0 Here we show the details of the original Omniglot dataset and the statistical details, and some examples of how the characters look in a different domain.

The meta-train, meta-test, and domain adaptation split of classes we used are based on the same split used in prior work.

There is no overlap of classes or instances among the three sets, i.e., they are all mutually exclusive both at instance and class-level.

7 APPENDIX: MODEL CONFIGURATION AND TRAINING For MLDA, we followed training procedures adopted similar to BID38 and BID28 .

Specifically, for CycleGAN, we changed the parameters related to image dimensions (scaling and cropping pre-processing) to keep the generated image size fixed to the original size i.e. 28 × 28 for Omniglot/Omniglot-M and 84 × 84 for OfficeHome Clipart/Product.

The generative networks are the same as the original work BID38 , each including two stride-2 convolutions with residual blocks, and two fractionally-strided convolutions with stride 1 2 .

For all experiments, we used 6 blocks to generate images.

We initialized the learning rate to 0.0002 and kept this learning rate for training till 100 epochs.

The model after each epoch was used to translate source task to target task.

Weights were initialized with Gaussian distribution with mean 0 and standard deviation 0.02.

We use the Adam solver with a batch size of 1.

We also modified the loss function for diffent settings of MLDA.

Specifically, for MLDA, we removed the losses related to target → source (B → A) mapping and set λ idt = 0.

For MLDA+idt, we set λ idt = 0.1.

For MLDA+idt+revMap, we kept the loss function the same as the original CycleGAN.For Prototypical Networks, we followed the best hyperparameter settings in BID28 .

We used the same embedding architecture in the original work, including four convoluational blocks, each of which comprises a 64-filter 3 × 3 convolution, batch normalization layer, ReLU activation, and 2 × 2 max-pooling layer.

This results in 64-dimensional output space for Omniglot/Omniglotm and 1600-dimensional output space for HomeOffice Clipart/Product.

For Omniglot/Omniglotm experiments, the learning rate was set to 0.001 and reduced by half every 2K iterations, starting from iteration 2K.

The network is trained for a total of 20K iterations.

For OfficeHome Product/Clipart experiments, we initialized the learning rate to 0.001 and decayed the learning rate by half every 25K iterations, starting from iteration 25K.

The model is trained up to 100K iterations.

We also use Adam solver to optimize the networks.

Following BID28 , we chose squared Euclidean distance to perform kNN classification as this metric showed superior performance in prior work.

For all the baselines, we reused the official code and ran them with default hyperparameters.

We only modified parameters to make the models compatible with the image resolution and number of classes in Omniglot/Omniglot-M and Product/Clipart datasets.

In all experiments, we set N c classes and N S support points per class identical at training and test-time.

We also fixed 15 query points per class per episode in all experiments.

We computed the classification accuracy by averaging over 600 randomly generated episodes from the Meta-test set.

@highlight

Meta Learning for Few Shot learning assumes that training tasks and test tasks are drawn from the same distribution. What do you do if they are not? Meta Learning with task-level Domain Adaptation!

@highlight

This paper proposes a model combining unsupervised adversarial domain adaptation with prototypical networks that performs better than few-shot learning baselines on few-shot learning tasks with domain shift.

@highlight

The authors proposed meta domain adaptation to address domain shift scenario in meta learning setup, demonstrating performance improvements in several experiments.