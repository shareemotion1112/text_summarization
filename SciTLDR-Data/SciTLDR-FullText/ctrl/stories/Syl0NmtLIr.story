In cognitive systems, the role of a working memory is crucial for visual reasoning and decision making.

Tremendous progress has been made in understanding the mechanisms of the human/animal working memory, as well as in formulating different frameworks of artificial neural networks.

In the case of humans, the visual working memory (VWM) task is a standard one in which the subjects are presented with a sequence of images, each of which needs to be identified as to whether it was already seen or not.



Our work is a study of multiple ways to learn a working memory model using recurrent neural networks that learn to remember input images across timesteps.

We train these neural networks to solve the working memory task by training them with a sequence of images in supervised and reinforcement learning settings.

The supervised setting uses image sequences with their corresponding labels.

The reinforcement learning setting is inspired by the popular view in neuroscience that the working memory in the prefrontal cortex is modulated by a dopaminergic mechanism.

We consider the VWM task as an environment that rewards the agent when it remembers past information and penalizes it for forgetting.

We quantitatively estimate the performance of these models on sequences of images from a standard image dataset (CIFAR-100).

Further, we evaluate their ability to remember and recall as they are increasingly trained over episodes.

Based on our analysis, we establish that a gated recurrent neural network model with long short-term memory units trained using reinforcement learning is powerful and more efficient in temporally consolidating the input spatial information.



This work is an initial analysis as a part of our ultimate goal to use artificial neural networks to model the behavior and information processing of the working memory of the brain and to use brain imaging data captured from human subjects during the VWM cognitive task to understand various memory mechanisms of the brain.

Memory is an essential component for solving many tasks intelligently.

Most sequential tasks 24 involve the need for a mechanism to maintain a context.

In the brain, working memory serves as 25 a work space to encode and maintain the most relevant information over a short period of time, in 26 order to use it to guide behavior for cognitive tasks.

Several cognitive tasks have been proposed 27 in the Neuropsychology literature to study and understand the working memory in animals.

The

Visual Working Memory Task (VWM task) [1] or the classic N-back task is one of the most simple On the other hand, with artificial intelligence systems, there has been very good progress in models 32 that learn from sequences of inputs using artificial neural networks as memory for all types of learning (supervised, unsupervised and reinforcement).

We intend to use these developments as an ideal 34 opportunity for synergy to computationally model the working memory system of the brain.

As memory is an important aspect of both artificial intelligence and neuroscience, there are some 36 good studies that helped choose our models as discussed in Section 2.

For all experiments in both supervised and reinforcement learning settings, 100 images were drawn The problem solved by all the models is a binary classification problem, predicting unseen(0)/seen(1).

The performance of all the models in the experiments were measured using the accuracy metric 67 calculated based on the number of correct predictions for the 100 images in a sequence (as a %).

This 68 evaluation was repeated for 10 independent trials as a part of ablation studies.

<|TLDR|>

@highlight

LSTMs can more effectively model the working memory if they are learned using reinforcement learning, much like the dopamine system that modulates the memory in the prefrontal cortex    