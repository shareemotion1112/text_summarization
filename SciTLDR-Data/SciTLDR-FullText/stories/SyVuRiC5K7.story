The goal of few-shot learning is to learn a classifier that generalizes well even when trained with a limited number of training instances per class.

The recently introduced meta-learning approaches tackle this problem by learning a generic classifier across a large number of multiclass classification tasks and generalizing the model to a new task.

Yet, even with such meta-learning, the low-data problem in the novel classification task still remains.

In this paper, we propose Transductive Propagation Network (TPN), a novel meta-learning framework for transductive inference that classifies the entire test set at once to alleviate the low-data problem.

Specifically, we propose to learn to propagate labels from labeled instances to unlabeled test instances, by learning a graph construction module that exploits the manifold structure in the data.

TPN jointly learns both the parameters of feature embedding and the graph construction in an end-to-end manner.

We validate TPN on multiple benchmark datasets, on which it largely outperforms existing few-shot learning approaches and achieves the state-of-the-art results.

@highlight

We propose a novel meta-learning framework for transductive inference that classifies the entire test set at once to alleviate the low-data problem.

@highlight


This paper proposes to address few-shot learning in a transductive way by learning a label propagation model in an end-to-end manner, the first to learn label propagation for transductive few-shot learning and produced effective empirical results. 

@highlight

This paper proposes a meta-learning framework that leverages unlabeled data by learning the graph-based label propogation in an end-to-end manner.

@highlight

Studies few-host learning in a transductive setting: using meta learning to learn to propagate labels from training samples to test samples. 