While machine learning models achieve human-comparable performance on sequential data, exploiting structured knowledge is still a challenging problem.

Spatio-temporal graphs have been proved to be a useful tool to abstract interaction graphs and previous works exploits carefully designed feed-forward architecture to preserve such structure.

We argue to scale such network design to real-world problem, a model needs to automatically learn a meaningful representation of the possible relations.

Learning such interaction structure is not trivial: on the one hand, a model has to discover the hidden relations between different problem factors in an unsupervised way; on the other hand, the mined relations have to be interpretable.



In this paper, we propose an attention module able to project a graph sub-structure in a fixed size embedding, preserving the influence that the neighbours exert on a given vertex.

On a comprehensive evaluation done on real-world as well as toy task, we found our model competitive against strong baselines.

<|TLDR|>

@highlight

A graph neural network able to automatically learn and leverage a dynamic interactive graph structure