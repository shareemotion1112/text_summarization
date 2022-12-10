We study how the topology of a data set comprising two components representing two classes of objects in a binary classification problem changes as it passes through the layers of a well-trained neural network, i.e., one with perfect accuracy on training set and a generalization error of less than 1%.

The goal is to shed light on two well-known mysteries in deep neural networks: (i) a nonsmooth activation function like ReLU outperforms a smooth one like hyperbolic tangent; (ii) successful neural network architectures rely on having many layers, despite the fact that a shallow network is able to approximate any function arbitrary well.

We performed extensive experiments on persistent homology of a range of point cloud data sets.

The results consistently demonstrate the following: (1) Neural networks operate by changing topology, transforming a topologically complicated data set into a topologically simple one as it passes through the layers.

No matter how complicated the topology of the data set we begin with, when passed through a well-trained neural network, the Betti numbers of both components invariably reduce to their lowest possible values: zeroth Betti number is one and all higher Betti numbers are zero.

Furthermore, (2) the reduction in Betti numbers is significantly faster for ReLU activation compared to hyperbolic tangent activation --- consistent with the fact that the former define nonhomeomorphic maps (that change topology) whereas the latter define homeomorphic maps (that preserve topology).

Lastly, (3) shallow and deep networks process the same data set differently --- a shallow network operates mainly through changing geometry and changes topology only in its final layers, a deep network spreads topological changes more evenly across all its layers.

<|TLDR|>

@highlight

We show that neural networks operate by changing topologly of a data set and explore how architectural choices effect this change.