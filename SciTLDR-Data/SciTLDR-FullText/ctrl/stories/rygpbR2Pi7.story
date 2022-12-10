Pruning neural networks for wiring length efficiency is considered.

Three techniques are proposed and experimentally tested: distance-based regularization, nested-rank pruning, and layer-by-layer bipartite matching.

The first two algorithms are used in the training and pruning phases, respectively, and the third is used in the arranging neurons phase.

Experiments show that distance-based regularization with weight based pruning tends to perform the best, with or without layer-by-layer bipartite matching.

These results suggest that these techniques may be useful in creating neural networks for implementation in widely deployed specialized circuits.

3.

After iterating between the previous two steps a sufficient number of times, apply layer-by-layer bipartite matching to further optimize the energy of the layouts.

The algorithm uses the realization that finding the optimal permutation of nodes in one layer that minimizes the wiring length to the nodes of other layers assuming their positions are fixed is equivalent to the weighted bipartite matching problem, for which the Hungarian algorithm is polynomial-time and exact BID19 .

Apply this optimization algorithm layer by layer to the nodes of the pruned network.

We run pruning experiments on a fully-connected neural network for MNIST, which contains two hidden layers of 300 and 100 units, respectively (this is the standard LeNet-300-100 architecture that has been widely studied in the pruning literature).

We also try pruning the fully connected layers of a 10-layer convolutional network trained on the street-view house numbers dataset BID20 .

We show energy-accuracy curves for one setting of hyperparameters for each of these datasets in FIG0 .In TAB1 we show a subset of the results of a hyperparameter grid search for these two datasets.

We record the accuracy and energy after each pruning iteration, and then for each set of hyperparameters choose the model with the lowest energy greater than some threshold accuracy.

For each target accuracy we show the weight-based result (which is comparable to the technique of BID2 and forms a baseline) and the results on the distance-based regularization technique.

We found that nested rank pruning can perform better than pure weight based pruning, however distance-based regularization tends to outperform techniques that use nested-rank pruning, although sometimes distance-based regularization with nested-rank pruning performs best in the lower accuracy, low energy regime as can be seen in the right graph of FIG0 .

In these tables we obtain a wide range of values at the highest accuracy (which we suspect is due to randomness in initial accuracy) but more consistency at the lower accuracies.

For MNIST, our best performing set of hyperparameters results in a compression ratio of 1.64 percent at 98%, comparable to state-of-the art results for this initial architecture and dataset BID21 .

In Table 3 we apply the bipartite matching heuristic to the best performing network obtained using weight-based regularization and the best performing network using weight-distance based regularization for each target accuracy.

Across both datasets the distance-based regularization outperforms weight-based regularization on average across four trials, in some cases by close to 70%.

In this paper we consider the novel problem of learning accurate neural networks that have low total wiring length because this corresponds to energy consumption in the fundamental limit.

We introduce weight-distance regularization, nested rank pruning, and layer-by-layer bipartite matching and show through ablation studies that all of these algorithms are effective, and can even reach state-of-the-art compression ratios.

The results suggests that these techniques may be worth the computational effort if the neural network is to be widely deployed, if significantly lower energy is worth the slight decrease in accuracy, or if the application is to be deployed on either a specialized circuit or general purpose processor.

Table 2 : Average and standard deviation over four trials for Street View House Numbers task on both the wiring length metric (energy) and remaining edges metric (edges).

We note that with the appropriate hyperparameter setting our algorithm outperforms the baseline weight based techniques (p=0) often on both the energy and number of remaining edges metric.

Table 3 : Results of applying the bipartite matching algorithm on the best performing weight-based pruning network and best performing distance-based regularization method before and after applying layer-by-layer bipartite matching.

Average and standard deviation over 4 trials presented.

<|TLDR|>

@highlight

Three new algorithms with ablation studies to prune neural network to optimize for wiring length, as opposed to number of remaining weights.