Spectral Graph Convolutional Networks (GCNs) are a generalization of convolutional networks to learning on graph-structured data.

Applications of spectral GCNs have been successful, but limited to a few problems where the graph is fixed, such as shape correspondence and node classification.

In this work, we address this limitation by revisiting a particular family of spectral graph networks, Chebyshev GCNs, showing its efficacy in solving graph classification tasks with a variable graph structure and size.

Current GCNs also restrict graphs to have at most one edge between any pair of nodes.

To this end, we propose a novel multigraph network that learns from multi-relational graphs.

We explicitly model different types of edges: annotated edges, learned edges with abstract meaning, and hierarchical edges.

We also experiment with different ways to fuse the representations extracted from different edge types.

This restriction is sometimes implied from a dataset, however, we relax this restriction for all kinds of datasets.

We achieve state-of-the-art results on a variety of chemical, social, and vision graph classification benchmarks.

<|TLDR|>

@highlight

A novel approach to graph classification based on spectral graph convolutional networks and its extension to multigraphs with learnable relations and hierarchical structure. We show state-of-the art results on chemical, social and image datasets.