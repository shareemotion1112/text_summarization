Place and grid-cells are known to aid navigation in animals and humans.

Together with concept cells, they allow humans to form an internal representation of the external world, namely the concept space.

We investigate the presence of such a space in deep neural networks by plotting the activation profile of its hidden layer neurons.

Although place cell and concept-cell like properties are found, grid-cell like firing patterns are absent thereby indicating a lack of path integration or feature transformation functionality in trained networks.

Overall, we present a plausible inadequacy in current deep learning practices that restrict deep networks from performing analogical reasoning and memory retrieval tasks.

Cells in the hippocampal region of the rat brain signal the animal's location in space.

This phenomenon 11 has provided support to the idea that the rat hippocampus operates like a cognitive map [1] .

Specific 12 cells, termed as the "place cells" are known to selectively fire when the animal is in certain locations 13 in the environment [1] .

The discovery of place cells was followed by the discovery of "grid" cells, 14 that selectively fire at multiple discrete and regularly spaced locations [2] .

Extensive animal studies, is generally attributed to their ability to identify optimal discriminative features in a given dataset.

This learnt feature space can be thought of as a "concept" space for the DNN.

The ability to perform 29 relational reasoning depends on the network's ability to navigate over this concept space.

To solve 30 analogical problems, the network needs to apply the desired feature transformation to arrive at the 31 correct solution.

For instance in the image space, the network needs to navigate the visual concept 32 space to understand that an OR operation between a "dog image" and a "ball image" could lead to an 33 image of "a dog playing with a ball".

The ability to navigate the concept space depends on the properties of the constituent neurons.

Akin activity.

However, the ability to navigate the concept space relies on the DNN's ability to perform 40 "path integration", or the ability to localize its position in the concept space given specific feature 41 transformations.

Grid cells are known to be responsible for path integration in rodents and humans.

Therefore, the ability of the DNN to solve analogical problems relies on the presence of grid cell-like 43 properties among DNN neurons.

In this work 1 , we aim to investigate a particular class of DNN, namely convolutional neural network 45 (CNN).

We investigate the firing properties of neurons, specifically in the final and pre-final layers, to 46 understand their activation patterns while the network performs classification.

We trained a CNN to identify hand-written digits 0 − 9 in the MNIST dataset.

The dataset has 60,000

Once the network was trained to yield a satisfactory classification accuracy on the testset, we 57 investigated the activation properties of constituent neurons.

We limited our analysis to the pre- corresponds to the concept space learned by the network, also referred hereafter as the network 64 concept space.

We generated the aforementioned spaces for the testset images (10,000 points).

Figure   65 1 shows the two concept spaces.

To observe the firing property of the neuron across the concept space, the tSNE plots were color-coded 67 using the respective neuron's activation values.

Figure 2 shows the firing pattern of Layer 7 neurons.

All the final layer neurons showed place-cell like activity, with preference to one cluster.

Given the 69 training paradigm, this was expected.

Further, we extended our analysis to observe the firing pattern 70 of the previous layer.

The pre-final layer had 50 neurons.

However, none of the neurons had grid-cell like firing property.

We used Principal Component Analysis (PCA) and obtained 11 PCs that significantly explained the Thus, the trained CNN would fail to perform analogical problems, similar to other deep models 88 trained on more complicated datasets.

The network analyzed here was a fairly simple and shallow network as compared to practical deep

<|TLDR|>

@highlight

We investigated if simple deep networks possess grid cell-like artificial neurons while memory retrieval in the learned concept space.