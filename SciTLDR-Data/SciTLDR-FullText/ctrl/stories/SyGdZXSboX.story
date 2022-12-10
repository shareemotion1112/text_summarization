Neural networks offer high-accuracy solutions to a range of problems, but are computationally costly to run in production systems.

We propose a technique called Deep Learning Approximation to take an already-trained neural network model and build a faster (and almost equally accurate) network by manipulating the network structure and coefficients without requiring re-training or access to the training data.

Speedup is achieved by applying a sequential series of independent optimizations that reduce the floating-point operations (FLOPs) required to perform a forward pass.

An optimal lossy approximation is chosen for each layer by weighing the relative accuracy loss and FLOP reduction.

On PASCAL VOC 2007 with the YOLO network, we show an end-to-end 2x speedup in a network forward pass with a $5$\% drop in mAP that can be re-gained by finetuning, enabling this network (and others like it) to be deployed in compute-constrained systems.

FLOPs for DISPLAYFORM0 An optimal approximation is chosen by calculating the runtime and accuracy loss from all possible Table 1 .

When chaining approximations, R for is the ratio 55 of the final output FLOPs to the FLOPs from W .

A is the product of the accuracy scores for each 56 approximation in the chain, since any error introduced by the first will be carried over to the next.

with the absolute speedup, with the exception of the ResNet50 network, as shown in TAB3 .

Table 61 4 shows that the input parameter p can be chosen based on the desired runtime / accuracy tradeoff.

of an improvement in runtime from DLA (for example ResNet50 in TAB3 ).

Additionally, pushing 64 beyond the 2x speedup observed on YOLO without significant accuracy loss is not possible with

<|TLDR|>

@highlight

Decompose weights to use fewer FLOPs with SVD