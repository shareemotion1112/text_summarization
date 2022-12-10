As neural networks grow deeper and wider, learning networks with hard-threshold activations is becoming increasingly important, both for network quantization, which can drastically reduce time and energy requirements, and for creating large integrated systems of deep networks, which may have non-differentiable components and must avoid vanishing and exploding gradients for effective learning.

However, since gradient descent is not applicable to hard-threshold functions, it is not clear how to learn them in a principled way.

We address this problem by observing that setting targets for hard-threshold hidden units in order to minimize loss is a discrete optimization problem, and can be solved as such.

The discrete optimization goal is to find a set of targets such that each unit, including the output, has a linearly separable problem to solve.

Given these targets, the network decomposes into individual perceptrons, which can then be learned with standard convex approaches.

Based on this, we develop a recursive mini-batch algorithm for learning deep hard-threshold networks that includes the popular but poorly justified straight-through estimator as a special case.

Empirically, we show that our algorithm improves classification accuracy in a number of settings, including for AlexNet and ResNet-18 on ImageNet, when compared to the straight-through estimator.

<|TLDR|>

@highlight

We learn deep networks of hard-threshold units by setting hidden-unit targets using combinatorial optimization and weights by convex optimization, resulting in improved performance on ImageNet.

@highlight

The paper explains and generalizes approaches for learning neural nets with hard activation.

@highlight

This paper examines the problem of optimizing deep networks of hard-threshold units.

@highlight

The paper discusses the problem of optimizing neural networks with hard threshold and proposes a novel solution to it with a collection of heuristics/approximations.