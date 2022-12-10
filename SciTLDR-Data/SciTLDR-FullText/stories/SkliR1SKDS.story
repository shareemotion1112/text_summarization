Data augmentation techniques, e.g., flipping or cropping, which systematically enlarge the training dataset by explicitly generating more training samples, are effective in improving the generalization performance of deep neural networks.

In the supervised setting, a common practice for data augmentation is to assign the same label to all augmented samples of the same source.

However, if the augmentation results in large distributional discrepancy among them (e.g., rotations), forcing their label invariance may be too difficult to solve and often hurts the performance.

To tackle this challenge, we suggest a simple yet effective idea of learning the joint distribution of the original and self-supervised labels of augmented samples.

The joint learning framework is easier to train, and enables an aggregated inference combining the predictions from different augmented samples for improving the performance.

Further, to speed up the aggregation process, we also propose a knowledge transfer technique, self-distillation, which transfers the knowledge of augmentation into the model itself.

We demonstrate the effectiveness of our data augmentation framework on various fully-supervised settings including the few-shot and imbalanced classification scenarios.

@highlight

We propose a simple self-supervised data augmentation technique which improves performance of fully-supervised scenarios including few-shot learning and imbalanced classification.