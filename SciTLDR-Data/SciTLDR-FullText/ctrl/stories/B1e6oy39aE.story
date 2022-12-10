This work provides theoretical and empirical evidence that invariance-inducing regularizers can increase predictive accuracy for worst-case spatial transformations (spatial robustness).

Evaluated on these adversarially transformed examples, we demonstrate that adding regularization on top of standard or adversarial training reduces the relative error by 20% for CIFAR10 without increasing the computational cost.

This outperforms handcrafted networks that were explicitly designed to be spatial-equivariant.

Furthermore, we observe for SVHN, known to have inherent variance in orientation, that robust training also improves standard accuracy on the test set.

<|TLDR|>

@highlight

for spatial transformations robust minimizer also minimizes standard accuracy; invariance-inducing regularization leads to better robustness than specialized architectures