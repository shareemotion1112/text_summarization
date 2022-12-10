This paper addresses unsupervised domain adaptation, the setting where labeled training data is available on a source domain, but the goal is to have good performance on a target domain with only unlabeled data.

Like much of previous work, we seek to align the learned representations of the source and target domains while preserving discriminability.

The way we accomplish alignment is by learning to perform auxiliary self-supervised task(s) on both domains simultaneously.

Each self-supervised task brings the two domains closer together along the direction relevant to that task.

Training this jointly with the main task classifier on the source domain is shown to successfully generalize to the unlabeled target domain.

The presented objective is straightforward to implement and easy to optimize.

We achieve state-of-the-art results on four out of seven standard benchmarks, and competitive results on segmentation adaptation.

We also demonstrate that our method composes well with another popular pixel-level adaptation method.

@highlight

We use self-supervision on both domain to align them for unsupervised domain adaptation.