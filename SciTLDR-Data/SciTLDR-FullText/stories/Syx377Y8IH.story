One of the fundamental problems in supervised classification and in machine learning in general, is the modelling of non-parametric invariances that exist in data.

Most prior art has focused on enforcing priors in the form of invariances to parametric nuisance transformations that are expected to be present in data.

However, learning non-parametric invariances directly from data remains an important open problem.

In this paper, we introduce a new architectural layer for convolutional networks which is capable of learning general invariances from data itself.

This layer can learn invariance to non-parametric transformations and interestingly, motivates and incorporates permanent random connectomes there by being called Permanent Random Connectome Non-Parametric Transformation Networks (PRC-NPTN).

PRC-NPTN networks are initialized with random connections (not just weights) which are a small subset of the connections in a fully connected convolution layer.

Importantly, these connections in PRC-NPTNs once initialized remain permanent throughout training and testing.

Random connectomes makes these architectures loosely more biologically plausible than many other mainstream network architectures which require highly ordered structures.

We motivate randomly initialized connections as a simple method to learn invariance from data itself while invoking invariance towards multiple nuisance transformations simultaneously.

We find that these randomly initialized permanent connections have positive effects on generalization, outperform much larger ConvNet baselines and the recently proposed Non-Parametric Transformation Network (NPTN) on benchmarks that enforce learning invariances from the data itself.

invariances directly from the data, with the only prior being the structure that allows them to do so.

with an enhanced ability to learn non-parametric invariances through permanent random connectivity.

we do not explore these biological connections in more detail, it is still an interesting observation.

The common presence of random connections in the cortex at a local level leads us to ask: Is it

Both train and test data were augmented leading to an increase in overall complexity of the problem.

No architecture was altered in anyway between the two transformations i.e. they were not designed Discussion.

We present all test errors for this experiment in Table.

@highlight

A layer modelling local random connectomes in the cortex within deep networks capable of learning general non-parametric invariances from the data itself.