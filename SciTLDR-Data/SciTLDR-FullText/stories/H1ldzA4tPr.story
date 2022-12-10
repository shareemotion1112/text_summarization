Finding an embedding space for a linear approximation of a nonlinear dynamical system enables efficient system identification and control synthesis.

The Koopman operator theory lays the foundation for identifying the nonlinear-to-linear coordinate transformations with data-driven methods.

Recently, researchers have proposed to use deep neural networks as a more expressive class of basis functions for calculating the Koopman operators.

These approaches, however, assume a fixed dimensional state space; they are therefore not applicable to scenarios with a variable number of objects.

In this paper, we propose to learn compositional Koopman operators, using graph neural networks to encode the state into object-centric embeddings and using a block-wise linear transition matrix to regularize the shared structure across objects.

The learned dynamics can quickly adapt to new environments of unknown physical parameters and produce control signals to achieve a specified goal.

Our experiments on manipulating ropes and controlling soft robots show that the proposed method has better efficiency and generalization ability than existing baselines.

@highlight

Learning compositional Koopman operators for efficient system identification and model-based control.