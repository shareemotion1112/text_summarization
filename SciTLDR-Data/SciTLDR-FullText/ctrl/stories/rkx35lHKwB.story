A fundamental trait of intelligence is the ability to achieve goals in the face of novel circumstances.

In this work, we address one such setting which requires solving a task with a novel set of actions.

Empowering machines with this ability requires generalization in the way an agent perceives its available actions along with the way it uses these actions to solve tasks.

Hence, we propose a framework to enable generalization over both these aspects: understanding an action’s functionality, and using actions to solve tasks through reinforcement learning.

Specifically, an agent interprets an action’s behavior using unsupervised representation learning over a collection of data samples reflecting the diverse properties of that action.

We employ a reinforcement learning architecture which works over these action representations, and propose regularization metrics essential for enabling generalization in a policy.

We illustrate the generalizability of the representation learning method and policy, to enable zero-shot generalization to previously unseen actions on challenging sequential decision-making environments.

Our results and videos can be found at sites.google.com/view/action-generalization/

<|TLDR|>

@highlight

We address the problem of generalization of reinforcement learning to unseen action spaces.