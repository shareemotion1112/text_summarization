The current dominant paradigm for imitation learning relies on strong supervision of expert actions to learn both 'what' and 'how' to imitate.

We pursue an alternative paradigm wherein an agent first explores the world without any expert supervision and then distills its experience into a goal-conditioned skill policy with a novel forward consistency loss.

In our framework, the role of the expert is only to communicate the goals (i.e., what to imitate) during inference.

The learned policy is then employed to mimic the expert (i.e., how to imitate) after seeing just a sequence of images demonstrating the desired task.

Our method is 'zero-shot' in the sense that the agent never has access to expert actions during training or for the task demonstration at inference.

We evaluate our zero-shot imitator in two real-world settings: complex rope manipulation with a Baxter robot and navigation in previously unseen office environments with a TurtleBot.

Through further experiments in VizDoom simulation, we provide evidence that better mechanisms for exploration lead to learning a more capable policy which in turn improves end task performance.

Videos, models, and more details are available at https://pathak22.github.io/zeroshot-imitation/.

<|TLDR|>

@highlight

Agents can learn to imitate solely visual demonstrations (without actions) at test time after learning from their own experience without any form of supervision at training time.

@highlight

This paper proposes and approach for zero-shot visual learning by learning parametric skill functions.

@highlight

A paper about imitation of a task presented just during inference, where learning is performed in a self-supervised manner and during training the agent explores related but different tasks.

@highlight

Proposes a method for sidestepping the issue of expensive expert demonstration by using the random exploration of an agent to learn generalizable skills which can be applied without specific pretraining