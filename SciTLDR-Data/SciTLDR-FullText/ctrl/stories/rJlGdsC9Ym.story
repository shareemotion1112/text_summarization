Curriculum learning consists in learning a difficult task by first training on an easy version of it, then on more and more difficult versions and finally on the difficult task.

To make this learning efficient, given a curriculum and the current learning state of an agent, we need to find what are the good next tasks to train the agent on.

Teacher-Student algorithms assume that the good next tasks are the ones on which the agent is making the fastest progress or digress.

We first simplify and improve them.

However, two problematic situations where the agent is mainly trained on tasks it can't learn yet or it already learnt may occur.

Therefore, we introduce a new algorithm using min max ordered curriculums that assumes that the good next tasks are the ones that are learnable but not learnt yet.

It outperforms Teacher-Student algorithms on small curriculums and significantly outperforms them on sophisticated ones with numerous tasks.

Curriculum learning.

An agent with no prior knowledge can learn a lot of tasks by reinforcement, i.e. by reinforcing (taking more often) actions that lead to higher reward.

But, for some very hard tasks, it is impossible.

Let's consider the following task:Figure 1: The agent (in red) receives a reward of 1 when it picks up the blue ball in the adjacent room.

To do so, it has to first open the gray box, take the key inside and then open the locked door.

This is an easy task for humans because we have prior knowledge: we know that a key can be picked up, that we can open a locked door with a key, etc...

However, most of the time, the agent starts with no prior knowledge, i.e. it starts by acting randomly.

Therefore, it has a probability near 0 of achieving the task in a decent number of time-steps, so it has a probability near 0 of getting reward, so it can't learn the task by reinforcement.

One solution to still learn this task is to do curriculum learning BID0 ), i.e. to first train the agent on an easy version of the task, where it can get reward and learn, then train on more and more difficult versions using the previously learnt policy and finally, train on the difficult task.

Learning by curriculum may be decomposed into two parts:1.

Defining the curriculum, i.e. the set of tasks the agent may be trained on.

2.

Defining the program, i.e. the sequence of curriculum's tasks it will be trained on.

These two parts can be done online, during training.

Curriculum learning algorithms.

Defining a curriculum and a program can be done manually, e.g. by defining a hand-chosen performance threshold for advancement to the next task BID6 ; BID5 ).However, if an efficient algorithm is found, it may save us a huge amount of time in the future.

Besides, efficient (and more efficient than humans) algorithms are likely to exist because they can easily mix in different tasks (what is hard for humans) and then:• avoid catastrophic forgetting by continuously retraining on easier tasks;• quickly detect learnable but not learnt yet tasks.

Hence, it motivates the research of curriculum learning algorithms.

Curriculum learning algorithms can be grouped into two categories:1.

curriculum algorithms: algorithms that define the curriculum; 2. program algorithms: algorithms that define the program, i.e. that decide, given a curriculum and the learning state of the agent, what are the good next tasks to train the agent on.

In this paper, we will focus on program algorithms, in the reinforcement learning context.

Recently, several such algorithms emerged, focused on the notion of learning progress BID4 ; BID3 BID2 ).

BID4 proposed four algorithms (called Teacher-Student) based on the assumption that the good next tasks are the ones on which the agent is making the fastest progress or digress.

We first simplify and improve Teacher-Student algorithms (section 4).

However, even improved, two problematic situations where the agent is mainly trained on tasks it can't learn or it already learnt may occur.

Therefore, we introduce a new algorithm (section 5), focused on the notion of mastering rate, based on the assumption that the good next tasks are the ones that are learnable but not learnt yet.

We show that this algorithm outperforms Teacher-Student algorithms on small curriculums and significantly outperforms them on sophisticated ones with numerous tasks.

2.1 CURRICULUM LEARNING First, let's recall some general curriculum learning notions defined in BID3 .

A curriculum C is a set of tasks {c 1 , ..., c n }.

A sample x is a drawn from one of the tasks of C. A distribution d over C is a family of non-negative summing to one numbers indexed by DISPLAYFORM0 Without loss of generality, we propose to perceive a distribution d over tasks C as coming from an attention a over tasks, i.e. d := ∆(a).

An attention a over C is a family of non-negative numbers indexed by C, i.e. a = (a c ) c∈C with a c ≥ 0.

Intuitively, a c represents the interest given to task c. Let A C be the set of attentions over C. BID4 ; BID3 BID2 , several distribution converters are used (without using this terminology): DISPLAYFORM1 • the argmax distribution converter: ∆ Amax (a) c := 1 if c = argmax c a c 0 otherwise .A greedy version of it is used in BID4 DISPLAYFORM2 and u the uniform distribution over C.• the exponential distribution converter: BID3 ).

DISPLAYFORM3 • the Boltzmann distribution converter: BID4 ).

DISPLAYFORM4 • The powered distribution converter: BID2 ).

DISPLAYFORM5 An attention function a : N → A C is a time-varying sequence of attentions over C. A program d can be rewritten using this notion: d(t) := ∆(a(t)) for a given attention converter ∆.Finally, a program algorithm can be defined as follows: DISPLAYFORM6 An agent A; DISPLAYFORM7

The Teacher-Student paper BID4 ) presents four attention functions called Online, Naive, Window and Sampling.

They are all based on the idea that the attention must be given by the absolute value of an estimate of the learning progress over tasks, i.e. A(t) := |β(t)| where β c (t) is an estimate of the learning progress of the agent A on task c.

For the Window attention function, they first estimate the "instantaneous" learning progress of the agent A on task c by the slope β Linreg c (t) of the linear regression of the points (t 1 , r t1 ), ..., (t K , r t K ) where t 1 , ..., t K are the K last time-steps when the agent was trained on a sample of c and where r ti is the return got by the agent at these time-steps.

From this instantaneous learning progress, they define β c as the weighted moving average of β DISPLAYFORM0 For all the algorithms, a Boltzmann or greedy argmax distribution converter is used.

For example, here is the GAmax Window program algorithm proposed in the paper: DISPLAYFORM1 An agent A; β := 0 ; for t ← 1 to T do a := |β| ; d := ∆ GAmax (a) ; Draw a task c from d and then a sample x from c ; Train A on x and observe return r t ; β Linreg c := slope of lin.

reg.

of (t 1 , r t1 ), ..., (t K , r t K ) ; β c := αβ Three curriculums were used to evaluate the algorithms, called BlockedUnlockPickup, KeyCorridor and ObstructedMaze (see appendix A for screenshots of all tasks).

They are all composed of Gym MiniGrid environments BID1 ).These environments are partially observable and at each time-step, the agent receives a 7 × 7 × 3 image (figure 1) along with a textual instruction.

Some environments require language understanding and memory to be efficiently solved, but the ones chosen in the three curriculums don't.

The agent gets a reward of 1 − n nmax when the instruction is executed in n steps with n ≤ n max .

Otherwise, it gets a reward of 0.

Before simplifying and improving Teacher-Student algorithms, here are some suggestions about which distribution converters and attention functions of the Teacher-Student paper to use and not to use.

First, in this paper, two distribution converters are proposed: the greedy argmax and the Boltzmann ones.

We don't recommend to use the Boltzmann distribution converter because τ is very hard to tune in order to get a distribution that is neither deterministic nor uniform.

Second, four attention functions are proposed: the Online, Naive, Window and Sampling ones.

We don't recommend to use:• the Naive attention function because it is a naive version of the Window one and performs worst (see figure 5 in BID4 );• the Sampling attention function because it performs worst than the Window one (see figure 5 in BID4 ).

Moreover, the reason it was introduced was to avoid hyperparameters but it still require to tune a ε to avoid deterministic distributions (see algorithm 8 in BID4 )...

It remains the Online and Window attention functions.

But, the Online one is similar to the Window one when K = 1.Finally, among all what is proposed in this paper, we only recommend to use the Window attention function with the greedy argmax distribution converter, i.e. to use the GAmax Window algorithm (algorithm 2).

This is the only one we will consider in the rest of this section.

Now, let's see how we can simplify and improve the GAmax Window algorithm. .

This corresponds to the powered distribution converter when p = 1.

We then replace the greedy argmax distribution converter by a greedy proportional one, and improve performances (figures 2 and 3).

Algorithm 3: GProp Linreg algorithm input: A curriculum C ; An agent A; β Linreg := 0 ; for t ← 1 to T do a := |β Linreg | ; d := ∆ GP rop (a) ; Draw a task c from d and then a sample x from c ; Train A on x and observe return r t ; β Linreg c := slope of lin.

reg.

FIG6 , ..., (t K , r t K ) ;In the rest of this article, this algorithm will be referred as our "baseline".

The two following figures show that:• the GAmax Linreg algorithm performs similarly to the GAmax Window algorithm, as asserted before.

It even seems a bit more stable because the gap between the first and last quartile is smaller.• the GProp Linreg algorithm performs better than the GAmax Linreg and GAmax Window algorithm, as asserted before.

Algorithms introduced in BID4 ; BID3 BID2 , and in particular Teacher-Student algorithms and the baseline algorithm, are focused on the notion of learning progress, based on the assumption that the good next tasks are the ones on which the agent is making the fastest progress or digress.

However, two problematic situations may occur:1.

The agent may be mainly trained on tasks it already learnt.

The frame B of the figure 4 shows that, around time-step 500k, the agent already learnt Unlock and UnlockPickup but is still trained 90% of the time on them, i.e. on tasks it already learnt.

2.

It may be mainly trained on tasks it can't learn yet.

The more the curriculum has tasks, the more it occurs:• The frame A of the figure 4 shows that, initially, the agent doesn't learn Unlock but is trained 66% of the time on UnlockPickup and BlockedUnlockPickup, i.e. on tasks it can't learn yet.• The figure 5 shows that agents spend most of the time training on the hardest task of the ObstructedMaze curriculum whereas they have not learnt yet the easy tasks..

The agent with seed 6 was trained on the BlockedUnlockPickup curriculum using the baseline algorithm.

The return and probability during training are plotted.

Two particular moments of the training are framed.

Figure 5: 10 agents (seeds) were trained on the ObstructedMaze curriculum using the baseline algorithm.

The mean return and mean probability during training are plotted.

To overcome these issues, we introduce a new algorithm, focused on the notion of mastering rate, based on the assumption that the good next tasks to train on are the ones that are learnable but not learnt yet.

Why this assumption?

Because it can't be otherwise.

Mainly training on learnt tasks or not learnable ones is a lost of time.

This time must be spent training on respectively harder or easier tasks.

In subsection 5.1, we first define what are learnt and learnable tasks and then, in subsection 5.2, we present this new algorithm.

Learnt tasks.

A min-max curriculum is a curriculum C = {c 1 , ..., c n } along with:• a family (m 1 c ) c∈C where m 1 c is an estimation of the minimum mean return the agent would get on task c. It should be higher than the true minimum mean return.• and a family (M 1 c ) c∈C where M 1 c is an estimation of the maximum mean return the agent would get on task c. It should be lower than the true maximum mean return.

On such a curriculum, we can define, for a task c:• the live mean returnr c (t) byr c (1) := m 1 c andr c (t) = (r t1 + ... + r t K )/K where t 1 , ..., t K are the K last time-steps when the agent was trained on a sample of c and where r ti is the return got by the agent at these time-steps;• the live minimum mean return by m c (1) := m From this, we can define, for a task c, the mastering rate M c (t) :=r DISPLAYFORM0 Mc(t)−mc(t) .

Intuitively, a task c would be said "learnt" if M c (t) is near 1 and "not learnt" if M c (t) is near 0.

FIG6 , ..., (t K , r t K ) ; Finally, we can remark that the MR algorithm is just a more general version of Teacher-Student algorithms and the baseline algorithm.

If we consider min-max ordered curriculums without edges (i.e. just curriculums), if δ = 0, and if we use the GProp dist converter instead of the Prop one, then the MR algorithm is exactly the GProp Linreg algorithm.

The MR algorithm with δ = 0.6 (see appendix B for the min-max ordered curriculums given to this algorithm) outperforms the baseline algorithm on all the curriculums, especially on:• KeyCorridor where the median return of the baseline is near 0 after 10M time-steps on S4R3, S5R3 and S6R3 while the first quartile of the MR algorithm is higher than 0.8 after 6M time-steps (see FIG8 ).•

ObstructedMaze where the last quartile of the baseline is near 0 after 10M time-steps on all the tasks while the last quartile of the MR algorithm is higher than 0.7 after 5M time-steps on 1Dl, 1Dlh, 1Dlhb, 2Dl, 2Dlhb (see FIG9 )..

A CURRICULUMS Three curriculums were used to evaluate the algorithms: BlockedUnlockPickup (3 tasks), KeyCorridor (6 tasks) and ObstructedMaze (9 tasks).(a) Unlock.

nmax = 288 DISPLAYFORM0 Figure 8: BlockedUnlockPickup curriculum.

In Unlock, the agent has to open the locked door.

In the others, it has to pick up the box.

In UnlockPickup, the door is locked and, in BlockedUnlockPickup, it is locked and blocked by a ball.

The position and color of the door and the box are random.

Here are the min-max ordered curriculums given to the MR algorithm in subsection 5.3.For every task c, we set m 1 c to 0 and M 1 c to 0.5.

The real maximum mean return is around 0.9 but we preferred to take a much lower maximum estimation to show we don't need an accurate one to get the algorithm working.

FIG6 : GProp Linreg and MR with δ = 0.6 were each tested with 10 different agents (seeds) on the BlockedUnlockPickup curriculum.

The median return during training, between the first and last quartile, are plotted.

<|TLDR|>

@highlight

We present a new algorithm for learning by curriculum based on the notion of mastering rate that outperforms previous algorithms.